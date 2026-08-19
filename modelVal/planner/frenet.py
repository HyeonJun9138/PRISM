"""Frenet-frame optimal trajectory planner, ported 1:1 from the JS original
(D:\\FrenetFramePath\\src\\planner\\frenetPlanner.js + models/*.js, rev 2026-08).

Semantics are kept identical -- same candidate lattice, same quintic/quartic
boundary solves, same cost terms and per-pool normalization -- so that the
paper can truthfully say "the planner is fixed; only the risk input varies".

One deliberate generalization over the JS demo: obstacles are supplied by a
RiskProvider whose disks may vary with lookahead time t (`disks_batch`).
A constant provider reproduces the original static-obstacle behavior exactly;
the CV / PRISM-mixture providers feed moving, uncertainty-weighted disks into
the very same cost formula.

Disk row layout (M disks x 5): [x, y, hard_radius, sigma, gain]
  sigma == NaN -> JS default  max(10, hard_radius + safety_buffer)
  gain  == NaN -> JS default  (hard_radius + vehicle_radius) / 12
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

EPS = 1e-9


def _js_round(v: float, step: float) -> float:
    """JS Math.round(v/step)*step (half rounds toward +infinity)."""
    return math.floor(v / step + 0.5) * step


def unique_sorted(values: Sequence[float], step: float = 0.01) -> list[float]:
    seen: set[str] = set()
    out: list[float] = []
    for v in values:
        r = _js_round(v, step)
        key = f"{r:.2f}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return sorted(out)


# --------------------------------------------------------------------------
# polynomials (models/polynomials.js)
# --------------------------------------------------------------------------

class QuinticPolynomial:
    """Position-boundary quintic: (p,v,a) at 0 and T."""

    def __init__(self, start: tuple[float, float, float],
                 end: tuple[float, float, float], T: float) -> None:
        p0, v0, a0 = start
        p1, v1, a1 = end
        t2, t3 = T * T, T * T * T
        t4, t5 = t3 * T, t3 * T * T
        self.c0, self.c1, self.c2 = p0, v0, a0 / 2.0
        A = np.array([[t3, t4, t5],
                      [3 * t2, 4 * t3, 5 * t4],
                      [6 * T, 12 * t2, 20 * t3]], dtype=np.float64)
        b = np.array([p1 - (self.c0 + self.c1 * T + self.c2 * t2),
                      v1 - (self.c1 + 2 * self.c2 * T),
                      a1 - 2 * self.c2], dtype=np.float64)
        try:
            c345 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c345 = np.zeros(3)
        self.c3, self.c4, self.c5 = (float(c345[0]), float(c345[1]),
                                     float(c345[2]))

    def eval_batch(self, t: np.ndarray) -> tuple[np.ndarray, ...]:
        t2, t3 = t * t, t * t * t
        t4, t5 = t3 * t, t3 * t * t
        p = self.c0 + self.c1 * t + self.c2 * t2 + self.c3 * t3 \
            + self.c4 * t4 + self.c5 * t5
        v = self.c1 + 2 * self.c2 * t + 3 * self.c3 * t2 + 4 * self.c4 * t3 \
            + 5 * self.c5 * t4
        a = 2 * self.c2 + 6 * self.c3 * t + 12 * self.c4 * t2 \
            + 20 * self.c5 * t3
        j = 6 * self.c3 + 24 * self.c4 * t + 60 * self.c5 * t2
        return p, v, a, j

    def eval(self, t: float) -> dict[str, float]:
        p, v, a, j = self.eval_batch(np.asarray([t], dtype=np.float64))
        return {"p": float(p[0]), "v": float(v[0]),
                "a": float(a[0]), "j": float(j[0])}


class QuarticPolynomial:
    """Velocity-boundary quartic: (p,v,a) at 0, (v,a) at T."""

    def __init__(self, start: tuple[float, float, float],
                 end_velocity: tuple[float, float], T: float) -> None:
        p0, v0, a0 = start
        v1, a1 = end_velocity
        t2, t3 = T * T, T * T * T
        self.c0, self.c1, self.c2 = p0, v0, a0 / 2.0
        b0 = v1 - self.c1 - 2 * self.c2 * T
        b1 = a1 - 2 * self.c2
        det = (3 * t2) * (12 * t2) - (4 * t3) * (6 * T)
        if abs(det) < EPS:
            self.c3 = self.c4 = 0.0
        else:
            self.c3 = (b0 * (12 * t2) - (4 * t3) * b1) / det
            self.c4 = ((3 * t2) * b1 - b0 * (6 * T)) / det

    def eval_batch(self, t: np.ndarray) -> tuple[np.ndarray, ...]:
        t2, t3 = t * t, t * t * t
        t4 = t3 * t
        p = self.c0 + self.c1 * t + self.c2 * t2 + self.c3 * t3 + self.c4 * t4
        v = self.c1 + 2 * self.c2 * t + 3 * self.c3 * t2 + 4 * self.c4 * t3
        a = 2 * self.c2 + 6 * self.c3 * t + 12 * self.c4 * t2
        j = 6 * self.c3 + 24 * self.c4 * t
        return p, v, a, j

    def eval(self, t: float) -> dict[str, float]:
        p, v, a, j = self.eval_batch(np.asarray([t], dtype=np.float64))
        return {"p": float(p[0]), "v": float(v[0]),
                "a": float(a[0]), "j": float(j[0])}


# --------------------------------------------------------------------------
# reference path (models/referencePath.js)
# --------------------------------------------------------------------------

def _normalize_angle(a: np.ndarray | float):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ReferencePath:
    """Polyline reference path with arclength frames (s, theta, kappa)."""

    def __init__(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            raise ValueError("ReferencePath expects [N>=2, 2] xy points")
        # drop near-duplicate vertices like the JS builder (> 0.2 m spacing)
        keep = [0]
        for i in range(1, pts.shape[0]):
            if np.hypot(*(pts[i] - pts[keep[-1]])) > 0.2:
                keep.append(i)
        pts = pts[keep]

        seg = np.diff(pts, axis=0)
        ds = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.x = pts[:, 0]
        self.y = pts[:, 1]
        self.total_length = float(self.s[-1])

        n = pts.shape[0]
        prev_i = np.maximum(0, np.arange(n) - 1)
        next_i = np.minimum(n - 1, np.arange(n) + 1)
        theta = np.arctan2(self.y[next_i] - self.y[prev_i],
                           self.x[next_i] - self.x[prev_i])

        kappa = np.zeros(n)
        if n > 2:
            d_theta = _normalize_angle(theta[2:] - theta[:-2])
            d_s = np.maximum(self.s[2:] - self.s[:-2], 1.0)
            kappa[1:-1] = d_theta / d_s
            kappa[0] = kappa[1]
            kappa[-1] = kappa[-2]
        self.kappa = kappa
        # unwrapped copy so np.interp reproduces lerpAngle (shortest arc)
        self._theta_unwrapped = np.unwrap(theta)

    # ------------------------------------------------------------- factories
    @classmethod
    def demo(cls, kind: str) -> "ReferencePath":
        """The four JS demo corridors (used by the port-fidelity test)."""
        pts: list[tuple[float, float]] = []
        if kind == "sCurve":
            for xx in np.arange(0.0, 1900.0 + 1e-6, 5.0):
                pts.append((xx, 72 * math.sin(xx / 260) + 18 * math.sin(xx / 75)))
        elif kind == "intersection":
            for xx in np.arange(0.0, 720.0 + 1e-6, 5.0):
                pts.append((xx, 0.0))
            cx, cy, r = 720.0, 165.0, 165.0
            a = -math.pi / 2
            while a <= 0.001:
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                a += math.pi / 90
            for yy in np.arange(165.0, 1120.0 + 1e-6, 5.0):
                pts.append((885.0, yy))
        elif kind == "uTurn":
            for xx in np.arange(0.0, 760.0 + 1e-6, 5.0):
                pts.append((xx, -135.0))
            cx, cy, r = 760.0, 0.0, 135.0
            a = -math.pi / 2
            while a <= math.pi / 2 + 0.001:
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                a += math.pi / 92
            for xx in np.arange(760.0, -130.0 - 1e-6, -5.0):
                pts.append((xx, 135.0))
        else:  # straight
            for xx in np.arange(0.0, 1900.0 + 1e-6, 5.0):
                pts.append((xx, 0.0))
        return cls(np.asarray(pts))

    # --------------------------------------------------------------- queries
    def frame_at_batch(self, s: np.ndarray) -> dict[str, np.ndarray]:
        sq = np.clip(s, 0.0, self.total_length)
        return {
            "x": np.interp(sq, self.s, self.x),
            "y": np.interp(sq, self.s, self.y),
            "theta": np.interp(sq, self.s, self._theta_unwrapped),
            "kappa": np.interp(sq, self.s, self.kappa),
            "s": sq,
        }

    def point_at_batch(self, s: np.ndarray, d: np.ndarray) -> dict[str, np.ndarray]:
        f = self.frame_at_batch(s)
        return {
            "x": f["x"] - d * np.sin(f["theta"]),
            "y": f["y"] + d * np.cos(f["theta"]),
            "theta": f["theta"],
            "kappa": f["kappa"],
            "s": f["s"],
        }

    def point_at(self, s: float, d: float = 0.0) -> dict[str, float]:
        p = self.point_at_batch(np.asarray([s], dtype=np.float64),
                                np.asarray([d], dtype=np.float64))
        return {k: float(v[0]) for k, v in p.items()}

    def project(self, x: float, y: float) -> dict[str, float]:
        """World point -> (s, d, theta_ref, kappa_ref). d > 0 is left of path."""
        ax, ay = self.x[:-1], self.y[:-1]
        bx, by = self.x[1:], self.y[1:]
        ex, ey = bx - ax, by - ay
        seg_len2 = np.maximum(ex * ex + ey * ey, EPS)
        t = np.clip(((x - ax) * ex + (y - ay) * ey) / seg_len2, 0.0, 1.0)
        px, py = ax + t * ex, ay + t * ey
        dist2 = (x - px) ** 2 + (y - py) ** 2
        i = int(np.argmin(dist2))
        s = float(self.s[i] + t[i] * math.sqrt(seg_len2[i]))
        f = self.frame_at_batch(np.asarray([s]))
        theta = float(f["theta"][0])
        d = -(x - float(f["x"][0])) * math.sin(theta) \
            + (y - float(f["y"][0])) * math.cos(theta)
        return {"s": s, "d": d, "theta": theta,
                "kappa": float(f["kappa"][0]),
                "dist": math.sqrt(float(dist2[i]))}


# --------------------------------------------------------------------------
# risk providers
# --------------------------------------------------------------------------

class RiskProvider(Protocol):
    def disks_batch(self, ts: np.ndarray) -> np.ndarray:
        """[S, M, 5] risk disks for each lookahead time in ts (seconds from
        the plan epoch). M may be 0. Columns: x, y, hard_r, sigma, gain."""
        ...


class StaticDisks:
    """Constant obstacle set == the JS demo semantics (also the
    'no prediction' arm: last observed positions frozen in time)."""

    def __init__(self, disks: np.ndarray) -> None:
        d = np.asarray(disks, dtype=np.float64)
        if d.size == 0:
            d = np.zeros((0, 5))
        if d.shape[1] == 3:  # (x, y, radius) -> JS default sigma/gain
            pad = np.full((d.shape[0], 2), np.nan)
            d = np.concatenate([d, pad], axis=1)
        self.disks = d

    def disks_batch(self, ts: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.disks, (len(ts),) + self.disks.shape)


# --------------------------------------------------------------------------
# planner (planner/frenetPlanner.js)
# --------------------------------------------------------------------------

@dataclass
class FrenetState:
    s: float = 0.0
    d: float = 0.0
    h: float = 0.0
    sd: float = 0.0   # ds/dt
    sdd: float = 0.0
    dd: float = 0.0   # dd/dt
    ddd: float = 0.0
    hd: float = 0.0
    hdd: float = 0.0


@dataclass
class PlannerConfig:
    desired_speed: float = 32.0
    max_lateral_accel: float = 4.5
    max_climb_rate: float = 8.0
    max_vertical_accel: float = 2.2
    vehicle_radius: float = 7.0
    corridor_width: float = 88.0
    altitude_band: float = 70.0
    horizon: float = 6.2
    lateral_step: float = 9.0
    altitude_step: float = 14.0
    safety_buffer: float = 8.0
    weight_jerk: float = 1.2
    weight_safety: float = 3.2
    weight_offset: float = 1.8
    weight_fuel: float = 1.4
    weight_speed: float = 0.7      # JS hardcodes 0.7 * raw.speed
    # Absolute centerline-return preference on the TERMINAL offset,
    # |target_d| / corridor half-width, NOT pool-normalized. The four
    # pool-relative costs make "park offset and fly straight" (zero jerk)
    # a stable equilibrium in clear air; this term charges parked offset
    # on an absolute scale so the gentlest return candidate eventually
    # wins. 0.0 (default) == the JS-parity behavior.
    weight_center: float = 0.0
    plan_dt: float = 0.2
    replan_period: float = 0.35
    reference_altitude: float = 120.0
    max_total_accel: Optional[float] = None   # None -> JS readConfig formula
    speed_cap_lo: float = 6.0                 # JS clamp(v, 6, 75)
    speed_cap_hi: float = 75.0
    speed_abs_max: float = 72.0               # JS Math.min(72, 1.08*desired)
    # Extra speed-grid rungs as fractions of desired_speed (clamped to the
    # envelope caps like every rung). Empty tuple == the JS original grid;
    # PRISM adds a deep-brake rung so "arrive later" is a plannable escape.
    extra_speed_factors: tuple = ()
    # Candidate duration window (JS hardcodes clamp(h∓0.7, 2.5, 10)) plus
    # optional extra absolute-seconds rungs. Defaults == the JS original;
    # PRISM raises the ceiling to the prediction horizon and keeps one
    # short rung for tight reactive maneuvers.
    duration_min: float = 2.5
    duration_max: float = 10.0
    extra_durations: tuple = ()
    # Separation standoff subtracted from clearance INSIDE the soft cost:
    # potential is maximal anywhere within `soft_standoff` of a disk edge
    # and decays by sigma only beyond it. 0 reproduces the JS original
    # (prediction sigma alone); PRISM sets the NMAC radius here so sigma
    # buys buffer ON TOP of the separation standard, not instead of it.
    soft_standoff: float = 0.0
    # Risk is evaluated only for sample times <= risk_horizon (seconds).
    # inf == JS original. PRISM plans past the prediction horizon (route
    # continuity / graceful corridor return) but ALL arms may charge risk
    # only where a forecast exists -- otherwise the CV arm could fake a
    # 30 s forecast by extrapolation while the mixture cannot.
    risk_horizon: float = math.inf
    # False == JS original (all costs are time integrals). True divides the
    # COMFORT integrals (jerk/offset/fuel) by T so mixed-duration grids
    # compete fairly; safety stays ABSOLUTE (an event mass, not a rate) --
    # otherwise long candidates always lose on accumulated fuel and the
    # far-sighted rungs can never win, killing anticipatory avoidance.
    time_normalized_costs: bool = False

    def __post_init__(self) -> None:
        if self.max_total_accel is None:
            self.max_total_accel = max(
                5.5, self.max_lateral_accel + self.max_vertical_accel + 1.2)


@dataclass
class Candidate:
    d_poly: QuinticPolynomial
    h_poly: QuinticPolynomial
    s_poly: QuarticPolynomial
    T: float
    target_d: float
    target_h: float
    target_speed: float
    samples: dict[str, np.ndarray]
    valid: bool
    collision: bool
    boundary: bool
    violation: float
    min_clearance: float
    max_lat_accel: float
    max_total_accel: float
    raw: dict[str, float]
    norm: dict[str, float] = field(default_factory=lambda: {
        "jerk": 0.0, "safety": 0.0, "offset": 0.0, "fuel": 0.0})
    total_cost: float = math.inf


@dataclass
class PlanResult:
    candidates: list[Candidate]
    best: Optional[Candidate]
    metrics: dict


def candidate_targets(state: FrenetState, config: PlannerConfig,
                      mode: str) -> dict[str, list[float]]:
    half = config.corridor_width / 2 - config.vehicle_radius - 1.5
    vertical_half = config.altitude_band / 2 - config.vehicle_radius - 1.5

    d_values: list[float] = []
    d = -half
    while d <= half + EPS:
        d_values.append(d)
        d += config.lateral_step
    d_values += [0.0, state.d, float(np.clip(state.d, -half, half))]

    if mode == "2d":
        h_values = [0.0]
    else:
        h_values = []
        h = -vertical_half
        while h <= vertical_half + EPS:
            h_values.append(h)
            h += config.altitude_step
        h_values += [0.0, state.h, float(np.clip(state.h, -vertical_half,
                                                 vertical_half))]
        h_values = unique_sorted(h_values, 0.1)

    speeds = unique_sorted([
        config.desired_speed * 0.72,
        config.desired_speed * 0.88,
        config.desired_speed,
        min(config.speed_abs_max, config.desired_speed * 1.08),
    ] + [config.desired_speed * f for f in config.extra_speed_factors], 0.1)
    speeds = [float(np.clip(v, config.speed_cap_lo, config.speed_cap_hi))
              for v in speeds]

    lo, hi = config.duration_min, config.duration_max
    durations = unique_sorted([
        float(np.clip(config.horizon - 0.7, lo, hi)),
        float(np.clip(config.horizon, lo, hi)),
        float(np.clip(config.horizon + 0.7, lo, hi)),
    ] + [float(np.clip(t, lo, hi)) for t in config.extra_durations], 0.1)

    return {"d_values": unique_sorted(d_values, 0.1), "h_values": h_values,
            "speeds": speeds, "durations": durations}


def _fuel_power(speed, sa, lat_accel, hv, ha, config, mode):
    cruise = (speed / max(config.desired_speed, 1.0)) ** 2
    if mode == "3d":
        accel = 0.075 * (sa ** 2 + lat_accel ** 2 + 1.35 * ha ** 2)
        climb = 0.18 * np.maximum(0.0, hv) ** 2
        descent = 0.055 * np.maximum(0.0, -hv) ** 2
        return 0.16 + cruise + accel + climb + descent
    return 0.16 + cruise + 0.075 * (sa ** 2 + lat_accel ** 2)


def evaluate_candidate(state: FrenetState, target_d: float, target_h: float,
                       target_speed: float, T: float, config: PlannerConfig,
                       path: ReferencePath, risk: RiskProvider,
                       mode: str) -> Candidate:
    d_poly = QuinticPolynomial((state.d, state.dd, state.ddd),
                               (target_d, 0.0, 0.0), T)
    h_poly = QuinticPolynomial((state.h, state.hd, state.hdd),
                               (target_h, 0.0, 0.0), T)
    s_poly = QuarticPolynomial((state.s, max(0.1, state.sd), state.sdd),
                               (target_speed, 0.0), T)

    steps = max(4, math.ceil(T / config.plan_dt))
    t = np.minimum(T, np.arange(steps + 1, dtype=np.float64) * config.plan_dt)

    sp, sv, sa, sj = s_poly.eval_batch(t)
    dp, dv, da, dj = d_poly.eval_batch(t)
    hp, hv, ha, hj = h_poly.eval_batch(t)

    frame = path.frame_at_batch(sp)
    theta_ref, kappa_ref = frame["theta"], frame["kappa"]
    x = frame["x"] - dp * np.sin(theta_ref)
    y = frame["y"] + dp * np.cos(theta_ref)
    z = config.reference_altitude + hp
    horizontal_speed = np.hypot(sv * (1 - kappa_ref * dp), dv)
    speed = np.hypot(horizontal_speed, hv) if mode == "3d" else horizontal_speed
    heading = theta_ref + np.arctan2(dv, np.maximum(0.1, sv))

    half = config.corridor_width / 2
    vertical_half = config.altitude_band / 2

    # ------------------------------------------------------- integral costs
    jerk_sq = sj ** 2 + dj ** 2 + (hj ** 2 if mode == "3d" else 0.0)
    cost_jerk = float(np.sum(jerk_sq) * config.plan_dt)
    offset_term = (dp / max(half, 1.0)) ** 2
    if mode == "3d":
        offset_term = offset_term + 0.55 * (hp / max(vertical_half, 1.0)) ** 2
    cost_offset = float(np.sum(offset_term) * config.plan_dt)

    violation = 0.0
    boundary = False
    lat_excess = np.abs(dp) + config.vehicle_radius - half
    mask = lat_excess > 0
    if np.any(mask):
        boundary = True
        violation += float(np.sum(lat_excess[mask]))
    if mode == "3d":
        vert_excess = np.abs(hp) + config.vehicle_radius - vertical_half
        mask = vert_excess > 0
        if np.any(mask):
            boundary = True
            violation += float(np.sum(vert_excess[mask]))

    # ------------------------------------------------ curvature (3-point)
    curvature = np.zeros_like(x)
    if len(x) >= 3:
        axv, ayv = x[:-2], y[:-2]
        bxv, byv = x[1:-1], y[1:-1]
        cxv, cyv = x[2:], y[2:]
        ab = np.hypot(bxv - axv, byv - ayv)
        bc = np.hypot(cxv - bxv, cyv - byv)
        ac = np.hypot(cxv - axv, cyv - ayv)
        cross = (bxv - axv) * (cyv - ayv) - (byv - ayv) * (cxv - axv)
        denom = np.maximum(ab * bc * ac, EPS)
        curvature[1:-1] = np.clip(2 * cross / denom, -0.25, 0.25)
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

    lat_accel = horizontal_speed ** 2 * np.abs(curvature)
    total_accel = (np.hypot(np.hypot(sa, lat_accel), ha) if mode == "3d"
                   else np.hypot(sa, lat_accel))
    max_lat_accel = float(np.max(lat_accel))
    max_tot_accel = float(np.max(total_accel))

    cost_fuel = float(np.sum(
        _fuel_power(speed, sa, lat_accel, hv, ha, config, mode))
        * config.plan_dt)

    lat_over = lat_accel - config.max_lateral_accel
    violation += float(np.sum(lat_over[lat_over > 0]) * 0.7)
    if mode == "3d":
        climb_over = np.abs(hv) - config.max_climb_rate
        violation += float(np.sum(climb_over[climb_over > 0]) * 0.9)
        vert_over = np.abs(ha) - config.max_vertical_accel
        violation += float(np.sum(vert_over[vert_over > 0]) * 0.8)
    sa_over = np.abs(sa) - config.max_total_accel
    violation += float(np.sum(sa_over[sa_over > 0]) * 0.5)

    # ----------------------------------------------------- risk disk sweep
    # only samples within the risk horizon see (and pay for) risk disks
    rmask = t <= config.risk_horizon
    tr = t[rmask]
    xr, yr = x[rmask], y[rmask]
    zr = z[rmask]
    disks = risk.disks_batch(tr)                      # [Sr, M, 5]
    collision = False
    min_clearance = math.inf
    cost_safety = 0.0
    if disks.shape[0] > 0 and disks.shape[1] > 0:
        dx = xr[:, None] - disks[:, :, 0]
        dy = yr[:, None] - disks[:, :, 1]
        if mode == "3d":
            dz = zr[:, None] - config.reference_altitude  # disks fly at ref alt
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            dist = np.hypot(dx, dy)
        hard_r = disks[:, :, 2]
        clearance = dist - hard_r - config.vehicle_radius
        min_clearance = float(np.min(clearance))

        hit = clearance < 0
        if np.any(hit):
            collision = True
            violation += float(np.sum(40.0 + np.abs(clearance[hit]) * 7.0))

        sigma = disks[:, :, 3]
        sigma = np.where(np.isnan(sigma),
                         np.maximum(10.0, hard_r + config.safety_buffer),
                         sigma)
        gain = disks[:, :, 4]
        gain = np.where(np.isnan(gain),
                        (hard_r + config.vehicle_radius) / 12.0,
                        gain)
        soft = np.maximum(0.0, clearance - config.soft_standoff)
        cost_safety = float(np.sum(
            np.exp(-soft ** 2 / (2 * sigma ** 2)) * gain) * config.plan_dt)

    speed_track = ((target_speed - config.desired_speed)
                   / max(1.0, config.desired_speed)) ** 2
    if config.time_normalized_costs:
        cost_jerk /= T
        cost_offset /= T
        cost_fuel /= T
    valid = (violation <= 1e-6) and (not collision) and (not boundary)

    samples = {"t": t, "s": sp, "d": dp, "h": hp, "x": x, "y": y, "z": z,
               "theta": heading, "speed": speed,
               "horizontal_speed": horizontal_speed,
               "sv": sv, "sa": sa, "dv": dv, "da": da,
               "curvature": curvature, "lat_accel": lat_accel,
               "total_accel": total_accel}

    return Candidate(
        d_poly=d_poly, h_poly=h_poly, s_poly=s_poly, T=T,
        target_d=target_d, target_h=target_h, target_speed=target_speed,
        samples=samples, valid=valid, collision=collision, boundary=boundary,
        violation=violation, min_clearance=min_clearance,
        max_lat_accel=max_lat_accel, max_total_accel=max_tot_accel,
        raw={"jerk": cost_jerk, "safety": cost_safety, "offset": cost_offset,
             "fuel": cost_fuel, "speed": speed_track})


def normalize_and_score(pool: list[Candidate], config: PlannerConfig,
                        emergency_mode: bool) -> None:
    max_jerk = max(max((c.raw["jerk"] for c in pool), default=0.0), 1e-6)
    max_safety = max(max((c.raw["safety"] for c in pool), default=0.0), 1e-6)
    max_offset = max(max((c.raw["offset"] for c in pool), default=0.0), 1e-6)
    max_fuel = max(max((c.raw["fuel"] for c in pool), default=0.0), 1e-6)
    half_c = max(config.corridor_width / 2 - config.vehicle_radius - 1.5, 1.0)

    for c in pool:
        c.norm["jerk"] = c.raw["jerk"] / max_jerk
        c.norm["safety"] = c.raw["safety"] / max_safety
        c.norm["offset"] = c.raw["offset"] / max_offset
        c.norm["fuel"] = c.raw["fuel"] / max_fuel
        c.norm["center"] = min(1.0, abs(c.target_d) / half_c)
        weighted = (config.weight_jerk * c.norm["jerk"]
                    + config.weight_safety * c.norm["safety"]
                    + config.weight_offset * c.norm["offset"]
                    + config.weight_fuel * c.norm["fuel"]
                    + config.weight_speed * c.raw["speed"]
                    + config.weight_center * c.norm["center"])
        if config.time_normalized_costs:
            # per-second costs tie across durations in clear air; prefer
            # the longest plan (stable route, less replan churn)
            weighted -= 1e-4 * c.T
        emergency = (c.violation * 2.0 + (500.0 if c.collision else 0.0)) \
            if emergency_mode else 0.0
        c.total_cost = weighted + emergency


def plan_trajectory(state: FrenetState, config: PlannerConfig,
                    path: ReferencePath, risk: RiskProvider,
                    mode: str = "2d") -> PlanResult:
    targets = candidate_targets(state, config, mode)
    candidates: list[Candidate] = []
    for T in targets["durations"]:
        for target_speed in targets["speeds"]:
            for target_d in targets["d_values"]:
                for target_h in targets["h_values"]:
                    candidates.append(evaluate_candidate(
                        state, target_d, target_h, target_speed, T,
                        config, path, risk, mode))

    valid = [c for c in candidates if c.valid]
    pool = valid if valid else candidates
    normalize_and_score(pool, config, emergency_mode=not valid)
    pool = sorted(pool, key=lambda c: c.total_cost)
    best = pool[0] if pool else None

    return PlanResult(
        candidates=candidates, best=best,
        metrics={"total": len(candidates), "valid": len(valid),
                 "min_clearance": best.min_clearance if best else math.inf,
                 "max_lat_accel": best.max_lat_accel if best else 0.0})


def eval_trajectory_at(traj: Candidate, t: float) -> FrenetState:
    tt = float(np.clip(t, 0.0, traj.T))
    sv = traj.s_poly.eval(tt)
    dv = traj.d_poly.eval(tt)
    hv = traj.h_poly.eval(tt)
    return FrenetState(
        s=sv["p"], d=dv["p"], h=hv["p"],
        sd=max(0.0, sv["v"]), sdd=sv["a"],
        dd=dv["v"], ddd=dv["a"],
        hd=hv["v"], hdd=hv["a"])
