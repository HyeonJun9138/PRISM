"""Inverse construction of encounter geometries (doc 3.4).

WHY BACKWARDS
-------------
If traffic is placed at random, a dangerous encounter essentially never
happens: the probability that two independently placed, independently headed
aircraft pass within a few hundred metres inside a 60 s window is tiny. A
dataset built that way is almost entirely "nothing happens", so the risk-map
and the collision-avoidance policy downstream would train on noise. Doc 3.4
therefore inverts the problem: the OUTCOME (closest-point-of-approach distance
and time) is sampled first and the initial condition is solved for in closed
form.

THE CLOSED FORM
---------------
Work in the relative frame. With ownship position ``p_o`` and velocity ``v_o``
at ``t = 0`` and an intruder at ``p_i`` with velocity ``v_i``, define

    p_rel(t) = p_rel(0) + v_rel * t,      v_rel = v_i - v_o.

For straight-line motion the squared separation ``|p_rel(t)|^2`` is a convex
parabola in ``t``; its minimum is where its derivative vanishes:

    d/dt |p_rel|^2 = 2 (p_rel(0) + v_rel t) . v_rel = 0
    =>  t_cpa = - (p_rel(0) . v_rel) / |v_rel|^2                        (1)
        d_cpa = | p_rel(0) + v_rel * t_cpa |                            (2)

Now go the other way. Choose the target ``t*`` [s] and ``d*`` [m], choose
``v_rel`` (fixed once the intruder speed and heading are chosen), and let
``n_hat`` be a UNIT vector perpendicular to ``v_rel`` (two choices, decided by
a coin flip -- this is what puts the intruder on the left or the right). Set

    p_rel(0) = -v_rel * t*  +  d* * n_hat                               (3)
    p_i(0)   = p_o + p_rel(0)

Substituting (3) into (1), and using ``n_hat . v_rel = 0``:

    t_cpa = -( -|v_rel|^2 t* + d* (n_hat . v_rel) ) / |v_rel|^2 = t*
    p_rel(t*) = -v_rel t* + d* n_hat + v_rel t* = d* n_hat
    d_cpa = |d* n_hat| = d*                                            QED

So (3) reproduces the requested pair EXACTLY, for any ``v_rel != 0``. The
construction is verified numerically inside :func:`build_encounter` and raises
if the round trip is off by more than ``tol`` -- a silent geometry bug would
poison every downstream metric.

APPROACH ANGLE
--------------
``chi`` is the absolute angle between the two VELOCITY vectors (doc 3.4 and
:class:`~core.types.GeometryType`):

    HEAD_ON     150 deg .. 180 deg
    CROSSING     30 deg .. 150 deg
    OVERTAKING    0 deg ..  30 deg

The intruder heading is ``psi_i = psi_o +/- chi`` with the sign drawn as a
coin flip, so both mirror images of every geometry appear in the dataset.

DEGENERACY
----------
``|v_rel| -> 0`` (a co-speed overtaking case) has no well-defined CPA: the two
bodies never approach. The intruder speed is therefore drawn subject to
``|v_rel| >= v_rel_min_ms``, retrying with a fresh draw of the whole geometry
if the platform envelope cannot satisfy it at that ``chi``.

WHAT THE SAMPLED VALUES MEAN AFTERWARDS
---------------------------------------
From ``t = 0`` the agents evolve FREELY: they turn, accelerate and react
according to their own stochastic decision rules (doc 3.3) and are pushed by
wind (doc 3.6). The realised CPA will therefore differ from ``d*`` -- often by
a lot. ``d*`` and ``t*`` are the DESIGN INTENT, recorded as episode metadata
for stratified analysis (doc 3.4 "side effect"); the REALISED minimum
separation is measured after the fact by :mod:`analysis.metrics`. The two must
never be confused.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from core.state import AgentState, PerformanceEnvelope, angle_diff, wrap_angle
from core.types import GEOMETRY_NAMES, GeometryType
from scenario.config import EncounterConfig

__all__ = [
    "GEOMETRY_ANGLE_RANGES_RAD",
    "DESIGNED_GEOMETRIES",
    "compute_cpa",
    "classify_geometry",
    "sample_geometry",
    "sample_approach_angle",
    "build_encounter",
    "geometry_of_meta",
]


#: Approach-angle band [rad] per geometry class, as (low, high) inclusive of
#: the low edge and exclusive of the high edge except for HEAD_ON, which is
#: closed at pi. Doc 3.4.
GEOMETRY_ANGLE_RANGES_RAD: dict[int, tuple[float, float]] = {
    int(GeometryType.OVERTAKING): (0.0, math.radians(30.0)),
    int(GeometryType.CROSSING): (math.radians(30.0), math.radians(150.0)),
    int(GeometryType.HEAD_ON): (math.radians(150.0), math.pi),
}

#: Geometries that can actually be designed. BACKGROUND is a label for traffic
#: that was NOT designed, so it is never a target of :func:`build_encounter`.
DESIGNED_GEOMETRIES: tuple[GeometryType, ...] = (
    GeometryType.HEAD_ON,
    GeometryType.CROSSING,
    GeometryType.OVERTAKING,
)


# --------------------------------------------------------------------------
# Forward geometry (also used by analysis.metrics)
# --------------------------------------------------------------------------


def compute_cpa(
    p_rel: Sequence[float],
    v_rel: Sequence[float],
    clamp_future: bool = False,
) -> tuple[float, float]:
    """Closest point of approach of two bodies in straight-line relative motion.

    Implements equations (1) and (2) of the module docstring.

    Args:
        p_rel: relative position ``(dx, dy)`` [m] at the evaluation instant,
            intruder minus ownship.
        v_rel: relative velocity ``(dvx, dvy)`` [m/s], intruder minus ownship.
        clamp_future: if True the time to CPA is clamped to ``>= 0``, i.e. a
            receding pair reports its CURRENT separation at ``tcpa = 0``. If
            False (the default, and what the inverse construction verifies
            against) the raw stationary point is returned and may be negative,
            meaning the CPA already happened.

    Returns:
        ``(cpa_dist, tcpa)`` with the distance [m] at the closest approach and
        the time [s] until it. For a zero relative velocity the pair never
        approaches: the current separation is returned with ``tcpa = 0``.
    """
    px = float(p_rel[0])
    py = float(p_rel[1])
    vx = float(v_rel[0])
    vy = float(v_rel[1])
    vv = vx * vx + vy * vy
    if vv <= 0.0:
        return math.hypot(px, py), 0.0
    t = -(px * vx + py * vy) / vv
    if clamp_future and t < 0.0:
        t = 0.0
    cx = px + vx * t
    cy = py + vy * t
    return math.hypot(cx, cy), t


def classify_geometry(own_psi: float, intruder_psi: float) -> GeometryType:
    """Label an encounter geometry from the two headings (doc 3.4).

    Used for BACKGROUND traffic, which is never designed but still deserves a
    geometry label in the metadata so that stratified analysis can include it.

    Args:
        own_psi: ownship heading [rad], CCW from +X.
        intruder_psi: intruder heading [rad], CCW from +X.

    Returns:
        :class:`~core.types.GeometryType` -- OVERTAKING below 30 deg of
        heading difference, CROSSING from 30 to 150 deg, HEAD_ON at 150 deg
        and above.
    """
    chi = abs(angle_diff(float(intruder_psi), float(own_psi)))
    if chi < GEOMETRY_ANGLE_RANGES_RAD[int(GeometryType.OVERTAKING)][1]:
        return GeometryType.OVERTAKING
    if chi < GEOMETRY_ANGLE_RANGES_RAD[int(GeometryType.CROSSING)][1]:
        return GeometryType.CROSSING
    return GeometryType.HEAD_ON


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------


def sample_geometry(
    cfg: EncounterConfig,
    rng: np.random.Generator,
) -> GeometryType:
    """Draw a designed geometry class from the configured weights.

    BACKGROUND is excluded and its weight ignored: it labels undesigned
    traffic, so designing it would be a contradiction. If the remaining
    weights are all zero the three classes are drawn uniformly.

    Args:
        cfg: encounter configuration holding ``geometry_weights``.
        rng: generator; pass the ``'scenario'`` stream.

    Returns:
        One of :data:`DESIGNED_GEOMETRIES`.
    """
    weights = [max(0.0, float(cfg.geometry_weights.get(int(g), 0.0)))
               for g in DESIGNED_GEOMETRIES]
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0] * len(DESIGNED_GEOMETRIES)
        total = float(len(DESIGNED_GEOMETRIES))
    u = float(rng.random()) * total
    acc = 0.0
    for g, w in zip(DESIGNED_GEOMETRIES, weights):
        acc += w
        if u < acc:
            return g
    return DESIGNED_GEOMETRIES[-1]


def sample_approach_angle(
    geometry: GeometryType,
    rng: np.random.Generator,
) -> float:
    """Draw the absolute approach angle ``chi`` [rad] for a geometry class.

    Args:
        geometry: designed geometry class.
        rng: generator; pass the ``'scenario'`` stream.

    Returns:
        ``chi`` [rad] in ``[0, pi]``, uniform inside the class band.

    Raises:
        ValueError: for :attr:`~core.types.GeometryType.BACKGROUND`, which has
            no band.
    """
    key = int(geometry)
    if key not in GEOMETRY_ANGLE_RANGES_RAD:
        raise ValueError(
            f"geometry {GEOMETRY_NAMES.get(key, key)} has no approach-angle "
            "band; only HEAD_ON / CROSSING / OVERTAKING can be designed"
        )
    lo, hi = GEOMETRY_ANGLE_RANGES_RAD[key]
    return float(rng.uniform(lo, hi))


# --------------------------------------------------------------------------
# Inverse construction
# --------------------------------------------------------------------------


def build_encounter(
    ownship_state: AgentState,
    geometry: GeometryType,
    cfg: EncounterConfig,
    envelope: PerformanceEnvelope,
    rng: np.random.Generator,
    *,
    speed_frac_min: float = 0.30,
    v_rel_min_ms: float = 2.0,
    min_range_m: float = 0.0,
    max_range_m: Optional[float] = None,
    max_attempts: int = 64,
    tol: float = 1e-6,
) -> tuple[AgentState, dict[str, Any]]:
    """Solve for an intruder initial state that yields a requested CPA.

    Implements equation (3) of the module docstring. Draw order per attempt is
    fixed and MUST NOT be reordered, because reproducibility of a dataset
    depends on the exact sequence of draws from the ``'scenario'`` stream:

        1. ``d*``      target CPA distance [m], uniform in the config range
        2. ``t*``      target time to CPA [s], uniform in the config range
        3. ``chi``     approach angle [rad], uniform in the geometry band
        4. sign of chi (left/right mirror of the geometry)
        5. side of the miss (which side of ``v_rel`` the ownship passes)
        6. ``v_i``     intruder airspeed [m/s], uniform in the allowed band

    The ownship velocity used is its AIR-relative velocity: the design is a
    pure kinematic construction and deliberately ignores wind, since the
    agents' free evolution invalidates the exact geometry anyway.

    Args:
        ownship_state: ownship state at ``t = 0``; only position, speed and
            heading are read.
        geometry: which geometry class to build; BACKGROUND is rejected.
        cfg: encounter configuration supplying the CPA/TCPA sampling ranges.
        envelope: performance envelope of the INTRUDER platform, used for the
            speed band.
        rng: generator; pass the ``'scenario'`` stream.
        speed_frac_min: lower edge of the intruder speed band, expressed as a
            fraction [-] of ``envelope.v_max``, floored by ``envelope.v_min``.
            The default 0.30 keeps a designed intruder in transit rather than
            hovering, which would make the encounter meaningless.
        v_rel_min_ms: minimum acceptable relative speed [m/s]; guards the
            ``|v_rel| -> 0`` degeneracy.
        min_range_m: minimum acceptable initial separation [m]. Use it to stop
            an encounter starting already on top of the ownship.
        max_range_m: optional maximum initial separation [m]; use the region
            half-diagonal to keep the intruder inside the simulated area.
        max_attempts: number of resampling attempts [-] before giving up.
        tol: absolute tolerance [m] and [s] for the numerical self-check.

    Returns:
        ``(intruder_state, meta)``. ``intruder_state`` is an
        :class:`~core.state.AgentState` with zero initial turn rate and
        acceleration. ``meta`` is a JSON-safe dict:

            ``geometry`` int, ``geometry_name`` str,
            ``cpa_dist`` design d* [m], ``tcpa`` design t* [s],
            ``chi_rad`` / ``chi_deg`` approach angle,
            ``chi_sign`` +1 if the intruder heading is CCW of the ownship's,
            ``side_sign`` +1 if the miss vector is +90 deg from ``v_rel``,
            ``v_i`` intruder airspeed [m/s], ``psi_i`` heading [rad],
            ``v_rel`` [m/s] pair, ``speed_rel`` [m/s],
            ``range0`` initial separation [m], ``bearing0`` [rad] absolute
            bearing of the intruder from the ownship,
            ``attempts`` [-], ``verify_cpa_err`` [m], ``verify_tcpa_err`` [s].

    Raises:
        ValueError: if ``geometry`` is BACKGROUND, or no attempt satisfied the
            relative-speed and range constraints.
        AssertionError: never; a failed self-check raises ``ValueError`` with
            the measured error, so a geometry bug cannot pass silently.
    """
    if int(geometry) not in GEOMETRY_ANGLE_RANGES_RAD:
        raise ValueError(
            "build_encounter: BACKGROUND traffic is not designed; use "
            "classify_geometry() to label it instead"
        )

    p_ox = float(ownship_state.x)
    p_oy = float(ownship_state.y)
    psi_o = float(ownship_state.psi)
    v_o = float(ownship_state.v)
    v_ox = v_o * math.cos(psi_o)
    v_oy = v_o * math.sin(psi_o)

    v_hi = float(envelope.v_max)
    v_lo = max(float(envelope.v_min), float(speed_frac_min) * v_hi)
    if v_lo > v_hi:
        v_lo = v_hi

    d_lo = float(cfg.cpa_dist_min_m)
    d_hi = float(cfg.cpa_dist_max_m)
    t_lo = float(cfg.tcpa_min_s)
    t_hi = float(cfg.tcpa_max_s)

    last_reason = "no attempt was made"
    n_attempts = max(1, int(max_attempts))

    for attempt in range(1, n_attempts + 1):
        # -- 1..5: geometry draws -----------------------------------------
        d_star = float(rng.uniform(d_lo, d_hi))
        t_star = float(rng.uniform(t_lo, t_hi))
        chi = sample_approach_angle(geometry, rng)
        chi_sign = 1.0 if float(rng.random()) < 0.5 else -1.0
        side_sign = 1.0 if float(rng.random()) < 0.5 else -1.0

        psi_i = wrap_angle(psi_o + chi_sign * chi)

        # -- 6: intruder speed, subject to the |v_rel| guard ---------------
        v_i = float(rng.uniform(v_lo, v_hi)) if v_hi > v_lo else v_hi
        speed_rel = _relative_speed(v_i, psi_i, v_ox, v_oy)
        if speed_rel < float(v_rel_min_ms):
            # |v_rel|^2 is a convex parabola in v_i, so its maximum over the
            # closed speed band is attained at one of the two endpoints. Pick
            # the better endpoint deterministically -- no extra draw, so the
            # stream advance stays independent of this branch.
            s_lo = _relative_speed(v_lo, psi_i, v_ox, v_oy)
            s_hi = _relative_speed(v_hi, psi_i, v_ox, v_oy)
            if s_hi >= s_lo:
                v_i, speed_rel = v_hi, s_hi
            else:
                v_i, speed_rel = v_lo, s_lo
            if speed_rel < float(v_rel_min_ms):
                last_reason = (
                    f"relative speed {speed_rel:.3f} m/s below the "
                    f"{v_rel_min_ms} m/s floor at chi={math.degrees(chi):.1f} deg"
                )
                continue

        v_ix = v_i * math.cos(psi_i)
        v_iy = v_i * math.sin(psi_i)
        v_rx = v_ix - v_ox
        v_ry = v_iy - v_oy

        # -- equation (3) --------------------------------------------------
        # n_hat: unit normal to v_rel, side picked by the coin flip. The
        # +90 deg rotation of (vx, vy) is (-vy, vx).
        n_x = side_sign * (-v_ry) / speed_rel
        n_y = side_sign * (v_rx) / speed_rel
        p_rx = -v_rx * t_star + d_star * n_x
        p_ry = -v_ry * t_star + d_star * n_y

        p_ix = p_ox + p_rx
        p_iy = p_oy + p_ry
        range0 = math.hypot(p_rx, p_ry)

        if range0 < float(min_range_m):
            last_reason = (
                f"initial range {range0:.1f} m below min_range_m="
                f"{min_range_m:.1f} m"
            )
            continue
        if max_range_m is not None and range0 > float(max_range_m):
            last_reason = (
                f"initial range {range0:.1f} m above max_range_m="
                f"{float(max_range_m):.1f} m"
            )
            continue

        state = AgentState(
            x=p_ix, y=p_iy, v=v_i, psi=psi_i, omega=0.0, a=0.0
        )

        # -- numerical self-check, from the FINAL state --------------------
        # Recomputing the velocity from the stored (wrapped) heading also
        # catches any error introduced by the angle round trip.
        vx_chk = state.v * math.cos(state.psi)
        vy_chk = state.v * math.sin(state.psi)
        cpa_chk, tcpa_chk = compute_cpa(
            (state.x - p_ox, state.y - p_oy),
            (vx_chk - v_ox, vy_chk - v_oy),
            clamp_future=False,
        )
        err_d = abs(cpa_chk - d_star)
        err_t = abs(tcpa_chk - t_star)
        if err_d > tol or err_t > tol:
            raise ValueError(
                "build_encounter: inverse construction failed its self-check "
                f"(|d - d*| = {err_d:.3e} m, |t - t*| = {err_t:.3e} s, "
                f"tol = {tol:.1e}). Requested d*={d_star:.3f} m, "
                f"t*={t_star:.3f} s, chi={math.degrees(chi):.2f} deg, "
                f"|v_rel|={speed_rel:.3f} m/s."
            )

        meta: dict[str, Any] = {
            "geometry": int(geometry),
            "geometry_name": GEOMETRY_NAMES[int(geometry)],
            "cpa_dist": float(d_star),
            "tcpa": float(t_star),
            "chi_rad": float(chi),
            "chi_deg": float(math.degrees(chi)),
            "chi_sign": float(chi_sign),
            "side_sign": float(side_sign),
            "v_i": float(v_i),
            "psi_i": float(state.psi),
            "v_rel": [float(v_rx), float(v_ry)],
            "speed_rel": float(speed_rel),
            "range0": float(range0),
            "bearing0": float(wrap_angle(math.atan2(p_ry, p_rx))),
            "attempts": int(attempt),
            "verify_cpa_err": float(err_d),
            "verify_tcpa_err": float(err_t),
        }
        return state, meta

    raise ValueError(
        f"build_encounter: no valid {GEOMETRY_NAMES[int(geometry)]} encounter "
        f"after {n_attempts} attempts (last reason: {last_reason}). Check the "
        "intruder speed band against the ownship speed and the range limits."
    )


def _relative_speed(
    v_i: float,
    psi_i: float,
    v_ox: float,
    v_oy: float,
) -> float:
    """Magnitude [m/s] of ``v_i - v_o`` for an intruder speed and heading.

    Args:
        v_i: intruder airspeed [m/s].
        psi_i: intruder heading [rad], CCW from +X.
        v_ox: ownship velocity East component [m/s].
        v_oy: ownship velocity North component [m/s].

    Returns:
        Relative speed [m/s], always non-negative.
    """
    return math.hypot(v_i * math.cos(psi_i) - v_ox, v_i * math.sin(psi_i) - v_oy)


def geometry_of_meta(meta: Mapping[str, Any]) -> GeometryType:
    """Read the geometry class back out of a :func:`build_encounter` meta dict.

    Args:
        meta: the metadata dict returned by :func:`build_encounter`.

    Returns:
        The :class:`~core.types.GeometryType` recorded in the dict.
    """
    return GeometryType(int(meta["geometry"]))
