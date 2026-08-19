"""Risk providers -- the ONLY component that differs between experiment arms.

Every arm feeds the same planner (frenet.plan_trajectory) the same disk
format; what changes is where the disks sit over lookahead time t and how
wide/heavy they are:

  none   last observed position, frozen in time (the JS demo semantics)
  cv     constant-velocity extrapolation, uncertainty growing linearly in t
  model  PRISM K-branch mixture: one disk per (intruder, branch), centered on
         mu_k(t), sized by the predicted sigma_k(t), weighted by pi_k

Fairness rules baked in here (referenced by the paper's experiment section):
  * hard_radius is identical across arms (physical extent is arm-agnostic);
  * the soft-potential GAIN follows the same base formula across arms; the
    model arm splits it by branch probability (sum over branches == base);
  * the baseline sigma for none/cv is a swept experiment knob (default:
    NMAC radius), NOT tuned per scenario.

All positions are world metres; time t is seconds after the last update
(the plan epoch). Observation cadence is LOG_DT (0.5 s), PRED_STEPS==30.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

LOG_DT = 0.5          # s between logged observation rows (dataGen 2 Hz)
PRED_STEPS = 30       # model horizon steps (15 s)


def default_gain(hard_r: float, vehicle_radius: float) -> float:
    """The JS demo's obstacleSizeGain -- kept as the shared base gain."""
    return (hard_r + vehicle_radius) / 12.0


class BaseRisk:
    """Shared plumbing: observation extraction + constant-disk fallback."""

    name = "base"

    def __init__(self, hard_radius: float = 30.0, vehicle_radius: float = 6.0,
                 sigma_base: float = 152.4, sigma_floor: float = 10.0) -> None:
        # hard_radius is a deconfliction FLOOR (never plan closer than
        # this), deliberately far below the NMAC radius: in dense traffic a
        # hard NMAC tube per intruder blocks the whole corridor and forces
        # permanent emergency mode, where violation depth -- not the arms'
        # uncertainty fields -- decides the ranking. Separation is instead
        # produced by the soft sigma potential (the experimental variable);
        # NMAC/LoWC stay OUTCOME metrics measured by the simulator.
        self.hard_radius = float(hard_radius)
        self.vehicle_radius = float(vehicle_radius)
        self.sigma_base = float(sigma_base)     # none/cv baseline sigma [m]
        self.sigma_floor = float(sigma_floor)
        self._disks0 = np.zeros((0, 5))         # current-position disks

    # ------------------------------------------------------------ helpers
    def _observed_now(self, ep_dict: dict, t0: int) -> tuple[np.ndarray, list[int]]:
        det = np.asarray(ep_dict["obs_detected"][t0], dtype=bool)
        slots = [a for a in range(det.shape[0]) if det[a]]
        pos = np.asarray([ep_dict["obs_pos"][t0, a] for a in slots],
                         dtype=np.float64) if slots else np.zeros((0, 2))
        return pos, slots

    def _const_rows(self, pos: np.ndarray, sigma: float) -> np.ndarray:
        m = pos.shape[0]
        if m == 0:
            return np.zeros((0, 5))
        rows = np.empty((m, 5))
        rows[:, 0:2] = pos
        rows[:, 2] = self.hard_radius
        rows[:, 3] = max(self.sigma_floor, sigma)
        rows[:, 4] = default_gain(self.hard_radius, self.vehicle_radius)
        return rows

    # ---------------------------------------------------------- interface
    def update(self, ep_dict: dict, t0: int,
               preds: Optional[dict] = None) -> None:
        raise NotImplementedError

    def disks_batch(self, ts: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._disks0,
                               (len(ts),) + self._disks0.shape)

    def describe(self) -> dict:
        return {"name": self.name, "n_disks": int(self._disks0.shape[0])}

    def viz(self) -> dict:
        """JSON-safe sketch of what THIS arm feeds the planner, so the
        console can draw the planner's world-view (not the model's)."""
        return {"kind": self.name,
                "pts": [[round(float(r[0]), 1), round(float(r[1]), 1),
                         round(float(r[3]), 1)] for r in self._disks0]}


class NoPredictionRisk(BaseRisk):
    """Arm 1: reactive. Last observed positions frozen over the horizon."""

    name = "none"

    def update(self, ep_dict: dict, t0: int,
               preds: Optional[dict] = None) -> None:
        pos, _ = self._observed_now(ep_dict, t0)
        self._disks0 = self._const_rows(pos, self.sigma_base)


class CVRisk(BaseRisk):
    """Arm 2: constant-velocity extrapolation from observed positions.

    Velocity is a least-squares linear fit over the DETECTED rows of the
    last `window_steps` logged samples (position-only sensors -- same
    doctrine as the prediction model). Two-point differencing is NOT used:
    with tens-of-metres measurement noise it turns into a velocity error of
    dozens of m/s and the extrapolation thrashes frame to frame; the
    windowed fit averages the noise down and keeps the arm a fair,
    non-strawman baseline. Uncertainty grows linearly:
    sigma(t) = sqrt(sigma0^2 + (growth*t)^2).
    """

    name = "cv"

    def __init__(self, sigma_growth: float = 4.0, sigma0: float = 15.0,
                 window_steps: int = 8, **kw) -> None:
        super().__init__(**kw)
        self.sigma_growth = float(sigma_growth)   # m per s of lookahead
        self.sigma0 = float(sigma0)
        self.window_steps = int(window_steps)     # 8 x 0.5 s = 4 s window
        self._pos = np.zeros((0, 2))
        self._vel = np.zeros((0, 2))

    def update(self, ep_dict: dict, t0: int,
               preds: Optional[dict] = None) -> None:
        det = np.asarray(ep_dict["obs_detected"], dtype=bool)
        obs = np.asarray(ep_dict["obs_pos"], dtype=np.float64)
        j0 = max(0, t0 - self.window_steps + 1)
        pos_list, vel_list = [], []
        for a in range(det.shape[1]):
            if not det[t0, a]:
                continue
            idx = [j for j in range(j0, t0 + 1) if det[j, a]]
            v = np.zeros(2)
            if len(idx) >= 2:
                ts = (np.asarray(idx, dtype=np.float64) - t0) * LOG_DT
                ps = obs[idx, a, :]
                tc = ts - ts.mean()
                denom = float((tc ** 2).sum())
                if denom > 1e-9:
                    v = ((tc[:, None] * (ps - ps.mean(axis=0))).sum(axis=0)
                         / denom)
            pos_list.append(obs[t0, a])
            vel_list.append(v)
        self._pos = np.asarray(pos_list) if pos_list else np.zeros((0, 2))
        self._vel = np.asarray(vel_list) if vel_list else np.zeros((0, 2))
        self._disks0 = self._const_rows(self._pos, self.sigma0)

    def disks_batch(self, ts: np.ndarray) -> np.ndarray:
        m = self._pos.shape[0]
        s = len(ts)
        if m == 0:
            return np.zeros((s, 0, 5))
        t = np.asarray(ts, dtype=np.float64)[:, None]            # [S,1]
        out = np.empty((s, m, 5))
        out[:, :, 0] = self._pos[None, :, 0] + self._vel[None, :, 0] * t
        out[:, :, 1] = self._pos[None, :, 1] + self._vel[None, :, 1] * t
        out[:, :, 2] = self.hard_radius
        out[:, :, 3] = np.maximum(
            self.sigma_floor,
            np.sqrt(self.sigma0 ** 2 + (self.sigma_growth * t) ** 2))
        out[:, :, 4] = default_gain(self.hard_radius, self.vehicle_radius)
        return out

    def viz(self) -> dict:
        return {"kind": self.name,
                "tracks": [[round(float(p[0]), 1), round(float(p[1]), 1),
                            round(float(v[0]), 2), round(float(v[1]), 2)]
                           for p, v in zip(self._pos, self._vel)],
                "sigma0": self.sigma0, "growth": self.sigma_growth}


class ModelRisk(BaseRisk):
    """Arm 3: PRISM mixture. One disk per (intruder, branch) at mu_k(t).

    sigma_k(t) is the isotropic equivalent of the predicted per-axis sigmas:
    sqrt((sx^2 + sy^2)/2) == sqrt(trace(Sigma)/2), which is rotation
    invariant, so the target-frame axes of the network output are safe to
    use directly. Branch gain = base_gain * pi_k (sums to the base gain).
    Branches below pi_min are dropped.
    """

    name = "model"

    def __init__(self, pi_min: float = 0.05, sigma_scale: float = 1.0,
                 **kw) -> None:
        super().__init__(**kw)
        self.pi_min = float(pi_min)
        # Post-hoc temperature from experiments/recalibrate.py, which fits a
        # single scalar as sigma -> T * sigma. 1.0 leaves the raw network
        # sigma untouched (the condition every batch before 2026-08-17 ran
        # under); 1.665 is the value fitted on the OOD-adjacent calibration
        # runs. Applying it here is exactly equivalent to scaling both axis
        # sigmas, because the isotropic equivalent is homogeneous of degree 1.
        self.sigma_scale = float(sigma_scale)
        self._anchor = np.zeros((0, 2))    # [M,2] obs position at t=0
        self._mu = np.zeros((0, PRED_STEPS, 2))
        self._sigma = np.zeros((0, PRED_STEPS))
        self._gain = np.zeros((0,))
        self._static = np.zeros((0, 5))    # unpredicted-but-detected tracks
        self.using_fallback = True

    def update(self, ep_dict: dict, t0: int,
               preds: Optional[dict] = None) -> None:
        preds = preds or {}
        anchors, mus, sigmas, gains = [], [], [], []
        base = default_gain(self.hard_radius, self.vehicle_radius)
        covered: set[int] = set()
        for slot, p in preds.items():
            anchor = np.asarray(ep_dict["obs_pos"][t0, int(slot)],
                                dtype=np.float64)
            if not np.all(np.isfinite(anchor)):
                continue
            mu_w = np.asarray(p["mu_w"], dtype=np.float64)       # [K,30,2]
            sig = np.asarray(p["sigma"], dtype=np.float64)       # [K,30,2]
            pi = np.asarray(p["pi"], dtype=np.float64)           # [K]
            iso = (np.sqrt((sig[:, :, 0] ** 2 + sig[:, :, 1] ** 2) / 2.0)
                   * self.sigma_scale)
            took_any = False
            for k in range(mu_w.shape[0]):
                if pi[k] < self.pi_min:
                    continue
                anchors.append(anchor)
                mus.append(mu_w[k])
                sigmas.append(iso[k])
                gains.append(base * float(pi[k]))
                took_any = True
            if took_any:
                covered.add(int(slot))
        if anchors:
            self._anchor = np.asarray(anchors)
            self._mu = np.asarray(mus)
            self._sigma = np.asarray(sigmas)
            self._gain = np.asarray(gains)
        else:
            self._anchor = np.zeros((0, 2))

        # coverage union: detected tracks the model could NOT predict yet
        # (young/intermittent tracks) still get a reactive static disk --
        # a real DAA stack never leaves a live track unprotected, and the
        # arms must cover the same traffic set to compare fairly.
        det = np.asarray(ep_dict["obs_detected"][t0], dtype=bool)
        rest = [a for a in range(det.shape[0])
                if det[a] and a not in covered]
        pos = np.asarray([ep_dict["obs_pos"][t0, a] for a in rest],
                         dtype=np.float64) if rest else np.zeros((0, 2))
        self._static = self._const_rows(pos, self.sigma_base)
        self.using_fallback = self._anchor.shape[0] == 0

    def disks_batch(self, ts: np.ndarray) -> np.ndarray:
        static = np.broadcast_to(
            self._static, (len(ts),) + self._static.shape)
        if self._anchor.shape[0] == 0:
            return static
        m = self._anchor.shape[0]
        s = len(ts)
        t = np.clip(np.asarray(ts, dtype=np.float64), 0.0,
                    PRED_STEPS * LOG_DT)

        # timeline per disk: pos(0)=anchor, pos(i*0.5)=mu[i-1]
        # fractional index into that timeline:
        f = t / LOG_DT                                   # [S] in [0, 30]
        i0 = np.floor(f).astype(int)                     # lower node
        frac = (f - i0)[:, None]                         # [S,1]
        i0 = np.clip(i0, 0, PRED_STEPS)                  # nodes 0..30
        i1 = np.clip(i0 + 1, 0, PRED_STEPS)

        # node arrays with the anchor prepended: [M, 31, 2] / [M, 31]
        nodes = np.concatenate([self._anchor[:, None, :], self._mu], axis=1)
        signode = np.concatenate([np.full((m, 1), self.sigma_floor),
                                  self._sigma], axis=1)

        p0 = nodes[:, i0, :]                             # [M,S,2]
        p1 = nodes[:, i1, :]
        pos = p0 + (p1 - p0) * frac[None, :, :]          # [M,S,2]
        s0 = signode[:, i0]
        s1 = signode[:, i1]
        sig = s0 + (s1 - s0) * frac[None, :, 0]          # [M,S]

        out = np.empty((s, m, 5))
        out[:, :, 0] = pos[:, :, 0].T
        out[:, :, 1] = pos[:, :, 1].T
        out[:, :, 2] = self.hard_radius
        out[:, :, 3] = np.maximum(self.sigma_floor, sig.T)
        out[:, :, 4] = self._gain[None, :]
        return np.concatenate([static, out], axis=1)

    def describe(self) -> dict:
        return {"name": self.name,
                "n_disks": int(self._anchor.shape[0]
                               + self._static.shape[0]),
                "n_static": int(self._static.shape[0]),
                "fallback": self.using_fallback}

    def viz(self) -> dict:
        # the mixture itself is already on screen as the prediction fans;
        # only the UNPREDICTED tracks (static coverage) need extra ink
        return {"kind": self.name,
                "pts": [[round(float(r[0]), 1), round(float(r[1]), 1),
                         round(float(r[3]), 1)] for r in self._static]}


RISK_KINDS = {"none": NoPredictionRisk, "cv": CVRisk, "model": ModelRisk}


def make_risk(kind: str, **kw) -> BaseRisk:
    if kind not in RISK_KINDS:
        raise ValueError(f"unknown risk kind '{kind}' "
                         f"(choose from {sorted(RISK_KINDS)})")
    return RISK_KINDS[kind](**kw)
