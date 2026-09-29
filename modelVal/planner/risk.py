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

Review round 1 additions (Reviewer 3; every default reproduces the runs made
before 2026-09-26 bit for bit):
  * track_rule ("slot" | "current", R3 major 3): which detections the CV
    velocity fit may use. "slot" = every detection of the physical agent
    slot across track drops (oracle re-association, the legacy behaviour);
    "current" = only detections of the track open at t0. The class default
    stays "slot" so callers that pass no rule (live console, replay_case)
    keep their old behaviour; experiments/batch.py always passes the rule.
  * disk_kinds() (R3 major 4): per-disk label aligned with the M axis of
    disks_batch -- KIND_STATIC (observed position, frozen over lookahead)
    or KIND_FORECAST (a disk that moves with lookahead time).
  * update_hard()/hard_disks_batch() (R3 major 4): ONE arm-independent hard
    geometry -- the observed positions at t0 of the intruders detected at
    t0, static over lookahead, with the shared hard_radius. Used only when
    PlannerConfig.hard_geom == "shared".
  * floor_stats()/branch_floor (R3 major 5): how often sigma_floor raises
    a forecast disk's sigma (lookahead nodes 0.5..15 s; the t=0 anchor node
    is excluded because it is set to the floor by construction).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

LOG_DT = 0.5          # s between logged observation rows (dataGen 2 Hz)
PRED_STEPS = 30       # model horizon steps (15 s)

KIND_STATIC = 0       # disk at an observed position, frozen over lookahead
KIND_FORECAST = 1     # disk that moves with lookahead time (CV / model)
TRACK_RULES = ("slot", "current")

#: lookahead of forecast node i (i = 1..30): i * LOG_DT seconds
NODE_T = np.arange(1, PRED_STEPS + 1, dtype=np.float64) * LOG_DT
#: node columns (0-based into a [.., 30] per-step array) of the 5/10/15 s
#: lookaheads reported by the branch-floor statistic
FLOOR_REPORT_STEPS = {5: 9, 10: 19, 15: 29}


def default_gain(hard_r: float, vehicle_radius: float) -> float:
    """The JS demo's obstacleSizeGain -- kept as the shared base gain."""
    return (hard_r + vehicle_radius) / 12.0


def iso_reduce(sig: np.ndarray, iso: str) -> np.ndarray:
    """(sigma_x, sigma_y) [..., 2] -> the planner's single isotropic scale."""
    if iso == "max":
        return np.maximum(sig[..., 0], sig[..., 1])
    if iso == "min":
        return np.minimum(sig[..., 0], sig[..., 1])
    return np.sqrt((sig[..., 0] ** 2 + sig[..., 1] ** 2) / 2.0)


def branch_floor_counts(preds: Optional[dict], iso: str, sigma_scale: float,
                        pi_min: float, floor: float) -> dict[int, list[int]]:
    """{5|10|15: [n_below, n_total]} over the RETAINED branches (pi >= pi_min)
    of every predicted track: how many isotropic branch sigmas (after --iso
    and --sigma-scale, i.e. exactly the value the floor is applied to) lie
    below the floor before flooring, at 5/10/15 s lookahead."""
    out = {h: [0, 0] for h in FLOOR_REPORT_STEPS}
    for p in (preds or {}).values():
        sig = np.asarray(p["sigma"], dtype=np.float64)        # [K,30,2]
        pi = np.asarray(p["pi"], dtype=np.float64)            # [K]
        keep = pi >= pi_min
        if not keep.any():
            continue
        red = iso_reduce(sig[keep], iso) * sigma_scale        # [k,30]
        for h, col in FLOOR_REPORT_STEPS.items():
            out[h][0] += int((red[:, col] < floor).sum())
            out[h][1] += int(red.shape[0])
    return out


class BaseRisk:
    """Shared plumbing: observation extraction + constant-disk fallback."""

    name = "base"

    def __init__(self, hard_radius: float = 30.0, vehicle_radius: float = 6.0,
                 sigma_base: float = 152.4, sigma_floor: float = 10.0,
                 track_rule: str = "slot") -> None:
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
        if str(track_rule) not in TRACK_RULES:
            raise ValueError(f"track_rule must be one of {TRACK_RULES}, "
                             f"got {track_rule!r}")
        self.track_rule = str(track_rule)
        self._disks0 = np.zeros((0, 5))         # current-position disks
        self._hard0: Optional[np.ndarray] = None   # shared hard geometry
        # {5|10|15: [n_below, n_total]} of the last update (predictor arms)
        self.branch_floor: Optional[dict[int, list[int]]] = None

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

    def disk_kinds(self) -> np.ndarray:
        """[M] int8 kind of every disk on the M axis of disks_batch."""
        return np.full(self._disks0.shape[0], KIND_STATIC, dtype=np.int8)

    # ---------------------------------------- shared hard geometry (R3 M4)
    def update_hard(self, ep_dict: dict, t0: int) -> None:
        """Refresh the arm-independent hard geometry: observed positions at
        t0 of the intruders detected at t0 (the 'none' arm's disk centres),
        with the shared hard_radius. sigma/gain are carried only to keep the
        5-column disk format; the hard check reads columns 0..2."""
        pos, _ = self._observed_now(ep_dict, t0)
        self._hard0 = self._const_rows(pos, self.sigma_base)

    def hard_disks_batch(self, ts: np.ndarray) -> np.ndarray:
        """[S, M_obs, 5] shared hard disks, static over lookahead."""
        if self._hard0 is None:
            raise RuntimeError("hard_disks_batch() before update_hard(): the "
                               "shared hard geometry was never prepared")
        return np.broadcast_to(self._hard0, (len(ts),) + self._hard0.shape)

    # ------------------------------------------------ floor stats (R3 M5)
    def floor_stats(self) -> tuple[int, int]:
        """(n_raised, n_total) over the FORECAST disks' pre-floor sigma at
        lookahead nodes 0.5..15 s. Static disks are not forecasts: (0, 0)."""
        return 0, 0

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
        self._slots: list[int] = []               # agent slot of each track

    def sigma_schedule(self, t: np.ndarray) -> np.ndarray:
        """CV's prescribed (pre-floor) sigma at lookahead t [s]."""
        t = np.asarray(t, dtype=np.float64)
        return np.sqrt(self.sigma0 ** 2 + (self.sigma_growth * t) ** 2)

    def update(self, ep_dict: dict, t0: int,
               preds: Optional[dict] = None) -> None:
        det = np.asarray(ep_dict["obs_detected"], dtype=bool)
        obs = np.asarray(ep_dict["obs_pos"], dtype=np.float64)
        # "current": only rows of the track open at t0 (det[t0, a] implies
        # obs_track_id[t0, a] >= 0). A reacquired object gets a NEW id from
        # the tracker, so its closed earlier track is invisible here, exactly
        # as it is to a current-track predictor. "slot" keeps the legacy
        # expression untouched (bit-identical to the pre-review runs).
        tid = (np.asarray(ep_dict["obs_track_id"])
               if self.track_rule == "current" else None)
        j0 = max(0, t0 - self.window_steps + 1)
        pos_list, vel_list, slots = [], [], []
        for a in range(det.shape[1]):
            if not det[t0, a]:
                continue
            if tid is None:
                idx = [j for j in range(j0, t0 + 1) if det[j, a]]
            else:
                cur = tid[t0, a]
                idx = [j for j in range(j0, t0 + 1)
                       if det[j, a] and tid[j, a] == cur]
            slots.append(a)
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
        self._slots = slots
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

    def disk_kinds(self) -> np.ndarray:
        # every CV disk is an extrapolation (moves with t), even a track
        # whose fitted velocity happens to be zero
        return np.full(self._pos.shape[0], KIND_FORECAST, dtype=np.int8)

    def floor_stats(self) -> tuple[int, int]:
        m = self._pos.shape[0]
        raised = int((self.sigma_schedule(NODE_T) < self.sigma_floor).sum())
        return m * raised, m * PRED_STEPS

    def forecast_log(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """[(slot, mu [30,2] world m, sigma [30] isotropic, pre-floor)] of
        every CV track: the arm's forecast as an isotropic Gaussian."""
        sched = self.sigma_schedule(NODE_T)
        return [(int(a), p[None, :] + v[None, :] * NODE_T[:, None], sched)
                for a, p, v in zip(self._slots, self._pos, self._vel)]

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
                 iso: str = "rms", gain_mode: str = "pi", **kw) -> None:
        super().__init__(**kw)
        self.pi_min = float(pi_min)
        # --- rebuttal knobs (reviewer 2, comment VI). Defaults reproduce
        # every batch run before 2026-09-07 exactly; only a non-default
        # value changes behaviour.
        #
        # iso: how the per-axis (sigma_x, sigma_y) pair is reduced to the
        #   ONE isotropic scale the planner's disk format carries.
        #     'rms' -- sqrt((sx^2+sy^2)/2) == sqrt(trace/2). Rotation
        #              invariant, but it UNDER-states the major axis by up
        #              to sqrt(2) when the ellipse is elongated.
        #     'max' -- max(sx, sy). The smallest disk that contains the
        #              1-sigma ellipse, i.e. the conservative bound 'rms'
        #              is not guaranteed to provide.
        #     'min' -- min(sx, sy). The optimistic opposite; included so
        #              the sensitivity grid brackets 'rms' from both sides
        #              rather than only from above.
        if str(iso) not in ("rms", "max", "min"):
            raise ValueError(f"iso must be rms|max|min, got {iso!r}")
        self.iso = str(iso)
        # gain_mode: how the shared base gain is split across branches.
        #     'pi'      -- gain_k = base * pi_k, so the branch-weighted peak
        #                  equals a single-hypothesis obstacle's peak. This
        #                  is a probability-weighted risk score.
        #     'uniform' -- gain_k = base for EVERY retained branch, i.e. the
        #                  planner treats each surviving mode as if it were
        #                  certain (a worst-case / risk-averse aggregation).
        #                  Tests whether pi-weighting dilutes the value of a
        #                  multimodal forecast in the closed loop.
        if str(gain_mode) not in ("pi", "uniform"):
            raise ValueError(f"gain_mode must be pi|uniform, got {gain_mode!r}")
        self.gain_mode = str(gain_mode)
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
            if self.iso == "max":
                red = np.maximum(sig[:, :, 0], sig[:, :, 1])
            elif self.iso == "min":
                red = np.minimum(sig[:, :, 0], sig[:, :, 1])
            else:
                red = np.sqrt((sig[:, :, 0] ** 2 + sig[:, :, 1] ** 2) / 2.0)
            iso = red * self.sigma_scale
            took_any = False
            for k in range(mu_w.shape[0]):
                if pi[k] < self.pi_min:
                    continue
                anchors.append(anchor)
                mus.append(mu_w[k])
                sigmas.append(iso[k])
                gains.append(base if self.gain_mode == "uniform"
                             else base * float(pi[k]))
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
        # R3 major 5: how often the floor would raise the NETWORK's branch
        # sigma (computed before any decomposition arm overrides _sigma)
        self.branch_floor = branch_floor_counts(
            preds, self.iso, self.sigma_scale, self.pi_min, self.sigma_floor)

    def disk_kinds(self) -> np.ndarray:
        # disks_batch order: static fallback rows first, then branch disks
        n_static = self._static.shape[0]
        n_fc = self._anchor.shape[0]
        return np.concatenate([np.full(n_static, KIND_STATIC, dtype=np.int8),
                               np.full(n_fc, KIND_FORECAST, dtype=np.int8)])

    def floor_stats(self) -> tuple[int, int]:
        # the pre-floor sigma at node i*0.5 s IS self._sigma[:, i-1] (the
        # interpolation weight is exactly 0 on a node); node 0 (anchor) is
        # excluded -- it is set to the floor by construction
        if self._anchor.shape[0] == 0:
            return 0, 0
        return int((self._sigma < self.sigma_floor).sum()), int(self._sigma.size)

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


class ModelCenterCVSigmaRisk(ModelRisk):
    """Decomposition arm: PRISM branch CENTRES, CV's prescribed sigma.

    Reviewer 1 comment 3 / reviewer 2 comment V ask why the learned mixture
    does not beat CV in the closed loop, given that it clearly beats CV on
    the prediction metrics. The model arm changes TWO things at once
    relative to CV -- where the risk sits over lookahead time, and how wide
    it grows -- so the batch cannot attribute the outcome to either. This
    arm keeps the learned multi-branch centres mu_k(t) and the pi-weighted
    gains but replaces the predicted sigma_k(t) with CV's prescribed
    schedule sqrt(sigma0^2 + (growth*t)^2), isolating the CENTRE.
    """

    name = "mc_cvsig"

    def __init__(self, sigma_growth: float = 4.0, sigma0: float = 15.0,
                 **kw) -> None:
        super().__init__(**kw)
        self.sigma_growth = float(sigma_growth)
        self.sigma0 = float(sigma0)

    def update(self, ep_dict: dict, t0: int,
               preds: Optional[dict] = None) -> None:
        super().update(ep_dict, t0, preds)
        # overwrite the per-node sigma timeline with the CV schedule; node i
        # of self._mu is lookahead (i+1)*LOG_DT seconds (node 0 == anchor is
        # prepended with sigma_floor inside disks_batch).
        if self._sigma.shape[0]:
            t = (np.arange(1, PRED_STEPS + 1, dtype=np.float64) * LOG_DT)
            sched = np.sqrt(self.sigma0 ** 2 + (self.sigma_growth * t) ** 2)
            self._sigma = np.broadcast_to(
                sched, self._sigma.shape).copy()


class CVCenterModelSigmaRisk(CVRisk):
    """Decomposition arm: CV centres, PRISM's predicted sigma.

    The complement of :class:`ModelCenterCVSigmaRisk`: constant-velocity
    extrapolation decides WHERE the risk sits, while the learned predictor
    decides HOW WIDE it is. A track the predictor could not serve keeps
    CV's own schedule, so the traffic set covered is identical to the CV
    arm's. Together the two arms decompose the model arm's closed-loop
    effect into a centre term and an uncertainty term.
    """

    name = "cvc_msig"

    def __init__(self, pi_min: float = 0.05, sigma_scale: float = 1.0,
                 iso: str = "rms", **kw) -> None:
        super().__init__(**kw)
        self.pi_min = float(pi_min)
        self.sigma_scale = float(sigma_scale)
        if str(iso) not in ("rms", "max", "min"):
            raise ValueError(f"iso must be rms|max|min, got {iso!r}")
        self.iso = str(iso)
        self._msig = np.zeros((0, PRED_STEPS))   # [M, 30] per CV track
        self._has_model = np.zeros((0,), dtype=bool)

    def update(self, ep_dict: dict, t0: int,
               preds: Optional[dict] = None) -> None:
        # CVRisk.update fills self._pos / self._vel over the DETECTED slots,
        # in ascending slot order; mirror that ordering here.
        super().update(ep_dict, t0, preds)
        preds = preds or {}
        det = np.asarray(ep_dict["obs_detected"][t0], dtype=bool)
        slots = [a for a in range(det.shape[0]) if det[a]]
        rows, has = [], []
        for a in slots:
            p = preds.get(a)
            if p is None:
                rows.append(np.zeros(PRED_STEPS)); has.append(False)
                continue
            sig = np.asarray(p["sigma"], dtype=np.float64)   # [K,30,2]
            pi = np.asarray(p["pi"], dtype=np.float64)       # [K]
            if self.iso == "max":
                red = np.maximum(sig[:, :, 0], sig[:, :, 1])
            elif self.iso == "min":
                red = np.minimum(sig[:, :, 0], sig[:, :, 1])
            else:
                red = np.sqrt((sig[:, :, 0] ** 2 + sig[:, :, 1] ** 2) / 2.0)
            keep = pi >= self.pi_min
            if not keep.any():
                rows.append(np.zeros(PRED_STEPS)); has.append(False)
                continue
            w = pi[keep] / pi[keep].sum()
            rows.append((w[:, None] * red[keep]).sum(axis=0)
                        * self.sigma_scale)
            has.append(True)
        self._msig = (np.asarray(rows) if rows
                      else np.zeros((0, PRED_STEPS)))
        self._has_model = (np.asarray(has, dtype=bool) if has
                           else np.zeros((0,), dtype=bool))
        self.branch_floor = branch_floor_counts(
            preds, self.iso, self.sigma_scale, self.pi_min, self.sigma_floor)

    def floor_stats(self) -> tuple[int, int]:
        m = self._pos.shape[0]
        if m == 0:
            return 0, 0
        sched = self.sigma_schedule(NODE_T)                   # [30]
        pre = np.where(self._has_model[:, None], self._msig, sched[None, :])
        return int((pre < self.sigma_floor).sum()), int(pre.size)

    def disks_batch(self, ts: np.ndarray) -> np.ndarray:
        out = super().disks_batch(ts)          # CV centres + CV sigma
        m = self._pos.shape[0]
        if m == 0 or not self._has_model.any():
            return out
        t = np.clip(np.asarray(ts, dtype=np.float64), 0.0,
                    PRED_STEPS * LOG_DT)
        f = t / LOG_DT
        i0 = np.clip(np.floor(f).astype(int), 0, PRED_STEPS)
        i1 = np.clip(i0 + 1, 0, PRED_STEPS)
        frac = (f - np.floor(f))
        node = np.concatenate(
            [np.full((m, 1), self.sigma_floor), self._msig], axis=1)
        sig = (node[:, i0] + (node[:, i1] - node[:, i0]) * frac[None, :]).T
        swap = self._has_model[None, :]
        out[:, :, 3] = np.where(
            swap, np.maximum(self.sigma_floor, sig), out[:, :, 3])
        return out


RISK_KINDS = {"none": NoPredictionRisk, "cv": CVRisk, "model": ModelRisk,
              "mc_cvsig": ModelCenterCVSigmaRisk,
              "cvc_msig": CVCenterModelSigmaRisk}


def make_risk(kind: str, **kw) -> BaseRisk:
    if kind not in RISK_KINDS:
        raise ValueError(f"unknown risk kind '{kind}' "
                         f"(choose from {sorted(RISK_KINDS)})")
    return RISK_KINDS[kind](**kw)
