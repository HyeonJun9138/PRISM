"""IMM baseline: the classical tracking answer to this problem (doc 8.2).

Interacting Multiple Model filter over the OBSERVATION sequence, with a bank
that mirrors the simulator's true manoeuvre set (doc 3.7 modes), which is the
most favourable honest configuration a reviewer could ask for:

  * CV : white-accel constant velocity     state [x y vx vy]
  * CT : coordinated turn, EKF, turn rate  state [x y vx vy w]
         ESTIMATED in the state
  * CA : white-jerk constant acceleration  state [x y vx vy ax ay]

Measurements are the logged 2 Hz observed positions with their true per-point
covariance R (obs_R); missed detections are coast steps (time update only).
Prediction at t0: each model's state is propagated open-loop over the 15 s
horizon and the result is a 3-component GAUSSIAN MIXTURE weighted by the IMM
model probabilities -- directly comparable to the K=3 mixture the paper's
model emits, including NLL and coverage.

No learning. Tunable process noises are fitted by grid search on the valid
split (main.py fit-imm) and frozen into configs/imm.json.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from common.constants import HIST_STEPS, PRED_STEPS

LOG_DT = 0.5
N_MODELS = 3            # CV, CT, CA


@dataclass
class IMMParams:
    q_cv: float = 0.6        # white accel PSD [m^2/s^3] for CV
    q_ct: float = 0.8        # accel PSD for CT position/velocity channels
    q_w: float = 0.01        # turn-rate random walk PSD [rad^2/s^3]
    q_ca: float = 0.4        # white jerk PSD for CA
    p_stay: float = 0.95     # TPM diagonal
    r_infl: float = 1.0      # measurement covariance inflation

    def tpm(self) -> np.ndarray:
        off = (1.0 - self.p_stay) / (N_MODELS - 1)
        m = np.full((N_MODELS, N_MODELS), off)
        np.fill_diagonal(m, self.p_stay)
        return m

    def to_dict(self) -> dict:
        return {k: float(getattr(self, k)) for k in
                ("q_cv", "q_ct", "q_w", "q_ca", "p_stay", "r_infl")}

    @classmethod
    def from_dict(cls, d: dict) -> "IMMParams":
        return cls(**{k: float(v) for k, v in d.items()})


# ------------------------------------------------------------ model dynamics

def _f_cv(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F @ x, F


def _q_cv(q: float, dt: float) -> np.ndarray:
    d3, d2 = dt ** 3 / 3.0, dt ** 2 / 2.0
    Q1 = q * np.array([[d3, d2], [d2, dt]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return Q


def _f_ct(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordinated turn with estimated w. Returns (x_next, Jacobian)."""
    px, py, vx, vy, w = x
    if abs(w) < 1e-6:
        xn = np.array([px + vx * dt, py + vy * dt, vx, vy, w])
        J = np.eye(5)
        J[0, 2] = J[1, 3] = dt
        # d(pos)/dw and d(vel)/dw limits as w->0
        J[0, 4] = -0.5 * vy * dt * dt
        J[1, 4] = 0.5 * vx * dt * dt
        J[2, 4] = -vy * dt
        J[3, 4] = vx * dt
        return xn, J
    s, c = math.sin(w * dt), math.cos(w * dt)
    xn = np.array([
        px + (vx * s - vy * (1 - c)) / w,
        py + (vx * (1 - c) + vy * s) / w,
        vx * c - vy * s,
        vx * s + vy * c,
        w,
    ])
    J = np.eye(5)
    J[0, 2] = s / w
    J[0, 3] = -(1 - c) / w
    J[1, 2] = (1 - c) / w
    J[1, 3] = s / w
    J[2, 2] = c; J[2, 3] = -s
    J[3, 2] = s; J[3, 3] = c
    J[0, 4] = (vx * (w * dt * c - s) + vy * (w * dt * s - (1 - c))) / w ** 2
    J[1, 4] = (vx * (w * dt * s - (1 - c)) - vy * (w * dt * c - s)) / w ** 2
    J[2, 4] = dt * (-vx * s - vy * c)
    J[3, 4] = dt * (vx * c - vy * s)
    return xn, J


def _q_ct(q: float, qw: float, dt: float) -> np.ndarray:
    Q = np.zeros((5, 5))
    Q[:4, :4] = _q_cv(q, dt)
    Q[4, 4] = qw * dt
    return Q


def _f_ca(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(6)
    F[0, 2] = F[1, 3] = dt
    F[2, 4] = F[3, 5] = dt
    F[0, 4] = F[1, 5] = 0.5 * dt * dt
    return F @ x, F


def _q_ca(q: float, dt: float) -> np.ndarray:
    d5, d4, d3, d2 = dt**5/20, dt**4/8, dt**3/6, dt**2/2
    Q1 = q * np.array([[d5, d4, d3], [d4, dt**3/3, d2], [d3, d2, dt]])
    Q = np.zeros((6, 6))
    Q[np.ix_([0, 2, 4], [0, 2, 4])] = Q1
    Q[np.ix_([1, 3, 5], [1, 3, 5])] = Q1
    return Q


_DIMS = (4, 5, 6)
_POS = ((0, 1), (0, 1), (0, 1))         # position indices per model


# ---------------------------------------------------------------- the filter

@dataclass
class _ModelState:
    x: np.ndarray
    P: np.ndarray


class IMMTracker:
    """One target's IMM over its logged observation row.

    ``init_gap_fix`` (default False = pre-round-1 behaviour): the two-point
    velocity initialisation divides by the time between the first two
    detections. The legacy code does not advance that time over the misses
    in between (step(None) returns early while initialisation is pending), so
    a track whose first two detections are k misses apart starts with (k+1)x
    its velocity. Found in review round 1 (2026-09-26); on 30 test_id
    episodes the fix moved IMM minADE 101.9 -> 74.6 m under rule "slot". Kept
    opt-in so the published IMM rows reproduce; see baselines/main.py
    --imm-init-gap-fix.
    """

    def __init__(self, params: IMMParams, init_gap_fix: bool = False):
        self.p = params
        self.mu = np.full(N_MODELS, 1.0 / N_MODELS)
        self.models: list[_ModelState] | None = None
        self._last_z: np.ndarray | None = None
        self._init_pending = True
        self.init_gap_fix = bool(init_gap_fix)

    # ------------------------------------------------------- initialisation
    def _init_from(self, z0: np.ndarray, z1: np.ndarray, gap_s: float,
                   R: np.ndarray) -> None:
        v = (z1 - z0) / max(gap_s, 1e-6)
        big_v = 100.0
        states = []
        for dim in _DIMS:
            x = np.zeros(dim)
            x[0:2] = z1
            x[2:4] = v
            P = np.eye(dim) * 1.0
            P[0:2, 0:2] = R * 2.0
            P[2, 2] = P[3, 3] = big_v
            if dim >= 5:
                P[4, 4] = 0.09           # w ~ +-0.3 rad/s prior (CT)
            if dim == 6:
                P[4, 4] = P[5, 5] = 4.0  # accel prior (CA)
            states.append(_ModelState(x, P))
        self.models = states
        self._init_pending = False

    # ------------------------------------------------------------ transitions
    def _propagate(self, m: _ModelState, idx: int, dt: float) -> _ModelState:
        p = self.p
        if idx == 0:
            xn, F = _f_cv(m.x, dt)
            Q = _q_cv(p.q_cv, dt)
        elif idx == 1:
            xn, F = _f_ct(m.x, dt)
            Q = _q_ct(p.q_ct, p.q_w, dt)
        else:
            xn, F = _f_ca(m.x, dt)
            Q = _q_ca(p.q_ca, dt)
        return _ModelState(xn, F @ m.P @ F.T + Q)

    def _mix(self) -> list[_ModelState]:
        """Standard IMM mixing to common-dimension inputs per model."""
        tpm = self.p.tpm()
        c_j = tpm.T @ self.mu                       # predicted model probs
        c_j = np.maximum(c_j, 1e-12)
        mixed = []
        for j in range(N_MODELS):
            w = tpm[:, j] * self.mu / c_j[j]        # mixing weights
            dim = _DIMS[j]
            x = np.zeros(dim)
            parts = []
            for i in range(N_MODELS):
                xi = self._cast(self.models[i], dim)
                parts.append((w[i], xi))
                x += w[i] * xi.x
            P = np.zeros((dim, dim))
            for w_i, xi in parts:
                d = (xi.x - x)[:, None]
                P += w_i * (xi.P + d @ d.T)
            mixed.append(_ModelState(x, P))
        self._c_pred = c_j
        return mixed

    @staticmethod
    def _cast(m: _ModelState, dim: int) -> _ModelState:
        """Project a model state into another model's dimension.

        ONLY the kinematic prefix [x y vx vy] has shared semantics. The 5th+
        states mean different things per model (CT: turn rate, CA: accel);
        copying them across banks injects an acceleration into the turn-rate
        slot and destroys the CT filter -- they are re-initialised to their
        model's prior instead.
        """
        d0 = m.x.shape[0]
        if d0 == dim:
            return _ModelState(m.x.copy(), m.P.copy())
        x = np.zeros(dim)
        n = min(d0, dim, 4)
        x[:n] = m.x[:n]
        P = np.eye(dim) * 1.0
        P[:n, :n] = m.P[:n, :n]
        for k in range(n, dim):            # model-specific extras: fresh prior
            P[k, k] = 0.09 if dim == 5 else 4.0
        return _ModelState(x, P)

    # ------------------------------------------------------------------ step
    def step(self, z: np.ndarray | None, R: np.ndarray | None,
             dt: float = LOG_DT) -> None:
        """One logged step: z=None means a missed detection (coast)."""
        if self._init_pending:
            if z is None:
                if self.init_gap_fix and self._last_z is not None:
                    self._gap += dt          # time since the first detection grows
                return
            if self._last_z is None:
                self._last_z = z.copy()
                self._gap = dt
                return
            self._init_from(self._last_z, z, self._gap, R)
            return
        if self.models is None:
            return

        mixed = self._mix()
        liks = np.ones(N_MODELS) * 1e-12
        for j in range(N_MODELS):
            pred = self._propagate(mixed[j], j, dt)
            if z is not None:
                H = np.zeros((2, _DIMS[j]))
                H[0, 0] = H[1, 1] = 1.0
                Rm = R * self.p.r_infl
                S = H @ pred.P @ H.T + Rm
                Sinv = np.linalg.inv(S)
                innov = z - pred.x[0:2]
                K = pred.P @ H.T @ Sinv
                x = pred.x + K @ innov
                P = (np.eye(_DIMS[j]) - K @ H) @ pred.P
                mahal = float(innov @ Sinv @ innov)
                liks[j] = math.exp(-0.5 * mahal) / \
                    (2 * math.pi * math.sqrt(max(np.linalg.det(S), 1e-12)))
                self.models[j] = _ModelState(x, P)
            else:
                self.models[j] = pred
        if z is not None:
            mu = self._c_pred * np.maximum(liks, 1e-300)
            self.mu = mu / mu.sum()
        else:
            self.mu = self._c_pred
        if z is not None:
            self._last_z = z.copy()
        else:
            self._gap = getattr(self, "_gap", dt) + dt

    @property
    def ready(self) -> bool:
        return self.models is not None

    # ------------------------------------------------------------- predict
    def predict(self, horizon_steps: int = PRED_STEPS
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Open-loop forecast: (mu_pos [K,H,2], sigma [K,H,2], rho [K,H],
        pi [K]) as a Gaussian mixture over the model bank."""
        K, H = N_MODELS, horizon_steps
        mu_pos = np.zeros((K, H, 2))
        sig = np.zeros((K, H, 2))
        rho = np.zeros((K, H))
        for j in range(K):
            m = _ModelState(self.models[j].x.copy(), self.models[j].P.copy())
            for h in range(H):
                m = self._propagate(m, j, LOG_DT)
                mu_pos[j, h] = m.x[0:2]
                Pp = m.P[0:2, 0:2]
                sx = math.sqrt(max(Pp[0, 0], 1e-6))
                sy = math.sqrt(max(Pp[1, 1], 1e-6))
                sig[j, h] = (sx, sy)
                rho[j, h] = float(np.clip(Pp[0, 1] / (sx * sy), -0.99, 0.99))
        return mu_pos, sig, rho, self.mu.copy()


# --------------------------------------------------------------- per-episode

def imm_predictions_for_episode(ep: dict, t0_list: dict[int, list[int]],
                                params: IMMParams, rule: str = "slot",
                                init_gap_fix: bool = False
                                ) -> dict[tuple[int, int], tuple]:
    """Filter each requested agent ONCE over its row, snapshotting forecasts
    at every requested t0 (windows overlap heavily; refiltering per sample
    would cost 20x).

    Args:
        ep: episode arrays dict (needs obs_track_id under rule "current").
        t0_list: {agent_slot: sorted [t0, ...]} anchors to forecast at.
        params: tuned IMM noise set.
        rule: track rule (modelTraining/data/dataset.py module doc).
            "slot" (default, pre-round-1 behaviour): ONE filter over the whole
            physical slot row, coasting through track drops -- oracle
            re-association. "current": a detection whose track id differs
            from the running filter's starts a FRESH filter (a reacquired
            object is a new track: new id, no memory of the old one), and a
            forecast at t0 exists only if the filter belongs to the track
            open at t0 (obs_track_id[t0] >= 0 and equal to the filter's id).
        init_gap_fix: IMMTracker option (class doc); False = legacy.

    Returns:
        {(agent, t0): (mu[K,H,2], sigma, rho, pi)}; an anchor without a
        ready filter (or, under "current", without an open track) is absent.
    """
    if rule not in ("slot", "current"):
        raise ValueError(f"unknown track rule {rule!r}")
    out: dict[tuple[int, int], tuple] = {}
    det_all = ep["obs_detected"].astype(bool)
    if rule == "slot":
        for a, anchors in t0_list.items():
            tracker = IMMTracker(params, init_gap_fix)
            want = set(int(t) for t in anchors)
            t_max = max(want)
            for t in range(0, t_max + 1):
                if det_all[t, a]:
                    z = ep["obs_pos"][t, a].astype(np.float64)
                    R = ep["obs_R"][t, a].astype(np.float64)
                    tracker.step(z, R)
                else:
                    tracker.step(None, None)
                if t in want and tracker.ready:
                    out[(a, t)] = tracker.predict()
        return out

    if "obs_track_id" not in ep:
        raise KeyError("rule 'current' needs obs_track_id in the episode arrays")
    tid_all = np.asarray(ep["obs_track_id"])
    for a, anchors in t0_list.items():
        tracker: IMMTracker | None = None
        trk_id = None
        want = set(int(t) for t in anchors)
        t_max = max(want)
        for t in range(0, t_max + 1):
            if det_all[t, a]:
                tid = int(tid_all[t, a])
                if tracker is None or tid != trk_id:
                    tracker, trk_id = IMMTracker(params, init_gap_fix), tid  # new track
                z = ep["obs_pos"][t, a].astype(np.float64)
                R = ep["obs_R"][t, a].astype(np.float64)
                tracker.step(z, R)
            elif tracker is not None:
                tracker.step(None, None)
            if t in want and tracker is not None and tracker.ready:
                open_id = int(tid_all[t, a])
                if open_id >= 0 and open_id == trk_id:
                    out[(a, t)] = tracker.predict()
    return out
