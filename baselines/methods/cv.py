"""Constant-velocity baselines (doc 8.2 -- the mandatory floor).

Two point variants and one probabilistic wrapper:

  * CV-obs : velocity from the target's LAST TWO DETECTIONS. Sees exactly what
    the model sees (observation layer only) -- the honest baseline row.
  * CV-gt  : velocity from ground truth at t0. The unbeatable-information
    upper bound for any constant-velocity method; matches the dataGen 8.3
    difficulty diagnostic.
  * FittedSigma : per-horizon-step residual covariance fitted on TRAIN
    residuals, expressed in the CV heading frame (along/cross track) and
    attached to any point method. This is "global calibration": on average
    correct, per-situation blind -- precisely the contrast the paper draws
    against situation-dependent model sigma.
"""
from __future__ import annotations

import numpy as np

from common.constants import HIST_STEPS, PRED_STEPS
from data.dataset import window_detections

LOG_DT = 0.5


def cv_predict_obs(ep: dict, t0: int, target: int,
                   fit_window: int = 5, rule: str = "slot") -> np.ndarray | None:
    """CV extrapolation from the observed track. [PRED,2] world metres.

    Velocity comes from a least-squares line fit over the last up-to-
    ``fit_window`` detections (~2.5 s). Two-point differencing amplifies the
    position noise into ~10-20 m/s of velocity error and produced a 600 m
    residual sigma at 15 s -- a strawman no reviewer would accept as "the CV
    baseline". The LS fit is what any real tracker front-end would do and
    remains strictly constant-velocity.

    ``rule`` is the track rule of the neural feature builder
    (modelTraining/data/dataset.py): "current" fits only detections of the
    track OPEN at t0 -- a reacquired object's earlier, closed tracks are
    invisible, as the stated tracking model says; "slot" (the default, the
    pre-round-1 behaviour) uses every detection of the physical slot, i.e.
    oracle re-association. The harness always passes track_rule(data_cfg).
    Returns None when fewer than two usable detections exist.
    """
    h0 = t0 - HIST_STEPS + 1
    det = window_detections(ep, t0, rule)[:, target]
    hs = np.flatnonzero(det)
    if hs.size < 2:
        return None
    use = hs[-fit_window:]
    ts = use.astype(np.float64) * LOG_DT
    ps = ep["obs_pos"][h0 + use, target].astype(np.float64)
    t_mean = ts.mean()
    denom = float(((ts - t_mean) ** 2).sum())
    if denom < 1e-9:
        return None
    v = ((ts - t_mean)[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / denom
    p_at_last = ps.mean(axis=0) + v * (ts[-1] - t_mean)
    # extrapolate from the LAST OBSERVATION time, not t0: if the track went
    # stale, CV honestly carries that staleness forward
    dt_to_t0 = (HIST_STEPS - 1 - hs[-1]) * LOG_DT
    steps = np.arange(1, PRED_STEPS + 1, dtype=np.float64) * LOG_DT + dt_to_t0
    return p_at_last[None, :] + v[None, :] * steps[:, None]


def cv_predict_gt(ep: dict, t0: int, target: int) -> np.ndarray:
    """CV from ground-truth position/velocity at t0 (information upper bound)."""
    p = ep["gt_pos"][t0, target].astype(np.float64)
    v = ep["gt_vel"][t0, target].astype(np.float64)
    steps = np.arange(1, PRED_STEPS + 1, dtype=np.float64) * LOG_DT
    return p[None, :] + v[None, :] * steps[:, None]


class FittedSigma:
    """Per-step residual covariance in the along/cross-track frame.

    fit(): accumulate residuals (pred - gt) rotated into the prediction's
    heading frame; store std_along[h], std_cross[h].
    attach(): given point predictions, return (sigma[PRED,2], rho[PRED]) in
    WORLD axes for the metrics code (rho from the frame rotation).
    """

    def __init__(self) -> None:
        self.sa = np.full(PRED_STEPS, np.nan)
        self.sc = np.full(PRED_STEPS, np.nan)
        self._acc: list[tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------ fit
    def add_residuals(self, pred: np.ndarray, gt: np.ndarray,
                      valid: np.ndarray) -> None:
        """pred/gt [PRED,2] world; valid [PRED] bool."""
        v = pred[-1] - pred[0]
        n = np.hypot(v[0], v[1])
        c, s = (v[0] / n, v[1] / n) if n > 1e-6 else (1.0, 0.0)
        R = np.array([[c, s], [-s, c]])          # world -> along/cross
        r = (pred - gt) @ R.T
        r[~valid.astype(bool)] = np.nan
        self._acc.append((r[:, 0], r[:, 1]))

    def finalize(self, min_sigma: float = 1.0) -> None:
        a = np.stack([x for x, _ in self._acc])   # [N,PRED]
        c = np.stack([y for _, y in self._acc])
        self.sa = np.maximum(np.nanstd(a, axis=0), min_sigma)
        self.sc = np.maximum(np.nanstd(c, axis=0), min_sigma)
        self._acc.clear()

    # --------------------------------------------------------------- attach
    def attach(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (sigma_world[PRED,2], rho_world[PRED]) for this trajectory."""
        v = pred[-1] - pred[0]
        n = np.hypot(v[0], v[1])
        c, s = (v[0] / n, v[1] / n) if n > 1e-6 else (1.0, 0.0)
        # frame cov diag(sa^2, sc^2) rotated to world: R^T D R with R above
        sa2, sc2 = self.sa ** 2, self.sc ** 2
        cxx = c * c * sa2 + s * s * sc2
        cyy = s * s * sa2 + c * c * sc2
        cxy = c * s * (sa2 - sc2)
        sx = np.sqrt(cxx)
        sy = np.sqrt(cyy)
        rho = np.clip(cxy / (sx * sy + 1e-12), -0.99, 0.99)
        return np.stack([sx, sy], axis=1), rho

    def to_dict(self) -> dict:
        return {"sa": self.sa.tolist(), "sc": self.sc.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FittedSigma":
        f = cls()
        f.sa = np.asarray(d["sa"], dtype=np.float64)
        f.sc = np.asarray(d["sc"], dtype=np.float64)
        return f
