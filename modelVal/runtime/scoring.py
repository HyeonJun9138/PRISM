"""Rolling live accuracy: score past predictions the moment reality catches up.

Every prediction made at step t0 is held in a ring; when the simulation
reaches t0+10 / t0+20 / t0+30 logged steps (5/10/15 s), the realised GT
position is compared against what the model said back then:

  * err  -- displacement error [m] of the BEST branch (min over K)
  * cov90 -- was the realised point inside the mixture's 90% region?
             (Monte-Carlo HPD with 32 samples, same estimator family as
             modelTraining.model.metrics, cheap enough per frame)

Aggregates are rolling over the last ~400 scored points per horizon, so the
bottom panel reacts to what the model is doing NOW, not the session average.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

HORIZON_STEPS = (10, 20, 30)          # 5 s / 10 s / 15 s at LOG_DT = 0.5
MC_SAMPLES = 32
CHI_WINDOW = 400


class RollingScore:
    def __init__(self) -> None:
        self.pending: deque = deque(maxlen=200)   # (t0, {agent: pred})
        self.err = {h: deque(maxlen=CHI_WINDOW) for h in HORIZON_STEPS}
        self.cov = {h: deque(maxlen=CHI_WINDOW) for h in HORIZON_STEPS}
        self.n_scored = 0

    def clear_pending(self) -> None:
        """Scenario rotated: old predictions reference dead indices."""
        self.pending.clear()

    def add_predictions(self, t0: int, preds: dict[int, dict[str, Any]]) -> None:
        if preds:
            self.pending.append((int(t0), preds))

    # ------------------------------------------------------------------ tick
    def settle(self, ep_dict: dict, now_step: int) -> None:
        """Score every pending prediction whose horizons have been reached."""
        for t0, preds in list(self.pending):
            done = True
            for h in HORIZON_STEPS:
                tk = t0 + h
                if tk > now_step:
                    done = False
                    continue
                if tk == now_step:            # score exactly once, on arrival
                    self._score_horizon(ep_dict, t0, h, preds)
            if done and t0 + HORIZON_STEPS[-1] <= now_step:
                try:
                    self.pending.remove((t0, preds))
                except ValueError:
                    pass

    def _score_horizon(self, ep_dict: dict, t0: int, h: int,
                       preds: dict[int, dict[str, Any]]) -> None:
        step_idx = h - 1                      # prediction array is 0-based
        tk = t0 + h
        rng = np.random.default_rng(tk)
        for a, p in preds.items():
            if a >= ep_dict["gt_valid"].shape[1] or not ep_dict["gt_valid"][tk, a]:
                continue
            gt = ep_dict["gt_pos"][tk, a].astype(np.float64)
            if not np.isfinite(gt).all():
                continue
            mu = p["mu_w"][:, step_idx]        # [K,2] world
            err = float(np.min(np.hypot(mu[:, 0] - gt[0], mu[:, 1] - gt[1])))
            self.err[h].append(err)
            self.cov[h].append(1.0 if self._inside_90(p, step_idx, gt) else 0.0)
            self.n_scored += 1

    # ------------------------------------------------------------- coverage
    @staticmethod
    def _inside_90(p: dict[str, Any], step_idx: int, gt_world: np.ndarray) -> bool:
        """MC-HPD membership test on the mixture at one step (world axes).

        sigma/rho live in the target frame; rotating a distribution and its
        sample set by the same R changes nothing about HPD membership, so the
        test runs in the frame: transform gt into the frame instead.
        """
        mu_w = p["mu_w"][:, step_idx].astype(np.float64)     # [K,2]
        sig = p["sigma"][:, step_idx].astype(np.float64)     # [K,2]
        rho = p["rho"][:, step_idx].astype(np.float64)       # [K]
        pi = p["pi"].astype(np.float64)
        K = mu_w.shape[0]
        theta = float(p["theta"])
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, s], [-s, c]])                      # world -> frame rot

        def logpdf(pts_w: np.ndarray) -> np.ndarray:
            d = (pts_w[:, None, :] - mu_w[None, :, :]) @ R.T  # [S,K,2] frame
            zx = d[..., 0] / sig[None, :, 0]
            zy = d[..., 1] / sig[None, :, 1]
            omr2 = np.clip(1.0 - rho ** 2, 1e-6, None)[None, :]
            q = (zx ** 2 - 2 * rho[None, :] * zx * zy + zy ** 2) / omr2
            log_n = (-0.5 * q - np.log(sig[None, :, 0]) - np.log(sig[None, :, 1])
                     - 0.5 * np.log(omr2) - math.log(2 * math.pi))
            m = log_n.max(axis=1, keepdims=True)
            return (m[:, 0] + np.log((pi[None, :] * np.exp(log_n - m)).sum(axis=1)
                                     + 1e-300))

        rng = np.random.default_rng(int(abs(gt_world[0] * 7 + gt_world[1])) & 0xFFFF)
        comp = rng.choice(K, size=MC_SAMPLES, p=pi / pi.sum())
        e1 = rng.standard_normal(MC_SAMPLES)
        e2 = rng.standard_normal(MC_SAMPLES)
        fx = sig[comp, 0] * e1
        fy = sig[comp, 1] * (rho[comp] * e1 + np.sqrt(np.clip(1 - rho[comp] ** 2,
                                                              1e-6, None)) * e2)
        Rinv = R.T
        samples_w = mu_w[comp] + np.stack([fx, fy], axis=1) @ Rinv.T
        thresh = np.quantile(logpdf(samples_w), 0.10)
        return bool(logpdf(gt_world[None, :])[0] >= thresh)

    # ------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n_scored": self.n_scored}
        for h in HORIZON_STEPS:
            sec = h // 2
            e = np.asarray(self.err[h], dtype=np.float64)
            cv = np.asarray(self.cov[h], dtype=np.float64)
            out[f"err_{sec}s"] = round(float(np.median(e)), 1) if e.size else None
            out[f"cov90_{sec}s"] = round(float(cv.mean()), 3) if cv.size else None
            out[f"n_{sec}s"] = int(e.size)
        return out
