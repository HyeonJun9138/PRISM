"""Score any baseline with modelTraining's OWN metric code.

Every method here emits the same `out` dict the neural model does
(mu / log_sigma / rho / mode_logits in POS_SCALE units), so
model.metrics.batch_metrics computes minADE / minFDE / NLL / MC-HPD coverage
IDENTICALLY for every row of the paper's comparison table. Point methods get
sigma from a FittedSigma wrapper (fitted on train residuals), which makes
their NLL/coverage well-defined -- and demonstrably situation-blind.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from common.constants import N_TYPES, POS_SCALE, PRED_STEPS
from model.metrics import MetricAccumulator, batch_metrics

from harness.data import future_of, iter_eval_groups
from methods.cv import FittedSigma, cv_predict_gt, cv_predict_obs
from methods.imm import IMMParams, imm_predictions_for_episode

BATCH = 512


class _Collector:
    """Accumulate per-sample mixtures, flush through batch_metrics."""

    def __init__(self):
        self.acc = MetricAccumulator()
        self.buf: list[dict] = []
        self.wall_ms: list[float] = []

    def add(self, mu_kh2: np.ndarray, sig_kh2: np.ndarray, rho_kh: np.ndarray,
            pi_k: np.ndarray, fut: np.ndarray, val: np.ndarray,
            gt_type: int, intent: int) -> None:
        self.buf.append(dict(mu=mu_kh2, sig=sig_kh2, rho=rho_kh, pi=pi_k,
                             fut=fut, val=val, ty=gt_type, it=intent))
        if len(self.buf) >= BATCH:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        B = len(self.buf)
        K = self.buf[0]["mu"].shape[0]
        mu = np.stack([b["mu"] for b in self.buf]) / POS_SCALE
        sig = np.stack([b["sig"] for b in self.buf]) / POS_SCALE
        rho = np.stack([b["rho"] for b in self.buf])
        pi = np.stack([b["pi"] for b in self.buf])
        out = {
            "mu": torch.as_tensor(mu, dtype=torch.float32),
            "log_sigma": torch.as_tensor(np.log(np.clip(sig, 1e-4, None)),
                                         dtype=torch.float32),
            "rho": torch.as_tensor(rho, dtype=torch.float32),
            "mode_logits": torch.as_tensor(np.log(np.clip(pi, 1e-9, None)),
                                           dtype=torch.float32),
            "type_logits": torch.zeros(B, N_TYPES),
        }
        batch = {
            "y": torch.as_tensor(
                np.stack([b["fut"] for b in self.buf]) / POS_SCALE,
                dtype=torch.float32),
            "y_valid": torch.as_tensor(np.stack([b["val"] for b in self.buf]),
                                       dtype=torch.float32),
            "type_label": torch.as_tensor([b["ty"] for b in self.buf]),
            "meta_intent": torch.as_tensor([b["it"] for b in self.buf]),
        }
        self.acc.update(batch_metrics(out, batch, mc_samples=128))
        self.buf.clear()

    def report(self, name: str, runs: list[str], n_params: int = 0) -> dict:
        self.flush()
        res = self.acc.result()
        by_type = res.pop("by_type", {})
        from common.constants import TYPE_NAMES
        return {
            "method": name,
            "eval_runs": runs,
            "n_samples": self.acc.n,
            "learnable_params": n_params,
            "mean_latency_ms": round(float(np.mean(self.wall_ms)), 3)
                               if self.wall_ms else None,
            "overall": {k: v for k, v in res.items()
                        if isinstance(v, (int, float))},
            "by_type": {TYPE_NAMES[t] if 0 <= t < N_TYPES else str(t): v
                        for t, v in by_type.items()},
        }


def _point_to_mixture(pred: np.ndarray, fitted: FittedSigma
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig, rho = fitted.attach(pred)
    return (pred[None], sig[None], rho[None], np.array([1.0]))


# ------------------------------------------------------------------ fitting

def fit_cv_sigma(train_runs: list[str], data_cfg: dict, variant: str,
                 max_samples: int = 20000) -> FittedSigma:
    f = FittedSigma()
    n = 0
    for ep, groups in iter_eval_groups(train_runs, data_cfg, max_samples):
        for a, t0s in groups.items():
            for t0 in t0s:
                pred = (cv_predict_obs(ep, t0, a) if variant == "obs"
                        else cv_predict_gt(ep, t0, a))
                if pred is None:
                    continue
                fut, val = future_of(ep, t0, a)
                f.add_residuals(pred, fut, val)
                n += 1
    f.finalize()
    return f


# --------------------------------------------------------------- evaluation

def evaluate_cv(runs: list[str], data_cfg: dict, fitted: FittedSigma,
                variant: str) -> dict:
    col = _Collector()
    for ep, groups in iter_eval_groups(runs, data_cfg):
        for a, t0s in groups.items():
            ty = int(ep["gt_type"][a])
            for t0 in t0s:
                t_start = time.perf_counter()
                pred = (cv_predict_obs(ep, t0, a) if variant == "obs"
                        else cv_predict_gt(ep, t0, a))
                if pred is None:
                    continue
                mu, sig, rho, pi = _point_to_mixture(pred, fitted)
                col.wall_ms.append((time.perf_counter() - t_start) * 1000)
                fut, val = future_of(ep, t0, a)
                intent = int(ep["gt_intent"][t0, a]) if "gt_intent" in ep else -1
                col.add(mu, sig, rho, pi, fut, val, ty, intent)
    return col.report(f"CV-{variant}+fitsigma", runs)


def evaluate_imm(runs: list[str], data_cfg: dict, params: IMMParams) -> dict:
    col = _Collector()
    for ep, groups in iter_eval_groups(runs, data_cfg):
        t_start = time.perf_counter()
        preds = imm_predictions_for_episode(ep, groups, params)
        n_ep = max(1, len(preds))
        per = (time.perf_counter() - t_start) * 1000 / n_ep
        for a, t0s in groups.items():
            ty = int(ep["gt_type"][a])
            for t0 in t0s:
                got = preds.get((a, t0))
                if got is None:
                    continue
                mu, sig, rho, pi = got
                col.wall_ms.append(per)
                fut, val = future_of(ep, t0, a)
                intent = int(ep["gt_intent"][t0, a]) if "gt_intent" in ep else -1
                col.add(mu, sig, rho, pi, fut, val, ty, intent)
    return col.report("IMM(CV/CT/CA)", runs)


def tune_imm(valid_runs: list[str], data_cfg: dict,
             max_samples: int = 1200) -> tuple[IMMParams, list[dict]]:
    """Coarse grid on process noises, scored by mean mixture NLL."""
    # pass 1 [0.2..2.5]x[0.003..0.08] and pass 2 [0.05..0.5]x[0.05..1.0] both
    # optimised at the (q min, q_w max) corner; third re-centring below. If
    # the corner persists, IMM is trading sharpness for coverage by inflating
    # the turn-rate walk -- accept and document.
    grid_q = [0.01, 0.03, 0.05, 0.1]
    grid_w = [0.7, 2.0, 6.0]
    history = []
    best = None
    groups_cache = list(iter_eval_groups(valid_runs, data_cfg, max_samples))
    for q in grid_q:
        for qw in grid_w:
            params = IMMParams(q_cv=q * 0.6, q_ct=q, q_w=qw, q_ca=q * 0.8)
            col = _Collector()
            for ep, groups in groups_cache:
                preds = imm_predictions_for_episode(ep, groups, params)
                for a, t0s in groups.items():
                    ty = int(ep["gt_type"][a])
                    for t0 in t0s:
                        got = preds.get((a, t0))
                        if got is None:
                            continue
                        fut, val = future_of(ep, t0, a)
                        col.add(*got, fut, val, ty, -1)
            rep = col.report("tune", valid_runs)
            nll = rep["overall"].get("NLL", math.inf)
            history.append({"q": q, "q_w": qw, "NLL": nll,
                            "minADE": rep["overall"].get("minADE")})
            if best is None or nll < best[0]:
                best = (nll, params)
    return best[1], history
