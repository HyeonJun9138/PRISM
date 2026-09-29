"""Score any baseline with modelTraining's OWN metric code.

Every method here emits the same `out` dict the neural model does
(mu / log_sigma / rho / mode_logits in POS_SCALE units), so
model.metrics.batch_metrics computes minADE / minFDE / NLL / MC-HPD coverage
IDENTICALLY for every row of the paper's comparison table. Point methods get
sigma from a FittedSigma wrapper (fitted on train residuals), which makes
their NLL/coverage well-defined -- and demonstrably situation-blind.

Review round 1: the same model.metrics.extended_metrics (top-1 / mixture-mean
ADE/FDE, trajectory NLL, energy score, branch diagnostics), the same pooled
strata (track-history class, type, intent, optional avoidance labels), the
same per-sample file format and the same seeded generators as
modelTraining/analysis/evaluate.py. Observation-based methods read detections
under track_rule(data_cfg) (CV-obs: current-track window; IMM: filter
restarted on a new track id).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from common.constants import N_TYPES, POS_SCALE, TYPE_NAMES
from data.dataset import index_track_rule, track_rule
from model.metrics import (ES_SEED, MC_SEED, METRIC_DEFINITIONS,
                           MetricAccumulator, SampleTable, batch_metrics,
                           extended_metrics, load_avoid_labels, make_generator,
                           overall_extended, save_per_sample, standard_strata)

from harness.data import FILE_KEY, future_of, iter_eval_groups, reacq_class
from methods.cv import FittedSigma, cv_predict_gt, cv_predict_obs
from methods.imm import IMMParams, imm_predictions_for_episode

BATCH = 512


class _Collector:
    """Accumulate per-sample mixtures, flush through batch_metrics (and, if
    ``extended``, extended_metrics), keep the per-sample table + keys."""

    def __init__(self, extended: bool = True):
        self.acc = MetricAccumulator()
        self.buf: list[dict] = []
        self.wall_ms: list[float] = []
        self.extended = extended
        self.table = SampleTable()
        self.mc_gen = make_generator("cpu", MC_SEED)
        self.es_gen = make_generator("cpu", ES_SEED)
        self.files: list[str] = []
        self._file_pos: dict[str, int] = {}
        self.keys: list[tuple[int, int, int]] = []      # (file_idx, t0, agent)
        self.n_skipped = 0

    def add(self, mu_kh2: np.ndarray, sig_kh2: np.ndarray, rho_kh: np.ndarray,
            pi_k: np.ndarray, fut: np.ndarray, val: np.ndarray,
            gt_type: int, intent: int, key: tuple | None = None,
            reacq: int = -1) -> None:
        """``key`` = (episode path, t0, agent) of the sample."""
        if key is not None:
            f, t0, a = key
            if f not in self._file_pos:
                self._file_pos[f] = len(self.files)
                self.files.append(f)
            self.keys.append((self._file_pos[f], int(t0), int(a)))
        self.buf.append(dict(mu=mu_kh2, sig=sig_kh2, rho=rho_kh, pi=pi_k,
                             fut=fut, val=val, ty=gt_type, it=intent, re=reacq))
        if len(self.buf) >= BATCH:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        B = len(self.buf)
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
            "meta_reacq": torch.as_tensor([b["re"] for b in self.buf]),
        }
        m = batch_metrics(out, batch, mc_samples=128, generator=self.mc_gen)
        ext = extended_metrics(out, batch, self.es_gen) if self.extended else None
        self.table.add(m["_per_sample"], ext)
        self.acc.update(m)
        self.buf.clear()

    def report(self, name: str, runs: list[str], n_params: int = 0,
               labels_dir: str | None = None,
               per_sample_path: str | None = None,
               meta: dict | None = None) -> dict:
        self.flush()
        res = self.acc.result()
        by_type = res.pop("by_type", {})
        overall = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        rep = {
            "method": name,
            "eval_runs": runs,
            "n_samples": self.acc.n,
            "learnable_params": n_params,
            "mean_latency_ms": round(float(np.mean(self.wall_ms)), 3)
                               if self.wall_ms else None,
            "overall": overall,
            "by_type": {TYPE_NAMES[t] if 0 <= t < N_TYPES else str(t): v
                        for t, v in by_type.items()},
        }
        if not self.extended or not len(self.table):
            return rep
        cols = self.table.finalize()
        overall.update(overall_extended(cols))            # new keys only
        avoid, avoid_stats = None, None
        if labels_dir and self.keys:
            k = np.asarray(self.keys, np.int64)
            avoid, avoid_stats = load_avoid_labels(self.files, k[:, 0], k[:, 1],
                                                   k[:, 2], labels_dir)
        rep.update({"n_skipped": self.n_skipped,
                    "mc_seed": MC_SEED, "es_seed": ES_SEED, "batch_size": BATCH,
                    "strata": standard_strata(cols, avoid),
                    "avoid_labels": avoid_stats,
                    "metric_definitions": METRIC_DEFINITIONS,
                    "per_sample_path": None})
        if per_sample_path and self.keys:
            k = np.asarray(self.keys, np.int64)
            extra = {} if avoid is None else {"avoid": avoid}
            ps_meta = {"method": name, "method_name": name, "n_samples": self.acc.n}
            ps_meta.update(meta or {})               # the caller's tag wins
            save_per_sample(per_sample_path, self.files, k[:, 0], k[:, 1], k[:, 2],
                            cols, ps_meta, extra)
            rep["per_sample_path"] = str(per_sample_path).replace("\\", "/")
        return rep


def _point_to_mixture(pred: np.ndarray, fitted: FittedSigma
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig, rho = fitted.attach(pred)
    return (pred[None], sig[None], rho[None], np.array([1.0]))


def _rules(data_cfg: dict) -> dict:
    return {"track_rule": track_rule(data_cfg),
            "index_rule": index_track_rule(data_cfg)}


# ------------------------------------------------------------------ fitting

def fit_cv_sigma(train_runs: list[str], data_cfg: dict, variant: str,
                 max_samples: int = 20000) -> FittedSigma:
    rule = track_rule(data_cfg)
    f = FittedSigma()
    n = 0
    for ep, groups in iter_eval_groups(train_runs, data_cfg, max_samples):
        for a, t0s in groups.items():
            for t0 in t0s:
                pred = (cv_predict_obs(ep, t0, a, rule=rule) if variant == "obs"
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
                variant: str, labels_dir: str | None = None,
                per_sample_path: str | None = None, meta: dict | None = None) -> dict:
    rule = track_rule(data_cfg)
    col = _Collector()
    for ep, groups in iter_eval_groups(runs, data_cfg):
        for a, t0s in groups.items():
            ty = int(ep["gt_type"][a])
            for t0 in t0s:
                t_start = time.perf_counter()
                pred = (cv_predict_obs(ep, t0, a, rule=rule) if variant == "obs"
                        else cv_predict_gt(ep, t0, a))
                if pred is None:
                    col.n_skipped += 1
                    continue
                mu, sig, rho, pi = _point_to_mixture(pred, fitted)
                col.wall_ms.append((time.perf_counter() - t_start) * 1000)
                fut, val = future_of(ep, t0, a)
                intent = int(ep["gt_intent"][t0, a]) if "gt_intent" in ep else -1
                col.add(mu, sig, rho, pi, fut, val, ty, intent,
                        key=(ep[FILE_KEY], t0, a), reacq=reacq_class(ep, t0, a))
    rep = col.report(f"CV-{variant}+fitsigma", runs, labels_dir=labels_dir,
                     per_sample_path=per_sample_path, meta=meta)
    rep.update(_rules(data_cfg))
    return rep


def evaluate_imm(runs: list[str], data_cfg: dict, params: IMMParams,
                 labels_dir: str | None = None,
                 per_sample_path: str | None = None, meta: dict | None = None,
                 init_gap_fix: bool = False) -> dict:
    rule = track_rule(data_cfg)
    col = _Collector()
    for ep, groups in iter_eval_groups(runs, data_cfg):
        t_start = time.perf_counter()
        preds = imm_predictions_for_episode(ep, groups, params, rule=rule,
                                            init_gap_fix=init_gap_fix)
        n_ep = max(1, len(preds))
        per = (time.perf_counter() - t_start) * 1000 / n_ep
        for a, t0s in groups.items():
            ty = int(ep["gt_type"][a])
            for t0 in t0s:
                got = preds.get((a, t0))
                if got is None:
                    col.n_skipped += 1
                    continue
                mu, sig, rho, pi = got
                col.wall_ms.append(per)
                fut, val = future_of(ep, t0, a)
                intent = int(ep["gt_intent"][t0, a]) if "gt_intent" in ep else -1
                col.add(mu, sig, rho, pi, fut, val, ty, intent,
                        key=(ep[FILE_KEY], t0, a), reacq=reacq_class(ep, t0, a))
    rep = col.report("IMM(CV/CT/CA)", runs, labels_dir=labels_dir,
                     per_sample_path=per_sample_path, meta=meta)
    rep.update(_rules(data_cfg))
    rep["imm_init_gap_fix"] = bool(init_gap_fix)
    return rep


def tune_imm(valid_runs: list[str], data_cfg: dict,
             max_samples: int = 1200) -> tuple[IMMParams, list[dict]]:
    """Coarse grid on process noises, scored by mean mixture NLL."""
    # pass 1 [0.2..2.5]x[0.003..0.08] and pass 2 [0.05..0.5]x[0.05..1.0] both
    # optimised at the (q min, q_w max) corner; third re-centring below. If
    # the corner persists, IMM is trading sharpness for coverage by inflating
    # the turn-rate walk -- accept and document.
    rule = track_rule(data_cfg)
    grid_q = [0.01, 0.03, 0.05, 0.1]
    grid_w = [0.7, 2.0, 6.0]
    history = []
    best = None
    groups_cache = list(iter_eval_groups(valid_runs, data_cfg, max_samples))
    for q in grid_q:
        for qw in grid_w:
            params = IMMParams(q_cv=q * 0.6, q_ct=q, q_w=qw, q_ca=q * 0.8)
            col = _Collector(extended=False)
            for ep, groups in groups_cache:
                preds = imm_predictions_for_episode(ep, groups, params, rule=rule)
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


# --------------------------------------------------- IMM tuning (round 1)
# The pre-round-1 IMM tuning protocol, reconstructed and VERIFIED on
# 2026-09-26: re-running the recorded pass (tune_imm above) with the
# pre-change code reproduces all 12 NLLs stored in
# Logs/Models/baselines/imm_params.json bit-for-bit, and the frozen point
# q=0.05, q_w=1.0 scores 2.13224 -- the file note's "val NLL 2.132".
#   data       valid_full5x, the FIRST max_samples=1200 samples of the index
#              in iter_eval_groups order (5 episodes); never a test split
#   objective  mean per-step marginal mixture NLL ("NLL"), minimised
#   family     q_cv = 0.6 q, q_ct = q, q_ca = 0.8 q, and q_w;
#              p_stay = 0.95 and r_infl = 1.0 fixed
#   search     three coarse passes; final = argmin over ALL evaluated points
#              (the legacy pick q=0.05, q_w=1.0 came from pass 2 and beat
#              every pass-3 point on the same 1200 samples)
# Passes 1-2 are documented only as ranges (tune_imm comment): here they are
# discretised at 4 x 4 points including both range ends; pass 3 is exact.
# The legacy "re-centring" step becomes a rule: while the argmin lies on the
# edge of the evaluated q or q_w values, evaluate the 3 x 3 neighbourhood
# {x/3, x, 3x} around it (at most max_extra times).
TUNE_PASSES = (
    ("pass1", (0.2, 0.5, 1.0, 2.5), (0.003, 0.01, 0.03, 0.08)),
    ("pass2", (0.05, 0.1, 0.2, 0.5), (0.05, 0.2, 0.5, 1.0)),
    ("pass3", (0.01, 0.03, 0.05, 0.1), (0.7, 2.0, 6.0)),
)
TUNE_MAX_SAMPLES = 1200

_TUNE_STATE: dict = {}


def _imm_family(q: float, q_w: float) -> IMMParams:
    return IMMParams(q_cv=q * 0.6, q_ct=q, q_w=q_w, q_ca=q * 0.8)


def _tune_init(valid_runs: list[str], data_cfg: dict, max_samples: int,
               init_gap_fix: bool) -> None:
    """Per-process tuning state: the capped valid sample groups."""
    _TUNE_STATE["cache"] = list(iter_eval_groups(valid_runs, data_cfg, max_samples))
    _TUNE_STATE["rule"] = track_rule(data_cfg)
    _TUNE_STATE["gap"] = bool(init_gap_fix)


def _tune_point(point: tuple[float, float]) -> dict:
    """Score one (q, q_w) on the cached tuning samples."""
    q, qw = point
    t_start = time.perf_counter()
    col = _Collector()
    params = _imm_family(q, qw)
    for ep, groups in _TUNE_STATE["cache"]:
        preds = imm_predictions_for_episode(ep, groups, params, rule=_TUNE_STATE["rule"],
                                            init_gap_fix=_TUNE_STATE["gap"])
        for a, t0s in groups.items():
            ty = int(ep["gt_type"][a])
            for t0 in t0s:
                got = preds.get((a, t0))
                if got is None:
                    col.n_skipped += 1
                    continue
                fut, val = future_of(ep, t0, a)
                col.add(*got, fut, val, ty, -1)
    o = col.report("tune", [])["overall"]
    return {"q": q, "q_w": qw, "NLL": o.get("NLL", math.inf),
            "minADE": o.get("minADE"), "minFDE": o.get("minFDE"),
            "trajNLL": o.get("trajNLL"), "ES": o.get("ES"),
            "cov90_15s": o.get("cov90_15s"), "n": col.acc.n,
            "n_skipped": col.n_skipped,
            "wall_s": round(time.perf_counter() - t_start, 2)}


def tune_imm_protocol(valid_runs: list[str], data_cfg: dict,
                      init_gap_fix: bool = False,
                      max_samples: int = TUNE_MAX_SAMPLES,
                      passes: tuple = TUNE_PASSES, workers: int = 1,
                      max_extra: int = 3, log=print
                      ) -> tuple[IMMParams, list[dict], dict]:
    """The pre-round-1 tuning protocol (block comment above) under the track
    rule of ``data_cfg`` and the given IMM init option. ``workers`` > 1 scores
    grid points in that many processes (results do not depend on it).

    Returns (params, history of every evaluated point, best point).
    """
    from concurrent.futures import ProcessPoolExecutor
    init_args = (list(valid_runs), dict(data_cfg), int(max_samples), bool(init_gap_fix))
    done: dict[tuple[float, float], dict] = {}
    history: list[dict] = []
    ex = (ProcessPoolExecutor(max_workers=int(workers), initializer=_tune_init,
                              initargs=init_args) if workers > 1 else None)
    if ex is None:
        _tune_init(*init_args)
    try:
        def run(points, tag):
            todo = [p for p in dict.fromkeys(points) if p not in done]
            res = list(ex.map(_tune_point, todo)) if ex else [_tune_point(p) for p in todo]
            for r in res:
                r["pass"] = tag
                done[(r["q"], r["q_w"])] = r
                history.append(r)
                log(f"  {tag:7s} q={r['q']:<8g} q_w={r['q_w']:<8g} NLL={r['NLL']:.5f} "
                    f"minADE={r['minADE']:.2f} n={r['n']} ({r['wall_s']:.1f}s)")

        for tag, gq, gw in passes:
            run([(float(q), float(w)) for q in gq for w in gw], tag)
        for i in range(int(max_extra)):
            best = min(done.values(), key=lambda r: r["NLL"])
            qs = sorted({k[0] for k in done})
            ws = sorted({k[1] for k in done})
            if best["q"] not in (qs[0], qs[-1]) and best["q_w"] not in (ws[0], ws[-1]):
                break
            run([(round(best["q"] * f, 6), round(best["q_w"] * g, 6))
                 for f in (1 / 3, 1.0, 3.0) for g in (1 / 3, 1.0, 3.0)], f"extra{i + 1}")
    finally:
        if ex is not None:
            ex.shutdown()
    best = min(done.values(), key=lambda r: r["NLL"])
    return _imm_family(best["q"], best["q_w"]), history, best
