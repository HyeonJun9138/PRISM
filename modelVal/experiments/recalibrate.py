"""Post-hoc temperature (variance) recalibration for the PRISM predictor.

Fits ONE scalar T on the calibration runs: sigma -> T * sigma, chosen to
minimise mixture NLL on the calibration set. Reports before/after NLL and HPD
coverage on a HELD-OUT OOD evaluation set, and the transfer of the same T to
the factor-wise OOD sets of DECISIONS.md D5.5 (R3 major 2: "specific to
recalibration under the tested shift unless transfer ... is evaluated").

Evaluation set (review round 1, R3 major 2 / R2-IV) -- read before editing
--------------------------------------------------------------------------
The calibration runs (``*test_ood_cal_*``: 300 episodes, master seeds
300000-300299) are bit-identical regenerations of OOD test episodes 0-299
(same preset ``test_ood``, same seeds). The released recalibration.json scored
the fitted T on ALL 1200 test episodes (full5x_a: episode_index 0-599,
full5x_b: 600-1199), so 300 of its episodes -- 129,938 of its 520,079 samples
-- were the fitting episodes. The default evaluation set is now the test
episodes whose master seed is >= 300300 (episode_index 300-1199), with seeds
read from each run's index.jsonl, and the script ASSERTS zero seed overlap
between the fitting and evaluation sets before anything is scored.
``--legacy`` restores the released evaluation set and reproduces the released
file (the gate, ``--expect``).

Note on the Sep-7 ``--holdout`` flag this replaces: it looked up a ``file``
key that index.jsonl rows do not have, so it excluded nothing and
Paper/rebuttal/recalibration_holdout.json also scored all 520,079 samples.

HPD Monte Carlo (recorded in the output under ``hpd_mc``)
---------------------------------------------------------
Per (sample, horizon step): HPD_MC = 96 draws from the predicted mixture; the
coverage indicator is log p(GT) >= the (1 - nominal) quantile of the draws'
log-densities (torch.quantile, linear). Standard normals come from a
torch.Generator seeded 7 and mixture components from one seeded
``--comp-seed`` (default 8), both re-seeded for every coverage call, so the
before/after blocks share their draws.

The released script drew the components with ``torch.multinomial`` WITHOUT a
generator, i.e. from torch's default CPU generator, which on this Windows
build is seeded from the steady clock at process start
(``torch.initial_seed()`` differs on every run). Its coverage values are one
unseeded Monte-Carlo realisation and cannot be reproduced bit for bit -- the
Sep-7 rerun on the same 520,079 samples already differed in the 4th decimal.
``--expect`` therefore checks the non-MC fields (model, T, cal_runs,
t0_stride, n_cal, n_test, both NLLs) EXACTLY and, with ``--mc-replicates R``,
each released coverage value against R replicate component streams (the only
unseeded input): it must lie within Z_MAX replicate SDs.

Samples: t0 in range(HIST_STEPS - 1, T - PRED_STEPS, 4), eligibility
>= min_obs usable detections under ``index_track_rule(data_cfg)``, GT valid
at t0 and >= 27 valid future steps; features follow the model's own
``track_rule`` (legacy checkpoints: "slot"). Episodes are featurised in a
process pool (``--workers``) and forwarded in the released order and batch
composition (512 consecutive samples).

  python -m experiments.recalibrate --out <json> [--model <ckpt>] [--legacy]
         [--transfer auto|none] [--workers 10] [--expect <released json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.constants import HIST_STEPS, PRED_STEPS  # noqa: E402
from data.dataset import (build_sample, index_track_rule,  # noqa: E402
                          track_rule, window_detections)
from model.network import build_model  # noqa: E402

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

MODEL = f"{PRISM_ROOT}/Logs/Models/20260813_170429_full5x_s4/final/model_final.pt"
CAL_RUNS = sorted(glob.glob(rf"{PRISM_ROOT}/Logs/Dataset/*test_ood_cal_*"))
TEST_RUNS = [rf"{PRISM_ROOT}/Logs/Dataset/20260813_023139_test_ood_full5x_a",
             rf"{PRISM_ROOT}/Logs/Dataset/20260813_023139_test_ood_full5x_b"]
T0_STRIDE = 4            # match training-index stride; keeps runtime sane
MIN_FUTURE_VALID = 27
BATCH = 512
HPD_MC = 96              # MC samples per (sample, step) for the HPD threshold
NORMAL_SEED = 7
COMPONENT_SEED = 8       # mixture-component generator seed (--comp-seed)
#: MC-consistency gate: |released - replicate mean| <= Z_MAX replicate SDs
Z_MAX = 4.0
#: fields of the released recalibration.json that involve no MC draw
DETERMINISTIC_FIELDS = ("model", "T", "cal_runs", "t0_stride", "n_cal",
                        "n_test")
HORIZON_STEPS = {5: 9, 10: 19, 15: 29}
NOMINALS = (0.5, 0.7, 0.9)
#: first master seed of the held-out evaluation set (episode_index 300)
EVAL_MIN_SEED = 300300
#: factor-wise OOD sets of DECISIONS.md D5.5, in the order of its table
FACTOR_ORDER = ("control", "density", "wind", "intent", "platform", "all")
FACTOR_GLOB = f"{PRISM_ROOT}/Logs/Dataset/*_ood_{{}}_rebut"


# ============================================================ episode sets

def run_episodes(run: str) -> list[dict]:
    """Episodes of a run in file order, with master seeds from index.jsonl.

    Raises when files and index rows disagree -- a silently unmatched file is
    exactly how the Sep-7 hold-out flag failed.
    """
    rows = {}
    with open(os.path.join(run, "index.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[int(r["episode_index"])] = r
    files = sorted(glob.glob(os.path.join(run, "episodes", "*.npz")))
    out = []
    for f in files:
        idx = int(os.path.basename(f)[3:9])
        if idx not in rows:
            raise KeyError(f"{f}: episode_index {idx} missing from index.jsonl")
        out.append({"run": run, "episode_index": idx,
                    "seed": int(rows[idx]["seed"]), "path": f})
    if len(out) != len(rows):
        raise ValueError(f"{run}: {len(rows)} index rows but {len(out)} files")
    return out


def seed_overlap(fit_seeds, eval_seeds) -> list[int]:
    return sorted(set(int(s) for s in fit_seeds) & set(int(s) for s in eval_seeds))


def assert_no_seed_overlap(fit_seeds, eval_seeds, what: str = "") -> None:
    ov = seed_overlap(fit_seeds, eval_seeds)
    if ov:
        raise AssertionError(
            f"{what}: {len(ov)} master seeds are in both the fitting and the "
            f"evaluation set (e.g. {ov[:5]}); the evaluation is not held out")


def select_eval(test_eps: list[dict], legacy: bool,
                min_seed: int = EVAL_MIN_SEED) -> list[dict]:
    return list(test_eps) if legacy else [e for e in test_eps
                                          if e["seed"] >= int(min_seed)]


def _seed_summary(eps: list[dict]) -> dict:
    s = [e["seed"] for e in eps]
    i = [e["episode_index"] for e in eps]
    return {"n_episodes": len(eps),
            "seed_min": min(s) if s else None, "seed_max": max(s) if s else None,
            "episode_index_min": min(i) if i else None,
            "episode_index_max": max(i) if i else None,
            "seeds_contiguous": bool(s) and (max(s) - min(s) + 1 == len(set(s)))}


def find_factor_runs() -> dict[str, str]:
    out = {}
    for f in FACTOR_ORDER:
        hits = sorted(glob.glob(FACTOR_GLOB.format(f)))
        if hits:
            out[f] = hits[-1].replace("\\", "/")
    return out


# ======================================================= sample building

def _episode_samples(task):
    """Worker: every eligible (t0, agent) sample of one episode file."""
    path, data_cfg, stride, min_fut = task
    with np.load(path) as z:
        ep = {k: z[k] for k in z.files}
    gtv = ep["gt_valid"].astype(bool)
    T, N = gtv.shape
    rule = index_track_rule(data_cfg)
    min_obs = max(2, int(data_cfg.get("min_obs", 3)))
    xs, ms, ys, yvs, meta = [], [], [], [], []
    for t0 in range(HIST_STEPS - 1, T - PRED_STEPS, stride):
        n_det = window_detections(ep, t0, rule).sum(axis=0)
        n_fut = gtv[t0 + 1: t0 + 1 + PRED_STEPS].sum(axis=0)
        for a in range(N):
            # same set and order as the released loop: detections, GT at t0,
            # then (there: after building) >= min_fut valid future steps
            if n_det[a] < min_obs or not gtv[t0, a] or n_fut[a] < min_fut:
                continue
            s = build_sample(ep, t0, a, data_cfg, with_labels=True)
            xs.append(s["x"]); ms.append(s["agent_mask"])
            ys.append(s["y"]); yvs.append(s["y_valid"])
            meta.append((t0, a, int(s["meta_reacq"])))
    if not xs:
        return {"path": path, "n": 0}
    return {"path": path, "n": len(xs), "x": np.stack(xs), "m": np.stack(ms),
            "y": np.stack(ys), "yv": np.stack(yvs),
            "meta": np.asarray(meta, dtype=np.int32)}


class _Forwarder:
    """Forwards samples in consecutive batches of exactly BATCH (the released
    forward_all batch composition), keeping only the outputs."""

    def __init__(self, model, device, batch=BATCH):
        self.model, self.device, self.batch = model, device, int(batch)
        self.outs = {k: [] for k in ("mu", "sigma", "rho", "pi", "y", "yv")}
        self.pend = {k: [] for k in ("x", "m", "y", "yv")}
        self.n_pend = 0

    def _run(self, x, m, y, yv):
        xt = torch.as_tensor(x).to(self.device)
        mt = torch.as_tensor(m).to(self.device)
        with torch.no_grad():
            o = self.model(xt, mt)
        self.outs["mu"].append(o["mu"].float().cpu())
        self.outs["sigma"].append(o["log_sigma"].float().exp().cpu())
        self.outs["rho"].append(o["rho"].float().cpu())
        self.outs["pi"].append(torch.softmax(o["mode_logits"].float(), 1).cpu())
        self.outs["y"].append(torch.as_tensor(y))
        self.outs["yv"].append(torch.as_tensor(yv))

    def add(self, r: dict) -> None:
        if not r["n"]:
            return
        for k in self.pend:
            self.pend[k].append(r[k])
        self.n_pend += r["n"]
        if self.n_pend >= self.batch:
            cat = {k: np.concatenate(v) for k, v in self.pend.items()}
            n_full = (self.n_pend // self.batch) * self.batch
            for i in range(0, n_full, self.batch):
                self._run(*(np.ascontiguousarray(cat[k][i:i + self.batch])
                            for k in ("x", "m", "y", "yv")))
            self.pend = {k: ([cat[k][n_full:]] if n_full < self.n_pend else [])
                         for k in cat}
            self.n_pend -= n_full

    def finish(self) -> dict:
        if self.n_pend:
            cat = {k: np.ascontiguousarray(np.concatenate(v))
                   for k, v in self.pend.items()}
            self._run(cat["x"], cat["m"], cat["y"], cat["yv"])
            self.pend = {k: [] for k in self.pend}
            self.n_pend = 0
        return {k: torch.cat(v) for k, v in self.outs.items()}


def forward_set(pool, model, device, eps: list[dict], data_cfg: dict,
                label: str, stride: int = T0_STRIDE,
                min_fut: int = MIN_FUTURE_VALID):
    """Featurise ``eps`` in the pool (ordered) and forward on ``device``.
    Returns (outputs, per-episode sample counts)."""
    tasks = [(e["path"], data_cfg, stride, min_fut) for e in eps]
    fw = _Forwarder(model, device)
    counts = []
    t_start = time.time()
    it = (pool.imap(_episode_samples, tasks, chunksize=1) if pool is not None
          else map(_episode_samples, tasks))
    for i, r in enumerate(it):
        fw.add(r)
        counts.append(int(r["n"]))
        if (i + 1) % 100 == 0 or i + 1 == len(tasks):
            print(f"  [{label}] {i + 1}/{len(tasks)} episodes, "
                  f"{sum(counts)} samples, {time.time() - t_start:.0f}s",
                  flush=True)
    return fw.finish(), counts


# ================================================================ metrics

def mixture_nll(d, T=1.0):
    """Step-mean mixture NLL (scaled units), masked by y_valid."""
    mu, sig, rho, pi, y, yv = (d["mu"], d["sigma"] * T, d["rho"], d["pi"],
                               d["y"], d["yv"])
    B, K, S, _ = mu.shape
    dx = (y[:, None, :, 0] - mu[..., 0]) / sig[..., 0]
    dy = (y[:, None, :, 1] - mu[..., 1]) / sig[..., 1]
    omr = 1 - rho ** 2
    z = (dx ** 2 - 2 * rho * dx * dy + dy ** 2) / omr
    logN = -(z / 2) - torch.log(2 * math.pi * sig[..., 0] * sig[..., 1]
                                * torch.sqrt(omr))
    w = torch.log(pi[:, :, None].clamp_min(1e-9)) + logN      # [B,K,S]
    step_ll = torch.logsumexp(w, dim=1)                        # [B,S]
    mask = yv.bool()
    return float((-step_ll[mask]).mean())


def coverage(d, T, gen, comp_gen, hpd_mc=HPD_MC):
    """HPD MC coverage at NOMINALS x HORIZON_STEPS (released arithmetic;
    every draw from the given seeded generators)."""
    mu, sig, rho, pi, y, yv = (d["mu"], d["sigma"] * T, d["rho"], d["pi"],
                               d["y"], d["yv"])
    B, K, S, _ = mu.shape
    out = {}
    for h_lab, s_idx in HORIZON_STEPS.items():
        m, sg, rh = mu[:, :, s_idx], sig[:, :, s_idx], rho[:, :, s_idx]
        valid = yv[:, s_idx].bool()
        # draw component indices then correlated normals
        comp = torch.multinomial(pi, hpd_mc, replacement=True,
                                 generator=comp_gen)               # [B,M]
        idx = comp[..., None]
        mm = torch.gather(m, 1, idx.expand(-1, -1, 2))             # [B,M,2]
        ss = torch.gather(sg, 1, idx.expand(-1, -1, 2))
        rr = torch.gather(rh, 1, comp)
        e1 = torch.randn(B, hpd_mc, generator=gen)
        e2 = torch.randn(B, hpd_mc, generator=gen)
        sx = mm[..., 0] + ss[..., 0] * e1
        sy = mm[..., 1] + ss[..., 1] * (rr * e1
                                        + torch.sqrt(1 - rr ** 2) * e2)

        def dens(px, py):
            dx = (px[:, :, None] - m[:, None, :, 0]) / sg[:, None, :, 0]
            dy = (py[:, :, None] - m[:, None, :, 1]) / sg[:, None, :, 1]
            r = rh[:, None, :]
            omr = 1 - r ** 2
            z = (dx ** 2 - 2 * r * dx * dy + dy ** 2) / omr
            logN = -(z / 2) - torch.log(
                2 * math.pi * sg[:, None, :, 0] * sg[:, None, :, 1]
                * torch.sqrt(omr))
            return torch.logsumexp(
                torch.log(pi[:, None, :].clamp_min(1e-9)) + logN, dim=2)

        samp_d = dens(sx, sy)                                      # [B,M]
        gt_d = dens(y[:, None, s_idx, 0], y[:, None, s_idx, 1])[:, 0]
        for nom in NOMINALS:
            thr = torch.quantile(samp_d, 1 - nom, dim=1)
            cov = (gt_d >= thr)[valid].float().mean()
            out[f"cov{int(nom*100)}_{h_lab}s"] = round(float(cov), 4)
    return out


def evaluate(d, T: float, comp_seed: int = COMPONENT_SEED,
             hpd_mc: int = HPD_MC, with_nll: bool = True) -> dict:
    """before (T = 1) / after (T) blocks; both use the same freshly seeded
    normal and component streams (common random numbers)."""
    res = {}
    for key, tt in (("before", 1.0), ("after", T)):
        cov = coverage(d, tt, torch.Generator().manual_seed(NORMAL_SEED),
                       torch.Generator().manual_seed(int(comp_seed)), hpd_mc)
        res[key] = ({"NLL": round(mixture_nll(d, tt), 4), **cov} if with_nll
                    else cov)
    return res


def fit_temperature(cal) -> tuple[float, dict]:
    """Golden-section search on NLL(T), exactly as released."""
    lo, hi = 0.8, 2.6
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = mixture_nll(cal, a), mixture_nll(cal, b)
    for _ in range(24):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = mixture_nll(cal, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = mixture_nll(cal, b)
    T = round((lo + hi) / 2, 3)
    info = {"method": "golden-section on mixture NLL, bracket [0.8, 2.6], "
                      "24 iterations, T rounded to 3 decimals",
            "T_unrounded": (lo + hi) / 2,
            "cal_NLL_T1": round(mixture_nll(cal, 1.0), 4),
            "cal_NLL_T": round(mixture_nll(cal, T), 4)}
    return T, info


# =================================================================== main

def main(argv=None):
    from experiments.rollout_score import (compare_fields, file_sha256,
                                           guard_output_path, json_roundtrip)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True,
                    help="output JSON (never Paper/draft_work or the released "
                         "Paper/rebuttal/*.json)")
    ap.add_argument("--legacy", action="store_true",
                    help="released evaluation set (all 1200 test episodes, "
                         "incl. the 300 fitting episodes)")
    ap.add_argument("--comp-seed", type=int, default=COMPONENT_SEED)
    ap.add_argument("--eval-min-seed", type=int, default=EVAL_MIN_SEED)
    ap.add_argument("--T-fixed", type=float, default=None,
                    help="skip the fit and use this T")
    ap.add_argument("--transfer", default="auto",
                    help="'auto' = the D5.5 factor-wise OOD runs, 'none', or "
                         "a comma-separated list of run dirs")
    ap.add_argument("--index-track-rule", choices=("current", "slot"),
                    default=None, help="override eligibility rule only")
    ap.add_argument("--hpd-mc", type=int, default=HPD_MC)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--expect", default=None,
                    help="released JSON: non-MC fields (and both NLLs) must "
                         "match exactly")
    ap.add_argument("--mc-replicates", type=int, default=0,
                    help="with --expect: replicate component streams for the "
                         "MC-consistency gate of the released coverage values")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out = guard_output_path(args.out, args.overwrite)
    sets_out = (out[:-5] if out.endswith(".json") else out) + "_sets.json"
    guard_output_path(sets_out, args.overwrite)
    t_all = time.time()

    # --- model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") or ck["cfg"]["model"]
    data_cfg = dict(ck.get("data_cfg") or ck["cfg"]["data"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if args.index_track_rule:
        data_cfg["index_track_rule"] = args.index_track_rule

    # --- sets and the overlap check (before any scoring)
    cal_eps = [e for r in CAL_RUNS for e in run_episodes(r)]
    test_all = [e for r in TEST_RUNS for e in run_episodes(r)]
    eval_eps = select_eval(test_all, args.legacy, args.eval_min_seed)
    cal_seeds = [e["seed"] for e in cal_eps]
    overlap = seed_overlap(cal_seeds, [e["seed"] for e in eval_eps])
    if not args.legacy:
        assert_no_seed_overlap(cal_seeds, [e["seed"] for e in eval_eps],
                               "calibration vs evaluation")
    if args.legacy or args.transfer == "none":
        transfer = {}
    elif args.transfer == "auto":
        transfer = find_factor_runs()
    else:
        transfer = {os.path.basename(p.rstrip("/\\")): p
                    for p in args.transfer.split(",") if p}
    transfer_eps = {k: run_episodes(v) for k, v in transfer.items()}
    for k, eps in transfer_eps.items():
        assert_no_seed_overlap(cal_seeds, [e["seed"] for e in eps],
                               f"calibration vs transfer set {k}")
    print(f"model {args.model}\n  track_rule={track_rule(data_cfg)} "
          f"index_track_rule={index_track_rule(data_cfg)}")
    print(f"cal: {len(CAL_RUNS)} runs / {len(cal_eps)} episodes; eval: "
          f"{len(eval_eps)} episodes ({'LEGACY all' if args.legacy else 'held out'}"
          f", seed overlap with cal = {len(overlap)}); transfer: "
          f"{list(transfer)}", flush=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers) if args.workers > 1 else None
    timing = {}
    try:
        t = time.time()
        cal, cal_counts = forward_set(pool, model, device, cal_eps, data_cfg, "cal")
        timing["cal_forward_s"] = round(time.time() - t, 1)
        print(f"cal samples: {cal['mu'].shape[0]}", flush=True)
        if args.T_fixed is None:
            T, fit_info = fit_temperature(cal)
        else:
            T, fit_info = float(args.T_fixed), {
                "method": "fixed by --T-fixed",
                "cal_NLL_T1": round(mixture_nll(cal, 1.0), 4),
                "cal_NLL_T": round(mixture_nll(cal, float(args.T_fixed)), 4)}
        print(f"T = {T}  (cal NLL {fit_info['cal_NLL_T1']:.4f} -> "
              f"{fit_info['cal_NLL_T']:.4f})", flush=True)
        n_cal = int(cal["mu"].shape[0])
        del cal

        t = time.time()
        test, test_counts = forward_set(pool, model, device, eval_eps, data_cfg,
                                        "eval")
        n_test = int(test["mu"].shape[0])
        print(f"eval samples: {n_test}", flush=True)
        ev = evaluate(test, T, args.comp_seed, args.hpd_mc)
        timing["eval_s"] = round(time.time() - t, 1)
        reps = []
        if args.expect and args.mc_replicates > 0:
            t = time.time()
            for r in range(args.mc_replicates):
                reps.append(evaluate(test, T, args.comp_seed + 1 + r,
                                     args.hpd_mc, with_nll=False))
                print(f"  MC replicate {r + 1}/{args.mc_replicates} "
                      f"({time.time() - t:.0f}s)", flush=True)
            timing["mc_replicates_s"] = round(time.time() - t, 1)
        del test

        transfer_res, transfer_counts = {}, {}
        for k, eps in transfer_eps.items():
            t = time.time()
            d, cnt = forward_set(pool, model, device, eps, data_cfg, k)
            transfer_counts[k] = cnt
            transfer_res[k] = {"run": transfer[k], **_seed_summary(eps),
                               "n_samples": int(d["mu"].shape[0]),
                               **evaluate(d, T, args.comp_seed, args.hpd_mc)}
            timing[f"transfer_{k}_s"] = round(time.time() - t, 1)
            del d
            print(f"  {k}: n={transfer_res[k]['n_samples']} before "
                  f"cov90@15={transfer_res[k]['before']['cov90_15s']} "
                  f"NLL={transfer_res[k]['before']['NLL']} -> after "
                  f"cov90@15={transfer_res[k]['after']['cov90_15s']} "
                  f"NLL={transfer_res[k]['after']['NLL']}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ---- released fields first, in the released order
    res = {"model": args.model, "T": T, "cal_runs": CAL_RUNS,
           "t0_stride": T0_STRIDE, "n_cal": n_cal, "n_test": n_test,
           "before": ev["before"], "after": ev["after"]}
    # ---- R3 major 2 additions
    res["eval_set"] = {
        "legacy": bool(args.legacy),
        "rule": ("all test episodes (released set; includes the 300 fitting "
                 "episodes)" if args.legacy else
                 f"test episodes with master seed >= {args.eval_min_seed}"),
        "test_runs": TEST_RUNS, **_seed_summary(eval_eps)}
    res["fit_set"] = {"cal_runs": [r.replace("\\", "/") for r in CAL_RUNS],
                      **_seed_summary(cal_eps)}
    res["seed_overlap"] = {
        "n_fit_seeds": len(set(cal_seeds)),
        "n_eval_seeds": len({e["seed"] for e in eval_eps}),
        "n_overlap": len(overlap),
        "overlap_seed_min": overlap[0] if overlap else None,
        "overlap_seed_max": overlap[-1] if overlap else None,
        "asserted_zero": not args.legacy,
        "note": ("the calibration runs regenerate test_ood seeds "
                 "300000-300299 with the same preset, so equal seeds are "
                 "identical episodes")}
    res["n_test_samples_on_overlap_episodes"] = int(sum(
        c for e, c in zip(eval_eps, test_counts) if e["seed"] in set(overlap)))
    res["sampling"] = {"t0_stride": T0_STRIDE, "min_future_valid": MIN_FUTURE_VALID,
                       "min_obs": max(2, int(data_cfg.get("min_obs", 3))),
                       "index_track_rule": index_track_rule(data_cfg),
                       "feature_track_rule": track_rule(data_cfg)}
    res["hpd_mc"] = {
        "draws_per_sample_step": int(args.hpd_mc),
        "normal_draws": f"torch.Generator seeded {NORMAL_SEED}, re-seeded per "
                        "coverage call (before/after share the normals)",
        "component_draws": (f"torch.Generator seeded {args.comp_seed}, "
                            "re-seeded per coverage call (released: unseeded "
                            "torch default generator, clock-seeded per process)"),
        "threshold": "torch.quantile of the draws' log-densities at 1 - nominal",
        "indicator": "log p(GT) >= threshold, GT-valid samples at the step"}
    res["nll_definition"] = ("mean over GT-valid (sample, step) pairs of the "
                             "negative log mixture density, positions in units "
                             "of POS_SCALE = 100 m")
    res["fit"] = fit_info
    if transfer_res:
        res["transfer"] = transfer_res
    res["sets_sidecar"] = sets_out
    res["timing_s"] = {**timing, "total": round(time.time() - t_all, 1),
                       "workers": int(args.workers)}
    res["source"] = {"script": os.path.abspath(__file__).replace("\\", "/"),
                     "script_sha256": file_sha256(os.path.abspath(__file__)),
                     "dataset_py_sha256": file_sha256(
                         sys.modules["data.dataset"].__file__),
                     "torch": torch.__version__, "device": str(device)}

    gate_fail = False
    if args.expect:
        with open(args.expect, encoding="utf-8") as fh:
            expected = json.load(fh)
        det_exp = {k: expected[k] for k in DETERMINISTIC_FIELDS if k in expected}
        for blk in ("before", "after"):
            if blk in expected and "NLL" in expected[blk]:
                det_exp[blk] = {"NLL": expected[blk]["NLL"]}
        det_mism = compare_fields(json_roundtrip(res), det_exp)
        mc_exp = {blk: {k: v for k, v in expected[blk].items() if k != "NLL"}
                  for blk in ("before", "after") if blk in expected}
        gate = {"expected_file": args.expect.replace("\\", "/"),
                "deterministic_fields": sorted(DETERMINISTIC_FIELDS)
                + ["before/NLL", "after/NLL"],
                "deterministic_exact": not det_mism,
                "deterministic_mismatches": det_mism,
                "mc_fields_exact_with_this_seed": not compare_fields(
                    json_roundtrip(res), mc_exp),
                "note": "released coverage values came from an unseeded "
                        "(clock-seeded) torch default generator: exact equality "
                        "is not attainable, consistency is tested against "
                        "replicate component streams"}
        if reps:
            from experiments.rollout_score import mc_consistency
            cons = mc_consistency(mc_exp, reps, unit_of=lambda p: 1e-4,
                                  z_max=Z_MAX)
            cons["replicate_comp_seeds"] = [args.comp_seed + 1,
                                            args.comp_seed + len(reps)]
            gate["mc_consistency"] = cons
            gate["pass"] = (not det_mism) and cons["pass"]
        else:
            gate["pass"] = None if not det_mism else False
        res["gate"] = gate
        gate_fail = gate["pass"] is False

    def _ep_list(eps, counts):
        return [{"run": e["run"].replace("\\", "/"),
                 "episode_index": e["episode_index"], "seed": e["seed"],
                 "n_samples": int(c)} for e, c in zip(eps, counts)]

    sets = {"model": args.model,
            "fit": _ep_list(cal_eps, cal_counts),
            "eval": _ep_list(eval_eps, test_counts),
            "overlap_seeds": overlap,
            "transfer": {k: _ep_list(transfer_eps[k], transfer_counts[k])
                         for k in transfer_eps}}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(sets_out, "w", encoding="utf-8") as fh:
        json.dump(sets, fh)
    print(json.dumps(res["before"], indent=1))
    print(json.dumps(res["after"], indent=1))
    if args.expect:
        g = res["gate"]
        print("GATE deterministic:", "EXACT" if g["deterministic_exact"]
              else g["deterministic_mismatches"])
        if "mc_consistency" in g:
            c = g["mc_consistency"]
            print(f"GATE MC consistency: max|z| = {c['max_abs_z']} over "
                  f"{c['n_fields']} fields, {c['n_outside_replicate_range']} "
                  f"outside the {c['n_replicates']}-replicate range -> "
                  f"{'PASS' if c['pass'] else 'FAIL'}")
    print("saved ->", out)
    return 2 if gate_fail else 0


if __name__ == "__main__":
    sys.exit(main())
