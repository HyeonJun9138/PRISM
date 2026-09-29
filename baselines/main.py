"""PRISM baselines (doc 8.2) -- the comparison rows of the paper's tables.

Methods, all scored on EXACTLY the neural model's evaluation samples with
EXACTLY modelTraining's metric code:

  cv-obs   constant velocity from the last DETECTIONS (+fitted sigma)
  cv-gt    constant velocity from GT state at t0 -- information upper bound
  imm      IMM {CV, CT(EKF, w estimated), CA} over the observed track,
           emitting a 3-Gaussian mixture (NLL/coverage-comparable)
  pointnet the K=1 MSE-trained network (train via modelTraining CLI with
           configs/pointnet.json; evaluated there, sigma-wrapped here later)

Usage:
  python main.py fit       --train-run <dir> [--valid-run <dir>]   # sigma + IMM tuning
  python main.py evaluate  --run <dir> [--run ...] [--methods cv-obs,cv-gt,imm]
  python main.py all       --train-run ... --valid-run ... --run ...
  python main.py tune-imm  --valid-run <dir> [--track-rule current] [--imm-init-gap-fix]
                           [--workers 4]     # pre-round-1 tuning protocol, re-run
  python main.py selftest

Track rule (review round 1, R3 major 3; modelTraining/data/dataset.py doc):
  --job/--model  the neural model being compared: its data config supplies the
                 eligibility knobs and the default track rule (a legacy config
                 without the key = "slot"), so both rows score the same samples
  --track-rule   override: "current" (the stated tracking model) or "slot"
                 (oracle re-association, pre-round-1 behaviour). Default: the
                 compared model's rule, else "current".
  --index-rule   override eligibility only (data_cfg["index_track_rule"])
Parameters: rule "slot" reads the pre-round-1 fits in Logs/Models/baselines/
(read-only); every new fit -- and every report -- goes to
Logs/Models/baselines_r3/, rule-tagged, never overwriting an existing file.
Rule "current" needs `fit --track-rule current` first for the CV sigma (the
residuals of a different predictor); IMM keeps the pre-round-1 tuned noise
unless baselines_r3/imm_params_<rule>.json exists or --imm-file is given.
--imm-init-gap-fix: opt-in fix of the IMM velocity initialisation over
misses (methods/imm.py IMMTracker doc); outputs are named "imm-gapfix".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bridge  # noqa: E402,F401

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")

LEGACY_DIR = f"{PRISM_ROOT}/Logs/Models/baselines"       # pre-round-1, read-only
OUT_DIR = f"{PRISM_ROOT}/Logs/Models/baselines_r3"       # every new write
MODELS_ROOT = f"{PRISM_ROOT}/Logs/Models"
CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

TRACK_RULES = ("current", "slot")


def _free_path(path: str) -> str:
    """``path`` if unused, else a timestamped sibling -- never overwrite."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"


def _save(obj: dict, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    canonical = os.path.join(OUT_DIR, name)
    path = _free_path(canonical).replace("\\", "/")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    if os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(canonical)) \
            and name.startswith(("cv_fitted_sigma_", "imm_params_")):
        print(f"  note: {canonical} already exists and stays the default for "
              f"evaluate; pass --sigma-file/--imm-file {path} to use this fit")
    return path


def _model_path(args) -> str | None:
    if getattr(args, "model", None):
        return args.model
    if getattr(args, "job", None):
        for rel in (("final", "model_final.pt"), ("checkpoints", "best.pt")):
            cand = os.path.join(MODELS_ROOT, args.job, *rel)
            if os.path.exists(cand):
                return cand
        sys.exit(f"no model_final.pt / best.pt under job {args.job}")
    return None


def _data_cfg(args) -> tuple[dict, str | None]:
    from harness.data import resolve_data_cfg
    mp = _model_path(args)
    mcfg = None
    if mp:
        import torch
        ck = torch.load(mp, map_location="cpu", weights_only=False)
        mcfg = ck.get("data_cfg") or ck.get("cfg", {}).get("data")
        if mcfg is None:
            sys.exit(f"{mp} carries no data config")
    return resolve_data_cfg(mcfg, args.track_rule, args.index_rule), mp


def _param_file(kind: str, rule: str, override: str | None,
                gap_fix: bool = False) -> str:
    """Parameter file of a method under a rule (module doc). IMM with the
    init-gap fix looks for baselines_r3/imm_params_<rule>_gapfix.json first
    (written by `tune-imm --imm-init-gap-fix`)."""
    if override:
        return override
    names = ([f"{kind}_{rule}_gapfix.json"] if gap_fix else []) + [f"{kind}_{rule}.json"]
    for name in names:
        r3 = os.path.join(OUT_DIR, name)
        if os.path.exists(r3):
            return r3
    r3 = os.path.join(OUT_DIR, names[-1])
    legacy = os.path.join(LEGACY_DIR, f"{kind}.json")
    if rule == "slot" or kind == "imm_params":
        return legacy
    sys.exit(f"no {kind} fitted under track rule '{rule}': run "
             f"`python main.py fit --track-rule {rule} --train-run ...` first "
             f"(writes {r3})")


def cmd_fit(args) -> None:
    from harness.evaluate import fit_cv_sigma, tune_imm
    data_cfg, mp = _data_cfg(args)
    rule = data_cfg["track_rule"]
    print(f"track rule: {rule}  (eligibility: {data_cfg.get('index_track_rule', rule)})")

    print("fitting CV residual sigma (train split)...")
    t0 = time.time()
    f_obs = fit_cv_sigma(args.train_run, data_cfg, "obs")
    f_gt = fit_cv_sigma(args.train_run, data_cfg, "gt")
    path = _save({"obs": f_obs.to_dict(), "gt": f_gt.to_dict(),
                  "fitted_on": args.train_run, "data_cfg": data_cfg,
                  "compared_model": mp}, f"cv_fitted_sigma_{rule}.json")
    print(f"  done ({time.time()-t0:.0f}s)  "
          f"sigma_along@15s: obs={f_obs.sa[-1]:.1f}m gt={f_gt.sa[-1]:.1f}m -> {path}")

    if args.valid_run:
        print("tuning IMM process noise (valid split, coarse grid)...")
        t0 = time.time()
        params, hist = tune_imm(args.valid_run, data_cfg)
        path = _save({"params": params.to_dict(), "grid": hist,
                      "tuned_on": args.valid_run, "data_cfg": data_cfg},
                     f"imm_params_{rule}.json")
        best = min(hist, key=lambda h: h["NLL"])
        print(f"  done ({time.time()-t0:.0f}s)  best NLL={best['NLL']:.3f} "
              f"@ q={best['q']} q_w={best['q_w']} -> {path}")
    else:
        print("  no --valid-run: IMM keeps the pre-round-1 tuned noise "
              f"({LEGACY_DIR}/imm_params.json)")


def cmd_evaluate(args) -> None:
    from harness.evaluate import evaluate_cv, evaluate_imm
    from methods.cv import FittedSigma
    from methods.imm import IMMParams
    from model.metrics import code_fingerprint, file_sha256

    data_cfg, mp = _data_cfg(args)
    rule = data_cfg["track_rule"]
    from data.dataset import index_track_rule
    irule = index_track_rule(data_cfg)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tag = args.tag or "eval"
    fp = code_fingerprint()
    print(f"track rule: {rule}  eligibility: {irule}  "
          f"compared model: {mp or '-'}  code_fp {fp['sha256'][:12]}")
    if args.labels and not os.path.isdir(args.labels):
        sys.exit(f"labels dir not found: {args.labels}")

    for m in methods:
        print(f"[{m}] evaluating on {len(args.run)} run(s)...")
        t0 = time.time()
        # file-name tag of the method; the opt-in IMM fix gets its own name
        mname = m + ("-gapfix" if m == "imm" and args.imm_init_gap_fix else "")
        ps = None
        if args.per_sample:
            os.makedirs(args.per_sample, exist_ok=True)
            ps = _free_path(os.path.join(
                args.per_sample, f"persample_{mname}_{tag}_{rule}_idx{irule}.npz"))
        meta = {"method": mname, "track_rule": rule, "index_rule": irule,
                "run_dirs": args.run, "compared_model": mp, "code_fp": fp,
                "imm_init_gap_fix": bool(m == "imm" and args.imm_init_gap_fix)}
        if m in ("cv-obs", "cv-gt"):
            pfile = _param_file("cv_fitted_sigma", rule, args.sigma_file)
            with open(pfile, encoding="utf-8") as f:
                fitted = FittedSigma.from_dict(
                    json.load(f)["obs" if m == "cv-obs" else "gt"])
            rep = evaluate_cv(args.run, data_cfg, fitted, m.split("-")[1],
                              labels_dir=args.labels, per_sample_path=ps, meta=meta)
        elif m == "imm":
            pfile = _param_file("imm_params", rule, args.imm_file,
                                gap_fix=args.imm_init_gap_fix)
            with open(pfile, encoding="utf-8") as f:
                params = IMMParams.from_dict(json.load(f)["params"])
            rep = evaluate_imm(args.run, data_cfg, params,
                               labels_dir=args.labels, per_sample_path=ps, meta=meta,
                               init_gap_fix=args.imm_init_gap_fix)
        else:
            print(f"  unknown method '{m}', skipping")
            continue
        rep["wall_s"] = round(time.time() - t0, 1)
        rep.update({"params_file": pfile.replace("\\", "/"),
                    "params_sha256": file_sha256(pfile),
                    "data_cfg": data_cfg, "compared_model": mp,
                    "run_dirs": [str(r).replace("\\", "/") for r in args.run],
                    "code_fp": fp, "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        path = _save(rep, f"report_{mname}_{tag}_{rule}_idx{irule}.json")
        o = rep["overall"]
        print(f"  n={rep['n_samples']} (skipped {rep.get('n_skipped', 0)})  "
              f"minADE={o.get('minADE', float('nan')):.3f}m  "
              f"NLL={o.get('NLL', float('nan')):.4f}  "
              f"cov90@15s={o.get('cov90_15s', float('nan')):.4f}  "
              f"top1ADE={o.get('top1ADE', float('nan')):.3f}m  "
              f"ES={o.get('ES', float('nan')):.2f}m  -> {path}")
    return


def cmd_tune_imm(args) -> None:
    """Re-tune the IMM process noise with the verified pre-round-1 protocol
    (harness/evaluate.py TUNE_PASSES block) under the requested track rule and
    IMM init option; writes baselines_r3/imm_params_<rule>[_gapfix][_cap<N>].json."""
    from harness.data import iter_eval_groups
    from harness.evaluate import TUNE_MAX_SAMPLES, TUNE_PASSES, tune_imm_protocol
    from model.metrics import code_fingerprint, file_sha256
    data_cfg, mp = _data_cfg(args)
    rule = data_cfg["track_rule"]
    workers = max(1, min(int(args.workers), os.cpu_count() or 1))
    cap = int(args.max_samples)
    groups = list(iter_eval_groups(args.valid_run, data_cfg, cap))
    n_used = sum(len(v) for _, g in groups for v in g.values())
    print(f"track rule: {rule}  eligibility: {data_cfg.get('index_track_rule', rule)}  "
          f"gap fix: {bool(args.imm_init_gap_fix)}  cap: {cap} "
          f"({n_used} samples, {len(groups)} episodes)  workers: {workers}")
    t0 = time.time()
    params, hist, best = tune_imm_protocol(args.valid_run, data_cfg,
                                           init_gap_fix=args.imm_init_gap_fix,
                                           max_samples=cap, workers=workers)
    wall = round(time.time() - t0, 1)
    legacy_file = os.path.join(LEGACY_DIR, "imm_params.json")
    with open(legacy_file, encoding="utf-8") as f:
        legacy = json.load(f)
    name = args.name or (f"imm_params_{rule}{'_gapfix' if args.imm_init_gap_fix else ''}"
                         f"{'' if cap == TUNE_MAX_SAMPLES else f'_cap{cap}'}.json")
    out = {"params": params.to_dict(), "best": best, "grid": hist,
           "tuned_on": args.valid_run, "data_cfg": data_cfg, "compared_model": mp,
           "imm_init_gap_fix": bool(args.imm_init_gap_fix),
           "max_samples": cap, "n_samples_used": n_used, "n_episodes_used": len(groups),
           "tuning_episodes": [str(ep["_file"]) for ep, _ in groups],
           "protocol": {
               "objective": "mean per-step marginal mixture NLL ('NLL'), minimised",
               "family": "q_cv=0.6q, q_ct=q, q_ca=0.8q, q_w; p_stay=0.95, r_infl=1.0",
               "passes": [[t, list(q), list(w)] for t, q, w in TUNE_PASSES],
               "selection": "argmin over all evaluated points; 3x3 {x/3,x,3x} "
                            "re-centring while the argmin is on the grid edge",
               "source": "pre-round-1 protocol, verified bit-exact against "
                         "Logs/Models/baselines/imm_params.json (harness/evaluate.py)"},
           "legacy": {"file": legacy_file.replace("\\", "/"),
                      "sha256": file_sha256(legacy_file), "params": legacy["params"]},
           "workers": workers, "wall_s": wall, "code_fp": code_fingerprint(),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = _save(out, name)
    print(f"  done ({wall:.0f}s, {len(hist)} points)  best NLL={best['NLL']:.5f} "
          f"@ q={best['q']} q_w={best['q_w']} -> {path}")


def cmd_all(args) -> None:
    cmd_fit(args)
    for tag, runs in (("test_id", args.run), ):
        args.tag = tag
        cmd_evaluate(args)


def cmd_selftest(args) -> None:
    from selftests.test_baselines import run_all
    from selftests.test_track_rule import run_all as run_track_rule
    ok = run_all()
    ok = run_track_rule() and ok
    sys.exit(0 if ok else 1)


def _rule_args(p) -> None:
    p.add_argument("--job", default=None,
                   help="neural model job being compared (its data config and rule)")
    p.add_argument("--model", default=None, help="... or its checkpoint path")
    p.add_argument("--track-rule", choices=TRACK_RULES, default=None,
                   help="override the track rule (default: the compared "
                        "model's, else 'current')")
    p.add_argument("--index-rule", choices=TRACK_RULES, default=None,
                   help="override eligibility only")


def _eval_args(p) -> None:
    p.add_argument("--methods", default="cv-obs,cv-gt,imm")
    p.add_argument("--tag", default="")
    p.add_argument("--per-sample", default=None,
                   help="directory for per-sample .npz files (one per method)")
    p.add_argument("--labels", default=None,
                   help="avoidance labels dir (<labels>/<run name>/<episode>.npz)")
    p.add_argument("--sigma-file", default=None, help="CV sigma JSON override")
    p.add_argument("--imm-file", default=None, help="IMM params JSON override")
    p.add_argument("--imm-init-gap-fix", action="store_true",
                   help="opt-in fix of the IMM two-point velocity init over "
                        "misses (methods/imm.py IMMTracker doc); off = the "
                        "published IMM rows")


def main() -> None:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit CV sigma + tune IMM")
    f.add_argument("--train-run", action="append", required=True)
    f.add_argument("--valid-run", action="append", default=None)
    _rule_args(f)

    e = sub.add_parser("evaluate", help="score methods on eval runs")
    e.add_argument("--run", action="append", required=True)
    _rule_args(e)
    _eval_args(e)

    a = sub.add_parser("all", help="fit then evaluate")
    a.add_argument("--train-run", action="append", required=True)
    a.add_argument("--valid-run", action="append", default=None)
    a.add_argument("--run", action="append", required=True)
    _rule_args(a)
    _eval_args(a)

    ti = sub.add_parser("tune-imm", help="re-tune IMM noise (pre-round-1 protocol)")
    ti.add_argument("--valid-run", action="append", required=True)
    _rule_args(ti)
    ti.add_argument("--imm-init-gap-fix", action="store_true")
    ti.add_argument("--max-samples", type=int, default=1200,
                    help="tuning sample cap (pre-round-1 protocol: 1200)")
    ti.add_argument("--workers", type=int, default=1,
                    help="processes scoring grid points in parallel")
    ti.add_argument("--name", default=None, help="output file name override")

    sub.add_parser("selftest", help="run unit tests")

    args = ap.parse_args()
    {"fit": cmd_fit, "evaluate": cmd_evaluate, "all": cmd_all,
     "tune-imm": cmd_tune_imm, "selftest": cmd_selftest}[args.command](args)


if __name__ == "__main__":
    main()
