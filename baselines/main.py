"""PRISM baselines (doc 8.2) -- the comparison rows of the paper's tables.

Methods, all scored on EXACTLY the neural model's evaluation samples with
EXACTLY modelTraining's metric code:

  cv-obs   constant velocity from the last two DETECTIONS (+fitted sigma)
  cv-gt    constant velocity from GT state at t0 -- information upper bound
  imm      IMM {CV, CT(EKF, w estimated), CA} over the observed track,
           emitting a 3-Gaussian mixture (NLL/coverage-comparable)
  pointnet the K=1 MSE-trained network (train via modelTraining CLI with
           configs/pointnet.json; evaluated there, sigma-wrapped here later)

Usage:
  python main.py fit       --train-run <dir> [--valid-run <dir>]   # sigma + IMM tuning
  python main.py evaluate  --run <dir> [--run ...] [--methods cv-obs,cv-gt,imm]
  python main.py all       --train-run ... --valid-run ... --run ...
  python main.py selftest
Reports land in $PRISM_ROOT/Logs/Models/baselines/ as JSON, one per (method, run-set).
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

OUT_DIR = f"{PRISM_ROOT}/Logs/Models/baselines"
CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

#: the neural model's data config -- keep in lockstep with the trained model
DATA_CFG = {"max_agents": 10, "t0_stride": 4, "min_obs": 3,
            "min_future_valid": 27}


def _save(obj: dict, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name).replace("\\", "/")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def cmd_fit(args) -> None:
    from harness.evaluate import fit_cv_sigma, tune_imm
    from methods.imm import IMMParams

    print("fitting CV residual sigma (train split)...")
    t0 = time.time()
    f_obs = fit_cv_sigma(args.train_run, DATA_CFG, "obs")
    f_gt = fit_cv_sigma(args.train_run, DATA_CFG, "gt")
    _save({"obs": f_obs.to_dict(), "gt": f_gt.to_dict(),
           "fitted_on": args.train_run}, "cv_fitted_sigma.json")
    print(f"  done ({time.time()-t0:.0f}s)  "
          f"sigma_along@15s: obs={f_obs.sa[-1]:.1f}m gt={f_gt.sa[-1]:.1f}m")

    if args.valid_run:
        print("tuning IMM process noise (valid split, coarse grid)...")
        t0 = time.time()
        params, hist = tune_imm(args.valid_run, DATA_CFG)
        _save({"params": params.to_dict(), "grid": hist,
               "tuned_on": args.valid_run}, "imm_params.json")
        best = min(hist, key=lambda h: h["NLL"])
        print(f"  done ({time.time()-t0:.0f}s)  best NLL={best['NLL']:.3f} "
              f"@ q={best['q']} q_w={best['q_w']}")
    else:
        _save({"params": IMMParams().to_dict(), "grid": [],
               "tuned_on": "defaults"}, "imm_params.json")
        print("  no --valid-run: IMM keeps default noise (retune later)")


def cmd_evaluate(args) -> None:
    from harness.evaluate import evaluate_cv, evaluate_imm
    from methods.cv import FittedSigma
    from methods.imm import IMMParams

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sig_path = os.path.join(OUT_DIR, "cv_fitted_sigma.json")
    imm_path = os.path.join(OUT_DIR, "imm_params.json")
    tag = args.tag or "eval"
    reports = []

    for m in methods:
        print(f"[{m}] evaluating on {len(args.run)} run(s)...")
        t0 = time.time()
        if m in ("cv-obs", "cv-gt"):
            with open(sig_path, encoding="utf-8") as f:
                fitted = FittedSigma.from_dict(
                    json.load(f)["obs" if m == "cv-obs" else "gt"])
            rep = evaluate_cv(args.run, DATA_CFG, fitted, m.split("-")[1])
        elif m == "imm":
            with open(imm_path, encoding="utf-8") as f:
                params = IMMParams.from_dict(json.load(f)["params"])
            rep = evaluate_imm(args.run, DATA_CFG, params)
        else:
            print(f"  unknown method '{m}', skipping")
            continue
        rep["wall_s"] = round(time.time() - t0, 1)
        path = _save(rep, f"report_{m}_{tag}.json")
        o = rep["overall"]
        print(f"  minADE={o.get('minADE', float('nan')):.1f}m  "
              f"NLL={o.get('NLL', float('nan')):.3f}  "
              f"cov90@15s={o.get('cov90_15s', float('nan')):.3f}  "
              f"-> {path}")
        reports.append(rep)
    return


def cmd_all(args) -> None:
    cmd_fit(args)
    for tag, runs in (("test_id", args.run), ):
        args.tag = tag
        cmd_evaluate(args)


def cmd_selftest(args) -> None:
    from selftests.test_baselines import run_all
    sys.exit(0 if run_all() else 1)


def main() -> None:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit CV sigma + tune IMM")
    f.add_argument("--train-run", action="append", required=True)
    f.add_argument("--valid-run", action="append", default=None)

    e = sub.add_parser("evaluate", help="score methods on eval runs")
    e.add_argument("--run", action="append", required=True)
    e.add_argument("--methods", default="cv-obs,cv-gt,imm")
    e.add_argument("--tag", default="")

    a = sub.add_parser("all", help="fit then evaluate")
    a.add_argument("--train-run", action="append", required=True)
    a.add_argument("--valid-run", action="append", default=None)
    a.add_argument("--run", action="append", required=True)
    a.add_argument("--methods", default="cv-obs,cv-gt,imm")
    a.add_argument("--tag", default="")

    sub.add_parser("selftest", help="run unit tests")

    args = ap.parse_args()
    {"fit": cmd_fit, "evaluate": cmd_evaluate, "all": cmd_all,
     "selftest": cmd_selftest}[args.command](args)


if __name__ == "__main__":
    main()

