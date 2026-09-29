"""PRISM prediction-model training system - the only top-level entry point.

Subsystem map (which part of 논문관련.txt each package implements):
  data/       [5.1] sample construction: target-centric frame, observation-only
              features, GT futures as labels; episode-level train/val split [8.1]
  model/      [5.1][5.2] K-mode mixture-Gaussian network + type aux head;
              losses (WTA + mixture NLL ramp); metrics [9.1][9.2][9.3]
  training/   the training engine: CUDA/AMP, warmup-cosine, checkpoints,
              graceful stop -> packaged final model + model card [8.4]
  analysis/   post-hoc evaluation report (stratified, reliability, latency)
  gui/        HTML GUI (stdlib server, vanilla JS): register data runs, tune
              hyperparameters, live charts/logs, checkpoints, stop/resume

Usage:
  python main.py                    # launch the GUI (default)
  python main.py gui [--port 8790] [--no-browser]
  python main.py train --train-run $PRISM_ROOT/Logs/<run> [--val-run ...]
                       [--epochs 60] [--seed 0] [--tag x] [--config file.json]
  python main.py resume --job <job_id> [--checkpoint last.pt]
  python main.py evaluate --job <job_id> --run $PRISM_ROOT/Logs/<run> [<run> ...]
                       [--index-rule current|slot] [--out report.json]
                       [--per-sample samples.npz] [--labels <labels dir>]
  python main.py demo               # synthetic end-to-end smoke run (no real data needed)
  python main.py selftest           # unit + integration tests
  python main.py list-jobs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.constants import LOGS_ROOT, MODELS_ROOT           # noqa: E402
from common.utils import deep_update, read_json               # noqa: E402


def _load_config(path: str | None, overrides: dict) -> dict:
    base = read_json(os.path.join(os.path.dirname(__file__), "config", "defaults.json"))
    if path:
        base = deep_update(base, read_json(path))
    return deep_update(base, overrides)


def _run_blocking(job, fn, *args) -> None:
    """Run a job function on the queue worker and tail its log to stdout."""
    from training.job import REGISTRY, TERMINAL
    REGISTRY.submit(job, fn, *args)
    cursor = 0
    while True:
        lines, cursor = job.logs_after(cursor)
        for line in lines:
            print(line)
        if job.status in TERMINAL:
            break
        time.sleep(0.5)
    print(f"[{job.id}] -> {job.status}")
    if job.error:
        print(job.error)
        sys.exit(1)


# ------------------------------------------------------------------ commands

def cmd_gui(args) -> None:
    from gui.server import serve
    serve(open_browser=not args.no_browser, port=args.port)


def cmd_train(args) -> None:
    from training.job import REGISTRY
    from training.trainer import run_training
    overrides: dict = {}
    if args.epochs is not None:
        overrides.setdefault("optim", {})["epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.tag:
        overrides["tag"] = args.tag
    cfg = _load_config(args.config, overrides)
    roster = {"train": args.train_run, "val": args.val_run or []}
    job = REGISTRY.create_job(cfg.get("tag", "run"))
    print(f"job: {job.id}\ndir: {job.dir}")
    _run_blocking(job, run_training, cfg, roster, None)


def cmd_resume(args) -> None:
    from training.job import REGISTRY
    from training.trainer import run_training
    src = os.path.join(MODELS_ROOT, args.job)
    saved = read_json(os.path.join(src, "config.json"))
    ck = os.path.join(src, "checkpoints", args.checkpoint)
    if not os.path.exists(ck):
        sys.exit(f"checkpoint not found: {ck}")
    job = REGISTRY.create_job(f"resume_{args.job[-6:]}")
    print(f"job: {job.id}")
    _run_blocking(job, run_training, saved["config"], saved["roster"], ck)


def cmd_evaluate(args) -> None:
    import functools
    from analysis.evaluate import run_evaluation
    from training.job import REGISTRY
    model = args.model
    if not model and args.job:
        cand = os.path.join(MODELS_ROOT, args.job, "final", "model_final.pt")
        if not os.path.exists(cand):
            cand = os.path.join(MODELS_ROOT, args.job, "checkpoints", "best.pt")
        model = cand
    if not model or not os.path.exists(model):
        sys.exit(f"model not found: {model}")
    if args.labels and not os.path.isdir(args.labels):
        sys.exit(f"labels dir not found: {args.labels}")
    job = REGISTRY.create_job("eval", kind="eval")
    fn = functools.partial(run_evaluation, index_rule=args.index_rule,
                           out_path=args.out, per_sample_path=args.per_sample,
                           labels_dir=args.labels)
    _run_blocking(job, fn, model, args.run)


def cmd_demo(args) -> None:
    """Synthetic end-to-end run: proves the whole pipeline without real data."""
    import tempfile
    from data.synthetic import make_synthetic_run
    from training.job import REGISTRY
    from training.trainer import run_training
    tmp = tempfile.mkdtemp(prefix="prism_demo_")
    run_dir = make_synthetic_run(os.path.join(tmp, "synth_run"), n_episodes=8)
    print(f"synthetic data: {run_dir}")
    cfg = _load_config(None, {
        "tag": "demo", "data": {"t0_stride": 8},
        "optim": {"epochs": args.epochs, "batch_size": 64, "num_workers": 0,
                  "warmup_steps": 20},
        "checkpoint": {"every_epochs": 2},
    })
    job = REGISTRY.create_job("demo")
    print(f"job: {job.id}")
    _run_blocking(job, run_training, cfg, {"train": [run_dir], "val": []}, None)
    card = read_json(os.path.join(job.dir, "final", "model_card.json"))
    print(json.dumps({k: card[k] for k in ("final_model_path", "parameters",
                                           "best", "latency_ms")}, indent=2))


def cmd_validate_data(args) -> None:
    """Prove a dataGen run is consumable end-to-end: index it, build a real
    batch, run the model forward. Prints ONE JSON line (machine-readable;
    dataGen's `dataset` pipeline parses it for its training-ready stamp)."""
    import numpy as np
    import torch
    from data.dataset import build_sample_index
    from data.loader import make_loader
    from model.network import build_model

    cfg = _load_config(None, {})
    result: dict = {"ok": False}
    try:
        idx = build_sample_index(args.run, cfg["data"])
        dl = make_loader(idx, cfg["data"], batch_size=min(64, max(2, len(idx))),
                         shuffle=True, num_workers=0)
        batch = next(iter(dl))
        for key in ("x", "agent_mask", "y", "y_valid", "type_label"):
            if key not in batch:
                raise KeyError(f"batch missing '{key}'")
            if torch.isnan(batch[key].float()).any():
                raise ValueError(f"NaN in batch['{key}']")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg["model"]).to(device)
        with torch.no_grad():
            out = model(batch["x"].to(device), batch["agent_mask"].to(device))
        result = {
            "ok": True,
            "n_samples": int(len(idx)),
            "n_episodes": int(len(idx.files)),
            "batch_shape": list(batch["x"].shape),
            "output_traj_shape": list(out["mu"].shape),
            "types_seen": sorted(set(batch["type_label"].tolist())),
            "device": str(device),
        }
    except Exception as e:  # noqa: BLE001 - the caller wants a verdict, not a trace
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["ok"] else 1)


def cmd_selftest(args) -> None:
    from tests.test_pipeline import run_all
    ok = run_all(verbose=True)
    ok = __import__("tests.test_r3_metrics", fromlist=["run_all"]).run_all(verbose=True) and ok
    sys.exit(0 if ok else 1)


def cmd_list_jobs(args) -> None:
    from training.job import REGISTRY
    for snap in REGISTRY.list_jobs():
        p = snap.get("progress", {})
        print(f"{snap.get('id','?'):40s} {snap.get('status','?'):10s} "
              f"epoch {p.get('epoch','-')}/{p.get('epochs','-')}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    g = sub.add_parser("gui", help="launch the training GUI (default)")
    g.add_argument("--port", type=int, default=None)
    g.add_argument("--no-browser", action="store_true")

    t = sub.add_parser("train", help="headless training (for the runner PCs)")
    t.add_argument("--train-run", action="append", required=True,
                   help="dataset run dir (repeatable)")
    t.add_argument("--val-run", action="append", default=None)
    t.add_argument("--config", default=None, help="JSON config overriding defaults")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--tag", default=None)

    r = sub.add_parser("resume", help="resume a job from its checkpoint")
    r.add_argument("--job", required=True)
    r.add_argument("--checkpoint", default="last.pt")

    e = sub.add_parser("evaluate", help="write eval_report.json for a model")
    e.add_argument("--job", default=None)
    e.add_argument("--model", default=None)
    e.add_argument("--run", nargs="+", action="extend", required=True,
                   help="dataset run dir(s); repeatable")
    e.add_argument("--index-rule", choices=("current", "slot"), default=None,
                   help="override sample ELIGIBILITY only (data_cfg "
                        "index_track_rule); the model's feature rule is kept")
    e.add_argument("--out", default=None,
                   help="also write the report here (never overwrites)")
    e.add_argument("--per-sample", default=None,
                   help="write per-sample arrays (.npz) keyed by (file, t0, agent)")
    e.add_argument("--labels", default=None,
                   help="avoidance labels dir: <labels>/<run name>/<episode>.npz "
                        "with reaction_active [T,N]")

    d = sub.add_parser("demo", help="synthetic end-to-end smoke run")
    d.add_argument("--epochs", type=int, default=3)

    vd = sub.add_parser("validate-data",
                        help="prove a dataGen run feeds the trainer (JSON out)")
    vd.add_argument("--run", action="append", required=True,
                    help="dataset run dir (repeatable)")

    sub.add_parser("selftest", help="run the test suite")
    sub.add_parser("list-jobs", help="list all jobs under $PRISM_ROOT/Models")

    args = ap.parse_args()
    cmd = args.command or "gui"
    if cmd == "gui" and args.command is None:
        args.port = None
        args.no_browser = False
    {"gui": cmd_gui, "train": cmd_train, "resume": cmd_resume,
     "evaluate": cmd_evaluate, "demo": cmd_demo, "selftest": cmd_selftest,
     "validate-data": cmd_validate_data,
     "list-jobs": cmd_list_jobs}[cmd](args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows DataLoader workers
    main()
