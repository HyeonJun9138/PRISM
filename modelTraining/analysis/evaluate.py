"""Post-training evaluation: paper-grade numbers for one trained model.

Produces eval_report.json with:
  * overall minADE/minFDE/NLL/type accuracy
  * reliability table: nominal vs empirical coverage (50/70/90 x 5/10/15 s)
  * stratified breakdown by platform type and by intent   (doc 8.1 층화 분석)
  * inference latency (batch 1 / 16)                       (doc 8.4)
"""
from __future__ import annotations

import os

import torch

from common.constants import (COVERAGE_LEVELS, COVERAGE_STEPS, INTENT_NAMES,
                              N_TYPES, TYPE_NAMES)
from common.utils import atomic_write_json
from data.dataset import build_sample_index
from data.loader import loader_with_fallback, to_device
from model.metrics import MetricAccumulator, batch_metrics
from model.network import build_model
from training.job import JobState
from training.trainer import _measure_latency


def run_evaluation(job: JobState, model_path: str, run_dirs: list[str],
                   batch_size: int = 256) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    job.set_status("indexing")
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") or ck.get("cfg", {}).get("model")
    data_cfg = ck.get("data_cfg") or ck.get("cfg", {}).get("data")
    if model_cfg is None or data_cfg is None:
        raise RuntimeError(f"{model_path} carries no config - not a trainer artifact")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    job.log(f"loaded {model_path} ({sum(p.numel() for p in model.parameters()):,} params)")

    idx = build_sample_index(run_dirs, data_cfg,
                             lambda d, t, n: job.set_progress(pct=100.0 * d / max(1, t)))
    job.log(f"eval index: {len(idx)} samples from {len(idx.files)} episodes")
    loader = loader_with_fallback(idx, data_cfg, batch_size, False, 2, job.log)

    job.set_status("evaluating")
    acc = MetricAccumulator()
    intents: list[torch.Tensor] = []
    ades: list[torch.Tensor] = []
    n_batches = max(1, len(loader))
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if job.stop_requested:
                break
            batch = to_device(batch, device)
            out = model(batch["x"], batch["agent_mask"])
            m = batch_metrics(out, batch, mc_samples=128)
            intents.append(m["_per_sample"]["intent"])
            ades.append(m["_per_sample"]["minADE"])
            acc.update(m)
            if bi % 10 == 0:
                job.set_progress(pct=100.0 * bi / n_batches)

    res = acc.result()
    by_type = res.pop("by_type", {})
    reliability = [{"nominal": p, "horizon_s": (s + 1) // 2,
                    "empirical": res.get(f"cov{int(p*100)}_{(s+1)//2}s")}
                   for p in COVERAGE_LEVELS for s in COVERAGE_STEPS]

    by_intent = {}
    if intents:
        lab = torch.cat(intents); ade = torch.cat(ades)
        for i in lab.unique().tolist():
            if i < 0:
                continue
            sel = lab == i
            name = INTENT_NAMES[i] if 0 <= i < len(INTENT_NAMES) else str(i)
            by_intent[name] = {"minADE": float(ade[sel].mean()), "n": int(sel.sum())}

    report = {
        "model": model_path.replace("\\", "/"),
        "eval_runs": run_dirs,
        "n_samples": acc.n,
        "overall": {k: v for k, v in res.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")},
        "reliability": reliability,
        "by_type": {TYPE_NAMES[t] if 0 <= t < N_TYPES else str(t): v
                    for t, v in by_type.items()},
        "by_intent": by_intent,
        "latency_ms": _measure_latency(model, device),
    }
    out_path = os.path.join(job.dir, "eval_report.json")
    atomic_write_json(out_path, report)
    job.summary = {"eval_report": out_path.replace("\\", "/"),
                   "minADE": report["overall"].get("minADE"),
                   "NLL": report["overall"].get("NLL")}
    job.set_status("stopped" if job.stop_requested else "finished", pct=100.0)
    job.log(f"eval report written: {out_path}")
