"""Post-training evaluation: paper-grade numbers for one trained model.

Produces eval_report.json with:
  * overall minADE/minFDE/NLL/type accuracy
  * reliability table: nominal vs empirical coverage (50/70/90 x 5/10/15 s)
  * stratified breakdown by platform type and by intent   (doc 8.1 층화 분석)
  * inference latency (batch 1 / 16)                       (doc 8.4)

Review round 1 (R3 major 3 / 7, minor 1) -- all ADDED, no legacy value moves:
  * top-1 / mixture-mean ADE/FDE, trajectory NLL, energy score, branch
    entropy / separation / distinct count (model/metrics.py module doc)
  * "strata": every metric, legacy and new, pooled per stratum: all, track
    history class (meta_reacq), type, intent and -- with ``labels_dir`` -- the
    avoidance-window label of another pipeline
  * provenance: index (eligibility) rule, feature rule, model path, run dirs,
    n_samples, code fingerprint
  * optional per-sample file keyed by (episode path, t0, agent) for paired
    comparisons (model.metrics.join_per_sample)

Reproducibility: the deterministic legacy scalars (minADE, minFDE, NLL,
type_acc, pi/use, mode_entropy) reproduce pre-round-1 reports bit-for-bit.
The MC coverage of those reports drew from the unseeded global RNG and so
carries run-to-run MC noise (model/metrics.py module doc); evaluations now
draw coverage and energy-score samples from two dedicated generators seeded
MC_SEED / ES_SEED, so a report reproduces exactly for a given index, batch
size (256) and device.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from common.constants import (COVERAGE_LEVELS, COVERAGE_STEPS, INTENT_NAMES,
                              N_TYPES, TYPE_NAMES)
from common.utils import atomic_write_json
from data.dataset import (TRACK_RULES, build_sample_index, index_track_rule,
                          track_rule)
from data.loader import loader_with_fallback, to_device
from model.metrics import (ES_SEED, MC_SEED, METRIC_DEFINITIONS,
                           MetricAccumulator, SampleTable, batch_metrics,
                           code_fingerprint, extended_metrics, file_sha256,
                           load_avoid_labels, make_generator, overall_extended,
                           save_per_sample, standard_strata)
from model.network import build_model
from training.job import JobState
from training.trainer import _measure_latency


def _free_path(path: str) -> str:
    """``path`` if nothing is there yet, else a timestamped sibling: an
    evaluation never overwrites an existing result file."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"


def run_evaluation(job: JobState, model_path: str, run_dirs: list[str],
                   batch_size: int = 256, index_rule: str | None = None,
                   out_path: str | None = None,
                   per_sample_path: str | None = None,
                   labels_dir: str | None = None) -> None:
    """Evaluate one checkpoint on ``run_dirs``; write eval_report.json.

    Args:
        index_rule: None keeps the model's own eligibility rule; "current" or
            "slot" overrides data_cfg["index_track_rule"] ONLY -- which
            samples are scored. The feature rule the model was trained with
            (track_rule(data_cfg); legacy configs = "slot") is never changed.
        out_path: also write the report here (never overwrites; a timestamped
            sibling is used if the file exists).
        per_sample_path: write per-sample arrays (.npz) keyed by
            (episode path, t0, agent).
        labels_dir: directory of avoidance labels (<labels_dir>/<run name>/
            <episode>.npz with reaction_active [T,N]); adds the "avoid" stratum.
    """
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    job.set_status("indexing")
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") or ck.get("cfg", {}).get("model")
    data_cfg = ck.get("data_cfg") or ck.get("cfg", {}).get("data")
    if model_cfg is None or data_cfg is None:
        raise RuntimeError(f"{model_path} carries no config - not a trainer artifact")
    data_cfg = dict(data_cfg)
    if index_rule is not None:
        if index_rule not in TRACK_RULES:
            raise ValueError(f"index_rule must be one of {TRACK_RULES}, got {index_rule!r}")
        data_cfg["index_track_rule"] = index_rule
    feat_rule, elig_rule = track_rule(data_cfg), index_track_rule(data_cfg)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    job.log(f"loaded {model_path} ({sum(p.numel() for p in model.parameters()):,} params)")
    job.log(f"track rules: features={feat_rule} "
            f"({'from config' if 'track_rule' in data_cfg else 'legacy config, no key'}), "
            f"eligibility={elig_rule}"
            f"{' (--index-rule override)' if index_rule is not None else ''}")

    idx = build_sample_index(run_dirs, data_cfg,
                             lambda d, t, n: job.set_progress(pct=100.0 * d / max(1, t)))
    job.log(f"eval index: {len(idx)} samples from {len(idx.files)} episodes")
    avoid, avoid_stats = None, None
    if labels_dir:
        avoid, avoid_stats = load_avoid_labels(idx.files, idx.file_idx, idx.t0,
                                               idx.agent, labels_dir)
        job.log(f"avoidance labels: {avoid_stats['n_samples_active']} active / "
                f"{avoid_stats['n_samples_inactive']} inactive / "
                f"{avoid_stats['n_samples_unknown']} unknown samples "
                f"({avoid_stats['n_episodes_missing']}/{avoid_stats['n_episodes']} "
                f"episodes without a label file)")
    loader = loader_with_fallback(idx, data_cfg, batch_size, False, 2, job.log)

    job.set_status("evaluating")
    acc = MetricAccumulator()
    table = SampleTable()
    mc_gen = make_generator(device, MC_SEED)    # coverage draws (reproducible)
    es_gen = make_generator(device, ES_SEED)    # energy-score draws
    intents: list[torch.Tensor] = []
    ades: list[torch.Tensor] = []
    n_batches = max(1, len(loader))
    offset = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if job.stop_requested:
                break
            batch = to_device(batch, device)
            out = model(batch["x"], batch["agent_mask"])
            m = batch_metrics(out, batch, mc_samples=128, generator=mc_gen)
            ext = extended_metrics(out, batch, es_gen)
            nb = int(m["n"])
            # per-sample rows must line up with the index (loader not shuffled)
            if not np.array_equal(batch["meta_t0"].cpu().numpy(),
                                  idx.t0[offset:offset + nb]):
                raise RuntimeError("loader order diverged from the sample index")
            offset += nb
            table.add(m["_per_sample"], ext)
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

    cols = table.finalize() if len(table) else {}
    overall = {k: v for k, v in res.items()
               if isinstance(v, (int, float)) and not k.startswith("_")}
    if cols:
        overall.update(overall_extended(cols))       # new keys only
    avoid_eval = None if avoid is None else avoid[:offset]
    report = {
        "model": model_path.replace("\\", "/"),
        "eval_runs": run_dirs,
        "n_samples": acc.n,
        "overall": overall,
        "reliability": reliability,
        "by_type": {TYPE_NAMES[t] if 0 <= t < N_TYPES else str(t): v
                    for t, v in by_type.items()},
        "by_intent": by_intent,
        "latency_ms": _measure_latency(model, device),
        # ------------------------------------------ review round 1 additions
        "index_rule": elig_rule,
        "feature_rule": feat_rule,
        "index_rule_override": index_rule,
        "data_cfg": data_cfg,
        "run_dirs": [str(r).replace("\\", "/") for r in run_dirs],
        "model_sha256": file_sha256(model_path),
        "n_index": len(idx),
        "complete": offset == len(idx),
        "code_fp": code_fingerprint(),
        "mc_seed": MC_SEED, "es_seed": ES_SEED, "batch_size": batch_size,
        "strata": standard_strata(cols, avoid_eval) if cols else {},
        "avoid_labels": avoid_stats,
        "metric_definitions": METRIC_DEFINITIONS,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t_start, 1),
        "per_sample_path": None,
    }
    if per_sample_path and cols:
        ps = _free_path(per_sample_path)
        meta = {k: report[k] for k in ("model", "model_sha256", "run_dirs",
                                       "index_rule", "feature_rule", "n_samples",
                                       "code_fp", "created")}
        meta["eval_job"] = job.id
        extra = {} if avoid_eval is None else {"avoid": avoid_eval}
        save_per_sample(ps, idx.files, idx.file_idx[:offset], idx.t0[:offset],
                        idx.agent[:offset], cols, meta, extra)
        report["per_sample_path"] = ps.replace("\\", "/")
        job.log(f"per-sample arrays written: {ps}")

    out_file = os.path.join(job.dir, "eval_report.json")
    atomic_write_json(out_file, report)
    if out_path:
        extra_out = _free_path(out_path)
        d = os.path.dirname(extra_out)
        if d:
            os.makedirs(d, exist_ok=True)
        atomic_write_json(extra_out, report)
        job.log(f"eval report also written: {extra_out}")
    job.summary = {"eval_report": out_file.replace("\\", "/"),
                   "minADE": report["overall"].get("minADE"),
                   "NLL": report["overall"].get("NLL")}
    job.set_status("stopped" if job.stop_requested else "finished", pct=100.0)
    job.log(f"eval report written: {out_file}")
