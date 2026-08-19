"""The training engine: CUDA + AMP(bf16), warmup-cosine schedule, graceful
stop, checkpoint family (periodic / best / last) and a final packaged model.

Stop semantics (user requirement):
  * graceful stop -> finish the current step, run a final validation, save
    last.pt, package final/model_final.pt from the BEST checkpoint so far
    (falling back to last), write model_card.json, status='stopped'.
  * force stop    -> break out immediately, still save last.pt and package,
    skip the final validation.

Everything worth putting in a paper is persisted: metrics.jsonl (every point),
train.log, config.json, model_card.json (environment, versions, data roster,
parameter count, latency measurement, best metrics, seeds).
"""
from __future__ import annotations

import math
import os
import platform
import time

import numpy as np
import torch

from common.constants import MODELS_ROOT, N_TYPES, TYPE_NAMES
from common.utils import EMA, atomic_write_json, deep_update, read_json, set_seed
from data.dataset import SampleIndex, build_sample_index, split_train_val
from data.loader import loader_with_fallback, to_device
from model.losses import compute_losses, mixture_weight_at
from model.metrics import MetricAccumulator, batch_metrics
from model.network import build_model
from training.job import JobState

LOG_EVERY_STEPS = 20


# ================================================================ public API

def run_training(job: JobState, config: dict, roster: dict,
                 resume_from: str | None = None) -> None:
    """Entry point executed on the job worker thread.

    roster = {"train": [run_dir,...], "val": [run_dir,...]}   (val may be [])
    """
    cfg = config
    atomic_write_json(os.path.join(job.dir, "config.json"),
                      {"config": cfg, "roster": roster, "resume_from": resume_from})
    set_seed(int(cfg.get("seed", 0)))
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    job.log(f"device: {device} "
            f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")

    # ------------------------------------------------------------- indexing
    job.set_status("indexing")
    t_idx = time.time()

    def idx_progress(done, total, n_samples):
        job.set_progress(pct=100.0 * done / max(1, total),
                         phase_detail=f"indexing {done}/{total} episodes, {n_samples} samples")

    train_idx = build_sample_index(roster["train"], cfg["data"], idx_progress)
    if roster.get("val"):
        val_idx = build_sample_index(roster["val"], cfg["data"])
        job.log(f"index: train={len(train_idx)} samples from {len(train_idx.files)} eps; "
                f"val={len(val_idx)} samples from {len(val_idx.files)} eps (separate runs)")
    else:
        train_idx, val_idx = split_train_val(
            train_idx, float(cfg["data"].get("val_fraction", 0.1)),
            int(cfg.get("seed", 0)))
        job.log(f"index: split by episode -> train={len(train_idx)} / val={len(val_idx)} samples")
    job.log(f"indexing took {time.time() - t_idx:.1f}s")
    if len(train_idx) == 0:
        raise RuntimeError("no eligible training samples - check data roster / eligibility settings")
    # optional deterministic val subsample: on large datasets a full val pass
    # every epoch costs minutes; a fixed 30-50k subset tracks the same curves.
    val_cap = int(cfg["optim"].get("val_max_samples", 0))
    if val_cap and len(val_idx) > val_cap:
        rng = np.random.default_rng(int(cfg.get("seed", 0)) + 7)
        pick = np.sort(rng.choice(len(val_idx), size=val_cap, replace=False))
        mask = np.zeros(len(val_idx), dtype=bool)
        mask[pick] = True
        val_idx = val_idx.subset(mask)
        job.log(f"val subsampled to {len(val_idx)} samples (val_max_samples)")

    train_idx.save(os.path.join(job.dir, "index_train.npz"))
    val_idx.save(os.path.join(job.dir, "index_val.npz"))

    # --------------------------------------------------------------- setup
    opt_cfg = cfg["optim"]
    train_loader = loader_with_fallback(
        train_idx, cfg["data"], int(opt_cfg["batch_size"]), True,
        int(opt_cfg.get("num_workers", 4)), job.log, int(cfg.get("seed", 0)),
        int(cfg["data"].get("max_samples_per_epoch", 0)))
    val_loader = loader_with_fallback(
        val_idx, cfg["data"], int(opt_cfg["batch_size"]), False,
        min(2, int(opt_cfg.get("num_workers", 4))), job.log) if len(val_idx) else None

    model = build_model(cfg["model"]).to(device)
    n_params = model.count_parameters()
    job.log(f"model: d={cfg['model'].get('d_model')} K={cfg['model'].get('n_modes')} "
            f"params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]),
                            weight_decay=float(opt_cfg["weight_decay"]))
    epochs = int(opt_cfg["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = int(opt_cfg.get("warmup_steps", 1000))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    amp_mode = str(opt_cfg.get("amp", "bf16")).lower()
    use_amp = device.type == "cuda" and amp_mode in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    job.log(f"amp: {'off' if not use_amp else amp_mode}; "
            f"steps/epoch={steps_per_epoch}, total={total_steps}")

    start_epoch = 0
    global_step = 0
    best = {"value": float("inf"), "epoch": -1, "path": None}
    if resume_from:
        start_epoch, global_step, best = _load_checkpoint(
            resume_from, model, opt, sched, scaler, device, job)

    # ---------------------------------------------------------- train loop
    job.set_status("running", epochs=epochs, steps_per_epoch=steps_per_epoch)
    step_ema = EMA(0.05)
    stopped = False
    ckpt_cfg = cfg["checkpoint"]
    best_key = str(ckpt_cfg.get("best_metric", "val/minADE")).split("/")[-1]
    patience = int(opt_cfg.get("early_stop_patience", 0))
    since_best = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        w_mix = mixture_weight_at(epoch, epochs, cfg["loss"])
        t_step = time.time()
        for it, batch in enumerate(train_loader):
            if job.stop_requested:
                stopped = True
                break
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(batch["x"], batch["agent_mask"])
            loss, comps = compute_losses(out, batch, cfg["loss"], w_mix)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"]))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"]))
                opt.step()
            sched.step()
            global_step += 1

            dt = time.time() - t_step
            t_step = time.time()
            sps = step_ema.update(batch["x"].shape[0] / max(1e-6, dt))
            if global_step % LOG_EVERY_STEPS == 0 or it == steps_per_epoch - 1:
                remaining = (total_steps - global_step) * (dt if step_ema.value is None
                                                           else batch["x"].shape[0] / max(1e-3, step_ema.value))
                gpu_gb = (torch.cuda.memory_allocated() / 1e9) if device.type == "cuda" else 0.0
                job.set_progress(epoch=epoch + 1, step=global_step,
                                 pct=100.0 * global_step / total_steps,
                                 eta_s=remaining, samples_per_s=round(sps, 1),
                                 gpu_mem_gb=round(gpu_gb, 2))
                job.push_metric({"split": "train", "step": global_step,
                                 "epoch": epoch + 1, "lr": sched.get_last_lr()[0],
                                 "samples_per_s": round(sps, 1), **comps})
        if stopped:
            break

        # ------------------------------------------------------ validation
        if val_loader is not None and ((epoch + 1) % int(opt_cfg.get("val_every", 1)) == 0
                                       or epoch == epochs - 1):
            vm = _validate(model, val_loader, device, use_amp, amp_dtype, job)
            vm.update({"split": "val", "step": global_step, "epoch": epoch + 1})
            by_type = vm.pop("by_type", {})
            job.push_metric(vm)
            if by_type:
                job.push_metric({"split": "val_by_type", "step": global_step,
                                 "epoch": epoch + 1,
                                 "by_type": {TYPE_NAMES[t] if 0 <= t < N_TYPES else str(t): v
                                             for t, v in by_type.items()}})
            job.log(f"epoch {epoch + 1}/{epochs}  "
                    f"minADE={vm.get('minADE', float('nan')):.1f}m  "
                    f"minFDE={vm.get('minFDE', float('nan')):.1f}m  "
                    f"NLL={vm.get('NLL', float('nan')):.3f}  "
                    f"cov90@15s={vm.get('cov90_15s', float('nan')):.2f}  "
                    f"type_acc={vm.get('type_acc', float('nan')):.2f}")

            metric_val = vm.get(best_key)
            if metric_val is not None and metric_val < best["value"]:
                best = {"value": float(metric_val), "epoch": epoch + 1, "path": None}
                best["path"] = _save_checkpoint(job, model, opt, sched, scaler, cfg,
                                                epoch + 1, global_step, best, name="best.pt")
                job.log(f"new best {best_key}={metric_val:.3f} @ epoch {epoch + 1}")
                since_best = 0
            else:
                since_best += 1
                if patience and since_best >= patience:
                    job.log(f"early stop: no {best_key} improvement for {patience} validations")
                    break

        # ------------------------------------------------------ checkpoint
        if (epoch + 1) % int(ckpt_cfg.get("every_epochs", 5)) == 0:
            _save_checkpoint(job, model, opt, sched, scaler, cfg,
                             epoch + 1, global_step, best,
                             name=f"ckpt_ep{epoch + 1:04d}.pt")
            _prune_checkpoints(job, int(ckpt_cfg.get("keep_last", 5)))

    # ------------------------------------------------------------ finalise
    job.set_status("finalizing")
    last_path = _save_checkpoint(job, model, opt, sched, scaler, cfg,
                                 epoch + 1 if 'epoch' in dir() else start_epoch,
                                 global_step, best, name="last.pt")
    final_metrics = {}
    if stopped and not job.force_requested and val_loader is not None:
        job.log("graceful stop: running final validation")
        final_metrics = _validate(model, val_loader, device, use_amp, amp_dtype, job)
        final_metrics.pop("by_type", None)

    src = best["path"] if best["path"] and os.path.exists(best["path"]) else last_path
    card = _package_final(job, src, cfg, roster, n_params, best, final_metrics,
                          device, model)
    job.summary = {"best": {k: v for k, v in best.items() if k != "path"},
                   "params": n_params, "final_model": card["final_model_path"],
                   "stopped_by_user": stopped}
    job.set_status("stopped" if stopped else "finished", pct=100.0)
    job.log(f"done ({'stopped by user' if stopped else 'complete'}). "
            f"final model: {card['final_model_path']}")


# ================================================================= internals

@torch.no_grad()
def _validate(model, val_loader, device, use_amp, amp_dtype, job) -> dict:
    model.eval()
    acc = MetricAccumulator()
    for batch in val_loader:
        if job.force_requested:
            break
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(batch["x"], batch["agent_mask"])
        acc.update(batch_metrics(out, batch))
    model.train()
    return acc.result()


def _save_checkpoint(job, model, opt, sched, scaler, cfg, epoch, step, best,
                     name) -> str:
    path = os.path.join(job.dir, "checkpoints", name)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "cfg": cfg, "epoch": epoch, "step": step,
                "best": {k: v for k, v in best.items() if k != "path"}}, tmp)
    os.replace(tmp, path)
    job.push_metric({"split": "ckpt", "step": step, "epoch": epoch, "name": name})
    return path


def _prune_checkpoints(job, keep: int) -> None:
    d = os.path.join(job.dir, "checkpoints")
    periodic = sorted(f for f in os.listdir(d) if f.startswith("ckpt_ep"))
    for f in periodic[:-keep]:
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


def _load_checkpoint(path, model, opt, sched, scaler, device, job):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    if ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    best = ck.get("best", {"value": float("inf"), "epoch": -1})
    best.setdefault("path", None)
    job.log(f"resumed from {path} (epoch {ck['epoch']}, step {ck['step']})")
    return int(ck["epoch"]), int(ck["step"]), best


@torch.no_grad()
def _measure_latency(model, device) -> dict:
    """Inference latency for the paper's onboard-feasibility row (doc 8.4)."""
    from common.constants import F_DIM, HIST_STEPS
    model.eval()
    res = {}
    for bs in (1, 16):
        x = torch.zeros(bs, 8, HIST_STEPS, F_DIM, device=device)
        m = torch.ones(bs, 8, device=device)
        for _ in range(3):
            model(x, m)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        n = 30
        for _ in range(n):
            model(x, m)
        if device.type == "cuda":
            torch.cuda.synchronize()
        res[f"batch{bs}_ms"] = round((time.time() - t0) / n * 1000.0, 3)
    model.train()
    return res


def _package_final(job, src_ckpt, cfg, roster, n_params, best, final_metrics,
                   device, model) -> dict:
    final_dir = os.path.join(job.dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    final_path = os.path.join(final_dir, "model_final.pt")
    torch.save({"model": ck["model"], "model_cfg": cfg["model"],
                "data_cfg": cfg["data"], "epoch": ck["epoch"],
                "source_checkpoint": os.path.basename(src_ckpt)}, final_path)

    latency = {}
    try:
        latency = _measure_latency(model, device)
    except Exception as e:  # noqa: BLE001
        job.log(f"latency measurement skipped: {e!r}")

    card = {
        "job_id": job.id,
        "final_model_path": final_path.replace("\\", "/"),
        "source_checkpoint": os.path.basename(src_ckpt),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
        "data_roster": roster,
        "parameters": n_params,
        "best": {k: v for k, v in best.items() if k != "path"},
        "final_validation": final_metrics,
        "latency_ms": latency,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "os": platform.platform(),
        },
        "type_names": TYPE_NAMES,
    }
    atomic_write_json(os.path.join(final_dir, "model_card.json"), card)
    return card
