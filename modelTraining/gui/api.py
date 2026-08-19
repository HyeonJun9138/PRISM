"""JSON API + SSE event source behind gui/server.py.

Routes (all JSON):
  GET  /api/env                      torch/cuda/device facts
  GET  /api/data/runs                dataset runs under $PRISM_ROOT/Logs
  GET  /api/config/defaults          default config + GUI field specs
  GET  /api/jobs                     job history (live + on disk)
  GET  /api/jobs/detail?job_id=      snapshot + checkpoints + summary
  GET  /api/jobs/metrics?job_id=&after=   persisted metrics (chart rebuild)
  POST /api/train/start              {config?, roster, tag?, seeds?} -> job ids
  POST /api/train/resume             {job_id, checkpoint} -> new job
  POST /api/train/stop               {job_id, force?}
  POST /api/eval/start               {job_id | model_path, runs} -> eval job
  SSE  /api/train/stream?job_id=     events: status / metric / log
"""
from __future__ import annotations

import glob
import os
import time

from common.constants import (GUI_PORT, LOGS_ROOT, MODELS_ROOT, N_TYPES,
                              TYPE_NAMES)
from common.utils import deep_update, read_json, read_jsonl
from data.dataset import discover_runs, validate_episode_schema
from training.job import REGISTRY, TERMINAL

_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "defaults.json")


# ================================================================== routing

def handle(method: str, path: str, query: dict, body: dict | None):
    try:
        route = ROUTES.get((method, path))
        if route is None:
            return 404, {"error": f"no route {method} {path}"}
        return route(query, body or {})
    except Exception as e:  # noqa: BLE001 - every API error must reach the UI
        import traceback
        return 500, {"error": str(e), "trace": traceback.format_exc(limit=8)}


# ================================================================ endpoints

def _env(q, b):
    import torch
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "python": __import__("platform").python_version(),
            "models_root": MODELS_ROOT, "logs_root": LOGS_ROOT}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["bf16"] = torch.cuda.is_bf16_supported()
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / 1e9, 1)
    return 200, info


def _data_runs(q, b):
    runs = discover_runs()
    for r in runs:
        eps = sorted(glob.glob(os.path.join(r["path"], "episodes", "*.npz")))
        r["schema_ok"] = True
        if eps:
            problems = validate_episode_schema(eps[0])
            r["schema_ok"] = not problems
            r["schema_problems"] = problems
    return 200, {"runs": runs}


def _config_defaults(q, b):
    cfg = read_json(_DEFAULTS_PATH)
    return 200, {"config": cfg, "fields": FIELD_SPECS,
                 "type_names": TYPE_NAMES, "n_types": N_TYPES}


def _jobs(q, b):
    return 200, {"jobs": REGISTRY.list_jobs(), "current": REGISTRY.current_id}


def _job_detail(q, b):
    job_id = q.get("job_id", "")
    job = REGISTRY.get(job_id)
    if job is not None:
        snap = job.snapshot()
    else:
        p = os.path.join(MODELS_ROOT, job_id, "job.json")
        if not os.path.exists(p):
            return 404, {"error": f"unknown job {job_id}"}
        snap = read_json(p)
        snap["offline"] = True
    ck_dir = os.path.join(MODELS_ROOT, job_id, "checkpoints")
    cks = []
    if os.path.isdir(ck_dir):
        for f in sorted(os.listdir(ck_dir)):
            if f.endswith(".pt"):
                st = os.stat(os.path.join(ck_dir, f))
                cks.append({"name": f, "mb": round(st.st_size / 1e6, 1),
                            "mtime": time.strftime("%H:%M:%S", time.localtime(st.st_mtime))})
    snap["checkpoints"] = cks
    final = os.path.join(MODELS_ROOT, job_id, "final", "model_card.json")
    if os.path.exists(final):
        snap["model_card"] = read_json(final)
    rep = os.path.join(MODELS_ROOT, job_id, "eval_report.json")
    if os.path.exists(rep):
        snap["eval_report"] = read_json(rep)
    cfgp = os.path.join(MODELS_ROOT, job_id, "config.json")
    if os.path.exists(cfgp):
        snap["config"] = read_json(cfgp)
    return 200, snap


def _job_metrics(q, b):
    job_id = q.get("job_id", "")
    after = int(q.get("after", 0))
    job = REGISTRY.get(job_id)
    if job is not None:
        items, cursor = job.metrics_after(after)
    else:
        items = read_jsonl(os.path.join(MODELS_ROOT, job_id, "metrics.jsonl"))[after:]
        cursor = after + len(items)
    return 200, {"metrics": items[-20000:], "cursor": cursor}


def _train_start(q, b):
    roster = b.get("roster") or {}
    train_runs = [r for r in roster.get("train", []) if os.path.isdir(r)]
    val_runs = [r for r in roster.get("val", []) if os.path.isdir(r)]
    if not train_runs:
        return 400, {"error": "roster.train must list at least one existing run directory"}
    base = read_json(_DEFAULTS_PATH)
    cfg = deep_update(base, b.get("config") or {})
    tag = str(b.get("tag") or cfg.get("tag") or "run")[:40].replace(" ", "_")
    seeds = b.get("seeds") or [int(cfg.get("seed", 0))]
    from training.trainer import run_training
    ids = []
    for s in seeds:
        c = deep_update(cfg, {"seed": int(s)})
        job = REGISTRY.create_job(f"{tag}_s{int(s)}" if len(seeds) > 1 else tag)
        job.log(f"queued: train runs={train_runs} val runs={val_runs} seed={s}")
        REGISTRY.submit(job, run_training, c,
                        {"train": train_runs, "val": val_runs}, None)
        ids.append(job.id)
    return 200, {"job_ids": ids}


def _train_resume(q, b):
    src_id = str(b.get("job_id", ""))
    ck_name = str(b.get("checkpoint", "last.pt"))
    src_dir = os.path.join(MODELS_ROOT, src_id)
    ck_path = os.path.join(src_dir, "checkpoints", ck_name)
    cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(ck_path):
        return 404, {"error": f"checkpoint not found: {ck_path}"}
    if not os.path.exists(cfg_path):
        return 404, {"error": f"config not found for job {src_id}"}
    saved = read_json(cfg_path)
    cfg = deep_update(saved["config"], b.get("config") or {})
    from training.trainer import run_training
    job = REGISTRY.create_job(f"resume_{src_id[-6:]}")
    job.log(f"queued: resume {src_id}/{ck_name}")
    REGISTRY.submit(job, run_training, cfg, saved["roster"], ck_path)
    return 200, {"job_ids": [job.id]}


def _train_stop(q, b):
    ok = REGISTRY.request_stop(str(b.get("job_id", "")), bool(b.get("force")))
    return (200, {"ok": True}) if ok else (404, {"error": "unknown or offline job"})


def _eval_start(q, b):
    runs = [r for r in (b.get("runs") or []) if os.path.isdir(r)]
    if not runs:
        return 400, {"error": "runs must list at least one existing run directory"}
    model_path = b.get("model_path")
    if not model_path and b.get("job_id"):
        cand = os.path.join(MODELS_ROOT, b["job_id"], "final", "model_final.pt")
        if not os.path.exists(cand):
            cand = os.path.join(MODELS_ROOT, b["job_id"], "checkpoints", "best.pt")
        model_path = cand
    if not model_path or not os.path.exists(model_path):
        return 404, {"error": f"model not found: {model_path}"}
    from analysis.evaluate import run_evaluation
    job = REGISTRY.create_job("eval", kind="eval")
    job.log(f"queued: evaluate {model_path} on {runs}")
    REGISTRY.submit(job, run_evaluation, model_path, runs)
    return 200, {"job_ids": [job.id]}


ROUTES = {
    ("GET", "/api/env"): _env,
    ("GET", "/api/data/runs"): _data_runs,
    ("GET", "/api/config/defaults"): _config_defaults,
    ("GET", "/api/jobs"): _jobs,
    ("GET", "/api/jobs/detail"): _job_detail,
    ("GET", "/api/jobs/metrics"): _job_metrics,
    ("POST", "/api/train/start"): _train_start,
    ("POST", "/api/train/resume"): _train_resume,
    ("POST", "/api/train/stop"): _train_stop,
    ("POST", "/api/eval/start"): _eval_start,
}


# ===================================================================== SSE

def stream_events(job_id: str):
    """Generator of (event, payload). Ends shortly after the job terminates."""
    m_cur = 0
    l_cur = 0
    idle_after_end = 0
    while True:
        job = REGISTRY.get(job_id)
        if job is None:
            yield "status", {"status": "offline", "id": job_id}
            return
        metrics, m_cur = job.metrics_after(m_cur)
        if metrics:
            yield "metric", {"items": metrics[-500:]}
        logs, l_cur = job.logs_after(l_cur)
        if logs:
            yield "log", {"lines": logs[-100:]}
        yield "status", job.snapshot()
        if job.status in TERMINAL:
            idle_after_end += 1
            if idle_after_end >= 2:
                return
        time.sleep(0.7)


# ============================================================== field specs
# Rendered by the GUI as grouped, tooltipped inputs. Keep keys aligned with
# config/defaults.json.
FIELD_SPECS = [
    {"group": "Model", "key": "model.d_model", "label": "d_model", "type": "int",
     "min": 32, "max": 512, "help": "임베딩 차원. 128이면 ~1.2M 파라미터."},
    {"group": "Model", "key": "model.n_modes", "label": "K (갈래 수)", "type": "int",
     "min": 1, "max": 8, "help": "혼합 가우시안 갈래 수. K=1이면 단봉(rev.2와 동일)."},
    {"group": "Model", "key": "model.temporal_layers", "label": "temporal layers", "type": "int", "min": 1, "max": 6},
    {"group": "Model", "key": "model.agent_layers", "label": "agent layers", "type": "int", "min": 1, "max": 6},
    {"group": "Model", "key": "model.n_heads", "label": "attention heads", "type": "int", "min": 1, "max": 16},
    {"group": "Model", "key": "model.dropout", "label": "dropout", "type": "float", "min": 0, "max": 0.5},
    {"group": "Data", "key": "data.max_agents", "label": "max agents (A)", "type": "int",
     "min": 2, "max": 16, "help": "대상+자기항공기+주변. 슬롯 수."},
    {"group": "Data", "key": "data.t0_stride", "label": "t0 stride", "type": "int",
     "min": 1, "max": 20, "help": "에피소드에서 기준시각을 몇 스텝 간격으로 뽑을지. 작을수록 샘플 많음."},
    {"group": "Data", "key": "data.min_obs", "label": "min detections", "type": "int", "min": 2, "max": 20},
    {"group": "Data", "key": "data.val_fraction", "label": "val fraction", "type": "float",
     "min": 0.02, "max": 0.5, "help": "별도 val 런 미지정 시 에피소드 단위 분할 비율."},
    {"group": "Optim", "key": "optim.batch_size", "label": "batch size", "type": "int", "min": 8, "max": 2048},
    {"group": "Optim", "key": "optim.lr", "label": "learning rate", "type": "float", "min": 1e-6, "max": 0.01},
    {"group": "Optim", "key": "optim.epochs", "label": "epochs", "type": "int", "min": 1, "max": 1000},
    {"group": "Optim", "key": "optim.warmup_steps", "label": "warmup steps", "type": "int", "min": 0, "max": 20000},
    {"group": "Optim", "key": "optim.weight_decay", "label": "weight decay", "type": "float", "min": 0, "max": 0.1},
    {"group": "Optim", "key": "optim.grad_clip", "label": "grad clip", "type": "float", "min": 0.1, "max": 10},
    {"group": "Optim", "key": "optim.amp", "label": "AMP", "type": "select",
     "options": ["bf16", "fp16", "off"], "help": "RTX 30/40계열은 bf16 권장."},
    {"group": "Optim", "key": "optim.num_workers", "label": "data workers", "type": "int", "min": 0, "max": 16},
    {"group": "Optim", "key": "optim.early_stop_patience", "label": "early-stop patience", "type": "int",
     "min": 0, "max": 100, "help": "0 = 끔. N번 연속 val 개선 없으면 조기 종료."},
    {"group": "Loss", "key": "loss.w_mode", "label": "w_mode", "type": "float", "min": 0, "max": 5},
    {"group": "Loss", "key": "loss.w_type", "label": "w_type (보조)", "type": "float", "min": 0, "max": 5},
    {"group": "Loss", "key": "loss.w_mixture_max", "label": "w_mixture max", "type": "float",
     "min": 0, "max": 2, "help": "후반부에 혼합 NLL을 이 가중치까지 램프. pi 보정용."},
    {"group": "Loss", "key": "loss.mixture_ramp_start", "label": "mixture ramp start", "type": "float", "min": 0, "max": 1},
    {"group": "Checkpoint", "key": "checkpoint.every_epochs", "label": "save every N epochs", "type": "int", "min": 1, "max": 100},
    {"group": "Checkpoint", "key": "checkpoint.keep_last", "label": "keep last M", "type": "int", "min": 1, "max": 50},
    {"group": "Checkpoint", "key": "checkpoint.best_metric", "label": "best metric", "type": "select",
     "options": ["val/minADE", "val/minFDE", "val/NLL"]},
    {"group": "Run", "key": "seed", "label": "seed", "type": "int", "min": 0, "max": 999999},
    {"group": "Run", "key": "tag", "label": "tag", "type": "text"},
]
