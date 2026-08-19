"""Job state, persistence and the sequential job queue.

One Job = one training (or evaluation) run, owning a directory under
$PRISM_ROOT/Models/<job_id>/ that survives the GUI process:

  job.json           status snapshot (atomic-replaced on every change)
  config.json        exact resolved config
  metrics.jsonl      every metric point ever emitted (paper log)
  train.log          plain-text log
  checkpoints/       ckpt_ep####.pt, best.pt, last.pt
  final/             model_final.pt + model_card.json  (graceful finish/stop)
  eval_report.json   written by analysis.evaluate

The GUI streams live state from the in-memory JobState; after a restart it
re-reads the persisted files, so nothing about a run is ever lost.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from collections import deque

from common.constants import MODELS_ROOT
from common.utils import append_jsonl, atomic_write_json, now_stamp, read_json

TERMINAL = ("finished", "stopped", "failed", "aborted")


class JobState:
    def __init__(self, job_id: str, job_dir: str, kind: str = "train"):
        self.id = job_id
        self.dir = job_dir
        self.kind = kind
        self.lock = threading.RLock()
        self.status = "queued"
        self.progress: dict = {"phase": "queued", "pct": 0.0, "epoch": 0,
                               "epochs": 0, "step": 0, "steps_per_epoch": 0,
                               "eta_s": None, "samples_per_s": 0.0,
                               "gpu_mem_gb": 0.0}
        self.metrics: list[dict] = []          # in-memory mirror of metrics.jsonl
        self.logs: deque = deque(maxlen=2000)
        self.stop_requested = False
        self.force_requested = False
        self.error: str | None = None
        self.created = time.time()
        self.summary: dict = {}

    # ------------------------------------------------------------- mutation
    def log(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        entry = f"[{stamp}] {line}"
        with self.lock:
            self.logs.append(entry)
        try:
            with open(os.path.join(self.dir, "train.log"), "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except OSError:
            pass

    def push_metric(self, point: dict) -> None:
        point = dict(point)
        point["t"] = round(time.time(), 3)
        with self.lock:
            self.metrics.append(point)
        append_jsonl(os.path.join(self.dir, "metrics.jsonl"), point)

    def set_status(self, status: str, **progress) -> None:
        with self.lock:
            self.status = status
            self.progress.update(progress)
            self.progress["phase"] = status
        self.persist()

    def set_progress(self, **kv) -> None:
        with self.lock:
            self.progress.update(kv)

    def persist(self) -> None:
        with self.lock:
            snap = {"id": self.id, "kind": self.kind, "status": self.status,
                    "progress": dict(self.progress), "error": self.error,
                    "created": self.created, "summary": self.summary}
        atomic_write_json(os.path.join(self.dir, "job.json"), snap)

    # ---------------------------------------------------------------- views
    def snapshot(self) -> dict:
        with self.lock:
            return {"id": self.id, "kind": self.kind, "status": self.status,
                    "progress": dict(self.progress), "error": self.error,
                    "summary": dict(self.summary),
                    "n_metrics": len(self.metrics), "n_logs": len(self.logs)}

    def metrics_after(self, cursor: int) -> tuple[list[dict], int]:
        with self.lock:
            return self.metrics[cursor:], len(self.metrics)

    def logs_after(self, cursor: int) -> tuple[list[str], int]:
        with self.lock:
            items = list(self.logs)
        start = max(0, cursor)
        return items[start:], len(items)


class JobRegistry:
    """Owns live JobState objects and a single sequential worker thread.
    Sequential on purpose: one GPU, one training at a time; queued jobs
    (e.g. a 5-seed sweep) run back-to-back unattended."""

    def __init__(self, models_root: str = MODELS_ROOT):
        self.root = models_root
        os.makedirs(self.root, exist_ok=True)
        self.jobs: dict[str, JobState] = {}
        self.queue: deque = deque()
        self.lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self.current_id: str | None = None

    # ------------------------------------------------------------ creation
    def create_job(self, tag: str, kind: str = "train") -> JobState:
        job_id = f"{now_stamp()}_{tag}"
        job_dir = os.path.join(self.root, job_id)
        os.makedirs(os.path.join(job_dir, "checkpoints"), exist_ok=True)
        job = JobState(job_id, job_dir, kind)
        with self.lock:
            self.jobs[job_id] = job
        job.persist()
        return job

    def submit(self, job: JobState, target, *args) -> None:
        """Queue target(job, *args) for the worker thread."""
        with self.lock:
            self.queue.append((job, target, args))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_loop,
                                                name="job-worker", daemon=True)
                self._worker.start()

    def _run_loop(self) -> None:
        while True:
            with self.lock:
                if not self.queue:
                    self._worker = None
                    return
                job, target, args = self.queue.popleft()
                self.current_id = job.id
            if job.stop_requested:      # cancelled while queued
                job.set_status("stopped")
                continue
            try:
                target(job, *args)
            except Exception:  # noqa: BLE001
                job.error = traceback.format_exc(limit=20)
                job.log("FATAL: " + job.error.splitlines()[-1])
                job.set_status("failed")
            finally:
                with self.lock:
                    self.current_id = None

    # ------------------------------------------------------------- control
    def get(self, job_id: str) -> JobState | None:
        with self.lock:
            return self.jobs.get(job_id)

    def request_stop(self, job_id: str, force: bool = False) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.stop_requested = True
        job.force_requested = job.force_requested or force
        job.log("stop requested" + (" (force)" if force else " (graceful)"))
        return True

    # --------------------------------------------------------------- lists
    def list_jobs(self) -> list[dict]:
        """Live jobs + jobs persisted on disk from earlier sessions."""
        seen = {}
        with self.lock:
            for j in self.jobs.values():
                seen[j.id] = j.snapshot()
        if os.path.isdir(self.root):
            for name in os.listdir(self.root):
                if name in seen:
                    continue
                p = os.path.join(self.root, name, "job.json")
                if os.path.exists(p):
                    try:
                        snap = read_json(p)
                        snap["offline"] = True
                        seen[snap.get("id", name)] = snap
                    except Exception:
                        continue
        return sorted(seen.values(), key=lambda s: s.get("id", ""), reverse=True)


REGISTRY = JobRegistry()
