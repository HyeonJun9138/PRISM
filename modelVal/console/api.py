"""JSON API for the validation console. One global session, one predictor."""
from __future__ import annotations
import os

import time
import traceback
from typing import Any, Optional

from runtime.predictor import LivePredictor, list_models
from runtime.session import LiveSession

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

PREDICTOR = LivePredictor()
SESSION: Optional[LiveSession] = None


def _session() -> LiveSession:
    global SESSION
    if SESSION is None:
        SESSION = LiveSession(PREDICTOR)
    return SESSION


def handle(method: str, path: str, query: dict, body: dict | None):
    try:
        fn = ROUTES.get((method, path))
        if fn is None:
            return 404, {"error": f"no route {method} {path}"}
        return fn(query, body or {})
    except Exception as e:  # noqa: BLE001 - errors must reach the console UI
        return 500, {"error": f"{type(e).__name__}: {e}",
                     "trace": traceback.format_exc(limit=6)}


def _env(q, b):
    import torch
    return 200, {"cuda": torch.cuda.is_available(),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "model": PREDICTOR.info or None,
                 "pilot": ({"control": SESSION.control_mode,
                            "risk": SESSION.pilot.risk.name}
                           if SESSION is not None else None)}


def _models(q, b):
    return 200, {"models": list_models()}


def _model_load(q, b):
    import os
    path = os.path.abspath(str(b.get("path", "")))
    root = os.path.abspath(f"{PRISM_ROOT}/Logs/Models")
    # containment check: the console may be exposed for demos, and this
    # endpoint feeds torch.load -- restrict it to the models tree.
    if not path.startswith(root):
        return 403, {"error": "model path must live under Logs/Models"}
    info = PREDICTOR.load(path)
    if SESSION is not None:
        SESSION.reset_score()      # tiles must reflect THIS model only
        SESSION.rotate(seed=SESSION.seed)   # same-seed replay under new model
    return 200, info


def _control(q, b):
    s = _session()
    action = str(b.get("action", ""))
    if action == "play":
        s.paused = False
    elif action == "pause":
        s.paused = True
    elif action == "reset":
        s.rotate(seed=b.get("seed"))
    elif action == "preset":
        s.set_preset(str(b.get("preset", "test_ood")))
    elif action == "speed":
        s.speed = max(0.1, min(8.0, float(b.get("speed", 1.0))))
    else:
        return 400, {"error": f"unknown action '{action}'"}
    return 200, {"ok": True, "paused": s.paused, "speed": s.speed,
                 "preset": s.preset, "scenario": s.scenario_count}


def _keys(q, b):
    s = _session()
    s.keys.set(float(b.get("turn", 0.0)), float(b.get("accel", 0.0)),
               bool(b.get("brake", False)))
    return 200, {"ok": True}


def _pilot(q, b):
    """Switch ownship control source ('manual'|'planner') and/or the
    planner's risk arm ('none'|'cv'|'model')."""
    s = _session()
    state = s.set_pilot(control=b.get("control"), risk=b.get("risk"))
    return 200, {"ok": True, **state}


def _frames(q, b):
    """Polling fallback for networks whose middleboxes reset SSE streams:
    plain bounded GET responses always get through."""
    s = _session()
    after = int(q.get("after", 0))
    frames = s.latest_frames(after)[-4:]
    return 200, {"frames": frames,
                 "cursor": frames[-1]["seq"] if frames else after}


ROUTES = {
    ("GET", "/api/env"): _env,
    ("GET", "/api/models"): _models,
    ("GET", "/api/frames"): _frames,
    ("POST", "/api/model/load"): _model_load,
    ("POST", "/api/control"): _control,
    ("POST", "/api/keys"): _keys,
    ("POST", "/api/pilot"): _pilot,
}


def stream_frames():
    """SSE generator: every new logged frame (2 Hz sim-time, scaled by speed)."""
    s = _session()
    last = 0
    while True:
        frames = s.latest_frames(last)
        if frames:
            last = frames[-1]["seq"]
            for f in frames[-4:]:      # never flood a slow client
                yield f
        time.sleep(0.10)
