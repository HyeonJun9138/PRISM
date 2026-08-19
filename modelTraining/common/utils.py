"""Small shared utilities: seeding, atomic JSON i/o, timers, formatting."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices)."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed & 0x7FFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON via a temp file + os.replace so a crash never leaves a
    truncated file (paper-grade bookkeeping requirement)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: str, limit: int | None = None) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue    # tolerate a torn tail line from a crash
    if limit is not None:
        return out[-limit:]
    return out


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def fmt_duration(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def deep_update(base: dict, patch: dict) -> dict:
    """Recursively merge patch into a copy of base (patch wins)."""
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


class EMA:
    """Exponential moving average for throughput / ETA estimation."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x
        return self.value
