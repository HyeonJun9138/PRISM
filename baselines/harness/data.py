"""Shared sample enumeration: baselines are scored on EXACTLY the sample set
the neural model is evaluated on, or the comparison is meaningless.

The (episode, t0, target) triples come from modelTraining's own index builder
with the model's data config -- same eligibility (min_obs, future validity),
same stride. Episodes are loaded once and grouped so per-track filters (IMM)
run a single pass per agent.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from common.constants import PRED_STEPS
from data.dataset import build_sample_index

EP_KEYS = ("gt_pos", "gt_vel", "gt_valid", "gt_type", "gt_intent",
           "obs_detected", "obs_pos", "obs_R")


def iter_eval_groups(run_dirs: list[str], data_cfg: dict, max_samples: int = 0):
    """Yield (ep_dict, {agent: [t0,...]}) per episode file, in index order.

    max_samples > 0 caps the TOTAL sample count (spread over the leading
    files) -- used by the IMM tuner, not by the final evaluation.
    """
    idx = build_sample_index(run_dirs, data_cfg)
    order = np.lexsort((idx.t0, idx.agent, idx.file_idx))
    total = 0
    current_file = -1
    groups: dict[int, list[int]] = defaultdict(list)

    def flush(fi):
        ep = dict(np.load(idx.files[fi]))
        ep = {k: ep[k] for k in EP_KEYS if k in ep}
        return ep, dict(groups)

    for o in order:
        fi = int(idx.file_idx[o])
        if fi != current_file and current_file >= 0:
            yield flush(current_file)
            groups.clear()
        current_file = fi
        groups[int(idx.agent[o])].append(int(idx.t0[o]))
        total += 1
        if max_samples and total >= max_samples:
            break
    if current_file >= 0 and groups:
        yield flush(current_file)


def future_of(ep: dict, t0: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    """GT future [PRED,2] world metres + validity [PRED]."""
    sl = slice(t0 + 1, t0 + 1 + PRED_STEPS)
    fut = np.nan_to_num(ep["gt_pos"][sl, target].astype(np.float64), nan=0.0)
    val = ep["gt_valid"][sl, target].astype(np.float64)
    return fut, val
