"""Shared sample enumeration: baselines are scored on EXACTLY the sample set
the neural model is evaluated on, or the comparison is meaningless.

The (episode, t0, target) triples come from modelTraining's own index builder
with the model's data config -- same eligibility (min_obs, future validity,
index_track_rule), same stride. Episodes are loaded once and grouped so
per-track filters (IMM) run a single pass per agent.

Track identity (review round 1, R3 major 3): the observation-based methods
read detections under track_rule(data_cfg) exactly like the neural feature
builder (modelTraining/data/dataset.py module doc), which needs
obs_track_id -- hence its place in EP_KEYS.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from common.constants import HIST_STEPS, PRED_STEPS
from data.dataset import build_sample_index, window_detections

EP_KEYS = ("gt_pos", "gt_vel", "gt_valid", "gt_type", "gt_intent",
           "obs_detected", "obs_pos", "obs_R", "obs_track_id")

#: key under which iter_eval_groups stores the episode's file path
FILE_KEY = "_file"

#: the paper's full5x data config (pre-round-1 main.DATA_CFG); used when no
#: compared model is named
DEFAULT_DATA_CFG = {"max_agents": 10, "t0_stride": 4, "min_obs": 3,
                    "min_future_valid": 27}


def resolve_data_cfg(model_data_cfg: dict | None, track_rule_arg: str | None,
                     index_rule_arg: str | None) -> dict:
    """Data config of a baseline run: the compared model's data config (else
    DEFAULT_DATA_CFG) for identical eligibility, with its track rule unless
    overridden -- a legacy config without the key means "slot" -- and
    "current" when no model is named. ``index_rule_arg`` sets eligibility
    only (data_cfg["index_track_rule"])."""
    from data.dataset import TRACK_RULES, track_rule
    for r in (track_rule_arg, index_rule_arg):
        if r is not None and r not in TRACK_RULES:
            raise ValueError(f"track rule must be one of {TRACK_RULES}, got {r!r}")
    if model_data_cfg is not None:
        cfg = dict(model_data_cfg)
        rule = track_rule_arg or track_rule(cfg)
    else:
        cfg = dict(DEFAULT_DATA_CFG)
        rule = track_rule_arg or "current"
    cfg["track_rule"] = rule
    if index_rule_arg:
        cfg["index_track_rule"] = index_rule_arg
    return cfg


def iter_eval_groups(run_dirs: list[str], data_cfg: dict, max_samples: int = 0):
    """Yield (ep_dict, {agent: [t0,...]}) per episode file, in index order.

    ``ep_dict[FILE_KEY]`` is the episode path exactly as the index spells it
    (the key of per-sample outputs). max_samples > 0 caps the TOTAL sample
    count (spread over the leading files) -- used by the fitters/tuner, not
    by the final evaluation.
    """
    idx = build_sample_index(run_dirs, data_cfg)
    order = np.lexsort((idx.t0, idx.agent, idx.file_idx))
    total = 0
    current_file = -1
    groups: dict[int, list[int]] = defaultdict(list)

    def flush(fi):
        ep = dict(np.load(idx.files[fi]))
        ep = {k: ep[k] for k in EP_KEYS if k in ep}
        ep[FILE_KEY] = idx.files[fi]
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


def reacq_class(ep: dict, t0: int, target: int) -> int:
    """Track-history class of the target -- the value build_sample returns as
    meta_reacq (same definition, rule-independent): 0 = every detection in
    the history window belongs to the track open at t0; 1 = the window also
    holds detections of an earlier, closed track; 2 = no open track at t0;
    -1 = no track ids stored."""
    if "obs_track_id" not in ep:
        return -1
    W = slice(t0 - HIST_STEPS + 1, t0 + 1)
    slot_det = window_detections(ep, t0, "slot")[:, target]
    tid_t = np.asarray(ep["obs_track_id"][W])[:, target]
    cur = int(ep["obs_track_id"][t0][target])
    if cur < 0:
        return 2
    return int(bool(np.any(slot_det & (tid_t != cur))))
