"""Dataset discovery, sample indexing and feature construction.

Implements 논문관련.txt [5.1] (rev.3) exactly:

* one sample = (episode, anchor time t0, one target agent)
* input  = OBSERVATION-layer arrays only (never ground truth) over the past
  HIST_STEPS logged steps, for the target + ownship + nearest neighbours
* target = GROUND-TRUTH future positions over PRED_STEPS logged steps
* frame  = target-centric: origin at the target's LAST OBSERVED position,
  x-axis along the target's observed direction of motion.

Frame heading is estimated from the last two distinct valid observations of
the target -- NOT from GT heading -- because at inference time GT does not
exist. If the observed displacement is < HEADING_MIN_DISP (hovering target),
the frame falls back to world-axis alignment (theta = 0), which is
deterministic and available at inference time too.

Missing observations are NEVER interpolated (doc 3.5): undetected timesteps
carry zeroed kinematic features plus detected=0, and the gap/track-age
features let the model see how stale the information is. NaN discipline:
obs_pos / obs_R contain NaN where undetected, so every read of those arrays
is routed through np.where(detected, ..., 0) before any arithmetic result is
kept.

Track identity (data_cfg["track_rule"], review round 1, R3 major 3):
  * "current" -- a detection is usable only if it belongs to the track that
    is OPEN at t0 (obs_track_id[t0] >= 0). The tracker closes a track after
    drop_after_misses consecutive misses and hands a reacquired object a NEW
    id, so the predictor must not see the object's earlier, closed tracks.
    This is the tracking model the paper states.
  * "slot" -- every detection of the physical agent slot, across track drops.
    That is PERFECT re-association (an oracle the stated tracker does not
    have). It is the behaviour of every model trained before 2026-09-26, so
    a config WITHOUT the key means "slot": a legacy checkpoint is always fed
    the way it was trained. Kept as the oracle-association control.
data_cfg["index_track_rule"] overrides the rule used for sample ELIGIBILITY
only, so an oracle model can be scored on exactly the current-track sample
set (paired comparison) while still being fed its own inputs.
"""
from __future__ import annotations

import functools
import glob
import os
from dataclasses import dataclass, field

import numpy as np

from common.constants import (AGE_SCALE, F_DIM, GAP_SCALE, HIST_STEPS, LOG_DT,
                              LOGS_ROOT, POS_SCALE, PRED_STEPS, R_SCALE,
                              VEL_SCALE)
from common.utils import read_json, read_jsonl

HEADING_MIN_DISP = 0.3      # m; below this the target is treated as hovering
OWNSHIP_POS_STD = 0.5       # m; ownship knows its own position to ~GPS level

TRACK_RULES = ("current", "slot")


def track_rule(data_cfg: Mapping) -> str:
    """Feature-construction track rule of a data config (see module doc).

    A missing key is a legacy (pre-2026-09-26) config and means "slot".
    """
    rule = str(data_cfg.get("track_rule", "slot"))
    if rule not in TRACK_RULES:
        raise ValueError(f"track_rule must be one of {TRACK_RULES}, got {rule!r}")
    return rule


def index_track_rule(data_cfg: Mapping) -> str:
    """Eligibility track rule: ``index_track_rule`` if set, else track_rule."""
    rule = str(data_cfg.get("index_track_rule", track_rule(data_cfg)))
    if rule not in TRACK_RULES:
        raise ValueError(f"index_track_rule must be one of {TRACK_RULES}, got {rule!r}")
    return rule


def window_detections(ep: Mapping, t0: int, rule: str) -> np.ndarray:
    """[HIST, N] bool: detections in the history window ending at t0 that a
    predictor may use under ``rule`` ("current" or "slot", module doc).

    Under "current" an agent with no open track at t0 contributes nothing,
    and detections of its earlier, closed tracks are masked. Raises when the
    episode carries no obs_track_id: silently falling back to "slot" would
    reintroduce the oracle association this rule exists to remove.
    """
    if t0 < HIST_STEPS - 1:
        raise ValueError(f"t0={t0} leaves no full history window")
    h0 = t0 - HIST_STEPS + 1
    det = np.asarray(ep["obs_detected"][h0:t0 + 1]).astype(bool)
    if rule == "slot":
        return det
    if rule != "current":
        raise ValueError(f"unknown track rule {rule!r}")
    if "obs_track_id" not in ep:
        raise KeyError("track_rule='current' needs obs_track_id in the episode "
                       "arrays (pass it through every ep_dict builder)")
    tid = np.asarray(ep["obs_track_id"][h0:t0 + 1])
    cur = np.asarray(ep["obs_track_id"][t0])
    return det & (tid == cur[None, :]) & (cur >= 0)[None, :]

REQUIRED_KEYS = ["t", "gt_valid", "gt_pos", "gt_type", "obs_detected",
                 "obs_pos", "obs_R", "obs_track_age", "own_pos"]


# =============================================================== run discovery

def discover_runs(logs_root: str = LOGS_ROOT) -> list[dict]:
    """List dataset runs under Logs/ with cheap metadata (no NPZ opened)."""
    runs = []
    if not os.path.isdir(logs_root):
        return runs
    for name in sorted(os.listdir(logs_root)):
        run_dir = os.path.join(logs_root, name)
        if not os.path.isdir(run_dir):
            continue
        episodes = glob.glob(os.path.join(run_dir, "episodes", "*.npz"))
        if not episodes:
            continue
        info = {"run_id": name, "path": run_dir.replace("\\", "/"),
                "n_episodes": len(episodes), "preset": "", "created": ""}
        man_path = os.path.join(run_dir, "manifest.json")
        if os.path.exists(man_path):
            try:
                man = read_json(man_path)
                info["preset"] = str(man.get("preset", man.get("config", {}).get("name", "")))
                info["created"] = str(man.get("created", man.get("created_at", "")))
            except Exception:
                pass
        idx = read_jsonl(os.path.join(run_dir, "index.jsonl"))
        if idx:
            info["n_episodes"] = len(idx)
            nmac = sum(1 for r in idx if r.get("nmac_occurred"))
            info["nmac_episodes"] = nmac
        runs.append(info)
    return runs


def validate_episode_schema(path: str) -> list[str]:
    """Return a list of problems with one episode file (empty = OK)."""
    problems = []
    try:
        with np.load(path) as z:
            keys = set(z.files)
            for k in REQUIRED_KEYS:
                if k not in keys:
                    problems.append(f"missing key '{k}'")
            if problems:
                return problems
            T, N = z["gt_valid"].shape
            if z["obs_pos"].shape != (T, N, 2):
                problems.append(f"obs_pos shape {z['obs_pos'].shape} != {(T, N, 2)}")
            if z["obs_R"].shape != (T, N, 2, 2):
                problems.append(f"obs_R shape {z['obs_R'].shape} != {(T, N, 2, 2)}")
            if T < HIST_STEPS + PRED_STEPS + 1:
                problems.append(f"episode too short: T={T}")
    except Exception as e:  # noqa: BLE001 - surfaced to the GUI as-is
        problems.append(f"unreadable: {e}")
    return problems


# ============================================================== sample index

@dataclass
class SampleIndex:
    """Flat index of (file, t0, agent) triples plus the file table."""
    files: list[str] = field(default_factory=list)
    file_idx: np.ndarray = None     # [S] int32
    t0: np.ndarray = None           # [S] int32
    agent: np.ndarray = None        # [S] int16

    def __len__(self) -> int:
        return 0 if self.file_idx is None else int(self.file_idx.shape[0])

    def subset(self, mask: np.ndarray) -> "SampleIndex":
        return SampleIndex(self.files, self.file_idx[mask],
                           self.t0[mask], self.agent[mask])

    def save(self, path: str) -> None:
        np.savez_compressed(path, files=np.array(self.files, dtype=object),
                            file_idx=self.file_idx, t0=self.t0, agent=self.agent)

    @staticmethod
    def load(path: str) -> "SampleIndex":
        z = np.load(path, allow_pickle=True)
        return SampleIndex(list(z["files"]), z["file_idx"], z["t0"], z["agent"])


def build_sample_index(run_dirs: list[str], data_cfg: dict,
                       progress_cb=None) -> SampleIndex:
    """Scan every episode of the given runs and enumerate eligible samples.

    Eligibility per (t0, agent):
      * >= min_obs usable detections of the agent inside the history window
        under index_track_rule(data_cfg) (also guarantees the >=2 valid
        points the frame heading needs)
      * gt_valid at t0 and >= min_future_valid GT steps in the future window
    t0 runs over range(HIST_STEPS-1, T-PRED_STEPS, t0_stride).
    """
    stride = int(data_cfg.get("t0_stride", 5))
    min_obs = max(2, int(data_cfg.get("min_obs", 3)))
    min_fut = int(data_cfg.get("min_future_valid", 27))
    rule = index_track_rule(data_cfg)

    files: list[str] = []
    for rd in run_dirs:
        files.extend(sorted(glob.glob(os.path.join(rd, "episodes", "*.npz"))))
    if not files:
        raise FileNotFoundError(f"no episode NPZ files under {run_dirs}")

    fi_l, t0_l, ag_l = [], [], []
    for fi, path in enumerate(files):
        with np.load(path) as z:
            det = z["obs_detected"].astype(bool)       # [T,N]
            valid = z["gt_valid"].astype(bool)         # [T,N]
            tid = z["obs_track_id"] if rule == "current" else None
        T, N = det.shape
        # cumulative sums let each t0 window be tested in O(1)
        det_c = np.concatenate([np.zeros((1, N), np.int32),
                                np.cumsum(det, axis=0, dtype=np.int32)])
        val_c = np.concatenate([np.zeros((1, N), np.int32),
                                np.cumsum(valid, axis=0, dtype=np.int32)])
        ep_ids = {"obs_detected": det, "obs_track_id": tid}
        for t0 in range(HIST_STEPS - 1, T - PRED_STEPS, stride):
            h0 = t0 - HIST_STEPS + 1
            if rule == "slot":
                n_det = det_c[t0 + 1] - det_c[h0]                  # [N]
            else:
                n_det = window_detections(ep_ids, t0, rule).sum(axis=0)
            n_fut = val_c[t0 + 1 + PRED_STEPS] - val_c[t0 + 1]     # [N]
            ok = (n_det >= min_obs) & valid[t0] & (n_fut >= min_fut)
            for a in np.nonzero(ok)[0]:
                fi_l.append(fi); t0_l.append(t0); ag_l.append(a)
        if progress_cb and (fi % 20 == 0 or fi == len(files) - 1):
            progress_cb(fi + 1, len(files), len(fi_l))

    idx = SampleIndex(
        [f.replace("\\", "/") for f in files],
        np.asarray(fi_l, np.int32), np.asarray(t0_l, np.int32),
        np.asarray(ag_l, np.int16))
    return idx


def split_train_val(idx: SampleIndex, val_fraction: float,
                    seed: int) -> tuple[SampleIndex, SampleIndex]:
    """Split BY EPISODE, never by sample: samples of one episode share the
    same trajectories, so a sample-level split would leak (doc 8.1)."""
    n_files = len(idx.files)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_files)
    n_val = max(1, int(round(n_files * val_fraction))) if n_files > 1 else 0
    val_files = set(order[:n_val].tolist())
    val_mask = np.isin(idx.file_idx, list(val_files))
    return idx.subset(~val_mask), idx.subset(val_mask)


# ========================================================== feature building

#: Only the arrays build_sample actually reads. Loading (and caching) the
#: rest -- gt_vel, gt_psi, gt_mode, t -- roughly doubled the per-episode
#: memory and halved the effective cache size for nothing. obs_track_id is
#: [T,N] int32 (a few KB) and is needed by track_rule "current".
_NEEDED_KEYS = ("gt_pos", "gt_valid", "gt_type", "gt_intent",
                "obs_detected", "obs_pos", "obs_R", "obs_track_age", "own_pos",
                "obs_track_id")


@functools.lru_cache(maxsize=256)
def _load_episode(path: str) -> dict:
    """Per-process episode cache. Arrays are read-only after load.

    Cache sizing note: with a fully random sampler over thousands of episode
    files this cache thrashes no matter its size, which is why the loader
    uses a BLOCK shuffle (data/loader.py) -- consecutive batches draw from a
    window of files small enough to live in here.
    """
    with np.load(path) as z:
        ep = {k: z[k] for k in _NEEDED_KEYS if k in z.files}
    return ep


def _rot_matrix(theta: float) -> np.ndarray:
    """World -> frame rotation: v_frame = R @ v_world."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def build_sample(path: str | Mapping, t0: int, target: int, data_cfg: dict,
                 with_labels: bool = True) -> dict:
    """Construct one training sample. Returns float32 numpy arrays:

      x           [A, HIST, F_DIM]   features (A = max_agents slots)
      agent_mask  [A]                1 where the slot holds a real agent
      y           [PRED, 2]          GT future in frame, / POS_SCALE
      y_valid     [PRED]             1 where GT exists
      type_label  ()                 int64 platform type of the target
      meta_*                         ints for stratified evaluation

    ``path`` may also be an already-loaded episode dict (the live validation
    environment feeds the engine's in-memory arrays through the SAME feature
    code that built the training set -- reimplementing it would invite drift).
    ``with_labels=False`` skips the future-GT slice, for prediction at the
    live edge where the future does not exist yet; the frame transform of the
    sample is returned under ``frame`` either way.
    """
    A = int(data_cfg.get("max_agents", 8))
    ep = _load_episode(path) if isinstance(path, str) else path
    rule = track_rule(data_cfg)
    N = ep["obs_detected"].shape[1]
    h0 = t0 - HIST_STEPS + 1
    W = slice(h0, t0 + 1)

    # [H,N] bool; under "current" the earlier, closed tracks of an object are
    # masked exactly as if they had never been detected (module doc)
    det = window_detections(ep, t0, rule)
    # track-history class of the target, identical under both rules, for the
    # reacquisition breakdown R3 asks for: 0 = every window detection belongs
    # to the track open at t0; 1 = the window also holds detections of an
    # earlier, closed track (reacquired object); 2 = no open track at t0
    # (only reachable under "slot" eligibility); -1 = no track ids stored.
    reacq = -1
    if "obs_track_id" in ep:
        slot_det = window_detections(ep, t0, "slot")[:, target]
        tid_t = np.asarray(ep["obs_track_id"][W])[:, target]
        cur_t = int(ep["obs_track_id"][t0][target])
        if cur_t < 0:
            reacq = 2
        else:
            reacq = int(bool(np.any(slot_det & (tid_t != cur_t))))
    pos_raw = ep["obs_pos"][W].astype(np.float64)   # [H,N,2] NaN when ~det
    R_raw = ep["obs_R"][W].astype(np.float64)       # [H,N,2,2]
    age_raw = ep["obs_track_age"][W].astype(np.float64)
    own = ep["own_pos"][W].astype(np.float64)       # [H,2]

    # ---------------------------------------------------------------- frame
    hs = np.nonzero(det[:, target])[0]
    p_last = pos_raw[hs[-1], target]
    theta = 0.0
    for h_prev in hs[:-1][::-1]:
        d = p_last - pos_raw[h_prev, target]
        if float(np.hypot(d[0], d[1])) >= HEADING_MIN_DISP:
            theta = float(np.arctan2(d[1], d[0]))
            break
    Rm = _rot_matrix(theta)
    origin = p_last

    # ------------------------------------------------------- slot selection
    # slot 0 = target, slot 1 = ownship, slots 2.. = nearest other agents
    others = []
    for a in range(N):
        if a == target or not det[:, a].any():
            continue
        pl = pos_raw[np.nonzero(det[:, a])[0][-1], a]
        others.append((float(np.hypot(*(pl - origin))), a))
    others.sort()
    slots = [("agent", target), ("ownship", -1)] + \
            [("agent", a) for _, a in others[: max(0, A - 2)]]

    x = np.zeros((A, HIST_STEPS, F_DIM), dtype=np.float32)
    agent_mask = np.zeros(A, dtype=np.float32)

    for si, (kind, a) in enumerate(slots):
        agent_mask[si] = 1.0
        if kind == "ownship":
            d_s = np.ones(HIST_STEPS, dtype=bool)
            p_s = own
            R_s = np.broadcast_to(np.eye(2) * OWNSHIP_POS_STD ** 2,
                                  (HIST_STEPS, 2, 2))
            age_s = np.full(HIST_STEPS, AGE_SCALE)
        else:
            d_s = det[:, a]
            p_s = pos_raw[:, a]
            R_s = R_raw[:, a]
            age_s = age_raw[:, a]

        m = d_s.astype(np.float64)
        # relative position in frame; np.where kills the NaN of undetected rows
        rel = (p_s - origin) @ Rm.T
        rel = np.where(d_s[:, None], rel, 0.0)
        rel = np.nan_to_num(rel, nan=0.0)

        # finite-difference velocity over consecutive detected steps
        dv = np.zeros((HIST_STEPS, 2))
        dvalid = np.zeros(HIST_STEPS)
        both = d_s[1:] & d_s[:-1]
        step_v = (p_s[1:] - p_s[:-1]) / LOG_DT @ Rm.T
        dv[1:] = np.where(both[:, None], step_v, 0.0)
        dv = np.nan_to_num(dv, nan=0.0)
        dvalid[1:] = both

        # measurement covariance rotated into the frame
        Rf = Rm @ np.nan_to_num(R_s, nan=0.0) @ Rm.T          # [H,2,2]
        r00 = np.clip(Rf[:, 0, 0], 0.0, None)
        r11 = np.clip(Rf[:, 1, 1], 0.0, None)
        denom = np.sqrt(r00 * r11) + 1e-9
        rho_r = np.clip(Rf[:, 0, 1] / denom, -1.0, 1.0)

        # steps since the previous detection (0 at a detected step)
        gap = np.zeros(HIST_STEPS)
        last = -1
        for h in range(HIST_STEPS):
            if d_s[h]:
                last = h
                gap[h] = 0.0
            else:
                gap[h] = (h - last) if last >= 0 else (h + 1)

        x[si, :, 0] = rel[:, 0] / POS_SCALE
        x[si, :, 1] = rel[:, 1] / POS_SCALE
        x[si, :, 2] = m
        x[si, :, 3] = dv[:, 0] / VEL_SCALE
        x[si, :, 4] = dv[:, 1] / VEL_SCALE
        x[si, :, 5] = dvalid
        x[si, :, 6] = np.sqrt(r00) / R_SCALE * m
        x[si, :, 7] = np.sqrt(r11) / R_SCALE * m
        x[si, :, 8] = rho_r * m
        x[si, :, 9] = np.clip(age_s * m, 0, 10 * AGE_SCALE) / AGE_SCALE
        x[si, :, 10] = np.clip(gap, 0, 2 * GAP_SCALE) / GAP_SCALE
        x[si, :, 11] = 1.0 if kind == "ownship" else 0.0
        x[si, :, 12] = 1.0 if si == 0 else 0.0

    # ---------------------------------------------------------------- labels
    if with_labels:
        Wf = slice(t0 + 1, t0 + 1 + PRED_STEPS)
        fut = ep["gt_pos"][Wf, target].astype(np.float64)
        y_valid = ep["gt_valid"][Wf, target].astype(np.float32)
        y = ((np.nan_to_num(fut, nan=0.0) - origin) @ Rm.T) / POS_SCALE
        y = (y * y_valid[:, None]).astype(np.float32)
    else:
        y = np.zeros((PRED_STEPS, 2), dtype=np.float32)
        y_valid = np.zeros(PRED_STEPS, dtype=np.float32)

    return {
        "x": x,
        "agent_mask": agent_mask,
        "y": y.astype(np.float32),
        "y_valid": y_valid,
        "type_label": np.int64(ep["gt_type"][target]),
        "meta_intent": np.int64(ep["gt_intent"][t0, target]) if "gt_intent" in ep else np.int64(-1),
        "meta_t0": np.int64(t0),
        "meta_reacq": np.int64(reacq),
        # frame transform (origin [m], theta [rad]) -- predictions come out in
        # this frame; the live console inverts it to draw in world axes.
        "frame": np.array([origin[0], origin[1], theta], dtype=np.float64),
    }
