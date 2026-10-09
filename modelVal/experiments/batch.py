"""Headless planner-utility batch: (risk arm x episode) closed loop.

Design notes (paper experiment section mirrors these):
  * The PLANNER IS FROZEN (controller.py config); only the RiskProvider
    differs between arms. Episode indices are IDENTICAL across arms, so
    every comparison is paired on the same traffic.
  * Episode indices start at 10000 -- inside the test_ood seed block but
    far beyond the dataset's generated range, so scenarios come from the
    OOD distribution without reusing evaluation episodes. (--preset test_id
    draws the same indices from the test_id block instead.)
  * Safety is scored on GROUND TRUTH separations (the simulator's world),
    never on observations: sensors belong to the system under test.
  * 'unavoidable' flag: an NMAC whose intruder was first DETECTED with
    less lead time than a full-rate lateral displacement of one NMAC
    radius needs (t_need = sqrt(2*R_nmac/a_lat_max)); reported separately
    so late-detection physics does not dilute the arm comparison.

Usage:
  python -m experiments.batch --arm cv --episodes 100 --out <dir>
  (run one process per arm; only the predictor arms touch the GPU)

Review round 1 (Reviewer 3) additions
-------------------------------------
Every row field that existed before is still computed by the original code
(2 Hz, union-based), so with ``--track-rule slot`` and every other option at
its default the old rows reproduce bit for bit (the DECISIONS D5.1 gate).
New options (all apply to EVERY arm unless noted):

  --preset test_ood|test_id   BASE_SPLIT of config_for_episode (R3 M8)
  --track-rule current|slot   detections the arms may use (R3 M3): CV fits
                              velocity on the current track only; predictor
                              arms refuse a checkpoint trained under another
                              rule unless --allow-rule-mismatch
  --hard-geom arm|shared      hard-constraint geometry (R3 M4)
  --sigma-floor F             floor on every disk sigma (R3 M5; 10.0)
  --plateau-scale S           soft_standoff = NMAC radius x S (1.0)
  --w-safety W                planner weight_safety (12.0)
  --safety-norm pool|absolute  (+ --safety-ref R) safety-term normalization
  --safety-ref-probe          record the per-cycle pool max of raw safety
  --log-preds DIR             closed-loop forecasts for scoring (R3 minor 5):
                              DIR/<arm>_<index:05d>.npz per episode (absolute
                              index, so disjoint index shards may share DIR;
                              an existing file is refused), every 4th replan;
                              forecasts + GT futures + intruder labels +
                              the ownship trajectory at every logged frame
                              (field list: save_pred_log). Strata are derived
                              by experiments/aggregate_r3.py from these raw
                              arrays with the definitions as parameters.

The FIRST stdout line is a banner with the arm, track rule, model path and
model rule and every knob (DECISIONS D5.3: silent degradation must be
impossible to miss).

New row fields (JSONL, one row per episode)
  meta     preset, track_rule, allow_rule_mismatch, hard_geom, sigma_floor,
           plateau_scale, w_safety, safety_norm, safety_ref,
           safety_ref_probe, model_path, model_track_rule, overrides,
           code_fp (sha256 over the sorted .py files of modelVal,
           modelTraining and baselines, __pycache__ excluded; computed once
           per process -- final only after code freeze, other sessions edit
           some of those files), code_fp_files
  planner  (R3 M4) n_cycles, feasible_frac_mean, feasible_frac_p10,
           emergency_cycles, emergency_frac, emerg_collision_static,
           emerg_collision_forecast, emerg_boundary_only,
           emerg_boundary_dynamic, emerg_dynamic_only, emerg_unexplained,
           pen_depth_mean_m / pen_depth_max_m (selected candidate, emergency
           cycles with a collision), emerg_dyn_violation_mean/_max,
           emerg_boundary_excess_max_m, attrib_inconsistent (must be 0),
           cycles = [[t, n_valid, n_total, emergency 0/1, coll_kind -1/0/1,
           pen_depth_m], ...] one entry per replan.
           Emergency cause precedence: collision (by the kind of the deepest
           penetrating disk: static = observed-position disks, forecast =
           moving disks) > boundary (+dynamic -> emerg_boundary_dynamic) >
           dynamic only.
  floor    (R3 M5) floor_active_frac / floor_n: fraction of forecast-disk
           sigma values at lookahead 0.5..15 s (t=0 anchor excluded) that
           sigma_floor raised, pooled over the episode's cycles
           (floor_active_frac_cycle_mean: unweighted mean of the per-cycle
           fractions);
           branch_sigma_below_floor_{5,10,15}s / branch_sigma_n_{..}s
           (predictor arms): retained-branch isotropic sigma (after --iso,
           --sigma-scale) below the floor before flooring
  probe    safety_pool_max_p50/_p90/_max (+ _nz_p50/_nz_p90/n_nz over cycles
           with any risk) when --safety-ref-probe, else null
  timing   (R3 minor 2) cycle_ms_p50/_p99/_max and deadline_miss (cycles
           over 500 ms). cycle = wall time of predictor inference + the
           planner's replan (risk update + candidate sweep), measured on a
           SHARED CPU/GPU -- other processes run concurrently, so these are
           upper-bound-ish indications, not a real-time benchmark.
  safety   (R3 M6; 20 Hz = every engine tick incl. t=0 and the final tick)
           min_sep_20hz (and min_sep_engine, the engine's own 20 Hz
           accounting, as a cross-check);
           union20_events_{152,76,45}, union20_time_below_{152,76,45}_s,
           union20_event_list {"152"|"76"|"45": [event, ...]} with event =
             {slot, t_entry, t_min, t_exit, min_sep, time_below_s,
              lead_phys_s, lead_track_s, unavoidable_old, unavoidable_new};
           pairwise_encounters and pairwise = [segment, ...], one per
             contiguous tick run of ONE intruder's separation < 152.4 m:
             {slot, gt_type, gt_intent, track_id, t_entry, t_min, t_exit,
              min_sep, tb_152_s, tb_76_s, tb_45_s, lead_phys_s,
              lead_track_s, turning_at_track_start, turning_at_entry,
              reaction_active, unavoidable_old, unavoidable_new}
           nmac_events_avoidable_new + events_new_screen (aligned with the
             old 2 Hz 'events' list: {lead_track_s, unavoidable_new}).
           Definitions: lead_phys_s = closest approach - first-ever
           detection of the slot (the old screen); lead_track_s = threshold
           entry - first valid detection available by the as-of frame of
           the track open at entry; null for closed/missing/invalid tracks.
           unavoidable_old/new = unrounded lead < t_need
           (a null lead counts as unavoidable). turning_* = gt mode CT_LEFT
           or CT_RIGHT at the track's first detection / at the entry tick.
           reaction_active = the intruder's reaction layer LATCHED (the
           evasive turn is being blended in) at any tick in
           [t_entry - 15 s, t_exit]; REACTIVE_AVOID agents evade the ownship
           through their intent instead (see gt_intent == 4).
           For union events the responsible slot is the argmin at the
           minimum tick and lead_track_s is that slot's own 152.4 m entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

import numpy as np  # noqa: E402

from core.constants import (DWC_HMD_M, LOG_DECIMATION,  # noqa: E402
                            NMAC_RADIUS_M, SIM_DT)
from scenario.presets import config_for_episode  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402

from planner.controller import PlannerPilot  # noqa: E402
from planner.risk import make_risk  # noqa: E402

EP_KEYS = ("gt_pos", "gt_valid", "gt_type", "gt_intent",
           "obs_detected", "obs_pos", "obs_R", "obs_track_age", "own_pos",
           "obs_track_id")
BASE_SPLIT = "test_ood"          # default --preset (replay_case imports it)
PRESETS = ("test_ood", "test_id")
INDEX_OFFSET = 10_000
DURATION_S = 90.0
#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")
MODEL_PATH = f"{PRISM_ROOT}/Logs/Models/20260813_170429_full5x_s4/final/model_final.pt"
MODEL_ARMS = ("model", "mc_cvsig", "cvc_msig")

#: separation thresholds [m] of the 20 Hz endpoints: NMAC, half NMAC, severe
THRESHOLDS = ((NMAC_RADIUS_M, "152"), (76.2, "76"), (45.0, "45"))
T_NEED_S = math.sqrt(2.0 * NMAC_RADIUS_M / 4.5)     # ~8.2 s full-rate
DEADLINE_MS = 500.0              # one 2 Hz update period
PRED_LOG_EVERY = 4               # log forecasts every 4th replan (2 s)
REACT_LOOKBACK_S = 15.0          # reaction window before threshold entry
TURN_MODES = (1, 2)              # ManeuverMode.CT_LEFT / CT_RIGHT
PRED_T = 30                      # forecast steps logged (15 s at 0.5 s)
# Relative to THIS file (modelVal/experiments/batch.py), never absolute: a
# runner snapshot lives at another root (C:/PRISM/r3/code_<tag>/...), and an
# absolute root rewritten by the runner path patch pointed the fingerprint at
# a different, older tree there.
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FP_ROOTS = tuple(os.path.join(_CODE_ROOT, d).replace("\\", "/")
                 for d in ("modelVal", "modelTraining", "baselines"))


# ======================================================================
# provenance
# ======================================================================

def code_fingerprint(roots=FP_ROOTS) -> tuple[str, int]:
    """sha256 over (relative path, bytes) of every .py file under roots,
    in sorted path order; __pycache__ excluded. Returns (hex, n_files)."""
    files = []
    for root in roots:
        base = os.path.dirname(os.path.abspath(root))
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for fn in filenames:
                if fn.endswith(".py"):
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, base).replace("\\", "/")
                    files.append((rel, full))
    h = hashlib.sha256()
    for rel, full in sorted(files):
        h.update(rel.encode("utf-8") + b"\0")
        with open(full, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return h.hexdigest(), len(files)


# ======================================================================
# 20 Hz ground-truth recorder
# ======================================================================

class TickRecorder:
    """Ownship/intruder positions, reaction latch and manoeuvre mode at every
    engine tick. Read-only on the simulation (attribute reads only); the
    separation arithmetic is vectorized once at the end of the episode."""

    def __init__(self, scenario) -> None:
        self.scn = scenario
        self.t: list[float] = []
        self.own: list[tuple[float, float]] = []
        self.xs: list[list[float]] = []
        self.ys: list[list[float]] = []
        self.lat: list[list[bool]] = []
        self.mode: list[list[int]] = []

    def record(self, t: float) -> None:
        own = self.scn.ownship.state
        nan = math.nan
        xs, ys, lat, mode = [], [], [], []
        for ag in self.scn.agents:
            if ag.alive:
                st = ag.state
                xs.append(st.x)
                ys.append(st.y)
                r = ag.reaction
                lat.append(r is not None and r.is_latched)
                mode.append(ag.mode_code)
            else:
                xs.append(nan)
                ys.append(nan)
                lat.append(False)
                mode.append(-1)
        self.t.append(t)
        self.own.append((own.x, own.y))
        self.xs.append(xs)
        self.ys.append(ys)
        self.lat.append(lat)
        self.mode.append(mode)

    def arrays(self) -> dict:
        own = np.asarray(self.own, dtype=np.float64)              # [T,2]
        X = np.asarray(self.xs, dtype=np.float64)                 # [T,N]
        Y = np.asarray(self.ys, dtype=np.float64)
        sep = np.hypot(X - own[:, 0:1], Y - own[:, 1:2])
        sep = np.where(np.isfinite(sep), sep, np.inf)
        return {"t": np.asarray(self.t, dtype=np.float64), "sep": sep,
                "lat": np.asarray(self.lat, dtype=bool),
                "mode": np.asarray(self.mode, dtype=np.int16)}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index ranges of the contiguous True runs of a 1-D mask."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0].tolist(),
                    np.nonzero(d == -1)[0].tolist()))


class TrackBook:
    """Track lifecycles from the logged arrays (2 Hz), for lead_track_s."""

    def __init__(self, ep, n_logged: int) -> None:
        self.t = np.asarray(ep.t[:n_logged], dtype=np.float64)
        self.tid = np.asarray(ep.obs_track_id[:n_logged])
        self.det = np.asarray(ep.obs_detected[:n_logged], dtype=bool)
        self.mode = np.asarray(ep.gt_mode[:n_logged])

    def frame_at(self, t: float) -> int:
        """Latest logged frame with time <= t (-1 before the first frame)."""
        return int(np.searchsorted(self.t, t, side="right")) - 1

    def track_at_entry(self, slot: int, t_entry: float):
        """(track_id, first_frame, first_detection_s) of the open as-of
        track with a past valid detection; otherwise None. A current miss
        does not close an ID; closed IDs and future hits are never used."""
        i = self.frame_at(t_entry)
        if i < 0:
            return None
        col = self.tid[:i + 1, slot]
        if col[i] < 0:
            return None
        tid = int(col[i])
        rows = np.nonzero((col == tid) & self.det[:i + 1, slot])[0]
        if rows.size == 0:
            return None
        j = int(rows[0])
        return tid, j, float(self.t[j])

    def screen_at_entry(self, slot: int, t_entry: float) -> dict:
        """Read-only event provenance; timestamps below are not rounded."""
        i = self.frame_at(t_entry)
        trk = self.track_at_entry(slot, t_entry)
        lead = None if trk is None else float(t_entry) - trk[2]
        reason = ('before_first_observation' if i < 0 else
                  'closed_asof_track' if self.tid[i, slot] < 0 else
                  'open_track_without_past_detection' if trk is None else
                  'insufficient_lead' if lead < T_NEED_S else None)
        return {'screen_entry_s': float(t_entry), 'screen_threshold_m': NMAC_RADIUS_M,
                'asof_frame': None if i < 0 else i,
                'asof_time_s': None if i < 0 else float(self.t[i]),
                'asof_track_id': None if i < 0 else int(self.tid[i, slot]),
                'asof_detected': None if i < 0 else bool(self.det[i, slot]),
                'track_open_asof': bool(i >= 0 and self.tid[i, slot] >= 0),
                'first_detection_frame': None if trk is None else trk[1],
                'first_detection_s': None if trk is None else trk[2],
                'lead_track_raw_s': lead, 'exclusion_reason': reason}


def _r(v, nd):
    return None if v is None else round(float(v), nd)


def safety_endpoints(arr: dict, book: TrackBook, first_det: dict,
                     agents) -> dict:
    """R3 major 6 endpoints from the 20 Hz separation array."""
    tt, sep, lat, mode = arr["t"], arr["sep"], arr["lat"], arr["mode"]
    n_t, n_ag = sep.shape
    # Endpoint samples still determine counts/minima; only the intervals
    # [t_i, t_(i+1)) carry exposure. The last tick has zero time weight.
    dt = np.r_[np.diff(tt), 0.0] if n_t else np.empty(0)
    if not np.all(np.isfinite(tt)) or np.any(dt < 0):
        raise ValueError('scoring ticks must be finite and nondecreasing')
    back = int(round(REACT_LOOKBACK_S / SIM_DT))
    out: dict = {'screen_version': 'open_asof_v2',
                 'endpoint_precision_version': 'raw_minimum_v2',
                 'time_integration': 'left_tick_interval_v2',
                 'scoring_start_s': float(tt[0]) if n_t else None,
                 'scoring_end_s': float(tt[-1]) if n_t else None,
                 'exposure_s': float(tt[-1] - tt[0]) if n_t else 0.0}
    umin = sep.min(axis=1) if n_ag else np.full(n_t, np.inf)
    out["min_sep_20hz"] = _r(float(umin.min()), 1) if n_t else None

    # ---- pairwise encounters: one intruder's contiguous run below 152.4 m
    pair: list[dict] = []
    seg_of: dict[int, list[tuple[int, int, dict]]] = {}
    for slot in range(n_ag):
        col = sep[:, slot]
        for a, b in _runs(col < NMAC_RADIUS_M):
            seg = col[a:b]
            k = a + int(np.argmin(seg))
            t_entry, t_min, t_exit = tt[a], tt[k], tt[b - 1]
            det_t = first_det.get(slot)
            lead_phys = None if det_t is None else t_min - det_t
            trk = book.track_at_entry(slot, t_entry)
            lead_track = None if trk is None else t_entry - trk[2]
            audit = book.screen_at_entry(slot, t_entry)
            ag = agents[slot]
            s = {"slot": slot, "gt_type": int(ag.platform),
                 "gt_intent": int(ag.intent_code),
                 "track_id": None if trk is None else trk[0],
                 "t_entry": _r(t_entry, 2), "t_min": _r(t_min, 2),
                 "t_exit": _r(t_exit, 2), "min_sep": _r(seg[k - a], 1),
                 "min_sep_raw_m": float(seg[k - a])}
            for thr, lbl in THRESHOLDS:
                s[f"tb_{lbl}_s"] = _r(float(dt[a:b][seg < thr].sum()), 2)
                s[f"tb_{lbl}_legacy_tick_s"] = _r(float((seg < thr).sum()) * SIM_DT, 2)
            s.update({
                "lead_phys_s": _r(lead_phys, 2),
                "lead_track_s": _r(lead_track, 2),
                "lead_track_raw_s": None if lead_track is None else float(lead_track),
                "screen_audit": audit,
                "turning_at_track_start": (
                    None if trk is None
                    else bool(int(book.mode[trk[1], slot]) in TURN_MODES)),
                "turning_at_entry": bool(int(mode[a, slot]) in TURN_MODES),
                "reaction_active": bool(lat[max(0, a - back):b, slot].any()),
                "unavoidable_old": bool(lead_phys is None
                                        or lead_phys < T_NEED_S),
                "unavoidable_new": bool(lead_track is None
                                        or lead_track < T_NEED_S),
            })
            pair.append(s)
            seg_of.setdefault(slot, []).append((a, b, s))
    pair.sort(key=lambda s: (s["t_entry"], s["slot"]))
    out["pairwise_encounters"] = len(pair)

    # ---- union events (min over intruders) at every threshold, 20 Hz
    lists: dict[str, list[dict]] = {}
    for thr, lbl in THRESHOLDS:
        inside = umin < thr
        out[f"union20_time_below_{lbl}_s"] = _r(float(dt[inside].sum()), 2)
        out[f"union20_time_below_{lbl}_legacy_tick_s"] = _r(float(inside.sum()) * SIM_DT, 2)
        evs = []
        for a, b in _runs(inside):
            k = a + int(np.argmin(umin[a:b]))
            slot = int(np.argmin(sep[k]))
            seg = next((s for (sa, sb, s) in seg_of.get(slot, [])
                        if sa <= k < sb), None)
            det_t = first_det.get(slot)
            lead_phys = None if det_t is None else tt[k] - det_t
            lead_track = None if seg is None else seg["lead_track_raw_s"]
            evs.append({
                "slot": slot, "t_entry": _r(tt[a], 2), "t_min": _r(tt[k], 2),
                "t_exit": _r(tt[b - 1], 2), "min_sep": _r(umin[k], 1),
                "min_sep_raw_m": float(umin[k]),
                "time_below_s": _r(float(dt[a:b].sum()), 2),
                "time_below_legacy_tick_s": _r((b - a) * SIM_DT, 2),
                "lead_phys_s": _r(lead_phys, 2), "lead_track_s": _r(lead_track, 2),
                "lead_track_raw_s": lead_track,
                "screen_audit": None if seg is None else dict(seg["screen_audit"]),
                "unavoidable_old": bool(lead_phys is None
                                        or lead_phys < T_NEED_S),
                "unavoidable_new": bool(lead_track is None
                                        or lead_track < T_NEED_S)})
        out[f"union20_events_{lbl}"] = len(evs)
        lists[lbl] = evs
    out["union20_event_list"] = lists
    out["pairwise"] = pair
    return out


# ======================================================================
# per-episode summaries
# ======================================================================

def _pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=np.float64), q))


def planner_summary(diags: list[dict]) -> dict:
    """R3 major 4 decision-rule diagnostics over an episode's replans."""
    n = len(diags)
    out: dict = {"n_cycles": n}
    if n == 0:
        return out
    ff = [d["n_valid"] / d["n_total"] if d["n_total"] else 0.0 for d in diags]
    em = [d for d in diags if d["emergency"]]
    out["feasible_frac_mean"] = round(float(np.mean(ff)), 4)
    out["feasible_frac_p10"] = round(_pct(ff, 10), 4)
    out["emergency_cycles"] = len(em)
    out["emergency_frac"] = round(len(em) / n, 4)
    cause = {"emerg_collision_static": 0, "emerg_collision_forecast": 0,
             "emerg_boundary_only": 0, "emerg_boundary_dynamic": 0,
             "emerg_dynamic_only": 0, "emerg_unexplained": 0}
    pens, dyns, bexc = [], [], []
    inconsistent = 0
    for d in em:
        if not d.get("attrib_consistent", True):
            inconsistent += 1
        dyn = float(d.get("dyn_violation", 0.0))
        dyns.append(dyn)
        bexc.append(float(d.get("boundary_excess_m", 0.0)))
        if d.get("collision"):
            pens.append(float(d["pen_depth_m"]))
            cause["emerg_collision_static" if d["coll_kind"] == 0
                  else "emerg_collision_forecast"] += 1
        elif d.get("boundary"):
            cause["emerg_boundary_dynamic" if dyn > 0
                  else "emerg_boundary_only"] += 1
        elif dyn > 0:
            cause["emerg_dynamic_only"] += 1
        else:
            cause["emerg_unexplained"] += 1
    out.update(cause)
    out["pen_depth_mean_m"] = round(float(np.mean(pens)), 2) if pens else None
    out["pen_depth_max_m"] = round(float(np.max(pens)), 2) if pens else None
    out["emerg_dyn_violation_mean"] = (round(float(np.mean(dyns)), 3)
                                       if dyns else None)
    out["emerg_dyn_violation_max"] = (round(float(np.max(dyns)), 3)
                                      if dyns else None)
    out["emerg_boundary_excess_max_m"] = (round(float(np.max(bexc)), 2)
                                          if bexc else None)
    out["attrib_inconsistent"] = inconsistent
    return out


def floor_summary(diags: list[dict], predictor_arm: bool) -> dict:
    raised = sum(int(d["floor"][0]) for d in diags if "floor" in d)
    total = sum(int(d["floor"][1]) for d in diags if "floor" in d)
    per_cycle = [d["floor"][0] / d["floor"][1] for d in diags
                 if "floor" in d and d["floor"][1] > 0]
    out = {"floor_active_frac": round(raised / total, 5) if total else None,
           "floor_n": total,
           # unweighted mean of the per-cycle fractions (cycles that had
           # forecast disks); floor_active_frac above is pooled over values
           "floor_active_frac_cycle_mean": (round(float(np.mean(per_cycle)), 5)
                                            if per_cycle else None)}
    for h in (5, 10, 15):
        below = tot = 0
        for d in diags:
            bf = d.get("branch_floor")
            if bf:
                below += int(bf[h][0])
                tot += int(bf[h][1])
        out[f"branch_sigma_below_floor_{h}s"] = (
            round(below / tot, 5) if (predictor_arm and tot) else None)
        out[f"branch_sigma_n_{h}s"] = tot if predictor_arm else None
    return out


def probe_summary(diags: list[dict], probe: bool) -> dict:
    keys = ("safety_pool_max_p50", "safety_pool_max_p90",
            "safety_pool_max_max", "safety_pool_max_nz_p50",
            "safety_pool_max_nz_p90", "safety_pool_max_n_nz")
    if not probe:
        return {k: None for k in keys}
    v = np.asarray([d["safety_pool_max"] for d in diags
                    if "safety_pool_max" in d], dtype=np.float64)
    nz = v[v > 0]
    f = lambda a, q: (round(float(np.percentile(a, q)), 6)  # noqa: E731
                      if a.size else None)
    return {"safety_pool_max_p50": f(v, 50), "safety_pool_max_p90": f(v, 90),
            "safety_pool_max_max": (round(float(v.max()), 6)
                                    if v.size else None),
            "safety_pool_max_nz_p50": f(nz, 50),
            "safety_pool_max_nz_p90": f(nz, 90),
            "safety_pool_max_n_nz": int(nz.size)}


def timing_summary(cycle_ms: list[float]) -> dict:
    if not cycle_ms:
        return {"cycle_ms_p50": None, "cycle_ms_p99": None,
                "cycle_ms_max": None, "deadline_miss": 0}
    c = np.asarray(cycle_ms, dtype=np.float64)
    return {"cycle_ms_p50": round(float(np.percentile(c, 50)), 1),
            "cycle_ms_p99": round(float(np.percentile(c, 99)), 1),
            "cycle_ms_max": round(float(c.max()), 1),
            "deadline_miss": int((c > DEADLINE_MS).sum())}


# ======================================================================
# closed-loop prediction log (R3 minor 5)
# ======================================================================

def pred_log_path(log_dir: str, arm: str, index: int) -> str:
    """<log_dir>/<arm>_<index:05d>.npz.

    The name depends only on the arm and the ABSOLUTE episode index (the
    loop index i, i.e. --start + k, never a shard-local counter), and
    '{:05d}' is injective on non-negative integers (a 6-digit index simply
    widens), so shards over DISJOINT index ranges can share one --log-preds
    directory without collision. Overlapping ranges, or two conditions of
    the same arm pointed at one directory, WOULD collide: main() refuses to
    start when a target file exists, and save_pred_log creates the file
    exclusively, so even a concurrent overlap fails loudly instead of
    silently replacing a forecast log."""
    return os.path.join(log_dir, f"{arm}_{int(index):05d}.npz")


REACTIVE_AVOID = 4               # IntentType.REACTIVE_AVOID


def save_pred_log(path: str, entries: list[dict], K: int, kind: str,
                  ep, n_logged: int, arr: dict, meta: dict,
                  own: dict, ref: dict, overwrite: bool = False) -> int:
    """One compressed npz per episode; returns the number of forecasts.

    Per forecast [P]: t0 (log index), t_s, slot, mu/sigma/rho/pi/theta,
    gt/gt_valid (30 GT steps after t0), gt_type, gt_intent (at t0),
    reaction_latched (the intruder's reaction layer LATCHED at any 20 Hz
    tick in [t, t+15 s]), intent_reactive (gt_intent == REACTIVE_AVOID at
    t), own_intervening (the selected plan at t leaves the nominal -- kept
    for transparency; its fixed definition flags most forecasts, so the
    aggregator derives the ownship stratum from the raw trajectory below).
    Ownship at every logged frame [F] (2 Hz, incl. the final frame):
    own_t, own_x, own_y, own_d (signed lateral offset from the planner's
    reference path, + = left), own_d_nominal (signed lateral offset from
    the ownship's nominal straight ground track own_track, + = left),
    own_v (the ownship state speed -- the variable the planner's
    desired_speed refers to), own_psi.
    Scalars: v_desired, corridor_width, lateral_step, hard_radius,
    ref_kind ('corridor' | 'nominal_track'), own_d_t0 (own_d of the
    ownship's start position: the scenario places the nominal track off
    the corridor centreline); own_track [x0, y0, vgx, vgy]; reference
    polyline ref_x/ref_y.
    meta_json: the run's knobs (condition + code_fp) so a directory that
    mixes conditions can be detected by the aggregator."""
    P = len(entries)
    f32 = np.float32
    mu = np.full((P, K, PRED_T, 2), np.nan, f32)
    sigma = np.full((P, K, PRED_T, 2), np.nan, f32)
    rho = np.zeros((P, K, PRED_T), f32)
    pi = np.zeros((P, K), f32)
    gt = np.full((P, PRED_T, 2), np.nan, f32)
    gt_valid = np.zeros((P, PRED_T), bool)
    t0s = np.zeros(P, np.int32)
    t_s = np.zeros(P, f32)
    slots = np.zeros(P, np.int32)
    theta = np.zeros(P, f32)
    own_int = np.zeros(P, bool)
    latched = np.zeros(P, bool)
    gtype = np.full(P, -1, np.int8)
    gintent = np.full(P, -1, np.int8)
    lat, tt = arr["lat"], arr["t"]
    fwd = int(round(REACT_LOOKBACK_S / SIM_DT))
    for p, e in enumerate(entries):
        k = e["mu"].shape[0]
        mu[p, :k] = e["mu"]
        sigma[p, :k] = e["sigma"]
        rho[p, :k] = e["rho"]
        pi[p, :k] = e["pi"]
        t0s[p], t_s[p], slots[p] = e["t0"], e["t"], e["slot"]
        theta[p] = e["theta"]
        own_int[p] = e["own_intervening"]
        rows = e["t0"] + 1 + np.arange(PRED_T)
        ok = rows < n_logged
        gt[p, ok] = ep.gt_pos[rows[ok], e["slot"]]
        gt_valid[p, ok] = ep.gt_valid[rows[ok], e["slot"]]
        it = int(np.searchsorted(tt, e["t"] - 1e-6))       # tick of forecast
        latched[p] = bool(lat[it:it + fwd + 1, e["slot"]].any())
        gtype[p] = int(ep.gt_type[e["slot"]])
        gintent[p] = int(ep.gt_intent[e["t0"], e["slot"]])
    intent_reactive = gintent == REACTIVE_AVOID
    payload = dict(
        t0=t0s, t_s=t_s, slot=slots, mu=mu, sigma=sigma, rho=rho,
        pi=pi, theta=theta, gt=gt, gt_valid=gt_valid,
        gt_type=gtype, gt_intent=gintent,
        reaction_latched=latched, intent_reactive=intent_reactive,
        own_intervening=own_int,
        own_t=np.asarray(own["t"], f32), own_x=np.asarray(own["x"], f32),
        own_y=np.asarray(own["y"], f32), own_d=np.asarray(own["d"], f32),
        own_d_nominal=np.asarray(own["d_nominal"], f32),
        own_v=np.asarray(own["v"], f32), own_psi=np.asarray(own["psi"], f32),
        own_track=np.asarray(ref["own_track"], np.float64),
        own_d_t0=np.float64(ref["own_d_t0"]),
        v_desired=np.float64(ref["v_desired"]),
        corridor_width=np.float64(ref["corridor_width"]),
        lateral_step=np.float64(ref["lateral_step"]),
        hard_radius=np.float64(ref["hard_radius"]),
        ref_kind=np.array(ref["ref_kind"]),
        ref_x=np.asarray(ref["x"], np.float64),
        ref_y=np.asarray(ref["y"], np.float64),
        kind=np.array(kind), arm=np.array(meta["arm"]),
        index=np.int32(meta["index"]), preset=np.array(meta["preset"]),
        track_rule=np.array(meta["track_rule"]),
        meta_json=np.array(meta.get("meta_json", "{}")))
    # exclusive create: a concurrent writer of the same (arm, index) --
    # overlapping shards -- gets FileExistsError instead of a silent replace
    with open(path, "wb" if overwrite else "xb") as fh:
        np.savez_compressed(fh, **payload)
    return P


# ======================================================================
# one closed-loop episode
# ======================================================================

def run_episode(arm: str, index: int, predictor,
                overrides: dict | None = None,
                risk_kw: dict | None = None, *,
                preset: str = BASE_SPLIT,
                pilot_kw: dict | None = None,
                model_kw: dict | None = None,
                probe_safety: bool = False,
                pred_log_dir: str | None = None,
                meta: dict | None = None,
                overwrite: bool = False) -> dict:
    """Run one paired episode. ``risk_kw`` goes to make_risk (every arm),
    ``pilot_kw`` to PlannerPilot, ``model_kw`` only labels the legacy
    sigma_scale/pi_min/iso/gain_mode row fields, ``meta`` is copied onto the
    row (knobs, provenance). ``overwrite`` only lets the forecast log replace
    an existing npz."""
    cfg = config_for_episode(preset, INDEX_OFFSET + index)
    over = dict(overrides or {})
    enc = {k: over.pop(k) for k in ("n_designed_min", "n_designed_max")
           if k in over}
    if enc:
        from dataclasses import replace as dc_replace
        over["encounter"] = dc_replace(cfg.encounter, **enc)
    cfg = cfg.copy_with(duration_s=DURATION_S,
                        name=f"batch/{arm}/{index:05d}", **over)

    pilot = PlannerPilot(make_risk(arm, **(risk_kw or {})),
                         **(pilot_kw or {}))
    engine = SimulationEngine(cfg, ownship_controller=pilot.command)
    scn = engine.scenario
    pilot.bind(scn.world, scn.ownship.envelope, scn.ownship_track)
    ep = engine.ep
    mkw = model_kw if model_kw is not None else (risk_kw or {})

    sep_series: list[np.ndarray] = []      # per logged frame: [N] gt dist
    own_xy: list[tuple[float, float]] = []
    omega_series: list[float] = []
    v_series: list[float] = []
    plan_ms: list[float] = []
    infer_ms: list[float] = []
    first_det: dict[int, float] = {}       # slot -> sim t of first detection
    t_series: list[float] = []
    # review-round additions (read-only on the simulation)
    rec = TickRecorder(scn)
    diags: list[dict] = []
    cycles: list[list] = []
    cycle_ms: list[float] = []
    plog: list[dict] | None = (
        [] if (pred_log_dir and arm != "none") else None)
    # ownship at every logged frame, only for the forecast log (read-only)
    own_log: list[tuple[float, float, float, float, float]] = []

    wall0 = time.perf_counter()
    rec.record(engine.t)                   # t = 0, before the first tick
    own_xy0 = (scn.ownship.state.x, scn.ownship.state.y)
    while engine.step(log=True):
        rec.record(engine.t)
        if engine._tick % LOG_DECIMATION != 0:
            continue
        t0 = engine._log_index - 1
        if t0 < 0:
            continue
        ep_dict = {k: getattr(ep, k) for k in EP_KEYS}

        t_cyc = time.perf_counter()
        preds = {}
        if predictor is not None and predictor.loaded:
            t_inf = time.perf_counter()
            preds = predictor.predict_frame(ep_dict, t0)
            infer_ms.append((time.perf_counter() - t_inf) * 1000.0)

        pilot.replan(engine.t, ep_dict, t0, preds, scn.ownship.state)
        cycle_ms.append((time.perf_counter() - t_cyc) * 1000.0)
        plan_ms.append(pilot.plan_ms)

        d = pilot.cycle_diagnostics(probe_safety)
        diags.append(d)
        cycles.append([round(engine.t, 2), d.get("n_valid"), d.get("n_total"),
                       int(bool(d.get("emergency"))), int(d.get("coll_kind", -1)),
                       round(float(d.get("pen_depth_m", 0.0)), 2)])
        if plog is not None and t0 % PRED_LOG_EVERY == 0:
            own_int = bool(d.get("own_intervening", False))
            if arm in MODEL_ARMS:
                for slot, p in preds.items():
                    plog.append({"t0": t0, "t": engine.t, "slot": int(slot),
                                 "mu": p["mu_w"], "sigma": p["sigma"],
                                 "rho": p["rho"], "pi": p["pi"],
                                 "theta": float(p["theta"]),
                                 "own_intervening": own_int})
            elif hasattr(pilot.risk, "forecast_log"):
                for slot, mu, sig in pilot.risk.forecast_log():
                    plog.append({"t0": t0, "t": engine.t, "slot": slot,
                                 "mu": mu[None],
                                 "sigma": np.stack([sig, sig], -1)[None],
                                 "rho": np.zeros((1, PRED_T)),
                                 "pi": np.ones(1), "theta": 0.0,
                                 "own_intervening": own_int})

        own = scn.ownship.state
        dists = np.full(len(scn.agents), np.inf)
        for slot, ag in enumerate(scn.agents):
            if not ag.alive:
                continue
            dists[slot] = math.hypot(ag.state.x - own.x, ag.state.y - own.y)
            if bool(ep.obs_detected[t0, slot]) and slot not in first_det:
                first_det[slot] = engine.t
        sep_series.append(dists)
        own_xy.append((own.x, own.y))
        omega_series.append(abs(math.degrees(own.omega)))
        v_series.append(own.v)
        t_series.append(engine.t)
        if plog is not None:
            own_log.append((engine.t, own.x, own.y, own.v, own.psi))
    rec.record(engine.t)                   # the final tick ends the loop
    if plog is not None:                   # the final logged frame (90.0 s)
        st = scn.ownship.state
        own_log.append((engine.t, st.x, st.y, st.v, st.psi))

    wall_s = time.perf_counter() - wall0
    sep = np.asarray(sep_series)                      # [F, N]
    ts = np.asarray(t_series)
    min_by_frame = sep.min(axis=1)
    log_dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.5

    # ---- NMAC events: contiguous frames below the ring, per nearest slot
    events = []
    ev_frames = []                                    # (i, k, slot) per event
    inside = min_by_frame < NMAC_RADIUS_M
    i = 0
    t_need = math.sqrt(2.0 * NMAC_RADIUS_M / 4.5)     # ~8.2 s full-rate
    while i < len(inside):
        if not inside[i]:
            i += 1
            continue
        j = i
        while j < len(inside) and inside[j]:
            j += 1
        k = int(np.argmin(min_by_frame[i:j])) + i
        slot = int(np.argmin(sep[k]))
        det_t = first_det.get(slot)
        lead = (ts[k] - det_t) if det_t is not None else 0.0
        events.append({
            "t": round(float(ts[k]), 1),
            "slot": slot,
            "min_sep": round(float(min_by_frame[k]), 1),
            "duration_s": round((j - i) * log_dt, 1),
            "detect_lead_s": round(float(lead), 1),
            "unavoidable": bool(lead < t_need),
        })
        ev_frames.append((i, k, slot))
        i = j

    # ---- path economy: flown length vs corridor progress
    xy = np.asarray(own_xy)
    path_len = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 \
        else 0.0
    progress = 0.0
    if pilot.path is not None and len(xy) > 1:
        progress = (pilot.path.project(*xy[-1])["s"]
                    - pilot.path.project(*xy[0])["s"])
    elongation = (path_len / progress - 1.0) if progress > 100.0 else None

    # ---- LoWC events (same contiguous-segment rule as NMAC)
    lowc_inside = min_by_frame < DWC_HMD_M
    lowc_events = 0
    i = 0
    while i < len(lowc_inside):
        if lowc_inside[i]:
            lowc_events += 1
            while i < len(lowc_inside) and lowc_inside[i]:
                i += 1
        else:
            i += 1

    # ---- review round: 20 Hz safety endpoints + new-screen for 2 Hz events
    n_logged = int(engine._log_index)
    book = TrackBook(ep, n_logged)
    arr = rec.arrays()
    safety = safety_endpoints(arr, book, first_det, scn.agents)
    ev_new = []
    for (i0, k, slot) in ev_frames:
        f = k                                   # the slot's own 2 Hz entry
        while f - 1 >= i0 and sep[f - 1, slot] < NMAC_RADIUS_M:
            f -= 1
        trk = book.track_at_entry(slot, float(ts[f]))
        lt = None if trk is None else float(ts[f]) - trk[2]
        ev_new.append({"lead_track_s": _r(lt, 2),
                       "lead_track_raw_s": lt,
                       "screen_audit": book.screen_at_entry(slot, float(ts[f])),
                       "unavoidable_new": bool(lt is None or lt < T_NEED_S)})

    pred_logged = None
    if plog is not None:
        if arm in MODEL_ARMS:
            K = (int(plog[0]["mu"].shape[0]) if plog
                 else int(getattr(predictor, "info", {}).get("K", 3)))
            kind = "mixture"
        else:
            K, kind = 1, "gaussian"
        m = dict(meta or {})
        ol = np.asarray(own_log, dtype=np.float64).reshape(-1, 5)
        own_d = [pilot.path.project(x, y)["d"] for x, y in ol[:, 1:3]]
        # the scenario places the ownship's nominal (undisturbed) ground
        # track at a lateral OFFSET from the corridor centreline, so own_d
        # carries that placement offset; own_d_nominal is the signed offset
        # from the nominal straight track itself (+ = left of it)
        x0, y0, vgx, vgy = (float(v) for v in scn.ownship_track)
        un = math.hypot(vgx, vgy)
        ux, uy = ((vgx / un, vgy / un) if un > 1e-9 else (1.0, 0.0))
        own_d_nom = ux * (ol[:, 2] - y0) - uy * (ol[:, 1] - x0)
        pcfg = pilot.cfg
        corridors = getattr(scn.world, "corridors", None) or []
        ref = {"x": pilot.path.x, "y": pilot.path.y,
               "v_desired": pcfg.desired_speed,
               "corridor_width": pcfg.corridor_width,
               "lateral_step": pcfg.lateral_step,
               "hard_radius": pilot.risk.hard_radius,
               "ref_kind": "corridor" if corridors else "nominal_track",
               "own_track": (x0, y0, vgx, vgy),
               "own_d_t0": pilot.path.project(*own_xy0)["d"]}
        pred_logged = save_pred_log(
            pred_log_path(pred_log_dir, arm, index), plog, K,
            kind, ep, n_logged, arr,
            {"arm": arm, "index": index, "preset": preset,
             "track_rule": m.get("track_rule", ""),
             "meta_json": json.dumps(m, sort_keys=True)},
            {"t": ol[:, 0], "x": ol[:, 1], "y": ol[:, 2], "d": own_d,
             "d_nominal": own_d_nom, "v": ol[:, 3], "psi": ol[:, 4]}, ref,
            overwrite=overwrite)

    om = np.asarray(omega_series)
    row = {
        "arm": arm,
        "index": index,
        # written on every row so a batch is self-describing: earlier runs
        # carry no such field and are all sigma_scale == 1.0
        "sigma_scale": float(mkw.get("sigma_scale", 1.0)),
        "pi_min": float(mkw.get("pi_min", 0.05)),
        "iso": str(mkw.get("iso", "rms")),
        "gain_mode": str(mkw.get("gain_mode", "pi")),
        "seed": int(cfg.seed),
        "n_agents": int(sum(1 for a in scn.agents)),
        "sim_s": round(float(ts[-1]), 1) if len(ts) else 0.0,
        "wall_s": round(wall_s, 1),
        "min_sep": round(float(min_by_frame.min()), 1),
        "sep_p5": round(float(np.percentile(min_by_frame, 5)), 1),
        "sep_median": round(float(np.median(min_by_frame)), 1),
        "nmac_events": len(events),
        "nmac_events_avoidable": sum(1 for e in events
                                     if not e["unavoidable"]),
        "nmac_duration_s": round(float(inside.sum() * log_dt), 1),
        "lowc_fraction": round(float((min_by_frame < DWC_HMD_M).mean()), 3),
        "lowc_events": lowc_events,
        "path_len_m": round(path_len, 0),
        "corridor_progress_m": round(float(progress), 0),
        "path_elongation": (round(elongation, 4)
                            if elongation is not None else None),
        "abs_omega_mean": round(float(om.mean()), 2),
        "abs_omega_p95": round(float(np.percentile(om, 95)), 2),
        "v_mean": round(float(np.mean(v_series)), 1),
        "v_std": round(float(np.std(v_series)), 2),
        "plan_ms_mean": round(float(np.mean(plan_ms)), 1),
        "plan_ms_p95": round(float(np.percentile(plan_ms, 95)), 1),
        "infer_ms_mean": (round(float(np.mean(infer_ms)), 1)
                          if infer_ms else None),
        "events": events,
    }
    # ------------------------------------------------ review-round fields
    row.update(meta or {})
    row["preset"] = preset
    row["nmac_events_avoidable_new"] = sum(1 for e in ev_new
                                           if not e["unavoidable_new"])
    row["events_new_screen"] = ev_new
    row["min_sep_engine"] = round(float(engine.min_separation_m), 1)
    row.update({k: v for k, v in safety.items()
                if k not in ("union20_event_list", "pairwise")})
    row.update(planner_summary(diags))
    row.update(floor_summary(diags, arm in MODEL_ARMS))
    row.update(probe_summary(diags, probe_safety))
    row.update(timing_summary(cycle_ms))
    row["pred_logged"] = pred_logged
    row["union20_event_list"] = safety["union20_event_list"]
    row["pairwise"] = safety["pairwise"]
    row["cycles"] = cycles
    return row


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("none", "cv", "model", "mc_cvsig", "cvc_msig"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing non-empty <arm>.jsonl "
                         "and existing --log-preds npz files (Logs/ holds the "
                         "paper's raw data: CLAUDE.md rule 5)")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--n-min", type=int, default=None)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--designed-min", type=int, default=None)
    ap.add_argument("--designed-max", type=int, default=None)
    ap.add_argument("--preset", default=BASE_SPLIT, choices=PRESETS,
                    help="scenario split of config_for_episode; indices stay "
                         "INDEX_OFFSET + i (R3 major 8: matched ID/OOD)")
    ap.add_argument("--track-rule", default="current",
                    choices=("current", "slot"),
                    help="current: arms use only the track open at t0 (the "
                         "stated tracking model, R3 major 3); slot: legacy "
                         "oracle re-association (reproduces pre-review runs)")
    ap.add_argument("--allow-rule-mismatch", action="store_true",
                    help="let a predictor arm run a checkpoint whose track "
                         "rule differs from --track-rule")
    ap.add_argument("--hard-geom", default="arm", choices=("arm", "shared"),
                    help="arm: hard collision check on the arm's own disks "
                         "(original); shared: one arm-independent geometry "
                         "(observed positions at t0) for the hard check, arm "
                         "disks for the soft cost (R3 major 4 control)")
    ap.add_argument("--sigma-floor", type=float, default=10.0,
                    help="floor on every disk sigma, all arms (R3 major 5)")
    ap.add_argument("--plateau-scale", type=float, default=1.0,
                    help="soft_standoff = NMAC radius x this (1.0 = 152.4 m)")
    ap.add_argument("--w-safety", type=float, default=12.0,
                    help="planner weight_safety (12.0 = frozen planner)")
    ap.add_argument("--safety-norm", default="pool",
                    choices=("pool", "absolute"),
                    help="pool: safety / pool max (original); absolute: "
                         "safety / --safety-ref")
    ap.add_argument("--safety-ref", type=float, default=None,
                    help="fixed safety divisor for --safety-norm absolute "
                         "(choose it from a --safety-ref-probe pilot)")
    ap.add_argument("--safety-ref-probe", action="store_true",
                    help="record per-cycle pool max of raw safety cost "
                         "(no behaviour change)")
    ap.add_argument("--log-preds", default=None, metavar="DIR",
                    help="write closed-loop forecasts + GT futures, one npz "
                         "per episode (R3 minor 5)")
    ap.add_argument("--sigma-scale", type=float, default=1.0,
                    help="post-hoc temperature applied as sigma -> T*sigma "
                         "in the model arm (1.0 = raw network sigma; 1.665 = "
                         "the value fitted by experiments/recalibrate.py on "
                         "the OOD-adjacent calibration runs)")
    ap.add_argument("--pi-min", type=float, default=0.05,
                    help="branch-pruning threshold on mixture weight "
                         "(0.05 = the untested engineering value the paper "
                         "reports; 0 keeps every branch). Reviewer 2 VI.")
    ap.add_argument("--iso", default="rms", choices=("rms", "max", "min"),
                    help="how (sigma_x, sigma_y) is reduced to the planner's "
                         "single isotropic scale: rms == sqrt(trace/2) (the "
                         "paper's choice), max == the disk that bounds the "
                         "1-sigma ellipse, min == the optimistic opposite")
    ap.add_argument("--gain-mode", default="pi", choices=("pi", "uniform"),
                    help="pi == gain_k scales with branch probability (the "
                         "paper's choice); uniform == every retained branch "
                         "carries the full base gain (risk-averse worst-case "
                         "aggregation)")
    args = ap.parse_args()

    if args.sigma_scale != 1.0 and args.arm not in MODEL_ARMS:
        ap.error("--sigma-scale needs an arm that consumes the predictor")
    for flag, val, default in (("--pi-min", args.pi_min, 0.05),
                               ("--iso", args.iso, "rms"),
                               ("--gain-mode", args.gain_mode, "pi")):
        if val != default and args.arm not in MODEL_ARMS:
            ap.error(f"{flag} needs an arm that consumes the predictor")
    if args.safety_norm == "absolute" and not (
            args.safety_ref is not None and args.safety_ref > 0):
        ap.error("--safety-norm absolute needs --safety-ref > 0")
    if args.safety_ref is not None and args.safety_norm != "absolute":
        ap.error("--safety-ref only applies with --safety-norm absolute")
    for flag, val in (("--sigma-floor", args.sigma_floor),
                      ("--plateau-scale", args.plateau_scale),
                      ("--w-safety", args.w_safety)):
        if not (val >= 0.0 and math.isfinite(val)):
            ap.error(f"{flag} must be a finite value >= 0")

    model_kw = {}
    if args.arm in MODEL_ARMS:
        model_kw = {"sigma_scale": args.sigma_scale, "pi_min": args.pi_min,
                    "iso": args.iso}
        if args.arm != "cvc_msig":        # cvc_msig has no branch gains
            model_kw["gain_mode"] = args.gain_mode
    # every arm gets the rule and the floor (defaults == pre-review values)
    risk_kw = {**model_kw, "track_rule": args.track_rule,
               "sigma_floor": args.sigma_floor}
    pilot_kw = {"plateau_scale": args.plateau_scale,
                "w_safety": args.w_safety, "safety_norm": args.safety_norm,
                "safety_ref": args.safety_ref, "hard_geom": args.hard_geom}

    overrides = {}
    if args.density is not None:
        overrides["density_per_km2"] = args.density
    if args.n_min is not None:
        overrides["n_agents_min"] = args.n_min
    if args.n_max is not None:
        overrides["n_agents_max"] = args.n_max
    if args.designed_min is not None:
        overrides["n_designed_min"] = args.designed_min
    if args.designed_max is not None:
        overrides["n_designed_max"] = args.designed_max

    out_path = os.path.join(args.out, f"{args.arm}.jsonl")
    if (os.path.exists(out_path) and os.path.getsize(out_path) > 0
            and not args.overwrite):
        sys.exit(f"[{args.arm}] refusing to overwrite {out_path} "
                 f"(pass --overwrite if that is really intended)")

    predictor = None
    info: dict = {}
    caught: list = []
    # Every arm that reads the mixture needs the predictor loaded, not just
    # 'model'. Without this the decomposition arms silently degrade: mc_cvsig
    # receives no branches and falls back to the static disks of the 'none'
    # arm, and cvc_msig keeps CV's own sigma and reproduces the 'cv' arm
    # exactly. Both failures are invisible in the output -- the batch runs
    # clean and writes plausible rows -- so the guard is spelled out here.
    if args.arm in MODEL_ARMS:
        import warnings
        # torch/model-build warnings are held back and re-emitted AFTER the
        # banner, so the banner is the first line of a combined log
        with warnings.catch_warnings(record=True) as caught:
            from runtime.predictor import LivePredictor
            predictor = LivePredictor()
            info = predictor.load(args.model)

    fp, n_fp = code_fingerprint()
    knobs = {
        "preset": args.preset, "track_rule": args.track_rule,
        "allow_rule_mismatch": bool(args.allow_rule_mismatch),
        "hard_geom": args.hard_geom, "sigma_floor": args.sigma_floor,
        "plateau_scale": args.plateau_scale, "w_safety": args.w_safety,
        "safety_norm": args.safety_norm, "safety_ref": args.safety_ref,
        "safety_ref_probe": bool(args.safety_ref_probe),
        "model_path": info.get("path"),
        "model_track_rule": info.get("track_rule"),
        "overrides": dict(overrides),
        "code_fp": fp, "code_fp_files": n_fp,
    }
    mismatch = bool(predictor is not None
                    and info.get("track_rule") != args.track_rule)
    # FIRST log line: everything that decides what this process measures
    print(f"[{args.arm}] BATCH arm={args.arm} track_rule={args.track_rule} "
          f"model={info.get('path')} model_rule={info.get('track_rule')} "
          f"device={info.get('device')} "
          f"{'RULE-MISMATCH ' if mismatch else ''}"
          f"preset={args.preset} hard_geom={args.hard_geom} "
          f"sigma_floor={args.sigma_floor} plateau_scale={args.plateau_scale} "
          f"w_safety={args.w_safety} safety_norm={args.safety_norm} "
          f"safety_ref={args.safety_ref} probe={args.safety_ref_probe} "
          f"sigma_scale={args.sigma_scale} pi_min={args.pi_min} "
          f"iso={args.iso} gain_mode={args.gain_mode} "
          f"overrides={json.dumps(overrides, sort_keys=True)} "
          f"episodes={args.start}..{args.start + args.episodes - 1} "
          f"log_preds={args.log_preds} out={out_path} code_fp={fp[:16]}",
          flush=True)
    for w in caught:
        print(f"[{args.arm}] model-load warning: {w.category.__name__}: "
              f"{w.message}", flush=True)
    if mismatch and not args.allow_rule_mismatch:
        sys.exit(f"[{args.arm}] track-rule mismatch: checkpoint "
                 f"{info.get('path')} was trained under "
                 f"'{info.get('track_rule')}' but --track-rule is "
                 f"'{args.track_rule}'. Use a matching checkpoint, or pass "
                 f"--allow-rule-mismatch for a deliberate control.")

    if args.log_preds and args.arm != "none" and not args.overwrite:
        # shards over disjoint index ranges never share a file name (see
        # pred_log_path); an existing target means an overlapping range or a
        # second condition of this arm in the same directory -- refuse
        clash = [pred_log_path(args.log_preds, args.arm, i)
                 for i in range(args.start, args.start + args.episodes)
                 if os.path.exists(pred_log_path(args.log_preds, args.arm, i))]
        if clash:
            sys.exit(f"[{args.arm}] refusing to overwrite {len(clash)} "
                     f"forecast log(s) in {args.log_preds}, e.g. {clash[0]} "
                     f"(overlapping shard or another condition of this arm; "
                     f"use a new --log-preds dir or --overwrite)")

    os.makedirs(args.out, exist_ok=True)
    if args.log_preds:
        os.makedirs(args.log_preds, exist_ok=True)

    done = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(args.start, args.start + args.episodes):
            try:
                row = run_episode(args.arm, i, predictor, overrides, risk_kw,
                                  preset=args.preset, pilot_kw=pilot_kw,
                                  model_kw=model_kw,
                                  probe_safety=args.safety_ref_probe,
                                  pred_log_dir=args.log_preds, meta=knobs,
                                  overwrite=args.overwrite)
            except Exception as e:  # noqa: BLE001 - keep the batch alive
                row = {"arm": args.arm, "index": i,
                       "error": f"{type(e).__name__}: {e}", **knobs}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 10 == 0 or done == args.episodes:
                print(f"[{args.arm}] {done}/{args.episodes} episodes",
                      flush=True)
    print(f"[{args.arm}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
