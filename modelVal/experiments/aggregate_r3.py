"""Aggregate review-round (Reviewer 3) planner batches into one JSON.

  python -m experiments.aggregate_r3 --runs [LABEL=]PATH[,PATH...] ... \\
         [--preds LABEL=DIR ...] [--own-ref reference|nominal|at_t] \\
         [--own-dev-m 30] [--own-vfrac 0.9] [--preds-counts-only] \\
         --out OUT.json [--force]

Inputs
  --runs   one entry per condition: <arm>.jsonl files written by
           experiments/batch.py. Several comma-separated files are merged
           by episode index (index shards, DECISIONS D5.4); a duplicated
           index keeps the first row and is counted. LABEL defaults to
           "<parent dir>/<arm>". Error rows are counted and dropped.
  --preds  LABEL=DIR of a --log-preds directory; the files
           <arm>_<index>.npz of the label's arm are scored (closed-loop
           coverage, R3 minor 5). A directory whose files record different
           run conditions (meta_json, code_fp ignored) is refused.
  --own-ref, --own-dev-m, --own-vfrac   definition and parameters of the
           ownship stratum (below).
  --preds-counts-only  report the stratum counts and skip every coverage
           computation, so the split can be judged for degeneracy before
           any outcome is looked at.
  --out    output JSON; an existing file is refused unless --force.
Pairing: every comparison is restricted to the episode indices present in
both conditions whose rows describe the same traffic (equal seed and
n_agents, and equal preset/overrides where both rows record them); shared
indices that fail this are counted as unpaired_traffic_mismatch (e.g. a
test_id run against a test_ood run). Rows written before the review round
lack the 20 Hz / new-screen fields; endpoints that need them are null.

Definitions
  exposure_hours = sum(exposure_s) / 3600 for corrected rows. Legacy rows
  lacking exposure_s retain sim_s, with explicit fallback provenance.
  Event families (per-episode counts):
    union2    the original 2 Hz union NMAC events (row 'events'), 152.4 m
    union20   20 Hz union events (min over intruders), thresholds 152/76/45
    pairwise  20 Hz per-intruder encounters (runs below 152.4 m) whose
              min_sep_raw_m is below the threshold (152/76/45), falling
              back to displayed min_sep only in legacy records without raw
    *_severe  152.4 m union encounters whose minimum is below 45 m;
              union20_45 counts direct below-45 m runs, NOT this family
  Screens: all = unscreened; old = drop events with lead_phys_s < t_need
  (first-ever detection -> closest approach: the paper's screen); new = drop
  events with lead_track_s < t_need (current-track detection -> threshold
  entry). t_need = sqrt(2*152.4/4.5) s. A null lead counts as unavoidable.
  Times below threshold [s]: union20_tb_{thr}, pairwise_tb_{thr} (summed
  over encounters, so overlaps with different intruders count twice),
  union2_tb_152 (= the old nmac_duration_s, 2 Hz).
  Strata (pairwise encounters and union20 152 m events, the latter mapped
  to the responsible intruder's encounter): class HARD (gt_type Bird=0 or
  Unknown=6, or gt_intent SearchWander=3) / EASY; turning_at_track_start
  yes / no / unknown (never tracked before entry); reaction_active yes / no.

Output schema (JSON)
  {"schema": "aggregate_r3/2", "created", "seed", "B", "mc_draws",
   "t_need_s", "preds_counts_only", "elapsed_s",
   "inputs":  {LABEL: {paths, episodes, errors, duplicates, arm,
                        knobs: {field: [distinct values]}, code_fp: [...]}},
   "per_arm": {LABEL: {
       episodes, exposure_hours,
       events: {ENDPOINT: {total, per_hour}} for ENDPOINT in
               union2_152_{all,old,new}, union2_severe_{all,old,new},
               union20_{152,76,45}_{all,old,new},
               union20_severe_{all,old,new},
               pairwise_{152,76,45}_{all,old,new},
       time_below: {union20_tb_{thr} | pairwise_tb_{thr} | union2_tb_152:
                    {total_s, per_hour_s}},
       min_sep_20hz: {median, p5, mean}, min_sep_2hz: {median, p5},
       planner: {cycles, feasible_frac_mean (mean of episode means),
                 feasible_frac_p10_median, emergency_cycles,
                 emergency_frac, causes {...}, pen_depth_mean_m,
                 pen_depth_max_m, attrib_inconsistent},
       floor: {floor_active_frac (pooled), floor_n,
               branch_sigma_below_floor_{5,10,15}s (pooled), ..._n},
       probe: {safety_pool_max_p50_median, _p90_median, _max} | null,
       timing: {cycle_ms_p50_median, cycle_ms_p99_max, cycle_ms_max,
                deadline_miss, cycles},
       strata: {FAMILY: {STRATUM: {all, old, new}}} counts}},
   "paired": {"A|B": {
       n, indices [first, last], unpaired_traffic_mismatch,
       min_sep_20hz: {wins, losses, ties (A larger = win), sign_p,
                      mean_diff, median_diff {est, ci}, p5_diff {est, ci}},
       min_sep_2hz: same shape,
       events: {ENDPOINT: {wins, losses, ties (A fewer = win), sign_p,
                            total_a, total_b, rate_a, rate_b,
                            rate_diff {est, ci}, rate_ratio {est, ci,
                            undefined_frac}}},
       time_below: {ENDPOINT: same as events},
       strata: {FAMILY: {STRATUM: {SCREEN: {total_a, total_b,
                rate_diff {est, ci}}}}}}},
   "coverage": {LABEL: {dir, arm, files, forecasts, kind, strata_source,
       strata_params, conditions, code_fp,
       strata_counts: {forecasts: {all|interv0_react0|interv0_react1|
                                   interv1_react0|interv1_react1: n},
                       marginals: {intervening_in_horizon: {yes, no},
                                   intruder_reacting: {yes, no}},
                       components: {dev_crit, speed_crit (the lateral /
                                    speed parts of intervening_in_horizon),
                                    reaction_latched, intent_reactive,
                                    own_intervening_at_t},
                       valid_points: {CELL: {"5s"|"10s"|"15s": n}}},
       cells: {CELL: {"5s"|"10s"|"15s": {n, cov50, cov70, cov90}}}}}}
       (cells is absent with --preds-counts-only; the counts are computed
       and printed before any coverage number)
  ci = [2.5, 97.5] percentiles of a paired episode bootstrap (B draws of
  the paired episode indices with replacement, fixed seed); a rate ratio
  is undefined in a draw whose B-condition total is 0 (reported as
  undefined_frac; the CI uses the defined draws). sign_p is the exact
  two-sided binomial sign test on non-tied episodes (log-space CDF, so n
  in the thousands is fine).
  Coverage: HPD region membership of the GT position at 5/10/15 s.
  Mixtures (K > 1): Monte Carlo, mc_draws samples per point from the
  predicted mixture, inside iff logpdf(GT) >= the (1-p) quantile of the
  samples' logpdf (the modelTraining estimator); single Gaussians (K = 1,
  e.g. the CV arm): closed form, Mahalanobis^2 <= -2 ln(1-p).
  Closed-loop strata per forecast made at time t, from the raw arrays:
    intervening_in_horizon (--own-ref reference, the default) = max |own_d|
      over the logged frames in [t, t+15 s] > --own-dev-m (default 30 m,
      the planner's hard radius) OR min own_v over [t, t+15 s] <
      --own-vfrac (0.90) x v_desired. Alternatives (--own-ref): nominal =
      the same with own_d_nominal (offset from the ownship's nominal
      straight track: the scenario places that track OFF the corridor
      centreline, so own_d carries a placement offset before any
      avoidance); at_t = |own_d(tau) - own_d(t)| > own_dev_m OR
      |own_v(tau) - own_v(t)| > (1 - own_vfrac) x v_desired for some tau in
      [t, t+15 s] (manoeuvre during the horizon);
    intruder_reacting = reaction_latched OR intent_reactive
      (reaction layer latched at any 20 Hz tick in [t, t+15 s];
       gt_intent == REACTIVE_AVOID at t).
  Cells interv{0|1}_react{0|1}. strata_source "legacy_stored_flags" marks
  a directory written before these arrays existed: its cells then use the
  stored own_intervening / intruder_reacting flags and the parameters do
  not apply.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
from itertools import combinations

import numpy as np

SEED = 20260926
B_DEFAULT = 10_000
MC_DEFAULT = 256
T_NEED_S = math.sqrt(2.0 * 152.4 / 4.5)
THR = (("152", 152.4), ("76", 76.2), ("45", 45.0))
SCREENS = ("all", "old", "new")
LEVELS = (0.5, 0.7, 0.9)
HORIZONS = ((5, 9), (10, 19), (15, 29))       # (seconds, step index)
HARD_TYPES = (0, 6)                           # Bird, Unknown
HARD_INTENTS = (3,)                           # SearchWander
STRATUM_KEYS = ("class=HARD", "class=EASY", "turning_at_track_start=yes",
                "turning_at_track_start=no", "turning_at_track_start=unknown",
                "reaction_active=yes", "reaction_active=no")
KNOB_FIELDS = ("preset", "track_rule", "allow_rule_mismatch", "hard_geom",
               "sigma_floor", "plateau_scale", "w_safety", "safety_norm",
               "safety_ref", "safety_ref_probe", "sigma_scale", "pi_min",
               "iso", "gain_mode", "model_path", "model_track_rule")


# ======================================================================
# statistics helpers
# ======================================================================

def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test (p = 1/2), ties excluded.
    The CDF is summed in log space: math.comb at n ~ thousands overflows
    floats (DECISIONS D5.14)."""
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    ln2 = math.log(2.0)
    lg = math.lgamma
    logs = [lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * ln2
            for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return float(min(1.0, 2.0 * tail))


def wlt(a: np.ndarray, b: np.ndarray, better: str) -> dict:
    """Wins/losses/ties of A vs B per paired episode."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if better == "low":
        w, l = int((d < 0).sum()), int((d > 0).sum())
    else:
        w, l = int((d > 0).sum()), int((d < 0).sum())
    return {"wins": w, "losses": l, "ties": int(d.size - w - l),
            "sign_p": round(sign_test(w, l), 6)}


def boot_indices(n: int, B: int, seed: int) -> np.ndarray:
    """[B, n] paired-episode resampling indices (with replacement)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n), dtype=np.int32)


def _ci(x: np.ndarray) -> list | None:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def rate_boot(ca: np.ndarray, cb: np.ndarray, ha: np.ndarray,
              hb: np.ndarray, idx: np.ndarray) -> dict:
    """Point estimates + paired bootstrap CIs of the difference and ratio of
    per-hour rates (sum counts / sum hours) of A and B."""
    ra0 = ca.sum() / ha.sum() if ha.sum() > 0 else math.nan
    rb0 = cb.sum() / hb.sum() if hb.sum() > 0 else math.nan
    ra = ca[idx].sum(axis=1) / ha[idx].sum(axis=1)
    rb = cb[idx].sum(axis=1) / hb[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rb > 0, ra / rb, np.nan)
    und = float(np.mean(~np.isfinite(ratio)))
    return {"total_a": _num(ca.sum()), "total_b": _num(cb.sum()),
            "rate_a": _num(ra0, 4), "rate_b": _num(rb0, 4),
            "rate_diff": {"est": _num(ra0 - rb0, 4), "ci": _ci(ra - rb)},
            "rate_ratio": {"est": (_num(ra0 / rb0, 4) if rb0 > 0 else None),
                           "ci": _ci(ratio),
                           "undefined_frac": round(und, 4)}}


def stat_boot(a: np.ndarray, b: np.ndarray, idx: np.ndarray, q: float) -> dict:
    """Difference of a per-condition quantile (median = 50) with CI."""
    est = float(np.percentile(a, q) - np.percentile(b, q))
    da = np.percentile(a[idx], q, axis=1)
    db = np.percentile(b[idx], q, axis=1)
    return {"est": round(est, 3), "ci": _ci(da - db)}


def _num(v, nd=None):
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    if nd is None:
        return int(v) if v.is_integer() else v
    return round(v, nd)


# ======================================================================
# loading
# ======================================================================

def parse_spec(spec: str) -> tuple[str | None, list[str]]:
    if "=" in spec and not os.path.exists(spec):
        label, paths = spec.split("=", 1)
        return label, [p for p in paths.split(",") if p]
    return None, [p for p in spec.split(",") if p]


def load_runs(paths: list[str]) -> tuple[dict[int, dict], dict]:
    rows: dict[int, dict] = {}
    errors = dups = 0
    arm = None
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                arm = arm or r.get("arm")
                if "error" in r:
                    errors += 1
                    continue
                i = int(r["index"])
                if i in rows:
                    dups += 1
                    continue
                rows[i] = r
    return rows, {"errors": errors, "duplicates": dups, "arm": arm}


# ======================================================================
# per-episode endpoints
# ======================================================================

def _screen_counts(evs: list[dict], pred=None) -> dict[str, int]:
    evs = [e for e in evs if pred is None or pred(e)]
    return {"all": len(evs),
            "old": sum(1 for e in evs if not e["unavoidable_old"]),
            "new": sum(1 for e in evs if not e["unavoidable_new"])}


def _minimum_m(event: dict) -> float:
    """Raw 20 Hz event minimum; legacy displayed value is an explicit
    fallback, not evidence that a rounded threshold value is truly equal."""
    return float(event['min_sep_raw_m'] if event.get('min_sep_raw_m') is not None
                 else event['min_sep'])


def endpoint_minimum_provenance(rows: list[dict]) -> dict:
    """Keep endpoint precision changes separate from track-screen changes.
    Counts cover the records filtered by raw minima, not all direct runs."""
    out = {'policy': 'raw_minimum_preferred_legacy_display_fallback',
           'pairwise_raw_events': 0, 'pairwise_legacy_rounded_events': 0,
           'union152_raw_events': 0, 'union152_legacy_rounded_events': 0,
           'legacy_union2_events': sum(len(r.get('events', [])) for r in rows)}
    for r in rows:
        for family, events in (('pairwise', r.get('pairwise') or []),
                               ('union152', (r.get('union20_event_list') or {}).get('152', []))):
            for event in events:
                suffix = ('raw_events' if event.get('min_sep_raw_m') is not None
                          else 'legacy_rounded_events')
                out[f'{family}_{suffix}'] += 1
    return out


def episode_endpoints(r: dict) -> dict[str, float | None]:
    """Per-episode event counts / times; None where the row lacks fields."""
    out: dict[str, float | None] = {}
    ev2 = r.get("events", [])
    scr = r.get("events_new_screen")
    new_ok = scr is not None and len(scr) == len(ev2)
    pairs = list(zip(ev2, scr if new_ok else [None] * len(ev2)))
    for name, max_sep in (("152", math.inf), ("severe", 45.0)):
        sel = [(e, s) for e, s in pairs if e["min_sep"] < max_sep]
        out[f"union2_{name}_all"] = len(sel)
        out[f"union2_{name}_old"] = sum(1 for e, _ in sel
                                        if not e["unavoidable"])
        out[f"union2_{name}_new"] = (sum(1 for _, s in sel
                                         if not s["unavoidable_new"])
                                     if new_ok else None)
    ul = r.get("union20_event_list")
    pw = r.get("pairwise")
    for lbl, thr in THR:
        if ul is not None:
            for s, v in _screen_counts(ul.get(lbl, [])).items():
                out[f"union20_{lbl}_{s}"] = v
        else:
            for s in SCREENS:
                out[f"union20_{lbl}_{s}"] = None
        if pw is not None:
            c = _screen_counts(pw, lambda e, t=thr: _minimum_m(e) < t)
            for s, v in c.items():
                out[f"pairwise_{lbl}_{s}"] = v
        else:
            for s in SCREENS:
                out[f"pairwise_{lbl}_{s}"] = None
    if ul is not None:
        c = _screen_counts(ul.get("152", []), lambda e: _minimum_m(e) < 45.0)
        for s, v in c.items():
            out[f"union20_severe_{s}"] = v
    else:
        for s in SCREENS:
            out[f"union20_severe_{s}"] = None
    return out


def episode_times(r: dict) -> dict[str, float | None]:
    out: dict[str, float | None] = {"union2_tb_152": r.get("nmac_duration_s")}
    pw = r.get("pairwise")
    for lbl, _ in THR:
        out[f"union20_tb_{lbl}"] = r.get(f"union20_time_below_{lbl}_s")
        out[f"pairwise_tb_{lbl}"] = (sum(float(s[f"tb_{lbl}_s"]) for s in pw)
                                     if pw is not None else None)
    return out


def episode_exposure_s(r: dict) -> float:
    """Prefer measured scoring exposure; never silently infer 90 s for
    a legacy row. The original last-replan sim_s is retained in the input."""
    value = float(r['exposure_s'] if r.get('exposure_s') is not None
                  else r.get('sim_s', 0.0))
    if not math.isfinite(value) or value < 0:
        raise ValueError('episode exposure must be finite and nonnegative')
    return value


def exposure_provenance(rows: list[dict]) -> dict:
    corrected = sum(r.get('exposure_s') is not None for r in rows)
    legacy = sum(r.get('exposure_s') is None and 'sim_s' in r for r in rows)
    return {'scoring_exposure_rows': corrected,
            'legacy_sim_s_fallback_rows': legacy,
            'missing_exposure_rows': len(rows) - corrected - legacy,
            'legacy_last_replan_hours': sum(float(r.get('sim_s', 0.0))
                                           for r in rows) / 3600.0}


def _stratum_keys(seg: dict) -> dict[str, str]:
    hard = (int(seg.get("gt_type", -1)) in HARD_TYPES
            or int(seg.get("gt_intent", -1)) in HARD_INTENTS)
    ts = seg.get("turning_at_track_start")
    return {"class": "HARD" if hard else "EASY",
            "turning_at_track_start": ("unknown" if ts is None
                                       else ("yes" if ts else "no")),
            "reaction_active": "yes" if seg.get("reaction_active") else "no"}


def episode_strata(r: dict) -> dict | None:
    """{FAMILY: {AXIS=VALUE: {screen: count}}} for pairwise encounters
    (152 m and severe <45 m) and union20 152 m events."""
    pw = r.get("pairwise")
    ul = r.get("union20_event_list")
    if pw is None or ul is None:
        return None
    fam: dict[str, list[tuple[dict, dict]]] = {
        "pairwise_152": [(s, s) for s in pw],
        "pairwise_45": [(s, s) for s in pw if _minimum_m(s) < 45.0],
        "union20_152": []}
    for e in ul.get("152", []):
        seg = next((s for s in pw if s["slot"] == e["slot"]
                    and s["t_entry"] <= e["t_min"] <= s["t_exit"]), None)
        fam["union20_152"].append((e, seg or {}))
    out: dict = {}
    for name, items in fam.items():
        # every stratum is always reported, zero counts included
        d: dict[str, dict[str, int]] = {k: {s: 0 for s in SCREENS}
                                        for k in STRATUM_KEYS}
        for ev, seg in items:
            for axis, val in _stratum_keys(seg).items():
                key = f"{axis}={val}"
                c = d.setdefault(key, {s: 0 for s in SCREENS})
                c["all"] += 1
                c["old"] += int(not ev["unavoidable_old"])
                c["new"] += int(not ev["unavoidable_new"])
        out[name] = d
    return out


# ======================================================================
# per-condition summary
# ======================================================================

def _col(rows: list[dict], fn) -> np.ndarray | None:
    vals = [fn(r) for r in rows]
    if any(v is None for v in vals):
        return None
    return np.asarray(vals, dtype=np.float64)


def summarize(label: str, rows: dict[int, dict]) -> dict:
    v = [rows[i] for i in sorted(rows)]
    n = len(v)
    if n == 0:
        return {"episodes": 0}
    hours = sum(episode_exposure_s(r) for r in v) / 3600.0
    out: dict = {"episodes": n, "exposure_hours": round(hours, 4),
                 "exposure_provenance": exposure_provenance(v),
                 "endpoint_minimum_provenance": endpoint_minimum_provenance(v)}

    eps = [episode_endpoints(r) for r in v]
    ev: dict = {}
    for k in eps[0]:
        c = _col(eps, lambda e, k=k: e[k])
        ev[k] = (None if c is None else
                 {"total": int(c.sum()),
                  "per_hour": round(float(c.sum()) / hours, 4) if hours else None})
    out["events"] = ev
    tms = [episode_times(r) for r in v]
    tb: dict = {}
    for k in tms[0]:
        c = _col(tms, lambda e, k=k: e[k])
        tb[k] = (None if c is None else
                 {"total_s": round(float(c.sum()), 2),
                  "per_hour_s": round(float(c.sum()) / hours, 3) if hours else None})
    out["time_below"] = tb

    for key, name in (("min_sep_20hz", "min_sep_20hz"), ("min_sep", "min_sep_2hz")):
        c = _col(v, lambda r, key=key: r.get(key))
        out[name] = (None if c is None else
                     {"median": round(float(np.median(c)), 2),
                      "p5": round(float(np.percentile(c, 5)), 2),
                      "mean": round(float(c.mean()), 2)})

    # planner decision-rule diagnostics
    if all("n_cycles" in r for r in v):
        cyc = sum(int(r["n_cycles"]) for r in v)
        em = sum(int(r["emergency_cycles"]) for r in v)
        causes = {}
        for k in ("emerg_collision_static", "emerg_collision_forecast",
                  "emerg_boundary_only", "emerg_boundary_dynamic",
                  "emerg_dynamic_only", "emerg_unexplained"):
            causes[k] = sum(int(r.get(k, 0)) for r in v)
        pens = [(r["pen_depth_mean_m"], r["emerg_collision_static"]
                 + r["emerg_collision_forecast"]) for r in v
                if r.get("pen_depth_mean_m") is not None]
        npen = sum(w for _, w in pens)
        out["planner"] = {
            "cycles": cyc,
            "feasible_frac_mean": round(float(np.mean(
                [r["feasible_frac_mean"] for r in v])), 4),
            "feasible_frac_p10_median": round(float(np.median(
                [r["feasible_frac_p10"] for r in v])), 4),
            "emergency_cycles": em,
            "emergency_frac": round(em / cyc, 5) if cyc else None,
            "causes": causes,
            "pen_depth_mean_m": (round(sum(p * w for p, w in pens) / npen, 3)
                                 if npen else None),
            "pen_depth_max_m": (round(max(r["pen_depth_max_m"] for r in v
                                          if r.get("pen_depth_max_m")
                                          is not None), 2)
                                if pens else None),
            "attrib_inconsistent": sum(int(r.get("attrib_inconsistent", 0))
                                       for r in v)}
    else:
        out["planner"] = None

    if all("floor_n" in r for r in v):
        fn_ = sum(int(r["floor_n"]) for r in v)
        fr = sum(float(r["floor_active_frac"] or 0.0) * int(r["floor_n"])
                 for r in v)
        fl = {"floor_active_frac": round(fr / fn_, 5) if fn_ else None,
              "floor_n": fn_}
        for h in (5, 10, 15):
            tot = sum(int(r.get(f"branch_sigma_n_{h}s") or 0) for r in v)
            bel = sum(float(r.get(f"branch_sigma_below_floor_{h}s") or 0.0)
                      * int(r.get(f"branch_sigma_n_{h}s") or 0) for r in v)
            fl[f"branch_sigma_below_floor_{h}s"] = (round(bel / tot, 5)
                                                    if tot else None)
            fl[f"branch_sigma_n_{h}s"] = tot
        out["floor"] = fl
    else:
        out["floor"] = None

    if all(r.get("safety_pool_max_p50") is not None for r in v):
        out["probe"] = {
            "safety_pool_max_p50_median": round(float(np.median(
                [r["safety_pool_max_p50"] for r in v])), 6),
            "safety_pool_max_p90_median": round(float(np.median(
                [r["safety_pool_max_p90"] for r in v])), 6),
            "safety_pool_max_max": round(float(max(
                r["safety_pool_max_max"] for r in v)), 6)}
    else:
        out["probe"] = None

    if all(r.get("cycle_ms_p50") is not None for r in v):
        out["timing"] = {
            "cycle_ms_p50_median": round(float(np.median(
                [r["cycle_ms_p50"] for r in v])), 1),
            "cycle_ms_p99_max": round(float(max(r["cycle_ms_p99"] for r in v)), 1),
            "cycle_ms_max": round(float(max(r["cycle_ms_max"] for r in v)), 1),
            "deadline_miss": sum(int(r["deadline_miss"]) for r in v),
            "cycles": sum(int(r.get("n_cycles", 0)) for r in v),
            "note": "shared CPU/GPU; not a real-time benchmark"}
    else:
        out["timing"] = None

    st = [episode_strata(r) for r in v]
    if all(s is not None for s in st):
        agg: dict = {}
        for s in st:
            for fam, d in s.items():
                fa = agg.setdefault(fam, {})
                for key, c in d.items():
                    ca = fa.setdefault(key, {x: 0 for x in SCREENS})
                    for x in SCREENS:
                        ca[x] += c[x]
        out["strata"] = agg
    else:
        out["strata"] = None
    return out


# ======================================================================
# paired comparison
# ======================================================================

def same_traffic(a: dict, b: dict) -> bool:
    """Paired rows must describe the SAME traffic: equal seed and agent
    count, and equal preset/overrides where both rows record them."""
    if a.get("seed") != b.get("seed") or a.get("n_agents") != b.get("n_agents"):
        return False
    for k in ("preset", "overrides"):
        if k in a and k in b and a[k] != b[k]:
            return False
    return True


def compare(ra: dict[int, dict], rb: dict[int, dict], B: int,
            seed: int) -> dict:
    common = sorted(set(ra) & set(rb))
    idx_ep = [i for i in common if same_traffic(ra[i], rb[i])]
    n = len(idx_ep)
    if n == 0:
        return {"n": 0, "unpaired_traffic_mismatch": len(common)}
    A = [ra[i] for i in idx_ep]
    Bv = [rb[i] for i in idx_ep]
    bidx = boot_indices(n, B, seed)
    ha = np.asarray([episode_exposure_s(r) / 3600.0 for r in A])
    hb = np.asarray([episode_exposure_s(r) / 3600.0 for r in Bv])
    out: dict = {"n": n, "indices": [idx_ep[0], idx_ep[-1]],
             "unpaired_traffic_mismatch": len(common) - n,
             "exposure_provenance_a": exposure_provenance(A),
             "exposure_provenance_b": exposure_provenance(Bv),
             "endpoint_minimum_provenance_a": endpoint_minimum_provenance(A),
             "endpoint_minimum_provenance_b": endpoint_minimum_provenance(Bv)}

    for key, name in (("min_sep_20hz", "min_sep_20hz"), ("min_sep", "min_sep_2hz")):
        a = _col(A, lambda r, key=key: r.get(key))
        b = _col(Bv, lambda r, key=key: r.get(key))
        if a is None or b is None:
            out[name] = None
            continue
        d = wlt(a, b, "high")
        d["mean_diff"] = round(float((a - b).mean()), 3)
        d["median_diff"] = stat_boot(a, b, bidx, 50)
        d["p5_diff"] = stat_boot(a, b, bidx, 5)
        out[name] = d

    ea = [episode_endpoints(r) for r in A]
    eb = [episode_endpoints(r) for r in Bv]
    evs: dict = {}
    for k in ea[0]:
        a = _col(ea, lambda e, k=k: e[k])
        b = _col(eb, lambda e, k=k: e[k])
        if a is None or b is None:
            evs[k] = None
            continue
        d = wlt(a, b, "low")
        d.update(rate_boot(a, b, ha, hb, bidx))
        evs[k] = d
    out["events"] = evs

    ta = [episode_times(r) for r in A]
    tb = [episode_times(r) for r in Bv]
    tms: dict = {}
    for k in ta[0]:
        a = _col(ta, lambda e, k=k: e[k])
        b = _col(tb, lambda e, k=k: e[k])
        if a is None or b is None:
            tms[k] = None
            continue
        d = wlt(a, b, "low")
        d.update(rate_boot(a, b, ha, hb, bidx))
        tms[k] = d
    out["time_below"] = tms

    sa = [episode_strata(r) for r in A]
    sb = [episode_strata(r) for r in Bv]
    if all(s is not None for s in sa + sb):
        keys: dict[str, set] = {}
        for s in sa + sb:
            for fam, d in s.items():
                keys.setdefault(fam, set()).update(d)
        strata: dict = {}
        for fam in sorted(keys):
            for key in sorted(keys[fam]):
                for scr in SCREENS:
                    a = np.asarray([s[fam].get(key, {}).get(scr, 0) for s in sa],
                                   dtype=np.float64)
                    b = np.asarray([s[fam].get(key, {}).get(scr, 0) for s in sb],
                                   dtype=np.float64)
                    rb_ = rate_boot(a, b, ha, hb, bidx)
                    strata.setdefault(fam, {}).setdefault(key, {})[scr] = {
                        "total_a": rb_["total_a"], "total_b": rb_["total_b"],
                        "rate_diff": rb_["rate_diff"]}
        out["strata"] = strata
    else:
        out["strata"] = None
    return out


# ======================================================================
# closed-loop prediction coverage (R3 minor 5)
# ======================================================================

def _rot(theta: np.ndarray) -> np.ndarray:
    """[P,2,2] world -> target-frame rotation (v_frame = R @ v_world)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


def _gauss_logpdf(d: np.ndarray, sig: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """log N of frame offsets d [...,2] with per-axis sig [...,2], rho [...]."""
    zx = d[..., 0] / sig[..., 0]
    zy = d[..., 1] / sig[..., 1]
    omr2 = np.clip(1.0 - rho ** 2, 1e-6, None)
    q = (zx ** 2 - 2.0 * rho * zx * zy + zy ** 2) / omr2
    return (-0.5 * q - np.log(sig[..., 0]) - np.log(sig[..., 1])
            - 0.5 * np.log(omr2) - math.log(2.0 * math.pi))


def hpd_inside(mu_f: np.ndarray, sig: np.ndarray, rho: np.ndarray,
               pi: np.ndarray, gt_f: np.ndarray, rng: np.random.Generator,
               S: int) -> dict[float, np.ndarray]:
    """HPD membership of gt for [P] points; mixtures by MC, K=1 closed form.

    mu_f [P,K,2], sig [P,K,2], rho [P,K], pi [P,K], gt_f [P,2] -- all in
    the target frame (rotation preserves densities)."""
    P, K = pi.shape
    if K == 1:
        d = gt_f - mu_f[:, 0]
        zx = d[:, 0] / sig[:, 0, 0]
        zy = d[:, 1] / sig[:, 0, 1]
        r = rho[:, 0]
        omr2 = np.clip(1.0 - r ** 2, 1e-6, None)
        m2 = (zx ** 2 - 2.0 * r * zx * zy + zy ** 2) / omr2
        return {p: m2 <= -2.0 * math.log(1.0 - p) for p in LEVELS}
    w = np.clip(pi, 0.0, None)
    w = w / w.sum(axis=1, keepdims=True)
    logw = np.log(np.clip(w, 1e-300, None))

    def mix_logpdf(x: np.ndarray) -> np.ndarray:           # x [P,S,2]
        d = x[:, :, None, :] - mu_f[:, None, :, :]           # [P,S,K,2]
        ln = _gauss_logpdf(d, sig[:, None], rho[:, None]) + logw[:, None]
        m = ln.max(axis=2, keepdims=True)
        return (m[..., 0] + np.log(np.exp(ln - m).sum(axis=2)))

    u = rng.random((P, S))
    comp = (u[:, :, None] > np.cumsum(w, axis=1)[:, None, :]).sum(axis=2)
    comp = np.minimum(comp, K - 1)                            # [P,S]
    e1 = rng.standard_normal((P, S))
    e2 = rng.standard_normal((P, S))
    sgx = np.take_along_axis(sig[..., 0], comp, axis=1)
    sgy = np.take_along_axis(sig[..., 1], comp, axis=1)
    rr = np.take_along_axis(rho, comp, axis=1)
    mx = np.take_along_axis(mu_f[..., 0], comp, axis=1)
    my = np.take_along_axis(mu_f[..., 1], comp, axis=1)
    xs = np.stack([mx + sgx * e1,
                   my + sgy * (rr * e1 + np.sqrt(np.clip(1 - rr ** 2, 1e-6,
                                                         None)) * e2)], -1)
    lp_s = mix_logpdf(xs)                                     # [P,S]
    lp_g = mix_logpdf(gt_f[:, None, :])[:, 0]                 # [P]
    return {p: lp_g >= np.quantile(lp_s, 1.0 - p, axis=1) for p in LEVELS}


PRED_HORIZON_S = 15.0
CELLS = ("all", "interv0_react0", "interv0_react1", "interv1_react0",
         "interv1_react1")
RAW_STRATA_KEYS = ("own_t", "own_d", "own_v", "v_desired",
                   "reaction_latched", "intent_reactive")
CONDITION_IGNORE = ("code_fp", "code_fp_files")


def pred_files(d: str, arm: str) -> list[str]:
    """<arm>_<index>.npz files of one arm (exact arm match: 'cv' never picks
    up 'cvc_msig_*')."""
    pat = re.compile(rf"^{re.escape(arm)}_\d{{5,}}\.npz$")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if pat.match(f))


OWN_REFS = ("reference", "nominal", "at_t")
OWN_REF_DOC = {
    "reference": "max|own_d| over [t, t+15 s] > own_dev_m OR min own_v "
                 "over [t, t+15 s] < own_vfrac * v_desired (own_d = offset "
                 "from the corridor reference path)",
    "nominal": "max|own_d_nominal| over [t, t+15 s] > own_dev_m OR min "
               "own_v over [t, t+15 s] < own_vfrac * v_desired (offset from "
               "the ownship's nominal straight ground track)",
    "at_t": "max|own_d(tau) - own_d(t)| over tau in [t, t+15 s] > own_dev_m "
            "OR max|own_v(tau) - own_v(t)| > (1 - own_vfrac) * v_desired "
            "(manoeuvre DURING the horizon, relative to the ownship state "
            "at the forecast time)"}


def forecast_strata(z, own_dev_m: float, own_vfrac: float,
                    horizon_s: float = PRED_HORIZON_S,
                    own_ref: str = "reference") -> dict[str, np.ndarray]:
    """Per-forecast stratum flags from the RAW pred-log arrays (module doc).
    ``z`` is an npz or any mapping with the same keys. ``own_ref`` selects
    the ownship-intervention definition (OWN_REF_DOC); dev_crit / speed_crit
    are its lateral and speed parts."""
    if own_ref not in OWN_REFS:
        raise ValueError(f"own_ref must be one of {OWN_REFS}, got {own_ref!r}")
    t = np.asarray(z["t_s"], dtype=np.float64)                # [P]
    ot = np.asarray(z["own_t"], dtype=np.float64)             # [F]
    key_d = "own_d_nominal" if own_ref == "nominal" else "own_d"
    names = set(z.files) if hasattr(z, "files") else set(z)
    if key_d not in names:
        raise SystemExit(f"own_ref={own_ref!r} needs {key_d!r}, which this "
                         f"pred log does not store")
    od = np.asarray(z[key_d], dtype=np.float64)
    ov = np.asarray(z["own_v"], dtype=np.float64)
    vdes = float(np.asarray(z["v_desired"]))
    win = ((ot[None, :] >= t[:, None] - 1e-6)
           & (ot[None, :] <= t[:, None] + horizon_s + 1e-6))  # [P,F]
    if ot.size == 0:
        dev = np.zeros(t.size, bool)
        spd = np.zeros(t.size, bool)
    elif own_ref == "at_t":
        i0 = np.clip(np.searchsorted(ot, t - 1e-6), 0, ot.size - 1)
        dd = np.where(win, np.abs(od[None, :] - od[i0][:, None]), -np.inf)
        dv = np.where(win, np.abs(ov[None, :] - ov[i0][:, None]), -np.inf)
        dev = dd.max(axis=1) > own_dev_m
        spd = dv.max(axis=1) > (1.0 - own_vfrac) * vdes
    else:
        maxd = np.where(win, np.abs(od)[None, :], -np.inf).max(axis=1)
        minv = np.where(win, ov[None, :], np.inf).min(axis=1)
        dev = maxd > own_dev_m
        spd = minv < own_vfrac * vdes
    latched = np.asarray(z["reaction_latched"], dtype=bool)
    intent = np.asarray(z["intent_reactive"], dtype=bool)
    return {"intervening": dev | spd, "dev_crit": dev,
            "speed_crit": spd, "reacting": latched | intent,
            "reaction_latched": latched, "intent_reactive": intent,
            "window_frames": win.sum(axis=1)}


def _file_strata(z, own_dev_m: float, own_vfrac: float,
                 own_ref: str = "reference"):
    """(source, flags) of one pred-log file; legacy files (written before
    the raw arrays existed) fall back to their stored flags."""
    names = set(z.files) if hasattr(z, "files") else set(z)
    if all(k in names for k in RAW_STRATA_KEYS):
        return "raw", forecast_strata(z, own_dev_m, own_vfrac,
                                      own_ref=own_ref)
    return "legacy_stored_flags", {
        "intervening": np.asarray(z["own_intervening"], dtype=bool),
        "reacting": np.asarray(z["intruder_reacting"], dtype=bool)}


def _cell_names(fl: dict) -> np.ndarray:
    return np.array([f"interv{int(i)}_react{int(r)}"
                     for i, r in zip(fl["intervening"], fl["reacting"])],
                    dtype=object)


def _usable(z, step: int) -> np.ndarray:
    """[P] forecasts scorable at a horizon step (GT present, finite params)."""
    valid = np.asarray(z["gt_valid"][:, step], dtype=bool)
    mu_w = np.asarray(z["mu"][:, :, step], dtype=np.float64)
    sig = np.asarray(z["sigma"][:, :, step], dtype=np.float64)
    gt_w = np.asarray(z["gt"][:, step], dtype=np.float64)
    return (valid & np.isfinite(gt_w).all(1) & np.isfinite(mu_w).all((1, 2))
            & np.isfinite(sig).all((1, 2)) & (sig > 0).all((1, 2)))


def pred_dir_counts(d: str, arm: str, own_dev_m: float,
                    own_vfrac: float, own_ref: str = "reference") -> dict:
    """Stratum counts of one --log-preds directory. Reads no coverage
    outcome: only the stratum flags and GT availability."""
    files = pred_files(d, arm)
    fc = {c: 0 for c in CELLS}
    vp = {c: {f"{h}s": 0 for h, _ in HORIZONS} for c in CELLS}
    comp = {"dev_crit": 0, "speed_crit": 0, "reaction_latched": 0,
            "intent_reactive": 0, "own_intervening_at_t": 0}
    marg = {"intervening_in_horizon": {"yes": 0, "no": 0},
            "intruder_reacting": {"yes": 0, "no": 0}}
    sources, kinds, conds, fps = set(), set(), {}, set()
    n_fc = 0
    for fp in files:
        z = np.load(fp)
        kinds.add(str(z["kind"]))
        if "meta_json" in z.files:
            meta = json.loads(str(z["meta_json"]))
            if meta.get("code_fp"):
                fps.add(meta["code_fp"])
            cond = {k: v for k, v in meta.items() if k not in CONDITION_IGNORE}
            conds.setdefault(json.dumps(cond, sort_keys=True), []).append(
                os.path.basename(fp))
        src, fl = _file_strata(z, own_dev_m, own_vfrac, own_ref)
        sources.add(src)
        if len(sources) > 1:
            raise SystemExit(f"{d}: mixes raw and legacy pred-log formats "
                             f"({sorted(sources)}); score them separately")
        P = int(z["t0"].shape[0])
        n_fc += P
        if P == 0:
            continue
        cell = _cell_names(fl)
        fc["all"] += P
        for c in CELLS[1:]:
            fc[c] += int((cell == c).sum())
        iv, rv = fl["intervening"], fl["reacting"]
        marg["intervening_in_horizon"]["yes"] += int(iv.sum())
        marg["intervening_in_horizon"]["no"] += int((~iv).sum())
        marg["intruder_reacting"]["yes"] += int(rv.sum())
        marg["intruder_reacting"]["no"] += int((~rv).sum())
        for k in ("dev_crit", "speed_crit", "reaction_latched",
                  "intent_reactive"):
            if k in fl:
                comp[k] += int(fl[k].sum())
            else:
                comp[k] = None
        comp["own_intervening_at_t"] += int(
            np.asarray(z["own_intervening"], dtype=bool).sum())
        for h, step in HORIZONS:
            ok = _usable(z, step)
            vp["all"][f"{h}s"] += int(ok.sum())
            for c in CELLS[1:]:
                vp[c][f"{h}s"] += int((ok & (cell == c)).sum())
    if len(conds) > 1:
        raise SystemExit(f"{d}: files of arm {arm!r} record {len(conds)} "
                         f"different run conditions (e.g. "
                         f"{[v[0] for v in conds.values()]}); one directory "
                         f"must hold one condition")
    src = sources.pop() if sources else None
    return {
        "dir": d.replace("\\", "/"), "arm": arm, "files": len(files),
        "forecasts": n_fc, "kind": sorted(kinds), "strata_source": src,
        "strata_params": (
            {"own_ref": own_ref, "own_dev_m": own_dev_m,
             "own_vfrac": own_vfrac, "horizon_s": PRED_HORIZON_S,
             "intervening_in_horizon": OWN_REF_DOC[own_ref],
             "intruder_reacting": "reaction_latched OR intent_reactive"}
            if src == "raw" else
            {"note": "legacy files: stored own_intervening / "
                     "intruder_reacting flags; parameters not applied"}),
        "conditions": [json.loads(k) for k in conds],
        "code_fp": sorted(fps),
        "strata_counts": {"forecasts": fc, "marginals": marg,
                          "components": comp, "valid_points": vp}}


def pred_dir_coverage(d: str, arm: str, own_dev_m: float, own_vfrac: float,
                      S: int, seed: int, chunk: int = 2048,
                      own_ref: str = "reference") -> dict:
    """HPD coverage per stratum cell (run AFTER pred_dir_counts)."""
    acc = {c: {f"{h}s": {"n": 0, **{f"cov{int(p*100)}": 0 for p in LEVELS}}
               for h, _ in HORIZONS} for c in CELLS}
    rng = np.random.default_rng(seed)
    for fp in pred_files(d, arm):
        z = np.load(fp)
        P = int(z["t0"].shape[0])
        if P == 0:
            continue
        _, fl = _file_strata(z, own_dev_m, own_vfrac, own_ref)
        cell = _cell_names(fl)
        R = _rot(z["theta"].astype(np.float64))               # [P,2,2]
        for h, step in HORIZONS:
            mu_w = z["mu"][:, :, step].astype(np.float64)      # [P,K,2]
            sig = z["sigma"][:, :, step].astype(np.float64)
            rho = z["rho"][:, :, step].astype(np.float64)
            pi = z["pi"].astype(np.float64)
            gt_w = z["gt"][:, step].astype(np.float64)
            ii = np.nonzero(_usable(z, step))[0]
            for c0 in range(0, ii.size, chunk):
                sl = ii[c0:c0 + chunk]
                Rs = R[sl]
                mu_f = np.einsum("pij,pkj->pki", Rs, mu_w[sl])
                gt_f = np.einsum("pij,pj->pi", Rs, gt_w[sl])
                ins = hpd_inside(mu_f, sig[sl], rho[sl], pi[sl], gt_f, rng, S)
                for cname in ("all",) + tuple(sorted(set(cell[sl]))):
                    m = np.ones(sl.size, bool) if cname == "all" \
                        else (cell[sl] == cname)
                    a = acc[cname][f"{h}s"]
                    a["n"] += int(m.sum())
                    for p in LEVELS:
                        a[f"cov{int(p*100)}"] += int(ins[p][m].sum())
    for c in CELLS:
        for a in acc[c].values():
            n = a["n"]
            for p in LEVELS:
                k = f"cov{int(p*100)}"
                a[k] = round(a[k] / n, 4) if n else None
    return acc


def format_counts(label: str, c: dict) -> str:
    sc = c["strata_counts"]
    f = sc["forecasts"]
    m = sc["marginals"]
    k = sc["components"]
    return (f"strata {label} [{c['strata_source']}]: forecasts={f['all']} "
            + " ".join(f"{x}={f[x]}" for x in CELLS[1:])
            + f" | intervening yes/no={m['intervening_in_horizon']['yes']}/"
              f"{m['intervening_in_horizon']['no']} reacting yes/no="
              f"{m['intruder_reacting']['yes']}/{m['intruder_reacting']['no']}"
              f" | dev={k['dev_crit']} speed={k['speed_crit']} "
              f"latched={k['reaction_latched']} "
              f"intent={k['intent_reactive']} "
              f"own_int_at_t={k['own_intervening_at_t']}")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="[LABEL=]PATH[,PATH...] per condition")
    ap.add_argument("--preds", nargs="*", default=[],
                    help="LABEL=DIR of --log-preds output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out file")
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mc", type=int, default=MC_DEFAULT)
    ap.add_argument("--own-dev-m", type=float, default=30.0,
                    help="ownship stratum: max |lateral offset| over [t, "
                         "t+15 s] above this [m] counts as intervening "
                         "(30 = the planner's hard radius)")
    ap.add_argument("--own-vfrac", type=float, default=0.90,
                    help="ownship stratum: min speed over [t, t+15 s] below "
                         "this fraction of v_desired counts as intervening")
    ap.add_argument("--own-ref", default="reference", choices=OWN_REFS,
                    help="ownship-intervention definition: reference = "
                         "offset from the corridor reference path (the "
                         "default definition); nominal = offset from the "
                         "ownship's nominal straight track; at_t = lateral "
                         "/ speed change during [t, t+15 s] relative to the "
                         "state at the forecast time")
    ap.add_argument("--preds-counts-only", action="store_true",
                    help="report the pred-log stratum counts and skip the "
                         "coverage computation entirely")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"refusing to overwrite {args.out} (use --force)")

    t_start = time.perf_counter()
    conds: dict[str, dict[int, dict]] = {}
    inputs: dict[str, dict] = {}
    for spec in args.runs:
        label, paths = parse_spec(spec)
        rows, info = load_runs(paths)
        if label is None:
            parent = os.path.basename(os.path.dirname(os.path.abspath(paths[0])))
            label = f"{parent}/{info['arm']}"
        if label in conds:
            raise SystemExit(f"duplicate label {label!r}: name it LABEL=PATH")
        conds[label] = rows
        knobs = {}
        for k in KNOB_FIELDS:
            vals = sorted({json.dumps(r.get(k)) for r in rows.values()})
            knobs[k] = [json.loads(x) for x in vals]
        inputs[label] = {"paths": [p.replace("\\", "/") for p in paths],
                         "episodes": len(rows), **info, "knobs": knobs,
                         "code_fp": sorted({r.get("code_fp") for r in rows.values()
                                            if r.get("code_fp")})}

    out = {"schema": "aggregate_r3/2",
           "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seed": args.seed, "B": args.boot, "mc_draws": args.mc,
           "t_need_s": round(T_NEED_S, 4), "inputs": inputs,
           "per_arm": {lab: summarize(lab, rows) for lab, rows in conds.items()},
           "paired": {}, "coverage": {}}
    for la, lb in combinations(list(conds), 2):
        out["paired"][f"{la}|{lb}"] = compare(conds[la], conds[lb],
                                             args.boot, args.seed)
    pred_specs = []
    for spec in args.preds:
        if "=" not in spec:
            raise SystemExit(f"--preds needs LABEL=DIR, got {spec!r}")
        label, d = spec.split("=", 1)
        arm = inputs.get(label, {}).get("arm")
        if arm is None:        # no run with this label: the dir must hold one arm
            arms = {os.path.basename(p).rsplit("_", 1)[0]
                    for p in glob.glob(os.path.join(d, "*.npz"))}
            if len(arms) != 1:
                raise SystemExit(f"--preds {spec!r}: label matches no --runs "
                                 f"entry and the dir holds arms {sorted(arms)}")
            arm = arms.pop()
        pred_specs.append((label, d, arm))
    # 1) stratum counts for every directory BEFORE any coverage number, so
    #    the split can be judged for degeneracy without seeing outcomes
    for label, d, arm in pred_specs:
        out["coverage"][label] = pred_dir_counts(d, arm, args.own_dev_m,
                                                 args.own_vfrac, args.own_ref)
        print(format_counts(label, out["coverage"][label]), flush=True)
    # 2) coverage per cell
    if not args.preds_counts_only:
        for label, d, arm in pred_specs:
            out["coverage"][label]["cells"] = pred_dir_coverage(
                d, arm, args.own_dev_m, args.own_vfrac, args.mc, args.seed,
                own_ref=args.own_ref)
    out["preds_counts_only"] = bool(args.preds_counts_only)
    out["elapsed_s"] = round(time.perf_counter() - t_start, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for lab, s in out["per_arm"].items():
        ev = s.get("events", {})
        g = lambda k: (ev.get(k) or {}).get("per_hour")  # noqa: E731
        print(f"{lab:40s} ep={s.get('episodes')} h={s.get('exposure_hours')} "
              f"u2_152 old/h={g('union2_152_old')} u20_152 all/h="
              f"{g('union20_152_all')} new/h={g('union20_152_new')} "
              f"pw_152 all/h={g('pairwise_152_all')} "
              f"minsep20 med={(s.get('min_sep_20hz') or {}).get('median')}")
    for k, v in out["paired"].items():
        ms = v.get("min_sep_20hz") or {}
        print(f"{k}: n={v.get('n')} minsep20 W/L/T={ms.get('wins')}/"
              f"{ms.get('losses')}/{ms.get('ties')} p={ms.get('sign_p')}")
    for k, v in out["coverage"].items():
        if "cells" not in v:
            continue
        a = v["cells"]["all"]
        print(f"coverage {k}: forecasts={v['forecasts']} "
              + " ".join(f"{h}:{a[h]['cov90']}(n={a[h]['n']})" for h in a))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
