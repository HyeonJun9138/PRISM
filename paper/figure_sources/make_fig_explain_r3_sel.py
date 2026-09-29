# -*- coding: utf-8 -*-
"""PRISM fig_explain r3 (rule-selected case): prediction and path selection
at one closed-loop instant of arm C (PRISM) of the episode that the
pre-registered rule (D:/PRISM/Paper/figures_r3/case_rule_r3.txt,
select_case_r3.py) selects. Replaces the ep244 version
(make_fig_explain_r3.py; its outputs were copied to fig_explain_r3_ep244*
and caption_explain_r3_ep244.txt before this script overwrote the stems).

Data: D:/PRISM/Paper/figures_r3/data/scene_<idx>.json (recorded at the
instant by select_case_r3.py final), case_<idx>.json, gate_<idx>.json,
candidates_<cond>.json. Gate, recording-run compare and the rule file's sha256
are asserted.

Instant (Step 5 of the rule, applied online by the recorder in arm C): the
first replanning cycle before A's closest approach at which the selected
plan's lateral offset target changes by more than 30 m relative to the
previously selected plan while the critical intruder has an open track and a
PRISM forecast; fallbacks (largest change) are stated in the caption if used.

Panels (same window, world metres)
  (a) forecasts: the time-swept HPD map of make_fig_hdr.py (its StepHPD /
      swept_levels / display_schedule / prism_nodes, colour scale, contour
      levels, mean-path widths, detection dots and true-future dashes are
      imported and reused), combined over the intruders in view by the
      minimum level; pi labels of the critical intruder.
  (b) the planner's own risk field at that cycle from the recorded disks
      (risk.disks_batch at the evaluation lookaheads 0..15 s), shown as the
      maximum over the lookaheads of the soft kernel sum / g0; the candidate
      pool (admissible dark, rejected light) and the selected path. The
      kernel is checked against the planner: the safety cost of every
      candidate recomputed from the recorded disks equals the planner's raw
      safety cost.

Outputs (D:/PRISM/Paper/figures_r3): fig_explain_r3.{png,pdf},
fig_explain_r3_panel_{a,b}.{png,pdf}, caption_explain_r3.txt, and the
"fig_explain_r3" entry of fig_case_r3_manifest.json (the ep244 entry is kept
as "fig_explain_r3_ep244").

Imports make_fig_hdr.py for its HPD functions only (its module-level imports
load the live D:/PRISM tree; no value of this figure passes through them).
sys.dont_write_bytecode is set so no .pyc is written into the live tree.

Run: python
     D:/PRISM/Paper/figures/make_fig_explain_r3_sel.py [--cond ood40] [--index N]
"""
from __future__ import annotations

import os

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

ROOT = (os.environ["PRISM_ROOT"] + '')
HDR_PATH = f"{ROOT}/Paper/figures/make_fig_hdr.py"
_spec = importlib.util.spec_from_file_location("make_fig_hdr", HDR_PATH)
HDR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(HDR)           # also sets the HDR figure style

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

OUT = f"{ROOT}/Paper/figures_r3"
RULE_FILE = f"{OUT}/case_rule_r3.txt"
MANIFEST = f"{OUT}/fig_case_r3_manifest.json"
TYPE_NAMES = ["Bird", "Multirotor", "Fixed-wing UAS", "eVTOL",
              "General aviation", "Debris", "Unknown"]
TYPE_KR = ["조류", "멀티로터", "고정익 무인기", "eVTOL", "경항공기(GA)",
           "부유 잔해", "미확인 기체"]
TYPE_PHRASE = {"Bird": "bird", "Multirotor": "multirotor",
               "Fixed-wing UAS": "fixed-wing UAS", "eVTOL": "eVTOL",
               "General aviation": "general-aviation aircraft",
               "Debris": "piece of debris", "Unknown": "unknown aircraft"}


def art(w: str) -> str:
    return "an" if w[0].lower() in "aeiou" else "a"


OWN_PAST_S = 20.0
OWN_PAST_VIEW_S = 10.0
RISK_N = 600
RISK_BLANK = 0.01
PICK = "#0072B2"
CAND_OK, CAND_BAD = "#8c8c8c", "#d4d4d4"
_OR = plt.get_cmap("Oranges")
RISK_CMAP = LinearSegmentedColormap.from_list(
    "planner_risk", [_OR(v) for v in np.linspace(0.10, 0.85, 256)])
STROKE = [pe.withStroke(linewidth=1.8, foreground="white")]

ap = argparse.ArgumentParser()
ap.add_argument("--cond", default="ood40")
ap.add_argument("--index", type=int, default=None)
ARGS = ap.parse_args()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arr(v) -> np.ndarray:
    def fix(x):
        if isinstance(x, list):
            return [fix(y) for y in x]
        return np.nan if x is None else x
    return np.asarray(fix(v), dtype=float)


# ------------------------------------------------------------------ load
SELN_PATH = f"{OUT}/data/selection_r3.json"      # union over the conditions
SELN = json.load(open(SELN_PATH, encoding="utf-8"))
COND = ARGS.cond if ARGS.index is not None else SELN["selection"]["cond"]
CAND_PATH = f"{OUT}/data/candidates_{COND}.json"
CAND = json.load(open(CAND_PATH, encoding="utf-8"))
IDX = ARGS.index if ARGS.index is not None else SELN["selection"]["index"]
SCENE = f"{OUT}/data/scene_{IDX}.json"
CASE = f"{OUT}/data/case_{IDX}.json"
GATE = f"{OUT}/data/gate_{IDX}.json"
S = json.load(open(SCENE, encoding="utf-8"))
G = json.load(open(GATE, encoding="utf-8"))
META = S["meta"]
assert G["pass"] and META["gate"]["pass"] and META["record_check"]["pass"],\
    "gate / record check did not pass"
assert META["episode"]["index"] == IDX == G["episode_index"]
assert META["episode"]["condition"] == f"r3_{COND}"
RULE_META = META["rule"]
SHA_ORIG = RULE_META.get("sha256_original", RULE_META.get("sha256"))
assert SHA_ORIG == CAND["rule_sha256"] == SELN["rule_sha256"],\
    "original rule sha256 disagrees between the data files"
RULE_TXT = open(RULE_FILE, encoding="utf-8").read()
assert RULE_TXT.startswith(RULE_META.get("original_text", RULE_META["text"])),\
    "the rule file no longer starts with the original (pre-registered) text"
AMENDED = bool(RULE_META.get("amended"))
assert sha256(RULE_FILE) == (RULE_META["sha256_amended"] if AMENDED
                             else SHA_ORIG), "rule file changed"
RULE = S["fig9_rule"]
A1 = RULE.get("amendment") == "A1"           # amended Step 5 (post hoc)
CYC_PATH = f"{OUT}/data/scene_{IDX}_cycles.json"
CYCLES = json.load(open(CYC_PATH, encoding="utf-8")) if A1 else None
ROW_T = RULE["row_at_instant"] if A1 else None
if A1:
    assert abs(float(S["t"]) - float(ROW_T["t"])) < 1e-9
    assert any(abs(r["t"] - float(S["t"])) < 1e-9 for r in CYCLES["table"])
T_FIG = float(S["t"])
D = S["arms"]["model"]
assert abs(D["t"] - T_FIG) < 1e-9
CRIT = META["critical"]["slot"]
LABEL = META["arm_labels"]
OWN = np.asarray(D["own"][:2], float)
PSI = float(D["own"][2])
OWN_HIST = arr(D["own_hist"])
TRACKS = {int(t["slot"]): t for t in D["tracks"]}
PL = D["planner"]
RK = D["risk"]
G0 = float(RK["base_gain"])
LOOK = np.asarray(D["disk_lookaheads"], float)
DISKS = np.asarray(D["disks"], float)            # [31, M, 5]
SMAP = [tuple(x) for x in D["disk_slot_map"]]
CANDS = D["candidates"]
SEL = int(D["selected"])
T_MIN_A = float(META["critical"]["t_min_a"])
assert T_FIG < T_MIN_A

FC = {}
for f in D["forecasts"]:
    FC[int(f["slot"])] = {
        "slot": int(f["slot"]), "mu_w": np.asarray(f["mu_w"], float),
        "sigma": np.asarray(f["sigma"], float),
        "rho": np.asarray(f["rho"], float), "pi": np.asarray(f["pi"], float),
        "theta": float(f["theta"]), "origin": np.asarray(f["origin"], float),
        "theta_check": float(f["frame_theta_check"]),
        "used_pos": arr(f["used_pos"]),
        "type": TYPE_NAMES[TRACKS[int(f["slot"])]["gt_type"]]}
CRIT_FC = CRIT in FC
assert CRIT_FC or A1 or RULE["why"] == "fallback_max_change_any",\
    "critical intruder has no forecast although the rule requires one"

# the previous selected plan (from the C-arm cycle list of the case file)
case = json.load(open(CASE, encoding="utf-8"))
CYC = case["arms"]["model"]["cycles"]
k_now = next(i for i, c in enumerate(CYC) if abs(c["t"] - T_FIG) < 1e-6)
PREV_D = float(CYC[k_now - 1]["sel"]["target_d"]) if k_now > 0 else None
NOW_D = float(CYC[k_now]["sel"]["target_d"])
assert abs(NOW_D - CANDS[SEL]["target_d"]) < 1e-9
if PREV_D is not None and not A1:
    assert abs(abs(NOW_D - PREV_D) - RULE["chosen_d_change"]) < 1e-6

# ============================================================ validation
VAL = {}
log_lines = []


def log(s: str) -> None:
    log_lines.append(s)
    print(s)


vr, st, hr_cfg = PL["vehicle_radius"], PL["soft_standoff"], RK["hard_radius"]
dt_plan, horizon = PL["plan_dt"], PL["risk_horizon"]
worst_abs = worst_rel = 0.0
for c in CANDS:
    t = np.asarray(c["t"], float)
    m = t <= horizon
    j = np.rint(t[m] / (LOOK[1] - LOOK[0])).astype(int)
    assert np.allclose(LOOK[j], t[m], atol=1e-12, rtol=0), "off-grid sample"
    dk = DISKS[j]
    x, y = np.asarray(c["x"], float)[m], np.asarray(c["y"], float)[m]
    clr = np.hypot(x[:, None] - dk[:, :, 0], y[:, None] - dk[:, :, 1])\
        - dk[:, :, 2] - vr
    soft = np.maximum(0.0, clr - st)
    cost = float(np.sum(np.exp(-soft ** 2 / (2 * dk[:, :, 3] ** 2))
                        * dk[:, :, 4]) * dt_plan)
    ref = c["raw"]["safety"]
    worst_abs = max(worst_abs, abs(cost - ref))
    worst_rel = max(worst_rel, abs(cost - ref) / max(abs(ref), 1e-12))
VAL["safety_recomputed"] = {"n_candidates": len(CANDS),
                            "max_abs_diff": worst_abs,
                            "max_rel_diff": worst_rel,
                            "pass": worst_rel < 1e-9}
log(f"[V1] safety cost of {len(CANDS)} candidates recomputed from the recorded "
    f"disks: max |diff| {worst_abs:.2e}, max rel {worst_rel:.2e} "
    f"{'PASS' if VAL['safety_recomputed']['pass'] else 'FAIL'}")

floor, pi_min = RK["sigma_floor"], RK["pi_min"]
d_ctr = d_sig = d_gain = 0.0
kept = set()
for mi, (slot, k) in enumerate(SMAP):
    tr = TRACKS[slot]
    anchor = np.asarray(tr["obs_t0"], float)
    col = DISKS[:, mi, :]
    if k < 0:
        d_ctr = max(d_ctr, float(np.abs(col[:, :2] - anchor).max()))
        d_sig = max(d_sig, float(np.abs(col[:, 3] - max(floor,
                                                        RK["sigma_base"])).max()))
        d_gain = max(d_gain, float(np.abs(col[:, 4] - G0).max()))
        continue
    kept.add((slot, k))
    f = FC[slot]
    iso = np.sqrt((f["sigma"][k, :, 0] ** 2 + f["sigma"][k, :, 1] ** 2) / 2.0)
    ctr = np.vstack([anchor, f["mu_w"][k]])
    sig = np.concatenate([[floor], np.maximum(floor, iso)])
    d_ctr = max(d_ctr, float(np.abs(col[:, :2] - ctr).max()))
    d_sig = max(d_sig, float(np.abs(col[:, 3] - sig).max()))
    d_gain = max(d_gain, float(np.abs(col[:, 4] - G0 * f["pi"][k]).max()))
expect = {(s, k) for s, f in FC.items() if TRACKS[s]["detected_t0"]
          for k in range(len(f["pi"])) if f["pi"][k] >= pi_min}
dropped_no_det = sorted(s for s in FC if not TRACKS[s]["detected_t0"])
pruned = sorted((s, k) for s, f in FC.items() for k in range(len(f["pi"]))
                if f["pi"][k] < pi_min)
VAL["disks_from_forecasts"] = {
    "max_abs_centre_m": d_ctr, "max_abs_sigma_m": d_sig,
    "max_abs_gain": d_gain, "branch_set_equal": kept == expect,
    "forecasts_without_detection_at_t0_not_used": dropped_no_det,
    "branches_pruned_pi_below_min": pruned,
    "pass": bool(kept == expect and d_ctr < 1e-3 and d_sig < 1e-3
                 and d_gain < 1e-6)}
log(f"[V2] disks vs forecasts: max |d centre| {d_ctr:.1e} m, |d sigma| "
    f"{d_sig:.1e} m, |d gain| {d_gain:.1e}; retained-branch set equal "
    f"{kept == expect}; forecasts not used (no detection at t0): "
    f"{dropped_no_det}; pruned branches: {pruned} "
    f"{'PASS' if VAL['disks_from_forecasts']['pass'] else 'FAIL'}")

dth = max([abs(math.remainder(f["theta"] - f["theta_check"], 2 * math.pi))
           for f in FC.values()] or [0.0])
VAL["frame_theta"] = {"max_abs_diff_rad": dth, "pass": dth < 1e-6}
log(f"[V3] forecast theta vs build_sample frame theta: max |diff| {dth:.1e} rad "
    f"{'PASS' if dth < 1e-6 else 'FAIL'}")
assert all(v["pass"] for v in VAL.values()), "validation failed"


# ============================================================== HPD maps
def hpd_case(f: dict) -> dict:
    nodes = HDR.prism_nodes(f, np.zeros(2))
    sched = HDR.display_schedule(nodes, f["origin"], True)
    node_tab = [HDR.StepHPD(cp) for cp in nodes]
    tab = [node_tab[j] if j is not None else HDR.StepHPD(cp)
           for _, cp, j in sched]
    lam = min(float(np.linalg.eigvalsh(S_)[0]) for _, cp, _ in sched
              for _, _, S_ in cp)
    return {"nodes": nodes, "sched": sched, "tab": tab, "lam_min": lam,
            "mass_min": min(t.mass for t in tab),
            "box": [HDR.comps_box(cp, 4.0) for _, cp, _ in sched]}


sel = CANDS[SEL]
st_ = np.asarray(sel["t"], float)
sel_xy = np.stack([sel["x"], sel["y"]], 1)
tr_c = TRACKS[CRIT]
t_own = np.arange(len(OWN_HIST)) * 0.5 + 0.5
H = {}
pts = [OWN[None], OWN_HIST[t_own >= T_FIG - OWN_PAST_VIEW_S],
       arr(tr_c["gt_future"]), arr(tr_c["gt_t0"])[None],
       sel_xy[st_ <= horizon]]
if CRIT_FC:
    H[CRIT] = hpd_case(FC[CRIT])
    lo = np.min([b[0] for b in H[CRIT]["box"]], 0)
    hi = np.max([b[1] for b in H[CRIT]["box"]], 0)
    ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2
    g_prov = HDR.make_grid((ctr[0] - half, ctr[0] + half, ctr[1] - half,
                            ctr[1] + half), HDR.PROVISIONAL_N)
    Lc = HDR.swept_levels([cp for _, cp, _ in H[CRIT]["sched"]],
                          H[CRIT]["tab"], g_prov)
    m = Lc <= HDR.L_BLANK
    pts += [np.stack([g_prov["X"][m], g_prov["Y"][m]], 1), FC[CRIT]["used_pos"]]
P = np.vstack([p[np.isfinite(p).all(1)] for p in pts if len(p)])
lo, hi = P.min(0), P.max(0)
ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * 1.10 + 60.0
half = float(np.ceil(half / 50.0) * 50.0)
TICK = HDR.nice_step(2 * half)
EXT = (ctr[0] - half, ctr[0] + half, ctr[1] - half, ctr[1] + half)

for s, f in FC.items():
    if s == CRIT:
        continue
    lo_s = np.min([mu - 4.0 * np.sqrt(np.diag(Sg))
                   for nd in HDR.prism_nodes(f, np.zeros(2))
                   for _, mu, Sg in nd], 0)
    hi_s = np.max([mu + 4.0 * np.sqrt(np.diag(Sg))
                   for nd in HDR.prism_nodes(f, np.zeros(2))
                   for _, mu, Sg in nd], 0)
    if (hi_s[0] < EXT[0] or lo_s[0] > EXT[1] or hi_s[1] < EXT[2]
            or lo_s[1] > EXT[3]):
        continue
    H[s] = hpd_case(f)
GRID = HDR.make_grid(EXT)
L = np.ones_like(GRID["X"])
for s in list(H):
    H[s]["L"] = HDR.swept_levels([cp for _, cp, _ in H[s]["sched"]],
                                 H[s]["tab"], GRID)
    if s != CRIT and not (H[s]["L"] <= HDR.L_BLANK).any():
        del H[s]
        continue
    L = np.minimum(L, H[s]["L"])
IN_VIEW = sorted(H, key=lambda s: (s != CRIT, s))
VAL["hpd"] = {str(s): {"time_points": len(H[s]["sched"]),
                       "min_fine_grid_mass": round(H[s]["mass_min"], 6),
                       "min_eigenvalue_m2": round(H[s]["lam_min"], 4),
                       "pass": H[s]["mass_min"] >= HDR.MASS_MIN
                       and H[s]["lam_min"] > 0}
              for s in IN_VIEW}
for s in IN_VIEW:
    v = VAL["hpd"][str(s)]
    log(f"[V4] HPD slot {s} ({FC[s]['type']}): {v['time_points']} time points, "
        f"min fine-grid mass {v['min_fine_grid_mass']}, min covariance "
        f"eigenvalue {v['min_eigenvalue_m2']} m^2 "
        f"{'PASS' if v['pass'] else 'FAIL'}")
assert all(v["pass"] for v in VAL["hpd"].values())

# ============================================================ risk field
RG = HDR.make_grid(EXT, RISK_N)
reach = hr_cfg + vr + st
n_used = 0
for mi in range(DISKS.shape[1]):
    col = DISKS[:, mi, :]
    far = (np.minimum.reduce([col[:, 0] - EXT[0], EXT[1] - col[:, 0],
                              col[:, 1] - EXT[2], EXT[3] - col[:, 1]])
           < -(reach + 6.0 * col[:, 3]))
    if far.all():
        continue
    n_used += 1
Rt = np.zeros((len(LOOK),) + RG["X"].shape)
for j in range(len(LOOK)):
    acc = np.zeros_like(RG["X"])
    for mi in range(DISKS.shape[1]):
        cx, cy, hr, sg, gn = DISKS[j, mi]
        if (cx < EXT[0] - reach - 6 * sg or cx > EXT[1] + reach + 6 * sg
                or cy < EXT[2] - reach - 6 * sg or cy > EXT[3] + reach + 6 * sg):
            continue
        soft = np.maximum(0.0, np.hypot(RG["X"] - cx, RG["Y"] - cy)
                          - hr - vr - st)
        acc += gn * np.exp(-soft ** 2 / (2.0 * sg ** 2))
    Rt[j] = acc
R = Rt.max(axis=0) / G0
VMAX = max(1.0, float(R.max()))
log(f"[V5] risk field: {n_used} of {DISKS.shape[1]} disks reach the window; "
    f"max kernel / g0 in view {R.max():.3f}")

ok = [i for i, c in enumerate(CANDS) if c["valid"]]
bad = [i for i, c in enumerate(CANDS) if not c["valid"]]
why = {"collision": sum(1 for i in bad if CANDS[i]["collision"]),
       "boundary_only": sum(1 for i in bad if CANDS[i]["boundary"]
                            and not CANDS[i]["collision"]),
       "dynamics_only": sum(1 for i in bad if not CANDS[i]["collision"]
                            and not CANDS[i]["boundary"])}
assert len(ok) == D["n_valid"] and len(CANDS) == D["n_total"]
EMERGENCY = D["n_valid"] == 0            # planner fell back to least violation
assert CANDS[SEL]["valid"] or EMERGENCY, "invalid selection outside an emergency"
if EMERGENCY:
    assert all(CANDS[SEL]["violation"] <= c["violation"] + 1e-9 for c in CANDS),\
        "emergency selection is not the least-violating candidate"

# [V6] hard-radius conflicts of every candidate recomputed from the recorded
# disks (clearance = distance - r_disk - r_veh < 0 at a sample time within
# the risk horizon) must reproduce the planner's collision flag; the disks
# are then attributed to slots, so the caption can name the aircraft that
# blocks the pool.
conf_slots = {}
v6_ok = True
for i, c in enumerate(CANDS):
    t = np.asarray(c["t"], float)
    m = t <= horizon
    j = np.rint(t[m] / (LOOK[1] - LOOK[0])).astype(int)
    dk = DISKS[j]
    x, y = np.asarray(c["x"], float)[m], np.asarray(c["y"], float)[m]
    clr = np.hypot(x[:, None] - dk[:, :, 0], y[:, None] - dk[:, :, 1])\
        - dk[:, :, 2] - vr
    hit = (clr < 0.0).any(axis=0)
    v6_ok &= bool(hit.any()) == bool(c["collision"])
    for mi in np.nonzero(hit)[0]:
        conf_slots.setdefault(int(SMAP[mi][0]), set()).add(i)
CONFLICT_BY_SLOT = {s: len(v) for s, v in sorted(conf_slots.items(),
                                                key=lambda kv: -len(kv[1]))}
VAL["hard_conflicts"] = {"pass": bool(v6_ok),
                         "n_colliding": sum(1 for c in CANDS if c["collision"]),
                         "candidates_in_conflict_by_slot": CONFLICT_BY_SLOT}
log(f"[V6] hard-radius conflicts recomputed from the recorded disks reproduce "
    f"the planner's collision flags for all {len(CANDS)} candidates: "
    f"{'PASS' if v6_ok else 'FAIL'}; conflicting candidates by slot "
    f"{CONFLICT_BY_SLOT}")
assert v6_ok

# nearest aircraft at the instant (ground truth) and the blocking aircraft
_near = [(float(np.hypot(*(arr(tr["gt_t0"]) - OWN))), s) for s, tr in TRACKS.items()
         if np.all(np.isfinite(arr(tr["gt_t0"])))]
NEAR_D, NEAR_S = min(_near)
BLOCK_S = next(iter(CONFLICT_BY_SLOT)) if CONFLICT_BY_SLOT else None
BLOCK_D = (float(np.hypot(*(arr(TRACKS[BLOCK_S]["gt_t0"]) - OWN)))
           if BLOCK_S is not None else None)
if A1:
    # the scene at the instant must agree with the A1 table row
    assert CONFLICT_BY_SLOT.get(CRIT, 0) == ROW_T["n_infeasible_by_bird"],\
        "V6 bird conflicts differ from the A1 table"
    assert EMERGENCY == bool(ROW_T["emergency"])
    assert (not EMERGENCY) or RULE["why"] == "no_non_emergency_cycle"
    log(f"[V7] A1 consistency at t = {T_FIG:g} s: bird hard-check failures "
        f"{CONFLICT_BY_SLOT.get(CRIT, 0)} == table {ROW_T['n_infeasible_by_bird']}; "
        f"emergency {EMERGENCY}; bird share {ROW_T['bird_share']:.3f} PASS")


# ================================================================ drawing
def own_marker():
    return (3, 0, math.degrees(PSI) - 90.0)


def draw_ownship(ax):
    m = t_own >= T_FIG - OWN_PAST_S
    ax.plot(OWN_HIST[m, 0], OWN_HIST[m, 1], color=HDR.INK, lw=1.0,
            ls=(0, (4, 2)), zorder=7)
    ax.plot(*OWN, marker=own_marker(), ms=8.0, mfc=HDR.INK, mec="white",
            mew=0.8, ls="none", zorder=10)


def setup(ax, title):
    ax.set_xlim(EXT[0], EXT[1])
    ax.set_ylim(EXT[2], EXT[3])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(TICK))
    ax.set_aspect("equal")
    ax.apply_aspect()
    ax.grid(True, lw=0.4, color="#ececec")
    for sp in ax.spines.values():
        sp.set_zorder(9)
    ax.set_title(title, pad=3.5, color="#1a1a1a")
    ax.set_xlabel("x [m]", labelpad=1.5)


def place(ax, xy, text, cands, avoid, placed, **kw):
    r = ax.figure.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    best = None
    for dx, dy, ha, va in cands:
        t = ax.annotate(text, xy=xy, xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, **kw)
        bb = t.get_window_extent(r)
        sc = 100.0 * sum(bb.overlaps(b) for b in placed)
        sc += float(((avoid[:, 0] > bb.x0) & (avoid[:, 0] < bb.x1)
                     & (avoid[:, 1] > bb.y0) & (avoid[:, 1] < bb.y1)).sum())
        if (bb.x0 < axbb.x0 + 2 or bb.x1 > axbb.x1 - 2
                or bb.y0 < axbb.y0 + 2 or bb.y1 > axbb.y1 - 2):
            sc += 1000.0
        if best is None or sc < best[0]:
            if best is not None:
                best[1].remove()
            best = (sc, t, bb)
        else:
            t.remove()
    placed.append(best[2])
    return best[1]


AROUND = [(0, 9, "center", "bottom"), (0, -9, "center", "top"),
          (9, 0, "left", "center"), (-9, 0, "right", "center"),
          (8, 7, "left", "bottom"), (-8, 7, "right", "bottom"),
          (8, -7, "left", "top"), (-8, -7, "right", "top")]


def intruder_name(s: int) -> str:
    n = FC[s]["type"].lower() if s in FC else\
        TYPE_NAMES[TRACKS[s]["gt_type"]].lower()
    return f"{n} (critical)" if s == CRIT else n


def draw_forecasts(ax):
    rgba = HDR.CMAP(HDR.NORM(np.clip(L, 0.0, HDR.L_BLANK)))
    rgba[..., 3] = (L <= HDR.L_BLANK).astype(float)
    ax.imshow(rgba, extent=EXT, origin="lower", interpolation="nearest",
              zorder=1)
    ds = (EXT[1] - EXT[0]) / 150.0
    lines, obs, ends_c = [], [], []
    for s in IN_VIEW:
        f = FC[s]
        fut = np.vstack([arr(TRACKS[s]["gt_t0"])[None],
                         arr(TRACKS[s]["gt_future"])])
        fut = fut[np.isfinite(fut).all(1)]
        lines += [fut] + list(f["mu_w"])
        obs.append(f["used_pos"])
    Dp = np.vstack([np.vstack(obs)] + [HDR.resample(ln, ds) for ln in lines])\
        if obs else np.zeros((0, 2))
    polys, cl_pts = [], []
    ax._clabels = []
    if IN_VIEW:
        cs = ax.contour(GRID["X"], GRID["Y"], L, levels=list(HDR.LEVELS),
                        colors=HDR.CONTOUR_INK, linewidths=0.55, zorder=3)
        polys = [s_ for seg in cs.allsegs for s_ in seg if len(s_) > 1]
        spots = [tuple(p) for p in HDR.contour_label_spots(cs, Dp, EXT)
                 if p is not None]
        if spots:
            for t in ax.clabel(cs, levels=list(HDR.LEVELS), manual=spots,
                               inline=True, inline_spacing=1.5, fontsize=5.6,
                               fmt={lv: f"{int(round(lv * 100))} %"
                                    for lv in HDR.LEVELS}):
                t.set_color(HDR.CONTOUR_INK)
                t.set_zorder(3.5)
                t.set_path_effects([pe.withStroke(linewidth=1.6,
                                                  foreground="white")])
                cl_pts.append(t.get_position())
                ax._clabels.append(t)
    halo = lambda w: [pe.Stroke(linewidth=w + HDR.HALO_PT,  # noqa: E731
                                foreground="white"), pe.Normal()]
    for s in IN_VIEW:
        f = FC[s]
        ax.plot(f["used_pos"][:, 0], f["used_pos"][:, 1], "o", ms=2.1,
                mfc=HDR.OBS, mec="white", mew=0.3, ls="none", zorder=5)
        for k in np.argsort(-f["pi"]):
            lw = HDR.LW_PER_PI * float(f["pi"][k])
            ax.plot(f["mu_w"][k][:, 0], f["mu_w"][k][:, 1], color=HDR.INK,
                    lw=lw, solid_capstyle="round", zorder=6,
                    path_effects=halo(lw))
            ax.plot(*f["mu_w"][k][-1], "o", ms=2.2, mfc=HDR.INK, mec="white",
                    mew=0.45, zorder=6.6)
            if s == CRIT:
                ends_c.append((int(k), f["mu_w"][k][-1]))
        fut = np.vstack([arr(TRACKS[s]["gt_t0"])[None],
                         arr(TRACKS[s]["gt_future"])])
        fut = fut[np.isfinite(fut).all(1)]
        ax.plot(fut[:, 0], fut[:, 1], color=HDR.TRUE_FUT, lw=0.9, ls=HDR.DASH,
                dash_capstyle="butt", zorder=6.4, path_effects=halo(0.9))
    if not CRIT_FC:                      # fallback_any: GT only
        fut = np.vstack([arr(tr_c["gt_t0"])[None], arr(tr_c["gt_future"])])
        fut = fut[np.isfinite(fut).all(1)]
        ax.plot(fut[:, 0], fut[:, 1], color=HDR.TRUE_FUT, lw=0.9, ls=HDR.DASH,
                dash_capstyle="butt", zorder=6.4, path_effects=halo(0.9))
        ax.plot(*fut[0], "s", ms=3.0, mfc="none", mec=HDR.TRUE_FUT, mew=0.8,
                zorder=6.5)
    draw_ownship(ax)
    avoid = [Dp] + [HDR.resample(s_, ds) for s_ in polys]
    avoid.append(HDR.resample(OWN_HIST[t_own >= T_FIG - OWN_PAST_S], ds))
    ax._prism_labels = (ends_c, FC[CRIT]["pi"] if CRIT_FC else np.zeros(0),
                        np.vstack([p for p in avoid if len(p)]),
                        np.array(cl_pts, float).reshape(-1, 2))
    return np.vstack([p for p in avoid if len(p)])


def draw_risk(ax):
    rgba = RISK_CMAP(Normalize(0.0, VMAX)(np.clip(R, 0.0, VMAX)))
    rgba[..., 3] = (R >= RISK_BLANK).astype(float) * 0.92
    ax.imshow(rgba, extent=EXT, origin="lower", interpolation="nearest",
              zorder=1)
    for i in bad:
        c = CANDS[i]
        ax.plot(c["x"], c["y"], color=CAND_BAD, lw=0.3, zorder=3)
    for i in ok:
        if i == SEL:
            continue
        c = CANDS[i]
        ax.plot(c["x"], c["y"], color=CAND_OK, lw=0.3, alpha=0.75, zorder=4)
    c = CANDS[SEL]
    halo = [pe.Stroke(linewidth=2.9, foreground="white"), pe.Normal()]
    ax.plot(c["x"], c["y"], color=PICK, lw=1.9, zorder=8, path_effects=halo,
            solid_capstyle="round")
    t = np.asarray(c["t"], float)
    k15 = int(np.argmin(np.abs(t - horizon)))
    ax.plot(c["x"][k15], c["y"][k15], marker="o", ms=3.4, mfc="white",
            mec=PICK, mew=1.1, zorder=9)
    for s in sorted(set(IN_VIEW) | {CRIT}):
        p = arr(TRACKS[s]["obs_t0"])
        if np.all(np.isfinite(p)):
            ax.plot(*p, "o", ms=2.6, mfc=HDR.OBS, mec="white", mew=0.4,
                    zorder=6)
    draw_ownship(ax)
    return k15


def add_intruder_labels(ax, avoid_data):
    placed = []
    avoid = ax.transData.transform(avoid_data)
    for s in sorted(set(IN_VIEW) | {CRIT}, key=lambda s: (s != CRIT, s)):
        p = arr(TRACKS[s]["obs_t0"])
        if not np.all(np.isfinite(p)):
            p = FC[s]["origin"] if s in FC else arr(TRACKS[s]["gt_t0"])
        if not (EXT[0] < p[0] < EXT[1] and EXT[2] < p[1] < EXT[3]):
            continue
        place(ax, p, intruder_name(s), AROUND, avoid, placed, fontsize=6.0,
              color="#1a1a1a", zorder=8.5, path_effects=STROKE)
    return placed


DIRS16 = [(math.cos(a), math.sin(a))
          for a in np.deg2rad(np.arange(0.0, 360.0, 22.5))]
RADII_PI = (10.0, 14.0, 19.0, 25.0, 32.0, 40.0, 50.0)


def label_pi(fig, ax, boxes):
    ends, pi, avoid_data, _ = ax._prism_labels
    if not ends:
        return
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    pt = fig.dpi / 72.0
    A = ax.transData.transform(avoid_data)
    placed = list(boxes) + [t.get_window_extent(r)
                            for t in getattr(ax, "_clabels", [])]
    o = ax.transData.transform(OWN)
    placed.append(Bbox.from_extents(o[0] - 7 * pt, o[1] - 7 * pt,
                                    o[0] + 7 * pt, o[1] + 7 * pt))
    Pend = np.array([ax.transData.transform(e[1]) for e in ends])
    for i in np.argsort([-pi[k] for k, _ in ends]):
        k, xy = ends[i]
        best = None
        for rr in RADII_PI:
            for cx, cy in DIRS16:
                t = ax.annotate(rf"$\pi$ = {pi[k]:.2f}", xy=xy,
                                xytext=(cx * rr, cy * rr),
                                textcoords="offset points", ha="center",
                                va="center", fontsize=6.1)
                bb = t.get_window_extent(r)
                t.remove()
                bx = bb.expanded(1.08, 1.3)
                sc = 1000.0 * sum(bx.overlaps(b) for b in placed)
                sc += 8.0 * float(((A[:, 0] > bx.x0) & (A[:, 0] < bx.x1)
                                   & (A[:, 1] > bx.y0)
                                   & (A[:, 1] < bx.y1)).sum())
                if (bx.x0 < axbb.x0 + 2 or bx.x1 > axbb.x1 - 2
                        or bx.y0 < axbb.y0 + 2 or bx.y1 > axbb.y1 - 2):
                    sc += 5000.0
                c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
                if len(Pend) > 1 and int(np.argmin(
                        np.hypot(*(Pend - c).T))) != int(i):
                    sc += 300.0
                sc += 0.6 * rr
                if best is None or sc < best[0]:
                    best = (sc, (cx * rr, cy * rr), bb, rr)
        _, off, bb, rr = best
        kw = dict(xy=xy, xytext=off, textcoords="offset points",
                  fontsize=6.1, color=HDR.INK, ha="center", va="center",
                  zorder=8)
        if rr > 13.0:
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.4, color=HDR.INK,
                                    alpha=0.7, shrinkA=0.0, shrinkB=2.5)
        t = ax.annotate(rf"$\pi$ = {pi[k]:.2f}", **kw)
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
        placed.append(bb)


def key_hpd(fig, rect, legend_y, ncol=3):
    h, lab = HDR.legend_entries(False)
    h = h + [Line2D([], [], color=HDR.INK, lw=0, marker=own_marker(), ms=6,
                    mec="white", mew=0.6),
             Line2D([], [], color=HDR.INK, lw=1.0, ls=(0, (4, 2)))]
    lab = lab + ["Ownship (heading)", f"Ownship, past {OWN_PAST_S:.0f} s"]
    fig.legend(handles=h, labels=lab, loc="center",
               bbox_to_anchor=(rect[0] + rect[2] / 2, legend_y), ncol=ncol,
               frameon=False, columnspacing=1.1, handlelength=1.9,
               handletextpad=0.45, borderaxespad=0.1)
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=HDR.NORM, cmap=HDR.CMAP), cax=cax,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#8c8c8c")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 0.9])
    cb.set_ticklabels(["0", "25", "50", "75", "90"])
    cax.tick_params(labelsize=6.0, length=1.8, width=0.4, pad=1.2)
    for lv in HDR.LEVELS:
        cax.axvline(lv, color=HDR.CONTOUR_INK, lw=0.8)
    cax.set_title("HPD level L [%], swept 0\u201315 s, min over intruders; "
                  "blank above 95 %", fontsize=6.2, pad=2.5)


def key_risk(fig, rect, legend_y, ncol=2):
    h = [Line2D([], [], color=CAND_OK, lw=0.9),
         Line2D([], [], color=CAND_BAD, lw=0.9),
         Line2D([], [], color=PICK, lw=1.9, marker="o", ms=3.4, mfc="white",
                mec=PICK, mew=1.0, markevery=[1]),
         Line2D([], [], color=HDR.OBS, lw=0, marker="o", ms=2.8,
                mec="white", mew=0.4)]
    T_sel = float(CANDS[SEL]["T"])
    dot = ("dot: 15 s" if T_sel >= horizon - 1e-9
           else f"dot: plan end, {T_sel:g} s")
    lab = [f"admissible candidates ({len(ok)})",
           f"rejected candidates ({len(bad)})",
           (f"selected: least violation ({dot})" if EMERGENCY
            else f"selected path ({dot})"),
           f"detections at t = {T_FIG:g} s"]
    if not ok:                            # emergency cycle: no admissible entry
        h, lab = h[1:], lab[1:]
        lab[0] = f"candidates, none admissible ({len(bad)})"
        ncol = 1
    fig.legend(handles=h, labels=lab, loc="center",
               bbox_to_anchor=(rect[0] + rect[2] / 2, legend_y), ncol=ncol,
               frameon=False, columnspacing=1.1, handlelength=1.9,
               handletextpad=0.45, borderaxespad=0.1, labelspacing=0.32)
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0.0, VMAX),
                                     cmap=RISK_CMAP), cax=cax,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#8c8c8c")
    step = 0.25 if VMAX <= 1.5 else 0.5
    cb.set_ticks(np.arange(0.0, VMAX + 1e-9, step))
    cax.tick_params(labelsize=6.0, length=1.8, width=0.4, pad=1.2)
    cax.set_title(r"planner risk kernel / $g_0$, max over lookahead "
                  "0\u201315 s", fontsize=6.2, pad=2.5)


TITLE_A = "PRISM forecasts, HPD map"
TITLE_B = "Planner risk field, candidates"


def save(fig, stem, files):
    for ext in ("png", "pdf"):
        p = f"{OUT}/{stem}.{ext}"
        fig.savefig(p, dpi=300)
        files.append(p)
    plt.close(fig)


files: list[str] = []

# --------------------------------------------------------- combined figure
W, s_in, gap, bottom_keys, top = 7.0, 2.95, 0.22, 1.22, 0.30
Hh = bottom_keys + s_in + top
fig = plt.figure(figsize=(W, Hh))
x0 = W - 0.12 - (2 * s_in + gap)
axa = fig.add_axes([x0 / W, bottom_keys / Hh, s_in / W, s_in / Hh])
axb = fig.add_axes([(x0 + s_in + gap) / W, bottom_keys / Hh, s_in / W,
                    s_in / Hh])
setup(axa, f"(a) {TITLE_A}")
setup(axb, f"(b) {TITLE_B}")
axa.set_ylabel("y [m]", labelpad=1.5)
axb.tick_params(axis="y", labelleft=False)
av_a = draw_forecasts(axa)
k15 = draw_risk(axb)
lab_a = add_intruder_labels(axa, np.vstack([av_a, OWN[None]]))
add_intruder_labels(axb, np.vstack(
    [np.stack([CANDS[SEL]["x"], CANDS[SEL]["y"]], 1), OWN[None]]))
label_pi(fig, axa, lab_a)
key_hpd(fig, [x0 / W + 0.03, 0.18 / Hh, s_in / W - 0.06, 0.07 / Hh],
        legend_y=0.66 / Hh, ncol=3)
key_risk(fig, [(x0 + s_in + gap) / W + 0.03, 0.18 / Hh, s_in / W - 0.06,
               0.07 / Hh], legend_y=0.66 / Hh, ncol=2)
save(fig, "fig_explain_r3", files)

# --------------------------------------- stand-alone panels (same size)
PW, PH = 3.3, 4.3
for key in ("a", "b"):
    fp = plt.figure(figsize=(PW, PH))
    ax = fp.add_axes([0.17, 1.25 / PH, 0.78, 0.78 * PW / PH])
    setup(ax, TITLE_A if key == "a" else TITLE_B)
    ax.set_ylabel("y [m]", labelpad=1.5)
    if key == "a":
        av = draw_forecasts(ax)
        lab = add_intruder_labels(ax, np.vstack([av, OWN[None]]))
        label_pi(fp, ax, lab)
        key_hpd(fp, [0.20, 0.18 / PH, 0.70, 0.075 / PH], legend_y=0.68 / PH,
                ncol=2)
    else:
        draw_risk(ax)
        add_intruder_labels(ax, np.vstack(
            [np.stack([CANDS[SEL]["x"], CANDS[SEL]["y"]], 1), OWN[None]]))
        key_risk(fp, [0.20, 0.18 / PH, 0.70, 0.075 / PH], legend_y=0.68 / PH,
                 ncol=2)
    save(fp, f"fig_explain_r3_panel_{key}", files)

# ================================================================= caption
cc_c = case["arms"]["model"]["closest_crit"]
cc_a = case["arms"]["none"]["closest_crit"]
prov = META["provenance"]
selc = CANDS[SEL]
d_crit = float(np.hypot(*(arr(TRACKS[CRIT]["gt_t0"]) - OWN)))
typ_c = TYPE_NAMES[TRACKS[CRIT]["gt_type"]]
pi_c = (", ".join(f"{p:.2f}" for p in np.sort(FC[CRIT]["pi"])[::-1])
        if CRIT_FC else None)
others = [s for s in IN_VIEW if s != CRIT]
dchg = ((abs(NOW_D - PREV_D) if PREV_D is not None else 0.0) if A1
        else RULE["chosen_d_change"])
n_adm = len(ok)
why = RULE["why"]
if A1:
    n_ib, n_ibo = ROW_T["n_infeasible_by_bird"], ROW_T["n_infeasible_bird_only"]
    share = ROW_T["bird_share"]
    cf_chg = ROW_T["cf_best_changes"]
    cf_d = ROW_T["cf_best_target_d"]
    largest = ROW_T["bird_is_largest_share"]
    excl = [r for r in CYCLES["table"] if not r["emergency"]
            and r["n_infeasible_by_bird"] >= 1]
    hard_en = (f" {n_ib} of the {len(CANDS)} candidates fail the hard check on "
               f"the bird's forecast disks ({n_ibo} on the bird's disks alone), and"
               if n_ib else " No candidate fails the hard check on the bird's "
               "forecast disks yet, but")
    hard_kr = (f" 후보 {len(CANDS)}개 중 {n_ib}개가 조류의 예측 원판에서 경성 검사에 "
               f"실패하고(조류 원판만으로 {n_ibo}개), " if n_ib else
               " 조류의 예측 원판에서 경성 검사에 실패하는 후보는 아직 없지만, ")
    later_en = (f" From {excl[0]['t']:g} s the bird's disks also exclude up to "
                f"{max(r['n_infeasible_by_bird'] for r in excl)} candidates "
                "outright (notes)." if excl and excl[0]["t"] > T_FIG + 1e-9 else "")
    later_kr = (f" {excl[0]['t']:g}초부터는 조류의 원판이 후보를 최대 "
                f"{max(r['n_infeasible_by_bird'] for r in excl)}개까지 직접 제외한다"
                "(노트 참조)." if excl and excl[0]["t"] > T_FIG + 1e-9 else "")
    if not RULE["fallback"]:
        inst_en = (
            "The instant is chosen by rule: the first replanning cycle between "
            f"5 s and A's closest approach ({T_MIN_A:g} s) that is not an "
            "emergency cycle and at which the bird's forecast enters the "
            "decision (Amendment A1 of the pre-registered rule, declared after "
            f"the first draw; notes).{hard_en} the bird carries "
            f"{share * 100:.0f} % of the selected path's safety cost"
            + (", the largest share of any aircraft" if largest else "")
            + ". With the bird's disks removed the planner would select "
            + (f"a different path (lateral offset target {cf_d:+.0f} m instead of "
               f"{NOW_D:+.0f} m)." if cf_chg else "the same path.")
            + later_en)
        inst_kr = (
            f"시점은 규칙으로 정했다. 5초와 A군 최근접({T_MIN_A:g}초) 사이의 재계획 "
            "주기 가운데 비상 주기가 아니면서 조류의 예측이 결정에 들어가는 첫 "
            f"주기이다(사전 등록 규칙의 개정 A1, 첫 작도 후에 선언. 노트 참조).{hard_kr}"
            f"조류는 선택 경로 안전 비용의 {share * 100:.0f}%를 차지한다"
            + "(모든 비행체 중 가장 큰 비중)" * bool(largest)
            + ". 조류 원판을 제거하면 플래너는 "
            + (f"다른 경로(횡방향 목표 {NOW_D:+.0f} m 대신 {cf_d:+.0f} m)를 고른다."
               if cf_chg else "같은 경로를 고른다.")
            + later_kr)
    else:
        inst_en = (
            "The instant is a fallback of Amendment A1 (declared after the first "
            f"draw; notes): between 5 s and A's closest approach ({T_MIN_A:g} s) "
            "there is no non-emergency cycle at which a candidate fails the hard "
            "check on the bird's disks or the bird carries the largest share "
            "(above 0.5) of the selected path's safety cost, so the non-emergency "
            f"cycle with the largest bird share ({share * 100:.0f} %) is shown; "
            "the bird's forecast entered no non-emergency decision before A's "
            "closest approach.")
        inst_kr = (
            "시점은 개정 A1의 대체 규칙으로 정했다(첫 작도 후에 선언. 노트 참조). "
            f"5초와 A군 최근접({T_MIN_A:g}초) 사이에 조류 원판에서 경성 검사에 "
            "실패하는 후보가 있거나 조류가 선택 경로 안전 비용의 가장 큰 비중(0.5 "
            "초과)을 차지하는 비상 아닌 주기가 없어, 조류 비중이 가장 큰 비상 아닌 "
            f"주기({share * 100:.0f}%)를 보인다. A군 최근접 이전에 조류의 예측은 "
            "비상 아닌 결정에 들어가지 않았다.")
elif why == "first_rule":
    inst_en = ("The instant is chosen by rule: the first replanning cycle before "
               "A's closest approach "
               f"({T_MIN_A:g} s) at which the selected plan's lateral offset "
               "target changes by more than 30 m relative to the previously "
               f"selected plan (here from {PREV_D:.1f} m to {NOW_D:.1f} m, "
               f"{dchg:.1f} m) while the critical intruder has an open track "
               "and a PRISM forecast.")
    inst_kr = (f"시점은 규칙으로 정했다. A군 최근접({T_MIN_A:g}초) 이전에 선택 "
               "경로의 횡방향 편위 목표가 직전 선택 경로보다 30 m 넘게 바뀌고"
               f"(여기서는 {PREV_D:.1f} m에서 {NOW_D:.1f} m로, {dchg:.1f} m) 핵심 "
               "침입기가 열린 트랙과 PRISM 예측을 모두 가진 첫 재계획 주기이다.")
elif why == "fallback_max_change_eligible":
    inst_en = ("The instant is a fallback: no cycle before A's closest approach "
               f"({T_MIN_A:g} s) changed the selected plan's lateral offset by "
               "more than 30 m while the critical intruder had an open track "
               "and a PRISM forecast, so the cycle with the largest such change "
               f"({PREV_D:.1f} m to {NOW_D:.1f} m, {dchg:.1f} m) among the "
               "cycles with an open track and a forecast is shown.")
    inst_kr = (f"시점은 대체 규칙으로 정했다. A군 최근접({T_MIN_A:g}초) 이전에 핵심 "
               "침입기가 열린 트랙과 PRISM 예측을 가진 주기 가운데 횡방향 편위가 "
               "30 m 넘게 바뀐 주기가 없어, 그중 변화가 가장 큰 주기"
               f"({PREV_D:.1f} m에서 {NOW_D:.1f} m로, {dchg:.1f} m)를 보인다.")
else:
    inst_en = ("The instant is a second fallback: before A's closest approach "
               "the critical intruder never had an open track and a forecast at "
               "a replanning cycle, so the cycle with the largest lateral "
               f"change of the selected plan ({dchg:.1f} m) is shown.")
    inst_kr = ("시점은 두 번째 대체 규칙으로 정했다. A군 최근접 이전에 핵심 침입기가 "
               "열린 트랙과 예측을 가진 재계획 주기가 없어, 선택 경로의 횡방향 "
               f"변화가 가장 큰 주기({dchg:.1f} m)를 보인다.")

own_d = float((OWN - np.array([D["ref_path"]["x0"], D["ref_path"]["y0"]]))
              @ np.array([-(D["ref_path"]["y1"] - D["ref_path"]["y0"]),
                          D["ref_path"]["x1"] - D["ref_path"]["x0"]])
              / math.hypot(D["ref_path"]["x1"] - D["ref_path"]["x0"],
                           D["ref_path"]["y1"] - D["ref_path"]["y0"]))
# rejection reasons of the non-admissible candidates
REJ = {"collision": sum(1 for i in bad if CANDS[i]["collision"]),
       "boundary_only": sum(1 for i in bad if CANDS[i]["boundary"]
                            and not CANDS[i]["collision"]),
       "dynamics_only": sum(1 for i in bad if not CANDS[i]["collision"]
                            and not CANDS[i]["boundary"])}
rej = [(n, en_, kr_) for n, en_, kr_ in (
    (REJ["collision"], "a hard-radius conflict", "경성 반경 충돌"),
    (REJ["boundary_only"], "the corridor boundary", "회랑 경계"),
    (REJ["dynamics_only"], "the acceleration limits", "가속도 한계")) if n]
rej_en = ("" if not rej else (f"rejected by {rej[0][1]}" if len(rej) == 1 else
          "rejected (" + ", ".join(f"{n} by {e}" for n, e, _ in rej) + ")"))
rej_kr = ("" if not rej else (f"{rej[0][2]}로 기각된 후보" if len(rej) == 1 else
          "기각 후보(" + ", ".join(f"{k} {n}개" for n, _, k in rej) + ")"))
NUM_EN = ["no", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine"]
PLURAL_EN = {"bird": "birds", "multirotor": "multirotors",
             "fixed-wing UAS": "fixed-wing UAS", "eVTOL": "eVTOLs",
             "general-aviation aircraft": "general-aviation aircraft",
             "piece of debris": "pieces of debris",
             "unknown aircraft": "unknown aircraft"}
if others:
    groups: dict = {}
    for s_ in others:
        groups.setdefault(TRACKS[s_]["gt_type"], []).append(
            "/".join(f"{p:.2f}" for p in np.sort(FC[s_]["pi"])[::-1]))
    parts_en, parts_kr = [], []
    for gt, pis in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tp = TYPE_PHRASE[TYPE_NAMES[gt]]
        n = len(pis)
        head = (f"{art(tp)} {tp}" if n == 1 else
                f"{NUM_EN[n] if n < 10 else n} {PLURAL_EN[tp]}")
        parts_en.append(f"{head} (pi = " + "; ".join(pis) + ")")
        parts_kr.append(f"{TYPE_KR[gt]} {n}기(π = " + "; ".join(pis) + ")")
    if len(parts_en) > 1:
        lst_en = ", ".join(parts_en[:-1]) + " and " + parts_en[-1]
    else:
        lst_en = parts_en[0]
    also_en = f" Also in view: {lst_en}."
    also_kr = " " + ", ".join(parts_kr) + "도 시야에 있다."
else:
    also_en = also_kr = ""
back_en = ("back to the corridor centre line from its current lateral "
           f"offset of {own_d:.0f} m" if abs(selc["target_d"]) < 1e-6 else
           f"to a lateral offset of {selc['target_d']:.0f} m (now "
           f"{own_d:.0f} m)")
back_kr = ("회랑 중심선으로 돌아간다(현재 횡방향 편위 "
           f"{own_d:.0f} m, 목표 0 m)" if abs(selc["target_d"]) < 1e-6
           else f"횡방향 편위 {selc['target_d']:.0f} m(현재 {own_d:.0f} m)로 "
           "간다")
spd_en = ("at the desired speed of "
          f"{selc['target_speed']:.0f} m/s"
          if abs(selc["target_speed"] - PL["desired_speed"]) <= 0.05 + 1e-9 else
          f"at {selc['target_speed']:.0f} m/s (desired "
          f"{PL['desired_speed']:.0f} m/s)")
spd_kr = (f"기준 속도 {selc['target_speed']:.0f} m/s로"
          if abs(selc["target_speed"] - PL["desired_speed"]) <= 0.05 + 1e-9 else
          f"{selc['target_speed']:.0f} m/s(기준 {PL['desired_speed']:.0f} "
          "m/s)로")
T_SEL = float(selc["T"])
dot_en = ("the open dot marks 15 s, the end of the risk horizon"
          if T_SEL >= horizon - 1e-9 else
          f"the open dot marks the end of the {T_SEL:g} s plan")
dot_kr = ("빈 원은 위험 지평의 끝인 15초이다" if T_SEL >= horizon - 1e-9 else
          f"빈 원은 {T_SEL:g}초 계획의 끝이다")
if EMERGENCY:
    blk_tp = TYPE_PHRASE[TYPE_NAMES[TRACKS[BLOCK_S]["gt_type"]]]
    blk_kr = TYPE_KR[TRACKS[BLOCK_S]["gt_type"]]
    n_blk = CONFLICT_BY_SLOT[BLOCK_S]
    cand_en = (
        f"Grey lines are the {len(CANDS)} candidate paths. At this cycle none "
        f"is admissible: {REJ['collision']} conflict with a hard radius"
        + (f" and {REJ['dynamics_only']} exceed the acceleration limits"
           if REJ["dynamics_only"] else "")
        + (f" and {REJ['boundary_only']} leave the corridor" if REJ["boundary_only"]
           else "")
        + ", so the planner applies its emergency rule and selects the "
        f"least-violating candidate (blue): {T_SEL:g} s long, {back_en}, "
        f"{spd_en}; {dot_en}. The conflicts come from the forecast disks of "
        f"{art(blk_tp)} {blk_tp} {BLOCK_D:.0f} m from the ownship ({n_blk} of the "
        f"{REJ['collision']} conflicting candidates); the intervention at this "
        "instant answers that aircraft, not the critical intruder.")
    cand_kr = (
        f"회색 선은 후보 경로 {len(CANDS)}개이다. 이 주기에는 허용 후보가 없다. "
        f"{REJ['collision']}개는 경성 반경과 충돌하고"
        + (f" {REJ['dynamics_only']}개는 가속도 한계를 넘는다" if REJ["dynamics_only"]
           else "")
        + (f" {REJ['boundary_only']}개는 회랑을 벗어난다" if REJ["boundary_only"]
           else "")
        + ". 따라서 플래너는 비상 규칙에 따라 위반이 가장 작은 후보를 고른다"
        f"(파란 선). 길이 {T_SEL:g}초이며, {spd_kr} {back_kr}. {dot_kr}. 충돌은 "
        f"자기 비행체에서 {BLOCK_D:.0f} m 떨어진 {blk_kr}의 예측 원판에서 온다"
        f"(충돌 후보 {REJ['collision']}개 중 {n_blk}개). 이 시점의 개입은 핵심 "
        "침입기가 아니라 그 비행체에 대한 것이다.")
else:
    cand_en = (
        f"Grey lines are the {len(CANDS)} candidate paths: {n_adm} admissible "
        "(dark)" + (f" and {len(bad)} {rej_en} (light)" if bad else "")
        + f". Blue is the selected path: {T_SEL:g} s long, {back_en}, {spd_en}; "
        f"{dot_en}.")
    cand_kr = (
        f"회색 선은 후보 경로 {len(CANDS)}개이다. 진한 선은 허용 후보 {n_adm}개"
        + (f"이고, 옅은 선은 {rej_kr} {len(bad)}개" if bad else "")
        + f"이다. 파란 선은 선택 경로이다. 길이 {T_SEL:g}초이며, {spd_kr} "
        f"{back_kr}. {dot_kr}.")
cc_rel_en = ("In C the closest approach to this intruder "
             f"({cc_c['sep']:.1f} m at {round(cc_c['t'], 2):g} s) comes "
             + ("after" if cc_c["t"] > T_FIG else "before") + " this instant; in "
             f"A it was {cc_a['sep']:.1f} m at {round(cc_a['t'], 2):g} s.")
cc_rel_kr = (f"C군에서 이 침입기와의 최근접({cc_c['sep']:.1f} m, "
             f"{round(cc_c['t'], 2):g}초)은 이 시점 "
             + ("뒤" if cc_c["t"] > T_FIG else "앞") + "에 온다. A군에서는 "
             f"{cc_a['sep']:.1f} m({round(cc_a['t'], 2):g}초)였다.")
crit_en = (f"The critical intruder (the {typ_c.lower()} of Figure 11) has "
           f"pi = {pi_c} and is {d_crit:.0f} m from the ownship."
           if CRIT_FC else
           f"The critical intruder (the {typ_c.lower()} of Figure 11, "
           f"{d_crit:.0f} m from the ownship) has no forecast at this cycle; "
           "its true position and future are drawn (square, dashed).")
crit_kr = (f"핵심 침입기(그림 11의 {TYPE_KR[TRACKS[CRIT]['gt_type']]})는 "
           f"π = {pi_c}이고 자기 비행체에서 {d_crit:.0f} m 떨어져 있다."
           if CRIT_FC else
           f"핵심 침입기(그림 11의 {TYPE_KR[TRACKS[CRIT]['gt_type']]}, 자기 "
           f"비행체에서 {d_crit:.0f} m)는 이 주기에 예측이 없다. 실제 위치와 "
           "미래를 그렸다(사각형, 파선).")

en = (
    "Figure 9. Prediction and path selection in arm C (PRISM) of the case in "
    f"Figure 11, at t = {T_FIG:g} s. {inst_en} "
    "(a) Predicted position distributions of the intruders in view, drawn as "
    "the time-swept highest predictive density (HPD) map of Figure X. For "
    "each intruder the K = 3 mixture (full covariance with rho) is evaluated "
    "from 0 to 15 s ahead, each cell takes its HPD level, and the colour shows "
    "the smallest level over time and over intruders; dark lines mark the "
    "50 % and 90 % levels, and cells above 95 % are blank. For display only, "
    "the map is evaluated every 0.1 s: each branch's mean and covariance "
    "matrix are interpolated linearly between the 0.5 s forecast steps, "
    "starting at t from the last detection with half the covariance of the "
    "first step. Black lines are branch means (width proportional to pi_k), "
    "dots are the detections in the 10 s history used by the predictor, "
    "dashed lines are the true futures, and the triangle is the ownship with "
    f"its past {OWN_PAST_S:.0f} s dashed. {crit_en}{also_en} (b) The planner's "
    "risk field at the same cycle, computed as the planner does. Each branch "
    f"with pi_k >= {pi_min:g} becomes a disk that moves along the branch mean, "
    "with the isotropic sigma = sqrt((sigma_x^2 + sigma_y^2)/2) floored at "
    f"{floor:g} m and the gain g0·pi_k (g0 = {G0:g}); a detected track "
    f"without a forecast keeps a static disk (sigma = {RK['sigma_base']:g} m, "
    "gain g0). The kernel g·exp(-s^2/(2 sigma^2)) is not normalised; s is "
    "the clearance beyond r_hard + r_veh + the NMAC standoff "
    f"({hr_cfg:g} + {vr:g} + {st:g} m), and s = 0 inside it. The colour "
    "shows the kernel sum divided by g0, maximised over the planner's "
    "evaluation times 0-15 s ahead (every 0.5 s); a candidate is charged "
    f"only at its own sample times. {cand_en} The HPD "
    f"map in (a) is not the planner's risk field; (b) is. {cc_rel_en}")
kr = (
    f"그림 9. 그림 11 사례의 C군(PRISM)에서 t = {T_FIG:g}초의 예측과 경로 "
    f"선택이다. {inst_kr} (a) 시야 안 침입기들의 "
    "예측 위치 분포를 그림 X의 시간 누적 최고밀도영역(HPD) 지도로 그렸다. "
    "침입기마다 K = 3 혼합 분포(ρ를 포함한 완전 공분산)를 0~15초 앞까지 계산하고 "
    "각 셀의 HPD 수준을 구한다. 색은 시간과 침입기에 걸친 최소 수준이다. 진한 "
    "선은 50%와 90% 수준이고, 95%를 넘는 셀은 비워 두었다. 지도는 표시용으로만 "
    "0.1초 간격으로 계산하였다. 0.5초 예측 스텝 사이에서 갈래별 평균과 공분산 "
    "행렬을 선형 보간하며, t에서는 마지막 탐지 위치와 첫 스텝 공분산의 절반에서 "
    "출발한다. 검은 선은 갈래 평균이며 굵기는 π_k에 비례한다. 점은 예측기가 "
    "사용한 10초 이력의 탐지이고, 파선은 실제 미래이다. 삼각형은 자기 비행체이며 "
    f"검은 파선은 지난 {OWN_PAST_S:.0f}초의 항적이다. {crit_kr}{also_kr} (b) 같은 "
    f"주기에서 플래너가 계산하는 위험장이다. π_k ≥ {pi_min:g}인 갈래마다 갈래 "
    "평균을 따라 움직이는 원판을 둔다. σ는 √((σx² + σy²)/2)로 등방화하고 "
    f"{floor:g} m 하한을 두며, 이득은 g0·π_k(g0 = {G0:g})이다. 예측이 없는 탐지 "
    f"트랙은 정적 원판(σ = {RK['sigma_base']:g} m, 이득 g0)으로 남는다. 커널 "
    "g·exp(−s²/(2σ²))는 정규화하지 않는다. s는 경성 반경, 기체 반경, NMAC "
    f"이격({hr_cfg:g} + {vr:g} + {st:g} m)을 넘는 여유 거리이며, 그 안에서는 "
    "0이다. 색은 커널 합을 g0로 나눈 값이며, 플래너의 평가 시각(0~15초 앞, "
    "0.5초 간격)에 걸친 최댓값이다. 후보 경로는 자기 표본 시각에서만 비용을 "
    f"받는다. {cand_kr} (a)의 HPD 지도는 플래너의 "
    f"위험장이 아니며, (b)가 위험장이다. {cc_rel_kr}")

gate = META["gate"]
flags_now = CYC[k_now]["crit"]


def a1_table_lines() -> list[str]:
    tb = CYCLES["table"]
    out = [f"{'t':>5s} {'emerg':>5s} {'n_valid':>7s} {'inf_bird':>8s} "
           f"{'bird_only':>9s} {'share':>6s} {'sel_saf':>9s} "
           f"{'largest(slot:share)':>19s} {'tgt_d':>7s} {'T':>5s} {'cf_chg':>6s} "
           f"{'cf_d':>7s} {'fc':>2s} {'use':>3s} hard-check failures by slot"]
    for r in tb:
        out.append(
            f"{r['t']:5.1f} {int(r['emergency']):5d} {r['n_valid']:7d} "
            f"{r['n_infeasible_by_bird']:8d} {r['n_infeasible_bird_only']:9d} "
            f"{r['bird_share']:6.3f} {r['sel_safety_raw']:9.3g} "
            f"{(str(r['largest_share_slot']) + ':' + format(r['largest_share'], '.3f')):>19s} "
            f"{r['sel_target_d']:7.1f} {r['sel_T']:5.1f} {int(r['cf_best_changes']):6d} "
            f"{r['cf_best_target_d']:7.1f} "
            f"{int(r['crit_forecast']):2d} {int(r['crit_planner_uses']):3d} "
            f"{r.get('infeasible_by_slot', {})}")
    return out


if A1:
    tb = CYCLES["table"]
    val_line = (
        f"per-cycle reproduction of the planner over the {len(tb)} cycles: "
        f"collision-flag mismatches {max(r['val_collision_mismatch'] for r in tb)}, "
        f"raw safety max rel diff {max(r['val_safety_max_rel'] for r in tb):.1e}, "
        f"total cost max abs diff {max(r['val_total_max_abs'] for r in tb):.1e}, "
        f"selected candidate reproduced in all cycles "
        f"{all(r['val_best_reproduced'] for r in tb)}, emergency flag reproduced "
        f"{all(r['val_emergency_reproduced'] for r in tb)}.")
    instant_note = (
        f"instant: t = {T_FIG:g} s (log index t0 = {D['t0']}), by AMENDMENT A1 of "
        f"Step 5 in {RULE_FILE} (original text sha256 {SHA_ORIG}, amended file "
        f"sha256 {RULE_META['sha256_amended']}; declared post hoc after the first "
        "draw at the Step 5 instant, which was an emergency cycle). Amended rule: "
        "over the C-arm cycles with 5 s <= t < t_min compute (i) the number of "
        "candidates whose hard check fails on a disk of the bird, (ii) the bird's "
        "share of the selected candidate's raw safety cost, (iii) the emergency "
        "flag; take the first non-emergency cycle with (i) >= 1 or ((ii) > 0.5 and "
        "the largest intruder share); else the non-emergency cycle with the "
        f"largest (ii). Result: {RULE['why']} (fallback {RULE['fallback']}); "
        f"non-emergency cycles {RULE['n_non_emergency']} of {RULE['n_cycles']}; "
        "cycles in which the bird's forecast enters a non-emergency decision: "
        f"{RULE['n_non_emergency_cycles_where_bird_enters']}; bird drove a "
        "non-emergency decision before A's closest approach: "
        f"{RULE['bird_drove_a_non_emergency_decision']}. At the instant: "
        f"n_infeasible_by_bird {ROW_T['n_infeasible_by_bird']} (bird only "
        f"{ROW_T['n_infeasible_bird_only']}), bird_share {ROW_T['bird_share']:.3f} "
        f"(largest share slot {ROW_T['largest_share_slot']} = "
        f"{ROW_T['largest_share']:.3f}; shares by slot {ROW_T['shares_by_slot']}), "
        f"selected target_d {ROW_T['sel_target_d']:.1f} m (previous cycle "
        f"{PREV_D:.1f} m), selected candidate's raw safety cost "
        f"{ROW_T['sel_safety_raw']:.3g} (bird part {ROW_T['sel_safety_bird_raw']:.3g}), "
        f"selection changes without the bird's disks: {ROW_T['cf_best_changes']} "
        f"(counterfactual target_d {ROW_T['cf_best_target_d']:.1f} m, "
        f"{ROW_T['cf_n_valid']} admissible without the bird); flags at the "
        f"instant {flags_now}. The original Step 5 result was t = "
        f"{RULE['step5_original']['chosen_t']} s ('{RULE['step5_original']['why']}', "
        f"|delta target_d| {RULE['step5_original']['chosen_d_change']:.1f} m), an "
        "emergency cycle whose 255 hard-radius conflicts came from the "
        "general-aviation aircraft (slot 15); its scene is kept as "
        f"data/scene_{IDX}_step5_original.json. Per-cycle table: {CYC_PATH}.")
else:
    instant_note = (
        f"instant: t = {T_FIG:g} s (log index t0 = {D['t0']}); pre-registered Step "
        f"5 of {RULE_FILE} (sha256 {SHA_ORIG}): the first C-arm cycle "
        f"with t < t_min ({T_MIN_A:g} s) at which |target_d(t) - target_d(t - 0.5 s)| "
        "> 30 m while the critical intruder has an open track (obs_track_id >= 0) "
        "and a PRISM forecast; fallbacks: largest change among eligible cycles, "
        f"then largest change regardless. Result: {why}; first_rule_t "
        f"{RULE['first_rule_t']} (change {RULE['first_rule_d_change']}); max eligible "
        f"change {RULE['max_elig_d_change']} at {RULE['max_elig_t']} s; max change "
        f"{RULE['max_any_d_change']} at {RULE['max_any_t']} s; chosen change "
        f"{dchg:.3f} m (target_d {PREV_D} -> {NOW_D}); flags at the instant "
        f"{flags_now} (planner_uses = detected at t0 and at least one branch with "
        f"pi >= {pi_min:g}).")
notes = [
    "--- notes (not part of the caption) ---",
    instant_note,
    *(["A1 PER-CYCLE TABLE (arm C, 5 s <= t < t_min; emerg = no admissible "
       "candidate; inf_bird = candidates failing the hard check on a bird disk; "
       "bird_only = of these, admissible but for the bird; share = bird's share "
       "of the selected candidate's raw safety cost; sel_saf = that raw safety "
       "cost (kernel sum x dt, base gain g0 = 3); largest = slot with the "
       "largest share; tgt_d/T = selected plan; cf_chg/cf_d = selection changes "
       "without the bird's disks / the target_d it would select; fc/use = bird "
       "forecast exists / used by the planner):",
       *["  " + ln for ln in a1_table_lines()], val_line] if A1 else []),
    "reference: the stored own_intervening flag of the planner (controller."
    "cycle_diagnostics) uses |target_d| > lateral_step/2 or target_speed < "
    "0.95 x desired (relative to the nominal), and aggregate_r3's default "
    "stratum uses max |own_d| > 30 m over the horizon; the 30 m change "
    "relative to the previous plan is the rule fixed for this figure.",
    f"critical intruder: slot {CRIT}, {typ_c}, "
    f"{['corridor transit', 'delivery waypoints', 'loiter/hover', 'search wander', 'reactive avoid', 'passive drift'][META['critical']['gt_intent']]} intent, "
    f"open track id {TRACKS[CRIT]['track_id_t0']}, "
    + (f"{len(FC[CRIT]['used_pos'])} detections of the open track in the 10 s "
       "history; " if CRIT_FC else "no forecast at this cycle; ")
    + f"distance to the ownship at t {d_crit:.1f} m (ground truth); closest "
    f"approach in C {cc_c['sep']:.1f} m at {round(cc_c['t'], 2):g} s, in A "
    f"{cc_a['sep']:.1f} m at {round(cc_a['t'], 2):g} s.",
    f"forecasts in view (a): {[(s, FC[s]['type']) for s in IN_VIEW]}; "
    f"forecasts not used by the planner at this cycle (no detection at t0): "
    f"{dropped_no_det} ("
    + ("none of them in the view" if not set(dropped_no_det) & set(IN_VIEW)
       else f"in the view: {sorted(set(dropped_no_det) & set(IN_VIEW))}")
    + f"); branches pruned by pi < {pi_min:g}: "
    f"{pruned if pruned else 'none'}.",
    f"planner at this cycle: {D['n_total']} candidates, {D['n_valid']} "
    f"admissible; rejected: {REJ}; emergency cycle (no admissible candidate, "
    f"least-violation fallback): {EMERGENCY}. Selected candidate index {SEL}: T "
    f"{selc['T']:g} s, target_d {selc['target_d']:.2f} m (previous cycle "
    f"{PREV_D}), target speed {selc['target_speed']:.2f} m/s, total cost "
    f"{selc['total_cost']:.4f}, min clearance {selc['min_clearance']:.1f} m, "
    f"violation {selc['violation']:.3f}, valid {selc['valid']}, collision "
    f"{selc['collision']}. [V6] hard-radius conflicts recomputed from the "
    "recorded disks reproduce the planner's collision flags; conflicting "
    f"candidates by slot {CONFLICT_BY_SLOT} (slot types "
    f"{ {s: TYPE_NAMES[TRACKS[s]['gt_type']] for s in CONFLICT_BY_SLOT} }); "
    f"nearest aircraft at the instant: slot {NEAR_S} "
    f"({TYPE_NAMES[TRACKS[NEAR_S]['gt_type']]}) at {NEAR_D:.0f} m; the C arm's "
    "emergency cycles (n_valid = 0) in this episode: "
    f"{[c['t'] for c in CYC if c['n_valid'] == 0]}.",
    f"risk field: disks from risk.disks_batch at lookaheads 0..15 s every "
    f"0.5 s ({DISKS.shape[1]} disks: {sum(1 for s_, k in SMAP if k < 0)} "
    f"static, {sum(1 for s_, k in SMAP if k >= 0)} branch); planner values "
    f"vehicle_radius {vr:g} m, hard_radius {hr_cfg:g} m, soft_standoff "
    f"{st:g} m, sigma_floor {floor:g} m, pi_min {pi_min:g}, iso "
    f"{RK['iso']}, gain_mode {RK['gain_mode']}, base gain g0 {G0:g}; grid "
    f"{RISK_N} x {RISK_N}; max kernel / g0 in view {R.max():.3f}.",
    f"validation: [V1] safety cost of all {len(CANDS)} candidates recomputed "
    "from the recorded disks with the drawn kernel equals the planner's raw "
    f"safety cost (max rel diff {VAL['safety_recomputed']['max_rel_diff']:.1e}); "
    "[V2] every disk follows from the recorded forecasts and detections by "
    "the stated reduction (max |diff| centre "
    f"{VAL['disks_from_forecasts']['max_abs_centre_m']:.1e} m, sigma "
    f"{VAL['disks_from_forecasts']['max_abs_sigma_m']:.1e} m, gain "
    f"{VAL['disks_from_forecasts']['max_abs_gain']:.1e}; retained-branch set "
    "equal); [V3] forecast frame angle = build_sample frame angle; [V4] HPD: "
    "fine-grid mass >= 0.995 and positive-definite interpolated covariances "
    "at every time point: "
    + "; ".join(f"slot {s} min mass {VAL['hpd'][str(s)]['min_fine_grid_mass']}"
                for s in IN_VIEW) + ".",
    f"HPD method and parameters: make_fig_hdr.py (sha256 {sha256(HDR_PATH)}) "
    "StepHPD / swept_levels / display_schedule (0.1 s, node 0 = last "
    "detection with 0.5 x the first step's covariance) / prism_nodes, "
    "display grid 720 x 720; combined over intruders by the minimum level. "
    "All K = 3 branches enter the HPD map (the planner's pi pruning does not "
    "apply to the forecast distribution).",
    f"view: world metres, x {EXT[0]:.0f}..{EXT[1]:.0f}, y {EXT[2]:.0f}.."
    f"{EXT[3]:.0f}; holds the ownship and its past {OWN_PAST_S:.0f} s, the "
    "critical intruder's swept 95 % region, detections and true future, and "
    "the selected path up to 15 s.",
    f"data: {SCENE} (sha256 {sha256(SCENE)}), {CASE} (sha256 {sha256(CASE)}); "
    f"selector/recorder Paper/figures/select_case_r3.py (sha256 "
    f"{prov['script_sha256']}); gate file {GATE} (sha256 {sha256(GATE)}); "
    f"candidates {CAND_PATH} (sha256 {sha256(CAND_PATH)}); selection "
    f"{SELN_PATH} (sha256 {sha256(SELN_PATH)}; conditions used "
    f"{SELN['conditions_in_order']}, survivors {SELN['n_survivors']}, selected "
    f"rank {SELN['selection']['rank_1based']}).",
    f"model: {prov['model_local']} (sha256 {prov['model_sha256']}); code: "
    f"snapshot {prov['snapshot_dir']}, fingerprint "
    f"{prov['code_fp_runner_layout']['fp']} (runner layout, = code_fp of the "
    f"batch rows) / {prov['code_fp_local_layout']['fp']} (local layout of the "
    "replay).",
    "gate (batch row / replay row, RTX 4090 runner / RTX 3080), every "
    "deterministic field equal for all three arms ("
    + ", ".join(f"{LABEL[a]} {gate['n_deterministic_fields'][a]}"
                for a in ('none', 'cv', 'model')) + " fields); key fields: "
    + "; ".join(f"{LABEL[a]}: " + ", ".join(
        f"{k} {gate['key_fields'][a][k]['batch']}/{gate['key_fields'][a][k]['replay']}"
        for k in ("min_sep_20hz", "min_sep", "sep_p5", "nmac_events",
                  "union20_events_152", "path_len_m", "n_cycles", "pred_logged"))
        for a in ("none", "cv", "model")) + ".",
    "supersedes: the ep244 version (fig_explain_r3_ep244*, "
    "caption_explain_r3_ep244.txt; instant 66.5 s = the submitted figure's "
    "time) and the submitted Figure 9 (pre-review model, 24-aircraft "
    "density-40 variant of the scenario).",
]
cap_path = f"{OUT}/caption_explain_r3.txt"
with open(cap_path, "w", encoding="utf-8") as f:
    f.write("[EN]\n" + en + "\n\n[KR]\n" + kr + "\n\n" + "\n".join(notes) + "\n")
files.append(cap_path)

# ================================================================ manifest
man = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST)\
    else {}
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
old = man.get("fig_explain_r3")
if old is not None and "fig_explain_r3_ep244" not in man and\
        "ep244" in json.dumps(old):
    man["fig_explain_r3_ep244"] = man.pop("fig_explain_r3")
man["fig_explain_r3"] = {
    "description": "Prediction (HPD swept map) and path selection (planner "
                   f"risk field, candidates) in arm C at t = {T_FIG:g} s of "
                   f"the rule-selected case (episode {IDX}, r3_{COND}); "
                   "replaces the ep244 version.",
    "script": {"path": os.path.abspath(__file__).replace("\\", "/"),
               "sha256": sha256(os.path.abspath(__file__))},
    "hpd_code": {"path": HDR_PATH, "sha256": sha256(HDR_PATH)},
    "sources": {"scene": {"path": SCENE, "sha256": sha256(SCENE)},
                "case": {"path": CASE, "sha256": sha256(CASE)},
                "gate": {"path": GATE, "sha256": sha256(GATE)},
                "candidates": {"path": CAND_PATH, "sha256": sha256(CAND_PATH)},
                "selection": {"path": SELN_PATH, "sha256": sha256(SELN_PATH)},
                "rule": {"path": RULE_FILE, "sha256_original": SHA_ORIG,
                         "sha256_current": sha256(RULE_FILE)}},
    "instant": {"t_s": T_FIG, "t0": D["t0"], "rule": RULE, "amended": A1,
                "target_d_prev": PREV_D, "target_d_now": NOW_D},
    "rule_file": {"path": RULE_FILE, "sha256_original": SHA_ORIG,
                  "sha256_current": sha256(RULE_FILE), "amended": AMENDED},
    **({"cycles_table": {"path": CYC_PATH, "sha256": sha256(CYC_PATH),
                         "n_cycles": len(CYCLES["table"])}} if A1 else {}),
    "values": {"critical_slot": CRIT, "critical_type": typ_c,
               "critical_has_forecast": CRIT_FC,
               "critical_pi": FC[CRIT]["pi"].tolist() if CRIT_FC else None,
               "critical_distance_m": round(d_crit, 1),
               "forecasts_in_view": IN_VIEW,
               "n_candidates": len(CANDS), "n_admissible": n_adm,
               "emergency_cycle": EMERGENCY,
               "conflicting_candidates_by_slot": CONFLICT_BY_SLOT,
               "blocking_slot": BLOCK_S, "blocking_distance_m": BLOCK_D,
               "nearest_slot": NEAR_S, "nearest_distance_m": round(NEAR_D, 1),
               "rejected_by": REJ, "selected": {
                   "index": SEL, "T": selc["T"], "target_d": selc["target_d"],
                   "target_speed": selc["target_speed"],
                   "total_cost": selc["total_cost"],
                   "min_clearance": selc["min_clearance"]},
               "risk_max_over_g0_in_view": round(float(R.max()), 4),
               "base_gain": G0},
    "validation": VAL,
    "view": {"ext_m": [round(v, 1) for v in EXT], "tick_m": TICK},
    "files": files,
    "generated_utc": now}
man["generated_utc"] = now
with open(MANIFEST + ".tmp", "w", encoding="utf-8") as f:
    json.dump(man, f, indent=1, ensure_ascii=False, default=float)
os.replace(MANIFEST + ".tmp", MANIFEST)
for p in files:
    print("[saved]", p)
print("[saved]", MANIFEST)
