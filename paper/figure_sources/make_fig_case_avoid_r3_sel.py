# -*- coding: utf-8 -*-
"""PRISM fig_case_avoid r3 (rule-selected case): the paired closed-loop
example of Figure 11, drawn from the episode that the pre-registered rule of
D:/PRISM/Paper/figures_r3/case_rule_r3.txt selects (select_case_r3.py). It
replaces the hand-picked episode 244 (make_fig_case_avoid_r3.py; its outputs
were copied to fig_case_avoid_r3_ep244*.{png,pdf} and
caption_case_avoid_r3_ep244.txt before this script overwrote the stems).

Data (D:/PRISM/Paper/figures_r3/data): candidates_<cond>.json (pool, survivor
table, selection), case_<idx>.json (20 Hz trajectories and separations of the
three arms, 2 Hz GT of all agents, per-cycle plans), gate_<idx>.json
(hook-free replay == batch row for every deterministic field), scene_<idx>.json
(the Figure 9 instant). The gate, the recording-run row compare and the rule
file's sha256 are asserted here.

Panels (same axes, equal aspect, world metres, a window around arm A's
closest approach)
  (a) A: static observation, (b) B: CV, (c) C: PRISM
      ownship track (black, arrowheads), the critical intruder as it flew in
      that arm's own closed loop (orange; marks every 10 s), other traffic
      (grey), the closest approach to the critical intruder (connector +
      value; red cross inside the NMAC radius); in (c) a hollow blue dot on
      the ownship marks the Figure 9 instant.
  (d) 20 Hz separation to the critical intruder per arm (solid) and to the
      nearest aircraft of any kind (dotted), NMAC radius dashed, Figure 9
      instant as a vertical line.
Arm names follow the revised Table 11; no internal label (condition tag, arm
key, run name) is drawn.

Outputs (D:/PRISM/Paper/figures_r3): fig_case_avoid_r3.{png,pdf} (letters
drawn), fig_case_avoid_r3_panel_{a,b,c,d}.{png,pdf}, fig_case_avoid_r3_legend
.{png,pdf}, caption_case_avoid_r3.txt (caption EN/KR + notes: rule text,
survivor table, gate table, data files, model sha256, snapshot fingerprint),
and the "fig_case_avoid_r3" entry of fig_case_r3_manifest.json (the ep244
entry is kept as "fig_case_avoid_r3_ep244").

Run: python
     D:/PRISM/Paper/figures/make_fig_case_avoid_r3_sel.py [--cond ood40] [--index N]
"""
from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = (os.environ["PRISM_ROOT"] + '')
OUT = f"{ROOT}/Paper/figures_r3"
RULE_FILE = f"{OUT}/case_rule_r3.txt"
MANIFEST = f"{OUT}/fig_case_r3_manifest.json"
NMAC = 152.4
ARMS = ("none", "cv", "model")
LETTER = {"none": "a", "cv": "b", "model": "c"}
TYPE_NAMES = ["Bird", "Multirotor", "Fixed-wing UAS", "eVTOL",
              "General aviation", "Debris", "Unknown"]
TYPE_KR = ["조류", "멀티로터", "고정익 무인기", "eVTOL", "경항공기(GA)",
           "부유 잔해", "미확인 기체"]
INTENT_EN = ["corridor transit", "delivery waypoints", "loiter/hover",
             "search wander", "reactive avoid", "passive drift"]
INTENT_KR = ["회랑 순항", "배달 경유점 추종", "체공/정지비행", "탐색 배회",
             "반응 회피", "수동 표류"]
TYPE_PHRASE = {"Bird": "bird", "Multirotor": "multirotor",
               "Fixed-wing UAS": "fixed-wing UAS", "eVTOL": "eVTOL",
               "General aviation": "general-aviation aircraft",
               "Debris": "piece of debris", "Unknown": "unknown aircraft"}


def art(w: str) -> str:
    return "an" if w[0].lower() in "aeiou" else "a"


def plural(n: int, w: str) -> str:
    return f"{n} {w}" + ("" if n == 1 else "s")


def ts(v) -> str:
    """Times: 2 decimals, trailing zeros dropped (32.95, 34.2, 10)."""
    return f"{round(float(v), 2):g}"

ap = argparse.ArgumentParser()
ap.add_argument("--cond", default="ood40")
ap.add_argument("--index", type=int, default=None)
ARGS = ap.parse_args()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.4,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.6,
    "axes.linewidth": 0.5, "axes.edgecolor": "#8c8c8c",
    "xtick.color": "#8c8c8c", "ytick.color": "#8c8c8c",
    "xtick.labelcolor": "#404040", "ytick.labelcolor": "#404040",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "grid.linewidth": 0.4, "grid.color": "#e8e8e8", "grid.alpha": 1.0,
    "axes.axisbelow": True, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, ORANGE, BLUE, RED = "#1a1a1a", "#E69F00", "#0072B2", "#D55E00"
GRAY_OTH = "#cfcfcf"
ARM_COL = {"none": "#000000", "cv": "#009E73", "model": "#0072B2"}
ARM_LS = {"none": "solid", "cv": (0, (5, 1.6, 1.2, 1.6)), "model": "solid"}
STROKE = [pe.withStroke(linewidth=1.8, foreground="white")]


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
DATA = f"{OUT}/data/case_{IDX}.json"
GATE = f"{OUT}/data/gate_{IDX}.json"
SCENE = f"{OUT}/data/scene_{IDX}.json"
D = json.load(open(DATA, encoding="utf-8"))
G = json.load(open(GATE, encoding="utf-8"))
S9 = json.load(open(SCENE, encoding="utf-8"))
META = D["meta"]
assert G["pass"] and META["gate"]["pass"] and META["record_check"]["pass"],\
    "gate / record check did not pass"
assert META["episode"]["index"] == IDX == G["episode_index"]
RULE_TXT = open(RULE_FILE, encoding="utf-8").read()
RULE_META = META["rule"]
SHA_ORIG = RULE_META.get("sha256_original", RULE_META.get("sha256"))
assert SHA_ORIG == CAND["rule_sha256"] == SELN["rule_sha256"],\
    "original rule sha256 disagrees between the data files"
ORIG_TXT = RULE_META.get("original_text", RULE_META["text"])
assert RULE_TXT.startswith(ORIG_TXT),\
    "the rule file no longer starts with the original (pre-registered) text"
AMENDED = bool(RULE_META.get("amended"))
assert sha256(RULE_FILE) == (RULE_META["sha256_amended"] if AMENDED
                             else SHA_ORIG), "rule file changed"
A1 = META.get("fig9_amended")            # Amendment A1 result (None before it)
CYC_PATH = f"{OUT}/data/scene_{IDX}_cycles.json"
CYCLES = (json.load(open(CYC_PATH, encoding="utf-8"))
          if A1 and os.path.exists(CYC_PATH) else None)
CRIT = META["critical"]["slot"]
LABEL = META["arm_labels"]
T_MIN_A = float(META["critical"]["t_min_a"])
T_FIG = float(S9["t"])
SEL = SELN["selection"]
SURV = sorted(SELN["survivors"], key=lambda r: (r["c_min_rec"], r["index"]))
assert (SEL["index"] == IDX and SEL["cond"] == COND) or ARGS.index is not None
assert META["episode"]["condition"] == f"r3_{COND}"
CONDS_USED = SELN["conditions_in_order"]           # ood40 first, then the extension
EXTENDED = len(CONDS_USED) > 1
COND_EN, COND_KR = META["episode"]["condition_en"], META["episode"]["condition_kr"]
N_AG = META["episode"]["n_agents"]
COND_DESC = {"ood40": ("the nominal-load OOD condition", "표준 부하 OOD 조건"),
             "id24": ("the in-distribution condition with 24 aircraft",
                      "24기 분포 내 조건"),
             "id40": ("the in-distribution condition", "분포 내 조건"),
             "ood100": ("the saturated OOD condition", "포화 OOD 조건"),
             "sparse": ("the sparse OOD condition", "희소 OOD 조건")}

A = {}
for a in ARMS:
    X = D["arms"][a]
    A[a] = {"t20": arr(X["t20"]), "own20": arr(X["own20"]),
            "crit20": arr(X["crit20"]), "sep_crit": arr(X["sep_crit20"]),
            "sep_min": arr(X["sep_min20"]),
            "sep_min_slot": np.asarray(X["sep_min_slot20"]),
            "t2": arr(X["t2"]), "gt2": arr(X["gt2"]),
            "cc": X["closest_crit"], "em": X["episode_min"],
            "cycles": X["cycles"], "row": X["row_key"],
            "gt_type": X["gt_type"]}
    assert round(X["episode_min"]["sep"], 1) == X["row_key"]["min_sep_20hz"]
assert abs(A["none"]["cc"]["t"] - T_MIN_A) < 1e-6
assert round(A["none"]["cc"]["sep"], 1) == META["critical"]["min_sep_a"]

# ------------------------------------------------------------- the window
T_MAP = (max(0.0, math.floor((T_MIN_A - 20.0) / 5.0) * 5.0),
         min(90.0, math.ceil((T_MIN_A + 10.0) / 5.0) * 5.0))
T_MARKS = [float(t) for t in range(0, 91, 10) if T_MAP[0] <= t <= T_MAP[1]]
T_STRIP = (max(0.0, T_MAP[0] - 10.0), min(90.0, T_MAP[1] + 10.0))
T_ARROWS = [T_MAP[0] + f * (T_MAP[1] - T_MAP[0]) for f in (0.2, 0.5, 0.8)]

# ----------------------------------------------------- facts for the caption
own_dev = {}
for a in ("cv", "model"):
    t = A[a]["t20"]
    m = t <= T_MIN_A + 1e-9
    d = np.hypot(*(A[a]["own20"][m] - A["none"]["own20"][m]).T)
    k = int(np.argmin(np.abs(t - T_MIN_A)))
    own_dev[a] = {"max_m": round(float(d.max()), 1),
                  "t_of_max": round(float(t[m][int(np.argmax(d))]), 2),
                  "at_t_min_m": round(float(np.hypot(
                      *(A[a]["own20"][k] - A["none"]["own20"][k]))), 1)}


def first_diff(a: str, thr: float):
    d = np.hypot(*(A[a]["crit20"] - A["none"]["crit20"]).T)
    i = np.nonzero(d > thr)[0]
    return None if not i.size else round(float(A[a]["t20"][i[0]]), 2)


def crit_dev_until_tmin(a: str) -> float:
    """Critical intruder: max 2 Hz distance between arm a and arm A up to
    A's closest approach (frames alive in both)."""
    g, g0, t2 = A[a]["gt2"][CRIT], A["none"]["gt2"][CRIT], A[a]["t2"]
    m = t2 <= T_MIN_A + 1e-9
    d = np.hypot(*(g[m] - g0[m]).T)
    return round(float(np.nanmax(d)), 1)


def first_traffic_diff(a: str, thr: float):
    g, g0 = A[a]["gt2"], A["none"]["gt2"]
    d = np.hypot(g[..., 0] - g0[..., 0], g[..., 1] - g0[..., 1])   # [N,F]
    mism = np.isfinite(g).all(-1) ^ np.isfinite(g0).all(-1)
    hit = (np.nan_to_num(d, nan=0.0) > thr) | mism
    if not hit.any():
        return None, []
    f = int(np.nonzero(hit.any(axis=0))[0][0])
    return round(float(A[a]["t2"][f]), 2), np.nonzero(hit[:, f])[0].tolist()


def first_plan_diff(a: str):
    for ca, cn in zip(A[a]["cycles"], A["none"]["cycles"]):
        ka = (ca["sel"]["T"], ca["sel"]["target_d"], ca["sel"]["target_speed"])
        kn = (cn["sel"]["T"], cn["sel"]["target_d"], cn["sel"]["target_speed"])
        if ka != kn:
            return ca["t"]
    return None


def own_offset_at(a: str, t: float) -> float | None:
    if t is None:
        return None
    i = int(np.argmin(np.abs(A[a]["t20"] - t)))
    return round(float(np.hypot(*(A[a]["own20"][i] - A["none"]["own20"][i]))), 2)


ATT_C = META["attributability_C_vs_A"]
ATT_B = META["attributability_B_vs_A"]
FACTS = {
    "episode_index": IDX, "condition": META["episode"]["condition"],
    "critical_slot": CRIT,
    "critical_type": TYPE_NAMES[META["critical"]["gt_type"]],
    "critical_intent": INTENT_EN[META["critical"]["gt_intent"]],
    "closest_to_critical": {a: {"sep_m": round(A[a]["cc"]["sep"], 1),
                                "t_s": round(A[a]["cc"]["t"], 2)} for a in ARMS},
    "episode_min": {a: {"sep_m": round(A[a]["em"]["sep"], 1),
                        "t_s": round(A[a]["em"]["t"], 2),
                        "slot": A[a]["em"]["slot"],
                        "type": TYPE_NAMES[A[a]["gt_type"][A[a]["em"]["slot"]]]}
                    for a in ARMS},
    "time_below_nmac_crit_s": {a: round(float(np.nansum(A[a]["sep_crit"] < NMAC))
                                        * 0.05, 2) for a in ARMS},
    "ownship_offset_from_A_until_A_cpa_m": own_dev,
    "critical_intruder_max_dev_from_A_until_A_cpa_2hz_m": {
        a: crit_dev_until_tmin(a) for a in ("cv", "model")},
    "first_plan_difference_vs_A_s": {a: first_plan_diff(a)
                                     for a in ("cv", "model")},
    "critical_intruder_first_differs_from_A_s": {
        a: {"gt_0.01m": first_diff(a, 0.01), "gt_1m": first_diff(a, 1.0),
            "gt_30m": first_diff(a, 30.0)} for a in ("cv", "model")},
    "traffic_first_differs_from_A_gt_1m": {
        a: dict(zip(("t_s", "slots"), first_traffic_diff(a, 1.0)))
        for a in ("cv", "model")},
    "attributability_C_vs_A": ATT_C, "attributability_B_vs_A": ATT_B,
    "fig9_instant_s": T_FIG, "fig9_rule": S9["fig9_rule"],
    "rows": {a: A[a]["row"] for a in ARMS},
    "a_arm_event": {k: META["critical"][k] for k in
                    ("t_entry_a", "t_exit_a", "lead_track_s_a", "track_id_a")},
    "selection": {"conditions_in_order": CONDS_USED, "extended": EXTENDED,
                  "pool_sizes": SELN["pool_sizes"],
                  "n_survivors_by_cond": SELN["n_survivors_by_cond"],
                  "n_survivors": SELN["n_survivors"],
                  "rank_1based": SEL["rank_1based"],
                  "dropped_by_cond_status": SELN["dropped_by_cond_status"]},
}
for a in ("cv", "model"):
    FACTS["traffic_first_differs_from_A_gt_1m"][a]["own_offset_m"] = own_offset_at(
        a, FACTS["traffic_first_differs_from_A_gt_1m"][a]["t_s"])
assert abs(FACTS["critical_intruder_max_dev_from_A_until_A_cpa_2hz_m"]["model"]
           - ATT_C["crit_max_dev_2hz_m"]) < 0.11
assert FACTS["critical_intruder_max_dev_from_A_until_A_cpa_2hz_m"]["model"] <= 30.0

# ------------------------------------------------------------- the view
pts = []
for a in ARMS:
    m = (A[a]["t20"] >= T_MAP[0]) & (A[a]["t20"] <= T_MAP[1])
    cr = A[a]["crit20"][m]
    pts += [A[a]["own20"][m], cr[np.isfinite(cr).all(1)],
            np.array([A[a]["cc"]["own"], A[a]["cc"]["intr"]])]
    em = A[a]["em"]
    if em["slot"] != CRIT and T_MAP[0] <= em["t"] <= T_MAP[1]:
        pts.append(np.array([em["own"], em["intr"]]))
P = np.vstack(pts)
lo, hi = P.min(0), P.max(0)
pad = 0.06 * (hi - lo).max()
XL = (lo[0] - pad, hi[0] + pad)
YL = (lo[1] - pad, hi[1] + pad)


def fmt_m(v: float) -> str:
    return f"{v:.1f} m"


def fmt_t(v: float) -> str:
    return f"{round(float(v), 2):g} s"


# ------------------------------------------------------- label placement
def place(ax, xy, text, cands, avoid, placed, max_score=None, **kw):
    r = ax.figure.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    best = None
    for dx, dy, ha, va in cands:
        t = ax.annotate(text, xy=xy, xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, **kw)
        bb = t.get_window_extent(r)
        sc = 100.0 * sum(bb.overlaps(b) for b in placed)
        if len(avoid):
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
    if max_score is not None and best[0] > max_score:
        best[1].remove()
        return None
    placed.append(best[2])
    return best[1]


AROUND = [(8, -2, "left", "top"), (8, 2, "left", "bottom"),
          (-8, -2, "right", "top"), (-8, 2, "right", "bottom"),
          (0, -9, "center", "top"), (0, 9, "center", "bottom"),
          (14, 8, "left", "bottom"), (-14, 8, "right", "bottom"),
          (14, -8, "left", "top"), (-14, -8, "right", "top")]
NEAR = [(4, 3, "left", "bottom"), (4, -3, "left", "top"),
        (-4, 3, "right", "bottom"), (-4, -3, "right", "top"),
        (0, 6, "center", "bottom"), (0, -6, "center", "top"),
        (7, 0, "left", "center"), (-7, 0, "right", "center")]


# ---------------------------------------------------------------- drawing
def draw_map(ax, a: str, title: str | None):
    X = A[a]
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_aspect("equal")
    ax.apply_aspect()
    ax.grid(True)
    tr_data = []
    t2, gt2 = X["t2"], X["gt2"]
    mw = (t2 >= T_MAP[0]) & (t2 <= T_MAP[1])
    for s in range(gt2.shape[0]):
        if s == CRIT:
            continue
        tr = gt2[s][mw]
        ok = np.isfinite(tr).all(1)
        if ok.sum() < 2:
            continue
        ax.plot(tr[ok, 0], tr[ok, 1], color=GRAY_OTH, lw=0.55, zorder=2,
                solid_capstyle="round")
        tr_data.append(tr[ok])
    t20 = X["t20"]
    m20 = (t20 >= T_MAP[0]) & (t20 <= T_MAP[1])
    cr = X["crit20"][m20]
    ax.plot(cr[:, 0], cr[:, 1], color=ORANGE, lw=1.3, zorder=4,
            solid_capstyle="round")
    own = X["own20"][m20]
    ax.plot(own[:, 0], own[:, 1], color=INK, lw=1.3, zorder=6,
            solid_capstyle="round")
    tr_data += [cr[np.isfinite(cr).all(1)], own]
    for tm in T_ARROWS:
        i = int(np.argmin(np.abs(t20 - tm)))
        if i + 4 < len(t20):
            ax.annotate("", xy=X["own20"][i + 4], xytext=X["own20"][i], zorder=6,
                        arrowprops=dict(arrowstyle="-|>", lw=0, color=INK,
                                        mutation_scale=7.5, shrinkA=0,
                                        shrinkB=0))
    # Figure 9 instant on the ownship of arm C
    if a == "model" and T_MAP[0] <= T_FIG <= T_MAP[1]:
        i = int(np.argmin(np.abs(t20 - T_FIG)))
        ax.plot(*X["own20"][i], marker="o", ms=3.6, mfc="white", mec=BLUE,
                mew=1.0, ls="none", zorder=7)
    # closest approach to the critical intruder
    cc = X["cc"]
    o, q = np.asarray(cc["own"]), np.asarray(cc["intr"])
    below = cc["sep"] < NMAC
    ax.plot([o[0], q[0]], [o[1], q[1]], color=RED if below else INK,
            lw=0.9, zorder=8)
    tr_data.append(np.linspace(o, q, 20))
    mid = (o + q) / 2
    if below:
        ax.plot(*mid, marker="x", ms=6.5, mec=RED, mew=1.5, ls="none",
                zorder=9)
    marks = []
    for tm in T_MARKS:
        i = int(np.argmin(np.abs(t20 - tm)))
        p = X["crit20"][i]
        if np.all(np.isfinite(p)):
            ax.plot(*p, marker="o", ms=2.4, mfc=ORANGE, mec="white", mew=0.4,
                    ls="none", zorder=5)
            marks.append((tm, p))
    em = X["em"]
    em_in_view = em["slot"] != CRIT and T_MAP[0] <= em["t"] <= T_MAP[1]
    if em_in_view:
        o2, q2 = np.asarray(em["own"]), np.asarray(em["intr"])
        ax.plot([o2[0], q2[0]], [o2[1], q2[1]], color="#555555", lw=0.8,
                ls=(0, (1, 1.2)), zorder=8)
        tr_data.append(np.linspace(o2, q2, 20))
    avoid = ax.transData.transform(np.vstack(tr_data))
    placed = []
    place(ax, mid, f"{fmt_m(cc['sep'])}\n{fmt_t(cc['t'])}", AROUND, avoid,
          placed, fontsize=6.4, color=RED if below else INK,
          fontweight="bold", zorder=9, path_effects=STROKE, linespacing=1.05)
    if em_in_view:
        typ = TYPE_NAMES[X["gt_type"][em["slot"]]].lower()
        place(ax, (o2 + q2) / 2,
              f"episode min {fmt_m(em['sep'])}\n({typ}, {fmt_t(em['t'])})",
              AROUND, avoid, placed, fontsize=5.8, color="#333333", zorder=9,
              path_effects=STROKE, linespacing=1.05)
    for tm, p in marks:
        place(ax, p, f"{tm:.0f} s", NEAR, avoid, placed, max_score=6.0,
              fontsize=5.6, color=INK, zorder=7, path_effects=STROKE)
    if title:
        ax.set_title(title, pad=3.5)
    ax.set_xlabel("x [m]", labelpad=1.5)


def strip_ylim() -> float:
    vals = []
    for a in ARMS:
        m = (A[a]["t20"] >= T_STRIP[0]) & (A[a]["t20"] <= T_STRIP[1])
        for k in ("sep_crit", "sep_min"):
            v = A[a][k][m]
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(np.percentile(v, 98))
    top = max(400.0, min(1500.0, math.ceil(max(vals) / 100.0) * 100.0))
    return top


Y_TOP = strip_ylim()
STRIP_LABELS = [(0, 6, "center", "bottom"), (6, 6, "left", "bottom"),
                (-6, 6, "right", "bottom"), (0, -7, "center", "top"),
                (8, -6, "left", "top"), (-8, -6, "right", "top"),
                (14, 10, "left", "bottom"), (-14, 10, "right", "bottom")]


def draw_strip(ax):
    ax.set_xlim(*T_STRIP)
    ax.set_ylim(0, Y_TOP)
    avoid_pts = []
    for a in ARMS:
        X = A[a]
        t = X["t20"]
        m = (t >= T_STRIP[0]) & (t <= T_STRIP[1])
        ax.plot(t, X["sep_crit"], color=ARM_COL[a], lw=1.1, ls=ARM_LS[a],
                zorder=4)
        ax.plot(t, X["sep_min"], color=ARM_COL[a], lw=0.6, ls=(0, (1, 1.3)),
                alpha=0.9, zorder=3)
        for k in ("sep_crit", "sep_min"):
            v = X[k][m]
            ok = np.isfinite(v) & (v <= Y_TOP)
            avoid_pts.append(np.stack([t[m][ok], v[ok]], 1))
    ax.axhline(NMAC, color=RED, lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax.axvline(T_FIG, color=BLUE, lw=0.6, ls=(0, (3.2, 1.8)), zorder=1)
    ax.grid(True)
    ax.set_xlabel("t [s]", labelpad=1.5)
    ax.set_ylabel("separation [m]", labelpad=1.5)
    avoid = ax.transData.transform(np.vstack(avoid_pts))
    placed = []
    ax.text(T_STRIP[0] + 0.4, NMAC - 14, "NMAC 152.4 m", fontsize=5.8,
            color=RED, va="top", zorder=6, path_effects=STROKE)
    r = ax.figure.canvas.get_renderer()
    placed.append(ax.texts[-1].get_window_extent(r))
    ax.text(T_FIG + 0.3, Y_TOP * 0.985, f"{T_FIG:g} s (Figure 9)", fontsize=5.6,
            color=BLUE, va="top", ha="left", zorder=6, path_effects=STROKE)
    placed.append(ax.texts[-1].get_window_extent(r))
    for a in ARMS:
        cc = A[a]["cc"]
        ax.plot(cc["t"], cc["sep"], marker="o", ms=3.2, mfc=ARM_COL[a],
                mec="white", mew=0.5, zorder=5)
        place(ax, (cc["t"], min(cc["sep"], Y_TOP)), fmt_m(cc["sep"]),
              STRIP_LABELS, avoid, placed, fontsize=6.0, color=ARM_COL[a],
              zorder=6, path_effects=STROKE)
    em = A["model"]["em"]
    if em["slot"] != CRIT and T_STRIP[0] <= em["t"] <= T_STRIP[1]:
        typ = TYPE_NAMES[A["model"]["gt_type"][em["slot"]]].lower()
        ax.plot(em["t"], em["sep"], marker="o", ms=3.0, mfc="white",
                mec=ARM_COL["model"], mew=0.9, zorder=5, clip_on=False)
        place(ax, (em["t"], em["sep"]), f"{fmt_m(em['sep'])} ({typ})",
              STRIP_LABELS, avoid, placed, fontsize=5.8, color=ARM_COL["model"],
              zorder=6, path_effects=STROKE)


def legend_handles():
    h = [Line2D([], [], color=INK, lw=1.3, label="Ownship"),
         Line2D([], [], color=ORANGE, lw=1.3, marker="o", ms=2.6,
                mec="white", mew=0.4,
                label=f"Critical intruder ({TYPE_PHRASE[FACTS['critical_type']]}), "
                      "marks every 10 s"),
         Line2D([], [], color=GRAY_OTH, lw=0.9, label="Other traffic"),
         Line2D([], [], color=INK, lw=0.9,
                label="Closest approach to the critical intruder"),
         Line2D([], [], color=RED, lw=0.9, marker="x", ms=5, mew=1.3,
                label="Closest approach inside 152.4 m"),
         Line2D([], [], color=BLUE, lw=0, marker="o", ms=3.6, mfc="white",
                mec=BLUE, mew=1.0,
                label=f"Figure 9 instant ({T_FIG:g} s, arm C)")]
    h2 = [Line2D([], [], color=ARM_COL[a], lw=1.1, ls=ARM_LS[a],
                 label=f"{LABEL[a]}") for a in ARMS]
    h2 += [Line2D([], [], color="#555555", lw=0.6, ls=(0, (1, 1.3)),
                  label="nearest aircraft of any kind"),
           Line2D([], [], color=RED, lw=0.7, ls=(0, (4, 2)),
                  label="NMAC radius")]
    return h, h2


def save(fig, stem, files):
    for ext in ("png", "pdf"):
        p = f"{OUT}/{stem}.{ext}"
        fig.savefig(p, dpi=300)
        files.append(p)
    plt.close(fig)


files: list[str] = []
h1, h2 = legend_handles()
LEG1 = dict(loc="center", ncol=3, frameon=False, columnspacing=1.3,
            handlelength=2.1, handletextpad=0.4, borderaxespad=0.0)
LEG2 = dict(loc="center", ncol=5, frameon=False, columnspacing=1.3,
            handlelength=2.4, handletextpad=0.4, borderaxespad=0.0)

# ---------------------------------------------------------- combined figure
W = 7.0
span_x, span_y = XL[1] - XL[0], YL[1] - YL[0]
mw_in = 2.02
mh_in = mw_in * span_y / span_x
left, gap = 0.52, 0.16
leg1_h, title_pad, strip_gap, strip_h = 0.50, 0.24, 0.62, 1.10
leg2_h, bottom = 0.24, 0.36
H = leg1_h + title_pad + mh_in + strip_gap + strip_h + bottom + leg2_h
fig = plt.figure(figsize=(W, H))
y_strip = bottom + leg2_h
y_map = y_strip + strip_h + strip_gap
x0 = left + (W - left - 0.12 - 3 * mw_in - 2 * gap) / 2
for i, a in enumerate(ARMS):
    ax = fig.add_axes([(x0 + i * (mw_in + gap)) / W, y_map / H,
                       mw_in / W, mh_in / H])
    draw_map(ax, a, f"({LETTER[a]}) {LABEL[a]}")
    if i == 0:
        ax.set_ylabel("y [m]", labelpad=1.5)
    else:
        ax.tick_params(axis="y", labelleft=False)
axs = fig.add_axes([x0 / W, y_strip / H, (3 * mw_in + 2 * gap) / W,
                    strip_h / H])
draw_strip(axs)
axs.set_title("(d) Separation, 20 Hz", pad=3.0, fontsize=7.0)
fig.legend(handles=h1, bbox_to_anchor=(0.5, 1 - (leg1_h / 2) / H), **LEG1)
fig.legend(handles=h2, bbox_to_anchor=(0.5, (leg2_h / 2 + 0.02) / H), **LEG2)
save(fig, "fig_case_avoid_r3", files)

# ------------------------------------------- stand-alone panels, same size
PANEL_W = 2.45
ax_w = 0.735 * PANEL_W
PANEL_H = ax_w * span_y / span_x + 0.47 + 0.25
for a in ARMS:
    fp = plt.figure(figsize=(PANEL_W, PANEL_H))
    axp = fp.add_axes([0.235, 0.47 / PANEL_H, 0.735,
                       (PANEL_H - 0.47 - 0.25) / PANEL_H])
    draw_map(axp, a, LABEL[a])
    axp.set_ylabel("y [m]", labelpad=1.5)
    save(fp, f"fig_case_avoid_r3_panel_{LETTER[a]}", files)
fs = plt.figure(figsize=(W, 1.75))
axs = fs.add_axes([0.075, 0.24, 0.905, 0.66])
draw_strip(axs)
axs.set_title("Separation, 20 Hz", pad=3.0, fontsize=7.0)
save(fs, "fig_case_avoid_r3_panel_d", files)
fl = plt.figure(figsize=(W, 0.72))
fl.legend(handles=h1, bbox_to_anchor=(0.5, 0.64), **LEG1)
fl.legend(handles=h2, bbox_to_anchor=(0.5, 0.17), **LEG2)
save(fl, "fig_case_avoid_r3_legend", files)

# ------------------------------------------------------------------ caption
F = FACTS
cc, em = F["closest_to_critical"], F["episode_min"]
tb = F["time_below_nmac_crit_s"]
od = F["ownship_offset_from_A_until_A_cpa_m"]
fd = F["critical_intruder_first_differs_from_A_s"]
td = F["traffic_first_differs_from_A_gt_1m"]
pdiff = F["first_plan_difference_vs_A_s"]
ev = F["a_arm_event"]
prov = META["provenance"]
typ_c = TYPE_PHRASE[F["critical_type"]]
kr_crit = TYPE_KR[META["critical"]["gt_type"]]
POOL_ENTRY = next(x for x in CAND["pool"] if x["index"] == IDX)
N_QUAL_A = int(POOL_ENTRY["n_qualifying_segments_a"])
# batch rows of this episode (read only): the screen's unavoidable flags
ROWS = {}
for a in ARMS:
    with open(f"{ROOT}/Logs/Planner/r3_{COND}/{a}.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            r_ = json.loads(ln)
            if r_.get("index") == IDX:
                ROWS[a] = r_
                break
assert len(ROWS) == 3


def screen_unavoidable(a: str, slot: int):
    segs = [s for s in ROWS[a]["pairwise"] if s["slot"] == slot]
    return None if not segs else any(s["unavoidable_new"] for s in segs)
# other aircraft within 152.4 m in any arm (20 Hz GT), by slot
OTHER = {a: {p["slot"]: (round(p["min_sep"], 1), round(p["t"], 2), p["gt_type"])
             for p in D["arms"][a]["per_slot_min"]
             if p["slot"] != CRIT and p["min_sep"] is not None
             and p["min_sep"] < NMAC} for a in ARMS}
SLOTS_OTHER = sorted(set().union(*[set(v) for v in OTHER.values()]),
                     key=lambda s: min(OTHER[a][s][0] for a in ARMS
                                       if s in OTHER[a]))
ARM_SHORT = {"none": "A", "cv": "B", "model": "C"}
other_en_parts, other_kr_parts = [], []
COMMON_SLOT = None                # an aircraft all three arms pass inside NMAC
for s in SLOTS_OTHER:
    arms_s = [a for a in ARMS if s in OTHER[a]]
    if len(arms_s) != 3:
        continue                  # single-arm passes: notes only
    gt = next(OTHER[a][s][2] for a in arms_s)
    tp, kr_tp = TYPE_PHRASE[TYPE_NAMES[gt]], TYPE_KR[gt]
    vals_en = ", ".join(f"{ARM_SHORT[a]} {OTHER[a][s][0]:.1f} m at "
                        f"{ts(OTHER[a][s][1])} s" for a in arms_s)
    vals_kr = ", ".join(f"{ARM_SHORT[a]} {OTHER[a][s][0]:.1f} m/"
                        f"{ts(OTHER[a][s][1])}초" for a in arms_s)
    unav = [screen_unavoidable(a, s) for a in ARMS]
    avoid_en = (" the batch screen counts this pass as avoidable in every arm"
                if all(u is False for u in unav) else
                " the batch screen counts this pass as unavoidable in "
                + ", ".join(ARM_SHORT[a] for a, u in zip(ARMS, unav) if u))
    avoid_kr = ("배치 선별은 세 군 모두에서 이 통과를 회피 가능으로 본다"
                if all(u is False for u in unav) else
                ", ".join(ARM_SHORT[a] for a, u in zip(ARMS, unav) if u)
                + "군에서 회피 불가능으로 본다")
    t_lo = min(OTHER[a][s][1] for a in arms_s)
    t_hi = max(OTHER[a][s][1] for a in arms_s)
    if COMMON_SLOT is None and A["model"]["em"]["slot"] == s:
        COMMON_SLOT = s
        other_en_parts.append(
            f"The episode minimum of C, {OTHER['model'][s][0]:.1f} m at "
            f"{ts(OTHER['model'][s][1])} s (open circle in (d)), is {art(tp)} {tp} "
            f"that all three arms pass inside the NMAC radius at "
            f"{t_lo:.0f}-{t_hi:.0f} s ({vals_en});{avoid_en}.")
        other_kr_parts.append(
            f"C의 에피소드 최소 이격 {OTHER['model'][s][0]:.1f} m"
            f"({ts(OTHER['model'][s][1])}초, (d)의 빈 원)은 {kr_tp}와의 이격이며, "
            f"세 군 모두 이 {kr_tp} 옆을 {t_lo:.0f}~{t_hi:.0f}초에 NMAC 반경 안에서 "
            f"지나간다({vals_kr}). {avoid_kr}.")
    else:
        other_en_parts.append(f"All three arms also pass {art(tp)} {tp} inside "
                              f"the NMAC radius ({vals_en});{avoid_en}.")
        other_kr_parts.append(f"세 군 모두 {kr_tp} 옆을 NMAC 반경 안에서 "
                              f"지나간다({vals_kr}). {avoid_kr}.")
if N_QUAL_A > 1:
    other_en_parts.append(
        f"In A the {typ_c} is the "
        + ("smaller of two" if N_QUAL_A == 2 else f"smallest of {N_QUAL_A}")
        + " approaches below 45 m that the batch screen does not flag as "
        "unavoidable, so the rule makes it the critical intruder.")
    other_kr_parts.append(
        f"A군에서 {kr_crit}는 회피 불가능으로 표시되지 않은 45 m 미만 접근 "
        f"{N_QUAL_A}개 가운데 가장 작아 규칙에 따라 핵심 침입기가 된다.")
other_en = (" " + " ".join(other_en_parts)) if other_en_parts else ""
other_kr = (" " + " ".join(other_kr_parts)) if other_kr_parts else ""
FACTS["other_encounters_below_nmac"] = {
    a: {str(s): {"min_sep_m": v[0], "t_s": v[1], "type": TYPE_NAMES[v[2]]}
        for s, v in OTHER[a].items()} for a in ARMS}
FACTS["a_arm_qualifying_segments"] = N_QUAL_A
int_en = F["critical_intent"]
int_kr = INTENT_KR[META["critical"]["gt_intent"]]
dev_c = F["critical_intruder_max_dev_from_A_until_A_cpa_2hz_m"]["model"]
dev_b = F["critical_intruder_max_dev_from_A_until_A_cpa_2hz_m"]["cv"]
own_c = od["model"]["at_t_min_m"]
n_surv = SELN["n_survivors"]
rank = SEL["rank_1based"]
n_pool_first = SELN["pool_sizes"]["ood40"]
n_eps_first = SELN["n_episodes"]["ood40"]
n_surv_first = SELN["n_survivors_by_cond"]["ood40"]
n_pool_total = sum(SELN["pool_sizes"][c] for c in CONDS_USED)
n_eps_total = sum(SELN["n_episodes"][c] for c in CONDS_USED)
below = {a: cc[a]["sep_m"] < NMAC for a in ARMS}
if EXTENDED:
    ext = CONDS_USED[1:]
    ext_en = (" The rule was applied to the nominal-load OOD condition first "
              f"({n_eps_first} episodes, {plural(n_pool_first, 'candidate')}, "
              f"{plural(n_surv_first, 'survivor')}); because fewer than 5 "
              "survived, the pool was extended in the pre-registered order to "
              + ", ".join(f"{COND_DESC[c][0]} ({SELN['n_episodes'][c]} episodes, "
                          f"{plural(SELN['pool_sizes'][c], 'candidate')}, "
                          f"{plural(SELN['n_survivors_by_cond'][c], 'survivor')})"
                          for c in ext)
              + f", giving {plural(n_surv, 'survivor')} in all.")
    ext_kr = (f" 규칙은 표준 부하 OOD 조건에 먼저 적용하였다({n_eps_first}개 "
              f"에피소드, 후보 {n_pool_first}개, 생존 {n_surv_first}개). 생존자가 "
              "5개 미만이어서 사전 등록한 순서대로 후보군을 "
              + ", ".join(f"{COND_DESC[c][1]}({SELN['n_episodes'][c]}개 에피소드, "
                          f"후보 {SELN['pool_sizes'][c]}개, 생존 "
                          f"{SELN['n_survivors_by_cond'][c]}개)" for c in ext)
              + f"으로 넓혀 생존자는 모두 {n_surv}개이다.")
else:
    ext_en = ext_kr = ""


def ts(v):
    return f"{round(float(v), 2):g}"


def tw(a, b):
    return f"{a:.0f}-{b:.0f}"


em_c = em["model"]
em_note_en = ""
em_note_kr = ""
if em_c["slot"] != CRIT and em_c["slot"] != COMMON_SLOT:
    em_tp = TYPE_PHRASE[em_c["type"]]
    em_note_en = (f" The episode minimum of C is {em_c['sep_m']:.1f} m, to "
                  f"{art(em_tp)} {em_tp} at {ts(em_c['t_s'])} s (open circle in "
                  "(d)).")
    em_note_kr = (f" C의 에피소드 최소 이격은 {ts(em_c['t_s'])}초의 "
                  f"{TYPE_KR[TYPE_NAMES.index(em_c['type'])]}와의 "
                  f"{em_c['sep_m']:.1f} m이다((d)의 빈 원).")

# closing sentences: what the difference in separation is (Amendment A1)
if A1 is not None:
    row_t = A1["row_at_instant"]
    n_enter = A1["n_non_emergency_cycles_where_bird_enters"]
    TB = CYCLES["table"] if CYCLES else []
    # contiguous runs of emergency cycles, each with its dominant conflicting
    # slot (the aircraft whose disks fail the most candidates) and whether
    # the bird's disks are among the conflicts
    EM_RUNS = []
    for r in TB:
        if not r["emergency"]:
            continue
        ibs = {int(k): v for k, v in r.get("infeasible_by_slot", {}).items()}
        dom = max(ibs, key=ibs.get) if ibs else None
        if EM_RUNS and abs(r["t"] - EM_RUNS[-1]["t_end"] - 0.5) < 1e-6:
            g = EM_RUNS[-1]
            g["t_end"] = r["t"]
            g["doms"].append(dom)
            g["bird"] |= CRIT in ibs
            g["n"] += 1
        else:
            EM_RUNS.append({"t_start": r["t"], "t_end": r["t"], "doms": [dom],
                            "bird": CRIT in ibs, "n": 1})
    for g in EM_RUNS:
        ds = [d for d in g["doms"] if d is not None]
        g["dom"] = max(set(ds), key=ds.count) if ds else None
    FACTS["emergency_runs_C"] = [{"t_start": g["t_start"], "t_end": g["t_end"],
                                  "n_cycles": g["n"], "dominant_slot": g["dom"],
                                  "bird_in_conflicts": g["bird"]} for g in EM_RUNS]
    first_run = next((g for g in EM_RUNS if g["dom"] is not None
                      and g["dom"] != CRIT), None)
    later_bird_runs = [g for g in EM_RUNS if g["bird"] and g is not first_run]
    excl = [r for r in TB if not r["emergency"] and r["n_infeasible_by_bird"] >= 1]
    n_tot = TB[0]["n_total"] if TB else None

    def tr_en(g):
        return f"{g['t_start']:g}-{g['t_end']:g} s"

    def tr_kr(g):
        return f"{g['t_start']:g}~{g['t_end']:g}초"

    if first_run is not None:
        fr_tp = TYPE_PHRASE[TYPE_NAMES[A["model"]["gt_type"][first_run["dom"]]]]
        fr_kr = TYPE_KR[A["model"]["gt_type"][first_run["dom"]]]
    if A1["bird_drove_a_non_emergency_decision"]:
        tail_en = (
            f"Here the critical intruder's path in C stays within {dev_c:.1f} m "
            f"of its path in A up to {ts(T_MIN_A)} s, so the difference in "
            "separation is the ownship's. "
            + (f"The ownship of C is displaced first by its manoeuvre around "
               f"{art(fr_tp)} {fr_tp} ({tr_en(first_run)}, emergency cycles caused "
               "by that aircraft's forecast disks); " if first_run else "")
            + f"from {A1['chosen_t']:g} s the bird's forecast enters the planner's "
            f"decisions (Figure 9; {n_enter} non-emergency cycles before "
            f"{ts(T_MIN_A)} s)"
            + (f", from {excl[0]['t']:g} s its disks exclude up to "
               f"{max(r['n_infeasible_by_bird'] for r in excl)} of the {n_tot} "
               "candidates outright" if excl else "")
            + (", and " + " and ".join(
                f"a further run of emergency cycles ({tr_en(g)}) has the bird's "
                "disks among the hard conflicts" for g in later_bird_runs)
               if later_bird_runs else "")
            + f". At {ts(T_MIN_A)} s the ownship of C is {own_c:.1f} m from the "
            "ownship of A.")
        tail_kr = (
            f"이 사례에서 핵심 침입기의 C군 경로는 {ts(T_MIN_A)}초까지 A군 경로의 "
            f"{dev_c:.1f} m 안에 있다. 따라서 이격의 차이는 자기 비행체의 것이다. "
            + (f"C의 자기 비행체는 먼저 {fr_kr} 회피 기동({tr_kr(first_run)}, 그 "
               "비행체의 예측 원판이 만든 비상 주기)으로 변위된다. "
               if first_run else "")
            + f"{A1['chosen_t']:g}초부터 조류의 예측이 플래너의 결정에 들어가고"
            f"(그림 9, {ts(T_MIN_A)}초 이전 비상 아닌 주기 {n_enter}개)"
            + (f", {excl[0]['t']:g}초부터는 조류의 원판이 후보 {n_tot}개 중 최대 "
               f"{max(r['n_infeasible_by_bird'] for r in excl)}개를 직접 제외한다"
               if excl else "")
            + ("".join(f". {tr_kr(g)}의 비상 주기에서도 조류의 원판이 경성 충돌에 "
                       "포함된다" for g in later_bird_runs) if later_bird_runs else "")
            + f". {ts(T_MIN_A)}초에 C의 자기 비행체는 A의 자기 비행체에서 "
            f"{own_c:.1f} m 떨어져 있다.")
    else:
        tail_en = (
            f"Here the critical intruder's path in C stays within {dev_c:.1f} m "
            f"of its path in A up to {ts(T_MIN_A)} s, so the difference in "
            "separation is the ownship's: a displaced path, not anticipation of "
            "the bird. "
            + (f"The ownship of C is moved off A's path by its manoeuvre around "
               f"{art(fr_tp)} {fr_tp} ({tr_en(first_run)}, emergency cycles "
               "caused by that aircraft's forecast disks), " if first_run else "")
            + f"and between 5 s and {ts(T_MIN_A)} s the bird's forecast enters no "
            "non-emergency decision of the planner (no candidate is excluded by "
            "its disks and it never carries the largest share of the selected "
            f"path's safety cost; Figure 9 notes). At {ts(T_MIN_A)} s the ownship "
            f"of C is {own_c:.1f} m from the ownship of A.")
        tail_kr = (
            f"이 사례에서 핵심 침입기의 C군 경로는 {ts(T_MIN_A)}초까지 A군 경로의 "
            f"{dev_c:.1f} m 안에 있다. 따라서 이격의 차이는 자기 비행체의 것이다. "
            "다만 조류를 예견한 회피가 아니라 변위된 경로이다. "
            + (f"C의 자기 비행체는 {fr_kr} 회피 기동({tr_kr(first_run)}, 그 비행체의 "
               "예측 원판이 만든 비상 주기)으로 A의 경로에서 벗어나고, "
               if first_run else "")
            + f"5초부터 {ts(T_MIN_A)}초까지 조류의 예측은 플래너의 비상 아닌 결정에 "
            "들어가지 않는다(조류 원판이 제외하는 후보가 없고, 선택 경로 안전 "
            f"비용에서 가장 큰 비중을 차지한 적이 없다. 그림 9 노트). {ts(T_MIN_A)}초에 "
            f"C의 자기 비행체는 A의 자기 비행체에서 {own_c:.1f} m 떨어져 있다.")
else:
    tail_en = (
        f"Here the critical intruder's path in C stays within {dev_c:.1f} m "
        f"of its path in A up to {ts(T_MIN_A)} s, so the difference in separation "
        f"is the ownship's: at {ts(T_MIN_A)} s the ownship of C is {own_c:.1f} m "
        "from the ownship of A.")
    tail_kr = (
        f"이 사례에서 핵심 침입기의 C군 경로는 {ts(T_MIN_A)}초까지 A군 경로의 "
        f"{dev_c:.1f} m 안에 있다. 따라서 이격의 차이는 자기 비행체의 것이다. "
        f"{ts(T_MIN_A)}초에 C의 자기 비행체는 A의 자기 비행체에서 {own_c:.1f} m "
        "떨어져 있다.")

en = (
    f"Figure 11. Paired closed-loop example chosen by a pre-registered rule. "
    f"Episode {IDX} of the {COND_EN}, {N_AG} aircraft, 90 s. The episode was "
    "flown three times from the same scenario seed and initial state with the "
    "same planner and the three risk inputs of Table 11. (a) A: static "
    f"observation, (b) B: CV, (c) C: PRISM; {tw(*T_MAP)} s, world metres, same "
    f"axes. Black: ownship. Orange: the critical intruder ({art(typ_c)} "
    f"{typ_c}, {int_en} intent) as it flew in each arm's own closed loop, marks "
    "every 10 s. Grey: other traffic. The connector marks the closest approach "
    "to the critical intruder, computed at 20 Hz on ground truth: "
    f"A {cc['none']['sep_m']:.1f} m at {ts(cc['none']['t_s'])} s, "
    f"B {cc['cv']['sep_m']:.1f} m at {ts(cc['cv']['t_s'])} s, "
    f"C {cc['model']['sep_m']:.1f} m at {ts(cc['model']['t_s'])} s. A red cross "
    "marks a closest approach inside the NMAC radius (152.4 m). The hollow "
    f"blue dot in (c) is the ownship at the instant of Figure 9 ({T_FIG:g} s). "
    "(d) Separation to the critical intruder (solid) and to the nearest "
    f"aircraft of any kind (dotted), 20 Hz, {tw(*T_STRIP)} s; the vertical line "
    f"marks the instant of Figure 9.{em_note_en}{other_en} Selection rule, fixed before "
    "any replay: an episode is a candidate if arm A came within 45 m of an "
    "intruder whose encounter was not unavoidable and arm C never came within "
    "152.4 m of that intruder; arms A and C of every candidate were replayed "
    "through the frozen code, and the replay had to reproduce every "
    "deterministic field of the batch record; the intruder's path in C had to "
    "stay within 30 m of its path in A at every 2 Hz frame up to A's closest "
    f"approach.{ext_en} "
    + ("" if EXTENDED else
       f"In the {n_eps_first} episodes of this condition this gives "
       f"{n_pool_first} candidates and {n_surv} survivors. ")
    + "Of the survivors, the episode with the median C-arm closest approach "
    f"was taken (lower median, rank {rank} of {n_surv}). The traffic is "
    "matched on initial conditions only; "
    "it reacts to the ownship and draws its decisions from shared random "
    f"streams. {tail_en}")
kr = (
    "그림 11. 사전 등록 규칙으로 고른 폐루프 대응 사례이다. "
    f"{COND_KR}의 에피소드 {IDX}이며, 비행체 {N_AG}기, 90초이다. 같은 시나리오 "
    "시드와 초기 상태에서 세 번 비행하였다. 플래너는 같고 위험 입력만 표 11의 세 "
    "가지로 바꾸었다. (a) A: 정적 관측, (b) B: CV, (c) C: PRISM이다. "
    f"{T_MAP[0]:.0f}~{T_MAP[1]:.0f}초를 같은 축의 세계 좌표(m)로 그렸다. 검은 "
    f"선은 자기 비행체이다. 주황 선은 핵심 침입기({kr_crit}, {int_kr} 의도)이며 "
    "각 군의 폐루프에서 실제로 날아간 경로이다. 점은 10초 간격이다. 회색은 다른 "
    "교통이다. 연결선은 핵심 침입기와의 최근접이며 지상 진실로 20 Hz에서 "
    f"계산하였다. A {cc['none']['sep_m']:.1f} m({ts(cc['none']['t_s'])}초), "
    f"B {cc['cv']['sep_m']:.1f} m({ts(cc['cv']['t_s'])}초), "
    f"C {cc['model']['sep_m']:.1f} m({ts(cc['model']['t_s'])}초)이다. 빨간 X는 "
    "NMAC 반경(152.4 m) 안의 최근접이다. (c)의 빈 파란 점은 그림 9 시점"
    f"({T_FIG:g}초)의 자기 비행체이다. (d) 핵심 침입기(실선)와 가장 가까운 "
    f"비행체(점선)까지의 이격이며 20 Hz, {T_STRIP[0]:.0f}~{T_STRIP[1]:.0f}초이다. "
    f"세로선은 그림 9의 시점이다.{em_note_kr}{other_kr} 선택 규칙은 재생 전에 고정하였다. "
    "A군이 회피 불가능하지 않은 침입기의 45 m 안에 들어가고 C군은 그 침입기의 "
    "152.4 m 안에 들어간 적이 없는 에피소드가 후보이다. 모든 후보의 A군과 C군을 "
    "동결 코드로 재생하고, 재생이 배치 기록의 결정적 필드를 전부 재현해야 한다. "
    "침입기의 C군 경로가 A군 최근접까지 모든 2 Hz 프레임에서 A군 경로의 30 m "
    f"안에 있어야 한다.{ext_kr} "
    + ("" if EXTENDED else
       f"이 조건의 {n_eps_first}개 에피소드에서 후보는 {n_pool_first}개, 생존자는 "
       f"{n_surv}개이다. ")
    + f"생존자 가운데 C군 최근접 거리가 중앙값인 에피소드를 취한다(낮은 중앙값, "
    f"{n_surv}개 중 {rank}번째). 교통은 초기 조건만 일치한다. 교통은 자기 비행체에 반응하고 "
    f"공유 난수열에서 결정을 추출한다. {tail_kr}")

gate = META["gate"]
KEYS = ("seed", "n_agents", "min_sep", "min_sep_20hz", "sep_p5",
        "nmac_events", "union20_events_152", "union20_events_76",
        "union20_events_45", "path_len_m", "abs_omega_p95", "n_cycles",
        "emergency_cycles", "pred_logged")


def gate_lines():
    out = [f"{'field':20s} " + " | ".join(f"{LABEL[a]:>24s}" for a in ARMS)]
    for k in KEYS:
        cells = []
        for a in ARMS:
            kf = gate["key_fields"][a][k]
            cells.append(f"{kf['batch']} / {kf['replay']}"
                         + ("" if kf["equal"] else " MISMATCH"))
        out.append(f"{k:20s} " + " | ".join(f"{c:>24s}" for c in cells))
    return out


def survivor_lines():
    out = [f"{'cond':7s} {'idx':>5s} {'slot':>4s} {'class':16s} {'intent':15s} "
           f"{'A min':>8s} {'B min':>10s} {'C min':>8s} {'t_min':>6s} "
           f"{'own@t_min':>9s} {'crit dev':>8s} {'1st>1m':>7s}"]
    for r in SURV:
        b = f"{r['b_min_row']} m" if r["b_min_row"] is not None else "> 152.4 m"
        at = r["attributability"]
        mark = (" <- selected" if r["index"] == IDX and r["cond"] == COND
                else "")
        out.append(f"{r['cond']:7s} {r['index']:5d} {r['slot']:4d} "
                   f"{r['type_name']:16s} {r['intent_name']:15s} "
                   f"{r['a_min_rec']:6.1f} m {b:>10s} {r['c_min_rec']:6.1f} m "
                   f"{r['t_min_a']:6.2f} {at['own_div_at_t_min_m']:7.1f} m "
                   f"{at['crit_max_dev_2hz_m']:6.1f} m "
                   f"{str(at['any_agent_first_gt_1m_2hz_s']):>7s}{mark}")
    return out


def dropped_lines():
    out = []
    for cnd in CONDS_USED:
        cj = json.load(open(f"{OUT}/data/candidates_{cnd}.json", encoding="utf-8"))
        for r in cj["dropped"]:
            x = ""
            if r["status"] == "attributability_failed":
                at = r["attributability"]
                x = (f"critical intruder {at['crit_max_dev_2hz_m']} m from its A "
                     f"path at {at['crit_max_dev_t_s']} s; C min {r['c_min_rec']} m")
            elif r["status"] == "gate_failed":
                x = json.dumps({a: r["gate"][a]["mismatches"][:3]
                                for a in r["gate"]})[:200]
            out.append(f"  {cnd:7s} ep{r['index']:04d} slot {r['slot']:2d} "
                       f"{r['type_name']:16s} {r['status']:22s} {x}")
    return out


sc = META["record_check"]["sep_check_20hz"]
notes = [
    "--- notes (not part of the caption) ---",
    "PRE-REGISTERED RULE (verbatim copy of "
    f"{RULE_FILE}; original text sha256 {SHA_ORIG}, written before any replay"
    + (f"; the file now also carries Amendment A1 to Step 5, declared post hoc, "
       f"file sha256 {RULE_META['sha256_amended']}" if AMENDED else "") + "):",
    *["  | " + ln for ln in RULE_TXT.rstrip("\n").split("\n")],
    "selection result: conditions in the pre-registered order "
    f"{CONDS_USED} (extended beyond ood40: {EXTENDED}); episodes "
    f"{ {c: SELN['n_episodes'][c] for c in CONDS_USED} }; pool sizes "
    f"{ {c: SELN['pool_sizes'][c] for c in CONDS_USED} } ({n_pool_total} candidates, "
    f"{2 * n_pool_total} replays of A and C, gate result per candidate in the "
    "candidates_<cond>.json files); survivors by condition "
    f"{SELN['n_survivors_by_cond']} ({n_surv} in all); dropped by condition and "
    f"reason {SELN['dropped_by_cond_status']}; selected {COND} ep{IDX} = rank "
    f"{rank} of {n_surv} by the C-arm closest approach (lower median); selection "
    f"file {SELN_PATH} (sha256 {sha256(SELN_PATH)}).",
    "SURVIVOR TABLE (all conditions used, sorted by C min; A/C from the recorded "
    "20 Hz GT, B from the B batch row; t_min = A's closest approach; own@t_min = "
    "ownship distance A-C at t_min; crit dev = max 2 Hz distance of the critical "
    "intruder between A and C up to t_min; 1st>1m = first 2 Hz frame with any "
    "agent > 1 m apart between A and C):",
    *["  " + ln for ln in survivor_lines()],
    "DROPPED CANDIDATES:",
    *dropped_lines(),
    f"data: {DATA} (sha256 {sha256(DATA)}); {SCENE} (sha256 {sha256(SCENE)}); "
    f"gate file {GATE} (sha256 {sha256(GATE)}); candidates {CAND_PATH} (sha256 "
    f"{sha256(CAND_PATH)}); written by Paper/figures/select_case_r3.py (sha256 "
    f"{prov['script_sha256']}).",
    "batch rows (read only): " + "; ".join(
        f"{a}: {r['file']} line {r['line_no']} sha256 {r['line_sha256']}"
        for a, r in prov["batch_rows"].items()),
    f"batch command: {prov['batch_command']}",
    f"model: {prov['model_local']} (sha256 {prov['model_sha256']}; the rows "
    f"name the runner copy {prov['model_runner_in_rows']}, same file); "
    "track rule current, K = 3, seed 1.",
    f"code: frozen snapshot {prov['snapshot_dir']}; batch.code_fingerprint "
    f"{prov['code_fp_runner_layout']['fp']} in the runner layout (equal to the "
    f"code_fp of the rows: {prov['runner_fp_matches_rows']}), "
    f"{prov['code_fp_local_layout']['fp']} in the local layout used for the "
    "replay; the two layouts differ only by 'D:/PRISM' -> 'C:/PRISM' in "
    f"{prov['layout_equivalence']['path_rewrite_only']} files (all others "
    "identical). Imports resolved to the snapshot only (module check in the "
    "gate file); no snapshot source file was edited.",
    "gate (hook-free replay): replay row == batch row for every deterministic "
    f"field ({', '.join(str(gate['n_deterministic_fields'][a]) for a in ARMS)} "
    "fields for A/B/C, incl. the per-cycle list, pairwise segments and union "
    "event lists; wall-clock timing fields excluded; model path and code_fp "
    "differ by layout only). Batch (RTX 4090 runner) / replay (RTX 3080):",
    *["  " + ln for ln in gate_lines()],
    "recording run: rows equal the batch rows again (all deterministic "
    "fields); recorded 20 Hz minima: " + "; ".join(
        f"{LABEL[a]} {sc[a]['min_sep_20hz_recorded']} m (row "
        f"{sc[a]['min_sep_20hz_row']})" for a in ARMS) + ".",
    f"critical intruder: slot {CRIT} ({F['critical_type']}, intent "
    f"{F['critical_intent']}); arm A segment {ev['t_entry_a']}-{ev['t_exit_a']} s, "
    f"track id {ev['track_id_a']}, lead from track start {ev['lead_track_s_a']} s; "
    "time below 152.4 m (20 Hz): "
    + ", ".join(f"{LABEL[a]} {tb[a]} s" for a in ARMS) + ".",
    "episode minimum per arm (20 Hz, any aircraft): " + "; ".join(
        f"{LABEL[a]} {em[a]['sep_m']} m at {em[a]['t_s']:.2f} s "
        f"({em[a]['type']}, slot {em[a]['slot']})" for a in ARMS) + ".",
    "other aircraft within 152.4 m (20 Hz GT, per arm; slot: type, min at t): "
    + "; ".join(f"{LABEL[a]}: " + (", ".join(
        f"slot {s} {TYPE_NAMES[v[2]]} {v[0]} m at {v[1]} s"
        for s, v in sorted(OTHER[a].items())) or "none") for a in ARMS)
    + f". Arm A has {N_QUAL_A} pairwise segment(s) below 45 m not flagged "
    "unavoidable_new; the rule takes the smallest as the critical intruder.",
    "attributability (C vs A): critical intruder max 2 Hz deviation "
    f"{ATT_C['crit_max_dev_2hz_m']} m at {ATT_C['crit_max_dev_t_s']} s over "
    f"{ATT_C['n_frames_checked']} frames up to t_min (alive in both arms in all "
    f"of them); first 20 Hz tick with the critical intruder > 1 m from A: "
    f"{ATT_C['crit_first_gt_1m_20hz_s']} s; first 2 Hz frame with any agent > 1 m "
    f"from A: {ATT_C['any_agent_first_gt_1m_2hz_s']} s (slots "
    f"{ATT_C['any_agent_first_gt_1m_slots']}); ownship C-A distance at t_min "
    f"{ATT_C['own_div_at_t_min_m']} m, max before t_min "
    f"{ATT_C['own_div_max_before_t_min_m']} m, first > 1 m at "
    f"{ATT_C['own_first_gt_1m_20hz_s']} s. (B vs A, information): critical "
    f"intruder max deviation {ATT_B['crit_max_dev_2hz_m']} m; ownship B-A at "
    f"t_min {ATT_B['own_div_at_t_min_m']} m.",
    "pairing: first different selected plan vs A at "
    + ", ".join(f"{LABEL[a]} {pdiff[a]} s" for a in ("cv", "model"))
    + "; critical intruder first differs from A by > 0.01 / 1 / 30 m at "
    + "; ".join(f"{LABEL[a]} {fd[a]['gt_0.01m']} / {fd[a]['gt_1m']} / "
                f"{fd[a]['gt_30m']} s" for a in ("cv", "model"))
    + "; first 2 Hz frame with any aircraft > 1 m from A: "
    + "; ".join(f"{LABEL[a]} {td[a]['t_s']} s (slots {td[a]['slots']}, ownship "
                f"offset {td[a]['own_offset_m']} m)" for a in ("cv", "model"))
    + ". Mechanism (dataGen, unchanged): the reaction layer draws from the "
    "shared 'perception' and 'branch_choice' streams only when a threat, the "
    "ownship included, is inside its trigger geometry, so a small ownship "
    "difference can shift those streams for every later agent.",
    "ownship offset from arm A up to the A-arm closest approach: "
    + "; ".join(f"{LABEL[a]} max {od[a]['max_m']} m (at {ts(od[a]['t_of_max'])} s), "
                f"{od[a]['at_t_min_m']} m at t_min" for a in ("cv", "model")) + ".",
    (f"Figure 9 instant: t = {T_FIG:g} s by Amendment A1 ('{A1['why']}', fallback "
     f"{A1['fallback']}); non-emergency cycles {A1['n_non_emergency']} of "
     f"{A1['n_cycles']} in [{A1['table_range_s'][0]:g}, {A1['table_range_s'][1]:g}) s; "
     f"cycles where the bird's forecast enters a non-emergency decision: "
     f"{A1['n_non_emergency_cycles_where_bird_enters']}; bird drove a non-emergency "
     f"decision before A's closest approach: "
     f"{A1['bird_drove_a_non_emergency_decision']}; at the instant "
     f"n_infeasible_by_bird {A1['row_at_instant']['n_infeasible_by_bird']}, "
     f"bird_share {A1['row_at_instant']['bird_share']:.3f}, selected target_d "
     f"{A1['row_at_instant']['sel_target_d']:.1f} m; the original Step 5 instant "
     f"was {A1['step5_original']['chosen_t']} s ('{A1['step5_original']['why']}', "
     "an emergency cycle) and its scene is kept as "
     f"data/scene_{IDX}_step5_original.json; per-cycle table in {CYC_PATH}."
     if A1 is not None else
     f"Figure 9 instant: t = {T_FIG:g} s, rule result "
     f"'{S9['fig9_rule']['why']}' (first_rule_t {S9['fig9_rule']['first_rule_t']}, "
     f"chosen |delta target_d| {S9['fig9_rule']['chosen_d_change']:.1f} m)."),
    ("emergency cycles of arm C before A's closest approach (from the A1 table), "
     "as contiguous runs with the dominant conflicting slot: "
     + "; ".join(
         f"{g['t_start']:g}-{g['t_end']:g} s ({g['n_cycles']} cycles, dominant "
         f"slot {g['dominant_slot']}"
         + (f" = {TYPE_NAMES[A['model']['gt_type'][g['dominant_slot']]]}"
            if g["dominant_slot"] is not None else "")
         + f", bird disks among the conflicts: {g['bird_in_conflicts']})"
         for g in FACTS["emergency_runs_C"])
     + ". Per-cycle hard-check failures by slot: "
     + ", ".join(f"{r['t']:g} s {r.get('infeasible_by_slot', {})}"
                 for r in (CYCLES["table"] if CYCLES else []) if r["emergency"])
     + "." if CYCLES else "A1 table not available."),
    f"view: world metres, equal aspect; x {XL[0]:.0f}..{XL[1]:.0f}, y "
    f"{YL[0]:.0f}..{YL[1]:.0f}; window {tw(*T_MAP)} s = [t_min - 20, t_min + 10] s "
    "rounded outward to 5 s and clipped to the episode; holds every arm's "
    "ownship and critical-intruder track over the window and both endpoints of "
    f"every closest-approach connector; strip {tw(*T_STRIP)} s, y up to "
    f"{Y_TOP:.0f} m.",
    "supersedes: episode 244 of r3_ood100 (files fig_case_avoid_r3_ep244*, "
    "caption_case_avoid_r3_ep244.txt, data case_ep244_r3.json), in which the "
    "critical intruder itself flew a different path in C (not attributable to "
    "the ownship), and the submitted Figure 11 (pre-review runs and model).",
]
cap_path = f"{OUT}/caption_case_avoid_r3.txt"
with open(cap_path, "w", encoding="utf-8") as f:
    f.write("[EN]\n" + en + "\n\n[KR]\n" + kr + "\n\n" + "\n".join(notes) + "\n")
files.append(cap_path)

# ----------------------------------------------------------------- manifest
man = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST)\
    else {}
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
for key in ("fig_case_avoid_r3", "provenance", "gate"):
    old = man.get(key)
    if old is None or key + "_ep244" in man:
        continue
    txt = json.dumps(old)
    if "ep244" in txt or "244" in str(old.get("description", "")) or (
            key == "gate" and old.get("episode_index") is None):
        man[key + "_ep244"] = man.pop(key)
# the ep244 gate summary is kept from its own gate file (the earlier manifest
# entry carried no episode tag)
GATE_244 = f"{OUT}/data/gate_ep244_r3.json"
if "gate_ep244" not in man and os.path.exists(GATE_244):
    g244 = json.load(open(GATE_244, encoding="utf-8"))
    man["gate_ep244"] = {
        "episode_index": g244["episode_index"], "condition": "r3_ood100",
        "pass": g244["pass"],
        "key_fields": {a: g244["table"][a]["key_fields"] for a in ARMS},
        "n_deterministic_fields": {a: g244["table"][a]["n_deterministic_fields"]
                                   for a in ARMS},
        "mismatches": {a: g244["table"][a]["deterministic_mismatches"]
                       for a in ARMS},
        "source": {"path": GATE_244, "sha256": sha256(GATE_244)}}
man["provenance"] = {
    "episode_index": IDX, "condition": META["episode"]["condition"],
    "snapshot": prov["snapshot_dir"],
    "code_fp_runner_layout": prov["code_fp_runner_layout"],
    "code_fp_local_layout": prov["code_fp_local_layout"],
    "layout_equivalence": {k: prov["layout_equivalence"][k] for k in
                           ("equivalent", "identical", "path_rewrite_only",
                            "other", "only_local", "only_runner")},
    "model": prov["model_local"], "model_sha256": prov["model_sha256"],
    "batch_rows": prov["batch_rows"], "batch_command": prov["batch_command"],
    "selector": {"path": prov["script"], "sha256": prov["script_sha256"]},
    "rule_file": {"path": RULE_FILE, "sha256_original": SHA_ORIG,
                  "sha256_current": sha256(RULE_FILE), "amended": AMENDED},
    "candidates_file": {"path": CAND_PATH, "sha256": sha256(CAND_PATH)},
    "selection_file": {"path": SELN_PATH, "sha256": sha256(SELN_PATH)},
    "gate_file": {"path": GATE, "sha256": sha256(GATE), "pass": G["pass"]}}
man["gate"] = {"episode_index": IDX, "pass": gate["pass"],
               "key_fields": gate["key_fields"],
               "n_deterministic_fields": gate["n_deterministic_fields"],
               "mismatches": gate["mismatches"]}
man["fig_case_avoid_r3"] = {
    "description": f"Paired closed-loop case selected by the pre-registered "
                   f"rule: episode {IDX} of r3_{COND}, arms A/B/C, 20 Hz "
                   "separations; replaces the ep244 case.",
    "script": {"path": os.path.abspath(__file__).replace("\\", "/"),
               "sha256": sha256(os.path.abspath(__file__))},
    "sources": {"case": {"path": DATA, "sha256": sha256(DATA)},
                "scene (Figure 9 instant only)": {"path": SCENE,
                                                  "sha256": sha256(SCENE)},
                "candidates": {"path": CAND_PATH, "sha256": sha256(CAND_PATH)},
                "selection": {"path": SELN_PATH, "sha256": sha256(SELN_PATH)},
                "rule": {"path": RULE_FILE, "sha256_original": SHA_ORIG,
                         "sha256_current": sha256(RULE_FILE), "amended": AMENDED},
                **({"cycles (Amendment A1)": {"path": CYC_PATH,
                                              "sha256": sha256(CYC_PATH)}}
                   if CYCLES else {})},
    "values": FACTS,
    "fig9_amended": A1,
    "view": {"xlim_m": [round(v, 1) for v in XL],
             "ylim_m": [round(v, 1) for v in YL], "t_map_s": T_MAP,
             "t_strip_s": T_STRIP, "t_marks_s": T_MARKS,
             "strip_ymax_m": Y_TOP},
    "labels": LABEL,
    "files": files,
    "generated_utc": now}
man["generated_utc"] = now
with open(MANIFEST + ".tmp", "w", encoding="utf-8") as f:
    json.dump(man, f, indent=1, ensure_ascii=False, default=float)
os.replace(MANIFEST + ".tmp", MANIFEST)
for p in files:
    print("[saved]", p)
print("[saved]", MANIFEST)
