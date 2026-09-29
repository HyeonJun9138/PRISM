# -*- coding: utf-8 -*-
"""Figures for the Reviewer-3 (R3) revision round of the PRISM paper.

    python make_r3_figs.py                                   # all twelve figures
    python make_r3_figs.py --only planner_forest,rollout     # a subset
    python make_r3_figs.py --list                            # inputs present / missing

Every plotted number, sample size and interval bound is read at run time from
the JSON / JSONL / NPZ sources below; nothing is typed in from memory. A panel
(or a row, or a seed) whose input has not landed yet is drawn as a neutral
"not available" placeholder (no status text on the figure itself); the missing
file is recorded in the manifest's pending list, and the script still exits 0,
so it can simply be rerun when the inputs arrive. An unexpected exception
inside a figure (for example a changed schema) is printed with its traceback,
recorded in the manifest, and makes the exit code 1 after the other figures
are built.

Presentation rules (revision round 1, author's instruction): panel titles of
at most six words; no model tags, condition keys, file names or settings lists
on the drawn figure (they live in the caption and the manifest); arms are
named "A: static observation", "B: CV", "C: PRISM"; one legend per figure,
outside the panels; log-ratio axes state their direction; every figure is
17 cm wide (full width) with no text below 7 pt at print size.

Sources (schema of the planner aggregates: modelVal/experiments/aggregate_r3.py)
  Paper/rebuttal/r3/planner/core_{sparse,ood40,ood100,id40,id24}.json
      core_ood8.json is never read (duplicate of ood40, DECISIONS D6.13)
  Paper/rebuttal/r3/planner/shared40_300.json, sens300_*.json, *iso*.json
      (paired indices 0-299; the older sens_* / cal40 / decomp40 / shared40
      files are never read)
  Paper/rebuttal/r3/recal_r3_k3_s1.json
  Paper/rebuttal/r3/rollout_r3_k3_s1_{idheldout_T1,oodheldout_T1,oodheldout_T1695}.json
  Paper/rebuttal/r3/rollout_cond_*_{aligned,legacy}.json      (submitted model)
  Paper/rebuttal/r3/pred_p5_p6_p8*.json   (newest by mtime; only P8 is plotted)
  Paper/rebuttal/param_sensitivity_results.json   (case descriptions, caption)
  Logs/Analysis/eval_r3/<tag>_<set>_report.json
  Logs/Analysis/eval_r3/r3_k3_s<seed>_{testid,testood}_persample.npz   (per class)
  Logs/Analysis/eval_r3/persample_<method>_<set>_revised_current_idxcurrent.npz
  Logs/Models/baselines_r3/report_<method>_<set>_revised_current_idxcurrent.json
  Logs/Models/<job>_<tag>/{metrics.jsonl, job.json, config.json}  (completed jobs)
  Logs/Planner/r3_<cond>/<arm>*.jsonl   agent count / density of a condition,
      taken from the rows the aggregate was built from (its inputs.paths)
  modelVal/experiments/aggregate_r3.py  event thresholds and t_need (imported)

Outputs (only files whose name starts with r3fig_ are written)
  Paper/figures_r3/r3fig_<name>.png (300 dpi) and .pdf
  Paper/figures_r3/r3fig_manifest.json   per figure: inputs with sha256, the
      numbers plotted, pending items, caption; merge-updated under --only
  Paper/figures_r3/r3fig_captions.md     caption drafts, rendered from the manifest

Style: rcParams of make_fig_cases.py with the font floor raised to 7 pt; arm
colours of make_rebuttal_figs.py (A static observation #90a4ae, B CV #4fc3f7,
C PRISM #ab47bc) as marker fills, the darker shade of the same hue family
(#455a64, #0288d1, #7b1fa2) for lines and marker edges, and a fixed marker and
line style per arm (A circle / dotted, B square / dashed, C diamond / solid),
so the arms stay distinguishable in grayscale and under colour-vision
deficiency. The ink trio passes the dataviz validator's CVD and normal-vision
separation checks; A is deliberately low-chroma (the no-forecast reference).
Ablation variants and the IMM baseline use Okabe-Ito colours.

Run:  python D:/PRISM/Paper/figures/make_r3_figs.py
"""
from __future__ import annotations

import os

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator,  # noqa: E402
                               MaxNLocator, NullLocator)
from matplotlib.transforms import Bbox  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# the aggregator is imported by path below; never leave a bytecode cache in
# modelVal/ (or anywhere else) as a side effect of drawing figures
sys.dont_write_bytecode = True

ROOT = (os.environ["PRISM_ROOT"] + '')
R3 = f"{ROOT}/Paper/rebuttal/r3"
PL = f"{R3}/planner"
EV = f"{ROOT}/Logs/Analysis/eval_r3"
MODELS = f"{ROOT}/Logs/Models"
BASELINES = f"{MODELS}/baselines_r3"
PARAM_SENS_JSON = f"{ROOT}/Paper/rebuttal/param_sensitivity_results.json"
OUT = f"{ROOT}/Paper/figures_r3"
PREFIX = "r3fig_"
MANIFEST = f"{OUT}/{PREFIX}manifest.json"
CAPTIONS = f"{OUT}/{PREFIX}captions.md"
AGG_PY = f"{ROOT}/modelVal/experiments/aggregate_r3.py"
SCRIPT = os.path.abspath(__file__).replace("\\", "/")
DPI = 300
W_FULL = 17.0 / 2.54          # full-width figure [in]; every figure here is full width
W_COL = 8.5 / 2.54            # single column [in] (kept for reference; unused)
FS_MIN = 7.0                  # font floor at print size [pt]
FS_TXT = 7.0                  # in-panel annotations
FS_TITLE = 7.6                # panel titles

# ------------------------------------------------------------------ style
# make_fig_cases.py rcParams with every size lifted to the 7 pt floor
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_MIN, "ytick.labelsize": FS_MIN,
    "legend.fontsize": FS_MIN,
    "axes.linewidth": 0.5, "axes.edgecolor": "#8c8c8c",
    "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#8c8c8c", "ytick.color": "#8c8c8c",
    "xtick.labelcolor": "#404040", "ytick.labelcolor": "#404040",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
    "grid.linewidth": 0.4, "grid.color": "#e8e8e8", "grid.alpha": 1.0,
    "axes.axisbelow": True, "savefig.dpi": DPI,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ARMS = ("none", "cv", "model")
ARM_FILL = {"none": "#90a4ae", "cv": "#4fc3f7", "model": "#ab47bc"}
ARM_INK = {"none": "#455a64", "cv": "#0288d1", "model": "#7b1fa2"}
ARM_MK = {"none": "o", "cv": "s", "model": "D"}
ARM_LS = {"none": (0, (1.1, 1.5)), "cv": (0, (4.2, 2.0)), "model": "-"}
ARM_NAME = {"none": "A: static observation", "cv": "B: CV", "model": "C: PRISM"}
ARM_SHORT = {"none": "static", "cv": "CV", "model": "PRISM"}
ARM_LET = {"none": "A", "cv": "B", "model": "C"}
IMM_COL, IMM_MK = "#E69F00", "^"          # IMM baseline (Okabe-Ito orange)
GT_COL, GT_MK = "#333333", "X"            # CV on the true state (upper bound)
ARROW = "→"                          # direction marker on ratio axes
INK, MUTE, WARN = "#1a1a1a", "#6f6f6f", "#D55E00"
REF = "#555555"            # reference lines (ratio = 1, nominal level, uniform)
SEP = "#d4d4d4"            # row / group separators
ZEBRA = "#f5f5f5"          # alternate-row band in forest plots
ORACLE = "#333333"         # the submitted (oracle-association) model, s4
# nominal levels 0.5 / 0.7 / 0.9: one-hue ordinal ramp (validator: ordinal pass)
LEVEL_COL = {50: "#ba68c8", 70: "#8e24aa", 90: "#4a148c"}
# ablation variants (Okabe-Ito based; validator: normal-vision pass, CVD in
# the 6-8 band, so every variant also has its own marker and dash pattern)
ABLATIONS = (
    ("r3_k3_s0", "full model (K = 3)", ARM_INK["model"], "-", "D"),
    ("r3_k1_s0", "K = 1", "#E69F00", (0, (4.0, 1.6)), "^"),
    ("r3_noneigh_s0", "no neighbours", "#009E73", (0, (1.2, 1.2)), "v"),
    ("r3_notype_s0", "no type head", "#D55E00", (0, (5.0, 1.5, 1.2, 1.5)), "<"),
    ("r3_mdn_s0", "standard MDN training", "#CC79A7", (0, (2.4, 1.4)), "p"),
    ("r3_pointnet_s0", "Point-MSE (K = 1)", "#333333", (0, (1.0, 1.0, 3.0, 1.0)), "X"),
)
SEEDS = tuple(range(7))             # K = 3 seeds 0..6 (DECISIONS D6.5)
ONE_X_TAG = "r3_1x_s0"


def _planner_seed(default: int = 1) -> int:
    """Seed of the checkpoint the planner's model arm actually loaded, read
    from the core aggregates' run knobs (model_path ..._r3_k3_s<seed>); the
    default (DECISIONS D6.9) is used only if no aggregate is readable."""
    for key in ("ood40", "id40", "sparse", "ood100", "id24"):
        p = f"{ROOT}/Paper/rebuttal/r3/planner/core_{key}.json"
        try:
            with open(p, encoding="utf-8") as f:
                mp = json.load(f)["inputs"]["model"]["knobs"]["model_path"]
            m = re.search(r"_r3_k3_s(\d+)[/\\]", str(mp[0] if isinstance(mp, list)
                                                    else mp))
            if m:
                return int(m.group(1))
        except (OSError, KeyError, IndexError, ValueError, TypeError):
            continue
    print(f"  [warn] planner seed not found in the core aggregates; using {default}")
    return default


PLANNER_SEED = _planner_seed()


# ============================================================ provenance
def norm(p: str) -> str:
    return os.path.abspath(p).replace("\\", "/")


_SHA: dict = {}
_JSON: dict = {}


def file_info(path: str) -> dict:
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA[key] = h.hexdigest()
    return {"sha256": _SHA[key], "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(
                timespec="seconds")}


def read_json(path: str):
    st = os.stat(path)
    key = (path, st.st_mtime_ns)
    if key not in _JSON:
        with open(path, encoding="utf-8-sig") as f:
            _JSON[key] = json.load(f)
    return _JSON[key]


class Ctx:
    """Book-keeping of one figure: inputs read, inputs missing, values."""

    def __init__(self, name: str):
        self.name = name
        self.inputs: dict = {}
        self.missing: list = []
        self.values: dict = {}
        self.pending: list = []
        self.notes: list = []
        self.caption = ""
        self.files: list = []
        self.size_cm = None
        self.size_px = None
        self.error = None

    def json(self, path: str):
        path = norm(path)
        if not os.path.isfile(path):
            if path not in self.missing:
                self.missing.append(path)
            return None
        self.inputs[path] = file_info(path)
        return read_json(path)

    def use(self, path: str) -> bool:
        path = norm(path)
        if os.path.isfile(path):
            self.inputs[path] = file_info(path)
            return True
        if path not in self.missing:
            self.missing.append(path)
        return False

    def pend(self, item: str, missing) -> None:
        self.pending.append({"item": item,
                             "missing": [norm(m) if ("/" in m or "\\" in m)
                                         else m for m in missing]})

    def status(self) -> str:
        """complete: nothing missing; partial: some items PENDING; pending:
        no input found at all (every panel is a placeholder)."""
        if self.error:
            return "error"
        if self.pending:
            return "partial" if self.inputs else "pending"
        return "complete"


def newest(pattern: str):
    c = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    return norm(max(c, key=os.path.getmtime)) if c else None


def _load_thresholds():
    """Event thresholds [m] and t_need [s] from the aggregator module."""
    try:
        spec = importlib.util.spec_from_file_location("_agg_r3", AGG_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {k: float(v) for k, v in mod.THR}, float(mod.T_NEED_S)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [warn] thresholds not importable from {AGG_PY}: {exc}")
        return None, None


THR, T_NEED = _load_thresholds()


def thr_txt(key: str) -> str:
    """'152.4 m' from the aggregator; the key itself if it was not importable."""
    if THR and key in THR:
        return f"{THR[key]:g} m"
    return f"{key} m (key)"


# ============================================================ small helpers
def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_p(p):
    if p is None:
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fmt_n(n):
    return "n/a" if n is None else f"{int(n):,}"


def mk(arm: str, size: float = 4.0, filled: bool = True) -> dict:
    """Marker style of an arm: fill colour, darker edge, fixed shape."""
    s = size * (0.9 if ARM_MK[arm] == "D" else 1.0)
    if not filled:
        return dict(marker=ARM_MK[arm], ms=s, mfc="white", mec=ARM_INK[arm],
                    mew=0.85)
    return dict(marker=ARM_MK[arm], ms=s, mfc=ARM_FILL[arm],
                mec=ARM_INK[arm], mew=0.65)


def ytrans(ax):
    """x in axes fraction, y in data."""
    return mtransforms.blended_transform_factory(ax.transAxes, ax.transData)


def xtrans(ax):
    """x in data, y in axes fraction."""
    return mtransforms.blended_transform_factory(ax.transData, ax.transAxes)


def pending_panel(ax, ctx: Ctx, item: str, missing, note: str = "") -> None:
    """Neutral placeholder for a panel whose input is missing. The figure
    shows no status text and no file name; both go to the manifest."""
    ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center",
            va="center", fontsize=FS_TXT, color=MUTE, style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d0d0d0")
    ctx.pend(item + (f" ({note})" if note else ""), missing)


def pending_row(ax, y: float, text: str) -> None:
    """Neutral placeholder for a missing row (the reason is in the manifest)."""
    ax.text(0.5, y, "not available", transform=ytrans(ax), ha="center",
            va="center", fontsize=FS_TXT, color=MUTE, style="italic",
            zorder=6)


def _plotted_points(ax):
    """Everything drawn in the panel, as axes-fraction points: (data, ref).
    data = lines (densified), error-bar segments, bar tops and direct labels
    in data coordinates; ref = full-width / full-height reference lines."""
    ax.get_xlim()                  # force any pending autoscale first
    ax.get_ylim()
    inv = ax.transAxes.inverted()
    pts, ref = [], []
    fr = np.linspace(0.0, 1.0, 41)

    def add(xy):
        xy = np.asarray(xy, float)
        if xy.ndim == 2 and len(xy):
            xy = xy[np.isfinite(xy).all(1)]
            if len(xy):
                pts.append(inv.transform(ax.transData.transform(xy)))
                return pts[-1]
        return None

    for ln in ax.lines:
        tr = ln.get_transform()
        xy = np.asarray(ln.get_xydata(), float)
        if not len(xy) or not np.isfinite(xy).all():
            continue
        if tr == ax.get_yaxis_transform():                    # axhline
            y = inv.transform(ax.transData.transform(
                [[ax.get_xlim()[0], xy[0, 1]]]))[0, 1]
            ref.append(np.column_stack([fr, np.full_like(fr, y)]))
        elif tr == ax.get_xaxis_transform():                  # axvline
            x = inv.transform(ax.transData.transform(
                [[xy[0, 0], ax.get_ylim()[0]]]))[0, 0]
            ref.append(np.column_stack([np.full_like(fr, x), fr]))
        elif tr == ax.transData:
            P = add(xy)
            if P is not None and len(P) > 1 and ln.get_linestyle() not in (
                    "None", "none", "", " "):
                t = np.linspace(0, 1, 12)[1:-1]
                for a, b in zip(P[:-1], P[1:]):
                    pts.append(a + np.outer(t, b - a))
    for coll in ax.collections:
        if coll.get_transform() != ax.transData or\
                not hasattr(coll, "get_segments"):
            continue
        for s in coll.get_segments():
            add(s)
    for p in ax.patches:
        if p.get_transform() != ax.transData or not hasattr(p, "get_bbox"):
            continue
        b = p.get_bbox()
        add([[b.x0, b.y1], [b.x1, b.y1], [(b.x0 + b.x1) / 2, b.y1]])
    for t in ax.texts:                        # direct labels in data coords
        if t.get_transform() != ax.transData:
            continue
        P = add([t.get_position()])
        if P is not None:
            w = 0.012 * len(t.get_text())
            sgn = -1.0 if t.get_ha() == "right" else 1.0
            pts.append(P + np.array([[sgn * w * f, 0.0] for f in
                                     (0.33, 0.66, 1.0)]))
    stack = (lambda L: np.vstack(L) if L else np.zeros((0, 2)))
    return stack(pts), stack(ref)


def auto_note(ax, text: str, fontsize: float = 5.6, color: str = MUTE,
              prefer=("tr", "tl", "br", "bl"), weight: str = "normal"):
    """Put a short note where it covers the least: every left / right /
    centre position on a vertical grid is scored by the plotted data inside
    its box (weight 1) and reference lines inside it (weight 0.25); ties go
    to the preferred corners, then to positions near the top or bottom edge.
    Call after the limits are final."""
    P, R = _plotted_points(ax)
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    w_pt, h_pt = pos.width * fw * 72, pos.height * fh * 72
    lines = text.split("\n")
    tw = (max(len(s) for s in lines) * fontsize * 0.55 + 6) / w_pt
    th = (len(lines) * fontsize * 1.3 + 4) / h_pt
    m, pad = 0.03, 0.02
    ytop, ybot = 1 - m - th, m

    def count(A, x0, y0):
        if not len(A):
            return 0
        return int(((A[:, 0] >= x0 - pad) & (A[:, 0] <= x0 + tw + pad)
                    & (A[:, 1] >= y0 - pad) & (A[:, 1] <= y0 + th + pad)).sum())

    cands = []
    for side in ("r", "l", "c"):
        x0 = {"r": 1 - m - tw, "l": m, "c": 0.5 - tw / 2}[side]
        for y0 in np.linspace(ybot, max(ybot, ytop), 21):
            corner = ("t" if abs(y0 - ytop) < 1e-9 else
                      "b" if abs(y0 - ybot) < 1e-9 else "m") + side
            rank = prefer.index(corner) if corner in prefer else len(prefer)
            edge = min(abs(y0 - ytop), abs(y0 - ybot))
            cost = count(P, x0, y0) + 0.25 * count(R, x0, y0)
            cands.append(((cost, rank, round(edge, 4)), side, x0, y0))
    _, side, x0, y0 = min(cands, key=lambda c: c[0])
    ha = {"r": "right", "l": "left", "c": "center"}[side]
    x = {"r": x0 + tw, "l": x0, "c": x0 + tw / 2}[side]
    ax.text(x, y0 + th / 2, text, transform=ax.transAxes, fontsize=fontsize,
            color=color, weight=weight, ha=ha, va="center", zorder=7,
            linespacing=1.15, multialignment=ha)
    return side


def pick_ticks(lo: float, hi: float, n_max: int = 5) -> list:
    """Round ratio ticks inside [lo, hi] on a log axis, never crowded."""
    prio = [1.0, 2.0, 0.5, 1.5, 0.75, 3.0, 2.5, 0.8, 1.25, 0.6, 0.9, 4.0,
            0.7, 1.1, 5.0, 0.4, 0.3]
    span = math.log10(hi) - math.log10(lo)
    sep = span / (n_max + 0.5)
    out: list = []
    for t in prio:
        if lo <= t <= hi and all(abs(math.log10(t) - math.log10(c)) >= sep
                                 for c in out):
            out.append(t)
    return sorted(out)


def log_ratio_axis(ax, lo: float, hi: float, n_max: int = 5) -> None:
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    t = pick_ticks(lo, hi, n_max)
    ax.xaxis.set_major_locator(FixedLocator(t))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in t]))
    ax.xaxis.set_minor_locator(NullLocator())


def ratio_limits(vals, pad: float = 0.07):
    vals = [v for v in vals if v is not None and v > 0]
    lo, hi = (min(vals + [1.0]), max(vals + [1.0])) if vals else (0.5, 2.0)
    s = (math.log(hi) - math.log(lo)) * pad
    return math.exp(math.log(lo) - s), math.exp(math.log(hi) + s)


def forest_point(ax, y, est, ci, arm, filled=True, size=4.0, lw=0.9,
                 color=None):
    if est is None:
        return
    col = color or ARM_INK[arm]
    if ci and ci[0] is not None and ci[1] is not None:
        ax.plot([ci[0], ci[1]], [y, y], color=col, lw=lw,
                solid_capstyle="butt", zorder=3)
    ax.plot([est], [y], ls="none", zorder=4, **mk(arm, size, filled))


def event_entry(pp: dict, family: str, key: str) -> dict:
    e = pp[family][key]
    rr = e["rate_ratio"]
    return {"ratio": rr["est"], "ci95": rr["ci"],
            "undefined_frac": rr.get("undefined_frac"),
            "rate_a_per_h": e["rate_a"], "rate_b_per_h": e["rate_b"],
            "total_a": e["total_a"], "total_b": e["total_b"],
            "wins": e["wins"], "losses": e["losses"], "ties": e["ties"],
            "sign_p": e["sign_p"], "n_paired": pp.get("n"),
            "indices": pp.get("indices")}


def traffic_meta(ctx: Ctx, agg: dict, label: str | None = None) -> dict:
    """Agent count, density and preset of an aggregate's traffic, read from
    the planner rows it was built from (first path of one of its inputs)."""
    out = {"n_agents": None, "density_per_km2": None, "preset": None,
           "sim_s": None, "rows": None}
    try:
        lab = label or next(iter(agg["inputs"]))
        p = norm(agg["inputs"][lab]["paths"][0])
    except (KeyError, StopIteration, IndexError):
        return out
    if not ctx.use(p):
        ctx.notes.append(f"planner rows not found (agent count unknown): {p}")
        return out
    na, dens, pre, sim = set(), set(), set(), set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            na.add(r.get("n_agents"))
            pre.add(r.get("preset"))
            sim.add(r.get("sim_s"))
            dens.add((r.get("overrides") or {}).get("density_per_km2"))
    out.update(n_agents=sorted(x for x in na if x is not None),
               density_per_km2=sorted(x for x in dens if x is not None),
               preset=sorted(x for x in pre if x),
               sim_s=sorted(x for x in sim if x is not None), rows=p)
    return out


def one(v):
    """The single value of a distinct-value list, else the list."""
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v


def job_name(model_path) -> str | None:
    if not model_path:
        return None
    parts = str(model_path).replace("\\", "/").split("/")
    return parts[-3] if len(parts) >= 3 else str(model_path)


def save(fig, ctx: Ctx, width_cm: float = 17.0) -> None:
    """Write png + pdf at exactly `width_cm`: the tight bounding box of
    everything drawn (legends and outside labels included) is padded
    symmetrically to the target width, so nothing is clipped and the print
    width is known. Content wider than the target is saved as is and noted
    in the manifest (a layout bug to fix, never silent)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())        # inches
    pad = 0.02
    x0, x1, y0, y1 = bb.x0 - pad, bb.x1 + pad, bb.y0 - pad, bb.y1 + pad
    target = width_cm / 2.54
    if x1 - x0 > target + 0.01:
        ctx.notes.append(f"content is {(x1 - x0) * 2.54:.2f} cm wide, "
                         f"more than the {width_cm:g} cm target")
    else:
        extra = (target - (x1 - x0)) / 2
        x0, x1 = x0 - extra, x1 + extra
    box = Bbox([[x0, y0], [x1, y1]])
    ctx.files = []
    for ext in ("png", "pdf"):
        p = f"{OUT}/{ctx.name}.{ext}"
        assert os.path.basename(p).startswith(PREFIX)
        fig.savefig(p, dpi=DPI, bbox_inches=box, pad_inches=0,
                    facecolor="white")
        ctx.files.append(p)
    plt.close(fig)
    ctx.size_cm = [round(float(x1 - x0) * 2.54, 2), round(float(y1 - y0) * 2.54, 2)]
    try:
        h_px, w_px = matplotlib.image.imread(ctx.files[0]).shape[:2]
        ctx.size_px = [int(w_px), int(h_px)]
    except Exception:                                         # noqa: BLE001
        ctx.size_px = None


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=3.2, fontsize=FS_TITLE)


# ======================================================================
# 1. planner forest
# ======================================================================
CORE = (("sparse", "sparse"), ("ood40", "nominal load"), ("ood100", "saturated"),
        ("id40", "nominal"), ("id24", "matched load"))
EVENT_KEYS = (("union20_152_new", "152"), ("union20_76_new", "76"),
              ("union20_severe_new", "45"))
TB_KEY = "union20_tb_45"


def cond_label(key: str, desc: str, preset) -> str:
    tag = {"test_ood": "OOD", "test_id": "ID"}.get(preset, str(preset or ""))
    return f"{tag} {desc}".strip()


def gfmt(v) -> str:
    """Compact number (8.0 -> '8'); lists of one value are unwrapped."""
    v = one(v)
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return "n/a" if v in (None, []) else str(v)


def fig_planner_forest(ctx: Ctx) -> None:
    rows = [(key, desc, ctx.json(f"{PL}/core_{key}.json"))
            for key, desc in CORE]
    for key, desc, d in rows:
        if d is None:
            ctx.pend(f"condition {key} (all panels)", [f"{PL}/core_{key}.json"])
    V: dict = {}
    for key, desc, d in rows:
        if d is None:
            continue
        A, P = d["per_arm"], d["paired"]
        kn = d["inputs"]["model"]["knobs"]
        v = {"file": norm(f"{PL}/core_{key}.json"),
             "preset": one(kn.get("preset")),
             "hard_geom": one(kn.get("hard_geom")),
             "model_path": one(kn.get("model_path")),
             "episodes": {a: A[a]["episodes"] for a in ARMS},
             "exposure_hours": {a: A[a]["exposure_hours"] for a in ARMS},
             "bootstrap_B": d.get("B"), "bootstrap_seed": d.get("seed"),
             "t_need_s": d.get("t_need_s"),
             "traffic": traffic_meta(ctx, d, "none")}
        for ek, _ in EVENT_KEYS:
            v[ek] = {pr: event_entry(P[pr], "events", ek)
                     for pr in ("cv|model", "none|model")}
            v[ek]["per_arm_rate_per_h"] = {a: A[a]["events"][ek]["per_hour"]
                                           for a in ARMS}
            v[ek]["per_arm_total"] = {a: A[a]["events"][ek]["total"]
                                      for a in ARMS}
        v[TB_KEY] = {
            "per_arm": {a: {"per_hour_s": A[a]["time_below"][TB_KEY]["per_hour_s"],
                            "total_s": A[a]["time_below"][TB_KEY]["total_s"]}
                        for a in ARMS},
            "sign_test": {pr: {**{k: P[pr]["time_below"][TB_KEY][k]
                                  for k in ("wins", "losses", "ties", "sign_p")},
                               "n": P[pr]["n"]}
                          for pr in ("cv|model", "none|model", "none|cv")}}
        if len(set(v["exposure_hours"].values())) != 1:
            ctx.notes.append(f"{key}: exposure hours differ between arms "
                             f"{v['exposure_hours']}")
        V[key] = v
    ctx.values["conditions"] = V
    ctx.values["thresholds_m"] = THR
    ctx.values["t_need_s_aggregator"] = T_NEED

    n = len(rows)
    fig = plt.figure(figsize=(W_FULL, 3.15))
    gs = fig.add_gridspec(1, 4, left=0.16, right=0.99, top=0.83,
                          bottom=0.155, wspace=0.14)
    axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
    allr = [x for v in V.values() for ek, _ in EVENT_KEYS
            for pr in ("cv|model", "none|model")
            for x in [v[ek][pr]["ratio"]] + list(v[ek][pr]["ci95"] or [])]
    lo, hi = ratio_limits(allr)
    titles = [f"(a) NMAC radius, < {thr_txt('152')}",
              f"(b) < {thr_txt('76')}",
              f"(c) severe, < {thr_txt('45')}",
              f"(d) time below {thr_txt('45')}"]
    presets = [V[k]["preset"] if k in V else None for k, _, _ in rows]

    def frame(ax):
        for i in range(n):
            if i % 2 == 1:
                ax.axhspan(i - 0.5, i + 0.5, color=ZEBRA, lw=0, zorder=0)
        for i in range(1, n):
            if presets[i] and presets[i - 1] and presets[i] != presets[i - 1]:
                ax.axhline(i - 0.5, color="#9e9e9e", lw=0.6, zorder=1)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])

    for j, (ek, _) in enumerate(EVENT_KEYS):
        ax = axes[j]
        frame(ax)
        ax.axvline(1.0, color=REF, lw=0.7, zorder=1)
        for i, (key, desc, d) in enumerate(rows):
            if d is None:
                pending_row(ax, i, f"core_{key}.json")
                continue
            e1, e2 = V[key][ek]["cv|model"], V[key][ek]["none|model"]
            forest_point(ax, i - 0.17, e1["ratio"], e1["ci95"], "cv", True,
                         4.3, 1.15)
            forest_point(ax, i + 0.17, e2["ratio"], e2["ci95"], "none",
                         True, 4.1, 1.0)
        log_ratio_axis(ax, lo, hi, n_max=5)
        ax.grid(True, axis="x")
        panel_title(ax, titles[j])
    # one shared axis label under (a)-(c): the ratio and its direction
    x_abc = 0.5 * (axes[0].get_position().x0 + axes[2].get_position().x1)
    fig.text(x_abc, 0.035, "event-rate ratio, B / C and A / C (log scale)"
             f"     {ARROW} fewer events with PRISM",
             ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"],
             color=INK)
    for i, (key, desc, d) in enumerate(rows):
        pr = presets[i] if d is not None else None
        axes[0].text(-0.05, i, cond_label(key, desc, pr) if d is not None
                     else cond_label(key, desc, None),
                     transform=ytrans(axes[0]), ha="right", va="center",
                     fontsize=plt.rcParams["axes.labelsize"],
                     color=INK if d is not None else MUTE)

    # (d) time below the severe threshold, per arm
    ax = axes[3]
    frame(ax)
    tmax = max([V[k][TB_KEY]["per_arm"][a]["per_hour_s"] for k in V
                for a in ARMS] + [1e-9])
    offs = (("cv", -0.22), ("model", 0.0), ("none", 0.22))
    for i, (key, desc, d) in enumerate(rows):
        if d is None:
            pending_row(ax, i, f"core_{key}.json")
            continue
        for arm, off in offs:
            x = V[key][TB_KEY]["per_arm"][arm]["per_hour_s"]
            ax.plot([0, x], [i + off, i + off], color=ARM_INK[arm], lw=0.55,
                    alpha=0.55, solid_capstyle="butt", zorder=2)
            ax.plot([x], [i + off], ls="none", zorder=4, **mk(arm, 3.9, True))
    ax.set_xlim(0, tmax * 1.1)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, axis="x")
    panel_title(ax, titles[3])
    ax.set_xlabel("s per exposure hour", labelpad=1.5)

    # one legend: the arm each marker stands for. In (a)-(c) the marker of
    # arm B or A sits at that arm's ratio to C (bar: 95 % CI); (d) shows all
    # three arms.
    hl = [Line2D([], [], color=ARM_INK["none"], lw=1.0, **mk("none", 4.1, True)),
          Line2D([], [], color=ARM_INK["cv"], lw=1.15, **mk("cv", 4.3, True)),
          Line2D([], [], ls="none", **mk("model", 4.1, True))]
    fig.legend(hl, [ARM_NAME["none"] + "   (A / C in a–c)",
                    ARM_NAME["cv"] + "   (B / C in a–c)", ARM_NAME["model"]],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.56, 0.905), columnspacing=2.2,
               handlelength=2.0, handletextpad=0.5, borderaxespad=0.0)
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    if not V:
        ctx.caption = ("PENDING: none of the core planner aggregates "
                       "(planner/core_*.json) was found.")
        return
    any_v = next(iter(V.values()))
    mp = job_name(any_v["model_path"])
    parts = []
    for key, desc, d in rows:
        if key not in V:
            parts.append(f"{key}: PENDING")
            continue
        v = V[key]
        tr = v["traffic"]
        parts.append(
            f"{cond_label(key, desc, v['preset'])} ({key}): preset "
            f"{v['preset']}, {gfmt(tr['n_agents'])} agents, density "
            f"{gfmt(tr['density_per_km2'])} per km², "
            f"{v['episodes']['model']} paired episodes, "
            f"{v['exposure_hours']['model']:.2f} h of exposure per arm")
    tests = []
    for key, desc, d in rows:
        if key not in V:
            continue
        st = V[key][TB_KEY]["sign_test"]
        seg = "; ".join(
            f"{lab} {st[pr]['wins']}/{st[pr]['losses']}/{st[pr]['ties']}, "
            f"{fmt_p(st[pr]['sign_p'])}"
            for pr, lab in (("cv|model", "B vs C"), ("none|model", "A vs C"),
                            ("none|cv", "A vs B")))
        tests.append(f"{key}: {seg}")
    tn = any_v["t_need_s"]
    ctx.caption = (
        "Closed-loop planner outcomes in five traffic conditions (rows). The "
        "planner is identical in all arms; only the risk input differs: A = "
        "static observation (the observed positions as static obstacles), "
        "B = constant-velocity (CV) extrapolation, C = PRISM "
        f"(checkpoint {mp}). (a)–(c) Ratio of per-hour event rates, B/C "
        "(squares) and A/C (circles), on a log axis; a ratio "
        "above 1 means fewer events in arm C. Events are union events over "
        "all intruders at 20 Hz whose minimum separation falls below "
        f"{thr_txt('152')} (a) or {thr_txt('76')} (b); (c) counts the "
        f"{thr_txt('152')} events whose minimum is below {thr_txt('45')}. "
        "Events whose lead time from current-track detection to threshold "
        f"entry is shorter than t_need = {fmt(tn, 2)} s, or unknown, are "
        "excluded as unavoidable (current-track screen). Bars: 95 % "
        "confidence intervals from a paired episode bootstrap "
        f"(B = {fmt_n(any_v['bootstrap_B'])} resamples of the paired "
        f"episodes, seed {any_v['bootstrap_seed']}). (d) Time during which "
        f"the minimum separation to any intruder is below {thr_txt('45')} "
        "(20 Hz, no avoidability screen), seconds per exposure hour, per "
        "arm. Conditions: " + "; ".join(parts) + ". Exact two-sided "
        "sign tests on the per-episode time below "
        f"{thr_txt('45')} (episodes with less time for the first arm / for "
        "the second arm / ties): " + "; ".join(tests) + ".")


# ======================================================================
# 2. shared hard geometry
# ======================================================================
SHARED_ROWS = (
    ("hdr", "vs PRISM, arm geometry"),
    ("cv_arm|model_arm", "B / C", "cv", True),
    ("none_arm|model_arm", "A / C", "none", True),
    ("hdr", "vs PRISM, shared geometry"),
    ("cv_sh|model_sh", "B / C", "cv", False),
    ("none_sh|model_sh", "A / C", "none", False),
    ("hdr", "same arm: arm / shared"),
    ("cv_arm|cv_sh", "B", "cv", True),
    ("model_arm|model_sh", "C", "model", True),
    ("none_arm|none_sh", "A", "none", True),
)
SH_KEYS = (("union20_152_new", "152"), ("union20_severe_new", "45"))


def fig_shared_geometry(ctx: Ctx) -> None:
    path = f"{PL}/shared40_300.json"
    d = ctx.json(path)
    fig = plt.figure(figsize=(W_FULL, 4.7))
    gt = fig.add_gridspec(1, 2, left=0.08, right=0.99, top=0.865,
                          bottom=0.605, wspace=0.22)
    gb = fig.add_gridspec(1, 2, left=0.235, right=0.99, top=0.455,
                          bottom=0.11, wspace=0.1)
    ax_a, ax_b = fig.add_subplot(gt[0, 0]), fig.add_subplot(gt[0, 1])
    ax_c, ax_d = fig.add_subplot(gb[0, 0]), fig.add_subplot(gb[0, 1])
    if d is None:
        for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
            pending_panel(ax, ctx, f"panel ({lab})", [path])
        save(fig, ctx)
        ctx.caption = "PENDING: planner/shared40_300.json not found."
        return
    A, P = d["per_arm"], d["paired"]
    tr = traffic_meta(ctx, d, "none_arm")
    geo = {lab: one(d["inputs"][lab]["knobs"].get("hard_geom"))
           for lab in d["inputs"]}
    vals: dict = {"file": norm(path), "hard_geom": geo, "traffic": tr,
                  "episodes": {k: v["episodes"] for k, v in A.items()},
                  "exposure_hours": {k: v["exposure_hours"]
                                     for k, v in A.items()},
                  "bootstrap_B": d.get("B"), "bootstrap_seed": d.get("seed"),
                  "rates_per_h": {}, "totals": {}, "ratios": {}}
    for ek, _ in SH_KEYS:
        vals["rates_per_h"][ek] = {k: v["events"][ek]["per_hour"]
                                   for k, v in A.items()}
        vals["totals"][ek] = {k: v["events"][ek]["total"]
                              for k, v in A.items()}
        vals["ratios"][ek] = {r[0]: event_entry(P[r[0]], "events", ek)
                              for r in SHARED_ROWS if r[0] != "hdr"
                              and r[0] in P}
    ctx.values = vals

    # (a), (b): rates per arm, arm geometry (filled) vs shared (open)
    for ax, (ek, tk), lab in ((ax_a, SH_KEYS[0], "a"), (ax_b, SH_KEYS[1], "b")):
        top = 0.0
        for i, arm in enumerate(ARMS):
            ra = A[f"{arm}_arm"]["events"][ek]["per_hour"]
            rs = A[f"{arm}_sh"]["events"][ek]["per_hour"]
            top = max(top, ra, rs)
            ax.plot([i - 0.15, i + 0.15], [ra, rs], color="#b0b0b0", lw=0.7,
                    zorder=2)
            ax.plot([i - 0.15], [ra], ls="none", zorder=4, **mk(arm, 5.0, True))
            ax.plot([i + 0.15], [rs], ls="none", zorder=4,
                    **mk(arm, 5.0, False))
            # arm-geometry value above its (filled) marker, shared-geometry
            # value below its (open) marker: no horizontal collisions
            ax.annotate(f"{ra:.2f}", (i - 0.15, ra), xytext=(0, 5.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FS_TXT, color=INK)
            ax.annotate(f"{rs:.2f}", (i + 0.15, rs), xytext=(0, -5.5),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=FS_TXT, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels([ARM_NAME[a] for a in ARMS])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, top * 1.28)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(True, axis="y")
        ax.set_ylabel("events per exposure hour")
        name = ("NMAC radius" if tk == "152" else "severe")
        panel_title(ax, f"({lab}) {name}, < {thr_txt(tk)}")
    hl = [Line2D([], [], ls="none", marker="o", ms=4.6, mfc="#8c8c8c",
                 mec="#4d4d4d", mew=0.65),
          Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white",
                 mec="#4d4d4d", mew=0.85)]
    fig.legend(hl, ["arm geometry: hard constraint on the arm's own risk disks",
                    "shared geometry: hard constraint on common disks"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.535, 0.915), columnspacing=2.0,
               handletextpad=0.3, borderaxespad=0.0)

    # (c), (d): paired ratios
    ys, y = [], 0.0
    for r in SHARED_ROWS:
        if r[0] == "hdr" and ys:
            y += 0.45
        ys.append(y)
        y += 1.0
    ymax = y - 1.0
    allr = [x for ek, _ in SH_KEYS for e in vals["ratios"][ek].values()
            for x in [e["ratio"]] + list(e["ci95"] or [])]
    lo, hi = ratio_limits(allr)
    for ax, (ek, tk), lab in ((ax_c, SH_KEYS[0], "c"), (ax_d, SH_KEYS[1], "d")):
        ax.axvline(1.0, color=REF, lw=0.7, zorder=1)
        for yy, r in zip(ys, SHARED_ROWS):
            if r[0] == "hdr":
                if ax is ax_c:
                    ax.text(-0.025, yy, r[1], transform=ytrans(ax), ha="right",
                            va="center", fontsize=FS_TXT, color=INK,
                            style="italic")
                if yy > 0:
                    ax.axhline(yy - 0.72, color=SEP, lw=0.5, zorder=1)
                continue
            key, txt, arm, filled = r
            if key not in P:
                pending_row(ax, yy, key)
                continue
            e = vals["ratios"][ek][key]
            forest_point(ax, yy, e["ratio"], e["ci95"], arm, filled, 4.3, 1.0)
            if ax is ax_c:
                ax.text(-0.025, yy, txt, transform=ytrans(ax), ha="right",
                        va="center", fontsize=plt.rcParams["axes.labelsize"],
                        color=INK)
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.set_yticks([])
        log_ratio_axis(ax, lo, hi, n_max=6)
        ax.grid(True, axis="x")
        name = ("NMAC radius" if tk == "152" else "severe")
        panel_title(ax, f"({lab}) paired ratios, {name}, < {thr_txt(tk)}")
    x_cd = 0.5 * (ax_c.get_position().x0 + ax_d.get_position().x1)
    fig.text(x_cd, 0.03, "rate ratio, first / second (log scale)"
             f"     {ARROW} fewer events in the second (denominator) run",
             ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"],
             color=INK)
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    pr0 = P.get("cv_sh|model_sh", {})
    idx = pr0.get("indices")
    ident = P.get("none_arm|none_sh")
    ident_txt = ""
    if ident:
        e1 = ident["events"][SH_KEYS[0][0]]
        e2 = ident["events"][SH_KEYS[1][0]]
        ident_txt = (" For arm A the two geometries gave the same per-episode "
                     f"counts in {e1['ties']} of {ident['n']} episodes "
                     f"({thr_txt('152')}) and {e2['ties']} of {ident['n']} "
                     f"({thr_txt('45')}).")
    ctx.caption = (
        "Hard-constraint geometry control at nominal OOD load "
        f"(preset {gfmt(tr['preset'])}, {gfmt(tr['n_agents'])} agents, density "
        f"{gfmt(tr['density_per_km2'])} per km²; paired episode indices "
        f"{idx[0] if idx else 'n/a'}-{idx[1] if idx else 'n/a'}, "
        f"{pr0.get('n', 'n/a')} episodes, "
        f"{A['model_sh']['exposure_hours']:.2f} h per arm). A = static "
        "observation, B = CV, C = PRISM. Arm geometry "
        "(filled markers; run knob hard_geom = "
        f"{geo.get('model_arm')}): each arm checks hard feasibility against "
        "its own risk disks, as in the planner-outcome figure. Shared geometry "
        f"(open markers; hard_geom = {geo.get('model_sh')}): every arm checks "
        "hard feasibility against the same disks at the observed intruder "
        "positions at t0, while the soft risk term stays arm-specific "
        "(definition: R3_RESULTS.md §4). (a) NMAC-radius events "
        f"(< {thr_txt('152')}) and (b) severe events (< {thr_txt('45')}) per "
        "exposure hour, current-track screen. (c, d) Paired ratios of "
        "per-hour rates (first / second run; above 1 = fewer events in the "
        "second) with 95 % confidence intervals from a paired episode "
        f"bootstrap (B = {fmt_n(d.get('B'))}, seed {d.get('seed')}): between "
        "arms within each geometry, and arm geometry / shared geometry within "
        "each arm." + ident_txt)


# ======================================================================
# 3. one-factor sensitivity
# ======================================================================
SENS_ORDER = ("floor1", "floor20", "plat05", "plat2", "wsafe6", "wsafe24",
              "normabs", "pimin20")
# figure labels: readable names, no code identifiers; the exact knob values
# (including S_REF) are written into the caption from KNOB_FULL
KNOB_FMT = {
    "sigma_floor": lambda v: rf"$\sigma$ floor {v:g} m",
    "plateau_scale": lambda v: f"plateau ×{v:g}",
    "w_safety": lambda v: f"safety weight {v:g}",
    "safety_norm": lambda v: f"{v} normalisation",
    "safety_ref": lambda v: None,                 # caption only
    "pi_min": lambda v: rf"$\pi_{{min}}$ {v:g}",
    "iso": lambda v: f"isotropic reduction: {v}",
    "gain_mode": lambda v: f"gain mode {v}",
    "sigma_scale": lambda v: rf"$\sigma$ ×{v:g}",
    "hard_geom": lambda v: f"hard geometry {v}",
}
KNOB_FULL = dict(KNOB_FMT, **{
    "safety_ref": lambda v: (f"S_REF {v:g}" if isinstance(v, (int, float))
                             else f"S_REF {v}"),
    "w_safety": lambda v: f"safety weight w_safety {v:g}",
})
KNOB_SKIP = ("model_path", "model_track_rule", "track_rule",
             "allow_rule_mismatch", "safety_ref_probe", "preset")


def plain(s: str) -> str:
    """Mathtext label -> plain text (manifest and captions)."""
    return (s.replace(r"$\sigma$", "σ").replace(r"$\pi_{min}$", "π_min")
            .replace("$", ""))


def knob_changes(d: dict, arm: str, base: str) -> list:
    ka, kb = d["inputs"][arm]["knobs"], d["inputs"][base]["knobs"]
    out = []
    for k in ka:
        if k in KNOB_SKIP:
            continue
        if ka.get(k) != kb.get(k):
            out.append((k, one(kb.get(k)), one(ka.get(k))))
    return out


def knob_label(changes: list, table: dict | None = None) -> str:
    table = table or KNOB_FMT
    parts = []
    for k, _, new in changes:
        f = table.get(k)
        try:
            s = f(new) if f else f"{k} {new}"
        except (TypeError, ValueError):
            s = f"{k} {new}"
        if s:
            parts.append(s)
    return ", ".join(parts) if parts else "(no knob change recorded)"


def sens_rows(d: dict, fname: str):
    """Settings of one sensitivity aggregate: list of (label, pair key,
    knob changes, arm-C-only flag), plus notes on anything odd."""
    arms = list(d["per_arm"])
    rows, notes = [], []
    sufs = sorted({a.split("_", 1)[1] for a in arms if "_" in a} - {"base"})
    for s in sufs:
        if f"cv_{s}|model_{s}" in d["paired"]:
            key, c_only = f"cv_{s}|model_{s}", False
        elif f"model_{s}" in arms and f"cv_base|model_{s}" in d["paired"]:
            key, c_only = f"cv_base|model_{s}", True
        else:
            continue
        ch = knob_changes(d, f"model_{s}", "model_base")
        if not c_only:
            chc = knob_changes(d, f"cv_{s}", "cv_base")
            if chc != ch:
                notes.append(f"{fname}: CV and PRISM knob changes differ "
                             f"{chc} vs {ch}")
        rows.append((knob_label(ch) + (" (PRISM arm only)" if c_only else ""),
                     key, ch, c_only))
    return rows, notes


def fig_sensitivity(ctx: Ctx) -> None:
    extra = sorted({os.path.basename(p)[len("sens300_"):-len(".json")]
                    for p in glob.glob(f"{PL}/sens300_*.json")} - set(SENS_ORDER))
    iso_files = sorted(norm(p) for p in glob.glob(f"{PL}/*iso*.json"))
    names = list(SENS_ORDER) + extra
    files = {nm: norm(f"{PL}/sens300_{nm}.json") for nm in names}
    for p in iso_files:
        nm = os.path.basename(p)[:-5]
        if nm.startswith("sens300_"):
            continue                      # already listed through `extra`
        names.append(nm)
        files[nm] = p
    loaded = {nm: ctx.json(files[nm]) for nm in names}

    # the baseline pair is carried by every file; check that it is the same
    base, base_src, sigs = None, None, {}
    for nm in names:
        d = loaded[nm]
        if d and "cv_base|model_base" in d["paired"]:
            pb = d["paired"]["cv_base|model_base"]
            sigs[nm] = json.dumps({ek: [pb["events"][ek]["total_a"],
                                        pb["events"][ek]["total_b"],
                                        pb["events"][ek]["rate_ratio"]["est"],
                                        pb["events"][ek]["rate_ratio"]["ci"]]
                                   for ek, _ in SH_KEYS}, sort_keys=True)
            if base is None:
                base, base_src = d, nm
    consistent = len(set(sigs.values())) <= 1
    if not consistent:
        ctx.notes.append("baseline pair cv_base|model_base differs between "
                         f"sensitivity files: {sorted(sigs)}")

    rows: list = []
    if base is not None:
        rows.append(dict(label="baseline", nm=base_src,
                         key="cv_base|model_base", c_only=False, state="ok"))
    else:
        rows.append(dict(label="baseline", state="missing",
                         missing=f"{PL}/sens300_*.json"))
    for nm in names:
        d = loaded[nm]
        if d is None:
            rows.append(dict(label=nm, state="missing", missing=files[nm]))
            continue
        got, notes = sens_rows(d, nm)
        ctx.notes += notes
        if not got:
            rows.append(dict(label=f"{nm}: schema not recognised",
                             state="bad", missing=files[nm]))
            ctx.notes.append(f"{files[nm]}: no cv_X|model_X or "
                             "cv_base|model_X pair found")
        for lab, key, ch, c_only in got:
            rows.append(dict(label=lab, nm=nm, key=key, c_only=c_only,
                             state="ok"))
    if not iso_files:
        rows.append(dict(label="isotropic reduction (iso = max / min)",
                         state="missing", missing=f"{PL}/*iso*.json"))

    vals = {"baseline_source": base_src,
            "baseline_identical_across_files": consistent,
            "files": {nm: files[nm] for nm in names}, "rows": []}
    nrow = len(rows)
    fig_h = 1.35 + 0.245 * nrow
    fig = plt.figure(figsize=(W_FULL, fig_h))
    gs = fig.add_gridspec(1, 2, left=0.315, right=0.99,
                          top=1 - 0.62 / fig_h, bottom=0.56 / fig_h, wspace=0.08)
    axes = [fig.add_subplot(gs[0, j]) for j in range(2)]
    allr = []
    for rw in rows:
        if rw["state"] == "ok":
            for ek, _ in SH_KEYS:
                e = loaded[rw["nm"]]["paired"][rw["key"]]["events"][ek][
                    "rate_ratio"]
                allr += [e["est"]] + list(e["ci"] or [])
    lo, hi = ratio_limits(allr)
    for j, (ek, tk) in enumerate(SH_KEYS):
        ax = axes[j]
        for i in range(nrow):
            if i % 2 == 1:
                ax.axhspan(i - 0.5, i + 0.5, color=ZEBRA, lw=0, zorder=0)
        ax.axvline(1.0, color=REF, lw=0.7, zorder=1)
        bref = None
        for i, rw in enumerate(rows):
            if rw["state"] != "ok":
                pending_row(ax, i, os.path.basename(rw["missing"])
                            if rw["state"] == "missing"
                            else "schema not recognised")
                if j == 0:
                    ctx.pend(f"row '{rw['label']}'", [rw["missing"]])
                continue
            d = loaded[rw["nm"]]
            pp = d["paired"][rw["key"]]
            e = event_entry(pp, "events", ek)
            if rw["label"] == "baseline":
                bref = e["ratio"]
            forest_point(ax, i, e["ratio"], e["ci95"], "cv", True, 4.3, 1.1)
            if j == 0:
                arm_a, arm_b = rw["key"].split("|")
                vals["rows"].append({
                    "label": plain(rw["label"]), "file": files[rw["nm"]],
                    "pair": rw["key"], "arm_C_only": rw["c_only"],
                    "rates_per_h": {x: {k2: d["per_arm"][x]["events"][k2][
                        "per_hour"] for k2, _ in SH_KEYS}
                        for x in (arm_a, arm_b)},
                    "n_paired": pp.get("n"), "indices": pp.get("indices"),
                    "exposure_hours": d["per_arm"][arm_b]["exposure_hours"],
                    "knob_changes": ([] if rw["label"] == "baseline" else
                                     knob_changes(d, arm_b, "model_base")),
                    "ratios": {k2: event_entry(pp, "events", k2)
                               for k2, _ in SH_KEYS}})
        if bref is not None:
            ax.axvline(bref, color="#bdbdbd", lw=0.8, zorder=1)
        ax.set_ylim(nrow - 0.5, -0.5)
        ax.set_yticks([])
        log_ratio_axis(ax, lo, hi, n_max=5)
        ax.grid(True, axis="x")
        name = ("NMAC radius" if tk == "152" else "severe")
        panel_title(ax, f"({'ab'[j]}) {name}, < {thr_txt(tk)}")
    for ax in axes:
        ax.axhline(0.5, color="#9e9e9e", lw=0.6, zorder=1)
    for i, rw in enumerate(rows):
        axes[0].text(-0.03, i, rw["label"], transform=ytrans(axes[0]),
                     ha="right", va="center",
                     fontsize=plt.rcParams["axes.labelsize"],
                     weight="bold" if (i == 0 and base is not None) else "normal",
                     color=INK if rw["state"] == "ok" else MUTE)
    x_ab = 0.5 * (axes[0].get_position().x0 + axes[1].get_position().x1)
    fig.text(x_ab, 0.11 / fig_h, "event-rate ratio B / C, CV / PRISM (log scale)"
             f"     {ARROW} fewer events with PRISM",
             ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"],
             color=INK)
    hl = [Line2D([], [], color=ARM_INK["cv"], lw=1.1, **mk("cv", 4.3, True)),
          Line2D([], [], color="#bdbdbd", lw=0.8),
          Line2D([], [], color=REF, lw=0.7)]
    fig.legend(hl, ["B / C rate ratio with 95 % CI", "baseline ratio",
                    "ratio 1 (no difference)"],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(x_ab, 1 - 0.3 / fig_h), columnspacing=2.0,
               handlelength=2.0, handletextpad=0.5, borderaxespad=0.0)
    ctx.values = vals
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    if base is None:
        ctx.caption = "PENDING: no sens300_*.json aggregate was found."
        return
    bk = base["inputs"]["model_base"]["knobs"]
    bdesc = ", ".join(KNOB_FULL[k](one(bk[k])) for k in
                      ("sigma_floor", "plateau_scale", "w_safety",
                       "safety_norm", "safety_ref", "pi_min", "iso",
                       "gain_mode") if k in bk and KNOB_FULL[k](one(bk[k])))
    bdesc = plain(bdesc)
    row_desc = "; ".join(
        plain(knob_label(r["knob_changes"], KNOB_FULL))
        + (" (PRISM arm only)" if r["arm_C_only"] else "")
        for r in vals["rows"] if r["label"] != "baseline")
    pb = base["paired"]["cv_base|model_base"]
    tr = traffic_meta(ctx, base, "cv_base")
    ctx.caption = (
        "One-factor sensitivity of the closed-loop comparison B/C "
        "(CV / PRISM) at nominal OOD load (preset "
        f"{gfmt(tr['preset'])}, {gfmt(tr['n_agents'])} agents, density "
        f"{gfmt(tr['density_per_km2'])} per km²; paired episode indices "
        f"{pb['indices'][0]}-{pb['indices'][1]}, {pb['n']} episodes, "
        f"{base['per_arm']['model_base']['exposure_hours']:.2f} h per arm and "
        "setting). Each row changes the listed setting(s) in both arms "
        f"relative to the baseline ({bdesc}); rows in order: {row_desc}. A "
        "row marked 'PRISM arm only' was run for PRISM alone and is paired "
        "with the baseline CV arm. (a) "
        f"NMAC-radius events (< {thr_txt('152')}), (b) severe events "
        f"(< {thr_txt('45')}), current-track screen. Points: ratio of "
        "per-hour event rates (above 1 = fewer events with PRISM); bars: 95 % "
        "confidence intervals from a paired "
        f"episode bootstrap (B = {fmt_n(base.get('B'))}, seed "
        f"{base.get('seed')}). Light grey vertical line: baseline estimate; "
        "dark line: ratio 1. The "
        "baseline pair is identical in every sensitivity file: "
        f"{'yes' if consistent else 'NO (see manifest notes)'}." +
        ("" if iso_files else " Isotropic-reduction rows are not available "
         "(no planner/*iso*.json aggregate at build time)."))


# ======================================================================
# 4. closed-loop calibration
# ======================================================================
COV_SETS = (("id40", "ID nominal", "testid"),
            ("ood40", "OOD nominal load", "testood"))
CELL_ORDER = ("all", "interv0_react0", "interv0_react1", "interv1_react0",
              "interv1_react1")
CELL_LABEL = {"offline": "offline\ntest set", "all": "closed loop\nall",
              "interv0_react0": "own −\nintr −", "interv0_react1": "own −\nintr +",
              "interv1_react0": "own +\nintr −", "interv1_react1": "own +\nintr +"}


def _hkeys(cells: dict) -> list:
    return sorted(cells["all"], key=lambda h: float(h.rstrip("s")))


def _levels(cells: dict, hk: str) -> list:
    return sorted(int(k[3:]) for k in cells["all"][hk] if k.startswith("cov"))


def _closed_loop_inputs(ctx: Ctx):
    """Core aggregates (with their coverage blocks) and the offline reports
    of the same checkpoint, plus the horizons and nominal levels read from
    whichever coverage block exists. Shared by the full (supplement) and the
    compact (main-text) closed-loop calibration figures."""
    data = {}
    for key, lab, split in COV_SETS:
        d = ctx.json(f"{PL}/core_{key}.json")
        rep = ctx.json(f"{EV}/r3_k3_s{PLANNER_SEED}_{split}_report.json")
        data[key] = (d, rep)
    hks, levels = ["5s", "10s", "15s"], None
    for key, (d, rep) in data.items():
        cov = (d or {}).get("coverage", {})
        if "model" in cov and "cells" in cov["model"]:
            hks = _hkeys(cov["model"]["cells"])
            levels = _levels(cov["model"]["cells"], hks[0])
            break
    return data, hks, (levels if levels is not None else [])


def _strata_text(sp) -> str:
    """Definition of the manoeuvre strata from the aggregate's parameters."""
    if not sp:
        return "strata definitions: see aggregate_r3.py"
    return (f"own + = ownship manoeuvre within [t, t + "
            f"{sp.get('horizon_s'):g} s] relative to its state at the "
            f"forecast time t (lateral change > {sp.get('own_dev_m'):g} m "
            f"or speed change > {(1 - sp.get('own_vfrac')) * 100:.0f} % of "
            f"the desired speed; own_ref = {sp.get('own_ref')}); intr + = "
            f"intruder reacting ({sp.get('intruder_reacting')}: its "
            "reaction layer is latched at some 20 Hz tick within the "
            "horizon, or its intent at t is REACTIVE_AVOID)")


def _closed_loop_legend(fig, fig_h: float) -> None:
    hl = [Line2D([], [], color=ARM_INK["model"], ls="-", lw=0.95,
                 **mk("model", 3.3, True)),
          Line2D([], [], color=ARM_INK["cv"], ls=ARM_LS["cv"], lw=0.95,
                 **mk("cv", 3.3, True)),
          Line2D([], [], color=ARM_INK["model"], ls=(0, (1.0, 1.3)), lw=0.9,
                 marker="D", ms=3.2, mfc="white", mec=ARM_INK["model"],
                 mew=0.8),
          Line2D([], [], color=REF, lw=0.7)]
    fig.legend(hl, ["C: PRISM, forecasts in the loop",
                    "B: CV, forecasts in the loop",
                    "C: PRISM, offline test set", "nominal level"],
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.54, 1 - 0.28 / fig_h),
               columnspacing=1.6, handlelength=2.2, handletextpad=0.5,
               borderaxespad=0.0)


def fig_closed_loop_calibration(ctx: Ctx) -> None:
    data, hks, levels = _closed_loop_inputs(ctx)
    nr = max(len(levels), 1)
    fig_h = 1.55 * nr + 1.05
    fig, axes = plt.subplots(nr, 2, figsize=(W_FULL, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.995, top=1 - 0.5 / fig_h,
                        bottom=0.7 / fig_h, hspace=0.14, wspace=0.05)
    groups = ["offline"] + list(CELL_ORDER)
    gx = {g: [4 * gi + k for k in range(len(hks))] for gi, g in enumerate(groups)}
    vals: dict = {}
    for c, (key, lab, split) in enumerate(COV_SETS):
        d, rep = data[key]
        cov = (d or {}).get("coverage", {})
        ok = all(a in cov and "cells" in cov[a] for a in ("cv", "model"))
        vals[key] = {"file": norm(f"{PL}/core_{key}.json"),
                     "offline_report": norm(
                         f"{EV}/r3_k3_s{PLANNER_SEED}_{split}_report.json"),
                     "closed_loop": {}, "offline": {}, "n": {},
                     "strata_params": (cov.get("model") or {}).get(
                         "strata_params"),
                     "pred_files": {a: (cov.get(a) or {}).get("files")
                                    for a in ("cv", "model")},
                     "forecasts": {a: (cov.get(a) or {}).get("forecasts")
                                   for a in ("cv", "model")},
                     "dirs": {a: (cov.get(a) or {}).get("dir")
                              for a in ("cv", "model")}}
        for r, L in enumerate(levels or [None]):
            ax = axes[r, c]
            if not ok or L is None:
                pending_panel(ax, ctx, f"{key} closed-loop coverage",
                              [f"{PL}/core_{key}.json (coverage block)"])
                continue
            ax.axhline(L / 100, color=REF, lw=0.7, zorder=1)
            for gi in range(1, len(groups)):
                ax.axvline(4 * gi - 1.0, color=SEP if gi > 1 else "#9e9e9e",
                           lw=0.5 if gi > 1 else 0.7, zorder=1)
            if rep is not None:
                ys = [rep["overall"].get(f"cov{L}_{h}") for h in hks]
                vals[key]["offline"][f"cov{L}"] = dict(zip(hks, ys))
                vals[key]["offline"]["n_samples"] = rep.get("n_samples")
                ax.plot(gx["offline"], ys, color=ARM_INK["model"], lw=0.9,
                        ls=(0, (1.0, 1.3)), marker="D", ms=3.2, mfc="white",
                        mec=ARM_INK["model"], mew=0.8, zorder=4)
            else:
                ax.text(np.mean(gx["offline"]), 0.5, "not available",
                        color=MUTE, fontsize=FS_TXT, ha="center", va="center",
                        rotation=90, style="italic")
                if r == 0:
                    ctx.pend(f"{key} offline reference", [
                        f"{EV}/r3_k3_s{PLANNER_SEED}_{split}_report.json"])
            for cell in CELL_ORDER:
                for arm in ("cv", "model"):
                    C = cov[arm]["cells"][cell]
                    ys = [C[h][f"cov{L}"] for h in hks]
                    vals[key]["closed_loop"].setdefault(arm, {}).setdefault(
                        cell, {})[f"cov{L}"] = dict(zip(hks, ys))
                    vals[key]["n"].setdefault(arm, {})[cell] = {
                        h: C[h]["n"] for h in hks}
                    ax.plot(gx[cell], ys, color=ARM_INK[arm], ls=ARM_LS[arm],
                            lw=0.95, zorder=4 if arm == "model" else 3,
                            **mk(arm, 3.3, True))
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(True, axis="y")
            ax.set_xlim(-0.8, gx[groups[-1]][-1] + 0.8)
            ax.set_xticks([x for g in groups for x in gx[g]])
            if r == len(levels) - 1:
                ax.set_xticklabels([h.rstrip("s") for g in groups for h in hks],
                                   fontsize=FS_MIN)
                ax.tick_params(axis="x", pad=1.5)
                for g in groups:
                    ax.text(np.mean(gx[g]), -0.135, CELL_LABEL[g],
                            transform=xtrans(ax), ha="center", va="top",
                            fontsize=FS_TXT, color=INK, linespacing=1.1)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"HPD coverage\n(nominal {L / 100:.1f})")
            else:
                ax.set_yticklabels([])
            if r == 0:
                panel_title(ax, f"({'ab'[c]}) {lab}")
    _closed_loop_legend(fig, fig_h)
    fig.text(0.54, 0.012, "numbers under the panels: forecast horizon [s]; "
             "own = ownship manoeuvring, intr = intruder reacting",
             ha="center", va="bottom", fontsize=FS_TXT, color=MUTE)
    ctx.values = vals
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    parts = []
    sp = None
    for key, lab, split in COV_SETS:
        v = vals.get(key, {})
        if not v.get("closed_loop"):
            parts.append(f"{key}: PENDING")
            continue
        sp = sp or v["strata_params"]
        d = data[key][0]
        tr = traffic_meta(ctx, d, "model")
        n15 = {a: {cell: v["n"][a][cell][hks[-1]] for cell in CELL_ORDER}
               for a in ("cv", "model")}
        parts.append(
            f"{key} (preset {gfmt(tr['preset'])}, {gfmt(tr['n_agents'])} "
            f"agents, density {gfmt(tr['density_per_km2'])} per km²): forecast "
            f"logs of {v['pred_files']['model']} episodes per arm; scored "
            f"forecasts at {hks[-1].rstrip('s')} s: PRISM all "
            f"{fmt_n(n15['model']['all'])} (strata "
            + ", ".join(fmt_n(n15['model'][cc]) for cc in CELL_ORDER[1:])
            + f"), CV all {fmt_n(n15['cv']['all'])} (strata "
            + ", ".join(fmt_n(n15['cv'][cc]) for cc in CELL_ORDER[1:])
            + f"); offline test set n = {fmt_n(v['offline'].get('n_samples'))}")
    d0 = data[COV_SETS[0][0]][0] or data[COV_SETS[1][0]][0]
    mc = d0.get("mc_draws") if d0 else None
    hz = ", ".join(h.rstrip("s") for h in hks)
    own = _strata_text(sp)
    mdl = None
    for key, _, _ in COV_SETS:
        rep = data[key][1]
        if rep:
            mdl = job_name(rep.get("model"))
            break
    ctx.caption = (
        "Closed-loop HPD coverage of the forecasts the planner consumed, "
        "(a) ID nominal (id40) and (b) OOD nominal load (ood40). "
        "Rows: nominal level (horizontal line). Within each group the three "
        f"points are the {hz} s horizons. Groups: offline test-set coverage "
        f"of the same PRISM checkpoint ({mdl or 'n/a'}; evaluation report "
        "'overall' value, current-track test set); closed-loop coverage "
        "pooled over all logged forecasts; and four strata, "
        f"{own}. PRISM coverage by Monte Carlo HPD membership ({mc} draws "
        "per forecast point), CV (single Gaussian) in closed form. Point "
        "estimates; the source files carry no intervals. "
        + " ".join(p + "." for p in parts))


# ---------------------------------------------------------------------
# 4b. compact main-text variant: nominal 0.9 only, manoeuvre strata
# ---------------------------------------------------------------------
MAIN_CELLS = ("all", "interv1_react0", "interv0_react1", "interv1_react1")
MAIN_LABEL = {"offline": "offline\ntest set", "all": "closed loop\nall",
              "interv1_react0": "own\nonly", "interv0_react1": "intruder\nonly",
              "interv1_react1": "both"}
MAIN_LEVEL = 90


def fig_closed_loop_calibration_main(ctx: Ctx) -> None:
    """Main-text variant of the closed-loop calibration figure: the nominal
    0.9 level only, the pooled closed loop and the three strata in which
    someone manoeuvres (own only, intruder only, both). The 0.5 / 0.7 levels
    and the stratum with neither manoeuvre remain in the full figure."""
    data, hks, levels = _closed_loop_inputs(ctx)
    L = MAIN_LEVEL if MAIN_LEVEL in levels else (max(levels) if levels else MAIN_LEVEL)
    fig_h = 2.3
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, fig_h), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, top=1 - 0.46 / fig_h,
                        bottom=0.66 / fig_h, wspace=0.06)
    groups = ["offline"] + list(MAIN_CELLS)
    gx = {g: [4 * gi + k for k in range(len(hks))] for gi, g in enumerate(groups)}
    vals: dict = {"level": L / 100, "horizons": hks, "cells": list(MAIN_CELLS),
                  "cell_labels": {c: MAIN_LABEL[c].replace("\n", " ")
                                  for c in MAIN_CELLS}, "sets": {}}
    for c, (key, lab, split) in enumerate(COV_SETS):
        ax = axes[c]
        d, rep = data[key]
        cov = (d or {}).get("coverage", {})
        ok = all(a in cov and "cells" in cov[a] for a in ("cv", "model"))
        V = {"file": norm(f"{PL}/core_{key}.json"),
             "offline_report": norm(
                 f"{EV}/r3_k3_s{PLANNER_SEED}_{split}_report.json"),
             "offline": {}, "closed_loop": {}, "n": {},
             "strata_params": (cov.get("model") or {}).get("strata_params"),
             "pred_files": {a: (cov.get(a) or {}).get("files")
                            for a in ("cv", "model")},
             "mc_draws": (d or {}).get("mc_draws")}
        vals["sets"][key] = V
        panel_title(ax, f"({'ab'[c]}) {lab}")
        if not ok:
            pending_panel(ax, ctx, f"{key} closed-loop coverage",
                          [f"{PL}/core_{key}.json (coverage block)"])
            continue
        ax.axhline(L / 100, color=REF, lw=0.7, zorder=1)
        for gi in range(1, len(groups)):
            ax.axvline(4 * gi - 1.0, color=SEP if gi > 1 else "#9e9e9e",
                       lw=0.5 if gi > 1 else 0.7, zorder=1)
        if rep is not None:
            ys = [rep["overall"].get(f"cov{L}_{h}") for h in hks]
            V["offline"] = dict(zip(hks, ys))
            V["offline_n"] = rep.get("n_samples")
            V["model"] = rep.get("model")
            ax.plot(gx["offline"], ys, color=ARM_INK["model"], lw=0.9,
                    ls=(0, (1.0, 1.3)), marker="D", ms=3.2, mfc="white",
                    mec=ARM_INK["model"], mew=0.8, zorder=4)
        else:
            ax.text(np.mean(gx["offline"]), 0.5, "not available", color=MUTE,
                    fontsize=FS_TXT, ha="center", va="center", rotation=90,
                    style="italic")
            ctx.pend(f"{key} offline reference",
                     [f"{EV}/r3_k3_s{PLANNER_SEED}_{split}_report.json"])
        for cell in MAIN_CELLS:
            for arm in ("cv", "model"):
                C = cov[arm]["cells"][cell]
                ys = [C[h][f"cov{L}"] for h in hks]
                V["closed_loop"].setdefault(arm, {})[cell] = dict(zip(hks, ys))
                V["n"].setdefault(arm, {})[cell] = {h: C[h]["n"] for h in hks}
                ax.plot(gx[cell], ys, color=ARM_INK[arm], ls=ARM_LS[arm],
                        lw=0.95, zorder=4 if arm == "model" else 3,
                        **mk(arm, 3.3, True))
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, axis="y")
        ax.set_xlim(-0.8, gx[groups[-1]][-1] + 0.8)
        ax.set_xticks([x for g in groups for x in gx[g]])
        ax.set_xticklabels([h.rstrip("s") for g in groups for h in hks],
                           fontsize=FS_MIN)
        ax.tick_params(axis="x", pad=1.5)
        for g in groups:
            ax.text(np.mean(gx[g]), -0.2, MAIN_LABEL[g], transform=xtrans(ax),
                    ha="center", va="top", fontsize=FS_TXT, color=INK,
                    linespacing=1.1)
        if c == 0:
            ax.set_ylabel(f"HPD coverage\n(nominal {L / 100:.1f})")
    _closed_loop_legend(fig, fig_h)
    fig.text(0.54, 0.01, "numbers under the panels: forecast horizon [s]; "
             "own = ownship manoeuvring, intruder = intruder reacting, within "
             "the forecast horizon",
             ha="center", va="bottom", fontsize=FS_TXT, color=MUTE)
    ctx.values = vals
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    h15 = hks[-1]
    S = {}
    for key, lab, split in COV_SETS:
        V = vals["sets"][key]
        if not V.get("closed_loop"):
            continue
        pr = [v for cell in MAIN_CELLS
              for v in V["closed_loop"]["model"][cell].values()]
        cv = [v for cell in MAIN_CELLS
              for v in V["closed_loop"]["cv"][cell].values()]
        S[key] = {"pooled_15s": V["closed_loop"]["model"]["all"][h15],
                  "offline_15s": V["offline"].get(h15),
                  "prism_range": [min(pr), max(pr)],
                  "cv_range": [min(cv), max(cv)],
                  "cv_pooled": V["closed_loop"]["cv"]["all"],
                  "n_15s": {a: {cell: V["n"][a][cell][h15] for cell in MAIN_CELLS}
                            for a in ("cv", "model")}}
        V["summary"] = S[key]
    if len(S) < 2:
        ctx.caption = ("Closed-loop calibration (main-text variant): "
                       + "; ".join(f"{k}: not available" for k, *_ in COV_SETS
                                   if k not in S) + ".")
        return
    a, b = S["id40"], S["ood40"]
    d0 = data["id40"][0] or data["ood40"][0]
    sp = vals["sets"]["id40"].get("strata_params") or vals["sets"]["ood40"].get(
        "strata_params")
    mdl = job_name(vals["sets"]["id40"].get("model") or vals["sets"]["ood40"].get("model"))
    n_txt = "; ".join(
        f"{lab}: PRISM " + ", ".join(
            f"{vals['cell_labels'][cell]} {fmt_n(S[key]['n_15s']['model'][cell])}"
            for cell in MAIN_CELLS)
        + "; CV " + ", ".join(
            f"{vals['cell_labels'][cell]} {fmt_n(S[key]['n_15s']['cv'][cell])}"
            for cell in MAIN_CELLS)
        for key, lab, split in COV_SETS)
    ctx.caption = (
        "Calibration of the forecasts the planner actually consumed, at the "
        f"nominal {L / 100:.1f} level: PRISM's forecasts keep their offline "
        f"calibration in the closed loop ({h15.rstrip('s')} s coverage "
        f"{a['pooled_15s']:.3f} pooled versus {a['offline_15s']:.3f} offline "
        f"on ID nominal, {b['pooled_15s']:.3f} versus {b['offline_15s']:.3f} "
        "on OOD nominal load; "
        f"{a['prism_range'][0]:.2f}–{a['prism_range'][1]:.2f} and "
        f"{b['prism_range'][0]:.2f}–{b['prism_range'][1]:.2f} over the "
        "manoeuvre strata and horizons), whereas the constant-velocity "
        "forecasts the same planner consumed cover only "
        f"{a['cv_range'][0]:.2f}–{a['cv_range'][1]:.2f} (ID) and "
        f"{b['cv_range'][0]:.2f}–{b['cv_range'][1]:.2f} (OOD) of the nominal "
        f"{L} %. (a) ID nominal (id40), (b) OOD nominal load (ood40); "
        f"checkpoint {mdl or 'n/a'}. Within each group the three points are "
        f"the {', '.join(h.rstrip('s') for h in hks)} s horizons. Groups: "
        "offline test-set coverage of the same checkpoint (evaluation report "
        "'overall' value, current-track test set); closed-loop coverage "
        "pooled over all logged forecasts; and the strata in which someone "
        "manoeuvres — own only (own +, intr −), intruder only (own −, "
        f"intr +), both (own +, intr +), where {_strata_text(sp)}. PRISM "
        "coverage by Monte Carlo HPD membership "
        f"({vals['sets']['id40'].get('mc_draws') or (d0 or {}).get('mc_draws')} "
        "draws per forecast point), CV (single Gaussian) in closed form. Point "
        "estimates; the source files carry no intervals. Scored forecasts at "
        f"{h15.rstrip('s')} s — {n_txt}. The 0.5 and 0.7 levels and the stratum "
        "with neither manoeuvre are in the supplementary version of this "
        "figure.")


# ======================================================================
# 5. reacquisition
# ======================================================================
REACQ_SPLITS = (("testid", "ID", "s4_testid_indexcurrent"),
                ("testood", "OOD", "s4_testood_indexcurrent"))


def fig_reacquisition(ctx: Ctx) -> None:
    p8path = newest(f"{R3}/pred_p5_p6_p8*.json")
    p8 = ctx.json(p8path) if p8path else None
    if p8path is None:
        ctx.missing.append(norm(f"{R3}/pred_p5_p6_p8*.json"))
    fig, axes = plt.subplots(2, 4, figsize=(W_FULL, 4.3))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.885, bottom=0.12,
                        hspace=0.6, wspace=0.45)
    vals: dict = {"p8_file": p8path}
    seed_off = {s: -0.33 + 0.07 * s for s in SEEDS}
    classes = ("0", "1")
    class_name = {"0": "one track", "1": "reacquired"}
    for r, (split, lab, orc_tag) in enumerate(REACQ_SPLITS):
        reps = {s: ctx.json(f"{EV}/r3_k3_s{s}_{split}_report.json")
                for s in SEEDS}
        orc = ctx.json(f"{EV}/{orc_tag}_report.json")
        have = [s for s in SEEDS if reps[s] is not None]
        miss = [s for s in SEEDS if reps[s] is None]
        V = {"seeds_present": have, "seeds_missing": miss, "per_seed": {},
             "oracle": None, "n": {},
             "oracle_report": norm(f"{EV}/{orc_tag}_report.json"),
             "n_samples": (reps[have[0]].get("n_samples") if have else None)}
        ns = {}
        for s in have:
            S = reps[s]["strata"]["reacq"]
            V["per_seed"][s] = {c: {"minADE": S[c]["minADE"], "NLL": S[c]["NLL"],
                                    "n": S[c]["n"]} for c in classes}
            ns[f"seed{s}"] = tuple(S[c]["n"] for c in classes)
        if orc is not None:
            S = orc["strata"]["reacq"]
            V["oracle"] = {c: {"minADE": S[c]["minADE"], "NLL": S[c]["NLL"],
                               "n": S[c]["n"]} for c in classes}
            V["oracle_model"] = orc.get("model")
            V["oracle_rules"] = {"index_rule": orc.get("index_rule"),
                                 "feature_rule": orc.get("feature_rule")}
            ns["oracle"] = tuple(S[c]["n"] for c in classes)
        V["class_sizes_equal"] = len(set(ns.values())) <= 1
        if not V["class_sizes_equal"]:
            ctx.notes.append(f"{split}: reacq class sizes differ between "
                             f"models {ns}")
        V["n"] = ns
        for j, met in enumerate(("minADE", "NLL")):
            ax = axes[r, j]
            unit = " [m]" if met == "minADE" else ""
            panel_title(ax, f"({'abcdefgh'[4 * r + j]}) {lab}: {met}{unit}")
            if not have and orc is None:
                pending_panel(ax, ctx, f"{lab} {met}",
                              [f"{EV}/r3_k3_s*_{split}_report.json"])
                continue
            for ci, c in enumerate(classes):
                for s in have:
                    ax.plot([ci + seed_off[s]], [V["per_seed"][s][c][met]],
                            ls="none", zorder=4, **mk("model", 3.6, True))
                if V["oracle"]:
                    ax.plot([ci + 0.26], [V["oracle"][c][met]], ls="none",
                            marker="*", ms=7.0, mfc=ORACLE, mec="white",
                            mew=0.4, zorder=5)
            ref = (V["per_seed"][have[0]] if have else V["oracle"])
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"{class_name[c]}\nn = {fmt_n(ref[c]['n'])}"
                                for c in classes], fontsize=FS_MIN)
            ax.set_xlim(-0.6, 1.6)
            ys = [v[c][met] for v in V["per_seed"].values() for c in classes]
            ys += [V["oracle"][c][met] for c in classes] if V["oracle"] else []
            span = (max(ys) - min(ys)) or 1.0
            ax.set_ylim(min(ys) - 0.1 * span, max(ys) + 0.12 * span)
            ax.grid(True, axis="y")
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if V["oracle"] is None and j == 0:
                ctx.pend(f"{lab} oracle-association model",
                         [f"{EV}/{orc_tag}_report.json"])
            if miss and j == 0:
                ctx.pend(f"{lab} seeds {', '.join(map(str, miss))}",
                         [f"{EV}/r3_k3_s{s}_{split}_report.json" for s in miss])
        # P8: paired per-sample differences with episode-cluster CIs
        P8 = ((p8 or {}).get("P8") or {}).get(split)
        V["p8"] = P8
        for j, met in enumerate(("minADE", "NLL")):
            ax = axes[r, 2 + j]
            unit = " [m]" if met == "minADE" else ""
            panel_title(ax, f"({'abcdefgh'[4 * r + 2 + j]}) {lab}: PRISM − oracle")
            ax.set_ylabel(f"{met} difference{unit}", labelpad=2)
            if not P8:
                pending_panel(ax, ctx, f"{lab} P8 paired difference ({met})",
                              [p8path or f"{R3}/pred_p5_p6_p8*.json"],
                              note=f"no P8 block '{split}'")
                continue
            ax.axhline(0, color=REF, lw=0.7, zorder=1)
            seeds8 = sorted(int(k[4:]) for k in P8 if k.startswith("seed"))
            cats = tuple(f"reacq{c}" for c in classes)
            for ci, cat in enumerate(cats):
                for s in seeds8:
                    e = P8[f"seed{s}"][met][cat]
                    x = ci + seed_off.get(s, 0.0) + 0.1
                    ax.plot([x, x], e["ci"], color=ARM_INK["model"], lw=0.9,
                            solid_capstyle="butt", zorder=3)
                    ax.plot([x], [e["est"]], ls="none", zorder=4,
                            **mk("model", 3.4, True))
            nsz = {cat: P8[f"seed{seeds8[0]}"][met][cat]["n"] for cat in cats}
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([f"{class_name[cat[-1]]}\nn = {fmt_n(nsz[cat])}"
                                for cat in cats], fontsize=FS_MIN)
            ax.set_xlim(-0.6, len(cats) - 0.4)
            ax.grid(True, axis="y")
            ax.yaxis.set_major_locator(MaxNLocator(5))
        vals[split] = V
    hl = [Line2D([], [], ls="none", **mk("model", 3.6, True)),
          Line2D([], [], ls="none", marker="*", ms=7.0, mfc=ORACLE,
                 mec="white", mew=0.4),
          Line2D([], [], color=ARM_INK["model"], lw=0.9)]
    fig.legend(hl, ["C: PRISM, current-track inputs (one marker per seed)",
                    "oracle track association (submitted model)",
                    "95 % CI (episode-cluster bootstrap)"],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.535, 0.95), columnspacing=1.2,
               handlelength=1.6, handletextpad=0.4, borderaxespad=0.0)
    ctx.values = vals
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    defs = None
    for s in SEEDS:
        pth = norm(f"{EV}/r3_k3_s{s}_testid_report.json")
        if os.path.isfile(pth):
            defs = read_json(pth).get("metric_definitions", {}).get("reacq")
            break
    if defs:
        keep = [x.strip() for x in defs.split(";")
                if x.strip().split(" ")[0] in classes]
        defs = "; ".join(keep)
    ci_txt = None
    if p8:
        flat = " ".join((p8.get("definitions") or "").split())
        m = re.search(r"CIs?:\s*([^.]+)", flat)
        ci_txt = m.group(1).strip() if m else None
    orc_i = vals.get("testid", {})
    orc_desc = "n/a"
    if orc_i.get("oracle_model"):
        rr = orc_i.get("oracle_rules") or {}
        orc_desc = (f"{job_name(orc_i['oracle_model'])}, the submitted model, "
                    f"with {rr.get('feature_rule')}-rule input features on the "
                    f"{rr.get('index_rule')}-track sample set")
    seeds_txt = {sp: ", ".join(map(str, vals[sp]["seeds_present"])) or "none"
                 for sp, _, _ in REACQ_SPLITS if sp in vals}
    p8seeds = {sp: ", ".join(str(int(k[4:])) for k in (vals[sp]["p8"] or {})
                             if k.startswith("seed")) or "none"
               for sp, _, _ in REACQ_SPLITS if sp in vals}
    nsamp = {sp: vals[sp].get("n_samples") for sp, _, _ in REACQ_SPLITS
             if sp in vals}
    pend_txt = ""
    if ctx.pending:
        pend_txt = (" Not available at build time: "
                    + "; ".join(p["item"] for p in ctx.pending) + ".")
    ctx.caption = (
        "Prediction error by reacquisition class on the current-track test "
        f"sets (ID n = {fmt_n(nsamp.get('testid'))}, OOD n = "
        f"{fmt_n(nsamp.get('testood'))} samples). Classes (evaluation-report "
        f"definition; 'one track' = class 0, 'reacquired' = class 1): "
        f"{defs or 'n/a'}. (a, b, e, f) minADE [m] and "
        "position-marginal mixture NLL [(100 m)² units] per class for the "
        "current-track PRISM K = 3 models (diamonds, one per seed; ID seeds "
        f"{seeds_txt.get('testid')}, OOD seeds {seeds_txt.get('testood')}) "
        f"and the oracle-association model (star; {orc_desc}). (c, d, g, h) "
        "Pre-registered analysis P8: per-sample difference, current-track "
        "PRISM seed minus oracle, joined on (episode file, t0, agent), per "
        f"class; bars: 95 % confidence intervals ({ci_txt or 'see P8 file'}); "
        "where no bar is visible the interval is narrower than the marker. P8 "
        f"seeds: ID {p8seeds.get('testid')}, OOD {p8seeds.get('testood')} "
        f"(file {os.path.basename(p8path) if p8path else 'none'})." + pend_txt)


# ======================================================================
# 6. rollout diagnostics
# ======================================================================
ROLL = (("rollout_r3_k3_s1_idheldout_T1.json", "ID held-out"),
        ("rollout_r3_k3_s1_oodheldout_T1.json", "OOD held-out"),
        ("rollout_r3_k3_s1_oodheldout_T1695.json", "OOD held-out"))
COND_MK = ("^", "v", "p")


def _cc(g: dict):
    hs = sorted(g, key=lambda h: float(h.rstrip("s")))
    noms = sorted(g[hs[0]], key=lambda n: int(n[3:]))
    return hs, noms


def fig_rollout(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(W_FULL, 4.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.095,
                        hspace=0.5, wspace=0.36)
    vals: dict = {"sets": {}, "legacy_vs_aligned": {}}
    ymin_all = 1.0
    pit_top = 0.0
    loaded = []
    for fn, lab in ROLL:
        d = ctx.json(f"{R3}/{fn}")
        loaded.append((fn, lab, d))
        if d is not None:
            g = d["bootstrap"]["group"]["conditional_coverage"]
            hs, noms = _cc(g)
            ymin_all = min([ymin_all] + [g[h][n]["mean_ci95"][0] for h in hs
                                         for n in noms])
            pit_top = max([pit_top] + d["bootstrap"]["group"]["pit_hist"][
                "15s"]["ci95_hi"])
    for k, (fn, lab, d) in enumerate(loaded):
        ax, axp = axes[0, k], axes[1, k]
        if d is None:
            pending_panel(ax, ctx, f"rollout coverage {fn}", [f"{R3}/{fn}"])
            pending_panel(axp, ctx, f"rollout PIT {fn}", [f"{R3}/{fn}"])
            panel_title(ax, f"({'abc'[k]}) {lab}")
            panel_title(axp, f"({'efg'[k]}) {lab}")
            continue
        b = d["bootstrap"]
        g = b["group"]["conditional_coverage"]
        hs, noms = _cc(g)
        xs = [float(h.rstrip("s")) for h in hs]
        V = {"file": norm(f"{R3}/{fn}"), "run": d.get("run"),
             "model": d.get("model"), "T": d.get("T"),
             "alignment": d.get("alignment"), "n_groups": d.get("n_groups"),
             "n_scored": d.get("n_scored"), "units": b["group"]["n_units"],
             "B": b.get("B"), "boot_seed": b.get("seed"),
             "branches_per_group": d.get("branches_per_group"),
             "n_episode_files": d.get("n_episode_files"), "coverage": {},
             "pit_15s": b["group"]["pit_hist"]["15s"]}
        for nom in noms:
            L = int(nom[3:])
            m = np.array([g[h][nom]["mean"] for h in hs])
            lo = np.array([g[h][nom]["mean_ci95"][0] for h in hs])
            hi = np.array([g[h][nom]["mean_ci95"][1] for h in hs])
            V["coverage"][nom] = {h: {"mean": g[h][nom]["mean"],
                                      "ci95": g[h][nom]["mean_ci95"],
                                      "n_groups": g[h][nom].get("n_groups")}
                                  for h in hs}
            ax.axhline(L / 100, color=REF, lw=0.6, zorder=1)
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], color=LEVEL_COL.get(L, INK),
                        lw=1.0, marker="D", ms=3.0, mfc=LEVEL_COL.get(L, INK),
                        mec="white", mew=0.4, capsize=1.6, elinewidth=0.75,
                        zorder=4)
            ax.text(xs[-1] + 0.9, m[-1], f"{L / 100:.1f}", color=LEVEL_COL.get(L, INK),
                    fontsize=FS_TXT, va="center", ha="left")
        ax.set_xlim(xs[0] - 1.5, xs[-1] + 3.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_ylim(max(0.0, ymin_all - 0.05), 1.0)
        ax.grid(True, axis="y")
        ax.set_xlabel("horizon [s]", labelpad=1.0)
        if k == 0:
            ax.set_ylabel("mean conditional coverage")
        panel_title(ax, f"({'abc'[k]}) {lab}, T = {d.get('T'):g}")
        # PIT at 15 s
        pb = b["group"]["pit_hist"]["15s"]
        hist = np.array(pb["hist"])
        lo, hi = np.array(pb["ci95_lo"]), np.array(pb["ci95_hi"])
        nb = len(hist)
        cen = (np.arange(nb) + 0.5) / nb
        axp.bar(cen, hist, width=0.84 / nb, color=ARM_FILL["model"],
                edgecolor=ARM_INK["model"], lw=0.5, zorder=3)
        axp.errorbar(cen, hist, yerr=[hist - lo, hi - hist], fmt="none",
                     ecolor=INK, elinewidth=0.6, capsize=1.1, zorder=4)
        axp.axhline(1.0 / nb, color=REF, lw=0.7, zorder=5)
        axp.set_xlim(0, 1)
        axp.set_ylim(0, pit_top * 1.12)
        axp.set_xticks([0, 0.5, 1])
        axp.set_xticklabels(["0", "0.5", "1"])
        axp.grid(True, axis="y")
        axp.set_xlabel("PIT at 15 s", labelpad=1.0)
        if k == 0:
            axp.set_ylabel("fraction")
        panel_title(axp, f"({'efg'[k]}) {lab}, T = {d.get('T'):g}")
        vals["sets"][fn] = V

    # (d) submitted model: legacy offset vs timestamp-aligned, nominal 0.9
    ax = axes[0, 3]
    pairs = []
    for pa in sorted(glob.glob(f"{R3}/rollout_cond_*_aligned.json")):
        pl = pa.replace("_aligned.json", "_legacy.json")
        da, dl = ctx.json(pa), ctx.json(pl)
        if da is None or dl is None:
            continue
        tag = "ID" if "test_id" in os.path.basename(pa) else "OOD"
        pairs.append((tag, da.get("T"), pa, pl, da, dl))
    pairs.sort(key=lambda t: (t[0] != "ID", t[1]))
    if not pairs:
        pending_panel(ax, ctx, "legacy vs aligned (submitted model)",
                      [f"{R3}/rollout_cond_*_aligned.json / _legacy.json"])
    else:
        nomk = "nom90"
        L = int(nomk[3:])
        ax.axhline(L / 100, color=REF, lw=0.6, zorder=1)
        lo_all = 1.0
        for i, (tag, T, pa, pl, da, dl) in enumerate(pairs):
            key = f"{tag}, T = {T:g}"
            vv = {}
            mids = {}
            for jj, (dd, filled) in enumerate(((dl, False), (da, True))):
                g = dd["bootstrap"]["group"]["conditional_coverage"]
                hs, _ = _cc(g)
                xs = np.array([float(h.rstrip("s")) for h in hs])
                off = (i - (len(pairs) - 1) / 2) * 1.3 + (0.36 if filled else -0.36)
                m = np.array([g[h][nomk]["mean"] for h in hs])
                lo = np.array([g[h][nomk]["mean_ci95"][0] for h in hs])
                hi = np.array([g[h][nomk]["mean_ci95"][1] for h in hs])
                lo_all = min(lo_all, float(lo.min()))
                mids[filled] = (xs + off, m)
                ax.errorbar(xs + off, m, yerr=[m - lo, hi - m], fmt="none",
                            ecolor=ORACLE, elinewidth=0.7, capsize=1.2, zorder=3)
                ax.plot(xs + off, m, ls="none", marker=COND_MK[i % 3], ms=3.8,
                        mfc=ORACLE if filled else "white", mec=ORACLE, mew=0.75,
                        zorder=4)
                if filled:
                    (x0, y0), (x1, y1) = mids[False], mids[True]
                    for a0, b0, a1, b1 in zip(x0, y0, x1, y1):
                        ax.plot([a0, a1], [b0, b1], color="#a0a0a0", lw=0.6,
                                zorder=2)
                vv["aligned" if filled else "legacy"] = {
                    "file": norm(pa if filled else pl),
                    "alignment": dd.get("alignment"),
                    "model_minus_rollout_time_s": dd.get(
                        "model_minus_rollout_time_s"),
                    "units": dd["bootstrap"]["group"]["n_units"],
                    "coverage": {nk: {h: {"mean": g[h][nk]["mean"],
                                          "ci95": g[h][nk]["mean_ci95"]}
                                      for h in hs} for nk in g[hs[0]]}}
            vals["legacy_vs_aligned"][key] = vv
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlim(xs[0] - 2.5, xs[-1] + 2.5)
        ax.set_ylim(max(0.0, lo_all - 0.04), 1.0)
        ax.grid(True, axis="y")
        ax.set_xlabel("horizon [s]", labelpad=1.0)
        panel_title(ax, "(d) submitted model")
        vals["legacy_vs_aligned_model"] = pairs[0][4].get("model")
        vals["legacy_vs_aligned_level"] = L / 100
    # (h): the figure's single legend, in the empty slot; group headers are
    # entries with an invisible handle
    axl = axes[1, 3]
    axl.axis("off")
    blank = Line2D([], [], ls="none", marker=None)
    hs = [blank]
    ts = ["(a)–(c), (e)–(g)"]
    hs += [Line2D([], [], color=LEVEL_COL[L], lw=1.0, marker="D", ms=3.0,
                  mfc=LEVEL_COL[L], mec="white", mew=0.4) for L in (50, 70, 90)]
    ts += ["nominal 0.5 (95 % CI)", "nominal 0.7", "nominal 0.9"]
    hs += [Line2D([], [], color=REF, lw=0.7),
           Patch(fc=ARM_FILL["model"], ec=ARM_INK["model"], lw=0.5)]
    ts += ["nominal / uniform", "PIT fraction (95 % CI)"]
    if pairs:
        hs.append(blank)
        ts.append(f"(d) submitted model, {vals['legacy_vs_aligned_level']:.1f}")
        for i, (tag, T, *_r) in enumerate(pairs):
            hs.append(Line2D([], [], ls="none", marker=COND_MK[i % 3], ms=3.8,
                             mfc=ORACLE, mec=ORACLE))
            ts.append(f"{tag}, T = {T:g}")
        dl0 = pairs[0][5]
        hs += [Line2D([], [], ls="none", marker="o", ms=3.8, mfc="white",
                      mec=ORACLE, mew=0.75),
               Line2D([], [], ls="none", marker="o", ms=3.8, mfc=ORACLE,
                      mec=ORACLE)]
        ts += [f"legacy offset ({dl0.get('model_minus_rollout_time_s'):g} s)",
               "timestamp-aligned"]
    leg = axl.legend(hs, ts, loc="center left", frameon=False,
                     bbox_to_anchor=(-0.32, 0.5), handlelength=1.6,
                     borderaxespad=0.0, labelspacing=0.32, handletextpad=0.5)
    leg._legend_box.align = "left"
    for txt, h in zip(leg.get_texts(), hs):
        if h is blank:
            txt.set_fontweight("bold")
    ctx.values = vals
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    parts = []
    for fn, lab, d in loaded:
        if d is None:
            parts.append(f"{fn}: PENDING")
            continue
        parts.append(f"{lab}, T = {d.get('T'):g}: run "
                     f"{os.path.basename(str(d.get('run')))}, "
                     f"{d.get('n_episode_files')} episode files, "
                     f"{fmt_n(d['bootstrap']['group']['n_units'])} scored "
                     f"groups of {fmt_n(d.get('n_groups'))}, "
                     f"{fmt_n(d['bootstrap']['group']['pit_hist']['15s'].get('n_values'))} "
                     "PIT values at 15 s")
    d0 = next((d for _, _, d in loaded if d is not None), None)
    pa_txt = ""
    if pairs:
        pa_txt = (" (d) Submitted model "
                  f"({job_name(pairs[0][4].get('model'))}): nominal 0.9 "
                  "coverage scored with the legacy time alignment (open markers; "
                  "model time minus rollout time = "
                  f"{pairs[0][5].get('model_minus_rollout_time_s'):g} s) and with "
                  "timestamp alignment (filled; "
                  f"{pairs[0][4].get('model_minus_rollout_time_s'):g} s), for "
                  + ", ".join(f"{t} T = {T:g} ({fmt_n(da['bootstrap']['group']['n_units'])} / "
                              f"{fmt_n(dl['bootstrap']['group']['n_units'])} groups aligned / legacy)"
                              for t, T, _pa, _pl, da, dl in pairs) + ".")
    ctx.caption = (
        "Rollout-based conditional calibration of the planner checkpoint "
        f"({job_name(d0.get('model')) if d0 else 'n/a'}). "
        f"Reference: {d0.get('reference_note') if d0 else 'n/a'}; "
        f"{one(d0.get('branches_per_group')) if d0 else 'n/a'} branch futures "
        "per group (episode, origin step, target slot). (a)–(c) Mean over "
        "groups of the per-group HPD coverage at nominal 0.5 / 0.7 / 0.9 "
        "(horizontal lines) versus horizon; whiskers: 95 % percentile "
        f"intervals of a group bootstrap (B = {d0['bootstrap']['B'] if d0 else 'n/a'}, "
        f"seed {d0['bootstrap']['seed'] if d0 else 'n/a'}). T scales the "
        "predicted standard deviations. (e)–(g) PIT histogram at 15 s "
        "(bars) with 95 % group-bootstrap intervals; line: uniform. "
        + "; ".join(parts) + "." + pa_txt)


# ======================================================================
# 7. recalibration transfer
# ======================================================================
def fig_recal_transfer(ctx: Ctx) -> None:
    path = f"{R3}/recal_r3_k3_s1.json"
    d = ctx.json(path)
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.9), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.8, bottom=0.17,
                        wspace=0.08)
    if d is None:
        for ax, lab in zip(axes, "ab"):
            pending_panel(ax, ctx, f"panel ({lab})", [path])
        save(fig, ctx)
        ctx.caption = "PENDING: recal_r3_k3_s1.json not found."
        return
    es = d.get("eval_set", {})
    rows = [("held-out OOD test set", d["n_test"], d["before"], d["after"],
             f"ep. {es.get('episode_index_min')}–{es.get('episode_index_max')}")]
    for k, v in d.get("transfer", {}).items():
        if isinstance(v, dict) and "before" in v:
            rows.append((SET_LABEL.get(k, k), v.get("n_samples"), v["before"],
                         v["after"], os.path.basename(str(v.get("run")))))
    covk = "cov90_15s"
    L = int(covk[3:5])
    vals = {"file": norm(path), "T": d["T"], "n_cal": d["n_cal"],
            "rows": [{"set": r[0], "n": r[1], "run_or_range": r[4],
                      "source_key": (list(d.get("transfer", {}))[i - 1]
                                     if i else "eval_set"),
                      "before": {covk: r[2].get(covk), "NLL": r[2].get("NLL")},
                      "after": {covk: r[3].get(covk), "NLL": r[3].get("NLL")}}
                     for i, r in enumerate(rows)]}
    n = len(rows)
    for j, (met, xl) in enumerate(((covk, f"coverage of the nominal {L} % HPD "
                                          "region at 15 s"),
                                   ("NLL", "position-marginal NLL [(100 m)² units]"))):
        ax = axes[j]
        for i in range(n):
            if i % 2 == 1:
                ax.axhspan(i - 0.5, i + 0.5, color=ZEBRA, lw=0, zorder=0)
        if met == covk:
            ax.axvline(L / 100, color=REF, lw=0.7, zorder=1)
        for i, (lab, nn, bf, af, _x) in enumerate(rows):
            xb, xa = bf.get(met), af.get(met)
            if xb is None or xa is None:
                pending_row(ax, i, f"{lab} {met}")
                continue
            ax.annotate("", xy=(xa, i), xytext=(xb, i),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                        color=ARM_INK["model"], shrinkA=3.2,
                                        shrinkB=3.4, mutation_scale=6.5),
                        zorder=3)
            ax.plot([xb], [i], ls="none", zorder=4, **mk("model", 4.2, False))
            ax.plot([xa], [i], ls="none", zorder=5, **mk("model", 4.2, True))
            for x, ha, dx in ((xb, "right" if xb < xa else "left",
                               -5 if xb < xa else 5),
                              (xa, "left" if xb < xa else "right",
                               5 if xb < xa else -5)):
                ax.annotate(f"{x:.3f}" if met == covk else f"{x:.2f}", (x, i),
                            xytext=(dx, 0), textcoords="offset points", ha=ha,
                            va="center", fontsize=FS_TXT, color=MUTE)
        ax.axhline(0.5, color="#9e9e9e", lw=0.6, zorder=1)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        xs = [r[2].get(met) for r in rows] + [r[3].get(met) for r in rows]
        xs = [x for x in xs if x is not None] + ([L / 100] if met == covk else [])
        span = max(xs) - min(xs)
        ax.set_xlim(min(xs) - 0.24 * span, max(xs) + 0.24 * span)
        ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, axis="x")
        ax.set_xlabel(xl, labelpad=1.5)
        panel_title(ax, f"({'ab'[j]}) " + (f"{L} % coverage at 15 s"
                                           if met == covk else "NLL"))
    for i, (lab, nn, bf, af, x) in enumerate(rows):
        axes[0].text(-0.03, i - 0.16, lab, transform=ytrans(axes[0]),
                     ha="right", va="center",
                     fontsize=plt.rcParams["axes.labelsize"], color=INK)
        axes[0].text(-0.03, i + 0.22, f"n = {fmt_n(nn)}",
                     transform=ytrans(axes[0]), ha="right", va="center",
                     fontsize=FS_TXT, color=MUTE)
    hl = [Line2D([], [], ls="none", **mk("model", 4.2, False)),
          Line2D([], [], ls="none", **mk("model", 4.2, True))]
    fig.legend(hl, ["before recalibration (T = 1)",
                    f"after recalibration (T = {d['T']:g})"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.59, 0.905), columnspacing=2.0,
               handletextpad=0.4, borderaxespad=0.0)
    ctx.values = vals
    save(fig, ctx)
    fs = d.get("fit_set", {})
    fit = d.get("fit", {})
    ctx.caption = (
        "Scalar temperature recalibration of the planner checkpoint "
        f"({job_name(d.get('model'))}) and its transfer. T = {d['T']:g} "
        f"(unrounded {fit.get('T_unrounded')}) was fitted "
        f"({fit.get('method', 'n/a')}) on {fmt_n(d['n_cal'])} samples of the "
        f"calibration runs (seeds {fs.get('seed_min')}-{fs.get('seed_max')}) "
        "and evaluated unchanged on the held-out OOD test set (preset "
        "test_ood, episodes "
        f"{es.get('episode_index_min')}-{es.get('episode_index_max')}, seeds "
        f"{es.get('seed_min')}-{es.get('seed_max')}, {fmt_n(d['n_test'])} "
        f"samples; seed overlap with the fit set "
        f"{(d.get('seed_overlap') or {}).get('n_overlap')}) and on the "
        "factor-wise OOD sets (one shifted factor per set; control = no "
        "shift; 'all factors' = every factor shifted; n per row). (a) "
        "Coverage of the nominal 90 % HPD region at "
        "15 s, (b) position-marginal mixture NLL, before (open, T = 1) and "
        f"after (filled) scaling. HPD by Monte Carlo "
        f"({(d.get('hpd_mc') or {}).get('draws_per_sample_step')} draws per "
        "sample and step, seeded). Point estimates; the source carries no "
        "intervals.")


# ======================================================================
# 8. training curves
# ======================================================================
def find_job(tag: str):
    js = [j for j in sorted(glob.glob(f"{MODELS}/*_{tag}"))
          if os.path.isfile(f"{j}/final/model_final.pt")
          and os.path.isfile(f"{j}/metrics.jsonl")]
    return norm(js[-1]) if js else None


def load_job(ctx: Ctx, tag: str):
    j = find_job(tag)
    ov = None
    if j is None:
        # an interrupted job documented in Paper/rebuttal/r3/job_overrides.json
        # (DECISIONS D6.17): its own metrics plus the resumed job's, and the
        # checkpoint that was actually evaluated
        ovp = f"{R3}/job_overrides.json"
        ov = (json.load(open(ovp, encoding="utf-8")) if os.path.isfile(ovp)
              else {}).get(tag)
        if ov is None:
            miss = norm(f"{MODELS}/*_{tag}/final/model_final.pt")
            if miss not in ctx.missing:
                ctx.missing.append(miss)
            return None
        j = norm(ov["job_dir"])
    ctx.use(f"{j}/metrics.jsonl")
    jj = ctx.json(f"{j}/job.json") or {}
    cfg = ctx.json(f"{j}/config.json") or {}
    if ov is not None:
        jj = dict(jj, status="interrupted+resumed",
                  summary=dict(jj.get("summary") or {}, best=ov["best"]))
    val = {}
    for mf in [f"{j}/metrics.jsonl"] + [norm(p) for p in (ov or {}).get("extra_metrics", [])]:
        if mf != f"{j}/metrics.jsonl":
            ctx.use(mf)
        with open(mf, encoding="utf-8-sig") as f:
            for line in f:
                if '"val"' not in line:
                    continue
                r = json.loads(line)
                if r.get("split") == "val":
                    val[int(r["epoch"])] = r           # last row of an epoch
    ep = sorted(val)
    c = cfg.get("config", {})
    return {"dir": j, "job": os.path.basename(j), "tag": tag,
            "status": jj.get("status"),
            "epochs": ep,
            "NLL": [val[e].get("NLL") for e in ep],
            "minADE": [val[e].get("minADE") for e in ep],
            "cov90_15s": [val[e].get("cov90_15s") for e in ep],
            "best": (jj.get("summary") or {}).get("best"),
            "val_roster": (cfg.get("roster") or {}).get("val"),
            "train_roster": (cfg.get("roster") or {}).get("train"),
            "val_max_samples": (c.get("optim") or {}).get("val_max_samples"),
            "best_metric": (c.get("checkpoint") or {}).get("best_metric"),
            "n_modes": (c.get("model") or {}).get("n_modes"),
            "loss_variant": (c.get("loss") or {}).get("variant")}


def fig_training(ctx: Ctx) -> None:
    seeds = {s: load_job(ctx, f"r3_k3_s{s}") for s in SEEDS}
    abl = {t[0]: load_job(ctx, t[0]) for t in ABLATIONS}
    onex = load_job(ctx, ONE_X_TAG)
    have = [s for s in SEEDS if seeds[s]]
    miss = [s for s in SEEDS if not seeds[s]]
    fig, axes = plt.subplots(2, 3, figsize=(W_FULL, 4.5), sharex=False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.85, bottom=0.1,
                        hspace=0.42, wspace=0.3)
    mets = (("NLL", "validation NLL [(100 m)² units]"),
            ("minADE", "validation minADE [m]"))

    def curve(ax, J, met, color, ls, marker, lw, ms=2.2, alpha=1.0,
              best=False, z=3):
        y = [np.nan if v is None else v for v in J[met]]
        ax.plot(J["epochs"], y, color=color, ls=ls, lw=lw, marker=marker,
                ms=ms, mfc=color, mec=color, alpha=alpha, zorder=z)
        sel = J.get("best_metric") or "val/NLL"      # the metric the job kept its checkpoint by
        if best and J.get("best") and sel == f"val/{met}":
            b = J["best"]
            ax.plot([b["epoch"]], [b["value"]], ls="none", marker="o", ms=5.0,
                    mfc="none", mec=color, mew=0.9, zorder=z + 1)

    # column 1: K = 3 seeds
    for r, (met, yl) in enumerate(mets):
        ax = axes[r, 0]
        if not have:
            pending_panel(ax, ctx, f"K=3 seeds {met}",
                          [f"{MODELS}/*_r3_k3_s*"])
            continue
        for s in have:
            main = (s == PLANNER_SEED)
            curve(ax, seeds[s], met, ARM_INK["model"], "-",
                  "D" if main else None, 1.5 if main else 0.7,
                  ms=2.0, alpha=1.0 if main else 0.55, best=True,
                  z=5 if main else 3)
        ax.set_ylabel(yl)
        ax.grid(True, axis="y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        panel_title(ax, f"({'ad'[r]}) K = 3 seeds")
        if miss and r == 0:
            ctx.pend(f"K=3 seeds {', '.join(map(str, miss))}",
                     [f"{MODELS}/*_r3_k3_s{s}" for s in miss])
    # column 2: ablations (seed 0)
    abl_pending = [t for t in ABLATIONS if abl[t[0]] is None]
    for r, (met, yl) in enumerate(mets):
        ax = axes[r, 1]
        drawn = 0
        for tag, lab, col, ls, mkr in ABLATIONS:
            J = abl[tag]
            if J is None or all(v is None for v in J[met]):
                continue
            if tag == "r3_pointnet_s0" and met == "NLL":
                # loss variant "mse": sigma/rho/pi receive no gradient, so its
                # validation NLL scores an untrained spread head (~2.4); drawn
                # it would only stretch the axis of the trained models. The
                # caption says so; no note on the figure.
                continue
            curve(ax, J, met, col, ls, mkr, 1.4 if tag == "r3_k3_s0" else 1.0,
                  ms=2.4, best=True, z=5 if tag == "r3_k3_s0" else 3)
            drawn += 1
        if not drawn:
            pending_panel(ax, ctx, f"ablations {met}",
                          [f"{MODELS}/*_{t[0]}" for t in ABLATIONS])
            continue
        ax.grid(True, axis="y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        panel_title(ax, f"({'be'[r]}) ablations, seed 0")
        if abl_pending and r == 0:
            ctx.pend("ablations " + ", ".join(t[1] for t in abl_pending),
                     [f"{MODELS}/*_{t[0]}" for t in abl_pending])
    # column 3: data scale
    for r, (met, yl) in enumerate(mets):
        ax = axes[r, 2]
        if onex is None:
            pending_panel(ax, ctx, f"data scale 1x vs 5x ({met})",
                          [f"{MODELS}/*_{ONE_X_TAG}"],
                          note=f"1x corpus model, tag {ONE_X_TAG}")
            panel_title(ax, f"({'cf'[r]}) data scale: 1x vs 5x corpus")
            continue
        for s in have:
            curve(ax, seeds[s], met, ARM_INK["model"], "-", None, 0.7,
                  alpha=0.55)
        curve(ax, onex, met, ORACLE, (0, (4.0, 1.6)), "X", 1.3, ms=2.6,
              best=True, z=5)
        ax.grid(True, axis="y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        panel_title(ax, f"({'cf'[r]}) data scale: 1x vs 5x corpus")
        if have and onex["val_roster"] != seeds[have[0]]["val_roster"] and r == 0:
            # said in the caption; no note on the figure
            ctx.notes.append("1x and 5x jobs use different validation rosters")
    for ax in axes[1, :]:
        if ax.get_xticks().size:                 # not a placeholder
            ax.set_xlabel("epoch", labelpad=1.0)
    # legend: the thick purple line is the full K = 3 model in every column
    # (seed of the planner model in a, d; seed 0 in b, e), so it is one entry
    hs = [Line2D([], [], color=ARM_INK["model"], lw=1.5, marker="D", ms=2.0),
          Line2D([], [], color=ARM_INK["model"], lw=0.7, alpha=0.55),
          Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none",
                 mec=ARM_INK["model"], mew=0.9)]
    ts = ["full model, K = 3 (planner seed in a, d; seed 0 in b, e)",
          "other K = 3 seeds", "kept checkpoint"]
    for tag, lab, col, ls, mkr in ABLATIONS[1:]:
        if abl[tag] is not None:
            hs.append(Line2D([], [], color=col, ls=ls, lw=1.0, marker=mkr,
                             ms=2.6))
            ts.append(lab)
    if onex is not None:
        hs.append(Line2D([], [], color=ORACLE, ls=(0, (4.0, 1.6)), lw=1.3,
                         marker="X", ms=2.6))
        ts.append("1x corpus")
    fig.legend(hs, ts, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.535, 0.895), columnspacing=1.6,
               handlelength=2.2, handletextpad=0.45, borderaxespad=0.0)
    vals = {"seeds": {s: seeds[s] for s in have},
            "ablations": {t: abl[t] for t in abl if abl[t]},
            "one_x": onex, "seeds_missing": miss,
            "ablations_missing": [t[0] for t in abl_pending]}
    ctx.values = vals
    save(fig, ctx)
    ref = seeds[have[0]] if have else next((j for j in abl.values() if j), None)
    ctx.caption = (
        "Training evidence from the per-epoch validation rows of each "
        "completed job (metrics.jsonl). Validation set: "
        f"{', '.join(os.path.basename(v) for v in (ref or {}).get('val_roster') or []) or 'n/a'}"
        f", at most {fmt_n((ref or {}).get('val_max_samples'))} samples per "
        "evaluation; current-track rule. Top row: position-marginal mixture "
        "NLL; bottom row: minADE. (a, d) PRISM K = 3, seeds "
        f"{', '.join(map(str, have)) or 'none'} (thick line: seed {PLANNER_SEED}, "
        "the planner model; open circle: epoch of the kept checkpoint, drawn in "
        "the panel of the metric the job selected by, from job.json: validation "
        "NLL for the K = 3 and ablation jobs, validation minADE for Point-MSE "
        "and the 1x-corpus job). (b, e) Ablations, all seed 0: "
        + ", ".join(t[1] for t in ABLATIONS if abl[t[0]]) +
        (f"; not available: {', '.join(t[1] for t in abl_pending)}" if abl_pending else "")
        + (". Point-MSE is drawn in (e) only: it is trained by squared error on the mean, so its spread "
           "head receives no gradient and its NLL is not a trained quantity" if abl.get("r3_pointnet_s0") else "")
        + f". (c, f) Data scale: 1x corpus model (tag {ONE_X_TAG}; 85 epochs, "
        "configuration of the submitted 1x model) against the 5x-corpus K = 3 seeds; "
        "the two use different validation sets, so compare them on the common test "
        "samples of Table 10" + ("" if onex else " — not available (no completed job)")
        + (f". Missing K = 3 seeds: {', '.join(map(str, miss))}." if miss else "."))


# ======================================================================
# 9 / 10. per-set dot plots (factor-wise OOD, parameter sensitivity)
# ======================================================================
def cov_nominal(key: str):
    """Nominal level of a coverage key ('cov90_15s' -> 0.9), else None."""
    m = re.match(r"cov(\d+)_", key)
    return int(m.group(1)) / 100 if m else None


METRIC_TITLE = {"minADE": "minADE", "NLL": "NLL",
                "cov90_15s": "90 % HPD coverage, 15 s",
                "type_acc": "type accuracy"}
FACTOR_SETS = ("control", "density", "wind", "intent", "platform", "all")
PARAM_SETS = ("baseline", "dwell_hi", "dwell_lo", "notice_hi", "notice_lo",
              "sensor_2x")
# readable row names; the evaluation-set keys stay in the manifest and the
# caption. Names that carry a number are checked against the case
# description of param_sensitivity_results.json at run time (set_label()).
SET_LABEL = {
    "control": "control", "density": "density", "wind": "wind",
    "intent": "intent", "platform": "platform", "all": "all factors",
    "baseline": "baseline", "dwell_hi": "dwell ×1.3", "dwell_lo": "dwell ×0.7",
    "notice_hi": "notice +0.2", "notice_lo": "notice −0.2",
    "sensor_2x": "sensor noise ×2",
}


def param_case_descriptions(ctx: Ctx | None = None) -> dict:
    """case -> description of the simulator-parameter perturbation sets."""
    p = norm(PARAM_SENS_JSON)
    if not os.path.isfile(p):
        return {}
    if ctx is not None:
        ctx.use(p)
    out: dict = {}

    def walk(o):
        if isinstance(o, dict):
            if "case" in o and isinstance(o.get("description"), str):
                out.setdefault(o["case"], o["description"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(read_json(p))
    return out


def set_label(key: str, desc: dict) -> str:
    """Display name of an evaluation set. A name with a number is used only
    when that number occurs in the recorded case description; otherwise
    the key itself is shown (never a number the sources do not carry)."""
    lab = SET_LABEL.get(key, key)
    nums = re.findall(r"\d+(?:\.\d+)?", lab)
    if not nums:
        return lab
    d = desc.get(key, "")
    return lab if all(n in d for n in nums) else key


def _set_dotplot(ctx: Ctx, prefix: str, sets, metrics, ref_set: str,
                 title_word: str, ref_name: str) -> dict:
    reps = {s: ctx.json(f"{EV}/r3_k3_s{PLANNER_SEED}_{prefix}{s}_report.json")
            for s in sets}
    desc = param_case_descriptions(ctx) if prefix == "sens_" else {}
    labels = {s: set_label(s, desc) for s in sets}
    have = [s for s in sets if reps[s] is not None]
    n = len(sets)
    fig_h = 1.15 + 0.32 * n
    fig, axes = plt.subplots(1, len(metrics), figsize=(W_FULL, fig_h),
                             sharey=True)
    fig.subplots_adjust(left=0.125, right=0.985, top=1 - 0.55 / fig_h,
                        bottom=0.5 / fig_h, wspace=0.1)
    vals = {"sets": {}, "labels": labels, "descriptions": desc,
            "missing": [s for s in sets if reps[s] is None]}
    for s in have:
        o = reps[s]["overall"]
        vals["sets"][s] = {"n_samples": reps[s].get("n_samples"),
                           "eval_runs": reps[s].get("eval_runs"),
                           "model": reps[s].get("model"),
                           **{m: o.get(m) for m, _ in metrics}}
    for j, (met, xl) in enumerate(metrics):
        nominal = cov_nominal(met)
        ax = axes[j]
        panel_title(ax, f"({'abc'[j]}) {METRIC_TITLE.get(met, met)}")
        if not have:
            pending_panel(ax, ctx, f"{title_word} {met}",
                          [f"{EV}/r3_k3_s{PLANNER_SEED}_{prefix}{s}_report.json"
                           for s in sets])
            continue
        for i in range(n):
            if i % 2 == 1:
                ax.axhspan(i - 0.5, i + 0.5, color=ZEBRA, lw=0, zorder=0)
        if nominal is not None:
            ax.axvline(nominal, color=REF, lw=0.7, zorder=1)
        elif ref_set in have and vals["sets"][ref_set].get(met) is not None:
            ax.axvline(vals["sets"][ref_set][met], color="#bdbdbd", lw=0.8,
                       zorder=1)
        xs = [vals["sets"][s][met] for s in have
              if vals["sets"][s].get(met) is not None]
        xs += [nominal] if nominal is not None else []
        for i, s in enumerate(sets):
            if reps[s] is None:
                pending_row(ax, i, f"r3_k3_s{PLANNER_SEED}_{prefix}{s}_report.json")
                if j == 0:
                    ctx.pend(f"set '{s}'", [
                        f"{EV}/r3_k3_s{PLANNER_SEED}_{prefix}{s}_report.json"])
                continue
            x = vals["sets"][s].get(met)
            if x is None:
                continue
            ax.plot([x], [i], ls="none", zorder=4, **mk("model", 4.4, True))
            ax.annotate(f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}", (x, i),
                        xytext=(5, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=FS_TXT, color=MUTE)
        if xs:
            lo, hi = min(xs), max(xs)
            span = (hi - lo) or abs(hi) * 0.1 or 1.0
            ax.set_xlim(lo - 0.12 * span, hi + 0.34 * span)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f"{labels[s]}\nn = {fmt_n(vals['sets'][s]['n_samples'])}"
             if s in have else labels[s] for s in sets], fontsize=FS_MIN)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, axis="x")
        ax.set_xlabel(xl, labelpad=1.5)
    hl = [Line2D([], [], ls="none", **mk("model", 4.4, True)),
          Line2D([], [], color="#bdbdbd", lw=0.8),
          Line2D([], [], color=REF, lw=0.7)]
    fig.legend(hl, ["C: PRISM (planner checkpoint)", f"{ref_name} value",
                    "nominal level"],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.555, 1 - 0.3 / fig_h), columnspacing=2.2,
               handlelength=2.0, handletextpad=0.5, borderaxespad=0.0)
    ctx.values = vals
    save(fig, ctx)
    return vals


def fig_factor_ood(ctx: Ctx) -> None:
    vals = _set_dotplot(
        ctx, "factor_", FACTOR_SETS,
        (("minADE", "minADE [m]"),
         ("cov90_15s", "90 % HPD coverage at 15 s"),
         ("type_acc", "type accuracy")),
        "control", "factor-wise OOD", "control")
    runs = "; ".join(f"{vals['labels'][s]} ({s}): "
                     f"{os.path.basename(str(one(v.get('eval_runs'))))}"
                     for s, v in vals["sets"].items())
    mdl = job_name(next((v.get("model") for v in vals["sets"].values()), None))
    ctx.caption = (
        "Factor-wise OOD evaluation of the planner checkpoint "
        f"({mdl or 'n/a'}): one test set per shifted factor, control = no "
        "shift, 'all factors' = all factors shifted together. (a) minADE [m], "
        "(b) coverage of the nominal 90 % HPD region at 15 s (dark line: "
        "nominal level), (c) auxiliary type-classification accuracy; grey "
        "lines in (a) and (c): control value. Current-track samples; n per "
        "set on the axis. Point "
        "estimates (report 'overall'); the source reports carry no "
        f"intervals. Test runs: {runs or 'n/a'}."
        + (f" Not available: {', '.join(vals['missing'])}." if vals["missing"]
           else ""))


def fig_param_sensitivity(ctx: Ctx) -> None:
    vals = _set_dotplot(
        ctx, "sens_", PARAM_SETS,
        (("minADE", "minADE [m]"),
         ("NLL", "position-marginal NLL [(100 m)² units]"),
         ("cov90_15s", "90 % HPD coverage at 15 s")),
        "baseline", "parameter sensitivity", "baseline")
    desc = vals["descriptions"]
    rows_txt = "; ".join(
        f"{vals['labels'][s]} ({s}): {desc.get(s, 'no description recorded')}"
        for s in PARAM_SETS if s in vals["sets"])
    runs = "; ".join(f"{s}: {os.path.basename(str(one(v.get('eval_runs'))))}"
                     for s, v in vals["sets"].items())
    ctx.caption = (
        "Simulator-parameter sensitivity of the planner checkpoint: the "
        "unchanged model evaluated on test sets regenerated with one "
        "perturbed parameter family each. Rows: "
        f"{rows_txt}. (a) minADE [m], (b) "
        "position-marginal mixture NLL, (c) coverage of the nominal 90 % HPD "
        "region at 15 s (dark line: nominal level); grey lines in (a) and "
        "(b): baseline value. Point estimates (report 'overall'); n per set "
        "on the axis. "
        + (f"Test runs: {runs}. " if runs else "")
        + (f"Not available: {', '.join(vals['missing'])}." if vals["missing"]
           else ""))


# ======================================================================
# 11. type-aware prediction: per-class confusion, error and calibration
# ======================================================================
TYPE_CONST_PY = f"{ROOT}/modelTraining/common/constants.py"
TYPE_NAMES_FALLBACK = ("Bird", "Multirotor", "FixedWingUAS", "eVTOL", "GA",
                       "Debris", "Unknown")
TYPE_DISPLAY = {"Bird": "Bird", "Multirotor": "Multirotor",
                "FixedWingUAS": "Fixed-wing UAS", "eVTOL": "eVTOL",
                "GA": "General aviation", "Debris": "Debris",
                "Unknown": "Unknown"}
TYPE_DISPLAY_2L = {"FixedWingUAS": "Fixed-wing\nUAS", "GA": "General\naviation"}
PC_SPLITS = (("testid", "test_id", "ID"), ("testood", "test_ood", "OOD"))
PC_BASELINES = (("cv-obs", "B: CV, observed track", ARM_INK["cv"],
                 ARM_FILL["cv"], ARM_MK["cv"]),
                ("imm-gapfix", "IMM (CV / CT / CA)", IMM_COL, IMM_COL, IMM_MK),
                ("cv-gt", "CV, true state (oracle)", GT_COL,
                 "white", GT_MK))
PC_B, PC_SEED = 2000, 20260926
CM_CMAP = LinearSegmentedColormap.from_list(
    "prism_seq", ["#ffffff", "#e1bee7", "#ab47bc", "#4a148c"])


def type_names(ctx: Ctx) -> list:
    """TYPE_NAMES in the order of modelTraining/common/constants.py (parsed
    from the source text, so no training-side import is needed)."""
    p = norm(TYPE_CONST_PY)
    try:
        with open(p, encoding="utf-8") as f:
            m = re.search(r"^TYPE_NAMES\s*=\s*\[([^\]]*)\]", f.read(), re.M)
        names = [s.strip().strip("\"'") for s in m.group(1).split(",")
                 if s.strip()]
        ctx.use(p)
        return names
    except (OSError, AttributeError):
        ctx.notes.append(f"TYPE_NAMES not parsed from {p}; fallback order used")
        return list(TYPE_NAMES_FALLBACK)


def load_persample(ctx: Ctx, path: str):
    if not ctx.use(path):
        return None
    z = np.load(path, allow_pickle=True)
    keys = ("file_idx", "type_label", "type_pred", "minADE", "NLL", "cov",
            "files", "cov_levels", "cov_horizons_s")
    return {k: z[k] for k in keys if k in z.files}


def cluster_ci(vals, groups, n_groups: int, B: int = PC_B, seed: int = PC_SEED):
    """Mean of each column of `vals` with a 95 % percentile interval from an
    episode-cluster bootstrap: episodes (groups) are resampled with
    replacement B times, every resample re-weights the per-episode sums."""
    vals = np.asarray(vals, float)
    if vals.ndim == 1:
        vals = vals[:, None]
    S = np.zeros((n_groups, vals.shape[1]))
    np.add.at(S, groups, vals)
    C = np.bincount(groups, minlength=n_groups).astype(float)
    rng = np.random.default_rng(seed)
    W = np.empty((B, n_groups))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n_groups, n_groups),
                           minlength=n_groups)
    est = (W @ S) / (W @ C)[:, None]
    return (vals.mean(0), np.percentile(est, 2.5, axis=0),
            np.percentile(est, 97.5, axis=0))


def fig_per_class(ctx: Ctx) -> None:
    names = type_names(ctx)
    K = len(names)
    tag = f"r3_k3_s{PLANNER_SEED}"
    D: dict = {}
    for split, tset, short in PC_SPLITS:
        m = load_persample(ctx, f"{EV}/{tag}_{split}_persample.npz")
        rep = ctx.json(f"{EV}/{tag}_{split}_report.json")
        bl = {}
        for meth, *_ in PC_BASELINES:
            bl[meth] = {
                "ps": load_persample(
                    ctx, f"{EV}/persample_{meth}_{tset}_revised_current_idxcurrent.npz"),
                "rep": ctx.json(
                    f"{BASELINES}/report_{meth}_{tset}_revised_current_idxcurrent.json")}
        D[split] = {"model": m, "report": rep, "baselines": bl}

    # ---------------------------------------------------------- numbers
    vals: dict = {"type_names": names, "planner_seed": PLANNER_SEED,
                  "bootstrap": {"B": PC_B, "seed": PC_SEED,
                                "unit": "episode file (cluster)"},
                  "splits": {}}
    for split, tset, short in PC_SPLITS:
        m = D[split]["model"]
        if m is None:
            continue
        lab, prd = m["type_label"].astype(int), m["type_pred"].astype(int)
        nf = len(m["files"])
        hz = [int(h) for h in m["cov_horizons_s"]]
        lv = [float(x) for x in m["cov_levels"]]
        i90 = lv.index(max(lv))
        cm = np.zeros((K, K), int)
        np.add.at(cm, (lab, prd), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rn = np.where(cm.sum(1, keepdims=True) > 0,
                          cm / np.maximum(cm.sum(1, keepdims=True), 1), np.nan)
        V = {"n_samples": int(len(lab)), "n_episode_files": nf,
             "model": None,
             "accuracy": float((lab == prd).mean()),
             "classes_present": [names[i] for i in range(K) if cm[i].sum()],
             "classes_predicted": [names[j] for j in range(K) if cm[:, j].sum()],
             "confusion_counts": cm.tolist(),
             "confusion_row_normalised": np.round(rn, 4).tolist(),
             "cov_horizons_s": hz, "cov_levels": lv, "per_class": {}}
        if D[split]["report"]:
            V["model"] = D[split]["report"].get("model")
            V["report_type_acc"] = D[split]["report"]["overall"].get("type_acc")
        cov90 = m["cov"][:, i90, :].astype(float)          # [N, horizons]
        for i in range(K):
            sel = lab == i
            if not sel.any():
                continue
            cols = np.column_stack([m["minADE"][sel], m["NLL"][sel], cov90[sel]])
            mu, lo, hi = cluster_ci(cols, m["file_idx"][sel].astype(int), nf)
            top = int(np.argmax(cm[i]))
            pc = {"n": int(sel.sum()),
                  "minADE": {"mean": float(mu[0]), "ci95": [float(lo[0]), float(hi[0])]},
                  "NLL": {"mean": float(mu[1]), "ci95": [float(lo[1]), float(hi[1])]},
                  "cov90": {f"{h}s": {"mean": float(mu[2 + k]),
                                      "ci95": [float(lo[2 + k]), float(hi[2 + k])]}
                            for k, h in enumerate(hz)},
                  "predicted_as_most": names[top],
                  "predicted_as_share": float(rn[i, top]),
                  "correct_share": float(rn[i, i]),
                  "baselines": {}}
            for meth, *_ in PC_BASELINES:
                b = D[split]["baselines"][meth]
                ent = {"report_by_type": None, "persample": None}
                if b["rep"] and names[i] in (b["rep"].get("by_type") or {}):
                    bt = b["rep"]["by_type"][names[i]]
                    ent["report_by_type"] = {"minADE": bt.get("minADE"), "n": bt.get("n")}
                    ent["method_name"] = b["rep"].get("method")
                ps = b["ps"]
                if ps is not None:
                    sb = ps["type_label"].astype(int) == i
                    if sb.any():
                        c2 = np.column_stack([ps["minADE"][sb],
                                              ps["cov"][sb][:, i90, -1]])
                        mu2, lo2, hi2 = cluster_ci(c2, ps["file_idx"][sb].astype(int),
                                                   len(ps["files"]))
                        ent["persample"] = {
                            "n": int(sb.sum()),
                            "minADE": {"mean": float(mu2[0]),
                                       "ci95": [float(lo2[0]), float(hi2[0])]},
                            f"cov90_{hz[-1]}s": {"mean": float(mu2[1]),
                                                 "ci95": [float(lo2[1]), float(hi2[1])]}}
                        r = ent["report_by_type"]
                        if r and r["minADE"] is not None and\
                                abs(r["minADE"] - mu2[0]) > 0.01:
                            ctx.notes.append(
                                f"{split} {meth} {names[i]}: per-sample mean "
                                f"{mu2[0]:.3f} differs from report by_type "
                                f"{r['minADE']:.3f}")
                pc["baselines"][meth] = ent
            V["per_class"][names[i]] = pc
        vals["splits"][split] = V
    ctx.values = vals

    id_v, ood_v = vals["splits"].get("testid"), vals["splits"].get("testood")
    trained = list(id_v["classes_present"]) if id_v else []
    vals["trained_classes"] = trained
    vals["unseen_classes"] = ([c for c in names if c not in trained]
                              if id_v else [])

    # ---------------------------------------------------------- figure
    fig = plt.figure(figsize=(W_FULL, 4.9))
    gt = fig.add_gridspec(1, 3, left=0.095, right=0.975, top=0.865,
                          bottom=0.6, wspace=0.42, width_ratios=[1.2, 1.3, 1.3])
    gb = fig.add_gridspec(1, 1, left=0.095, right=0.975, top=0.415, bottom=0.135)
    ax_a, ax_b, ax_c = (fig.add_subplot(gt[0, k]) for k in range(3))
    ax_d = fig.add_subplot(gb[0, 0])
    methods = [("model", "C: PRISM", ARM_INK["model"], ARM_FILL["model"],
                ARM_MK["model"])] + list(PC_BASELINES)
    offs = np.linspace(-0.3, 0.3, len(methods))

    def method_style(col, fill, mkr, size=4.2):
        return dict(marker=mkr, ms=size, mfc=fill, mec=col, mew=0.7,
                    ls="none")

    def error_panel(ax, V, classes, ymax):
        for j, (meth, _lab, col, fill, mkr) in enumerate(methods):
            xs, ys, lo, hi = [], [], [], []
            for i, c in enumerate(classes):
                pc = V["per_class"].get(c)
                if pc is None:
                    continue
                if meth == "model":
                    e = pc["minADE"]
                else:
                    e = (pc["baselines"][meth]["persample"] or {}).get("minADE")
                    if e is None:                 # report point value only
                        r = pc["baselines"][meth]["report_by_type"]
                        if not r or r["minADE"] is None:
                            continue
                        e = {"mean": r["minADE"], "ci95": [r["minADE"], r["minADE"]]}
                xs.append(i + offs[j])
                ys.append(e["mean"])
                lo.append(e["ci95"][0])
                hi.append(e["ci95"][1])
            if not xs:
                continue
            ys, lo, hi = map(np.asarray, (ys, lo, hi))
            ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none",
                        ecolor=col, elinewidth=0.8, capsize=1.5, zorder=3)
            ax.plot(xs, ys, zorder=4, clip_on=False,
                    **method_style(col, fill, mkr))
        ax.set_xlim(-0.6, len(classes) - 0.4 + (0.25 if len(classes) > 4 else 0))
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(True, axis="y")
        ax.set_ylabel("minADE [m]")

    ymax = 1.0
    for V in (id_v, ood_v):
        if V:
            for pc in V["per_class"].values():
                ymax = max(ymax, pc["minADE"]["ci95"][1])
                for b in pc["baselines"].values():
                    e = (b["persample"] or {}).get("minADE")
                    r = b["report_by_type"]
                    ymax = max(ymax, e["ci95"][1] if e else 0.0,
                               (r or {}).get("minADE") or 0.0)
    ymax *= 1.08

    # (a) ID confusion matrix, trained classes only
    if id_v is None:
        pending_panel(ax_a, ctx, "ID confusion", [f"{EV}/{tag}_testid_persample.npz"])
    else:
        rows = [names.index(c) for c in id_v["classes_present"]]
        cols = sorted(set(rows) | {names.index(c) for c in id_v["classes_predicted"]})
        M = np.array(id_v["confusion_row_normalised"], float)[np.ix_(rows, cols)]
        ax_a.imshow(M, cmap=CM_CMAP, vmin=0, vmax=1, aspect="equal",
                    interpolation="nearest")
        for r in range(len(rows)):
            for c in range(len(cols)):
                v = M[r, c]
                ax_a.text(c, r, f"{100 * v:.1f} %", ha="center", va="center",
                          fontsize=FS_TXT, color="white" if v > 0.55 else INK)
        ax_a.set_xticks(range(len(cols)))
        ax_a.set_xticklabels([TYPE_DISPLAY_2L.get(names[j], TYPE_DISPLAY[names[j]])
                              .replace("Multirotor", "Multi-\nrotor")
                              for j in cols], fontsize=FS_MIN)
        ax_a.set_yticks(range(len(rows)))
        ax_a.set_yticklabels(
            [f"{TYPE_DISPLAY_2L.get(names[i], TYPE_DISPLAY[names[i]])}"
             for i in rows], fontsize=FS_MIN)
        ax_a.tick_params(length=0)
        ax_a.set_xlabel("predicted class", labelpad=2)
        ax_a.set_ylabel("true class", labelpad=2)
        for sp in ax_a.spines.values():
            sp.set_visible(False)
    panel_title(ax_a, "(a) Class from 10 s of track")

    # (b) ID error by class
    if id_v is None:
        pending_panel(ax_b, ctx, "ID error by class", [f"{EV}/{tag}_testid_persample.npz"])
    else:
        cls = id_v["classes_present"]
        error_panel(ax_b, id_v, cls, ymax)
        ax_b.set_xticks(range(len(cls)))
        ax_b.set_xticklabels([TYPE_DISPLAY_2L.get(c, TYPE_DISPLAY[c]) for c in cls],
                             fontsize=FS_MIN)
    panel_title(ax_b, "(b) Error by class")

    # (c) ID calibration by class: 90 % coverage at each horizon
    if id_v is None:
        pending_panel(ax_c, ctx, "ID calibration by class",
                      [f"{EV}/{tag}_testid_persample.npz"])
    else:
        cls = id_v["classes_present"]
        hz = id_v["cov_horizons_s"]
        L = max(id_v["cov_levels"])
        ax_c.axhline(L, color=REF, lw=0.7, zorder=1)
        hoff = np.linspace(-0.25, 0.25, len(hz))
        xt, xl = [], []
        lo_all, hi_all = L, L
        for i, c in enumerate(cls):
            pc = id_v["per_class"][c]
            xs = i + hoff
            ys = np.array([pc["cov90"][f"{h}s"]["mean"] for h in hz])
            lo = np.array([pc["cov90"][f"{h}s"]["ci95"][0] for h in hz])
            hi = np.array([pc["cov90"][f"{h}s"]["ci95"][1] for h in hz])
            lo_all, hi_all = min(lo_all, lo.min()), max(hi_all, hi.max())
            ax_c.plot(xs, ys, color=ARM_INK["model"], lw=0.8, zorder=3)
            ax_c.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none",
                          ecolor=ARM_INK["model"], elinewidth=0.8, capsize=1.5,
                          zorder=3)
            ax_c.plot(xs, ys, zorder=4, **method_style(ARM_INK["model"],
                                                       ARM_FILL["model"], "D", 4.0))
            xt += list(xs)
            xl += [f"{h}" for h in hz]
            ax_c.text(i, 0.96, TYPE_DISPLAY_2L.get(c, TYPE_DISPLAY[c]),
                      transform=xtrans(ax_c), ha="center", va="top",
                      fontsize=FS_MIN, color="#404040", linespacing=1.05)
            if i:
                ax_c.axvline(i - 0.5, color=SEP, lw=0.5, zorder=1)
        pad = max(0.02, (hi_all - lo_all) * 0.6)
        ax_c.set_ylim(min(lo_all, L) - pad, max(hi_all, L) + 2.2 * pad)
        ax_c.set_xlim(-0.6, len(cls) - 0.4)
        ax_c.set_xticks(xt)
        ax_c.set_xticklabels(xl, fontsize=FS_MIN)
        ax_c.tick_params(axis="x", pad=1.5)
        ax_c.grid(True, axis="y")
        ax_c.yaxis.set_major_locator(MaxNLocator(5))
        ax_c.set_ylabel(f"{L * 100:.0f} % HPD coverage")
        ax_c.set_xlabel("forecast horizon [s]", labelpad=1.5)
    panel_title(ax_c, "(c) Calibration by class")

    # (d) OOD error by class, unseen classes marked
    if ood_v is None:
        pending_panel(ax_d, ctx, "OOD error by class",
                      [f"{EV}/{tag}_testood_persample.npz"])
    else:
        cls = [c for c in names if c in trained and c in ood_v["per_class"]] +\
              [c for c in names if c not in trained and c in ood_v["per_class"]]
        n_tr = sum(1 for c in cls if c in trained)
        if n_tr < len(cls):
            ax_d.axvspan(n_tr - 0.5, len(cls) - 0.5, color="#f1f1f1", lw=0,
                         zorder=0)
            ax_d.text(n_tr - 0.5 + 0.06, 0.97, "classes not in the training set",
                      transform=xtrans(ax_d), ha="left", va="top",
                      fontsize=FS_TXT, color="#404040", style="italic")
            ax_d.text(n_tr - 0.5 - 0.06, 0.97, "trained classes",
                      transform=xtrans(ax_d), ha="right", va="top",
                      fontsize=FS_TXT, color="#404040", style="italic")
        error_panel(ax_d, ood_v, cls, ymax)
        labels = []
        for c in cls:
            base = TYPE_DISPLAY_2L.get(c, TYPE_DISPLAY[c])
            if c not in trained:
                pc = ood_v["per_class"][c]
                base += (f"\n{ARROW} {TYPE_DISPLAY[pc['predicted_as_most']]} "
                         f"{100 * pc['predicted_as_share']:.0f} %")
            labels.append(base)
        ax_d.set_xticks(range(len(cls)))
        ax_d.set_xticklabels(labels, fontsize=FS_MIN, linespacing=1.1)
        vals["ood_class_order"] = cls
    panel_title(ax_d, "(d) OOD error by class")

    hl = [Line2D([], [], **method_style(col, fill, mkr)) for _m, _l, col, fill, mkr
          in methods]
    hl.append(Line2D([], [], color=REF, lw=0.7))
    fig.legend(hl, [lab for _m, lab, *_ in methods] + ["nominal level"],
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.535, 0.925), columnspacing=1.4,
               handlelength=1.3, handletextpad=0.4, borderaxespad=0.0)
    save(fig, ctx)

    # ---------------------------------------------------------- caption
    if id_v is None or ood_v is None:
        ctx.caption = "Per-class figure: per-sample arrays not found (see manifest)."
        return

    def pcs(V, c):
        return V["per_class"][c]

    def mth(V, c, meth):
        b = pcs(V, c)["baselines"][meth]
        e = (b["persample"] or {}).get("minADE")
        return e["mean"] if e else (b["report_by_type"] or {}).get("minADE")

    hz = id_v["cov_horizons_s"]
    unseen = [c for c in vals["ood_class_order"] if c not in trained]
    diag = ", ".join(f"{TYPE_DISPLAY[c]} {100 * pcs(id_v, c)['correct_share']:.1f} %"
                     for c in trained)
    absorb = ", ".join(
        f"{TYPE_DISPLAY[c]} {ARROW} {TYPE_DISPLAY[pcs(ood_v, c)['predicted_as_most']]} "
        f"({100 * pcs(ood_v, c)['predicted_as_share']:.0f} %)" for c in unseen)
    ood_err = sorted(((pcs(ood_v, c)["minADE"]["mean"], c) for c in unseen),
                     reverse=True)
    worst = ", ".join(f"{TYPE_DISPLAY[c]} {e:.1f} m" for e, c in ood_err[:2])
    easiest_e, easiest_c = ood_err[-1]
    per_id = "; ".join(
        f"{TYPE_DISPLAY[c]} (n = {fmt_n(pcs(id_v, c)['n'])}): PRISM "
        f"{pcs(id_v, c)['minADE']['mean']:.1f} m "
        f"[{pcs(id_v, c)['minADE']['ci95'][0]:.1f}, {pcs(id_v, c)['minADE']['ci95'][1]:.1f}], "
        f"CV {fmt(mth(id_v, c, 'cv-obs'), 1)} m, IMM {fmt(mth(id_v, c, 'imm-gapfix'), 1)} m, "
        f"CV on the true state {fmt(mth(id_v, c, 'cv-gt'), 1)} m; 90 % coverage at "
        + "/".join(str(h) for h in hz) + " s = "
        + "/".join(f"{pcs(id_v, c)['cov90'][f'{h}s']['mean']:.3f}" for h in hz)
        for c in trained)
    per_ood = "; ".join(
        f"{TYPE_DISPLAY[c]} (n = {fmt_n(pcs(ood_v, c)['n'])}"
        + ("" if c in trained else ", unseen") + f"): PRISM "
        f"{pcs(ood_v, c)['minADE']['mean']:.1f} m "
        f"[{pcs(ood_v, c)['minADE']['ci95'][0]:.1f}, {pcs(ood_v, c)['minADE']['ci95'][1]:.1f}], "
        f"CV {fmt(mth(ood_v, c, 'cv-obs'), 1)} m, IMM {fmt(mth(ood_v, c, 'imm-gapfix'), 1)} m, "
        f"CV on the true state {fmt(mth(ood_v, c, 'cv-gt'), 1)} m, 90 % coverage at "
        f"{hz[-1]} s {pcs(ood_v, c)['cov90'][f'{hz[-1]}s']['mean']:.3f}"
        for c in vals["ood_class_order"])
    ctx.caption = (
        "Type-aware prediction. From 10 s of observed track alone the network "
        f"recovers the object class ((a): ID accuracy {id_v['accuracy']:.3f}; "
        f"correct share per true class {diag}), its forecast error is "
        "class-specific and below both online baselines (CV, IMM) in every "
        "class (b), and its 90 % "
        "HPD coverage stays near nominal in every trained class (c); on the OOD "
        "set (d) the classes absent from training are absorbed into the "
        f"nearest trained class ({absorb}) and the largest OOD errors fall on "
        f"{worst}, while {TYPE_DISPLAY[easiest_c]} ({easiest_e:.1f} m) is the "
        "easiest class of all. "
        f"Planner checkpoint {job_name(id_v.get('model')) or 'n/a'} (seed "
        f"{PLANNER_SEED}); current-track test sets, ID n = {fmt_n(id_v['n_samples'])} "
        f"samples from {fmt_n(id_v['n_episode_files'])} episodes, OOD n = "
        f"{fmt_n(ood_v['n_samples'])} from {fmt_n(ood_v['n_episode_files'])} "
        "episodes. (a) Row-normalised confusion of the auxiliary type head "
        "(argmax over all seven classes; rows = true class, columns = "
        "predicted class; only trained classes are ever predicted, so the OOD "
        f"accuracy over seven classes is {ood_v['accuracy']:.3f}). (b, d) minADE "
        "per true class for PRISM (C), CV extrapolation of the observed track "
        "(B), an IMM filter (CV / CT / CA) and CV extrapolation of the true "
        "state (an information upper bound not available to any online "
        "method); bars: 95 % percentile intervals of an episode-cluster "
        f"bootstrap (B = {fmt_n(PC_B)} resamples of episodes, seed {PC_SEED}); "
        "the baseline point values equal the by_type field of the baseline "
        "reports. (c) Coverage of the nominal 90 % HPD region at "
        + ", ".join(str(h) for h in hz) + " s per class (Monte-Carlo HPD "
        "membership of the evaluation), line: nominal; same bootstrap. In (d) "
        "the shaded classes were absent from training and the arrow gives the "
        f"trained class they are most often assigned to. ID per class: {per_id}. "
        f"OOD per class: {per_ood}.")


# ======================================================================
# registry, manifest, CLI
# ======================================================================
FIGURES = {
    "r3fig_planner_forest": fig_planner_forest,
    "r3fig_shared_geometry": fig_shared_geometry,
    "r3fig_sensitivity": fig_sensitivity,
    "r3fig_closed_loop_calibration": fig_closed_loop_calibration,
    "r3fig_closed_loop_calibration_main": fig_closed_loop_calibration_main,
    "r3fig_reacquisition": fig_reacquisition,
    "r3fig_rollout": fig_rollout,
    "r3fig_recal_transfer": fig_recal_transfer,
    "r3fig_training": fig_training,
    "r3fig_factor_ood": fig_factor_ood,
    "r3fig_param_sensitivity": fig_param_sensitivity,
    "r3fig_per_class": fig_per_class,
}


def resolve(only) -> list:
    if not only:
        return list(FIGURES)
    want = [w.strip() for x in only for w in x.split(",") if w.strip()]
    out = []
    for w in want:
        k = w if w.startswith(PREFIX) else PREFIX + w
        if k not in FIGURES:
            raise SystemExit(f"unknown figure {w!r}; choose from "
                             f"{[f[len(PREFIX):] for f in FIGURES]}")
        out.append(k)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def write_outputs(man: dict) -> None:
    man["generated_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    man["script"] = SCRIPT
    man["script_sha256"] = file_info(SCRIPT)["sha256"]
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(man), f, indent=1, ensure_ascii=False)
    os.replace(tmp, MANIFEST)
    lines = ["# R3 figure captions (drafts)", "",
             "Generated by `Paper/figures/make_r3_figs.py` from "
             "`r3fig_manifest.json`; do not edit by hand. Every number is "
             "read from the source files listed in the manifest.", ""]
    for i, name in enumerate(FIGURES, start=1):
        e = man["figures"].get(name)
        lines.append(f"## Figure R3-{i}: `{name}`")
        lines.append("")
        if not e:
            lines += ["Not built yet.", ""]
            continue
        pend = e.get("pending") or []
        st = e.get("status")
        lines.append(f"Status: **{st}** (built {e.get('generated_utc')})"
                     + (" — pending: " + "; ".join(p["item"] for p in pend)
                        if pend else ""))
        lines.append("")
        lines.append(e.get("caption") or "")
        lines.append("")
    tmp = CAPTIONS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, CAPTIONS)


def list_inputs() -> None:
    def show(p):
        print(f"   {'ok     ' if os.path.isfile(p) else 'MISSING'} {p}")
    print("planner aggregates")
    for k, _ in CORE:
        show(f"{PL}/core_{k}.json")
    show(f"{PL}/shared40_300.json")
    for s in SENS_ORDER:
        show(f"{PL}/sens300_{s}.json")
    iso = glob.glob(f"{PL}/*iso*.json")
    print(f"   {'ok     ' if iso else 'MISSING'} {PL}/*iso*.json {iso}")
    print("prediction")
    show(f"{R3}/recal_r3_k3_s1.json")
    for fn, _ in ROLL:
        show(f"{R3}/{fn}")
    for p in sorted(glob.glob(f"{R3}/rollout_cond_*.json")):
        show(p)
    print(f"   P8 newest: {newest(f'{R3}/pred_p5_p6_p8*.json')}")
    for split in ("testid", "testood"):
        for s in SEEDS:
            show(f"{EV}/r3_k3_s{s}_{split}_report.json")
    show(f"{EV}/s4_testid_indexcurrent_report.json")
    show(f"{EV}/s4_testood_indexcurrent_report.json")
    for s in FACTOR_SETS:
        show(f"{EV}/r3_k3_s{PLANNER_SEED}_factor_{s}_report.json")
    for s in PARAM_SETS:
        show(f"{EV}/r3_k3_s{PLANNER_SEED}_sens_{s}_report.json")
    print("per class")
    for split, tset, _ in PC_SPLITS:
        show(f"{EV}/r3_k3_s{PLANNER_SEED}_{split}_persample.npz")
        for meth, *_ in PC_BASELINES:
            show(f"{EV}/persample_{meth}_{tset}_revised_current_idxcurrent.npz")
            show(f"{BASELINES}/report_{meth}_{tset}_revised_current_idxcurrent.json")
    print("training jobs (completed)")
    for t in [f"r3_k3_s{s}" for s in SEEDS] + [a[0] for a in ABLATIONS[1:]]\
            + [ONE_X_TAG]:
        print(f"   {t:18s} {find_job(t) or 'MISSING'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="figure names (with or without the r3fig_ prefix), "
                         "comma- or space-separated")
    ap.add_argument("--list", action="store_true",
                    help="report which inputs are present and exit")
    args = ap.parse_args()
    if args.list:
        list_inputs()
        return 0
    names = resolve(args.only)
    os.makedirs(OUT, exist_ok=True)
    man = {"figures": {}}
    if os.path.isfile(MANIFEST):
        try:
            with open(MANIFEST, encoding="utf-8") as f:
                man = json.load(f)
            man.setdefault("figures", {})
        except (OSError, json.JSONDecodeError):
            man = {"figures": {}}
    failed = []
    for name in names:
        print(f"building {name}")
        ctx = Ctx(name)
        try:
            FIGURES[name](ctx)
        except Exception:                                    # noqa: BLE001
            ctx.error = traceback.format_exc()
            print(ctx.error)
            failed.append(name)
            plt.close("all")
        entry = {
            "status": ctx.status(),
            "generated_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "files": ctx.files,
            "size_cm": ctx.size_cm,
            "size_px": ctx.size_px,
            "inputs": ctx.inputs,
            "missing_inputs": ctx.missing,
            "pending": ctx.pending,
            "notes": ctx.notes,
            "values": ctx.values,
            "caption": ctx.caption,
        }
        if ctx.error:
            entry["error"] = ctx.error
        man["figures"][name] = entry
        tag = ctx.status()
        extra = (f"; pending: {'; '.join(p['item'] for p in ctx.pending)}"
                 if ctx.pending else "")
        print(f"  -> {tag}{extra}; size {ctx.size_cm} cm, {ctx.size_px} px")
        for nt in ctx.notes:
            print(f"  note: {nt}")
    man["figures"] = {k: man["figures"][k] for k in FIGURES
                      if k in man["figures"]}
    write_outputs(man)
    print(f"manifest -> {MANIFEST}\ncaptions -> {CAPTIONS}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
