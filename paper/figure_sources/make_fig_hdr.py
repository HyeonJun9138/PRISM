# -*- coding: utf-8 -*-
"""PRISM fig_hdr v2: the predictive position distribution as a time-swept
highest predictive density (HPD) map.

Versions
  v1  2026-09-26, frozen byte-identical as make_fig_hdr_v1.py (SHA-256
      a2e3f3a9...b3469, recorded in figures_r3/fig_hdr_manifest.json); its
      outputs carry no suffix. Swept map over the 30 forecast steps, panels
      a-d.
  v2  this file. Outputs carry --suffix (default "_v2"); v1 files are kept.
      Applied: HPD terminology as in the manuscript; display-only
      interpolation of the swept map every 0.1 s (--no-interp gives v1's
      sweep); per-time 90 % HPD contours at 5/10/15 s with the rollout
      positions at those times on the rollout panels, swept map drawn
      lighter there; 15 s rollout endpoints as dots; a second rule-based
      rollout case from the HARD classes (panels e, f); per-time inclusion
      counts in the manifest; one more validation (interpolated covariance).
      Rejected / deferred: none.

Why. make_fig_cases.py draws each branch as ONE "1-sigma ellipse at 15 s". In
2-D a 1-sigma ellipse holds only 39 % of the mass (1 - e^-1/2), and a single
endpoint ellipse hides the predicted corridor (REVISION_PLAN_round1_v3.md
section 6; R3 minor 1, R3 major 5).

Method
  1. Forecast step t = 0.5 .. 15 s: p_t(x) = sum_k pi_k N(x; mu_kt, Sigma_kt),
     full covariance with rho. The network works in the target frame, so
     Sigma_world = R Sigma_frame R^T with R = [[c,-s],[s,c]] (theta = frame
     angle): the SAME frame->world matrix predictor.py applies to mu.
  2. HPD level L_t(x) = mass of {p_t > p_t(x)}. Per step, a fine grid covers
     every branch (union of mu_k +- 5 sigma_k, 16 cells per smallest principal
     sigma, lattice shifted by an irrational sub-cell phase, see PHASE); cell
     densities are sorted descending and cumulated times the cell area. The
     cumulative sum is NOT renormalised by the grid total (the analytic total
     is 1, so mass outside the grid counts as the lowest densities) and the
     grid must hold >= 99.5 % (asserted). L_t depends on x only through
     p_t(x), so the sorted density -> level table transfers exactly to the
     display grid, where p_t is evaluated in closed form (only inside each
     step's +-5 sigma box; outside it L_t > 0.9999 and the cell stays blank).
  3. Swept map L(x) = min_t L_t(x): a cell is inside the 90 % region at SOME
     time iff L(x) <= 0.9. Single-hue sequential colour (dark = low L), blank
     where L > 0.95.
  4. Display interpolation (swept map ONLY): the map is evaluated every
     SUBSTEP_S = 0.1 s. Between two forecast nodes each branch's mean and each
     covariance ENTRY are interpolated linearly (never sigma and rho
     separately); a convex combination of two positive-definite matrices is
     positive definite (checked in validation [10]). Node 0 (t = 0) is the
     last detection at t0 with covariance NODE0_COV_SCALE = 0.5 x the first
     forecast step's covariance: a stated display choice, the model has no
     t = 0 output. Branch weights pi_k are constant over the horizon.
     Coverage numbers, per-time contours and every comparison with rollouts
     use the forecast steps themselves. --no-interp sweeps the 30 forecast
     steps only (v1).
  Validation ([1]..[10]) is written to <out>/hdr_validation<suffix>.txt; any
  failed check aborts before a figure is drawn.

Panels
  (a) PRISM, (b) CV: the case given by --case-* (default: the sample of the
      published fig_cases panel (a); see case_notes()).
  (c) PRISM, (d) CV: rollout group chosen by ROLLOUT_RULE.
  (e) PRISM, (f) CV: rollout group chosen by the same rule restricted to the
      HARD classes (HARD_RULE) of the test_ood rollout run; dropped with a
      note if no group passes (the rule is not relaxed).
  On rollout panels the swept map is a light background; thin labelled
  lines are the per-time 90 % HPD contours at 5/10/15 s, and grey markers
  the 60 rollout positions at those times (triangle / square / dot).
  CV = the planner's CV arm (modelVal/planner/risk.py CVRisk on the open
  track's detections): least-squares velocity over the last 4 s of
  detections, anchored at the detection at t0, sigma(t) = sqrt(15^2+(4t)^2)
  isotropic.
  Rollout timing: logged index i is t = (i+1) * 0.5 s; rollouts fork at
  origin_step * 0.5 s = 30.0 s, so the history ends at index origin_step - 1
  and rollout step k is t = 30.0 + 0.5 (k+1) s = model step k.

Outputs (in --out, default D:/PRISM/Paper/figures_r3; <s> = --suffix)
  fig_hdr<s>.png/.pdf                 combined figure (3 x 2, or 2 x 2)
  fig_hdr_panel_{a..f}<s>.png/.pdf    stand-alone panels, identical size
  fig_hdr_legend<s>.png/.pdf          legend + colour key only
  hdr_validation<s>.txt               validation numbers
  caption_hdr<s>.txt                  draft caption (EN + KR) and notes
  fig_hdr_manifest<s>.json            sources, SHA-256, rules, counts, values

CPU only (the GPU is shared). About a minute end to end.

Run:  python D:/PRISM/Paper/figures/make_fig_hdr.py
      [--model <final/model_final.pt>] [--out <dir>] [--suffix _v2]
      [--no-interp] [--case-run <run> --case-ep N --case-t0 N --case-slot N]
"""
from __future__ import annotations

import os

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import deque
from datetime import datetime, timezone

import numpy as np

ROOT = (os.environ["PRISM_ROOT"] + '')
sys.path.insert(0, f"{ROOT}/modelVal")
import bridge  # noqa: E402,F401  -- puts dataGen + modelTraining on sys.path

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

import torch  # noqa: E402

from common.constants import (HIST_STEPS, INTENT_NAMES, LOG_DT,  # noqa: E402
                              POS_SCALE, PRED_STEPS, TYPE_NAMES)
from data.dataset import (build_sample, track_rule,  # noqa: E402
                          window_detections)
from planner.risk import CVRisk  # noqa: E402
from runtime.predictor import LivePredictor  # noqa: E402
from simulation.rollout import FROZEN_SPREAD_M  # noqa: E402  (dataGen, 1 m)

# ------------------------------------------------------------------ inputs
MODEL_DEFAULT = f"{ROOT}/Logs/Models/20260813_170429_full5x_s4/final/model_final.pt"
OUT_DEFAULT = f"{ROOT}/Paper/figures_r3"
RUN_A = f"{ROOT}/Logs/Dataset/20260813_023139_test_ood_full5x_a"
CASE_DEFAULT = (RUN_A, 29, 31, 1)          # published fig_cases panel (a)
# the three published fig_cases samples (fig_manifest.json, cache meta)
PUBLISHED = {"a": (RUN_A, 29, 31, 1), "b": (RUN_A, 1, 79, 1),
             "c": (RUN_A, 9, 19, 4)}
RUN_ID = f"{ROOT}/Logs/Dataset/20260814_101540_test_id_rollout_eval"
RUN_OOD = f"{ROOT}/Logs/Dataset/20260814_101541_test_ood_rollout_eval"
ROLLOUT_RUNS = (RUN_ID, RUN_OOD)
EP_KEYS = ("t", "gt_pos", "gt_valid", "gt_type", "gt_intent", "gt_mode",
           "obs_detected", "obs_pos", "obs_R", "obs_track_age", "own_pos",
           "obs_track_id")

# ---------------------------------------------------------- HPD parameters
N_SIG_BOX = 5.0          # fine-grid half width per branch [sigma, per axis]
CELLS_PER_SIGMA = 16     # fine-grid cells per smallest principal sigma
FINE_MAX_SIDE = 3000     # cap on fine-grid cells per side (warned if hit)
# Irrational sub-cell phase of the fine lattice. Without it a Gaussian centre
# sits exactly on a lattice corner (mu - lo = 5 sigma = whole cells), and the
# lattice-point count of a symmetric region errs coherently (Gauss circle
# problem): +1.9 % area at 8 cells/sigma in the first v1 validation run.
PHASE = np.array([0.6180339887, 0.4142135624])
MASS_MIN = 0.995         # a grid must hold at least this much of p_t
L_BLANK = 0.95           # cells with L above this are left blank
LEVELS = (0.5, 0.9)      # swept-map contour levels (panels a, b)
DISPLAY_N = 720          # display-grid cells per side
PROVISIONAL_N = 300      # coarse pass that fixes the view
AREA_TOL = 0.02          # validation tolerance on HPD areas
SUBSTEP_S = 0.1          # display interpolation step of the swept map [s]
NODE0_COV_SCALE = 0.5    # node 0 covariance = 0.5 x first-step covariance
PER_TIME = ((9, "5 s"), (19, "10 s"), (29, "15 s"))   # node index, label
PER_TIME_LEVEL = 0.9
# one line pattern per time (overlapping regions stay traceable), matching
# the one marker style per time used for the rollout positions
PER_TIME_DASH = {9: (0, (1.1, 1.1)), 19: (0, (3.2, 1.4)), 29: "solid"}

# --------------------------------------------------- case-selection rules
MIN_OBS_SELECT = 3       # open-track detections in the 10 s window
CV_WINDOW = 8            # CVRisk default: 8 x 0.5 s = 4 s
SEP_FACTOR = 3.0         # 2-means separation > 3 x mean within-cluster std
HARD_TYPES = (0, 6)      # Bird, Unknown  (TYPE_NAMES order)
HARD_INTENTS = (3,)      # SearchWander   (INTENT_NAMES order)
ROLLOUT_RULE = (
    "Pool all rollout groups of both rollout-evaluation runs (test_id and "
    "test_ood, 100 episodes each; branch at 30.0 s). Keep groups whose target "
    "(i) is alive at 30.0 s, with all 60 rollouts and the logged future valid "
    "over the whole 15 s horizon; (ii) has >= 3 detections of its currently "
    "open track in the 10 s history (eligible under both the current-track "
    "and the slot rule) and is detected at 30.0 s with >= 2 open-track "
    "detections in the last 4 s (the planner's CV arm can extrapolate it); "
    "(iii) is not frozen: 15 s rollout spread > 1 m (dataGen "
    "FROZEN_SPREAD_M, a point mass below it). "
    "Split the 60 rollout endpoints at 15 s by 2-means (deterministic: best "
    "SSE of a principal-axis split and a farthest-pair split, Lloyd "
    "iterations); keep the group if both clusters hold >= 2 endpoints and the "
    "centroid separation exceeds 3 x the mean within-cluster std, where a "
    "cluster's std is the RMS distance of its endpoints from its centroid "
    "(sqrt(trace(cov)), ddof = 1). Among the kept groups pick the one with "
    "the median 15 s rollout spread (sqrt(trace(cov)) of all 60 endpoints, "
    "dataGen's rollout_statistics definition; lower median for an even "
    "count). No model output enters the selection.")
HARD_RULE = (
    "ROLLOUT_RULE applied to the test_ood rollout run only, after keeping "
    "the groups of the HARD classes: target platform Bird (gt_type 0) or "
    "Unknown (gt_type 6), or intent SearchWander (gt_intent 3) at the branch "
    "step. Lower median of the 15 s rollout spread. Not relaxed if empty.")

# ------------------------------------------------------------------- style
# identical to make_fig_cases.py so the new panels sit with the old ones
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.2,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.6,
    "axes.linewidth": 0.5, "axes.edgecolor": "#8c8c8c",
    "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#8c8c8c", "ytick.color": "#8c8c8c",
    "xtick.labelcolor": "#404040", "ytick.labelcolor": "#404040",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
    "grid.linewidth": 0.4, "grid.color": "#e8e8e8", "grid.alpha": 1.0,
    "axes.axisbelow": True, "savefig.dpi": 300,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# ColorBrewer "Blues" (single hue, lightness monotone -> survives greyscale
# print and every colour-vision deficiency; checked in validation), truncated
# so the pale end still reads against white and the dark end leaves room for
# the black mean paths.
_BLUES = plt.get_cmap("Blues")
CMAP = LinearSegmentedColormap.from_list(
    "prism_hpd", [_BLUES(v) for v in np.linspace(0.90, 0.16, 256)])
NORM = Normalize(0.0, L_BLANK)
INK, OBS = "#1a1a1a", "#242424"
CONTOUR_INK = "#0e2f52"          # swept contours (a, b): dark navy
PER_TIME_INK = "#262626"         # per-time 90 % contours (rollout panels)
TRUE_FUT = "#D55E00"             # Okabe-Ito vermillion, dashed
ROLL = "#4d4d4d"
ROLL_BG_ALPHA = 0.45             # swept map opacity on the rollout panels
ROLL_MARKERS = {9: dict(marker="^", ms=2.3, mfc="none", mew=0.4),
                19: dict(marker="s", ms=2.0, mfc="none", mew=0.4),
                29: dict(marker="o", ms=1.7, mfc=ROLL, mew=0.0)}
LW_PER_PI = 2.0                  # mean-path width [pt] = LW_PER_PI * pi_k
HALO_PT = 0.8                    # white halo added around mean paths [pt]
DASH = (0, (3.4, 1.9))

PANEL_FIGSIZE = (2.36, 2.30)
PANEL_MARGINS = dict(left=0.233, right=0.974, bottom=0.157, top=0.917)
TYPE_LABEL = {"Bird": "Bird", "Multirotor": "Multirotor",
              "FixedWingUAS": "Fixed-wing UAS", "eVTOL": "eVTOL",
              "GA": "General aviation", "Debris": "Debris",
              "Unknown": "Unknown"}
INTENT_LABEL = {"CorridorTransit": "corridor transit",
                "DeliveryWP": "delivery waypoints",
                "LoiterHover": "loiter/hover", "SearchWander": "search wander",
                "ReactiveAvoid": "reactive avoid",
                "PassiveDrift": "passive drift"}
# Korean terms as in manuscript_kr_v06.md section 3.2 / table 1
TYPE_KR = {"Bird": "조류", "Multirotor": "멀티로터",
           "FixedWingUAS": "고정익 무인기", "eVTOL": "eVTOL",
           "GA": "경항공기(GA)", "Debris": "부유 잔해", "Unknown": "미확인 기체"}
INTENT_KR = {"CorridorTransit": "회랑 순항", "DeliveryWP": "배달 경유점 추종",
             "LoiterHover": "체공 관측", "SearchWander": "탐색 배회",
             "ReactiveAvoid": "반응 회피", "PassiveDrift": "수동 표류"}


# ================================================================ HPD core
def cov_frame(sx: float, sy: float, rho: float) -> np.ndarray:
    return np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])


def frame_to_world(theta: float) -> np.ndarray:
    """R with v_world = R v_frame (predictor.py R_inv, build_sample's Rm^T)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def mixture_logpdf(X: np.ndarray, Y: np.ndarray, comps) -> np.ndarray:
    """log sum_k w_k N([X,Y]; mu_k, S_k), full 2x2 covariances."""
    out = None
    for w, mu, S in comps:
        if w <= 0.0:
            continue
        a, b, c = float(S[0, 0]), float(S[0, 1]), float(S[1, 1])
        det = a * c - b * b
        if not det > 0.0:
            raise ValueError(f"covariance not positive definite: {S}")
        dx = X - mu[0]
        dy = Y - mu[1]
        q = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det
        lg = math.log(w) - 0.5 * q - math.log(2.0 * math.pi * math.sqrt(det))
        out = lg if out is None else np.logaddexp(out, lg)
    return out


def comps_box(comps, nsig: float):
    lo = np.min([mu - nsig * np.sqrt(np.diag(S)) for _, mu, S in comps], 0)
    hi = np.max([mu + nsig * np.sqrt(np.diag(S)) for _, mu, S in comps], 0)
    return lo, hi


class StepHPD:
    """HPD level function of one mixture: L(x) = P[p(X) > p(x)]."""

    def __init__(self, comps):
        lo, hi = comps_box(comps, N_SIG_BOX)
        smin = min(math.sqrt(float(np.linalg.eigvalsh(S)[0]))
                   for _, _, S in comps)
        h = np.full(2, smin / CELLS_PER_SIGMA)
        n = np.ceil((hi - lo) / h).astype(int) + 1     # +1: room for PHASE
        self.capped = bool((n > FINE_MAX_SIDE).any())
        if self.capped:
            n = np.minimum(n, FINE_MAX_SIDE)
            h = np.maximum(h, (hi - lo) / (n - 1))
        n = np.maximum(n, 16)
        hx, hy = float(h[0]), float(h[1])
        # cells cover [lo - PHASE h, lo - PHASE h + n h] which contains [lo, hi]
        xs = lo[0] - PHASE[0] * hx + (np.arange(n[0]) + 0.5) * hx
        ys = lo[1] - PHASE[1] * hy + (np.arange(n[1]) + 0.5) * hy
        X, Y = np.meshgrid(xs, ys)
        lp = mixture_logpdf(X, Y, comps).ravel()
        lp_desc = np.sort(lp)[::-1]
        m = np.exp(lp_desc) * (hx * hy)
        cum = np.cumsum(m)
        self.mass = float(cum[-1])       # analytic total is 1: no renormalise
        if self.mass < MASS_MIN:
            raise AssertionError(f"fine grid holds {self.mass:.5f} < {MASS_MIN}")
        self.lp_asc = lp_desc[::-1].copy()
        self.L_asc = (cum - m)[::-1].copy()   # mass strictly above the cell
        self.cells = int(lp.size)
        self.h = (hx, hy)
        self.smin = smin
        self.cells_per_sigma = smin / max(hx, hy)

    def level(self, logp: np.ndarray) -> np.ndarray:
        return np.interp(logp, self.lp_asc, self.L_asc,
                         left=self.mass, right=0.0)


def make_grid(ext, n=DISPLAY_N) -> dict:
    x0, x1, y0, y1 = ext
    hx, hy = (x1 - x0) / n, (y1 - y0) / n
    xs = x0 + (np.arange(n) + 0.5) * hx
    ys = y0 + (np.arange(n) + 0.5) * hy
    X, Y = np.meshgrid(xs, ys)
    return {"X": X, "Y": Y, "xs": xs, "ys": ys, "A": hx * hy, "ext": ext}


def box_slice(comps, g):
    """Display-grid block inside the mixture's +-5 sigma box (None if empty).
    Outside it every branch is > 5 Mahalanobis units away, L_t > 0.9999."""
    lo, hi = comps_box(comps, N_SIG_BOX)
    i0 = int(np.searchsorted(g["xs"], lo[0]))
    i1 = int(np.searchsorted(g["xs"], hi[0], side="right"))
    j0 = int(np.searchsorted(g["ys"], lo[1]))
    j1 = int(np.searchsorted(g["ys"], hi[1], side="right"))
    if i1 <= i0 or j1 <= j0:
        return None
    return (slice(j0, j1), slice(i0, i1))


def step_levels(comps, table, g) -> np.ndarray:
    """L_t on the display grid (1 outside the step's box)."""
    L = np.ones_like(g["X"])
    sub = box_slice(comps, g)
    if sub is not None:
        L[sub] = table.level(mixture_logpdf(g["X"][sub], g["Y"][sub], comps))
    return L


def swept_levels(comps_list, tables, g) -> np.ndarray:
    """L(x) = min_t L_t(x) on a display grid."""
    L = np.ones_like(g["X"])
    for comps, tb in zip(comps_list, tables):
        sub = box_slice(comps, g)
        if sub is None:
            continue
        Lt = tb.level(mixture_logpdf(g["X"][sub], g["Y"][sub], comps))
        L[sub] = np.minimum(L[sub], Lt)
    return L


def direct_levels(logp: np.ndarray, cell_area: float):
    """HPD levels by sorting the given grid's own cells (cross check)."""
    lp = logp.ravel()
    order = np.argsort(-lp, kind="stable")
    m = np.exp(lp[order]) * cell_area
    cum = np.cumsum(m)
    L = np.empty_like(lp)
    L[order] = cum - m
    return L.reshape(logp.shape), float(cum[-1])


def label_components(mask: np.ndarray) -> int:
    """Connected components of a boolean grid, 8-connectivity (BFS)."""
    lab = np.zeros(mask.shape, np.int32)
    ny, nx = mask.shape
    n = 0
    for i0, j0 in zip(*np.nonzero(mask)):
        if lab[i0, j0]:
            continue
        n += 1
        lab[i0, j0] = n
        q = deque([(i0, j0)])
        while q:
            i, j = q.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < ny and 0 <= b < nx and mask[a, b]\
                            and not lab[a, b]:
                        lab[a, b] = n
                        q.append((a, b))
    return n


def contour_areas(X, Y, L, levels):
    """Area enclosed by each level's contour polygons (shoelace, summed)."""
    fig = plt.figure()
    cs = plt.contour(X, Y, L, levels=list(levels))
    out = {}
    for lev, segs in zip(levels, cs.allsegs):
        a = 0.0
        for s in segs:
            x, y = s[:, 0], s[:, 1]
            a += 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                                 - np.dot(y, np.roll(x, -1))))
        out[lev] = (a, len(segs))
    plt.close(fig)
    return out


# ============================================================= schedules
def interp_comps(ca, cb, f: float):
    """Branch-wise linear interpolation of mean and covariance ENTRIES."""
    out = []
    for (wa, ma, Sa), (wb, mb, Sb) in zip(ca, cb):
        if abs(wa - wb) > 1e-12:
            raise AssertionError("branch weights must be constant in time")
        out.append((wa, (1.0 - f) * ma + f * mb, (1.0 - f) * Sa + f * Sb))
    return out


def display_schedule(nodes, origin_rel, interp: bool):
    """Times and mixtures for the swept map: [(t, comps, node_index|None)].

    interp=False: the 30 forecast steps (v1). interp=True: node 0 at t = 0
    (last detection, NODE0_COV_SCALE x first-step covariance) plus every
    SUBSTEP_S between the 0.5 s nodes; nodes enter as the SAME objects."""
    if not interp:
        return [(LOG_DT * (j + 1), nodes[j], j) for j in range(len(nodes))]
    node0 = [(w, np.asarray(origin_rel, float).copy(), NODE0_COV_SCALE * S)
             for (w, _, S) in nodes[0]]
    n_sub = int(round(LOG_DT / SUBSTEP_S))
    sched = [(0.0, node0, None)]
    prev = node0
    for j in range(len(nodes)):
        for s in range(1, n_sub):
            sched.append((LOG_DT * j + s * SUBSTEP_S,
                          interp_comps(prev, nodes[j], s / n_sub), None))
        sched.append((LOG_DT * (j + 1), nodes[j], j))
        prev = nodes[j]
    return sched


# ================================================================ helpers
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episode(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in EP_KEYS if k in z.files}


def open_track_detections(ep: dict, t0: int, a: int, window: int):
    """[window] bool: detections of the track open at t0, ending at t0."""
    tid = ep["obs_track_id"][t0 - window + 1:t0 + 1, a]
    cur = int(ep["obs_track_id"][t0, a])
    det = ep["obs_detected"][t0 - window + 1:t0 + 1, a].astype(bool)
    return det & (tid == cur) & (cur >= 0), cur


def eligibility(ep: dict, t0: int, a: int) -> dict:
    """Model (current-track) and CV-arm eligibility of one target."""
    det_h, cur = open_track_detections(ep, t0, a, HIST_STEPS)
    det_cv, _ = open_track_detections(ep, t0, a, CV_WINDOW)
    return {"alive": bool(ep["gt_valid"][t0, a]),
            "n_open_track_10s": int(det_h.sum()),
            "n_slot_10s": int(ep["obs_detected"][t0 - HIST_STEPS + 1:t0 + 1, a]
                              .astype(bool).sum()),
            "detected_t0": bool(ep["obs_detected"][t0, a]),
            "track_id_t0": cur,
            "n_open_track_4s": int(det_cv.sum()),
            "future_valid": bool(ep["gt_valid"][t0 + 1:t0 + 1 + PRED_STEPS, a]
                                 .all())}


def eligible(e: dict) -> bool:
    return (e["alive"] and e["future_valid"]
            and e["n_open_track_10s"] >= MIN_OBS_SELECT
            and e["detected_t0"] and e["track_id_t0"] >= 0
            and e["n_open_track_4s"] >= 2)


def spread(P: np.ndarray) -> float:
    return float(np.sqrt(np.trace(np.cov(P.T, ddof=1))))


def kmeans2(P: np.ndarray):
    """Deterministic 2-means: best SSE of two fixed initialisations."""
    P = np.asarray(P, float)
    inits = []
    d = P - P.mean(0)
    _, V = np.linalg.eigh(d.T @ d)
    inits.append((d @ V[:, -1] > 0).astype(int))
    D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    i, j = np.unravel_index(int(np.argmax(D)), D.shape)
    inits.append((np.hypot(*(P - P[j]).T) < np.hypot(*(P - P[i]).T))
                 .astype(int))
    best = None
    for lab in inits:
        if lab.sum() in (0, len(P)):
            continue
        for _ in range(200):
            C = np.array([P[lab == k].mean(0) for k in (0, 1)])
            new = np.argmin(np.hypot(*(P[:, None] - C[None])
                                     .transpose(2, 0, 1)), 1)
            if (new == lab).all() or new.sum() in (0, len(P)):
                break
            lab = new
        C = np.array([P[lab == k].mean(0) for k in (0, 1)])
        sse = float(sum(((P[lab == k] - C[k]) ** 2).sum() for k in (0, 1)))
        if best is None or sse < best[0] - 1e-9:
            best = (sse, lab.copy(), C)
    return best


# ============================================================ forecasts
class Forecaster:
    def __init__(self, model_path: str):
        self.pred = LivePredictor()
        self.pred.device = torch.device("cpu")      # GPU is shared
        self.info = self.pred.load(model_path)
        self.data_cfg = dict(self.pred.data_cfg)

    def prism(self, ep: dict, t0: int, a: int) -> dict:
        res = self.pred.predict_frame(ep, t0)
        if a not in res:
            raise SystemExit(
                f"target slot {a} at t0={t0} is not eligible for this model "
                f"(track rule '{self.info['track_rule']}', min_obs "
                f"{self.data_cfg.get('min_obs', 3)})")
        r = res[a]
        # the frame-coordinate mean, straight from the network, to check that
        # the rotation used for Sigma reproduces predictor.py's mu_w
        s = build_sample(ep, t0, a, self.data_cfg, with_labels=False)
        with torch.no_grad():
            o = self.pred.model(torch.as_tensor(s["x"][None]),
                                torch.as_tensor(s["agent_mask"][None]))
        mu_f = o["mu"][0].float().numpy() * POS_SCALE
        R = frame_to_world(r["theta"])
        mu_chk = mu_f @ R.T + s["frame"][:2]
        return {"mu_w": r["mu_w"].astype(float), "sigma": r["sigma"].astype(float),
                "rho": r["rho"].astype(float), "pi": r["pi"].astype(float),
                "theta": float(r["theta"]), "origin": s["frame"][:2].copy(),
                "mu_f": mu_f, "rot_check_m": float(np.abs(mu_chk - r["mu_w"]).max())}


def prism_nodes(f: dict, org: np.ndarray):
    R = frame_to_world(f["theta"])
    K = len(f["pi"])
    return [[(float(f["pi"][k]), f["mu_w"][k, t] - org,
              R @ cov_frame(f["sigma"][k, t, 0], f["sigma"][k, t, 1],
                            f["rho"][k, t]) @ R.T)
             for k in range(K)] for t in range(PRED_STEPS)]


def cv_forecast(ep: dict, t0: int, a: int) -> dict:
    """The planner's CV arm (CVRisk) on the detections of the open track."""
    tid = ep["obs_track_id"]
    cur = tid[t0]
    det = ep["obs_detected"].astype(bool) & (tid == cur[None, :])\
        & (cur >= 0)[None, :]
    epm = dict(ep)
    epm["obs_detected"] = det
    risk = CVRisk()
    risk.update(epm, t0)
    slots = [s for s in range(det.shape[1]) if det[t0, s]]
    i = slots.index(a)
    ts = np.arange(1, PRED_STEPS + 1) * LOG_DT
    D = risk.disks_batch(ts)[:, i, :]
    j0 = max(0, t0 - risk.window_steps + 1)
    return {"mean": D[:, :2].astype(float), "sigma": D[:, 3].astype(float),
            "anchor": risk._pos[i].astype(float),
            "vel": risk._vel[i].astype(float),
            "n_fit": int(det[j0:t0 + 1, a].sum()),
            "sigma0": risk.sigma0, "growth": risk.sigma_growth,
            "window_s": risk.window_steps * LOG_DT}


def cv_nodes(cv: dict, org: np.ndarray):
    return [[(1.0, cv["mean"][t] - org, np.eye(2) * cv["sigma"][t] ** 2)]
            for t in range(PRED_STEPS)]


# ====================================================== case assembly
def build_case(fc: Forecaster, run: str, ep_idx: int, t0: int, a: int,
               interp: bool, rollout=None) -> dict:
    path = f"{run}/episodes/ep_{ep_idx:06d}.npz"
    ep = load_episode(path)
    e = eligibility(ep, t0, a)
    if not eligible(e):
        raise SystemExit(f"case {path} t0={t0} slot={a} fails eligibility: {e}")
    org = ep["gt_pos"][t0, a].astype(float)
    f = fc.prism(ep, t0, a)
    cv = cv_forecast(ep, t0, a)
    used = window_detections(ep, t0, track_rule(fc.data_cfg))[:, a]
    W = slice(t0 - HIST_STEPS + 1, t0 + 1)
    obs = (ep["obs_pos"][W, a].astype(float) - org)[used]
    fut = np.vstack([np.zeros((1, 2)),
                     ep["gt_pos"][t0 + 1:t0 + 1 + PRED_STEPS, a]
                     .astype(float) - org])
    c = {"path": path, "run": run, "ep": ep_idx, "t0": t0, "slot": a,
         "t_s": float(ep["t"][t0]), "org": org, "elig": e,
         "type": TYPE_NAMES[int(ep["gt_type"][a])],
         "intent": INTENT_NAMES[int(ep["gt_intent"][t0, a])],
         "prism": f, "cv": cv, "obs": obs, "fut": fut,
         "mu": f["mu_w"] - org, "cv_mean": cv["mean"] - org,
         "nodes_prism": prism_nodes(f, org), "nodes_cv": cv_nodes(cv, org),
         "n_used": int(used.sum()), "track_rule": track_rule(fc.data_cfg)}
    # node 0 of the display interpolation: the last detection (the model's
    # frame origin; also CVRisk's anchor, both the detection at t0)
    c["sched_prism"] = display_schedule(c["nodes_prism"], f["origin"] - org,
                                        interp)
    c["sched_cv"] = display_schedule(c["nodes_cv"], cv["anchor"] - org, interp)
    if rollout is not None:
        c["roll"] = np.concatenate(
            [np.zeros((rollout.shape[0], 1, 2)), rollout - org], axis=1)
    # plain facts about the input and the outcome, for the notes
    tids = sorted({int(v) for v in ep["obs_track_id"][W, a][used]})
    gspd = np.hypot(*np.diff(ep["gt_pos"][t0:t0 + 1 + PRED_STEPS, a]
                             .astype(float), axis=0).T) / LOG_DT
    c["facts"] = {
        "track_ids_in_input": tids,
        "branch_mean_speed_15s_mps": [
            round(float(np.hypot(*(f["mu_w"][k, -1] - f["origin"]))
                        / (PRED_STEPS * LOG_DT)), 1)
            for k in np.argsort(-f["pi"])],
        "true_speed_range_mps": [round(float(gspd.min()), 1),
                                 round(float(gspd.max()), 1)],
        "cv_speed_mps": round(float(np.hypot(*cv["vel"])), 1)}
    return c


def scan_rollout_groups(runs):
    """One pass over every rollout group: stage flags and cluster statistics
    for both selection rules (no model output involved)."""
    recs = []
    for run in runs:
        for epf in sorted(glob.glob(f"{run}/episodes/ep_*.npz")):
            rof = epf.replace("episodes", "rollouts").replace(".npz", "_rollouts.npz")
            if not os.path.exists(rof):
                continue
            ep = load_episode(epf)
            with np.load(rof) as z:
                origin = z["origin_step"].astype(int)
                slot = z["agent_slot"].astype(int)
                pos = z["pos"]
                valid = z["valid"].astype(bool)
            for g in range(len(origin)):
                t0 = int(origin[g]) - 1
                if abs(float(ep["t"][t0]) - origin[g] * LOG_DT) > 1e-6:
                    raise AssertionError(f"{epf}: t[{t0}] != branch time")
                a = int(slot[g])
                r = {"run": run, "episode": os.path.basename(epf),
                     "ep": int(os.path.basename(epf)[3:9]), "group": g,
                     "slot": a, "t0": t0, "origin_step": int(origin[g]),
                     "gt_type": int(ep["gt_type"][a]) if a >= 0 else -1,
                     "gt_intent": int(ep["gt_intent"][t0, a]) if a >= 0 else -1,
                     "alive": bool(a >= 0 and ep["gt_valid"][t0, a]),
                     "rollouts_valid": bool(valid[g].all()),
                     "eligible": False, "not_frozen": False,
                     "two_cluster": False}
                recs.append(r)
                if not (r["alive"] and r["rollouts_valid"]):
                    continue
                r["eligible"] = eligible(eligibility(ep, t0, a))
                if not r["eligible"]:
                    continue
                E = pos[g, :, PRED_STEPS - 1, :].astype(float)
                r["spread15_m"] = spread(E)
                r["not_frozen"] = r["spread15_m"] > FROZEN_SPREAD_M
                if not r["not_frozen"]:
                    continue
                sse, lab, C = kmeans2(E)
                nk = [int((lab == k).sum()) for k in (0, 1)]
                if min(nk) < 2:
                    continue
                sep = float(np.hypot(*(C[0] - C[1])))
                stds = [spread(E[lab == k]) for k in (0, 1)]
                r.update(separation_m=sep, cluster_std_m=stds, cluster_n=nk)
                r["two_cluster"] = sep > SEP_FACTOR * float(np.mean(stds))
    return recs


def apply_rule(recs, runs, hard: bool):
    """Stage counts and the lower-median-spread group (None if none pass)."""
    rs = [r for r in recs if r["run"] in runs]
    counts = {"groups": len(rs)}
    if hard:
        rs = [r for r in rs if r["gt_type"] in HARD_TYPES
              or r["gt_intent"] in HARD_INTENTS]
        counts["hard_class"] = len(rs)
    for stage in ("alive", "rollouts_valid", "eligible", "not_frozen",
                  "two_cluster"):
        rs = [r for r in rs if r[stage]]
        counts[stage] = len(rs)
    if not rs:
        return None, counts
    rs.sort(key=lambda r: (r["spread15_m"], r["run"], r["ep"], r["group"]))
    i = (len(rs) - 1) // 2
    choice = dict(rs[i])
    choice["rank"], choice["n_kept"] = i + 1, len(rs)
    choice["median_rank"] = f"{i + 1} of {len(rs)}"
    spreads = np.array([r["spread15_m"] for r in rs])
    choice["kept_spread_quartiles_m"] = [round(float(q), 1) for q in
                                         np.percentile(spreads, (25, 50, 75))]
    return choice, counts


def load_rollout(sel) -> np.ndarray:
    with np.load(f"{sel['run']}/rollouts/{sel['episode'][:-4]}_rollouts.npz") as z:
        assert int(z["origin_step"][sel["group"]]) - 1 == sel["t0"]
        assert int(z["agent_slot"][sel["group"]]) == sel["slot"]
        return z["pos"][sel["group"]].astype(float)          # [60,30,2]


# ============================================================= validation
def validate(cases, log):
    """Analytic checks of the HPD pipeline; raises on any failure."""
    ok = True

    def one_step(comps, pad=3.5):
        lo, hi = comps_box(comps, pad)
        ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2
        g = make_grid((ctr[0] - half, ctr[0] + half,
                       ctr[1] - half, ctr[1] + half))
        tb = StepHPD(comps)
        return g, tb, swept_levels([comps], [tb], g)

    def area_test(name, comps):
        nonlocal ok
        S = comps[0][2]
        sq = math.sqrt(float(np.linalg.det(S)))
        g, tb, L = one_step(comps)
        ca = contour_areas(g["X"], g["Y"], L, LEVELS)
        lp = mixture_logpdf(g["X"], g["Y"], comps)
        log(f"  {name}: sqrt(det Sigma) = {sq:.1f} m^2, fine-grid mass "
            f"{tb.mass:.6f}, display cell {math.sqrt(g['A']):.2f} m")
        for p in LEVELS:
            ana = math.pi * sq * (-2.0 * math.log(1.0 - p))
            cells = float((L <= p).sum()) * g["A"]
            poly, nseg = ca[p]
            inside = float((np.exp(lp) * g["A"])[L <= p].sum())
            e1, e2 = cells / ana - 1.0, poly / ana - 1.0
            good = abs(e1) <= AREA_TOL and abs(e2) <= AREA_TOL\
                and abs(inside - p) <= 0.01 and nseg == 1
            ok &= good
            log(f"    {int(p * 100)} %: analytic {ana:10.1f} m^2 | grid cells "
                f"{cells:10.1f} ({e1 * 100:+.2f} %) | contour polygon "
                f"{poly:10.1f} ({e2 * 100:+.2f} %, {nseg} piece) | mass inside "
                f"{inside:.4f}  {'PASS' if good else 'FAIL'}")

    log("[1] single Gaussian, one step: HPD area vs pi*sqrt(det S)*chi2_2(p), "
        f"chi2_2(p) = -2 ln(1-p), tolerance {AREA_TOL * 100:.0f} %")
    area_test("isotropic, sigma = 50 m",
              [(1.0, np.array([120.0, -40.0]), np.eye(2) * 50.0 ** 2)])
    R = frame_to_world(math.radians(35.0))
    Sc = R @ cov_frame(80.0, 30.0, 0.6) @ R.T
    area_test("correlated, frame sigma 80 x 30 m, rho 0.6, theta 35 deg",
              [(1.0, np.array([-60.0, 90.0]), Sc)])

    log("[2] two-branch mixture: 90 % HPD region connectivity (8-neighbour)")
    for name, d, expect in (("separated, means 420 m apart", 420.0, 2),
                            ("overlapping, means 60 m apart (info)", 60.0, 1)):
        comps = [(0.6, np.array([0.0, 0.0]), np.diag([40.0 ** 2, 60.0 ** 2])),
                 (0.4, np.array([d, 0.0]), np.diag([55.0 ** 2, 35.0 ** 2]))]
        g, _, L = one_step(comps)
        lp = mixture_logpdf(g["X"], g["Y"], comps)
        ncomp = {p: label_components(L <= p) for p in LEVELS}
        inside = {p: float((np.exp(lp) * g["A"])[L <= p].sum()) for p in LEVELS}
        good = ncomp[0.9] == expect and all(abs(inside[p] - p) <= 0.01
                                            for p in LEVELS)
        ok &= good
        log(f"  {name}: components at 50 % = {ncomp[0.5]}, at 90 % = "
            f"{ncomp[0.9]} (expected {expect}); mass inside 50 %/90 % = "
            f"{inside[0.5]:.4f}/{inside[0.9]:.4f}  {'PASS' if good else 'FAIL'}")

    log("[3] frame -> world covariance rotation (Monte Carlo, 400k samples)")
    rng = np.random.default_rng(7)
    for sx, sy, rho, th in ((120.0, 40.0, 0.5, 2.1), (30.0, 90.0, -0.7, -0.6)):
        e1, e2 = rng.standard_normal((2, 400_000))
        xf = np.stack([sx * e1, sy * (rho * e1 + math.sqrt(1 - rho ** 2) * e2)])
        Rw = frame_to_world(th)
        emp = np.cov(Rw @ xf)
        ana = Rw @ cov_frame(sx, sy, rho) @ Rw.T
        rel = float(np.abs(emp - ana).max() / np.abs(ana).max())
        good = rel < 0.01
        ok &= good
        log(f"  sx={sx:.0f} sy={sy:.0f} rho={rho:+.1f} theta={th:+.1f}: max "
            f"|cov(R x) - R S R^T| / max|R S R^T| = {rel:.4f}  "
            f"{'PASS' if good else 'FAIL'}")
    for c in cases:
        good = c["prism"]["rot_check_m"] < 0.05
        ok &= good
        log(f"  {c['tag']}: max |R mu_frame + origin - predictor mu_w| = "
            f"{c['prism']['rot_check_m']:.2e} m (same R as used for Sigma)  "
            f"{'PASS' if good else 'FAIL'}")
    if not ok:
        raise AssertionError("HPD validation failed (see log)")
    return ok


def validate_colours(log):
    """Lightness of the ramp: monotone for normal, greyscale and CVD views."""
    def lin(c):
        c = np.asarray(c, float)
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def lstar(rgb_lin):
        Y = rgb_lin @ np.array([0.2126, 0.7152, 0.0722])
        f = np.where(Y > 216 / 24389, np.cbrt(Y), (24389 / 27 * Y + 16) / 116)
        return 116 * f - 16

    machado = {
        "protan": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
        "deutan": np.array([[0.367322, 0.860646, -0.227968],
                            [0.280085, 0.672501, 0.047413],
                            [-0.011820, 0.042940, 0.968881]]),
        "tritan": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]])}
    Ls = np.linspace(0.0, L_BLANK, 20)
    ok = True
    log("[4] colour ramp (Blues 0.90 -> 0.16): CIE L* along L = 0 .. 0.95, "
        f"full opacity and at the rollout-panel opacity {ROLL_BG_ALPHA}")
    for alpha in (1.0, ROLL_BG_ALPHA):
        srgb = alpha * CMAP(NORM(Ls))[:, :3] + (1.0 - alpha)   # over white
        rgb = lin(srgb)
        views = {"normal": rgb}
        views.update({k: np.clip(rgb @ M.T, 0, 1) for k, M in machado.items()})
        for k, v in views.items():
            Lv = lstar(v)
            mono = bool(np.all(np.diff(Lv) > 0))
            ok &= mono
            log(f"  alpha {alpha:.2f} {k:7s}: L* {Lv[0]:5.1f} -> {Lv[-1]:5.1f}, "
                f"min step {np.diff(Lv).min():.2f}, monotone "
                f"{'PASS' if mono else 'FAIL'}")
    marks = {"mean path": INK, "swept contour": CONTOUR_INK,
             "per-time contour": PER_TIME_INK, "true future": TRUE_FUT,
             "rollout": ROLL}
    rgbm = {k: lstar(lin(matplotlib.colors.to_rgb(v))[None])[0]
            for k, v in marks.items()}
    log("  line L*: " + ", ".join(f"{k} {v:.0f}" for k, v in rgbm.items())
        + "; ramp L* at L = 0 / 0.5 / 0.9: "
        + " / ".join(f"{lstar(lin(CMAP(NORM(x))[:3])[None])[0]:.0f}"
                     for x in (0.0, 0.5, 0.9)))
    return ok


def validate_panels(cases, log, interp: bool):
    """Checks on the actual figure grids (run after the maps exist)."""
    ok = True
    log("[5] fine-grid mass per swept-map time point (must be >= "
        f"{MASS_MIN}); resolution (cells per smallest sigma, cap = WARN)")
    for c in cases:
        for arm in ("prism", "cv"):
            tbs = c[f"tab_{arm}"]
            mmin = min(t.mass for t in tbs)
            caps = sum(t.capped for t in tbs)
            cells = [t.cells for t in tbs]
            cps = min(t.cells_per_sigma for t in tbs)
            good = mmin >= MASS_MIN
            ok &= good
            log(f"  {c['tag']} {arm:5s}: min mass {mmin:.6f} over {len(tbs)} "
                f"time points, fine grid {min(cells)}..{max(cells)} cells, "
                f"min {cps:.1f} cells/sigma, cap hits {caps}"
                f"{' (WARN)' if caps else ''}  {'PASS' if good else 'FAIL'}")

    log("[6] CV swept map vs closed form min_t [1 - exp(-|x - m_t|^2 / "
        "(2 sigma_t^2))] over the same time points, display grid")
    for c in cases:
        g = c["grid"]
        La = np.ones_like(g["X"])
        iso = True
        for _, comps, _ in c["sched_cv"]:
            (_, m, S), = comps
            iso &= bool(S[0, 1] == 0.0 and S[0, 0] == S[1, 1])
            r2 = (g["X"] - m[0]) ** 2 + (g["Y"] - m[1]) ** 2
            np.minimum(La, 1.0 - np.exp(-r2 / (2.0 * S[0, 0])), out=La)
        err = np.abs(c["L_cv"] - La)
        band = La <= L_BLANK
        good = iso and float(err[band].max()) < 0.01
        ok &= good
        log(f"  {c['tag']}: {len(c['sched_cv'])} time points, isotropic "
            f"{iso}; max |L_grid - L_closed| = {err[band].max():.5f} (cells "
            f"with L <= {L_BLANK}), mean {err[band].mean():.6f}  "
            f"{'PASS' if good else 'FAIL'}")

    # The 95 % region of the 15 s step lies inside the view (the view holds
    # the swept L <= 0.95 region), so for those cells {p > p(x)} is wholly on
    # the display grid and a direct sort of its cells must give the same L.
    log("[7] fine-table transfer vs direct sort of the display grid's own "
        "cells, PRISM, forecast step 15 s, cells with L_15 <= 0.95")
    for c in cases:
        g = c["grid"]
        comps = c["nodes_prism"][PRED_STEPS - 1]
        lp = mixture_logpdf(g["X"], g["Y"], comps)
        Ld, mass = direct_levels(lp, g["A"])
        Lt = c["node_tab_prism"][PRED_STEPS - 1].level(lp)
        band = Lt <= L_BLANK
        err = np.abs(Lt - Ld)[band]
        good = float(err.max()) < 0.01
        ok &= good
        log(f"  {c['tag']}: display grid holds {mass:.4f} of p_15; max |dL| "
            f"= {err.max():.5f}, mean {err.mean():.6f} over {int(band.sum())} "
            f"cells  {'PASS' if good else 'FAIL'}")

    # Independent route: the Monte-Carlo HPD threshold the paper's scoring
    # code uses (experiments/rollout_score.py): sample the mixture, take the
    # (1-p) quantile of the sample log-densities. Its level on the grid table
    # must be p, and it must classify the rollout positions the same way.
    log("[8] grid table vs Monte-Carlo HPD threshold (400k mixture samples), "
        "rollout cases, forecast steps 5 / 10 / 15 s")
    rng = np.random.default_rng(11)
    for c in cases:
        if "roll" not in c:
            continue
        for arm in ("prism", "cv"):
            for t, _ in PER_TIME:
                comps = c[f"nodes_{arm}"][t]
                hd = c[f"node_tab_{arm}"][t]
                w = np.array([cp[0] for cp in comps])
                ks = rng.choice(len(comps), size=400_000, p=w / w.sum())
                smp = np.empty((ks.size, 2))
                for k, (_, mu, S) in enumerate(comps):
                    m = ks == k
                    smp[m] = rng.multivariate_normal(mu, S, int(m.sum()))
                lps = mixture_logpdf(smp[:, 0], smp[:, 1], comps)
                E = c["roll"][:, t + 1, :]
                lpe = mixture_logpdf(E[:, 0], E[:, 1], comps)
                Le = hd.level(lpe)
                parts = []
                for p in LEVELS:
                    thr = float(np.quantile(lps, 1.0 - p))
                    Lthr = float(hd.level(np.array([thr]))[0])
                    n_mc = int((lpe >= thr).sum())
                    n_grid = int((Le <= p).sum())
                    good = abs(Lthr - p) < 0.005 and abs(n_mc - n_grid) <= 1
                    ok &= good
                    parts.append(f"{int(p * 100)} %: L(thr_MC) = {Lthr:.4f}, "
                                 f"inside MC {n_mc}/{len(E)} grid {n_grid}/"
                                 f"{len(E)} {'PASS' if good else 'FAIL'}")
                log(f"  {c['tag']} {arm:5s} {LOG_DT * (t + 1):4.1f} s | "
                    + " | ".join(parts))

    # The CV panels restrict the fit to the open track; the planner's CVRisk
    # uses every detection of the slot. Report whether that changed anything.
    log("[9] CV forecast (open-track detections) vs planner CVRisk on the "
        "unmodified arrays (information)")
    for c in cases:
        ep = load_episode(c["path"])
        risk = CVRisk()
        risk.update(ep, c["t0"])
        det = np.asarray(ep["obs_detected"][c["t0"]], bool)
        i = [s for s in range(det.shape[0]) if det[s]].index(c["slot"])
        D = risk.disks_batch(np.arange(1, PRED_STEPS + 1) * LOG_DT)[:, i, :]
        dm = float(np.abs(D[:, :2] - c["cv"]["mean"]).max())
        dsg = float(np.abs(D[:, 3] - c["cv"]["sigma"]).max())
        c["cv_equals_cvrisk"] = bool(dm < 1e-6 and dsg < 1e-6)
        log(f"  {c['tag']}: max |d mean| = {dm:.2e} m, max |d sigma| = "
            f"{dsg:.2e} m -> "
            + ("identical to CVRisk" if c["cv_equals_cvrisk"]
               else "differs (a closed track fell in the 4 s window)"))

    log("[10] display interpolation: covariance positive definite at every "
        "time point, equal to the node covariance at the nodes, node 0 = last "
        f"detection with {NODE0_COV_SCALE} x first-step covariance")
    if not interp:
        log("  --no-interp: the swept map uses the 30 forecast steps only "
            "(nothing to check)")
    for c in cases if interp else ():
        for arm in ("prism", "cv"):
            sched = c[f"sched_{arm}"]
            nodes = c[f"nodes_{arm}"]
            lam_min, rel_min = np.inf, np.inf
            for _, comps, _ in sched:
                for _, _, S in comps:
                    ev = np.linalg.eigvalsh(S)
                    lam_min = min(lam_min, float(ev[0]))
                    rel_min = min(rel_min, float(ev[0] / ev[1]))
            at_nodes = [(j, comps) for _, comps, j in sched if j is not None]
            dmax = max(max(float(np.abs(Sa - Sb).max()) +
                           float(np.abs(ma - mb).max())
                           for (_, ma, Sa), (_, mb, Sb) in zip(comps, nodes[j]))
                       for j, comps in at_nodes)
            node0 = sched[0][1]
            origin = (c["prism"]["origin"] if arm == "prism"
                      else c["cv"]["anchor"]) - c["org"]
            d0 = max(max(float(np.abs(S0 - NODE0_COV_SCALE * S1).max()),
                         float(np.abs(m0 - origin).max()))
                     for (_, m0, S0), (_, _, S1) in zip(node0, nodes[0]))
            times = np.array([t for t, _, _ in sched])
            dt_ok = bool(np.allclose(np.diff(times), SUBSTEP_S))
            good = (lam_min > 0.0 and dmax == 0.0 and d0 < 1e-9
                    and len(at_nodes) == PRED_STEPS and dt_ok)
            ok &= good
            log(f"  {c['tag']} {arm:5s}: {len(sched)} time points "
                f"({times[0]:.1f}..{times[-1]:.1f} s, step {SUBSTEP_S} s: "
                f"{dt_ok}); min eigenvalue {lam_min:.3f} m^2, min eigenvalue "
                f"ratio {rel_min:.4f}; max |diff| at the {len(at_nodes)} nodes "
                f"{dmax:.1e}; node 0 max |diff| {d0:.1e}  "
                f"{'PASS' if good else 'FAIL'}")
    return ok


# ============================================================== rendering
def nice_step(span: float) -> float:
    raw = span / 4.0
    mag = 10.0 ** np.floor(np.log10(max(raw, 1e-9)))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def fit_view(c: dict):
    """Square view holding both arms' swept L <= 0.95 regions and the data."""
    comps_all = ([cp for _, cp, _ in c["sched_prism"]]
                 + [cp for _, cp, _ in c["sched_cv"]])
    boxes = [comps_box(cp, 4.0) for cp in comps_all]
    lo = np.min([b[0] for b in boxes], 0)
    hi = np.max([b[1] for b in boxes], 0)
    ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2
    g = make_grid((ctr[0] - half, ctr[0] + half, ctr[1] - half, ctr[1] + half),
                  PROVISIONAL_N)
    # the older observation trail may leave the frame rather than dictate a
    # mostly empty square (fast targets): only the last 4 detections count
    pts = [c["fut"], c["obs"][-4:], c["cv_mean"]] + list(c["mu"])
    if "roll" in c:
        pts.append(c["roll"].reshape(-1, 2))
    for arm in ("prism", "cv"):
        L = swept_levels([cp for _, cp, _ in c[f"sched_{arm}"]],
                         c[f"tab_{arm}"], g)
        m = L <= L_BLANK
        pts.append(np.stack([g["X"][m], g["Y"][m]], 1))
    P = np.vstack([p for p in pts if len(p)])
    lo, hi = P.min(0), P.max(0)
    ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * 1.08 + 10
    step = nice_step(2 * half)
    half = float(np.ceil(half / (step / 2)) * (step / 2))   # half-tick rounding
    return (ctr[0] - half, ctr[0] + half, ctr[1] - half, ctr[1] + half), step


def compute_maps(c: dict):
    """Fine tables (forecast nodes shared with the swept schedule), view,
    swept maps and, for rollout cases, the per-time level fields."""
    for arm in ("prism", "cv"):
        c[f"node_tab_{arm}"] = [StepHPD(cp) for cp in c[f"nodes_{arm}"]]
        c[f"tab_{arm}"] = [c[f"node_tab_{arm}"][j] if j is not None
                           else StepHPD(cp) for _, cp, j in c[f"sched_{arm}"]]
    ext, c["tick"] = fit_view(c)
    c["ext"] = ext
    c["grid"] = g = make_grid(ext)
    for arm in ("prism", "cv"):
        c[f"L_{arm}"] = swept_levels([cp for _, cp, _ in c[f"sched_{arm}"]],
                                     c[f"tab_{arm}"], g)
        if "roll" in c:
            c[f"Lt_{arm}"] = {t: step_levels(c[f"nodes_{arm}"][t],
                                             c[f"node_tab_{arm}"][t], g)
                              for t, _ in PER_TIME}


def inclusion_counts(c: dict) -> dict:
    """Rollout positions inside each arm's per-time HPD region, computed on
    the forecast steps themselves (never on the interpolated display map)."""
    out = {}
    for arm in ("prism", "cv"):
        out[arm] = {}
        for t, lab in PER_TIME:
            comps = c[f"nodes_{arm}"][t]
            E = c["roll"][:, t + 1, :]
            Le = c[f"node_tab_{arm}"][t].level(
                mixture_logpdf(E[:, 0], E[:, 1], comps))
            out[arm][lab.replace(" ", "")] = {
                f"{int(p * 100)}": int((Le <= p).sum()) for p in LEVELS}
    out["n_rollouts"] = int(c["roll"].shape[0])
    return out


def resample(P: np.ndarray, step: float) -> np.ndarray:
    """Points every `step` along a polyline (both ends kept)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    if s[-1] <= 0.0:
        return P[:1]
    q = np.append(np.arange(0.0, s[-1], step), s[-1])
    return np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], 1)


def seg_length(s: np.ndarray) -> float:
    return float(np.hypot(*np.diff(s, axis=0).T).sum()) if len(s) > 1 else 0.0


def far_vertex(seg, data_pts, taken, ext, margin):
    """Vertex of `seg` farthest from data and from labels already taken,
    kept `margin` inside the view; None if no vertex qualifies."""
    V = seg[::2]
    inside = ((V[:, 0] > ext[0] + margin) & (V[:, 0] < ext[1] - margin)
              & (V[:, 1] > ext[2] + margin) & (V[:, 1] < ext[3] - margin))
    V = V[inside]
    if not len(V):
        return None, -np.inf
    d = np.min(np.hypot(V[:, None, 0] - data_pts[None, :, 0],
                        V[:, None, 1] - data_pts[None, :, 1]), 1)
    if taken:
        C = np.array(taken)
        d = np.minimum(d, 0.8 * np.min(np.hypot(
            V[:, None, 0] - C[None, :, 0], V[:, None, 1] - C[None, :, 1]), 1))
    i = int(np.argmax(d))
    return V[i], float(d[i])


def contour_label_spots(cs, data_pts: np.ndarray, ext):
    """Swept contours (a, b): one label per level, on the vertex farthest
    from the data and the other level's label, 9 % inside the view."""
    m = 0.09 * (ext[1] - ext[0])
    chosen, out = [], []
    for segs in cs.allsegs:
        best, bscore = None, -np.inf
        for seg in segs:
            if len(seg) < 12:                   # too small to carry a label
                continue
            v, sc = far_vertex(seg, data_pts, chosen, ext, m)
            if v is not None and sc > bscore:
                best, bscore = v, sc
        out.append(best)
        if best is not None:
            chosen.append(best)
    return out


def draw_per_time(ax, c, arm, data_pts, ext):
    """Per-time 90 % HPD contours at 5/10/15 s, each component labelled:
    inline if long enough, else a small text beside it. Returns the label
    positions (data coords) and the contour polylines."""
    span = ext[1] - ext[0]
    taken, polys = [], []
    for t, lab in PER_TIME:
        cs = ax.contour(c["grid"]["X"], c["grid"]["Y"], c[f"Lt_{arm}"][t],
                        levels=[PER_TIME_LEVEL], colors=PER_TIME_INK,
                        linewidths=0.75, linestyles=[PER_TIME_DASH[t]],
                        zorder=3)
        segs = [s for s in cs.allsegs[0] if len(s) > 2]
        polys += segs
        inline, side = [], []
        for s in segs:
            if seg_length(s) >= 0.16 * span and len(s) >= 12:
                v, _ = far_vertex(s, data_pts, taken, ext, 0.06 * span)
                if v is not None:
                    inline.append(tuple(v))
                    taken.append(v)
                    continue
            side.append(s)
        if inline:
            for tx in ax.clabel(cs, levels=[PER_TIME_LEVEL], manual=inline,
                                inline=True, inline_spacing=1.2, fontsize=5.4,
                                fmt={PER_TIME_LEVEL: lab}):
                tx.set_color(PER_TIME_INK)
                tx.set_zorder(3.5)
                tx.set_path_effects([pe.withStroke(linewidth=1.5,
                                                   foreground="white")])
        for s in side:                      # small blob: label beside it
            ctr = s.mean(0)
            r = float(np.hypot(*(s - ctr).T).max())
            dist = float(np.hypot(*ctr))
            u = ctr / dist if dist > 1e-6 else np.array([1.0, 0.0])
            nrm = np.array([-u[1], u[0]])      # left of the outward direction
            pos = ctr + nrm * (r + 0.035 * span)
            if not (ext[0] < pos[0] < ext[1] and ext[2] < pos[1] < ext[3]):
                pos = ctr - nrm * (r + 0.035 * span)
            tx = ax.text(pos[0], pos[1], lab, fontsize=5.4, color=PER_TIME_INK,
                         ha="center", va="center", zorder=3.5)
            tx.set_path_effects([pe.withStroke(linewidth=1.5,
                                               foreground="white")])
            taken.append(pos)
    return taken, polys


def draw_panel(ax, c: dict, arm: str, title: str, ylabel: bool,
               ytick_labels: bool = True):
    L = c[f"L_{arm}"]
    ext = c["ext"]
    rollcase = "roll" in c
    rgba = CMAP(NORM(np.clip(L, 0.0, L_BLANK)))
    rgba[..., 3] = (L <= L_BLANK) * (ROLL_BG_ALPHA if rollcase else 1.0)
    ax.imshow(rgba, extent=ext, origin="lower", interpolation="nearest",
              zorder=1)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(c["tick"]))
    ax.set_aspect("equal")
    ax.grid(True, lw=0.4, color="#ececec")
    for sp in ax.spines.values():
        sp.set_zorder(9)

    # label-avoidance points: every polyline resampled at one spacing, so the
    # placement cost measures drawn length, not vertex density
    ds = (ext[1] - ext[0]) / 60.0
    lines = [c["fut"]] + (list(c["mu"]) if arm == "prism" else [c["cv_mean"]])
    marks = []
    if rollcase:        # every 6th rollout: the bundles are near-identical
        lines += list(c["roll"][::6])
        marks = [c["roll"][:, t + 1, :] for t, _ in PER_TIME]
        ax.add_collection(LineCollection(list(c["roll"]), colors=ROLL,
                                         linewidths=0.25, alpha=0.3, zorder=2))
    D = np.vstack([c["obs"]] + [resample(ln, ds) for ln in lines] + marks)

    cl_pts, polys = [], []
    if not rollcase:
        cs = ax.contour(c["grid"]["X"], c["grid"]["Y"], L, levels=list(LEVELS),
                        colors=CONTOUR_INK, linewidths=0.55, zorder=3)
        polys = [s for seg in cs.allsegs for s in seg if len(s) > 1]
        # one call for all levels: clabel(levels=[one non-first level])
        # indexes past its level list in matplotlib 3.10. Each spot lies on
        # its own level's contour, so the nearest-contour match is right.
        spots = [tuple(p) for p in contour_label_spots(cs, D, ext)
                 if p is not None]
        if spots:
            for t in ax.clabel(cs, levels=list(LEVELS), manual=spots,
                               inline=True, inline_spacing=1.5, fontsize=5.6,
                               fmt={lv: f"{int(round(lv * 100))} %"
                                    for lv in LEVELS}):
                t.set_color(CONTOUR_INK)
                t.set_zorder(3.5)
                t.set_path_effects([pe.withStroke(linewidth=1.6,
                                                  foreground="white")])
                cl_pts.append(t.get_position())
    else:
        cl_pts, polys = draw_per_time(ax, c, arm, D, ext)

    ax.plot(c["obs"][:, 0], c["obs"][:, 1], "o", ms=2.1, mfc=OBS, mec="white",
            mew=0.3, ls="none", zorder=5)
    halo = lambda w: [pe.Stroke(linewidth=w + HALO_PT, foreground="white"),  # noqa: E731
                      pe.Normal()]
    ends = []
    if arm == "prism":
        pi, mu = c["prism"]["pi"], c["mu"]
        for k in np.argsort(-pi):
            lw = LW_PER_PI * float(pi[k])
            ax.plot(mu[k][:, 0], mu[k][:, 1], color=INK, lw=lw,
                    solid_capstyle="round", zorder=6, path_effects=halo(lw))
            ax.plot(*mu[k][-1], "o", ms=2.2, mfc=INK, mec="white", mew=0.45,
                    zorder=6.6)
            ends.append((int(k), mu[k][-1]))
    else:
        m = c["cv_mean"]
        lw = LW_PER_PI * 1.0
        ax.plot(m[:, 0], m[:, 1], color=INK, lw=lw, solid_capstyle="round",
                zorder=6, path_effects=halo(lw))
        ax.plot(*m[-1], "o", ms=2.2, mfc=INK, mec="white", mew=0.45,
                zorder=6.6)
    # the truth goes on top of the forecast (as in make_fig_cases.py)
    ax.plot(c["fut"][:, 0], c["fut"][:, 1], color=TRUE_FUT, lw=0.9, ls=DASH,
            dash_capstyle="butt", zorder=6.4, path_effects=halo(0.9))
    if rollcase:        # rollout positions at the per-time contour times
        for t, _ in PER_TIME:
            P = c["roll"][:, t + 1, :]
            ax.plot(P[:, 0], P[:, 1], ls="none", color=ROLL, mec=ROLL,
                    zorder=6.5, **ROLL_MARKERS[t])

    avoid = [D] + [resample(s, ds) for s in polys]
    if ends:
        ax._prism_labels = (ends, c["prism"]["pi"],
                            np.vstack([p for p in avoid if len(p)]),
                            np.array(cl_pts, float).reshape(-1, 2))

    ax.set_title(title, pad=3.5, color="#1a1a1a")
    ax.set_xlabel("x [m]", labelpad=1.5)
    if ylabel:
        ax.set_ylabel("y [m]", labelpad=1.5)
    if not ytick_labels:
        ax.tick_params(axis="y", labelleft=False)


DIRS = np.array([[np.cos(t), np.sin(t)]
                 for t in np.deg2rad(np.arange(0, 360, 22.5))])
RADII = np.array([13.0, 18.0, 24.0, 31.0, 39.0])


def label_branches(ax):
    """pi labels (make_fig_cases.py placement, ported): inside the axes,
    nearest their own branch end, clear of data, contours, contour labels
    and each other."""
    ends, pi, avoid, cl_pts = ax._prism_labels
    fig = ax.figure
    pt = fig.dpi / 72.0
    bb = ax.get_window_extent()
    hw, hh, pad = 17.0 * pt, 4.6 * pt, 2.5 * pt
    P = np.array([ax.transData.transform(e[1]) for e in ends])
    A = ax.transData.transform(avoid)

    def clutter(c, pts, scale, weight):
        if not len(pts):
            return 0.0
        dx = np.clip(np.abs(pts[:, 0] - c[0]) - hw, 0, None)
        dy = np.clip(np.abs(pts[:, 1] - c[1]) - hh, 0, None)
        return weight * float(np.exp(-np.hypot(dx, dy) / (scale * pt)).sum())

    # contour labels count as labels already placed
    placed = [ax.transData.transform(p) for p in cl_pts]
    slot = {}
    for i in np.argsort(-np.array([pi[e[0]] for e in ends])):
        best, best_cost = None, np.inf
        for r in RADII:
            for d in DIRS:
                c = P[i] + d * r * pt
                if (c[0] - hw < bb.x0 + pad or c[0] + hw > bb.x1 - pad
                        or c[1] - hh < bb.y0 + pad or c[1] + hh > bb.y1 - pad):
                    continue
                if len(P) > 1 and int(np.argmin(np.hypot(*(P - c).T))) != int(i):
                    continue
                cost = 2.4 * r
                cost += clutter(c, A, 5.0, 90.0)
                cost += clutter(c, P, 7.0, 260.0)
                # a label on a label is worse than a label across a line
                cost += clutter(c, np.array(placed) if placed
                                else np.zeros((0, 2)), 9.0, 4000.0)
                if r > 20:
                    lead = P[i] + np.outer(np.linspace(0.25, 0.9, 6), c - P[i])
                    cost += sum(clutter(q, A, 3.5, 26.0) for q in lead)
                if cost < best_cost:
                    best, best_cost = c, cost
        if best is None:
            best = np.clip(P[i] + np.array([16.0, 11.0]) * pt,
                           [bb.x0 + pad + hw, bb.y0 + pad + hh],
                           [bb.x1 - pad - hw, bb.y1 - pad - hh])
        placed.append(best)
        slot[int(i)] = best
    inv = ax.transData.inverted()
    for i, (k, xy) in enumerate(ends):
        c = slot[i]
        kw = dict(xy=xy, xytext=inv.transform(c), textcoords="data",
                  fontsize=6.1, color=INK, ha="center", va="center", zorder=8)
        if float(np.hypot(*(c - P[i]))) > 20 * pt:
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.4, color=INK,
                                    alpha=0.7, shrinkA=0.0, shrinkB=2.5)
        t = ax.annotate(rf"$\pi$ = {pi[k]:.2f}", **kw)
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])


def finish_labels(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        if hasattr(ax, "_prism_labels"):
            label_branches(ax)


def legend_entries(with_roll: bool):
    h = [Line2D([], [], color=OBS, marker="o", ms=2.6, ls="none"),
         Line2D([], [], color=TRUE_FUT, lw=0.9, ls=DASH),
         Line2D([], [], color=INK, lw=LW_PER_PI * 0.5)]
    lab = ["Observed", "True future", r"Mean path (width $\propto\ \pi$)"]
    if with_roll:
        h += [Line2D([], [], color=ROLL, lw=0.6, alpha=0.6),
              tuple(Line2D([], [], ls="none", color=ROLL, mec=ROLL,
                           **{**ROLL_MARKERS[t], "ms": ROLL_MARKERS[t]["ms"] + 0.8})
                    for t, _ in PER_TIME),
              tuple(Line2D([], [], color=PER_TIME_INK, lw=0.75,
                           ls=PER_TIME_DASH[t]) for t, _ in PER_TIME)]
        lab += ["Rollout futures (60)", "Rollouts at 5 / 10 / 15 s",
                "90 % HPD at 5 / 10 / 15 s"]
    return h, lab


def add_key(fig, legend_y: float, cbar_rect, with_roll: bool):
    """Legend + colour key; the key's dark marks are the swept levels."""
    h, lab = legend_entries(with_roll)
    fig.legend(handles=h, labels=lab, loc="center",
               bbox_to_anchor=(0.5, legend_y), ncol=3 if with_roll else 4,
               frameon=False, columnspacing=1.2,
               handlelength=3.0 if with_roll else 1.9,
               handletextpad=0.45, borderaxespad=0.1,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)})
    cax = fig.add_axes(cbar_rect)
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#8c8c8c")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 0.9])
    cb.set_ticklabels(["0", "25", "50", "75", "90"])
    cax.tick_params(labelsize=6.0, length=1.8, width=0.4, pad=1.2)
    for lv in LEVELS:
        cax.axvline(lv, color=CONTOUR_INK, lw=0.8)
    y = cbar_rect[1] + cbar_rect[3] / 2
    fig.text(cbar_rect[0] - 0.012, y, "HPD level L [%]", ha="right",
             va="center", fontsize=6.6)
    fig.text(cbar_rect[0] + cbar_rect[2] + 0.012, y,
             "swept 0\u201315 s; lines in (a, b): 50 % and 90 %; "
             "blank above 95 %" if with_roll else
             "swept 0\u201315 s; dark lines: 50 % and 90 %; blank above 95 %",
             ha="left", va="center", fontsize=6.0, color="#404040")


def case_title(c: dict) -> str:
    return (f"{TYPE_LABEL.get(c['type'], c['type'])} \u00b7 "
            f"{INTENT_LABEL.get(c['intent'], c['intent'])}")


def render(rows, out: str, dpi: int, sfx: str):
    """rows: list of (case, keys); first row is the non-rollout case."""
    spec, titles = {}, {}
    for c, (kp, kc) in rows:
        spec[kp], spec[kc] = (c, "prism"), (c, "cv")
        titles[kp] = f"PRISM \u00b7 {case_title(c)}"
        titles[kc] = f"CV \u00b7 {case_title(c)}"
    keys = [k for _, ks in rows for k in ks]
    left_keys = [ks[0] for _, ks in rows]
    with_roll = any("roll" in c for c, _ in rows)
    files = []

    # ---- combined (letters drawn here only; MDPI sets them for the
    # stand-alone panels, as in make_fig_cases.py)
    n = len(rows)
    s, left, gapx, bottom, gapy = 2.08, 0.50, 0.16, 0.34, 0.50
    ptop = bottom + n * s + (n - 1) * gapy
    key_h = 0.98 if with_roll else 0.72
    W, H = 4.95, ptop + key_h
    fig = plt.figure(figsize=(W, H))
    for r, (c, (kp, kc)) in enumerate(rows):
        y = bottom + (n - 1 - r) * (s + gapy)
        for col, key in enumerate((kp, kc)):
            ax = fig.add_axes([(left + col * (s + gapx)) / W, y / H,
                               s / W, s / H])
            cc, arm = spec[key]
            draw_panel(ax, cc, arm, f"({key}) {titles[key]}",
                       ylabel=col == 0, ytick_labels=col == 0)
    if with_roll:
        add_key(fig, legend_y=(ptop + 0.76) / H,
                cbar_rect=[0.20, (ptop + 0.36) / H, 0.26, 0.08 / H],
                with_roll=True)
    else:
        add_key(fig, legend_y=(ptop + 0.60) / H,
                cbar_rect=[0.22, (ptop + 0.33) / H, 0.28, 0.08 / H],
                with_roll=False)
    finish_labels(fig)
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"fig_hdr{sfx}.{ext}")
        fig.savefig(p, dpi=dpi)
        files.append(p)
    plt.close(fig)

    # ---- stand-alone panels, identical pixel size, no bbox_inches
    for key in keys:
        c, arm = spec[key]
        f1 = plt.figure(figsize=PANEL_FIGSIZE)
        ax1 = f1.add_subplot(111)
        f1.subplots_adjust(**PANEL_MARGINS)
        draw_panel(ax1, c, arm, titles[key], ylabel=key in left_keys)
        finish_labels(f1)
        for ext in ("png", "pdf"):
            p = os.path.join(out, f"fig_hdr_panel_{key}{sfx}.{ext}")
            f1.savefig(p, dpi=dpi)
            files.append(p)
        plt.close(f1)

    # ---- legend + colour key only
    fl = plt.figure(figsize=(2 * PANEL_FIGSIZE[0], 0.74 if with_roll else 0.52))
    if with_roll:
        add_key(fl, legend_y=0.72, cbar_rect=[0.20, 0.20, 0.26, 0.11],
                with_roll=True)
    else:
        add_key(fl, legend_y=0.76, cbar_rect=[0.22, 0.30, 0.28, 0.15],
                with_roll=False)
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"fig_hdr_legend{sfx}.{ext}")
        fl.savefig(p, dpi=dpi)
        files.append(p)
    plt.close(fl)
    return files


# ============================================================ caption text
def fmt_pi(c):
    return ", ".join(f"{p:.2f}" for p in np.sort(c["prism"]["pi"])[::-1])


def case_notes(ca, pub):
    """(a,b) selection sentence, EN and KR, built from computed eligibility."""
    rel = lambda p: p.replace(f"{ROOT}/", "")  # noqa: E731
    if (ca["run"], ca["ep"], ca["t0"], ca["slot"]) != CASE_DEFAULT:
        return (f"a user-specified sample ({rel(ca['path'])}, t0 = {ca['t0']}, "
                f"slot {ca['slot']})",
                f"지정한 표본이다({rel(ca['path'])}, t0 = {ca['t0']}, "
                f"슬롯 {ca['slot']}).")
    pb, pc = pub["b"], pub["c"]
    miss_b = "" if pb["detected_t0"] else (
        ", and sample (b) has no detection at t0, so the planner's CV arm "
        "has no forecast for it")
    miss_b_kr = "" if pb["detected_t0"] else (
        " 또한 (b)는 t0 탐지가 없어 CV 군이 예측하지 못한다.")
    en = ("the sample of the published case figure, panel (a) (OOD test run "
          "_a, ep_000029, t0 = 31, slot 1). The published (b) and (c) samples "
          "are not reused: under the current-track rule their open tracks hold "
          f"{pb['n_open_track_10s']} and {pc['n_open_track_10s']} detections in "
          f"the 10 s history (below min_obs = {MIN_OBS_SELECT}; the slot rule "
          f"counted {pb['n_slot_10s']} and {pc['n_slot_10s']}){miss_b}")
    kr = ("기존 사례 그림 (a)의 표본이다(OOD 시험 런 _a, ep_000029, t0 = 31, "
          "슬롯 1). 기존 (b), (c) 표본은 쓰지 않았다. 현재 트랙 규칙에서 열린 "
          f"트랙의 10초 탐지가 각각 {pb['n_open_track_10s']}개, "
          f"{pc['n_open_track_10s']}개로 min_obs = {MIN_OBS_SELECT}에 못 "
          f"미친다(슬롯 규칙으로는 {pb['n_slot_10s']}개, "
          f"{pc['n_slot_10s']}개).{miss_b_kr}")
    return en, kr


def incl90(inc, arm):
    return "/".join(str(inc[arm][lab.replace(" ", "")]["90"])
                    for _, lab in PER_TIME)


def write_caption(path, ca, rolls, pub, model_path, info, interp, hard_note):
    """rolls: list of (case, sel, counts, inc, keys, rule_name)."""
    rel = lambda p: p.replace(f"{ROOT}/", "")  # noqa: E731
    note_en, note_kr = case_notes(ca, pub)

    def tgt(c, kr=False):
        if kr:
            return (f"{TYPE_KR.get(c['type'], c['type'])}, "
                    f"{INTENT_KR.get(c['intent'], c['intent'])}")
        return (f"{TYPE_LABEL.get(c['type'], c['type'])}, "
                f"{INTENT_LABEL.get(c['intent'], c['intent'])}")

    same_cv = all(c.get("cv_equals_cvrisk") for c in [ca] + [r[0] for r in rolls])
    cv_src = "detections" if same_cv else "open-track detections"
    cv_src_kr = "탐지" if same_cv else "열린 트랙의 탐지"
    pk = " and ".join(f"({kp}, {kc})" for _, _, _, _, (kp, kc), _ in rolls)
    pk_kr = ", ".join(f"({kp}, {kc})" for _, _, _, _, (kp, kc), _ in rolls)
    interp_en = (
        " For display only, the swept map is evaluated every 0.1 s: each "
        "branch's mean and covariance matrix are interpolated linearly between "
        "the 0.5 s forecast steps, starting at t0 from the last detection with "
        "half the covariance of the first step. All coverage numbers use the "
        "forecast steps themselves." if interp else
        " The swept map uses the 30 forecast steps only.")
    interp_kr = (
        " 누적 지도는 표시용으로만 0.1초 간격으로 계산하였다. 0.5초 예측 스텝 사이에서 "
        "갈래별 평균과 공분산 행렬을 선형 보간하며, t0에서는 마지막 탐지 위치와 첫 스텝 "
        "공분산의 절반에서 출발한다. 포함 비율과 개수는 모두 예측 스텝 자체로 계산하였다."
        if interp else " 누적 지도는 30개 예측 스텝만으로 계산하였다.")

    roll_en, roll_kr, sel_en, sel_kr = [], [], [], []
    for c, sel, counts, inc, (kp, kc), rule in rolls:
        roll_en.append(
            f"({kp}) PRISM and ({kc}) CV; target {tgt(c)}, pi = {fmt_pi(c)}; "
            f"rollouts inside the 90 % HPD region at 5/10/15 s: PRISM "
            f"{incl90(inc, 'prism')}, CV {incl90(inc, 'cv')} of "
            f"{inc['n_rollouts']}.")
        roll_kr.append(
            f"({kp}) PRISM, ({kc}) CV. 대상은 {tgt(c, True)}이다(π = {fmt_pi(c)}). "
            f"5/10/15초 90% HPD 영역 안의 롤아웃은 PRISM {incl90(inc, 'prism')}개, "
            f"CV {incl90(inc, 'cv')}개이다({inc['n_rollouts']}개 중).")
        run_r = os.path.basename(sel["run"])
        where = (f"rank {sel['rank']} of {sel['n_kept']}; {run_r}, "
                 f"{sel['episode']}, slot {sel['slot']}; spread "
                 f"{sel['spread15_m']:.1f} m, cluster separation "
                 f"{sel['separation_m']:.1f} m")
        where_kr = (f"{sel['n_kept']}개 중 {sel['rank']}번째; {run_r}, "
                    f"{sel['episode']}, 슬롯 {sel['slot']}; 퍼짐 "
                    f"{sel['spread15_m']:.1f} m, 군집 간 거리 "
                    f"{sel['separation_m']:.1f} m")
        if rule == "rollout":
            sel_en.append(
                f"({kp}, {kc}): among the rollout groups of both rollout-"
                "evaluation runs whose target both arms can predict, the "
                "groups whose 60 rollout endpoints at 15 s split into two "
                "2-means clusters with centroid separation > 3 x the mean "
                "within-cluster std; of these, the group with the median 15 s "
                f"rollout spread ({where}).")
            sel_kr.append(
                f"({kp}, {kc})는 두 롤아웃 평가 런의 롤아웃 그룹 가운데 두 군이 모두 "
                "예측할 수 있는 그룹에서 고른다. 15초 롤아웃 끝점 60개가 2-means로 두 "
                "군집으로 나뉘고 중심 간 거리가 군집 내 표준편차 평균의 3배를 넘는 "
                f"그룹만 남긴다. 그중 15초 롤아웃 퍼짐이 중앙값인 그룹이다({where_kr}).")
        else:
            sel_en.append(
                f"({kp}, {kc}): the same rule on the test_ood rollout run, "
                "restricted to the HARD classes (bird or unknown platform, or "
                f"search-wander intent), selected without model output ({where}).")
            sel_kr.append(
                f"({kp}, {kc})는 같은 규칙을 test_ood 롤아웃 런의 어려운 클래스(조류 또는 "
                "미확인 기체, 또는 탐색 배회 의도)에만 적용하여 모델 출력 없이 골랐다"
                f"({where_kr}).")
    if hard_note:
        sel_en.append(hard_note[0])
        sel_kr.append(hard_note[1])

    en = f"""Figure X. Predictive position distribution drawn as a time-swept highest predictive density (HPD) map.
For every forecast step t = 0.5-15 s the K = 3 mixture p_t(x) = sum_k pi_k N(x; mu_kt, Sigma_kt) (full covariance with rho, rotated from the target frame to world axes) is evaluated on a grid, and each cell receives its HPD level L_t(x), the probability mass of the region where p_t exceeds p_t(x). The colour shows the swept level L(x) = min_t L_t(x), so a cell lies inside the 90 % HPD region at some time exactly when L(x) <= 0.9. Darker cells lie inside smaller regions; cells with L > 0.95 are blank.{interp_en} Black lines are branch means (width proportional to pi_k; pi labelled at 15 s), dots are the detections in the 10 s history used by the predictor, and the dashed line is the true future. Axes are metres relative to the target's true position at t0.
(a) PRISM and (b) constant-velocity (CV) forecast; target {tgt(ca)}, pi = {fmt_pi(ca)}; dark lines mark the swept 50 % and 90 % levels. In the rollout panels {pk} the swept map is drawn lighter, thin labelled lines are the 90 % HPD regions of the forecast at 5, 10 and 15 s (dotted, dashed, solid), and grey markers are the positions of 60 branch rollouts at those times (triangle 5 s, square 10 s, dot 15 s). The rollouts share the simulated past and continue from 30.0 s with the agents' decision, perception, wind and sensor random streams re-seeded. {' '.join(roll_en)} The CV forecast is the planner's CV arm: least-squares velocity over the last 4 s of {cv_src}, anchored at the detection at t0, isotropic sigma(t) = sqrt(15^2 + (4t)^2) m, drawn on the same colour scale.
The map overlays per-time position marginals. The 90 % region holds 90 % of the position mass at each forecast time separately; it is not the probability that the whole trajectory stays inside it, and a point inside the swept region may lie outside the 15 s region. The map is also not the planner's risk field. The planner reduces each branch to an isotropic sigma = sqrt((sigma_x^2 + sigma_y^2)/2) with a 10 m floor, drops branches with pi < 0.05, and applies an unnormalised kernel with peak gain proportional to pi_k that is flat within the NMAC standoff.
Case selection. (a, b): {note_en}. {' '.join(sel_en)}
"""
    kr = f"""그림 X. 예측 위치 분포의 시간 누적 최고밀도영역(HPD) 수준 지도.
예측 스텝 t = 0.5~15초마다 K = 3 혼합 밀도 p_t(x) = Σ_k π_k N(x; μ_kt, Σ_kt)를 격자에서 계산한다. 공분산은 ρ를 포함하며 대상 좌표계에서 세계 좌표계로 회전한다. 각 셀의 HPD 수준 L_t(x)는 p_t가 p_t(x)보다 큰 영역의 확률 질량이다. 색은 누적 수준 L(x) = min_t L_t(x)이다. 따라서 L(x) ≤ 0.9인 셀은 어느 한 시점에서 90% HPD 영역 안에 든다. 진할수록 작은 영역에 속하고, L > 0.95인 셀은 비워 두었다.{interp_kr} 검은 선은 갈래 평균 경로이며 굵기는 π_k에 비례하고, 15초 끝점에 π를 표기하였다. 점은 예측기가 사용한 10초 이력의 탐지이고 파선은 실제 미래이다. 좌표는 t0 시점 대상의 실제 위치 기준 m이다.
(a) PRISM, (b) 등속(CV) 예측. 대상은 {tgt(ca, True)}이다(π = {fmt_pi(ca)}). 진한 선은 누적 50%와 90% 수준이다. 롤아웃 패널 {pk_kr}에서는 누적 지도를 옅게 그렸다. 가는 선과 시간 표기는 5, 10, 15초 예측의 90% HPD 영역이고(점선, 파선, 실선), 회색 표식은 같은 시점의 분기 롤아웃 60개 위치이다(삼각형 5초, 사각형 10초, 점 15초). 롤아웃은 같은 과거에서 30.0초부터 의사결정·인지·바람·센서 난수만 바꾸어 다시 굴린 미래이다. {' '.join(roll_kr)} CV 예측은 플래너의 CV 군과 같다. 속도는 최근 4초 {cv_src_kr}로 최소제곱 적합하고, t0 탐지 위치에서 출발하며, σ(t) = √(15² + (4t)²) m 등방이다. 색 척도는 PRISM과 같다.
이 지도는 시점별 위치 주변분포의 HPD 수준을 겹친 것이다. 90% 영역은 각 시점의 위치 질량 90%를 담는다. 궤적 전체가 그 안에 머물 확률은 아니며, 누적 영역 안의 점이 15초 영역 밖에 있을 수 있다. 또한 플래너가 쓰는 위험장도 아니다. 플래너는 갈래마다 σ = √((σx² + σy²)/2)로 등방화하고, 10 m 하한을 두고, π < 0.05인 갈래를 버린다. 커널은 정규화하지 않으며, 최대 이득은 π_k에 비례하고 NMAC 이격 안에서 평평하다.
사례 선택. (a, b)는 {note_kr} {' '.join(sel_kr)}
"""

    def facts(c):
        f = c["facts"]
        return (f"input holds {c['n_used']}/20 detections from track ids "
                f"{f['track_ids_in_input']} (model rule '{c['track_rule']}'); "
                f"branch mean speed over 15 s (pi desc) "
                f"{f['branch_mean_speed_15s_mps']} m/s, true speed "
                f"{f['true_speed_range_mps'][0]}-{f['true_speed_range_mps'][1]}"
                f" m/s, CV speed {f['cv_speed_mps']} m/s (fit on "
                f"{c['cv']['n_fit']} detections; "
                + ("equal to CVRisk on the unmodified arrays)"
                   if c.get("cv_equals_cvrisk") else
                   "differs from CVRisk: a closed track fell in the window)"))

    lines = [
        "--- notes (not part of the caption) ---",
        f"model: {rel(model_path)} (epoch {info.get('epoch')}, track rule "
        f"'{info.get('track_rule')}'); rerun with --model for the retrained "
        "checkpoint.",
        f"case (a,b): {rel(ca['path'])}, t0 = {ca['t0']} (t = {ca['t_s']:.1f} "
        f"s), slot {ca['slot']}; {facts(ca)}."]
    for c, sel, counts, inc, (kp, kc), rule in rolls:
        lines.append(
            f"case ({kp},{kc}): {rel(sel['run'])}/episodes/{sel['episode']} + "
            f"rollouts/{sel['episode'][:-4]}_rollouts.npz group {sel['group']}; "
            f"history ends at index {sel['t0']} (t = {c['t_s']:.1f} s = branch "
            f"time), rollout step k at 30.0 + 0.5 (k + 1) s; {facts(c)}.")
        lines.append(
            f"case ({kp},{kc}) rule stage counts: {json.dumps(counts)}; kept-"
            f"group spread quartiles [m]: {sel['kept_spread_quartiles_m']}.")
        lines.append(
            f"case ({kp},{kc}) rollout positions inside the per-time HPD "
            f"region (50 % / 90 %, of {inc['n_rollouts']}), forecast steps "
            "themselves: " + "; ".join(
                f"{arm.upper() if arm == 'cv' else 'PRISM'} " + ", ".join(
                    f"{lab} {inc[arm][lab.replace(' ', '')]['50']}/"
                    f"{inc[arm][lab.replace(' ', '')]['90']}"
                    for _, lab in PER_TIME) for arm in ("prism", "cv"))
            + " (state-conditioned, one group each, descriptive only).")
    lines += [
        ("display interpolation: every 0.1 s between the 0.5 s forecast steps, "
         "branch-wise linear in the mean and in the covariance entries (a "
         "convex combination of positive-definite matrices, checked); node 0 "
         "= last detection at t0 with 0.5 x the first step's covariance (a "
         "display choice: the model has no t = 0 output); used for the swept "
         "map only." if interp else
         "display interpolation off (--no-interp): swept map over the 30 "
         "forecast steps, as in v1."),
        f"rollout rule (full): {ROLLOUT_RULE}",
        f"HARD rule: {HARD_RULE}",
        "episode 244 (planner case) is not drawn: case_ep244.json holds the "
        "closed-loop trajectories only, and scene_244_t60/t66.json hold branch "
        "mu_w, sigma (frame axes, rounded) and pi but no rho, no frame angle "
        "and no observation arrays, so the full covariance cannot be rebuilt; "
        "reproducing the scene needs the closed-loop planner replay "
        "(dump_scene.py / replay_case.py)."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("[EN]\n" + en + "\n[KR]\n" + kr + "\n" + "\n".join(lines) + "\n")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--suffix", default="_v2",
                    help="appended to every output file name (v1 had none)")
    ap.add_argument("--no-interp", action="store_true",
                    help="sweep the 30 forecast steps only (v1 swept map)")
    ap.add_argument("--case-run", default=CASE_DEFAULT[0])
    ap.add_argument("--case-ep", type=int, default=CASE_DEFAULT[1])
    ap.add_argument("--case-t0", type=int, default=CASE_DEFAULT[2])
    ap.add_argument("--case-slot", type=int, default=CASE_DEFAULT[3])
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    interp, sfx = not args.no_interp, args.suffix
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log(f"make_fig_hdr.py (v2)  {now}  suffix '{sfx}'  display interpolation "
        f"{'on (0.1 s)' if interp else 'off'}")
    fc = Forecaster(args.model)
    log(f"model {args.model}  epoch {fc.info['epoch']}  track rule "
        f"'{fc.info['track_rule']}'  device {fc.info['device']}")

    recs = scan_rollout_groups(ROLLOUT_RUNS)
    sel_c, counts_c = apply_rule(recs, ROLLOUT_RUNS, hard=False)
    sel_h, counts_h = apply_rule(recs, (RUN_OOD,), hard=True)
    if sel_c is None:
        raise SystemExit("no rollout group satisfies ROLLOUT_RULE")
    for name, sel, counts in (("rollout rule", sel_c, counts_c),
                              ("HARD rule", sel_h, counts_h)):
        log(f"{name} stage counts {json.dumps(counts)}")
        if sel is None:
            log(f"{name}: no group passes; the rule is not relaxed")
            continue
        log(f"{name} choice: {os.path.basename(sel['run'])} {sel['episode']} "
            f"group {sel['group']} slot {sel['slot']} type "
            f"{TYPE_NAMES[sel['gt_type']]} intent "
            f"{INTENT_NAMES[sel['gt_intent']]} spread {sel['spread15_m']:.1f} m "
            f"sep {sel['separation_m']:.1f} m cluster std "
            f"{[round(s_, 1) for s_ in sel['cluster_std_m']]} n "
            f"{sel['cluster_n']} (median rank {sel['median_rank']})")

    ca = build_case(fc, args.case_run, args.case_ep, args.case_t0,
                    args.case_slot, interp)
    ca["tag"] = "case (a,b)"
    rolls = []
    for sel, counts, keys, rule in ((sel_c, counts_c, ("c", "d"), "rollout"),
                                    (sel_h, counts_h, ("e", "f"), "hard")):
        if sel is None:
            continue
        c = build_case(fc, sel["run"], sel["ep"], sel["t0"], sel["slot"],
                       interp, rollout=load_rollout(sel))
        c["tag"] = f"case ({keys[0]},{keys[1]})"
        if abs(c["t_s"] - sel["origin_step"] * LOG_DT) > 1e-6:
            raise AssertionError(f"{c['tag']}: history does not end at the "
                                 "branch time")
        rolls.append([c, sel, counts, None, keys, rule])
    hard_note = None
    if sel_h is None:
        hard_note = ("(e, f): no group of the HARD classes on the test_ood run "
                     "passed the rule (stage counts in the notes); the rule was "
                     "not relaxed and panels e, f are not drawn.",
                     "(e, f): test_ood 런의 어려운 클래스에서는 규칙을 통과한 그룹이 없어 "
                     "그리지 않았다. 규칙은 완화하지 않았다.")
    cases = [ca] + [r[0] for r in rolls]
    # why the other two published samples are not reused (computed, not typed)
    pub = {k: eligibility(load_episode(f"{r}/episodes/ep_{e:06d}.npz"), t, s_)
           for k, (r, e, t, s_) in PUBLISHED.items()}
    log("published fig_cases samples, eligibility: "
        + json.dumps({k: {q: v[q] for q in ("n_slot_10s", "n_open_track_10s",
                                              "detected_t0")}
                      for k, v in pub.items()}))

    vpath = os.path.join(args.out, f"hdr_validation{sfx}.txt")
    try:
        log("\n=== HPD validation ===")
        validate(cases, log)
        log("colour check:")
        col_ok = validate_colours(log)
        for c in cases:
            compute_maps(c)
        pan_ok = validate_panels(cases, log, interp)
        if not (col_ok and pan_ok):
            raise AssertionError("panel / colour validation failed (see log)")
    finally:
        with open(vpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    log("\n=== cases ===")
    for r in rolls:
        r[3] = inclusion_counts(r[0])
    for c in cases:
        log(f"{c['tag']}: {c['path']} t0={c['t0']} (t={c['t_s']:.1f} s) "
            f"slot={c['slot']} {c['type']}/{c['intent']} pi="
            f"{np.round(c['prism']['pi'], 3).tolist()} used det={c['n_used']} "
            f"cv fit n={c['cv']['n_fit']} cv speed="
            f"{float(np.hypot(*c['cv']['vel'])):.1f} m/s view="
            f"{[round(v) for v in c['ext']]}")
    for c, _, _, inc, _, _ in rolls:
        log(f"{c['tag']} rollout positions inside the per-time HPD region "
            f"(50 %/90 %, of {inc['n_rollouts']}): " + json.dumps(
                {k: v for k, v in inc.items() if k != "n_rollouts"}))
    with open(vpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    rows = [(ca, ("a", "b"))] + [(r[0], r[4]) for r in rolls]
    files = render(rows, args.out, args.dpi, sfx)
    cap = os.path.join(args.out, f"caption_hdr{sfx}.txt")
    write_caption(cap, ca, [tuple(r) for r in rolls], pub, args.model,
                  fc.info, interp, hard_note)

    def case_meta(c):
        pi = c["prism"]["pi"]
        order = np.argsort(-pi)
        return {"episode_file": c["path"], "episode_sha256": sha256(c["path"]),
                "t0": c["t0"], "t_s": c["t_s"], "slot": c["slot"],
                "gt_type": c["type"], "gt_intent_t0": c["intent"],
                "eligibility": c["elig"],
                "pi_desc": [round(float(pi[k]), 3) for k in order],
                "sigma15_m_desc": [[round(float(v), 1) for v in
                                    c["prism"]["sigma"][k, -1]] for k in order],
                "cv": {"speed_mps": round(float(np.hypot(*c["cv"]["vel"])), 2),
                       "n_fit": c["cv"]["n_fit"],
                       "sigma15_m": round(float(c["cv"]["sigma"][-1]), 1),
                       "equals_planner_cvrisk": c.get("cv_equals_cvrisk")},
                "facts": c["facts"],
                "view_m": [round(float(v), 1) for v in c["ext"]],
                "n_detections_used": c["n_used"],
                "swept_map_time_points": len(c["sched_prism"])}

    man = {"figure": f"fig_hdr{sfx}", "version": "v2",
           "terminology": "highest predictive density (HPD) region",
           "generated_utc": now,
           "script": {"path": os.path.abspath(__file__).replace("\\", "/"),
                      "sha256": sha256(os.path.abspath(__file__)),
                      "v1_snapshot": "make_fig_hdr_v1.py"},
           "model": {"path": args.model, "sha256": sha256(args.model),
                     **{k: fc.info[k] for k in ("epoch", "track_rule", "K")}},
           "method": {"fine_grid": {"n_sigma_box": N_SIG_BOX,
                                    "cells_per_sigma": CELLS_PER_SIGMA,
                                    "max_side": FINE_MAX_SIDE,
                                    "mass_min": MASS_MIN},
                      "display_n": DISPLAY_N, "blank_above": L_BLANK,
                      "swept_contours_ab": list(LEVELS),
                      "per_time_contour_level": PER_TIME_LEVEL,
                      "per_time_steps_s": [LOG_DT * (t + 1) for t, _ in PER_TIME],
                      "display_interpolation": {
                          "enabled": interp, "substep_s": SUBSTEP_S,
                          "scheme": "branch-wise linear in mean and "
                                    "covariance entries between forecast "
                                    "steps; weights constant",
                          "node0": "last detection at t0, covariance "
                                   f"{NODE0_COV_SCALE} x first forecast step",
                          "used_for": "swept map only; coverage and "
                                      "inclusion counts use forecast steps"},
                      "colormap": "matplotlib Blues 0.90 -> 0.16",
                      "view": "square; both arms' swept L <= 0.95, branch "
                              "means, CV mean, true future, rollouts, last 4 "
                              "detections; ticks on round steps",
                      "rollout_panel_background_alpha": ROLL_BG_ALPHA,
                      "mean_path_lw_pt_per_pi": LW_PER_PI},
           "case_ab": {**case_meta(ca), "rule": case_notes(ca, pub)[0],
                       "published_samples_eligibility": pub}}
    for c, sel, counts, inc, (kp, kc), rule in rolls:
        man[f"case_{kp}{kc}"] = {
            **case_meta(c),
            "rule": ROLLOUT_RULE if rule == "rollout" else HARD_RULE,
            "selection": {k: v for k, v in sel.items()},
            "stage_counts": counts,
            "rollout_file": f"{sel['run']}/rollouts/"
                            f"{sel['episode'][:-4]}_rollouts.npz",
            "inclusion_counts_per_time": inc}
    if sel_h is None:
        man["case_ef"] = {"rule": HARD_RULE, "stage_counts": counts_h,
                          "result": "no group passed; not relaxed, not drawn"}
    man["files"] = files + [cap, vpath]
    mp = os.path.join(args.out, f"fig_hdr_manifest{sfx}.json")
    with open(mp + ".tmp", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    os.replace(mp + ".tmp", mp)
    for p in man["files"] + [mp]:
        print("[saved]", p)


if __name__ == "__main__":
    main()
