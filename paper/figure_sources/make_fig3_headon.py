# -*- coding: utf-8 -*-
"""Figure 3 -- a purpose-built head-on encounter run in the simulator.

Rather than mining a random episode, this drives dataGen directly with a
scenario stripped down to the one thing the figure has to show:

  * exactly two traffic agents, both eVTOL on corridor transit (straight legs)
  * one inverse-designed HEAD-ON encounter against the ownship
  * no wind, no no-fly zones -- nothing that bends a path for another reason
  * the rev.3 reaction layer left on, so the intruder evades on its own

Seeds are swept and scored, and the clearest run is plotted: original heading,
detection range, and what the track did after being seen.

Run:  python make_fig3_headon.py
Out:  D:/PRISM/Paper/figures/fig3_headon/
"""
from __future__ import annotations

import os

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, (os.environ["PRISM_ROOT"] + '/dataGen'))

from core.types import GeometryType, IntentType, PlatformType   # noqa: E402
from scenario.config import (ScenarioConfig, EncounterConfig,   # noqa: E402
                             PlatformMixConfig, IntentMixConfig, WindConfig)
from simulation.engine import SimulationEngine                  # noqa: E402

OUT_DIR = (os.environ["PRISM_ROOT"] + '/Paper/figures/fig3_headon')
SEEDS = range(4000, 4400)
PANEL_IN = (2.75, 3.30)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.color": "#b0b0b0",
    "grid.alpha": 0.35, "lines.solid_capstyle": "round",
    "savefig.dpi": 400, "figure.dpi": 140,
})

PCOL = {0: "#e8c547", 1: "#4fc3f7", 2: "#66bb6a", 3: "#ab47bc",
        4: "#ef5350", 5: "#90a4ae", 6: "#546e7a"}
PNAME = {0: "Bird", 1: "Multirotor", 2: "Fixed-wing UAS", 3: "eVTOL",
         4: "General aviation", 5: "Debris", 6: "Unknown"}
INK, WARN, GHOST = "#333333", "#D55E00", "#8a8a8a"


#: Fixed-wing UAS is the intruder: on corridor transit it holds a straight
#: leg, and its envelope (bank 40-50 deg, onset 25-35 deg/s^2) makes the
#: avoidance turn legible. An eVTOL reacts too -- but passenger-comfort
#: limits cap its whole-episode heading change near 28 deg, which does not
#: read as a turn on the page.
INTRUDER = PlatformType.FIXED_WING_UAS


def head_on_config(seed: int) -> ScenarioConfig:
    """Two straight corridor legs, one designed head-on, no wind."""
    return ScenarioConfig(
        name="fig3_headon", seed=int(seed), duration_s=60.0, area_km2=4.0,
        density_per_km2=0.5, n_agents_min=2, n_agents_max=2,
        ownship_speed_ms=55.0,
        platform_mix=PlatformMixConfig(weights={int(INTRUDER): 1.0}),
        intent_mix=IntentMixConfig(
            weights={int(IntentType.CORRIDOR_TRANSIT): 1.0}),
        encounter=EncounterConfig(
            n_designed_min=1, n_designed_max=1,
            # the validator requires the upper bound to clear DWC_HMD_M
            # (1219.2 m) so RiskLevel LOW/NONE stay reachable; tight cases
            # are then picked out by score_run() instead of by the sampler
            cpa_dist_min_m=40.0, cpa_dist_max_m=1400.0,
            tcpa_min_s=26.0, tcpa_max_s=40.0,
            geometry_weights={int(GeometryType.HEAD_ON): 1.0}),
        wind=WindConfig(speed_min_ms=0.0, speed_max_ms=0.0),
        corridor_enabled=True, nfz_count=0, reaction_enabled=True,
    )


def turn_profile(psi, valid):
    """Unwrapped heading and the step of the sharpest sustained turn."""
    p = np.unwrap(np.where(valid, psi, np.nan))
    rate = np.abs(np.diff(p))
    rate[~(valid[:-1] & valid[1:])] = 0.0
    return p, rate


def score_run(ep, meta):
    """Legibility score for one run; None when the case is not usable."""
    gv, gt, det = ep.gt_valid, ep.gt_pos, ep.obs_detected
    own, psi = ep.own_pos, ep.gt_psi
    if meta.reaction_events < 1:
        return None
    best = None
    T = gv.shape[0]
    for a in range(gv.shape[1]):
        v = gv[:, a]
        if v.sum() < 110:
            continue
        d = np.where(v, np.hypot(gt[:, a, 0] - own[:, 0],
                                 gt[:, a, 1] - own[:, 1]), np.inf)
        icpa = int(np.argmin(d))
        sep = float(d[icpa])
        if not (150.0 < sep < 900.0) or icpa < 20 or icpa > T - 8:
            continue
        seen = v & det[:, a]
        if seen.sum() < 0.60 * v.sum():
            continue                       # the intruder must be observable
        i0 = int(np.nonzero(seen)[0][0])
        p, rate = turn_profile(psi[:, a], v)

        # onset = first heading rate that stays high for three steps, after
        # acquisition and before the closest approach
        onset = None
        hot = rate > np.radians(1.5)
        for i in range(max(i0, 8), icpa - 2):
            if hot[i] and hot[i + 1] and hot[i + 2]:
                onset = i
                break
        if onset is None:
            continue

        # a straight leg before the turn is what makes "original heading"
        # unambiguous on the page
        pre = float(np.degrees(abs(p[max(0, onset - 16)] - p[onset])))
        if pre > 10.0:
            continue
        j = min(T - 1, onset + 30)
        turn = float(np.degrees(np.nanmax(np.abs(p[onset:j] - p[onset]))))
        if not (25.0 <= turn <= 80.0):
            continue                       # a loop does not read as avoidance

        # genuine head-on: at the turn the intruder is closing nearly
        # nose to nose with the ownship
        vo, vi = own[onset] - own[onset - 4], gt[onset, a] - gt[onset - 4, a]
        no, ni = np.hypot(*vo), np.hypot(*vi)
        if no < 1e-6 or ni < 1e-6:
            continue
        rel = np.degrees(np.arccos(
            np.clip(float(vo @ vi) / (no * ni), -1.0, 1.0)))
        if rel < 130.0:
            continue

        # what the turn actually bought: hold the pre-turn velocity and see
        # how close the intruder would have come to the same ownship track
        vel = (gt[onset, a] - gt[onset - 1, a])
        ghost = gt[onset, a] + np.outer(np.arange(T - onset), vel)
        d_hold = float(np.min(np.hypot(ghost[:, 0] - own[onset:, 0],
                                       ghost[:, 1] - own[onset:, 1])))
        d_real = float(np.min(d[onset:]))
        if d_hold > 500.0 or d_real <= d_hold:
            continue                       # the turn has to have helped

        s = (round(d_real - d_hold, 1), -abs(turn - 45.0))
        if best is None or s > best[0]:
            best = (s, dict(agent=a, icpa=icpa, sep=sep, turn=turn,
                            onset=onset, first_det=i0, rel_bearing=rel,
                            d_hold=d_hold, d_real=d_real))
    return best


def sweep():
    best = None
    failed = 0
    for sd in SEEDS:
        try:
            eng = SimulationEngine(head_on_config(sd))
            ep, meta = eng.run_episode()
        except Exception as e:                    # report, never swallow
            if not failed:
                print(f"  !! seed {sd}: {type(e).__name__}: {e}")
            failed += 1
            continue
        r = score_run(ep, meta)
        if r is None:
            continue
        if best is None or r[0] > best[0]:
            best = (r[0], sd, ep, meta, r[1])
            print(f"  seed {sd}: turn {r[1]['turn']:.0f} deg, "
                  f"sep {r[1]['sep']:.0f} m, "
                  f"reactions {meta.reaction_events}")
    if failed:
        print(f"  ({failed} seeds failed to build)")
    return best


def plot(ep, meta, info, seed):
    a, icpa, onset = info["agent"], info["icpa"], info["onset"]
    gv, gt, det = ep.gt_valid, ep.gt_pos, ep.obs_detected
    own, t = ep.own_pos, ep.t
    v = gv[:, a]
    c = PCOL[int(ep.gt_type[a])]

    seen = np.nonzero(v & det[:, a])[0]
    i0 = int(seen[0])
    rng0 = float(np.hypot(*(gt[i0, a] - own[i0])))

    fig = plt.figure(figsize=PANEL_IN, layout="constrained")
    ax = fig.add_subplot(1, 1, 1)

    # ownship: scripted straight leg
    ax.plot(own[:, 0], own[:, 1], color=INK, lw=1.3, ls="--", zorder=4)
    ax.plot(own[0, 0], own[0, 1], "s", ms=3.6, mfc=INK, mec="white",
            mew=0.5, zorder=6)

    # the path not taken: pre-turn velocity held to the end of the episode
    vel = gt[onset, a] - gt[onset - 1, a]
    n_rest = len(t) - onset
    ext = gt[onset, a] + np.outer(np.arange(n_rest), vel)
    ax.plot(ext[:, 0], ext[:, 1], color=GHOST, lw=1.0, ls=(0, (1.3, 1.9)),
            zorder=3)
    ax.annotate(f"original heading held:\nwould pass at "
                f"{info['d_hold']:,.0f} m",
                xy=(ext[int(0.72 * n_rest), 0], ext[int(0.72 * n_rest), 1]),
                xytext=(-8, 0), textcoords="offset points", fontsize=5.9,
                color=GHOST, ha="right", va="center", zorder=9)
    # where that ghost would have been at its own closest approach
    dg = np.hypot(ext[:, 0] - own[onset:, 0], ext[:, 1] - own[onset:, 1])
    kg = int(np.argmin(dg))
    ax.plot([own[onset + kg, 0], ext[kg, 0]],
            [own[onset + kg, 1], ext[kg, 1]], color=GHOST, lw=0.7, ls=":",
            zorder=3)

    # intruder: solid where the sensor had it, dashed where it did not
    ax.plot(*np.where((v & det[:, a])[:, None], gt[:, a], np.nan).T,
            color=c, lw=2.0, zorder=5)
    ax.plot(*np.where((v & ~det[:, a])[:, None], gt[:, a], np.nan).T,
            color=c, lw=1.6, ls=(0, (2.2, 1.6)), zorder=5)

    # acquisition, reaction onset, closest approach
    ax.plot([own[i0, 0], gt[i0, a, 0]], [own[i0, 1], gt[i0, a, 1]],
            color=GHOST, lw=0.7, ls=(0, (4, 2)), zorder=3)
    ax.plot(gt[i0, a, 0], gt[i0, a, 1], "o", ms=4.8, mfc="white", mec=c,
            mew=1.3, zorder=8)
    ax.annotate(f"first detection\n{rng0:,.0f} m", xy=gt[i0, a],
                xytext=(-7, 7), textcoords="offset points", fontsize=6.0,
                color=c, ha="right", va="bottom", zorder=9)

    ax.plot(gt[onset, a, 0], gt[onset, a, 1], "o", ms=8.0, mfc="none",
            mec=WARN, mew=1.3, zorder=8)
    ax.annotate(f"avoidance turn\nt = {t[onset]:.0f} s", xy=gt[onset, a],
                xytext=(7, -7), textcoords="offset points", fontsize=6.0,
                color=WARN, ha="left", va="top", zorder=9)

    ax.plot([own[icpa, 0], gt[icpa, a, 0]], [own[icpa, 1], gt[icpa, a, 1]],
            color=WARN, lw=0.9, ls=":", zorder=6)
    mid = 0.5 * (own[icpa] + gt[icpa, a])
    ax.annotate(f"CPA {info['sep']:,.0f} m", xy=mid, xytext=(-6, -8),
                textcoords="offset points", fontsize=6.2, color=WARN,
                ha="right", va="top", zorder=9)

    pts = np.vstack([gt[v, a], own])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    h = 0.5 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 1.18 + 80.0
    ax.set_xlim(cx - h, cx + h)
    ax.set_ylim(cy - h, cy + h)
    ax.set_box_aspect(1.0)
    ax.grid(True)
    ax.set_xlabel("x [m]", labelpad=1.5)
    ax.set_ylabel("y [m]", labelpad=1.5)
    ax.set_title(f"Designed head-on encounter "
                 f"({info['rel_bearing']:.0f}\u00b0 relative)\n"
                 f"{info['turn']:.0f}\u00b0 avoidance turn opens "
                 f"{info['d_hold']:,.0f} m \u2192 {info['d_real']:,.0f} m",
                 pad=4, color=INK)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext_ in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"headon_seed{seed}.{ext_}"),
                    facecolor="white")
    plt.close(fig)

    # legend strip, matching the other panels
    # label follows INTRUDER; it used to say "eVTOL" while the run was
    # actually flying a fixed-wing UAS
    h_ = [Line2D([], [], color=c, lw=2.0,
                 label=f"Intruder ({PNAME[int(ep.gt_type[a])]})"),
          Line2D([], [], color=INK, lw=1.3, ls="--", label="Ownship"),
          Line2D([], [], color="#777777", lw=2.0, label="detected"),
          Line2D([], [], color="#777777", lw=1.6, ls=(0, (2.2, 1.6)),
                 label="sensor gap"),
          Line2D([], [], color=GHOST, lw=0.95, ls=(0, (1.3, 1.9)),
                 label="original heading"),
          Line2D([], [], color=WARN, lw=0, marker="o", ms=6, mfc="none",
                 mew=1.3, label="avoidance turn")]
    figl = plt.figure(figsize=(6.9, 0.42), layout="constrained")
    axl = figl.add_subplot(1, 1, 1)
    axl.axis("off")
    axl.legend(handles=h_, loc="center", ncol=6, frameon=False,
               columnspacing=1.1, handlelength=1.7, handletextpad=0.45)
    for ext_ in ("png", "pdf"):
        figl.savefig(os.path.join(OUT_DIR, f"headon_legend.{ext_}"),
                     facecolor="white")
    plt.close(figl)


def main():
    print(f"sweeping {len(SEEDS)} seeds for a clean head-on ...")
    best = sweep()
    if best is None:
        sys.exit("no usable head-on encounter found; widen SEEDS or the "
                 "scoring window in score_run()")
    _, seed, ep, meta, info = best
    print(f"\nchosen seed {seed}: agent {info['agent']}, "
          f"turn {info['turn']:.1f} deg, CPA {info['sep']:.0f} m at "
          f"t={ep.t[info['icpa']]:.1f} s, onset t={ep.t[info['onset']]:.1f} s, "
          f"reaction_events={meta.reaction_events}, "
          f"min_sep={meta.min_separation_m:.0f} m")
    plot(ep, meta, info, seed)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
