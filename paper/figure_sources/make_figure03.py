"""Landscape preview of Figure 3 using the published simulated encounter."""

from __future__ import annotations

import os

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path((os.environ["PRISM_ROOT"] + ''))
sys.path.insert(0, str(ROOT / "Paper" / "figures"))
import make_fig3_headon as source  # noqa: E402


OUTPUT = ROOT / "Paper" / "figures_r3" / "figure_03.png"


def main():
    seed = 4183
    episode, metadata = source.SimulationEngine(source.head_on_config(seed)).run_episode()
    selection = source.score_run(episode, metadata)
    if selection is None:
        raise RuntimeError("Published Figure 3 encounter was not reproduced")
    info = selection[1]
    agent = info["agent"]
    onset = info["onset"]
    first = info["first_det"]
    cpa = info["icpa"]

    own = episode.own_pos
    intr = episode.gt_pos[:, agent]
    valid = episode.gt_valid[:, agent]
    velocity = intr[onset] - intr[onset - 1]
    ghost = intr[onset] + np.outer(np.arange(len(episode.t) - onset), velocity)

    # A rigid rotation aligns the ownship's heading with the horizontal
    # axis. It changes neither relative distances nor the turn angle.
    direction = own[-1] - own[0]
    angle = float(np.arctan2(direction[1], direction[0]))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    origin = own[onset]

    def rotate(points):
        return (points - origin) @ rotation.T

    own_r, intr_r, ghost_r = rotate(own), rotate(intr), rotate(ghost)
    first_range = float(np.linalg.norm(intr[first] - own[first]))
    assert abs(first_range - 2377.934) < 0.1
    assert abs(info["turn"] - 57.2468) < 0.01
    assert abs(info["d_hold"] - 84.4778) < 0.01
    assert abs(info["d_real"] - 481.8816) < 0.01
    assert np.allclose(np.linalg.norm(intr[cpa] - own[cpa]),
                       np.linalg.norm(intr_r[cpa] - own_r[cpa]), atol=0.01)

    green, orange, ink, gray = "#2f9e44", "#d55e00", "#303030", "#777777"
    fig = plt.figure(figsize=(7.1, 2.75), dpi=400, facecolor="white")
    ax = fig.add_axes([0.075, 0.18, 0.695, 0.71])
    fig.add_artist(Line2D([0.795, 0.795], [0.10, 0.90],
                          transform=fig.transFigure, color="#d9d9d9", lw=0.65))

    ax.plot(own_r[:, 0], own_r[:, 1], color=ink, lw=1.5,
            ls=(0, (5, 3)), zorder=2)
    ax.annotate("", xy=(1780, 0), xytext=(1670, 0),
                arrowprops=dict(arrowstyle="->", color=ink, lw=1.45,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=7)
    ax.text(2160, 20, "Ownship path", ha="right", va="bottom",
            fontsize=7.1, color=ink)
    ax.plot(ghost_r[:, 0], ghost_r[:, 1], color=gray, lw=1.25,
            ls=(0, (1.4, 2.0)), zorder=1)
    intr_r[~valid] = np.nan
    ax.plot(intr_r[:, 0], intr_r[:, 1], color=green, lw=2.05,
            solid_capstyle="butt", solid_joinstyle="round", zorder=4)
    ax.annotate("", xy=intr_r[13], xytext=intr_r[7],
                arrowprops=dict(arrowstyle="->", color=green, lw=1.65,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=7)

    # The original and turned closest-approach links are left visible.
    ghost_sep = np.linalg.norm(ghost - own[onset:], axis=1)
    ghost_cpa = int(np.argmin(ghost_sep))
    ax.plot([ghost_r[ghost_cpa, 0], own_r[onset + ghost_cpa, 0]],
            [ghost_r[ghost_cpa, 1], own_r[onset + ghost_cpa, 1]],
            color=gray, lw=0.9, ls=(0, (1.3, 1.6)), zorder=3)
    ax.plot([intr_r[cpa, 0], own_r[cpa, 0]],
            [intr_r[cpa, 1], own_r[cpa, 1]],
            color=orange, lw=1.0, ls=(0, (2, 2)), zorder=3)

    ax.plot(intr_r[first, 0], intr_r[first, 1], marker="o", ms=5.8,
            mfc="white", mec=green, mew=1.45, zorder=6)
    ax.annotate("first detection", xy=intr_r[first], xytext=(9, 12),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=7.4, color=green)
    ax.plot(intr_r[onset, 0], intr_r[onset, 1], marker="o", ms=8.3,
            mfc="none", mec=orange, mew=1.45, zorder=6)
    ax.annotate("turn", xy=intr_r[onset], xytext=(-11, 12),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7.4, color=orange)

    ax.set_xlim(675, 2225)
    ax.set_ylim(-110, 510)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Along-track distance (m)", fontsize=8.0, labelpad=1)
    ax.set_ylabel("Cross-track distance (m)", fontsize=8.0, labelpad=1)
    ax.tick_params(labelsize=7.0, length=2.5, width=0.6, pad=1.5)
    ax.grid(True, color="#d9d9d9", lw=0.45)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)

    fig.text(0.825, 0.89, "Head-on encounter", fontsize=8.0,
             fontweight="semibold", color=ink)
    fig.text(0.825, 0.83, f"{info['rel_bearing']:.0f}° relative approach",
             fontsize=7.1, color=ink)

    def key(y, color, style, label, lw=1.5):
        fig.add_artist(Line2D([0.825, 0.865], [y, y],
                              transform=fig.transFigure,
                              color=color, lw=lw, ls=style))
        fig.text(0.878, y, label, fontsize=7.0, va="center", color=ink)

    key(0.71, green, "-", "Intruder", 1.9)
    key(0.62, ink, (0, (5, 3)), "Ownship", 1.4)
    key(0.53, gray, (0, (1.4, 2.0)), "No-turn path", 1.25)
    fig.text(0.825, 0.405, f"First seen: {first_range:,.0f} m",
             fontsize=7.0, color=green)
    fig.text(0.825, 0.342, f"{info['turn']:.0f}° turn (≈10 s)",
             fontsize=7.0, color=orange)
    fig.text(0.825, 0.225, "Closest approach", fontsize=7.0, color=ink)
    fig.text(0.825, 0.155, f"{info['d_hold']:.0f} → {info['d_real']:.0f} m",
             fontsize=10.5, fontweight="semibold", color=ink)
    fig.text(0.825, 0.10, "No turn     With turn", fontsize=6.35, color=gray)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=400, facecolor="white", pil_kwargs={"compress_level": 6})
    plt.close(fig)
    print(OUTPUT)
    print(f"rotation={-np.degrees(angle):.2f} deg; seed={seed}; "
          f"CPA={info['d_hold']:.1f}->{info['d_real']:.1f} m")


if __name__ == "__main__":
    main()
