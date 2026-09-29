"""Figure 6 decision-sequence schematic for author review.

This is intentionally a conceptual diagram, not a logged t+1 simulator
snapshot or the quantitative PRISM risk kernel.  It must be described as a
schematic if inserted in the manuscript.
"""

import os

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


OUT = Path((os.environ["PRISM_ROOT"] + '/Paper/figures_r3'))
PNG = OUT / "figure_06.png"
PDF = OUT / "figure_06.pdf"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.8,
    "axes.titlesize": 10.1, "axes.labelsize": 8.8,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "axes.linewidth": 0.65, "axes.edgecolor": "#92989C",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

BLUE = "#0069A8"
GRAY = "#8B969E"
INK = "#20272B"
REF = "#59646B"

x = np.linspace(25, 535, 620)


def candidate(offset, turn_shift=0.0, return_shift=0.0):
    """A smooth lateral excursion that returns towards the reference."""
    return offset * 0.5 * (np.tanh((x - (165 + turn_shift)) / 65)
                           - np.tanh((x - (455 + return_shift)) / 80))


offsets = [-100, -80, -60, -40, -20, 20, 40, 60, 80]
candidate_specs = [(offset, shift, shift * 0.55)
                   for offset in offsets for shift in (-16.0, 0.0, 16.0)]
selected = candidate(-100)

# Compact, locally displayed encounter-risk contours.  Values are relative
# visual levels for this schematic, not planner scores or probabilities.
gx = np.linspace(-15, 535, 650)
gy = np.linspace(-175, 175, 420)
X, Y = np.meshgrid(gx, gy)
risk = np.exp(-0.5 * (((X - 310) / 45) ** 2
                      + ((Y - 4) / 42) ** 2))
levels = [0.055, 0.14, 0.26, 0.42, 0.62, 0.80, 1.01]
colors = ListedColormap([
    "#FFF0D9", "#FFDBB4", "#FDBF86", "#FA9B5A", "#EB6740", "#C9252C"
])

fig = plt.figure(figsize=(8.6, 3.3), facecolor="white")
ax = fig.add_axes([0.067, 0.19, 0.425, 0.69])
bx = fig.add_axes([0.565, 0.19, 0.425, 0.69])


def frame(a):
    a.set_xlim(-15, 535)
    a.set_ylim(-175, 175)
    a.set_aspect("equal", adjustable="box")
    a.axhspan(-140, 140, fc="#F3F6F7", ec="none", zorder=0)
    for d in (-140, 140):
        a.axhline(d, color="#C9D0D3", lw=0.8, zorder=1)
    a.contourf(X, Y, risk, levels=levels, cmap=colors,
               antialiased=True, alpha=0.95, zorder=2)
    a.plot(310, 4, marker="x", ms=5.7, mew=1.3, color=INK,
           zorder=10)
    a.set_xticks([0, 100, 200, 300, 400, 500])
    a.set_yticks([-150, -100, -50, 0, 50, 100, 150])
    a.tick_params(direction="out", length=2.5, width=0.6,
                  color="#879096")
    a.set_xlabel(r"Along-track position $s$ [m]", labelpad=2)
    a.set_ylabel(r"Lateral position $d$ [m]", labelpad=2)


frame(ax)
frame(bx)

# Before: the reference continues straight through the red core while the
# planner has not yet distinguished among its lateral candidate paths.
for offset, turn_shift, return_shift in candidate_specs:
    ax.plot(x, candidate(offset, turn_shift, return_shift),
            color=GRAY, lw=0.82, alpha=0.58, zorder=5)
ax.plot([25, 535], [0, 0], color=INK, lw=1.5,
        ls=(0, (5, 3)), zorder=6)
ax.plot(25, 0, marker=(3, 0, -90), ms=9.6, mfc=INK,
        mec="white", mew=0.9, zorder=11)
ax.annotate("straight course", xy=(175, 0), xytext=(82, 25),
            ha="left", va="bottom", fontsize=8.1, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK,
                            shrinkA=1, shrinkB=3), zorder=12)
ax.annotate("risk contour", xy=(332, 38), xytext=(385, 93),
            ha="left", va="center", fontsize=8.1, color="#9B3F24",
            bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.3),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#B55D38",
                            shrinkA=2, shrinkB=3), zorder=12)
ax.text(450, -115, "candidate paths", ha="right", va="center",
        fontsize=8.0, color="#66747D",
        bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.2), zorder=12)
ax.set_title("(a) Before: candidate paths are evaluated", loc="left", pad=5)

# After: the same candidates remain visible but only the lower, peripheral
# route is selected.  The position marker indicates the next schematic step.
for offset, turn_shift, return_shift in candidate_specs:
    if offset == -100 and turn_shift == 0:
        continue
    bx.plot(x, candidate(offset, turn_shift, return_shift),
            color=GRAY, lw=0.78, alpha=0.22, zorder=4)
bx.plot([25, 535], [0, 0], color=REF, lw=1.0,
        ls=(0, (5, 3)), alpha=0.42, zorder=4)
next_x = 112.0
next_y = float(np.interp(next_x, x, selected))
before = x <= next_x
after = x >= next_x
bx.plot(x[before], selected[before], color=BLUE, lw=2.4,
        alpha=0.40, solid_capstyle="round", zorder=8)
bx.plot(x[after], selected[after], color=BLUE, lw=3.0,
        solid_capstyle="round", zorder=9)
bx.plot(next_x, next_y, marker=(3, 0, -98), ms=10.0,
        mfc=BLUE, mec="white", mew=0.9, zorder=12)
bx.annotate("", xy=(266, float(np.interp(266, x, selected))),
            xytext=(232, float(np.interp(232, x, selected))),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=9,
                            lw=1.6, color=BLUE), zorder=11)
bx.annotate("selected path", xy=(371, float(np.interp(371, x, selected))),
            xytext=(445, -122), ha="right", va="center", fontsize=8.2,
            color=BLUE, bbox=dict(fc="white", ec="none", alpha=0.82,
                                  pad=1.2),
            arrowprops=dict(arrowstyle="-", lw=0.7, color=BLUE,
                            shrinkA=2, shrinkB=2), zorder=12)
bx.text(495, 112, "not selected", ha="right", va="center",
        fontsize=8.0, color="#7D8991",
        bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.2), zorder=12)
bx.set_title(r"(b) Next step ($t+1$): one path is selected",
             loc="left", pad=5)

fig.text(0.985, 0.97, "SCHEMATIC", ha="right", va="top",
         color="#68757D", fontsize=7.6)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(PNG, dpi=360)
fig.savefig(PDF)
plt.close(fig)
print(PNG)
print(PDF)
