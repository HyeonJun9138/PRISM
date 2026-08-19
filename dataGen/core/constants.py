"""Physical constants, timing constants and separation criteria for PRISM.

Every quantity is SI (metres, seconds, radians, kilograms) unless the symbol
name says otherwise. These values are part of the dataset contract: changing
any of the timing constants invalidates previously generated datasets, and
changing a separation threshold invalidates both the reward function (doc 7.3)
and the evaluation metrics (doc 9.4), which MUST agree.

Frame convention used everywhere in dataGen:
    Local tangent plane, +X = East [m], +Y = North [m].
    psi = heading [rad], measured CCW from +X (standard math convention,
    NOT compass), wrapped to (-pi, pi].
    Positive turn rate omega = left turn = CCW = increasing psi.
"""

from __future__ import annotations
import os

import math

__all__ = [
    "G",
    "SIM_DT",
    "LOG_DT",
    "LOG_DECIMATION",
    "HIST_STEPS",
    "PRED_STEPS",
    "HIST_SECONDS",
    "PRED_SECONDS",
    "NMAC_RADIUS_M",
    "DWC_HMD_M",
    "DWC_TAU_MOD_S",
    "EPS_V",
    "LOGS_ROOT",
    "DEG",
    "RAD",
    "TWO_PI",
]

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------

#: Standard gravitational acceleration [m/s^2] (ISO 80000-3 / CGPM).
G: float = 9.80665

#: Radians per degree [rad/deg] -- multiply a degree value by this.
DEG: float = math.pi / 180.0

#: Degrees per radian [deg/rad] -- multiply a radian value by this.
RAD: float = 180.0 / math.pi

#: 2*pi [rad], provided so angle-wrapping code never re-derives it.
TWO_PI: float = 2.0 * math.pi

# --------------------------------------------------------------------------
# Timing (doc 3.7: integrate fine, log coarse)
# --------------------------------------------------------------------------

#: Integration timestep [s] -- 20 Hz. Used by simulation.engine only.
SIM_DT: float = 0.05

#: Logging timestep [s] -- 2 Hz. The sampling rate of the stored dataset.
LOG_DT: float = 0.5

#: Number of integration steps per logged step [-]. Must equal LOG_DT/SIM_DT.
LOG_DECIMATION: int = 10

#: Observation history window length [logged steps] -- 10 s at LOG_DT.
HIST_STEPS: int = 20

#: Prediction horizon length [logged steps] -- 15 s at LOG_DT.
PRED_STEPS: int = 30

#: Observation history window duration [s].
HIST_SECONDS: float = HIST_STEPS * LOG_DT

#: Prediction horizon duration [s].
PRED_SECONDS: float = PRED_STEPS * LOG_DT

# --------------------------------------------------------------------------
# Separation criteria (doc 9.4). 2D paper -> horizontal component only.
# --------------------------------------------------------------------------

#: Near Mid-Air Collision horizontal radius [m] = 500 ft.
#: Commonly cited NMAC definition (horizontal 500 ft / vertical 100 ft); only
#: the horizontal condition applies in this 2D study.
NMAC_RADIUS_M: float = 152.4

#: DAA Well Clear horizontal miss distance (HMD) threshold [m] = 4000 ft.
#: WARNING / ACTION REQUIRED BEFORE PUBLICATION (doc 9.4, doc 12.5):
#: this value is the commonly quoted figure and MUST be verified against the
#: RTCA DO-365 (DAA MOPS) source text before any number derived from it is
#: reported in the paper. Do not treat it as sourced until that check is done.
DWC_HMD_M: float = 1219.2

#: DAA Well Clear modified tau (tau_mod) threshold [s].
#: Same verification requirement as DWC_HMD_M: confirm against RTCA DO-365.
DWC_TAU_MOD_S: float = 35.0

# --------------------------------------------------------------------------
# Numerical guards
# --------------------------------------------------------------------------

#: Minimum speed [m/s] used in the denominator of bank/tilt-limited turn-rate
#: formulae, so that a near-zero airspeed cannot produce an infinite cap.
EPS_V: float = 0.5

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: Dataset output root. Forward slashes on purpose (Windows-safe in Python).
#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

LOGS_ROOT: str = f"{PRISM_ROOT}/Logs/Dataset"
