"""Shared constants for the PRISM prediction-model training system.

All physical quantities are SI (metres, seconds). The dataset is produced by
$PRISM_ROOT/dataGen and stored under $PRISM_ROOT/Logs; trained models are written to
$PRISM_ROOT/Models. See 논문관련.txt [5.1] (rev.3) for the input/output contract
this trainer implements.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------- data layout
LOG_DT = 0.5            # s per logged step (2 Hz)
HIST_STEPS = 20         # observation history window (10 s)
PRED_STEPS = 30         # prediction horizon (15 s)

# ------------------------------------------------------------- normalisation
# Features and targets are scaled before entering the network; predictions are
# un-scaled back to metres for every reported metric.
POS_SCALE = 100.0       # m   -> feature/target unit
VEL_SCALE = 20.0        # m/s -> feature unit
R_SCALE = 10.0          # m (sqrt-covariance) -> feature unit
AGE_SCALE = 20.0        # logged steps
GAP_SCALE = 20.0        # logged steps

# ------------------------------------------------------------------ features
# Per-timestep feature vector, in this exact order (doc 5.1 rev.3):
FEATURES = [
    "dx", "dy",             # relative position in target frame (masked to 0)
    "detected",             # 0/1 detection mask
    "dvx", "dvy",           # finite-difference velocity (consecutive detections)
    "dvalid",               # 0/1 diff-velocity validity
    "sqrt_R00", "sqrt_R11", # measurement std devs in target frame
    "rho_R",                # measurement correlation in target frame
    "track_age",            # normalised
    "gap",                  # steps since last detection, normalised
    "is_ownship",           # slot flag
    "is_target",            # slot flag
]
F_DIM = len(FEATURES)   # 13

# ------------------------------------------------------------------- classes
# PlatformType enum of dataGen/core/types.py. Order is load-bearing.
TYPE_NAMES = ["Bird", "Multirotor", "FixedWingUAS", "eVTOL", "GA", "Debris", "Unknown"]
N_TYPES = len(TYPE_NAMES)
INTENT_NAMES = ["CorridorTransit", "DeliveryWP", "LoiterHover",
                "SearchWander", "ReactiveAvoid", "PassiveDrift"]

# ------------------------------------------------------------------- outputs
# Per mode, per step: (mu_x, mu_y, log_sigma_x, log_sigma_y, rho_raw)
PARAMS_PER_STEP = 5
LOG_SIGMA_MIN = -5.0    # sigma >= e^-5 * POS_SCALE = 0.67 m
LOG_SIGMA_MAX = 3.0     # sigma <= e^3  * POS_SCALE = 2008 m
RHO_LIMIT = 0.95        # rho = RHO_LIMIT * tanh(raw)

# --------------------------------------------------------------------- paths
#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

LOGS_ROOT = f"{PRISM_ROOT}/Logs/Dataset"
MODELS_ROOT = f"{PRISM_ROOT}/Logs/Models"

# ----------------------------------------------------------------------- gui
GUI_HOST = "127.0.0.1"
GUI_PORT = 8790         # dataGen GUI uses 8770; keep disjoint

# -------------------------------------------------------------------- eval
COVERAGE_LEVELS = (0.5, 0.7, 0.9)
COVERAGE_STEPS = (9, 19, 29)        # 0-based step index -> 5 s, 10 s, 15 s
COVERAGE_MC_SAMPLES = 64            # Monte-Carlo HPD samples per point
