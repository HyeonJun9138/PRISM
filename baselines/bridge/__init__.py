"""Path bridge: makes dataGen (physics) and modelTraining (model) importable.

modelVal deliberately owns NO physics and NO model code. The validation
environment must exercise the exact simulator the dataset came from and the
exact feature pipeline the model was trained on -- a reimplementation that
drifted by one detail would validate nothing. Import this module FIRST.

Import order note: dataGen is inserted ahead of modelTraining, and only
non-colliding package names are ever imported from each root
(dataGen: core/scenario/agents/environment/sensors/simulation;
modelTraining: common/data/model). Both roots also have 'analysis' and 'gui'
packages -- modelVal must never import those two names.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAGEN_ROOT = os.path.join(os.path.dirname(_HERE), "dataGen")
TRAINING_ROOT = os.path.join(os.path.dirname(_HERE), "modelTraining")

for _p in (TRAINING_ROOT, DATAGEN_ROOT):   # dataGen ends up FIRST
    if _p not in sys.path:
        sys.path.insert(0, _p)


def assert_bridge() -> dict:
    """Import one marker from each root and return their versions."""
    from core.constants import LOG_DT            # dataGen
    from common.constants import PRED_STEPS      # modelTraining
    return {"datagen": DATAGEN_ROOT, "training": TRAINING_ROOT,
            "log_dt": LOG_DT, "pred_steps": PRED_STEPS}
