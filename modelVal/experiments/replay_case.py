"""Replay one paired batch episode per arm, recording trajectories.

Produces the case-study data for the '실제 회피' figure: same traffic,
different risk input, different outcome. Deterministic re-run of the
batch protocol (config_for_episode(test_ood, 10000+index)).

  python -m experiments.replay_case --index 244 --threat 22
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

import numpy as np  # noqa: E402

from core.constants import LOG_DECIMATION  # noqa: E402
from scenario.presets import config_for_episode  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402

from planner.controller import PlannerPilot  # noqa: E402
from planner.risk import make_risk  # noqa: E402
from experiments.batch import (EP_KEYS, BASE_SPLIT, INDEX_OFFSET,  # noqa: E402
                               DURATION_S, MODEL_PATH)

PLAN_SNAP_T = (55.0, 62.0, 68.0)      # plan polylines near the encounter


def run(arm, index, threat_slot, predictor):
    cfg = config_for_episode(BASE_SPLIT, INDEX_OFFSET + index)
    cfg = cfg.copy_with(duration_s=DURATION_S, name=f"case/{arm}/{index}")
    pilot = PlannerPilot(make_risk(arm))
    engine = SimulationEngine(cfg, ownship_controller=pilot.command)
    scn = engine.scenario
    pilot.bind(scn.world, scn.ownship.envelope, scn.ownship_track)
    ep = engine.ep

    rec = {"arm": arm, "t": [], "own": [], "threat": [], "sep": [],
           "plans": {}, "others": {}}
    while engine.step(log=True):
        if engine._tick % LOG_DECIMATION != 0:
            continue
        t0 = engine._log_index - 1
        if t0 < 0:
            continue
        ep_dict = {k: getattr(ep, k) for k in EP_KEYS}
        preds = {}
        if predictor is not None and predictor.loaded:
            preds = predictor.predict_frame(ep_dict, t0)
        pilot.replan(engine.t, ep_dict, t0, preds, scn.ownship.state)

        own = scn.ownship.state
        rec["t"].append(round(engine.t, 1))
        rec["own"].append([round(own.x, 1), round(own.y, 1)])
        th = scn.agents[threat_slot]
        rec["threat"].append([round(th.state.x, 1), round(th.state.y, 1)]
                             if th.alive else None)
        sep = min((math.hypot(a.state.x - own.x, a.state.y - own.y)
                   for a in scn.agents if a.alive), default=float("inf"))
        rec["sep"].append(round(sep, 1))
        for ts in PLAN_SNAP_T:
            if abs(engine.t - ts) < 0.26 and str(ts) not in rec["plans"]:
                w = pilot.wire(engine.t)
                rec["plans"][str(ts)] = w.get("poly", [])
        # sample a few other agents sparsely for context
        if t0 % 4 == 0:
            for s, a in enumerate(scn.agents):
                if a.alive and s != threat_slot:
                    rec["others"].setdefault(str(s), []).append(
                        [round(a.state.x), round(a.state.y)])
    rec["min_sep"] = min(rec["sep"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--threat", type=int, required=True)
    args = ap.parse_args()

    from runtime.predictor import LivePredictor
    predictor = LivePredictor()
    predictor.load(MODEL_PATH)

    out = {"index": args.index, "threat_slot": args.threat, "arms": {}}
    for arm in ("none", "model"):
        pred = predictor if arm == "model" else None
        out["arms"][arm] = run(arm, args.index, args.threat, pred)
        print(arm, "min_sep", out["arms"][arm]["min_sep"], flush=True)

    dst = rf"{PRISM_ROOT}/results/case_ep{args.index}.json"
    json.dump(out, open(dst, "w", encoding="utf-8"))
    print("saved ->", dst)


if __name__ == "__main__":
    main()
#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

