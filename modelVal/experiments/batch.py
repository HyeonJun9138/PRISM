"""Headless planner-utility batch: (risk arm x episode) closed loop.

Design notes (paper experiment section mirrors these):
  * The PLANNER IS FROZEN (controller.py config); only the RiskProvider
    differs between arms. Episode indices are IDENTICAL across arms, so
    every comparison is paired on the same traffic.
  * Episode indices start at 10000 -- inside the test_ood seed block but
    far beyond the dataset's generated range, so scenarios come from the
    OOD distribution without reusing evaluation episodes.
  * Safety is scored on GROUND TRUTH separations (the simulator's world),
    never on observations: sensors belong to the system under test.
  * 'unavoidable' flag: an NMAC whose intruder was first DETECTED with
    less lead time than a full-rate lateral displacement of one NMAC
    radius needs (t_need = sqrt(2*R_nmac/a_lat_max)); reported separately
    so late-detection physics does not dilute the arm comparison.

Usage:
  python -m experiments.batch --arm cv --episodes 100 --out <dir>
  (run one process per arm; only --arm model touches the GPU)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

import numpy as np  # noqa: E402

from core.constants import (DWC_HMD_M, LOG_DECIMATION,  # noqa: E402
                            NMAC_RADIUS_M)
from scenario.presets import config_for_episode  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402

from planner.controller import PlannerPilot  # noqa: E402
from planner.risk import make_risk  # noqa: E402

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")

EP_KEYS = ("gt_pos", "gt_valid", "gt_type", "gt_intent",
           "obs_detected", "obs_pos", "obs_R", "obs_track_age", "own_pos")
BASE_SPLIT = "test_ood"
INDEX_OFFSET = 10_000
DURATION_S = 90.0
MODEL_PATH = f"{PRISM_ROOT}/Logs/Models/20260813_170429_full5x_s4/final/model_final.pt"


def run_episode(arm: str, index: int, predictor,
                overrides: dict | None = None,
                risk_kw: dict | None = None) -> dict:
    cfg = config_for_episode(BASE_SPLIT, INDEX_OFFSET + index)
    over = dict(overrides or {})
    enc = {k: over.pop(k) for k in ("n_designed_min", "n_designed_max")
           if k in over}
    if enc:
        from dataclasses import replace as dc_replace
        over["encounter"] = dc_replace(cfg.encounter, **enc)
    cfg = cfg.copy_with(duration_s=DURATION_S,
                        name=f"batch/{arm}/{index:05d}", **over)

    pilot = PlannerPilot(make_risk(arm, **(risk_kw or {})))
    engine = SimulationEngine(cfg, ownship_controller=pilot.command)
    scn = engine.scenario
    pilot.bind(scn.world, scn.ownship.envelope, scn.ownship_track)
    ep = engine.ep

    sep_series: list[np.ndarray] = []      # per logged frame: [N] gt dist
    own_xy: list[tuple[float, float]] = []
    omega_series: list[float] = []
    v_series: list[float] = []
    plan_ms: list[float] = []
    infer_ms: list[float] = []
    first_det: dict[int, float] = {}       # slot -> sim t of first detection
    t_series: list[float] = []

    wall0 = time.perf_counter()
    while engine.step(log=True):
        if engine._tick % LOG_DECIMATION != 0:
            continue
        t0 = engine._log_index - 1
        if t0 < 0:
            continue
        ep_dict = {k: getattr(ep, k) for k in EP_KEYS}

        preds = {}
        if predictor is not None and predictor.loaded:
            t_inf = time.perf_counter()
            preds = predictor.predict_frame(ep_dict, t0)
            infer_ms.append((time.perf_counter() - t_inf) * 1000.0)

        pilot.replan(engine.t, ep_dict, t0, preds, scn.ownship.state)
        plan_ms.append(pilot.plan_ms)

        own = scn.ownship.state
        dists = np.full(len(scn.agents), np.inf)
        for slot, ag in enumerate(scn.agents):
            if not ag.alive:
                continue
            dists[slot] = math.hypot(ag.state.x - own.x, ag.state.y - own.y)
            if bool(ep.obs_detected[t0, slot]) and slot not in first_det:
                first_det[slot] = engine.t
        sep_series.append(dists)
        own_xy.append((own.x, own.y))
        omega_series.append(abs(math.degrees(own.omega)))
        v_series.append(own.v)
        t_series.append(engine.t)

    wall_s = time.perf_counter() - wall0
    sep = np.asarray(sep_series)                      # [F, N]
    ts = np.asarray(t_series)
    min_by_frame = sep.min(axis=1)
    log_dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.5

    # ---- NMAC events: contiguous frames below the ring, per nearest slot
    events = []
    inside = min_by_frame < NMAC_RADIUS_M
    i = 0
    t_need = math.sqrt(2.0 * NMAC_RADIUS_M / 4.5)     # ~8.2 s full-rate
    while i < len(inside):
        if not inside[i]:
            i += 1
            continue
        j = i
        while j < len(inside) and inside[j]:
            j += 1
        k = int(np.argmin(min_by_frame[i:j])) + i
        slot = int(np.argmin(sep[k]))
        det_t = first_det.get(slot)
        lead = (ts[k] - det_t) if det_t is not None else 0.0
        events.append({
            "t": round(float(ts[k]), 1),
            "slot": slot,
            "min_sep": round(float(min_by_frame[k]), 1),
            "duration_s": round((j - i) * log_dt, 1),
            "detect_lead_s": round(float(lead), 1),
            "unavoidable": bool(lead < t_need),
        })
        i = j

    # ---- path economy: flown length vs corridor progress
    xy = np.asarray(own_xy)
    path_len = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 \
        else 0.0
    progress = 0.0
    if pilot.path is not None and len(xy) > 1:
        progress = (pilot.path.project(*xy[-1])["s"]
                    - pilot.path.project(*xy[0])["s"])
    elongation = (path_len / progress - 1.0) if progress > 100.0 else None

    # ---- LoWC events (same contiguous-segment rule as NMAC)
    lowc_inside = min_by_frame < DWC_HMD_M
    lowc_events = 0
    i = 0
    while i < len(lowc_inside):
        if lowc_inside[i]:
            lowc_events += 1
            while i < len(lowc_inside) and lowc_inside[i]:
                i += 1
        else:
            i += 1

    om = np.asarray(omega_series)
    return {
        "arm": arm,
        "index": index,
        # written on every row so a batch is self-describing: earlier runs
        # carry no such field and are all sigma_scale == 1.0
        "sigma_scale": float((risk_kw or {}).get("sigma_scale", 1.0)),
        "seed": int(cfg.seed),
        "n_agents": int(sum(1 for a in scn.agents)),
        "sim_s": round(float(ts[-1]), 1) if len(ts) else 0.0,
        "wall_s": round(wall_s, 1),
        "min_sep": round(float(min_by_frame.min()), 1),
        "sep_p5": round(float(np.percentile(min_by_frame, 5)), 1),
        "sep_median": round(float(np.median(min_by_frame)), 1),
        "nmac_events": len(events),
        "nmac_events_avoidable": sum(1 for e in events
                                     if not e["unavoidable"]),
        "nmac_duration_s": round(float(inside.sum() * log_dt), 1),
        "lowc_fraction": round(float((min_by_frame < DWC_HMD_M).mean()), 3),
        "lowc_events": lowc_events,
        "path_len_m": round(path_len, 0),
        "corridor_progress_m": round(float(progress), 0),
        "path_elongation": (round(elongation, 4)
                            if elongation is not None else None),
        "abs_omega_mean": round(float(om.mean()), 2),
        "abs_omega_p95": round(float(np.percentile(om, 95)), 2),
        "v_mean": round(float(np.mean(v_series)), 1),
        "v_std": round(float(np.std(v_series)), 2),
        "plan_ms_mean": round(float(np.mean(plan_ms)), 1),
        "plan_ms_p95": round(float(np.percentile(plan_ms, 95)), 1),
        "infer_ms_mean": (round(float(np.mean(infer_ms)), 1)
                          if infer_ms else None),
        "events": events,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("none", "cv", "model"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--n-min", type=int, default=None)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--designed-min", type=int, default=None)
    ap.add_argument("--designed-max", type=int, default=None)
    ap.add_argument("--sigma-scale", type=float, default=1.0,
                    help="post-hoc temperature applied as sigma -> T*sigma "
                         "in the model arm (1.0 = raw network sigma; 1.665 = "
                         "the value fitted by experiments/recalibrate.py on "
                         "the OOD-adjacent calibration runs)")
    args = ap.parse_args()

    if args.sigma_scale != 1.0 and args.arm != "model":
        ap.error("--sigma-scale only applies to the model arm")
    risk_kw = ({"sigma_scale": args.sigma_scale}
               if args.arm == "model" else {})

    overrides = {}
    if args.density is not None:
        overrides["density_per_km2"] = args.density
    if args.n_min is not None:
        overrides["n_agents_min"] = args.n_min
    if args.n_max is not None:
        overrides["n_agents_max"] = args.n_max
    if args.designed_min is not None:
        overrides["n_designed_min"] = args.designed_min
    if args.designed_max is not None:
        overrides["n_designed_max"] = args.designed_max

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.arm}.jsonl")

    predictor = None
    if args.arm == "model":
        from runtime.predictor import LivePredictor
        predictor = LivePredictor()
        info = predictor.load(args.model)
        print(f"[{args.arm}] model {info['path']} ({info['device']})",
              flush=True)

    done = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(args.start, args.start + args.episodes):
            try:
                row = run_episode(args.arm, i, predictor, overrides, risk_kw)
            except Exception as e:  # noqa: BLE001 - keep the batch alive
                row = {"arm": args.arm, "index": i,
                       "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 10 == 0 or done == args.episodes:
                print(f"[{args.arm}] {done}/{args.episodes} episodes",
                      flush=True)
    print(f"[{args.arm}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
