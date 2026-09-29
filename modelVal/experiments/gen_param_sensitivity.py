"""Simulator-parameter sensitivity datasets (reviewer 2, comment II).

Reviewer 2 observes that several motion, perception and avoidance constants are
simulator design values, and asks what influence they have on the results.  The
honest answer is a measurement: perturb one family of constants at a time,
regenerate an in-distribution test set under the perturbation, and evaluate the
UNCHANGED headline checkpoint on it.  A large metric swing means the reported
numbers are an artefact of that choice; a small one means they are not.

Three perturbations, chosen because each is a different kind of assumption:

    dwell     manoeuvre commitment time, scaled by 0.7 and by 1.3.
              This is the behavioural assumption the difficulty of the dataset
              rests on: shorter dwell means more frequent mode changes and a
              less predictable target.
    notice    the probability that a class notices a developing conflict,
              shifted by -0.2 and +0.2 and clipped to [0.05, 0.99].
              This is the perception assumption inside the traffic, and it
              controls how much interaction structure exists to be learned.
    sensor    range and bearing noise, both multiplied by 2.
              This is the perception assumption on the ownship side, and it is
              the one a real radar is most likely to violate.

``dataGen/`` is frozen, so nothing here edits it.  The constants are patched in
memory, in the already-imported modules, before any scenario is built, and the
patch is recorded verbatim in the run manifest so a reader can tell what the
run actually contained.

    python -m experiments.gen_param_sensitivity --case dwell_lo --episodes 200
    python -m experiments.gen_param_sensitivity --list
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

from scenario.presets import config_for_episode  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from storage.writer import DatasetWriter  # noqa: E402

#: Base split.  test_id is the in-distribution control, so any change measured
#: here is attributable to the perturbation and not to a distribution shift.
BASE_SPLIT = "test_id"
#: Episode indices are shared with the unperturbed test_id run, which makes the
#: comparison paired at the scenario level.
START_INDEX = 0

CASES = {
    "dwell_lo": {"kind": "dwell", "factor": 0.7,
                 "description": "manoeuvre dwell times scaled by 0.7"},
    "dwell_hi": {"kind": "dwell", "factor": 1.3,
                 "description": "manoeuvre dwell times scaled by 1.3"},
    "notice_lo": {"kind": "notice", "delta": -0.2,
                  "description": "conflict notice probability shifted by -0.2"},
    "notice_hi": {"kind": "notice", "delta": +0.2,
                  "description": "conflict notice probability shifted by +0.2"},
    "sensor_2x": {"kind": "sensor", "factor": 2.0,
                  "description": "range and bearing noise multiplied by 2"},
    "baseline": {"kind": "none",
                 "description": "no perturbation; regenerates the control so "
                                "the comparison is like for like"},
}


def _scale_dataclass_fields(obj, names, fn):
    """Return a copy of frozen dataclass ``obj`` with ``names`` mapped by fn."""
    if not dataclasses.is_dataclass(obj):
        return obj, {}
    changed = {}
    over = {}
    for f in dataclasses.fields(obj):
        if f.name in names:
            old = getattr(obj, f.name)
            if isinstance(old, (int, float)):
                new = fn(float(old))
                over[f.name] = new
                changed[f.name] = [old, new]
    if not over:
        return obj, {}
    return dataclasses.replace(obj, **over), changed


def apply_patch(case: str) -> dict:
    """Patch the in-memory constants for ``case`` and return an audit record."""
    spec = CASES[case]
    kind = spec["kind"]
    audit: dict = {"case": case, "kind": kind,
                   "description": spec["description"], "edits": {}}
    if kind == "none":
        return audit

    if kind == "dwell":
        import agents.platforms as P
        factor = float(spec["factor"])
        names = {n for n in dir(P) if "DWELL" in n.upper()}
        for n in sorted(names):
            v = getattr(P, n)
            if isinstance(v, (int, float)):
                setattr(P, n, type(v)(v * factor))
                audit["edits"][f"platforms.{n}"] = [v, getattr(P, n)]
            elif isinstance(v, dict):
                new = {}
                for k, x in v.items():
                    if isinstance(x, (int, float)):
                        new[k] = type(x)(x * factor)
                    elif isinstance(x, (tuple, list)):
                        new[k] = type(x)(type(e)(e * factor)
                                         if isinstance(e, (int, float)) else e
                                         for e in x)
                    else:
                        new[k] = x
                setattr(P, n, new)
                audit["edits"][f"platforms.{n}"] = ["dict", f"x{factor}"]
        # per-platform stability profiles carry their own dwell bands
        prof = getattr(P, "PLATFORM_STABILITY", None) or \
            getattr(P, "STABILITY_PROFILES", None)
        if isinstance(prof, dict):
            fields = {"dwell_min_s", "dwell_max_s", "dwell_mean_s",
                      "dwell_min", "dwell_max"}
            for k, v in list(prof.items()):
                nv, ch = _scale_dataclass_fields(v, fields,
                                                 lambda x: x * factor)
                if ch:
                    prof[k] = nv
                    audit["edits"][f"stability[{k}]"] = ch

    elif kind == "notice":
        import agents.reaction as R
        delta = float(spec["delta"])
        prof = getattr(R, "REACTION_PROFILES", None)
        if prof is None:
            for n in dir(R):
                v = getattr(R, n)
                if isinstance(v, dict) and v and dataclasses.is_dataclass(
                        next(iter(v.values()))):
                    prof = v
                    break
        if isinstance(prof, dict):
            for k, v in list(prof.items()):
                nv, ch = _scale_dataclass_fields(
                    v, {"notice_prob"},
                    lambda x: min(0.99, max(0.05, x + delta)))
                if ch:
                    prof[k] = nv
                    audit["edits"][f"reaction[{k}].notice_prob"] = \
                        ch["notice_prob"]

    elif kind == "sensor":
        # Sensor noise is a field of ScenarioConfig.sensor, not a module
        # constant, so it is applied per episode in cfg_transform() rather
        # than patched here. Record the intent so the audit is complete.
        f = float(spec["factor"])
        audit["edits"] = {
            "sensor.sigma_r0_m": ["x", f],
            "sensor.sigma_r_frac": ["x", f],
            "sensor.sigma_theta_rad": ["x", f],
        }
        return audit

    if not audit["edits"]:
        raise RuntimeError(
            f"case {case!r} patched nothing; the constant names moved. "
            f"Inspect the module rather than generating a run that silently "
            f"reproduces the baseline.")
    return audit


def cfg_transform(case: str, cfg):
    """Apply the perturbations that live in the config rather than in code."""
    spec = CASES[case]
    if spec["kind"] != "sensor":
        return cfg
    f = float(spec["factor"])
    s = cfg.sensor
    return cfg.copy_with(sensor=dataclasses.replace(
        s,
        sigma_r0_m=float(s.sigma_r0_m) * f,
        sigma_r_frac=float(s.sigma_r_frac) * f,
        sigma_theta_rad=float(s.sigma_theta_rad) * f))


def generate(case: str, n: int, tag: str) -> str:
    audit = apply_patch(case)
    writer = DatasetWriter()
    run_dir = writer.open_run(
        cfg_transform(case, config_for_episode(BASE_SPLIT, START_INDEX)),
        preset=f"sens_{case}", tag=tag)
    print(f"case      : {case}  ({audit['description']})")
    print(f"edits     : {len(audit['edits'])}")
    for k, v in list(audit["edits"].items())[:8]:
        print(f"            {k}: {v}")
    print(f"run       : {run_dir}")
    nmac = 0
    try:
        for i in range(n):
            cfg = cfg_transform(
                case, config_for_episode(BASE_SPLIT, START_INDEX + i))
            engine = SimulationEngine(cfg, episode_index=i)
            ep, meta = engine.run_episode()
            writer.write_episode(ep, meta)
            nmac += int(bool(meta.nmac_occurred))
            if (i + 1) % 25 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n}  nmac {nmac}", flush=True)
        writer.add_split(f"sens_{case}", n_episodes=n, seed_base=0,
                         episode_range=[0, n - 1],
                         notes=json.dumps(audit, ensure_ascii=False))
    except BaseException:
        writer.close(status="aborted")
        raise
    writer.close()
    with open(os.path.join(run_dir, "sensitivity_patch.json"), "w",
              encoding="utf-8") as f:
        json.dump(audit, f, indent=1, ensure_ascii=False)
    print(f"done: {n} episodes ({nmac} with an NMAC) -> {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--tag", default="sens")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k in sorted(CASES):
            print(f"{k:12s} {CASES[k]['description']}")
        return
    if not args.case:
        ap.error("--case is required unless --list is given")
    generate(args.case, int(args.episodes), args.tag)


if __name__ == "__main__":
    main()
