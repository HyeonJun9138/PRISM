"""Per-step reaction-layer labels by deterministic replay of dataset episodes.

R3 major 7 asks to compare the full and no-neighbour predictors "on windows
containing an avoidance response within the forecast horizon". The dataset
stores only a per-episode count of armed reactions (index.jsonl
``reaction_events``), so this script REPLAYS each episode with dataGen's own
SimulationEngine (read-only use of the frozen dataGen tree) and records, per
logged step and agent slot, the state of the agent's reaction layer
(dataGen/agents/reaction.py, state dict travelling under REACTION_STATE_KEY):
idle -> delay -> latched -> cooldown -> idle. The engine arms it in its 2 Hz
perception pass (simulation/engine.py step(), ticks with tick % 10 == 0);
Agent.step (agents/base.py) then runs ReactionLayer.tick() followed by
.apply() every 0.05 s tick, and only the LATCHED phase blends the evasive
turn into the command stream.

Label arrays (per logged step i = interval (t_i - LOG_DT, t_i], 10 ticks)
  reaction_active [T,N] bool  the layer of slot n was LATCHED (evasive turn
                              applied) after at least one tick of the interval
                              while the agent was alive
  reaction_armed  [T,N] bool  a reaction was armed (idle -> delay) by the
                              perception pass of the interval; the episode sum
                              must equal index.jsonl ``reaction_events``
  reaction_phase  [T,N] int8  phase at the logged instant: -1 no layer
                              (DEBRIS) or not alive, 0 idle, 1 delay,
                              2 latched, 3 cooldown
  reaction_threat [T,N] int16 threat of a non-idle layer at the logged
                              instant: >= 0 agent slot, -1 ownship, -2 none
  t [T] float32, gt_valid [T,N], gt_type [N]  copied from the replay (equal to
                              the logged file by construction of the check)

Reproduction: configs are rebuilt the way the generator built them --
dataGen/main.py ``_cmd_generate`` / ``_generate_split``:
``scenario.presets.config_for_episode(preset, episode_index)`` and
``SimulationEngine(cfg, episode_index=i)`` with the default ownship controller;
factor-wise OOD runs (preset ``ood_<factor>``) use
modelVal/experiments/gen_factor_ood.config_for. The episode index and master
seed come from index.jsonl (the seed must equal the rebuilt config's). EVERY
replayed episode is compared with its logged file -- gt_pos max |diff| == 0
on finite entries with an identical NaN pattern, plus t, gt_valid,
obs_detected and obs_track_id -- and the armed count with index.jsonl; a
failing episode is recorded as a mismatch and gets NO label file.

Paths: nothing but the two defaults below is hard-coded. dataGen is found as
<PRISM root>/dataGen next to modelVal, or --datagen-root / PRISM_DATAGEN_ROOT.

  python -m experiments.reaction_labels --run 20260813_023139_test_id_full5x
         [--dataset-root $PRISM_ROOT/Logs/Dataset]
         [--out-root $PRISM_ROOT/Logs/Analysis/reaction_labels]
         [--workers 16] [--episodes N] [--start S] [--every K] [--resume]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELVAL_ROOT = os.path.dirname(_HERE)
REPO_ROOT = os.path.dirname(MODELVAL_ROOT)      # dataGen lives next to modelVal
#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", REPO_ROOT).replace("\\", "/")
DEFAULT_DATASET_ROOT = f"{PRISM_ROOT}/Logs/Dataset"
DEFAULT_OUT_ROOT = f"{PRISM_ROOT}/Logs/Analysis/reaction_labels"
LABEL_VERSION = 1
PHASE_CODES = {"idle": 0, "delay": 1, "latched": 2, "cooldown": 3}
PHASE_NAMES = {v: k for k, v in PHASE_CODES.items()}
#: sample-window geometry used for the window statistic (modelTraining
#: common.constants: HIST_STEPS 20, PRED_STEPS 30, training t0 stride 4)
HIST_STEPS, PRED_STEPS, WINDOW_STRIDE = 20, 30, 4


def setup_datagen(root: str | None = None) -> str:
    """Put dataGen on sys.path (first) and return its absolute root."""
    root = os.path.abspath(root or os.environ.get("PRISM_DATAGEN_ROOT")
                           or os.path.join(REPO_ROOT, "dataGen"))
    if not os.path.isdir(os.path.join(root, "simulation")):
        raise FileNotFoundError(f"no dataGen tree at {root}")
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    if MODELVAL_ROOT not in sys.path:
        sys.path.append(MODELVAL_ROOT)
    return root


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================ run metadata

def resolve_run(spec: str, dataset_root: str) -> str:
    cand = spec if os.path.isdir(spec) else os.path.join(dataset_root, spec)
    if not os.path.isdir(cand):
        raise SystemExit(f"run not found: {spec} (dataset root {dataset_root})")
    return os.path.abspath(cand).replace("\\", "/")


def read_index(run_dir: str) -> list[dict]:
    rows = []
    with open(os.path.join(run_dir, "index.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def config_kind(preset: str) -> tuple[str, str]:
    """('preset', name) for dataGen splits, ('factor', axis) for
    gen_factor_ood runs; raises for anything this script cannot rebuild."""
    from scenario.presets import PRESET_NAMES
    if preset in PRESET_NAMES:
        return "preset", preset
    if preset.startswith("ood_"):
        from experiments.gen_factor_ood import FACTORS
        if preset[4:] in FACTORS:
            return "factor", preset[4:]
    raise ValueError(f"preset {preset!r}: episode configs of this run cannot "
                     "be rebuilt (not a dataGen split and not a factor-wise "
                     "OOD run)")


def build_config(kind: str, name: str, index: int):
    if kind == "preset":
        from scenario.presets import config_for_episode
        return config_for_episode(name, int(index))
    if kind == "factor":
        from experiments.gen_factor_ood import config_for
        return config_for(name, int(index))
    raise ValueError(kind)


def check_manifest(man: dict, kind: str, name: str) -> dict:
    """Does the manifest's run config equal the config the generator passed
    to open_run? (make_preset(preset) / config_for(factor, 0)); a difference
    means overrides this script does not know about."""
    if kind == "preset":
        from scenario.presets import make_preset
        ref = make_preset(name).to_dict()
    else:
        from experiments.gen_factor_ood import config_for
        ref = config_for(name, 0).to_dict()
    ref = json.loads(json.dumps(ref))
    got = man.get("config", {})
    diff = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
    return {"matches_rebuild": not diff, "differing_keys": diff}


# ================================================================== replay

def _phase(layer) -> int:
    return PHASE_CODES[str(layer.state["phase"])]


def replay_episode(task: dict) -> dict:
    """Replay one episode, verify it against the logged file, write labels.

    Returns a result dict; ``ok`` False means mismatch (no file written).
    """
    t_start = time.perf_counter()
    setup_datagen(task["datagen_root"])
    from core.constants import LOG_DECIMATION
    from simulation.engine import SimulationEngine

    idx = int(task["episode_index"])
    res = {"episode_index": idx, "seed": int(task["seed"]), "ok": False}
    cfg = build_config(task["kind"], task["name"], idx)
    if int(cfg.seed) != int(task["seed"]):
        res["reason"] = f"seed: rebuilt {cfg.seed} != index {task['seed']}"
        return res
    if task.get("duration_s") is not None and \
            float(cfg.duration_s) != float(task["duration_s"]):
        res["reason"] = (f"duration: rebuilt {cfg.duration_s} != manifest "
                         f"{task['duration_s']}")
        return res

    engine = SimulationEngine(cfg, episode_index=idx)
    T, N = int(engine.n_steps), int(engine.n_agents)
    agents = engine.scenario.agents
    slot_of = {int(a.agent_id): s for s, a in enumerate(agents)}
    active = np.zeros((T, N), dtype=bool)
    armed = np.zeros((T, N), dtype=bool)
    phase_log = np.full((T, N), -1, dtype=np.int8)
    threat_log = np.full((T, N), -2, dtype=np.int16)
    prev = [(_phase(a.reaction) if a.reaction is not None else -1)
            for a in agents]

    while not engine.is_done:
        j = engine.n_logged                    # frame this tick belongs to
        engine.step(log=True)
        for s, ag in enumerate(agents):
            layer = ag.reaction
            if layer is None:
                continue
            code = _phase(layer)
            if prev[s] == 0 and code != 0:     # only perceive() leaves idle
                armed[j, s] = True
            if code == 2 and ag.alive:
                active[j, s] = True
            prev[s] = code
        if engine.tick % LOG_DECIMATION == 0:
            for s, ag in enumerate(agents):
                layer = ag.reaction
                if layer is None or not ag.alive:
                    continue
                code = _phase(layer)
                phase_log[j, s] = code
                if code != 0:
                    tid = int(layer.state["threat_id"])
                    threat_log[j, s] = -1 if tid == -1 else slot_of.get(tid, -3)
    t_sim = time.perf_counter() - t_start

    # ------------------------------------------------ verify against the log
    ep = engine.ep
    with np.load(task["episode_path"]) as z:
        logged = {k: z[k] for k in ("t", "gt_pos", "gt_valid", "obs_detected",
                                    "obs_track_id")}
    problems = []
    if logged["gt_pos"].shape != ep.gt_pos.shape:
        problems.append(f"gt_pos shape {ep.gt_pos.shape} != logged "
                        f"{logged['gt_pos'].shape}")
    else:
        fin_r = np.isfinite(ep.gt_pos)
        fin_l = np.isfinite(logged["gt_pos"])
        n_nan_diff = int((fin_r != fin_l).sum())
        both = fin_r & fin_l
        max_diff = (float(np.abs(ep.gt_pos[both].astype(np.float64)
                                 - logged["gt_pos"][both]).max())
                    if both.any() else 0.0)
        res["gt_pos_max_abs_diff"] = max_diff
        res["gt_pos_nan_pattern_diff"] = n_nan_diff
        if n_nan_diff:
            problems.append(f"gt_pos NaN pattern differs at {n_nan_diff} entries")
        if max_diff != 0.0:
            problems.append(f"gt_pos max |diff| = {max_diff!r}")
        for k in ("t", "gt_valid", "obs_detected", "obs_track_id"):
            if not np.array_equal(getattr(ep, k), logged[k]):
                problems.append(f"{k} differs")
    n_armed = int(armed.sum())
    res["arms"] = n_armed
    res["arms_engine"] = int(engine.reaction_events)
    res["arms_index"] = task.get("reaction_events")
    if n_armed != int(engine.reaction_events):
        problems.append(f"armed-label count {n_armed} != engine counter "
                        f"{engine.reaction_events}")
    if task.get("reaction_events") is not None and \
            int(engine.reaction_events) != int(task["reaction_events"]):
        problems.append(f"engine reaction_events {engine.reaction_events} != "
                        f"index {task['reaction_events']}")
    if problems:
        res["reason"] = "; ".join(problems)
        res["wall_s"] = round(time.perf_counter() - t_start, 3)
        return res

    arrays = {"reaction_active": active, "reaction_armed": armed,
              "reaction_phase": phase_log, "reaction_threat": threat_log,
              "t": np.asarray(ep.t, dtype=np.float32),
              "gt_valid": np.asarray(ep.gt_valid, dtype=bool),
              "gt_type": np.asarray(ep.gt_type, dtype=np.int8),
              "episode_index": np.int64(idx), "seed": np.int64(task["seed"]),
              "label_version": np.int64(LABEL_VERSION)}
    if task.get("out_path"):
        tmp = task["out_path"] + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, task["out_path"])
    res["ok"] = True
    res["stats"] = label_stats(active, phase_log, threat_log, armed,
                               arrays["gt_valid"], arrays["gt_type"])
    res["sim_s"] = round(t_sim, 3)
    res["wall_s"] = round(time.perf_counter() - t_start, 3)
    if task.get("return_arrays"):
        res["arrays"] = arrays
    return res


# ================================================================== stats

N_TYPES = 7


def label_stats(active, phase, threat, armed, gt_valid, gt_type) -> dict:
    """Counts (per platform type, index = PlatformType value) from one
    episode's label arrays; summed over episodes by the caller."""
    T, N = active.shape
    types = np.asarray(gt_type, dtype=np.int64)
    alive = np.asarray(gt_valid, dtype=bool)
    per = {k: [0] * N_TYPES for k in ("agents", "alive_agent_steps",
                                      "active_agent_steps", "arms",
                                      "windows", "windows_with_active")}
    ph_counts = {k: 0 for k in ("no_layer", "idle", "delay", "latched",
                                "cooldown")}
    lat_threat = {"ownship": 0, "agent": 0, "unknown": 0}
    t0s = list(range(HIST_STEPS - 1, T - PRED_STEPS, WINDOW_STRIDE))
    for n in range(N):
        ty = int(types[n])
        if ty < 0 or ty >= N_TYPES:
            continue
        per["agents"][ty] += int(alive[:, n].any())
        per["alive_agent_steps"][ty] += int(alive[:, n].sum())
        per["active_agent_steps"][ty] += int((active[:, n] & alive[:, n]).sum())
        per["arms"][ty] += int(armed[:, n].sum())
        for t0 in t0s:
            if not alive[t0, n]:
                continue
            per["windows"][ty] += 1
            per["windows_with_active"][ty] += int(
                active[t0 + 1:t0 + 1 + PRED_STEPS, n].any())
    ph = np.asarray(phase)[alive]
    ph_counts["no_layer"] = int((ph == -1).sum())
    for name, code in PHASE_CODES.items():
        ph_counts[name] = int((ph == code).sum())
    lat = (np.asarray(phase) == 2) & alive
    thr = np.asarray(threat)[lat]
    lat_threat["ownship"] = int((thr == -1).sum())
    lat_threat["agent"] = int((thr >= 0).sum())
    lat_threat["unknown"] = int((thr < -1).sum())
    return {"per_type": per, "phase_at_logged_instants": ph_counts,
            "latched_threat": lat_threat}


def _merge_stats(acc: dict | None, s: dict) -> dict:
    if acc is None:
        return json.loads(json.dumps(s))
    for k, v in s["per_type"].items():
        acc["per_type"][k] = [a + b for a, b in zip(acc["per_type"][k], v)]
    for grp in ("phase_at_logged_instants", "latched_threat"):
        for k, v in s[grp].items():
            acc[grp][k] += v
    return acc


def summarise(acc: dict | None) -> dict:
    from core.types import PlatformType
    if acc is None:
        return {}
    per = acc["per_type"]
    tot = {k: int(sum(v)) for k, v in per.items()}

    def frac(a, b):
        return round(a / b, 6) if b else None

    out = {
        "alive_agent_steps": tot["alive_agent_steps"],
        "active_agent_steps": tot["active_agent_steps"],
        "active_fraction_of_alive_agent_steps": frac(tot["active_agent_steps"],
                                                     tot["alive_agent_steps"]),
        "arms": tot["arms"],
        "windows_stride4": {
            "definition": "(t0, agent) with t0 in range(19, T-30, 4) and the "
                          "agent alive at t0; 'with_active' = reaction_active "
                          "anywhere in t0+1..t0+30 (the 15 s horizon). Not "
                          "filtered by the predictor's detection eligibility.",
            "n_windows": tot["windows"],
            "n_with_active": tot["windows_with_active"],
            "fraction": frac(tot["windows_with_active"], tot["windows"])},
        "per_platform": {},
        "phase_fraction_at_logged_instants": {
            k: frac(v, tot["alive_agent_steps"])
            for k, v in acc["phase_at_logged_instants"].items()},
        "latched_threat_at_logged_instants": acc["latched_threat"],
    }
    for ty in range(N_TYPES):
        if not per["alive_agent_steps"][ty]:
            continue
        out["per_platform"][PlatformType(ty).name] = {
            "agents": per["agents"][ty],
            "alive_agent_steps": per["alive_agent_steps"][ty],
            "active_agent_steps": per["active_agent_steps"][ty],
            "active_fraction": frac(per["active_agent_steps"][ty],
                                    per["alive_agent_steps"][ty]),
            "arms": per["arms"][ty],
            "windows_stride4": per["windows"][ty],
            "windows_with_active": per["windows_with_active"][ty],
            "window_fraction": frac(per["windows_with_active"][ty],
                                    per["windows"][ty])}
    return out


def stats_from_file(path: str) -> dict:
    with np.load(path) as z:
        return label_stats(z["reaction_active"], z["reaction_phase"],
                           z["reaction_threat"], z["reaction_armed"],
                           z["gt_valid"], z["gt_type"])


# ==================================================================== main

def process_run(run_dir: str, args, datagen_root: str) -> dict:
    from storage.manifest import source_fingerprint

    name = os.path.basename(run_dir.rstrip("/"))
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    kind, cname = config_kind(str(man["preset"]))
    mcheck = check_manifest(man, kind, cname)
    fp = source_fingerprint(datagen_root)
    man_fp = man.get("source", {}).get("sha256")
    rows = read_index(run_dir)
    sel = rows[args.start:]
    if args.every > 1:
        sel = sel[::args.every]
    if args.episodes is not None:
        sel = sel[:args.episodes]
    out_dir = os.path.join(args.out_root, name).replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, args.summary_name)
    if os.path.exists(summary_path) and not (args.overwrite or args.resume):
        raise SystemExit(f"{summary_path} exists; pass --resume or --overwrite")
    duration = man.get("timing", {}).get("duration_s")

    print(f"run {name}: preset {man['preset']} ({kind}:{cname}), "
          f"{len(sel)}/{len(rows)} episodes, manifest config matches rebuild: "
          f"{mcheck['matches_rebuild']}, dataGen fingerprint "
          f"{'==' if fp['sha256'] == man_fp else '!='} manifest", flush=True)

    tasks, resumed = [], []
    for r in sel:
        idx = int(r["episode_index"])
        ep_path = os.path.join(run_dir, "episodes", f"ep_{idx:06d}.npz")
        out_path = os.path.join(out_dir, f"ep_{idx:06d}.npz")
        if os.path.exists(out_path):
            if args.resume:
                resumed.append((idx, out_path, r))
                continue
            if not args.overwrite:
                raise SystemExit(f"{out_path} exists; pass --resume or --overwrite")
        tasks.append({"datagen_root": datagen_root, "kind": kind, "name": cname,
                      "episode_index": idx, "seed": int(r["seed"]),
                      "reaction_events": r.get("reaction_events"),
                      "duration_s": duration, "episode_path": ep_path,
                      "out_path": out_path})

    t_wall = time.time()
    results = []
    if args.workers > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(replay_episode, tasks,
                                                        chunksize=1)):
                results.append(res)
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    print(f"  {i + 1}/{len(tasks)} replayed, "
                          f"{sum(not x['ok'] for x in results)} mismatches, "
                          f"{time.time() - t_wall:.0f}s", flush=True)
    else:
        for i, task in enumerate(tasks):
            results.append(replay_episode(task))
            print(f"  {i + 1}/{len(tasks)} ep {task['episode_index']}: "
                  f"{'ok' if results[-1]['ok'] else 'MISMATCH'} "
                  f"{results[-1].get('wall_s')}s", flush=True)
    wall = time.time() - t_wall

    acc = None
    for res in results:
        if res["ok"]:
            acc = _merge_stats(acc, res["stats"])
    for idx, path, r in resumed:
        acc = _merge_stats(acc, stats_from_file(path))
    results.sort(key=lambda x: x["episode_index"])
    ok = [x for x in results if x["ok"]]
    walls = np.array([x["wall_s"] for x in ok], dtype=float)
    summary = {
        "run": name, "run_dir": run_dir, "preset": man.get("preset"),
        "tag": man.get("tag"), "master_seed": man.get("master_seed"),
        "config_rebuild": f"{kind}:{cname}", "manifest_check": mcheck,
        "duration_s": duration,
        "datagen_root": datagen_root.replace("\\", "/"),
        "datagen_fingerprint": fp.get("sha256"),
        "manifest_fingerprint": man_fp,
        "fingerprint_match": fp.get("sha256") == man_fp,
        "selection": {"start": args.start, "every": args.every,
                      "episodes": args.episodes, "n_selected": len(sel),
                      "n_in_run": len(rows),
                      "episode_indices": [int(r["episode_index"]) for r in sel]
                      if len(sel) != len(rows) else "all"},
        "n_replayed": len(results), "n_resumed": len(resumed),
        "n_ok": len(ok) + len(resumed),
        "n_mismatch": sum(not x["ok"] for x in results),
        "mismatches": [{k: x.get(k) for k in ("episode_index", "seed", "reason")}
                       for x in results if not x["ok"]],
        "verification": {
            "gt_pos_max_abs_diff_max": (max(x["gt_pos_max_abs_diff"] for x in ok)
                                        if ok else None),
            "gt_pos_nan_pattern_diff_total": (sum(x["gt_pos_nan_pattern_diff"]
                                                  for x in ok) if ok else None),
            "armed_equals_index_reaction_events": sum(
                1 for x in ok if x["arms"] == x["arms_index"]),
            "also_compared": ["t", "gt_valid", "obs_detected", "obs_track_id"]},
        **summarise(acc),
        "timing": {
            "workers": int(args.workers), "wall_s": round(wall, 1),
            "per_episode_wall_s_mean": (round(float(walls.mean()), 3)
                                        if walls.size else None),
            "per_episode_wall_s_median": (round(float(np.median(walls)), 3)
                                          if walls.size else None),
            "per_episode_wall_s_max": (round(float(walls.max()), 3)
                                       if walls.size else None),
            "per_episode_sim_s_mean": (round(float(np.mean([x["sim_s"] for x in ok])), 3)
                                       if ok else None)},
        "definitions": {
            "reaction_active": "reaction layer LATCHED (evasive turn applied) "
                               "after >= 1 integration tick of the logging "
                               "interval (t-0.5 s, t], agent alive",
            "reaction_armed": "reaction armed (idle -> delay) by the perception "
                              "pass at the start of the interval",
            "reaction_phase": "-1 no layer/not alive, 0 idle, 1 delay, "
                              "2 latched, 3 cooldown (at the logged instant)",
            "reaction_threat": ">=0 agent slot, -1 ownship, -2 none"},
        "label_version": LABEL_VERSION,
        "script_sha256": _sha256(os.path.abspath(__file__)),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"  ok {summary['n_ok']}, mismatches {summary['n_mismatch']}, "
          f"active fraction {summary.get('active_fraction_of_alive_agent_steps')}"
          f", wall {wall:.1f}s -> {summary_path}", flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True,
                    help="run directory name (under --dataset-root) or path; "
                         "repeatable")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--datagen-root", default=None,
                    help="dataGen tree (default: <PRISM root>/dataGen next to "
                         "modelVal, or $PRISM_DATAGEN_ROOT)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first S episodes (index order)")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every K-th episode after --start (a documented "
                         "subset)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="at most N episodes after --start/--every")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes whose label file exists (their stats "
                         "are recomputed from the file)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-name", default="summary.json")
    args = ap.parse_args(argv)
    args.out_root = os.path.abspath(args.out_root).replace("\\", "/")
    root = setup_datagen(args.datagen_root)
    os.environ["PRISM_DATAGEN_ROOT"] = root
    bad = 0
    for spec in args.run:
        s = process_run(resolve_run(spec, args.dataset_root), args, root)
        bad += s["n_mismatch"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
