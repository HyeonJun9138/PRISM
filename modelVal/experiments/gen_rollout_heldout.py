"""Held-out branch-rollout run for the R3 rollout diagnostics (majors 1/2).

Why: the released OOD rollout set (20260814_101541_test_ood_rollout_eval,
test_ood episodes 0-99, master seeds 300000-300099) is bit-identical to
test_ood_cal_a -- episodes the recalibration temperature was fitted on -- so
any post-scaling rollout diagnostic on it is in-sample. This script writes a
FRESH rollout run, by default test_ood episode indices 1200-1299 (master
seeds 301200-301299): inside the test_ood seed block, after the 1200 OOD
evaluation episodes (0-1199) and therefore also after the 300 calibration
episodes (0-299). Before anything is generated it proves this: every seed is
in the preset's block, and no run under --dataset-root or --out-root holds
any of them (all index.jsonl seeds are read).

Generation path = dataGen ``main.py generate --preset P --episodes N
--start-index S --tag T --rollouts K --branch-t B`` (``_cmd_generate``), with
the frozen dataGen tree used read-only: per episode
``scenario.presets.config_for_episode(P, i)`` ->
``SimulationEngine(cfg, episode_index=i).run_episode()`` ->
``DatasetWriter.write_episode`` -> dataGen's own ``_write_branch_rollouts``
(reset, deterministic replay to B, K branches with re-seeded decision
streams). The only change is parallelism: workers run the engine and
``_write_branch_rollouts`` against a capturing writer, and the main process
writes episodes and rollout bundles through the real DatasetWriter in episode
order. The run directory (episodes/, rollouts/, preview/, index.jsonl,
manifest.json, config.json, README.txt) is the one ``_cmd_generate`` writes,
plus verification.json.

Verification (``--verify-episodes K``):
  (a) for the first K episodes, a fresh plain generation of the same index
      reproduces the written episode bit for bit (all arrays; the part up to
      the branch time is reported separately), and the rollout engine's
      re-logged past up to the fork equals the written past;
  (b) every rollout bundle: dataGen's validate_rollouts contract, shapes
      [N,K,H,2] / [N,K,H], origin_step = round(B / LOG_DT), agent slots
      0..N-1, the branch seeds dataGen derives, valid == finite(pos), and
      (first K episodes) branch step 0 on the logged continuation at
      B + LOG_DT; dataGen's storage.manifest.verify_run must report nothing.
``--reference RUN`` instead regenerates episode ``--reference-index`` of an
existing main.py-generated rollout run in memory and compares episode,
rollout bundle and index row bit for bit (end-to-end proof of the path).

No path is hard-coded: roots default to <PRISM root>/Logs/Dataset and
<PRISM root>/dataGen, with PRISM root = the parent of modelVal.

  python -m experiments.gen_rollout_heldout --preset test_ood --start 1200
         --episodes 100 --workers 20 [--dataset-root ...] [--out-root ...]
  python -m experiments.gen_rollout_heldout --check-only
  python -m experiments.gen_rollout_heldout --reference <run> --reference-index 0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELVAL_ROOT = os.path.dirname(_HERE)
#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(MODELVAL_ROOT)).replace("\\", "/")
if MODELVAL_ROOT not in sys.path:
    sys.path.insert(0, MODELVAL_ROOT)
from experiments.reaction_labels import setup_datagen  # noqa: E402

DEFAULT_DATASET_ROOT = os.path.join(PRISM_ROOT, "Logs", "Dataset").replace("\\", "/")
PRESETS = ("test_id", "test_ood")
DEFAULT_START, DEFAULT_EPISODES = 1200, 100
DEFAULT_ROLLOUTS, DEFAULT_BRANCH_T = 60, 30.0
DEFAULT_TAG = "rollout_heldout"
#: time-indexed episode arrays compared against the rollout engine's re-log
RELOG_KEYS = ("t", "own_pos", "own_vel", "own_psi", "gt_valid", "gt_pos",
              "gt_vel", "gt_psi", "gt_mode", "gt_intent", "obs_detected",
              "obs_pos", "obs_R", "obs_t_valid", "obs_track_id",
              "obs_track_age")
ROLLOUT_KEYS = {"origin_step", "agent_slot", "branch_seed", "pos", "valid"}

_DGM = None


def datagen_main(root: str):
    """dataGen/main.py, loaded read-only under a private module name so it
    can never be confused with modelVal/main.py or modelTraining/main.py."""
    global _DGM
    if _DGM is None:
        spec = importlib.util.spec_from_file_location(
            "prism_datagen_main", os.path.join(root, "main.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DGM = mod
    return _DGM


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


# ============================================================ seed check

def seed_check(preset: str, start: int, n: int, roots, ignore=()) -> dict:
    """Seeds of episodes start..start+n-1: inside the preset's block and in
    no run under ``roots`` (every index.jsonl read)."""
    from scenario.presets import episode_seed, seed_block, split_of_seed
    lo, hi = seed_block(preset)
    seeds = [episode_seed(preset, i) for i in range(start, start + n)]
    inside = all(lo <= s < hi and split_of_seed(s) == preset for s in seeds)
    target = set(seeds)
    scanned, overlaps, no_seed = {}, {}, []
    block_max = None
    for root in dict.fromkeys(os.path.abspath(r) for r in roots):
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            ip = os.path.join(root, d, "index.jsonl")
            if d in ignore or not os.path.isfile(ip):
                continue
            ss = set()
            with open(ip, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line).get("seed")
                    if s is None:
                        no_seed.append(d)
                        break
                    ss.add(int(s))
            scanned[f"{root}/{d}".replace("\\", "/")] = len(ss)
            inter = sorted(target & ss)
            if inter:
                overlaps[d] = {"n": len(inter), "first": inter[0], "last": inter[-1]}
            in_block = [s for s in ss if lo <= s < hi]
            if in_block:
                block_max = (max(in_block) if block_max is None
                             else max(block_max, max(in_block)))
    return {"preset": preset, "seed_block": [int(lo), int(hi)],
            "episode_range": [int(start), int(start + n - 1)],
            "seed_range": [int(min(seeds)), int(max(seeds))],
            "inside_block": bool(inside), "n_runs_scanned": len(scanned),
            "n_seeds_scanned": int(sum(scanned.values())),
            "max_existing_seed_in_block": block_max,
            "runs_without_seed_key": sorted(set(no_seed)),
            "ignored_runs": sorted(ignore),
            "overlapping_runs": overlaps,
            "ok": bool(inside) and not overlaps and not no_seed}


# ============================================================ generation

class _CaptureWriter:
    """Stands in for DatasetWriter inside dataGen's _write_branch_rollouts:
    records the bundle instead of writing it (the main process writes it)."""

    def __init__(self):
        self.bundles = []

    def write_rollouts(self, episode_index, arrays, **_kw):
        self.bundles.append((int(episode_index),
                             {k: np.array(v, copy=True) for k, v in arrays.items()}))
        return ""


def generate_episode(task: dict) -> dict:
    """One episode exactly as _cmd_generate makes it (worker side)."""
    root = setup_datagen(task["datagen_root"])
    dgm = datagen_main(root)
    from scenario.presets import config_for_episode
    from simulation.engine import SimulationEngine

    i = int(task["episode_index"])
    t0 = time.perf_counter()
    cfg = config_for_episode(task["preset"], i)
    engine = SimulationEngine(cfg, episode_index=i)
    ep, meta = engine.run_episode()
    t_ep = time.perf_counter() - t0
    written = copy.deepcopy(ep)      # what _cmd_generate hands to write_episode
    out = {"episode_index": i, "seed": int(cfg.seed), "ep": written,
           "meta": meta, "t_episode_s": round(t_ep, 3), "rollouts": None,
           "n_slots": 0, "t_rollouts_s": 0.0}
    k = int(task["rollouts"])
    if k:
        cap = _CaptureWriter()
        t1 = time.perf_counter()
        n_slots = dgm._write_branch_rollouts(cap, engine, i, k,
                                             float(task["branch_t"]))
        out["t_rollouts_s"] = round(time.perf_counter() - t1, 3)
        if len(cap.bundles) != 1 or cap.bundles[0][0] != i:
            raise RuntimeError(f"episode {i}: _write_branch_rollouts wrote "
                               f"{[b[0] for b in cap.bundles]}")
        out["rollouts"] = cap.bundles[0][1]
        out["n_slots"] = int(n_slots)
        a, b = ep.to_npz_dict(), written.to_npz_dict()
        out["episode_untouched_by_rollouts"] = all(_same(a[x], b[x]) for x in a)
        if task.get("return_relogged"):
            n = int(engine.n_logged)
            e2 = engine.ep.to_npz_dict()
            out["relogged"] = {"n_logged": n,
                               **{x: np.array(e2[x][:n]) for x in RELOG_KEYS}}
    out["wall_s"] = round(time.perf_counter() - t0, 3)
    return out


# ============================================================ verification

def check_bundle(r: dict, n_agents: int, seed: int, k: int, branch_t: float) -> list[str]:
    """(b) contract of one written rollout bundle."""
    from core.constants import LOG_DT, PRED_STEPS
    from simulation.rollout import branch_seed_block
    from storage.writer import validate_rollouts
    p = [f"validate_rollouts: {x}" for x in validate_rollouts(r)]
    if set(r) != ROLLOUT_KEYS:
        p.append(f"keys {sorted(r)} != {sorted(ROLLOUT_KEYS)}")
        return p
    if r["pos"].shape != (n_agents, k, PRED_STEPS, 2) or r["pos"].dtype != np.float32:
        p.append(f"pos {r['pos'].shape} {r['pos'].dtype}")
    if r["valid"].shape != (n_agents, k, PRED_STEPS) or r["valid"].dtype != np.bool_:
        p.append(f"valid {r['valid'].shape} {r['valid'].dtype}")
    else:
        if not np.array_equal(r["valid"], np.isfinite(r["pos"]).all(axis=-1)):
            p.append("valid != isfinite(pos)")
    if not np.all(r["origin_step"] == int(round(branch_t / LOG_DT))):
        p.append(f"origin_step {np.unique(r['origin_step'])}")
    if not np.array_equal(r["agent_slot"], np.arange(n_agents, dtype=np.int32)):
        p.append("agent_slot != 0..N-1")
    want = np.tile(np.asarray(branch_seed_block(int(seed) + int(round(branch_t)), k),
                              dtype=np.int64), (n_agents, 1))
    if not np.array_equal(r["branch_seed"], want):
        p.append("branch_seed != dataGen branch_seed_block(seed + branch_t, K)")
    return p


def verify_episode(run_dir: str, preset: str, idx: int, seed: int, k: int,
                   branch_t: float, relogged: dict | None) -> dict:
    """(a) regeneration checks + (b) timing / divergence of one episode."""
    from core.constants import LOG_DT
    from scenario.presets import config_for_episode
    from simulation.engine import SimulationEngine
    from simulation.rollout import frozen_agents

    with np.load(os.path.join(run_dir, "episodes", f"ep_{idx:06d}.npz")) as z:
        w = {x: z[x] for x in z.files}
    T = w["t"].shape[0]
    tt = w["t"].astype(np.float64)
    fork = np.flatnonzero(tt == float(branch_t))
    past = tt <= float(branch_t)
    eng = SimulationEngine(config_for_episode(preset, idx), episode_index=idx)
    fresh = eng.run_episode()[0].to_npz_dict()
    timed = [x for x in fresh if x in RELOG_KEYS]
    res = {"episode_index": idx, "seed": seed,
           "a_fresh_full_identical_keys": sorted(x for x in fresh if _same(fresh[x], w.get(x))),
           "a_fresh_full_differing_keys": sorted(x for x in fresh if not _same(fresh[x], w.get(x))),
           "a_fresh_past_differing_keys": sorted(x for x in timed if not _same(fresh[x][past], w[x][past])),
           "a_past_steps": int(past.sum()), "a_fork_index": int(fork[0]) if fork.size == 1 else None,
           "n_keys": len(fresh)}
    if relogged is not None:
        n = int(relogged["n_logged"])
        res["a_relogged_n_steps"] = n
        res["a_relogged_past_differing_keys"] = sorted(
            x for x in RELOG_KEYS if not _same(relogged[x], w[x][:n]))
        res["a_relogged_ends_at_branch_time"] = bool(n >= 1 and float(relogged["t"][-1]) == float(branch_t))
    with np.load(os.path.join(run_dir, "rollouts", f"ep_{idx:06d}_rollouts.npz")) as z:
        r = {x: z[x] for x in z.files}
    N = w["gt_pos"].shape[1]
    res["b_bundle_problems"] = check_bundle(r, N, seed, k, branch_t)
    if fork.size == 1 and fork[0] + 2 < T:
        f0 = int(fork[0])
        gt = w["gt_pos"].astype(np.float64)
        d1, d2 = [], []
        for a in range(N):
            ok = r["valid"][a, :, 0]
            if not ok.any() or not (w["gt_valid"][f0 + 1, a] and w["gt_valid"][f0 + 2, a]):
                continue
            pa = r["pos"][a, ok, 0].astype(np.float64)
            d1.append(float(np.min(np.linalg.norm(pa - gt[f0 + 1, a], axis=1))))
            d2.append(float(np.min(np.linalg.norm(pa - gt[f0 + 2, a], axis=1))))
        res["b_step0_vs_log_t_plus_0.5s_median_m"] = float(np.median(d1)) if d1 else None
        res["b_step0_vs_log_t_plus_1.0s_median_m"] = float(np.median(d2)) if d2 else None
        res["b_alive_at_fork"] = int(w["gt_valid"][f0].sum())
        res["b_alive_at_fork_all_branches_valid_k0"] = int(
            sum(bool(w["gt_valid"][f0, a] and r["valid"][a, :, 0].all()) for a in range(N)))
    fz = frozen_agents(np.transpose(r["pos"], (1, 0, 2, 3)).astype(np.float64))
    res["b_live_agents_at_horizon"] = int(fz["n_live"])
    res["b_frozen_fraction_at_horizon"] = round(float(fz["fraction"]), 4)
    res["b_valid_fraction_by_horizon"] = {
        f"{int(round((h + 1) * LOG_DT))}s": round(float(r["valid"][:, :, h].mean()), 4)
        for h in (9, 19, 29)}
    res["ok"] = (not res["a_fresh_full_differing_keys"]
                 and not res["a_fresh_past_differing_keys"]
                 and not res.get("a_relogged_past_differing_keys")
                 and res.get("a_relogged_ends_at_branch_time", True)
                 and not res["b_bundle_problems"]
                 and (res.get("b_step0_vs_log_t_plus_0.5s_median_m") or 0.0) < 1e-3)
    return res


def reproduce_reference(ref_run: str, idx: int, datagen_root: str) -> dict:
    """Regenerate episode ``idx`` of an existing main.py rollout run in memory
    and compare episode arrays, rollout bundle and index row bit for bit."""
    from core.constants import LOG_DT
    with open(os.path.join(ref_run, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    with np.load(os.path.join(ref_run, "rollouts", f"ep_{idx:06d}_rollouts.npz")) as z:
        ref_ro = {x: z[x] for x in z.files}
    with np.load(os.path.join(ref_run, "episodes", f"ep_{idx:06d}.npz")) as z:
        ref_ep = {x: z[x] for x in z.files}
    row = None
    with open(os.path.join(ref_run, "index.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and int(json.loads(line)["episode_index"]) == idx:
                row = json.loads(line)
    k = int(ref_ro["branch_seed"].shape[1])
    branch_t = float(ref_ro["origin_step"][0]) * LOG_DT
    t0 = time.perf_counter()
    g = generate_episode({"datagen_root": datagen_root, "preset": man["preset"],
                          "episode_index": idx, "rollouts": k, "branch_t": branch_t})
    ep = g["ep"].to_npz_dict()
    meta_row = json.loads(json.dumps(g["meta"].to_dict()))
    return {"reference_run": ref_run.replace("\\", "/"), "episode_index": idx,
            "preset": man["preset"], "rollouts": k, "branch_t": branch_t,
            "episode_keys_differing": sorted(x for x in set(ep) | set(ref_ep)
                                             if x not in ep or x not in ref_ep
                                             or not _same(ep[x], ref_ep[x])),
            "rollout_keys_differing": sorted(x for x in set(g["rollouts"]) | set(ref_ro)
                                             if x not in g["rollouts"] or x not in ref_ro
                                             or not _same(g["rollouts"][x], ref_ro[x])),
            "index_row_identical": meta_row == row,
            "identical": None, "wall_s": round(time.perf_counter() - t0, 1)}


# ==================================================================== main

def _timing(results: list[dict]) -> dict:
    if not results:
        return {}
    w = np.array([r["wall_s"] for r in results], dtype=float)
    e = np.array([r["t_episode_s"] for r in results], dtype=float)
    ro = np.array([r["t_rollouts_s"] for r in results], dtype=float)
    return {"n": int(w.size), "per_episode_wall_s_mean": round(float(w.mean()), 2),
            "per_episode_wall_s_median": round(float(np.median(w)), 2),
            "per_episode_wall_s_max": round(float(w.max()), 2),
            "episode_part_s_mean": round(float(e.mean()), 2),
            "rollout_part_s_mean": round(float(ro.mean()), 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=PRESETS, default="test_ood")
    ap.add_argument("--start", type=int, default=DEFAULT_START)
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    ap.add_argument("--branch-t", dest="branch_t", type=float, default=DEFAULT_BRANCH_T)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                    help="scanned for existing seeds (every <run>/index.jsonl)")
    ap.add_argument("--out-root", default=None,
                    help="where the new run directory is created "
                         "(default: --dataset-root); also scanned")
    ap.add_argument("--datagen-root", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify-episodes", type=int, default=2,
                    help="regeneration checks (a) on the first K episodes")
    ap.add_argument("--ignore-run", action="append", default=[],
                    help="run name excluded from the seed scan (only for an "
                         "aborted earlier attempt of this same generation)")
    ap.add_argument("--check-only", action="store_true",
                    help="run the seed checks and exit")
    ap.add_argument("--reference", default=None,
                    help="existing main.py rollout run to reproduce in memory")
    ap.add_argument("--reference-index", type=int, default=0)
    args = ap.parse_args(argv)
    root = setup_datagen(args.datagen_root)
    os.environ["PRISM_DATAGEN_ROOT"] = root
    args.out_root = os.path.abspath(args.out_root or args.dataset_root).replace("\\", "/")

    if args.reference:
        ref = args.reference if os.path.isdir(args.reference) else \
            os.path.join(args.dataset_root, args.reference)
        rep = reproduce_reference(ref, args.reference_index, root)
        rep["identical"] = (not rep["episode_keys_differing"]
                            and not rep["rollout_keys_differing"]
                            and rep["index_row_identical"])
        print(json.dumps(rep, indent=1))
        return 0 if rep["identical"] else 1

    check = seed_check(args.preset, args.start, args.episodes,
                       [args.dataset_root, args.out_root], tuple(args.ignore_run))
    print(json.dumps({k: v for k, v in check.items()}, indent=1))
    if not check["ok"]:
        print("SEED CHECK FAILED -- nothing generated")
        return 1
    if args.check_only:
        return 0

    from scenario.presets import make_preset, seed_block
    from storage.manifest import verify_run
    from storage.writer import DatasetWriter
    os.makedirs(args.out_root, exist_ok=True)
    writer = DatasetWriter(logs_root=args.out_root)
    run_dir = writer.open_run(make_preset(args.preset), preset=args.preset,
                              tag=args.tag)
    print(f"run: {run_dir}", flush=True)
    idxs = list(range(args.start, args.start + args.episodes))
    tasks = [{"datagen_root": root, "preset": args.preset, "episode_index": i,
              "rollouts": args.rollouts, "branch_t": args.branch_t,
              "return_relogged": j < args.verify_episodes}
             for j, i in enumerate(idxs)]
    results, relogged = [], {}
    t_wall = time.time()
    pool = (mp.get_context("spawn").Pool(args.workers)
            if args.workers > 1 and len(tasks) > 1 else None)
    try:
        it = (pool.imap(generate_episode, tasks, chunksize=1) if pool is not None
              else map(generate_episode, tasks))
        for j, res in enumerate(it):
            writer.write_episode(res["ep"], res["meta"])
            if res["rollouts"] is not None:
                writer.write_rollouts(res["episode_index"], res["rollouts"])
            if "relogged" in res:
                relogged[res["episode_index"]] = res.pop("relogged")
            results.append({k: v for k, v in res.items()
                            if k not in ("ep", "meta", "rollouts")})
            print(f"  {j + 1}/{len(tasks)} ep {res['episode_index']} "
                  f"({res['wall_s']:.1f}s: episode {res['t_episode_s']:.1f}s + "
                  f"rollouts {res['t_rollouts_s']:.1f}s) "
                  f"{time.time() - t_wall:.0f}s elapsed", flush=True)
        block = seed_block(args.preset)
        writer.add_split(
            args.preset, n_episodes=len(idxs), seed_base=int(block[0]),
            seed_block=[int(block[0]), int(block[1])],
            episode_range=[idxs[0], idxs[-1]],
            notes=(f"generated by modelVal/experiments/gen_rollout_heldout.py "
                   f"= main.py generate --start-index {args.start} --rollouts "
                   f"{args.rollouts} --branch-t {args.branch_t:g} path, "
                   f"parallel over episodes; held-out rollout set for review "
                   f"R3 majors 1/2 (seeds disjoint from every run under the "
                   f"dataset root)"))
    except BaseException:
        writer.close(status="aborted")
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    wall = time.time() - t_wall
    timing = {**_timing(results), "workers": int(args.workers),
              "run_wall_s": round(wall, 1)}
    writer.close(extra={
        "generator": {"script": os.path.abspath(__file__).replace("\\", "/"),
                      "script_sha256": _sha256(os.path.abspath(__file__)),
                      "datagen_main_sha256": _sha256(os.path.join(root, "main.py")),
                      "equivalent_command": (
                          f"python main.py generate --preset {args.preset} "
                          f"--episodes {args.episodes} --start-index {args.start} "
                          f"--tag {args.tag} --rollouts {args.rollouts} "
                          f"--branch-t {args.branch_t:g}")},
        "seed_check": check, "timing": timing})

    # ------------------------------------------------------ verification
    t_v = time.time()
    bundle_problems = {}
    for r in results:
        i = r["episode_index"]
        with np.load(os.path.join(run_dir, "rollouts", f"ep_{i:06d}_rollouts.npz")) as z:
            b = {x: z[x] for x in z.files}
        with np.load(os.path.join(run_dir, "episodes", f"ep_{i:06d}.npz")) as z:
            n_agents = int(z["gt_pos"].shape[1])
        pr = check_bundle(b, n_agents, r["seed"], args.rollouts, args.branch_t)
        if pr or not r.get("episode_untouched_by_rollouts", True):
            bundle_problems[i] = pr + ([] if r.get("episode_untouched_by_rollouts", True)
                                       else ["episode changed by the rollout pass"])
    per_ep = [verify_episode(run_dir, args.preset, i, results[j]["seed"],
                             args.rollouts, args.branch_t, relogged.get(i))
              for j, i in enumerate(idxs[:args.verify_episodes])]
    vr = verify_run(run_dir, deep=True)
    ver = {"run_dir": run_dir, "seed_check": check, "timing": timing,
           "bundle_checks_all_episodes": {"n": len(results),
                                          "problems": bundle_problems},
           "regeneration_checks": per_ep,
           "datagen_verify_run_problems": vr,
           "verify_wall_s": round(time.time() - t_v, 1)}
    ver["ok"] = (not bundle_problems and all(p["ok"] for p in per_ep) and not vr)
    with open(os.path.join(run_dir, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump(ver, fh, indent=1)
    print(json.dumps({"ok": ver["ok"], "timing": timing,
                      "bundle_problems": bundle_problems,
                      "verify_run_problems": vr[:10],
                      "regeneration": [{k: p[k] for k in (
                          "episode_index", "ok", "a_fresh_full_differing_keys",
                          "a_fresh_past_differing_keys",
                          "a_relogged_past_differing_keys", "b_bundle_problems",
                          "b_step0_vs_log_t_plus_0.5s_median_m",
                          "b_step0_vs_log_t_plus_1.0s_median_m",
                          "b_frozen_fraction_at_horizon") if k in p}
                          for p in per_ep]}, indent=1))
    print(f"-> {run_dir}")
    return 0 if ver["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
