"""Tests for storage.writer / storage.manifest.

Covers the two properties the dataset depends on:
    * an episode survives the NPZ round trip with dtypes and the NaN pattern
      intact (doc 3.5: a gap must stay a gap),
    * a finished run passes storage.manifest.verify_run.

Run directly::

    python $PRISM_ROOT/dataGen/tests/test_storage.py

All temporary data is written into a throwaway directory under the system
temp dir, never into $PRISM_ROOT/Logs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.constants import LOG_DT                                    # noqa: E402
from core.types import IntentType, ManeuverMode, PlatformType, RiskLevel  # noqa: E402
from scenario.config import ScenarioConfig                            # noqa: E402
from storage import manifest as mf                                    # noqa: E402
from storage import schema                                            # noqa: E402
from storage.schema import EpisodeArrays, EpisodeMeta                 # noqa: E402
from storage.writer import (                                          # noqa: E402
    DatasetWriter,
    build_preview,
    iter_run,
    list_runs,
    read_episode,
    read_index,
    read_preview,
    read_rollouts,
    validate_rollouts,
)

_FAILURES: list[str] = []
_CHECKS = 0


def check(condition: bool, message: str) -> None:
    """Record one assertion result."""
    global _CHECKS
    _CHECKS += 1
    if not condition:
        _FAILURES.append(message)


def check_close(a: float, b: float, tol: float, message: str) -> None:
    """Record one approximate-equality assertion."""
    check(abs(float(a) - float(b)) <= tol, f"{message} ({a!r} vs {b!r})")


# --------------------------------------------------------------------------
# Synthetic episode
# --------------------------------------------------------------------------


def make_episode(episode_index: int = 0, T: int = 120, N: int = 3,
                 seed: int = 7) -> tuple[EpisodeArrays, EpisodeMeta]:
    """Build a small, schema-valid episode with holes in every layer.

    Slot 0 is a full-length agent, slot 1 appears late (so the early steps are
    padding), slot 2 is never used at all. Detections are dropped on a fixed
    pattern so the NaN policy of the observation layer is exercised.

    Args:
        episode_index: index recorded in the metadata.
        T: logged steps.
        N: agent slot capacity.
        seed: RNG seed for the synthetic tracks.

    Returns:
        ``(EpisodeArrays, EpisodeMeta)``.
    """
    rng = np.random.default_rng(seed)
    ep = EpisodeArrays.allocate(T, N)
    ep.t[:] = (np.arange(T) * LOG_DT).astype(np.float32)

    # ownship: straight, due East at 55 m/s
    ep.own_pos[:, 0] = (np.arange(T) * 55.0 * LOG_DT).astype(np.float32)
    ep.own_pos[:, 1] = 0.0
    ep.own_vel[:, 0] = np.float32(55.0)
    ep.own_vel[:, 1] = np.float32(0.0)
    ep.own_psi[:] = np.float32(0.0)

    start = {0: 0, 1: 30}
    types = {0: PlatformType.MULTIROTOR, 1: PlatformType.FIXED_WING_UAS}
    for slot, t0 in start.items():
        steps = np.arange(t0, T)
        # gentle constant turn so the track is not exactly CV
        psi = 0.5 * np.pi - 0.02 * (steps - t0)
        v = 25.0
        x = 900.0 + np.cumsum(np.cos(psi) * v * LOG_DT) - np.cos(psi[0]) * v * LOG_DT
        y = -400.0 + np.cumsum(np.sin(psi) * v * LOG_DT) - np.sin(psi[0]) * v * LOG_DT
        ep.gt_valid[t0:, slot] = True
        ep.gt_pos[t0:, slot, 0] = x.astype(np.float32)
        ep.gt_pos[t0:, slot, 1] = y.astype(np.float32)
        ep.gt_vel[t0:, slot, 0] = (np.cos(psi) * v).astype(np.float32)
        ep.gt_vel[t0:, slot, 1] = (np.sin(psi) * v).astype(np.float32)
        ep.gt_psi[t0:, slot] = np.arctan2(np.sin(psi), np.cos(psi)).astype(np.float32)
        ep.gt_mode[t0:, slot] = np.int8(ManeuverMode.CT_RIGHT)
        ep.gt_intent[t0:, slot] = np.int8(
            IntentType.DELIVERY_WP if slot == 0 else IntentType.CORRIDOR_TRANSIT
        )
        ep.gt_type[slot] = np.int8(types[slot])
        ep.agent_ids[slot] = np.int32(100 + slot)

        # observations: a live track with periodic misses
        detected = np.ones(steps.size, dtype=bool)
        detected[::4] = False                     # every 4th update is missed
        age = 0
        for k, step in enumerate(steps):
            age += 1
            ep.obs_track_id[step, slot] = np.int32(10 * (slot + 1))
            ep.obs_track_age[step, slot] = np.int32(age)
            if detected[k]:
                ep.obs_detected[step, slot] = True
                noise = rng.normal(0.0, 4.0, size=2)
                ep.obs_pos[step, slot, 0] = np.float32(x[k] + noise[0])
                ep.obs_pos[step, slot, 1] = np.float32(y[k] + noise[1])
                ep.obs_R[step, slot] = np.array([[16.0, 1.5], [1.5, 25.0]],
                                                dtype=np.float32)
                # The measurement was true 0.2 s BEFORE the frame it lands in
                # (sensor latency); see storage.schema obs_t_valid.
                ep.obs_t_valid[step, slot] = np.float32(
                    float(ep.t[step]) - 0.2)

    meta = EpisodeMeta(
        episode_index=int(episode_index),
        seed=1234 + int(episode_index),
        density=40.0,
        realised_density_per_km2=0.28,
        wind_speed=6.0,
        wind_dir=0.3,
        n_agents=2,
        geometry_type=[1],
        cpa_dist=[420.0],
        tcpa=[6.0],
        min_separation_m=420.0,
        risk_level=int(RiskLevel.MEDIUM),
        platform_counts={int(PlatformType.MULTIROTOR): 1,
                         int(PlatformType.FIXED_WING_UAS): 1},
        intent_counts={int(IntentType.DELIVERY_WP): 1,
                       int(IntentType.CORRIDOR_TRANSIT): 1},
        nmac_occurred=False,
        dwc_violated=True,
        dwc_alerted=True,
        duration_s=float(T * LOG_DT),
    )
    return ep, meta


def make_config(T: int = 120) -> ScenarioConfig:
    """A ScenarioConfig whose duration matches the synthetic episode length."""
    return ScenarioConfig(name="test_storage", seed=1234,
                          duration_s=float(T * LOG_DT))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_synthetic_episode_is_valid() -> None:
    """The fixture itself must satisfy the storage contract."""
    ep, meta = make_episode()
    check(schema.validate(ep) == [], f"fixture episode invalid: {schema.validate(ep)}")
    check(meta.validate() == [], f"fixture meta invalid: {meta.validate()}")


def test_npz_roundtrip(tmp: str) -> None:
    """Dtypes, values and the NaN pattern must survive a write/read cycle."""
    ep, meta = make_episode()
    cfg = make_config()
    w = DatasetWriter(logs_root=tmp)
    run_dir = w.open_run(cfg, preset="unit", tag="roundtrip",
                         include_fingerprint=False)
    path = w.write_episode(ep, meta)
    w.close()

    back = read_episode(path, validate=True)
    for name, (dtype, _shape) in schema.EPISODE_ARRAY_SPEC.items():
        a = getattr(ep, name)
        b = getattr(back, name)
        check(b.dtype == np.dtype(dtype),
              f"{name}: dtype changed on round trip ({b.dtype})")
        check(a.shape == b.shape, f"{name}: shape changed on round trip")
        if a.dtype.kind == "f":
            check(np.array_equal(np.isnan(a), np.isnan(b)),
                  f"{name}: NaN pattern changed on round trip")
            check(np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]),
                  f"{name}: finite values changed on round trip")
        else:
            check(np.array_equal(a, b), f"{name}: values changed on round trip")

    # the NaN policy must really be non-trivial in the fixture
    check(np.any(np.isnan(ep.obs_pos)), "fixture has no observation gaps to test")
    check(np.any(np.isnan(ep.gt_pos)), "fixture has no GT padding to test")

    # GT and OBS live in separate, explicitly prefixed key groups on disk
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
    check({k for k in keys if k.startswith("gt_")} ==
          {k for k in schema.EPISODE_ARRAY_SPEC if k.startswith("gt_")},
          "gt_* key group on disk does not match the schema")
    check({k for k in keys if k.startswith("obs_")} ==
          {k for k in schema.EPISODE_ARRAY_SPEC if k.startswith("obs_")},
          "obs_* key group on disk does not match the schema")
    shutil.rmtree(run_dir, ignore_errors=True)


def test_three_episode_run(tmp: str) -> None:
    """A complete 3-episode run must verify, list and iterate correctly."""
    cfg = make_config()
    w = DatasetWriter(logs_root=tmp)
    run_dir = w.open_run(cfg, preset="unit", tag="threeep",
                         include_fingerprint=False)
    w.add_split("train", n_episodes=3, seed_base=1234, episode_range=(0, 3),
                notes="unit test split")
    for i in range(3):
        ep, meta = make_episode(episode_index=i, seed=11 + i)
        w.write_episode(ep, meta)
    # branch rollouts for one episode
    B, K, H = 2, 5, 6
    rollouts = {
        "origin_step": np.array([4, 9], dtype=np.int32),
        "agent_slot": np.array([0, 0], dtype=np.int32),
        "branch_seed": np.arange(B * K, dtype=np.int64).reshape(B, K),
        "pos": np.zeros((B, K, H, 2), dtype=np.float32),
        "valid": np.ones((B, K, H), dtype=np.bool_),
    }
    roll_path = w.write_rollouts(1, rollouts)
    w.close()

    problems = mf.verify_run(run_dir)
    check(problems == [], f"verify_run reported problems: {problems}")

    metas = read_index(run_dir)
    check(len(metas) == 3, f"index.jsonl should hold 3 lines, got {len(metas)}")
    check([m.episode_index for m in metas] == [0, 1, 2],
          "index.jsonl episode order is wrong")

    seen = [(m.episode_index if m else -1, e.n_steps)
            for m, e in iter_run(run_dir, validate=True)]
    check(len(seen) == 3, "iter_run did not yield 3 episodes")
    check(all(idx >= 0 for idx, _ in seen), "iter_run lost the metadata pairing")

    man = mf.read_manifest(run_dir)
    check(man["status"] == "complete", "manifest status should be 'complete'")
    check(man["counts"]["n_episodes"] == 3, "manifest episode count is wrong")
    check(man["counts"]["n_index_lines"] == 3, "manifest index count is wrong")
    check(man["counts"]["n_rollout_files"] == 1, "manifest rollout count is wrong")
    check("train" in man["splits"], "split provenance was not recorded")
    check(mf.validate_manifest(man) == [],
          f"validate_manifest: {mf.validate_manifest(man)}")

    back = read_rollouts(roll_path)
    check(set(back) == set(rollouts), "rollout keys changed on round trip")
    check(back["branch_seed"].dtype == np.int64, "rollout dtype changed")
    check(validate_rollouts(back) == [], "round-tripped rollouts fail validation")

    runs = list_runs(tmp)
    check(any(r["run_dir"] == run_dir for r in runs), "list_runs missed the run")
    check(all(r["status"] in {"complete", "open", "aborted", "unreadable"}
              for r in runs), "list_runs returned a bad status")

    summary = mf.summarize_run(run_dir)
    check(summary["n_episodes"] == 3, "summarize_run episode count is wrong")
    check(summary["dwc_rate"] == 1.0, "summarize_run dwc_rate is wrong")
    shutil.rmtree(run_dir, ignore_errors=True)


def test_verify_run_detects_damage(tmp: str) -> None:
    """A deleted episode file and a stray index line must both be caught."""
    cfg = make_config()
    w = DatasetWriter(logs_root=tmp)
    run_dir = w.open_run(cfg, preset="unit", tag="damage",
                         include_fingerprint=False)
    for i in range(3):
        ep, meta = make_episode(episode_index=i, seed=21 + i)
        w.write_episode(ep, meta)
    w.close()
    check(mf.verify_run(run_dir) == [], "the undamaged run should verify")

    os.remove(mf.episode_path(run_dir, 1))
    problems = mf.verify_run(run_dir)
    check(any("no episode file" in p for p in problems),
          f"a missing episode file was not detected: {problems}")
    check(any("index.jsonl has 3 line" in p for p in problems),
          f"the index/file count mismatch was not detected: {problems}")
    shutil.rmtree(run_dir, ignore_errors=True)


def test_partial_run_is_readable(tmp: str) -> None:
    """A run killed mid-flight (never closed) must still load."""
    cfg = make_config()
    w = DatasetWriter(logs_root=tmp)
    run_dir = w.open_run(cfg, preset="unit", tag="partial",
                         include_fingerprint=False)
    for i in range(2):
        ep, meta = make_episode(episode_index=i, seed=31 + i)
        w.write_episode(ep, meta)
    # deliberately NOT closing: simulate a killed process
    w._index_fh.flush()                                   # noqa: SLF001

    metas = read_index(run_dir)
    check(len(metas) == 2, f"partial run lost index lines ({len(metas)})")
    eps = list(iter_run(run_dir, validate=True))
    check(len(eps) == 2, "partial run lost episode files")
    man = mf.read_manifest(run_dir)
    check(man["status"] == "open", "an unclosed run should still say 'open'")
    w.close("aborted")
    shutil.rmtree(run_dir, ignore_errors=True)


def test_preview_json_is_strict(tmp: str) -> None:
    """The GUI preview must be strict JSON: NaN encoded as null, not NaN."""
    ep, meta = make_episode()
    preview = build_preview(ep, meta, stride=2)
    text = json.dumps(preview, allow_nan=False)          # raises if NaN leaked
    reparsed = json.loads(text)
    check(reparsed["n_steps"] == 60, "preview decimation is wrong")
    check(len(reparsed["agents"]) == 2, "preview dropped an agent")
    slot0 = reparsed["agents"][0]
    check(any(v is None for v in slot0["obs_x"]),
          "preview should encode an observation gap as null")
    check("meta" in reparsed and reparsed["meta"]["episode_index"] == 0,
          "preview lost the episode metadata")

    cfg = make_config()
    w = DatasetWriter(logs_root=tmp)
    run_dir = w.open_run(cfg, preset="unit", tag="preview",
                         include_fingerprint=False)
    w.write_episode(ep, meta)
    w.close()
    loaded = read_preview(mf.preview_path(run_dir, 0))
    check(loaded["episode_index"] == 0, "preview file did not round-trip")
    readme = os.path.join(run_dir, mf.README_NAME)
    check(os.path.isfile(readme), "README.txt was not written")
    with open(readme, "r", encoding="utf-8") as fh:
        text = fh.read()
    check("NEVER interpolated" in text or "NEVER" in text,
          "README does not document the NaN policy")
    check("obs_track_age" in text, "README does not document every array")
    shutil.rmtree(run_dir, ignore_errors=True)


def test_writer_rejects_bad_input(tmp: str) -> None:
    """Validation must refuse a corrupted episode and a duplicate index."""
    cfg = make_config()
    w = DatasetWriter(logs_root=tmp)
    run_dir = w.open_run(cfg, preset="unit", tag="reject",
                         include_fingerprint=False)

    ep, meta = make_episode()
    # NOTE: EpisodeArrays.from_npz_dict uses np.asarray, which does NOT copy
    # when the dtype already matches -- deep-copy before mutating.
    bad = EpisodeArrays.from_npz_dict(
        {k: v.copy() for k, v in ep.to_npz_dict().items()}
    )
    bad.obs_pos[0, 2, :] = 0.0            # a detection-free slot must stay NaN
    raised = False
    try:
        w.write_episode(bad, meta)
    except ValueError:
        raised = True
    check(raised, "writer accepted an episode that breaks the NaN policy")

    w.write_episode(ep, meta)
    raised = False
    try:
        w.write_episode(ep, meta)
    except ValueError:
        raised = True
    check(raised, "writer accepted a duplicate episode index")

    raised = False
    try:
        w.write_rollouts(0, {"pos": np.zeros((2, 3, 4, 2), dtype=np.float64)})
    except ValueError:
        raised = True
    check(raised, "writer accepted rollouts with the wrong dtype")
    w.close()
    shutil.rmtree(run_dir, ignore_errors=True)


def main() -> int:
    """Run every test and report. Returns a process exit code."""
    tmp = tempfile.mkdtemp(prefix="prism_storage_test_")
    try:
        test_synthetic_episode_is_valid()
        test_npz_roundtrip(tmp)
        test_three_episode_run(tmp)
        test_verify_run_detects_damage(tmp)
        test_partial_run_is_readable(tmp)
        test_preview_json_is_strict(tmp)
        test_writer_rejects_bad_input(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} of {_CHECKS} checks")
        for f in _FAILURES:
            print("  -", f)
        return 1
    print(f"ok  ({_CHECKS} checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
