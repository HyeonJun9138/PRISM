"""Tests for the simulation engine, the scenario builder and the split presets.

The four properties that everything downstream rests on:

* **Determinism.** ``(config, seed)`` reproduces an episode bit-for-bit, or
  doc 10's replayability claim is false and no result in the paper can be
  regenerated.
* **Snapshot / restore exactness.** Restoring and re-running must reproduce
  the original trajectory exactly, or the branch rollouts of doc 3.3 start
  from a different situation than they claim to.
* **Schema compliance.** A 60 s episode is exactly 120 logged frames and
  passes :func:`storage.schema.validate` with no complaints -- including the
  NaN policy, which is what keeps a missed detection a genuine gap.
* **Split separation.** The doc 8.1 splits differ on parameter axes, not just
  on seeds.

Run with::

    python -m unittest discover -s $PRISM_ROOT/dataGen/tests -t $PRISM_ROOT/dataGen
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.constants import (  # noqa: E402
    LOG_DECIMATION,
    LOG_DT,
    NMAC_RADIUS_M,
    PRED_STEPS,
    SIM_DT,
)
from core.types import IntentType, PlatformType  # noqa: E402
from scenario import presets  # noqa: E402
from scenario.builder import ARENA_AREA_KM2, build_scenario  # noqa: E402
from simulation.engine import SimulationEngine, default_ownship_controller  # noqa: E402
from simulation.episode import (  # noqa: E402
    episode_seeds,
    generate_batch,
    run_episode,
    separation_series,
)
from storage.schema import EPISODE_ARRAY_SPEC, validate  # noqa: E402


def _train_cfg(seed: int = 12345):
    """A training-split config with a fixed seed, for fast repeatable tests."""
    return presets.make_preset("train", seed=seed)


class TestPresets(unittest.TestCase):
    """The doc 8.1 splits must be separated structurally, not by convention."""

    def test_seed_blocks_are_disjoint(self) -> None:
        presets.assert_no_seed_overlap()
        seen: set[int] = set()
        for name in presets.PRESET_NAMES:
            lo, hi = presets.seed_block(name)
            probe = {lo, lo + 1, hi - 1}
            self.assertTrue(seen.isdisjoint(probe), f"{name} block collides")
            seen |= probe

    def test_parameter_axes_are_separated(self) -> None:
        presets.assert_parameter_axis_separation()

    def test_ood_adds_unseen_platforms_and_intents(self) -> None:
        train = presets.split_spec("train")
        ood = presets.split_spec("test_ood")
        added_p = ood.platform_support() - train.platform_support()
        added_i = ood.intent_support() - train.intent_support()
        self.assertIn(int(PlatformType.BIRD), added_p)
        self.assertIn(int(PlatformType.GA), added_p)
        self.assertIn(int(IntentType.SEARCH_WANDER), added_i)
        self.assertIn(int(IntentType.REACTIVE_AVOID), added_i)

    def test_density_axis_actually_moves_the_agent_count(self) -> None:
        # A density band that the n_agents clamp flattens would make doc 8.1's
        # primary split axis decorative.
        lo, hi = presets.split_spec("train").agent_count_band()
        self.assertLess(lo, hi, "training density band does not change n_agents")
        ood_lo, _ = presets.split_spec("test_ood").agent_count_band()
        self.assertGreater(ood_lo, hi, "OOD density does not exceed the train band")

    def test_episode_seed_stays_inside_its_block(self) -> None:
        self.assertEqual(presets.split_of_seed(presets.episode_seed("valid", 3)),
                         "valid")
        with self.assertRaises(ValueError):
            presets.episode_seed("train", presets.SEED_BLOCK_SIZE)

    def test_presets_round_trip_through_disk(self) -> None:
        paths = presets.write_presets()
        self.assertEqual(len(paths), len(presets.PRESET_NAMES))
        for name, path in zip(presets.PRESET_NAMES, paths):
            self.assertTrue(os.path.isfile(path), path)
            loaded = presets.load_preset(name)
            self.assertEqual(loaded.to_dict(), presets.make_preset(name).to_dict())
            with open(path, "r", encoding="utf-8") as fh:
                json.load(fh)  # must be parseable UTF-8 JSON

    def test_every_preset_config_validates(self) -> None:
        for name in presets.PRESET_NAMES:
            for index in (0, 1, 17):
                cfg = presets.config_for_episode(name, index)
                self.assertEqual(cfg.validate(), [], f"{name}#{index}")


class TestBuilder(unittest.TestCase):
    """Scenario assembly: population, encounters, intent resolution."""

    def test_population_matches_density(self) -> None:
        cfg = _train_cfg()
        sc = build_scenario(cfg)
        self.assertEqual(sc.n_agents, cfg.n_background())
        self.assertEqual(len(sc.agent_ids), sc.n_agents)
        self.assertEqual(len(set(sc.agent_ids)), sc.n_agents)

    def test_designed_encounters_are_recorded(self) -> None:
        cfg = _train_cfg()
        sc = build_scenario(cfg)
        self.assertGreaterEqual(len(sc.designed), cfg.encounter.n_designed_min)
        self.assertLessEqual(len(sc.designed), cfg.encounter.n_designed_max)
        for enc in sc.designed:
            self.assertTrue(cfg.encounter.cpa_dist_min_m - 1e-6
                            <= enc.cpa_dist_target
                            <= cfg.encounter.cpa_dist_max_m + 1e-6)
            self.assertTrue(cfg.encounter.tcpa_min_s - 1e-6
                            <= enc.tcpa_target
                            <= cfg.encounter.tcpa_max_s + 1e-6)
            self.assertLess(enc.slot, sc.n_agents)

    def test_debris_always_drifts_passively(self) -> None:
        # OOD is the only split that spawns DEBRIS at all.
        found = False
        for index in range(12):
            sc = build_scenario(presets.config_for_episode("test_ood", index))
            for slot, ptype in enumerate(sc.platforms):
                if int(ptype) == int(PlatformType.DEBRIS):
                    found = True
                    self.assertEqual(sc.intents[slot], int(IntentType.PASSIVE_DRIFT))
        self.assertTrue(found, "no DEBRIS agent appeared in 12 OOD episodes")

    def test_ownship_leg_fits_inside_the_arena(self) -> None:
        cfg = _train_cfg()
        sc = build_scenario(cfg)
        x0, y0, vgx, vgy = sc.ownship_track
        xmin, ymin, xmax, ymax = sc.world.bounds
        for tau in (0.0, cfg.duration_s):
            self.assertTrue(xmin <= x0 + vgx * tau <= xmax)
            self.assertTrue(ymin <= y0 + vgy * tau <= ymax)

    def test_arena_is_larger_than_the_density_footprint(self) -> None:
        self.assertGreater(ARENA_AREA_KM2, presets.DENSITY_REFERENCE_AREA_KM2)
        sc = build_scenario(_train_cfg())
        self.assertAlmostEqual(sc.realised_density_per_km2,
                               sc.n_agents / ARENA_AREA_KM2, places=9)


class TestEpisodeShape(unittest.TestCase):
    """The storage contract: 120 frames, right dtypes, clean NaN policy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ep, cls.meta = run_episode(_train_cfg(), seed=2024)

    def test_sixty_seconds_yields_exactly_120_frames(self) -> None:
        self.assertEqual(self.ep.n_steps, 120)
        self.assertAlmostEqual(float(self.ep.t[0]), LOG_DT, places=5)
        self.assertAlmostEqual(float(self.ep.t[-1]), 60.0, places=4)
        dt = np.diff(self.ep.t.astype(np.float64))
        np.testing.assert_allclose(dt, LOG_DT, rtol=0, atol=1e-4)

    def test_schema_validate_is_clean(self) -> None:
        self.assertEqual(validate(self.ep), [])
        self.assertEqual(self.meta.validate(), [])

    def test_dtypes_match_the_spec(self) -> None:
        for name, (dtype, _shape) in EPISODE_ARRAY_SPEC.items():
            self.assertEqual(getattr(self.ep, name).dtype, np.dtype(dtype), name)

    def test_observation_gaps_are_never_interpolated(self) -> None:
        missed = ~self.ep.obs_detected
        self.assertTrue(np.all(np.isnan(self.ep.obs_pos[missed])))
        self.assertTrue(np.all(np.isnan(self.ep.obs_R[missed])))
        seen = self.ep.obs_detected
        self.assertTrue(np.all(np.isfinite(self.ep.obs_pos[seen])))

    def test_ground_truth_and_observation_differ(self) -> None:
        # If they matched, the sensor layer would not be doing anything and the
        # dataset would be teaching the model a problem that does not exist.
        seen = self.ep.obs_detected
        self.assertTrue(np.any(seen), "no detections at all")
        err = np.hypot(*(self.ep.obs_pos[seen] - self.ep.gt_pos[seen]).T)
        self.assertGreater(float(np.median(err)), 0.5)
        self.assertLess(float(np.median(err)), 200.0)

    def test_some_detections_are_missing(self) -> None:
        rate = float(self.ep.obs_detected.sum()) / max(int(self.ep.gt_valid.sum()), 1)
        self.assertGreater(rate, 0.05)
        self.assertLess(rate, 0.999, "Pd < 1 must produce real gaps (doc 3.5)")

    def test_track_age_zero_iff_no_live_track(self) -> None:
        no_track = self.ep.obs_track_id < 0
        self.assertTrue(np.all(self.ep.obs_track_age[no_track] == 0))
        self.assertTrue(np.all(self.ep.obs_track_age[~no_track] > 0))

    def test_headings_are_wrapped(self) -> None:
        valid = self.ep.gt_valid
        psi = self.ep.gt_psi[valid].astype(np.float64)
        self.assertTrue(np.all(psi > -math.pi - 1e-5))
        self.assertTrue(np.all(psi <= math.pi + 1e-5))


class TestDeterminism(unittest.TestCase):
    """doc 10: every stage's output must be reproducible from its seed."""

    def test_same_seed_reproduces_every_array(self) -> None:
        a, ma = run_episode(_train_cfg(), seed=777)
        b, mb = run_episode(_train_cfg(), seed=777)
        for name in EPISODE_ARRAY_SPEC:
            np.testing.assert_array_equal(
                getattr(a, name), getattr(b, name), err_msg=name
            )
        self.assertEqual(ma.to_dict(), mb.to_dict())

    def test_different_seeds_produce_different_episodes(self) -> None:
        a, _ = run_episode(_train_cfg(), seed=777)
        b, _ = run_episode(_train_cfg(), seed=778)
        self.assertFalse(np.array_equal(a.gt_pos, b.gt_pos, equal_nan=True))

    def test_rerunning_one_engine_rewinds_cleanly(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=99))
        first, _ = engine.run_episode()
        first_pos = first.gt_pos.copy()
        second, _ = engine.run_episode()
        np.testing.assert_array_equal(first_pos, second.gt_pos)


class TestSnapshotRestore(unittest.TestCase):
    """Branch rollouts are only meaningful if restore is exact."""

    def test_restore_reproduces_the_trajectory(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=31337))
        engine.advance_to(20.0)
        snap = engine.snapshot()

        engine.advance_to(45.0, log=False)
        reference = engine.gt_positions().copy()
        ref_own = (engine.scenario.ownship.state.x, engine.scenario.ownship.state.y)
        ref_t = engine.t

        engine.restore(snap)
        self.assertAlmostEqual(engine.t, 20.0, places=9)
        engine.advance_to(45.0, log=False)

        np.testing.assert_array_equal(reference, engine.gt_positions())
        self.assertAlmostEqual(engine.scenario.ownship.state.x, ref_own[0], places=12)
        self.assertAlmostEqual(engine.scenario.ownship.state.y, ref_own[1], places=12)
        self.assertAlmostEqual(engine.t, ref_t, places=12)

    def test_one_snapshot_restores_repeatedly(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=5150))
        engine.advance_to(10.0)
        snap = engine.snapshot()
        runs = []
        for _ in range(3):
            engine.restore(snap)
            engine.advance(200, log=False)
            runs.append(engine.gt_positions().copy())
        np.testing.assert_array_equal(runs[0], runs[1])
        np.testing.assert_array_equal(runs[0], runs[2])

    def test_restore_also_rewinds_the_observation_chain(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=606))
        engine.advance_to(15.0)
        snap = engine.snapshot()
        engine.advance_to(30.0)
        obs_a = engine.ep.obs_pos[:60].copy()
        det_a = engine.ep.obs_detected[:60].copy()

        engine.restore(snap)
        engine._log_index = int(round(15.0 / LOG_DT))  # rewind the write cursor
        engine.advance_to(30.0)
        np.testing.assert_array_equal(det_a, engine.ep.obs_detected[:60])
        np.testing.assert_array_equal(obs_a, engine.ep.obs_pos[:60])

    def test_restoring_a_foreign_snapshot_is_refused(self) -> None:
        a = SimulationEngine(_train_cfg(seed=1))
        b = SimulationEngine(presets.make_preset("test_ood", seed=300001))
        with self.assertRaises(ValueError):
            a.restore(b.snapshot())


class TestDatasetDifficulty(unittest.TestCase):
    """doc 3.1 / 8.3: the data must be hard for a constant-velocity extrapolator."""

    def test_cv_fifteen_second_error_is_the_right_order_of_magnitude(self) -> None:
        # doc 8.3: too small and CV wins outright, so the paper collapses; too
        # large and the trajectories are noise nobody could predict. The stated
        # target is the same order of magnitude as the NMAC radius.
        errors = []
        for seed in (11, 22, 33):
            ep, _ = run_episode(_train_cfg(), seed=seed)
            horizon = PRED_STEPS
            for i in range(0, ep.n_steps - horizon):
                mask = ep.gt_valid[i] & ep.gt_valid[i + horizon]
                if not np.any(mask):
                    continue
                pred = ep.gt_pos[i][mask] + ep.gt_vel[i][mask] * (horizon * LOG_DT)
                errors.append(np.hypot(*(pred - ep.gt_pos[i + horizon][mask]).T))
        err = np.concatenate(errors)
        median = float(np.median(err))
        self.assertGreater(median, 0.1 * NMAC_RADIUS_M,
                           "trajectories are too straight: CV would win")
        self.assertLess(median, 10.0 * NMAC_RADIUS_M,
                        "trajectories are too random to be predictable at all")

    def test_agents_actually_manoeuvre(self) -> None:
        ep, _ = run_episode(_train_cfg(), seed=4242)
        modes = ep.gt_mode[ep.gt_valid]
        self.assertGreater(len(np.unique(modes)), 1,
                           "every agent stayed in one maneuver mode")
        turning = float(np.mean((modes == 1) | (modes == 2)))
        self.assertGreater(turning, 0.02, "almost no coordinated turns occur")


class TestOwnshipController(unittest.TestCase):
    """The ownship hook is where the PPO policy of doc 7 will be plugged in."""

    def test_default_controller_holds_the_corridor(self) -> None:
        ep, _ = run_episode(_train_cfg(), seed=808)
        psi = np.unwrap(ep.own_psi.astype(np.float64))
        self.assertLess(float(np.max(np.abs(psi - psi[0]))), math.radians(15.0))

    def test_custom_controller_is_used(self) -> None:
        calls: list[float] = []

        def turning(t, state, envelope, context):
            calls.append(t)
            return envelope.max_turn_rate(state.v), 0.0

        ep, _ = run_episode(_train_cfg(), seed=808,
                            ownship_controller=turning, validate=True)
        self.assertGreater(len(calls), 1000)
        psi = np.unwrap(ep.own_psi.astype(np.float64))
        self.assertGreater(float(np.max(np.abs(psi - psi[0]))), math.radians(90.0))

    def test_default_controller_respects_the_envelope(self) -> None:
        sc = build_scenario(_train_cfg())
        env = sc.ownship.envelope
        omega, accel = default_ownship_controller(
            0.0, sc.ownship.state, env,
            {"psi_ref": sc.ownship.psi_ref + 3.0, "v_ref": 1e6, "world": None},
        )
        self.assertLessEqual(abs(omega), env.max_turn_rate(sc.ownship.state.v) + 1e-12)
        self.assertLessEqual(accel, env.a_max + 1e-12)


class TestMetadata(unittest.TestCase):
    """doc 9.4 metrics and doc 3.4's stratification labels."""

    def test_min_separation_is_measured_at_the_integration_rate(self) -> None:
        """min_separation_m comes from 20 Hz state, not the 2 Hz log.

        The logged series undersamples the closest point of approach: the true
        minimum can fall between two logged frames, and the error is one-sided
        (the logged minimum is never below the true one). Measured at up to
        19 m, 12% of NMAC_RADIUS_M -- enough to label a real NMAC as safe.
        """
        engine = SimulationEngine(_train_cfg(seed=1234))
        ep, meta = engine.run_episode()
        series = separation_series(ep)
        logged = float(np.nanmin(
            np.where(series["valid"], series["range_m"], np.inf)
        ))
        # Never above the logged minimum, and never absurdly below it.
        self.assertLessEqual(meta.min_separation_m, logged + 1e-6)
        self.assertGreater(meta.min_separation_m, 0.0)
        self.assertGreater(meta.min_separation_m, logged - 200.0)

        # And it equals an independent 20 Hz recomputation.
        fine = float("inf")
        replay = SimulationEngine(_train_cfg(seed=1234))
        while not replay.is_done:
            replay.step(log=True)
            ox = replay.scenario.ownship.state.x
            oy = replay.scenario.ownship.state.y
            for agent in replay.scenario.agents:
                if agent.alive:
                    fine = min(fine, math.hypot(agent.state.x - ox,
                                                agent.state.y - oy))
        self.assertAlmostEqual(meta.min_separation_m, min(fine, logged), places=6)

    def test_reactive_avoid_agents_actually_react(self) -> None:
        """The REACTIVE_AVOID label must carry its own behaviour (doc 3.2).

        agents.behavior.reactive_avoid_decision was unreachable dead code and a
        tighter local fallback ran instead: 68 agents produced 9 reactions and
        87% of the class flew a plain wander, i.e. was behaviourally identical
        to SEARCH_WANDER. Without this assertion that regresses silently.
        """
        from agents.intent import (
            RA_CPA_TRIGGER_M, RA_DETECT_RANGE_M, RA_MIN_CONDITIONAL_FRACTION,
            RA_MIN_REACTING_FRACTION, RA_MIN_TRIGGER_RATE, RA_TCPA_MAX_S,
        )
        from core.types import IntentType
        from scenario.presets import config_for_episode

        n_agents = triggers = reacted = qualified = qual_reacted = 0
        for index in range(8):
            engine = SimulationEngine(config_for_episode("test_ood", index))
            ep, _meta = engine.run_episode()
            rel = ep.gt_pos.astype(np.float64) - ep.own_pos.astype(np.float64)[:, None, :]
            relv = ep.gt_vel.astype(np.float64) - ep.own_vel.astype(np.float64)[:, None, :]
            rng_m = np.hypot(rel[..., 0], rel[..., 1])
            dot = rel[..., 0] * relv[..., 0] + rel[..., 1] * relv[..., 1]
            v2 = np.maximum(relv[..., 0] ** 2 + relv[..., 1] ** 2, 1e-9)
            tcpa = np.clip(-dot / v2, 0.0, None)
            cpa = np.hypot(rel[..., 0] + relv[..., 0] * tcpa,
                           rel[..., 1] + relv[..., 1] * tcpa)
            for slot, agent in enumerate(engine.scenario.agents):
                ctrl = agent.intent
                if getattr(ctrl, "INTENT", None) is not IntentType.REACTIVE_AVOID:
                    continue
                n_agents += 1
                n = int(ctrl.state_dict.get("avoids", 0))
                triggers += n
                reacted += 1 if n else 0
                # Did this agent EVER see a geometry the model should fire on?
                qualifies = np.any(
                    ep.gt_valid[:, slot]
                    & (rng_m[:, slot] < RA_DETECT_RANGE_M)
                    & (dot[:, slot] < 0.0)
                    & (tcpa[:, slot] <= RA_TCPA_MAX_S)
                    & (cpa[:, slot] < RA_CPA_TRIGGER_M)
                )
                if qualifies:
                    qualified += 1
                    qual_reacted += 1 if n else 0
        self.assertGreater(n_agents, 10, "no REACTIVE_AVOID agents to measure")
        rate = triggers / n_agents
        frac = reacted / n_agents
        self.assertGreaterEqual(
            rate, RA_MIN_TRIGGER_RATE,
            f"only {rate:.2f} reactions per REACTIVE_AVOID agent "
            f"({triggers} over {n_agents} agents)")
        self.assertGreaterEqual(
            frac, RA_MIN_REACTING_FRACTION,
            f"only {frac * 100:.0f}% of REACTIVE_AVOID agents ever reacted")
        # The assertion with teeth: the raw rate above is capped by how often
        # the ownship develops a conflict at all, which is a scenario property.
        # This one is purely the reaction model's own responsibility.
        self.assertGreater(qualified, 5, "too few qualifying geometries to judge")
        self.assertGreaterEqual(
            qual_reacted / qualified, RA_MIN_CONDITIONAL_FRACTION,
            f"only {qual_reacted}/{qualified} agents with a well-clear-losing "
            f"geometry actually reacted")

    def test_nmac_implies_dwc(self) -> None:
        result = generate_batch(_train_cfg(), 8)
        for meta in result.metas:
            if meta.nmac_occurred:
                self.assertTrue(meta.dwc_violated)
            self.assertEqual(meta.validate(), [])

    def test_nmac_flag_agrees_with_the_threshold(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=2468))
        ep, meta = engine.run_episode()
        self.assertEqual(meta.nmac_occurred, meta.min_separation_m < NMAC_RADIUS_M)

    def test_designed_encounter_metadata_is_aligned(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=13579))
        ep, meta = engine.run_episode()
        n = len(engine.scenario.designed)
        self.assertEqual(len(meta.geometry_type), n)
        self.assertEqual(len(meta.cpa_dist), n)
        self.assertEqual(len(meta.tcpa), n)
        for value in meta.tcpa:
            self.assertTrue(0.0 <= value <= engine.cfg.duration_s + 1e-6)

    def test_platform_counts_sum_to_n_agents(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=97))
        _, meta = engine.run_episode()
        self.assertEqual(sum(meta.platform_counts.values()), meta.n_agents)
        self.assertEqual(sum(meta.intent_counts.values()), meta.n_agents)

    def test_meta_round_trips_through_json(self) -> None:
        _, meta = run_episode(_train_cfg(), seed=64)
        text = json.dumps(meta.to_dict())
        self.assertEqual(type(meta).from_dict(json.loads(text)).to_dict(),
                         meta.to_dict())


class TestBatch(unittest.TestCase):
    """One batch path shared by the CLI and the GUI."""

    def test_batch_uses_consecutive_seeds_and_calls_back(self) -> None:
        cfg = _train_cfg(seed=500)
        seen: list[tuple[int, int]] = []
        sunk: list[int] = []
        result = generate_batch(
            cfg, 3,
            progress_cb=lambda i, n, m: seen.append((i, n)),
            sink=lambda arrays, meta: sunk.append(meta.episode_index),
        )
        self.assertEqual(result.seeds, [500, 501, 502])
        self.assertEqual(seen, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(sunk, [0, 1, 2])
        self.assertEqual(result.n_ok, 3)
        self.assertEqual(result.n_failed, 0)

    def test_batch_rejects_duplicate_seeds(self) -> None:
        with self.assertRaises(ValueError):
            episode_seeds(_train_cfg(), 3, [1, 1, 2])
        with self.assertRaises(ValueError):
            episode_seeds(_train_cfg(), 3, [1, 2])

    def test_summary_reports_rates(self) -> None:
        result = generate_batch(_train_cfg(seed=900), 5)
        summary = result.summary()
        self.assertEqual(summary["n_ok"], 5)
        self.assertTrue(0.0 <= summary["nmac_rate"] <= 1.0)
        self.assertTrue(0.0 <= summary["dwc_rate"] <= 1.0)


class TestGuiFrame(unittest.TestCase):
    """``current_frame`` is sent at ~20 fps, so it must stay small and JSON-safe."""

    def test_frame_is_json_serialisable(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=246))
        engine.advance_to(5.0)
        frame = engine.current_frame()
        text = json.dumps(frame)
        self.assertEqual(json.loads(text)["t"], frame["t"])

    def test_frame_carries_what_the_gui_draws(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=246))
        engine.advance_to(5.0)
        frame = engine.current_frame()
        for key in ("t", "wind", "own", "agents", "statics"):
            self.assertIn(key, frame)
        self.assertEqual(len(frame["agents"]), engine.scenario.n_agents)
        entry = frame["agents"][0]
        for key in ("id", "x", "y", "psi", "type", "mode", "intent",
                    "det", "tid", "age", "cpa_d", "cpa_t"):
            self.assertIn(key, entry)

    def test_frame_stays_compact_without_statics(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=246))
        engine.advance_to(5.0)
        size = len(json.dumps(engine.current_frame(include_statics=False)))
        self.assertLess(size, 8000, f"frame is {size} bytes; too big for 20 fps")


class TestTiming(unittest.TestCase):
    """The integrate-fine / log-coarse contract of doc 3.7."""

    def test_ten_integration_ticks_per_logged_frame(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=1))
        self.assertEqual(engine.n_logged, 0)
        engine.advance(LOG_DECIMATION - 1)
        self.assertEqual(engine.n_logged, 0)
        engine.step()
        self.assertEqual(engine.n_logged, 1)
        self.assertAlmostEqual(engine.t, LOG_DECIMATION * SIM_DT, places=12)

    def test_clock_does_not_drift(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=1))
        engine.advance(1200)
        self.assertEqual(engine.t, 60.0)
        self.assertTrue(engine.is_done)

    def test_logging_can_be_suppressed(self) -> None:
        engine = SimulationEngine(_train_cfg(seed=1))
        engine.advance(LOG_DECIMATION, log=False)
        self.assertEqual(engine.n_logged, 0)
        self.assertTrue(np.all(engine.ep.t == 0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
