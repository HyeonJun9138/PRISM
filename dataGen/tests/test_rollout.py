"""Tests for branch rollouts -- the calibration ground truth of doc 3.3.

The whole paper rests on one claim: replaying the same situation N times
yields the TRUE distribution of futures, against which a predictor's stated
uncertainty can be scored. Three things must therefore hold, and each has a
test here:

* **Identical seeds give identical futures.** If they did not, the rollout
  would be adding noise of its own and the "true" distribution would be an
  artefact of the harness.
* **Different seeds give different futures.** If they did not, the true
  distribution would be a point mass and every predictor -- however
  overconfident -- would score as perfectly calibrated. This is the silent
  failure mode, so the degeneracy check is itself tested against an engine
  deliberately broken to reproduce it.
* **Only the FUTURE is re-randomised.** The past, the population and the
  platform envelopes must survive a branch untouched, or the branches are
  rollouts of different scenarios.

Run with::

    python $PRISM_ROOT/dataGen/tests/test_rollout.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.constants import LOG_DT, PRED_SECONDS, PRED_STEPS  # noqa: E402
from scenario import presets  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.rollout import (  # noqa: E402
    BRANCH_STREAMS,
    DegenerateRolloutError,
    branch_rollout,
    branch_seed_block,
    chi2_2dof_quantile,
    coverage_at,
    rollout_statistics,
)

#: Branch point [s]. A logged instant, well inside the episode so the horizon
#: fits without the engine having to run past its own duration.
BRANCH_T: float = 20.0


def _engine(seed: int = 4242) -> SimulationEngine:
    """A training-split engine with a fixed seed."""
    return SimulationEngine(presets.make_preset("train", seed=seed))


class _FrozenEngine(SimulationEngine):
    """An engine whose branch re-seeding does nothing.

    Reproduces the exact defect :class:`DegenerateRolloutError` exists to
    catch -- a reseed that never reaches the agents' decision draws, e.g.
    because the mode machines were left bound to pre-reseed generators.
    """

    def reseed_branch(self, branch_seed, names):  # noqa: D102 - see class docstring
        return None


class TestBranchRolloutShape(unittest.TestCase):
    """Shape, dtype and the default prediction grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _engine()
        cls.rollouts = branch_rollout(cls.engine, BRANCH_T, 12)

    def test_shape_matches_the_prediction_horizon(self) -> None:
        self.assertEqual(
            self.rollouts.shape,
            (12, self.engine.n_agents, PRED_STEPS, 2),
        )

    def test_dtype_is_float64(self) -> None:
        # Analysis product, not stored data: coverage statistics must not
        # inherit the float32 rounding of the storage schema.
        self.assertEqual(self.rollouts.dtype, np.float64)

    def test_default_horizon_is_fifteen_seconds(self) -> None:
        self.assertEqual(PRED_STEPS, int(round(PRED_SECONDS / LOG_DT)))

    def test_positions_are_finite_for_live_agents(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.rollouts)))

    def test_shorter_horizon_gives_fewer_steps(self) -> None:
        short = branch_rollout(self.engine, BRANCH_T, 2, horizon_s=5.0)
        self.assertEqual(short.shape[2], int(round(5.0 / LOG_DT)))


class TestBranchDivergence(unittest.TestCase):
    """The correctness property the calibration ground truth depends on."""

    def test_identical_seeds_are_bit_identical(self) -> None:
        engine = _engine(seed=101)
        rollouts = branch_rollout(
            engine, BRANCH_T, 4,
            branch_seeds=[9, 9, 9, 9], require_divergence=False,
        )
        for b in range(1, 4):
            np.testing.assert_array_equal(rollouts[0], rollouts[b])

    def test_different_seeds_diverge(self) -> None:
        engine = _engine(seed=101)
        rollouts = branch_rollout(engine, BRANCH_T, 6,
                                  branch_seeds=[1, 2, 3, 4, 5, 6])
        pairwise_equal = [
            np.array_equal(rollouts[0], rollouts[b], equal_nan=True)
            for b in range(1, 6)
        ]
        self.assertFalse(any(pairwise_equal), "no branch differed from branch 0")

    def test_divergence_is_physically_meaningful(self) -> None:
        # A few centimetres of float noise would satisfy "not equal" while
        # still being a point mass in practice. The spread has to be big
        # enough that a probabilistic predictor can actually beat a point
        # estimate on it.
        engine = _engine(seed=202)
        rollouts = branch_rollout(engine, BRANCH_T, 16)
        spread = rollout_statistics(rollouts)["spread"][:, -1]
        self.assertGreater(float(np.nanmax(spread)), 20.0)

    def test_uncertainty_grows_with_the_horizon(self) -> None:
        engine = _engine(seed=303)
        rollouts = branch_rollout(engine, BRANCH_T, 16)
        spread = rollout_statistics(rollouts)["spread"]
        early = float(np.nanmedian(spread[:, 0]))
        late = float(np.nanmedian(spread[:, -1]))
        self.assertGreater(late, early,
                           "the future did not get less certain with time")

    def test_degeneracy_is_detected_and_reported(self) -> None:
        engine = _FrozenEngine(presets.make_preset("train", seed=404))
        with self.assertRaises(DegenerateRolloutError) as ctx:
            branch_rollout(engine, BRANCH_T, 4, branch_seeds=[1, 2, 3, 4])
        message = str(ctx.exception)
        self.assertIn("bit-identical", message)
        self.assertIn("distinct seeds", message)

    def test_degeneracy_check_can_be_disabled(self) -> None:
        engine = _FrozenEngine(presets.make_preset("train", seed=404))
        rollouts = branch_rollout(engine, BRANCH_T, 3,
                                  branch_seeds=[1, 2, 3], require_divergence=False)
        self.assertEqual(rollouts.shape[0], 3)


class TestPastIsPreserved(unittest.TestCase):
    """A branch must re-randomise the future only -- never the situation."""

    def test_all_branches_start_from_the_same_state(self) -> None:
        engine = _engine(seed=505)
        engine.advance_to(BRANCH_T)
        start = engine.gt_positions().copy()
        branch_rollout(engine, BRANCH_T, 5)
        np.testing.assert_array_equal(start, engine.gt_positions())

    def test_population_survives_a_branch(self) -> None:
        engine = _engine(seed=505)
        platforms = list(engine.scenario.platforms)
        intents = list(engine.scenario.intents)
        envelopes = [a.envelope.v_max for a in engine.scenario.agents]
        branch_rollout(engine, BRANCH_T, 4)
        self.assertEqual(platforms, list(engine.scenario.platforms))
        self.assertEqual(intents, list(engine.scenario.intents))
        self.assertEqual(envelopes,
                         [a.envelope.v_max for a in engine.scenario.agents])

    def test_engine_clock_is_restored(self) -> None:
        engine = _engine(seed=606)
        branch_rollout(engine, BRANCH_T, 3)
        self.assertAlmostEqual(engine.t, BRANCH_T, places=9)

    def test_rollouts_do_not_touch_the_logged_episode(self) -> None:
        engine = _engine(seed=707)
        engine.advance_to(BRANCH_T)
        logged = engine.n_logged
        gt_before = engine.ep.gt_pos.copy()
        obs_before = engine.ep.obs_pos.copy()
        branch_rollout(engine, BRANCH_T, 4)
        self.assertEqual(engine.n_logged, logged)
        np.testing.assert_array_equal(gt_before, engine.ep.gt_pos)
        np.testing.assert_array_equal(obs_before, engine.ep.obs_pos)

    def test_episode_continues_correctly_after_a_branch(self) -> None:
        # The dataset must be unaffected by having rollouts taken from it.
        plain = _engine(seed=808)
        plain.run_episode()
        reference = plain.ep.gt_pos.copy()

        branched = _engine(seed=808)
        branched.advance_to(BRANCH_T)
        branch_rollout(branched, BRANCH_T, 4)
        while not branched.is_done:
            branched.step()
        np.testing.assert_array_equal(reference, branched.ep.gt_pos)

    def test_branch_streams_exclude_the_situation_streams(self) -> None:
        for name in ("placement", "platform", "intent", "scenario"):
            self.assertNotIn(name, BRANCH_STREAMS)
        for name in ("mode_decision", "branch_choice"):
            self.assertIn(name, BRANCH_STREAMS)


class TestSeeds(unittest.TestCase):
    """Seed handling and argument validation."""

    def test_default_seeds_are_distinct_and_reproducible(self) -> None:
        a = branch_seed_block(1000, 8)
        self.assertEqual(len(set(a)), 8)
        self.assertEqual(a, branch_seed_block(1000, 8))

    def test_default_seeds_reproduce_the_same_rollout(self) -> None:
        first = branch_rollout(_engine(seed=909), BRANCH_T, 4)
        second = branch_rollout(_engine(seed=909), BRANCH_T, 4)
        np.testing.assert_array_equal(first, second)

    def test_invalid_arguments_are_rejected(self) -> None:
        engine = _engine(seed=111)
        with self.assertRaises(ValueError):
            branch_rollout(engine, BRANCH_T, 0)
        with self.assertRaises(ValueError):
            branch_rollout(engine, BRANCH_T, 2, horizon_s=0.0)
        with self.assertRaises(ValueError):
            branch_rollout(engine, BRANCH_T, 3, branch_seeds=[1, 2])


class TestRolloutStatistics(unittest.TestCase):
    """Moments of the empirical distribution."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _engine(seed=1212)
        cls.rollouts = branch_rollout(cls.engine, BRANCH_T, 20)
        cls.stats = rollout_statistics(cls.rollouts)

    def test_shapes(self) -> None:
        n = self.engine.n_agents
        self.assertEqual(self.stats["mean"].shape, (n, PRED_STEPS, 2))
        self.assertEqual(self.stats["cov"].shape, (n, PRED_STEPS, 2, 2))
        self.assertEqual(self.stats["std"].shape, (n, PRED_STEPS, 2))
        self.assertEqual(self.stats["spread"].shape, (n, PRED_STEPS))
        self.assertEqual(self.stats["count"].shape, (n, PRED_STEPS))

    def test_counts_match_the_branch_number(self) -> None:
        self.assertTrue(np.all(self.stats["count"] == 20))

    def test_mean_matches_numpy(self) -> None:
        expected = self.rollouts.mean(axis=0)
        np.testing.assert_allclose(self.stats["mean"], expected, atol=1e-9)

    def test_covariance_is_symmetric_and_psd(self) -> None:
        cov = self.stats["cov"]
        np.testing.assert_allclose(cov[..., 0, 1], cov[..., 1, 0], atol=1e-9)
        self.assertTrue(np.all(cov[..., 0, 0] >= -1e-12))
        self.assertTrue(np.all(cov[..., 1, 1] >= -1e-12))
        det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
        self.assertTrue(np.all(det >= -1e-6))

    def test_spread_is_the_root_trace(self) -> None:
        cov = self.stats["cov"]
        expected = np.sqrt(cov[..., 0, 0] + cov[..., 1, 1])
        np.testing.assert_allclose(self.stats["spread"], expected, atol=1e-9)

    def test_single_branch_has_no_covariance(self) -> None:
        stats = rollout_statistics(self.rollouts[:1])
        self.assertTrue(np.all(np.isnan(stats["cov"])))
        self.assertTrue(np.all(np.isfinite(stats["mean"])))

    def test_missing_agents_are_dropped_not_imputed(self) -> None:
        arr = self.rollouts.copy()
        arr[0, 0, :, :] = np.nan
        stats = rollout_statistics(arr)
        self.assertEqual(int(stats["count"][0, 0]), arr.shape[0] - 1)
        self.assertTrue(np.isfinite(stats["mean"][0, 0]).all())

    def test_bad_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rollout_statistics(np.zeros((3, 4, 5)))


class TestCoverage(unittest.TestCase):
    """The empirical-coverage helper the analysis stage reuses."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _engine(seed=1313)
        cls.rollouts = branch_rollout(cls.engine, BRANCH_T, 40)

    def test_chi2_quantile_closed_form(self) -> None:
        # 2 dof: CDF is 1 - exp(-c/2), so the median is 2*ln(2).
        self.assertAlmostEqual(chi2_2dof_quantile(0.5), 2.0 * math.log(2.0), places=12)
        self.assertAlmostEqual(chi2_2dof_quantile(0.0), 0.0, places=12)
        self.assertAlmostEqual(chi2_2dof_quantile(0.9), 4.605170185988091, places=9)
        with self.assertRaises(ValueError):
            chi2_2dof_quantile(1.0)
        with self.assertRaises(ValueError):
            chi2_2dof_quantile(-0.1)

    def test_coverage_is_a_probability(self) -> None:
        result = coverage_at(self.rollouts, 0.9)
        cov = result["coverage"]
        finite = cov[np.isfinite(cov)]
        self.assertGreater(finite.size, 0)
        self.assertTrue(np.all(finite >= 0.0))
        self.assertTrue(np.all(finite <= 1.0))
        self.assertTrue(0.0 <= result["overall"] <= 1.0)

    def test_coverage_tracks_the_nominal_level(self) -> None:
        # Not an equality: where the true distribution forks it is genuinely
        # non-Gaussian, and the gap is the signal of doc 9.5 rather than an
        # error. A loose band still catches an inverted or broken ellipse.
        for level in (0.5, 0.9):
            overall = coverage_at(self.rollouts, level)["overall"]
            self.assertGreater(overall, level - 0.25, f"level {level}")
            self.assertLess(overall, min(1.0, level + 0.35), f"level {level}")

    def test_higher_level_covers_more(self) -> None:
        low = coverage_at(self.rollouts, 0.5)["overall"]
        high = coverage_at(self.rollouts, 0.95)["overall"]
        self.assertGreaterEqual(high, low)

    def test_gaussian_sample_is_calibrated(self) -> None:
        # Sanity check on the mathematics itself, independent of the simulator:
        # a genuine bivariate normal must cover its own 90% ellipse ~90% of
        # the time.
        rng = np.random.default_rng(0)
        sample = rng.normal(size=(4000, 1, 1, 2)) * np.array([30.0, 10.0])
        result = coverage_at(sample, 0.9)
        self.assertAlmostEqual(result["overall"], 0.9, delta=0.02)

    def test_precomputed_stats_are_reused(self) -> None:
        stats = rollout_statistics(self.rollouts)
        a = coverage_at(self.rollouts, 0.9)
        b = coverage_at(self.rollouts, 0.9, stats=stats)
        np.testing.assert_array_equal(a["coverage"], b["coverage"])

    def test_bad_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            coverage_at(np.zeros((2, 3)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
