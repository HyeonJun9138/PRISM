"""Correctness tests for the onboard sensor model and the track manager.

The four checks demanded by the dataGen spec are, in order:

    1. the empirical detection rate at short range equals
       ``pd_max * rcs_factor`` within Monte-Carlo tolerance;
    2. the empirical Cartesian covariance of 20 000 samples matches the
       reported ``R`` at both 200 m and 2000 m -- this is the real correctness
       check on the polar-to-Cartesian Jacobian, and it is the reason the
       stored ``obs_R`` can be trusted as a model input;
    3. a forced dropout longer than ``drop_after_misses`` yields a NEW
       ``track_id`` with the age reset to zero;
    4. no interpolated value ever appears -- every undetected slot is NaN.

Run directly::

    python $PRISM_ROOT/dataGen/tests/test_sensors.py

Units: metres, seconds, radians.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE).replace("\\", "/")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.rng import RNGHub                                    # noqa: E402
from core.state import AgentState, wrap_angle                  # noqa: E402
from core.types import IntentType, ManeuverMode, PlatformType  # noqa: E402
from scenario.config import SensorConfig                       # noqa: E402
from sensors.model import Measurement, SensorModel, SensorTarget, as_target  # noqa: E402
from sensors.tracker import TrackManager                       # noqa: E402
from storage import schema                                     # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _flat_rcs(value: float = 1.0) -> dict[int, float]:
    """Return an RCS table [-] with the same factor for every platform type."""
    return {int(p): float(value) for p in PlatformType}


def _hub(seed: int = 12345) -> RNGHub:
    """Return a fresh RNG hub with the canonical sensor streams available."""
    return RNGHub(seed)


def _streams(hub: RNGHub):
    """Return the ``(noise, detect, latency)`` generator triple."""
    return (
        hub.stream("sensor_noise"),
        hub.stream("sensor_detect"),
        hub.stream("sensor_latency"),
    )


def _target(aid: int, x: float, y: float, ptype: PlatformType,
            v: float = 0.0, psi: float = 0.0) -> SensorTarget:
    """Build a stationary sensor target at ``(x, y)`` [m]."""
    return SensorTarget(aid, AgentState(x=x, y=y, v=v, psi=psi), ptype)


# --------------------------------------------------------------------------
# 1. detection probability
# --------------------------------------------------------------------------


class TestDetectionProbability(unittest.TestCase):
    """Pd(r) shape, RCS scaling and the empirical short-range rate."""

    def test_pd_curve_shape(self) -> None:
        """Pd is flat inside r_full, decays smoothly, and is zero past r_max."""
        cfg = SensorConfig(pd_max=0.9, r_full_m=800.0, r_max_m=3000.0,
                           rcs_factor=_flat_rcs(1.0))
        sm = SensorModel(cfg)
        self.assertAlmostEqual(sm.detection_probability(0.0), 0.9, places=12)
        self.assertAlmostEqual(sm.detection_probability(799.0), 0.9, places=12)
        self.assertAlmostEqual(sm.detection_probability(800.0), 0.9, places=12)
        self.assertAlmostEqual(sm.detection_probability(3000.0), 0.0, places=12)
        self.assertAlmostEqual(sm.detection_probability(5000.0), 0.0, places=12)
        # Monotone decreasing and continuous across the transition band.
        rs = np.linspace(800.0, 3000.0, 200)
        pds = np.array([sm.detection_probability(float(r)) for r in rs])
        self.assertTrue(np.all(np.diff(pds) <= 1e-12))
        mid = sm.detection_probability(1900.0)
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 0.9)

    def test_pd_is_clipped_to_one(self) -> None:
        """pd_max * rcs > 1 saturates at 1 instead of exceeding it."""
        cfg = SensorConfig(pd_max=0.95, rcs_factor=_flat_rcs(1.0))
        cfg.rcs_factor[int(PlatformType.GA)] = 1.3
        sm = SensorModel(cfg)
        self.assertAlmostEqual(
            sm.detection_probability(100.0, PlatformType.GA), 1.0, places=12
        )

    def test_empirical_short_range_rate(self) -> None:
        """Empirical detection rate matches pd_max*rcs within MC tolerance.

        20 000 update opportunities per platform type at r = 200 m (well
        inside ``r_full_m``), so the range falloff is exactly 1 and the only
        factors are ``pd_max`` and the per-type RCS multiplier.
        """
        pd_max = 0.95
        rcs = {int(p): 1.0 for p in PlatformType}
        rcs[int(PlatformType.BIRD)] = 0.40
        rcs[int(PlatformType.MULTIROTOR)] = 0.60
        cfg = SensorConfig(
            pd_max=pd_max,
            r_full_m=800.0,
            r_max_m=3000.0,
            update_rate_hz=5.0,
            async_phase=False,
            latency_s=0.0,
            latency_jitter_s=0.0,
            rcs_factor=rcs,
        )
        sm = SensorModel(cfg, phase_seed=7)
        rng_n, rng_d, rng_l = _streams(_hub(2024))

        own = AgentState(x=0.0, y=0.0, v=50.0, psi=0.3)
        n_each = 40
        targets = []
        for i in range(n_each):
            ang = 2.0 * math.pi * i / n_each
            targets.append(
                _target(i, 200.0 * math.cos(ang), 200.0 * math.sin(ang),
                        PlatformType.BIRD)
            )
            targets.append(
                _target(1000 + i, 200.0 * math.cos(ang), 200.0 * math.sin(ang),
                        PlatformType.MULTIROTOR)
            )

        n_steps = 500  # 40 targets x 500 opportunities = 20 000 per type
        hits = {int(PlatformType.BIRD): 0, int(PlatformType.MULTIROTOR): 0}
        opps = {int(PlatformType.BIRD): 0, int(PlatformType.MULTIROTOR): 0}
        ptype_of = {t.agent_id: int(t.platform_type) for t in targets}
        for k in range(n_steps):
            t = 0.2 * k
            for m in sm.measure(t, own, targets, rng_n, rng_d, rng_l):
                p = ptype_of[m.agent_id]
                opps[p] += 1
                hits[p] += int(m.detected)

        for ptype, expected in (
            (int(PlatformType.BIRD), pd_max * 0.40),
            (int(PlatformType.MULTIROTOR), pd_max * 0.60),
        ):
            self.assertEqual(opps[ptype], n_each * n_steps)
            rate = hits[ptype] / opps[ptype]
            sigma = math.sqrt(expected * (1.0 - expected) / opps[ptype])
            self.assertLess(
                abs(rate - expected), 4.0 * sigma,
                msg=(f"platform {ptype}: empirical Pd {rate:.4f} vs expected "
                     f"{expected:.4f} (4 sigma = {4.0 * sigma:.4f})"),
            )

    def test_no_detection_beyond_r_max(self) -> None:
        """A target past r_max is never detected, so misses are deterministic."""
        cfg = SensorConfig(pd_max=1.0, r_full_m=800.0, r_max_m=3000.0,
                           update_rate_hz=5.0, async_phase=False,
                           latency_s=0.0, latency_jitter_s=0.0,
                           rcs_factor=_flat_rcs(1.0))
        sm = SensorModel(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(5))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgts = [_target(0, 4000.0, 0.0, PlatformType.MULTIROTOR)]
        for k in range(50):
            for m in sm.measure(0.2 * k, own, tgts, rng_n, rng_d, rng_l):
                self.assertFalse(m.detected)
                self.assertTrue(np.all(np.isnan(m.pos_xy)))


# --------------------------------------------------------------------------
# 2. covariance / Jacobian
# --------------------------------------------------------------------------


class TestMeasurementCovariance(unittest.TestCase):
    """The reported R must be the covariance the samples actually have."""

    #: Ownship heading [rad] used for the empirical runs; deliberately not
    #: zero, so an error in the rotation into world axes cannot hide.
    PSI_OWN = 0.7
    #: Bearing [rad] of the target relative to the ownship heading.
    THETA_REL = -0.9

    def _run_samples(self, r_true: float, n: int = 20000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect ``n`` measurements of a static target at range ``r_true``.

        Returns:
            ``(positions [n,2] m, mean reported R [2,2] m^2,
            analytic R at the true geometry [2,2] m^2)``.
        """
        cfg = SensorConfig(
            sigma_r0_m=3.0,
            sigma_r_frac=0.01,
            sigma_theta_rad=0.0087,
            pd_max=1.0,
            r_full_m=5000.0,
            r_max_m=6000.0,
            update_rate_hz=5.0,
            async_phase=False,
            latency_s=0.0,
            latency_jitter_s=0.0,
            rcs_factor=_flat_rcs(1.0),
        )
        sm = SensorModel(cfg, phase_seed=3)
        rng_n, rng_d, rng_l = _streams(_hub(98765))

        own = AgentState(x=100.0, y=-50.0, v=55.0, psi=self.PSI_OWN)
        alpha = self.PSI_OWN + self.THETA_REL
        tx = own.x + r_true * math.cos(alpha)
        ty = own.y + r_true * math.sin(alpha)

        n_targets = 50
        targets = [
            _target(i, tx, ty, PlatformType.FIXED_WING_UAS) for i in range(n_targets)
        ]
        n_steps = int(math.ceil(n / n_targets))

        pos = np.empty((n_steps * n_targets, 2), dtype=np.float64)
        rsum = np.zeros((2, 2), dtype=np.float64)
        k = 0
        for step in range(n_steps):
            for m in sm.measure(0.2 * step, own, targets, rng_n, rng_d, rng_l):
                self.assertTrue(m.detected)
                pos[k] = m.pos_xy
                rsum += m.R_2x2
                k += 1
        self.assertEqual(k, n_steps * n_targets)

        sigma_r = cfg.sigma_r0_m + cfg.sigma_r_frac * r_true
        r_analytic = SensorModel.cartesian_covariance(
            r_true, self.THETA_REL, self.PSI_OWN, sigma_r, cfg.sigma_theta_rad
        )
        return pos[:k], rsum / float(k), r_analytic

    def _check(self, r_true: float) -> None:
        """Empirical covariance vs reported R at one range, 10% tolerance."""
        pos, r_reported, r_analytic = self._run_samples(r_true)
        cov = np.cov(pos.T, bias=False)

        # -- the reported R must equal the analytic one --------------------
        self.assertTrue(
            np.allclose(r_reported, r_analytic, rtol=0.08, atol=1e-9),
            msg=f"r={r_true}: reported R {r_reported} vs analytic {r_analytic}",
        )

        # -- Frobenius agreement of empirical vs reported ------------------
        rel = (np.linalg.norm(cov - r_reported, ord="fro")
               / np.linalg.norm(r_reported, ord="fro"))
        self.assertLess(
            rel, 0.10,
            msg=f"r={r_true}: ||cov - R||_F / ||R||_F = {rel:.4f}\ncov={cov}\nR={r_reported}",
        )

        # -- the physically meaningful decomposition -----------------------
        # Along the line of sight the spread must be sigma_r; across it, it
        # must be r*sigma_theta. This is what makes the error ellipse
        # range-elongated and is the property the Jacobian encodes.
        alpha = self.PSI_OWN + self.THETA_REL
        u = np.array([math.cos(alpha), math.sin(alpha)])       # along range
        w = np.array([-math.sin(alpha), math.cos(alpha)])      # across range
        sigma_r = 3.0 + 0.01 * r_true
        sigma_t = r_true * 0.0087

        var_along = float(u @ cov @ u)
        var_across = float(w @ cov @ w)
        self.assertLess(abs(math.sqrt(var_along) / sigma_r - 1.0), 0.10,
                        msg=f"r={r_true}: along-range sigma {math.sqrt(var_along):.3f} "
                            f"vs {sigma_r:.3f}")
        self.assertLess(abs(math.sqrt(var_across) / sigma_t - 1.0), 0.10,
                        msg=f"r={r_true}: across-range sigma {math.sqrt(var_across):.3f} "
                            f"vs {sigma_t:.3f}")

        # -- the reported R is a usable covariance -------------------------
        self.assertAlmostEqual(r_reported[0, 1], r_reported[1, 0], places=9)
        eig = np.linalg.eigvalsh(r_reported)
        self.assertGreater(float(eig.min()), 0.0)

    def test_covariance_at_200m(self) -> None:
        """20 000 samples at 200 m match the reported covariance."""
        self._check(200.0)

    def test_covariance_at_2000m(self) -> None:
        """20 000 samples at 2000 m match the reported covariance."""
        self._check(2000.0)

    def test_ellipse_grows_and_rotates_with_geometry(self) -> None:
        """The error ellipse is aligned with the line of sight and grows with r.

        Along the LOS the standard deviation is exactly ``sigma_r``; across it
        the standard deviation is exactly ``r * sigma_theta`` and therefore
        scales linearly with range. Both are checked against the analytic
        values, and the eigenvector of the larger axis is checked to lie along
        the correct geometric direction.
        """
        th, psi, st = 0.3, 0.0, 0.0087
        u = np.array([math.cos(th + psi), math.sin(th + psi)])   # along range
        w = np.array([-math.sin(th + psi), math.cos(th + psi)])  # across range
        acrosses = []
        for r in (200.0, 2000.0, 6000.0):
            sr = 3.0 + 0.01 * r
            cov = SensorModel.cartesian_covariance(r, th, psi, sr, st)
            self.assertAlmostEqual(math.sqrt(float(u @ cov @ u)), sr, places=9)
            self.assertAlmostEqual(math.sqrt(float(w @ cov @ w)), r * st, places=9)
            acrosses.append(math.sqrt(float(w @ cov @ w)))
        # linear growth of the across-range axis with range
        self.assertAlmostEqual(acrosses[1] / acrosses[0], 10.0, places=6)
        self.assertAlmostEqual(acrosses[2] / acrosses[0], 30.0, places=6)
        # With a bearing-noise-dominated sensor the across-range axis becomes
        # the major one: sigma_theta = 0.02 rad beats sigma_r_frac = 0.01.
        cov_bearing = SensorModel.cartesian_covariance(
            2000.0, th, psi, 3.0 + 0.01 * 2000.0, 0.02
        )
        self.assertGreater(
            float(w @ cov_bearing @ w), float(u @ cov_bearing @ u)
        )
        # a rotated ownship rotates the ellipse rigidly (same eigenvalues)
        cov_a = SensorModel.cartesian_covariance(1500.0, 0.3, 0.0, 18.0, st)
        cov_b = SensorModel.cartesian_covariance(1500.0, 0.3, 1.1, 18.0, st)
        self.assertTrue(
            np.allclose(np.linalg.eigvalsh(cov_a), np.linalg.eigvalsh(cov_b))
        )
        self.assertFalse(np.allclose(cov_a, cov_b))

    def test_covariance_matches_manual_jacobian(self) -> None:
        """cartesian_covariance equals J D J^T rotated, computed by hand."""
        r, th, psi, sr, st = 1234.0, 0.42, -1.1, 15.34, 0.0087
        jac = np.array([[math.cos(th), -r * math.sin(th)],
                        [math.sin(th), r * math.cos(th)]])
        d = np.diag([sr * sr, st * st])
        rot = np.array([[math.cos(psi), -math.sin(psi)],
                        [math.sin(psi), math.cos(psi)]])
        expected = rot @ (jac @ d @ jac.T) @ rot.T
        got = SensorModel.cartesian_covariance(r, th, psi, sr, st)
        self.assertTrue(np.allclose(got, expected, rtol=1e-12, atol=1e-12))


# --------------------------------------------------------------------------
# 3. track identity: dropout -> new id
# --------------------------------------------------------------------------


class TestTrackManager(unittest.TestCase):
    """Track opening, ageing, dropping and deliberate re-acquisition."""

    def _cfg(self, drop: int = 3) -> SensorConfig:
        return SensorConfig(
            pd_max=1.0,
            r_full_m=800.0,
            r_max_m=3000.0,
            update_rate_hz=5.0,
            async_phase=False,
            latency_s=0.0,
            latency_jitter_s=0.0,
            drop_after_misses=drop,
            rcs_factor=_flat_rcs(1.0),
        )

    def test_forced_dropout_yields_new_track_id(self) -> None:
        """A dropout longer than drop_after_misses resets id and age.

        The target is teleported past ``r_max_m`` so Pd is exactly zero and
        the misses are deterministic, then brought back into range.
        """
        drop = 3
        cfg = self._cfg(drop)
        sm = SensorModel(cfg, phase_seed=11)
        tm = TrackManager.from_config(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(31337))

        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgt = _target(42, 200.0, 0.0, PlatformType.MULTIROTOR)

        ages: list[int] = []
        ids: list[int] = []
        n_visible_1, n_gap, n_visible_2 = 6, 4, 3
        first_id = -1
        second_id = -1
        closed_at: int | None = None

        for k in range(n_visible_1 + n_gap + n_visible_2):
            t = 0.2 * k
            if n_visible_1 <= k < n_visible_1 + n_gap:
                tgt.state.x = 5000.0          # far outside r_max -> Pd = 0
            else:
                tgt.state.x = 200.0
            recs = tm.update(t, sm.measure(t, own, [tgt], rng_n, rng_d, rng_l))
            rec = recs[42]
            ids.append(rec.track_id)
            ages.append(rec.track_age)
            if k == n_visible_1 - 1:
                first_id = rec.track_id
            if rec.track_id < 0 and closed_at is None:
                closed_at = k
            if k == n_visible_1 + n_gap:
                second_id = rec.track_id

        # the first track opened at id 0 and aged one step per detection
        self.assertEqual(ids[0], 0)
        self.assertEqual(ages[0], 0)
        self.assertEqual(first_id, 0)
        self.assertEqual(ages[n_visible_1 - 1], n_visible_1 - 1)

        # it closed exactly on the drop-th consecutive miss
        self.assertEqual(closed_at, n_visible_1 + drop - 1)
        for k in range(closed_at, n_visible_1 + n_gap):
            self.assertEqual(ids[k], -1)
            self.assertEqual(ages[k], 0)

        # re-acquisition is a NEW identity with a reset age
        self.assertGreater(second_id, first_id)
        self.assertEqual(second_id, 1)
        self.assertEqual(ages[n_visible_1 + n_gap], 0)
        self.assertEqual(ages[-1], n_visible_2 - 1)

    def test_track_survives_a_short_gap(self) -> None:
        """Fewer than drop_after_misses misses keep the same track id."""
        drop = 4
        cfg = self._cfg(drop)
        sm = SensorModel(cfg, phase_seed=11)
        tm = TrackManager.from_config(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(4242))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgt = _target(7, 300.0, 100.0, PlatformType.EVTOL_UAM)

        ids: list[int] = []
        for k in range(10):
            t = 0.2 * k
            tgt.state.x = 5000.0 if 3 <= k < 6 else 300.0   # 3 misses < drop=4
            recs = tm.update(t, sm.measure(t, own, [tgt], rng_n, rng_d, rng_l))
            ids.append(recs[7].track_id)
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(ids[0], 0)

    def test_track_ids_are_unique_and_monotonic(self) -> None:
        """Ids are never reused, even across many open/close cycles."""
        cfg = self._cfg(1)
        sm = SensorModel(cfg, phase_seed=1)
        tm = TrackManager.from_config(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(9))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgt = _target(3, 250.0, 0.0, PlatformType.MULTIROTOR)
        seen: list[int] = []
        for k in range(20):
            t = 0.2 * k
            tgt.state.x = 250.0 if k % 2 == 0 else 5000.0
            recs = tm.update(t, sm.measure(t, own, [tgt], rng_n, rng_d, rng_l))
            tid = recs[3].track_id
            if tid >= 0 and (not seen or seen[-1] != tid):
                seen.append(tid)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(len(seen), 10)

    def test_snapshot_restore_round_trip(self) -> None:
        """Tracker state survives a snapshot/restore cycle unchanged."""
        cfg = self._cfg(3)
        sm = SensorModel(cfg, phase_seed=2)
        tm = TrackManager.from_config(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(77))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgts = [_target(i, 200.0 + 10.0 * i, 30.0 * i, PlatformType.MULTIROTOR)
                for i in range(4)]
        for k in range(6):
            t = 0.2 * k
            tm.update(t, sm.measure(t, own, tgts, rng_n, rng_d, rng_l))

        snap = tm.snapshot()
        before = tm.records()
        tm2 = TrackManager(drop_after_misses=99, first_track_id=999)
        tm2.restore(snap)
        after = tm2.records()
        self.assertEqual(tm2.drop_after_misses, cfg.drop_after_misses)
        self.assertEqual(tm2.next_track_id, tm.next_track_id)
        self.assertEqual(sorted(before), sorted(after))
        for aid, rec in before.items():
            self.assertEqual(rec.track_id, after[aid].track_id)
            self.assertEqual(rec.track_age, after[aid].track_age)
            self.assertTrue(
                np.allclose(rec.pos_xy, after[aid].pos_xy, equal_nan=True)
            )

    def test_returned_records_are_copies(self) -> None:
        """Mutating a returned record cannot corrupt the tracker."""
        cfg = self._cfg(3)
        sm = SensorModel(cfg, phase_seed=2)
        tm = TrackManager.from_config(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(5))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgts = [_target(0, 200.0, 0.0, PlatformType.MULTIROTOR)]
        recs = tm.update(0.0, sm.measure(0.0, own, tgts, rng_n, rng_d, rng_l))
        recs[0].track_id = 12345
        recs[0].pos_xy[0] = 1e9
        again = tm.update(0.2, sm.measure(0.2, own, tgts, rng_n, rng_d, rng_l))
        self.assertEqual(again[0].track_id, 0)
        self.assertLess(abs(float(again[0].pos_xy[0])), 1e4)


# --------------------------------------------------------------------------
# 4. no interpolation, ever
# --------------------------------------------------------------------------


class TestNoInterpolation(unittest.TestCase):
    """Undetected slots must be NaN -- the gap is the signal (doc 3.5)."""

    def test_undetected_slots_are_nan(self) -> None:
        """Over a long run with Pd ~ 0.5, every miss is NaN in both layers."""
        cfg = SensorConfig(
            pd_max=0.5,
            r_full_m=800.0,
            r_max_m=3000.0,
            update_rate_hz=5.0,
            async_phase=True,
            latency_s=0.2,
            latency_jitter_s=0.05,
            drop_after_misses=4,
            rcs_factor=_flat_rcs(1.0),
        )
        sm = SensorModel(cfg, phase_seed=17)
        tm = TrackManager.from_config(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(555))

        own = AgentState(0.0, 0.0, 55.0, 0.2)
        tgts = [
            SensorTarget(i, AgentState(300.0 + 40.0 * i, -200.0 + 55.0 * i,
                                       25.0, 0.5 * i), PlatformType.MULTIROTOR)
            for i in range(6)
        ]

        n_miss = 0
        n_hit = 0
        n_no_track = 0
        for k in range(600):                       # 30 s at SIM_DT = 0.05 s
            t = 0.05 * k
            for tgt in tgts:                       # let the truth move
                tgt.state.x += 25.0 * math.cos(tgt.state.psi) * 0.05
                tgt.state.y += 25.0 * math.sin(tgt.state.psi) * 0.05
            recs = tm.update(t, sm.measure(t, own, tgts, rng_n, rng_d, rng_l))
            for rec in recs.values():
                if rec.detected:
                    n_hit += 1
                    self.assertTrue(np.all(np.isfinite(rec.pos_xy)))
                    self.assertTrue(np.all(np.isfinite(rec.R_2x2)))
                    self.assertGreaterEqual(rec.track_id, 0)
                    self.assertTrue(math.isfinite(rec.t_valid))
                else:
                    n_miss += 1
                    self.assertTrue(np.all(np.isnan(rec.pos_xy)),
                                    "undetected slot carries a position")
                    self.assertTrue(np.all(np.isnan(rec.R_2x2)),
                                    "undetected slot carries a covariance")
                    self.assertTrue(math.isnan(rec.t_valid))
                    if rec.track_id < 0:
                        n_no_track += 1
                        self.assertEqual(rec.track_age, 0)
        self.assertGreater(n_hit, 100)
        self.assertGreater(n_miss, 100)
        self.assertGreater(n_no_track, 0, "no track was ever dropped")

    def test_measurement_miss_payload_is_nan(self) -> None:
        """A miss Measurement itself carries NaN, not a stale position."""
        cfg = SensorConfig(pd_max=1.0, r_full_m=100.0, r_max_m=200.0,
                           update_rate_hz=5.0, async_phase=False,
                           latency_s=0.0, latency_jitter_s=0.0,
                           rcs_factor=_flat_rcs(1.0))
        sm = SensorModel(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(1))
        own = AgentState(0.0, 0.0, 40.0, 0.0)
        tgt = _target(0, 50.0, 0.0, PlatformType.MULTIROTOR)
        m0 = sm.measure(0.0, own, [tgt], rng_n, rng_d, rng_l)[0]
        self.assertTrue(m0.detected)
        tgt.state.x = 500.0
        m1 = sm.measure(0.2, own, [tgt], rng_n, rng_d, rng_l)[0]
        self.assertFalse(m1.detected)
        self.assertTrue(np.all(np.isnan(m1.pos_xy)))
        self.assertTrue(np.all(np.isnan(m1.R_2x2)))

    def test_fills_a_schema_compliant_episode(self) -> None:
        """The observation layer written straight through passes schema.validate.

        This is the contract check that matters for the writer: the tracker's
        NaN / -1 policy is exactly the one :mod:`storage.schema` enforces, so
        the writer may copy the records verbatim with no gap filling.
        """
        cfg = SensorConfig(
            pd_max=0.7, r_full_m=800.0, r_max_m=3000.0, update_rate_hz=5.0,
            async_phase=True, latency_s=0.2, latency_jitter_s=0.05,
            drop_after_misses=3, rcs_factor=_flat_rcs(1.0),
        )
        sm = SensorModel(cfg, phase_seed=4)
        tm = TrackManager.from_config(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(808))

        n_steps, n_agents = 40, 3
        ep = schema.EpisodeArrays.allocate(n_steps, n_agents)
        own = AgentState(0.0, 0.0, 55.0, 0.1)
        tgts = [
            SensorTarget(i, AgentState(400.0 + 100.0 * i, 100.0 * i, 20.0,
                                       0.3 * i - 0.4), PlatformType.MULTIROTOR)
            for i in range(n_agents)
        ]
        ep.agent_ids[:] = [t.agent_id for t in tgts]
        ep.gt_type[:] = [int(t.platform_type) for t in tgts]

        for k in range(n_steps):
            t = 0.5 * k                      # log at LOG_DT
            for tgt in tgts:
                tgt.state.x += 20.0 * math.cos(tgt.state.psi) * 0.5
                tgt.state.y += 20.0 * math.sin(tgt.state.psi) * 0.5
            recs = tm.update(t, sm.measure(t, own, tgts, rng_n, rng_d, rng_l))

            ep.t[k] = t
            ep.own_pos[k] = (own.x, own.y)
            ep.own_vel[k] = own.ground_velocity((0.0, 0.0), 0.0)
            ep.own_psi[k] = wrap_angle(own.psi)
            for slot, tgt in enumerate(tgts):
                ep.gt_valid[k, slot] = True
                ep.gt_pos[k, slot] = (tgt.state.x, tgt.state.y)
                ep.gt_vel[k, slot] = tgt.state.ground_velocity((0.0, 0.0), 0.0)
                ep.gt_psi[k, slot] = wrap_angle(tgt.state.psi)
                ep.gt_mode[k, slot] = int(ManeuverMode.CV)
                ep.gt_intent[k, slot] = int(IntentType.CORRIDOR_TRANSIT)
                rec = recs.get(tgt.agent_id)
                if rec is None:
                    continue
                ep.obs_track_id[k, slot] = rec.track_id
                ep.obs_track_age[k, slot] = rec.track_age
                if rec.detected:
                    ep.obs_detected[k, slot] = True
                    ep.obs_pos[k, slot] = rec.pos_xy
                    ep.obs_R[k, slot] = rec.R_2x2
                    ep.obs_t_valid[k, slot] = np.float32(rec.t_valid)

        problems = schema.validate(ep)
        self.assertEqual(problems, [], msg="\n".join(problems))
        self.assertTrue(bool(np.any(ep.obs_detected)))
        self.assertTrue(bool(np.any(~ep.obs_detected)))


# --------------------------------------------------------------------------
# update rate, async phase, latency
# --------------------------------------------------------------------------


class TestRateAndLatency(unittest.TestCase):
    """Update scheduling, phase de-synchronisation and delayed reporting."""

    def test_update_rate_is_respected(self) -> None:
        """A 5 Hz sensor stepped at 20 Hz fires once every four calls."""
        cfg = SensorConfig(update_rate_hz=5.0, async_phase=False, pd_max=1.0,
                           r_full_m=800.0, r_max_m=3000.0, latency_s=0.0,
                           latency_jitter_s=0.0, rcs_factor=_flat_rcs(1.0))
        sm = SensorModel(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(3))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgts = [_target(0, 300.0, 0.0, PlatformType.MULTIROTOR)]
        n = 0
        n_steps = 400                                   # 20 s at 0.05 s
        for k in range(n_steps):
            n += len(sm.measure(0.05 * k, own, tgts, rng_n, rng_d, rng_l))
        self.assertEqual(n, 100)                        # 20 s * 5 Hz

    def test_async_phase_desynchronises_tracks(self) -> None:
        """With async phasing, targets do not all update on the same tick."""
        cfg = SensorConfig(update_rate_hz=5.0, async_phase=True, pd_max=1.0,
                           r_full_m=800.0, r_max_m=3000.0, latency_s=0.0,
                           latency_jitter_s=0.0, rcs_factor=_flat_rcs(1.0))
        sm = SensorModel(cfg, phase_seed=99)
        rng_n, rng_d, rng_l = _streams(_hub(3))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgts = [_target(i, 300.0, 20.0 * i, PlatformType.MULTIROTOR)
                for i in range(8)]
        per_tick: list[int] = []
        for k in range(200):
            per_tick.append(
                len(sm.measure(0.05 * k, own, tgts, rng_n, rng_d, rng_l))
            )
        self.assertLess(max(per_tick), len(tgts),
                        "all tracks updated on the same tick")
        self.assertGreater(sum(per_tick), 0)
        # every target still gets its full share of updates
        self.assertAlmostEqual(sum(per_tick) / len(tgts), 50.0, delta=2.0)

    def test_latency_reports_a_stale_position(self) -> None:
        """A report at time t describes the target as it was at t - latency."""
        cfg = SensorConfig(
            sigma_r0_m=0.0, sigma_r_frac=0.0, sigma_theta_rad=0.0,
            pd_max=1.0, r_full_m=5000.0, r_max_m=6000.0,
            update_rate_hz=5.0, async_phase=False,
            latency_s=0.2, latency_jitter_s=0.0, rcs_factor=_flat_rcs(1.0),
        )
        sm = SensorModel(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(6))
        own = AgentState(0.0, 0.0, 0.0, 0.0)
        speed = 20.0
        tgt = _target(0, 500.0, 0.0, PlatformType.FIXED_WING_UAS,
                      v=speed, psi=0.0)
        dt = 0.05
        for k in range(200):
            t = dt * k
            tgt.state.x = 500.0 + speed * t
            for m in sm.measure(t, own, [tgt], rng_n, rng_d, rng_l):
                if t < 0.5:
                    continue                    # buffer edge: t_valid clamps
                self.assertTrue(m.detected)
                self.assertAlmostEqual(m.t_valid, t - cfg.latency_s, places=9)
                expected_x = 500.0 + speed * (t - cfg.latency_s)
                self.assertAlmostEqual(float(m.pos_xy[0]), expected_x, places=6)
                self.assertAlmostEqual(float(m.pos_xy[1]), 0.0, places=6)

    def test_sensor_model_snapshot_restore(self) -> None:
        """Snapshot/restore reproduces the same measurement stream."""
        cfg = SensorConfig(pd_max=0.8, r_full_m=800.0, r_max_m=3000.0,
                           update_rate_hz=5.0, async_phase=True,
                           latency_s=0.2, latency_jitter_s=0.05,
                           rcs_factor=_flat_rcs(1.0))
        own = AgentState(0.0, 0.0, 50.0, 0.0)
        tgts = [_target(i, 300.0 + 20.0 * i, 10.0 * i, PlatformType.MULTIROTOR)
                for i in range(3)]

        sm = SensorModel(cfg, phase_seed=21)
        hub = _hub(1234)
        rng_n, rng_d, rng_l = _streams(hub)
        for k in range(20):
            sm.measure(0.05 * k, own, tgts, rng_n, rng_d, rng_l)
        snap_sensor = sm.snapshot()
        snap_rng = hub.snapshot()
        tail_a = [
            [m.to_dict() for m in sm.measure(0.05 * k, own, tgts, rng_n, rng_d, rng_l)]
            for k in range(20, 40)
        ]

        sm2 = SensorModel(cfg, phase_seed=0)
        sm2.restore(snap_sensor)
        hub2 = _hub(0)
        hub2.restore(snap_rng)
        rng_n2, rng_d2, rng_l2 = _streams(hub2)
        tail_b = [
            [m.to_dict() for m in
             sm2.measure(0.05 * k, own, tgts, rng_n2, rng_d2, rng_l2)]
            for k in range(20, 40)
        ]
        self.assertEqual(tail_a, tail_b)


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------


class TestTargetAdapter(unittest.TestCase):
    """as_target accepts the duck-typed agent objects of agents.base."""

    def test_duck_typed_agent(self) -> None:
        """An object with agent_id / state / platform_type is accepted."""

        class FakeAgent:
            def __init__(self) -> None:
                self.agent_id = 5
                self.state = AgentState(1.0, 2.0, 3.0, 0.4)
                self.platform_type = PlatformType.EVTOL_UAM

        tgt = as_target(FakeAgent())
        self.assertEqual(tgt.agent_id, 5)
        self.assertEqual(tgt.platform_type, PlatformType.EVTOL_UAM)
        self.assertAlmostEqual(tgt.state.x, 1.0)

    def test_tuple_forms(self) -> None:
        """(id, state) and (id, state, type) tuples are accepted."""
        st = AgentState(0.0, 0.0, 10.0, 0.0)
        self.assertEqual(as_target((3, st)).platform_type, PlatformType.UNKNOWN)
        self.assertEqual(
            as_target((3, st, int(PlatformType.BIRD))).platform_type,
            PlatformType.BIRD,
        )

    def test_rejects_unusable_object(self) -> None:
        """An object with neither id nor state raises TypeError."""
        with self.assertRaises(TypeError):
            as_target(object())

    def test_disabled_sensor_is_a_passthrough(self) -> None:
        """enabled=False copies ground truth exactly (debug path only)."""
        cfg = SensorConfig(enabled=False, update_rate_hz=5.0,
                           async_phase=False, rcs_factor=_flat_rcs(1.0))
        sm = SensorModel(cfg)
        rng_n, rng_d, rng_l = _streams(_hub(2))
        own = AgentState(10.0, -5.0, 50.0, 0.9)
        tgt = _target(1, 1234.0, -567.0, PlatformType.BIRD)
        m = sm.measure(0.0, own, [tgt], rng_n, rng_d, rng_l)[0]
        self.assertTrue(m.detected)
        self.assertAlmostEqual(float(m.pos_xy[0]), 1234.0, places=9)
        self.assertAlmostEqual(float(m.pos_xy[1]), -567.0, places=9)
        self.assertIsInstance(m, Measurement)


if __name__ == "__main__":
    unittest.main(verbosity=2)
