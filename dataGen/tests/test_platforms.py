"""Self-consistency tests for the performance-envelope catalogue.

Run directly:
    python $PRISM_ROOT/dataGen/tests/test_platforms.py

The point of these tests is not coverage for its own sake. An envelope with a
degenerate field (v_min >= v_max, a zero turn cap, a non-negative a_min) does
not crash: it silently produces trajectories that are trivially predictable,
which is the one failure mode doc 3.1 says destroys the paper. So every field
is checked over many random draws rather than once.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

# The package root ($PRISM_ROOT/dataGen) is normally put on sys.path by main.py.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np  # noqa: E402

from agents.platforms import (  # noqa: E402
    OOD_PLATFORMS,
    PARAMETER_PROVENANCE,
    PLATFORM_DEFAULTS,
    TRAINING_PLATFORMS,
    UNKNOWN_BANK_CAP_RAD,
    UNKNOWN_BASE_CHOICES,
    UNKNOWN_DWELL_SCALE,
    UNKNOWN_OMEGA_DOT_FACTOR,
    UNKNOWN_OMEGA_MAX_FACTOR,
    UNKNOWN_V_MAX_FACTOR,
    _self_check,
    default_envelope,
    envelope_summary_table,
    is_ood_platform,
    make_envelope,
    make_envelope_with_meta,
    mode_dwell_scale,
)
from core.constants import EPS_V, G, RAD  # noqa: E402
from core.state import PerformanceEnvelope  # noqa: E402
from core.types import PlatformType  # noqa: E402

#: Number of independent draws per platform.
N_DRAWS = 200


def _rng(seed: int) -> np.random.Generator:
    """Fresh generator [-] for a test case."""
    return np.random.default_rng(seed)


class TestCatalogue(unittest.TestCase):
    """The static table itself must be complete and well formed."""

    def test_self_check_clean(self) -> None:
        self.assertEqual(_self_check(), [])

    def test_every_platform_present(self) -> None:
        for ptype in PlatformType:
            self.assertIn(int(ptype), PLATFORM_DEFAULTS, ptype.name)

    def test_split_partitions_all_platforms(self) -> None:
        covered = set(TRAINING_PLATFORMS) | set(OOD_PLATFORMS)
        self.assertEqual(covered, set(PlatformType))
        self.assertEqual(set(TRAINING_PLATFORMS) & set(OOD_PLATFORMS), set())

    def test_is_ood_platform_matches_split(self) -> None:
        for ptype in PlatformType:
            self.assertEqual(is_ood_platform(ptype), ptype in OOD_PLATFORMS, ptype.name)

    def test_provenance_status_values(self) -> None:
        self.assertTrue(PARAMETER_PROVENANCE)
        for key, value in PARAMETER_PROVENANCE.items():
            self.assertEqual(len(value), 2, key)
            self.assertIn(value[0], ("traced", "assumed"), key)
            self.assertTrue(value[1].strip(), key)

    def test_summary_table_is_json_safe(self) -> None:
        rows = envelope_summary_table()
        self.assertEqual(len(rows), len(list(PlatformType)))
        allowed = (bool, int, float, str, list)
        for row in rows:
            for key, value in row.items():
                self.assertIsInstance(key, str)
                self.assertIsInstance(value, allowed, f"{key} -> {type(value)}")
            self.assertGreater(row["v_max_ms"], row["v_min_ms"], row["label"])
            self.assertGreater(row["omega_max_deg_s"], 0.0, row["label"])


class TestEnvelopeConsistency(unittest.TestCase):
    """Every sampled envelope must be physically self-consistent."""

    def _assert_consistent(self, env: PerformanceEnvelope, label: str) -> None:
        self.assertLess(env.v_min, env.v_max, f"{label}: v_min < v_max")
        self.assertGreaterEqual(env.v_min, 0.0, f"{label}: v_min >= 0")
        self.assertLess(env.a_min, 0.0, f"{label}: a_min < 0")
        self.assertGreater(env.a_max, 0.0, f"{label}: a_max > 0")
        self.assertGreater(env.omega_max, 0.0, f"{label}: omega_max > 0")
        self.assertGreater(env.omega_dot_max, 0.0, f"{label}: omega_dot_max > 0")
        self.assertGreater(env.lateral_accel_max, 0.0, f"{label}: lateral > 0")
        self.assertGreaterEqual(env.bank_max, 0.0, f"{label}: bank_max >= 0")
        self.assertLess(env.bank_max, 0.5 * math.pi, f"{label}: bank_max < 90 deg")
        self.assertGreaterEqual(env.wind_susceptibility, 0.0, f"{label}: wind >= 0")
        self.assertLessEqual(env.wind_susceptibility, 1.0, f"{label}: wind <= 1")
        self.assertGreater(env.mass, 0.0, f"{label}: mass > 0")
        self.assertGreater(env.drag_area, 0.0, f"{label}: drag_area > 0")
        if env.bank_limited:
            self.assertGreater(env.bank_max, 0.0, f"{label}: banked needs bank_max")
        for value in env.to_dict().values():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), f"{label}: non-finite field")

    def test_all_platforms_all_draws(self) -> None:
        for ptype in PlatformType:
            rng = _rng(1000 + int(ptype))
            for i in range(N_DRAWS):
                env = make_envelope(ptype, rng)
                self._assert_consistent(env, f"{ptype.name}#{i}")

    def test_default_envelope_consistent(self) -> None:
        for ptype in PlatformType:
            self._assert_consistent(default_envelope(ptype), f"default {ptype.name}")

    def test_turn_rate_cap_positive_and_finite(self) -> None:
        for ptype in PlatformType:
            rng = _rng(2000 + int(ptype))
            for _ in range(50):
                env = make_envelope(ptype, rng)
                for v in (0.0, EPS_V, 1.0, env.v_min, 0.5 * (env.v_min + env.v_max),
                          env.v_max, 2.0 * env.v_max):
                    cap = env.max_turn_rate(v)
                    self.assertGreater(cap, 0.0, ptype.name)
                    self.assertTrue(math.isfinite(cap), ptype.name)
                    self.assertLessEqual(cap, env.omega_max + 1e-12, ptype.name)

    def test_speed_dependent_cap_is_non_increasing(self) -> None:
        """Bank/tilt-limited platforms must turn worse as they speed up."""
        for ptype in (PlatformType.FIXED_WING_UAS, PlatformType.EVTOL_UAM,
                      PlatformType.GA, PlatformType.MULTIROTOR):
            rng = _rng(3000 + int(ptype))
            env = make_envelope(ptype, rng)
            speeds = np.linspace(max(env.v_min, EPS_V), env.v_max, 25)
            caps = [env.max_turn_rate(float(v)) for v in speeds]
            for lo, hi in zip(caps, caps[1:]):
                self.assertGreaterEqual(lo + 1e-12, hi, ptype.name)
            self.assertLess(caps[-1], caps[0], f"{ptype.name}: cap must actually fall")

    def test_bank_limited_formula(self) -> None:
        for ptype in (PlatformType.FIXED_WING_UAS, PlatformType.EVTOL_UAM,
                      PlatformType.GA):
            env = make_envelope(ptype, _rng(4000 + int(ptype)))
            self.assertTrue(env.bank_limited, ptype.name)
            v = 0.5 * (env.v_min + env.v_max)
            expected = min(G * math.tan(env.bank_max) / max(v, EPS_V), env.omega_max)
            self.assertAlmostEqual(env.max_turn_rate(v), expected, places=12)

    def test_flags_match_platform_semantics(self) -> None:
        expect_hover = {
            PlatformType.MULTIROTOR: True,
            PlatformType.EVTOL_UAM: True,
            PlatformType.BIRD: False,
            PlatformType.FIXED_WING_UAS: False,
            PlatformType.GA: False,
            PlatformType.DEBRIS: False,
        }
        for ptype, hover in expect_hover.items():
            env = make_envelope(ptype, _rng(5000 + int(ptype)))
            self.assertEqual(env.can_hover, hover, ptype.name)
            self.assertEqual(
                env.is_powered, ptype is not PlatformType.DEBRIS, ptype.name
            )

    def test_debris_is_unpowered_and_fully_advected(self) -> None:
        rng = _rng(6006)
        for _ in range(50):
            env = make_envelope(PlatformType.DEBRIS, rng)
            self.assertFalse(env.is_powered)
            self.assertFalse(env.can_hover)
            self.assertAlmostEqual(env.wind_susceptibility, 1.0, places=12)
            self.assertGreaterEqual(env.mass, 0.2 - 1e-9)
            self.assertLessEqual(env.mass, 2.0 + 1e-9)
            self.assertGreaterEqual(env.drag_area, 0.01 - 1e-9)
            self.assertLessEqual(env.drag_area, 0.08 + 1e-9)

    def test_speed_bands_respect_the_doc_table(self) -> None:
        """Sampled speeds must stay inside the documented family range."""
        for ptype in PlatformType:
            if ptype is PlatformType.UNKNOWN:
                continue  # UNKNOWN is defined by leaving its band, see below
            lo, hi = PLATFORM_DEFAULTS[int(ptype)]["doc_speed_range_ms"]
            rng = _rng(7000 + int(ptype))
            for _ in range(100):
                env = make_envelope(ptype, rng)
                self.assertGreaterEqual(env.v_min, lo - 1e-9, ptype.name)
                self.assertLessEqual(env.v_max, hi + 1e-9, ptype.name)

    def test_agents_of_a_class_are_not_clones(self) -> None:
        for ptype in PlatformType:
            rng = _rng(8000 + int(ptype))
            v_maxes = {round(make_envelope(ptype, rng).v_max, 9) for _ in range(30)}
            self.assertGreater(len(v_maxes), 1, ptype.name)


class TestUnknownIsTrulyOutOfDistribution(unittest.TestCase):
    """UNKNOWN must exceed the envelope of the base it was built from."""

    def test_meta_reports_base_and_factors(self) -> None:
        rng = _rng(9001)
        seen_bases = set()
        for _ in range(N_DRAWS):
            env, meta = make_envelope_with_meta(PlatformType.UNKNOWN, rng)
            self.assertTrue(meta["is_ood"])
            self.assertEqual(meta["platform"], int(PlatformType.UNKNOWN))
            self.assertIsNotNone(meta["base_platform"])
            base_platform = PlatformType(int(meta["base_platform"]))
            self.assertIn(base_platform, UNKNOWN_BASE_CHOICES)
            seen_bases.add(base_platform)
            self.assertAlmostEqual(meta["dwell_scale"], UNKNOWN_DWELL_SCALE, places=12)
            base = PerformanceEnvelope.from_dict(meta["base_envelope"])

            f = meta["factors"]
            self.assertGreaterEqual(f["v_max"], UNKNOWN_V_MAX_FACTOR[0] - 1e-9)
            self.assertLessEqual(f["v_max"], UNKNOWN_V_MAX_FACTOR[1] + 1e-9)
            self.assertGreaterEqual(f["omega_max"], UNKNOWN_OMEGA_MAX_FACTOR[0] - 1e-9)
            self.assertLessEqual(f["omega_max"], UNKNOWN_OMEGA_MAX_FACTOR[1] + 1e-9)
            self.assertGreaterEqual(f["omega_dot"], UNKNOWN_OMEGA_DOT_FACTOR[0] - 1e-9)
            self.assertLessEqual(f["omega_dot"], UNKNOWN_OMEGA_DOT_FACTOR[1] + 1e-9)

            # The three headline caps must genuinely exceed the base.
            self.assertGreater(env.v_max, base.v_max)
            self.assertGreater(env.omega_max, base.omega_max)
            self.assertGreater(env.omega_dot_max, base.omega_dot_max)
            self.assertLessEqual(env.bank_max, UNKNOWN_BANK_CAP_RAD + 1e-12)

            # And the EFFECTIVE, speed-dependent cap must exceed it too --
            # scaling omega_max alone would do nothing to a bank-limited base.
            for v in np.linspace(0.0, base.v_max * 1.6, 20):
                self.assertGreater(
                    env.max_turn_rate(float(v)),
                    base.max_turn_rate(float(v)),
                    f"base={base_platform.name} v={v:.1f}",
                )
        self.assertGreater(len(seen_bases), 1, "base platform should vary")

    def test_dwell_scale_is_shorter_only_for_unknown(self) -> None:
        for ptype in PlatformType:
            scale = mode_dwell_scale(ptype)
            if ptype is PlatformType.UNKNOWN:
                self.assertLess(scale, 1.0)
                self.assertGreater(scale, 0.0)
            else:
                self.assertEqual(scale, 1.0, ptype.name)

    def test_unknown_envelope_is_still_self_consistent(self) -> None:
        rng = _rng(9100)
        for _ in range(N_DRAWS):
            env = make_envelope(PlatformType.UNKNOWN, rng)
            self.assertLess(env.v_min, env.v_max)
            self.assertLess(env.a_min, 0.0)
            self.assertGreater(env.a_max, 0.0)
            self.assertGreater(env.omega_max, 0.0)
            self.assertGreater(env.omega_dot_max, 0.0)
            self.assertLessEqual(env.wind_susceptibility, 1.0)
            self.assertGreaterEqual(env.wind_susceptibility, 0.0)


class TestReproducibility(unittest.TestCase):
    """Seeded generation is a paper requirement (doc 10)."""

    def test_same_seed_same_envelope(self) -> None:
        for ptype in PlatformType:
            a = make_envelope(ptype, _rng(4242)).to_dict()
            b = make_envelope(ptype, _rng(4242)).to_dict()
            self.assertEqual(a, b, ptype.name)

    def test_draw_count_depends_only_on_platform(self) -> None:
        """Two agents of the same class must consume the same number of draws."""
        for ptype in PlatformType:
            rng_a = _rng(555)
            make_envelope(ptype, rng_a)
            make_envelope(ptype, rng_a)
            after_two = rng_a.random()

            rng_b = _rng(555)
            make_envelope(ptype, rng_b)
            rng_b_state_mid = rng_b.bit_generator.state
            make_envelope(ptype, rng_b)
            self.assertNotEqual(rng_b_state_mid, rng_b.bit_generator.state, ptype.name)
            self.assertEqual(after_two, rng_b.random(), ptype.name)


class TestPhysicalPlausibility(unittest.TestCase):
    """Sanity checks that would catch a unit slip (deg vs rad, etc.)."""

    def test_turn_rates_are_in_a_sane_degree_range(self) -> None:
        limits_deg = {
            PlatformType.BIRD: (30.0, 90.0),
            PlatformType.MULTIROTOR: (30.0, 120.0),
            PlatformType.FIXED_WING_UAS: (5.0, 60.0),
            PlatformType.EVTOL_UAM: (1.0, 20.0),
            PlatformType.GA: (2.0, 20.0),
            PlatformType.DEBRIS: (1.0, 40.0),
        }
        for ptype, (lo, hi) in limits_deg.items():
            rng = _rng(11000 + int(ptype))
            for _ in range(50):
                env = make_envelope(ptype, rng)
                deg = env.omega_max * RAD
                self.assertGreaterEqual(deg, lo, ptype.name)
                self.assertLessEqual(deg, hi, ptype.name)

    def test_cruise_turn_radius_is_kilometre_scale_for_fast_platforms(self) -> None:
        for ptype in (PlatformType.EVTOL_UAM, PlatformType.GA):
            env = make_envelope(ptype, _rng(12000 + int(ptype)))
            v = 0.5 * (env.v_min + env.v_max)
            radius = env.turn_radius(v)
            self.assertGreater(radius, 100.0, ptype.name)
            self.assertLess(radius, 20000.0, ptype.name)

    def test_multirotor_tilt_cap_matches_lateral_accel(self) -> None:
        env = make_envelope(PlatformType.MULTIROTOR, _rng(13000))
        self.assertFalse(env.bank_limited)
        self.assertTrue(env.can_hover)
        v = 10.0
        self.assertAlmostEqual(
            env.max_turn_rate(v),
            min(env.lateral_accel_max / v, env.omega_max),
            places=12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
