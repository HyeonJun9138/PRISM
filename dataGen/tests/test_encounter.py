"""Tests for the inverse encounter construction (doc 3.4).

The central claim of :mod:`environment.encounter` is that the initial
condition it solves for reproduces the REQUESTED closest-point-of-approach
distance and time exactly, for every geometry class. If that were only
approximately true, every stratified analysis in the paper would be labelled
with numbers that do not describe the data.

Run directly:
    python $PRISM_ROOT/dataGen/tests/test_encounter.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.rng import RNGHub                                    # noqa: E402
from core.state import AgentState, PerformanceEnvelope, wrap_angle  # noqa: E402
from core.types import GeometryType                            # noqa: E402
from environment.encounter import (                            # noqa: E402
    DESIGNED_GEOMETRIES,
    GEOMETRY_ANGLE_RANGES_RAD,
    build_encounter,
    classify_geometry,
    compute_cpa,
    geometry_of_meta,
    sample_geometry,
)
from scenario.config import EncounterConfig                    # noqa: E402
from environment.wind import WindField                         # noqa: E402
from environment.world import Corridor, NoFlyZone, World       # noqa: E402
from scenario.config import ScenarioConfig, WindConfig         # noqa: E402


#: Number of random draws per geometry class [-].
N_DRAWS = 500

#: Absolute tolerance [m] / [s] for the CPA round trip.
TOL = 1e-6


_failures: list[str] = []
_checks = 0


def check(condition: bool, message: str) -> None:
    """Record one assertion.

    Args:
        condition: the predicate that must hold.
        message: description printed if it does not.
    """
    global _checks
    _checks += 1
    if not condition:
        _failures.append(message)


def _fixed_wing_envelope() -> PerformanceEnvelope:
    """A fixed-wing UAS envelope (doc 3.2 (3) initial values, 20-40 m/s)."""
    return PerformanceEnvelope(
        v_min=20.0,
        v_max=40.0,
        a_max=2.0,
        a_min=-3.0,
        omega_max=0.35,
        omega_dot_max=0.30,
        bank_max=math.radians(35.0),
        bank_limited=True,
        lateral_accel_max=0.0,
        can_hover=False,
        is_powered=True,
        wind_susceptibility=1.0,
        mass=12.0,
        drag_area=0.25,
    )


def _multirotor_envelope() -> PerformanceEnvelope:
    """A small multirotor envelope (doc 3.2 (3) initial values, 0-20 m/s)."""
    return PerformanceEnvelope(
        v_min=0.0,
        v_max=20.0,
        a_max=4.0,
        a_min=-4.0,
        omega_max=1.20,
        omega_dot_max=1.50,
        bank_max=math.radians(35.0),
        bank_limited=False,
        lateral_accel_max=6.0,
        can_hover=True,
        is_powered=True,
        wind_susceptibility=1.0,
        mass=3.0,
        drag_area=0.12,
    )


def _evtol_envelope() -> PerformanceEnvelope:
    """An eVTOL cruise envelope (doc 3.2 (3) initial values, 50-80 m/s)."""
    return PerformanceEnvelope(
        v_min=50.0,
        v_max=80.0,
        a_max=1.5,
        a_min=-2.0,
        omega_max=0.20,
        omega_dot_max=0.15,
        bank_max=math.radians(25.0),
        bank_limited=True,
        lateral_accel_max=0.0,
        can_hover=True,
        is_powered=True,
        wind_susceptibility=0.8,
        mass=1200.0,
        drag_area=2.5,
    )


# --------------------------------------------------------------------------
# compute_cpa
# --------------------------------------------------------------------------


def test_compute_cpa_basics() -> None:
    """Analytic cases for the forward CPA formula."""
    # Head-on along +X, closing at 20 m/s from 200 m, offset 5 m laterally.
    d, t = compute_cpa((200.0, 5.0), (-20.0, 0.0))
    check(abs(t - 10.0) < 1e-12, f"head-on tcpa: expected 10 s, got {t}")
    check(abs(d - 5.0) < 1e-12, f"head-on cpa: expected 5 m, got {d}")

    # Zero relative velocity -> never approaches.
    d, t = compute_cpa((30.0, 40.0), (0.0, 0.0))
    check(abs(d - 50.0) < 1e-12, f"static cpa: expected 50 m, got {d}")
    check(t == 0.0, f"static tcpa: expected 0 s, got {t}")

    # Receding pair -> negative tcpa unless clamped.
    d, t = compute_cpa((100.0, 0.0), (20.0, 0.0))
    check(t < 0.0, f"receding tcpa should be negative, got {t}")
    d_c, t_c = compute_cpa((100.0, 0.0), (20.0, 0.0), clamp_future=True)
    check(t_c == 0.0, f"clamped tcpa should be 0 s, got {t_c}")
    check(abs(d_c - 100.0) < 1e-12, f"clamped cpa: expected 100 m, got {d_c}")


# --------------------------------------------------------------------------
# classify_geometry
# --------------------------------------------------------------------------


def test_classify_geometry() -> None:
    """Band edges of the geometry classifier."""
    cases = [
        (0.0, GeometryType.OVERTAKING),
        (29.9, GeometryType.OVERTAKING),
        (30.1, GeometryType.CROSSING),
        (90.0, GeometryType.CROSSING),
        (149.9, GeometryType.CROSSING),
        (150.1, GeometryType.HEAD_ON),
        (180.0, GeometryType.HEAD_ON),
        (-90.0, GeometryType.CROSSING),
        (-170.0, GeometryType.HEAD_ON),
    ]
    for deg, expect in cases:
        got = classify_geometry(0.0, math.radians(deg))
        check(got == expect, f"classify_geometry({deg} deg) = {got}, want {expect}")
    # Invariance to the ownship heading: only the difference matters.
    for own_deg in (-179.0, -45.0, 0.0, 91.0, 179.0):
        own = math.radians(own_deg)
        got = classify_geometry(own, wrap_angle(own + math.radians(170.0)))
        check(
            got == GeometryType.HEAD_ON,
            f"classify_geometry is not heading-invariant at own={own_deg} deg",
        )


# --------------------------------------------------------------------------
# The main claim: the inverse construction is exact
# --------------------------------------------------------------------------


def test_inverse_construction_reproduces_cpa() -> None:
    """500 random draws per geometry class must reproduce (d*, t*) exactly."""
    cfg = EncounterConfig()
    hub = RNGHub(20260812)
    rng = hub.stream("scenario")
    place = hub.stream("placement")
    envelopes = {
        "fixed_wing": _fixed_wing_envelope(),
        "multirotor": _multirotor_envelope(),
        "evtol": _evtol_envelope(),
    }

    worst_d = 0.0
    worst_t = 0.0
    for geometry in DESIGNED_GEOMETRIES:
        lo, hi = GEOMETRY_ANGLE_RANGES_RAD[int(geometry)]
        n_ok = 0
        for i in range(N_DRAWS):
            # A fresh, arbitrary ownship state every draw.
            own = AgentState(
                x=float(place.uniform(-1000.0, 1000.0)),
                y=float(place.uniform(-1000.0, 1000.0)),
                v=float(place.uniform(30.0, 80.0)),
                psi=wrap_angle(float(place.uniform(-math.pi, math.pi))),
            )
            name = list(envelopes)[i % len(envelopes)]
            env = envelopes[name]
            state, meta = build_encounter(own, geometry, cfg, env, rng)

            # Round trip through the forward formula, from the returned state.
            p_rel = (state.x - own.x, state.y - own.y)
            v_rel = (
                state.v * math.cos(state.psi) - own.v * math.cos(own.psi),
                state.v * math.sin(state.psi) - own.v * math.sin(own.psi),
            )
            cpa, tcpa = compute_cpa(p_rel, v_rel)
            err_d = abs(cpa - meta["cpa_dist"])
            err_t = abs(tcpa - meta["tcpa"])
            worst_d = max(worst_d, err_d)
            worst_t = max(worst_t, err_t)
            check(
                err_d <= TOL,
                f"{geometry.name}/{name} draw {i}: CPA error {err_d:.3e} m > {TOL}",
            )
            check(
                err_t <= TOL,
                f"{geometry.name}/{name} draw {i}: TCPA error {err_t:.3e} s > {TOL}",
            )

            # Design values must respect the configured ranges.
            check(
                cfg.cpa_dist_min_m <= meta["cpa_dist"] <= cfg.cpa_dist_max_m,
                f"{geometry.name} draw {i}: d* {meta['cpa_dist']} outside config range",
            )
            check(
                cfg.tcpa_min_s <= meta["tcpa"] <= cfg.tcpa_max_s,
                f"{geometry.name} draw {i}: t* {meta['tcpa']} outside config range",
            )

            # The approach angle must sit in the requested band, and the
            # resulting heading pair must classify back to the same class.
            chi = meta["chi_rad"]
            check(
                lo - 1e-12 <= chi <= hi + 1e-12,
                f"{geometry.name} draw {i}: chi {math.degrees(chi)} deg out of band",
            )
            check(
                classify_geometry(own.psi, state.psi) == geometry,
                f"{geometry.name} draw {i}: classify_geometry disagrees with design",
            )
            check(
                geometry_of_meta(meta) == geometry,
                f"{geometry.name} draw {i}: meta geometry mismatch",
            )

            # Speed must be inside the platform envelope, heading wrapped.
            check(
                env.v_min - 1e-9 <= state.v <= env.v_max + 1e-9,
                f"{geometry.name} draw {i}: speed {state.v} outside envelope",
            )
            check(
                -math.pi < state.psi <= math.pi + 1e-15,
                f"{geometry.name} draw {i}: psi {state.psi} not wrapped",
            )
            check(
                state.omega == 0.0 and state.a == 0.0,
                f"{geometry.name} draw {i}: intruder must start in CV",
            )
            check(
                meta["speed_rel"] >= 2.0 - 1e-9,
                f"{geometry.name} draw {i}: |v_rel| {meta['speed_rel']} below floor",
            )
            n_ok += 1
        check(n_ok == N_DRAWS, f"{geometry.name}: only {n_ok}/{N_DRAWS} draws built")

    print(f"    worst CPA error  = {worst_d:.3e} m")
    print(f"    worst TCPA error = {worst_t:.3e} s")


def test_inverse_construction_side_and_range_options() -> None:
    """Both miss sides occur, and the range limits are honoured."""
    cfg = EncounterConfig()
    rng = np.random.default_rng(7)
    env = _fixed_wing_envelope()
    own = AgentState(x=0.0, y=0.0, v=55.0, psi=0.0)

    sides = set()
    for _ in range(200):
        _, meta = build_encounter(own, GeometryType.CROSSING, cfg, env, rng)
        sides.add(meta["side_sign"])
    check(sides == {1.0, -1.0}, f"both miss sides must occur, got {sides}")

    for _ in range(100):
        state, meta = build_encounter(
            own, GeometryType.HEAD_ON, cfg, env, rng,
            min_range_m=500.0, max_range_m=4000.0,
        )
        r = math.hypot(state.x - own.x, state.y - own.y)
        check(
            500.0 - 1e-9 <= r <= 4000.0 + 1e-9,
            f"range constraint violated: {r} m",
        )
        check(abs(r - meta["range0"]) < 1e-9, "meta range0 disagrees with the state")


def test_degenerate_and_invalid_inputs() -> None:
    """Guards: BACKGROUND is rejected, impossible constraints raise."""
    cfg = EncounterConfig()
    rng = np.random.default_rng(11)
    env = _fixed_wing_envelope()
    own = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0)

    try:
        build_encounter(own, GeometryType.BACKGROUND, cfg, env, rng)
        check(False, "BACKGROUND geometry must raise ValueError")
    except ValueError:
        check(True, "")

    # An impossible range window must raise rather than loop or return junk.
    try:
        build_encounter(
            own, GeometryType.CROSSING, cfg, env, rng,
            min_range_m=1.0e9, max_attempts=8,
        )
        check(False, "impossible min_range_m must raise ValueError")
    except ValueError:
        check(True, "")

    # Co-speed overtaking: the |v_rel| guard must still find a solution by
    # moving the intruder speed to an envelope endpoint.
    own_match = AgentState(x=0.0, y=0.0, v=30.0, psi=0.0)
    for _ in range(200):
        state, meta = build_encounter(
            own_match, GeometryType.OVERTAKING, cfg, env, rng
        )
        check(
            meta["speed_rel"] >= 2.0 - 1e-9,
            f"co-speed overtaking produced |v_rel| = {meta['speed_rel']}",
        )
        p_rel = (state.x, state.y)
        v_rel = (
            state.v * math.cos(state.psi) - own_match.v,
            state.v * math.sin(state.psi),
        )
        cpa, tcpa = compute_cpa(p_rel, v_rel)
        check(abs(cpa - meta["cpa_dist"]) <= TOL, "co-speed CPA round trip failed")
        check(abs(tcpa - meta["tcpa"]) <= TOL, "co-speed TCPA round trip failed")


def test_sample_geometry_respects_weights() -> None:
    """BACKGROUND is never designed and zero-weight classes never appear."""
    rng = np.random.default_rng(3)
    cfg = EncounterConfig(
        geometry_weights={
            int(GeometryType.HEAD_ON): 1.0,
            int(GeometryType.CROSSING): 0.0,
            int(GeometryType.OVERTAKING): 0.0,
            int(GeometryType.BACKGROUND): 99.0,
        }
    )
    got = {sample_geometry(cfg, rng) for _ in range(500)}
    check(got == {GeometryType.HEAD_ON}, f"weighted sampling gave {got}")

    cfg_zero = EncounterConfig(
        geometry_weights={int(GeometryType.BACKGROUND): 1.0}
    )
    got_zero = {sample_geometry(cfg_zero, rng) for _ in range(500)}
    check(
        got_zero == set(DESIGNED_GEOMETRIES),
        f"all-zero weights must fall back to uniform, got {got_zero}",
    )


def test_determinism() -> None:
    """The same seed must rebuild the identical encounter."""
    cfg = EncounterConfig()
    env = _evtol_envelope()
    own = AgentState(x=10.0, y=-20.0, v=55.0, psi=0.7)
    out = []
    for _ in range(2):
        rng = RNGHub(4242).stream("scenario")
        s, m = build_encounter(own, GeometryType.CROSSING, cfg, env, rng)
        out.append((s.x, s.y, s.v, s.psi, m["cpa_dist"], m["tcpa"]))
    check(out[0] == out[1], f"encounter construction is not deterministic: {out}")


# --------------------------------------------------------------------------
# Companion smoke tests for the rest of the environment package
# --------------------------------------------------------------------------


def test_wind_field() -> None:
    """Steady wind sampling, gust statistics and snapshot round trip."""
    cfg = WindConfig(speed_min_ms=4.0, speed_max_ms=4.0, dir_min_rad=0.0,
                     dir_max_rad=0.0)
    rng = np.random.default_rng(1)
    wf = WindField(cfg, rng)
    wx, wy = wf.at(0.0, (0.0, 0.0))
    check(abs(wx - 4.0) < 1e-12 and abs(wy) < 1e-12, f"steady wind = {(wx, wy)}")

    # Gusts off: step must not consume the stream.
    r1 = np.random.default_rng(5)
    before = r1.bit_generator.state
    wf.step(0.05, r1)
    check(r1.bit_generator.state == before, "gust-off step consumed the rng")

    # Gusts on: stationary std must approach gust_sigma_ms.
    gcfg = WindConfig(speed_min_ms=0.0, speed_max_ms=0.0, dir_min_rad=0.0,
                      dir_max_rad=0.0, gust_enabled=True, gust_sigma_ms=2.0,
                      gust_tau_s=5.0)
    gwf = WindField(gcfg, np.random.default_rng(2))
    grng = np.random.default_rng(3)
    samples = []
    for k in range(40000):
        gwf.step(0.05, grng)
        if k > 2000:
            samples.append(gwf.gust[0])
    sd = float(np.std(np.asarray(samples)))
    check(abs(sd - 2.0) < 0.25, f"OU stationary sigma = {sd:.3f}, want ~2.0")

    snap = gwf.snapshot()
    g_before = gwf.gust
    gwf.step(0.5, grng)
    gwf.restore(snap)
    check(gwf.gust == g_before, "wind snapshot/restore round trip failed")
    check("wx" in gwf.to_render_dict(), "to_render_dict must expose wx")


def test_world() -> None:
    """Bounds, containment, corridor projection and NFZ queries."""
    cfg = ScenarioConfig(area_km2=4.0, corridor_enabled=True, nfz_count=2)
    world = World(cfg, np.random.default_rng(9))
    check(abs(world.side_m - 2000.0) < 1e-9, f"side = {world.side_m} m")
    check(abs(world.half_m - 1000.0) < 1e-9, f"half = {world.half_m} m")
    check(len(world.corridors) == 1, "one corridor expected")
    check(len(world.nfz) <= 2, "at most nfz_count zones")

    check(world.contains((0.0, 0.0)), "origin must be inside")
    check(world.out_of_bounds((1500.0, 0.0)), "far point must be outside")

    # Deep inside: no containment command.
    check(
        world.containment_heading((0.0, 0.0), 0.0) is None,
        "no containment deep inside the region",
    )
    # Near the +X wall heading outward: must be commanded back inward.
    psi_des = world.containment_heading((995.0, 0.0), 0.0)
    check(psi_des is not None, "containment must trigger near the wall")
    if psi_des is not None:
        check(
            math.cos(psi_des) < 0.0,
            f"containment heading must point inward, got {math.degrees(psi_des)} deg",
        )
    # Near the wall but heading inward: leave it alone.
    check(
        world.containment_heading((995.0, 0.0), math.pi) is None,
        "inward-flying agent inside the band must not be commanded",
    )

    # Hard reflection fallback.
    st = AgentState(x=1200.0, y=0.0, v=30.0, psi=0.0)
    hit = world.enforce_bounds(st)
    check(hit, "enforce_bounds must fire outside the wall")
    check(abs(st.x - 1000.0) < 1e-9, f"x clamped to the wall, got {st.x}")
    check(abs(abs(st.psi) - math.pi) < 1e-9, f"heading mirrored, got {st.psi}")

    # Corridor projection on an explicit corridor.
    w2 = World(
        cfg,
        corridors=[Corridor((-1000.0, 0.0), (1000.0, 0.0), 100.0)],
        nfz=[NoFlyZone((500.0, 500.0), 200.0)],
    )
    q = w2.nearest_corridor((0.0, 50.0))
    check(q.index == 0, "corridor index")
    check(abs(q.distance - 50.0) < 1e-9, f"corridor distance = {q.distance}")
    check(abs(q.signed_offset - 50.0) < 1e-9, "left of p0->p1 must be positive")
    check(q.inside, "50 m offset is inside a 100 m half-width corridor")
    q2 = w2.nearest_corridor((0.0, -50.0))
    check(abs(q2.signed_offset + 50.0) < 1e-9, "right of p0->p1 must be negative")
    q3 = w2.nearest_corridor((2000.0, 0.0))
    check(abs(q3.s_along - 2000.0) < 1e-9, "s_along clamps to the segment length")
    check(abs(q3.distance - 1000.0) < 1e-9, f"clamped distance = {q3.distance}")

    check(w2.inside_nfz((500.0, 500.0)), "zone centre is inside the zone")
    check(not w2.inside_nfz((0.0, 0.0)), "origin is outside the zone")
    idx, d = w2.nearest_nfz((500.0, 500.0))
    check(idx == 0 and d < 0.0, f"signed NFZ distance inside must be negative: {d}")

    # Free-position sampling avoids the zones and the margin band.
    prng = np.random.default_rng(13)
    for _ in range(500):
        p = w2.sample_free_position(prng, nfz_clearance=10.0)
        check(w2.contains(p, 0.0), f"sampled position out of bounds: {p}")
        check(not w2.inside_nfz(p), f"sampled position inside a NFZ: {p}")

    # Snapshot round trip.
    snap = world.snapshot()
    rebuilt = World.from_snapshot(snap)
    check(rebuilt.snapshot() == snap, "world snapshot round trip failed")
    check("corridors" in world.to_render_dict(), "render dict must list corridors")


def main() -> int:
    """Run every test and report.

    Returns:
        Process exit code [-]: 0 if all checks passed, 1 otherwise.
    """
    tests = [
        test_compute_cpa_basics,
        test_classify_geometry,
        test_inverse_construction_reproduces_cpa,
        test_inverse_construction_side_and_range_options,
        test_degenerate_and_invalid_inputs,
        test_sample_geometry_respects_weights,
        test_determinism,
        test_wind_field,
        test_world,
    ]
    for fn in tests:
        n_before = len(_failures)
        fn()
        status = "ok" if len(_failures) == n_before else "FAIL"
        print(f"[{status}] {fn.__name__}")
    print(f"\n{_checks} checks, {len(_failures)} failures")
    for msg in _failures[:40]:
        print("  - " + msg)
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
