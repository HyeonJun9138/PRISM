"""Tests for :mod:`agents.intent`.

Run directly::

    python $PRISM_ROOT/dataGen/tests/test_intent.py

Every ``test_*`` function is also collectable by pytest, but the module needs
no third-party package: ``main()`` runs the whole suite with plain asserts and
a check counter.

What is verified (the three properties the task calls for, plus the guards
that protect them):

1. every controller runs a full 60 s episode at ``SIM_DT`` with no NaN/inf,
2. every command respects the platform envelope at every single step
   (``|omega_cmd| <= max_turn_rate(v)`` and ``a_min <= a_cmd <= a_max``),
3. ``snapshot -> restore -> rerun`` reproduces an identical trajectory
   BIT-FOR-BIT when the generator state is restored as well -- the property
   that branch rollouts (doc 3.3) depend on.

Because ``agents/platforms.py`` and ``agents/base.py`` are written by parallel
tasks, this file defines its own minimal agent container and its own envelope
table. Both are test fixtures only: the envelopes here are the doc 3.2 (3)
initial values and are NOT the authority for dataset generation.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE).replace("\\", "/")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.intent import (  # noqa: E402
    INTENT_CLASSES,
    CorridorTransitIntent,
    DeliveryWaypointIntent,
    IntentController,
    LoiterHoverIntent,
    PassiveDriftIntent,
    ReactiveAvoidIntent,
    SearchWanderIntent,
    cross_track_guidance,
    make_intent,
    orbit_guidance,
    resolve_intent_type,
)
from core.constants import SIM_DT  # noqa: E402
from core.state import AgentState, PerformanceEnvelope, wrap_angle  # noqa: E402
from core.types import IntentType, PlatformType  # noqa: E402

CHECKS = 0


def check(condition: bool, message: str) -> None:
    """Assert ``condition`` and count the check."""
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------
# Fixtures: minimal agent + envelope table (doc 3.2 (3) initial values)
# --------------------------------------------------------------------------

DEG = math.pi / 180.0


def envelope_for(ptype: PlatformType) -> PerformanceEnvelope:
    """Test-fixture performance envelope [SI] for one platform class."""
    if ptype is PlatformType.MULTIROTOR:
        return PerformanceEnvelope(
            v_min=0.0, v_max=20.0, a_max=3.5, a_min=-3.5,
            omega_max=1.2, omega_dot_max=1.5, bank_max=30.0 * DEG,
            bank_limited=False, lateral_accel_max=6.0,
            can_hover=True, is_powered=True, wind_susceptibility=0.9,
            mass=2.5, drag_area=0.08,
        )
    if ptype is PlatformType.FIXED_WING_UAS:
        return PerformanceEnvelope(
            v_min=18.0, v_max=40.0, a_max=2.0, a_min=-2.5,
            omega_max=0.60, omega_dot_max=0.8, bank_max=35.0 * DEG,
            bank_limited=True, lateral_accel_max=0.0,
            can_hover=False, is_powered=True, wind_susceptibility=0.55,
            mass=12.0, drag_area=0.25,
        )
    if ptype is PlatformType.EVTOL_UAM:
        # can_hover is False on purpose: doc 7.2 limits this study to the
        # cruise phase, where hover transitions are excluded.
        return PerformanceEnvelope(
            v_min=35.0, v_max=80.0, a_max=1.5, a_min=-2.0,
            omega_max=0.35, omega_dot_max=0.35, bank_max=25.0 * DEG,
            bank_limited=True, lateral_accel_max=0.0,
            can_hover=False, is_powered=True, wind_susceptibility=0.35,
            mass=2200.0, drag_area=2.2,
        )
    if ptype is PlatformType.BIRD:
        return PerformanceEnvelope(
            v_min=8.0, v_max=25.0, a_max=5.0, a_min=-5.0,
            omega_max=2.0, omega_dot_max=4.0, bank_max=50.0 * DEG,
            bank_limited=True, lateral_accel_max=0.0,
            can_hover=False, is_powered=True, wind_susceptibility=1.0,
            mass=1.0, drag_area=0.05,
        )
    if ptype is PlatformType.GA:
        return PerformanceEnvelope(
            v_min=30.0, v_max=70.0, a_max=1.5, a_min=-2.0,
            omega_max=0.30, omega_dot_max=0.3, bank_max=25.0 * DEG,
            bank_limited=True, lateral_accel_max=0.0,
            can_hover=False, is_powered=True, wind_susceptibility=0.3,
            mass=1100.0, drag_area=1.5,
        )
    # DEBRIS / UNKNOWN: unpowered ballistic body.
    return PerformanceEnvelope(
        v_min=0.0, v_max=30.0, a_max=0.0, a_min=0.0,
        omega_max=0.05, omega_dot_max=0.05, bank_max=0.0,
        bank_limited=False, lateral_accel_max=0.0,
        can_hover=False, is_powered=False, wind_susceptibility=1.0,
        mass=0.5, drag_area=0.20,
    )


class FakeAgent:
    """Minimal stand-in for ``agents.base.Agent`` (``.state`` + ``.envelope``)."""

    def __init__(self, state: AgentState, envelope: PerformanceEnvelope,
                 platform_type: PlatformType) -> None:
        self.state = state
        self.envelope = envelope
        self.platform_type = platform_type


class FakeWorld:
    """Minimal stand-in for ``environment.world.World`` (region extent only)."""

    def __init__(self, side_m: float = 2000.0) -> None:
        self.region_side_m = float(side_m)


def spawn_agent(ptype: PlatformType, x: float = 0.0, y: float = 0.0,
                psi: float = 0.0, v: float | None = None) -> FakeAgent:
    """Create a fake agent of ``ptype`` at a nominal cruise speed."""
    env = envelope_for(ptype)
    if v is None:
        v = env.clamp_speed(0.65 * env.v_max)
    return FakeAgent(AgentState(x=x, y=y, v=float(v), psi=float(psi)),
                     env, ptype)


def integrate(agent: FakeAgent, omega_cmd, a_cmd, dt: float) -> None:
    """Advance the fake agent one step with the contract's rate limiting.

    Mirrors what ``core.integrator`` will do: rate-limit ``omega`` toward the
    command with ``omega_dot_max``, clamp the accel, integrate speed, heading
    and position (forward Euler). ``(None, None)`` = passive body -> coast.
    """
    s = agent.state
    env = agent.envelope
    if omega_cmd is None or a_cmd is None:
        omega_cmd, a_cmd = 0.0, 0.0
    target = env.clamp_turn_rate(float(omega_cmd), s.v)
    delta = target - s.omega
    lim = env.omega_dot_max * dt
    if delta > lim:
        delta = lim
    elif delta < -lim:
        delta = -lim
    s.omega = s.omega + delta
    s.a = env.clamp_accel(float(a_cmd))
    s.v = env.clamp_speed(s.v + s.a * dt)
    s.psi = wrap_angle(s.psi + s.omega * dt)
    s.x = s.x + s.v * math.cos(s.psi) * dt
    s.y = s.y + s.v * math.sin(s.psi) * dt


POWERED_PLATFORMS = (
    PlatformType.MULTIROTOR,
    PlatformType.FIXED_WING_UAS,
    PlatformType.EVTOL_UAM,
    PlatformType.BIRD,
)

POWERED_INTENTS = (
    IntentType.CORRIDOR_TRANSIT,
    IntentType.DELIVERY_WP,
    IntentType.LOITER_HOVER,
    IntentType.SEARCH_WANDER,
    IntentType.REACTIVE_AVOID,
)

DURATION_S = 60.0
N_STEPS = int(round(DURATION_S / SIM_DT))


def run_episode(
    controller: IntentController,
    agent: FakeAgent,
    rng: np.random.Generator,
    n_steps: int = N_STEPS,
    ownship: FakeAgent | None = None,
    t0: float = 0.0,
    check_limits: bool = True,
    label: str = "",
) -> list[tuple[float, float, float, float]]:
    """Run ``n_steps`` of closed-loop simulation, returning the trajectory.

    Returns:
        A list of ``(x, y, psi, v)`` tuples, one per step, for bit-exact
        comparison.
    """
    traj: list[tuple[float, float, float, float]] = []
    env = agent.envelope
    for k in range(n_steps):
        t = t0 + k * SIM_DT
        own_state = ownship.state if ownship is not None else None
        omega_cmd, a_cmd = controller.command(agent, t, own_state, rng)
        if check_limits:
            if omega_cmd is None:
                assert a_cmd is None, f"{label}: half-passive command"
            else:
                cap = env.max_turn_rate(agent.state.v)
                assert math.isfinite(omega_cmd), f"{label}: omega not finite"
                assert math.isfinite(a_cmd), f"{label}: accel not finite"
                assert abs(omega_cmd) <= cap + 1e-9, (
                    f"{label}: omega {omega_cmd} exceeds cap {cap} "
                    f"at v={agent.state.v}"
                )
                assert env.a_min - 1e-9 <= a_cmd <= env.a_max + 1e-9, (
                    f"{label}: accel {a_cmd} outside "
                    f"[{env.a_min}, {env.a_max}]"
                )
        integrate(agent, omega_cmd, a_cmd, SIM_DT)
        if ownship is not None:
            integrate(ownship, 0.0, 0.0, SIM_DT)
        s = agent.state
        assert math.isfinite(s.x) and math.isfinite(s.y), f"{label}: NaN position"
        assert math.isfinite(s.psi) and math.isfinite(s.v), f"{label}: NaN state"
        assert -math.pi - 1e-12 < s.psi <= math.pi + 1e-12, f"{label}: psi unwrapped"
        traj.append((s.x, s.y, s.psi, s.v))
    return traj


# --------------------------------------------------------------------------
# 1. Guidance primitives
# --------------------------------------------------------------------------


def test_cross_track_guidance() -> None:
    """Cross-track law turns toward the centreline and is bounded."""
    # On the line, heading east: desired heading equals the path heading.
    psi = cross_track_guidance(0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.06, 0.4)
    check(abs(psi) < 1e-12, "on-centreline desired heading must equal path heading")
    # 200 m to the LEFT (north) of an eastbound path -> must aim right (south).
    psi = cross_track_guidance(0.0, 200.0, 0.0, 0.0, 0.0, 30.0, 0.06, 0.6)
    check(psi < 0.0, "left of the path must command a right (CW) intercept")
    # 200 m to the RIGHT -> aim left.
    psi2 = cross_track_guidance(0.0, -200.0, 0.0, 0.0, 0.0, 30.0, 0.06, 0.6)
    check(psi2 > 0.0, "right of the path must command a left (CCW) intercept")
    check(abs(psi + psi2) < 1e-12, "cross-track law must be antisymmetric")
    # The intercept angle is clipped.
    psi3 = cross_track_guidance(0.0, 100000.0, 0.0, 0.0, 0.0, 30.0, 0.06, 0.2)
    check(abs(psi3) <= 0.2 + 1e-12, "intercept angle must respect the clip")
    # Works on a non-zero path heading too (rotational consistency).
    ph = 1.0
    psi4 = cross_track_guidance(-200.0 * math.sin(ph), 200.0 * math.cos(ph),
                                0.0, 0.0, ph, 30.0, 0.06, 0.6)
    check(abs(wrap_angle(psi4 - ph) - (psi - 0.0)) < 1e-9,
          "cross-track law must be invariant to path rotation")


def test_orbit_guidance() -> None:
    """Orbit field is tangential on the circle and inward far outside it."""
    # On the circle at bearing 0 from the centre, CCW -> heading +90 deg.
    psi = orbit_guidance(100.0, 0.0, 0.0, 0.0, 100.0, +1)
    check(abs(wrap_angle(psi - 0.5 * math.pi)) < 1e-12,
          "CCW orbit must be tangential (+90 deg) on the circle")
    psi = orbit_guidance(100.0, 0.0, 0.0, 0.0, 100.0, -1)
    check(abs(wrap_angle(psi + 0.5 * math.pi)) < 1e-12,
          "CW orbit must be tangential (-90 deg) on the circle")
    # Far outside -> point at the centre regardless of turn direction.
    for d in (+1, -1):
        psi = orbit_guidance(1.0e7, 0.0, 0.0, 0.0, 100.0, d)
        check(abs(wrap_angle(psi - math.pi)) < 1e-4,
              "far outside the orbit the field must point at the centre")
    # Inside the circle -> the heading rotates back off the tangent so that it
    # gains an outward radial component (at bearing 0 that means 0 < psi < 90).
    psi = orbit_guidance(10.0, 0.0, 0.0, 0.0, 100.0, +1)
    check(0.0 < psi < 0.5 * math.pi,
          "inside the orbit the CCW field must acquire an outward component")
    check(math.cos(psi) > 0.0,
          "inside the orbit the velocity must point away from the centre")


# --------------------------------------------------------------------------
# 2. Compatibility guard / fallback table
# --------------------------------------------------------------------------


def test_fallback_unpowered_becomes_passive() -> None:
    """Any intent on an unpowered body falls back to PASSIVE_DRIFT."""
    agent = spawn_agent(PlatformType.DEBRIS, v=12.0)
    rng = np.random.default_rng(1)
    for it in IntentType:
        eff = resolve_intent_type(it, agent.envelope)
        check(eff is IntentType.PASSIVE_DRIFT,
              f"unpowered {it.name} must fall back to PASSIVE_DRIFT")
        ctrl = make_intent(it, agent, None, rng)
        check(isinstance(ctrl, PassiveDriftIntent),
              f"unpowered {it.name} must build a PassiveDriftIntent")
        check(ctrl.requested_type is it, "requested_type must record the request")
        check(ctrl.intent_type is IntentType.PASSIVE_DRIFT,
              "intent_type must report the effective intent")


def test_fallback_passive_on_powered() -> None:
    """PASSIVE_DRIFT on a powered platform falls back to CORRIDOR_TRANSIT."""
    for pt in POWERED_PLATFORMS:
        agent = spawn_agent(pt)
        ctrl = make_intent(IntentType.PASSIVE_DRIFT, agent, FakeWorld(),
                           np.random.default_rng(2))
        check(isinstance(ctrl, CorridorTransitIntent),
              f"{pt.name}: PASSIVE_DRIFT must fall back to CORRIDOR_TRANSIT")
        check(ctrl.requested_type is IntentType.PASSIVE_DRIFT,
              "the requested intent must still be visible after a fallback")


def test_fallback_loiter_variant() -> None:
    """LOITER_HOVER picks the orbit variant when the platform cannot hover."""
    hoverer = spawn_agent(PlatformType.MULTIROTOR)
    ctrl = make_intent(IntentType.LOITER_HOVER, hoverer, FakeWorld(),
                       np.random.default_rng(3))
    check(isinstance(ctrl, LoiterHoverIntent), "must build a LoiterHoverIntent")
    check(ctrl.variant() == "hover", "a hover-capable platform gets the hover variant")
    for pt in (PlatformType.FIXED_WING_UAS, PlatformType.EVTOL_UAM,
               PlatformType.BIRD):
        agent = spawn_agent(pt)
        ctrl = make_intent(IntentType.LOITER_HOVER, agent, FakeWorld(),
                           np.random.default_rng(4))
        check(ctrl.variant() == "orbit",
              f"{pt.name} cannot hover -> orbit variant required")
        check(ctrl.intent_type is IntentType.LOITER_HOVER,
              "the orbit fallback must keep the LOITER_HOVER label")
        r = float(ctrl.state_dict["radius"])
        r_min = agent.envelope.turn_radius(agent.state.v)
        check(r >= r_min - 1e-6,
              f"{pt.name}: orbit radius {r:.1f} m below min turn radius "
              f"{r_min:.1f} m")


def test_intent_class_table() -> None:
    """Every IntentType has a controller class, mapped to the right label."""
    for it in IntentType:
        check(it in INTENT_CLASSES, f"{it.name} missing from INTENT_CLASSES")
        check(INTENT_CLASSES[it].INTENT is it,
              f"{it.name} maps to a class with a different INTENT label")


# --------------------------------------------------------------------------
# 3. 60 s runs: no NaN, envelope respected every step
# --------------------------------------------------------------------------


def test_all_controllers_run_60s_within_envelope() -> None:
    """Each intent x platform runs 60 s cleanly and never violates the envelope."""
    world = FakeWorld(2000.0)
    for pt in POWERED_PLATFORMS:
        for it in POWERED_INTENTS:
            agent = spawn_agent(pt, x=-300.0, y=150.0, psi=0.4)
            ownship = spawn_agent(PlatformType.EVTOL_UAM, x=900.0, y=-400.0,
                                  psi=math.pi * 0.75, v=55.0)
            ctrl = make_intent(it, agent, world, np.random.default_rng(11))
            traj = run_episode(ctrl, agent, np.random.default_rng(12),
                               ownship=ownship, label=f"{pt.name}/{it.name}")
            check(len(traj) == N_STEPS,
                  f"{pt.name}/{it.name}: wrong step count")
            # Path length, not displacement: a loiter orbit legitimately ends
            # up back where it started.
            path = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                       for a, b in zip(traj, traj[1:]))
            if agent.envelope.v_min > 0.0:
                check(path > 0.5 * agent.envelope.v_min * DURATION_S,
                      f"{pt.name}/{it.name}: flew only {path:.1f} m in "
                      f"{DURATION_S:.0f} s despite v_min="
                      f"{agent.envelope.v_min:.1f} m/s")
            check(isinstance(ctrl.describe(), str) and ctrl.describe(),
                  f"{pt.name}/{it.name}: describe() must return a non-empty string")


def test_passive_drift_returns_none() -> None:
    """PASSIVE_DRIFT emits no commands at all, for the whole episode."""
    agent = spawn_agent(PlatformType.DEBRIS, v=14.0, psi=-0.6)
    ctrl = make_intent(IntentType.PASSIVE_DRIFT, agent, FakeWorld(),
                       np.random.default_rng(21))
    rng = np.random.default_rng(22)
    for k in range(N_STEPS):
        omega, a = ctrl.command(agent, k * SIM_DT, None, rng)
        assert omega is None and a is None, "passive drift must command nothing"
        integrate(agent, omega, a, SIM_DT)
    check(True, "passive drift returned (None, None) for 1200 steps")
    check(math.isfinite(agent.state.x) and math.isfinite(agent.state.y),
          "passive body must stay finite")


def test_hover_variant_slows_to_v_min() -> None:
    """The hover variant drives a hover-capable platform toward v_min."""
    agent = spawn_agent(PlatformType.MULTIROTOR, v=15.0)
    ctrl = make_intent(IntentType.LOITER_HOVER, agent, FakeWorld(),
                       np.random.default_rng(31))
    run_episode(ctrl, agent, np.random.default_rng(32), label="hover")
    check(agent.state.v <= agent.envelope.v_min + 1.0,
          f"hovering multirotor should be near v_min, got {agent.state.v:.2f} m/s")


def test_orbit_variant_holds_its_circle() -> None:
    """The orbit variant converges to and holds the sampled radius."""
    agent = spawn_agent(PlatformType.FIXED_WING_UAS, v=28.0)
    ctrl = make_intent(IntentType.LOITER_HOVER, agent, FakeWorld(),
                       np.random.default_rng(41))
    st = ctrl.state_dict
    cx, cy, radius = float(st["cx"]), float(st["cy"]), float(st["radius"])
    traj = run_episode(ctrl, agent, np.random.default_rng(42), label="orbit")
    errs = [abs(math.hypot(x - cx, y - cy) - radius) for x, y, _, _ in traj[200:]]
    check(max(errs) < 0.25 * radius,
          f"orbit radius error {max(errs):.1f} m too large for R={radius:.1f} m")


def test_delivery_captures_waypoints() -> None:
    """DELIVERY_WP advances through waypoints and therefore turns periodically."""
    for pt in (PlatformType.MULTIROTOR, PlatformType.FIXED_WING_UAS,
               PlatformType.EVTOL_UAM):
        agent = spawn_agent(pt)
        ctrl = make_intent(IntentType.DELIVERY_WP, agent, FakeWorld(),
                           np.random.default_rng(51))
        run_episode(ctrl, agent, np.random.default_rng(52), label=f"wp/{pt.name}")
        captures = int(ctrl.state_dict["captures"])
        check(captures >= 1,
              f"{pt.name}: DELIVERY_WP captured no waypoint in 60 s")
        check(int(ctrl.state_dict["idx"]) < len(ctrl.state_dict["wp"]),
              f"{pt.name}: waypoint index ran past the route")


def test_search_wander_reorients() -> None:
    """SEARCH_WANDER fires Poisson reorientations and produces real curvature."""
    agent = spawn_agent(PlatformType.BIRD)
    ctrl = make_intent(IntentType.SEARCH_WANDER, agent, FakeWorld(),
                       np.random.default_rng(61))
    traj = run_episode(ctrl, agent, np.random.default_rng(62), label="wander")
    check(int(ctrl.state_dict["reorients"]) >= 1,
          "SEARCH_WANDER produced no reorientation in 60 s")
    headings = [p[2] for p in traj]
    swept = sum(abs(wrap_angle(b - a)) for a, b in zip(headings, headings[1:]))
    check(swept > 2.0,
          f"SEARCH_WANDER swept only {math.degrees(swept):.0f} deg of heading")


def test_reactive_avoid_turns_away() -> None:
    """REACTIVE_AVOID latches a turn away from a converging ownship."""
    agent = spawn_agent(PlatformType.BIRD, x=0.0, y=0.0, psi=0.0, v=15.0)
    # Ownship 600 m out on a +45 deg relative bearing, flying at the agent.
    ownship = spawn_agent(PlatformType.EVTOL_UAM, x=424.0, y=424.0,
                          psi=-0.75 * math.pi, v=55.0)
    ctrl = make_intent(IntentType.REACTIVE_AVOID, agent, FakeWorld(),
                       np.random.default_rng(71))
    rng = np.random.default_rng(72)
    omega, _ = ctrl.command(agent, 0.0, ownship.state, rng)
    check(int(ctrl.state_dict["avoids"]) == 1,
          "REACTIVE_AVOID did not latch on a converging ownship")
    check(int(ctrl.state_dict["avoid_latch"]["sign"]) == -1,
          "must turn RIGHT (negative omega) away from an ownship on the left")
    check(omega is not None and omega < 0.0,
          "the evasive command must be a right turn")
    cap = agent.envelope.max_turn_rate(agent.state.v)
    check(abs(omega) > 0.5 * cap,
          "the evasive turn must use a high fraction of the turn-rate cap")
    # And it does NOT react to an ownship that is far away and receding.
    agent2 = spawn_agent(PlatformType.BIRD, x=0.0, y=0.0, psi=0.0, v=15.0)
    far = spawn_agent(PlatformType.EVTOL_UAM, x=-5000.0, y=5000.0, psi=0.75,
                      v=55.0)
    ctrl2 = make_intent(IntentType.REACTIVE_AVOID, agent2, FakeWorld(),
                        np.random.default_rng(73))
    run_episode(ctrl2, agent2, np.random.default_rng(74), ownship=far,
                label="ra-far")
    check(int(ctrl2.state_dict["avoids"]) == 0,
          "REACTIVE_AVOID must ignore a distant receding ownship")


def test_corridor_stays_near_centreline() -> None:
    """CORRIDOR_TRANSIT converges onto its centreline (the easy-to-predict class)."""
    agent = spawn_agent(PlatformType.EVTOL_UAM, x=0.0, y=0.0, psi=0.0, v=60.0)
    ctrl = make_intent(IntentType.CORRIDOR_TRANSIT, agent, FakeWorld(),
                       np.random.default_rng(81))
    st = ctrl.state_dict
    ox, oy, ph = float(st["ox"]), float(st["oy"]), float(st["ph"])
    traj = run_episode(ctrl, agent, np.random.default_rng(82), label="corridor")
    e_end = abs(-math.sin(ph) * (traj[-1][0] - ox)
                + math.cos(ph) * (traj[-1][1] - oy))
    check(e_end < 20.0,
          f"corridor transit ended {e_end:.1f} m off its centreline")


# --------------------------------------------------------------------------
# 4. Snapshot / restore -> bit-exact replay (doc 3.3)
# --------------------------------------------------------------------------


def _snapshot_restore_case(it: IntentType, pt: PlatformType) -> None:
    """Run one intent x platform through the snapshot/restore replay check."""
    world = FakeWorld(2000.0)
    label = f"{pt.name}/{it.name}"
    agent = spawn_agent(pt, x=120.0, y=-80.0, psi=-0.9)
    ownship = spawn_agent(PlatformType.EVTOL_UAM, x=700.0, y=650.0,
                          psi=-2.2, v=55.0)
    ctrl = make_intent(it, agent, world, np.random.default_rng(101))
    rng = np.random.default_rng(202)

    half = N_STEPS // 3
    run_episode(ctrl, agent, rng, n_steps=half, ownship=ownship, label=label)

    # Capture EVERYTHING that determines the future.
    snap = ctrl.snapshot()
    rng_state = copy.deepcopy(rng.bit_generator.state)
    agent_state = agent.state.copy()
    own_state = ownship.state.copy()
    t_mid = half * SIM_DT

    traj_a = run_episode(ctrl, agent, rng, n_steps=N_STEPS - half,
                         ownship=ownship, t0=t_mid, label=label)

    # Restore and replay.
    ctrl.restore(snap)
    rng.bit_generator.state = copy.deepcopy(rng_state)
    agent.state = agent_state.copy()
    ownship.state = own_state.copy()
    traj_b = run_episode(ctrl, agent, rng, n_steps=N_STEPS - half,
                         ownship=ownship, t0=t_mid, label=label)

    check(traj_a == traj_b,
          f"{label}: snapshot->restore->rerun was not bit-exact")

    # A fresh controller restored from the same snapshot must also match: this
    # is what the branch-rollout driver actually does.
    fresh = make_intent(it, spawn_agent(pt), world, np.random.default_rng(999))
    fresh.restore(snap)
    agent.state = agent_state.copy()
    ownship.state = own_state.copy()
    rng.bit_generator.state = copy.deepcopy(rng_state)
    traj_c = run_episode(fresh, agent, rng, n_steps=N_STEPS - half,
                         ownship=ownship, t0=t_mid, label=label)
    check(traj_a == traj_c,
          f"{label}: a fresh controller restored from the snapshot diverged")


def test_snapshot_restore_bit_exact() -> None:
    """snapshot -> restore -> rerun is bit-exact for every intent and platform."""
    for pt in POWERED_PLATFORMS:
        for it in POWERED_INTENTS:
            _snapshot_restore_case(it, pt)
    _snapshot_restore_case(IntentType.PASSIVE_DRIFT, PlatformType.DEBRIS)


def test_snapshot_is_json_safe() -> None:
    """The snapshot contains only JSON-safe types (it goes into AgentSnapshot)."""
    import json

    world = FakeWorld()
    for pt in POWERED_PLATFORMS:
        for it in POWERED_INTENTS:
            agent = spawn_agent(pt)
            ctrl = make_intent(it, agent, world, np.random.default_rng(301))
            run_episode(ctrl, agent, np.random.default_rng(302), n_steps=200,
                        label=f"json/{pt.name}/{it.name}")
            text = json.dumps(ctrl.snapshot())
            back = json.loads(text)
            check(back["intent"] == int(it) or it is IntentType.PASSIVE_DRIFT,
                  "snapshot must record the effective intent")
            ctrl.restore(back)
            check(isinstance(ctrl.describe(), str),
                  "a JSON round-tripped snapshot must still be usable")


def test_construction_is_deterministic() -> None:
    """Same seed + same agent -> identical intent construction."""
    world = FakeWorld()
    for it in POWERED_INTENTS:
        a1 = spawn_agent(PlatformType.FIXED_WING_UAS, x=10.0, y=20.0, psi=0.3)
        a2 = spawn_agent(PlatformType.FIXED_WING_UAS, x=10.0, y=20.0, psi=0.3)
        c1 = make_intent(it, a1, world, np.random.default_rng(7))
        c2 = make_intent(it, a2, world, np.random.default_rng(7))
        check(c1.snapshot() == c2.snapshot(),
              f"{it.name}: construction is not reproducible from the seed")


# --------------------------------------------------------------------------
# 5. The design rule: branch rollouts must actually spread (doc 3.1 / 3.3)
# --------------------------------------------------------------------------


def _branch_spread(it: IntentType, pt: PlatformType, n_branches: int = 24) -> float:
    """Std-dev [m] of the final position over N decision-reseeded rollouts."""
    world = FakeWorld()
    base_agent = spawn_agent(pt)
    ctrl = make_intent(it, base_agent, world, np.random.default_rng(401))
    snap = ctrl.snapshot()
    state0 = base_agent.state.copy()
    xs: list[float] = []
    ys: list[float] = []
    for b in range(n_branches):
        agent = FakeAgent(state0.copy(), base_agent.envelope, pt)
        ctrl.restore(snap)
        run_episode(ctrl, agent, np.random.default_rng(9000 + b),
                    check_limits=False, label=f"branch/{it.name}")
        xs.append(agent.state.x)
        ys.append(agent.state.y)
    ax = np.asarray(xs, dtype=np.float64)
    ay = np.asarray(ys, dtype=np.float64)
    return float(np.sqrt(ax.var() + ay.var()))


def test_multimodality_ordering() -> None:
    """SEARCH_WANDER must spread far more than CORRIDOR_TRANSIT.

    This is the doc 3.1 design rule expressed as a test: if the wander class
    does not branch, a CV extrapolator wins everywhere and the calibration
    story has nothing to measure.
    """
    easy = _branch_spread(IntentType.CORRIDOR_TRANSIT, PlatformType.BIRD)
    hard = _branch_spread(IntentType.SEARCH_WANDER, PlatformType.BIRD)
    check(hard > 100.0,
          f"SEARCH_WANDER branch spread only {hard:.1f} m over 60 s")
    check(hard > 10.0 * max(easy, 1e-6),
          f"SEARCH_WANDER spread {hard:.1f} m is not clearly above "
          f"CORRIDOR_TRANSIT {easy:.1f} m")


def test_wander_beats_cv_extrapolation() -> None:
    """A 15 s CV extrapolation of a wandering bird must miss by a useful margin.

    Doc 8.3: if the CV error is tiny the dataset is too easy. The check here is
    deliberately loose (order-of-magnitude, not a tuned threshold) -- the real
    diagnostic lives in ``analysis/cv_baseline.py``.
    """
    agent = spawn_agent(PlatformType.BIRD)
    ctrl = make_intent(IntentType.SEARCH_WANDER, agent, FakeWorld(),
                       np.random.default_rng(501))
    traj = run_episode(ctrl, agent, np.random.default_rng(502), label="cv")
    horizon = int(round(15.0 / SIM_DT))
    errs: list[float] = []
    for k in range(0, len(traj) - horizon, 20):
        x, y, psi, v = traj[k]
        px = x + v * math.cos(psi) * 15.0
        py = y + v * math.sin(psi) * 15.0
        errs.append(math.hypot(traj[k + horizon][0] - px,
                               traj[k + horizon][1] - py))
    median = float(np.median(np.asarray(errs, dtype=np.float64)))
    check(median > 20.0,
          f"CV 15 s median error only {median:.1f} m -- wander is too weak")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


TESTS = [
    test_cross_track_guidance,
    test_orbit_guidance,
    test_fallback_unpowered_becomes_passive,
    test_fallback_passive_on_powered,
    test_fallback_loiter_variant,
    test_intent_class_table,
    test_all_controllers_run_60s_within_envelope,
    test_passive_drift_returns_none,
    test_hover_variant_slows_to_v_min,
    test_orbit_variant_holds_its_circle,
    test_delivery_captures_waypoints,
    test_search_wander_reorients,
    test_reactive_avoid_turns_away,
    test_corridor_stays_near_centreline,
    test_snapshot_restore_bit_exact,
    test_snapshot_is_json_safe,
    test_construction_is_deterministic,
    test_multimodality_ordering,
    test_wander_beats_cv_extrapolation,
]


def main() -> int:
    """Run every test, printing one line per test. Returns a process exit code."""
    failures = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected error
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed, "
          f"{CHECKS} explicit checks")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
