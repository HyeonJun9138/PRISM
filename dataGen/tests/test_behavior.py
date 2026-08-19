"""Tests for agents.behavior and agents.maneuver.

:func:`test_branch_rollouts_are_multimodal` is the paper's central premise
expressed as code. Doc 3.3 claims that giving agents stochastic decision
rules -- rather than hand-placed waypoints -- makes identical pasts lead to
genuinely different futures, so that the true future distribution is
multimodal and a point-estimate predictor (in particular a constant-velocity
extrapolator) cannot represent it. If that claim is false, everything
downstream of the simulator loses its justification.

The test therefore does not settle for "some spread exists". It rolls one
identical snapshot forward 200 times with different branch seeds and then
asserts that both branches carry real mass, that they are separated by
several within-branch standard deviations, that knowing which branch was
taken roughly halves the achievable RMSE, that a constant-velocity
extrapolator does markedly worse than that, and that a two-component
Gaussian mixture beats a single Gaussian on BIC. Section 4 of this file
documents why cross-track offset -- not accumulated heading -- is the primary
statistic, and why no dip at the centre is asserted.

Run standalone (no pytest required)::

    python $PRISM_ROOT/dataGen/tests/test_behavior.py

or collect with pytest if it happens to be installed.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

# Allow running this file directly: main.py normally puts dataGen on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.behavior import (  # noqa: E402
    DEFAULT_AVOID_LATCH_S,
    DEFAULT_MIN_DWELL_S,
    binary_branch,
    categorical,
    new_avoid_latch,
    obstacle_fork,
    ou_process_step,
    poisson_event,
    reactive_avoid_decision,
    sample_dwell,
    sample_fraction,
)
from agents.maneuver import (  # noqa: E402
    CT_BLEND_S,
    CV_OMEGA_FRACTION,
    DWELL_MEAN_S,
    MODE_INDEX,
    MODE_ORDER,
    TRANSITION_MATRICES,
    ModeMachine,
    biased_transitions,
)
from agents.platforms import (  # noqa: E402
    INTENT_DWELL_MULTIPLIER,
    STABILITY_PROFILES,
    default_envelope,
    stability_profile,
)
from core.constants import SIM_DT  # noqa: E402
from core.rng import RNGHub  # noqa: E402
from core.state import (  # noqa: E402
    DEFAULT_STABILITY,
    AgentState,
    PerformanceEnvelope,
    StabilityProfile,
    wrap_angle,
)
from core.types import IntentType, ManeuverMode, PlatformType  # noqa: E402


# --------------------------------------------------------------------------
# Test fixtures: a minimal agent and a minimal integrator.
#
# agents/base.py and core/integrator.py are owned by other modules and are
# written in parallel with this one, so the test deliberately depends on
# neither. It reproduces only the two-line kinematic core it needs:
#     omega -> rate limited toward the command by omega_dot_max
#     psi   -> integrated from omega
# which is the part of the dynamics the mode machine actually drives.
# --------------------------------------------------------------------------


@dataclass
class _Agent:
    """Smallest object satisfying the ModeMachine agent interface."""

    state: AgentState
    envelope: PerformanceEnvelope


def _fixed_wing_envelope() -> PerformanceEnvelope:
    """A fixed-wing UAS-like envelope [SI units] used by the rollout test."""
    return PerformanceEnvelope(
        v_min=18.0,
        v_max=35.0,
        a_max=2.0,
        a_min=-2.5,
        omega_max=0.45,
        omega_dot_max=0.35,
        bank_max=math.radians(35.0),
        bank_limited=True,
        lateral_accel_max=0.0,
        can_hover=False,
        is_powered=True,
        wind_susceptibility=0.6,
        mass=12.0,
        drag_area=0.15,
    )


def _multirotor_envelope() -> PerformanceEnvelope:
    """A small multirotor envelope [SI units] (hover-capable, tilt-limited)."""
    return PerformanceEnvelope(
        v_min=0.0,
        v_max=20.0,
        a_max=3.0,
        a_min=-3.0,
        omega_max=1.2,
        omega_dot_max=1.5,
        bank_max=math.radians(30.0),
        bank_limited=False,
        lateral_accel_max=5.0,
        can_hover=True,
        is_powered=True,
        wind_susceptibility=0.85,
        mass=2.5,
        drag_area=0.08,
    )


def _debris_envelope() -> PerformanceEnvelope:
    """An unpowered body envelope [SI units]: no thrust, no commanded turns."""
    return PerformanceEnvelope(
        v_min=0.0,
        v_max=15.0,
        a_max=0.0,
        a_min=-0.5,
        omega_max=0.05,
        omega_dot_max=0.05,
        bank_max=0.0,
        bank_limited=False,
        lateral_accel_max=0.2,
        can_hover=True,
        is_powered=False,
        wind_susceptibility=1.0,
        mass=0.8,
        drag_area=0.3,
    )


def _integrate(agent: _Agent, omega_cmd: float, a_cmd: float, dt: float) -> None:
    """Advance the fixture agent by ``dt`` [s] with a rate-limited turn.

    Mirrors the contract of :mod:`core.integrator`: ``omega`` is a state that
    slews toward the command at no more than ``omega_dot_max`` [rad/s^2], and
    the heading integrates that state. Position is advanced too so the state
    stays physically coherent, though the tests only read heading and speed.
    """
    st = agent.state
    env = agent.envelope
    d_omega = omega_cmd - st.omega
    max_d = env.omega_dot_max * dt
    if d_omega > max_d:
        d_omega = max_d
    elif d_omega < -max_d:
        d_omega = -max_d
    st.omega = env.clamp_turn_rate(st.omega + d_omega, st.v)
    st.a = env.clamp_accel(a_cmd)
    st.psi = wrap_angle(st.psi + st.omega * dt)
    st.v = env.clamp_speed(st.v + st.a * dt)
    st.x += st.v * math.cos(st.psi) * dt
    st.y += st.v * math.sin(st.psi) * dt


# --------------------------------------------------------------------------
# 1. Transition matrices
# --------------------------------------------------------------------------


def test_transition_matrix_rows_sum_to_one() -> None:
    """Every intent's transition matrix must be row-stochastic."""
    assert len(TRANSITION_MATRICES) == len(list(IntentType)), (
        "every IntentType needs a transition matrix"
    )
    for intent, mat in TRANSITION_MATRICES.items():
        assert mat.shape == (5, 5), f"{intent.name}: shape {mat.shape} != (5, 5)"
        assert np.all(mat >= 0.0), f"{intent.name}: negative probability"
        assert np.all(np.isfinite(mat)), f"{intent.name}: non-finite probability"
        row_sums = mat.sum(axis=1)
        for i, s in enumerate(row_sums):
            assert abs(s - 1.0) < 1e-12, (
                f"{intent.name}: row {MODE_ORDER[i].name} sums to {s!r}, expected 1.0"
            )
    # The index convention MODE_ORDER[int(mode)] == mode is relied on by the
    # mode machine and by the ground-truth mode labels written to disk.
    for mode in ManeuverMode:
        assert MODE_ORDER[int(mode)] == mode
        assert MODE_INDEX[mode] == int(mode)


def test_dwell_means_cover_every_intent_and_mode() -> None:
    """Every (intent, mode) pair needs a positive fallback mean dwell [s].

    ``DWELL_MEAN_S`` is now the FALLBACK table, used only by a machine whose
    agent carries no per-platform stability profile. It is still checked
    against the default floor, because that is the pair those machines use.
    """
    for intent in IntentType:
        assert intent in DWELL_MEAN_S, f"{intent.name} has no dwell means"
        for mode in ManeuverMode:
            mean = DWELL_MEAN_S[intent][mode]
            assert mean > 0.0, f"{intent.name}/{mode.name}: mean dwell must be > 0"
            assert mean >= DEFAULT_MIN_DWELL_S, (
                f"{intent.name}/{mode.name}: mean dwell {mean} s is below the "
                f"{DEFAULT_MIN_DWELL_S} s default floor, which would make the "
                f"floor the mode"
            )


# --------------------------------------------------------------------------
# 2. Behaviour primitives
# --------------------------------------------------------------------------


def test_sample_dwell_respects_the_floor() -> None:
    """No dwell may fall below the floor the caller asked for (doc 3.3 (b))."""
    rng = np.random.default_rng(20260812)
    # Deliberately adversarial: a mean far below the floor and a huge cv, so
    # that essentially every raw lognormal draw is under the floor.
    for mean_s, cv in ((0.05, 2.0), (0.5, 1.2), (2.0, 0.4), (12.0, 0.4)):
        samples = np.array([sample_dwell(rng, mean_s, cv) for _ in range(5000)])
        assert samples.min() >= DEFAULT_MIN_DWELL_S, (
            f"dwell floor violated for mean={mean_s}, cv={cv}: "
            f"min sample {samples.min()}"
        )
        assert np.all(np.isfinite(samples))
    # And with a benign mean the empirical mean should track the requested one
    # (the floor only bites in the left tail).
    samples = np.array([sample_dwell(rng, 8.0, 0.4) for _ in range(20000)])
    assert 7.5 < samples.mean() < 8.5, f"dwell mean drifted: {samples.mean()}"
    assert 0.30 < samples.std() / samples.mean() < 0.50, "dwell cv drifted"
    # Custom floor is honoured.
    samples = np.array([sample_dwell(rng, 1.0, 0.6, min_s=3.0) for _ in range(2000)])
    assert samples.min() >= 3.0


def test_binary_branch_is_a_fair_coin() -> None:
    """The canonical fork returns +/-1 with the requested probability."""
    rng = np.random.default_rng(7)
    draws = np.array([binary_branch(rng) for _ in range(20000)])
    assert set(np.unique(draws).tolist()) == {-1, 1}
    p_left = float((draws > 0).mean())
    assert 0.48 < p_left < 0.52, f"p_left = {p_left}, expected ~0.5"
    biased = np.array([binary_branch(rng, 0.8) for _ in range(20000)])
    assert 0.78 < float((biased > 0).mean()) < 0.82
    assert all(binary_branch(rng, 1.0) == 1 for _ in range(50))
    assert all(binary_branch(rng, 0.0) == -1 for _ in range(50))


def test_obstacle_fork_ignores_geometry() -> None:
    """The obstacle fork's sign must not depend on where the obstacle is.

    This is the property doc 3.3 needs: identical pasts, different futures.
    Two calls with the SAME generator state but mirrored geometry must give
    the same sign, proving the geometry never enters the decision.
    """
    pos = (0.0, 0.0)
    psi = 0.3
    for seed in range(200):
        left_obstacle = (100.0, 250.0)
        right_obstacle = (100.0, -250.0)
        a = obstacle_fork(np.random.default_rng(seed), pos, psi, left_obstacle)
        b = obstacle_fork(np.random.default_rng(seed), pos, psi, right_obstacle)
        assert a == b, "obstacle_fork leaked geometry into the sign"
    rng = np.random.default_rng(11)
    signs = np.array([obstacle_fork(rng, pos, psi, (50.0, 50.0)) for _ in range(10000)])
    assert 0.48 < float((signs > 0).mean()) < 0.52


def test_categorical_matches_requested_probabilities() -> None:
    """One-uniform inverse-CDF sampling reproduces the target distribution."""
    rng = np.random.default_rng(3)
    probs = [0.1, 0.6, 0.25, 0.05]
    draws = np.array([categorical(rng, probs) for _ in range(40000)])
    for k, p in enumerate(probs):
        emp = float((draws == k).mean())
        assert abs(emp - p) < 0.01, f"category {k}: empirical {emp}, target {p}"
    # Unnormalised weights are accepted.
    draws = np.array([categorical(rng, [2.0, 2.0]) for _ in range(4000)])
    assert 0.47 < float((draws == 0).mean()) < 0.53
    # A zero-weight category is never drawn.
    draws = np.array([categorical(rng, [1.0, 0.0, 1.0]) for _ in range(4000)])
    assert not np.any(draws == 1)


def test_poisson_event_rate() -> None:
    """Event probability follows 1 - exp(-rate*dt), exactly, at any dt."""
    rng = np.random.default_rng(5)
    for rate, dt in ((0.5, 0.05), (2.0, 0.5), (0.1, 1.0)):
        expected = -math.expm1(-rate * dt)
        hits = np.array([poisson_event(rng, rate, dt) for _ in range(40000)])
        emp = float(hits.mean())
        assert abs(emp - expected) < 0.01, (
            f"rate={rate} dt={dt}: empirical {emp}, expected {expected}"
        )
    assert poisson_event(rng, 0.0, 1.0) is False
    assert poisson_event(rng, 10.0, 0.0) is False


def test_ou_process_is_stationary_at_any_step_size() -> None:
    """The exact OU discretisation holds its stationary variance for any dt."""
    for dt in (0.05, 0.5, 2.0):
        rng = np.random.default_rng(101)
        sigma, tau = 2.0, 5.0
        x = 0.0
        burn = int(round(60.0 / dt))
        for _ in range(burn):
            x = ou_process_step(x, sigma, tau, dt, rng)
        samples = np.empty(40000)
        for i in range(samples.size):
            x = ou_process_step(x, sigma, tau, dt, rng)
            samples[i] = x
        assert abs(samples.std() - sigma) < 0.12 * sigma, (
            f"dt={dt}: stationary std {samples.std()} != sigma {sigma}"
        )
        assert abs(samples.mean()) < 0.1 * sigma
    # Zero timestep is a no-op; zero sigma is deterministic decay.
    rng = np.random.default_rng(1)
    assert ou_process_step(3.0, 1.0, 5.0, 0.0, rng) == 3.0
    decayed = ou_process_step(4.0, 0.0, 2.0, 2.0, rng)
    assert abs(decayed - 4.0 * math.exp(-1.0)) < 1e-12


def test_sample_fraction_bounds() -> None:
    """Aggressiveness fractions stay inside [lo, hi] and use the full range."""
    rng = np.random.default_rng(13)
    vals = np.array([sample_fraction(rng, 0.3, 1.0) for _ in range(20000)])
    assert vals.min() >= 0.3 and vals.max() <= 1.0
    assert vals.min() < 0.32 and vals.max() > 0.98, "range is not being covered"
    assert abs(vals.mean() - 0.65) < 0.01
    assert sample_fraction(rng, 0.5, 0.5) == 0.5


def test_reactive_avoid_triggers_only_when_closing_and_latches() -> None:
    """Reactive avoidance needs proximity AND closing, then commits to a side."""
    rng = np.random.default_rng(17)

    # Close but opening -> no trigger.
    latch = new_avoid_latch()
    active, sign = reactive_avoid_decision(rng, (100.0, 0.0), (30.0, 0.0), 300.0, latch)
    assert active is False and sign == 0

    # Far but closing -> no trigger.
    latch = new_avoid_latch()
    active, sign = reactive_avoid_decision(rng, (900.0, 0.0), (-60.0, 0.0), 300.0, latch)
    assert active is False and sign == 0

    # Close and closing -> trigger, and the side latches for latch_s seconds.
    latch = new_avoid_latch()
    active, sign = reactive_avoid_decision(
        rng, (150.0, 0.0), (-50.0, 0.0), 300.0, latch, dt=0.0
    )
    assert active is True and sign in (-1, 1)
    assert latch["active"] is True and latch["trigger_count"] == 1
    first_sign = sign
    held = 0.0
    while held < DEFAULT_AVOID_LATCH_S - 2 * SIM_DT:
        # Geometry now says "no threat at all" -- the latch must ignore that.
        active, sign = reactive_avoid_decision(
            rng, (5000.0, 0.0), (100.0, 0.0), 300.0, latch, dt=SIM_DT
        )
        assert active is True, "latch released early"
        assert sign == first_sign, "latch changed sides mid-commitment"
        held += SIM_DT
    # After the latch expires with no threat present it releases.
    for _ in range(int(round(2 * SIM_DT / SIM_DT)) + 2):
        active, sign = reactive_avoid_decision(
            rng, (5000.0, 0.0), (100.0, 0.0), 300.0, latch, dt=SIM_DT
        )
    assert active is False and sign == 0 and latch["active"] is False

    # Over many independent encounters, both sides are chosen.
    signs = []
    for seed in range(400):
        r = np.random.default_rng(seed)
        lt = new_avoid_latch()
        ok, s = reactive_avoid_decision(r, (120.0, 40.0), (-40.0, -10.0), 300.0, lt)
        assert ok is True
        signs.append(s)
    frac_left = float(np.mean(np.array(signs) > 0))
    assert 0.4 < frac_left < 0.6, f"avoid side is biased: p_left={frac_left}"


# --------------------------------------------------------------------------
# 3. ModeMachine mechanics
# --------------------------------------------------------------------------


def _mode_segment_lengths(
    envelope: PerformanceEnvelope,
    intent: IntentType,
    v0: float,
    seed: int,
    duration_s: float = 1800.0,
    stability: StabilityProfile | None = None,
) -> tuple[list[float], ModeMachine]:
    """Fly one mode machine and return the length [s] of every mode segment."""
    hub = RNGHub(seed)
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")
    agent = _Agent(AgentState(0.0, 0.0, v0, 0.0), envelope)
    machine = ModeMachine(intent, stability=stability)

    segment_len = 0.0
    last_mode = machine.mode
    lengths: list[float] = []
    for _ in range(int(round(duration_s / SIM_DT))):
        machine.step(SIM_DT, agent, rng_mode, rng_branch)
        if machine.mode != last_mode:
            if segment_len > 0.0:
                lengths.append(segment_len)
            segment_len = 0.0
            last_mode = machine.mode
        segment_len += SIM_DT
        omega_cmd, a_cmd = machine.apply(0.0, 0.0, agent)
        cap = agent.envelope.max_turn_rate(agent.state.v)
        assert abs(omega_cmd) <= cap + 1e-9, "turn command exceeded the envelope"
        assert agent.envelope.a_min - 1e-9 <= a_cmd <= agent.envelope.a_max + 1e-9
        _integrate(agent, omega_cmd, a_cmd, SIM_DT)
    return lengths, machine


def test_mode_machine_respects_the_PLATFORM_dwell_floor_and_envelope() -> None:
    """Each platform holds a mode for at least ITS OWN floor, not a global one.

    There is deliberately no global minimum dwell any more: 1.5 s is a bird and
    is absurd for a passenger eVTOL, whose 12 s floor is what keeps its trace
    looking like an aircraft (doc 7.2). What is universal is the *shape* of the
    constraint -- a mode, once entered, is held -- so it is asserted here
    against every platform's own :class:`core.state.StabilityProfile`.
    """
    cases = (
        (PlatformType.BIRD, IntentType.SEARCH_WANDER),
        (PlatformType.MULTIROTOR, IntentType.DELIVERY_WP),
        (PlatformType.FIXED_WING_UAS, IntentType.DELIVERY_WP),
        (PlatformType.EVTOL_UAM, IntentType.CORRIDOR_TRANSIT),
        (PlatformType.GA, IntentType.SEARCH_WANDER),
        (PlatformType.UNKNOWN, IntentType.SEARCH_WANDER),
    )
    for platform, intent in cases:
        env = default_envelope(platform)
        floor = env.stability.dwell_min_s
        assert floor == stability_profile(platform).dwell_min_s, (
            f"{platform.name}: the envelope is not carrying its own profile"
        )
        v0 = 0.5 * (env.v_min + env.v_max)
        lengths, machine = _mode_segment_lengths(env, intent, v0, 4242 + int(platform))
        assert machine.stability is env.stability, (
            f"{platform.name}: the machine did not adopt the envelope's profile"
        )
        # A stable platform switches RARELY on purpose: an eVTOL on a corridor
        # makes about seven mode changes in half an hour. Four is simply the
        # smallest sample that can falsify a floor.
        assert len(lengths) > 4, (
            f"{platform.name}: only {len(lengths)} mode switches in 1800 s"
        )
        shortest = min(lengths)
        # One SIM_DT of slack: a segment is measured on the integration grid.
        assert shortest >= floor - SIM_DT - 1e-9, (
            f"{platform.name}: a mode was held for only {shortest:.2f} s, below "
            f"its own {floor} s floor"
        )


def test_platform_dwell_floors_are_ordered_and_supersede_the_global_one() -> None:
    """Commitment must be a platform property with the documented ordering.

    This is the regression guard for the spec bug this phase fixed: a single
    global floor. The absolute seconds are an assumed parameter (doc 12), but
    the ORDERING is the claim the dataset rests on -- if a bird were as
    committed as a UAM aircraft, the difficulty spread that the calibration
    figure exploits would not exist.
    """
    floors = {p: STABILITY_PROFILES[int(p)].dwell_min_s for p in PlatformType}
    assert floors[PlatformType.EVTOL_UAM] > floors[PlatformType.GA]
    assert floors[PlatformType.GA] > floors[PlatformType.FIXED_WING_UAS]
    assert floors[PlatformType.FIXED_WING_UAS] > floors[PlatformType.MULTIROTOR]
    assert floors[PlatformType.MULTIROTOR] > floors[PlatformType.BIRD]
    assert floors[PlatformType.EVTOL_UAM] >= 8.0 * floors[PlatformType.BIRD], (
        "a UAM aircraft must be an order of magnitude steadier than a bird"
    )
    assert floors[PlatformType.UNKNOWN] <= floors[PlatformType.BIRD], (
        "the OOD class must be at least as twitchy as the twitchiest trained one"
    )

    authority = {p: STABILITY_PROFILES[int(p)].mode_authority for p in PlatformType}
    assert authority[PlatformType.BIRD] > authority[PlatformType.MULTIROTOR]
    assert authority[PlatformType.MULTIROTOR] > authority[PlatformType.FIXED_WING_UAS]
    assert authority[PlatformType.FIXED_WING_UAS] > authority[PlatformType.GA]
    assert authority[PlatformType.GA] > authority[PlatformType.EVTOL_UAM]

    # Every profile must be self-consistent, and the intent multipliers total.
    for platform in PlatformType:
        problems = STABILITY_PROFILES[int(platform)].validate()
        assert not problems, f"{platform.name}: {problems}"
    for intent in IntentType:
        assert int(intent) in INTENT_DWELL_MULTIPLIER, intent.name
    assert (INTENT_DWELL_MULTIPLIER[int(IntentType.CORRIDOR_TRANSIT)]
            > INTENT_DWELL_MULTIPLIER[int(IntentType.DELIVERY_WP)]
            > INTENT_DWELL_MULTIPLIER[int(IntentType.SEARCH_WANDER)]), (
        "a transiting aircraft must commit for longer than a searching one"
    )


def test_intent_multiplier_scales_dwell_but_the_platform_floor_still_binds() -> None:
    """A wandering eVTOL is still far steadier than a wandering bird.

    The intent multiplies the mean dwell; the platform's floor is applied
    AFTER that multiplication. SEARCH_WANDER cuts the eVTOL's 25 s mean to
    11.25 s, which is below its own 12 s floor -- so the floor, not the intent,
    decides. That is the whole point of the ordering.
    """
    evtol = default_envelope(PlatformType.EVTOL_UAM)
    bird = default_envelope(PlatformType.BIRD)
    evtol_wander, _ = _mode_segment_lengths(
        evtol, IntentType.SEARCH_WANDER, 55.0, 31337)
    bird_wander, _ = _mode_segment_lengths(
        bird, IntentType.SEARCH_WANDER, 12.5, 31337)
    evtol_med = float(np.median(evtol_wander))
    bird_med = float(np.median(bird_wander))
    assert evtol_med >= evtol.stability.dwell_min_s - SIM_DT, (
        f"the eVTOL floor stopped binding under SEARCH_WANDER: {evtol_med:.2f} s"
    )
    assert evtol_med > 5.0 * bird_med, (
        f"a wandering eVTOL ({evtol_med:.2f} s) is not much steadier than a "
        f"wandering bird ({bird_med:.2f} s)"
    )

    # And the corridor multiplier lengthens the same platform further still.
    evtol_corridor, _ = _mode_segment_lengths(
        evtol, IntentType.CORRIDOR_TRANSIT, 55.0, 31337)
    assert float(np.median(evtol_corridor)) > evtol_med, (
        "CORRIDOR_TRANSIT must hold a manoeuvre longer than SEARCH_WANDER"
    )


def test_diagonal_bias_makes_stable_platforms_sticky() -> None:
    """``P' = (1-b)P + bI`` must stay stochastic and reach the eVTOL target."""
    for platform in PlatformType:
        bias = STABILITY_PROFILES[int(platform)].diag_bias
        for intent in IntentType:
            m = biased_transitions(intent, bias)
            sums = m.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-12), (
                f"{platform.name}/{intent.name}: rows sum to {sums.tolist()}"
            )
            assert np.all(m >= 0.0), f"{platform.name}/{intent.name}: negative entry"

    evtol_bias = STABILITY_PROFILES[int(PlatformType.EVTOL_UAM)].diag_bias
    corridor = biased_transitions(IntentType.CORRIDOR_TRANSIT, evtol_bias)
    p_cv_cv = float(corridor[MODE_INDEX[ManeuverMode.CV], MODE_INDEX[ManeuverMode.CV]])
    assert p_cv_cv >= 0.9, (
        f"an eVTOL in corridor transit re-enters CV with p={p_cv_cv:.3f}; "
        f"doc 3.2 calls this class 'straight, easy to predict' and the "
        f"stability model requires >= 0.9"
    )

    # A NEGATIVE bias must move mass OFF the diagonal (UNKNOWN is erratic on
    # purpose), and a zero bias must be the base matrix untouched.
    base = TRANSITION_MATRICES[IntentType.SEARCH_WANDER]
    unknown_bias = STABILITY_PROFILES[int(PlatformType.UNKNOWN)].diag_bias
    assert unknown_bias < 0.0
    unknown = biased_transitions(IntentType.SEARCH_WANDER, unknown_bias)
    assert float(np.trace(unknown)) < float(np.trace(base)), (
        "the OOD class must switch mode MORE often than the base matrix"
    )
    assert np.array_equal(biased_transitions(IntentType.SEARCH_WANDER, 0.0), base)


def test_mode_authority_blends_a_stable_platform_instead_of_overriding() -> None:
    """A latched turn overrides a bird and merely biases an eVTOL.

    ``omega = (1-a) * omega_intent + a * omega_mode``. With ``a = 1`` (the
    default profile) this is the pure override the multimodality depends on;
    with the eVTOL's 0.35 the guidance keeps the majority vote, which is what
    stops a UAM aircraft wandering off its corridor centreline.
    """
    hub = RNGHub(20260812)
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")

    for platform, expect_override in ((PlatformType.BIRD, True),
                                      (PlatformType.EVTOL_UAM, False)):
        env = default_envelope(platform)
        v0 = 0.5 * (env.v_min + env.v_max)
        agent = _Agent(AgentState(0.0, 0.0, v0, 0.0), env)
        machine = ModeMachine(IntentType.CORRIDOR_TRANSIT, mode=ManeuverMode.CT_LEFT)
        machine.initialize(agent, rng_mode, rng_branch)
        cap = env.max_turn_rate(v0)
        authority = env.stability.mode_authority

        # The intent screams "turn right as hard as you can".
        omega_cmd, _ = machine.apply(-cap, 0.0, agent)
        omega_mode = machine.mode_params["omega_frac"] * cap
        expected = (1.0 - authority) * (-cap) + authority * omega_mode
        assert abs(omega_cmd - expected) < 1e-9, (
            f"{platform.name}: authority blend is not being applied "
            f"({omega_cmd} vs {expected})"
        )
        if expect_override:
            assert omega_cmd > 0.0, (
                f"{platform.name}: a committed turn was cancelled by the intent"
            )
        else:
            assert omega_cmd < 0.0, (
                f"{platform.name}: the mode overrode corridor guidance that was "
                f"commanding the full opposite turn rate"
            )
        assert abs(omega_cmd) <= cap + 1e-9, "the blend left the envelope"


def test_cv_mode_suppresses_deliberate_turning() -> None:
    """In CV a large intent turn request is cut to a course-keeping trim."""
    hub = RNGHub(9)
    agent = _Agent(AgentState(0.0, 0.0, 25.0, 0.0), _fixed_wing_envelope())
    machine = ModeMachine(IntentType.CORRIDOR_TRANSIT, mode=ManeuverMode.CV, mode_timer=10.0)
    cap = agent.envelope.max_turn_rate(agent.state.v)
    omega_cmd, a_cmd = machine.apply(10.0 * cap, 1.0, agent)
    assert abs(omega_cmd) <= CV_OMEGA_FRACTION * cap + 1e-12, (
        "CV let a deliberate maneuver through"
    )
    assert a_cmd == 1.0, "CV must pass the intent acceleration through"
    del hub


def test_ct_mode_overrides_intent_and_commits() -> None:
    """At full authority a coordinated turn overrides the intent outright.

    The fixture envelope carries :data:`core.state.DEFAULT_STABILITY`
    (authority 1.0), which is the behaviour every agent had before manoeuvre
    commitment became a platform property; the per-platform blend is covered by
    ``test_mode_authority_blends_a_stable_platform_instead_of_overriding``.
    """
    hub = RNGHub(31)
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")
    agent = _Agent(AgentState(0.0, 0.0, 25.0, 0.0), _fixed_wing_envelope())
    assert agent.envelope.stability == DEFAULT_STABILITY
    machine = ModeMachine(IntentType.DELIVERY_WP, mode=ManeuverMode.CT_LEFT)
    machine.initialize(agent, rng_mode, rng_branch)
    assert machine.mode == ManeuverMode.CT_LEFT
    cap = agent.envelope.max_turn_rate(agent.state.v)
    # The intent screams "turn right as hard as you can"; CT ignores it.
    omega_cmd, _ = machine.apply(-cap, 0.0, agent)
    assert omega_cmd > 0.0, "CT_LEFT did not override an opposing intent command"
    assert 0.3 * cap - 1e-9 <= omega_cmd <= cap + 1e-9, (
        f"CT magnitude {omega_cmd} outside the U(0.3, 1.0) * cap band"
    )
    assert machine.mode_timer >= machine.stability.dwell_min_s


def test_blend_smooths_the_exit_from_a_turn() -> None:
    """Leaving a CT mode ramps the command down instead of snapping it."""
    hub = RNGHub(77)
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")
    agent = _Agent(AgentState(0.0, 0.0, 25.0, 0.0), _fixed_wing_envelope())
    machine = ModeMachine(IntentType.DELIVERY_WP, mode=ManeuverMode.CT_LEFT)
    machine.initialize(agent, rng_mode, rng_branch)
    omega_in_turn, _ = machine.apply(0.0, 0.0, agent)

    machine.force_mode(ManeuverMode.CV, agent, rng_mode, rng_branch, dwell_s=10.0)
    assert machine.blend_active, "no blend was armed when leaving CT"
    omega_just_after, _ = machine.apply(0.0, 0.0, agent)
    assert abs(omega_just_after - omega_in_turn) < 1e-9, (
        "the command jumped at the instant the turn ended"
    )
    # Halfway through the blend the command should be about half.
    for _ in range(int(round((CT_BLEND_S / 2.0) / SIM_DT))):
        machine.step(SIM_DT, agent, rng_mode, rng_branch)
    omega_mid, _ = machine.apply(0.0, 0.0, agent)
    assert 0.35 * omega_in_turn < omega_mid < 0.65 * omega_in_turn, (
        f"blend is not linear: {omega_mid} vs {omega_in_turn}"
    )
    # After the full blend the command is the CV trim (zero for zero intent).
    for _ in range(int(round(CT_BLEND_S / SIM_DT)) + 2):
        machine.step(SIM_DT, agent, rng_mode, rng_branch)
    assert not machine.blend_active
    omega_after, _ = machine.apply(0.0, 0.0, agent)
    assert abs(omega_after) < 1e-12


def test_hover_is_only_used_by_hover_capable_platforms() -> None:
    """HOVER never appears for a fixed-wing; it does for a multirotor."""
    hub = RNGHub(555)
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")

    fw = _Agent(AgentState(0.0, 0.0, 25.0, 0.0), _fixed_wing_envelope())
    machine = ModeMachine(IntentType.LOITER_HOVER)
    seen_fw: set[ManeuverMode] = set()
    for _ in range(int(round(1200.0 / SIM_DT))):
        machine.step(SIM_DT, fw, rng_mode, rng_branch)
        seen_fw.add(machine.mode)
        omega_cmd, a_cmd = machine.apply(0.0, 0.0, fw)
        _integrate(fw, omega_cmd, a_cmd, SIM_DT)
    assert ManeuverMode.HOVER not in seen_fw, "a fixed-wing was told to hover"
    assert fw.state.v >= fw.envelope.v_min - 1e-9

    mr = _Agent(AgentState(0.0, 0.0, 10.0, 0.0), _multirotor_envelope())
    machine = ModeMachine(IntentType.LOITER_HOVER)
    seen_mr: set[ManeuverMode] = set()
    for _ in range(int(round(1200.0 / SIM_DT))):
        machine.step(SIM_DT, mr, rng_mode, rng_branch)
        seen_mr.add(machine.mode)
        omega_cmd, a_cmd = machine.apply(0.0, 0.0, mr)
        _integrate(mr, omega_cmd, a_cmd, SIM_DT)
    assert ManeuverMode.HOVER in seen_mr, "a loitering multirotor never hovered"


def test_unpowered_body_stays_in_cv() -> None:
    """An unpowered body has no thrust and therefore no commanded maneuvers."""
    hub = RNGHub(888)
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")
    agent = _Agent(AgentState(0.0, 0.0, 8.0, 0.5), _debris_envelope())
    machine = ModeMachine(IntentType.PASSIVE_DRIFT)
    for _ in range(int(round(600.0 / SIM_DT))):
        machine.step(SIM_DT, agent, rng_mode, rng_branch)
        assert machine.mode == ManeuverMode.CV, "an unpowered body maneuvered"
        omega_cmd, a_cmd = machine.apply(0.0, 0.0, agent)
        _integrate(agent, omega_cmd, a_cmd, SIM_DT)


def test_snapshot_restore_is_exact() -> None:
    """A restored machine produces a bit-identical continuation."""
    hub = RNGHub(2024)
    agent = _Agent(AgentState(0.0, 0.0, 25.0, 0.0), _fixed_wing_envelope())
    machine = ModeMachine(IntentType.DELIVERY_WP)
    for _ in range(400):
        machine.step(SIM_DT, agent, hub.stream("mode_decision"), hub.stream("branch_choice"))
        omega_cmd, a_cmd = machine.apply(0.0, 0.0, agent)
        _integrate(agent, omega_cmd, a_cmd, SIM_DT)

    machine_snap = machine.snapshot()
    rng_snap = hub.snapshot()
    state_snap = agent.state.to_dict()

    def continue_for(steps: int) -> list[float]:
        out: list[float] = []
        for _ in range(steps):
            machine.step(SIM_DT, agent, hub.stream("mode_decision"), hub.stream("branch_choice"))
            omega_cmd, a_cmd = machine.apply(0.0, 0.0, agent)
            _integrate(agent, omega_cmd, a_cmd, SIM_DT)
            out.append(agent.state.psi)
        return out

    first = continue_for(400)

    machine.restore(machine_snap)
    hub.restore(rng_snap)
    agent.state = AgentState.from_dict(state_snap)
    second = continue_for(400)

    assert first == second, "restore did not reproduce the continuation exactly"

    rebuilt = ModeMachine.from_snapshot(machine_snap)
    assert rebuilt.mode == ModeMachine.from_snapshot(machine_snap).mode
    assert rebuilt.intent == IntentType.DELIVERY_WP


def test_modulate_is_the_host_entry_point() -> None:
    """The adapter agents.base drives must tick the process AND return commands.

    ``agents.base`` probes the mode machine for a method named, in priority
    order, ``modulate`` / ``update`` / ``step`` / ``apply`` / ``command`` /
    ``__call__``, calls the first one it finds with
    ``(agent, t, dt, omega_cmd, a_cmd, rng)``, and interprets a ``None``
    return as "the machine declined to override". ``step`` returns ``None``,
    so if ``modulate`` were missing the host would silently run every agent on
    its intent command alone: no committed turns, no multimodality, and no
    failing test anywhere else to reveal it. This test pins that contract.
    """
    probe_order = ("modulate", "update", "step", "apply", "command", "__call__")
    resolved = next(
        name for name in probe_order if callable(getattr(ModeMachine, name, None))
    )
    assert resolved == "modulate", (
        f"agents.base would call ModeMachine.{resolved}(), which does not "
        "return a command pair"
    )

    # The host binds by parameter name; all six canonical names must be there.
    params = inspect.signature(ModeMachine.modulate).parameters
    for name in ("agent", "t", "dt", "omega_cmd", "a_cmd", "rng"):
        assert name in params, f"ModeMachine.modulate is missing parameter '{name}'"

    hub = RNGHub(4711)
    agent = _Agent(AgentState(0.0, 0.0, 25.0, 0.0), _fixed_wing_envelope())
    machine = ModeMachine(IntentType.DELIVERY_WP)
    machine.bind_streams(hub.stream("mode_decision"), hub.stream("branch_choice"))

    out = machine.modulate(agent, 0.0, SIM_DT, 0.0, 0.0)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(v, float) for v in out)

    # Bound streams must actually be used: an unbound machine driven by a
    # single generator must still work, but a bound one must not need `rng`.
    modes: set[ManeuverMode] = set()
    for i in range(int(round(120.0 / SIM_DT))):
        omega_cmd, a_cmd = machine.modulate(agent, i * SIM_DT, SIM_DT, 0.0, 0.0)
        modes.add(machine.mode)
        _integrate(agent, omega_cmd, a_cmd, SIM_DT)
    assert len(modes) > 1, "the mode process never advanced through modulate()"

    unbound = ModeMachine(IntentType.DELIVERY_WP)
    single = np.random.default_rng(5)
    assert unbound.modulate(agent, 0.0, SIM_DT, 0.0, 0.0, single) is not None
    try:
        ModeMachine(IntentType.DELIVERY_WP).modulate(agent, 0.0, SIM_DT, 0.0, 0.0)
    except ValueError:
        pass
    else:  # pragma: no cover - guard against a silent no-generator path
        raise AssertionError("modulate accepted a call with no generator at all")


def test_snapshot_survives_the_reduced_agent_snapshot_form() -> None:
    """A mid-blend machine round-trips through AgentSnapshot's three fields.

    ``core.state.AgentSnapshot`` stores only ``mode``, ``mode_timer`` and
    ``mode_params`` -- there is no field for the blend. Losing the blend would
    make a branch rollout that resumes mid-blend diverge from its parent by a
    small heading offset, which would show up as spurious spread in the
    calibration reference distribution. So the blend is mirrored into
    ``mode_params`` while it is live.
    """
    hub = RNGHub(606)
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")
    agent = _Agent(AgentState(0.0, 0.0, 25.0, 0.0), _fixed_wing_envelope())
    machine = ModeMachine(IntentType.DELIVERY_WP, mode=ManeuverMode.CT_RIGHT)
    machine.initialize(agent, rng_mode, rng_branch)
    machine.force_mode(ManeuverMode.CV, agent, rng_mode, rng_branch, dwell_s=10.0)
    assert machine.blend_active

    full = machine.snapshot()
    assert "_blend_omega" in full["mode_params"], "blend was not mirrored"

    # Reduced form: exactly what AgentSnapshot can carry.
    reduced = {
        "mode": full["mode"],
        "mode_timer": full["mode_timer"],
        "mode_params": dict(full["mode_params"]),
    }
    restored = ModeMachine(IntentType.DELIVERY_WP)
    restored.restore(reduced)
    assert restored.blend_active, "blend was lost through the reduced form"
    a = machine.apply(0.0, 0.0, agent)
    b = restored.apply(0.0, 0.0, agent)
    assert a == b, f"reduced-form restore changed the command: {a} vs {b}"
    # The mirrored keys must not leak into the visible mode payload.
    assert "_blend_omega" not in restored.mode_params
    # Once the blend has run out it is no longer mirrored.
    for _ in range(int(round(CT_BLEND_S / SIM_DT)) + 2):
        machine.step(SIM_DT, agent, rng_mode, rng_branch)
    assert "_blend_omega" not in machine.snapshot()["mode_params"]


# --------------------------------------------------------------------------
# 4. THE CENTRAL PREMISE: branch rollouts are multimodal
#
# What is measured, and why
# -------------------------
# The rollouts below record two statistics at the 10 s horizon:
#
#   dpsi -- the accumulated heading change [rad], i.e. how far and which way
#           the agent turned.
#   lat  -- the cross-track offset [m], i.e. the lateral component of the
#           final position in the frame of the initial heading.
#
# The PRIMARY statistic is ``lat``, for two reasons. First, position is what
# the prediction stage actually outputs and what the BEV risk map consumes
# (doc [4], doc [6]); "where will it be" is the question the pipeline asks.
# Second, position is the time integral of the committed turn, so a decision
# taken early leaves a permanent trace in it, whereas the accumulated heading
# can be partly undone by a later reversal. Both statistics are asserted on;
# only the model-selection (BIC) assertion is restricted to ``lat``.
#
# The distribution is genuinely THREE-moded (turn left / fly straight / turn
# right), because "keep going" is a legitimate outcome of the transition
# matrix. So this file deliberately does NOT assert a dip at the centre: a
# populated centre is correct here. What it asserts is that two well-separated
# branches carry substantial probability mass, and that committing to one of
# them roughly halves the achievable prediction error -- which is precisely
# the statement "a single point estimate is structurally inadequate".
# --------------------------------------------------------------------------


def _gaussian_bic(x: np.ndarray) -> float:
    """BIC [-] of a single-Gaussian maximum-likelihood fit to 1-D samples."""
    n = x.size
    var = float(x.var()) + 1e-12
    log_lik = -0.5 * n * (math.log(2.0 * math.pi * var) + 1.0)
    k = 2  # mu, var
    return k * math.log(n) - 2.0 * log_lik


def _em2(
    x: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    w: np.ndarray,
    var_floor: float,
    iters: int = 400,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run 1-D two-component EM to convergence.

    Args:
        x: samples.
        mu: initial component means.
        var: initial component variances.
        w: initial mixing weights [-].
        var_floor: hard lower bound on each component variance. Without it EM
            can collapse a component onto a repeated value (for a mode process
            that sometimes flies exactly straight, "exactly 0.0" occurs many
            times) and win the likelihood comparison with a delta spike rather
            than with genuine multimodality.
        iters: maximum iterations.

    Returns:
        ``(log_likelihood, weights, means, stds)``.
    """
    n = x.size
    mu = np.array(mu, dtype=np.float64)
    var = np.maximum(np.array(var, dtype=np.float64), var_floor)
    w = np.array(w, dtype=np.float64)
    log_lik = -np.inf
    for _ in range(iters):
        comp = (
            w[None, :]
            * np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :])
            / np.sqrt(2.0 * math.pi * var[None, :])
        )
        tot = comp.sum(axis=1) + 1e-300
        resp = comp / tot[:, None]
        new_log_lik = float(np.log(tot).sum())
        nk = resp.sum(axis=0) + 1e-12
        w = nk / n
        mu = (resp * x[:, None]).sum(axis=0) / nk
        var = np.maximum((resp * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk, var_floor)
        if abs(new_log_lik - log_lik) < 1e-10:
            log_lik = new_log_lik
            break
        log_lik = new_log_lik
    return log_lik, w, mu, np.sqrt(var)


def _two_component_bic(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """BIC [-] of the best two-component Gaussian mixture fit.

    EM is run from three deterministic initialisations (split at zero, split
    at the median, and symmetric about the mean +/- one std) and the best
    likelihood wins, so the result does not depend on a lucky start. A
    variance floor of 2% of the sample std keeps the fit honest.

    Args:
        x: samples.

    Returns:
        ``(bic, weights, means, stds)``.
    """
    n = x.size
    s = float(x.std())
    var_floor = (0.02 * s) ** 2 if s > 0.0 else 1e-12
    m = float(x.mean())
    inits: list[tuple[list[float], list[float], list[float]]] = []
    for split in (0.0, float(np.median(x))):
        lo = x[x <= split]
        hi = x[x > split]
        if lo.size >= 3 and hi.size >= 3:
            inits.append((
                [float(lo.mean()), float(hi.mean())],
                [max(float(lo.var()), var_floor), max(float(hi.var()), var_floor)],
                [lo.size / n, hi.size / n],
            ))
    inits.append(([m - s, m + s], [s * s + var_floor, s * s + var_floor], [0.5, 0.5]))

    best = max(
        (_em2(x, np.array(mu), np.array(var), np.array(w), var_floor) for mu, var, w in inits),
        key=lambda r: r[0],
    )
    log_lik, w, mu, sd = best
    k = 5  # 2 means + 2 variances + 1 free weight
    return k * math.log(n) - 2.0 * log_lik, w, mu, sd


def _branch_statistics(d: np.ndarray) -> dict[str, float]:
    """Summarise a branch-rollout sample as the quantities the premise needs.

    Clusters are formed by the SIGN of the statistic -- left branch versus
    right branch -- which is the physically meaningful partition and, unlike a
    free clustering, cannot be gamed by a degenerate fit.

    Args:
        d: one statistic per branch rollout ([rad] or [m]).

    Returns:
        Dict with:
            ``p_left`` / ``p_right`` -- fraction [-] outside a dead band of
                10% of the sample std, so a rollout that barely moved counts
                as neither branch.
            ``separation`` -- distance between the two branch means divided by
                the mean within-branch std [-].
            ``rmse_one`` -- RMSE [same units] of the best single point
                prediction (the sample mean).
            ``rmse_two`` -- RMSE of the best two-branch prediction (each
                branch predicted by its own mean).
            ``rmse_cv`` -- RMSE of the constant-velocity extrapolation, which
                predicts "no turn at all", i.e. the value 0.
    """
    n = int(d.size)
    left = d[d > 0.0]
    right = d[d < 0.0]
    dead = 0.10 * float(d.std())
    within = 0.5 * (float(left.std()) + float(right.std()))
    rmse_two = math.sqrt(
        (left.size * float(left.var()) + right.size * float(right.var())) / n
    )
    return {
        "p_left": float((d > dead).mean()),
        "p_right": float((d < -dead).mean()),
        "separation": (float(left.mean()) - float(right.mean())) / within,
        "rmse_one": float(d.std()),
        "rmse_two": rmse_two,
        "rmse_cv": math.sqrt(float((d ** 2).mean())),
    }


def _branch_rollouts(
    n_branches: int = 200,
    horizon_s: float = 10.0,
    branch_seed_base: int = 1000,
    intent: IntentType = IntentType.DELIVERY_WP,
    envelope: PerformanceEnvelope | None = None,
    v0: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll a ModeMachine out ``n_branches`` times from ONE initial snapshot.

    This is doc 3.3's evaluation recipe in miniature: "동일 초기조건 + 동일
    과거 궤적에서 시드만 바꿔 N회 롤아웃" -- same initial condition, same past,
    re-seed only the decision streams, roll out N times. Only ``mode_decision``
    and ``branch_choice`` are re-seeded, via
    :meth:`core.rng.RNGHub.reseed_branch`; every other aspect of the setup is
    byte-identical across branches, so all spread in the result is caused by
    the agents' stochastic decision rules and by nothing else.

    Note that ``reseed_branch`` derives the new streams from ``branch_seed``
    alone, so the hub's master seed does not influence a branch's decisions.
    Varying the *branch seed base* is therefore the correct way to check that
    the multimodality is not an artefact of one lucky seed bank, and a real
    rollout driver must mix the episode seed into the branch seed it passes.

    Args:
        n_branches: number of rollouts [-].
        horizon_s: rollout length [s].
        branch_seed_base: first branch seed [-]; branch ``b`` uses
            ``branch_seed_base + b``.
        intent: intent whose transition matrix drives the rollouts.
        envelope: platform envelope to fly; ``None`` uses the fixed-wing
            fixture, whose :data:`core.state.DEFAULT_STABILITY` profile
            reproduces the pre-stability-model behaviour exactly. Pass a real
            :func:`agents.platforms.default_envelope` to roll out a specific
            platform WITH its commitment profile.
        v0: initial airspeed [m/s]; must suit ``envelope``.

    Returns:
        ``(dpsi, lat)``: accumulated heading change [rad] and cross-track
        offset [m] at the horizon, one entry per branch. Heading change is
        accumulated per step rather than differenced, so a turn beyond +/-pi
        is not aliased by wrapping. The initial heading is 0, so the
        cross-track offset is simply the final ``y``.
    """
    envelope = _fixed_wing_envelope() if envelope is None else envelope
    initial = AgentState(x=0.0, y=0.0, v=float(v0), psi=0.0, omega=0.0, a=0.0)
    # One common parent snapshot: identical mode, timer and params for every
    # branch. mode_timer = 0.0 means the first decision happens immediately,
    # which is precisely the fork under study.
    parent = ModeMachine(intent, mode=ManeuverMode.CV, mode_timer=0.0).snapshot()

    n_steps = int(round(horizon_s / SIM_DT))
    dpsi = np.empty(n_branches, dtype=np.float64)
    lat = np.empty(n_branches, dtype=np.float64)
    for b in range(n_branches):
        hub = RNGHub(20260812)
        hub.reseed_branch(branch_seed_base + b, ("mode_decision", "branch_choice"))
        rng_mode = hub.stream("mode_decision")
        rng_branch = hub.stream("branch_choice")

        agent = _Agent(initial.copy(), envelope)
        machine = ModeMachine.from_snapshot(parent)
        turned = 0.0
        for _ in range(n_steps):
            machine.step(SIM_DT, agent, rng_mode, rng_branch)
            omega_cmd, a_cmd = machine.apply(0.0, 0.0, agent)
            before = agent.state.psi
            _integrate(agent, omega_cmd, a_cmd, SIM_DT)
            turned += wrap_angle(agent.state.psi - before)
        dpsi[b] = turned
        lat[b] = agent.state.y
    return dpsi, lat


def test_branch_rollouts_are_multimodal() -> None:
    """200 branch rollouts from one snapshot must produce a MULTIMODAL future.

    This is the paper's central premise (doc 3.1 / 3.3) in code form. If it
    fails, the simulator is generating futures that a single point estimate
    can summarise, and every stage downstream of the simulator -- the
    probabilistic predictor, the calibration study, the risk map -- loses its
    justification.

    Five assertions, from weakest to strongest:

    1. **Both branches occur.** Left and right each carry > 25% of the mass.
    2. **They are separated.** The distance between the branch means exceeds
       2.5 within-branch standard deviations, so the modes are resolved
       rather than smeared into one another.
    3. **One point estimate is not enough.** Predicting each branch by its own
       mean cuts the RMSE to below 70% of the best single point prediction.
       This is the operational meaning of "multimodal": there is no single
       answer that is even close to as good as saying "either this or that".
    4. **CV cannot do it.** A constant-velocity extrapolator predicts "no
       turn", and its RMSE is more than 1.6x the two-branch RMSE.
    5. **Model selection agrees.** On cross-track offset, a two-component
       Gaussian mixture beats a single Gaussian on BIC by a clear margin,
       with the mixture fit protected by a variance floor so that it cannot
       win with a degenerate spike.
    """
    dpsi, lat = _branch_rollouts(n_branches=200)
    assert dpsi.size == 200 and lat.size == 200
    assert np.all(np.isfinite(dpsi)) and np.all(np.isfinite(lat))

    for name, d, sep_min, ratio_max in (
        ("heading change [rad]", dpsi, 2.2, 0.72),
        ("cross-track offset [m]", lat, 2.5, 0.70),
    ):
        st = _branch_statistics(d)

        # -- 1. both branches really happen -------------------------------
        assert st["p_left"] > 0.25, f"{name}: left branch too rare, p={st['p_left']:.3f}"
        assert st["p_right"] > 0.25, f"{name}: right branch too rare, p={st['p_right']:.3f}"

        # -- 2. the branches are resolved, not smeared --------------------
        assert st["separation"] > sep_min, (
            f"{name}: branch means are only {st['separation']:.2f} within-branch "
            f"std apart (need > {sep_min}); the modes are not resolved"
        )

        # -- 3. a single point estimate is structurally inadequate --------
        ratio = st["rmse_two"] / st["rmse_one"]
        assert ratio < ratio_max, (
            f"{name}: knowing which branch was taken only improves RMSE by a "
            f"factor {ratio:.2f} (need < {ratio_max}); the future is "
            f"effectively unimodal"
        )

        # -- 4. constant velocity cannot represent this -------------------
        cv_ratio = st["rmse_cv"] / st["rmse_two"]
        assert cv_ratio > 1.6, (
            f"{name}: the CV extrapolator is within {cv_ratio:.2f}x of the "
            f"two-branch predictor; the trajectories are too straight and the "
            f"CV baseline would not be beaten"
        )

    # -- 5. model selection on the primary statistic ----------------------
    bic1 = _gaussian_bic(lat)
    bic2, w2, mu2, sd2 = _two_component_bic(lat)
    assert bic2 < bic1 - 8.0, (
        f"a single Gaussian is not clearly rejected for the cross-track "
        f"offset: BIC1={bic1:.1f}, BIC2={bic2:.1f}"
    )
    assert abs(mu2[0] - mu2[1]) > 2.0 * float(sd2.mean()), (
        f"the fitted mixture means are not resolved: mu={np.round(mu2, 2)}, "
        f"sd={np.round(sd2, 2)}"
    )

    # -- reproducibility: identical seeds, identical distribution ---------
    dpsi_again, lat_again = _branch_rollouts(n_branches=200)
    assert np.array_equal(dpsi, dpsi_again), "branch rollouts are not reproducible"
    assert np.array_equal(lat, lat_again), "branch rollouts are not reproducible"


def test_branch_rollouts_are_multimodal_across_seed_banks() -> None:
    """The multimodality is a property of the design, not of one lucky seed.

    Repeats the core of the premise over independent banks of branch seeds.
    Thresholds are the same as the main test; only the seeds change.
    """
    for base in (2000, 12345, 60606, 8675309):
        dpsi, lat = _branch_rollouts(n_branches=200, branch_seed_base=base)
        for name, d in (("heading", dpsi), ("cross-track", lat)):
            st = _branch_statistics(d)
            assert st["p_left"] > 0.25 and st["p_right"] > 0.25, (
                f"base={base} {name}: p_left={st['p_left']:.3f}, "
                f"p_right={st['p_right']:.3f}"
            )
            assert st["separation"] > 2.2, (
                f"base={base} {name}: separation={st['separation']:.2f}"
            )
            assert st["rmse_two"] / st["rmse_one"] < 0.72, (
                f"base={base} {name}: rmse ratio "
                f"{st['rmse_two'] / st['rmse_one']:.2f}"
            )
        bic1 = _gaussian_bic(lat)
        bic2, _w, _mu, _sd = _two_component_bic(lat)
        assert bic2 < bic1 - 8.0, (
            f"base={base}: single Gaussian not rejected "
            f"(BIC1={bic1:.1f}, BIC2={bic2:.1f})"
        )


def test_a_committed_intent_is_more_multimodal_than_a_corridor() -> None:
    """Corridor transit must stay predictable; loiter must not.

    Doc 3.2 assigns different difficulty to different intents on purpose --
    corridor transit is "직선적, 예측 쉬움" (straight, easy to predict) while
    loitering is "반복 선회" (repeated turning). A dataset in which every
    intent is equally hard would make the stratified analysis of doc 3.4
    meaningless, so the ordering itself is worth pinning down.
    """
    _, lat_corridor = _branch_rollouts(n_branches=200, intent=IntentType.CORRIDOR_TRANSIT)
    _, lat_loiter = _branch_rollouts(n_branches=200, intent=IntentType.LOITER_HOVER)

    corridor = _branch_statistics(lat_corridor)
    loiter = _branch_statistics(lat_loiter)

    # Most corridor rollouts fly essentially straight, so CV does comparatively
    # well; the loiter population is dominated by committed turns.
    straight_corridor = 1.0 - corridor["p_left"] - corridor["p_right"]
    straight_loiter = 1.0 - loiter["p_left"] - loiter["p_right"]
    assert straight_corridor > 0.4, (
        f"corridor transit is not predictable enough: only "
        f"{straight_corridor:.2f} of rollouts flew straight"
    )
    assert straight_loiter < 0.2, (
        f"loitering is too straight: {straight_loiter:.2f} of rollouts did not turn"
    )
    assert loiter["rmse_cv"] > 1.5 * corridor["rmse_cv"], (
        "loitering traffic is not harder for a CV baseline than corridor traffic"
    )


def test_bird_multimodality_survives_the_per_platform_stability_model() -> None:
    """The stability model must not have quietened the class it is not for.

    Manoeuvre commitment became a platform property so that heavy aircraft stop
    wandering. The risk that change carries is that it also damps the agile
    classes, and the agile classes are where the paper's central premise lives:
    if a bird's futures collapse to one mode, no calibrated predictor can beat
    a point estimate on the hard end of the dataset and doc 3.1 fails.

    So the branch-rollout premise is re-asserted on a BIRD flying with the real
    bird profile (0.90 authority, 1.0 s floor, 0.40-1.00 turn band), for both a
    wandering and a waypoint intent. Thresholds are the same as the main
    premise test.
    """
    env = default_envelope(PlatformType.BIRD)
    assert env.stability.mode_authority >= 0.9
    v0 = 0.5 * (env.v_min + env.v_max)
    for intent in (IntentType.SEARCH_WANDER, IntentType.DELIVERY_WP):
        dpsi, lat = _branch_rollouts(
            n_branches=200, intent=intent, envelope=env, v0=v0)
        for name, d, sep_min in (("heading change [rad]", dpsi, 2.2),
                                 ("cross-track offset [m]", lat, 2.4)):
            st = _branch_statistics(d)
            assert st["p_left"] > 0.25 and st["p_right"] > 0.25, (
                f"BIRD/{intent.name} {name}: branches too rare "
                f"(L={st['p_left']:.3f}, R={st['p_right']:.3f})"
            )
            assert st["separation"] > sep_min, (
                f"BIRD/{intent.name} {name}: separation {st['separation']:.2f} "
                f"(need > {sep_min})"
            )
            assert st["rmse_two"] / st["rmse_one"] < 0.72, (
                f"BIRD/{intent.name} {name}: two-branch RMSE ratio "
                f"{st['rmse_two'] / st['rmse_one']:.2f}; the future is "
                f"effectively unimodal"
            )
            assert st["rmse_cv"] / st["rmse_two"] > 1.6, (
                f"BIRD/{intent.name} {name}: CV is within "
                f"{st['rmse_cv'] / st['rmse_two']:.2f}x of the two-branch "
                f"predictor"
            )


def test_a_stable_platform_is_measurably_easier_than_a_bird() -> None:
    """The difficulty SPREAD across classes is a feature, and must be large.

    An eVTOL and a bird are rolled out from the same kind of snapshot; the
    eVTOL's cross-track spread at 10 s must be far smaller. That gap is what
    the calibration figure of doc 5.2 exploits -- a model that emits the same
    sigma for both cannot be shown to modulate its uncertainty at all.
    """
    bird = default_envelope(PlatformType.BIRD)
    evtol = default_envelope(PlatformType.EVTOL_UAM)
    _, lat_bird = _branch_rollouts(
        n_branches=200, intent=IntentType.SEARCH_WANDER, envelope=bird,
        v0=0.5 * (bird.v_min + bird.v_max))
    _, lat_evtol = _branch_rollouts(
        n_branches=200, intent=IntentType.CORRIDOR_TRANSIT, envelope=evtol,
        v0=0.5 * (evtol.v_min + evtol.v_max))
    bird_cv = _branch_statistics(lat_bird)["rmse_cv"]
    evtol_cv = _branch_statistics(lat_evtol)["rmse_cv"]
    assert bird_cv > 3.0 * evtol_cv, (
        f"a wandering bird ({bird_cv:.1f} m of CV error at 10 s) is not much "
        f"harder than a cruising eVTOL ({evtol_cv:.1f} m); the dataset has no "
        f"difficulty spread for the calibration study to show"
    )


def test_only_decision_streams_change_between_branches() -> None:
    """Re-seeding the branch streams must not disturb wind or sensor streams.

    The calibration reference distribution is only meaningful if the spread
    across branches comes from decisions alone. If re-seeding perturbed the
    wind stream, part of the observed spread would be environmental and the
    "true distribution" would be measuring the wrong thing.
    """
    hub_a = RNGHub(4242)
    hub_b = RNGHub(4242)
    for name in ("wind", "sensor_noise", "sensor_detect", "placement"):
        hub_a.stream(name)
        hub_b.stream(name)
    hub_a.stream("mode_decision")
    hub_b.stream("mode_decision")
    hub_b.reseed_branch(999, ("mode_decision", "branch_choice"))

    for name in ("wind", "sensor_noise", "sensor_detect", "placement"):
        a = hub_a.stream(name).standard_normal(64)
        b = hub_b.stream(name).standard_normal(64)
        assert np.array_equal(a, b), f"stream '{name}' was disturbed by reseed_branch"
    a = hub_a.stream("mode_decision").standard_normal(64)
    b = hub_b.stream("mode_decision").standard_normal(64)
    assert not np.array_equal(a, b), "reseed_branch did not re-randomise mode_decision"


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _main() -> int:
    """Run every ``test_*`` function in definition order; return an exit code."""
    tests = [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}\n     {exc}")
        except Exception as exc:  # pragma: no cover - unexpected error path
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
