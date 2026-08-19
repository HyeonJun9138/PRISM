"""Semi-Markov maneuver-mode process -- the source of trajectory multimodality.

This module implements doc 3.3 directly. An agent does not follow waypoints
placed on a map; it holds a *maneuver mode* for a randomly drawn dwell time
and then samples a new mode from a transition matrix conditioned on its
intent. Two constraints from doc 3.3 are structural, not tunable:

    (a) 물리적으로 가능한 만큼만 -- only what is physically possible.
        Every commanded turn rate is expressed as a fraction of
        :meth:`core.state.PerformanceEnvelope.max_turn_rate` at the *current*
        airspeed, so a command can never exceed the platform's envelope.

    (b) 방금 하던 기동을 몇 초는 유지 -- hold the current maneuver for a few
        seconds. Enforced by a dwell floor in
        :func:`agents.behavior.sample_dwell`. Re-deciding every step yields
        jitter, not maneuvers, and jitter is something a constant-velocity
        extrapolator predicts *well*.

        That floor is PER PLATFORM, not global. "A few seconds" is right for a
        bird and absurd for a passenger eVTOL, which holds a manoeuvre for tens
        of seconds; a single global floor produces a mushy wandering trace for
        every heavy aircraft in the dataset, which doc 7.2 flags as the first
        thing a reviewer notices. Commitment therefore arrives as a
        :class:`core.state.StabilityProfile` carried by the agent's
        performance envelope (table: :data:`agents.platforms.STABILITY_PROFILES`)
        and shapes four things here -- the dwell floor, the mean dwell, the
        turn-aggressiveness band, and how hard a mode overrides the intent.

Mode authority (why a coordinated turn BLENDS rather than overrides)
-------------------------------------------------------------------
A latched turn used to replace the intent's command outright. For a bird that
is right: the turn is a decision, and it must survive long enough to matter.
For an eVTOL it is wrong -- the override fights the corridor guidance and the
aircraft wanders off its centreline. So the two are combined by the platform's
``mode_authority``::

    omega_cmd = (1 - authority) * omega_intent + authority * omega_mode

A bird (0.90) still executes fully committed, decisive turns, so the branch
multimodality that the whole paper rests on is preserved. An eVTOL (0.35)
stays glued to its corridor and merely biases. Both ends are then saturated at
the envelope cap and rate-limited by the integrator exactly as before.

Why the left/right coin lives in a separate RNG stream
------------------------------------------------------
When a transition selects a coordinated turn, *which way* it turns is drawn
from the ``branch_choice`` stream, never from ``mode_decision``. Those are
the two streams :meth:`core.rng.RNGHub.reseed_branch` re-randomises for
branch rollouts. Re-running the same initial condition with N different
branch seeds therefore produces genuinely different turn directions while
wind, placement and sensor noise replay bit-identically -- which is exactly
the empirical "true" future distribution doc 3.3 uses as the calibration
reference.

Frame / sign convention (same as the whole package):
    psi [rad] CCW from +X; positive omega [rad/s] = left = CCW.

Expected agent interface
------------------------
:meth:`ModeMachine.step` and :meth:`ModeMachine.apply` take an ``agent``
object and read exactly two attributes from it:

    ``agent.state``    -> :class:`core.state.AgentState`
    ``agent.envelope`` -> :class:`core.state.PerformanceEnvelope`

For convenience during unit testing and for callers that have not built an
agent wrapper yet, a bare ``AgentState`` is rejected (an envelope is
mandatory) but a plain object / ``SimpleNamespace`` carrying those two
attributes is accepted, as is the alias ``agent.perf`` for the envelope.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from agents.behavior import (
    DEFAULT_MIN_DWELL_S,
    binary_branch,
    categorical,
    sample_dwell,
    sample_fraction,
)
from agents.platforms import intent_dwell_multiplier
from core.constants import EPS_V, PRED_SECONDS
from core.state import DEFAULT_STABILITY, AgentState, PerformanceEnvelope, StabilityProfile
from core.types import IntentType, ManeuverMode

__all__ = [
    "MODE_ORDER",
    "MODE_INDEX",
    "TRANSITION_MATRICES",
    "DWELL_MEAN_S",
    "TURN_FRACTION_RANGE",
    "ACCEL_FRACTION_RANGE",
    "CT_BLEND_S",
    "CV_TRACK_GAIN",
    "CV_OMEGA_FRACTION",
    "HOVER_TAU_S",
    "biased_transitions",
    "ModeMachine",
]


# --------------------------------------------------------------------------
# Mode ordering: row/column index of the transition matrices.
# --------------------------------------------------------------------------

#: Canonical row/column order of :data:`TRANSITION_MATRICES`. The index of a
#: mode equals its ``IntEnum`` value, so ``MODE_ORDER[int(mode)] is mode``.
MODE_ORDER: tuple[ManeuverMode, ...] = (
    ManeuverMode.CV,
    ManeuverMode.CT_LEFT,
    ManeuverMode.CT_RIGHT,
    ManeuverMode.CA,
    ManeuverMode.HOVER,
)

#: Reverse lookup ``ManeuverMode -> matrix index``.
MODE_INDEX: dict[ManeuverMode, int] = {m: i for i, m in enumerate(MODE_ORDER)}

_N_MODES: int = len(MODE_ORDER)
_I_CV = MODE_INDEX[ManeuverMode.CV]
_I_CTL = MODE_INDEX[ManeuverMode.CT_LEFT]
_I_CTR = MODE_INDEX[ManeuverMode.CT_RIGHT]
_I_CA = MODE_INDEX[ManeuverMode.CA]
_I_HOVER = MODE_INDEX[ManeuverMode.HOVER]


def _matrix(rows: list[list[float]]) -> np.ndarray:
    """Build a row-stochastic 5x5 transition matrix [-] from nested lists."""
    m = np.asarray(rows, dtype=np.float64)
    if m.shape != (_N_MODES, _N_MODES):
        raise ValueError(f"transition matrix must be {_N_MODES}x{_N_MODES}, got {m.shape}")
    return m


# --------------------------------------------------------------------------
# Transition matrices, one per IntentType (doc 3.2 (1) x doc 3.7 mode set).
#
# Row = current mode, column = next mode, both in MODE_ORDER.
# Entries are the probability [-] of choosing that next mode when the dwell
# of the current mode expires. Rows are asserted row-stochastic at import.
#
# These are the BASE matrices, common to every platform. A platform's
# ``StabilityProfile.diag_bias`` is blended in per agent by
# :func:`biased_transitions`, so an eVTOL sees a stickier version of the same
# matrix and an UNKNOWN a twitchier one.
#
# Reading guide for the numbers:
#   * CT/CA/HOVER "diagonal" entries are small on purpose -- re-entering the
#     same turn direction merely extends it, which the dwell distribution
#     already covers; the interesting transitions are the ones that change the
#     qualitative behaviour. The CV diagonal is the exception: for the corridor
#     intent it is deliberately large, because "still flying straight" is the
#     defining property of that class rather than a missed opportunity to turn.
#   * CT_LEFT -> CT_RIGHT (and vice versa) is the S-turn / zig-zag entry, so
#     it is the dominant off-diagonal for the wandering intents.
#   * The HOVER column is non-zero only where hovering makes sense; it is
#     additionally masked at runtime for platforms whose envelope reports
#     ``can_hover == False``.
# --------------------------------------------------------------------------

TRANSITION_MATRICES: dict[IntentType, np.ndarray] = {
    # Corridor transit: straight and predictable (doc 3.2: "직선적, 예측 쉬움").
    # Turns are occasional course corrections, and the agent always falls back
    # to CV. This intent is the one a CV baseline does reasonably well on --
    # it exists to keep the dataset from being uniformly hard.
    # The 0.86 CV->CV entry is what makes an aircraft in cruise stay in cruise:
    # with a platform diag_bias of 0.35 an eVTOL ends up at 0.909, i.e. it
    # leaves cruise roughly once per ten decision epochs. It is also the reason
    # CORRIDOR_TRANSIT is the EASY end of the difficulty spread that
    # analysis.cv_baseline.EXPECTED_DIFFICULTY grades against.
    IntentType.CORRIDOR_TRANSIT: _matrix([
        # CV    CT_L   CT_R   CA     HOVER
        [0.86,  0.05,  0.05,  0.04,  0.00],   # from CV
        [0.70,  0.05,  0.10,  0.15,  0.00],   # from CT_LEFT
        [0.70,  0.10,  0.05,  0.15,  0.00],   # from CT_RIGHT
        [0.75,  0.09,  0.09,  0.07,  0.00],   # from CA
        [1.00,  0.00,  0.00,  0.00,  0.00],   # from HOVER (recover to cruise)
    ]),
    # Delivery / survey waypoints: periodic turning (doc 3.2: "주기적 선회").
    # High CV -> CT mass, so a leg change is the dominant event. A turn is
    # followed mostly by a straight leg (CT -> CV = 0.55) rather than by an
    # immediate reversal: a survey grid does contain 180 deg reversals, but it
    # flies them as turn-leg-turn, not as one continuous S. Keeping the direct
    # CT -> opposite-CT term modest is what lets a committed turn survive the
    # 15 s prediction horizon instead of being cancelled inside it, which is
    # the difference between a multimodal future and a smeared unimodal one.
    IntentType.DELIVERY_WP: _matrix([
        [0.30,  0.28,  0.28,  0.14,  0.00],
        [0.55,  0.08,  0.17,  0.20,  0.00],
        [0.55,  0.17,  0.08,  0.20,  0.00],
        [0.40,  0.24,  0.24,  0.12,  0.00],
        [0.60,  0.18,  0.18,  0.04,  0.00],
    ]),
    # Loiter / hover observation: repeated turning plus low-speed station
    # keeping (doc 3.2: "반복 선회, 저속/정지 근접").
    IntentType.LOITER_HOVER: _matrix([
        [0.18,  0.26,  0.26,  0.10,  0.20],
        [0.15,  0.30,  0.20,  0.10,  0.25],
        [0.15,  0.20,  0.30,  0.10,  0.25],
        [0.25,  0.20,  0.20,  0.10,  0.25],
        [0.20,  0.20,  0.20,  0.10,  0.30],
    ]),
    # Search / wander: zig-zag with random reversals (doc 3.2: "지그재그,
    # 랜덤 방향전환"). The CT -> opposite-CT entry dominates. OOD split.
    IntentType.SEARCH_WANDER: _matrix([
        [0.20,  0.33,  0.33,  0.14,  0.00],
        [0.20,  0.08,  0.52,  0.15,  0.05],
        [0.20,  0.52,  0.08,  0.15,  0.05],
        [0.25,  0.30,  0.30,  0.10,  0.05],
        [0.30,  0.28,  0.28,  0.09,  0.05],
    ]),
    # Reactive avoid: nominal behaviour is ordinary transit; the actual
    # avoidance maneuver is FORCED by agents.behavior.reactive_avoid_decision
    # through ModeMachine.force_mode, not produced by this matrix. OOD split.
    IntentType.REACTIVE_AVOID: _matrix([
        [0.55,  0.16,  0.16,  0.13,  0.00],
        [0.55,  0.10,  0.15,  0.15,  0.05],
        [0.55,  0.15,  0.10,  0.15,  0.05],
        [0.60,  0.15,  0.15,  0.10,  0.00],
        [0.70,  0.12,  0.12,  0.06,  0.00],
    ]),
    # Passive drift (debris): unpowered, advected by the wind. Almost pure CV;
    # the runtime mask for ``is_powered == False`` collapses this to CV anyway,
    # the matrix is kept complete so the structure is uniform. OOD split.
    IntentType.PASSIVE_DRIFT: _matrix([
        [0.94,  0.02,  0.02,  0.02,  0.00],
        [0.90,  0.04,  0.04,  0.02,  0.00],
        [0.90,  0.04,  0.04,  0.02,  0.00],
        [0.90,  0.03,  0.03,  0.04,  0.00],
        [1.00,  0.00,  0.00,  0.00,  0.00],
    ]),
}


def _assert_row_stochastic() -> None:
    """Validate every transition matrix at import time.

    A silently non-stochastic row would bias the mode process in a way that
    is nearly invisible in trajectory plots but corrupts the ground-truth
    mode labels stored in the dataset, so this is checked eagerly rather
    than in a test only.

    Raises:
        AssertionError: if any matrix has the wrong shape, a negative entry,
            or a row that does not sum to 1 within 1e-12.
    """
    for intent, mat in TRANSITION_MATRICES.items():
        assert mat.shape == (_N_MODES, _N_MODES), (
            f"TRANSITION_MATRICES[{intent!r}] has shape {mat.shape}, "
            f"expected ({_N_MODES}, {_N_MODES})"
        )
        assert np.all(np.isfinite(mat)), f"TRANSITION_MATRICES[{intent!r}] has non-finite entries"
        assert np.all(mat >= 0.0), f"TRANSITION_MATRICES[{intent!r}] has a negative entry"
        sums = mat.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-12), (
            f"TRANSITION_MATRICES[{intent!r}] rows must sum to 1.0, got {sums.tolist()}"
        )
    missing = [i for i in IntentType if i not in TRANSITION_MATRICES]
    assert not missing, f"TRANSITION_MATRICES is missing intents: {missing}"


_assert_row_stochastic()


# --------------------------------------------------------------------------
# Per-platform diagonal bias
# --------------------------------------------------------------------------

#: Cache of biased matrices, keyed by ``(intent, rounded diag_bias)``. A dozen
#: entries at most (7 platforms x 6 intents), all read-only.
_BIASED_CACHE: dict[tuple[int, float], np.ndarray] = {}


def biased_transitions(intent: IntentType, diag_bias: float) -> np.ndarray:
    """Blend a platform's diagonal bias into an intent's transition matrix.

    ``P' = (1 - b) P + b I``, then rows are renormalised. The result is the
    stickiness of the mode process expressed as one number: a stable platform
    (eVTOL ``b = 0.35``) re-enters the mode it is already flying far more often
    than an agile one, which is what turns a long dwell into a long *behaviour*
    rather than a long sequence of alternating short ones.

    A NEGATIVE bias (only ``PlatformType.UNKNOWN``, ``b = -0.10``) does the
    opposite: it moves mass OFF the diagonal so the OOD class changes mode more
    often than any trained platform. Negative blending can drive a small
    diagonal entry below zero, so entries are clipped at zero before the rows
    are renormalised -- which is why renormalisation is not a no-op here even
    though the unclipped blend is already row-stochastic.

    Args:
        intent: the intent whose base matrix is being biased.
        diag_bias: the platform's bias [-] in ``(-1, 1)``.

    Returns:
        A row-stochastic ``[5,5]`` float64 matrix. The array is CACHED and
        shared between agents -- treat it as read-only.

    Raises:
        ValueError: if the bias is outside ``(-1, 1)``.
    """
    it = IntentType(int(intent))
    b = float(diag_bias)
    if not (-1.0 < b < 1.0):
        raise ValueError(f"diag_bias must be in (-1, 1), got {diag_bias!r}")
    key = (int(it), round(b, 12))
    cached = _BIASED_CACHE.get(key)
    if cached is not None:
        return cached
    base = TRANSITION_MATRICES[it]
    if b == 0.0:
        _BIASED_CACHE[key] = base
        return base
    m = (1.0 - b) * base + b * np.eye(_N_MODES, dtype=np.float64)
    np.clip(m, 0.0, None, out=m)
    row_sums = m.sum(axis=1, keepdims=True)
    # A row cannot sum to zero: the blend only ever removes mass from the
    # diagonal, and every base row has off-diagonal mass. Guard anyway rather
    # than emit NaNs into the mode process.
    row_sums[row_sums <= 0.0] = 1.0
    m /= row_sums
    m.flags.writeable = False
    _BIASED_CACHE[key] = m
    return m


# --------------------------------------------------------------------------
# FALLBACK dwell means [s], per (intent, mode).
#
# These set the *time scale* of the multimodality. Too short and the mode
# process becomes jitter that averages out to a straight line (CV wins);
# too long and a 60 s episode contains only one decision (no branching).
#
# This table is used ONLY by a mode machine whose agent carries no per-platform
# stability profile, i.e. a hand-built :class:`core.state.PerformanceEnvelope`
# (unit tests, GUI probes), which gets :data:`core.state.DEFAULT_STABILITY`.
# Every agent built through :mod:`agents.platforms` instead uses
#
#     mean dwell = profile.dwell_mean_s * INTENT_DWELL_MULTIPLIER[intent]
#
# with the platform's own floor applied after the multiplication, because
# manoeuvre commitment is a property of the airframe, not of the intent alone.
# --------------------------------------------------------------------------

DWELL_MEAN_S: dict[IntentType, dict[ManeuverMode, float]] = {
    IntentType.CORRIDOR_TRANSIT: {
        ManeuverMode.CV: 12.0,
        ManeuverMode.CT_LEFT: 4.0,
        ManeuverMode.CT_RIGHT: 4.0,
        ManeuverMode.CA: 5.0,
        ManeuverMode.HOVER: 3.0,
    },
    # 6 s of turning at a typical fixed-wing rate (~0.15-0.27 rad/s) is a
    # 50-90 deg leg change, which is what a waypoint transition physically is.
    IntentType.DELIVERY_WP: {
        ManeuverMode.CV: 6.0,
        ManeuverMode.CT_LEFT: 6.0,
        ManeuverMode.CT_RIGHT: 6.0,
        ManeuverMode.CA: 3.5,
        ManeuverMode.HOVER: 4.0,
    },
    IntentType.LOITER_HOVER: {
        ManeuverMode.CV: 3.0,
        ManeuverMode.CT_LEFT: 6.0,
        ManeuverMode.CT_RIGHT: 6.0,
        ManeuverMode.CA: 2.5,
        ManeuverMode.HOVER: 6.0,
    },
    IntentType.SEARCH_WANDER: {
        ManeuverMode.CV: 2.5,
        ManeuverMode.CT_LEFT: 3.0,
        ManeuverMode.CT_RIGHT: 3.0,
        ManeuverMode.CA: 2.5,
        ManeuverMode.HOVER: 3.0,
    },
    IntentType.REACTIVE_AVOID: {
        ManeuverMode.CV: 8.0,
        ManeuverMode.CT_LEFT: 3.5,
        ManeuverMode.CT_RIGHT: 3.5,
        ManeuverMode.CA: 3.0,
        ManeuverMode.HOVER: 3.0,
    },
    IntentType.PASSIVE_DRIFT: {
        ManeuverMode.CV: 20.0,
        ManeuverMode.CT_LEFT: 4.0,
        ManeuverMode.CT_RIGHT: 4.0,
        ManeuverMode.CA: 6.0,
        ManeuverMode.HOVER: 3.0,
    },
}

#: Coefficient of variation [-] of every dwell distribution.
DWELL_CV: float = 0.4

#: FALLBACK turn aggressiveness range [-] sampled on entering CT_LEFT /
#: CT_RIGHT, as a fraction of ``envelope.max_turn_rate(v)``. Doc-mandated
#: U(0.3, 1.0). Superseded per platform by
#: ``StabilityProfile.turn_fraction_range`` -- a bird commits to 0.4-1.0 of its
#: (very large) cap, an eVTOL to 0.15-0.45 of its (small) one.
#:
#: The machine reads the band off the profile, never off this constant, so the
#: two are pinned together below rather than left to drift apart silently.
TURN_FRACTION_RANGE: tuple[float, float] = (0.3, 1.0)


def _assert_default_profile_matches_the_documented_fallbacks() -> None:
    """Pin :data:`core.state.DEFAULT_STABILITY` to the doc-mandated fallbacks.

    The no-profile path must be exactly the behaviour that existed before
    manoeuvre commitment became a platform property, because that is what a
    hand-built envelope (unit test, GUI probe) still gets. If someone edits
    ``DEFAULT_STABILITY`` these constants stop describing anything, so the
    agreement is checked at import instead of documented in prose.

    Raises:
        AssertionError: if the default profile and the fallback constants
            disagree.
    """
    assert DEFAULT_STABILITY.turn_fraction_range == TURN_FRACTION_RANGE, (
        f"DEFAULT_STABILITY turn band {DEFAULT_STABILITY.turn_fraction_range} "
        f"!= documented fallback {TURN_FRACTION_RANGE}"
    )
    assert DEFAULT_STABILITY.dwell_min_s == DEFAULT_MIN_DWELL_S, (
        f"DEFAULT_STABILITY.dwell_min_s {DEFAULT_STABILITY.dwell_min_s} != "
        f"agents.behavior.DEFAULT_MIN_DWELL_S {DEFAULT_MIN_DWELL_S}"
    )
    assert DEFAULT_STABILITY.dwell_mean_s is None, (
        "DEFAULT_STABILITY must select the per-(intent, mode) DWELL_MEAN_S "
        "table, not a platform mean"
    )
    assert DEFAULT_STABILITY.mode_authority == 1.0, (
        "the no-profile path must keep the full CT override"
    )
    assert DEFAULT_STABILITY.diag_bias == 0.0, (
        "the no-profile path must use the unbiased transition matrix"
    )


_assert_default_profile_matches_the_documented_fallbacks()

#: Longitudinal acceleration range [-] sampled on entering CA, as a fraction
#: of ``envelope.a_max`` (speeding up) or ``|envelope.a_min|`` (slowing down).
ACCEL_FRACTION_RANGE: tuple[float, float] = (0.3, 1.0)

#: Duration [s] of the linear blend applied when LEAVING a coordinated turn.
CT_BLEND_S: float = 1.0

#: Gain [-] applied to the intent's turn command while in CV. Small but
#: non-zero: CV must suppress *deliberate* maneuvering yet still allow gentle
#: course keeping (holding a corridor line against a crosswind), otherwise an
#: agent in CV would be unable to correct at all and every heading change in
#: the dataset would carry a CT label, which is not what CV means.
CV_TRACK_GAIN: float = 0.25

#: Additional hard cap on the CV turn rate, as a fraction [-] of
#: ``envelope.max_turn_rate(v)``. This is the quantitative definition of
#: "nearly constant velocity" used for the ground-truth CV mode label.
CV_OMEGA_FRACTION: float = 0.15

#: First-order time constant [s] used by HOVER to bleed airspeed toward
#: ``envelope.v_min``.
HOVER_TAU_S: float = 1.0


# --------------------------------------------------------------------------
# Agent access helpers
# --------------------------------------------------------------------------


def _unpack_agent(agent: Any) -> tuple[AgentState, PerformanceEnvelope]:
    """Extract ``(state, envelope)`` from an agent-like object.

    Args:
        agent: an object exposing ``.state`` (:class:`core.state.AgentState`)
            and ``.envelope`` (or the alias ``.perf``, a
            :class:`core.state.PerformanceEnvelope`).

    Returns:
        The agent's kinematic state and performance envelope.

    Raises:
        TypeError: if either attribute is missing.
    """
    state = getattr(agent, "state", None)
    if state is None:
        raise TypeError(
            "ModeMachine requires an agent exposing '.state' "
            f"(core.state.AgentState); got {type(agent).__name__}"
        )
    env = getattr(agent, "envelope", None)
    if env is None:
        env = getattr(agent, "perf", None)
    if env is None:
        raise TypeError(
            "ModeMachine requires an agent exposing '.envelope' "
            f"(core.state.PerformanceEnvelope); got {type(agent).__name__}"
        )
    return state, env


# --------------------------------------------------------------------------
# The mode machine
# --------------------------------------------------------------------------


class ModeMachine:
    """Semi-Markov maneuver-mode process for one agent (doc 3.3).

    Lifecycle per integration step, driven by the caller (normally
    :mod:`simulation.engine` through :mod:`agents.base`)::

        machine.step(SIM_DT, agent, rng_mode, rng_branch)
        omega_cmd, a_cmd = machine.apply(omega_intent, a_intent, agent)
        # -> hand the commands to core.integrator

    The machine owns the *mode*; the intent owns the *goal*. What each mode
    does with the intent's commands is documented on :meth:`apply`.

    Attributes are exposed read-only through properties; use
    :meth:`snapshot` / :meth:`restore` to persist the machine inside a
    :class:`core.state.AgentSnapshot`.
    """

    __slots__ = (
        "_intent",
        "_mode",
        "_mode_timer",
        "_mode_params",
        "_blend_omega",
        "_blend_timer",
        "_transitions",
        "_dwell_means",
        "_stability",
        "_dwell_multiplier",
        "_rng_mode",
        "_rng_branch",
    )

    def __init__(
        self,
        intent: IntentType,
        mode: ManeuverMode = ManeuverMode.CV,
        mode_timer: float = 0.0,
        mode_params: Mapping[str, Any] | None = None,
        stability: StabilityProfile | None = None,
    ) -> None:
        """Create a mode machine.

        Args:
            intent: the agent's high-level intent; selects the transition
                matrix and the fallback dwell means.
            mode: initial maneuver mode.
            mode_timer: remaining dwell [s] of ``mode``. The default 0.0
                means "decide a new mode on the very first
                :meth:`step`", which is the normal way to start an episode.
            mode_params: initial mode payload (see :meth:`mode_params`).
            stability: the platform's manoeuvre-commitment profile. ``None``
                means "adopt it from the agent's envelope the first time one is
                seen" (see :meth:`_adopt_stability`), so a machine built
                without it still behaves like its platform. Passing it
                explicitly -- which :mod:`scenario.builder` does -- is
                preferred, because the transition matrix is biased once at
                construction rather than on the first step.

        Raises:
            KeyError: if ``intent`` has no transition matrix.
        """
        self._intent: IntentType = IntentType(int(intent))
        self._stability: StabilityProfile | None = stability
        self._dwell_multiplier: float = intent_dwell_multiplier(self._intent)
        self._transitions: np.ndarray = biased_transitions(
            self._intent, 0.0 if stability is None else stability.diag_bias
        )
        self._dwell_means: dict[ManeuverMode, float] = DWELL_MEAN_S[self._intent]
        self._mode: ManeuverMode = ManeuverMode(int(mode))
        self._mode_timer: float = float(mode_timer)
        self._mode_params: dict[str, Any] = dict(mode_params or {})
        # Blend bookkeeping: value of the turn-rate command at the instant the
        # last coordinated turn was left, and the remaining blend time [s].
        self._blend_omega: float = 0.0
        self._blend_timer: float = 0.0
        # Optional generators bound by :meth:`bind_streams`, used by the
        # single-generator host convention of :meth:`modulate`.
        self._rng_mode: np.random.Generator | None = None
        self._rng_branch: np.random.Generator | None = None

    # -- read-only view ---------------------------------------------------

    @property
    def intent(self) -> IntentType:
        """The intent whose transition matrix and dwell means are in use."""
        return self._intent

    @property
    def mode(self) -> ManeuverMode:
        """The maneuver mode currently being executed."""
        return self._mode

    @property
    def mode_timer(self) -> float:
        """Dwell time [s] remaining before the next mode decision."""
        return self._mode_timer

    @property
    def mode_params(self) -> dict[str, Any]:
        """Latched payload of the current mode (JSON-safe scalars).

        Keys, by mode:
            CT_LEFT / CT_RIGHT:
                ``omega_sign`` [-] ``+1`` left / ``-1`` right,
                ``omega_frac`` [-] fraction of the envelope turn-rate cap,
                ``omega_cmd`` [rad/s] the command evaluated at latch time.
            CA:
                ``a_frac`` [-] fraction of the envelope accel limit,
                ``a_sign`` [-] ``+1`` accelerate / ``-1`` decelerate,
                ``a_cmd`` [m/s^2] the latched longitudinal command.
            HOVER:
                ``v_target`` [m/s] the speed the mode drives toward.
            CV: empty.
        """
        return dict(self._mode_params)

    @property
    def blend_active(self) -> bool:
        """True while the post-coordinated-turn blend is still running."""
        return self._blend_timer > 0.0

    @property
    def stability(self) -> StabilityProfile:
        """The manoeuvre-commitment profile in force.

        Before an agent has been seen and none was passed to the constructor,
        this reports :data:`core.state.DEFAULT_STABILITY` -- which is also what
        the machine would use if it were stepped right now.
        """
        return DEFAULT_STABILITY if self._stability is None else self._stability

    @property
    def transitions(self) -> np.ndarray:
        """The EFFECTIVE (diagonal-biased) transition matrix [-]; read-only."""
        return self._transitions

    # -- lifecycle --------------------------------------------------------

    def initialize(
        self,
        agent: Any,
        rng_mode: np.random.Generator,
        rng_branch: np.random.Generator,
    ) -> None:
        """Latch parameters and a dwell for the CURRENT mode, without switching.

        Use this when an episode builder wants an agent to start already
        committed to a specific mode (e.g. a designed encounter that begins
        mid-turn). The ordinary path -- construct with ``mode_timer = 0.0``
        and let the first :meth:`step` decide -- does not need it.

        Args:
            agent: object exposing ``.state`` and ``.envelope``.
            rng_mode: ``mode_decision`` stream generator.
            rng_branch: ``branch_choice`` stream generator.
        """
        state, env = _unpack_agent(agent)
        self._adopt_stability(env)
        self._mode = self._legalise(self._mode, env)
        self._latch_params(self._mode, state, env, rng_mode, rng_branch)
        self._mode_timer = self._draw_dwell(self._mode, rng_mode)

    def step(
        self,
        dt: float,
        agent: Any,
        rng_mode: np.random.Generator,
        rng_branch: np.random.Generator,
    ) -> None:
        """Advance the mode process by ``dt`` [s].

        Runs down the dwell timer and, when it expires, samples the next mode
        from the intent's transition matrix. The transition is factorised
        into two independent draws so that the direction of a turn can be
        re-randomised on its own:

            1. ``rng_mode`` selects the mode *category* -- CV, "a coordinated
               turn", CA or HOVER -- using the row of the transition matrix
               with the two CT columns merged.
            2. ``rng_branch`` flips the left/right coin whenever the category
               is "a coordinated turn", with
               ``p_left = P(CT_LEFT) / (P(CT_LEFT) + P(CT_RIGHT))``.

        Step 2 is the fork that branch rollouts re-randomise, and factoring
        it out of step 1 is what guarantees that re-seeding ``branch_choice``
        alone changes turn directions while leaving the *rhythm* of the mode
        process (when decisions happen, how long they last) untouched.

        Modes illegal for the platform are masked out before the draw: HOVER
        requires a platform that can actually come to a stop (``can_hover``
        AND ``v_min <= EPS_V``; see :meth:`_can_hover`), and an unpowered body
        (``is_powered == False``) is pinned to CV. The dwell drawn for the new
        mode is truncated at the prediction horizon (see :meth:`_draw_dwell`).

        Args:
            dt: elapsed simulation time [s], normally
                :data:`core.constants.SIM_DT`.
            agent: object exposing ``.state`` and ``.envelope``.
            rng_mode: ``mode_decision`` stream generator.
            rng_branch: ``branch_choice`` stream generator.
        """
        state, env = _unpack_agent(agent)
        self._adopt_stability(env)
        d = float(dt)

        if self._blend_timer > 0.0:
            self._blend_timer = max(0.0, self._blend_timer - d)

        self._mode_timer -= d
        if self._mode_timer > 0.0:
            return

        previous = self._mode
        nxt = self._sample_next_mode(previous, env, rng_mode, rng_branch)

        # Leaving a coordinated turn: remember the command we were holding so
        # apply() can blend out of it instead of snapping the heading.
        if previous in (ManeuverMode.CT_LEFT, ManeuverMode.CT_RIGHT) and nxt != previous:
            self._blend_omega = self._ct_command(state, env)
            self._blend_timer = CT_BLEND_S
        elif nxt != previous:
            self._blend_omega = 0.0
            self._blend_timer = 0.0

        self._mode = nxt
        self._latch_params(nxt, state, env, rng_mode, rng_branch)
        # Carry the (small, negative) overshoot into the new dwell so that the
        # mode process does not slowly drift late relative to wall time.
        floor = self.stability.dwell_min_s
        overshoot = -self._mode_timer
        dwell = self._draw_dwell(nxt, rng_mode)
        self._mode_timer = max(0.5 * floor, dwell - overshoot)

    def force_mode(
        self,
        mode: ManeuverMode,
        agent: Any,
        rng_mode: np.random.Generator,
        rng_branch: np.random.Generator,
        dwell_s: float | None = None,
        omega_frac: float | None = None,
    ) -> None:
        """Override the mode immediately, bypassing the transition matrix.

        Intended for intents that must pre-empt the nominal mode process --
        principally ``IntentType.REACTIVE_AVOID``, where
        :func:`agents.behavior.reactive_avoid_decision` has already latched a
        side and the agent must commit to it now (doc 3.2 rev.2). The forced
        mode still respects the dwell floor, so a pre-empted maneuver is just
        as committed as a nominally chosen one.

        Args:
            mode: mode to switch to. It is legalised against the envelope
                (HOVER on a non-hovering platform falls back to CV).
            agent: object exposing ``.state`` and ``.envelope``.
            rng_mode: ``mode_decision`` stream generator (dwell, magnitudes).
            rng_branch: ``branch_choice`` stream generator (unused when the
                caller already fixed the direction, kept for signature
                symmetry with :meth:`step`).
            dwell_s: explicit dwell [s]; floored at the platform's
                ``StabilityProfile.dwell_min_s``. ``None`` samples from the
                platform's dwell distribution.
            omega_frac: explicit turn aggressiveness [-] for a CT mode as a
                fraction of the envelope cap. ``None`` samples from the
                platform's turn-fraction band.
        """
        state, env = _unpack_agent(agent)
        self._adopt_stability(env)
        target = self._legalise(ManeuverMode(int(mode)), env)
        previous = self._mode
        if previous in (ManeuverMode.CT_LEFT, ManeuverMode.CT_RIGHT) and target != previous:
            self._blend_omega = self._ct_command(state, env)
            self._blend_timer = CT_BLEND_S
        self._mode = target
        self._latch_params(target, state, env, rng_mode, rng_branch, omega_frac=omega_frac)
        floor = self.stability.dwell_min_s
        if dwell_s is None:
            self._mode_timer = self._draw_dwell(target, rng_mode)
        else:
            self._mode_timer = max(floor, float(dwell_s))

    # -- host adapter -----------------------------------------------------

    def bind_streams(
        self,
        rng_mode: np.random.Generator,
        rng_branch: np.random.Generator,
    ) -> None:
        """Bind the two decision streams for use by :meth:`modulate`.

        :meth:`step` takes both generators explicitly, but the agent host
        (:mod:`agents.base`) drives a mode machine with a SINGLE generator per
        step. Binding lets the episode builder hand the machine the real
        ``branch_choice`` stream up front, so the left/right coin keeps coming
        from the stream that :meth:`core.rng.RNGHub.reseed_branch`
        re-randomises. Without this the fork would be drawn from whatever
        single generator the host passes, and if that generator is a per-agent
        private child (``RNGHub.spawn_child``) rather than a hub stream, branch
        rollouts would replay identically and the "true" future distribution
        would collapse to a point.

        The bound generators are references, not copies, and are deliberately
        NOT part of :meth:`snapshot`: their state belongs to
        :meth:`core.rng.RNGHub.snapshot`.

        Args:
            rng_mode: the ``mode_decision`` stream generator.
            rng_branch: the ``branch_choice`` stream generator.
        """
        self._rng_mode = rng_mode
        self._rng_branch = rng_branch

    def modulate(
        self,
        agent: Any,
        t: float,
        dt: float,
        omega_cmd: float,
        a_cmd: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float]:
        """Tick the mode process and shape the intent's command, in one call.

        This is the entry point :mod:`agents.base` probes for first; it is a
        thin adapter over :meth:`step` followed by :meth:`apply`, returning the
        ``(omega_cmd, a_cmd)`` pair the host expects. Calling ``step`` alone
        would tick the timers but return ``None``, which the host reads as
        "the machine declined to override" -- the mode process would then have
        no effect on the trajectories at all.

        Stream selection, in priority order:
            * generators bound by :meth:`bind_streams` (correct for branch
              rollouts -- use this path);
            * otherwise ``rng`` for both the mode draw and the branch coin.

        Args:
            agent: object exposing ``.state`` and ``.envelope``.
            t: current simulation time [s]. Accepted for the host's calling
               convention; the mode process is time-invariant and depends only
               on elapsed ``dt``, so it is not used.
            dt: elapsed simulation time [s] since the previous call.
            omega_cmd: turn rate [rad/s] proposed by the intent layer.
            a_cmd: longitudinal acceleration [m/s^2] proposed by the intent.
            rng: fallback generator used when no streams are bound.

        Returns:
            ``(omega_cmd, a_cmd)`` in [rad/s] and [m/s^2], clipped to the
            envelope.

        Raises:
            ValueError: if no generator is available from either source.
        """
        rng_mode = self._rng_mode if self._rng_mode is not None else rng
        if rng_mode is None:
            raise ValueError(
                "ModeMachine.modulate needs a generator: call bind_streams() "
                "or pass rng="
            )
        rng_branch = self._rng_branch if self._rng_branch is not None else rng_mode
        del t  # time-invariant process; accepted only for the host convention
        self.step(dt, agent, rng_mode, rng_branch)
        return self.apply(omega_cmd, a_cmd, agent)

    # -- command shaping --------------------------------------------------

    def apply(
        self,
        omega_cmd_from_intent: float,
        a_cmd_from_intent: float,
        agent: Any,
    ) -> tuple[float, float]:
        """Combine the intent's commands with the current maneuver mode.

        Per-mode behaviour:

        ``CV``
            The acceleration passes through unchanged; the turn command is
            *suppressed* -- scaled by :data:`CV_TRACK_GAIN` and hard-capped
            at :data:`CV_OMEGA_FRACTION` of the envelope's turn-rate limit.
            The residual authority exists only for gentle course keeping;
            what CV must not do is execute a deliberate maneuver, because the
            ground-truth mode label ``CV`` is a claim about the trajectory.

        ``CT_LEFT`` / ``CT_RIGHT``
            The latched target turn rate is combined with the intent's
            request in proportion to the platform's ``mode_authority``::

                omega = (1 - authority) * omega_intent + authority * omega_mode

            At ``authority = 1`` this is the pure override the mode process
            was built on: the maneuver becomes a genuine commitment ("turn
            left at 60% of my limit for the next 4.2 s") instead of a nudge
            that the intent's waypoint tracking would immediately cancel, and
            committed turns are what a constant-velocity extrapolator cannot
            follow. A bird (0.90) is essentially there. But a full override on
            a STABLE platform is wrong: it fights the corridor guidance and
            makes a passenger eVTOL wander off its centreline, which doc 7.2
            calls out as the artefact reviewers spot. At 0.35 the mode still
            biases the aircraft -- the manoeuvre is real and labelled -- while
            the guidance keeps it on the corridor. Both inputs are already
            inside the envelope, so their convex combination is too. The
            acceleration passes through.

        ``CA``
            The turn command passes through (so the agent may still track its
            intent while changing speed); the acceleration is **overridden**
            by the latched value.

        ``HOVER``
            Legal only when ``envelope.can_hover``; otherwise this method
            degrades to CV rather than commanding an impossible state. The
            acceleration is overridden by a first-order law driving the
            airspeed toward ``envelope.v_min`` with time constant
            :data:`HOVER_TAU_S`; the (suppressed) intent turn command passes
            through so the agent can still point where it wants while
            stopping.

        **Blend out of a coordinated turn.** Snapping the turn-rate command
        from a latched CT value to zero would produce a corner in the heading
        trace at the exact instants where the mode label changes -- an
        artefact a predictor can exploit and a physically wrong signal. So
        for :data:`CT_BLEND_S` (1.0 s) after leaving a CT mode the returned
        turn rate is a linear interpolation::

            f     = blend_timer / CT_BLEND_S            # 1 -> 0
            omega = f * omega_at_exit + (1 - f) * omega_of_new_mode

        The blend is applied in the command domain only. The integrator's own
        ``omega_dot_max`` rate limit still applies on top of it, so the
        result can never violate the roll-onset limit; the blend simply
        prevents the command itself from being discontinuous. No blend is
        applied when *entering* a CT mode, because the rate limiter alone
        gives a physically correct roll-in and the entry is a real, labelled
        event.

        Args:
            omega_cmd_from_intent: turn rate [rad/s] requested by the intent
                layer, positive = left / CCW.
            a_cmd_from_intent: longitudinal acceleration [m/s^2] requested by
                the intent layer.
            agent: object exposing ``.state`` and ``.envelope``.

        Returns:
            ``(omega_cmd, a_cmd)`` in [rad/s] and [m/s^2], both already
            clipped to the platform envelope at the current airspeed.
        """
        state, env = _unpack_agent(agent)
        self._adopt_stability(env)
        v = float(state.v)
        omega_intent = float(omega_cmd_from_intent)
        a_intent = float(a_cmd_from_intent)
        mode = self._mode

        if mode == ManeuverMode.CT_LEFT or mode == ManeuverMode.CT_RIGHT:
            authority = self.stability.mode_authority
            omega_mode = self._ct_command(state, env)
            omega = (1.0 - authority) * omega_intent + authority * omega_mode
            a_cmd = a_intent
        elif mode == ManeuverMode.CA:
            omega = self._cv_omega(omega_intent, v, env)
            a_cmd = float(self._mode_params.get("a_cmd", a_intent))
        elif mode == ManeuverMode.HOVER and env.can_hover:
            omega = self._cv_omega(omega_intent, v, env)
            v_target = float(self._mode_params.get("v_target", env.v_min))
            a_cmd = (v_target - v) / HOVER_TAU_S
        else:  # CV, or HOVER requested by a platform that cannot hover
            omega = self._cv_omega(omega_intent, v, env)
            a_cmd = a_intent

        if self._blend_timer > 0.0:
            f = self._blend_timer / CT_BLEND_S
            if f > 1.0:
                f = 1.0
            omega = f * self._blend_omega + (1.0 - f) * omega

        return env.clamp_turn_rate(omega, v), env.clamp_accel(a_cmd)

    # -- persistence ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture the machine state as a JSON-safe dict.

        The result is what :class:`core.state.AgentSnapshot` stores in
        ``mode`` / ``mode_timer`` / ``mode_params``, plus the blend
        bookkeeping. Blend state must be included: restoring without it would
        make a branch rollout that starts mid-blend take a slightly different
        path from the parent run, which would show up as spurious spread in
        the calibration reference distribution.

        :class:`core.state.AgentSnapshot` has fields for ``mode``,
        ``mode_timer`` and ``mode_params`` but none for the blend, so when a
        blend is in progress its two values are ALSO mirrored into
        ``mode_params`` under ``_blend_omega`` / ``_blend_timer``.
        :meth:`restore` reads either location. That way a caller which
        persists only the three ``AgentSnapshot`` fields still round-trips a
        mid-blend machine exactly. The mirrored keys are written only while a
        blend is live (a fraction of the timeline) and are cleared on the next
        mode entry, since :meth:`_latch_params` replaces ``mode_params``
        wholesale.

        The stability profile is carried too. It is immutable for the lifetime
        of an agent, but a machine that has not been stepped yet has not
        adopted one, so a snapshot taken before the first step and one taken
        after must still restore to the same behaviour -- and a snapshot must
        never depend on an envelope the restoring caller may not have.

        Returns:
            ``{"intent", "mode", "mode_timer", "mode_params",
            "blend_omega", "blend_timer", "stability"}``.
        """
        params = dict(self._mode_params)
        if self._blend_timer > 0.0:
            params["_blend_omega"] = float(self._blend_omega)
            params["_blend_timer"] = float(self._blend_timer)
        return {
            "intent": int(self._intent),
            "mode": int(self._mode),
            "mode_timer": float(self._mode_timer),
            "mode_params": params,
            "blend_omega": float(self._blend_omega),
            "blend_timer": float(self._blend_timer),
            "stability": (None if self._stability is None
                          else self._stability.to_dict()),
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        """Restore from a :meth:`snapshot` result.

        Accepts either a full :meth:`snapshot` dict or the reduced form held
        by :class:`core.state.AgentSnapshot` (``mode`` / ``mode_timer`` /
        ``mode_params``), recovering the blend from the mirrored
        ``_blend_omega`` / ``_blend_timer`` keys inside ``mode_params`` when
        the top-level entries are absent.

        Args:
            snap: dict produced by :meth:`snapshot`. ``intent`` is honoured
                if present, so a snapshot fully re-creates the machine.
        """
        if "intent" in snap:
            self._intent = IntentType(int(snap["intent"]))
            self._dwell_means = DWELL_MEAN_S[self._intent]
            self._dwell_multiplier = intent_dwell_multiplier(self._intent)
        if "stability" in snap:
            raw = snap["stability"]
            self._stability = (None if raw is None
                               else StabilityProfile.from_dict(dict(raw)))
        # Rebuild the biased matrix from whatever (intent, profile) now hold:
        # restoring the base matrix here would silently drop a platform's
        # diagonal bias, and every branch rollout starts with a restore.
        self._transitions = biased_transitions(
            self._intent,
            0.0 if self._stability is None else self._stability.diag_bias,
        )
        self._mode = ManeuverMode(int(snap.get("mode", ManeuverMode.CV)))
        self._mode_timer = float(snap.get("mode_timer", 0.0))
        params = dict(snap.get("mode_params", {}))
        mirrored_omega = params.pop("_blend_omega", None)
        mirrored_timer = params.pop("_blend_timer", None)
        self._mode_params = params
        if "blend_omega" in snap:
            self._blend_omega = float(snap["blend_omega"])
        else:
            self._blend_omega = float(mirrored_omega or 0.0)
        if "blend_timer" in snap:
            self._blend_timer = float(snap["blend_timer"])
        else:
            self._blend_timer = float(mirrored_timer or 0.0)

    @classmethod
    def from_snapshot(cls, snap: Mapping[str, Any]) -> "ModeMachine":
        """Build a new machine directly from a :meth:`snapshot` result."""
        machine = cls(IntentType(int(snap.get("intent", IntentType.CORRIDOR_TRANSIT))))
        machine.restore(snap)
        return machine

    # -- internals --------------------------------------------------------

    def _adopt_stability(self, env: PerformanceEnvelope) -> None:
        """Late-bind the platform's stability profile from an agent's envelope.

        The profile belongs to the airframe, so the envelope is its natural
        carrier; but a machine can legitimately be constructed before any agent
        exists (an episode builder, a snapshot, a unit test), which is why the
        constructor argument is optional. The first time a machine sees an
        agent it takes the envelope's profile and re-derives the biased
        transition matrix from it.

        This runs at most once per machine and never overrides an explicitly
        supplied profile, so it cannot change behaviour mid-episode -- which
        would break branch rollouts, since the profile would then be mutable
        state that :meth:`snapshot` has to reproduce at exactly the right
        moment.

        Args:
            env: the agent's performance envelope.
        """
        if self._stability is not None:
            return
        profile = getattr(env, "stability", None)
        if profile is None:
            profile = DEFAULT_STABILITY
        self._stability = profile
        self._transitions = biased_transitions(self._intent, profile.diag_bias)

    def _draw_dwell(self, mode: ManeuverMode, rng_mode: np.random.Generator) -> float:
        """Sample a dwell [s] for ``mode``, truncated at the prediction horizon.

        The lognormal dwell has a long right tail. When an agent's remaining
        dwell outlives the 15 s prediction horizon AND its intent supplies no
        continuous stochastic term (delivery waypoints, corridor cross-track),
        NO decision can occur inside the horizon and the empirical future
        distribution measured by :func:`simulation.rollout.branch_rollout`
        collapses to a literal point mass -- measured at 200 branches producing
        2 distinct endpoints and a spread of 4e-4 m. A calibration target with
        sigma == 0 tells a model the correct answer is "no uncertainty", which
        is exactly the degeneracy doc 3.3 exists to prevent.

        Truncating at :data:`core.constants.PRED_SECONDS` guarantees every
        powered agent reaches at least one decision epoch inside every
        prediction window. It does NOT make stable platforms twitchy: what
        governs how long an aircraft actually flies straight is the diagonal
        of its (platform-biased) transition matrix, not the dwell tail. An
        eVTOL in corridor cruise re-decides with P(stay CV) = 0.909, so its
        expected uninterrupted cruise leg is 15 / (1 - 0.909) ~ 165 s, far
        longer than a 60 s episode -- and a CV -> CV re-latch has an empty
        parameter payload, i.e. it is a no-op on the trajectory.

        The truncation never fights the platform's commitment floor: the cap
        is raised to ``dwell_min_s`` for any platform whose floor is already
        at or above the horizon.

        The tail is REDRAWN uniformly inside ``[floor, cap]``, never clamped to
        ``cap``. Clamping puts an atom of probability on exactly one dwell
        length, and for a platform whose mean dwell is well above the horizon
        (the eVTOL, mean 25 s) that atom carries most of the mass -- so its
        mode changes land on an exact 15 s grid, resonate with the branch
        times, and the fork ends up at the very last step of the prediction
        window where it has had no time to displace anything. Measured: eight
        of sixteen frozen agents sat at ``mode_timer == 15.00`` exactly. A
        redraw removes the atom and the resonance with it.

        Args:
            mode: the mode being entered.
            rng_mode: ``mode_decision`` stream generator.

        Returns:
            The dwell [s], in ``[dwell_min_s, max(PRED_SECONDS, dwell_min_s)]``.
        """
        floor = float(self.stability.dwell_min_s)
        dwell = sample_dwell(rng_mode, self._dwell_mean(mode), DWELL_CV, floor)
        cap = PRED_SECONDS if PRED_SECONDS > floor else floor
        if dwell <= cap:
            return dwell
        if cap <= floor:
            return cap
        return float(rng_mode.uniform(floor, cap))

    def _dwell_mean(self, mode: ManeuverMode) -> float:
        """Mean dwell [s] for ``mode`` under the current platform and intent.

        With a per-platform profile the mean is
        ``profile.dwell_mean_s * INTENT_DWELL_MULTIPLIER[intent]`` and does not
        depend on the mode: how long an object commits to a manoeuvre is a
        property of the object and its mission, not of which manoeuvre it
        happens to be flying. Without one (a hand-built envelope) the legacy
        per-(intent, mode) table :data:`DWELL_MEAN_S` is used instead.

        Args:
            mode: the mode whose dwell is being drawn.

        Returns:
            The mean dwell [s]; the platform's floor is applied afterwards, by
            :func:`agents.behavior.sample_dwell`.
        """
        fallback = float(self._dwell_means.get(mode, 4.0))
        return self.stability.dwell_mean_for(self._dwell_multiplier, fallback)

    @staticmethod
    def _can_hover(env: PerformanceEnvelope) -> bool:
        """True if the platform can actually COME TO A STOP, not merely tilt.

        ``can_hover`` is an airframe-capability flag: an eVTOL has rotors and
        can hover on paper, but its cruise envelope has ``v_min`` around
        30 m/s, so a HOVER command only drives it to its minimum cruise speed.
        Entering HOVER anyway wrote ``gt_mode = HOVER`` onto eVTOLs cruising at
        31-69 m/s and onto UNKNOWN bodies at up to 101 m/s. ``gt_mode`` is an
        auxiliary supervised label (doc 5.1); a mode head trained on it would
        learn that HOVER sometimes means 69 m/s. Requiring the speed envelope
        to reach a standstill keeps the flag intact for every other consumer
        while making the LABEL truthful.
        """
        return bool(env.can_hover) and float(env.v_min) <= EPS_V

    @classmethod
    def _legalise(cls, mode: ManeuverMode, env: PerformanceEnvelope) -> ManeuverMode:
        """Map a requested mode onto one the platform can actually fly."""
        if not env.is_powered:
            return ManeuverMode.CV
        if mode == ManeuverMode.HOVER and not cls._can_hover(env):
            return ManeuverMode.CV
        return mode

    def _allowed_mask(self, env: PerformanceEnvelope) -> np.ndarray:
        """Boolean mask [5] of modes this platform may enter."""
        mask = np.ones(_N_MODES, dtype=bool)
        if not self._can_hover(env):
            mask[_I_HOVER] = False
        if not env.is_powered:
            mask[:] = False
            mask[_I_CV] = True
        return mask

    def _sample_next_mode(
        self,
        current: ManeuverMode,
        env: PerformanceEnvelope,
        rng_mode: np.random.Generator,
        rng_branch: np.random.Generator,
    ) -> ManeuverMode:
        """Draw the next mode: category from ``rng_mode``, side from ``rng_branch``."""
        row = self._transitions[MODE_INDEX[current]].copy()
        row *= self._allowed_mask(env)
        total = float(row.sum())
        if total <= 0.0:
            return ManeuverMode.CV
        row /= total

        p_ct = float(row[_I_CTL] + row[_I_CTR])
        # Category weights: [CV, any-CT, CA, HOVER]. Merging the two CT
        # columns is what removes the direction decision from this draw.
        cat_weights = (float(row[_I_CV]), p_ct, float(row[_I_CA]), float(row[_I_HOVER]))
        cat = categorical(rng_mode, cat_weights)

        if cat == 0:
            return ManeuverMode.CV
        if cat == 2:
            return ManeuverMode.CA
        if cat == 3:
            return ManeuverMode.HOVER
        # cat == 1: a coordinated turn. THE fork -- drawn from branch_choice.
        p_left = 0.5 if p_ct <= 0.0 else float(row[_I_CTL]) / p_ct
        sign = binary_branch(rng_branch, p_left)
        return ManeuverMode.CT_LEFT if sign > 0 else ManeuverMode.CT_RIGHT

    def _latch_params(
        self,
        mode: ManeuverMode,
        state: AgentState,
        env: PerformanceEnvelope,
        rng_mode: np.random.Generator,
        rng_branch: np.random.Generator,
        omega_frac: float | None = None,
    ) -> None:
        """Sample and store the payload of a freshly entered mode."""
        v = float(state.v)
        if mode == ManeuverMode.CT_LEFT or mode == ManeuverMode.CT_RIGHT:
            frac = (
                sample_fraction(rng_mode, *self.stability.turn_fraction_range)
                if omega_frac is None
                else float(omega_frac)
            )
            sign = 1 if mode == ManeuverMode.CT_LEFT else -1
            self._mode_params = {
                "omega_sign": int(sign),
                "omega_frac": float(frac),
                "omega_cmd": float(sign * frac * env.max_turn_rate(v)),
            }
        elif mode == ManeuverMode.CA:
            frac = sample_fraction(rng_mode, *ACCEL_FRACTION_RANGE)
            sign = self._accel_sign(state, env, rng_mode)
            a_cmd = frac * (env.a_max if sign > 0 else env.a_min)
            self._mode_params = {
                "a_sign": int(sign),
                "a_frac": float(frac),
                "a_cmd": float(env.clamp_accel(a_cmd)),
            }
        elif mode == ManeuverMode.HOVER:
            self._mode_params = {"v_target": float(env.v_min)}
        else:
            self._mode_params = {}

    @staticmethod
    def _accel_sign(
        state: AgentState,
        env: PerformanceEnvelope,
        rng_mode: np.random.Generator,
    ) -> int:
        """Choose accelerate (+1) or decelerate (-1) for a CA mode.

        A fair coin, except at the edges of the speed envelope: an agent
        already at 90% of ``v_max`` may only decelerate and one near ``v_min``
        may only accelerate. Without that guard a large share of CA segments
        would be immediately saturated by the speed clamp and would be
        indistinguishable from CV, polluting the mode labels.
        """
        v = float(state.v)
        span = float(env.v_max) - float(env.v_min)
        if span <= 0.0:
            return -1
        if v >= env.v_min + 0.9 * span:
            return -1
        if v <= env.v_min + 0.1 * span:
            return 1
        return 1 if float(rng_mode.random()) < 0.5 else -1

    def _ct_command(self, state: AgentState, env: PerformanceEnvelope) -> float:
        """Latched coordinated-turn command [rad/s] at the CURRENT airspeed.

        The *fraction* of the envelope is what is latched, not the absolute
        turn rate: re-evaluating ``frac * max_turn_rate(v)`` every step keeps
        the command physically admissible as the airspeed changes (a
        bank-limited platform loses turn rate as it speeds up), which is
        constraint (a) of doc 3.3.
        """
        params = self._mode_params
        sign = float(params.get("omega_sign", 1 if self._mode == ManeuverMode.CT_LEFT else -1))
        frac = float(params.get("omega_frac", self.stability.turn_fraction_range[0]))
        return sign * frac * env.max_turn_rate(float(state.v))

    @staticmethod
    def _cv_omega(omega_intent: float, v: float, env: PerformanceEnvelope) -> float:
        """Suppressed turn command [rad/s] used by CV / CA / HOVER."""
        cap = CV_OMEGA_FRACTION * env.max_turn_rate(v)
        omega = CV_TRACK_GAIN * float(omega_intent)
        if omega > cap:
            return cap
        if omega < -cap:
            return -cap
        return omega

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        prof = self.stability
        return (
            f"ModeMachine(intent={self._intent.name}, mode={self._mode.name}, "
            f"timer={self._mode_timer:.2f}s, authority={prof.mode_authority:.2f}, "
            f"dwell_min={prof.dwell_min_s:.1f}s, params={self._mode_params})"
        )
