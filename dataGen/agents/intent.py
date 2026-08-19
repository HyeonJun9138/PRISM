"""Intent layer: "what is this object trying to do" (doc 3.2 rev.2).

An :class:`IntentController` is the *policy* of one traffic agent. Every
simulation step it converts the agent's own kinematic state (and, for
``REACTIVE_AVOID`` only, the ownship state) into a pair of low-level
commands::

    (omega_cmd [rad/s], a_cmd [m/s^2])

``omega_cmd`` is a *commanded* turn rate: positive = left = CCW = increasing
psi. It is always pre-clipped here to the platform's speed-dependent
:meth:`~core.state.PerformanceEnvelope.max_turn_rate`, so the integrator only
has to apply the roll-onset rate limit ``omega_dot_max``. ``a_cmd`` is a
commanded longitudinal acceleration, pre-clipped to ``[a_min, a_max]``.
``PASSIVE_DRIFT`` returns ``(None, None)``: that is the sentinel telling
``Agent.step`` to route the body through the passive (drag/advection)
integrator instead of the powered one.

Design rule (doc 3.1). The purpose of these policies is NOT realism, it is to
embed structure a constant-velocity extrapolator cannot reproduce, and to do
so through *stochastic decision rules* (doc 3.3) rather than hand-placed
waypoints on a map. The six intents are deliberately spread over a difficulty
range, because the calibration story needs an easy end and a hard end:

    CORRIDOR_TRANSIT  easiest  -- near-CV, tiny centreline corrections
    DELIVERY_WP       medium   -- periodic committed turns, stochastic route
    LOITER_HOVER      medium   -- persistent curvature (orbit) or near-standstill
    REACTIVE_AVOID    hard     -- gentle wander + latched evasive turns
    SEARCH_WANDER     hardest  -- OU heading walk + Poisson branch turns
    PASSIVE_DRIFT     n/a      -- unpowered, no commands at all

Branch-rollout correctness (doc 3.3). Every mutable value a controller owns
lives in a single JSON-safe dict, returned whole by :meth:`snapshot` and
restored whole by :meth:`restore`. Nothing that influences the future may live
in a Python attribute outside that dict -- an omission there does not raise,
it silently makes the N branch rollouts start from slightly different
situations, which would corrupt the "true distribution" used to score
calibration. The generated route of a ``DELIVERY_WP`` agent, the loiter centre
of a ``LOITER_HOVER`` agent and the OU value of a ``SEARCH_WANDER`` agent are
therefore all part of the snapshot, not construction-time attributes.

Compatibility fallback table (applied by :func:`resolve_intent_type` and
:func:`make_intent`; the *requested* type is kept on the controller as
``requested_type`` so the GUI and the metadata can show that a substitution
happened):

    requested          condition                         effective
    ----------------   -------------------------------   -----------------
    any                envelope.is_powered is False      PASSIVE_DRIFT
    PASSIVE_DRIFT      envelope.is_powered is True       CORRIDOR_TRANSIT
    LOITER_HOVER       envelope.can_hover is False       LOITER_HOVER
                                                         (orbit variant --
                                                          same class, the
                                                          hover variant is
                                                          simply not selected)
    anything else      --                                unchanged

The ``LOITER_HOVER`` case is handled *inside* the controller rather than by
swapping the class, so the stored ``gt_intent`` label stays truthful: a
fixed-wing loitering on a circular orbit is still "loiter", it just cannot
station-keep at zero airspeed.

Units are SI throughout: metres, seconds, radians, m/s, m/s^2.
"""

from __future__ import annotations

import importlib
import inspect
import math
from typing import Any, Callable, Mapping, Optional

import numpy as np

from agents.behavior import new_avoid_latch, reactive_avoid_decision
from core.constants import DWC_HMD_M, DWC_TAU_MOD_S, EPS_V, SIM_DT
from core.state import AgentState, PerformanceEnvelope, angle_diff, wrap_angle
from core.types import INTENT_NAMES, IntentType, PlatformType

__all__ = [
    "cross_track_guidance",
    "orbit_guidance",
    "heading_rate_command",
    "speed_accel_command",
    "ou_step",
    "IntentController",
    "CorridorTransitIntent",
    "DeliveryWaypointIntent",
    "LoiterHoverIntent",
    "SearchWanderIntent",
    "ReactiveAvoidIntent",
    "PassiveDriftIntent",
    "INTENT_CLASSES",
    "INTENT_FALLBACK_DOC",
    "resolve_intent_type",
    "PLATFORM_INTENT_COMPATIBILITY",
    "compatible_intents",
    "make_intent",
]


# ==========================================================================
# Tuning constants. Every value is an ASSUMED behavioural parameter (doc 12
# provenance table) -- they shape difficulty, not physics, and physics is
# enforced separately by PerformanceEnvelope.
# ==========================================================================

#: Fallback square-region side [m] when the world exposes no extent.
DEFAULT_REGION_SIDE_M: float = 2000.0

#: Fallback integration step [s] used on the very first command() call, before
#: two successive ``t`` values are available to difference.
DEFAULT_DT_S: float = SIM_DT

#: Largest plausible step [s]; anything above is treated as a discontinuity
#: (episode restart / snapshot restore) and replaced by DEFAULT_DT_S.
MAX_PLAUSIBLE_DT_S: float = 1.0

# -- CORRIDOR_TRANSIT ------------------------------------------------------
CORRIDOR_HEADING_GAIN: float = 0.35        # [1/s] heading error -> turn rate
CORRIDOR_CROSS_GAIN: float = 0.06          # [1/s] cross-track error gain
CORRIDOR_MAX_INTERCEPT_RAD: float = 25.0 * math.pi / 180.0
CORRIDOR_OMEGA_FRACTION: float = 0.30      # [-] of the envelope turn-rate cap
CORRIDOR_OFFSET_M: float = 120.0           # [m] initial lateral offset spread
CORRIDOR_HEADING_DEV_RAD: float = 8.0 * math.pi / 180.0
CORRIDOR_SPEED_TIMER_MIN_S: float = 10.0
CORRIDOR_SPEED_TIMER_MAX_S: float = 30.0
CORRIDOR_SPEED_SIGMA_FRAC: float = 0.04    # [-] of nominal cruise speed
CORRIDOR_SPEED_GAIN: float = 0.25          # [1/s]
CORRIDOR_GOAL_RANGE_M: float = 20000.0     # [m] distance of the notional goal

# -- DELIVERY_WP -----------------------------------------------------------
WP_CAPTURE_TURN_FACTOR: float = 3.0        # [-] capture radius / min turn radius
WP_CAPTURE_MIN_M: float = 60.0
WP_CAPTURE_REGION_FRACTION: float = 0.25   # [-] cap relative to region side
WP_LEG_FACTOR_MIN: float = 1.6             # [-] leg length / capture radius
WP_LEG_FACTOR_MAX: float = 3.2
WP_LEG_MIN_M: float = 250.0
WP_TURN_MIN_RAD: float = 30.0 * math.pi / 180.0
WP_TURN_MAX_RAD: float = 120.0 * math.pi / 180.0
WP_N_MIN: int = 4
WP_N_MAX: int = 8                          # inclusive
WP_MAX_DWELL_S: float = 45.0               # [s] forced advance guard
WP_REPLAN_PROB: float = 0.35               # [-] branch point at each capture
WP_HEADING_GAIN: float = 0.9               # [1/s]
WP_OMEGA_FRACTION: float = 0.85            # [-]
WP_SPEED_GAIN: float = 0.4                 # [1/s]
WP_LIST_TRIM_LEN: int = 64                 # [-] bound the stored route length

# -- LOITER_HOVER ----------------------------------------------------------
LOITER_RADIUS_FACTOR_MIN: float = 1.2      # [-] orbit radius / min turn radius
LOITER_RADIUS_FACTOR_MAX: float = 2.5
LOITER_RADIUS_MIN_M: float = 60.0
LOITER_ORBIT_FIELD_GAIN: float = 1.0       # [-] vector-field convergence gain
LOITER_HEADING_GAIN: float = 0.7           # [1/s]
LOITER_OMEGA_FRACTION: float = 1.0         # [-] orbit needs the full cap
LOITER_SPEED_GAIN: float = 0.4             # [1/s]
HOVER_SPEED_MARGIN_MS: float = 0.5         # [m/s] above v_min while hovering
HOVER_SPEED_GAIN: float = 0.8              # [1/s]
HOVER_DRIFT_TAU_S: float = 6.0             # [s] OU correlation time
HOVER_DRIFT_SIGMA_FRAC: float = 0.10       # [-] of the turn-rate cap
HOVER_STATION_RADIUS_M: float = 80.0       # [m] tolerated station drift
HOVER_RETURN_GAIN: float = 0.5             # [1/s]

# -- SEARCH_WANDER ---------------------------------------------------------
SEARCH_OU_TAU_S: float = 3.0
SEARCH_OU_SIGMA_FRAC: float = 0.30         # [-] of the turn-rate cap
SEARCH_REORIENT_RATE_HZ: float = 1.0 / 7.0 # [1/s] Poisson intensity
SEARCH_TURN_MIN_RAD: float = 35.0 * math.pi / 180.0
SEARCH_TURN_MAX_RAD: float = 145.0 * math.pi / 180.0
SEARCH_TURN_OMEGA_FRACTION: float = 0.65   # [-]
SEARCH_OU_OMEGA_FRACTION: float = 0.55     # [-]
SEARCH_SPEED_TIMER_MIN_S: float = 6.0
SEARCH_SPEED_TIMER_MAX_S: float = 18.0
SEARCH_SPEED_LO_FRAC: float = 0.45         # [-] of v_max
SEARCH_SPEED_HI_FRAC: float = 1.0
SEARCH_SPEED_GAIN: float = 0.5             # [1/s]

# -- REACTIVE_AVOID --------------------------------------------------------
RA_OU_TAU_S: float = 5.0
RA_OU_SIGMA_FRAC: float = 0.18             # [-] gentler baseline wander
RA_REORIENT_RATE_HZ: float = 1.0 / 12.0
RA_TURN_MIN_RAD: float = 25.0 * math.pi / 180.0
RA_TURN_MAX_RAD: float = 80.0 * math.pi / 180.0
RA_TURN_OMEGA_FRACTION: float = 0.50
RA_OU_OMEGA_FRACTION: float = 0.45
# The trigger is a conjunction of range, closure, time-to-CPA and projected
# miss distance. With the original (700 m, 18 s, 250 m) thresholds the three
# conditions almost never co-occurred for traffic that was not deliberately
# aimed at the ownship: 68 REACTIVE_AVOID agents over 10 test_ood episodes
# produced 9 reactions, and ~87% of agents carrying the REACTIVE_AVOID label
# flew a plain wander for the whole 60 s -- behaviourally identical to
# SEARCH_WANDER, which makes the intent label uninformative for most of its
# own class and removes doc 3.2's interaction structure ("futures that depend
# on the ownship's own choices") from the data.
#
# The relaxed thresholds are anchored to the study's own separation criteria
# rather than picked: an agent reacts exactly when the developing geometry
# would cost DAA WELL CLEAR within the well-clear lookahead -- the projected
# miss is below :data:`core.constants.DWC_HMD_M` and the CPA arrives within
# :data:`core.constants.DWC_TAU_MOD_S`. That is the same criterion doc 9.4
# uses to call an encounter unsafe, so the reaction fires on precisely the
# encounters the paper cares about.
#
# There is a STRUCTURAL ceiling on how often this can fire, and it is not a
# threshold problem: over 10 test_ood episodes only 76% of REACTIVE_AVOID
# agents ever come within 2500 m of the ownship at all and only 52% ever
# project a CPA inside the well-clear HMD. Raising the rate to "every agent
# reacts every episode" would mean aiming every reactive agent at the ownship,
# which fabricates the interaction instead of letting it emerge from the
# geometry -- the opposite of doc 3.3. What IS asserted (see
# :data:`RA_MIN_TRIGGER_RATE`) is that essentially every agent whose geometry
# qualifies does react.
RA_DETECT_RANGE_M: float = 2500.0          # [m] reaction distance
RA_TCPA_MAX_S: float = DWC_TAU_MOD_S       # [s] only react to imminent CPA
RA_CPA_TRIGGER_M: float = DWC_HMD_M        # [m] projected miss that alarms
RA_LATCH_MIN_S: float = 4.0
RA_LATCH_MAX_S: float = 8.0
RA_OMEGA_FRACTION: float = 0.85            # [-] evasive turn authority
RA_REFRACTORY_S: float = 4.0               # [s] relax time after a latch
#: Minimum reactions per REACTIVE_AVOID agent per episode below which the
#: intent label is no longer carrying its own behaviour and the class is
#: indistinguishable from SEARCH_WANDER. Measured 0.41; the old broken trigger
#: gave 0.13. Asserted by ``main.py selftest``.
RA_MIN_TRIGGER_RATE: float = 0.25
#: Minimum share [-] of REACTIVE_AVOID agents that react at least once.
#: Measured 0.31; the old broken trigger gave 0.10.
RA_MIN_REACTING_FRACTION: float = 0.20
#: Minimum share [-] of agents WHOSE GEOMETRY QUALIFIES that must react. This
#: is the assertion with teeth: the raw rate above is capped by how often the
#: ownship actually develops a conflict with a given agent (a scenario
#: property), whereas this one is entirely a property of the reaction model.
#: Measured 1.00 (33 of 33).
RA_MIN_CONDITIONAL_FRACTION: float = 0.90


# ==========================================================================
# agents.behavior adapter
# ==========================================================================
#
# ``agents/behavior.py`` was written by a parallel task, so ``binary_branch``
# (the stochastic left/right coin that creates multimodality) is still bound
# lazily and defensively: the module is imported on first use, its signature is
# inspected and the arguments are bound BY NAME from a pool of aliases, and the
# return value is normalised (bool / +-1 / 0-1). A missing module falls back to
# the local reference implementation, which has the documented semantics, so
# behaviour is identical either way up to the exact random draws.
#
# ``reactive_avoid_decision`` is NOT bound this way any more, and must not be.
# Its mandatory parameters (``rel_pos``, ``rel_vel``, ``react_radius``,
# ``latch``) cannot be supplied from an alias pool, so ``_bind_kwargs``
# returned None and a *different*, undocumented local trigger silently
# generated every dataset. It is now imported directly at the top of this
# module and called with an explicit argument list; there is exactly one
# implementation of the reactive-avoidance model in dataGen.

_BEHAVIOR_CACHE: dict[str, Optional[Callable[..., Any]]] = {}


def _behavior_attr(name: str) -> Optional[Callable[..., Any]]:
    """Return ``agents.behavior.<name>`` if importable and callable, else None."""
    if name in _BEHAVIOR_CACHE:
        return _BEHAVIOR_CACHE[name]
    fn: Optional[Callable[..., Any]] = None
    try:
        mod = importlib.import_module("agents.behavior")
    except Exception:  # module not written yet / import error -> use fallback
        mod = None
    if mod is not None:
        cand = getattr(mod, name, None)
        if callable(cand):
            fn = cand
    _BEHAVIOR_CACHE[name] = fn
    return fn


def _bind_kwargs(
    fn: Callable[..., Any],
    pool: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Build a keyword-argument dict for ``fn`` from ``pool``.

    Args:
        fn: the callable to bind.
        pool: candidate values keyed by every plausible parameter name.

    Returns:
        The kwargs dict, or ``None`` if ``fn`` declares a mandatory parameter
        that the pool cannot supply (in which case the caller falls back to the
        local reference implementation).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    kwargs: dict[str, Any] = {}
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if p.name in pool:
            kwargs[p.name] = pool[p.name]
        elif p.default is inspect.Parameter.empty:
            return None
    return kwargs


def _as_sign(value: Any) -> int:
    """Normalise a branch result to ``+1`` (left / CCW) or ``-1`` (right / CW).

    Accepts ``True``/``False``, ``+1``/``-1`` and ``1``/``0`` conventions:
    anything strictly positive is left, everything else is right.
    """
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else -1
    try:
        return 1 if float(value) > 0.0 else -1
    except (TypeError, ValueError):
        return -1


def _binary_branch(rng: np.random.Generator, p_left: float = 0.5) -> int:
    """Stochastic left/right branch [-]: ``+1`` = left (CCW), ``-1`` = right.

    Delegates to ``agents.behavior.binary_branch`` when available; the local
    fallback is a single Bernoulli draw with probability ``p_left`` of left.

    Args:
        rng: generator supplying the draw (a decision stream).
        p_left: probability [-] in 0..1 of choosing the left branch.

    Returns:
        ``+1`` or ``-1``.
    """
    fn = _behavior_attr("binary_branch")
    if fn is not None:
        kw = _bind_kwargs(fn, {
            "rng": rng, "generator": rng, "gen": rng, "random_state": rng,
            "p_left": float(p_left), "p": float(p_left),
            "prob_left": float(p_left), "p_true": float(p_left),
            "bias": float(p_left),
        })
        if kw is not None:
            return _as_sign(fn(**kw))
    return 1 if float(rng.random()) < float(p_left) else -1


# ==========================================================================
# Guidance primitives
# ==========================================================================


def cross_track_guidance(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    path_heading: float,
    v: float,
    k_cross: float,
    max_intercept: float,
) -> float:
    """Desired heading [rad] that drives an agent onto a straight centreline.

    The centreline passes through ``(origin_x, origin_y)`` with direction
    ``path_heading``. The signed cross-track error is positive when the agent
    lies to the LEFT of the path; the intercept angle is
    ``atan2(-k_cross * e, v)``, which is naturally bounded by +-pi/2 and is
    further clipped to ``max_intercept`` so the agent never turns broadside.

    Args:
        x: agent East position [m].
        y: agent North position [m].
        origin_x: any point on the centreline, East [m].
        origin_y: any point on the centreline, North [m].
        path_heading: centreline direction [rad], CCW from +X.
        v: airspeed [m/s]; floored by ``EPS_V`` in the denominator.
        k_cross: cross-track gain [1/s].
        max_intercept: maximum intercept angle [rad], positive.

    Returns:
        Desired heading [rad], wrapped to (-pi, pi].
    """
    ch = math.cos(path_heading)
    sh = math.sin(path_heading)
    # Left normal of the path is (-sin, cos): e > 0 means "left of the path".
    e = -sh * (x - origin_x) + ch * (y - origin_y)
    vv = v if v > EPS_V else EPS_V
    corr = math.atan2(-k_cross * e, vv)
    lim = abs(max_intercept)
    if corr > lim:
        corr = lim
    elif corr < -lim:
        corr = -lim
    return wrap_angle(path_heading + corr)


def orbit_guidance(
    x: float,
    y: float,
    centre_x: float,
    centre_y: float,
    radius: float,
    direction: int,
    k_field: float = LOITER_ORBIT_FIELD_GAIN,
) -> float:
    """Desired heading [rad] of a circular-orbit vector field.

    On the circle the field is tangential; outside it points inward, and at
    large distance it degenerates to "fly straight at the centre" (for
    ``k_field = 1``), which is the standard well-behaved orbit field.

    Args:
        x: agent East position [m].
        y: agent North position [m].
        centre_x: orbit centre East [m].
        centre_y: orbit centre North [m].
        radius: orbit radius [m], must be > 0.
        direction: ``+1`` for CCW (left-hand orbit), ``-1`` for CW.
        k_field: convergence gain [-]; 1.0 gives the classic field.

    Returns:
        Desired heading [rad], wrapped to (-pi, pi].
    """
    dx = x - centre_x
    dy = y - centre_y
    lam = math.atan2(dy, dx)
    d = math.hypot(dx, dy)
    r = radius if radius > 1e-6 else 1e-6
    sgn = 1.0 if direction >= 0 else -1.0
    return wrap_angle(lam + sgn * (0.5 * math.pi + k_field * math.atan((d - r) / r)))


def heading_rate_command(
    psi: float,
    psi_desired: float,
    v: float,
    envelope: PerformanceEnvelope,
    gain: float,
    omega_fraction: float = 1.0,
) -> float:
    """Proportional heading-hold turn-rate command [rad/s], envelope-clipped.

    Args:
        psi: current heading [rad].
        psi_desired: desired heading [rad].
        v: airspeed [m/s], used for the speed-dependent turn-rate cap.
        envelope: platform performance envelope.
        gain: proportional gain [1/s].
        omega_fraction: fraction [-] in 0..1 of the envelope cap this
            controller is allowed to use (a "gentleness" knob).

    Returns:
        Turn-rate command [rad/s], magnitude <= ``max_turn_rate(v)``.
    """
    cap = envelope.max_turn_rate(v) * max(0.0, min(1.0, omega_fraction))
    omega = gain * angle_diff(psi_desired, psi)
    if omega > cap:
        return cap
    if omega < -cap:
        return -cap
    return omega


def speed_accel_command(
    v: float,
    v_target: float,
    envelope: PerformanceEnvelope,
    gain: float,
) -> float:
    """Proportional longitudinal acceleration command [m/s^2], envelope-clipped.

    Args:
        v: current airspeed [m/s].
        v_target: target airspeed [m/s]; clipped to the envelope first.
        envelope: platform performance envelope.
        gain: proportional gain [1/s].

    Returns:
        Acceleration command [m/s^2] inside ``[a_min, a_max]``.
    """
    vt = envelope.clamp_speed(float(v_target))
    return envelope.clamp_accel(gain * (vt - v))


def ou_step(
    value: float,
    dt: float,
    tau: float,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    """One exact discrete step of a zero-mean Ornstein-Uhlenbeck process.

    Uses the exact transition ``x' = x*a + sigma*sqrt(1-a^2)*N(0,1)`` with
    ``a = exp(-dt/tau)``, so the sampled path is independent of ``dt`` in
    distribution (unlike an Euler-Maruyama step).

    Args:
        value: current process value (units of ``sigma``).
        dt: step length [s], >= 0.
        tau: correlation time [s], > 0.
        sigma: stationary standard deviation (same units as ``value``).
        rng: generator supplying the Gaussian draw.

    Returns:
        The updated process value.
    """
    t = tau if tau > 1e-9 else 1e-9
    a = math.exp(-max(dt, 0.0) / t)
    return value * a + sigma * math.sqrt(max(1.0 - a * a, 0.0)) * float(
        rng.standard_normal()
    )


# ==========================================================================
# Agent / world duck-typing helpers
# ==========================================================================
#
# ASSUMPTION about agents/base.py (written by a parallel task): an Agent
# exposes ``.state -> AgentState`` and ``.envelope -> PerformanceEnvelope``.
# The alternative attribute names below are accepted so a naming difference
# cannot block integration.


def _agent_state(agent: Any) -> AgentState:
    """Extract the :class:`~core.state.AgentState` of an agent-like object."""
    if isinstance(agent, AgentState):
        return agent
    for name in ("state", "agent_state", "kinematics"):
        s = getattr(agent, name, None)
        if isinstance(s, AgentState):
            return s
    raise TypeError("agent does not expose an AgentState via .state")


def _agent_envelope(agent: Any) -> PerformanceEnvelope:
    """Extract the :class:`~core.state.PerformanceEnvelope` of an agent."""
    for name in ("envelope", "perf", "performance", "limits"):
        e = getattr(agent, name, None)
        if isinstance(e, PerformanceEnvelope):
            return e
    raise TypeError("agent does not expose a PerformanceEnvelope via .envelope")


def _world_extent(world: Any) -> tuple[float, float, float, float]:
    """Return the world bounding box ``(xmin, ymin, xmax, ymax)`` [m].

    Accepts a world exposing ``bounds``/``extent`` (4 numbers),
    ``region_side_m`` (float or callable), ``size_m``, or a ``config`` with
    ``region_side_m()``. Falls back to a square of
    :data:`DEFAULT_REGION_SIDE_M` centred on the origin.
    """
    if world is not None:
        for name in ("bounds", "extent"):
            b = getattr(world, name, None)
            if isinstance(b, (tuple, list)) and len(b) == 4:
                x0, y0, x1, y1 = (float(v) for v in b)
                return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        for holder in (world, getattr(world, "config", None),
                       getattr(world, "cfg", None)):
            if holder is None:
                continue
            for name in ("region_side_m", "side_m", "size_m"):
                s = getattr(holder, name, None)
                if callable(s):
                    try:
                        s = s()
                    except Exception:
                        s = None
                if isinstance(s, (int, float)) and float(s) > 0.0:
                    h = 0.5 * float(s)
                    return (-h, -h, h, h)
    h = 0.5 * DEFAULT_REGION_SIDE_M
    return (-h, -h, h, h)


def _world_corridor(world: Any) -> Optional[tuple[float, float, float]]:
    """Return ``(origin_x [m], origin_y [m], heading [rad])`` of a corridor.

    Accepts a world exposing ``corridor`` as an object with
    ``origin_xy``/``origin`` plus ``heading_rad``/``heading``, or as a
    3-sequence. Returns ``None`` when the world defines no corridor, in which
    case :class:`CorridorTransitIntent` synthesises one from the agent's own
    initial heading.
    """
    if world is None:
        return None
    c = getattr(world, "corridor", None)
    if c is None:
        return None
    if isinstance(c, (tuple, list)) and len(c) == 3:
        return (float(c[0]), float(c[1]), float(c[2]))
    origin = None
    for name in ("origin_xy", "origin", "centre_xy", "center_xy"):
        o = getattr(c, name, None)
        if isinstance(o, (tuple, list)) and len(o) >= 2:
            origin = (float(o[0]), float(o[1]))
            break
    heading = None
    for name in ("heading_rad", "heading", "psi"):
        h = getattr(c, name, None)
        if isinstance(h, (int, float)):
            heading = float(h)
            break
    if origin is None or heading is None:
        return None
    return (origin[0], origin[1], heading)


def _json_copy(value: Any) -> Any:
    """Recursively copy a JSON-safe structure of dicts / lists / scalars."""
    if isinstance(value, dict):
        return {str(k): _json_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return float(value)


def _cruise_speed(state: AgentState, envelope: PerformanceEnvelope) -> float:
    """Nominal cruise airspeed [m/s]: the current speed, sanitised.

    Uses the agent's spawn speed when it is a sensible cruise value, otherwise
    65 % of ``v_max``. Never returns a value below ``EPS_V`` for a powered
    platform, so guidance denominators stay finite.
    """
    v = float(state.v)
    if not math.isfinite(v) or v < max(envelope.v_min, 0.25 * envelope.v_max):
        v = 0.65 * envelope.v_max
    v = envelope.clamp_speed(v)
    if v < EPS_V:
        v = max(EPS_V, min(envelope.v_max, 0.5 * envelope.v_max))
    return v


def _min_turn_radius(envelope: PerformanceEnvelope, v: float) -> float:
    """Minimum turn radius [m] at speed ``v``, finite and >= 1 m."""
    r = envelope.turn_radius(v)
    if not math.isfinite(r) or r <= 1.0:
        return 1.0
    return float(r)


# ==========================================================================
# Base controller
# ==========================================================================


class IntentController:
    """Base class of every intent policy.

    Subclasses implement :meth:`_build` (sample the initial intent state from
    the construction generator) and :meth:`_command` (produce the per-step
    commands). All mutable data lives in ``self._st``, a JSON-safe dict that
    :meth:`snapshot` returns and :meth:`restore` accepts, so branch rollouts
    resume bit-exactly.

    Attributes:
        requested_type: the intent originally asked for (before the
            compatibility fallback).
    """

    #: Effective intent label of this class (what is stored as ``gt_intent``).
    INTENT: IntentType = IntentType.CORRIDOR_TRANSIT

    def __init__(
        self,
        agent: Any,
        world: Any,
        rng: np.random.Generator,
        requested_type: Optional[IntentType] = None,
    ) -> None:
        """Construct and sample the intent's initial state.

        Args:
            agent: the agent this controller drives; must expose ``.state``
                and ``.envelope``.
            world: the world object (may be ``None``); used only for the
                region extent and an optional corridor definition.
            rng: generator used for ALL construction-time randomness.
            requested_type: the intent asked for before any fallback.
        """
        self.requested_type: IntentType = IntentType(
            self.INTENT if requested_type is None else requested_type
        )
        self._st: dict[str, Any] = {"last_t": None}
        #: The ``branch_choice`` generator, bound by :meth:`bind_streams`.
        #: Deliberately NOT part of the snapshot: it is a live reference to a
        #: hub-owned object, and re-binding after every restore / reseed is
        #: what stops a controller drawing from a stale generator.
        self._rng_branch: Optional[np.random.Generator] = None
        state = _agent_state(agent)
        envelope = _agent_envelope(agent)
        self._build(state, envelope, world, rng)

    # -- public API -------------------------------------------------------

    def bind_streams(
        self,
        rng_mode: np.random.Generator,
        rng_branch: np.random.Generator,
    ) -> None:
        """Bind the decision streams (mirrors ``ModeMachine.bind_streams``).

        An intent controller is handed ONE generator per step by the agent
        host, and :mod:`simulation.engine` passes ``mode_decision``. Any
        genuine left/right FORK an intent takes must come from
        ``branch_choice`` instead, or re-seeding ``branch_choice`` alone --
        the contract branch rollouts rely on -- would not flip it.

        Args:
            rng_mode: the ``mode_decision`` stream (accepted for signature
                symmetry; the base class does not use it).
            rng_branch: the ``branch_choice`` stream.
        """
        self._rng_branch = rng_branch

    def _branch_rng(
        self,
        fallback: np.random.Generator,
    ) -> np.random.Generator:
        """The fork generator: ``branch_choice`` when bound, else ``fallback``."""
        return self._rng_branch if self._rng_branch is not None else fallback

    @property
    def intent_type(self) -> IntentType:
        """Effective intent type [-] of this controller."""
        return IntentType(self.INTENT)

    @property
    def state_dict(self) -> dict[str, Any]:
        """Live (mutable) intent-state dict. Do not copy-and-keep; see snapshot."""
        return self._st

    def command(
        self,
        agent: Any,
        t: float,
        ownship_state: Optional[AgentState],
        rng: np.random.Generator,
    ) -> tuple[Optional[float], Optional[float]]:
        """Produce this step's ``(omega_cmd [rad/s], a_cmd [m/s^2])``.

        The step length is derived as ``t - t_previous`` (the controller is not
        handed ``dt``); the first call, and any implausible difference, uses
        :data:`DEFAULT_DT_S`. This keeps the controller callable from the
        engine, the GUI stepper and the branch-rollout driver alike, and keeps
        ``dt`` a deterministic function of the snapshot.

        Args:
            agent: the driven agent (``.state`` / ``.envelope``).
            t: simulation time [s] since episode start.
            ownship_state: ownship kinematic state, or ``None`` when the
                ownship is not modelled. Only ``REACTIVE_AVOID`` reads it.
            rng: decision-stream generator for this step.

        Returns:
            ``(omega_cmd, a_cmd)``; both ``None`` for an unpowered body, which
            tells ``Agent.step`` to use the passive integrator.
        """
        state = _agent_state(agent)
        envelope = _agent_envelope(agent)
        last = self._st.get("last_t")
        if last is None:
            dt = DEFAULT_DT_S
        else:
            dt = float(t) - float(last)
            if not (0.0 < dt <= MAX_PLAUSIBLE_DT_S) or not math.isfinite(dt):
                dt = DEFAULT_DT_S
        self._st["last_t"] = float(t)
        return self._command(state, envelope, float(t), dt, ownship_state, rng)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe, fully sufficient copy of the controller state.

        Returns:
            ``{"intent": int, "requested": int, "st": {...}}``. Feeding this to
            :meth:`restore` reproduces every future command exactly, provided
            the generator state is restored too.
        """
        return {
            "intent": int(self.INTENT),
            "requested": int(self.requested_type),
            "st": _json_copy(self._st),
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        """Restore from a :meth:`snapshot` result (or a bare state dict).

        Args:
            snap: the dict produced by :meth:`snapshot`. A bare intent-state
                dict (no ``"st"`` key) is also accepted, for convenience when
                reading back ``AgentSnapshot.intent_state``.
        """
        if "st" in snap:
            self._st = _json_copy(dict(snap["st"]))
            if "requested" in snap:
                self.requested_type = IntentType(int(snap["requested"]))
        else:
            self._st = _json_copy(dict(snap))

    def describe(self) -> str:
        """One-line human-readable description for the GUI inspector panel."""
        return INTENT_NAMES[int(self.INTENT)]

    # -- subclass hooks ---------------------------------------------------

    def _build(
        self,
        state: AgentState,
        envelope: PerformanceEnvelope,
        world: Any,
        rng: np.random.Generator,
    ) -> None:
        """Sample the initial intent state into ``self._st`` (subclass hook)."""
        raise NotImplementedError

    def _command(
        self,
        state: AgentState,
        envelope: PerformanceEnvelope,
        t: float,
        dt: float,
        ownship_state: Optional[AgentState],
        rng: np.random.Generator,
    ) -> tuple[Optional[float], Optional[float]]:
        """Per-step command (subclass hook). See :meth:`command`."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}({self.describe()})"


# ==========================================================================
# CORRIDOR_TRANSIT
# ==========================================================================


class CorridorTransitIntent(IntentController):
    """Long straight transit along an air corridor (UAM passenger, long-range
    delivery).

    Deliberately the EASY-to-predict class: the agent holds a fixed centreline
    with a small-gain cross-track law and only rarely nudges its speed. A
    constant-velocity extrapolator does well here, and that is the point --
    the dataset needs an easy end to contrast against ``SEARCH_WANDER`` when
    the calibration story is told (doc 3.1 / 8.1).
    """

    INTENT = IntentType.CORRIDOR_TRANSIT

    def _build(self, state, envelope, world, rng) -> None:
        """Sample the centreline, the initial lateral offset and cruise speed."""
        v_nom = _cruise_speed(state, envelope)
        corridor = _world_corridor(world)
        if corridor is None:
            # No corridor in the world: synthesise one through the spawn point
            # along the spawn heading, with a small deviation so agents on the
            # same corridor are not perfectly parallel.
            ph = wrap_angle(
                float(state.psi)
                + float(rng.uniform(-CORRIDOR_HEADING_DEV_RAD,
                                    CORRIDOR_HEADING_DEV_RAD))
            )
            off = float(rng.uniform(-CORRIDOR_OFFSET_M, CORRIDOR_OFFSET_M))
            ox = float(state.x) - (-math.sin(ph)) * off
            oy = float(state.y) - (math.cos(ph)) * off
        else:
            ox, oy, ph = corridor
            ph = wrap_angle(
                ph + float(rng.uniform(-CORRIDOR_HEADING_DEV_RAD,
                                       CORRIDOR_HEADING_DEV_RAD))
            )
        self._st.update({
            "ox": float(ox),
            "oy": float(oy),
            "ph": float(ph),
            "v_nom": float(v_nom),
            "v_t": float(v_nom),
            "speed_timer": float(rng.uniform(CORRIDOR_SPEED_TIMER_MIN_S,
                                             CORRIDOR_SPEED_TIMER_MAX_S)),
            "goal_x": float(ox + CORRIDOR_GOAL_RANGE_M * math.cos(ph)),
            "goal_y": float(oy + CORRIDOR_GOAL_RANGE_M * math.sin(ph)),
        })

    def _command(self, state, envelope, t, dt, ownship_state, rng):
        """Hold the centreline; resample the cruise speed on a slow timer."""
        st = self._st
        st["speed_timer"] = float(st["speed_timer"]) - dt
        if st["speed_timer"] <= 0.0:
            sigma = CORRIDOR_SPEED_SIGMA_FRAC * float(st["v_nom"])
            st["v_t"] = float(
                envelope.clamp_speed(
                    float(st["v_nom"]) + sigma * float(rng.standard_normal())
                )
            )
            st["speed_timer"] = float(
                rng.uniform(CORRIDOR_SPEED_TIMER_MIN_S, CORRIDOR_SPEED_TIMER_MAX_S)
            )
        psi_des = cross_track_guidance(
            state.x, state.y, float(st["ox"]), float(st["oy"]), float(st["ph"]),
            state.v, CORRIDOR_CROSS_GAIN, CORRIDOR_MAX_INTERCEPT_RAD,
        )
        omega = heading_rate_command(
            state.psi, psi_des, state.v, envelope,
            CORRIDOR_HEADING_GAIN, CORRIDOR_OMEGA_FRACTION,
        )
        a = speed_accel_command(state.v, float(st["v_t"]), envelope,
                                CORRIDOR_SPEED_GAIN)
        return (omega, a)

    def describe(self) -> str:
        """Describe the corridor heading and target speed."""
        st = self._st
        return (
            f"Corridor transit: centreline {math.degrees(float(st['ph'])):.0f} deg, "
            f"target {float(st['v_t']):.1f} m/s"
        )


# ==========================================================================
# DELIVERY_WP
# ==========================================================================


def _leg_step(
    x: float,
    y: float,
    heading: float,
    capture_r: float,
    rng: np.random.Generator,
    extent: tuple[float, float, float, float],
    first: bool,
) -> tuple[float, float, float]:
    """Generate one route leg, returning ``(next_x, next_y, leg_heading)`` [m, rad].

    The turn between successive legs is a :func:`_binary_branch` sign times a
    magnitude in ``[WP_TURN_MIN_RAD, WP_TURN_MAX_RAD]``, so the route is made of
    committed, clearly visible turns rather than random jitter. The leg is at
    least ``WP_LEG_FACTOR_MIN`` capture radii long, which guarantees the capture
    logic cannot skip several waypoints in one step. If the leg would leave the
    world, the heading is re-aimed at the region centre (plus a random spread).
    """
    length = float(rng.uniform(WP_LEG_FACTOR_MIN * capture_r,
                               WP_LEG_FACTOR_MAX * capture_r))
    if length < WP_LEG_MIN_M:
        length = WP_LEG_MIN_M
    h = float(heading)
    if not first:
        sign = _binary_branch(rng, 0.5)
        h = wrap_angle(
            h + sign * float(rng.uniform(WP_TURN_MIN_RAD, WP_TURN_MAX_RAD))
        )
    nx = x + length * math.cos(h)
    ny = y + length * math.sin(h)
    xmin, ymin, xmax, ymax = extent
    margin = 0.10 * max(xmax - xmin, ymax - ymin)
    if not (xmin - margin <= nx <= xmax + margin
            and ymin - margin <= ny <= ymax + margin):
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        h = wrap_angle(
            math.atan2(cy - y, cx - x)
            + float(rng.uniform(-0.7, 0.7))
        )
        nx = x + length * math.cos(h)
        ny = y + length * math.sin(h)
    return (nx, ny, h)


class DeliveryWaypointIntent(IntentController):
    """Waypoint sequence: multi-drop delivery or a survey grid.

    Proportional heading control to the active waypoint; the waypoint advances
    on a capture radius of about ``3 x`` the minimum turn radius, on a
    passed-the-plane test (so a large aircraft whose turn radius exceeds the
    capture radius cannot orbit a waypoint forever) or on a dwell timeout.

    Multimodality source (doc 3.3): the route is generated stochastically at
    construction, and at every capture there is a :data:`WP_REPLAN_PROB`
    chance of re-drawing the NEXT leg from the decision stream. Branch
    rollouts therefore diverge at waypoint captures -- committed, physically
    reachable futures rather than jitter.
    """

    INTENT = IntentType.DELIVERY_WP

    def _build(self, state, envelope, world, rng) -> None:
        """Sample the capture radius and the initial stochastic route."""
        v_nom = _cruise_speed(state, envelope)
        extent = _world_extent(world)
        side = max(extent[2] - extent[0], extent[3] - extent[1])
        r_min = _min_turn_radius(envelope, v_nom)
        capture = WP_CAPTURE_TURN_FACTOR * r_min
        capture = max(WP_CAPTURE_MIN_M,
                      min(capture, WP_CAPTURE_REGION_FRACTION * side))
        n = int(rng.integers(WP_N_MIN, WP_N_MAX + 1))
        wps: list[list[float]] = []
        x, y, h = float(state.x), float(state.y), float(state.psi)
        for i in range(n):
            x, y, h = _leg_step(x, y, h, capture, rng, extent, first=(i == 0))
            wps.append([x, y])
        self._st.update({
            "wp": wps,
            "idx": 0,
            "prev": [float(state.x), float(state.y)],
            "leg_psi": float(h),
            "capture_r": float(capture),
            "dwell": 0.0,
            "v_nom": float(v_nom),
            "v_t": float(v_nom),
            "extent": [float(e) for e in extent],
            "captures": 0,
        })

    def _extend_route(self, rng: np.random.Generator) -> None:
        """Append fresh legs when the route runs out; keep the list bounded."""
        st = self._st
        wps: list[list[float]] = st["wp"]
        x, y = (wps[-1] if wps else st["prev"])
        h = float(st["leg_psi"])
        extent = tuple(float(e) for e in st["extent"])  # type: ignore[assignment]
        for _ in range(WP_N_MIN):
            x, y, h = _leg_step(x, y, h, float(st["capture_r"]), rng,
                                extent, first=False)  # type: ignore[arg-type]
            wps.append([x, y])
        st["leg_psi"] = float(h)
        if len(wps) > WP_LIST_TRIM_LEN:
            drop = len(wps) - WP_LIST_TRIM_LEN
            del wps[:drop]
            st["idx"] = max(0, int(st["idx"]) - drop)

    def _replan_next(self, rng: np.random.Generator) -> None:
        """Re-draw the upcoming waypoint (the stochastic branch point)."""
        st = self._st
        wps: list[list[float]] = st["wp"]
        i = int(st["idx"])
        if i >= len(wps):
            return
        px, py = (wps[i - 1] if i > 0 else st["prev"])
        extent = tuple(float(e) for e in st["extent"])  # type: ignore[assignment]
        nx, ny, h = _leg_step(float(px), float(py), float(st["leg_psi"]),
                              float(st["capture_r"]), rng,
                              extent, first=False)  # type: ignore[arg-type]
        wps[i] = [nx, ny]
        st["leg_psi"] = float(h)

    def _command(self, state, envelope, t, dt, ownship_state, rng):
        """Steer to the active waypoint and advance it when captured/passed."""
        st = self._st
        wps: list[list[float]] = st["wp"]
        if int(st["idx"]) >= len(wps):
            self._extend_route(rng)
            wps = st["wp"]
        i = int(st["idx"])
        wx, wy = float(wps[i][0]), float(wps[i][1])
        px, py = (wps[i - 1] if i > 0 else st["prev"])
        dx = wx - float(state.x)
        dy = wy - float(state.y)
        dist = math.hypot(dx, dy)
        st["dwell"] = float(st["dwell"]) + dt

        # Leg direction (from the previous waypoint to the active one) defines
        # the "passed the waypoint" half-plane test.
        lx = wx - float(px)
        ly = wy - float(py)
        passed = (lx * dx + ly * dy) < 0.0 and math.hypot(lx, ly) > 1e-6
        if (dist <= float(st["capture_r"]) or passed
                or float(st["dwell"]) >= WP_MAX_DWELL_S):
            st["prev"] = [wx, wy]
            st["idx"] = i + 1
            st["dwell"] = 0.0
            st["captures"] = int(st["captures"]) + 1
            if int(st["idx"]) >= len(wps):
                self._extend_route(rng)
            if float(rng.random()) < WP_REPLAN_PROB:
                self._replan_next(rng)
            wps = st["wp"]
            i = int(st["idx"])
            wx, wy = float(wps[i][0]), float(wps[i][1])
            dx = wx - float(state.x)
            dy = wy - float(state.y)
            dist = math.hypot(dx, dy)

        psi_des = math.atan2(dy, dx) if dist > 1e-6 else float(state.psi)
        omega = heading_rate_command(
            state.psi, psi_des, state.v, envelope,
            WP_HEADING_GAIN, WP_OMEGA_FRACTION,
        )
        a = speed_accel_command(state.v, float(st["v_t"]), envelope,
                                WP_SPEED_GAIN)
        return (omega, a)

    def describe(self) -> str:
        """Describe the active waypoint and the capture radius."""
        st = self._st
        i = int(st["idx"])
        wps = st["wp"]
        if 0 <= i < len(wps):
            wp = f"({wps[i][0]:.0f}, {wps[i][1]:.0f}) m"
        else:
            wp = "(replanning)"
        return (
            f"Delivery waypoints: leg {i + 1}/{len(wps)} -> {wp}, "
            f"capture {float(st['capture_r']):.0f} m, "
            f"{int(st['captures'])} captured"
        )


# ==========================================================================
# LOITER_HOVER
# ==========================================================================


class LoiterHoverIntent(IntentController):
    """Station keeping: infrastructure inspection, filming, holding.

    Two variants, selected at construction from the platform envelope:

    * ``hover`` (``can_hover``): the airspeed is driven to ``v_min`` plus a
      small margin and the heading drifts on a slow OU process; a gentle
      restoring term pulls the body back when wind has advected it more than
      :data:`HOVER_STATION_RADIUS_M` from the station.
    * ``orbit`` (everything else, e.g. fixed-wing): a circular orbit of a
      sampled radius >= :data:`LOITER_RADIUS_FACTOR_MIN` times the minimum turn
      radius, flown with the vector field of :func:`orbit_guidance`. The turn
      direction is a :func:`_binary_branch` draw.

    This is the fallback documented in the module header: a fixed-wing asked to
    LOITER_HOVER keeps the ``LOITER_HOVER`` label (it really is loitering) but
    is given the orbit variant, which its envelope can actually fly.
    """

    INTENT = IntentType.LOITER_HOVER

    def _build(self, state, envelope, world, rng) -> None:
        """Choose the hover or orbit variant and sample its parameters."""
        v_nom = _cruise_speed(state, envelope)
        hover = bool(envelope.can_hover)
        direction = _binary_branch(rng, 0.5)
        if hover:
            v_t = envelope.clamp_speed(envelope.v_min + HOVER_SPEED_MARGIN_MS)
            self._st.update({
                "variant": "hover",
                "cx": float(state.x),
                "cy": float(state.y),
                "radius": 0.0,
                "dir": int(direction),
                "v_nom": float(v_nom),
                "v_t": float(v_t),
                "ou": 0.0,
            })
            return
        r_min = _min_turn_radius(envelope, v_nom)
        radius = float(rng.uniform(LOITER_RADIUS_FACTOR_MIN,
                                   LOITER_RADIUS_FACTOR_MAX)) * r_min
        radius = max(radius, LOITER_RADIUS_MIN_M)
        # Place the centre abeam so the agent starts exactly on its orbit,
        # tangentially, and never has to fly a transient capture arc.
        bearing = wrap_angle(float(state.psi) + direction * 0.5 * math.pi)
        self._st.update({
            "variant": "orbit",
            "cx": float(state.x) + radius * math.cos(bearing),
            "cy": float(state.y) + radius * math.sin(bearing),
            "radius": float(radius),
            "dir": int(direction),
            "v_nom": float(v_nom),
            "v_t": float(v_nom),
            "ou": 0.0,
        })

    def _command(self, state, envelope, t, dt, ownship_state, rng):
        """Fly the orbit, or hold the station at minimum airspeed."""
        st = self._st
        if st["variant"] == "orbit":
            psi_des = orbit_guidance(
                state.x, state.y, float(st["cx"]), float(st["cy"]),
                float(st["radius"]), int(st["dir"]), LOITER_ORBIT_FIELD_GAIN,
            )
            omega = heading_rate_command(
                state.psi, psi_des, state.v, envelope,
                LOITER_HEADING_GAIN, LOITER_OMEGA_FRACTION,
            )
            a = speed_accel_command(state.v, float(st["v_t"]), envelope,
                                    LOITER_SPEED_GAIN)
            return (omega, a)

        cap = envelope.max_turn_rate(state.v)
        st["ou"] = ou_step(float(st["ou"]), dt, HOVER_DRIFT_TAU_S,
                           HOVER_DRIFT_SIGMA_FRAC * cap, rng)
        dx = float(st["cx"]) - float(state.x)
        dy = float(st["cy"]) - float(state.y)
        drift = math.hypot(dx, dy)
        if drift > HOVER_STATION_RADIUS_M:
            psi_des = math.atan2(dy, dx)
            omega = heading_rate_command(
                state.psi, psi_des, state.v, envelope,
                HOVER_RETURN_GAIN, LOITER_OMEGA_FRACTION,
            )
        else:
            omega = envelope.clamp_turn_rate(float(st["ou"]), state.v)
        a = speed_accel_command(state.v, float(st["v_t"]), envelope,
                                HOVER_SPEED_GAIN)
        return (omega, a)

    def variant(self) -> str:
        """Return ``"hover"`` or ``"orbit"`` -- which variant was selected."""
        return str(self._st.get("variant", "orbit"))

    def describe(self) -> str:
        """Describe the station or the orbit."""
        st = self._st
        if st["variant"] == "orbit":
            turn = "CCW" if int(st["dir"]) >= 0 else "CW"
            return (
                f"Loiter (orbit): R={float(st['radius']):.0f} m {turn} about "
                f"({float(st['cx']):.0f}, {float(st['cy']):.0f}) m"
            )
        return (
            f"Loiter (hover): station ({float(st['cx']):.0f}, "
            f"{float(st['cy']):.0f}) m, target {float(st['v_t']):.1f} m/s"
        )


# ==========================================================================
# SEARCH_WANDER
# ==========================================================================


class _WanderCore:
    """Shared wander machinery for SEARCH_WANDER and REACTIVE_AVOID.

    The wander is the sum of two stochastic mechanisms, which is what makes it
    hard for a constant-velocity extrapolator:

    * a continuous Ornstein-Uhlenbeck turn rate (smooth, unpredictable
      curvature), and
    * Poisson-triggered discrete reorientations of a committed magnitude,
      whose left/right sign comes from :func:`_binary_branch`. The committed
      turn is held until the requested heading change has been flown, which is
      the doc 3.3 (b) dwell constraint -- no per-step re-decision chatter.

    This is a helper, not a controller: it operates on the owner's state dict
    so that everything stays inside a single snapshot.
    """

    @staticmethod
    def build(st: dict[str, Any], rng: np.random.Generator) -> None:
        """Initialise the wander keys in ``st``."""
        st["ou"] = 0.0
        st["turn_left"] = 0.0

    @staticmethod
    def command_omega(
        st: dict[str, Any],
        state: AgentState,
        envelope: PerformanceEnvelope,
        dt: float,
        rng: np.random.Generator,
        ou_tau: float,
        ou_sigma_frac: float,
        ou_omega_frac: float,
        rate_hz: float,
        turn_min: float,
        turn_max: float,
        turn_omega_frac: float,
    ) -> float:
        """Advance the wander by ``dt`` and return the turn-rate command [rad/s]."""
        cap = envelope.max_turn_rate(state.v)
        st["ou"] = ou_step(float(st["ou"]), dt, ou_tau, ou_sigma_frac * cap, rng)
        remaining = float(st["turn_left"])
        if abs(remaining) > 1e-9:
            sign = 1.0 if remaining > 0.0 else -1.0
            omega = sign * turn_omega_frac * cap
            remaining -= omega * dt
            if (sign > 0.0 and remaining <= 0.0) or (sign < 0.0 and remaining >= 0.0):
                remaining = 0.0
            st["turn_left"] = float(remaining)
            return envelope.clamp_turn_rate(omega, state.v)
        p_fire = 1.0 - math.exp(-max(rate_hz, 0.0) * max(dt, 0.0))
        if float(rng.random()) < p_fire:
            sign = _binary_branch(rng, 0.5)
            magnitude = float(rng.uniform(turn_min, turn_max))
            st["turn_left"] = float(sign * magnitude)
            omega = sign * turn_omega_frac * cap
            return envelope.clamp_turn_rate(omega, state.v)
        omega = float(st["ou"])
        lim = ou_omega_frac * cap
        if omega > lim:
            omega = lim
        elif omega < -lim:
            omega = -lim
        return envelope.clamp_turn_rate(omega, state.v)


class SearchWanderIntent(IntentController):
    """Search-and-rescue sweep, or a bird foraging (doc 3.2 rev.2).

    HIGHEST multimodality of the six intents and the star of the OOD split
    (doc 8.1): the Poisson reorientations are genuine coin flips, so replaying
    the same initial condition with a different ``branch_choice`` seed produces
    visibly different, physically reachable futures. This is the class that
    makes the "true distribution" of doc 3.3 non-degenerate, and therefore the
    class the calibration argument depends on.
    """

    INTENT = IntentType.SEARCH_WANDER

    def _build(self, state, envelope, world, rng) -> None:
        """Initialise the OU wander and the speed-variation timer."""
        v_nom = _cruise_speed(state, envelope)
        _WanderCore.build(self._st, rng)
        self._st.update({
            "v_nom": float(v_nom),
            "v_t": float(v_nom),
            "speed_timer": float(rng.uniform(SEARCH_SPEED_TIMER_MIN_S,
                                             SEARCH_SPEED_TIMER_MAX_S)),
            "reorients": 0,
        })

    def _command(self, state, envelope, t, dt, ownship_state, rng):
        """Wander: OU curvature plus Poisson committed reorientations."""
        st = self._st
        before = abs(float(st["turn_left"])) > 1e-9
        omega = _WanderCore.command_omega(
            st, state, envelope, dt, rng,
            SEARCH_OU_TAU_S, SEARCH_OU_SIGMA_FRAC, SEARCH_OU_OMEGA_FRACTION,
            SEARCH_REORIENT_RATE_HZ, SEARCH_TURN_MIN_RAD, SEARCH_TURN_MAX_RAD,
            SEARCH_TURN_OMEGA_FRACTION,
        )
        if (not before) and abs(float(st["turn_left"])) > 1e-9:
            st["reorients"] = int(st["reorients"]) + 1
        st["speed_timer"] = float(st["speed_timer"]) - dt
        if st["speed_timer"] <= 0.0:
            lo = max(envelope.v_min, SEARCH_SPEED_LO_FRAC * envelope.v_max)
            hi = max(lo + 1e-6, SEARCH_SPEED_HI_FRAC * envelope.v_max)
            st["v_t"] = float(envelope.clamp_speed(float(rng.uniform(lo, hi))))
            st["speed_timer"] = float(
                rng.uniform(SEARCH_SPEED_TIMER_MIN_S, SEARCH_SPEED_TIMER_MAX_S)
            )
        a = speed_accel_command(state.v, float(st["v_t"]), envelope,
                                SEARCH_SPEED_GAIN)
        return (omega, a)

    def describe(self) -> str:
        """Describe the current wander phase."""
        st = self._st
        rem = float(st["turn_left"])
        if abs(rem) > 1e-9:
            side = "left" if rem > 0.0 else "right"
            phase = f"committed {side} turn, {abs(math.degrees(rem)):.0f} deg left"
        else:
            phase = f"OU drift {math.degrees(float(st['ou'])):.2f} deg/s"
        return (
            f"Search wander: {phase}, {int(st['reorients'])} reorientations, "
            f"target {float(st['v_t']):.1f} m/s"
        )


# ==========================================================================
# REACTIVE_AVOID
# ==========================================================================


def _relative_geometry(
    state: AgentState,
    ownship_state: AgentState,
) -> tuple[float, float, float, float]:
    """Relative geometry to the ownship.

    Uses air-relative velocities (wind is unknown to the intent layer and
    advects both bodies almost equally, so it largely cancels in the relative
    vector).

    Args:
        state: this agent's state.
        ownship_state: the ownship's state.

    Returns:
        ``(range_m, tcpa_s, cpa_m, closing_speed_ms)``. ``tcpa_s`` is negative
        when the pair is diverging; ``closing_speed_ms`` is positive when the
        range is shrinking.
    """
    dx = float(ownship_state.x) - float(state.x)
    dy = float(ownship_state.y) - float(state.y)
    r = math.hypot(dx, dy)
    ovx, ovy = ownship_state.air_velocity()
    svx, svy = state.air_velocity()
    rvx = ovx - svx
    rvy = ovy - svy
    v2 = rvx * rvx + rvy * rvy
    if v2 < 1e-9:
        return (r, -1.0, r, 0.0)
    tcpa = -(dx * rvx + dy * rvy) / v2
    cx = dx + rvx * tcpa
    cy = dy + rvy * tcpa
    cpa = math.hypot(cx, cy)
    closing = -(dx * rvx + dy * rvy) / r if r > 1e-6 else 0.0
    return (r, tcpa, cpa, closing)


class ReactiveAvoidIntent(IntentController):
    """Baseline wander that reacts to the approaching ownship (doc 3.2 rev.2).

    The ONLY intent that reads the ownship state. When
    :func:`_reactive_avoid_decision` fires, the agent latches a turn AWAY from
    the ownship at :data:`RA_OMEGA_FRACTION` of its turn-rate cap for a sampled
    latch duration, then relaxes through a refractory period back to its
    gentle wander.

    Note the deliberate framing (doc 3.2 rev.2 removed "pursuit"): this is a
    one-sided reaction, exactly as observed in bird avoidance of aircraft. It
    creates interaction structure in the dataset -- futures that depend on the
    ownship's own choices -- without importing any pursuit-evasion
    game-theoretic claim the paper would then have to defend.
    """

    INTENT = IntentType.REACTIVE_AVOID

    def _build(self, state, envelope, world, rng) -> None:
        """Initialise the gentle wander plus the avoidance latch bookkeeping."""
        v_nom = _cruise_speed(state, envelope)
        _WanderCore.build(self._st, rng)
        self._st.update({
            "v_nom": float(v_nom),
            "v_t": float(v_nom),
            # THE latch, owned by agents.behavior.reactive_avoid_decision and
            # mutated in place by it. A plain dict of primitives, so it rides
            # inside ``intent_state`` through snapshot / restore unchanged and
            # a branch rollout resumes mid-evasion exactly where it was.
            "avoid_latch": new_avoid_latch(),
            "refractory": 0.0,
            "avoids": 0,
        })

    def _command(self, state, envelope, t, dt, ownship_state, rng):
        """Latch an evasive turn when alarmed, else wander gently.

        The trigger, the commitment timer and the side coin all live in
        :func:`agents.behavior.reactive_avoid_decision`; this method only
        supplies the geometry, decides how long to commit and turns the
        latched side into a turn-rate command.
        """
        st = self._st
        st["refractory"] = max(0.0, float(st["refractory"]) - dt)
        latch = st["avoid_latch"]
        was_active = bool(latch.get("active", False))

        active = False
        sign = 0
        if ownship_state is not None:
            dx = float(ownship_state.x) - float(state.x)
            dy = float(ownship_state.y) - float(state.y)
            ovx, ovy = ownship_state.air_velocity()
            svx, svy = state.air_velocity()
            # The side is drawn from branch_choice (the fork stream), the
            # commitment DURATION from the step generator: doc 3.3's
            # factorisation is "branch_choice decides which way, mode_decision
            # decides the rhythm", and re-seeding branch_choice alone must
            # flip sides without moving decision times.
            duration = float(rng.uniform(RA_LATCH_MIN_S, RA_LATCH_MAX_S))
            active, sign = reactive_avoid_decision(
                self._branch_rng(rng),
                (dx, dy),
                (ovx - svx, ovy - svy),
                RA_DETECT_RANGE_M,
                latch,
                dt=dt,
                latch_s=duration,
                p_left=self._p_left_away(state, dx, dy),
                tcpa_max_s=RA_TCPA_MAX_S,
                cpa_max_m=RA_CPA_TRIGGER_M,
                armed=float(st["refractory"]) <= 0.0,
            )

        if active:
            if not was_active:
                st["avoids"] = int(st["avoids"]) + 1
                st["turn_left"] = 0.0
            cap = envelope.max_turn_rate(state.v)
            omega = float(sign) * RA_OMEGA_FRACTION * cap
            a = speed_accel_command(state.v, float(st["v_t"]), envelope,
                                    SEARCH_SPEED_GAIN)
            return (envelope.clamp_turn_rate(omega, state.v), a)

        if was_active:
            # The commitment just expired: relax through the refractory period
            # instead of immediately re-arming on the same encounter.
            st["refractory"] = float(RA_REFRACTORY_S)
            st["turn_left"] = 0.0

        omega = _WanderCore.command_omega(
            st, state, envelope, dt, rng,
            RA_OU_TAU_S, RA_OU_SIGMA_FRAC, RA_OU_OMEGA_FRACTION,
            RA_REORIENT_RATE_HZ, RA_TURN_MIN_RAD, RA_TURN_MAX_RAD,
            RA_TURN_OMEGA_FRACTION,
        )
        a = speed_accel_command(state.v, float(st["v_t"]), envelope,
                                SEARCH_SPEED_GAIN)
        return (omega, a)

    @staticmethod
    def _p_left_away(state: AgentState, dx: float, dy: float) -> float:
        """Probability [-] of choosing LEFT so the turn goes AWAY from the threat.

        Deterministic (0 or 1) at any resolvable relative bearing -- a bird
        turns away from the aircraft, not at random. Inside a narrow dead-ahead
        or dead-astern cone the "away" side does not exist, so the choice
        becomes a genuine 50/50 fork, which is one of the branch points doc 3.3
        wants the multimodality to come from.
        """
        bearing = math.atan2(dy, dx)
        rel = angle_diff(bearing, float(state.psi))
        if abs(rel) < 0.15 or abs(rel) > (math.pi - 0.15):
            return 0.5
        return 0.0 if rel > 0.0 else 1.0

    def describe(self) -> str:
        """Describe the latch state."""
        st = self._st
        latch = st.get("avoid_latch", {})
        if bool(latch.get("active", False)):
            side = "left" if int(latch.get("sign", 0)) > 0 else "right"
            return (
                f"Reactive avoid: LATCHED {side}, "
                f"{float(latch.get('timer', 0.0)):.1f} s remaining "
                f"({int(st['avoids'])} reactions)"
            )
        if float(st["refractory"]) > 0.0:
            return (
                f"Reactive avoid: relaxing, "
                f"{float(st['refractory']):.1f} s refractory"
            )
        return f"Reactive avoid: monitoring ({int(st['avoids'])} reactions)"


# ==========================================================================
# PASSIVE_DRIFT
# ==========================================================================


class PassiveDriftIntent(IntentController):
    """Unpowered body (DEBRIS): no commands at all.

    :meth:`command` returns ``(None, None)``, the sentinel that makes
    ``Agent.step`` route the body through the passive integrator (drag +
    wind advection with ``wind_susceptibility``) instead of the powered one.
    The controller still keeps a snapshot dict so that the snapshot/restore
    contract is uniform across intents.
    """

    INTENT = IntentType.PASSIVE_DRIFT

    def _build(self, state, envelope, world, rng) -> None:
        """Record the spawn state; nothing stochastic happens here."""
        self._st.update({
            "spawn_x": float(state.x),
            "spawn_y": float(state.y),
            "spawn_v": float(state.v),
        })

    def _command(self, state, envelope, t, dt, ownship_state, rng):
        """Always ``(None, None)``: the body is not controlled."""
        return (None, None)

    def describe(self) -> str:
        """Describe the passive body."""
        return (
            "Passive drift: unpowered, advected by wind "
            f"(spawn speed {float(self._st['spawn_v']):.1f} m/s)"
        )


# ==========================================================================
# Factory and compatibility guard
# ==========================================================================


#: Effective controller class for each :class:`~core.types.IntentType`.
INTENT_CLASSES: dict[IntentType, type[IntentController]] = {
    IntentType.CORRIDOR_TRANSIT: CorridorTransitIntent,
    IntentType.DELIVERY_WP: DeliveryWaypointIntent,
    IntentType.LOITER_HOVER: LoiterHoverIntent,
    IntentType.SEARCH_WANDER: SearchWanderIntent,
    IntentType.REACTIVE_AVOID: ReactiveAvoidIntent,
    IntentType.PASSIVE_DRIFT: PassiveDriftIntent,
}

#: Human-readable fallback table (also reproduced in the module docstring).
INTENT_FALLBACK_DOC: tuple[tuple[str, str, str], ...] = (
    ("<any>", "envelope.is_powered is False", "PASSIVE_DRIFT"),
    ("PASSIVE_DRIFT", "envelope.is_powered is True", "CORRIDOR_TRANSIT"),
    ("LOITER_HOVER", "envelope.can_hover is False",
     "LOITER_HOVER (orbit variant, same class)"),
)


def resolve_intent_type(
    intent_type: IntentType | int,
    envelope: PerformanceEnvelope,
) -> IntentType:
    """Apply the platform-compatibility fallback to a requested intent.

    An intent a platform physically cannot execute must never reach the
    integrator as an invalid command; it is substituted here instead. The
    ``LOITER_HOVER`` hover-vs-orbit choice is NOT a type substitution -- the
    controller selects the orbit variant internally, so the stored
    ``gt_intent`` label stays truthful.

    Args:
        intent_type: the requested intent.
        envelope: the platform's performance envelope.

    Returns:
        The effective intent type.
    """
    it = IntentType(int(intent_type))
    if not envelope.is_powered:
        return IntentType.PASSIVE_DRIFT
    if it is IntentType.PASSIVE_DRIFT:
        return IntentType.CORRIDOR_TRANSIT
    return it


#: Which intents each PLATFORM class is allowed to be given (doc 3.2 mission
#: plausibility, NOT a physical limit -- physics is enforced by
#: :class:`core.state.PerformanceEnvelope`).
#:
#: Without this, platform and intent are drawn independently and the dataset
#: contains a passenger eVTOL flying a random-zigzag search pattern and a
#: general-aviation aircraft doing the same. Measured consequence: the eVTOL
#: class's 15 s CV error is a MIXTURE (17.4 m on a corridor, 179.0 m
#: search-wandering) rather than a class, so the platform label -- an auxiliary
#: supervised target in doc 5.1 -- does not identify a behaviour, and the
#: platform-axis HARD/EASY band separation is depressed by the confound.
#:
#: The exclusions are deliberately few, and each has a mission reason:
#:   * SEARCH_WANDER is doc 3.2's "지그재그, 랜덤 방향전환" -- random zigzag.
#:     A passenger UAM and a manned light aircraft do not fly it. (A GA
#:     SAR *grid* is a systematic pattern; that is DELIVERY_WP here.)
#:   * A bird does not transit a corridor or fly a delivery route.
#:   * DEBRIS is unpowered and is pinned to PASSIVE_DRIFT by
#:     :func:`resolve_intent_type`; it is listed for completeness.
#:   * UNKNOWN is deliberately unconstrained -- being un-classifiable is the
#:     whole point of the class.
PLATFORM_INTENT_COMPATIBILITY: dict[int, frozenset[IntentType]] = {
    int(PlatformType.BIRD): frozenset({
        IntentType.SEARCH_WANDER,
        IntentType.REACTIVE_AVOID,
        IntentType.LOITER_HOVER,
    }),
    int(PlatformType.MULTIROTOR): frozenset({
        IntentType.CORRIDOR_TRANSIT,
        IntentType.DELIVERY_WP,
        IntentType.LOITER_HOVER,
        IntentType.SEARCH_WANDER,
        IntentType.REACTIVE_AVOID,
    }),
    int(PlatformType.FIXED_WING_UAS): frozenset({
        IntentType.CORRIDOR_TRANSIT,
        IntentType.DELIVERY_WP,
        IntentType.LOITER_HOVER,
        IntentType.SEARCH_WANDER,
        IntentType.REACTIVE_AVOID,
    }),
    int(PlatformType.EVTOL_UAM): frozenset({
        IntentType.CORRIDOR_TRANSIT,
        IntentType.DELIVERY_WP,
        IntentType.LOITER_HOVER,
        IntentType.REACTIVE_AVOID,
    }),
    int(PlatformType.GA): frozenset({
        IntentType.CORRIDOR_TRANSIT,
        IntentType.DELIVERY_WP,
        IntentType.LOITER_HOVER,
        IntentType.REACTIVE_AVOID,
    }),
    int(PlatformType.DEBRIS): frozenset({IntentType.PASSIVE_DRIFT}),
    int(PlatformType.UNKNOWN): frozenset({
        IntentType.CORRIDOR_TRANSIT,
        IntentType.DELIVERY_WP,
        IntentType.LOITER_HOVER,
        IntentType.SEARCH_WANDER,
        IntentType.REACTIVE_AVOID,
    }),
}


def compatible_intents(platform: PlatformType | int) -> frozenset[IntentType]:
    """Intents ``platform`` may be assigned; every intent for an unknown key."""
    return PLATFORM_INTENT_COMPATIBILITY.get(
        int(platform), frozenset(IntentType)
    )


def make_intent(
    intent_type: IntentType | int,
    agent: Any,
    world: Any,
    rng: np.random.Generator,
) -> IntentController:
    """Build the controller for ``intent_type``, applying the fallback table.

    Args:
        intent_type: the requested :class:`~core.types.IntentType`.
        agent: the agent to drive; must expose ``.state`` and ``.envelope``.
        world: the world object, or ``None``.
        rng: generator for construction-time randomness (the ``'intent'`` or a
            per-agent decision stream).

    Returns:
        A ready :class:`IntentController`; ``controller.intent_type`` is the
        effective intent and ``controller.requested_type`` is what was asked
        for, so a substitution is visible to the GUI and the metadata.
    """
    requested = IntentType(int(intent_type))
    envelope = _agent_envelope(agent)
    effective = resolve_intent_type(requested, envelope)
    cls = INTENT_CLASSES[effective]
    return cls(agent, world, rng, requested_type=requested)
