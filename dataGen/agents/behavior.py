"""Stochastic decision primitives -- the single place where randomness enters
agent behaviour.

Doc 3.3 is the design rule this module exists to serve::

    "지도에 '여기가 갈림길'이라고 좌표를 찍지 않는다.
     물체에게 확률적 결정 규칙만 부여하면 갈림길은 저절로 생긴다."

    Do NOT mark fork points on a map. Give the agents stochastic decision
    rules and the forks appear by themselves.

Every branching future in the dataset therefore has to be traceable to one of
the handful of functions below. Keeping them in one module makes the
branching structure *auditable*: if a reviewer asks "where does the
multimodality come from?", the answer is a finite list of call sites of
:func:`binary_branch`, :func:`obstacle_fork`, :func:`categorical`,
:func:`poisson_event` and :func:`reactive_avoid_decision`.

Two hard rules, enforced by convention throughout dataGen:

1. **No module-level RNG.** Every function takes an explicit
   ``numpy.random.Generator``. This is what lets
   :meth:`core.rng.RNGHub.reseed_branch` re-randomise *only* the decision
   streams during branch rollouts while wind, placement and sensor noise
   replay bit-identically (doc 3.3, "동일 초기조건 + 동일 과거 궤적에서
   시드만 바꿔 N회 롤아웃").
2. **One draw per decision.** The samplers use explicit inverse-CDF /
   single-uniform constructions rather than convenience helpers whose
   internal draw count is an implementation detail, so that the number of
   variates consumed per decision is stable and replay is exact.

Sign convention (identical to the rest of dataGen):
    ``+1`` = left = CCW = increasing psi;  ``-1`` = right = CW.

Units are SI: seconds [s], metres [m], metres per second [m/s], radians [rad].
"""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_MIN_DWELL_S",
    "DEFAULT_DWELL_CV",
    "DEFAULT_AVOID_LATCH_S",
    "sample_dwell",
    "binary_branch",
    "obstacle_fork",
    "categorical",
    "poisson_event",
    "ou_process_step",
    "sample_fraction",
    "reactive_avoid_decision",
    "new_avoid_latch",
]


# --------------------------------------------------------------------------
# Module constants
# --------------------------------------------------------------------------

#: DEFAULT minimum dwell time [s] of a maneuver mode, used only when no
#: per-platform :class:`core.state.StabilityProfile` is available.
#:
#: doc 3.3 constraint (b) -- "방금 하던 기동을 몇 초는 유지", hold the current
#: maneuver for a few seconds -- is a statement about every agent, but "a few
#: seconds" is NOT one number. 1.5 s is a bird; a passenger eVTOL holds a
#: manoeuvre for tens of seconds and re-deciding every 1.5 s would give it the
#: trace of a leaf in a breeze. The real floors live per platform in
#: :data:`agents.platforms.STABILITY_PROFILES` and reach this module as the
#: ``min_s`` argument of :func:`sample_dwell`. This constant is the value used
#: when a caller supplies no floor at all.
DEFAULT_MIN_DWELL_S: float = 1.5

#: Default coefficient of variation [-] (std / mean) of a dwell distribution.
DEFAULT_DWELL_CV: float = 0.4

#: Default latch duration [s] of a reactive-avoidance decision. Once a bird
#: has picked a side it commits to it for this long instead of re-deciding.
DEFAULT_AVOID_LATCH_S: float = 4.0


# --------------------------------------------------------------------------
# Dwell times
# --------------------------------------------------------------------------


def sample_dwell(
    rng: np.random.Generator,
    mean_s: float,
    cv: float = DEFAULT_DWELL_CV,
    min_s: float = DEFAULT_MIN_DWELL_S,
) -> float:
    """Sample a semi-Markov mode dwell time [s].

    The dwell is lognormal, parameterised by its *mean* and *coefficient of
    variation* rather than by the underlying normal moments, because those
    are the quantities a scenario author actually reasons about ("turns last
    about 5 s, give or take 40%"). For a lognormal with mean ``m`` and cv
    ``c``::

        sigma^2 = ln(1 + c^2)
        mu      = ln(m) - sigma^2 / 2

    The lognormal is right-skewed, which is the physically correct shape for
    a maneuver duration: many short corrections, a long tail of committed
    turns, and never a negative value.

    The result is finally floored at ``min_s``. That floor is not cosmetic --
    it is constraint (b) of doc 3.3. Without it the mode process degenerates
    into per-step re-decision, which produces high-frequency heading jitter
    that a constant-velocity extrapolator predicts *better* than a committed
    turn, exactly inverting the premise of the paper.

    The floor is supplied BY THE CALLER because it is a per-platform quantity
    (:attr:`core.state.StabilityProfile.dwell_min_s`), and it is applied AFTER
    the intent has scaled ``mean_s``: a wandering eVTOL draws a short mean but
    is still held to its 12 s commitment floor, so it stays far steadier than a
    wandering bird whose floor is 1 s.

    Args:
        rng: explicit generator (``mode_decision`` stream in dataGen).
        mean_s: target mean dwell [s]; must be > 0.
        cv: coefficient of variation [-] (std / mean); must be > 0.
        min_s: hard floor [s] applied after sampling; the platform's
            ``dwell_min_s``, not a global constant.

    Returns:
        Dwell time [s], always ``>= min_s``.

    Raises:
        ValueError: if ``mean_s <= 0`` or ``cv <= 0``.
    """
    m = float(mean_s)
    c = float(cv)
    if not (m > 0.0):
        raise ValueError(f"sample_dwell: mean_s must be > 0, got {mean_s!r}")
    if not (c > 0.0):
        raise ValueError(f"sample_dwell: cv must be > 0, got {cv!r}")
    sigma = math.sqrt(math.log1p(c * c))
    mu = math.log(m) - 0.5 * sigma * sigma
    d = float(rng.lognormal(mean=mu, sigma=sigma))
    floor = float(min_s)
    return d if d > floor else floor


# --------------------------------------------------------------------------
# The canonical forks
# --------------------------------------------------------------------------


def binary_branch(rng: np.random.Generator, p_left: float = 0.5) -> int:
    """The canonical fork: turn left or turn right [-].

    This single coin flip is the atom of multimodality in PRISM. It must be
    drawn from the ``branch_choice`` stream wherever the outcome is a
    genuinely open decision, because that is the stream
    :meth:`core.rng.RNGHub.reseed_branch` re-randomises to generate the
    empirical "true" future distribution used for calibration scoring.

    Args:
        rng: explicit generator (``branch_choice`` stream).
        p_left: probability [-] in ``[0, 1]`` of the left (CCW, ``+1``)
            outcome. Defaults to a fair coin.

    Returns:
        ``+1`` for left / CCW / increasing psi, ``-1`` for right / CW.

    Raises:
        ValueError: if ``p_left`` is outside ``[0, 1]``.
    """
    p = float(p_left)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"binary_branch: p_left must be in [0,1], got {p_left!r}")
    return 1 if float(rng.random()) < p else -1


def obstacle_fork(
    rng: np.random.Generator,
    pos: Sequence[float],
    psi: float,
    obstacle_centre: Sequence[float],
    p_left: float = 0.5,
) -> int:
    """Pick a side when an obstacle is met -- by coin flip, not by geometry.

    Doc 3.3 gives this as *the* worked example of how forks are created:
    "장애물 만나면 좌/우 중 무작위 선택" -- on meeting an obstacle, choose
    left or right at random.

    **The sign is deliberately NOT a function of the geometry.** Choosing the
    "cheaper" side would make the future a deterministic function of the
    past, and a deterministic future is exactly what a constant-velocity
    extrapolator (and any point-estimate predictor) can learn to nail. The
    whole paper rests on the opposite property: *identical pasts must lead to
    genuinely different futures*, so that a calibrated probabilistic
    predictor beats a point estimate for a reason that is real and not an
    artefact of noise.

    The geometry arguments are still taken, and are used only to compute the
    bookkeeping returned through the module docstring's audit trail (they let
    a caller log which obstacle triggered the fork and how the agent was
    oriented at the time). They never influence the returned sign.

    Args:
        rng: explicit generator (``branch_choice`` stream).
        pos: agent position ``(x, y)`` [m].
        psi: agent heading [rad], CCW from +X.
        obstacle_centre: obstacle centre ``(x, y)`` [m].
        p_left: probability [-] of the left (``+1``) outcome; keep at 0.5
            unless a scenario deliberately models a side bias (e.g. a
            right-of-way rule), in which case the bias must be documented in
            the scenario config.

    Returns:
        ``+1`` for left / CCW, ``-1`` for right / CW.

    Raises:
        ValueError: if ``pos`` or ``obstacle_centre`` is not length 2.
    """
    if len(pos) != 2 or len(obstacle_centre) != 2:
        raise ValueError(
            "obstacle_fork: pos and obstacle_centre must be 2-element (x, y) [m]"
        )
    # Geometry is read but intentionally unused for the decision; touching it
    # here keeps the call signature honest and makes the "no geometry in the
    # sign" property visible at the one place it matters.
    float(psi)
    float(pos[0]) - float(obstacle_centre[0])
    return binary_branch(rng, p_left)


def categorical(rng: np.random.Generator, probs: Sequence[float]) -> int:
    """Sample an index from a categorical distribution, one uniform draw.

    Implemented as an explicit inverse-CDF lookup so that exactly one variate
    is consumed per decision regardless of the number of categories. That
    stability matters for replay: adding a category must not shift the number
    of draws taken by unrelated later decisions in the same stream.

    Args:
        rng: explicit generator.
        probs: non-negative weights [-]; they are normalised internally, so
            an unnormalised weight vector is accepted.

    Returns:
        Index in ``[0, len(probs))``.

    Raises:
        ValueError: if ``probs`` is empty, contains a negative or non-finite
            entry, or sums to zero.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("categorical: probs must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(p)):
        raise ValueError("categorical: probs must be finite")
    if np.any(p < 0.0):
        raise ValueError("categorical: probs must be non-negative")
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError("categorical: probs must sum to a positive value")
    u = float(rng.random()) * total
    acc = 0.0
    last = int(p.size - 1)
    for i in range(last):
        acc += float(p[i])
        if u < acc:
            return i
    return last


def poisson_event(rng: np.random.Generator, rate_hz: float, dt: float) -> bool:
    """Did a Poisson event of rate ``rate_hz`` occur within ``dt``?

    Uses the exact Bernoulli probability ``1 - exp(-rate * dt)`` rather than
    the small-``dt`` approximation ``rate * dt``, so the event rate stays
    correct even if a caller uses the logging timestep (0.5 s) instead of the
    integration timestep (0.05 s).

    Args:
        rng: explicit generator.
        rate_hz: event rate [1/s]; must be >= 0. Zero never fires.
        dt: elapsed time [s]; must be >= 0.

    Returns:
        True if at least one event occurred in the interval.

    Raises:
        ValueError: if ``rate_hz`` or ``dt`` is negative.
    """
    r = float(rate_hz)
    d = float(dt)
    if r < 0.0:
        raise ValueError(f"poisson_event: rate_hz must be >= 0, got {rate_hz!r}")
    if d < 0.0:
        raise ValueError(f"poisson_event: dt must be >= 0, got {dt!r}")
    if r == 0.0 or d == 0.0:
        return False
    p = -math.expm1(-r * d)
    return bool(float(rng.random()) < p)


def ou_process_step(
    x: float,
    sigma: float,
    tau: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """Advance a scalar Ornstein-Uhlenbeck process by ``dt``.

    Exact (not Euler-Maruyama) discretisation of ``dx = -x/tau dt + ... dW``
    with stationary standard deviation ``sigma``::

        alpha = exp(-dt / tau)
        x'    = alpha * x + sigma * sqrt(1 - alpha^2) * N(0, 1)

    Being exact means the stationary variance is ``sigma^2`` for *any* step
    size, so the same gust or wander model behaves identically whether it is
    ticked at 20 Hz (SIM_DT) or 2 Hz (LOG_DT). ``sigma`` is the stationary
    std of ``x`` itself, NOT the diffusion coefficient.

    Used for (a) wind gusts in :mod:`environment.wind` and (b) slow heading
    drift in the search/wander intent, where an OU process gives correlated
    wandering instead of white-noise jitter.

    Args:
        x: current process value; units are the caller's ([m/s] for gusts,
            [rad] for heading drift).
        sigma: stationary standard deviation, same units as ``x``; >= 0.
        tau: correlation time [s]; must be > 0.
        dt: timestep [s]; must be >= 0.
        rng: explicit generator (``wind`` or ``mode_decision`` stream).

    Returns:
        The new process value, same units as ``x``.

    Raises:
        ValueError: if ``tau <= 0``, ``sigma < 0`` or ``dt < 0``.
    """
    s = float(sigma)
    t = float(tau)
    d = float(dt)
    if t <= 0.0:
        raise ValueError(f"ou_process_step: tau must be > 0, got {tau!r}")
    if s < 0.0:
        raise ValueError(f"ou_process_step: sigma must be >= 0, got {sigma!r}")
    if d < 0.0:
        raise ValueError(f"ou_process_step: dt must be >= 0, got {dt!r}")
    if d == 0.0:
        return float(x)
    alpha = math.exp(-d / t)
    noise_scale = s * math.sqrt(max(0.0, 1.0 - alpha * alpha))
    return float(alpha * float(x) + noise_scale * float(rng.standard_normal()))


def sample_fraction(
    rng: np.random.Generator,
    lo: float = 0.3,
    hi: float = 1.0,
) -> float:
    """Sample an aggressiveness fraction [-] uniformly in ``[lo, hi]``.

    Used when a maneuver mode is entered to decide *how hard* the maneuver
    is, as a fraction of the platform's physical limit. The lower bound is
    deliberately well above zero: a fraction near zero would make the mode
    indistinguishable from constant velocity, which both wastes a dwell
    period and blurs the mode label stored as ground truth.

    Args:
        rng: explicit generator (``mode_decision`` stream).
        lo: lower bound [-].
        hi: upper bound [-]; must be >= ``lo``.

    Returns:
        A fraction in ``[lo, hi]``.

    Raises:
        ValueError: if ``hi < lo``.
    """
    a = float(lo)
    b = float(hi)
    if b < a:
        raise ValueError(f"sample_fraction: hi ({hi!r}) must be >= lo ({lo!r})")
    if b == a:
        return a
    return a + (b - a) * float(rng.random())


# --------------------------------------------------------------------------
# Reactive avoidance (IntentType.REACTIVE_AVOID, doc 3.2 rev.2)
# --------------------------------------------------------------------------


def new_avoid_latch() -> dict[str, Any]:
    """Create a fresh, inactive avoidance latch.

    The latch is a plain JSON-safe dict so it can live directly inside
    :attr:`core.state.AgentSnapshot.intent_state` and survive
    snapshot/restore for branch rollouts.

    Returns:
        ``{"active": False, "sign": 0, "timer": 0.0, "trigger_count": 0}``
        where ``timer`` is the remaining commitment time [s].
    """
    return {"active": False, "sign": 0, "timer": 0.0, "trigger_count": 0}


def reactive_avoid_decision(
    rng: np.random.Generator,
    rel_pos: Sequence[float],
    rel_vel: Sequence[float],
    react_radius: float,
    latch: MutableMapping[str, Any] | Mapping[str, Any] | None,
    dt: float = 0.0,
    latch_s: float = DEFAULT_AVOID_LATCH_S,
    p_left: float = 0.5,
    tcpa_max_s: float = float("inf"),
    cpa_max_m: float = float("inf"),
    armed: bool = True,
) -> tuple[bool, int]:
    """Decide whether to break away from an approaching aircraft, and which way.

    Models the bird-avoidance behaviour of ``IntentType.REACTIVE_AVOID``
    (doc 3.2 rev.2: "반응 회피 -- 자기 항공기 접근에 반응하여 진로 변경").
    This is THE reactive-avoidance model: ``agents.intent.ReactiveAvoidIntent``
    calls it directly, and there is deliberately no second implementation of
    the trigger anywhere in dataGen.

    Four conditions must hold simultaneously for a trigger:

    1. **Proximity** -- ``|rel_pos| < react_radius``.
    2. **Closing** -- ``rel_pos . rel_vel < 0``, i.e. the range is shrinking.
       Without this term a bird would flare at an aircraft that is already
       past and opening, which is neither realistic nor useful.
    3. **Imminent** -- the constant-velocity time to closest approach is at
       most ``tcpa_max_s``.
    4. **Threatening** -- the projected miss distance at that CPA is below
       ``cpa_max_m``. Conditions 3 and 4 are what make the reaction a response
       to a *developing conflict* rather than to mere proximity: an aircraft
       700 m away on a track that will pass 900 m clear is not a threat.

    Conditions 3 and 4 default to "off" (infinite thresholds), which reduces
    the trigger to the proximity-and-closing model.

    Once triggered the decision **latches**: the side is a coin flip taken
    once, and it is then held for ``latch_s`` seconds. The latch is the
    reactive-intent counterpart of the mode-dwell floor -- it is what makes
    the bird *commit* to one side and therefore what makes the two outcomes
    separate into two distinguishable future modes rather than smearing into
    one indecisive blur.

    The latch dict is mutated in place when it is mutable, so the caller's
    ``intent_state`` stays up to date without a second bookkeeping step.

    Args:
        rng: explicit generator (``branch_choice`` stream -- the side choice
            is a genuine fork and must be re-randomised in branch rollouts).
        rel_pos: position of the threat relative to this agent ``(dx, dy)``
            [m], i.e. ``threat_pos - own_pos``.
        rel_vel: velocity of the threat relative to this agent ``(dvx, dvy)``
            [m/s], i.e. ``threat_vel - own_vel``.
        react_radius: reaction range [m]; a trigger requires the slant range
            to be below this.
        latch: latch state from :func:`new_avoid_latch`, or ``None`` for a
            stateless one-shot evaluation.
        dt: time [s] elapsed since the previous call, used to run down the
            latch timer. Pass 0.0 to inspect without ageing the latch.
        latch_s: commitment duration [s] applied when a new trigger fires.
        p_left: probability [-] of choosing the left (``+1``) side. The caller
            biases this away from the threat; ``0.5`` is the genuinely
            ambiguous head-on / dead-astern case, where the side becomes a
            pure fork.
        tcpa_max_s: react only when the projected CPA is this soon [s].
        cpa_max_m: react only when the projected miss distance is below this
            [m].
        armed: when False the geometry is not tested at all (an existing latch
            still ages normally). Used for the post-manoeuvre refractory
            period, so a relaxing agent cannot immediately re-trigger.

    Returns:
        ``(active, sign)`` where ``active`` is True while an avoidance
        maneuver should be executed and ``sign`` is ``+1`` (left / CCW) or
        ``-1`` (right / CW). When ``active`` is False, ``sign`` is ``0``.

    Raises:
        ValueError: if ``rel_pos`` or ``rel_vel`` is not length 2, or if
            ``react_radius`` is negative.
    """
    if len(rel_pos) != 2 or len(rel_vel) != 2:
        raise ValueError(
            "reactive_avoid_decision: rel_pos and rel_vel must be 2-element vectors"
        )
    r_react = float(react_radius)
    if r_react < 0.0:
        raise ValueError(
            f"reactive_avoid_decision: react_radius must be >= 0, got {react_radius!r}"
        )

    state: dict[str, Any]
    if latch is None:
        state = new_avoid_latch()
    else:
        state = latch if isinstance(latch, dict) else dict(latch)
        state.setdefault("active", False)
        state.setdefault("sign", 0)
        state.setdefault("timer", 0.0)
        state.setdefault("trigger_count", 0)

    # 1. Age the existing commitment.
    if bool(state["active"]):
        timer = float(state["timer"]) - max(0.0, float(dt))
        if timer > 0.0:
            state["timer"] = timer
            return True, int(state["sign"])
        state["active"] = False
        state["sign"] = 0
        state["timer"] = 0.0

    # 2. No live commitment -> test the trigger geometry.
    if not armed:
        return False, 0
    dx = float(rel_pos[0])
    dy = float(rel_pos[1])
    vx = float(rel_vel[0])
    vy = float(rel_vel[1])
    rng_sq = dx * dx + dy * dy
    dot = dx * vx + dy * vy
    if rng_sq >= r_react * r_react or dot >= 0.0:
        return False, 0

    # Constant-velocity CPA of the relative state. dot < 0 above guarantees a
    # positive tcpa, so no clamping is needed here.
    v2 = vx * vx + vy * vy
    if v2 <= 1e-12:
        return False, 0
    tcpa = -dot / v2
    if tcpa > float(tcpa_max_s):
        return False, 0
    mx = dx + vx * tcpa
    my = dy + vy * tcpa
    if math.hypot(mx, my) >= float(cpa_max_m):
        return False, 0

    sign = binary_branch(rng, p_left)
    state["active"] = True
    state["sign"] = int(sign)
    state["timer"] = float(latch_s)
    state["trigger_count"] = int(state["trigger_count"]) + 1
    return True, int(sign)
