"""Agent state, performance envelope and snapshot containers.

Units (doc convention, repeated on every field below):
    position   [m]      in the local tangent plane, +X East, +Y North
    speed      [m/s]    AIRSPEED magnitude (ground speed for unpowered bodies)
    heading    [rad]    psi, CCW from +X, wrapped to (-pi, pi]
    turn rate  [rad/s]  positive = left = CCW = increasing psi
    accel      [m/s^2]  longitudinal, along the body heading

The snapshot dataclasses exist for one reason: doc 3.3 requires branch
rollouts that restart from an identical situation and re-randomise only the
decision streams. A snapshot must therefore FULLY determine the future
evolution of an agent -- kinematic state, maneuver mode and its dwell timer,
intent bookkeeping, and the RNG bit-generator state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from core.constants import EPS_V, G, TWO_PI
from core.types import ManeuverMode

__all__ = [
    "wrap_angle",
    "angle_diff",
    "AgentState",
    "StabilityProfile",
    "DEFAULT_STABILITY",
    "PerformanceEnvelope",
    "AgentSnapshot",
    "WorldSnapshot",
]


# --------------------------------------------------------------------------
# Small angle utilities -- used by every module, defined once here.
# --------------------------------------------------------------------------


def wrap_angle(a: float) -> float:
    """Wrap an angle [rad] to the half-open interval (-pi, pi].

    Note the interval is open at -pi and closed at +pi, so exactly -pi maps to
    +pi. This is the canonical wrap for every heading in dataGen.

    Args:
        a: angle [rad], any magnitude.

    Returns:
        The equivalent angle [rad] in (-pi, pi].
    """
    w = math.fmod(a + math.pi, TWO_PI)
    if w <= 0.0:
        w += TWO_PI
    return w - math.pi


def angle_diff(a: float, b: float) -> float:
    """Signed smallest difference ``a - b`` [rad], wrapped to (-pi, pi].

    Args:
        a: first angle [rad].
        b: second angle [rad].

    Returns:
        The signed angular difference [rad]; positive means ``a`` is CCW of
        ``b``.
    """
    return wrap_angle(a - b)


# --------------------------------------------------------------------------
# Kinematic state
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AgentState:
    """Instantaneous kinematic state of one agent (2D, flat-earth).

    Attributes:
        x: East position [m].
        y: North position [m].
        v: AIRSPEED magnitude [m/s]. For unpowered bodies (DEBRIS) this is
            interpreted as the ground speed, because such bodies are advected
            with the air mass and have no airspeed of their own.
        psi: heading [rad], CCW from +X, wrapped to (-pi, pi].
        omega: CURRENT turn rate [rad/s]. This is a state, not a command: the
            integrator rate-limits it toward the commanded value using
            ``PerformanceEnvelope.omega_dot_max`` (roll onset).
        a: CURRENT longitudinal acceleration [m/s^2], along the heading.
    """

    x: float
    y: float
    v: float
    psi: float
    omega: float = 0.0
    a: float = 0.0

    def ground_velocity(
        self,
        wind_xy: Sequence[float],
        susceptibility: float,
    ) -> tuple[float, float]:
        """Ground velocity [m/s] = air-relative velocity + advected wind.

        Args:
            wind_xy: ambient wind vector ``(wx, wy)`` [m/s] in world axes
                (the direction the air is moving TOWARD).
            susceptibility: fraction 0..1 of the wind that advects this body
                (``PerformanceEnvelope.wind_susceptibility``). Pass 0.0 for a
                body whose ``v`` already is a ground speed, so the wind is not
                counted twice.

        Returns:
            ``(vx, vy)`` ground velocity components [m/s], +X East, +Y North.
        """
        s = float(susceptibility)
        vx = self.v * math.cos(self.psi) + s * float(wind_xy[0])
        vy = self.v * math.sin(self.psi) + s * float(wind_xy[1])
        return vx, vy

    def air_velocity(self) -> tuple[float, float]:
        """Air-relative velocity ``(vx, vy)`` [m/s], ignoring wind."""
        return self.v * math.cos(self.psi), self.v * math.sin(self.psi)

    def normalized(self) -> "AgentState":
        """Return a copy whose heading is wrapped to (-pi, pi]."""
        return AgentState(self.x, self.y, self.v, wrap_angle(self.psi),
                          self.omega, self.a)

    def copy(self) -> "AgentState":
        """Return an independent copy of this state."""
        return AgentState(self.x, self.y, self.v, self.psi, self.omega, self.a)

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain JSON-safe dict of floats (SI units)."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "v": float(self.v),
            "psi": float(self.psi),
            "omega": float(self.omega),
            "a": float(self.a),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentState":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            v=float(d["v"]),
            psi=float(d["psi"]),
            omega=float(d.get("omega", 0.0)),
            a=float(d.get("a", 0.0)),
        )


# --------------------------------------------------------------------------
# Manoeuvre stability (how COMMITTED a platform is, doc 3.3 (b))
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StabilityProfile:
    """How committed a platform is to the manoeuvre it is currently flying.

    The performance envelope says what a platform *cannot* do. This says what
    it *would not* do: a passenger eVTOL in corridor cruise does not re-decide
    its manoeuvre every couple of seconds, and a bird does. Manoeuvre
    commitment is therefore a PLATFORM PROPERTY, not one global constant --
    the earlier global 1.5 s dwell floor was right for a bird and absurd for a
    UAM aircraft, and doc 7.2 warns that the mushy wandering trace it produces
    is exactly what a reviewer notices first.

    The per-platform table lives in :data:`agents.platforms.STABILITY_PROFILES`
    and is attached to the envelope by :func:`agents.platforms.make_envelope`;
    :class:`agents.maneuver.ModeMachine` is the only consumer. The dataclass is
    frozen so that one instance can be shared safely as a default, and because
    an agent's commitment must not drift during an episode -- if it could, it
    would have to be part of :class:`AgentSnapshot` for branch rollouts to be
    exact.

    Attributes:
        dwell_min_s: hard floor [s] on the dwell of any manoeuvre mode. Applied
            AFTER the intent multiplier, so a wandering eVTOL is still far
            steadier than a wandering bird.
        dwell_mean_s: mean dwell [s] of a manoeuvre mode BEFORE the per-intent
            multiplier of :data:`agents.platforms.INTENT_DWELL_MULTIPLIER`.
            ``None`` selects the legacy per-(intent, mode) table
            :data:`agents.maneuver.DWELL_MEAN_S`, which is what a bare
            envelope built outside :mod:`agents.platforms` gets.
        turn_frac_lo: lower end [-] of the ``U(lo, hi)`` fraction of
            :meth:`PerformanceEnvelope.max_turn_rate` a CT mode commits to.
        turn_frac_hi: upper end [-] of that fraction.
        mode_authority: 0..1 [-], how strongly the maneuver mode overrides the
            intent's guidance command::

                omega_cmd = (1 - authority) * omega_intent
                            + authority * omega_mode

            1.0 is a full override (a bird's committed, decisive turn); 0.35
            merely biases an aircraft that stays glued to its corridor
            centreline.
        diag_bias: extra probability mass [-] added to the diagonal of the
            intent's transition matrix, ``P' = (1-b) P + b I`` renormalised.
            Positive makes a platform hold its current mode; NEGATIVE (only
            ``UNKNOWN``) makes it switch more often than any trained class,
            which is one of the two ways the OOD class leaves the training
            distribution.
    """

    dwell_min_s: float = 1.5
    dwell_mean_s: float | None = None
    turn_frac_lo: float = 0.30
    turn_frac_hi: float = 1.00
    mode_authority: float = 1.0
    diag_bias: float = 0.0

    @property
    def turn_fraction_range(self) -> tuple[float, float]:
        """The ``(lo, hi)`` turn-aggressiveness band [-], ordered."""
        lo = float(self.turn_frac_lo)
        hi = float(self.turn_frac_hi)
        return (lo, hi) if lo <= hi else (hi, lo)

    def dwell_mean_for(self, intent_multiplier: float, fallback_s: float) -> float:
        """Effective mean dwell [s] under an intent multiplier.

        Args:
            intent_multiplier: the intent's dwell multiplier [-] (see
                :data:`agents.platforms.INTENT_DWELL_MULTIPLIER`).
            fallback_s: mean dwell [s] to use when this profile carries no
                ``dwell_mean_s`` (the legacy per-(intent, mode) value).

        Returns:
            A strictly positive mean dwell [s]. Note the floor
            :attr:`dwell_min_s` is NOT applied here: it is applied to the
            sampled dwell, after the multiplier, by
            :func:`agents.behavior.sample_dwell`.
        """
        if self.dwell_mean_s is None:
            base = float(fallback_s)
        else:
            base = float(self.dwell_mean_s) * float(intent_multiplier)
        return base if base > 1e-6 else 1e-6

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the profile is usable."""
        problems: list[str] = []
        if not (self.dwell_min_s > 0.0):
            problems.append(f"dwell_min_s must be > 0, got {self.dwell_min_s!r}")
        if self.dwell_mean_s is not None and not (self.dwell_mean_s > 0.0):
            problems.append(f"dwell_mean_s must be > 0 or None, got {self.dwell_mean_s!r}")
        lo, hi = self.turn_fraction_range
        if not (0.0 <= lo <= hi <= 1.0):
            problems.append(f"turn fraction band must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
        if not (0.0 <= self.mode_authority <= 1.0):
            problems.append(f"mode_authority must be in [0,1], got {self.mode_authority!r}")
        if not (-1.0 < self.diag_bias < 1.0):
            problems.append(f"diag_bias must be in (-1,1), got {self.diag_bias!r}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-safe dict (seconds and dimensionless)."""
        return {
            "dwell_min_s": float(self.dwell_min_s),
            "dwell_mean_s": (None if self.dwell_mean_s is None
                             else float(self.dwell_mean_s)),
            "turn_frac_lo": float(self.turn_frac_lo),
            "turn_frac_hi": float(self.turn_frac_hi),
            "mode_authority": float(self.mode_authority),
            "diag_bias": float(self.diag_bias),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StabilityProfile":
        """Rebuild from :meth:`to_dict` output; ``None`` gives the default."""
        if not d:
            return DEFAULT_STABILITY
        mean = d.get("dwell_mean_s", None)
        return cls(
            dwell_min_s=float(d.get("dwell_min_s", 1.5)),
            dwell_mean_s=None if mean is None else float(mean),
            turn_frac_lo=float(d.get("turn_frac_lo", 0.30)),
            turn_frac_hi=float(d.get("turn_frac_hi", 1.00)),
            mode_authority=float(d.get("mode_authority", 1.0)),
            diag_bias=float(d.get("diag_bias", 0.0)),
        )


#: Profile used when nothing better is known: a full-authority mode machine on
#: the legacy per-(intent, mode) dwell table with a 1.5 s floor. It reproduces
#: the pre-stability-model behaviour EXACTLY, so a :class:`PerformanceEnvelope`
#: built by hand (unit tests, GUI probes) behaves as it always did. Every
#: envelope built through :mod:`agents.platforms` carries a real per-platform
#: profile instead.
DEFAULT_STABILITY: StabilityProfile = StabilityProfile()


# --------------------------------------------------------------------------
# Performance envelope
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PerformanceEnvelope:
    """Hard physical limits of one platform class (doc 3.2 (3)).

    The envelope is what an agent CANNOT do. Decision logic proposes; this
    object clips. Keeping the clipping here (and only here) is what makes the
    "physically possible only" constraint of doc 3.3 (a) auditable.

    Attributes:
        v_min: minimum airspeed [m/s] (0.0 for hover-capable platforms).
        v_max: maximum airspeed [m/s].
        a_max: maximum longitudinal acceleration [m/s^2], positive.
        a_min: maximum longitudinal deceleration [m/s^2], NEGATIVE.
        omega_max: absolute turn-rate cap [rad/s], always applied last.
        omega_dot_max: turn-rate rate limit [rad/s^2] (roll onset), positive.
        bank_max: maximum bank angle [rad] for bank-limited platforms.
        bank_limited: True for aerodynamically banked turns; the turn-rate cap
            then falls as G*tan(bank_max)/v.
        lateral_accel_max: maximum lateral acceleration [m/s^2] from tilt, used
            for multirotor-style platforms (bank_limited False, can_hover True).
        can_hover: True if the platform may command zero airspeed.
        is_powered: False for DEBRIS (no thrust, purely advected/ballistic).
        wind_susceptibility: fraction 0..1 of the ambient wind that advects
            this body [-]. Light bodies approach 1.0, heavy fast bodies less.
        mass: body mass [kg]. Passive dynamics only (unpowered bodies).
        drag_area: drag area Cd*A [m^2]. Passive dynamics only.
        stability: how committed the platform is to a manoeuvre once it starts
            one (see :class:`StabilityProfile`). Physically it belongs with the
            envelope -- it is a property of the airframe and its mission, fixed
            for the lifetime of an agent -- so it rides along here and is
            deliberately NOT part of :class:`AgentSnapshot`.
    """

    v_min: float
    v_max: float
    a_max: float
    a_min: float
    omega_max: float
    omega_dot_max: float
    bank_max: float
    bank_limited: bool
    lateral_accel_max: float
    can_hover: bool
    is_powered: bool
    wind_susceptibility: float
    mass: float
    drag_area: float
    stability: StabilityProfile = DEFAULT_STABILITY

    def max_turn_rate(self, v: float) -> float:
        """Effective, speed-dependent turn-rate cap [rad/s] at airspeed ``v``.

        Three regimes (exactly as specified in the dataGen contract):
            * tilt limited  (not bank_limited and can_hover):
                  cap = lateral_accel_max / max(v, EPS_V)
            * bank limited  (bank_limited):
                  cap = G * tan(bank_max) / max(v, EPS_V)
            * otherwise:
                  cap = omega_max
        The result is always additionally clipped by ``omega_max``.

        Args:
            v: airspeed [m/s]; may be zero or negative-ish noise, it is
                floored by ``EPS_V`` inside the denominator.

        Returns:
            Non-negative maximum magnitude of the turn rate [rad/s].
        """
        vv = v if v > EPS_V else EPS_V
        if (not self.bank_limited) and self.can_hover:
            cap = self.lateral_accel_max / vv
        elif self.bank_limited:
            cap = G * math.tan(self.bank_max) / vv
        else:
            cap = self.omega_max
        return min(cap, self.omega_max)

    def clamp_speed(self, v: float) -> float:
        """Clip an airspeed [m/s] to ``[v_min, v_max]``."""
        if v < self.v_min:
            return self.v_min
        if v > self.v_max:
            return self.v_max
        return v

    def clamp_accel(self, a: float) -> float:
        """Clip a longitudinal acceleration [m/s^2] to ``[a_min, a_max]``."""
        if a < self.a_min:
            return self.a_min
        if a > self.a_max:
            return self.a_max
        return a

    def clamp_turn_rate(self, omega: float, v: float) -> float:
        """Clip a turn rate [rad/s] to +/- :meth:`max_turn_rate` at speed ``v``."""
        cap = self.max_turn_rate(v)
        if omega > cap:
            return cap
        if omega < -cap:
            return -cap
        return omega

    def turn_radius(self, v: float) -> float:
        """Minimum turn radius [m] at airspeed ``v`` [m/s].

        Returns ``inf`` when the effective turn-rate cap is zero (a body that
        cannot turn at all).
        """
        cap = self.max_turn_rate(v)
        if cap <= 0.0:
            return float("inf")
        return max(v, 0.0) / cap

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-safe dict (SI units)."""
        return {
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "a_max": float(self.a_max),
            "a_min": float(self.a_min),
            "omega_max": float(self.omega_max),
            "omega_dot_max": float(self.omega_dot_max),
            "bank_max": float(self.bank_max),
            "bank_limited": bool(self.bank_limited),
            "lateral_accel_max": float(self.lateral_accel_max),
            "can_hover": bool(self.can_hover),
            "is_powered": bool(self.is_powered),
            "wind_susceptibility": float(self.wind_susceptibility),
            "mass": float(self.mass),
            "drag_area": float(self.drag_area),
            "stability": self.stability.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerformanceEnvelope":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            v_min=float(d["v_min"]),
            v_max=float(d["v_max"]),
            a_max=float(d["a_max"]),
            a_min=float(d["a_min"]),
            omega_max=float(d["omega_max"]),
            omega_dot_max=float(d["omega_dot_max"]),
            bank_max=float(d["bank_max"]),
            bank_limited=bool(d["bank_limited"]),
            lateral_accel_max=float(d["lateral_accel_max"]),
            can_hover=bool(d["can_hover"]),
            is_powered=bool(d["is_powered"]),
            wind_susceptibility=float(d["wind_susceptibility"]),
            mass=float(d["mass"]),
            drag_area=float(d["drag_area"]),
            stability=StabilityProfile.from_dict(d.get("stability")),
        )


# --------------------------------------------------------------------------
# Snapshots (branch-rollout support, doc 3.3)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AgentSnapshot:
    """Everything needed to resume one agent bit-exactly.

    Attributes:
        state: kinematic state (see :class:`AgentState`).
        mode: currently executed maneuver mode.
        mode_timer: seconds [s] remaining in the current mode dwell. Dwell is
            the doc 3.3 (b) constraint that prevents per-step re-decision
            chatter.
        mode_params: mode payload, e.g. ``{"omega_cmd": 0.12}`` [rad/s] for a
            coordinated turn or ``{"a_cmd": -1.5}`` [m/s^2] for CA. Plain
            JSON-safe scalars only.
        intent_state: intent bookkeeping, e.g. waypoint index, loiter centre
            [m], avoidance latch flags. Plain JSON-safe values only.
        rng_state: ``numpy`` Generator ``bit_generator.state`` dict for this
            agent's private stream, or an empty dict if the agent draws only
            from shared hub streams.
    """

    state: AgentState
    mode: ManeuverMode = ManeuverMode.CV
    mode_timer: float = 0.0
    mode_params: dict[str, Any] = field(default_factory=dict)
    intent_state: dict[str, Any] = field(default_factory=dict)
    rng_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-safe dict."""
        return {
            "state": self.state.to_dict(),
            "mode": int(self.mode),
            "mode_timer": float(self.mode_timer),
            "mode_params": dict(self.mode_params),
            "intent_state": dict(self.intent_state),
            "rng_state": dict(self.rng_state),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentSnapshot":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            state=AgentState.from_dict(d["state"]),
            mode=ManeuverMode(int(d.get("mode", 0))),
            mode_timer=float(d.get("mode_timer", 0.0)),
            mode_params=dict(d.get("mode_params", {})),
            intent_state=dict(d.get("intent_state", {})),
            rng_state=dict(d.get("rng_state", {})),
        )


@dataclass(slots=True)
class WorldSnapshot:
    """Complete, resumable state of the whole simulation at one instant.

    Attributes:
        t: simulation time [s] since episode start.
        ownship: snapshot of the ownship agent.
        agents: snapshots of all traffic agents, index-aligned with
            ``agent_ids``.
        agent_ids: stable integer ids of the traffic agents [-].
        rng_state: :meth:`core.rng.RNGHub.snapshot` output for all live
            streams (wind, sensor, decision, ...).
    """

    t: float
    ownship: AgentSnapshot
    agents: list[AgentSnapshot] = field(default_factory=list)
    agent_ids: list[int] = field(default_factory=list)
    rng_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-safe dict."""
        return {
            "t": float(self.t),
            "ownship": self.ownship.to_dict(),
            "agents": [a.to_dict() for a in self.agents],
            "agent_ids": [int(i) for i in self.agent_ids],
            "rng_state": dict(self.rng_state),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorldSnapshot":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            t=float(d["t"]),
            ownship=AgentSnapshot.from_dict(d["ownship"]),
            agents=[AgentSnapshot.from_dict(a) for a in d.get("agents", [])],
            agent_ids=[int(i) for i in d.get("agent_ids", [])],
            rng_state=dict(d.get("rng_state", {})),
        )
