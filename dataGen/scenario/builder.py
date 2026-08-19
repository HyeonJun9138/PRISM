"""Assemble a runnable :class:`Scenario` from a :class:`ScenarioConfig`.

This module is the single integration point of dataGen: it wires the world,
the wind field, the ownship, the designed encounters (doc 3.4) and the
background traffic together, and hands :mod:`simulation.engine` an object that
needs no further construction.

Three structural decisions are recorded here, because none of them is obvious
from the configuration alone.

**1. Density footprint vs simulated arena.**
``ScenarioConfig.area_km2`` is the *density reference footprint* over which
doc 8.1's ``density_per_km2`` is defined -- see :mod:`scenario.presets` for why
that area is 0.25 km^2. A 500 m box is far too small to fly a 60 s cruise leg
in, so the simulated arena is a separate, larger region: the
:class:`environment.world.World` is built from a config copy whose
``area_km2`` is :data:`ARENA_AREA_KM2`.

``density_per_km2`` is therefore the doc 8.1 SPLIT AXIS -- the knob that sets
how many aircraft an episode contains -- and not the areal density realised
inside the arena. The realised figure is
:attr:`Scenario.realised_density_per_km2`; report that one whenever a paper
figure is about spacing rather than about the split. The two differ by the
ratio of the arena to the reference footprint, and pretending otherwise would
put an aircraft every ~150 m, at which point an NMAC is unavoidable in every
single episode and doc 3.4's stratification has nothing left to stratify.

**2. Where multimodality comes from.**
Nothing here places a waypoint on a map to create a fork -- doc 3.3 forbids
exactly that. Every branch point is a stochastic decision taken at run time by
:mod:`agents.intent` / :mod:`agents.maneuver`, drawn from the ``branch_choice``
stream so that :meth:`core.rng.RNGHub.reseed_branch` can re-randomise it and
only it.

**3. Bound decision streams must be re-bound after a reseed.**
:meth:`agents.maneuver.ModeMachine.bind_streams` stores generator REFERENCES,
while :meth:`core.rng.RNGHub.reseed_branch` REPLACES the generator objects in
the hub. A machine bound before a reseed would keep drawing from the
pre-branch generator, every branch would replay identically, and the empirical
"true" future distribution -- the whole calibration ground truth -- would
collapse to a point. :meth:`Scenario.bind_streams` exists to be called again
after every reseed and after every restore;
:class:`simulation.engine.SimulationEngine` does so automatically.

Units are SI: metres, seconds, radians, m/s, kilograms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from agents.base import Agent
from agents.intent import compatible_intents, make_intent, resolve_intent_type
from agents.reaction import ReactionLayer, reaction_profile
from agents.maneuver import ModeMachine
from agents.platforms import is_ood_platform, make_envelope
from core.kinematics import step_powered
from core.rng import RNGHub
from core.state import AgentSnapshot, AgentState, PerformanceEnvelope, wrap_angle
from core.types import IntentType, ManeuverMode, PlatformType
from environment.encounter import build_encounter, sample_geometry
from environment.wind import WindField
from environment.world import World
from scenario.config import ScenarioConfig
from sensors.model import SensorModel
from sensors.tracker import TrackManager

__all__ = [
    "ARENA_AREA_KM2",
    "BACKENDS",
    "resolve_intent",
    "ground_velocity_of",
    "Ownship",
    "DesignedEncounter",
    "Scenario",
    "build_scenario",
]


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Area [km^2] of the simulated arena: a 6 km x 6 km box.
#:
#: The lower bound is flight geometry -- a 55 m/s ownship covers 3300 m in a
#: 60 s episode, and the leg must fit inside the arena with margin or the doc
#: 3.4 inverse design stops being exact over the whole ``tcpa`` range.
#:
#: The value above that bound was chosen by measurement, trading two things
#: that pull in opposite directions at a FIXED agent count (doc 3.7 fixes it
#: at 5-15):
#:
#:   * shrink the arena and the ownship's swept corridor covers more of it, so
#:     background traffic manufactures NMACs on its own -- at 16 km^2 roughly
#:     70% of episodes end up NMAC and doc 3.4's stratification degenerates to
#:     a single risk class;
#:   * grow the arena and traffic drifts outside the 3 km sensor horizon, so
#:     the observation layer empties out -- at 64 km^2 a quarter of the agents
#:     are never detected at all.
#:
#: At 36 km^2 roughly half the episodes reach NMAC (a clean split of the
#: designed encounters into violating and non-violating cases, which is
#: exactly what doc 3.4 asks for), about half of all logged agent-steps carry
#: a detection, and ~90% of agents are seen at least once.
ARENA_AREA_KM2: float = 36.0

#: Minimum designed-encounter range [m] at t = 0. Closer than this and the
#: encounter is effectively already at CPA.
_MIN_DESIGN_RANGE_M: float = 200.0

#: Designed-encounter range cap [m] as a multiple of the sensor's maximum
#: range, so an intruder is at least plausibly acquirable at spawn.
_DESIGN_RANGE_SENSOR_FACTOR: float = 1.25

#: Which module backs each role. Recorded in the episode manifest so a dataset
#: can always be traced to the code that produced it.
BACKENDS: dict[str, str] = {
    "agent": "agents.base",
    "intent": "agents.intent",
    "maneuver": "agents.maneuver",
    "platforms": "agents.platforms",
    "integrator": "core.kinematics",
    "wind": "environment.wind",
    "world": "environment.world",
    "encounter": "environment.encounter",
    "sensor": "sensors.model",
    "tracker": "sensors.tracker",
}


# --------------------------------------------------------------------------
# Intent resolution
# --------------------------------------------------------------------------


def resolve_intent(
    platform: PlatformType | int,
    intent: IntentType | int,
    envelope: PerformanceEnvelope,
    weights: Optional[Mapping[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> IntentType:
    """Coerce a sampled intent onto one the platform can plausibly fly.

    The compatibility tables themselves live in :mod:`agents.intent` and are
    not duplicated here -- two sources of truth for the same rule is how a
    dataset silently ends up containing a steering debris object. Two layers
    are applied, in order:

    1. **Mission plausibility** --
       :data:`agents.intent.PLATFORM_INTENT_COMPATIBILITY`. When the drawn
       intent is not one the platform would ever be flying, it is REDRAWN from
       the same intent mix restricted to the compatible set and renormalised,
       rather than mapped to a fixed fallback: a fixed fallback would pile all
       rejected draws onto one intent and distort the mix.
    2. **Physical capability** -- :func:`agents.intent.resolve_intent_type`,
       plus an explicit platform-keyed guarantee that ``DEBRIS`` is always
       ``PASSIVE_DRIFT`` whatever the mix says. That already follows from
       ``envelope.is_powered == False``, but stating it against the platform
       class as well means a future envelope edit cannot quietly give debris
       an intent.

    Args:
        platform: platform class drawn from the platform mix.
        intent: intent drawn from the intent mix.
        envelope: the platform's sampled performance envelope.
        weights: the intent mix, for the restricted redraw. Without it (and
            without ``rng``) an incompatible intent falls back to the
            lowest-numbered compatible one, which is only used by callers that
            have no mix -- dataset generation always passes both.
        rng: the ``'intent'`` stream generator, for the restricted redraw.

    Returns:
        The effective :class:`~core.types.IntentType`.
    """
    if int(platform) == int(PlatformType.DEBRIS):
        return IntentType.PASSIVE_DRIFT

    allowed = compatible_intents(platform)
    it = IntentType(int(intent))
    if allowed and it not in allowed:
        pool = {int(k): float(v) for k, v in (weights or {}).items()
                if IntentType(int(k)) in allowed and float(v) > 0.0}
        if pool and rng is not None:
            it = IntentType(_sample_from_weights(pool, rng))
        else:
            it = min(allowed, key=int)
    return resolve_intent_type(it, envelope)


# --------------------------------------------------------------------------
# Kinematics helper
# --------------------------------------------------------------------------


def ground_velocity_of(
    state: AgentState,
    envelope: PerformanceEnvelope,
    wind_xy: Sequence[float],
) -> tuple[float, float]:
    """Ground velocity ``(vx, vy)`` [m/s] of a body, counting the wind once.

    For a POWERED body ``state.v`` is an airspeed, so the wind is added on top
    of it scaled by ``wind_susceptibility``. For an UNPOWERED body ``state.v``
    is already a GROUND speed (see :class:`core.state.AgentState`) with the
    drift inside it, so adding wind again would double-count. This function is
    the one place that distinction is made; everything else calls it.

    Args:
        state: kinematic state.
        envelope: the body's performance envelope.
        wind_xy: ambient wind ``(wx, wy)`` [m/s].

    Returns:
        ``(vx, vy)`` ground velocity [m/s], +X East, +Y North.
    """
    s = envelope.wind_susceptibility if envelope.is_powered else 0.0
    return state.ground_velocity(wind_xy, s)


# --------------------------------------------------------------------------
# Ownship
# --------------------------------------------------------------------------


@dataclass
class Ownship:
    """The data-generation ownship.

    This is NOT the RL policy. For dataset generation the ownship flies a
    simple corridor profile so the designed encounter geometries of doc 3.4
    stay valid; :class:`simulation.engine.SimulationEngine` takes a pluggable
    ``ownship_controller`` callable so the PPO policy of doc 7 can be dropped
    in later without touching this class or the storage schema.

    Attributes:
        state: kinematic state [m], [m/s], [rad].
        envelope: performance envelope.
        psi_ref: nominal corridor heading [rad], CCW from +X.
        v_ref: nominal cruise airspeed [m/s].
        agent_id: always ``-1``; the ownship never occupies a traffic slot and
            is stored in the ``own_*`` arrays instead.
        platform: airframe class, for display only.
    """

    state: AgentState
    envelope: PerformanceEnvelope
    psi_ref: float
    v_ref: float
    agent_id: int = -1
    platform: PlatformType = PlatformType.EVTOL_UAM

    def ground_velocity(self, wind_xy: Sequence[float]) -> tuple[float, float]:
        """Ground velocity ``(vx, vy)`` [m/s] under the given wind."""
        return ground_velocity_of(self.state, self.envelope, wind_xy)

    def step(
        self,
        omega_cmd: float,
        a_cmd: float,
        wind_xy: Sequence[float],
        dt: float,
    ) -> None:
        """Advance the ownship by ``dt`` [s] under a control command.

        Uses the same envelope-clipping, rate-limited integrator as the
        traffic (:func:`core.kinematics.step_powered`), so an ownship
        trajectory is exactly as flyable as an intruder's.

        Args:
            omega_cmd: commanded turn rate [rad/s], positive = left / CCW.
            a_cmd: commanded longitudinal acceleration [m/s^2].
            wind_xy: ambient wind ``(wx, wy)`` [m/s].
            dt: timestep [s].
        """
        self.state = step_powered(
            self.state, float(omega_cmd), float(a_cmd), self.envelope, wind_xy, dt
        )

    def snapshot(self) -> AgentSnapshot:
        """Capture the ownship state.

        The engine adds its own auxiliary bookkeeping to
        ``intent_state`` afterwards (see
        :meth:`simulation.engine.SimulationEngine.snapshot`); this method only
        owns the ownship's own fields.
        """
        return AgentSnapshot(
            state=self.state.copy(),
            mode=ManeuverMode.CV,
            mode_timer=0.0,
            mode_params={},
            intent_state={"psi_ref": float(self.psi_ref), "v_ref": float(self.v_ref)},
            rng_state={},
        )

    def restore(self, snap: AgentSnapshot) -> None:
        """Restore from a :meth:`snapshot` result."""
        self.state = snap.state.copy()
        self.psi_ref = float(snap.intent_state.get("psi_ref", self.psi_ref))
        self.v_ref = float(snap.intent_state.get("v_ref", self.v_ref))


# --------------------------------------------------------------------------
# Scenario container
# --------------------------------------------------------------------------


@dataclass
class DesignedEncounter:
    """Bookkeeping for one inverse-designed encounter (doc 3.4).

    Attributes:
        slot: index [-] of the hosting agent in :attr:`Scenario.agents`, which
            is also its column in the episode arrays.
        agent_id: ground-truth id [-] of the hosting agent.
        geometry: :class:`~core.types.GeometryType` integer.
        cpa_dist_target: designed CPA distance d* [m].
        tcpa_target: designed time to CPA t* [s] from episode start.
        meta: the full metadata dict returned by
            :func:`environment.encounter.build_encounter`, kept verbatim so the
            design can be audited after the fact.
    """

    slot: int
    agent_id: int
    geometry: int
    cpa_dist_target: float
    tcpa_target: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """Everything :class:`simulation.engine.SimulationEngine` needs to run.

    Attributes:
        cfg: the configuration this scenario was built from (not mutated).
        world: static context -- arena bounds, corridors, no-fly zones.
        wind: the wind field.
        ownship: the data-generation ownship.
        agents: traffic agents; the list index IS the episode's agent slot.
        agent_ids: stable ids [-], index-aligned with ``agents``.
        platforms: platform-type integers per slot.
        intents: effective intent integers per slot.
        designed: the designed encounters, in construction order.
        sensor: the onboard surveillance model.
        tracker: the track manager.
        wind_speed: realised steady wind speed [m/s].
        wind_dir: realised steady wind direction [rad], CCW from +X.
        ownship_track: ``(x0, y0, vgx, vgy)`` [m], [m/s] of the ownship's
            nominal straight GROUND track; used for traffic seeding and by the
            default ownship controller.
    """

    cfg: ScenarioConfig
    world: World
    wind: WindField
    ownship: Ownship
    agents: list[Agent]
    agent_ids: list[int]
    platforms: list[int]
    intents: list[int]
    designed: list[DesignedEncounter]
    sensor: SensorModel
    tracker: TrackManager
    wind_speed: float
    wind_dir: float
    ownship_track: tuple[float, float, float, float]

    @property
    def n_agents(self) -> int:
        """Number of traffic agents [-]."""
        return len(self.agents)

    @property
    def realised_density_per_km2(self) -> float:
        """Traffic density [aircraft/km^2] actually realised inside the arena.

        This is the number to quote when a statement is about physical
        spacing. ``cfg.density_per_km2`` is the doc 8.1 split axis and is
        larger by ``ARENA_AREA_KM2 / cfg.area_km2``; see the module docstring.
        """
        return float(len(self.agents)) / ARENA_AREA_KM2

    def bind_streams(self, rng_hub: RNGHub) -> None:
        """(Re-)bind every mode machine AND intent to the hub's live streams.

        MUST be called after :meth:`core.rng.RNGHub.reseed_branch` and after
        :meth:`core.rng.RNGHub.restore`, because both replace the generator
        objects the machines hold references to. See the module docstring for
        why a stale binding silently destroys the branch-rollout ground truth.

        Intents are bound for the same reason: ``REACTIVE_AVOID`` takes a
        genuine left/right fork (which way to break away), and that draw has
        to come from ``branch_choice`` or re-seeding ``branch_choice`` alone
        would not re-randomise it.

        Args:
            rng_hub: the hub whose ``mode_decision`` / ``branch_choice``
                streams the machines should draw from.
        """
        rng_mode = rng_hub.stream("mode_decision")
        rng_branch = rng_hub.stream("branch_choice")
        for agent in self.agents:
            machine = agent.maneuver
            if machine is not None and hasattr(machine, "bind_streams"):
                machine.bind_streams(rng_mode, rng_branch)
            intent = agent.intent
            if intent is not None and hasattr(intent, "bind_streams"):
                intent.bind_streams(rng_mode, rng_branch)

    def platform_counts(self) -> dict[int, int]:
        """``{platform_type_int: count}`` over the traffic population."""
        out: dict[int, int] = {}
        for p in self.platforms:
            out[int(p)] = out.get(int(p), 0) + 1
        return out

    def intent_counts(self) -> dict[int, int]:
        """``{intent_type_int: count}`` over the traffic population."""
        out: dict[int, int] = {}
        for i in self.intents:
            out[int(i)] = out.get(int(i), 0) + 1
        return out

    def statics(self) -> dict[str, Any]:
        """JSON-safe static context for the GUI (bounds, corridors, no-fly zones)."""
        return self.world.to_render_dict()


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------


def _sample_from_weights(
    weights: dict[int, float],
    rng: np.random.Generator,
) -> int:
    """Draw one key from an unnormalised, int-keyed weight table.

    Exactly one uniform variate is consumed regardless of the number of
    categories, so adding a zero-weight category to a mix cannot shift the
    draws of anything downstream in the same stream.

    Args:
        weights: ``{key: weight}`` with non-negative weights.
        rng: generator.

    Returns:
        The chosen key.

    Raises:
        ValueError: if the table is empty or every weight is zero.
    """
    keys = sorted(int(k) for k in weights)
    vals = [max(0.0, float(weights[k])) for k in keys]
    total = sum(vals)
    if not keys or total <= 0.0:
        raise ValueError("weight table is empty or sums to zero")
    u = float(rng.random()) * total
    acc = 0.0
    for k, w in zip(keys, vals):
        acc += w
        if u < acc:
            return k
    return keys[-1]


def _nominal_speed(env: PerformanceEnvelope, rng: np.random.Generator) -> float:
    """Draw a sensible cruise airspeed [m/s] inside an envelope.

    The band is the middle of ``[v_min, v_max]``, so a spawned agent keeps
    authority to both accelerate and decelerate and a multirotor with
    ``v_min = 0`` does not spawn motionless.

    Args:
        env: performance envelope.
        rng: generator.

    Returns:
        An airspeed [m/s] inside ``[v_min, v_max]``.
    """
    lo = env.v_min + 0.25 * (env.v_max - env.v_min)
    hi = env.v_min + 0.85 * (env.v_max - env.v_min)
    return float(env.clamp_speed(float(rng.uniform(lo, hi))))


def _slide_into(start: float, delta: float, lo: float, hi: float) -> float:
    """Shift a 1-D segment ``[start, start+delta]`` so it lies inside ``[lo, hi]``.

    The segment is translated, never shortened: the ownship must cover its
    full ``v * duration`` leg or the designed encounters lose their geometry.
    A segment longer than the interval is centred instead, which is the best
    available and is only reachable with a mis-sized arena.

    Args:
        start: segment start [m].
        delta: signed segment length [m].
        lo: lower bound [m].
        hi: upper bound [m].

    Returns:
        The adjusted start coordinate [m].
    """
    a = min(start, start + delta)
    b = max(start, start + delta)
    if (b - a) > (hi - lo):
        return start + (0.5 * (lo + hi) - 0.5 * (a + b))
    if a < lo:
        return start + (lo - a)
    if b > hi:
        return start - (b - hi)
    return start


def _ownship_track(
    cfg: ScenarioConfig,
    world: World,
    envelope: PerformanceEnvelope,
    wind_xy: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[AgentState, float, tuple[float, float, float, float]]:
    """Lay out the ownship's nominal straight cruise leg.

    The leg follows corridor 0 when the world has one, and is positioned so
    that the whole ``duration_s`` fits inside the arena. That is what keeps
    the doc 3.4 inverse design exact: the ownship really does hold a constant
    ground velocity for the entire episode, so a CPA solved analytically at
    t = 0 is the CPA that actually happens.

    Args:
        cfg: scenario configuration.
        world: the arena.
        envelope: ownship envelope; its ``wind_susceptibility`` sets the drift.
        wind_xy: steady wind ``(wx, wy)`` [m/s].
        rng: ``'placement'`` stream generator.

    Returns:
        ``(state, psi_ref, (x0, y0, vgx, vgy))`` -- initial ownship state,
        nominal heading [rad], and the ground-track parameters [m], [m/s].
    """
    if world.corridors:
        corr = world.corridors[0]
        psi = corr.heading()
        cx = 0.5 * (corr.p0[0] + corr.p1[0])
        cy = 0.5 * (corr.p0[1] + corr.p1[1])
    else:
        psi = float(wrap_angle(rng.uniform(-math.pi, math.pi)))
        cx = 0.0
        cy = 0.0

    v = float(envelope.clamp_speed(cfg.ownship_speed_ms))
    s = float(envelope.wind_susceptibility)
    vgx = v * math.cos(psi) + s * wind_xy[0]
    vgy = v * math.sin(psi) + s * wind_xy[1]
    dx = vgx * float(cfg.duration_s)
    dy = vgy * float(cfg.duration_s)

    xmin, ymin, xmax, ymax = world.bounds
    pad = 50.0
    x0 = _slide_into(cx - 0.5 * dx, dx, xmin + pad, xmax - pad)
    y0 = _slide_into(cy - 0.5 * dy, dy, ymin + pad, ymax - pad)

    state = AgentState(x=float(x0), y=float(y0), v=v, psi=float(psi))
    return state, float(psi), (float(x0), float(y0), float(vgx), float(vgy))


def _seed_background_position(
    world: World,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Draw a spawn position [m] for one piece of background traffic.

    Uniform over the free area of the arena (outside the no-fly zones), which
    is deliberate rather than convenient. Doc 3.4 opens by observing that
    "무작위 배치하면 위험 조우가 거의 안 생긴다" -- random placement almost
    never produces a risky encounter -- and that this is precisely why the
    dangerous geometries are INVERSE-DESIGNED instead. Background traffic must
    therefore be genuinely uninformative about risk: if it were seeded along
    the ownship's track it would manufacture near-misses on its own, every
    episode would land in :class:`~core.types.RiskLevel` HIGH, and the
    designed CPA distance would stop being the controlled variable that the
    stratified analysis of doc 3.4 depends on.

    Args:
        world: the arena; supplies bounds and no-fly-zone clearance.
        rng: ``'placement'`` stream generator.

    Returns:
        ``(x, y)`` spawn position [m].
    """
    return world.sample_free_position(rng, nfz_clearance=50.0)


class _NoneSafeIntent:
    """Adapter that normalises an "I have no command" proposal to ``None``.

    KNOWN CROSS-MODULE MISMATCH (report it upstream, do not paper over it
    silently): :meth:`agents.intent.IntentController.command` documents and
    returns ``(None, None)`` for an unpowered body, whereas
    :meth:`agents.base.Agent.step` tests ``proposal is None`` and otherwise
    does ``float(proposal[0])`` -- which raises ``TypeError`` on the tuple. A
    ``DEBRIS`` agent therefore cannot be stepped without this adapter.

    Rather than editing either sibling module (both are owned elsewhere), the
    two conventions are reconciled at the one place that builds the agent.
    Every other attribute -- ``intent_type``, ``requested_type``, ``snapshot``,
    ``restore``, ``describe`` -- is forwarded unchanged, so
    :meth:`agents.base.Agent.snapshot` still captures the real controller's
    state and branch rollouts stay exact.

    Attributes:
        _inner: the wrapped :class:`agents.intent.IntentController`.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        """Wrap ``inner``.

        Args:
            inner: the intent controller to delegate to.
        """
        self._inner = inner

    def command(
        self,
        agent: Any,
        t: float,
        ownship_state: Optional[AgentState],
        rng: Optional[np.random.Generator],
    ) -> Optional[tuple[float, float]]:
        """Return ``(omega_cmd [rad/s], a_cmd [m/s^2])``, or ``None``.

        Args:
            agent: the driven agent.
            t: simulation time [s].
            ownship_state: ownship state, or ``None``.
            rng: decision-stream generator.

        Returns:
            The controller's command pair, or ``None`` when either component
            is ``None`` (an uncontrolled, unpowered body).
        """
        proposal = self._inner.command(agent, t, ownship_state, rng)
        if proposal is None:
            return None
        omega, accel = proposal[0], proposal[1]
        if omega is None or accel is None:
            return None
        return float(omega), float(accel)

    def __getattr__(self, name: str) -> Any:
        """Forward every other attribute to the wrapped controller."""
        return getattr(self._inner, name)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_NoneSafeIntent({self._inner!r})"


def _make_agent(
    slot: int,
    platform: PlatformType,
    envelope: PerformanceEnvelope,
    state: AgentState,
    requested_intent: IntentType,
    world: World,
    rng_intent: np.random.Generator,
    rng_mode: np.random.Generator,
    rng_branch: np.random.Generator,
    reaction_enabled: bool = True,
) -> Agent:
    """Build one fully wired traffic agent.

    Assembly order matters: the intent controller needs the agent (it reads
    ``.state`` / ``.envelope`` at construction), and the mode machine needs the
    intent's EFFECTIVE type, because the transition matrix and the dwell means
    of :mod:`agents.maneuver` are keyed by it. Building the machine from the
    *requested* intent would give a substituted agent the wrong maneuver
    rhythm while its ``gt_intent`` label said otherwise.

    Args:
        slot: agent slot index [-]; also the agent id.
        platform: airframe class.
        envelope: sampled performance envelope.
        state: initial kinematic state.
        requested_intent: intent drawn from the mix, before resolution.
        world: static context handed to the intent controller.
        rng_intent: ``'intent'`` stream (construction-time randomness).
        rng_mode: ``'mode_decision'`` stream, bound into the machine.
        rng_branch: ``'branch_choice'`` stream, bound into the machine.

    Returns:
        A ready :class:`agents.base.Agent`. Its ``rng`` is deliberately
        ``None``: every decision draw comes from a hub stream passed in at
        step time, so :meth:`core.rng.RNGHub.reseed_branch` reaches all of
        them. A private per-agent generator would be invisible to the reseed
        and would make branch rollouts degenerate.
    """
    agent = Agent(
        agent_id=int(slot),
        platform=PlatformType(int(platform)),
        envelope=envelope,
        state=state,
        intent=None,
        maneuver=None,
        alive=True,
        spawn_t=0.0,
        rng=None,
        is_ood=bool(is_ood_platform(PlatformType(int(platform)))),
    )
    controller = make_intent(requested_intent, agent, world, rng_intent)
    controller.bind_streams(rng_mode, rng_branch)
    agent.intent = _NoneSafeIntent(controller)
    # The stability profile rides on the envelope (agents.platforms). Passing
    # it explicitly means the diagonal-biased transition matrix is built once,
    # here, rather than being adopted on the machine's first step.
    machine = ModeMachine(
        IntentType(int(controller.intent_type)), stability=envelope.stability
    )
    machine.bind_streams(rng_mode, rng_branch)
    agent.maneuver = machine

    # rev.3 reaction layer (doc 3.2(4)): every powered platform reacts to
    # threatening traffic. REACTIVE_AVOID intents already evade the ownship,
    # so their layer excludes it -- the two mechanisms never double-command.
    if reaction_enabled and envelope.is_powered:
        profile = reaction_profile(agent.platform)
        if profile is not None:
            agent.reaction = ReactionLayer(
                profile,
                exclude_ownship=(
                    IntentType(int(controller.intent_type))
                    == IntentType.REACTIVE_AVOID
                ),
            )
    return agent


# --------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------


def build_scenario(
    cfg: ScenarioConfig,
    rng_hub: Optional[RNGHub] = None,
) -> Scenario:
    """Assemble a complete, runnable scenario.

    Construction order. Each stage consumes only its own named stream, so
    changing one stage cannot perturb another and an episode stays replayable
    from ``(config, seed)`` alone (doc 10):

    1. ``'wind'``          -- steady wind speed and direction.
    2. ``'placement'``     -- arena, corridors, no-fly zones, ownship leg,
       background spawn positions.
    3. ``'platform'``      -- platform draws and their performance envelopes.
    4. ``'scenario'``      -- how many designed encounters, their geometries
       and the inverse design itself.
    5. ``'intent'``        -- intent draws and intent-controller construction.
    6. ``'mode_decision'`` / ``'branch_choice'`` -- bound into every mode
       machine, not drawn from here.

    Args:
        cfg: the scenario configuration. It is NOT mutated.
        rng_hub: the hub to draw from; a fresh ``RNGHub(cfg.seed)`` is created
            when omitted.

    Returns:
        A :class:`Scenario` with its decision streams already bound.

    Raises:
        ValueError: if ``cfg`` fails validation, or if a mix cannot produce a
            single agent.
    """
    problems = cfg.validate()
    if problems:
        raise ValueError(
            f"invalid ScenarioConfig {cfg.name!r}: " + "; ".join(problems)
        )
    hub = RNGHub(int(cfg.seed)) if rng_hub is None else rng_hub

    rng_wind = hub.stream("wind")
    rng_place = hub.stream("placement")
    rng_plat = hub.stream("platform")
    rng_scen = hub.stream("scenario")
    rng_intent = hub.stream("intent")
    rng_mode = hub.stream("mode_decision")
    rng_branch = hub.stream("branch_choice")

    # 1. wind ---------------------------------------------------------------
    wind = WindField(cfg.wind, rng_wind)
    wind_xy = wind.steady

    # 2. arena and static context ------------------------------------------
    arena_cfg = cfg.copy_with(area_km2=ARENA_AREA_KM2)
    world = World(arena_cfg, rng_place)

    # 3. ownship ------------------------------------------------------------
    own_env = make_envelope(PlatformType.EVTOL_UAM, rng_plat)
    own_state, psi_ref, track = _ownship_track(cfg, world, own_env, wind_xy, rng_place)
    ownship = Ownship(
        state=own_state,
        envelope=own_env,
        psi_ref=psi_ref,
        v_ref=float(own_state.v),
    )

    # The inverse design of doc 3.4 assumes the ownship travels in a straight
    # line at constant velocity. Under a steady wind the ownship's GROUND
    # track is still exactly straight, so the design is solved against a
    # pseudo-state whose (v, psi) ARE the ownship ground velocity. Designing
    # against the airspeed instead would put every designed CPA off by the
    # ownship's wind drift -- up to 8 m/s x 45 s = 360 m, which is larger than
    # the entire CPA sampling range.
    vgx, vgy = track[2], track[3]
    design_state = AgentState(
        x=own_state.x,
        y=own_state.y,
        v=math.hypot(vgx, vgy),
        psi=wrap_angle(math.atan2(vgy, vgx)),
    )

    # 4. population ---------------------------------------------------------
    n_traffic = int(cfg.n_background())
    n_lo = int(cfg.encounter.n_designed_min)
    n_hi = int(cfg.encounter.n_designed_max)
    n_designed = int(rng_scen.integers(n_lo, n_hi + 1)) if n_hi > n_lo else n_lo
    n_designed = max(0, min(n_designed, n_traffic))

    platform_weights = dict(cfg.platform_mix.weights)
    intent_weights = dict(cfg.intent_mix.weights)
    max_design_range = float(cfg.sensor.r_max_m) * _DESIGN_RANGE_SENSOR_FACTOR

    agents: list[Agent] = []
    agent_ids: list[int] = []
    platforms: list[int] = []
    intents: list[int] = []
    designed: list[DesignedEncounter] = []

    for slot in range(n_traffic):
        ptype = PlatformType(_sample_from_weights(platform_weights, rng_plat))
        env = make_envelope(ptype, rng_plat)

        if slot < n_designed:
            geometry = sample_geometry(cfg.encounter, rng_scen)
            state, meta = build_encounter(
                design_state,
                geometry,
                cfg.encounter,
                env,
                rng_scen,
                min_range_m=_MIN_DESIGN_RANGE_M,
                max_range_m=max_design_range,
            )
            designed.append(DesignedEncounter(
                slot=slot,
                agent_id=slot,
                geometry=int(meta["geometry"]),
                cpa_dist_target=float(meta["cpa_dist"]),
                tcpa_target=float(meta["tcpa"]),
                meta=dict(meta),
            ))
        else:
            x, y = _seed_background_position(world, rng_place)
            state = AgentState(
                x=float(x),
                y=float(y),
                v=_nominal_speed(env, rng_place),
                psi=float(wrap_angle(rng_place.uniform(-math.pi, math.pi))),
            )

        requested = resolve_intent(
            ptype,
            IntentType(_sample_from_weights(intent_weights, rng_intent)),
            env,
            weights=intent_weights,
            rng=rng_intent,
        )
        agent = _make_agent(
            slot, ptype, env, state, requested, world,
            rng_intent, rng_mode, rng_branch,
            reaction_enabled=bool(getattr(cfg, "reaction_enabled", True)),
        )

        agents.append(agent)
        agent_ids.append(int(agent.agent_id))
        platforms.append(int(ptype))
        intents.append(int(agent.intent_code))

    # 5. observation layer --------------------------------------------------
    sensor = SensorModel(cfg.sensor, phase_seed=int(cfg.seed))
    tracker = TrackManager.from_config(cfg.sensor)

    scenario = Scenario(
        cfg=cfg,
        world=world,
        wind=wind,
        ownship=ownship,
        agents=agents,
        agent_ids=agent_ids,
        platforms=platforms,
        intents=intents,
        designed=designed,
        sensor=sensor,
        tracker=tracker,
        wind_speed=float(math.hypot(wind_xy[0], wind_xy[1])),
        wind_dir=float(wrap_angle(math.atan2(wind_xy[1], wind_xy[0]))),
        ownship_track=track,
    )
    scenario.bind_streams(hub)
    return scenario
