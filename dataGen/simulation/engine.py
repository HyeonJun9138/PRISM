"""The fixed-step simulation engine: one tick, one episode, one snapshot.

Timing contract (doc 3.7 "integrate fine, log coarse"):

* the state is integrated at :data:`core.constants.SIM_DT` = 0.05 s (20 Hz);
* a frame is logged every :data:`core.constants.LOG_DECIMATION` = 10 ticks,
  i.e. at :data:`core.constants.LOG_DT` = 0.5 s (2 Hz);
* a 60 s episode is 1200 integration ticks and exactly 120 logged frames, at
  ``t = 0.5, 1.0, ... 60.0`` s.

The first logged frame is at ``t = LOG_DT``, not at ``t = 0``. That follows
directly from the required tick order (log AFTER advancing), and it means the
logged series covers the whole 60 s of simulated time rather than stopping
half a step short of it.

Snapshot / restore is not a convenience here -- it is the mechanism the
calibration ground truth of doc 3.3 is built on (:mod:`simulation.rollout`).
Everything that can influence the future is captured: agent states, mode
machines, intent bookkeeping, wind, the sensor schedule and history, the
tracker, the per-slot logged track age, the pending measurement batch, and
every RNG stream. Anything left out would make a restored run diverge from
the original and would silently corrupt the "true" future distribution.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

from core.constants import LOG_DECIMATION, SIM_DT
from core.rng import RNGHub
from core.state import (
    AgentState,
    PerformanceEnvelope,
    WorldSnapshot,
    angle_diff,
    wrap_angle,
)
from environment.encounter import compute_cpa
from scenario.builder import Scenario, build_scenario
from scenario.config import ScenarioConfig
from storage.schema import EpisodeArrays, EpisodeMeta

__all__ = [
    "ENGINE_AUX_KEY",
    "OwnshipController",
    "default_ownship_controller",
    "SimulationEngine",
]


#: Key under which the engine stores its auxiliary state inside the ownship
#: snapshot's ``intent_state``. :class:`core.state.WorldSnapshot` has a fixed
#: field set owned by ``core.state``; rather than widening that shared
#: contract, the engine nests its bookkeeping under one reserved key in the
#: one free-form dict it owns.
ENGINE_AUX_KEY: str = "__engine__"

#: Heading-hold proportional gain [1/s] of the default ownship controller.
_OWN_HEADING_GAIN: float = 0.6

#: Speed-hold proportional gain [1/s] of the default ownship controller.
_OWN_SPEED_GAIN: float = 0.30

#: Signature of a pluggable ownship controller.
#:
#: ``(t [s], state, envelope, context) -> (omega_cmd [rad/s], a_cmd [m/s^2])``
#:
#: ``context`` carries at least ``psi_ref`` [rad], ``v_ref`` [m/s],
#: ``wind_xy`` [m/s], ``world`` and ``track``. The PPO policy of doc 7 drops in
#: here: it is the only ownship-side coupling point, so swapping it cannot
#: change the traffic model, the sensor model or the storage schema.
OwnshipController = Callable[
    [float, AgentState, PerformanceEnvelope, dict], "tuple[float, float]"
]


def default_ownship_controller(
    t: float,
    state: AgentState,
    envelope: PerformanceEnvelope,
    context: dict,
) -> tuple[float, float]:
    """Fly the nominal cruise corridor: hold heading, hold speed.

    Deliberately trivial. The ownship must hold a constant GROUND velocity for
    the whole episode or the inverse-designed encounters of doc 3.4 stop
    landing where they were designed to; a cross-track controller would fight
    the steady wind drift that the design already accounts for, so only
    heading and speed are regulated. The world's containment heading is
    honoured as a safety net, but with the track laid out by
    :func:`scenario.builder._ownship_track` it never fires.

    Args:
        t: current simulation time [s] (unused; the profile is time-invariant).
        state: ownship kinematic state.
        envelope: ownship performance envelope (the caller clips again).
        context: see :data:`OwnshipController`.

    Returns:
        ``(omega_cmd [rad/s], a_cmd [m/s^2])``.
    """
    del t
    psi_ref = float(context.get("psi_ref", state.psi))
    world = context.get("world")
    if world is not None:
        psi_bound = world.containment_heading((state.x, state.y), state.psi)
        if psi_bound is not None:
            psi_ref = float(psi_bound)
    omega = _OWN_HEADING_GAIN * angle_diff(psi_ref, state.psi)
    accel = _OWN_SPEED_GAIN * (float(context.get("v_ref", state.v)) - state.v)
    return envelope.clamp_turn_rate(omega, state.v), envelope.clamp_accel(accel)


class SimulationEngine:
    """Drives one episode of one scenario.

    Attributes:
        cfg: the scenario configuration.
        rng_hub: the RNG hub every stochastic decision draws from.
        scenario: the assembled world, agents and observation chain.
        ep: the logged episode buffer, filled in place as the run proceeds.
    """

    def __init__(
        self,
        cfg: ScenarioConfig,
        rng_hub: Optional[RNGHub] = None,
        ownship_controller: Optional[OwnshipController] = None,
        episode_index: int = 0,
    ) -> None:
        """Build the scenario and allocate the episode buffer.

        Args:
            cfg: scenario configuration; NOT mutated.
            rng_hub: hub to draw from; a fresh ``RNGHub(cfg.seed)`` when
                omitted.
            ownship_controller: pluggable ownship control law; defaults to
                :func:`default_ownship_controller`.
            episode_index: index [-] recorded in the episode metadata.

        Raises:
            ValueError: if ``cfg`` fails validation (raised by the builder).
        """
        self.cfg: ScenarioConfig = cfg
        self.episode_index: int = int(episode_index)
        self.rng_hub: RNGHub = RNGHub(int(cfg.seed)) if rng_hub is None else rng_hub
        self._controller: OwnshipController = (
            default_ownship_controller if ownship_controller is None
            else ownship_controller
        )
        self.scenario: Scenario = build_scenario(cfg, self.rng_hub)

        self.n_steps: int = int(cfg.n_logged_steps())
        self.total_ticks: int = self.n_steps * LOG_DECIMATION
        self.n_agents: int = max(1, self.scenario.n_agents)

        self._tick: int = 0
        self._log_index: int = 0
        self._pending: list[Any] = []
        self._track_id = np.full((self.n_agents,), -1, dtype=np.int64)
        self._track_age = np.zeros((self.n_agents,), dtype=np.int64)
        self.ep: EpisodeArrays = self._allocate()
        self._min_sep_m: float = float("inf")
        self.reaction_events: int = 0
        self._observe_separation()

    # -- construction ------------------------------------------------------

    def _allocate(self) -> EpisodeArrays:
        """Allocate the episode buffer and fill its time-invariant columns."""
        ep = EpisodeArrays.allocate(self.n_steps, self.n_agents)
        for slot, agent in enumerate(self.scenario.agents):
            ep.gt_type[slot] = np.int8(int(agent.platform))
            ep.agent_ids[slot] = np.int32(int(agent.agent_id))
        return ep

    def reset(self) -> None:
        """Rebuild the scenario from scratch and rewind to ``t = 0``.

        The hub is recreated from ``cfg.seed``, so a reset engine reproduces
        the original episode exactly.
        """
        self.rng_hub = RNGHub(int(self.cfg.seed))
        self.scenario = build_scenario(self.cfg, self.rng_hub)
        self.n_agents = max(1, self.scenario.n_agents)
        self._tick = 0
        self._log_index = 0
        self._pending = []
        self._track_id = np.full((self.n_agents,), -1, dtype=np.int64)
        self._track_age = np.zeros((self.n_agents,), dtype=np.int64)
        self.ep = self._allocate()
        self._min_sep_m = float("inf")
        #: reaction-layer latches armed while logging (rev.3 event counter);
        #: branch rollouts (log=False) perceive but do not count.
        self.reaction_events = 0
        self._observe_separation()

    # -- read-only views ---------------------------------------------------

    @property
    def t(self) -> float:
        """Current simulation time [s].

        Derived from the integer tick count rather than accumulated, so 1200
        steps of 0.05 s land on exactly 60.0 s with no floating-point drift.
        """
        return self._tick * SIM_DT

    @property
    def tick(self) -> int:
        """Number of integration steps taken [-]."""
        return self._tick

    @property
    def n_logged(self) -> int:
        """Number of frames logged so far [-]."""
        return self._log_index

    @property
    def is_done(self) -> bool:
        """True once the configured episode duration has been simulated."""
        return self._tick >= self.total_ticks

    @property
    def min_separation_m(self) -> float:
        """Smallest ownship-to-agent horizontal distance [m] seen SO FAR.

        Accumulated at :data:`core.constants.SIM_DT` (20 Hz), not at the 2 Hz
        logging rate. The logged arrays undersample the closest point of
        approach by up to half a logging interval of closure -- measured at up
        to 19 m, i.e. 12% of :data:`core.constants.NMAC_RADIUS_M` -- and the
        bias is one-sided (the logged minimum is never below the true one), so
        an episode whose true CPA lands just inside the NMAC radius would be
        labelled safe. Safety ground truth must not be a sampling artefact.
        """
        return float(self._min_sep_m)

    def _observe_separation(self) -> None:
        """Fold the current instant into the running minimum separation."""
        sc = self.scenario
        ox = float(sc.ownship.state.x)
        oy = float(sc.ownship.state.y)
        m = self._min_sep_m
        for agent in sc.agents:
            if not agent.alive:
                continue
            d = math.hypot(float(agent.state.x) - ox, float(agent.state.y) - oy)
            if d < m:
                m = d
        self._min_sep_m = m

    def gt_positions(self) -> np.ndarray:
        """True positions ``[N,2]`` [m] right now; NaN for a retired slot."""
        out = np.full((self.n_agents, 2), np.nan, dtype=np.float64)
        for slot, agent in enumerate(self.scenario.agents):
            if agent.alive:
                out[slot, 0] = agent.state.x
                out[slot, 1] = agent.state.y
        return out

    # -- stepping ----------------------------------------------------------

    def step(self, log: bool = True) -> bool:
        """Advance the simulation by exactly one :data:`core.constants.SIM_DT`.

        Tick order, as specified:

        1. the wind field advances;
        2. every live agent advances, each feeling the wind at ITS OWN
           position and seeing the ownship state from the START of the tick,
           so no agent gains an unfair half-step of information;
        3. the ownship advances under the pluggable controller;
        4. ``t`` advances;
        5. the sensor is sampled (this both pushes the 20 Hz history the
           latency model interpolates and collects any measurement that came
           due), and on every tenth tick a frame is logged.

        Args:
            log: when False the frame is still simulated and the sensor still
                runs, but nothing is written to :attr:`ep`. Branch rollouts
                use this so a rollout cannot overwrite the episode being
                logged.

        Returns:
            True while the episode still has ticks left, False once
            :attr:`is_done`.
        """
        dt = SIM_DT
        t0 = self.t
        sc = self.scenario

        # 1. wind
        sc.wind.step(dt, self.rng_hub.stream("wind"))

        # 2. traffic
        own_state = sc.ownship.state
        rng_mode = self.rng_hub.stream("mode_decision")

        # 2a. reaction-layer perception pass (rev.3, doc 3.2(4)). Runs at the
        # logging cadence (2 Hz), not 20 Hz: that is a plausible avoidance
        # scan rate AND it keeps the O(N^2) geometry sweep off the hot path.
        # Agents are visited in list order and the threat table is built once,
        # so the shared-stream draws stay deterministic for a fixed seed.
        if self._tick % LOG_DECIMATION == 0:
            threats: list[tuple[int, float, float, float, float]] = [
                (-1, own_state.x, own_state.y,
                 own_state.v * math.cos(own_state.psi),
                 own_state.v * math.sin(own_state.psi))
            ]
            for ag in sc.agents:
                if not ag.alive:
                    continue
                threats.append((int(ag.agent_id), ag.state.x, ag.state.y,
                                ag.state.v * math.cos(ag.state.psi),
                                ag.state.v * math.sin(ag.state.psi)))
            rng_percep = self.rng_hub.stream("perception")
            rng_branch = self.rng_hub.stream("branch_choice")
            for ag in sc.agents:
                if ag.alive and ag.reaction is not None:
                    armed = ag.reaction.perceive(ag, threats, rng_percep, rng_branch)
                    if armed and log:
                        self.reaction_events += 1

        for agent in sc.agents:
            if not agent.alive:
                continue
            wind_xy = sc.wind.at(t0, (agent.state.x, agent.state.y))
            agent.step(t0, dt, wind_xy, own_state, rng_mode)

        # 3. ownship
        own_wind = sc.wind.at(t0, (own_state.x, own_state.y))
        omega_cmd, a_cmd = self._controller(
            t0, sc.ownship.state, sc.ownship.envelope, self._controller_context(own_wind)
        )
        sc.ownship.step(omega_cmd, a_cmd, own_wind, dt)

        # 4. clock
        self._tick += 1
        t1 = self.t

        # 5. observation chain + logging
        self._sample_sensor(t1)
        if log:
            # 20 Hz safety accounting. Gated on `log` for the same reason the
            # frame write is: a branch rollout must not fold its alternative
            # future into the episode being measured.
            self._observe_separation()
        if self._tick % LOG_DECIMATION == 0:
            self._flush_frame(t1, log=log)

        return not self.is_done

    def advance(self, n_ticks: int, log: bool = True) -> None:
        """Take ``n_ticks`` integration steps.

        Args:
            n_ticks: number of steps [-].
            log: forwarded to :meth:`step`.
        """
        for _ in range(int(n_ticks)):
            self.step(log=log)

    def advance_to(self, t_target: float, log: bool = True) -> None:
        """Step until the simulation time reaches ``t_target`` [s].

        Args:
            t_target: target time [s]; a target already passed is a no-op.
            log: forwarded to :meth:`step`.
        """
        target_tick = int(round(float(t_target) / SIM_DT))
        while self._tick < target_tick:
            self.step(log=log)

    def _controller_context(self, wind_xy: Sequence[float]) -> dict:
        """Assemble the context dict handed to the ownship controller."""
        sc = self.scenario
        return {
            "psi_ref": sc.ownship.psi_ref,
            "v_ref": sc.ownship.v_ref,
            "wind_xy": (float(wind_xy[0]), float(wind_xy[1])),
            "world": sc.world,
            "track": sc.ownship_track,
            "engine": self,
        }

    # -- observation chain -------------------------------------------------

    def _sample_sensor(self, t: float) -> None:
        """Run the sensor for one tick, accumulating measurements that came due.

        The sensor is polled at the full 20 Hz integration rate on purpose:
        :meth:`sensors.model.SensorModel.measure` pushes its latency history
        buffer on every call, so polling only at the 2 Hz logging rate would
        leave the 0.2 s latency model interpolating across 0.5 s gaps. The
        per-target schedule inside ``measure`` still fires only at
        ``update_rate_hz``, so the number of measurements is unaffected.

        Args:
            t: current simulation time [s].
        """
        sc = self.scenario
        live = [a for a in sc.agents if a.alive]
        if not live:
            return
        self._pending.extend(sc.sensor.measure(
            t,
            sc.ownship.state,
            live,
            self.rng_hub.stream("sensor_noise"),
            self.rng_hub.stream("sensor_detect"),
            self.rng_hub.stream("sensor_latency"),
        ))

    def _flush_frame(self, t: float, log: bool) -> None:
        """Fold the pending measurements into the tracker and log one frame.

        The measurements accumulated since the previous logged frame are
        applied as ONE tracker batch. That is what makes ``obs_detected`` mean
        "a fresh measurement arrived during this logging interval", which is
        the only honest reading when the sensor (5 Hz) runs faster than the
        log (2 Hz).

        Args:
            t: logged time [s].
            log: when False the tracker still advances -- it must, or the
                track lifecycle would depend on whether anyone was watching --
                but nothing is written to :attr:`ep`.
        """
        sc = self.scenario
        records = sc.tracker.update(t, self._pending)
        self._pending.clear()

        # Per-slot logged-step track age. The tracker counts SENSOR updates;
        # the schema wants LOGGED steps, and wants age == 0 to be an exact
        # synonym for "no live track", so the counter is kept here.
        for slot, agent in enumerate(sc.agents):
            rec = records.get(int(agent.agent_id))
            track_id = -1 if rec is None else int(rec.track_id)
            if track_id < 0:
                self._track_id[slot] = -1
                self._track_age[slot] = 0
            elif track_id != int(self._track_id[slot]):
                self._track_id[slot] = track_id
                self._track_age[slot] = 1
            else:
                self._track_age[slot] += 1

        if not log or self._log_index >= self.n_steps:
            return

        i = self._log_index
        ep = self.ep

        own = sc.ownship
        own_wind = sc.wind.at(t, (own.state.x, own.state.y))
        ovx, ovy = own.ground_velocity(own_wind)
        ep.t[i] = np.float32(t)
        ep.own_pos[i, 0] = np.float32(own.state.x)
        ep.own_pos[i, 1] = np.float32(own.state.y)
        ep.own_vel[i, 0] = np.float32(ovx)
        ep.own_vel[i, 1] = np.float32(ovy)
        ep.own_psi[i] = np.float32(wrap_angle(own.state.psi))

        for slot, agent in enumerate(sc.agents):
            if agent.alive:
                st = agent.state
                wind_xy = sc.wind.at(t, (st.x, st.y))
                vx, vy = agent.ground_velocity(wind_xy)
                ep.gt_valid[i, slot] = True
                ep.gt_pos[i, slot, 0] = np.float32(st.x)
                ep.gt_pos[i, slot, 1] = np.float32(st.y)
                ep.gt_vel[i, slot, 0] = np.float32(vx)
                ep.gt_vel[i, slot, 1] = np.float32(vy)
                ep.gt_psi[i, slot] = np.float32(wrap_angle(st.psi))
                ep.gt_mode[i, slot] = np.int8(int(agent.mode_code))
                ep.gt_intent[i, slot] = np.int8(int(agent.intent_code))

            ep.obs_track_id[i, slot] = np.int32(self._track_id[slot])
            ep.obs_track_age[i, slot] = np.int32(self._track_age[slot])

            rec = records.get(int(agent.agent_id))
            if rec is not None and rec.detected and agent.alive:
                ep.obs_detected[i, slot] = True
                ep.obs_pos[i, slot, :] = np.asarray(rec.pos_xy, dtype=np.float32)
                ep.obs_R[i, slot, :, :] = np.asarray(rec.R_2x2, dtype=np.float32)
                # The time the measurement was TRUE, not the time it is logged
                # at. obs_R covers only the polar noise; without t_valid the
                # staleness (latency + age inside the 0.5 s log interval) is
                # unmodelled and unrecoverable, which biases the observation
                # layer's own uncertainty low. See storage.schema.
                ep.obs_t_valid[i, slot] = np.float32(rec.t_valid)

        self._log_index += 1

    # -- episode -----------------------------------------------------------

    def run_episode(self) -> tuple[EpisodeArrays, EpisodeMeta]:
        """Run the whole episode and return its arrays and metadata.

        Rewinds first if the engine has already been stepped, so calling this
        twice on one engine yields identical results.

        Returns:
            ``(arrays, meta)``. The metadata is computed by
            :func:`simulation.episode.compute_episode_meta` -- imported inside
            the function because ``simulation.episode`` constructs engines,
            and a module-level import either way would be circular.
        """
        from simulation.episode import compute_episode_meta

        if self._tick != 0:
            self.reset()
        while not self.is_done:
            self.step(log=True)
        meta = compute_episode_meta(
            self.ep,
            self.scenario,
            episode_index=self.episode_index,
            min_separation_m=self.min_separation_m,
            reaction_events=self.reaction_events,
        )
        return self.ep, meta

    # -- snapshot / restore ------------------------------------------------

    def snapshot(self) -> WorldSnapshot:
        """Capture the complete resumable state of the simulation.

        Returns:
            A :class:`core.state.WorldSnapshot`. The engine's own bookkeeping
            (clock, wind, sensor schedule and history, tracker, logged track
            age, pending measurement batch) lives under
            :data:`ENGINE_AUX_KEY` inside the ownship snapshot's
            ``intent_state``.
        """
        sc = self.scenario
        own_snap = sc.ownship.snapshot()
        own_snap.intent_state[ENGINE_AUX_KEY] = {
            "tick": int(self._tick),
            "log_index": int(self._log_index),
            "wind": copy.deepcopy(sc.wind.snapshot()),
            "sensor": copy.deepcopy(sc.sensor.snapshot()),
            "tracker": copy.deepcopy(sc.tracker.snapshot()),
            "track_id": self._track_id.tolist(),
            "track_age": self._track_age.tolist(),
            "pending": copy.deepcopy(self._pending),
            "min_sep_m": float(self._min_sep_m),
        }
        return WorldSnapshot(
            t=self.t,
            ownship=own_snap,
            agents=[a.snapshot() for a in sc.agents],
            agent_ids=[int(a.agent_id) for a in sc.agents],
            rng_state=self.rng_hub.snapshot(),
        )

    def restore(self, snap: WorldSnapshot) -> None:
        """Resume from a :meth:`snapshot` result.

        The snapshot is NOT consumed: every mutable payload is deep-copied on
        the way in, so the same snapshot can be restored any number of times.
        That is exactly what :func:`simulation.rollout.branch_rollout` does --
        restore, reseed, run, restore, reseed, run.

        Args:
            snap: a snapshot produced by this engine (or by an engine built
                from the same configuration and seed).

        Raises:
            ValueError: if the snapshot describes a different number of agents,
                which would silently mis-align every agent slot.
        """
        sc = self.scenario
        if len(snap.agents) != len(sc.agents):
            raise ValueError(
                f"snapshot has {len(snap.agents)} agents but this engine has "
                f"{len(sc.agents)}: the snapshot came from a different scenario"
            )
        aux = dict(snap.ownship.intent_state.get(ENGINE_AUX_KEY, {}))

        sc.ownship.restore(snap.ownship)
        for agent, agent_snap in zip(sc.agents, snap.agents):
            agent.restore(copy.deepcopy(agent_snap))

        self.rng_hub.restore(snap.rng_state)
        # The mode machines hold REFERENCES to the hub's generator objects and
        # RNGHub.restore may create new ones; re-binding here is what stops a
        # restored run from drawing out of a stale generator.
        sc.bind_streams(self.rng_hub)

        if "wind" in aux:
            sc.wind.restore(copy.deepcopy(aux["wind"]))
        if "sensor" in aux:
            sc.sensor.restore(copy.deepcopy(aux["sensor"]))
        if "tracker" in aux:
            sc.tracker.restore(copy.deepcopy(aux["tracker"]))
        self._tick = int(aux.get("tick", int(round(snap.t / SIM_DT))))
        self._log_index = int(aux.get("log_index", self._log_index))
        if "track_id" in aux:
            self._track_id = np.asarray(aux["track_id"], dtype=np.int64).copy()
        if "track_age" in aux:
            self._track_age = np.asarray(aux["track_age"], dtype=np.int64).copy()
        self._pending = copy.deepcopy(list(aux.get("pending", [])))
        self._min_sep_m = float(aux.get("min_sep_m", self._min_sep_m))

    def reseed_branch(self, branch_seed: int, names: Sequence[str]) -> None:
        """Re-randomise the named streams and re-bind the mode machines.

        Always prefer this over calling
        :meth:`core.rng.RNGHub.reseed_branch` directly: the hub REPLACES the
        generator objects, and :class:`agents.maneuver.ModeMachine` holds
        references to them. Reseeding without re-binding leaves every machine
        drawing from the pre-branch generator, all branches replay identically
        and the empirical future distribution collapses to a point -- the
        exact degeneracy :mod:`simulation.rollout` checks for.

        Args:
            branch_seed: seed [-] identifying this branch.
            names: stream names to re-randomise.
        """
        self.rng_hub.reseed_branch(int(branch_seed), list(names))
        self.scenario.bind_streams(self.rng_hub)

    # -- GUI ---------------------------------------------------------------

    def current_frame(self, include_statics: bool = True) -> dict[str, Any]:
        """A compact, JSON-serialisable snapshot of the current instant.

        Sent to the Canvas 2D GUI at roughly 20 fps, so every float is rounded
        to the precision the display can actually resolve: 0.1 m for
        positions, 1 mrad for angles.

        Args:
            include_statics: include the world's static context (bounds,
                corridors, no-fly zones). They never change during an episode,
                so a streaming client can ask for them once.

        Returns:
            A dict with ``t``, ``wind``, ``own``, ``agents`` and optionally
            ``statics``. Each agent entry carries its ground truth (position,
            heading, speed, platform, mode, intent), its observation status
            (detected, track id, logged track age, measured position, the
            covariance as a flat ``[Rxx, Rxy, Ryx, Ryy]``) and the live
            constant-velocity CPA against the ownship.
        """
        sc = self.scenario
        t = self.t
        own = sc.ownship
        own_wind = sc.wind.at(t, (own.state.x, own.state.y))
        ovx, ovy = own.ground_velocity(own_wind)
        designed_slots = {d.slot for d in sc.designed}

        agents: list[dict[str, Any]] = []
        for slot, agent in enumerate(sc.agents):
            st = agent.state
            wind_xy = sc.wind.at(t, (st.x, st.y))
            vx, vy = agent.ground_velocity(wind_xy)
            dx = st.x - own.state.x
            dy = st.y - own.state.y
            cpa_d, cpa_t = compute_cpa((dx, dy), (vx - ovx, vy - ovy), clamp_future=True)

            entry: dict[str, Any] = {
                "id": int(agent.agent_id),
                "x": round(float(st.x), 1),
                "y": round(float(st.y), 1),
                "psi": round(float(wrap_angle(st.psi)), 3),
                "v": round(float(st.v), 2),
                "type": int(agent.platform),
                "mode": int(agent.mode_code),
                "intent": int(agent.intent_code),
                "alive": bool(agent.alive),
                "ood": bool(agent.is_ood),
                "designed": bool(slot in designed_slots),
                "rng_m": round(float(math.hypot(dx, dy)), 1),
                "cpa_d": round(float(cpa_d), 1),
                "cpa_t": round(float(cpa_t), 2),
                "det": bool(self.ep.obs_detected[max(self._log_index - 1, 0), slot])
                if self._log_index > 0 else False,
                "tid": int(self._track_id[slot]),
                "age": int(self._track_age[slot]),
            }
            if entry["det"] and self._log_index > 0:
                j = self._log_index - 1
                entry["ox"] = round(float(self.ep.obs_pos[j, slot, 0]), 1)
                entry["oy"] = round(float(self.ep.obs_pos[j, slot, 1]), 1)
                R = self.ep.obs_R[j, slot]
                entry["R"] = [round(float(v), 2) for v in
                              (R[0, 0], R[0, 1], R[1, 0], R[1, 1])]
            agents.append(entry)

        frame: dict[str, Any] = {
            "t": round(float(t), 3),
            "tick": int(self._tick),
            "log_index": int(self._log_index),
            "n_steps": int(self.n_steps),
            "duration_s": float(self.cfg.duration_s),
            "wind": sc.wind.to_render_dict(),
            "own": {
                "x": round(float(own.state.x), 1),
                "y": round(float(own.state.y), 1),
                "psi": round(float(wrap_angle(own.state.psi)), 3),
                "v": round(float(own.state.v), 2),
                "vx": round(float(ovx), 2),
                "vy": round(float(ovy), 2),
                "type": int(own.platform),
            },
            "agents": agents,
        }
        if include_statics:
            frame["statics"] = sc.statics()
        return frame

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"SimulationEngine(name={self.cfg.name!r}, seed={self.cfg.seed}, "
                f"t={self.t:.2f}/{self.cfg.duration_s:.1f} s, "
                f"agents={self.scenario.n_agents}, "
                f"logged={self._log_index}/{self.n_steps})")
