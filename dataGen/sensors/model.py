"""Onboard surveillance sensor model: polar noise, Pd, async rate, latency.

Why this module exists, and why it is not optional (doc 3.5, doc 5.2)
---------------------------------------------------------------------
PRISM addresses *non-cooperative* traffic, defined in the paper as an object
that is **not** identified by any cooperative surveillance means -- no ADS-B
Out, no Remote ID, no transponder reply. Hobby drones without Remote ID,
light aircraft without ADS-B Out, birds and transponder-failed aircraft are
the concrete cases. For such an object the ownship possesses exactly one
information channel: its own onboard surveillance sensor (radar or EO/IR),
which returns a geometric track and nothing else.

Three consequences follow, and they are the reason the dataset is built the
way it is:

1. **The prediction model must be trained on observations, not on truth.**
   Ground truth exists in the simulator only because the simulator generated
   it; at inference time it does not exist. Storing GT and OBS in separate
   layers (doc 4) and feeding only OBS to the model is therefore not a
   modelling preference, it is the physical situation.

2. **Type and intent labels are unavailable at inference time.** A radar
   track carries range, bearing and their noise -- not "this is an eVTOL on a
   delivery mission". This is the physical justification for treating type
   inference as an *auxiliary* task predicted from kinematics (doc 5.2)
   rather than as an input feature.

3. **Missing data is information.** Pd < 1, a finite update rate and finite
   latency mean the track is intermittent and stale. A gap must be presented
   to the model as a gap, so that the model learns to widen its predictive
   distribution when the track breaks. Interpolating across a miss would
   destroy exactly the signal that makes the predictor calibrated under real
   track loss. Nothing in this module or in :mod:`sensors.tracker` ever
   invents a sample: a miss produces NaN plus ``detected = False``.

Measurement construction pipeline, per target and per update opportunity:

    1. geometry     -- range r [m] and relative bearing theta [rad] from the
                       ownship, both evaluated at the *delayed* time t_valid
    2. detection    -- Pd(r) * rcs_factor(type), Bernoulli draw
    3. noise        -- sampled in POLAR space and then transformed, with the
                       covariance propagated through the polar->Cartesian
                       Jacobian and rotated into world axes
    4. update rate  -- per-track schedule with an independent phase offset
    5. latency      -- fixed delay plus Gaussian jitter, served from a small
                       per-target ring buffer of past true states

Units are SI throughout: metres, seconds, radians. Angles follow the global
convention (psi CCW from +X = East, wrapped to (-pi, pi]).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable, Mapping, Sequence

import numpy as np

from core.rng import stable_hash
from core.state import AgentState, wrap_angle
from core.types import PlatformType
from scenario.config import SensorConfig

__all__ = [
    "Measurement",
    "SensorTarget",
    "as_target",
    "SensorModel",
]


#: Hard cap on ring-buffer length [samples]; time-based pruning normally keeps
#: the buffers far shorter than this. Guards against an unexpectedly high
#: call rate growing memory without bound.
_MAX_HISTORY = 4096

#: Numerical slack [s] used when testing whether an update is due.
_T_EPS = 1e-9


def _nan_pos() -> np.ndarray:
    """Return a ``[2]`` float64 array of NaN (no position available)."""
    return np.full((2,), np.nan, dtype=np.float64)


def _nan_cov() -> np.ndarray:
    """Return a ``[2,2]`` float64 array of NaN (no covariance available)."""
    return np.full((2, 2), np.nan, dtype=np.float64)


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Measurement:
    """One sensor report about one target at one update opportunity.

    A ``Measurement`` is emitted whenever the per-target update schedule
    fires, whether or not the target was detected. The miss case carries
    ``detected = False`` and NaN payload; it is emitted on purpose so that
    :class:`sensors.tracker.TrackManager` can count consecutive misses.
    Targets that are simply not due for an update produce no ``Measurement``
    at all -- a non-opportunity is not a miss.

    Attributes:
        agent_id: ground-truth id [-] of the observed agent. The tracker maps
            this to a ``track_id``; the id itself is never stored in the
            observation layer (the ownship cannot know it).
        t_emit: time [s] at which the report becomes available to the tracker,
            i.e. the current simulation time.
        t_valid: time [s] the reported geometry was actually true,
            ``t_emit - tau`` where ``tau`` is the latency of this report.
        pos_xy: measured position ``[2]`` [m] in WORLD axes (+X East,
            +Y North); NaN when ``detected`` is False.
        R_2x2: measurement covariance ``[2,2]`` [m^2] in WORLD axes; NaN when
            ``detected`` is False. Symmetric positive definite otherwise.
        detected: True if the Bernoulli detection draw succeeded.
        range_m: true range [m] at ``t_valid`` (diagnostic / GUI only; it is
            NOT part of the stored observation layer).
        bearing_rad: true bearing [rad] relative to the ownship heading at
            ``t_valid``, wrapped to (-pi, pi] (diagnostic only).
        pd: detection probability [-] used for the Bernoulli draw
            (diagnostic only).
    """

    agent_id: int
    t_emit: float
    t_valid: float
    pos_xy: np.ndarray = field(default_factory=_nan_pos)
    R_2x2: np.ndarray = field(default_factory=_nan_cov)
    detected: bool = False
    range_m: float = float("nan")
    bearing_rad: float = float("nan")
    pd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (NaN becomes ``None``)."""

        def _f(v: float) -> Any:
            return None if not math.isfinite(float(v)) else float(v)

        pos = [_f(self.pos_xy[0]), _f(self.pos_xy[1])]
        cov = [[_f(self.R_2x2[i][j]) for j in range(2)] for i in range(2)]
        return {
            "agent_id": int(self.agent_id),
            "t_emit": float(self.t_emit),
            "t_valid": float(self.t_valid),
            "pos_xy": pos,
            "R_2x2": cov,
            "detected": bool(self.detected),
            "range_m": _f(self.range_m),
            "bearing_rad": _f(self.bearing_rad),
            "pd": float(self.pd),
        }


@dataclass(slots=True)
class SensorTarget:
    """Minimal description of an observable object.

    ``SensorModel.measure`` accepts either these, or any duck-typed agent
    object exposing ``agent_id`` / ``state`` / ``platform_type`` (see
    :func:`as_target`), so that :mod:`agents.base` can hand its own agent
    objects straight through without an adapter.

    Attributes:
        agent_id: ground-truth agent id [-].
        state: true kinematic state at the current simulation time.
        platform_type: airframe class, used only to look up the RCS factor.
    """

    agent_id: int
    state: AgentState
    platform_type: PlatformType = PlatformType.UNKNOWN


def as_target(obj: Any) -> SensorTarget:
    """Coerce an agent-like object into a :class:`SensorTarget`.

    Accepted forms:
        * a :class:`SensorTarget` (returned unchanged);
        * any object with ``state`` plus one of ``agent_id`` / ``aid`` /
          ``id``, and optionally one of ``platform_type`` / ``ptype`` /
          ``type`` (this covers the agent objects of :mod:`agents.base`);
        * a ``(agent_id, state)`` or ``(agent_id, state, platform_type)``
          tuple / list.

    Args:
        obj: the object to coerce.

    Returns:
        An equivalent :class:`SensorTarget`. The ``state`` is referenced, not
        copied -- the sensor only reads from it.

    Raises:
        TypeError: if no agent id or no state can be found on ``obj``.
    """
    if isinstance(obj, SensorTarget):
        return obj
    if isinstance(obj, (tuple, list)):
        if len(obj) == 2:
            return SensorTarget(int(obj[0]), obj[1], PlatformType.UNKNOWN)
        if len(obj) >= 3:
            return SensorTarget(int(obj[0]), obj[1], PlatformType(int(obj[2])))
        raise TypeError("sensor target tuple must be (agent_id, state[, type])")

    aid: Any = None
    for name in ("agent_id", "aid", "id"):
        if hasattr(obj, name):
            aid = getattr(obj, name)
            break
    if aid is None:
        raise TypeError(f"object {type(obj).__name__} exposes no agent id")

    state: Any = getattr(obj, "state", None)
    if state is None:
        if all(hasattr(obj, f) for f in ("x", "y", "psi")):
            state = obj
        else:
            raise TypeError(f"object {type(obj).__name__} exposes no state")

    ptype: Any = None
    for name in ("platform_type", "ptype", "type"):
        if hasattr(obj, name):
            ptype = getattr(obj, name)
            break
    ptype_i = int(ptype) if ptype is not None else int(PlatformType.UNKNOWN)
    try:
        pt = PlatformType(ptype_i)
    except ValueError:
        pt = PlatformType.UNKNOWN
    return SensorTarget(int(aid), state, pt)


# --------------------------------------------------------------------------
# Sensor model
# --------------------------------------------------------------------------


class SensorModel:
    """Degrades ground truth into an onboard surveillance measurement stream.

    The model owns two pieces of mutable state, both of which are captured by
    :meth:`snapshot` so that branch rollouts (doc 3.3) can restart cleanly:

        * a per-target ring buffer of past true states, used to serve delayed
          reports (the latency implementation);
        * a per-target update schedule with an independent phase offset, so
          that tracks do not all refresh on the same tick.

    The phase offset is derived from a stable CRC-32 hash of
    ``(phase_seed, agent_id)``, not from a random stream. That keeps the
    schedule identical when a branch rollout re-randomises the decision
    streams, which is exactly what doc 3.3 requires: the spread of the branch
    futures must come from the agents' decisions, never from a different
    sensor sampling pattern.
    """

    def __init__(self, cfg: SensorConfig, phase_seed: int = 0) -> None:
        """Create a sensor model.

        Args:
            cfg: sensor parameters (see :class:`scenario.config.SensorConfig`).
            phase_seed: integer [-] mixed into the per-target update-phase
                hash. Pass the scenario seed so that two scenarios do not
                share the same phase pattern; the value is recorded by
                :meth:`snapshot`.
        """
        self._cfg = cfg
        self._phase_seed = int(phase_seed)
        self._own_hist: Deque[tuple[float, float, float, float]] = deque(
            maxlen=_MAX_HISTORY
        )
        self._hist: dict[int, Deque[tuple[float, float, float, float]]] = {}
        self._next_update_t: dict[int, float] = {}
        self._rcs: dict[int, float] = {
            int(k): float(v) for k, v in cfg.rcs_factor.items()
        }

    # -- properties --------------------------------------------------------

    @property
    def cfg(self) -> SensorConfig:
        """The sensor configuration this model was built with."""
        return self._cfg

    @property
    def update_period_s(self) -> float:
        """Nominal sensor update period [s] = ``1 / update_rate_hz``."""
        rate = float(self._cfg.update_rate_hz)
        return 1.0 / rate if rate > 0.0 else float("inf")

    @property
    def history_horizon_s(self) -> float:
        """Length [s] of the retained true-state history.

        Sized to cover the worst realistic latency (mean plus six jitter
        standard deviations) with half a second of margin, so a delayed
        lookup practically never clamps at the buffer edge.
        """
        return (
            float(self._cfg.latency_s)
            + 6.0 * float(self._cfg.latency_jitter_s)
            + 0.5
        )

    # -- public physics ----------------------------------------------------

    def rcs_factor(self, platform_type: PlatformType | int) -> float:
        """Detectability multiplier [-] for a platform type.

        Small, low-reflectivity bodies (birds, debris, small multirotors) get
        a factor below 1 and are therefore genuinely harder to hold. That
        asymmetry is realistic and it is wanted in the dataset: the predictor
        must cope with the fact that the hardest-to-predict objects are also
        the ones whose tracks break most often.

        Args:
            platform_type: airframe class (enum or raw int).

        Returns:
            Positive multiplier applied to the range-dependent Pd; 1.0 for a
            type absent from the configuration table.
        """
        return float(self._rcs.get(int(platform_type), 1.0))

    def range_falloff(self, r: float) -> float:
        """Range-dependent detection factor [-] in 0..1.

        Unity inside ``r_full_m``, then a raised-cosine decay to exactly zero
        at ``r_max_m``. A smooth falloff (rather than a hard step) is used so
        that targets near the edge of the sensor envelope produce realistic
        intermittent tracks rather than a sharp on/off boundary.

        Args:
            r: range [m], non-negative.

        Returns:
            The multiplicative range factor [-].
        """
        r_full = float(self._cfg.r_full_m)
        r_max = float(self._cfg.r_max_m)
        if r <= r_full:
            return 1.0
        if r >= r_max:
            return 0.0
        span = r_max - r_full
        if span <= 0.0:
            return 0.0
        u = (float(r) - r_full) / span
        return 0.5 * (1.0 + math.cos(math.pi * u))

    def detection_probability(
        self,
        r: float,
        platform_type: PlatformType | int = PlatformType.UNKNOWN,
    ) -> float:
        """Detection probability [-] of one target at range ``r``.

        ``Pd = clip(pd_max * range_falloff(r) * rcs_factor(type), 0, 1)``.
        The clip matters for large, highly reflective platforms whose RCS
        factor exceeds 1: their Pd saturates at 1 instead of exceeding it.

        Args:
            r: range [m] from the ownship to the target.
            platform_type: airframe class of the target.

        Returns:
            Probability in 0..1.
        """
        if not self._cfg.enabled:
            return 1.0
        pd = (
            float(self._cfg.pd_max)
            * self.range_falloff(float(r))
            * self.rcs_factor(platform_type)
        )
        if pd < 0.0:
            return 0.0
        if pd > 1.0:
            return 1.0
        return pd

    def noise_sigmas(self, r: float) -> tuple[float, float]:
        """Polar measurement standard deviations at range ``r``.

        Args:
            r: range [m].

        Returns:
            ``(sigma_r, sigma_theta)`` with ``sigma_r`` in [m] and
            ``sigma_theta`` in [rad]. Range noise grows with range as
            ``sigma_r0_m + sigma_r_frac * r``; bearing noise is constant in
            angle, which is what makes the Cartesian error ellipse stretch
            *across* the line of sight at long range.
        """
        sr = float(self._cfg.sigma_r0_m) + float(self._cfg.sigma_r_frac) * float(r)
        return sr, float(self._cfg.sigma_theta_rad)

    @staticmethod
    def cartesian_covariance(
        r: float,
        theta_rel: float,
        psi_own: float,
        sigma_r: float,
        sigma_theta: float,
    ) -> np.ndarray:
        """Propagate polar noise into a world-frame Cartesian covariance.

        In the ownship-relative frame (x along the ownship heading) the
        polar-to-Cartesian map has Jacobian

            J = [[cos th, -r sin th],
                 [sin th,  r cos th]]

        so the relative-frame covariance is ``R_rel = J diag(sr^2, st^2) J^T``.
        Rotating the relative frame into world axes by the ownship heading
        gives ``R_world = C R_rel C^T`` with ``C = Rot(psi_own)``.

        The result is the range-elongated ellipse a real sensor produces: the
        along-range standard deviation is ``sigma_r`` while the across-range
        standard deviation is ``r * sigma_theta`` and therefore grows linearly
        with range. Getting this Jacobian right is what makes the stored
        ``obs_R`` a usable uncertainty input rather than a decoration.

        Args:
            r: range [m].
            theta_rel: bearing [rad] relative to the ownship heading.
            psi_own: ownship heading [rad], CCW from +X.
            sigma_r: range standard deviation [m].
            sigma_theta: bearing standard deviation [rad].

        Returns:
            ``[2,2]`` float64 symmetric positive-definite covariance [m^2] in
            world axes (+X East, +Y North).
        """
        ct = math.cos(theta_rel)
        st = math.sin(theta_rel)
        jac = np.array([[ct, -r * st], [st, r * ct]], dtype=np.float64)
        d = np.array(
            [[sigma_r * sigma_r, 0.0], [0.0, sigma_theta * sigma_theta]],
            dtype=np.float64,
        )
        r_rel = jac @ d @ jac.T
        cp = math.cos(psi_own)
        sp = math.sin(psi_own)
        rot = np.array([[cp, -sp], [sp, cp]], dtype=np.float64)
        r_world = rot @ r_rel @ rot.T
        # Enforce exact symmetry against float round-off; storage/schema.py
        # validates symmetry and positive definiteness.
        return 0.5 * (r_world + r_world.T)

    # -- main entry point --------------------------------------------------

    def measure(
        self,
        t: float,
        ownship_state: AgentState,
        agents: Iterable[Any],
        rng_noise: np.random.Generator,
        rng_detect: np.random.Generator,
        rng_latency: np.random.Generator | None = None,
    ) -> list[Measurement]:
        """Produce the sensor reports available at simulation time ``t``.

        Call this once per integration step (``SIM_DT``): the per-target
        schedule then resolves the sensor update rate exactly, and the true
        state history is sampled finely enough for the latency lookup. Calling
        it more slowly than ``update_rate_hz`` simply yields at most one
        report per target per call.

        Args:
            t: current simulation time [s].
            ownship_state: true ownship state at ``t``.
            agents: observable objects; anything :func:`as_target` accepts.
            rng_noise: generator for the polar noise draws (stream
                ``'sensor_noise'``).
            rng_detect: generator for the Bernoulli detection draws (stream
                ``'sensor_detect'``).
            rng_latency: generator for latency jitter (stream
                ``'sensor_latency'``). Falls back to ``rng_noise`` when None,
                so a two-stream caller still works.

        Returns:
            One :class:`Measurement` per target whose update schedule fired at
            this time, detections and misses alike. Targets not due for an
            update are absent from the list.
        """
        t = float(t)
        if rng_latency is None:
            rng_latency = rng_noise

        targets = [as_target(a) for a in agents]
        self._push_history(t, ownship_state, targets)

        out: list[Measurement] = []
        period = self.update_period_s
        for tgt in targets:
            aid = int(tgt.agent_id)
            if not self._is_due(aid, t, period):
                continue
            out.append(self._measure_one(t, tgt, rng_noise, rng_detect, rng_latency))
        return out

    # -- internals ---------------------------------------------------------

    def _measure_one(
        self,
        t: float,
        tgt: SensorTarget,
        rng_noise: np.random.Generator,
        rng_detect: np.random.Generator,
        rng_latency: np.random.Generator,
    ) -> Measurement:
        """Build one report for one target that is due for an update."""
        cfg = self._cfg
        aid = int(tgt.agent_id)

        # -- latency -------------------------------------------------------
        if cfg.enabled and (cfg.latency_s > 0.0 or cfg.latency_jitter_s > 0.0):
            jitter = (
                float(rng_latency.normal(0.0, float(cfg.latency_jitter_s)))
                if cfg.latency_jitter_s > 0.0
                else 0.0
            )
            tau = float(cfg.latency_s) + jitter
            if tau < 0.0:
                tau = 0.0
        else:
            tau = 0.0
        t_valid = t - tau

        own_x, own_y, own_psi = self._sample_history(self._own_hist, t_valid)
        hist = self._hist.get(aid)
        if hist is None or not hist:
            tgt_x, tgt_y = float(tgt.state.x), float(tgt.state.y)
        else:
            tgt_x, tgt_y, _ = self._sample_history(hist, t_valid)

        # -- geometry at the delayed time ----------------------------------
        dx = tgt_x - own_x
        dy = tgt_y - own_y
        r = math.hypot(dx, dy)
        theta = wrap_angle(math.atan2(dy, dx) - own_psi)

        # -- sensor disabled: perfect passthrough (debug only) -------------
        if not cfg.enabled:
            return Measurement(
                agent_id=aid,
                t_emit=t,
                t_valid=t_valid,
                pos_xy=np.array([tgt_x, tgt_y], dtype=np.float64),
                R_2x2=np.eye(2, dtype=np.float64) * 1e-6,
                detected=True,
                range_m=r,
                bearing_rad=theta,
                pd=1.0,
            )

        # -- detection ------------------------------------------------------
        pd = self.detection_probability(r, tgt.platform_type)
        detected = bool(float(rng_detect.random()) < pd)
        if not detected:
            return Measurement(
                agent_id=aid,
                t_emit=t,
                t_valid=t_valid,
                pos_xy=_nan_pos(),
                R_2x2=_nan_cov(),
                detected=False,
                range_m=r,
                bearing_rad=theta,
                pd=pd,
            )

        # -- noise: sampled in POLAR space, then transformed ----------------
        sigma_r, sigma_theta = self.noise_sigmas(r)
        dr = float(rng_noise.normal(0.0, sigma_r)) if sigma_r > 0.0 else 0.0
        dth = float(rng_noise.normal(0.0, sigma_theta)) if sigma_theta > 0.0 else 0.0
        r_meas = r + dr
        if r_meas < 0.0:
            # A negative range is unphysical; reflect it through the origin so
            # the reported point stays in front of the sensor.
            r_meas = abs(r_meas)
        th_meas = wrap_angle(theta + dth)

        alpha = own_psi + th_meas  # world-frame bearing of the measured point
        px = own_x + r_meas * math.cos(alpha)
        py = own_y + r_meas * math.sin(alpha)

        # The reported covariance is evaluated at the MEASURED geometry,
        # because a real sensor knows only what it measured. The difference
        # from the true-geometry covariance is O(sigma_r / r) and vanishes as
        # the target approaches.
        sigma_r_m, sigma_theta_m = self.noise_sigmas(r_meas)
        cov = self.cartesian_covariance(
            r_meas, th_meas, own_psi, sigma_r_m, sigma_theta_m
        )

        return Measurement(
            agent_id=aid,
            t_emit=t,
            t_valid=t_valid,
            pos_xy=np.array([px, py], dtype=np.float64),
            R_2x2=cov,
            detected=True,
            range_m=r,
            bearing_rad=theta,
            pd=pd,
        )

    def _is_due(self, aid: int, t: float, period: float) -> bool:
        """Advance the per-target schedule and report whether it fired.

        The first sight of a target arms its schedule at ``t + phase*period``
        (or at ``t`` when async phasing is disabled), so that targets acquired
        at the same instant still refresh on different ticks.
        """
        nxt = self._next_update_t.get(aid)
        if nxt is None:
            if self._cfg.async_phase and math.isfinite(period):
                nxt = t + self._phase(aid) * period
            else:
                nxt = t
            self._next_update_t[aid] = nxt
        if t + _T_EPS < nxt:
            return False
        if not math.isfinite(period) or period <= 0.0:
            self._next_update_t[aid] = float("inf")
            return True
        # Catch up if the caller stepped by more than one sensor period.
        while nxt <= t + _T_EPS:
            nxt += period
        self._next_update_t[aid] = nxt
        return True

    def _phase(self, aid: int) -> float:
        """Deterministic per-target update phase [-] in 0..1."""
        h = stable_hash(f"sensor_phase:{self._phase_seed}:{int(aid)}")
        return (h & 0xFFFFFFFF) / 4294967296.0

    def _push_history(
        self,
        t: float,
        ownship_state: AgentState,
        targets: Sequence[SensorTarget],
    ) -> None:
        """Append the current true states and prune stale buffer entries."""
        self._own_hist.append(
            (
                t,
                float(ownship_state.x),
                float(ownship_state.y),
                float(ownship_state.psi),
            )
        )
        self._prune(self._own_hist, t)
        seen: set[int] = set()
        for tgt in targets:
            aid = int(tgt.agent_id)
            seen.add(aid)
            buf = self._hist.get(aid)
            if buf is None:
                buf = deque(maxlen=_MAX_HISTORY)
                self._hist[aid] = buf
            buf.append(
                (t, float(tgt.state.x), float(tgt.state.y), float(tgt.state.psi))
            )
            self._prune(buf, t)
        # Forget agents that stopped existing, so long episodes cannot leak.
        horizon = self.history_horizon_s
        stale = [
            aid
            for aid, buf in self._hist.items()
            if aid not in seen and (not buf or buf[-1][0] < t - horizon)
        ]
        for aid in stale:
            self._hist.pop(aid, None)
            self._next_update_t.pop(aid, None)

    def _prune(
        self,
        buf: Deque[tuple[float, float, float, float]],
        t: float,
    ) -> None:
        """Drop entries older than the latency horizon, keeping >= 2 samples."""
        cutoff = t - self.history_horizon_s
        while len(buf) > 2 and buf[0][0] < cutoff and buf[1][0] < cutoff:
            buf.popleft()

    @staticmethod
    def _sample_history(
        buf: Deque[tuple[float, float, float, float]],
        t_query: float,
    ) -> tuple[float, float, float]:
        """Linearly interpolate a buffered true state at ``t_query``.

        This interpolates GROUND TRUTH between two finely spaced integration
        samples in order to evaluate a delayed report -- it is emphatically
        NOT interpolation of a missing observation, which is forbidden
        (doc 3.5). Queries before the oldest sample clamp to that sample,
        which only happens in the first few steps of an episode.

        Args:
            buf: ring buffer of ``(t, x, y, psi)`` entries, ascending in t.
            t_query: time [s] to evaluate.

        Returns:
            ``(x, y, psi)`` with x, y in [m] and psi in [rad], wrapped.
        """
        if not buf:
            return 0.0, 0.0, 0.0
        if t_query >= buf[-1][0]:
            e = buf[-1]
            return e[1], e[2], e[3]
        if t_query <= buf[0][0]:
            e = buf[0]
            return e[1], e[2], e[3]
        prev = buf[0]
        for entry in buf:
            if entry[0] >= t_query:
                t0, x0, y0, p0 = prev
                t1, x1, y1, p1 = entry
                span = t1 - t0
                w = 0.0 if span <= 0.0 else (t_query - t0) / span
                psi = wrap_angle(p0 + w * wrap_angle(p1 - p0))
                return x0 + w * (x1 - x0), y0 + w * (y1 - y0), psi
            prev = entry
        e = buf[-1]
        return e[1], e[2], e[3]

    # -- snapshot / restore -------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture the full mutable state (history buffers and schedule).

        Returns:
            A JSON-safe dict; agent ids become string keys.
        """
        return {
            "phase_seed": int(self._phase_seed),
            "own_hist": [list(e) for e in self._own_hist],
            "hist": {
                str(aid): [list(e) for e in buf] for aid, buf in self._hist.items()
            },
            "next_update_t": {
                str(aid): float(v) for aid, v in self._next_update_t.items()
            },
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        """Restore the state captured by :meth:`snapshot`.

        Args:
            snap: a mapping produced by :meth:`snapshot`.
        """
        self._phase_seed = int(snap.get("phase_seed", self._phase_seed))
        self._own_hist = deque(
            (tuple(float(v) for v in e) for e in snap.get("own_hist", [])),
            maxlen=_MAX_HISTORY,
        )
        self._hist = {}
        for aid, entries in dict(snap.get("hist", {})).items():
            self._hist[int(aid)] = deque(
                (tuple(float(v) for v in e) for e in entries), maxlen=_MAX_HISTORY
            )
        self._next_update_t = {
            int(aid): float(v)
            for aid, v in dict(snap.get("next_update_t", {})).items()
        }

    def reset(self) -> None:
        """Clear all history and schedules (start of a fresh episode)."""
        self._own_hist.clear()
        self._hist.clear()
        self._next_update_t.clear()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"SensorModel(rate={self._cfg.update_rate_hz} Hz, "
            f"pd_max={self._cfg.pd_max}, tracks={len(self._hist)})"
        )
