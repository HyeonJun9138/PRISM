"""The :class:`Agent` container: one traffic object in the 2D world.

An agent is the binding of four things that are otherwise independent:

    envelope   what it physically cannot do          (agents.platforms)
    intent     what it is trying to achieve          (agents.intent)
    maneuver   the discrete manoeuvre it is latched  (agents.maneuver)
    state      where it is right now                 (core.state)

Per-step control flow (doc 3.3):

    1. the intent proposes ``(omega_cmd, a_cmd)``; PASSIVE_DRIFT proposes
       nothing and returns ``None``
    2. the maneuver mode machine may OVERRIDE that proposal while a discrete
       manoeuvre is latched, and it is what enforces the dwell-time constraint
       of doc 3.3 (b)
    3. the command is clipped to the performance envelope -- doc 3.3 (a) -- and
       handed to the integrator: powered bodies to ``step_powered``, unpowered
       bodies (DEBRIS) to ``step_passive``

Snapshot / restore is load-bearing, not a convenience. doc 3.3 generates the
calibration ground truth by replaying the same situation N times with only the
decision streams reseeded. If :meth:`Agent.snapshot` omitted any piece of
state, the N rollouts would not start from an identical situation and the
"true" distribution used to score calibration would be silently wrong. The
snapshot therefore captures the kinematic state, the mode and its dwell timer,
the mode payload, ALL intent bookkeeping, the agent-level flags and the
agent's private RNG bit-generator state.

Units: SI everywhere (m, m/s, m/s^2, rad, rad/s, s).
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from core.state import AgentSnapshot, AgentState, PerformanceEnvelope, wrap_angle
from core.types import IntentType, ManeuverMode, PlatformType

__all__ = [
    "MACHINE_STATE_KEY",
    "AGENT_STATE_KEY",
    "REACTION_STATE_KEY",
    "Agent",
]


# --------------------------------------------------------------------------
# Reserved snapshot keys
# --------------------------------------------------------------------------
#
# AgentSnapshot has a fixed field set (state / mode / mode_timer / mode_params
# / intent_state / rng_state). Two things still need to survive a round trip:
# extra bookkeeping owned by the mode machine, and the agent-level flags. They
# are nested under reserved keys inside the two free-form dicts rather than by
# widening the shared contract, which is owned by core.state.

#: Reserved key inside ``AgentSnapshot.mode_params`` holding any extra mode
#: machine state beyond mode / mode_timer / mode_params.
MACHINE_STATE_KEY: str = "_machine"

#: Key under which the reaction layer's state travels inside
#: ``AgentSnapshot.intent_state`` (rev.3; mirrors MACHINE_STATE_KEY).
REACTION_STATE_KEY: str = "_reaction"

#: Reserved key inside ``AgentSnapshot.intent_state`` holding the agent-level
#: flags (id, platform, alive, spawn time, OOD mark).
AGENT_STATE_KEY: str = "_agent"


# --------------------------------------------------------------------------
# JSON-safety helpers (snapshots must survive storage.writer / GUI transport)
# --------------------------------------------------------------------------


def _coerce(value: Any) -> tuple[bool, Any]:
    """Convert a value to a JSON-safe form if possible.

    Args:
        value: any object taken from an intent or mode-machine attribute.

    Returns:
        ``(ok, converted)``. ``ok`` is False when the value cannot be
        represented as plain JSON data, in which case it must be dropped from
        the snapshot rather than silently serialised as a repr.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True, value
    if isinstance(value, (np.bool_,)):
        return True, bool(value)
    if isinstance(value, np.integer):
        return True, int(value)
    if isinstance(value, np.floating):
        return True, float(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biufc" and value.size <= 4096:
            return True, value.tolist()
        return False, None
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            ok, conv = _coerce(item)
            if not ok:
                return False, None
            out.append(conv)
        return True, out
    if isinstance(value, Mapping):
        d: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                return False, None
            ok, conv = _coerce(v)
            if not ok:
                return False, None
            d[k] = conv
        return True, d
    return False, None


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Coerce a mapping to a JSON-safe dict, dropping unrepresentable entries."""
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            continue
        ok, conv = _coerce(v)
        if ok:
            out[k] = conv
    return out


def _attribute_names(obj: Any) -> list[str]:
    """Public instance-attribute names of ``obj`` (handles ``__slots__``)."""
    names: list[str] = []
    seen: set[str] = set()
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for name in d:
            if not name.startswith("_") and name not in seen:
                seen.add(name)
                names.append(name)
    for klass in type(obj).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if not name.startswith("_") and name not in seen:
                seen.add(name)
                names.append(name)
    return names


#: Save / load method-name pairs probed on intent and mode-machine objects, in
#: priority order. The first pair whose BOTH members are callable is used, so a
#: component that implements its own serialisation always wins over the
#: attribute-scraping fallback.
_STATE_METHOD_PAIRS: tuple[tuple[str, str], ...] = (
    ("to_state_dict", "from_state_dict"),
    ("state_dict", "load_state_dict"),
    ("get_state", "set_state"),
    ("snapshot_state", "restore_state"),
    ("snapshot", "restore"),
)


def _capture_component(obj: Any, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Capture the resumable state of an intent or mode-machine object.

    Args:
        obj: the component, or ``None``.
        exclude: attribute names to skip in the fallback path (used for fields
            that already have a dedicated :class:`AgentSnapshot` slot).

    Returns:
        A JSON-safe dict. Empty when ``obj`` is ``None`` or carries no
        representable state.
    """
    if obj is None:
        return {}
    for save_name, load_name in _STATE_METHOD_PAIRS:
        saver = getattr(obj, save_name, None)
        loader = getattr(obj, load_name, None)
        if callable(saver) and callable(loader):
            try:
                raw = saver()
            except TypeError:
                continue
            return _coerce_dict(raw)
    skip = set(exclude)
    out: dict[str, Any] = {}
    for name in _attribute_names(obj):
        if name in skip:
            continue
        ok, conv = _coerce(getattr(obj, name, None))
        if ok:
            out[name] = conv
    return out


def _restore_component(obj: Any, data: Mapping[str, Any]) -> None:
    """Restore an intent or mode-machine object captured by :func:`_capture_component`.

    Args:
        obj: the component, or ``None`` (then nothing happens).
        data: the dict produced by :func:`_capture_component`.
    """
    if obj is None or not data:
        return
    for save_name, load_name in _STATE_METHOD_PAIRS:
        saver = getattr(obj, save_name, None)
        loader = getattr(obj, load_name, None)
        if callable(saver) and callable(loader):
            loader(dict(data))
            return
    for name, value in data.items():
        if name.startswith("_"):
            continue
        try:
            setattr(obj, name, copy.deepcopy(value))
        except AttributeError:
            # Read-only property on the component: nothing sensible to do, and
            # raising here would abort an otherwise valid branch rollout.
            continue


# --------------------------------------------------------------------------
# Cross-module call binding
# --------------------------------------------------------------------------
#
# agents/maneuver.py and core/integrator.py are written by other modules. Only
# the intent call signature is pinned by the shared contract
# (``intent.command(agent, t, ownship_state, rng)``); the other two are
# described in prose. The binder below inspects the callee once and maps our
# canonical argument names onto whatever it actually declares, falling back to
# a positional call in canonical order. If binding is impossible the resulting
# TypeError is re-raised with the signature it saw, so an integration mismatch
# is a loud, one-line diagnosis instead of a silent wrong-argument bug.

_BIND_CACHE: dict[Any, tuple[tuple[str, str, bool], ...] | None] = {}


def _build_binding(
    fn: Any,
    canonical_order: Sequence[str],
    aliases: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, str, bool], ...] | None:
    """Work out how to call ``fn`` with our canonical arguments.

    Args:
        fn: the callable to bind.
        canonical_order: our argument names, in the order we would pass them
            positionally.
        aliases: canonical name -> accepted parameter names on the callee.

    Returns:
        A tuple of ``(param_name, canonical_name, positional_only)`` in
        declaration order, or ``None`` when the signature could not be mapped
        (the caller then falls back to a plain positional call).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    lookup: dict[str, str] = {}
    for canon in canonical_order:
        lookup.setdefault(canon, canon)
        for alias in aliases.get(canon, ()):
            lookup.setdefault(alias, canon)

    plan: list[tuple[str, str, bool]] = []
    used: set[str] = set()
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        canon = lookup.get(param.name)
        if canon is None or canon in used:
            if param.default is not param.empty:
                continue  # optional parameter we know nothing about: leave it
            return None
        used.add(canon)
        plan.append((param.name, canon, param.kind == param.POSITIONAL_ONLY))
    if not plan:
        return None
    return tuple(plan)


def _adaptive_call(
    fn: Any,
    values: Mapping[str, Any],
    canonical_order: Sequence[str],
    aliases: Mapping[str, tuple[str, ...]],
) -> Any:
    """Call ``fn`` with ``values``, tolerating parameter-name differences.

    Args:
        fn: the callable.
        values: canonical name -> argument value.
        canonical_order: fallback positional order.
        aliases: canonical name -> accepted parameter names on the callee.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        TypeError: if neither the bound nor the positional call is accepted;
            the message names the signature that was actually found.
    """
    key = getattr(fn, "__func__", fn)
    try:
        plan = _BIND_CACHE[key]
        cached = True
    except (KeyError, TypeError):
        plan = _build_binding(fn, canonical_order, aliases)
        cached = False
    if not cached:
        try:
            _BIND_CACHE[key] = plan
        except TypeError:
            pass

    try:
        if plan is None:
            return fn(*[values[name] for name in canonical_order])
        args = [values[c] for (_n, c, pos_only) in plan if pos_only]
        kwargs = {n: values[c] for (n, c, pos_only) in plan if not pos_only}
        return fn(*args, **kwargs)
    except TypeError as exc:
        try:
            sig_text = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig_text = "<unavailable>"
        raise TypeError(
            f"agents.base could not call {getattr(fn, '__qualname__', fn)!r}: "
            f"expected arguments {tuple(canonical_order)} but the callee "
            f"declares {sig_text}. Original error: {exc}"
        ) from exc


#: Parameter-name aliases accepted for the integrator entry points.
_INTEGRATOR_ALIASES: dict[str, tuple[str, ...]] = {
    "state": ("s", "st", "agent_state"),
    "envelope": ("env", "envl", "perf", "performance", "limits"),
    "omega_cmd": ("omega", "omega_command", "omega_c", "w_cmd", "turn_rate_cmd"),
    "a_cmd": ("accel_cmd", "a_command", "acc_cmd", "accel", "a_long_cmd"),
    "dt": ("h", "delta_t", "timestep", "step", "sim_dt"),
    "wind_xy": ("wind", "wind_vec", "wind_vector", "wind_ms", "w_xy"),
}

#: Canonical positional order for ``step_powered``.
_POWERED_ORDER: tuple[str, ...] = (
    "state",
    "envelope",
    "omega_cmd",
    "a_cmd",
    "dt",
    "wind_xy",
)

#: Canonical positional order for ``step_passive``.
_PASSIVE_ORDER: tuple[str, ...] = ("state", "envelope", "dt", "wind_xy")

#: Parameter-name aliases accepted for the mode machine's update method.
_MACHINE_ALIASES: dict[str, tuple[str, ...]] = {
    "agent": ("ag", "owner", "self_agent"),
    "t": ("time", "t_s", "sim_time"),
    "dt": ("h", "delta_t", "timestep", "step", "sim_dt"),
    "omega_cmd": ("omega", "omega_command", "omega_c", "w_cmd", "turn_rate_cmd"),
    "a_cmd": ("accel_cmd", "a_command", "acc_cmd", "accel", "a_long_cmd"),
    "rng": ("generator", "gen", "random", "rand"),
}

#: Canonical positional order for the mode machine's update method.
_MACHINE_ORDER: tuple[str, ...] = ("agent", "t", "dt", "omega_cmd", "a_cmd", "rng")

#: Method names probed on the mode machine, in priority order.
_MACHINE_METHODS: tuple[str, ...] = (
    "modulate",
    "update",
    "step",
    "apply",
    "command",
    "__call__",
)

#: Modules searched for ``step_powered`` / ``step_passive``, in priority order.
_INTEGRATOR_MODULES: tuple[str, ...] = ("core.integrator", "core.kinematics")

_INTEGRATOR_FUNCS: dict[str, Any] = {}


def _integrator_func(name: str) -> Any:
    """Resolve ``step_powered`` / ``step_passive`` lazily and cache it.

    The import is deferred so that this module can be imported (and its
    platform catalogue tested) before :mod:`core.integrator` exists.

    Args:
        name: ``"step_powered"`` or ``"step_passive"``.

    Returns:
        The integrator callable.

    Raises:
        ImportError: if no integrator module could be imported.
        AttributeError: if none of them exports ``name``.
    """
    fn = _INTEGRATOR_FUNCS.get(name)
    if fn is not None:
        return fn
    last_error: BaseException | None = None
    tried: list[str] = []
    for mod_name in _INTEGRATOR_MODULES:
        try:
            module = __import__(mod_name, fromlist=["*"])
        except ImportError as exc:  # module not written / not importable yet
            last_error = exc
            continue
        tried.append(mod_name)
        candidate = getattr(module, name, None)
        if callable(candidate):
            _INTEGRATOR_FUNCS[name] = candidate
            return candidate
    if not tried:
        raise ImportError(
            f"agents.base needs {name}, but none of {_INTEGRATOR_MODULES} could "
            f"be imported ({last_error})."
        )
    raise AttributeError(
        f"agents.base needs {name}, but none of {tuple(tried)} exports it."
    )


def _interpret_machine_result(
    result: Any,
    omega_cmd: float,
    a_cmd: float,
) -> tuple[float, float]:
    """Normalise whatever the mode machine returned into ``(omega_cmd, a_cmd)``.

    Accepted forms: ``None`` (command unchanged), a 2-sequence, a 3-sequence
    whose first element is the resulting mode, or a mapping keyed by
    ``omega_cmd`` / ``omega`` and ``a_cmd`` / ``a``.

    Args:
        result: the mode machine's return value.
        omega_cmd: the turn rate [rad/s] proposed by the intent.
        a_cmd: the longitudinal acceleration [m/s^2] proposed by the intent.

    Returns:
        ``(omega_cmd, a_cmd)`` in SI units.

    Raises:
        TypeError: if the return value has no recognisable form.
    """
    if result is None:
        return omega_cmd, a_cmd
    if isinstance(result, Mapping):
        w = result.get("omega_cmd", result.get("omega", omega_cmd))
        a = result.get("a_cmd", result.get("a", a_cmd))
        return float(w), float(a)
    if isinstance(result, (tuple, list, np.ndarray)):
        seq = list(result)
        if len(seq) == 2:
            return float(seq[0]), float(seq[1])
        if len(seq) == 3:
            # (mode, omega_cmd, a_cmd)
            return float(seq[1]), float(seq[2])
    raise TypeError(
        "agents.base: the maneuver mode machine returned "
        f"{type(result).__name__}; expected None, a 2-sequence "
        "(omega_cmd, a_cmd), a 3-sequence (mode, omega_cmd, a_cmd) or a mapping."
    )


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Agent:
    """One traffic object: envelope + intent + mode machine + kinematic state.

    Attributes:
        agent_id: stable integer id [-], unique within an episode. Written to
            ``EpisodeArrays.agent_ids``.
        platform: airframe class; the auxiliary ground-truth label of doc 5.1.
        envelope: hard physical limits (see :mod:`agents.platforms`).
        state: current kinematic state [m, m/s, rad].
        intent: an ``IntentController`` exposing
            ``command(agent, t, ownship_state, rng) -> (omega_cmd, a_cmd) |
            None``. ``None`` means "no command at all" and is what a
            PASSIVE_DRIFT intent returns.
        maneuver: a ``ModeMachine`` exposing ``mode``, ``mode_timer``,
            ``mode_params`` and an update method that may override the intent's
            command while a discrete manoeuvre is latched.
        alive: False once the agent has left the region or been retired; a dead
            agent ignores :meth:`step` and is logged as invalid.
        spawn_t: simulation time [s] at which the agent entered the episode.
        behavior: optional personality object (:mod:`agents.behavior`) shared
            with the intent / mode machine. Not stepped here.
        rng: the agent's PRIVATE generator, normally
            ``RNGHub.spawn_child('mode_decision', agent_id)``. Its bit-generator
            state is captured by :meth:`snapshot`. May be ``None`` when the
            agent draws only from shared hub streams, whose state is captured by
            ``RNGHub.snapshot`` instead.
        is_ood: True for agents built as out-of-distribution (doc 8.1), i.e.
            ``PlatformType.UNKNOWN``. Recorded in metadata, never fed to the
            model.
    """

    agent_id: int
    platform: PlatformType
    envelope: PerformanceEnvelope
    state: AgentState
    intent: Any = None
    maneuver: Any = None
    alive: bool = True
    spawn_t: float = 0.0
    behavior: Any = None
    rng: np.random.Generator | None = None
    is_ood: bool = False
    #: rev.3 reaction layer (agents/reaction.py), or None (DEBRIS, disabled).
    #: Driven by the engine's 2 Hz perception pass; consulted every step.
    reaction: Any = None

    # -- read-only views --------------------------------------------------

    @property
    def mode(self) -> ManeuverMode:
        """Current maneuver mode; CV when no mode machine is attached."""
        raw = getattr(self.maneuver, "mode", ManeuverMode.CV)
        try:
            return ManeuverMode(int(raw))
        except (TypeError, ValueError):
            return ManeuverMode.CV

    @property
    def mode_code(self) -> int:
        """Current maneuver mode as the int stored in ``gt_mode`` [-]."""
        return int(self.mode)

    @property
    def intent_type(self) -> IntentType | None:
        """The intent's :class:`~core.types.IntentType`, or None if unknown."""
        raw = getattr(self.intent, "intent_type", None)
        if raw is None:
            raw = getattr(self.intent, "kind", None)
        if raw is None:
            return None
        try:
            return IntentType(int(raw))
        except (TypeError, ValueError):
            return None

    @property
    def intent_code(self) -> int:
        """Intent as the int stored in ``gt_intent``; -1 when unknown (sentinel)."""
        value = self.intent_type
        return -1 if value is None else int(value)

    @property
    def type_code(self) -> int:
        """Platform as the int stored in ``gt_type`` [-]."""
        return int(self.platform)

    @property
    def position(self) -> tuple[float, float]:
        """Current position ``(x, y)`` [m]."""
        return self.state.x, self.state.y

    def ground_velocity(self, wind_xy: Sequence[float]) -> tuple[float, float]:
        """Ground velocity ``(vx, vy)`` [m/s] under the given wind.

        Unpowered bodies carry a ground speed in ``state.v`` already, so their
        susceptibility is passed as 0.0 here to avoid counting the wind twice;
        the passive integrator is what applies the wind to them.

        Args:
            wind_xy: ambient wind ``(wx, wy)`` [m/s] in world axes.

        Returns:
            ``(vx, vy)`` [m/s], +X East, +Y North.
        """
        susceptibility = (
            self.envelope.wind_susceptibility if self.envelope.is_powered else 0.0
        )
        return self.state.ground_velocity(wind_xy, susceptibility)

    def distance_to(self, other: "Agent | AgentState") -> float:
        """Horizontal distance [m] to another agent or state."""
        s = other.state if isinstance(other, Agent) else other
        dx = self.state.x - s.x
        dy = self.state.y - s.y
        return float(np.hypot(dx, dy))

    # -- simulation -------------------------------------------------------

    def step(
        self,
        t: float,
        dt: float,
        wind_xy: Sequence[float],
        ownship_state: AgentState | None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Advance the agent by one integration step (doc 3.3).

        Order of operations:
            1. the intent proposes a command, or ``None`` for PASSIVE_DRIFT;
            2. the mode machine may override the proposal while a discrete
               manoeuvre is latched, and ticks its own dwell timer;
            3. the command is clipped to the envelope -- doc 3.3 (a) -- and
               integrated by ``step_powered`` or ``step_passive`` depending on
               ``envelope.is_powered``.

        When the intent returns ``None`` the mode machine is bypassed entirely:
        such an agent is not making decisions, so there is nothing to modulate
        and the passive integrator owns its motion.

        Args:
            t: current simulation time [s].
            dt: integration timestep [s], normally ``core.constants.SIM_DT``.
            wind_xy: ambient wind ``(wx, wy)`` [m/s] in world axes.
            ownship_state: the ownship's state, for interaction-aware intents
                (REACTIVE_AVOID). May be ``None`` when stepping the ownship
                itself or when no ownship exists.
            rng: generator for this step's stochastic decisions. Defaults to
                :attr:`rng`. Passing the shared ``mode_decision`` stream is also
                valid -- branch rollouts reseed either path correctly.
        """
        if not self.alive:
            return

        gen = rng if rng is not None else self.rng

        # 1. intent proposal ------------------------------------------------
        proposal: Any = None
        if self.intent is not None:
            proposal = self.intent.command(self, t, ownship_state, gen)

        # ``None`` and ``(None, None)`` are the SAME sentinel: "this agent is
        # not commanded". agents/intent.py returns the pair form (its
        # ``_command`` hook always yields a 2-tuple, so PASSIVE_DRIFT yields
        # ``(None, None)``), while a bare ``None`` is what a controller that
        # declines entirely returns. Accepting only the bare form used to raise
        # ``TypeError: float() argument ... not 'NoneType'`` on the first step
        # of every DEBRIS / PASSIVE_DRIFT agent.
        if proposal is None or proposal[0] is None or proposal[1] is None:
            omega_cmd = 0.0
            a_cmd = 0.0
            decision_active = False
        else:
            omega_cmd = float(proposal[0])
            a_cmd = float(proposal[1])
            decision_active = True

        # 2. mode-machine modulation / override ------------------------------
        if decision_active and self.maneuver is not None:
            omega_cmd, a_cmd = self._run_mode_machine(t, dt, omega_cmd, a_cmd, gen)

        # 2b. reaction layer (rev.3): a latched evasion blends into the
        # command stream AFTER the mode machine, so a committed avoidance
        # manoeuvre cannot be overridden by a routine mode change. The engine
        # arms it via perceive() at 2 Hz; here we only advance its timers and
        # apply the blend. Unpowered agents never carry a layer.
        if decision_active and self.reaction is not None:
            self.reaction.tick(dt)
            omega_cmd, a_cmd = self.reaction.apply(omega_cmd, a_cmd, self)

        # 3. envelope clipping, then integration ----------------------------
        # The integrator clips as well; doing it here too costs nothing and
        # makes the doc 3.3 (a) guarantee hold even if a future integrator
        # forgets, so an over-eager decision rule can never leak into the data.
        omega_cmd = self.envelope.clamp_turn_rate(omega_cmd, self.state.v)
        a_cmd = self.envelope.clamp_accel(a_cmd)

        if self.envelope.is_powered:
            result = _adaptive_call(
                _integrator_func("step_powered"),
                {
                    "state": self.state,
                    "envelope": self.envelope,
                    "omega_cmd": omega_cmd,
                    "a_cmd": a_cmd,
                    "dt": float(dt),
                    "wind_xy": wind_xy,
                },
                _POWERED_ORDER,
                _INTEGRATOR_ALIASES,
            )
        else:
            result = _adaptive_call(
                _integrator_func("step_passive"),
                {
                    "state": self.state,
                    "envelope": self.envelope,
                    "dt": float(dt),
                    "wind_xy": wind_xy,
                },
                _PASSIVE_ORDER,
                _INTEGRATOR_ALIASES,
            )

        if isinstance(result, AgentState):
            self.state = result
        # A ``None`` return means the integrator mutated ``self.state`` in place.

        self.state.psi = wrap_angle(self.state.psi)

    def _run_mode_machine(
        self,
        t: float,
        dt: float,
        omega_cmd: float,
        a_cmd: float,
        rng: np.random.Generator | None,
    ) -> tuple[float, float]:
        """Pass the intent's command through the maneuver mode machine.

        Args:
            t: current simulation time [s].
            dt: integration timestep [s].
            omega_cmd: turn rate proposed by the intent [rad/s].
            a_cmd: longitudinal acceleration proposed by the intent [m/s^2].
            rng: generator for the mode-switch draws.

        Returns:
            The possibly overridden ``(omega_cmd, a_cmd)`` in SI units.

        Raises:
            AttributeError: if the attached mode machine exposes none of
                :data:`_MACHINE_METHODS`.
        """
        method = None
        for name in _MACHINE_METHODS:
            candidate = getattr(self.maneuver, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None:
            raise AttributeError(
                "agents.base: the attached maneuver mode machine "
                f"({type(self.maneuver).__name__}) exposes none of "
                f"{_MACHINE_METHODS}."
            )
        result = _adaptive_call(
            method,
            {
                "agent": self,
                "t": float(t),
                "dt": float(dt),
                "omega_cmd": omega_cmd,
                "a_cmd": a_cmd,
                "rng": rng,
            },
            _MACHINE_ORDER,
            _MACHINE_ALIASES,
        )
        return _interpret_machine_result(result, omega_cmd, a_cmd)

    # -- snapshot / restore (doc 3.3 branch rollouts) ---------------------

    def snapshot(self) -> AgentSnapshot:
        """Capture everything that determines this agent's future evolution.

        Captured: kinematic state, maneuver mode + dwell timer + mode payload
        (plus any extra machine bookkeeping under
        :data:`MACHINE_STATE_KEY`), all intent bookkeeping (plus the agent-level
        flags under :data:`AGENT_STATE_KEY`), and the private RNG bit-generator
        state.

        NOT captured, because they are immutable for the lifetime of an agent:
        the performance envelope, the platform class and the behaviour
        parameters. If a future change makes any of those mutable, it MUST be
        added here or branch rollouts will start from different situations and
        the calibration ground truth of doc 3.3 becomes silently wrong.

        Returns:
            A JSON-safe :class:`~core.state.AgentSnapshot`.
        """
        mode = self.mode
        mode_timer = float(getattr(self.maneuver, "mode_timer", 0.0) or 0.0)
        mode_params = _coerce_dict(getattr(self.maneuver, "mode_params", {}) or {})

        machine_extra = _capture_component(
            self.maneuver, exclude=("mode", "mode_timer", "mode_params")
        )
        machine_extra.pop(MACHINE_STATE_KEY, None)
        if machine_extra:
            mode_params[MACHINE_STATE_KEY] = machine_extra

        intent_state = _capture_component(self.intent)
        intent_state[AGENT_STATE_KEY] = {
            "agent_id": int(self.agent_id),
            "platform": int(self.platform),
            "alive": bool(self.alive),
            "spawn_t": float(self.spawn_t),
            "is_ood": bool(self.is_ood),
        }
        if self.reaction is not None:
            # rev.3: latch/delay/cooldown timers are future-determining state;
            # omitting them would silently corrupt branch rollouts (doc 3.3).
            intent_state[REACTION_STATE_KEY] = self.reaction.snapshot()

        rng_state: dict[str, Any] = {}
        if self.rng is not None:
            rng_state = copy.deepcopy(self.rng.bit_generator.state)

        return AgentSnapshot(
            state=self.state.copy(),
            mode=mode,
            mode_timer=mode_timer,
            mode_params=mode_params,
            intent_state=intent_state,
            rng_state=rng_state,
        )

    def restore(self, snap: AgentSnapshot) -> None:
        """Restore this agent from a :meth:`snapshot` result.

        The agent must be the same object family the snapshot came from (same
        platform, same intent class, same mode-machine class); a snapshot does
        not carry constructors, only state.

        Args:
            snap: the snapshot to resume from.
        """
        self.state = snap.state.copy()
        self.state.psi = wrap_angle(self.state.psi)

        mode_params = dict(snap.mode_params)
        machine_extra = mode_params.pop(MACHINE_STATE_KEY, None)
        if self.maneuver is not None:
            if isinstance(machine_extra, Mapping):
                _restore_component(self.maneuver, machine_extra)
            for attr, value in (
                ("mode", ManeuverMode(int(snap.mode))),
                ("mode_timer", float(snap.mode_timer)),
                ("mode_params", copy.deepcopy(mode_params)),
            ):
                if hasattr(self.maneuver, attr):
                    try:
                        setattr(self.maneuver, attr, value)
                    except AttributeError:
                        continue

        intent_state = dict(snap.intent_state)
        agent_flags = intent_state.pop(AGENT_STATE_KEY, None)
        reaction_state = intent_state.pop(REACTION_STATE_KEY, None)
        if self.reaction is not None and isinstance(reaction_state, Mapping):
            self.reaction.restore(reaction_state)
        _restore_component(self.intent, intent_state)
        if isinstance(agent_flags, Mapping):
            self.alive = bool(agent_flags.get("alive", self.alive))
            self.spawn_t = float(agent_flags.get("spawn_t", self.spawn_t))
            self.is_ood = bool(agent_flags.get("is_ood", self.is_ood))

        if snap.rng_state:
            if self.rng is None:
                self.rng = np.random.default_rng()
            self.rng.bit_generator.state = copy.deepcopy(snap.rng_state)

    # -- misc -------------------------------------------------------------

    def retire(self) -> None:
        """Mark the agent as no longer participating (left the region, etc.)."""
        self.alive = False

    def describe(self) -> str:
        """One-line human-readable summary, for GUI tooltips and debugging."""
        intent = self.intent_type
        intent_name = "?" if intent is None else intent.name
        return (
            f"#{self.agent_id} {PlatformType(int(self.platform)).name} "
            f"intent={intent_name} mode={self.mode.name} "
            f"v={self.state.v:.1f} m/s psi={np.degrees(self.state.psi):.0f} deg "
            f"@({self.state.x:.0f}, {self.state.y:.0f}) m"
            f"{' OOD' if self.is_ood else ''}"
        )
