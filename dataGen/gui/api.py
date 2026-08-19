"""JSON API layer for the PRISM dataGen GUI.

This module contains every endpoint implementation. :mod:`gui.server` owns only
the HTTP plumbing (routing, static files, Server-Sent Events); everything that
knows about the simulator lives here.

Design constraint
-----------------
``gui`` is written in parallel with ``simulation``, ``sensors``, ``storage`` and
``analysis``. Therefore every dependency on those packages goes through the
small *bridge* section below, which

* imports the module lazily (so the GUI starts even when it is missing),
* probes a short list of plausible symbol names,
* reports what it found through ``GET /api/meta`` -> ``bridge`` so the operator
  can see at a glance which subsystem is wired up,
* raises :class:`ApiError` with an actionable message when a required symbol is
  genuinely absent, instead of a 500 traceback.

Preferred integration hook
--------------------------
If ``simulation.engine``'s engine object implements

    def gui_frame(self) -> dict

returning the frame dictionary documented in :func:`frame_schema_doc`, the GUI
uses it verbatim and every generic-extraction heuristic is bypassed. That is
the cheapest way for the simulation author to make the visualisation exact.

Units: SI throughout (metres, seconds, radians, m/s). Angles follow the global
convention -- ``psi`` is CCW from +X (East), wrapped to (-pi, pi].
"""

from __future__ import annotations

import glob
import inspect
import json
import math
import os
import queue
import re
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from core.constants import (
    DWC_HMD_M,
    DWC_TAU_MOD_S,
    EPS_V,
    G,
    HIST_STEPS,
    LOG_DECIMATION,
    LOG_DT,
    LOGS_ROOT,
    NMAC_RADIUS_M,
    PRED_STEPS,
    SIM_DT,
)
from core.state import AgentState, wrap_angle
from core.types import (
    GEOMETRY_NAMES,
    INTENT_NAMES,
    MODE_NAMES,
    PLATFORM_COLORS,
    RISK_NAMES,
    TYPE_NAMES,
    GeometryType,
    IntentType,
    ManeuverMode,
    PlatformType,
    RiskLevel,
)
from scenario.config import ScenarioConfig
from storage.schema import EpisodeArrays, EpisodeMeta

__all__ = [
    "ApiError",
    "Api",
    "SimSession",
    "Job",
    "APP_ROOT",
    "STATIC_ROOT",
    "frame_schema_doc",
    "to_jsonable",
]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: ``$PRISM_ROOT/dataGen`` -- the package root that ``main.py`` puts on sys.path.
APP_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(
    "\\", "/"
)

#: Directory served at ``/``.
STATIC_ROOT: str = APP_ROOT + "/gui/static"

#: Where presets written as JSON files live.
PRESET_DIR: str = APP_ROOT + "/config/presets"

#: ``config/defaults.json``.
DEFAULTS_PATH: str = APP_ROOT + "/config/defaults.json"

#: Only these characters may appear in a run directory name coming from a URL.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


# --------------------------------------------------------------------------
# Errors and JSON sanitising
# --------------------------------------------------------------------------


class ApiError(Exception):
    """An error that should be returned as a clean JSON envelope.

    Attributes:
        status: HTTP status code [-].
        code: short machine-readable slug.
        message: human-readable one-liner.
        detail: optional extra payload (traceback tail, missing symbol names).
    """

    def __init__(
        self,
        message: str,
        status: int = 400,
        code: str = "bad_request",
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = str(code)
        self.message = str(message)
        self.detail = detail

    def envelope(self) -> dict[str, Any]:
        """Return the JSON body for this error."""
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "detail": to_jsonable(self.detail),
            },
        }


def to_jsonable(obj: Any) -> Any:
    """Convert an arbitrary object into something ``json.dumps`` accepts.

    NaN and +/-Inf become ``None``: ``JSON.parse`` in the browser rejects the
    ``NaN`` literal that Python's ``json`` would otherwise emit, and the GUI
    relies on ``null`` to mean "no measurement" (doc 3.5 -- a gap stays a gap).

    Args:
        obj: any value.

    Returns:
        A structure made only of dict/list/str/int/float/bool/None.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return to_jsonable(asdict(obj))
        except Exception:  # dataclasses holding ndarrays
            return to_jsonable({f_.name: getattr(obj, f_.name)
                                for f_ in obj.__dataclass_fields__.values()})
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())
    return str(obj)


def frame_schema_doc() -> dict[str, Any]:
    """Return a machine-readable description of the GUI frame dictionary.

    Exposed at ``GET /api/meta`` so the simulation author can implement
    ``Engine.gui_frame()`` against an unambiguous target.
    """
    return {
        "t": "float [s] -- simulated time since episode start",
        "step": "int [-] -- logged step index",
        "own": {
            "x": "float [m] East", "y": "float [m] North",
            "vx": "float [m/s] ground", "vy": "float [m/s] ground",
            "v": "float [m/s] airspeed", "psi": "float [rad] CCW from +X",
            "omega": "float [rad/s] current turn rate",
        },
        "wind": {"wx": "float [m/s]", "wy": "float [m/s]",
                 "speed": "float [m/s]", "dir": "float [rad] toward"},
        "agents": [{
            "id": "int [-] stable ground-truth agent id",
            "type": "int -- PlatformType", "intent": "int -- IntentType",
            "mode": "int -- ManeuverMode", "mode_timer": "float [s] dwell left",
            "x": "float [m]", "y": "float [m]",
            "vx": "float [m/s] ground", "vy": "float [m/s] ground",
            "v": "float [m/s] airspeed", "psi": "float [rad]",
            "omega": "float [rad/s]", "omega_max": "float [rad/s] current cap",
            "det": "bool -- obs_detected",
            "ox": "float [m] or null -- measured position",
            "oy": "float [m] or null",
            "R": "[[m^2,m^2],[m^2,m^2]] or null -- obs covariance",
            "track_id": "int, -1 when no live track",
            "track_age": "int [logged steps]",
        }],
        "static": {
            "bounds": "[xmin, ymin, xmax, ymax] in m",
            "corridors": [{"x1": "m", "y1": "m", "x2": "m", "y2": "m",
                           "half_width": "m"}],
            "nfz": [{"x": "m", "y": "m", "r": "m"}],
        },
    }


# ==========================================================================
# Bridge to the modules written in parallel
# ==========================================================================


class _Bridge:
    """Lazy, name-probing importer for the simulator subsystems."""

    #: module -> candidate symbol names, in preference order.
    WANTED: dict[str, tuple[str, ...]] = {
        "simulation.engine": ("SimulationEngine", "Engine", "SimEngine", "Simulator"),
        "simulation.episode": (
            "run_episode", "generate_episode", "simulate_episode", "make_episode",
            "EpisodeRunner",
        ),
        "simulation.rollout": (
            "branch_rollout", "rollout_branches", "branch_fan", "rollout_fan",
            "run_branches", "rollout",
        ),
        "storage.writer": (
            "DatasetWriter", "RunWriter", "EpisodeWriter", "write_run",
            "write_episode",
        ),
        "storage.manifest": (
            "read_manifest", "load_manifest", "Manifest", "RunManifest",
        ),
        "analysis.cv_baseline": (
            "run_diagnostic", "diagnose_run", "cv_diagnosis",
            "cv_baseline_report", "run_diagnosis",
        ),
        "scenario.presets": (
            "load_preset", "make_preset", "PRESETS", "list_presets",
            "get_preset", "build_preset", "preset",
        ),
    }

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def module(self, name: str) -> Optional[Any]:
        """Import ``name`` once; return ``None`` if it is absent or broken."""
        with self._lock:
            if name in self._cache:
                return self._cache[name]
        mod: Optional[Any]
        try:
            import importlib

            mod = importlib.import_module(name)
        except Exception:
            mod = None
        with self._lock:
            self._cache[name] = mod
        return mod

    def symbol(self, module: str, *candidates: str) -> Optional[Any]:
        """Return the first attribute of ``module`` present in ``candidates``."""
        mod = self.module(module)
        if mod is None:
            return None
        names = candidates or self.WANTED.get(module, ())
        for cand in names:
            obj = getattr(mod, cand, None)
            if obj is not None:
                return obj
        return None

    def require(self, module: str, *candidates: str) -> Any:
        """Like :meth:`symbol` but raise a helpful :class:`ApiError`."""
        obj = self.symbol(module, *candidates)
        if obj is None:
            wanted = list(candidates or self.WANTED.get(module, ()))
            raise ApiError(
                f"module '{module}' does not provide any of {wanted}",
                status=503,
                code="subsystem_missing",
                detail={"module": module, "wanted": wanted,
                        "imported": self.module(module) is not None},
            )
        return obj

    def status(self) -> dict[str, Any]:
        """Report, per module, whether it imports and which symbol was found."""
        out: dict[str, Any] = {}
        for mod_name, cands in self.WANTED.items():
            mod = self.module(mod_name)
            found = None
            if mod is not None:
                for c in cands:
                    if getattr(mod, c, None) is not None:
                        found = c
                        break
            out[mod_name] = {"imported": mod is not None, "symbol": found}
        return out


BRIDGE = _Bridge()


def _first_attr(obj: Any, names: Iterable[str]) -> Any:
    """Return the first existing, non-None attribute of ``obj``."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _call_flex(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature can accept.

    Two filters are applied, and the second one matters: a keyword whose name
    is ALREADY bound by one of the positional arguments is dropped, because
    passing it twice raises ``TypeError: got multiple values for argument``.
    That is exactly what happens when a sibling module names its first
    parameter ``run_dir`` or ``path`` -- the very names the GUI offers as
    keyword hints.

    Args:
        fn: callable to invoke.
        *args: positional arguments, passed through unchanged.
        **kwargs: candidate keyword arguments; names the callable cannot accept
            (or that the positional arguments already bind) are dropped.

    Returns:
        Whatever ``fn`` returns.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    params = sig.parameters

    # names that the positional arguments will bind
    bound: set[str] = set()
    n = 0
    for p in params.values():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            n = len(args)
            break
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if n < len(args):
                bound.add(p.name)
                n += 1

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        accepted = {k: v for k, v in kwargs.items() if k not in bound}
    else:
        accepted = {k: v for k, v in kwargs.items()
                    if k in params and k not in bound
                    and params[k].kind is not inspect.Parameter.POSITIONAL_ONLY}
    return fn(*args, **accepted)


def _bind_call(fn: Callable[..., Any], aliases: Mapping[str, Any],
               pool: list[Any]) -> Any:
    """Call ``fn`` binding its parameters by NAME, never by guessed position.

    This is how the GUI survives not knowing the exact signature a sibling
    module settled on. Parameters whose name appears in ``aliases`` receive
    that value; remaining required parameters are filled, in declaration order,
    from ``pool``. Optional parameters are left at their defaults, so a writer
    that grew a new keyword argument still works.

    Args:
        fn: the callable to invoke.
        aliases: parameter-name -> value.
        pool: ordered fallback values for required parameters not in
            ``aliases``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        ApiError: if a required parameter cannot be supplied.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*pool)
    remaining = list(pool)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if p.name in aliases:
            value = aliases[p.name]
        elif p.default is not inspect.Parameter.empty:
            continue
        elif remaining:
            value = remaining.pop(0)
        else:
            raise ApiError(
                f"cannot supply required parameter '{p.name}' of "
                f"{getattr(fn, '__qualname__', fn)}",
                status=503, code="signature_mismatch",
                detail={"signature": str(sig), "known": sorted(aliases)},
            )
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[p.name] = value
    return fn(*args, **kwargs)


def _construct(cls: Any, cfg: ScenarioConfig, seed: int) -> Any:
    """Instantiate an engine-like class, trying the plausible signatures.

    Args:
        cls: the class (or factory) found in ``simulation.engine``.
        cfg: scenario configuration.
        seed: master seed [-].

    Returns:
        The constructed engine.

    Raises:
        ApiError: if every construction attempt failed; the detail carries each
            attempt's exception so the mismatch is obvious.
    """
    attempts: list[tuple[str, str]] = []
    trials: list[tuple[str, Callable[[], Any]]] = [
        ("cls(config=cfg, seed=seed)", lambda: _call_flex(cls, config=cfg, seed=seed)),
        ("cls(cfg=cfg, seed=seed)", lambda: _call_flex(cls, cfg=cfg, seed=seed)),
        ("cls(scenario=cfg, seed=seed)", lambda: _call_flex(cls, scenario=cfg, seed=seed)),
        ("cls(cfg, seed)", lambda: cls(cfg, seed)),
        ("cls(cfg)", lambda: cls(cfg)),
    ]
    for label, fn in trials:
        try:
            obj = fn()
            if obj is not None:
                return obj
        except Exception as exc:  # try the next signature
            attempts.append((label, f"{type(exc).__name__}: {exc}"))
    raise ApiError(
        "could not construct the simulation engine with any known signature",
        status=503,
        code="engine_construct_failed",
        detail={"attempts": attempts},
    )


def _engine_reset(engine: Any) -> None:
    """Reset an engine in place if it supports it."""
    for name in ("reset", "restart", "rewind"):
        fn = getattr(engine, name, None)
        if callable(fn):
            _call_flex(fn)
            return


def _engine_time(engine: Any, fallback: float) -> float:
    """Read the engine's simulated time [s], falling back to our own counter."""
    for name in ("t", "time", "sim_time", "t_s"):
        v = getattr(engine, name, None)
        if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)):
            return float(v)
        if callable(v):
            try:
                r = v()
                if isinstance(r, (int, float, np.floating)):
                    return float(r)
            except Exception:
                pass
    return float(fallback)


def _engine_step_logged(engine: Any) -> None:
    """Advance the engine by exactly one logged step (``LOG_DT`` seconds).

    Preference order:
        1. ``step_logged()`` / ``step_log()`` / ``log_step()`` -- explicit.
        2. ``step()`` called ``LOG_DECIMATION`` times (the SIM_DT convention).
        3. ``advance(dt)`` / ``update(dt)`` with ``dt = SIM_DT``.
    """
    for name in ("step_logged", "step_log", "log_step", "step_logstep"):
        fn = getattr(engine, name, None)
        if callable(fn):
            fn()
            return
    fn = getattr(engine, "step", None)
    if callable(fn):
        try:
            sig = inspect.signature(fn)
            takes_dt = any(
                p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect.Parameter.empty
                for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            takes_dt = False
        for _ in range(LOG_DECIMATION):
            fn(SIM_DT) if takes_dt else fn()
        return
    for name in ("advance", "update", "tick"):
        fn = getattr(engine, name, None)
        if callable(fn):
            for _ in range(LOG_DECIMATION):
                _call_flex(fn, SIM_DT, dt=SIM_DT)
            return
    raise ApiError(
        "the simulation engine exposes no step()/advance() method",
        status=503,
        code="engine_no_step",
    )


def _state_tuple(st: Any) -> tuple[float, float, float, float, float, float]:
    """Return ``(x, y, v, psi, omega, a)`` from an AgentState-like object."""
    return (
        float(getattr(st, "x", 0.0)),
        float(getattr(st, "y", 0.0)),
        float(getattr(st, "v", 0.0)),
        float(getattr(st, "psi", 0.0)),
        float(getattr(st, "omega", 0.0)),
        float(getattr(st, "a", 0.0)),
    )


def _ground_vel(st: Any, wind_xy: tuple[float, float], susc: float) -> tuple[float, float]:
    """Ground velocity [m/s] of a state, using the contract method when present."""
    fn = getattr(st, "ground_velocity", None)
    if callable(fn):
        try:
            vx, vy = fn(wind_xy, susc)
            return float(vx), float(vy)
        except Exception:
            pass
    v = float(getattr(st, "v", 0.0))
    psi = float(getattr(st, "psi", 0.0))
    return v * math.cos(psi) + wind_xy[0] * susc, v * math.sin(psi) + wind_xy[1] * susc


def _engine_wind(engine: Any) -> tuple[float, float]:
    """Steady wind vector [m/s] as ``(wx, wy)``; ``(0, 0)`` when unavailable."""
    for name in ("wind_xy", "wind_vector", "wind", "steady_wind"):
        v = getattr(engine, name, None)
        if callable(v):
            try:
                v = _call_flex(v)
            except Exception:
                v = None
        if v is None:
            continue
        if isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        wx = _first_attr(v, ("wx", "x", "u"))
        wy = _first_attr(v, ("wy", "y", "v"))
        if wx is not None and wy is not None:
            return float(wx), float(wy)
        spd = _first_attr(v, ("speed", "speed_ms"))
        drc = _first_attr(v, ("dir", "direction", "dir_rad"))
        if spd is not None and drc is not None:
            return float(spd) * math.cos(float(drc)), float(spd) * math.sin(float(drc))
    return 0.0, 0.0


def _engine_static(engine: Any, cfg: ScenarioConfig) -> dict[str, Any]:
    """Extract world bounds, corridors and no-fly zones, with safe defaults."""
    half = 0.5 * cfg.region_side_m()
    out: dict[str, Any] = {
        "bounds": [-half, -half, half, half],
        "corridors": [],
        "nfz": [],
    }
    world = _first_attr(engine, ("world", "env", "environment"))
    src = world if world is not None else engine

    bounds = _first_attr(src, ("bounds", "world_bounds", "extent"))
    if callable(bounds):
        try:
            bounds = _call_flex(bounds)
        except Exception:
            bounds = None
    if isinstance(bounds, (list, tuple, np.ndarray)) and len(bounds) == 4:
        out["bounds"] = [float(b) for b in bounds]

    corridors = _first_attr(src, ("corridors", "corridor_list", "corridor"))
    if callable(corridors):
        try:
            corridors = _call_flex(corridors)
        except Exception:
            corridors = None
    if isinstance(corridors, (list, tuple)):
        for c in corridors:
            if isinstance(c, Mapping):
                d = dict(c)
            else:
                d = {
                    "x1": _first_attr(c, ("x1", "ax", "start_x")),
                    "y1": _first_attr(c, ("y1", "ay", "start_y")),
                    "x2": _first_attr(c, ("x2", "bx", "end_x")),
                    "y2": _first_attr(c, ("y2", "by", "end_y")),
                    "half_width": _first_attr(c, ("half_width", "hw", "width_half")),
                }
            if d.get("x1") is None and isinstance(c, (list, tuple)) and len(c) >= 4:
                d = {"x1": c[0], "y1": c[1], "x2": c[2], "y2": c[3],
                     "half_width": c[4] if len(c) > 4 else 150.0}
            if d.get("x1") is None:
                continue
            out["corridors"].append({
                "x1": float(d["x1"]), "y1": float(d["y1"]),
                "x2": float(d["x2"]), "y2": float(d["y2"]),
                "half_width": float(d.get("half_width") or 150.0),
            })

    nfz = _first_attr(src, ("nfz", "nfzs", "no_fly_zones", "keepout", "nfz_list"))
    if callable(nfz):
        try:
            nfz = _call_flex(nfz)
        except Exception:
            nfz = None
    if isinstance(nfz, (list, tuple)):
        for z in nfz:
            if isinstance(z, Mapping):
                d = dict(z)
            elif isinstance(z, (list, tuple, np.ndarray)) and len(z) >= 3:
                d = {"x": z[0], "y": z[1], "r": z[2]}
            else:
                d = {"x": _first_attr(z, ("x", "cx", "centre_x", "center_x")),
                     "y": _first_attr(z, ("y", "cy", "centre_y", "center_y")),
                     "r": _first_attr(z, ("r", "radius", "radius_m"))}
            if d.get("x") is None or d.get("r") is None:
                continue
            out["nfz"].append({"x": float(d["x"]), "y": float(d["y"]),
                               "r": float(d["r"])})
    return out


def _obs_lookup(engine: Any) -> dict[int, dict[str, Any]]:
    """Return ``{agent_id: observation dict}`` for the current step.

    Accepts a list/dict of mappings or of objects with the usual attribute
    names. Returns an empty mapping when the engine exposes no observations,
    in which case the GUI simply shows no OBS layer.
    """
    raw = None
    for name in ("observations", "latest_observations", "obs", "tracks",
                 "current_observations", "sensor_tracks"):
        v = getattr(engine, name, None)
        if callable(v):
            try:
                v = _call_flex(v)
            except Exception:
                v = None
        if v is not None:
            raw = v
            break
    if raw is None:
        return {}
    items: list[Any]
    if isinstance(raw, Mapping):
        items = []
        for k, v in raw.items():
            d = dict(v) if isinstance(v, Mapping) else {"_obj": v}
            d.setdefault("agent_id", k)
            items.append(d)
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return {}

    out: dict[int, dict[str, Any]] = {}
    for it in items:
        if isinstance(it, Mapping):
            get = lambda names, m=it: next(
                (m[n] for n in names if n in m and m[n] is not None), None
            )
        else:
            get = lambda names, o=it: _first_attr(o, names)
        aid = get(("agent_id", "id", "gt_id", "truth_id"))
        if aid is None:
            continue
        det = get(("detected", "det", "is_detected"))
        pos = get(("pos", "xy", "position"))
        ox = get(("x_obs", "ox", "x"))
        oy = get(("y_obs", "oy", "y"))
        if pos is not None and (ox is None or oy is None):
            try:
                ox, oy = float(pos[0]), float(pos[1])
            except Exception:
                ox = oy = None
        R = get(("R", "R_cov", "cov", "covariance"))
        tid = get(("track_id", "tid"))
        age = get(("track_age", "age"))
        rec: dict[str, Any] = {
            "det": bool(det) if det is not None else (ox is not None),
            "ox": None if ox is None else float(ox),
            "oy": None if oy is None else float(oy),
            "R": None,
            "track_id": int(tid) if tid is not None else -1,
            "track_age": int(age) if age is not None else 0,
        }
        if R is not None:
            try:
                Rm = np.asarray(R, dtype=np.float64).reshape(2, 2)
                if np.all(np.isfinite(Rm)):
                    rec["R"] = [[float(Rm[0, 0]), float(Rm[0, 1])],
                                [float(Rm[1, 0]), float(Rm[1, 1])]]
            except Exception:
                rec["R"] = None
        if rec["ox"] is None or rec["oy"] is None:
            rec["det"] = False
        out[int(aid)] = rec
    return out


def _snapshot_of(engine: Any) -> Any:
    """Return the engine's :class:`core.state.WorldSnapshot`-like object."""
    for name in ("snapshot", "world_snapshot", "get_snapshot", "state", "world_state"):
        v = getattr(engine, name, None)
        if callable(v):
            try:
                r = _call_flex(v)
            except Exception:
                continue
            if r is not None and hasattr(r, "agents"):
                return r
        elif v is not None and hasattr(v, "agents"):
            return v
    return None


def _normalize_statics(raw: Any, cfg: ScenarioConfig) -> dict[str, Any]:
    """Normalise an engine static-context dict into the GUI's shape.

    Accepts corridors given either as ``{'p0':[x,y],'p1':[x,y],'half_width':w}``
    (the ``simulation.engine`` form) or as ``{'x1','y1','x2','y2'}``, and
    no-fly zones given as ``{'center':[x,y],'radius':r}`` or ``{'x','y','r'}``.

    Args:
        raw: the engine's statics mapping.
        cfg: scenario config, used for the default square bounds.

    Returns:
        ``{'bounds', 'corridors', 'nfz'}`` in GUI units [m].
    """
    half = 0.5 * cfg.region_side_m()
    out: dict[str, Any] = {"bounds": [-half, -half, half, half],
                           "corridors": [], "nfz": []}
    if not isinstance(raw, Mapping):
        return out
    b = raw.get("bounds")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        out["bounds"] = [float(v) for v in b]
    for c in (raw.get("corridors") or []):
        if not isinstance(c, Mapping):
            continue
        p0, p1 = c.get("p0"), c.get("p1")
        if p0 is not None and p1 is not None:
            x1, y1 = float(p0[0]), float(p0[1])
            x2, y2 = float(p1[0]), float(p1[1])
        elif c.get("x1") is not None:
            x1, y1 = float(c["x1"]), float(c["y1"])
            x2, y2 = float(c["x2"]), float(c["y2"])
        else:
            continue
        hw = c.get("half_width", c.get("hw", 150.0))
        out["corridors"].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "half_width": float(hw)})
    for z in (raw.get("nfz") or raw.get("nfzs") or []):
        if not isinstance(z, Mapping):
            continue
        ctr = z.get("center", z.get("centre"))
        if ctr is not None:
            x, y = float(ctr[0]), float(ctr[1])
        elif z.get("x") is not None:
            x, y = float(z["x"]), float(z["y"])
        else:
            continue
        r = z.get("radius", z.get("r"))
        if r is None:
            continue
        out["nfz"].append({"x": x, "y": y, "r": float(r)})
    return out


def _merge_logged_row(engine: Any, by_id: dict[int, dict[str, Any]],
                      own: dict[str, Any]) -> None:
    """Fill the observation layer from the engine's live episode buffer.

    ``simulation.engine.SimulationEngine`` fills a
    :class:`storage.schema.EpisodeArrays` as it runs, and that buffer is the
    only place the measured position and its 2x2 covariance exist. Reading the
    most recent logged row gives the GUI exactly the observation layer the
    dataset will contain -- which is the point of the OBS overlay: the
    range-elongated 1-sigma ellipses are what make the sensor model legible.

    Ground velocities are taken from the same row, so the drawn velocity
    vectors include wind advection instead of being airspeed vectors.

    Args:
        engine: the simulation engine.
        by_id: agent records keyed by ground-truth agent id, modified in place.
        own: the ownship record, modified in place.
    """
    ep = getattr(engine, "ep", None)
    if not isinstance(ep, EpisodeArrays):
        return
    try:
        k = int(getattr(engine, "n_logged", 0)) - 1
    except Exception:
        return
    if k < 0 or k >= ep.n_steps:
        return
    try:
        if np.all(np.isfinite(ep.own_vel[k])):
            own["vx"] = float(ep.own_vel[k, 0])
            own["vy"] = float(ep.own_vel[k, 1])
        ids = ep.agent_ids
        for j in range(ep.n_agents):
            aid = int(ids[j])
            rec = by_id.get(aid)
            if rec is None:
                continue
            if bool(ep.gt_valid[k, j]):
                rec["vx"] = float(ep.gt_vel[k, j, 0])
                rec["vy"] = float(ep.gt_vel[k, j, 1])
            det = bool(ep.obs_detected[k, j])
            rec["det"] = det
            if det:
                rec["ox"] = float(ep.obs_pos[k, j, 0])
                rec["oy"] = float(ep.obs_pos[k, j, 1])
                R = ep.obs_R[k, j].astype(np.float64)
                rec["R"] = [[float(R[0, 0]), float(R[0, 1])],
                            [float(R[1, 0]), float(R[1, 1])]]
            else:
                rec["ox"] = rec["oy"] = None
                rec["R"] = None
            rec["track_id"] = int(ep.obs_track_id[k, j])
            rec["track_age"] = int(ep.obs_track_age[k, j])
    except Exception:
        return  # a partially written buffer must never break the stream


def _merge_agent_internals(engine: Any, by_id: dict[int, dict[str, Any]],
                           own: Optional[dict[str, Any]] = None) -> None:
    """Add turn rate, dwell timer and the current turn-rate cap to each agent.

    These are agent internals, not dataset fields, so no compact engine frame
    carries them -- but the inspector needs them: ``omega`` against
    ``PerformanceEnvelope.max_turn_rate(v)`` is how an operator sees whether a
    platform is saturating its performance envelope (doc 3.2 (3)), and the
    dwell timer is how the "hold a maneuver for a few seconds" constraint of
    doc 3.3 becomes visible.

    Every lookup is optional and guarded: a simulator that exposes none of
    these still streams frames, the inspector just shows ``--``.

    Args:
        engine: the simulation engine.
        by_id: agent records keyed by agent id, modified in place.
        own: the ownship record, modified in place when available.
    """
    scenario = getattr(engine, "scenario", None)
    if own is not None:
        own_obj = getattr(scenario, "ownship", None)
        st = getattr(own_obj, "state", None)
        if st is not None:
            try:
                own["omega"] = float(getattr(st, "omega", 0.0))
                own["accel"] = float(getattr(st, "a", 0.0))
            except (TypeError, ValueError):
                pass
        env = getattr(own_obj, "envelope", None)
        mtr = getattr(env, "max_turn_rate", None)
        if callable(mtr):
            try:
                own["omega_max"] = float(mtr(float(own.get("v") or 0.0)))
            except Exception:
                pass
    agents = getattr(scenario, "agents", None)
    if not isinstance(agents, (list, tuple)):
        return
    for ag in agents:
        try:
            aid = int(getattr(ag, "agent_id", -1))
        except (TypeError, ValueError):
            continue
        rec = by_id.get(aid)
        if rec is None:
            continue
        try:
            st = getattr(ag, "state", None)
            if st is not None and rec.get("omega") is None:
                rec["omega"] = float(getattr(st, "omega", 0.0))
                rec["accel"] = float(getattr(st, "a", 0.0))
            env = getattr(ag, "envelope", None)
            mtr = getattr(env, "max_turn_rate", None)
            if callable(mtr):
                rec["omega_max"] = float(mtr(float(rec.get("v") or 0.0)))
                rec["v_min"] = float(getattr(env, "v_min", float("nan")))
                rec["v_max"] = float(getattr(env, "v_max", float("nan")))
                rec["can_hover"] = bool(getattr(env, "can_hover", False))
            man = getattr(ag, "maneuver", None)
            for attr in ("timer", "mode_timer", "dwell_remaining"):
                tv = getattr(man, attr, None)
                if isinstance(tv, (int, float)):
                    rec["mode_timer"] = float(tv)
                    break
            if rec.get("mode_timer") is None:
                tv = getattr(ag, "mode_timer", None)
                if isinstance(tv, (int, float)):
                    rec["mode_timer"] = float(tv)
            rec["ood"] = bool(getattr(ag, "is_ood", rec.get("ood") or False))
        except Exception:
            continue


def _frame_from_current_frame(engine: Any, raw: Mapping[str, Any],
                              cfg: ScenarioConfig, step: int) -> dict[str, Any]:
    """Adapt ``SimulationEngine.current_frame()`` to the GUI frame schema.

    The engine's compact frame carries ground truth, detection flags and its
    own CPA solution but not the measured position or its covariance; those are
    merged in from the live episode buffer by :func:`_merge_logged_row`.

    Args:
        engine: the simulation engine (for the buffer merge).
        raw: the engine's ``current_frame()`` result.
        cfg: scenario configuration.
        step: fallback logged-step index [-].

    Returns:
        A GUI frame dictionary.
    """
    w = raw.get("wind") or {}
    wx = float(w.get("wx", 0.0))
    wy = float(w.get("wy", 0.0))
    ro = raw.get("own") or {}
    psi_o = float(ro.get("psi", 0.0))
    v_o = float(ro.get("v", 0.0))
    own = {
        "x": float(ro.get("x", 0.0)), "y": float(ro.get("y", 0.0)),
        "v": v_o, "psi": wrap_angle(psi_o),
        "vx": float(ro.get("vx", v_o * math.cos(psi_o))),
        "vy": float(ro.get("vy", v_o * math.sin(psi_o))),
        "omega": ro.get("omega"),
        "type": ro.get("type"),
    }

    agents: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for a in (raw.get("agents") or []):
        if not isinstance(a, Mapping):
            continue
        if a.get("alive") is False:
            continue
        psi = float(a.get("psi", 0.0))
        v = float(a.get("v", 0.0))
        rec = {
            "id": int(a.get("id", -1)),
            "type": int(a.get("type", int(PlatformType.UNKNOWN))),
            "intent": int(a.get("intent", -1)),
            "mode": int(a.get("mode", -1)),
            "mode_timer": a.get("mode_timer"),
            "x": float(a.get("x", 0.0)), "y": float(a.get("y", 0.0)),
            "v": v, "psi": wrap_angle(psi),
            "vx": float(a.get("vx", v * math.cos(psi))),
            "vy": float(a.get("vy", v * math.sin(psi))),
            "omega": a.get("omega"), "omega_max": a.get("omega_max"),
            "det": bool(a.get("det", False)),
            "ox": a.get("ox"), "oy": a.get("oy"), "R": a.get("R"),
            "track_id": int(a.get("tid", a.get("track_id", -1))),
            "track_age": int(a.get("age", a.get("track_age", 0))),
            "engine_cpa_m": a.get("cpa_d"), "engine_tcpa_s": a.get("cpa_t"),
            "engine_range_m": a.get("rng_m"),
            "designed": a.get("designed"), "ood": a.get("ood"),
        }
        agents.append(rec)
        by_id[rec["id"]] = rec

    _merge_agent_internals(engine, by_id, own)
    _merge_logged_row(engine, by_id, own)

    return to_jsonable({
        "t": float(raw.get("t", step * LOG_DT)),
        "step": int(raw.get("log_index", step)),
        "n_steps": int(raw.get("n_steps", 0)) or None,
        "own": own,
        "wind": {"wx": wx, "wy": wy,
                 "speed": float(w.get("speed_ms", math.hypot(wx, wy))),
                 "dir": float(w.get("dir_rad", math.atan2(wy, wx)))},
        "agents": agents,
        "static": _normalize_statics(raw.get("statics"), cfg),
    })


def _extract_frame(engine: Any, cfg: ScenarioConfig, step: int,
                   fallback_t: float) -> dict[str, Any]:
    """Build a GUI frame dict from an engine, honouring ``gui_frame()`` first.

    Args:
        engine: the simulation engine.
        cfg: scenario config (used only for default world bounds).
        step: logged step index [-].
        fallback_t: time [s] to use if the engine exposes none.

    Returns:
        The frame dictionary described by :func:`frame_schema_doc`.
    """
    hook = getattr(engine, "gui_frame", None)
    if callable(hook):
        fr = dict(_call_flex(hook))
        fr.setdefault("step", step)
        fr.setdefault("t", fallback_t)
        fr.setdefault("static", _engine_static(engine, cfg))
        return to_jsonable(fr)

    cur = getattr(engine, "current_frame", None)
    if callable(cur):
        try:
            raw = _call_flex(cur, include_statics=True)
        except Exception:
            raw = None
        if isinstance(raw, Mapping) and "agents" in raw:
            return _frame_from_current_frame(engine, raw, cfg, step)

    snap = _snapshot_of(engine)
    wx, wy = _engine_wind(engine)
    t = _engine_time(engine, fallback_t)
    if snap is not None:
        t = float(getattr(snap, "t", t))

    # ownship ------------------------------------------------------------
    own_state = None
    if snap is not None:
        own_snap = getattr(snap, "ownship", None)
        own_state = getattr(own_snap, "state", own_snap)
    if own_state is None:
        own_state = _first_attr(engine, ("ownship_state", "own_state"))
        own_obj = _first_attr(engine, ("ownship", "own"))
        if own_state is None and own_obj is not None:
            own_state = getattr(own_obj, "state", own_obj)
    if own_state is None:
        own_state = AgentState(0.0, 0.0, float(cfg.ownship_speed_ms), 0.0)
    ox_, oy_, ov, opsi, oomega, _oa = _state_tuple(own_state)
    ovx, ovy = _ground_vel(own_state, (wx, wy), 1.0)

    # agents -------------------------------------------------------------
    agent_snaps: list[Any] = []
    agent_ids: list[int] = []
    if snap is not None:
        agent_snaps = list(getattr(snap, "agents", []) or [])
        agent_ids = [int(i) for i in (getattr(snap, "agent_ids", []) or [])]
    if not agent_snaps:
        lst = _first_attr(engine, ("agents", "traffic", "intruders"))
        if isinstance(lst, (list, tuple)):
            agent_snaps = list(lst)
    if len(agent_ids) != len(agent_snaps):
        agent_ids = list(range(len(agent_snaps)))

    types = _first_attr(engine, ("platform_types", "agent_types", "types"))
    if callable(types):
        try:
            types = _call_flex(types)
        except Exception:
            types = None
    envelopes = _first_attr(engine, ("envelopes", "performance", "perf_envelopes"))
    if callable(envelopes):
        try:
            envelopes = _call_flex(envelopes)
        except Exception:
            envelopes = None

    obs = _obs_lookup(engine)
    agents: list[dict[str, Any]] = []
    for i, asn in enumerate(agent_snaps):
        st = getattr(asn, "state", asn)
        x, y, v, psi, omega, _a = _state_tuple(st)
        aid = int(agent_ids[i]) if i < len(agent_ids) else i

        env = None
        if isinstance(envelopes, (list, tuple)) and i < len(envelopes):
            env = envelopes[i]
        elif isinstance(envelopes, Mapping):
            env = envelopes.get(aid)
        susc = 1.0
        omega_max = None
        if env is not None:
            susc = float(getattr(env, "wind_susceptibility", 1.0))
            mtr = getattr(env, "max_turn_rate", None)
            if callable(mtr):
                try:
                    omega_max = float(mtr(v))
                except Exception:
                    omega_max = None
        vx, vy = _ground_vel(st, (wx, wy), susc)

        ptype = None
        if isinstance(types, (list, tuple, np.ndarray)) and i < len(types):
            ptype = int(types[i])
        elif isinstance(types, Mapping) and aid in types:
            ptype = int(types[aid])
        else:
            istate = getattr(asn, "intent_state", None)
            if isinstance(istate, Mapping):
                for key in ("platform_type", "type", "ptype"):
                    if key in istate:
                        ptype = int(istate[key])
                        break
            if ptype is None:
                pv = _first_attr(asn, ("platform_type", "ptype", "type"))
                if pv is not None:
                    ptype = int(pv)
        if ptype is None:
            ptype = int(PlatformType.UNKNOWN)

        intent = None
        istate = getattr(asn, "intent_state", None)
        if isinstance(istate, Mapping):
            for key in ("intent", "intent_type", "type"):
                if key in istate:
                    try:
                        intent = int(istate[key])
                        break
                    except Exception:
                        pass
        if intent is None:
            iv = _first_attr(asn, ("intent", "intent_type"))
            if iv is not None:
                try:
                    intent = int(iv)
                except Exception:
                    intent = None
        if intent is None:
            intent = -1

        mode = getattr(asn, "mode", None)
        mode_i = int(mode) if mode is not None else -1
        rec = {
            "id": aid, "type": ptype, "intent": intent, "mode": mode_i,
            "mode_timer": float(getattr(asn, "mode_timer", 0.0) or 0.0),
            "x": x, "y": y, "vx": vx, "vy": vy, "v": v,
            "psi": wrap_angle(psi), "omega": omega, "omega_max": omega_max,
            "det": False, "ox": None, "oy": None, "R": None,
            "track_id": -1, "track_age": 0,
        }
        o = obs.get(aid)
        if o:
            rec.update(o)
        agents.append(rec)

    return to_jsonable({
        "t": t,
        "step": int(step),
        "own": {"x": ox_, "y": oy_, "vx": ovx, "vy": ovy, "v": ov,
                "psi": wrap_angle(opsi), "omega": oomega},
        "wind": {"wx": wx, "wy": wy, "speed": math.hypot(wx, wy),
                 "dir": math.atan2(wy, wx)},
        "agents": agents,
        "static": _engine_static(engine, cfg),
    })


# ==========================================================================
# Live simulation sessions
# ==========================================================================


class SimSession:
    """One live, interactive simulation driven by its own thread.

    The driver thread is the only place the engine is stepped, so the SSE
    stream, the transport buttons and the rollout endpoint can never race.
    Subscribers receive frames through bounded queues; a slow browser loses old
    frames instead of stalling the simulation.
    """

    def __init__(self, sim_id: str, cfg: ScenarioConfig) -> None:
        """Create and start a session.

        Args:
            sim_id: opaque identifier handed back to the client.
            cfg: the scenario configuration to simulate.
        """
        self.sim_id = sim_id
        self.cfg = cfg
        self.lock = threading.RLock()
        self.closed = False
        self.playing = False
        self.speed = 1.0
        self.pending_steps = 0
        self.reset_requested = False
        self.step_index = 0
        self.seq = 0
        self.created = time.time()
        self.last_touch = time.time()
        self.error: Optional[str] = None
        self._subs: list["queue.Queue[dict[str, Any]]"] = []

        cls = BRIDGE.require("simulation.engine")
        self.engine = _construct(cls, cfg, int(cfg.seed))
        self.n_log_steps = max(1, int(round(cfg.duration_s / LOG_DT)))
        self.frame = self._frame()
        self._thread = threading.Thread(
            target=self._run, name=f"sim-{sim_id}", daemon=True
        )
        self._thread.start()

    # -- frame plumbing ---------------------------------------------------

    def _frame(self) -> dict[str, Any]:
        """Build the current frame and stamp it with session status."""
        fr = _extract_frame(self.engine, self.cfg, self.step_index,
                            self.step_index * LOG_DT)
        self.seq += 1
        fr["seq"] = self.seq
        fr["sim_id"] = self.sim_id
        fr["playing"] = bool(self.playing)
        fr["speed"] = float(self.speed)
        fr["n_steps"] = int(self.n_log_steps)
        fr["done"] = bool(self.step_index >= self.n_log_steps)
        return fr

    def _publish(self, fr: dict[str, Any]) -> None:
        """Push a frame to every subscriber, dropping the oldest when full."""
        with self.lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(fr)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(fr)
                except Exception:
                    pass

    def subscribe(self) -> "queue.Queue[dict[str, Any]]":
        """Register a new SSE listener and prime it with the current frame."""
        q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=64)
        with self.lock:
            self._subs.append(q)
            self.last_touch = time.time()
            fr = self.frame
        try:
            q.put_nowait(fr)
        except queue.Full:
            pass
        return q

    def unsubscribe(self, q: "queue.Queue[dict[str, Any]]") -> None:
        """Remove an SSE listener."""
        with self.lock:
            if q in self._subs:
                self._subs.remove(q)
            self.last_touch = time.time()

    @property
    def n_subscribers(self) -> int:
        """Number of attached SSE listeners [-]."""
        with self.lock:
            return len(self._subs)

    # -- transport --------------------------------------------------------

    #: How long :meth:`control` waits [s] for the driver thread to publish the
    #: frame a ``step`` / ``reset`` asked for. The GUI renders from SSE and
    #: ignores the response frame, but an API client that single-steps and
    #: reads ``d.frame`` would otherwise get the frame from BEFORE its own
    #: command every time -- step #1 looked like a no-op and the final frame
    #: was never returned by any control call.
    _SETTLE_TIMEOUT_S: float = 1.0

    def control(self, action: str, speed: Optional[float] = None) -> dict[str, Any]:
        """Apply a transport command.

        For ``step`` and ``reset`` this blocks (briefly) until the driver
        thread has actually produced the resulting frame, so the returned
        ``frame`` is the state AFTER the command. On timeout the current frame
        is returned rather than raising -- a slow step must not fail the call.

        Args:
            action: one of ``play``, ``pause``, ``toggle``, ``step``, ``reset``.
            speed: playback rate multiplier [-]; ``1.0`` is real time.

        Returns:
            The frame in effect after the command took effect.
        """
        wait_for_seq = -1
        with self.lock:
            self.last_touch = time.time()
            if speed is not None and math.isfinite(float(speed)):
                self.speed = float(min(max(float(speed), 0.05), 50.0))
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
            elif action == "toggle":
                self.playing = not self.playing
            elif action == "step":
                self.playing = False
                self.pending_steps += 1
                wait_for_seq = int(self.frame.get("seq", 0))
            elif action == "reset":
                self.playing = False
                self.pending_steps = 0
                self.reset_requested = True
                wait_for_seq = int(self.frame.get("seq", 0))
            elif action == "speed":
                pass
            else:
                raise ApiError(f"unknown control action '{action}'",
                               code="bad_action")
            if wait_for_seq < 0:
                return self.frame

        deadline = time.time() + self._SETTLE_TIMEOUT_S
        while time.time() < deadline:
            with self.lock:
                fr = self.frame
                if int(fr.get("seq", 0)) != wait_for_seq or self.closed:
                    return fr
            time.sleep(0.005)
        with self.lock:
            return self.frame

    def close(self) -> None:
        """Stop the driver thread and release subscribers."""
        with self.lock:
            self.closed = True
            self._subs.clear()

    # -- driver -----------------------------------------------------------

    def _run(self) -> None:
        """Driver loop: the only writer of engine state."""
        while not self.closed:
            try:
                do_step = False
                with self.lock:
                    if self.reset_requested:
                        self.reset_requested = False
                        _engine_reset(self.engine)
                        self.step_index = 0
                        self.error = None
                        self.frame = self._frame()
                        published = self.frame
                    elif self.pending_steps > 0:
                        self.pending_steps -= 1
                        do_step = True
                        published = None
                    elif self.playing and self.step_index < self.n_log_steps:
                        do_step = True
                        published = None
                    else:
                        if self.playing and self.step_index >= self.n_log_steps:
                            self.playing = False
                            self.frame = self._frame()
                            published = self.frame
                        else:
                            published = None
                if published is not None:
                    self._publish(published)
                    continue
                if not do_step:
                    time.sleep(0.02)
                    continue
                with self.lock:
                    _engine_step_logged(self.engine)
                    self.step_index += 1
                    self.frame = self._frame()
                    fr = self.frame
                    delay = LOG_DT / max(self.speed, 0.05) if self.playing else 0.0
                self._publish(fr)
                if delay > 0.0:
                    time.sleep(min(delay, 2.0))
            except Exception as exc:  # keep the session alive but report
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.playing = False
                    fr = dict(self.frame)
                fr["error"] = self.error
                self._publish(fr)
                time.sleep(0.25)


# ==========================================================================
# Batch jobs
# ==========================================================================


@dataclass
class Job:
    """A background dataset-generation job.

    Attributes:
        job_id: opaque identifier.
        tag: user-supplied run tag.
        n_episodes: total episodes requested [-].
        done: episodes completed [-].
        state: ``queued`` | ``running`` | ``done`` | ``error`` | ``cancelled``.
        started: wall-clock start [s since epoch].
        finished: wall-clock finish [s since epoch], 0 while running.
        run_path: output directory once known.
        message: last status line.
        log: bounded tail of log lines.
        cancel: cooperative cancellation flag.
    """

    job_id: str
    tag: str
    n_episodes: int
    done: int = 0
    state: str = "queued"
    started: float = 0.0
    finished: float = 0.0
    run_path: str = ""
    message: str = ""
    log: deque = field(default_factory=lambda: deque(maxlen=400))
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, line: str) -> None:
        """Append a timestamped line to the log tail."""
        with self.lock:
            stamp = time.strftime("%H:%M:%S")
            self.log.append(f"[{stamp}] {line}")
            self.message = line

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe progress report including an ETA."""
        with self.lock:
            elapsed = (self.finished or time.time()) - self.started if self.started else 0.0
            rate = (self.done / elapsed) if (elapsed > 0 and self.done > 0) else 0.0
            remaining = max(self.n_episodes - self.done, 0)
            eta = (remaining / rate) if rate > 0 else None
            return {
                "job_id": self.job_id,
                "tag": self.tag,
                "state": self.state,
                "n_episodes": int(self.n_episodes),
                "done": int(self.done),
                "progress": (self.done / self.n_episodes) if self.n_episodes else 0.0,
                "elapsed_s": float(elapsed),
                "eta_s": eta,
                "rate_eps": rate,
                "run_path": self.run_path,
                "message": self.message,
                "log": list(self.log)[-80:],
            }


# ==========================================================================
# Run directory access (read side is deliberately layout-agnostic)
# ==========================================================================


def _safe_run_dir(run: str) -> str:
    """Resolve a run name to an absolute directory inside ``LOGS_ROOT``.

    ``'latest'`` resolves to the newest run, exactly as ``main.py --run
    latest`` and the README do. The CLI and the HTTP API document the same
    argument, so they must accept the same values.

    Args:
        run: directory name taken from a URL path segment, or ``'latest'``.

    Returns:
        Absolute path with forward slashes.

    Raises:
        ApiError: on an unsafe name or a path that escapes ``LOGS_ROOT``.
    """
    if (run or "").strip() == "latest":
        newest = _latest_run_dir()
        if not newest:
            raise ApiError(f"no runs found under {LOGS_ROOT}", status=404,
                           code="run_not_found")
        return newest
    if not _SAFE_NAME.match(run or ""):
        raise ApiError(f"illegal run name '{run}'", code="bad_run_name")
    root = os.path.abspath(LOGS_ROOT)
    path = os.path.abspath(os.path.join(root, run))
    if os.path.commonpath([root, path]) != root:
        raise ApiError("run path escapes the logs root", status=403,
                       code="path_escape")
    if not os.path.isdir(path):
        raise ApiError(f"run '{run}' not found under {LOGS_ROOT}", status=404,
                       code="run_not_found")
    return path.replace("\\", "/")


def _latest_run_dir() -> str:
    """Newest run directory under ``LOGS_ROOT``, or ``''`` when there is none.

    Delegates to :func:`storage.writer.latest_run` and falls back to an mtime
    scan so the endpoint still answers if that helper is unavailable.
    """
    fn = BRIDGE.symbol("storage.writer", "latest_run")
    if callable(fn):
        try:
            got = fn()
            if got:
                return str(got).replace("\\", "/")
        except Exception:
            pass
    root = os.path.abspath(LOGS_ROOT)
    if not os.path.isdir(root):
        return ""
    dirs = [os.path.join(root, n) for n in os.listdir(root)
            if os.path.isdir(os.path.join(root, n))]
    if not dirs:
        return ""
    return max(dirs, key=os.path.getmtime).replace("\\", "/")


def _read_json(path: str) -> Any:
    """Read a UTF-8 JSON file, returning ``None`` on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _run_manifest(run_dir: str) -> dict[str, Any]:
    """Load the run-level manifest, preferring :mod:`storage.manifest`.

    Returns:
        The manifest dict (possibly ``{}``) with an added ``source`` key.
    """
    reader = BRIDGE.symbol("storage.manifest", "read_manifest", "load_manifest")
    if callable(reader):
        try:
            m = to_jsonable(_call_flex(reader, run_dir, path=run_dir,
                                       run_dir=run_dir))
            if isinstance(m, dict) and m:
                m = dict(m)
                m["source"] = "storage.manifest.read_manifest"
                return m
        except Exception:
            pass
    for name in ("manifest.json", "index.json", "run.json", "dataset.json"):
        d = _read_json(f"{run_dir}/{name}")
        if isinstance(d, dict):
            d = dict(d)
            d["source"] = name
            return d
    return {"source": "none"}


def _run_episode_metas(run_dir: str) -> tuple[list[dict[str, Any]], str]:
    """Return the per-episode metadata rows of a run, and where they came from.

    Resolution order (first non-empty wins):
        1. ``storage.writer.read_index`` -- the authoritative reader.
        2. ``index.jsonl`` / ``episodes.jsonl`` / ``meta.jsonl``.
        3. an ``episodes`` list inside a manifest-like JSON file.
        4. ``gui_index.json`` written by this GUI's fallback batch path.
        5. per-episode metadata JSON files.
        6. a bare ``.npz`` scan, yielding index-only rows.

    Args:
        run_dir: absolute run directory.

    Returns:
        ``(rows, source_label)``.
    """
    reader = BRIDGE.symbol("storage.writer", "read_index")
    if callable(reader):
        try:
            rows = _call_flex(reader, run_dir, run_dir=run_dir, path=run_dir)
            out = [to_jsonable(r) for r in (rows or [])]
            if out:
                return out, "storage.writer.read_index"
        except Exception:
            pass

    for name in ("index.jsonl", "episodes.jsonl", "meta.jsonl"):
        p = f"{run_dir}/{name}"
        if not os.path.isfile(p):
            continue
        rows2: list[dict[str, Any]] = []
        try:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows2.append(obj)
        except Exception:
            pass
        if rows2:
            return rows2, name

    for name in ("manifest.json", "index.json", "run.json", "gui_index.json"):
        d = _read_json(f"{run_dir}/{name}")
        if isinstance(d, dict):
            eps = d.get("episodes") or d.get("index")
            if isinstance(eps, list) and eps:
                return [e for e in eps if isinstance(e, dict)], name
        elif isinstance(d, list) and d:
            return [e for e in d if isinstance(e, dict)], name

    metas = _episode_meta_files(run_dir)
    if metas:
        rows3 = [d for d in (_read_json(p) for p in metas) if isinstance(d, dict)]
        if rows3:
            return rows3, "per-episode json"

    files = _episode_files(run_dir)
    if files:
        return [{"episode_index": i} for i in range(len(files))], "npz scan"
    return [], "none"


def _episode_files(run_dir: str) -> list[str]:
    """Return the sorted list of episode ``.npz`` files in a run directory."""
    hits: list[str] = []
    for pattern in ("*.npz", "episodes/*.npz", "data/*.npz", "**/*.npz"):
        hits = sorted(
            p.replace("\\", "/")
            for p in glob.glob(os.path.join(run_dir, pattern), recursive=True)
        )
        if hits:
            break
    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _episode_meta_files(run_dir: str) -> list[str]:
    """Return sorted per-episode metadata JSON files, if the writer made any."""
    out: list[str] = []
    for pattern in ("meta/*.json", "episodes/*.json", "*_meta.json"):
        hits = sorted(
            p.replace("\\", "/") for p in glob.glob(os.path.join(run_dir, pattern))
        )
        out.extend(h for h in hits if os.path.basename(h) not in
                   ("manifest.json", "index.json", "run.json", "config.json"))
        if out:
            break
    return out


def _load_episode_arrays(npz_path: str) -> EpisodeArrays:
    """Load an ``EpisodeArrays`` from a ``.npz`` file written by the writer.

    Uses ``storage.writer.read_episode`` when it exists so the GUI reads
    exactly what the writer wrote; falls back to a direct ``np.load``.
    """
    reader = BRIDGE.symbol("storage.writer", "read_episode")
    if callable(reader):
        try:
            ep = _call_flex(reader, npz_path, path=npz_path, validate=False)
            if isinstance(ep, EpisodeArrays):
                return ep
        except Exception:
            pass
    with np.load(npz_path, allow_pickle=False) as z:
        missing = [k for k in EpisodeArrays.__dataclass_fields__ if k not in z.files]
        if missing:
            raise ApiError(
                f"episode file is missing arrays {missing}",
                status=500, code="episode_incomplete",
                detail={"file": npz_path, "present": list(z.files)},
            )
        return EpisodeArrays.from_npz_dict({k: z[k] for k in z.files
                                            if k in EpisodeArrays.__dataclass_fields__})


def _episode_file_for(run_dir: str, index: int,
                      files: Optional[list[str]] = None) -> Optional[str]:
    """Resolve the ``.npz`` path of one episode.

    Prefers ``storage.manifest.episode_path`` (the writer's own naming rule)
    and falls back to positional lookup in the sorted file list.

    Args:
        run_dir: absolute run directory.
        index: episode index [-].
        files: pre-computed sorted file list, to avoid re-globbing.

    Returns:
        The path, or ``None`` when the episode does not exist.
    """
    ep_path = BRIDGE.symbol("storage.manifest", "episode_path")
    if callable(ep_path):
        try:
            p = str(ep_path(run_dir, int(index))).replace("\\", "/")
            if os.path.isfile(p):
                return p
        except Exception:
            pass
    files = files if files is not None else _episode_files(run_dir)
    if 0 <= index < len(files):
        return files[index]
    return None


def _frames_from_arrays(ep: EpisodeArrays, stride: int = 1) -> list[dict[str, Any]]:
    """Convert stored arrays into the same frame dicts the live view consumes.

    The one difference from a live frame: ``v`` is the GROUND-speed magnitude,
    because the dataset stores ground velocity and not airspeed. ``omega`` and
    ``mode_timer`` are ``None`` for the same reason -- they are agent internals
    that the on-disk schema deliberately does not carry.

    Args:
        ep: loaded episode arrays.
        stride: keep every ``stride``-th logged step [-] (decimation for very
            long episodes).

    Returns:
        A list of frame dictionaries.
    """
    T, N = ep.n_steps, ep.n_agents
    used = ep.used_slots()
    frames: list[dict[str, Any]] = []
    for k in range(0, T, max(1, int(stride))):
        agents: list[dict[str, Any]] = []
        for j in range(N):
            if not used[j]:
                continue
            valid = bool(ep.gt_valid[k, j])
            det = bool(ep.obs_detected[k, j])
            if not valid and not det:
                continue
            rec: dict[str, Any] = {
                "id": int(ep.agent_ids[j]),
                "slot": j,
                "type": int(ep.gt_type[j]),
                "intent": int(ep.gt_intent[k, j]),
                "mode": int(ep.gt_mode[k, j]),
                "mode_timer": None,
                "valid": valid,
                "det": det,
                "track_id": int(ep.obs_track_id[k, j]),
                "track_age": int(ep.obs_track_age[k, j]),
                "omega_max": None,
                "omega": None,
            }
            if valid:
                rec["x"] = float(ep.gt_pos[k, j, 0])
                rec["y"] = float(ep.gt_pos[k, j, 1])
                rec["vx"] = float(ep.gt_vel[k, j, 0])
                rec["vy"] = float(ep.gt_vel[k, j, 1])
                rec["psi"] = float(ep.gt_psi[k, j])
                rec["v"] = math.hypot(rec["vx"], rec["vy"])
            else:
                rec["x"] = rec["y"] = rec["vx"] = rec["vy"] = None
                rec["psi"] = rec["v"] = None
            if det:
                rec["ox"] = float(ep.obs_pos[k, j, 0])
                rec["oy"] = float(ep.obs_pos[k, j, 1])
                R = ep.obs_R[k, j].astype(np.float64)
                rec["R"] = [[float(R[0, 0]), float(R[0, 1])],
                            [float(R[1, 0]), float(R[1, 1])]]
            else:
                rec["ox"] = rec["oy"] = None
                rec["R"] = None
            agents.append(rec)
        frames.append({
            "t": float(ep.t[k]),
            "step": int(k),
            "own": {
                "x": float(ep.own_pos[k, 0]), "y": float(ep.own_pos[k, 1]),
                "vx": float(ep.own_vel[k, 0]), "vy": float(ep.own_vel[k, 1]),
                "v": float(math.hypot(ep.own_vel[k, 0], ep.own_vel[k, 1])),
                "psi": float(ep.own_psi[k]), "omega": None,
            },
            "agents": agents,
        })
    return to_jsonable(frames)


def _bounds_of(ep: EpisodeArrays) -> list[float]:
    """Return ``[xmin, ymin, xmax, ymax]`` [m] covering ownship and traffic."""
    xs = [float(np.nanmin(ep.own_pos[:, 0])), float(np.nanmax(ep.own_pos[:, 0]))]
    ys = [float(np.nanmin(ep.own_pos[:, 1])), float(np.nanmax(ep.own_pos[:, 1]))]
    if np.any(ep.gt_valid):
        gp = ep.gt_pos[ep.gt_valid]
        if gp.size:
            xs += [float(np.nanmin(gp[:, 0])), float(np.nanmax(gp[:, 0]))]
            ys += [float(np.nanmin(gp[:, 1])), float(np.nanmax(gp[:, 1]))]
    pad = 200.0
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


# ==========================================================================
# CV baseline diagnosis (doc 8.3)
# ==========================================================================

#: Fallback verdict bands, expressed as median FDE at 15 s divided by the NMAC
#: radius. Doc 8.3 says the median 15 s CV error should land within the same
#: order of magnitude as the NMAC radius for the avoidance problem to be
#: meaningfully hard.
CV_TOO_EASY_RATIO: float = 0.20
CV_TOO_RANDOM_RATIO: float = 5.00


def _hist_of(values: Any) -> dict[str, Any]:
    """Histogram + order statistics of an error sample.

    Args:
        values: 1-D array-like of horizontal errors [m].

    Returns:
        ``{n, mean, p50, p90, p95, max, hist:{edges, counts}}`` -- all entries
        are ``None`` when the sample is empty.
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None,
                "max": None, "hist": {"edges": [], "counts": []}}
    hi = float(np.percentile(a, 99.0))
    if not (hi > 0.0):
        hi = float(max(a.max(), 1.0))
    counts, edges = np.histogram(np.clip(a, 0.0, hi), bins=40, range=(0.0, hi))
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50.0)),
        "p90": float(np.percentile(a, 90.0)),
        "p95": float(np.percentile(a, 95.0)),
        "max": float(a.max()),
        "hist": {"edges": [float(e) for e in edges],
                 "counts": [int(c) for c in counts]},
    }


def _cv_diagnose_module(run_dir: str, max_episodes: Optional[int] = None
                        ) -> dict[str, Any]:
    """CV diagnosis using :mod:`analysis.cv_baseline` as the authority.

    The verdict, thresholds and per-class breakdown all come from
    ``analysis.cv_baseline.diagnose`` so the GUI can never disagree with the
    number the paper reports. Only the histograms are computed here, from the
    same merged raw error arrays, because the analysis module reports order
    statistics rather than binned distributions.

    Args:
        run_dir: absolute run directory.
        max_episodes: stop after this many episodes [-] (``None`` = all).

    Returns:
        A GUI-shaped diagnosis dict (see :meth:`Api.diagnose`).

    Raises:
        ApiError: if the analysis module raises or the run has no episodes.
    """
    cvb = BRIDGE.module("analysis.cv_baseline")
    wr = BRIDGE.module("storage.writer")
    if cvb is None or wr is None:
        raise ApiError("analysis.cv_baseline / storage.writer unavailable",
                       status=503, code="subsystem_missing")
    horizons = tuple(getattr(cvb, "DEFAULT_HORIZONS_S", (5.0, 10.0, 15.0)))
    diag_h = float(getattr(cvb, "DIAGNOSTIC_HORIZON_S", 15.0))
    merged = cvb.CVErrors.empty(horizons)
    n = 0
    for _meta, ep in wr.iter_run(run_dir, validate=False, with_meta=False):
        merged.extend(cvb.cv_errors(ep, horizons))
        n += 1
        if max_episodes is not None and n >= int(max_episodes):
            break
    if n == 0:
        raise ApiError(f"run '{os.path.basename(run_dir)}' contains no episodes",
                       status=404, code="no_episodes")
    report = to_jsonable(cvb.diagnose(merged, diagnostic_horizon_s=diag_h))

    per_h = []
    for h in horizons:
        key = f"{float(h):g}"
        stats = _hist_of(merged.errors.get(float(h), np.zeros((0,))))
        src = (report.get("per_horizon") or {}).get(key, {})
        if src.get("median") is not None:
            stats["p50"] = src.get("median")
            stats["p90"] = src.get("p90")
            stats["mean"] = src.get("mean")
            stats["n"] = int(src.get("n") or stats["n"])
        stats["horizon_s"] = float(h)
        stats["verdict"] = src.get("verdict")
        per_h.append(stats)

    def _groups(table: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, rec in (table or {}).items():
            if not isinstance(rec, Mapping):
                continue
            # 'expected' / the band limits come straight from
            # analysis.cv_baseline: a class is judged against its OWN expected
            # difficulty, so a verdict shown without its band would be
            # unreadable (a 60 m median is a pass for an eVTOL and a failure
            # for a bird).
            out[str(k)] = {"label": rec.get("name", str(k)),
                           "p50": rec.get("median"), "p90": rec.get("p90"),
                           "n": int(rec.get("n") or 0),
                           "expected": rec.get("expected"),
                           "band_lo_m": rec.get("band_lo_m"),
                           "band_hi_m": rec.get("band_hi_m"),
                           "verdict": rec.get("verdict")}
        return out

    overall = report.get("overall") or {}
    med = overall.get("median")
    thr = report.get("thresholds") or {}
    nmac = float(thr.get("nmac_radius_m") or NMAC_RADIUS_M)
    return {
        "run": os.path.basename(run_dir),
        "source": "analysis.cv_baseline",
        "verdict": report.get("verdict", "NO_DATA"),
        "note": "; ".join(report.get("warnings") or []) or
                "verdict and thresholds come from analysis.cv_baseline (doc 8.3).",
        "n_episodes_scanned": int(report.get("n_episodes") or n),
        "n_samples": int(overall.get("n") or 0),
        "nmac_radius_m": nmac,
        "dwc_hmd_m": DWC_HMD_M,
        "median_fde_15s_m": med,
        "ratio_to_nmac": (float(med) / nmac) if isinstance(med, (int, float)) else None,
        "bands": {
            "too_easy_below": thr.get("too_easy_fraction_of_nmac"),
            "too_random_above": thr.get("too_random_multiple_of_nmac"),
        },
        "thresholds": thr,
        "diagnostic_horizon_s": report.get("diagnostic_horizon_s", diag_h),
        "warnings": report.get("warnings") or [],
        "horizons": per_h,
        "by_platform": _groups(report.get("by_platform")),
        "by_intent": _groups(report.get("by_intent")),
        "gate": report.get("gate") or {},
        "band_separation": report.get("band_separation") or {},
    }


def _cv_diagnose_fallback(run_dir: str, max_episodes: int = 200,
                          time_stride: int = 4) -> dict[str, Any]:
    """Compute the CV-extrapolation diagnosis directly from stored episodes.

    Used only when ``analysis.cv_baseline`` is unavailable. The definition is
    doc 8.3's: ``future = position + velocity * horizon`` with no learning.

    Args:
        run_dir: absolute run directory.
        max_episodes: cap on episodes scanned [-], to keep the UI responsive.
        time_stride: sample every ``time_stride``-th anchor step [-].

    Returns:
        A diagnosis dict with ``verdict``, per-horizon statistics and
        histograms, plus per-platform and per-intent breakdowns.
    """
    files = _episode_files(run_dir)
    if not files:
        raise ApiError(f"no episode .npz files under {run_dir}", status=404,
                       code="no_episodes")
    horizons = [5.0, 10.0, 15.0]
    hsteps = [int(round(h / LOG_DT)) for h in horizons]
    errs: dict[int, list[float]] = {h: [] for h in hsteps}
    by_plat: dict[int, list[float]] = {}
    by_intent: dict[int, list[float]] = {}
    n_ep = 0
    n_samples = 0

    for path in files[:max_episodes]:
        try:
            ep = _load_episode_arrays(path)
        except Exception:
            continue
        n_ep += 1
        T, N = ep.n_steps, ep.n_agents
        pos = ep.gt_pos.astype(np.float64)
        vel = ep.gt_vel.astype(np.float64)
        valid = ep.gt_valid
        hmax = max(hsteps)
        for k in range(0, max(T - hmax, 0), max(1, int(time_stride))):
            live = valid[k]
            if not np.any(live):
                continue
            for hs in hsteps:
                tgt = valid[k + hs] & live
                if not np.any(tgt):
                    continue
                pred = pos[k] + vel[k] * (hs * LOG_DT)
                d = np.hypot(pred[:, 0] - pos[k + hs][:, 0],
                             pred[:, 1] - pos[k + hs][:, 1])
                sel = d[tgt]
                errs[hs].extend(float(x) for x in sel if math.isfinite(x))
                if hs == hsteps[-1]:
                    n_samples += int(np.count_nonzero(tgt))
                    idx = np.nonzero(tgt)[0]
                    for j in idx:
                        pt = int(ep.gt_type[j])
                        it = int(ep.gt_intent[k, j])
                        e = float(d[j])
                        if math.isfinite(e):
                            by_plat.setdefault(pt, []).append(e)
                            by_intent.setdefault(it, []).append(e)

    def _stats(vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {"n": 0, "mean": None, "p50": None, "p90": None,
                    "p95": None, "max": None,
                    "hist": {"edges": [], "counts": []}}
        a = np.asarray(vals, dtype=np.float64)
        hi = float(np.percentile(a, 99.0))
        hi = hi if hi > 0 else 1.0
        counts, edges = np.histogram(np.clip(a, 0.0, hi), bins=40, range=(0.0, hi))
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50.0)),
            "p90": float(np.percentile(a, 90.0)),
            "p95": float(np.percentile(a, 95.0)),
            "max": float(a.max()),
            "hist": {"edges": [float(e) for e in edges],
                     "counts": [int(c) for c in counts]},
        }

    per_h = []
    for h, hs in zip(horizons, hsteps):
        s = _stats(errs[hs])
        s["horizon_s"] = float(h)
        per_h.append(s)

    med15 = per_h[-1]["p50"]
    if med15 is None:
        verdict = "NO_DATA"
        ratio = None
    else:
        ratio = float(med15) / NMAC_RADIUS_M
        if ratio < CV_TOO_EASY_RATIO:
            verdict = "TOO_EASY"
        elif ratio > CV_TOO_RANDOM_RATIO:
            verdict = "TOO_RANDOM"
        else:
            verdict = "OK"

    notes = {
        "TOO_EASY": "CV 15 s error is far below the NMAC radius: trajectories "
                    "are too straight. Increase turn frequency / turn rate in "
                    "the behaviour rules (doc 8.3, doc 3.2).",
        "TOO_RANDOM": "CV 15 s error dwarfs the NMAC radius: motion is close to "
                      "noise. Increase the maneuver dwell time so a mode is "
                      "held for several seconds (doc 3.3 constraint b).",
        "OK": "CV 15 s median error is within the same order of magnitude as "
              "the NMAC radius: the dataset is a meaningful avoidance problem "
              "(doc 8.3).",
        "NO_DATA": "No valid prediction windows were found in this run.",
    }[verdict]

    return {
        "run": os.path.basename(run_dir),
        "source": "gui_fallback",
        "verdict": verdict,
        "note": notes,
        "n_episodes_scanned": n_ep,
        "n_samples": n_samples,
        "nmac_radius_m": NMAC_RADIUS_M,
        "dwc_hmd_m": DWC_HMD_M,
        "median_fde_15s_m": med15,
        "ratio_to_nmac": ratio,
        "bands": {"too_easy_below": CV_TOO_EASY_RATIO,
                  "too_random_above": CV_TOO_RANDOM_RATIO},
        "horizons": per_h,
        "by_platform": {
            str(k): dict(_stats(v), label=TYPE_NAMES.get(k, str(k)))
            for k, v in sorted(by_plat.items()) if k >= 0
        },
        "by_intent": {
            str(k): dict(_stats(v), label=INTENT_NAMES.get(k, str(k)))
            for k, v in sorted(by_intent.items()) if k >= 0
        },
    }


# ==========================================================================
# The API facade
# ==========================================================================


class Api:
    """Implementation of every ``/api/*`` endpoint.

    One instance is shared by all HTTP worker threads, so every mutable
    registry it owns is guarded by a lock.
    """

    #: Sessions idle for longer than this are reaped [s].
    SESSION_TTL_S: float = 1800.0

    def __init__(self) -> None:
        self._sims: dict[str, SimSession] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        os.makedirs(LOGS_ROOT, exist_ok=True)

    # -- helpers ----------------------------------------------------------

    def _session(self, sim_id: str) -> SimSession:
        """Look up a live session or raise 404."""
        with self._lock:
            s = self._sims.get(str(sim_id))
        if s is None or s.closed:
            raise ApiError(f"no live simulation '{sim_id}'", status=404,
                           code="sim_not_found")
        s.last_touch = time.time()
        return s

    def _reap(self) -> None:
        """Close sessions with no subscribers that have been idle too long."""
        now = time.time()
        with self._lock:
            dead = [
                k for k, s in self._sims.items()
                if s.n_subscribers == 0 and (now - s.last_touch) > self.SESSION_TTL_S
            ]
            for k in dead:
                try:
                    self._sims[k].close()
                except Exception:
                    pass
                self._sims.pop(k, None)

    @staticmethod
    def _config_from_body(body: Mapping[str, Any]) -> ScenarioConfig:
        """Build a validated :class:`ScenarioConfig` from a request body.

        Accepts ``{"config": {...}}``, ``{"preset": "name"}`` or a bare config
        dict. Raises :class:`ApiError` listing every validation problem.
        """
        cfg_d = body.get("config") if isinstance(body, Mapping) else None
        preset = body.get("preset") if isinstance(body, Mapping) else None
        if cfg_d is None and preset:
            cfg = Api._preset_config(str(preset))
        elif cfg_d is None and isinstance(body, Mapping) and "name" in body:
            cfg = ScenarioConfig.from_dict(body)
        elif cfg_d is None:
            cfg = ScenarioConfig()
        else:
            cfg = ScenarioConfig.from_dict(cfg_d)
        if isinstance(body, Mapping) and body.get("seed") is not None:
            try:
                cfg.seed = int(body["seed"])
            except (TypeError, ValueError):
                pass
        problems = cfg.validate()
        if problems:
            raise ApiError("scenario configuration is invalid",
                           code="config_invalid", detail={"problems": problems})
        return cfg

    @staticmethod
    def _preset_config(name: str) -> ScenarioConfig:
        """Resolve a preset name to a config, file presets first."""
        for p in Api._preset_files():
            if os.path.splitext(os.path.basename(p))[0] == name:
                d = _read_json(p)
                if isinstance(d, dict):
                    return ScenarioConfig.from_dict(d)
        table = BRIDGE.symbol("scenario.presets", "PRESETS")
        if isinstance(table, Mapping) and name in table:
            v = table[name]
            return v if isinstance(v, ScenarioConfig) else ScenarioConfig.from_dict(v)
        for cand in ("load_preset", "make_preset", "get_preset", "build_preset",
                     "preset"):
            getter = BRIDGE.symbol("scenario.presets", cand)
            if not callable(getter):
                continue
            try:
                v = getter(name)
            except Exception:
                continue
            if isinstance(v, ScenarioConfig):
                return v
            if isinstance(v, Mapping):
                return ScenarioConfig.from_dict(v)
        raise ApiError(f"unknown preset '{name}'", status=404, code="preset_not_found")

    @staticmethod
    def _preset_files() -> list[str]:
        """Every ``config/presets/*.json`` path, sorted."""
        return sorted(
            p.replace("\\", "/") for p in glob.glob(PRESET_DIR + "/*.json")
        )

    # -- GET /api/meta ----------------------------------------------------

    def meta(self) -> dict[str, Any]:
        """Enum tables, colours, constants and bridge status for the UI."""
        return {
            "platforms": [
                {"value": int(p), "name": TYPE_NAMES[int(p)],
                 "color": PLATFORM_COLORS[p]}
                for p in PlatformType
            ],
            "modes": [{"value": int(m), "name": MODE_NAMES[int(m)]}
                      for m in ManeuverMode],
            "intents": [{"value": int(i), "name": INTENT_NAMES[int(i)]}
                        for i in IntentType],
            "geometries": [{"value": int(g), "name": GEOMETRY_NAMES[int(g)]}
                           for g in GeometryType],
            "risks": [{"value": int(r), "name": RISK_NAMES[int(r)]}
                      for r in RiskLevel],
            "constants": {
                "G": G, "SIM_DT": SIM_DT, "LOG_DT": LOG_DT,
                "LOG_DECIMATION": LOG_DECIMATION, "HIST_STEPS": HIST_STEPS,
                "PRED_STEPS": PRED_STEPS, "NMAC_RADIUS_M": NMAC_RADIUS_M,
                "DWC_HMD_M": DWC_HMD_M, "DWC_TAU_MOD_S": DWC_TAU_MOD_S,
                "EPS_V": EPS_V, "LOGS_ROOT": LOGS_ROOT,
            },
            "bridge": BRIDGE.status(),
            "frame_schema": frame_schema_doc(),
            "server_time": time.time(),
        }

    # -- GET /api/presets -------------------------------------------------

    def presets(self) -> dict[str, Any]:
        """List every preset that can be resolved, with its full config."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in self._preset_files():
            name = os.path.splitext(os.path.basename(p))[0]
            d = _read_json(p)
            if not isinstance(d, dict):
                continue
            seen.add(name)
            out.append({"name": name, "source": "file", "path": p, "config": d})
        table = BRIDGE.symbol("scenario.presets", "PRESETS")
        if isinstance(table, Mapping):
            for name, v in table.items():
                if str(name) in seen:
                    continue
                try:
                    cfg = v if isinstance(v, ScenarioConfig) else ScenarioConfig.from_dict(v)
                except Exception:
                    continue
                seen.add(str(name))
                out.append({"name": str(name), "source": "scenario.presets",
                            "path": None, "config": cfg.to_dict()})
        names: list[str] = []
        declared = BRIDGE.symbol("scenario.presets", "PRESET_NAMES")
        if isinstance(declared, (list, tuple)):
            names.extend(str(n) for n in declared)
        lister = BRIDGE.symbol("scenario.presets", "list_presets")
        if callable(lister):
            try:
                names.extend(str(n) for n in lister())
            except Exception:
                pass
        for name in names:
            if name in seen:
                continue
            try:
                cfg = self._preset_config(name)
            except Exception:
                continue
            seen.add(name)
            out.append({"name": name, "source": "scenario.presets",
                        "path": None, "config": cfg.to_dict()})
        if not out:
            out.append({"name": "default", "source": "builtin", "path": None,
                        "config": ScenarioConfig().to_dict()})
        out.sort(key=lambda r: r["name"])
        return {"presets": out}

    # -- GET /api/config/defaults ----------------------------------------

    def config_defaults(self) -> dict[str, Any]:
        """The default scenario configuration, from disk when present."""
        d = _read_json(DEFAULTS_PATH)
        if isinstance(d, dict):
            try:
                cfg = ScenarioConfig.from_dict(d)
                return {"config": cfg.to_dict(), "source": DEFAULTS_PATH,
                        "problems": cfg.validate()}
            except Exception:
                pass
        cfg = ScenarioConfig()
        return {"config": cfg.to_dict(), "source": "ScenarioConfig()",
                "problems": cfg.validate()}

    # -- POST /api/sim/new ------------------------------------------------

    def sim_new(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Create a live session and return its first frame."""
        self._reap()
        cfg = self._config_from_body(body)
        sim_id = uuid.uuid4().hex[:12]
        session = SimSession(sim_id, cfg)
        with self._lock:
            self._sims[sim_id] = session
        return {"sim_id": sim_id, "config": cfg.to_dict(), "frame0": session.frame}

    # -- POST /api/sim/control -------------------------------------------

    def sim_control(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Apply play / pause / step / reset / speed to a live session."""
        sim_id = str(body.get("sim_id", ""))
        action = str(body.get("action", "")).lower()
        speed = body.get("speed")
        session = self._session(sim_id)
        frame = session.control(action, None if speed is None else float(speed))
        return {"sim_id": sim_id, "playing": session.playing,
                "speed": session.speed, "frame": frame,
                "error": session.error}

    # -- GET /api/sim/agent ----------------------------------------------

    def sim_agent(self, sim_id: str, agent_id: int) -> dict[str, Any]:
        """Full inspector detail for one agent of a live session."""
        session = self._session(sim_id)
        with session.lock:
            frame = session.frame
        rec = next((a for a in frame.get("agents", [])
                    if int(a.get("id", -1)) == int(agent_id)), None)
        if rec is None:
            raise ApiError(f"agent {agent_id} is not in the current frame",
                           status=404, code="agent_not_found")
        own = frame.get("own", {})
        detail = dict(rec)
        detail.update(self._relative_geometry(own, rec))
        detail["type_name"] = TYPE_NAMES.get(int(rec.get("type", -1)), "?")
        detail["intent_name"] = INTENT_NAMES.get(int(rec.get("intent", -1)), "?")
        detail["mode_name"] = MODE_NAMES.get(int(rec.get("mode", -1)), "?")
        detail["t"] = frame.get("t")
        return {"agent": to_jsonable(detail)}

    @staticmethod
    def _relative_geometry(own: Mapping[str, Any],
                           rec: Mapping[str, Any]) -> dict[str, Any]:
        """Range, bearing, CPA distance and TCPA under constant velocity.

        Args:
            own: ownship sub-dict of a frame.
            rec: one agent record of a frame.

        Returns:
            ``{'range_m', 'bearing_rad', 'rel_bearing_rad', 'cpa_m', 'tcpa_s'}``
            with ``None`` entries where the geometry is undefined.
        """
        try:
            px = float(rec["x"]) - float(own["x"])
            py = float(rec["y"]) - float(own["y"])
            vx = float(rec["vx"]) - float(own["vx"])
            vy = float(rec["vy"]) - float(own["vy"])
        except (TypeError, KeyError, ValueError):
            return {"range_m": None, "bearing_rad": None, "rel_bearing_rad": None,
                    "cpa_m": None, "tcpa_s": None}
        rng = math.hypot(px, py)
        brg = math.atan2(py, px)
        rel = wrap_angle(brg - float(own.get("psi", 0.0)))
        vv = vx * vx + vy * vy
        if vv <= 1e-9:
            return {"range_m": rng, "bearing_rad": brg, "rel_bearing_rad": rel,
                    "cpa_m": rng, "tcpa_s": None}
        tcpa = -(px * vx + py * vy) / vv
        tcpa_c = max(tcpa, 0.0)
        cpa = math.hypot(px + vx * tcpa_c, py + vy * tcpa_c)
        return {"range_m": rng, "bearing_rad": brg, "rel_bearing_rad": rel,
                "cpa_m": cpa, "tcpa_s": tcpa}

    # -- POST /api/rollout ------------------------------------------------

    def rollout(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Branch-rollout fan from the current state of a live session.

        The fan is what makes multimodality visible: the same initial condition
        replayed with re-seeded decision streams (doc 3.3).
        """
        sim_id = str(body.get("sim_id", ""))
        n_branches = int(body.get("n_branches", 12))
        horizon_s = float(body.get("horizon_s", PRED_STEPS * LOG_DT))
        n_branches = max(1, min(n_branches, 200))
        horizon_s = max(LOG_DT, min(horizon_s, 120.0))
        session = self._session(sim_id)
        fn = BRIDGE.require("simulation.rollout")
        with session.lock:
            engine = session.engine
            base_frame = session.frame
            snap = _snapshot_of(engine)
            n_steps = int(round(horizon_s / LOG_DT))
            branch_t = _engine_time(engine, session.step_index * LOG_DT)
            try:
                raw = _bind_call(
                    fn,
                    {"engine": engine, "sim": engine, "simulator": engine,
                     "world": engine, "session": engine,
                     "snapshot": snap, "world_snapshot": snap, "state": snap,
                     "branch_t": branch_t, "t": branch_t, "t0": branch_t,
                     "start_t": branch_t, "from_t": branch_t,
                     "n_branches": n_branches, "n": n_branches,
                     "branches": n_branches, "n_rollouts": n_branches,
                     "horizon_s": horizon_s, "horizon": horizon_s,
                     "n_steps": n_steps, "steps": n_steps,
                     "dt": LOG_DT, "branch_seed": int(body.get("seed", 0)),
                     "seed": int(body.get("seed", 0)),
                     # a visual check must never hard-fail on degeneracy; the
                     # GUI detects and REPORTS it instead (see below).
                     "require_divergence": False},
                    [n_branches, horizon_s],
                )
            except ApiError:
                raise
            except Exception as exc:
                raise ApiError(
                    "branch rollout failed inside simulation.rollout",
                    status=500, code="rollout_failed",
                    detail={"exception": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=6)},
                ) from exc
        fan = self._normalize_fan(raw, base_frame, n_branches)
        fan["horizon_s"] = horizon_s
        fan["dt"] = LOG_DT
        fan["n_branches"] = n_branches
        fan["branch_t"] = branch_t
        fan["spread_m"] = self._fan_spread(fan)
        if n_branches > 1 and fan["spread_m"] is not None and fan["spread_m"] < 1.0:
            fan["warning"] = (
                "branches did not diverge (final-point spread "
                f"{fan['spread_m']:.2f} m). The futures are effectively "
                "deterministic, so a CV extrapolator will win -- the agents' "
                "decision rules are not actually stochastic (doc 3.3)."
            )
        return fan

    @staticmethod
    def _fan_spread(fan: Mapping[str, Any]) -> Optional[float]:
        """Mean over agents of the std-dev of branch END points [m].

        This is the single number that says whether the dataset is multimodal:
        if it is ~0, every branch produced the same future and the premise of
        the paper (doc 3.3) is not being met by the simulator.
        """
        spreads: list[float] = []
        for a in fan.get("agents", []):
            ends = []
            for br in a.get("branches", []):
                if br and br[-1] is not None:
                    p = br[-1]
                    if (p[0] is not None and p[1] is not None
                            and math.isfinite(p[0]) and math.isfinite(p[1])):
                        ends.append((float(p[0]), float(p[1])))
            if len(ends) < 2:
                continue
            arr = np.asarray(ends, dtype=np.float64)
            centre = arr.mean(axis=0)
            spreads.append(float(np.sqrt(
                np.mean(np.sum((arr - centre) ** 2, axis=1)))))
        return float(np.mean(spreads)) if spreads else None

    @staticmethod
    def _normalize_fan(raw: Any, base_frame: Mapping[str, Any],
                       n_branches: int) -> dict[str, Any]:
        """Coerce whatever ``simulation.rollout`` returned into a drawable fan.

        Accepted shapes:
            * ndarray ``[B, N, S, 2]`` -- branch, agent, step, (x, y).
            * ndarray ``[B, S, 2]``    -- single tracked agent.
            * ``{"agents": [{"id":.., "branches": [[[x,y],...], ...]}]}``.
            * list of per-branch lists of :class:`WorldSnapshot`-like objects.

        Returns:
            ``{"agents": [{"id": int, "branches": [[[x, y], ...], ...]}]}``.
        """
        ids = [int(a["id"]) for a in base_frame.get("agents", [])]

        if isinstance(raw, Mapping) and "agents" in raw:
            return to_jsonable({"agents": raw["agents"]})

        arr = None
        if isinstance(raw, np.ndarray):
            arr = raw
        elif isinstance(raw, (list, tuple)) and raw and isinstance(
            raw[0], (np.ndarray, list, tuple)
        ):
            try:
                arr = np.asarray(raw, dtype=np.float64)
            except Exception:
                arr = None

        def _poly(track: np.ndarray) -> list[list[float]]:
            """Trim a branch at the first non-finite sample.

            ``simulation.rollout`` marks an agent that is no longer alive with
            NaN; a polyline must stop there rather than carry a hole that would
            break the canvas path.
            """
            pts: list[list[float]] = []
            for s in range(track.shape[0]):
                x, y = float(track[s, 0]), float(track[s, 1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    break
                pts.append([x, y])
            return pts

        if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[-1] >= 2:
            B, N = arr.shape[0], arr.shape[1]
            agents = []
            for j in range(N):
                aid = ids[j] if j < len(ids) else j
                branches = [_poly(arr[b, j]) for b in range(B)]
                if not any(len(p) > 1 for p in branches):
                    continue
                agents.append({"id": aid, "branches": branches})
            return {"agents": agents}

        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 2:
            B = arr.shape[0]
            aid = ids[0] if ids else 0
            return {"agents": [{
                "id": aid,
                "branches": [_poly(arr[b]) for b in range(B)],
            }]}

        # list of per-branch snapshot sequences
        if isinstance(raw, (list, tuple)) and raw:
            per_agent: dict[int, list[list[list[float]]]] = {}
            for b, seq in enumerate(raw[:n_branches]):
                if not isinstance(seq, (list, tuple)):
                    continue
                tracks: dict[int, list[list[float]]] = {}
                for snap in seq:
                    aids = list(getattr(snap, "agent_ids", []) or [])
                    asn = list(getattr(snap, "agents", []) or [])
                    for j, a in enumerate(asn):
                        st = getattr(a, "state", a)
                        aid = int(aids[j]) if j < len(aids) else j
                        tracks.setdefault(aid, []).append(
                            [float(getattr(st, "x", 0.0)),
                             float(getattr(st, "y", 0.0))]
                        )
                for aid, pts in tracks.items():
                    per_agent.setdefault(aid, []).append(pts)
            if per_agent:
                return {"agents": [{"id": k, "branches": v}
                                   for k, v in sorted(per_agent.items())]}

        raise ApiError(
            "simulation.rollout returned a shape the GUI does not understand",
            status=500, code="rollout_shape",
            detail={"type": type(raw).__name__,
                    "expected": "ndarray [B,N,S,2] or {'agents':[{'id','branches'}]}"},
        )

    # -- POST /api/generate -----------------------------------------------

    #: Keys ``POST /api/generate`` understands at the top level. Anything else
    #: is REJECTED rather than ignored: this subcommand writes a dataset to
    #: disk, and silently substituting the default episode count for a
    #: misspelled key ("episodes" instead of "n_episodes") wrote a 10-episode
    #: run when 2 were asked for. ``episodes`` is accepted as an alias because
    #: it is the CLI's own flag name.
    _GENERATE_KEYS: frozenset[str] = frozenset({
        "config", "preset", "name", "seed", "n_episodes", "episodes", "tag",
    })

    def generate(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Start a background dataset-generation job.

        Raises:
            ApiError: on an unknown top-level key, or an out-of-range episode
                count.
        """
        if isinstance(body, Mapping) and "name" not in body:
            unknown = sorted(set(body) - self._GENERATE_KEYS)
            if unknown:
                raise ApiError(
                    "unknown key(s) in the generate request",
                    code="unknown_key",
                    detail={"unknown": unknown,
                            "accepted": sorted(self._GENERATE_KEYS)},
                )
        cfg = self._config_from_body(body)
        raw_n = body.get("n_episodes", body.get("episodes", 10))
        try:
            n_episodes = int(raw_n)
        except (TypeError, ValueError):
            raise ApiError(f"n_episodes must be an integer, got {raw_n!r}",
                           code="bad_n_episodes") from None
        if n_episodes < 1 or n_episodes > 200000:
            raise ApiError("n_episodes must be between 1 and 200000",
                           code="bad_n_episodes")
        tag = str(body.get("tag") or cfg.name or "run").strip()
        if not _SAFE_NAME.match(tag):
            tag = re.sub(r"[^A-Za-z0-9_.\-]", "_", tag) or "run"
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, tag=tag, n_episodes=n_episodes)
        with self._lock:
            self._jobs[job_id] = job
        th = threading.Thread(target=self._generate_worker,
                              args=(job, cfg, n_episodes, tag),
                              name=f"gen-{job_id}", daemon=True)
        th.start()
        return {"job_id": job_id, "n_episodes": n_episodes, "tag": tag}

    def _generate_worker(self, job: Job, cfg: ScenarioConfig,
                         n_episodes: int, tag: str) -> None:
        """Worker thread body for :meth:`generate`."""
        job.state = "running"
        job.started = time.time()
        job.emit(f"starting: {n_episodes} episodes, scenario '{cfg.name}'")
        try:
            runner = BRIDGE.require("simulation.episode")
            writer_sym = BRIDGE.require("storage.writer")
            writer, run_dir = self._open_writer(writer_sym, cfg, tag, job)
            job.run_path = run_dir
            job.emit(f"output directory: {run_dir}")
            run_fn = runner if callable(runner) else None
            runner_obj = None
            if run_fn is None or inspect.isclass(runner):
                runner_obj = _construct(runner, cfg, int(cfg.seed))
                run_fn = _first_attr(runner_obj, ("run", "run_episode", "generate",
                                                  "__call__"))
            metas: list[dict[str, Any]] = []
            for i in range(n_episodes):
                if job.cancel.is_set():
                    job.state = "cancelled"
                    job.emit("cancelled by user")
                    break
                seed = int(cfg.seed) + i
                ep_cfg = cfg.copy_with(seed=seed)
                result = _bind_call(
                    run_fn,
                    {"config": ep_cfg, "cfg": ep_cfg, "scenario": ep_cfg,
                     "scenario_config": ep_cfg,
                     "episode_index": i, "index": i, "idx": i,
                     "seed": seed, "master_seed": seed},
                    [ep_cfg, i, seed],
                )
                ep, meta = self._split_episode_result(result, i, seed, cfg)
                self._write_episode(writer, run_dir, i, ep, meta, job)
                if meta is not None:
                    metas.append(to_jsonable(meta))
                job.done = i + 1
                if (i + 1) % max(1, n_episodes // 50) == 0 or i == n_episodes - 1:
                    job.emit(f"episode {i + 1}/{n_episodes} written")
            self._close_writer(writer, job)
            if job.state != "cancelled":
                job.state = "done"
                job.emit(f"finished: {job.done} episodes -> {run_dir}")
            self._write_gui_index(run_dir, metas, cfg, job)
        except ApiError as exc:
            job.state = "error"
            job.emit(f"ERROR {exc.code}: {exc.message}")
            if exc.detail:
                job.emit("detail: " + json.dumps(to_jsonable(exc.detail))[:400])
        except Exception as exc:
            job.state = "error"
            job.emit(f"ERROR {type(exc).__name__}: {exc}")
            for line in traceback.format_exc(limit=8).splitlines()[-8:]:
                job.emit(line)
        finally:
            job.finished = time.time()

    @staticmethod
    def _split_episode_result(result: Any, index: int, seed: int,
                              cfg: ScenarioConfig
                              ) -> tuple[Any, Optional[Any]]:
        """Split whatever the episode runner returned into (arrays, meta)."""
        if isinstance(result, tuple) and len(result) >= 2:
            return result[0], result[1]
        if isinstance(result, Mapping):
            return (result.get("arrays") or result.get("episode"),
                    result.get("meta"))
        ep = _first_attr(result, ("arrays", "episode"))
        meta = _first_attr(result, ("meta", "metadata"))
        if ep is not None:
            return ep, meta
        return result, None

    @staticmethod
    def _open_writer(writer_sym: Any, cfg: ScenarioConfig, tag: str,
                     job: Job) -> tuple[Any, str]:
        """Open a run for writing and return ``(writer, run_dir)``.

        The expected contract is :class:`storage.writer.DatasetWriter`:
        ``DatasetWriter(logs_root).open_run(config, preset=..., tag=...)``
        returns the run directory. A plain callable writer is accepted too, in
        which case the GUI creates a timestamped directory itself.

        Args:
            writer_sym: the class or function found in ``storage.writer``.
            cfg: scenario config for this run.
            tag: user-supplied run tag.
            job: the job, used for log lines.

        Returns:
            ``(writer_object_or_callable, absolute_run_dir)``.

        Raises:
            ApiError: if a writer class could not be constructed or opened.
        """
        if inspect.isclass(writer_sym):
            obj = None
            for label, mk in (
                ("cls(LOGS_ROOT)", lambda: writer_sym(LOGS_ROOT)),
                ("cls()", lambda: writer_sym()),
            ):
                try:
                    obj = mk()
                    if obj is not None:
                        job.emit(f"writer: {writer_sym.__name__} via {label}")
                        break
                except Exception:
                    obj = None
            if obj is None:
                raise ApiError("could not construct storage writer", status=503,
                               code="writer_construct_failed",
                               detail={"class": getattr(writer_sym, "__name__", "?")})
            open_run = getattr(obj, "open_run", None)
            if callable(open_run):
                try:
                    run_dir = _bind_call(
                        open_run,
                        {"config": cfg, "cfg": cfg, "scenario": cfg,
                         "preset": str(cfg.name), "tag": tag,
                         "master_seed": int(cfg.seed)},
                        [cfg],
                    )
                except Exception as exc:
                    raise ApiError(
                        "storage writer refused to open a run",
                        status=500, code="writer_open_failed",
                        detail={"exception": f"{type(exc).__name__}: {exc}"},
                    ) from exc
                run_dir = str(run_dir or getattr(obj, "run_dir", "")).replace("\\", "/")
                if not run_dir:
                    raise ApiError("storage writer returned no run directory",
                                   status=500, code="writer_open_failed")
                return obj, run_dir
            run_dir = f"{LOGS_ROOT}/{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(run_dir, exist_ok=True)
            return obj, run_dir

        job.emit(f"writer: function {getattr(writer_sym, '__name__', writer_sym)}")
        run_dir = f"{LOGS_ROOT}/{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(run_dir, exist_ok=True)
        try:
            from scenario.config import save_scenario_config
            save_scenario_config(cfg, f"{run_dir}/config.json")
        except Exception as exc:
            job.emit(f"warning: could not write config.json ({exc})")
        return writer_sym, run_dir

    @staticmethod
    def _write_episode(writer: Any, run_dir: str, index: int, ep: Any,
                       meta: Any, job: Job) -> None:
        """Hand one episode to the writer, binding arguments by name."""
        target: Any = None
        if not (inspect.isfunction(writer) or inspect.isbuiltin(writer)):
            for name in ("write_episode", "write", "append", "add", "save_episode"):
                fn = getattr(writer, name, None)
                if callable(fn):
                    target = fn
                    break
        if target is None:
            target = writer
        _bind_call(
            target,
            {"ep": ep, "episode": ep, "arrays": ep, "episode_arrays": ep,
             "meta": meta, "metadata": meta, "episode_meta": meta,
             "index": index, "episode_index": index,
             "run_dir": run_dir, "path": run_dir, "root": run_dir},
            [ep, meta, index],
        )

    @staticmethod
    def _close_writer(writer: Any, job: Job) -> None:
        """Close / finalise the writer if it exposes such a method."""
        for name in ("close", "finalize", "finalise", "flush", "finish"):
            fn = getattr(writer, name, None)
            if callable(fn):
                try:
                    _call_flex(fn)
                except Exception as exc:
                    job.emit(f"warning: writer.{name}() raised {exc}")
                return

    @staticmethod
    def _write_gui_index(run_dir: str, metas: list[dict[str, Any]],
                         cfg: ScenarioConfig, job: Job) -> None:
        """Write ``gui_index.json`` as a fallback episode table for Replay.

        Only written when the real writer produced no manifest; it never
        overwrites ``manifest.json`` or ``index.json``.
        """
        if not metas:
            return
        for name in ("manifest.json", "index.json", "run.json", "index.jsonl"):
            if os.path.isfile(f"{run_dir}/{name}"):
                return
        try:
            with open(f"{run_dir}/gui_index.json", "w", encoding="utf-8") as fh:
                json.dump({"scenario": cfg.to_dict(), "episodes": metas},
                          fh, indent=1, ensure_ascii=False)
        except Exception as exc:
            job.emit(f"warning: could not write gui_index.json ({exc})")

    # -- GET /api/jobs/{id} ----------------------------------------------

    def job(self, job_id: str) -> dict[str, Any]:
        """Progress report for a background job."""
        with self._lock:
            j = self._jobs.get(str(job_id))
        if j is None:
            raise ApiError(f"no job '{job_id}'", status=404, code="job_not_found")
        return j.snapshot()

    def job_cancel(self, job_id: str) -> dict[str, Any]:
        """Request cooperative cancellation of a running job."""
        with self._lock:
            j = self._jobs.get(str(job_id))
        if j is None:
            raise ApiError(f"no job '{job_id}'", status=404, code="job_not_found")
        j.cancel.set()
        j.emit("cancellation requested")
        return j.snapshot()

    def jobs(self) -> dict[str, Any]:
        """List every job this server has started."""
        with self._lock:
            js = list(self._jobs.values())
        return {"jobs": [j.snapshot() for j in js]}

    # -- GET /api/runs ----------------------------------------------------

    def runs(self) -> dict[str, Any]:
        """List dataset runs found under ``LOGS_ROOT``."""
        root = LOGS_ROOT
        out: list[dict[str, Any]] = []
        try:
            names = sorted(os.listdir(root))
        except OSError:
            names = []
        for name in names:
            path = f"{root}/{name}"
            if not os.path.isdir(path) or not _SAFE_NAME.match(name):
                continue
            files = _episode_files(path)
            man = _run_manifest(path)
            try:
                metas, meta_src = _run_episode_metas(path)
            except Exception:
                metas, meta_src = [], "error"
            n_meta = len(metas)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            cfg_d = _read_json(f"{path}/config.json")
            out.append({
                "name": name,
                "path": path,
                "n_episode_files": len(files),
                "n_meta": n_meta,
                "modified": mtime,
                "modified_str": time.strftime("%Y-%m-%d %H:%M",
                                              time.localtime(mtime)) if mtime else "",
                "manifest_source": man.get("source"),
                "index_source": meta_src,
                "status": man.get("status"),
                "run_id": man.get("run_id"),
                "preset": man.get("preset"),
                "scenario_name": (cfg_d or {}).get("name") if isinstance(cfg_d, dict) else None,
            })
        out.sort(key=lambda r: r["modified"], reverse=True)
        return {"root": root, "runs": out}

    # -- GET /api/runs/{run}/index ---------------------------------------

    def run_index(self, run: str) -> dict[str, Any]:
        """Stratified metadata table of one run, for the Replay tab."""
        run_dir = _safe_run_dir(run)
        episodes, source = _run_episode_metas(run_dir)
        files = _episode_files(run_dir)

        rows: list[dict[str, Any]] = []
        for i, e in enumerate(episodes):
            try:
                m = EpisodeMeta.from_dict(e) if isinstance(e, Mapping) else EpisodeMeta()
                d = m.to_dict()
            except Exception:
                d = dict(e) if isinstance(e, Mapping) else {}
            d.setdefault("episode_index", i)
            d["row"] = i
            d["risk_name"] = RISK_NAMES.get(int(d.get("risk_level", 0) or 0), "?")
            geo = d.get("geometry_type") or []
            d["geometry_names"] = [GEOMETRY_NAMES.get(int(g), "?") for g in geo]
            cpa = d.get("cpa_dist") or []
            d["cpa_min"] = min((float(c) for c in cpa), default=None)
            d["tcpa_first"] = (float(d["tcpa"][0])
                               if d.get("tcpa") else None)
            d["file"] = _episode_file_for(run_dir, int(d["episode_index"]), files)
            rows.append(d)
        return {"run": run, "path": run_dir, "source": source,
                "n_episode_files": len(files), "episodes": to_jsonable(rows)}

    # -- GET /api/runs/{run}/episode/{i} ----------------------------------

    def run_episode(self, run: str, index: int) -> dict[str, Any]:
        """Decimated replay payload for one stored episode."""
        run_dir = _safe_run_dir(run)
        files = _episode_files(run_dir)
        if not files:
            raise ApiError(f"run '{run}' contains no .npz episode files",
                           status=404, code="no_episodes")
        i = int(index)
        path = _episode_file_for(run_dir, i, files)
        if path is None:
            raise ApiError(f"episode {i} out of range (0..{len(files) - 1})",
                           status=404, code="episode_out_of_range")
        ep = _load_episode_arrays(path)
        stride = 1 if ep.n_steps <= 1200 else max(1, ep.n_steps // 1200)
        frames = _frames_from_arrays(ep, stride=stride)
        idx = self.run_index(run)
        meta = next((r for r in idx["episodes"]
                     if int(r.get("episode_index", -1)) == i), {})
        return {
            "run": run, "episode": i, "file": path,
            "n_steps": ep.n_steps, "n_agents": ep.n_agents,
            "stride": stride, "dt": LOG_DT * stride,
            "bounds": _bounds_of(ep),
            "gt_type": [int(v) for v in ep.gt_type.tolist()],
            "agent_ids": [int(v) for v in ep.agent_ids.tolist()],
            "meta": meta,
            "frames": frames,
        }

    # -- POST /api/diagnose ----------------------------------------------

    def diagnose(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Run the CV-extrapolation data-quality diagnosis over a run.

        Doc 8.3 makes this step 2 of the whole project: if the CV 15 s error is
        tiny the dataset is too easy and the paper collapses; if it is enormous
        the motion is noise.
        """
        run = str(body.get("run", ""))
        run_dir = _safe_run_dir(run)
        max_ep = body.get("max_episodes")
        max_ep = int(max_ep) if max_ep else None
        cvb = BRIDGE.module("analysis.cv_baseline")
        if cvb is not None and hasattr(cvb, "cv_errors") and hasattr(cvb, "diagnose"):
            try:
                d = _cv_diagnose_module(run_dir, max_ep)
                d["run"] = run
                return d
            except ApiError:
                raise
            except Exception as exc:
                fb = _cv_diagnose_fallback(run_dir,
                                           max_episodes=max_ep or 200)
                fb["run"] = run
                fb["module_error"] = f"{type(exc).__name__}: {exc}"
                return fb
        d = _cv_diagnose_fallback(run_dir, max_episodes=max_ep or 200)
        d["run"] = run
        return d

    # -- session access for the SSE handler -------------------------------

    def sim_session(self, sim_id: str) -> SimSession:
        """Public accessor used by :mod:`gui.server` to open an SSE stream."""
        return self._session(sim_id)

    def shutdown(self) -> None:
        """Close every live session (called on server shutdown)."""
        with self._lock:
            sims = list(self._sims.values())
            self._sims.clear()
        for s in sims:
            try:
                s.close()
            except Exception:
                pass
