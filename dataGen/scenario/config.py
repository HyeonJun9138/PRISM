"""Scenario configuration dataclasses with lossless JSON round-tripping.

Every knob that changes what a dataset contains lives here, so an episode can
be replayed from ``(ScenarioConfig, seed)`` alone (doc 10).

Serialisation rule: dictionaries keyed by an enum integer (platform mix,
intent mix, geometry weights, RCS factors) are written to JSON with STRING
keys, because JSON has no integer keys. ``from_dict`` accepts either form and
converts back to ``int``. Therefore ``to_dict(from_dict(d)) == d`` for any
``d`` produced by ``to_dict`` -- the round trip is lossless in both directions.

Units are SI: metres, seconds, radians, m/s. Angles follow the global
convention (CCW from +X = East, wrapped to (-pi, pi]).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.constants import DWC_HMD_M, LOG_DT, NMAC_RADIUS_M
from core.types import GeometryType, IntentType, PlatformType

__all__ = [
    "SensorConfig",
    "WindConfig",
    "EncounterConfig",
    "PlatformMixConfig",
    "IntentMixConfig",
    "ScenarioConfig",
    "load_scenario_config",
    "save_scenario_config",
]


# --------------------------------------------------------------------------
# int-keyed dict <-> JSON helpers
# --------------------------------------------------------------------------


def _weights_to_json(d: Mapping[int, float]) -> dict[str, float]:
    """Convert an int-keyed weight dict to a JSON-safe str-keyed dict."""
    return {str(int(k)): float(v) for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))}


def _weights_from_json(d: Mapping[Any, Any]) -> dict[int, float]:
    """Convert a str- or int-keyed weight mapping back to int keys."""
    return {int(k): float(v) for k, v in d.items()}


def _check_weights(
    weights: Mapping[int, float],
    enum_cls: type,
    label: str,
) -> list[str]:
    """Validate a weight table: known keys, non-negative values, positive sum."""
    problems: list[str] = []
    valid = {int(m) for m in enum_cls}
    if not weights:
        problems.append(f"{label}: weight table is empty")
        return problems
    for k, v in weights.items():
        if int(k) not in valid:
            problems.append(f"{label}: unknown key {k} for {enum_cls.__name__}")
        if not math.isfinite(float(v)):
            problems.append(f"{label}: weight for key {k} is not finite")
        elif float(v) < 0.0:
            problems.append(f"{label}: negative weight {v} for key {k}")
    total = sum(float(v) for v in weights.values() if math.isfinite(float(v)))
    if total <= 0.0:
        problems.append(f"{label}: weights sum to {total}, must be > 0")
    return problems


def _normalize(weights: Mapping[int, float]) -> dict[int, float]:
    """Return the weight table normalised to a probability distribution."""
    total = sum(float(v) for v in weights.values())
    if total <= 0.0:
        n = max(len(weights), 1)
        return {int(k): 1.0 / n for k in weights}
    return {int(k): float(v) / total for k, v in weights.items()}


# --------------------------------------------------------------------------
# Sensor
# --------------------------------------------------------------------------


def _default_rcs_factor() -> dict[int, float]:
    """Default per-platform detectability multipliers [-] (1.0 = nominal).

    Small, low-reflectivity bodies (birds, debris, small multirotors) are
    harder to hold on an onboard surveillance sensor than a GA aircraft.
    These are ASSUMED values (doc 12 requires them to be marked as such in the
    parameter-provenance table until sourced).
    """
    return {
        int(PlatformType.BIRD): 0.40,
        int(PlatformType.MULTIROTOR): 0.60,
        int(PlatformType.FIXED_WING_UAS): 0.80,
        int(PlatformType.EVTOL_UAM): 1.20,
        int(PlatformType.GA): 1.30,
        int(PlatformType.DEBRIS): 0.30,
        int(PlatformType.UNKNOWN): 1.00,
    }


@dataclass
class SensorConfig:
    """Onboard surveillance sensor model parameters (doc 3.5).

    The targets are non-cooperative, so there is no ADS-B / Remote ID: only a
    noisy, intermittent, delayed onboard track exists. Initial values should
    be traced to DO-365 sensor performance requirements before publication
    (doc 12.1).

    Attributes:
        enabled: if False the observation layer copies ground truth exactly
            (debug only -- never for dataset generation).
        sigma_r0_m: range noise floor [m].
        sigma_r_frac: range noise proportional term [-], added as
            ``sigma_r = sigma_r0_m + sigma_r_frac * range``.
        sigma_theta_rad: bearing noise standard deviation [rad].
        pd_max: maximum detection probability [-] in 0..1.
        r_full_m: range [m] within which Pd equals ``pd_max``.
        r_max_m: range [m] at which Pd has decayed to zero.
        update_rate_hz: sensor track update rate [Hz].
        async_phase: if True each track gets a random phase offset [-], so
            tracks do not update on the same tick.
        latency_s: fixed measurement latency [s].
        latency_jitter_s: standard deviation of latency jitter [s].
        drop_after_misses: consecutive misses [-] after which the track is
            dropped; a re-acquisition receives a NEW track_id.
        rcs_factor: per :class:`~core.types.PlatformType` detectability
            multiplier [-] applied to Pd.
        clutter_enabled: false-alarm injection (doc 3.5 second-tier extension).
            Keep False for the first paper.
    """

    enabled: bool = True
    sigma_r0_m: float = 3.0
    sigma_r_frac: float = 0.01
    sigma_theta_rad: float = 0.0087
    pd_max: float = 0.95
    r_full_m: float = 800.0
    r_max_m: float = 3000.0
    update_rate_hz: float = 5.0
    async_phase: bool = True
    latency_s: float = 0.2
    latency_jitter_s: float = 0.05
    drop_after_misses: int = 6
    rcs_factor: dict[int, float] = field(default_factory=_default_rcs_factor)
    clutter_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "enabled": bool(self.enabled),
            "sigma_r0_m": float(self.sigma_r0_m),
            "sigma_r_frac": float(self.sigma_r_frac),
            "sigma_theta_rad": float(self.sigma_theta_rad),
            "pd_max": float(self.pd_max),
            "r_full_m": float(self.r_full_m),
            "r_max_m": float(self.r_max_m),
            "update_rate_hz": float(self.update_rate_hz),
            "async_phase": bool(self.async_phase),
            "latency_s": float(self.latency_s),
            "latency_jitter_s": float(self.latency_jitter_s),
            "drop_after_misses": int(self.drop_after_misses),
            "rcs_factor": _weights_to_json(self.rcs_factor),
            "clutter_enabled": bool(self.clutter_enabled),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SensorConfig":
        """Rebuild from :meth:`to_dict` output (missing keys take defaults)."""
        base = cls()
        rcs = d.get("rcs_factor")
        return cls(
            enabled=bool(d.get("enabled", base.enabled)),
            sigma_r0_m=float(d.get("sigma_r0_m", base.sigma_r0_m)),
            sigma_r_frac=float(d.get("sigma_r_frac", base.sigma_r_frac)),
            sigma_theta_rad=float(d.get("sigma_theta_rad", base.sigma_theta_rad)),
            pd_max=float(d.get("pd_max", base.pd_max)),
            r_full_m=float(d.get("r_full_m", base.r_full_m)),
            r_max_m=float(d.get("r_max_m", base.r_max_m)),
            update_rate_hz=float(d.get("update_rate_hz", base.update_rate_hz)),
            async_phase=bool(d.get("async_phase", base.async_phase)),
            latency_s=float(d.get("latency_s", base.latency_s)),
            latency_jitter_s=float(d.get("latency_jitter_s", base.latency_jitter_s)),
            drop_after_misses=int(d.get("drop_after_misses", base.drop_after_misses)),
            rcs_factor=_weights_from_json(rcs) if rcs is not None else base.rcs_factor,
            clutter_enabled=bool(d.get("clutter_enabled", base.clutter_enabled)),
        )

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        p: list[str] = []
        if self.sigma_r0_m < 0.0:
            p.append("sensor.sigma_r0_m must be >= 0 m")
        if self.sigma_r_frac < 0.0:
            p.append("sensor.sigma_r_frac must be >= 0")
        if self.sigma_theta_rad < 0.0:
            p.append("sensor.sigma_theta_rad must be >= 0 rad")
        if not (0.0 < self.pd_max <= 1.0):
            p.append("sensor.pd_max must be in (0, 1]")
        if self.r_full_m <= 0.0:
            p.append("sensor.r_full_m must be > 0 m")
        if self.r_max_m <= self.r_full_m:
            p.append("sensor.r_max_m must be > sensor.r_full_m")
        if self.update_rate_hz <= 0.0:
            p.append("sensor.update_rate_hz must be > 0 Hz")
        if self.latency_s < 0.0:
            p.append("sensor.latency_s must be >= 0 s")
        if self.latency_jitter_s < 0.0:
            p.append("sensor.latency_jitter_s must be >= 0 s")
        if self.drop_after_misses < 1:
            p.append("sensor.drop_after_misses must be >= 1")
        p.extend(_check_weights(self.rcs_factor, PlatformType, "sensor.rcs_factor"))
        for k, v in self.rcs_factor.items():
            if float(v) <= 0.0:
                p.append(f"sensor.rcs_factor[{k}] must be > 0")
        return p


# --------------------------------------------------------------------------
# Wind
# --------------------------------------------------------------------------


@dataclass
class WindConfig:
    """Steady wind (and optional gust) sampling ranges (doc 3.6).

    Wind is what makes a constant-velocity extrapolator miss systematically,
    and is the physical origin of aleatoric uncertainty in the dataset.

    Attributes:
        speed_min_ms: minimum steady wind speed [m/s].
        speed_max_ms: maximum steady wind speed [m/s].
        dir_min_rad: minimum wind direction [rad], CCW from +X (East), giving
            the direction the air moves TOWARD.
        dir_max_rad: maximum wind direction [rad], same convention.
        gust_enabled: enable the Ornstein-Uhlenbeck gust process (second-tier
            extension; keep False for the first paper).
        gust_sigma_ms: stationary standard deviation of the gust [m/s].
        gust_tau_s: gust correlation time [s].
    """

    speed_min_ms: float = 0.0
    speed_max_ms: float = 8.0
    dir_min_rad: float = -math.pi
    dir_max_rad: float = math.pi
    gust_enabled: bool = False
    gust_sigma_ms: float = 1.5
    gust_tau_s: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "speed_min_ms": float(self.speed_min_ms),
            "speed_max_ms": float(self.speed_max_ms),
            "dir_min_rad": float(self.dir_min_rad),
            "dir_max_rad": float(self.dir_max_rad),
            "gust_enabled": bool(self.gust_enabled),
            "gust_sigma_ms": float(self.gust_sigma_ms),
            "gust_tau_s": float(self.gust_tau_s),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WindConfig":
        """Rebuild from :meth:`to_dict` output (missing keys take defaults)."""
        base = cls()
        return cls(
            speed_min_ms=float(d.get("speed_min_ms", base.speed_min_ms)),
            speed_max_ms=float(d.get("speed_max_ms", base.speed_max_ms)),
            dir_min_rad=float(d.get("dir_min_rad", base.dir_min_rad)),
            dir_max_rad=float(d.get("dir_max_rad", base.dir_max_rad)),
            gust_enabled=bool(d.get("gust_enabled", base.gust_enabled)),
            gust_sigma_ms=float(d.get("gust_sigma_ms", base.gust_sigma_ms)),
            gust_tau_s=float(d.get("gust_tau_s", base.gust_tau_s)),
        )

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        p: list[str] = []
        if self.speed_min_ms < 0.0:
            p.append("wind.speed_min_ms must be >= 0 m/s")
        if self.speed_max_ms < self.speed_min_ms:
            p.append("wind.speed_max_ms must be >= wind.speed_min_ms")
        if self.dir_max_rad < self.dir_min_rad:
            p.append("wind.dir_max_rad must be >= wind.dir_min_rad")
        if (self.dir_max_rad - self.dir_min_rad) > (2.0 * math.pi + 1e-9):
            p.append("wind direction range spans more than 2*pi rad")
        if self.gust_enabled:
            if self.gust_sigma_ms < 0.0:
                p.append("wind.gust_sigma_ms must be >= 0 m/s")
            if self.gust_tau_s <= 0.0:
                p.append("wind.gust_tau_s must be > 0 s when gusts are enabled")
        return p


# --------------------------------------------------------------------------
# Encounter geometry
# --------------------------------------------------------------------------


def _default_geometry_weights() -> dict[int, float]:
    """Default sampling weights over encounter geometries [-] (doc 3.4/3.7)."""
    return {
        int(GeometryType.HEAD_ON): 0.30,
        int(GeometryType.CROSSING): 0.45,
        int(GeometryType.OVERTAKING): 0.25,
        int(GeometryType.BACKGROUND): 0.0,
    }


@dataclass
class EncounterConfig:
    """Inverse-design parameters for designed encounters (doc 3.4).

    Random placement almost never produces a risky encounter, so the target
    CPA distance and time-to-CPA are sampled FIRST and the initial conditions
    are solved backwards from them.

    Attributes:
        n_designed_min: minimum number of designed encounters per episode [-].
            **0, deliberately.** With a floor of 1 every episode is guaranteed
            a designed conflict, ``dwc_violated`` is True in 100% of episodes
            and :class:`~core.types.RiskLevel` ``NONE`` is an unreachable enum
            member -- a nominal, no-conflict episode does not exist anywhere in
            the dataset. Doc 3.4 asks for a risk-STRATIFIED set, and a
            downstream risk-map / PPO consumer that sees a conflict 100% of the
            time cannot learn not to evade phantoms.
        n_designed_max: maximum number of designed encounters per episode [-].
        cpa_dist_min_m: minimum target closest-point-of-approach distance [m].
        cpa_dist_max_m: maximum target CPA distance [m]. The range must
            straddle the NMAC radius so that both violating and non-violating
            cases occur for stratified analysis (doc 3.4), and it must reach
            ABOVE :data:`core.constants.DWC_HMD_M` (1219.2 m) or every designed
            encounter is inside the well-clear threshold before the simulation
            starts and the LOW stratum is unreachable too.
        tcpa_min_s: minimum time to CPA [s] from episode start.
        tcpa_max_s: maximum time to CPA [s] from episode start.
        geometry_weights: sampling weights [-] keyed by
            :class:`~core.types.GeometryType`.
    """

    n_designed_min: int = 0
    n_designed_max: int = 3
    cpa_dist_min_m: float = 30.0
    cpa_dist_max_m: float = 2500.0
    tcpa_min_s: float = 15.0
    tcpa_max_s: float = 45.0
    geometry_weights: dict[int, float] = field(
        default_factory=_default_geometry_weights
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "n_designed_min": int(self.n_designed_min),
            "n_designed_max": int(self.n_designed_max),
            "cpa_dist_min_m": float(self.cpa_dist_min_m),
            "cpa_dist_max_m": float(self.cpa_dist_max_m),
            "tcpa_min_s": float(self.tcpa_min_s),
            "tcpa_max_s": float(self.tcpa_max_s),
            "geometry_weights": _weights_to_json(self.geometry_weights),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EncounterConfig":
        """Rebuild from :meth:`to_dict` output (missing keys take defaults)."""
        base = cls()
        gw = d.get("geometry_weights")
        return cls(
            n_designed_min=int(d.get("n_designed_min", base.n_designed_min)),
            n_designed_max=int(d.get("n_designed_max", base.n_designed_max)),
            cpa_dist_min_m=float(d.get("cpa_dist_min_m", base.cpa_dist_min_m)),
            cpa_dist_max_m=float(d.get("cpa_dist_max_m", base.cpa_dist_max_m)),
            tcpa_min_s=float(d.get("tcpa_min_s", base.tcpa_min_s)),
            tcpa_max_s=float(d.get("tcpa_max_s", base.tcpa_max_s)),
            geometry_weights=(
                _weights_from_json(gw) if gw is not None else base.geometry_weights
            ),
        )

    def geometry_probabilities(self) -> dict[int, float]:
        """Geometry weights normalised to probabilities [-]."""
        return _normalize(self.geometry_weights)

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        p: list[str] = []
        if self.n_designed_min < 0:
            p.append("encounter.n_designed_min must be >= 0")
        if self.n_designed_max < self.n_designed_min:
            p.append("encounter.n_designed_max must be >= n_designed_min")
        if self.cpa_dist_min_m < 0.0:
            p.append("encounter.cpa_dist_min_m must be >= 0 m")
        if self.cpa_dist_max_m <= self.cpa_dist_min_m:
            p.append("encounter.cpa_dist_max_m must be > cpa_dist_min_m")
        if self.tcpa_min_s <= 0.0:
            p.append("encounter.tcpa_min_s must be > 0 s")
        if self.tcpa_max_s < self.tcpa_min_s:
            p.append("encounter.tcpa_max_s must be >= tcpa_min_s")
        if not (self.cpa_dist_min_m < NMAC_RADIUS_M < self.cpa_dist_max_m):
            p.append(
                "encounter CPA range must straddle NMAC_RADIUS_M "
                f"({NMAC_RADIUS_M} m) so violating and non-violating cases "
                "both occur (doc 3.4)"
            )
        if self.cpa_dist_max_m <= DWC_HMD_M:
            p.append(
                "encounter.cpa_dist_max_m must exceed DWC_HMD_M "
                f"({DWC_HMD_M} m): with every designed CPA inside the "
                "well-clear threshold, RiskLevel LOW and NONE are unreachable "
                "and the dataset is 2-level with a 4-level label (doc 3.4)"
            )
        p.extend(
            _check_weights(
                self.geometry_weights, GeometryType, "encounter.geometry_weights"
            )
        )
        return p


# --------------------------------------------------------------------------
# Mixes
# --------------------------------------------------------------------------


def _default_platform_weights() -> dict[int, float]:
    """Default training platform mix [-] (doc 3.7 / 8.1).

    BIRD, GA and DEBRIS carry zero weight: they are reserved for the
    out-of-distribution split and must be switched on by a preset, never by
    the default training config.
    """
    return {
        int(PlatformType.BIRD): 0.0,
        int(PlatformType.MULTIROTOR): 0.40,
        int(PlatformType.FIXED_WING_UAS): 0.35,
        int(PlatformType.EVTOL_UAM): 0.25,
        int(PlatformType.GA): 0.0,
        int(PlatformType.DEBRIS): 0.0,
    }


def _default_intent_weights() -> dict[int, float]:
    """Default training intent mix [-] (doc 3.7 / 8.1).

    SEARCH_WANDER, REACTIVE_AVOID and PASSIVE_DRIFT are OOD-only and carry
    zero weight here.
    """
    return {
        int(IntentType.CORRIDOR_TRANSIT): 0.40,
        int(IntentType.DELIVERY_WP): 0.35,
        int(IntentType.LOITER_HOVER): 0.25,
        int(IntentType.SEARCH_WANDER): 0.0,
        int(IntentType.REACTIVE_AVOID): 0.0,
        int(IntentType.PASSIVE_DRIFT): 0.0,
    }


@dataclass
class PlatformMixConfig:
    """Categorical distribution over platform types.

    Attributes:
        weights: unnormalised weights [-] keyed by
            :class:`~core.types.PlatformType`.
    """

    weights: dict[int, float] = field(default_factory=_default_platform_weights)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {"weights": _weights_to_json(self.weights)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PlatformMixConfig":
        """Rebuild from :meth:`to_dict` output (missing keys take defaults)."""
        w = d.get("weights")
        if w is None:
            return cls()
        return cls(weights=_weights_from_json(w))

    def probabilities(self) -> dict[int, float]:
        """Weights normalised to probabilities [-]."""
        return _normalize(self.weights)

    def active_types(self) -> list[int]:
        """Platform-type integers with strictly positive weight."""
        return [int(k) for k, v in sorted(self.weights.items()) if float(v) > 0.0]

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        return _check_weights(self.weights, PlatformType, "platform_mix.weights")


@dataclass
class IntentMixConfig:
    """Categorical distribution over intent types.

    Attributes:
        weights: unnormalised weights [-] keyed by
            :class:`~core.types.IntentType`.
    """

    weights: dict[int, float] = field(default_factory=_default_intent_weights)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {"weights": _weights_to_json(self.weights)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "IntentMixConfig":
        """Rebuild from :meth:`to_dict` output (missing keys take defaults)."""
        w = d.get("weights")
        if w is None:
            return cls()
        return cls(weights=_weights_from_json(w))

    def probabilities(self) -> dict[int, float]:
        """Weights normalised to probabilities [-]."""
        return _normalize(self.weights)

    def active_types(self) -> list[int]:
        """Intent-type integers with strictly positive weight."""
        return [int(k) for k, v in sorted(self.weights.items()) if float(v) > 0.0]

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        return _check_weights(self.weights, IntentType, "intent_mix.weights")


# --------------------------------------------------------------------------
# Top-level scenario
# --------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Full description of one episode family (doc 3.7 minimum configuration).

    Attributes:
        name: human-readable scenario / preset name.
        seed: master seed [-] handed to :class:`core.rng.RNGHub`.
        duration_s: episode duration [s]; must be an integer multiple of
            ``LOG_DT`` so the logged step count is exact.
        area_km2: simulated square region area [km^2]; the side length is
            ``sqrt(area_km2)`` km.
        density_per_km2: traffic density [aircraft/km^2] used to size the
            background population (doc 8.1 splits on this axis).
        n_agents_min: hard floor on the number of traffic agents [-].
        n_agents_max: hard ceiling on the number of traffic agents [-].
        ownship_speed_ms: nominal ownship cruise airspeed [m/s].
        platform_mix: platform-type distribution.
        intent_mix: intent-type distribution.
        encounter: designed-encounter inverse-design parameters.
        sensor: onboard sensor model parameters.
        wind: wind sampling ranges.
        corridor_enabled: place a nominal cruise corridor as static context.
        nfz_count: number of no-fly zones [-] used as static context.
    """

    name: str = "default"
    seed: int = 0
    duration_s: float = 60.0
    area_km2: float = 4.0
    density_per_km2: float = 40.0
    n_agents_min: int = 5
    n_agents_max: int = 15
    ownship_speed_ms: float = 55.0
    platform_mix: PlatformMixConfig = field(default_factory=PlatformMixConfig)
    intent_mix: IntentMixConfig = field(default_factory=IntentMixConfig)
    encounter: EncounterConfig = field(default_factory=EncounterConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    corridor_enabled: bool = True
    nfz_count: int = 2
    #: rev.3 reaction layer (agents/reaction.py): every powered platform
    #: evades threatening traffic. Off only for ablation studies.
    reaction_enabled: bool = True

    # -- derived quantities ----------------------------------------------

    def n_background(self) -> int:
        """Number of traffic agents [-] implied by density, area and bounds.

        Returns:
            ``round(density_per_km2 * area_km2)`` clipped to
            ``[n_agents_min, n_agents_max]``.
        """
        raw = int(round(float(self.density_per_km2) * float(self.area_km2)))
        return max(int(self.n_agents_min), min(int(self.n_agents_max), raw))

    def region_side_m(self) -> float:
        """Side length [m] of the square simulation region."""
        return math.sqrt(max(float(self.area_km2), 0.0)) * 1000.0

    def n_logged_steps(self) -> int:
        """Number of logged samples [-] in one episode (``T`` in the schema)."""
        return int(round(float(self.duration_s) / LOG_DT))

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole scenario to a JSON-safe nested dict."""
        return {
            "name": str(self.name),
            "seed": int(self.seed),
            "duration_s": float(self.duration_s),
            "area_km2": float(self.area_km2),
            "density_per_km2": float(self.density_per_km2),
            "n_agents_min": int(self.n_agents_min),
            "n_agents_max": int(self.n_agents_max),
            "ownship_speed_ms": float(self.ownship_speed_ms),
            "platform_mix": self.platform_mix.to_dict(),
            "intent_mix": self.intent_mix.to_dict(),
            "encounter": self.encounter.to_dict(),
            "sensor": self.sensor.to_dict(),
            "wind": self.wind.to_dict(),
            "corridor_enabled": bool(self.corridor_enabled),
            "nfz_count": int(self.nfz_count),
            "reaction_enabled": bool(self.reaction_enabled),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScenarioConfig":
        """Rebuild from :meth:`to_dict` output (missing keys take defaults)."""
        base = cls()
        return cls(
            name=str(d.get("name", base.name)),
            seed=int(d.get("seed", base.seed)),
            duration_s=float(d.get("duration_s", base.duration_s)),
            area_km2=float(d.get("area_km2", base.area_km2)),
            density_per_km2=float(d.get("density_per_km2", base.density_per_km2)),
            n_agents_min=int(d.get("n_agents_min", base.n_agents_min)),
            n_agents_max=int(d.get("n_agents_max", base.n_agents_max)),
            ownship_speed_ms=float(d.get("ownship_speed_ms", base.ownship_speed_ms)),
            platform_mix=PlatformMixConfig.from_dict(d.get("platform_mix", {})),
            intent_mix=IntentMixConfig.from_dict(d.get("intent_mix", {})),
            encounter=EncounterConfig.from_dict(d.get("encounter", {})),
            sensor=SensorConfig.from_dict(d.get("sensor", {})),
            wind=WindConfig.from_dict(d.get("wind", {})),
            corridor_enabled=bool(d.get("corridor_enabled", base.corridor_enabled)),
            nfz_count=int(d.get("nfz_count", base.nfz_count)),
            reaction_enabled=bool(d.get("reaction_enabled", base.reaction_enabled)),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string (UTF-8 safe, ASCII not forced)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ScenarioConfig":
        """Parse a JSON string produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def copy_with(self, **overrides: Any) -> "ScenarioConfig":
        """Return a copy of this config with top-level fields replaced.

        Args:
            **overrides: any top-level field name of :class:`ScenarioConfig`.

        Returns:
            A new config; nested sub-configs are rebuilt from a dict round
            trip, so the copy shares no mutable state with the original.
        """
        d = self.to_dict()
        for k, v in overrides.items():
            if k not in d:
                raise KeyError(f"unknown ScenarioConfig field: {k}")
            d[k] = v.to_dict() if hasattr(v, "to_dict") else v
        return ScenarioConfig.from_dict(d)

    # -- validation -------------------------------------------------------

    def validate(self) -> list[str]:
        """Return every problem found; an empty list means the config is OK."""
        p: list[str] = []
        if not str(self.name).strip():
            p.append("name must be a non-empty string")
        if int(self.seed) < 0:
            p.append("seed must be >= 0")
        if self.duration_s <= 0.0:
            p.append("duration_s must be > 0 s")
        else:
            steps = self.duration_s / LOG_DT
            if abs(steps - round(steps)) > 1e-9:
                p.append(f"duration_s must be an integer multiple of LOG_DT={LOG_DT} s")
        if self.area_km2 <= 0.0:
            p.append("area_km2 must be > 0 km^2")
        if self.density_per_km2 < 0.0:
            p.append("density_per_km2 must be >= 0 per km^2")
        if self.n_agents_min < 1:
            p.append("n_agents_min must be >= 1")
        if self.n_agents_max < self.n_agents_min:
            p.append("n_agents_max must be >= n_agents_min")
        if self.ownship_speed_ms <= 0.0:
            p.append("ownship_speed_ms must be > 0 m/s")
        if self.nfz_count < 0:
            p.append("nfz_count must be >= 0")
        if self.encounter.n_designed_max > self.n_agents_max:
            p.append(
                "encounter.n_designed_max exceeds n_agents_max: there would "
                "not be enough agents to host the designed encounters"
            )
        if self.encounter.tcpa_max_s > self.duration_s:
            p.append("encounter.tcpa_max_s exceeds duration_s: CPA would never occur")
        p.extend(self.platform_mix.validate())
        p.extend(self.intent_mix.validate())
        p.extend(self.encounter.validate())
        p.extend(self.sensor.validate())
        p.extend(self.wind.validate())
        return p


# --------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------


def load_scenario_config(path: str) -> ScenarioConfig:
    """Load a :class:`ScenarioConfig` from a UTF-8 JSON file.

    Args:
        path: filesystem path (forward slashes are fine on Windows).

    Returns:
        The parsed configuration.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return ScenarioConfig.from_dict(json.load(fh))


def save_scenario_config(cfg: ScenarioConfig, path: str, indent: int = 2) -> None:
    """Write a :class:`ScenarioConfig` to a UTF-8 JSON file.

    Args:
        cfg: configuration to write.
        path: destination path; parent directories are created if missing.
        indent: JSON indentation width [-].
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
