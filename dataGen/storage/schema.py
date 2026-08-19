"""On-disk dataset contract: array layout, NaN policy and episode metadata.

Doc section [4] is explicit: **ground truth and observations are stored
separately**. The learning input is the observation layer (noisy, missing,
delayed); the target and the evaluation reference is the ground-truth layer.
Nothing in the observation layer may be interpolated to hide a miss -- a gap
must stay a gap so the predictor learns to widen sigma when the track breaks
(doc 3.5).

Shapes use two symbols:
    T -- number of logged steps  (duration_s / LOG_DT, e.g. 60 s -> 120)
    N -- agent slot capacity     (max traffic agents; unused slots are padding)

Storage dtypes are float32 for all continuous quantities (internal
computation stays float64), int8 for enum labels, int32 for ids and counters,
bool for masks.

NaN / sentinel policy enforced by :func:`validate`:
    * ``gt_pos`` / ``gt_vel`` / ``gt_psi``  : finite where ``gt_valid``,
      NaN elsewhere (no stale values in padding slots).
    * ``gt_mode`` / ``gt_intent``           : valid enum where ``gt_valid``,
      ``-1`` elsewhere.
    * ``gt_type``                           : valid enum for used slots,
      ``-1`` for unused slots.
    * ``obs_pos`` / ``obs_R``               : finite where ``obs_detected``,
      NaN elsewhere -- NEVER interpolated.
    * ``obs_t_valid``                       : finite where ``obs_detected``,
      NaN elsewhere; the time the measurement was actually true, which is
      EARLIER than the logged time by the sensor latency plus the age of the
      measurement inside the logging interval.
    * ``obs_track_id``                      : ``>= 0`` wherever a track is
      live (a detection requires a live track); ``-1`` when no live track.
      A live track MAY have ``obs_detected == False`` on a missed update.
    * ``obs_track_age``                     : ``>= 0``; exactly ``0`` where
      ``obs_track_id == -1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.types import GeometryType, IntentType, ManeuverMode, PlatformType, RiskLevel

__all__ = [
    "EPISODE_ARRAY_SPEC",
    "EpisodeArrays",
    "EpisodeMeta",
    "validate",
]


#: name -> (dtype, symbolic shape). 'T' and 'N' are resolved per episode.
EPISODE_ARRAY_SPEC: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "t": (np.float32, ("T",)),
    "own_pos": (np.float32, ("T", 2)),
    "own_vel": (np.float32, ("T", 2)),
    "own_psi": (np.float32, ("T",)),
    "gt_valid": (np.bool_, ("T", "N")),
    "gt_pos": (np.float32, ("T", "N", 2)),
    "gt_vel": (np.float32, ("T", "N", 2)),
    "gt_psi": (np.float32, ("T", "N")),
    "gt_mode": (np.int8, ("T", "N")),
    "gt_intent": (np.int8, ("T", "N")),
    "gt_type": (np.int8, ("N",)),
    "obs_detected": (np.bool_, ("T", "N")),
    "obs_pos": (np.float32, ("T", "N", 2)),
    "obs_R": (np.float32, ("T", "N", 2, 2)),
    "obs_t_valid": (np.float32, ("T", "N")),
    "obs_track_id": (np.int32, ("T", "N")),
    "obs_track_age": (np.int32, ("T", "N")),
    "agent_ids": (np.int32, ("N",)),
}


@dataclass
class EpisodeArrays:
    """All logged arrays of a single episode.

    Attributes:
        t: [T] float32 -- logged time [s] since episode start.
        own_pos: [T,2] float32 -- ownship position [m], (East, North).
        own_vel: [T,2] float32 -- ownship GROUND velocity [m/s].
        own_psi: [T] float32 -- ownship heading [rad], CCW from +X, in (-pi, pi].
        gt_valid: [T,N] bool -- agent slot occupied and alive at this step.
        gt_pos: [T,N,2] float32 -- true agent position [m].
        gt_vel: [T,N,2] float32 -- true agent GROUND velocity [m/s].
        gt_psi: [T,N] float32 -- true agent heading [rad].
        gt_mode: [T,N] int8 -- :class:`~core.types.ManeuverMode`, -1 if invalid.
        gt_intent: [T,N] int8 -- :class:`~core.types.IntentType`, -1 if invalid.
        gt_type: [N] int8 -- :class:`~core.types.PlatformType`, -1 if the slot
            is unused. Constant over time (an agent never changes airframe).
        obs_detected: [T,N] bool -- a measurement exists at this step.
        obs_pos: [T,N,2] float32 -- measured position [m], NaN where not
            detected. NEVER interpolate across a gap (doc 3.5).
        obs_R: [T,N,2,2] float32 -- measurement covariance [m^2] in Cartesian
            world axes, NaN where not detected.
        obs_t_valid: [T,N] float32 -- the time [s] at which ``obs_pos`` was
            actually true, NaN where not detected. It is EARLIER than ``t``:
            the sensor runs at ``update_rate_hz`` and reports through a
            latency ``tau``, so the measurement folded into logged step ``t``
            was emitted somewhere inside ``(t - LOG_DT, t]`` and was true at
            ``t_emit - tau``. ``obs_R`` describes ONLY the polar measurement
            noise at ``t_valid``; the staleness ``t - t_valid`` is a separate,
            recoverable error term. Storing it is what keeps the observation
            layer's uncertainty budget complete -- without it a consumer
            silently treats a stale position as a current one and the input
            layer's own covariance is optimistic by roughly 50% in variance.
        obs_track_id: [T,N] int32 -- tracker id [-]; -1 when no live track. A
            re-acquisition after a drop gets a NEW id.
        obs_track_age: [T,N] int32 -- number of logged steps [-] the track has
            existed; 0 when no live track.
        agent_ids: [N] int32 -- stable ground-truth agent id per slot; -1 for
            unused slots.
    """

    t: np.ndarray
    own_pos: np.ndarray
    own_vel: np.ndarray
    own_psi: np.ndarray
    gt_valid: np.ndarray
    gt_pos: np.ndarray
    gt_vel: np.ndarray
    gt_psi: np.ndarray
    gt_mode: np.ndarray
    gt_intent: np.ndarray
    gt_type: np.ndarray
    obs_detected: np.ndarray
    obs_pos: np.ndarray
    obs_R: np.ndarray
    obs_t_valid: np.ndarray
    obs_track_id: np.ndarray
    obs_track_age: np.ndarray
    agent_ids: np.ndarray

    # -- construction -----------------------------------------------------

    @classmethod
    def allocate(cls, n_steps: int, n_agents: int) -> "EpisodeArrays":
        """Allocate an empty, contract-compliant episode buffer.

        Continuous arrays start as NaN and label arrays as -1, so a slot that
        is never written cannot masquerade as a valid zero-valued sample.

        Args:
            n_steps: T, number of logged steps [-].
            n_agents: N, agent slot capacity [-].

        Returns:
            A zero-filled / NaN-filled :class:`EpisodeArrays`.
        """
        T = int(n_steps)
        N = int(n_agents)
        nan = float("nan")
        return cls(
            t=np.zeros((T,), dtype=np.float32),
            own_pos=np.zeros((T, 2), dtype=np.float32),
            own_vel=np.zeros((T, 2), dtype=np.float32),
            own_psi=np.zeros((T,), dtype=np.float32),
            gt_valid=np.zeros((T, N), dtype=np.bool_),
            gt_pos=np.full((T, N, 2), nan, dtype=np.float32),
            gt_vel=np.full((T, N, 2), nan, dtype=np.float32),
            gt_psi=np.full((T, N), nan, dtype=np.float32),
            gt_mode=np.full((T, N), -1, dtype=np.int8),
            gt_intent=np.full((T, N), -1, dtype=np.int8),
            gt_type=np.full((N,), -1, dtype=np.int8),
            obs_detected=np.zeros((T, N), dtype=np.bool_),
            obs_pos=np.full((T, N, 2), nan, dtype=np.float32),
            obs_R=np.full((T, N, 2, 2), nan, dtype=np.float32),
            obs_t_valid=np.full((T, N), nan, dtype=np.float32),
            obs_track_id=np.full((T, N), -1, dtype=np.int32),
            obs_track_age=np.zeros((T, N), dtype=np.int32),
            agent_ids=np.full((N,), -1, dtype=np.int32),
        )

    # -- shape helpers ----------------------------------------------------

    @property
    def n_steps(self) -> int:
        """T, the number of logged steps [-]."""
        return int(self.t.shape[0])

    @property
    def n_agents(self) -> int:
        """N, the agent slot capacity [-]."""
        return int(self.gt_valid.shape[1])

    def used_slots(self) -> np.ndarray:
        """Boolean [N] mask of slots that are valid at least once."""
        return np.any(self.gt_valid, axis=0)

    # -- serialisation ----------------------------------------------------

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Return ``{name: array}`` in the exact order of the schema spec."""
        return {name: getattr(self, name) for name in EPISODE_ARRAY_SPEC}

    @classmethod
    def from_npz_dict(cls, d: Mapping[str, Any]) -> "EpisodeArrays":
        """Rebuild from a mapping produced by :meth:`to_npz_dict`.

        Arrays are cast to the schema dtype, so a loader that widened dtypes
        (e.g. ``np.load`` of an old file) still produces a valid object.
        """
        kwargs = {}
        for name, (dtype, _shape) in EPISODE_ARRAY_SPEC.items():
            kwargs[name] = np.asarray(d[name], dtype=dtype)
        return cls(**kwargs)


@dataclass
class EpisodeMeta:
    """Per-episode metadata for stratified splitting and analysis (doc 4).

    Attributes:
        episode_index: index [-] of the episode within its dataset.
        seed: master seed [-] used to generate the episode (replay key).
        density: the doc 8.1 density SPLIT AXIS [aircraft/km^2]. This is the
            knob that sets how many aircraft the episode contains, defined
            over ``ScenarioConfig.area_km2`` (a 0.25 km^2 reference
            footprint), NOT the areal density of the simulated arena. It is
            roughly 60x larger than what the scene actually contains and must
            never be quoted as a scene property -- use
            ``realised_density_per_km2`` for that.
        realised_density_per_km2: the density [aircraft/km^2] actually
            realised inside the simulated arena (``n_agents`` divided by the
            arena area). This is the number to report in any spacing-related
            paper text.
        wind_speed: steady wind speed [m/s].
        wind_dir: steady wind direction [rad], CCW from +X (East), toward
            which the air moves.
        n_agents: number of traffic agents actually simulated [-].
        geometry_type: :class:`~core.types.GeometryType` per designed
            encounter, in encounter order.
        cpa_dist: achieved closest-point-of-approach distance [m] per designed
            encounter, in the same order as ``geometry_type``.
        tcpa: time of closest approach [s] since episode start, per designed
            encounter.
        min_separation_m: minimum ownship-to-any-agent horizontal distance [m]
            over the whole episode.
        risk_level: :class:`~core.types.RiskLevel` label of the episode.
        platform_counts: number of agents [-] per platform-type integer.
        intent_counts: number of agents [-] per intent-type integer.
        nmac_occurred: True if horizontal separation fell below the NMAC
            radius. Measured at the 20 Hz integration rate, not at the 2 Hz
            logging rate -- see ``min_separation_m``.
        dwc_violated: True if DAA Well Clear was ACTUALLY LOST, i.e. the
            achieved ``min_separation_m`` fell below ``DWC_HMD_M``.
        dwc_alerted: True if the DAA Well Clear *alerting* predicate fired at
            any logged step: the constant-velocity projection of the relative
            state had a horizontal miss below ``DWC_HMD_M`` arriving within
            ``DWC_TAU_MOD_S``. This is a much wider net than
            ``dwc_violated`` -- in a 36 km^2 arena it fires in nearly every
            episode -- and the two must not be confused.
        duration_s: episode duration [s].
    """

    episode_index: int = 0
    seed: int = 0
    density: float = 0.0
    realised_density_per_km2: float = 0.0
    wind_speed: float = 0.0
    wind_dir: float = 0.0
    n_agents: int = 0
    geometry_type: list[int] = field(default_factory=list)
    cpa_dist: list[float] = field(default_factory=list)
    tcpa: list[float] = field(default_factory=list)
    min_separation_m: float = float("inf")
    risk_level: int = int(RiskLevel.NONE)
    platform_counts: dict[int, int] = field(default_factory=dict)
    intent_counts: dict[int, int] = field(default_factory=dict)
    nmac_occurred: bool = False
    dwc_violated: bool = False
    dwc_alerted: bool = False
    duration_s: float = 0.0
    #: rev.3: reaction-layer latches armed during the logged episode [-].
    reaction_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (int-keyed counts become str keys).

        ``min_separation_m`` is written as ``None`` when it is infinite,
        because JSON has no infinity literal that round-trips portably.
        """
        msep = float(self.min_separation_m)
        return {
            "episode_index": int(self.episode_index),
            "seed": int(self.seed),
            "density": float(self.density),
            "realised_density_per_km2": float(self.realised_density_per_km2),
            "wind_speed": float(self.wind_speed),
            "wind_dir": float(self.wind_dir),
            "n_agents": int(self.n_agents),
            "geometry_type": [int(g) for g in self.geometry_type],
            "cpa_dist": [float(c) for c in self.cpa_dist],
            "tcpa": [float(t) for t in self.tcpa],
            "min_separation_m": None if not np.isfinite(msep) else msep,
            "risk_level": int(self.risk_level),
            "platform_counts": {str(int(k)): int(v)
                                for k, v in sorted(self.platform_counts.items())},
            "intent_counts": {str(int(k)): int(v)
                              for k, v in sorted(self.intent_counts.items())},
            "nmac_occurred": bool(self.nmac_occurred),
            "dwc_violated": bool(self.dwc_violated),
            "dwc_alerted": bool(self.dwc_alerted),
            "duration_s": float(self.duration_s),
            "reaction_events": int(self.reaction_events),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EpisodeMeta":
        """Rebuild from :meth:`to_dict` output (missing keys take defaults)."""
        msep = d.get("min_separation_m", None)
        return cls(
            episode_index=int(d.get("episode_index", 0)),
            seed=int(d.get("seed", 0)),
            density=float(d.get("density", 0.0)),
            realised_density_per_km2=float(d.get("realised_density_per_km2", 0.0)),
            wind_speed=float(d.get("wind_speed", 0.0)),
            wind_dir=float(d.get("wind_dir", 0.0)),
            n_agents=int(d.get("n_agents", 0)),
            geometry_type=[int(g) for g in d.get("geometry_type", [])],
            cpa_dist=[float(c) for c in d.get("cpa_dist", [])],
            tcpa=[float(t) for t in d.get("tcpa", [])],
            min_separation_m=float("inf") if msep is None else float(msep),
            risk_level=int(d.get("risk_level", int(RiskLevel.NONE))),
            platform_counts={int(k): int(v)
                             for k, v in dict(d.get("platform_counts", {})).items()},
            intent_counts={int(k): int(v)
                           for k, v in dict(d.get("intent_counts", {})).items()},
            nmac_occurred=bool(d.get("nmac_occurred", False)),
            dwc_violated=bool(d.get("dwc_violated", False)),
            dwc_alerted=bool(d.get("dwc_alerted", d.get("dwc_violated", False))),
            duration_s=float(d.get("duration_s", 0.0)),
            reaction_events=int(d.get("reaction_events", 0)),
        )

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the metadata is consistent."""
        p: list[str] = []
        if self.episode_index < 0:
            p.append("meta.episode_index must be >= 0")
        if self.seed < 0:
            p.append("meta.seed must be >= 0")
        if self.n_agents < 0:
            p.append("meta.n_agents must be >= 0")
        if self.duration_s <= 0.0:
            p.append("meta.duration_s must be > 0 s")
        n_enc = len(self.geometry_type)
        if len(self.cpa_dist) != n_enc or len(self.tcpa) != n_enc:
            p.append("meta.geometry_type / cpa_dist / tcpa must have equal length")
        valid_geom = {int(g) for g in GeometryType}
        for g in self.geometry_type:
            if int(g) not in valid_geom:
                p.append(f"meta.geometry_type contains unknown value {g}")
        if int(self.risk_level) not in {int(r) for r in RiskLevel}:
            p.append(f"meta.risk_level {self.risk_level} is not a RiskLevel")
        valid_plat = {int(t) for t in PlatformType}
        for k, v in self.platform_counts.items():
            if int(k) not in valid_plat:
                p.append(f"meta.platform_counts has unknown platform key {k}")
            if int(v) < 0:
                p.append(f"meta.platform_counts[{k}] must be >= 0")
        valid_intent = {int(i) for i in IntentType}
        for k, v in self.intent_counts.items():
            if int(k) not in valid_intent:
                p.append(f"meta.intent_counts has unknown intent key {k}")
            if int(v) < 0:
                p.append(f"meta.intent_counts[{k}] must be >= 0")
        total_plat = sum(int(v) for v in self.platform_counts.values())
        if self.platform_counts and total_plat != int(self.n_agents):
            p.append(
                f"meta.platform_counts sum ({total_plat}) != n_agents "
                f"({self.n_agents})"
            )
        if self.nmac_occurred and not self.dwc_violated:
            p.append("meta.nmac_occurred implies dwc_violated must also be True")
        if self.dwc_violated and not self.dwc_alerted:
            p.append("meta.dwc_violated implies dwc_alerted must also be True")
        return p


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate(ep: EpisodeArrays) -> list[str]:
    """Check one episode against the storage contract.

    Verifies array presence, dtype, shape consistency (all arrays must agree
    on T and N), and the full NaN / sentinel policy documented at the top of
    this module.

    Args:
        ep: the episode buffer to check.

    Returns:
        A list of human-readable problems. Empty means the episode is
        contract-compliant and safe to write.
    """
    p: list[str] = []

    # -- presence and dtype ------------------------------------------------
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, _shape) in EPISODE_ARRAY_SPEC.items():
        arr = getattr(ep, name, None)
        if arr is None:
            p.append(f"{name}: missing")
            continue
        if not isinstance(arr, np.ndarray):
            p.append(f"{name}: not a numpy array (got {type(arr).__name__})")
            continue
        arrays[name] = arr
        if arr.dtype != np.dtype(dtype):
            p.append(f"{name}: dtype is {arr.dtype}, expected {np.dtype(dtype)}")
    if len(arrays) != len(EPISODE_ARRAY_SPEC):
        return p  # cannot check shapes reliably

    # -- shape consistency -------------------------------------------------
    T = int(arrays["t"].shape[0]) if arrays["t"].ndim == 1 else -1
    if T < 0:
        p.append("t: must be 1-D with shape [T]")
        return p
    N = int(arrays["gt_valid"].shape[1]) if arrays["gt_valid"].ndim == 2 else -1
    if N < 0:
        p.append("gt_valid: must be 2-D with shape [T,N]")
        return p
    dims = {"T": T, "N": N}
    for name, (_dtype, shape) in EPISODE_ARRAY_SPEC.items():
        expected = tuple(dims[s] if isinstance(s, str) else int(s) for s in shape)
        if arrays[name].shape != expected:
            p.append(f"{name}: shape {arrays[name].shape}, expected {expected}")
    if p:
        return p  # element-wise checks below assume shapes are right

    # -- time axis ---------------------------------------------------------
    t = arrays["t"]
    if T > 0 and not np.all(np.isfinite(t)):
        p.append("t: contains non-finite values")
    elif T > 1 and not np.all(np.diff(t.astype(np.float64)) > 0.0):
        p.append("t: must be strictly increasing")

    # -- ownship -----------------------------------------------------------
    for name in ("own_pos", "own_vel", "own_psi"):
        if not np.all(np.isfinite(arrays[name])):
            p.append(f"{name}: contains NaN/Inf (ownship state is always known)")
    psi = arrays["own_psi"].astype(np.float64)
    if psi.size and (np.any(psi <= -np.pi - 1e-5) or np.any(psi > np.pi + 1e-5)):
        p.append("own_psi: values outside the wrapped interval (-pi, pi]")

    # -- ground truth ------------------------------------------------------
    valid = arrays["gt_valid"]
    invalid = ~valid
    for name in ("gt_pos", "gt_vel"):
        a = arrays[name]
        if np.any(~np.isfinite(a[valid])):
            p.append(f"{name}: non-finite value where gt_valid is True")
        if np.any(~np.isnan(a[invalid])):
            p.append(f"{name}: must be NaN where gt_valid is False")
    gpsi = arrays["gt_psi"]
    if np.any(~np.isfinite(gpsi[valid])):
        p.append("gt_psi: non-finite value where gt_valid is True")
    if np.any(~np.isnan(gpsi[invalid])):
        p.append("gt_psi: must be NaN where gt_valid is False")
    vpsi = gpsi[valid].astype(np.float64)
    if vpsi.size and (np.any(vpsi <= -np.pi - 1e-5) or np.any(vpsi > np.pi + 1e-5)):
        p.append("gt_psi: values outside the wrapped interval (-pi, pi]")

    valid_modes = np.array(sorted(int(m) for m in ManeuverMode), dtype=np.int64)
    mode = arrays["gt_mode"].astype(np.int64)
    if np.any(~np.isin(mode[valid], valid_modes)):
        p.append("gt_mode: value outside ManeuverMode where gt_valid is True")
    if np.any(mode[invalid] != -1):
        p.append("gt_mode: must be -1 where gt_valid is False")

    valid_intents = np.array(sorted(int(i) for i in IntentType), dtype=np.int64)
    intent = arrays["gt_intent"].astype(np.int64)
    if np.any(~np.isin(intent[valid], valid_intents)):
        p.append("gt_intent: value outside IntentType where gt_valid is True")
    if np.any(intent[invalid] != -1):
        p.append("gt_intent: must be -1 where gt_valid is False")

    used = np.any(valid, axis=0)
    valid_types = np.array(sorted(int(t_) for t_ in PlatformType), dtype=np.int64)
    gtype = arrays["gt_type"].astype(np.int64)
    if np.any(~np.isin(gtype[used], valid_types)):
        p.append("gt_type: value outside PlatformType for a used slot")
    if np.any(gtype[~used] != -1):
        p.append("gt_type: must be -1 for slots that are never valid")

    ids = arrays["agent_ids"].astype(np.int64)
    if np.any(ids[used] < 0):
        p.append("agent_ids: must be >= 0 for a used slot")
    if np.any(ids[~used] != -1):
        p.append("agent_ids: must be -1 for slots that are never valid")
    used_ids = ids[used]
    if used_ids.size != np.unique(used_ids).size:
        p.append("agent_ids: duplicate id across used slots")

    # -- observations ------------------------------------------------------
    det = arrays["obs_detected"]
    undet = ~det
    opos = arrays["obs_pos"]
    if np.any(~np.isfinite(opos[det])):
        p.append("obs_pos: non-finite value where obs_detected is True")
    if np.any(~np.isnan(opos[undet])):
        p.append("obs_pos: must be NaN where obs_detected is False "
                 "(never interpolate a gap, doc 3.5)")
    oR = arrays["obs_R"]
    if np.any(~np.isfinite(oR[det])):
        p.append("obs_R: non-finite value where obs_detected is True")
    if np.any(~np.isnan(oR[undet])):
        p.append("obs_R: must be NaN where obs_detected is False")
    if np.any(det):
        Rd = oR[det].astype(np.float64)          # [K,2,2]
        if np.any(np.abs(Rd[:, 0, 1] - Rd[:, 1, 0]) > 1e-3):
            p.append("obs_R: covariance blocks must be symmetric")
        if np.any(Rd[:, 0, 0] <= 0.0) or np.any(Rd[:, 1, 1] <= 0.0):
            p.append("obs_R: diagonal entries must be > 0 m^2")
        detR = Rd[:, 0, 0] * Rd[:, 1, 1] - Rd[:, 0, 1] * Rd[:, 1, 0]
        if np.any(detR <= 0.0):
            p.append("obs_R: covariance must be positive definite")

    tv = arrays["obs_t_valid"]
    if np.any(~np.isfinite(tv[det])):
        p.append("obs_t_valid: non-finite value where obs_detected is True")
    if np.any(~np.isnan(tv[undet])):
        p.append("obs_t_valid: must be NaN where obs_detected is False")
    if np.any(det):
        # A measurement is always STALE relative to the frame it is logged in:
        # never from the future, and never older than one logging interval plus
        # the latency the sensor was configured with. The upper bound is left
        # generous (10 s) because the latency is a config value the schema does
        # not see; the point of the check is to catch a sign error or a stale
        # buffer, not to re-derive the sensor model.
        tcol = np.broadcast_to(t.astype(np.float64)[:, None], det.shape)[det]
        lag = tcol - tv[det].astype(np.float64)
        if np.any(lag < -1e-4):
            p.append("obs_t_valid: measurement time is AFTER the logged time")
        if np.any(lag > 10.0):
            p.append("obs_t_valid: measurement is more than 10 s stale")

    tid = arrays["obs_track_id"].astype(np.int64)
    if np.any(tid[det] < 0):
        p.append("obs_track_id: must be >= 0 wherever obs_detected is True")
    age = arrays["obs_track_age"].astype(np.int64)
    if np.any(age < 0):
        p.append("obs_track_age: must be >= 0")
    if np.any(age[tid < 0] != 0):
        p.append("obs_track_age: must be 0 where obs_track_id is -1")
    if np.any(det & ~valid):
        p.append("obs_detected: a detection exists where gt_valid is False")

    return p
