"""Run one episode, or a whole batch, and derive the episode metadata.

The metadata computed here is what makes doc 8.1's stratified analysis and
doc 9.4's safety reporting possible at all: realised minimum separation, NMAC
occurrence, DAA Well Clear violation, the achieved CPA of every designed
encounter, and the platform / intent census.

:func:`generate_batch` is deliberately the ONLY batch path. The CLI in
``main.py`` and the GUI's batch job both call it, so a dataset produced from
the browser and a dataset produced from the command line cannot drift apart.

Separation criteria (doc 9.4, 2D -- horizontal component only):

* **NMAC** -- horizontal separation below
  :data:`core.constants.NMAC_RADIUS_M` (500 ft) at any logged step.
* **DWC** -- the RTCA DO-365 style pair of conditions, evaluated per logged
  step against the constant-velocity projection of the current relative
  state: the projected horizontal miss distance is below
  :data:`core.constants.DWC_HMD_M` AND the modified tau is at or below
  :data:`core.constants.DWC_TAU_MOD_S`. The threshold VALUES still carry the
  "verify against DO-365 before publication" warning of
  :mod:`core.constants`; this module only fixes how they are combined.

Units are SI: metres, seconds, radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from analysis.metrics import risk_level_from_separation
from core.constants import DWC_HMD_M, DWC_TAU_MOD_S, NMAC_RADIUS_M
from core.types import RiskLevel
from scenario.builder import Scenario
from scenario.config import ScenarioConfig
from storage.schema import EpisodeArrays, EpisodeMeta
from storage.schema import validate as validate_arrays

__all__ = [
    "EpisodeResult",
    "BatchResult",
    "separation_series",
    "compute_episode_meta",
    "episode_seeds",
    "run_episode",
    "generate_batch",
]


#: Callback invoked after each episode of a batch:
#: ``(index [-], total [-], meta) -> None``.
ProgressCallback = Callable[[int, int, EpisodeMeta], None]

#: Consumer of a finished episode, e.g. ``storage.writer``:
#: ``(arrays, meta) -> None``.
EpisodeSink = Callable[[EpisodeArrays, EpisodeMeta], None]


@dataclass
class EpisodeResult:
    """One finished episode.

    Attributes:
        arrays: the logged episode buffer.
        meta: the derived metadata.
    """

    arrays: EpisodeArrays
    meta: EpisodeMeta


@dataclass
class BatchResult:
    """Outcome of a :func:`generate_batch` run.

    Attributes:
        seeds: the master seeds [-] used, in generation order.
        metas: metadata of every episode that completed.
        episodes: the full episodes, only when ``collect=True`` -- 5000
            episodes of arrays do not fit in memory, so the default path
            streams them to a ``sink`` instead.
        failures: ``(seed, message)`` for every episode that raised, when
            ``stop_on_error`` is False.
    """

    seeds: list[int] = field(default_factory=list)
    metas: list[EpisodeMeta] = field(default_factory=list)
    episodes: list[EpisodeResult] = field(default_factory=list)
    failures: list[tuple[int, str]] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        """Number of episodes that completed [-]."""
        return len(self.metas)

    @property
    def n_failed(self) -> int:
        """Number of episodes that raised [-]."""
        return len(self.failures)

    def summary(self) -> dict[str, Any]:
        """Aggregate counters for a manifest or a progress readout.

        Returns:
            ``{"n_ok", "n_failed", "nmac_rate", "dwc_rate",
            "min_separation_m", "risk_counts"}``. Rates are fractions [-] over
            the episodes that completed.
        """
        n = max(self.n_ok, 1)
        nmac = sum(1 for m in self.metas if m.nmac_occurred)
        dwc = sum(1 for m in self.metas if m.dwc_violated)
        seps = [m.min_separation_m for m in self.metas
                if np.isfinite(m.min_separation_m)]
        risk: dict[int, int] = {}
        for m in self.metas:
            risk[int(m.risk_level)] = risk.get(int(m.risk_level), 0) + 1
        return {
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "nmac_rate": nmac / n,
            "dwc_rate": dwc / n,
            "min_separation_m": float(min(seps)) if seps else float("inf"),
            "risk_counts": risk,
        }


# --------------------------------------------------------------------------
# Separation analysis
# --------------------------------------------------------------------------


def separation_series(ep: EpisodeArrays) -> dict[str, np.ndarray]:
    """Per-step, per-agent ownship separation geometry.

    Everything doc 9.4 needs is derived once, vectorised, from the ground
    truth layer -- never from the observation layer, because a metric computed
    from noisy tracks would measure the sensor rather than the encounter.

    Args:
        ep: a logged episode.

    Returns:
        A dict of ``[T,N]`` float64 arrays (NaN where ``gt_valid`` is False):

        * ``range_m``   -- horizontal separation [m];
        * ``range_rate``-- its time derivative [m/s], negative when closing;
        * ``hmd_m``     -- horizontal miss distance [m] of the constant-velocity
          projection of the current relative state;
        * ``tcpa_s``    -- time [s] to that projected CPA, clamped to >= 0;
        * ``tau_mod_s`` -- modified tau [s]; ``+inf`` for a non-closing pair,
          negative once the pair is already inside the HMD threshold.

        Plus ``valid`` -- the ``[T,N]`` boolean ground-truth mask.
    """
    valid = np.asarray(ep.gt_valid, dtype=bool)
    gt_pos = np.asarray(ep.gt_pos, dtype=np.float64)
    gt_vel = np.asarray(ep.gt_vel, dtype=np.float64)
    own_pos = np.asarray(ep.own_pos, dtype=np.float64)[:, None, :]
    own_vel = np.asarray(ep.own_vel, dtype=np.float64)[:, None, :]

    rel = gt_pos - own_pos           # [T,N,2]
    relv = gt_vel - own_vel          # [T,N,2]

    r = np.hypot(rel[..., 0], rel[..., 1])
    r_safe = np.where(r > 1e-9, r, 1e-9)
    dot = rel[..., 0] * relv[..., 0] + rel[..., 1] * relv[..., 1]
    range_rate = dot / r_safe

    vv = relv[..., 0] ** 2 + relv[..., 1] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        tcpa = np.where(vv > 1e-12, -dot / np.where(vv > 1e-12, vv, 1.0), 0.0)
    tcpa = np.maximum(tcpa, 0.0)
    mx = rel[..., 0] + relv[..., 0] * tcpa
    my = rel[..., 1] + relv[..., 1] * tcpa
    hmd = np.hypot(mx, my)

    # tau_mod = (HMD*^2 - r^2) / (r * r_dot), only meaningful while closing.
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = (DWC_HMD_M ** 2 - r ** 2) / (r_safe * range_rate)
    tau_mod = np.where(range_rate < -1e-9, tau, np.inf)

    nan = np.nan
    def _mask(a: np.ndarray, fill: float = nan) -> np.ndarray:
        out = np.asarray(a, dtype=np.float64).copy()
        out[~valid] = fill
        return out

    return {
        "valid": valid,
        "range_m": _mask(r),
        "range_rate": _mask(range_rate),
        "hmd_m": _mask(hmd),
        "tcpa_s": _mask(tcpa),
        "tau_mod_s": _mask(tau_mod, np.inf),
    }


def compute_episode_meta(
    ep: EpisodeArrays,
    scenario: Scenario,
    episode_index: int = 0,
    min_separation_m: Optional[float] = None,
    reaction_events: int = 0,
) -> EpisodeMeta:
    """Derive the metadata of a finished episode.

    Args:
        ep: the logged episode.
        scenario: the scenario it was generated from (for the census, the
            wind and the designed-encounter slots).
        episode_index: index [-] of this episode within its dataset.
        min_separation_m: the minimum ownship-to-agent distance [m] measured
            by the engine at ``SIM_DT``. Pass it whenever it is available:
            ``ep`` is sampled at ``LOG_DT``, and a minimum taken over a 2 Hz
            series is biased HIGH by up to half a logging interval of closure
            (measured at 19 m, 12% of the NMAC radius), which silently turns a
            true NMAC into ``nmac_occurred = False``. Omitting it falls back to
            the logged series, which is correct only for an episode that was
            loaded from disk rather than just simulated.

    Returns:
        A populated :class:`storage.schema.EpisodeMeta`.
    """
    cfg = scenario.cfg
    series = separation_series(ep)
    valid = series["valid"]
    rng_m = series["range_m"]
    hmd = series["hmd_m"]
    tau = series["tau_mod_s"]

    if np.any(valid):
        logged_min = float(np.nanmin(np.where(valid, rng_m, np.inf)))
    else:
        logged_min = float("inf")
    if min_separation_m is None:
        min_sep = logged_min
    else:
        # min() rather than a plain override: the engine's accumulator and the
        # logged series see the same trajectory, so the fine one can only be
        # smaller. Taking the minimum makes the field monotone under any future
        # change to either path instead of trusting one blindly.
        min_sep = min(logged_min, float(min_separation_m))

    nmac = bool(min_sep < NMAC_RADIUS_M)
    # Two distinct things, previously conflated into one field:
    #   ALERT -- at some instant the constant-velocity projection of the
    #            relative state said well clear would be lost within tau_mod.
    #            This is the DAA *alerting* predicate, and in a 36 km^2 arena
    #            with doc 3.7 traffic it fires in essentially every episode:
    #            an intruder 3 km away closing at 100 m/s with a 500 m
    #            projected miss already satisfies it.
    #   LOST  -- well clear was ACTUALLY lost, i.e. the achieved minimum
    #            separation fell below the HMD threshold.
    # `dwc_violated` now means LOST, which is what its name says and what
    # doc 9.4 grades; the alert is reported alongside it. Conflating them made
    # `dwc_violated` True in 100% of episodes and therefore uninformative.
    alerted = bool(np.any(valid & (hmd < DWC_HMD_M) & (tau <= DWC_TAU_MOD_S)))
    dwc = bool(np.isfinite(min_sep) and min_sep < DWC_HMD_M)
    # An NMAC is by construction a loss of well clear (NMAC_RADIUS_M is far
    # inside DWC_HMD_M); assert the implication so EpisodeMeta.validate() holds
    # even under a future threshold edit.
    dwc = dwc or nmac
    alerted = alerted or dwc

    # THE ladder lives in analysis.metrics, not here: doc 9.4 requires the
    # dataset label and the evaluation metric to agree, and they cannot if the
    # threshold rule exists in two places.
    risk = RiskLevel(risk_level_from_separation(min_sep, nmac, dwc, alerted))

    geometry_type: list[int] = []
    cpa_dist: list[float] = []
    tcpa: list[float] = []
    t_axis = np.asarray(ep.t, dtype=np.float64)
    for enc in scenario.designed:
        slot = int(enc.slot)
        geometry_type.append(int(enc.geometry))
        if slot >= rng_m.shape[1]:
            cpa_dist.append(float("nan"))
            tcpa.append(float("nan"))
            continue
        col = np.where(valid[:, slot], rng_m[:, slot], np.inf)
        if not np.any(np.isfinite(col)):
            cpa_dist.append(float("nan"))
            tcpa.append(float("nan"))
            continue
        k = int(np.argmin(col))
        cpa_dist.append(float(col[k]))
        tcpa.append(float(t_axis[k]))

    return EpisodeMeta(
        episode_index=int(episode_index),
        seed=int(cfg.seed),
        density=float(cfg.density_per_km2),
        realised_density_per_km2=float(scenario.realised_density_per_km2),
        wind_speed=float(scenario.wind_speed),
        wind_dir=float(scenario.wind_dir),
        n_agents=int(scenario.n_agents),
        geometry_type=geometry_type,
        cpa_dist=cpa_dist,
        tcpa=tcpa,
        min_separation_m=min_sep,
        risk_level=int(risk),
        platform_counts=scenario.platform_counts(),
        intent_counts=scenario.intent_counts(),
        nmac_occurred=nmac,
        dwc_violated=dwc,
        dwc_alerted=alerted,
        duration_s=float(cfg.duration_s),
        reaction_events=int(reaction_events),
    )


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def episode_seeds(
    cfg: ScenarioConfig,
    n: int,
    seeds: Optional[Sequence[int]] = None,
) -> list[int]:
    """Resolve the master seeds [-] of a batch.

    Args:
        cfg: base configuration; ``cfg.seed`` is the first seed of the default
            consecutive block.
        n: number of episodes [-].
        seeds: explicit seeds; must have length ``n`` when given.

    Returns:
        A list of ``n`` master seeds.

    Raises:
        ValueError: if ``n`` is negative, or ``seeds`` has the wrong length,
            or ``seeds`` contains duplicates (which would silently generate
            the same episode twice and bias every rate reported from the set).
    """
    if int(n) < 0:
        raise ValueError(f"n must be >= 0, got {n!r}")
    if seeds is None:
        return [int(cfg.seed) + i for i in range(int(n))]
    out = [int(s) for s in seeds]
    if len(out) != int(n):
        raise ValueError(f"seeds has length {len(out)} but n is {int(n)}")
    if len(set(out)) != len(out):
        raise ValueError("seeds contains duplicates: episodes would repeat")
    return out


def run_episode(
    cfg: ScenarioConfig,
    seed: Optional[int] = None,
    episode_index: int = 0,
    ownship_controller: Optional[Callable[..., tuple[float, float]]] = None,
    validate: bool = True,
) -> tuple[EpisodeArrays, EpisodeMeta]:
    """Generate one complete episode.

    Args:
        cfg: scenario configuration; NOT mutated. When ``seed`` is given a
            copy carrying that seed is used instead.
        seed: master seed [-]; defaults to ``cfg.seed``.
        episode_index: index [-] recorded in the metadata.
        ownship_controller: pluggable ownship control law; see
            :data:`simulation.engine.OwnshipController`.
        validate: check the result against the storage contract before
            returning it. Leave this on for dataset generation -- a schema
            violation that reaches disk is far more expensive than the check.

    Returns:
        ``(arrays, meta)``.

    Raises:
        ValueError: if ``validate`` is True and the episode or its metadata
            violates the storage contract.
    """
    from simulation.engine import SimulationEngine

    episode_cfg = cfg if seed is None else cfg.copy_with(seed=int(seed))
    engine = SimulationEngine(
        episode_cfg,
        ownship_controller=ownship_controller,
        episode_index=int(episode_index),
    )
    arrays, meta = engine.run_episode()

    if validate:
        problems = validate_arrays(arrays)
        if problems:
            raise ValueError(
                f"episode {episode_index} (seed {episode_cfg.seed}) violates the "
                "storage contract: " + "; ".join(problems)
            )
        meta_problems = meta.validate()
        if meta_problems:
            raise ValueError(
                f"episode {episode_index} (seed {episode_cfg.seed}) metadata is "
                "inconsistent: " + "; ".join(meta_problems)
            )
    return arrays, meta


def generate_batch(
    cfg: ScenarioConfig,
    n: int,
    seeds: Optional[Sequence[int]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    sink: Optional[EpisodeSink] = None,
    collect: bool = False,
    episode_index_start: int = 0,
    ownship_controller: Optional[Callable[..., tuple[float, float]]] = None,
    validate: bool = True,
    stop_on_error: bool = True,
) -> BatchResult:
    """Generate ``n`` episodes -- the one code path the CLI and the GUI share.

    Args:
        cfg: base configuration.
        n: number of episodes [-].
        seeds: explicit master seeds; consecutive from ``cfg.seed`` when
            omitted.
        progress_cb: called as ``progress_cb(i, n, meta)`` after each episode,
            with ``i`` the 1-based count of completed episodes. The GUI uses
            this to stream progress; the CLI to print it.
        sink: called as ``sink(arrays, meta)`` for each episode, before the
            arrays are dropped. This is the streaming hand-off to
            ``storage.writer``; without it (and without ``collect``) the
            arrays are discarded and only the metadata is kept.
        collect: also keep every episode in :attr:`BatchResult.episodes`.
            Memory-hungry -- intended for tests and small interactive runs,
            not for the 5000-episode dataset of doc 3.7.
        episode_index_start: index [-] of the first episode.
        ownship_controller: pluggable ownship control law.
        validate: validate every episode against the storage contract.
        stop_on_error: re-raise the first failure. When False the failure is
            recorded in :attr:`BatchResult.failures` and generation continues,
            which is what a long unattended run wants.

    Returns:
        A :class:`BatchResult`.

    Raises:
        Exception: whatever an episode raised, when ``stop_on_error`` is True.
    """
    total = int(n)
    resolved = episode_seeds(cfg, total, seeds)
    result = BatchResult(seeds=list(resolved))

    for i, seed in enumerate(resolved):
        index = int(episode_index_start) + i
        try:
            arrays, meta = run_episode(
                cfg,
                seed=seed,
                episode_index=index,
                ownship_controller=ownship_controller,
                validate=validate,
            )
        except Exception as exc:  # noqa: BLE001 - recorded and re-raised below
            if stop_on_error:
                raise
            result.failures.append((int(seed), f"{type(exc).__name__}: {exc}"))
            continue

        if sink is not None:
            sink(arrays, meta)
        if collect:
            result.episodes.append(EpisodeResult(arrays=arrays, meta=meta))
        result.metas.append(meta)
        if progress_cb is not None:
            progress_cb(i + 1, total, meta)

    return result
