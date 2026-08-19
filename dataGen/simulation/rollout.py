"""Branch rollouts: the empirical "true" future distribution (doc 3.3).

This module is why the paper is allowed to make any claim about calibration at
all. Doc 1 states the justification for simulating rather than using recorded
flights: *the same situation can be replayed N times, so the true distribution
of futures is knowable* -- recorded data has exactly one future per past and
can never score a probabilistic prediction properly.

The procedure, and the reason each step is what it is:

1. **Snapshot the world at** ``branch_t``. Everything that can influence the
   future is captured, so every branch starts from a bit-identical situation.
2. **Restore, then re-seed only the decision streams.** The PAST must be
   identical -- same placement, same platforms, same wind realisation, same
   trajectory up to ``branch_t`` -- so the spread that appears afterwards is
   caused purely by the agents' stochastic decision rules and not by a
   different world. That is what makes the empirical spread a legitimate
   target distribution rather than an artefact of resampling everything.
3. **Run forward and record TRUE positions** at the
   :data:`core.constants.PRED_STEPS` logged prediction times. Ground truth,
   never the observation layer: the target of a predictor is where the
   aircraft actually went.

The resulting empirical distribution IS what calibration is scored against.

**The failure mode this module actively hunts for.** If snapshot/restore
over-restores -- restoring the decision streams too, or leaving a mode machine
bound to a stale generator -- every branch replays identically, the "true"
distribution collapses to a point mass, and every predictor scores as
perfectly calibrated no matter how wrong it is. That failure is silent and
would invalidate the paper's central claim, so :func:`branch_rollout` detects
it explicitly and raises :class:`DegenerateRolloutError`.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from core.constants import LOG_DECIMATION, LOG_DT, PRED_SECONDS

__all__ = [
    "BRANCH_STREAMS",
    "FROZEN_SPREAD_M",
    "MAX_FROZEN_FRACTION",
    "DegenerateRolloutError",
    "branch_rollout",
    "branch_seed_block",
    "frozen_agents",
    "rollout_statistics",
    "coverage_at",
    "chi2_2dof_quantile",
]


#: Streams re-randomised per branch.
#:
#: Wider than :data:`core.rng.BRANCH_STREAMS` on purpose. The two decision
#: streams are what generate multimodality; ``wind`` is included because gust
#: realisation is part of the aleatoric future (doc 3.6), and the two sensor
#: streams because a branch is a complete alternative future including what
#: the ownship would have observed in it. ``placement``, ``platform``,
#: ``intent`` and ``scenario`` are deliberately EXCLUDED: those define the
#: situation, and re-randomising them would change the past.
BRANCH_STREAMS: tuple[str, ...] = (
    "mode_decision",
    "branch_choice",
    "sensor_noise",
    "sensor_detect",
    "wind",
    # rev.3: the reaction layer's notice/delay draws are future stochastic
    # decisions too. Doc 3.3 warns that omitting this stream silently narrows
    # the branch distribution and corrupts the calibration ground truth.
    "perception",
)

#: Stride [-] between the default per-branch seeds. A prime keeps consecutive
#: branch blocks from aliasing onto each other when several branch points of
#: one episode are rolled out.
_SEED_STRIDE: int = 7919

#: RMS radial spread [m] at the final horizon step below which an agent's
#: empirical future is treated as a POINT MASS. 1 m over 15 s is three orders
#: of magnitude below the 15 s CV error of even the easiest powered class, so
#: anything under it carries no distributional information at all.
FROZEN_SPREAD_M: float = 1.0

#: Fraction [-] of live agents that may be frozen before
#: :func:`branch_rollout` refuses the rollout. Some agents are legitimately
#: deterministic -- an unpowered DEBRIS body advected by a steady wind has no
#: decision to make, and doc 3.7 keeps the gust off in the first-tier
#: configuration -- so the threshold is a majority rule, not zero tolerance.
MAX_FROZEN_FRACTION: float = 0.5


class DegenerateRolloutError(RuntimeError):
    """Branches given different seeds produced identical futures.

    This is never a "the agents happened to agree" situation: with distinct
    seeds on the decision streams the mode process alone re-randomises dwell
    lengths and turn directions, so bit-identical branches mean the reseeding
    did not reach the decision path. The usual causes are:

    * a mode machine still holding a reference to a pre-reseed generator (fix:
      call :meth:`simulation.engine.SimulationEngine.reseed_branch`, which
      re-binds, rather than :meth:`core.rng.RNGHub.reseed_branch` directly);
    * agents drawing from a private ``RNGHub.spawn_child`` generator, which a
      hub reseed cannot reach;
    * a snapshot that restores the decision streams after the reseed.
    """


def branch_seed_block(base_seed: int, n_branches: int) -> list[int]:
    """Default per-branch seeds derived from one base seed.

    Args:
        base_seed: base seed [-] of this branch point.
        n_branches: number of branches [-].

    Returns:
        ``n_branches`` distinct, reproducible seeds.
    """
    return [int(base_seed) + i * _SEED_STRIDE for i in range(int(n_branches))]


def branch_rollout(
    engine: Any,
    branch_t: float,
    n_branches: int,
    horizon_s: float = PRED_SECONDS,
    branch_seeds: Optional[Sequence[int]] = None,
    streams: Sequence[str] = BRANCH_STREAMS,
    require_divergence: bool = True,
) -> np.ndarray:
    """Roll the world forward ``n_branches`` times from one shared past.

    The engine is advanced to ``branch_t`` if it has not reached it yet, and
    is left exactly as it was found: the branch-point snapshot is restored
    once more before returning, so an episode being logged is untouched by
    having rollouts taken from it. Rollout stepping runs with logging
    disabled for the same reason.

    Args:
        engine: a :class:`simulation.engine.SimulationEngine`.
        branch_t: time [s] at which the futures fork. Should be a logged
            instant (a multiple of :data:`core.constants.LOG_DT`); the
            prediction times are ``branch_t + LOG_DT ... branch_t + horizon_s``
            to match the engine's log convention.
        n_branches: number of branches [-]; doc 3.7 asks for 50-100.
        horizon_s: prediction horizon [s]; defaults to
            :data:`core.constants.PRED_SECONDS` (15 s).
        branch_seeds: explicit per-branch seeds; derived from the engine's
            master seed and ``branch_t`` when omitted.
        streams: stream names to re-randomise per branch; see
            :data:`BRANCH_STREAMS`.
        require_divergence: raise :class:`DegenerateRolloutError` when
            distinct seeds yield bit-identical branches. Only pass False when
            deliberately testing the identical-seed case.

    Returns:
        ``[n_branches, n_agents, n_pred_steps, 2]`` float64 positions [m].
        NaN marks an agent that is not alive at that step. float64, not the
        storage float32: this is an analysis product, and coverage statistics
        computed from it must not inherit storage rounding.

    Raises:
        ValueError: if ``n_branches`` < 1, ``horizon_s`` <= 0, or
            ``branch_seeds`` has the wrong length or contains duplicates.
        DegenerateRolloutError: see :class:`DegenerateRolloutError`.
    """
    n = int(n_branches)
    if n < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches!r}")
    if float(horizon_s) <= 0.0:
        raise ValueError(f"horizon_s must be > 0 s, got {horizon_s!r}")

    n_pred = int(round(float(horizon_s) / LOG_DT))
    if n_pred < 1:
        raise ValueError(
            f"horizon_s {horizon_s} s is shorter than one logged step ({LOG_DT} s)"
        )

    if branch_seeds is None:
        base = int(getattr(engine.rng_hub, "master_seed", 0))
        base += int(round(float(branch_t) / LOG_DT)) * 1_000_003
        seeds = branch_seed_block(base, n)
    else:
        seeds = [int(s) for s in branch_seeds]
        if len(seeds) != n:
            raise ValueError(
                f"branch_seeds has length {len(seeds)} but n_branches is {n}"
            )

    engine.advance_to(float(branch_t), log=True)
    base_snapshot = engine.snapshot()

    n_agents = int(engine.n_agents)
    out = np.full((n, n_agents, n_pred, 2), np.nan, dtype=np.float64)

    for b, seed in enumerate(seeds):
        engine.restore(base_snapshot)
        engine.reseed_branch(int(seed), streams)
        for k in range(n_pred):
            for _ in range(LOG_DECIMATION):
                engine.step(log=False)
            out[b, :, k, :] = engine.gt_positions()

    engine.restore(base_snapshot)

    if require_divergence and n > 1 and len(set(seeds)) > 1:
        _assert_divergence(out, seeds, horizon_s=float(horizon_s))
    return out


def frozen_agents(
    rollouts: np.ndarray,
    tol_m: float = FROZEN_SPREAD_M,
    stats: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]:
    """Which agents' empirical futures are point masses.

    The per-agent counterpart of :func:`_assert_divergence`. An array-wide
    "some branch differs somewhere" test passes on a rollout where 90% of the
    agents are frozen, because one wandering bird is enough to satisfy it --
    and every calibration score taken from a frozen agent is a point mass with
    sigma = 0, which tells a model the correct uncertainty is zero.

    Args:
        rollouts: ``[B,N,K,2]`` positions [m] from :func:`branch_rollout`.
        tol_m: RMS radial spread [m] at the final step at or below which an
            agent counts as frozen.
        stats: a :func:`rollout_statistics` result to reuse.

    Returns:
        ``{'frozen': [N] bool, 'live': [N] bool, 'spread_m': [N] float,
        'n_live': int, 'n_frozen': int, 'fraction': float}``. ``fraction`` is
        the frozen share of LIVE agents, and 0.0 when nothing is live.
    """
    arr = np.asarray(rollouts, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"rollouts must have shape [B,N,K,2], got {arr.shape}")
    st = rollout_statistics(arr) if stats is None else stats
    spread = np.asarray(st["spread"], dtype=np.float64)[:, -1]     # [N] at horizon
    live = np.isfinite(spread)
    frozen = live & (spread <= float(tol_m))
    n_live = int(live.sum())
    n_frozen = int(frozen.sum())
    return {
        "frozen": frozen,
        "live": live,
        "spread_m": spread,
        "n_live": n_live,
        "n_frozen": n_frozen,
        "fraction": (n_frozen / n_live) if n_live else 0.0,
    }


def _assert_divergence(
    rollouts: np.ndarray,
    seeds: Sequence[int],
    horizon_s: float = PRED_SECONDS,
) -> None:
    """Raise if the branches do not carry a usable future distribution.

    Two checks, because they catch two different failures:

    1. **Whole-array**: every branch bit-identical to branch 0 means the
       reseeding never reached the decision path at all (see
       :class:`DegenerateRolloutError`). NaN is treated as equal to NaN so a
       retired agent does not read as a difference. Always applied.
    2. **Per agent**: more than :data:`MAX_FROZEN_FRACTION` of the LIVE agents
       having a spread at or below :data:`FROZEN_SPREAD_M` at the horizon.
       Check 1 alone passes as long as *any one* agent moved, so a rollout in
       which almost everything is a point mass reported "diverged".

       Applied only at or beyond the FULL prediction horizon. What guarantees
       a fork is the dwell truncation in
       :meth:`agents.maneuver.ModeMachine._draw_dwell`, and it guarantees a
       decision epoch within :data:`core.constants.PRED_SECONDS` -- not within
       an arbitrary shorter window. A 5 s rollout of aircraft whose commitment
       floors are 2-12 s is *legitimately* near-deterministic, and failing it
       would be a false alarm.

    Args:
        rollouts: ``[B,N,K,2]`` positions [m].
        seeds: the seeds used, for the error message.
        horizon_s: the horizon [s] the rollout covered.

    Raises:
        DegenerateRolloutError: on either condition.
    """
    first = rollouts[0]
    identical = True
    for b in range(1, rollouts.shape[0]):
        if not np.array_equal(rollouts[b], first, equal_nan=True):
            identical = False
            break
    if identical:
        raise DegenerateRolloutError(
            f"all {rollouts.shape[0]} branches are bit-identical despite "
            f"{len(set(int(s) for s in seeds))} distinct seeds "
            f"({[int(s) for s in seeds[:4]]}...). The re-seeding is not reaching "
            "the agents' decision draws, so the empirical future distribution is a "
            "point mass and every calibration score computed from it is "
            "meaningless. See DegenerateRolloutError for the usual causes."
        )

    if float(horizon_s) + 1e-9 < PRED_SECONDS:
        return
    fz = frozen_agents(rollouts)
    if fz["n_live"] and fz["fraction"] > MAX_FROZEN_FRACTION:
        raise DegenerateRolloutError(
            f"{fz['n_frozen']} of {fz['n_live']} live agents have a branch "
            f"spread <= {FROZEN_SPREAD_M:g} m at the horizon "
            f"({fz['fraction'] * 100:.0f}%, limit "
            f"{MAX_FROZEN_FRACTION * 100:.0f}%). The rollout as a whole is not "
            "bit-identical, but most agents' empirical futures ARE point "
            "masses, so a per-agent calibration score taken from it would be "
            "fitted against sigma = 0. Usual cause: mode dwells outliving the "
            "prediction horizon (see agents.maneuver.ModeMachine._draw_dwell)."
        )


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def rollout_statistics(rollouts: np.ndarray) -> dict[str, np.ndarray]:
    """Per-agent, per-step moments of an empirical future distribution.

    Branches where an agent is not alive (NaN) are dropped rather than
    imputed, so an agent that retires part way through still produces valid
    statistics for the steps it was present.

    Args:
        rollouts: ``[B,N,K,2]`` positions [m] from :func:`branch_rollout`.

    Returns:
        A dict of arrays:

        * ``count``  ``[N,K]`` int64  -- branches contributing [-];
        * ``mean``   ``[N,K,2]``      -- mean position [m];
        * ``cov``    ``[N,K,2,2]``    -- covariance [m^2], the unbiased
          (``N-1``) estimate, NaN where fewer than 2 branches contribute;
        * ``std``    ``[N,K,2]``      -- per-axis standard deviation [m];
        * ``spread`` ``[N,K]``        -- RMS radial spread [m] about the mean,
          i.e. ``sqrt(trace(cov))``, a single scalar for plotting;
        * ``frozen`` ``[N]`` bool     -- agents whose spread at the LAST step
          is at or below :data:`FROZEN_SPREAD_M`, i.e. whose empirical future
          is a point mass. Surfaced here so a caller reading the statistics
          cannot miss the degenerate rows: their covariance is ~0 and any
          calibration score computed against them is meaningless;
        * ``frozen_fraction`` scalar  -- frozen share [-] of live agents.

    Raises:
        ValueError: if ``rollouts`` is not 4-D with a trailing axis of 2.
    """
    arr = np.asarray(rollouts, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(
            f"rollouts must have shape [B,N,K,2], got {arr.shape}"
        )
    B, N, K, _ = arr.shape

    finite = np.isfinite(arr).all(axis=-1)                  # [B,N,K]
    count = finite.sum(axis=0).astype(np.int64)             # [N,K]

    mean = np.full((N, K, 2), np.nan, dtype=np.float64)
    cov = np.full((N, K, 2, 2), np.nan, dtype=np.float64)
    std = np.full((N, K, 2), np.nan, dtype=np.float64)
    spread = np.full((N, K), np.nan, dtype=np.float64)

    for i in range(N):
        for k in range(K):
            m = finite[:, i, k]
            c = int(count[i, k])
            if c < 1:
                continue
            pts = arr[m, i, k, :]                            # [c,2]
            mu = pts.mean(axis=0)
            mean[i, k, :] = mu
            if c < 2:
                continue
            d = pts - mu
            c_mat = (d.T @ d) / (c - 1)
            cov[i, k, :, :] = c_mat
            var = np.diag(c_mat)
            std[i, k, :] = np.sqrt(np.maximum(var, 0.0))
            spread[i, k] = math.sqrt(max(float(var.sum()), 0.0))

    last = spread[:, -1] if K > 0 else np.full((N,), np.nan)
    live = np.isfinite(last)
    frozen = live & (last <= FROZEN_SPREAD_M)
    n_live = int(live.sum())
    return {
        "count": count,
        "mean": mean,
        "cov": cov,
        "std": std,
        "spread": spread,
        "frozen": frozen,
        "frozen_fraction": float(int(frozen.sum()) / n_live) if n_live else 0.0,
    }


def chi2_2dof_quantile(level: float) -> float:
    """Chi-squared quantile with 2 degrees of freedom, in closed form.

    For 2 dof the CDF is ``1 - exp(-c/2)``, so the quantile is
    ``-2 * ln(1 - level)``. Having it in closed form is what lets the
    coverage check run with numpy alone -- no scipy anywhere in dataGen.

    Args:
        level: nominal probability [-] in ``[0, 1)``.

    Returns:
        The squared-Mahalanobis threshold [-] enclosing ``level`` of a
        bivariate normal.

    Raises:
        ValueError: if ``level`` is outside ``[0, 1)``.
    """
    p = float(level)
    if not (0.0 <= p < 1.0):
        raise ValueError(f"level must be in [0, 1), got {level!r}")
    return -2.0 * math.log(1.0 - p)


def coverage_at(
    rollouts: np.ndarray,
    level: float = 0.9,
    stats: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Empirical coverage of the Gaussian fit to an empirical distribution.

    A self-consistency probe, and the reference implementation the analysis
    stage reuses against MODEL-predicted moments. Fitting a Gaussian to the
    branches and then asking what fraction of those same branches falls inside
    its ``level`` ellipse answers one question: how Gaussian is the true
    future? Where a genuine fork exists the empirical distribution is
    bimodal, the fitted ellipse is too wide in the middle, and the measured
    coverage rises above ``level`` -- so a sustained gap between measured and
    nominal coverage is a MAP OF THE MULTIMODAL REGIONS (doc 9.5: "sigma가
    커진 지점 vs 실제 갈림길 일치도"), not a defect.

    Args:
        rollouts: ``[B,N,K,2]`` positions [m].
        level: nominal coverage [-] in ``[0, 1)``, e.g. 0.9.
        stats: a :func:`rollout_statistics` result to reuse; recomputed when
            omitted.

    Returns:
        A dict:

        * ``coverage``  ``[N,K]`` -- measured fraction [-] inside the ellipse,
          NaN where the fit is undefined;
        * ``count``     ``[N,K]`` int64 -- branches contributing [-];
        * ``nominal``   scalar float -- ``level``;
        * ``threshold`` scalar float -- the Mahalanobis-squared threshold [-];
        * ``overall``   scalar float -- count-weighted mean coverage [-] over
          all valid (agent, step) cells.

    Raises:
        ValueError: if ``rollouts`` has the wrong shape or ``level`` is out of
            range.
    """
    arr = np.asarray(rollouts, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"rollouts must have shape [B,N,K,2], got {arr.shape}")
    threshold = chi2_2dof_quantile(level)
    st = rollout_statistics(arr) if stats is None else stats
    mean = st["mean"]
    cov = st["cov"]

    _B, N, K, _ = arr.shape
    coverage = np.full((N, K), np.nan, dtype=np.float64)
    used = np.zeros((N, K), dtype=np.int64)

    for i in range(N):
        for k in range(K):
            c_mat = cov[i, k]
            if not np.all(np.isfinite(c_mat)):
                continue
            det = c_mat[0, 0] * c_mat[1, 1] - c_mat[0, 1] * c_mat[1, 0]
            if not np.isfinite(det) or det <= 0.0:
                continue
            inv = np.array(
                [[c_mat[1, 1], -c_mat[0, 1]], [-c_mat[1, 0], c_mat[0, 0]]],
                dtype=np.float64,
            ) / det
            pts = arr[:, i, k, :]
            m = np.isfinite(pts).all(axis=-1)
            if not np.any(m):
                continue
            d = pts[m] - mean[i, k]
            maha = np.einsum("bi,ij,bj->b", d, inv, d)
            coverage[i, k] = float(np.mean(maha <= threshold))
            used[i, k] = int(m.sum())

    valid = np.isfinite(coverage)
    if np.any(valid):
        w = used[valid].astype(np.float64)
        overall = float(np.sum(coverage[valid] * w) / max(float(np.sum(w)), 1.0))
    else:
        overall = float("nan")

    return {
        "coverage": coverage,
        "count": st["count"],
        "nominal": float(level),
        "threshold": float(threshold),
        "overall": overall,
    }
