"""Doc section 9 evaluation metrics, frozen before the predictor exists.

The prediction model is not built yet, but the metric *definitions* are what a
reviewer checks and what every later number depends on. Freezing them now --
with unit tests -- means the numbers reported in the paper cannot silently
change when the model lands.

What lives here, mapped to the doc:

    9.1 prediction accuracy   ADE, FDE, minADE_k, minFDE_k
    9.2 distribution fit      Gaussian NLL, CRPS (closed form and ensemble),
                              energy score
    9.3 calibration           empirical coverage, reliability curve, PIT
    9.4 downstream safety     minimum separation, NMAC count, DAA Well Clear
                              violation, risk-level labelling

Shape convention. A *sample* is one predicted future for one agent at one
time origin. Arrays are ``[..., H, 2]`` where ``H`` is the number of predicted
steps and the trailing axis is ``(x, y)`` in metres; multimodal predictions
add a mode axis, ``[..., K, H, 2]``. Every function accepts an optional
boolean ``mask`` of shape ``[..., H]`` marking which steps are scored, so a
target that leaves the scene mid-horizon does not have to be dropped whole.

Numerical policy. Everything is computed in float64 regardless of the storage
dtype, and reductions ignore masked-out entries rather than filling them with
zeros (a zero error is not the same as no error).

Angles: [rad] CCW from +X. Distances: [m]. Times: [s]. Covariances: [m^2].
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from core.constants import (
    DWC_HMD_M,
    DWC_TAU_MOD_S,
    LOG_DT,
    NMAC_RADIUS_M,
)
from core.types import RiskLevel
from storage.schema import EpisodeArrays

__all__ = [
    # 9.1
    "displacement_error",
    "ade_per_sample",
    "fde_per_sample",
    "ade",
    "fde",
    "min_ade_k",
    "min_fde_k",
    # 9.2
    "mahalanobis_sq",
    "gaussian_nll",
    "gaussian_nll_diag",
    "crps_gaussian",
    "crps_ensemble",
    "energy_score",
    # 9.3
    "chi2_quantile_2dof",
    "empirical_coverage",
    "reliability_curve",
    "pit_values",
    "calibration_error",
    # 9.4
    "LOW_RISK_MARGIN_FACTOR",
    "separation_distances",
    "min_separation",
    "nmac_events",
    "tau_mod",
    "horizontal_miss_distance",
    "dwc_violation",
    "risk_level_from_separation",
    "risk_stratification",
    "separation_metrics",
]


# --------------------------------------------------------------------------
# Small numeric helpers (numpy has no erf; scipy is banned)
# --------------------------------------------------------------------------

_erf_vec = np.vectorize(math.erf, otypes=[np.float64])
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _std_normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, elementwise, via :func:`math.erf`."""
    return 0.5 * (1.0 + _erf_vec(np.asarray(z, dtype=np.float64) * _INV_SQRT2))


def _std_normal_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF, elementwise."""
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def _as2(a: np.ndarray) -> np.ndarray:
    """Cast to float64 and check the trailing axis is the 2-D plane."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.shape[-1] != 2:
        raise ValueError(f"expected trailing axis of size 2, got shape {arr.shape}")
    return arr


def _masked_mean(values: np.ndarray, mask: np.ndarray | None, axis: Any = None) -> np.ndarray:
    """Mean over ``axis`` ignoring masked-out entries; empty -> NaN.

    Args:
        values: array of finite scores.
        mask: boolean array broadcastable to ``values``; None means all valid.
        axis: reduction axis (or None for a full reduction).

    Returns:
        The masked mean; NaN wherever nothing was left to average.
    """
    v = np.asarray(values, dtype=np.float64)
    if mask is None:
        m = np.ones_like(v, dtype=bool)
    else:
        m = np.broadcast_to(np.asarray(mask, dtype=bool), v.shape)
    m = m & np.isfinite(v)
    num = np.sum(np.where(m, v, 0.0), axis=axis)
    den = np.sum(m, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return out


# --------------------------------------------------------------------------
# 9.1 Prediction accuracy
# --------------------------------------------------------------------------


def displacement_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-step Euclidean displacement error [m].

    Args:
        pred: ``[..., H, 2]`` predicted positions [m].
        gt: ``[..., H, 2]`` true positions [m], broadcastable against ``pred``.

    Returns:
        ``[..., H]`` errors [m].
    """
    return np.linalg.norm(_as2(pred) - _as2(gt), axis=-1)


def ade_per_sample(pred: np.ndarray, gt: np.ndarray,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Average Displacement Error [m] per sample: mean over the valid steps.

    Args:
        pred: ``[..., H, 2]`` predicted positions [m].
        gt: ``[..., H, 2]`` true positions [m].
        mask: ``[..., H]`` boolean, which steps are scored.

    Returns:
        ``[...]`` per-sample ADE [m]; NaN where a sample has no valid step.
    """
    return _masked_mean(displacement_error(pred, gt), mask, axis=-1)


def fde_per_sample(pred: np.ndarray, gt: np.ndarray,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Final Displacement Error [m] per sample.

    With a mask, the "final" step is the LAST VALID step of that sample, not
    blindly ``H-1``; a target that disappears at step 20 of 30 is scored at
    step 20 instead of being compared against nothing.

    Args:
        pred: ``[..., H, 2]`` predicted positions [m].
        gt: ``[..., H, 2]`` true positions [m].
        mask: ``[..., H]`` boolean, which steps are scored.

    Returns:
        ``[...]`` per-sample FDE [m]; NaN where a sample has no valid step.
    """
    err = displacement_error(pred, gt)                       # [..., H]
    H = err.shape[-1]
    if mask is None:
        m = np.isfinite(err)
    else:
        m = np.broadcast_to(np.asarray(mask, dtype=bool), err.shape) & np.isfinite(err)
    idx = np.arange(H)
    last = np.where(m, idx, -1).max(axis=-1)                 # [...]
    ok = last >= 0
    safe = np.where(ok, last, 0)
    picked = np.take_along_axis(err, safe[..., None], axis=-1)[..., 0]
    return np.where(ok, picked, np.nan)


def ade(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Dataset-level ADE [m]: mean of the per-sample ADE.

    Two-stage averaging (steps within a sample, then samples) is the standard
    definition and keeps every sample equally weighted regardless of how many
    steps it contributed.
    """
    return float(_masked_mean(ade_per_sample(pred, gt, mask), None, axis=None))


def fde(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Dataset-level FDE [m]: mean of the per-sample FDE."""
    return float(_masked_mean(fde_per_sample(pred, gt, mask), None, axis=None))


def min_ade_k(pred_k: np.ndarray, gt: np.ndarray,
              mask: np.ndarray | None = None) -> float:
    """minADE_k [m] for a multimodal prediction (doc 9.1).

    For each sample the best of the K hypotheses is taken, then averaged over
    samples. This is the standard "at least one mode was right" score used for
    multimodal trajectory prediction; it must always be reported next to the
    unimodal ADE, because minADE_k improves trivially with larger K.

    Args:
        pred_k: ``[..., K, H, 2]`` predicted positions [m], K hypotheses.
        gt: ``[..., H, 2]`` true positions [m].
        mask: ``[..., H]`` boolean over the true trajectory's steps.

    Returns:
        minADE_k [m].
    """
    p = _as2(pred_k)
    g = _as2(gt)[..., None, :, :]                            # [..., 1, H, 2]
    m = None if mask is None else np.asarray(mask, dtype=bool)[..., None, :]
    per_mode = ade_per_sample(p, g, m)                       # [..., K]
    best = np.nanmin(np.where(np.isfinite(per_mode), per_mode, np.inf), axis=-1)
    best = np.where(np.isfinite(best), best, np.nan)
    return float(_masked_mean(best, None, axis=None))


def min_fde_k(pred_k: np.ndarray, gt: np.ndarray,
              mask: np.ndarray | None = None) -> float:
    """minFDE_k [m] for a multimodal prediction (doc 9.1).

    Args:
        pred_k: ``[..., K, H, 2]`` predicted positions [m].
        gt: ``[..., H, 2]`` true positions [m].
        mask: ``[..., H]`` boolean over the true trajectory's steps.

    Returns:
        minFDE_k [m].
    """
    p = _as2(pred_k)
    g = _as2(gt)[..., None, :, :]
    m = None if mask is None else np.asarray(mask, dtype=bool)[..., None, :]
    per_mode = fde_per_sample(p, g, m)                       # [..., K]
    best = np.nanmin(np.where(np.isfinite(per_mode), per_mode, np.inf), axis=-1)
    best = np.where(np.isfinite(best), best, np.nan)
    return float(_masked_mean(best, None, axis=None))


# --------------------------------------------------------------------------
# 9.2 Distribution fit
# --------------------------------------------------------------------------


def _cov_invdet(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form inverse and determinant of a stack of 2x2 covariances.

    Args:
        cov: ``[..., 2, 2]`` covariance matrices [m^2].

    Returns:
        ``(inv, det)`` with shapes ``[...,2,2]`` and ``[...]``.

    Raises:
        ValueError: if any determinant is not strictly positive (a covariance
            must be positive definite; silently regularising would fake the
            calibration result the paper is about).
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.shape[-2:] != (2, 2):
        raise ValueError(f"covariance must be [...,2,2], got {c.shape}")
    a = c[..., 0, 0]
    b = c[..., 0, 1]
    d = c[..., 1, 0]
    e = c[..., 1, 1]
    det = a * e - b * d
    if np.any(~(det > 0.0)):
        raise ValueError("covariance is singular or not positive definite")
    inv = np.empty_like(c)
    inv[..., 0, 0] = e / det
    inv[..., 0, 1] = -b / det
    inv[..., 1, 0] = -d / det
    inv[..., 1, 1] = a / det
    return inv, det


def mahalanobis_sq(mu: np.ndarray, cov: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance [-] of ``y`` under ``N(mu, cov)``.

    Args:
        mu: ``[..., 2]`` predicted means [m].
        cov: ``[..., 2, 2]`` covariances [m^2].
        y: ``[..., 2]`` observed positions [m].

    Returns:
        ``[...]`` squared Mahalanobis distances (dimensionless).
    """
    inv, _det = _cov_invdet(cov)
    d = _as2(y) - _as2(mu)
    return np.einsum("...i,...ij,...j->...", d, inv, d)


def gaussian_nll(mu: np.ndarray, cov: np.ndarray, y: np.ndarray,
                 mask: np.ndarray | None = None,
                 reduce: bool = True) -> float | np.ndarray:
    """Negative log-likelihood [nats] of a 2-D Gaussian prediction (doc 9.2).

    ``NLL = 0.5 * (d^T S^-1 d + ln det S + 2 ln 2pi)`` with ``d = y - mu``.
    The constant term is kept so the value is a real log density and can be
    compared across papers, not just across our own ablations.

    Args:
        mu: ``[..., 2]`` predicted means [m].
        cov: ``[..., 2, 2]`` predicted covariances [m^2].
        y: ``[..., 2]`` true positions [m].
        mask: ``[...]`` boolean, which entries are scored.
        reduce: return the masked mean (True) or the per-entry array (False).

    Returns:
        Mean NLL [nats] or the per-entry array.
    """
    inv, det = _cov_invdet(cov)
    d = _as2(y) - _as2(mu)
    quad = np.einsum("...i,...ij,...j->...", d, inv, d)
    nll = 0.5 * (quad + np.log(det) + 2.0 * math.log(2.0 * math.pi))
    if not reduce:
        return nll
    return float(_masked_mean(nll, mask, axis=None))


def gaussian_nll_diag(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                      mask: np.ndarray | None = None,
                      reduce: bool = True) -> float | np.ndarray:
    """NLL [nats] for an axis-aligned Gaussian given per-axis sigmas.

    Args:
        mu: ``[..., 2]`` predicted means [m].
        sigma: ``[..., 2]`` predicted standard deviations [m], > 0.
        y: ``[..., 2]`` true positions [m].
        mask: ``[...]`` boolean, which entries are scored.
        reduce: return the masked mean (True) or the per-entry array (False).

    Returns:
        Mean NLL [nats] or the per-entry array.

    Raises:
        ValueError: if any sigma is not strictly positive.
    """
    s = np.asarray(sigma, dtype=np.float64)
    if np.any(~(s > 0.0)):
        raise ValueError("sigma must be strictly positive")
    d = _as2(y) - _as2(mu)
    z = d / s
    nll = np.sum(0.5 * z * z + np.log(s) + 0.5 * math.log(2.0 * math.pi), axis=-1)
    if not reduce:
        return nll
    return float(_masked_mean(nll, mask, axis=None))


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                  mask: np.ndarray | None = None,
                  reduce: bool = True) -> float | np.ndarray:
    """Continuous Ranked Probability Score [m] of a scalar Gaussian (doc 9.2).

    Closed form (Gneiting & Raftery)::

        CRPS = sigma * ( z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ),
        z = (y - mu) / sigma

    CRPS has the unit of the predicted variable [m] and is minimised by the
    true predictive distribution, which is exactly the property the
    calibration claim of doc 9.3 needs. Apply it per coordinate; the caller
    averages over x and y, or uses :func:`energy_score` for a genuinely
    multivariate score.

    Args:
        mu: predicted means [m], any shape.
        sigma: predicted standard deviations [m], > 0, broadcastable to ``mu``.
        y: observations [m], broadcastable to ``mu``.
        mask: boolean, which entries are scored.
        reduce: return the masked mean (True) or the per-entry array (False).

    Returns:
        Mean CRPS [m] or the per-entry array.

    Raises:
        ValueError: if any sigma is not strictly positive.
    """
    m_ = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    o = np.asarray(y, dtype=np.float64)
    if np.any(~(s > 0.0)):
        raise ValueError("sigma must be strictly positive")
    z = (o - m_) / s
    crps = s * (z * (2.0 * _std_normal_cdf(z) - 1.0)
                + 2.0 * _std_normal_pdf(z) - _INV_SQRT_PI)
    if not reduce:
        return crps
    return float(_masked_mean(crps, mask, axis=None))


def crps_ensemble(samples: np.ndarray, y: np.ndarray,
                  reduce: bool = True) -> float | np.ndarray:
    """CRPS [m] of a scalar ENSEMBLE forecast (branch rollouts, doc 3.3).

    Uses the exact empirical form::

        CRPS = mean_i |x_i - y| - (1 / (2 M^2)) * sum_i sum_j |x_i - x_j|

    computed in ``O(M log M)`` via the sorted identity
    ``sum_i sum_j |x_i - x_j| = 4 * sum_i (i - (M-1)/2) * x_(i)``.

    Args:
        samples: ``[..., M]`` ensemble members [m] (M >= 1).
        y: ``[...]`` observations [m].
        reduce: return the mean (True) or the per-entry array (False).

    Returns:
        Mean CRPS [m] or the per-entry array.

    Raises:
        ValueError: if the ensemble axis is empty.
    """
    x = np.asarray(samples, dtype=np.float64)
    o = np.asarray(y, dtype=np.float64)
    M = x.shape[-1]
    if M < 1:
        raise ValueError("crps_ensemble: the ensemble axis must be non-empty")
    term1 = np.mean(np.abs(x - o[..., None]), axis=-1)
    xs = np.sort(x, axis=-1)
    i = np.arange(M, dtype=np.float64)
    weights = 2.0 * i - (M - 1.0)
    pair_sum = 2.0 * np.sum(xs * weights, axis=-1)   # = sum_i sum_j |x_i - x_j|
    crps = term1 - pair_sum / (2.0 * M * M)
    if not reduce:
        return crps
    return float(np.mean(crps))


def energy_score(samples: np.ndarray, y: np.ndarray,
                 reduce: bool = True) -> float | np.ndarray:
    """Energy score [m] -- the multivariate generalisation of CRPS.

    ``ES = mean_i ||x_i - y|| - 0.5 * mean_ij ||x_i - x_j||``. This is the
    right score for the 2-D branch-rollout distribution of doc 3.3, because it
    penalises a prediction that gets each coordinate's marginal right while
    getting their joint structure (the actual fork in the road) wrong.

    Args:
        samples: ``[..., M, 2]`` ensemble members [m].
        y: ``[..., 2]`` observations [m].
        reduce: return the mean (True) or the per-entry array (False).

    Returns:
        Mean energy score [m] or the per-entry array.

    Raises:
        ValueError: if the ensemble axis is empty.
    """
    x = _as2(samples)
    o = _as2(y)
    M = x.shape[-2]
    if M < 1:
        raise ValueError("energy_score: the ensemble axis must be non-empty")
    term1 = np.mean(np.linalg.norm(x - o[..., None, :], axis=-1), axis=-1)
    diff = x[..., :, None, :] - x[..., None, :, :]          # [..., M, M, 2]
    term2 = np.mean(np.linalg.norm(diff, axis=-1), axis=(-2, -1))
    es = term1 - 0.5 * term2
    if not reduce:
        return es
    return float(np.mean(es))


# --------------------------------------------------------------------------
# 9.3 Calibration -- the core of the paper
# --------------------------------------------------------------------------


def chi2_quantile_2dof(p: float | np.ndarray) -> np.ndarray:
    """Quantile of the chi-square distribution with 2 degrees of freedom.

    For 2 dof the CDF is ``F(x) = 1 - exp(-x/2)``, so the quantile is exactly
    ``-2 ln(1 - p)``. No special-function library is needed, which is why the
    2-D coverage test below is closed form.

    Args:
        p: probability level in [0, 1).

    Returns:
        The quantile [-] (dimensionless, a squared Mahalanobis radius).

    Raises:
        ValueError: if ``p`` is outside [0, 1).
    """
    q = np.asarray(p, dtype=np.float64)
    if np.any(q < 0.0) or np.any(q >= 1.0):
        raise ValueError("chi2_quantile_2dof: p must be in [0, 1)")
    return -2.0 * np.log1p(-q)


def empirical_coverage(mu: np.ndarray, cov: np.ndarray, y: np.ndarray,
                       level: float = 0.9,
                       mask: np.ndarray | None = None) -> float:
    """Fraction of truths inside the nominal ``level`` confidence ellipse.

    Doc 9.3: "does the 90% region actually contain 90%". A well-calibrated
    predictor returns ``level``; a typical over-confident deep model returns
    substantially less.

    Args:
        mu: ``[..., 2]`` predicted means [m].
        cov: ``[..., 2, 2]`` predicted covariances [m^2].
        y: ``[..., 2]`` true positions [m].
        level: nominal probability level in [0, 1).
        mask: ``[...]`` boolean, which entries are scored.

    Returns:
        Empirical coverage in [0, 1]; NaN if nothing was scored.
    """
    d2 = mahalanobis_sq(mu, cov, y)
    inside = (d2 <= chi2_quantile_2dof(float(level))).astype(np.float64)
    return float(_masked_mean(inside, mask, axis=None))


def reliability_curve(mu: np.ndarray, cov: np.ndarray, y: np.ndarray,
                      levels: Sequence[float] | None = None,
                      mask: np.ndarray | None = None) -> dict[str, Any]:
    """Reliability (calibration) curve for 2-D Gaussian predictions (doc 9.3).

    The plot the paper needs: x = stated probability, y = observed frequency.
    A calibrated model sits on the diagonal; the baselines do not.

    Args:
        mu: ``[..., 2]`` predicted means [m].
        cov: ``[..., 2, 2]`` predicted covariances [m^2].
        y: ``[..., 2]`` true positions [m].
        levels: nominal levels to evaluate; default 0.05 .. 0.95 in 19 steps.
        mask: ``[...]`` boolean, which entries are scored.

    Returns:
        ``{'nominal': [L] float, 'empirical': [L] float, 'n': int,
        'ece': float, 'mce': float}`` where ECE/MCE are the mean and maximum
        absolute gap to the diagonal.
    """
    lv = (np.linspace(0.05, 0.95, 19) if levels is None
          else np.asarray(list(levels), dtype=np.float64))
    d2 = mahalanobis_sq(mu, cov, y)
    if mask is None:
        m = np.isfinite(d2)
    else:
        m = np.broadcast_to(np.asarray(mask, dtype=bool), d2.shape) & np.isfinite(d2)
    vals = d2[m]
    n = int(vals.size)
    emp = np.full(lv.shape, np.nan, dtype=np.float64)
    if n:
        thresholds = chi2_quantile_2dof(lv)
        emp = np.array([float(np.mean(vals <= th)) for th in thresholds])
    gap = np.abs(emp - lv)
    return {
        "nominal": [float(v) for v in lv],
        "empirical": [float(v) for v in emp],
        "n": n,
        "ece": float(np.mean(gap)) if n else float("nan"),
        "mce": float(np.max(gap)) if n else float("nan"),
    }


def pit_values(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
               mask: np.ndarray | None = None) -> np.ndarray:
    """Probability Integral Transform values for a scalar Gaussian (doc 9.3).

    ``PIT = Phi((y - mu) / sigma)``. Under perfect calibration these are
    uniform on [0, 1], so the PIT histogram is flat; a U shape means
    over-confidence and a dome means under-confidence.

    Args:
        mu: predicted means [m].
        sigma: predicted standard deviations [m], > 0.
        y: observations [m].
        mask: boolean; masked-out entries are dropped from the output.

    Returns:
        A flat 1-D array of PIT values in [0, 1].

    Raises:
        ValueError: if any scored sigma is not strictly positive.
    """
    m_ = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    o = np.asarray(y, dtype=np.float64)
    z = (o - m_) / s
    keep = np.isfinite(z)
    if mask is not None:
        keep = keep & np.broadcast_to(np.asarray(mask, dtype=bool), z.shape)
    if np.any(~(np.broadcast_to(s, z.shape)[keep] > 0.0)):
        raise ValueError("sigma must be strictly positive where scored")
    return _std_normal_cdf(z[keep])


def calibration_error(curve: Mapping[str, Any]) -> float:
    """Expected calibration error [-] of a :func:`reliability_curve` result."""
    return float(curve["ece"])


# --------------------------------------------------------------------------
# 9.4 Downstream safety
# --------------------------------------------------------------------------

#: An episode is labelled LOW risk when the minimum separation stayed above
#: the DWC HMD threshold but below this multiple of it -- "well clear margin
#: eroded but not violated" (see core.types.RiskLevel). A convention of this
#: study, not a standard.
LOW_RISK_MARGIN_FACTOR: float = 2.0


def separation_distances(own_pos: np.ndarray, gt_pos: np.ndarray,
                         gt_valid: np.ndarray | None = None) -> np.ndarray:
    """Horizontal ownship-to-agent distance [m] at every step and slot.

    Args:
        own_pos: ``[T,2]`` ownship positions [m].
        gt_pos: ``[T,N,2]`` agent positions [m] (NaN where invalid).
        gt_valid: ``[T,N]`` validity mask; when given, invalid entries are
            forced to NaN so they cannot win a minimum.

    Returns:
        ``[T,N]`` distances [m], NaN where the agent is not valid.
    """
    o = np.asarray(own_pos, dtype=np.float64)
    g = np.asarray(gt_pos, dtype=np.float64)
    d = np.linalg.norm(g - o[:, None, :], axis=-1)
    if gt_valid is not None:
        d = np.where(np.asarray(gt_valid, dtype=bool), d, np.nan)
    return d


def min_separation(own_pos: np.ndarray, gt_pos: np.ndarray,
                   gt_valid: np.ndarray | None = None) -> dict[str, Any]:
    """Minimum horizontal separation [m] over an episode (doc 9.4).

    Args:
        own_pos: ``[T,2]`` ownship positions [m].
        gt_pos: ``[T,N,2]`` agent positions [m].
        gt_valid: ``[T,N]`` validity mask.

    Returns:
        ``{'min_separation_m': float, 'per_agent_m': [N] float,
        'argmin_step': int, 'argmin_slot': int}``. ``min_separation_m`` is
        ``inf`` and the indices are ``-1`` if no agent was ever valid.
    """
    d = separation_distances(own_pos, gt_pos, gt_valid)
    per_agent = np.full(d.shape[1], np.inf, dtype=np.float64)
    finite = np.isfinite(d)
    for j in range(d.shape[1]):
        col = d[:, j][finite[:, j]]
        if col.size:
            per_agent[j] = float(np.min(col))
    if not np.any(finite):
        return {"min_separation_m": float("inf"), "per_agent_m": per_agent,
                "argmin_step": -1, "argmin_slot": -1}
    filled = np.where(finite, d, np.inf)
    flat = int(np.argmin(filled))
    step, slot = np.unravel_index(flat, filled.shape)
    return {
        "min_separation_m": float(filled[step, slot]),
        "per_agent_m": per_agent,
        "argmin_step": int(step),
        "argmin_slot": int(slot),
    }


def nmac_events(own_pos: np.ndarray, gt_pos: np.ndarray,
                gt_valid: np.ndarray | None = None,
                radius_m: float = NMAC_RADIUS_M) -> dict[str, Any]:
    """Near Mid-Air Collision detection (doc 9.4, horizontal component only).

    The 2D scope of this paper means only the horizontal 500 ft condition of
    the usual NMAC definition is applied; the paper must state that.

    Args:
        own_pos: ``[T,2]`` ownship positions [m].
        gt_pos: ``[T,N,2]`` agent positions [m].
        gt_valid: ``[T,N]`` validity mask.
        radius_m: NMAC horizontal radius [m].

    Returns:
        ``{'occurred': bool, 'n_agents': int, 'n_steps': int,
        'agent_slots': list[int], 'first_step': int}`` -- ``n_agents`` counts
        distinct agents that breached, ``n_steps`` counts (step, agent) pairs
        below the radius, ``first_step`` is ``-1`` if none.
    """
    d = separation_distances(own_pos, gt_pos, gt_valid)
    breach = np.isfinite(d) & (d < float(radius_m))
    slots = [int(j) for j in np.nonzero(np.any(breach, axis=0))[0]]
    steps = np.nonzero(np.any(breach, axis=1))[0]
    return {
        "occurred": bool(len(slots) > 0),
        "n_agents": int(len(slots)),
        "n_steps": int(np.count_nonzero(breach)),
        "agent_slots": slots,
        "first_step": int(steps[0]) if steps.size else -1,
    }


def tau_mod(rel_pos: np.ndarray, rel_vel: np.ndarray,
            dmod_m: float = DWC_HMD_M) -> np.ndarray:
    """Modified tau [s] -- the DAA Well Clear time-to-conflict term.

    ``tau_mod = (DMOD^2 - r^2) / (r * rdot)`` for a closing geometry
    (``rdot < 0``), where ``r = |rel_pos|`` and ``rdot`` is the range rate.
    It is 0 when the aircraft are already inside DMOD, and ``+inf`` when they
    are not closing.

    WARNING (doc 9.4 / 12.5): ``DWC_HMD_M`` and ``DWC_TAU_MOD_S`` are the
    commonly quoted DO-365 figures and MUST be verified against the standard's
    source text before any number derived from them is published.

    Args:
        rel_pos: ``[..., 2]`` intruder position relative to ownship [m].
        rel_vel: ``[..., 2]`` intruder velocity relative to ownship [m/s].
        dmod_m: distance modifier [m], the HMD threshold.

    Returns:
        ``[...]`` modified tau [s]; ``inf`` where not closing.
    """
    p = _as2(rel_pos)
    v = _as2(rel_vel)
    r = np.linalg.norm(p, axis=-1)
    dmod = float(dmod_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        rdot = np.sum(p * v, axis=-1) / r
        tm = (dmod * dmod - r * r) / (r * rdot)
    inside = r <= dmod
    closing = rdot < 0.0
    out = np.where(inside, 0.0, np.where(closing, tm, np.inf))
    return np.where(np.isfinite(r), out, np.inf)


def horizontal_miss_distance(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
    """Projected horizontal miss distance [m] at the closest point of approach.

    Straight-line projection of the current relative geometry: the distance at
    ``tcpa = -(p.v)/|v|^2`` when that is in the future, otherwise the current
    range (the CPA is already behind).

    Args:
        rel_pos: ``[..., 2]`` relative position [m].
        rel_vel: ``[..., 2]`` relative velocity [m/s].

    Returns:
        ``[...]`` HMD [m].
    """
    p = _as2(rel_pos)
    v = _as2(rel_vel)
    vv = np.sum(v * v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tcpa = np.where(vv > 0.0, -np.sum(p * v, axis=-1) / np.maximum(vv, 1e-12), 0.0)
    tcpa = np.maximum(tcpa, 0.0)
    closest = p + v * tcpa[..., None]
    return np.linalg.norm(closest, axis=-1)


def dwc_violation(own_pos: np.ndarray, own_vel: np.ndarray,
                  gt_pos: np.ndarray, gt_vel: np.ndarray,
                  gt_valid: np.ndarray | None = None,
                  hmd_m: float = DWC_HMD_M,
                  tau_s: float = DWC_TAU_MOD_S) -> dict[str, Any]:
    """DAA Well Clear violation test (doc 9.4, horizontal only).

    Well clear is lost when BOTH conditions hold at the same instant:
        * the projected horizontal miss distance is below ``hmd_m``, and
        * ``tau_mod`` is at or below ``tau_s`` (which includes the case of
          already being inside the HMD threshold, where ``tau_mod = 0``).

    This construction guarantees the invariant that
    :class:`storage.schema.EpisodeMeta` asserts: an NMAC (range < 152.4 m)
    forces range < HMD, hence ``tau_mod = 0`` and ``HMD_projected <= range``,
    so a DWC violation is always flagged too.

    WARNING: the thresholds are the commonly quoted DO-365 values and must be
    verified against the standard before publication.

    Args:
        own_pos: ``[T,2]`` ownship positions [m].
        own_vel: ``[T,2]`` ownship ground velocities [m/s].
        gt_pos: ``[T,N,2]`` agent positions [m].
        gt_vel: ``[T,N,2]`` agent ground velocities [m/s].
        gt_valid: ``[T,N]`` validity mask.
        hmd_m: HMD threshold [m].
        tau_s: tau_mod threshold [s].

    Returns:
        ``{'violated': bool, 'n_agents': int, 'n_steps': int,
        'agent_slots': list[int], 'first_step': int, 'min_tau_mod_s': float,
        'min_hmd_m': float}``.
    """
    o_p = np.asarray(own_pos, dtype=np.float64)[:, None, :]
    o_v = np.asarray(own_vel, dtype=np.float64)[:, None, :]
    rel_p = np.asarray(gt_pos, dtype=np.float64) - o_p
    rel_v = np.asarray(gt_vel, dtype=np.float64) - o_v

    valid = (np.ones(rel_p.shape[:2], dtype=bool) if gt_valid is None
             else np.asarray(gt_valid, dtype=bool))
    valid = valid & np.all(np.isfinite(rel_p), axis=-1) & np.all(np.isfinite(rel_v), axis=-1)

    safe_p = np.where(valid[..., None], rel_p, np.array([1e9, 0.0]))
    safe_v = np.where(valid[..., None], rel_v, np.array([1.0, 0.0]))

    tm = tau_mod(safe_p, safe_v, dmod_m=hmd_m)
    hmd = horizontal_miss_distance(safe_p, safe_v)
    viol = valid & (hmd < float(hmd_m)) & (tm <= float(tau_s))

    slots = [int(j) for j in np.nonzero(np.any(viol, axis=0))[0]]
    steps = np.nonzero(np.any(viol, axis=1))[0]
    tm_valid = tm[valid]
    hmd_valid = hmd[valid]
    return {
        "violated": bool(len(slots) > 0),
        "n_agents": int(len(slots)),
        "n_steps": int(np.count_nonzero(viol)),
        "agent_slots": slots,
        "first_step": int(steps[0]) if steps.size else -1,
        "min_tau_mod_s": float(np.min(tm_valid)) if tm_valid.size else float("inf"),
        "min_hmd_m": float(np.min(hmd_valid)) if hmd_valid.size else float("inf"),
    }


def risk_level_from_separation(min_separation_m: float, nmac: bool,
                               dwc: bool, alerted: Optional[bool] = None) -> int:
    """Map separation outcomes to a :class:`~core.types.RiskLevel` (doc 8.1).

    THE definition. :func:`simulation.episode.compute_episode_meta` calls this
    rather than repeating the rule, because doc 9.4 requires the dataset labels
    and the evaluation metrics to agree and two copies of a threshold ladder
    is how they stop agreeing.

    HIGH   -- an NMAC occurred.
    MEDIUM -- DAA Well Clear was actually LOST but no NMAC, i.e. the achieved
              minimum separation fell below ``DWC_HMD_M``.
    LOW    -- well clear was kept, but the DAA well-clear ALERTING predicate
              fired: at some instant the constant-velocity projection of the
              relative state was going to lose well clear within ``tau_mod``.
              An alert that did not become a violation.
    NONE   -- no alert at any point.

    The previous LOW rule ("separation stayed above HMD but below
    ``LOW_RISK_MARGIN_FACTOR`` x HMD") is kept as the fallback for callers that
    have no alert flag, but it was UNREACHABLE in the generator: any pair
    closing inside the HMD threshold has a small positive ``tau_mod`` and
    therefore satisfies the DWC predicate by construction, so LOW and NONE were
    dead enum members and every episode came out MEDIUM or HIGH.

    Args:
        min_separation_m: minimum horizontal separation [m] of the episode.
        nmac: whether an NMAC occurred.
        dwc: whether well clear was lost.
        alerted: whether the well-clear alerting predicate ever fired. ``None``
            falls back to the margin rule described above.

    Returns:
        The integer risk level.
    """
    if nmac:
        return int(RiskLevel.HIGH)
    if dwc:
        return int(RiskLevel.MEDIUM)
    if alerted is not None:
        return int(RiskLevel.LOW) if bool(alerted) else int(RiskLevel.NONE)
    if np.isfinite(min_separation_m) and \
            float(min_separation_m) < LOW_RISK_MARGIN_FACTOR * DWC_HMD_M:
        return int(RiskLevel.LOW)
    return int(RiskLevel.NONE)


def risk_stratification(risk_levels: Sequence[int]) -> dict[str, Any]:
    """Coverage of the four :class:`~core.types.RiskLevel` strata over a run.

    Doc 3.4 asks for a risk-STRATIFIED dataset. A four-level label column in
    which only two levels ever occur is a two-class dataset wearing a
    four-class label, and nothing in the pipeline noticed for 60 episodes.
    This is the check that makes that visible.

    Args:
        risk_levels: one integer :class:`~core.types.RiskLevel` per episode.

    Returns:
        ``{'counts': {name: n}, 'fractions': {name: f}, 'empty': [names],
        'n_episodes': int, 'n_populated': int}``.
    """
    levels = [int(r) for r in risk_levels]
    n = len(levels)
    counts = {r.name: 0 for r in RiskLevel}
    for r in levels:
        try:
            counts[RiskLevel(r).name] += 1
        except ValueError:
            continue
    empty = [k for k, v in counts.items() if v == 0]
    return {
        "counts": counts,
        "fractions": {k: (v / n if n else 0.0) for k, v in counts.items()},
        "empty": empty,
        "n_episodes": n,
        "n_populated": len(counts) - len(empty),
    }


def separation_metrics(ep: EpisodeArrays) -> dict[str, Any]:
    """All doc 9.4 safety numbers for one episode, ready for EpisodeMeta.

    This is the function :mod:`simulation.episode` should call to fill
    ``EpisodeMeta.min_separation_m``, ``nmac_occurred``, ``dwc_violated`` and
    ``risk_level``, so that the dataset labels and the evaluation metrics can
    never drift apart (doc 9.4 requires them to agree).

    Args:
        ep: the episode arrays.

    Returns:
        ``{'min_separation_m', 'risk_level', 'nmac_occurred', 'dwc_violated',
        'nmac', 'dwc', 'per_agent_min_m', 'cpa_step', 'cpa_slot',
        'cpa_time_s'}``.
    """
    own_pos = ep.own_pos.astype(np.float64)
    own_vel = ep.own_vel.astype(np.float64)
    gt_pos = ep.gt_pos.astype(np.float64)
    gt_vel = ep.gt_vel.astype(np.float64)
    valid = ep.gt_valid.astype(bool)

    sep = min_separation(own_pos, gt_pos, valid)
    nm = nmac_events(own_pos, gt_pos, valid)
    dw = dwc_violation(own_pos, own_vel, gt_pos, gt_vel, valid)
    step = int(sep["argmin_step"])
    return {
        "min_separation_m": float(sep["min_separation_m"]),
        "per_agent_min_m": sep["per_agent_m"],
        "cpa_step": step,
        "cpa_slot": int(sep["argmin_slot"]),
        "cpa_time_s": (float(ep.t[step]) if step >= 0 and step < ep.n_steps
                       else float("nan")),
        "nmac_occurred": bool(nm["occurred"]),
        "dwc_violated": bool(dw["violated"]),
        "risk_level": risk_level_from_separation(
            sep["min_separation_m"], nm["occurred"], dw["violated"]),
        "nmac": nm,
        "dwc": dw,
        "log_dt_s": float(LOG_DT),
    }
