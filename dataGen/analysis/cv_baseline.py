"""Constant-velocity extrapolation: performance floor AND data-difficulty gauge.

Doc 8.3 gives this module two jobs.

**Role 1 -- the mandatory performance floor.** ``future position = current
position + current velocity * dt``. No learning, no parameters. A reviewer's
first question about any trajectory-prediction paper is "how much better than
CV are you"; without this number the paper is rejected on sight (doc 8.2).

**Role 2 -- the data-difficulty diagnostic.** Doc 3.1 is blunt: if the
simulator only produces straight lines, CV becomes unbeatable and the whole
paper collapses. So the *first* thing to run after the simulator exists
(doc 10 step 2) is CV against the generated data:

    CV 15 s error far too small -> data too easy   -> raise turn frequency
    CV 15 s error far too large -> data too random -> raise mode dwell time
    somewhere in between        -> OK

Doc 8.3 explicitly says the judgement is **relative, not absolute** -- it
depends on the speed scale -- and offers one anchor: the median CV error at
15 s should land within the same order of magnitude as the NMAC radius
(:data:`core.constants.NMAC_RADIUS_M` = 152.4 m), because that is the length
scale at which a collision-avoidance decision is actually made. If CV is
already accurate to a small fraction of an NMAC radius, prediction is not the
bottleneck and the downstream RL problem is trivial. If CV is wrong by many
NMAC radii, the futures are so dispersed that no predictor -- ours included --
can say anything useful, and the calibration story degenerates into "sigma is
huge everywhere".

The thresholds below turn that qualitative anchor into reproducible numbers.
They are a **convention of this study**, not a standard, and the docstring of
:func:`diagnose` says so in the verdict payload as well. They are deliberately
tighter than a literal "same order of magnitude" (which would be a 0.1x..10x
band) because the useful working range is narrower than the band in which the
statement is merely true.

A final trap this module guards against: an aggregate "OK" can hide a class of
traffic that is trivially straight. A dataset where eVTOL cruise dominates and
never turns will look fine on average while teaching the model nothing about
the multimodality that the paper is built on. So :func:`diagnose` breaks the
verdict down **by PlatformType and by IntentType** as well.

WHY THERE IS NO SINGLE "RIGHT" CV ERROR
---------------------------------------
The classes are NOT meant to be equally hard, and grading them against one
global band would be a category error. A calibrated model must emit a SMALL
sigma for a stable eVTOL in corridor cruise and a LARGE one for a wandering
bird; that is the entire claim of doc section 1. Heterogeneous difficulty is
therefore a FEATURE of this dataset. If every object were equally
unpredictable there would be nothing for the model to modulate, and the
calibration figure would have nothing to show.

So each class is judged against ITS OWN expected band
(:data:`EXPECTED_DIFFICULTY`, :data:`BAND_LIMITS`):

    a low CV error on EVTOL_UAM  -> PASS. It is supposed to be easy.
    a low CV error on BIRD       -> FAILURE. It is supposed to defeat CV.

and the dataset-level gate (:func:`gate_report`) asks three questions instead
of one:

    (a) do the HARD classes genuinely defeat CV?
    (b) are the EASY classes genuinely easy?
    (c) is there clear SEPARATION between the two bands?

(c) is reported as the ratio of the HARD-class to the EASY-class median error
at 15 s and must exceed :data:`MIN_BAND_SEPARATION`. That spread is the thing
the calibration figure exploits.

Units: positions [m], velocities [m/s], horizons [s], errors [m].
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from core.constants import LOG_DT, NMAC_RADIUS_M
from core.kinematics import predict_cv
from core.types import INTENT_NAMES, TYPE_NAMES, IntentType, PlatformType
from storage.schema import EpisodeArrays

__all__ = [
    "DEFAULT_HORIZONS_S",
    "DIAGNOSTIC_HORIZON_S",
    "TOO_EASY_FRACTION_OF_NMAC",
    "TOO_RANDOM_MULTIPLE_OF_NMAC",
    "TOO_EASY_MEDIAN_M",
    "TOO_RANDOM_MEDIAN_M",
    "MIN_SAMPLES_FOR_VERDICT",
    "EASY",
    "MEDIUM",
    "HARD",
    "EXPECTED_DIFFICULTY",
    "BAND_LIMITS",
    "MIN_BAND_SEPARATION",
    "expected_difficulty",
    "band_of_platform",
    "band_of_intent",
    "VERDICT_OK",
    "VERDICT_TOO_EASY",
    "VERDICT_TOO_RANDOM",
    "VERDICT_INSUFFICIENT",
    "CVErrors",
    "cv_predict",
    "cv_errors",
    "diagnose",
    "gate_report",
    "run_diagnostic",
    "format_report",
]


# --------------------------------------------------------------------------
# Thresholds (a convention of this study -- see the module docstring)
# --------------------------------------------------------------------------

#: Horizons [s] reported by default: the short, mid and full prediction range.
DEFAULT_HORIZONS_S: tuple[float, ...] = (5.0, 10.0, 15.0)

#: Horizon [s] the verdict is taken at. It is the full prediction horizon of
#: the paper (doc 3.7: 15 s = 30 logged steps), because that is where a
#: predictor's structural advantage over CV either exists or does not.
DIAGNOSTIC_HORIZON_S: float = 15.0

#: Below ``TOO_EASY_FRACTION_OF_NMAC * NMAC_RADIUS_M`` the median CV error is
#: so small that a CV extrapolator already resolves the traffic to well inside
#: a near-mid-air-collision radius. Prediction cannot be the bottleneck, so the
#: dataset carries no evidence for the paper's claim. 0.25 is chosen so that
#: the whole 90% error mass would still fit inside one NMAC radius.
TOO_EASY_FRACTION_OF_NMAC: float = 0.25

#: Above ``TOO_RANDOM_MULTIPLE_OF_NMAC * NMAC_RADIUS_M`` the median CV error
#: exceeds several NMAC radii, i.e. the true future is dispersed over an area
#: far larger than the conflict geometry itself. 3.0 keeps the median inside
#: about one third of the DAA Well Clear HMD threshold (1219.2 m), so a
#: violation of well clear is still a *predictable* event rather than noise.
TOO_RANDOM_MULTIPLE_OF_NMAC: float = 3.0

#: Derived absolute thresholds [m] on the median error at DIAGNOSTIC_HORIZON_S.
TOO_EASY_MEDIAN_M: float = TOO_EASY_FRACTION_OF_NMAC * NMAC_RADIUS_M     # 38.1 m
TOO_RANDOM_MEDIAN_M: float = TOO_RANDOM_MULTIPLE_OF_NMAC * NMAC_RADIUS_M  # 457.2 m

#: Fewer samples than this and no verdict is issued: a per-class breakdown on
#: a handful of samples says more about sampling noise than about the data.
MIN_SAMPLES_FOR_VERDICT: int = 200

VERDICT_OK: str = "OK"
VERDICT_TOO_EASY: str = "TOO_EASY: increase turn frequency"
VERDICT_TOO_RANDOM: str = "TOO_RANDOM: increase mode dwell time"
VERDICT_INSUFFICIENT: str = "INSUFFICIENT_DATA"


# --------------------------------------------------------------------------
# Expected difficulty per class (see the module docstring)
# --------------------------------------------------------------------------

EASY: str = "EASY"
MEDIUM: str = "MEDIUM"
HARD: str = "HARD"

#: What each class is DESIGNED to be, given the per-platform stability model of
#: :data:`agents.platforms.STABILITY_PROFILES` and the intent difficulty ladder
#: of doc 3.2. Keyed by :class:`~core.types.PlatformType` and by
#: :class:`~core.types.IntentType`; the two axes are graded independently.
#:
#:   EASY   long manoeuvre commitment / low mode authority: an eVTOL or GA
#:          aircraft in cruise, or anything on a corridor. CV should track it.
#:   MEDIUM committed but frequent manoeuvres: sUAS classes and task-driven
#:          intents.
#:   HARD   short commitment, high authority, large turn-rate cap: birds, the
#:          OOD class, and the wandering / reacting intents. CV must fail here
#:          or the paper has no evidence for a probabilistic predictor.
#:
#: PASSIVE_DRIFT is EASY (an unpowered body advected by a slowly varying wind
#: is very nearly constant-velocity) and DEBRIS is EASY for the same reason --
#: note that "easy" is about predictability, NOT about being harmless.
EXPECTED_DIFFICULTY: dict[str, dict[int, str]] = {
    "platform": {
        int(PlatformType.EVTOL_UAM): EASY,
        int(PlatformType.GA): EASY,
        int(PlatformType.DEBRIS): EASY,
        int(PlatformType.FIXED_WING_UAS): MEDIUM,
        int(PlatformType.MULTIROTOR): MEDIUM,
        int(PlatformType.BIRD): HARD,
        int(PlatformType.UNKNOWN): HARD,
    },
    "intent": {
        int(IntentType.CORRIDOR_TRANSIT): EASY,
        int(IntentType.PASSIVE_DRIFT): EASY,
        int(IntentType.DELIVERY_WP): MEDIUM,
        int(IntentType.LOITER_HOVER): MEDIUM,
        int(IntentType.SEARCH_WANDER): HARD,
        int(IntentType.REACTIVE_AVOID): HARD,
    },
}

#: Acceptable band [m] for the median 15 s CV error of a class, per expected
#: difficulty. Every number is a multiple of the NMAC radius R = 152.4 m,
#: because that is the length scale at which a collision-avoidance decision is
#: actually made -- a position error is "large" only relative to the protection
#: volume it might put an intruder inside or outside of:
#:
#:   EASY   (0.05 .. 1.00) x R. Upper: CV must resolve the class to inside one
#:          NMAC radius -- that is what "predictable" means operationally.
#:          Lower: below ~7.6 m over 15 s a POWERED object is flying a pure
#:          straight line and carries no signal at all, which is degenerate
#:          rather than easy. (Objects that are constant-velocity by
#:          construction are exempt: see :data:`CV_BY_CONSTRUCTION`.)
#:   MEDIUM (0.25 .. 3.00) x R -- the original doc 8.3 global band.
#:   HARD   (0.50 .. 3.00) x R. Lower: at half the protection radius the CV
#:          prediction and the truth no longer agree about the conflict -- the
#:          predicted and actual NMAC disks overlap by well under two thirds of
#:          their area -- so an avoidance decision taken on a CV estimate is
#:          materially wrong, which is exactly the claim the HARD classes exist
#:          to support. A stricter 0.75 R was considered and rejected: it is
#:          not tied to anything geometric, and it would grade REACTIVE_AVOID
#:          (median 0.57 R, p90 1.68 R) as too easy purely because most of its
#:          time origins fall in the monitoring phase between latched evasions.
#:          Upper: the same TOO_RANDOM ceiling as everything else -- beyond
#:          three NMAC radii no predictor can say anything useful and
#:          calibration degenerates to "sigma is huge everywhere".
#:
#: The EASY and HARD bands deliberately OVERLAP. They are not a partition of
#: the error axis; they are two different questions asked of two different
#: kinds of traffic. What keeps them apart in practice is
#: :data:`MIN_BAND_SEPARATION`, which is measured on the data rather than
#: assumed by the thresholds.
BAND_LIMITS: dict[str, tuple[float, float]] = {
    EASY: (0.05 * NMAC_RADIUS_M, 1.00 * NMAC_RADIUS_M),    #   7.6 .. 152.4 m
    MEDIUM: (0.25 * NMAC_RADIUS_M, 3.00 * NMAC_RADIUS_M),  #  38.1 .. 457.2 m
    HARD: (0.50 * NMAC_RADIUS_M, 3.00 * NMAC_RADIUS_M),    #  76.2 .. 457.2 m
}

#: Classes that are constant-velocity BY CONSTRUCTION, and are therefore exempt
#: from the lower bound of their band (the upper bound still applies).
#:
#: An unpowered body has no manoeuvre modes at all: :mod:`agents.base` routes it
#: to the passive integrator, where it is advected by the wind and slowed by
#: drag. Under a steady wind that is a straight line, so CV predicts it almost
#: exactly -- around 5 m at 15 s. That is the physics of the class, not a
#: degenerate simulator, and the doc 8.3 advice attached to TOO_EASY ("increase
#: turn frequency") is meaningless for an object with no thrust. Flagging it
#: would train the reader to ignore the flag.
#:
#: The exemption is deliberately narrow: it covers only DEBRIS and its forced
#: PASSIVE_DRIFT intent. A POWERED class that collapses to a straight line is
#: still a real defect and is still reported.
CV_BY_CONSTRUCTION: dict[str, frozenset[int]] = {
    "platform": frozenset({int(PlatformType.DEBRIS)}),
    "intent": frozenset({int(IntentType.PASSIVE_DRIFT)}),
}

#: Required ratio [-] of the HARD-band to the EASY-band median 15 s error.
#:
#: 2.0 is chosen, not 1.0-and-a-bit, for two reasons. (1) Meaning: the ratio of
#: CV errors is the ratio of the sigmas a calibrated model has to emit, and a
#: factor below ~2 puts the two predictive distributions on top of each other
#: -- a sigma-versus-class figure would show no visible modulation, so the
#: dataset could not evidence the claim even if the model were perfect.
#: (2) Noise: a per-class median over a few thousand samples carries a sampling
#: error of a few per cent, so 2.0 is many standard errors clear of 1.0 and the
#: gate cannot pass or fail on sampling luck.
MIN_BAND_SEPARATION: float = 2.0


# --------------------------------------------------------------------------
# Error container
# --------------------------------------------------------------------------


@dataclass
class CVErrors:
    """Per-horizon CV extrapolation errors with parallel class labels.

    For every horizon the three arrays have the same length: one entry per
    (episode, agent slot, time origin) sample that had a complete future
    window. Different horizons have different lengths, because a longer
    horizon admits fewer time origins.

    Attributes:
        horizons_s: the horizons [s] in reporting order.
        errors: horizon [s] -> [S] float64 Euclidean position error [m].
        platform: horizon [s] -> [S] int8 :class:`~core.types.PlatformType`
            of the agent the sample came from.
        intent: horizon [s] -> [S] int8 :class:`~core.types.IntentType` of the
            agent **at the time origin** (intent can change during a flight).
        n_episodes: number of episodes the samples were drawn from [-].
    """

    horizons_s: tuple[float, ...] = DEFAULT_HORIZONS_S
    errors: dict[float, np.ndarray] = field(default_factory=dict)
    platform: dict[float, np.ndarray] = field(default_factory=dict)
    intent: dict[float, np.ndarray] = field(default_factory=dict)
    n_episodes: int = 0

    @classmethod
    def empty(cls, horizons_s: Sequence[float] = DEFAULT_HORIZONS_S) -> "CVErrors":
        """Create an empty container ready to be extended."""
        hs = tuple(float(h) for h in horizons_s)
        return cls(
            horizons_s=hs,
            errors={h: np.zeros((0,), dtype=np.float64) for h in hs},
            platform={h: np.zeros((0,), dtype=np.int8) for h in hs},
            intent={h: np.zeros((0,), dtype=np.int8) for h in hs},
            n_episodes=0,
        )

    def extend(self, other: "CVErrors") -> "CVErrors":
        """Append another container's samples in place and return ``self``.

        Raises:
            ValueError: if the horizon sets differ.
        """
        if tuple(self.horizons_s) != tuple(other.horizons_s):
            raise ValueError(
                f"cannot merge CVErrors with horizons {other.horizons_s} into "
                f"{self.horizons_s}"
            )
        for h in self.horizons_s:
            self.errors[h] = np.concatenate([self.errors[h], other.errors[h]])
            self.platform[h] = np.concatenate([self.platform[h], other.platform[h]])
            self.intent[h] = np.concatenate([self.intent[h], other.intent[h]])
        self.n_episodes += int(other.n_episodes)
        return self

    def n_samples(self, horizon_s: float | None = None) -> int:
        """Number of samples [-] at one horizon (default: the diagnostic one)."""
        h = self._pick(horizon_s)
        return int(self.errors[h].size)

    def _pick(self, horizon_s: float | None) -> float:
        """Resolve a horizon key, defaulting to the longest available one."""
        if horizon_s is None:
            return max(self.horizons_s) if self.horizons_s else 0.0
        h = float(horizon_s)
        if h in self.errors:
            return h
        raise KeyError(f"horizon {h} s not present; have {sorted(self.errors)}")


# --------------------------------------------------------------------------
# The baseline itself
# --------------------------------------------------------------------------


def cv_predict(pos: np.ndarray, vel: np.ndarray, horizons_s: Sequence[float]) -> np.ndarray:
    """Constant-velocity extrapolation -- the whole baseline, in one line.

    This is a thin adapter over :func:`core.kinematics.predict_cv`, which is
    the ONE definition of the CV model in the codebase. The baseline the paper
    is measured against must not exist twice: if the diagnostic and the
    simulator's own CV mode could drift apart, a "CV cannot predict this"
    claim would be unfalsifiable. The only thing added here is accepting any
    ``Sequence[float]`` for ``horizons_s`` (the diagnostic passes tuples).

    Args:
        pos: ``[..., 2]`` current positions [m].
        vel: ``[..., 2]`` current ground velocities [m/s].
        horizons_s: horizons [s] to extrapolate to.

    Returns:
        ``[..., H, 2]`` predicted positions [m], where ``H = len(horizons_s)``.
    """
    return predict_cv(pos, vel, np.asarray(list(horizons_s), dtype=np.float64))


def _velocity_field(ep: EpisodeArrays, velocity_source: str) -> np.ndarray:
    """Return the ``[T,N,2]`` velocity field [m/s] used as the CV state.

    Args:
        ep: the episode.
        velocity_source: ``'gt'`` uses the logged ground-truth ground velocity
            (the *best case* for CV, which is what a difficulty diagnostic
            wants: it removes sensor noise from the verdict). ``'diff'`` uses
            a backward finite difference of ground-truth positions, which is
            what a predictor that only sees positions could reconstruct; the
            first step falls back to the forward difference.

    Raises:
        ValueError: on an unknown source.
    """
    if velocity_source == "gt":
        return ep.gt_vel.astype(np.float64)
    if velocity_source == "diff":
        pos = ep.gt_pos.astype(np.float64)
        vel = np.full_like(pos, np.nan)
        if pos.shape[0] >= 2:
            vel[1:] = (pos[1:] - pos[:-1]) / float(LOG_DT)
            vel[0] = vel[1]
        return vel
    raise ValueError(f"velocity_source must be 'gt' or 'diff', got {velocity_source!r}")


def _contiguous_valid(valid: np.ndarray, span: int) -> np.ndarray:
    """Mask of origins whose next ``span`` steps are all valid.

    Args:
        valid: ``[T,N]`` boolean validity mask.
        span: number of steps [-] the window extends beyond the origin.

    Returns:
        ``[T,N]`` boolean mask, False for origins that run past the end of the
        episode or whose window contains an invalid step. Requiring the *whole*
        window (not just the endpoints) rejects an agent that left the slot and
        a different agent that later re-entered it.
    """
    T = valid.shape[0]
    out = valid.copy()
    for k in range(1, int(span) + 1):
        shifted = np.zeros_like(valid)
        if T > k:
            shifted[: T - k] = valid[k:]
        out &= shifted
    return out


def cv_errors(
    ep: EpisodeArrays,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    velocity_source: str = "gt",
) -> CVErrors:
    """Evaluate CV extrapolation over one episode.

    Every valid traffic agent and every time origin with a complete future
    window contributes one sample per horizon. The ownship is excluded: it is
    not a prediction target.

    Args:
        ep: the episode arrays.
        horizons_s: horizons [s]; each must be a whole multiple of
            :data:`core.constants.LOG_DT`.
        velocity_source: ``'gt'`` (default) or ``'diff'``; see
            :func:`_velocity_field`.

    Returns:
        A :class:`CVErrors` container.

    Raises:
        ValueError: if a horizon is not a multiple of LOG_DT.
    """
    hs = tuple(float(h) for h in horizons_s)
    out = CVErrors.empty(hs)
    out.n_episodes = 1

    T, N = ep.gt_valid.shape
    if T == 0 or N == 0:
        return out

    pos = ep.gt_pos.astype(np.float64)                # [T,N,2]
    vel = _velocity_field(ep, velocity_source)        # [T,N,2]
    valid = ep.gt_valid.astype(bool)                  # [T,N]
    gt_type = ep.gt_type.astype(np.int8)              # [N]
    gt_intent = ep.gt_intent.astype(np.int8)          # [T,N]

    for h in hs:
        steps_f = h / float(LOG_DT)
        steps = int(round(steps_f))
        if abs(steps_f - steps) > 1e-9 or steps <= 0:
            raise ValueError(
                f"horizon {h} s is not a positive whole multiple of LOG_DT="
                f"{LOG_DT} s"
            )
        if steps >= T:
            continue
        origins = _contiguous_valid(valid, steps)     # [T,N]
        origins[T - steps:] = False                   # no full future window
        if not np.any(origins):
            continue
        oi, oj = np.nonzero(origins)                  # time index, slot index
        p0 = pos[oi, oj]                              # [S,2]
        v0 = vel[oi, oj]                              # [S,2]
        target = pos[oi + steps, oj]                  # [S,2]
        finite = (np.all(np.isfinite(p0), axis=1)
                  & np.all(np.isfinite(v0), axis=1)
                  & np.all(np.isfinite(target), axis=1))
        if not np.any(finite):
            continue
        p0, v0, target = p0[finite], v0[finite], target[finite]
        pred = p0 + v0 * float(h)
        err = np.linalg.norm(pred - target, axis=1)
        out.errors[h] = err
        out.platform[h] = gt_type[oj[finite]]
        out.intent[h] = gt_intent[oi[finite], oj[finite]]
    return out


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------


def _stats(err: np.ndarray) -> dict[str, Any]:
    """Summary statistics [m] of one error array (empty -> NaNs)."""
    if err.size == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "median": nan, "p90": nan,
                "p95": nan, "max": nan, "std": nan}
    e = np.asarray(err, dtype=np.float64)
    return {
        "n": int(e.size),
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90.0)),
        "p95": float(np.percentile(e, 95.0)),
        "max": float(np.max(e)),
        "std": float(np.std(e)),
    }


def expected_difficulty(axis: str, label: int) -> str:
    """Expected difficulty band of one class.

    Args:
        axis: ``'platform'`` or ``'intent'``.
        label: the enum value.

    Returns:
        ``EASY`` / ``MEDIUM`` / ``HARD``; ``MEDIUM`` for an unknown label, so a
        future enum member is graded against the neutral band rather than
        silently escaping the gate.
    """
    return EXPECTED_DIFFICULTY.get(axis, {}).get(int(label), MEDIUM)


def band_of_platform(platform: int) -> str:
    """Expected difficulty of a :class:`~core.types.PlatformType`."""
    return expected_difficulty("platform", int(platform))


def band_of_intent(intent: int) -> str:
    """Expected difficulty of an :class:`~core.types.IntentType`."""
    return expected_difficulty("intent", int(intent))


def class_band_limits(axis: str, label: int) -> tuple[float, float]:
    """Error band [m] one class is judged against, exemptions applied.

    Args:
        axis: ``'platform'`` or ``'intent'``.
        label: the enum value.

    Returns:
        ``(lo, hi)`` in metres. ``lo`` is 0.0 for the constant-velocity-by-
        construction classes of :data:`CV_BY_CONSTRUCTION`.
    """
    lo, hi = BAND_LIMITS[expected_difficulty(axis, label)]
    if int(label) in CV_BY_CONSTRUCTION.get(axis, frozenset()):
        lo = 0.0
    return float(lo), float(hi)


def _verdict_in_band(median_m: float, n: int, lo: float, hi: float) -> str:
    """Verdict for a median error [m] against explicit band limits [m]."""
    if n < MIN_SAMPLES_FOR_VERDICT or not np.isfinite(median_m):
        return VERDICT_INSUFFICIENT
    if median_m < lo:
        return VERDICT_TOO_EASY
    if median_m > hi:
        return VERDICT_TOO_RANDOM
    return VERDICT_OK


def _verdict(median_m: float, n: int, band: str = MEDIUM) -> str:
    """Map a median 15 s error [m] and a sample count to a verdict string.

    The band a class is judged in comes from :data:`EXPECTED_DIFFICULTY`, not
    from one global rule: a 30 m median is a PASS for an eVTOL and a FAILURE
    for a bird. ``MEDIUM`` reproduces the original global band exactly, which
    is why it is what the aggregate verdict uses -- an aggregate mixes classes
    of every difficulty, so the neutral band is the only defensible one for it.

    Args:
        median_m: median error [m] at the diagnostic horizon.
        n: sample count [-].
        band: ``EASY`` / ``MEDIUM`` / ``HARD``.

    Returns:
        One of the ``VERDICT_*`` strings.
    """
    lo, hi = BAND_LIMITS.get(band, BAND_LIMITS[MEDIUM])
    return _verdict_in_band(median_m, n, lo, hi)


def _group_report(
    err: np.ndarray,
    labels: np.ndarray,
    names: Mapping[int, str],
    axis: str,
) -> dict[str, Any]:
    """Per-class statistics and banded verdict at the diagnostic horizon.

    Args:
        err: ``[S]`` errors [m] at the diagnostic horizon.
        labels: ``[S]`` integer class label parallel to ``err``.
        names: label -> display name.
        axis: ``'platform'`` or ``'intent'``; selects the expectation table.

    Returns:
        ``{str(label): {'name', 'expected', 'band_lo_m', 'band_hi_m',
        'verdict', **stats}}``.
    """
    out: dict[str, Any] = {}
    if err.size == 0:
        return out
    lab = np.asarray(labels, dtype=np.int64)
    for value in sorted(int(v) for v in np.unique(lab)):
        if value < 0:
            continue
        sel = err[lab == value]
        st = _stats(sel)
        band = expected_difficulty(axis, value)
        lo, hi = class_band_limits(axis, value)
        out[str(value)] = {
            "name": names.get(value, f"#{value}"),
            "expected": band,
            "band_lo_m": float(lo),
            "band_hi_m": float(hi),
            "verdict": _verdict_in_band(st["median"], st["n"], lo, hi),
            **st,
        }
    return out


def _band_pool(
    err: np.ndarray,
    labels: np.ndarray,
    axis: str,
    band: str,
) -> dict[str, Any]:
    """Statistics [m] of every sample whose class sits in ``band``.

    Pooling the raw samples rather than averaging per-class medians is
    deliberate: the separation the calibration figure will show is between the
    two POPULATIONS the model sees, weighted as they actually occur in the
    dataset.

    Args:
        err: ``[S]`` errors [m].
        labels: ``[S]`` class labels parallel to ``err``.
        axis: ``'platform'`` or ``'intent'``.
        band: the difficulty band to pool.

    Returns:
        ``{'classes': [names], **stats}``.
    """
    if err.size == 0:
        return {"classes": [], **_stats(np.zeros((0,)))}
    lab = np.asarray(labels, dtype=np.int64)
    table = EXPECTED_DIFFICULTY.get(axis, {})
    names = TYPE_NAMES if axis == "platform" else INTENT_NAMES
    wanted = [v for v in sorted(set(int(x) for x in np.unique(lab)))
              if v >= 0 and table.get(v, MEDIUM) == band]
    if not wanted:
        return {"classes": [], **_stats(np.zeros((0,)))}
    sel = np.isin(lab, wanted)
    return {
        "classes": [names.get(v, f"#{v}") for v in wanted],
        **_stats(err[sel]),
    }


def _separation(
    err: np.ndarray,
    labels: np.ndarray,
    axis: str,
) -> dict[str, Any]:
    """HARD-vs-EASY median-error ratio [-] along one class axis.

    Args:
        err: ``[S]`` errors [m] at the diagnostic horizon.
        labels: ``[S]`` class labels parallel to ``err``.
        axis: ``'platform'`` or ``'intent'``.

    Returns:
        ``{'axis', 'hard', 'easy', 'ratio', 'required', 'passed', 'reason'}``.
        ``ratio`` is NaN and ``passed`` False when either band is missing or
        too thin to judge.
    """
    hard = _band_pool(err, labels, axis, HARD)
    easy = _band_pool(err, labels, axis, EASY)
    enough = (hard["n"] >= MIN_SAMPLES_FOR_VERDICT
              and easy["n"] >= MIN_SAMPLES_FOR_VERDICT)
    ratio = float("nan")
    reason = ""
    if not enough:
        reason = (f"insufficient samples to compare bands on {axis} "
                  f"(HARD n={hard['n']}, EASY n={easy['n']})")
    elif not (easy["median"] > 0.0):
        reason = f"EASY {axis} median is zero; the classes are degenerate"
    else:
        ratio = float(hard["median"] / easy["median"])
        if ratio < MIN_BAND_SEPARATION:
            reason = (f"{axis} band separation {ratio:.2f}x is below the "
                      f"required {MIN_BAND_SEPARATION:.1f}x")
    return {
        "axis": axis,
        "hard": hard,
        "easy": easy,
        "ratio": ratio,
        "required": float(MIN_BAND_SEPARATION),
        "passed": bool(np.isfinite(ratio) and ratio >= MIN_BAND_SEPARATION),
        "reason": reason,
    }


def gate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """The dataset-level gate: three questions, not one verdict.

    A single global verdict cannot express what this dataset is for. The
    classes are deliberately of different difficulty -- that spread is what
    lets the paper show a model modulating its uncertainty -- so the gate asks:

        (a) do the HARD classes genuinely defeat CV?
        (b) are the EASY classes genuinely easy (and not degenerate)?
        (c) is the separation between the bands large enough to see?

    Args:
        report: a :func:`diagnose` result.

    Returns:
        ``{'passed', 'hard_ok', 'easy_ok', 'separation_ok', 'reasons',
        'separation': {'platform': {...}, 'intent': {...}}}``. ``passed`` is
        False whenever a condition could not be evaluated, so an empty or thin
        dataset never gates green by accident.
    """
    reasons: list[str] = []
    checked = {HARD: 0, EASY: 0}
    failed = {HARD: 0, EASY: 0}
    for axis, key in (("platform", "by_platform"), ("intent", "by_intent")):
        for rec in report.get(key, {}).values():
            band = rec.get("expected", MEDIUM)
            if band not in (HARD, EASY) or rec["verdict"] == VERDICT_INSUFFICIENT:
                continue
            checked[band] += 1
            if rec["verdict"] != VERDICT_OK:
                failed[band] += 1
                reasons.append(
                    f"{axis} '{rec['name']}' is expected {band} but is "
                    f"{rec['verdict']} (median {rec['median']:.1f} m, band "
                    f"{rec['band_lo_m']:.1f}-{rec['band_hi_m']:.1f} m)"
                )
    if checked[HARD] == 0:
        reasons.append("no HARD class had enough samples to judge")
    if checked[EASY] == 0:
        reasons.append("no EASY class had enough samples to judge")

    separation = report.get("band_separation", {})
    sep_ok = True
    for axis in ("platform", "intent"):
        rec = separation.get(axis)
        if rec is None:
            sep_ok = False
            reasons.append(f"no band separation was computed for {axis}")
            continue
        if not rec["passed"]:
            sep_ok = False
            if rec["reason"]:
                reasons.append(rec["reason"])

    hard_ok = checked[HARD] > 0 and failed[HARD] == 0
    easy_ok = checked[EASY] > 0 and failed[EASY] == 0

    # Is the gate even ANSWERABLE on this data? Doc 8.1 cuts the splits on
    # parameter axes and deliberately keeps the HARD classes (BIRD, UNKNOWN,
    # SEARCH_WANDER, REACTIVE_AVOID) out of training. A 'train' run therefore
    # contains ZERO hard samples -- question (a) has no data to answer it, and
    # reporting FAIL there would be a category error: it would blame the data
    # for a property of the split. That is distinct from a dataset that
    # contains both bands and fails on merit, and from an empty dataset, which
    # must still fail. So: exactly one band absent while the other is well
    # populated => single-band split => NOT APPLICABLE. Everything else,
    # including empty, is judged normally.
    hard_n = max([int(separation[a]["hard"]["n"]) for a in separation] or [0])
    easy_n = max([int(separation[a]["easy"]["n"]) for a in separation] or [0])
    single_band = (hard_n == 0) != (easy_n == 0)
    populated = max(hard_n, easy_n) >= MIN_SAMPLES_FOR_VERDICT
    applicable = not (single_band and populated)
    passed = bool(hard_ok and easy_ok and sep_ok)
    if applicable:
        status = "PASS" if passed else "FAIL"
    else:
        status = "N/A"
        reasons.insert(0, (
            "this data contains only the "
            f"{'EASY' if hard_n == 0 else 'HARD'} band, so the dataset gate "
            "cannot be judged on it -- doc 8.1 keeps the hard classes out of "
            "training by design. Diagnose over all splits (--preset all) to "
            "evaluate the gate."
        ))
    return {
        "passed": passed,
        "status": status,
        "applicable": bool(applicable),
        "hard_ok": bool(hard_ok),
        "easy_ok": bool(easy_ok),
        "separation_ok": bool(sep_ok),
        "n_hard_classes": int(checked[HARD]),
        "n_easy_classes": int(checked[EASY]),
        "n_hard_samples": hard_n,
        "n_easy_samples": easy_n,
        "reasons": reasons,
        "separation": separation,
    }


def diagnose(
    errors: CVErrors,
    diagnostic_horizon_s: float = DIAGNOSTIC_HORIZON_S,
) -> dict[str, Any]:
    """Turn CV errors into the doc 8.3 data-difficulty verdict.

    The verdict is taken from the **median** error at
    ``diagnostic_horizon_s``, because the median is robust to the handful of
    agents that happen to execute a hard turn right at the origin, whereas the
    mean is not. The p90 is reported alongside so that a dataset with a benign
    median but a pathological tail is visible.

    The thresholds are a convention of this study, derived from the NMAC radius
    as doc 8.3 suggests; they are echoed in the returned dict under
    ``'thresholds'`` so that any figure produced from this output can state
    them. The paper judges difficulty on relative feel, not on hitting an
    absolute number -- treat a verdict as a prompt to look at the trajectories,
    not as a pass/fail gate.

    Args:
        errors: output of :func:`cv_errors` (possibly merged over a run).
        diagnostic_horizon_s: horizon [s] the verdict is taken at.

    Returns:
        A JSON-safe dict::

            {
              'verdict': str,                # overall
              'diagnostic_horizon_s': float,
              'thresholds': {...},           # the constants used
              'n_episodes': int,
              'per_horizon': {'5.0': {n, mean, median, p90, p95, max, std,
                                      'median_over_nmac': float,
                                      'verdict': str}, ...},
              'by_platform': {'1': {name, expected, band_lo_m, band_hi_m,
                                    verdict, ...}, ...},
              'by_intent':   {'0': {name, expected, ...}, ...},
              'band_separation': {'platform': {...}, 'intent': {...}},
              'gate': {...},                 # see gate_report
              'warnings': [str, ...],        # subgroups that are not OK
            }

        ``verdict`` is the AGGREGATE band judgement and keeps its original
        global-band meaning. The per-class entries are judged against their own
        expected band, and ``gate`` is the dataset-level pass/fail.
    """
    per_horizon: dict[str, Any] = {}
    for h in errors.horizons_s:
        st = _stats(errors.errors.get(h, np.zeros((0,))))
        st["median_over_nmac"] = (float(st["median"] / NMAC_RADIUS_M)
                                  if np.isfinite(st["median"]) else float("nan"))
        st["verdict"] = _verdict(st["median"], st["n"])
        per_horizon[f"{float(h):g}"] = st

    hkey = float(diagnostic_horizon_s)
    err_h = errors.errors.get(hkey, np.zeros((0,), dtype=np.float64))
    plat_h = errors.platform.get(hkey, np.zeros((0,), dtype=np.int8))
    int_h = errors.intent.get(hkey, np.zeros((0,), dtype=np.int8))
    overall = _stats(err_h)
    verdict = _verdict(overall["median"], overall["n"])

    by_platform = _group_report(err_h, plat_h, TYPE_NAMES, "platform")
    by_intent = _group_report(err_h, int_h, INTENT_NAMES, "intent")
    band_separation = {
        "platform": _separation(err_h, plat_h, "platform"),
        "intent": _separation(err_h, int_h, "intent"),
    }

    warnings: list[str] = []
    for group_label, table in (("platform", by_platform), ("intent", by_intent)):
        for key, rec in table.items():
            if rec["verdict"] not in (VERDICT_OK, VERDICT_INSUFFICIENT):
                warnings.append(
                    f"{group_label} '{rec['name']}': {rec['verdict']} "
                    f"(median {rec['median']:.1f} m over {rec['n']} samples, "
                    f"expected {rec['expected']} = "
                    f"{rec['band_lo_m']:.1f}-{rec['band_hi_m']:.1f} m)"
                )
    if verdict == VERDICT_OK and warnings:
        warnings.insert(0, "aggregate verdict is OK but at least one class is not "
                           "-- an easy class can hide inside a mixed dataset")

    result: dict[str, Any] = {
        "verdict": verdict,
        "diagnostic_horizon_s": float(diagnostic_horizon_s),
        "velocity_source_note": "errors were produced by analysis.cv_baseline.cv_errors",
        "thresholds": {
            "nmac_radius_m": float(NMAC_RADIUS_M),
            "too_easy_median_m": float(TOO_EASY_MEDIAN_M),
            "too_random_median_m": float(TOO_RANDOM_MEDIAN_M),
            "too_easy_fraction_of_nmac": float(TOO_EASY_FRACTION_OF_NMAC),
            "too_random_multiple_of_nmac": float(TOO_RANDOM_MULTIPLE_OF_NMAC),
            "min_samples_for_verdict": int(MIN_SAMPLES_FOR_VERDICT),
            "band_limits_m": {k: [float(v[0]), float(v[1])]
                              for k, v in BAND_LIMITS.items()},
            "min_band_separation": float(MIN_BAND_SEPARATION),
            "note": "convention of this study, derived from NMAC radius per "
                    "doc 8.3; the judgement is relative, not absolute, and "
                    "each class is judged against its own expected band",
        },
        "n_episodes": int(errors.n_episodes),
        "overall": overall,
        "per_horizon": per_horizon,
        "by_platform": by_platform,
        "by_intent": by_intent,
        "band_separation": band_separation,
        "warnings": warnings,
    }
    result["gate"] = gate_report(result)
    return result


def run_diagnostic(
    run_dir: str,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    velocity_source: str = "gt",
    max_episodes: int | None = None,
    diagnostic_horizon_s: float = DIAGNOSTIC_HORIZON_S,
) -> dict[str, Any]:
    """Run the CV diagnostic over a whole dataset run.

    This is doc 10 step 2: generate a run, then immediately call this. If the
    verdict is not ``OK``, adjust the doc 3.2 personality parameters (turn
    frequency, mode dwell time) and regenerate before building anything
    downstream.

    Args:
        run_dir: a ``<LOGS_ROOT>/<run_id>`` directory produced by
            :class:`storage.writer.DatasetWriter`.
        horizons_s: horizons [s] to report.
        velocity_source: ``'gt'`` or ``'diff'``.
        max_episodes: stop after this many episodes (None = all).
        diagnostic_horizon_s: horizon [s] the verdict is taken at.

    Returns:
        The :func:`diagnose` dict with ``'run_dir'`` and ``'n_episodes'``
        filled in.

    Raises:
        FileNotFoundError: if ``run_dir`` does not exist.
    """
    from storage.writer import iter_run  # local import: keeps analysis import-cheap

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    merged = CVErrors.empty(horizons_s)
    n = 0
    for _meta, ep in iter_run(run_dir, validate=False, with_meta=False):
        merged.extend(cv_errors(ep, horizons_s, velocity_source=velocity_source))
        n += 1
        if max_episodes is not None and n >= int(max_episodes):
            break
    report = diagnose(merged, diagnostic_horizon_s=diagnostic_horizon_s)
    report["run_dir"] = str(run_dir).replace("\\", "/")
    report["velocity_source"] = str(velocity_source)
    return report


def format_report(report: Mapping[str, Any], width: int = 78) -> str:
    """Render a :func:`diagnose` / :func:`run_diagnostic` dict as plain text.

    Args:
        report: the diagnosis dict.
        width: rule width [characters].

    Returns:
        A multi-line string ready for ``print``.
    """
    lines: list[str] = []
    a = lines.append
    a("=" * width)
    a("CV EXTRAPOLATION BASELINE / DATA DIFFICULTY DIAGNOSTIC  (doc 8.3)")
    a("=" * width)
    if "run_dir" in report:
        a(f"run        : {report['run_dir']}")
    a(f"episodes   : {report.get('n_episodes', 0)}")
    a(f"NMAC radius: {report['thresholds']['nmac_radius_m']:.1f} m   "
      f"OK band for the {report['diagnostic_horizon_s']:g} s median: "
      f"{report['thresholds']['too_easy_median_m']:.1f} .. "
      f"{report['thresholds']['too_random_median_m']:.1f} m")
    a("")
    a(f"{'horizon':>8}  {'n':>9}  {'median[m]':>10}  {'p90[m]':>9}  "
      f"{'mean[m]':>9}  {'med/NMAC':>9}")
    a("-" * width)
    for key, st in report.get("per_horizon", {}).items():
        a(f"{key + ' s':>8}  {st['n']:>9}  {st['median']:>10.1f}  "
          f"{st['p90']:>9.1f}  {st['mean']:>9.1f}  {st['median_over_nmac']:>9.2f}")
    a("")
    for title, key in (("BY PLATFORM TYPE", "by_platform"),
                       ("BY INTENT TYPE", "by_intent")):
        table = report.get(key, {})
        if not table:
            continue
        a(title)
        a("-" * width)
        a(f"  {'class':<18} {'expect':>7} {'band[m]':>13} {'n':>8}  "
          f"{'median[m]':>10}  {'p90[m]':>9}  verdict")
        for _k, rec in table.items():
            band = (f"{rec.get('band_lo_m', float('nan')):.0f}-"
                    f"{rec.get('band_hi_m', float('nan')):.0f}")
            a(f"  {rec['name']:<18} {rec.get('expected', '?'):>7} {band:>13} "
              f"{rec['n']:>8}  {rec['median']:>10.1f}  {rec['p90']:>9.1f}  "
              f"{rec['verdict']}")
        a("")

    sep = report.get("band_separation", {})
    if sep:
        a("BAND SEPARATION  (HARD median / EASY median at "
          f"{report.get('diagnostic_horizon_s', DIAGNOSTIC_HORIZON_S):g} s; "
          f"the spread the calibration figure exploits)")
        a("-" * width)
        for axis in ("platform", "intent"):
            rec = sep.get(axis)
            if rec is None:
                continue
            hard, easy = rec["hard"], rec["easy"]
            ratio = rec["ratio"]
            ratio_txt = f"{ratio:.2f}x" if np.isfinite(ratio) else "n/a"
            mark = "OK" if rec["passed"] else "FAIL"
            a(f"  by {axis:<9} HARD {hard['median']:>8.1f} m (n={hard['n']:<7}) "
              f"/ EASY {easy['median']:>7.1f} m (n={easy['n']:<7}) "
              f"= {ratio_txt:>7}  need >= {rec['required']:.1f}x  [{mark}]")
            if hard["classes"]:
                a(f"      HARD: {', '.join(hard['classes'])}")
            if easy["classes"]:
                a(f"      EASY: {', '.join(easy['classes'])}")
        a("")

    for w in report.get("warnings", []):
        a(f"  ! {w}")
    if report.get("warnings"):
        a("")
    a("=" * width)
    a(f"AGGREGATE VERDICT: {report.get('verdict', '?')}   "
      f"(global band, mixed classes)")
    gate = report.get("gate")
    if gate:
        status = gate.get("status", "PASS" if gate["passed"] else "FAIL")
        a(f"DATASET GATE     : {status}"
          f"   hard_defeat_cv={'y' if gate['hard_ok'] else 'n'}"
          f"  easy_are_easy={'y' if gate['easy_ok'] else 'n'}"
          f"  separation={'y' if gate['separation_ok'] else 'n'}")
        for reason in gate.get("reasons", []):
            a(f"  - {reason}")
    a("=" * width)
    return "\n".join(lines)
