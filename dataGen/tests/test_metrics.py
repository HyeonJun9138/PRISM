"""Tests for analysis.metrics and analysis.cv_baseline.

The prediction model does not exist yet; these tests freeze the metric
definitions (doc 9) so that the numbers reported in the paper cannot silently
change later. Where a closed form exists, the test hand-computes it; where it
does not, the test checks the estimator against a large sample from a known
distribution.

Run directly::

    python $PRISM_ROOT/dataGen/tests/test_metrics.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis import cv_baseline as cvb                              # noqa: E402
from analysis import metrics as M                                     # noqa: E402
from core.constants import DWC_HMD_M, LOG_DT, NMAC_RADIUS_M           # noqa: E402
from core.types import IntentType, ManeuverMode, PlatformType, RiskLevel  # noqa: E402
from storage.schema import EpisodeArrays                              # noqa: E402

#: DWC tau_mod threshold [s], mirrored here so the DWC test reads plainly.
DWC_TAU_ABOVE = 35.0

_FAILURES: list[str] = []
_CHECKS = 0


def check(condition: bool, message: str) -> None:
    """Record one assertion result."""
    global _CHECKS
    _CHECKS += 1
    if not condition:
        _FAILURES.append(message)


def close(a: float, b: float, tol: float, message: str) -> None:
    """Record one approximate-equality assertion."""
    ok = math.isfinite(float(a)) and abs(float(a) - float(b)) <= tol
    check(ok, f"{message}: got {a!r}, expected {b!r} +- {tol}")


# --------------------------------------------------------------------------
# 9.1 ADE / FDE, hand-computed
# --------------------------------------------------------------------------


def test_ade_fde_hand_computed() -> None:
    """ADE and FDE against errors that can be read off by eye."""
    # one sample, three steps, errors 5, 0, 10  ->  ADE = 5, FDE = 10
    pred = np.zeros((1, 3, 2))
    gt = np.array([[[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]]])
    err = M.displacement_error(pred, gt)
    check(np.allclose(err, [[5.0, 0.0, 10.0]]), f"displacement_error wrong: {err}")
    close(M.ade(pred, gt), 5.0, 1e-12, "ADE of (5,0,10)")
    close(M.fde(pred, gt), 10.0, 1e-12, "FDE of (5,0,10)")

    # two samples: sample ADEs 5 and 2  ->  dataset ADE = 3.5
    pred2 = np.zeros((2, 2, 2))
    gt2 = np.array([
        [[3.0, 4.0], [3.0, 4.0]],       # errors 5, 5 -> ADE 5, FDE 5
        [[0.0, 1.0], [3.0, 0.0]],       # errors 1, 3 -> ADE 2, FDE 3
    ])
    close(M.ade(pred2, gt2), 3.5, 1e-12, "two-sample ADE")
    close(M.fde(pred2, gt2), 4.0, 1e-12, "two-sample FDE")
    check(np.allclose(M.ade_per_sample(pred2, gt2), [5.0, 2.0]),
          "ade_per_sample wrong")

    # masking: the last step of sample 1 is dropped -> ADE 5, FDE 5 for it
    mask = np.array([[True, True], [True, False]])
    close(M.ade(pred2, gt2, mask), (5.0 + 1.0) / 2.0, 1e-12, "masked ADE")
    close(M.fde(pred2, gt2, mask), (5.0 + 1.0) / 2.0, 1e-12,
          "masked FDE must use the last VALID step")


def test_min_ade_fde_k() -> None:
    """minADE_k / minFDE_k must pick the best hypothesis per sample."""
    gt = np.array([[[10.0, 0.0], [20.0, 0.0]]])              # [1,2,2]
    pred_k = np.array([[
        [[10.0, 0.0], [20.0, 0.0]],                          # perfect mode
        [[0.0, 0.0], [0.0, 0.0]],                            # errors 10, 20
    ]])                                                      # [1,2,2,2]
    close(M.min_ade_k(pred_k, gt), 0.0, 1e-12, "minADE_2 with a perfect mode")
    close(M.min_fde_k(pred_k, gt), 0.0, 1e-12, "minFDE_2 with a perfect mode")
    # unimodal ADE of the bad mode alone is 15, of the good mode 0
    close(M.ade(pred_k[:, 1], gt), 15.0, 1e-12, "ADE of the bad mode")
    # a genuinely split prediction: best ADE is min(15, 3) = 3
    pred_k2 = np.array([[
        [[0.0, 0.0], [0.0, 0.0]],                            # ADE 15
        [[8.0, 0.0], [16.0, 0.0]],                           # errors 2, 4 -> 3
    ]])
    close(M.min_ade_k(pred_k2, gt), 3.0, 1e-12, "minADE_2 picks the better mode")
    close(M.min_fde_k(pred_k2, gt), 4.0, 1e-12, "minFDE_2 picks the better mode")


# --------------------------------------------------------------------------
# 9.2 NLL / CRPS
# --------------------------------------------------------------------------


def test_gaussian_nll_closed_form() -> None:
    """NLL at the mean of a unit Gaussian is exactly ln(2*pi)."""
    mu = np.zeros((1, 2))
    cov = np.eye(2)[None, :, :]
    y = np.zeros((1, 2))
    close(M.gaussian_nll(mu, cov, y), math.log(2.0 * math.pi), 1e-12,
          "NLL at the mean of N(0, I)")
    # one sigma away on x: NLL increases by exactly 0.5
    y1 = np.array([[1.0, 0.0]])
    close(M.gaussian_nll(mu, cov, y1), math.log(2.0 * math.pi) + 0.5, 1e-12,
          "NLL one sigma from the mean")
    # scaling the covariance by 4 adds ln(4) and quarters the quadratic term
    cov4 = 4.0 * np.eye(2)[None, :, :]
    close(M.gaussian_nll(mu, cov4, y1),
          0.5 * (0.25 + math.log(16.0) + 2.0 * math.log(2.0 * math.pi)), 1e-12,
          "NLL with a scaled covariance")
    # the diagonal form must agree with the full form
    close(M.gaussian_nll_diag(mu, np.array([[2.0, 2.0]]), y1),
          float(M.gaussian_nll(mu, cov4, y1)), 1e-12,
          "gaussian_nll_diag disagrees with gaussian_nll")
    # a singular covariance must be rejected, not silently regularised
    raised = False
    try:
        M.gaussian_nll(mu, np.zeros((1, 2, 2)), y)
    except ValueError:
        raised = True
    check(raised, "a singular covariance was accepted")


def test_crps_closed_form_and_ensemble() -> None:
    """CRPS closed form, and the ensemble estimator converging to it."""
    # CRPS of N(0,1) evaluated at its mean = (sqrt(2)-1)/sqrt(pi)
    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    close(M.crps_gaussian(np.array([0.0]), np.array([1.0]), np.array([0.0])),
          expected, 1e-12, "CRPS of N(0,1) at the mean")
    # CRPS scales linearly with sigma
    close(M.crps_gaussian(np.array([0.0]), np.array([3.0]), np.array([0.0])),
          3.0 * expected, 1e-12, "CRPS must scale with sigma")
    # a perfectly sharp forecast that is right has CRPS -> 0
    tiny = float(M.crps_gaussian(np.array([0.0]), np.array([1e-6]),
                                 np.array([0.0])))
    check(tiny < 1e-6, f"a sharp, correct forecast should score ~0, got {tiny}")
    # an over-confident forecast that is wrong must score worse than a wide one
    sharp_wrong = float(M.crps_gaussian(np.array([0.0]), np.array([0.1]),
                                        np.array([5.0])))
    wide_wrong = float(M.crps_gaussian(np.array([0.0]), np.array([3.0]),
                                       np.array([5.0])))
    check(sharp_wrong > wide_wrong,
          "CRPS must punish confident-and-wrong more than uncertain-and-wrong")

    # ensemble estimator vs closed form
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, size=(1, 40000))
    ens = float(M.crps_ensemble(samples, np.array([0.0])))
    close(ens, expected, 0.01, "crps_ensemble should converge to the closed form")

    # hand-computable 2-member ensemble: x = {0, 2}, y = 1
    #   term1 = (1 + 1)/2 = 1 ; pair_sum = |0-0|+|0-2|+|2-0|+|2-2| = 4
    #   CRPS  = 1 - 4/(2*4) = 0.5
    close(M.crps_ensemble(np.array([[0.0, 2.0]]), np.array([1.0])), 0.5, 1e-12,
          "two-member ensemble CRPS")


def test_energy_score() -> None:
    """Energy score reduces to |x - y| for a one-member ensemble."""
    x = np.array([[[3.0, 4.0]]])                            # [1,1,2]
    y = np.array([[0.0, 0.0]])
    close(M.energy_score(x, y), 5.0, 1e-12,
          "energy score of a degenerate ensemble is the distance")
    # a spread-out ensemble centred on the truth beats a tight wrong one
    rng = np.random.default_rng(3)
    good = rng.normal(0.0, 50.0, size=(1, 500, 2))
    bad = rng.normal(0.0, 1.0, size=(1, 500, 2)) + 300.0
    check(float(M.energy_score(good, y)) < float(M.energy_score(bad, y)),
          "energy score must prefer the ensemble covering the truth")


# --------------------------------------------------------------------------
# 9.3 Calibration
# --------------------------------------------------------------------------


def test_chi2_quantile() -> None:
    """The 2-dof chi-square quantile is -2 ln(1-p); check known values."""
    close(M.chi2_quantile_2dof(0.0), 0.0, 1e-12, "chi2 quantile at p=0")
    close(M.chi2_quantile_2dof(0.5), 2.0 * math.log(2.0), 1e-12,
          "chi2 quantile at p=0.5")
    close(M.chi2_quantile_2dof(0.9), 4.605170185988091, 1e-9,
          "chi2 quantile at p=0.9 (textbook 4.605)")
    close(M.chi2_quantile_2dof(0.95), 5.991464547107979, 1e-9,
          "chi2 quantile at p=0.95 (textbook 5.991)")


def test_coverage_of_known_gaussian() -> None:
    """Empirical coverage of samples from the stated Gaussian ~= nominal."""
    rng = np.random.default_rng(12345)
    n = 200000
    # correlated, anisotropic covariance so the Mahalanobis path is exercised
    cov = np.array([[400.0, 120.0], [120.0, 100.0]])
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(n, 2))
    y = z @ L.T
    mu = np.zeros((n, 2))
    covs = np.broadcast_to(cov, (n, 2, 2))
    for level in (0.5, 0.9, 0.95):
        emp = M.empirical_coverage(mu, covs, y, level=level)
        close(emp, level, 0.01, f"coverage at nominal {level}")

    # an over-confident model (sigma halved -> covariance quartered) must
    # under-cover, which is exactly the failure mode doc 9.3 is about
    over = M.empirical_coverage(mu, covs / 4.0, y, level=0.9)
    check(over < 0.75, f"an over-confident model should under-cover, got {over}")

    curve = M.reliability_curve(mu, covs, y)
    check(curve["n"] == n, "reliability_curve dropped samples")
    check(curve["ece"] < 0.01, f"a calibrated model should have tiny ECE, "
                               f"got {curve['ece']}")
    bad_curve = M.reliability_curve(mu, covs / 4.0, y)
    check(bad_curve["ece"] > curve["ece"],
          "ECE must be larger for the over-confident model")
    check(M.calibration_error(curve) == curve["ece"], "calibration_error mismatch")


def test_pit_uniformity() -> None:
    """PIT values of a correct Gaussian forecast are uniform on [0,1]."""
    rng = np.random.default_rng(99)
    n = 100000
    y = rng.normal(2.0, 3.0, size=n)
    pit = M.pit_values(np.full(n, 2.0), np.full(n, 3.0), y)
    check(pit.shape == (n,), "pit_values returned the wrong shape")
    close(float(np.mean(pit)), 0.5, 0.01, "PIT mean of a calibrated forecast")
    hist, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
    dev = float(np.max(np.abs(hist / n - 0.1)))
    check(dev < 0.01, f"PIT histogram is not flat (max deviation {dev})")
    # over-confident forecast -> U-shaped histogram (tails over-populated)
    pit_bad = M.pit_values(np.full(n, 2.0), np.full(n, 1.0), y)
    hist_bad, _ = np.histogram(pit_bad, bins=10, range=(0.0, 1.0))
    check(hist_bad[0] + hist_bad[-1] > hist_bad[4] + hist_bad[5],
          "an over-confident forecast should produce a U-shaped PIT histogram")


# --------------------------------------------------------------------------
# 9.4 Separation
# --------------------------------------------------------------------------


def _straight_episode(T: int = 120, N: int = 2,
                      intruder_offset: float = 800.0) -> EpisodeArrays:
    """Ownship flying East; slot 0 crossing it, slot 1 far away and parallel."""
    ep = EpisodeArrays.allocate(T, N)
    t = np.arange(T) * LOG_DT
    ep.t[:] = t.astype(np.float32)
    ep.own_pos[:, 0] = (50.0 * t).astype(np.float32)
    ep.own_vel[:, 0] = np.float32(50.0)
    ep.own_psi[:] = np.float32(0.0)

    # slot 0: crosses the ownship track from the north at mid-episode
    x0 = 50.0 * t[T // 2]
    ep.gt_valid[:, 0] = True
    ep.gt_pos[:, 0, 0] = np.float32(x0)
    ep.gt_pos[:, 0, 1] = (intruder_offset - 40.0 * t).astype(np.float32)
    ep.gt_vel[:, 0, 0] = np.float32(0.0)
    ep.gt_vel[:, 0, 1] = np.float32(-40.0)
    ep.gt_psi[:, 0] = np.float32(-0.5 * math.pi)
    ep.gt_mode[:, 0] = np.int8(ManeuverMode.CV)
    ep.gt_intent[:, 0] = np.int8(IntentType.CORRIDOR_TRANSIT)
    ep.gt_type[0] = np.int8(PlatformType.FIXED_WING_UAS)
    ep.agent_ids[0] = np.int32(1)

    # slot 1: 20 km to the north, parallel -- never a factor
    ep.gt_valid[:, 1] = True
    ep.gt_pos[:, 1, 0] = (50.0 * t).astype(np.float32)
    ep.gt_pos[:, 1, 1] = np.float32(20000.0)
    ep.gt_vel[:, 1, 0] = np.float32(50.0)
    ep.gt_vel[:, 1, 1] = np.float32(0.0)
    ep.gt_psi[:, 1] = np.float32(0.0)
    ep.gt_mode[:, 1] = np.int8(ManeuverMode.CV)
    ep.gt_intent[:, 1] = np.int8(IntentType.CORRIDOR_TRANSIT)
    ep.gt_type[1] = np.int8(PlatformType.EVTOL_UAM)
    ep.agent_ids[1] = np.int32(2)
    return ep


def test_separation_and_nmac() -> None:
    """Minimum separation, NMAC detection and the risk-level ladder."""
    own = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    gt = np.array([
        [[0.0, 500.0], [np.nan, np.nan]],
        [[100.0, 100.0], [np.nan, np.nan]],
        [[200.0, 300.0], [0.0, 0.0]],
    ])
    valid = np.array([[True, False], [True, False], [True, True]])
    d = M.separation_distances(own, gt, valid)
    check(np.isnan(d[0, 1]) and np.isnan(d[1, 1]),
          "invalid slots must not produce a distance")
    close(d[1, 0], 100.0, 1e-9, "distance at step 1")

    res = M.min_separation(own, gt, valid)
    close(res["min_separation_m"], 100.0, 1e-9, "minimum separation")
    check(res["argmin_step"] == 1 and res["argmin_slot"] == 0,
          f"CPA index wrong: {res['argmin_step']}, {res['argmin_slot']}")
    close(res["per_agent_m"][1], 200.0, 1e-9, "per-agent minimum for slot 1")

    nm = M.nmac_events(own, gt, valid)
    check(nm["occurred"], "100 m separation is inside the NMAC radius")
    check(nm["n_agents"] == 1, f"one agent should breach, got {nm['n_agents']}")
    check(nm["first_step"] == 1, f"first NMAC step wrong: {nm['first_step']}")
    check(100.0 < NMAC_RADIUS_M, "sanity: the fixture must be inside NMAC")

    # nothing close at all
    far = np.array([[[0.0, 5000.0]], [[0.0, 5000.0]], [[0.0, 5000.0]]])
    far_valid = np.ones((3, 1), dtype=bool)
    nm2 = M.nmac_events(own, far, far_valid)
    check(not nm2["occurred"], "a 5 km separation must not be an NMAC")
    check(nm2["first_step"] == -1, "first_step should be -1 when nothing happens")


def test_tau_mod_and_dwc() -> None:
    """tau_mod and the DWC gate, on geometries computed by hand."""
    # already inside DMOD -> tau_mod is 0 by definition
    tm = M.tau_mod(np.array([[0.0, 500.0]]), np.array([[0.0, 0.0]]))
    close(tm[0], 0.0, 1e-12, "tau_mod inside DMOD")
    # opening geometry -> infinite
    tm = M.tau_mod(np.array([[5000.0, 0.0]]), np.array([[100.0, 0.0]]))
    check(math.isinf(float(tm[0])), "tau_mod must be inf when not closing")
    # closing: r=5000, rdot=-100 -> (DMOD^2 - r^2)/(r*rdot)
    r, rdot = 5000.0, -100.0
    expect = (DWC_HMD_M ** 2 - r ** 2) / (r * rdot)
    tm = M.tau_mod(np.array([[5000.0, 0.0]]), np.array([[-100.0, 0.0]]))
    close(tm[0], expect, 1e-6, "tau_mod closing geometry")
    check(expect > DWC_TAU_ABOVE, "fixture should sit above the tau threshold")

    # HMD projection: crossing geometry with a 300 m miss
    hmd = M.horizontal_miss_distance(np.array([[5000.0, 300.0]]),
                                     np.array([[-100.0, 0.0]]))
    close(hmd[0], 300.0, 1e-9, "projected HMD")
    # CPA already behind -> current range
    hmd2 = M.horizontal_miss_distance(np.array([[100.0, 0.0]]),
                                      np.array([[50.0, 0.0]]))
    close(hmd2[0], 100.0, 1e-9, "HMD when the CPA is behind")

    # far away and slow to close: HMD is small but tau_mod > 35 s -> no violation
    own_pos = np.array([[0.0, 0.0]])
    own_vel = np.array([[100.0, 0.0]])
    gt_pos = np.array([[[5000.0, 300.0]]])
    gt_vel = np.array([[[0.0, 0.0]]])
    valid = np.ones((1, 1), dtype=bool)
    dw = M.dwc_violation(own_pos, own_vel, gt_pos, gt_vel, valid)
    check(not dw["violated"],
          f"tau_mod {dw['min_tau_mod_s']:.1f} s > 35 s must not violate DWC")

    # same geometry, much closer -> tau_mod drops below 35 s -> violation
    gt_pos_near = np.array([[[2000.0, 300.0]]])
    dw2 = M.dwc_violation(own_pos, own_vel, gt_pos_near, gt_vel, valid)
    check(dw2["violated"], "a closing 300 m-miss at 2 km must violate DWC")
    close(dw2["min_hmd_m"], 300.0, 1.0, "DWC minimum HMD")
    check(dw2["first_step"] == 0, "DWC first step wrong")


def test_nmac_implies_dwc_invariant() -> None:
    """schema.EpisodeMeta asserts NMAC => DWC; the metrics must guarantee it."""
    rng = np.random.default_rng(5)
    T = 40
    own_pos = np.cumsum(rng.normal(0.0, 30.0, size=(T, 2)), axis=0)
    own_vel = rng.normal(0.0, 40.0, size=(T, 2))
    gt_pos = own_pos[:, None, :] + rng.normal(0.0, 120.0, size=(T, 1, 2))
    gt_vel = rng.normal(0.0, 40.0, size=(T, 1, 2))
    valid = np.ones((T, 1), dtype=bool)
    nm = M.nmac_events(own_pos, gt_pos, valid)
    dw = M.dwc_violation(own_pos, own_vel, gt_pos, gt_vel, valid)
    check(nm["occurred"], "fixture should contain an NMAC")
    check(dw["violated"], "an NMAC must always imply a DWC violation")
    rl = M.risk_level_from_separation(50.0, nm["occurred"], dw["violated"])
    check(rl == int(RiskLevel.HIGH), "NMAC must map to RiskLevel.HIGH")
    check(M.risk_level_from_separation(900.0, False, True) == int(RiskLevel.MEDIUM),
          "a DWC violation without NMAC must map to MEDIUM")
    check(M.risk_level_from_separation(1500.0, False, False) == int(RiskLevel.LOW),
          "an eroded margin must map to LOW")
    check(M.risk_level_from_separation(50000.0, False, False) == int(RiskLevel.NONE),
          "a clean episode must map to NONE")


def test_separation_metrics_on_episode() -> None:
    """separation_metrics must agree with the component functions."""
    ep = _straight_episode()
    res = M.separation_metrics(ep)
    d = M.separation_distances(ep.own_pos.astype(np.float64),
                               ep.gt_pos.astype(np.float64),
                               ep.gt_valid.astype(bool))
    close(res["min_separation_m"], float(np.nanmin(d)), 1e-6,
          "separation_metrics minimum disagrees with the raw distances")
    check(res["cpa_slot"] == 0, "the crossing agent should own the CPA")
    check(0.0 <= res["cpa_time_s"] <= float(ep.t[-1]),
          "CPA time is outside the episode")
    check(res["risk_level"] in {int(r) for r in RiskLevel},
          "risk_level is not a RiskLevel")
    check(not res["nmac_occurred"] or res["dwc_violated"],
          "separation_metrics broke the NMAC => DWC invariant")


# --------------------------------------------------------------------------
# CV baseline (doc 8.3)
# --------------------------------------------------------------------------


def _cv_episode(turn_rate: float, T: int = 120, N: int = 3,
                speed: float = 25.0) -> EpisodeArrays:
    """An episode whose agents all fly a constant turn at ``turn_rate`` rad/s."""
    ep = EpisodeArrays.allocate(T, N)
    t = np.arange(T) * LOG_DT
    ep.t[:] = t.astype(np.float32)
    ep.own_pos[:, 0] = (50.0 * t).astype(np.float32)
    ep.own_vel[:, 0] = np.float32(50.0)
    ep.own_psi[:] = np.float32(0.0)
    for slot in range(N):
        psi = 0.3 * slot + turn_rate * t
        vx = speed * np.cos(psi)
        vy = speed * np.sin(psi)
        x = 500.0 * slot + np.cumsum(vx) * LOG_DT
        y = 300.0 * slot + np.cumsum(vy) * LOG_DT
        ep.gt_valid[:, slot] = True
        ep.gt_pos[:, slot, 0] = x.astype(np.float32)
        ep.gt_pos[:, slot, 1] = y.astype(np.float32)
        ep.gt_vel[:, slot, 0] = vx.astype(np.float32)
        ep.gt_vel[:, slot, 1] = vy.astype(np.float32)
        ep.gt_psi[:, slot] = np.arctan2(np.sin(psi), np.cos(psi)).astype(np.float32)
        ep.gt_mode[:, slot] = np.int8(
            ManeuverMode.CV if turn_rate == 0.0 else ManeuverMode.CT_LEFT)
        ep.gt_intent[:, slot] = np.int8(IntentType.CORRIDOR_TRANSIT)
        ep.gt_type[slot] = np.int8(PlatformType.FIXED_WING_UAS)
        ep.agent_ids[slot] = np.int32(slot)
    return ep


def test_cv_predict_and_errors() -> None:
    """CV extrapolation is exact on a straight line and grows on a turn."""
    pred = cvb.cv_predict(np.array([0.0, 0.0]), np.array([10.0, 0.0]),
                          (5.0, 10.0, 15.0))
    check(np.allclose(pred, [[50.0, 0.0], [100.0, 0.0], [150.0, 0.0]]),
          f"cv_predict is wrong: {pred}")

    straight = _cv_episode(turn_rate=0.0)
    errs = cvb.cv_errors(straight)
    for h in cvb.DEFAULT_HORIZONS_S:
        e = errs.errors[h]
        check(e.size > 0, f"no CV samples at horizon {h}")
        check(float(np.max(e)) < 1.0,
              f"CV must be near-exact on straight tracks, max error {np.max(e)}")

    turning = _cv_episode(turn_rate=0.08)
    errs_t = cvb.cv_errors(turning)
    med5 = float(np.median(errs_t.errors[5.0]))
    med15 = float(np.median(errs_t.errors[15.0]))
    check(med15 > med5, "CV error must grow with the horizon")
    check(errs_t.errors[15.0].size < errs_t.errors[5.0].size,
          "a longer horizon must admit fewer time origins")
    check(errs_t.platform[15.0].size == errs_t.errors[15.0].size,
          "platform labels must be parallel to the errors")

    # horizons must be whole multiples of LOG_DT
    raised = False
    try:
        cvb.cv_errors(straight, horizons_s=(0.3,))
    except ValueError:
        raised = True
    check(raised, "a non-multiple-of-LOG_DT horizon was accepted")


def test_cv_diagnosis_verdicts() -> None:
    """The three doc 8.3 verdicts must fire on data built to trigger them."""
    straight = _cv_episode(turn_rate=0.0)
    errs = cvb.CVErrors.empty()
    for _ in range(3):
        errs.extend(cvb.cv_errors(straight))
    diag = cvb.diagnose(errs)
    check(diag["verdict"] == cvb.VERDICT_TOO_EASY,
          f"straight-line data must be TOO_EASY, got {diag['verdict']}")
    check(diag["overall"]["n"] >= cvb.MIN_SAMPLES_FOR_VERDICT,
          "the fixture must supply enough samples for a verdict")

    # synthetic error populations placed on either side of the OK band
    def _fake(median_m: float) -> cvb.CVErrors:
        e = cvb.CVErrors.empty()
        n = 5000
        vals = np.full(n, median_m, dtype=np.float64)
        for h in e.horizons_s:
            e.errors[h] = vals
            e.platform[h] = np.full(n, int(PlatformType.MULTIROTOR), dtype=np.int8)
            e.intent[h] = np.full(n, int(IntentType.DELIVERY_WP), dtype=np.int8)
        e.n_episodes = 1
        return e

    check(cvb.diagnose(_fake(2.0 * NMAC_RADIUS_M))["verdict"] == cvb.VERDICT_OK,
          "2x the NMAC radius should be OK")
    check(cvb.diagnose(_fake(10.0 * NMAC_RADIUS_M))["verdict"] == cvb.VERDICT_TOO_RANDOM,
          "10x the NMAC radius should be TOO_RANDOM")
    check(cvb.diagnose(_fake(0.05 * NMAC_RADIUS_M))["verdict"] == cvb.VERDICT_TOO_EASY,
          "0.05x the NMAC radius should be TOO_EASY")

    # too few samples -> no verdict rather than a noisy one
    thin = cvb.CVErrors.empty()
    for h in thin.horizons_s:
        thin.errors[h] = np.full(10, 300.0)
        thin.platform[h] = np.zeros(10, dtype=np.int8)
        thin.intent[h] = np.zeros(10, dtype=np.int8)
    check(cvb.diagnose(thin)["verdict"] == cvb.VERDICT_INSUFFICIENT,
          "a tiny sample must yield INSUFFICIENT_DATA")


def test_cv_class_breakdown_catches_hidden_easy_class() -> None:
    """An 'OK on average' dataset with one trivially straight class must warn."""
    n = 4000
    e = cvb.CVErrors.empty()
    hard = np.full(n, 2.0 * NMAC_RADIUS_M)
    easy = np.full(n, 0.02 * NMAC_RADIUS_M)
    vals = np.concatenate([hard, easy])
    plat = np.concatenate([
        np.full(n, int(PlatformType.MULTIROTOR), dtype=np.int8),
        np.full(n, int(PlatformType.EVTOL_UAM), dtype=np.int8),
    ])
    intent = np.concatenate([
        np.full(n, int(IntentType.DELIVERY_WP), dtype=np.int8),
        np.full(n, int(IntentType.CORRIDOR_TRANSIT), dtype=np.int8),
    ])
    for h in e.horizons_s:
        e.errors[h] = vals
        e.platform[h] = plat
        e.intent[h] = intent
    e.n_episodes = 1
    diag = cvb.diagnose(e)
    check(str(int(PlatformType.EVTOL_UAM)) in diag["by_platform"],
          "the platform breakdown is missing a class")
    check(diag["by_platform"][str(int(PlatformType.EVTOL_UAM))]["verdict"]
          == cvb.VERDICT_TOO_EASY,
          "the trivially straight platform class was not flagged")
    check(diag["by_intent"][str(int(IntentType.CORRIDOR_TRANSIT))]["verdict"]
          == cvb.VERDICT_TOO_EASY,
          "the trivially straight intent class was not flagged")
    check(len(diag["warnings"]) >= 2,
          f"hidden easy classes must raise warnings: {diag['warnings']}")
    text = cvb.format_report(diag)
    check("VERDICT" in text and "BY PLATFORM TYPE" in text,
          "format_report did not render the breakdown")


def _labelled_errors(rows: list[tuple[float, int, int, int]]) -> "cvb.CVErrors":
    """Build a CVErrors from ``(median_m, n, platform, intent)`` groups."""
    e = cvb.CVErrors.empty()
    vals = np.concatenate([np.full(n, m, dtype=np.float64) for m, n, _p, _i in rows])
    plat = np.concatenate([np.full(n, p, dtype=np.int8) for _m, n, p, _i in rows])
    intent = np.concatenate([np.full(n, i, dtype=np.int8) for _m, n, _p, i in rows])
    for h in e.horizons_s:
        e.errors[h] = vals
        e.platform[h] = plat
        e.intent[h] = intent
    e.n_episodes = 1
    return e


def test_expected_difficulty_covers_every_class() -> None:
    """Every platform and intent must have a declared difficulty expectation.

    A class with no expectation would be graded against the neutral MEDIUM
    band by default, which is a silent way for a new platform to escape the
    gate entirely.
    """
    for platform in PlatformType:
        band = cvb.band_of_platform(int(platform))
        check(band in (cvb.EASY, cvb.MEDIUM, cvb.HARD),
              f"{platform.name}: bad expected difficulty {band!r}")
        check(int(platform) in cvb.EXPECTED_DIFFICULTY["platform"],
              f"{platform.name} is missing from EXPECTED_DIFFICULTY")
    for intent in IntentType:
        check(int(intent) in cvb.EXPECTED_DIFFICULTY["intent"],
              f"{intent.name} is missing from EXPECTED_DIFFICULTY")

    # The declared expectations must match the stability model that produces
    # them: the platforms with the longest manoeuvre commitment are the EASY
    # ones, and the twitchiest are HARD.
    from agents.platforms import STABILITY_PROFILES
    for platform in PlatformType:
        if platform is PlatformType.DEBRIS:
            continue  # unpowered: its profile is inert by construction
        dwell = STABILITY_PROFILES[int(platform)].dwell_min_s
        band = cvb.band_of_platform(int(platform))
        if band == cvb.EASY:
            check(dwell >= 8.0,
                  f"{platform.name} is graded EASY but only commits {dwell} s")
        if band == cvb.HARD:
            check(dwell <= 2.0,
                  f"{platform.name} is graded HARD but commits {dwell} s")

    for band, (lo, hi) in cvb.BAND_LIMITS.items():
        check(0.0 <= lo < hi, f"{band}: bad band limits ({lo}, {hi})")
    check(cvb.BAND_LIMITS[cvb.HARD][0] > cvb.BAND_LIMITS[cvb.EASY][0],
          "the HARD band must demand a larger error than the EASY band")


def test_each_class_is_judged_against_its_own_band() -> None:
    """A low CV error is a PASS for an eVTOL and a FAILURE for a bird.

    This is the whole point of the banded diagnostic. Heterogeneous difficulty
    is a feature of the dataset -- a calibrated model must emit a small sigma
    for a stable eVTOL and a large one for a wandering bird -- so one global
    band would flag the design as a defect.
    """
    n = 3000
    modest = 60.0  # [m] at 15 s: unremarkable for an eVTOL, far too good for a bird
    easy = _labelled_errors([(modest, n, int(PlatformType.EVTOL_UAM),
                              int(IntentType.CORRIDOR_TRANSIT))])
    hard = _labelled_errors([(modest, n, int(PlatformType.BIRD),
                              int(IntentType.SEARCH_WANDER))])
    easy_rec = cvb.diagnose(easy)["by_platform"][str(int(PlatformType.EVTOL_UAM))]
    hard_rec = cvb.diagnose(hard)["by_platform"][str(int(PlatformType.BIRD))]
    check(easy_rec["expected"] == cvb.EASY and easy_rec["verdict"] == cvb.VERDICT_OK,
          f"a {modest} m median on an eVTOL must PASS, got {easy_rec['verdict']}")
    check(hard_rec["expected"] == cvb.HARD
          and hard_rec["verdict"] == cvb.VERDICT_TOO_EASY,
          f"a {modest} m median on a BIRD must FAIL, got {hard_rec['verdict']}")

    # An unpowered body IS constant velocity; a near-zero error is its physics,
    # not a defect, so it is exempt from the lower bound (and only it is).
    debris = cvb.diagnose(_labelled_errors([
        (3.0, n, int(PlatformType.DEBRIS), int(IntentType.PASSIVE_DRIFT))]))
    rec = debris["by_platform"][str(int(PlatformType.DEBRIS))]
    check(rec["verdict"] == cvb.VERDICT_OK,
          f"debris drifting on a steady wind must PASS, got {rec['verdict']}")
    check(rec["band_lo_m"] == 0.0, "the debris exemption is not being applied")
    lo_evtol, _hi = cvb.class_band_limits("platform", int(PlatformType.EVTOL_UAM))
    check(lo_evtol > 0.0, "the exemption leaked to a powered class")


def test_dataset_gate_needs_hard_easy_and_separation() -> None:
    """The gate passes only when all three conditions hold."""
    n = 3000
    good = _labelled_errors([
        (200.0, n, int(PlatformType.BIRD), int(IntentType.SEARCH_WANDER)),
        (40.0, n, int(PlatformType.EVTOL_UAM), int(IntentType.CORRIDOR_TRANSIT)),
    ])
    gate = cvb.diagnose(good)["gate"]
    check(gate["passed"], f"a well-separated dataset must pass: {gate['reasons']}")
    check(gate["separation"]["platform"]["ratio"] == 5.0,
          f"separation ratio is wrong: {gate['separation']['platform']['ratio']}")

    # (a) the HARD classes must genuinely defeat CV
    soft = _labelled_errors([
        (20.0, n, int(PlatformType.BIRD), int(IntentType.SEARCH_WANDER)),
        (10.0, n, int(PlatformType.EVTOL_UAM), int(IntentType.CORRIDOR_TRANSIT)),
    ])
    gate = cvb.diagnose(soft)["gate"]
    check(not gate["passed"] and not gate["hard_ok"],
          "a dataset whose birds are easy must fail the gate")

    # (b) the EASY classes must genuinely be easy
    wild = _labelled_errors([
        (200.0, n, int(PlatformType.BIRD), int(IntentType.SEARCH_WANDER)),
        (400.0, n, int(PlatformType.EVTOL_UAM), int(IntentType.CORRIDOR_TRANSIT)),
    ])
    gate = cvb.diagnose(wild)["gate"]
    check(not gate["passed"] and not gate["easy_ok"],
          "a dataset whose eVTOLs wander must fail the gate")

    # (c) separation: both bands inside their own limits, but too close
    flat = _labelled_errors([
        (100.0, n, int(PlatformType.BIRD), int(IntentType.SEARCH_WANDER)),
        (100.0, n, int(PlatformType.EVTOL_UAM), int(IntentType.CORRIDOR_TRANSIT)),
    ])
    diag = cvb.diagnose(flat)
    gate = diag["gate"]
    check(gate["hard_ok"] and gate["easy_ok"],
          "the per-class verdicts should both be OK in the separation case")
    check(not gate["passed"] and not gate["separation_ok"],
          "equally difficult classes must fail the separation condition")
    check(abs(diag["band_separation"]["platform"]["ratio"] - 1.0) < 1e-9,
          "separation ratio should be 1.0 when both bands have equal error")

    # An empty dataset must never gate green by accident.
    check(not cvb.diagnose(cvb.CVErrors.empty())["gate"]["passed"],
          "an empty diagnosis passed the gate")

    text = cvb.format_report(cvb.diagnose(good))
    check("BAND SEPARATION" in text and "DATASET GATE" in text,
          "format_report does not show the banded verdict")


def test_cv_run_diagnostic(tmp_root: str) -> None:
    """run_diagnostic must aggregate a real run written by DatasetWriter."""
    import shutil

    from scenario.config import ScenarioConfig
    from storage.schema import EpisodeMeta
    from storage.writer import DatasetWriter

    cfg = ScenarioConfig(name="test_cv", seed=1, duration_s=60.0)
    w = DatasetWriter(logs_root=tmp_root)
    run_dir = w.open_run(cfg, preset="unit", tag="cvdiag",
                         include_fingerprint=False)
    for i in range(2):
        ep = _cv_episode(turn_rate=0.08)
        meta = EpisodeMeta(
            episode_index=i, seed=100 + i, density=40.0, n_agents=3,
            platform_counts={int(PlatformType.FIXED_WING_UAS): 3},
            intent_counts={int(IntentType.CORRIDOR_TRANSIT): 3},
            duration_s=60.0, min_separation_m=800.0,
            risk_level=int(RiskLevel.LOW),
        )
        w.write_episode(ep, meta)
    w.close()

    report = cvb.run_diagnostic(run_dir)
    check(report["n_episodes"] == 2, "run_diagnostic lost an episode")
    check(report["overall"]["n"] > 0, "run_diagnostic found no samples")
    check("15" in report["per_horizon"], "per-horizon table is missing 15 s")
    check(report["verdict"] in {cvb.VERDICT_OK, cvb.VERDICT_TOO_EASY,
                                cvb.VERDICT_TOO_RANDOM,
                                cvb.VERDICT_INSUFFICIENT},
          f"unexpected verdict {report['verdict']}")
    check(report["run_dir"].endswith(os.path.basename(run_dir)),
          "run_diagnostic did not record the run directory")
    shutil.rmtree(run_dir, ignore_errors=True)


def main() -> int:
    """Run every test and report. Returns a process exit code."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="prism_metrics_test_")
    try:
        test_ade_fde_hand_computed()
        test_min_ade_fde_k()
        test_gaussian_nll_closed_form()
        test_crps_closed_form_and_ensemble()
        test_energy_score()
        test_chi2_quantile()
        test_coverage_of_known_gaussian()
        test_pit_uniformity()
        test_separation_and_nmac()
        test_tau_mod_and_dwc()
        test_nmac_implies_dwc_invariant()
        test_separation_metrics_on_episode()
        test_cv_predict_and_errors()
        test_cv_diagnosis_verdicts()
        test_cv_class_breakdown_catches_hidden_easy_class()
        test_expected_difficulty_covers_every_class()
        test_each_class_is_judged_against_its_own_band()
        test_dataset_gate_needs_hard_easy_and_separation()
        test_cv_run_diagnostic(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} of {_CHECKS} checks")
        for f in _FAILURES:
            print("  -", f)
        return 1
    print(f"ok  ({_CHECKS} checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
