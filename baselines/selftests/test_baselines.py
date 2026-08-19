# -*- coding: utf-8 -*-
"""Baseline correctness tests. Run via `python main.py selftest`.

Protected properties:
  1. CV-obs reproduces a straight track exactly (zero error, sensor-perfect)
  2. FittedSigma: residuals drawn from a known Gaussian -> coverage ~ nominal
  3. CT-EKF converges to a known constant turn rate
  4. IMM: model probabilities follow a CV->turn switch on synthetic data
  5. IMM mixture predict: straight flier stays tight, turner curves
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

from common.constants import HIST_STEPS, PRED_STEPS               # noqa: E402
from methods.cv import FittedSigma, cv_predict_obs                # noqa: E402
from methods.imm import IMMParams, IMMTracker                     # noqa: E402

LOG_DT = 0.5
PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


def _straight_ep(v=(20.0, 5.0), n_agents=1, T=70):
    """Minimal episode dict: one agent, perfect detections, straight."""
    t = np.arange(T) * LOG_DT
    pos = np.stack([100 + v[0] * t, -50 + v[1] * t], axis=1)
    ep = {
        "gt_pos": pos[:, None, :].astype(np.float32),
        "gt_vel": np.tile(np.array(v, np.float32), (T, 1))[:, None, :],
        "gt_valid": np.ones((T, 1), bool),
        "gt_type": np.array([3], np.int8),
        "gt_intent": np.zeros((T, 1), np.int8),
        "obs_detected": np.ones((T, 1), bool),
        "obs_pos": pos[:, None, :].astype(np.float32),
        "obs_R": np.tile(np.eye(2, dtype=np.float32) * 4.0, (T, 1, 1, 1)),
    }
    return ep


def test_cv_exact():
    ep = _straight_ep()
    pred = cv_predict_obs(ep, t0=30, target=0)
    gt = ep["gt_pos"][31:31 + PRED_STEPS, 0]
    err = np.max(np.hypot(*(pred - gt).T))
    check(err < 1e-3, f"CV exact on straight track (max err {err:.2e} m)")


def test_fitted_sigma_coverage():
    rng = np.random.default_rng(0)
    f = FittedSigma()
    heading = np.array([1.0, 0.0])
    base = np.cumsum(np.tile(heading * 10, (PRED_STEPS, 1)), axis=0)
    for _ in range(3000):
        noise = rng.normal(0, [8.0, 3.0], size=(PRED_STEPS, 2))
        f.add_residuals(base + noise, base, np.ones(PRED_STEPS))
    f.finalize()
    check(abs(f.sa[-1] - 8.0) < 0.5, f"sigma_along fitted {f.sa[-1]:.2f} ~ 8")
    check(abs(f.sc[-1] - 3.0) < 0.5, f"sigma_cross fitted {f.sc[-1]:.2f} ~ 3")
    # nominal-coverage check at the last step, 1-sigma ellipse ~ 39.3%
    sig, rho = f.attach(base)
    inside = 0
    n = 4000
    for _ in range(n):
        e = rng.normal(0, [8.0, 3.0])
        m = (e[0] / sig[-1, 0]) ** 2 + (e[1] / sig[-1, 1]) ** 2
        inside += m <= 1.0
    frac = inside / n
    check(abs(frac - 0.393) < 0.03, f"1-sigma coverage {frac:.3f} ~ 0.393")


def _run_tracker(traj, R_std=2.0, params=None):
    tr = IMMTracker(params or IMMParams())
    R = np.eye(2) * R_std ** 2
    for z in traj:
        tr.step(z + np.zeros(2), R)
    return tr


def test_ct_convergence():
    w_true = 0.15                      # rad/s
    v, psi = 30.0, 0.0
    p = np.zeros(2)
    traj = []
    for _ in range(80):
        psi += w_true * LOG_DT
        p = p + v * LOG_DT * np.array([math.cos(psi), math.sin(psi)])
        traj.append(p.copy())
    tr = _run_tracker(traj)
    w_est = tr.models[1].x[4]
    check(abs(w_est - w_true) < 0.03,
          f"CT-EKF turn rate {w_est:.3f} ~ {w_true}")
    check(tr.mu[1] > 0.5, f"IMM weights CT on a turning target (mu={tr.mu})")


def test_imm_switch():
    v = 25.0
    p = np.zeros(2)
    traj = []
    psi = 0.0
    for k in range(120):
        w = 0.0 if k < 60 else 0.2
        psi += w * LOG_DT
        p = p + v * LOG_DT * np.array([math.cos(psi), math.sin(psi)])
        traj.append(p.copy())
    tr = IMMTracker(IMMParams())
    R = np.eye(2) * 4.0
    mu_mid = mu_end = None
    for k, z in enumerate(traj):
        tr.step(z, R)
        if k == 55:
            mu_mid = tr.mu.copy()
        if k == 119:
            mu_end = tr.mu.copy()
    check(mu_mid[0] > 0.5, f"CV dominant while straight (mu={np.round(mu_mid,2)})")
    check(mu_end[1] > 0.5, f"CT dominant after switch (mu={np.round(mu_end,2)})")


def test_imm_predict_shapes():
    ep = _straight_ep()
    from methods.imm import imm_predictions_for_episode
    preds = imm_predictions_for_episode(ep, {0: [30, 40]}, IMMParams())
    check((0, 30) in preds and (0, 40) in preds, "predictions at both anchors")
    mu, sig, rho, pi = preds[(0, 30)]
    check(mu.shape == (3, PRED_STEPS, 2), f"mu shape {mu.shape}")
    check(abs(pi.sum() - 1.0) < 1e-6, "pi sums to 1")
    gt = ep["gt_pos"][31:31 + PRED_STEPS, 0]
    best = min(np.mean(np.hypot(*(mu[k] - gt).T)) for k in range(3))
    check(best < 5.0, f"straight-track IMM minADE {best:.2f} m < 5")
    check(np.all(sig[:, -1] > sig[:, 0]), "sigma grows with horizon")


def run_all() -> bool:
    for fn in (test_cv_exact, test_fitted_sigma_coverage, test_ct_convergence,
               test_imm_switch, test_imm_predict_shapes):
        fn()
    print(f"test_baselines: {PASS} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
