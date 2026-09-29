# -*- coding: utf-8 -*-
"""Track-rule tests for the baselines (review round 1, R3 major 3).
Run via `python main.py selftest` or standalone.

Protected properties:
  1. rule "slot" is the pre-round-1 code, bit-for-bit (CV-obs and IMM vs
     verbatim copies of the old functions)
  2. CV-obs under "current" fits only the track open at t0: exact on a
     reacquired target whose earlier track flew another velocity, and equal
     to the prediction on an episode with the old track deleted
  3. IMM under "current" restarts on a new track id: the forecast equals a
     fresh filter run on the new track alone; no forecast without an open
     track; the coasting old track still forecasts before it closes
  4. harness: EP_KEYS carries obs_track_id; reacq_class == build_sample's
     meta_reacq on every sample; sample counts follow the index rule and
     mismatched rules are counted as skipped, never silently dropped
  5. data-config resolution: default rule = compared model's (legacy -> slot),
     else "current"; overrides; per-sample file keyed like the model's
  6. IMM init-gap fix is opt-in and correct (legacy default unchanged)
  7. IMM tuner: parallel == serial, argmin pick, edge re-centring
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

from common.constants import HIST_STEPS, PRED_STEPS               # noqa: E402
from data.dataset import build_sample, build_sample_index         # noqa: E402
from methods.cv import FittedSigma, cv_predict_obs                # noqa: E402
from methods.imm import (IMMParams, IMMTracker,                   # noqa: E402
                         imm_predictions_for_episode)

LOG_DT = 0.5
PASS = FAIL = 0
T_EP = 80
V1 = np.array([10.0, 0.0])
V2 = np.array([0.0, 12.0])


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


# ------------------------------------------------ verbatim pre-round-1 code

def _legacy_cv_predict_obs(ep, t0, target, fit_window=5):
    h0 = t0 - HIST_STEPS + 1
    det = ep["obs_detected"][h0:t0 + 1, target].astype(bool)
    hs = np.flatnonzero(det)
    if hs.size < 2:
        return None
    use = hs[-fit_window:]
    ts = use.astype(np.float64) * LOG_DT
    ps = ep["obs_pos"][h0 + use, target].astype(np.float64)
    t_mean = ts.mean()
    denom = float(((ts - t_mean) ** 2).sum())
    if denom < 1e-9:
        return None
    v = ((ts - t_mean)[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / denom
    p_at_last = ps.mean(axis=0) + v * (ts[-1] - t_mean)
    dt_to_t0 = (HIST_STEPS - 1 - hs[-1]) * LOG_DT
    steps = np.arange(1, PRED_STEPS + 1, dtype=np.float64) * LOG_DT + dt_to_t0
    return p_at_last[None, :] + v[None, :] * steps[:, None]


def _legacy_imm(ep, t0_list, params):
    out = {}
    det_all = ep["obs_detected"].astype(bool)
    for a, anchors in t0_list.items():
        tracker = IMMTracker(params)
        want = set(int(t) for t in anchors)
        for t in range(0, max(want) + 1):
            if det_all[t, a]:
                tracker.step(ep["obs_pos"][t, a].astype(np.float64),
                             ep["obs_R"][t, a].astype(np.float64))
            else:
                tracker.step(None, None)
            if t in want and tracker.ready:
                out[(a, t)] = tracker.predict()
    return out


# ------------------------------------------------------------------ fixture

def _episode(seed: int = 0) -> dict:
    """3 agents, noise-free observations. Agent 0 flies V1 on track id 3
    (detected 0..14), is missed 15..21 (id 3 coasting to 17, closed -1 at
    18..21) while it turns to V2, and is reacquired as id 7 from step 22.
    Agent 1: one track, always detected. Agent 2: random misses, one track
    closed and reopened at step 40."""
    rng = np.random.default_rng(seed)
    T, N = T_EP, 3
    gt = np.zeros((T, N, 2))
    p = np.array([0.0, 0.0])
    for t in range(T):
        gt[t, 0] = p
        p = p + (V1 if t < 15 else V2) * LOG_DT
    gt[:, 1] = np.array([500.0, 0.0]) + np.arange(T)[:, None] * np.array([-8.0, 3.0]) * LOG_DT
    ang = np.cumsum(np.full(T, 0.05))
    gt[:, 2] = np.array([-300.0, 200.0]) + np.cumsum(
        15 * LOG_DT * np.stack([np.cos(ang), np.sin(ang)], 1), axis=0)
    det = np.ones((T, N), bool)
    tid = np.zeros((T, N), np.int32)
    det[15:22, 0] = False
    tid[:, 0] = 3; tid[18:22, 0] = -1; tid[22:, 0] = 7
    tid[:, 1] = 5
    miss = rng.random(T) < 0.2
    det[:, 2] = ~miss
    det[33:40, 2] = False                        # long gap -> new track
    tid[:, 2] = 1; tid[36:40, 2] = -1; tid[40:, 2] = 9
    obs = np.where(det[..., None], gt, np.nan)
    R = np.where(det[..., None, None], np.eye(2) * 4.0, np.nan)
    age = np.zeros((T, N), np.int32)
    for a in range(N):
        n = 0
        for t in range(T):
            n = n + 1 if (tid[t, a] >= 0 and t > 0 and tid[t - 1, a] == tid[t, a]) else 0
            age[t, a] = n if tid[t, a] >= 0 else 0
    return {"t": (np.arange(T) * LOG_DT).astype(np.float32),
            "gt_pos": gt.astype(np.float32),
            "gt_vel": np.gradient(gt, LOG_DT, axis=0).astype(np.float32),
            "gt_valid": np.ones((T, N), bool),
            "gt_type": np.array([2, 1, 3], np.int8),
            "gt_intent": np.zeros((T, N), np.int8),
            "own_pos": np.zeros((T, 2), np.float32),
            "obs_detected": det, "obs_pos": obs.astype(np.float32),
            "obs_R": R.astype(np.float32), "obs_track_id": tid,
            "obs_track_age": age}


def _run_dir(tmp: str) -> str:
    run = os.path.join(tmp, "20990101_000000_trackrule")
    os.makedirs(os.path.join(run, "episodes"))
    for i in range(2):
        np.savez_compressed(os.path.join(run, "episodes", f"ep_{i:06d}.npz"), **_episode(i))
    return run


# -------------------------------------------------------------------- tests

def test_slot_is_legacy():
    ep = _episode(1)
    same = True
    for a in range(3):
        for t0 in range(HIST_STEPS - 1, T_EP - PRED_STEPS):
            new, old = cv_predict_obs(ep, t0, a, rule="slot"), _legacy_cv_predict_obs(ep, t0, a)
            if (new is None) != (old is None) or (new is not None and not np.array_equal(new, old)):
                same = False
    check(same, "CV-obs slot == pre-round-1 code on every (t0, agent)")
    anchors = {a: list(range(HIST_STEPS - 1, T_EP - PRED_STEPS, 3)) for a in range(3)}
    new = imm_predictions_for_episode(ep, anchors, IMMParams(), rule="slot")
    old = _legacy_imm(ep, anchors, IMMParams())
    check(new.keys() == old.keys() and all(
        all(np.array_equal(x, y) for x, y in zip(new[k], old[k])) for k in old),
        "IMM slot == pre-round-1 code (all anchors, all outputs)")


def test_cv_current():
    ep = _episode(0)
    t0 = 24                                   # current track: detections 22, 23, 24
    fut = ep["gt_pos"][t0 + 1:t0 + 1 + PRED_STEPS, 0].astype(np.float64)
    cur = cv_predict_obs(ep, t0, 0, rule="current")
    slot = cv_predict_obs(ep, t0, 0, rule="slot")
    err_c = float(np.max(np.hypot(*(cur - fut).T)))
    err_s = float(np.max(np.hypot(*(slot - fut).T)))
    check(err_c < 1e-3, f"CV current exact on the reacquired track (max err {err_c:.2e} m)")
    check(err_s > 20.0, f"CV slot mixes the old track in (max err {err_s:.1f} m)")
    ep_del = {k: v.copy() for k, v in ep.items()}
    ep_del["obs_detected"][:15, 0] = False
    check(np.array_equal(cur, cv_predict_obs(ep_del, t0, 0, rule="slot")),
          "CV current == CV on the episode with the closed track deleted")
    check(cv_predict_obs(ep, 20, 0, rule="current") is None,
          "CV current: no open track at t0 -> no prediction")


def test_imm_current():
    ep = _episode(0)
    p = IMMParams()
    got = imm_predictions_for_episode(ep, {0: [16, 20, 30]}, p, rule="current")
    fresh = IMMTracker(p)
    for t in range(22, 31):                    # the new track alone
        fresh.step(ep["obs_pos"][t, 0].astype(np.float64), ep["obs_R"][t, 0].astype(np.float64))
    ref = fresh.predict()
    check((0, 30) in got and all(np.array_equal(x, y) for x, y in zip(got[(0, 30)], ref)),
          "IMM current at t0=30 == fresh filter on track 7 only")
    check((0, 20) not in got, "IMM current: no forecast while no track is open")
    check((0, 16) in got, "IMM current: the still-open old track (coasting) forecasts")
    old_only = IMMTracker(p)
    for t in range(0, 17):
        if ep["obs_detected"][t, 0]:
            old_only.step(ep["obs_pos"][t, 0].astype(np.float64), ep["obs_R"][t, 0].astype(np.float64))
        else:
            old_only.step(None, None)
    check(all(np.array_equal(x, y) for x, y in zip(got[(0, 16)], old_only.predict())),
          "IMM current at t0=16 == filter on track 3")
    slot = imm_predictions_for_episode(ep, {0: [30]}, p, rule="slot")
    fut = ep["gt_pos"][31:31 + PRED_STEPS, 0]
    best = lambda pr: min(float(np.mean(np.hypot(*(pr[0][k] - fut).T))) for k in range(3))  # noqa: E731
    check(best(got[(0, 30)]) < best(slot[(0, 30)]),
          f"restarted filter beats the slot filter after the turn "
          f"({best(got[(0, 30)]):.1f} vs {best(slot[(0, 30)]):.1f} m)")


def test_imm_init_gap():
    """Opt-in fix: the two-point velocity init spans the misses in between.
    Legacy (default) keeps dividing by one step -- 4x velocity here."""
    v = np.array([12.0, -4.0])
    R = np.eye(2) * 4.0
    for fix, factor in ((False, 4.0), (True, 1.0)):
        tr = IMMTracker(IMMParams(), init_gap_fix=fix)
        tr.step(np.zeros(2), R)
        for _ in range(3):
            tr.step(None, None)                  # three misses
        tr.step(v * 4 * LOG_DT, R)               # second detection, 2 s later
        check(np.allclose(tr.models[0].x[2:4], factor * v),
              f"IMM init velocity (init_gap_fix={fix}) = {tr.models[0].x[2:4]} "
              f"expected {factor} x {v}")
    ep = _episode(0)
    a = imm_predictions_for_episode(ep, {2: [45]}, IMMParams(), rule="current")
    b = imm_predictions_for_episode(ep, {2: [45]}, IMMParams(), rule="current",
                                    init_gap_fix=False)
    check(all(np.array_equal(x, y) for x, y in zip(a[(2, 45)], b[(2, 45)])),
          "init_gap_fix defaults to the legacy behaviour")


def test_harness(tmp: str):
    from harness.data import (EP_KEYS, FILE_KEY, iter_eval_groups, reacq_class,
                              resolve_data_cfg)
    from harness.evaluate import evaluate_cv, evaluate_imm
    from model.metrics import load_per_sample

    check("obs_track_id" in EP_KEYS, "EP_KEYS carries obs_track_id")
    run = _run_dir(tmp)
    base = {"max_agents": 4, "t0_stride": 1, "min_obs": 3, "min_future_valid": 27}
    slot_cfg = dict(base, track_rule="slot")
    eps = list(iter_eval_groups([run], slot_cfg))
    check(all("obs_track_id" in ep and FILE_KEY in ep for ep, _ in eps),
          "iter_eval_groups passes obs_track_id and the file key")
    idx = build_sample_index([run], slot_cfg)
    mism = 0
    classes = set()
    for i in range(len(idx)):
        f, t0, a = idx.files[idx.file_idx[i]], int(idx.t0[i]), int(idx.agent[i])
        with np.load(f) as z:
            ep = {k: z[k] for k in z.files}
        r = reacq_class(ep, t0, a)
        classes.add(r)
        mism += int(r != int(build_sample(f, t0, a, slot_cfg)["meta_reacq"]))
    check(mism == 0, f"reacq_class == build_sample meta_reacq ({mism} mismatches)")
    check(classes == {0, 1, 2}, f"fixture covers reacq classes 0/1/2 ({sorted(classes)})")

    fitted = FittedSigma.from_dict({"sa": [10.0] * PRED_STEPS, "sc": [5.0] * PRED_STEPS})
    n_idx = {r: len(build_sample_index([run], dict(base, track_rule=r))) for r in ("slot", "current")}
    check(n_idx["current"] < n_idx["slot"], f"current index smaller {n_idx}")
    for rule in ("slot", "current"):
        cfg = dict(base, track_rule=rule)
        ps = os.path.join(tmp, f"ps_{rule}.npz")
        rep = evaluate_cv([run], cfg, fitted, "obs", per_sample_path=ps)
        check(rep["n_samples"] == n_idx[rule] and rep["n_skipped"] == 0,
              f"cv-obs {rule}: n={rep['n_samples']} of index {n_idx[rule]}, "
              f"skipped {rep['n_skipped']}")
        check(rep["track_rule"] == rule and "top1ADE" in rep["overall"]
              and "reacq" in rep["strata"], f"cv-obs {rule}: report keys")
        d = load_per_sample(ps)
        check(len(d["t0"]) == rep["n_samples"], f"cv-obs {rule}: per-sample rows")
        rep_i = evaluate_imm([run], cfg, IMMParams())
        check(rep_i["n_samples"] == n_idx[rule] and rep_i["n_skipped"] == 0,
              f"imm {rule}: n={rep_i['n_samples']} of index {n_idx[rule]}, "
              f"skipped {rep_i['n_skipped']}")
    # features "current" on the slot index: samples without an open track
    # cannot be predicted and must be COUNTED, not silently dropped
    mixed = dict(base, track_rule="current", index_track_rule="slot")
    rep = evaluate_cv([run], mixed, fitted, "obs")
    check(rep["n_skipped"] > 0 and rep["n_samples"] + rep["n_skipped"] == n_idx["slot"],
          f"mismatched rules are counted: n={rep['n_samples']} skipped={rep['n_skipped']}")

    legacy = {"max_agents": 10, "t0_stride": 4, "min_obs": 3, "min_future_valid": 27}
    check(resolve_data_cfg(legacy, None, None)["track_rule"] == "slot",
          "legacy model config -> slot")
    check(resolve_data_cfg(dict(legacy, track_rule="current"), None, None)["track_rule"] == "current",
          "current-trained model -> current")
    check(resolve_data_cfg(None, None, None)["track_rule"] == "current", "no model -> current")
    r = resolve_data_cfg(legacy, "current", "slot")
    check(r["track_rule"] == "current" and r["index_track_rule"] == "slot"
          and r["t0_stride"] == 4, "overrides keep the model's eligibility knobs")


def test_tuner(tmp: str):
    """tune_imm_protocol: workers do not change results; argmin over all
    points; the edge rule re-centres a 3x3 {x/3, x, 3x} pass."""
    from harness.data import DEFAULT_DATA_CFG
    from harness.evaluate import tune_imm_protocol
    run = _run_dir(os.path.join(tmp, "tuner"))
    cfg = dict(DEFAULT_DATA_CFG, t0_stride=2, track_rule="current")
    passes = (("p", (0.05, 0.5), (0.1, 1.0)),)
    kw = dict(init_gap_fix=True, max_samples=40, passes=passes, log=lambda s: None)
    p1, h1, b1 = tune_imm_protocol([run], cfg, workers=1, **kw)
    p2, h2, b2 = tune_imm_protocol([run], cfg, workers=2, **kw)
    key = lambda h: [(r["q"], r["q_w"], r["NLL"], r["n"]) for r in h]  # noqa: E731
    check(key(h1) == key(h2) and p1.to_dict() == p2.to_dict(),
          "tuner: 2 workers == serial (every point, same pick)")
    check(b1["NLL"] == min(r["NLL"] for r in h1), "tuner: pick = argmin over all points")
    core = {(r["q"], r["q_w"]) for r in h1 if r["pass"] == "p"}
    extra = [r for r in h1 if r["pass"].startswith("extra")]
    check(len(core) == 4 and all(r["n"] == 40 for r in h1),
          f"tuner: 2x2 pass on the 40-sample cap ({len(core)} points)")
    # on a 2x2 grid every point is an edge point -> at least one 3x3 re-centring
    check(len(extra) >= 1, f"tuner: edge argmin re-centred ({len(extra)} extra points)")


def run_all() -> bool:
    global PASS, FAIL
    PASS = FAIL = 0
    tmp = tempfile.mkdtemp(prefix="prism_bl_track_")
    try:
        test_slot_is_legacy()
        test_cv_current()
        test_imm_current()
        test_imm_init_gap()
        test_harness(tmp)
        test_tuner(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_track_rule: {PASS} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
