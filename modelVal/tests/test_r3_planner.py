"""Review-round (Reviewer 3) planner-batch checks. Standalone script:

  python tests/test_r3_planner.py        (exit code != 0 on failure)

Covers: disk_kinds alignment with disks_batch; the shared hard geometry is
the 'none' arm's geometry (and the none arm plans identically under
hard_geom arm/shared); the CV fit ignores earlier-track rows under the
current-track rule; the sigma-floor statistics against an independent
reconstruction through disks_batch; collision attribution; the 20 Hz
pairwise/union segmentation and both avoidability screens on a synthetic
separation series; the sign-test and paired-bootstrap helpers; the
closed-loop forecast log (file naming across shards, exclusive create,
stored fields) and the aggregator's parameterized strata and counts.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bridge  # noqa: E402,F401

import numpy as np  # noqa: E402

from planner.frenet import (FrenetState, PlannerConfig,  # noqa: E402
                            ReferencePath, evaluate_candidate,
                            explain_candidate, plan_trajectory)
from planner.risk import (KIND_FORECAST, KIND_STATIC, NODE_T,  # noqa: E402
                          PRED_STEPS, branch_floor_counts, make_risk)
from experiments import aggregate_r3 as agg  # noqa: E402
from experiments.batch import (T_NEED_S, TrackBook,  # noqa: E402
                               _runs, pred_log_path, safety_endpoints,
                               save_pred_log)

T_STEPS, N_AG = 25, 3
T0 = 24


def synthetic_ep() -> dict:
    """Three agents detected at T0. Agent 0 was tracked as id 5 on rows
    14..17 (moving +x at 20 m/s), lost, then reacquired as id 7 on rows
    20..24 moving +y at 10 m/s. Agent 1: one steady track (id 2), moving -y.
    Agent 2: a single detection at T0 (id 9)."""
    det = np.zeros((T_STEPS, N_AG), bool)
    pos = np.full((T_STEPS, N_AG, 2), np.nan)
    tid = np.full((T_STEPS, N_AG), -1, np.int32)
    for j in range(14, 18):
        det[j, 0] = True
        tid[j, 0] = 5
        pos[j, 0] = (1000.0 + 20.0 * 0.5 * (j - 14), 30.0)
    for j in range(20, 25):
        det[j, 0] = True
        tid[j, 0] = 7
        pos[j, 0] = (1400.0, 30.0 + 10.0 * 0.5 * (j - 20))
    for j in range(10, 25):
        det[j, 1] = True
        tid[j, 1] = 2
        pos[j, 1] = (600.0, 300.0 - 15.0 * 0.5 * (j - 10))
    det[T0, 2] = True
    tid[T0, 2] = 9
    pos[T0, 2] = (300.0, -20.0)
    return {"obs_detected": det, "obs_pos": pos, "obs_track_id": tid}


def synthetic_preds(ep: dict, slots=(0, 1), sig_fn=None, pi=(0.5, 0.3, 0.2)):
    out = {}
    for a in slots:
        p0 = ep["obs_pos"][T0, a]
        K = len(pi)
        mu = np.zeros((K, PRED_STEPS, 2))
        for k in range(K):
            ang = 0.4 * (k - 1)
            v = 15.0 * np.array([math.cos(ang), math.sin(ang)])
            mu[k] = p0[None, :] + v[None, :] * NODE_T[:, None]
        s = (np.broadcast_to((5.0 + np.arange(PRED_STEPS))[None, :, None],
                             (K, PRED_STEPS, 2)).copy()
             if sig_fn is None else sig_fn(K))
        out[a] = {"mu_w": mu.astype(np.float32), "sigma": s.astype(np.float32),
                  "rho": np.zeros((K, PRED_STEPS), np.float32),
                  "pi": np.asarray(pi, np.float32), "theta": 0.0}
    return out


def updated(kind: str, ep: dict, preds=None, **kw):
    r = make_risk(kind, **kw)
    r.update(ep, T0, preds)
    return r


TS = np.arange(0.0, 15.0 + 1e-9, 0.5)


class DiskKindsTests(unittest.TestCase):
    def test_alignment_every_arm(self):
        ep = synthetic_ep()
        preds = synthetic_preds(ep)
        for kind in ("none", "cv", "model", "mc_cvsig", "cvc_msig"):
            r = updated(kind, ep, preds)
            d = r.disks_batch(TS)
            k = r.disk_kinds()
            self.assertEqual(k.shape[0], d.shape[1], kind)
            moving = np.any(np.abs(d[:, :, 0:2] - d[0:1, :, 0:2]) > 1e-9,
                            axis=(0, 2))
            # a static disk never moves; every moving disk is a forecast
            self.assertFalse(np.any(moving & (k == KIND_STATIC)), kind)
            if kind == "none":
                self.assertTrue(np.all(k == KIND_STATIC))
            if kind in ("cv", "cvc_msig"):
                self.assertTrue(np.all(k == KIND_FORECAST))
            if kind in ("model", "mc_cvsig"):
                # agents 0,1 predicted (3 branches each), agent 2 static
                self.assertEqual(list(k), [0] + [1] * 6, kind)

    def test_model_without_preds_is_all_static(self):
        r = updated("model", synthetic_ep(), {})
        self.assertTrue(np.all(r.disk_kinds() == KIND_STATIC))
        self.assertEqual(r.disk_kinds().shape[0], r.disks_batch(TS).shape[1])


class SharedHardGeometryTests(unittest.TestCase):
    def test_equals_none_arm_geometry(self):
        ep = synthetic_ep()
        preds = synthetic_preds(ep)
        ref = updated("none", ep).disks_batch(TS)[:, :, :3]
        for kind in ("none", "cv", "model", "mc_cvsig", "cvc_msig"):
            r = updated(kind, ep, preds)
            r.update_hard(ep, T0)
            h = r.hard_disks_batch(TS)[:, :, :3]
            np.testing.assert_array_equal(h, ref, err_msg=kind)

    def test_hard_disks_need_update_hard(self):
        r = updated("cv", synthetic_ep())
        with self.assertRaises(RuntimeError):
            r.hard_disks_batch(TS)

    def _plan(self, risk, hard_geom):
        cfg = PlannerConfig(desired_speed=30.0, corridor_width=200.0,
                            lateral_step=20.0, horizon=12.0,
                            duration_max=15.0, plan_dt=0.5, risk_horizon=15.0,
                            soft_standoff=152.4, hard_geom=hard_geom)
        path = ReferencePath.demo("straight")
        st = FrenetState(s=100.0, d=0.0, sd=30.0)
        return plan_trajectory(st, cfg, path, risk), cfg

    def test_none_arm_plans_identically(self):
        ep = synthetic_ep()
        r = updated("none", ep)
        r.update_hard(ep, T0)
        a, _ = self._plan(r, "arm")
        b, _ = self._plan(r, "shared")
        self.assertEqual(len(a.candidates), len(b.candidates))
        for ca, cb in zip(a.candidates, b.candidates):
            self.assertEqual((ca.valid, ca.collision, ca.violation,
                              ca.min_clearance, ca.raw["safety"]),
                             (cb.valid, cb.collision, cb.violation,
                              cb.min_clearance, cb.raw["safety"]))
        self.assertEqual(a.best.total_cost, b.best.total_cost)

    def test_shared_ignores_forecast_collisions(self):
        # an intruder 600 m ahead of the ownship (s=100 -> x=700), closing
        # head-on at 40 m/s: its CV forecast disks meet the ownship near
        # t = 8.6 s, while its observed position is reached by no candidate
        # within the 15 s risk horizon
        det = np.zeros((T_STEPS, 1), bool)
        pos = np.full((T_STEPS, 1, 2), np.nan)
        tid = np.full((T_STEPS, 1), -1, np.int32)
        for j in range(16, 25):
            det[j, 0] = True
            tid[j, 0] = 1
            pos[j, 0] = (700.0 - 40.0 * 0.5 * (j - 24), 0.0)
        ep = {"obs_detected": det, "obs_pos": pos, "obs_track_id": tid}
        r = updated("cv", ep)
        r.update_hard(ep, T0)
        a, cfg_a = self._plan(r, "arm")
        b, cfg_b = self._plan(r, "shared")
        self.assertTrue(any(c.collision for c in a.candidates))
        self.assertLess(sum(c.collision for c in b.candidates),
                        sum(c.collision for c in a.candidates))
        # soft cost is the arm's in both modes
        for ca, cb in zip(a.candidates, b.candidates):
            self.assertAlmostEqual(ca.raw["safety"], cb.raw["safety"],
                                   places=12)
        hit = next(c for c in a.candidates if c.collision)
        ex = explain_candidate(hit, cfg_a, r)
        self.assertTrue(ex["collision"])
        self.assertEqual(ex["coll_kind"], KIND_FORECAST)
        self.assertGreater(ex["pen_depth_m"], 0.0)
        self.assertAlmostEqual(ex["pen_depth_m"], -hit.min_clearance, places=9)


class ExplainTests(unittest.TestCase):
    def test_static_collision_and_breakdown(self):
        det = np.ones((T_STEPS, 1), bool)
        pos = np.tile(np.array([[[400.0, 0.0]]]), (T_STEPS, 1, 1))
        ep = {"obs_detected": det, "obs_pos": pos,
              "obs_track_id": np.zeros((T_STEPS, 1), np.int32)}
        r = updated("none", ep)
        cfg = PlannerConfig(desired_speed=30.0, corridor_width=200.0,
                            lateral_step=20.0, horizon=12.0,
                            duration_max=15.0, plan_dt=0.5, risk_horizon=15.0)
        path = ReferencePath.demo("straight")
        st = FrenetState(s=100.0, d=0.0, sd=30.0)
        c = evaluate_candidate(st, 0.0, 0.0, 30.0, 12.0, cfg, path, r, "2d")
        self.assertTrue(c.collision)
        ex = explain_candidate(c, cfg, r)
        self.assertEqual(ex["coll_kind"], KIND_STATIC)
        self.assertTrue(ex["coll_static"] and not ex["coll_forecast"])
        self.assertAlmostEqual(ex["pen_depth_m"], -c.min_clearance, places=9)
        # violation = collision penalty + boundary + dynamic parts
        self.assertFalse(ex["boundary"])
        # a candidate far off the corridor edge is a boundary violation
        c2 = evaluate_candidate(st, 120.0, 0.0, 30.0, 6.0, cfg, path, r, "2d")
        ex2 = explain_candidate(c2, cfg, r)
        self.assertTrue(ex2["boundary"] and c2.boundary)
        self.assertGreater(ex2["boundary_excess_m"], 0.0)


class CurrentTrackFitTests(unittest.TestCase):
    def test_cv_ignores_closed_track(self):
        ep = synthetic_ep()
        cur = updated("cv", ep, track_rule="current")
        slot = updated("cv", ep, track_rule="slot")
        i0 = cur._slots.index(0)
        # current: only rows 20..24 of track 7 (+y at 10 m/s) are fitted
        np.testing.assert_allclose(cur._vel[i0], [0.0, 10.0], atol=1e-9)
        # slot: the window 17..24 also holds row 17 of the closed track 5
        self.assertGreater(abs(slot._vel[slot._slots.index(0)][0]), 1.0)
        # a steady single-track agent is fitted identically by both rules
        i1 = cur._slots.index(1)
        np.testing.assert_array_equal(cur._vel[i1],
                                      slot._vel[slot._slots.index(1)])
        np.testing.assert_allclose(cur._vel[i1], [0.0, -15.0], atol=1e-9)

    def test_current_needs_track_ids(self):
        ep = synthetic_ep()
        ep.pop("obs_track_id")
        updated("cv", ep, track_rule="slot")             # legacy: fine
        with self.assertRaises(KeyError):
            updated("cv", ep, track_rule="current")

    def test_bad_rule_rejected(self):
        with self.assertRaises(ValueError):
            make_risk("none", track_rule="oracle")


class FloorStatTests(unittest.TestCase):
    """floor_stats vs an independent count through disks_batch: the same
    provider with sigma_floor=0 exposes the pre-floor sigma at the nodes."""

    def _prefloor_count(self, kind, ep, preds, floor, **kw):
        r0 = updated(kind, ep, preds, sigma_floor=0.0, **kw)
        d = r0.disks_batch(NODE_T)                     # [30, M, 5]
        fc = r0.disk_kinds() == KIND_FORECAST
        pre = d[:, fc, 3]
        return int((pre < floor).sum()), int(pre.size)

    def test_model_and_cv_arms(self):
        ep = synthetic_ep()
        for scale in (1.0, 0.5):
            preds = synthetic_preds(ep)
            for kind in ("model", "mc_cvsig", "cvc_msig", "cv"):
                kw = {} if kind == "cv" else {"sigma_scale": scale}
                for floor in (10.0, 30.0):
                    r = updated(kind, ep, preds, sigma_floor=floor, **kw)
                    self.assertEqual(r.floor_stats(),
                                     self._prefloor_count(kind, ep, preds,
                                                          floor, **kw),
                                     (kind, scale, floor))

    def test_known_counts(self):
        ep = synthetic_ep()
        preds = synthetic_preds(ep)                     # sigma = 5 + i
        r = updated("model", ep, preds, sigma_floor=10.0)
        # nodes 0.5..2.5 s (5 of 30) per retained branch, 6 branches
        self.assertEqual(r.floor_stats(), (6 * 5, 6 * 30))
        # CV: sqrt(15^2 + (4t)^2) < 30  <=>  t < 6.495 s -> 12 nodes
        r = updated("cv", ep, sigma_floor=30.0)
        self.assertEqual(r.floor_stats(), (3 * 12, 3 * 30))
        self.assertEqual(updated("none", ep).floor_stats(), (0, 0))
        # branch sigma below the floor at 5/10/15 s (cols 9/19/29 = 14/24/34)
        bf = branch_floor_counts(preds, "rms", 0.5, 0.05, 10.0)
        self.assertEqual(bf, {5: [6, 6], 10: [0, 6], 15: [0, 6]})
        # pi_min 0.25 keeps branches 0.5 and 0.3 of both tracks
        bf = branch_floor_counts(preds, "rms", 1.0, 0.25, 15.0)
        self.assertEqual(bf, {5: [4, 4], 10: [0, 4], 15: [0, 4]})


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        n = 401                                           # 0..20 s at 20 Hz
        tt = np.round(np.arange(n) * 0.05, 10)
        s0 = np.minimum(100.0 + 60.0 * np.abs(tt - 6.0),
                        40.0 + 80.0 * np.abs(tt - 12.5))
        s1 = 120.0 + 50.0 * np.abs(tt - 7.5)
        sep = np.stack([s0, s1], axis=1)
        lat = np.zeros((n, 2), bool)
        lat[20, 1] = True                                 # agent 1 at t=1.0
        lat[300, 0] = True                                # agent 0 at t=15
        mode = np.zeros((n, 2), np.int16)
        mode[tt >= 11.0, 0] = 2                           # CT_RIGHT
        self.arr = {"t": tt, "sep": sep, "lat": lat, "mode": mode}
        # 2 Hz track book: frames at 0.5 .. 20.0
        ft = np.arange(1, 41) * 0.5
        tid = np.full((40, 2), -1, np.int32)
        det = np.zeros((40, 2), bool)
        gm = np.zeros((40, 2), np.int8)
        for i, t in enumerate(ft):
            if 1.0 <= t <= 4.0:
                tid[i, 0], det[i, 0] = 3, True
            if t >= 6.0:
                tid[i, 0], det[i, 0] = 9, True
            if t >= 2.0:
                tid[i, 1], det[i, 1] = 4, True
        gm[ft == 6.0, 0] = 1                              # CT_LEFT at start
        det[ft == 7.0, 0] = False                         # coasting miss
        ep = SimpleNamespace(t=ft.astype(np.float32), obs_track_id=tid,
                             obs_detected=det, gt_mode=gm)
        self.book = TrackBook(ep, 40)
        self.first_det = {0: 1.0, 1: 2.0}
        self.agents = [SimpleNamespace(platform=0, intent_code=1),
                       SimpleNamespace(platform=1, intent_code=3)]
        self.out = safety_endpoints(self.arr, self.book, self.first_det,
                                    self.agents)

    def test_runs_helper(self):
        self.assertEqual(_runs(np.array([0, 1, 1, 0, 1], bool)),
                         [(1, 3), (4, 5)])
        self.assertEqual(_runs(np.array([1, 1], bool)), [(0, 2)])
        self.assertEqual(_runs(np.zeros(3, bool)), [])

    def test_pairwise_vs_union(self):
        o = self.out
        self.assertEqual(o["pairwise_encounters"], 3)
        self.assertEqual(o["union20_events_152"], 2)      # overlap merged
        self.assertEqual(o["union20_events_45"], 1)
        self.assertAlmostEqual(o["min_sep_20hz"], 40.0)
        segs = {(s["slot"], s["t_min"]): s for s in o["pairwise"]}
        a = segs[(0, 6.0)]
        b = segs[(1, 7.5)]
        c = segs[(0, 12.5)]
        self.assertEqual((a["t_entry"], a["t_exit"], a["min_sep"]),
                         (5.15, 6.85, 100.0))
        self.assertEqual((b["t_entry"], b["t_exit"]), (6.9, 8.1))
        self.assertAlmostEqual(c["tb_45_s"], 0.15)       # ticks 12.45..12.55
        self.assertAlmostEqual(c["tb_152_s"], 57 * 0.05)
        # union time below 152.4: the merged run 5.15..8.10 + 11.10..13.90
        self.assertAlmostEqual(o["union20_time_below_152_s"],
                               (60 + 57) * 0.05)
        ev = o["union20_event_list"]["152"][0]
        self.assertEqual((ev["t_entry"], ev["t_exit"], ev["slot"]),
                         (5.15, 8.1, 0))

    def test_screens_and_flags(self):
        segs = {(s["slot"], s["t_min"]): s for s in self.out["pairwise"]}
        a, b, c = segs[(0, 6.0)], segs[(1, 7.5)], segs[(0, 12.5)]
        # a: no open track at entry 5.15 (track 3 closed at 4.0) -> the
        # most recent track (3, first detected 1.0) -> lead 4.15 s
        self.assertEqual(a["track_id"], 3)
        self.assertAlmostEqual(a["lead_track_s"], 4.15)
        self.assertAlmostEqual(a["lead_phys_s"], 5.0)
        # c: reacquired track 9 (first det 6.0) open at entry 11.1:
        # old screen says avoidable (lead_phys 11.5), new screen does not
        self.assertEqual(c["track_id"], 9)
        self.assertAlmostEqual(c["lead_track_s"], 5.1)
        self.assertAlmostEqual(c["lead_phys_s"], 11.5)
        self.assertGreater(c["lead_phys_s"], T_NEED_S)
        self.assertFalse(c["unavoidable_old"])
        self.assertTrue(c["unavoidable_new"])
        self.assertTrue(c["turning_at_track_start"])      # CT_LEFT at 6.0
        self.assertTrue(c["turning_at_entry"])            # CT_RIGHT >= 11
        self.assertFalse(a["turning_at_entry"])
        self.assertFalse(c["reaction_active"])            # latch at 15 s
        self.assertTrue(b["reaction_active"])             # latch at 1.0 s
        self.assertEqual((b["gt_type"], b["gt_intent"]), (1, 3))
        self.assertAlmostEqual(b["lead_track_s"], 6.9 - 2.0)

    def test_never_tracked_is_unavoidable(self):
        book = TrackBook(SimpleNamespace(
            t=self.book.t, obs_track_id=np.full((40, 2), -1, np.int32),
            obs_detected=np.zeros((40, 2), bool),
            gt_mode=np.zeros((40, 2), np.int8)), 40)
        out = safety_endpoints(self.arr, book, {}, self.agents)
        for s in out["pairwise"]:
            self.assertIsNone(s["lead_track_s"])
            self.assertIsNone(s["lead_phys_s"])
            self.assertIsNone(s["turning_at_track_start"])
            self.assertTrue(s["unavoidable_old"] and s["unavoidable_new"])

    def test_aggregator_endpoints(self):
        row = {"events": [], "events_new_screen": [],
               "union20_event_list": self.out["union20_event_list"],
               "pairwise": self.out["pairwise"], "sim_s": 20.0}
        e = agg.episode_endpoints(row)
        self.assertEqual(e["pairwise_152_all"], 3)
        self.assertEqual(e["pairwise_45_all"], 1)
        self.assertEqual(e["pairwise_152_old"],
                         sum(not s["unavoidable_old"]
                             for s in self.out["pairwise"]))
        self.assertEqual(e["union20_severe_all"], 1)
        st = agg.episode_strata(row)
        for fam in st.values():                       # every stratum present
            self.assertEqual(set(fam), set(agg.STRATUM_KEYS))
        self.assertEqual(st["pairwise_152"]["class=HARD"]["all"], 3)


class StatsTests(unittest.TestCase):
    def test_sign_test_exact(self):
        self.assertAlmostEqual(agg.sign_test(0, 5), 0.0625, places=12)
        self.assertAlmostEqual(agg.sign_test(1, 9), 22 / 1024, places=12)
        self.assertAlmostEqual(agg.sign_test(9, 1), 22 / 1024, places=12)
        self.assertEqual(agg.sign_test(5, 5), 1.0)
        self.assertEqual(agg.sign_test(0, 0), 1.0)
        # exact rational reference at n = 300
        w, l = 130, 170
        n = w + l
        ref = 2 * sum(Fraction(math.comb(n, i), 2 ** n) for i in range(w + 1))
        self.assertAlmostEqual(agg.sign_test(w, l), float(ref), places=10)

    def test_sign_test_large_n(self):
        p = agg.sign_test(1300, 1400)          # n = 2700: no overflow
        self.assertTrue(math.isfinite(p))
        z = (100 - 1) / math.sqrt(2700)        # continuity-corrected normal
        approx = math.erfc(z / math.sqrt(2))
        self.assertAlmostEqual(p, approx, delta=0.003)

    def test_bootstrap(self):
        rng = np.random.default_rng(1)
        n = 200
        ca = rng.poisson(1.0, n).astype(float)
        cb = rng.poisson(1.3, n).astype(float)
        h = np.full(n, 90.0 / 3600.0)
        idx = agg.boot_indices(n, 2000, 20260926)
        np.testing.assert_array_equal(idx, agg.boot_indices(n, 2000, 20260926))
        same = agg.rate_boot(ca, ca, h, h, idx)
        self.assertEqual(same["rate_diff"]["ci"], [0.0, 0.0])
        self.assertEqual(same["rate_ratio"]["ci"], [1.0, 1.0])
        d = agg.rate_boot(ca, cb, h, h, idx)
        lo, hi = d["rate_diff"]["ci"]
        self.assertLessEqual(lo, d["rate_diff"]["est"])
        self.assertGreaterEqual(hi, d["rate_diff"]["est"])
        self.assertAlmostEqual(d["rate_a"], ca.sum() / h.sum(), places=3)
        z = agg.rate_boot(ca, np.zeros(n), h, h, idx)
        self.assertEqual(z["rate_ratio"]["undefined_frac"], 1.0)
        self.assertIsNone(z["rate_ratio"]["est"])
        m = agg.stat_boot(ca, cb, idx, 50)
        self.assertEqual(m["est"],
                         round(float(np.median(ca) - np.median(cb)), 3))
        w = agg.wlt(np.array([1, 2, 3.0]), np.array([2, 2, 1.0]), "low")
        self.assertEqual((w["wins"], w["losses"], w["ties"]), (1, 1, 1))

    def test_same_traffic(self):
        a = {"seed": 1, "n_agents": 5, "preset": "test_ood",
             "overrides": {"density_per_km2": 40.0}}
        self.assertTrue(agg.same_traffic(a, dict(a)))
        self.assertFalse(agg.same_traffic(a, {**a, "preset": "test_id"}))
        self.assertFalse(agg.same_traffic(a, {**a, "seed": 2}))
        self.assertFalse(agg.same_traffic(a, {**a, "overrides": {}}))
        self.assertTrue(agg.same_traffic(a, {"seed": 1, "n_agents": 5}))

    def test_hpd_closed_form_vs_mc(self):
        # K=1 closed form == MC estimator on a 1-component "mixture"
        rng = np.random.default_rng(3)
        P = 4000
        mu = np.zeros((P, 1, 2))
        sig = np.tile(np.array([[[30.0, 10.0]]]), (P, 1, 1))
        rho = np.full((P, 1), 0.4)
        L = np.linalg.cholesky(np.array([[900.0, 120.0], [120.0, 100.0]]))
        gt = rng.standard_normal((P, 2)) @ L.T
        cf = agg.hpd_inside(mu, sig, rho, np.ones((P, 1)), gt, rng, 256)
        mu2 = np.concatenate([mu, mu], axis=1)
        mc = agg.hpd_inside(mu2, np.concatenate([sig, sig], 1),
                            np.concatenate([rho, rho], 1),
                            np.full((P, 2), 0.5), gt, rng, 256)
        for p in agg.LEVELS:
            self.assertAlmostEqual(cf[p].mean(), p, delta=0.025)
            self.assertAlmostEqual(mc[p].mean(), p, delta=0.03)


def _pred_log_inputs(meta_extra=None):
    """Synthetic save_pred_log inputs: 60 logged frames (0.5..30 s), two
    intruders; slot 0's reaction layer is latched only at t = 10 s, slot 1
    is a REACTIVE_AVOID agent (intent 4). Three forecasts."""
    T, N, K = 60, 2, 3
    ep = SimpleNamespace(
        gt_pos=np.stack([np.stack([np.arange(T) * 1.0,
                                   np.full(T, 5.0 * a)], -1)
                         for a in range(N)], 1).astype(np.float32),
        gt_valid=np.ones((T, N), bool),
        gt_type=np.array([1, 0], np.int8),
        gt_intent=np.tile(np.array([[0, 4]], np.int8), (T, 1)))
    tt = np.round(np.arange(0, 601) * 0.05, 10)
    lat = np.zeros((601, N), bool)
    lat[200, 0] = True                                   # t = 10.0 s
    mk = lambda t0, t, slot: {                          # noqa: E731
        "t0": t0, "t": t, "slot": slot,
        "mu": np.zeros((K, 30, 2)), "sigma": np.ones((K, 30, 2)),
        "rho": np.zeros((K, 30)), "pi": np.full(K, 1.0 / K), "theta": 0.0,
        "own_intervening": True}
    entries = [mk(3, 2.0, 0), mk(39, 20.0, 0), mk(3, 2.0, 1)]
    own_t = np.arange(1, T + 1) * 0.5
    own = {"t": own_t, "x": own_t * 50.0, "y": np.zeros(T),
           "d": np.zeros(T), "d_nominal": np.zeros(T),
           "v": np.full(T, 50.0), "psi": np.zeros(T)}
    ref = {"x": np.array([0.0, 5000.0]), "y": np.array([0.0, 0.0]),
           "v_desired": 50.0, "corridor_width": 400.0, "lateral_step": 45.0,
           "hard_radius": 30.0, "ref_kind": "corridor",
           "own_track": (0.0, 0.0, 50.0, 0.0), "own_d_t0": 0.0}
    meta = {"arm": "cv", "index": 0, "preset": "test_ood",
            "track_rule": "current",
            "meta_json": json.dumps({"hard_geom": "arm", "code_fp": "x",
                                     **(meta_extra or {})}, sort_keys=True)}
    return entries, K, ep, T, {"t": tt, "lat": lat}, meta, own, ref


class PredLogTests(unittest.TestCase):
    def test_disjoint_shards_never_collide(self):
        arms = ("cv", "model", "mc_cvsig", "cvc_msig")
        shards = [range(0, 150), range(150, 300), range(300, 600),
                  range(99_990, 100_010)]
        names = {}
        for arm in arms:
            for sh in shards:
                for i in sh:
                    p = pred_log_path("D", arm, i)
                    self.assertNotIn(p, names, (arm, i, names.get(p)))
                    names[p] = (arm, i)
        self.assertEqual(os.path.basename(pred_log_path("D", "cv", 7)),
                         "cv_00007.npz")
        self.assertEqual(os.path.basename(pred_log_path("D", "cv", 123456)),
                         "cv_123456.npz")

    def test_arm_file_matching_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ("cv_00000.npz", "cv_00001.npz", "cvc_msig_00000.npz",
                      "cv_00002.tmp", "cvx_00000.npz"):
                open(os.path.join(d, f), "wb").close()
            got = [os.path.basename(p) for p in agg.pred_files(d, "cv")]
            self.assertEqual(got, ["cv_00000.npz", "cv_00001.npz"])
            self.assertEqual([os.path.basename(p) for p in
                              agg.pred_files(d, "cvc_msig")],
                             ["cvc_msig_00000.npz"])

    def test_fields_and_exclusive_create(self):
        entries, K, ep, T, arr, meta, own, ref = _pred_log_inputs()
        with tempfile.TemporaryDirectory() as d:
            p = pred_log_path(d, "cv", 0)
            self.assertEqual(save_pred_log(p, entries, K, "gaussian", ep, T,
                                           arr, meta, own, ref), 3)
            with self.assertRaises(FileExistsError):
                save_pred_log(p, entries, K, "gaussian", ep, T, arr, meta,
                              own, ref)
            save_pred_log(p, entries, K, "gaussian", ep, T, arr, meta, own,
                          ref, overwrite=True)
            with np.load(p) as z:
                self.assertNotIn("intruder_reacting", z.files)
                np.testing.assert_array_equal(z["reaction_latched"],
                                              [True, False, False])
                np.testing.assert_array_equal(z["intent_reactive"],
                                              [False, False, True])
                np.testing.assert_array_equal(z["gt_type"], [1, 1, 0])
                np.testing.assert_array_equal(z["gt_intent"], [0, 0, 4])
                np.testing.assert_array_equal(z["own_intervening"],
                                              [True, True, True])
                # forecast at t0=39 (20 s): GT rows 40..59 exist, 60.. do not
                self.assertEqual(int(z["gt_valid"][1].sum()), 20)
                self.assertEqual(z["own_t"].shape, (T,))
                for k in ("own_x", "own_y", "own_d", "own_d_nominal",
                          "own_v", "own_psi"):
                    self.assertEqual(z[k].shape, (T,), k)
                self.assertEqual(z["own_track"].shape, (4,))
                self.assertEqual(float(z["v_desired"]), 50.0)
                self.assertEqual(str(z["ref_kind"]), "corridor")
                self.assertEqual(json.loads(str(z["meta_json"]))["hard_geom"],
                                 "arm")

    def test_forecast_strata_parameters(self):
        own_t = np.arange(1, 61) * 0.5                      # 0.5 .. 30 s
        own_d = np.zeros(60)
        own_d[own_t == 12.0] = -40.0                        # |d| 40 at 12 s
        own_v = np.full(60, 50.0)
        own_v[own_t == 25.0] = 40.0                         # 0.8 v_des at 25 s
        z = {"t_s": np.array([2.0, 20.0, 26.0, 12.0]), "own_t": own_t,
             "own_d": own_d, "own_v": own_v, "v_desired": 50.0,
             "reaction_latched": np.array([1, 0, 0, 0], bool),
             "intent_reactive": np.array([0, 0, 1, 0], bool)}
        s = agg.forecast_strata(z, 30.0, 0.90)
        np.testing.assert_array_equal(s["dev_crit"], [1, 0, 0, 1])
        np.testing.assert_array_equal(s["speed_crit"], [0, 1, 0, 1])
        np.testing.assert_array_equal(s["intervening"], [1, 1, 0, 1])
        np.testing.assert_array_equal(s["reacting"], [1, 0, 1, 0])
        # 26 s: the window is truncated at the episode end (26..30 s)
        np.testing.assert_array_equal(s["window_frames"], [31, 21, 9, 31])
        # the definitions are parameters
        s = agg.forecast_strata(z, 50.0, 0.70)
        self.assertFalse(s["intervening"].any())
        s = agg.forecast_strata(z, 0.0, 0.90)               # d > 0 somewhere
        np.testing.assert_array_equal(s["dev_crit"], [1, 0, 0, 1])
        # a constant placement offset: 'reference' flags every forecast,
        # 'nominal' (offset from the own track) and 'at_t' (change during
        # the horizon) do not
        z2 = dict(z, own_d=own_d + 150.0, own_d_nominal=own_d)
        self.assertTrue(agg.forecast_strata(z2, 30.0, 0.5)["dev_crit"].all())
        np.testing.assert_array_equal(
            agg.forecast_strata(z2, 30.0, 0.5, own_ref="nominal")["dev_crit"],
            [1, 0, 0, 1])
        a = agg.forecast_strata(z2, 30.0, 0.90, own_ref="at_t")
        # at_t: |d(tau)-d(t)| = 40 only if 12 s is in [t, t+15] and d(t)=0;
        # the forecast AT 12 s has d(t) = -40 itself -> max change 40 too
        np.testing.assert_array_equal(a["dev_crit"], [1, 0, 0, 1])
        # speed change 10 m/s > 0.1 * 50 inside [20,35] and [12,27]
        np.testing.assert_array_equal(a["speed_crit"], [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            agg.forecast_strata(z, 30.0, 0.9, own_ref="centre")

    def test_counts_conditions_and_legacy(self):
        entries, K, ep, T, arr, meta, own, ref = _pred_log_inputs()
        with tempfile.TemporaryDirectory() as d:
            for i in (0, 1):
                save_pred_log(pred_log_path(d, "cv", i), entries, K,
                              "gaussian", ep, T, arr, meta, own, ref)
            save_pred_log(pred_log_path(d, "cvc_msig", 0), entries, K,
                          "gaussian", ep, T, arr, meta, own, ref)
            c = agg.pred_dir_counts(d, "cv", 30.0, 0.90)
            self.assertEqual((c["files"], c["forecasts"]), (2, 6))
            self.assertEqual(c["strata_source"], "raw")
            f = c["strata_counts"]["forecasts"]
            # own never deviates -> interv0; slot0@2s latched, slot1 intent 4
            self.assertEqual(f, {"all": 6, "interv0_react0": 2,
                                 "interv0_react1": 4, "interv1_react0": 0,
                                 "interv1_react1": 0})
            comp = c["strata_counts"]["components"]
            self.assertEqual((comp["reaction_latched"],
                              comp["intent_reactive"],
                              comp["own_intervening_at_t"]), (2, 2, 6))
            self.assertEqual(c["strata_counts"]["valid_points"]["all"]["15s"],
                             2 * (1 + 0 + 1))    # the 20 s forecast ends early
            self.assertEqual(len(c["conditions"]), 1)
            # a second condition of the same arm in the same dir is refused
            _, _, _, _, _, meta2, _, _ = _pred_log_inputs(
                {"hard_geom": "shared"})
            save_pred_log(pred_log_path(d, "cv", 2), entries, K, "gaussian",
                          ep, T, arr, meta2, own, ref)
            with self.assertRaises(SystemExit):
                agg.pred_dir_counts(d, "cv", 30.0, 0.90)
        with tempfile.TemporaryDirectory() as d:        # pre-change format
            P = 2
            np.savez(os.path.join(d, "cv_00000.npz"),
                     t0=np.zeros(P, np.int32), t_s=np.zeros(P, np.float32),
                     slot=np.zeros(P, np.int32),
                     mu=np.zeros((P, 1, 30, 2), np.float32),
                     sigma=np.ones((P, 1, 30, 2), np.float32),
                     rho=np.zeros((P, 1, 30), np.float32),
                     pi=np.ones((P, 1), np.float32),
                     theta=np.zeros(P, np.float32),
                     gt=np.zeros((P, 30, 2), np.float32),
                     gt_valid=np.ones((P, 30), bool),
                     own_intervening=np.array([True, False]),
                     intruder_reacting=np.array([True, True]),
                     kind=np.array("gaussian"))
            c = agg.pred_dir_counts(d, "cv", 30.0, 0.90)
            self.assertEqual(c["strata_source"], "legacy_stored_flags")
            self.assertEqual(c["strata_counts"]["forecasts"]["interv1_react1"],
                             1)
            cov = agg.pred_dir_coverage(d, "cv", 30.0, 0.90, 64, 1)
            self.assertEqual(cov["all"]["5s"]["n"], 2)
            self.assertEqual(cov["all"]["5s"]["cov90"], 1.0)   # GT at mean


if __name__ == "__main__":
    prog = unittest.main(exit=False, verbosity=2)
    sys.exit(0 if prog.result.wasSuccessful() else 1)
