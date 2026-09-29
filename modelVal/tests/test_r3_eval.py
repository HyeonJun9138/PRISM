"""Standalone checks for the review-round-1 R3 evaluation fixes (no pytest).

    cd $PRISM_ROOT/modelVal
    python tests/test_r3_eval.py

1. rollout scoring aligns model and branch futures by TIMESTAMP
   (experiments.rollout_score.aligned_t0 / check_step_alignment), and refuses
   to guess when the fork time is not a logged timestamp (R3 major 1);
2. the recalibration evaluation set is held out: the seed-overlap guard
   raises on overlap, and the default set (seed >= 300300) shares no seed
   with the calibration runs while the released set shares 300 (R3 major 2);
3. replaying one dataset episode with dataGen's engine reproduces the logged
   gt_pos exactly, reproduces its own labels, and a tampered log is rejected
   (experiments.reaction_labels, R3 major 7);
4. the held-out rollout generator (experiments.gen_rollout_heldout): its
   default seeds are inside the preset block and in no existing run (and an
   overlapping range is caught), its bundle check rejects a tampered bundle,
   and its generation path regenerates a released main.py rollout episode
   (test_id rollout_eval episode 0: episode, 60-branch bundle, index row) bit
   for bit (~25 s).

Tests that need the released datasets are skipped when those are absent.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import gen_rollout_heldout as gh  # noqa: E402
from experiments import reaction_labels as rl  # noqa: E402
from experiments import recalibrate as rc  # noqa: E402
from experiments import rollout_score as rs  # noqa: E402

#: Data/artifact root. Override with the PRISM_ROOT environment variable;
#: defaults to the repository root so a fresh clone works out of the box.
PRISM_ROOT = os.environ.get("PRISM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/")
DATASET = os.environ.get("PRISM_DATASET_ROOT", f"{PRISM_ROOT}/Logs/Dataset")
ROLLOUT_ID = os.path.join(DATASET, "20260814_101540_test_id_rollout_eval")
TEST_ID = os.path.join(DATASET, "20260813_023139_test_id_full5x")


def _logged_t(n=120, offset=0.0):
    return ((np.arange(n) + 1) * 0.5 + offset).astype(np.float32)


class TimestampAlignment(unittest.TestCase):
    def test_fork_index_is_the_logged_fork_time(self):
        t = _logged_t()
        t0 = rs.aligned_t0(t, 60)            # dataGen: origin_step = 60
        self.assertEqual(t0, 59)
        self.assertEqual(float(t[t0]), 30.0)
        # model step k and rollout step k share t = 30.5 + 0.5 k
        rs.check_step_alignment(t, t0, 30.0, [0, 9, 19, 29])

    def test_legacy_index_is_one_step_late(self):
        t = _logged_t()
        self.assertEqual(float(t[60]) - 60 * 0.5, 0.5)
        with self.assertRaises(AssertionError):
            rs.check_step_alignment(t, 60, 30.0, [0])

    def test_mismatch_raises(self):
        with self.assertRaises(AssertionError):          # no such timestamp
            rs.aligned_t0(_logged_t(offset=0.25), 60)
        with self.assertRaises(AssertionError):          # ambiguous
            rs.aligned_t0(np.array([29.5, 30.0, 30.0, 30.5], np.float32), 60)
        with self.assertRaises(AssertionError):          # fork beyond the log
            rs.aligned_t0(_logged_t(n=40), 60)

    @unittest.skipUnless(os.path.isdir(ROLLOUT_ID), "rollout run not present")
    def test_branches_start_one_step_after_the_fork(self):
        ep = np.load(os.path.join(ROLLOUT_ID, "episodes", "ep_000000.npz"))
        ro = np.load(os.path.join(ROLLOUT_ID, "rollouts",
                                  "ep_000000_rollouts.npz"))
        t0 = rs.aligned_t0(ep["t"], int(ro["origin_step"][0]))
        self.assertEqual(t0, 59)
        gt, pos, val = ep["gt_pos"], ro["pos"], ro["valid"]
        d_aligned, d_legacy = [], []
        for g, a in enumerate(ro["agent_slot"]):
            ok = val[g, :, 0]
            if not ok.any() or not np.isfinite(gt[t0 + 2, a]).all():
                continue
            p = pos[g, ok, 0]                       # rollout step 0 (30.5 s)
            d_aligned.append(np.min(np.linalg.norm(p - gt[t0 + 1, a], axis=1)))
            d_legacy.append(np.min(np.linalg.norm(p - gt[t0 + 2, a], axis=1)))
        # 0.5 s after the fork the logged continuation is (nearly always)
        # one of the branches, one step later it is not
        self.assertLess(np.median(d_aligned), 1e-3)
        self.assertGreater(np.median(d_legacy), 1.0)


class SeedOverlap(unittest.TestCase):
    def test_guard(self):
        self.assertEqual(rc.seed_overlap([1, 2, 3], [4, 5]), [])
        self.assertEqual(rc.seed_overlap([1, 2, 3], [3, 4]), [3])
        rc.assert_no_seed_overlap([1, 2], [3, 4])
        with self.assertRaises(AssertionError):
            rc.assert_no_seed_overlap([1, 2], [2, 3], "unit")

    @unittest.skipUnless(all(os.path.isdir(r) for r in rc.TEST_RUNS)
                         and len(rc.CAL_RUNS) == 3, "OOD runs not present")
    def test_default_eval_set_is_held_out(self):
        cal = [e for r in rc.CAL_RUNS for e in rc.run_episodes(r)]
        test = [e for r in rc.TEST_RUNS for e in rc.run_episodes(r)]
        cal_s = [e["seed"] for e in cal]
        held = rc.select_eval(test, legacy=False)
        self.assertEqual(len(cal), 300)
        self.assertEqual(len(held), 900)
        self.assertEqual(min(e["episode_index"] for e in held), 300)
        self.assertEqual(rc.seed_overlap(cal_s, [e["seed"] for e in held]), [])
        legacy = rc.select_eval(test, legacy=True)
        self.assertEqual(len(legacy), 1200)
        ov = rc.seed_overlap(cal_s, [e["seed"] for e in legacy])
        self.assertEqual((len(ov), ov[0], ov[-1]), (300, 300000, 300299))
        with self.assertRaises(AssertionError):
            rc.assert_no_seed_overlap(cal_s, [e["seed"] for e in legacy])


@unittest.skipUnless(os.path.isdir(TEST_ID), "test_id_full5x not present")
class ReplayDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = rl.setup_datagen(None)
        with open(os.path.join(TEST_ID, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        cls.kind, cls.name = rl.config_kind(man["preset"])
        cls.row = rl.read_index(TEST_ID)[0]
        idx = int(cls.row["episode_index"])
        cls.task = {"datagen_root": cls.root, "kind": cls.kind,
                    "name": cls.name, "episode_index": idx,
                    "seed": int(cls.row["seed"]),
                    "reaction_events": cls.row["reaction_events"],
                    "duration_s": man["timing"]["duration_s"],
                    "episode_path": os.path.join(TEST_ID, "episodes",
                                                 f"ep_{idx:06d}.npz"),
                    "out_path": None, "return_arrays": True}

    def test_replay_reproduces_log_and_labels(self):
        a = rl.replay_episode(dict(self.task))
        self.assertTrue(a["ok"], a.get("reason"))
        self.assertEqual(a["gt_pos_max_abs_diff"], 0.0)
        self.assertEqual(a["gt_pos_nan_pattern_diff"], 0)
        self.assertEqual(a["arms"], int(self.row["reaction_events"]))
        self.assertGreater(int(a["arrays"]["reaction_active"].sum()), 0)
        b = rl.replay_episode(dict(self.task))
        for k in ("reaction_active", "reaction_armed", "reaction_phase",
                  "reaction_threat", "t"):
            np.testing.assert_array_equal(a["arrays"][k], b["arrays"][k])
        # latched at the logged instant => active in that interval; DEBRIS
        # (type 5) carries no reaction layer and can never be active
        act, ph = a["arrays"]["reaction_active"], a["arrays"]["reaction_phase"]
        self.assertTrue(np.all(act[ph == 2]))
        self.assertFalse(act[:, a["arrays"]["gt_type"] == 5].any())

    def test_tampered_log_is_rejected(self):
        tmp = tempfile.mkdtemp()
        try:
            with np.load(self.task["episode_path"]) as z:
                arrs = {k: z[k] for k in z.files}
            g = arrs["gt_pos"].copy()
            i, n = np.argwhere(np.isfinite(g[:, :, 0]))[len(g) // 2]
            g[i, n, 0] += np.float32(0.001)
            arrs["gt_pos"] = g
            bad = os.path.join(tmp, "ep_bad.npz")
            np.savez_compressed(bad, **arrs)
            res = rl.replay_episode({**self.task, "episode_path": bad,
                                     "return_arrays": False})
            self.assertFalse(res["ok"])
            self.assertIn("gt_pos", res["reason"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(os.path.isdir(DATASET), "dataset root not present")
class HeldoutRolloutGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = rl.setup_datagen(None)

    def test_default_seed_range_is_fresh(self):
        # The release includes the already-generated held-out run. Check
        # freshness against the older corpus, then separately ensure that
        # the normal guard refuses to generate those released seeds again.
        released = "20260926_155613_test_ood_rollout_heldout"
        c = gh.seed_check("test_ood", gh.DEFAULT_START, gh.DEFAULT_EPISODES,
                          [DATASET], ignore=(released,))
        self.assertTrue(c["ok"], c)
        self.assertTrue(c["inside_block"])
        self.assertEqual(c["seed_range"], [301200, 301299])
        self.assertEqual(c["overlapping_runs"], {})
        self.assertGreater(c["n_runs_scanned"], 0)
        if os.path.isdir(os.path.join(DATASET, released)):
            reused = gh.seed_check("test_ood", gh.DEFAULT_START,
                                   gh.DEFAULT_EPISODES, [DATASET])
            self.assertFalse(reused["ok"])
            self.assertIn(released, reused["overlapping_runs"])
        # the calibration / evaluation episodes 0-1199 are caught
        bad = gh.seed_check("test_ood", 250, 100, [DATASET])
        self.assertFalse(bad["ok"])
        self.assertTrue(any("test_ood_cal" in r for r in bad["overlapping_runs"]))
        self.assertTrue(any("test_ood_full5x_a" in r for r in bad["overlapping_runs"]))

    @unittest.skipUnless(os.path.isdir(ROLLOUT_ID), "rollout run not present")
    def test_bundle_check_rejects_tampering(self):
        with np.load(os.path.join(ROLLOUT_ID, "rollouts",
                                  "ep_000000_rollouts.npz")) as z:
            r = {k: z[k] for k in z.files}
        seed = int(rl.read_index(ROLLOUT_ID)[0]["seed"])
        n = r["pos"].shape[0]
        self.assertEqual(gh.check_bundle(r, n, seed, 60, 30.0), [])
        bad = dict(r, valid=~r["valid"])
        self.assertTrue(gh.check_bundle(bad, n, seed, 60, 30.0))
        bad = dict(r, origin_step=r["origin_step"] + 1)
        self.assertTrue(gh.check_bundle(bad, n, seed, 60, 30.0))
        self.assertTrue(gh.check_bundle(r, n, seed + 1, 60, 30.0))

    @unittest.skipUnless(os.path.isdir(ROLLOUT_ID), "rollout run not present")
    def test_generation_path_reproduces_released_rollout_episode(self):
        rep = gh.reproduce_reference(ROLLOUT_ID, 0, self.root)
        self.assertEqual(rep["episode_keys_differing"], [])
        self.assertEqual(rep["rollout_keys_differing"], [])
        self.assertTrue(rep["index_row_identical"])
        self.assertEqual((rep["rollouts"], rep["branch_t"]), (60, 30.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
