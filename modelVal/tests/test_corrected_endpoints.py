"""Standalone regression tests for corrected endpoint screening, timing and precision."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.batch import TrackBook, safety_endpoints, T_NEED_S
from experiments import aggregate_r3 as agg

def book(t, ids, detected=None):
    ids = np.asarray(ids, dtype=np.int32).reshape(len(t), -1)
    det = ids >= 0 if detected is None else np.asarray(detected, bool).reshape(ids.shape)
    return TrackBook(SimpleNamespace(t=np.asarray(t, float), obs_track_id=ids,
                                    obs_detected=det, gt_mode=np.zeros_like(ids)), len(t))

def score(t, sep, b=None):
    sep = np.asarray(sep, float).reshape(len(t), -1)
    if b is None:
        b = book([0], [[1] * sep.shape[1]])
    return safety_endpoints({'t': np.asarray(t, float), 'sep': sep,
                            'lat': np.zeros_like(sep, bool), 'mode': np.zeros_like(sep, int)},
                           b, {i: 0 for i in range(sep.shape[1])},
                           [SimpleNamespace(platform=0, intent_code=0) for _ in range(sep.shape[1])])

class ScreenTests(unittest.TestCase):
    def test_closed_track_at_entry_is_ineligible(self):
        b = book([0, .5, 1, 1.5, 10], [7, 7, 7, -1, -1], [1, 0, 0, 0, 0])
        self.assertIsNone(b.track_at_entry(0, 10))
        self.assertIsNone(b.track_at_entry(0, 1.5))
        e = score([10, 10.05], [30, 200], b)['pairwise'][0]
        self.assertTrue(e['unavoidable_new'])
        self.assertIn('screen_audit', e)
        self.assertEqual(e['screen_audit']['exclusion_reason'], 'closed_asof_track')

    def test_open_track_with_missed_current_frame_remains_open(self):
        b = book([0, .5, 10], [7, 7, 7], [1, 0, 0])
        self.assertEqual(b.track_at_entry(0, 10), (7, 0, 0))
        e = score([10, 10.05], [30, 200], b)['pairwise'][0]
        self.assertFalse(e['unavoidable_new'])

    def test_reacquired_track_uses_only_new_first_detection(self):
        b = book([0, 1, 1.5, 2, 10], [7, -1, 9, 9, 9], [1, 0, 1, 0, 0])
        self.assertEqual(b.track_at_entry(0, 1.5), (9, 2, 1.5))
        self.assertEqual(b.track_at_entry(0, 10), (9, 2, 1.5))
        self.assertIsNone(b.track_at_entry(0, 1))

    def test_entry_before_first_logged_frame(self):
        self.assertIsNone(book([.5], [1]).track_at_entry(0, .49))

    def test_entry_between_frames_uses_last_available_frame(self):
        b = book([.5, 1], [-1, 1])
        self.assertIsNone(b.track_at_entry(0, .75))
        self.assertIsNone(b.track_at_entry(0, 1 - 5e-7))

    def test_future_detection_cannot_change_past_screen(self):
        b = book([0, .5, 1.5], [8, 8, 8], [0, 0, 1])
        self.assertIsNone(b.track_at_entry(0, .5))
        e = score([.5, .55], [30, 200], b)['pairwise'][0]
        self.assertTrue(e['unavoidable_new'])
        self.assertIn('screen_audit', e)
        self.assertEqual(e['screen_audit']['exclusion_reason'], 'open_track_without_past_detection')

    def test_entry_at_final_frame(self):
        self.assertEqual(book([0, .5], [1, 1]).track_at_entry(0, .5), (1, 0, 0))

    def test_lead_time_boundary_uses_unrounded_value(self):
        for lead, avoid in [(T_NEED_S - 1e-5, True), (T_NEED_S, False),
                            (T_NEED_S + 1e-5, False), (8.231, False)]:
            o = score([lead, lead + .05], [30, 200])
            for e in [o['pairwise'][0], o['union20_event_list']['152'][0],
                      o['union20_event_list']['45'][0]]:
                self.assertEqual(e['unavoidable_new'], avoid)

    def test_responsible_slot_uses_its_152_entry_and_first_tie_index(self):
        b = book([0, .5, 1, 2, 10], [[1, -1], [1, -1], [1, -1], [1, 2], [1, 2]])
        o = score([10, 10.05, 10.1, 10.15, 10.2, 10.25],
                  [[100, 200], [100, 200], [100, 100], [180, 30], [180, 100], [200, 200]], b)
        e = o['union20_event_list']['152'][0]
        self.assertEqual(e['slot'], 1)
        self.assertEqual(e['t_entry'], 10)
        self.assertIn('screen_audit', e)
        self.assertEqual(e['screen_audit']['screen_entry_s'], 10.1)
        self.assertAlmostEqual(e['screen_audit']['lead_track_raw_s'], 8.1)
        self.assertEqual(o['union20_event_list']['45'][0]['screen_audit'], e['screen_audit'])
        tie = score([10, 10.05], [[30, 30], [200, 200]], b)
        self.assertEqual(tie['union20_event_list']['152'][0]['slot'], 0)

class TimeTests(unittest.TestCase):
    def test_full_exposure_is_90_not_90_05(self):
        o = score(np.arange(1801) * .05, np.full(1801, 30))
        self.assertEqual(o['union20_time_below_45_s'], 90)
        self.assertEqual(o['pairwise'][0]['tb_45_s'], 90)

    def test_final_endpoint_has_count_but_no_duration(self):
        o = score([0, .05, .1], [200, 200, 30])
        self.assertEqual(o['union20_events_45'], 1)
        self.assertEqual(o['union20_time_below_45_s'], 0)
        self.assertEqual(o['union20_event_list']['45'][0]['time_below_s'], 0)
        self.assertEqual(o['pairwise'][0]['tb_45_s'], 0)

    def test_nonuniform_tick_intervals_use_actual_left_tick_duration(self):
        o = score([2, 2.1, 2.4, 2.9], [30, 200, 30, 30])
        self.assertAlmostEqual(o['union20_time_below_45_s'], .6)
        self.assertAlmostEqual(sum(p['tb_45_s'] for p in o['pairwise']), .6)

    def test_overlapping_pairwise_may_exceed_union_time(self):
        o = score([0, .05, .1], [[30, 30]] * 3)
        self.assertEqual(o['pairwise_encounters'], 2)
        self.assertEqual(o['union20_events_152'], 1)
        self.assertEqual(o['union20_time_below_45_s'], .1)
        self.assertEqual(sum(p['tb_45_s'] for p in o['pairwise']), .2)

    def test_severe152_encounter_differs_from_direct45_runs(self):
        o = score(np.arange(5) * .05, [100, 30, 100, 30, 200])
        e = agg.episode_endpoints(o)
        self.assertEqual(e['pairwise_45_all'], 1)
        self.assertEqual(e['union20_severe_all'], 1)
        self.assertEqual(e['union20_45_all'], 2)

class PrecisionTests(unittest.TestCase):
    def test_raw_44_96_is_severe_even_when_display_is_45(self):
        o = score([10, 10.05], [44.96, 200])
        e = agg.episode_endpoints(o)
        self.assertEqual(o['pairwise'][0]['min_sep'], 45.0)
        self.assertEqual(e['union20_45_all'], 1)
        self.assertEqual(e['pairwise_45_all'], 1)
        self.assertEqual(e['union20_severe_all'], 1)
        self.assertEqual(agg.episode_strata(o)['pairwise_45']['class=HARD']['all'], 1)
        self.assertIn('min_sep_raw_m', o['pairwise'][0])
        self.assertEqual(o['pairwise'][0]['min_sep_raw_m'], 44.96)
        self.assertEqual(o['union20_event_list']['152'][0]['min_sep_raw_m'], 44.96)

    def test_thresholds_use_strict_raw_boundary_not_display(self):
        for lbl, thr in [('45', 45.0), ('76', 76.2), ('152', 152.4)]:
            for delta, expected in [(-.04, 1), (0, 0), (.04, 0)]:
                with self.subTest(threshold=thr, delta=delta):
                    o = score([10, 10.05], [thr + delta, 200])
                    e = agg.episode_endpoints(o)
                    self.assertEqual(e[f'pairwise_{lbl}_all'], expected, (lbl, delta))
                    self.assertEqual(e[f'union20_{lbl}_all'], expected, (lbl, delta))
                    if lbl == '45':
                        self.assertEqual(e['union20_severe_all'], expected, delta)

    def test_legacy_rounded_fallback_is_explicit_and_preserved(self):
        ev = {'min_sep': 45.0, 'unavoidable_old': False, 'unavoidable_new': False,
              'slot': 0, 't_entry': 10, 't_min': 10, 't_exit': 10.05,
              'tb_152_s': .1, 'tb_76_s': .1, 'tb_45_s': 0}
        r = {'pairwise': [ev], 'union20_event_list': {'152': [ev]}, 'sim_s': 89.5}
        self.assertEqual(agg.episode_endpoints(r)['union20_severe_all'], 0)
        s = agg.summarize('legacy', {0: r})
        self.assertIn('endpoint_minimum_provenance', s)
        self.assertEqual(s['endpoint_minimum_provenance']['pairwise_legacy_rounded_events'], 1)
        self.assertEqual(s['endpoint_minimum_provenance']['union152_legacy_rounded_events'], 1)

    def test_precision_metadata_is_separate_from_track_screen_version(self):
        o = score([10, 10.05], [44.96, 200])
        self.assertIn('screen_version', o)
        self.assertEqual(o['screen_version'], 'open_asof_v2')
        self.assertIn('endpoint_precision_version', o)
        self.assertEqual(o['endpoint_precision_version'], 'raw_minimum_v2')
        self.assertFalse(o['pairwise'][0]['unavoidable_new'])
        s = agg.summarize('raw', {0: o})
        self.assertEqual(s['endpoint_minimum_provenance']['pairwise_raw_events'], 1)
        self.assertEqual(s['endpoint_minimum_provenance']['pairwise_legacy_rounded_events'], 0)

class ExposureTests(unittest.TestCase):
    def test_scoring_exposure_fields_match_real_endpoints(self):
        o = score([0, 90], [200, 30])
        for k in ['scoring_start_s', 'scoring_end_s', 'exposure_s']:
            self.assertIn(k, o)
        self.assertEqual((o['scoring_start_s'], o['scoring_end_s'], o['exposure_s']), (0, 90, 90))
        early = score([2, 15], [200, 30])
        self.assertEqual(early['exposure_s'], 13)

    def test_300_and_600_complete_episodes_use_7_5_and_15_hours(self):
        for n, hours in [(300, 7.5), (600, 15)]:
            rows = {i: {'sim_s': 89.5, 'exposure_s': 90} for i in range(n)}
            self.assertEqual(agg.summarize('x', rows)['exposure_hours'], hours)
            self.assertEqual(rows[0]['sim_s'], 89.5)

    def test_legacy_exposure_is_not_silently_promoted_to_corrected(self):
        s = agg.summarize('x', {0: {'sim_s': 89.5}})
        self.assertIn('exposure_provenance', s)
        self.assertEqual(s['exposure_provenance']['legacy_sim_s_fallback_rows'], 1)

    def test_paired_rates_use_actual_exposure_not_last_replan(self):
        r = {'seed': 1, 'n_agents': 1, 'sim_s': 89.5, 'exposure_s': 90,
             'events': [{'min_sep': 30, 'unavoidable': False}]}
        s = agg.compare({0: r}, {0: r}, B=10, seed=1)
        self.assertEqual(s['events']['union2_152_all']['rate_a'], 40)

if __name__ == '__main__':
    unittest.main(verbosity=2)
