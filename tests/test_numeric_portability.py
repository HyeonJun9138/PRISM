"""Duration ties and floating-statistic verification across Python versions."""
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_round2', ROOT/'tools/verify_round2.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
A = V.AGG


def row(parts):
    return {'index': 585, 'exposure_s': 90, 'nmac_duration_s': 0,
            'pairwise': [{'tb_152_s': x, 'tb_76_s': 0, 'tb_45_s': 0,
                          'min_sep': 100, 'min_sep_raw_m': 100,
                          'unavoidable_old': False, 'unavoidable_new': False} for x in parts],
            **{f'union20_time_below_{t}_s': 0 for t in ('152','76','45')}}


class DurationTies(unittest.TestCase):
    def test_weighted_penetration_mean_has_portable_three_decimal_rounding(self):
        # Released r3_wsafe24/cv nonempty penetration summaries and cycle weights.
        pens = [(17.27,4),(14.19,5),(10.33,3),(17.08,4),(13.27,4),(6.54,1),
                (3.16,2),(9.77,3),(10.57,6),(2.49,2),(13.2,4),(8.94,4),
                (20.54,5),(6.52,1),(6.86,1),(12.89,6),(11.5,4),(4.28,2),
                (17.08,6),(1.96,1),(9.06,1),(16.1,5),(13.93,4),(7.14,1),
                (8.44,15),(.04,1),(16.87,5),(.92,2),(5.93,4),(16.14,2),
                (13.48,4),(3.86,2),(8.01,1),(16.12,5),(2.85,4),(3.58,2),
                (3.88,3),(2.36,3),(7.43,2),(4.5,1),(16.65,8),(.03,1),
                (8.39,3),(4.1,2),(17.27,5),(1.3,2),(16.97,5),(21.12,5),
                (4.8,4),(12.82,4),(.11,1),(19.77,5)]
        rows={i:{**row([]),'n_cycles':w,'emergency_cycles':w,
                 'emerg_collision_static':w,'emerg_collision_forecast':0,
                 'pen_depth_mean_m':p,'pen_depth_max_m':p,
                 'feasible_frac_mean':0,'feasible_frac_p10':0}
              for i,(p,w) in enumerate(pens)}
        self.assertEqual(A.summarize('cv_wsafe24',rows)['planner']['pen_depth_mean_m'],11.598)

    def test_episode_585_is_exactly_326_ticks_in_any_order(self):
        for parts in itertools.permutations([5.2, 6.25, 4.85]):
            self.assertEqual(A.episode_time_ticks(row(parts))['pairwise_tb_152'], 326)
        self.assertEqual(A.episode_time_ticks(row([5.2, 5.2, 5.9]))['pairwise_tb_152'], 326)

    def test_equal_tick_durations_are_ties_in_paired_and_unrounded_outputs(self):
        ra = {585: row([5.2, 6.25, 4.85])}
        rb = {585: row([5.2, 5.2, 5.9])}
        got = A.compare(ra, rb, 16, 20260926)
        V.add_unrounded(got, ra, rb, 16)
        d = got['time_below']['pairwise_tb_152']
        self.assertEqual((d['wins'], d['losses'], d['ties']), (0, 0, 1))
        self.assertEqual(d['unrounded']['sign_p'], 1.0)
        self.assertEqual(d['rate_diff']['est'], 0)

    def test_canonical_seconds_fix_native_sum_false_tie(self):
        # 2.9 + 2.9 and 1.1 + 4.7 represent the same 116 ticks.
        self.assertEqual(A.episode_times(row([2.9,2.9]))['pairwise_tb_152'],
                         A.episode_times(row([1.1,4.7]))['pairwise_tb_152'])

    def test_one_tick_difference_is_not_tied(self):
        got = A.compare({585:row([16.3])}, {585:row([16.35])}, 16, 20260926)
        d = got['time_below']['pairwise_tb_152']
        self.assertEqual((d['wins'],d['losses'],d['ties']), (1,0,0))

    def test_off_grid_negative_nonfinite_and_boolean_durations_rejected(self):
        for bad in (0.051, -0.05, float('nan'), float('inf'), True):
            with self.subTest(bad=bad), self.assertRaises((ValueError,TypeError)):
                A.episode_times(row([bad]))


class NumericVerification(unittest.TestCase):
    def test_cli_rejects_modified_row_bytes_before_statistics(self):
        manifest = json.loads((ROOT/'results/round2/numbers_manifest_round2.json').read_text(encoding='utf-8'))
        run = next(iter(manifest['coverage']['inputs']))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'round2/rescored'/(run+'.jsonl')
            path.parent.mkdir(parents=True)
            path.write_text('{}\n',encoding='utf-8')
            result = subprocess.run([sys.executable,str(ROOT/'tools/verify_round2.py'),'--root',td],
                                    capture_output=True,text=True,encoding='utf-8')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Row checksum mismatch',result.stderr)

    def test_roundoff_in_float_statistics_accepted(self):
        V.assert_statistics_match({'sign_p':.3017578125000002}, {'sign_p':.3017578125000003})
        V.assert_statistics_match({'rate':0.0}, {'rate':5e-13})
        V.assert_statistics_match({'rate':1e6}, {'rate':1e6+5e-7})

    def test_counts_indices_hashes_bool_and_structure_remain_exact(self):
        pairs = [({'wins':169},{'wins':170}), ({'indices':[0,599]},{'indices':[0,598]}),
                 ({'sha256':'a'*64},{'sha256':'b'*64}), ({'ready':True},{'ready':1}),
                 ({'n':300},{'n':300.0}), ({'x':None},{'x':0.0}), ({'x':[]},{'x':[1]}),
                 ({'x':1},{'x':1,'y':2}),
                 ({'events':{'severe':{'total_a':1.0}}},{'events':{'severe':{'total_a':1.0+1e-13}}})]
        for got, want in pairs:
            with self.subTest(got=got), self.assertRaises(ValueError):
                V.assert_statistics_match(got,want)

    def test_materially_wrong_or_nonfinite_statistics_fail(self):
        for got,want in ((.30175,.30176),(0.,1e-8),(float('nan'),float('nan')),
                         (float('inf'),float('inf'))):
            with self.subTest(got=got), self.assertRaises(ValueError):
                V.assert_statistics_match({'sign_p':got},{'sign_p':want})

if __name__ == '__main__':
    unittest.main()
