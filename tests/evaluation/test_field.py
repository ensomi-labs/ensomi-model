import math
import unittest

import numpy as np

from ensomi_model.evaluation.case import Condition, EvalCase, Scope, Skeleton, Timing, evaluate
from ensomi_model.evaluation.field import (
    clip_domain, domain_cells, field_statistics, kernel_sum, normal_cdf, smooth)
from ensomi_model.evaluation.case import Spans
from ensomi_model.evaluation.harness.chain import (
    FieldOperator, SongNull, chart_events, continuation_cut, window_chart, windows)
from ensomi_model.evaluation.harness.trivial import HeadRate

from ._harness_charts import chart, hold, ln_chart, tap
from .test_harness_trivial import model


class KernelTests(unittest.TestCase):
    def test_normal_cdf(self):
        np.testing.assert_allclose(normal_cdf([0.0, 1.0, -1.959963984540054]), [0.5, 0.8413447460685429, 0.025])

    def test_kernel_sum_is_a_rate_in_seconds(self):
        t = np.arange(0.0, 100000.0, 100.0)  # ten per second
        r = kernel_sum([50000.0], t, np.ones(len(t)), 4000.0)
        self.assertAlmostEqual(float(r[0]), 10.0, places=6)

    def test_smooth_is_a_local_mean_defined_everywhere(self):
        t = np.array([0.0, 1000.0, 2000.0, 500000.0])
        x = np.array([1.0, 1.0, 1.0, 5.0])
        f = smooth([1000.0, 250000.0, 10 ** 7], t, x, 2000.0)
        self.assertAlmostEqual(f[0], 1.0)
        self.assertTrue(np.isfinite(f).all())
        self.assertAlmostEqual(f[2], 5.0)
        two = smooth([1000.0], t, np.stack([x, 2 * x], axis=1), 2000.0)
        np.testing.assert_allclose(two[0], [1.0, 2.0])


class FieldStatisticTests(unittest.TestCase):
    def test_cells_tile_the_domain(self):
        centres, lengths, which = domain_cells(((0.0, 250.0), (1000.0, 1100.0)), 100.0)
        np.testing.assert_allclose(centres, [50.0, 150.0, 225.0, 1050.0])
        np.testing.assert_allclose(lengths, [100.0, 100.0, 50.0, 100.0])
        np.testing.assert_array_equal(which, [0, 0, 0, 1])

    def test_constant_field_and_worst_window(self):
        t = np.arange(0.0, 60000.0, 500.0)
        x = np.ones((len(t), 2))
        st = field_statistics(t, x, ((0.0, 59500.0),), h_ms=2000.0, step_ms=100.0, scan_ms=16000.0)
        np.testing.assert_allclose(st['mean'], [1.0, 1.0])
        np.testing.assert_allclose(st['worst'], [1.0, 1.0])
        self.assertAlmostEqual(st['seconds'], 59.5)
        x[(t >= 20000) & (t < 30000), 0] = 5.0
        st = field_statistics(t, x, ((0.0, 59500.0),), h_ms=2000.0, step_ms=100.0, scan_ms=16000.0)
        self.assertGreater(st['worst'][0], 3.0)
        self.assertAlmostEqual(st['mean'][0], 1 + 4 * 10 / 59.5, delta=0.05)
        self.assertAlmostEqual(st['mean'][1], 1.0)

    def test_empty_domain(self):
        st = field_statistics(np.zeros(0), np.zeros((0, 2)), (), h_ms=2000.0, step_ms=100.0, scan_ms=16000.0)
        self.assertTrue(np.isnan(st['mean']).all())
        self.assertEqual(clip_domain(Spans.everything(), 5.0, 5.0), ())


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.m = model()

    def test_every_head_and_release_receives_a_score(self):
        c = ln_chart()
        op = FieldOperator(HeadRate(), self.m)
        out = op(EvalCase(c, Condition.of(star=3.2), Scope.whole()))
        times, kind, _ = chart_events(c)
        self.assertEqual(len(out['event_time']), len(times))
        self.assertTrue(np.isfinite(out['readout']).all())
        self.assertTrue(out['ranked'].all())
        self.assertTrue((kind == 3).any())
        self.assertTrue(np.isfinite(out['stats']['mean']).all())

    def test_continuation_scores_only_after_the_cut(self):
        c = ln_chart()
        t = continuation_cut(c)
        times, _, _ = chart_events(c)
        self.assertEqual(t, times[0] + 0.25 * (times[-1] - times[0]))
        out = FieldOperator(HeadRate(), self.m)(EvalCase(c, Condition.of(star=3.2), Scope.continuation(t)))
        self.assertTrue((out['time'][out['ranked']] >= t).all())
        self.assertEqual(out['domain'][0][0], t)
        self.assertLess(out['time'].min(), t)  # given heads are read as context

    def test_skeleton_skips_the_family(self):
        c = ln_chart(8)
        op = FieldOperator(HeadRate(), self.m)
        sk = EvalCase(c, Condition.of(Skeleton.from_chart(c), star=3.2))
        self.assertIn('skipped', evaluate(sk, [op])['results']['head_rate'])
        tm = EvalCase(c, Condition.of(Timing(c.musical_grid()[0]), star=3.2))
        self.assertNotIn('skipped', evaluate(tm, [op])['results']['head_rate'])

    def test_windows(self):
        c = ln_chart()
        w = windows(c)
        self.assertEqual(len(w), 6)
        times, _, _ = chart_events(c)
        self.assertEqual(w[0][2:], (times[0], times[-1]))
        part = window_chart(c, w[-1][2], w[-1][3])
        self.assertTrue(all(w[-1][2] <= o.start_time_ms <= w[-1][3] for o in part.objects))


class SongNullTests(unittest.TestCase):
    def test_p_values_weights_and_n_eff(self):
        rng = np.random.default_rng(0)
        n = 2000
        star = rng.uniform(2, 4, n)
        secs = rng.uniform(60, 240, n)
        stat = rng.normal(1.0, 0.1, n)
        null = SongNull(star, np.log(secs), np.arange(n), dict(mean_low=stat))
        p, n_eff = null.p_values([3.0, 3.0, 3.0], [120.0, 120.0, 120.0], dict(mean_low=np.array([5.0, -5.0, np.nan])))
        w = null.weights(3.0, 120.0)
        self.assertAlmostEqual(p['mean_low'][0], 1 / (1 + w.sum()))
        self.assertAlmostEqual(p['mean_low'][1], 1.0)
        self.assertTrue(np.isnan(p['mean_low'][2]))
        self.assertAlmostEqual(n_eff[0], w.sum() ** 2 / (w ** 2).sum())
        # Uniform p-values for queries drawn from the null.
        q = rng.normal(1.0, 0.1, 400)
        pq, _ = null.p_values(np.full(400, 3.0), np.full(400, 120.0), dict(mean_low=q))
        self.assertLess(abs(np.mean(pq['mean_low'] < 0.05) - 0.05), 0.04)

    def test_windows_of_one_song_count_once(self):
        null = SongNull(np.full(6, 3.0), np.log(np.full(6, 100.0)), np.zeros(6, dtype=np.int64),
                        dict(mean_low=np.arange(6.0)))
        _, n_eff = null.p_values([3.0], [100.0], dict(mean_low=np.array([0.0])))
        self.assertAlmostEqual(n_eff[0], 1.0)


if __name__ == '__main__':
    unittest.main()


class FoldScoringTests(unittest.TestCase):
    def test_a_song_is_scored_without_its_fold(self):
        from ensomi_model.evaluation.harness import cases
        from ensomi_model.evaluation.harness.trivial import fold_of
        cases.init_worker(cases.DEFAULT_FAMILY, model())
        op = cases.operator_for(fold_of('g42'))
        self.assertEqual(op.fold, fold_of('g42'))
        self.assertIs(cases.operator_for(fold_of('g42')), op)
        self.assertEqual(cases.load_family(cases.DEFAULT_FAMILY).name, 'head_rate')
