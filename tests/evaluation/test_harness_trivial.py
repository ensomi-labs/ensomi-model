import math
import unittest

import numpy as np

from ensomi_model.evaluation.case import Condition, EvalCase, Scope, Spans
from ensomi_model.evaluation.harness.trivial import (
    K_FOLDS, N_BINS, N_KEYS, V_BIN, BinnedKDE, HeadRate, group_bootstrap, star_band, value_bin, wilson)

from ._harness_charts import chart, hold, tap


def case(c, scope=None, star=3.2):
    return EvalCase(c, Condition.of(star=star), scope or Scope.whole())


class HeadRateTests(unittest.TestCase):
    def test_uniform_chart_reads_its_rate_everywhere(self):
        # Four heads per second for 60 s: the edge-corrected rate is 4/s at the ends as in the middle.
        c = chart([tap(t, (t // 250) % 4) for t in range(0, 60000, 250)])
        ev = HeadRate().values(case(c))
        rate = np.exp(ev.value)
        self.assertEqual(len(ev), 240)
        np.testing.assert_allclose(rate[[0, 120, -1]], 4.0, rtol=0.03)
        self.assertTrue(ev.scored.all())
        self.assertEqual(ev.domain, ((0.0, 59750.0),))
        np.testing.assert_array_equal(ev.key, np.full(240, int(star_band(3.2))))

    def test_chords_count_every_head_and_releases_do_not(self):
        taps = [tap(t, 0) for t in range(0, 30000, 500)]
        doubled = taps + [tap(t, 1) for t in range(0, 30000, 500)]
        held = [hold(t, t + 200, 0) for t in range(0, 30000, 500)]
        one = np.exp(HeadRate().values(case(chart(taps))).value)
        two = np.exp(HeadRate().values(case(chart(doubled))).value)
        np.testing.assert_allclose(two[::2], 2 * one, rtol=1e-12)
        np.testing.assert_array_equal(HeadRate().values(case(chart(held))).value,
                                      HeadRate().values(case(chart(taps))).value)

    def test_shift_by_whole_ms_is_bit_identical(self):
        objs = [tap(t, (t * 7) % 4) for t in sorted({int(x) for x in np.random.default_rng(0).uniform(0, 90000, 400)})]
        a = HeadRate().values(case(chart(objs)))
        b = HeadRate().values(case(chart([tap(o.start_time_ms + 1037, o.lane) for o in objs])))
        np.testing.assert_array_equal(a.value, b.value)
        np.testing.assert_array_equal(a.time_ms + 1037, b.time_ms)

    def test_reads_given_and_scored_spans_only(self):
        c = chart([tap(t, 0) for t in range(0, 40000, 250)])
        scope = Scope(Spans.of((10000, 20000)), Spans.of((20000, 30000)))
        ev = HeadRate().values(case(c, scope))
        self.assertEqual(ev.time_ms.min(), 10000)
        self.assertEqual(ev.time_ms.max(), 29750)
        self.assertEqual(int(ev.scored.sum()), 40)
        self.assertEqual(ev.domain, ((20000.0, 29750.0),))

    def test_needs_a_star(self):
        with self.assertRaises(ValueError):
            HeadRate().values(EvalCase(chart([tap(0, 0)]), Condition(), Scope.whole()))


def model(seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros((N_KEYS, K_FOLDS, N_BINS))
    for f in range(K_FOLDS):
        np.add.at(counts[6, f], value_bin(rng.normal(1.0, 0.3, 4000)), 1.0)
    return BinnedKDE.fit(counts)


class BinnedKDETests(unittest.TestCase):
    def test_density_is_a_density_and_bandwidth_is_selected(self):
        m = model()
        self.assertAlmostEqual(float(m.full[6].sum() * V_BIN), 1.0, places=9)
        self.assertIn(str(m.bandwidth), m.loglik)
        self.assertEqual(max(m.loglik, key=m.loglik.get), str(m.bandwidth))
        self.assertAlmostEqual(math.log(m.mode_rate(6)), 1.0, delta=0.1)
        self.assertIsNone(m.mode_rate(7))

    def test_cross_fitted_surprisal_uses_the_model_without_the_fold(self):
        m = model()
        v = np.array([1.0, 2.5])
        np.testing.assert_allclose(m.surprisal(v, [6, 6], 2), -np.log(m.without[6, 2][value_bin(v)]))
        np.testing.assert_allclose(m.surprisal(v, [6, 6], -1), -np.log(m.full[6][value_bin(v)]))
        self.assertTrue(np.isnan(m.surprisal(v, [7, 7], -1)).all())

    def test_ranks_are_mid_p_and_both_tails(self):
        m = model()
        s = m.surprisal(np.array([1.0, 2.8, -0.5]), [6, 6, 6], -1)
        lo, hi = m.ranks(s, [6, 6, 6], -1)
        np.testing.assert_allclose(lo + hi, 1.0)
        self.assertGreater(lo[0], 0.75)         # near the mode: not surprising, very typical
        self.assertLess(hi[0], 0.25)
        self.assertLess(lo[1], 0.001)           # far tail: surprising
        self.assertLess(lo[2], 0.001)
        n = m.counts[6].sum()
        self.assertGreaterEqual(lo.min(), 0.5 / (n + 1))
        lo_f, _ = m.ranks(s, [6, 6, 6], 3)
        self.assertGreaterEqual(lo_f.min(), 0.5 / (n - m.counts[6, 3].sum() + 1))

    def test_pickled_model_drops_its_rank_cache(self):
        import pickle
        m = model()
        m.ranks(m.surprisal(np.array([1.0]), [6], -1), [6], -1)
        self.assertTrue(m._tables)
        self.assertFalse(pickle.loads(pickle.dumps(m))._tables)


class IntervalTests(unittest.TestCase):
    def test_intervals(self):
        lo, hi = wilson(5, 100)
        self.assertTrue(lo < 0.05 < hi)
        blo, bhi = group_bootstrap(np.array([1, 0, 0, 0] * 50), np.repeat(np.arange(100), 2), draws=200)
        self.assertTrue(blo < 0.25 < bhi)


if __name__ == '__main__':
    unittest.main()
