import unittest

import numpy as np

from ensomi_model.evaluation.harness.arrays import Objects
from ensomi_model.evaluation.harness.cases import song_span, splice
from ensomi_model.evaluation.harness.chain import continuation_cut
from ensomi_model.evaluation.harness.descriptions import crowding, describe
from ensomi_model.evaluation.harness.injections import FAMILIES, Source, dose_label
from ensomi_model.evaluation.legality import violations
from ensomi_model.evaluation.osu_text import round_trip

from ._harness_charts import busy_chart, chart, hold, ln_chart, tap


def source(c=None, **kw):
    c = c or round_trip(busy_chart())
    grid = c.musical_grid()[0]
    return Source(c, grid, 4.0, mode_rate=lambda k: 6.0, span=song_span(c, grid), **kw)


def heads_per_bar(objs: Objects, grid):
    bar = grid.locate(objs.start)['bar']
    return np.bincount(bar - bar.min() if len(bar) else bar)


def written(objs: Objects, c):
    return round_trip(c.__class__(objs.to_objects(), c.red_lines))


def family(name):
    return next(f for f in FAMILIES if f.name == name)


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.src = source()
        self.base = heads_per_bar(self.src.objects, self.src.grid)

    def apply(self, name, dose, seed=0, src=None):
        return family(name).apply(src or self.src, dose, np.random.default_rng(seed))

    def test_every_family_and_dose_runs_and_is_seeded(self):
        donor = round_trip(busy_chart(24))
        self.src = source(donor=(donor, donor.musical_grid()[0]))
        for f in FAMILIES:
            for dose in f.doses:
                with self.subTest(family=f.name, dose=dose_label(dose)):
                    a = self.apply(f.name, dose, seed=1)
                    b = self.apply(f.name, dose, seed=1)
                    self.assertIsNotNone(a)
                    self.assertEqual(a.to_objects(), b.to_objects())

    def test_head_preserving_families_keep_heads_per_bar_and_stay_legal(self):
        for src in (self.src, source(round_trip(ln_chart()))):
            base = heads_per_bar(src.objects, src.grid)
            for f in FAMILIES:
                if not f.heads_unchanged:
                    continue
                for dose in f.doses:
                    with self.subTest(family=f.name, dose=dose_label(dose)):
                        out = written(self.apply(f.name, dose, src=src), src.chart)
                        self.assertEqual(violations(out.objects, song_span=src.span), {})
                        np.testing.assert_array_equal(heads_per_bar(Objects.from_chart(out), src.grid), base)

    def test_group_moves_are_legal_on_a_chart_with_long_notes(self):
        src = source(round_trip(ln_chart()))
        for name, dose in (('D6', 0.25), ('D6', 1.0), ('D7', 0.25), ('D7', 1.0), ('F1', 0.1), ('F1', 1.0),
                           ('D10', 'loop_first_quarter'), ('F2', 'replaced')):
            with self.subTest(family=name, dose=dose):
                out = self.apply(name, dose, src=src)
                self.assertEqual(violations(written(out, src.chart).objects, song_span=src.span), {})
                self.assertLessEqual(out.note['done'], out.note['requested'])
        d6 = self.apply('D6', 1.0, src=src)
        self.assertGreater(int(np.sum(d6.lane != src.objects.lane)), 0)
        d7 = self.apply('D7', 1.0, src=src)
        self.assertGreater(d7.note['done'], 0)

    def test_single_edit_families_write_integer_times(self):
        for name in ('D2', 'D2s', 'D3same', 'D3other', 'D4', 'D8', 'D9', 'F6', 'F7'):
            for dose in family(name).doses:
                with self.subTest(family=name, dose=dose_label(dose)):
                    out = self.apply(name, dose)
                    self.assertEqual(set(written(out, self.src.chart).objects), set(out.to_objects()))

    def test_density_drift_scales_heads(self):
        n = len(self.src.objects)
        for factor in (0.5, 0.67, 0.8, 1.25, 1.5, 2.0):
            with self.subTest(factor=factor):
                out = written(self.apply('D9', factor), self.src.chart)
                self.assertEqual(violations(out.objects), {})
                self.assertAlmostEqual(len(out.objects) / n, factor, delta=0.01)

    def test_short_holds_and_crowding(self):
        out = Objects.from_chart(written(self.apply('D2', 0.4), self.src.chart))
        added = out.hold & ~np.isin(out.start, self.src.objects.start[self.src.objects.hold])
        self.assertGreater(added.sum(), 0)
        self.assertTrue(np.all(out.end[added] - out.start[added] <= 500 / 8 + 0.5))
        out = Objects.from_chart(written(self.apply('D3other', 0.2), self.src.chart))
        gaps = crowding(out, 'other')
        self.assertGreaterEqual(int(((gaps > 0) & (gaps <= 40)).sum()), round(0.2 * self.src.objects.hold.sum()))

    def test_triplets_never_ln_and_degenerate(self):
        out = written(self.apply('D8', 1.0), self.src.chart)
        self.assertGreater(describe(out, self.src.grid)['triplet_share'], describe(self.src.chart, self.src.grid)['triplet_share'])
        self.assertFalse(Objects.from_chart(written(self.apply('F3', 'replaced'), self.src.chart)).hold.any())
        one = Objects.from_chart(written(self.apply('D10', 'one_lane'), self.src.chart))
        self.assertEqual(len(np.unique(one.lane)), 1)
        quad = written(self.apply('D10', 'all_quad'), self.src.chart)
        self.assertEqual(describe(quad, self.src.grid)['mean_chord'], 4.0)
        loop = written(self.apply('D10', 'loop_first_quarter'), self.src.chart)
        self.assertEqual(violations(loop.objects), {})
        self.assertGreater(describe(loop, self.src.grid)['bar_repeat_share'], 0.5)

    def test_mode_chart_tiles_one_unit(self):
        out = written(self.apply('F2', 'replaced'), self.src.chart)
        self.assertEqual(violations(out.objects), {})
        self.assertEqual(describe(out, self.src.grid)['unit_count_cv'], 0.0)

    def test_anchor_lengthens_runs(self):
        base = describe(self.src.chart, self.src.grid)['run_length']
        out = written(self.apply('D1', (8, 4)), self.src.chart)
        self.assertGreater(describe(out, self.src.grid)['run_length'], base)

    def test_small_chart_rounds_to_no_change(self):
        c = round_trip(chart([tap(0, 0), tap(500, 1), tap(1000, 2)]))
        out = family('D2').apply(source(c), 0.05, np.random.default_rng(0))
        self.assertEqual(out.to_objects(), Objects.from_chart(c).to_objects())


class SpliceTests(unittest.TestCase):
    def test_seam_conflicts_are_repaired(self):
        src = Objects.from_chart(chart([hold(0, 3000, 0), tap(1000, 1), tap(4000, 0), tap(5000, 1)]))
        inj = src.copy()
        inj.lane[2] = 0
        inj.start[2] = 2000.0
        inj.end[2] = 2000.0  # under the given hold
        out, n = splice(src, inj, 1500.0, in_place=True)
        self.assertEqual(n, 1)
        self.assertTrue(out.legal())
        self.assertIn(4000.0, out.start.tolist())  # reverted to its source state
        out, n = splice(src, inj, 1500.0, in_place=False)
        self.assertEqual((n, len(out)), (1, 3))
        clean, n = splice(src, src, 1500.0, in_place=True)
        self.assertEqual((n, len(clean)), (0, 4))

    def test_all_quad_under_continuation_is_legal_after_the_seam(self):
        c = round_trip(ln_chart())
        src = source(c)
        out = family('D10').apply(src, 'all_quad', np.random.default_rng(0))
        spliced, n = splice(src.objects, out, continuation_cut(c), in_place=False)
        self.assertTrue(spliced.legal(src.span))


if __name__ == '__main__':
    unittest.main()
