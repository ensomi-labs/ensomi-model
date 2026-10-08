import unittest

import numpy as np

from ensomi_model.evaluation.case import Condition, EvalCase, Scope, Skeleton, Timing, evaluate
from ensomi_model.evaluation.passages import PLACEHOLDER, Bars4Placeholder, HeadCountPerPassage
from ensomi_model.evaluation.redlines import RedLine

from ._harness_charts import chart, hold, tap


def run(c, scope=None, grid=None):
    grid = grid or c.musical_grid()[0]
    case = EvalCase(c, Condition.of(Timing(grid), star=3.0), scope or Scope.whole())
    return HeadCountPerPassage()(case)


class PassageTests(unittest.TestCase):
    def test_four_bars_trailing_dropped(self):
        # 120 BPM, a head on every beat for ten bars: two passages, two bars dropped.
        c = chart([tap(t, (t // 500) % 4) for t in range(0, 20000, 500)])
        out = run(c)
        self.assertEqual(out['segmentation'], PLACEHOLDER)
        np.testing.assert_array_equal(out['count'], [16, 16])
        np.testing.assert_array_equal(out['label'], [0, 4])
        self.assertEqual(out['diagnostics']['dropped_bars'], 2)
        self.assertEqual(out['diagnostics']['heads_dropped'], 8)

    def test_empty_passages_count_and_releases_do_not(self):
        c = chart([tap(0, 0), hold(500, 900, 1), tap(16000 + 500, 2), tap(31500, 3)])
        np.testing.assert_array_equal(run(c)['count'], [2, 0, 1, 1])

    def test_passages_stay_inside_a_segment(self):
        # Six bars at 120 BPM, then 150 BPM from 12000 ms: no passage crosses 12000.
        lines = ((0.0, 500.0, 4), (12000.0, 400.0, 4))
        objs = [tap(t, 0) for t in range(0, 12000, 500)] + [tap(12000 + k * 400, 1) for k in range(40)]
        out = run(chart(objs, lines))
        self.assertEqual([s[0] for s in out['segments']], [0, 1, 1])
        np.testing.assert_array_equal(out['count'], [16, 16, 16])
        np.testing.assert_allclose(out['canonical_bpm'], [120.0, 150.0, 150.0])
        self.assertEqual(out['diagnostics']['dropped_bars'], 2 + 2)

    def test_renotation_gives_identical_passages(self):
        objs = [tap(t, (t // 250) % 4) for t in range(0, 30000, 250)]
        a = run(chart(objs, ((0.0, 500.0, 4),)))
        b = run(chart(objs, ((0.0, 250.0, 4),)))
        for key in ('count', 'label'):
            np.testing.assert_array_equal(a[key], b[key])

    def test_continuation_starts_at_the_scored_bar(self):
        c = chart([tap(t, 0) for t in range(0, 32000, 500)])
        grid = c.musical_grid()[0]
        t = float(grid.time_of_bar(4))
        out = run(c, Scope.continuation(t), grid)
        np.testing.assert_array_equal(out['label'], [4, 8, 12])
        np.testing.assert_array_equal(out['count'], [16, 16, 16])

    def test_skeleton_skips_the_operator(self):
        c = chart([tap(t, 0) for t in range(0, 8000, 500)])
        case = EvalCase(c, Condition.of(Skeleton.from_chart(c), star=3.0))
        result = evaluate(case, [HeadCountPerPassage()])['results']['head_count_per_passage']
        self.assertIn('skipped', result)

    def test_layout_counts_short_segments_and_fold_changes(self):
        lines = ((0.0, 60000 / 158, 4), (60000 / 158 * 16, 60000 / 162, 4))
        c = chart([tap(t, 0) for t in np.arange(0, 30000, 60000 / 162).round()], lines)
        grid = c.musical_grid()[0]
        lay = Bars4Placeholder().layout(grid, 0, 10)
        self.assertEqual(lay['fold_changes'], 1)
        self.assertEqual(lay['short_segments'], 0)


if __name__ == '__main__':
    unittest.main()
