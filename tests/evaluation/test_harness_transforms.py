import math
import unittest

import numpy as np

from ensomi_model.evaluation.harness.cases import covariance, make_case, run_case
from ensomi_model.evaluation.harness.chain import FieldOperator, continuation_cut
from ensomi_model.evaluation.harness.transforms import TRANSFORMS
from ensomi_model.evaluation.harness.trivial import HeadRate
from ensomi_model.evaluation.osu_text import round_trip
from ensomi_model.evaluation.redlines import RedLine

from ._harness_charts import busy_chart, ln_chart
from .test_harness_trivial import model


def outputs(c, op, made=None, keep=False):
    grid = made.grid if made is not None and made.grid is not None else c.musical_grid()[0]
    t = continuation_cut(c)
    offset = made.time_offset if made is not None else 0.0
    out = {}
    for scope in ('whole', 'continuation'):
        rec, raw = run_case(make_case(c, grid, scope, 3.2, t=t), op, offset=offset, keep=keep)
        out[scope] = (rec, raw)
    return out


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.op = FieldOperator(HeadRate(), model())

    def test_every_invariant_transform_keeps_the_full_output(self):
        base = ln_chart(30)
        base = base.__class__(base.objects, (RedLine(0.0, 500.0, 4), RedLine(30000.0, 400.0, 4)))
        src = round_trip(base)
        want = outputs(src, self.op)
        for i, transform in enumerate(TRANSFORMS):
            with self.subTest(transform=transform.name):
                made = transform.apply(src, 11 + i)
                self.assertNotIsInstance(made, str)
                rt = round_trip(made.chart, **made.extras)
                got = outputs(rt, self.op, made)
                for scope in ('whole', 'continuation'):
                    same = got[scope][0]['full_hash'] == want[scope][0]['full_hash']
                    self.assertEqual(same, transform.stretch is None, scope)

    def test_stretch_scales_the_rate_by_the_inverse_factor(self):
        src = round_trip(busy_chart(40))
        base = outputs(src, self.op, keep=True)['whole'][1]
        for transform in TRANSFORMS:
            if transform.stretch is None:
                continue
            with self.subTest(transform=transform.name):
                made = transform.apply(src, 0)
                rt = round_trip(made.chart, **made.extras)
                cov = covariance(base, outputs(rt, self.op, made, keep=True)['whole'][1], transform.stretch)
                self.assertEqual(cov['cov_n'], len(src.objects))
                self.assertAlmostEqual(cov['cov_median'], 1.0, delta=0.01)

    def test_mirror_changes_lanes(self):
        src = round_trip(busy_chart(4))
        made = TRANSFORMS[0].apply(src, 0)
        lanes = np.array(sorted((o.start_time_ms, o.lane) for o in made.chart.objects))
        orig = np.array(sorted((o.start_time_ms, 3 - o.lane) for o in src.objects))
        np.testing.assert_array_equal(np.sort(lanes, axis=0), np.sort(orig, axis=0))

    def test_timing_lesion_leaves_no_red_line_and_another_grid(self):
        src = round_trip(busy_chart(4))
        made = next(t for t in TRANSFORMS if t.name == 'timing_lesion').apply(src, 0)
        self.assertEqual(round_trip(made.chart, **made.extras).red_lines, ())
        self.assertTrue(made.extras['green_lines'])
        self.assertNotEqual(made.grid, src.musical_grid()[0])


if __name__ == '__main__':
    unittest.main()
