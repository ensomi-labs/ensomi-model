import math
import unittest

import numpy as np

from ensomi_model.evaluation.harness.cases import covariance, make_case, run_case
from ensomi_model.evaluation.harness.chain import FieldOperator, continuation_cut
from ensomi_model.evaluation.harness.transforms import TRANSFORMS, expressive_ok
from ensomi_model.evaluation.harness.trivial import HeadRate
from ensomi_model.evaluation.osu_text import round_trip
from ensomi_model.evaluation.redlines import RedLine

from ._harness_charts import busy_chart, chart, ln_chart, tap
from .test_harness_trivial import model


def outputs(c, op, made=None, keep=False):
    grid = made.grid if made is not None and made.grid is not None else c.musical_grid()[0]
    t = continuation_cut(c)
    offset = made.time_offset if made is not None else 0.0
    out = {}
    for scope in ('whole', 'continuation'):
        rec, raw = run_case(make_case(c, grid, scope, 3.2, t=t, grid_from_chart=bool(made and made.grid_from_chart)),
                            op, offset=offset, keep=keep)
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
                self.assertTrue(expressive_ok(rt, made.check_expressive))
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

    def test_stretch_applies_across_the_fold_boundary(self):
        c = round_trip(busy_chart(8).__class__(busy_chart(8).objects, (RedLine(0.0, 60000 / 158.0, 4),)))
        stretch = next(t for t in TRANSFORMS if t.name == 'stretch_0.95')
        self.assertNotIsInstance(stretch.apply(c, 0), str)

    def test_mirror_changes_lanes(self):
        src = round_trip(busy_chart(4))
        made = TRANSFORMS[0].apply(src, 0)
        lanes = np.array(sorted((o.start_time_ms, o.lane) for o in made.chart.objects))
        orig = np.array(sorted((o.start_time_ms, 3 - o.lane) for o in src.objects))
        np.testing.assert_array_equal(np.sort(lanes, axis=0), np.sort(orig, axis=0))


class ExpressiveLineTests(unittest.TestCase):
    def test_stop_on_an_existing_line_offset_keeps_the_grid(self):
        # The median head sits on the second red line's offset: the inserted stop must not supersede it.
        objs = [tap(t, 0) for t in range(0, 8000, 500)] + [tap(8000 + k * 400, 1) for k in range(17)]
        c = round_trip(chart(objs, lines=((0.0, 500.0, 4), (8000.0, 400.0, 4))))
        heads = sorted(o.start_time_ms for o in c.objects)
        self.assertEqual(heads[len(heads) // 2], 8000.0)
        made = next(t for t in TRANSFORMS if t.name == 'expressive_lines').apply(c, 0)
        rt = round_trip(made.chart)
        self.assertTrue(expressive_ok(rt, made.check_expressive))
        self.assertEqual(rt.musical_grid()[0], c.musical_grid()[0])


if __name__ == '__main__':
    unittest.main()
