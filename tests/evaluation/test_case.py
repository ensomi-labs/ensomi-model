import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ensomi_model.evaluation.beats import LN_RELEASE
from ensomi_model.evaluation.case import (
    HEAD_TIMES, RELEASE_TIMES, Audio, Chart, Condition, Context, EvalCase, Scope, Skeleton, Spans, Timing,
    condition_from_chart, evaluate)

LANE_X = (64, 192, 320, 448)


def write_chart(directory: Path, name='a.osu', timing=('0,500,4,2,0,80,1,0',), objects=()):
    lines = ['osu file format v14', '', '[General]', 'Mode: 3', '', '[Difficulty]', 'CircleSize:4', '',
             '[TimingPoints]', *timing, '', '[HitObjects]']
    for start, lane, end in objects:
        lines.append(f'{LANE_X[lane]},192,{start},1,0,0:0:0:0:' if end is None
                     else f'{LANE_X[lane]},192,{start},128,0,{end}:0:0:0:0:')
    path = directory / name
    path.write_text('\n'.join(lines) + '\n')
    return path


OBJECTS = [(t, (t // 500) % 4, None) for t in range(0, 16000, 500)] + [(7750, 2, 8250)]


class SpansTests(unittest.TestCase):
    def test_parse_editor_timestamps_and_milliseconds(self):
        self.assertEqual(Spans.parse('00:01:000..00:02:500').intervals, ((1000.0, 2500.0),))
        self.assertEqual(Spans.parse('1:00:00:000..').intervals, ((3600000.0, math.inf),))
        self.assertEqual(Spans.parse('..30000, 40000..50000').intervals, ((-math.inf, 30000.0), (40000.0, 50000.0)))
        for bad in ('00:01:000', '1:2..3', 'a..b'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Spans.parse(bad)

    def test_set_operations(self):
        a, b = Spans.of((0, 10), (5, 20), (30, 40)), Spans.of((15, 35))
        self.assertEqual(a.intervals, ((0.0, 20.0), (30.0, 40.0)))
        self.assertEqual((a & b).intervals, ((15.0, 20.0), (30.0, 35.0)))
        self.assertEqual((a - b).intervals, ((0.0, 15.0), (35.0, 40.0)))
        self.assertEqual((~a).intervals, ((-math.inf, 0.0), (20.0, 30.0), (40.0, math.inf)))
        self.assertFalse(Spans())
        self.assertFalse(Spans.everything() & Spans())
        np.testing.assert_array_equal(a.contains([0, 19.9, 20, 30, 45]), [True, True, False, True, False])

    def test_bars_of_a_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            chart = Chart.from_osu(write_chart(Path(tmp), timing=('0,250,4,2,0,80,1,0',), objects=OBJECTS))
        grid, _ = chart.musical_grid()
        # Notated 240 BPM is canonical 120: a bar is four canonical beats, 2000 ms.
        self.assertEqual(Spans.bars(grid, 2, 4).intervals, ((4000.0, 8000.0),))


class ChartTests(unittest.TestCase):
    def test_from_osu_keeps_every_red_line_and_sections_by_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_chart(Path(tmp), timing=('0,500,4,2,0,80,1,0', '4000,1e-6,4,2,0,80,1,0',
                                                  '4001,500,4,2,0,80,1,0'), objects=OBJECTS)
            chart = Chart.from_osu(path)
        self.assertEqual(len(chart.red_lines), 3)
        grid, roles = chart.musical_grid()
        self.assertEqual([r.role for r in roles], ['musical', 'expressive', 'redundant'])
        part = chart.section(Spans.of((7500, 8000)))
        self.assertEqual([(o.start_time_ms, o.end_time_ms) for o in part.objects], [(7500.0, 7500.0), (7750.0, 8250.0)])
        self.assertEqual(part.red_lines, chart.red_lines)


class ConditionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.chart = Chart.from_osu(write_chart(Path(tmp.name), objects=OBJECTS))
        self.grid = self.chart.musical_grid()[0]

    def test_components_compose(self):
        context = Context.from_chart(self.chart, Spans.before(4000))
        condition = Timing(self.grid) | context | Audio('dataset/0/1/audio.mp3')
        self.assertEqual(condition.name, 'timing+context+audio')
        self.assertEqual(condition.fixed, frozenset({'grid'}))
        self.assertEqual(len(condition.context.chart.objects), 8)
        self.assertEqual(Condition().name, 'none')
        self.assertEqual((Skeleton.from_chart(self.chart) | context).name, 'skeleton+context')
        # Release times are given only on request: by default the generator chooses them.
        self.assertEqual(Condition.of(Skeleton.from_chart(self.chart)).fixed, frozenset({'grid', HEAD_TIMES}))
        self.assertEqual(Condition.of(Skeleton.from_chart(self.chart, releases=True)).fixed,
                         frozenset({'grid', HEAD_TIMES, RELEASE_TIMES}))
        with self.assertRaises(ValueError):
            Timing(self.grid) | Skeleton.from_chart(self.chart)
        self.assertEqual(condition_from_chart(self.chart, ['skeleton', 'context'], given=Spans.before(4000)).name,
                         'skeleton+context')
        with self.assertRaises(ValueError):
            condition_from_chart(self.chart, ['audio'])

    def test_scopes(self):
        scope = Scope.continuation(4000)
        self.assertEqual((scope.given.intervals, scope.scored.intervals),
                         (((-math.inf, 4000.0),), ((4000.0, math.inf),)))
        edit = Scope.edit(Spans.of((4000, 6000)))
        self.assertEqual(edit.given.intervals, ((-math.inf, 4000.0), (6000.0, math.inf)))
        with self.assertRaises(ValueError):
            Scope(Spans.before(5000), Spans.after(4000))

    def test_case_grid_masks_and_fixed_parts(self):
        context = Context.from_chart(self.chart, Spans.before(8000))
        with self.assertRaises(ValueError):  # nothing the condition fixed is scored
            EvalCase(self.chart, Condition.of(context), Scope.continuation(7000))
        case = EvalCase(self.chart, Condition.of(context), Scope.continuation(8000))
        self.assertEqual(case.grid_source, 'chart')
        events, given, scored = case.events()
        self.assertFalse((given & scored).any())
        # The long note starting at 7750 is given whole: its release at 8250 goes with its head.
        release = (events.kind == LN_RELEASE)
        self.assertTrue(given[release].all())
        self.assertEqual(int(scored.sum()), 16)
        other = EvalCase(self.chart, Timing(self.grid) | context, Scope.passage(Spans.of((8000, 12000))))
        self.assertEqual((other.grid_source, other.describe()['condition']), ('condition', 'timing+context'))

    def test_evaluate_skips_operators_on_fixed_aspects(self):
        class Density:
            name, judges = 'density', frozenset({HEAD_TIMES})

            def __call__(self, case):
                events, _, scored = case.events()
                return dict(events=int(scored.sum()))

        class Releases:
            name, judges = 'releases', frozenset({RELEASE_TIMES})

            def __call__(self, case):
                return dict(ok=True)

        timed = EvalCase(self.chart, Condition.of(Timing(self.grid)), Scope.passage(Spans.of((0, 4000))))
        result = evaluate(timed, [Density()])
        self.assertEqual(result['results']['density'], dict(events=8))
        self.assertEqual(result['case']['scored'], '0..4000')
        skeleton = EvalCase(self.chart, Condition.of(Skeleton.from_chart(self.chart)))
        results = evaluate(skeleton, [Density(), Releases()])['results']
        self.assertEqual(results['density'], dict(skipped='condition fixes head_times'))
        self.assertEqual(results['releases'], dict(ok=True))


if __name__ == '__main__':
    unittest.main()
