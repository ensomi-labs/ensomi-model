import unittest

import numpy as np

from ensomi_model.evaluation.beats import (
    LN_HEAD, LN_RELEASE, TAP, BeatGrid, chart_events, fold_for_bpm, in_spans, renotate)
from ensomi_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ensomi_model.osu_core.timing import RedTimingPoint


def grid(*points, span=(0.0, 60000.0)):
    return BeatGrid.from_timing_points([RedTimingPoint(*p) for p in points], span)


class FoldTests(unittest.TestCase):
    def test_fold_moves_bpm_into_80_160(self):
        cases = {60.0: 1, 79.9: 1, 80.0: 0, 120.0: 0, 159.9: 0, 160.0: -1, 240.0: -1, 320.0: -2, 640.0: -3}
        for bpm, fold in cases.items():
            with self.subTest(bpm=bpm):
                self.assertEqual(fold_for_bpm(bpm), fold)
                self.assertTrue(80.0 <= bpm * 2.0 ** fold < 160.0)

    def test_folding_rescales_beats_subdivisions_and_bars(self):
        fast = grid((0.0, 250.0, 4))   # notated 240 BPM
        self.assertEqual(fast.fold, -1)
        self.assertEqual(fast.canonical_bpm, 120.0)
        coords = fast.locate([62.5, 250.0, 1000.0, 2000.0])
        # A notated 1/4 at 240 is a canonical 1/8 at 120; a notated beat is half a canonical beat.
        np.testing.assert_allclose(coords['beat'], [0.125, 0.5, 2.0, 4.0])
        np.testing.assert_array_equal(coords['snap'], [8, 2, 1, 1])
        # A bar holds four canonical beats: two notated bars at 240.
        np.testing.assert_array_equal(coords['bar'], [0, 0, 0, 1])
        slow = grid((0.0, 1000.0, 4))  # notated 60 BPM
        self.assertEqual(slow.fold, 1)
        np.testing.assert_allclose(slow.locate([500.0, 2000.0])['beat'], [1.0, 4.0])

    def test_renotated_chart_gets_identical_canonical_coordinates(self):
        points = [RedTimingPoint(37.0, 500.0, 4), RedTimingPoint(30037.0, 400.0, 3)]
        times = np.array([37.0, 162.0, 287.0, 30037.0, 30170.333, 45000.0, 59000.0])
        base = BeatGrid.from_timing_points(points, (0.0, 60000.0))
        for factor in (-2, -1, 1, 2):
            with self.subTest(factor=factor):
                other = BeatGrid.from_timing_points(renotate(points, factor), (0.0, 60000.0))
                self.assertEqual(other.fold, base.fold - factor)
                a, b = base.locate(times), other.locate(times)
                for key in ('segment', 'snap', 'bar'):
                    np.testing.assert_array_equal(a[key], b[key])
                for key in ('beat', 'global_beat', 'bar_beat', 'residual_ms'):
                    np.testing.assert_allclose(a[key], b[key], atol=1e-9)

    def test_one_fold_per_chart_from_the_dominant_segment(self):
        g = grid((0.0, 60000.0 / 158.0, 4), (50000.0, 60000.0 / 162.0, 4))
        self.assertEqual(g.dominant, 0)
        self.assertEqual(g.fold, 0)  # the short 162 BPM segment is not folded to 81
        np.testing.assert_allclose(60000.0 / g.canonical_beat_lengths(), [158.0, 162.0])
        g = grid((0.0, 60000.0 / 158.0, 4), (10000.0, 60000.0 / 162.0, 4))
        self.assertEqual((g.dominant, g.fold), (1, -1))


class LocateTests(unittest.TestCase):
    def test_snap_and_residual(self):
        g = grid((1000.0, 500.0, 4))
        c = g.locate([1000.0, 1000.0 + 500.0 / 3, 1093.75, 1250.4, 1007.0, 990.0])
        np.testing.assert_array_equal(c['snap'][:4], [1, 3, 16, 2])
        self.assertAlmostEqual(c['residual_ms'][3], 0.4)
        self.assertGreater(c['snap'][4], 16)    # 7 ms off: only a very fine subdivision is within 2 ms
        self.assertAlmostEqual(c['beat'][5], -0.02)  # before the first red line: first segment extended

    def test_new_red_line_starts_a_new_bar_and_partial_bars_count(self):
        g = grid((0.0, 500.0, 4), (3000.0, 400.0, 4))  # 6 beats = 1.5 bars before the second line
        c = g.locate([2999.0, 3000.0, 3400.0])
        np.testing.assert_array_equal(c['segment'], [0, 1, 1])
        np.testing.assert_array_equal(c['bar'], [1, 2, 2])
        np.testing.assert_allclose(c['bar_beat'], [2.0, 0.0, 1.0])
        np.testing.assert_allclose(c['global_beat'][1:], [6.0, 7.0])
        g = grid((0.0, 500.0, 4), (4001.0, 400.0, 4))  # 1 ms past a bar line: no extra bar
        self.assertEqual(int(g.locate([4001.0])['bar'][0]), 2)

    def test_time_of_inverts_locate(self):
        g = grid((-12.5, 333.333, 4), (20000.0, 271.0, 7))
        times = np.linspace(-100.0, 59000.0, 997)
        c = g.locate(times)
        np.testing.assert_allclose(g.time_of(c['segment'], c['beat']), times, atol=1e-6)


class EventTests(unittest.TestCase):
    def test_holds_give_head_and_release_in_time_order(self):
        objects = [ManiaHitObject(500.0, 1500.0, 2, ManiaHitObjectKind.HOLD),
                   ManiaHitObject(1000.0, 1000.0, 0, ManiaHitObjectKind.TAP),
                   ManiaHitObject(0.0, 0.0, 3, ManiaHitObjectKind.TAP)]
        events = chart_events(objects, grid((0.0, 500.0, 4)))
        np.testing.assert_array_equal(events.time_ms, [0.0, 500.0, 1000.0, 1500.0])
        np.testing.assert_array_equal(events.kind, [TAP, LN_HEAD, TAP, LN_RELEASE])
        np.testing.assert_array_equal(events.lane, [3, 2, 0, 2])
        part = events.take(in_spans(events.time_ms, [(400.0, 600.0), (1500.0, 2000.0)]))
        np.testing.assert_array_equal(part.kind, [LN_HEAD, LN_RELEASE])
        np.testing.assert_allclose(part.coords['beat'], [1.0, 3.0])


if __name__ == '__main__':
    unittest.main()
