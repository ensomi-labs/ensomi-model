import math
import unittest

import numpy as np

from ensomi_model.evaluation.beats import (
    LN_HEAD, LN_RELEASE, TAP, BarStart, BeatGrid, Segment, chart_events, event_times, fold_for_bpm, renotate)
from ensomi_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ensomi_model.osu_core.timing import RedTimingPoint


def grid(*points):
    return BeatGrid.from_timing_points([RedTimingPoint(*p) for p in points])


def assert_same_coords(test, a, b):
    for key in ('segment', 'snap', 'bar'):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    for key in ('beat', 'global_beat', 'bar_beat', 'residual_ms'):
        np.testing.assert_allclose(a[key], b[key], atol=1e-9, err_msg=key)


class FoldTests(unittest.TestCase):
    def test_fold_moves_bpm_into_80_160(self):
        cases = {60.0: 1, 79.9: 1, 80.0: 0, 120.0: 0, 159.9: 0, 160.0: -1, 240.0: -1, 320.0: -2, 640.0: -3}
        for bpm, fold in cases.items():
            with self.subTest(bpm=bpm):
                self.assertEqual(fold_for_bpm(bpm), fold)
                self.assertTrue(80.0 <= bpm * 2.0 ** fold < 160.0)

    def test_folding_rescales_beats_subdivisions_and_bars(self):
        fast = grid((0.0, 250.0, 4))   # notated 240 BPM
        self.assertEqual((fast.segments[0].fold, fast.segments[0].canonical_bpm), (-1, 120.0))
        coords = fast.locate([62.5, 250.0, 1000.0, 2000.0])
        # A notated 1/4 at 240 is a canonical 1/8 at 120; a notated beat is half a canonical beat.
        np.testing.assert_allclose(coords['beat'], [0.125, 0.5, 2.0, 4.0])
        np.testing.assert_array_equal(coords['snap'], [8, 2, 1, 1])
        # A bar holds four canonical beats: two notated bars at 240.
        np.testing.assert_array_equal(coords['bar'], [0, 0, 0, 1])
        slow = grid((0.0, 1000.0, 4))  # notated 60 BPM
        self.assertEqual(slow.segments[0].fold, 1)
        np.testing.assert_allclose(slow.locate([500.0, 2000.0])['beat'], [1.0, 4.0])

    def test_every_segment_folds_on_its_own(self):
        g = grid((0.0, 60000.0 / 158.0, 4), (50000.0, 60000.0 / 162.0, 4))
        self.assertEqual([s.fold for s in g.segments], [0, -1])
        np.testing.assert_allclose(60000.0 / g.canonical_beat_lengths(), [158.0, 81.0])
        self.assertEqual(g.dominant((0.0, 60000.0)), 0)
        self.assertEqual(g.dominant((0.0, 120000.0)), 1)

    def test_renotating_the_chart_or_any_segment_keeps_coordinates(self):
        points = [RedTimingPoint(37.0, 500.0, 4), RedTimingPoint(30037.0, 400.0, 3),
                  RedTimingPoint(40037.0, 300.0, 4)]
        times = np.array([37.0, 162.0, 287.0, 30037.0, 30170.333, 35000.0, 40037.0, 40112.0, 59000.0])
        base = BeatGrid.from_timing_points(points).locate(times)
        for factor in (-2, -1, 1, 2, (1, 0, -1), (0, -1, 1), (2, 1, 0)):
            with self.subTest(factor=factor):
                other = BeatGrid.from_timing_points(renotate(points, factor))
                assert_same_coords(self, base, other.locate(times))

    def test_renotating_past_the_raw_red_line_limit_keeps_coordinates(self):
        # A real segment at 39.9 BPM: at half BPM it is 3,008.6 ms a beat, past the raw limit of 3,000.
        points = [RedTimingPoint(10460.0, 1504.313736, 4)]
        times = [10460.0, 10836.078434, 11212.156868]
        base, other = BeatGrid.from_timing_points(points), BeatGrid.from_timing_points(renotate(points, -1))
        self.assertEqual(other.segments[0].beat_length_ms, 3008.627472)
        np.testing.assert_array_equal(base.canonical_beat_lengths(), other.canonical_beat_lengths())
        assert_same_coords(self, base.locate(times), other.locate(times))

    def test_every_notation_that_folds_is_a_valid_segment(self):
        by_bpm = {80.0: 0, 160.0: -1, 40.0: 1, 320.0: -2, 20.0: 2, 1000.0: -3}
        folds = {60000.0 / bpm: fold for bpm, fold in by_bpm.items()}
        folds.update({1504.313736: 2, 3008.627472: 3, 0.01: -16, 1e6: 11})
        for length, fold in folds.items():
            with self.subTest(beat_length_ms=length):
                s = grid((0.0, length, 4)).segments[0]
                self.assertEqual(s.fold, fold)
                # The canonical BPM a grid reports is the fold's own quantity, so it cannot round out of range.
                self.assertEqual(s.canonical_bpm, 60000.0 / length * 2.0 ** fold)
                self.assertTrue(80.0 <= s.canonical_bpm < 160.0)


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
        c = g.locate([2500.0, 2999.0, 3000.0, 3400.0])
        np.testing.assert_array_equal(c['segment'], [0, 0, 1, 1])
        # 1 ms before the red line snaps onto its downbeat.
        np.testing.assert_array_equal(c['bar'], [1, 2, 2, 2])
        np.testing.assert_allclose(c['bar_beat'], [1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(c['global_beat'][2:], [6.0, 7.0])
        g = grid((0.0, 500.0, 4), (4001.0, 400.0, 4))  # 1 ms past a bar line: no extra bar
        self.assertEqual(int(g.locate([4001.0])['bar'][0]), 2)

    def test_bars_follow_bar_starts_not_segments(self):
        segments = (Segment(0.0, 500.0, 4), Segment(6000.0, 400.0, 4))
        # A 3-beat bar starts at 3000 within the first segment (within 1 ms of beat 6);
        # the second segment starts no bar, so bars of 3 run on across it.
        g = BeatGrid(segments, (BarStart(0.0, 4), BarStart(3001.0, 3)))
        c = g.locate([2500.0, 3000.0, 4500.0, 6000.0, 6400.0, 6800.0])
        np.testing.assert_array_equal(c['bar'], [1, 2, 3, 4, 4, 4])
        np.testing.assert_allclose(c['bar_beat'], [1.0, 0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(c['segment'], [0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(c['beat'][3:], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(g.time_of_bar([0, 1, 2, 3, 4, 5]), [0.0, 2000.0, 3000.0, 4500.0, 6000.0, 7200.0])

    def test_time_of_inverts_locate(self):
        g = grid((-12.5, 333.333, 4), (20000.0, 271.0, 7))
        times = np.linspace(-100.0, 59000.0, 997)
        c = g.locate(times)
        np.testing.assert_allclose(g.time_of(c['segment'], c['beat']), times, atol=1e-6)

    def test_invalid_grids_are_refused(self):
        with self.assertRaises(ValueError):
            BeatGrid((), (BarStart(0.0, 4),))
        bad = [Segment(0.0, length, 4) for length in (0.0, -500.0, math.inf, -math.inf, math.nan)]
        bad += [Segment(0.0, 500.0, 0), Segment(0.0, 500.0, -4), Segment(math.nan, 500.0, 4)]
        for segment in bad:
            with self.subTest(segment=segment), self.assertRaises(ValueError):
                BeatGrid((segment,), (BarStart(0.0, 4),))
        with self.assertRaises(ValueError):
            BeatGrid((Segment(0.0, 500.0, 4),), ())


class EventTests(unittest.TestCase):
    def test_holds_give_head_and_release_in_time_order(self):
        objects = [ManiaHitObject(500.0, 1500.0, 2, ManiaHitObjectKind.HOLD),
                   ManiaHitObject(1000.0, 1000.0, 0, ManiaHitObjectKind.TAP),
                   ManiaHitObject(0.0, 0.0, 3, ManiaHitObjectKind.TAP)]
        events = chart_events(objects, grid((0.0, 500.0, 4)))
        np.testing.assert_array_equal(events.time_ms, [0.0, 500.0, 1000.0, 1500.0])
        np.testing.assert_array_equal(events.kind, [TAP, LN_HEAD, TAP, LN_RELEASE])
        np.testing.assert_array_equal(events.lane, [3, 2, 0, 2])
        np.testing.assert_array_equal(events.head_ms, [0.0, 500.0, 1000.0, 500.0])
        part = events.take(events.kind != TAP)
        np.testing.assert_allclose(part.coords['beat'], [1.0, 3.0])
        np.testing.assert_array_equal(np.sort(event_times(objects)), [0.0, 500.0, 1000.0, 1500.0])


if __name__ == '__main__':
    unittest.main()
