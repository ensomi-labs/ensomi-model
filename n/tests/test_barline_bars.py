"""Barline art starts no bar (human decision, 2026-10-19), on inputs that can break it.

A red line that keeps the tempo and the notated grid in force and starts no
complete bar of its own meter is ignored altogether: no segment start, no bar
start, and the bar grid of the last other red line runs on through it. Lines
that change the timing (resets, tempo changes) and lines that start a complete
bar of their own meter (meter changes, re-barring, visual meters) still start
bars.
"""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.mapformat import MusicMap
from ensomi_n.timing import Reference, Segment, Timing

FAST = 300.0                      # 200 BPM notated
NORMAL = 500.0                    # 120 BPM
HI = 40 * FAST


def bars(segments, lo=0.0, hi=HI):
    b = Timing.from_segments(segments, lo, hi).bar_starts()
    return b[(b >= lo) & (b <= hi)]


def art_chart(step):
    """4/4 with barline art only: same-tempo lines on the grid, each holding less than a bar of its own meter.

    Lines at beats 9 and 10 (one beat each), 13 (omit-first-barline, two beats), 15 (meter 7, five beats,
    over the grid's bar at 16) and 26 (two beats); kiai lines at 20 (six beats) and 28 (four) hold
    complete bars on the grid's downbeats."""
    segs = [Segment(0.0, step, 4)]
    for beat, meter, bar in ((9, 4, True), (10, 4, True), (13, 4, False), (15, 7, True), (20, 4, True),
                             (26, 4, True), (28, 4, True), (32, 4, True)):
        segs.append(Segment(beat * step, step, meter, bar))
    return segs


@pytest.mark.parametrize('step', [FAST, NORMAL])
def test_barline_art_adds_no_bars(step):
    hi = 40 * step
    b_art = bars(art_chart(step), hi=hi)
    np.testing.assert_array_equal(b_art, np.arange(11) * 4 * step)       # 0, 4, ..., 40 beats: the single line's
    np.testing.assert_array_equal(b_art, bars([Segment(0.0, step, 4)], hi=hi))
    t = Timing.from_segments(art_chart(step), 0.0, hi)
    starts = t.beats[t.seg_start]
    np.testing.assert_array_equal(starts[starts >= 0], np.array([0, 20, 28, 32]) * step)
    # Beats, canonical lines and free factors are those of the single line.
    one = Timing.from_segments([Segment(0.0, step, 4)], 0.0, hi)
    np.testing.assert_allclose(t.beats, one.beats, rtol=0, atol=1e-9)
    np.testing.assert_allclose(t.canonical().t, one.canonical().t, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(t.canonical().factor, one.canonical().factor)


def test_a_beat_list_on_segments_reads_barline_art_alike():
    t = Timing.from_segments(art_chart(FAST), 0.0, HI)
    inside = (t.beats >= 0) & (t.beats <= HI)
    from_beats = Timing.from_beats(t.beats[inside], segments=art_chart(FAST))
    np.testing.assert_array_equal(from_beats.downbeat, t.downbeat[inside])
    np.testing.assert_array_equal(from_beats.seg_start, t.seg_start[inside])


def test_barline_art_scores_bars_one_against_the_single_line():
    heads = np.arange(40) * FAST
    for ref_segs, pred_segs in ((art_chart(FAST), [Segment(0.0, FAST, 4)]),
                                ([Segment(0.0, FAST, 4)], art_chart(FAST))):
        s = score(Reference.from_segments(ref_segs, heads), MusicMap(segments=tuple(pred_segs)))
        assert (s.bars_start, s.bars_same, s.bars_precision) == (1.0, 1.0, 1.0)
        assert s.bl20 == 1.0


def test_meter_changes_still_start_bars():
    # 3/4 from beat 16 for three bars, one 3/4 bar exactly from beat 25, then 4/4 from beat 28.
    segs = [Segment(0.0, FAST, 4), Segment(16 * FAST, FAST, 3), Segment(25 * FAST, FAST, 3),
            Segment(28 * FAST, FAST, 4)]
    np.testing.assert_array_equal(bars(segs), np.array([0, 4, 8, 12, 16, 19, 22, 25, 28, 32, 36, 40]) * FAST)


def test_rebarring_and_visual_meters_still_start_bars():
    # A same-tempo line at an odd beat that holds complete bars re-bars (beat 17: bars 17, 21);
    # meter 1 at beat 11 holding two beats starts two one-beat bars (a visual meter is still a bar).
    segs = [Segment(0.0, FAST, 4), Segment(11 * FAST, FAST, 1), Segment(13 * FAST, FAST, 4),
            Segment(17 * FAST, FAST, 4)]
    np.testing.assert_array_equal(bars(segs, hi=25 * FAST),
                                  np.array([0, 4, 8, 11, 12, 13, 17, 21, 25]) * FAST)


@pytest.mark.parametrize('step', [FAST, NORMAL])
def test_resets_and_tempo_changes_holding_less_than_a_bar_still_start_bars(step):
    # A reset half a beat off the grid at beat 16.5 holding two beats, then a line back at 18.5;
    # a tempo change at beat 30.5 (to 0.8 of the step) holding two of its beats.
    c = 30.5 * step
    segs = [Segment(0.0, step, 4), Segment(16.5 * step, step, 4), Segment(18.5 * step, step, 4),
            Segment(c, 0.8 * step, 4), Segment(c + 1.6 * step, step, 4)]
    got = bars(segs, hi=40 * step)
    want = np.concatenate([np.arange(5) * 4 * step, [16.5 * step], 18.5 * step + np.array([0, 4, 8]) * step,
                           [c], c + 1.6 * step + np.array([0, 4]) * step])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


def _seven_eight(notation):
    if notation == 'resets':
        return [Segment(k * 1750.0, 500.0, 4) for k in range(8)]
    if notation == 'meter7':
        return [Segment(0.0, 250.0, 7)]
    out, t = [], 0.0
    for k in range(16):
        meter = 4 if k % 2 == 0 else 3
        out.append(Segment(t, 250.0, meter))
        t += meter * 250.0
    return out


@pytest.mark.parametrize('ref_n', ['resets', 'meter7', 'alternating'])
@pytest.mark.parametrize('pred_n', ['resets', 'meter7', 'alternating'])
def test_seven_eight_notations_keep_their_bars_and_score_each_other_one(ref_n, pred_n):
    heads = [k * 1750.0 + j * 250.0 for k in range(8) for j in range(7)]
    ref = Reference.from_segments(_seven_eight(ref_n), heads)
    s = score(ref, MusicMap(segments=tuple(_seven_eight(pred_n))))
    assert s.bl == {20.0: 1.0, 10.0: 1.0, 5.0: 1.0}
    want = np.arange(8) * 1750.0
    if ref_n == 'alternating':     # bars of 4 and 3 eighths, as notated
        want = np.sort(np.concatenate([want, want + 1000.0]))
    b = ref.timing.bar_starts()
    np.testing.assert_array_equal(b[(b >= 0) & (b < 8 * 1750.0)], want)
    if ref_n == pred_n or 'alternating' not in (ref_n, pred_n):
        assert s.bars_start == 1.0 and s.bars_precision == 1.0
