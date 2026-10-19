"""Canonical beat lines on inputs that can break them: fractional boundaries, folds, odd meters."""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n.timing import Reference, Segment, Timing, fold_of


@pytest.mark.parametrize('bpm, fold', [(80.0, 0), (159.99, 0), (160.0, -1), (240.0, -1), (319.9, -1),
                                       (320.0, -2), (79.99, 1), (60.0, 1), (40.0, 1), (39.99, 2), (20.0, 2),
                                       (1000.0, -3), (128.0, 0), (93.75, 0)])
def test_fold_moves_tempo_into_80_160(bpm, fold):
    assert int(fold_of(60000.0 / bpm)[0]) == fold
    assert 80.0 <= bpm * 2.0 ** fold < 160.0


def test_fractional_boundaries_non_integer_tempi_and_both_folds():
    b_a = 60000.0 / 128.3
    b_b = 60000.0 / 97.0
    b_c = 60000.0 / 175.0         # fast: canonical 87.5 BPM
    a0 = 0.0
    b0 = a0 + 10.5 * b_a          # half a beat into A's eleventh beat
    c0 = b0 + 7.25 * b_b
    d0 = c0 + 6.5 * b_c           # slow segment starts between C's canonical lines
    e0 = d0 + 3700.0              # D at 60 BPM ends 200 ms past a canonical line
    segments = [Segment(a0, b_a), Segment(b0, b_b), Segment(c0, b_c), Segment(d0, 1000.0), Segment(e0, b_a)]
    hi = e0 + 20 * b_a
    heads = np.concatenate([[a0], np.arange(1, 21) * b_a + e0])
    ref = Reference.from_segments(segments, heads, lo_ms=a0, hi_ms=hi)
    got = ref.lines().t
    expected = np.concatenate([
        a0 + np.arange(11) * b_a,
        b0 + np.arange(8) * b_b,
        c0 + np.arange(4) * 2 * b_c,
        d0 + np.arange(8) * 500.0,
        e0 + np.arange(21) * b_a,
    ])
    assert len(got) == len(expected) == 52
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)
    # Inside each segment lines sit one canonical beat apart, within [80, 160) BPM.
    spacing = np.diff(got)
    assert spacing.max() <= 750.0 + 1e-9
    assert np.all(spacing[:10] == pytest.approx(b_a))


def test_non_integer_tempo_does_not_drift():
    step = 60000.0 / 137.25
    ref = Reference.from_segments([Segment(1234.5, step)], [1234.5, 1234.5 + 2000 * step])
    lines = ref.lines().t
    assert len(lines) == 2001
    assert lines[2000] == pytest.approx(1234.5 + 2000 * step, rel=0, abs=1e-6)


SEVEN_EIGHT_LINES = np.sort(np.concatenate([k * 1750.0 + np.array([0.0, 500.0, 1000.0, 1500.0])
                                            for k in range(8)]))


def _seven_eight(notation: str) -> list[Segment]:
    if notation == 'resets':      # 4/4 at quarter = 120 with a reset on every downbeat
        return [Segment(k * 1750.0, 500.0, 4) for k in range(8)]
    if notation == 'meter7':      # eighth = 240, meter 7
        return [Segment(0.0, 250.0, 7)]
    if notation == 'alternating':  # eighth = 240, bars of 4 and 3 eighths
        out, t = [], 0.0
        for k in range(16):
            meter = 4 if k % 2 == 0 else 3
            out.append(Segment(t, 250.0, meter))
            t += meter * 250.0
        return out
    raise ValueError(notation)


@pytest.mark.parametrize('notation', ['resets', 'meter7', 'alternating'])
def test_seven_eight_notations_give_the_same_lines(notation):
    timing = Timing.from_segments(_seven_eight(notation), 0.0, 8 * 1750.0 - 250.0)
    lines = timing.canonical().t
    inside = lines[(lines >= 0) & (lines < 8 * 1750.0)]
    np.testing.assert_array_equal(inside, SEVEN_EIGHT_LINES)


def test_seven_eight_downbeats():
    def bars(n):
        t = Timing.from_segments(_seven_eight(n), 0.0, 8 * 1750.0 - 250.0).bar_starts()
        return t[(t >= 0) & (t < 8 * 1750.0)]
    np.testing.assert_array_equal(bars('resets'), np.arange(8) * 1750.0)
    np.testing.assert_array_equal(bars('meter7'), np.arange(8) * 1750.0)
    np.testing.assert_array_equal(bars('alternating'), np.sort(np.concatenate([np.arange(8) * 1750.0,
                                                                               np.arange(8) * 1750.0 + 1000.0])))


def test_bar_shorter_than_a_canonical_beat_does_not_restart():
    # Meter 1 at 240 BPM (a meter set for visual effect): lines stay one canonical beat apart.
    lines = Timing.from_segments([Segment(0.0, 250.0, 1)], 0.0, 4000.0).canonical().t
    inside = lines[(lines >= 0) & (lines <= 4000.0)]
    np.testing.assert_array_equal(inside, np.arange(9) * 500.0)


def test_triple_meter_at_fast_notation_runs_on_and_is_free():
    # 3/4 at 200 BPM (reader fix, 2026-10-19): a 1.5-beat bar fixes no canonical phase, so the count
    # runs on from the red line, as R reads it, and the lines are marked free (refined when scored).
    c = Timing.from_segments([Segment(0.0, 300.0, 3)], 0.0, 3600.0).canonical()
    inside = (c.t >= 0) & (c.t <= 3600.0)
    np.testing.assert_array_equal(c.t[inside], [0, 600, 1200, 1800, 2400, 3000, 3600])
    assert np.all(c.factor[inside] == 2)


def test_backward_extension_keeps_the_anchor():
    timing = Timing.from_segments([Segment(5000.0, 250.0, 7)], 0.0, 9000.0)
    lines = timing.canonical().t
    before = lines[(lines >= 5000.0 - 2 * 1750.0) & (lines < 5000.0)]
    np.testing.assert_array_equal(before, np.sort(np.concatenate(
        [5000.0 - k * 1750.0 + np.array([0.0, 500.0, 1000.0, 1500.0]) for k in (1, 2)])))
    np.testing.assert_array_equal(timing.bar_starts()[timing.bar_starts() < 5000.0][-2:], [1500.0, 3250.0])


def test_jittered_beat_list_keeps_its_fold():
    rng = np.random.default_rng(1)
    for bpm, keep_every in ((155.0, 1), (165.0, 2)):
        step = 60000.0 / bpm
        beats = np.arange(200) * step + rng.uniform(-3.0, 3.0, 200)
        lines = Timing.from_beats(beats).canonical().t
        np.testing.assert_array_equal(lines, beats[::keep_every])


def test_beat_list_on_a_segment_grid_matches_the_segments():
    segments = [Segment(100.0, 400.0, 4), Segment(100.0 + 9.5 * 400.0, 1250.0, 4)]
    from_segments = Timing.from_segments(segments, 100.0, 20000.0)
    inside = (from_segments.beats >= 100.0) & (from_segments.beats <= 20000.0)
    beats = from_segments.beats[inside]
    from_beats = Timing.from_beats(beats, segments=segments)
    a = from_segments.canonical().t
    b = from_beats.canonical().t
    a = a[(a >= beats[0]) & (a < beats[-1])]
    b = b[(b >= beats[0]) & (b < beats[-1])]
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(from_beats.downbeat, from_segments.downbeat[inside])


def test_beats_must_increase():
    with pytest.raises(ValueError, match='increase strictly'):
        Timing.from_beats([0.0, 500.0, 500.0])


def test_head_weights_split_between_lines():
    ref = Reference.from_segments([Segment(0.0, 500.0)], [0.0, 125.0, 1000.0])
    w = ref.weights()
    np.testing.assert_allclose(w, [1.75, 0.25, 1.0])
    assert w.sum() == pytest.approx(3.0)
