"""The canonical phase rule (reader fix, 2026-10-19) and the level term on beat lists, on inputs that can break them."""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.eval.measures import _pred_beat_length
from ensomi_n.mapformat import MusicMap
from ensomi_n.timing import Reference, Segment, Timing

FAST = 300.0                      # 200 BPM notated, canonical 100 BPM


def lines(segments, lo=0.0, hi=12000.0):
    t = Timing.from_segments(segments, lo, hi)
    c = t.canonical()
    m = (c.t >= lo) & (c.t <= hi)
    return c.t[m], c.factor[m]


def test_barline_art_changes_nothing():
    # 4/4 at a notated 200 BPM. Barline art: lines on the grid at the same tempo, at odd and even
    # beats, holding less than a bar (meters 4 and 1, one with omit-first-barline), and kiai lines
    # on downbeats holding whole bars.
    plain = [Segment(0.0, FAST, 4)]
    art = [Segment(0.0, FAST, 4)]
    for beat, meter, bar in ((9, 4, True), (10, 4, True), (11, 1, True), (13, 4, False), (15, 4, True)):
        art.append(Segment(beat * FAST, FAST, meter, bar))
    art += [Segment(16 * FAST, FAST, 4), Segment(23 * FAST, FAST, 4), Segment(24 * FAST, FAST, 4),
            Segment(32 * FAST, FAST, 4)]
    t0, f0 = lines(plain)
    t1, f1 = lines(art)
    np.testing.assert_allclose(t1, t0, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(f1, f0)
    np.testing.assert_allclose(t0, np.arange(len(t0)) * 2 * FAST)
    assert np.all(f0 == 1)


@pytest.mark.parametrize('step', [60000.0 / 180.0, FAST])
def test_triple_meter_at_fast_notation_keeps_a_continuous_phase(step):
    t, f = lines([Segment(0.0, step, 3)], hi=40.5 * step)
    every_bar = lines([Segment(k * 3 * step, step, 3) for k in range(14)], hi=40.5 * step)
    np.testing.assert_allclose(t, np.arange(len(t)) * 2 * step, rtol=0, atol=1e-9)
    assert np.all(f[:len(t)] == 2)       # free: a 1.5-beat bar holds no canonical pulse
    np.testing.assert_allclose(every_bar[0], t, rtol=0, atol=1e-9)


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


SEVEN_EIGHT_HEADS = [k * 1750.0 + j * 250.0 for k in range(8) for j in range(7)]


@pytest.mark.parametrize('ref_n', ['resets', 'meter7', 'alternating'])
@pytest.mark.parametrize('pred_n', ['resets', 'meter7', 'alternating'])
def test_seven_eight_notations_score_each_other_one(ref_n, pred_n):
    ref = Reference.from_segments(_seven_eight(ref_n), SEVEN_EIGHT_HEADS)
    s = score(ref, MusicMap(segments=tuple(_seven_eight(pred_n))))
    assert s.bl == {20.0: 1.0, 10.0: 1.0, 5.0: 1.0}
    assert s.offset_ms == 0.0


def _waltz_ref(n_bars=24):
    heads = np.arange(3 * n_bars) * FAST
    return Reference.from_segments([Segment(0.0, FAST, 3)], heads)


def test_free_bars_accept_either_phase_and_nothing_else():
    ref = _waltz_ref()
    n = 3 * 24
    beats = np.arange(n) * FAST
    early = np.arange(-1, n) * FAST
    for pred in (MusicMap(beats=early),                          # a beat list whose count starts a beat early
                 MusicMap(beats=early, downbeats=np.arange(1, n + 1, 3)),   # ... with the right bars
                 MusicMap(segments=(Segment(0.0, 2 * FAST, 4),)),  # canonical grid, phase of bar 1
                 MusicMap(segments=(Segment(FAST, 2 * FAST, 4),)),  # canonical grid, the other phase
                 MusicMap(beats=beats, downbeats=np.arange(1, n, 3))):  # bars a beat late
        s = score(ref, pred)
        assert s.bl20 == 1.0
    assert score(ref, MusicMap(beats=beats + FAST / 2)).bl20 == 0.0   # half a notated beat
    s = score(ref, MusicMap(beats=beats))
    assert s.chance[20.0] == pytest.approx(40.0 / FAST)
    assert s.free_share == 1.0 and s.free_share_pred == 0.0


def test_a_free_prediction_against_a_set_phase_is_marked():
    # 6/4 at a notated 200 BPM sets the phase; a 3/4 reading starting at the odd bar is free.
    ref = Reference.from_segments([Segment(0.0, FAST, 6)], np.arange(72) * FAST)
    s = score(ref, MusicMap(segments=(Segment(3 * FAST, FAST, 3),)))
    assert s.bl20 == 1.0
    assert s.free_share_pred == pytest.approx(s.free_share) and s.free_share > 0.9
    assert score(ref, MusicMap(segments=(Segment(FAST, FAST, 6),))).bl20 == 0.0     # set phase, one beat off


def test_free_null_is_exact():
    # Guardrail 2: on a free reference, whole-grid shifts pass with chance 2 tau over the notated beat.
    ref = _waltz_ref()
    n = 300
    vals = [score(ref, MusicMap(segments=(Segment((k + 0.5) / n * FAST, FAST, 3),))).bl20 for k in range(n)]
    assert np.mean(vals) == pytest.approx(40.0 / FAST, abs=1.0 / n)


def test_fractional_boundary_and_song_end_with_a_free_segment():
    b_a = 60000.0 / 128.3
    c0 = 10.5 * b_a
    segs = [Segment(0.0, b_a, 4), Segment(c0, FAST, 3), Segment(c0 + 7.5 * FAST, b_a, 4)]
    heads = np.concatenate([np.arange(11) * b_a, c0 + np.arange(8) * FAST, c0 + 7.5 * FAST + np.arange(9) * b_a])
    ref = Reference.from_segments(segs, heads)
    expected = np.concatenate([np.arange(11) * b_a, c0 + np.arange(4) * 2 * FAST, c0 + 7.5 * FAST + np.arange(9) * b_a])
    np.testing.assert_allclose(ref.lines().t, expected, rtol=0, atol=1e-9)
    s = score(ref, MusicMap(segments=tuple(segs)))
    assert s.bl == {20.0: 1.0, 10.0: 1.0, 5.0: 1.0}
    assert score(ref, MusicMap(beats=heads, segments=tuple(segs))).bl20 == 1.0


@pytest.mark.parametrize('bpm', [123.0, 131.3, 97.0, 120.0])
def test_level_reads_a_frame_quantised_beat_list(bpm):
    step = 60000.0 / bpm
    beats = np.round((1000.0 + np.arange(240) * step) / 20.0) * 20.0      # raw tracker beats, 20 ms frames
    # Resets every 3 beats split the reference into short segments of one tempo.
    segs = [Segment(1000.0 + k * 3 * step, step, 4) for k in range(80)]
    ref = Reference.from_segments(segs, 1000.0 + np.arange(240) * step)
    s = score(ref, MusicMap(beats=beats))
    assert s.bl20 == 1.0 and s.level == 1.0
    if bpm in (123.0, 131.3):   # the median interval, E0's reading, is off by more than the 0.5 % tolerance
        assert abs(np.median(np.diff(beats)) / step - 1.0) > 0.005


def test_level_reads_a_jittered_beat_list():
    rng = np.random.default_rng(2)
    for bpm in (155.0, 123.0, 87.0):
        step = 60000.0 / bpm
        exact = np.arange(200) * step
        ref = Reference.from_segments([Segment(0.0, step, 4)], exact)
        assert score(ref, MusicMap(beats=exact + rng.uniform(-3.0, 3.0, 200))).level == 1.0


def test_level_reads_ramps_exact_and_on_frames():
    tempos = np.linspace(100.0, 140.0, 33)
    steps = 60000.0 / tempos
    starts = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    segs = [Segment(float(a), float(b)) for a, b in zip(starts, steps)]
    heads = np.concatenate([starts, starts[-1] + np.arange(1, 21) * steps[-1]])
    ref = Reference.from_segments(segs, heads)
    assert score(ref, MusicMap(beats=heads)).level == 1.0
    assert score(ref, MusicMap(beats=np.round(heads / 20.0) * 20.0 + 7.0)).level == 1.0
    # One beat of the ramp, read on frames, is within 0.5 % of its own tempo.
    t = Timing.from_beats(np.round(heads / 20.0) * 20.0)
    assert abs(_pred_beat_length(t, starts[16], starts[17], 0.0, starts[-1]) / steps[16] - 1.0) <= 0.005
