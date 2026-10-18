"""Each measure on a fixture with an absolute expected value."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ensomi_n.eval import equivariance, score, score_beats
from ensomi_n.eval.measures import level_class, subdivision_classes, weighted_median
from ensomi_n.labels import read_set
from ensomi_n.mapformat import MusicMap
from ensomi_n.timing import Reference, Segment, Timing

B = 500.0


def grid(n=32, step=B, meter=4, start=0.0) -> Reference:
    return Reference.from_segments([Segment(start, step, meter)], start + np.arange(n) * step)


def red(*segments) -> MusicMap:
    return MusicMap(segments=tuple(Segment(*s) for s in segments))


def test_reference_against_itself_scores_perfectly():
    s = score(grid(), red((0.0, B, 4)))
    assert s.bl == {20.0: 1.0, 10.0: 1.0, 5.0: 1.0}
    assert s.bl_beat == {20.0: 1.0, 10.0: 1.0, 5.0: 1.0}
    assert s.offset_ms == 0.0 and s.level == 1.0
    assert (s.bars_start, s.bars_same, s.bars_precision) == (1.0, 1.0, 1.0)
    assert s.edits == {'offset': 0.0, 'retime': 0.0, 'delete': 0.0, 'level': 0.0, 'meter': 0.0}
    assert s.edits_per_min == 0.0 and s.redo_s_per_min == 0.0 and s.hidden_per_min == 0.0
    assert s.n_lines == 32 and s.heads == 32.0


@pytest.mark.parametrize('shift, bl20, bl10, bl5', [(15.0, 1.0, 0.0, 0.0), (-15.0, 1.0, 0.0, 0.0),
                                                     (20.0, 1.0, 0.0, 0.0), (21.0, 0.0, 0.0, 0.0),
                                                     (7.0, 1.0, 1.0, 0.0), (B / 2, 0.0, 0.0, 0.0),
                                                     (B / 3, 0.0, 0.0, 0.0)])
def test_whole_grid_shift_counts_against_bl(shift, bl20, bl10, bl5):
    s = score(grid(), red((shift, B, 4)))
    assert s.bl == {20.0: bl20, 10.0: bl10, 5.0: bl5}
    expected_o = shift if abs(shift) < B / 2 else -B / 2
    assert s.offset_ms == pytest.approx(expected_o)
    if abs(shift) < B / 2:
        assert s.bl_o == {10.0: 1.0, 5.0: 1.0}
    assert s.edits['offset'] == (0.0 if abs(shift) <= 10 else 1.0)


def test_half_and_double_time_are_level_errors_only():
    half = score(grid(), red((0.0, 2 * B, 2)))
    assert half.bl20 == 1.0 and half.level == 0.0 and half.level_classes['octave'] == 1.0
    assert half.edits['level'] == 1.0 and half.edits['retime'] == 0.0
    double = score(grid(), red((0.0, B / 2, 8)))
    assert double.bl20 == 1.0 and double.level_classes['octave'] == 1.0 and double.bars_same == 1.0


def test_double_time_phase_is_fixed_by_its_anchor():
    # A double-time map whose segment starts on an off-beat has its canonical lines on the off-beats.
    s = score(grid(), red((-B / 2, B / 2, 8)))
    assert s.bl20 == 0.0 and s.level_classes['octave'] == 1.0


def test_song_end_and_partial_prediction():
    ref = grid(40)
    first_half = MusicMap(beats=np.arange(20) * B)
    s = score(ref, first_half)
    assert s.bl20 == 0.5 and s.bl_beat[20.0] == 0.5
    # Anything after the last head is not scored.
    rng = np.random.default_rng(0)
    tail = np.sort(39 * B + 1000.0 + rng.uniform(0, 30000, 50))
    assert score(ref, MusicMap(beats=np.concatenate([np.arange(40) * B, tail]))).bl20 == 1.0


def test_empty_prediction():
    for pred in (None, MusicMap(beats=[])):
        s = score(grid(), pred)
        assert s.bl == {20.0: 0.0, 10.0: 0.0, 5.0: 0.0}
        assert math.isnan(s.offset_ms)
        assert s.level == 0.0 and s.level_classes['missing'] == 1.0
        assert s.bars_start == 0.0 and s.bars_same == 0.0 and not s.pred_has_bars
        assert s.edits['offset'] == 1.0 and s.edits['retime'] == 1.0 and s.edits['meter'] == 7.0


def test_several_segments_fractional_boundary_scores_one(osu_writer):
    b_a, b_b = 60000.0 / 128.3, 60000.0 / 97.0
    b0 = 10.5 * b_a
    c0 = b0 + 7.25 * b_b
    reds = [(0.0, b_a, 4), (b0, b_b, 3), (c0, 60000.0 / 175.0, 7), (c0 + 4000.0, 1000.0, 4)]
    heads = [0.0, b0, c0, c0 + 4000.0, c0 + 9000.0]
    ref = read_set([osu_writer('ref.osu', reds, heads=heads)])
    s = score(ref, MusicMap.from_osu(osu_writer('pred.osu', reds)))
    assert s.bl == {20.0: 1.0, 10.0: 1.0, 5.0: 1.0}
    assert s.level == 1.0 and s.bars_same == 1.0 and sum(s.edits.values()) == 0.0
    # The same timing as a bare beat list, one segment per tempo dropped: still every line.
    beats = ref.timing.beats[(ref.timing.beats >= -1) & (ref.timing.beats <= c0 + 9001.0)]
    assert score(ref, MusicMap(beats=beats, segments=ref.segments)).bl20 == 1.0


def test_seven_eight_notations_score_each_other(osu_writer):
    heads = [k * 1750 + j * 250 for k in range(8) for j in range(7)]
    resets = read_set([osu_writer('a.osu', [(k * 1750, 500.0, 4) for k in range(8)], heads=heads)])
    meter7 = MusicMap.from_osu(osu_writer('b.osu', [(0, 250.0, 7)]))
    s = score(resets, meter7)
    assert s.bl20 == 1.0 and s.bars_start == 1.0 and s.bars_same == 1.0 and s.reset_recall == 1.0
    lines, t = [], 0
    for k in range(16):
        meter = 4 if k % 2 == 0 else 3
        lines.append((t, 250.0, meter))
        t += meter * 250
    alternating = MusicMap.from_osu(osu_writer('c.osu', lines))
    s = score(resets, alternating)
    assert s.bl20 == 1.0 and s.bars_start == 1.0 and s.bars_same == 0.0
    flat = score(resets, red((0.0, 500.0, 4)))
    assert flat.reset_recall == 0.0 and flat.bl20 < 1.0


def test_ramp_of_red_lines():
    tempos = np.linspace(100.0, 140.0, 33)
    steps = 60000.0 / tempos
    starts = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    segs = [Segment(float(a), float(b)) for a, b in zip(starts, steps)]
    end = starts[-1] + steps[-1] * 20
    heads = np.concatenate([starts, starts[-1] + np.arange(1, 21) * steps[-1]])
    ref = Reference.from_segments(segs, heads)
    np.testing.assert_allclose(ref.lines().t, heads, rtol=0, atol=1e-9)
    assert score(ref, MusicMap(segments=tuple(segs))).bl20 == 1.0
    assert score(ref, MusicMap(beats=heads)).bl20 == 1.0
    assert score(ref, MusicMap(beats=heads)).level == 1.0
    one_tempo = score(ref, red((0.0, float(np.mean(steps)), 4)))
    assert one_tempo.bl20 < 0.5 and one_tempo.level_classes['correct'] < 0.5
    assert end > 0


def test_head_weighting_and_offset_median():
    heads = np.concatenate([np.arange(16) * B, np.arange(16) * B + B / 2, np.arange(16, 32) * B])
    ref = Reference.from_segments([Segment(0.0, B, 4)], heads)
    w = ref.weights()
    np.testing.assert_allclose(w, [1.5] + [2.0] * 15 + [1.5] + [1.0] * 15)
    # The second red line, 100 ms late, starts before line 16: lines 0-15 pass, 16-31 fail.
    s = score(ref, red((0.0, B, 4), (15 * B + 100.0, B, 4)))
    assert s.bl20 == pytest.approx(31.5 / 48.0) and s.bl_beat[20.0] == 0.5
    assert s.offset_ms == 0.0
    # Mapper work: one segment re-timed, the late red line deleted, the four scored bars from
    # 6000 ms on unmatched (6000-8000 ends off a predicted bar start; the bar holding the last
    # head ends past it and is not scored), four failing bars hidden.
    assert s.edits == {'offset': 0.0, 'retime': 1.0, 'delete': 1.0, 'level': 0.0, 'meter': 4.0}
    assert s.redo_s_per_min == 0.0
    assert s.hidden_share == 1.0
    assert s.hidden_per_min == pytest.approx(4 / (31 * B / 60000.0))
    assert s.edits_per_min == pytest.approx(6 / (31 * B / 60000.0))


def test_offset_median_splits_ties_at_the_midpoint():
    assert weighted_median([-15.0, 15.0], [1.0, 1.0]) == 0.0
    assert weighted_median([-15.0, 15.0, 15.0], [1.0, 1.0, 1.0]) == 15.0
    beats = np.arange(32) * B + np.where(np.arange(32) % 2 == 0, -15.0, 15.0)
    assert score(grid(), MusicMap(beats=beats)).offset_ms == 0.0


@pytest.mark.parametrize('ratio, cls', [(1.0, 'correct'), (1.004, 'correct'), (2.0, 'octave'), (0.5, 'octave'),
                                        (4.0, 'octave'), (1.5, 'family'), (0.75, 'family'), (1.25, 'family'),
                                        (1.3, 'other'), (1.01, 'other'), (math.nan, 'missing')])
def test_level_classes(ratio, cls):
    assert level_class(ratio) == cls


def test_level_on_a_prediction_at_three_halves():
    s = score(grid(), red((0.0, B * 2 / 3, 4)))
    assert s.level_classes['family'] == 1.0 and s.edits['level'] == 1.0


def test_bars_against_a_three_beat_prediction():
    s = score(grid(), red((0.0, B, 3)))
    assert s.bars_start == pytest.approx(3 / 8)
    assert s.bars_same == 0.0
    assert s.bars_precision == pytest.approx(3 / 11)
    assert s.edits['meter'] == 7.0 and s.bl20 == 1.0


def test_downbeat_on_every_beat_earns_no_bars():
    beats = np.arange(32) * B
    s = score(grid(), MusicMap(beats=beats, downbeats=np.arange(32)))
    assert s.bars_start == 1.0 and s.bars_same == 0.0 and s.bars_precision == 0.25
    bare = score(grid(), MusicMap(beats=beats))
    assert bare.bars_start == 0.0 and bare.bars_same == 0.0 and not bare.pred_has_bars
    assert bare.edits['meter'] == 7.0


def test_long_tempo_change_boundaries():
    ref = Reference.from_segments([Segment(0.0, 500.0), Segment(20000.0, 400.0)],
                                  np.concatenate([np.arange(40) * 500.0, 20000.0 + np.arange(50) * 400.0]))
    s = score(ref, red((0.0, 500.0), (20000.0, 400.0)))
    assert (s.boundary_recall, s.boundary_precision) == (1.0, 1.0)
    late = score(ref, red((0.0, 500.0), (20300.0, 400.0)))
    assert (late.boundary_recall, late.boundary_precision) == (1.0, 1.0)
    too_late = score(ref, red((0.0, 500.0), (20500.0, 400.0)))
    assert (too_late.boundary_recall, too_late.boundary_precision) == (0.0, 0.0)
    flat = score(ref, red((0.0, 500.0)))
    assert flat.boundary_recall == 0.0 and math.isnan(flat.boundary_precision)
    assert math.isnan(score(grid(), red((0.0, B))).boundary_recall)


def test_subdivision_classes_and_the_issue_32_beat():
    step = 60000.0 / 128.0
    timing = Timing.from_segments([Segment(0.0, step)], 0.0, 10 * step)
    offsets = {0: [0, 1 / 4, 1 / 2], 1: [0, 1 / 3, 2 / 3], 2: [0, 1 / 2], 3: [0, 1 / 6, 1 / 8], 4: [0, 1 / 5],
               5: [0, 1 / 12], 6: [1 / 16, 3 / 4, 1 / 3], 7: [0]}
    heads = [k * step + x * step for k, xs in offsets.items() for x in xs]
    beats, divisors, classes = subdivision_classes(timing, heads)
    np.testing.assert_allclose(timing.beats[beats] / step, [0, 1, 2, 3, 4, 5, 6])
    assert divisors.tolist() == [4, 3, 2, 24, 0, 12, 48]
    assert classes.tolist() == ['duple', 'triple', 'plain', 'mixed', 'finer', 'mixed', 'mixed']
    ref = Reference.from_segments([Segment(0.0, step)], heads)
    probs = {'duple': [0, 1], 'triple': [1, 0], 'plain': [0, 0], 'mixed': [1, 1], 'finer': [1, 1]}
    sub = [probs[c] for c in classes] + [[0, 0]] * 3
    pred = MusicMap(beats=np.arange(10) * step, subdivision=sub)
    s = score(ref, pred)
    # Six of seven agree: 'finer' has no probability form and reads as mixed.
    assert s.subdivision_accuracy == pytest.approx(6 / 7)
    own = score(ref, MusicMap(segments=(Segment(0.0, step),), heads=heads))
    assert own.subdivision_accuracy == 1.0 and own.subdivision_kappa == 1.0


def test_calibration_and_confidence():
    beats = np.concatenate([np.arange(16) * B, np.arange(16, 32) * B + 100.0])
    sure = MusicMap(beats=beats, confidence=np.r_[np.ones(16), np.zeros(16)])
    s = score(grid(), sure)
    assert s.ece == 0.0 and s.auc == 1.0
    assert s.confident_fail == 0.0 and s.confident_predicted_fail == 0.0
    assert s.check_load_s_per_min > 0 and s.hidden_per_min == 0.0
    flat = score(grid(), MusicMap(beats=beats, confidence=np.full(32, 0.5)))
    assert flat.ece == 0.0 and flat.auc == 0.5 and math.isnan(flat.confident_fail)
    wrong = score(grid(), MusicMap(beats=beats, confidence=np.r_[np.zeros(16), np.ones(16)]))
    assert wrong.ece == 1.0 and wrong.auc == 0.0 and wrong.confident_fail == 1.0
    assert wrong.hidden_share == 1.0
    assert math.isnan(score(grid(), MusicMap(beats=beats)).ece)


def test_stratum_spans_restrict_measures_but_not_the_offset():
    heads = np.concatenate([np.arange(16) * B, np.arange(16) * B + B / 2, np.arange(16, 32) * B])
    ref = Reference.from_segments([Segment(0.0, B, 4)], heads)
    pred = red((0.0, B, 4), (15 * B + 100.0, B, 4))
    late = score(ref, pred, spans=[(16 * B, 31 * B)])
    assert late.bl20 == 0.0 and late.offset_ms == 0.0 and late.heads == 16.0
    early = score(ref, pred, spans=[(0.0, 15 * B)])
    assert early.bl20 == 1.0 and early.heads == pytest.approx(31.0)


def test_decoded_clock_needs_the_measured_constant():
    m = MusicMap(beats=np.arange(32) * B + 20.0, clock='decoded')
    with pytest.raises(ValueError, match='decoded'):
        score(grid(), m)
    s = score(grid(), m, decoded_to_osu_ms=-20.0)
    assert s.bl20 == 1.0 and s.offset_ms == 0.0


def test_rate_edit_equivariance():
    src = MusicMap(beats=np.arange(64) * B + 37.0, downbeats=np.arange(0, 64, 4))
    edit = MusicMap(beats=(np.arange(64) * B + 37.0) / 1.1, downbeats=np.arange(0, 64, 4))
    s = equivariance(src, edit, 1.1)
    assert s.bl20 == 1.0 and s.offset_ms == pytest.approx(0.0) and s.bars_same == 1.0
    drifted = MusicMap(beats=(np.arange(64) * B + 37.0) / 1.1 + 30.0)
    assert equivariance(src, drifted, 1.1).bl20 == 0.0


def test_score_beats_without_heads_weighs_every_line():
    ref = np.arange(20) * B
    s = score_beats(ref, ref + 15.0)
    assert s.bl20 == 1.0 and s.bl10 == 0.0 and s.offset_ms == pytest.approx(15.0)
    assert math.isnan(s.heads) and s.n_scored == 20


def test_swapped_arguments_are_refused():
    with pytest.raises(TypeError, match='reference comes first'):
        score(red((0.0, B, 4)), grid())
