"""Dense targets and interim strata on fixtures built to break them (guardrail 10, design §4.2 test list)."""
from __future__ import annotations

import numpy as np
import pytest

from ensomi_n.data import strata
from ensomi_n.labels import read_set
from ensomi_n.targets import RecLabels, dense


def _ref(osu_writer, red, heads, name='a.osu'):
    return read_set([osu_writer(name, red, heads)], key='k')


def _peaks(y):
    return np.flatnonzero(y == 1.0)


def test_beat_targets_sit_on_lines_with_exact_offsets(osu_writer):
    heads = [1003.0 + 500.0 * k for k in range(40)]
    lab = RecLabels.from_reference(_ref(osu_writer, [(1003.0, 500.0)], heads))
    tg = dense(lab, 0.0, 3000)
    lines = lab.lines[(lab.lines > 0) & (lab.lines < 30000)]
    centre = np.round(lines / 10.0).astype(int)
    assert set(_peaks(tg['beat'])) >= set(centre[(lines >= lab.mapped[0]) & (lines <= lab.mapped[1])])
    j = centre[5]
    assert tg['beat'][j - 1] == 0.5 and tg['beat'][j + 1] == 0.5
    assert tg['beat_off'][j] == pytest.approx(lines[5] - 10.0 * j)
    assert tg['beat_offmask'][j - 1] and tg['beat_off'][j - 1] == pytest.approx(lines[5] - 10.0 * (j - 1))
    assert np.all(tg['level'][tg['mask']] == 1)                   # 120 BPM: notated = canonical


def test_mapped_range_masks_song_end(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(21)]               # last head 11000
    lab = RecLabels.from_reference(_ref(osu_writer, [(1000.0, 500.0)], heads))
    tg = dense(lab, 0.0, 2000)
    t = 10.0 * np.arange(2000)
    assert not tg['mask'][t > 11500 + 1].any() and tg['mask'][(t >= 1000) & (t <= 11500)].all()
    assert not tg['mask'][t < 500 - 1].any()


def test_meter7_at_240_equals_7_8_with_resets(osu_writer):
    heads = [500.0 + 250.0 * k for k in range(56)]
    fast = RecLabels.from_reference(_ref(osu_writer, [(500.0, 250.0, 7)], heads, 'fast.osu'))
    bar_ms = 7 * 250.0
    red = [(500.0 + bar_ms * b, 500.0, 4) for b in range(8)]
    reset = RecLabels.from_reference(_ref(osu_writer, red, heads, 'reset.osu'))
    a, b = dense(fast, 0.0, 1500), dense(reset, 0.0, 1500)
    t = 10.0 * np.arange(1500)
    m = (t >= fast.lo_ms - 5) & (t <= fast.hi_ms + 5)       # beyond the heads the lines are extrapolation
    np.testing.assert_array_equal(a['mask'], b['mask'])
    np.testing.assert_array_equal(a['beat'][m], b['beat'][m])
    np.testing.assert_array_equal(a['down'][m], b['down'][m])
    assert np.all(a['level'][a['mask']] == 2) and np.all(b['level'][b['mask']] == 1)


def test_fractional_boundary_reset_is_a_phase_jump(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(10)] + [5833.3 + 500.0 * k for k in range(10)]
    lab = RecLabels.from_reference(_ref(osu_writer, [(1000.0, 500.0), (5833.3, 500.0)], heads))
    tg = dense(lab, 0.0, 1100)
    j = int(round(5833.3 / 10))
    assert tg['phase'][j - 1] > 0.6 and tg['phase'][j + 1] < 0.1    # jumps to 0 at the reset
    assert tg['beat_off'][j] == pytest.approx(5833.3 - 10.0 * j)


def test_stretch_scales_every_label_time(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(30)]
    lab = RecLabels.from_reference(_ref(osu_writer, [(1000.0, 500.0)], heads))
    r = 1.1
    tg = dense(lab, 0.0, 2000, rate=r)
    peaks = 10.0 * _peaks(tg['beat'] * tg['mask'])
    expected = lab.lines[(lab.lines * r >= lab.mapped[0] * r) & (lab.lines * r <= lab.mapped[1] * r)] * r
    assert np.max(np.abs(peaks[:, None] - expected[None, :]).min(axis=1)) <= 5.0
    assert tg['logp'][tg['phase_mask']] == pytest.approx(np.log(0.5 * r))


def test_fold_flip_moves_targets_between_beat_and_half(osu_writer):
    # Just under and just over 160 BPM: the canonical level flips from 160 to 80. The 160-level
    # beat lines must reappear as the 80-level beat lines plus its half-beat lines.
    def heads_of(beat):
        heads = [1003.0 + beat * k for k in range(60)]           # off the frame half-points
        lab = RecLabels.from_reference(_ref(osu_writer, [(1003.0, beat)], heads, f'{beat}.osu'))
        tg = dense(lab, 0.0, 3000)
        m = tg['mask']
        return lab, set(_peaks(tg['beat'] * m)), set(_peaks(tg['half'] * m))
    below, beat_b, half_b = heads_of(375.0001)
    above, beat_a, half_a = heads_of(374.9999)
    assert np.median(below.line_beat) == pytest.approx(375.0001)         # 159.99996 BPM stays
    assert np.median(above.line_beat) == pytest.approx(749.9998)         # 160.00004 BPM folds to 80
    assert beat_a | half_a == beat_b and not (beat_a & half_a)


def test_interim_strata(osu_writer):
    heads = [1000.0 + 500.0 * k for k in range(100)]
    s0 = strata.flags(_ref(osu_writer, [(1000.0, 500.0)], heads, 's0.osu'))
    assert s0['strata'] == ['S0'] and not s0['hard']
    rs = strata.flags(_ref(osu_writer, [(1000.0, 500.0), (20250.0, 500.0)], heads, 'rs.osu'))
    assert rs['flags']['reset'] and 'S4' in rs['strata'] and 'S9' in rs['strata']
    m7 = strata.flags(_ref(osu_writer, [(1000.0, 500.0, 7)], heads, 'm7.osu'))
    assert 'S2' in m7['strata']
    red = [(1000.0 + 3000.0 * k, 500.0 * (1 + 0.01 * k)) for k in range(12)]
    tm = strata.flags(_ref(osu_writer, red, heads, 'tm.osu'))
    assert tm['flags']['tempo_mapped'] and 'S3' in tm['strata'] and 'S4' not in tm['strata']
    red2 = [(1000.0, 500.0), (21000.0, 500.0), (41000.0, 500.0)]  # redundant lines merge
    assert strata.flags(_ref(osu_writer, red2, heads, 'red.osu'))['strata'] == ['S0']
    lv = strata.flags(_ref(osu_writer, [(1000.0, 60000 / 80.5)], heads, 'lv.osu'))
    assert 'S5' in lv['strata']


def test_random_draws_are_independent_of_the_hard_draw():
    """val-n1's random half and R1-train's rest hold hard recordings in about the pool's share."""
    from ensomi_n.data.manifest import select
    recs = [dict(rec=f'r{i:04d}', role='n_train') for i in range(3000)]
    hard = {r['rec']: (i % 2 == 0) for i, r in enumerate(recs)}
    sel = select(recs, hard)
    assert len(sel['val_hard']) == 150 and all(hard[k] for k in sel['val_hard'])
    assert 50 <= sel['val_random_hard'] <= 100          # about half of 150, not 0
    assert not set(sel['val_random']) & set(sel['val_hard'])
    rest = sel['train'][len(sel['train_hard']):]
    assert 150 <= sum(hard[k] for k in rest) <= 450     # hard ones remain in the random rest
    assert len(set(sel['train'])) == 1600 and not set(sel['train']) & (set(sel['val_hard']) | set(sel['val_random']))


def test_reader_flags_mark_the_census_causes_and_spare_resets(osu_writer):
    """Pre-registration item 24: redundant lines, odd bars and tempo lines at fast notation, scroll art."""
    from ensomi_n.data.strata import reader_flags
    heads = list(np.arange(1000.0, 30000.0, 250.0))
    cases = {
        'clean': ([(0.0, 500.0)], []),
        'fast_even': ([(0.0, 333.333)], []),
        'fast_odd': ([(0.0, 333.333, 3)], ['fast_odd_bar']),
        'redundant': ([(0.0, 500.0), (10000.0, 500.0, 3)], ['redundant']),
        'reset': ([(0.0, 500.0), (10120.0, 500.0)], []),
        'tempo_fast': ([(0.0, 500.0), (10000.0, 333.333)], ['fast_tempo_line']),
        'tempo_slow': ([(0.0, 500.0), (10000.0, 450.0)], []),
    }
    for name, (red, want) in cases.items():
        ref = read_set([osu_writer(f'{name}.osu', red, heads)])
        assert reader_flags(ref) == want, name
