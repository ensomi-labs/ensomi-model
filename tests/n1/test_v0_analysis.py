"""scripts/n1/v0_analysis.py on charts with known answers: reader-neutral heads, R's reference, phase tolerance."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('ensomi_n')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'n1'))

import v0_analysis as va  # noqa: E402
from ensomi_n.eval.strata import set_strata  # noqa: E402
from ensomi_n.labels import read_osu, reference_from_charts  # noqa: E402
from ensomi_n.mapformat import MusicMap  # noqa: E402
from ensomi_n.timing import Segment  # noqa: E402


def _write(path: Path, red, heads) -> Path:
    lines = ['osu file format v14', '', '[General]', 'AudioFilename: a.mp3', 'Mode: 3', '', '[Metadata]',
             'Title:t', 'Version:v', '', '[Difficulty]', 'CircleSize:4', '', '[TimingPoints]']
    lines += [f'{o!r},{b!r},{m},1,0,100,1,0' for o, b, m in red]
    lines += ['', '[HitObjects]'] + [f'64,192,{t!r},1,0,0:0:0:0:' for t in heads]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _item(tmp_path, red, heads):
    path = _write(tmp_path / 'c.osu', red, heads)
    chart = read_osu(path)
    ref = reference_from_charts([chart], 'c')
    st = set_strata(ref, [chart], tags=(), tags_known=False, key='c')
    return dict(key='c', rec='r', item='i', literal=False, fast57=False, ref=ref, st=st, strata={})


def _map(*segs):
    return MusicMap(segments=tuple(Segment(float(o), float(b), int(m)) for o, b, m in segs), model={'name': 't'})


def _vals(rec, arm, test):
    return dict(zip(va.FIELDS, rec['arms'][arm][test]['all']))


def test_union_chance_counts_distinct_phases():
    # factor 1, 500 ms: four phases 0, 1/3, 1/2, 2/3 apart by at least 1/6 (83 ms > 40 ms) -> 4 x 0.08
    np.testing.assert_allclose(va.union_chance(np.array([500.0]), np.array([1])), [0.32])
    # factor 2 (compared at 250 ms): 1/2 x 2 = 0 again, so three phases -> 3 x 0.16
    np.testing.assert_allclose(va.union_chance(np.array([250.0]), np.array([2])), [0.48])
    # wide windows saturate at 1
    np.testing.assert_allclose(va.union_chance(np.array([100.0]), np.array([1])), [1.0])


def test_exclusion_and_outside():
    n = np.array([0.0, 500.0, 1000.0, 1500.0])
    r = np.array([0.0, 520.0, 1000.0, 1500.0])
    heads = np.array([0.0, 250.0, 750.0, 1250.0])
    changed = np.array([False, True, True, False])
    ex = va.exclusion(heads, changed, n, r)
    assert ex == [(0.0, 1000.0)]
    np.testing.assert_array_equal(va.outside(heads, ex), [False, False, False, True])


def test_same_lines_score_one_everywhere(tmp_path):
    beat = 500.0
    x = _item(tmp_path, [(1000.0, beat, 4)], 1000.0 + beat * np.arange(64))
    rec = va.item_record(x, {'same': _map((1000.0, beat, 4)), 'half': _map((1250.0, beat, 4))})
    assert rec['changed'] == 0 and rec['neutral'] == rec['heads'] == 64 and rec['r_mismatch'] == 0
    for test in va.TESTS:
        v = _vals(rec, 'same', test)
        assert v['bl'] == 1.0 and v['pt'] == 1.0 and v['nz'] == 0.0
    # half a beat off at 120 BPM: BL-20 0, phase-tolerant 1 through the 1/2 shift, chance 4 x 40 / 500
    for test in va.TESTS:
        v = _vals(rec, 'half', test)
        assert v['bl'] == 0.0 and v['pt'] == 1.0 and v['nz'] == 1.0
        assert abs(v['ch'] - 0.08) < 1e-9 and abs(v['chp'] - 0.32) < 1e-9


def test_fast_notation_other_beat_is_forgiven_only_by_phase_tolerance(tmp_path):
    beat = 60000.0 / 180.0          # canonical 90 BPM, one notated beat = half a canonical beat
    x = _item(tmp_path, [(1000.0, beat, 4)], 1000.0 + beat * np.arange(96))
    rec = va.item_record(x, {'other': _map((1000.0 + beat, beat, 4))})
    assert rec['changed'] == 0
    v = _vals(rec, 'other', 'N')
    assert v['bl'] == 0.0 and v['pt'] == 1.0


def test_fast_five_four_separates_the_readers(tmp_path):
    # 5/4 at 220 BPM, one red line: N restarts canonical lines every bar (2 + 2 + 1 notated beats), R runs a
    # continuous half-time pulse. Heads on every notated beat.
    beat = 60000.0 / 220.0
    x = _item(tmp_path, [(1000.0, beat, 5)], 1000.0 + beat * np.arange(200))
    n_like = _map((1000.0, beat, 5))
    r_like = _map((1000.0, 2 * beat, 4))
    rec = va.item_record(x, {'n': n_like, 'r': r_like})
    assert rec['r_mismatch'] == 0
    assert rec['changed'] > 0 and rec['neutral'] < rec['heads']
    assert _vals(rec, 'n', 'N')['bl'] == 1.0 and _vals(rec, 'r', 'R')['bl'] == 1.0
    assert _vals(rec, 'r', 'N')['bl'] < 0.8 and _vals(rec, 'n', 'R')['bl'] < 0.8
    # one segment: a per-bar restart is not a constant phase, so phase tolerance does not make it pass
    assert rec['n_segments'] == 1 and _vals(rec, 'r', 'N')['pt'] < 0.9
    # on the neutral heads both readings pass
    assert _vals(rec, 'n', 'Nn')['bl'] == 1.0 and _vals(rec, 'r', 'Nn')['bl'] == 1.0


def test_a_tempo_change_gives_each_segment_its_own_shift(tmp_path):
    b1, b2 = 500.0, 400.0
    t2 = 1000.0 + 32 * b1
    heads = np.concatenate([1000.0 + b1 * np.arange(32), t2 + b2 * np.arange(40)])
    x = _item(tmp_path, [(1000.0, b1, 4), (t2, b2, 4)], heads)
    # first segment on time, the second a third of a beat late: BL-20 passes the 32 heads of the first segment
    # and the one on the line at the change (the prediction's first grid still has a beat there); the
    # second segment's best shift (2/3, a third early) passes all of its heads but that one
    rec = va.item_record(x, {'p': _map((1000.0, b1, 4), (t2 + b2 / 3.0, b2, 4))})
    v = _vals(rec, 'p', 'N')
    assert rec['n_segments'] == 2
    assert abs(v['bl'] - 33 / 72) < 1e-6 and abs(v['pt'] - 71 / 72) < 1e-6
