"""Mapper-to-mapper spread (`ensomi_n.eval.spread`) on synthetic pairs with known answers.

Plain asserts and no fixtures, so the module also runs without pytest.
"""
from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np

from ensomi_n.eval import spread
from ensomi_n.eval.census import Unit
from ensomi_n.eval.strata import set_strata
from ensomi_n.labels import Chart, RedLine, reference_from_charts
from ensomi_n.timing import Segment

BEAT = 500.0          # 120 BPM
START = 1000.0
N_BEATS = 240         # two minutes


def make_set(red, heads, key):
    heads = np.unique(np.asarray(heads, dtype=np.float64))
    chart = Chart(Path(f'{key}.osu'), 'audio.mp3', 3, 'Hard', None, tuple(RedLine(*r) for r in red), heads,
                  float(heads[-1]))
    ref = reference_from_charts([chart], key)
    return ref, set_strata(ref, [chart], tags=(), tags_known=False, key=key)


def beats(start=START, step=BEAT, n=N_BEATS, shift=0.0):
    return start + shift + np.arange(n) * step


def base(shift=0.0, key='a'):
    return make_set([(START + shift, BEAT, 4)], beats(shift=shift), key)


def pair(a, b):
    return spread.compare(a[0], b[0], a[1], b[1])


def test_copy_is_flagged_and_agrees_exactly():
    p = pair(base(key='a'), base(key='b'))
    assert p['flags'] == dict(census=True, c1=True, c2=True, c3=True)
    assert p['offset'] == 0.0
    for d in p['cells']['all']:
        assert np.all(np.abs(d['e']) < 1e-9)
        assert d['bars']['raw'] == d['bars']['n'] > 0
    assert p['level_pair'] == 'correct'


def test_offset_shift_separates_offset_from_shape():
    p = pair(base(key='a'), base(shift=7.0, key='b'))
    assert p['flags'] == dict(census=False, c1=False, c2=True, c3=True)
    assert math.isclose(p['offset'], 7.0, abs_tol=1e-9)
    ab, ba = p['cells']['all']
    assert math.isclose(ab['o'], 7.0, abs_tol=1e-9) and math.isclose(ba['o'], -7.0, abs_tol=1e-9)
    assert np.allclose(np.abs(ab['e']), 7.0)
    assert np.allclose(ab['e'] - ab['o'], 0.0)
    assert ab['bars']['raw'] == ab['bars']['n'] and ab['bars']['o'] == ab['bars']['n']


def test_shift_beyond_bar_tolerance_matches_bars_only_after_removing_the_offset():
    p = pair(base(key='a'), base(shift=30.0, key='b'))
    ab = p['cells']['all'][0]
    assert ab['bars']['raw'] == 0 and ab['bars']['o'] == ab['bars']['n'] > 0
    assert p['flags']['c1'] is False and p['flags']['c2'] is True


def test_octave_folds_out_of_lines_and_shows_in_level():
    b = make_set([(START, BEAT / 2, 4)], beats(step=BEAT / 2, n=2 * N_BEATS), 'b')
    p = pair(base(key='a'), b)
    assert p['level_pair'] == 'octave'
    assert p['flags']['c1'] is False
    ab = p['cells']['all'][0]
    assert np.all(np.abs(ab['e']) < 1e-9)
    assert ab['level']['octave'] > 0 and ab['level']['correct'] == 0
    assert ab['bars']['n'] == 0 and ab['bars']['other_level'] > 0


def test_downbeat_one_beat_late_is_counted_by_beats():
    b = make_set([(START + BEAT, BEAT, 4)], beats(), 'b')
    p = pair(base(key='a'), b)
    ab = p['cells']['all'][0]
    assert np.all(np.abs(ab['e']) < 1e-9)
    assert ab['bars']['raw'] == 0 and ab['bars']['o'] == 0
    assert ab['bars']['disp']['1'] == ab['bars']['n'] > 0
    assert p['flags']['c1'] is False and p['flags']['c2'] is False


def test_meter_three_against_four():
    b = make_set([(START, BEAT, 3)], beats(), 'b')
    ab = pair(base(key='a'), b)['cells']['all'][0]
    bars = ab['bars']
    # 4/4 against 3/4 from one downbeat: A's starts at 0, 12, 24 ... beats coincide with B's.
    assert 0 < bars['o'] < bars['n']
    assert bars['same'] == 0


def test_tempo_change_one_beat_late():
    change = START + 64 * BEAT
    heads_a = np.concatenate([beats(n=64), change + np.arange(150) * 400.0])
    a = make_set([(START, BEAT, 4), (change, 400.0, 4)], heads_a, 'a')
    heads_b = np.concatenate([beats(n=65), change + BEAT + np.arange(149) * 400.0])
    b = make_set([(START, BEAT, 4), (change + BEAT, 400.0, 4)], heads_b, 'b')
    ab = pair(a, b)['cells']['all'][0]
    ch = ab['changes']
    assert ch['n'] == 1
    assert ch['strict_half'] == 0 and ch['loose_half'] == 0
    assert ch['strict_near'] == 1 and ch['loose_near'] == 1


def test_reset_missing_in_the_other_set():
    reset = START + 64 * BEAT + BEAT / 2
    heads_a = np.concatenate([beats(n=64), reset + np.arange(150) * BEAT])
    a = make_set([(START, BEAT, 4), (reset, BEAT, 4)], heads_a, 'a')
    b = make_set([(START, BEAT, 4)], beats(n=220), 'b')
    p = pair(a, b)
    ab = p['cells']['all'][0]
    assert ab['resets']['n'] == 1 and ab['resets']['loose_half'] == 0
    assert ab['reset20'][0] == 1 and ab['reset20'][1] == 0
    assert 'S4' in p['strata'] or 'S9' in p['strata']


def test_shared_range_only():
    a = make_set([(0.0, BEAT, 4)], np.arange(0.0, 60000.0, BEAT), 'a')
    b = make_set([(0.0, BEAT, 4)], np.arange(30000.0, 90000.0, BEAT), 'b')
    p = pair(a, b)
    assert p['lo'] == 30000.0 and p['hi'] == 59500.0
    short = make_set([(0.0, BEAT, 4)], np.arange(50000.0, 90000.0, BEAT), 'c')
    assert pair(a, short)['skip'].startswith('shared range')


def test_shared_exact_non_integer_tempo():
    odd = 60000.0 / 174.97
    assert spread.shared_exact_tempo([Segment(0.0, odd)], [Segment(13.0, odd)])
    assert not spread.shared_exact_tempo([Segment(0.0, 500.0)], [Segment(13.0, 500.0)])
    assert not spread.shared_exact_tempo([Segment(0.0, odd)], [Segment(13.0, odd + 1e-3)])


def test_candidate_pairs_need_one_audio_file_and_two_set_ids():
    units = [Unit('k1', [], set_id='1', audio_sha256='x', recording='r'),
             Unit('k2', [], set_id='2', audio_sha256='x', recording='r'),
             Unit('k2b', [], set_id='2', audio_sha256='x', recording='r'),
             Unit('k3', [], set_id='3', audio_sha256='y', recording='r'),
             Unit('k4', [], set_id='4', audio_sha256='', recording='r')]
    got = sorted((a.key, b.key) for a, b in spread.candidate_pairs(units))
    assert got == [('k1', 'k2'), ('k1', 'k2b')]
    older = spread.candidate_pairs([Unit('new', [], set_id='900', audio_sha256='x'),
                                    Unit('old', [], set_id='12', audio_sha256='x')])
    assert [(a.key, b.key) for a, b in older] == [('old', 'new')]


def test_unsealed_drops_sealed_splits_and_the_hard_holdout():
    units = [Unit('a', [], split='fit_train', set_id='1'), Unit('b', [], split='holdout2', set_id='2'),
             Unit('c', [], split='std-taiko', set_id='3'), Unit('d', [], split='std-taiko', set_id='4'),
             Unit('e', [], split='fit_dev', set_id='5', audio_sha256='zz'),
             Unit('f', [], split='cal', set_id='6', recording='rec:x')]
    excl = dict(keys={'c'}, set_ids=set(), audio_sha256={'zz'}, recordings={'rec:x'})
    kept, drops = spread.unsealed(units, excl, ('holdout1', 'holdout2'))
    assert [u.key for u in kept] == ['a', 'd'] and kept[1].split == 'std-taiko-design'
    assert drops == {'sealed_split': 1, 'hard_holdout': 3}


def test_summary_views_and_known_values():
    copy = pair(base(key='a'), base(key='b'))
    shift = pair(base(key='c'), base(shift=7.0, key='d'))
    late = pair(base(key='e'), base(shift=12.0, key='f'))
    for p, rec in ((copy, 'r1'), (shift, 'r2'), (late, 'r3')):
        p.update(rec=rec, modes='mania4k', split='fit_dev')
    s = spread.summarize([copy, shift, late], draws=200, seed=1, strata_views=('C1',))
    assert s['C0']['all']['n_pairs'] == 3 and s['C1']['all']['n_pairs'] == 2 and s['C2']['all']['n_pairs'] == 0
    c1 = s['C1']['all']
    assert math.isclose(c1['lines']['le5']['v'], 0.0) and math.isclose(c1['lines']['le20']['v'], 1.0)
    assert math.isclose(c1['lines']['le10']['v'], 0.5)
    assert math.isclose(c1['lines_o']['le2']['v'], 1.0)
    assert abs(c1['lines']['q95']['v'] - 12.0) < 0.05 and abs(c1['lines_o']['q50']['v']) < 0.05
    assert c1['offset']['q50']['v'] in (7.0, 12.0)
    assert math.isclose(c1['offset']['le10']['v'], 0.5)
    assert c1['offset']['median_signed']['v'] > 0
    assert c1['lines']['le10']['lo'] <= 0.5 <= c1['lines']['le10']['hi']
    again = spread.summarize([copy, shift, late], draws=200, seed=1, strata_views=('C1',))
    assert again['C1']['all']['lines']['le10'] == c1['lines']['le10']
    assert 'S0' in s['C1'] and 'fast' in s['C1']
    md = spread.markdown(s, dict(pairs=3))
    assert '## Beat lines' in md and 'C1 raw' in md


def _osu(path: Path, creator: str, set_id: int, red, heads) -> None:
    lines = ['osu file format v14', '', '[General]', 'AudioFilename: audio.mp3', 'Mode: 3', '', '[Metadata]',
             'Title:t', f'Creator:{creator}', 'Version:Hard', f'BeatmapSetID:{set_id}', '', '[Difficulty]',
             'CircleSize:4', '', '[TimingPoints]']
    lines += [f'{o!r},{b!r},4,1,0,100,1,0' for o, b in red]
    lines += ['', '[HitObjects]'] + [f'64,192,{t!r},1,0,0:0:0:0:' for t in heads]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_run_reads_files_and_skips_shared_creators():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        heads = list(beats())
        _osu(root / 's1/a.osu', 'Alice', 1, [(START, BEAT)], heads)
        _osu(root / 's2/b.osu', 'Bob', 2, [(START + 5.0, BEAT)], [h + 5.0 for h in heads])
        _osu(root / 's3/c.osu', 'alice', 3, [(START + 9.0, BEAT)], [h + 9.0 for h in heads])
        units = [Unit(f's{i}', [f's{i}/{n}.osu'], source='4k', recording='rec:1', split='fit_dev', set_id=str(i),
                      mode='mania4k', audio_sha256='same') for i, n in ((1, 'a'), (2, 'b'), (3, 'c'))]
        res = spread.run(units, root)
        f = res['funnel']
        assert f['candidate_pairs'] == 3 and f['shared_creator'] == 1 and f['pairs'] == 2
        assert {(p['a'], p['b']) for p in res['pairs']} == {('s1', 's2'), ('s2', 's3')}
        offsets = sorted(round(p['offset'], 6) for p in res['pairs'])
        assert offsets == [4.0, 5.0]
        summary = spread.summarize(res['pairs'], draws=50, seed=0)
        out = root / 'out'
        spread.write(out, res, summary, dict(test=True))
        assert (out / 'summary.md').is_file() and (out / 'pairs.jsonl').read_text().count('\n') == 2


if __name__ == '__main__':
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as exc:  # noqa: BLE001 - report every test
                failed += 1
                print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    sys.exit(1 if failed else 0)
