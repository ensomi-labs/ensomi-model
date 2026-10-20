"""scripts/n1/logic_baseline.py on charts and maps with known answers: clock rule, attribution, agreement, tier B
pulse, bootstrap tables, v0 exposure."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('ensomi_n')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'n1'))

import logic_baseline as lb  # noqa: E402
from ensomi_n.labels import read_osu, reference_from_charts  # noqa: E402
from ensomi_n.mapformat import MusicMap  # noqa: E402
from ensomi_n.mapformat.logic import SCHEMA, read_logic_map  # noqa: E402
from ensomi_n.timing import Segment  # noqa: E402


def _write(path: Path, red, heads) -> Path:
    lines = ['osu file format v14', '', '[General]', 'AudioFilename: a.mp3', 'Mode: 3', '', '[Metadata]',
             'Title:t', 'Version:v', '', '[Difficulty]', 'CircleSize:4', '', '[TimingPoints]']
    lines += [f'{o!r},{b!r},{m},1,0,100,1,0' for o, b, m in red]
    lines += ['', '[HitObjects]'] + [f'64,192,{t!r},1,0,0:0:0:0:' for t in heads]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _ref(tmp_path, red, heads, name='c.osu'):
    path = _write(tmp_path / name, red, heads)
    return reference_from_charts([read_osu(path)], 'c'), path


def _logic(tmp_path, times_ms, anchor_index=0, signature='4/4'):
    t = [x / 1000.0 for x in times_ms]
    data = dict(schema=SCHEMA, song_id='901', mode='constant', native_downbeat_time_seconds=t[anchor_index],
                time_signature_events=[dict(audio_file_seconds=0.0, signature=signature)],
                tempo_events=[dict(audio_file_seconds=t[0], bpm=60.0 / (t[1] - t[0]))],
                beats=[dict(ordinal=i, audio_file_seconds=x) for i, x in enumerate(t)])
    p = tmp_path / 'logic.json'
    p.write_text(json.dumps(data))
    return p


def _beats(times):
    return MusicMap(beats=np.asarray(times, dtype=np.float64), model={'name': 't'})


CHART = [(1000.0, 500.0, 4)]
HEADS = 1000.0 + 500.0 * np.arange(80)


def test_clock_rule_end_to_end(tmp_path):
    """Markers 20 ms after the chart's lines on the decode: c20 passes with offset 0; c287 passes 8.7 ms early."""
    ref, _ = _ref(tmp_path, CHART, HEADS)
    m = read_logic_map(_logic(tmp_path, 1000.0 + 500.0 * np.arange(-2, 84) + 20.0, anchor_index=2))
    r20 = lb.score_span(ref, m.on_osu_clock(lb.CLOCK_CONSTANTS['c20']), None)
    assert r20['bl20'] == pytest.approx(1.0) and r20['offset_ms'] == pytest.approx(0.0, abs=1e-6)
    assert r20['attribution'] == 'pass' and r20['bars_start'] == pytest.approx(1.0)
    r28 = lb.score_span(ref, m.on_osu_clock(lb.CLOCK_CONSTANTS['c287']), None)
    assert r28['bl20'] == pytest.approx(1.0) and r28['offset_ms'] == pytest.approx(-8.7)
    with pytest.raises(ValueError, match='decoded clock'):
        lb.score_span(ref, m, None)


def test_attribution_classes(tmp_path):
    ref, _ = _ref(tmp_path, CHART, HEADS)
    # Double time with the fold on the off-beats: every chart line is among the arm's beats, none among its lines.
    fold = lb.score_span(ref, _beats(1250.0 + 250.0 * np.arange(-8, 170)), None)
    assert fold['bl20'] < 0.1 and fold['union20'] > 0.95
    assert fold['attribution'] == 'level: octave with the fold on the other phase'
    # 160 BPM against 120: a 4:3 pulse.
    other = lb.score_span(ref, _beats(1000.0 + 375.0 * np.arange(-4, 110)), None)
    assert other['level_classes']['family'] > 0.9 and other['bl20'] < 0.5
    assert other['attribution'] == 'level: another pulse (a 3:2, 4:3 or 5:4 family ratio)'
    shifted = lb.score_span(ref, _beats(1000.0 + 500.0 * np.arange(-2, 84) + 30.0), None)
    assert shifted['bl20'] == pytest.approx(0.0) and shifted['bl20_o'] == pytest.approx(1.0)
    assert shifted['attribution'] == 'whole-grid offset of clock size'
    half = lb.score_span(ref, _beats(1000.0 + 500.0 * np.arange(-2, 84) + 250.0), None)
    assert half['attribution'] == 'whole-grid phase shift'
    drift = lb.score_span(ref, _beats(1000.0 + 503.0 * np.arange(-2, 84)), None)
    assert drift['level_classes']['other'] > 0.9
    assert drift['attribution'] == 'tempo off by more than 0.5 % (no level ratio)'
    # Correct tempo, lines scattered by 30 ms around the chart's on alternate beats: no constant shift explains it.
    jitter = 1000.0 + 500.0 * np.arange(-2, 84) + np.where(np.arange(86) % 2, 30.0, -30.0)
    scattered = lb.score_span(ref, _beats(jitter), None)
    assert scattered['bl20'] < 0.1 and scattered['attribution'] == 'phase error'
    assert lb.attribution({'bl20': math.nan}) == 'unscored'


def test_agreement_groups_and_the_four_readings():
    a = 1000.0 + 500.0 * np.arange(20)
    lines = {'st-constant': a, 'st-variable': a + 5.0, 'v0s0': a + 50.0, 'v0s1': a + 51.0, 'r2': a[::2],
             'mapper': a}
    ag = lb.agreement(lines)
    assert ag['agree']['st-constant|st-variable'] and ag['agree']['v0s0|v0s1']
    assert not ag['agree']['st-constant|r2']           # R2 holds half the lines: one direction is 1, the other 0.5
    assert sorted(map(tuple, ag['groups'])) == [('mapper', 'st-constant', 'st-variable'), ('r2',), ('v0s0', 'v0s1')]
    sysv = lb.system_agreement(ag)
    assert sysv['pairs']['mapper|st'] and not sysv['pairs']['mapper|v0'] and not sysv['all_disagree']
    lines2 = {'st-constant': a + 100.0, 'st-variable': a + 100.0, 'v0s0': a + 50.0, 'v0s1': a + 50.0,
              'r2': a + 200.0, 'mapper': a}
    s2 = lb.system_agreement(lb.agreement(lines2))
    assert s2['all_disagree'] and s2['chart_alone']


def test_tier_b_pulse_reads_continuous_and_restarting_arms(tmp_path):
    """5/4 notated at 220 BPM: N restarts canonical lines each bar (notated beats 0, 2, 4), R runs on every 2."""
    nb = 60000.0 / 220.0
    heads = 1000.0 + nb * np.arange(200)
    ref, path = _ref(tmp_path, [(1000.0, nb, 5)], heads)
    import reader_census as rc
    r_t = rc.r_lines(str(path), ref.lo_ms, ref.hi_ms)[0]
    k = np.arange(-4, 210)
    continuous = 1000.0 + 2 * nb * k
    restart = np.sort(np.concatenate([1000.0 + 5 * nb * b + nb * np.array([0, 2, 4]) for b in range(-2, 45)]))
    res = lb.pulse(ref, r_t, {'cont': _beats(continuous), 'restart': _beats(restart),
                              'fine': _beats(1000.0 + nb * k)})
    assert res['n_only'] > 10 and res['r_only'] > 10
    assert res['arms']['cont']['native']['verdict'] == 'continuous pulse (R reading)'
    assert res['arms']['restart']['native']['verdict'] == 'restarts each bar (N reading)'
    assert res['arms']['fine']['native']['verdict'].startswith('too fine')
    assert res['arms']['cont']['native']['median_line_notated'] == pytest.approx(2.0)
    # Without bars of its own, a beat list at the notated level folds onto one continuous phase.
    assert res['arms']['fine']['canonical']['verdict'] == 'continuous pulse (R reading)'


def test_tables_oracle_and_held_out_pairing():
    def row(i, exposure, st_c, st_v, v0, r2):
        sc = {'st-constant@c20': {'whole': {'bl20': st_c}}, 'st-variable@c20': {'whole': {'bl20': st_v}},
              'v0s0@9': {'whole': {'bl20': v0}}, 'v0s1@9': {'whole': {'bl20': v0}}, 'r2': {'whole': {'bl20': r2}}}
        return dict(id=f'{i:03d}', v0_exposure=exposure, scores=sc)
    rows = [row(1, 'held-out', 0.9, 0.5, 0.8, 0.7), row(2, 'held-out', 0.3, 0.7, 0.8, 0.7),
            row(3, 'trained', 1.0, 1.0, 0.0, 0.7)]
    last = {'s0': 'v0s0@9', 's1': 'v0s1@9'}
    blk = lb.table_block(rows, last, 'whole', 'bl20', 'c20')
    assert blk['arms']['st-oracle']['value'] == pytest.approx((0.9 + 0.7 + 1.0) / 3)
    assert blk['paired']['st-constant - r2']['value'] == pytest.approx((0.2 - 0.4 + 0.3) / 3)
    held = blk['paired']['st-constant - v0mean (held-out)']
    assert held['n'] == 2 and held['value'] == pytest.approx((0.1 - 0.5) / 2)
    assert blk['paired']['st-constant - v0mean (trained-on, apart)']['value'] == pytest.approx(1.0)
    assert blk['v0_seed_spread'] == pytest.approx(0.0)
    b = lb.boot_mean(np.array([1.0, 1.0, math.nan]))
    assert b['n'] == 2 and b['lo'] == b['hi'] == 1.0


def test_v0_exposure_by_audio_and_by_recording():
    cur = [dict(audio_sha256='a', rec='ra', role='train', keep=True, source='4k', weight=3.0, audio_rel='x/a'),
           dict(audio_sha256='b', rec='rb', role='train', keep=False, reasons=['grid'], source='4k'),
           dict(audio_sha256='c', rec='rc', role='panel'),
           dict(audio_sha256='other', rec='rd', role='train', keep=True, source='4k', weight=1.5, audio_rel='y/o')]
    assert lb.v0_exposure('a', {'ra'}, cur)[0] == 'trained'
    status, why = lb.v0_exposure('b', {'rb'}, cur)
    assert status == 'held-out' and 'grid' in why
    assert lb.v0_exposure('c', {'rc'}, cur)[0] == 'held-out'
    status, why = lb.v0_exposure('d', {'rd'}, cur)
    assert status == 'trained' and 'another audio copy' in why
    assert lb.v0_exposure('e', {'re'}, cur) == ('held-out', 'no curated record')


def test_map_difference():
    a = dict(beats=[1.0, 2.0, 3.0], downbeats=[0], segments=[dict(start_ms=1.0, beat_ms=1.0)])
    d = lb.map_difference(a, dict(beats=[1.0, 2.0, 3.005], downbeats=[0], segments=[dict(start_ms=1.0, beat_ms=1.0)]))
    assert d['downbeats_equal'] and d['max_abs_ms'] == pytest.approx(0.005)
    d = lb.map_difference(a, dict(beats=[1.0, 2.0, 3.0], downbeats=[0], segments=[dict(start_ms=1.0, beat_ms=1.02)]))
    assert d['max_abs_ms'] == pytest.approx(0.02)
    assert lb.map_difference(a, dict(beats=[1.0, 2.0], downbeats=[1]))['max_abs_ms'] is None


def test_song_table_is_the_list():
    assert [s[1] for s in lb.SONGS] == [f'A{i:02d}' for i in range(1, 13)] + [f'B{i:02d}' for i in range(1, 10)] + \
        [f'C{i:02d}' for i in range(1, 28)]
    assert [s[0] for s in lb.SONGS] == [f'{i:03d}' for i in range(1, 49)]
    for sid, lid, tier, group, audio, span in lb.SONGS:
        assert (span is None) == (tier == 'B')
        assert span is None or span[1] > span[0]
    assert [s[3] for s in lb.SONGS if s[2] == 'C'] == [f'S{k}' for k in range(1, 10) for _ in range(3)]
