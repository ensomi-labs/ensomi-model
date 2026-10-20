"""scripts/n1/logic_baseline.py prep, score and report on a three-song fixture with known answers."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('ensomi_n')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'n1'))

import logic_baseline as lb  # noqa: E402
from ensomi_n.mapformat import MusicMap  # noqa: E402
from ensomi_n.mapformat.logic import SCHEMA  # noqa: E402
from ensomi_n.timing import Segment  # noqa: E402

NB = 500.0          # 120 BPM, 4/4, red line at 1000 ms
FAST = 60000.0 / 220.0


def _osu(path: Path, red, heads, audio='audio.mp3') -> None:
    lines = ['osu file format v14', '', '[General]', f'AudioFilename: {audio}', 'Mode: 3', '', '[Metadata]',
             'Title:t', 'Version:v', '', '[Difficulty]', 'CircleSize:4', '', '[TimingPoints]']
    lines += [f'{o!r},{b!r},{m},1,0,100,1,0' for o, b, m in red]
    lines += ['', '[HitObjects]'] + [f'64,192,{t!r},1,0,0:0:0:0:' for t in heads]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _logic(path: Path, times_ms, anchor, signature='4/4', mode='constant'):
    t = [x / 1000.0 for x in times_ms]
    labels = []
    for i in range(len(t)):
        bar = 0 if i < anchor else (i - anchor) // int(signature.split('/')[0]) + 1
        down = i >= anchor and (i - anchor) % int(signature.split('/')[0]) == 0
        labels.append(f'Downbeat, Bar {bar}' if down else f'Bar {bar} Beat {i}')
    data = dict(schema=SCHEMA, song_id=path.name[:3], mode=mode, native_downbeat_time_seconds=t[anchor],
                time_signature_events=[dict(audio_file_seconds=0.0, signature=signature)],
                tempo_events=[dict(audio_file_seconds=t[0], bpm=60.0 / (t[1] - t[0]))],
                beats=[dict(ordinal=i, audio_file_seconds=x, raw_ax_label=labels[i]) for i, x in enumerate(t)],
                validation=dict(status='pass'))
    path.write_text(json.dumps(data))


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    songs = [('001', 'A01', 'A', 'S1', 'dataset/0/1/audio.mp3', (5000.0, 15000.0)),
             ('002', 'B01', 'B', 'S2', 'dataset/0/2/audio.mp3', None),
             ('003', 'C01', 'C', 'S1', 'dataset/0/3/audio.mp3', (4000.0, 12000.0))]
    monkeypatch.setattr(lb, 'SONGS', songs)
    import ensomi_n.v0.evaluate as ev
    monkeypatch.setattr(ev, '_tag_lookup', lambda *a, **k: ({}, False))
    logic = tmp_path / 'logic'
    (logic / 'bulk' / 'maps').mkdir(parents=True)
    bulk = tmp_path / 'bulk'
    shas, split = {}, {'panel': [], 'train': []}
    heads = {'001': 1000.0 + NB * np.arange(60), '002': 1000.0 + FAST * np.arange(150),
             '003': 1000.0 + NB * np.arange(40)}
    red = {'001': [(1000.0, NB, 4)], '002': [(1000.0, FAST, 5)], '003': [(1000.0, NB, 4)]}
    for sid, _, _, _, audio, _ in songs:
        p = root / audio
        p.parent.mkdir(parents=True)
        p.write_bytes(b'not an mp3 ' + sid.encode())
        shas[sid] = hashlib.sha256(p.read_bytes()).hexdigest()
        chart = p.parent / f'chart-{sid}.osu'
        _osu(chart, red[sid], heads[sid])
        it = dict(audio_sha256=shas[sid], audio_rel=audio, rec=shas[sid], item=shas[sid],
                  charts=[str(chart.relative_to(root))])
        split['panel' if sid == '001' else 'train'].append(it)
    r0 = tmp_path / 'r0'
    r0.mkdir()
    (r0 / 'split.json').write_text(json.dumps(split))
    (r0 / 'recordings.json').write_text(json.dumps([]))
    curated = [dict(audio_sha256=shas['002'], rec=shas['002'], role='train', keep=True, source='4k', weight=3.0),
               dict(audio_sha256=shas['003'], rec=shas['003'], role='train', keep=False, reasons=['grid'])]
    (tmp_path / 'curated.json').write_text(json.dumps(curated))
    (logic / 'source-integrity.json').write_text(json.dumps(dict(sources=[
        dict(id=s[0], source_filename=f'{s[1]}.mp3', expected_sha256=shas[s[0]]) for s in songs])))
    # Smart Tempo: 001 constant on the chart (+20 ms on the decode, so c20 is exact), variable at double time with
    # the fold on the off-beats; 002 a continuous pulse at the canonical level; 003 both modes 30 ms late after c20.
    k = np.arange(-4, 70)
    _logic(logic / 'bulk' / 'maps' / '001__constant.json', 1000.0 + NB * k + 20.0, anchor=4)
    _logic(logic / 'bulk' / 'maps' / '001__variable.json', 1250.0 + 250.0 * np.arange(-8, 140) + 20.0, anchor=8,
           mode='variable')
    for mode in lb.MODES:
        _logic(logic / 'bulk' / 'maps' / f'002__{mode}.json', 1000.0 + 2 * FAST * np.arange(-2, 80) + 20.0, anchor=2,
               mode=mode)
        _logic(logic / 'bulk' / 'maps' / f'003__{mode}.json', 1000.0 + NB * np.arange(-2, 45) + 50.0, anchor=2,
               mode=mode)
    # v0: on the chart; R2: on the chart's canonical lines for 001 and 003, half a beat off for 002.
    for seed, run in lb.SEEDS.items():
        d = bulk / 'v0-maps' / run
        d.mkdir(parents=True)
        with gzip.open(d / 'maps-000100.jsonl.gz', 'wt') as f:
            for sid in shas:
                m = MusicMap(segments=(Segment(*red[sid][0]),), model={'name': 'v0'})
                f.write(json.dumps(dict(id=sid, item=shas[sid], map=m.to_json())) + '\n')
    (bulk / 'r2-export').mkdir(parents=True)
    for sid in shas:
        lines = [[1000.0, NB, 4]] if sid != '002' else [[1000.0 + FAST, 2 * FAST, 4]]
        (bulk / 'r2-export' / f'{shas[sid]}.json').write_text(json.dumps(dict(red_lines=lines)))
    out = tmp_path / 'out'
    args = ['--root', str(root), '--logic', str(logic), '--out', str(out), '--bulk', str(bulk), '--r0', str(r0),
            '--curated', str(tmp_path / 'curated.json'), '--r2-stored', str(tmp_path / 'none')]
    monkeypatch.chdir(root)
    return dict(args=args, out=out, shas=shas)


def test_prep_score_report(project):
    lb.main(['prep'] + project['args'])
    meta = json.loads((project['out'] / 'songs.json').read_text())
    assert not meta['problems'] and meta['maps_with_labels'] == 6
    exp = {s['id']: s['v0_exposure'] for s in meta['songs']}
    assert exp == {'001': 'held-out', '002': 'trained', '003': 'held-out'}
    assert all(s['lead_ms'] == 0.0 for s in meta['songs'])
    lb.main(['score'] + project['args'])
    rows = {r['id']: r for r in json.loads((project['out'] / 'items.json').read_text())}
    assert sorted(p.suffix for p in project['out'].iterdir()) == ['.json'] * len(list(project['out'].iterdir()))
    s1 = rows['001']['scores']
    assert s1['st-constant@c20']['hard']['bl20'] == pytest.approx(1.0)
    assert s1['st-constant@c287']['hard']['offset_ms'] == pytest.approx(-8.7)
    assert s1['st-variable@c20']['whole']['attribution'] == 'level: octave with the fold on the other phase'
    assert s1['r2']['whole']['bl20'] == pytest.approx(1.0) and s1['v0s0@000100']['whole']['bl20'] == pytest.approx(1.0)
    ag = rows['001']['agreement']['c20']
    assert ag['agree']['st-constant|mapper'] and not ag['agree']['st-variable|mapper']
    assert rows['003']['scores']['st-constant@c20']['hard']['attribution'] == 'whole-grid offset of clock size'
    pulse = rows['002']['pulse']['c20']
    assert pulse['arms']['st-constant']['native']['verdict'] == 'continuous pulse (R reading)'
    assert pulse['arms']['st-constant']['canonical']['verdict'] == 'continuous pulse (R reading)'
    # v0's map states the chart's 220 BPM 5/4 line: its beats hit both readings, its canonical lines restart.
    assert pulse['arms']['v0s0']['native']['verdict'].startswith('too fine')
    assert pulse['arms']['v0s0']['canonical']['verdict'] == 'restarts each bar (N reading)'
    # R2 here runs a continuous pulse one notated beat late: it hits N's own lines and misses the shared ones.
    assert pulse['arms']['r2']['canonical']['verdict'] == 'continuous pulse a notated beat off R'
    lb.main(['report'] + project['args'])
    res = json.loads((project['out'] / 'results.json').read_text())
    a = res['tiers']['A']['hard']['c20']['bl20']
    assert a['arms']['st-constant']['value'] == pytest.approx(1.0) and a['arms']['st-oracle']['value'] == pytest.approx(1.0)
    c = res['tiers']['C']['hard']['c20']['bl20']
    assert c['paired']['st-constant - v0mean (held-out)']['value'] == pytest.approx(-1.0)
    b = res['tiers']['B']['whole']['c20']['bl20']
    assert b['paired']['st-constant - v0mean (held-out)']['n'] == 0
    assert b['paired']['st-constant - v0mean (trained-on, apart)']['n'] == 1
    md = (project['out'] / 'RESULTS.md').read_text()
    assert '## Clock rule' in md and 'B01' in md and 'continuous pulse' in md
