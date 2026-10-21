"""scripts/n1/open_baselines on the three-song fixture of the Smart Tempo end-to-end test: the Smart Tempo run is
scored first, then prep, preflight, synthetic tool outputs through the converters, status, score and report.

The old arms must come out unchanged: the preflight reproduces the Smart Tempo results, and the new items keep
every old arm's numbers exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('ensomi_n')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'n1'))
sys.path.insert(0, str(ROOT / 'scripts' / 'n1' / 'open_baselines'))
sys.path.insert(0, str(ROOT / 'tests' / 'n1'))

import logic_baseline as lb  # noqa: E402
import obtools as tools  # noqa: E402
import open_baselines as ob  # noqa: E402
import test_logic_baseline_e2e as st  # noqa: E402
from test_logic_baseline_e2e import project  # noqa: E402,F401  (the Smart Tempo fixture, used by open_project)

RATE = 44100
FRAMES = RATE * 2


@pytest.fixture
def open_project(project, tmp_path):
    base = project
    args = base['args']
    get = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
    logic, out = Path(get['--logic']), Path(get['--out'])
    # The capture's records of the canonical decodes, and the decodes themselves (two seconds, distinct per song).
    wav_dir = tmp_path / 'inputs-wav'
    wav_dir.mkdir()
    inputs = []
    (logic / 'bulk' / 'raw-resu').mkdir(parents=True)
    for k, (sid, *_rest) in enumerate(lb.SONGS):
        x = (np.random.default_rng(k).standard_normal((FRAMES, 2)) * 0.05).astype(np.float32)
        tools.write_wav_float(wav_dir / f'{sid}.wav', x, RATE, extensible=True)
        sha = hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()
        inputs.append(dict(id=sid, prepared_input_path=str(wav_dir / f'{sid}.wav'),
                           conversion=dict(decoded_pcm_sha256=f'SHA256={sha}', sample_count=FRAMES,
                                           audio=dict(stream=dict(sample_rate=str(RATE), channels=2)))))
        for mode in lb.MODES:
            cap = dict(song_id=sid, mode=mode, canonical_pcm_matches=True,
                       audio=dict(pcm_sha256=sha, sample_count=FRAMES, sample_rate_hz=RATE, channels=2))
            (logic / 'bulk' / 'raw-resu' / f'{sid}__{mode}.capture.json').write_text(json.dumps(cap))
    (logic / 'bulk' / 'manifest.json').write_text(json.dumps(dict(schema='x', inputs=inputs)))
    lb.main(['prep'] + args)
    lb.main(['score'] + args)
    lb.main(['report'] + args)
    ob_out, ob_bulk = tmp_path / 'open-out', tmp_path / 'open-bulk'
    ob_args = ['--root', get['--root'], '--logic', str(logic), '--logic-out', str(out), '--logic-bulk', get['--bulk'],
               '--out', str(ob_out), '--bulk', str(ob_bulk), '--r0', get['--r0'], '--curated', get['--curated'],
               '--r2-stored', get['--r2-stored']]
    return dict(base, ob_args=ob_args, ob_out=ob_out, ob_bulk=ob_bulk, logic_out=out)


def _vamp_raw(run, times_s, labels, plugin):
    outs = {'0': [[int(t), int(round((t - int(t)) * 1e9)), lab, None] for t, lab in zip(times_s, labels)]}
    outs['1'] = [b for b in outs['0'] if b[2] == '1']
    params = {'bpb': float(run[10:])} if run.startswith('qm-bbt-bpb') else {}
    return dict(runner='ardour_vamp.lua', run=run, plugin=plugin, parameters=params, session_rate=RATE,
                readable_length=FRAMES, channels=2, channel=0, analyze_rv=0, outputs=outs)


def _write_tool_outputs(bulk: Path, jobs: dict) -> None:
    """What the Mac runs would leave: tracker beats 20 ms after the charts' lines on the decode; for the 5/4 song,
    bar starts every 4 beats at the default and every 5 beats when asked for 5; MiniBPM at 120 (the charts of 001
    and 003); TempoCNN at 120, and a failed TempoCNN run on 003."""
    nb, fast = st.NB, st.FAST

    def put(job, outs):
        for arm, d in outs.items():
            tools.dump(bulk / 'maps' / 'main' / arm / f"{job['id']}.json", d)
    for sid, job in jobs.items():
        audio = ob._audio(job)
        step = fast if sid == '002' else nb
        t = (1000.0 + step * np.arange(-2, 160 if sid == '002' else 60) + 20.0) / 1000.0
        for k in job['bpbs']:
            labels = [str((i - 2) % k + 1) for i in range(len(t))]
            raw = _vamp_raw(f'qm-bbt-bpb{k}', t, labels, 'libardourvampplugins:qm-barbeattracker')
            put(job, tools.vamp_outputs(raw, sid, audio))
        raw = _vamp_raw('qm-tempo', t, [''] * len(t), 'libardourvampplugins:qm-tempotracker')
        put(job, tools.vamp_outputs(raw, sid, audio))
        mb = dict(sample_rate=RATE, frames=FRAMES, whole=dict(bpm=120.0, candidates=[120.0]), middle10=None)
        put(job, tools.minibpm_outputs(mb, sid, audio))
        tc = dict(bpm=120, confidence=0.9, topCandidates=[], sampleRate=RATE, frames=FRAMES)
        if sid == '003':
            tc = dict(error='timed out after 900 s')
        put(job, tools.tempocnn_outputs(tc, sid, audio))


def test_open_baselines_end_to_end(open_project):
    project = open_project
    a = project['ob_args']
    ob.main(['prep'] + a)
    jobs = {j['id']: j for j in json.loads((project['ob_bulk'] / 'jobs.json').read_text())}
    assert sorted(jobs) == ['001', '002', '003', 'P44', 'P48']
    assert jobs['002']['oracle_bpbs'] == [5] and jobs['002']['bpbs'] == [4, 5] and jobs['001']['bpbs'] == [4]
    songs = json.loads((project['ob_out'] / 'open-songs.json').read_text())
    assert not songs['problems'] and all(s['ok'] for s in songs['songs'])

    ob.main(['preflight'] + a)
    cmp = json.loads((project['ob_out'] / 'preflight' / 'compare.json').read_text())
    assert cmp['results.json']['equal'] and cmp['items.json']['equal']

    _write_tool_outputs(project['ob_bulk'], {k: v for k, v in jobs.items() if v['kind'] == 'song'})
    ob.main(['status'] + a)
    status = json.loads((project['ob_out'] / 'run-status.json').read_text())
    assert status['counts']['tcnn'] == dict(ok=2, empty=0, failed=1, **{'not run': 0})
    assert status['failures'] == [dict(arm='tcnn', id='003', error='timed out after 900 s')]
    assert status['counts']['qm-bbt-m5'] == dict(ok=1, empty=0, failed=0, **{'not run': 2})

    ob.main(['score'] + a)
    old = {r['id']: r for r in json.loads((project['logic_out'] / 'items.json').read_text())}
    new = {r['id']: r for r in json.loads((project['ob_out'] / 'items.json').read_text())}
    for sid, r in old.items():
        for arm, v in r['scores'].items():
            assert new[sid]['scores'][arm] == v, (sid, arm)
        assert new[sid]['agreement'] == r['agreement']
    s1 = new['001']['scores']
    assert s1['qm-bbt@c20']['hard']['bl20'] == pytest.approx(1.0)
    assert s1['qm-bbt@c287']['hard']['offset_ms'] == pytest.approx(-8.7)
    assert s1['qm-tempo@c20']['hard']['pred_has_bars'] is False
    assert new['001']['open']['tempo']['mbpm-raw']['whole']['acc1'] == pytest.approx(1.0)
    assert new['001']['open']['stated_bpm']['tcnn'] == 120.0
    assert 'tcnn@c20' not in new['003']['scores'] and new['003']['open']['status']['tcnn'].startswith('failed')
    s2 = new['002']['scores']
    assert s2['qm-bbt-m5@c20']['whole']['bars_start'] > s2['qm-bbt@c20']['whole']['bars_start']
    assert new['002']['pulse_open']['c20']['arms']['qm-bbt-m5']['canonical']['verdict'] == 'restarts each bar (N reading)'
    ag = new['001']['agreement_open']['c20']
    assert ag['agree']['qm-bbt|mapper'] and ag['systems']['pairs']['mapper|qm-bbt']

    ob.main(['report'] + a)
    res = json.loads((project['ob_out'] / 'results.json').read_text())
    st_res = json.loads((project['logic_out'] / 'results.json').read_text())
    for tier, span in (('A', 'hard'), ('B', 'whole'), ('C', 'hard')):
        for cname in lb.CLOCK_CONSTANTS:
            for m in lb.MEASURES:
                mine = res['tiers'][tier][span][cname][m]['arms']
                theirs = st_res['tiers'][tier][span][cname][m]['arms']
                for arm in ('st-constant', 'st-variable', 'v0s0', 'v0s1', 'v0mean', 'r2'):
                    assert mine[arm] == theirs[arm], (tier, span, cname, m, arm)
    b = res['tiers']['B']['whole']['c20']['bars_start']['arms']
    assert b['qm-bbt-oracle']['value'] > b['qm-bbt']['value']
    c = res['tiers']['C']['hard']['c20']['bl20']
    assert c['arms']['tcnn']['n'] == 0 and c['arms']['qm-bbt']['value'] == pytest.approx(1.0)
    assert c['paired']['qm-bbt - st-constant']['value'] == pytest.approx(1.0)
    assert res['tempo']['A']['whole']['mbpm-raw']['acc1']['value'] == pytest.approx(1.0)
    md = (project['ob_out'] / 'RESULTS.md').read_text()
    assert 'Preflight: Smart Tempo `results.json` reproduced' in md and '`tcnn` 003: timed out' in md
    # The synced results directory holds Markdown and JSON only; logs and tool outputs stay in bulk.
    files = [p for p in project['ob_out'].rglob('*') if p.is_file()]
    assert {p.suffix for p in files} == {'.json', '.md'}
    assert sorted(p.name for p in files if p.suffix == '.md') == ['RESULTS.md', 'RESULTS.md']   # and the preflight's
