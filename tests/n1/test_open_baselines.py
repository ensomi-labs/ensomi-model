"""scripts/n1/open_baselines on inputs with known answers: Ardour's tempo rounding, canonical WAVs, the raw-to-
normalized converters, the tempo terms, probes, the preflight comparison, and the TempoCNN runner's WAV reader."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('ensomi_n')
OB = Path(__file__).resolve().parents[2] / 'scripts' / 'n1' / 'open_baselines'
sys.path.insert(0, str(OB))

import obtools as tools  # noqa: E402
import open_baselines as ob  # noqa: E402
from ensomi_n.mapformat.open_tools import read_open_map  # noqa: E402
from ensomi_n.timing import Reference, Segment  # noqa: E402

AUDIO = dict(pcm_sha256='ab' * 32, frames=441000, sample_rate_hz=44100, channels=2, channel_used=0)


def test_c_round_takes_halves_away_from_zero():
    assert [tools.c_round(x) for x in (2.5, 3.5, -2.5, 2.4999)] == [3.0, 4.0, -3.0, 2.0]


def test_ardour_rounding_over_a_whole_song_keeps_whole_bars():
    """A 138.97 s song at 113.5 BPM: 262.9 beats -> 263 -> 65.75 bars -> 66 bars = 264 beats. Past 18 bars no
    power of two applies; the two roundings move the beat count over the song by at most 2.5 beats."""
    qpm, beats = tools.ardour_rounded_tempo(113.5, 138.9735625)
    assert beats == 264 and qpm == pytest.approx(264 / (138.9735625 / 60))
    moved = []
    for est in np.linspace(55, 190, 271):
        q, b = tools.ardour_rounded_tempo(float(est), 200.0)
        moved.append(abs(q - est) * 200.0 / 60.0)
        assert b % 4 == 0 and moved[-1] <= 2.5 + 1e-9
    assert max(moved) > 2.0


def test_ardour_rounding_over_ten_seconds_lands_on_48_96_or_192():
    """The source estimate: MiniBPM's 55-190 BPM over ten seconds is 2 to 8 bars, which the power-of-two rule takes
    to 2, 4 or 8 bars. C's rounding matters at the halves: 60 BPM is 10 beats, 2.5 bars -> 3 -> 4 bars."""
    assert tools.ardour_rounded_tempo(113.5, 10.0) == (192.0, 32.0)
    assert tools.ardour_rounded_tempo(60.0, 10.0) == (96.0, 16.0)
    assert tools.ardour_rounded_tempo(55.0, 10.0) == (48.0, 8.0)
    seen = {tools.ardour_rounded_tempo(float(e), 10.0)[0] for e in np.linspace(55, 190, 1351)}
    assert seen == {48.0, 96.0, 192.0}
    # 17 and 18 bars stay as they are; a 0 ("too short") stays 0.
    assert tools.ardour_rounded_tempo(414.0, 10.0) == (408.0, 68.0)
    assert tools.ardour_rounded_tempo(0.0, 10.0) == (0.0, 0.0)
    # A 3-bar loop goes to 4 bars.
    assert tools.ardour_rounded_tempo(120.0, 6.0) == (160.0, 16.0)


def test_source_window_is_ten_seconds_from_the_middle():
    assert tools.ardour_source_window(441001 * 2, 44100) == ((882002 - 441000) // 2, 441000)
    assert tools.ardour_source_window(441000, 44100) == (0, 441000)
    assert tools.ardour_source_window(440999, 44100) is None


@pytest.mark.parametrize('extensible', [False, True])
def test_wav_info_hashes_the_payload_and_reads_both_float_headers(tmp_path, extensible):
    x = np.random.default_rng(0).standard_normal((1000, 2)).astype(np.float32) * 0.1
    p = tmp_path / 'a.wav'
    tools.write_wav_float(p, x, 48000, extensible=extensible)
    info = tools.wav_info(p)
    assert info['tag'] == tools.FLOAT and info['format_tag'] == (tools.EXTENSIBLE if extensible else tools.FLOAT)
    assert (info['frames'], info['channels'], info['sample_rate_hz']) == (1000, 2, 48000)
    assert info['pcm_sha256'] == hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()
    y, rate = tools.read_wav(p)
    assert rate == 48000 and np.array_equal(y, x)
    want = dict(pcm_sha256=info['pcm_sha256'], frames=1000, sample_rate_hz=48000, channels=2)
    assert ob.check_wav(p, want)[1] == []
    bad = ob.check_wav(p, dict(want, frames=999, pcm_sha256='0' * 64))[1]
    assert len(bad) == 2 and any('frames' in b for b in bad) and any('pcm_sha256' in b for b in bad)
    with pytest.raises(ValueError, match='RIFF'):
        (tmp_path / 'b.wav').write_bytes(b'nope' * 4)
        tools.wav_info(tmp_path / 'b.wav')


def test_canonical_record_needs_the_manifest_and_captures_to_agree():
    m = dict(conversion=dict(decoded_pcm_sha256='SHA256=' + 'AB' * 32, sample_count=10,
                             audio=dict(stream=dict(sample_rate='44100', channels=2))))
    cap = dict(audio=dict(pcm_sha256='ab' * 32, sample_count=10, sample_rate_hz=44100, channels=2),
               canonical_pcm_matches=True)
    rec = ob.canonical_record('001', m, {'constant': cap, 'variable': cap})
    assert rec['problems'] == [] and rec['expected']['pcm_sha256'] == 'ab' * 32
    other = dict(cap, audio=dict(cap['audio'], sample_count=11))
    assert len(ob.canonical_record('001', m, {'constant': cap, 'variable': other})['problems']) == 1


def _vamp_raw(run, beats, labels=None, bars=None, **kw):
    params = {'bpb': float(run[10:])} if run.startswith('qm-bbt-bpb') else {'method': 1.0}
    outs = {'0': [[int(t), int(round((t - int(t)) * 1e9)), (labels[i] if labels else f'{t:.2f} bpm'), None]
                  for i, t in enumerate(beats)]}
    if bars is not None:
        outs['1'] = [[int(t), int(round((t - int(t)) * 1e9)), str(i + 1), None] for i, t in enumerate(bars)]
    raw = dict(runner='ardour_vamp.lua', run=run, plugin='libardourvampplugins:qm-barbeattracker', parameters=params,
               session_rate=44100, readable_length=441000, channels=2, channel=0, analyze_rv=0, outputs=outs)
    raw.update(kw)
    return raw


def test_vamp_bar_beat_output_reads_bars_from_the_labels(tmp_path):
    t = 0.5 + 0.5 * np.arange(12)
    labels = [str((i + 3) % 4 + 1) for i in range(12)]          # a pickup: 4, then bars from beat 1
    raw = _vamp_raw('qm-bbt-bpb4', t, labels, bars=t[1::4])
    d = tools.vamp_outputs(raw, '001', AUDIO)['qm-bbt']
    assert d['status'] == 'ok' and d['downbeats'] == [1, 5, 9] and d['provenance']['bar_check']['equal']
    p = tmp_path / 'o.json'
    p.write_text(json.dumps(d))
    m = read_open_map(p)
    np.testing.assert_allclose(m.beats, 1000.0 * t, atol=1e-6)
    five = tools.vamp_outputs(_vamp_raw('qm-bbt-bpb5', t, ['1'] * 12), '001', AUDIO)
    assert list(five) == ['qm-bbt-m5'] and five['qm-bbt-m5']['status'] == 'ok'


def test_vamp_tempo_output_states_no_bars_and_failures_are_kept():
    t = 0.5 + 0.5 * np.arange(8)
    d = tools.vamp_outputs(_vamp_raw('qm-tempo', t, plugin='libardourvampplugins:qm-tempotracker'), '001', AUDIO)
    assert d['qm-tempo']['status'] == 'ok' and d['qm-tempo']['downbeats'] is None
    assert tools.vamp_outputs(_vamp_raw('qm-tempo', []), '001', AUDIO)['qm-tempo']['status'] == 'empty'
    cases = [dict(error='Readable.load failed'), dict(session_rate=48000), dict(readable_length=440000),
             dict(analyze_rv=-1), dict(parameters={'bpb': 3.0})]
    for kw in cases:
        out = tools.vamp_outputs(_vamp_raw('qm-bbt-bpb4', t, ['1'] * 8, **kw), '001', AUDIO)['qm-bbt']
        assert out['status'] == 'failed' and out['error'], kw
    shuffled = tools.vamp_outputs(_vamp_raw('qm-bbt-bpb4', [1.0, 0.5], ['1', '2']), '001', AUDIO)['qm-bbt']
    assert shuffled['status'] == 'failed' and 'increasing' in shuffled['error']


def test_minibpm_outputs_three_arms_on_the_file_start_grid():
    raw = dict(runner='minibpm_cli', sample_rate=44100, frames=441000 * 3, channels=2, format_tag=3,
               whole=dict(bpm=113.5, candidates=[113.5, 56.75]), middle10=dict(bpm=113.5, candidates=[113.5]))
    audio = dict(AUDIO, frames=441000 * 3)
    out = tools.minibpm_outputs(raw, '001', audio)
    assert sorted(out) == ['mbpm-clip', 'mbpm-raw', 'mbpm-source']
    assert out['mbpm-raw']['bpm'] == 113.5 and out['mbpm-source']['bpm'] == 192.0
    assert out['mbpm-clip']['bpm'] == pytest.approx(tools.ardour_rounded_tempo(113.5, 30.0)[0])
    assert all(d['grid'] == dict(anchor_s=0.0, meter=4) and d['status'] == 'ok' for d in out.values())
    zero = tools.minibpm_outputs(dict(raw, whole=dict(bpm=0.0, candidates=[]), middle10=None), '001', audio)
    assert zero['mbpm-raw']['status'] == 'empty' and zero['mbpm-source']['status'] == 'empty'
    assert all(d['status'] == 'failed' for d in tools.minibpm_outputs(dict(raw, frames=1), '001', audio).values())


def test_tempocnn_output():
    raw = dict(bpm=128, confidence=0.9, topCandidates=[dict(bpm=128, probability=0.5)], sampleRate=44100,
               frames=441000, opendaw_wav_decoder='accepted', opendaw_left_equal=True)
    d = tools.tempocnn_outputs(raw, '001', AUDIO)['tcnn']
    assert d['status'] == 'ok' and d['bpm'] == 128.0 and d['provenance']['confidence'] == 0.9
    assert tools.tempocnn_outputs(dict(raw, error='boom'), '001', AUDIO)['tcnn']['status'] == 'failed'
    assert tools.tempocnn_outputs(dict(raw, sampleRate=48000), '001', AUDIO)['tcnn']['status'] == 'failed'


def test_tempo_terms_against_a_chart():
    ref = Reference.from_segments([Segment(1000.0, 500.0, 4), Segment(17000.0, 60000.0 / 150.0, 4)],
                                  np.concatenate([1000.0 + 500.0 * np.arange(32), 17000.0 + 400.0 * np.arange(40)]))
    whole = ob.tempo_terms(ref, 120.0, None)
    share120 = (17000.0 - 1000.0) / (ref.hi_ms - 1000.0)
    assert whole['acc1'] == pytest.approx(share120) and whole['acc2'] == pytest.approx(share120)
    assert ob.tempo_terms(ref, 60.0, None)['acc1'] == 0.0
    assert ob.tempo_terms(ref, 60.0, [(2000.0, 10000.0)])['acc2'] == pytest.approx(1.0)
    assert ob.tempo_terms(ref, 124.0, [(2000.0, 10000.0)])['acc1'] == 1.0      # 3.3 % off
    assert ob.tempo_terms(ref, 125.0, [(2000.0, 10000.0)])['acc1'] == 0.0      # 4.2 % off
    assert ob.tempo_terms(ref, 150.0, [(18000.0, 30000.0)])['ratio'] == pytest.approx(1.0)
    none = ob.tempo_terms(ref, 0.0, None)          # the tool stated no tempo: what the user gets scores 0
    assert none['acc1'] == 0.0 and none['acc2'] == 0.0 and math.isnan(none['ratio'])


def test_click_probe_and_offsets():
    x, truth = tools.click_probe(48000)
    assert x.shape[1] == 2 and np.array_equal(x[:, 0], x[:, 1])
    assert truth[0] == 1.0 and np.allclose(np.diff(truth), 0.5)
    onset = int(round(truth[3] * 48000))
    assert x[onset - 1, 0] == 0.0 and x[onset + 10, 0] != 0.0
    res = tools.probe_offsets(truth + 0.010, truth)
    assert res['median_ms'] == pytest.approx(10.0) and res['within20'] == 1.0 and res['n'] == 61
    assert tools.probe_offsets([], truth)['median_ms'] is None


def test_json_diff_kinds():
    a = {'x': 1.0, 'p': 'a/b', 'l': [1, 2], 'f': True}
    assert ob.json_diff(a, a) == []
    d = ob.json_diff(a, {'x': 1.0 + 1e-6, 'p': 'c/b', 'l': [1], 'f': False})
    kinds = sorted(k for k, *_ in d)
    assert kinds == ['string', 'value', 'value', 'value']


def test_oracle_beats_per_bar_values():
    assert ob.oracle_bpbs([5]) == [5] and ob.oracle_bpbs([1, 4, 7, 20]) == [2, 4, 7, 16] and ob.oracle_bpbs([]) == []


def _fake_tool(path: Path, body: str) -> str:
    path.write_text(f'#!{sys.executable}\nimport json, sys, time\n{body}\n')
    path.chmod(0o755)
    return str(path)


def test_tool_runs_write_one_output_per_arm_and_record_failures(tmp_path):
    """The driver's plumbing with stand-in tools: arms named from the runs, a run without output and a timeout
    recorded as failed outputs with their reasons, and every song keeps a status."""
    bulk, out = tmp_path / 'bulk', tmp_path / 'out'
    jobs = [dict(id=sid, list_id=sid, kind='song', wav=str(tmp_path / f'{sid}.wav'), rate=44100, channels=2,
                 frames=441000, pcm_sha256='ab' * 32, lead_ms=0.0, oracle_bpbs=[5], bpbs=[4, 5]) for sid in ('001', '002')]
    tools.dump(bulk / 'jobs.json', jobs)
    raw = _vamp_raw('RUN', [0.5, 1.0, 1.5], ['1', '2', '3'])
    ardour = _fake_tool(tmp_path / 'ardour9-lua', f"""
wav, rate, sess, prefix, bpbs = sys.argv[2:7]
if wav.endswith('002.wav'):
    time.sleep(30)
for run in ['qm-bbt-bpb4', 'qm-tempo']:          # no output for bpb 5
    raw = json.loads({json.dumps(json.dumps(raw))})
    raw['run'] = run
    if run == 'qm-bbt-bpb4':
        raw['parameters'] = dict(bpb=4.0)
    open(prefix + '__' + run + '.raw.json', 'w').write(json.dumps(raw))
""")
    common = ['--bulk', str(bulk), '--out', str(out), '--timeout', '5']
    ob.main(['ardour', '--ardour-lua', ardour] + common)
    maps = bulk / 'maps' / 'main'
    first = {arm: json.loads((maps / arm / '001.json').read_text()) for arm in ('qm-bbt', 'qm-bbt-m5', 'qm-tempo')}
    assert first['qm-bbt']['status'] == 'ok' and first['qm-tempo']['status'] == 'ok'
    assert first['qm-bbt-m5']['status'] == 'failed' and 'no output' in first['qm-bbt-m5']['error']
    second = json.loads((maps / 'qm-bbt' / '002.json').read_text())
    assert second['status'] == 'failed' and 'timed out' in second['error']
    mb = _fake_tool(tmp_path / 'minibpm_cli', "print(json.dumps(dict(sample_rate=44100, frames=441000, "
                                               "whole=dict(bpm=120.0, candidates=[120.0]), middle10=dict(bpm=120.0))))")
    ob.main(['minibpm', '--minibpm-bin', mb, '--ids', '001'] + common)
    # 120 BPM over ten seconds is 20 beats, 5 bars, taken to 8 bars: 192 BPM.
    assert json.loads((maps / 'mbpm-source' / '001.json').read_text())['bpm'] == 192.0
    assert json.loads((maps / 'mbpm-raw' / '001.json').read_text())['bpm'] == 120.0
    node = _fake_tool(tmp_path / 'node', "json.dump(dict(bpm=121, confidence=0.5, sampleRate=44100, frames=441000), "
                                         "open(sys.argv[-1], 'w'))")
    ob.main(['tempocnn', '--node', node, '--opendaw', str(tmp_path), '--model', '', '--harness', str(tmp_path),
             '--ids', '001'] + common)
    assert json.loads((maps / 'tcnn' / '001.json').read_text())['bpm'] == 121.0
    ob.main(['status'] + common)
    st = json.loads((out / 'run-status.json').read_text())
    assert st['per_song']['qm-bbt'] == {'001': 'ok', '002': 'failed', 'P44': 'not run', 'P48': 'not run'}
    assert st['counts']['tcnn'] == {'ok': 1, 'empty': 0, 'failed': 0, 'not run': 1}
    assert ob.failed_tools({'qm-bbt-m5': {str(i): 'not run' for i in range(48)},
                            'tcnn': {str(i): 'not run' for i in range(5)}}) == {'tcnn': 5}


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
@pytest.mark.parametrize('extensible', [False, True])
def test_tempocnn_runner_reads_the_left_channel_of_both_float_headers(tmp_path, extensible):
    """Without openDAW's packages the runner stops at the import, after reading the WAV; what it read is recorded."""
    x = np.stack([np.linspace(-0.5, 0.5, 2000), np.zeros(2000)], axis=1).astype(np.float32)
    wav = tmp_path / 'a.wav'
    tools.write_wav_float(wav, x, 44100, extensible=extensible)
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'not a model')
    out = tmp_path / 'out.json'
    subprocess.run(['node', str(OB / 'tempocnn_one.mjs'), str(tmp_path), str(model), str(wav), str(out)],
                   check=True, capture_output=True, timeout=60)
    res = json.loads(out.read_text())
    assert (res['sampleRate'], res['frames'], res['channels']) == (44100, 2000, 2)
    assert res['format_tag'] == (0xFFFE if extensible else 3) and res['channel_used'] == 0
    assert res['opendaw_wav_decoder'].startswith('rejected') and 'error' in res
    assert res['model_sha256'] == hashlib.sha256(b'not a model').hexdigest()
