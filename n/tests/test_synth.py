"""Synthetic suite v0: absolute checks of the generator against its own ground truth.

- Rendered onsets sit at their ground-truth times within one sample, measured
  from the audio by a simple detector on each isolated layer, edits included.
- 7/8 with a red line per bar reads back through `ensomi_n.labels` and
  `ensomi_n.mapformat` with the generator's beats and downbeats.
- Ramps integrate to the right beat times (an independent numeric integral).
- A seed reproduces the clip byte for byte; another seed changes it.
- The ground truth scores itself 1 through `ensomi_n.eval`, and a half-beat
  shift 0.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.labels import read_set
from ensomi_n.mapformat import load
from ensomi_n.synth import SETTINGS, STRATA, build, make, write_clip
from ensomi_n.synth.detect import onsets
from ensomi_n.synth.e1 import bt_map, r2_map, reference, score_clip
from ensomi_n.synth.render import SAMPLE_RATE, render
from ensomi_n.synth.truth import ground_truth, music_map, osu_text, subdivision_class

SAMPLE_MS = 1000.0 / SAMPLE_RATE

ONSET_SETTINGS = ['control_click', 'control_a', 'edit_cut', 'edit_stutter', 'edit_tapestop', 'ramp_rit',
                  'rubato_high', 'mixsub_sim', 'meter_7_8_reset', 'poly_3_2', 'free_time']


def test_every_stratum_has_settings_and_the_brief_strata_exist():
    strata = {s.stratum for s in SETTINGS.values()}
    assert strata == set(STRATA)
    for st in ('S0', 'S1', 'S2', 'S3s', 'S3r', 'S3p', 'S9', 'S8', 'S6b', 'S6s', 'Sy'):
        assert st in strata
    assert sum(s.stratum == 'S0' for s in SETTINGS.values()) >= 3


@pytest.mark.parametrize('setting', ONSET_SETTINGS)
def test_rendered_onsets_sit_at_their_truth_within_a_sample(setting):
    song = build(setting, 0)
    gt = ground_truth(song, 60000.0)
    tape = [(a, b) for a, b, k in gt['no_pulse'] if 'tapestop' in k]
    checked = 0
    for name, lay in song.layers.items():
        if lay.voice == 'pad':
            continue
        x = render(song, layers=[name], fx=False)
        found = onsets(x, SAMPLE_RATE)
        truth = np.array(gt['layers'][name]['onsets_ms'])
        keep_t = np.array([not any(a <= t < b for a, b in tape) for t in truth], dtype=bool)
        keep_f = np.array([not any(a <= t < b for a, b in tape) for t in found], dtype=bool)
        truth, found = truth[keep_t], found[keep_f]
        truth = truth[truth < 60000.0 - SAMPLE_MS]
        assert len(found) == len(truth), (name, len(found), len(truth))
        if len(truth):
            err = np.abs(found - truth)
            assert err.max() < SAMPLE_MS + 1e-6, (name, float(err.max()))
            checked += len(truth)
    assert checked > 100


def test_onsets_after_an_edit_move_with_it():
    song = build('edit_cut', 0)
    gt = ground_truth(song, 60000.0)
    kick = np.array(gt['layers']['kick']['onsets_ms'])
    src = np.array(sorted(n.t for n in song.notes if n.layer == 'kick'))
    # Cuts remove time and inserts add it, so the heard kicks are not the source kicks.
    assert len(kick) != len(src) or np.max(np.abs(kick - src[:len(kick)])) > 50.0


def test_seven_eight_resets_read_back_through_labels_and_mapformat(tmp_path):
    clip = make('meter_7_8_reset', 0)
    write_clip(clip, tmp_path / 'audio', tmp_path / 'truth')
    gt = clip.truth
    t = np.array(gt['beats']['t_ms'])
    down = np.array(gt['beats']['downbeat'])
    segs = gt['segments']
    assert len(segs) == int(down.sum())          # a red line at every bar
    assert all(abs(s['beat_ms'] - segs[0]['beat_ms']) < 1e-9 for s in segs)
    bar_ms = np.diff([s['start_ms'] for s in segs])
    assert np.allclose(bar_ms, 3.5 * segs[0]['beat_ms'])
    # Through the reader.
    ref = read_set(tmp_path / 'truth' / clip.id / 'truth.osu')
    rb = ref.timing.beats
    inside = (rb >= t[0] - 1e-6) & (rb <= t[-1] + 1e-6)
    assert np.allclose(rb[inside], t, atol=1e-6)
    assert np.array_equal(ref.timing.downbeat[inside], down)
    # Through the music map.
    m = load(tmp_path / 'truth' / clip.id / 'map.json')
    assert m.clock == 'decoded'
    timing = m.timing(t[0], t[-1])
    assert np.allclose(timing.beats, t, atol=1e-9)
    assert np.array_equal(timing.bar_starts(), t[down])
    # The same canonical lines from both: every bar start, then every quarter inside the bar.
    lines_osu = ref.timing.canonical().t
    lines_map = timing.canonical().t
    lo, hi = t[0], t[-1]
    a = lines_osu[(lines_osu >= lo - 1e-6) & (lines_osu <= hi + 1e-6)]
    b = lines_map[(lines_map >= lo - 1e-6) & (lines_map <= hi + 1e-6)]
    assert np.allclose(a, b, atol=1e-6)
    assert len(a) == len(t)


def test_seven_eight_in_eighths_puts_bar_starts_on_canonical_lines():
    gt = ground_truth(build('meter_7_8', 0), 60000.0)
    t = np.array(gt['beats']['t_ms'])
    down = np.array(gt['beats']['downbeat'])
    m = music_map(gt, 'x.wav', {})
    lines = m.timing(t[0], t[-1]).canonical().t
    for b in t[down]:
        assert np.min(np.abs(lines - b)) < 1e-6
    # Seven eighths per bar give four lines: 0, 2, 4, 6 eighths.
    first, second = t[down][:2]
    assert np.sum((lines >= first - 1e-6) & (lines < second - 1e-6)) == 4
    assert gt['segments'] and len(gt['segments']) == 1 and gt['segments'][0]['meter'] == 7


@pytest.mark.parametrize('setting', ['ramp_accel', 'ramp_rit'])
def test_ramps_integrate_to_the_right_beat_times(setting):
    song = build(setting, 1)
    ramps = [p for p in song.tm.pieces if p.kind == 'ramp']
    assert ramps
    for p in ramps:
        n = int(round(p.b1 - p.b0))
        T = p.t1 - p.t0
        dt = 0.01
        tt = np.arange(0.0, T + dt, dt)
        bpm = p.bpm0 + (p.bpm1 - p.bpm0) * tt / T
        rate = bpm / 60000.0
        phi = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2.0 * np.diff(tt))])
        assert abs(np.interp(T, tt, phi) - n) < 1e-6
        k = np.arange(1, n)
        expect = np.interp(k, phi, tt) + p.t0
        got = np.array([song.tm.time(p.b0 + j) for j in k])
        assert np.max(np.abs(got - expect)) < 0.01
    gt = ground_truth(song, 60000.0)
    assert gt['segments'] is None and gt['segments_note']
    bpm = np.array(gt['beats']['bpm'])
    r = ramps[0]
    assert bpm.min() >= min(r.bpm0, r.bpm1) - 1.0 and bpm.max() <= max(r.bpm0, r.bpm1) + 1.0


def test_seed_reproduces_bytes_and_another_seed_differs(tmp_path):
    a = write_clip(make('control_click', 0), tmp_path / 'a', tmp_path / 'ta')
    b = write_clip(make('control_click', 0), tmp_path / 'b', tmp_path / 'tb')
    c = write_clip(make('control_click', 1), tmp_path / 'c', tmp_path / 'tc')
    assert a['audio_sha256'] == b['audio_sha256']
    assert (tmp_path / 'a' / 'control_click-s0.wav').read_bytes() == (tmp_path / 'b' / 'control_click-s0.wav').read_bytes()
    assert (tmp_path / 'ta' / 'control_click-s0' / 'map.json').read_bytes() == \
        (tmp_path / 'tb' / 'control_click-s0' / 'map.json').read_bytes()
    assert a['audio_sha256'] != c['audio_sha256']
    assert a['n_samples'] == 60 * SAMPLE_RATE


@pytest.mark.parametrize('setting', ['control_a', 'meter_7_8_reset', 'meter_alt_78_44', 'tempo_short', 'ramp_accel',
                                     'edit_cut', 'beatless_tempo', 'level_80_160'])
def test_truth_scores_itself_one_and_a_half_beat_shift_zero(setting):
    gt = ground_truth(build(setting, 1), 60000.0)
    m = music_map(gt, 'x.wav', {})
    for c in (0.0, -20.0):
        s = score_clip(m, m, c)
        assert s['bl20'] == 1.0 and s['bl5'] == 1.0
        assert abs(s['offset_ms']) < 1e-9
        assert s['bars_start'] == 1.0 and s['bars_same'] == 1.0
        assert s['level_fit'] == 1.0
    shifted = m.on_osu_clock(0.0)
    t = np.array(gt['beats']['t_ms'])
    # Half way to the next beat (a 7/8 bar's short last beat moves by a quarter of a quarter note).
    half = 0.5 * np.append(np.minimum(np.diff(t), np.array(gt['beats']['step_ms'])[:-1]), gt['beats']['step_ms'][-1])
    from ensomi_n.mapformat import MusicMap
    moved = MusicMap.from_beats(shifted.beats + half, shifted.downbeats, clock='decoded')
    assert score_clip(m, moved, 0.0)['bl20'] == 0.0


def test_beatless_span_has_no_beats_and_a_pulse_on_both_sides():
    gt = ground_truth(build('beatless', 0), 60000.0)
    t = np.array(gt['beats']['t_ms'])
    (a, b, kind), = gt['no_pulse']
    assert kind == 'beatless' and b - a >= 12000.0
    assert not np.any((t > a + 1e-6) & (t < b - 1e-6))
    assert np.sum(t < a) >= 16 and np.sum(t >= b) >= 16
    assert abs(t[t >= b][0] - b) < 1e-6           # the re-entry is a beat at the gap's end
    (sa, sb), = gt['spans']
    assert abs(sa - b) < 1e-6                       # the stratum span is the re-entry window


def test_stutter_keeps_the_grid_and_marks_its_events():
    song = build('edit_stutter', 0)
    gt = ground_truth(song, 60000.0)
    assert len(gt['segments']) == 1                 # a beat repeat replaces audio in place
    t = np.array(gt['beats']['t_ms'])
    d = np.diff(t)
    assert np.max(np.abs(d - d[0])) < SAMPLE_MS + 1e-6
    eff = np.array(gt['layers']['kick']['effect'])
    assert eff.sum() > 0 and len(gt['effects']) == song.params['stutters']


def test_cuts_and_inserts_are_resets():
    song = build('edit_cut', 0)
    gt = ground_truth(song, 60000.0)
    n_edits = len(song.params['edits'])
    # Each edit starts a red line off the previous grid (and one more at the next downbeat when the
    # edit falls inside a bar).
    assert len(gt['segments']) >= n_edits + 1
    t = np.array(gt['beats']['t_ms'])
    assert np.min(np.diff(t)) < 0.8 * np.median(np.diff(t))


def test_subdivision_classes():
    assert subdivision_class([0, '1/6', '1/8']) == ('mixed', 24)
    assert subdivision_class([0, '1/3', '1/4']) == ('mixed', 12)
    assert subdivision_class([0, '1/6']) == ('triple', 6)
    assert subdivision_class([0, '1/8']) == ('duple', 8)
    assert subdivision_class([0, '1/2']) == ('plain', 2)
    assert subdivision_class(['16/25']) == ('finer', 0)
    gt = ground_truth(build('mixsub_sim', 0), 60000.0)
    cls = np.array(gt['subdivision_class'])
    div = np.array(gt['subdivision_divisor'])
    assert np.all(div[cls == 'mixed'] == 24) and (cls == 'mixed').mean() > 0.5
    gt = ground_truth(build('mixsub_alt', 0), 60000.0)
    cls = gt['subdivision_class']
    assert set(cls) == {'triple', 'duple'}
    beat = gt['beats']['beat_in_bar']
    assert all(cls[k] != cls[k + 1] for k in range(40) if beat[k + 1] != 0)      # alternating inside each bar


def test_prediction_maps_for_the_arms():
    gt = ground_truth(build('control_a', 1), 60000.0)
    m = music_map(gt, 'x.wav', {})
    t = np.array(gt['beats']['t_ms'])
    down = np.flatnonzero(gt['beats']['downbeat'])
    # A BeatThis-like beat list, decoded, on 20 ms frames and 8 ms late: BL-20 1, level_fit 1.
    bt = bt_map(dict(beats_s=(np.round((t + 8.0) / 20.0) * 20.0 / 1000.0).tolist(),
                     downbeats_s=(np.round((t[down] + 8.0) / 20.0) * 20.0 / 1000.0).tolist()))
    s = score_clip(m, bt, -20.0)
    assert s['bl20'] == 1.0 and s['bars_start'] == 1.0 and s['level_fit'] == 1.0
    # R2's red line on osu!'s clock, 25 ms early: scored with c = -25, BL-20 1.
    step = gt['segments'][0]['beat_ms']
    r2 = r2_map([[t[0] - 25.0, step, 4]])
    assert score_clip(m, r2, -25.0)['bl20'] == 1.0
    assert score_clip(m, r2, 0.0)['bl20'] == 0.0     # the same map with the convention left in: 25 ms off


def test_cli_render_writes_manifest(tmp_path):
    from ensomi_n.synth.__main__ import main
    assert main(['render', '--audio', str(tmp_path / 'a'), '--truth', str(tmp_path / 't'), '--settings',
                 'control_click', '--seeds', '1', '--commit', 'abc']) == 0
    rows = [json.loads(x) for x in (tmp_path / 't' / 'manifest.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['commit'] == 'abc'
    import hashlib
    assert hashlib.sha256((tmp_path / 'a' / rows[0]['audio']).read_bytes()).hexdigest() == rows[0]['audio_sha256']
    ref = read_set(tmp_path / 't' / rows[0]['id'] / 'truth.osu')
    assert len(ref.timing.beats) >= rows[0]['n_beats']
