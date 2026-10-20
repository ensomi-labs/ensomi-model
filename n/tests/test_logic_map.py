"""Smart Tempo maps read as music maps: bars, labels, the clock, and scoring against a chart."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.mapformat import MusicMap
from ensomi_n.mapformat.logic import SCHEMA, ax_check, bar_numbers, downbeats, read_logic_map, summarize
from ensomi_n.timing import Reference, Segment


def logic_json(times_s, *, anchor_s, signatures=(('0', '4/4'),), labels=True, mode='constant', bpm=None):
    """A normalized map as the capture writes it, with labels numbered as the editor numbers them."""
    times_s = [float(x) for x in times_s]
    sigs = [(float(a), s) for a, s in signatures]
    data = dict(schema=SCHEMA, song_id='900', source_filename='x.mp3', mode=mode, logic_version='12.4',
                native_downbeat_time_seconds=float(anchor_s),
                time_signature_events=[dict(audio_file_seconds=a, signature=s, native_conf=-1) for a, s in sigs],
                tempo_events=[dict(audio_file_seconds=times_s[0], bpm=bpm or 60.0 / (times_s[1] - times_s[0]),
                                   native_conf=-1)],
                beats=[dict(ordinal=i, audio_file_seconds=t, native_conf=-1, native_onset=1, raw_ax_label=None,
                            downbeat_from_ax=None) for i, t in enumerate(times_s)],
                validation=dict(status='pass'))
    if labels:
        bars = bar_numbers(data)
        down = set(downbeats(data).tolist())
        for i, b in enumerate(data['beats']):
            b['raw_ax_label'] = f'Downbeat, Bar {bars[i]}' if i in down else f'Bar {bars[i]} Beat {i}'
            b['downbeat_from_ax'] = i in down
    return data


def write(tmp_path, data, name='m.json'):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_bars_count_from_the_anchor_with_a_pickup():
    t = 0.25 + 0.5 * np.arange(14)
    data = logic_json(t, anchor_s=t[2], labels=False)
    np.testing.assert_array_equal(downbeats(data), [2, 6, 10])
    np.testing.assert_array_equal(bar_numbers(data)[:7], [0, 0, 1, 1, 1, 1, 2])


def test_three_four_and_a_signature_change_at_a_bar_line():
    t = 0.1 + 0.4 * np.arange(16)
    np.testing.assert_array_equal(downbeats(logic_json(t, anchor_s=t[0], signatures=(('0', '3/4'),),
                                                       labels=False)), [0, 3, 6, 9, 12, 15])
    data = logic_json(t, anchor_s=t[0], signatures=(('0', '4/4'), (str(t[8]), '3/4')), labels=False)
    np.testing.assert_array_equal(downbeats(data), [0, 4, 8, 11, 14])


def test_labels_agree_with_the_reading_and_a_wrong_label_is_counted():
    t = 0.3 + 0.6 * np.arange(12)
    data = logic_json(t, anchor_s=t[1])
    assert ax_check(data) == dict(labels=12, unreadable=0, downbeat_mismatch=0, bar_mismatch=0, beat_mismatch=0,
                                  ok=True)
    data['beats'][5]['raw_ax_label'] = 'Downbeat, Bar 2'          # the reading puts bar 2 on beat 5 ...
    data['beats'][4]['raw_ax_label'] = 'Bar 1 Beat 4'
    assert ax_check(data)['ok']
    data['beats'][6]['raw_ax_label'] = 'Downbeat, Bar 3'          # ... but bar 3 on beat 9, not 6
    res = ax_check(data)
    assert res['downbeat_mismatch'] == 1 and res['bar_mismatch'] == 1 and not res['ok']
    unlabelled = logic_json(t, anchor_s=t[1], labels=False)
    assert ax_check(unlabelled)['labels'] == 0 and not ax_check(unlabelled)['ok']


def test_the_map_is_on_the_decoded_clock_with_the_lead_added(tmp_path):
    t = 0.25 + 0.5 * np.arange(10)
    m = read_logic_map(write(tmp_path, logic_json(t, anchor_s=t[2])), lead_ms=12.0)
    assert isinstance(m, MusicMap) and m.clock == 'decoded'
    np.testing.assert_allclose(m.beats, 1000.0 * t + 12.0)
    np.testing.assert_array_equal(m.downbeats, [2, 6])
    assert m.extra['logic']['signatures'] == ['4/4'] and m.extra['logic']['lead_ms'] == 12.0
    with pytest.raises(ValueError, match='decoded clock'):
        m.on_osu_clock()
    np.testing.assert_allclose(m.on_osu_clock(-20.0).beats, 1000.0 * t + 12.0 - 20.0)
    with pytest.raises(ValueError, match='lead_ms'):
        read_logic_map(write(tmp_path, logic_json(t, anchor_s=t[2])), lead_ms=-1.0)


def test_scored_against_a_chart_with_the_one_constant(tmp_path):
    """A chart at 120 BPM; Smart Tempo's markers 20 ms after its lines on the decode. With c = -20 ms every line
    passes and the offset is 0; with the other sourced constant, -28.7 ms, the whole grid sits 8.7 ms early."""
    seg = Segment(1000.0, 500.0, 4)
    heads = 1000.0 + 500.0 * np.arange(64)
    ref = Reference.from_segments([seg], heads)
    t = (1000.0 + 500.0 * np.arange(-2, 70) + 20.0) / 1000.0
    m = read_logic_map(write(tmp_path, logic_json(t, anchor_s=t[2])))
    s = score(ref, m, decoded_to_osu_ms=-20.0)
    assert s.bl20 == pytest.approx(1.0) and s.offset_ms == pytest.approx(0.0, abs=1e-6)
    assert s.bars_start == pytest.approx(1.0) and s.level == pytest.approx(1.0)
    s2 = score(ref, m, decoded_to_osu_ms=-28.7)
    assert s2.bl20 == pytest.approx(1.0) and s2.bl5 == pytest.approx(0.0) and s2.offset_ms == pytest.approx(-8.7)
    s3 = score(ref, m, decoded_to_osu_ms=5.0)
    assert s3.bl20 == pytest.approx(0.0) and s3.offset_ms == pytest.approx(25.0)


def test_a_double_time_map_folds_and_a_three_four_reading_keeps_the_lines(tmp_path):
    """Smart Tempo at twice the chart's tempo folds to the chart's canonical lines; bars are its own."""
    ref = Reference.from_segments([Segment(1000.0, 500.0, 4)], 1000.0 + 500.0 * np.arange(64))
    t = (1000.0 + 250.0 * np.arange(-4, 140)) / 1000.0
    m = read_logic_map(write(tmp_path, logic_json(t, anchor_s=t[4], signatures=(('0', '3/4'),))))
    s = score(ref, m, decoded_to_osu_ms=0.0)
    assert s.bl20 == pytest.approx(1.0)
    assert s.level_classes['octave'] == pytest.approx(1.0)


def test_refused_inputs(tmp_path):
    t = 0.25 + 0.5 * np.arange(8)
    off = logic_json(t, anchor_s=t[2] + 0.01, labels=False)
    with pytest.raises(ValueError, match='nearest beat'):
        downbeats(off)
    with pytest.raises(ValueError, match='n/4'):
        downbeats(logic_json(t, anchor_s=t[0], signatures=(('0', '6/8'),), labels=False))
    bad = logic_json(t, anchor_s=t[0], labels=False)
    bad['beats'][3]['audio_file_seconds'] = bad['beats'][2]['audio_file_seconds']
    with pytest.raises(ValueError, match='strictly increasing'):
        read_logic_map(write(tmp_path, bad))
    wrong = logic_json(t, anchor_s=t[0], labels=False)
    wrong['schema'] = 'something-else'
    with pytest.raises(ValueError, match='schema'):
        read_logic_map(write(tmp_path, wrong))
    shuffled = logic_json(t, anchor_s=t[0], labels=False)
    shuffled['beats'][1]['ordinal'] = 5
    with pytest.raises(ValueError, match='ordinals'):
        read_logic_map(write(tmp_path, shuffled))


def test_summary(tmp_path):
    t = 0.2 + 0.6 * np.arange(20)
    s = summarize(write(tmp_path, logic_json(t, anchor_s=t[0], mode='variable')))
    assert s['mode'] == 'variable' and s['beats'] == 20 and s['bars'] == 5 and s['ax']['ok']
    assert s['median_bpm'] == pytest.approx(100.0) and s['signatures'] == ['4/4']
