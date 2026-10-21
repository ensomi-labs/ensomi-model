"""Open-tool outputs read as music maps: beat lists, one-tempo grids, empty and failed runs, the clock."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.mapformat import MusicMap
from ensomi_n.mapformat.open_tools import SCHEMA, OpenRunFailed, load_open, read_open_map, stated_bpm
from ensomi_n.timing import Reference, Segment

REF = Reference.from_segments([Segment(1000.0, 500.0, 4)], 1000.0 + 500.0 * np.arange(64))


def out(tmp_path, name='o.json', **fields):
    data = dict(schema=SCHEMA, arm='qm-bbt', tool='ardour-vamp', song_id='900', status='ok', kind='beats',
                audio=dict(pcm_sha256='0' * 64, frames=1, sample_rate_hz=44100, channels=2, channel_used=0))
    data.update(fields)
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_beats_with_bars_score_through_the_one_constant(tmp_path):
    """Beats 20 ms after the chart's lines on the decode, bars every 4 from the chart's first bar line."""
    t = (1000.0 + 500.0 * np.arange(-2, 70) + 20.0) / 1000.0
    p = out(tmp_path, beats_s=t.tolist(), downbeats=list(range(2, 72, 4)))
    m = read_open_map(p, lead_ms=0.0)
    assert m.clock == 'decoded' and m.model['name'] == 'qm-bbt' and m.extra['open']['kind'] == 'beats'
    with pytest.raises(ValueError, match='decoded clock'):
        m.on_osu_clock()
    s = score(REF, m, decoded_to_osu_ms=-20.0)
    assert s.bl20 == pytest.approx(1.0) and s.offset_ms == pytest.approx(0.0, abs=1e-6)
    assert s.bars_start == pytest.approx(1.0) and s.level == pytest.approx(1.0)
    np.testing.assert_allclose(read_open_map(p, lead_ms=12.0).beats, 1000.0 * t + 12.0)


def test_beats_without_bars_state_no_bars(tmp_path):
    t = (1000.0 + 500.0 * np.arange(-2, 70)) / 1000.0
    m = read_open_map(out(tmp_path, arm='qm-tempo', beats_s=t.tolist(), downbeats=None))
    s = score(REF, m, decoded_to_osu_ms=0.0)
    assert s.bl20 == pytest.approx(1.0) and not s.pred_has_bars and s.bars_start == pytest.approx(0.0)


def test_a_tempo_grid_lies_where_the_host_anchors_it(tmp_path):
    """One BPM laid from the file start: the level term does not see the phase, BL-20 does."""
    on_phase = out(tmp_path, 'a.json', kind='tempo', bpm=120.0, grid=dict(anchor_s=0.020, meter=4))
    m = read_open_map(on_phase)
    assert m.segments == (Segment(20.0, 500.0, 4, True),) and m.beats is None
    s = score(REF, m, decoded_to_osu_ms=-20.0)
    # The host's bars run every 4 beats from the anchor: 0, 2000, ... ms, two beats off the chart's (1000, 3000).
    assert s.bl20 == pytest.approx(1.0) and s.level == pytest.approx(1.0) and s.bars_start == pytest.approx(0.0)
    on_bars = Reference.from_segments([Segment(2000.0, 500.0, 4)], 2000.0 + 500.0 * np.arange(64))
    assert score(on_bars, m, decoded_to_osu_ms=-20.0).bars_start == pytest.approx(1.0)
    off_phase = read_open_map(out(tmp_path, 'b.json', kind='tempo', bpm=120.0, grid=dict(anchor_s=0.270, meter=4)))
    s2 = score(REF, off_phase, decoded_to_osu_ms=-20.0)
    assert s2.bl20 == pytest.approx(0.0) and s2.level == pytest.approx(1.0)
    half = read_open_map(out(tmp_path, 'c.json', kind='tempo', bpm=60.0, grid=dict(anchor_s=0.020, meter=4)))
    assert score(REF, half, decoded_to_osu_ms=-20.0).level_classes['octave'] == pytest.approx(1.0)
    assert stated_bpm(load_open(on_phase)) == 120.0


def test_empty_runs_score_zero_and_failed_runs_refuse(tmp_path):
    m = read_open_map(out(tmp_path, 'e.json', status='empty', kind='tempo', bpm=0.0))
    assert isinstance(m, MusicMap) and len(m.beats) == 0
    s = score(REF, m, decoded_to_osu_ms=-20.0)
    assert s.bl20 == pytest.approx(0.0) and s.level == pytest.approx(0.0)
    assert np.isnan(stated_bpm(load_open(tmp_path / 'e.json')))
    with pytest.raises(OpenRunFailed, match='timed out'):
        read_open_map(out(tmp_path, 'f.json', status='failed', kind=None, error='timed out after 600 s'))


def test_refused_outputs(tmp_path):
    with pytest.raises(ValueError, match='schema'):
        read_open_map(out(tmp_path, schema='other', beats_s=[1.0, 2.0]))
    with pytest.raises(ValueError, match='strictly increasing'):
        read_open_map(out(tmp_path, beats_s=[1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match='status "empty"'):
        read_open_map(out(tmp_path, beats_s=[]))
    with pytest.raises(ValueError, match='indices'):
        read_open_map(out(tmp_path, beats_s=[1.0, 2.0], downbeats=[0, 2]))
    with pytest.raises(ValueError, match='positive tempo'):
        read_open_map(out(tmp_path, kind='tempo', bpm=0.0, grid=dict(anchor_s=0.0, meter=4)))
    with pytest.raises(ValueError, match='grid'):
        read_open_map(out(tmp_path, kind='tempo', bpm=120.0))
    with pytest.raises(ValueError, match='status'):
        read_open_map(out(tmp_path, status='done'))
    with pytest.raises(ValueError, match='lead_ms'):
        read_open_map(out(tmp_path, beats_s=[1.0, 2.0]), lead_ms=-1.0)
