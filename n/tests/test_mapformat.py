"""The music map: round trip, validation, and `.osu` read as a map."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ensomi_n.eval import score
from ensomi_n.labels import read_set
from ensomi_n.mapformat import FORMAT, VERSION, MusicMap, load
from ensomi_n.timing import Segment


def full_map() -> MusicMap:
    return MusicMap(beats=np.array([512.0, 1012.0, 1512.0, 2012.0]), downbeats=[0],
                    confidence=[0.97, None, 0.4, 1.0], subdivision=[[0.1, 0.8], None, [0.0, 0.0], [1.0, 1.0]],
                    segments=(Segment(512.0, 500.0, 4), Segment(1512.0, 500.0, 3, bar_at_start=False)),
                    heads=[512.0, 762.0, 512.0], model={'name': 'n1-a', 'version': '0.1.0'}, audio='audio.mp3',
                    extra={'alternatives': [{'kind': 'octave'}]})


def test_round_trip_keeps_every_field(tmp_path):
    m = full_map()
    path = tmp_path / 'map.json'
    m.dump(path)
    data = json.loads(path.read_text())
    assert data['format'] == FORMAT and data['version'] == VERSION == 1
    assert data['confidence'] == [0.97, None, 0.4, 1.0]
    assert data['subdivision'][1] is None
    assert data['segments'][1] == {'start_ms': 1512.0, 'beat_ms': 500.0, 'meter': 3, 'bar_at_start': False}
    assert data['heads'] == [512.0, 762.0]
    assert data['alternatives'] == [{'kind': 'octave'}]
    back = load(path)
    assert back.to_json() == m.to_json()
    np.testing.assert_array_equal(back.downbeats, [0])
    assert math.isnan(back.confidence[1])


@pytest.mark.parametrize('kwargs, message', [
    (dict(beats=[0.0, 500.0, 500.0]), 'strictly increasing'),
    (dict(beats=[0.0, math.nan]), 'strictly increasing'),
    (dict(beats=[0.0, 500.0], confidence=[0.5]), 'one entry per beat'),
    (dict(beats=[0.0, 500.0], confidence=[0.5, 1.5]), 'probabilities'),
    (dict(beats=[0.0, 500.0], subdivision=[[0.5, 0.5], [0.5, -0.1]]), 'probabilities'),
    (dict(beats=[0.0, 500.0], downbeats=[2]), 'index beats'),
    (dict(segments=(Segment(0.0, 500.0),), confidence=[0.5]), 'need beats'),
    (dict(), 'beats or segments'),
    (dict(beats=[0.0], clock='wall'), 'clock'),
])
def test_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MusicMap(**kwargs)


def test_format_and_version_are_checked():
    with pytest.raises(ValueError, match='Not a music map'):
        MusicMap.from_json({'format': 'other', 'version': 1, 'beats': []})
    with pytest.raises(ValueError, match='version'):
        MusicMap.from_json({'format': FORMAT, 'version': 2, 'beats': []})


def test_empty_prediction_is_a_map():
    m = MusicMap.from_json({'format': FORMAT, 'version': 1, 'beats': []})
    assert len(m.timing(0.0, 1000.0)) == 0


def test_osu_as_a_map_scores_like_its_json(osu_writer, tmp_path):
    reds = [(0.0, 60000.0 / 128.3, 4), (5000.0, 60000.0 / 175.0, 7)]
    heads = [0.0, 2000.0, 5000.0, 9000.0]
    ref = read_set([osu_writer('ref/x.osu', reds, heads=heads)])
    pred_path = osu_writer('pred/x.osu', [(12.0, 60000.0 / 128.3, 4), (5012.0, 60000.0 / 175.0, 7)], heads=heads)
    from_osu = load(pred_path)
    assert from_osu.segments == (Segment(12.0, 60000.0 / 128.3, 4), Segment(5012.0, 60000.0 / 175.0, 7))
    np.testing.assert_array_equal(from_osu.heads, heads)
    from_osu.dump(tmp_path / 'x.json')
    a = score(ref, from_osu).summary()
    b = score(ref, load(tmp_path / 'x.json')).summary()
    assert a == pytest.approx(b, nan_ok=True)
    assert a['bl20'] == 1.0 and a['bl10'] == 0.0 and a['offset_ms'] == pytest.approx(12.0)


def test_from_beats_for_a_training_loop():
    m = MusicMap.from_beats(np.arange(8) * 500.0, downbeats=[0, 4], confidence=np.full(8, 0.9),
                            model='arch-a', version='0.1')
    assert m.model == {'name': 'arch-a', 'version': '0.1'} and m.clock == 'osu'
    assert m.timing(0.0, 3500.0).bar_starts().tolist() == [0.0, 2000.0]
