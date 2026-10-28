"""Absolute checks for fractional resets, endpoints and label semantics."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from n1_redesign.probes.common import LATTICE, assert_open, interp, split
from n1_redesign.probes.labels import beat_labels, make_labels
from n1_redesign.probes.tokens import beat_tokens, frame_labels, frame_inputs, event_inputs


def test_split_is_specified_hash():
    for sha in ('a' * 64, 'b' * 64, 'c' * 64):
        h = int(hashlib.sha256(('n1r-probe:' + sha).encode()).hexdigest()[:8], 16) % 8
        assert split(sha) == ('probe-fit' if h in (0, 1) else 'probe-dev' if h == 2 else None)


def test_transitive_folder_rejected_before_read():
    e = dict(audio_sha256=set(), recordings=set(), set_ids=set(), folders={'dataset/open/7'})
    row = dict(sha256='unseen', memberships=[dict(audio_rel='dataset/open/7/a.mp3')])
    with pytest.raises(ValueError, match='Sealed folder'):
        assert_open(row, e)


def test_lattice_and_interpolation():
    assert len(LATTICE) == 24
    assert set(np.round(LATTICE * 48).astype(int)) == set(range(0, 48, 3)) | set(range(0, 48, 4))
    x = np.arange(20, dtype=np.float32)[:, None]
    np.testing.assert_array_equal(interp(x, [5, 125, 190]), [[.5], [12.5], [19]])


def test_mixed_is_ternary_unsnapped_excluded():
    b = np.arange(4) * 600.
    heads = np.array([0, 200, 300, 600, 900, 1200, 1377, 1800])
    y, n, bi = beat_labels(b, np.full(4, 600.), heads, np.array([1, 3, 2, 1, 2, 1, 0, 1]))
    np.testing.assert_array_equal(y, [1, 0, -1, -1])
    np.testing.assert_array_equal(n, [3, 2, 2, 1])


def test_fractional_reset_last_head_and_plain_stream(tmp_path):
    p = tmp_path / 'fixture.osu'
    p.write_text('''osu file format v14
[General]
AudioFilename: a.wav
Mode: 3
[Metadata]
Version: test
[Difficulty]
CircleSize: 4
[TimingPoints]
0,500,4,0,0,100,1,0
1125.5,400,3,0,0,100,1,0
1925.5,400,4,0,0,100,1,0
[HitObjects]
64,192,0,1,0,0:0:0:0:
64,192,250,1,0,0:0:0:0:
64,192,500,1,0,0:0:0:0:
64,192,1000,1,0,0:0:0:0:
64,192,1126,1,0,0:0:0:0:
64,192,1526,1,0,0:0:0:0:
64,192,1926,1,0,0:0:0:0:
64,192,2326,1,0,0:0:0:0:
''')
    # The rank lookup is optional; this fixture is intentionally outside ROOT.
    from n1_redesign.probes import labels as mod
    old = mod.ROOT
    mod.ROOT = tmp_path
    try:
        z = make_labels([str(p)], {})
    finally:
        mod.ROOT = old
    np.testing.assert_allclose(z['beats'], [0, 500, 1000, 1125.5, 1525.5, 1925.5, 2325.5])
    assert z['divisor'][-1] == 1
    assert z['head_beat'][-1] == len(z['beats']) - 1
    assert z['bib'][5] == 0
    token = beat_tokens(np.ones((300, 2), np.float32), z)
    np.testing.assert_array_equal(token, np.ones((len(z['beats']), 50)))


def test_attack_negatives_require_dense_beat():
    z = dict(beats=np.array([0., 500.]), length=np.array([500., 500.]),
             heads=np.array([0., 250., 700.]), count=np.array([2, 1]), head_beat=np.array([0, 0, 1]))
    y = frame_labels(z, 100)
    assert y[10] == 0
    assert y[60] == -1
    assert y[70] == 1
    assert y[73] == -1


def test_no_balanced_frames_keeps_feature_widths():
    z = dict(beats=np.array([0.]), length=np.array([500.]), bib=np.array([0]))
    d = frame_inputs(np.zeros((10, 160)), np.zeros((10, 23)),
                     np.zeros((10, 128)), np.array([], int), z)
    assert {k: v.shape for k, v in d.items()} == {
        'F': (0, 160), 'L': (0, 23), 'M': (0, 1280), 'Pos': (0, 29), 'flux': (0,)}


def test_event_features_interpolate_on_the_same_clock():
    z = dict(beats=np.array([0., 500.]), length=np.array([500., 500.]), bib=np.array([0, 1]))
    x = np.repeat(np.arange(100, dtype=np.float32)[:, None], 128, axis=1)
    d = event_inputs(x[:, :2], x[:, :1], x, np.array([105.]), z)
    np.testing.assert_allclose(d['F'], [[10.5, 10.5]])
    np.testing.assert_allclose(d['M'][0, :128], 6.5)
    np.testing.assert_allclose(d['M'][0, -128:], 1)


def test_standard_slider_repeat_and_tail_keep_ladder_heads(tmp_path, monkeypatch):
    from n1_redesign.probes import labels as mod
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    header = '''osu file format v14
[General]
AudioFilename: a.wav
Mode: 0
[Metadata]
Version: {version}
[Difficulty]
SliderMultiplier: 1.4
[TimingPoints]
0,500,4,0,0,100,1,0
[HitObjects]
'''
    objects = [
        '64,192,0,2,0,B|164:192,2,140,0|8|4,0:0|0:0|0:0,0:0:0:0:\n',
        ''.join(f'64,192,{t},1,0,0:0:0:0:\n' for t in (0, 1000)),
        ''.join(f'64,192,{t},1,0,0:0:0:0:\n' for t in (0, 250, 500, 750, 1000))]
    paths, ranks = [], {}
    for i, body in enumerate(objects):
        version = f'difficulty{i}'
        p = tmp_path / (version + '.osu')
        p.write_text(header.format(version=version) + body)
        paths.append(str(p)); ranks['.', version] = (i, i)
    z = make_labels(paths, ranks)
    np.testing.assert_allclose(z['retention'], [1, 0, .5, 0, 1])
    assert any((z['event_t'] == 500) & (z['event_bits'] == 8))
    assert any((z['event_t'] == 1000) & (z['event_bits'] == 4))
