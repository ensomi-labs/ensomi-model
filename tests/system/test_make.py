import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from ensomi_model.osu_core.hitobjects import parse_mania_hit_objects
from ensomi_model.system.copy import CopyConfig, CopyPath
from ensomi_model.system.make import System, clean_heads, difficulty_passes, parse_scope, resolve_requests
from ensomi_model.system.prior import Prior, fit_band

from .helpers import calibration, grid, skeleton, tiny_r2


def scopes(rng, n=400):
    return pd.DataFrame(dict(band=rng.integers(2, 6, n), length_s=rng.choice([8.0, 16.0, 32.0], n),
                             median_gap_ms=rng.uniform(80, 400, n), J_ex=rng.uniform(-0.4, 0.2, n),
                             S=rng.uniform(0.4, 0.95, n)))


def test_parse_scope_and_clean_heads():
    assert parse_scope('1000,12000,0.9,chordjack', 'jack') == dict(concept='jack', a_ms=1000.0, b_ms=12000.0, q=0.9,
                                                                    subtype='chordjack')
    with pytest.raises(ValueError):
        parse_scope('1000,12000,0.9,chordjack', 'stream')
    with pytest.raises(ValueError):
        parse_scope('1000,12000', 'stream')
    heads, dropped = clean_heads([5.0, 1.0, 2.5, 99.0, 50.0], 100.0)
    assert np.array_equal(heads, [1.0, 5.0, 50.0]) and dropped == 2


def test_requests_resolve_on_generated_rows():
    rng = np.random.default_rng(0)
    head, song = skeleton(rng, 200)
    cal = calibration()
    reqs = resolve_requests(head, song, [parse_scope('5000,15000,0.9', 'stream'), parse_scope('20000,30000,0.5', 'jack')],
                            3, cal, scopes(rng))
    s, j = reqs
    assert s['length_bin_s'] == 8.0 and s['neighbors'] > 0 and 'rows_carry_stream' in s
    assert j['reattack_floor_ms'] == 105.0 and isinstance(j['rows_allow_jack'], bool)
    assert s['band'][0] <= s['target'] <= s['band'][1] + 1e-9
    with pytest.raises(ValueError):          # overlapping scopes
        resolve_requests(head, song, [parse_scope('5000,15000,0.9', 'stream'), parse_scope('14000,30000,0.5', 'jack')],
                         3, cal, scopes(rng))


def test_difficulty_passes_map_clamp_and_correct():
    cal = calibration()
    calls = []

    def run(m):                                # a fake arrangement whose r grows with m
        calls.append(m)
        return 'a', 'g', dict(fallback_count=0)

    import ensomi_model.system.make as make
    r_of = {}
    make_song_r = make.song_r
    make.song_r = lambda head, song, a, g: 1.1 * calls[-1] ** 0.75
    try:
        a, g, tr, info = difficulty_passes(run, None, None, cal, 3, 0.9, base_r=1.2)
    finally:
        make.song_r = make_song_r
    values = cal['difficulty']['3']['values']
    assert np.isclose(info['target_r'], np.quantile(values, 0.9))
    assert info['domain'] == [0.8, 1.2] and 0.8 <= info['m_final'] <= 1.2
    assert len(info['passes']) == len(calls) <= 2
    assert info['clamped'] == (info['m_requested'] > 1.2 or info['m_requested'] < 0.8)
    assert info['base_r_source'] == 'prior' and info['above_tested_band']


class StubRows:
    def __init__(self, head, song, g, prepared=None):
        self.head, self.song, self.grid, self.prepared = head, song, g, prepared

    def prepare(self, audio):
        return self.prepared

    def heads(self, prepared, *, star=None, request=None, sections=(), seed=0, temperature=1.0):
        return dict(head_ms=self.head, K=len(self.head), grid=self.grid, segments=np.array([[0.0, 500.0, 4.0]]),
                    song_ms=self.song, request=2.3, beats={}, seconds=0.01)


def test_make_writes_a_playable_osz_and_its_receipt(tmp_path):
    rng = np.random.default_rng(1)
    head, song = skeleton(rng, 160)
    g = grid()
    X = rng.normal(size=(50, 4))
    Y = np.column_stack((np.full(50, 0.1), np.full(50, 1.4), np.full(50, 0.03), np.full(50, 0.01), np.full(50, 0.25)))
    prior = Prior(dict(method='median', bands={str(b): fit_band(X, Y) for b in range(2, 6)}))
    copy = CopyPath(CopyConfig())
    config = dict(default_q={str(b): 0.5 for b in range(2, 6)},
                  copy=dict(weights='copy.pt', default='on', strength=0.5, reason=None))
    system = System(config, StubRows(head, song, g), tiny_r2(), calibration(), prior, copy, scopes=scopes(rng))
    audio = tmp_path / 'song.ogg'
    audio.write_bytes(b'not really audio')
    prepared = dict(audio=audio, song=SimpleNamespace(features=rng.normal(size=(int(song / 10) + 10, 130)).astype(np.float16)),
                    grid=g, song_ms=song)
    out = tmp_path / 'out' / 'x.osz'
    receipt = system.make(str(audio), 3.4, out, ln='half', stream=['5000,15000,0.9'], jack=['20000,30000,0.5'],
                          seed=2, prepared=prepared)
    names = zipfile.ZipFile(out).namelist()
    assert sorted(names) == ['audio.ogg', 'generated.osu', 'receipt.json']
    saved = json.loads((out.with_suffix('') / 'receipt.json').read_text())
    assert saved['band'] == 3 and saved['r2_seed'] == 956 and saved['copy']['on'] and saved['copy']['strength'] == 0.5
    assert saved['controls']['ln']['mode'] == 'half' and np.isclose(saved['controls']['ln']['target_held_share'], 0.05)
    assert saved['controls']['difficulty']['q'] == 0.5 and saved['controls']['difficulty']['default']
    assert len(saved['controls']['sections']) == 2 and saved['export']['legal']
    assert saved['prior']['method'] == 'median' and saved['targets'][1] == pytest.approx(1.4 * saved['controls']['difficulty']['m_final'])
    parsed = parse_mania_hit_objects(out.with_suffix('') / 'generated.osu')
    assert np.array_equal(np.unique([o.start_time_ms for o in parsed]), head)
    text = (out.with_suffix('') / 'generated.osu').read_text()
    assert 'AudioFilename:audio.ogg' in text and 'Bookmarks:5000,15000,20000,30000' in text
    assert text.count('[Metadata]') == 1


def test_copy_setting_resolution():
    system = System(dict(copy=dict(weights='w.pt', default='off', strength=1.0, reason='jack rate above the band')),
                    None, None, None, None, copy=object())
    assert system.copy_setting() == (False, 1.0, 'jack rate above the band')
    assert system.copy_setting('on', 0.25)[:2] == (True, 0.25)
    with pytest.raises(ValueError):
        system.copy_setting('on', 1.5)
    with pytest.raises(ValueError):
        System(dict(copy={}), None, None, None, None).copy_setting('on')


def test_cli_main_reads_its_flags(tmp_path, monkeypatch, capsys):
    import ensomi_model.system.make as make
    rng = np.random.default_rng(4)
    head, song = skeleton(rng, 120)
    g = grid()
    X = rng.normal(size=(50, 4))
    Y = np.column_stack((np.full(50, 0.1), np.full(50, 1.4), np.full(50, 0.03), np.full(50, 0.01), np.full(50, 0.25)))
    prior = Prior(dict(method='median', bands={str(b): fit_band(X, Y) for b in range(2, 6)}))
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'x')
    prepared = dict(audio=audio, song=SimpleNamespace(features=np.zeros((int(song / 10) + 10, 130), np.float16)),
                    grid=g, song_ms=song)
    config = dict(default_q={str(b): 'off' for b in range(2, 6)}, copy=dict(default='off'))
    system = System(config, StubRows(head, song, g, prepared), tiny_r2(), calibration(), prior, None, scopes=scopes(rng))
    monkeypatch.setattr(make.System, 'load', classmethod(lambda cls, cfg: system))
    (tmp_path / 'system.json').write_text(json.dumps(config))
    out = tmp_path / 'o.osz'
    make.main(['--config', str(tmp_path / 'system.json'), '--audio', str(audio), '--star', '4.2', '--out', str(out),
               '--ln', 'tap', '--difficulty-q', '0.3', '--jack', '5000,15000,0.9', '--seed', '1'])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['band'] == 4 and printed['copy'] is False and printed['difficulty']['q'] == 0.3
    receipt = json.loads((tmp_path / 'o' / 'receipt.json').read_text())
    assert receipt['controls']['ln']['mode'] == 'tap' and receipt['controls']['ln']['ln_heads'] == 0
    assert receipt['r2_seed'] == 955 and receipt['controls']['sections'][0]['concept'] == 'jack'
    make.main(['--config', str(tmp_path / 'system.json'), '--audio', str(audio), '--star', '2.0', '--out', str(out),
               '--difficulty', 'default'])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['difficulty'] is None
