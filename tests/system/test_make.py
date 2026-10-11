import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from ensomi_model.osu_core.hitobjects import parse_mania_hit_objects
from ensomi_model.system.copy import CopyConfig, CopyPath
from ensomi_model.system.make import (System, clean_heads, difficulty_passes, file_sha, parse_scope, resolve_requests,
                                     song_metadata, whole_ms_releases)
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
        return dict(head_ms=self.head, K=len(self.head), grid=self.grid, segments=np.array([[0.4, 500.0, 4.0]]),
                    song_ms=self.song, request=2.3, beats={}, seconds=0.01)


def test_make_writes_a_playable_osz_and_its_receipt(tmp_path):
    rng = np.random.default_rng(1)
    head, song = skeleton(rng, 160)
    head = head + 0.37                      # the Listener's grid gives fractional times
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
                          seed=2, title='A Song', artist='Someone', prepared=prepared)
    names = zipfile.ZipFile(out).namelist()
    assert sorted(names) == ['audio.ogg', 'generated.osu', 'receipt.json']
    saved = json.loads((out.with_suffix('') / 'receipt.json').read_text())
    assert saved['band'] == 3 and saved['r2_seed'] == 956 and saved['copy']['on'] and saved['copy']['strength'] == 0.5
    assert saved['controls']['ln']['mode'] == 'half' and np.isclose(saved['controls']['ln']['target_held_share'], 0.05)
    assert saved['controls']['difficulty']['q'] == 0.5 and saved['controls']['difficulty']['default']
    assert len(saved['controls']['sections']) == 2 and saved['export']['legal']
    assert saved['prior']['method'] == 'median' and saved['targets'][1] == pytest.approx(1.4 * saved['controls']['difficulty']['m_final'])
    parsed = parse_mania_hit_objects(out.with_suffix('') / 'generated.osu')
    assert np.array_equal(np.unique([o.start_time_ms for o in parsed]), np.round(head))
    assert all(float(t).is_integer() for o in parsed for t in (o.start_time_ms, o.end_time_ms))
    text = (out.with_suffix('') / 'generated.osu').read_text()
    assert 'AudioFilename:audio.ogg' in text and 'Bookmarks:5000,15000,20000,30000' in text
    assert text.count('[Metadata]') == 1 and 'Title:A Song\n' in text and 'Artist:Someone\n' in text
    timing = text.split('[TimingPoints]\n')[1].split('\n')[0]
    assert timing.startswith('0,500') and saved['metadata'] == dict(title='A Song', artist='Someone')


def test_whole_ms_releases_stay_between_their_rows():
    n = np.nan
    gap = [[n] * 4, [10.4, 15.6, 19.7, n], [29.9, 20.2, n, n]]
    out = whole_ms_releases(np.array([10.0, 20.0]), 30.6, gap)
    assert np.array_equal(out, [[n] * 4, [11, 16, 19, n], [30, 21, n, n]], equal_nan=True)


def test_song_metadata_from_the_cache(tmp_path):
    audio = tmp_path / 'audio.mp3'
    audio.write_bytes(b'abc')
    osu = tmp_path / 'x.osu'
    osu.write_text('osu file format v14\n[Metadata]\nTitle:unravel\nArtist:Ado\n')
    pd.DataFrame(dict(path=[str(osu)], audio_sha256=[file_sha(audio)])).to_parquet(tmp_path / 'index.parquet')
    assert song_metadata(audio, tmp_path / 'index.parquet') == ('unravel', 'Ado')
    other = tmp_path / 'b.mp3'
    other.write_bytes(b'x')
    assert song_metadata(other, tmp_path / 'index.parquet') == ('b', 'Unknown')
    assert song_metadata(audio, None) == ('audio', 'Unknown')


def test_copy_setting_resolution():
    system = System(dict(copy=dict(weights='w.pt', default='off', strength=1.0, reason='jack rate above the band')),
                    None, None, None, None, copy=object())
    assert system.copy_setting() == (False, 1.0, 'jack rate above the band')
    assert system.copy_setting('on', 0.25)[:2] == (True, 0.25)
    with pytest.raises(ValueError):
        system.copy_setting('on', 1.5)
    with pytest.raises(ValueError):
        System(dict(copy={}), None, None, None, None).copy_setting('on')


def test_load_resolves_a_model_directory(tmp_path, monkeypatch):
    import ensomi_model.audio_rows.rowsgen as rowsgen
    import ensomi_model.r2.bakeoff_panel as panel
    from ensomi_model.system import prior as prior_module
    seen = {}
    monkeypatch.setattr(rowsgen.Rows, 'load', classmethod(lambda cls, model, beatthis: seen.update(rows=model, beatthis=beatthis)))
    monkeypatch.setattr(panel, 'load_model', lambda p: seen.update(r2=p))
    monkeypatch.setattr(prior_module.Prior, 'load', classmethod(lambda cls, p: seen.update(prior=p)))
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('rows.pt', 'r2.pt', 'prior.json', 'beat.ckpt'):
        (model / name).write_bytes(name.encode())
    (model / 'cal.json').write_text(json.dumps(dict(scope_table='scopes.parquet')))
    config = dict(rows_model='rows.pt', r2='r2.pt', prior='prior.json', calibration='cal.json', beatthis='final0')
    system = System.load(config, model)
    assert seen == dict(rows=str(model / 'rows.pt'), beatthis='final0', r2=str(model / 'r2.pt'), prior=str(model / 'prior.json'))
    assert system.cal['scope_table'] == str(model / 'scopes.parquet') and system.config == config
    System.load(dict(config, beatthis='beat.ckpt'), model)
    assert seen['beatthis'] == str(model / 'beat.ckpt')


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
    monkeypatch.setattr(make.System, 'load', classmethod(lambda cls, cfg, model_dir=None: system))
    (tmp_path / 'system.json').write_text(json.dumps(config))
    out = tmp_path / 'o.osz'
    make.main(['--config', str(tmp_path / 'system.json'), '--audio', str(audio), '--star', '4.2', '--out', str(out),
               '--ln', 'tap', '--difficulty-q', '0.3', '--jack', '5000,15000,0.9', '--seed', '1'])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['band'] == 4 and printed['copy'] is False and printed['difficulty']['q'] == 0.3
    receipt = json.loads((tmp_path / 'o' / 'receipt.json').read_text())
    assert receipt['controls']['ln']['mode'] == 'tap' and receipt['controls']['ln']['ln_heads'] == 0
    assert receipt['r2_seed'] == 955 and receipt['controls']['sections'][0]['concept'] == 'jack'
    assert receipt['metadata'] == dict(title='song', artist='Unknown')
    make.main(['--config', str(tmp_path / 'system.json'), '--audio', str(audio), '--star', '2.0', '--out', str(out),
               '--difficulty', 'default'])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['difficulty'] is None
    make.main(['--model-dir', str(tmp_path), '--audio', str(audio), '--star', '2.0', '--out', str(out), '--difficulty', 'off'])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])['difficulty'] is None
