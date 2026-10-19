"""End-to-end smoke of the v0 evaluation command on a fake project (two songs, CPU, a tiny model)."""
from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pa = pytest.importorskip('pyarrow')

SMALL = dict(d=32, d_mid=32, d_mem=32, n_latents=4, heads=2, mid_layers=1)


def _osu(path: Path, offset: float, beat: float, n_beats: int, meter: int = 4):
    heads = [offset + k * beat / 2 for k in range(2 * n_beats)]
    lines = ['osu file format v14', '[General]', 'AudioFilename: audio.mp3', 'Mode: 3', '[Metadata]',
             'Version: X', 'Creator: someone', 'BeatmapSetID: 1', '[TimingPoints]',
             f'{offset},{beat},{meter},1,0,100,1,0', '[HitObjects]'] + [f'64,192,{int(h)},1,0,0:0:0:0:' for h in heads]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def test_eval_main_end_to_end(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from ensomi_n.features.cache import mel_path
    from ensomi_n.labels import read_set
    from ensomi_n.v0 import evaluate as E
    from ensomi_n.v0.labels import V0Labels
    from ensomi_n.v0.model import V0
    from ensomi_n.v0.songs import memo_from_mel, memo_path
    root = tmp_path
    bulk = root / 'bulk'
    items, index, rows, recs = [], {}, [], []
    for i, beat in enumerate((500.0, 400.0, 600.0)):
        sha = f'{i}' * 64
        chart = f'dataset/s{i}/x.osu'
        _osu(root / chart, 1000.0, beat, int(50000 / beat))
        n = 6000
        mel = np.random.default_rng(i).normal(-6, 1, (n, 128)).astype(np.float32)
        mel[np.round(np.arange(1000.0, 55000.0, beat) / 10).astype(int)] += 4
        mp = mel_path(bulk / 'mel', sha)
        mp.parent.mkdir(parents=True, exist_ok=True)
        np.save(mp, mel.astype(np.float16))
        me = memo_path(bulk, sha)
        me.parent.mkdir(parents=True, exist_ok=True)
        np.save(me, memo_from_mel(mel))
        index[sha] = dict(sha=sha, frames=n, mean=float(mel.mean()), std=float(mel.std()))
        ref = read_set([root / chart], key=sha)
        lab = V0Labels.from_reference(ref, sha)
        lp = bulk / 'labels-v0b' / sha[:2] / f'{sha}.json'
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(lab.to_json()))
        it = dict(audio_sha256=sha, audio_rel=f'dataset/s{i}/audio.mp3', charts=[chart], rec=f'rec{i}', bt_key=sha)
        items.append(it)
        rows.append(dict(item=sha, audio_sha256=sha, audio_rel=it['audio_rel'], rec=it['rec'], ok=True,
                         role='val_hard' if i < 2 else 'cal', strata=['S0'], hard=False, reader_flags=[]))
        recs.append(dict(rec=it['rec'], charts=[chart], role='cal'))
        z = np.zeros(n // 2)
        z[np.round((np.arange(1000.0, 55000.0, beat) + 30) / 20).astype(int)] = 6.0
        bt = bulk / 'bt-final0-abc'
        bt.mkdir(parents=True, exist_ok=True)
        np.savez(bt / f'{sha}.npz', beat=z - 3, down=z - 3)
        r2 = bulk / 'r2-export'
        r2.mkdir(parents=True, exist_ok=True)
        (r2 / f'{sha}.json').write_text(json.dumps(dict(red_lines=[[1000.0, beat, 4]])))
    pq.write_table(pa.table(dict(path=[it['charts'][0] for it in items], api_tag_ids=[[] for _ in items],
                                 api_status=['ranked'] * 3)), str(root / 'corpus.parquet'))
    (bulk / 'song-index-v0.json').write_text(json.dumps(index))
    r0 = root / 'r0'
    r0.mkdir()
    (r0 / 'split.json').write_text(json.dumps(dict(panel=items[:2], val_hard=items[:2])))
    (r0 / 'recordings.json').write_text(json.dumps(recs))
    data = root / 'data'
    data.mkdir()
    (data / 'curated.json').write_text(json.dumps(rows))
    run = root / 'runs' / 'v0-test'
    run.mkdir(parents=True)
    (run / 'refs.json').write_text(json.dumps(dict(pos_weight=dict(beat=5.0, half=5.0, down=8.0, bound=10.0))))
    ck = root / 'bulk-runs' / 'v0-test'
    ck.mkdir(parents=True)
    torch.manual_seed(0)
    for t in (100, 200):
        torch.save(dict(model=V0(**SMALL).state_dict(), model_kw=SMALL), ck / f'ckpt-{t:06d}.pt')
    a = types.SimpleNamespace(data=str(data), run=str(run), r0=str(r0), bulk=str(bulk), bulk_runs=str(root / 'bulk-runs'),
                              root=str(root), processes=2, device='cpu', arms='v0,v0f1,v0zero,v0other,final0,r2,oracle',
                              checkpoints='last2', panels='panel', out='', decode_kw='', ablation=True, limit=0, tune=False,
                              corpus='corpus.parquet', tags='none.json')
    mod = E
    mod.main(a)
    out = run / 'eval'
    t = json.loads((out / 'compact-panel.json').read_text())
    arms = set(t['tables'])
    assert {'r2', 'final0', 'oracle', 'v0zero', 'v0other', 'v0@000200', 'v0@000100', 'v0f1@000200'} <= arms
    assert t['tables']['oracle']['all']['bl20']['value'] > 0.99
    assert t['tables']['r2']['all']['bl20']['value'] > 0.99
    assert 'bl20_nonfree' in t['tables']['oracle']['all']
    assert t['meta']['n_errors'] == 0
    pair = json.loads((out / 'paired-panel.json').read_text())
    assert any(k.startswith('v0@000200-r2:bl20_nonfree') for k in pair)
    ab = json.loads((out / 'ablation.json').read_text())
    assert {'own', 'zero', 'other1', 'other7', 'null_other1_minus_other7'} <= set(ab)
    a.tune, a.decode_kw = True, json.dumps({'base': {}, 'reset7': {'reset_penalty': 7.0}})
    mod.main(a)
    tune = json.loads((out / 'tune.json').read_text())
    assert set(tune['results']) == {'base', 'reset7'}
    assert E.pick_settings(out) in ({}, {'reset_penalty': 7.0})
