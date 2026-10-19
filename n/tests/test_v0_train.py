"""N1 v0 network, batches, whole-song inference and the resumable training job, on synthetic songs (CPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from ensomi_n.timing import Reference, Segment  # noqa: E402
from ensomi_n.v0 import data as D  # noqa: E402
from ensomi_n.v0.labels import V0Labels  # noqa: E402
from ensomi_n.v0.songs import memo_from_mel, memo_path  # noqa: E402

SMALL = dict(d=32, d_mid=32, d_mem=32, n_latents=4, heads=2, mid_layers=1)


def _song(tmp: Path, i: int, beat_ms: float, seconds: float = 40.0) -> D.Song:
    n = int(seconds * 100)
    rng = np.random.default_rng(i)
    mel = (rng.normal(-6, 0.5, (n, 128))).astype(np.float32)
    beats = np.arange(500.0, seconds * 1000 - 500, beat_ms)
    mel[np.round(beats / 10).astype(int)] += 4.0
    sha = f'{i:02d}' + 'a' * 62
    mp = tmp / 'mel' / 'x' / sha[:2] / f'{sha}.npy'
    mp.parent.mkdir(parents=True, exist_ok=True)
    np.save(mp, mel.astype(np.float16))
    me = memo_path(tmp, sha)
    me.parent.mkdir(parents=True, exist_ok=True)
    np.save(me, memo_from_mel(mel))
    ref = Reference.from_segments([Segment(500.0, beat_ms)], np.arange(500.0, seconds * 1000 - 600, beat_ms / 2))
    lab = V0Labels.from_reference(ref, sha)
    return D.Song(sha, str(mp), str(me), float(mel.mean()), float(mel.std()), n, lab, 1.0 + i, 'synthetic', ('S0',))


@pytest.fixture()
def songs(tmp_path):
    return [_song(tmp_path, i, b) for i, b in enumerate((500.0, 437.5, 612.0))]


def test_memo_shape_and_stretch(songs):
    s = songs[0]
    m = D.memo_input(s)
    assert m.shape == (s.frames // 4, 64)
    assert abs(len(D.memo_input(s, 1.1)) - int(len(m) * 1.1)) <= 1


def test_batch_forward_backward(songs):
    from ensomi_n.v0.model import V0
    sp = D.Sampler(songs, seed=0)
    x, tg, memo, mask, owner = sp.batch(2)
    assert x.shape == (4, D.CROP, 128) and owner.tolist() == [0, 0, 1, 1]
    assert memo.shape[0] == 2 and mask.any(axis=1).all()
    model = V0(**SMALL)
    xt, tt, mt, kt = D.to_torch(x, tg, memo, mask, 'cpu')
    mem = model.encoder.song_memory(mt, kt)[torch.from_numpy(owner)]
    out = model.heads(model.encoder(xt, mem))
    assert out['logits'].shape == (4, D.CROP, 3) and out['level'].shape == (4, D.CROP, 4)
    loss, parts = V0.loss(out, tt, dict(beat=5.0, half=5.0, down=8.0, bound=10.0))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if 'memory' in n and p.requires_grad)
    assert set(parts) >= {'beat', 'bound', 'level', 'phase', 'period', 'bar'}


def test_memory_changes_output_and_zero_mode(songs):
    from ensomi_n.v0.model import V0
    from ensomi_n.v0.infer import song_outputs
    torch.manual_seed(0)
    model = V0(**SMALL).eval()
    a = song_outputs(model, songs[0], 'cpu')
    b = song_outputs(model, songs[0], 'cpu', memory='other', other_song=songs[2])
    z = song_outputs(model, songs[0], 'cpu', memory='zero')
    assert a['beat'].shape == (songs[0].frames,) and a['level'].shape == (songs[0].frames, 4)
    assert np.abs(a['beat'] - b['beat']).max() > 0 and np.abs(a['beat'] - z['beat']).max() > 0


def test_windows_cover_every_frame():
    from ensomi_n.v0.infer import windows
    for n in (100, 2000, 2004, 5000, 12345):
        st = windows(n + (-n) % 4)
        cover = np.zeros(n + (-n) % 4, dtype=bool)
        for s in st:
            cover[s:s + D.CROP] = True
        assert cover.all()


def test_training_job_resumes(songs, tmp_path):
    from ensomi_n.v0 import train as T
    from ensomi_n.v0.data import fixed_crops
    from ensomi_n import probe
    crops = fixed_crops(songs[:2], per_song=1)
    ref = Reference.from_segments([Segment(500.0, 500.0)], np.arange(500.0, 39000.0, 250.0))
    items = [dict(song=songs[0], group='S0', item=songs[0].item, windows=probe.beat_windows(ref))]
    refs = dict(pos_weight=dict(beat=5.0, half=5.0, down=8.0, bound=10.0), probe_s0_chance=0.08, probe_s0_final0=0.8)
    run, ck = tmp_path / 'run', tmp_path / 'ck'
    old = (T.PROBE_EVERY_MIN, T.CKPT_EVERY_MIN, T.SONGS_PER_BATCH)
    T.PROBE_EVERY_MIN, T.CKPT_EVERY_MIN, T.SONGS_PER_BATCH = 0.02, 0.03, 1
    try:
        st = T.train(songs=songs, probe_crops=crops, probe_songs=songs, probe_items=items, refs=refs, run=run,
                     ck_dir=ck, seed=0, total_min=0.06, job_min=0.03, device='cpu', model_kw=SMALL)
        assert st['verdict'] in ('job_done', 'finished') and (run / 'status.json').exists()
        st2 = T.train(songs=songs, probe_crops=crops, probe_songs=songs, probe_items=items, refs=refs, run=run,
                      ck_dir=ck, seed=0, total_min=0.06, job_min=1.0, device='cpu', model_kw=SMALL)
        assert st2['step'] > st['step'] and st2['verdict'] == 'finished'
        assert (run / 'verdict.json').exists()
        lines = [json.loads(x) for x in (run / 'progress.jsonl').read_text().splitlines()]
        assert any(x.get('event') == 'probe' for x in lines)
    finally:
        T.PROBE_EVERY_MIN, T.CKPT_EVERY_MIN, T.SONGS_PER_BATCH = old
