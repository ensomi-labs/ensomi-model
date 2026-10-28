"""Feature capture preserves context selection, the padded tail and sealed exclusions."""
from types import SimpleNamespace

import numpy as np
import pytest

from ensomi_n.v0.featcache import check_membership


@pytest.mark.parametrize('blocked', ['audio', 'set', 'set_string', 'recording', 'key', 'path', 'foreign'])
def test_exclusions(blocked):
    row = dict(audio_sha256='a', rec='r', set_id=7, census_key='k', source='4k',
               audio_rel='dataset/0/7/audio.mp3')
    ex = dict(audio_sha256=set(), set_ids=set(), recordings=set(), keys=set())
    recs = []
    if blocked == 'audio':
        ex['audio_sha256'].add('a')
    elif blocked == 'set':
        ex['set_ids'].add(7)
    elif blocked == 'set_string':
        ex['set_ids'].add('7')
    elif blocked == 'recording':
        ex['recordings'].add('r')
    elif blocked == 'key':
        ex['keys'].add('k')
    elif blocked == 'path':
        recs.append(dict(rec='other', audio_rels=[row['audio_rel']]))
    else:
        row['audio_rel'] = 'dataset/netease/gold/audio.mp3'
    with pytest.raises(ValueError):
        check_membership([row], ex, recs)


@pytest.mark.parametrize('frames', [53, 4001, 19003])
def test_capture_uses_evaluator_windows_and_clock(tmp_path, frames):
    torch = pytest.importorskip('torch')
    from ensomi_n.v0.featcache import capture_song, evidence_from_logits, reproduce, write_npz
    from ensomi_n.v0.infer import song_outputs
    from ensomi_n.v0.model import Heads

    class Encoder(torch.nn.Module):
        memory = None

        def forward(self, x, mem, use_memory):
            # The first coordinate identifies position within a window, so seams are observable.
            h = torch.zeros((*x.shape[:2], 160), device=x.device)
            h[..., 0] = torch.arange(x.shape[1], device=x.device) / 2000
            h[..., 1] = x[..., 0]
            return h

    torch.manual_seed(3)
    model = SimpleNamespace(encoder=Encoder(), heads=Heads(160).eval())
    mel = np.zeros((frames, 128), dtype=np.float16)
    mel[:, 0] = np.arange(frames) / frames
    mp = tmp_path / 'mel.npy'
    np.save(mp, mel)
    song = SimpleNamespace(item='test', frames=frames, mel=mp, mean=0., std=1.)
    feat, logits, acts = capture_song(model, song, 'cpu')
    baseline = song_outputs(model, song, 'cpu')
    assert feat.shape == (frames, 160) and logits.shape == (frames, 23)
    assert np.array_equal(feat[:, 1], mel[:, 0])
    if frames == 4001:
        np.testing.assert_allclose(feat[[1499, 1500, 4000], 0], [1499 / 2000, 500 / 2000, 1996 / 2000])
    for k in baseline:
        np.testing.assert_array_equal(acts[k], baseline[k])
        np.testing.assert_allclose(evidence_from_logits(logits)[k], baseline[k], atol=2e-6)
    assert not model.heads.lin._forward_hooks
    saved = tmp_path / 'cache.npz'
    write_npz(saved, feat, logits)
    with np.load(saved, allow_pickle=False) as z:
        assert z['feat'].dtype == z['logits'].dtype == np.float16
        assert z['t0_ms'] == 0 and z['fps'] == 100
    np.savez_compressed(tmp_path / 'test.npz', **baseline)
    report = reproduce('test', logits, acts, saved, tmp_path)
    assert max(report['full_precision_max_abs'].values()) == 0
