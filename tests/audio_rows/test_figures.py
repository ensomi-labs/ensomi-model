import numpy as np
import torch

from ensomi_model.audio_rows import figures as FG
from ensomi_model.audio_rows.beatdata import EXTRA_DIM
from ensomi_model.audio_rows.lattice import snap
from ensomi_model.audio_rows.net import FigureNet, NetConfig, nll
from ensomi_model.audio_rows.rowmeasures import copy_readiness, locked
from ensomi_model.audio_rows.rowsgen import decode
from ensomi_model.r2.common import GridArrays, grid_from_arrays


def grid(offset_ms=100.0, beat_ms=500.0):
    return GridArrays.from_grid(grid_from_arrays(np.array([[offset_ms, beat_ms, 4]]), np.array([[offset_ms, 4]])))


def test_figures_share_positions_across_lattices_and_round_trip_through_the_grid():
    g = grid()
    beats = np.array([0.0, 0.5, 1.0, 1 + 1 / 3, 1 + 2 / 3, 2.25, 2.75])
    heads = g.time_of_beat(beats)
    t = snap(heads, g, 5_000.0)
    figs = FG.beat_figures(t['lattice'], t['mask'])
    b = -t['b0']
    assert figs[b:b + 3].tolist() == [1 | 1 << 12, 1 | 1 << 8 | 1 << 16, 1 << 6 | 1 << 18]   # 0, 1/2; triplets; 1/4, 3/4
    assert np.allclose(FG.head_times(g, t['b0'], figs), heads, atol=1e-3)


def test_vocab_puts_the_empty_figure_first_and_drops_rare_figures():
    v = FG.Vocab.fit([5, 5, 5, 0, 7, 7, 7, 7, 9], min_count=3)
    assert v.figures.tolist() == [0, 7, 5]
    assert v.encode([7, 9, 0]).tolist() == [1, -1, 0]


def tiny_net(V=6):
    members = FG.members(np.array([0, 1, 1 | (1 << 12), (1 << 6) | (1 << 18), 1 | (1 << 6) | (1 << 12) | (1 << 18), 1 << 12]))
    torch.manual_seed(0)
    return FigureNet(NetConfig(vocab=V, enc=8, ctx=16, ctx_heads=2, window=8, hidden=16, fig=8, heads=2, lags=8, key=4),
                     members[:V])


def batch(ids):
    L = len(ids)
    frames = 40 * L + 64
    rng = np.random.default_rng(0)
    pos = np.clip(np.arange(L)[:, None] * 40 + np.arange(24)[None] * 2, 0, frames - 1)
    return dict(mel=torch.from_numpy(rng.normal(size=(1, frames, 128)).astype(np.float32)),
                pos_frames=torch.from_numpy(pos)[None], beat_frames=torch.from_numpy(np.stack((pos[:, 0], pos[:, 0] + 40), 1))[None],
                extra=torch.zeros(1, L, EXTRA_DIM), ids=torch.tensor([ids]), valid=torch.ones(1, L, dtype=torch.bool))


def test_teacher_forcing_learns_a_repeating_figure_through_the_copy_heads():
    model = tiny_net()
    b = batch([1, 2, 3, 4] * 2 + [-1] + [1, 2, 3, 4])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        loss, m = nll(model, b, drop=0.0)
        opt.zero_grad()
        (loss.sum() / m.sum()).backward()
        opt.step()
        losses.append(loss.sum().item() / m.sum().item())
    assert m.sum() == 12 and losses[-1] < losses[0]
    with torch.no_grad():
        on, _ = nll(model, b, drop=0.0)
        off, _ = nll(model, b, drop=0.0, off=('copy',))
    assert torch.isfinite(on).all() and not torch.allclose(on, off)


def test_decode_emits_heads_inside_the_song_on_vocabulary_figures():
    model = tiny_net().eval()
    record = dict(vocab=[0, 1, 1 | (1 << 12), (1 << 6) | (1 << 18), 1 | (1 << 6) | (1 << 12) | (1 << 18), 1 << 12],
                  label_mean=2.0, label_std=1.0)
    g = grid()
    song_ms = 6_210.0
    mel = np.random.default_rng(1).normal(size=(621, 128)).astype(np.float16)
    heads, rec = decode(model, record, mel, g, song_ms, 2.5, sections=[(1000.0, 3000.0, 3.0)], seed=3)
    assert np.all(np.diff(heads) > 0) and heads.min() >= 0 and heads.max() <= song_ms - 2.0
    assert len(heads) == FG.counts(rec['figures']).sum()


def test_copy_readiness_and_locked_stretches_on_a_repeated_bar():
    beats = np.concatenate([b * 4 + np.array([0.0, 1.0, 1.5, 3.0]) for b in range(40)])
    assert copy_readiness(beats) > 0.8
    figs = np.tile([1, 0, 4097, 0], 40)
    desc = np.random.default_rng(0).normal(size=(160, 128))
    assert locked(figs, desc)['locked_share'] > 0.9
