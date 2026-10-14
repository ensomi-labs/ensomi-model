import numpy as np
import torch

from ensomi_model.audio_rows import figures as FG
from ensomi_model.audio_rows.beatdata import EXTRA_DIM
from ensomi_model.audio_rows.lattice import beat_range, snap
from ensomi_model.audio_rows.net import FigureNet, NetConfig, nll
from ensomi_model.audio_rows.rowmeasures import copy_readiness, locked
from ensomi_model.audio_rows.audio import Song
from ensomi_model.audio_rows.rowsgen import Rows, available_positions, decode, quiet_rank
from ensomi_model.r2.common import MIN_GAP_MS, GridArrays, grid_from_arrays


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


RECORD = dict(vocab=[0, 1, 1 | (1 << 12), (1 << 6) | (1 << 18), 1 | (1 << 6) | (1 << 12) | (1 << 18), 1 << 12],
              label_mean=2.0, label_std=1.0, star_map=dict(star=[2.0, 6.0], request=[1.5, 3.5]))


def test_section_tilt_pulls_a_section_toward_its_request_and_stays_inside_it():
    model = tiny_net().eval()
    g = grid()
    song_ms = 20_210.0
    mel = np.random.default_rng(1).normal(size=(2021, 128)).astype(np.float32)
    sec = [(5000.0, 15000.0, 5.0)]
    n_in = {}
    for gain in (0.0, 3.0):
        n_in[gain] = 0
        for seed in range(3):
            heads, rec = decode(model, RECORD, mel, g, song_ms, 1.0, sections=sec, seed=seed, section_gain=gain)
            n_in[gain] += int(((heads >= 5000.0) & (heads < 15000.0)).sum())
            start = np.clip(rec['edges'][:-1], 0, song_ms)
            inside = (start >= 5000.0) & (start < 15000.0)
            assert np.all(rec['tilts'][~inside] == 0)
            assert gain == 0.0 or rec['tilts'][inside][1:].max() > 0
    assert n_in[3.0] > n_in[0.0]


def test_quiet_rests_keep_the_current_rule_at_zero_and_the_trained_heads_on_every_beat():
    model = tiny_net().eval()
    g = grid()
    song_ms = 20_210.0
    mel = np.random.default_rng(1).normal(size=(2021, 128)).astype(np.float32)
    mel[:1000] += 1.0                                                   # a loud first half
    beat_frames = np.stack((np.arange(40) * 50, np.arange(40) * 50 + 50), 1)
    rank = quiet_rank(mel, beat_frames)
    assert rank[:15].min() > rank[25:].max() and rank.min() == 0 and rank.max() == 1
    run = lambda seed, **kw: decode(model, RECORD, mel, g, song_ms, 2.5, seed=seed, **kw)[0]
    differs = 0
    for seed in range(4):
        current, trained = run(seed, quiet_rests=0.0), run(seed, copy_rests=True)
        differs += len(current) != len(trained) or not np.array_equal(current, trained)
        assert np.array_equal(run(seed, quiet_rests=1.1), trained)
        heads = run(seed, quiet_rests=0.4)
        assert np.array_equal(run(seed), heads)                         # the default is .40
        assert np.all(np.diff(heads) > 0) and heads.max() <= song_ms - 2.0
    assert differs


def two_segments(beats):
    """Beats of 500 ms from 100 ms, then of 430 ms from ``beats`` beats later; the two segments' lines differ."""
    seg = np.array([[100.0, 500.0, 4], [100.0 + 500.0 * beats, 430.0, 4]])
    return GridArrays.from_grid(grid_from_arrays(seg, seg[:, [0, 2]]))


def test_whole_beats_start_every_segment_on_a_whole_beat_and_leave_one_segment_alone():
    for beats, whole in ((8.3, 8.0), (8.5, 9.0), (8.7, 9.0)):
        g = two_segments(beats).whole_beats()
        assert g.origins.tolist() == [0.0, whole] and g.time_of_beat(whole) == g.offsets[1]
    one = grid()
    for key in ('offsets', 'lengths', 'origins', 'meters'):
        assert np.array_equal(getattr(one.whole_beats(), key), getattr(one, key))


def test_a_beat_cut_short_keeps_its_positions_a_sixteenth_of_a_beat_before_the_next_segment():
    song_ms = 12_000.0
    for g in (two_segments(8.95).whole_beats(), grid()):
        b0, nb = beat_range(g, song_ms)
        edges = g.time_of_beat(b0 + np.arange(nb + 1))
        pos = g.time_of_beat(b0 + np.arange(nb)[:, None] + FG.POSITIONS[None] / 48.0)
        ok = available_positions(g, edges, pos, song_ms)
        inside = (pos >= 0) & (pos <= song_ms - MIN_GAP_MS)
        if len(g.offsets) == 1:
            assert np.array_equal(ok, inside)
            continue
        cut = 8 - b0                    # the boundary cuts beat 8 at 475 ms; its 11/12 and 15/16 lie 17 and 6 ms before
        assert edges[cut + 1] == g.offsets[1]
        assert ok[cut].tolist() == (FG.POSITIONS <= 42).tolist()
        assert np.array_equal(np.delete(ok, cut, 0), np.delete(inside, cut, 0))


def test_decode_puts_every_head_on_its_own_segments_lines_and_before_the_next_segment():
    vocab = [0, 1, 1 << 18, 1 | (1 << 6) | (1 << 12) | (1 << 18), 1 << 23]          # {}, 0, 3/4, quarters, 15/16
    torch.manual_seed(0)
    model = FigureNet(NetConfig(vocab=5, enc=8, ctx=16, ctx_heads=2, window=8, hidden=16, fig=8, heads=2, lags=8,
                                key=4), FG.members(np.array(vocab))).eval()
    record = dict(RECORD, vocab=vocab)
    song_ms = 12_000.0
    mel = np.random.default_rng(1).normal(size=(1200, 128)).astype(np.float32)
    for beats in (8.3, 8.7, 8.95):  # 0.3 beat left before the boundary gets no positions; 0.7 and 0.95 are cut short
        g = two_segments(beats)
        boundary = g.offsets[1]
        for seed in range(4):
            heads, rec = decode(model, record, mel, g, song_ms, 3.0, seed=seed)
            s = g.segment(heads)
            lines = (heads - g.offsets[s]) / g.lengths[s] * 48
            assert np.abs((lines - np.round(lines)) * g.lengths[s] / 48).max() < 1e-3
            assert np.isin(np.round(lines) % 48, FG.POSITIONS).all() and (s == 1).any()
            assert boundary in rec['edges']
            for k in np.flatnonzero(rec['figures']):
                start = rec['edges'][k]
                times = start + FG.POSITIONS[FG.members(rec['figures'][k:k + 1])[0]] / 48 * g.lengths[g.segment(start)]
                assert times.max() <= rec['edges'][k + 1] - g.lengths[g.segment(start)] / 16 + 1e-6
            assert np.diff(heads).min() >= g.lengths.min() / 16 - 1e-3   # head times are rounded to 0.001 ms
            if beats < 8.5:
                assert not ((heads >= 100.0 + 500.0 * 8) & (heads < boundary)).any()


def test_decode_keeps_the_continuous_beats_on_one_segment():
    model = tiny_net().eval()
    g = grid()
    mel = np.random.default_rng(1).normal(size=(621, 128)).astype(np.float32)
    heads, rec = decode(model, RECORD, mel, g, 6_210.0, 2.5, seed=3)
    assert np.array_equal(rec['edges'], g.time_of_beat(rec['b0'] + np.arange(rec['nb'] + 1)))
    assert np.array_equal(heads, np.unique(FG.head_times(g, rec['b0'], rec['figures'])))


def test_rows_gives_heads_on_the_listener_grid_for_a_star_request():
    class Listener:
        def listen(self, audio):
            mel = np.random.default_rng(2).normal(size=(1200, 130)).astype(np.float16)
            return Song(mel, np.array([[130.0, 500.0]]), 12_000.0, dict(listen=0.1))

    rows = Rows(tiny_net().eval(), RECORD, Listener())
    prepared = rows.prepare('song.ogg')
    r = rows.heads(prepared, star=3.0, sections=[dict(start_ms=4000.0, end_ms=8000.0, star=5.0)], seed=1)
    assert r['K'] == len(r['head_ms']) > 0 and r['request'] == 2.0
    assert np.all(np.diff(r['head_ms']) > 0) and r['head_ms'][-1] <= r['song_ms'] - 2.0
    assert r['segments'].shape == (1, 3) and r['grid'].bpm(np.array([1000.0]))[0] == 120.0


def test_copy_readiness_and_locked_stretches_on_a_repeated_bar():
    beats = np.concatenate([b * 4 + np.array([0.0, 1.0, 1.5, 3.0]) for b in range(40)])
    assert copy_readiness(beats) > 0.8
    figs = np.tile([1, 0, 4097, 0], 40)
    desc = np.random.default_rng(0).normal(size=(160, 128))
    assert locked(figs, desc)['locked_share'] > 0.9
