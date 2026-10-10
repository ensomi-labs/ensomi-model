"""Whole-song head rows from the figure generator, one canonical beat at a time.

``decode`` encodes the song's audio once, then per beat runs one GRU step, mixes the generator
and the copy heads (``net``), removes figures with a position outside [0, T - 2 ms] (R2 needs T at
least 2 ms after the last head), and samples a figure. The generator's logits are divided by
``temperature``; the copy heads are not. Heads sit exactly on the figure positions.

Requests are log(W_H / s) (``beatdata``): ``request`` for the song, ``sections`` a list of
(start_ms, end_ms, log W_H/s) applied to the beats starting in [start_ms, end_ms).
``star_request`` maps a star rating to a song request with the training record's map.
"""
from __future__ import annotations

import numpy as np
import torch

from ..r2.common import MIN_GAP_MS
from .audio import HOP_MS, MEL_BINS
from .figures import POSITIONS, head_times
from .lattice import beat_range
from .net import FigureNet


def star_request(record, star: float) -> float:
    m = record['star_map']
    return float(np.interp(star, m['star'], m['request']))


def song_inputs(grid, song_ms: float, record, request: float, sections=()):
    """Per-beat inputs of a whole song on ``grid``: b0, edges [nb+1], position times [nb, 24], extra [nb, E]."""
    b0, nb = beat_range(grid, song_ms)
    edges = grid.time_of_beat(b0 + np.arange(nb + 1))
    pos_ms = grid.time_of_beat(b0 + np.arange(nb)[:, None] + POSITIONS[None] / 48.0)
    beat_ms = np.clip(edges[:-1], 0.0, song_ms)
    std = lambda v: (v - record['label_mean']) / record['label_std']
    sec = np.zeros((nb, 2))
    for a, b, value in sections:
        sec[(beat_ms >= a) & (beat_ms < b)] = (std(value), 1.0)
    extra = np.column_stack((np.eye(4)[(b0 + np.arange(nb)) % 4], beat_ms / song_ms, grid.bpm(beat_ms) / 120.0,
                             np.full(nb, std(request)), sec)).astype(np.float32)
    return b0, edges, pos_ms, extra


@torch.no_grad()
def decode(model: FigureNet, record, mel, grid, song_ms: float, request: float, sections=(), seed: int = 0,
           temperature: float = 1.0, off=()):
    """(head times [K] ms, per-beat record) for a whole song; ``mel`` [F, >=128] standardised log-Mel.

    ``off``: 'copy' forces the gate to the generator (an ablation)."""
    cfg = model.cfg
    b0, edges, pos_ms, extra = song_inputs(grid, song_ms, record, request, sections)
    nb = len(extra)
    nf = len(mel)
    pos_frames = np.clip((pos_ms / HOP_MS).astype(np.int64), 0, nf - 1)
    bf = np.clip((edges / HOP_MS).astype(np.int64), 0, nf)
    beat_frames = np.stack((bf[:-1], np.maximum(bf[1:], bf[:-1] + 1)), 1).clip(0, nf)
    available = (pos_ms >= 0) & (pos_ms <= song_ms - MIN_GAP_MS)
    members = model.members.bool().numpy()
    allowed = torch.from_numpy(~(members[None] & ~available[:, None]).any(-1))          # [nb, V]
    t = lambda x: torch.from_numpy(np.asarray(x))[None]
    ex = t(extra)
    pos, A, q, k = model.audio(t(np.asarray(mel[:, :MEL_BINS], dtype=np.float32)), t(pos_frames), t(beat_frames),
                               ex, torch.ones(1, nb, dtype=torch.bool))
    pos, A, q, k, ex = pos[0], A[0], q[0], k[0], ex[0]
    V, lags = cfg.vocab, cfg.lags
    gen = torch.Generator().manual_seed(int(seed))
    ids = np.full(nb, -1, dtype=np.int64)
    gates = np.zeros((nb, 1 + cfg.heads), dtype=np.float32)
    h = None
    prev = V
    table_in, table_out = model.input_table(), model.output_table()
    for b in range(nb):
        x = torch.cat((A[b], table_in[prev], ex[b]))[None, None]
        out, h = model.gru(x, h)
        hb = out[0]
        logits = model.generator_logits(hb, pos[b:b + 1], table=table_out) / temperature
        j = np.arange(b - 1, b - lags - 1, -1)
        ok = j >= 0
        jj = np.where(ok, j, 0)
        back_ids = torch.from_numpy(np.where(ok, ids[jj], -1))
        eq = []
        for d in (1, 2):
            before = ids[b - d] if b - d >= 0 else -1
            bj = np.where(jj - d >= 0, ids[np.maximum(jj - d, 0)], -1)
            eq.append((ok & (bj == before) & (before >= 0)).astype(np.float32))
        scores = model.copy_scores(q[b:b + 1], k[torch.from_numpy(jj)][None], torch.from_numpy(np.stack(eq, -1))[None])
        g, logp, att = model.mixture(hb, logits, scores, torch.from_numpy(ok)[None])
        p_gen = logp[0].exp()
        if 'copy' in off:
            p = p_gen
        else:
            p = g[0, 0] * p_gen
            okt = torch.from_numpy(ok)
            for hd in range(cfg.heads):
                p_h = (att[0, hd, -1] * p_gen).index_add(0, back_ids[okt], att[0, hd, :-1][okt])
                p = p + g[0, 1 + hd] * p_h
        p = p * allowed[b]
        f = int(torch.multinomial(p / p.sum(), 1, generator=gen))
        ids[b] = f
        gates[b] = g[0].numpy()
        prev = f
    figures = np.asarray(record['vocab'], dtype=np.int64)[ids]
    heads = np.unique(head_times(grid, b0, figures))
    return heads, dict(b0=b0, nb=nb, ids=ids, figures=figures, edges=edges, gates=gates)
