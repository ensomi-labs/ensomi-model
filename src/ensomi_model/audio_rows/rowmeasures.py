"""Characterising measures of head rows on one song, for human and generated rows alike.

None is validated against the human's judgement; they characterise, and the human's screen decides.
``song_measures`` takes heads, the grid they are read on (a human chart's own grid, or the grid
the rows were generated on) and the song's standardised log-Mel:

- ``log_wh_s``: log(W_H / T), the song request scale (``beatdata``).
- ``chart_measures`` (``measures``): breathing (interior intensity-workload rho, rests in quiet
  and loud interior sections) and onset rank against a half-beat shift.
- Recurrence: per lag L in 4/8/16/32 beats, the correlation over beats b between the audio
  similarity of beats [b, b+2) and [b-L, b-L+2) (Pearson over the 128 Mel bins of the block means,
  each bin centred on the song) and the equality of the two blocks' figures (pairs where both
  blocks are empty excluded); ``recurrence_excess`` is its mean over lags minus the same with the
  audio similarity taken half the song away (shifted-audio null).
- Locked stretches: runs of 32 beats or more whose figure equals the figure 4 beats earlier,
  with at least one head; ``locked_share`` the share of beats in them, ``locked_changing_share`` of
  those runs whose mean lag-4 audio similarity is below the song's median.
- Drift: realised log(W_H / s) per 64-beat block minus ``request``; least-squares slope against
  the song position (0-1), and the mean of the last third minus that of the first third.
- Copy readiness: share of heads whose 9-entry rhythm key (8 inter-onset intervals in 1/48 beat
  and the beat phase, as in the R2 copy path) equals that of a head 4-32 beats earlier.

``pair_measures`` compares generated heads with one reference on the same grid: head F1 at 20 ms,
and on beats where both have the same nonzero head count, the share with the same figure.
"""
from __future__ import annotations

import numpy as np

from .audio import HOP_MS, MEL_BINS
from .data import beat_workload, section_label
from .evaluate import head_f1
from .figures import beat_figures
from .lattice import beat_range, snap
from .measures import chart_measures

LAGS = (4, 8, 16, 32)


def grid_figures(head_ms, grid, song_ms: float):
    """(b0, per-beat figures [nb], snap error ms [K]) of heads snapped to ``grid``."""
    t = snap(head_ms, grid, song_ms)
    return t['b0'], beat_figures(t['lattice'], t['mask']), t['err_ms']


def beat_descriptors(mel, edges_ms) -> np.ndarray:
    """[nb, 128] mean Mel per beat, each bin centred on the song's beats."""
    mel = np.asarray(mel[:, :MEL_BINS], dtype=np.float32)
    f = np.clip((np.asarray(edges_ms) / HOP_MS).astype(int), 0, len(mel))
    cs = np.concatenate((np.zeros((1, MEL_BINS), np.float32), np.cumsum(mel, 0)))
    n = np.maximum(f[1:] - f[:-1], 1)[:, None]
    d = (cs[np.maximum(f[1:], f[:-1] + 1).clip(max=len(mel))] - cs[f[:-1]]) / n
    return d - d.mean(0)


def block_similarity(desc, lag: int, shift: int = 0) -> np.ndarray:
    """[nb] Pearson similarity of 2-beat blocks b and b - lag (NaN where undefined); ``shift`` moves the audio."""
    nb = len(desc)
    blk = desc[:-1] + desc[1:]
    blk = blk - blk.mean(1, keepdims=True)
    blk = blk / (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
    out = np.full(nb, np.nan)
    b = np.arange(lag, nb - 1)
    bs, ls = (b + shift) % (nb - 1), (b - lag + shift) % (nb - 1)
    out[b] = (blk[bs] * blk[ls]).sum(1)
    return out


def _corr(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def recurrence(figures, desc) -> dict:
    nb = len(figures)
    out = {}
    for name, shift in (('aligned', 0), ('shifted', nb // 2)):
        rs = []
        for lag in LAGS:
            if nb < lag + 16:
                continue
            sim = block_similarity(desc, lag, shift)
            b = np.arange(lag, nb - 1)
            same = (figures[b] == figures[b - lag]) & (figures[b + 1] == figures[b - lag + 1])
            empty = (figures[b] == 0) & (figures[b + 1] == 0) & (figures[b - lag] == 0) & (figures[b - lag + 1] == 0)
            keep = ~empty & np.isfinite(sim[b])
            r = _corr(sim[b][keep], same[keep].astype(float))
            if r is not None:
                rs.append(r)
            if name == 'aligned':
                out[f'figure_repeat_lag{lag}'] = float(same[keep].mean()) if keep.any() else None
        out[f'recurrence_r_{name}'] = float(np.mean(rs)) if rs else None
    if out['recurrence_r_aligned'] is not None and out['recurrence_r_shifted'] is not None:
        out['recurrence_excess'] = out['recurrence_r_aligned'] - out['recurrence_r_shifted']
    else:
        out['recurrence_excess'] = None
    return out


def locked(figures, desc, min_run: int = 32) -> dict:
    nb = len(figures)
    rep = np.zeros(nb, dtype=bool)
    rep[4:] = figures[4:] == figures[:-4]
    sim = block_similarity(desc, 4)
    median = np.nanmedian(sim) if np.isfinite(sim).any() else np.nan
    in_run = changing = 0
    b = 0
    while b < nb:
        if not rep[b]:
            b += 1
            continue
        e = b
        while e < nb and rep[e]:
            e += 1
        if e - b >= min_run and (figures[b:e] != 0).any():
            in_run += e - b
            s = sim[b:e][np.isfinite(sim[b:e])]
            if len(s) and s.mean() < median:
                changing += e - b
        b = e
    return dict(locked_share=in_run / nb, locked_changing_share=changing / nb)


def drift(head_ms, song_ms: float, edges_ms, request: float, block: int = 64) -> dict:
    beat_ms = np.clip(edges_ms, 0.0, song_ms)
    nb = len(beat_ms) - 1
    if nb < 3 * block:
        return dict(drift_slope=None, drift_end_start=None)
    wh = beat_workload(head_ms, song_ms, beat_ms)
    starts = list(range(0, nb - block // 2, block))
    vals = np.array([section_label(wh, beat_ms, a, min(a + block, nb)) - request for a in starts])
    mid = np.array([(beat_ms[a] + beat_ms[min(a + block, nb)]) / 2 / song_ms for a in starts])
    slope = float(np.polyfit(mid, vals, 1)[0])
    third = len(vals) // 3
    return dict(drift_slope=slope, drift_end_start=float(vals[-third:].mean() - vals[:third].mean()))


def timing_keys(b, n):
    """[n, 9] int: inter-onset intervals k-4..k+3 in 1/48 beat (-1 where undefined) and the beat phase."""
    ioi = np.rint(48 * np.diff(b)).astype(np.int64)
    keys = np.full((n, 9), -1, dtype=np.int64)
    for col, off in enumerate(range(-4, 4)):
        i = np.arange(n) + off
        ok = (i >= 0) & (i < len(ioi))
        keys[ok, col] = ioi[i[ok]]
    keys[:, 8] = np.rint(48 * np.mod(b[:n], 1)).astype(np.int64) % 48
    return keys


def copy_readiness(head_beats) -> float | None:
    b = np.asarray(head_beats, dtype=np.float64)
    if len(b) < 10:
        return None
    keys = timing_keys(b, len(b))
    hit = 0
    lo = np.searchsorted(b, b - 32 - 1e-6, side='left')
    hi = np.searchsorted(b, b - 4 + 1e-6, side='right')
    for k in range(len(b)):
        if hi[k] > lo[k] and (keys[lo[k]:hi[k]] == keys[k]).all(1).any():
            hit += 1
    return hit / len(b)


def song_measures(head_ms, grid, song_ms: float, mel, request: float | None = None) -> dict:
    head_ms = np.asarray(head_ms, dtype=np.float64)
    b0, nb = beat_range(grid, song_ms)
    edges = grid.time_of_beat(b0 + np.arange(nb + 1))
    beat_ms = np.clip(edges, 0.0, song_ms)
    wh = beat_workload(head_ms, song_ms, beat_ms)
    out = dict(K=len(head_ms), beats=nb, log_wh_s=section_label(wh, beat_ms, 0, nb))
    cm = chart_measures(head_ms, song_ms, grid, mel)
    out.update({k: cm[k] for k in ('interior_intensity_workload_rho', 'interior_low_high_ratio', 'interior_low_rest',
                                   'interior_high_rest', 'onset_rank', 'shifted_onset_rank')})
    if cm['onset_rank'] is not None:
        out['onset_advantage'] = cm['onset_rank'] - cm['shifted_onset_rank']
    _, figures, _ = grid_figures(head_ms, grid, song_ms)
    desc = beat_descriptors(mel, edges)
    out.update(recurrence(figures, desc))
    out.update(locked(figures, desc))
    out.update(drift(head_ms, song_ms, edges, out['log_wh_s'] if request is None else request))
    out['copy_readiness'] = copy_readiness(grid.beat(head_ms))
    out['empty_beats'] = float((figures == 0).mean())
    return out


def pair_measures(gen_ms, ref_ms, grid, song_ms: float) -> dict:
    """Head F1 at 20 ms and figure agreement on equal-count beats, ``ref_ms`` snapped to ``grid``."""
    _, fg, _ = grid_figures(gen_ms, grid, song_ms)
    _, fr, _ = grid_figures(ref_ms, grid, song_ms)
    cg = np.array([bin(int(x)).count('1') for x in fg])
    cr = np.array([bin(int(x)).count('1') for x in fr])
    same = (cg == cr) & (cg > 0)
    return dict(f1_20=head_f1(gen_ms, ref_ms, 20)['f1'], equal_count_beats=int(same.sum()),
                figure_agreement=float((fg[same] == fr[same]).mean()) if same.any() else None)
