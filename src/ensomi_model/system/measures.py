"""Characterising measures of an arrangement on its rows, against the song's audio. None is validated against the
human's judgement; they characterise, and the human's screen decides.

``arrangement_measures(head_ms, song_ms, grid, actions, mel, target=None)``:

- ``J``: jack rate, rows sharing a lane with the previous row over all rows (texture split's J).
- ``R_tau``, ``S3short``, ``S3long``: share of row pairs tau beats apart (within .008 beat) with equal head masks,
  tau in 1, 2, 4 (short) and 8, 16, 32 (long) (texture split's R).
- ``D``: adjacent-row dependence, mutual information of consecutive head masks in 128-row windows minus a
  permutation null, bits (texture split's D).
- ``audio_recurrence``: whether the arrangement repeats where the audio repeats (2026-10-09 audio study,
  ``structure.recurrence``): over beat pairs 4-32 beats apart, the partial correlation of arrangement similarity
  (lanes on 24 ticks per beat) and audio similarity (song-centred unit log-Mel frames averaged at the integer beat
  ±50 ms), controlled for the lag and the rows' own similarity, minus the mean over 40 circular audio shifts.
- ``jack_sound``: whether jacks land on repeated sound (``splus_jacks``, re-implemented on the Listener's
  128-bin log-Mel): over rows with a gap of at most 600 ms, the partial correlation of the jack indicator with
  band-energy sameness of the two heads (minus the RMS difference of three band means), controlled for 8 local
  rows features, minus the mean over 40 circular shifts of the cue.
- ``nh``, ``held``, ``busy``: heads per row, held share, all-busy share; with ``target`` also ``track_rmse``, the
  RMS error of the 64-row running (held, heads, busy) against the kappa targets, per component.
"""
from __future__ import annotations

import numpy as np

from ..r2.theta import decode

TAUS = (1, 2, 4, 8, 16, 32)
BIT = 1 << np.arange(4)
COUNT = np.array([int(m).bit_count() for m in range(16)])
BANDS = ((0, 24), (24, 80), (80, 128))


def _residual(Y, X):
    A = np.column_stack((np.ones(len(X)), X))
    coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    return Y - A @ coef


def _corr(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def jack_rate(masks) -> float:
    m = np.asarray(masks)
    return float(((m[1:] & m[:-1]) != 0).sum() / len(m)) if len(m) else np.nan


def lag_pairs(b, tau):
    lo = np.searchsorted(b, b + tau - .008, side='left')
    hi = np.searchsorted(b, b + tau + .008, side='right')
    cnt = hi - lo
    ii = np.repeat(np.arange(len(b)), cnt)
    jj = np.concatenate([np.arange(a, z) for a, z in zip(lo, hi)]) if cnt.sum() else np.zeros(0, int)
    return ii, jj


def repeats(b, masks) -> dict:
    out = {}
    for tau in TAUS:
        i, j = lag_pairs(b, tau)
        out[f'R_{tau}'] = float((masks[i] == masks[j]).mean()) if len(i) else np.nan
    out['S3short'] = float(np.nanmean([out[f'R_{t}'] for t in TAUS[:3]]))
    out['S3long'] = float(np.nanmean([out[f'R_{t}'] for t in TAUS[3:]]))
    return out


def _entropy(counts):
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / total
    return -(p * np.log2(np.where(p > 0, p, 1))).sum(axis=-1)


def _mutual(prev, currents):
    rows = len(currents)
    ids = prev[None] * 16 + currents + 256 * np.arange(rows)[:, None]
    table = np.bincount(ids.ravel(), minlength=256 * rows).reshape(rows, 16, 16)
    return _entropy(table.sum(1)) + _entropy(table.sum(2)) - _entropy(table.reshape(rows, 256))


def dependence(masks, seed=20261007) -> float:
    vals = []
    for start in range(0, len(masks) - 127, 64):
        x = masks[start:start + 127]
        y = masks[start + 1:start + 128]
        rng = np.random.default_rng(seed ^ start)
        perms = np.stack([y[rng.permutation(127)] for _ in range(20)])
        mi = _mutual(x, np.vstack([y, perms]))
        vals.append(mi[0] - mi[1:].mean())
    return float(np.mean(vals)) if vals else np.nan


def beat_mel(mel, grid, b0, b1) -> np.ndarray:
    """[b1 - b0, 128] unit audio vectors at integer beats: song-centred unit frames averaged over the beat ±50 ms."""
    m = np.asarray(mel[:, :128], dtype=np.float32)
    m = m - m.mean(0)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-6)
    t = grid.time_of_beat(np.arange(b0, b1, dtype=np.float64))
    a = np.clip(np.floor((t - 50) / 10).astype(int), 0, len(m) - 1)
    z = np.clip(np.ceil((t + 50) / 10).astype(int), a + 1, len(m))
    cs = np.concatenate((np.zeros((1, 128), np.float32), np.cumsum(m, 0)))
    return (cs[z] - cs[a]) / (z - a)[:, None]


def _cos_rows(x, y):
    return np.sum(x * y, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9)


def audio_recurrence(b, heads, bm, b0, b1, shifts=40, seed=442) -> dict:
    """Arrangement-audio recurrence over beats [b0, b1) (``structure.recurrence`` of the audio study)."""
    nb = b1 - b0
    arr = np.zeros((nb, 24, 4), bool)
    sk = np.zeros((nb, 24), bool)
    ib = np.floor(b + 1e-9).astype(int)
    tick = np.clip(np.rint(24 * (b - ib)).astype(int), 0, 23)
    good = (ib >= b0) & (ib < b1)
    for i, t, h in zip(ib[good] - b0, tick[good], heads[good]):
        arr[i, t] |= h
        sk[i, t] = True
    arr = arr.reshape(nb, -1).astype(float)
    sk = sk.astype(float)
    parts = []
    for lag in range(4, 33):
        if nb <= lag:
            continue
        v = np.flatnonzero((sk[:-lag].sum(1) > 0) & (sk[lag:].sum(1) > 0))
        if not len(v):
            continue
        parts.append((np.full(len(v), lag), _cos_rows(arr[v], arr[v + lag]), _cos_rows(bm[v], bm[v + lag]),
                      _cos_rows(sk[v], sk[v + lag]), v))
    if not parts:
        return dict(audio_recurrence=np.nan, audio_recurrence_r=np.nan, audio_recurrence_null=np.nan)
    lag, a_cos, au_cos, sk_cos, i = (np.concatenate(x) for x in zip(*parts))
    X = np.column_stack((np.eye(29)[lag - 4], sk_cos))
    R = _residual(np.column_stack((a_cos, au_cos)), X)
    point = _corr(R[:, 0], R[:, 1])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(shifts):
        p = np.roll(bm, int(rng.integers(1, len(bm))), axis=0)
        null.append(_cos_rows(p[i], p[i + lag]))
    EN = _residual(np.column_stack([a_cos] + null), X)
    null_r = np.nanmean([_corr(EN[:, 0], EN[:, j]) for j in range(1, EN.shape[1])])
    return dict(audio_recurrence=point - null_r, audio_recurrence_r=point, audio_recurrence_null=float(null_r))


def local_rows(t, grid, song_ms) -> np.ndarray:
    """[K, 8] local rows features of the audio study (S columns 0-5, 10, 11)."""
    K = len(t)
    B = grid.beat(t)
    beat_len = grid.lengths[grid.segment(t)]
    nxt = np.append(t[1:], song_ms)
    prv = np.concatenate([[t[0] - 10000.0], t[:-1]])
    gn, gp = np.maximum(nxt - t, 1.0), np.maximum(t - prv, 1.0)
    bn = np.append(np.diff(B), (song_ms - t[-1]) / beat_len[-1])
    bp = np.concatenate([[8.0], np.diff(B)])
    ph = B - np.floor(B)
    den1 = np.searchsorted(t, t + 1000) - np.searchsorted(t, t - 1000)
    den4 = np.searchsorted(t, t + 4000) - np.searchsorted(t, t - 4000)
    return np.stack([np.log(gn), np.log(gp), np.log2(np.maximum(bn, 1e-3)), np.log2(np.maximum(bp, 1e-3)),
                     np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), den1, den4], 1)


def band_energies(mel, t) -> np.ndarray:
    """[K, 3] mean log-Mel of three bin ranges at each head's frame."""
    m = np.asarray(mel[:, :128], dtype=np.float32)
    f = np.clip(np.rint(np.asarray(t) / 10).astype(int), 0, len(m) - 1)
    return np.stack([m[f, a:z].mean(1) for a, z in BANDS], 1)


def jack_sound(t, heads, grid, song_ms, mel, shifts=40, seed=455) -> dict:
    jj = np.arange(1, len(t))
    jj = jj[(t[jj] - t[jj - 1]) <= 600]
    if len(jj) < 16:
        return dict(jack_sound=np.nan, jack_sound_r=np.nan)
    e = band_energies(mel, t)
    cue = -np.sqrt(np.mean((e[jj] - e[jj - 1]) ** 2, axis=1))
    y = (heads[jj] & heads[jj - 1]).any(1).astype(float)
    X = local_rows(t, grid, song_ms)[jj]
    rng = np.random.default_rng(seed)
    cols = [y, cue] + [np.roll(cue, int(s)) for s in rng.integers(1, len(jj), size=shifts)]
    R = _residual(np.column_stack(cols), X)
    point = _corr(R[:, 0], R[:, 1])
    null = np.nanmean([_corr(R[:, 0], R[:, k]) for k in range(2, R.shape[1])])
    return dict(jack_sound=point - null, jack_sound_r=point)


def tracking(actions, K, target) -> dict:
    held, heads, _ = decode(actions[:K])
    busy = (held | heads).all(1)
    cum = [np.concatenate(([0.0], np.cumsum(x))) for x in (held.sum(1), heads.sum(1), busy)]
    k = np.arange(1, K)
    start = np.maximum(0, k - 64)
    n = k - start
    running = np.stack([(cum[0][k] - cum[0][start]) / (4 * n), (cum[1][k] - cum[1][start]) / n,
                        (cum[2][k] - cum[2][start]) / n], 1)
    rmse = np.sqrt(np.mean((running - np.asarray(target)) ** 2, axis=0))
    return dict(track_rmse_held=float(rmse[0]), track_rmse_heads=float(rmse[1]), track_rmse_busy=float(rmse[2]))


def arrangement_measures(head_ms, song_ms, grid, actions, mel, target=None) -> dict:
    t = np.asarray(head_ms, dtype=np.float64)
    K = len(t)
    held, heads, ln = decode(actions[:K])
    masks = heads.astype(np.int64) @ BIT
    b = grid.beat(t)
    b0, b1 = int(np.ceil(b[0] - 1e-9)), int(np.floor(b[-1] + 1e-9))
    out = dict(K=K, nh=float(heads.sum(1).mean()), held=float(held.mean()), busy=float((held | heads).all(1).mean()),
               ln_heads=int(ln.sum()), J=jack_rate(masks), D=dependence(masks), **repeats(b, masks))
    out.update(audio_recurrence(b, heads, beat_mel(mel, grid, b0, b1), b0, b1) if b1 - b0 > 40 else
               dict(audio_recurrence=np.nan))
    out.update(jack_sound(t, heads, grid, song_ms, mel))
    if target is not None:
        out.update(tracking(actions, K, target))
    return out
