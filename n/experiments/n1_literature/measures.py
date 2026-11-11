"""Diagnostic statistics; none is validated as a perceptual quality measure."""
from __future__ import annotations

import numpy as np

from ensomi_n.eval.measures import nearest, weighted_median

SEED = 20261111


def robust_sd(x, axis=None):
    x = np.asarray(x, dtype=float)
    centre = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - centre), axis=axis)


def interval(values, statistic='mean', draws=2000):
    """Percentile bootstrap of independent recording values, omitting nonfinite values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(value=None, lo=None, hi=None, n=0)
    fun = {'mean': np.mean, 'median': np.median, 'robust_sd': robust_sd,
           'sd': lambda a, axis=None: np.std(a, axis=axis, ddof=1)}[statistic]
    rng = np.random.default_rng(SEED)
    bs = fun(x[rng.integers(len(x), size=(draws, len(x)))], axis=1)
    return dict(value=float(fun(x)), lo=float(np.quantile(bs, .025)),
                hi=float(np.quantile(bs, .975)), n=len(x))


def recording_values(rows, field):
    per = {}
    for r in rows:
        v = r.get(field)
        if v is not None and np.isfinite(v):
            per.setdefault(r['rec'], []).append(v)
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}


def summarize(rows, fields):
    return {f: interval(list(recording_values(rows, f).values())) for f in fields}


def spacing_class(times):
    """Osu2MIR classes on distinct red-line times; exactly 5 s belongs to class 2."""
    t = np.unique(np.asarray(times, dtype=float))
    if len(t) == 0:
        return 0
    if len(t) == 1:
        return 1
    return 2 if np.all(np.diff(t) >= 5000.) else 3


def matching(ref, pred, tolerance):
    """Maximum-cardinality one-to-one matching of ordered event times within tolerance."""
    a, b = np.sort(ref), np.sort(pred)
    i = j = hits = 0
    while i < len(a) and j < len(b):
        if b[j] < a[i] - tolerance - 1e-9:
            j += 1
        elif a[i] < b[j] - tolerance - 1e-9:
            i += 1
        else:
            hits += 1
            i += 1
            j += 1
    return dict(hits=hits, n_ref=len(a), n_pred=len(b),
                recall=hits / len(a) if len(a) else np.nan,
                precision=hits / len(b) if len(b) else np.nan,
                f=2 * hits / (len(a) + len(b)) if len(a) + len(b) else np.nan)


def auc(positive, negative):
    """P(positive > negative) + half the tie probability; undefined for an empty class."""
    p, n = np.asarray(positive, float), np.sort(np.asarray(negative, float))
    if not len(p) or not len(n):
        return np.nan
    less = np.searchsorted(n, p, side='left')
    equal = np.searchsorted(n, p, side='right') - less
    return float(np.mean((less + .5 * equal) / len(n)))


def boundary_auc(logits, boundaries, positive, lo, hi, radius=3):
    """Within-song AUC with equally max-filtered event and background scores.

    Times are ms, logits at 100 fps; background centers are spaced 100 ms and
    exclude 500 ms around every boundary. `positive` may select one boundary kind.
    """
    from scipy.ndimage import maximum_filter1d
    z = np.asarray(logits, float)
    if radius:
        z = maximum_filter1d(z, size=2 * radius + 1, mode='nearest')
    centers = np.arange(0, len(z), 10)
    t = centers * 10.
    distance, _ = nearest(t, np.asarray(boundaries))
    good = (t >= lo) & (t <= hi) & (np.abs(distance) >= 500.)
    pos = np.round(np.asarray(positive) / 10.).astype(int)
    pos = pos[(pos >= 0) & (pos < len(z))]
    return dict(auc=auc(z[pos], z[centers[good]]), n_positive=len(pos), n_negative=int(good.sum()))


def spectral_flux(wave, sr=24000):
    """Three normalized log-magnitude flux bands at 1 kHz, on the input PCM clock."""
    if sr != 24000:
        raise ValueError('spectral_flux requires 24-kHz PCM')
    n_fft, hop = 480, 24
    x = np.pad(np.asarray(wave, np.float32), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    bands = [(freq >= a) & (freq < b) for a, b in [(60, 250), (250, 2000), (2000, 10000)]]
    out = np.zeros((len(frames), 3), np.float32)
    last = None
    for a in range(0, len(frames), 4096):
        m = np.log1p(np.abs(np.fft.rfft(frames[a:a + 4096] * window, axis=1)))
        prev = m[:1] if last is None else last
        d = np.maximum(np.diff(np.concatenate([prev, m]), axis=0), 0.)
        out[a:a + len(m)] = np.stack([d[:, band].sum(1) for band in bands], axis=1)
        last = m[-1:]
    for k in range(3):
        p = out[:, k][out[:, k] > 0]
        if len(p):
            out[:, k] /= max(float(np.quantile(p, .95)), 1e-12)
    return out


def flux_peaks(flux):
    from scipy.signal import find_peaks
    f = np.asarray(flux)
    p = f[f > 0]
    return find_peaks(f, height=np.quantile(p, .75), distance=50)[0].astype(float) if len(p) else np.array([])


def onset_offset(lines, weights, peaks, model_offset, lag):
    """Symmetric local peak association about the chart/model midpoint, without peak reuse."""
    lines, weights, peaks = map(np.asarray, (lines, weights, peaks))
    e, j = nearest(lines + model_offset / 2, peaks - lag)
    candidate = np.flatnonzero((weights > 0) & (np.abs(e) <= 100.) & (j >= 0))
    chosen, used = [], set()
    for i in candidate[np.argsort(np.abs(e[candidate]), kind='stable')]:
        if j[i] not in used:
            chosen.append(i)
            used.add(j[i])
    ix = np.asarray(chosen, dtype=int)
    o = weighted_median(peaks[j[ix]] - lag - lines[ix], weights[ix]) if len(ix) else np.nan
    coverage = float(weights[ix].sum() / weights.sum()) if weights.sum() else np.nan
    if np.isfinite(o):
        diff = abs(o - model_offset) - abs(o)
        verdict = 'tie' if abs(diff) <= 2 else ('model' if diff < 0 else 'chart')
    else:
        verdict = 'undefined'
    return dict(flux_offset_ms=float(o), matched_peaks=len(ix), weight_coverage=coverage, verdict=verdict)
