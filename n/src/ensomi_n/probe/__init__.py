"""Decoder-free probe measures of rung 1 (design §5.1 R1): M1, M1-bar, M2 and M3.

An *activation* is a function of time sampled on a regular grid: ``Activation(values, t0_ms, step_ms)``;
it is read between samples by linear interpolation and is 0 outside its span. Impulse trains (R2's
beat lines, the identity fixture) are `Activation.impulses`.

- **M1, window phase recovery at 20 ms.** The reference canonical lines are cut into windows of at most
  8 lines that never cross a segment start of the reference's musical segments (redundant lines merged);
  a run of n lines gives ceil(n / 8) near-equal windows, so a short section is one window; windows of
  fewer than 2 lines are dropped. For each window, the shift s in (-L/2, L/2] (L the window's median
  canonical beat, 1 ms grid) maximising the summed activation at the lines plus s is found; the window
  passes when |s| <= 20 ms. A window whose best sum is 0 fails. Windows weigh by the heads on their lines
  (`Reference.weights`). An item's M1 is the weighted share of passing windows. Chance is 40 ms / L.
- **M1-bar.** The same over notated bar starts: windows of 4 bar starts within a run of one notated beat
  length (same-tempo resets inside a run allowed), shifts in (-B/2, B/2] with B the median bar length.
- **M2, local tempo.** Per M1 window, the dominant period of the activation (autocorrelation over a span of
  at least 6 s centred on the window, lags 0.2-1.5 s) or a given period track (A2), against the window's
  canonical beat: within 2 %, octave (x2, x1/2), 3:2 family (x3/2, x2/3, x3, x1/3), other.
- **M3.** Peak F-measure at 20 ms against the canonical lines in the scored range, one-to-one matching.

`aggregate` gives per-stratum means over items with a recording-cluster bootstrap. Everything here is
numpy; the fixtures in ``tests/test_probe.py`` check chance (40 ms / L), identity (1.0) and a 25 ms shift (0).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..data.strata import musical_segments

TOL_MS = 20.0
WINDOW_LINES = 8
BAR_WINDOW = 4
SHIFT_STEP_MS = 1.0


@dataclass
class Activation:
    values: np.ndarray
    t0_ms: float
    step_ms: float

    @classmethod
    def impulses(cls, times_ms, step_ms: float = 10.0, t0_ms: float = 0.0, end_ms: float | None = None) -> 'Impulses':
        """Unit triangles of half-width ``step_ms`` centred exactly at ``times_ms`` (peaks are not snapped)."""
        return Impulses(np.sort(np.asarray(times_ms, dtype=np.float64)), step_ms, t0_ms, end_ms)

    def at(self, times_ms) -> np.ndarray:
        pos = (np.asarray(times_ms, dtype=np.float64) - self.t0_ms) / self.step_ms
        n = len(self.values)
        i = np.floor(pos).astype(np.int64)
        f = pos - i
        ok = (i >= 0) & (i + 1 < n)
        out = np.zeros(pos.shape)
        ii = np.where(ok, i, 0)
        out[ok] = (self.values[ii] * (1 - f) + self.values[np.minimum(ii + 1, n - 1)] * f)[ok]
        return out


class Impulses:
    """An impulse train read as unit triangles of half-width ``width_ms`` around each time."""

    def __init__(self, times: np.ndarray, width_ms: float = 10.0, t0_ms: float = 0.0, end_ms: float | None = None):
        self.times, self.width, self.step_ms, self.t0_ms = times, width_ms, width_ms, t0_ms
        end = end_ms if end_ms is not None else (float(times.max()) + 1000.0 if len(times) else 1000.0)
        self._end = end

    @property
    def values(self) -> np.ndarray:
        return self.at(self.t0_ms + self.step_ms * np.arange(int((self._end - self.t0_ms) / self.step_ms) + 1))

    def at(self, times_ms) -> np.ndarray:
        t = np.asarray(times_ms, dtype=np.float64)
        out = np.zeros(t.shape)
        if not len(self.times):
            return out
        j = np.searchsorted(self.times, t)
        for c in (j - 1, j):
            ok = (c >= 0) & (c < len(self.times))
            d = np.abs(t - self.times[np.clip(c, 0, len(self.times) - 1)])
            out += np.where(ok, np.maximum(0.0, 1.0 - d / self.width), 0.0)
        return out


def beat_windows(ref) -> list[dict]:
    """M1 windows of a `Reference`: dicts with line indices, times, beat length and head weight."""
    lines = ref.lines()
    w = ref.weights(lines)
    t = lines.t
    if len(t) < 2:
        return []
    segs = [s for s in musical_segments(ref.segments or ()) if s['plausible']] if ref.segments else []
    bounds = np.array(sorted(s['start'] for s in segs)[1:]) if segs else np.zeros(0)
    run_id = np.searchsorted(bounds, t + 2.0, side='right') if len(bounds) else np.zeros(len(t), dtype=np.int64)
    out = []
    for r in np.unique(run_id):
        idx = np.flatnonzero(run_id == r)
        if len(idx) < 2:
            continue
        for part in np.array_split(idx, int(math.ceil(len(idx) / WINDOW_LINES))):
            if len(part) < 2:
                continue
            out.append(dict(idx=part, t=t[part], beat=float(np.median(lines.beat_ms[part])), weight=float(w[part].sum()),
                            run=int(r), run_len=len(idx)))
    return out


def bar_windows(ref) -> list[dict]:
    """M1-bar windows: 4 consecutive notated bar starts in a run of one notated beat length, within range."""
    tm = ref.timing
    if not tm.has_bars or not len(tm.beats):
        return []
    bi = np.flatnonzero(tm.downbeat)
    bt = tm.beats[bi]
    keep = (bt >= ref.lo_ms - 1) & (bt <= ref.hi_ms + 1)
    bi, bt = bi[keep], bt[keep]
    if len(bt) < 2:
        return []
    step = tm.step_ms[bi]
    lines, w = ref.lines(), ref.weights()
    run = np.concatenate([[0], np.cumsum(np.abs(step[1:] / step[:-1] - 1.0) > 1e-6)])
    out = []
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        for part in np.array_split(idx, max(int(math.ceil(len(idx) / BAR_WINDOW)), 1)):
            if len(part) < 2:
                continue
            times = bt[part]
            span = (lines.t >= times[0] - 1) & (lines.t < times[-1] + np.median(np.diff(times)))
            out.append(dict(t=times, beat=float(np.median(np.diff(times))), weight=float(w[span].sum())))
    return out


def recover(act: Activation, windows: list[dict], tol_ms: float = TOL_MS) -> tuple[np.ndarray, np.ndarray]:
    """(passed [W] bool, best shift [W] ms) for each window."""
    passed = np.zeros(len(windows), dtype=bool)
    best = np.full(len(windows), np.nan)
    for k, win in enumerate(windows):
        half = win['beat'] / 2.0
        shifts = np.arange(-half + SHIFT_STEP_MS, half + 1e-9, SHIFT_STEP_MS)
        s = act.at(win['t'][None, :] + shifts[:, None]).sum(axis=1)
        top = s.max()
        if top <= 0:
            continue
        ties = np.flatnonzero(s >= top - 1e-9 * max(abs(top), 1.0))
        shift = float(shifts[ties[len(ties) // 2]])     # the middle of a plateau, not its first point
        best[k] = shift
        passed[k] = abs(shift) <= tol_ms
    return passed, best


def weighted_share(passed: np.ndarray, windows: list[dict]) -> float:
    w = np.array([x['weight'] for x in windows], dtype=np.float64)
    if not len(w) or w.sum() <= 0:
        return float('nan')
    return float((passed * w).sum() / w.sum())


def chance(windows: list[dict], tol_ms: float = TOL_MS) -> float:
    w = np.array([x['weight'] for x in windows], dtype=np.float64)
    c = np.array([min(2 * tol_ms / x['beat'], 1.0) for x in windows])
    return float((c * w).sum() / w.sum()) if len(w) and w.sum() > 0 else float('nan')


def m1(act: Activation, windows: list[dict]) -> float:
    return weighted_share(recover(act, windows)[0], windows)


def _tempo_class(ratio: float) -> str:
    near = lambda q: abs(ratio / q - 1.0) <= 0.02
    if near(1.0):
        return 'tempo'
    if near(2.0) or near(0.5):
        return 'octave'
    if any(near(q) for q in (1.5, 2 / 3, 3.0, 1 / 3)):
        return '3:2'
    return 'other'


def dominant_period(act: Activation, centre_ms: float, span_ms: float = 6000.0, lo_ms: float = 200.0,
                    hi_ms: float = 1500.0) -> float:
    step = act.step_ms
    t = np.arange(centre_ms - span_ms / 2, centre_ms + span_ms / 2, step)
    x = act.at(t)
    x = x - x.mean()
    if not np.any(x):
        return float('nan')
    lags = np.arange(int(lo_ms / step), int(hi_ms / step) + 1)
    ac = np.array([np.dot(x[:-l], x[l:]) for l in lags])
    j = int(np.argmax(ac))
    frac = 0.0
    if 0 < j < len(ac) - 1:
        y0, y1, y2 = ac[j - 1], ac[j], ac[j + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return float((lags[j] + frac) * step)


def m2(act: Activation | None, windows: list[dict], period: Activation | None = None) -> dict:
    """Weighted shares of tempo classes over windows (``period`` is a period track in ms, for A2)."""
    counts = dict(tempo=0.0, octave=0.0, **{'3:2': 0.0}, other=0.0)
    total = 0.0
    for win in windows:
        c = float(np.mean(win['t']))
        if period is not None:
            vals = period.at(np.arange(win['t'][0], win['t'][-1] + 1, period.step_ms))
            vals = vals[vals > 0]
            p = float(np.median(vals)) if len(vals) else float('nan')
        else:
            p = dominant_period(act, c, span_ms=max(6000.0, win['t'][-1] - win['t'][0] + 2 * win['beat']))
        cls = _tempo_class(p / win['beat']) if np.isfinite(p) else 'other'
        counts[cls] += win['weight']
        total += win['weight']
    return {k: (v / total if total > 0 else float('nan')) for k, v in counts.items()}


def peaks(act: Activation, threshold: float = 0.5, radius_ms: float = 70.0) -> np.ndarray:
    """Local maxima over +-radius with value >= threshold, with parabolic sub-sample times (ms)."""
    v = act.values
    if len(v) < 3:
        return np.zeros(0)
    r = max(int(round(radius_ms / act.step_ms)), 1)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(v, r, constant_values=-np.inf)
    mx = sliding_window_view(pad, 2 * r + 1).max(axis=1)
    i = np.flatnonzero((v >= mx) & (v >= threshold))
    i = i[(i > 0) & (i < len(v) - 1)]
    if len(i):
        keep = np.concatenate([[True], np.diff(i) > r])     # one peak per plateau
        i = i[keep]
    y0, y1, y2 = v[i - 1], v[i], v[i + 1]
    den = y0 - 2 * y1 + y2
    frac = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
    return act.t0_ms + (i + frac) * act.step_ms


def phase_wraps(phase: Activation, kappa: Activation, min_kappa: float = 1.0) -> np.ndarray:
    """Times where a phase track wraps from above 0.75 to below 0.25, interpolated; F2's candidates."""
    p, k = phase.values, kappa.values
    i = np.flatnonzero((p[:-1] > 0.75) & (p[1:] < 0.25) & (k[1:] >= min_kappa))
    d = (p[i + 1] + 1.0) - p[i]
    frac = np.where(d > 0, (1.0 - p[i]) / d, 0.0)
    return phase.t0_ms + (i + frac) * phase.step_ms


def m3(pred_ms: np.ndarray, ref, tol_ms: float = TOL_MS) -> float:
    lines = ref.lines().t
    lines = lines[(lines >= ref.lo_ms - tol_ms) & (lines <= ref.hi_ms + tol_ms)]
    p = np.sort(np.asarray(pred_ms, dtype=np.float64))
    p = p[(p >= ref.lo_ms - tol_ms) & (p <= ref.hi_ms + tol_ms)]
    if not len(lines) or not len(p):
        return 0.0
    used = np.zeros(len(p), dtype=bool)
    hits = 0
    j = np.searchsorted(p, lines)
    for k, t in enumerate(lines):
        best, bd = -1, tol_ms + 1e-9
        for c in (j[k] - 1, j[k]):
            if 0 <= c < len(p) and not used[c] and abs(p[c] - t) <= bd:
                best, bd = c, abs(p[c] - t)
        if best >= 0:
            used[best] = True
            hits += 1
    prec, rec = hits / len(p), hits / len(lines)
    return 0.0 if hits == 0 else 2 * prec * rec / (prec + rec)


def bootstrap_mean(values: np.ndarray, clusters: list[str], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over items with a 95 % interval from resampling clusters (recordings)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    v = v[ok]
    cl = np.asarray(clusters)[ok]
    if not len(v):
        return float('nan'), float('nan'), float('nan')
    keys, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(keys))
    cnts = np.bincount(inv, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n, len(keys)))
    means = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff(a: np.ndarray, b: np.ndarray, clusters: list[str], n: int = 2000, seed: int = 0):
    """Paired mean difference a - b with a recording-cluster bootstrap interval."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return bootstrap_mean(a - b, clusters, n, seed)
