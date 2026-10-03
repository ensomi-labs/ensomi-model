"""The trivial evaluator of harness v0: one event family, head rate, and its model of normal.

Family (``HeadRate``): the rate of heads around each head, in heads per
second, from a Gaussian kernel in seconds (``SIGMA_S``), with every head of a
chord counted. The kernel mass outside the readable heads' extent is divided
out, so the first and last heads of a chart are not read as half as dense.
It reads heads in the scope's given and scored spans, judges head times, and
never reads red lines. The value of an event is the log of its rate.

Key: the requested star (the condition's star, for an injected or transformed
chart its source's, never recomputed), in bands of ``STAR_BAND_WIDTH``.

Model (``BinnedKDE``): per key, a density of the log rate on bins of
``V_BIN``, a Gaussian kernel of bandwidth chosen from ``BANDWIDTHS`` by
cross-fitted log-likelihood on the fit split (``K_FOLDS`` folds by song group),
and ``PSEUDO_EVENTS`` spread uniformly so no bin has zero density. Surprisal is
``-log density``. Ranks are mid-p ranks of the surprisal among cross-fitted fit
events at the key: a fit event's surprisal comes from the density without its
fold; a fit song is ranked against the events of the other folds.

Helpers kept from v0 for the harness: star and canonical-BPM bands (the BPM
band is used only to choose splice donors and to stratify reports), folds,
Wilson and song-group bootstrap intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from ..case import HEAD_TIMES, EvalCase
from ..field import EventValues, clip_domain, kernel_sum, normal_cdf

SIGMA_S = 4.0
K_FOLDS = 5
FOLD_SALT = 'ensomi-harness-v0-folds:'
STAR_BAND_WIDTH = 0.5
BPM_BAND_WIDTH = 20.0
BPM_BANDS = 4
CANONICAL_MIN = 80.0
N_KEYS = 24
V_MIN, V_BIN, N_BINS = -4.0, 0.002, 5000
BANDWIDTHS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
PSEUDO_EVENTS = 1.0


def star_band(star) -> np.ndarray:
    return np.floor(np.asarray(star, dtype=np.float64) / STAR_BAND_WIDTH).astype(np.int64)


def star_band_label(band: int) -> str:
    return f'{band * STAR_BAND_WIDTH:.1f}-{(band + 1) * STAR_BAND_WIDTH:.1f}'


def bpm_band(canonical_bpm) -> np.ndarray:
    band = np.floor((np.asarray(canonical_bpm, dtype=np.float64) - CANONICAL_MIN) / BPM_BAND_WIDTH)
    return np.clip(band, 0, BPM_BANDS - 1).astype(np.int64)


def bpm_band_label(band: int) -> str:
    return f'{CANONICAL_MIN + band * BPM_BAND_WIDTH:.0f}-{CANONICAL_MIN + (band + 1) * BPM_BAND_WIDTH:.0f}'


def fold_of(group_id: str) -> int:
    return int(hashlib.sha256((FOLD_SALT + group_id).encode()).hexdigest()[:8], 16) % K_FOLDS


def value_bin(value) -> np.ndarray:
    return np.clip(np.floor((np.asarray(value, dtype=np.float64) - V_MIN) / V_BIN), 0, N_BINS - 1).astype(np.int64)


@dataclass(frozen=True)
class HeadRate:
    """Heads per second around each head (Gaussian kernel, ``sigma_s`` seconds), as its log."""
    sigma_s: float = SIGMA_S
    name: str = 'head_rate'
    judges: frozenset[str] = frozenset((HEAD_TIMES,))

    def values(self, case: EvalCase) -> EventValues:
        star = case.condition.star
        if star is None:
            raise ValueError('The head-rate family is keyed by the requested star; the condition has none')
        heads = np.sort(np.array([o.start_time_ms for o in case.chart.objects], dtype=np.float64))
        readable = case.scope.given | case.scope.scored
        t = heads[readable.contains(heads)] if len(heads) else heads
        if not len(t):
            empty = np.zeros(0)
            return EventValues(empty, empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool), ())
        unique, inverse, count = np.unique(t, return_inverse=True, return_counts=True)
        sigma_ms = self.sigma_s * 1000.0
        raw = kernel_sum(unique, unique, count, sigma_ms)
        lo, hi = float(unique[0]), float(unique[-1])
        extent = clip_domain(readable, lo, hi)
        if extent:
            mass = np.zeros(len(unique))
            for a, b in extent:
                mass += normal_cdf((b - unique) / sigma_ms) - normal_cdf((a - unique) / sigma_ms)
        else:
            mass = np.ones(len(unique))
        value = np.log(raw / mass)[inverse]
        return EventValues(t, value, np.full(len(t), int(star_band(star))), case.scope.scored.contains(t),
                           clip_domain(case.scope.scored, lo, hi))

    def sufficient(self, values: EventValues, fold: int) -> dict:
        """Counts of value bins per key, for this fit song's fold."""
        out = {}
        bins = value_bin(values.value)
        for k in np.unique(values.key):
            idx, cnt = np.unique(bins[values.key == k], return_counts=True)
            out[(int(k), int(fold))] = (idx.astype(np.int32), cnt.astype(np.int64))
        return out

    def fit(self, stats: Sequence[dict]) -> 'BinnedKDE':
        counts = np.zeros((N_KEYS, K_FOLDS, N_BINS), dtype=np.float64)
        for chunk in stats:
            for (k, f), (idx, cnt) in chunk.items():
                if 0 <= k < N_KEYS:
                    np.add.at(counts[k, f], idx, cnt)
        return BinnedKDE.fit(counts)


def _kernel(bandwidth: float) -> np.ndarray:
    sd = bandwidth / V_BIN
    half = int(math.ceil(5 * sd))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sd) ** 2)
    return k / k.sum()


def _density(counts: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Smoothed density per unit of log rate; ``PSEUDO_EVENTS`` spread over every bin."""
    smoothed = np.convolve(counts, kernel, mode='same')
    total = counts.sum()
    return (smoothed + PSEUDO_EVENTS / N_BINS) / (total + PSEUDO_EVENTS) / V_BIN


@dataclass
class BinnedKDE:
    """Binned Gaussian-kernel densities of the log head rate per key: on all fit folds and without each fold."""
    counts: np.ndarray                      # [key, fold, bin]
    bandwidth: float
    loglik: dict[str, float]
    full: np.ndarray = field(init=False)    # [key, bin]
    without: np.ndarray = field(init=False)  # [key, fold, bin]
    has: np.ndarray = field(init=False)      # [key]: any fit event
    _tables: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        kernel = _kernel(self.bandwidth)
        total = self.counts.sum(axis=1)
        self.has = total.sum(axis=1) > 0
        self.full = np.stack([_density(total[k], kernel) for k in range(N_KEYS)])
        self.without = np.stack([np.stack([_density(total[k] - self.counts[k, f], kernel) for f in range(K_FOLDS)])
                                 for k in range(N_KEYS)])

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_tables'] = {}
        return state

    @classmethod
    def fit(cls, counts: np.ndarray) -> 'BinnedKDE':
        """Bandwidth with the largest cross-fitted log-likelihood of fit events (folds by song group)."""
        total = counts.sum(axis=1)
        loglik = {}
        for b in BANDWIDTHS:
            kernel = _kernel(b)
            ll = 0.0
            for k in np.flatnonzero(total.sum(axis=1) > 0):
                for f in range(K_FOLDS):
                    if counts[k, f].sum():
                        ll += float(counts[k, f] @ np.log(_density(total[k] - counts[k, f], kernel)))
            loglik[str(b)] = ll
        best = max(BANDWIDTHS, key=lambda b: loglik[str(b)])
        return cls(counts, best, loglik)

    def n(self, key: int) -> float:
        return float(self.counts[key].sum()) if 0 <= key < N_KEYS else 0.0

    def surprisal(self, value, key, fold: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        key = np.asarray(key, dtype=np.int64)
        out = np.full(len(value), np.nan)
        ok = (key >= 0) & (key < N_KEYS)
        if not ok.any():
            return out
        table = self.full if fold < 0 else self.without[:, fold]
        k = np.clip(key, 0, N_KEYS - 1)
        s = -np.log(table[k, value_bin(value)])
        use = ok & self.has[k]
        out[use] = s[use]
        return out

    def _table(self, key: int, fold: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Sorted distinct surprisals of fit events at ``key`` (outside ``fold``) and cumulative counts."""
        cached = self._tables.get((key, fold))
        if cached is not None:
            return cached
        values, weights = [], []
        for g in range(K_FOLDS):
            if g == fold:
                continue
            idx = np.flatnonzero(self.counts[key, g])
            values.append(-np.log(self.without[key, g, idx]))
            weights.append(self.counts[key, g, idx])
        v, w = np.concatenate(values), np.concatenate(weights)
        order = np.argsort(v, kind='stable')
        v, w = v[order], w[order]
        distinct, start = np.unique(v, return_index=True)
        summed = np.add.reduceat(w, start) if len(w) else w
        cum = np.concatenate([[0.0], np.cumsum(summed)])
        self._tables[(key, fold)] = (distinct, cum, float(cum[-1]))
        return self._tables[(key, fold)]

    def ranks(self, surprisal, key, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Mid-p ranks: ``u_low`` = share at least as surprising, ``u_high`` = share at most as surprising.

        The query counts as half a tie, so neither is 0 or 1 and ``u_low + u_high = 1``.
        """
        s = np.asarray(surprisal, dtype=np.float64)
        key = np.asarray(key, dtype=np.int64)
        u_low, u_high = np.full(len(s), np.nan), np.full(len(s), np.nan)
        for k in np.unique(key):
            if not (0 <= k < N_KEYS) or self.n(int(k)) == 0:
                continue
            q = (key == k) & np.isfinite(s)
            if not q.any():
                continue
            distinct, cum, n = self._table(int(k), fold)
            less = cum[np.searchsorted(distinct, s[q], 'left')]
            le = cum[np.searchsorted(distinct, s[q], 'right')]
            tie = le - less
            u_low[q] = (n - le + 0.5 * tie + 0.5) / (n + 1)
            u_high[q] = (less + 0.5 * tie + 0.5) / (n + 1)
        return u_low, u_high

    def mode_rate(self, key: int) -> float | None:
        """Heads per second at the mode of the full density at ``key``."""
        if self.n(key) == 0:
            return None
        return float(math.exp(V_MIN + (int(np.argmax(self.full[key])) + 0.5) * V_BIN))

    def describe(self) -> dict[str, Any]:
        keys = {}
        for k in range(N_KEYS):
            total = self.counts[k].sum(axis=0)
            if not total.sum():
                continue
            cdf = np.cumsum(total) / total.sum()
            q = {f'p{int(100 * p)}': round(float(np.exp(V_MIN + (np.searchsorted(cdf, p) + 0.5) * V_BIN)), 4)
                 for p in (0.05, 0.5, 0.95)}
            idx = np.flatnonzero(total)
            keys[star_band_label(k)] = dict(key=k, events=int(total.sum()), mode_rate=self.mode_rate(k),
                                            rate_quantiles=q, first_bin=int(idx[0]),
                                            counts=total[idx[0]:idx[-1] + 1].astype(np.int64).tolist())
        return dict(value='log heads per second', bins=dict(start=V_MIN, width=V_BIN, count=N_BINS),
                    bandwidth=self.bandwidth, bandwidth_candidates=list(BANDWIDTHS),
                    cross_fitted_loglik=self.loglik, pseudo_events=PSEUDO_EVENTS, k_folds=K_FOLDS,
                    fold_salt=FOLD_SALT, keys=keys)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def group_bootstrap(flags, groups, *, draws: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of a rate, resampling song groups with replacement."""
    flags = np.asarray(flags, dtype=np.float64)
    if not len(flags):
        return math.nan, math.nan
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    hits = np.bincount(g, weights=flags)
    size = np.bincount(g).astype(np.float64)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(hits), np.full(len(hits), 1.0 / len(hits)), size=draws)
    rates = (weights @ hits) / np.maximum(weights @ size, 1e-12)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))
