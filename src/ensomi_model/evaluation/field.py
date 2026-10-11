"""The event field: per-event values in physical time, their surprisal ranks, and a time field.

A chart is a set of hit objects with physical times in ms. An event family
scores the events it judges: for each it returns a time, a value and a key
(``EventValues``). Its model (``EventModel``), fitted on the corpus, turns a
value into a surprisal at the key and ranks the surprisal against cross-fitted
corpus events at the same key, in two tails: ``u_low`` small means surprising,
``u_high`` small means too typical. Each tail's ``-log u`` is smoothed over the
scored events by a Gaussian kernel in seconds; that is the field. A scope's
statistic is the time-average of the field over its scored spans, weights in
seconds; a second view is the worst sub-span of a fixed length. Every instant
of a scored span has a field value, so every head and release in it receives a
score; nothing is split or dropped.

The harness reads families only through ``EventFamily`` and ``EventModel``:
values, keys and surprisal ranks in, field and statistics out. Red lines are
never read here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .case import EvalCase

SQRT_2PI = math.sqrt(2.0 * math.pi)
CUTOFF_SIGMAS = 6.0
_ERF = np.frompyfunc(math.erf, 1, 1)


def normal_cdf(x) -> np.ndarray:
    """Standard normal CDF (libm ``erf``, elementwise)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + _ERF(x / math.sqrt(2.0)).astype(np.float64))


@dataclass(frozen=True)
class EventValues:
    """The events a family judges, in time order, with one value and one integer key each.

    ``scored`` marks events in the scope's scored spans (by head time);
    ``domain`` is the scored time the field averages over, a tuple of
    ``(start_ms, end_ms)`` intervals.
    """
    time_ms: np.ndarray
    value: np.ndarray
    key: np.ndarray
    scored: np.ndarray
    domain: tuple[tuple[float, float], ...]

    def __len__(self) -> int:
        return len(self.time_ms)


class EventModel(Protocol):
    """A family's model of normal, fitted on the fit split with cross-fitting by song group.

    ``fold`` is -1 for the model on every fit fold (calibration and generated
    charts) or the fold of a fit song (the model without that fold, and a
    reference without it).
    """
    def surprisal(self, value: np.ndarray, key: np.ndarray, fold: int) -> np.ndarray: ...

    def ranks(self, surprisal: np.ndarray, key: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]: ...

    def describe(self) -> Mapping[str, Any]: ...


class EventFamily(Protocol):
    """One description of a chart's events, with the aspects it judges (see ``case.evaluate``)."""
    name: str
    judges: frozenset[str]

    def values(self, case: EvalCase) -> EventValues: ...

    def sufficient(self, values: EventValues, fold: int) -> dict: ...

    def fit(self, stats: Sequence[dict]) -> EventModel: ...


def kernel_sum(query, times, weights, sigma_ms: float) -> np.ndarray:
    """Gaussian kernel sum in events per second: ``sum_j weights_j * phi(d_j) / sigma_s``, ``d_j`` in sigmas.

    Times and ``sigma_ms`` are in ms and must be sorted; terms beyond
    ``CUTOFF_SIGMAS`` are left out. Only differences of times enter, so a shift
    of every time by whole ms leaves the result bit for bit unchanged.
    """
    query = np.asarray(query, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    out = np.zeros(len(query))
    cut = CUTOFF_SIGMAS * sigma_ms
    for a in range(0, len(query), 256):
        q = query[a:a + 256]
        lo = int(np.searchsorted(times, q[0] - cut, 'left'))
        hi = int(np.searchsorted(times, q[-1] + cut, 'right'))
        if hi <= lo:
            continue
        d = (q[:, None] - times[None, lo:hi]) / sigma_ms
        k = np.where(np.abs(d) <= CUTOFF_SIGMAS, np.exp(-0.5 * d * d), 0.0)
        out[a:a + 256] = k @ weights[lo:hi]
    return out * (1000.0 / (sigma_ms * SQRT_2PI))


def smooth(query, times, x, h_ms: float) -> np.ndarray:
    """Nadaraya-Watson mean of ``x`` (rows aligned with sorted ``times``) at each query, Gaussian ``h_ms``.

    Events beyond ``CUTOFF_SIGMAS`` bandwidths are left out; a query with no
    event that close takes its nearest events, so the field is defined at every
    time. ``x`` may have several columns.
    """
    query = np.asarray(query, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    out = np.full((len(query), x2.shape[1]), np.nan)
    if not len(times) or not len(query):
        return out[:, 0] if squeeze else out
    cut = CUTOFF_SIGMAS * h_ms
    n = len(times)
    for a in range(0, len(query), 256):
        q = query[a:a + 256]
        lo = int(np.searchsorted(times, q[0] - cut, 'left'))
        hi = int(np.searchsorted(times, q[-1] + cut, 'right'))
        lo = max(0, min(lo, int(np.searchsorted(times, q[0], 'right')) - 1))
        hi = min(n, max(hi, int(np.searchsorted(times, q[-1], 'left')) + 1))
        d = (q[:, None] - times[None, lo:hi]) / h_ms
        logw = -0.5 * d * d
        inside = np.abs(d) <= CUTOFF_SIGMAS
        none = ~inside.any(axis=1)
        logw = np.where(inside | none[:, None], logw, -np.inf)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        out[a:a + 256] = w @ x2[lo:hi]
    return out[:, 0] if squeeze else out


def domain_cells(domain: Sequence[tuple[float, float]], step_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells of ``step_ms`` tiling each interval from its start: centres, lengths and interval index.

    The last cell of an interval is cut at its end. Cell edges sit at whole
    steps from the interval start, so a shift of the scope moves every cell
    with it.
    """
    centres, lengths, which = [], [], []
    for k, (lo, hi) in enumerate(domain):
        if not hi > lo:
            continue
        g = int(math.ceil((hi - lo) / step_ms))
        start = lo + np.arange(g) * step_ms
        length = np.full(g, float(step_ms))
        length[-1] = hi - start[-1]
        centres.append(start + length / 2)
        lengths.append(length)
        which.append(np.full(g, k))
    if not centres:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.concatenate(centres), np.concatenate(lengths), np.concatenate(which)


def field_statistics(times, x, domain, *, h_ms: float, step_ms: float, scan_ms: float) -> dict[str, np.ndarray]:
    """Time-average and worst ``scan_ms`` sub-span of the field of ``x`` over ``domain``.

    ``x`` holds one column per tail. The time-average weights cells by their
    length in seconds. The worst sub-span is the largest mean over runs of
    ``scan_ms / step_ms`` whole cells inside one interval; an interval shorter
    than that contributes its own mean. Empty domains give NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    k = x.shape[1]
    centres, lengths, which = domain_cells(domain, step_ms)
    seconds = float(lengths.sum()) / 1000.0
    if not len(centres) or not len(times):
        return dict(mean=np.full(k, np.nan), worst=np.full(k, np.nan), seconds=seconds)
    f = smooth(centres, times, x, h_ms)
    mean = (lengths[:, None] * f).sum(axis=0) / lengths.sum()
    run = max(1, int(round(scan_ms / step_ms)))
    worst = np.full(k, -np.inf)
    for i in np.unique(which):
        m = which == i
        fi, li = f[m], lengths[m]
        if len(fi) <= run:
            worst = np.maximum(worst, (li[:, None] * fi).sum(axis=0) / li.sum())
            continue
        cum = np.concatenate([np.zeros((1, k)), np.cumsum(li[:, None] * fi, axis=0)])
        lcum = np.concatenate([[0.0], np.cumsum(li)])
        means = (cum[run:] - cum[:-run]) / (lcum[run:] - lcum[:-run])[:, None]
        worst = np.maximum(worst, means.max(axis=0))
    return dict(mean=mean, worst=worst, seconds=seconds)


def clip_domain(spans, lo: float, hi: float) -> tuple[tuple[float, float], ...]:
    """Intervals of ``spans`` (a ``case.Spans``) inside ``[lo, hi]``."""
    out = []
    for a, b in spans.intervals:
        a, b = max(a, lo), min(b, hi)
        if b > a:
            out.append((float(a), float(b)))
    return tuple(out)
