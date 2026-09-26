"""Corpus-reference selection costs on explicit continuation observations.

The envelope describes ranked-chart attack rates, not physiological capacity.
Difficulty selects a reference range; it never alters committed gameplay state.
Holding and coordination remain separate response channels.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..scoped_style_modeling.dataset import ContractError
from .state import observe_continuation


@dataclass(frozen=True)
class AttackEnvelope:
    windows_ms: tuple[float, ...]
    stars: tuple[float, ...]
    maximum_hz: tuple[tuple[float, ...], ...]
    reference: str

    def __post_init__(self):
        windows, stars, rates = map(np.asarray, (self.windows_ms, self.stars, self.maximum_hz))
        if (windows.ndim != 1 or not len(windows) or np.any(windows <= 0) or
                stars.ndim != 1 or len(stars) < 2 or np.any(np.diff(stars) <= 0) or
                rates.shape != (len(stars), len(windows)) or np.any(rates <= 0) or
                not all(np.isfinite(v).all() for v in (windows, stars, rates))):
            raise ContractError('Attack envelope requires ordered difficulty knots and positive window rates')

    def rates(self, stars):
        """Interpolate reference Hz; requests beyond fitted knots use the endpoint."""
        values = np.asarray(self.maximum_hz)
        return np.stack([np.interp(stars, self.stars, values[:, j])
                         for j in range(len(self.windows_ms))], axis=-1)


def fit_attack_envelope(stars, groups, maxima_hz, *, windows_ms, knots, reference,
                        quantile=.99, band_width=1.):
    """Fit group-weighted chart maxima within difficulty bands.

    Inputs must already be restricted to the declared native-time TRAIN cohort.
    Each song group has equal total weight within a band. The fitted reference
    is monotone in difficulty; raw quantiles and cohort counts remain auditable.
    """
    stars, maxima = np.asarray(stars), np.asarray(maxima_hz)
    if (stars.ndim != 1 or len(groups) != len(stars) or
            maxima.shape != (len(stars), len(windows_ms)) or
            not np.isfinite(stars).all() or not np.isfinite(maxima).all() or
            not 0 < quantile < 1 or not np.isfinite(band_width) or band_width <= 0):
        raise ContractError('Envelope fitting requires aligned finite chart observations')
    raw, counts = [], []
    for knot in knots:
        indices = np.flatnonzero((stars >= knot-band_width/2) & (stars < knot+band_width/2))
        if not len(indices):
            raise ContractError('A requested envelope band contains no source charts')
        frequencies = Counter(groups[i] for i in indices)
        weights = np.array([1/frequencies[groups[i]] for i in indices])
        values = []
        for column in maxima[indices].T:
            order = np.argsort(column)
            index = np.searchsorted(np.cumsum(weights[order])/weights.sum(), quantile)
            values.append(float(column[order[min(index, len(order)-1)]]))
        raw.append(values)
        counts.append(dict(stars=knot, charts=len(indices), groups=len(frequencies)))
    monotone = np.maximum.accumulate(raw, axis=0)
    envelope = AttackEnvelope(tuple(windows_ms), tuple(knots), tuple(map(tuple, monotone)), reference)
    return envelope, dict(quantile=quantile, band_width=band_width, raw_hz=raw, bands=counts)


def sustained_response(state, continuation, end_ms, envelope, ranges):
    """Observe a private future and integrate excess separately by control range.

    Ranges are disjoint (start,end,stars) triples covering the response horizon;
    a missing stars value applies no selection cost. For each column/window,
    integrate squared positive relative rate excess in seconds, averaging window
    durations and summing columns. Attack and expiry events give an exact
    piecewise-constant integral, including the committed history's residual load.
    Returns the request-independent observations and a serializable selection
    report. The report is neither a star rating nor a complete gameplay response.
    """
    future, ranges = tuple(continuation), tuple(ranges)
    observations = observe_continuation(state, future, end_ms, windows_ms=envelope.windows_ms)
    if (not ranges or ranges[0][0] != state.time_ms or ranges[-1][1] != end_ms or
            any(b <= a or (d is not None and not np.isfinite(d)) for a, b, d in ranges) or
            any(left[1] != right[0] for left, right in zip(ranges, ranges[1:]))):
        raise ContractError('Response control ranges must partition the complete future horizon')
    attacks = [np.array((*state.attacks[c], *(r.time_ms for r in future if r.actions[c] in (1, 2))))
               for c in range(4)]
    widths = np.asarray(envelope.windows_ms)
    boundaries = [state.time_ms, end_ms, *(t for a, b, d in ranges for t in (a, b))]
    for times in attacks:
        boundaries.extend(times[(times > state.time_ms) & (times < end_ms)])
        for width in widths:
            expires = times+width
            boundaries.extend(expires[(expires > state.time_ms) & (expires < end_ms)])
    edges = np.unique(boundaries)
    left, duration = edges[:-1], np.diff(edges)/1000
    rates = np.empty((len(left), len(widths), 4))
    for c, times in enumerate(attacks):
        rates[:, :, c] = (np.searchsorted(times, left[:, None], side='right')-
            np.searchsorted(times, left[:, None]-widths, side='right'))*1000/widths
    reports = []
    for start, stop, stars in ranges:
        mask = (left >= start) & (left < stop)
        if stars is None:
            cost, peak = 0., None
        else:
            ratio = rates[mask]/envelope.rates(stars)[None, :, None]
            intensity = np.maximum(0., ratio-1.)**2
            cost = float((intensity.sum(-1).mean(-1)*duration[mask]).sum())
            peak = float(ratio.max())
        reports.append(dict(start_ms=start, end_ms=stop, stars=stars,
                            excess_seconds=cost, peak_ratio=peak))
    return observations, dict(excess_seconds=sum(r['excess_seconds'] for r in reports), ranges=reports)
