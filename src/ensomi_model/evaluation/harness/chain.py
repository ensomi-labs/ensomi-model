"""Score chain of harness v0 on the event field, for any family that follows ``field.EventFamily``.

Per case (``FieldOperator``): the family's values and keys; surprisal and
both-tail ranks from its model (cross-fitted for fit songs); each tail's
``-log u`` smoothed in seconds over the scored events (the field); the field's
time-average over the scored spans and its worst ``SCAN_S`` sub-span; and the
field read at every head and release of the chart in the scored spans, so that
every event receives a score.

Per song (``SongNull``): a statistic's p-value against cross-fitted fit songs,
kernel-weighted in log scored duration and requested star. For the
continuation scope the null is same-shape windows cut from fit songs at several
positions (``WINDOWS``), each with its first ``GIVEN_FRACTION`` given.

Estimator settings (reported, fixed before calibration is read): ``H_FIELD_S``,
``STEP_MS``, ``SCAN_S``, ``NULL_H_STAR``, ``NULL_H_LOG_SECONDS``,
``NULL_LOG_SECONDS_ROUND``, ``WINDOWS``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ...osu_core.hitobjects import ManiaHitObjectKind
from ..case import Chart, EvalCase
from ..field import EventFamily, EventModel, field_statistics, smooth

H_FIELD_S = 2.0
STEP_MS = 100.0
SCAN_S = 16.0
NULL_H_STAR = 0.25
NULL_H_LOG_SECONDS = 0.2
NULL_LOG_SECONDS_ROUND = 0.005
GIVEN_FRACTION = 0.25
WINDOWS = ((1.0, 0.0), (0.75, 0.0), (0.75, 0.25), (0.5, 0.0), (0.5, 0.25), (0.5, 0.5))
TAILS = ('low', 'high')
STATS = ('mean_low', 'mean_high', 'worst_low', 'worst_high')
HEAD, RELEASE = 1, 3


def chart_events(chart: Chart) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every head and long-note release: time, kind and the head time it belongs to; ordered by time."""
    rows = []
    for o in chart.objects:
        rows.append((o.start_time_ms, HEAD, o.lane, o.start_time_ms))
        if o.kind is ManiaHitObjectKind.HOLD:
            rows.append((o.end_time_ms, RELEASE, o.lane, o.start_time_ms))
    rows.sort()
    return (np.array([r[0] for r in rows], dtype=np.float64), np.array([r[1] for r in rows], dtype=np.int8),
            np.array([r[3] for r in rows], dtype=np.float64))


def continuation_cut(chart: Chart) -> float:
    """Start of the scored part of the continuation scope: the first ``GIVEN_FRACTION`` of the time
    from the chart's first to its last event (heads and releases) is given."""
    times, _, _ = chart_events(chart)
    return float(times[0] + GIVEN_FRACTION * (times[-1] - times[0]))


class FieldOperator:
    """A family and its model as a ``case.Operator``: per-event ranks, field, scope statistics, readouts."""

    def __init__(self, family: EventFamily, model: EventModel, fold: int = -1, *, h_field_s: float = H_FIELD_S,
                 step_ms: float = STEP_MS, scan_s: float = SCAN_S):
        self.family, self.model, self.fold = family, model, fold
        self.name, self.judges = family.name, family.judges
        self.h_ms, self.step_ms, self.scan_ms = h_field_s * 1000.0, step_ms, scan_s * 1000.0

    def __call__(self, case: EvalCase) -> dict[str, Any]:
        ev = self.family.values(case)
        s = self.model.surprisal(ev.value, ev.key, self.fold)
        u_low, u_high = self.model.ranks(s, ev.key, self.fold)
        x = np.stack([-np.log(u_low), -np.log(u_high)], axis=1) if len(ev) else np.zeros((0, 2))
        use = ev.scored & np.isfinite(x).all(axis=1)
        stats = field_statistics(ev.time_ms[use], x[use], ev.domain, h_ms=self.h_ms, step_ms=self.step_ms,
                                 scan_ms=self.scan_ms)
        times, kind, head = chart_events(case.chart)
        scored = case.scope.scored.contains(head)
        readout = smooth(times[scored], ev.time_ms[use], x[use], self.h_ms) if use.any() else \
            np.full((int(scored.sum()), 2), np.nan)
        return dict(time=ev.time_ms, value=ev.value, key=ev.key, scored=ev.scored, ranked=use, s=s, u_low=u_low,
                    u_high=u_high, x=x, domain=ev.domain, stats=stats, event_time=times[scored],
                    event_kind=kind[scored], readout=readout)


def window_chart(chart: Chart, a: float, b: float) -> Chart:
    """The objects of ``chart`` whose head lies in ``[a, b]``."""
    return Chart(tuple(o for o in chart.objects if a <= o.start_time_ms <= b), chart.red_lines, chart.keys,
                 chart.source)


def windows(chart: Chart) -> list[tuple[float, float, float, float]]:
    """Same-shape windows of a fit song for the continuation null: ``(fraction, position, start, end)``
    in ms over the chart's event extent."""
    times, _, _ = chart_events(chart)
    e0, e1 = float(times[0]), float(times[-1])
    return [(lam, rho, e0 + rho * (e1 - e0), e0 + (rho + lam) * (e1 - e0)) for lam, rho in WINDOWS]


@dataclass
class SongNull:
    """Fit songs (or windows) as an empirical null, kernel-weighted in log scored seconds and star.

    ``song`` indexes the fit chart each row comes from, so that the effective
    number of null songs counts a chart's windows once. A query's p-value is
    ``(1 + sum of weights of null rows at least as large) / (1 + sum of weights)``,
    the query counting as one null song at full weight.
    """
    star: np.ndarray
    log_seconds: np.ndarray
    song: np.ndarray
    stats: dict[str, np.ndarray]
    h_star: float = NULL_H_STAR
    h_log_seconds: float = NULL_H_LOG_SECONDS

    def __post_init__(self):
        self.star = np.asarray(self.star, dtype=np.float64)
        self.log_seconds = np.asarray(self.log_seconds, dtype=np.float64)
        ok = np.isfinite(self.star) & np.isfinite(self.log_seconds)
        self._order, self._sorted, self._valid = {}, {}, {}
        for name, v in self.stats.items():
            v = np.where(ok, np.asarray(v, dtype=np.float64), np.nan)
            order = np.argsort(np.where(np.isfinite(v), v, np.inf), kind='stable')
            self._order[name] = order
            self._sorted[name] = v[order]
            self._valid[name] = np.isfinite(v[order])
        self._ok = ok
        self._n_songs = int(self.song.max()) + 1 if len(self.song) else 0

    def __len__(self) -> int:
        return int(self._ok.sum())

    def weights(self, star: float, seconds: float) -> np.ndarray:
        log_s = round(math.log(seconds) / NULL_LOG_SECONDS_ROUND) * NULL_LOG_SECONDS_ROUND
        z = ((self.star - star) / self.h_star) ** 2 + ((self.log_seconds - log_s) / self.h_log_seconds) ** 2
        return np.where(self._ok, np.exp(-0.5 * z), 0.0)

    def p_values(self, star, seconds, stats: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """p-value per query and statistic, and the effective number of null songs per query."""
        star = np.asarray(star, dtype=np.float64)
        seconds = np.asarray(seconds, dtype=np.float64)
        n = len(star)
        out = {name: np.full(n, np.nan) for name in stats}
        n_eff = np.full(n, np.nan)
        ok = np.isfinite(star) & np.isfinite(seconds) & (seconds > 0)
        if not ok.any() or not len(self):
            return out, n_eff
        log_s = np.round(np.log(np.where(ok, seconds, 1.0)) / NULL_LOG_SECONDS_ROUND)
        pairs, inverse = np.unique(np.stack([np.where(ok, star, 0.0), log_s], axis=1), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        members = np.argsort(inverse, kind='stable')
        cuts = np.searchsorted(inverse[members], np.arange(len(pairs) + 1))
        for p in range(len(pairs)):
            q = members[cuts[p]:cuts[p + 1]]
            q = q[ok[q]]
            if not len(q):
                continue
            w = self.weights(float(star[q[0]]), float(seconds[q[0]]))
            by_song = np.bincount(self.song, weights=w, minlength=self._n_songs)
            n_eff[q] = w.sum() ** 2 / max(float((by_song ** 2).sum()), 1e-300)
            for name, values in stats.items():
                ws = w[self._order[name]] * self._valid[name]
                tail = np.concatenate([np.cumsum(ws[::-1])[::-1], [0.0]])
                v = np.asarray(values, dtype=np.float64)[q]
                idx = np.searchsorted(self._sorted[name], v, 'left')
                res = (1.0 + tail[idx]) / (1.0 + ws.sum())
                out[name][q] = np.where(np.isfinite(v), res, np.nan)
        return out, n_eff
