"""Aggregation over recordings: strata, the recording as the unit, and bootstrap intervals.

The source recording is the decision unit. All sets of one recording form one
unit: a recording's value of a measure is the mean over its sets (or over the
sets the strata table marks as its reference), and pooled counts are averaged
the same way, so every recording weighs 1. Intervals are 95 % percentile
bootstraps that resample recordings with replacement.

Per stratum, a measure is reported as ``{"value", "lo", "hi"}``. Means over
recordings skip recordings where the measure is NaN; ``n`` counts the rest.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
import os
from typing import Iterable, Sequence

import numpy as np

from .measures import (AUC_LEVELS, ECE_BINS, EDIT_TYPES, LEVEL_CLASSES, SUBDIVISION_CLASSES, TAUS, SetScore,
                       auc_from_hist, ece_from_bins, kappa)

S_BL_THRESHOLD = 0.9
RESET_UNION_MS = 20.0

#: Per-recording means, with whether each decides or characterises.
MEAN_MEASURES = (
    [f'bl{t:g}' for t in TAUS] + [f'bl{t:g}_beat' for t in TAUS] + [f'bl{t:g}_o' for t in TAUS if t != 20.0]
    + [f'chance{t:g}' for t in TAUS]
    + ['level'] + [f'level_{c}' for c in LEVEL_CLASSES]
    + ['bars_start', 'bars_same', 'bars_precision', 'check_load_s_per_min', 'hidden_per_min', 'edits_per_min']
    + [f'edits_{k}' for k in EDIT_TYPES] + ['redo_s_per_min']
)

ROLES = {
    'bl20': 'decides, by the human rule (d-bl20-no-offset); no measure is validated against listening yet',
    'offset': 'reported on its own term; never subtracted from BL-20',
}
CHARACTERISES = 'characterises'


@dataclass
class Entry:
    """One scored set and what the strata table says about it."""
    score: SetScore
    recording: str
    source: str = ''
    reference: bool | None = None
    strata: dict[str, list | None] = field(default_factory=dict)


def _recording_table(entries: Sequence[tuple[SetScore, str]]):
    """Per-recording scalar means and pooled counts, recordings in first-seen order."""
    order: list[str] = []
    groups: dict[str, list[SetScore]] = {}
    for s, rec in entries:
        if rec not in groups:
            order.append(rec)
            groups[rec] = []
        groups[rec].append(s)
    R = len(order)
    values = {name: np.full(R, np.nan) for name in MEAN_MEASURES + ['offset_ms']}
    pooled_keys = ('ece_bins', 'auc_hist', 'confident', 'kappa', 'boundaries', 'windows')
    pooled = {}
    resets = np.zeros((R, 2))
    counts = dict(n_sets=np.zeros(R), heads=np.zeros(R), minutes=np.zeros(R), n_scored=np.zeros(R))
    for r, rec in enumerate(order):
        sets = groups[rec]
        summaries = [s.summary() for s in sets]
        for name in values:
            v = np.array([float(x[name]) if x[name] is not None else np.nan for x in summaries], dtype=np.float64)
            if np.isfinite(v).any():
                values[name][r] = np.nanmean(v)
        for k in pooled_keys:
            arr = np.mean([s.suff[k] for s in sets], axis=0)
            pooled.setdefault(k, np.zeros((R,) + arr.shape))[r] = arr
        # Reset recall against the union of the recording's sets.
        t = np.concatenate([s.resets['t'] for s in sets]) if sets else np.zeros(0)
        ok = np.concatenate([s.resets['ok'] for s in sets]) if sets else np.zeros(0, dtype=bool)
        if len(t):
            order_t = np.argsort(t)
            t, ok = t[order_t], ok[order_t]
            group = np.concatenate([[0], np.cumsum(np.diff(t) > RESET_UNION_MS)])
            n_groups = int(group[-1]) + 1
            hit = np.zeros(n_groups, dtype=bool)
            np.logical_or.at(hit, group, ok)
            resets[r] = (n_groups, hit.sum())
        counts['n_sets'][r] = len(sets)
        counts['heads'][r] = np.nansum([s.heads for s in sets]) / len(sets)
        counts['minutes'][r] = np.mean([s.minutes for s in sets])
        counts['n_scored'][r] = np.mean([s.n_scored for s in sets])
    pooled['resets'] = resets
    return order, values, pooled, counts


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return math.nan
    order = np.argsort(x[ok])
    xs, cum = x[ok][order], np.cumsum(w[ok][order])
    k = int(np.searchsorted(cum, q * cum[-1] - 1e-12 * cum[-1], side='left'))
    if abs(cum[k] - q * cum[-1]) <= 1e-12 * cum[-1] and k + 1 < len(xs):
        return float((xs[k] + xs[k + 1]) / 2.0)
    return float(xs[k])


def _statistics(M: np.ndarray, values: dict, pooled: dict) -> dict[str, np.ndarray]:
    """Every aggregate statistic for each row of recording multiplicities ``M`` (rows, recordings)."""
    out: dict[str, np.ndarray] = {}
    for name in MEAN_MEASURES:
        v = values[name]
        fin = np.isfinite(v)
        num = M @ np.where(fin, v, 0.0)
        den = M @ fin.astype(np.float64)
        out[name] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    bl = values['bl20']
    fin = np.isfinite(bl)
    den = M @ fin.astype(np.float64)
    out['s_bl20'] = np.where(den > 0, (M @ ((bl >= S_BL_THRESHOLD) & fin).astype(np.float64))
                             / np.where(den > 0, den, 1.0), np.nan)
    o = values['offset_ms']
    fin = np.isfinite(o)
    den = M @ fin.astype(np.float64)
    for tol in (5.0, 10.0, 20.0):
        out[f'offset_within_{tol:g}'] = np.where(
            den > 0, (M @ ((np.abs(o) <= tol) & fin).astype(np.float64)) / np.where(den > 0, den, 1.0), np.nan)
    med = np.array([_weighted_quantile(o, m, 0.5) for m in M])
    out['offset_median'] = med
    out['offset_robust_sd'] = np.array([1.4826 * _weighted_quantile(np.abs(o - md), m, 0.5)
                                        for m, md in zip(M, med)])
    sums = {k: np.tensordot(M, v, axes=(1, 0)) for k, v in pooled.items()}
    out['ece'] = np.array([ece_from_bins(b) for b in sums['ece_bins']])
    out['auc'] = np.array([auc_from_hist(h) for h in sums['auc_hist']])
    kap = np.array([kappa(c) for c in sums['kappa']])
    out['subdivision_kappa'], out['subdivision_accuracy'] = kap[:, 0], kap[:, 1]
    c = sums['confident']
    out['confident_fail'] = np.where(c[:, 0] > 0, c[:, 1] / np.where(c[:, 0] > 0, c[:, 0], 1), np.nan)
    out['confident_predicted_fail'] = np.where(c[:, 0] > 0, c[:, 2] / np.where(c[:, 0] > 0, c[:, 0], 1), np.nan)
    b = sums['boundaries']
    out['boundary_recall'] = np.where(b[:, 0] > 0, b[:, 1] / np.where(b[:, 0] > 0, b[:, 0], 1), np.nan)
    out['boundary_precision'] = np.where(b[:, 2] > 0, b[:, 3] / np.where(b[:, 2] > 0, b[:, 2], 1), np.nan)
    rs = sums['resets']
    out['reset_recall'] = np.where(rs[:, 0] > 0, rs[:, 1] / np.where(rs[:, 0] > 0, rs[:, 0], 1), np.nan)
    wn = sums['windows']
    out['hidden_share'] = np.where(wn[:, 0] > 0, wn[:, 1] / np.where(wn[:, 0] > 0, wn[:, 0], 1), np.nan)
    return out


def aggregate(scores: Sequence[SetScore], recordings: Sequence[str] | None = None, *, bootstrap: int = 2000,
              seed: int = 0) -> dict:
    """Aggregate set scores into one stratum's report.

    ``recordings`` names each set's recording (default: one recording per
    set). Returns ``{measure: {"value", "lo", "hi"}}`` plus counts. With
    ``bootstrap`` 0 the interval bounds are NaN.
    """
    recordings = list(recordings) if recordings is not None else [s.key or str(i) for i, s in enumerate(scores)]
    if len(recordings) != len(scores):
        raise ValueError('One recording name per score')
    if not scores:
        return {'n_recordings': 0, 'n_sets': 0}
    order, values, pooled, counts = _recording_table(list(zip(scores, recordings)))
    R = len(order)
    point = _statistics(np.ones((1, R)), values, pooled)
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        M = rng.multinomial(R, np.full(R, 1.0 / R), size=bootstrap).astype(np.float64)
        boot = _statistics(M, values, pooled)
    else:
        boot = None
    out: dict = dict(n_recordings=R, n_sets=int(counts['n_sets'].sum()),
                     heads=float(np.nansum(counts['heads'])), minutes=float(counts['minutes'].sum()),
                     n_scored_lines=float(counts['n_scored'].sum()))
    n_finite = {name: int(np.isfinite(values[name]).sum()) for name in MEAN_MEASURES}
    for name, v in point.items():
        entry = {'value': _clean(v[0])}
        if boot is not None:
            b = boot[name][np.isfinite(boot[name])]
            entry['lo'] = _clean(np.percentile(b, 2.5)) if len(b) else None
            entry['hi'] = _clean(np.percentile(b, 97.5)) if len(b) else None
        else:
            entry['lo'] = entry['hi'] = None
        if name in n_finite:
            entry['n'] = n_finite[name]
        out[name] = entry
    return out


def _clean(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def read_strata(path: str | os.PathLike) -> dict[str, dict]:
    """Read a strata table (CSV with a header).

    Columns: ``key`` (required; the set key), ``recording``, ``stratum``
    (one name, or several separated by ``;``), ``source`` (for example
    ranked or loved), ``start_ms`` and ``end_ms`` (a span; both empty for the
    whole recording), ``reference`` (1 or 0). A key may take several rows.
    Returns ``{key: {"recording", "source", "reference", "strata": {name: spans or None}}}``.
    """
    table: dict[str, dict] = {}
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or 'key' not in reader.fieldnames:
            raise ValueError(f'{path}: a strata table needs a "key" column')
        for row in reader:
            key = (row.get('key') or '').strip()
            if not key:
                continue
            item = table.setdefault(key, dict(recording='', source='', reference=None, strata={}))
            rec = (row.get('recording') or '').strip()
            if rec:
                if item['recording'] and item['recording'] != rec:
                    raise ValueError(f'{path}: key {key!r} names two recordings')
                item['recording'] = rec
            src = (row.get('source') or '').strip()
            if src:
                item['source'] = src
            ref = (row.get('reference') or '').strip()
            if ref:
                item['reference'] = ref not in ('0', 'false', 'False', 'no')
            a, b = (row.get('start_ms') or '').strip(), (row.get('end_ms') or '').strip()
            span = (float(a), float(b)) if a and b else None
            for name in (row.get('stratum') or '').split(';'):
                name = name.strip()
                if not name:
                    continue
                spans = item['strata'].get(name, [])
                if span is None or spans is None:
                    item['strata'][name] = None
                else:
                    item['strata'][name] = spans + [span]
    return table


def build_strata(entries: Iterable[Entry], scores_by_span: dict | None = None, *, bootstrap: int = 2000,
                 seed: int = 0) -> dict:
    """Aggregate entries into ``all``, each stratum, and each ``stratum|source`` split.

    An entry's score for a stratum with spans comes from ``scores_by_span``
    (keyed by ``(set key, stratum)``); a stratum without spans uses the whole
    score. A recording with a set marked as reference counts only its marked
    sets; a set marked 0 never counts.
    """
    entries = list(entries)
    scores_by_span = scores_by_span or {}
    marked = {e.recording for e in entries if e.reference is True}
    groups: dict[str, list[tuple[SetScore, str]]] = {}
    for e in entries:
        if e.reference is False or (e.reference is None and e.recording in marked):
            continue
        names = {'all': None, **e.strata}
        for name, spans in names.items():
            s = e.score if spans is None else scores_by_span.get((e.score.key, name))
            if s is None:
                continue
            groups.setdefault(name, []).append((s, e.recording))
            if e.source:
                groups.setdefault(f'{name}|{e.source}', []).append((s, e.recording))
    return {name: aggregate([s for s, _ in rows], [r for _, r in rows], bootstrap=bootstrap, seed=seed)
            for name, rows in sorted(groups.items(), key=lambda kv: (kv[0] != 'all', kv[0]))}

