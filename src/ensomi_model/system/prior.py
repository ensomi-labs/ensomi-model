"""Corpus prior: what human charts of a star band do on head rows like these.

Generated rows have no arrangement, so the four things the demo took from a source chart come from the request
and this prior: the kappa 3 force targets (held share, heads per row, all-busy share), the natural LN level (the
held share), and the base r = sqrt(W / W_H) from which the difficulty map sets its heads-per-row multiplier.

Row features (``rows_features``, the same for human and generated rows): log rows per second, log median gap,
log W_H per second (the rows-only reference workload, ``r2.strain``), log canonical BPM at the median head.
Targets (``arrangement_targets``) of a human arrangement: held share (lanes held before each row), heads per
row, all-busy rows without a quad, quad rows, log r.

Three predictors, fitted per band on fit_train (``fit``): the band median, a per-band least-squares line on
the features, and the median of the 64 nearest fit_train charts of the band (standardised features). The fit
scores each on fit_dev by mean absolute error with a song-group bootstrap and keeps the plainest whose error on
every target is not worse than the best's (paired interval of the difference covers zero or favours it).
Run from the repository root::

    python -m ensomi_model.system.prior --cache artifacts/r2-cache/v1 --out <dir> --workers 8
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from ..r2.strain import StrainTrace

FEATURES = ('log_rows_s', 'log_gap_ms', 'log_wh_s', 'log_bpm')
TARGETS = ('held', 'heads', 'busy_free', 'quad', 'log_r')
METHODS = ('median', 'linear', 'knn')        # plainest first
NEIGHBOURS = 64


def rows_features(head_ms, song_ms: float, grid) -> np.ndarray:
    """[4] features of a head skeleton (FEATURES order)."""
    head_ms = np.asarray(head_ms, dtype=np.float64)
    seconds = song_ms / 1000.0
    wh = StrainTrace(head_ms, song_ms).reference_prefix[-1]
    gap = np.median(np.diff(head_ms)) if len(head_ms) > 1 else seconds * 1000.0
    bpm = float(np.median(grid.bpm(head_ms)))
    return np.array([np.log(len(head_ms) / seconds), np.log(max(gap, 1.0)), np.log(max(wh, 1e-6) / seconds),
                     np.log(bpm)])


def arrangement_targets(head_ms, song_ms: float, actions, gap) -> np.ndarray:
    """[5] targets of an arrangement (TARGETS order); actions [K+1, 4] with EOS."""
    from ..r2.theta import decode
    K = len(head_ms)
    held, heads, _ = decode(actions[:K])
    quad = heads.all(1)
    busy = (held | heads).all(1)
    r = StrainTrace(head_ms, song_ms).extend(actions, gap).scope(0, song_ms).ratio
    return np.array([held.mean(), heads.sum(1).mean(), (busy & ~quad).mean(), quad.mean(), np.log(r)])


def kappa_targets(pred: dict, ln_fraction: float = 1.0, multiplier: float = 1.0) -> np.ndarray:
    """The demo's ``source_targets`` from predicted targets: held share, heads per row, all-busy share."""
    return np.array([pred['held'] * ln_fraction, pred['heads'] * multiplier,
                     pred['busy_free'] * ln_fraction + pred['quad'] * min(1.0, multiplier)])


def _clip(name, v):
    if name in ('held', 'busy_free', 'quad'):
        return float(np.clip(v, 0.0, 1.0))
    if name == 'heads':
        return float(np.clip(v, 1.0, 4.0))
    return float(v)


class Prior:
    """A fitted prior: ``predict(features, band)`` -> dict of TARGETS, plus the method used."""

    def __init__(self, record: dict):
        self.record = record
        self.method = record['method']

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, features, band: int) -> dict:
        p = self.record['bands'][str(band)]
        x = np.asarray(features, dtype=np.float64)
        if self.method == 'median':
            v = np.asarray(p['median'])
        elif self.method == 'linear':
            v = np.r_[1.0, x] @ np.asarray(p['coef'])
        else:
            z = (x - np.asarray(p['mean'])) / np.asarray(p['sd'])
            table = np.asarray(p['table'])            # [n, 4 + 5] standardised features, then targets
            near = np.argsort(np.linalg.norm(table[:, :4] - z, axis=1), kind='stable')[:NEIGHBOURS]
            v = np.median(table[near, 4:], axis=0)
        return {name: _clip(name, x) for name, x in zip(TARGETS, v)}


def fit_band(X, Y) -> dict:
    """All three predictors' parameters for one band's fit_train charts."""
    coef = np.linalg.lstsq(np.column_stack((np.ones(len(X)), X)), Y, rcond=None)[0]
    mean, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return dict(n=len(X), median=np.median(Y, 0).tolist(), coef=coef.tolist(), mean=mean.tolist(), sd=sd.tolist(),
                table=np.column_stack(((X - mean) / sd, Y)).tolist())


def _chart_job(row):
    from ..r2.cache import load_chart
    from ..r2.common import GridArrays, grid_from_arrays
    dec = load_chart(Path(row['root']) / 'charts' / row['file'])
    grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
    try:
        x = rows_features(dec.head_ms, dec.song_ms, grid)
        y = arrangement_targets(dec.head_ms, dec.song_ms, dec.actions.astype(np.int64), dec.gap_release_ms)
    except Exception as exc:        # a chart the strain replay rejects is recorded, not fatal
        return row['sha'], None, None, f'{type(exc).__name__}: {exc}'[:200]
    return row['sha'], x, y, None


def _bootstrap_mae(err, groups, reps=1000, seed=712):
    """Mean of err [n, T] and the 2.5/97.5 % song-group bootstrap of that mean, per column."""
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    sums = np.zeros((G, err.shape[1]))
    np.add.at(sums, inv, err)
    counts = np.bincount(inv, minlength=G)
    w = np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=reps)
    boot = (w @ sums) / (w @ counts)[:, None]
    return err.mean(0), np.percentile(boot, [2.5, 97.5], axis=0)


def choose(errors: dict, groups) -> tuple[str, dict]:
    """The plainest method whose fit_dev error on every target is not worse than the best's. When none holds up
    on every target, the method with the smallest summed error ratio to each target's best method."""
    mae = {m: e.mean(0) for m, e in errors.items()}
    best = {t: min(METHODS, key=lambda m: mae[m][i]) for i, t in enumerate(TARGETS)}
    table = {}
    chosen = None
    for m in METHODS:
        ok = True
        rows = {}
        for i, t in enumerate(TARGETS):
            diff = errors[m][:, i:i + 1] - errors[best[t]][:, i:i + 1]
            mean, ci = _bootstrap_mae(diff, groups)
            rows[t] = dict(mae=float(mae[m][i]), minus_best=float(mean[0]), lo=float(ci[0, 0]), hi=float(ci[1, 0]),
                           best=best[t])
            ok &= bool(ci[0, 0] <= 0)
        table[m] = dict(targets=rows, holds_up=ok)
        if ok and chosen is None:
            chosen = m
    if chosen is None:
        ratio = {m: float(sum(mae[m][i] / mae[best[t]][i] for i, t in enumerate(TARGETS))) for m in METHODS}
        for m in METHODS:
            table[m]['summed_ratio_to_best'] = ratio[m]
        chosen = min(METHODS, key=ratio.get)
    return chosen, table


def fit(cache_root: Path, out: Path, workers: int, limit=None):
    import pyarrow.parquet as pq
    t0 = time.time()
    index = pq.read_table(cache_root / 'index.parquet').to_pandas()
    index = index[index.role.isin(('fit_train', 'fit_dev'))].sort_values('sha256')
    if limit:
        index = index.iloc[np.random.default_rng(0).permutation(len(index))[:limit]]
    rows = [dict(sha=r.sha256, file=r.file, root=str(cache_root)) for r in index.itertuples()]
    feats, targs, failed = {}, {}, {}
    with get_context('spawn').Pool(workers) as pool:
        for sha, x, y, err in pool.imap_unordered(_chart_job, rows, chunksize=32):
            if err:
                failed[sha] = err
            else:
                feats[sha], targs[sha] = x, y
    index = index[index.sha256.isin(feats)]
    record = dict(features=FEATURES, targets=TARGETS, neighbours=NEIGHBOURS, failed=failed, bands={}, dev={},
                  charts={role: int((index.role == role).sum()) for role in ('fit_train', 'fit_dev')})
    errors = {m: [] for m in METHODS}
    dev_groups = []
    for band in range(2, 6):
        tr = index[(index.role == 'fit_train') & (index.band == band)]
        dv = index[(index.role == 'fit_dev') & (index.band == band)]
        X, Y = np.stack([feats[s] for s in tr.sha256]), np.stack([targs[s] for s in tr.sha256])
        record['bands'][str(band)] = fit_band(X, Y)
        for m in METHODS:
            prior = Prior(dict(method=m, bands=record['bands']))
            pred = np.array([[prior.predict(feats[s], band)[t] for t in TARGETS] for s in dv.sha256])
            truth = np.stack([targs[s] for s in dv.sha256])
            errors[m].append(np.abs(pred - truth))
        dev_groups += list(dv.group_id)
        record['dev'][str(band)] = dict(charts=len(dv), train_charts=len(tr))
    errors = {m: np.concatenate(e) for m, e in errors.items()}
    method, table = choose(errors, np.asarray(dev_groups))
    record.update(method=method, selection=table, seconds=time.time() - t0,
                  rule='plainest of median, linear, knn whose fit_dev MAE on every target is not worse than the '
                       "best method's (paired song-group bootstrap interval of the difference reaches zero); when "
                       "none is, the smallest sum over targets of MAE / the target's best MAE")
    if method != 'knn':                 # the neighbour tables are only needed by the knn predictor
        for b in record['bands'].values():
            b.pop('table')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prior.json').write_text(json.dumps(record))
    print(json.dumps(dict(method=record['method'], charts=record['charts'], failed=len(failed),
                          selection={m: dict(holds_up=v['holds_up'],
                                             mae={t: round(x['mae'], 4) for t, x in v['targets'].items()})
                                     for m, v in table.items()}, seconds=round(record['seconds'], 1))), flush=True)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cache', type=Path, default=Path('artifacts/r2-cache/v1'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='random subset of charts, for a smoke run')
    a = p.parse_args(argv)
    fit(a.cache, a.out, a.workers, a.limit)


if __name__ == '__main__':
    main()
