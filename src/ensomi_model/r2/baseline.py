"""The frozen skeleton baseline b(S) of section difficulty (plan v4 section 7.1) and the proxy map g.

b(S) is a quadratic ridge (lambda 1, unpenalised intercept) on 23 head-time features of the
scope (A's ``interval_features``: duration, head rows, density, occupied span, gap moments and
quantiles, short-gap shares, 1 s and 4 s count statistics), refitted on fit_train cells of
every length the draw and the interface use (30, 60, 120 s at three phases and the whole
song) against Difficulty_nu. It reads head times in the scope and the scope bounds only, so
it is skeleton-only, span-local and mirror-invariant, and it is never fed to the model.

g is a linear ridge (lambda 1) from the committed difficulty proxies of a whole scope (mean
chord size / 4, same-lane repeat rate, LN share; the hold-length proxy, held-lane occupancy,
is kept out) to the residual Difficulty_nu - b(S), fitted on the same cells. The stage-2
relaxed-proxy term and its F3 calibration read it.

Fit on the mac after the relabel (writes ``labels/baseline-v1.npz`` and ``labels/baseline-v1.json``)::

    .venv/bin/python -m ensomi_model.r2.baseline --cache artifacts/r2-cache/v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .common import ContractError, band_of

FEATURE_NAMES = ['duration', 'rows', 'density', 'occupied_span', 'gap_mean', 'gap_sd', 'gap_q05', 'gap_q25',
                 'gap_q50', 'gap_q75', 'gap_q95', 'gap_lt50', 'gap_lt100', 'gap_lt200', 'gap_lt500',
                 'count1_mean', 'count1_sd', 'count1_q95', 'count1_max', 'count4_mean', 'count4_sd', 'count4_q95',
                 'count4_max']
PROXY_NAMES = ['chord_over_4', 'same_lane_repeat', 'ln_share']   # star_proxies indices 0, 1, 3
PROXY_INDEX = (0, 1, 3)
BASELINE_FILE = 'baseline-v1'
_PAIRS = np.triu_indices(len(FEATURE_NAMES))


def interval_features(head, a, b):
    h = np.asarray(head)[(np.asarray(head) >= a) & (np.asarray(head) < b)]
    gaps = np.diff(h) / 1000
    gaps = gaps if len(gaps) else np.array([0.])
    dur = (b - a) / 1000
    vals = [dur, len(h), len(h) / dur, (h[-1] - h[0]) / 1000 if len(h) > 1 else 0.,
            gaps.mean(), gaps.std(), *np.quantile(gaps, [.05, .25, .5, .75, .95]),
            *[(gaps < th).mean() for th in [.05, .1, .2, .5]]]
    for seconds in [1., 4.]:
        edges = np.arange(a, b + 1000 * seconds, 1000 * seconds)
        counts = np.histogram(h, bins=edges)[0] / seconds
        vals += [counts.mean(), counts.std(), np.quantile(counts, .95), counts.max()]
    return np.array(vals, dtype=float)


def design(Z):
    return np.column_stack([np.ones(len(Z)), Z, Z[:, _PAIRS[0]] * Z[:, _PAIRS[1]]])


class Baseline:
    def __init__(self, mu, sd, beta, g_beta, g_mu, g_sd, sha256=None):
        self.mu, self.sd, self.beta = np.asarray(mu), np.asarray(sd), np.asarray(beta)
        self.g_beta, self.g_mu, self.g_sd = np.asarray(g_beta), np.asarray(g_mu), np.asarray(g_sd)
        self.sha256 = sha256

    def predict(self, head_ms, a, b) -> float:
        z = (interval_features(head_ms, a, b) - self.mu) / self.sd
        return float(design(z[None])[0] @ self.beta)

    def proxy_residual(self, proxies):
        """g(proxies): proxies [..., 4] in ``features.star_proxies`` order (numpy or torch)."""
        x = proxies[..., list(PROXY_INDEX)]
        mu, sd, beta = self.g_mu, self.g_sd, self.g_beta
        try:
            import torch
            if isinstance(x, torch.Tensor):
                mu, sd, beta = (torch.as_tensor(v, dtype=x.dtype, device=x.device) for v in (mu, sd, beta))
                return beta[0] + (((x - mu) / sd) * beta[1:]).sum(-1)
        except ImportError:  # pragma: no cover
            pass
        return beta[0] + (((x - mu) / sd) * beta[1:]).sum(-1)

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = path.read_bytes()
        with np.load(path) as z:
            return cls(z['mu'], z['sd'], z['beta'], z['g_beta'], z['g_mu'], z['g_sd'], hashlib.sha256(raw).hexdigest())


def baseline_path(cache_root) -> Path:
    return Path(cache_root) / 'labels' / f'{BASELINE_FILE}.npz'


def load_baseline(cache_root, required=True):
    path = baseline_path(cache_root)
    if not path.exists():
        if required:
            raise ContractError(f'{path} is missing: fit the baseline after the relabel')
        return None
    return Baseline.load(path)


def _ridge_normal(rows_iter, p):
    A = np.zeros((p, p))
    c = np.zeros(p)
    for X, y in rows_iter:
        A += X.T @ X
        c += X.T @ y
    A += np.eye(p)
    A[0, 0] -= 1.0
    return np.linalg.solve(A, c)


def _group_bootstrap(groups, y, pred, rng, n=200):
    uniq = np.unique(groups)
    gi = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n):
        ix = np.concatenate([gi[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        sse = ((pred[ix] - y[ix]) ** 2)
        out.append([1 - sse.sum() / ((y[ix] - y[ix].mean()) ** 2).sum(), np.sqrt(sse.mean())])
    return np.std(out, axis=0, ddof=1).tolist()


def fit(cache_root: Path, chunk: int = 20000):
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree
    from .data import chart_from_cache
    from .cache import load_chart
    from .features import Interval, star_proxies
    from .labels import load_cells
    t0 = time.time()
    cells, complete, label_sha = load_cells(cache_root)
    if not complete:
        raise SystemExit('The v2 label file is missing or incomplete')
    table = pq.read_table(cache_root / 'index.parquet').to_pandas().sort_values('sha256')
    table = table[table.role.isin(['fit_train', 'fit_dev'])]
    X, y, role, group, length, sha_of, proxies, pos = [], [], [], [], [], [], [], []
    for i, r in enumerate(table.itertuples()):
        chart = chart_from_cache(load_chart(cache_root / 'charts' / r.file))
        for key, cs in cells.get(r.sha256, {}).items():
            for j, (a, b, v) in enumerate(sorted(cs)):
                X.append(interval_features(chart.head_ms, a, b))
                y.append(v)
                role.append(r.role)
                group.append(r.group_id)
                length.append('whole' if key == 'whole' else str(key[0]))
                sha_of.append(r.sha256)
                pos.append((str(key), j))
                proxies.append(star_proxies(chart, Interval(1, a, b, v), chart.K))
        if i % 2000 == 0:
            print(f'[{time.time() - t0:7.1f}s] features {i}/{len(table)} cells {len(y)}', flush=True)
    X, y, P = np.array(X), np.array(y), np.array(proxies)
    role, group, length = np.array(role), np.array(group), np.array(length)
    train, dev = role == 'fit_train', role == 'fit_dev'
    mu, sd = X[train].mean(0), np.maximum(X[train].std(0), 1e-8)
    Z = (X - mu) / sd
    tr = np.flatnonzero(train)
    p = design(Z[:1]).shape[1]
    beta = _ridge_normal(((design(Z[tr[s:s + chunk]]), y[tr[s:s + chunk]]) for s in range(0, len(tr), chunk)), p)
    pred = np.concatenate([design(Z[s:s + chunk]) @ beta for s in range(0, len(Z), chunk)])
    resid = y - pred
    tree = cKDTree(Z[train])
    dist, idx = tree.query(Z[dev], k=32, workers=4)
    w = 1 / np.maximum(dist, 1e-6)
    knn = (w * y[train][idx]).sum(1) / w.sum(1)
    Q = P[:, list(PROXY_INDEX)]
    g_mu, g_sd = Q[train].mean(0), np.maximum(Q[train].std(0), 1e-8)
    G = np.column_stack([np.ones(len(Q)), (Q - g_mu) / g_sd])
    g_beta = _ridge_normal([(G[train], resid[train])], G.shape[1])
    g_pred = G @ g_beta
    rng = np.random.default_rng(41005)

    def metrics(mask, pr, target):
        sse = (pr - target[mask]) ** 2
        return dict(n=int(mask.sum()), r2=float(1 - sse.sum() / ((target[mask] - target[mask].mean()) ** 2).sum()),
                    rmse=float(np.sqrt(sse.mean())), mae=float(np.abs(pr - target[mask]).mean()),
                    bootstrap_se_r2_rmse=_group_bootstrap(group[mask], target[mask], pr, rng))

    report = dict(ridge_dev=metrics(dev, pred[dev], y), knn32_dev=metrics(dev, knn, y),
                  ridge_dev_by_length={L: metrics(dev & (length == L), pred[dev & (length == L)], y)
                                       for L in sorted(set(length))},
                  g_dev=metrics(dev, g_pred[dev], resid))
    q = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    report['residual_quantiles'] = {split: dict(zip(map(str, q), np.quantile(resid[m], q).tolist()),
                                                sd=float(resid[m].std(ddof=1)))
                                    for split, m in (('fit_train', train), ('fit_dev', dev))}
    bands = np.array([band_of(v) for v in y])
    report['residual_sd_by_band_dev'] = {str(b): dict(n=int((dev & (bands == b)).sum()),
                                                      sd=float(resid[dev & (bands == b)].std(ddof=1)))
                                         for b in sorted(set(bands[dev]))}
    pairs = {}
    order = {}
    for i, (s, (key, j)) in enumerate(zip(sha_of, pos)):
        order[(s, key, j)] = i
    for (s, key, j), i in order.items():
        nxt = order.get((s, key, j + 1))
        if nxt is not None and key != 'whole':
            pairs.setdefault(length[i], []).append((resid[i], resid[nxt]))
    corr = {L: dict(pairs=len(v), r=float(np.corrcoef(np.array(v).T)[0, 1])) for L, v in pairs.items() if len(v) > 2}
    allp = np.array([x for v in pairs.values() for x in v])
    corr['pooled'] = dict(pairs=len(allp), r=float(np.corrcoef(allp.T)[0, 1]))
    report['adjacent_cell_residual_correlation'] = corr
    report['star_informative_rule'] = dict(threshold=0.5, fires=bool(corr['pooled']['r'] > 0.5))
    out = cache_root / 'labels' / f'{BASELINE_FILE}.npz'
    np.savez(out, mu=mu, sd=sd, beta=beta, g_beta=g_beta, g_mu=g_mu, g_sd=g_sd,
             feature_names=np.array(FEATURE_NAMES), proxy_names=np.array(PROXY_NAMES))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    summary = dict(file=out.name, sha256=sha, labels_sha256=label_sha, features=FEATURE_NAMES, proxies=PROXY_NAMES,
                   ridge_lambda=1, knn=32, cells_train=int(train.sum()), cells_dev=int(dev.sum()),
                   wall_s=time.time() - t0, **report)
    (cache_root / 'labels' / f'{BASELINE_FILE}.json').write_text(json.dumps(summary, indent=1))
    from .receipts import write_receipt
    write_receipt(cache_root / 'labels' / f'receipt-{BASELINE_FILE}.json', config=dict(cache=str(cache_root)),
                  inputs=dict(labels_sha256=label_sha), outputs=dict(sha256=sha), seed=41005, device='cpu',
                  ens_job=os.environ.get('ENS_JOB_ID'))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('features',)}, indent=1), flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='artifacts/r2-cache/v1')
    a = p.parse_args(argv)
    fit(Path(a.cache))


if __name__ == '__main__':
    sys.exit(main())
