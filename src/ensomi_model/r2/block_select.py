"""Best-of-four 64-row decoding with a fitted density or source-envelope score."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import load_chart
from .data import chart_from_cache
from .features import Chart
from .sampling import continue_chart

PHI = ('nh', 'held', 'c3', 'pent')
BLOCK = 64


def phi(chart, start, stop):
    """Heads, held share, triple-head rate, and nonempty-pattern entropy."""
    d = chart.derived()
    n = stop - start
    if not n:
        return np.zeros(4)
    counts = (d.cum_patterns[stop] - d.cum_patterns[start])[1:]
    p = counts[counts > 0] / counts.sum()
    return np.array([(d.cum_heads[stop] - d.cum_heads[start]) / n,
                     (d.cum_held[stop] - d.cum_held[start]) / (4 * n),
                     (d.cum_c3[stop] - d.cum_c3[start]) / n, -(p * np.log2(p)).sum()])


def predictors(chart, start, stop):
    """Running Phi, prefix size and five heads-only block timing coordinates."""
    head = chart.head_ms[start:stop]
    prev = chart.head_ms[np.maximum(np.arange(start, stop) - 1, 0)]
    dt = head - prev
    # At BOS there is no preceding head gap; use the first within-block gap.
    if start == 0:
        dt = dt[1:]
    if not len(dt):
        dt = np.array([chart.song_ms - head[-1]])
    logdt = np.log1p(dt)
    beat = chart.grid.beat(head)
    beat_step = float((beat[-1] - beat[0]) / max(len(beat) - 1, 1))
    return np.r_[phi(chart, 0, start), np.log1p(start), logdt.mean(), np.median(logdt), logdt.std(),
                 beat_step, chart.grid.bpm(head).mean() / 120.0]


@dataclass
class PhiScorer:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    precision: np.ndarray
    logdet: float

    def __call__(self, chart, start, stop, band):
        stop = min(stop, chart.K)
        if stop <= start:
            return 0.0
        x = np.r_[1.0, (predictors(chart, start, stop) - self.mean) / self.scale]
        residual = phi(chart, start, stop) - x @ self.coefficient
        return float(-0.5 * (residual @ self.precision @ residual + self.logdet + 4 * np.log(2 * np.pi)))


def fit_phi(cache_root, output_path, *, ridge=1.0, limit=None):
    """Fit Gaussian block residuals after ridge regression on fit_train only."""
    cache_root = Path(cache_root)
    table = pd.read_parquet(cache_root / 'index.parquet')
    table = table[table.role == 'fit_train'].sort_values('sha256')
    if limit is not None:
        table = table.iloc[:limit]
    xs, ys = [], []
    for row in table.itertuples():
        chart = chart_from_cache(load_chart(cache_root / 'charts' / row.file))
        for start in range(0, chart.K - BLOCK + 1, BLOCK):
            xs.append(predictors(chart, start, start + BLOCK))
            ys.append(phi(chart, start, start + BLOCK))
    x, y = np.asarray(xs), np.asarray(ys)
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    design = np.c_[np.ones(len(x)), (x - mean) / scale]
    penalty = np.diag(np.r_[0.0, np.full(x.shape[1], ridge)])
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    residual = y - design @ coefficient
    covariance = residual.T @ residual / len(residual) + np.eye(4) * ridge / len(residual)
    data = dict(role='fit_train', charts=len(table), blocks=len(x), block_rows=BLOCK, statistics=PHI,
                ridge=ridge, predictor_names=['running_' + n for n in PHI] +
                ['log_prefix_rows', 'mean_log_gap_ms', 'median_log_gap_ms', 'sd_log_gap_ms',
                 'mean_gap_beats', 'mean_bpm_over_120'],
                mean=mean.tolist(), scale=scale.tolist(), coefficient=coefficient.tolist(),
                precision=np.linalg.inv(covariance).tolist(), logdet=float(np.linalg.slogdet(covariance)[1]))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')
    return data


def load_scorer(kind, path, statistics=None):
    if kind == 'phi':
        data = json.loads(Path(path).read_text())
        return PhiScorer(*(np.asarray(data[k]) for k in ('mean', 'scale', 'coefficient', 'precision')),
                         data['logdet'])
    from .collapse_eval import envelope_score, load_reference, window_statistics
    reference = load_reference(path)

    def score(chart, start, stop, band):
        stop = min(stop, chart.K)
        if stop <= start:
            return 0.0
        # The source envelope is fitted to 64-row windows, including at a short tail.
        return envelope_score(window_statistics(chart, max(0, stop - BLOCK), stop), band, reference, statistics)
    return score


def candidate_seeds(seed, block, count=4):
    return [int(np.random.SeedSequence([seed, block, i]).generate_state(1, dtype=np.uint64)[0])
            for i in range(count)]


def select_blocks(model, chart, scorer, *, band, seed=954, stop=None, prefix_actions=None, prefix_gap=None,
                  candidates=4):
    """Return decisions and per-block scores; candidate RNGs depend only on seed/block/index.

    Each candidate branches from identical committed decisions and the immutable
    temporal cache. Only the winning branch supplies the next block's history.
    """
    stop = chart.K + 1 if stop is None else stop
    actions = np.empty((0, 4), dtype=np.int64) if prefix_actions is None else prefix_actions.copy()
    gap = np.empty((0, 4), dtype=np.float64) if prefix_gap is None else prefix_gap.copy()
    history = None
    blocks = []
    while len(actions) < stop:
        start = len(actions)
        end = min((start // BLOCK + 1) * BLOCK, stop)
        seeds = candidate_seeds(seed, start // BLOCK, candidates)
        branches, scores = [], []
        for candidate_seed in seeds:
            cached = {}
            a, g = continue_chart(model, chart.head_ms, chart.song_ms, chart.grid,
                                  prefix_actions=actions, prefix_gap=gap, seed=candidate_seed, stop=end,
                                  _history_cache=history, _history_out=cached)
            proposed = Chart(chart.head_ms, chart.song_ms, chart.grid, a, g)
            scores.append(float(scorer(proposed, start, end, band)))
            branches.append((a, g, (cached['cache'], cached['marks'])))
        best = int(np.argmax(scores))
        actions, gap, history = branches[best]
        blocks.append(dict(start=start, stop=end, seeds=seeds, scores=scores, selected=best))
    return actions, gap, dict(candidates=candidates, block_rows=BLOCK, blocks=blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fit the d0 conditional block-density scorer')
    parser.add_argument('--cache', default='artifacts/r2-cache/v1')
    parser.add_argument('--out', default='artifacts/r2-bakeoff-20261007/phi.json')
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--limit', type=int, help='Bound a smoke fit to this many source charts')
    args = parser.parse_args(argv)
    if args.ridge <= 0:
        parser.error('ridge must be positive')
    fit_phi(args.cache, args.out, ridge=args.ridge, limit=args.limit)


if __name__ == '__main__':
    main()
