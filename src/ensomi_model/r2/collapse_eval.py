"""Source-window collapse measures, paired chart comparisons and generation guards.

Windows contain 64 head rows. Degeneracy uses low pent/ng4 and high
rep1/loop/c4/fjack/bus4/lmax/hmax/hlock/lock relative to fit_dev 1/99 percentiles.
Repeated runs are averaged within chart before across-chart statistics. Undefined
coordinates remain null; G requires every selected measure in all four bands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

W = 64
BANDS = (2, 3, 4, 5)
MEASURES = tuple(f'm{i}' for i in range(1, 8))
HIGH = ('rep1', 'loop', 'c4', 'fjack', 'bus4', 'lmax', 'hmax', 'hlock', 'lock')
LOW = ('pent', 'ng4')
STATISTICS = ('pent', 'ng4', 'rep1', 'loop', 'nh', 'c3', 'c4', 'jack', 'fjack',
              'bus4', 'held', 'lmax', 'hmax', 'lock', 'hlock', 'single')
COHERENCE = ('nh', 'c3', 'held', 'pent')
SPREAD = ('nh', 'c3', 'c4', 'held', 'pent', 'lnlen')
DEFINITIONS = dict(
    m1='Pooled degenerate-window count / window count; any directional envelope exit.',
    m2='P(next window exits | current window exits); transitions stay within runs.',
    m3='Mean 1-correlation(first third,last third) across runs for nh,c3,held,pent.',
    m4='Mean V(8)/V(1) for held,nh; V(l)=between-chart variance of block[2+l]-mean(block[0:2]).',
    m5='Mean abs(1-SD(run)/SD(source)) for nh,c3,c4,held,pent,lnlen; paired charts per coordinate.',
    m6='Mean absolute paired band-mean offset for nh,held.',
    m7='Absolute paired band-mean shuffle-corrected adjacent-MI difference, bits.',
    lnlen='Median ln(max(closed hold length in ms,1)); at least 10 holds born in the evaluated span.',
    aggregation='Each run is one observation (m3-m5 across runs, others pooled or run means); the bootstrap resamples charts with all their runs. Continuations use generated suffix and matched source.',
    adjacency='128 rows, stride64,20 shuffles; seed=20261007 XOR first8 SHA hex XOR absolute start.',
    tail='Row-weighted exits in last10% of evaluated rows; final incomplete block uses trailing64-row window.',
    legal_export='Replay sampled decisions; cap-only export closes open holds midway to the next original head/EOS. Metrics exclude this closure.',
    comparison='Match chart,mode,seed,start,stop with b0; pairwise differences also use common runs. Bootstrap charts within each band, keeping all matched runs together.',
    G='Mean |m_system-m_source|/|m_b0-m_source| over selected measures and bands 2-5, without terms where '
      '|m_b0-m_source| < source chart-bootstrap SD; G_core also leaves out m1 and m6. Lower is better.',
)


def decode(actions):
    """Return held-before, heads and LN heads without replaying timing."""
    a = np.asarray(actions, dtype=np.int64)
    held = np.zeros(a.shape, dtype=bool)
    idx = np.arange(len(a))
    for lane in range(4):
        c = a[:, lane]
        reset = (c == 1) | (c == 3) | (c == 4)
        toggles = np.cumsum(c == 2)
        last = np.maximum.accumulate(np.where(reset, idx, -1))
        base = np.where(last >= 0, (c[np.maximum(last, 0)] == 4), 0)
        tbase = np.where(last >= 0, toggles[np.maximum(last, 0)], 0)
        held[1:, lane] = ((base + toggles - tbase) % 2)[:-1]
    return held, np.where(held, a >= 3, (a == 1) | (a == 2)), np.where(held, a == 4, a == 2)


def longest_lock(mask):
    idx = np.arange(len(mask))[:, None]
    last_zero = np.maximum.accumulate(np.where(mask, -1, idx), axis=0)
    return np.where(mask, idx - last_zero, 0).max(axis=1)


def entropy_counts(counts):
    counts = np.asarray(counts, float)
    p = counts[counts > 0] / max(counts.sum(), 1)
    return float(-(p * np.log2(p)).sum())


def _arrays(head_ms, actions):
    held, head, _ = decode(np.asarray(actions)[:len(head_ms)])
    hp = head @ np.array([1, 2, 4, 8])
    prev = np.vstack([np.zeros((1, 4), bool), head[:-1]])
    return held, head, hp, head & prev, longest_lock(held | head), longest_lock(head), np.r_[np.inf, np.diff(head_ms)]


def _window(head_ms, arrays, start, stop):
    held, head, hp, jack, lock, hlock, gaps = arrays
    sl = slice(start, stop)
    h = hp[sl]
    nh = head[sl].sum(1)
    counts = np.bincount(h, minlength=16)[1:]
    grams = h[:-3] * 4096 + h[1:-2] * 256 + h[2:-1] * 16 + h[3:]
    loops = [(hp[max(start, p):stop] == hp[max(start, p) - p:stop - p]).mean()
             for p in range(2, min(17, stop))]
    heads = max(int(nh.sum()), 1)
    lane = head[sl].sum(0)
    dt = gaps[sl]
    return dict(w0=start, frac=(start + (stop - start) / 2) / len(head_ms),
                pent=entropy_counts(counts), ng4=len(np.unique(grams)) / len(grams) if len(grams) else None,
                rep1=float((h[1:] == h[:-1]).mean()) if len(h) > 1 else None,
                loop=float(max(loops)) if loops else None, nh=float(nh.mean()),
                c3=float((nh >= 3).mean()), c4=float((nh == 4).mean()), single=float((nh == 1).mean()),
                jack=float(jack[sl].sum() / heads),
                fjack=float((jack[sl] & (dt < 100)[:, None]).sum() / heads),
                bus4=float(((held[sl] | head[sl]).sum(1) == 4).mean()), held=float(held[sl].mean()),
                lmax=float(lane.max() / heads), hmax=float(max(lane[:2].sum(), lane[2:].sum()) / heads),
                lock=int(lock[sl].max()), hlock=int(hlock[sl].max()), empty=int((nh == 0).sum()),
                dtmed=float(np.median(dt[np.isfinite(dt)])) if np.isfinite(dt).any() else None)


def seq_stats(head_ms, acts, marg=None, start=0):
    """Return complete nonoverlapping 64-row windows and whole-span locks."""
    K = len(head_ms)
    arrays = _arrays(head_ms, acts)
    rows = []
    for w0 in range(start, K - W + 1, W):
        row = _window(head_ms, arrays, w0, w0 + W)
        if marg is not None:
            m = np.asarray(marg, dtype=float)[w0:w0 + W]
            row['ent'] = float(-(np.where(m > 0, m * np.log(np.clip(m, 1e-12, 1)), 0)).sum((1, 2)).mean())
        else:
            row['ent'] = None
        rows.append(row)
    _, head, _, _, lock, hlock, _ = arrays
    return rows, dict(K=K, maxlock=int(lock[start:].max()) if K > start else 0,
                     maxhlock=int(hlock[start:].max()) if K > start else 0,
                     empty=int((head[start:].sum(1) == 0).sum()))


def window_statistics(chart, start=0, stop=None):
    """One block's measures, retaining prior history; ng4 is null below four rows."""
    stop = min(chart.n, chart.K) if stop is None else stop
    return _window(chart.head_ms[:stop], _arrays(chart.head_ms[:stop], chart.actions), start, stop)


def fit_envelope(windows):
    """Fit per-band linear 1/99-percentile bounds from source-window dictionaries."""
    return {str(b): {key: np.quantile([r[key] for r in windows if r['band'] == b], [.01, .99]).tolist()
                     for key in STATISTICS}
            for b in BANDS if any(r['band'] == b for r in windows)}


def build_reference(cache_root, output_path, *, limit=None):
    from .data import Corpus
    corpus = Corpus(cache_root, 'fit_dev', star_conditions=False, capacity=1)
    table = corpus.table.sort_values('sha256')
    if limit is not None:
        table = table.head(limit)
    windows = []
    for r in table.itertuples():
        chart = corpus.chart(r.sha256)
        rows, _ = seq_stats(chart.head_ms, chart.actions)
        windows.extend(dict(row, band=int(r.band)) for row in rows)
    result = dict(role='fit_dev', charts=len(table), windows=len(windows), limit=limit,
                  complete=limit is None, quantiles=[.01, .99], window_rows=W,
                  exits=dict(high=list(HIGH), low=list(LOW)), bands=fit_envelope(windows))
    _write_json(output_path, result)
    return result


def load_reference(path):
    return json.loads(Path(path).read_text())


def exits(stats, band, reference):
    envelope = reference['bands'][str(band)]
    return {key: bool(stats[key] > envelope[key][1]) for key in HIGH} | {
        key: bool(stats[key] < envelope[key][0]) for key in LOW}


def envelope_score(stats, band, reference, statistics):
    """Negative sum of distances outside each selected statistic's envelope, in envelope widths."""
    bounds = reference['bands'][str(band)]
    return -float(sum(max(bounds[k][0] - stats[k], 0, stats[k] - bounds[k][1]) / ((bounds[k][1] - bounds[k][0]) or 1.0)
                      for k in statistics))


def joint_metrics(x):
    pairs = np.bincount(x[:-1] * 16 + x[1:], minlength=256).reshape(16, 16)
    previous, following = entropy_counts(pairs.sum(1)), entropy_counts(pairs.sum(0))
    joint = entropy_counts(pairs.ravel())
    return np.array([joint - previous, previous + following - joint])


def adjacency(head_ms, actions, sha256, start=0):
    """Shuffle-corrected head-mask dependence in overlapping 128-row spans."""
    _, head, _ = decode(np.asarray(actions)[:len(head_ms)])
    masks = head @ np.array([1, 2, 4, 8])
    names = ('conditional_entropy', 'adjacent_MI', 'shuffle_conditional_entropy',
             'shuffle_adjacent_MI', 'conditional_entropy_minus_shuffle', 'adjacent_MI_minus_shuffle')
    rows = []
    for w0 in range(start, len(masks) - 127, W):
        x = masks[w0:w0 + 128]
        actual = joint_metrics(x)
        rng = np.random.default_rng(20261007 ^ int(sha256[:8], 16) ^ w0)
        null = np.mean([joint_metrics(x[rng.permutation(128)]) for _ in range(20)], axis=0)
        rows.append(dict(zip(names, np.r_[actual, null, actual - null]),
                         position=min(2, int((w0 - start + 63.5) * 3 / (len(masks) - start)))))
    return {name: {key: _mean([r[key] for r in rows if position is None or r['position'] == position])
                   for key in names}
            for name, position in [('all', None), ('early', 0), ('middle', 1), ('late', 2)]}


def _mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _complete_mean(values):
    return _mean(values) if values and all(v is not None for v in values) else None


def _ratio(a, b):
    return a / b if a is not None and b is not None and b != 0 else None


def _ln_length(chart, start, stop):
    # Only releases already committed in the measured span count; open holds are censored.
    held = {}
    lengths = []
    for k, row in enumerate(chart.actions[:min(stop + int(stop == chart.K), chart.n)]):
        for lane, code in enumerate(row):
            if lane in held and code:
                birth = held.pop(lane)
                end = chart.time(k) if code == 1 else chart.gap[k, lane]
                if birth >= start:
                    lengths.append(end - chart.head_ms[birth])
                if code == 4:
                    held[lane] = k
            elif lane not in held and code == 2:
                held[lane] = k
    return float(np.median(np.log(np.maximum(lengths, 1)))) if len(lengths) >= 10 else None


def summarize_sequence(chart, sha256, band, reference, *, start=0, stop=None):
    stop = min(chart.K, chart.n) if stop is None else stop
    head_ms = chart.head_ms[:stop]
    windows, whole = seq_stats(head_ms, chart.actions, start=start)
    window_exits = [exits(w, band, reference) for w in windows]
    flags = [any(flags.values()) for flags in window_exits]
    thirds = [{key: _mean([w[key] for w in windows
                          if min(2, int((w['w0'] - start + W / 2) * 3 / (stop - start))) == third])
               for key in COHERENCE} for third in range(3)]
    means = {key: _mean([w[key] for w in windows]) for key in STATISTICS}
    means['lnlen'] = _ln_length(chart, start, stop)
    tail_start = start + .9 * (stop - start)
    tail_n = tail_exits = 0.0
    for w, flag in zip(windows, flags):
        weight = max(0, w['w0'] + W - max(w['w0'], tail_start))
        tail_n += weight
        tail_exits += weight * flag
    covered = start + len(windows) * W
    if stop > covered and stop - start >= W:
        final = window_statistics(chart, stop - W, stop)
        weight = stop - max(covered, tail_start)
        tail_n += weight
        tail_exits += weight * any(exits(final, band, reference).values())
    drift = {key: {str(lag): (windows[2 + lag][key] - np.mean([w[key] for w in windows[:2]])
                              if len(windows) > 2 + lag else None)
                   for lag in (1, 8)} for key in ('nh', 'held')}
    return dict(means=means, thirds=thirds, drift=drift,
                exit_rates={key: _mean([row[key] for row in window_exits]) for key in HIGH + LOW},
                degenerate=_mean(flags), degenerate_n=sum(flags), entry_n=sum(not f for f in flags[:-1]),
                entry=sum(not a and b for a, b in zip(flags, flags[1:])),
                stay_n=sum(flags[:-1]), stay=sum(a and b for a, b in zip(flags, flags[1:])),
                windows=len(windows), tail_exits=_ratio(tail_exits, tail_n), tail_rows=tail_n,
                adjacency=adjacency(head_ms, chart.actions, sha256, start=start), **whole)


def _chart_values(records, field):
    keys = sorted({r['sha256'] for r in records})
    groups = [[r[field] for r in records if r['sha256'] == sha] for sha in keys]
    def average(path):
        def at(row):
            for key in path:
                row = row[key]
            return row
        return [_mean([at(row) for row in group]) for group in groups]
    return keys, average


def _band_data(records, field):
    """One value per run (each run is an observation); runs[c] indexes chart c's runs for the chart bootstrap."""
    records = sorted(records, key=_run_key)
    keys = sorted({r['sha256'] for r in records})
    runs = np.array([[i for i, r in enumerate(records) if r['sha256'] == k] for k in keys], dtype=int)
    runs = runs.reshape(len(keys), -1) if keys else np.empty((0, 0), int)

    def at(row, path):
        for key in path:
            row = row[key]
        return row
    paths = ([('means', k) for k in SPREAD]
             + [('thirds', t, k) for t in (0, 2) for k in COHERENCE]
             + [('drift', k, str(lag)) for k in ('nh', 'held') for lag in (1, 8)]
             + [(k,) for k in ('degenerate_n', 'windows', 'entry', 'entry_n', 'stay', 'stay_n')]
             + [('adjacency', 'all', 'adjacent_MI_minus_shuffle')])
    return keys, runs, {path: np.array([at(r[field], path) for r in records], dtype=float) for path in paths}


def _divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def _array_mean(values):
    return _divide(np.nansum(values, axis=-1), np.isfinite(values).sum(axis=-1))


def _paired_center(x, y):
    paired = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(paired, x, np.nan), np.where(paired, y, np.nan)
    return x - _array_mean(x)[..., None], y - _array_mean(y)[..., None]


def _metric_arrays(data, source):
    """Chart axis is last; a leading axis evaluates every bootstrap draw at once."""
    coherence = {}
    for key in COHERENCE:
        x, y = _paired_center(data['thirds', 0, key], data['thirds', 2, key])
        coherence[key] = _divide(np.nansum(x * y, axis=-1),
                                 np.sqrt(np.nansum(x * x, axis=-1) * np.nansum(y * y, axis=-1)))
    drift = {}
    for key in ('held', 'nh'):
        small, large = _paired_center(data['drift', key, '1'], data['drift', key, '8'])
        drift[key] = _divide(np.nansum(large * large, axis=-1), np.nansum(small * small, axis=-1))
    spread = {}
    for key in SPREAD:
        x, y = _paired_center(data['means', key], source['means', key])
        spread[key] = np.sqrt(_divide(np.nansum(x * x, axis=-1), np.nansum(y * y, axis=-1)))
    offsets = {key: _array_mean(data['means', key] - source['means', key]) for key in ('nh', 'held')}
    path = ('adjacency', 'all', 'adjacent_MI_minus_shuffle')
    adjacency_gap = _array_mean(source[path] - data[path])
    mean_components = lambda values: np.stack(list(values), axis=-1).mean(axis=-1)
    metrics = np.stack([
        _divide(_array_mean(data['degenerate_n',]), _array_mean(data['windows',])),
        _divide(_array_mean(data['stay',]), _array_mean(data['stay_n',])),
        mean_components(1 - c for c in coherence.values()), mean_components(drift.values()),
        mean_components(abs(1 - c) for c in spread.values()), mean_components(abs(c) for c in offsets.values()),
        abs(adjacency_gap)], axis=-1)
    return metrics, dict(entry_rate=_divide(_array_mean(data['entry',]), _array_mean(data['entry_n',])),
                         thirds_correlation=coherence, variance_ratios=drift, sd_ratios=spread,
                         paired_offsets=offsets, source_minus_run_adjacent_MI=adjacency_gap)


def _finite_json(value):
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    return float(value) if np.isfinite(value) else None


def band_metrics(records, *, field='sample', chart_ids=None):
    """m1..m7 plus components; chart_ids supports repeated paired bootstrap draws."""
    keys, runs, data = _band_data(records, field)
    _, _, source = _band_data(records, 'source')
    indices = runs[list(range(len(keys))) if chart_ids is None else [keys.index(k) for k in chart_ids]].ravel()
    metrics, details = _metric_arrays({k: v[indices] for k, v in data.items()},
                                      {k: v[indices] for k, v in source.items()})
    charts = len(keys) if chart_ids is None else len(chart_ids)
    return dict(metrics=_finite_json(dict(zip(MEASURES, metrics))), charts=charts, runs=len(indices),
                **_finite_json(details))


def aggregate(records, *, field='sample', selected=None):
    return {str(b): band_metrics([r for r in records if r['band'] == b], field=field,
                                 chart_ids=selected.get(str(b)) if selected else None)
            for b in BANDS}


def legal_export(chart):
    """Check complete export, or prefix replay plus deterministic export-only closure."""
    from .export import export_chart, minimal_header
    from .state import replay_decisions
    try:
        state, _ = replay_decisions(chart.head_ms, chart.song_ms, chart.actions, chart.gap)
        partial = not state.finalized
        if partial:
            K = chart.n
            song_ms = chart.time(K)
            actions = np.vstack([chart.actions, np.where(state.held, 2, 0)])
            release = (chart.head_ms[K - 1] + song_ms) / 2
            gap = np.vstack([chart.gap, np.where(state.held, release, np.nan)])
            head_ms = chart.head_ms[:K]
        else:
            head_ms, song_ms, actions, gap = chart.head_ms, chart.song_ms, chart.actions, chart.gap
        segments = np.column_stack([chart.grid.offsets, chart.grid.lengths, chart.grid.meters])
        objects, _ = export_chart(head_ms, song_ms, actions, gap, None, minimal_header(segments))
        matches = np.array_equal(np.unique([o.start_time_ms for o in objects]), head_ms)
        closure = ('midpoint-to-eos' if chart.n == chart.K else 'midpoint-to-next-head') if partial else None
        return dict(pass_=bool(matches), partial=partial, closure=closure)
    except Exception as exc:  # external generated decisions can be malformed
        return dict(pass_=False, error=f'{type(exc).__name__}: {exc}')


def teacher_forced_nll(checkpoint, cache_root, *, windows=64):
    """Reuse Evaluator.natural and its fixed fit_dev draw, with theta unknown."""
    import torch
    from .data import Corpus, build_natural_manifest
    from .evaluate import Evaluator
    from .model import R2Config, R2Model
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = R2Model(R2Config(**data['model_config'])).eval()
    model.load_state_dict(data['model'])
    corpus = Corpus(cache_root, 'fit_dev', star_conditions=False)
    manifest = dict(natural=dict(windows=build_natural_manifest(corpus, windows=windows)))
    ev = Evaluator(model, SimpleNamespace(cache=cache_root, seed_validation=954), manifest, star=False,
                   conditions=False)
    with torch.inference_mode():
        result = ev.natural('unknown')
    return dict(result, theta_mode='unknown', requested_windows=windows,
                scope='fixed natural fit_dev windows; not exhaustive corpus')


def guards(records, reference, nll=None, baseline_nll=None):
    per_band = {}
    for band in BANDS:
        group = [r for r in records if r['band'] == band]
        if not group:
            per_band[str(band)] = None
            continue
        _, vals = _chart_values(group, 'sample')
        _, src = _chart_values(group, 'source')
        envelope = reference['bands'][str(band)]
        means = {k: _mean(vals(('means', k))) for k in ('single', 'ng4')}
        inside = {k: envelope[k][0] <= value <= envelope[k][1] if value is not None else None
                  for k, value in means.items()}
        tail, tail_source = _mean(vals(('tail_exits',))), _mean(src(('tail_exits',)))
        per_band[str(band)] = dict(envelope=inside, means=means,
                                  bounds={k: envelope[k] for k in means},
                                  tail_exits=tail, source_tail_exits=tail_source,
                                  tail_pass=(tail <= 1.5 * tail_source)
                                  if tail is not None and tail_source is not None else None)
    return dict(legal_export=all(r['legal']['pass_'] for r in records) if records else None,
                no_head_lock_30=all(r['sample']['maxhlock'] < 30 for r in records) if records else None,
                nll_per_decision=nll, baseline_nll_per_decision=baseline_nll,
                nll_pass=nll <= baseline_nll + .02 if nll is not None and baseline_nll is not None else None,
                bands=per_band)


def _guard_all(values):
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def evaluate_system(panel_path, runs_dir, system, cache_root, output_path, *, reference_path,
                    checkpoint=None, measures=MEASURES, nll_windows=64):
    from .data import Corpus
    from .features import Chart
    _check_measures(measures)
    panel = json.loads(Path(panel_path).read_text())
    reference = load_reference(reference_path)
    corpus = Corpus(cache_root, 'fit_dev', star_conditions=False)
    records, sources = [], {}
    for path in sorted(Path(runs_dir).glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            sha = str(z['sha256'].item())
            source = corpus.chart(sha)
            actions, gap = z['actions'], z['gap_release_ms']
            start, seed = int(z['start']), int(z['seed'])
            band, mode = int(z['band']), str(z['mode'].item())
            chart = Chart(z['head_ms'], float(z['song_ms']), source.grid, actions, gap)
        stop = min(chart.n, chart.K)
        source_key = (sha, start, chart.n)
        if source_key not in sources:
            paired_source = source.with_decisions(source.actions[:chart.n], source.gap[:chart.n])
            sources[source_key] = summarize_sequence(paired_source, sha, band, reference, start=start, stop=stop)
        records.append(dict(sha256=sha, band=band, mode=mode, seed=seed, start=start, stop=stop,
                            sample=summarize_sequence(chart, sha, band, reference, start=start, stop=stop),
                            source=sources[source_key],
                            legal=legal_export(chart)))
    nll = teacher_forced_nll(checkpoint, cache_root, windows=nll_windows) if checkpoint and nll_windows else None
    result = dict(system=system, panel=str(panel_path), panel_seed=panel.get('seed'),
                  reference=str(reference_path), reference_complete=reference['complete'],
                  checkpoint=str(checkpoint) if checkpoint else None, measures=list(measures), definitions=DEFINITIONS,
                  bands=aggregate(records), source_bands=aggregate(records, field='source'), records=records,
                  nll=nll, guards=guards(records, reference, nll=nll['nll_per_decision'] if nll else None))
    _write_json(output_path, result)
    return result


def _check_measures(measures):
    if not measures or set(measures) - set(MEASURES) or len(set(measures)) != len(measures):
        raise ValueError(f'Measures must be a nonempty unique subset of {MEASURES}')


BAND_RELATIVE = ('m1', 'm6')  # envelope- and band-offset-based terms, left out of G_core


def _column_sd(values):
    """SD over the leading (bootstrap) axis, ignoring non-finite draws."""
    finite = np.isfinite(values)
    mean = _divide(np.where(finite, values, 0).sum(0), finite.sum(0))
    return np.sqrt(_divide(np.where(finite, (values - mean) ** 2, 0).sum(0), finite.sum(0)))


def _dropped(base_gap, sd):
    """A term leaves G when B0's distance to the source is below the source's own chart-bootstrap SD."""
    return np.isfinite(base_gap) & np.isfinite(sd) & (base_gap < sd)


def source_sd(records, *, bootstrap=200, seed=954):
    """Per band, chart-bootstrap SD of the source's m1..m7 on these runs (NaN for an empty band)."""
    rng = np.random.default_rng(seed)
    out = {}
    for band in BANDS:
        keys, runs, source = _band_data([r for r in records if r['band'] == band], 'source')
        if not keys:
            out[str(band)] = dict.fromkeys(MEASURES, np.nan)
            continue
        indices = runs[rng.integers(len(keys), size=(bootstrap, len(keys)))].reshape(bootstrap, runs.size)
        sampled = {k: v[indices] for k, v in source.items()}
        out[str(band)] = dict(zip(MEASURES, _column_sd(_metric_arrays(sampled, sampled)[0])))
    return out


def gap_score(system, source, baseline, measures, sd=None):
    """Two-sided g = |m_sys - m_src| / |m_b0 - m_src|; G averages kept terms, G_core also leaves out m1, m6.

    ``gaps`` keeps the signed ratios for reading. With ``sd`` (``source_sd``), a term is dropped when
    |m_b0 - m_src| is below the source's chart-bootstrap SD; undefined terms stay and make G undefined.
    """
    gaps, kept, dropped = {}, [], []
    for b in map(str, BANDS):
        gaps[b] = {}
        for m in measures:
            value, src, base = (x[b]['metrics'][m] for x in (system, source, baseline))
            gaps[b][m] = _ratio(_difference(value, src), _difference(base, src))
            base_gap = abs(base - src) if base is not None and src is not None else np.nan
            if sd is not None and _dropped(base_gap, sd[b][m]):
                dropped.append(f'band {b} {m}')
            else:
                kept.append((b, m))
    mean = lambda keys: _complete_mean([abs(gaps[b][m]) if gaps[b][m] is not None else None for b, m in keys])
    return dict(G=mean(kept), G_core=mean([k for k in kept if k[1] not in BAND_RELATIVE]), gaps=gaps,
                dropped=dropped)


def _difference(a, b):
    return a - b if a is not None and b is not None else None


def _run_key(record):
    return record['sha256'], record['mode'], record['seed'], record['start'], record['stop']


def _matched(systems):
    common = set.intersection(*[set(map(_run_key, rows)) for rows in systems])
    return [[r for r in rows if _run_key(r) in common] for rows in systems]


def bootstrap_difference(left, right, baseline, *, measures, bootstrap=2000, seed=954):
    """Paired, band-stratified chart bootstrap; all runs of a chart stay together."""
    left, right, baseline = _matched([left, right, baseline])
    groups = {str(b): sorted({r['sha256'] for r in left if r['band'] == b}) for b in BANDS}
    rng = np.random.default_rng(seed)
    columns = [MEASURES.index(m) for m in measures]
    points, draws = [], []
    for band in BANDS:
        rows = [[r for r in system if r['band'] == band] for system in (left, right, baseline)]
        data = [_band_data(system, 'sample')[2] for system in rows]
        _, runs, source = _band_data(rows[-1], 'source')
        source_m = _metric_arrays(source, source)[0]
        metrics = [_metric_arrays(d, source)[0] for d in data]
        count = len(groups[str(band)])
        indices = rng.integers(count, size=(bootstrap, count)) if count else np.empty((bootstrap, 0), int)
        indices = runs[indices].reshape(bootstrap, runs.size)
        sampled_source = {k: v[indices] for k, v in source.items()}
        sampled_metrics = [_metric_arrays({k: v[indices] for k, v in d.items()}, sampled_source)[0] for d in data]
        sampled_source_m = _metric_arrays(sampled_source, sampled_source)[0]
        drop = _dropped(np.abs(metrics[2] - source_m), _column_sd(sampled_source_m))
        keep = [c for c in columns if not drop[c]]
        two_sided = lambda m, src: _divide(np.abs(m[0] - src) - np.abs(m[1] - src), np.abs(m[2] - src))
        points.append(two_sided(metrics, source_m)[keep])
        draws.append(two_sided(sampled_metrics, sampled_source_m)[:, keep])
    points, draws = np.concatenate(points), np.concatenate(draws, axis=1)
    if not points.size:
        return dict(difference=None, ci90=None, valid_resamples=0, requested_resamples=bootstrap,
                    charts_by_band={b: len(ids) for b, ids in groups.items()}, matched_runs=len(left))
    point = _finite_json(points.mean())
    samples = draws.mean(axis=1)
    samples = samples[np.isfinite(samples)] if point is not None else np.array([])
    return dict(difference=point, ci90=np.quantile(samples, [.05, .95]).tolist() if len(samples) else None,
                valid_resamples=len(samples), requested_resamples=bootstrap,
                charts_by_band={b: len(ids) for b, ids in groups.items()}, matched_runs=len(left))


def ln_share(records):
    """Per band, chart-averaged held-share IQR, bus4 and lane-lock for runs and their matched sources."""
    out = {}
    for band in BANDS:
        _, run = _chart_values([r for r in records if r['band'] == band], 'sample')
        _, src = _chart_values([r for r in records if r['band'] == band], 'source')
        row = {}
        for prefix, values in (('', run), ('source_', src)):
            held = [v for v in values(('means', 'held')) if v is not None]
            row[prefix + 'held_iqr'] = np.percentile(held, [25, 75]).tolist() if held else None
            row[prefix + 'bus4'] = _mean(values(('means', 'bus4')))
            row[prefix + 'lock'] = _mean(values(('means', 'lock')))
        out[str(band)] = row
    return out


def compare_systems(paths, out, *, measures, bootstrap=2000, seed=954):
    """Write Markdown/JSON comparison and update each input JSON's NLL guard against b0."""
    _check_measures(measures)
    systems = {r['system']: r for r in [json.loads(Path(path).read_text()) for path in paths]}
    if 'b0' not in systems:
        raise ValueError('Comparison requires b0')
    base = systems['b0']
    scores = {}
    for name, system in systems.items():
        rows, matched_base = _matched([system['records'], base['records']])
        scores[name] = gap_score(aggregate(rows), aggregate(matched_base, field='source'),
                                 aggregate(matched_base), measures, source_sd(matched_base, seed=seed))
        scores[name]['matched_runs'] = len(rows)
        nll = system['nll']['nll_per_decision'] if system['nll'] else None
        base_nll = base['nll']['nll_per_decision'] if base['nll'] else None
        same_nll_draw = (system['nll'].get('requested_windows') == base['nll'].get('requested_windows')
                         if system['nll'] and base['nll'] else False)
        system['guards']['nll_pass'] = (nll <= base_nll + .02
                                        if nll is not None and base_nll is not None and same_nll_draw else None)
        system['guards']['baseline_nll_per_decision'] = base_nll
    for path in paths:
        original = json.loads(Path(path).read_text())
        _write_json(path, systems[original['system']])
    differences = {}
    names = list(systems)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            differences[f'{left} - {right}'] = bootstrap_difference(
                systems[left]['records'], systems[right]['records'], base['records'],
                measures=measures, bootstrap=bootstrap, seed=seed)
    result = dict(measures=list(measures), scores=scores, differences=differences,
                  guards={name: system['guards'] for name, system in systems.items()}, definitions=DEFINITIONS,
                  reference_complete={name: system.get('reference_complete') for name, system in systems.items()})
    def fmt(value):
        return 'unavailable' if value is None else f'{value:.4f}'
    status = lambda value: 'pass' if value is True else 'fail' if value is False else 'unavailable'
    lines = ['| System | G | G without m1, m6 | Runs matched to b0 | Legal export | No head-lock >=30 | NLL | Single/4-gram envelope | Tail exits |',
             '| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |']
    for name, score in scores.items():
        guard = systems[name]['guards']
        band_guards = list(guard['bands'].values())
        envelope = _guard_all([_guard_all([b['envelope'].get(k) for k in ('single', 'ng4')])
                               if b else None for b in band_guards])
        tail = _guard_all([b['tail_pass'] if b else None for b in band_guards])
        lines.append(f"| {name} | {fmt(score['G'])} | {fmt(score['G_core'])} | {score['matched_runs']} | {status(guard['legal_export'])} | "
                     f"{status(guard['no_head_lock_30'])} | {status(guard['nll_pass'])} | {status(envelope)} | {status(tail)} |")
    lines += ['', 'Selected measures: ' + ', '.join(measures) + '. G averages two-sided |m - m_src| / |m_b0 - m_src|; '
              'lower is better. Terms where |m_b0 - m_src| is below the source chart-bootstrap SD are dropped.', '',
              '| System | Dropped terms |', '| --- | --- |']
    lines += [f"| {name} | {', '.join(score['dropped']) or 'none'} |" for name, score in scores.items()]
    lines += ['', 'Signed per-term gaps (m - m_src) / (m_b0 - m_src), for reading:', '',
              '| System | Band | ' + ' | '.join(measures) + ' |', '| --- | --- |' + ' ---: |' * len(measures)]
    lines += [f"| {name} | {b} | " + ' | '.join(fmt(score['gaps'][b][m]) for m in measures) + ' |'
              for name, score in scores.items() for b in map(str, BANDS)]
    lines += ['',
              '| Difference | G difference | Paired chart-bootstrap 90% CI |', '| --- | ---: | --- |']
    for name, difference in differences.items():
        ci = difference['ci90']
        lines.append(f"| {name} | {fmt(difference['difference'])} | " +
                     (f'[{fmt(ci[0])}, {fmt(ci[1])}]' if ci else 'unavailable') + ' |')
    result['ln_share'] = {name: ln_share(system['records']) for name, system in systems.items()}
    lines += ['', 'LN-share measures (not in G): per-chart held share IQR, mean bus4 and mean lane-lock '
              '(rows), run vs matched source.', '',
              '| System | Band | Held IQR run | Held IQR source | bus4 run | bus4 source | Lane-lock run | Lane-lock source |',
              '| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    iqr = lambda q: 'unavailable' if q is None else f'[{q[0]:.3f}, {q[1]:.3f}]'
    for name, bands in result['ln_share'].items():
        for band, row in bands.items():
            lines.append(f"| {name} | {band} | {iqr(row['held_iqr'])} | {iqr(row['source_held_iqr'])} | "
                         f"{fmt(row['bus4'])} | {fmt(row['source_bus4'])} | {fmt(row['lock'])} | "
                         f"{fmt(row['source_lock'])} |")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    _write_json(out.with_suffix('.json'), result)
    return result


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    reference = sub.add_parser('reference')
    reference.add_argument('--cache', default='artifacts/r2-cache/v1')
    reference.add_argument('--out', required=True)
    reference.add_argument('--limit', type=int)
    evaluate = sub.add_parser('evaluate')
    for key in ('panel', 'runs', 'system', 'reference', 'out'):
        evaluate.add_argument('--' + key, required=True)
    evaluate.add_argument('--cache', default='artifacts/r2-cache/v1')
    evaluate.add_argument('--checkpoint')
    evaluate.add_argument('--nll-windows', type=int, default=64)
    evaluate.add_argument('--measures', default=','.join(MEASURES))
    compare = sub.add_parser('compare')
    compare.add_argument('systems', nargs='+')
    compare.add_argument('--out', required=True)
    compare.add_argument('--measures', required=True)
    compare.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.command == 'reference':
        build_reference(args.cache, args.out, limit=args.limit)
    elif args.command == 'evaluate':
        evaluate_system(args.panel, args.runs, args.system, args.cache, args.out,
                        reference_path=args.reference, checkpoint=args.checkpoint,
                        measures=args.measures.split(','), nll_windows=args.nll_windows)
    else:
        compare_systems(args.systems, args.out, measures=args.measures.split(','), bootstrap=args.bootstrap)


if __name__ == '__main__':
    main()
