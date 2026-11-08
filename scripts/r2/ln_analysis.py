"""Summarize LN probes with song-level intervals and inference noise floors."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from separation import disk_guard, write_json

METRICS = ['ln_share', 'inverse_gap_share', 'hold_p10_ms', 'hold_p50_ms', 'hold_p90_ms',
           'short_hold_share', 'near_head_share', 'co_release_share']


def interval(values, reps=2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(effect=np.nan, lo=np.nan, hi=np.nan, songs=0)
    rng = np.random.default_rng(20261108)
    draws = rng.integers(len(values), size=(reps, len(values)))
    lo, hi = np.percentile(values[draws].mean(1), [2.5, 97.5])
    return dict(effect=float(values.mean()), lo=float(lo), hi=float(hi), songs=len(values))


def rms(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.sqrt(np.mean(v * v))), len(v)) if len(v) else (np.nan, 0)


def pair_spread(table):
    diffs = [table[b] - table[a] for a, b in combinations(table.columns, 2)]
    return rms(pd.concat(diffs).to_numpy()) if diffs else (np.nan, 0)


def fr_floors(table, metric):
    baseline = table[table.arm == 'baseline']
    sampled = baseline[baseline.checkpoint == 56].pivot(index='sha', columns='seed', values=metric)
    seeded = baseline.groupby(['sha', 'checkpoint'])[metric].mean().unstack('checkpoint')
    s, sn = pair_spread(sampled)
    c, cn = pair_spread(seeded)
    return dict(sampling_rms=s, sampling_pairs=sn, checkpoint_rms=c, checkpoint_pairs=cn)


def scaled(record, floors):
    finite = [v for k, v in floors.items() if k.endswith('_rms') and np.isfinite(v)]
    floor = max(finite) if finite else np.nan
    ratio = (abs(record['effect']) / floor if floor > 0 else
             np.inf if floor == 0 and record['effect'] != 0 else np.nan)
    direction = record['lo'] > 0 or record['hi'] < 0
    verdict = 'beyond_inference_floor' if direction and ratio > 1 else (
        'within_inference_floor' if direction and np.isfinite(ratio) else 'unresolved')
    if record['songs'] < 2:
        verdict = 'insufficient_songs'
    return dict(record, **floors, effect_over_floor=ratio, verdict=verdict)


def strata(table, class_col='ln_class'):
    yield 'all', 'all', table
    for cls, rows in table.groupby(class_col):
        yield cls, 'all', rows
        for band, sub in rows.groupby('band'):
            yield cls, int(band), sub


def human_nulls(root):
    table = pd.read_csv(root / 'human-null.csv')
    rows = []
    for mode, m in table.groupby('mode'):
        for creator, c in m.groupby('same_creator'):
            for band_subset in ('all', 'same'):
                selected = c if band_subset == 'all' else c[c.same_band]
                for cls in ('all', 'normal', 'ln_heavy', 'inverse'):
                    subset = selected if cls == 'all' else selected[(selected.a_class == cls) & (selected.b_class == cls)]
                    for metric in METRICS:
                        valid = subset[np.isfinite(subset['diff_' + metric])]
                        values = valid['diff_' + metric].to_numpy()
                        spread, n = rms(values)
                        rows.append(dict(mode=mode, same_creator=creator, bands=band_subset, ln_class=cls,
                                         metric=metric, pairs=n, songs=valid.group_id.nunique(), rms=spread,
                                         p95_abs=np.percentile(abs(values), 95) if n else np.nan))
    result = pd.DataFrame(rows)
    result.to_csv(root / 'human-null-summary.csv', index=False)
    return result


def analyze_census(root):
    table = pd.read_csv(root / 'census.csv')
    details = pd.read_csv(root / 'code-candidate-counts.csv')
    hist = pd.read_csv(root / 'snapping-histogram.csv')
    rows = []
    for (role, cls), sub in table.groupby(['role', 'ln_class']):
        h = hist[(hist.role == role) & (hist.ln_class == cls)].groupby('error_ms').n.sum().sort_index()
        distribution = h.cumsum() / h.sum()
        d = details[(details.role == role) & (details.ln_class == cls)]
        r = dict(role=role, ln_class=cls, charts=len(sub), songs=sub.group_id.nunique(),
                 ln=int(sub.ln.sum()), exact=sub.exact_releases.sum() / sub.ln.sum(),
                 within1=sub.within1.sum() / sub.ln.sum(), within5=sub.within5.sum() / sub.ln.sum(),
                 on_row=sub.n_row_release.sum() / sub.ln.sum(),
                 four_anchor_within5=d.four_anchor_within5.sum() / sub.ln.sum(),
                 candidates_per_gap_release=d.candidate_total.sum() / d.candidate_releases.sum())
        for q in (.5, .95, .99, 1.):
            r[f'snap_p{int(q * 100)}_ms'] = float(distribution.index[distribution >= q - 1e-12][0])
        for state, codes in [('free', range(3)), ('held', range(5))]:
            counts = [d[f'{state}_{c}'].sum() for c in codes]
            for c, n in zip(codes, counts):
                r[f'{state}_{c}_share'] = n / sum(counts)
        for field in [c for c in d if c.startswith('candidate_') or c.startswith('snap_d')]:
            if field not in ('candidate_total', 'candidate_releases'):
                r[field + '_share'] = d[field].sum() / d.candidate_releases.sum()
        rows.append(r)
    pd.DataFrame(rows).to_csv(root / 'representation-summary.csv', index=False)
    code_rows = []
    for (role, cls, band), group in details.groupby(['role', 'ln_class', 'band']):
        for phase, prefix in [('head', ''), ('eos', 'eos_')]:
            for state, codes in [('free', range(3)), ('held', range(5))]:
                counts = [int(group[f'{prefix}{state}_{c}'].sum()) for c in codes]
                for code, n in zip(codes, counts):
                    code_rows.append(dict(role=role, ln_class=cls, band=band, phase=phase,
                        lane_state=state, code=code, n=n, opportunities=sum(counts),
                        share=n / sum(counts) if sum(counts) else np.nan))
    pd.DataFrame(code_rows).to_csv(root / 'code-distribution.csv', index=False)
    coverage_effects = []
    classes = ['normal', 'ln_heavy', 'inverse']
    for role, sub in table.groupby('role'):
        grouped = sub.groupby(['group_id', 'ln_class'])[['ln', 'within5', 'exact_releases']].sum()
        wide = grouped.unstack('ln_class').fillna(0)
        rng = np.random.default_rng(20261108)
        sampled = {metric: [] for metric in ('within5', 'exact_releases')}
        denominator = wide['ln'].reindex(columns=classes, fill_value=0).to_numpy()
        numerators = {metric: wide[metric].reindex(columns=classes, fill_value=0).to_numpy() for metric in sampled}
        for start in range(0, 2000, 50):
            indices = rng.integers(len(wide), size=(min(50, 2000 - start), len(wide)))
            count = denominator[indices].sum(1)
            for metric in sampled:
                numerator = numerators[metric][indices].sum(1)
                rates = np.divide(numerator, count, out=np.full_like(numerator, np.nan, dtype=float), where=count > 0)
                sampled[metric].append(rates - rates[:, :1])
        for metric in sampled:
            draws = np.concatenate(sampled[metric])
            rates = numerators[metric].sum(0) / denominator.sum(0)
            for j, cls in enumerate(classes[1:], 1):
                valid = draws[:, j][np.isfinite(draws[:, j])]
                lo, hi = np.percentile(valid, [2.5, 97.5])
                coverage_effects.append(dict(role=role, ln_class=cls, comparison='minus_normal', metric=metric,
                    effect=rates[j] - rates[0], lo=lo, hi=hi, valid_bootstraps=len(valid),
                    songs=int((denominator[:, j] > 0).sum()), normal_songs=int((denominator[:, 0] > 0).sum()),
                    inference_floor='not_applicable_deterministic_census'))
    pd.DataFrame(coverage_effects).to_csv(root / 'coverage-effects.csv', index=False)
    geometry = root / 'representation-miss-geometry.csv'
    miss = pd.read_csv(geometry if geometry.exists() else root / 'representation-misses.csv')
    miss['gap_bin'] = pd.cut(miss.gap_ms, [0, 40, 100, 250, 1000, np.inf], right=False).astype(str)
    miss['grid_position'] = 'off_other_grid'
    unresolved = np.ones(len(miss), bool)
    for denominator in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 64):
        distance = abs(miss.original_phase * denominator - np.round(miss.original_phase * denominator))
        match = unresolved & (distance / denominator * miss.canonical_beat_ms <= 1)
        miss.loc[match, 'grid_position'] = '1/' + str(denominator)
        unresolved &= ~match
    miss.to_csv(root / 'representation-miss-context.csv', index=False)
    grouped = []
    for variable in ('gap_bin', 'grid_position', 'code', 'band', 'heads_at_row', 'held_at_row'):
        for (cls, value), m in miss.groupby(['ln_class', variable]):
            grouped.append(dict(ln_class=cls, variable=variable, value=value, misses=len(m),
                                charts=m.sha.nunique(), median_error=m.error_ms.median(),
                                max_error=m.error_ms.max()))
    pd.DataFrame(grouped).to_csv(root / 'representation-miss-summary.csv', index=False)


def reweighted_prevalence(root, table):
    """Poststratification is descriptive where the panel covers a corpus cell."""
    census = pd.read_csv(root / 'census.csv')
    frequencies = census[census.role == 'fit_dev'].groupby(['ln_class', 'band']).size()
    rows = []
    for kappa, sub in table[(table['mode'] == 'bos') & (table.arm == 'baseline')].groupby('kappa'):
        cells = sub[['sha', 'source_class', 'band']].drop_duplicates()
        covered_cells = pd.MultiIndex.from_frame(cells[['source_class', 'band']].drop_duplicates())
        covered = frequencies.reindex(covered_cells).dropna()
        cell_n = cells.groupby(['source_class', 'band']).size()
        weights = cells.set_index('sha').apply(
            lambda row: frequencies.loc[(row.source_class, row.band)] /
            cell_n.loc[(row.source_class, row.band)] / covered.sum(), axis=1)
        for output in ('normal', 'ln_heavy', 'inverse'):
            values = sub.assign(hit=(sub.ln_class == output).astype(float))
            seed_table = values[values.checkpoint == 56].pivot(index='sha', columns='seed', values='hit')
            checkpoint_table = values.groupby(['sha', 'checkpoint']).hit.mean().unstack('checkpoint')

            def weighted_spread(wide):
                differences = pd.concat([wide[a] - wide[b] for a, b in combinations(wide.columns, 2)])
                w = weights.reindex(differences.index)
                return float(np.sqrt((differences.pow(2) * w).sum() / w.sum())), len(differences)

            sampling, sn = weighted_spread(seed_table)
            checkpoint, cn = weighted_spread(checkpoint_table)
            chart_rates = seed_table.mean(1)
            estimate = float((chart_rates * weights).sum())
            corpus_rate = float((census[census.role == 'fit_dev'].ln_class == output).mean())
            rng = np.random.default_rng(20261108)
            draws = np.zeros(2000)
            for cell, group in cells.groupby(['source_class', 'band']):
                rates = chart_rates.reindex(group.sha).to_numpy()
                draws += rates[rng.integers(len(rates), size=(2000, len(rates)))].mean(1) * covered.loc[cell] / covered.sum()
            lo, hi = np.percentile(draws - corpus_rate, [2.5, 97.5])
            result = dict(kappa=kappa, output_class=output, panel_conditional_rate=estimate,
                corpus_rate=corpus_rate, covered_charts=int(covered.sum()), total_charts=int(frequencies.sum()),
                **scaled(dict(effect=estimate - corpus_rate, lo=lo, hi=hi, songs=len(cells)),
                         dict(sampling_rms=sampling, sampling_pairs=sn,
                              checkpoint_rms=checkpoint, checkpoint_pairs=cn)))
            result['positive_runs'] = int(values.hit.sum())
            result['bootstrap_degenerate'] = bool(np.ptp(draws) == 0)
            if result['positive_runs'] == 0 and corpus_rate > 0:
                # Resampling zero events cannot bound an unobserved rare-event rate.
                result['verdict'] = 'unassessable_zero_count'
            rows.append(result)
    pd.DataFrame(rows).to_csv(root / 'fr-reweighted-prevalence.csv', index=False)


def analyze_fr(root, nulls):
    tables = [pd.read_json(p, lines=True) for p in (root / 'fr').glob('*.jsonl') if p.with_suffix('.done').exists()]
    table = pd.concat(tables, ignore_index=True)
    code_columns = [f'held_{i}' for i in range(5)] + [f'free_{i}' for i in range(3)]
    table['held_share'] = table[[f'held_{i}' for i in range(5)]].sum(1) / table[code_columns].sum(1)
    table.to_parquet(root / 'fr-summary.parquet', index=False)
    effects, levels, transitions = [], [], []
    for mode, m in table.groupby('mode'):
        human = m[m.arm == 'human'].set_index('sha')
        gen = m[m.arm != 'human']
        for cls, band, selected in strata(gen, 'source_class'):
            sources = human.loc[selected.sha.unique()]
            for kappa, sub in selected.groupby('kappa'):
                base56 = sub[(sub.arm == 'baseline') & (sub.checkpoint == 56)]
                for metric in METRICS:
                    floors = fr_floors(sub, metric)
                    generated = base56.groupby('sha')[metric].mean()
                    diff = generated - sources[metric]
                    result = dict(mode=mode, ln_class=cls, band=band, kappa=kappa, metric=metric,
                                  comparison='baseline_minus_human', **scaled(interval(diff), floors),
                                  generated_mean=generated.mean(), human_mean=sources.loc[generated.index, metric].mean(),
                                  mean_absolute_gap=diff.abs().mean())
                    for bands in ('all', 'same'):
                        null = nulls[(nulls['mode'] == mode) & ~nulls.same_creator & (nulls.bands == bands) &
                                     (nulls.ln_class == cls) & (nulls.metric == metric)].iloc[0]
                        result['human_' + bands + '_pairs'] = null.pairs
                        result['human_' + bands + '_p95'] = null.p95_abs
                        result['inside_human_' + bands] = (result['mean_absolute_gap'] <= null.p95_abs
                                                         if np.isfinite(null.p95_abs) else None)
                    effects.append(result)
                    for arm in ('fixed_forward', 'same_birth'):
                        altered = sub[sub.arm == arm].groupby('sha')[metric].mean()
                        effects.append(dict(mode=mode, ln_class=cls, band=band, kappa=kappa, metric=metric,
                                            comparison=arm + '_minus_baseline',
                                            **scaled(interval(altered - generated), floors)))
                for checkpoint, c in sub[sub.arm == 'baseline'].groupby('checkpoint'):
                    levels.append(dict(mode=mode, ln_class=cls, band=band, kappa=kappa, checkpoint=checkpoint,
                                       runs=len(c), songs=c.sha.nunique(), **{x: c[x].mean() for x in METRICS}))
                for arm, a in sub[sub.checkpoint == 56].groupby('arm'):
                    for output_class in ('normal', 'ln_heavy', 'inverse'):
                        per = a.assign(hit=(a.ln_class == output_class).astype(float)).groupby('sha').hit.mean()
                        observed = (sources.loc[per.index, 'ln_class'] == output_class).astype(float)
                        floors = fr_floors(sub.assign(hit=(sub.ln_class == output_class).astype(float)), 'hit')
                        transitions.append(dict(mode=mode, input_class=cls, band=band, kappa=kappa, arm=arm,
                            output_class=output_class, runs=len(a), generated_rate=per.mean(),
                            human_rate=observed.mean(), **scaled(interval(per - observed), floors)))
            zero = selected[(selected.arm == 'baseline') & (selected.checkpoint == 56) & (selected.kappa == 0)]
            three = selected[(selected.arm == 'baseline') & (selected.checkpoint == 56) & (selected.kappa == 3)]
            for metric in METRICS:
                floors0, floors3 = [fr_floors(selected[selected.kappa == k], metric) for k in (0, 3)]
                floors = {key: max(floors0[key], floors3[key]) for key in floors0}
                diff = three.groupby('sha')[metric].mean() - zero.groupby('sha')[metric].mean()
                effects.append(dict(mode=mode, ln_class=cls, band=band, kappa='3-0', metric=metric,
                                    comparison='kappa3_minus_kappa0', **scaled(interval(diff), floors)))
    pd.DataFrame(effects).to_csv(root / 'fr-effects.csv', index=False)
    pd.DataFrame(levels).to_csv(root / 'fr-levels.csv', index=False)
    pd.DataFrame(transitions).to_csv(root / 'fr-class-transitions.csv', index=False)
    context = []
    for mode, m in table.groupby('mode'):
        human = m[m.arm == 'human'].set_index('sha')
        for cls, band, selected in strata(m[m.arm != 'human'], 'source_class'):
            for kappa, sub in selected.groupby('kappa'):
                b56 = sub[(sub.arm == 'baseline') & (sub.checkpoint == 56)]
                by_song = b56.groupby('sha').held_share.mean()
                target = b56.drop_duplicates('sha').target.map(lambda t: t[0])
                context.append(dict(mode=mode, ln_class=cls, band=band, kappa=kappa,
                    human_held_share=human.loc[by_song.index, 'held_share'].mean(),
                    target_held_share=target.mean(), generated_held_share=by_song.mean(),
                    **scaled(interval(by_song - human.loc[by_song.index, 'held_share']), fr_floors(sub, 'held_share'))))
    pd.DataFrame(context).to_csv(root / 'prior-occupancy-context.csv', index=False)
    generated = table[table.arm != 'human']
    reweighted_prevalence(root, table)
    write_json(root / 'fr-integrity.json', dict(songs=generated.sha.nunique(), runs=len(generated),
               legal=int(generated.legal.sum()), fallbacks=int(generated.fallbacks.sum()),
               seconds_sum=float(generated.seconds.sum()), max_rss_bytes=float(generated.peak_rss_bytes.max())))


def load_tf(root, kind):
    table = pd.concat([pd.read_parquet(p) for p in (root / 'tf').glob('*-' + kind + '.parquet')], ignore_index=True)
    # Empty factor tables can make pandas retain object dtype after concatenation.
    table['eos'] = table.eos.astype(bool)
    return table


def likelihood_effects(sub, common):
    records = []
    for part in ('all', 'head', 'eos'):
        selected = sub if part == 'all' else sub[sub.eos == (part == 'eos')]
        if selected.empty:
            continue
        for unit in ('decision', 'release'):
            grouped = selected.groupby(['sha', 'checkpoint']).agg(pointer=('pointer_nll', 'sum'),
                                    n=('k', 'size'), releases=('gap_releases', 'sum'))
            denominator = grouped.n if unit == 'decision' else grouped.releases.replace(0, np.nan)
            values = (grouped.pointer / denominator).unstack('checkpoint')
            cp, cpn = pair_spread(values)
            baseline = selected[selected.checkpoint == 56]
            for lesion in ('forward', 'reverse', 'same_birth', 'hold_age', 'row_heads', 'other_hand'):
                col = lesion + ('_nll' if lesion in ('forward', 'reverse') else '_pointer_nll')
                differences = baseline.assign(diff=baseline[col] - baseline.pointer_nll).groupby('sha').agg(
                    diff=('diff', 'sum'), n=('k', 'size'), releases=('gap_releases', 'sum'))
                delta = differences['diff'] / (differences.n if unit == 'decision' else differences.releases.replace(0, np.nan))
                records.append(dict(common, part=part, unit=unit, lesion=lesion,
                    **scaled(interval(delta), dict(sampling_rms=0., sampling_pairs=0, checkpoint_rms=cp,
                                                   checkpoint_pairs=cpn))))
    return records


def analyze_tf(root):
    decisions, hazards, events = (load_tf(root, name) for name in ('decisions', 'hazards', 'events'))
    song_keys = ['sample', 'sha', 'group', 'ln_class', 'band', 'checkpoint', 'eos']
    count_columns = [c for c in decisions if ('nll' in c or '_code' in c or c.startswith('birth_') or
                     c.startswith('release_') or c in ('held', 'heads', 'gap_releases'))]
    compact = decisions.groupby(song_keys)[count_columns].sum(min_count=1)
    compact['decisions'] = decisions.groupby(song_keys).size()
    compact.reset_index().to_csv(root / 'tf-song-metrics.csv', index=False)
    event_columns = [c for c in events if c in ('observed', 'predicted') or c.endswith('_predicted')]
    compact = events.groupby(song_keys[:-1] + ['event'])[event_columns].mean()
    compact['n_factors'] = events.groupby(song_keys[:-1] + ['event']).size()
    compact.reset_index().to_csv(root / 'tf-pointer-song-metrics.csv', index=False)
    summaries, effects = [], []
    for sample, rows in decisions.groupby('sample'):
        for cls, band, sub in strata(rows):
            for checkpoint, c in sub.groupby('checkpoint'):
                for part in ('all', 'head', 'eos'):
                    chosen = c if part == 'all' else c[c.eos == (part == 'eos')]
                    if not len(chosen):
                        continue
                    summaries.append(dict(sample=sample, ln_class=cls, band=band, checkpoint=checkpoint, part=part,
                        decisions=len(chosen), songs=chosen.sha.nunique(), gap_releases=chosen.gap_releases.sum(),
                        code_nll=chosen.code_nll.mean(), pointer_nll=chosen.pointer_nll.mean(),
                        total_nll=(chosen.code_nll + chosen.pointer_nll).mean(),
                        pointer_nll_per_release=chosen.pointer_nll.sum() / max(chosen.gap_releases.sum(), 1),
                        conditional_orientation_tv=chosen.conditional_orientation_tv.mean(),
                        orientation_abs_log_ratio=chosen.orientation_abs_log_ratio.mean()))
            effects.extend(likelihood_effects(sub, dict(sample=sample, ln_class=cls, band=band)))
    pd.DataFrame(summaries).to_csv(root / 'tf-nll.csv', index=False)
    pd.DataFrame(effects).to_csv(root / 'tf-lesions.csv', index=False)
    bins = [0, 30, 60, 125, 250, 500, 1000, 2000, np.inf]
    for age in ('age_ms', 'previous_age_ms'):
        h = hazards.copy()
        h['age_bin'] = pd.cut(h[age], bins, right=False).astype(str)
        h.groupby(['sample', 'ln_class', 'band', 'checkpoint', 'eos', 'age_bin'], observed=True).agg(
            n=('k', 'size'), songs=('sha', 'nunique'), observed=('observed', 'mean'),
            predicted=('predicted', 'mean')).reset_index().to_csv(root / ('hazard-' + age + '.csv'), index=False)
    events['probability_bin'] = np.minimum((events.predicted * 10).astype(int), 9)
    events.groupby(['sample', 'ln_class', 'band', 'checkpoint', 'event'], observed=True).agg(
        n_factors=('k', 'size'), songs=('sha', 'nunique'), observed=('observed', 'mean'),
        predicted=('predicted', 'mean')).reset_index().to_csv(root / 'pointer-calibration.csv', index=False)
    events.assign(brier=(events.predicted - events.observed) ** 2).groupby(
        ['sample', 'ln_class', 'checkpoint', 'event', 'probability_bin'], observed=True).agg(
        n_factors=('k', 'size'), observed=('observed', 'mean'), predicted=('predicted', 'mean'),
        brier=('brier', 'mean')).reset_index().to_csv(root / 'pointer-reliability.csv', index=False)
    # Within-song calibration gaps get the same checkpoint floor as likelihood effects.
    calibration_effects = []
    head_hazards = hazards[~hazards.eos]
    frames = [('hazard', head_hazards.assign(event='release')), ('pointer', events)]
    for age in ('age_ms', 'previous_age_ms'):
        frames.append(('hazard_' + age, head_hazards.assign(
            event=pd.cut(head_hazards[age], bins, right=False).astype(str))))
    for name, frame in frames:
        for sample, sample_rows in frame.groupby('sample'):
            for cls, band, sub in strata(sample_rows):
                for event, e in sub.groupby('event'):
                    by = e.groupby(['sha', 'checkpoint'])[['predicted', 'observed']].mean()
                    cp, cpn = pair_spread(by.predicted.unstack('checkpoint'))
                    b = by.xs(56, level='checkpoint')
                    calibration_effects.append(dict(kind=name, sample=sample, ln_class=cls, band=band, event=event,
                        predicted=b.predicted.mean(), observed=b.observed.mean(),
                        **scaled(interval(b.predicted - b.observed), dict(sampling_rms=0., sampling_pairs=0,
                                                     checkpoint_rms=cp, checkpoint_pairs=cpn))))
    for sample, sample_rows in decisions[~decisions.eos].groupby('sample'):
        for cls, band, sub in strata(sample_rows):
            for dense, d in sub.assign(dense=sub.gap_ms <= 125).groupby('dense'):
                for state in ('free', 'held'):
                    g = d.assign(opportunities=4 - d.held if state == 'free' else d.held).groupby(
                        ['sha', 'checkpoint']).agg(expected=('birth_' + state + '_expected', 'sum'),
                        observed=('birth_' + state + '_observed', 'sum'), n=('opportunities', 'sum'))
                    g['predicted'] = g.expected / g.n.replace(0, np.nan)
                    g['observed'] = g.observed / g.n.replace(0, np.nan)
                    cp, cpn = pair_spread(g.predicted.unstack('checkpoint'))
                    b = g.xs(56, level='checkpoint')
                    calibration_effects.append(dict(kind='birth', sample=sample, ln_class=cls, band=band,
                        event=f'{state}_dense{int(dense)}', predicted=b.predicted.mean(), observed=b.observed.mean(),
                        **scaled(interval(b.predicted - b.observed), dict(sampling_rms=0., sampling_pairs=0,
                                                     checkpoint_rms=cp, checkpoint_pairs=cpn))))
    pd.DataFrame(calibration_effects).to_csv(root / 'tf-calibration-effects.csv', index=False)
    code_columns = [c for c in decisions if '_code' in c]
    decisions[~decisions.eos].groupby(['sample', 'ln_class', 'band', 'checkpoint'])[code_columns].sum().reset_index().to_csv(
        root / 'tf-code-calibration.csv', index=False)
    dense = decisions[~decisions.eos].assign(dense=lambda t: t.gap_ms <= 125, free=lambda t: 4 - t.held)
    dense.groupby(['sample', 'ln_class', 'band', 'checkpoint', 'dense']).agg(
        n=('k', 'size'), free=('free', 'sum'), held=('held', 'sum'), heads=('heads', 'sum'),
        birth_free_observed=('birth_free_observed', 'sum'), birth_free_expected=('birth_free_expected', 'sum'),
        birth_held_observed=('birth_held_observed', 'sum'), birth_held_expected=('birth_held_expected', 'sum')
        ).reset_index().to_csv(root / 'dense-births.csv', index=False)
    diagnostic = events.copy()
    diagnostic['gap_bin'] = pd.cut(diagnostic.gap_ms, [0, 40, 100, 250, 1000, np.inf], right=False).astype(str)
    diagnostic['age_bin'] = pd.cut(diagnostic.age_ms, bins, right=False).astype(str)
    identity = ['sample', 'sha', 'checkpoint', 'k', 'lane', 'orientation']
    together_index = pd.MultiIndex.from_frame(diagnostic.loc[diagnostic.event == 'together', identity])
    diagnostic['same_birth_present'] = pd.MultiIndex.from_frame(diagnostic[identity]).isin(together_index)
    rows = []
    context_effects = []
    for variable in ('gap_bin', 'age_bin', 'other_hand_held', 'row_heads', 'same_birth_present'):
        for keys, all_checkpoints in diagnostic.groupby(['sample', 'ln_class', 'event', variable]):
            sample, cls, event, value = keys
            group = all_checkpoints[all_checkpoints.checkpoint == 56]
            r = dict(sample=sample, ln_class=cls, event=event, variable=variable, value=value,
                     n=len(group), observed=group.observed.mean(), predicted=group.predicted.mean())
            for lesion in ('same_birth', 'hold_age', 'row_heads', 'other_hand'):
                r[lesion + '_predicted'] = group[lesion + '_predicted'].mean()
            rows.append(r)
            by = all_checkpoints.groupby(['sha', 'checkpoint'])[['predicted', 'observed']].mean()
            cp, cpn = pair_spread(by.predicted.unstack('checkpoint'))
            base = by.xs(56, level='checkpoint')
            floors = dict(sampling_rms=0., sampling_pairs=0, checkpoint_rms=cp, checkpoint_pairs=cpn)
            common = dict(sample=sample, ln_class=cls, event=event, variable=variable, value=value,
                          n_factors=len(group))
            context_effects.append(dict(common, comparison='baseline_minus_human',
                **scaled(interval(base.predicted - base.observed), floors)))
            for lesion in ('same_birth', 'hold_age', 'row_heads', 'other_hand'):
                altered = group.groupby('sha')[lesion + '_predicted'].mean()
                context_effects.append(dict(common, comparison=lesion + '_minus_baseline',
                    **scaled(interval(altered - base.predicted), floors)))
    pd.DataFrame(rows).to_csv(root / 'pointer-context.csv', index=False)
    pd.DataFrame(context_effects).to_csv(root / 'pointer-context-effects.csv', index=False)


def analyze_joint(root):
    records = [r for p in (root / 'joint-pointer').glob('*.json') for r in json.loads(p.read_text())]
    table = pd.DataFrame(records)
    table.to_parquet(root / 'joint-pointer-summary.parquet', index=False)
    compact = table.groupby(['sha', 'group', 'ln_class', 'band', 'checkpoint', 'arm', 'seed'])
    columns = [c for c in table if c.endswith(('_observed', '_predicted'))]
    per_song = compact[columns].mean()
    per_song['decisions'] = compact.size()
    per_song.reset_index().to_csv(root / 'joint-pointer-song-metrics.csv', index=False)
    results = []
    for cls, band, sub in strata(table):
        for event in ('short', 'near', 'together'):
            observed, predicted = event + '_observed', event + '_predicted'
            per = sub.groupby(['sha', 'checkpoint', 'seed', 'arm'])[[observed, predicted]].mean().reset_index()
            base = per[per.arm == 'none'].copy()
            base['arm'] = 'baseline'
            floors = fr_floors(base, predicted)
            b56 = base[base.checkpoint == 56].groupby('sha')[[observed, predicted]].mean()
            common = dict(ln_class=cls, band=band, event=event,
                          decisions=len(sub[['sha', 'k']].drop_duplicates()),
                          draws=int(sub.samples.sum()))
            results.append(dict(common, comparison='baseline_minus_human',
                predicted=b56[predicted].mean(), observed=b56[observed].mean(),
                **scaled(interval(b56[predicted] - b56[observed]), floors)))
            for arm in ('same_birth', 'fixed_forward'):
                altered = per[per.arm == arm].groupby('sha')[predicted].mean()
                results.append(dict(common, comparison=arm + '_minus_baseline',
                    **scaled(interval(altered - b56[predicted]), floors)))
    pd.DataFrame(results).to_csv(root / 'joint-pointer-effects.csv', index=False)


def analyze_diagnostics(root):
    def read(kind):
        return pd.concat([pd.read_parquet(p) for p in (root / 'diagnostics').glob('*-' + kind + '.parquet')
                          if p.with_name(p.name.removesuffix('-' + kind + '.parquet') + '.done').exists()],
                         ignore_index=True)
    decisions, hazards, releases = [read(k) for k in ('decisions', 'hazards', 'releases')]
    keys = ['ln_class', 'band', 'checkpoint', 'mode', 'kappa']
    decisions[~decisions.eos].groupby(keys + ['dense']).agg(
        decisions=('k', 'size'), songs=('sha', 'nunique'), free=('free', 'sum'), held=('held', 'sum'),
        heads=('heads', 'sum'), birth_free=('birth_free', 'sum'), birth_held=('birth_held', 'sum'),
        raw_birth_free_expected=('raw_birth_free_expected', 'sum'),
        raw_birth_held_expected=('raw_birth_held_expected', 'sum')).reset_index().to_csv(
            root / 'fr-dense-births.csv', index=False)
    hazards.groupby(keys + ['eos', 'age_bin']).agg(n=('k', 'size'), observed=('observed', 'mean'),
        raw_expected=('raw_expected', 'mean')).reset_index().to_csv(root / 'fr-hazard-age.csv', index=False)
    releases[releases.born_in_suffix].groupby(keys + ['gap_release']).agg(
        n=('k', 'size'), near=('near', 'sum'), hold_ms=('hold_ms', 'mean'),
        before_row_ms=('before_row_ms', 'mean')).reset_index().to_csv(root / 'fr-release-placement.csv', index=False)
    # Compare model-owned and human histories at exactly the same scored rows.
    human = load_tf(root, 'decisions')
    human = human[(human['sample'] == 'panel') & (human.checkpoint == 56) & ~human.eos]
    human = human[['sha', 'k', 'held', 'heads', 'birth_free_observed', 'birth_held_observed']].rename(
        columns={'held': 'human_held', 'heads': 'human_heads'})
    matched = decisions[~decisions.eos].merge(human, on=['sha', 'k'], validate='many_to_one')
    matched['human_free'] = 4 - matched.human_held
    groups = ['sha', 'ln_class', 'band', 'checkpoint', 'mode', 'kappa', 'seed', 'dense']
    sums = matched.groupby(groups).sum(numeric_only=True).reset_index()
    sums['arm'] = 'baseline'
    sums.to_csv(root / 'history-birth-song-counts.csv', index=False)
    effects = []
    for mode, m in sums.groupby('mode'):
        for cls, band, sub in strata(m):
            for (kappa, dense), s in sub.groupby(['kappa', 'dense']):
                for state in ('free', 'held'):
                    s = s.copy()
                    s['raw'] = s['raw_birth_' + state + '_expected'] / s[state].replace(0, np.nan)
                    s['actual'] = s['birth_' + state] / s[state].replace(0, np.nan)
                    s['source'] = s['birth_' + state + '_observed'] / s['human_' + state].replace(0, np.nan)
                    for metric, reference in [('raw', 'source'), ('actual', 'source'), ('actual', 'raw')]:
                        floors = fr_floors(s, metric)
                        b = s[s.checkpoint == 56].groupby('sha')[[metric, reference]].mean()
                        effects.append(dict(mode=mode, ln_class=cls, band=band, kappa=kappa, dense=dense,
                            lane_state=state, comparison=metric + '_minus_' + reference,
                            model_mean=b[metric].mean(), reference_mean=b[reference].mean(),
                            **scaled(interval(b[metric] - b[reference]), floors)))
    pd.DataFrame(effects).to_csv(root / 'history-birth-effects.csv', index=False)
    # Instrumentation must leave the random sequence and generated result unchanged.
    replay = pd.DataFrame([r for p in (root / 'diagnostics').glob('*-replay.json') for r in json.loads(p.read_text())])
    replay.to_csv(root / 'diagnostic-replay-metrics.csv', index=False)
    baseline = pd.read_parquet(root / 'fr-summary.parquet')
    baseline = baseline[baseline.arm == 'baseline']
    index = ['sha', 'checkpoint', 'mode', 'kappa', 'seed']
    merged = replay.merge(baseline, on=index, suffixes=('_replay', '_original'), validate='one_to_one')
    assert len(merged) == len(replay), 'Primary FR result missing for a diagnostic replay'
    for metric in METRICS:
        assert np.allclose(merged[metric + '_replay'], merged[metric + '_original'], equal_nan=True), metric
    write_json(root / 'diagnostic-integrity.json', dict(runs=len(merged), songs=merged.sha.nunique(),
                                                     identical_metrics=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['census', 'tf', 'fr', 'joint-pointer', 'diagnostics', 'all'])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.out)
    disk_guard(root)
    if args.command in ('census', 'all'):
        analyze_census(root)
    if args.command in ('tf', 'all'):
        analyze_tf(root)
    if args.command in ('fr', 'all'):
        analyze_fr(root, human_nulls(root))
    if args.command in ('joint-pointer', 'all'):
        analyze_joint(root)
    if args.command in ('diagnostics', 'all'):
        analyze_diagnostics(root)


if __name__ == '__main__':
    main()
