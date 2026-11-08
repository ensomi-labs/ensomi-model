#!/usr/bin/env python3
"""Analyze private audio-to-chart runs using a frozen stage-1 metric implementation.

No charts are repaired. Style statistics retain stage 1's definitions, including
its treatment of overlapping inverse gaps. Legal-chart summaries are separate.
Marginal hazards include only positive, complete holds inside the audio span.
Missing metrics stay missing; paired seeds are averaged within song before
song bootstrap intervals. One seed supplies no sampling floor.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd


def metric_code(snapshot):
    sys.path[:0] = [str(snapshot / 'src'), str(snapshot / 'scripts/r2')]
    import separation
    import ln_analysis
    import ln_probe
    import release_evaluate
    from ensomi_model.osu_core.hitobjects import parse_mania_hit_objects, ManiaHitObjectKind
    from ensomi_model.evaluation.legality import violations
    from ensomi_model.evaluation.redlines import read_red_lines, musical_grid
    from ensomi_model.r2.common import GridArrays
    from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
    from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
    from ensomi_model.research.chart.actions import EMPTY, TAP, LN_START, LN_CLOSE
    return SimpleNamespace(**locals())


def exact_events(events, code):
    grouped = defaultdict(dict)
    try:
        for t, lane, action in events:
            if lane in grouped[t]:
                raise ValueError(f'Multiple lane events at {t} ms on lane {lane}')
            grouped[t][lane] = action
        state = code.ExactReplayState()
        times = sorted(grouped)
        for i, t in enumerate(times):
            actions = tuple(grouped[t].get(lane, code.EMPTY) for lane in range(4))
            state = code.commit(state, code.CompleteRow(t, actions), is_terminal=i == len(times) - 1)
        return dict(complete=bool(state.is_complete and not any(state.occupancy)),
                    open_holds=sum(state.occupancy), rows=state.row_count)
    except Exception as e:
        return dict(complete=False, error=str(e))


def inspect_chart(path, source, source_grid, code):
    result = dict(path=str(path))
    try:
        objects = code.parse_mania_hit_objects(path)
    except Exception as e:
        return dict(result, parse=False, legal=False, error=str(e))
    result['parse'] = True
    result['violations'] = code.violations(objects, song_span=(0, source['audio_duration_ms']))
    result['grammar'] = not code.violations(objects)
    events = []
    hold = code.ManiaHitObjectKind.HOLD
    for o in objects:
        events.append((o.start_time_ms, o.lane, code.LN_START if o.kind is hold else code.TAP))
        if o.kind is hold:
            events.append((o.end_time_ms, o.lane, code.LN_CLOSE))
    result['replay'] = exact_events(events, code)
    result['holds_closed'] = result['replay']['complete']
    result['legal'] = not result['violations'] and result['replay']['complete']
    result['source_grid_metrics'] = code.separation.ln_statistics(objects, source_grid)
    try:
        grid, _ = code.musical_grid(code.read_red_lines(path), np.unique([o.start_time_ms for o in objects]))
        grid = code.GridArrays.from_grid(grid)
        result['metrics'] = code.separation.ln_statistics(objects, grid)
    except Exception as e:
        result['metric_error'] = str(e)
    complete = [o for o in objects if o.kind is hold and
                0 <= o.start_time_ms < o.end_time_ms <= source['audio_duration_ms']]
    result['duration_hazards'] = code.release_evaluate.duration_hazards(complete, 0)
    result['hazard_complete_holds'] = len(complete)
    result['hazard_excluded_holds'] = sum(o.kind is hold for o in objects) - len(complete)
    return result


def raw_mapper(path, code):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    active, errors, events = {}, Counter(), []
    previous = -np.inf
    for g in data['groups']:
        kind = g['type']
        if kind not in ('circle', 'hold_note', 'hold_note_end'):
            continue
        if not g['timestamp_present']:
            errors['missing_timestamp'] += 1
            continue
        if g['x'] is None:
            errors['missing_position'] += 1
            continue
        lane = min(3, max(0, int(g['x'] * 4 / 512)))
        t = float(g['time'])
        errors['time_reversal'] += t < previous
        previous = t
        if kind == 'hold_note_end':
            if lane not in active:
                errors['unmatched_end'] += 1
            elif t <= active[lane]:
                errors['nonpositive_hold'] += 1
            active.pop(lane, None)
            action = code.LN_CLOSE
        else:
            errors['head_while_held'] += lane in active
            if kind == 'hold_note':
                active[lane] = t
                action = code.LN_START
            else:
                action = code.TAP
        events.append((t, lane, action))
    errors['open_holds_at_end'] = len(active)
    replay = exact_events(events, code)
    return dict(errors={k: int(v) for k, v in errors.items() if v}, events=len(events),
                replay=replay, grammar=not any(errors.values()) and replay['complete'])


def inspect_runs(root, cache, code):
    panel = json.loads((root / 'results/panel.json').read_text())
    plan = json.loads((root / 'setup/jobs.json').read_text())
    jobs = {j['id']: j for j in plan['jobs']}
    completed = {r['id'] for r in (json.loads(s) for s in (root/'results/jobs.jsonl').read_text().splitlines())
                 if r['status']=='completed'}
    records = []
    for source in panel:
        dec = code.ln_probe.heldout_chart(source, cache)
        grid = code.separation.grid_of(dec)
        for system in ('r2', 'mug', 'mapper'):
            directory = root / 'outputs' / system / source['sha256']
            for path in sorted(directory.glob('*.osu')):
                if path.name.endswith('.unpatched.osu') or not path.with_suffix('.json').exists():
                    continue
                arm, seed = path.stem.rsplit('-', 1)
                if f'{system}/{source["sha256"]}/{path.stem}' not in completed:
                    continue
                receipt = json.loads(path.with_suffix('.json').read_text())
                result = dict(system=system, sha=source['sha256'], source_class=source['ln_class'],
                              band=source['band'], arm=arm, seed=int(seed), requested_star=source['star'],
                              source_ln_share=source['ln_share'], receipt=receipt,
                              **inspect_chart(path, source, grid, code))
                result['wall_seconds'] = receipt['wall_seconds']
                result['peak_rss_bytes'] = receipt['peak_rss_bytes']
                job = jobs[f'{system}/{source["sha256"]}/{path.stem}']
                if Path(job['log']).exists():
                    text = Path(job['log']).read_text(errors='replace')
                    wall = re.findall(r'^\s*([\d.]+) real', text, re.M)
                    if wall:
                        result['wall_seconds'] = float(wall[-1])
                if system == 'mapper':
                    result['unpatched'] = inspect_chart(path.with_suffix('.unpatched.osu'), source, grid, code)
                    result['raw_stream'] = raw_mapper(path.with_suffix('.events.json.gz'), code)
                    result['export_warnings'] = json.loads(path.with_suffix('.export-audit.json').read_text())
                    if result['parse'] and result['unpatched']['parse']:
                        key = lambda o: (o.start_time_ms, o.lane, o.end_time_ms, o.kind)
                        fixed = Counter(key(o) for o in code.parse_mania_hit_objects(path))
                        old = Counter(key(o) for o in code.parse_mania_hit_objects(path.with_suffix('.unpatched.osu')))
                        result['export_object_difference'] = dict(patched_only=sum((fixed-old).values()),
                            unpatched_only=sum((old-fixed).values()), common=sum((fixed & old).values()))
                records.append(result)
    return records


def summaries(records, human, code):
    rows, hazards = [], []
    for r in records:
        if 'metrics' not in r:
            continue
        h = human[r['sha']]['metrics']
        rows.append(dict(system=r['system'], arm=r['arm'], sha=r['sha'], seed=r['seed'], band=r['band'],
                         source_class=r['source_class'], legal=r['legal'], wall_seconds=r['wall_seconds'],
                         peak_rss_gb=r['peak_rss_bytes']/1e9, **r['metrics']))
        for item, ref in zip(r['duration_hazards'], human[r['sha']]['duration_hazards']):
            rate = item['releases']/item['risk'] if item['risk'] else np.nan
            human_rate = ref['releases']/ref['risk'] if ref['risk'] else np.nan
            hazards.append(dict(system=r['system'], arm=r['arm'], sha=r['sha'], seed=r['seed'],
                                source_class=r['source_class'], band=r['band'], legal=r['legal'],
                                **item, rate=rate, human_rate=human_rate, diff=rate-human_rate))
    frame = pd.DataFrame(rows)
    hazard_frame = pd.DataFrame(hazards)
    output = []
    for (system, arm), group in frame.groupby(['system', 'arm']):
        for sensitivity in ('all_parsed', 'legal_only'):
            selected = group if sensitivity == 'all_parsed' else group[group.legal]
            for cls, band, sub in code.ln_analysis.strata(selected, 'source_class'):
                for metric in code.ln_analysis.METRICS + ['class_attainment']:
                    if metric == 'class_attainment':
                        sub = sub.assign(class_attainment=(sub.ln_class == sub.source_class).astype(float))
                        reference = pd.Series({sha: 1. for sha in sub.sha.unique()})
                    else:
                        reference = pd.Series({sha: human[sha]['metrics'][metric] for sha in sub.sha.unique()}, dtype=float)
                    by_song = sub.groupby('sha')[metric].mean()
                    diff = by_song - reference.reindex(by_song.index)
                    matched = diff.dropna().index
                    paired = sub[sub.sha.isin(matched)].pivot(index='sha', columns='seed', values=metric)
                    floor, pairs = code.ln_analysis.pair_spread(paired)
                    output.append(dict(system=system, arm=arm, sensitivity=sensitivity, source_class=cls, band=band,
                                       metric=metric, **code.ln_analysis.interval(diff),
                                       generated_mean=by_song.reindex(matched).mean(), human_mean=reference.reindex(matched).mean(),
                                       mean_absolute_gap=diff.abs().mean(), sampling_rms=floor, sampling_pairs=pairs,
                                       runs=len(sub), finite_runs=int(sub[metric].notna().sum()),
                                       degenerate_interval=bool(len(diff.dropna()) and diff.dropna().nunique()==1)))
    hazard_summary = []
    for (system, arm, age), group in hazard_frame.groupby(['system', 'arm', 'age_lo_ms']):
        for cls, band, sub in code.ln_analysis.strata(group, 'source_class'):
            diff = sub.groupby('sha')['diff'].mean()
            matched = sub[sub.sha.isin(diff.dropna().index)]
            floor, pairs = code.ln_analysis.pair_spread(matched.pivot(index='sha', columns='seed', values='rate'))
            hazard_summary.append(dict(system=system, arm=arm, source_class=cls, band=band, age_lo_ms=age,
                                       **code.ln_analysis.interval(diff), sampling_rms=floor, sampling_pairs=pairs,
                                       generated_rate=matched.groupby('sha').rate.mean().mean(),
                                       human_rate=matched.groupby('sha').human_rate.mean().mean(),
                                       risk=int(sub.risk.sum()), releases=int(sub.releases.sum()), runs=len(sub)))
    return frame, pd.DataFrame(output), hazard_frame, pd.DataFrame(hazard_summary)


def comparisons(frame, code):
    """Cross-system differences use the same songs, never unmatched panel means."""
    pairs = [(('r2', 'shipped'), ('mug', 'natural')), (('r2', 'source'), ('mug', 'source')),
             (('r2', 'shipped'), ('mapper', 'natural')), (('r2', 'source'), ('mapper', 'source')),
             (('r2', 'source'), ('mapper', 'source-inverse'))]
    output = []
    for (system_a, arm_a), (system_b, arm_b) in pairs:
        a = frame[(frame.system==system_a)&(frame.arm==arm_a)]
        b = frame[(frame.system==system_b)&(frame.arm==arm_b)]
        shared = set(a.sha) & set(b.sha)
        for cls, band, selected_a in code.ln_analysis.strata(a[a.sha.isin(shared)], 'source_class'):
            selected_b = b[b.sha.isin(selected_a.sha)]
            for metric in code.ln_analysis.METRICS:
                mean_a = selected_a.groupby('sha')[metric].mean()
                mean_b = selected_b.groupby('sha')[metric].mean()
                diff = mean_a - mean_b
                matched = diff.dropna().index
                floors = []
                counts = []
                for table in (selected_a, selected_b):
                    wide = table[table.sha.isin(matched)].pivot(index='sha', columns='seed', values=metric)
                    floor, n = code.ln_analysis.pair_spread(wide)
                    floors.append(floor); counts.append(n)
                output.append(dict(system_a=system_a, arm_a=arm_a, system_b=system_b, arm_b=arm_b,
                                   source_class=cls, band=band, metric=metric, **code.ln_analysis.interval(diff),
                                   mean_a=mean_a.reindex(matched).mean(), mean_b=mean_b.reindex(matched).mean(),
                                   sampling_rms_a=floors[0], sampling_pairs_a=counts[0],
                                   sampling_rms_b=floors[1], sampling_pairs_b=counts[1],
                                   complete_sampling_floor=all(np.isfinite(f) for f in floors)))
    return pd.DataFrame(output)


def legality_summary(records, code):
    rows = []
    for r in records:
        row = {k: r[k] for k in ('system', 'arm', 'sha', 'source_class', 'band')}
        row.update({k: int(r.get(k, False)) for k in ('parse', 'grammar', 'holds_closed', 'legal')})
        row.update({k: r.get(k, 0) for k in ('hazard_complete_holds', 'hazard_excluded_holds')})
        if r['system'] == 'mug':
            row.update({k: r['receipt'][k] for k in ('orphan_held_frames', 'held_lanes_at_array_end')})
            row['moved_or_removed_holds'] = r['receipt']['postprocess']['removed_or_moved_holds']
        if r['system'] == 'mapper':
            row['raw_stream_grammar'] = int(r['raw_stream']['grammar'])
            row['unpatched_legal'] = int(r['unpatched']['legal'])
            for version in ('unpatched', 'patched'):
                warnings = r['export_warnings'][version]
                row[version+'_incomplete_heads'] = sum('Incomplete hold note' in w for w in warnings)
                row[version+'_unmatched_ends'] = sum('Unmatched hold end' in w for w in warnings)
                row[version+'_untimed_groups'] = sum('without a timestamp' in w for w in warnings)
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = []
    for (system, arm), group in frame.groupby(['system', 'arm']):
        for cls, band, sub in code.ln_analysis.strata(group, 'source_class'):
            totals = sub.select_dtypes(include='number').drop(columns='band').sum(min_count=1).to_dict()
            output.append(dict(system=system, arm=arm, source_class=cls, band=band,
                               songs=sub.sha.nunique(), runs=len(sub), **totals))
    return pd.DataFrame(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    code = metric_code(a.root / 'stage1-code')
    records = inspect_runs(a.root, a.cache, code)
    code.separation.write_json(a.root / 'results/charts.json', records)
    if not records:
        print('No completed chart receipts')
        return
    human = {r['sha']: r for r in json.loads((a.root / 'results/human.json').read_text())}
    tables = summaries(records, human, code)
    for name, table in zip(('runs', 'effects', 'hazards', 'hazard-effects'), tables):
        table.to_csv(a.root / 'results' / (name + '.csv'), index=False)
    costs = []
    for (system, arm), group in tables[0].groupby(['system', 'arm']):
        for cls, band, sub in code.ln_analysis.strata(group, 'source_class'):
            costs.append(dict(system=system, arm=arm, source_class=cls, band=band,
                              songs=sub.sha.nunique(), runs=len(sub), legal=int(sub.legal.sum()),
                              wall_mean_s=sub.wall_seconds.mean(), wall_median_s=sub.wall_seconds.median(),
                              wall_min_s=sub.wall_seconds.min(), wall_max_s=sub.wall_seconds.max(),
                              peak_rss_max_gb=sub.peak_rss_gb.max()))
    pd.DataFrame(costs).to_csv(a.root / 'results/costs.csv', index=False)
    legality_summary(records, code).to_csv(a.root / 'results/legality.csv', index=False)
    comparisons(tables[0], code).to_csv(a.root / 'results/comparisons.csv', index=False)
    # Score the identical exported objects with the fixed human grid. This is
    # a timing sensitivity check; generation still receives no source timing.
    source_records = [dict(r, metrics=r['source_grid_metrics']) for r in records
                      if 'source_grid_metrics' in r]
    source_tables = summaries(source_records, human, code)
    source_tables[0].to_csv(a.root / 'results/source-grid-runs.csv', index=False)
    source_tables[1].to_csv(a.root / 'results/source-grid-effects.csv', index=False)
    comparisons(source_tables[0], code).to_csv(a.root / 'results/source-grid-comparisons.csv', index=False)
    print(dict(outputs=len(records), legal=dict(Counter(r['system'] for r in records if r['legal']))))


if __name__ == '__main__':
    main()
