"""Paired chart-space comparisons with released inference floors.

Reads completed release_evaluate outputs, released whole-chart likelihoods,
and stage-1 free runs. A0 versus released 56M comes first. A1 versus A2 is a
conservative feature-blind common-support comparison: candidate scheduling
and completion still leak next-head information; a tie is inconclusive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ln_analysis import METRICS, fr_floors, interval, pair_spread, scaled, strata
from release_leaks import summarize
from separation import disk_guard

A2_NOTE = ('A2 is the feature-blind common-support control. Future-dependent opportunity scheduling '
           'and full-occupancy completion remain. An A1 advantage is interpreted conservatively as '
           'a lower bound on the benefit of next-head knowledge; a tie is inconclusive. '
           'This is an information comparison, not a formal finite-sample or optimization guarantee.')


def read_arm(path):
    if not (path / 'COMPLETE').exists():
        return []
    return [v for p in sorted(path.glob('*.json')) if 'whole_chart_nll' in (v := json.loads(p.read_text()))]


def paired_effect(current, reference, floors):
    pairs = pd.concat([current.rename('current'), reference.rename('reference')], axis=1)
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    return dict(current=float(pairs.current.mean()), reference=float(pairs.reference.mean()),
                **scaled(interval(pairs.current - pairs.reference), floors))


def analyze(root, stage1):
    arms = {a: read_arm(root / a) for a in ('A0', 'A1', 'A2')}
    arms = {a: rows for a, rows in arms.items() if rows}
    original = pd.read_csv(root / 'released-nll.csv')
    frames = {'released56': original[original.checkpoint_m == 56]}
    frames.update({a: pd.DataFrame(rows) for a, rows in arms.items()})
    comparisons = [(a, b) for a, b in [('A0', 'released56'), ('A1', 'A0'), ('A1', 'A2')]
                  if a in frames and b in frames]
    nll = []
    for a, b in comparisons:
        for cls, band, sub in strata(frames[a], 'source_class'):
            ref = frames[b][frames[b].sha.isin(sub.sha)]
            cp = original[original.sha.isin(sub.sha)].pivot(index='sha', columns='checkpoint_m', values='whole_chart_nll')
            floor, count = pair_spread(cp)
            effect = paired_effect(sub.set_index('sha').whole_chart_nll,
                ref.set_index('sha').whole_chart_nll,
                dict(sampling_rms=0., sampling_pairs=0, checkpoint_rms=floor, checkpoint_pairs=count))
            nll.append(dict(comparison=f'{a} - {b}', source_class=cls, band=band, **effect,
                            interpretation=A2_NOTE if b == 'A2' else 'Matched whole-chart NLL including EOS'))
    old = pd.DataFrame([json.loads(line) for p in sorted((stage1 / 'fr').glob('*.jsonl'))
                        for line in p.read_text().splitlines()])
    old = old[old.arm == 'baseline']
    runs = {'released56': old[old.checkpoint == 56]}
    for arm, rows in arms.items():
        runs[arm] = pd.DataFrame([dict(sha=r['sha'], source_class=r['source_class'], band=r['band'], **v)
                                  for r in rows for v in r['runs']])
    fr = []
    for a, b in comparisons:
        if runs[a].empty or runs[b].empty:
            continue
        for (mode, kappa), selected in runs[a].groupby(['mode', 'kappa']):
            for cls, band, sub in strata(selected, 'source_class'):
                ref = runs[b][(runs[b]['mode'] == mode) & (runs[b].kappa == kappa) & runs[b].sha.isin(sub.sha)]
                control = old[(old['mode'] == mode) & (old.kappa == kappa) & old.sha.isin(sub.sha)]
                for metric in METRICS:
                    floors = fr_floors(control, metric)
                    for name, frame in [('current', sub), ('reference', ref)]:
                        value, count = pair_spread(frame.pivot(index='sha', columns='seed', values=metric))
                        floors[name + '_sampling_rms'] = value
                        floors[name + '_sampling_pairs'] = count
                    effect = paired_effect(sub.groupby('sha')[metric].mean(), ref.groupby('sha')[metric].mean(), floors)
                    fr.append(dict(comparison=f'{a} - {b}', source_class=cls, band=band, mode=mode,
                        kappa=kappa, metric=metric, **effect, interpretation=A2_NOTE if b == 'A2' else
                        ('Primary representation comparison' if kappa == 0 else 'Decoder-portability diagnostic')))
    leaks, hazards, human_hazards, tails = [], [], [], []
    if arms:
        for r in next(iter(arms.values())):
            for human in r.get('human', []):
                for h in human['duration_hazards']:
                    human_hazards.append(dict(sha=r['sha'], source_class=r['source_class'], band=r['band'],
                                              mode=human['mode'], **h))
    for arm, rows in arms.items():
        for r in rows:
            for run in r['runs']:
                meta = dict(arm=arm, sha=r['sha'], source_class=r['source_class'], band=r['band'],
                            mode=run['mode'], kappa=run['kappa'], seed=run['seed'],
                            interpretation=A2_NOTE if arm == 'A2' else 'Common-chart descriptive counts')
                leaks.append(dict(meta, **run['leaks'], actual_full_forces=run['forced_full_occupancy'],
                                  actual_eos_forces=run['forced_eos']))
                for h in run['duration_hazards']:
                    hazards.append(dict(meta, **h))
                tails.append(dict(meta, eos_releases=run['eos_releases'], legal=run['legal'],
                                  fallbacks=run['fallbacks'], eos_offsets_ms=json.dumps(run['eos_release_offset_from_last_head_ms'])))
    leak_summaries = []
    if leaks:
        table = pd.DataFrame(leaks)
        for (arm, kappa), sub in table.groupby(['arm', 'kappa']):
            summary = summarize(sub)
            # summarize counts chart/seed records; report both denominators.
            summary = summary.rename(columns={'songs': 'chart_seed_runs'})
            summary['songs'] = summary.chart_seed_runs // 3
            summary['arm'], summary['kappa'] = arm, kappa
            summary['interpretation'] = A2_NOTE if arm == 'A2' else 'Counterfactual two-order replay on generated charts'
            for index, row in summary.iterrows():
                selected = sub[sub['mode'] == row['mode']]
                if row.source_class != 'all':
                    selected = selected[selected.source_class == row.source_class]
                if row.band != 'all':
                    selected = selected[selected.band == row.band]
                actual = selected.actual_full_forces.sum(min_count=1)
                summary.loc[index, 'actual_full_forces'] = actual
                summary.loc[index, 'actual_full_force_per_gap'] = actual / row.head_gaps if row.head_gaps else np.nan
                summary.loc[index, 'actual_full_force_per_full_gap'] = actual / row.full_occupancy_gaps if row.full_occupancy_gaps else np.nan
                summary.loc[index, 'actual_eos_forces'] = selected.actual_eos_forces.sum(min_count=1)
            leak_summaries.append(summary)
    return dict(arms=list(arms), nll=pd.DataFrame(nll), fr=pd.DataFrame(fr),
                leaks=pd.DataFrame(leaks), leak_summary=pd.concat(leak_summaries, ignore_index=True) if leak_summaries else pd.DataFrame(),
                duration_hazards=pd.DataFrame(hazards), human_duration_hazards=pd.DataFrame(human_hazards),
                eos=pd.DataFrame(tails))


def report(data, command):
    lines = ['# Short-budget release representation results', '',
             'Updated ' + datetime.now(timezone.utc).isoformat() + '.', '',
             'Completed evaluation arms: ' + ', '.join(data['arms']) + '.', '',
             '**A2 qualification:** ' + A2_NOTE, '',
             'All comparisons use matched songs and whole-chart likelihood through EOS. Per-decision NLL is never compared across arms. '
             'The larger of released checkpoint RMS and all available sampling RMS floors scales each effect. '
             'Intervals use 2,000 song bootstraps, seed 20261108. These one-training-seed measurements characterize behavior, not human quality.', '',
             '## A0 at the short budget, then the representation comparisons', '',
             'A0 minus released 56M is the short-training cost. A1 minus A0 changes joint ownership. '
             'A1 minus A2 tests next-head knowledge under the A2 qualification above. Negative NLL differences favor the first arm.', '',
             '| Comparison | Class | Songs | Whole-chart ΔNLL [95% CI] | Checkpoint RMS | Effect / floor |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    nll = data['nll']
    if not nll.empty:
        for _, r in nll[nll.band == 'all'].iterrows():
            lines.append(f'| {r.comparison} | {r.source_class} | {r.songs} | {r.effect:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | {r.checkpoint_rms:.2f} | {r.effect_over_floor:.2f} |')
    lines += ['', '**A2 qualification for the following table:** ' + A2_NOTE, '',
              'Primary free-running comparison: first-third prefix, κ=0. Differences are percentage points; '
              'closeness to source statistics must also be read from the saved human/run records. '
              'BOS, κ=3 portability, all eight style measures and class/band strata are in `fr.csv`.', '',
              '| Comparison | Class | Measure | Songs | Δ, pp [95% CI] | Effect / floor |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    fr = data['fr']
    if not fr.empty:
        selected = fr[(fr.band == 'all') & (fr['mode'] == 'prefix') & (fr.kappa == 0) &
                      fr.metric.isin(['ln_share', 'inverse_gap_share'])]
        for _, r in selected.iterrows():
            lines.append(f'| {r.comparison} | {r.source_class} | {r.metric} | {r.songs} | {100*r.effect:+.2f} [{100*r.lo:+.2f}, {100*r.hi:+.2f}] | {r.effect_over_floor:.2f} |')
    lines += ['', 'The source human–human nulls and counts remain in the stage-1 report: inverse–inverse pairs are absent, '
              'and the LN-heavy different-creator null spans only one song. They do not provide a broad inverse equivalence test.', '',
              '`leaks.csv` retains actual full-occupancy/EOS forces separately from the common-chart two-order replay. '
              '`leak_summary.csv` gives class/band replay and schedule rates. The schedule proxy is fixed by the supplied head rows; '
              'held-gap exposure depends on generated history. `duration_hazards.csv` uses complete generated holds born in the evaluated '
              'suffix, not human teacher-forced histories; source duration bins are in `human_duration_hazards.csv`. '
              'Those marginal hazards and `eos.csv` are descriptive; '
              'new hazard-bin and EOS checkpoint floors have not been generated. No conditional hazard rates with different decision units are compared.', '',
              'Exact analysis command: `' + command + '`.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['results', 'stage1', 'out']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--report', help='Also update an external report after writing the compact results')
    args = parser.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    disk_guard(root)
    data = analyze(Path(args.results), Path(args.stage1))
    for name, value in data.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(root / (name + '.csv'), index=False)
    import shlex, sys
    text = report(data, shlex.join(sys.argv))
    (root / 'REPORT.md').write_text(text)
    if args.report:
        target = Path(args.report)
        disk_guard(target.parent)
        target.write_text(text + '\nCompact result directory: ' + str(root.resolve()) + '.\n')
    (root / 'COMPLETE.json').write_text(json.dumps(dict(arms=data['arms']), indent=2) + '\n')
    print(json.dumps(dict(arms=data['arms'], nll=len(data['nll']), fr=len(data['fr']))), flush=True)


if __name__ == '__main__':
    main()
