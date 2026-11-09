"""Two-training-seed replication of the fixed release-representation comparison.

Training-seed floors are RMS song differences, seed 2 minus seed 1, separately
for A0 and A1. Free-running sampling seeds are averaged before that subtraction.
The larger arm floor is retained for two-seed means and the unrepeated A2
contrast; it does not estimate A2's own training variance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd

from ln_analysis import METRICS, fr_floors, interval, pair_spread, rms, strata
from release_compare import A2_NOTE, paired_effect, read_arm
from separation import disk_guard

ARMS = ('A0', 'A1', 'A2', 'A0-s2', 'A1-s2')


def validate(arms):
    expected = {r['sha']: (r['source_class'], r['band'], r['source_head_rows']) for r in arms['A0']}
    keys = {(mode, kappa, seed) for mode in ('bos', 'prefix') for kappa in (0, 3)
            for seed in (954, 955, 956)}
    for arm in ARMS:
        rows = arms[arm]
        actual = {r['sha']: (r['source_class'], r['band'], r['source_head_rows']) for r in rows}
        if actual != expected or len(actual) != len(rows):
            raise ValueError(f'{arm}: chart panel or metadata differs')
        for row in rows:
            runs = row['runs']
            if len(runs) != 12 or {(r['mode'], r['kappa'], r['seed']) for r in runs} != keys:
                raise ValueError(f'{arm}: sampling protocol differs')
            if not all(r['legal'] for r in runs):
                raise ValueError(f'{arm}: illegal generated chart')


def contrasts(values):
    yield 'A1 - A0', 'seed1', values['A1'], values['A0'], ('A1', 'A0')
    yield 'A1 - A0', 'seed2', values['A1-s2'], values['A0-s2'], ('A1-s2', 'A0-s2')
    yield ('A1 - A0', 'mean_of_two_seeds', (values['A1'] + values['A1-s2']) / 2,
           (values['A0'] + values['A0-s2']) / 2, ('A0', 'A1', 'A0-s2', 'A1-s2'))
    yield 'A1 - A2', 'seed1_only', values['A1'], values['A2'], ('A1', 'A2')


def measure(values, base_floors, sampling, metadata):
    floors = dict(base_floors)
    training, deltas = [], []
    for arm in ('A0', 'A1'):
        difference = values[arm + '-s2'] - values[arm]
        spread, count = rms(difference)
        floors[arm + '_training_seed_rms'] = spread
        floors[arm + '_training_seed_pairs'] = count
        training.append(dict(metadata, arm=arm, training_seed_rms=spread, training_seed_pairs=count,
                             **interval(difference)))
        deltas += [dict(metadata, arm=arm, sha=sha, seed2_minus_seed1=value)
                   for sha, value in difference.items() if np.isfinite(value)]
    effects = []
    for comparison, training_seed, current, reference, relevant in contrasts(values):
        ff = dict(floors)
        for arm in relevant:
            ff[arm + '_sampling_rms'], ff[arm + '_sampling_pairs'] = sampling[arm]
        result = paired_effect(current, reference, ff)
        result['verdict'] = result['verdict'].replace('inference_floor', 'measured_floor')
        effects.append(dict(metadata, comparison=comparison, training_seed=training_seed, **result,
            interpretation=A2_NOTE + ' A2 training-seed variation is unmeasured; max(A0,A1) is a proxy.'
            if comparison == 'A1 - A2' else 'Two matched recipe seeds; song bootstrap conditions on both training seeds.'))
    return effects, training, deltas


def analyze_rows(arms, original, old):
    validate(arms)
    frames = {arm: pd.DataFrame(rows) for arm, rows in arms.items()}
    runs = {arm: pd.DataFrame([dict(sha=r['sha'], source_class=r['source_class'], band=r['band'], **v)
                              for r in rows for v in r['runs']]) for arm, rows in arms.items()}
    nll, fr, training, deltas = [], [], [], []
    for cls, band, sub in strata(frames['A0'], 'source_class'):
        ids = sub.sha
        cp = original[original.sha.isin(ids)].pivot(index='sha', columns='checkpoint_m', values='whole_chart_nll')
        spread, count = pair_spread(cp)
        values = {a: frame[frame.sha.isin(ids)].set_index('sha').whole_chart_nll for a, frame in frames.items()}
        meta = dict(source_class=cls, band=band, metric='whole_chart_nll', mode='tf', kappa=None)
        rows, tt, dd = measure(values,
            dict(sampling_rms=0., sampling_pairs=0, checkpoint_rms=spread, checkpoint_pairs=count),
            {a: (0., 0) for a in ARMS}, meta)
        nll += rows; training += tt; deltas += dd
        for mode in ('bos', 'prefix'):
            for kappa in (0, 3):
                selected = {a: r[r.sha.isin(ids) & (r['mode'] == mode) & (r.kappa == kappa)]
                            for a, r in runs.items()}
                control = old[old.sha.isin(ids) & (old['mode'] == mode) & (old.kappa == kappa)]
                for metric in METRICS:
                    values = {a: r.groupby('sha')[metric].mean() for a, r in selected.items()}
                    sampling = {a: pair_spread(r.pivot(index='sha', columns='seed', values=metric))
                                for a, r in selected.items()}
                    meta = dict(source_class=cls, band=band, metric=metric, mode=mode, kappa=kappa)
                    rows, tt, dd = measure(values, fr_floors(control, metric), sampling, meta)
                    fr += rows; training += tt; deltas += dd
    return dict(nll=pd.DataFrame(nll), fr=pd.DataFrame(fr),
                training_seed_floors=pd.DataFrame(training), training_seed_deltas=pd.DataFrame(deltas))


def analyze(root, stage1):
    arms = {a: read_arm(root / a) for a in ARMS}
    if any(not rows for rows in arms.values()):
        raise ValueError('All five arm evaluations must be complete')
    original = pd.read_csv(root / 'released-nll.csv')
    old = pd.DataFrame([json.loads(line) for p in sorted((stage1 / 'fr').glob('*.jsonl'))
                        for line in p.read_text().splitlines()])
    return analyze_rows(arms, original, old[old.arm == 'baseline'])


def report(data, command, output):
    lines = ['# Second training seed: A0 and A1', '',
        'Updated ' + datetime.now(timezone.utc).isoformat() + '.', '',
        'Both arms use the frozen 8M recipe, four CPU threads; seed 1 is initialization/data 171/471, '
        'seed 2 is 172/472. Evaluation is the same 48 dev songs, EOS, BOS/prefix, κ0/3 and sampling seeds 954/955/956. '
        'A2 is reused from seed 1. No per-decision likelihood is compared.', '',
        'Training-seed RMS is sqrt(mean over songs of (arm seed2 − arm seed1)^2), separately for A0 and A1. '
        'Sampling seeds are averaged first for free-running statistics. This first pair changes initialization and data order together; '
        'it does not separate their contributions or establish a population estimate of training variance. '
        'Effect/floor uses the largest checkpoint RMS, compared-model sampling RMS, and either arm’s training-seed RMS. '
        'The two-seed mean retains the full measured floor. Its 2,000 song bootstrap (20261108) conditions on the two observed training seeds; '
        'songs are paired, with no doubling of the song count.', '',
        '## First measured training-seed variation', '',
        '| Arm | Class | Songs | Whole-chart training-seed RMS, nats | Mean seed2 − seed1, nats |',
        '| --- | --- | ---: | ---: | ---: |']
    floors = data['training_seed_floors']
    for _, r in floors[(floors.band == 'all') & (floors.metric == 'whole_chart_nll')].iterrows():
        lines.append(f'| {r.arm} | {r.source_class} | {r.training_seed_pairs} | {r.training_seed_rms:.2f} | {r.effect:+.2f} |')
    lines += ['', '## Joint-ownership contrast and next-head contrast', '',
        '**A2 qualification for this table:** ' + A2_NOTE + ' A2 has no second training seed; the max(A0,A1) training floor is a proxy.', '',
        '| Contrast | Training seed | Class | Songs | Whole-chart ΔNLL [95% CI] | Effect / largest floor |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for _, r in data['nll'][data['nll'].band == 'all'].iterrows():
        lines.append(f'| {r.comparison} | {r.training_seed} | {r.source_class} | {r.songs} | {r.effect:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | {r.effect_over_floor:.2f} |')
    lines += ['', '**A2 qualification for the following table:** ' + A2_NOTE + ' Its training floor is the measured A0/A1 proxy.', '',
        'Primary free running: first-third prefix, κ=0. Differences below are percentage points. '
        'All eight measures, BOS, κ3 portability, bands, each floor and every pair count are in the CSVs.', '',
        '| Contrast | Training seed | Class | Measure | Songs | Δ, pp [95% CI] | Effect / largest floor |',
        '| --- | --- | --- | --- | ---: | ---: | ---: |']
    fr = data['fr']
    selected = fr[(fr.band == 'all') & (fr['mode'] == 'prefix') & (fr.kappa == 0) &
                  fr.metric.isin(['ln_share', 'inverse_gap_share'])]
    for _, r in selected.iterrows():
        lines.append(f'| {r.comparison} | {r.training_seed} | {r.source_class} | {r.metric} | {r.songs} | {100*r.effect:+.2f} [{100*r.lo:+.2f}, {100*r.hi:+.2f}] | {r.effect_over_floor:.2f} |')
    lines += ['', 'These measures characterize behavior on supplied human rows; they are not human quality judgments. '
        'The inverse stratum has three songs, and no inverse–inverse human null pair exists. '
        'Whole-chart NLL includes EOS. κ0 is primary; κ3 applies controls after releases in A1 and is the existing decoder-portability diagnostic.', '',
        'Compact results: ' + str(output) + '. `training_seed_floors.csv` and `training_seed_deltas.csv` give the separate arm floors and song deltas; '
        '`nll.csv` and `fr.csv` give seed-1, seed-2, averaged A1−A0, and the seed-1-only A1−A2 contrast with updated ratios.', '',
        'Exact command: `' + command + '`.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('results', 'stage1', 'out', 'report'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    disk_guard(root)
    data = analyze(Path(args.results), Path(args.stage1))
    for name, table in data.items():
        table.to_csv(root / (name + '.csv'), index=False)
    text = report(data, shlex.join(sys.argv), root)
    (root / 'REPORT.md').write_text(text)
    target = Path(args.report)
    disk_guard(target.parent)
    target.write_text(text)
    (root / 'COMPLETE.json').write_text(json.dumps(dict(arms=list(ARMS)), indent=2) + '\n')
    print(json.dumps(dict(nll=len(data['nll']), fr=len(data['fr']))), flush=True)


if __name__ == '__main__':
    main()
