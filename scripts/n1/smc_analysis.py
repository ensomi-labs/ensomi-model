"""Matched SMC tables, descriptive strata and paired excerpt uncertainty.

Consumes smc_benchmark.py score files. No parameter or checkpoint is selected
here. Global shift optima are descriptive; held-out fold shifts are separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np

from ensomi_n.smc import METRICS, SHIFTS_MS, heldout_shifts, paired_bootstrap


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tags_and_confidence(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    codes = [line for line in lines if re.fullmatch(r'[a-z][1-4]', line)]
    if len(codes) != 1:
        raise ValueError(f'expected one annotator/confidence code: {text!r}')
    tags = {(line[1:-1] if line.startswith('(') and line.endswith(')') else line).lower()
            for line in lines if line not in codes and line != 'none'}
    return tags, codes[0][1], codes[0][0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    out = a.out
    if not json.loads((out / 'gate.json').read_text())['passed']:
        raise RuntimeError('BT-cv gate has not passed')
    scores = {path.stem: json.loads(path.read_text()) for path in sorted((out / 'scores').glob('*.json'))}
    reference = scores['BT-cv']['rows']
    keys = [r['key'] for r in reference]
    folds = [r['fold'] for r in reference]
    if len(keys) != 217 or len(set(keys)) != 217:
        raise ValueError('expected 217 unique excerpts')
    for name, d in scores.items():
        if [r['key'] for r in d['rows']] != keys:
            raise ValueError(f'{name}: excerpt order differs')
    values = {name: {k: np.array([r['metrics'][k] for r in d['rows']]) for k in METRICS}
              for name, d in scores.items()}
    # Average the two seeds within each excerpt, preserving paired sampling.
    for variant in ('full', 'peaks', 'canonical', 'f1decode', 'beatonly'):
        values['N1-mean-' + variant] = {
            k: (values['N1-s0-' + variant][k] + values['N1-s1-' + variant][k]) / 2 for k in METRICS}
    strata = {'all': np.ones(len(keys), dtype=bool)}
    for field in ('tempo_band', 'variability'):
        for label in sorted({r[field] for r in reference}):
            strata[label] = np.array([r[field] == label for r in reference])
    strata['outside_band'] = np.array([r['tempo_band'] != 'in_band' for r in reference])
    strata['easy'] = np.array([r['easy'] for r in reference])
    strata['hard'] = ~strata['easy']
    parsed = [tags_and_confidence(r['tag']) for r in reference]
    for tag in sorted({t for ts, _, _ in parsed for t in ts}):
        strata['tag:' + tag] = np.array([tag in ts for ts, _, _ in parsed])
    for code in sorted({c for _, c, _ in parsed}):
        strata['confidence:' + code] = np.array([c == code for _, c, _ in parsed])
    for annotator in sorted({an for _, _, an in parsed}):
        strata['annotator:' + annotator] = np.array([an == annotator for _, _, an in parsed])

    table, shift_table, flat, curve_table = [], [], [], []
    for name, d in scores.items():
        table.append(dict(arm=name, reading='raw', **d['mean']))
        if 'heldout_shift' in d:
            table.append(dict(arm=name, reading='heldout_shift', **d['heldout_shift']['mean']))
        curves = np.asarray(d['shift_curves'])
        curve = curves.mean(axis=0)
        curve_table.extend(dict(arm=name, shift_ms=int(shift), F=float(value))
                           for shift, value in zip(SHIFTS_MS, curve))
        order = sorted(range(len(SHIFTS_MS)), key=lambda i: (abs(SHIFTS_MS[i]), SHIFTS_MS[i]))
        best = max(order, key=lambda i: curve[i])
        heldout = heldout_shifts(curves, folds)
        indices = np.searchsorted(SHIFTS_MS, heldout)
        cv_f = curves[np.arange(len(keys)), indices]
        values[name]['heldout_shift_F'] = cv_f
        values[name]['best_shift_F'] = curves[:, best]
        shift_table.append(dict(arm=name, best_shift_ms=int(SHIFTS_MS[best]),
                                best_F=float(curve[best]), raw_F=d['mean']['F-measure'],
                                descriptive_gain=float(curve[best] - d['mean']['F-measure']),
                                heldout_F=float(cv_f.mean()),
                                fold_shifts=json.dumps({f: float(heldout[np.array(folds) == f][0]) for f in range(8)})))
        per_arm = []
        for i, row in enumerate(d['rows']):
            entry = {k: v for k, v in row.items() if k not in ('tag', 'metrics')}
            entry.update(arm=name, tag=row['tag'].replace('\n', '; '), **row['metrics'],
                         heldout_shift_ms=float(heldout[i]), heldout_shift_F=float(cv_f[i]))
            per_arm.append(entry)
            flat.append(entry)
        write_csv(out / 'scores' / f'{name}.csv', per_arm)
    write_csv(out / 'summary.csv', table)
    write_csv(out / 'shifts.csv', shift_table)
    write_csv(out / 'global-shift-curves.csv', curve_table)
    write_csv(out / 'per-excerpt.csv', flat)
    strata_table = [dict(stratum=label, n=int(mask.sum()), arm=name,
                         **{k: float(v[k][mask].mean()) for k in METRICS})
                    for label, mask in strata.items() for name, v in values.items()]
    write_csv(out / 'strata.csv', strata_table)
    failures = {name: dict(Counter(r['failure'] for r in d['rows'])) for name, d in scores.items()}
    ratios = {name: dict(Counter(r['ratio_class'] + ('_offbeat' if r['offbeat'] else '') for r in d['rows']))
              for name, d in scores.items()}

    pairs = [('N1-mean-full', 'BT-cv'), ('N1-s0-full', 'BT-cv'), ('N1-s1-full', 'BT-cv'),
             ('N1-s0-full', 'N1-s1-full'), ('N1-s0-full', 'N1-s0-prev-full'),
             ('N1-s1-full', 'N1-s1-prev-full'), ('BT-final0', 'BT-cv'),
             ('R2-timing', 'BT-final0'), ('BT-cv-n1decode', 'BT-cv'),
             ('N1-mean-canonical', 'BT-cv-n1decode'),
             ('N1-mean-beatonly', 'BT-cv-n1decode'), ('N1-mean-peaks', 'BT-cv')]
    for seed in ('s0', 's1', 'mean'):
        for variant in ('peaks', 'canonical', 'f1decode', 'beatonly'):
            pairs.append((f'N1-{seed}-{variant}', f'N1-{seed}-full'))
        pairs.append((f'N1-{seed}-canonical', f'N1-{seed}-peaks'))
    intervals = []
    main_strata = ['all', 'below77', 'in_band', 'above167', 'outside_band', 'steady', 'middle', 'high', 'easy', 'hard']
    for left, right in pairs:
        for label in main_strata:
            if label not in strata or not strata[label].any():
                continue
            mask = strata[label]
            for metric in ('F-measure', 'CMLt', 'AMLt'):
                intervals.append(dict(left=left, right=right, stratum=label, metric=metric,
                                      **paired_bootstrap(values[left][metric][mask], values[right][metric][mask])))
    # The AML-CML difference is paired within excerpts before comparing systems.
    for name in ('N1-s0-full', 'N1-s1-full', 'N1-mean-full'):
        for label in main_strata:
            if label not in strata or not strata[label].any():
                continue
            mask = strata[label]
            intervals.append(dict(left=name, right='BT-cv', stratum=label, metric='AMLt-CMLt',
                                  **paired_bootstrap((values[name]['AMLt'] - values[name]['CMLt'])[mask],
                                                     (values['BT-cv']['AMLt'] - values['BT-cv']['CMLt'])[mask])))
    for name in scores:
        intervals.append(dict(left=name + '+heldout_shift', right=name, stratum='all', metric='F-measure',
                              **paired_bootstrap(values[name]['heldout_shift_F'], values[name]['F-measure'])))
    for seed in ('s0', 's1'):
        name = f'N1-{seed}-full'
        for correction in ('heldout_shift_F', 'best_shift_F'):
            intervals.append(dict(left=name, right='BT-cv', stratum='all', metric=correction + '_gain',
                                  **paired_bootstrap(values[name][correction] - values[name]['F-measure'],
                                                     values['BT-cv'][correction] - values['BT-cv']['F-measure'])))
    # Tag/confidence comparisons remain descriptive, with no multiple-testing claim.
    for label, mask in strata.items():
        if label.startswith(('tag:', 'confidence:')):
            intervals.append(dict(left='N1-mean-full', right='BT-cv', stratum=label, metric='F-measure',
                                  **paired_bootstrap(values['N1-mean-full']['F-measure'][mask],
                                                     values['BT-cv']['F-measure'][mask])))
    write_csv(out / 'paired.csv', intervals)
    contrasts = {}
    gap = (values['N1-mean-full']['AMLt'] - values['N1-mean-full']['CMLt']
           - values['BT-cv']['AMLt'] + values['BT-cv']['CMLt'])
    for label, delta, first, second in [
        ('P3_gap_outside_minus_in', gap, 'outside_band', 'in_band'),
        ('P2_peak_gain_high_minus_steady', values['N1-mean-peaks']['F-measure'] - values['N1-mean-full']['F-measure'],
         'high', 'steady'),
        ('P1_gap_below_minus_in', values['N1-mean-full']['F-measure'] - values['BT-cv']['F-measure'],
         'below77', 'in_band'),
    ]:
        x, y = delta[strata[first]], delta[strata[second]]
        rng = np.random.default_rng(20261111)
        # Resample excerpts within each stratum; both systems stay paired.
        boot = x[rng.integers(len(x), size=(10000, len(x)))].mean(1) - y[rng.integers(len(y), size=(10000, len(y)))].mean(1)
        lo, hi = np.quantile(boot, [.025, .975])
        contrasts[label] = dict(difference=float(x.mean() - y.mean()), lo=float(lo), hi=float(hi),
                                first=first, second=second, n_first=len(x), n_second=len(y))
    (out / 'interactions.json').write_text(json.dumps(contrasts, indent=2))
    (out / 'diagnostics.json').write_text(json.dumps(dict(failures=failures, ratios=ratios,
                                                        strata_counts={k: int(v.sum()) for k, v in strata.items()}), indent=2))
    difference = values['N1-mean-full']['F-measure'] - values['BT-cv']['F-measure']
    shortlist = []
    manifest = {r['key']: r for r in json.loads((out / 'manifest.json').read_text())}
    for direction, selected in [('N1_better', np.argsort(-difference)[:5]), ('BT_better', np.argsort(difference)[:5])]:
        for i in selected:
            shortlist.append(dict(direction=direction, key=keys[i], file=manifest[keys[i]]['audio'],
                                  N1_s0_F=float(values['N1-s0-full']['F-measure'][i]),
                                  N1_s1_F=float(values['N1-s1-full']['F-measure'][i]),
                                  BT_cv_F=float(values['BT-cv']['F-measure'][i]),
                                  N1_ratio=scores['N1-s0-full']['rows'][i]['ratio_class'],
                                  BT_ratio=reference[i]['ratio_class'],
                                  N1_CMLt=float(values['N1-mean-full']['CMLt'][i]), BT_CMLt=reference[i]['metrics']['CMLt'],
                                  tempo_bpm=reference[i]['tempo_bpm'], ibi_cv=reference[i]['ibi_cv'],
                                  tags=reference[i]['tag'].replace('\n', '; ')))
    write_csv(out / 'listening-shortlist.csv', shortlist)
    print('tables written; all differences are paired over excerpts', flush=True)


if __name__ == '__main__':
    main()
