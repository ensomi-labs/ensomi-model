"""Exploratory sensitivity tables for the gated SMC stage-2 analysis.

This does not fit a readout. It describes matched gap changes, conditional
recovery fractions, precision/recall and the strict canonical-band subset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from ensomi_n.smc import paired_bootstrap
from smc_analysis import write_csv
from smc_benchmark import require_gate


def recovery(gain, ceiling_gain):
    """Paired recovery fraction, conditional on the existing fold selections."""
    rng = np.random.default_rng(20261111)
    indices = rng.integers(len(gain), size=(10000, len(gain)))
    denominator = ceiling_gain[indices].mean(axis=1)
    boot = gain[indices].mean(axis=1) / np.maximum(denominator, 1e-12)
    return dict(gain=float(gain.mean()), ceiling_gain=float(ceiling_gain.mean()),
                fraction=float(gain.mean() / ceiling_gain.mean()),
                lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out
    require_gate(out, out / 'manifest.json')
    stage = out / 'stage2'
    rows = json.loads((out / 'scores/BT-cv.json').read_text())['rows']
    keys = [r['key'] for r in rows]
    manifest = {r['key']: r for r in json.loads((out / 'manifest.json').read_text())}
    records = json.loads((stage / 'level-records.json').read_text())
    by = {arm: {r['key']: r for r in records if r['arm'] == arm} for arm in {r['arm'] for r in records}}

    def values(arm):
        return np.array([by[arm][key]['F70'] for key in keys])

    raw = (values('N1-s0-full') + values('N1-s1-full')) / 2
    ceiling = (values('N1-s0-full-ceiling') + values('N1-s1-full-ceiling')) / 2
    ridge = (values('N1-s0-full-ridge') + values('N1-s1-full-ridge')) / 2
    bt, btc = values('BT-cv'), values('BT-cv-ceiling')
    masks = {'all': np.ones(len(rows), bool)}
    masks.update({band: np.array([r['tempo_band'] == band for r in rows])
                  for band in ('below77', 'in_band', 'above167')})
    result = dict(gap_change={}, recovery={}, level_choices={}, precision_recall=[])
    for group, mask in masks.items():
        result['gap_change'][group] = paired_bootstrap((ceiling - btc)[mask], (raw - bt)[mask])
        for seed in ('mean', 's0', 's1'):
            a, b, c = ((raw, ceiling, ridge) if seed == 'mean' else
                      tuple(values(f'N1-{seed}-full' + suffix) for suffix in ('', '-ceiling', '-ridge')))
            gain, denominator = (c - a)[mask], (b - a)[mask]
            if denominator.mean() <= .01:
                raise ValueError('ceiling gain too small for an interpretable recovery ratio')
            result['recovery'][seed + ':' + group] = recovery(gain, denominator)
        for arm in ('N1-s0-full-ceiling', 'N1-s1-full-ceiling', 'BT-cv-ceiling'):
            result['level_choices'][arm + ':' + group] = dict(Counter(
                by[arm][key]['action'] for key, keep in zip(keys, mask) if keep))

    difference = ceiling - btc
    low, inside = difference[masks['below77']], difference[masks['in_band']]
    rng = np.random.default_rng(20261111)
    boot = (low[rng.integers(len(low), size=(10000, len(low)))].mean(axis=1)
            - inside[rng.integers(len(inside), size=(10000, len(inside)))].mean(axis=1))
    result['low_minus_inband_ceiling_gap'] = dict(kind='diagnostic ceiling',
        difference=float(low.mean() - inside.mean()), lo=float(np.quantile(boot, .025)),
        hi=float(np.quantile(boot, .975)))
    for arm in ('BT-cv', 'BT-cv-ceiling', 'N1-s0-full', 'N1-s1-full',
                'N1-s0-full-ceiling', 'N1-s1-full-ceiling'):
        precision_recall = []
        for key in keys:
            ref = np.loadtxt(manifest[key]['annotation'], ndmin=1)
            ref = ref[ref >= 5]
            root = stage if arm.endswith('ceiling') else out
            pred = np.loadtxt(root / 'beats' / arm / (key + '.beats'), ndmin=1)
            pred = pred[pred >= 5]
            count = by[arm][key]['n_matches']
            precision_recall.append([count / len(pred) if len(pred) else 0., count / len(ref)])
        for group, mask in masks.items():
            precision, recall = np.array(precision_recall)[mask].mean(axis=0)
            result['precision_recall'].append(dict(arm=arm, group=group, n=int(mask.sum()),
                kind='diagnostic ceiling' if arm.endswith('ceiling') else 'raw',
                precision70=float(precision), recall70=float(recall)))

    with (stage / 'evidence-per-excerpt.csv').open() as handle:
        probes = list(csv.DictReader(handle))
    strict = [r for r in rows if 80 <= r['tempo_bpm'] < 160]
    result['strict_canonical_band'] = dict(n=len(strict), metrics={},
        excluded_from_inband=[r['key'] for r in rows if r['tempo_band'] == 'in_band'
                              and not 80 <= r['tempo_bpm'] < 160])
    for metric in ('at_beat', 'peak50', 'sharpness', 'advantage'):
        v = {arm: np.array([float(next(p[metric] for p in probes if p['key'] == r['key'] and p['arm'] == arm))
                           for r in strict]) for arm in ('BT-cv', 'N1-s0', 'N1-s1')}
        result['strict_canonical_band']['metrics'][metric] = paired_bootstrap(
            (v['N1-s0'] + v['N1-s1']) / 2, v['BT-cv'])
    strict_mask = np.array([80 <= r['tempo_bpm'] < 160 for r in rows])
    result['strict_canonical_band']['gap'] = paired_bootstrap(difference[strict_mask], np.zeros(len(strict)))
    (stage / 'sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')

    if (stage / 'decoder-records.json').exists():
        fixed = {variant: json.loads((stage / 'decoder-candidates' / f'BT-cv-{variant}.json').read_text())
                 for variant in ('fixed', 'band35', 'loose')}
        values = {variant: np.array([r[0]['F70'] for r in rr]) for variant, rr in fixed.items()}
        band_recovery = {}
        for group in ('all', 'below77'):
            mask = masks[group]
            gain = (values['band35'] - values['fixed'])[mask]
            loss = (bt - values['fixed'])[mask]
            fraction = recovery(gain, loss)
            band_recovery[group] = dict(gain=fraction['gain'], loss=fraction['ceiling_gain'],
                recovery=fraction['fraction'], lo=fraction['lo'], hi=fraction['hi'],
                band_vs_loose=paired_bootstrap(values['band35'][mask], values['loose'][mask]))
        (stage / 'BT-band-recovery.json').write_text(json.dumps(band_recovery, indent=2) + '\n')

    shortlist = []
    selections = [('largest residual gap diagnostic ceiling', np.argsort(ceiling - btc)[:6]),
                  ('N1 advantage diagnostic ceiling', np.argsort(ceiling - btc)[-4:][::-1]),
                  ('largest N1 level gain diagnostic ceiling', np.argsort(ceiling - raw)[-6:][::-1])]
    for kind, indices in selections:
        for i in indices:
            row, key = rows[i], keys[i]
            shortlist.append(dict(selection=kind, key=key, audio=manifest[key]['audio'],
                tempo_band=row['tempo_band'], bpm=row['tempo_bpm'], ibi_cv=row['ibi_cv'],
                N1_raw_F=raw[i], BT_raw_F=bt[i], N1_diagnostic_ceiling_F=ceiling[i],
                BT_diagnostic_ceiling_F=btc[i], s0_action=by['N1-s0-full-ceiling'][key]['action'],
                s1_action=by['N1-s1-full-ceiling'][key]['action'], tags=row['tag']))
    write_csv(stage / 'listening-shortlist.csv', shortlist)
    print('sensitivity tables complete')


if __name__ == '__main__':
    main()
