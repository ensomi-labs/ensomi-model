"""SMC level ceilings, fold-selected decoder probes and adjusted tag associations.

Consumes the gated stage-1 output. All generated files belong under --out/stage2;
annotation-assisted rows are labelled diagnostic ceilings in every result table.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.special import expit

from ensomi_n.smc import map_beats, minimal_peaks, paired_bootstrap
from ensomi_n.smc_diagnosis import (LEVEL_ACTIONS, bootstrap_ols, evidence_probes,
    level_remappings, own_features, ridge_readout, select_folds, timing_metrics)
from smc_analysis import tags_and_confidence, write_csv
from smc_benchmark import require_gate

CEILING_ARMS = ('BT-cv', 'BT-cv-n1decode', 'N1-s0-full', 'N1-s1-full', 'N1-s0-peaks', 'N1-s1-peaks')
METRICS = ('F20', 'F35', 'F50', 'F70', 'CMLt', 'AMLt')
VARIANTS = {
    'fixed': {}, 'band55': {'bpm_lo': 55.}, 'band35': {'bpm_lo': 35.},
    'free-half': {'allow_half_free': True},
    'loose': {'sigma_cont': .08, 'step_penalty': 3., 'reset_penalty': 4.},
    'band35-loose': {'bpm_lo': 35., 'sigma_cont': .08, 'step_penalty': 3., 'reset_penalty': 4.},
    'freer': {'sigma_cont': .08, 'step_penalty': 3., 'reset_penalty': 4.,
              'fit_tol_ms': 0., 'min_constant_beats': 4, 'merge_phase_ms': 0.},
    'band35-free-half': {'bpm_lo': 35., 'allow_half_free': True},
}
# Source labels are retained in tag-mapping.json, including spelling variants.
TAG_PATTERNS = {
    'expressive timing': ('expressive timing',), 'missing bass': ('missing bass',),
    'weak transients': ('lack of transient',), 'slow tempo tag': ('slow tempo',),
    'gradual tempo changes': ('gradual tempo',), 'quiet accompaniment': ('quiet accompaniment',),
    'tempo discontinuities': ('tempo incontinu', 'tempo discontinu'),
    'pauses': ('pause',), 'ternary meter': ('ternary',), 'syncopation': ('syncopation',),
    'poor recording quality': ('poor',), 'rich ornamentation': ('rich ornamentation',),
    'low familiarity': ('low familiarity',), 'meter changes': ('changing time signature',),
    'metrical ambiguity': ('ambiguity of metrical', 'ambiguity in metrical'),
    'phase ambiguity': ('beat phase ambiguity',), 'wide dynamics': ('wide dynamic',),
    'intro': ('introduction of',), 'lack of harmonic change': ('lack of harmonic',),
    'no repetition': ('no repetition',),
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def readbeats(path):
    return np.loadtxt(path, ndmin=1)


def save_sequence(stage, arm, key, beats):
    p = stage / 'beats' / arm / (key + '.beats')
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, beats, fmt='%.10f')


def record(arm, kind, row, metrics, action='unchanged'):
    return dict(arm=arm, kind=kind, key=row['key'], fold=row['fold'], action=action, **metrics)


def levels(out, manifest, rows):
    stage = out / 'stage2'
    records, candidate_records, fits = [], [], {}
    folds = np.array([r['fold'] for r in rows])
    for arm in sorted(p.name for p in (out / 'beats').iterdir() if p.is_dir()):
        candidates, features, sequences = [], [], []
        for row in rows:
            key = row['key']
            r, p = readbeats(manifest[key]['annotation']), readbeats(out / 'beats' / arm / (key + '.beats'))
            records.append(record(arm, 'raw', row, timing_metrics(r, p)))
            if arm not in CEILING_ARMS:
                continue
            family = level_remappings(p)
            metrics = [timing_metrics(r, q) for q in family]
            best = int(np.argmax([m['F70'] for m in metrics]))
            records.append(record(arm + '-ceiling', 'diagnostic ceiling', row, metrics[best], LEVEL_ACTIONS[best]))
            save_sequence(stage, arm + '-ceiling', key, family[best])
            candidate_records.extend(record(arm, 'diagnostic ceiling candidate', row, m, action)
                                     for m, action in zip(metrics, LEVEL_ACTIONS))
            candidates.append(metrics)
            sequences.append(family)
            if arm in ('N1-s0-full', 'N1-s1-full'):
                with np.load(out / 'activations' / arm[:5] / (key + '.npz')) as z:
                    features.append(own_features(p, z))
        if features:
            utility = np.array([[m['F70'] for m in mm] for mm in candidates])
            chosen, fits[arm] = ridge_readout(features, utility, folds)
            for mode, choices in [('ridge', chosen), ('global', select_folds(utility, folds))]:
                for i, (row, choice) in enumerate(zip(rows, choices)):
                    name = arm + '-' + mode
                    records.append(record(name, 'tuned on SMC folds', row, candidates[i][choice], LEVEL_ACTIONS[choice]))
                    save_sequence(stage, name, row['key'], sequences[i][choice])
            dump(stage / (arm + '-features.json'), dict(keys=[r['key'] for r in rows], features=np.array(features).tolist()))
        print('levels', arm, flush=True)
    dump(stage / 'level-records.json', records)
    dump(stage / 'readout-fits.json', fits)
    write_csv(stage / 'level-per-excerpt.csv', records)
    write_csv(stage / 'level-candidates.csv', candidate_records)


def decode_job(job):
    out, arm, name, row, annotation = job
    from ensomi_n.v0.decode import DecodeConfig, Evidence, decode
    from ensomi_n.v0.evaluate import bt_evidence, v0_evidence
    source = arm.removesuffix('-beatonly')
    path = out / 'activations' / source / (row['key'] + '.npz')
    if arm == 'BT-cv':
        ev = bt_evidence(path, 0., 4000)
    else:
        with np.load(path) as z:
            ev = Evidence(beat=z['beat'], down=z['down']) if arm.endswith('-beatonly') else v0_evidence(dict(z))
    pm, diag = decode(ev, replace(DecodeConfig(reset_penalty=14.), **VARIANTS[name]))
    beats, outside = map_beats(pm, 40.)
    save_sequence(out / 'stage2', arm + '-' + name, row['key'], beats)
    return record(arm, 'decoder candidate', row, timing_metrics(readbeats(annotation), beats), name), diag, outside


def decoders(out, manifest, rows, workers):
    stage, folds = out / 'stage2', np.array([r['fold'] for r in rows])
    records, candidates, diagnostics = [], [], []
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for arm in ('BT-cv', 'N1-s0', 'N1-s1'):
            per = {}
            for name in VARIANTS:
                checkpoint = stage / 'decoder-candidates' / f'{arm}-{name}.json'
                if checkpoint.exists():
                    results = json.loads(checkpoint.read_text())
                else:
                    jobs = [(out, arm, name, row, manifest[row['key']]['annotation']) for row in rows]
                    results = list(pool.map(decode_job, jobs, chunksize=4))
                    dump(checkpoint, results)
                per[name] = [r[0] for r in results]
                candidates.extend(per[name])
                diagnostics.extend(dict(arm=arm, variant=name, key=row['key'], outside_audio=r[2], **r[1])
                                   for row, r in zip(rows, results))
                print('decoder', arm, name, flush=True)
            families = {'decoder-all': list(VARIANTS), 'decoder-band': ['fixed', 'band55', 'band35'],
                        'decoder-loose': ['fixed', 'loose'], 'decoder-half': ['fixed', 'free-half']}
            for label, names in families.items():
                utility = np.array([[per[name][i]['F70'] for name in names] for i in range(len(rows))])
                chosen = select_folds(utility, folds)
                for i, (row, choice) in enumerate(zip(rows, chosen)):
                    source = per[names[choice]][i]
                    result = dict(source, arm=arm + '-' + label, kind='tuned on SMC folds')
                    records.append(result)
                    beats = readbeats(stage / 'beats' / (arm + '-' + names[choice]) / (row['key'] + '.beats'))
                    save_sequence(stage, result['arm'], row['key'], beats)
            # Order fixes ties by distance from .5, then by lower threshold.
            thresholds = sorted([i / 10 for i in range(1, 10)], key=lambda t: (round(abs(t - .5), 2), t))
            per_peak = []
            for row in rows:
                with np.load(out / 'activations' / arm / (row['key'] + '.npz')) as z:
                    probability = expit(z['beat']) if arm == 'BT-cv' else z['beat']
                ref = readbeats(manifest[row['key']]['annotation'])
                seq = [minimal_peaks(probability, fps=50 if arm == 'BT-cv' else 100, threshold=t) for t in thresholds]
                mm = [timing_metrics(ref, b) for b in seq]
                per_peak.append((seq, mm))
                candidates.extend(record(arm, 'peak threshold candidate', row, m, str(t)) for t, m in zip(thresholds, mm))
            chosen = select_folds([[m['F70'] for m in pair[1]] for pair in per_peak], folds)
            for i, (row, choice) in enumerate(zip(rows, chosen)):
                name = arm + '-threshold'
                records.append(record(name, 'tuned on SMC folds', row, per_peak[i][1][choice], str(thresholds[choice])))
                save_sequence(stage, name, row['key'], per_peak[i][0][choice])
    dump(stage / 'decoder-records.json', records)
    write_csv(stage / 'decoder-per-excerpt.csv', records)
    write_csv(stage / 'decoder-candidates.csv', candidates)
    dump(stage / 'decoder-diagnostics.json', diagnostics)


def beatonly(out, manifest, rows, workers):
    """Fixed matched-evidence controls; no setting selection."""
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for source in ('N1-s0-beatonly', 'N1-s1-beatonly'):
            for name in ('fixed', 'band35', 'loose'):
                results = list(pool.map(decode_job, [(out, source, name, row, manifest[row['key']]['annotation'])
                                                    for row in rows], chunksize=4))
                dump(out / 'stage2' / 'matched-evidence' / f'{source}-{name}.json', results)
                records.extend(dict(r[0], arm=source + '-' + name, kind='exploratory fixed evidence control') for r in results)
                print('matched evidence', source, name, flush=True)
    dump(out / 'stage2' / 'matched-evidence-records.json', records)
    write_csv(out / 'stage2' / 'matched-evidence-per-excerpt.csv', records)


def records_by_arm(records, rows):
    result = {}
    for arm in sorted({r['arm'] for r in records}):
        d = {r['key']: r for r in records if r['arm'] == arm}
        if set(d) != {r['key'] for r in rows}:
            raise ValueError(f'{arm}: incomplete excerpt alignment')
        result[arm] = [d[r['key']] for r in rows]
    return result


def associations(out, manifest, rows):
    stage = out / 'stage2'
    by = records_by_arm(json.loads((stage / 'level-records.json').read_text()), rows)
    parsed = [tags_and_confidence(r['tag'])[0] for r in rows]
    mapping = {name: sorted({t for tags in parsed for t in tags if any(s in t for s in patterns)})
               for name, patterns in TAG_PATTERNS.items()}
    tagvalues = {name: np.array([bool(tags & set(literals)) for tags in parsed]) for name, literals in mapping.items()}
    dump(stage / 'tag-mapping.json', mapping)
    eligible = [k for k, x in tagvalues.items() if 15 <= x.sum() <= len(x) - 15]
    names = ['intercept', 'below77', 'above167', 'log1p_ibi_cv'] + eligible
    x = np.column_stack([np.ones(len(rows)), [r['tempo_band'] == 'below77' for r in rows],
                         [r['tempo_band'] == 'above167' for r in rows], np.log1p([r['ibi_cv'] for r in rows])]
                        + [tagvalues[k] for k in eligible])
    raw = np.array([[r['F70'] for r in by[f'N1-s{s}-full']] for s in (0, 1)])
    ceiling = np.array([[r['F70'] for r in by[f'N1-s{s}-full-ceiling']] for s in (0, 1)])
    bt, btc = (np.array([r['F70'] for r in by[a]]) for a in ('BT-cv', 'BT-cv-ceiling'))
    outcomes = {'N1 raw F': raw.mean(axis=0), 'raw N1-BT gap': raw.mean(axis=0) - bt,
                'N1 F diagnostic ceiling': ceiling.mean(axis=0),
                'N1-BT gap diagnostic ceiling': ceiling.mean(axis=0) - btc}
    for s in (0, 1):
        outcomes.update({f's{s} raw F': raw[s], f's{s} raw gap': raw[s] - bt,
                         f's{s} F diagnostic ceiling': ceiling[s], f's{s} gap diagnostic ceiling': ceiling[s] - btc})
    model = bootstrap_ols(x, np.column_stack(list(outcomes.values())))
    dump(stage / 'tag-model.json', dict(names=names, outcomes=list(outcomes), **model))
    write_csv(stage / 'tag-design.csv', [dict(key=r['key'], **dict(zip(names, xx))) for r, xx in zip(rows, x)])
    write_csv(stage / 'tag-counts.csv', [dict(tag=k, n=int(v.sum()), eligible=k in eligible,
        below77=int(sum(v[i] and r['tempo_band'] == 'below77' for i, r in enumerate(rows))),
        in_band=int(sum(v[i] and r['tempo_band'] == 'in_band' for i, r in enumerate(rows)))) for k, v in tagvalues.items()])
    corr = np.corrcoef(x[:, 1:], rowvar=False)
    dump(stage / 'tag-correlations.json', dict(names=names[1:], correlation=corr.tolist()))
    coef_rows = []
    for j, name in enumerate(names):
        supported = model['vif'][j] <= 10 and model['singular'] / model['samples'] <= .05 and name != 'above167'
        for k, outcome in enumerate(outcomes):
            coef_rows.append(dict(predictor=name, outcome=outcome, supported=supported,
                coefficient=model['coef'][j][k] if supported else None,
                lo=model['lo'][j][k] if supported else None, hi=model['hi'][j][k] if supported else None,
                vif=model['vif'][j]))
    write_csv(stage / 'tag-coefficients.csv', coef_rows)
    probes = []
    for row in rows:
        if row['tempo_band'] != 'in_band':
            continue
        ref = readbeats(manifest[row['key']]['annotation'])
        for arm in ('BT-cv', 'N1-s0', 'N1-s1'):
            with np.load(out / 'activations' / arm / (row['key'] + '.npz')) as z:
                p = expit(z['beat']) if arm == 'BT-cv' else z['beat']
            probes.append(dict(arm=arm, key=row['key'], kind='annotation-assisted evidence diagnostic',
                               **evidence_probes(p, ref, fps=50 if arm == 'BT-cv' else 100)))
    write_csv(stage / 'evidence-per-excerpt.csv', probes)
    mask = np.array([r['tempo_band'] == 'in_band' for r in rows])
    inrows = [r for r, ok in zip(rows, mask) if ok]
    ep = {a: [next(p for p in probes if p['key'] == r['key'] and p['arm'] == a) for r in inrows]
          for a in ('BT-cv', 'N1-s0', 'N1-s1')}
    emetrics = ('at_beat', 'peak50', 'sharpness', 'advantage')
    ey = {m: np.array([(ep['N1-s0'][i][m] + ep['N1-s1'][i][m]) / 2 - ep['BT-cv'][i][m]
                      for i in range(len(inrows))]) for m in emetrics}
    enames = ['intercept', 'log1p_ibi_cv', 'log_bpm'] + [k for k, v in tagvalues.items() if 10 <= v[mask].sum() <= mask.sum() - 10]
    ex = np.column_stack([np.ones(mask.sum()), np.log1p([r['ibi_cv'] for r in inrows]),
                          np.log([r['tempo_bpm'] for r in inrows])] + [tagvalues[k][mask] for k in enames[3:]])
    emodel = bootstrap_ols(ex, np.column_stack(list(ey.values())))
    dump(stage / 'evidence-model.json', dict(names=enames, outcomes=list(ey), **emodel))
    erows = []
    for tag, present in {'all': np.ones(mask.sum(), bool), **{k: v[mask] for k, v in tagvalues.items()}}.items():
        if not present.any():
            continue
        for m in emetrics:
            erows.append(dict(tag=tag, metric=m, kind='annotation-assisted evidence diagnostic',
                N1=float(np.mean([(ep['N1-s0'][i][m] + ep['N1-s1'][i][m]) / 2 for i in np.flatnonzero(present)])),
                BT=float(np.mean([ep['BT-cv'][i][m] for i in np.flatnonzero(present)])),
                **paired_bootstrap(ey[m][present], np.zeros(present.sum()))))
    write_csv(stage / 'evidence-by-tag.csv', erows)
    print('tag and evidence models complete', flush=True)


def summarize(out, rows):
    stage = out / 'stage2'
    records = json.loads((stage / 'level-records.json').read_text())
    if (stage / 'matched-evidence-records.json').exists():
        records += json.loads((stage / 'matched-evidence-records.json').read_text())
    if (stage / 'decoder-records.json').exists():
        records += json.loads((stage / 'decoder-records.json').read_text())
        for source in ('BT-cv', 'N1-s0', 'N1-s1'):
            for variant in VARIANTS:
                candidates = json.loads((stage / 'decoder-candidates' / f'{source}-{variant}.json').read_text())
                records += [dict(r[0], arm=source + '-fixed-' + variant,
                                 kind='fixed preregistered decoder') for r in candidates]
    by = records_by_arm(records, rows)
    # Seed means preserve excerpt pairing. Signed errors are summarized per seed.
    for arm in list(by):
        if 'N1-s0' in arm and arm.replace('N1-s0', 'N1-s1') in by:
            other = by[arm.replace('N1-s0', 'N1-s1')]
            name = arm.replace('N1-s0', 'N1-mean')
            by[name] = [dict(a, arm=name, **{m: (a[m] + b[m]) / 2 for m in METRICS}, signed_ms=None)
                        for a, b in zip(by[arm], other)]
    masks = {'all': np.ones(len(rows), bool)}
    masks.update({s: np.array([r['tempo_band'] == s for r in rows]) for s in ('below77', 'in_band', 'above167')})
    masks.update({s: np.array([r['variability'] == s for r in rows]) for s in ('steady', 'high')})
    masks['easy'] = np.array([r['easy'] for r in rows])
    summary = []
    for arm, rr in by.items():
        for group, mask in masks.items():
            sub = [r for r, ok in zip(rr, mask) if ok]
            errs = [r['signed_ms'] for r in sub if r['signed_ms'] is not None]
            summary.append(dict(arm=arm, kind=rr[0]['kind'], group=group, n=len(sub),
                **{m: float(np.mean([r[m] for r in sub])) for m in METRICS},
                median_signed_ms=float(np.median(errs)) if errs else None))
    write_csv(stage / 'summary.csv', summary)
    contrasts = []
    pairs = [('N1-mean-full', 'BT-cv'), ('N1-mean-full-ceiling', 'BT-cv-ceiling'),
             ('N1-mean-peaks-ceiling', 'BT-cv-ceiling'), ('BT-cv-n1decode-ceiling', 'BT-cv-ceiling'),
             ('N1-mean-full-ceiling', 'N1-mean-full'), ('N1-mean-full-ridge', 'N1-mean-full'),
             ('N1-s0-full-ridge', 'N1-s0-full'), ('N1-s1-full-ridge', 'N1-s1-full')]
    pairs += [('N1-mean-beatonly-' + v, 'N1-mean-beatonly-fixed') for v in ('band35', 'loose')]
    pairs.append(('N1-mean-beatonly-band35', 'BT-cv-fixed-band35'))
    for evidence in ('BT-cv', 'N1-mean'):
        base = 'BT-cv-n1decode' if evidence == 'BT-cv' else 'N1-mean-full'
        pairs += [(evidence + '-' + variant, base) for variant in ('decoder-all', 'decoder-band', 'decoder-loose', 'decoder-half')]
        pairs.append((evidence + '-threshold', 'BT-cv' if evidence == 'BT-cv' else 'N1-mean-peaks'))
        pairs += [(evidence + '-fixed-' + variant, base) for variant in VARIANTS if variant != 'fixed']
    for left, right in pairs:
        if left not in by or right not in by:
            continue
        for group, mask in masks.items():
            for m in ('F70', 'AMLt'):
                a, b = (np.array([r[m] for r in by[arm]])[mask] for arm in (left, right))
                contrasts.append(dict(left=left, right=right, kind='diagnostic ceiling' if 'ceiling' in left or 'ceiling' in right
                    else by[left][0]['kind'],
                    group=group, metric=m, **paired_bootstrap(a, b)))
    write_csv(stage / 'paired.csv', contrasts)
    choices = [dict(arm=arm, kind=rr[0]['kind'], action=k, n=v) for arm, rr in by.items() if 'mean' not in arm
               for k, v in Counter(r['action'] for r in rr).items()]
    write_csv(stage / 'choices.csv', choices)
    dump(stage / 'summary.json', dict(summary=summary, paired=contrasts, choices=choices))
    print('summary complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('levels', 'decoders', 'beatonly', 'tags', 'summary'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    require_gate(a.out, a.out / 'manifest.json')
    stage = a.out / 'stage2'
    lock = json.loads((stage / 'prereg-lock.json').read_text())
    if hashlib.sha256((stage / 'prereg.md').read_bytes()).hexdigest() != lock['sha256']:
        raise ValueError('preregistration hash mismatch')
    manifest = {r['key']: r for r in json.loads((a.out / 'manifest.json').read_text())}
    rows = json.loads((a.out / 'scores/BT-cv.json').read_text())['rows']
    start = time.time()
    if a.command == 'levels':
        levels(a.out, manifest, rows)
    elif a.command == 'decoders':
        decoders(a.out, manifest, rows, a.workers)
    elif a.command == 'tags':
        associations(a.out, manifest, rows)
    elif a.command == 'beatonly':
        beatonly(a.out, manifest, rows, a.workers)
    else:
        summarize(a.out, rows)
    dump(stage / (a.command + '-runtime.json'), dict(seconds=time.time() - start, command=a.command))


if __name__ == '__main__':
    main()
