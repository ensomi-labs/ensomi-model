"""P4 event-layer labels against metrical-position controls."""
from collections import defaultdict
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit

from .common import (OUT, auc, by_tempo, dump, features, interval, labels, log, manifests,
                     prereg_check, stable_seed, units)
from .models import DiskMatrix, Predictor, fit, load, save
from .tokens import event_inputs


def target(z, label, stratum=None):
    if label in ('clap', 'finish'):
        bits = z['event_bits']
        bit = 8 if label == 'clap' else 4
        positive = (bits >= 0) & ((bits & bit) != 0)
        negative = (bits >= 0) & ((bits & 14) == 0)
        use = z['event_eligible'] & (positive | negative)
        return use, positive[use].astype(np.int8)
    part = {'kick': 0, 'snare': 1, 'hat': 2}[label]
    use = (z['event_part'] >= 0) & (z['event_stratum'] == stratum)
    return use, (z['event_part'][use] == part).astype(np.int8)


def matrix(seed, rows, selected, name, label, stratum, cohort, mapped_path=None):
    width = {'Pos': 29, 'Pos+F': 189, 'Pos+M': 1309}[name]
    number = sum(len(y) for u, use, y in selected if u['cohort'] == cohort)
    x = (DiskMatrix(mapped_path, (number, width))
         if mapped_path else np.empty((number, width), np.float16))
    y, rec = np.empty(number, np.int8), np.empty(number, int)
    position_class = np.empty(number, int)
    offset = 0
    for i, (u, use, yy) in enumerate(selected):
        if u['cohort'] != cohort:
            continue
        f, l, m = features(rows[u['sha']]); z = labels(u)
        times = z['event_t'][use]
        for start in range(0, len(yy), 8192):
            d = event_inputs(f, l, m, times[start:start + 8192], z)
            xx = np.concatenate([d[k] for k in name.split('+')], axis=1)
            n = len(xx)
            if mapped_path:
                x.write(offset, xx)
            else:
                x[offset:offset + n] = xx
            y[offset:offset + n], rec[offset:offset + n] = yy[start:start + n], i
            position_class[offset:offset + n] = np.argmax(d['Pos'][:, :24], axis=1) + 24 * np.argmax(d['Pos'][:, 24:], axis=1)
            offset += n
    return x, y, rec, position_class


def permute(y, rec, position_class):
    yy = y.copy()
    for r in np.unique(rec):
        rng = np.random.default_rng(8100 + int(r))
        for c in np.unique(position_class[rec == r]):
            q = np.flatnonzero((rec == r) & (position_class == c))
            yy[q] = rng.permutation(yy[q])
    return yy


def matrix_backing(counts, width, resident_budget=4_000_000_000):
    """Prefer resident training rows for random batches; dev reads are sequential."""
    resident, backing = 0, {}
    for cohort in ('probe-fit', 'probe-dev'):
        size = counts[cohort] * width * 2
        backing[cohort] = resident + size > resident_budget
        if not backing[cohort]:
            resident += size
    return backing


def run(seed, keysounds=False):
    prereg_check()
    rows = manifests(seed)
    tasks = [(name, None) for name in ('clap', 'finish')]
    if keysounds:
        tasks = [(name, st) for name in ('kick', 'snare', 'hat') for st in (0, 1)]
    output = {}
    for label, st in tasks:
        selected = []
        for u in units():
            if u['sha'] not in rows:
                continue
            z = labels(u)
            if not keysounds and u['cohort'] in ('panel', 'val_hard') and z['mode'] != 3:
                continue
            if not keysounds and u['cohort'] == 'std_extension' and z['mode'] != 0:
                continue
            use, y = target(z, label, st)
            training = u['cohort'] in ('probe-fit', 'probe-dev')
            if len(y) and (not training or min(sum(y == 0), sum(y == 1)) >= 10):
                selected.append((u, use, y))
        counts = {c: sum(len(y) for u, _, y in selected if u['cohort'] == c)
                  for c in ('probe-fit', 'probe-dev')}
        label_counts = {}
        for cohort in ('probe-fit', 'probe-dev', 'panel', 'val_hard', 'std_extension', 'keysound_extension'):
            members = [(u, y) for u, _, y in selected if u['cohort'] == cohort]
            label_counts[cohort] = dict(recordings=len({u['rec'] for u, _ in members}),
                units=len(members), events=sum(len(y) for _, y in members),
                positive=sum(int(np.sum(y == 1)) for _, y in members),
                negative=sum(int(np.sum(y == 0)) for _, y in members))
        tag = label + (f':stratum{st}' if st is not None else '')
        if min(counts.values()) == 0:
            output[tag] = dict(unassessed='No training/dev examples',
                               counts=counts, label_counts=label_counts)
            continue
        predictors = {}
        for name in ('Pos', 'Pos+F', 'Pos+M'):
            jobs = []
            for null in (False, True):
                for refit in ([None] if null else [None, 0, 1, 2, 3, 4]):
                    suffix = '' if refit is None else f':boot{refit}'
                    path = OUT / 'models' / f'P4-s{seed}-{label}-{st}-{name}-null{int(null)}{suffix}.pt'
                    jobs.append((null, refit, suffix, path))
            if any(not path.exists() for _, _, _, path in jobs):
                width = {'Pos': 29, 'Pos+F': 189, 'Pos+M': 1309}[name]
                size = sum(counts.values()) * width * 2
                backing = matrix_backing(counts, width)
                mapped = any(backing.values())
                if mapped:
                    mapped_size = sum(counts[c] * width * 2 for c, use in backing.items() if use)
                    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
                    if used + mapped_size > 14.9e9:
                        raise RuntimeError('P4 mapped matrix would exceed the artifact disk budget')
                    log('P4_mapped_storage', seed=seed, label=label, inputs=name,
                        matrix_bytes=size, backing=backing, mapped_bytes=mapped_size,
                        projected_artifact_bytes=used + mapped_size)
                with tempfile.TemporaryDirectory(prefix='P4-matrix-', dir=OUT) as temporary:
                    directory = Path(temporary)
                    x, y, rec, pc = matrix(seed, rows, selected, name, label, st, 'probe-fit', directory / 'fit' if backing['probe-fit'] else None)
                    xd, yd, rd, pd = matrix(seed, rows, selected, name, label, st, 'probe-dev', directory / 'dev' if backing['probe-dev'] else None)
                    try:
                        for null, refit, suffix, path in jobs:
                            if not path.exists():
                                yy, yyd = (permute(y, rec, pc), permute(yd, rd, pd)) if null else (y, yd)
                                log('P4_fit', seed=seed, label=label, stratum=st, inputs=name, null=null, refit=refit, rows=len(y), mapped=mapped)
                                save(path, fit(x, yy, rec, xd, yyd, kind='logistic', seed=5400 + seed, bootstrap=refit))
                    finally:
                        for a in (x, xd):
                            if isinstance(a, np.memmap):
                                a._mmap.close()
                        del a, x, xd
            for null, refit, suffix, path in jobs:
                predictors[f'{name}:null{int(null)}' + suffix] = (name, null, Predictor(load(path)))
        by_rec = defaultdict(list)
        for u, use, yy in selected:
            if u['cohort'] not in ('probe-fit', 'probe-dev'):
                by_rec[u['rec']].append((u, use, yy))
        scores = defaultdict(lambda: defaultdict(dict))
        for rec, members in by_rec.items():
            groups = defaultdict(lambda: defaultdict(list))
            for u, use, yy in members:
                f, l, m = features(rows[u['sha']]); z = labels(u)
                d = event_inputs(f, l, m, z['event_t'][use], z)
                pc = np.argmax(d['Pos'][:, :24], axis=1) + 24 * np.argmax(d['Pos'][:, 24:], axis=1)
                cohort = 'panels' if u['cohort'] in ('panel', 'val_hard') else u['cohort']
                g = groups[cohort]; g['y'].append(yy); g['pc'].append(pc)
                for key, (name, null, pred) in predictors.items():
                    xx = np.concatenate([d[k] for k in name.split('+')], axis=1)
                    g[key].append(pred(xx))
            for cohort, g in groups.items():
                yy = np.concatenate(g['y'])
                if min(sum(yy == 0), sum(yy == 1)) < 10:
                    continue
                yn = permute(yy, np.repeat(stable_seed(rec), len(yy)), np.concatenate(g['pc']))
                for key, pp in g.items():
                    if key not in ('y', 'pc'):
                        scores[cohort][key][rec] = auc(yn if key.endswith('null1') else yy, np.concatenate(pp))
        output[tag] = dict(counts=counts, label_counts=label_counts,
                          summary={cohort: {name: dict(**interval(list(r.values())), recordings=r)
                          for name, r in g.items()} for cohort, g in scores.items()})
        output[tag]['by_tempo'] = {cohort: by_tempo(summary) for cohort, summary in output[tag]['summary'].items()}
    dump(OUT / f'P4{"b" if keysounds else "a"}-s{seed}.json', output)
