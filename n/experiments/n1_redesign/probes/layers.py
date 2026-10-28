"""P4 event-layer labels against metrical-position controls."""
from collections import defaultdict

import numpy as np
from scipy.special import expit

from .common import (OUT, auc, dump, features, interval, labels, log, manifests,
                     prereg_check, stable_seed, units)
from .models import Predictor, fit, load, save
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


def matrix(seed, rows, selected, name, label, stratum, cohort):
    width = {'Pos': 29, 'Pos+F': 189, 'Pos+M': 1309}[name]
    number = sum(len(y) for u, use, y in selected if u['cohort'] == cohort)
    x, y, rec = np.empty((number, width), np.float16), np.empty(number, np.int8), np.empty(number, int)
    position_class = np.empty(number, int)
    offset = 0
    for i, (u, use, yy) in enumerate(selected):
        if u['cohort'] != cohort:
            continue
        f, l, m = features(rows[u['sha']]); z = labels(u)
        d = event_inputs(f, l, m, z['event_t'][use], z)
        xx = np.concatenate([d[k] for k in name.split('+')], axis=1)
        n = len(yy)
        x[offset:offset + n], y[offset:offset + n], rec[offset:offset + n] = xx, yy, i
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
            use, y = target(z, label, st)
            if min(sum(y == 0), sum(y == 1)) >= 10:
                selected.append((u, use, y))
        counts = {c: sum(len(y) for u, _, y in selected if u['cohort'] == c)
                  for c in ('probe-fit', 'probe-dev')}
        tag = label + (f':stratum{st}' if st is not None else '')
        if min(counts.values()) == 0 or sum(counts.values()) * 1309 * 2 > 4e9:
            output[tag] = dict(unassessed='No training/dev examples or matrix would exceed RAM budget', counts=counts)
            continue
        predictors = {}
        for name in ('Pos', 'Pos+F', 'Pos+M'):
            x, y, rec, pc = matrix(seed, rows, selected, name, label, st, 'probe-fit')
            xd, yd, rd, pd = matrix(seed, rows, selected, name, label, st, 'probe-dev')
            for null in (False, True):
                for refit in ([None] if null else [None, 0, 1, 2, 3, 4]):
                    suffix = '' if refit is None else f':boot{refit}'
                    path = OUT / 'models' / f'P4-s{seed}-{label}-{st}-{name}-null{int(null)}{suffix}.pt'
                    if not path.exists():
                        yy, yyd = (permute(y, rec, pc), permute(yd, rd, pd)) if null else (y, yd)
                        log('P4_fit', seed=seed, label=label, stratum=st, inputs=name, null=null, refit=refit, rows=len(y))
                        save(path, fit(x, yy, rec, xd, yyd, kind='logistic', seed=5400 + seed, bootstrap=refit))
                    predictors[f'{name}:null{int(null)}' + suffix] = (name, null, Predictor(load(path)))
            del x, xd
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
                yn = permute(yy, np.repeat(stable_seed(rec), len(yy)), np.concatenate(g['pc']))
                for key, pp in g.items():
                    if key not in ('y', 'pc'):
                        scores[cohort][key][rec] = auc(yn if key.endswith('null1') else yy, np.concatenate(pp))
        output[tag] = dict(counts=counts, summary={cohort: {name: dict(**interval(list(r.values())), recordings=r)
                          for name, r in g.items()} for cohort, g in scores.items()})
    dump(OUT / f'P4{"b" if keysounds else "a"}-s{seed}.json', output)
