"""P1: frame attacks, metrical-position controls and shifted-label refits."""
from collections import defaultdict
import json

import numpy as np

from .common import (OUT, auc, dump, features, interval, labels, log, manifests,
                     nearest, prereg_check, song_auc, stable_seed, units)
from .models import Predictor, fit, load, save
from .tokens import attack_sample, frame_inputs, frame_labels

SETS = ('F', 'L', 'M', 'Pos', 'F+Pos', 'M+Pos')


def select(d, name):
    return np.concatenate([d[k] for k in name.split('+')], axis=1)


def training(seed, name, shifted=False):
    groups = defaultdict(list)
    rec_ids = {r: i for i, r in enumerate(sorted({u['rec'] for u in units()}))}
    for u in units():
        if u['cohort'] not in ('probe-fit', 'probe-dev'):
            continue
        p = OUT / 'frames' / f's{seed}' / ('shift' if shifted else 'real') / (u['id'] + '.npz')
        if not p.exists():
            continue
        with np.load(p) as d:
            groups[u['cohort']].append((select(d, name), d['y'], u['rec']))
    out = []
    for cohort in ('probe-fit', 'probe-dev'):
        g = groups[cohort]
        out.append((np.concatenate([a for a, _, _ in g]),
                    np.concatenate([b for _, b, _ in g]),
                    np.concatenate([np.repeat(rec_ids[r], len(b)) for _, b, r in g])))
    return out


def model_path(seed, kind, name, shifted=False, boot=None):
    suffix = '-shift' if shifted else ''
    if boot is not None:
        suffix += f'-boot{boot}'
    return OUT / 'models' / f'P1-s{seed}-{kind}-{name}{suffix}.pt'


def fit_models(seed, shifted=False, bootstrap=False):
    prereg_check()
    attack_sample(seed, shifted)
    for name in SETS:
        jobs = [(kind, b) for kind in ('logistic', 'mlp') for b in ([None] if not bootstrap else range(5))]
        if bootstrap and name not in ('F', 'M', 'Pos', 'F+Pos'):
            continue
        if bootstrap:
            jobs = [('mlp', b) for b in range(5)]
        if all(model_path(seed, kind, name, shifted, b).exists() for kind, b in jobs):
            continue
        (x, y, rec), (xd, yd, _) = training(seed, name, shifted)
        for kind, b in jobs:
            p = model_path(seed, kind, name, shifted, b)
            if p.exists():
                continue
            log('P1_fit', seed=seed, kind=kind, inputs=name, shifted=shifted, bootstrap=b,
                rows=len(y), dev_rows=len(yd))
            model = fit(x, y, rec, xd, yd, kind=kind, seed=3200 + seed,
                        bootstrap=b)
            save(p, model)
            log('P1_fit_done', seed=seed, kind=kind, inputs=name,
                shifted=shifted, bootstrap=b, epochs=model['epochs'], dev_bce=model['dev_bce'])
        del x, xd


def evaluate(seed, shifted=False, bootstrap=False):
    prereg_check()
    rows = manifests(seed)
    predictors = {}
    for name in SETS:
        for kind in ('logistic', 'mlp'):
            for b in (range(5) if bootstrap else [None]):
                p = model_path(seed, kind, name, shifted, b)
                if p.exists():
                    key = f'{kind}-{name}' + (f'-boot{b}' if b is not None else '')
                    predictors[key] = (name, Predictor(load(p)))
    by_rec = defaultdict(list)
    for u in units():
        if u['cohort'] not in ('probe-fit', 'probe-dev') and u['sha'] in rows:
            by_rec[u['rec']].append(u)
    recorded = defaultdict(lambda: defaultdict(dict))
    for i, (rec, members) in enumerate(sorted(by_rec.items())):
        groups = defaultdict(lambda: defaultdict(list))
        for u in members:
            f, l, m = features(rows[u['sha']])
            z = labels(u)
            y = frame_labels(z, len(f), shifted)
            idx = np.flatnonzero(y >= 0)
            off = abs(nearest(idx * 10., z['beats'])[0]) > 30
            pred = {k: [] for k in predictors}
            fluxes = []
            for start in range(0, len(idx), 8192):
                d = frame_inputs(f, l, m, idx[start:start + 8192], z)
                fluxes.append(d['flux'])
                for k, (name, model) in predictors.items():
                    pred[k].append(model(select(d, name)))
            if not len(idx):
                continue
            pred = {k: np.concatenate(v) for k, v in pred.items()}
            pred['flux'] = np.concatenate(fluxes)
            tempo = float(np.median(60000 / z['length']))
            band = 'ge160' if tempo >= 160 else '100-160' if tempo >= 100 else 'lt100'
            cohort = 'panels' if u['cohort'] in ('panel', 'val_hard') else u['cohort']
            for scope, q in [('all', np.ones(len(idx), bool)), ('offline', off)]:
                for cell in (cohort, cohort + ':' + u['cohort'], cohort + ':' + band):
                    group = groups[cell + ':' + scope]
                    group['y'].append(y[idx][q])
                    for k, pp in pred.items():
                        group[k].append(pp[q])
        for cell, g in groups.items():
            yy = np.concatenate(g['y'])
            if min(np.sum(yy == 0), np.sum(yy == 1)) < 50:
                continue
            for k, pp in g.items():
                if k != 'y':
                    recorded[cell][k][rec] = auc(yy, np.concatenate(pp))
        if i % 50 == 0:
            log('P1_read', seed=seed, recordings=i, shifted=shifted, bootstrap=bootstrap)
    out = {cell: {k: dict(**interval(list(v.values())), recordings=v) for k, v in g.items()}
           for cell, g in recorded.items()}
    suffix = '-shift' if shifted else ''
    suffix += '-bootstrap' if bootstrap else ''
    dump(OUT / f'P1-s{seed}{suffix}.json', out)
    log('P1_done', seed=seed, shifted=shifted, bootstrap=bootstrap)
