"""P2 ridge probes with streamed sufficient statistics and recording refits."""
from collections import defaultdict
import json

import numpy as np
from scipy.stats import spearmanr

from .common import OUT, by_tempo, dump, features, interp, interval, labels, log, manifests, prereg_check, stable_seed, units
from .tokens import event_inputs


def predictors(u, row):
    z = labels(u)
    q = np.isfinite(z['retention'])
    if not q.any():
        return None
    ht = z['hard_heads']
    pos = z['hard_pos']
    cls = np.full(len(ht), 6, int)
    for v, target in [(0, 0), (1, 1), (2, 1), (3, 1), (5, 2), (6, 3), (9, 4), (10, 5)]:
        cls[pos == v] = target
    median = np.median(np.diff(ht)) if len(ht) > 1 else 1000.
    prev = np.r_[median, np.diff(ht)]; nxt = np.r_[np.diff(ht), median]
    c = np.c_[np.eye(7)[cls], np.log(np.maximum(prev, 1)), np.log(np.maximum(nxt, 1))]
    f, logits, m = features(row)
    d = event_inputs(f, logits, m, ht[q], z)
    x = np.c_[c[q], interp(f, ht[q]), d['M'][:, :1152], z['chord'][q]].astype('float64')
    y = z['retention'][q].astype(float)
    return x, y, cls[q]


def permute_retention(y, cls, rec):
    null = y.copy()
    rng = np.random.default_rng(stable_seed('P2-null:' + rec))
    for k in range(7):
        ix = np.flatnonzero(cls == k)
        null[ix] = rng.permutation(null[ix])
    return null


def run(seed):
    prereg_check()
    cache, us = manifests(seed), units()
    width = 9 + 160 + 1152 + 1
    sets = dict(C=np.arange(9), CF=np.r_[np.arange(9), np.arange(9, 169)],
                CM=np.r_[np.arange(9), np.arange(169, width - 1)],
                CF_chord=np.r_[np.arange(169), width - 1],
                CM_chord=np.r_[np.arange(9), np.arange(169, width)])
    train_rec = sorted({u['rec'] for u in us if u['cohort'] == 'probe-fit'})
    rec_index = {r: i for i, r in enumerate(train_rec)}
    boot = [np.ones(len(train_rec))]
    for b in range(5):
        rng = np.random.default_rng(6100 + b)
        boot.append(np.bincount(rng.integers(len(train_rec), size=len(train_rec)), minlength=len(train_rec)))
    # Intercept is the last column; statistics are recording-normalized.
    xx = np.zeros((6, width + 1, width + 1), float)
    xy = np.zeros((6, width + 1, 2), float)
    devxx = np.zeros((width + 1, width + 1), float)
    devxy = np.zeros((width + 1, 2), float)
    devyy = np.zeros(2)
    nfit = ndev = 0
    for num, u in enumerate(us):
        if u['cohort'] not in ('probe-fit', 'probe-dev') or u['sha'] not in cache:
            continue
        data = predictors(u, cache[u['sha']])
        if data is None:
            continue
        x, y, cls = data
        null = permute_retention(y, cls, u['rec'])
        x = np.c_[x, np.ones(len(x))]
        target = np.c_[y, null]
        a, b = x.T @ x / len(x), x.T @ target / len(x)
        if u['cohort'] == 'probe-fit':
            nfit += 1
            for j in range(6):
                weight = boot[j][rec_index[u['rec']]]
                xx[j] += weight * a; xy[j] += weight * b
        else:
            ndev += 1
            devxx += a; devxy += b; devyy += np.mean(target ** 2, axis=0)
        if num % 200 == 0:
            log('P2_stats', seed=seed, units=num)
    models, selected = {}, {}
    for name, cols in sets.items():
        idx = np.r_[cols, width]
        for null in (0, 1):
            for refit in range(6 if null == 0 and name in ('C', 'CF', 'CM') else 1):
                a = xx[refit][np.ix_(idx, idx)]
                b = xy[refit, idx, null]
                mass = a[-1, -1]
                mean = a[:-1, -1] / mass
                sd = np.maximum(np.sqrt(np.maximum(np.diag(a)[:-1] / mass - mean ** 2, 0)), .01)
                transform = np.eye(len(idx)); transform[:-1, :-1] /= sd[:, None]
                transform[-1, :-1] = -mean / sd
                az, bz = transform.T @ a @ transform, transform.T @ b
                choices = []
                for alpha in (.1, 1., 10., 100.):
                    penalty = np.eye(len(idx)) * alpha; penalty[-1, -1] = 0
                    beta = transform @ np.linalg.solve(az + penalty, bz)
                    mse = (beta @ devxx[np.ix_(idx, idx)] @ beta - 2 * beta @ devxy[idx, null] + devyy[null]) / max(ndev, 1)
                    choices.append((float(mse), alpha, beta))
                mse, alpha, beta = min(choices, key=lambda t: t[0])
                key = f'{name}:null{null}:refit{refit}'
                models[key] = (idx, beta)
                selected[key] = dict(alpha=alpha, dev_mse=mse)
    predictions = defaultdict(lambda: defaultdict(list))
    for u in us:
        if u['cohort'] not in ('panel', 'val_hard') or u['sha'] not in cache:
            continue
        data = predictors(u, cache[u['sha']])
        if data is None:
            continue
        x, y, cls = data
        x = np.c_[x, np.ones(len(x))]
        predictions[u['rec']]['y'].append(y)
        predictions[u['rec']]['cls'].append(cls)
        for name, (idx, beta) in models.items():
            predictions[u['rec']][name].append(x[:, idx] @ beta)
    recordings = defaultdict(dict)
    for rec, group in predictions.items():
        y, cls = np.concatenate(group['y']), np.concatenate(group['cls'])
        if len(y) < 30:
            continue
        null = permute_retention(y, cls, rec)
        for name, values in group.items():
            if name in ('y', 'cls'):
                continue
            target = null if ':null1:' in name else y
            if np.std(target) == 0:
                continue
            score = float(spearmanr(target, np.concatenate(values)).statistic)
            if np.isfinite(score):
                recordings[name][rec] = score
    summary = {k: dict(**interval(list(r.values())), recordings=r) for k, r in recordings.items()}
    np.savez_compressed(OUT / 'models' / f'P2-s{seed}.npz',
        **{key + ':coefficients': beta for key, (idx, beta) in models.items()},
        **{key + ':columns': idx for key, (idx, beta) in models.items()})
    dump(OUT / f'P2-s{seed}.json', dict(summary=summary, by_tempo=by_tempo(summary), selected=selected, fit_units=nfit, dev_units=ndev))
    log('P2_done', seed=seed)
