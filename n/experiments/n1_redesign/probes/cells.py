"""P3 cell probes and M0 posteriors on chart and decoded beat grids."""
from collections import defaultdict
import json

import numpy as np
from scipy.special import expit

from .common import (OUT, PREV, LATTICE, dump, features, interp, labels, log,
                     manifests, nearest, prereg_check, song_auc, stable_seed, units)
from .models import Predictor, fit, load, save
from .attacks import model_path as attack_path
from .tokens import beat_tokens, pca_transform, token_load


def decoded(seed, u, z):
    from ensomi_n.mapformat import MusicMap
    p = PREV / 'e2-maps' / f's{seed}' / '009001' / (u['sha'] + '.json')
    m = MusicMap.from_json(json.loads(p.read_text())['full'])
    t = m.timing(float(z['beats'][0]), float(z['beats'][-1] + z['length'][-1]))
    keep = (t.beats >= z['beats'][0] - 50) & (t.beats <= z['beats'][-1] + 50)
    b = t.beats[keep]
    delta, _ = nearest(b, z['beats'])
    close = abs(delta) <= 50
    offset = float(np.median(delta[close])) if close.any() else 0.
    d, j = nearest(b - offset, z['beats'])
    matched = abs(d) <= 15
    return dict(beats=b, length=t.step_ms[keep], down=t.downbeat[keep],
                offset=offset, matched=matched, chart_index=j,
                red=np.array([[s.start_ms, s.beat_ms, s.meter] for s in m.segments]))


def inputs(seed, u, rows, grid='chart'):
    z = labels(u)
    f = None
    if grid == 'chart':
        t = token_load(seed, u)
        result = dict(T24=t['F'], Mel24=t['M'], T24shift=t['Fshift'], Mel24shift=t['Mshift'])
        g = z
    else:
        g = decoded(seed, u, z)
        f, _, m = features(rows[u['sha']])
        result = {}
        for name, xx in [('T24', f), ('Mel24', np.asarray(m, np.float32).reshape(len(m), 64, 2).mean(axis=-1))]:
            kind = 'F' if name == 'T24' else 'M'
            with np.load(OUT / 'tokens' / f'pca-s{seed}-{kind}.npz') as pca:
                result[name] = pca_transform(beat_tokens(xx, g), pca)
                result[name + 'shift'] = pca_transform(beat_tokens(xx, g, 1 / 32), pca)
    if f is None:
        f, _, _ = features(rows[u['sha']])
    attack = Predictor(load(attack_path(seed, 'mlp', 'F')))
    b, step = g['beats'], g['length']
    for suffix, shift in [('', 0.), ('shift', 1 / 32)]:
        at = b[:, None] + step[:, None] * (LATTICE[None, :] + shift)
        result['F_b' + suffix] = expit(attack(interp(f, at).reshape(-1, 160))).reshape(len(b), 24)
    y = z['y'].copy() if grid == 'chart' else z['y'][g['chart_index']].copy()
    if grid != 'chart':
        y[~g['matched']] = -1
    return result, y, g


def build_data(seed):
    rows = manifests(seed)
    for i, u in enumerate(units()):
        if u['sha'] not in rows:
            continue
        p = OUT / 'cells' / f's{seed}' / (u['id'] + '.npz')
        if p.exists():
            continue
        d, y, g = inputs(seed, u, rows)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, y=y, **{k: v.astype('float16') for k, v in d.items()})
        if i % 200 == 0:
            log('P3_inputs', seed=seed, units=i)


def data_path(seed, u):
    return OUT / 'cells' / f's{seed}' / (u['id'] + '.npz')


def model_path(seed, kind, name, null='real', boot=None):
    suffix = f'-boot{boot}' if boot is not None else ''
    return OUT / 'models' / f'P3-s{seed}-{kind}-{name}-{null}{suffix}.pt'


def training(seed, name, null):
    groups = defaultdict(list)
    rec_ids = {r: i for i, r in enumerate(sorted({u['rec'] for u in units()}))}
    for u in units():
        if u['cohort'] not in ('probe-fit', 'probe-dev') or not data_path(seed, u).exists():
            continue
        with np.load(data_path(seed, u)) as z:
            y = z['y']
            if min(np.sum(y == 0), np.sum(y == 1)) < 4:
                continue
            q = y >= 0
            x, y = z[name][q], y[q]
        if null == 'shuffle':
            y = np.random.default_rng(stable_seed('P3shuffle:' + u['rec'])).permutation(y)
        groups[u['cohort']].append((x, y, rec_ids[u['rec']]))
    return [(np.concatenate([a for a, _, _ in groups[c]]),
             np.concatenate([b for _, b, _ in groups[c]]),
             np.concatenate([np.repeat(r, len(b)) for _, b, r in groups[c]]))
            for c in ('probe-fit', 'probe-dev')]


def run(seed, bootstrap=False):
    prereg_check()
    build_data(seed)
    for name in ('T24', 'Mel24', 'F_b'):
        for null in (['real'] if bootstrap else ['real', 'shuffle']):
            (x, y, rec), (xd, yd, _) = training(seed, name, null)
            for kind in (['mlp'] if bootstrap else ['logistic', 'mlp']):
                for boot in (range(5) if bootstrap else [None]):
                    path = model_path(seed, kind, name, null, boot)
                    if not path.exists():
                        log('P3_fit', seed=seed, kind=kind, name=name, null=null, boot=boot, rows=len(y))
                        m = fit(x, y, rec, xd, yd, kind=kind, seed=4300 + seed, bootstrap=boot)
                        save(path, m)
    evaluate(seed, bootstrap)


def evaluate(seed, bootstrap=False):
    rows = manifests(seed)
    predictors = {}
    for name in ('T24', 'Mel24', 'F_b'):
        for null in (['real'] if bootstrap else ['real', 'shuffle']):
            for kind in (['mlp'] if bootstrap else ['logistic', 'mlp']):
                for boot in (range(5) if bootstrap else [None]):
                    p = model_path(seed, kind, name, null, boot)
                    key = f'{kind}-{name}-{null}' + (f'-boot{boot}' if boot is not None else '')
                    predictors[key] = (name, null, Predictor(load(p)))
    groups = defaultdict(lambda: defaultdict(list))
    unmatched = []
    for i, u in enumerate(units()):
        if not data_path(seed, u).exists():
            continue
        with np.load(data_path(seed, u)) as cache:
            d = dict(cache)
        if not bootstrap:
            p = expit(predictors['mlp-T24-real'][2](d['T24'])).astype('float32')
            path = OUT / 'm0' / f's{seed}' / (u['id'] + '.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, p)
        if u['cohort'] in ('probe-fit', 'probe-dev'):
            continue
        for grid in ('chart', 'decoded'):
            if grid == 'decoded':
                if u['cohort'] not in ('panel', 'val_hard'):
                    continue
                try:
                    d, yy, g = inputs(seed, u, rows, grid)
                except FileNotFoundError:
                    continue
                d['y'] = yy
                unmatched.append(dict(rec=u['rec'], share=float(1 - g['matched'].mean()), beats=len(yy)))
            yy = d['y']
            q = yy >= 0
            if not q.any():
                continue
            y = yy[q]
            cohort = 'panels' if u['cohort'] in ('panel', 'val_hard') else u['cohort']
            band = 'ge160' if np.median(60000 / labels(u)['length']) >= 160 else 'lt160'
            for sampling in (['real'] if bootstrap else ['real', 'shift']):
                pred = {k: m(d[name + ('shift' if sampling == 'shift' else '')][q])
                        for k, (name, null, m) in predictors.items()}
                for cell in (cohort, cohort + ':' + u['cohort'], cohort + ':' + band):
                    group = groups[f'{cell}:{grid}:{sampling}']
                    group['y'].append(y)
                    group['rec'].append(np.repeat(u['rec'], len(y)))
                    for k, p in pred.items():
                        # Shuffled-label read is also shuffled within song.
                        group[k].append(p)
            if i % 100 == 0:
                log('P3_read', seed=seed, units=i)
    out = {}
    for cell, g in groups.items():
        y, rec = np.concatenate(g['y']), np.concatenate(g['rec'])
        out[cell] = {}
        for k, p in g.items():
            if k in ('y', 'rec'):
                continue
            yy = y.copy()
            if '-shuffle' in k:
                for r in np.unique(rec):
                    ix = np.flatnonzero(rec == r)
                    yy[ix] = np.random.default_rng(stable_seed('P3shuffle:' + r)).permutation(yy[ix])
            out[cell][k] = song_auc(yy, np.concatenate(p), rec, minimum=4)
    out['unmatched'] = unmatched
    dump(OUT / (f'P3-s{seed}' + ('-bootstrap' if bootstrap else '') + '.json'), out)
    log('P3_done', seed=seed, bootstrap=bootstrap)
