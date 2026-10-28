"""Beat summaries, training-only PCA and frame/position probe inputs."""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .common import (OUT, LATTICE, dump, features, interp, labels, log, manifests,
                     nearest, prereg_check, stable_seed, units)


def beat_tokens(x, z, shift=0.):
    b, length = z['beats'], z['length']
    times = b[:, None] + length[:, None] * (LATTICE[None, :] + shift)
    lattice = interp(x, times)
    # The last frame before a fractional reset belongs only to the old beat.
    end = np.minimum(b + length, np.r_[b[1:], b[-1] + length[-1]])
    lo = np.clip(np.ceil(b / 10).astype(int), 0, len(x) - 1)
    hi = np.maximum(lo + 1, np.clip(np.ceil(end / 10).astype(int), 0, len(x)))
    cs = np.vstack([np.zeros((1, x.shape[1]), 'float32'), np.cumsum(x, axis=0, dtype='float32')])
    mean = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return np.concatenate([mean[:, None, :], lattice], axis=1).reshape(len(b), -1).astype('float32')


def standardize_song(t):
    return ((t - t.mean(axis=0)) / np.maximum(t.std(axis=0), 1e-5)).astype('float32')


def token_path(seed, u):
    return OUT / 'tokens' / f's{seed}' / (u['id'] + '.npz')


def token_load(seed, u):
    with np.load(token_path(seed, u), allow_pickle=False) as z:
        return {k: z[k].astype('float32') for k in z.files}


def pca_transform(t, pca):
    return (t - pca['mean']) @ pca['components'].T


def prepare_tokens(seed):
    prereg_check()
    rows, us = manifests(seed), units()
    for name, width in [('F', 160), ('M', 64)]:
        pp = OUT / 'tokens' / f'pca-s{seed}-{name}.npz'
        if pp.exists():
            continue
        fit = [u for u in us if u['cohort'] == 'probe-fit' and u['sha'] in rows]
        sizes = [min(64, len(labels(u)['beats'])) for u in fit]
        sample = np.empty((sum(sizes), 25 * width), np.float32)
        k = 0
        for u, size in zip(fit, sizes):
            f, _, m = features(rows[u['sha']])
            x = f if name == 'F' else np.asarray(m, np.float32).reshape(len(m), 64, 2).mean(axis=-1)
            t = beat_tokens(x, labels(u))
            idx = np.random.default_rng(stable_seed('pca:' + u['id'])).choice(len(t), size, replace=False)
            sample[k:k + size] = t[idx]
            k += size
        log('pca_fit_start', seed=seed, kind=name, shape=sample.shape)
        p = PCA(n_components=64, svd_solver='randomized', iterated_power=3, random_state=1701)
        p.fit(sample)
        pp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pp, mean=p.mean_, components=p.components_, explained=p.explained_variance_ratio_)
        del sample, p
        log('pca_fit_done', seed=seed, kind=name)
    pcas = {}
    for name in ('F', 'M'):
        with np.load(OUT / 'tokens' / f'pca-s{seed}-{name}.npz') as z:
            pcas[name] = dict(z)
    for i, u in enumerate(us):
        if u['sha'] not in rows:
            continue
        path = token_path(seed, u)
        if path.exists():
            continue
        f, l, m = features(rows[u['sha']])
        m = np.asarray(m, np.float32).reshape(len(m), 64, 2).mean(axis=-1)
        z = labels(u)
        out = {}
        for name, x in [('F', f), ('M', m)]:
            t = pca_transform(beat_tokens(x, z), pcas[name])
            out[name] = t.astype('float32')
            out[name + 'z'] = standardize_song(t)
            out[name + 'shift'] = pca_transform(beat_tokens(x, z, 1 / 32), pcas[name]).astype('float32')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        if i % 100 == 0:
            log('tokens_progress', seed=seed, units=i)
    log('tokens_done', seed=seed)


def frame_labels(z, frames, shifted=False):
    times = np.arange(frames) * 10.
    heads = z['heads']
    if shifted:
        heads = heads + z['length'][z['head_beat']] / 32
    dist = abs(nearest(times, np.sort(heads))[0])
    bi = np.clip(np.searchsorted(z['beats'], times, 'right') - 1, 0, len(z['beats']) - 1)
    within = (times >= z['beats'][0]) & (times < z['beats'][bi] + z['length'][bi])
    within &= times <= z['heads'][-1]
    y = np.full(frames, -1, np.int8)
    y[within & (z['count'][bi] >= 2) & (dist > 40)] = 0
    y[within & (dist <= 20)] = 1
    return y


def position(times, z):
    bi = np.clip(np.searchsorted(z['beats'], times + 2, 'right') - 1, 0, len(z['beats']) - 1)
    phase = (times - z['beats'][bi]) / z['length'][bi]
    # Circular cell distance also handles chart-rounded heads just before a line.
    d = abs((phase[:, None] - LATTICE[None, :] + .5) % 1 - .5)
    cells = np.argmin(d, axis=1)
    p = np.zeros((len(times), 29), np.float32)
    p[np.arange(len(times)), cells] = 1
    p[np.arange(len(times)), 24 + np.minimum(z['bib'][bi], 4)] = 1
    return p


def frame_inputs(f, l, m, idx, z):
    ii = np.clip(idx[:, None] + np.arange(-4, 5), 0, len(m) - 1)
    mel = np.asarray(m[ii], np.float32).reshape(len(idx), -1)
    flux = np.maximum(0, np.asarray(m[idx], np.float32) - m[np.maximum(idx - 1, 0)])
    return dict(F=f[idx], L=l[idx], M=np.concatenate([mel, flux], axis=1),
                Pos=position(idx * 10., z), flux=flux.mean(axis=1))


def attack_sample(seed, shifted=False):
    rows, us = manifests(seed), units()
    directory = OUT / 'frames' / f's{seed}' / ('shift' if shifted else 'real')
    directory.mkdir(parents=True, exist_ok=True)
    by_rec = {}
    for u in us:
        if u['cohort'] not in ('probe-fit', 'probe-dev') or u['sha'] not in rows:
            continue
        by_rec.setdefault(u['rec'], []).append(u)
    for n, (rec, group) in enumerate(sorted(by_rec.items())):
        for u in group:
            p = directory / (u['id'] + '.npz')
            if p.exists():
                continue
            z = labels(u)
            f, l, m = features(rows[u['sha']])
            y = frame_labels(z, len(f), shifted)
            rng = np.random.default_rng(stable_seed('attacks:' + u['id']))
            cap = 200 // len(group)
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            count = min(cap, len(pos), len(neg))
            ix = np.r_[rng.choice(pos, count, replace=False), rng.choice(neg, count, replace=False)]
            d = frame_inputs(f, l, m, ix, z)
            np.savez_compressed(p, y=y[ix], idx=ix, **{k: v.astype('float16') for k, v in d.items()})
        if n % 200 == 0:
            log('attack_samples', seed=seed, shifted=shifted, recordings=n)


def context(t, before=1, after=1):
    pad = np.pad(t, ((before, after), (0, 0)))
    return np.concatenate([pad[j:j + len(t)] for j in range(before + after + 1)], axis=1)


def bar_tokens(t, z):
    out = np.zeros((len(z['bar_ids']), 4 * t.shape[1]), np.float32)
    for i, b in enumerate(z['bar_ids']):
        q = np.flatnonzero(z['bars'] == b)[:4]
        out[i, :len(q) * t.shape[1]] = t[q].ravel()
    return out
