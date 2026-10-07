"""Chart vectors and a fit_train-only nearest-skeleton donor prior.

The ten coordinates are standardised on fit_train. The hand coordinate uses
both mirror orientations, so its mean is exactly one half and a mirror negates
its standardised value. Missing LN lengths become the training mean. Noise is
the standard deviation of first-half minus second-half coordinates on fit_train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import GridArrays, grid_from_arrays

THETA_FILE = 'artifacts/r2-theta/theta-v1.parquet'
COORDINATES = ('nh', 'c3', 'c4', 'jack', 'held', 'ln_share', 'ln_length', 'pent', 'loop', 'hand')
WHOLE = ['logK', 'song_s', 'rows_per_s', 'rows_per_beat', 'lead_frac', 'mapped_frac', 'bpm_med', 'bpm_sd', 'segments',
         'lg_q10', 'lg_q25', 'lg_q50', 'lg_q75', 'lg_q90', 'lb_q10', 'lb_q50', 'lb_q90', 'le40', 'le60', 'le80',
         'le120', 'le200', 'gt500', 'gt1000', 'reg', 'lg_sd', 'snap1', 'snap2', 'snap3', 'snap4', 'snap6', 'snap8',
         'snap12', 'snap16', 'snap_off', 'dens4_q10', 'dens4_q50', 'dens4_q90', 'dens4_cv', 'dens4_zero',
         'dens16_q10', 'dens16_q50', 'dens16_q90', 'dens16_cv', 'dens16_zero', 'isolated', 'breaks_per_min',
         'dense_row_frac']


def decode(actions):
    """Vectorised four-lane automaton: held-before, heads and LN heads."""
    a = np.asarray(actions, dtype=np.int64)
    held = np.zeros(a.shape, dtype=bool)
    index = np.arange(len(a))
    for lane in range(4):
        c = a[:, lane]
        reset = (c == 1) | (c == 3) | (c == 4)
        toggles = np.cumsum(c == 2)
        last = np.maximum.accumulate(np.where(reset, index, -1))
        base = np.where(last >= 0, c[np.maximum(last, 0)] == 4, 0)
        previous = np.where(last >= 0, toggles[np.maximum(last, 0)], 0)
        held[1:, lane] = ((base + toggles - previous) % 2)[:-1]
    return held, np.where(held, a >= 3, (a == 1) | (a == 2)), np.where(held, a == 4, a == 2)


def length_rows(head_ms, song_ms, grid, actions, gap, held, ln_head):
    """Birth rows and log2 beat lengths of closed holds, including EOS releases."""
    times = np.append(head_ms, song_ms)[:len(actions)]
    births = np.maximum.accumulate(np.where(ln_head, np.arange(len(actions))[:, None], -1), axis=0)
    before = np.vstack((np.full((1, 4), -1), births[:-1]))
    closed = held & (actions != 0)
    rows = before[closed]
    release = np.where(actions == 1, times[:, None], gap)[closed]
    length = np.log2(np.maximum(grid.beat(release) - grid.beat(times[rows]), 1e-4))
    return rows, length


def coordinates(held, heads, ln_heads, births, lengths, start, stop):
    """Chart statistics on a row span; entropy and best-period loop rates average 64-row windows."""
    h, l = heads[start:stop], ln_heads[start:stop]
    nh = h.sum(1)
    count = max(int(h.sum()), 1)
    previous = np.vstack((np.zeros((1, 4), bool), h[:-1]))
    patterns = (h * (1 << np.arange(4))).sum(1)
    pents, loops = [], []
    for a in range(0, len(h) - 63, 64):
        hist = np.bincount(patterns[a:a + 64], minlength=16)[1:]
        p = hist / max(hist.sum(), 1)
        pents.append(-(p[p > 0] * np.log2(p[p > 0])).sum())
        loops.append(max((patterns[max(a, lag):a + 64] ==
                          patterns[max(a, lag) - lag:a + 64 - lag]).mean() for lag in range(2, 17)))
    if not pents:
        hist = np.bincount(patterns, minlength=16)[1:]
        p = hist / max(hist.sum(), 1)
        pents = [-(p[p > 0] * np.log2(p[p > 0])).sum()]
        loops = [max(((patterns[lag:] == patterns[:-lag]).mean()
                      for lag in range(2, min(17, len(h)))), default=0.0)]
    length = lengths[(births >= start) & (births < stop)]
    return np.array([nh.mean(), (nh >= 3).mean(), (nh == 4).mean(), (h & previous).sum() / count,
                     held[start:stop].mean(), l.sum() / count, np.median(length) if len(length) else np.nan,
                     np.mean(pents), np.mean(loops), h[:, :2].sum() / count], dtype=np.float64)


def chart_coordinates(head_ms, song_ms, grid, actions, gap):
    K = len(head_ms)
    held, head, ln = decode(actions)
    births, lengths = length_rows(head_ms, song_ms, grid, actions, gap, held, ln)
    whole = coordinates(held, head, ln, births, lengths, 0, K)
    halves = [coordinates(held, head, ln, births, lengths, a, b) for a, b in ((0, K // 2), (K // 2, K))]
    return whole, halves[0] - halves[1]


def prefix_theta(table, head_ms, song_ms, grid, actions, gap):
    """Standardised theta of the committed rows only (continuations of a real prefix)."""
    held, head, ln = decode(actions)
    births, lengths = length_rows(head_ms, song_ms, grid, actions, gap, held, ln)
    raw = coordinates(held, head, ln, births, lengths, 0, len(actions))
    return np.nan_to_num((raw - table.mean) / table.sd)


def skeleton_features(head, song_ms, grid):
    """The 48 timing-only descriptors used by the nearest-neighbour prior."""
    head = np.asarray(head)
    beats = grid.beat(head)
    gaps, beat_gaps = np.diff(head), np.diff(beats)
    span, beat_span = max(head[-1] - head[0], 1.0), max(beats[-1] - beats[0], 1e-3)
    bpm = grid.bpm(head)
    out = dict(logK=np.log(len(head)), song_s=song_ms / 1000, rows_per_s=len(head) * 1000 / span,
               rows_per_beat=len(head) / beat_span, lead_frac=head[0] / song_ms, mapped_frac=span / song_ms,
               bpm_med=np.median(bpm), bpm_sd=np.std(np.log(bpm)), segments=np.log1p(len(grid.offsets)))
    lg, lb = np.log(np.maximum(gaps, 1)), np.log(np.maximum(beat_gaps, 1e-3))
    for q in (10, 25, 50, 75, 90):
        out[f'lg_q{q}'] = np.percentile(lg, q) if len(lg) else np.nan
    for q in (10, 50, 90):
        out[f'lb_q{q}'] = np.percentile(lb, q) if len(lb) else np.nan
    for t in (40, 60, 80, 120, 200):
        out[f'le{t}'] = (gaps <= t).mean() if len(gaps) else np.nan
    out.update(gt500=(gaps > 500).mean() if len(gaps) else np.nan,
               gt1000=(gaps > 1000).mean() if len(gaps) else np.nan,
               reg=(np.abs(np.diff(lg)) < .05).mean() if len(lg) > 1 else np.nan,
               lg_sd=lg.std() if len(lg) else np.nan)
    fraction = beats - np.floor(beats)
    snap = np.zeros(len(head), dtype=int)
    for d in (16, 12, 8, 6, 4, 3, 2, 1):
        snap = np.where(np.abs(fraction * d - np.round(fraction * d)) < .04 * (d / 4 if d > 4 else 1), d, snap)
    for d in (1, 2, 3, 4, 6, 8, 12, 16):
        out[f'snap{d}'] = (snap == d).mean()
    out['snap_off'] = (snap == 0).mean()
    for width in (4, 16):
        edges = np.arange(beats[0], max(beats[-1], beats[0] + 1e-6) + width, width)
        counts = np.histogram(beats, edges)[0] / width
        for q in (10, 50, 90):
            out[f'dens{width}_q{q}'] = np.percentile(counts, q)
        out[f'dens{width}_cv'] = counts.std() / max(counts.mean(), 1e-6)
        out[f'dens{width}_zero'] = (counts == 0).mean()
    isolated = np.zeros(len(head), bool)
    isolated[1:-1] = (beat_gaps[:-1] >= 1 - 1e-3) & (beat_gaps[1:] >= 1 - 1e-3)
    out.update(isolated=isolated.mean(), breaks_per_min=(beat_gaps > 4).sum() / (span / 60000),
               dense_row_frac=(np.append(gaps, np.inf) <= 60).mean())
    return np.array([out[c] for c in WHOLE])


class ThetaTable:
    def __init__(self, table, metadata):
        self.table, self.metadata = table, metadata
        self.mean, self.sd = np.array(metadata['mean']), np.array(metadata['sd'])
        self.noise_sd = np.array(metadata['noise_sd'])
        self.vectors = table.set_index('sha256')[[f'theta_{c}' for c in COORDINATES]]
        self.donors = table[table.role == 'fit_train'].sort_values('sha256').reset_index(drop=True)
        self.skeleton_mean = np.array(metadata['skeleton_mean'])
        self.skeleton_sd = np.array(metadata['skeleton_sd'])
        self.skeletons = self.standardize_skeleton(self.donors[WHOLE].to_numpy(float))

    @classmethod
    def load(cls, path=THETA_FILE):
        table = pq.read_table(path)
        return cls(table.to_pandas(), json.loads(table.schema.metadata[b'r2_theta']))

    @classmethod
    def fit(cls, table):
        table = table.copy()
        train = table.role == 'fit_train'
        raw = table[[f'raw_{c}' for c in COORDINATES]].to_numpy(float)
        mean, sd = np.nanmean(raw[train], axis=0), np.nanstd(raw[train], axis=0)
        mean[-1] = .5
        sd[-1] = np.sqrt(np.mean((raw[train, -1] - .5) ** 2))
        half_diff = table[[f'half_{c}' for c in COORDINATES]].to_numpy(float)[train] / sd
        noise = np.nanstd(half_diff, axis=0) / np.sqrt(2)  # SD of the two half values about their mean
        skel = table.loc[train, WHOLE].to_numpy(float)
        sm, ss = np.nanmean(skel, axis=0), np.nanstd(skel, axis=0) + 1e-9
        values = np.nan_to_num((raw - mean) / sd)
        for i, c in enumerate(COORDINATES):
            table[f'theta_{c}'] = values[:, i]
        metadata = dict(coordinates=list(COORDINATES), fit_role='fit_train', mean=mean.tolist(), sd=sd.tolist(),
                        noise_sd=noise.tolist(), skeleton_mean=sm.tolist(), skeleton_sd=ss.tolist())
        return cls(table, metadata)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(self.table, preserve_index=False)
        metadata = {**(table.schema.metadata or {}), b'r2_theta': json.dumps(self.metadata).encode()}
        pq.write_table(table.replace_schema_metadata(metadata), path)

    def vector(self, sha):
        return self.vectors.loc[sha].to_numpy(dtype=np.float64)

    def training_draw(self, sha, rng, dropout=.3):
        if rng.random() < dropout:
            return None
        return self.vector(sha) + rng.normal(size=len(COORDINATES)) * self.noise_sd

    def standardize_skeleton(self, raw):
        return np.clip(np.nan_to_num((raw - self.skeleton_mean) / self.skeleton_sd), -6, 6)

    def sample(self, *, head_ms, song_ms, grid, seed, source_sha256=None):
        raw = (self.table.set_index('sha256').loc[source_sha256, WHOLE].to_numpy(float)
               if source_sha256 is not None else skeleton_features(head_ms, song_ms, grid))
        distances = ((self.skeletons - self.standardize_skeleton(raw)) ** 2).sum(1)
        nearest = np.argsort(distances, kind='stable')[:32]
        digest = hashlib.sha256(np.asarray(head_ms, dtype='<f8').tobytes())
        digest.update(np.asarray([song_ms], dtype='<f8').tobytes())
        digest.update(str(int(seed)).encode())
        rng = np.random.default_rng(int.from_bytes(digest.digest()[:8], 'little'))
        donor = self.donors.iloc[int(rng.choice(nearest))]
        return self.vector(donor.sha256), dict(donor_sha256=donor.sha256)


def resolve_theta(mode, *, head_ms=None, song_ms=None, grid=None, seed=954,
                  table=None, source_sha256=None):
    if mode is None or (isinstance(mode, str) and mode == 'unknown'):
        return None, dict(mode='unknown', value=None)
    if not isinstance(mode, str):
        value = np.asarray(mode, dtype=np.float64)
        if value.shape != (10,) or not np.isfinite(value).all():
            raise ValueError('theta must be a finite standardised 10-vector')
        return value, dict(mode='vector', value=value.tolist())
    if mode not in ('prior', 'oracle'):
        raise ValueError('theta must be prior, unknown, oracle, or a standardised 10-vector')
    table = table if isinstance(table, ThetaTable) else ThetaTable.load(table or THETA_FILE)
    if mode == 'prior':
        value, record = table.sample(head_ms=head_ms, song_ms=song_ms, grid=grid, seed=seed,
                                     source_sha256=source_sha256)
    else:
        value, record = table.vector(source_sha256), dict(source_sha256=source_sha256)
    return value, dict(mode=mode, value=value.tolist(), **record)


def build_table(cache='artifacts/r2-cache/v1', skeleton='artifacts/r2-ln-level-20261007/data/charts.parquet',
                output=THETA_FILE, *, limit=None):
    from .cache import load_chart
    index = pd.read_parquet(Path(cache) / 'index.parquet')
    skeletons = pd.read_parquet(skeleton)
    table = skeletons[skeletons.role.isin(('fit_train', 'fit_dev'))].sort_values('sha256').reset_index(drop=True)
    if limit is not None:
        table = table.head(limit).copy()
    files = index.set_index('sha256').file
    rows, differences = [], []
    started = time.monotonic()
    for i, sha in enumerate(table.sha256):
        dec = load_chart(Path(cache) / 'charts' / files[sha])
        grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
        raw, delta = chart_coordinates(dec.head_ms, dec.song_ms, grid, dec.actions, dec.gap_release_ms)
        rows.append(raw)
        differences.append(delta)
        if (i + 1) % 1000 == 0:
            print(f'theta {i + 1}/{len(table)} {time.monotonic() - started:.1f}s', flush=True)
    for i, coordinate in enumerate(COORDINATES):
        table[f'raw_{coordinate}'] = np.asarray(rows)[:, i]
        table[f'half_{coordinate}'] = np.asarray(differences)[:, i]
    result = ThetaTable.fit(table)
    result.save(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='artifacts/r2-cache/v1')
    parser.add_argument('--skeleton', default='artifacts/r2-ln-level-20261007/data/charts.parquet')
    parser.add_argument('--out', default=THETA_FILE)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    table = build_table(args.cache, args.skeleton, args.out, limit=args.limit)
    print(json.dumps(dict(path=args.out, charts=len(table.table), fit_train=len(table.donors))))


if __name__ == '__main__':
    main()
