"""Interval labels: LN share (online, prefix sums) and tiled star (precomputed on the mac).

Tiled star of I = [a, a + L), L >= 30 s: the objects whose heads lie in I,
translated by -a, repeated with period L for n = ceil(240 s / L) copies (copy
order kept, source order within a copy); an LN of a non-final copy still held
at the next copy's first head in its lane ends 1 ms before that head; scored
with ``compute_mania_star_rating_20241007(objects, 4, clock_rate=1.0)``. A cut
that leaves a nonpositive hold makes the label invalid (recorded, not used).

Per chart at most nine intervals: up to four consecutive 30 s cells, up to four
consecutive 60 s cells (start chosen by a hash of the heads-only inputs), and
the whole song [0, T) when T >= 30 s. Build on the mac after the cache::

    .venv/bin/python -m ensomi_model.r2.labels --cache artifacts/r2-cache/v1 --workers 4
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

import numpy as np

from ..osu_core.difficulty import RawHitObject, compute_mania_star_rating_20241007, parse_osu_file
from .common import ContractError
from .splits import assert_fit

MIN_STAR_MS = 30_000.0
HORIZON_MS = 240_000.0
CELL_LENGTHS = (30_000.0, 60_000.0)
INTERVAL_SALT = 'r2-star-intervals-v1:'
TILING_VERSION = 'tiled-star-v1 (240s horizon, 1ms seam, 30s min, head ownership, untrimmed tails)'


def ln_share(head_counts: np.ndarray, ln_counts: np.ndarray, lo: int, hi: int):
    """LN share of the objects owned by rows [lo, hi) from per-row counts; None when empty."""
    n = int(head_counts[lo:hi].sum())
    return None if n == 0 else float(ln_counts[lo:hi].sum()) / n


def input_hash(head_ms, grid_segments, song_ms) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(head_ms, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(grid_segments, dtype=np.float64).tobytes())
    h.update(np.float64(song_ms).tobytes())
    return h.hexdigest()


def star_intervals(head_ms, grid_segments, song_ms) -> list[tuple[float, float, float]]:
    """(a, b, L) for at most nine intervals; depends only on heads, grid and T."""
    key = input_hash(head_ms, grid_segments, song_ms)
    out = []
    for L in CELL_LENGTHS:
        m = int(song_ms // L)
        r = min(4, m)
        if r <= 0:
            continue
        digest = hashlib.sha256(f'{INTERVAL_SALT}{key}:{int(L)}'.encode()).digest()
        j0 = int.from_bytes(digest[:8], 'big') % (m - r + 1)
        out += [(j * L, (j + 1) * L, L) for j in range(j0, j0 + r)]
    if song_ms >= MIN_STAR_MS:
        out.append((0.0, float(song_ms), float(song_ms)))
    seen, unique = set(), []
    for a, b, L in out:
        if (a, b) not in seen:
            seen.add((a, b))
            unique.append((a, b, L))
    return unique


def tile(objects: list[RawHitObject], a: float, length: float):
    """Tiled object list for [a, a + length) and the seam statistics; None if a cut is nonpositive."""
    owned = [o for o in objects if a <= o.start_time < a + length]
    n = math.ceil(HORIZON_MS / length)
    first = {}
    for o in owned:
        h = o.start_time - a
        first[o.column] = min(first.get(o.column, math.inf), h)
    tiled, cuts = [], 0
    for r in range(n):
        for o in owned:
            h, e = o.start_time - a + r * length, o.end_time - a + r * length
            if o.end_time > o.start_time and r < n - 1:
                v = (r + 1) * length + first[o.column]
                if e >= v:
                    e = v - 1.0
                    cuts += 1
                    if e <= h:
                        return None, dict(owned=len(owned), copies=n, cuts=cuts, invalid='nonpositive_cut')
            tiled.append(RawHitObject(start_time=h, end_time=e, column=o.column))
    return tiled, dict(owned=len(owned), copies=n, cuts=cuts)


def tiled_star(objects: list[RawHitObject], a: float, b: float):
    length = b - a
    if length < MIN_STAR_MS:
        raise ContractError('Tiled star needs an interval of at least 30 s')
    tiled, info = tile(objects, a, length)
    if tiled is None:
        return None, info
    if not tiled:
        return 0.0, info
    return float(compute_mania_star_rating_20241007(tiled, 4, clock_rate=1.0)), info


def _chart_labels(task):
    row, root, cache_root = task
    assert_fit(row)
    with np.load(Path(cache_root) / 'charts' / row['file']) as z:
        head_ms, seg, song_ms = z['head_ms'], z['grid_segments'], float(z['song_ms'])
    objects = parse_osu_file(Path(root) / row['path']).hit_objects
    labels = []
    for a, b, L in star_intervals(head_ms, seg, song_ms):
        value, info = tiled_star(objects, a, b)
        labels.append(dict(a=a, b=b, L=L, value=value, **info))
    return row['sha256'], dict(input_hash=input_hash(head_ms, seg, song_ms), labels=labels)


def build(cache_root: Path, root: Path, workers: int, limit=None):
    import pyarrow.parquet as pq
    t0 = time.time()
    index = pq.read_table(cache_root / 'index.parquet').to_pandas()
    index['eval_split'] = 'fit'  # the cache admits fit rows only (asserted when it was built)
    rows = index[['sha256', 'path', 'file', 'eval_split']].to_dict('records')
    if limit:
        rows = rows[:limit]
    out, done = {}, 0
    out_dir = cache_root / 'labels'
    out_dir.mkdir(exist_ok=True)
    partial = out_dir / 'star.partial.json.gz'
    with Pool(workers) as pool:
        for sha, value in pool.imap_unordered(_chart_labels, [(r, str(root), str(cache_root)) for r in rows],
                                              chunksize=4):
            out[sha] = value
            done += 1
            if done % 500 == 0:
                print(f'[{time.time() - t0:7.1f}s] {done}/{len(rows)}', flush=True)
                with gzip.open(partial, 'wt') as f:
                    json.dump(dict(complete=False, charts=out), f)
    payload = dict(complete=len(out) == len(index), version=TILING_VERSION,
                   calculator='compute_mania_star_rating_20241007', cache_index_sha256=
                   hashlib.sha256((cache_root / 'index.parquet').read_bytes()).hexdigest(), charts=out)
    target = out_dir / 'star.json.gz'
    with gzip.open(target, 'wt') as f:
        json.dump(payload, f)
    partial.unlink(missing_ok=True)
    n_labels = sum(len(v['labels']) for v in out.values())
    n_invalid = sum(l['value'] is None for v in out.values() for l in v['labels'])
    summary = dict(charts=len(out), complete=payload['complete'], labels=n_labels, invalid=n_invalid,
                   sha256=hashlib.sha256(target.read_bytes()).hexdigest(), wall_s=time.time() - t0)
    (out_dir / 'star_summary.json').write_text(json.dumps(summary, indent=1))
    from .receipts import write_receipt
    write_receipt(out_dir / 'receipt.json', config=dict(cache=str(cache_root), workers=workers, limit=limit),
                  inputs=dict(cache_index_sha256=payload['cache_index_sha256']), outputs=summary, seed=None,
                  device='cpu', ens_job=os.environ.get('ENS_JOB_ID'))
    print(json.dumps(summary), flush=True)
    return summary


def load_star_labels(cache_root: Path):
    """{sha256: [(a, b, value)]} for valid labels, and whether the file is complete."""
    path = Path(cache_root) / 'labels' / 'star.json.gz'
    if not path.exists():
        return {}, False
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    labels = {sha: [(l['a'], l['b'], l['value']) for l in v['labels'] if l['value'] is not None]
              for sha, v in data['charts'].items()}
    return labels, bool(data['complete'])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='artifacts/r2-cache/v1')
    p.add_argument('--root', default='.')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args(argv)
    build(Path(a.cache), Path(a.root), a.workers, a.limit)


if __name__ == '__main__':
    sys.exit(main())
