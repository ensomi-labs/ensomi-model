"""Difficulty labels: the cache relabel at the plan v4 cell lengths, through ``properties``.

Cells (version ``cells-v2``): for each length L in 30, 60, 120 s and each phase p in 0, 10,
20 s, the cells [p + i L, p + (i + 1) L) that end at or before T; plus the whole song [0, T]
when T >= 30 s. Each value is Difficulty_nu of the cache representation (the objects the
cached decisions expand to), so a label equals the readout of the same chart. An undefined
readout (no object in the cell, a nonpositive seam cut) is stored as ``null`` and never used.

Build on the mac after the cache (writes ``labels/star-v2.json.gz``, ``labels/star-v2_summary.json``
and ``labels/receipt-star-v2.json``; the v1 file ``labels/star.json.gz`` is kept and never written)::

    .venv/bin/python -m ensomi_model.r2.labels --cache artifacts/r2-cache/v1 --workers 4
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

import numpy as np

from .properties import CALCULATOR, MIN_STAR_MS, NU, NU_HASH, PROPERTIES_SOURCE_SHA256, TILING_VERSION, difficulty

CELL_LENGTHS_S = (30, 60, 120)
CELL_PHASES_S = (0, 10, 20)
CELLS_VERSION = 'cells-v2 (30/60/120 s at phases 0/10/20 s, ending by T; whole song [0, T] when T >= 30 s)'
LABEL_FILE = 'star-v2.json.gz'


def cells_of(song_ms: float):
    """[(a, b, L_s or 'whole', phase_s or None)] in a fixed order."""
    out = []
    for L in CELL_LENGTHS_S:
        for p in CELL_PHASES_S:
            a = 1000.0 * p
            while a + 1000.0 * L <= song_ms:
                out.append((a, a + 1000.0 * L, L, p))
                a += 1000.0 * L
    if song_ms >= MIN_STAR_MS:
        out.append((0.0, float(song_ms), 'whole', None))
    return out


def chart_objects(z):
    from .cache import objects_from_decisions
    return objects_from_decisions(z['head_ms'], z['actions'].astype(np.int64), z['gap_release_ms'], float(z['song_ms']))


def _chart_labels(task):
    row, cache_root = task
    with np.load(Path(cache_root) / 'charts' / row['file']) as z:
        song_ms = float(z['song_ms'])
        objects = chart_objects(z)
    labels = []
    for a, b, L, p in cells_of(song_ms):
        value, info = difficulty(objects, a, b, song_ms)
        labels.append(dict(a=a, b=b, L=L, phase=p, value=value, owned=info.get('owned'), cuts=info.get('cuts'),
                           invalid=info.get('invalid')))
    return row['sha256'], labels


def relabel(cache_root: Path, workers: int, limit=None):
    import pyarrow.parquet as pq
    t0 = time.time()
    index = pq.read_table(cache_root / 'index.parquet').to_pandas()
    rows = index[['sha256', 'file']].to_dict('records')
    if limit:
        rows = rows[:limit]
    out_dir = cache_root / 'labels'
    out_dir.mkdir(exist_ok=True)
    target = out_dir / LABEL_FILE
    if target.exists() and not limit:
        raise SystemExit(f'{target} exists; move it aside to relabel')
    out, done = {}, 0
    with Pool(workers) as pool:
        for sha, labels in pool.imap_unordered(_chart_labels, [(r, str(cache_root)) for r in rows], chunksize=4):
            out[sha] = labels
            done += 1
            if done % 1000 == 0:
                print(f'[{time.time() - t0:7.1f}s] {done}/{len(rows)}', flush=True)
    payload = dict(complete=len(out) == len(index), cells=CELLS_VERSION, nu=NU, nu_hash=NU_HASH,
                   properties_sha256=PROPERTIES_SOURCE_SHA256, tiling=TILING_VERSION, calculator=CALCULATOR,
                   source='cache representation (objects_from_decisions of the cached decisions)',
                   cache_index_sha256=hashlib.sha256((cache_root / 'index.parquet').read_bytes()).hexdigest(),
                   charts=out)
    if limit:
        target = out_dir / f'star-v2-limit{limit}.json.gz'
    with gzip.open(target, 'wt') as f:
        json.dump(payload, f)
    labels = [l for v in out.values() for l in v]
    by_len = {}
    for l in labels:
        key = str(l['L'])
        s = by_len.setdefault(key, dict(labels=0, undefined=0))
        s['labels'] += 1
        s['undefined'] += l['value'] is None
    summary = dict(charts=len(out), complete=payload['complete'], labels=len(labels),
                   undefined=sum(l['value'] is None for l in labels), by_length=by_len, nu_hash=NU_HASH,
                   file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), wall_s=time.time() - t0,
                   workers=workers)
    (out_dir / target.name.replace('.json.gz', '_summary.json')).write_text(json.dumps(summary, indent=1))
    from .receipts import write_receipt
    write_receipt(out_dir / f'receipt-{target.name.replace(".json.gz", "")}.json',
                  config=dict(cache=str(cache_root), workers=workers, limit=limit),
                  inputs=dict(cache_index_sha256=payload['cache_index_sha256']), outputs=summary, seed=None,
                  device='cpu', ens_job=os.environ.get('ENS_JOB_ID'))
    print(json.dumps(summary), flush=True)
    return summary


def label_path(cache_root) -> Path:
    return Path(cache_root) / 'labels' / LABEL_FILE


def load_cells(cache_root, path=None):
    """({sha: {(L_s, phase_s): [(a, b, value)], 'whole': [(0, T, value)]}}, complete, file sha256).

    Undefined labels are left out. Missing file: ({}, False, None)."""
    path = Path(path) if path else label_path(cache_root)
    if not path.exists():
        return {}, False, None
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw))
    if data.get('nu_hash') != NU_HASH:
        raise SystemExit(f'{path} was built under nu {data.get("nu_hash")}, not {NU_HASH}')
    out = {}
    for sha, labels in data['charts'].items():
        cells = {}
        for l in labels:
            if l['value'] is None:
                continue
            key = 'whole' if l['L'] == 'whole' else (int(l['L']), int(l['phase']))
            cells.setdefault(key, []).append((float(l['a']), float(l['b']), float(l['value'])))
        out[sha] = cells
    return out, bool(data['complete']), hashlib.sha256(raw).hexdigest()


def load_star_labels_v1(cache_root):
    """The v1 labels (from the original ``.osu``; 0.0 for an empty cell): {sha: [(a, b, value)]}."""
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
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args(argv)
    relabel(Path(a.cache), a.workers, a.limit)


if __name__ == '__main__':
    sys.exit(main())
