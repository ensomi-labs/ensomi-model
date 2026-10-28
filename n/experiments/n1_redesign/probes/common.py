"""Input boundaries, deterministic sampling and recording-balanced statistics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone
from functools import lru_cache
from types import SimpleNamespace

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path('/Users/l/ensomi/ensomi-model')
OUT = ROOT / 'artifacts/n1-redesign-20261028/probes'
CACHE = ROOT / 'artifacts/n1-redesign-20261028/featcache'
PREV = ROOT / 'artifacts/astra-consult-20261021/astra/exp'
LATTICE = np.array(sorted(set(range(0, 48, 3)) | set(range(0, 48, 4)))) / 48
assert len(LATTICE) == 24


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


def dump(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, default=serial) + '\n')
    tmp.replace(path)


def log(event, **kw):
    row = dict(time=utc(), event=event, **kw)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'job.jsonl').open('a') as f:
        f.write(json.dumps(row, default=serial) + '\n')
    print(json.dumps(row, default=serial), flush=True)


def stable_seed(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


def split(sha):
    h = stable_seed('n1r-probe:' + sha) % 8
    return 'probe-fit' if h in (0, 1) else 'probe-dev' if h == 2 else None


@lru_cache(maxsize=1)
def exclusion():
    from ensomi_n.eval.census import load_exclusion
    from ensomi_n.data.manifest import read_csv_shas
    e = load_exclusion(ROOT / 'artifacts/n1-eval/hard-holdout-v1',
                       '00771c6491ced746bc1dd536c0cd3ef453a01798a658ebe3330ab89edec327bb')
    for p in ('artifacts/timing-20261009/opt/holdout/holdout.csv',
              'artifacts/timing-20261009/opt/ft/pop/holdout2.csv'):
        e['audio_sha256'] |= read_csv_shas(ROOT / p)
    # Identifier metadata only; this also covers R2-heldout and sealed gold
    # already marked excluded by N's curation.
    recs = json.loads((ROOT / 'artifacts/n1-night-20261018/runs/r0-20261019a/recordings.json').read_text())
    for r in recs:
        if r['role'].startswith('excluded:'):
            e['recordings'].add(r['rec'])
            if r.get('audio_sha256'):
                e['audio_sha256'].add(r['audio_sha256'])
    import pyarrow.parquet as pq
    from ._exclusion import build_exclusion
    run = ROOT / 'artifacts/n1-redesign-20261028/census/run.json'
    args = SimpleNamespace(**json.loads(run.read_text())['args'])
    corpus = pq.read_table(ROOT / args.corpus, columns=[
        'group_id', 'set_dir', 'audio_sha256', 'eval_split']).to_pylist()
    audit = {}
    e['folders'] = build_exclusion(args, ROOT, corpus, audit)
    dump(OUT / 'exclusion-audit.json', dict(**audit, census_run_sha256=digest(run)))
    return e


def assert_open(row, e):
    sha = row.get('sha256', row.get('audio_sha256'))
    if sha in e['audio_sha256'] or row.get('rec') in e['recordings']:
        raise ValueError('Sealed recording in input list')
    if set(map(str, row.get('set_ids', []))) & set(map(str, e['set_ids'])):
        raise ValueError('Sealed set in input list')
    if row.get('audio_rel') and str(Path(row['audio_rel']).parent) in e.get('folders', set()):
        raise ValueError('Sealed folder in input list')
    if any(str(Path(p).parent) in e.get('folders', set()) for p in row.get('charts', [])):
        raise ValueError('Sealed chart folder in input list')
    for m in row.get('memberships', []):
        assert_open(dict(m, sha256=sha), e)


def manifests(seed, ready=True):
    dirs = [CACHE / f'v01-s{seed}-009001']
    if seed == 0 and (CACHE / 'v01-s1-009001/DONE').exists():
        done = json.loads((CACHE / 'v01-s1-009001/DONE').read_text())
        ext_manifest = CACHE / 'v01-s0-009001-ext/manifest.jsonl'
        assert len(ext_manifest.read_text().splitlines()) == done['0']['inputs']
        dirs.append(CACHE / 'v01-s0-009001-ext')
    if ready:
        marker = CACHE / 'summary.md' if seed == 0 else dirs[0] / 'DONE'
        if not marker.is_file():
            raise FileNotFoundError(f'Cache completion marker missing: {marker}')
    e, rows, rejected = exclusion(), {}, []
    for d in dirs:
        for line in (d / 'manifest.jsonl').read_text().splitlines():
            r = json.loads(line)
            try:
                assert_open(r, e)
            except ValueError as exc:
                rejected.append(dict(sha256=r['sha256'], reason=str(exc)))
                continue
            r['cache_dir'] = str(d)
            rows[r['sha256']] = r
    dump(OUT / f'excluded-manifest-s{seed}.json', rejected)
    return rows


@lru_cache(maxsize=1)
def extension_mel_index():
    path = ROOT / 'artifacts/n1-redesign-20261028/gpu/inference-songs.json'
    index = json.loads(path.read_text())
    inputs_path = OUT / 'inputs.json'
    inputs = json.loads(inputs_path.read_text())
    inputs['extension_mel_index'] = dict(path=str(path), sha256=digest(path),
        use='Read-only fallback for manifest-listed extension recordings missing from the bulk Mel cache')
    dump(inputs_path, inputs)
    return index


def mel_for(r):
    from ensomi_n.features.cache import mel_path
    path = mel_path(ROOT / 'artifacts/bulk/n1-cache/mel', r['sha256'])
    if not path.exists():
        assert r.get('ext'), 'A non-extension input is missing its bulk Mel cache'
        entry = extension_mel_index()[r['sha256']]
        expected = mel_path(ROOT / 'artifacts/n1-redesign-20261028/gpu/input-cache/mel', r['sha256'])
        assert Path(entry['mel']) == expected and entry['item'] == r['sha256']
        assert entry['frames'] == r['frames']
        path = expected
    return np.load(path, mmap_mode='r', allow_pickle=False)


def features(r):
    assert Path(r['npz']) == Path(r['sha256'][:2]) / (r['sha256'] + '.npz')
    p = Path(r['cache_dir']) / r['npz']
    with np.load(p, allow_pickle=False) as z:
        f, l = z['feat'].astype('float32'), z['logits'].astype('float32')
        assert float(z['t0_ms']) == 0 and float(z['fps']) == 100
    m = mel_for(r)
    assert f.shape == (len(m), 160) and l.shape == (len(m), 23)
    assert len(f) == r['frames']
    assert r['t0_ms'] == 0 and r['fps'] == 100
    return f, l, m


def interp(x, times):
    t = np.clip(np.asarray(times) / 10, 0, len(x) - 1)
    lo = t.astype('int64')
    w = (t - lo)[..., None]
    return (x[lo] * (1 - w) + x[np.minimum(lo + 1, len(x) - 1)] * w).astype('float32')


def nearest(times, grid):
    t, g = np.asarray(times), np.asarray(grid)
    if not len(g):
        return np.full(t.shape, np.inf), np.zeros(t.shape, int)
    j = np.clip(np.searchsorted(g, t), 0, len(g) - 1)
    k = np.maximum(j - 1, 0)
    j = np.where(abs(t - g[k]) <= abs(t - g[j]), k, j)
    return t - g[j], j


def auc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p)
    n = int(y.sum())
    if n == 0 or n == len(y):
        return float('nan')
    return float((rankdata(p)[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def interval(values, seed=20261028):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, mean=None, ci95=None, halfwidth=None)
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(len(v), size=len(v))].mean() for _ in range(1000)])
    lo, hi = np.quantile(bs, [.025, .975])
    return dict(n=len(v), mean=float(v.mean()), ci95=[float(lo), float(hi)],
                halfwidth=float((hi - lo) / 2))


def song_auc(y, p, rec, minimum=1):
    rows = {}
    for r in np.unique(rec):
        q = rec == r
        yy = y[q]
        if np.sum(yy == 0) >= minimum and np.sum(yy == 1) >= minimum:
            rows[str(r)] = auc(yy, p[q])
    return dict(**interval(list(rows.values())), recordings=rows,
                pooled_auc=auc(y, p))


def paired(a, b):
    keys = sorted(set(a) & set(b))
    return interval([a[k] - b[k] for k in keys])


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def unit_path(u):
    return OUT / 'labels' / (u['id'] + '.npz')


def labels(u):
    with np.load(unit_path(u), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def units():
    return json.loads((OUT / 'units.json').read_text())


def prereg_check():
    p = OUT / 'prereg.md'
    record = json.loads((OUT / 'prereg-lock.json').read_text())
    assert digest(p) == record['sha256'], 'Preregistration changed after freezing'
    return record
