"""Fixed cohorts, absolute tempo bands and artifact ownership for X5d."""
import os
import sys
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import base
from base import ROOT, np, json, hashlib, MusicMap, dump, append

OUT = base.OUT / 'x5d'
CACHE = ROOT / 'artifacts/n1-redesign-20261028/featcache'
BANDS = np.arange(-2, 5)
SEEDS = ('s0', 's1')


def bucket(audio):
    return int(hashlib.sha256(('n1r-probe:' + audio).encode()).hexdigest()[:8], 16) % 8


def band_of(beat_ms):
    return np.floor(np.log2(60000. / np.asarray(beat_ms) / 80.)).astype(int)


def training_rows():
    path = ROOT / 'artifacts/n1-night-20261018/runs/v0data2-20261019/curated.json'
    return {r['audio_sha256']: r for r in json.loads(path.read_text())
            if r['role'] == 'train' and r.get('keep') and bucket(r['audio_sha256']) < 3}


def cache_root(seed):
    return CACHE / f'v01-{seed}-009001'


def manifest(seed):
    path = cache_root(seed) / 'manifest.jsonl'
    if not path.exists():
        return {}
    # A producer may be appending its final row; only newline-terminated rows are committed.
    data = path.read_bytes()
    data = data[:data.rfind(b'\n') + 1]
    rows = [json.loads(line) for line in data.splitlines()]
    out = {r['sha256']: r for r in rows}
    if len(out) != len(rows):
        raise ValueError('Duplicate audio in feature manifest')
    return out


def chart_references(row):
    from ensomi_n.labels import read_osu, reference_from_charts
    refs = []
    for member in row['memberships']:
        paths = [ROOT / p for p in member['charts']]
        for path in paths:
            rel = path.relative_to(ROOT)
            if rel.parts[:2] not in (('dataset', '0'), ('dataset', 'std-taiko')) or '..' in rel.parts:
                raise ValueError('Chart outside the named open cache membership')
        refs.append(reference_from_charts([read_osu(p) for p in paths]))
    return refs


def targets(refs, times):
    """Mean one-hot chart labels; unmapped frames have zero weight."""
    target = np.zeros((len(times), len(BANDS)), np.float32)
    count = np.zeros(len(times), np.float32)
    for ref in refs:
        step = regime_steps(ref, times)
        valid = base.inside(times, ref) & np.isfinite(step) & (step > 0)
        ix = np.flatnonzero(valid)
        band = band_of(step[ix])
        if np.any((band < BANDS[0]) | (band > BANDS[-1])):
            raise ValueError('Target outside D2 band support')
        target[ix, band - BANDS[0]] += 1
        count[ix] += 1
    target /= np.maximum(count[:, None], 1)
    return target, count > 0


def regime_steps(ref, times):
    """Read half-open red-line intervals without the timing matcher's tolerance."""
    if not ref.segments:
        raise ValueError('Band targets require chart red lines')
    starts = np.array([s.start_ms for s in ref.segments])
    which = np.searchsorted(starts, times, side='right') - 1
    step = np.full(len(times), np.nan)
    valid = which >= 0
    step[valid] = np.array([s.beat_ms for s in ref.segments])[which[valid]]
    return step


def check_prereg():
    identity = json.loads((OUT / 'prereg-identity.json').read_text())
    if hashlib.sha256((OUT / 'prereg.md').read_bytes()).hexdigest() != identity['sha256']:
        raise RuntimeError('Preregistration changed')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()
