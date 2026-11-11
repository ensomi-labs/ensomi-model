"""Read only the explicitly selected unsealed evaluation and pair inputs."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

NIGHT = Path('artifacts/n1-night-20261018/runs')
CACHE = Path('artifacts/n1-redesign-20261028/featcache')
PAIRS = Path('artifacts/bulk/mapper-spread-20261021/run/pairs.jsonl')


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(clean_json(data), indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    with Path(path).open('w') as f:
        for r in rows:
            f.write(json.dumps(clean_json(r), allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def storage_guard(out):
    free = shutil.disk_usage(out).free
    size = sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())
    if free < 100 * 1024**3 or size > 2_000_000_000:
        raise RuntimeError(f'storage limit: free={free}, new_bytes={size}')
    return dict(free_bytes=free, output_bytes=size)


def allowed_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or p.parts[:2] not in [('dataset', '0'), ('dataset', 'std-taiko')]:
        raise ValueError(f'not a selected chart/audio path: {relative}')
    return root / p


def items(root, panels=('panel', 'val_hard')):
    from ensomi_n.v0.evaluate import eval_items
    if set(panels) - {'panel', 'val_hard'}:
        raise ValueError('only the two unsealed panels may be read')
    split = json.loads((root / NIGHT / 'r0-20261019a/split.json').read_text())
    selected = {p: split[p] for p in panels}
    for entries in selected.values():
        for x in entries:
            allowed_path(root, x['audio_rel'])
            for c in x['charts']:
                allowed_path(root, c)
    data = eval_items(selected, panels, root)
    raw = {(p, x['audio_sha256'], tuple(sorted(x['charts']))): x for p, xs in selected.items() for x in xs}
    for x in data:
        if 'error' not in x:
            x['audio_rel'] = raw[x['panel'], x['item'], tuple(sorted(x['charts']))]['audio_rel']
    return data


def maps(root, seed, panel):
    from ensomi_n.mapformat import MusicMap
    p = root / f'artifacts/bulk/n1-v01-analysis/v01-s{seed}-20261020/last3/maps-009001-{panel}.jsonl.gz'
    return {r['key']: MusicMap.from_json(r['map']) for r in read_jsonl(p)}


def scores(root, seed, panel):
    return read_jsonl(root / NIGHT / f'v01-s{seed}-20261020/eval/items-{panel}.jsonl')


def pair_inputs(root):
    """Only sets named in the already seal-filtered mapper-pair output are opened."""
    from ensomi_n.labels import read_osu, reference_from_charts
    rows = read_jsonl(root / PAIRS)
    if any(set(r['split'].split('|')) - {'fit_train', 'fit_dev', 'cal', 'std-taiko-design'} for r in rows):
        raise ValueError('pair output contains a sealed or unknown split')
    refs, charts = {}, {}
    for key in sorted({r[k] for r in rows for k in ('a', 'b')}):
        parts = Path(key).parts
        if len(parts) < 2 or parts[0] not in ('0', 'std-taiko'):
            raise ValueError(f'unknown pair key {key}')
        folder = allowed_path(root, str(Path('dataset', *parts[:2])))
        cs = [read_osu(p) for p in sorted(folder.glob('*.osu'))]
        if len(parts) > 2:
            cs = [c for c in cs if Path(c.audio).stem == '/'.join(parts[2:])]
        if len({c.audio.casefold() for c in cs}) != 1:
            raise ValueError(f'ambiguous pair audio for {key}')
        charts[key] = cs
        refs[key] = reference_from_charts(cs, key)
    return rows, refs, charts
