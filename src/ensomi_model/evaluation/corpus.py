"""Inventory of every ``.osu`` file under ``dataset/`` for corpus-referenced evaluation.

One row per file: identity and hashes, the acquisition it came from, header
fields and osu! API metadata, and for 4K osu!mania charts the computed star
rating and a summary of the chart on its own beat grid. Files are joined into
song groups and each group gets an evaluation split. The Parquet output is
committed so that every machine and every later operator reads the same corpus.

Run from the repository root on the machine that holds ``dataset/``::

    python -m ensomi_model.evaluation.corpus --r1-catalog <catalog.json>
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..osu_core.difficulty import compute_mania_star_rating_20241007, parse_osu_file
from ..osu_core.hitobjects import parse_mania_hit_objects
from ..osu_core.metadata import parse_osu_metadata
from ..osu_core.timing import parse_red_timing_points
from .beats import COMMON_DENOMINATORS, LN_RELEASE, BeatGrid, chart_events, renotate

FORMAT = 'ensomi-eval/corpus-inventory-v1'
SPLIT_SALT = 'ensomi-eval-split-v1:'
HELDOUT_PERCENT, CALIBRATION_PERCENT = 15, 15
DEFAULT_OUTPUT = 'data/evaluation/corpus-inventory.parquet'
ERROR_SEPARATOR = ' | '

_SIZE_MARKERS = ('tv size', 'tv ver', 'tv version', 'short ver', 'short version', 'cut ver', 'cut version',
                 'game ver', 'game version', 'game size', 'full ver', 'full version', 'extended ver',
                 'extended version', 'movie ver', 'anime ver', 'size ver', 'sped up ver')
_TRAILING_GROUP = re.compile(r'\s*(\(([^()]*)\)|\[([^\[\]]*)\]|-([^-]+)-|~([^~]+)~|～([^～]+)～)\s*$')


def normalized(value) -> str:
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).casefold().split())


def song_title(title) -> str:
    """Normalised title without trailing size or cut markers such as "(TV Size)"."""
    text = normalized(title)
    while (match := _TRAILING_GROUP.search(text)) is not None:
        inner = next(g for g in match.groups()[1:] if g is not None).replace('.', '')
        if not any(marker in inner for marker in _SIZE_MARKERS) or not text[:match.start()].strip():
            break
        text = text[:match.start()].strip()
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _on_grid(snap: np.ndarray) -> float | None:
    return float(np.isin(snap, COMMON_DENOMINATORS).mean()) if len(snap) else None


def _same_coords(a: dict, b: dict) -> bool:
    return (np.array_equal(a['snap'], b['snap']) and np.array_equal(a['bar'], b['bar'])
            and np.allclose(a['beat'], b['beat'], rtol=0, atol=1e-9)
            and np.allclose(a['bar_beat'], b['bar_beat'], rtol=0, atol=1e-9))


def _reason(exc: Exception, path: Path) -> str:
    """Exception text without the file's absolute path, which differs per machine."""
    return str(exc).replace(str(path), '<file>')


def _mania_summary(path: Path) -> dict:
    """Counts, computed star and beat-grid summary of one 4K chart; errors per stage."""
    out, errors = {}, []
    try:
        objects = parse_mania_hit_objects(path, expected_key_count=4)
    except Exception as exc:  # noqa: BLE001 - recorded per file, never raised
        return dict(error=f'objects: {_reason(exc, path)}')
    holds = sum(o.kind.value == 'HOLD' for o in objects)
    out.update(n_taps=len(objects) - holds, n_lns=holds)
    if objects:
        out.update(first_ms=min(o.start_time_ms for o in objects), last_ms=max(o.end_time_ms for o in objects))
    try:
        parsed = parse_osu_file(path)
        out['star'] = compute_mania_star_rating_20241007(parsed.hit_objects, 4, clock_rate=1.0)
    except Exception as exc:  # noqa: BLE001
        errors.append(f'star: {_reason(exc, path)}')
    try:
        points = parse_red_timing_points(path)
        if not points:
            raise ValueError('no red timing point')
        if not objects:
            raise ValueError('no hit objects')
        span = (out['first_ms'], out['last_ms'])
        grid = BeatGrid.from_timing_points(points, span)
        events = chart_events(objects, grid)
        coords, releases = events.coords, events.kind == LN_RELEASE
        bpms = 60000.0 / np.array([s.beat_length_ms for s in grid.segments])
        back = grid.time_of(coords['segment'], coords['beat'])
        out.update(n_red_lines=len(grid.segments), distinct_bpms=len(set(np.round(bpms, 2))),
                   min_bpm=float(bpms.min()), max_bpm=float(bpms.max()), dominant_bpm=grid.dominant_bpm,
                   fold=grid.fold, canonical_bpm=grid.canonical_bpm,
                   head_on_grid=_on_grid(coords['snap'][~releases]),
                   release_on_grid=_on_grid(coords['snap'][releases]),
                   roundtrip_max_ms=float(np.abs(back - events.time_ms).max()))
        checks = []
        for factor in (1, -1):
            try:
                other = BeatGrid.from_timing_points(renotate(points, factor), span)
            except ValueError:
                continue
            checks.append(other.fold == grid.fold - factor and _same_coords(coords, other.locate(events.time_ms)))
        out['renotation_invariant'] = all(checks) if checks else None
    except Exception as exc:  # noqa: BLE001
        errors.append(f'timing: {_reason(exc, path)}')
    if errors:
        out['error'] = ERROR_SEPARATOR.join(errors)
    return out


def inspect_file(task: tuple[str, str]) -> dict:
    """Hashes, header fields and, for 4K osu!mania, the chart summary of one file."""
    repo_root, rel = task
    path = Path(repo_root) / rel
    data = path.read_bytes()
    row = dict(path=rel, set_dir=str(Path(rel).parent), size_bytes=len(data),
               sha256=hashlib.sha256(data).hexdigest(), md5=hashlib.md5(data).hexdigest())
    try:
        meta = parse_osu_metadata(path)
    except Exception as exc:  # noqa: BLE001
        row['error'] = f'header: {_reason(exc, path)}'
        return row
    row.update(mode=meta.mode, circle_size=meta.circle_size, keys=meta.key_count, version=meta.version,
               artist=meta.artist, title=meta.title, creator=meta.creator, beatmap_id=meta.beatmap_id,
               beatmap_set_id=meta.beatmap_set_id, audio_filename=meta.audio_filename)
    if meta.mode == 3 and meta.key_count == 4:
        row.update(_mania_summary(path))
    return row


def hash_audio(task: tuple[str, str]) -> tuple[str, str | None]:
    repo_root, rel = task
    path = Path(repo_root) / rel
    return rel, (_sha256(path) if path.is_file() else None)


def load_acquisitions(dataset_root: Path) -> dict[int, str]:
    """Beatmap set id to the acquisition record that added it."""
    origin = {}
    for record in sorted((dataset_root / 'metadata' / 'acquisitions').glob('*.json')):
        value = json.loads(record.read_text())
        for entry in value.get('sets', []):
            origin[int(entry['beatmapSetId'])] = value['acquisitionId']
    return origin


def load_r1_catalog(path: Path | None) -> dict[str, tuple[str, str]]:
    """File SHA-256 to R1's (split, group id), from the catalog R1 was trained with."""
    if path is None:
        return {}
    return {e['source_sha256']: (e['split'], e['group_id']) for e in json.loads(path.read_text())}


def _api_columns(row: dict, set_meta: dict | None) -> dict:
    """osu! API fields of the beatmap this file is, matched by checksum, else by id."""
    if not set_meta:
        return {}
    beatmaps = set_meta.get('beatmaps') or []
    match = next((b for b in beatmaps if b.get('checksum') == row['md5']), None)
    how = 'checksum'
    if match is None and row.get('beatmap_id'):
        match, how = next((b for b in beatmaps if b.get('id') == row['beatmap_id']), None), 'beatmap_id'
    if match is None:
        return dict(api_match='none')
    tags = match.get('top_tag_ids') or []
    return dict(api_match=how, api_beatmap_id=match.get('id'), api_status=match.get('status'),
                api_star=match.get('difficulty_rating'), api_bpm=match.get('bpm'), api_mode=match.get('mode'),
                api_cs=match.get('cs'), api_total_length_s=match.get('total_length'),
                api_hit_length_s=match.get('hit_length'), api_ranked_date=set_meta.get('ranked_date'),
                api_tag_ids=[int(t['tag_id']) for t in tags], api_tag_counts=[int(t['count']) for t in tags])


def group_rows(rows: list[dict]) -> list[str]:
    """Song group of each row: union of set, beatmap, song name, audio and R1 group.

    A group's id is the smallest file SHA-256 among its members, so it is stable
    while that file stays in the dataset.
    """
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, row in enumerate(rows):
        tokens = [('dir', row['set_dir'])]
        for key in ('beatmap_set_id',):
            if row.get(key) and row[key] > 0:
                tokens.append(('set', row[key]))
        if Path(row['set_dir']).name.isdigit():
            tokens.append(('set', int(Path(row['set_dir']).name)))
        for key in ('beatmap_id', 'api_beatmap_id'):
            if row.get(key) and row[key] > 0:
                tokens.append(('beatmap', row[key]))
        artist, title = normalized(row.get('artist')), song_title(row.get('title'))
        if artist and title:
            tokens.append(('song', artist, title))
        for key in ('audio_sha256', 'r1_group_id'):
            if row.get(key):
                tokens.append((key, row[key]))
        for token in tokens:
            if token in owner:
                a, b = root(i), root(owner[token])
                parent[max(a, b)] = min(a, b)
            else:
                owner[token] = i
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[root(i)].append(row['sha256'])
    ids = {r: min(shas) for r, shas in members.items()}
    return [ids[root(i)] for i in range(len(rows))]


def split_of(group_id: str, r1_trained: bool) -> str:
    """Evaluation split of a song group, by hash; R1's TRAIN songs are never held out."""
    bucket = int(hashlib.sha256((SPLIT_SALT + group_id).encode()).hexdigest()[:16], 16) % 100
    if bucket < HELDOUT_PERCENT:
        return 'fit' if r1_trained else 'heldout'
    return 'calibration' if bucket < HELDOUT_PERCENT + CALIBRATION_PERCENT else 'fit'


SCHEMA = pa.schema([
    ('path', pa.string()), ('set_dir', pa.string()), ('origin', pa.string()), ('size_bytes', pa.int64()),
    ('sha256', pa.string()), ('md5', pa.string()),
    ('mode', pa.int8()), ('circle_size', pa.float32()), ('keys', pa.int8()), ('version', pa.string()),
    ('artist', pa.string()), ('title', pa.string()), ('creator', pa.string()), ('beatmap_id', pa.int64()),
    ('beatmap_set_id', pa.int64()), ('audio_filename', pa.string()), ('audio_present', pa.bool_()),
    ('audio_sha256', pa.string()),
    ('api_match', pa.string()), ('api_beatmap_id', pa.int64()), ('api_status', pa.string()),
    ('api_star', pa.float64()), ('api_bpm', pa.float64()), ('api_mode', pa.string()), ('api_cs', pa.float32()),
    ('api_total_length_s', pa.int32()), ('api_hit_length_s', pa.int32()), ('api_ranked_date', pa.string()),
    ('api_tag_ids', pa.list_(pa.int32())), ('api_tag_counts', pa.list_(pa.int32())),
    ('n_taps', pa.int32()), ('n_lns', pa.int32()), ('first_ms', pa.float64()), ('last_ms', pa.float64()),
    ('star', pa.float64()), ('n_red_lines', pa.int32()), ('distinct_bpms', pa.int32()),
    ('min_bpm', pa.float64()), ('max_bpm', pa.float64()), ('dominant_bpm', pa.float64()), ('fold', pa.int8()),
    ('canonical_bpm', pa.float64()), ('head_on_grid', pa.float64()), ('release_on_grid', pa.float64()),
    ('roundtrip_max_ms', pa.float64()), ('renotation_invariant', pa.bool_()), ('error', pa.string()),
    ('group_id', pa.string()), ('r1_split', pa.string()), ('r1_group_id', pa.string()), ('eval_split', pa.string()),
])


def build_inventory(repo_root: Path, dataset: str = 'dataset', *, r1_catalog: Path | None = None,
                    workers: int = 1) -> tuple[pa.Table, dict]:
    repo_root = repo_root.resolve()
    dataset_root = repo_root / dataset
    files = sorted(str(p.relative_to(repo_root)) for p in dataset_root.rglob('*.osu') if p.is_file())
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(inspect_file, [(str(repo_root), f) for f in files], chunksize=16))
        audio = sorted({str(Path(r['set_dir']) / r['audio_filename']) for r in rows if r.get('audio_filename')})
        audio_hash = dict(pool.map(hash_audio, [(str(repo_root), a) for a in audio], chunksize=8))
    acquisitions, r1 = load_acquisitions(dataset_root), load_r1_catalog(r1_catalog)
    set_meta = {}
    for row in rows:
        directory = repo_root / row['set_dir']
        if row['set_dir'] not in set_meta:
            meta_path = directory / 'metadata.json'
            set_meta[row['set_dir']] = json.loads(meta_path.read_text()) if meta_path.is_file() else None
        row.update(_api_columns(row, set_meta[row['set_dir']]))
        if row.get('audio_filename'):
            key = str(Path(row['set_dir']) / row['audio_filename'])
            row['audio_sha256'], row['audio_present'] = audio_hash.get(key), audio_hash.get(key) is not None
        name = Path(row['set_dir']).name
        supplemental = Path(dataset, 'supplemental')
        if Path(row['set_dir']).is_relative_to(supplemental):
            row['origin'] = 'supplemental/' + Path(row['set_dir']).relative_to(supplemental).parts[0]
        else:
            row['origin'] = acquisitions.get(int(name), 'base') if name.isdigit() else 'base'
        if row['sha256'] in r1:
            row['r1_split'], row['r1_group_id'] = r1[row['sha256']]
    groups = group_rows(rows)
    trained = {g for g, row in zip(groups, rows) if row.get('r1_split') == 'train'}
    for group, row in zip(groups, rows):
        row['group_id'], row['eval_split'] = group, split_of(group, group in trained)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    return table, summarize(rows, r1_catalog)


def summarize(rows: list[dict], r1_catalog: Path | None) -> dict:
    mania = [r for r in rows if r.get('mode') == 3 and r.get('keys') == 4]
    target = [r for r in mania if r.get('api_status') in ('ranked', 'loved')]
    timed = [r for r in target if r.get('head_on_grid') is not None]
    matched = [r for r in mania if r.get('api_match') == 'checksum' and r.get('star') is not None
               and r.get('api_star') is not None]
    star_gap = [abs(r['star'] - r['api_star']) for r in matched]
    return dict(
        format=FORMAT, files=len(rows), mania_4k=len(mania),
        origin=dict(Counter(r['origin'] for r in rows)),
        mode_keys=dict(Counter(f"mode{r.get('mode')}-{r.get('keys')}k" for r in rows)),
        mania_4k_status=dict(Counter(str(r.get('api_status')) for r in mania)),
        mania_4k_api_match=dict(Counter(str(r.get('api_match')) for r in mania)),
        ranked_loved_star_bands=dict(Counter(
            f'{min(int(r["api_star"]), 8)}' for r in target if r.get('api_star') is not None)),
        errors=dict(Counter(e.split(':', 1)[0] for r in rows for e in (r.get('error') or '').split(ERROR_SEPARATOR) if e)),
        error_reasons=dict(Counter(e[:100] for r in rows for e in (r.get('error') or '').split(ERROR_SEPARATOR)
                                   if e).most_common(12)),
        groups=len({r['group_id'] for r in rows}),
        largest_group_files=max(Counter(r['group_id'] for r in rows).values()) if rows else 0,
        split_ranked_loved_4k=dict(Counter(r['eval_split'] for r in target)),
        split_groups_ranked_loved_4k={s: len({r['group_id'] for r in target if r['eval_split'] == s})
                                      for s in ('fit', 'calibration', 'heldout')},
        r1_catalog=None if r1_catalog is None else dict(
            sha256=_sha256(r1_catalog), matched=dict(Counter(r['r1_split'] for r in rows if r.get('r1_split')))),
        r1_trained_in_heldout=sum(r.get('r1_split') == 'train' and r['eval_split'] == 'heldout' for r in rows),
        star_vs_api=dict(charts=len(star_gap), max_abs=max(star_gap) if star_gap else None,
                         over_1e_4=sum(g > 1e-4 for g in star_gap),
                         ranked_loved_max_abs=max((abs(r['star'] - r['api_star']) for r in matched
                                                   if r.get('api_status') in ('ranked', 'loved')), default=None)),
        grid_ranked_loved_head_on_grid_lt_0_9_multi_bpm=sum(r['head_on_grid'] < .9 and r['distinct_bpms'] > 1
                                                             for r in timed),
        grid=dict(
            ranked_loved_with_grid=len(timed),
            head_on_grid_ge_0_9=sum(r['head_on_grid'] >= .9 for r in timed),
            head_on_grid_median=float(np.median([r['head_on_grid'] for r in timed])) if timed else None,
            fold=dict(Counter(int(r['fold']) for r in timed)),
            renotation_invariant=dict(Counter(str(r.get('renotation_invariant')) for r in mania)),
            roundtrip_max_ms=max((r['roundtrip_max_ms'] for r in mania if r.get('roundtrip_max_ms') is not None),
                                 default=None)),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--dataset', default='dataset', help='dataset directory relative to the repo root')
    parser.add_argument('--r1-catalog', type=Path, help="R1's training catalog.json (marks R1 TRAIN songs)")
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Parquet path relative to the repo root')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    table, summary = build_inventory(args.repo_root, args.dataset, r1_catalog=args.r1_catalog, workers=args.workers)
    output = args.repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression='zstd')
    summary['inventory_sha256'] = _sha256(output)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
