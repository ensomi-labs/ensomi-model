"""Inference-only measurements of R2 and its source charts.

Inputs are an existing fit cache and a released model directory. Outputs belong
in an ignored artifact directory. No command opens sealed corpus partitions or
fits a model. Run with the repository's src directory on PYTHONPATH.

Run census first, then panel, details, snapping and nulls. miss-geometry adds
endpoint distances after details. --root resolves source paths in the cache
index; --out is the common result directory. panel requires --existing-panel
and --exclude-panel; nulls requires an existing --pairs CSV with a/b chart
hashes and same_creator/same_band fields. All output directories must have at
least 100 GB free. The scripts do not download or rebuild a cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from functools import lru_cache

import numpy as np
import pandas as pd

from ensomi_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ensomi_model.r2.cache import load_chart, objects_from_decisions
from ensomi_model.r2.common import GridArrays, grid_from_arrays
from ensomi_model.r2.splits import role_of

HOLD = ManiaHitObjectKind.HOLD
TAP = ManiaHitObjectKind.TAP
SELECTION_SEED = 20261108


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    disk_guard(path.parent)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def disk_guard(path):
    if shutil.disk_usage(path).free < 100_000_000_000:
        raise RuntimeError('Free disk is below 100 GB; stopping before writing')


def grid_of(dec):
    return GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))


def source_objects(dec):
    gap = np.where(np.isfinite(dec.release_orig_ms), dec.release_orig_ms, dec.gap_release_ms)
    return objects_from_decisions(dec.head_ms, dec.actions, gap, dec.song_ms)


def ln_statistics(objects, grid, start_ms=-np.inf):
    """Class definitions use original release times and all same-lane successors.

    With start_ms, score heads born in the suffix. Successors are still found
    in the complete chart; holds crossing the cut do not become new LN heads.
    """
    selected = [o for o in objects if o.start_time_ms >= start_ms]
    holds = [o for o in selected if o.kind is HOLD]
    gaps, short = [], []
    for lane in range(4):
        lane_objects = sorted((o for o in objects if o.lane == lane), key=lambda o: o.start_time_ms)
        for a, b in zip(lane_objects, lane_objects[1:]):
            if a.kind is HOLD and a.start_time_ms >= start_ms:
                gap = b.start_time_ms - a.end_time_ms
                beat_ms = float(grid.lengths[grid.segment(a.end_time_ms)])
                gaps.append(gap)
                short.append(gap <= max(40., beat_ms / 4) + 1e-6)
    share = len(holds) / len(selected) if selected else np.nan
    inverse = float(np.mean(short)) if short else np.nan
    kind = ('inverse' if share >= .8 and len(gaps) >= 32 and inverse >= .5 else
            'ln_heavy' if share >= .5 else 'normal')
    lengths = np.array([o.end_time_ms - o.start_time_ms for o in holds])
    starts_by_lane = [np.sort([o.start_time_ms for o in objects if o.lane != lane]) for lane in range(4)]
    near = sum(np.searchsorted(starts_by_lane[o.lane], o.end_time_ms + 40, side='right') >
               np.searchsorted(starts_by_lane[o.lane], o.end_time_ms + 1, side='left') for o in holds)
    by_birth = {}
    for o in holds:
        by_birth.setdefault(o.start_time_ms, []).append(o.end_time_ms)
    pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_birth.values())
    together = sum(abs(a - b) <= 5 for v in by_birth.values() for i, a in enumerate(v) for b in v[i + 1:])
    result = dict(ln_class=kind, objects=len(selected), ln=len(holds), ln_share=share,
                  inverse_gap_n=len(gaps), inverse_gap_share=inverse,
                  near_head_share=near / len(holds) if holds else np.nan,
                  same_birth_pairs=pairs, co_release_share=together / pairs if pairs else np.nan)
    for q in (10, 50, 90):
        result[f'hold_p{q}_ms'] = float(np.percentile(lengths, q)) if len(lengths) else np.nan
    result['short_hold_share'] = float(np.mean(lengths <= 60)) if len(lengths) else np.nan
    return result


def header(path):
    values = {}
    with Path(path).open(encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            if line.strip() == '[HitObjects]':
                break
            if ':' in line:
                k, v = line.split(':', 1)
                if k in ('AudioFilename', 'Creator'):
                    values[k] = v.strip()
    return values


def census_one(spec):
    row, cache, root = spec
    assert row['role'] in ('fit_train', 'fit_dev') and role_of(row['group_id']) == row['role']
    dec = load_chart(Path(cache) / 'charts' / row['file'])
    stats = ln_statistics(source_objects(dec), grid_of(dec))
    error = np.abs(dec.gap_release_ms - dec.release_orig_ms)
    error = error[np.isfinite(error)]
    # Row releases have exactly zero snapping error and belong in the LN denominator.
    all_errors = np.r_[np.zeros(row['n_row_release']), error]
    assert len(all_errors) == stats['ln']
    meta = header(Path(root) / row['path'])
    out = dict(row, **stats, creator=meta.get('Creator', ''),
               audio=str(Path(root) / Path(row['path']).parent / meta['AudioFilename']),
               exact_releases=int(np.sum(all_errors <= 1e-6)), within1=int(np.sum(all_errors <= 1)),
               within5=int(np.sum(all_errors <= 5)))
    for q in (50, 95, 99, 100):
        out[f'snap_p{q}_ms'] = float(np.percentile(all_errors, q)) if len(all_errors) else np.nan
    return out


def census(args):
    index = pd.read_parquet(Path(args.cache) / 'index.parquet')
    assert index.role.isin(['fit_train', 'fit_dev']).all()
    jobs = [(r, args.cache, args.root) for r in index.to_dict('records')]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        rows = []
        for i, row in enumerate(pool.map(census_one, jobs, chunksize=32)):
            rows.append(row)
            if (i + 1) % 2000 == 0:
                print(json.dumps(dict(census=i + 1, total=len(jobs))), flush=True)
    disk_guard(args.out)
    table = pd.DataFrame(rows)
    table.to_csv(Path(args.out) / 'census.csv', index=False)
    counts = table.groupby(['role', 'band', 'ln_class']).agg(
        charts=('sha256', 'size'), songs=('group_id', 'nunique'), ln=('ln', 'sum'),
        on_row=('n_row_release', 'sum'), in_gap=('n_gap_release', 'sum'),
        exact=('exact_releases', 'sum'), within5=('within5', 'sum')).reset_index()
    counts.to_csv(Path(args.out) / 'class-coverage.csv', index=False)
    print(counts.to_string(index=False), flush=True)


def select_panel(args):
    table = pd.read_csv(Path(args.out) / 'census.csv')
    existing = json.loads(Path(args.existing_panel).read_text())
    preferred = {s['key'] for s in existing['songs']}
    exclusions = json.loads(Path(args.exclude_panel).read_text())
    table = table[(table.role == 'fit_dev') & ~table.group_id.isin(exclusions['excluded_group_id'])].copy()
    # A duplicated audio hash occurring in train is not an unseen song.
    full = pd.read_csv(Path(args.out) / 'census.csv')
    train_audio = set(full.loc[full.role == 'fit_train', 'audio_sha256'])
    table = table[~table.audio_sha256.isin(train_audio)]
    table['order'] = [hashlib.sha256(f'{SELECTION_SEED}:{s}'.encode()).hexdigest() for s in table.sha256]
    table['preferred'] = table.audio_sha256.isin(preferred)
    table = table.sort_values(['preferred', 'order'], ascending=[False, True])
    selected, groups, audio = [], set(), set()

    def pick(candidates, n):
        for row in candidates.to_dict('records'):
            if n <= 0:
                break
            if row['group_id'] in groups or row['audio_sha256'] in audio or not Path(row['audio']).is_file():
                continue
            selected.append(row)
            groups.add(row['group_id']); audio.add(row['audio_sha256'])
            n -= 1

    # Rare classes first so normal charts cannot consume their songs.
    for kind in ('inverse', 'ln_heavy', 'normal'):
        for band in range(2, 6):
            pick(table[(table.ln_class == kind) & (table.band == band)], 4)
    for band in range(2, 6):
        pick(table[table.band == band], 12 - sum(r['band'] == band for r in selected))
    write_json(Path(args.out) / 'panel.json', dict(seed=SELECTION_SEED, charts=selected,
               counts=pd.DataFrame(selected).groupby(['band', 'ln_class']).size().to_dict(),
               excluded_train_audio=len(train_audio), preferred_audio=len(preferred)))
    print(pd.DataFrame(selected).groupby(['band', 'ln_class']).size(), flush=True)


def mask_statistics(head, actions, config, start=0):
    from ensomi_model.r2.theta import decode
    _, heads, _ = decode(actions[:len(head)])
    last = np.full(4, -np.inf)
    runs, quad = np.zeros(4, int), 0
    counts = dict(reattack=0, quad=0, jack=0)
    maximum = 0
    for k, (t, h) in enumerate(zip(head, heads)):
        tercile = np.searchsorted(config['gap_tercile_edges_ms'], t - head[k - 1] if k else np.inf,
                                 side='right')
        if k >= start:
            counts['reattack'] += int((h & (t - last < config['reattack_floor_ms'])).any())
            counts['quad'] += int(h.all() and quad >= config['quad_run_cap'])
            counts['jack'] += int((h & (runs >= config['jack_run_caps'][tercile])).any())
        runs = np.where(h, runs + 1, 0)
        quad = quad + 1 if h.all() else 0
        last[h] = t
        if k >= start:
            maximum = max(maximum, int(runs.max()))
    return {**{'mask_' + k: v for k, v in counts.items()}, 'max_jack_run': maximum}


def code_counts(actions, start=0, stop=None):
    from ensomi_model.r2.theta import decode
    held, _, _ = decode(actions)
    stop = len(actions) if stop is None else stop
    a, h = actions[start:stop], held[start:stop]
    return {f'{state}_{code}': int(np.sum((h if state == 'held' else ~h) & (a == code)))
            for state, codes in (('held', range(5)), ('free', range(3))) for code in codes}


def detail_one(spec):
    from ensomi_model.r2.candidates import build_candidates, SNAP_ONE_HOT
    from ensomi_model.r2.theta import decode
    row, cache = spec
    assert row['role'] in ('fit_train', 'fit_dev') and role_of(row['group_id']) == row['role']
    dec = load_chart(Path(cache) / 'charts' / row['file'])
    grid = grid_of(dec)
    held, heads, births = decode(dec.actions)
    times = np.r_[dec.head_ms, dec.song_ms]
    last_birth = np.maximum.accumulate(np.where(births, np.arange(len(times))[:, None], -1), axis=0)
    before_birth = np.vstack([np.full((1, 4), -1), last_birth[:-1]])
    counts = code_counts(dec.actions, stop=dec.K)
    counts.update({'eos_' + k: v for k, v in code_counts(dec.actions, start=dec.K).items()})
    counts.update(candidate_releases=0, candidate_total=0, four_anchor_within5=row['n_row_release'])
    for name in ('midpoint', 'eighth', 'quarter', 'grid'):
        counts['candidate_' + name] = 0
    for d in (*SNAP_ONE_HOT, 0):
        counts[f'snap_d{d}'] = 0
    misses = []
    for k in np.flatnonzero(np.isfinite(dec.release_orig_ms).any(1)):
        c = build_candidates(times[k - 1], times[k], grid)
        for lane in np.flatnonzero(np.isfinite(dec.release_orig_ms[k])):
            original = dec.release_orig_ms[k, lane]
            snapped = dec.gap_release_ms[k, lane]
            ix = int(np.searchsorted(c.times, snapped))
            assert ix < len(c.times) and c.times[ix] == snapped
            counts['candidate_releases'] += 1
            counts['candidate_total'] += len(c.times)
            for name, flag in zip(('midpoint', 'eighth', 'quarter', 'grid'), c.flags[ix]):
                counts['candidate_' + name] += int(flag)
            snap_flags = c.static[ix, 13:23]
            denom = SNAP_ONE_HOT[int(snap_flags.argmax())] if snap_flags.any() else 0
            counts[f'snap_d{denom}'] += 1
            anchors = c.times[c.flags[:, :3].any(1)]
            counts['four_anchor_within5'] += int(len(anchors) > 0 and np.min(abs(anchors - original)) <= 5)
            err = abs(original - snapped)
            if err > 5:
                local, seg = grid.local_beat(original)
                misses.append(dict(sha=row['sha256'], role=row['role'], band=row['band'],
                    ln_class=row['ln_class'], k=int(k), lane=int(lane), original_ms=original,
                    snapped_ms=snapped, error_ms=err, gap_ms=times[k] - times[k - 1],
                    hold_ms=original - times[before_birth[k, lane]], code=int(dec.actions[k, lane]),
                    original_phase=float(local % 1), chosen_denominator=denom,
                    canonical_beat_ms=float(grid.lengths[seg]), candidates=len(c.times),
                    heads_at_row=int(heads[k].sum()), held_at_row=int(held[k].sum())))
    dense = np.r_[False, np.diff(dec.head_ms) <= 125]
    for label, mask in (('dense', dense), ('sparse', ~dense)):
        counts[label + '_free'] = int((~held[:dec.K][mask]).sum())
        counts[label + '_held'] = int(held[:dec.K][mask].sum())
        counts[label + '_birth_free'] = int((births[:dec.K][mask] & ~held[:dec.K][mask]).sum())
        counts[label + '_birth_held'] = int((births[:dec.K][mask] & held[:dec.K][mask]).sum())
        counts[label + '_heads'] = int(heads[:dec.K][mask].sum())
    return dict(sha=row['sha256'], role=row['role'], band=row['band'], ln_class=row['ln_class'], **counts), misses


def details(args):
    out = Path(args.out)
    rows = pd.read_csv(out / 'census.csv').to_dict('records')
    if args.limit:
        rows = rows[:args.limit]
    counts, misses = [], []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        for i, (count, miss) in enumerate(pool.map(detail_one, [(r, args.cache) for r in rows], chunksize=8)):
            counts.append(count); misses.extend(miss)
            if (i + 1) % 500 == 0:
                print(json.dumps(dict(details=i + 1, total=len(rows))), flush=True)
    disk_guard(out)
    pd.DataFrame(counts).to_csv(out / 'code-candidate-counts.csv', index=False)
    pd.DataFrame(misses).to_csv(out / 'representation-misses.csv', index=False)
    print(json.dumps(dict(charts=len(counts), misses=len(misses))), flush=True)
    inverse = sorted((r for r in rows if r['role'] == 'fit_dev' and r['ln_class'] == 'inverse'),
                     key=lambda r: r['sha256'])[:2]
    examples = []
    from ensomi_model.r2.theta import decode
    for row in inverse:
        dec = load_chart(Path(args.cache) / 'charts' / row['file'])
        held, _, _ = decode(dec.actions)
        candidates = np.flatnonzero((dec.actions == 4).any(1))
        center = int(candidates[0]) if len(candidates) else dec.K // 3
        start, stop = max(0, center - 2), min(dec.K + 1, center + 6)
        times = np.r_[dec.head_ms, dec.song_ms]
        examples.append(dict(sha=row['sha256'], band=row['band'], source_class=row['ln_class'],
            start_row=start, time_origin_ms=times[start], rows=[dict(k=k, t=times[k] - times[start],
            held=held[k].astype(int), codes=dec.actions[k],
            gap_release=dec.gap_release_ms[k] - times[start], eos=k == dec.K) for k in range(start, stop)]))
    write_json(out / 'inverse-examples.json', examples)


@lru_cache(maxsize=32)
def null_chart(cache, sha):
    dec = load_chart(Path(cache) / 'charts' / (sha + '.npz'))
    return dec, source_objects(dec), grid_of(dec)


def null_one(spec):
    pair, cache, classes = spec
    a, ao, ag = null_chart(cache, pair['a'])
    b, bo, bg = null_chart(cache, pair['b'])
    records = []
    for mode, cut in [('bos', max(a.head_ms[0], b.head_ms[0])),
                      ('prefix', max(a.head_ms[a.K // 3], b.head_ms[b.K // 3]))]:
        am, bm = ln_statistics(ao, ag, cut), ln_statistics(bo, bg, cut)
        row = dict(pair, mode=mode, cut_ms=cut, a_class=classes[pair['a']], b_class=classes[pair['b']])
        for key in am:
            if key != 'ln_class':
                row['a_' + key], row['b_' + key], row['diff_' + key] = am[key], bm[key], bm[key] - am[key]
        records.append(row)
    return records


def nulls(args):
    out = Path(args.out)
    census_table = pd.read_csv(out / 'census.csv')
    allowed = set(census_table.sha256)
    classes = dict(zip(census_table.sha256, census_table.ln_class))
    pairs = pd.read_csv(args.pairs)
    assert set(pairs.a).union(pairs.b) <= allowed, 'Human null includes an uncached chart'
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(null_one, [(r, args.cache, classes) for r in pairs.to_dict('records')], chunksize=16):
            rows.extend(result)
    disk_guard(out)
    pd.DataFrame(rows).to_csv(out / 'human-null.csv', index=False)
    print(json.dumps(dict(pairs=len(pairs), rows=len(rows), songs=pairs.group_id.nunique())), flush=True)


def snap_one(spec):
    row, cache = spec
    assert row['role'] in ('fit_train', 'fit_dev') and role_of(row['group_id']) == row['role']
    dec = load_chart(Path(cache) / 'charts' / row['file'])
    error = np.abs(dec.gap_release_ms - dec.release_orig_ms)
    error = error[np.isfinite(error)]
    # Human times and cand-v1 targets are whole milliseconds in this cache.
    assert np.allclose(error, np.round(error))
    hist = np.bincount(np.round(error).astype(int))
    if not len(hist):
        hist = np.zeros(1, int)
    hist[0] += row['n_row_release']
    return [dict(role=row['role'], band=row['band'], ln_class=row['ln_class'], error_ms=i, n=int(n))
            for i, n in enumerate(hist) if n]


def snapping(args):
    out = Path(args.out)
    rows = pd.read_csv(out / 'census.csv').to_dict('records')
    hist = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(snap_one, [(r, args.cache) for r in rows], chunksize=64):
            hist.extend(result)
    disk_guard(out)
    pd.DataFrame(hist).groupby(['role', 'band', 'ln_class', 'error_ms']).n.sum().reset_index().to_csv(
        out / 'snapping-histogram.csv', index=False)


def miss_geometry(args):
    out = Path(args.out)
    misses = pd.read_csv(out / 'representation-misses.csv')
    result = []
    for sha, rows in misses.groupby('sha'):
        dec = load_chart(Path(args.cache) / 'charts' / (sha + '.npz'))
        times = np.r_[dec.head_ms, dec.song_ms]
        rows = rows.copy()
        rows['before_row_ms'] = times[rows.k] - rows.original_ms
        rows['after_previous_ms'] = rows.original_ms - times[rows.k - 1]
        rows['signed_snap_ms'] = rows.snapped_ms - rows.original_ms
        result.append(rows)
    disk_guard(out)
    pd.concat(result).to_csv(out / 'representation-miss-geometry.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['census', 'panel', 'details', 'nulls', 'snapping', 'miss-geometry'])
    parser.add_argument('--cache', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--existing-panel')
    parser.add_argument('--exclude-panel')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--pairs')
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {'census': census, 'panel': select_panel, 'details': details, 'nulls': nulls,
     'snapping': snapping, 'miss-geometry': miss_geometry}[args.command](args)


if __name__ == '__main__':
    main()
