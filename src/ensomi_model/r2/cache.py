"""R2 chart cache: heads-only inputs and teacher-forcing decisions, one NumPy file per chart.

Per chart: parse objects, take sorted distinct head times, merge rows closer
than 2 ms into the earlier time, build the grid with
``musical_grid(red_lines, head_times)`` (never the release-aware chart grid),
read the song length T from the audio container with ``soundfile.info``, and
require T >= t_last + 2 ms and T > the last source release. Source LN releases
become labels: a release on a head row is code 1; one inside a gap is snapped
to the nearest candidate of that gap (ties earlier) with the error recorded.

Run on the mac from the repository root::

    .venv/bin/python -m ensomi_model.r2.cache --out artifacts/r2-cache/v1 --workers 6
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

import numpy as np

from ..evaluation.legality import violations
from ..evaluation.redlines import musical_grid, read_red_lines
from ..osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from .candidates import build_candidates, snap_label
from .common import (CACHE_VERSION, CANDIDATE_VERSION, MIN_GAP_MS, ContractError, GridArrays, band_of,
                     grid_from_arrays, grid_to_arrays)
from .splits import assert_fit, population, role_of, write_assignment

TAP_KIND, HOLD_KIND = ManiaHitObjectKind.TAP, ManiaHitObjectKind.HOLD


class Excluded(Exception):
    def __init__(self, reason, detail=''):
        super().__init__(f'{reason}: {detail}')
        self.reason, self.detail = reason, detail


@dataclass
class ChartDecisions:
    head_ms: np.ndarray          # [K] float64
    actions: np.ndarray          # [K+1,4] int8, row K is EOS
    gap_release_ms: np.ndarray   # [K+1,4] float64 snapped gap releases, NaN when absent
    release_orig_ms: np.ndarray  # [K+1,4] float64 original gap release time, NaN when absent
    grid_segments: np.ndarray    # [S,3] offset, notated beat length, meter
    grid_bars: np.ndarray        # [B,2]
    song_ms: float
    stats: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.head_ms)


def read_duration_ms(audio_path: Path) -> float:
    import soundfile
    info = soundfile.info(str(audio_path))
    value = float(info.duration) * 1000.0
    if not math.isfinite(value) or value <= 0:
        raise Excluded('audio_bad_duration', str(value))
    return value


def decisions_from_objects(objects, red_lines, song_ms: float) -> ChartDecisions:
    """Pure conversion used by the cache and by tests; raises Excluded with a reason."""
    if not objects:
        raise Excluded('empty')
    if any(o.start_time_ms < 0 for o in objects):
        raise Excluded('negative_time')
    bad = violations(objects, song_span=(0.0, song_ms))
    if bad:
        reason = 'after_song' if set(bad) == {'after_song'} else 'illegal_source'
        raise Excluded(reason, json.dumps(bad))
    distinct = sorted({float(o.start_time_ms) for o in objects})
    rows, row_of = [], {}
    merged = 0
    for t in distinct:
        if rows and t - rows[-1] < MIN_GAP_MS:
            merged += 1
        else:
            rows.append(t)
        row_of[t] = len(rows) - 1
    head_ms = np.array(rows, dtype=np.float64)
    K = len(rows)
    moved = [ManiaHitObject(rows[row_of[o.start_time_ms]], o.end_time_ms if o.kind is HOLD_KIND else rows[row_of[o.start_time_ms]],
                            o.lane, o.kind) for o in objects]
    if merged:
        bad = violations(moved, song_span=(0.0, song_ms))
        if bad:
            raise Excluded('merge_collision', json.dumps(bad))
    last_release = max((o.end_time_ms for o in objects if o.kind is HOLD_KIND), default=-math.inf)
    if not song_ms >= head_ms[-1] + MIN_GAP_MS:
        raise Excluded('song_shorter_than_heads', f'T={song_ms} last_head={head_ms[-1]}')
    if not song_ms > last_release:
        raise Excluded('song_not_after_release', f'T={song_ms} last_release={last_release}')
    try:
        grid, _ = musical_grid(red_lines, head_ms)
    except ValueError as exc:
        raise Excluded('grid', str(exc)) from exc
    garr = GridArrays.from_grid(grid)
    actions = np.zeros((K + 1, 4), dtype=np.int8)
    gap = np.full((K + 1, 4), np.nan)
    orig = np.full((K + 1, 4), np.nan)
    row_release = np.zeros((K + 1, 4), dtype=bool)
    errors = []
    n_row_release = 0
    for o in moved:
        if o.kind is not HOLD_KIND:
            continue
        e = float(o.end_time_ms)
        q = int(np.searchsorted(head_ms, e, side='left'))
        if q < K and head_ms[q] == e:
            row_release[q, o.lane] = True
            n_row_release += 1
            continue
        a = head_ms[q - 1]
        b = head_ms[q] if q < K else song_ms
        cands = build_candidates(a, b, garr)
        idx, err = snap_label(e, cands)
        if not np.isnan(gap[q, o.lane]):
            raise Excluded('double_release', f'lane {o.lane} decision {q}')
        gap[q, o.lane] = cands.times[idx]
        orig[q, o.lane] = e
        errors.append(err)
    for o in moved:
        r = row_of[o.start_time_ms]
        if actions[r, o.lane] != 0 or row_release[r, o.lane]:
            raise Excluded('lane_conflict', f'lane {o.lane} row {r}')
        code = 1 if o.kind is TAP_KIND else 2
        actions[r, o.lane] = code + (2 if not np.isnan(gap[r, o.lane]) else 0)
    gap_only = ~np.isnan(gap) & (actions == 0)
    actions[gap_only] = 2
    actions[row_release] = 1
    seg, bars = grid_to_arrays(grid)
    errors = np.asarray(errors, dtype=np.float64)
    stats = dict(K=K, merged_rows=merged, n_objects=len(objects),
                 n_ln=sum(o.kind is HOLD_KIND for o in objects), n_row_release=n_row_release,
                 n_gap_release=int(len(errors)), snap_abs=np.abs(errors).tolist())
    return ChartDecisions(head_ms, actions, gap, orig, seg, bars, float(song_ms), stats)


def objects_from_decisions(head_ms, actions, gap_release_ms, song_ms=None):
    """Expand decisions back into hit objects (lanes 0..3), checking the code semantics."""
    K = len(head_ms)
    held = [None] * 4
    out = []
    for k in range(len(actions)):
        t = float(head_ms[k]) if k < K else None
        for lane in range(4):
            c = int(actions[k, lane])
            if held[lane] is not None:
                if c == 0:
                    if k == K:
                        raise ContractError('EOS must close every held lane')
                    continue
                if c == 1:
                    if k == K:
                        raise ContractError('EOS cannot release on a row')
                    end = t
                else:
                    end = float(gap_release_ms[k, lane])
                    lo = float(head_ms[k - 1])
                    hi = t if k < K else float(song_ms)
                    if not lo < end < hi:
                        raise ContractError('Gap release outside its gap')
                out.append(ManiaHitObject(held[lane], end, lane, HOLD_KIND))
                held[lane] = None
                if c in (3, 4):
                    if k == K:
                        raise ContractError('EOS has no heads')
                    if c == 3:
                        out.append(ManiaHitObject(t, t, lane, TAP_KIND))
                    else:
                        held[lane] = t
            else:
                if c in (3, 4) or (k == K and c != 0):
                    raise ContractError(f'Code {c} on a free lane at decision {k}')
                if c == 1:
                    out.append(ManiaHitObject(t, t, lane, TAP_KIND))
                elif c == 2:
                    held[lane] = t
    if any(h is not None for h in held):
        raise ContractError('Decisions leave an open hold')
    return out


# ---- per-row worker ---------------------------------------------------------------------------

def process_row(row: dict, root: str, out_dir: str):
    assert_fit(row)
    try:
        path = Path(root) / row['path']
        if not row.get('audio_present', True):
            raise Excluded('audio_missing', 'corpus flag')
        audio = Path(root) / row['set_dir'] / row['audio_filename']
        if not audio.is_file():
            raise Excluded('audio_missing', str(audio))
        try:
            song_ms = read_duration_ms(audio)
        except Excluded:
            raise
        except Exception as exc:  # unreadable container
            raise Excluded('audio_unreadable', f'{type(exc).__name__}: {exc}'[:200]) from exc
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise Excluded('sha_mismatch')
        try:
            objects = parse_mania_hit_objects(path)
            red_lines = read_red_lines(path)
        except ValueError as exc:
            raise Excluded('parse', str(exc)[:200]) from exc
        dec = decisions_from_objects(objects, red_lines, song_ms)
        back = objects_from_decisions(dec.head_ms, dec.actions, dec.gap_release_ms, dec.song_ms)
        if violations(back, song_span=(0.0, dec.song_ms)):
            raise Excluded('roundtrip_illegal')
        if sorted((o.start_time_ms, o.lane) for o in back) != sorted(
                (dec.head_ms[int(np.searchsorted(dec.head_ms, o.start_time_ms, side='right')) - 1], o.lane)
                for o in objects):
            raise Excluded('roundtrip_heads')
    except Excluded as exc:
        return dict(sha256=row['sha256'], path=row['path'], reason=exc.reason, detail=exc.detail), None
    name = row['sha256'] + '.npz'
    target = Path(out_dir) / 'charts' / name
    tmp = target.with_suffix('.tmp.npz')
    np.savez(tmp, head_ms=dec.head_ms, actions=dec.actions, gap_release_ms=dec.gap_release_ms,
             release_orig_ms=dec.release_orig_ms, grid_segments=dec.grid_segments, grid_bars=dec.grid_bars,
             song_ms=np.array(dec.song_ms))
    os.replace(tmp, target)
    file_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    st = dec.stats
    snap = np.asarray(st.pop('snap_abs'))
    entry = dict(sha256=row['sha256'], path=row['path'], group_id=row['group_id'], role=role_of(row['group_id']),
                 star=float(row['star']), band=band_of(float(row['star'])), song_ms=dec.song_ms,
                 file=name, file_sha256=file_sha, audio_sha256=row.get('audio_sha256'),
                 snap_n=int(len(snap)), snap_gt1=int((snap > 1).sum()), snap_gt5=int((snap > 5).sum()),
                 snap_max=float(snap.max()) if len(snap) else 0.0, **st)
    return entry, None


def _work(args):
    row, root, out_dir = args
    return process_row(row, root, out_dir)


def build(corpus: Path, out_dir: Path, root: Path, workers: int, limit: int | None = None):
    import pandas as pd
    import pyarrow.parquet as pq
    t0 = time.time()
    df = pq.read_table(corpus).to_pandas()
    pop = population(df)
    for split in pop.eval_split.unique():
        if split != 'fit':
            raise ContractError('Population contains a non-fit row')
    if limit:
        pop = pop.sample(limit, random_state=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'charts').mkdir(exist_ok=True)
    split_sha = write_assignment(out_dir / 'splits.json', {g: role_of(g) for g in sorted(pop.group_id.unique())})
    cols = ['path', 'set_dir', 'audio_filename', 'audio_present', 'audio_sha256', 'sha256', 'group_id',
            'star', 'eval_split']
    rows = pop[cols].to_dict('records')
    entries, excluded = [], []
    with Pool(workers) as pool:
        for i, (entry, _) in enumerate(pool.imap_unordered(_work, [(r, str(root), str(out_dir)) for r in rows],
                                                            chunksize=8)):
            (excluded if 'reason' in entry else entries).append(entry)
            if (i + 1) % 1000 == 0:
                print(f'[{time.time() - t0:7.1f}s] {i + 1}/{len(rows)} kept={len(entries)} excluded={len(excluded)}',
                      flush=True)
    index = pd.DataFrame(entries).sort_values('sha256').reset_index(drop=True)
    index.to_parquet(out_dir / 'index.parquet', index=False)
    index_sha = hashlib.sha256((out_dir / 'index.parquet').read_bytes()).hexdigest()
    by_band = {}
    for band, g in index.groupby('band'):
        n = int(g.snap_n.sum())
        by_band[str(band)] = dict(charts=int(len(g)), groups=int(g.group_id.nunique()), gap_ln=n,
                                  row_ln=int(g.n_row_release.sum()),
                                  moved_gt1ms=int(g.snap_gt1.sum()), moved_gt5ms=int(g.snap_gt5.sum()),
                                  share_moved_gt5ms=(float(g.snap_gt5.sum()) / n) if n else 0.0,
                                  max_error_ms=float(g.snap_max.max()))
    reasons = Counter(e['reason'] for e in excluded)
    summary = dict(version=CACHE_VERSION, candidates=CANDIDATE_VERSION, corpus=str(corpus),
                   corpus_sha256=hashlib.sha256(Path(corpus).read_bytes()).hexdigest(),
                   population=len(rows), kept=len(entries), excluded=len(excluded),
                   excluded_by_reason=dict(reasons.most_common()),
                   groups=int(index.group_id.nunique()),
                   roles={r: dict(charts=int((index.role == r).sum()),
                                  groups=int(index[index.role == r].group_id.nunique())) for r in ('fit_train', 'fit_dev')},
                   merged_rows=int(index.merged_rows.sum()), charts_with_merges=int((index.merged_rows > 0).sum()),
                   head_rows=int(index.K.sum()), snap_by_band=by_band,
                   index_sha256=index_sha, splits_sha256=split_sha, wall_s=time.time() - t0,
                   duration_reader='soundfile.info (libsndfile)')
    (out_dir / 'excluded.json').write_text(json.dumps(excluded, indent=1))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    from .receipts import write_receipt
    write_receipt(out_dir / 'receipt.json', config=dict(corpus=str(corpus), out=str(out_dir), workers=workers,
                                                        limit=limit), inputs=dict(corpus_sha256=summary['corpus_sha256']),
                  outputs=dict(index_sha256=index_sha, splits_sha256=split_sha), seed=None, device='cpu',
                  ens_job=os.environ.get('ENS_JOB_ID'))
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# ---- loading ----------------------------------------------------------------------------------

@dataclass
class CacheIndex:
    root: Path
    table: object  # pandas DataFrame
    summary: dict

    @classmethod
    def open(cls, root: Path):
        import pyarrow.parquet as pq
        root = Path(root)
        return cls(root, pq.read_table(root / 'index.parquet').to_pandas(), json.loads((root / 'summary.json').read_text()))

    def load(self, sha256: str) -> ChartDecisions:
        row = self.table.loc[self.table.sha256 == sha256].iloc[0]
        return load_chart(self.root / 'charts' / row.file)


def load_chart(path: Path) -> ChartDecisions:
    with np.load(path) as z:
        return ChartDecisions(z['head_ms'], z['actions'], z['gap_release_ms'], z['release_orig_ms'],
                              z['grid_segments'], z['grid_bars'], float(z['song_ms']))


def chart_grid(dec: ChartDecisions):
    return grid_from_arrays(dec.grid_segments, dec.grid_bars)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--corpus', default='data/r2-corpus.parquet')
    p.add_argument('--out', default='artifacts/r2-cache/v1')
    p.add_argument('--root', default='.')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args(argv)
    build(Path(a.corpus), Path(a.out), Path(a.root), a.workers, a.limit)


if __name__ == '__main__':
    sys.exit(main())
