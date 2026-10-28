"""Pure mark-reader functions copied from the authorized stage-0 reader.

Session paths, identities and calibration values are supplied by the caller.
"""
from collections import Counter
import numpy as np
from ensomi_n.eval.measures import nearest


def nearest_grid(grid, times):
    distance, index = nearest(np.asarray(times), np.asarray(grid))
    return -distance, index


def period(grid, times):
    grid = np.asarray(grid)
    times = np.asarray(times)
    j = np.clip(np.searchsorted(grid, times, side='right') - 1, 0, len(grid) - 2)
    return np.diff(grid)[j]


def pulse_labels(proposals, times):
    """The old scoring gate, reproduced before applying any listener offset."""
    keep = np.zeros(len(times), dtype=bool)
    for q in proposals:
        grid = np.asarray(q['beats_ms'])
        d, idx = nearest_grid(grid, times)
        gaps = np.diff(grid)
        left = np.where(idx > 0, gaps[np.clip(idx - 1, 0, len(gaps) - 1)], np.nan)
        right = np.where(idx < len(grid) - 1, gaps[np.clip(idx, 0, len(gaps) - 1)], np.nan)
        gate = np.minimum(50., .2 * np.nanmean(np.vstack([left, right]), axis=0))
        keep |= np.abs(d) <= gate
    return keep


def read_marks(grid, marks, offset):
    grid = np.asarray(grid, float)
    beats = sorted([m for m in marks if m['kind'] == 'beat'], key=lambda m: m['time_ms'])
    unbeats = sorted([m for m in marks if m['kind'] == 'unbeat'], key=lambda m: m['time_ms'])
    times = np.array([m['time_ms'] - offset for m in beats])
    candidates = [(float(t), 'line', 1) for t in grid]
    for a, b in zip(grid[:-1], grid[1:]):
        for div in (2, 3, 4):
            for k in range(1, div):
                candidates.append((float(a + (b - a) * k / div), 'cell', div))
    candidates.sort(key=lambda x: (x[0], x[2]))
    ct = np.array([c[0] for c in candidates])
    rows = []
    for m, t in zip(beats, times):
        j = int(np.argmin(np.abs(ct - t)))
        c, kind, div = candidates[j]
        raw = m.get('method') == 'key-playing' and not m.get('nudged_ms', 0)
        window = 35. if raw else 20.
        residual = t - c
        if abs(residual) > window:
            kind, div = 'between', None
        rows.append(dict(id=m['id'], time_ms=m['time_ms'], corrected_ms=float(t),
            method=m.get('method'), window_ms=window, nearest_ms=c, residual_ms=float(residual),
            initial_class=kind, **{'class': kind}, subdivision=div,
            beat_ms=float(period(grid, [t])[0]), unbeat=None))
    pairs = []
    for u in unbeats:
        t = u['time_ms'] - offset
        j = int(np.argmin(np.abs(times - t))) if len(times) else None
        paired = j is not None and abs(times[j] - t) <= .5 * period(grid, [t])[0]
        pairs.append(dict(unbeat_id=u['id'], time_ms=u['time_ms'], corrected_ms=t,
            beat_id=beats[j]['id'] if paired else None,
            distance_ms=float(abs(times[j] - t)) if j is not None else None))
        if paired:
            rows[j]['class'], rows[j]['unbeat'] = 'accent', u['id']
    for j, row in enumerate(rows):
        if row['class'] != 'between':
            continue
        adjacent = [k for k in (j - 1, j + 1) if 0 <= k < len(rows)
                    and abs(times[k] - times[j]) <= 2 * row['beat_ms']]
        if not any(rows[k]['initial_class'] == 'between' for k in adjacent):
            row['class'] = 'accent'
    runs = []
    cuts = [0]
    for i in range(1, len(rows)):
        if times[i] - times[i - 1] > 2 * period(grid, [(times[i] + times[i - 1]) / 2])[0]:
            cuts.append(i)
    cuts.append(len(rows))
    for start, end in zip(cuts[:-1], cuts[1:]):
        if end - start < 3:
            continue
        gaps = np.diff(times[start:end])
        local = period(grid, (times[start:end - 1] + times[start + 1:end]) / 2)
        residuals, multiples = {}, {}
        for d in (2, 3, 4):
            mult = np.maximum(1, np.round(gaps / (local / d)))
            residuals[d] = float(np.median(np.abs(gaps - mult * local / d)))
            multiples[d] = mult.astype(int).tolist()
        valid = [d for d in (2, 3, 4) if residuals[d] <= 20]
        selected = valid[0] if valid else None
        runs.append(dict(first=start, last=end - 1, n=end - start, subdivision=selected,
            compatible=valid, median_gap_residual_ms=residuals,
            gap_cells=multiples.get(selected), classes=dict(Counter(r['class'] for r in rows[start:end]))))
    return rows, pairs, runs
