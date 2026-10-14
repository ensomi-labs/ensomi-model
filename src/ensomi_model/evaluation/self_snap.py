"""Whether a chart's notes sit on its own red lines, and its head rows keep a sixteenth of a beat apart.

A release check for generated charts. For an ``.osu`` file it counts:

- ``off_snap``: heads and releases more than ``SNAP_TOLERANCE_MS`` from every
  1/d line, d in ``COMMON_DENOMINATORS`` (1/1 to 1/16, 1/3, 1/6 and 1/12), of
  the red line in force;
- ``close_rows``: consecutive head rows nearer than ``MIN_ROW_GAP_BEATS`` of the
  earlier row's beat, by more than ``SNAP_TOLERANCE_MS``.

Beats are the red lines' as written, not canonical. A time up to
``SNAP_TOLERANCE_MS`` before a red line belongs to that line: object times are
whole milliseconds, and a segment's first head can round to just before its
line's fractional offset. Times before the first red line use the first, as
osu! does.

    python -m ensomi_model.evaluation.self_snap <chart.osu>...

prints one JSON line per chart and exits with 1 when any count is not zero.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from ..osu_core.hitobjects import parse_mania_hit_objects
from ..osu_core.timing import require_red_timing_points
from .beats import COMMON_DENOMINATORS, SNAP_TOLERANCE_MS, event_times

MIN_ROW_GAP_BEATS = 1 / 16


def check(path: str | Path) -> dict:
    # Red lines sharing an offset keep the last, as osu! applies them.
    lines = {p.offset_ms: p.beat_length_ms for p in require_red_timing_points(path)}
    offsets, lengths = np.array(list(lines)), np.array(list(lines.values()))
    objects = parse_mania_hit_objects(path, expected_key_count=None)

    def line_of(times):
        return np.clip(np.searchsorted(offsets, times + SNAP_TOLERANCE_MS, side='right') - 1, 0, len(offsets) - 1)

    times = np.sort(event_times(objects))
    line = line_of(times)
    beat = (times - offsets[line]) / lengths[line]
    error = np.min([np.abs(beat - np.round(beat * d) / d) for d in COMMON_DENOMINATORS], axis=0) * lengths[line]
    off = times[error > SNAP_TOLERANCE_MS]

    rows = np.unique([o.start_time_ms for o in objects])
    gaps = np.diff(rows)
    close = np.flatnonzero(gaps < lengths[line_of(rows[:-1])] * MIN_ROW_GAP_BEATS - SNAP_TOLERANCE_MS)
    return dict(chart=str(path), events=len(times), head_rows=len(rows), off_snap=len(off), close_rows=len(close),
                first_off_snap_ms=float(off[0]) if len(off) else None,
                first_close_rows_ms=[float(rows[close[0]]), float(rows[close[0] + 1])] if len(close) else None)


def main(paths: list[str]) -> int:
    failed = False
    for path in paths:
        result = check(path)
        print(json.dumps(result), flush=True)
        failed |= bool(result['off_snap'] or result['close_rows'])
    return int(failed)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
