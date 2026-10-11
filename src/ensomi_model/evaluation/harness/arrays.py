"""Columnar view of a chart's objects, with the lane-occupancy queries and placement units injections need.

Placement units are the harness's, never the evaluator's: four canonical bars
inside one musical segment of the grid the harness knows, from the bar of the
first head to the bar of the last, a trailing incomplete group left out
(``unit_layout``). Injections use them to decide where to edit; the event-field
evaluator does not read bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from ..beats import BeatGrid
from ..case import Chart
from ..legality import violations_arrays

UNIT_BARS = 4


@dataclass
class Objects:
    """Objects as arrays, ordered by head time then lane. ``end == start`` for taps."""
    start: np.ndarray
    end: np.ndarray
    lane: np.ndarray
    hold: np.ndarray
    note: dict | None = None

    @classmethod
    def from_chart(cls, chart: Chart) -> 'Objects':
        objs = sorted(chart.objects, key=lambda o: (o.start_time_ms, o.lane, o.end_time_ms))
        return cls(np.array([o.start_time_ms for o in objs], dtype=np.float64),
                   np.array([o.end_time_ms for o in objs], dtype=np.float64),
                   np.array([o.lane for o in objs], dtype=np.int64),
                   np.array([o.kind is ManiaHitObjectKind.HOLD for o in objs], dtype=bool))

    @classmethod
    def concat(cls, parts: list['Objects']) -> 'Objects':
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in ('start', 'end', 'lane', 'hold')))

    def copy(self) -> 'Objects':
        return Objects(self.start.copy(), self.end.copy(), self.lane.copy(), self.hold.copy())

    def take(self, mask) -> 'Objects':
        return Objects(self.start[mask], self.end[mask], self.lane[mask], self.hold[mask])

    def __len__(self) -> int:
        return len(self.start)

    def violations(self, song_span: tuple[float, float] | None = None, keys: int = 4) -> dict[str, int]:
        return violations_arrays(self.start, self.end, self.lane, self.hold, keys=keys, song_span=song_span)

    def legal(self, song_span: tuple[float, float] | None = None) -> bool:
        return not self.violations(song_span)

    def written(self) -> 'Objects':
        """Times as the ``.osu`` file carries them (integer ms); a hold keeps at least 1 ms."""
        start = np.round(self.start) + 0.0  # half to even, as ``written_ms``
        end = np.round(self.end) + 0.0
        end = np.where(self.hold, np.maximum(end, start + 1.0), start)
        return Objects(start, end, self.lane.copy(), self.hold.copy(), self.note)

    def to_objects(self) -> tuple[ManiaHitObject, ...]:
        out = []
        for s, e, lane, h in zip(self.start.tolist(), self.end.tolist(), self.lane.tolist(), self.hold.tolist()):
            out.append(ManiaHitObject(s, e if h else s, int(lane), ManiaHitObjectKind.HOLD if h else ManiaHitObjectKind.TAP))
        return tuple(sorted(out, key=lambda o: (o.start_time_ms, o.lane)))

    def free(self, lane: int, a: float, b: float, skip: int | None = None) -> bool:
        """No object on ``lane`` occupies any instant of ``[a, b]`` (object ``skip`` ignored)."""
        end = np.where(self.hold, self.end, self.start)
        hit = (self.lane == lane) & (self.start <= b) & (end >= a)
        if skip is not None:
            hit[skip] = False
        return not hit.any()

    def free_at(self, lane: int, times: np.ndarray) -> np.ndarray:
        """For each time, whether no object on ``lane`` occupies it."""
        times = np.asarray(times, dtype=np.float64)
        mask = self.lane == lane
        if not mask.any():
            return np.ones(len(times), dtype=bool)
        s = self.start[mask]
        e = np.where(self.hold[mask], self.end[mask], s)
        order = np.argsort(s, kind='stable')
        s, e = s[order], np.maximum.accumulate(e[order])
        idx = np.searchsorted(s, times, side='right') - 1
        return ~((idx >= 0) & (e[np.clip(idx, 0, None)] >= times))


def unit_layout(grid: BeatGrid, first_bar: int, last_bar: int) -> dict[str, Any]:
    """Placement units over global bars ``first_bar..last_bar``: groups of ``UNIT_BARS`` inside one segment."""
    bar_ids = np.arange(first_bar, last_bar + 1)
    starts = grid.time_of_bar(bar_ids) if len(bar_ids) else np.zeros(0)
    offsets = np.array([s.offset_ms for s in grid.segments])
    segment = np.clip(np.searchsorted(offsets, starts + 1e-6, side='right') - 1, 0, len(offsets) - 1)
    first = []
    i = 0
    while i < len(bar_ids):
        j = i
        while j < len(bar_ids) and segment[j] == segment[i]:
            j += 1
        first += [int(bar_ids[i + k * UNIT_BARS]) for k in range((j - i) // UNIT_BARS)]
        i = j
    first_arr = np.array(first, dtype=np.int64)
    start_ms = grid.time_of_bar(first_arr) if len(first_arr) else np.zeros(0)
    end_ms = grid.time_of_bar(first_arr + UNIT_BARS) if len(first_arr) else np.zeros(0)
    return dict(first_bar=first_arr, start_ms=start_ms, end_ms=end_ms)


def object_units(o: Objects, grid: BeatGrid) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """Placement unit of each object by its head's bar (-1 outside every unit), and the layout."""
    if not len(o):
        return None, None
    bar = grid.locate(o.start)['bar'].astype(np.int64)
    lay = unit_layout(grid, int(bar.min()), int(bar.max()))
    index = np.full(int(bar.max() - bar.min()) + 1, -1, dtype=np.int64)
    for p, b in enumerate(lay['first_bar']):
        index[b - bar.min():b - bar.min() + UNIT_BARS] = p
    return index[bar - bar.min()], lay


def rows(start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct head times and the row index of each object."""
    times = np.unique(start)
    return times, np.searchsorted(times, start)


def canonical_beat_at(grid: BeatGrid, times) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    offsets = np.array([s.offset_ms for s in grid.segments])
    seg = np.clip(np.searchsorted(offsets, times, side='right') - 1, 0, len(offsets) - 1)
    return grid.canonical_beat_lengths()[seg]


def local_beat(grid: BeatGrid, times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment, canonical beats from the segment offset, and canonical beat length of each time."""
    times = np.asarray(times, dtype=np.float64)
    offsets = np.array([s.offset_ms for s in grid.segments])
    seg = np.clip(np.searchsorted(offsets, times, side='right') - 1, 0, len(offsets) - 1)
    length = grid.canonical_beat_lengths()[seg]
    return seg, (times - offsets[seg]) / length, length


def bar_fraction(grid: BeatGrid, times) -> tuple[np.ndarray, np.ndarray]:
    """Global bar of each time (as ``locate`` places it) and the fraction of that bar elapsed, linear in ms."""
    times = np.asarray(times, dtype=np.float64)
    if not len(times):
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    bar = grid.locate(times)['bar'].astype(np.int64)
    start, end = grid.time_of_bar(bar), grid.time_of_bar(bar + 1)
    return bar, (times - start) / np.maximum(end - start, 1e-9)


def time_at_bar_fraction(grid: BeatGrid, bar, fraction) -> np.ndarray:
    bar = np.asarray(bar, dtype=np.int64)
    start, end = grid.time_of_bar(bar), grid.time_of_bar(bar + 1)
    return start + np.asarray(fraction, dtype=np.float64) * (end - start)
