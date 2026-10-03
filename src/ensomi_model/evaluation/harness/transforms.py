"""Must-not-flag transforms: changes to a chart that describe the same chart to the evaluator.

Each transform returns the transformed chart, the ``.osu`` writer extras it
needs, and the grid of its ``Timing`` condition: by default the transformed
chart's own musical grid, or a grid the transform supplies (per-segment
renotation, stretch of the condition grid), or none (the chart's own grid read
by the case, against the same grid given by the condition). It returns a reason
string when it does not apply to a chart (a renotation or stretch that would
move a red line across the plausible BPM range).

Every transform but the time-stretch must leave the evaluator's whole output
identical; a shift is compared after shifting the output back
(``time_offset``). A time-stretch is a covariance test instead: quantities in
seconds move by the stretch factor (``stretch``), so its output is compared
with the source's through the stated scaling, not by hash.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ...osu_core.hitobjects import ManiaHitObject
from ..beats import BarStart, BeatGrid, Segment
from ..case import Chart
from ..redlines import EXPRESSIVE, MUSICAL, RedLine


@dataclass(frozen=True)
class Transformed:
    chart: Chart
    extras: dict
    grid_from_chart: bool = False
    check_expressive: tuple[RedLine, ...] = ()
    grid: BeatGrid | None = None
    time_offset: float = 0.0


@dataclass(frozen=True)
class Transform:
    name: str
    title: str
    apply: Callable[[Chart, int], 'Transformed | str']
    stretch: float | None = None


def _objects(chart: Chart, fn) -> tuple[ManiaHitObject, ...]:
    return tuple(fn(o) for o in chart.objects)


def mirror(chart: Chart, seed: int) -> Transformed:
    return Transformed(replace(chart, objects=_objects(chart, lambda o: replace(o, lane=chart.keys - 1 - o.lane))), {})


def _renotated(lines, factors) -> tuple[RedLine, ...] | None:
    out = tuple(replace(line, beat_length_ms=line.beat_length_ms / 2.0 ** f) for line, f in zip(lines, factors))
    if [a.plausible for a in out] != [b.plausible for b in lines]:
        return None
    return out


def renotate_whole(factor: int):
    def apply(chart: Chart, seed: int) -> Transformed | str:
        lines = _renotated(chart.red_lines, [factor] * len(chart.red_lines))
        if lines is None:
            return 'a red line would cross the plausible BPM range'
        return Transformed(replace(chart, red_lines=lines), {})
    return apply


def renotate_segments(chart: Chart, seed: int) -> Transformed:
    """Each musical segment of the condition grid renotated by its own factor in {-1, +1} (``beats.renotate``
    on the grid, bar starts kept); the chart is unchanged.

    Renotating one segment's red lines in the file instead is not a must-not-flag
    change: in osu! it changes that segment's scroll speed, and ``redlines`` may
    then read a bar line as a detour.
    """
    grid, _ = chart.musical_grid()
    factors = np.random.default_rng(seed).choice([-1, 1], size=len(grid.segments))
    segments = tuple(replace(s, beat_length_ms=s.beat_length_ms / 2.0 ** int(f)) for s, f in zip(grid.segments, factors))
    return Transformed(chart, {}, grid=BeatGrid(segments, grid.bar_starts))


def shift(ms: float):
    def apply(chart: Chart, seed: int) -> Transformed:
        objects = _objects(chart, lambda o: replace(o, start_time_ms=o.start_time_ms + ms, end_time_ms=o.end_time_ms + ms))
        lines = tuple(replace(line, offset_ms=line.offset_ms + ms) for line in chart.red_lines)
        return Transformed(replace(chart, objects=objects, red_lines=lines), {}, time_offset=ms)
    return apply


def add_expressive(chart: Chart, seed: int) -> Transformed | str:
    """An implausible red line (a stop) at the median head, and a superseded line at the first red line."""
    if not chart.red_lines or not chart.objects:
        return 'no red line or no object'
    heads = sorted(o.start_time_ms for o in chart.objects)
    stop = RedLine(float(heads[len(heads) // 2]), 1e-3, 4)
    superseded = RedLine(chart.red_lines[0].offset_ms, 1234.5, 3)
    lines = list(chart.red_lines)
    lines.insert(0, superseded)
    # Before any line at the same offset: a later line at one offset supersedes an earlier one.
    position = next((i for i, line in enumerate(lines) if line.offset_ms >= stop.offset_ms), len(lines))
    lines.insert(position, stop)
    return Transformed(replace(chart, red_lines=tuple(lines)), {}, check_expressive=(superseded, stop))


def green_and_hitsounds(chart: Chart, seed: int) -> Transformed:
    rng = np.random.default_rng(seed)
    heads = sorted({o.start_time_ms for o in chart.objects})[::4]
    greens = [(float(t), float(rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]))) for t in heads]
    return Transformed(chart, dict(green_lines=greens, hitsound_seed=seed))


def object_order(chart: Chart, seed: int) -> Transformed:
    return Transformed(chart, dict(order_seed=seed))


def own_grid(chart: Chart, seed: int) -> Transformed:
    return Transformed(chart, {}, grid_from_chart=True)


def stretch(factor: float, *, supplied_grid: bool = False):
    """Every time and beat length multiplied by ``factor``.

    With ``supplied_grid`` the condition grid is the source's musical grid
    stretched (segments and bar starts); otherwise the stretched chart's red
    lines are classified again and its own musical grid is used.
    """
    def apply(chart: Chart, seed: int) -> Transformed | str:
        grid, _ = chart.musical_grid()
        lines = tuple(replace(line, offset_ms=line.offset_ms * factor, beat_length_ms=line.beat_length_ms * factor)
                      for line in chart.red_lines)
        if [a.plausible for a in lines] != [b.plausible for b in chart.red_lines]:
            return 'a red line would cross the plausible BPM range'
        objects = _objects(chart, lambda o: replace(o, start_time_ms=o.start_time_ms * factor,
                                                   end_time_ms=o.end_time_ms * factor))
        stretched = replace(chart, objects=objects, red_lines=lines)
        if not supplied_grid:
            return Transformed(stretched, {})
        new = BeatGrid(tuple(Segment(s.offset_ms * factor, s.beat_length_ms * factor, s.meter) for s in grid.segments),
                       tuple(BarStart(b.time_ms * factor, b.meter) for b in grid.bar_starts))
        return Transformed(stretched, {}, grid=new)
    return apply


TRANSFORMS: tuple[Transform, ...] = (
    Transform('mirror', 'lane mirror', mirror),
    Transform('renotate_x2', 'whole-chart renotation at double BPM', renotate_whole(1)),
    Transform('renotate_half', 'whole-chart renotation at half BPM', renotate_whole(-1)),
    Transform('renotate_segments', 'per-segment renotation of the grid, half or double per segment', renotate_segments),
    Transform('shift_37', 'chart and grid shifted by 37 ms', shift(37.0)),
    Transform('shift_1000', 'chart and grid shifted by 1000 ms', shift(1000.0)),
    Transform('expressive_lines', 'added expressive red lines (stop, superseded)', add_expressive),
    Transform('green_hitsounds', 'green lines and hitsounds', green_and_hitsounds),
    Transform('object_order', 'object order in the file shuffled', object_order),
    Transform('own_grid', "chart's own grid instead of the same grid from the condition", own_grid),
    Transform('stretch_1.05', 'time-stretch x1.05, grid from the stretched red lines', stretch(1.05), stretch=1.05),
    Transform('stretch_0.95', 'time-stretch x0.95, grid from the stretched red lines', stretch(0.95), stretch=0.95),
    Transform('stretch_1.05_grid', 'time-stretch x1.05, condition grid stretched', stretch(1.05, supplied_grid=True),
              stretch=1.05),
    Transform('stretch_0.95_grid', 'time-stretch x0.95, condition grid stretched', stretch(0.95, supplied_grid=True),
              stretch=0.95),
)


def expressive_ok(chart: Chart, inserted: tuple[RedLine, ...]) -> bool:
    """Whether every inserted line is classified expressive in ``chart``."""
    if not inserted:
        return True
    _, roles = chart.musical_grid()
    remaining = list(inserted)
    for line, role in zip(chart.red_lines, roles):
        for k, want in enumerate(remaining):
            if line.offset_ms == want.offset_ms and line.beat_length_ms == want.beat_length_ms and line.meter == want.meter:
                if role.role != EXPRESSIVE:
                    return False
                remaining.pop(k)
                break
    return not remaining

