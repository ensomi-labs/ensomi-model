"""Must-not-flag transforms: changes to a chart that describe the same chart to the evaluator.

Each transform returns the transformed chart, the ``.osu`` writer extras it
needs, and the grid of its ``Timing`` condition: by default the transformed
chart's own musical grid, or a grid the transform supplies. It returns a reason
string when it does not apply to a chart (a stretch that would move a red line
across the plausible BPM range).

The timing lesion removes every red line, adds green lines and gives the
condition an unrelated grid: one test that the evaluator reads no timing point.

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


@dataclass(frozen=True)
class Transformed:
    chart: Chart
    extras: dict
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


def shift(ms: float):
    def apply(chart: Chart, seed: int) -> Transformed:
        objects = _objects(chart, lambda o: replace(o, start_time_ms=o.start_time_ms + ms, end_time_ms=o.end_time_ms + ms))
        lines = tuple(replace(line, offset_ms=line.offset_ms + ms) for line in chart.red_lines)
        return Transformed(replace(chart, objects=objects, red_lines=lines), {}, time_offset=ms)
    return apply


def object_order(chart: Chart, seed: int) -> Transformed:
    return Transformed(chart, dict(order_seed=seed))


def timing_lesion(chart: Chart, seed: int) -> Transformed:
    """No red line in the file, green lines on every fourth distinct head, a seeded unrelated condition grid."""
    rng = np.random.default_rng(seed)
    heads = sorted({o.start_time_ms for o in chart.objects})[::4]
    greens = [(float(t), float(rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]))) for t in heads]
    offset, beat = float(rng.uniform(0.0, 1000.0)), float(60000.0 / rng.uniform(80.0, 160.0))
    grid = BeatGrid((Segment(offset, beat, 4),), (BarStart(offset, 4),))
    return Transformed(replace(chart, red_lines=()), dict(green_lines=greens), grid=grid)


def stretch(factor: float):
    """Every time and beat length multiplied by ``factor``; the stretched red lines are classified again."""
    def apply(chart: Chart, seed: int) -> Transformed | str:
        lines = tuple(replace(line, offset_ms=line.offset_ms * factor, beat_length_ms=line.beat_length_ms * factor)
                      for line in chart.red_lines)
        if [a.plausible for a in lines] != [b.plausible for b in chart.red_lines]:
            return 'a red line would cross the plausible BPM range'
        objects = _objects(chart, lambda o: replace(o, start_time_ms=o.start_time_ms * factor,
                                                   end_time_ms=o.end_time_ms * factor))
        return Transformed(replace(chart, objects=objects, red_lines=lines), {})
    return apply


TRANSFORMS: tuple[Transform, ...] = (
    Transform('mirror', 'lane mirror', mirror),
    Transform('shift_37', 'chart and grid shifted by 37 ms', shift(37.0)),
    Transform('shift_1000', 'chart and grid shifted by 1000 ms', shift(1000.0)),
    Transform('object_order', 'object order in the file shuffled', object_order),
    Transform('timing_lesion', 'red lines removed, green lines added, unrelated condition grid', timing_lesion),
    Transform('stretch_1.05', 'time-stretch x1.05, grid from the stretched red lines', stretch(1.05), stretch=1.05),
    Transform('stretch_0.95', 'time-stretch x0.95, grid from the stretched red lines', stretch(0.95), stretch=0.95),
)
