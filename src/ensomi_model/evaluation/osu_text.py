"""A chart written as ``.osu`` text and read back, as a generator's output would be.

Hit object times are written as integer milliseconds, as osu! writes them, so a
chart built with fractional times is judged after the rounding a generator's
file would carry. Red lines are written as given (offset and beat length at
full float precision, meter, omit-first-barline flag). Optional extras that
must not change any evaluation: green lines (inherited timing points),
hitsounds and samples on hit objects, and a shuffled object order.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Sequence

from ..osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from .case import Chart
from .redlines import RedLine


def lane_x(lane: int, keys: int = 4) -> int:
    if not 0 <= lane < keys:
        raise ValueError(f'lane outside 0..{keys - 1}: {lane}')
    return int((lane + 0.5) * 512 / keys)


def osu_text(chart: Chart, *, green_lines: Sequence[tuple[float, float]] = (), hitsound_seed: int | None = None,
             order_seed: int | None = None, title: str = 'harness') -> str:
    """``.osu`` v14 text of ``chart``.

    ``green_lines`` are ``(time_ms, scroll_multiplier)`` inherited points.
    ``hitsound_seed`` gives every object a seeded hitsound and sample set;
    ``order_seed`` shuffles the order of the hit object lines.
    """
    keys = chart.keys
    lines = ['osu file format v14', '', '[General]', 'AudioFilename: audio.mp3', 'Mode: 3', '',
             '[Metadata]', f'Title:{title}', 'Artist:harness', 'Creator:harness', 'Version:v0', '',
             '[Difficulty]', 'HPDrainRate:8', f'CircleSize:{keys}', 'OverallDifficulty:8', 'ApproachRate:5',
             'SliderMultiplier:1.4', 'SliderTickRate:1', '', '[TimingPoints]']
    points = [(line.offset_ms, 0, _red(line)) for line in chart.red_lines]
    points += [(float(t), 1, f'{t!r},{-100.0 / sv!r},4,1,0,100,0,0') for t, sv in green_lines]
    lines += [text for _, _, text in sorted(points, key=lambda p: (p[0], p[1]))] if green_lines else \
        [text for _, _, text in points]
    lines += ['', '[HitObjects]']
    sounds = random.Random(hitsound_seed) if hitsound_seed is not None else None
    objects = sorted(chart.objects, key=lambda o: (_ms(o.start_time_ms), o.lane))
    rows = [_object_line(o, keys, sounds) for o in objects]
    if order_seed is not None:
        random.Random(order_seed).shuffle(rows)
    return '\n'.join(lines + rows) + '\n'


def _red(line: RedLine) -> str:
    return f'{line.offset_ms!r},{line.beat_length_ms!r},{line.meter},1,0,100,1,{8 if line.omit_first_barline else 0}'


def written_ms(t: float) -> float:
    """The time as an ``.osu`` file carries it: integer ms (Python rounding, half to even)."""
    return float(int(round(float(t))))


def _ms(t: float) -> int:
    return int(written_ms(t))


def _object_line(o: ManiaHitObject, keys: int, sounds: random.Random | None) -> str:
    hitsound, sample = (0, '0:0:0:0:') if sounds is None else (
        sounds.choice((0, 2, 4, 8, 10)), f'{sounds.randint(0, 3)}:{sounds.randint(0, 3)}:0:{sounds.randint(5, 100)}:')
    x = lane_x(o.lane, keys)
    if o.kind is ManiaHitObjectKind.HOLD:
        return f'{x},192,{_ms(o.start_time_ms)},128,{hitsound},{_ms(o.end_time_ms)}:{sample}'
    return f'{x},192,{_ms(o.start_time_ms)},1,{hitsound},{sample}'


def read_osu_text(text: str, *, keys: int = 4, source: str | None = None) -> Chart:
    """``Chart.from_osu`` on ``text`` (through a temporary file)."""
    handle, path = tempfile.mkstemp(suffix='.osu', prefix='harness-')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as out:
            out.write(text)
        chart = Chart.from_osu(Path(path), keys=keys)
    finally:
        os.unlink(path)
    return Chart(chart.objects, chart.red_lines, chart.keys, source)


def round_trip(chart: Chart, **extras) -> Chart:
    """``chart`` written as ``.osu`` text (see ``osu_text``) and parsed back."""
    return read_osu_text(osu_text(chart, **extras), keys=chart.keys, source=chart.source)
