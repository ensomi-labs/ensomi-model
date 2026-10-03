"""What one evaluation judges: a chart, the condition it was made under, a scope.

A condition composes any of three components: ``Timing`` (a beat grid) or
``Skeleton`` (a grid plus head and release times), ``Context`` (parts of a chart
given in advance) and ``Audio`` (the full song). Components combine with ``|``:
``Skeleton.from_chart(ref) | Context.from_chart(ref, Spans.parse('..00:30:000'))``.
Every result names its combination.

A scope is two span sets on the chart's time: ``given`` (what measures may read
as context) and ``scored`` (what they judge). Both are free: a continuation, a
local edit, or one passage with nothing given. Nothing a condition fixed is
scored: scored spans may not overlap the condition's context, and an operator
that judges an aspect the condition fixed (the grid, or head and release times
under a skeleton) is not run.

The grid of a case is the condition's when it has one; otherwise the chart's
own musical grid, whose fit to the audio is the timing evaluation's question.
Objects belong to spans by their head time; a long note's release goes with
its head. Operators (rhythm, arrangement, load) are not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from ..osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from .beats import BeatGrid, ChartEvents, chart_events, event_times
from .redlines import LineRole, RedLine, musical_grid, read_red_lines

GRID, HEAD_TIMES, RELEASE_TIMES = 'grid', 'head_times', 'release_times'


@dataclass(frozen=True)
class Spans:
    """A union of half-open ``[start, end)`` intervals in ms, sorted and merged."""
    intervals: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        merged: list[list[float]] = []
        for start, end in sorted((float(a), float(b)) for a, b in self.intervals):
            if math.isnan(start) or math.isnan(end):
                raise ValueError('Span bounds must not be NaN')
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        object.__setattr__(self, 'intervals', tuple((a, b) for a, b in merged))

    @classmethod
    def of(cls, *intervals: tuple[float, float]) -> 'Spans':
        return cls(tuple(intervals))

    @classmethod
    def everything(cls) -> 'Spans':
        return cls(((-math.inf, math.inf),))

    @classmethod
    def before(cls, t_ms: float) -> 'Spans':
        return cls(((-math.inf, t_ms),))

    @classmethod
    def after(cls, t_ms: float) -> 'Spans':
        return cls(((t_ms, math.inf),))

    @classmethod
    def parse(cls, text: str) -> 'Spans':
        """Spans from text: comma-separated ``start..end``, either side open.

        Bounds are osu! editor timestamps (``mm:ss:mmm``, or ``h:mm:ss:mmm``) or
        plain milliseconds: ``'00:12:345..00:40:000'``, ``'..30000'``,
        ``'01:00:000..'``.
        """
        intervals = []
        for part in filter(None, (p.strip() for p in text.split(','))):
            if '..' not in part:
                raise ValueError(f'A span needs "start..end", got {part!r}')
            start, end = (s.strip() for s in part.split('..', 1))
            intervals.append((_parse_time(start, -math.inf), _parse_time(end, math.inf)))
        return cls(tuple(intervals))

    @classmethod
    def bars(cls, grid: BeatGrid, first: int, stop: int) -> 'Spans':
        """Global bars ``first`` to ``stop - 1`` of ``grid``."""
        start, end = grid.time_of_bar([first, stop])
        return cls(((float(start), float(end)),))

    def __bool__(self) -> bool:
        return bool(self.intervals)

    def __or__(self, other: 'Spans') -> 'Spans':
        return Spans(self.intervals + other.intervals)

    def __invert__(self) -> 'Spans':
        edges, out = -math.inf, []
        for start, end in self.intervals:
            out.append((edges, start))
            edges = end
        out.append((edges, math.inf))
        return Spans(tuple(out))

    def __and__(self, other: 'Spans') -> 'Spans':
        return ~(~self | ~other)

    def __sub__(self, other: 'Spans') -> 'Spans':
        return self & ~other

    def contains(self, times_ms) -> np.ndarray:
        times = np.asarray(times_ms, dtype=np.float64)
        mask = np.zeros(times.shape, dtype=bool)
        for start, end in self.intervals:
            mask |= (times >= start) & (times < end)
        return mask

    def describe(self) -> str:
        return ','.join(f'{_format_time(a)}..{_format_time(b)}' for a, b in self.intervals) or 'none'


def _parse_time(text: str, open_value: float) -> float:
    """Milliseconds from ``''`` (open), a number, ``mm:ss:mmm`` or ``h:mm:ss:mmm``."""
    if not text:
        return open_value
    fields = text.split(':')
    if len(fields) == 1:
        return float(text)
    if len(fields) not in (3, 4) or not all(f.isdigit() for f in fields):
        raise ValueError(f'Expected milliseconds or an osu! editor timestamp mm:ss:mmm, got {text!r}')
    *hours_minutes, seconds, ms = (int(f) for f in fields)
    hours, minutes = hours_minutes if len(hours_minutes) == 2 else (0, hours_minutes[0])
    return float(((hours * 60 + minutes) * 60 + seconds) * 1000 + ms)


def _format_time(ms: float) -> str:
    return '' if math.isinf(ms) else f'{ms:g}'


@dataclass(frozen=True)
class Chart:
    """A mania chart: hit objects and red lines as written, plus where it came from."""
    objects: tuple[ManiaHitObject, ...]
    red_lines: tuple[RedLine, ...] = ()
    keys: int = 4
    source: str | None = None

    @classmethod
    def from_osu(cls, path: str | Path, *, keys: int = 4) -> 'Chart':
        return cls(tuple(parse_mania_hit_objects(path, expected_key_count=keys)), tuple(read_red_lines(path)),
                   keys, str(path))

    def musical_grid(self) -> tuple[BeatGrid, list[LineRole]]:
        """The chart's own grid: its musical red lines (see ``redlines``)."""
        return musical_grid(self.red_lines, event_times(self.objects))

    def section(self, spans: Spans) -> 'Chart':
        """The objects whose head lies in ``spans``; red lines are kept whole."""
        heads = np.array([o.start_time_ms for o in self.objects])
        keep = spans.contains(heads) if len(heads) else np.zeros(0, dtype=bool)
        return replace(self, objects=tuple(o for o, k in zip(self.objects, keep) if k))


class _Component:
    """A condition component; ``a | b`` composes components into a ``Condition``."""
    slot: str

    def __or__(self, other: '_Component | Condition') -> 'Condition':
        return Condition.of(self) | other


@dataclass(frozen=True)
class Timing(_Component):
    """The beat grid the generator was given."""
    grid: BeatGrid
    slot = 'timing'
    name = 'timing'


@dataclass(frozen=True)
class Skeleton(_Component):
    """A grid plus every head time, and release times when the generator was given them.

    ``release_ms`` is None when releases were not given, as the human decided
    for the generator on 2026-10-03; release times are then the generator's.
    """
    grid: BeatGrid
    head_ms: tuple[float, ...]
    release_ms: tuple[float, ...] | None = None
    slot = 'timing'
    name = 'skeleton'

    @classmethod
    def from_chart(cls, chart: Chart, grid: BeatGrid | None = None, *, releases: bool = False) -> 'Skeleton':
        heads = tuple(sorted(o.start_time_ms for o in chart.objects))
        given = (tuple(sorted(o.end_time_ms for o in chart.objects if o.kind is ManiaHitObjectKind.HOLD))
                 if releases else None)
        return cls(grid if grid is not None else chart.musical_grid()[0], heads, given)


@dataclass(frozen=True)
class Context(_Component):
    """Parts of a chart given in advance: the objects whose heads lie in ``spans``."""
    chart: Chart
    spans: Spans
    slot = 'context'
    name = 'context'

    @classmethod
    def from_chart(cls, chart: Chart, spans: Spans) -> 'Context':
        return cls(chart.section(spans), spans)


@dataclass(frozen=True)
class Audio(_Component):
    """The full song the generator heard, read as log-Mel through ``ensomi_model.features``."""
    path: str
    sha256: str | None = None
    slot = 'audio'
    name = 'audio'


@dataclass(frozen=True)
class Condition:
    """Any combination of a timing (or skeleton), a context and an audio component.

    ``star`` is a scalar request the generator was given; references are
    conditioned on it when present.
    """
    timing: Timing | Skeleton | None = None
    context: Context | None = None
    audio: Audio | None = None
    star: float | None = None

    @classmethod
    def of(cls, *components: _Component, star: float | None = None) -> 'Condition':
        condition = cls(star=star)
        for component in components:
            condition = condition | component
        return condition

    def __or__(self, other: '_Component | Condition') -> 'Condition':
        parts = other.components if isinstance(other, Condition) else (other,)
        out = self
        for part in parts:
            if getattr(out, part.slot) is not None:
                raise ValueError(f'Condition already has a {part.slot} component')
            out = replace(out, **{part.slot: part})
        if isinstance(other, Condition) and other.star is not None:
            if out.star is not None:
                raise ValueError('Condition already has a star request')
            out = replace(out, star=other.star)
        return out

    @property
    def components(self) -> tuple[_Component, ...]:
        return tuple(c for c in (self.timing, self.context, self.audio) if c is not None)

    @property
    def name(self) -> str:
        return '+'.join(c.name for c in self.components) or 'none'

    @property
    def fixed(self) -> frozenset[str]:
        """Aspects of the whole chart the condition fixed: ``grid``, ``head_times``, ``release_times``."""
        if self.timing is None:
            return frozenset()
        if not isinstance(self.timing, Skeleton):
            return frozenset((GRID,))
        return frozenset((GRID, HEAD_TIMES) + ((RELEASE_TIMES,) if self.timing.release_ms is not None else ()))

    @property
    def given_spans(self) -> Spans:
        return self.context.spans if self.context is not None else Spans()


@dataclass(frozen=True)
class Scope:
    """The given part (read as context) and the scored part of a chart's time."""
    given: Spans = field(default_factory=Spans)
    scored: Spans = field(default_factory=Spans.everything)

    def __post_init__(self):
        if self.given & self.scored:
            raise ValueError(f'Given and scored spans overlap: {(self.given & self.scored).describe()}')

    @classmethod
    def whole(cls) -> 'Scope':
        return cls()

    @classmethod
    def continuation(cls, t_ms: float) -> 'Scope':
        return cls(Spans.before(t_ms), Spans.after(t_ms))

    @classmethod
    def edit(cls, spans: Spans) -> 'Scope':
        return cls(~spans, spans)

    @classmethod
    def passage(cls, spans: Spans) -> 'Scope':
        return cls(Spans(), spans)

    def describe(self) -> str:
        return f'given={self.given.describe()} scored={self.scored.describe()}'


@dataclass(frozen=True)
class EvalCase:
    """A chart, the condition it was made under, and the scope it is judged on."""
    chart: Chart
    condition: Condition = field(default_factory=Condition)
    scope: Scope = field(default_factory=Scope)

    def __post_init__(self):
        fixed = self.condition.given_spans & self.scope.scored
        if fixed:
            raise ValueError(f'Scored spans overlap the condition context: {fixed.describe()}')

    @property
    def grid(self) -> BeatGrid:
        if self.condition.timing is not None:
            return self.condition.timing.grid
        return self.chart.musical_grid()[0]

    @property
    def grid_source(self) -> str:
        return 'condition' if self.condition.timing is not None else 'chart'

    def events(self) -> tuple[ChartEvents, np.ndarray, np.ndarray]:
        """Events on the case grid, with masks of the given and the scored ones."""
        events = chart_events(self.chart.objects, self.grid)
        return events, self.scope.given.contains(events.head_ms), self.scope.scored.contains(events.head_ms)

    def describe(self) -> dict[str, Any]:
        return dict(source=self.chart.source, condition=self.condition.name, star=self.condition.star,
                    fixed=sorted(self.condition.fixed), grid_source=self.grid_source,
                    given=self.scope.given.describe(), scored=self.scope.scored.describe())


class Operator(Protocol):
    """A measure on a case. ``judges`` names the aspects it scores (e.g. ``head_times``)."""
    name: str
    judges: frozenset[str]

    def __call__(self, case: EvalCase) -> Mapping[str, Any]: ...


def evaluate(case: EvalCase, operators: Iterable[Operator]) -> dict[str, Any]:
    """Run each operator on ``case``; one that judges an aspect the condition fixed is skipped."""
    results: dict[str, Any] = {}
    for operator in operators:
        clash = operator.judges & case.condition.fixed
        results[operator.name] = (dict(skipped=f'condition fixes {", ".join(sorted(clash))}') if clash
                                  else dict(operator(case)))
    return dict(case=case.describe(), results=results)


def condition_from_chart(source: Chart, components: Sequence[str], *, given: Spans = Spans(),
                         audio: Audio | None = None, star: float | None = None) -> Condition:
    """The condition a generator would receive from ``source``.

    ``components`` names any of ``timing``, ``skeleton`` (head times) or
    ``skeleton+releases``, ``context`` (the objects of ``source`` whose heads
    lie in ``given``) and ``audio``.
    """
    names = set(components)
    unknown = names - {'timing', 'skeleton', 'skeleton+releases', 'context', 'audio'}
    if unknown:
        raise ValueError(f'Unknown condition components: {sorted(unknown)}')
    parts: list[_Component] = []
    if len(names & {'timing', 'skeleton', 'skeleton+releases'}) > 1:
        raise ValueError('A condition has one timing or skeleton component')
    if names & {'skeleton', 'skeleton+releases'}:
        parts.append(Skeleton.from_chart(source, releases='skeleton+releases' in names))
    elif 'timing' in names:
        parts.append(Timing(source.musical_grid()[0]))
    if 'context' in names:
        parts.append(Context.from_chart(source, given))
    if 'audio' in names:
        if audio is None:
            raise ValueError('The audio component needs an Audio reference')
        parts.append(audio)
    return Condition.of(*parts, star=star)
