"""Passages of a scored span, and the one-family trivial evaluator built on them.

A segmentation turns a case (chart, grid, scope) into passages: each has a span
in ms, the musical segment ids it covers, its key fields, and a beat-clock
label used to compare outputs across transforms. It also says which passage
each event of ``case.events()`` belongs to.

``Bars4Placeholder`` (``bars4-v0-placeholder``) is the only segmentation here:
four consecutive canonical bars, never crossing a musical segment boundary,
from the bar of the scored span's start (the first scored head when the span
is open) to the bar of the last scored head; a trailing incomplete group of
bars in a segment is dropped and counted. It is a placeholder under review: a
four-bar passage doubles its length in ms across a fold change such as 158 to
162 BPM (canonical 158, then 81).

``HeadCountPerPassage`` is the trivial evaluator's operator: the number of
heads (taps and long-note heads) in each passage. It judges head times, so
``evaluate`` skips it under a skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol

import numpy as np

from .beats import LN_RELEASE, BeatGrid
from .case import HEAD_TIMES, EvalCase

PLACEHOLDER = 'bars4-v0-placeholder'


@dataclass(frozen=True)
class PassageSet:
    """Passages of one case, in time order.

    ``label`` identifies a passage on the beat clock (here its first global
    bar), so it is unchanged by shifts, renotation and time-stretch.
    ``event_passage`` is the passage index of each event of ``case.events()``,
    -1 when the event lies in no passage. ``diagnostics`` holds counts a
    reviewer of the segmentation needs (dropped bars, short segments).
    """
    start_ms: np.ndarray
    end_ms: np.ndarray
    segments: tuple[tuple[int, ...], ...]
    canonical_bpm: np.ndarray
    label: np.ndarray
    event_passage: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.start_ms)


class Segmentation(Protocol):
    name: str

    def __call__(self, case: EvalCase) -> PassageSet: ...


def scored_bar_range(case: EvalCase, coords: Mapping[str, np.ndarray], scored_heads: np.ndarray) -> tuple[int, int] | None:
    """First and last global bar of the scored span, or None when no head is scored.

    The scope's scored part must be one interval: everything, or ``[t, inf)``
    (a continuation). When finite, the first bar is the first that starts at or
    after ``t``; a continuation cut just before a bar start (see the harness)
    keeps heads snapped onto that bar start in the scored part.
    """
    intervals = case.scope.scored.intervals
    if len(intervals) != 1 or not math.isinf(intervals[0][1]):
        raise NotImplementedError('Passages support a whole-chart or a continuation scope only')
    if not scored_heads.any():
        return None
    start = intervals[0][0]
    if math.isinf(start):
        first = int(coords['bar'][scored_heads].min())
    else:
        first = int(case.grid.locate([start])['bar'][0])
        first += bool(case.grid.time_of_bar([first])[0] < start - 1e-9)
    return first, int(coords['bar'][scored_heads].max())


@dataclass(frozen=True)
class Bars4Placeholder:
    bars: int = 4
    name: str = PLACEHOLDER

    def layout(self, grid: BeatGrid, first_bar: int, last_bar: int) -> dict[str, Any]:
        """Passages over global bars ``first_bar..last_bar``: groups of ``bars`` inside one segment."""
        bar_ids = np.arange(first_bar, last_bar + 1)
        starts = grid.time_of_bar(bar_ids) if len(bar_ids) else np.zeros(0)
        offsets = np.array([s.offset_ms for s in grid.segments])
        segment = np.clip(np.searchsorted(offsets, starts + 1e-6, side='right') - 1, 0, len(offsets) - 1)
        first, seg_of, dropped_bars, dropped_chunks, runs, short_segments = [], [], 0, 0, 0, 0
        i = 0
        while i < len(bar_ids):
            j = i
            while j < len(bar_ids) and segment[j] == segment[i]:
                j += 1
            runs += 1
            run = j - i
            short_segments += run < self.bars
            for k in range(run // self.bars):
                first.append(int(bar_ids[i + k * self.bars]))
                seg_of.append(int(segment[i]))
            dropped_bars += run % self.bars
            dropped_chunks += run % self.bars > 0
            i = j
        first_arr = np.array(first, dtype=np.int64)
        start_ms = grid.time_of_bar(first_arr) if len(first_arr) else np.zeros(0)
        end_ms = grid.time_of_bar(first_arr + self.bars) if len(first_arr) else np.zeros(0)
        cbpm = np.array([grid.segments[s].canonical_bpm for s in seg_of], dtype=np.float64)
        folds = [grid.segments[int(s)].fold for s in np.unique(segment)] if len(segment) else []
        return dict(first_bar=first_arr, segment=np.array(seg_of, dtype=np.int64), start_ms=start_ms,
                    end_ms=end_ms, canonical_bpm=cbpm, dropped_bars=int(dropped_bars),
                    dropped_passages=int(dropped_chunks), segment_runs=int(runs),
                    short_segments=int(short_segments), bar_segment=segment, bar_ids=bar_ids,
                    fold_changes=int(sum(a != b for a, b in zip(folds, folds[1:]))))

    def __call__(self, case: EvalCase) -> PassageSet:
        events, _, scored = case.events()
        heads = events.kind != LN_RELEASE
        bounds = scored_bar_range(case, events.coords, heads & scored)
        if bounds is None:
            empty = np.zeros(0)
            return PassageSet(empty, empty, (), empty, np.zeros(0, dtype=np.int64),
                              np.full(len(events), -1, dtype=np.int64),
                              dict(dropped_passages=0, dropped_bars=0, heads_dropped=0, scored_heads=0,
                                   short_segments=0, segment_runs=0, fold_changes=0))
        lay = self.layout(case.grid, *bounds)
        bar = events.coords['bar']
        index = np.full(bounds[1] - bounds[0] + 1, -1, dtype=np.int64)
        for p, b in enumerate(lay['first_bar']):
            index[b - bounds[0]:b - bounds[0] + self.bars] = p
        inside = scored & (bar >= bounds[0]) & (bar <= bounds[1])
        event_passage = np.full(len(events), -1, dtype=np.int64)
        event_passage[inside] = index[bar[inside] - bounds[0]]
        scored_heads = heads & scored
        return PassageSet(lay['start_ms'], lay['end_ms'], tuple((int(s),) for s in lay['segment']),
                          lay['canonical_bpm'], lay['first_bar'], event_passage,
                          dict(dropped_bars=lay['dropped_bars'],
                               dropped_passages=lay['dropped_passages'],
                               heads_dropped=int((scored_heads & (event_passage < 0)).sum()),
                               scored_heads=int(scored_heads.sum()), short_segments=lay['short_segments'],
                               segment_runs=lay['segment_runs'], fold_changes=lay['fold_changes']))


@dataclass(frozen=True)
class HeadCountPerPassage:
    """Heads per passage, in canonical beats: the trivial evaluator's one family."""
    segmentation: Segmentation = field(default_factory=Bars4Placeholder)
    name: str = 'head_count_per_passage'
    judges: frozenset[str] = frozenset((HEAD_TIMES,))

    def __call__(self, case: EvalCase) -> dict[str, Any]:
        passages = self.segmentation(case)
        events, _, scored = case.events()
        heads = (events.kind != LN_RELEASE) & scored & (passages.event_passage >= 0)
        count = np.bincount(passages.event_passage[heads], minlength=len(passages)).astype(np.int64)
        return dict(segmentation=self.segmentation.name, count=count, label=passages.label,
                    segments=passages.segments, canonical_bpm=passages.canonical_bpm,
                    start_ms=passages.start_ms, end_ms=passages.end_ms, diagnostics=dict(passages.diagnostics))
