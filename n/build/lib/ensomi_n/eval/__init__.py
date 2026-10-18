"""The N line's timing evaluation (E0): measures, nulls, aggregation and the command line.

For a training loop:

- `score_beats` scores in-memory beat arrays against reference beats;
- `score` scores a `MusicMap` (or a `Timing`, or a beat array) against a
  `Reference` from `ensomi_n.labels.read_set`;
- `aggregate` turns set scores into stratum means with recording-bootstrap
  intervals.

``python -m ensomi_n.eval score`` runs the same on files; see ``__main__``.
"""
from __future__ import annotations

from typing import Sequence

from ..mapformat import MusicMap
from ..timing import Reference, Segment
from .measures import DECIDING_TAU, TAUS, SetScore, score
from .nulls import chance, check as check_nulls, table as null_table
from .report import aggregate, build_strata, read_strata

__all__ = ['score', 'score_beats', 'equivariance', 'SetScore', 'aggregate', 'build_strata', 'read_strata', 'chance',
           'null_table', 'check_nulls', 'TAUS', 'DECIDING_TAU']


def score_beats(ref_beats, pred_beats, heads=None, *, ref_downbeats=None, ref_segments: Sequence[Segment] | None = None,
                pred_downbeats=None, pred_confidence=None, pred_subdivision=None,
                pred_segments: Sequence[Segment] | None = None, lo_ms: float | None = None,
                hi_ms: float | None = None, key: str = '', **kwargs) -> SetScore:
    """Score predicted beats against reference beats, both in ms on one clock.

    ``heads`` weight the reference lines; without them every line weighs 1
    and BL equals BL per beat. Downbeats are index lists or boolean masks.
    The scored range defaults to the heads' span, else the reference beats'.
    Other keywords go to `score`.
    """
    ref = Reference.from_beats(ref_beats, heads, downbeats=ref_downbeats, segments=ref_segments, lo_ms=lo_ms,
                               hi_ms=hi_ms, key=key)
    pred = MusicMap.from_beats(pred_beats, pred_downbeats, pred_confidence, pred_subdivision,
                               segments=pred_segments)
    return score(ref, pred, **kwargs)


def equivariance(src: MusicMap, edit: MusicMap, rate: float, **kwargs) -> SetScore:
    """Score a rate edit's map against the source's map played at ``rate``.

    A model is equivariant when its map of audio sped up by ``rate`` equals
    its map of the source with every time divided by ``rate``; that scores
    BL 1 and offset 0 here. The source map, scaled, is the reference, every
    line weighing 1. Both maps need beats.
    """
    if src.beats is None or not len(src.beats):
        raise ValueError('The source map needs beats')
    timing = src.timing(0.0, 0.0).scaled(1.0 / rate)
    segments = None if not src.segments else tuple(
        Segment(s.start_ms / rate, s.beat_ms / rate, s.meter, s.bar_at_start) for s in src.segments)
    ref = Reference(timing, float(timing.beats[0]), float(timing.beats[-1]), None, segments)
    return score(ref, edit, **kwargs)
