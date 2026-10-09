"""Continuous phase and tempo from BeatThis's sub-frame peaks, after the legacy fit.

The legacy fitter reads the beat probability on its 20 ms frames: its offsets come from a 1 ms
search on a pulse template sampled at the frame times and its tempos from a 0.5 BPM step with
fractional candidates. Measured 2026-10-09 (`report-measure.md`, M3 and M4), its tempo on
exact synthetic tracks is off by up to 0.1 % and its phase moves by 4-21 ms on some songs when
the beats shift by a few ms inside the frame. ``refine_segments`` keeps the legacy segmentation,
tempo octave and downbeat phase and replaces each segment's (offset, beat length) by a weighted
robust least-squares line through the sub-frame peak times of the beats it covers, so the grid
follows the peaks rather than the frame grid. When every segment's tempo agrees within
``TEMPO_TOLERANCE`` (the rule ``grid.corrected_grid`` applies afterwards), one line is fitted to
the whole song's peaks, so the tempo is not the longest segment's extrapolated.
``downbeat_phase`` then moves the first offset by whole beats to the bar phase BeatThis's
downbeat activations carry.

Sub-frame peak position: the centroid of the beat probability over the peak frame and its two
neighbours on each side. Calibrated 2026-10-09 on BeatThis ``final0``'s response to synthetic
clicks with sample-exact onsets at four phases inside the frame (`report-step3.md`, E2): on
frame-commensurate tempi, where every beat shares one frame phase and the bias does not average
out, the integer frame varies by 10 ms with the phase, the parabola through the three logits by
6 ms, the probability centroid by 2.3 ms.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ...evaluation.beats import canonical_beat_length
from .config import GridFitterConfig
from .types import TimingSegment

MIN_PEAKS = 8               # fewer on-grid peaks: the legacy values stay
MIN_ON_GRID_SHARE = 0.4     # a smaller share of the segment's peaks on its grid: the legacy values stay
WINDOW_BEATS = 0.25         # a peak belongs to the nearest beat when within this fraction of a beat ...
WINDOW_MAX_MS = 50.0        # ... and within this many ms
TRIM_SIGMAS = 3.0           # second pass drops residuals beyond this many robust SDs (floor 5 ms)
TEMPO_TOLERANCE = 0.003


CENTROID_FRAMES = 2         # the centroid runs over the peak frame and this many frames on each side


def beat_peaks(beat_logits, frame_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Sub-frame beat peak times (ms) and probabilities from BeatThis beat logits.

    Peaks are local maxima over ±3 frames with logit > 0 (BeatThis's "minimal" rule); adjacent
    equal maxima merge to their mean frame. Each peak's time is the probability-weighted centroid
    of the frames within ``CENTROID_FRAMES`` of it; frame j is centred at j / frame_rate_hz. The
    probability is the peak frame's.
    """
    x = np.asarray(beat_logits, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        return np.zeros(0), np.zeros(0)
    padded = np.pad(x, 3, constant_values=-np.inf)
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 7).max(axis=1)
    candidates = np.flatnonzero((x >= local_max) & (x > 0.0))
    if candidates.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    probability = 1.0 / (1.0 + np.exp(-x))
    frames, heights = [], []
    for group in np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1):
        centre = int(round(float(np.mean(group))))
        heights.append(float(probability[group].max()))
        lo, hi = centre - CENTROID_FRAMES, centre + CENTROID_FRAMES + 1
        if lo < 0 or hi > n:
            frames.append(float(np.mean(group)))
            continue
        k = np.arange(lo, hi, dtype=np.float64)
        frames.append(float(np.dot(k, probability[lo:hi]) / probability[lo:hi].sum()))
    return np.asarray(frames) * 1000.0 / frame_rate_hz, np.asarray(heights)


def _line(times, weights, offset_ms, beat_length_ms):
    """Weighted least squares of t ≈ o + n L, with n the beat index on the given grid."""
    n = np.round((times - offset_ms) / beat_length_ms)
    sw = np.sqrt(weights)
    design = np.column_stack((np.ones_like(n), n)) * sw[:, None]
    (offset, length), *_ = np.linalg.lstsq(design, times * sw, rcond=None)
    return float(offset), float(length)


def _refine_one(times, weights, offset_ms, beat_length_ms):
    """(offset, beat length, peaks used), or None when too few peaks sit on the grid."""
    if times.shape[0] == 0:
        return None
    window = min(WINDOW_BEATS * beat_length_ms, WINDOW_MAX_MS)
    residual = times - (offset_ms + np.round((times - offset_ms) / beat_length_ms) * beat_length_ms)
    keep = np.abs(residual) < window
    if keep.sum() < MIN_PEAKS or keep.mean() < MIN_ON_GRID_SHARE:
        return None
    offset, length = _line(times[keep], weights[keep], offset_ms, beat_length_ms)
    residual = times - (offset + np.round((times - offset) / length) * length)
    scale = 1.4826 * float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
    keep = np.abs(residual) < min(window, max(TRIM_SIGMAS * scale, 5.0))
    if keep.sum() < MIN_PEAKS:
        return offset, length, int((np.abs(residual) < window).sum())
    offset, length = _line(times[keep], weights[keep], offset, length)
    return offset, length, int(keep.sum())


def _aligned_lengths(lengths):
    """Beat lengths rescaled by powers of two to the first one's octave."""
    lengths = np.asarray(lengths, dtype=np.float64)
    return lengths * 2.0 ** np.round(np.log2(lengths[0] / lengths))


def refine_segments(segments: Sequence[TimingSegment], beat_logits, frame_rate_hz: float,
                    config: GridFitterConfig = GridFitterConfig()) -> tuple[TimingSegment, ...]:
    """The legacy segments with each (offset, beat length) refitted to the sub-frame beat peaks.

    Beat index 0 of every refit is the segment's legacy offset beat, so the downbeat phase the
    legacy fitter chose is kept. Offsets are rounded to ``config.refine_offset_quantum_ms``.
    """
    segments = tuple(segments)
    if not segments:
        return segments
    times, weights = beat_peaks(beat_logits, frame_rate_hz)
    starts = [-np.inf] + [s.offset_ms for s in segments[1:]]
    ends = [s.offset_ms for s in segments[1:]] + [np.inf]
    refined, used = [], []
    for segment, start, end in zip(segments, starts, ends):
        inside = (times >= start) & (times < end)
        fit = _refine_one(times[inside], weights[inside], segment.offset_ms, segment.beat_length_ms)
        if fit is None:
            refined.append(segment)
            used.append(0)
        else:
            refined.append(TimingSegment(offset_ms=fit[0], beat_length_ms=fit[1]))
            used.append(fit[2])
    if len(refined) > 1:
        aligned = _aligned_lengths([s.beat_length_ms for s in refined])
        if np.all(np.abs(aligned / aligned[0] - 1.0) < TEMPO_TOLERANCE):
            frame_ms = 1000.0 / frame_rate_hz
            span = np.diff([*[s.offset_ms for s in refined], np.asarray(beat_logits).shape[0] * frame_ms])
            longest = refined[int(np.argmax(span))]
            fit = _refine_one(times, weights, longest.offset_ms, longest.beat_length_ms)
            if fit is not None:
                refined = [TimingSegment(offset_ms=fit[0], beat_length_ms=fit[1])]
    quantum = config.refine_offset_quantum_ms
    return tuple(TimingSegment(offset_ms=round(s.offset_ms / quantum) * quantum, beat_length_ms=s.beat_length_ms)
                 for s in refined)


def downbeat_phase(segments: Sequence[TimingSegment], beat_logits, downbeat_logits, frame_rate_hz: float,
                   config: GridFitterConfig = GridFitterConfig()) -> tuple[TimingSegment, ...]:
    """The segments with the first one's bar phase taken from BeatThis's downbeat activations.

    R2 and the head model count the grid's bars from the first offset (meter 4, canonical beats).
    The first segment's beats from the first peak to the segment's end fall into one class per beat
    of a canonical bar; the class with the largest summed downbeat probability holds the bar's
    first beat. The offset moves to that class's beat nearest it, the later one when the class lies
    half a bar away on both sides, so it changes by whole beats and stays within half a canonical
    bar of where it was; the line, its tempo and the later segments are unchanged. R2 also reads a
    16-beat phase from the first offset: on fit_train (population-weighted) it agreed with the
    chart's on 46 % of charts with the legacy's offset, 50 % with this rule, 43 % with ties going
    earlier and 21 % with the offset moved back to the first peak. The first offset sits in the
    chart's bar on 615 of 812 single-tempo fit_train charts with this rule and on 530 with the
    legacy's downbeat (2026-10-09, ``~/ensomi/.sync/cp/scratch/timing/bar/report-bar.md``).
    With no downbeat logits, no peaks or fewer beats than one bar, the segments are returned unchanged.
    """
    segments = tuple(segments)
    if not segments or downbeat_logits is None:
        return segments
    times, _ = beat_peaks(beat_logits, frame_rate_hz)
    if times.shape[0] == 0:
        return segments
    offset, length = segments[0].offset_ms, segments[0].beat_length_ms
    bar = 4.0 * canonical_beat_length(length)
    period = max(int(round(bar / length)), 1)
    end = segments[1].offset_ms if len(segments) > 1 else times[-1]
    beats = np.arange(np.ceil((times[0] - offset) / length), np.floor((end - offset) / length) + 1)
    if beats.shape[0] < period:
        return segments
    x = np.asarray(downbeat_logits, dtype=np.float64)
    probability = np.interp(offset + beats * length, np.arange(x.shape[0]) * 1000.0 / frame_rate_hz,
                            1.0 / (1.0 + np.exp(-x)))
    score = [probability[(beats - beats[0]) % period == j].sum() for j in range(period)]
    shift = (int(beats[0]) + int(np.argmax(score))) % period
    if shift > period / 2:
        shift -= period
    anchor = offset + shift * length
    quantum = config.refine_offset_quantum_ms
    return (TimingSegment(offset_ms=round(anchor / quantum) * quantum, beat_length_ms=length),) + segments[1:]
