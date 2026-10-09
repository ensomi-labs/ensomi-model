"""Each segment's phase from the Mel onsets the mappers' beats follow.

BeatThis's sub-frame peaks give the grid its tempo and segments, but their lag behind the music
changes with the instrumentation, so a grid fitted through them carries a per-song phase error
(9 ms robust SD on single-tempo fit_train charts). The onsets of the Listener's own log-Mel sit
much closer to the mappers' beats: within a chart, 2.7 ms robust SD against 6.5-6.8 ms for the
peaks, and the mappers sit a near-constant 15 ms before them.

``mel_phase`` keeps every segment's tempo and moves its offset so that the median onset over its
beats sits ``MAPPER_OFFSET_MS`` after the beat on the chart's clock (``grid.corrected_grid``'s 30 ms
lag applied). The onset of a beat is the largest local maximum of the full-band onset envelope
(``onset_envelope``) within ``ONSET_WINDOW_MS`` of the beat plus ``ONSET_CENTRE_MS``, placed by a
parabola through the envelope's three samples. Beats count from the first to the last BeatThis
peak. A segment with fewer than ``MIN_SEGMENT_ONSETS`` onsets takes the song's median, and a song
with fewer than ``MIN_SONG_ONSETS`` keeps its grid. When ``corrected_grid`` keeps one segment of
several (one tempo), every segment moves by that segment's shift, so the corrected grid moves as
one line.

Chosen on fit_train among 1,440 settings (onset band, search centre and window, estimator, slope,
song or segment scope) by five-fold cross-fitting, S1-20 first, then H-20, then S1-10; the offset
is the population-weighted median over fit_train charts of the committed grid's median head
residual minus the onset median. On fit_dev, against the committed grid on the same BeatThis
``final0`` logits (806 charts, 322 songs; song-cluster bootstrap):

- charts with 90 % of heads within 20 ms of the 1/4-1/6-1/8 lattice: 0.808 to 0.846, +0.038
  [+0.013, +0.065]; heads within 20 ms 0.920 to 0.943, +0.023 [+0.012, +0.038];
- single-tempo charts S1-20 +0.029 [+0.010, +0.051], multi-tempo +0.075 [-0.024, +0.180];
- each chart's own median head residual added to the committed grid reaches 0.846 and 0.945, so at
  20 ms this rule recovers the per-song phase error; at 10 ms that oracle holds 0.721 of charts
  against this rule's 0.597.

It gained at each of the BeatThis checkpoints ``final0``, ``final1`` and ``final2`` on a 92-song
panel and 40 multi-tempo songs, moved no bar phase or tempo, and added no whole-grid flip under
sub-frame shifts. On a holdout sealed before the rule was built and looked at once (289 fit_train
songs no earlier round had used, 519 charts), against the committed grid on the same logits: S1-20
0.784 to 0.840, +0.056 [+0.025, +0.090]; heads within 20 ms 0.897 to 0.928, +0.031 [+0.015,
+0.052]; S1-10 +0.125 [+0.076, +0.176]. Details:
``~/ensomi/.sync/cp/scratch/timing/opt/report-opt.md`` (2026-10-09).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..grid import LAG_MS, corrected_grid
from .peaks import beat_peaks
from .types import TimingSegment

ONSET_CENTRE_MS = 16.0      # the onset search is centred this far after the beat ...
ONSET_WINDOW_MS = 60.0      # ... and reaches this far on each side
MAPPER_OFFSET_MS = -14.873  # the beat sits this far from the median onset (set on fit_train)
MIN_SEGMENT_ONSETS = 32
MIN_SONG_ONSETS = 8
ENVELOPE_HOP_MS = 10.0
ENVELOPE_FIRST_MS = 15.0    # envelope sample k lies between the centres of Mel frames k - 1 and k


def onset_envelope(log_mel) -> np.ndarray:
    """[F] summed positive log-Mel differences between consecutive frames; sample 0 is 0."""
    log_mel = np.asarray(log_mel, dtype=np.float64)
    return np.r_[0.0, np.maximum(np.diff(log_mel, axis=0), 0.0).sum(axis=1)]


def onset_times(envelope, times_ms, window_ms: float = ONSET_WINDOW_MS) -> np.ndarray:
    """The largest local maximum of the envelope within ``window_ms`` of each time, NaN where none."""
    env = np.asarray(envelope, dtype=np.float64)
    times_ms = np.asarray(times_ms, dtype=np.float64)
    out = np.full(times_ms.shape[0], np.nan)
    peaks = np.flatnonzero(np.r_[False, (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]), False])
    if peaks.shape[0] == 0:
        return out
    at = peaks * ENVELOPE_HOP_MS + ENVELOPE_FIRST_MS
    lo = np.searchsorted(at, times_ms - window_ms)
    hi = np.searchsorted(at, times_ms + window_ms)
    best = np.full(times_ms.shape[0], -1)
    value = np.full(times_ms.shape[0], -np.inf)
    for k in range(int((hi - lo).max(initial=0))):
        j = np.minimum(lo + k, peaks.shape[0] - 1)
        v = np.where(lo + k < hi, env[peaks[j]], -np.inf)
        better = v > value
        best, value = np.where(better, j, best), np.where(better, v, value)
    found = best >= 0
    i = peaks[best[found]]
    y0, y1, y2 = env[i - 1], env[i], env[np.minimum(i + 1, env.shape[0] - 1)]
    curvature = y0 - 2.0 * y1 + y2
    safe = np.abs(curvature) > 1e-12
    delta = np.where(safe, np.clip(0.5 * (y0 - y2) / np.where(safe, curvature, 1.0), -0.5, 0.5), 0.0)
    out[found] = i * ENVELOPE_HOP_MS + ENVELOPE_FIRST_MS + ENVELOPE_HOP_MS * delta
    return out


def _beats(grid, lo: float, hi: float):
    """Beat times of each segment of a sorted (offset, length) grid inside [lo, hi], and their segment."""
    offsets, lengths = grid[:, 0], grid[:, 1]
    last_segment = grid.shape[0] - 1
    ends = np.r_[offsets[1:], max(hi, offsets[-1]) + lengths[-1]]
    times, index = [], []
    for j in range(grid.shape[0]):
        first = int(np.floor(min(lo - offsets[0], 0.0) / lengths[0])) if j == 0 else 0
        if j < last_segment:     # lines strictly before the next segment's offset
            last = int(np.ceil((ends[j] - offsets[j]) / lengths[j] - 1e-9)) - 1
        else:
            last = int(np.ceil((ends[j] - offsets[j]) / lengths[j]))
        n = np.arange(first, last + 1)
        times.append(offsets[j] + n * lengths[j])
        index.append(np.full(n.shape[0], j))
    times, index = np.concatenate(times), np.concatenate(index)
    keep = (times >= lo) & (times <= hi)
    return times[keep], index[keep]


def mel_phase(segments: Sequence[TimingSegment], log_mel, beat_logits, frame_rate_hz: float,
              song_ms: float) -> tuple[TimingSegment, ...]:
    """The segments with each offset moved to the Mel onsets of its beats; tempos unchanged.

    ``log_mel`` is the Listener's [F, bins] log-Mel (frame i covers [10 i, 10 i + 40) ms) on the
    same clock as the segments and ``beat_logits``.
    """
    segments = tuple(segments)
    if not segments:
        return segments
    times, probability = beat_peaks(beat_logits, frame_rate_hz)
    times = times[probability > 0.5]
    if times.shape[0] == 0:
        return segments
    raw = np.array([(s.offset_ms, s.beat_length_ms) for s in segments], dtype=np.float64)
    order = np.argsort(raw[:, 0], kind='stable')
    grid = corrected_grid(raw, song_ms)[1][:, :2]              # the chart's clock: LAG_MS earlier
    beats, index = _beats(grid, times[0] - LAG_MS, times[-1] - LAG_MS)
    offset = onset_times(onset_envelope(log_mel), beats + ONSET_CENTRE_MS) - beats
    found = np.isfinite(offset)
    if found.sum() < MIN_SONG_ONSETS:
        return segments
    song = float(np.median(offset[found]))
    shift = np.array([float(np.median(offset[found & (index == j)]))
                      if (found & (index == j)).sum() >= MIN_SEGMENT_ONSETS else song
                      for j in range(grid.shape[0])]) + MAPPER_OFFSET_MS
    if grid.shape[0] != raw.shape[0]:                           # one tempo: corrected_grid kept one line
        shift = np.full(raw.shape[0], shift[0])
    out = list(segments)
    for j, k in enumerate(order):
        out[k] = TimingSegment(offset_ms=segments[k].offset_ms + float(shift[j]), beat_length_ms=segments[k].beat_length_ms)
    return tuple(out)
