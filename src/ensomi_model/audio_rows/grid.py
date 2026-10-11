"""Chart-free corrections of the fitter's segments, giving the one grid heads and R2 share.

In order:

1. Lag: every offset moves 30 ms earlier. Uncorrected, the fitted grid sits a median 30.0 ms
   later than the chart's red lines (4,667 tempo-matched fit_train charts without phase jumps).
2. V2: when every segment's tempo is octave-equivalent to the longest one's within 0.3 %, only the
   longest segment is kept. Its notated BPM is rounded when within 0.25 of an integer, keeping
   the beat nearest the segment's middle in place.
3. Boundary repair: a segment within 0.3 % of 80 or 160 canonical BPM becomes exactly 80 BPM, so
   one tempo cannot fold to both ends of [80, 160) and flip R2's ``bpm / 120`` input.

All segments get meter 4; the fitter has no meter.
"""
from __future__ import annotations

import numpy as np

from ..evaluation.beats import canonical_beat_length
from ..r2.common import GridArrays, grid_from_arrays

LAG_MS = 30.0
TEMPO_TOLERANCE = 0.003
BPM_ROUNDING = 0.25


def corrected_grid(segments, song_ms: float, lag_ms: float = LAG_MS):
    """``(GridArrays, seg [S, 3])`` from the fitter's (offset_ms, beat_length_ms) segments.

    ``seg`` rows are (offset_ms, canonical beat length ms, meter), the red lines for export.
    """
    s = np.asarray(segments, dtype=np.float64)
    s = s[np.argsort(s[:, 0])]
    offsets, lengths = s[:, 0] - lag_ms, s[:, 1]
    ends = np.r_[offsets[1:], max(song_ms, offsets[-1] + 1)]
    longest = int(np.argmax(ends - offsets))
    canonical = np.array([canonical_beat_length(x) for x in lengths])
    # Compare octave-equivalent tempos before folding, which can put one tempo at both 80 and 160.
    aligned = canonical * 2.0 ** np.round(np.log2(canonical[longest] / canonical))
    if np.all(np.abs(aligned / aligned[longest] - 1) < TEMPO_TOLERANCE):
        offset, length = offsets[longest], lengths[longest]
        bpm = 60000.0 / length
        if abs(bpm - round(bpm)) <= BPM_ROUNDING:
            new_length = 60000.0 / round(bpm)
            offset += round((ends[longest] - offset) / (2 * length)) * (length - new_length)
            length = new_length
        offsets, lengths = np.array([offset]), np.array([length])
    canonical = np.array([canonical_beat_length(x) for x in lengths])
    bpm = 60000.0 / canonical
    canonical[(np.abs(bpm / 80 - 1) < TEMPO_TOLERANCE) | (np.abs(bpm / 160 - 1) < TEMPO_TOLERANCE)] = 750.0
    seg = np.column_stack((offsets, canonical, np.full(len(offsets), 4.0)))
    return GridArrays.from_grid(grid_from_arrays(seg, seg[:, [0, 2]])), seg
