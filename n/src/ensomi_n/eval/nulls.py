"""Nulls of BL-τ with fixed expected values, and a check that the harness meets them.

With no fitted offset, a prediction at the right tempo but a random phase
passes a reference line exactly when its own line falls within τ of it. So
the chance of BL-τ is ``2τ / B`` at canonical beat length B, both under one
uniform shift of the whole grid and under an independent phase per beat
(lines stay ordered, since each moves less than half a beat, so only a line's
own counterpart can fall within τ < B / 2). At τ = 20 ms that is 0.053 at
80 BPM and 0.107 at the 160 BPM limit, under the 0.15 bound the design sets
for a deciding statistic.

`check` runs the real scorer on synthetic grids and compares: the reference
against itself (1), shifts of half and a third of a beat (0), the mean over
equally spaced whole-grid shifts and one draw of independent per-beat phases
(each within 0.01 of ``2τ / B``).
"""
from __future__ import annotations

import numpy as np

from ..mapformat import MusicMap
from ..timing import Reference, Segment
from .measures import TAUS, score

CANONICAL_TEMPI = (80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0)
TOLERANCE = 0.01


def chance(bpm: float, tau_ms: float) -> float:
    """BL-τ chance at random phase for a canonical tempo: ``min(1, 2τ / B)``."""
    return min(1.0, 2.0 * tau_ms * bpm / 60000.0)


def table(tempi=CANONICAL_TEMPI, taus=TAUS) -> list[dict]:
    """The exact chance per canonical tempo. 160 BPM is the open upper limit of the canonical range."""
    return [dict(bpm=float(b), beat_ms=60000.0 / b, **{f'bl{t:g}': chance(b, t) for t in taus}) for b in tempi]


def _grid(bpm: float, n_beats: int) -> Reference:
    step = 60000.0 / bpm
    heads = np.arange(n_beats) * step
    return Reference.from_segments([Segment(0.0, step, 4)], heads, key=f'null-{bpm:g}')


def check(bpm: float, *, phases: int = 1000, n_beats: int = 64, phase_beats: int = 20000,
          seed: int = 0) -> dict:
    """Run every null at one canonical tempo through `score`; returns expected and observed values.

    Whole-grid shifts are ``(k + 0.5) / phases`` of a beat for k < phases,
    so their mean is within ``1 / phases`` of the exact chance. Per-beat
    phases are drawn once from ``U[-B/2, B/2)`` with ``seed``; the prediction
    declares the reference's beat length as its segment so the drawn beats
    keep their fold.
    """
    step = 60000.0 / bpm
    ref = _grid(bpm, n_beats)
    out: dict = dict(bpm=bpm, beat_ms=step, expected={f'bl{t:g}': chance(bpm, t) for t in TAUS})
    fixed = {}
    for name, shift in (('self', 0.0), ('half_beat', step / 2), ('third_beat', step / 3)):
        s = score(ref, MusicMap(segments=(Segment(shift, step, 4),)))
        fixed[name] = {f'bl{t:g}': s.bl[t] for t in TAUS}
    out['fixed'] = fixed
    sums = {t: 0.0 for t in TAUS}
    for k in range(phases):
        s = score(ref, MusicMap(segments=(Segment((k + 0.5) / phases * step, step, 4),)))
        for t in TAUS:
            sums[t] += s.bl[t]
    out['whole_shift'] = {f'bl{t:g}': sums[t] / phases for t in TAUS}
    rng = np.random.default_rng(seed)
    big = _grid(bpm, phase_beats)
    beats = np.arange(phase_beats) * step + rng.uniform(-step / 2, step / 2, phase_beats)
    s = score(big, MusicMap(beats=beats, segments=(Segment(0.0, step, 4),)))
    out['beat_phase'] = {f'bl{t:g}': s.bl[t] for t in TAUS}
    exp = out['expected']
    out['pass'] = bool(
        all(fixed['self'][k] == 1.0 and fixed['half_beat'][k] == 0.0 and fixed['third_beat'][k] == 0.0
            for k in exp)
        and all(abs(out['whole_shift'][k] - exp[k]) <= TOLERANCE for k in exp)
        and all(abs(out['beat_phase'][k] - exp[k]) <= TOLERANCE for k in exp))
    return out


def check_all(tempi=(80.0, 100.0, 120.0, 140.0, 159.0), **kwargs) -> list[dict]:
    return [check(b, **kwargs) for b in tempi]
