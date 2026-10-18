"""Interim hard-strata flags from a reference's red lines and heads (design §5.1, R0).

These stand in for `ensomi_n.eval` strata until a census exists, and every
output that uses them says ``interim``. They are recording-level flags: a
recording is in a stratum when the flag holds anywhere in its mapped range.

Musical segments are the reference's red lines with redundant ones merged: a
line at the previous segment's beat length whose start is on that segment's
beat grid (within ``MERGE_MS``) and whose meter is the same continues it.

Flags (design §5.1 list, then the eval design's S5 and S9 definitions):

- ``multi_tempo``: two musical segments in range whose canonical tempi differ by more than 0.5 %;
- ``reset``: a same-tempo line off the previous segment's beat grid;
- ``short``: a musical segment in range shorter than 8 s (clipped to the mapped range);
- ``meter``: a musical segment in range with meter other than 4, or meters that change;
- ``tempo_mapped``: 0.8 or more of four or more musical lines in range open sections under 8 s;
- ``mixed``: 1 % or more of the notated beats in range are mixed (3 and 4 divide the divisor);
- ``level``: dominant canonical tempo within 3 % of 80 or 160 BPM, or consecutive musical
  segments at a ratio of 2, 1/2, 3/2, 2/3, 4/3, 3/4, 5/4 or 4/5 (within 0.5 %);
- ``glitch``: a reset in a one-tempo recording, an implausible red line holding heads, or a
  segment shorter than one canonical beat holding heads.

Strata: S1 mixed, S2 meter, S3 tempo_mapped, S4 short or reset outside S3, S5 level,
S9 glitch, S0 none of the flags (so meter 4, one tempo, no reset, not near 80/160).
"""
from __future__ import annotations

import math

import numpy as np

from ..timing import MERGE_MS, canonical_beat_ms

SHORT_MS = 8000.0
SAME_TEMPO = 0.005
RATIOS = (2.0, 0.5, 1.5, 2 / 3, 4 / 3, 0.75, 1.25, 0.8)
HARD = ('multi_tempo', 'reset', 'short', 'meter', 'tempo_mapped', 'mixed', 'level', 'glitch')
STRATA = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S9')


def musical_segments(segments) -> list[dict]:
    """Plausible segments with redundant lines merged; each dict has start, beat, meter, reset."""
    out: list[dict] = []
    for s in segments:
        if not s.plausible:
            out.append(dict(start=s.start_ms, beat=math.nan, meter=s.meter, reset=False, plausible=False))
            continue
        if out and out[-1]['plausible']:
            prev = out[-1]
            same = abs(s.beat_ms / prev['beat'] - 1.0) <= 1e-6
            if same:
                k = (s.start_ms - prev['start']) / prev['beat']
                on_grid = abs(k - round(k)) * prev['beat'] <= MERGE_MS
                if on_grid and s.meter == prev['meter']:
                    continue                        # redundant: the same timing continues
                out.append(dict(start=s.start_ms, beat=s.beat_ms, meter=s.meter, reset=not on_grid, plausible=True))
                continue
        out.append(dict(start=s.start_ms, beat=s.beat_ms, meter=s.meter, reset=False, plausible=True))
    return out


def flags(ref) -> dict:
    """The interim flags and strata of one reference (`ensomi_n.timing.Reference`)."""
    from ..eval.measures import subdivision_classes
    lo, hi = ref.lo_ms, ref.hi_ms
    segs = musical_segments(ref.segments or ())
    heads = ref.heads if ref.heads is not None else np.zeros(0)
    ends = [segs[k + 1]['start'] if k + 1 < len(segs) else math.inf for k in range(len(segs))]
    inr = [k for k in range(len(segs)) if ends[k] > lo and segs[k]['start'] <= hi]
    plaus = [k for k in inr if segs[k]['plausible']]
    canon = {k: float(canonical_beat_ms(segs[k]['beat'])) for k in plaus}
    dur = {k: min(ends[k], hi) - max(segs[k]['start'], lo) for k in inr}

    def holds_heads(k) -> bool:
        a, b = segs[k]['start'], ends[k]
        return bool(np.any((heads >= a - MERGE_MS) & (heads < b - MERGE_MS)))

    f = {}
    tempi = [canon[k] for k in plaus]
    f['multi_tempo'] = bool(tempi) and (max(tempi) / min(tempi) - 1.0 > SAME_TEMPO)
    f['reset'] = any(segs[k]['reset'] for k in plaus)
    f['short'] = any(dur[k] < SHORT_MS for k in plaus) and len(plaus) > 1
    meters = {segs[k]['meter'] for k in plaus}
    f['meter'] = bool(meters) and (meters != {4})
    opened = [k for k in plaus if k > 0 or segs[k]['start'] >= lo]
    f['tempo_mapped'] = len(plaus) >= 4 and np.mean([dur[k] < SHORT_MS for k in plaus]) >= 0.8
    b = ref.timing.beats
    in_beats = int(np.sum((b >= lo - MERGE_MS) & (b <= hi + MERGE_MS)))
    _, _, cls = subdivision_classes(ref.timing, heads)
    n_mixed = int(np.sum(cls == 'mixed'))
    f['mixed'] = in_beats > 0 and n_mixed / in_beats >= 0.01
    level = False
    if plaus:
        weights = np.array([dur[k] for k in plaus])
        dom = canon[plaus[int(np.argmax(weights))]]
        bpm = 60000.0 / dom
        level = abs(bpm / 80.0 - 1.0) <= 0.03 or abs(bpm / 160.0 - 1.0) <= 0.03
        for x, y in zip(plaus, plaus[1:]):
            r = segs[x]['beat'] / segs[y]['beat']
            if any(abs(r / q - 1.0) <= SAME_TEMPO for q in RATIOS):
                level = True
    f['level'] = bool(level)
    implausible_heads = any(holds_heads(k) for k in inr if not segs[k]['plausible'])
    tiny = any(dur[k] < canon[k] and holds_heads(k) for k in plaus if k != plaus[-1])
    f['glitch'] = bool((f['reset'] and not f['multi_tempo']) or implausible_heads or tiny)
    f = {k: bool(v) for k, v in f.items()}
    strata = []
    if f['mixed']:
        strata.append('S1')
    if f['meter']:
        strata.append('S2')
    if f['tempo_mapped']:
        strata.append('S3')
    if (f['short'] or f['reset']) and not f['tempo_mapped']:
        strata.append('S4')
    if f['level']:
        strata.append('S5')
    if f['glitch']:
        strata.append('S9')
    if not any(f[k] for k in HARD):
        strata.append('S0')
    return dict(flags=f, strata=strata, hard=any(f[k] for k in HARD), n_musical=len(plaus),
                n_red=len(ref.segments or ()), mixed_share=round(n_mixed / in_beats, 5) if in_beats else 0.0,
                n_opened=len(opened))
