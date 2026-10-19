"""Hard-timing strata from labels alone (N1 evaluation design §2, research notes ``n1-eval-20261018``).

A stratum names spans of a beatmap set: the measures of `ensomi_n.eval.score`
run on those spans (``spans=``), and on the whole set where a stratum holds
whole recordings (spans ``None``). `set_strata` computes one set's strata from
its red lines (N's reader: resets, notated bars and detours kept) and its heads
(the union over the set's difficulties); `recording_strata` combines the sets of
one recording, the decision unit. Nothing here opens audio.

S10 (music where BeatThis fails) needs BeatThis logits, so it is not computed
here (`NEEDS_LOGITS`).

**Red lines.** Every red line gets a kind, judged against the grid in force
(the last line that is not a detour):

- ``implausible``: a tempo outside [20, 1000] BPM; it places no beats.
- ``first``: the first plausible line.
- ``same``: the notated tempo of the grid in force within 0.5 %, on its
  notated beat grid within 2 ms, and drifting from it by 2 ms or less over its
  own segment. It may restart the bar.
- ``reset``: the same notated tempo (0.5 %), off the grid by more than 2 ms;
  this is E0's reset (`measures._resets`).
- ``detour``: a run of lines at another tempo after which a line returns to
  the grid in force (its tempo within 0.5 %, on its grid within 2 ms), when
  every head inside the run lies on a binary subdivision (1/1 to 1/16) of the
  grid's canonical beat, or the run lasts no more than one bar of the grid. A
  run whose lines are all an octave of the grid in phase is a level change,
  not a detour.
- ``tempo``: every other change.

Musical lines are ``first``, ``tempo`` and ``reset``. A section runs from a
``first`` or ``tempo`` line to the next one, the last ending at the last
head; resets do not open sections (reading: a 7/8 bar timed with a reset is
not a tempo map), and count for S4 and S9 instead.

**Bars.** Bars start at every red line (osu! restarts the bar there) and
repeat every ``meter`` notated beats, as `ensomi_n.timing` places them. A bar
is counted when both its ends lie between the first and last head and it
touches no detour run and no implausible span. Its length is in notated beats
of its own segment, rounded to 1/4 beat. A bar ending at, or holding, a
``tempo`` line short of its meter is cut.

**Strata** (constants below; deviations from §2 are marked "reading"):

- S0 control: in every set, one musical line, every meter 3 or 4, no reset, no
  detour run or implausible line holding heads, no mixed notated beat, no
  voted timing tag, a canonical tempo more than 3 % from 80 and 160, and no
  rate edit (reading: version or audio names such as "1.1x").
- S1 mixed subdivision: 1 % or more of the mapped notated beats (beats holding
  a head) are mixed; a beat is classed only with two or more heads, as in §3.4
  (reading). Spans: the bars holding a mixed beat.
- S2 irregular and changing meter: a counted bar not cut short with a length
  outside {2, 3, 4, 6, 8, 12} (irregular); or two consecutive counted bars of
  different lengths, the first bar of the set not counted, cut bars included
  (changing; this is what "two bar lengths within 8 consecutive bars" reduces
  to). Spans: the irregular bars and both bars of each change.
- S3 tempo-mapped and performed: 0.8 or more of a set's sections are under
  8 s, with at least 8 sections (reading: a minimum, so a short intro of
  tempo changes does not count), or the voted tag meta/variable timing (whole
  set); ramps: 4 or more consecutive musical lines each moving the tempo the
  same way by under 5 % (4 steps), or the voted tag meta/accelerating bpm.
  Spans: whole set, or each ramp from its first line to 8 s past its last
  (or the next musical line).
- S4 short sections and resets: a section under 8 s or under 16 canonical
  beats, or a reset, in a recording outside S3. Spans: the
  short sections, and the bars before and after each reset.
- S5 metrical-level ambiguity: musical sections at a canonical tempo within
  3 % of 80 or 160, 8 s or more in all (reading: a minimum length); a section
  at 2 or 1/2 of the previous section's notated tempo starting on its grid; a
  section whose canonical tempo is 3/2, 2/3, 4/3, 3/4, 5/4 or 4/5 of the
  previous one's; or sets of one recording whose longest sections are
  notated at different octaves (whole recording).
- S6 sparse and beatless: gaps of 4 s or more between heads; windows of 8
  canonical beats with fewer than 4 heads (reading: canonical beats, so the
  notated level does not change the rate); and the 4 bars after either.
- S7 swing: the voted tag meta/swing, or 4 or more counted bars in which, of
  the heads at 1/3, 1/2 or 2/3 of a beat, at least 2 and at least 50 % sit at
  2/3, and under 10 % at 1/2 and under 10 % at 1/3 (reading: the 1/3 clause
  keeps triplet runs out). The beat is the notated beat, and also two notated
  beats where the notated tempo is 160 BPM or more (reading).
- S8 polyrhythm: runs of 6 or more consecutive heads of one difficulty at a
  constant spacing of 2/3, 3/4, 4/3 or 3/2 canonical beats (reading:
  canonical, so double-time notation does not turn triplet streams into
  hemiolas) within one tempo, crossing a bar line (reading of "spanning 2
  bars").
- S9 glitch, stutter and time-stretch: a detour run or an implausible line
  holding heads; a reset in a set with no tempo change; heads off every
  divisor up to 1/16 of the notated beat; a segment shorter than one
  canonical beat holding heads.

Rate edits are not folded here; the recording table decides which sets form
one recording.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Sequence

import numpy as np

from ..timing import MERGE_MS, Reference, Segment, Timing, canonical_beat_ms
from .measures import DIVISORS, LEVEL_FAMILY, SAME_TEMPO, subdivision_classes

__all__ = ['STRATA', 'HARD', 'NEEDS_LOGITS', 'NAMES', 'TIMING_TAGS', 'Line', 'SetStrata', 'classify_lines',
           'set_strata', 'recording_strata', 'flags', 'is_rate_edit', 'same_timing']

STRATA = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9')
HARD = STRATA[1:]
NEEDS_LOGITS = ('S10',)
NAMES = {
    'S0': 'control (easy)', 'S1': 'mixed subdivision', 'S2': 'irregular and changing meter',
    'S3': 'tempo-mapped and performed', 'S4': 'short sections and resets', 'S5': 'metrical-level ambiguity',
    'S6': 'sparse and beatless passages', 'S7': 'swing', 'S8': 'polyrhythm and polymeter',
    'S9': 'glitch, stutter and time-stretch', 'S10': 'music where BeatThis fails (needs logits; not computed)',
}
TIMING_TAGS = ('meta/swing', 'meta/accelerating bpm', 'meta/variable timing', 'tech/complex snap')

MUSICAL_KINDS = ('first', 'tempo', 'reset')
LEVEL_NEAR = 0.03
LEVEL_NEAR_MIN_MS = 8000.0
MIXED_SHARE = 0.01
REGULAR_BAR_BEATS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
BAR_QUANTUM = 0.25
SHORT_SECTION_MS = 8000.0
SHORT_SECTION_BEATS = 16.0
TEMPO_MAPPED_SHARE = 0.8
TEMPO_MAPPED_MIN_LINES = 8
RAMP_STEPS = 4
RAMP_MAX_STEP = 0.05
RAMP_MIN_STEP = 1e-4
GAP_MS = 4000.0
SPARSE_WINDOW_BEATS = 8
SPARSE_MIN_HEADS = 4
REENTRY_BARS = 4
SWING_MIN_BARS = 4
SWING_MIN_HEADS = 2
SWING_TWO_THIRDS = 0.5
SWING_HALF = 0.1
SWING_THIRD = 0.1
SWING_DOUBLE_BPM = 160.0
POLY_SPACINGS = (2 / 3, 3 / 4, 4 / 3, 3 / 2)
POLY_MIN_HEADS = 6
BINARY = (1, 2, 4, 8, 16)
DETOUR_MAX_RUN = 256
OCTAVES = (-3, -2, -1, 1, 2, 3)

Spans = list  # list of (start_ms, end_ms); None in a membership means the whole set


@dataclass(frozen=True)
class Line:
    """A red line with its kind (module docstring) and the end of its segment (the next line, any kind)."""
    start_ms: float
    beat_ms: float
    meter: int
    bar_at_start: bool
    kind: str
    end_ms: float

    @property
    def canonical_ms(self) -> float:
        return float(canonical_beat_ms(self.beat_ms)) if 0 < self.beat_ms < math.inf else math.nan


@dataclass
class SetStrata:
    """One set's strata.

    ``members`` maps each stratum the set is in to its spans (None: the whole
    set). ``detail`` holds, per stratum, the counts behind the decision.
    ``span_heads`` counts the set's heads inside each member's spans.
    """
    key: str
    lo_ms: float
    hi_ms: float
    heads: int
    lines: list[Line]
    members: dict[str, Spans | None]
    detail: dict[str, dict]
    span_heads: dict[str, int]
    s0: bool
    s3: bool
    s4_spans: Spans
    s4_heads: int
    dominant_bpm: float
    timing: tuple = field(default=(), repr=False)

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for line in self.lines:
            kinds[line.kind] = kinds.get(line.kind, 0) + 1
        return dict(key=self.key, heads=self.heads, lo_ms=self.lo_ms, hi_ms=self.hi_ms, kinds=kinds,
                    strata=sorted(self.members), dominant_bpm=self.dominant_bpm)


# ---------------------------------------------------------------- helpers

def _grid_error(start: float, beat: float, t) -> np.ndarray:
    pos = (np.asarray(t, dtype=np.float64) - start) / beat
    return np.abs(pos - np.round(pos)) * beat


def _on_divisors(x: np.ndarray, step: np.ndarray, divisors: Iterable[int]) -> np.ndarray:
    """Whether each position ``x`` (in beats of length ``step``) is within ``MERGE_MS`` of some ``k/d``."""
    ok = np.zeros(x.shape, dtype=bool)
    for d in divisors:
        ok |= np.abs(x - np.round(x * d) / d) * step <= MERGE_MS + 1e-9
    return ok


def _count_in(times: np.ndarray, a: float, b: float) -> int:
    """Number of sorted ``times`` in ``[a, b)``."""
    return int(np.searchsorted(times, b, side='left') - np.searchsorted(times, a, side='left'))


def _merge(spans: Iterable[tuple[float, float]]) -> Spans:
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _clip(spans: Iterable[tuple[float, float]], lo: float, hi: float) -> Spans:
    return _merge((max(a, lo), min(b, hi)) for a, b in spans)


def _heads_in(heads: np.ndarray, spans: Spans | None) -> int:
    """Heads inside the spans, ends included; all heads for None. Spans are merged, so none counts twice."""
    if spans is None:
        return int(len(heads))
    return int(sum(np.searchsorted(heads, b, side='right') - np.searchsorted(heads, a, side='left')
                   for a, b in spans))


_RATE = re.compile(r'(?<![\w.])(?:x\s*(\d(?:\.\d{1,3})?)(?![\d.])|(\d(?:\.\d{1,3})?)\s*x(?![a-z0-9]))', re.I)


def is_rate_edit(*names: str) -> bool:
    """Whether a difficulty or audio name marks a rate edit ("1.1x", "x1.2", "0.9x"), not 1.0x."""
    for name in names:
        for m in _RATE.finditer(name or ''):
            v = float(m.group(1) or m.group(2))
            if 0.5 <= v <= 2.0 and abs(v - 1.0) > 1e-9:
                return True
    return False


def same_timing(a: Sequence[Segment], b: Sequence[Segment], tol_ms: float = 1.0) -> bool:
    """Whether two sets' plausible red lines are one timing (a copy): same count, offsets within
    ``tol_ms``, beat lengths drifting apart by under ``tol_ms`` over 1,000 beats, same meters."""
    pa = [s for s in a if s.plausible]
    pb = [s for s in b if s.plausible]
    return len(pa) == len(pb) and all(
        abs(x.start_ms - y.start_ms) <= tol_ms and abs(x.beat_ms - y.beat_ms) * 1000.0 <= tol_ms
        and x.meter == y.meter for x, y in zip(pa, pb))


# ---------------------------------------------------------------- red lines

def classify_lines(segments: Sequence[Segment], heads, lo_ms: float, hi_ms: float) -> list[Line]:
    """The kind of every red line (module docstring), in offset order."""
    segs = list(segments)
    n = len(segs)
    h = np.asarray(heads if heads is not None else [], dtype=np.float64)
    ends = [segs[i + 1].start_ms if i + 1 < n else math.inf for i in range(n)]
    kinds = ['implausible' if not s.plausible else '' for s in segs]
    P = [i for i in range(n) if segs[i].plausible]
    if P:
        starts = np.array([segs[i].start_ms for i in P])
        beats = np.array([segs[i].beat_ms for i in P])
        pend = np.array([ends[i] for i in P])
        kinds[P[0]] = 'first'

        def seg_beats(k: int) -> float:
            a, b = starts[k], min(pend[k], max(hi_ms, starts[k]))
            return max(b - a, 0.0) / beats[k]

        def continuation(k: int, g: int) -> str:
            """Kind of line k at the grid g's tempo: same, reset, or tempo when it drifts."""
            if _grid_error(starts[g], beats[g], starts[k]) > MERGE_MS + 1e-9:
                return 'reset'
            return 'same' if abs(beats[k] - beats[g]) * seg_beats(k) <= MERGE_MS + 1e-9 else 'tempo'

        def detour_end(k: int, g: int) -> int | None:
            hi_j = min(len(P), k + 1 + DETOUR_MAX_RUN)
            if hi_j <= k + 1:
                return None
            cand = np.arange(k + 1, hi_j)
            same = np.abs(beats[cand] / beats[g] - 1.0) <= SAME_TEMPO
            on = _grid_error(starts[g], beats[g], starts[cand]) <= MERGE_MS + 1e-9
            hit = np.flatnonzero(same & on)
            if not len(hit):
                return None
            j = int(cand[hit[0]])
            run = range(k, j)
            octave = all(
                any(abs(beats[g] / beats[m] / 2.0 ** e - 1.0) <= SAME_TEMPO for e in OCTAVES)
                and _grid_error(starts[g], min(beats[g], beats[m]), starts[m]) <= MERGE_MS + 1e-9 for m in run)
            if octave:
                return None
            a, b = starts[k], starts[j]
            inside = h[(h >= a) & (h < b)]
            c = float(canonical_beat_ms(beats[g]))
            binary = bool(np.all(_on_divisors((inside - starts[g]) / c, np.full(len(inside), c), BINARY)))
            short = b - a <= max(segs[P[g]].meter, 1) * beats[g] + MERGE_MS
            return j if (binary or short) else None

        g, k = 0, 1
        while k < len(P):
            if abs(beats[k] / beats[g] - 1.0) <= SAME_TEMPO:
                kinds[P[k]] = continuation(k, g)
                g, k = k, k + 1
                continue
            j = detour_end(k, g)
            if j is not None:
                for m in range(k, j):
                    kinds[P[m]] = 'detour'
                kinds[P[j]] = continuation(j, g)
                g, k = j, j + 1
                continue
            kinds[P[k]] = 'tempo'
            g, k = k, k + 1
    return [Line(s.start_ms, s.beat_ms, int(s.meter), bool(s.bar_at_start), kind, end)
            for s, kind, end in zip(segs, kinds, ends)]


def _sections(lines: Sequence[Line], lo: float, hi: float) -> list[tuple[Line, float, float]]:
    """First and tempo lines with their sections ``[start, end)``; the last ends at ``hi``, the first may start
    at ``lo`` when the first head precedes it. Sections outside ``[lo, hi]`` are dropped."""
    mus = [line for line in lines if line.kind in ('first', 'tempo')]
    out = []
    for i, line in enumerate(mus):
        start = min(line.start_ms, lo) if i == 0 else line.start_ms
        end = mus[i + 1].start_ms if i + 1 < len(mus) else max(hi, line.start_ms)
        end = min(end, hi) if i + 1 == len(mus) else end
        if end > lo and start < hi and end > start:
            out.append((line, start, end))
    return out


# ---------------------------------------------------------------- bars

@dataclass
class _Bars:
    start: np.ndarray
    end: np.ndarray
    length: np.ndarray      # notated beats, rounded to BAR_QUANTUM
    counted: np.ndarray     # both ends in range, no detour or implausible span touched
    cut: np.ndarray         # ends at or holds a tempo line short of its meter
    meter: np.ndarray
    first_beat: np.ndarray  # index of the bar's first notated beat


def _bars(timing: Timing, lines: Sequence[Line], lo: float, hi: float, excluded: Spans) -> _Bars:
    idx = np.flatnonzero(timing.downbeat) if timing.has_bars else np.zeros(0, dtype=np.int64)
    if len(idx) < 2:
        z = np.zeros(0)
        return _Bars(z, z, z, z.astype(bool), z.astype(bool), z, z.astype(np.int64))
    start = timing.beats[idx[:-1]]
    end = timing.beats[idx[1:]]
    step = timing.step_ms[idx[:-1]]
    raw = (end - start) / step
    length = np.round(raw / BAR_QUANTUM) * BAR_QUANTUM
    plausible = [line for line in lines if line.kind != 'implausible']
    starts = np.array([line.start_ms for line in plausible])
    meters_at = np.array([line.meter for line in plausible])
    which = np.clip(np.searchsorted(starts, start + MERGE_MS, side='right') - 1, 0, max(len(starts) - 1, 0))
    meter = meters_at[which].astype(np.float64) if len(starts) else np.full(len(start), 4.0)
    tempo_t = np.array([line.start_ms for line in lines if line.kind == 'tempo'])
    cut = np.zeros(len(start), dtype=bool)
    if len(tempo_t):
        # A tempo line at the bar's end, or strictly inside it, while the bar is short of its meter.
        k_end = np.searchsorted(tempo_t, end - MERGE_MS, side='left')
        at_end = (k_end < len(tempo_t)) & (np.abs(tempo_t[np.minimum(k_end, len(tempo_t) - 1)] - end) <= MERGE_MS)
        k_in = np.searchsorted(tempo_t, start + MERGE_MS, side='right')
        inside = (k_in < len(tempo_t)) & (tempo_t[np.minimum(k_in, len(tempo_t) - 1)] < end - MERGE_MS)
        cut = (at_end | inside) & (np.abs(raw - meter) * step > MERGE_MS)
    counted = (start >= lo - MERGE_MS) & (end <= hi + MERGE_MS)
    for a, b in excluded:
        counted &= ~((start < b) & (end > a))
    return _Bars(start, end, length, counted, cut, meter, idx[:-1])


# ---------------------------------------------------------------- the strata of one set

def set_strata(ref: Reference, charts: Sequence = (), *, tags: Iterable[str] = (), tags_known: bool = True,
               rate_edit: bool = False, key: str | None = None) -> SetStrata:
    """The strata of one set (module docstring).

    ``ref`` comes from `ensomi_n.labels.read_set` or `reference_from_charts`;
    ``charts`` are the set's charts (`ensomi_n.labels.read_osu`), used for the
    per-difficulty head runs of S8 (without them, the union's heads). ``tags``
    are the set's voted tags; with ``tags_known`` False the tag clauses are
    skipped and S0 cannot check them (recorded in its detail).
    """
    key = ref.key if key is None else key
    tags = {t.strip().lower() for t in tags if t and t.strip()}
    lo, hi = ref.lo_ms, ref.hi_ms
    heads = ref.heads if ref.heads is not None else np.zeros(0)
    scorable = heads[ref.scorable(heads)]
    timing = ref.timing
    segs = ref.segments or ()
    lines = classify_lines(segs, heads, lo, hi)
    sections = _sections(lines, lo, hi)
    members: dict[str, Spans | None] = {}
    detail: dict[str, dict] = {}

    # Spans that the bar strata leave out: detour runs and implausible spans.
    detour_runs: list[tuple[float, float]] = []
    run_start = None
    for i, line in enumerate(lines):
        if line.kind == 'detour':
            run_start = line.start_ms if run_start is None else run_start
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if nxt is None or nxt.kind != 'detour':
                detour_runs.append((run_start, line.end_ms if math.isfinite(line.end_ms) else hi))
                run_start = None
    implausible = [(line.start_ms, line.end_ms if math.isfinite(line.end_ms) else hi)
                   for line in lines if line.kind == 'implausible']
    bars = _bars(timing, lines, lo, hi, _merge(detour_runs + implausible))

    def bars_holding(times: np.ndarray) -> Spans:
        if not len(bars.start) or not len(times):
            return []
        k = np.searchsorted(bars.start, times, side='right') - 1
        ok = (k >= 0) & (times < bars.end[np.maximum(k, 0)])
        return _merge((bars.start[i], bars.end[i]) for i in np.unique(k[ok]))

    def around(t: float) -> tuple[float, float]:
        """The bar before ``t`` and the bar after: previous downbeat to the next one."""
        db = timing.bar_starts()
        a = db[db < t - MERGE_MS]
        b = db[db > t + MERGE_MS]
        return (float(a[-1]) if len(a) else t, float(b[0]) if len(b) else t)

    musical = [line for line in lines if line.kind in MUSICAL_KINDS]
    resets = [line for line in lines if line.kind == 'reset' and lo - MERGE_MS <= line.start_ms <= hi + MERGE_MS]
    n_tempo = sum(1 for line in lines if line.kind == 'tempo')

    # S1: mixed notated beats.
    in_range = scorable[(scorable >= lo) & (scorable <= hi)]
    beat_idx = np.searchsorted(timing.beats, in_range + MERGE_MS, side='right') - 1
    mapped = len(np.unique(beat_idx[beat_idx >= 0]))
    rb, rdiv, rcls = subdivision_classes(timing, in_range)
    mixed = rb[rcls == 'mixed']
    mixed_share = len(mixed) / mapped if mapped else 0.0
    canon = timing.canonical()
    ct = canon.t
    keep_c = np.concatenate([[True], np.diff(ct) > 0]) if len(ct) else np.zeros(0, dtype=bool)
    canon_timing = Timing(ct[keep_c], np.zeros(int(keep_c.sum()), dtype=bool), np.zeros(int(keep_c.sum()), dtype=bool),
                          canon.beat_ms[keep_c], True, False) if len(ct) else Timing.empty()
    cb, _, ccls = subdivision_classes(canon_timing, in_range)
    c_idx = np.searchsorted(canon_timing.beats, in_range + MERGE_MS, side='right') - 1
    c_mapped = len(np.unique(c_idx[c_idx >= 0]))
    detail['S1'] = dict(mapped_beats=mapped, mixed_beats=int(len(mixed)), mixed_share=mixed_share,
                        finer_beats=int((rcls == 'finer').sum()),
                        canonical_mixed_share=(int((ccls == 'mixed').sum()) / c_mapped) if c_mapped else 0.0)
    if mapped and mixed_share >= MIXED_SHARE - 1e-12:
        members['S1'] = bars_holding(timing.beats[mixed] + MERGE_MS)

    # S2: irregular and changing meter.
    cnt = np.flatnonzero(bars.counted)
    irregular = [i for i in cnt if not bars.cut[i] and not np.any(np.isclose(bars.length[i], REGULAR_BAR_BEATS))]
    changes = []
    for a, b in zip(cnt[1:], cnt[2:]):
        if b == a + 1 and (bars.length[a] != bars.length[b] or bars.cut[a] or bars.cut[b]):
            changes.append((a, b))
    s2_spans = [(bars.start[i], bars.end[i]) for i in irregular]
    s2_spans += [(bars.start[a], bars.end[b]) for a, b in changes]
    lengths = sorted({float(bars.length[i]) for i in cnt})
    detail['S2'] = dict(counted_bars=int(len(cnt)), irregular_bars=len(irregular), changes=len(changes),
                        cut_bars=int(bars.cut[cnt].sum()) if len(cnt) else 0, bar_lengths=lengths[:12],
                        irregular=bool(irregular), changing=bool(changes),
                        meters=sorted({line.meter for line in lines if line.kind != 'implausible'})[:12])
    if irregular or changes:
        members['S2'] = _merge(s2_spans)

    # S3: tempo-mapped and ramps.
    short_flags = [(end - start) < SHORT_SECTION_MS for _, start, end in sections]
    share_short = (sum(short_flags) / len(sections)) if sections else 0.0
    tempo_mapped = (len(sections) >= TEMPO_MAPPED_MIN_LINES and share_short >= TEMPO_MAPPED_SHARE - 1e-12)
    tag_variable = tags_known and 'meta/variable timing' in tags
    ramps = []
    run = []
    for i in range(1, len(musical)):
        rel = musical[i].beat_ms / musical[i - 1].beat_ms - 1.0
        moving = RAMP_MIN_STEP < abs(rel) < RAMP_MAX_STEP
        if moving and run and np.sign(rel) == np.sign(run[-1][1]):
            run.append((i, rel))
        else:
            if len(run) >= RAMP_STEPS:
                ramps.append(run)
            run = [(i, rel)] if moving else []
    if len(run) >= RAMP_STEPS:
        ramps.append(run)
    ramp_spans = []
    for r in ramps:
        first, last = r[0][0] - 1, r[-1][0]
        end = musical[last].start_ms + SHORT_SECTION_MS
        if last + 1 < len(musical):
            end = min(end, musical[last + 1].start_ms)
        ramp_spans.append((musical[first].start_ms, end))
    tag_accel = tags_known and 'meta/accelerating bpm' in tags
    s3 = tempo_mapped or tag_variable or bool(ramps) or tag_accel
    detail['S3'] = dict(musical_lines=len(sections), short_share=share_short, tempo_mapped=tempo_mapped,
                        tag_variable_timing=bool(tag_variable), ramps=len(ramps), tag_accelerating=bool(tag_accel))
    if tempo_mapped or tag_variable or tag_accel:
        members['S3'] = None
    elif ramps:
        members['S3'] = _clip(ramp_spans, lo, hi)

    # S4 (applied per recording: only outside S3).
    short_spans = [(start, end) for (line, start, end) in sections
                   if (end - start) < max(SHORT_SECTION_MS, SHORT_SECTION_BEATS * line.canonical_ms)]
    reset_spans = [around(line.start_ms) for line in resets]
    s4_spans = _clip(short_spans + reset_spans, lo, hi)
    detail['S4'] = dict(short_sections=len(short_spans), resets=len(resets))

    # S5: metrical-level ambiguity.
    near = []
    for line, start, end in sections:
        bpm_c = 60000.0 / line.canonical_ms
        if abs(bpm_c / 80.0 - 1.0) <= LEVEL_NEAR or abs(bpm_c / 160.0 - 1.0) <= LEVEL_NEAR:
            near.append((max(start, lo), min(end, hi)))
    near = _merge(near)
    near_ms = sum(b - a for a, b in near)
    level_changes, modulations = [], []
    for i in range(1, len(sections)):
        prev, cur = sections[i - 1][0], sections[i][0]
        if cur.kind != 'tempo':
            continue
        span = (sections[i][1], sections[i][2])
        ratio = prev.beat_ms / cur.beat_ms
        if any(abs(ratio / r - 1.0) <= SAME_TEMPO for r in (2.0, 0.5)) and \
                _grid_error(prev.start_ms, prev.beat_ms, cur.start_ms) <= MERGE_MS + 1e-9:
            level_changes.append(span)
        rc = prev.canonical_ms / cur.canonical_ms
        if any(abs(rc / q - 1.0) <= SAME_TEMPO for q in LEVEL_FAMILY):
            modulations.append(span)
    s5_spans = (near if near_ms >= LEVEL_NEAR_MIN_MS else []) + level_changes + modulations
    longest = max(sections, key=lambda s: s[2] - s[1]) if sections else None
    dominant_bpm = 60000.0 / longest[0].beat_ms if longest else math.nan
    detail['S5'] = dict(near_80_160_ms=near_ms, level_changes=len(level_changes), modulations=len(modulations),
                        dominant_notated_bpm=dominant_bpm)
    if s5_spans:
        members['S5'] = _clip(s5_spans, lo, hi)

    # S6: sparse and beatless.
    gaps = []
    if len(heads) > 1:
        d = np.diff(heads)
        gaps = [(float(heads[i]), float(heads[i + 1])) for i in np.flatnonzero(d >= GAP_MS)]
    lines_in = ref.lines().t
    sparse = []
    if len(lines_in) > SPARSE_WINDOW_BEATS:
        a = lines_in[:-SPARSE_WINDOW_BEATS]
        b = lines_in[SPARSE_WINDOW_BEATS:]
        count = np.searchsorted(heads, b, side='left') - np.searchsorted(heads, a, side='left')
        ok = (count < SPARSE_MIN_HEADS) & (a >= lo - MERGE_MS) & (b <= hi + MERGE_MS)
        sparse = [(float(x), float(y)) for x, y in zip(a[ok], b[ok])]
    quiet = _merge(gaps + sparse)
    db = timing.bar_starts()
    reentry = []
    for _, e in quiet:
        k = int(np.searchsorted(db, e - MERGE_MS, side='left'))
        end = float(db[k + REENTRY_BARS]) if k + REENTRY_BARS < len(db) else hi
        reentry.append((e, max(end, e)))
    detail['S6'] = dict(gaps=len(gaps), sparse_windows=len(sparse), quiet_spans=len(quiet),
                        quiet_ms=float(sum(b - a for a, b in _clip(quiet, lo, hi))))
    if quiet:
        members['S6'] = _clip(quiet + reentry, lo, hi)

    # S7: swing.
    swing_bars = []
    tag_swing = tags_known and 'meta/swing' in tags
    for i in cnt:
        a, b = bars.start[i], bars.end[i]
        hs = scorable[(scorable >= a - MERGE_MS) & (scorable < b - MERGE_MS)]
        if len(hs) < SWING_MIN_HEADS:
            continue
        j = np.clip(np.searchsorted(timing.beats, hs + MERGE_MS, side='right') - 1, 0, len(timing.beats) - 1)
        step = timing.step_ms[j]
        x = (hs - timing.beats[j]) / step

        def swung(x: np.ndarray, step: np.ndarray) -> bool:
            n13 = int((np.abs(x - 1 / 3) * step <= MERGE_MS).sum())
            n12 = int((np.abs(x - 1 / 2) * step <= MERGE_MS).sum())
            n23 = int((np.abs(x - 2 / 3) * step <= MERGE_MS).sum())
            tot = n13 + n12 + n23
            return (n23 >= SWING_MIN_HEADS and n23 / tot >= SWING_TWO_THIRDS and n12 / tot < SWING_HALF
                    and n13 / tot < SWING_THIRD)

        ok = swung(x, step)
        if not ok and 60000.0 / timing.step_ms[bars.first_beat[i]] >= SWING_DOUBLE_BPM - 1e-9:
            pair = bars.first_beat[i] + 2 * ((j - bars.first_beat[i]) // 2)
            pair = np.clip(pair, 0, len(timing.beats) - 1)
            x2 = (hs - timing.beats[pair]) / (2.0 * step)
            ok = swung(x2, 2.0 * step)
        if ok:
            swing_bars.append((a, b))
    detail['S7'] = dict(swing_bars=len(swing_bars), tag_swing=bool(tag_swing))
    if tag_swing:
        members['S7'] = None
    elif len(swing_bars) >= SWING_MIN_BARS:
        members['S7'] = _merge(swing_bars)

    # S8: polyrhythm runs, per difficulty.
    poly = []
    head_sets = [c.heads for c in charts] if charts else [heads]
    for hs in head_sets:
        hs = np.asarray(hs, dtype=np.float64)
        hs = hs[ref.scorable(hs)]
        if len(hs) < POLY_MIN_HEADS:
            continue
        d = np.diff(hs)
        st = timing.step_at(hs)
        c = canonical_beat_ms(np.nan_to_num(st, nan=0.0))
        same_seg = (st[:-1] == st[1:]) & np.isfinite(st[:-1])
        for s in POLY_SPACINGS:
            m = (np.abs(d - s * c[:-1]) <= MERGE_MS + 1e-9) & same_seg
            if not m.any():
                continue
            edges = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if b - a >= POLY_MIN_HEADS - 1:
                    first, last = hs[a], hs[b]
                    if np.any((db > first + MERGE_MS) & (db <= last + MERGE_MS)):
                        poly.append((float(first), float(last)))
    detail['S8'] = dict(runs=len(poly))
    if poly:
        members['S8'] = _merge(poly)

    # S9: glitch, stutter and time-stretch.
    detour_ev = [(a, b) for a, b in detour_runs if _count_in(heads, a, b)]
    impl_ev = [(a, b) for a, b in implausible if _count_in(heads, a, b)]
    one_tempo_resets = [around(line.start_ms) for line in resets] if n_tempo == 0 else []
    j = np.searchsorted(timing.beats, in_range + MERGE_MS, side='right') - 1
    ok = j >= 0
    off = np.zeros(len(in_range), dtype=bool)
    off_spans = []
    if ok.any():
        jj = j[ok]
        step = timing.step_ms[jj]
        x = (in_range[ok] - timing.beats[jj]) / step
        off_ok = ~_on_divisors(x, step, DIVISORS)
        off[np.flatnonzero(ok)[off_ok]] = True
        off_spans = [(float(timing.beats[k]), float(timing.beats[k] + timing.step_ms[k])) for k in np.unique(jj[off_ok])]
    sub_beat = [(line.start_ms, line.end_ms) for line in lines
                if line.kind != 'implausible' and math.isfinite(line.end_ms)
                and line.end_ms - line.start_ms < line.canonical_ms and _count_in(heads, line.start_ms, line.end_ms)]
    detail['S9'] = dict(detour_runs=len(detour_runs), detour_runs_with_heads=len(detour_ev),
                        implausible_with_heads=len(impl_ev), one_tempo_resets=len(one_tempo_resets),
                        off_lattice_heads=int(off.sum()), sub_beat_segments=len(sub_beat))
    s9_spans = detour_ev + impl_ev + one_tempo_resets + off_spans + sub_beat
    if s9_spans:
        members['S9'] = _clip(s9_spans, lo, hi) or [(lo, lo)]

    # S0: control.
    fails = []
    if len([1 for _ in sections]) != 1:
        fails.append('musical_lines')
    in_lines = [line for line in lines if line.kind not in ('implausible', 'detour')
                and line.start_ms < hi and line.end_ms > lo]
    if any(line.meter not in (3, 4) for line in in_lines):
        fails.append('meter')
    if resets:
        fails.append('reset')
    if detour_ev or impl_ev:
        fails.append('detour_or_spike')
    if len(mixed):
        fails.append('mixed_beat')
    if tags_known and tags & set(TIMING_TAGS):
        fails.append('timing_tag')
    if sections:
        bpm_c = 60000.0 / sections[0][0].canonical_ms
        if abs(bpm_c / 80.0 - 1.0) <= LEVEL_NEAR or abs(bpm_c / 160.0 - 1.0) <= LEVEL_NEAR:
            fails.append('near_80_160')
    if rate_edit:
        fails.append('rate_edit')
    s0 = not fails
    detail['S0'] = dict(fails=fails, tags_checked=bool(tags_known))
    if s0:
        members['S0'] = None

    span_heads = {name: _heads_in(heads, spans) for name, spans in members.items()}
    timing_sig = tuple((s.start_ms, s.beat_ms, s.meter) for s in segs if s.plausible)
    return SetStrata(key, lo, hi, int(len(heads)), lines, members, detail, span_heads, s0, s3, s4_spans,
                     _heads_in(heads, s4_spans), dominant_bpm, timing_sig)


# ---------------------------------------------------------------- recordings

def recording_strata(sets: Sequence[SetStrata]) -> dict[str, dict[str, Spans | None]]:
    """The strata of one recording from its sets: ``{stratum: {set key: spans or None}}``.

    S0 needs every set in S0. S4 holds only when no set is in S3. S5 also
    holds, for every set, when the sets' longest sections are notated at
    different octaves. Every other stratum holds where any set holds it, on
    that set's spans.
    """
    sets = list(sets)
    out: dict[str, dict[str, Spans | None]] = {}
    if not sets:
        return out
    for name in STRATA:
        if name in ('S0', 'S4'):
            continue
        got = {s.key: s.members[name] for s in sets if name in s.members}
        if got:
            out[name] = got
    if all(s.s0 for s in sets):
        out['S0'] = {s.key: None for s in sets}
    if not any(s.s3 for s in sets):
        got = {s.key: s.s4_spans for s in sets if s.s4_spans}
        if got:
            out['S4'] = got
    bpms = [s.dominant_bpm for s in sets if math.isfinite(s.dominant_bpm)]
    octave = any(any(abs(a / b / 2.0 ** e - 1.0) <= SAME_TEMPO for e in OCTAVES) for a in bpms for b in bpms)
    if octave:
        out['S5'] = {s.key: None for s in sets}
    return out


def flags(ref: Reference, charts: Sequence = (), *, tags: Iterable[str] = (), tags_known: bool = False) -> dict:
    """One set's strata as plain values, for a training manifest (the interface R1's items read).

    ``strata`` lists the strata of the set taken as its own recording;
    ``hard`` is whether any of S1-S9 holds; ``flags`` holds a bool per
    stratum and per criterion (`ensomi_n.eval.census.criteria` names). Every
    value is JSON-serialisable.
    """
    s = set_strata(ref, charts, tags=tags, tags_known=tags_known)
    member = recording_strata([s])
    names = sorted(member, key=STRATA.index)
    f = {name: name in member for name in STRATA}
    d = s.detail
    f.update({
        'S2.irregular': bool(d['S2']['irregular']), 'S2.changing': bool(d['S2']['changing']),
        'S3.tempo_mapped': bool(d['S3']['tempo_mapped']), 'S3.ramps': d['S3']['ramps'] > 0,
        'S4.short_section': d['S4']['short_sections'] > 0, 'S4.reset': d['S4']['resets'] > 0,
        'S6.gap': d['S6']['gaps'] > 0, 'S9.detour_with_heads': d['S9']['detour_runs_with_heads'] > 0,
        'S9.off_lattice_head': d['S9']['off_lattice_heads'] > 0,
    })
    kinds: dict[str, int] = {}
    for line in s.lines:
        kinds[line.kind] = kinds.get(line.kind, 0) + 1
    return dict(flags=f, strata=names, hard=any(n in member for n in HARD), n_musical=d['S3']['musical_lines'],
                n_red=len(s.lines), mixed_share=round(float(d['S1']['mixed_share']), 5), line_kinds=kinds,
                strata_version='census-v1')
