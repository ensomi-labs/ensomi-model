"""The measures of one reference set against one prediction.

`score` returns a `SetScore`. The definitions follow the N1 evaluation design
(research notes, ``n1-eval-20261018`` §3); the package README states each one.
In brief:

- e_b is the signed distance from each reference canonical line b to the
  nearest predicted canonical line, prediction minus reference. Where either
  side's bars fix no canonical phase (free bars, `ensomi_n.timing`), both
  sides are compared at the level that fits the bar (`compared_lines`), so
  either canonical phase passes there.
- BL-τ is the head-weighted share of reference lines with ``|e_b| <= τ``.
  No offset is fitted or subtracted. τ = 20 ms is the deciding measure by
  the human's rule; 10 and 5 ms characterise and are also given with the set
  offset removed. Its chance is 2τ over the spacing each line is compared
  at, computed per line (`chance`); ``free_share`` is the head share
  compared at a refined level, and ``free_share_pred`` the part refined only
  because of the prediction's own bars.
- The set offset o is the head-weighted median of e_b over the whole scored
  range, reported on its own.
- Level, bars, boundaries, resets, subdivision class, calibration and the
  mapper-work proxies characterise. None of them removes o: bars and resets
  match within 20 ms of the reference, as BL-20 does, so a whole-grid phase
  shift fails them where it fails BL-20. Removing o would let any shift under
  half a beat pass, which puts the chance of the bar-start share at 1/meter.
- The ``_o`` variants (``bl10_o``, ``bl5_o``, ``bars_*_o``,
  ``reset_recall_o``) are the same measures with o removed. They
  characterise how the timing is shaped apart from its phase, and never
  decide.

Every share is computed on the reference's scored range: from the last
canonical line at or before the first head to the first at or after the
last. ``spans`` restricts the measures to stratum spans; o still comes from
the whole range, as one offset serves the whole song.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ..mapformat import MusicMap
from ..timing import MERGE_MS, Lines, Reference, Timing, canonical_beat_ms

TAUS = (20.0, 10.0, 5.0)
DECIDING_TAU = 20.0
LEVEL_TOLERANCE = 0.005
#: A beat list's beat length is read over at least this many intervals, centred on the reference run.
LEVEL_MIN_INTERVALS = 12
#: Intervals further than this from their median (missed or extra beats) break a chain of beats.
LEVEL_CHAIN_TOLERANCE = 0.25
#: Consecutive reference runs form one tempo trend while each step moves the tempo the same way by under this.
LEVEL_TREND_STEP = 0.05
LEVEL_FAMILY = (3 / 2, 2 / 3, 4 / 3, 3 / 4, 5 / 4, 4 / 5)
LEVEL_CLASSES = ('correct', 'octave', 'family', 'other', 'missing')
SAME_TEMPO = 0.005
SPAN_TEMPO_BEATS = 0.01
LONG_SPAN_MS = 8000.0
OFFSET_EDIT_MS = 10.0
WINDOW_MS = 4000.0
DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)
SUBDIVISION_CLASSES = ('plain', 'duple', 'triple', 'mixed', 'finer')
ECE_BINS = 10
AUC_LEVELS = 1001
PRED_MARGIN_MS = 6000.0
EDIT_TYPES = ('offset', 'retime', 'delete', 'level', 'meter')


def compared_lines(ref_lines: Lines, pred_lines: Lines, ref_exact: bool, pred_exact: bool, *,
                   ref_all: Lines | None = None, ref_end: float = -math.inf,
                   pred_end: float = -math.inf) -> tuple[Lines, Lines, np.ndarray]:
    """The reference and predicted lines at the level they are compared at.

    Each line interval is refined (`Lines.refine`) by the larger of its own
    free factor and that of the other side's interval holding its middle, so
    a free bar on either side is compared at the level that fits it, and
    everywhere else nothing changes. The middle, not the start, is looked up:
    a prediction shifted by a few ms then still meets the same intervals. ``ref_all`` (the reference's lines
    beyond the scored range) gives the reference's factor at predicted lines
    outside ``ref_lines``; ``ref_end`` and ``pred_end`` are each side's last
    notated beat (`Lines.refine`). Returns both refined line sets and, per refined
    reference line, whether it was refined only because of the prediction.
    """
    ref_all = ref_lines if ref_all is None else ref_all

    def mids(lines: Lines) -> np.ndarray:
        return np.append((lines.t[:-1] + lines.t[1:]) / 2.0, lines.t[-1:]) if len(lines) else lines.t

    own_r, own_p = ref_lines.factor, pred_lines.factor
    f_r = np.maximum(own_r, pred_lines.factor_at(mids(ref_lines))) if len(ref_lines) else own_r
    f_p = np.maximum(own_p, ref_all.factor_at(mids(pred_lines))) if len(pred_lines) else own_p
    ref_c, parent = ref_lines.refine(f_r, ref_exact, ref_end)
    pred_c, _ = pred_lines.refine(f_p, pred_exact, pred_end)
    by_pred = (f_r > own_r)[parent] if len(parent) else np.zeros(0, dtype=bool)
    return ref_c, pred_c, by_pred


def nearest(ref_t: np.ndarray, pred_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed error to the nearest predicted line, and its index (-1 and NaN without lines).

    An exact tie goes to the earlier line.
    """
    ref_t = np.asarray(ref_t, dtype=np.float64)
    if not len(pred_t):
        return np.full(ref_t.shape, np.nan), np.full(ref_t.shape, -1, dtype=np.int64)
    i = np.searchsorted(pred_t, ref_t)
    lo = np.clip(i - 1, 0, len(pred_t) - 1)
    hi = np.clip(i, 0, len(pred_t) - 1)
    e_lo, e_hi = pred_t[lo] - ref_t, pred_t[hi] - ref_t
    use_hi = np.abs(e_hi) < np.abs(e_lo)
    return np.where(use_hi, e_hi, e_lo), np.where(use_hi, hi, lo)


def weighted_median(x, w) -> float:
    """Median of ``x`` under weights ``w``; the midpoint where the weight splits exactly. NaN when empty."""
    x, w = np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return math.nan
    order = np.argsort(x[ok], kind='stable')
    xs, cum = x[ok][order], np.cumsum(w[ok][order])
    half = cum[-1] / 2.0
    k = int(np.searchsorted(cum, half - 1e-12 * cum[-1], side='left'))
    if abs(cum[k] - half) <= 1e-12 * cum[-1] and k + 1 < len(xs):
        return float((xs[k] + xs[k + 1]) / 2.0)
    return float(xs[k])


def share(ok, w) -> float:
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    return float((np.asarray(ok, dtype=bool) * w).sum() / total) if total > 0 else math.nan


def level_class(ratio: float) -> str:
    """Class of a predicted-over-reference notated tempo ratio."""
    if not math.isfinite(ratio) or ratio <= 0:
        return 'missing'
    if abs(ratio - 1.0) <= LEVEL_TOLERANCE:
        return 'correct'
    if any(abs(ratio / 2.0 ** k - 1.0) <= LEVEL_TOLERANCE for k in (-3, -2, -1, 1, 2, 3)):
        return 'octave'
    if any(abs(ratio / q - 1.0) <= LEVEL_TOLERANCE for q in LEVEL_FAMILY):
        return 'family'
    return 'other'


def tempo_spans(timing: Timing) -> list[tuple[float, float, float]]:
    """Runs of beats at one notated tempo: ``(start_ms, end_ms, beat_ms)``.

    A run ends where a beat's step leaves the run's first step by more than
    0.5 % (exact steps, so same-tempo resets join one run) or 1 % (running
    medians of a bare beat list). The last run ends one step after its last beat.
    """
    b, step = timing.beats, timing.step_ms
    if not len(b):
        return []
    tol = SAME_TEMPO if timing.exact_steps else SPAN_TEMPO_BEATS
    spans, start, ref = [], 0, step[0]
    for i in range(1, len(b)):
        if not (math.isfinite(step[i]) and math.isfinite(ref)) or abs(step[i] / ref - 1.0) > tol:
            spans.append((float(b[start]), float(b[i]), float(ref)))
            start, ref = i, step[i]
    last_step = step[-1] if math.isfinite(step[-1]) else 0.0
    spans.append((float(b[start]), float(b[-1] + last_step), float(ref)))
    return spans


def long_changes(spans: Sequence[tuple[float, float, float]]) -> np.ndarray:
    """Tempo changes between runs of 8 s or more.

    For each pair of consecutive long runs (shorter runs between them
    skipped) whose tempi differ by more than 1 %, the change is where the
    earlier run ends.
    """
    long = [s for s in spans if s[1] - s[0] >= LONG_SPAN_MS]
    out = [a[1] for a, b in zip(long, long[1:])
           if math.isfinite(a[2]) and math.isfinite(b[2]) and abs(b[2] / a[2] - 1.0) > SPAN_TEMPO_BEATS]
    return np.array(out, dtype=np.float64)


def _class_of_divisor(d: int) -> str:
    if d == 0:
        return 'finer'
    if d <= 2:
        return 'plain'
    if d & (d - 1) == 0:
        return 'duple'
    return 'mixed' if d % 4 == 0 else 'triple'


def subdivision_classes(timing: Timing, heads) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notated beats with two or more heads: their index, required divisor and class.

    Each head belongs to the notated beat it falls in, a head within
    ``MERGE_MS`` before a beat belonging to that beat. It snaps to the smallest
    d in (1, 2, 3, 4, 6, 8, 12, 16) within ``MERGE_MS``, measured on the beat's
    step; a head off every d is finer. The beat's divisor is the least common
    multiple of its heads' d (0 when any head is finer). Classes: plain (1 or
    2), duple (a power of two, 4 or more), triple (3 divides it, 4 does not),
    mixed (3 and 4 divide it), finer.
    """
    b, step = timing.beats, timing.step_ms
    h = np.asarray(heads if heads is not None else [], dtype=np.float64)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype='<U6'))
    if not len(b) or not len(h):
        return empty
    i = np.searchsorted(b, h + MERGE_MS, side='right') - 1
    ok = (i >= 0) & np.isfinite(step[np.maximum(i, 0)])
    h, i = h[ok], i[ok]
    if not len(h):
        return empty
    x = (h - b[i]) / step[i]
    d = np.zeros(len(h), dtype=np.int64)
    for dv in DIVISORS:
        err = np.abs(x - np.round(x * dv) / dv) * step[i]
        hit = (d == 0) & (err <= MERGE_MS)
        d[hit] = dv
    order = np.argsort(i, kind='stable')
    i, d = i[order], d[order]
    beats, first, counts = np.unique(i, return_index=True, return_counts=True)
    many = counts >= 2
    if not many.any():
        return empty
    lcm = np.lcm.reduceat(np.maximum(d, 1), first)
    finer = np.minimum.reduceat(d, first) == 0
    div = np.where(finer, 0, lcm)[many]
    return beats[many], div, np.array([_class_of_divisor(int(v)) for v in div])


def _class_of_probabilities(p: np.ndarray) -> str:
    if not np.all(np.isfinite(p)):
        return ''
    triple, duple = p[0] >= 0.5, p[1] >= 0.5
    return 'mixed' if triple and duple else 'triple' if triple else 'duple' if duple else 'plain'


@dataclass
class _Pred:
    timing: Timing
    confidence: np.ndarray | None
    subdivision: np.ndarray | None
    heads: np.ndarray | None
    red_lines: np.ndarray
    has_map: bool


def _prediction(pred, lo: float, hi: float, decoded_to_osu_ms: float | None) -> _Pred:
    if pred is None:
        return _Pred(Timing.empty(), None, None, None, np.zeros(0), False)
    if isinstance(pred, Timing):
        spans = tempo_spans(pred)
        reds = pred.beats[pred.seg_start] if pred.seg_start.any() else np.array([a for a, _, _ in spans])
        return _Pred(pred, None, None, None, reds, True)
    if not isinstance(pred, MusicMap):
        pred = MusicMap.from_beats(pred)
    m = pred.on_osu_clock(decoded_to_osu_ms)
    timing = m.timing(lo - PRED_MARGIN_MS, hi + PRED_MARGIN_MS)
    return _Pred(timing, m.beat_confidence(timing), m.subdivision, m.heads, m.red_lines(timing), True)


class _Region:
    """The scored range, or its overlap with stratum spans."""

    def __init__(self, lo: float, hi: float, spans):
        self.whole = spans is None
        if self.whole:
            self.parts = [(lo, hi)]
        else:
            self.parts = [(max(float(a), lo), min(float(b), hi)) for a, b in spans]
            self.parts = [(a, b) for a, b in self.parts if b > a]

    def contains(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        ok = np.zeros(t.shape, dtype=bool)
        for a, b in self.parts:
            ok |= (t >= a) & (t <= b)
        return ok

    def overlap(self, a, b) -> np.ndarray:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        total = np.zeros(np.broadcast(a, b).shape)
        for lo, hi in self.parts:
            total += np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None)
        return total

    @property
    def ms(self) -> float:
        return float(sum(b - a for a, b in self.parts))


@dataclass
class SetScore:
    """The measures of one reference set against one prediction.

    Shares are NaN where nothing is scored (no heads in the spans, no bars in
    the reference). ``beats`` holds per-line arrays: ``t``, ``e`` (ms),
    ``w`` (head weight), ``p`` (confidence, NaN without) and ``scored``.
    ``suff`` holds the counts that pool across sets (calibration, kappa,
    boundaries, resets, windows).
    """
    key: str
    bl: dict[float, float]
    bl_beat: dict[float, float]
    bl_o: dict[float, float]
    chance: dict[float, float]
    offset_ms: float
    level: float
    level_classes: dict[str, float]
    bars_start: float
    bars_same: float
    bars_precision: float
    bars_start_o: float
    bars_same_o: float
    bars_precision_o: float
    free_share: float
    free_share_pred: float
    pred_has_bars: bool
    boundary_recall: float
    boundary_precision: float
    reset_recall: float
    reset_recall_o: float
    subdivision_kappa: float
    subdivision_accuracy: float
    ece: float
    auc: float
    confident_fail: float
    confident_predicted_fail: float
    check_load_s_per_min: float
    hidden_per_min: float
    hidden_share: float
    edits: dict[str, float]
    edits_per_min: float
    redo_s_per_min: float
    minutes: float
    heads: float
    n_lines: int
    n_scored: int
    beats: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    suff: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    resets: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    @property
    def bl20(self) -> float:
        return self.bl[20.0]

    @property
    def bl10(self) -> float:
        return self.bl[10.0]

    @property
    def bl5(self) -> float:
        return self.bl[5.0]

    def summary(self) -> dict:
        """The scalar measures, with string keys, for a report."""
        out = {'key': self.key}
        for tau in TAUS:
            t = f'{tau:g}'
            out[f'bl{t}'] = self.bl[tau]
            out[f'bl{t}_beat'] = self.bl_beat[tau]
            out[f'chance{t}'] = self.chance[tau]
        for tau, v in self.bl_o.items():
            out[f'bl{tau:g}_o'] = v
        for name in ('offset_ms', 'level', 'bars_start', 'bars_same', 'bars_precision', 'bars_start_o',
                     'bars_same_o', 'bars_precision_o', 'free_share', 'free_share_pred', 'pred_has_bars',
                     'boundary_recall',
                     'boundary_precision', 'reset_recall', 'reset_recall_o', 'subdivision_kappa',
                     'subdivision_accuracy', 'ece', 'auc', 'confident_fail', 'confident_predicted_fail',
                     'check_load_s_per_min', 'hidden_per_min', 'hidden_share', 'edits_per_min',
                     'redo_s_per_min', 'minutes', 'heads', 'n_lines', 'n_scored'):
            out[name] = getattr(self, name)
        out.update({f'level_{k}': v for k, v in self.level_classes.items()})
        out.update({f'edits_{k}': v for k, v in self.edits.items()})
        return out


def kappa(confusion: np.ndarray) -> tuple[float, float]:
    """Cohen's kappa and accuracy of a confusion matrix (NaN when empty)."""
    c = np.asarray(confusion, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return math.nan, math.nan
    po = np.trace(c) / n
    pe = float((c.sum(axis=0) * c.sum(axis=1)).sum() / n ** 2)
    return (float((po - pe) / (1 - pe)) if pe < 1 else math.nan), float(po)


def ece_from_bins(bins: np.ndarray) -> float:
    """Expected calibration error from per-bin (count, sum p, sum passing)."""
    n = bins[:, 0].sum()
    if n <= 0:
        return math.nan
    used = bins[:, 0] > 0
    return float(np.abs(bins[used, 1] - bins[used, 2]).sum() / n)


def auc_from_hist(hist: np.ndarray) -> float:
    """AUC of low confidence for failing beats, from per-level (failing, passing) counts.

    The probability that a failing beat has lower confidence than a passing
    one, ties counting half; NaN without both.
    """
    fail, ok = hist[:, 0], hist[:, 1]
    nf, no = fail.sum(), ok.sum()
    if nf <= 0 or no <= 0:
        return math.nan
    ok_above = np.cumsum(ok[::-1])[::-1] - ok
    return float((fail * (ok_above + 0.5 * ok)).sum() / (nf * no))


def score(ref: Reference, pred: MusicMap | Timing | Sequence[float] | None, *, spans=None,
          flag_below: float = 0.5, confident_at: float = 0.9,
          decoded_to_osu_ms: float | None = None) -> SetScore:
    """Score one prediction against one reference.

    ``pred`` is a `MusicMap`, a `Timing`, a beat array, or None for an empty
    prediction. ``spans`` is a list of ``(start_ms, end_ms)`` stratum spans.
    ``flag_below`` is the confidence under which a window counts as flagged
    "listen here"; ``confident_at`` the confidence from which a beat counts
    as marked confident. Raises ``ValueError`` for a map on the decoded clock
    without ``decoded_to_osu_ms``.
    """
    if not isinstance(ref, Reference):
        swapped = isinstance(pred, Reference) and isinstance(ref, (MusicMap, Timing))
        raise TypeError('score(ref, pred): the reference comes first' if swapped
                        else f'score needs a Reference first, got {type(ref).__name__}')
    lo, hi = ref.lo_ms, ref.hi_ms
    region = _Region(lo, hi, spans)
    p = _prediction(pred, lo, hi, decoded_to_osu_ms)
    lines, pred_lines, by_pred = compared_lines(ref.lines(), p.timing.canonical(), ref.timing.exact_steps,
                                                p.timing.exact_steps, ref_all=ref.timing.canonical(),
                                                ref_end=float(ref.timing.beats[-1]) if len(ref.timing) else -math.inf,
                                                pred_end=float(p.timing.beats[-1]) if len(p.timing) else -math.inf)
    e, j = nearest(lines.t, pred_lines.t)
    w_all = ref.weights(lines)

    # The set offset over the whole range.
    offset = weighted_median(e, w_all)
    o = offset if math.isfinite(offset) else 0.0

    if region.whole:
        w = w_all
        in_region = np.ones(len(lines), dtype=bool)
    else:
        heads = ref.heads
        sub = Reference(ref.timing, lo, hi, None if heads is None else heads[region.contains(heads)],
                        ref.segments, ref.unscorable, ref.key)
        w = sub.weights(lines)
        in_region = region.contains(lines.t)
        if heads is None:
            w = w * in_region
    scored = w > 0
    err = np.abs(e)
    finite = np.isfinite(e)

    bl = {tau: share(finite & (err <= tau + 1e-9), w) for tau in TAUS}
    beat_mask = in_region
    bl_beat = {tau: share(finite[beat_mask] & (err[beat_mask] <= tau + 1e-9), np.ones(beat_mask.sum()))
               for tau in TAUS}
    bl_o = {tau: share(finite & (np.abs(e - o) <= tau + 1e-9), w) for tau in TAUS if tau != DECIDING_TAU}
    chance = {tau: float((w * np.minimum(1.0, 2 * tau / lines.beat_ms)).sum() / w.sum()) if w.sum() > 0
              else math.nan for tau in TAUS}
    refined = lines.factor > 1
    free_share = share(refined, w)
    free_share_pred = share(by_pred, w)
    passing = finite & (err <= DECIDING_TAU + 1e-9)

    # Confidence of the predicted line nearest each reference line.
    pb = np.full(len(lines), np.nan)
    if p.confidence is not None and len(pred_lines):
        ok = j >= 0
        conf = p.confidence
        pb[ok] = np.fmin(conf[pred_lines.src_lo[j[ok]]], conf[pred_lines.src_hi[j[ok]]])
        both_nan = ok & ~np.isfinite(pb)
        pb[both_nan] = np.nan

    # Level, per reference segment; a beat list's beat length is read over each run at one tempo.
    ref_segments = _ref_segments(ref)
    level_dur = {c: 0.0 for c in LEVEL_CLASSES}
    seg_class = []
    run_step: dict[int, float] = {}
    runs = _level_runs(ref_segments)
    for run, ((ra, rb), (ga, gb)) in enumerate(zip(runs, _trend_spans(ref_segments))):
        a_c, b_c = max(ra, lo), min(rb, hi)
        run_step[run] = _pred_beat_length(p.timing, a_c, b_c, ga, gb) if b_c > a_c else math.nan
    runs_of = _run_index(ref_segments)
    for (a, b, step), run in zip(ref_segments, runs_of):
        a_c, b_c = max(a, lo), min(b, hi)
        dur = float(region.overlap(a_c, b_c)) if b_c > a_c else 0.0
        if p.timing.exact_steps:
            pstep = _pred_beat_length(p.timing, a_c, b_c) if b_c > a_c else math.nan
        else:
            pstep = run_step[run]
        cls = level_class(step / pstep) if math.isfinite(pstep) and pstep > 0 else 'missing'
        seg_class.append((a, b, cls, dur))
        level_dur[cls] += dur
    total_dur = sum(level_dur.values())
    level = level_dur['correct'] / total_dur if total_dur > 0 else math.nan
    level_classes = {c: (v / total_dur if total_dur > 0 else math.nan) for c, v in level_dur.items()}

    # Bars, matched with no offset removed; the _o variants characterise only.
    ref_bars = ref.timing.bar_starts()
    pred_bars = p.timing.bar_starts() + 0.0
    bars = bars_o = (math.nan, math.nan, math.nan)
    bar_rows = []
    bar_windows = []
    if ref.timing.has_bars and len(ref_bars):
        # A bar start is scored inside the range; a bar's length only when its end is too, since
        # the chart's bars past the last head are extrapolated, as its beats there are unscored.
        ends = np.append(ref_bars[1:], np.nan)
        keep = region.overlap(ref_bars, np.where(np.isfinite(ends), ends, ref_bars)) > 0
        start_keep = region.contains(ref_bars)
        same_keep = start_keep & region.contains(ends)
        pb_in = pred_bars[region.contains(pred_bars)]
        same, bars = _bar_terms(ref_bars, ends, pred_bars, pb_in, start_keep, same_keep)
        _, bars_o = _bar_terms(ref_bars, ends, pred_bars - o, pb_in - o, start_keep, same_keep)
        bar_rows = [(float(a), float(b), bool(s)) for a, b, s, k in zip(ref_bars, ends, same, same_keep) if k]
        bar_windows = [(float(a), float(b) if math.isfinite(b) else hi) for a, b, k in zip(ref_bars, ends, keep) if k]

    # Boundaries: long tempo changes.
    ref_changes = long_changes(tempo_spans(ref.timing))
    ref_changes = ref_changes[region.contains(ref_changes)]
    pred_changes = long_changes(tempo_spans(p.timing)) if len(p.timing) else np.zeros(0)
    pred_changes = pred_changes[region.contains(pred_changes)]
    tol_ref = canonical_beat_ms(np.nan_to_num(ref.timing.step_at(ref_changes), nan=0.0)) if len(ref_changes) \
        else np.zeros(0)
    matched_ref = _within(ref_changes, pred_changes, tol_ref)
    tol_pred = canonical_beat_ms(np.nan_to_num(ref.timing.step_at(pred_changes), nan=0.0)) if len(pred_changes) \
        else np.zeros(0)
    matched_pred = _within(pred_changes, ref_changes, tol_pred)
    boundary_recall = float(matched_ref.mean()) if len(ref_changes) else math.nan
    boundary_precision = float(matched_pred.mean()) if len(pred_changes) else math.nan

    # Same-tempo phase resets.
    reset_t = _resets(ref)
    reset_t = reset_t[region.contains(reset_t)]
    pred_marks = np.union1d(p.timing.beats[p.timing.seg_start], pred_bars)
    reset_ok = _within(reset_t, pred_marks, DECIDING_TAU)
    reset_ok_o = _within(reset_t + o, pred_marks, DECIDING_TAU)
    reset_recall = float(reset_ok.mean()) if len(reset_t) else math.nan
    reset_recall_o = float(reset_ok_o.mean()) if len(reset_t) else math.nan

    # Subdivision class.
    confusion = np.zeros((len(SUBDIVISION_CLASSES), len(SUBDIVISION_CLASSES)))
    if ref.heads is not None:
        heads = ref.heads[region.contains(ref.heads) & ref.scorable(ref.heads)]
        rb, _, rc = subdivision_classes(ref.timing, heads)
        pred_cls = _pred_classes(p, ref.timing, rb)
        for a, b in zip(rc, pred_cls):
            if b:
                confusion[SUBDIVISION_CLASSES.index(a), SUBDIVISION_CLASSES.index(b)] += 1
    subdivision_kappa, subdivision_accuracy = kappa(confusion)

    # Calibration over scored lines with a confidence.
    cal = scored & np.isfinite(pb)
    bins = np.zeros((ECE_BINS, 3))
    hist = np.zeros((AUC_LEVELS, 2))
    conf_counts = np.zeros(3)
    if cal.any():
        pc, okc = pb[cal], passing[cal]
        k = np.minimum((pc * ECE_BINS).astype(np.int64), ECE_BINS - 1)
        np.add.at(bins, (k, 0), 1.0)
        np.add.at(bins, (k, 1), pc)
        np.add.at(bins, (k, 2), okc.astype(np.float64))
        lev = np.clip(np.round(pc * (AUC_LEVELS - 1)).astype(np.int64), 0, AUC_LEVELS - 1)
        np.add.at(hist, (lev, np.where(okc, 1, 0)), 1.0)
        conf = pc >= confident_at
        conf_counts = np.array([conf.sum(), (conf & ~okc).sum(), (1.0 - pc[conf]).sum()], dtype=np.float64)
    ece = ece_from_bins(bins)
    auc = auc_from_hist(hist)
    confident_fail = conf_counts[1] / conf_counts[0] if conf_counts[0] > 0 else math.nan
    confident_predicted_fail = conf_counts[2] / conf_counts[0] if conf_counts[0] > 0 else math.nan

    # Mapper work: windows.
    minutes = region.ms / 60000.0
    if bar_windows:
        windows = bar_windows
    else:
        edges = np.arange(lo, hi + WINDOW_MS, WINDOW_MS)
        windows = [(float(a), float(min(a + WINDOW_MS, hi))) for a in edges[:-1]] or [(lo, hi)]
    win_fail = win_flag = 0
    flagged_ms = 0.0
    hidden = 0
    for a, b in windows:
        if region.overlap(a, b) <= 0:
            continue
        inside = scored & (lines.t >= a) & (lines.t < b)
        if not inside.any():
            continue
        fails = bool((~passing[inside]).any())
        flagged = bool(np.any(pb[inside] < flag_below)) if np.isfinite(pb[inside]).any() else False
        win_fail += fails
        win_flag += flagged
        flagged_ms += float(region.overlap(a, b)) if flagged else 0.0
        hidden += fails and not flagged
    check_load = flagged_ms / 1000.0 / minutes if minutes > 0 else math.nan
    hidden_per_min = hidden / minutes if minutes > 0 else math.nan
    hidden_share = hidden / win_fail if win_fail else math.nan

    # Mapper work: red-line edits and redo time.
    edits = {k: 0.0 for k in EDIT_TYPES}
    edits['offset'] = 0.0 if math.isfinite(offset) and abs(offset) <= OFFSET_EDIT_MS else 1.0
    redo_ms = 0.0
    for a, b, cls, dur in seg_class:
        if dur <= 0:
            continue
        inside = scored & (lines.t >= a) & (lines.t < b)
        if inside.any():
            edits['retime'] += float((~passing[inside]).any())
            if passing[inside].mean() < 0.5:
                redo_ms += dur
    runs = 0
    previous_wrong = False
    for a, b, cls, dur in seg_class:
        if dur <= 0:
            continue
        wrong = cls != 'correct'
        runs += wrong and not previous_wrong
        previous_wrong = wrong
    edits['level'] = float(runs)
    ref_reds = _ref_red_lines(ref)
    pred_reds = p.red_lines[region.contains(p.red_lines)] if len(p.red_lines) else np.zeros(0)
    if len(pred_reds):
        tol = canonical_beat_ms(np.nan_to_num(ref.timing.step_at(pred_reds), nan=0.0))
        edits['delete'] = float((~_within(pred_reds, ref_reds, tol)).sum())
    edits['meter'] = float(sum(1 for _, _, s in bar_rows if not s))
    edits_per_min = sum(edits.values()) / minutes if minutes > 0 else math.nan

    pooled = dict(
        ece_bins=bins, auc_hist=hist, confident=conf_counts, kappa=confusion,
        boundaries=np.array([len(ref_changes), matched_ref.sum(), len(pred_changes), matched_pred.sum()],
                            dtype=np.float64),
        resets=np.array([len(reset_t), reset_ok.sum(), reset_ok_o.sum()], dtype=np.float64),
        windows=np.array([win_fail, hidden], dtype=np.float64),
    )
    return SetScore(
        key=ref.key, bl=bl, bl_beat=bl_beat, bl_o=bl_o, chance=chance, offset_ms=offset, level=level,
        level_classes=level_classes, bars_start=bars[0], bars_same=bars[1], bars_precision=bars[2],
        bars_start_o=bars_o[0], bars_same_o=bars_o[1], bars_precision_o=bars_o[2],
        free_share=free_share, free_share_pred=free_share_pred, pred_has_bars=bool(p.timing.has_bars and len(pred_bars)), boundary_recall=boundary_recall,
        boundary_precision=boundary_precision, reset_recall=reset_recall, reset_recall_o=reset_recall_o,
        subdivision_kappa=subdivision_kappa,
        subdivision_accuracy=subdivision_accuracy, ece=ece, auc=auc, confident_fail=confident_fail,
        confident_predicted_fail=confident_predicted_fail, check_load_s_per_min=check_load,
        hidden_per_min=hidden_per_min, hidden_share=hidden_share, edits=edits, edits_per_min=edits_per_min,
        redo_s_per_min=redo_ms / 1000.0 / minutes if minutes > 0 else math.nan, minutes=minutes,
        heads=float(w.sum()) if ref.heads is not None else math.nan, n_lines=int(len(lines)),
        n_scored=int(scored.sum()),
        beats=dict(t=lines.t, e=e, w=w, p=pb, scored=scored), suff=pooled,
        resets=dict(t=reset_t, ok=reset_ok, ok_o=reset_ok_o))


def _run_index(segments: Sequence[tuple[float, float, float]]) -> list[int]:
    """Run number of each reference segment: consecutive segments within 0.5 % of one tempo share a run."""
    out, run, ref_step = [], -1, math.nan
    for _, _, step in segments:
        if not (math.isfinite(ref_step) and abs(step / ref_step - 1.0) <= LEVEL_TOLERANCE):
            run, ref_step = run + 1, step
        out.append(run)
    return out


def _level_runs(segments: Sequence[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """``(start, end)`` of each run of reference segments at one tempo (resets and redundant lines join)."""
    runs: list[list[float]] = []
    for (a, b, _), run in zip(segments, _run_index(segments)):
        if run == len(runs):
            runs.append([a, b])
        else:
            runs[run][1] = b
    return [(a, b) for a, b in runs]


def _trend_spans(segments: Sequence[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """For each run (`_level_runs`), the span of the runs around it that continue one tempo trend.

    Going outwards from a run, each next run must move the tempo the same
    way as the step before it, by under ``LEVEL_TREND_STEP``, and hold fewer
    than ``LEVEL_MIN_INTERVALS`` beats (a long run at one tempo ends a
    ramp); the first step on each side sets that side's way, and at a
    turning point only the left side is kept.
    """
    steps = [segments[i][2] for i in [_run_index(segments).index(r) for r in range(len(_level_runs(segments)))]]
    runs = _level_runs(segments)
    out = []
    for r in range(len(runs)):
        def extend(direction: int) -> tuple[int, int]:
            k, way = r, 0
            while 0 <= k + direction < len(runs):
                change = steps[k + direction] / steps[k] - 1.0
                sign = 1 if change > 0 else -1
                a, b = runs[k + direction]
                long_run = (b - a) / steps[k + direction] >= LEVEL_MIN_INTERVALS
                if abs(change) >= LEVEL_TREND_STEP or (way and sign != way) or long_run:
                    break
                way, k = sign, k + direction
            return k, way
        left, way_l = extend(-1)
        right, way_r = extend(1)
        if way_l and way_r and way_l == way_r:
            right = r      # going right from a turning point reverses the trend seen going left
        out.append((runs[left][0], runs[right][1]))
    return out


def chain_beat_length(beats: np.ndarray, at: float | None = None) -> float:
    """The beat length of a beat list at beat number ``at``, robust to frame grids and missed or extra beats.

    Intervals within 25 % of their median link beats into chains; a missed
    or an extra beat breaks a chain. In each chain the beat times are fitted
    against beat number by least squares, with a quadratic term when the
    chain has four or more beats, and the slope is read at ``at`` (beat
    number from ``beats[0]``; default the middle). A frame grid moves this
    far less than any one interval or the median of them: at 123 BPM on
    20 ms frames the intervals are 480 and 500 ms, the median 1.6 % off. On a
    window centred on ``at`` the quadratic term drops out, so the slope is
    the linear one; off centre (at a list's ends) it keeps a ramp unbiased.
    Chains are pooled with weights n (n^2 - 1) / 12 for n beats. NaN with
    fewer than two beats.
    """
    x = np.asarray(beats, dtype=np.float64)
    if len(x) < 2:
        return math.nan
    d = np.diff(x)
    med = float(np.median(d))
    if not med > 0:
        return math.nan
    at = (len(x) - 1) / 2.0 if at is None else float(at)
    ok = np.abs(d / med - 1.0) <= LEVEL_CHAIN_TOLERANCE
    num = den = 0.0
    k = 0
    while k < len(d):
        if not ok[k]:
            k += 1
            continue
        j = k
        while j < len(d) and ok[j]:
            j += 1
        y = x[k:j + 1]
        n = len(y)
        i = np.arange(n, dtype=np.float64) - (at - k)
        if n >= 4:
            coef = np.polyfit(i, y, 2)
            slope = float(coef[1])
        else:
            slope = float(np.polyfit(i, y, 1)[0])
        weight = n * (n * n - 1) / 12.0
        num += weight * slope
        den += weight
        k = j
    return num / den if den > 0 else med


def _pred_beat_length(timing: Timing, a: float, b: float, ga: float = -math.inf, gb: float = math.inf) -> float:
    """The predicted notated beat length over the reference span ``[a, b)``.

    With exact steps, the step in force at the span's middle. For a bare beat
    list, `chain_beat_length` over the predicted intervals that start inside
    the span (or the interval holding its middle when none does), read at
    the span's centre. When those are fewer than ``LEVEL_MIN_INTERVALS``,
    the window is widened evenly on both sides to that many, but never past
    ``[ga, gb)``, the reference's monotone tempo trend around the span
    (`_trend_spans`); where it is clipped it extends the other way. So a
    frame grid averages out, a one-beat segment of a ramp reads its own
    tempo, and no window reaches across a kink in the reference's tempo.
    """
    mid = (a + b) / 2.0
    if timing.exact_steps or len(timing) < 2:
        return float(timing.step_at(mid)[0])
    beats = timing.beats
    n_int = len(beats) - 1
    j0 = int(np.searchsorted(beats, a - MERGE_MS, side='left'))
    j1 = min(int(np.searchsorted(beats, b - MERGE_MS, side='left')), n_int)
    if j1 <= j0:
        k = int(np.searchsorted(beats, mid, side='right')) - 1
        if not 0 <= k < n_int:
            return math.nan
        j0, j1 = k, k + 1
    centre = (j0 + j1) / 2.0
    need = LEVEL_MIN_INTERVALS - (j1 - j0)
    lo_j = int(np.searchsorted(beats, ga - MERGE_MS, side='left'))
    hi_j = max(min(int(np.searchsorted(beats, gb - MERGE_MS, side='left')), n_int), j1)
    lo_j = min(lo_j, j0)
    if need > 0:
        e = (need + 1) // 2
        j0, j1 = j0 - e, j1 + e
        if j0 < lo_j:
            j0, j1 = lo_j, min(hi_j, j1 + (lo_j - j0))
        if j1 > hi_j:
            j0, j1 = max(lo_j, j0 - (j1 - hi_j)), hi_j
    return chain_beat_length(beats[j0:j1 + 1], centre - j0)


def _bar_terms(ref_bars: np.ndarray, ends: np.ndarray, pred_bars: np.ndarray, pred_in: np.ndarray,
               start_keep: np.ndarray, same_keep: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Bar matches within 20 ms: each reference bar's 'same' flag, and the start, same and precision shares.

    A bar start matches a predicted one within 20 ms. A bar is the same when
    its start and end both match and no predicted bar start lies between.
    Precision is the share of ``pred_in`` (the predicted bar starts in the
    range) matching a reference bar start. Shares are NaN where nothing is
    scored. The caller passes the predicted bars as they are for the
    reported values, or moved by -o for the ``_o`` variants.
    """
    starts_ok = _within(ref_bars, pred_bars, DECIDING_TAU)
    ends_ok = _within(ends, pred_bars, DECIDING_TAU) & np.isfinite(ends)
    inside = np.zeros(len(ref_bars), dtype=bool)
    if len(pred_bars):
        a_idx = np.searchsorted(pred_bars, ref_bars + DECIDING_TAU + 1e-9, side='right')
        b_idx = np.searchsorted(pred_bars, np.nan_to_num(ends, nan=np.inf) - DECIDING_TAU - 1e-9, side='left')
        inside = b_idx > a_idx
    same = starts_ok & ends_ok & ~inside
    start_share = float(starts_ok[start_keep].mean()) if start_keep.any() else math.nan
    same_share = float(same[same_keep].mean()) if same_keep.any() else math.nan
    if len(pred_in):
        precision = float(_within(pred_in, ref_bars, DECIDING_TAU).mean())
    elif start_keep.any():
        precision = 0.0 if len(pred_bars) else math.nan
    else:
        precision = math.nan
    return same, (start_share, same_share, precision)


def _within(times: np.ndarray, targets: np.ndarray, tol) -> np.ndarray:
    """Whether each time has a target within ``tol`` (scalar or per time)."""
    t = np.asarray(times, dtype=np.float64)
    g = np.asarray(targets, dtype=np.float64)
    if not len(g) or not len(t):
        return np.zeros(t.shape, dtype=bool)
    d, _ = nearest(t, np.sort(g))
    return np.isfinite(d) & (np.abs(d) <= np.asarray(tol) + 1e-9)


def _ref_segments(ref: Reference) -> list[tuple[float, float, float]]:
    """``(start, end, beat_ms)`` of each plausible reference segment, or the tempo runs of a beat list."""
    if ref.segments:
        segs = list(ref.segments)
        out = []
        for k, s in enumerate(segs):
            end = segs[k + 1].start_ms if k + 1 < len(segs) else math.inf
            if s.plausible:
                out.append((s.start_ms, end, s.beat_ms))
        if out and out[0][0] > ref.lo_ms:
            out[0] = (-math.inf, out[0][1], out[0][2])
        return out
    return [(a, b, s) for a, b, s in tempo_spans(ref.timing)]


def _ref_red_lines(ref: Reference) -> np.ndarray:
    if ref.segments:
        return np.array([s.start_ms for s in ref.segments if s.plausible], dtype=np.float64)
    return np.array([a for a, _, _ in tempo_spans(ref.timing)], dtype=np.float64)


def _resets(ref: Reference) -> np.ndarray:
    """Same-tempo red lines off the previous segment's notated grid by more than ``MERGE_MS``."""
    if not ref.segments:
        return np.zeros(0)
    out = []
    segs = [s for s in ref.segments if s.plausible]
    for a, b in zip(segs, segs[1:]):
        if abs(b.beat_ms / a.beat_ms - 1.0) > SAME_TEMPO:
            continue
        pos = (b.start_ms - a.start_ms) / a.beat_ms
        if abs(pos - round(pos)) * a.beat_ms > MERGE_MS:
            out.append(b.start_ms)
    return np.array(out, dtype=np.float64)


def _pred_classes(p: _Pred, ref_timing: Timing, ref_beats: np.ndarray) -> list[str]:
    """The predicted class at each reference notated beat, '' where none."""
    if not len(ref_beats) or not len(p.timing):
        return [''] * len(ref_beats)
    b, step = ref_timing.beats, ref_timing.step_ms
    nxt = np.append(b[1:], np.inf)
    mid = b[ref_beats] + 0.5 * np.minimum(step[ref_beats], nxt[ref_beats] - b[ref_beats])
    pt = p.timing
    jj = np.searchsorted(pt.beats, mid, side='right') - 1
    covered = (jj >= 0) & (mid <= pt.beats[np.maximum(jj, 0)] + np.nan_to_num(pt.step_ms[np.maximum(jj, 0)]))
    if p.subdivision is not None and len(p.subdivision) == len(pt.beats):
        return [_class_of_probabilities(p.subdivision[k]) if c else '' for k, c in zip(jj, covered)]
    if p.heads is not None and len(p.heads):
        pbeats, _, pcls = subdivision_classes(pt, p.heads)
        lookup = dict(zip(pbeats.tolist(), pcls.tolist()))
        return [lookup.get(int(k), '') if c else '' for k, c in zip(jj, covered)]
    return [''] * len(ref_beats)
