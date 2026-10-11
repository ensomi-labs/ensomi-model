"""Defect injections at controlled, seeded doses (harness v0 development families).

D1 to D10 follow the families fixed on 2026-10-03, with the review's amendments
(D1 dosed by run length and run count, D2 with a seconds variant, D3 split by
hand relation with lanes 0-1 one hand and 2-3 the other); F1 to F7 are the
review's added families. F8 (real rejected outputs) is not scorable here: its
conditions and grids are not reconstructed.

Every injection works on the round-tripped source chart and its grid (the grid
the condition gives; the harness knows it, the evaluator does not read it) and
returns objects; the harness writes them to ``.osu``, parses them back, checks
legality and discards illegal results. A dose given in ms is a property of the
injection; no evaluator threshold is involved. Fraction doses round half up
(``floor(f * n + 0.5)``), so a small chart can receive no change; such no-ops
are counted, not hidden. Families that edit one object at a time round every
new time to integer ms (``written_ms``) before checking that the lane is free,
so the check holds for the file the harness writes.

Families that move groups of objects build legal charts where a legal one
exists, rather than leave the harness to discard them: D6 swaps lanes only
between instants no right-hand hold crosses; D7 redraws a bar's permutation
(up to ``TRIES`` times) or moves on to another bar; the copy families (D10
loop, F1, F2) place each copied block only where the result stays legal and
inside the song. The edits made against the edits requested are recorded in
``Objects.note`` (``requested``, ``done``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np

from ..beats import BeatGrid
from ..case import Chart
from ..osu_text import written_ms
from .arrays import UNIT_BARS, Objects, bar_fraction, canonical_beat_at, object_units, rows, time_at_bar_fraction
from .chain import continuation_cut
from .descriptions import next_heads, run_lengths
from .trivial import star_band

WHOLE, CONTINUATION = 'whole', 'continuation'
NOT_APPLICABLE_HEADS = 'not applicable: changes head times, which a skeleton fixes'
TRIES = 20


@dataclass
class Source:
    """The chart an injection starts from, with what some families need besides it."""
    chart: Chart
    grid: BeatGrid
    star: float
    mode_rate: Callable[[int], float | None] = lambda key: None
    donor: tuple[Chart, BeatGrid] | None = None
    span: tuple[float, float] | None = None
    objects: Objects = field(init=False)

    def __post_init__(self):
        self.objects = Objects.from_chart(self.chart)


def _k(fraction: float, n: int) -> int:
    return int(math.floor(fraction * n + 0.5))


def _members(row: np.ndarray, n_rows: int) -> list[np.ndarray]:
    order = np.argsort(row, kind='stable')
    cuts = np.searchsorted(row[order], np.arange(n_rows + 1))
    return [order[cuts[r]:cuts[r + 1]] for r in range(n_rows)]


def _end(o: Objects, i: int) -> float:
    return float(o.end[i]) if o.hold[i] else float(o.start[i])


def copy_bars(o: Objects, grid: BeatGrid, first: int, n_bars: int, dest: int, dest_grid: BeatGrid) -> Objects:
    """Objects whose head lies in bars ``first..first+n_bars-1``, moved bar for bar to start at ``dest``.

    Positions map by the fraction of the bar elapsed. A long note whose release
    falls at or after the end of the copied block becomes a tap, so a copy never
    reaches into what follows it.
    """
    bar, frac = bar_fraction(grid, o.start)
    sel = (bar >= first) & (bar < first + n_bars)
    if not sel.any():
        return Objects(np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool),
                       dict(bar_offset=np.zeros(0, dtype=np.int64)))
    start = time_at_bar_fraction(dest_grid, dest + bar[sel] - first, frac[sel])
    hold, end = o.hold[sel].copy(), start.copy()
    if hold.any():
        rbar, rfrac = bar_fraction(grid, o.end[sel][hold])
        inside = rbar - first < n_bars
        released = time_at_bar_fraction(dest_grid, dest + rbar - first, rfrac)
        end_h = np.where(inside, released, start[hold])
        hold_idx = np.flatnonzero(hold)
        end[hold_idx] = end_h
        hold[hold_idx[~inside]] = False
    out = Objects(start, np.where(hold, end, start), o.lane[sel].copy(), hold).written()
    out.note = dict(bar_offset=bar[sel] - first)
    return out


def place_blocks(o: Objects, bar: np.ndarray, blocks, span, limit: int | None = None) -> tuple[Objects, int]:
    """Replace bars by copied blocks one at a time, keeping each only if the chart stays legal.

    ``blocks`` yields ``(first_bar, n_bars, copy)``: the objects whose head lies
    in those bars (``bar`` gives each object's bar) are replaced by ``copy``.
    Stops after ``limit`` blocks are placed. Returns the chart and the number
    of blocks placed.
    """
    cur, cur_bar, done = o.copy(), bar.copy(), 0
    for first, n_bars, block in blocks:
        if limit is not None and done >= limit:
            break
        keep = (cur_bar < first) | (cur_bar >= first + n_bars)
        trial = Objects.concat([cur.take(keep), block])
        if trial.legal(span):
            offset = block.note['bar_offset'] if block.note else np.zeros(len(block), dtype=np.int64)
            cur, cur_bar = trial, np.concatenate([cur_bar[keep], first + np.asarray(offset, dtype=np.int64)])
            done += 1
    return cur, done


def head_bars(o: Objects, grid: BeatGrid) -> np.ndarray:
    return grid.locate(o.start)['bar'].astype(np.int64)


def continuation_bars(o: Objects, grid: BeatGrid) -> tuple[int, int, int]:
    """First bar, last bar and first scored bar of a continuation: the first 25% of bars are given."""
    bar = head_bars(o, grid)
    b0, b1 = int(bar.min()), int(bar.max())
    return b0, b1, b0 + (b1 - b0 + 1) // 4


# ---- families ------------------------------------------------------------------------------

def anchor(src: Source, dose, rng) -> Objects:
    """D1: new same-lane runs; dose = (run-length multiplier, run-count multiplier) of the chart's own runs.

    The chart's runs of length 2 or more give a count C0 and mean length L0
    (1 and 2 when it has none). round(mC * C0) new runs of round(mL * L0) rows
    are written by moving, in each row of a random window, one object onto a
    random lane when that lane is free. Times are kept, so head counts are.
    """
    m_len, m_count = dose
    o = src.objects.copy()
    times, row = rows(o.start)
    n_rows = len(times)
    if n_rows < 2:
        return o
    runs = run_lengths(o)
    long = runs[runs >= 2]
    c0, l0 = (len(long), float(long.mean())) if len(long) else (1, 2.0)
    n_new, length = max(1, _k(m_count, c0)), int(min(n_rows, max(2, math.floor(m_len * l0 + 0.5))))
    members = _members(row, n_rows)
    for _ in range(n_new):
        r0, lane = int(rng.integers(0, n_rows - length + 1)), int(rng.integers(0, 4))
        for r in range(r0, r0 + length):
            idx = members[r]
            if (o.lane[idx] == lane).any():
                continue
            for i in rng.permutation(idx):
                if o.free(lane, o.start[i], _end(o, i), skip=int(i)):
                    o.lane[i] = lane
                    break
    return o


def short_holds(unit: str):
    def apply(src: Source, fraction, rng) -> Objects:
        """D2: a fraction of taps become holds of 1/16 to 1/8 canonical beat (or 10 to 40 ms)."""
        o = src.objects.copy()
        taps = np.flatnonzero(~o.hold)
        need, done = _k(fraction, len(taps)), 0
        beat = canonical_beat_at(src.grid, o.start)
        for i in rng.permutation(taps):
            if done >= need:
                break
            length = rng.uniform(1 / 16, 1 / 8) * beat[i] if unit == 'beats' else rng.uniform(10.0, 40.0)
            end = max(written_ms(o.start[i] + length), o.start[i] + 1.0)
            if o.free(o.lane[i], o.start[i], end, skip=int(i)):
                o.hold[i], o.end[i] = True, end
                done += 1
        return o
    return apply


def release_crowding(relation: str):
    def apply(src: Source, fraction, rng) -> Objects:
        """D3: a fraction of releases move to 5-40 ms before the next head on the partner lanes."""
        o = src.objects.copy()
        holds = np.flatnonzero(o.hold)
        need, done = _k(fraction, len(holds)), 0
        for i in rng.permutation(holds):
            if done >= need:
                break
            nxt = next_heads(o, int(o.lane[i]), relation, [o.end[i]], inclusive=True)[0]
            if not math.isfinite(nxt):
                continue
            new = written_ms(nxt - rng.uniform(5.0, 40.0))
            if new > o.start[i] and o.free(o.lane[i], o.start[i], new, skip=int(i)):
                o.end[i] = new
                done += 1
        return o
    return apply


def ln_concentration(src: Source, fraction, rng) -> Objects:
    """D4: LN status moves from the least to the most dense units; heads and LN count kept.

    The holds of the least dense placement units (a fraction of all holds)
    become taps; as many taps of the densest units become holds with the
    removed lengths in canonical beats, where the lane is free.
    """
    o = src.objects.copy()
    unit, lay = object_units(o, src.grid)
    if unit is None:
        return o
    inside = unit >= 0
    if not inside.any():
        return o
    density = np.bincount(unit[inside], minlength=len(lay['first_bar']))
    dens = np.where(inside, density[np.clip(unit, 0, None)], -1)
    holds = np.flatnonzero(o.hold & inside)
    m = _k(fraction, len(holds))
    if m == 0:
        return o
    jitter = rng.random(len(o))
    take = holds[np.lexsort((jitter[holds], dens[holds]))][:m]
    beat = canonical_beat_at(src.grid, o.start)
    lengths = rng.permutation((o.end[take] - o.start[take]) / beat[take])
    o.hold[take], o.end[take] = False, o.start[take]
    taps = np.flatnonzero(~o.hold & inside)
    taps = taps[~np.isin(taps, take)]
    order = taps[np.lexsort((jitter[taps], -dens[taps]))]
    j = 0
    for length in lengths:
        while j < len(order):
            i = int(order[j])
            j += 1
            end = max(written_ms(o.start[i] + length * beat[i]), o.start[i] + 1.0)
            if o.free(o.lane[i], o.start[i], end, skip=i):
                o.hold[i], o.end[i] = True, end
                break
    return o


def jitter(src: Source, sigma, rng) -> Objects:
    """D5: Gaussian jitter of every head, sigma in ms; releases stay where they were."""
    o = src.objects.copy()
    o.start = o.start + rng.normal(0.0, float(sigma), len(o))
    o.end = np.where(o.hold, o.end, o.start)
    return o


def hand_swap(src: Source, fraction, rng) -> Objects:
    """D6: lanes 2 and 3 swapped inside a fraction of placement units (not a mirror).

    Each chosen unit's edges move forward to the first instant no lane-2 or
    lane-3 hold crosses, and every lane-2 and lane-3 object with its head
    between the moved edges is swapped. No hold then crosses an edge, so the
    swap keeps a legal chart legal.
    """
    o = src.objects.copy()
    unit, lay = object_units(o, src.grid)
    if unit is None or not len(lay['first_bar']):
        return o
    n = len(lay['first_bar'])
    chosen = rng.choice(n, size=_k(fraction, n), replace=False)
    right = o.lane >= 2
    hs, he = o.start[right & o.hold], o.end[right & o.hold]

    def clean(c: float) -> float:
        while True:
            cover = (hs < c) & (he >= c)
            if not cover.any():
                return c
            c = float(he[cover].max()) + 0.5

    sel = np.zeros(len(o), dtype=bool)
    for u in chosen:
        a, b = clean(float(lay['start_ms'][u])), clean(float(lay['end_ms'][u]))
        sel |= right & (o.start >= a) & (o.start < b)
    o.lane[sel] = 5 - o.lane[sel]
    o.note = dict(requested=int(len(chosen)), done=int(len(chosen)))
    return o


def row_shuffle(src: Source, fraction, rng) -> Objects:
    """D7: inside a fraction of bars with two rows or more, row contents are permuted; row times kept.

    A row's objects move together, holds with their rows. A permutation that
    would make the chart illegal is redrawn, up to ``TRIES`` times; a bar
    with no legal draw is left alone and another eligible bar is taken.
    """
    o = src.objects.copy()
    bar = head_bars(o, src.grid)
    times, row = rows(o.start)
    eligible = [int(b) for b in np.unique(bar) if len(np.unique(row[bar == b])) >= 2]
    need, done = _k(fraction, len(eligible)), 0
    for b in rng.permutation(np.array(eligible, dtype=np.int64)):
        if done >= need:
            break
        in_bar = np.flatnonzero(bar == b)
        r_in = np.unique(row[in_bar])
        for _ in range(TRIES):
            target = dict(zip(r_in.tolist(), times[r_in[rng.permutation(len(r_in))]].tolist()))
            shift = np.array([target[int(row[i])] for i in in_bar]) - o.start[in_bar]
            trial = o.copy()
            trial.start[in_bar] += shift
            trial.end[in_bar] = np.where(o.hold[in_bar], o.end[in_bar] + shift, trial.start[in_bar])
            if trial.legal(src.span):
                o = trial
                done += 1
                break
    o.note = dict(requested=int(need), done=int(done))
    return o


def triplet_substitution(src: Source, fraction, rng) -> Objects:
    """D8: a fraction of heads on canonical 1/4 positions move to the nearest 1/3 in the same beat."""
    o = src.objects.copy()
    loc = src.grid.locate(o.start)
    cand = np.flatnonzero(loc['snap'] == 4)
    need, done = _k(fraction, len(cand)), 0
    for i in rng.permutation(cand):
        if done >= need:
            break
        snapped = round(float(loc['beat'][i]) * 4) / 4
        whole = math.floor(snapped + 1e-9)
        target = whole + (1 / 3 if abs(snapped - whole - 0.25) < 1e-6 else 2 / 3)
        new = written_ms(src.grid.time_of(int(loc['segment'][i]), target))
        end = o.end[i] + new - o.start[i] if o.hold[i] else new
        if o.free(o.lane[i], new, end, skip=int(i)):
            o.start[i], o.end[i] = new, end
            done += 1
    return o


def density_drift(src: Source, factor, rng) -> Objects:
    """D9: heads thinned (random removal) or thickened (taps added at free lanes of rows and midpoints)."""
    o = src.objects.copy()
    n = len(o)
    if factor < 1:
        keep = np.sort(rng.permutation(n)[:n - _k(1 - factor, n)])
        return o.take(keep)
    need = _k(factor - 1, n)
    points = np.unique(o.start)
    times = points.copy()
    for _ in range(4):
        if sum(int(o.free_at(lane, times).sum()) for lane in range(4)) >= need:
            break
        gaps = np.diff(points)
        mids = (points[:-1] + points[1:])[gaps >= 2] / 2
        times = np.union1d(times, [written_ms(m) for m in mids])
        points = np.union1d(points, mids)
    cand_t = [times[o.free_at(lane, times)] for lane in range(4)]
    t = np.concatenate(cand_t)
    lane = np.concatenate([np.full(len(c), k) for k, c in enumerate(cand_t)])
    pick = rng.choice(len(t), size=min(need, len(t)), replace=False)
    extra = Objects(t[pick], t[pick].copy(), lane[pick].astype(np.int64), np.zeros(len(pick), dtype=bool))
    return Objects.concat([o, extra])


def degenerate(src: Source, variant, rng) -> Objects:
    """D10: one lane (one tap per row); every row a four-note chord; the first quarter of bars looped.

    The loop copies the first quarter of the bars over each following block of
    as many bars, block by block where the chart stays legal.
    """
    o = src.objects
    times, _ = rows(o.start)
    if variant == 'one_lane':
        return Objects(times.copy(), times.copy(), np.full(len(times), int(rng.integers(0, 4))),
                       np.zeros(len(times), dtype=bool))
    if variant == 'all_quad':
        return Objects(np.repeat(times, 4), np.repeat(times, 4), np.tile(np.arange(4), len(times)),
                       np.zeros(4 * len(times), dtype=bool))
    b0, b1, _ = continuation_bars(o, src.grid)
    q = max(1, (b1 - b0 + 1) // 4)
    bar = head_bars(o, src.grid)
    starts = list(range(b0 + q, b1 + 1, q))
    blocks = ((s0, min(q, b1 - s0 + 1), copy_bars(o, src.grid, b0, min(q, b1 - s0 + 1), s0, src.grid))
              for s0 in starts)
    out, done = place_blocks(o, bar, blocks, src.span)
    out.note = dict(requested=len(starts), done=done)
    return out


def repeat_bar(src: Source, fraction, rng) -> Objects:
    """F1: one random non-empty bar copied over a fraction of the other bars, where the chart stays legal."""
    o = src.objects
    bar = head_bars(o, src.grid)
    b0, b1 = int(bar.min()), int(bar.max())
    source_bar = int(rng.choice(np.unique(bar)))
    others = np.array([b for b in range(b0, b1 + 1) if b != source_bar], dtype=np.int64)
    need = _k(fraction, len(others))
    if not len(others) or not need:
        out = o.copy()
        out.note = dict(requested=int(need), done=0)
        return out
    blocks = ((int(t), 1, copy_bars(o, src.grid, source_bar, 1, int(t), src.grid)) for t in rng.permutation(others))
    out, done = place_blocks(o, bar, blocks, src.span, limit=need)
    out.note = dict(requested=int(need), done=int(done))
    return out


def mode_chart(src: Source, dose, rng) -> Objects:
    """F2 (per song): every placement unit replaced, where the chart stays legal, by the chart's unit whose
    head rate is closest to the fit-split modal head rate at its key."""
    o = src.objects
    unit, lay = object_units(o, src.grid)
    mode = src.mode_rate(int(star_band(src.star)))
    if unit is None or not len(lay['first_bar']) or mode is None:
        out = o.copy()
        out.note = dict(requested=0, done=0)
        return out
    n = len(lay['first_bar'])
    counts = np.bincount(unit[unit >= 0], minlength=n)
    seconds = (lay['end_ms'] - lay['start_ms']) / 1000.0
    gap = np.abs(np.log(np.maximum(counts, 0.5) / seconds) - np.log(mode))
    best = int(rng.choice(np.flatnonzero(gap == gap.min())))
    bar = head_bars(o, src.grid)
    blocks = ((int(f), UNIT_BARS, copy_bars(o, src.grid, int(lay['first_bar'][best]), UNIT_BARS, int(f), src.grid))
              for f in lay['first_bar'])
    out, done = place_blocks(o, bar, blocks, src.span)
    out.note = dict(requested=n, done=done)
    return out


def never_ln(src: Source, dose, rng) -> Objects:
    """F3 (per song): every hold becomes a tap at its head."""
    o = src.objects.copy()
    o.hold[:] = False
    o.end = o.start.copy()
    return o


def context_break(src: Source, dose, rng) -> Objects | None:
    """F4: the scored part of a continuation replaced by the scored part of another chart at the same key.

    The donor's bars from the bar of its own continuation cut are copied bar for
    bar onto the source's bars from the bar of the source's cut; the harness
    then keeps the source's given part and the copy after the cut.
    """
    if src.donor is None:
        return None
    o = src.objects
    donor_chart, donor_grid = src.donor
    d = Objects.from_chart(donor_chart)
    b1 = int(head_bars(o, src.grid).max())
    db1 = int(head_bars(d, donor_grid).max())
    s0 = int(src.grid.locate([continuation_cut(src.chart)])['bar'][0])
    d0 = int(donor_grid.locate([continuation_cut(donor_chart)])['bar'][0])
    n = min(b1 - s0 + 1, db1 - d0 + 1)
    if n <= 0:
        return None
    return Objects.concat([o.take(head_bars(o, src.grid) < s0), copy_bars(d, donor_grid, d0, n, s0, src.grid)])


def chord_merge(src: Source, fraction, rng) -> Objects:
    """F5: a fraction of objects move into the nearest other row of their bar where the lane is free."""
    o = src.objects.copy()
    bar = head_bars(o, src.grid)
    times, row = rows(o.start)
    row_bar = np.zeros(len(times), dtype=np.int64)
    row_bar[row] = bar
    need, done = _k(fraction, len(o)), 0
    for i in rng.permutation(len(o)):
        if done >= need:
            break
        cand = times[(row_bar == bar[i]) & (times != o.start[i])]
        length = o.end[i] - o.start[i] if o.hold[i] else 0.0
        for t in cand[np.argsort(np.abs(cand - o.start[i]), kind='stable')]:
            if o.free(o.lane[i], t, t + length, skip=int(i)):
                o.start[i], o.end[i] = t, t + length
                done += 1
                break
    o.end = np.where(o.hold, o.end, o.start)
    return o


def offgrid_release(src: Source, fraction, rng) -> Objects:
    """F6: a fraction of releases moved by a uniform amount within +-1/8 canonical beat."""
    o = src.objects.copy()
    holds = np.flatnonzero(o.hold)
    need, done = _k(fraction, len(holds)), 0
    beat = canonical_beat_at(src.grid, o.start)
    for i in rng.permutation(holds):
        if done >= need:
            break
        for _ in range(5):
            new = written_ms(o.end[i] + rng.uniform(-1.0, 1.0) * beat[i] / 8)
            if new > o.start[i] and o.free(o.lane[i], o.start[i], new, skip=int(i)):
                o.end[i] = new
                done += 1
                break
    return o


def near_duplicate(src: Source, fraction, rng) -> Objects:
    """F7: in a fraction of chords, one note moves 5-19 ms earlier or later (not a chord any more)."""
    o = src.objects.copy()
    times, row = rows(o.start)
    chord_rows = np.flatnonzero(np.bincount(row) >= 2)
    if not len(chord_rows):
        return o
    members = _members(row, len(times))
    for r in rng.choice(chord_rows, size=_k(fraction, len(chord_rows)), replace=False):
        i = int(rng.choice(members[r]))
        delta = rng.uniform(5.0, 19.0) * (1 if rng.random() < 0.5 else -1)
        new = written_ms(o.start[i] + delta)
        end = o.end[i] + new - o.start[i] if o.hold[i] else new
        if o.free(o.lane[i], new, end, skip=i):
            o.start[i], o.end[i] = new, end
    return o


# ---- registry ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Family:
    name: str
    title: str
    doses: tuple
    description: str | dict
    apply: Callable[[Source, Any, np.random.Generator], Objects | None]
    heads_unchanged: bool = False
    skeleton: str = 'applicable'
    scopes: tuple[str, ...] = (WHOLE, CONTINUATION)
    population: tuple[float, ...] = ()
    dose_unit: str = ''
    in_place: bool = True

    def description_of(self, dose) -> str:
        return self.description[dose] if isinstance(self.description, dict) else self.description


def dose_label(dose) -> str:
    if isinstance(dose, tuple):
        return f'Lx{dose[0]:g},Cx{dose[1]:g}'
    return f'{dose:g}' if isinstance(dose, (int, float)) else str(dose)


POPULATION_FRACTIONS = (0.1, 0.25, 0.5, 1.0)

FAMILIES: tuple[Family, ...] = (
    Family('D1', 'anchor runs (length x count)', tuple((L, C) for L in (1, 2, 4, 8) for C in (1, 2, 4)),
           'run_length', anchor, heads_unchanged=True, dose_unit='multiples of own run length, run count'),
    Family('D2', 'short holds <= 1/8 canonical beat', (0.05, 0.10, 0.20, 0.40), 'short_hold_beats',
           short_holds('beats'), heads_unchanged=True, dose_unit='fraction of taps'),
    Family('D2s', 'short holds 10-40 ms', (0.05, 0.10, 0.20, 0.40), 'short_hold_ms', short_holds('ms'),
           heads_unchanged=True, dose_unit='fraction of taps'),
    Family('D3same', 'release crowding, same hand', (0.05, 0.10, 0.20), 'crowd_same_hand',
           release_crowding('same'), heads_unchanged=True, dose_unit='fraction of releases'),
    Family('D3other', 'release crowding, other hand', (0.05, 0.10, 0.20), 'crowd_other_hand',
           release_crowding('other'), heads_unchanged=True, dose_unit='fraction of releases'),
    Family('D4', 'LN concentration on dense units', (0.25, 0.5, 1.0), 'ln_density_corr', ln_concentration,
           heads_unchanged=True, dose_unit='fraction of holds moved'),
    Family('D5', 'head jitter', (2, 5, 10, 20), 'head_residual_ms', jitter, skeleton=NOT_APPLICABLE_HEADS,
           dose_unit='sigma ms'),
    Family('D6', 'lanes 2 and 3 swapped', (0.25, 0.5, 1.0), 'adjacent_step_share', hand_swap,
           heads_unchanged=True, dose_unit='fraction of placement units'),
    Family('D7', 'row shuffle inside bars', (0.25, 0.5, 1.0), 'jack_share', row_shuffle, heads_unchanged=True,
           skeleton='not applicable: moves chord sizes between given head times', dose_unit='fraction of bars'),
    Family('D8', '1/4 moved to 1/3', (0.1, 0.25, 0.5, 1.0), 'triplet_share', triplet_substitution,
           heads_unchanged=True, skeleton=NOT_APPLICABLE_HEADS, dose_unit='fraction of 1/4 heads'),
    Family('D9', 'density drift', (0.5, 0.67, 0.8, 1.25, 1.5, 2.0), 'density', density_drift,
           skeleton=NOT_APPLICABLE_HEADS, dose_unit='head-count factor', in_place=False),
    Family('D10', 'degenerate', ('one_lane', 'all_quad', 'loop_first_quarter'),
           {'one_lane': 'max_lane_share', 'all_quad': 'mean_chord', 'loop_first_quarter': 'bar_repeat_share'},
           degenerate, skeleton=NOT_APPLICABLE_HEADS, dose_unit='variant', in_place=False),
    Family('F1', 'graded repetition of one bar', (0.1, 0.25, 0.5, 1.0), 'bar_repeat_share', repeat_bar,
           skeleton=NOT_APPLICABLE_HEADS, dose_unit='fraction of bars', in_place=False),
    Family('F2', 'population collapse (mode chart)', ('replaced',), 'unit_count_cv', mode_chart,
           skeleton=NOT_APPLICABLE_HEADS, population=POPULATION_FRACTIONS, dose_unit='fraction of songs',
           in_place=False),
    Family('F3', 'never writing LN', ('replaced',), 'ln_share', never_ln, heads_unchanged=True,
           population=POPULATION_FRACTIONS, dose_unit='fraction of songs'),
    Family('F4', 'context break under continuation', ('splice',), 'density_log_ratio', context_break,
           skeleton='not applicable: the skeleton fixes the scored heads', scopes=(CONTINUATION,),
           dose_unit='whole scored part', in_place=False),
    Family('F5', 'graded chord-size shift (merge)', (0.05, 0.1, 0.2, 0.4), 'mean_chord', chord_merge,
           heads_unchanged=True, skeleton=NOT_APPLICABLE_HEADS, dose_unit='fraction of objects'),
    Family('F6', 'off-grid releases', (0.1, 0.25, 0.5, 1.0), 'release_residual_ms', offgrid_release,
           heads_unchanged=True, dose_unit='fraction of releases'),
    Family('F7', 'near-duplicate heads', (0.1, 0.25, 0.5, 1.0), 'near_duplicate_share', near_duplicate,
           skeleton=NOT_APPLICABLE_HEADS, dose_unit='fraction of chords'),
)

NOT_SCORABLE = {'F8': 'real rejected outputs: not scorable, conditions and grids not reconstructed'}


def family_spec() -> list[dict]:
    """What defines the family set, for its hash in the receipt."""
    return [dict(name=f.name, title=f.title, doses=[dose_label(d) for d in f.doses],
                 description=f.description, apply=f.apply.__qualname__, heads_unchanged=f.heads_unchanged,
                 skeleton=f.skeleton, scopes=list(f.scopes), population=list(f.population), in_place=f.in_place)
            for f in FAMILIES]
