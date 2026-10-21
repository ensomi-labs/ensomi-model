"""Mapper-to-mapper timing spread on same-audio pairs, from labels only.

The question (research notes ``mapper-spread-20261021``): how far apart are two
osu! mappers' timings of one audio file, per quantity, once copied timing is
set aside? Nothing here opens audio or runs a model. The reader, the canonical
lines and the measures are the evaluation's own (`ensomi_n.labels`,
`ensomi_n.timing`, `ensomi_n.eval.measures`, `ensomi_n.eval.strata`).

**Pairs.** Two sets form a pair when they name one audio file (same sha256),
have different beatmap-set ids and share no creator, as the census pairs them
(`ensomi_n.eval.census`). The older set (lower set id) is the reference ``a``.
Both are compared over the shared range, from the later of the two first heads
to the earlier of the two last heads; a pair needs 20 s of it and 32 scored
lines each way.

**Copy rules**, nested, each excluding more than the one before:

- ``census``: identical plausible red lines within 1 ms (`strata.same_timing`).
  Reported to reconcile with the census; every such pair is also C1.
- ``C1``, the same grid: in both directions, 0.99 or more of the head weight on
  canonical lines within 1 ms of the other's, 0.99 or more of the duration at
  the same notated tempo (within 0.5 %), and 0.99 or more of the bar starts
  matched within 20 ms (*bars* below).
- ``C2``, the same grid up to one shift: C1, or the same with lines and bar
  starts compared after removing the pair's offset. It adds re-offset copies,
  and also independent timings that differ only in offset, which labels cannot
  tell apart.
- ``C3``: C2, or one exact beat length (within 1e-6 ms) shared at a tempo more
  than 0.01 BPM from every integer.

View ``C0`` keeps every pair; view ``Ck`` drops the pairs rule ``Ck`` flags.

**Per direction** (reference x, other mapper y), on the shared range:

- *lines*: x's canonical lines with the signed error to y's nearest canonical
  line, head-weighted, as BL-τ reads them (`measures.score`, free bars refined
  there). ``o`` is the head-weighted median error over the whole shared range;
  the offset-removed error is ``e - o``.
- *level*: the duration of x's segments at which y's notated tempo is the
  same, an octave, a family ratio or other (`measures.level_class`).
- *bars*: x's bar starts where y's notated tempo is the same (within 0.5 %).
  A start is matched when y has a bar start within 20 ms, raw or after moving
  it by ``o``. An unmatched start's displacement is counted in x's notated
  beats: a whole number within 20 ms, else fractional. A matched bar whose end
  y also starts a bar at, with no y bar start between, has the same length
  (meter agreement).
- *tempo changes and resets*: x's ``tempo`` and ``reset`` lines
  (`strata.classify_lines`) in the range, matched by a y line within half a
  canonical beat (``half``) or two canonical beats (``near``) of the shorter
  of the two beat lengths around x's line. Strict matching takes a y line of
  the same kind (``tempo`` or ``first`` for a change, ``reset`` for a reset);
  loose matching any plausible y line. Also the evaluation's long-change
  terms and 20 ms reset recall (`measures.score`).

A pair's strata are the census strata of its two sets
(`strata.recording_strata`). A stratum's cell restricts lines, level, bars and
events to the union of the two sets' spans; the offset stays the whole-range
one.

**Pooling and intervals.** In a cell every recording weighs 1, split evenly
over its pairs there, then evenly over the pair's two directions (lines by head
weight within a direction, bars evenly). Tempo changes and resets pool event
counts with the pair weights. Intervals are 95 % percentile bootstraps over
recordings.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..timing import MERGE_MS, Reference, Timing, canonical_beat_ms
from .measures import LEVEL_CLASSES, LEVEL_TOLERANCE, level_class, nearest, score
from .strata import STRATA, recording_strata, same_timing

__all__ = ['candidate_pairs', 'unsealed', 'shape_flags', 'shared_exact_tempo', 'direction', 'compare', 'run',
           'summarize', 'markdown', 'write']

TAUS_MS = (2.0, 5.0, 10.0, 20.0)
QUANTILES = (0.5, 0.75, 0.9, 0.95)
COPY_MS = 1.0
COPY_SHARE = 0.99
EXACT_BEAT_MS = 1e-6
ROUND_BPM = 0.01
FAST_BPM = 200.0
BAR_TOL_MS = 20.0
HALF_BEAT = 0.5
NEAR_BEATS = 2.0
MIN_SHARED_MS = 20000.0
MIN_LINES = 32
PRED_MARGIN_MS = 6000.0
BIN_MS = 0.05
MAX_MS = 400.0
VIEWS = ('C0', 'C1', 'C2', 'C3')
BAR_CLASSES = ('1', '2', '3', '4+', 'fractional')
EVENT_KEYS = ('n', 'strict_half', 'loose_half', 'strict_near', 'loose_near')
ROBUST = 1.4826


# ---------------------------------------------------------------- pairs and copies

def _order(u) -> tuple:
    return (0, int(u.set_id), u.key) if str(u.set_id).isdigit() else (1, 0, u.key)


def candidate_pairs(units: Sequence) -> list[tuple]:
    """Pairs of units on one audio file (same non-empty sha256) with different set ids, older first.

    Creators are checked later, once the files are read (`run`).
    """
    by_sha: dict[str, list] = {}
    for u in units:
        if u.audio_sha256:
            by_sha.setdefault(u.audio_sha256, []).append(u)
    out = []
    for _, us in sorted(by_sha.items()):
        us = sorted(us, key=_order)
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a, b = us[i], us[j]
                if a.set_id and a.set_id == b.set_id:
                    continue
                out.append((a, b))
    return out


def unsealed(units: Sequence, exclusion: dict, sealed: Iterable[str]) -> tuple[list, dict]:
    """The units outside every sealed split and the sealed hard holdout, with the drop counts.

    ``exclusion`` is `census.load_exclusion`'s result. A unit goes when its split
    is sealed, or its key, set id, audio sha256 or recording is listed. The
    hard holdout's pool split ``std-taiko`` becomes ``std-taiko-design``.
    """
    sealed = set(sealed)
    kept, drops = [], {'sealed_split': 0, 'hard_holdout': 0}
    for u in units:
        if u.split in sealed:
            drops['sealed_split'] += 1
        elif (u.key in exclusion['keys'] or (u.set_id and u.set_id in exclusion['set_ids'])
              or (u.audio_sha256 and u.audio_sha256 in exclusion['audio_sha256'])
              or u.recording in exclusion['recordings']):
            drops['hard_holdout'] += 1
        else:
            if u.split == 'std-taiko':
                u.split = 'std-taiko-design'
            kept.append(u)
    return kept, drops


def shared_exact_tempo(a_segments, b_segments) -> bool:
    """Whether the two sets share an exact beat length (1e-6 ms) at a tempo more than 0.01 BPM from every integer."""
    bb = np.array([s.beat_ms for s in b_segments if s.plausible], dtype=np.float64)
    for s in a_segments:
        if not s.plausible or not len(bb):
            continue
        bpm = 60000.0 / s.beat_ms
        if abs(bpm - round(bpm)) <= ROUND_BPM:
            continue
        if np.any(np.abs(bb - s.beat_ms) <= EXACT_BEAT_MS):
            return True
    return False


def _share_within(d: dict, shift: float, tol: float = COPY_MS) -> float:
    w = d['w']
    total = w.sum()
    return float((w * (np.abs(d['e'] - shift) <= tol + 1e-9)).sum() / total) if total > 0 else math.nan


def _level_share(d: dict) -> float:
    total = sum(d['level'].values())
    return d['level'].get('correct', 0.0) / total if total > 0 else math.nan


def _bar_share(d: dict, key: str) -> float:
    n = d['bars']['n']
    return d['bars'][key] / n if n else 1.0


def shape_flags(dirs: Sequence[dict], a_segments, b_segments) -> dict:
    """The copy flags of a pair from its two whole-range directions (module docstring)."""
    same_level = all(_level_share(d) >= COPY_SHARE for d in dirs)
    c1 = same_level and all(_share_within(d, 0.0) >= COPY_SHARE and _bar_share(d, 'raw') >= COPY_SHARE
                            for d in dirs)
    c2 = c1 or (same_level and all(_share_within(d, d['o']) >= COPY_SHARE and _bar_share(d, 'o') >= COPY_SHARE
                                   for d in dirs))
    return dict(census=bool(same_timing(a_segments, b_segments)), c1=bool(c1), c2=bool(c2),
                c3=bool(c2 or shared_exact_tempo(a_segments, b_segments)))


# ---------------------------------------------------------------- one direction

def _in(t, lo: float, hi: float, spans) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    ok = (t >= lo) & (t <= hi)
    if spans is None:
        return ok
    m = np.zeros(t.shape, dtype=bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return ok & m


def _shared(ref: Reference, lo: float, hi: float) -> Reference:
    h = ref.heads
    if h is not None:
        h = h[(h >= lo) & (h <= hi)]
    return Reference.from_segments(ref.segments, h, lo_ms=lo, hi_ms=hi, key=ref.key, meta=ref.meta)


def _near(times, targets, tol) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    g = np.sort(np.asarray(targets, dtype=np.float64))
    if not len(t) or not len(g):
        return np.zeros(t.shape, dtype=bool)
    d, _ = nearest(t, g)
    return np.isfinite(d) & (np.abs(d) <= np.asarray(tol) + 1e-9)


def _bars(xt: Timing, yt: Timing, lo: float, hi: float, spans, o: float) -> dict:
    xb, yb = xt.bar_starts(), np.sort(yt.bar_starts())
    ends = np.append(xb[1:], np.nan)
    sx, sy = xt.step_at(xb), yt.step_at(xb)
    level = np.isfinite(sx) & np.isfinite(sy) & (np.abs(sy / np.where(sx > 0, sx, np.nan) - 1.0) <= LEVEL_TOLERANCE)
    inside = _in(xb, lo, hi, spans)
    use = inside & level
    start, end, step = xb[use], ends[use], sx[use]
    raw = _near(start, yb, BAR_TOL_MS)
    d_o, _ = nearest(start + o, yb)
    m_o = np.isfinite(d_o) & (np.abs(d_o) <= BAR_TOL_MS + 1e-9)
    disp = {c: 0 for c in BAR_CLASSES}
    for d, s in zip(d_o[~m_o], step[~m_o]):
        k = round(d / s) if math.isfinite(d) else 0
        if k != 0 and abs(d - k * s) <= BAR_TOL_MS:
            disp[str(min(abs(k), 4)) + ('+' if abs(k) >= 4 else '')] += 1
        else:
            disp['fractional'] += 1
    has_end = np.isfinite(end) & (end <= hi + MERGE_MS)
    end_ok = _near(np.nan_to_num(end + o, nan=-1e18), yb, BAR_TOL_MS) & has_end
    a_idx = np.searchsorted(yb, start + o + BAR_TOL_MS + 1e-9, side='right')
    b_idx = np.searchsorted(yb, np.nan_to_num(end + o, nan=np.inf) - BAR_TOL_MS - 1e-9, side='left')
    same = m_o & end_ok & ~(b_idx > a_idx)
    return dict(n=int(use.sum()), other_level=int((inside & ~level).sum()), raw=int(raw.sum()), o=int(m_o.sum()),
                with_end=int(has_end.sum()), matched_with_end=int((m_o & has_end).sum()), same=int(same.sum()),
                disp=disp)


def _events(x_lines, y_lines, kinds: tuple, strict: tuple, lo: float, hi: float, spans) -> dict:
    out = {k: 0 for k in EVENT_KEYS}
    y_strict = [ln.start_ms for ln in y_lines if ln.kind in strict]
    y_any = [ln.start_ms for ln in y_lines if ln.kind != 'implausible']
    prev = None
    for ln in x_lines:
        if ln.kind == 'implausible':
            continue
        if ln.kind in kinds and _in([ln.start_ms], lo, hi, spans)[0]:
            beat = float(canonical_beat_ms(min(ln.beat_ms, prev.beat_ms) if prev is not None else ln.beat_ms))
            out['n'] += 1
            for name, f in (('half', HALF_BEAT), ('near', NEAR_BEATS)):
                out[f'strict_{name}'] += int(_near([ln.start_ms], y_strict, f * beat)[0])
                out[f'loose_{name}'] += int(_near([ln.start_ms], y_any, f * beat)[0])
        prev = ln
    return out


def direction(x: Reference, y: Reference, x_lines, y_lines, lo: float, hi: float, spans=None,
              o: float | None = None) -> dict:
    """One direction of a pair, x as reference and y as the other mapper (module docstring).

    ``o`` is the whole-range offset; when None it is computed here, which is
    right only for ``spans`` None.
    """
    xr = _shared(x, lo, hi)
    yt = Timing.from_segments(y.segments, lo - PRED_MARGIN_MS, hi + PRED_MARGIN_MS)
    s = score(xr, yt, spans=spans)
    if o is None:
        o = s.offset_ms
    o = float(o) if math.isfinite(o) else 0.0
    t, e, w = s.beats['t'], s.beats['e'], s.beats['w']
    keep = (w > 0) & np.isfinite(e)
    inside = _in(t, lo, hi, spans) & np.isfinite(e)
    minutes = s.minutes if math.isfinite(s.minutes) else 0.0
    level = {c: (float(v) * minutes * 60000.0 if math.isfinite(v) else 0.0) for c, v in s.level_classes.items()}
    return dict(o=o, e=e[keep], w=w[keep], eu=e[inside], level=level,
                bars=_bars(xr.timing, yt, lo, hi, spans, o),
                changes=_events(x_lines, y_lines, ('tempo',), ('tempo', 'first'), lo, hi, spans),
                resets=_events(x_lines, y_lines, ('reset',), ('reset',), lo, hi, spans),
                long=[float(v) for v in s.suff['boundaries']], reset20=[float(v) for v in s.suff['resets']])


def _union(spans_by_set: dict):
    if any(v is None for v in spans_by_set.values()):
        return None
    spans = sorted((float(a), float(b)) for v in spans_by_set.values() for a, b in v)
    out: list[list[float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def compare(a_ref: Reference, b_ref: Reference, a_strata, b_strata) -> dict:
    """Every cell of one pair, ``a`` the reference (older set). ``skip`` says why a pair is left out."""
    lo, hi = max(a_ref.lo_ms, b_ref.lo_ms), min(a_ref.hi_ms, b_ref.hi_ms)
    if not hi - lo >= MIN_SHARED_MS:
        return dict(skip='shared range under 20 s')
    xl, yl = a_strata.lines, b_strata.lines
    whole = [direction(a_ref, b_ref, xl, yl, lo, hi), direction(b_ref, a_ref, yl, xl, lo, hi)]
    if any(len(d['e']) < MIN_LINES for d in whole):
        return dict(skip='under 32 scored lines in a direction')
    cells = {'all': whole}
    member = recording_strata([a_strata, b_strata])
    for name in STRATA:
        if name in member:
            spans = _union(member[name])
            cells[name] = [direction(a_ref, b_ref, xl, yl, lo, hi, spans, whole[0]['o']),
                           direction(b_ref, a_ref, yl, xl, lo, hi, spans, whole[1]['o'])]
    bpm_a, bpm_b = a_strata.dominant_bpm, b_strata.dominant_bpm
    ratio = bpm_b / bpm_a if bpm_a and math.isfinite(bpm_a) and math.isfinite(bpm_b) else math.nan
    return dict(lo=lo, hi=hi, flags=shape_flags(whole, a_ref.segments, b_ref.segments),
                offset=(whole[0]['o'] - whole[1]['o']) / 2.0, bpm=(bpm_a, bpm_b), level_pair=level_class(ratio),
                fast=bool(max(bpm_a, bpm_b) >= FAST_BPM) if math.isfinite(ratio) else False,
                strata=sorted(member, key=STRATA.index), cells=cells)


# ---------------------------------------------------------------- the run

def run(units: Sequence, root, *, progress: Callable[[str], None] | None = None) -> dict:
    """Read the sets of every candidate pair once and compare each cross-creator pair.

    Returns ``pairs`` (`compare`'s result with identities), ``funnel`` (counts
    at each filter) and ``errors``.
    """
    from . import census
    say = progress or (lambda msg: None)
    cands = candidate_pairs(units)
    by_rec: dict[str, list] = {}
    for u in units:
        by_rec.setdefault(u.recording, []).append(u)
    other_audio = sum(1 for us in by_rec.values() for i in range(len(us)) for j in range(i + 1, len(us))
                      if us[i].audio_sha256 != us[j].audio_sha256 and not (us[i].set_id and us[i].set_id == us[j].set_id))
    needed = {id(u): u for pair in cands for u in pair}
    say(f'{len(cands)} candidate pairs over {len(needed)} sets')
    results = {r.unit.key: r for r in census.compute(list(needed.values()), root,
                                                     on_set=lambda u, ref, charts, paths, st: {'ref': ref})}
    funnel = dict(units=len(units), units_with_audio_sha=sum(1 for u in units if u.audio_sha256),
                  pairs_same_recording_other_audio=other_audio, candidate_pairs=len(cands), sets_read=len(results),
                  unreadable=0, shared_creator=0, skipped={}, pairs=0)
    pairs, errors = [], []
    for a, b in cands:
        ra, rb = results.get(a.key), results.get(b.key)
        if ra is None or rb is None or ra.strata is None or rb.strata is None or 'ref' not in ra.extra \
                or 'ref' not in rb.extra:
            funnel['unreadable'] += 1
            continue
        if not ra.creators or not rb.creators or ra.creators & rb.creators:
            funnel['shared_creator'] += 1
            continue
        try:
            c = compare(ra.extra['ref'], rb.extra['ref'], ra.strata, rb.strata)
        except Exception as exc:  # noqa: BLE001 - one failing pair is counted, never fatal
            errors.append(dict(a=a.key, b=b.key, error=f'{type(exc).__name__}: {exc}'))
            continue
        if 'skip' in c:
            funnel['skipped'][c['skip']] = funnel['skipped'].get(c['skip'], 0) + 1
            continue
        modes = '-'.join(sorted({a.mode or a.source, b.mode or b.source}))
        c.update(a=a.key, b=b.key, set_a=a.set_id, set_b=b.set_id, split='|'.join(sorted({a.split, b.split})),
                 rec=a.recording, modes=modes, status='|'.join(sorted({a.status, b.status})))
        pairs.append(c)
    funnel['pairs'] = len(pairs)
    for name in ('census',) + VIEWS[1:]:
        funnel[f'flag_{name}'] = sum(p['flags'][name if name == 'census' else name.lower()] for p in pairs)
    funnel['recordings'] = len({p['rec'] for p in pairs})
    funnel['errors'] = len(errors)
    return dict(pairs=pairs, funnel=funnel, errors=errors)


# ---------------------------------------------------------------- pooling and the bootstrap

class _Boot:
    """Multiplicities of B recording resamples, and 95 % percentile intervals."""

    def __init__(self, n_rec: int, draws: int, seed: int):
        self.n = n_rec
        if draws > 0 and n_rec > 0:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n_rec, size=(draws, n_rec))
            self.M = np.zeros((draws, n_rec))
            np.add.at(self.M, (np.repeat(np.arange(draws), n_rec), idx.ravel()), 1.0)
        else:
            self.M = np.zeros((0, n_rec))

    @staticmethod
    def _ci(point: float, draws: np.ndarray) -> dict:
        d = draws[np.isfinite(draws)] if len(draws) else draws
        if not len(d):
            return dict(v=point, lo=math.nan, hi=math.nan)
        return dict(v=point, lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))

    def ratio(self, num: np.ndarray, den: np.ndarray) -> dict:
        tot = den.sum()
        point = float(num.sum() / tot) if tot > 0 else math.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            draws = (self.M @ num) / (self.M @ den)
        return self._ci(point, draws)

    def hist_quantiles(self, H: np.ndarray, qs=QUANTILES) -> dict:
        def q_of(h: np.ndarray) -> np.ndarray:
            c = np.cumsum(h, axis=-1)
            tot = c[..., -1:]
            out = []
            for q in qs:
                i = np.argmax(c >= q * tot - 1e-12 * tot, axis=-1)
                v = (i + 0.5) * BIN_MS
                v = np.where(i >= H.shape[1] - 1, math.inf, v)
                out.append(np.where(tot[..., 0] > 0, v, math.nan))
            return np.array(out)
        point = q_of(H.sum(axis=0))
        draws = [q_of(self.M[k:k + 250] @ H) for k in range(0, len(self.M), 250)]
        draws = np.concatenate(draws, axis=1) if draws else np.zeros((len(qs), 0))
        return {f'q{int(round(q * 100))}': self._ci(float(point[i]), draws[i]) for i, q in enumerate(qs)}

    def wquantile(self, values: np.ndarray, w: np.ndarray, rec: np.ndarray, q: float,
                  fn: Callable | None = None) -> dict:
        def one(weights):
            v = values if fn is None else fn(values, weights)
            ok = weights > 0
            if not ok.any():
                return math.nan
            order = np.argsort(v[ok], kind='stable')
            cw = np.cumsum(weights[ok][order])
            return float(v[ok][order][np.searchsorted(cw, q * cw[-1] - 1e-12 * cw[-1])])
        point = one(w)
        draws = np.array([one(w * m[rec]) for m in self.M])
        return self._ci(point, draws)


def _hist(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    nb = int(round(MAX_MS / BIN_MS)) + 1
    i = np.minimum((np.abs(values) / BIN_MS).astype(np.int64), nb - 1)
    return np.bincount(i, weights=weights, minlength=nb)


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    ok = w > 0
    order = np.argsort(v[ok], kind='stable')
    cw = np.cumsum(w[ok][order])
    return float(v[ok][order][np.searchsorted(cw, 0.5 * cw[-1] - 1e-12 * cw[-1])])


def _mad_about_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.abs(values - _wmedian(values, weights))


def summarize_cell(items: Sequence[tuple[dict, list]], *, draws: int = 2000, seed: int = 20261021) -> dict:
    """The statistics of one cell: ``items`` holds (pair, its two directions in the cell)."""
    recs = sorted({p['rec'] for p, _ in items})
    if not items:
        return dict(n_pairs=0, n_rec=0)
    ri = {r: i for i, r in enumerate(recs)}
    per_rec = {r: 0 for r in recs}
    for p, _ in items:
        per_rec[p['rec']] += 1
    boot = _Boot(len(recs), draws, seed)
    n = len(recs)
    nb = int(round(MAX_MS / BIN_MS)) + 1
    H_raw, H_o = np.zeros((n, nb)), np.zeros((n, nb))
    sh = {k: np.zeros(n) for k in [f'raw{t:g}' for t in TAUS_MS] + [f'o{t:g}' for t in TAUS_MS]
          + [f'u{t:g}' for t in TAUS_MS] + ['tot', 'utot']}
    H_u = np.zeros((n, nb))
    off_v, off_s, off_w, off_r = [], [], [], []
    lvl_pair = {c: np.zeros(n) for c in LEVEL_CLASSES}
    lvl_dur = {c: np.zeros(n) for c in LEVEL_CLASSES}
    lvl_den, lvl_pair_den = np.zeros(n), np.zeros(n)
    bars = {k: np.zeros(n) for k in ('raw', 'o', 'den', 'meter', 'meter_den', 'same_all', 'same_all_den',
                                      'items', 'other_level', 'bars_all')}
    disp = {c: np.zeros(n) for c in BAR_CLASSES}
    ev = {f'{kind}_{side}_{k}': np.zeros(n) for kind in ('changes', 'resets') for side in ('rec', 'prec')
          for k in EVENT_KEYS}
    longs = {k: np.zeros(n) for k in ('n_ref', 'm_ref', 'n_pred', 'm_pred')}
    r20 = {k: np.zeros(n) for k in ('n', 'ok', 'ok_o', 'n_b', 'ok_b')}
    lines_n = 0
    for p, dirs in items:
        r = ri[p['rec']]
        wp = 1.0 / per_rec[p['rec']]
        live = [d for d in dirs if d['w'].sum() > 0]
        for d in live:
            share = wp / len(live)
            wl = share * d['w'] / d['w'].sum()
            lines_n += len(d['e'])
            H_raw[r] += _hist(d['e'], wl)
            H_o[r] += _hist(d['e'] - d['o'], wl)
            for t in TAUS_MS:
                sh[f'raw{t:g}'][r] += (wl * (np.abs(d['e']) <= t + 1e-9)).sum()
                sh[f'o{t:g}'][r] += (wl * (np.abs(d['e'] - d['o']) <= t + 1e-9)).sum()
            sh['tot'][r] += share
        live_u = [d for d in dirs if len(d['eu'])]
        for d in live_u:
            share = wp / len(live_u)
            wu = np.full(len(d['eu']), share / len(d['eu']))
            H_u[r] += _hist(d['eu'], wu)
            for t in TAUS_MS:
                sh[f'u{t:g}'][r] += (wu * (np.abs(d['eu']) <= t + 1e-9)).sum()
            sh['utot'][r] += share
        off_v.append(abs(p['offset']))
        off_s.append(p['offset'])
        off_w.append(wp)
        off_r.append(r)
        lvl_pair[p['level_pair']][r] += wp
        lvl_pair_den[r] += wp
        live_l = [d for d in dirs if sum(d['level'].values()) > 0]
        for d in live_l:
            tot = sum(d['level'].values())
            for c in LEVEL_CLASSES:
                lvl_dur[c][r] += wp / len(live_l) * d['level'].get(c, 0.0) / tot
        if live_l:
            lvl_den[r] += wp
        live_b = [d for d in dirs if d['bars']['n'] > 0]
        for d in live_b:
            b, share = d['bars'], wp / len(live_b)
            bars['raw'][r] += share * b['raw'] / b['n']
            bars['o'][r] += share * b['o'] / b['n']
            for c in BAR_CLASSES:
                disp[c][r] += share * b['disp'][c] / b['n']
        if live_b:
            bars['den'][r] += wp
        live_m = [d for d in dirs if d['bars']['matched_with_end'] > 0]
        for d in live_m:
            bars['meter'][r] += wp / len(live_m) * d['bars']['same'] / d['bars']['matched_with_end']
        if live_m:
            bars['meter_den'][r] += wp
        live_e = [d for d in dirs if d['bars']['with_end'] > 0]
        for d in live_e:
            bars['same_all'][r] += wp / len(live_e) * d['bars']['same'] / d['bars']['with_end']
        if live_e:
            bars['same_all_den'][r] += wp
        for d in dirs:
            bars['other_level'][r] += wp * d['bars']['other_level']
            bars['bars_all'][r] += wp * (d['bars']['other_level'] + d['bars']['n'])
        for kind in ('changes', 'resets'):
            for side, d in (('rec', dirs[0]), ('prec', dirs[1])):
                for k in EVENT_KEYS:
                    ev[f'{kind}_{side}_{k}'][r] += wp * d[kind][k]
        nr, mr, npd, mp = dirs[0]['long']
        for k, v in (('n_ref', nr), ('m_ref', mr), ('n_pred', npd), ('m_pred', mp)):
            longs[k][r] += wp * v
        r20['n'][r] += wp * dirs[0]['reset20'][0]
        r20['ok'][r] += wp * dirs[0]['reset20'][1]
        r20['ok_o'][r] += wp * dirs[0]['reset20'][2]
        r20['n_b'][r] += wp * dirs[1]['reset20'][0]
        r20['ok_b'][r] += wp * dirs[1]['reset20'][1]
    ov, os_, ow, orr = (np.array(x, dtype=np.float64) for x in (off_v, off_s, off_w, off_r))
    orr = orr.astype(np.int64)
    out = dict(n_pairs=len(items), n_rec=n, n_lines=lines_n)
    out['lines'] = dict(boot.hist_quantiles(H_raw), **{f'le{t:g}': boot.ratio(sh[f'raw{t:g}'], sh['tot'])
                                                        for t in TAUS_MS})
    out['lines_o'] = dict(boot.hist_quantiles(H_o), **{f'le{t:g}': boot.ratio(sh[f'o{t:g}'], sh['tot'])
                                                        for t in TAUS_MS})
    out['lines_u'] = dict(boot.hist_quantiles(H_u), **{f'le{t:g}': boot.ratio(sh[f'u{t:g}'], sh['utot'])
                                                        for t in TAUS_MS})
    out['offset'] = {f'q{int(round(q * 100))}': boot.wquantile(ov, ow, orr, q) for q in QUANTILES}
    out['offset'].update({f'le{t:g}': boot.ratio(np.bincount(orr, ow * (ov <= t + 1e-9), n), np.bincount(orr, ow, n))
                          for t in TAUS_MS})
    out['offset']['median_signed'] = boot.wquantile(os_, ow, orr, 0.5)
    rsd = boot.wquantile(os_, ow, orr, 0.5, fn=_mad_about_median)
    out['offset']['rsd_signed'] = {k: (v * ROBUST if isinstance(v, float) else v) for k, v in rsd.items()}
    out['offset']['rsd_abs'] = {k: v * ROBUST for k, v in out['offset']['q50'].items()}
    out['level_pair'] = {c: boot.ratio(lvl_pair[c], lvl_pair_den) for c in LEVEL_CLASSES}
    out['level_dur'] = {c: boot.ratio(lvl_dur[c], lvl_den) for c in LEVEL_CLASSES}
    out['bars'] = dict(start_raw=boot.ratio(bars['raw'], bars['den']), start_o=boot.ratio(bars['o'], bars['den']),
                       meter_given_start=boot.ratio(bars['meter'], bars['meter_den']),
                       same_bar_o=boot.ratio(bars['same_all'], bars['same_all_den']),
                       other_level_share=boot.ratio(bars['other_level'], bars['bars_all']),
                       disp={c: boot.ratio(disp[c], bars['den']) for c in BAR_CLASSES})
    for kind in ('changes', 'resets'):
        block = dict(n_ref=float(ev[f'{kind}_rec_n'].sum()), n_other=float(ev[f'{kind}_prec_n'].sum()))
        for k in EVENT_KEYS[1:]:
            block[f'recall_{k}'] = boot.ratio(ev[f'{kind}_rec_{k}'], ev[f'{kind}_rec_n'])
            block[f'precision_{k}'] = boot.ratio(ev[f'{kind}_prec_{k}'], ev[f'{kind}_prec_n'])
        out[kind] = block
    out['long_changes'] = dict(recall=boot.ratio(longs['m_ref'], longs['n_ref']),
                               precision=boot.ratio(longs['m_pred'], longs['n_pred']),
                               n_ref=float(longs['n_ref'].sum()))
    out['reset20'] = dict(recall=boot.ratio(r20['ok'], r20['n']), recall_o=boot.ratio(r20['ok_o'], r20['n']),
                          precision=boot.ratio(r20['ok_b'], r20['n_b']), n_ref=float(r20['n'].sum()))
    return out


def _view(pairs: Sequence[dict], view: str) -> list[dict]:
    if view == 'C0':
        return list(pairs)
    return [p for p in pairs if not p['flags'][view.lower()]]


def summarize(pairs: Sequence[dict], *, draws: int = 2000, seed: int = 20261021,
              strata_views: Sequence[str] = ('C1', 'C2')) -> dict:
    """Cells per view: ``all`` for every view; S0-S9, ``slow`` and ``fast`` for ``strata_views``."""
    out = {}
    for view in VIEWS:
        ps = _view(pairs, view)
        cells = {'all': [(p, p['cells']['all']) for p in ps]}
        if view in strata_views:
            for name in STRATA:
                cells[name] = [(p, p['cells'][name]) for p in ps if name in p['cells']]
            cells['slow'] = [(p, p['cells']['all']) for p in ps if not p['fast']]
            cells['fast'] = [(p, p['cells']['all']) for p in ps if p['fast']]
        out[view] = {name: summarize_cell(items, draws=draws, seed=seed) for name, items in cells.items()}
        out[view]['all']['modes'] = _count(p['modes'] for p in ps)
        out[view]['all']['splits'] = _count(p['split'] for p in ps)
    return out


def _count(xs: Iterable[str]) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- outputs

def _f(s: dict | None, digits: int = 1, pct: bool = False) -> str:
    """``value [lo, hi]``; '—' when nothing was scored, '>400' past the histogram."""
    if not isinstance(s, dict) or s.get('v') is None or (isinstance(s['v'], float) and math.isnan(s['v'])):
        return '—'
    k = 100.0 if pct else 1.0

    def one(x) -> str:
        if x is None or not isinstance(x, (int, float)) or math.isnan(x):
            return '?'
        return '>400' if math.isinf(x) else f'{x * k:.{digits}f}'
    return f"{one(s['v'])} [{one(s.get('lo'))}, {one(s.get('hi'))}]"


def _n(c: dict) -> str:
    flag = '' if c.get('n_pairs', 0) >= 10 else ' (n<10, not read)'
    return f"{c.get('n_pairs', 0)} / {c.get('n_rec', 0)}{flag}"


def markdown(summary: dict, funnel: dict) -> str:
    L = ['# Mapper-to-mapper spread: summary tables', '',
         'Cells: value [95 % recording-bootstrap interval]. Times in ms; shares in %. n = pairs / recordings.', '',
         '## Funnel', '', '| Step | Count |', '| --- | ---: |']
    for k, v in funnel.items():
        L.append(f'| {k} | {v} |')
    L += ['', '## Beat lines, pooled (head-weighted, recording-equal)', '',
          '| View | n | |e| p50 | p75 | p90 | p95 | ≤2 ms | ≤5 ms | ≤10 ms | ≤20 ms |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for block, label in (('lines', 'raw'), ('lines_o', 'offset removed'), ('lines_u', 'raw, unweighted')):
        for view in VIEWS:
            c = summary[view]['all']
            if not c.get('n_pairs'):
                continue
            b = c[block]
            L.append(f"| {view} {label} | {_n(c)} | {_f(b['q50'])} | {_f(b['q75'])} | {_f(b['q90'])} | "
                     f"{_f(b['q95'])} | {_f(b['le2'], 1, True)} | {_f(b['le5'], 1, True)} | {_f(b['le10'], 1, True)} | "
                     f"{_f(b['le20'], 1, True)} |")
    L += ['', '## Offset difference per pair (newer minus older set)', '',
          '| View | n | |o| p50 | p75 | p90 | p95 | rSD (1.4826 med |o|) | rSD signed (MAD) | median signed | '
          '≤2 ms | ≤5 ms | ≤10 ms | ≤20 ms |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for view in VIEWS:
        c = summary[view]['all']
        if not c.get('n_pairs'):
            continue
        o = c['offset']
        L.append(f"| {view} | {_n(c)} | {_f(o['q50'])} | {_f(o['q75'])} | {_f(o['q90'])} | {_f(o['q95'])} | "
                 f"{_f(o['rsd_abs'])} | {_f(o['rsd_signed'])} | {_f(o['median_signed'])} | {_f(o['le2'], 1, True)} | "
                 f"{_f(o['le5'], 1, True)} | {_f(o['le10'], 1, True)} | {_f(o['le20'], 1, True)} |")
    for view in [v for v in VIEWS if 'S1' in summary[v] or 'slow' in summary[v]]:
        L += ['', f'## Per stratum and tempo, view {view}', '',
              '| Cell | n | raw p50 | raw p95 | raw ≤5 | raw ≤20 | o-rem p50 | o-rem p95 | o-rem ≤5 | o-rem ≤20 | '
              '|o| p50 | offset rSD |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in ('all',) + STRATA + ('slow', 'fast'):
            c = summary[view].get(name)
            if not c or not c.get('n_pairs'):
                L.append(f'| {name} | 0 | | | | | | | | | | |')
                continue
            a, b, o = c['lines'], c['lines_o'], c['offset']
            L.append(f"| {name} | {_n(c)} | {_f(a['q50'])} | {_f(a['q95'])} | {_f(a['le5'], 1, True)} | "
                     f"{_f(a['le20'], 1, True)} | {_f(b['q50'])} | {_f(b['q95'])} | {_f(b['le5'], 1, True)} | "
                     f"{_f(b['le20'], 1, True)} | {_f(o['q50'])} | {_f(o['rsd_abs'])} |")
    L += ['', '## Level', '', '| View | n | pair: same | octave | family | other | duration: same | octave | family | '
          'other |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for view in VIEWS:
        c = summary[view]['all']
        if not c.get('n_pairs'):
            continue
        lp, ld = c['level_pair'], c['level_dur']
        L.append(f"| {view} | {_n(c)} | " + ' | '.join(_f(lp[k], 1, True) for k in ('correct', 'octave', 'family', 'other'))
                 + ' | ' + ' | '.join(_f(ld[k], 1, True) for k in ('correct', 'octave', 'family', 'other')) + ' |')
    for view in [v for v in VIEWS if 'S1' in summary[v]]:
        L += ['', f'## Bars and meter, view {view} (bars where both notate the same tempo)', '',
              '| Cell | n | start raw | start o-rem | off 1 beat | 2 | 3 | 4+ | fractional | meter given start | '
              'same bar | bars at other level |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in ('all',) + STRATA + ('slow', 'fast'):
            c = summary[view].get(name)
            if not c or not c.get('n_pairs'):
                continue
            b = c['bars']
            L.append(f"| {name} | {_n(c)} | {_f(b['start_raw'], 1, True)} | {_f(b['start_o'], 1, True)} | "
                     + ' | '.join(_f(b['disp'][k], 1, True) for k in BAR_CLASSES)
                     + f" | {_f(b['meter_given_start'], 1, True)} | {_f(b['same_bar_o'], 1, True)} | "
                       f"{_f(b['other_level_share'], 1, True)} |")
        L += ['', f'## Tempo changes and resets, view {view} (older set as reference)', '',
              '| Cell | n | kind | n ref / other | recall strict ½ | prec strict ½ | recall loose ½ | prec loose ½ | '
              'recall strict 2 | prec strict 2 |',
              '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in ('all',) + STRATA + ('slow', 'fast'):
            c = summary[view].get(name)
            if not c or not c.get('n_pairs'):
                continue
            for kind in ('changes', 'resets'):
                e = c[kind]
                L.append(f"| {name} | {_n(c)} | {kind} | {e['n_ref']:.1f} / {e['n_other']:.1f} | "
                         f"{_f(e['recall_strict_half'], 1, True)} | {_f(e['precision_strict_half'], 1, True)} | "
                         f"{_f(e['recall_loose_half'], 1, True)} | {_f(e['precision_loose_half'], 1, True)} | "
                         f"{_f(e['recall_strict_near'], 1, True)} | {_f(e['precision_strict_near'], 1, True)} |")
            lc, rr = c['long_changes'], c['reset20']
            L.append(f"| {name} | {_n(c)} | long changes (≥8 s runs, 1 beat) | {lc['n_ref']:.1f} | "
                     f"{_f(lc['recall'], 1, True)} | {_f(lc['precision'], 1, True)} | | | | |")
            L.append(f"| {name} | {_n(c)} | reset 20 ms (lines or bars) | {rr['n_ref']:.1f} | "
                     f"{_f(rr['recall'], 1, True)} | {_f(rr['precision'], 1, True)} | o-rem {_f(rr['recall_o'], 1, True)} "
                     f"| | | |")
    L += ['', 'Event counts n are pair-weighted (each recording sums to 1).', '']
    return '\n'.join(L) + '\n'


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else (None if math.isnan(float(x)) else str(float(x)))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def pair_row(p: dict) -> dict:
    """A pair's identity, flags and whole-range per-direction scalars, for ``pairs.jsonl``."""
    row = {k: p[k] for k in ('a', 'b', 'set_a', 'set_b', 'split', 'rec', 'modes', 'status', 'lo', 'hi', 'flags',
                             'offset', 'bpm', 'level_pair', 'fast', 'strata')}
    for name, d in zip(('ab', 'ba'), p['cells']['all']):
        e, w = d['e'], d['w']
        tot = w.sum()
        row[name] = dict(o=d['o'], lines=int(len(e)),
                         le5=float((w * (np.abs(e) <= 5 + 1e-9)).sum() / tot) if tot else None,
                         le20=float((w * (np.abs(e) <= 20 + 1e-9)).sum() / tot) if tot else None,
                         le5_o=float((w * (np.abs(e - d['o']) <= 5 + 1e-9)).sum() / tot) if tot else None,
                         le20_o=float((w * (np.abs(e - d['o']) <= 20 + 1e-9)).sum() / tot) if tot else None,
                         bars=d['bars'], changes=d['changes'], resets=d['resets'], level=d['level'])
    return row


def write(out_dir, result: dict, summary: dict, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'pairs.jsonl').open('w', encoding='utf-8') as f:
        for p in result['pairs']:
            f.write(json.dumps(_jsonable(pair_row(p))) + '\n')
    arrays = {}
    for i, p in enumerate(result['pairs']):
        for name, d in zip(('ab', 'ba'), p['cells']['all']):
            arrays[f'{i}_{name}_e'] = d['e']
            arrays[f'{i}_{name}_w'] = d['w']
    np.savez_compressed(out / 'lines.npz', **arrays)
    doc = dict(meta, funnel=result['funnel'], errors=result['errors'], summary=summary)
    (out / 'summary.json').write_text(json.dumps(_jsonable(doc), indent=1) + '\n', encoding='utf-8')
    (out / 'summary.md').write_text(markdown(summary, result['funnel']), encoding='utf-8')
