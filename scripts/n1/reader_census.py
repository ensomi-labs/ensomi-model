"""E3, the reader census: where N's `.osu` reader and R's give a set different beat lines.

R's reader is `ensomi_model.evaluation.redlines.musical_grid` (AGPL, the R line):
it keeps musical lines and drops resets as ``short_bar``, detours and
implausible lines, and places canonical beats from each musical segment's
offset (`evaluation.beats`). N's is `ensomi_n.labels` with
`ensomi_n.timing.Timing.canonical` (MIT): every red line kept. This script is
outside ``n/`` because it imports the R line.

For one set (N's timing chart, events from that chart as R's evaluation reads
them), `compare` reports:

- ``changed_heads``: heads of the set (the union over its difficulties) whose
  two surrounding beat lines differ between the readers by more than 20 ms, or
  that one reader leaves without a line on either side; ``changed_share`` is
  their share of the heads in N's scored range;
- ``n_only`` and ``r_only``: lines of one reader in the scored range with no
  line of the other within 20 ms;
- ``bars_unmatched``: N's bar starts in range with no R bar start within 20 ms
  (R's bars are canonical, F3, so this differs by construction at fast
  notation; it is reported apart and is not E3's trigger);
- ``roles``: R's role and reason counts for the chart's lines, which say why.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ensomi_model.evaluation.redlines import EXPRESSIVE, MUSICAL, musical_grid, read_red_lines

TOL_MS = 20.0
MERGE_MS = 2.0


def r_events(path) -> np.ndarray:
    """Head times, plus hold and spinner ends, of one chart, as R's evaluation reads events."""
    out, section = [], None
    with Path(path).open('r', encoding='utf-8-sig', errors='replace') as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue
            if section != 'HitObjects' or not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            t, kind = float(parts[2]), int(float(parts[3]))
            out.append(t)
            if kind & 128 and len(parts) > 5:
                out.append(float(parts[5].split(':', 1)[0]))
            elif kind & 8 and len(parts) > 5:
                out.append(float(parts[5]))
    return np.sort(np.asarray(out, dtype=np.float64))


def r_lines(path, lo: float, hi: float, red=None):
    """R's canonical beat lines and bar starts over ``[lo, hi]`` (one line of margin), and its roles."""
    red = read_red_lines(path) if red is None else red
    grid, roles = musical_grid(red, r_events(path))
    segs = grid.segments
    lines = []
    for i, s in enumerate(segs):
        c = s.canonical_beat_length_ms
        end = segs[i + 1].offset_ms if i + 1 < len(segs) else hi + 2 * c
        start = s.offset_ms
        k0 = math.floor((lo - start) / c) - 1 if i == 0 else 0
        k1 = math.ceil((min(end, hi + 2 * c) - start) / c)
        k = np.arange(k0, k1 + 1)
        t = start + k * c
        lines.append(t[(t < end - MERGE_MS) & (t >= lo - 2 * c) & (t <= hi + 2 * c)])
    lines_t = np.unique(np.concatenate(lines)) if lines else np.zeros(0)
    loc = grid.locate(np.array([lo, hi]))
    bars = grid.time_of_bar(np.arange(int(loc['bar'][0]) - 1, int(loc['bar'][1]) + 2))
    bars = np.unique(bars[(bars >= lo - MERGE_MS) & (bars <= hi + MERGE_MS)])
    return lines_t, bars, roles


def _around(lines: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.searchsorted(lines, h + 1e-9, side='right') - 1
    prev = np.where(k >= 0, lines[np.clip(k, 0, len(lines) - 1)], np.nan) if len(lines) else np.full(len(h), np.nan)
    nxt = np.where(k + 1 < len(lines), lines[np.clip(k + 1, 0, len(lines) - 1)], np.nan) if len(lines) \
        else np.full(len(h), np.nan)
    return prev, nxt


def _unmatched(a: np.ndarray, b: np.ndarray) -> int:
    if not len(a):
        return 0
    if not len(b):
        return int(len(a))
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), dtype=np.int64)
    d = np.minimum(np.abs(b[i] - a), np.abs(b[np.maximum(i - 1, 0)] - a))
    return int((d > TOL_MS).sum())


def changed_mask(heads: np.ndarray, n_lines: np.ndarray, r_t: np.ndarray) -> np.ndarray:
    """Per head: the two lines around it differ between the readers by more than 20 ms, or a reader has
    none on one side (a head on one of N's lines needs only the line before it to agree)."""
    n_prev, n_next = _around(n_lines, heads)
    r_prev, r_next = _around(r_t, heads)
    on_n = np.isin(np.round(heads, 6), np.round(n_lines, 6))
    with np.errstate(invalid='ignore'):
        same_prev = np.abs(n_prev - r_prev) <= TOL_MS
        same_next = np.abs(n_next - r_next) <= TOL_MS
    return ~(same_prev & (same_next | on_n))


def compare(ref, timing_chart, n_lines: np.ndarray | None = None, r_side=None) -> dict:
    """E3's comparison for one set (module docstring). ``ref`` is the set's N `Reference`; ``n_lines``
    replaces N's scored lines (E0's, for the before count); ``r_side`` is `r_lines`'s result, reused."""
    lo, hi = ref.lo_ms, ref.hi_ms
    n_lines = ref.lines().t if n_lines is None else n_lines
    r_t, r_bars, roles = r_lines(timing_chart, lo, hi) if r_side is None else r_side
    heads = ref.heads[(ref.heads >= lo) & (ref.heads <= hi)]
    changed = changed_mask(heads, n_lines, r_t)
    in_n = n_lines[(n_lines >= lo - MERGE_MS) & (n_lines <= hi + MERGE_MS)]
    in_r = r_t[(r_t >= lo - MERGE_MS) & (r_t <= hi + MERGE_MS)]
    n_bars = ref.timing.bar_starts()
    n_bars = n_bars[(n_bars >= lo - MERGE_MS) & (n_bars <= hi + MERGE_MS)]
    counts: dict[str, int] = {}
    for role in roles:
        name = role.role if role.role != EXPRESSIVE else f'expressive:{role.reason}'
        if role.role == MUSICAL:
            name = f'musical:{role.reason}'
        counts[name] = counts.get(name, 0) + 1
    return dict(heads=int(len(heads)), changed_heads=int(changed.sum()), changed_t=heads[changed],
                changed_share=float(changed.mean()) if len(heads) else 0.0,
                n_only=_unmatched(in_n, in_r), r_only=_unmatched(in_r, in_n), n_lines=int(len(in_n)),
                r_lines=int(len(in_r)), n_bars=int(len(n_bars)), bars_unmatched=_unmatched(n_bars, r_bars),
                roles=counts)


def causes(changed_t: np.ndarray, lines) -> dict[str, int]:
    """Changed heads by the kind of N's red line governing them (`ensomi_n.eval.strata.Line`), with the
    notated meter and tempo of that line for ``first``, ``same`` and ``tempo`` lines at 160 BPM or more."""
    out: dict[str, int] = {}
    if not len(changed_t) or not lines:
        return out
    starts = np.array([line.start_ms for line in lines])
    k = np.clip(np.searchsorted(starts, changed_t + MERGE_MS, side='right') - 1, 0, len(lines) - 1)
    for i, n in zip(*np.unique(k, return_counts=True)):
        line = lines[int(i)]
        name = line.kind
        if name in ('first', 'same', 'tempo') and 60000.0 / line.beat_ms >= 160.0 - 1e-9:
            name = f'{name}@fast meter {line.meter}'
        out[name] = out.get(name, 0) + int(n)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


LABELS = {
    'reset': 'resets kept by N (R drops them as short bars)',
    'detour': 'a detour run kept by N (R keeps the grid in force)',
    'implausible': 'a stop or spike line: N places no beats in its span',
    'same': 'redundant red lines restart N\'s canonical count where R continues',
    'tempo': 'tempo lines R drops or reads otherwise',
    'first': 'the first line\'s grid',
}


def why(result: dict, kinds: dict, cause: dict | None = None) -> str:
    """A short reason for a changed set: the leading cause by changed heads, then R's dropped lines."""
    parts = []
    if cause:
        top = next(iter(cause))
        base = top.split('@', 1)[0]
        if '@fast' in top:
            parts.append(f"{cause[top]} heads under {base} lines at a notated tempo of 160 BPM or more with "
                         f"{top.split('@fast ', 1)[1]}: N re-anchors the fast-beat count at downbeats and segment "
                         f"starts, R at musical lines only")
        else:
            parts.append(f'{cause[top]} heads: {LABELS.get(base, base)}')
    return '; '.join(parts + [_why_roles(result, kinds)])


def _why_roles(result: dict, kinds: dict) -> str:
    """A short reason for a changed set: what R drops that N keeps, from R's roles and N's line kinds."""
    roles = result['roles']
    parts = []
    for reason, label in (('short_bar', 'short-bar lines'), ('detour', 'detour lines'),
                          ('implausible', 'implausible-BPM lines'), ('empty', 'empty lines')):
        n = roles.get(f'expressive:{reason}', 0)
        if n:
            parts.append(f'R drops {n} {label}')
    if roles.get('musical:phase'):
        parts.append(f"R reads {roles['musical:phase']} phase lines")
    n_kinds = ', '.join(f'{k} {v}' for k, v in sorted(kinds.items()) if k != 'first')
    if not parts:
        parts.append('no line dropped by R')
    return '; '.join(parts) + (f' (N line kinds: {n_kinds})' if n_kinds else '')


# ---------------------------------------------------------------- before and after the reader fix

def lead(cause: dict) -> str:
    """The leading cause of a changed set as the census names it: ``same``, ``same@fast``, ``reset``, ..."""
    top = next(iter(cause), 'none')
    return top.split('@', 1)[0] + ('@fast' if '@fast' in top else '')


def _r_as_lines(r_t: np.ndarray):
    from ensomi_n.timing import Lines
    gap = np.diff(r_t)
    beat = np.append(gap, gap[-1:]) if len(gap) else np.zeros(len(r_t))
    idx = np.arange(len(r_t))
    return Lines(r_t, beat, idx, idx)


def span_mask(t: np.ndarray, spans) -> np.ndarray:
    if spans is None:
        return np.ones(len(t), dtype=bool)
    ok = np.zeros(len(t), dtype=bool)
    for a, b in spans:
        ok |= (t >= a) & (t < b)
    return ok


def free_sums(ref, lines_c, span_sets: dict) -> dict:
    """Per span set: head weight ``w``, weight on free lines ``wf``, and chance-weighted sums ``c``, ``cf``.

    ``lines_c`` are N's scored lines refined by their own free factors, the lines BL-20 compares at
    against a prediction with no free bars of its own; a line's chance is ``min(1, 40 ms / spacing)``.
    Weights are `Reference.weights` of the heads inside each span set (None: all heads).
    """
    import dataclasses
    chance = np.minimum(1.0, 2.0 * TOL_MS / lines_c.beat_ms)
    free = lines_c.factor > 1
    heads = ref.heads if ref.heads is not None else np.zeros(0)
    out = {}
    for name, spans in span_sets.items():
        sub = dataclasses.replace(ref, heads=heads[span_mask(heads, spans)])
        w = sub.weights(lines_c)
        out[name] = dict(w=float(w.sum()), wf=float(w[free].sum()), c=float((w * chance).sum()),
                         cf=float((w * chance)[free].sum()))
    return out


def e3_set(ref, timing_chart, st, e0_lines: np.ndarray, with_detail: bool) -> dict:
    """E3 for one set with E0's lines and with N's current lines, against R's, in one pass.

    ``old`` and ``new`` are `compare` results (``changed_t`` dropped) with their causes. ``new`` also
    gives ``changed_free`` (changed heads inside free spans, `ensomi_n.timing`), ``changed_refined``
    (heads still changed when free spans are compared at the refined level on both sides, as BL-20
    compares them), and ``free_heads`` (heads in free spans). ``free`` holds `free_sums` for the whole
    set (``whole``), each stratum's spans and S4's spans. With ``with_detail`` (unsealed sets only),
    ``same_diag`` holds `same_diagnostic` for a set whose E0 changes lead with ``same`` at normal tempo.
    """
    from ensomi_n.eval.measures import compared_lines
    lo, hi = ref.lo_ms, ref.hi_ms
    red = read_red_lines(timing_chart)
    r_side = r_lines(timing_chart, lo, hi, red)
    r_t = r_side[0]
    heads = ref.heads[(ref.heads >= lo) & (ref.heads <= hi)]
    old = compare(ref, timing_chart, e0_lines, r_side)
    old_t = old.pop('changed_t')
    old['causes'] = causes(old_t, st.lines)
    lines = ref.lines()
    new = compare(ref, timing_chart, lines.t, r_side)
    new_t = new.pop('changed_t')
    new['causes'] = causes(new_t, st.lines)
    changed = changed_mask(heads, lines.t, r_t)
    in_free = lines.factor_at(heads) > 1
    end = float(ref.timing.beats[-1]) if len(ref.timing) else -math.inf
    n_c, r_c, _ = compared_lines(lines, _r_as_lines(r_t), True, False, ref_all=ref.timing.canonical(),
                                 ref_end=end, pred_end=float(r_t[-1]) if len(r_t) else -math.inf)
    refined = changed_mask(heads, n_c.t, r_c.t) if len(r_t) else changed
    new.update(changed_free=int((changed & in_free).sum()), changed_refined=int(refined.sum()),
               free_heads=int(in_free.sum()))
    own_c, _ = lines.refine(lines.factor, True, end)
    spans = {'whole': None, 'S4spans': st.s4_spans or []}
    spans.update({k: v for k, v in st.members.items() if v is not None})
    out = dict(old=old, new=new, free=free_sums(ref, own_c, spans))
    if with_detail and old['changed_heads'] and lead(old['causes']) == 'same':
        out['same_diag'] = same_diagnostic(ref, st, red, r_side[2], old_t, e0_lines, lines.t, r_t)
    return out


def _role_name(role) -> str:
    if role.role == EXPRESSIVE:
        return f'expressive:{role.reason}'
    if role.role == MUSICAL:
        return f'musical:{role.reason}'
    return role.role


def same_diagnostic(ref, st, red, roles, old_t, e0_lines, n_t, r_t) -> dict:
    """The §4 reading of a set whose E0 changes lead with redundant lines at normal tempo, at h, its first
    changed head under one of N's ``same`` lines.

    ``prior`` is the nearest of N's red lines before h's line that is not ``same``, with N's kind and R's
    role; ``between`` counts R's roles of N's ``same`` lines from there to h's line; ``r_grid_ms`` is the
    offset of R's musical line in force at h, and ``stale`` says whether it lies before ``prior`` (R keeps
    a grid from before it). ``cascade`` holds when ``prior`` is a reset R drops as ``short_bar``, every
    line between is dropped by R too, and R's grid is stale: §4's reading.
    """
    lines = st.lines
    starts = np.array([line.start_ms for line in lines])
    gov = np.clip(np.searchsorted(starts, old_t + MERGE_MS, side='right') - 1, 0, len(lines) - 1)
    under_same = [k for k, g_ in enumerate(gov) if lines[int(g_)].kind == 'same']
    first = under_same[0] if under_same else 0
    h, g = float(old_t[first]), int(gov[first])
    p = g
    while p > 0 and lines[p].kind == 'same':
        p -= 1
    r_off = np.array([r.offset_ms for r in red])

    def r_role(t: float) -> str:
        k = np.flatnonzero(np.abs(r_off - t) <= 0.5)
        return _role_name(roles[int(k[-1])]) if len(k) else 'none'

    musical = [r.offset_ms for r, role in zip(red, roles) if role.role == MUSICAL and r.offset_ms <= h + MERGE_MS]
    r_grid = musical[-1] if musical else math.nan
    between: dict[str, int] = {}
    for line in lines[p + 1:g + 1]:
        name = r_role(line.start_ms)
        between[name] = between.get(name, 0) + 1
    prior = lines[p]
    prior_role = r_role(prior.start_ms)
    stale = bool(math.isfinite(r_grid) and r_grid < prior.start_ms - MERGE_MS)
    dropped = all(k.startswith('expressive') for k in between) if between else True
    cascade = prior.kind == 'reset' and prior_role == 'expressive:short_bar' and dropped and stale

    def line_row(line) -> dict:
        return dict(start_ms=round(line.start_ms, 3), bpm=round(60000.0 / line.beat_ms, 3), meter=line.meter,
                    kind=line.kind, r_role=r_role(line.start_ms))

    n_prev, n_next = _around(n_t, np.array([h]))
    r_prev, r_next = _around(r_t, np.array([h]))
    e_prev, e_next = _around(e0_lines, np.array([h]))
    return dict(h_ms=round(h, 3), changed=int(len(old_t)), prior=line_row(prior), prior_index=p, line_index=g,
                governing=line_row(lines[g]), next=line_row(lines[g + 1]) if g + 1 < len(lines) else None,
                between=between, r_grid_ms=r_grid, stale=stale, cascade=bool(cascade),
                around=dict(e0=[float(e_prev[0]), float(e_next[0])], n=[float(n_prev[0]), float(n_next[0])],
                            r=[float(r_prev[0]), float(r_next[0])]))
