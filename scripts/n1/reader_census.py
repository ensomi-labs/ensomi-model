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


def r_lines(path, lo: float, hi: float):
    """R's canonical beat lines and bar starts over ``[lo, hi]`` (one line of margin), and its roles."""
    red = read_red_lines(path)
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


def compare(ref, timing_chart) -> dict:
    """E3's comparison for one set (module docstring). ``ref`` is the set's N `Reference`."""
    lo, hi = ref.lo_ms, ref.hi_ms
    n_lines = ref.lines().t
    r_t, r_bars, roles = r_lines(timing_chart, lo, hi)
    heads = ref.heads[(ref.heads >= lo) & (ref.heads <= hi)]
    n_prev, n_next = _around(n_lines, heads)
    r_prev, r_next = _around(r_t, heads)
    on_n = np.isin(np.round(heads, 6), np.round(n_lines, 6))
    with np.errstate(invalid='ignore'):
        same_prev = np.abs(n_prev - r_prev) <= TOL_MS
        same_next = np.abs(n_next - r_next) <= TOL_MS
    changed = ~(same_prev & (same_next | on_n))
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
