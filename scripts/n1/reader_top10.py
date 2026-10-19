"""Renders of the charts whose beat lines N's reader changes most against R's, for the human.

For each chart, one window around its first changed head: the chart's red lines
(BPM, meter, N's kind, R's role), N's canonical lines after the reader fix, E0's
lines before it, R's lines, and the set's heads (the union over its
difficulties; heads whose surrounding lines differ after the fix drawn dark and
tall). Free spans (bars holding no canonical pulse, scored at the refined level)
are shaded. Every chart gets a plain SVG written here; a PNG is added when
matplotlib imports. `render` is called by ``census_run.py`` with the top lists;
unsealed splits only, which the caller guarantees.
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#a8a7a2'
FREE = '#e9e8e3'
SERIES = {'n': '#2a78d6', 'e0': '#1baf7a', 'r': '#eb6834'}   # categorical slots 1, 3, 2
ROWS = (('red', 'red lines'), ('n', 'N, after the fix'), ('e0', 'N before (E0)'), ('r', 'R'),
        ('heads', 'heads'))
WINDOW_S = 10.0
LEAD_S = 1.5


def _window(changed_t: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    h = float(changed_t[0]) if len(changed_t) else lo
    a = max(lo - 500.0, h - LEAD_S * 1000.0)
    return a, min(max(hi + 500.0, a + 1000.0), a + WINDOW_S * 1000.0)


def primitives(item: dict, a: float, b: float) -> list[tuple]:
    """Drawing primitives in data units: x in seconds, y in rows (0 at the top, one unit per row)."""
    out: list[tuple] = []
    sec = lambda t: t / 1000.0  # noqa: E731
    for f0, f1 in item['free_spans']:
        x0, x1 = max(f0, a), min(f1, b)
        if x1 > x0:
            out.append(('rect', sec(x0), sec(x1), 0.0, len(ROWS), FREE))
    rows = {name: k for k, (name, _) in enumerate(ROWS)}
    for name in ('n', 'e0', 'r'):
        y = rows[name]
        for t in item['lines'][name]:
            if a <= t <= b:
                out.append(('vline', sec(t), y + 0.1, y + 0.9, SERIES[name], 2.0))
    y = rows['heads']
    changed = set(np.round(item['changed_t'], 3).tolist())
    for t in item['heads']:
        if a <= t <= b:
            hit = round(float(t), 3) in changed
            out.append(('vline', sec(t), y + (0.1 if hit else 0.45), y + 0.9, INK if hit else MUTED, 1.0))
    y = rows['red']
    shown = [line for line in item['red_lines'] if a <= line['start_ms'] <= b]
    crowded = len(shown) > 16      # label only the lines that are not redundant when many are drawn
    for k, line in enumerate(shown):
        out.append(('vline', sec(line['start_ms']), y + 0.05, y + 0.95, INK2, 1.0))
        if crowded and line['kind'] == 'same':
            continue
        label = f"{line['bpm']:g} BPM m{line['meter']} {line['kind']} / R {line['r_role'].split(':')[-1]}"
        out.append(('text', sec(line['start_ms']) + 0.02, y - 0.25 + 0.3 * (k % 2), label, 7.5, INK))
    return out


def _svg(item: dict, prims: list[tuple], a: float, b: float, path: Path) -> None:
    w, left, right, top, row_h = 1400, 130, 20, 46, 34
    h = top + row_h * len(ROWS) + 46
    sx = lambda x: left + (x - a / 1000.0) / ((b - a) / 1000.0) * (w - left - right)  # noqa: E731
    sy = lambda y: top + y * row_h  # noqa: E731
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
         f'font-family="Helvetica, Arial, sans-serif"><rect width="{w}" height="{h}" fill="{SURFACE}"/>',
         f'<text x="{left}" y="18" font-size="13" fill="{INK}">{html.escape(item["title"])}</text>',
         f'<text x="{left}" y="35" font-size="10.5" fill="{INK2}">{html.escape(item["subtitle"])}</text>']
    for kind, *p in prims:
        if kind == 'rect':
            x0, x1, y0, y1, fill = p
            e.append(f'<rect x="{sx(x0):.1f}" y="{sy(y0):.1f}" width="{sx(x1) - sx(x0):.1f}" '
                     f'height="{sy(y1) - sy(y0):.1f}" fill="{fill}"/>')
        elif kind == 'vline':
            x, y0, y1, color, width = p
            e.append(f'<line x1="{sx(x):.2f}" x2="{sx(x):.2f}" y1="{sy(y0):.1f}" y2="{sy(y1):.1f}" '
                     f'stroke="{color}" stroke-width="{width}"/>')
        elif kind == 'text':
            x, y, s, size, color = p
            e.append(f'<text x="{sx(x):.1f}" y="{sy(y):.1f}" font-size="{size}" fill="{color}">'
                     f'{html.escape(s)}</text>')
    for k, (name, label) in enumerate(ROWS):
        color = SERIES.get(name, INK2)
        e.append(f'<text x="{left - 8}" y="{sy(k + 0.6):.1f}" font-size="11" text-anchor="end" '
                 f'fill="{INK}">{html.escape(label)}</text>')
        if name in SERIES:
            e.append(f'<rect x="{left - 124}" y="{sy(k + 0.3):.1f}" width="6" height="{0.4 * row_h:.1f}" '
                     f'fill="{color}"/>')
    base = sy(len(ROWS))
    e.append(f'<line x1="{left}" x2="{w - right}" y1="{base:.1f}" y2="{base:.1f}" stroke="{MUTED}"/>')
    for s in range(math.ceil(a / 1000.0), math.floor(b / 1000.0) + 1):
        e.append(f'<line x1="{sx(s):.1f}" x2="{sx(s):.1f}" y1="{base:.1f}" y2="{base + 4:.1f}" stroke="{INK2}"/>'
                 f'<text x="{sx(s):.1f}" y="{base + 16:.1f}" font-size="10" text-anchor="middle" '
                 f'fill="{INK2}">{s}</text>')
    e.append(f'<text x="{w - right}" y="{base + 32:.1f}" font-size="10" text-anchor="end" fill="{INK2}">'
             f'time (s); shaded: free spans (N); dark tall heads: a line around them has no line of the other '
             f'reader within 20 ms (N after the fix against R)'
             f'</text></svg>')
    path.write_text('\n'.join(e) + '\n', encoding='utf-8')


def _png(item: dict, prims: list[tuple], a: float, b: float, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(14, 2.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for kind, *p in prims:
        if kind == 'rect':
            x0, x1, y0, y1, fill = p
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=fill, lw=0))
        elif kind == 'vline':
            x, y0, y1, color, width = p
            ax.plot([x, x], [y0, y1], color=color, lw=width * 0.8)
        elif kind == 'text':
            x, y, s, size, color = p
            ax.text(x, y, s, fontsize=size * 0.8, color=color, va='center')
    ax.set_xlim(a / 1000.0, b / 1000.0)
    ax.set_ylim(len(ROWS), 0)
    ax.set_yticks([k + 0.5 for k in range(len(ROWS))])
    ax.set_yticklabels([label for _, label in ROWS], fontsize=8)
    for k, (name, _) in enumerate(ROWS):
        if name in SERIES:
            ax.get_yticklabels()[k].set_color(SERIES[name])
    ax.set_xlabel('time (s); shaded: free spans (N); dark tall heads: a line around them has no line of the other '
                  'reader within 20 ms (N after the fix against R)',
                  fontsize=7, color=INK2)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_title(item['title'] + '\n' + item['subtitle'], fontsize=8, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def chart_item(ref, st, timing_chart: str, e0_lines: np.ndarray, meta: dict, row: dict) -> dict:
    """What one render needs, from the set's N reference and strata, R's reader and E0's lines."""
    import reader_census as rc
    lo, hi = ref.lo_ms, ref.hi_ms
    red = rc.read_red_lines(timing_chart)
    r_t, _, roles = rc.r_lines(timing_chart, lo, hi, red)
    lines = ref.lines()
    heads = ref.heads[(ref.heads >= lo) & (ref.heads <= hi)]
    changed = rc.changed_matched(heads, lines.t, r_t)
    f = lines.factor
    spans, k = [], 0
    while k < len(f) - 1:
        if f[k] > 1:
            j = k
            while j < len(f) - 1 and f[j] > 1:
                j += 1
            spans.append((float(lines.t[k]), float(lines.t[j])))
            k = j
        else:
            k += 1
    r_off = np.array([r.offset_ms for r in red])
    red_rows = []
    for line in st.lines:
        idx = np.flatnonzero(np.abs(r_off - line.start_ms) <= 0.5)
        role = rc._role_name(roles[int(idx[-1])]) if len(idx) else 'none'
        red_rows.append(dict(start_ms=line.start_ms, bpm=round(60000.0 / line.beat_ms, 2), meter=line.meter,
                             kind=line.kind, r_role=role))
    changed_t = heads[changed]
    first_t = changed_t
    if row.get('window_by') == 'refined':   # start at the first head BL-20 can see changed
        from ensomi_n.eval.measures import compared_lines
        end = float(ref.timing.beats[-1]) if len(ref.timing) else -math.inf
        n_c, r_c, _ = compared_lines(lines, rc._r_as_lines(r_t), True, False, ref_all=ref.timing.canonical(),
                                     ref_end=end, pred_end=float(r_t[-1]) if len(r_t) else -math.inf)
        refined = heads[rc.changed_matched(heads, n_c.t, r_c.t)]
        first_t = refined if len(refined) else changed_t
    a, b = _window(first_t, lo, hi)
    title = f"{row['key']} ({row['split']}): {meta.get('artist', '')} - {meta.get('title', '')} [{meta.get('version', '')}]"
    sub = (f"{row['changed_heads']} of {row['heads']} heads changed after the fix (matched rule) "
           f"({row['changed_share']:.3f}; {row['changed_free']} in free spans, {row['changed_refined']} still "
           f"changed refined); before: {row['changed_e0']}; window {a / 1000:.1f}-{b / 1000:.1f} s")
    return dict(title=title, subtitle=sub, free_spans=spans, heads=heads, changed_t=changed_t,
                lines=dict(n=lines.t, e0=e0_lines, r=r_t), red_lines=red_rows, window=(a, b))


def render(items: list[tuple[dict, dict]], out: Path, first: list[str] | None = None) -> dict:
    """Write one SVG (and a PNG when matplotlib imports) per ``(row, item)``, and ``index.md``.

    ``first`` lists the keys of the top 10 by the share of heads still changed when free spans are
    compared refined (what BL-20 can see); the other items are the rest of the top 10 by changed share,
    listed apart.
    """
    strict = set(first if first is not None else [row['key'] for row, _ in items])
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
        backend = f'matplotlib {matplotlib.__version__}'
    except Exception as exc:  # noqa: BLE001
        backend = f'no matplotlib ({type(exc).__name__}); SVG only'
    rows, errors = [], []
    for n, (row, item) in enumerate(items, 1):
        stem = f"{n:02d}-{row['set_id'] or 'set'}-{row['key'].replace('/', '_').replace('@', '_')}"[:120]
        a, b = item['window']
        prims = primitives(item, a, b)
        _svg(item, prims, a, b, out / f'{stem}.svg')
        png = ''
        if backend.startswith('matplotlib'):
            try:
                _png(item, prims, a, b, out / f'{stem}.png')
                png = f'{stem}.png'
            except Exception as exc:  # noqa: BLE001 - the SVG stands
                errors.append(f'{stem}: {type(exc).__name__}: {exc}')
        rows.append(dict(row, svg=f'{stem}.svg', png=png, window_s=[round(a / 1000, 1), round(b / 1000, 1)]))
    lines = ['# Reader top 10 after the fix (unsealed splits, one chart per recording)', '',
             f'Backend: {backend}. Rows: the chart\'s red lines (BPM, meter, N\'s kind, R\'s role), N\'s lines '
             'after the reader fix, E0\'s lines before it, R\'s lines, and the heads (dark and tall where a line '
             'around a head has no line of the other reader within 20 ms, N after the fix against R). Shaded: free '
             'spans. Each '
             'window starts 1.5 s before the first changed head (in the first list, the first still changed with '
             'free spans refined) and runs 10 s.', '']
    head = ['| # | Set | Artist | Title | Split | Window (s) | Changed heads | Reason | File |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for title, keep in (('By share of heads still changed with free spans compared refined', True),
                        ('Also in the top 10 by share of changed heads (free spans compared as lines)', False)):
        part = [(n, r) for n, r in enumerate(rows, 1) if (r['key'] in strict) == keep]
        if not part:
            continue
        lines += [f'## {title}', ''] + head
        for n, r in part:
            lines.append(f"| {n} | `{r['key']}` | {r['artist']} | {r['title']} [{r['version']}] | {r['split']} | "
                         f"{r['window_s'][0]}-{r['window_s'][1]} | {r['changed_heads']} of {r['heads']} | "
                         f"{r['reason']} | {r['png'] or r['svg']} |")
        lines.append('')
    if errors:
        lines += ['', 'PNG errors (SVG written): ' + '; '.join(errors)]
    (out / 'index.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(backend=backend, rows=rows, errors=errors)
