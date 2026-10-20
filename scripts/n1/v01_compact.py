"""N1 v0.1: per-stratum values from the evaluations' ``compact-<panel>.json``, side by side (numpy-free).

Each ``--run label=<eval dir>`` contributes its last checkpoint's arm (``v0@<tag>``), and its checkpoint spread
(max - min over the v0 arms it holds); the first run also contributes R2. For BL-20 non-free, level,
bars_start, bars_same and reset recall, per stratum. Characterises; the decision tables come from
`v01_compare.py` on matched recordings.

    python scripts/n1/v01_compact.py --run 'v0 s0=runs/v0-s0-20261019/eval' --run 'v0 s1=...' --run 'v0.1 s0=...' \
        [--panels panel,val_hard] --out-md X
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STRATA = ['all'] + [f'S{k}' for k in range(10)]
MEASURES = ('bl20_nonfree', 'level', 'bars_start', 'bars_same', 'reset_recall')


def _v(row, m):
    x = (row or {}).get(m) or {}
    v = x.get('value')
    return v if v is not None and math.isfinite(v) else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--panels', default='panel,val_hard')
    p.add_argument('--out-md', required=True)
    a = p.parse_args(argv)
    runs = [r.split('=', 1) for r in a.run]
    lines = []
    for panel in a.panels.split(','):
        comp = {lab: json.load(open(Path(d) / f'compact-{panel}.json'))['tables'] for lab, d in runs}
        arms = {lab: sorted(k for k in t if k.startswith('v0@') and '|' not in k) for lab, t in comp.items()}
        r2 = comp[runs[0][0]]['r2']
        for m in MEASURES:
            labs = [lab for lab, _ in runs]
            lines += [f'**{m}, {panel}** (value; checkpoint spread in brackets)', '',
                      '| Stratum | n | R2 | ' + ' | '.join(labs) + ' |', '|' + ' --- |' * (3 + len(labs))]
            for st in STRATA:
                vals = []
                n = None
                for lab in labs:
                    t, ar = comp[lab], arms[lab]
                    v = _v(t[ar[-1]].get(st), m)
                    n = n or (t[ar[-1]].get(st) or {}).get('n')
                    cs = [_v(t[x].get(st), m) for x in ar]
                    cs = [c for c in cs if c is not None]
                    sp = f' ({max(cs) - min(cs):.3f})' if len(cs) > 1 else ''
                    vals.append('' if v is None else f'{v:.3f}{sp}')
                rv = _v(r2.get(st), m)
                if n is None and rv is None:
                    continue
                lines.append(f"| {st} | {n or ''} | {'' if rv is None else f'{rv:.3f}'} | " + ' | '.join(vals) + ' |')
            lines.append('')
    Path(a.out_md).write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
