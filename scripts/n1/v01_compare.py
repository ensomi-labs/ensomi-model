"""N1 v0.1 against both v0 seeds, its own minute-100 checkpoint and R2, by the v0.1 report's comparison rule.

Reads `scripts/n1/v0_analysis.py` items files (branch ``n1/v0-eval``; one directory per run, holding
``items-<panel>.jsonl`` with every checkpoint decoded) and writes markdown tables and a JSON of every
cell. numpy only; run on the control plane with that script's directory importable:

    python scripts/n1/v01_compare.py --analysis <dir of v0_analysis.py> \
        --v0 <v0 s0 items dir> --v0 <v0 s1 items dir> --v01 <v0.1 s0 items dir> [--v01 <v0.1 s1 items dir>] \
        [--ctrl <v0.1 s0 control items dir> [--ctrl <v0.1 s1 control items dir>]] --out-json X --out-md Y

Rule (v0.1 report §3), per panel, test (``N`` = BL-20 non-free on full spans, ``Nn`` = on reader-neutral
heads) and stratum, on the recordings every compared arm scores (matched rows):
- each run's value is its last checkpoint's; its checkpoint spread c is max - min over its checkpoints;
- m = max(seed spreads, checkpoint spreads); m_c = max(checkpoint spreads);
- finishing: **helps** when v0.1 is above both v0 seeds, v0.1 - mean(v0) > m, and v0.1 - control > m_c with its
  paired interval above 0 (every v0.1 seed's); **hurts** the mirror; **no change** when |v0.1 - mean(v0)| <= m and
  |v0.1 - control| <= m_c; otherwise **unresolved**. Without a control the control conditions are dropped and
  the class is marked so;
- against R2: **ahead** when Δ > m and every v0.1 seed's paired interval lies above 0, **behind** the mirror,
  **level** when |Δ| <= m, otherwise **unresolved**. Cells under 10 recordings are shown, not classed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

STRATA = ['all'] + [f'S{k}' for k in range(10)]
TESTS = {'N': 'BL-20 non-free, full spans', 'Nn': 'BL-20 non-free, reader-neutral heads (1b)'}


def _load_analysis(path: Path):
    spec = importlib.util.spec_from_file_location('v0_analysis', path / 'v0_analysis.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Run:
    def __init__(self, A, d: Path, panel: str, label: str):
        self.items = A.load_items(d / f'items-{panel}.jsonl')
        self.arms = sorted({a for x in self.items for a in x['arms'] if a.startswith('v0@')})
        self.last, self.label, self.A = self.arms[-1], label, A

    def rec(self, arm, test, st):
        return self.A.per_recording(self.items, arm, test, st, 'bl')


def cell(A, runs: dict, test: str, st: str) -> dict | None:
    v0s, v01s, ctrls, r2src = runs['v0'], runs['v01'], runs['ctrl'], runs['v0'][0]
    r2 = r2src.rec('r2', test, st)
    lasts = {r.label: r.rec(r.last, test, st) for r in v0s + v01s + ctrls}
    common = set(r2)
    for d in lasts.values():
        common &= set(d)
    if not common:
        return None
    keys = sorted(common)
    mean = lambda d: float(np.mean([d[k] for k in keys]))
    vals = {lab: mean(d) for lab, d in lasts.items()}
    spread = {}
    for r in v0s + v01s + ctrls:
        if len(r.arms) > 1:
            cv = [mean(r.rec(a, test, st)) if common <= set(r.rec(a, test, st)) else math.nan for a in r.arms]
            spread[r.label] = float(np.nanmax(cv) - np.nanmin(cv))
    v0v = [vals[r.label] for r in v0s]
    v01v = [vals[r.label] for r in v01s]
    sig_v0 = max(v0v) - min(v0v) if len(v0v) > 1 else 0.0
    sig_v01 = max(v01v) - min(v01v) if len(v01v) > 1 else 0.0
    cs = [spread.get(r.label, 0.0) for r in v0s + v01s]
    m = max([sig_v0, sig_v01] + cs)
    m_c = max(cs)
    v01m, v0m = float(np.mean(v01v)), float(np.mean(v0v))
    pd = lambda a, b: A.paired_delta(a, b, keys)
    out = dict(n=len(keys), r2=mean(r2), values=vals, spread=spread, sigma_v0=sig_v0, sigma_v01=sig_v01, m=m, m_c=m_c,
               d_v0=v01m - v0m, d_r2=v01m - mean(r2))
    out['paired_v01_v0'] = {f'{a.label}-{b.label}': pd(lasts[a.label], lasts[b.label]) for a in v01s for b in v0s}
    out['paired_v01_r2'] = {a.label: pd(lasts[a.label], r2) for a in v01s}
    out['paired_v0_r2'] = {b.label: pd(lasts[b.label], r2) for b in v0s}
    out['paired_ctrl'] = {a.label: pd(lasts[a.label], lasts[c.label]) for a, c in zip(v01s, ctrls)}
    if len(keys) < 10:
        out['finish'] = out['vs_r2'] = 'n<10'
        return out
    above = min(v01v) > max(v0v) and out['d_v0'] > m
    below = max(v01v) < min(v0v) and out['d_v0'] < -m
    if ctrls:
        cd = list(out['paired_ctrl'].values())
        c_up = all(c['delta'] > m_c and c['lo'] > 0 for c in cd)
        c_dn = all(c['delta'] < -m_c and c['hi'] < 0 for c in cd)
        c_flat = all(abs(c['delta']) <= m_c for c in cd)
        out['finish'] = ('helps' if above and c_up else 'hurts' if below and c_dn else
                         'no change' if abs(out['d_v0']) <= m and c_flat else 'unresolved')
    else:
        out['finish'] = ('above v0 (no control)' if above else 'below v0 (no control)' if below else
                         'within m (no control)' if abs(out['d_v0']) <= m else 'unresolved (no control)')
    pr = list(out['paired_v01_r2'].values())
    out['vs_r2'] = ('ahead' if out['d_r2'] > m and all(p['lo'] > 0 for p in pr) else
                    'behind' if out['d_r2'] < -m and all(p['hi'] < 0 for p in pr) else
                    'level' if abs(out['d_r2']) <= m else 'unresolved')
    return out


def _f(x, nd=3):
    return '' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.{nd}f}'


def _s(x, nd=3):
    return '' if x is None or not math.isfinite(x) else f'{x:+.{nd}f}'


def _ci(p):
    return '' if not p else f"{_s(p['delta'])} [{_s(p['lo'])}, {_s(p['hi'])}]"


def md(res: dict, labels: dict) -> str:
    lines = []
    for panel, tests in res.items():
        for test, rows in tests.items():
            lines += [f'**{TESTS[test]}, {panel}**', '']
            v0l, v01l, cl = labels['v0'], labels['v01'], labels['ctrl']
            head = (['Stratum', 'n', 'R2'] + v0l + v01l + cl + [f'{a} − {b}' for a in v01l for b in v0l] +
                    [f'{a} − {c}' for a, c in zip(v01l, cl)] + ['m', 'm_c', 'finishing'] +
                    [f'{a} − R2' for a in v01l] + ['v0 − R2 (mean)', 'vs R2'])
            lines += ['| ' + ' | '.join(head) + ' |', '|' + ' --- |' * len(head)]
            for st in STRATA:
                c = rows.get(st)
                if c is None:
                    continue
                v0d = np.mean([c['paired_v0_r2'][b]['delta'] for b in v0l])
                row = ([st, str(c['n']), _f(c['r2'])] + [_f(c['values'][x]) for x in v0l + v01l + cl] +
                       [_ci(c['paired_v01_v0'][f'{a}-{b}']) for a in v01l for b in v0l] +
                       [_ci(c['paired_ctrl'][a]) for a in v01l[:len(cl)]] +
                       [_f(c['m']), _f(c['m_c']), c['finish']] + [_ci(c['paired_v01_r2'][a]) for a in v01l] +
                       [_s(float(v0d)), c['vs_r2']])
                lines.append('| ' + ' | '.join(row) + ' |')
            lines.append('')
    return '\n'.join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', required=True, help='directory holding v0_analysis.py')
    p.add_argument('--v0', action='append', required=True)
    p.add_argument('--v01', action='append', required=True)
    p.add_argument('--ctrl', action='append', default=[])
    p.add_argument('--panels', default='panel,val_hard')
    p.add_argument('--tests', default='N,Nn')
    p.add_argument('--out-json', required=True)
    p.add_argument('--out-md', required=True)
    a = p.parse_args(argv)
    A = _load_analysis(Path(a.analysis))
    labels = dict(v0=[f'v0 s{i}' for i in range(len(a.v0))], v01=[f'v0.1 s{i}' for i in range(len(a.v01))],
                  ctrl=[f'v0.1 s{i}@ctrl' for i in range(len(a.ctrl))])
    res = {}
    for panel in a.panels.split(','):
        runs = {k: [Run(A, Path(d), panel, lab) for d, lab in zip(getattr(a, k), labels[k])] for k in ('v0', 'v01', 'ctrl')}
        res[panel] = {t: {st: c for st in STRATA if (c := cell(A, runs, t, st)) is not None} for t in a.tests.split(',')}
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    Path(a.out_md).write_text(md(res, labels))
    print(md(res, labels))


if __name__ == '__main__':
    main()
