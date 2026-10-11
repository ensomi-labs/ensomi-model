"""Analysis and report of harness v0 on the event field: floor, detection, transforms, receipts.

Every table cell the run could not fill prints "untested".
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..corpus import SPLIT_SALT
from .cases import CONDITIONS, COV_EDGES, INJECTION_STARS, SCOPES
from .chain import STATS
from .injections import FAMILIES, NOT_SCORABLE, dose_label, family_spec
from .transforms import TRANSFORMS
from .trivial import FOLD_SALT, group_bootstrap, star_band, star_band_label, wilson

NOMINAL = 0.05
UNTESTED = 'untested'
BOOT_SEED = 20261004


def _rate(flags: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, Any]:
    flags = np.asarray(flags, dtype=bool)
    n, k = int(len(flags)), int(np.sum(flags))
    if n == 0:
        return dict(n=0, k=0, rate=None)
    lo, hi = wilson(k, n)
    blo, bhi = group_bootstrap(flags, groups, seed=seed)
    se = math.sqrt(NOMINAL * (1 - NOMINAL) / n)
    return dict(n=n, k=k, rate=k / n, wilson=[lo, hi], bootstrap=[blo, bhi], groups=int(len(np.unique(groups))),
                two_se=2 * se, within_two_se=abs(k / n - NOMINAL) <= 2 * se)


def _fmt_rate(block: dict | None) -> str:
    if not block or not block.get('n'):
        return UNTESTED
    ci = block.get('bootstrap') or block.get('wilson')
    return f"{100 * block['rate']:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] (n={block['n']})"


def _pct(x) -> str:
    return UNTESTED if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{100 * x:.1f}%'


def _num(x, fmt='.4g') -> str:
    return UNTESTED if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def _uniformity(p: np.ndarray, bins: int = 20) -> dict[str, Any]:
    p = p[np.isfinite(p)]
    if not len(p):
        return dict(n=0)
    hist = np.histogram(p, bins=bins, range=(0, 1))[0]
    xs = np.sort(p)
    ks = float(np.max(np.maximum(np.arange(1, len(xs) + 1) / len(xs) - xs, xs - np.arange(len(xs)) / len(xs))))
    return dict(n=int(len(p)), histogram=hist.tolist(), ks_distance=ks)


class Columns:
    """Records of the calibration pass as arrays."""

    def __init__(self, cal: dict):
        rec = cal['records']
        self.n = len(rec)
        get = lambda k, d=None: [r.get(k, d) for r in rec]  # noqa: E731
        self.kind, self.name = np.array(get('kind')), np.array(get('name'))
        self.dose = np.array([str(x) for x in get('dose', '')])
        self.scope, self.condition = np.array(get('scope')), np.array(get('condition'))
        self.status = np.array(get('status', ''))
        self.same_family = np.array([x is True for x in get('same_family')])
        self.same_full = np.array([x is True for x in get('same_full')])
        self.compared = np.array([x is not None for x in get('same_family')])
        self.noop = np.array([x is True for x in get('noop')])
        self.desc = np.array([np.nan if x is None else float(x) for x in get('desc')])
        self.seam = np.array([x or 0 for x in get('seam_repaired', 0)])
        self.requested = np.array([np.nan if x is None else float(x) for x in get('requested')])
        self.done = np.array([np.nan if x is None else float(x) for x in get('done')])
        self.chart = cal['chart']
        self.p = cal['p']
        self.stats = cal['stats']
        self.seconds = cal['seconds']
        self.n_eff = cal['n_eff']
        for k in ('scored_heads', 'ranked_heads', 'scored_events', 'scored_events_with_score', 'scored_releases',
                  'releases_with_score'):
            setattr(self, k, np.array([r.get(k, 0) or 0 for r in rec], dtype=np.int64))
        self.reason = get('reason')


def analyse(fit: dict, cal: dict, model_info: dict, args) -> dict[str, Any]:
    rows = cal['rows']
    c = Columns(cal)
    groups = np.array([r['group_id'] for r in rows])
    out: dict[str, Any] = dict(family=model_info['family'], model=model_info)
    out['counts'] = counts(fit, cal, c, rows)
    original = {(int(c.chart[i]), c.scope[i]): i for i in np.flatnonzero((c.kind == 'original'))}
    identity = {(int(c.chart[i]), c.scope[i], c.condition[i]): i for i in np.flatnonzero(c.kind == 'identity')}
    out['coverage'] = coverage(c, original)
    out['floor'] = {sc: floor(c, original, rows, groups, sc) for sc in SCOPES}
    out['null'] = null_summary(fit, c, original)
    quant = description_quantiles(fit)
    cal_desc = [r.get('desc') or {} for r in cal['results']]
    cells, by_cell = [], defaultdict(list)
    for i in np.flatnonzero((c.kind == 'injection') & (c.condition == 'timing')):
        by_cell[(c.name[i], c.dose[i], c.scope[i])].append(i)
    for f in FAMILIES:
        for d in f.doses:
            label = dose_label(d)
            for sc in SCOPES:
                cells.append(detection_cell(f, label, sc, np.array(by_cell.get((f.name, label, sc), []), dtype=np.int64),
                                            c, groups, identity, rows, quant, cal_desc, f.description_of(d)))
    out['detection'] = cells
    out['population'] = population_rows(by_cell, c, groups, identity, args.seed)
    out['plausible_dose'] = plausible_doses(cells)
    out['must_not_flag'] = must_not_flag(c, rows)
    out['covariance'] = covariance_summary(cal, c)
    out['applicability'] = applicability(cells, c)
    out['edge_cases'] = edge_cases(fit, c, original)
    return out


def counts(fit, cal, c: Columns, rows) -> dict[str, Any]:
    res = fit['results']
    return dict(
        fit_charts=len(fit['rows']), fit_errors=int(sum('error' in r for r in res)),
        fit_null_errors=int(sum('error' in r for r in fit['null_results'])),
        fit_error_samples=[(r['path'], r['error']) for r in res + fit['null_results'] if 'error' in r][:20],
        fit_illegal=int(sum(bool(r.get('legal')) for r in res)),
        fit_illegal_codes=dict(sum((Counter(r.get('legal') or {}) for r in res), Counter())),
        fit_heads=int(sum(r.get('heads', 0) for r in res)),
        null_songs=len(fit['null']['whole']), null_windows=len(fit['null']['continuation']),
        calibration_charts=len(rows), calibration_errors=int(sum('error' in r for r in cal['results'])),
        calibration_error_samples=[(r['path'], r['error']) for r in cal['results'] if 'error' in r][:20],
        calibration_illegal_sources=int(sum(bool(r.get('legal')) for r in cal['results'])),
        calibration_illegal_codes=dict(sum((Counter(r.get('legal') or {}) for r in cal['results']), Counter())),
        injection_charts=int(sum(INJECTION_STARS[0] <= r['star'] < INJECTION_STARS[1] for r in rows)),
        injection_skipped=int(sum('inject_skipped' in r for r in cal['results'])),
        cases=c.n, case_status=dict(Counter(f'{k}/{cd}:{s}' for k, cd, s in zip(c.kind, c.condition, c.status))),
        case_errors=[(rows[int(c.chart[i])]['path'], c.kind[i], c.name[i], c.dose[i], c.reason[i])
                     for i in np.flatnonzero(c.status == 'error')][:30])


def coverage(c: Columns, original: dict) -> dict[str, Any]:
    out = {}
    for label, sel in (('original whole', np.array([i for (ch, sc), i in original.items() if sc == 'whole'])),
                       ('original continuation',
                        np.array([i for (ch, sc), i in original.items() if sc == 'continuation'])),
                       ('every timing case', np.flatnonzero((c.condition == 'timing') & (c.status == 'ok')))):
        sel = np.asarray(sel, dtype=np.int64)
        ok = sel[c.status[sel] == 'ok'] if len(sel) else sel
        ev, ev_ok = int(c.scored_events[ok].sum()), int(c.scored_events_with_score[ok].sum())
        hd, hd_ok = int(c.scored_heads[ok].sum()), int(c.ranked_heads[ok].sum())
        rl, rl_ok = int(c.scored_releases[ok].sum()), int(c.releases_with_score[ok].sum())
        short = int(np.sum((c.scored_events_with_score[ok] < c.scored_events[ok]) | (c.ranked_heads[ok] < c.scored_heads[ok])))
        out[label] = dict(cases=int(len(sel)), not_ok=int(len(sel) - len(ok)), events=ev, events_scored=ev_ok,
                          heads=hd, heads_ranked=hd_ok, releases=rl, releases_scored=rl_ok,
                          cases_short_of_full=short, share=(ev_ok / ev) if ev else None,
                          head_share=(hd_ok / hd) if hd else None)
    return out


def floor(c: Columns, original: dict, rows, groups, sc: str) -> dict[str, Any]:
    idx = np.array([original[(ch, sc)] for ch in range(len(rows)) if (ch, sc) in original], dtype=np.int64)
    block: dict[str, Any] = dict(songs=int(len(idx)))
    ok = idx[np.isfinite(c.p['mean_low'][idx]) & np.isfinite(c.p['mean_high'][idx])]
    block['unscored'] = int(len(idx) - len(ok))
    ch = c.chart[ok]
    sb = star_band(np.array([rows[int(x)]['star'] for x in ch]))
    secs = c.seconds[ok]
    edges = np.quantile(secs, [0.2, 0.4, 0.6, 0.8]) if len(secs) else np.zeros(4)
    quint = np.searchsorted(edges, secs, side='right')
    block['duration_quintile_edges_s'] = edges.tolist()
    for view, (lo_name, hi_name) in (('mean', ('mean_low', 'mean_high')), ('worst16', ('worst_low', 'worst_high'))):
        res = {}
        for tail, name in (('low', lo_name), ('high', hi_name)):
            flags = c.p[name][ok] < NOMINAL
            bands = [dict(star_band=star_band_label(int(b)), **_rate(flags[sb == b], groups[ch][sb == b], BOOT_SEED + int(b)))
                     for b in np.unique(sb)]
            quints = [dict(quintile=int(q) + 1, **_rate(flags[quint == q], groups[ch][quint == q], BOOT_SEED + 50 + int(q)))
                      for q in range(5)]
            in_target = (np.array([rows[int(x)]['star'] for x in ch]) >= INJECTION_STARS[0]) & \
                (np.array([rows[int(x)]['star'] for x in ch]) < INJECTION_STARS[1])
            res[tail] = dict(pooled=_rate(flags, groups[ch], BOOT_SEED), by_star_band=bands, by_duration_quintile=quints,
                             pooled_2_to_7=_rate(flags[in_target], groups[ch][in_target], BOOT_SEED + 99),
                             uniformity=_uniformity(c.p[name][ok]))
        block[view] = res
    return block


def null_summary(fit: dict, c: Columns, original: dict) -> dict[str, Any]:
    out = {}
    for sc in SCOPES:
        idx = np.array([i for (ch, s), i in original.items() if s == sc], dtype=np.int64)
        ne = c.n_eff[idx]
        ne = ne[np.isfinite(ne)]
        out[sc] = dict(null_rows=len(fit['null'][sc]),
                       n_eff_calibration_originals=dict(
                           n=int(len(ne)), min=float(ne.min()) if len(ne) else None,
                           p5=float(np.quantile(ne, 0.05)) if len(ne) else None,
                           median=float(np.median(ne)) if len(ne) else None,
                           below_20=int(np.sum(ne < 20)), below_100=int(np.sum(ne < 100))))
    ws = fit['whole_stats']
    out['fit_whole_statistics'] = {k: dict(median=float(np.nanmedian(v)), p95=float(np.nanquantile(v, 0.95)))
                                   for k, v in ws.items() if k in STATS}
    meta = fit['window_meta']
    out['windows_per_shape'] = dict(Counter(f'{a:g}@{b:g}' for a, b in meta)) if len(meta) else {}
    return out


def description_quantiles(fit: dict) -> dict:
    """5th and 95th percentile of each targeted description over fit charts, per star band."""
    values: dict[tuple, list] = defaultdict(list)
    for row, r in zip(fit['rows'], fit['results']):
        if 'error' in r:
            continue
        key = int(star_band(row['star']))
        for k, v in r['desc'].items():
            if v is not None and np.isfinite(v):
                values[(k, key)].append(v)
    return {k: (float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)), len(v)) for k, v in values.items() if len(v) >= 2}


def detection_cell(f, label, sc, idx, c: Columns, groups, identity, rows, quant, cal_desc, desc_name) -> dict[str, Any]:
    cell: dict[str, Any] = dict(family=f.name, title=f.title, dose=label, scope=sc, condition='timing',
                                description=desc_name, cases=int(len(idx)))
    if sc not in f.scopes:
        cell['cell'] = 'not applicable: needs a given span'
        return cell
    st = c.status[idx] if len(idx) else np.array([])
    ok = idx[st == 'ok'] if len(idx) else idx
    cell.update(discarded=int(np.sum(st == 'discarded')), not_applicable=int(np.sum(st == 'n/a')),
                errors=int(np.sum(st == 'error')),
                discard_reasons=dict(Counter(code.split(':')[0] for i in idx if c.status[i] == 'discarded'
                                             for code in (c.reason[i] or '').split(','))))
    attempted = int(len(idx) - cell['not_applicable'] - cell['errors'])
    cell['attempted'] = attempted
    cell['discard_rate'] = cell['discarded'] / attempted if attempted else None
    cell['noop'] = int(np.sum(c.noop[ok]))
    cell['seam_repaired_cases'] = int(np.sum(c.seam[idx] > 0)) if len(idx) else 0
    req, done = c.requested[ok], c.done[ok]
    m = np.isfinite(req) & (req > 0)
    cell['realized_dose'] = float(np.mean(done[m] / req[m])) if m.any() else None
    cell['compared'] = int(np.sum(c.compared[ok]))
    cell['family_identical'] = int(np.sum(c.same_family[ok]))
    scored = ok[np.isfinite(c.p['mean_low'][ok])]
    src = np.array([identity.get((int(c.chart[i]), sc, 'timing'), -1) for i in scored], dtype=np.int64)
    src = src[(src >= 0)]
    src = src[np.isfinite(c.p['mean_low'][src])] if len(src) else src
    cell['unscored'] = int(len(ok) - len(scored))
    seed = int(hashlib.sha256(f'{f.name}{label}{sc}'.encode()).hexdigest()[:6], 16)
    g = groups[c.chart[scored]] if len(scored) else np.array([])
    for key, name, s in (('low', 'mean_low', 0), ('high', 'mean_high', 1), ('worst_low', 'worst_low', 4)):
        cell[key] = _rate(c.p[name][scored] < NOMINAL, g, seed + s) if len(scored) else dict(n=0)
    gs = groups[c.chart[src]] if len(src) else np.array([])
    cell['source_low'] = _rate(c.p['mean_low'][src] < NOMINAL, gs, seed + 2) if len(src) else dict(n=0)
    cell['source_high'] = _rate(c.p['mean_high'][src] < NOMINAL, gs, seed + 3) if len(src) else dict(n=0)
    strata = defaultdict(list)
    for i in scored:
        strata[int(star_band(rows[int(c.chart[i])]['star']))].append(i)
    cell['strata'] = [dict(star_band=star_band_label(k), n=len(v), low=float(np.mean(c.p['mean_low'][v] < NOMINAL)),
                           high=float(np.mean(c.p['mean_high'][v] < NOMINAL)))
                      for k, v in sorted(strata.items()) if len(v) >= 30]
    inside, src_inside = [], []
    want = 'continuation' if f.name == 'F4' else 'whole'
    if sc == want:
        for i in ok:
            r = rows[int(c.chart[i])]
            key = (desc_name, int(star_band(r['star'])))
            v = c.desc[i]
            if key in quant and np.isfinite(v):
                lo, hi, _ = quant[key]
                inside.append(lo <= v <= hi)
                sv = cal_desc[int(c.chart[i])].get(desc_name)
                if sv is not None and np.isfinite(sv):
                    src_inside.append(lo <= sv <= hi)
    cell['plausible_inside_share'] = float(np.mean(inside)) if inside else None
    cell['plausible_n'] = len(inside)
    cell['source_inside_share'] = float(np.mean(src_inside)) if src_inside else None
    cell['plausible'] = (cell['plausible_inside_share'] >= 0.5) if inside else None
    return cell


def population_rows(by_cell, c: Columns, groups, identity, seed) -> list[dict]:
    out = []
    for f in FAMILIES:
        if not f.population:
            continue
        for sc in f.scopes:
            idx = by_cell.get((f.name, dose_label(f.doses[0]), sc), [])
            pairs = [(i, identity.get((int(c.chart[i]), sc, 'timing'))) for i in idx if c.status[i] == 'ok']
            pairs = [(i, j) for i, j in pairs if j is not None and np.isfinite(c.p['mean_low'][i])
                     and np.isfinite(c.p['mean_low'][j])]
            if not pairs:
                out += [dict(family=f.name, scope=sc, fraction=fr, cell=UNTESTED) for fr in f.population]
                continue
            inj, src = np.array([i for i, _ in pairs]), np.array([j for _, j in pairs])
            g = groups[c.chart[inj]]
            for frac in f.population:
                rng = np.random.default_rng([seed, int(frac * 1000), len(f.name)])
                replaced = np.zeros(len(pairs), dtype=bool)
                replaced[rng.choice(len(pairs), size=int(math.floor(frac * len(pairs) + 0.5)), replace=False)] = True
                pick = np.where(replaced, inj, src)
                out.append(dict(family=f.name, scope=sc, fraction=frac, songs=len(pairs), replaced=int(replaced.sum()),
                                low=_rate(c.p['mean_low'][pick] < NOMINAL, g, BOOT_SEED + int(frac * 100)),
                                high=_rate(c.p['mean_high'][pick] < NOMINAL, g, BOOT_SEED + int(frac * 100) + 1)))
    return out


def plausible_doses(cells: list[dict]) -> dict[str, Any]:
    out = {}
    for f in FAMILIES:
        mine = [x for x in cells if x['family'] == f.name and x['scope'] == ('continuation' if f.name == 'F4' else 'whole')]
        flags = {x['dose']: x.get('plausible') for x in mine}
        if f.name == 'D9':
            out[f.name] = {d: _largest(order, flags) for d, order in
                           (('thinning', ['0.8', '0.67', '0.5']), ('thickening', ['1.25', '1.5', '2']))}
        elif f.name in ('D1', 'D10') or f.population or len(f.doses) == 1:
            out[f.name] = dict(plausible_cells=[d for d, v in flags.items() if v],
                               implausible_cells=[d for d, v in flags.items() if v is False],
                               untested_cells=[d for d, v in flags.items() if v is None])
        else:
            out[f.name] = _largest([dose_label(d) for d in f.doses], flags)
    return out


def _largest(order: list[str], flags: dict) -> str:
    best = None
    for d in order:
        v = flags.get(d)
        if v is None:
            return best or UNTESTED
        if not v:
            return best or 'below the smallest dose'
        best = d
    return best or UNTESTED


def must_not_flag(c: Columns, rows) -> list[dict]:
    out = []
    for t in TRANSFORMS:
        for sc in SCOPES:
            for cond in CONDITIONS:
                idx = np.flatnonzero((c.kind == 'transform') & (c.name == t.name) & (c.scope == sc) & (c.condition == cond))
                st = c.status[idx]
                compared = idx[c.compared[idx]]
                bad = compared[~c.same_full[compared]]
                out.append(dict(transform=t.name, title=t.title, scope=sc, condition=cond,
                                judged_by='covariance' if t.stretch else 'full output hash',
                                cases=int(len(idx)), ok=int(np.sum(st == 'ok')), skipped=int(np.sum(st == 'skipped')),
                                not_applicable=int(np.sum(st == 'n/a')), discarded=int(np.sum(st == 'discarded')),
                                errors=int(np.sum(st == 'error')), compared=int(len(compared)),
                                mismatches=int(len(bad)),
                                mismatched_charts=[rows[int(c.chart[i])]['path'] for i in bad[:20]],
                                not_applicable_reasons=dict(Counter(c.reason[i] for i in idx if c.status[i] == 'n/a')),
                                error_samples=[c.reason[i] for i in idx if c.status[i] == 'error'][:5]))
    return out


def covariance_summary(cal: dict, c: Columns) -> list[dict]:
    out = []
    centres = (COV_EDGES[:-1] + COV_EDGES[1:]) / 2
    for t in TRANSFORMS:
        if not t.stretch:
            continue
        idx = np.flatnonzero((c.kind == 'transform') & (c.name == t.name) & (c.scope == 'whole') & (c.condition == 'timing')
                             & (c.status == 'ok'))
        recs = [cal['records'][i] for i in idx if cal['records'][i].get('cov_n')]
        if not recs:
            out.append(dict(transform=t.name, stretch=t.stretch, charts=0, verdict=UNTESTED))
            continue
        hist = np.sum([r['cov_hist'] for r in recs], axis=0)
        cdf = np.cumsum(hist) / hist.sum()
        median_log = float(centres[int(np.searchsorted(cdf, 0.5))])
        medians = np.array([r['cov_median'] for r in recs])
        out.append(dict(transform=t.name, stretch=t.stretch, charts=len(recs), heads=int(hist.sum()),
                        pooled_median_ratio=float(math.exp(median_log)),
                        pooled_within_1pct=float(sum(r['cov_within_1pct'] * r['cov_n'] for r in recs) / hist.sum()),
                        pooled_p5_ratio=float(math.exp(centres[int(np.searchsorted(cdf, 0.05))])),
                        pooled_p95_ratio=float(math.exp(centres[int(np.searchsorted(cdf, 0.95))])),
                        charts_median_within_1pct=float(np.mean(np.abs(medians - 1) <= 0.01)),
                        clipped_heads=int(hist[0] + hist[-1]),
                        not_compared=int(len(idx) - len(recs))))
    return out


def applicability(cells: list[dict], c: Columns) -> list[dict]:
    out = []
    by = {(x['family'], x['dose'], x['scope']): x for x in cells}
    for f in FAMILIES:
        for cond in CONDITIONS:
            for sc in SCOPES:
                if sc not in f.scopes:
                    cell = 'not applicable: needs a given span'
                elif cond == 'timing':
                    n = sum((by[(f.name, dose_label(d), sc)].get('low') or {}).get('n', 0) or 0 for d in f.doses)
                    cell = f'scored ({n} injected songs over {len(f.doses)} doses)' if n else UNTESTED
                elif f.skeleton != 'applicable':
                    cell = f.skeleton
                else:
                    idx = np.flatnonzero((c.kind == 'injection') & (c.name == f.name) & (c.scope == sc) & (c.condition == cond))
                    skipped = int(np.sum(c.status[idx] == 'skipped'))
                    reasons = sorted({c.reason[i] for i in idx if c.status[i] == 'skipped'})
                    cell = (f'skipped by evaluate on {skipped} of {len(idx)} cases ({"; ".join(reasons)})'
                            if len(idx) else UNTESTED)
                out.append(dict(family=f.name, condition=cond, scope=sc, cell=cell))
    for name, why in NOT_SCORABLE.items():
        for cond in CONDITIONS:
            for sc in SCOPES:
                out.append(dict(family=name, condition=cond, scope=sc, cell=why))
    sk = np.flatnonzero((c.kind == 'identity') & (c.condition == 'skeleton'))
    out.append(dict(family='identity', condition='skeleton', scope='both',
                    cell=f'skipped by evaluate on {int(np.sum(c.status[sk] == "skipped"))} of {len(sk)} cases'))
    return out


def edge_cases(fit: dict, c: Columns, original: dict) -> dict[str, Any]:
    idx = np.array(list(original.values()), dtype=np.int64)
    ok = idx[c.status[idx] == 'ok']
    return dict(
        calibration_original_cases_both_scopes=int(len(idx)),
        zero_scored_time=int(np.sum(~(c.seconds[ok] > 0))),
        unscored=int(np.sum(~np.isfinite(c.p['mean_low'][ok]))),
        shortest_scored_s=float(np.nanmin(c.seconds[ok])) if len(ok) else None,
        longest_scored_s=float(np.nanmax(c.seconds[ok])) if len(ok) else None,
        fit_songs_without_null=int(len(fit['rows']) - len(fit['null']['whole'])))


# ---- outputs -------------------------------------------------------------------------------------

def source_files(root: Path) -> list[Path]:
    base = root / 'src' / 'ensomi_model'
    files = sorted(p for p in (base / 'evaluation').glob('*.py'))
    files += sorted((base / 'evaluation' / 'harness').glob('*.py'))
    files += [base / 'osu_core' / 'hitobjects.py', base / 'osu_core' / 'timing.py']
    return files


def receipt(args, argv, phases, out: Path) -> dict[str, Any]:
    from .run import sha256_file
    root = Path('.').resolve()
    prov = None
    if args.provenance and Path(args.provenance).is_file():
        prov = json.loads(Path(args.provenance).read_text())
    spec = family_spec()
    return dict(
        entry_point=' '.join(['.venv/bin/python', '-m', 'ensomi_model.evaluation.harness.run'] + list(argv)),
        git=prov, git_note='HEAD and dirty-file digest of the control plane at launch; the working tree is uncommitted, so every source file used is hashed below',
        source_sha256={str(p.relative_to(root)): sha256_file(p) for p in source_files(root)},
        corpus=args.corpus, corpus_sha256=sha256_file(Path(args.corpus)), split_salt=SPLIT_SALT, fold_salt=FOLD_SALT,
        conditions=list(CONDITIONS), scopes=list(SCOPES), seeds=dict(base=args.seed, bootstrap=BOOT_SEED),
        evaluator=dict(family_spec=getattr(args, 'family', None)),
        family_set_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        fit_limit=args.fit_limit, calibration_limit=args.calibration_limit, phases_s=phases)


def write_outputs(out: Path, fit, cal, analysis, model_info, args, phases, argv) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .run import sha256_file
    rec = cal['records']
    cols = dict(path=[cal['rows'][int(i)]['path'] for i in cal['chart']],
                group_id=[cal['rows'][int(i)]['group_id'] for i in cal['chart']],
                star=cal['star'], seconds=cal['seconds'], n_eff=cal['n_eff'])
    for k in ('kind', 'name', 'scope', 'condition', 'status', 'reason', 'family_hash', 'full_hash'):
        cols[k] = [None if r.get(k) is None else str(r.get(k)) for r in rec]
    cols['dose'] = [str(r.get('dose', '')) for r in rec]
    for k in ('same_family', 'same_full', 'noop'):
        cols[k] = [r.get(k) for r in rec]
    for k in ('seam_repaired', 'requested', 'done', 'scored_heads', 'ranked_heads', 'scored_events',
              'scored_events_with_score'):
        cols[k] = [None if r.get(k) is None else float(r.get(k)) for r in rec]
    cols['desc'] = [None if r.get('desc') is None or not np.isfinite(r['desc']) else float(r['desc']) for r in rec]
    for name in STATS:
        cols[name] = cal['stats'][name]
        cols[f'p_{name}'] = cal['p'][name]
    pq.write_table(pa.table(cols), out / 'cases.parquet', compression='zstd')
    pq.write_table(pa.table(dict(path=[r['path'] for r in fit['rows']], group_id=[r['group_id'] for r in fit['rows']],
                                 star=fit['star'], fold=fit['fold'], seconds=fit['whole_stats']['seconds'],
                                 **{k: fit['whole_stats'][k] for k in STATS})), out / 'null_whole.parquet',
                   compression='zstd')
    analysis['receipt'] = receipt(args, argv, phases, out)
    analysis['receipt']['outputs_sha256'] = {p.name: sha256_file(p) for p in (out / 'cases.parquet', out / 'null_whole.parquet',
                                                                              out / 'model.json', out / 'access_log.jsonl')}
    (out / 'report.json').write_text(json.dumps(analysis, indent=1, default=_json) + '\n')
    (out / 'report.md').write_text(render(analysis))
    final = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'receipt.json'}
    (out / 'receipt.json').write_text(json.dumps(dict(outputs_sha256=final), indent=1) + '\n')


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _plaus(cell: dict) -> str:
    if cell['scope'] != ('continuation' if cell['family'] == 'F4' else 'whole'):
        return 'n/a'
    return _pct(cell.get('plausible_inside_share')) if cell.get('plausible_n') else UNTESTED


def render(a: dict) -> str:
    L = []
    r = a['receipt']
    m = a['model']
    es = m['estimator_settings']
    head = (f"Family `{a['family']}`" + (f": heads per second around each head, Gaussian kernel sigma {es['rate_kernel_sigma_s']} s, "
                                         'edge-corrected; key: requested star in bands of 0.5' if es.get('rate_kernel_sigma_s') else '')
            + f"; per-event surprisal ranks in both tails, every song scored with the model and reference without its "
            f"fold; field: -log u smoothed with h = {es['field_kernel_h_s']} s; scope statistic: time-average of the "
            f"field over the scored spans (weights in seconds); worst {es['scan_s']:g} s sub-span; "
            f"song null kernel-weighted in log scored seconds (h {es['null_h_log_seconds']}) and star (h {es['null_h_star']}).")
    L += [f"# Calibration harness v0: family `{a['family']}`", '', head, '',
          '## Receipt', '',
          f"- Entry point: `{r['entry_point']}`",
          f"- Git (control plane at launch): `{(r.get('git') or {}).get('head', UNTESTED)}` on `{(r.get('git') or {}).get('branch', UNTESTED)}`, "
          f"dirty files {(r.get('git') or {}).get('dirty_files', UNTESTED)}, diff digest `{(r.get('git') or {}).get('diff_sha256_16', UNTESTED)}`",
          f"- Corpus `{r['corpus']}` SHA-256 `{r['corpus_sha256']}`",
          f"- Split salt `{r['split_salt']}`, fold salt `{r['fold_salt']}`, seeds {r['seeds']}",
          f"- Conditions {r['conditions']}, scopes {r['scopes']}, family-set SHA-256 `{r['family_set_sha256'][:16]}`",
          f"- Subsample: fit {r['fit_limit'] or 'all'}, calibration {r['calibration_limit'] or 'all'} song groups; "
          f"phases (s) {json.dumps({k: (round(v) if isinstance(v, (int, float)) else v) for k, v in r['phases_s'].items()})}",
          '- Source files (SHA-256, first 16):', '']
    L += [f'  - `{p}` `{h[:16]}`' for p, h in r['source_sha256'].items()]
    L += ['', '- Outputs (SHA-256, first 16): ' + ', '.join(f'`{k}` `{v[:16]}`' for k, v in r['outputs_sha256'].items()), '']
    c = a['counts']
    L += ['## Counts', '',
          f"Fit charts {c['fit_charts']} (errors {c['fit_errors']} and {c['fit_null_errors']} in the two passes; illegal as read "
          f"{c['fit_illegal']} {c['fit_illegal_codes']}; {c['fit_heads']} heads); song null: {c['null_songs']} fit songs, "
          f"{c['null_windows']} continuation windows. Calibration charts {c['calibration_charts']} (errors {c['calibration_errors']}, "
          f"illegal after round trip {c['calibration_illegal_sources']} {c['calibration_illegal_codes']}); injected charts "
          f"(2 <= star < 7) {c['injection_charts']}, skipped as illegal sources {c['injection_skipped']}; cases {c['cases']}.", '']
    if c['case_errors'] or c['fit_error_samples'] or c['calibration_error_samples']:
        L += ['Errors (first samples):', ''] + [f'- {e}' for e in (c['fit_error_samples'] + c['calibration_error_samples']
                                                                    + c['case_errors'])[:30]] + ['']
    for sc in SCOPES:
        fl = a['floor'][sc]
        for view in ('mean', 'worst16'):
            what = 'time-average' if view == 'mean' else 'worst 16 s sub-span'
            lo, hi = fl[view]['low'], fl[view]['high']
            L += [f'## Floor: false-alarm rate at nominal 5%, {sc} scope, {what}', '',
                  f"Songs {fl['songs']}, unscored {fl['unscored']}. Group-bootstrap 95% intervals; "
                  'within 2SE uses the binomial SE at 5%.', '',
                  '| Stratum | Low tail | within 2SE | High tail | within 2SE |', '| --- | --- | --- | --- | --- |']
            L.append(f"| pooled | {_fmt_rate(lo['pooled'])} | {lo['pooled'].get('within_two_se', UNTESTED)} | {_fmt_rate(hi['pooled'])} | {hi['pooled'].get('within_two_se', UNTESTED)} |")
            L.append(f"| 2-7 stars | {_fmt_rate(lo['pooled_2_to_7'])} | {lo['pooled_2_to_7'].get('within_two_se', UNTESTED)} | {_fmt_rate(hi['pooled_2_to_7'])} | {hi['pooled_2_to_7'].get('within_two_se', UNTESTED)} |")
            for bl, bh in zip(lo['by_star_band'], hi['by_star_band']):
                small = '' if bl['n'] >= 100 else ' (n < 100)'
                L.append(f"| star {bl['star_band']}{small} | {_fmt_rate(bl)} | {bl.get('within_two_se', UNTESTED)} | {_fmt_rate(bh)} | {bh.get('within_two_se', UNTESTED)} |")
            edges = fl['duration_quintile_edges_s']
            for ql, qh in zip(lo['by_duration_quintile'], hi['by_duration_quintile']):
                q = ql['quintile']
                span = f"{'' if q == 1 else f'{edges[q - 2]:.0f}'}..{'' if q == 5 else f'{edges[q - 1]:.0f}'} s"
                L.append(f"| duration quintile {q} ({span}) | {_fmt_rate(ql)} | {ql.get('within_two_se', UNTESTED)} | {_fmt_rate(qh)} | {qh.get('within_two_se', UNTESTED)} |")
            L += ['', f"p-value KS distance to uniform: low {_num(lo['uniformity'].get('ks_distance'))}, "
                  f"high {_num(hi['uniformity'].get('ks_distance'))}.", '']
    L += ['', '## Detection by family, dose and scope (timing condition)', '',
          'Detection = share of injected songs with song p < 0.05 (time-average statistic) per tail, group-bootstrap 95% interval. '
          'Identical = injected family hash equal to the source\'s. Realized = mean share of requested edits made. '
          'Seam = cases with a repaired seam conflict. Plausible = share of injected charts whose targeted description is inside '
          'the central 90% of fit charts at their star band (whole scope; F4 continuation).', '',
          '| Family | Dose | Scope | Low tail | High tail | Source low | Discarded | No-op | Identical | Realized | Seam | Plausible |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in a['detection']:
        if 'cell' in x:
            L.append(f"| {x['family']} | {x['dose']} | {x['scope']} | {x['cell']} | | | | | | | | |")
            continue
        ident = f"{x['family_identical']}/{x['compared']}" if x.get('compared') else UNTESTED
        disc = f"{x['discarded']} ({_pct(x['discard_rate'])})" if x.get('discard_rate') is not None else UNTESTED
        L.append(f"| {x['family']} | {x['dose']} | {x['scope']} | {_fmt_rate(x.get('low'))} | {_fmt_rate(x.get('high'))} | "
                 f"{_fmt_rate(x.get('source_low'))} | {disc} | {x.get('noop', UNTESTED)} | {ident} | {_pct(x.get('realized_dose'))} | "
                 f"{x.get('seam_repaired_cases', 0)} | {_plaus(x)} |")
    L += ['', '### Population families (dose = fraction of songs replaced)', '', '| Family | Scope | Fraction | Low tail | High tail |',
          '| --- | --- | --- | --- | --- |']
    for p in a['population']:
        L.append(f"| {p['family']} | {p['scope']} | {p['fraction']} | {_fmt_rate(p.get('low'))} | {_fmt_rate(p.get('high'))} |")
    L += ['', 'Population coverage (P to Q) is not implemented in v0: F2 and F3 are read per song only.', '',
          '### Corpus-plausible dose', '']
    L += [f'- {fam}: {json.dumps(v)}' for fam, v in a['plausible_dose'].items()]
    L += ['', '## Must-not-flag and time-stretch', '',
          '| Transform | Scope | Condition | Judged by | OK | Skipped | Compared | Differ | n/a | Discarded | Errors |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in a['must_not_flag']:
        L.append(f"| {x['title']} | {x['scope']} | {x['condition']} | {x['judged_by']} | {x['ok']} | {x['skipped']} | {x['compared']} | "
                 f"{x['mismatches'] if x['judged_by'] != 'covariance' else 'n/a (expected)'} | "
                 f"{x['not_applicable']} | {x['discarded']} | {x['errors']} |")
    for x in a['must_not_flag']:
        if x['mismatched_charts'] and x['judged_by'] != 'covariance':
            L += ['', f"Differing charts, {x['transform']} / {x['scope']} / {x['condition']} (first 20):", '']
            L += [f'- `{pth}`' for pth in x['mismatched_charts']]
        if x['not_applicable_reasons'] and x['condition'] == 'timing':
            L.append(f"- {x['transform']} / {x['scope']} not applicable: {x['not_applicable_reasons']}")
        if x['error_samples']:
            L.append(f"- {x['transform']} / {x['scope']} / {x['condition']} errors: {x['error_samples']}")
    L += ['', 'Time-stretch covariance (whole scope, timing): ratio = stretched per-head rate x s / source per-head rate, heads in time order.', '',
          '| Transform | Charts | Heads | Pooled median ratio | p5 | p95 | Heads within 1% | Charts with median within 1% |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in a['covariance']:
        if not x.get('charts'):
            L.append(f"| {x['transform']} | 0 | | {UNTESTED} | | | | |")
            continue
        L.append(f"| {x['transform']} | {x['charts']} | {x['heads']} | {x['pooled_median_ratio']:.4f} | {x['pooled_p5_ratio']:.4f} | "
                 f"{x['pooled_p95_ratio']:.4f} | {_pct(x['pooled_within_1pct'])} | {_pct(x['charts_median_within_1pct'])} |")
    cov = a['coverage']
    L += ['', '## Coverage', '', '| Cases | Charts | Events scored | Heads ranked | Releases scored | Cases short of 100% |',
          '| --- | --- | --- | --- | --- | --- |']
    for k, v in cov.items():
        L.append(f"| {k} | {v['cases']} ({v['not_ok']} not evaluated) | {v['events_scored']}/{v['events']} | {v['heads_ranked']}/{v['heads']} | "
                 f"{v['releases_scored']}/{v['releases']} | {v['cases_short_of_full']} |")
    L += ['', '## Applicability (family x condition x scope)', '', '| Family | Condition | Scope | Cell |', '| --- | --- | --- | --- |']
    L += [f"| {x['family']} | {x['condition']} | {x['scope']} | {x['cell']} |" for x in a['applicability']]
    L += ['', '## Model and estimator settings', '',
          f"- Value: {m['value']}; bins from {m['bins']['start']} by {m['bins']['width']} ({m['bins']['count']}); "
          f"bandwidth {m['bandwidth']} chosen from {m['bandwidth_candidates']} by cross-fitted log-likelihood "
          f"{json.dumps({k: round(v) for k, v in m['cross_fitted_loglik'].items()})}; pseudo-events {m['pseudo_events']}; "
          f"{m['k_folds']} folds by song group (salt `{m['fold_salt']}`). Fixed before calibration was read.",
          f"- Estimator settings: {json.dumps(es)}", '',
          '| Star band | Fit heads | Modal rate (/s) | Rate p5 | p50 | p95 |', '| --- | --- | --- | --- | --- | --- |']
    for band, v in m['keys'].items():
        q = v['rate_quantiles']
        L.append(f"| {band} | {v['events']} | {_num(v['mode_rate'], '.3g')} | {q['p5']} | {q['p50']} | {q['p95']} |")
    nl = a['null']
    L += ['', 'Song nulls: ' + '; '.join(f"{sc}: {nl[sc]['null_rows']} rows, n_eff over calibration originals "
                                          f"min {_num(nl[sc]['n_eff_calibration_originals']['min'], '.1f')}, p5 "
                                          f"{_num(nl[sc]['n_eff_calibration_originals']['p5'], '.1f')}, median "
                                          f"{_num(nl[sc]['n_eff_calibration_originals']['median'], '.1f')}, below 20: "
                                          f"{nl[sc]['n_eff_calibration_originals']['below_20']}, below 100: "
                                          f"{nl[sc]['n_eff_calibration_originals']['below_100']}" for sc in SCOPES),
          f"Windows per shape (fraction@position): {nl['windows_per_shape']}.",
          f"Fit-song statistics (median, p95): {json.dumps({k: [round(v['median'], 3), round(v['p95'], 3)] for k, v in nl['fit_whole_statistics'].items()})}.",
          '', f"Edge cases: {json.dumps(a['edge_cases'])}", '']
    return '\n'.join(L)
