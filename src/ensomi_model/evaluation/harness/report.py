"""Analysis and report of harness v0 on the event field: claims, floor, detection, transforms, receipts.

Every table cell the run could not fill prints "untested". The pre-registered
claims E-C0 to E-C6 of run-2 are judged exactly as written in the brief; every
other number is post hoc and labelled so.
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
DIAGNOSIS_JOB = '20261003-hv0b-diag1'
# E-C3: families that leave head and release times unchanged (lane-only: D1, D6; D7 as the brief lists it) and
# families that change only releases (D2, D2s, D3same, D3other, D4, F3, F6).
E_C3_FAMILIES = ('D1', 'D6', 'D7', 'D2', 'D2s', 'D3same', 'D3other', 'D4', 'F3', 'F6')

DIAGNOSES = [
    dict(failure='C4: time-stretch changed the beat-clock output (75 to 235 charts per cell)',
         verdict='expected consequence of the old design; not a transform bug',
         evidence=f'job {DIAGNOSIS_JOB}, run-2/diag/diagnosis-run1.json. The transform multiplies every time and '
                  'beat length exactly. With the stretched grid supplied, 119 of run-1\'s 154 x1.05 mismatches '
                  'change a head\'s bar only after the .osu integer-ms rounding and 35 also in exact float times '
                  '(x0.95: 83 and 5 of 88); 94% of the heads that change bar change snap denominator. On 300 '
                  'random calibration charts at x1.05, 37 change bars in exact float times and 20 more after '
                  'rounding. The bar of a head depends on absolute 2 ms tolerances (beats.locate snapping, bar '
                  'starts within 2 ms of a beat, redlines drift checks) that do not scale with a stretch; with the '
                  'chart\'s own red lines the line roles change too (87 of 154 mismatched charts after rounding).',
         fix='none needed in the harness; the event-field evaluator reads no grid, and E-C5 (covariance) '
             'replaces the stretch invariance. The 80/160 fold restriction on the stretch is dropped.'),
    dict(failure='C4: added expressive red lines changed the output on 5 charts (and 1 error)',
         verdict='harness bug (transform), fixed after run-1 by the previous worker; verified here',
         evidence=f'job {DIAGNOSIS_JOB}: the inserted stop landed after an existing line at the same offset and '
                  'superseded it. With the stop inserted before any line at its offset, all 6 charts keep '
                  'their grid and every inserted line is classified expressive.',
         fix='insert before any line at the same offset (transforms.add_expressive); the unit test fixture of '
             'that fix was wrong (median head at 8400 ms, not 8000) and is corrected.'),
    dict(failure='C5: D6 0.25 discarded 36%, D7 0.25 85%, F1 0.1 48% (and D10 all_quad 26% under continuation)',
         verdict='harness bug: the injections wrote illegal charts they could have avoided',
         evidence=f'job {DIAGNOSIS_JOB}, 300 calibration charts at 2 to 7 stars: every discard is an overlap on a '
                  'chart with long notes (D6 90 of 270 LN charts, 0 of 30 without; D7 251 and 0; F1 141 and 0); '
                  'D10 all_quad under continuation: 74 discards, every one with a given hold crossing the cut. '
                  'D6 swapped lanes inside units that holds cross; D7 moved holds with their rows onto later '
                  'notes; F1 copied a bar under a hold reaching in from the bar before (and into bars starting '
                  'before the song); the continuation splice put scored notes under given holds.',
         fix='D6 moves unit edges to instants no right-hand hold crosses; D7 redraws a bar\'s permutation up to '
             '20 times, else takes another bar; D10 loop, F1 and F2 place each copied block only where the chart '
             'stays legal; copies are rounded to integer ms before the check; the continuation splice repairs '
             'seam conflicts (in-place families revert the object, others drop it) and counts them.'),
]

BUG_FIXES = [
    ('smoke-1 (previous worker)', 'Continuation cut exactly at a bar start: heads snapped onto the first scored bar were given by ms but scored by bar.', 'Obsolete: the continuation cut is now in time (first 25% of the event extent) and the evaluator reads no bars.'),
    ('smoke-2 (previous worker)', 'Per-segment renotation applied to the chart\'s red lines changes scroll speed and red-line roles.', 'Renotate the condition grid per segment; kept.'),
    ('run-1 (previous worker)', 'Expressive-lines transform inserted the stop after an existing line at the same offset.', 'Insert before any line at the same offset; kept, test fixture corrected in run-2.'),
    ('run-1 (previous worker)', 'Single-edit injections checked lane freeness on fractional times, then the writer rounded them.', 'Round each new time to integer ms before the check; kept, extended to copied blocks.'),
    ('run-2 review', 'D6, D7, F1, F2, D10 loop and the continuation splice wrote avoidable illegal charts.', 'Legal construction (see the diagnoses).'),
    ('run-2 review', 'F4 spliced from the bar-based quarter of bars while the scope cut is in time.', 'F4 copies from the bar of each chart\'s own time cut.'),
]

OBSERVATIONS = [
    'Corpus charts are not all legal as read; fit charts are used as read for the model, calibration charts that are illegal after the round trip are not injected.',
    'Placement units (four canonical bars inside one musical segment) remain the harness\'s way to choose where D4, D6 and F2 edit; the evaluator never reads them.',
    'Under the skeleton condition the head-rate family judges head times, which the skeleton fixes, so evaluate skips it on every case; the applicability table counts those skips.',
]

FAILED_PATHS = [
    'Calibration songs scored with the model on every fit fold while the null\'s fit songs were scored without their own fold: not exchangeable (the less noisy model never reaches the reference\'s most typical surprisals). Smoke-5 (job 20261003-hv0b-smoke5, 107 calibration charts, not a result) flagged 0 of 107 songs on the high tail. Fixed before run-2: every song is scored as a fit song of its group\'s fold.',
    'Ranks on a fine histogram of surprisal (1e-3 nats) were rejected before coding: near the mode one bin would hold about 4% of events and cap the high tail.',
    'A subsampled reference per key was rejected for exact mid-p ranks on the binned density (at most 5 x 5000 distinct surprisals per key).',
    'A song null over unique (star, exact duration) pairs was estimated at 10^9+ operations for the window null; log duration is rounded to 0.005 for the weights instead.',
]

LEDGER = [
    ('Is a "song" a chart or a song group?', 'A chart (one .osu) is the unit of the song statistic, the floor and detection; the song group is the bootstrap unit and the fold unit.', 'Kept from run-1; star is per chart.'),
    ('Which scope is E-C1 judged on?', 'Whole scope. The continuation floor is reported post hoc.', 'E-C1 names no scope; run-1 judged C1 on the whole scope.'),
    ('Duration quintiles in E-C1', 'Quintiles of the scored seconds of the calibration songs with a p-value, equal counts.', 'Stratification only; durations are not results.'),
    ('Which events does the head-rate family judge, and what does E-C3 compare?', 'Heads. Its field and statistics are built from heads; every release in the scored spans receives the field value at its time (E-C0). E-C3 compares the family hash (heads, their values and ranks, the field at heads, statistics, scored time); the release readouts are in the full hash only, which E-C4 compares and E-C3 reports post hoc.', 'The brief: the family reads heads only, so release-only changes leave its output identical; a release readout is the same field read at another time.'),
    ('Domain of the time-average', 'Scored spans clipped to the extent of the heads the family reads; releases after the last head get a readout but add no time.', 'Keeps the head family\'s statistic a function of heads only.'),
    ('Field over given events?', 'No: the field and its readouts use scored heads only; given heads enter as context of the head rate.', 'Given spans are context; scored spans are judged.'),
    ('Which families does E-C3 cover?', 'D1, D6, D7 (head and release times unchanged as the brief lists them; D1 and D6 are lane-only) and D2, D2s, D3same, D3other, D4, F3, F6 (releases only). D8 and F5, flagged head-preserving in v0 because they keep head counts per bar, move head times and are not in the claim.', 'The brief names D6, D7, lane-only changes, D2, D3, off-grid releases and never-LN; D4 also changes releases only.'),
    ('Edge correction of the head rate', 'The kernel mass inside the readable heads\' extent is divided out.', 'Without it the first and last 4 s of every chart read half as dense; a generated chart would inherit the same artefact.'),
    ('Key', 'Star bands of 0.5 (the requested star, never recomputed).', 'Brief allows bands or a kernel; bands give exact per-key references.'),
    ('Density estimator', 'Binned (0.002 log rate) Gaussian KDE per key, bandwidth chosen from {0.01, 0.02, 0.04, 0.08, 0.16, 0.32} by cross-fitted log-likelihood on fit, one pseudo-event spread uniformly.', 'A rule fixed before calibration; the pseudo-event keeps every surprisal finite.'),
    ('Bandwidth grid extended', 'The grid was {0.01 to 0.08} until the first smoke run (job 20261003-hv0b-smoke4, killed after its fit pass) chose 0.08, its upper edge, on 1,155 fit charts; 0.16 and 0.32 were added before any calibration chart of that run was scored.', 'A selection at the edge of its grid is not a selection; the decision used fit-split likelihoods only.'),
    ('Tails', 'u_low = share of cross-fitted fit events at the key at least as surprising (mid-p), u_high = share at most as surprising.', 'Same convention as run-1; E-C2 expects both D9 directions on the low tail.'),
    ('Continuation cut', 'First 25% of the time from the chart\'s first to last event (heads and releases), from the source chart; grid-free.', 'Brief: first 25% of time given.'),
    ('Continuation null', 'Six same-shape windows per fit song (fractions 1, 3/4, 3/4, 1/2, 1/2, 1/2 of its extent at positions 0; 0, 1/4; 0, 1/4, 1/2), each with its first 25% given and only its own heads read; kernel-weighted in log scored seconds and star; n_eff counts a song once.', 'Brief: same-shape windows cut from fit songs at several positions.'),
    ('Song p-value', 'p = (1 + sum of null weights at least as large) / (1 + sum of null weights); kernel weights exp(-z^2/2) in star (h 0.25) and log scored seconds (h 0.2, the query\'s log seconds rounded to 0.005).', 'The query counts as one null song at full weight; rounding lets the cases of one chart share one weight vector.'),
    ('Seam conflicts under continuation', 'A scored object that a given hold covers reverts to its source state for in-place families, otherwise is dropped; counted per case.', 'The given part is fixed by the condition; a generator could not write under a given hold.'),
    ('Placement units for injections', 'Four canonical bars inside one musical segment (the previous placeholder layout), moved to harness/arrays.py.', 'The harness may use the grid it knows; the evaluator does not read it.'),
    ('Descriptions for the corpus-plausible dose', 'Keyed by star band only; density and F4\'s density ratio in heads per second.', 'The evaluator\'s key is the star; the family is in seconds.'),
    ('F2 mode chart', 'The placement unit whose head rate is closest to the modal head rate of the fit density at the key, tiled over every unit where the chart stays legal.', 'Per-chart surrogate of the most common pattern at the key under the new model.'),
    ('Time-stretch transforms', 'Applied to every chart whose red lines stay plausible; the 80/160 fold restriction is dropped; judged by E-C5.', 'The fold restriction protected a beat-clock evaluator.'),
    ('E-C5 statistic', 'Per head, log of (stretched rate x s / source rate), heads matched in time order, pooled over every compared chart; the median event must lie within 1% of 1.', 'The claim names the median event; per-chart medians are reported post hoc.'),
    ('Skeleton cases', 'Every identity, transform and head-preserving injection is also run under the skeleton condition; evaluate skips the family; the cells count the skips.', 'Makes the skip visible per cell instead of inferred.'),
    ('How are calibration and generated songs scored?', 'As a fit song of the fold of their song group is: the density and the rank reference without that fold.', 'The song null is cross-fitted fit songs; the songs it judges must be scored by the same procedure to be exchangeable with it.'),
    ('How does another family plug in?', '--family module:attribute names any field.EventFamily (or a factory); one family per run; per-family results only.', 'Combining several families (the minimum p-value of P1) needs p-values of fit songs per family and is not built (unfinished).'),
    ('Smoke runs', 'smoke-4 (killed after its fit pass) and smoke-5 (300 fit and 30 calibration song groups) checked mechanics and timing; their calibration numbers are not results.', 'Only the bandwidth grid (fit-split likelihood) and the fold scoring bug changed after them.'),
    ('Unscored songs', 'A song with no ranked head, a zero scored time or no fit events at its key has no p-value and is counted as unscored.', 'No borrowing across keys.'),
]


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
    out['claims'] = claims(out)
    out['diagnoses'] = DIAGNOSES
    out['bug_fixes'] = [dict(found_in=f, bug=b, fix=x) for f, b, x in BUG_FIXES]
    out['observations'] = OBSERVATIONS
    out['failed_paths'] = FAILED_PATHS
    out['ledger'] = [dict(question=q, choice=ch, why=w) for q, ch, w in LEDGER]
    previous = getattr(args, 'previous', None)
    if previous and Path(previous).is_file():
        out['previous_run'] = previous_summary(json.loads(Path(previous).read_text()), out, previous)
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
    for label, sel in (('original whole (E-C0)', np.array([i for (ch, sc), i in original.items() if sc == 'whole'])),
                       ('original continuation (post hoc)',
                        np.array([i for (ch, sc), i in original.items() if sc == 'continuation'])),
                       ('every timing case (post hoc)', np.flatnonzero((c.condition == 'timing') & (c.status == 'ok')))):
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
    for view, (lo_name, hi_name) in (('mean', ('mean_low', 'mean_high')), ('worst16 (post hoc)', ('worst_low', 'worst_high'))):
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
    cell['full_identical'] = int(np.sum(c.same_full[ok]))
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
                                judged_by='E-C5 covariance' if t.stretch else 'E-C4 full output hash',
                                cases=int(len(idx)), ok=int(np.sum(st == 'ok')), skipped=int(np.sum(st == 'skipped')),
                                not_applicable=int(np.sum(st == 'n/a')), discarded=int(np.sum(st == 'discarded')),
                                errors=int(np.sum(st == 'error')), compared=int(len(compared)),
                                mismatches=int(len(bad)),
                                family_mismatches=int(np.sum(~c.same_family[compared])),
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


def claims(a: dict) -> dict[str, Any]:
    out = {}
    cov = a['coverage']['original whole (E-C0)']
    ok = cov['events'] > 0 and cov['events_scored'] == cov['events'] and cov['heads_ranked'] == cov['heads'] \
        and cov['not_ok'] == 0
    out['E-C0'] = dict(verdict=_verdict(ok if cov['cases'] else None),
                       detail=f"{cov['events_scored']} of {cov['events']} heads and releases scored "
                              f"({cov['heads_ranked']} of {cov['heads']} heads ranked) in {cov['cases']} calibration "
                              f"charts; {cov['not_ok']} charts without an evaluation; {cov['cases_short_of_full']} short of 100%")
    fl = a['floor']['whole']['mean']
    pooled, outside, ok, any_cell = [], [], True, False
    for tail in ('low', 'high'):
        blk = fl[tail]
        pooled.append(f"{tail} pooled {100 * blk['pooled']['rate']:.2f}% (n={blk['pooled']['n']})"
                      if blk['pooled'].get('n') else f'{tail} pooled {UNTESTED}')
        cells = [('pooled', blk['pooled'])] + [(f"star {b['star_band']}", b) for b in blk['by_star_band'] if b['n'] >= 100] \
            + [(f"duration quintile {q['quintile']}", q) for q in blk['by_duration_quintile']]
        for label, cell in cells:
            if not cell.get('n'):
                continue
            any_cell = True
            if not cell['within_two_se']:
                ok = False
                outside.append(f"{tail} {label}: {100 * cell['rate']:.2f}% (n={cell['n']}, 2SE {100 * cell['two_se']:.2f} pp)")
    out['E-C1'] = dict(verdict=_verdict(ok if any_cell else None),
                       detail='; '.join(pooled) + '; outside 2SE: ' + ('; '.join(outside) or 'none'))
    d9 = {x['dose']: x for x in a['detection'] if x['family'] == 'D9' and x['scope'] == 'whole'}
    vals, ok = [], True
    for dose in ('2', '0.5'):
        x = d9.get(dose)
        if not x or not (x.get('low') or {}).get('n'):
            ok = None
            vals.append(f'x{dose}: {UNTESTED}')
            continue
        vals.append(f"x{dose}: {100 * x['low']['rate']:.1f}% (n={x['low']['n']})")
        ok = (ok and x['low']['rate'] >= 0.8) if ok is not None else None
    out['E-C2'] = dict(verdict=_verdict(ok), detail='; '.join(vals),
                       dose_response={x['dose']: (x.get('low') or {}).get('rate') for x in d9.values()})
    fails, total, full_diff = [], 0, []
    for x in a['detection']:
        if x['family'] not in E_C3_FAMILIES or 'compared' not in x:
            continue
        total += x['compared']
        if x['compared'] and x['family_identical'] != x['compared']:
            fails.append(f"{x['family']} {x['dose']} {x['scope']}: {x['compared'] - x['family_identical']} of {x['compared']}")
        if x['compared'] and x['full_identical'] != x['compared']:
            full_diff.append(f"{x['family']} {x['dose']} {x['scope']}: {x['compared'] - x['full_identical']} of {x['compared']}")
    without_d7 = [f for f in fails if not f.startswith('D7 ')]
    out['E-C3'] = dict(verdict=_verdict(not fails if total else None), families=list(E_C3_FAMILIES),
                       compared=total, failures=fails,
                       post_hoc_without_d7=_verdict(not without_d7 if total else None),
                       post_hoc_full_hash_differences=full_diff)
    bad, tested = [], 0
    for m in a['must_not_flag']:
        if m['judged_by'].startswith('E-C4') and m['condition'] == 'timing':
            tested += m['compared']
            if m['mismatches'] or m['errors']:
                bad.append(f"{m['transform']}/{m['scope']}: {m['mismatches']} of {m['compared']} differ, {m['errors']} errors")
    out['E-C4'] = dict(verdict=_verdict(not bad if tested else None), compared=tested, failures=bad)
    cv = a['covariance']
    ok = all(x.get('charts') and abs(x['pooled_median_ratio'] - 1) <= 0.01 for x in cv) if cv else None
    out['E-C5'] = dict(verdict=_verdict(ok if cv and all(x.get('charts') for x in cv) else None),
                       detail='; '.join(f"{x['transform']}: median ratio {x['pooled_median_ratio']:.4f} over "
                                        f"{x['heads']} heads in {x['charts']} charts" if x.get('charts') else
                                        f"{x['transform']}: {UNTESTED}" for x in cv))
    ill, rates, untested = [], {}, []
    for f in FAMILIES:
        smallest = [dose_label(d) for d in f.doses] if f.name == 'D10' else [dose_label(f.doses[0])]
        for d in smallest:
            for sc in f.scopes:
                x = next(x for x in a['detection'] if x['family'] == f.name and x['dose'] == d and x['scope'] == sc)
                r = x.get('discard_rate')
                rates[f'{f.name} {d} {sc}'] = r
                if r is None:
                    untested.append(f'{f.name} {d} {sc}')
                elif r > 0.2:
                    ill.append(f'{f.name} {d} {sc}: {100 * r:.1f}%')
    out['E-C6'] = dict(verdict='pass (reported)' if not untested else 'fail: unreported cells',
                       smallest_dose_discard_rates=rates, ill_posed=ill, unreported=untested)
    return out


def _verdict(ok) -> str:
    return UNTESTED if ok is None else ('pass' if ok else 'fail')


def previous_summary(prev: dict, now: dict, path: str) -> dict[str, Any]:
    """Run-1 (bar passages) beside this run: its claims, and discard rates that changed."""
    def cells(a):
        return {(x['family'], x['dose'], x['scope']): x for x in a['detection']}
    p, n = cells(prev), cells(now)
    changed = []
    for key, x in n.items():
        q = p.get(key)
        if q is None or 'cell' in x:
            continue
        if q.get('discard_rate') is not None and x.get('discard_rate') is not None and \
                abs(q['discard_rate'] - x['discard_rate']) > 0.005:
            changed.append(dict(family=key[0], dose=key[1], scope=key[2], run1=q['discard_rate'], now=x['discard_rate']))
    return dict(path=path, receipt_outputs=prev.get('receipt', {}).get('outputs_sha256'),
                claims={k: dict(verdict=v.get('verdict'), detail=v.get('detail'), failures=v.get('failures'))
                        for k, v in prev['claims'].items()},
                discard_rate_changes=changed)


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
            f"field over the scored spans (weights in seconds); worst {es['scan_s']:g} s sub-span reported post hoc; "
            f"song null kernel-weighted in log scored seconds (h {es['null_h_log_seconds']}) and star (h {es['null_h_star']}). ")
    L += ['# Calibration harness v0, run-2: trivial event-field evaluator (head rate)', '',
          head + 'Pre-registered claims E-C0 to E-C6 are judged as written; every other number is post hoc.', '',
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
    L += ['## Pre-registered claims (run-2)', '', '| Claim | Verdict | Detail |', '| --- | --- | --- |']
    cl = a['claims']
    L.append(f"| E-C0 coverage | {cl['E-C0']['verdict']} | {cl['E-C0']['detail']} |")
    L.append(f"| E-C1 floor | {cl['E-C1']['verdict']} | {cl['E-C1']['detail']} |")
    L.append(f"| E-C2 the key | {cl['E-C2']['verdict']} | D9 low-tail detection, whole chart, timing: {cl['E-C2']['detail']} |")
    L.append(f"| E-C3 specificity | {cl['E-C3']['verdict']} | {', '.join(cl['E-C3']['families'])}: {cl['E-C3']['compared']} injected cases compared on the family hash; "
             f"differing: {'; '.join(cl['E-C3']['failures']) or 'none'}. Post hoc without D7: {cl['E-C3']['post_hoc_without_d7']} |")
    L.append(f"| E-C4 must-not-flag | {cl['E-C4']['verdict']} | {cl['E-C4']['compared']} transformed cases compared on the full output hash; "
             f"failures: {'; '.join(cl['E-C4']['failures']) or 'none'} |")
    L.append(f"| E-C5 stretch covariance | {cl['E-C5']['verdict']} | {cl['E-C5']['detail']} |")
    L.append(f"| E-C6 legality | {cl['E-C6']['verdict']} | ill-posed at the smallest dose (>20% discarded): "
             f"{'; '.join(cl['E-C6']['ill_posed']) or 'none'} |")
    L += ['']
    for sc in SCOPES:
        fl = a['floor'][sc]
        for view in ('mean', 'worst16 (post hoc)'):
            what = 'time-average' if view == 'mean' else 'worst 16 s sub-span'
            tag = '(E-C1)' if (sc, view) == ('whole', 'mean') else '(post hoc)'
            lo, hi = fl[view]['low'], fl[view]['high']
            L += [f'## Floor: false-alarm rate at nominal 5%, {sc} scope, {what} {tag}', '',
                  f"Songs {fl['songs']}, unscored {fl['unscored']}. Group-bootstrap 95% intervals; "
                  'within 2SE uses the binomial SE at 5%.', '',
                  '| Stratum | Low tail | within 2SE | High tail | within 2SE |', '| --- | --- | --- | --- | --- |']
            L.append(f"| pooled | {_fmt_rate(lo['pooled'])} | {lo['pooled'].get('within_two_se', UNTESTED)} | {_fmt_rate(hi['pooled'])} | {hi['pooled'].get('within_two_se', UNTESTED)} |")
            L.append(f"| 2-7 stars (post hoc) | {_fmt_rate(lo['pooled_2_to_7'])} | {lo['pooled_2_to_7'].get('within_two_se', UNTESTED)} | {_fmt_rate(hi['pooled_2_to_7'])} | {hi['pooled_2_to_7'].get('within_two_se', UNTESTED)} |")
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
    L += ['## D9 dose-response (whole chart, timing condition)', '',
          '| Factor | Low tail | High tail | Worst 16 s low (post hoc) | Source low | Discarded | Plausible share |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for x in a['detection']:
        if x['family'] == 'D9' and x['scope'] == 'whole':
            L.append(f"| {x['dose']} | {_fmt_rate(x.get('low'))} | {_fmt_rate(x.get('high'))} | {_fmt_rate(x.get('worst_low'))} | "
                     f"{_fmt_rate(x.get('source_low'))} | {x.get('discarded', UNTESTED)} | {_pct(x.get('plausible_inside_share'))} |")
    L += ['', '## Detection by family, dose and scope (timing condition)', '',
          'Detection = share of injected songs with song p < 0.05 (time-average statistic) per tail, group-bootstrap 95% interval. '
          'Identical = injected family hash equal to the source\'s (full hash in brackets). Realized = mean share of requested edits made. '
          'Seam = cases with a repaired seam conflict. Plausible = share of injected charts whose targeted description is inside '
          'the central 90% of fit charts at their star band (whole scope; F4 continuation).', '',
          '| Family | Dose | Scope | Low tail | High tail | Source low | Discarded | No-op | Identical | Realized | Seam | Plausible |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in a['detection']:
        if 'cell' in x:
            L.append(f"| {x['family']} | {x['dose']} | {x['scope']} | {x['cell']} | | | | | | | | |")
            continue
        ident = f"{x['family_identical']}/{x['compared']} [{x['full_identical']}]" if x.get('compared') else UNTESTED
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
    L += ['', '## Must-not-flag (E-C4) and time-stretch (E-C5)', '',
          '| Transform | Scope | Condition | Judged by | OK | Skipped | Compared | Differ (full) | Differ (family) | n/a | Discarded | Errors |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in a['must_not_flag']:
        L.append(f"| {x['title']} | {x['scope']} | {x['condition']} | {x['judged_by']} | {x['ok']} | {x['skipped']} | {x['compared']} | "
                 f"{x['mismatches'] if not x['judged_by'].startswith('E-C5') else 'n/a (expected)'} | "
                 f"{x['family_mismatches'] if not x['judged_by'].startswith('E-C5') else 'n/a'} | {x['not_applicable']} | {x['discarded']} | {x['errors']} |")
    for x in a['must_not_flag']:
        if x['mismatched_charts'] and not x['judged_by'].startswith('E-C5'):
            L += ['', f"Differing charts, {x['transform']} / {x['scope']} / {x['condition']} (first 20):", '']
            L += [f'- `{pth}`' for pth in x['mismatched_charts']]
        if x['not_applicable_reasons'] and x['condition'] == 'timing':
            L.append(f"- {x['transform']} / {x['scope']} not applicable: {x['not_applicable_reasons']}")
        if x['error_samples']:
            L.append(f"- {x['transform']} / {x['scope']} / {x['condition']} errors: {x['error_samples']}")
    L += ['', 'Time-stretch covariance (whole scope, timing): ratio = stretched per-head rate x s / source per-head rate, heads in time order.', '',
          '| Transform | Charts | Heads | Pooled median ratio | p5 | p95 | Heads within 1% | Charts with median within 1% (post hoc) |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in a['covariance']:
        if not x.get('charts'):
            L.append(f"| {x['transform']} | 0 | | {UNTESTED} | | | | |")
            continue
        L.append(f"| {x['transform']} | {x['charts']} | {x['heads']} | {x['pooled_median_ratio']:.4f} | {x['pooled_p5_ratio']:.4f} | "
                 f"{x['pooled_p95_ratio']:.4f} | {_pct(x['pooled_within_1pct'])} | {_pct(x['charts_median_within_1pct'])} |")
    cov = a['coverage']
    L += ['', '## Coverage (E-C0)', '', '| Cases | Charts | Events scored | Heads ranked | Releases scored | Cases short of 100% |',
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
    L += ['## Run-1 failures diagnosed', '', '| Failure | Verdict | Evidence | Fix |', '| --- | --- | --- | --- |']
    L += [f"| {d['failure']} | {d['verdict']} | {d['evidence']} | {d['fix']} |" for d in a['diagnoses']]
    L += ['', '## Bugs found and fixed', '', '| Found in | Bug | Fix |', '| --- | --- | --- |']
    L += [f"| {x['found_in']} | {x['bug']} | {x['fix']} |" for x in a['bug_fixes']]
    pr = a.get('previous_run')
    if pr:
        L += ['', '## Run-1 (bar passages) for reference', '', f"Report `{pr['path']}`.", '', '| Claim | Run-1 verdict | Run-1 detail |', '| --- | --- | --- |']
        for k, v in pr['claims'].items():
            L.append(f"| {k} | {v['verdict']} | {v.get('detail') or '; '.join(v.get('failures') or []) or ''} |")
        L += ['', 'Discard rates that changed (run-1 -> run-2):', '', '| Family | Dose | Scope | Run-1 | Run-2 |', '| --- | --- | --- | --- | --- |']
        L += [f"| {x['family']} | {x['dose']} | {x['scope']} | {_pct(x['run1'])} | {_pct(x['now'])} |" for x in pr['discard_rate_changes']]
    L += ['', '## Observations', ''] + [f'- {x}' for x in a['observations']]
    L += ['', '## Failed paths', ''] + [f'- {x}' for x in a['failed_paths']]
    L += ['', '## Question ledger', '', '| Question | Choice | Why |', '| --- | --- | --- |']
    L += [f"| {x['question']} | {x['choice']} | {x['why']} |" for x in a['ledger']]
    L += ['']
    return '\n'.join(L)
