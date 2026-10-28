"""Paired margins and conservative both-seed screening verdicts."""
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .common import OUT, digest, dump, interval, paired, prereg_check, utc


@lru_cache(maxsize=48)
def read(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def stat(data, *path):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data if isinstance(data, dict) and 'recordings' in data else None


def number(value):
    return 'unavailable' if value is None else f'{value:.4f}'


def confidence(value):
    return 'unavailable' if value is None else '[' + ', '.join(number(v) for v in value) + ']'


def average_stats(stats):
    stats = [s for s in stats if s is not None]
    if not stats:
        return None
    keys = set.intersection(*(set(s['recordings']) for s in stats))
    values = {k: float(np.mean([s['recordings'][k] for s in stats])) for k in keys}
    return dict(**interval(list(values.values())), recordings=values)


def merge_stats(stats):
    """Join read cohorts while keeping one contribution per recording."""
    grouped = defaultdict(list)
    for item in stats:
        if item:
            for rec, value in item['recordings'].items():
                grouped[rec].append(value)
    if not grouped:
        return None
    values = {rec: float(np.mean(v)) for rec, v in grouped.items()}
    return dict(**interval(list(values.values())), recordings=values)


def scaled_stats(item, scale):
    if item is None:
        return None
    values = {rec: value * scale for rec, value in item['recordings'].items()}
    return dict(**interval(list(values.values())), recordings=values)


def difference_stats(left, right):
    if left is None or right is None:
        return None
    values = {r: left['recordings'][r] - right['recordings'][r]
              for r in set(left['recordings']) & set(right['recordings'])}
    return dict(**interval(list(values.values())), recordings=values)


def independent_difference(left, right):
    a, b = np.array(list(left['recordings'].values())), np.array(list(right['recordings'].values()))
    if not len(a) or not len(b):
        return dict(n=0, mean=None, ci95=None, halfwidth=None)
    rng = np.random.default_rng(20261028)
    samples = [a[rng.integers(len(a), size=len(a))].mean() - b[rng.integers(len(b), size=len(b))].mean()
               for _ in range(1000)]
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(n=min(len(a), len(b)), n_left=len(a), n_right=len(b), mean=float(a.mean() - b.mean()),
                ci95=[float(lo), float(hi)], halfwidth=float((hi - lo) / 2), paired=False)


def compare(a, b, *, threshold, direction='above', refits=None, episode_spreads=None,
            require_refits=False, paired_recordings=True):
    """Compare paired recording means; unavailable seeds/noise never pass."""
    results = {}
    base = b if b is not None else a
    spread = None
    if all(base.get(s) and base[s]['mean'] is not None for s in (0, 1)):
        common = sorted(set(base[0]['recordings']) & set(base[1]['recordings']))
        if common:
            spread = abs(float(np.mean([base[0]['recordings'][k] for k in common])) -
                         float(np.mean([base[1]['recordings'][k] for k in common])))
    for seed in (0, 1):
        left, right = a.get(seed), b.get(seed) if b is not None else None
        if left is None or b is not None and right is None:
            results[str(seed)] = dict(status='unassessed', reason='Missing seed/statistic')
            continue
        diff = paired(left['recordings'], right['recordings']) if right is not None else interval(list(left['recordings'].values()))
        if right is not None and not paired_recordings:
            diff = independent_difference(left, right)
        if right is not None and '_curve_points' in left and '_curve_points' in right:
            diff = curve_contrast(left, right, diff)
        components = dict(seed_spread=spread, bootstrap_halfwidth=diff['halfwidth'])
        if refits is not None:
            vals = refits.get(seed, [])
            components['twice_refit_sd'] = float(2 * np.std(vals, ddof=1)) if len(vals) == 5 else None
        elif require_refits:
            components['twice_refit_sd'] = None
        if episode_spreads is not None:
            components['episode_spread'] = episode_spreads.get(seed)
        available = all(x is not None and np.isfinite(x) for x in components.values())
        margin = max(components.values()) if available else None
        enough = diff['n'] >= 40
        passes = bool(available and enough and diff['mean'] is not None and (
            diff['mean'] >= threshold + margin if direction == 'above' else diff['mean'] <= threshold - margin))
        results[str(seed)] = dict(status='holds' if passes else 'does_not_hold' if available and enough else 'inconclusive',
            left={k: left[k] for k in ('n', 'mean', 'ci95')},
            right={k: right[k] for k in ('n', 'mean', 'ci95')} if right else None,
            contrast=diff, margin=margin, margin_components=components, power=enough)
    holds = all(results[str(s)]['status'] == 'holds' for s in (0, 1))
    return dict(holds_both_seeds=holds, threshold=threshold, direction=direction, seeds=results)


def probe_stats(name, path, key, suffix=''):
    return {s: stat(read(f'{name}-s{s}{suffix}.json'), *path, key) for s in (0, 1)}


def bootstrap_values(name, path, a, b=None):
    result = {}
    for s in (0, 1):
        data = read(f'{name}-s{s}-bootstrap.json')
        vals = []
        for i in range(5):
            aa = stat(data, *path, a + f'-boot{i}')
            bb = stat(data, *path, b + f'-boot{i}') if b else None
            if aa and (b is None or bb):
                v = paired(aa['recordings'], bb['recordings'])['mean'] if bb else aa['mean']
                if v is not None:
                    vals.append(v)
        result[s] = vals
    return result


def verdict(predictions, retire):
    if any(v['holds_both_seeds'] for v in retire.values()):
        return 'retire under mapper-agreement screen'
    if predictions and all(v['holds_both_seeds'] for v in predictions.values()):
        return 'pass under mapper-agreement screen'
    return 'inconclusive'


def probe_verdicts():
    output = {}
    a = probe_stats('P1', ['panels:offline'], 'mlp-F')
    m = probe_stats('P1', ['panels:offline'], 'mlp-M')
    fp = probe_stats('P1', ['panels:offline'], 'mlp-F+Pos')
    pos = probe_stats('P1', ['panels:offline'], 'mlp-Pos')
    ref = lambda x, y=None: bootstrap_values('P1', ['panels:offline'], x, y)
    pred = dict(F_at_least_080=compare(a, None, threshold=.8, refits=ref('mlp-F')),
                F_over_M_003=compare(a, m, threshold=.03, refits=ref('mlp-F', 'mlp-M')),
                FPos_over_Pos_005=compare(fp, pos, threshold=.05, refits=ref('mlp-F+Pos', 'mlp-Pos')))
    retire = dict(F_below_070=compare(a, None, threshold=.7, direction='below', refits=ref('mlp-F')),
                  F_no_better_M=compare(a, m, threshold=0, direction='below', refits=ref('mlp-F', 'mlp-M')))
    output['P1'] = dict(verdict=verdict(pred, retire), predictions=pred, retire=retire,
        shifted_null=probe_stats('P1', ['panels:offline'], 'mlp-F', '-shift'),
        shifted_null_expected_upper=.60)
    path = ['panels:chart:real']
    a = probe_stats('P3', path, 'mlp-T24-real')
    m = probe_stats('P3', path, 'mlp-Mel24-real')
    fb = probe_stats('P3', path, 'mlp-F_b-real')
    decoded = probe_stats('P3', ['panels:decoded:real'], 'mlp-T24-real')
    ref = lambda x, y=None: bootstrap_values('P3', path, x, y)
    grid_refits = {}
    for seed in (0, 1):
        data = read(f'P3-s{seed}-bootstrap.json')
        v = []
        for b in range(5):
            aa = stat(data, 'panels:decoded:real', f'mlp-T24-real-boot{b}')
            bb = stat(data, 'panels:chart:real', f'mlp-T24-real-boot{b}')
            if aa and bb:
                v.append(paired(aa['recordings'], bb['recordings'])['mean'])
        grid_refits[seed] = v
    pred = dict(T24_at_least_080=compare(a, None, threshold=.8, refits=ref('mlp-T24-real')),
                T24_over_Mel_003=compare(a, m, threshold=.03, refits=ref('mlp-T24-real', 'mlp-Mel24-real')),
                Fb_at_least_075=compare(fb, None, threshold=.75, refits=ref('mlp-F_b-real')),
                decoded_within_005=compare(decoded, a, threshold=-.05, refits=grid_refits))
    retire = dict(T24_below_070=compare(a, None, threshold=.7, direction='below', refits=ref('mlp-T24-real')),
                  T24_no_more_Mel_001=compare(a, m, threshold=.01, direction='below', refits=ref('mlp-T24-real', 'mlp-Mel24-real')))
    output['P3'] = dict(verdict=verdict(pred, retire), predictions=pred, retire=retire)
    f = probe_stats('P5a', ['summary'], 'all:exact:auc:F')
    m = probe_stats('P5a', ['summary'], 'all:exact:auc:M')
    precision = probe_stats('P5a', ['summary'], 'all:p5:F')
    periodic = probe_stats('P5a', ['summary'], 'all:p5:periodic')
    pred = dict(F_at_least_070=compare(f, None, threshold=.7),
                F_over_Mel_003=compare(f, m, threshold=.03),
                P5_over_periodic=compare(precision, periodic, threshold=0))
    retire = dict(F_no_better_Mel=compare(f, m, threshold=0, direction='below'))
    bothlow = [compare(x, None, threshold=.6, direction='below') for x in (f, m)]
    output['P5a'] = dict(verdict=verdict(pred, retire), predictions=pred, retire=retire,
                         untrained_similarity_retire=all(x['holds_both_seeds'] for x in bothlow), both_below_060=bothlow)
    ti = probe_stats('P5b', ['summary'], 'failing:tier_i')
    br = probe_stats('P5b', ['B_summary'], 'B_repeat')
    tracked = probe_stats('P5b', ['B_summary'], 'B_tracked_repeat')
    pred = dict(T_recurrence_050=compare(ti, None, threshold=.5),
                B_repeat_at_most_030=compare(br, None, threshold=.3, direction='below'))
    retire = dict(T_recurrence_below_020=compare(ti, None, threshold=.2, direction='below'))
    priority = compare(probe_stats('P5b', ['B_summary'], 'B_tracked_any'), None, threshold=.3)
    output['P5b'] = dict(verdict=verdict(pred, retire), predictions=pred, retire=retire,
                         B_tracked_repeat=tracked,
                         B_worth_building=priority, priority=dict(B_tracked_repeat_at_least_030=priority),
                         priority_statement='B propagation by repeats is worth building' if priority['holds_both_seeds'] else 'B priority threshold not established',
                         posthoc_nontrivial=probe_stats('P5b', ['B_summary'], 'B_tracked_nontrivial'))
    f = probe_stats('P5c', ['summary'], 'feature')
    raw = probe_stats('P5c', ['summary'], 'raw')
    rand = probe_stats('P5c', ['summary'], 'random')
    pred = dict(average_over_raw_002=compare(f, raw, threshold=.02),
                average_over_random_002=compare(f, rand, threshold=.02))
    retire = dict(average_within_random_0005=compare(f, rand, threshold=.005, direction='below'))
    output['P5c'] = dict(verdict=verdict(pred, retire), predictions=pred, retire=retire)
    c = probe_stats('P2', ['summary'], 'C:null0:refit0')
    f = probe_stats('P2', ['summary'], 'CF:null0:refit0')
    m = probe_stats('P2', ['summary'], 'CM:null0:refit0')
    def refit_ridge(left, right):
        vals = {}
        for seed in (0, 1):
            data = read(f'P2-s{seed}.json'); v = []
            for b in range(1, 6):
                aa = stat(data, 'summary', f'{left}:null0:refit{b}')
                bb = stat(data, 'summary', f'{right}:null0:refit{b}')
                if aa and bb:
                    v.append(paired(aa['recordings'], bb['recordings'])['mean'])
            vals[seed] = v
        return vals
    pred = dict(feature_gain_010=compare(f, c, threshold=.1, refits=refit_ridge('CF', 'C')),
                feature_over_Mel_003=compare(f, m, threshold=.03, refits=refit_ridge('CF', 'CM')))
    retirement = [compare(v, c, threshold=.05, direction='below', refits=refit_ridge(name, 'C'))
                  for v, name in ((f, 'CF'), (m, 'CM'))]
    output['P2'] = dict(verdict='retire under mapper-agreement screen' if all(v['holds_both_seeds'] for v in retirement)
                       else verdict(pred, {}), predictions=pred,
                       retire=dict(feature_gain_below_005=retirement[0], Mel_gain_below_005=retirement[1]),
                       retirement_requires='both audio gains below threshold on both seeds')
    for experiment, tags in [('P4a', ['clap', 'finish']),
                             ('P4b', [f'{part}:stratum{st}' for part in ('kick', 'snare', 'hat') for st in (0, 1)])]:
        per_target = {}
        for tag in tags:
            cohorts = ['panels', 'keysound_extension'] if experiment == 'P4b' else ['panels']
            def layer_stat(data, key):
                return merge_stats([stat(data, tag, 'summary', c, key) for c in cohorts])
            def layer_seeds(key):
                return {s: layer_stat(read(f'{experiment}-s{s}.json'), key) for s in (0, 1)}
            base = layer_seeds('Pos:null0')
            f = layer_seeds('Pos+F:null0')
            m = layer_seeds('Pos+M:null0')
            def refit_layers(left, right):
                vals = {}
                for seed in (0, 1):
                    data = read(f'{experiment}-s{seed}.json'); v = []
                    for b in range(5):
                        aa = layer_stat(data, f'{left}:null0:boot{b}')
                        bb = layer_stat(data, f'{right}:null0:boot{b}')
                        if aa and bb:
                            v.append(paired(aa['recordings'], bb['recordings'])['mean'])
                    vals[seed] = v
                return vals
            pred = dict(feature_gain_005=compare(f, base, threshold=.05, refits=refit_layers('Pos+F', 'Pos'))) if not tag.endswith('stratum1') else {}
            if experiment == 'P4a':
                pred['feature_over_Mel_002'] = compare(f, m, threshold=.02, refits=refit_layers('Pos+F', 'Pos+M'))
            retirement = [compare(v, base, threshold=.02, direction='below', refits=refit_layers(name, 'Pos'))
                          for v, name in ((f, 'Pos+F'), (m, 'Pos+M'))]
            per_target[tag] = dict(read_cohorts=cohorts, predictions=pred, retire=dict(feature_gain_below_002=retirement[0], Mel_gain_below_002=retirement[1]),
                verdict='retire under mapper-agreement screen' if all(v['holds_both_seeds'] for v in retirement) else verdict(pred, {}))
        output[experiment] = dict(verdict='characterises; see per-target power and margins', targets=per_target,
            predictions={f'{tag}:{k}': v for tag, row in per_target.items() for k, v in row['predictions'].items()},
            retire={f'{tag}:{k}': v for tag, row in per_target.items() for k, v in row['retire'].items()})
        if experiment == 'P4b':
            strata_checks = {}
            for part in ('kick', 'snare', 'hat'):
                def gain(seed, stratum, boot=None):
                    data = read(f'P4b-s{seed}.json')
                    suffix = '' if boot is None else f':boot{boot}'
                    def get(name):
                        return merge_stats([stat(data, f'{part}:stratum{stratum}', 'summary', c, name + ':null0' + suffix)
                                            for c in ('panels', 'keysound_extension')])
                    return difference_stats(get('Pos+F'), get('Pos'))
                refits = {}
                for seed in (0, 1):
                    values = []
                    for b in range(5):
                        full, kit = gain(seed, 1, b), gain(seed, 0, b)
                        if full and kit and full['mean'] is not None and kit['mean'] is not None:
                            values.append(full['mean'] - kit['mean'])
                    refits[seed] = values
                strata_checks[part] = compare({s: gain(s, 1) for s in (0, 1)},
                    {s: gain(s, 0) for s in (0, 1)}, threshold=0, direction='below',
                    refits=refits, paired_recordings=False)
            output[experiment]['full_below_kit'] = strata_checks
            output[experiment]['predictions'].update({part + ':full_below_kit': v for part, v in strata_checks.items()})
            output[experiment]['retirement_requires'] = 'Both F and Mel gains below .02 in both strata on both feature seeds; each stratum needs 40 recordings'
            output[experiment]['parts'] = {}
            for part, strata_check in strata_checks.items():
                kit, full = per_target[part + ':stratum0'], per_target[part + ':stratum1']
                retire_all = all(v['holds_both_seeds'] for item in (kit, full) for v in item['retire'].values())
                passed = strata_check['holds_both_seeds'] and all(v['holds_both_seeds'] for v in kit['predictions'].values())
                output[experiment]['parts'][part] = dict(verdict='retire under mapper-agreement screen' if retire_all else
                    'pass under mapper-agreement screen' if passed else 'inconclusive')
    return output


def episode_stats(arm, metric, kind='real', cell='all'):
    return {s: stat(read(f'P7-s{s}.json'), 'summary', f'{arm}:{kind}:{cell}', metric) for s in (0, 1)}


def episode_spread(arm, metric, other=None, kind='real', other_kind='real', other_scale=1.):
    result = {}
    for seed in (0, 1):
        data = read(f'P7-s{seed}.json')
        spreads = []
        strata = [['sim0', 'sim1']]
        if not kind.startswith('confirming-'):
            strata.append(['pos0', 'pos1', 'pos2'])
        for cells in strata:
            values = []
            for cell in cells:
                a = stat(data, 'summary', f'{arm}:{kind}:{cell}', metric)
                b = scaled_stats(stat(data, 'summary', f'{other}:{other_kind}:{cell}', metric), other_scale) if other else None
                if a and (not other or b):
                    value = paired(a['recordings'], b['recordings'])['mean'] if b else a['mean']
                    if value is not None:
                        values.append(value)
            if len(values) == len(cells):
                spreads.append(max(values) - min(values))
        learners = [stat(data, 'summary', f'learned{k}:{kind}:all', metric) for k in (0, 1)]
        if all(x and x['mean'] is not None for x in learners):
            spreads.append(abs(learners[0]['mean'] - learners[1]['mean']))
        result[seed] = max(spreads) if len(spreads) == len(strata) + 1 else None
    return result


def curve_read(seed, arm, away):
    data = read(f'P7-s{seed}.json')
    if not data:
        return None
    points = []
    for key, metrics in data.get('curves', {}).items():
        if key.split(':')[0] != arm:
            continue
        a, y = metrics.get('M4_away'), metrics.get('M3i')
        if a and y and a['mean'] is not None and y['mean'] is not None:
            points.append((a['mean'], float(key.split(':')[1]), y, a))
    # Follow the gain sweep in its registered order; sorting by away-share
    # could connect nonadjacent settings and hide a non-monotone curve.
    points.sort(key=lambda p: p[1])
    for left, right in zip(points[:-1], points[1:]):
        if min(left[0], right[0]) <= away <= max(left[0], right[0]) and right[0] != left[0]:
            weight = (away - left[0]) / (right[0] - left[0])
            keys = set(left[2]['recordings']) & set(right[2]['recordings'])
            values = {r: left[2]['recordings'][r] * (1 - weight) + right[2]['recordings'][r] * weight for r in keys}
            return dict(**interval(list(values.values())), recordings=values,
                        away=away, gains=[left[1], right[1]], interpolation_weight=weight,
                        method='First crossing between adjacent gains; matched-arm contrast bootstraps crossing and recovery jointly',
                        _curve_points=points)
    return None


def curve_contrast(left, right, fallback):
    """Bootstrap recording weights, including movement of the curve crossing."""
    all_points = left['_curve_points'] + right['_curve_points']
    recs = sorted(set().union(*(set(p[k]['recordings']) for p in all_points for k in (2, 3))))
    n = len(recs)
    if not n:
        return dict(fallback, halfwidth=None, ci95=None, bootstrap_crossings=0)
    def arrays(points):
        return [np.array([[p[k]['recordings'].get(r, np.nan) for r in recs] for p in points]) for k in (3, 2)]
    la, ra = arrays(left['_curve_points']), arrays(right['_curve_points'])
    def crossing(matrix, weights, target):
        values = []
        for x in matrix:
            valid = np.isfinite(x)
            denom = valid @ weights
            values.append(np.divide(np.nan_to_num(x) @ weights, denom,
                out=np.full(len(x), np.nan), where=denom > 0))
        xx, yy = values
        for i in range(len(xx) - 1):
            if np.isfinite([xx[i], xx[i + 1], yy[i], yy[i + 1]]).all() and min(xx[i:i + 2]) <= target <= max(xx[i:i + 2]) and xx[i] != xx[i + 1]:
                w = (target - xx[i]) / (xx[i + 1] - xx[i])
                return yy[i] * (1 - w) + yy[i + 1] * w
        return None
    rng = np.random.default_rng(20261028)
    values = []
    for _ in range(1000):
        weights = np.bincount(rng.integers(n, size=n), minlength=n)
        a, b = crossing(la, weights, left['away']), crossing(ra, weights, right['away'])
        if a is not None and b is not None:
            values.append(a - b)
    # A target that disappears in any bootstrap draw has unsupported crossing
    # uncertainty. Preserve the point estimate but prevent a verdict.
    if len(values) != 1000:
        return dict(fallback, halfwidth=None, ci95=None, bootstrap_crossings=len(values),
                    limitation='Matched away-share was not crossed in every recording resample')
    lo, hi = np.quantile(values, [.025, .975])
    point_left = crossing(la, np.ones(n), left['away'])
    point_right = crossing(ra, np.ones(n), right['away'])
    return dict(fallback, mean=float(point_left - point_right), ci95=[float(lo), float(hi)],
                halfwidth=float((hi - lo) / 2), bootstrap_crossings=1000)


def propagation_verdict():
    graph, local, random, bias = [episode_stats(a, 'M3i') for a in ('graph', 'local', 'random', 'bias')]
    pred = dict(graph_over_local_010=compare(graph, local, threshold=.1,
                    episode_spreads=episode_spread('graph', 'M3i', 'local')),
                graph_over_random=compare(graph, random, threshold=0,
                    episode_spreads=episode_spread('graph', 'M3i', 'random')),
                graph_over_bias=compare(graph, bias, threshold=0,
                    episode_spreads=episode_spread('graph', 'M3i', 'bias')),
                away_at_most_020=compare(episode_stats('graph', 'M4_away'), None, threshold=.2, direction='below',
                    episode_spreads=episode_spread('graph', 'M4_away')))
    for kind in ('confirming-fail', 'confirming-nofail'):
        pred[kind + '_M8_001'] = compare(episode_stats('graph', 'M8', kind), None, threshold=.01, direction='below',
                                        episode_spreads=episode_spread('graph', 'M8', kind=kind))
    retire = dict(away_over_035=compare(episode_stats('graph', 'M4_away'), None, threshold=.35,
                                      episode_spreads=episode_spread('graph', 'M4_away')))
    similarity_no_gain = compare(graph, local, threshold=0, direction='below',
                                  episode_spreads=episode_spread('graph', 'M3i', 'random'))
    for seed, row in similarity_no_gain['seeds'].items():
        noise = pred['graph_over_random']['seeds'][seed]
        row['margin'] = noise.get('margin')
        row['margin_components'] = noise.get('margin_components')
        if row.get('power') and row.get('margin') is not None and row['contrast']['mean'] is not None:
            row['status'] = 'holds' if row['contrast']['mean'] <= row['margin'] else 'does_not_hold'
        else:
            row['status'] = 'inconclusive'
    similarity_no_gain['holds_both_seeds'] = all(r['status'] == 'holds' for r in similarity_no_gain['seeds'].values())
    similarity_no_gain['rule'] = 'Gain over local is no larger than the random-comparison noise margin.'
    retire['no_gain_over_local'] = similarity_no_gain
    learned = {}
    for arm in ('learned0', 'learned1'):
        real = episode_stats(arm, 'M3i')
        nulls = {kind: episode_stats(arm, 'M3i', kind) for kind in ('shuffle', 'uninformative')}
        ratios = {}
        for kind, null in nulls.items():
            ratios[kind] = {}
            for seed in (0, 1):
                if real[seed] and null[seed] and real[seed]['mean'] is not None and real[seed]['mean'] > 0:
                    ratios[kind][str(seed)] = dict(ratio=null[seed]['mean'] / real[seed]['mean'],
                                                  numerator=null[seed]['mean'], denominator=real[seed]['mean'])
                else:
                    ratios[kind][str(seed)] = dict(ratio=None, reason='Zero or unavailable recovery denominator')
        comparisons = {}
        retiring = {}
        for away in (.10, .20):
            aa = {s: curve_read(s, arm, away) for s in (0, 1)}
            bb = {s: curve_read(s, 'graph', away) for s in (0, 1)}
            comparisons[str(away)] = compare(aa, bb, threshold=.05,
                                        episode_spreads=episode_spread(arm, 'M3i', 'graph'))
            retiring[str(away)] = compare(aa, bb, threshold=0, direction='below',
                                        episode_spreads=episode_spread(arm, 'M3i', 'graph'))
        null_checks = {kind: compare(null, {s: scaled_stats(real[s], .2) for s in (0, 1)},
            threshold=0, direction='below', episode_spreads=episode_spread(
                arm, 'M3i', arm, kind=kind, other_scale=.2)) for kind, null in nulls.items()}
        shuffle_check = compare(nulls['shuffle'], {s: scaled_stats(real[s], .5) for s in (0, 1)},
            threshold=0, episode_spreads=episode_spread(arm, 'M3i', arm, kind='shuffle', other_scale=.5))
        positive_denominator = all(real[s] and real[s]['mean'] is not None and real[s]['mean'] > 0 for s in (0, 1))
        null_good = positive_denominator and all(v['holds_both_seeds'] for v in null_checks.values())
        shuffle_bad = positive_denominator and shuffle_check['holds_both_seeds']
        learned_verdict = ('retire under mapper-agreement screen' if shuffle_bad or all(v['holds_both_seeds'] for v in retiring.values())
                           else 'pass under mapper-agreement screen' if null_good and all(v['holds_both_seeds'] for v in comparisons.values())
                           else 'inconclusive')
        learned[arm] = dict(verdict=learned_verdict, matched_away=comparisons, retire_matched_away=retiring,
            null_retention=ratios, null_retention_checks=null_checks, shuffle_retire_check=shuffle_check,
            null_retention_limit_met=null_good, shuffle_retire=shuffle_bad,
            over_random=compare(real, random, threshold=0, episode_spreads=episode_spread(arm, 'M3i', 'random')))
    invariant = {str(s): read(f'P7-invariants-s{s}.json') for s in (0, 1)}
    invariant_ok = all(invariant[str(s)] and invariant[str(s)].get('complete') and invariant[str(s)].get('breaches') == 0 for s in (0, 1))
    invariant_breach = any(v and v.get('breaches', 0) for v in invariant.values())
    if not invariant_ok:
        for item in learned.values():
            item['verdict'] = 'stopped: invariant breach' if invariant_breach else 'inconclusive: incomplete invariant evidence'
    packet_arms = []
    if invariant_ok:
        if pred['graph_over_random']['holds_both_seeds']:
            packet_arms.append('graph')
        packet_arms += [a for a, v in learned.items() if v['over_random']['holds_both_seeds']]
    positions = {arm: {str(s): {str(k): {metric: stat(read(f'P7-s{s}.json'), 'summary', f'{arm}:real:pos{k}', metric)
                               for metric in ('M1e', 'M3i')} for k in range(4)} for s in (0, 1)}
                 for arm in ('graph', 'learned0', 'learned1')}
    self_null = {arm: episode_stats(arm, 'changed', 'self') for arm in ('mark', 'local', 'graph', 'learned0', 'learned1', 'bias', 'random')}
    self_status = {arm: {str(s): 'unassessed' if not v[s] or not v[s]['n'] else
                   'pass' if max(v[s]['recordings'].values()) == 0 else 'fail'
                   for s in (0, 1)} for arm, v in self_null.items()}
    bias_equivalent = all(row.get('power') and row.get('margin') is not None and
        abs(row['contrast']['mean']) <= row['margin'] for row in pred['graph_over_bias']['seeds'].values())
    decision = verdict(pred, retire) if invariant_ok else 'stopped: invariant breach' if invariant_breach else 'inconclusive'
    return dict(verdict=decision, predictions=pred, retire=retire,
                learned=learned, invariants=invariant, packet_eligible_arms=packet_arms,
                mark_position_estimates=positions, self_label_null=self_null, self_label_status=self_status,
                bias_interpretation='similar = the rest of the song' if bias_equivalent else 'equivalence to per-song bias not established',
                M10_note='Position spreads are included in the preregistered episode margin. Flatness under that same margin is not an independent equivalence test; report the position estimates directly.')


def build():
    read.cache_clear()
    lock = prereg_check()
    verdicts = probe_verdicts()
    verdicts['P7'] = propagation_verdict()
    results = dict(generated_utc=utc(), prereg=lock, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        evidence='Mapper agreement only; no ear-validated measure and no adoption.',
        inputs=read('inputs.json'), cohorts={str(s): read(f'counts-s{s}.json') for s in (0, 1)},
        verdicts=verdicts, packet=read('packet-key/key.json'),
        deviations=read('deviations.json') or [], failed_paths=read('failed-paths.json') or [])
    results['design_checks'] = dict(simulator=read('P6.json'), session={str(s):
        {k: v for k, v in (read(f'P8-s{s}.json') or {}).items() if k in
         ('expectation_results', 'missing', 'attack_marks', 'design_only')} for s in (0, 1)})
    results['limitations'] = [
        'Charts screen mapper agreement; none of these measures has been validated against the ear.',
        'The probe-fit and probe-dev songs trained the frozen trunk; this is in-sample representation evidence.',
        'P3 T24 is a 64-dimensional unsupervised PCA projection. Its retirement does not prove that all 4000 raw token dimensions lack thirds.',
        'Only checkpoint 009001 is available. Seed spread and probe refits do not measure neighboring-checkpoint noise.',
        'P7 supplies chart timing and changes only cell posteriors; it cannot recover pulse, red lines, bars or between-cell spans.',
        'P7 keeps its preregistered feature-token graph and reports Mel as a separate arm. Retirement of that graph does not establish that Mel-based or trained similarity must fail.',
        'The P7 tuning objective excludes the marked neighborhood. Mark-beat-only therefore has zero recovery objective; the preregistered lower-gain tie rule can select zero gain for this control.',
        'The secondary U probe was not fitted. Unbeat-pair class expectations are unassessed.']
    shifted = verdicts['P1']['shifted_null']
    if all(shifted[s] and shifted[s]['mean'] is not None and shifted[s]['mean'] > .60 for s in (0, 1)):
        results['limitations'].append('The refitted +1/32-beat attack-label null exceeds its expected .60 AUC ceiling on both seeds; the proposed null does not separate the attack mechanism.')
    if results['design_checks']['simulator'] and results['design_checks']['simulator']['simulator_optimistic']:
        results['limitations'].append('P6 is out of at least one fixed band: label P7 simulator optimistic. Session evidence remains design-only.')
    # Keep the narrative report free of packet identities; the complete private
    # packet key remains in its own owner and is represented here by composition.
    if results['packet']:
        results['packet'] = dict(path=str(OUT / 'packet'), composition=results['packet']['composition'])
    expected = ['P1', 'P2', 'P3', 'P4a', 'P4b', 'P5a', 'P5b', 'P5c', 'P7', 'P8']
    results['unfinished'] = [f'{p}-s{s}' for p in expected for s in (0, 1) if read(f'{p}-s{s}.json') is None]
    results['unfinished'] += [f'{p}-s{s}-bootstrap' for p in ('P1', 'P3') for s in (0, 1)
                              if read(f'{p}-s{s}-bootstrap.json') is None]
    results['unfinished'] += [f'P1-s{s}-shift' for s in (0, 1) if read(f'P1-s{s}-shift.json') is None]
    if read('P6.json') is None:
        results['unfinished'].append('P6')
    results['unfinished'] += [f'P5d-s{s} (stretch)' for s in (0, 1) if read(f'P5d-s{s}.json') is None]
    results['unfinished'].append('P7 secondary U class and its P8 class-change expectation')
    for experiment in ('P4a', 'P4b'):
        for seed in (0, 1):
            for tag, item in (read(f'{experiment}-s{seed}.json') or {}).items():
                if item.get('unassessed'):
                    results['unfinished'].append(f'{experiment}-s{seed} {tag}: {item["unassessed"]}')
    results['experiment_files'] = {p.name: dict(sha256=digest(p), path=str(p)) for p in sorted(OUT.glob('P*.json'))}
    results['runtime_sha256'] = digest(OUT / 'runtime.json')
    results['posthoc_attack_null_overlap'] = {str(s): read(f'P1-null-overlap-s{s}.json') for s in (0, 1)}
    from .plots import write
    results['figures'] = write()
    events = [json.loads(line) for line in (OUT / 'job.jsonl').read_text().splitlines()]
    results['resources'] = dict(peak_recorded_process_rss_bytes=max(e.get('peak_rss_bytes', 0) for e in events),
        artifact_file_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        numerical_thread_limit=4, device='cpu', niceness=10)
    dump(OUT / 'results.json', results)
    display = dict(verdicts)
    for arm, item in verdicts['P7']['learned'].items():
        display['P7 ' + arm] = dict(verdict=item['verdict'],
            predictions={**{'matched_away_' + key: value for key, value in item['matched_away'].items()},
                         **{'null_retention_' + key: value for key, value in item['null_retention_checks'].items()}},
            retire={**{'matched_away_' + key: value for key, value in item['retire_matched_away'].items()},
                    'shuffle_retention_over_050': item['shuffle_retire_check']})
    lines = ['# N1 frozen-feature probes', '', results['evidence'], '',
             f"Preregistration: `{lock['sha256']}`, frozen {lock['utc']}.",
             f"Code: `{results['source_commit']}` on `codex/n1-probes`.", '',
             '## Screening verdicts', '', '| Experiment | Both-seed verdict |', '| --- | --- |']
    for name, v in display.items():
        lines.append(f"| {name} | {v['verdict']} |")
    lines += ['', 'The numbers, recording intervals, paired contrasts, margin components and power checks for every criterion are in results.json. Raw per-seed summaries and strata are in the hashed experiment files listed there. Missing evidence prevents a pass.', '',
              '![Feature and Mel comparisons](probe-comparison.png)' if str(OUT / 'probe-comparison.png') in results['figures'] else 'Probe comparison figure unavailable.', '',
              '## Interpretation limits', '', *['- ' + item for item in results['limitations']], '',
              '## Cohorts and counts', '',
              '| Seed | Cohort | Recordings | Audio | Units | Beats | Heads | Ternary | Binary-fine | Ladder heads | Nontrivial bars | Inconsistent grids | Rank fallback |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    count_keys = ('recordings', 'audio_inputs', 'units', 'beats', 'heads', 'ternary', 'binary_fine',
                  'laddered_heads', 'nontrivial_bars', 'timing_inconsistent', 'hardest_rank_fallback')
    for seed, cohorts in results['cohorts'].items():
        for name, counts in (cohorts or {}).items():
            lines.append('| ' + ' | '.join([seed, name] + [str(counts.get(k, 'unavailable')) for k in count_keys]) + ' |')
    lines += ['', '## Criterion estimates', '']
    for name, v in display.items():
        lines += [f'### {name}', '', '| Criterion | Seed | Estimate / contrast | 95% CI | Margin | n | Status |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for category in ('predictions', 'retire', 'priority'):
            for criterion, c in v.get(category, {}).items():
                for s, val in c['seeds'].items():
                    d = val.get('contrast', {})
                    mean = d.get('mean')
                    lines.append(f"| {category}: {criterion} | {s} | {number(mean)} | {confidence(d.get('ci95'))} | {number(val.get('margin'))} | {d.get('n', 0)} | {val['status']} |")
        lines.append('')
    lines += ['## P7 arm readings', '',
              'All entries are recording-balanced. The per-seed files also contain lag, tempo, mark-position, single/run and confirming-count strata.', '',
              '| Arm | Seed | Metric | Mean | 95% CI | Recordings |', '| --- | --- | --- | --- | --- | --- |']
    for arm in ('mark', 'local', 'graph', 'learned0', 'learned1', 'repeat', 'bias', 'random', 'mel', 'periodic', 'everywhere'):
        for seed in (0, 1):
            for metric in ('M1e', 'M2', 'M3i', 'M3ii', 'R_binary_flipped', 'M4_toward', 'M4_away', 'M4_unlabelled', 'M7', 'tier2_precision'):
                item = episode_stats(arm, metric)[seed]
                if item:
                    lines.append(f"| {arm} | {seed} | {metric} | {number(item['mean'])} | {confidence(item['ci95'])} | {item['n']} |")
            for kind in ('confirming-fail', 'confirming-nofail'):
                item = episode_stats(arm, 'M8', kind)[seed]
                if item:
                    lines.append(f"| {arm} | {seed} | M8 {kind} | {number(item['mean'])} | {confidence(item['ci95'])} | {item['n']} |")
    lines.append('')
    lines += ['## P7 invariants and packet', '', '```json', json.dumps(verdicts['P7']['invariants'], indent=2), '```', '',
              '![Propagation gain curves](P7-curves.png)' if str(OUT / 'P7-curves.png') in results['figures'] else 'Propagation curves unavailable.', '',
              'Self-label nulls (zero MAP changes expected):', '', '```json', json.dumps(verdicts['P7']['self_label_status'], indent=2), '```', '',
              json.dumps(results['packet'], indent=2) if results['packet'] else 'Packet not yet built.', '',
              '## Private design checks', '',
              '| Seed | Arm | Expectation | Passage / scope | Outcome |', '| --- | --- | --- | --- | --- |']
    for seed, data in results['design_checks']['session'].items():
        for arm, expectations in data.get('expectation_results', {}).items():
            for expectation, value in expectations.items():
                for scope, status in (value.items() if isinstance(value, dict) else [('all', value)]):
                    lines.append(f'| {seed} | {arm} | {expectation} | {scope} | {status} |')
    simulator = results['design_checks']['simulator']
    if simulator:
        bounds = simulator['fixed_bands']
        lines += ['', '| Simulator measure | Estimate | Fixed band | Outcome |', '| --- | --- | --- | --- |']
        for key, value, band in [('robust_sd', 'simulator_residual_robust_sd_ms', str(bounds['robust_sd_ms']) + ' ms'),
                                 ('beyond15', 'simulator_beyond15_share', str(bounds['tail_share'])),
                                 ('offpulse', 'offpulse_difference', '+/- ' + str(bounds['offpulse_tolerance'])),
                                 ('phase_tv', 'phase_tv', '<= ' + str(bounds['phase_tv_max']))]:
            status = 'in band' if simulator['bands'][key] else 'out of band'
            lines.append(f"| {key} | {number(simulator[value])} | {band} | {status} |")
        coverage = simulator['owner_chart_coverage']
        lines += ['', f"Owner off-pulse chart-head coverage: {number(coverage['share'])}, {coverage['n']} marks. Run-length distributions and passage details are in P6.json; reached regions and source-cell posteriors are in P8-s0.json and P8-s1.json."]
    lines += ['', '## Deviations and failed paths', '', json.dumps(results['deviations'], indent=2), '',
              json.dumps(results['failed_paths'], indent=2), '', '## Unfinished', '',
              ', '.join(results['unfinished']) or 'None.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    return results
