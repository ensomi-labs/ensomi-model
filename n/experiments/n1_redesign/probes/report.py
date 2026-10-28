"""Paired margins and conservative both-seed screening verdicts."""
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .common import OUT, digest, dump, interval, paired, prereg_check, utc


def read(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def stat(data, *path):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data if isinstance(data, dict) and 'recordings' in data else None


def average_stats(stats):
    stats = [s for s in stats if s is not None]
    if not stats:
        return None
    keys = set.intersection(*(set(s['recordings']) for s in stats))
    values = {k: float(np.mean([s['recordings'][k] for s in stats])) for k in keys}
    return dict(**interval(list(values.values())), recordings=values)


def compare(a, b, *, threshold, direction='above', refits=None, episode_spreads=None,
            require_refits=False):
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
        shifted_null=probe_stats('P1', ['panels:offline'], 'mlp-F', '-shift'))
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
    output['P5b'] = dict(verdict=verdict(pred, retire), predictions=pred, retire=retire,
                         B_tracked_repeat=tracked,
                         B_worth_building=compare(probe_stats('P5b', ['B_summary'], 'B_tracked_any'), None, threshold=.3))
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
            path = [tag, 'summary', 'panels']
            base = probe_stats(experiment, path, 'Pos:null0')
            f = probe_stats(experiment, path, 'Pos+F:null0')
            m = probe_stats(experiment, path, 'Pos+M:null0')
            def refit_layers(left, right):
                vals = {}
                for seed in (0, 1):
                    data = read(f'{experiment}-s{seed}.json'); v = []
                    for b in range(5):
                        aa = stat(data, *path, f'{left}:null0:boot{b}')
                        bb = stat(data, *path, f'{right}:null0:boot{b}')
                        if aa and bb:
                            v.append(paired(aa['recordings'], bb['recordings'])['mean'])
                    vals[seed] = v
                return vals
            pred = dict(feature_gain_005=compare(f, base, threshold=.05, refits=refit_layers('Pos+F', 'Pos')))
            if experiment == 'P4a':
                pred['feature_over_Mel_002'] = compare(f, m, threshold=.02, refits=refit_layers('Pos+F', 'Pos+M'))
            retirement = [compare(v, base, threshold=.02, direction='below', refits=refit_layers(name, 'Pos'))
                          for v, name in ((f, 'Pos+F'), (m, 'Pos+M'))]
            per_target[tag] = dict(predictions=pred, retire=dict(feature_gain_below_002=retirement[0], Mel_gain_below_002=retirement[1]),
                verdict='retire under mapper-agreement screen' if all(v['holds_both_seeds'] for v in retirement) else verdict(pred, {}))
        output[experiment] = dict(verdict='characterises; see per-target power and margins', targets=per_target,
            predictions={f'{tag}:{k}': v for tag, row in per_target.items() for k, v in row['predictions'].items()},
            retire={f'{tag}:{k}': v for tag, row in per_target.items() for k, v in row['retire'].items()})
    return output


def episode_stats(arm, metric, kind='real', cell='all'):
    return {s: stat(read(f'P7-s{s}.json'), 'summary', f'{arm}:{kind}:{cell}', metric) for s in (0, 1)}


def episode_spread(arm, metric, other=None):
    result = {}
    for seed in (0, 1):
        data = read(f'P7-s{seed}.json')
        spreads = []
        for cells in (['sim0', 'sim1'], ['pos0', 'pos1', 'pos2']):
            values = []
            for cell in cells:
                a = stat(data, 'summary', f'{arm}:real:{cell}', metric)
                b = stat(data, 'summary', f'{other}:real:{cell}', metric) if other else None
                if a and (not other or b):
                    value = paired(a['recordings'], b['recordings'])['mean'] if b else a['mean']
                    if value is not None:
                        values.append(value)
            if len(values) == len(cells):
                spreads.append(max(values) - min(values))
        learners = [stat(data, 'summary', f'learned{k}:real:all', metric) for k in (0, 1)]
        if all(learners):
            spreads.append(abs(learners[0]['mean'] - learners[1]['mean']))
        result[seed] = max(spreads) if len(spreads) == 3 else None
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
            points.append((a['mean'], float(key.split(':')[1]), y))
    points.sort(key=lambda p: (p[0], p[1]))
    for left, right in zip(points[:-1], points[1:]):
        if left[0] <= away <= right[0] and right[0] > left[0]:
            weight = (away - left[0]) / (right[0] - left[0])
            keys = set(left[2]['recordings']) & set(right[2]['recordings'])
            values = {r: left[2]['recordings'][r] * (1 - weight) + right[2]['recordings'][r] * weight for r in keys}
            return dict(**interval(list(values.values())), recordings=values,
                        away=away, gains=[left[1], right[1]], interpolation_weight=weight)
    return None


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
                                        episode_spreads=episode_spread('graph', 'M8'))
    retire = dict(away_over_035=compare(episode_stats('graph', 'M4_away'), None, threshold=.35,
                                      episode_spreads=episode_spread('graph', 'M4_away')))
    similarity_no_gain = compare(graph, local, threshold=0, direction='below',
                                  episode_spreads=episode_spread('graph', 'M3i', 'random'))
    for row in similarity_no_gain['seeds'].values():
        if row.get('power') and row.get('margin') is not None and row['contrast']['mean'] is not None:
            row['status'] = 'holds' if row['contrast']['mean'] <= row['margin'] else 'does_not_hold'
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
                if real[seed] and null[seed] and real[seed]['mean'] > 0:
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
        null_good = all(ratios[kind][str(seed)]['ratio'] is not None and ratios[kind][str(seed)]['ratio'] <= .2
                        for kind in ratios for seed in (0, 1))
        shuffle_bad = all(ratios['shuffle'][str(seed)]['ratio'] is not None and ratios['shuffle'][str(seed)]['ratio'] > .5 for seed in (0, 1))
        learned_verdict = ('retire under mapper-agreement screen' if shuffle_bad or all(v['holds_both_seeds'] for v in retiring.values())
                           else 'pass under mapper-agreement screen' if null_good and all(v['holds_both_seeds'] for v in comparisons.values())
                           else 'inconclusive')
        learned[arm] = dict(verdict=learned_verdict, matched_away=comparisons, retire_matched_away=retiring,
            null_retention=ratios, null_retention_limit_met=null_good, shuffle_retire=shuffle_bad,
            over_random=compare(real, random, threshold=0, episode_spreads=episode_spread(arm, 'M3i', 'random')))
    invariant = {str(s): read(f'P7-invariants-s{s}.json') for s in (0, 1)}
    invariant_ok = all(invariant[str(s)] and invariant[str(s)].get('complete') and invariant[str(s)].get('breaches') == 0 for s in (0, 1))
    packet_arms = []
    if invariant_ok:
        if pred['graph_over_random']['holds_both_seeds']:
            packet_arms.append('graph')
        packet_arms += [a for a, v in learned.items() if v['over_random']['holds_both_seeds']]
    positions = {arm: {str(s): {str(k): stat(read(f'P7-s{s}.json'), 'summary', f'{arm}:real:pos{k}', 'M3i')
                               for k in range(4)} for s in (0, 1)} for arm in ('graph', 'learned0', 'learned1')}
    self_null = {arm: episode_stats(arm, 'changed', 'self') for arm in ('mark', 'local', 'graph', 'learned0', 'learned1', 'bias', 'random')}
    return dict(verdict=verdict(pred, retire), predictions=pred, retire=retire,
                learned=learned, invariants=invariant, packet_eligible_arms=packet_arms,
                mark_position_estimates=positions, self_label_null=self_null,
                M10_note='Position spreads are included in the preregistered episode margin. Flatness under that same margin is not an independent equivalence test; report the position estimates directly.')


def build():
    lock = prereg_check()
    verdicts = probe_verdicts()
    verdicts['P7'] = propagation_verdict()
    results = dict(generated_utc=utc(), prereg=lock, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        evidence='Mapper agreement only; no ear-validated measure and no adoption.',
        inputs=read('inputs.json'), cohorts={str(s): read(f'counts-s{s}.json') for s in (0, 1)},
        verdicts=verdicts, packet=read('packet-key/key.json'),
        deviations=read('deviations.json') or [], failed_paths=read('failed-paths.json') or [])
    # Keep the public-facing report free of identities; the complete private
    # packet key remains in its own owner and is represented here by composition.
    if results['packet']:
        results['packet'] = dict(path=str(OUT / 'packet'), composition=results['packet']['composition'])
    expected = ['P1', 'P2', 'P3', 'P4a', 'P4b', 'P5a', 'P5b', 'P5c', 'P7', 'P8']
    results['unfinished'] = [f'{p}-s{s}' for p in expected for s in (0, 1) if read(f'{p}-s{s}.json') is None]
    if read('P6.json') is None:
        results['unfinished'].append('P6')
    results['unfinished'].append('P5d (stretch)')
    results['experiment_files'] = {p.name: dict(sha256=digest(p), path=str(p)) for p in sorted(OUT.glob('P*.json'))}
    results['runtime_sha256'] = digest(OUT / 'runtime.json')
    dump(OUT / 'results.json', results)
    lines = ['# N1 frozen-feature probes', '', results['evidence'], '',
             f"Preregistration: `{lock['sha256']}`, frozen {lock['utc']}.",
             f"Code: `{results['source_commit']}` on `codex/n1-probes`.", '',
             '## Screening verdicts', '', '| Experiment | Both-seed verdict |', '| --- | --- |']
    for name, v in verdicts.items():
        lines.append(f"| {name} | {v['verdict']} |")
    lines += ['', 'The numbers, recording intervals, paired contrasts, margin components and power checks for every criterion are in results.json. Raw per-seed summaries and strata are in the hashed experiment files listed there. Missing evidence prevents a pass.', '',
              '## Cohorts and counts', '', '```json', json.dumps(results['cohorts'], indent=2), '```', '',
              '## Criterion estimates', '']
    for name, v in verdicts.items():
        lines += [f'### {name}', '', '| Criterion | Seed | Estimate / contrast | 95% CI | Margin | n | Status |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for category in ('predictions', 'retire'):
            for criterion, c in v.get(category, {}).items():
                for s, val in c['seeds'].items():
                    d = val.get('contrast', {})
                    mean = d.get('mean')
                    lines.append(f"| {category}: {criterion} | {s} | {mean if mean is not None else 'unavailable'} | {d.get('ci95')} | {val.get('margin')} | {d.get('n', 0)} | {val['status']} |")
        lines.append('')
    lines += ['## P7 invariants and packet', '', '```json', json.dumps(verdicts['P7']['invariants'], indent=2), '```', '',
              json.dumps(results['packet'], indent=2) if results['packet'] else 'Packet not yet built.', '',
              '## Deviations and failed paths', '', json.dumps(results['deviations'], indent=2), '',
              json.dumps(results['failed_paths'], indent=2), '', '## Unfinished', '',
              ', '.join(results['unfinished']) or 'None.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    return results
