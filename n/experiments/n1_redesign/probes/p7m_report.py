"""Recording-balanced P7m contrasts, null diagnostics and gain curves."""
import json
from pathlib import Path
import subprocess

import numpy as np

from .common import OUT, digest, dump, interval, paired, utc
from .p7m import DEST, NEW_ARMS, REFERENCES, CURVE_GAINS, frozen
from .report import compare, stat


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def metric(data, arm, name, kind='real', cell='all'):
    return stat(data, 'summary', f'{arm}:{kind}:{cell}', name)


def compact(value):
    if value is None:
        return None
    return {k: v for k, v in value.items() if k != 'recordings' and not k.startswith('_')}


def episode_spread(data, arm, name, other=None, kind='real'):
    """Retain P7's simulator/position contrasts and learned-training noise floor."""
    strata = [['sim0', 'sim1']]
    if not kind.startswith('confirming-'):
        strata.append(['pos0', 'pos1', 'pos2'])
    spreads = []
    for cells in strata:
        values = []
        for cell in cells:
            a = metric(data, arm, name, kind, cell)
            b = metric(data, other, name, kind, cell) if other else None
            if a and (not other or b):
                value = paired(a['recordings'], b['recordings'])['mean'] if b else a['mean']
                if value is not None:
                    values.append(value)
        if len(values) == len(cells):
            spreads.append(max(values) - min(values))
    learners = [metric(data, f'learned{k}', name, kind) for k in (0, 1)]
    if all(x and x['mean'] is not None for x in learners):
        spreads.append(abs(learners[0]['mean'] - learners[1]['mean']))
    return max(spreads) if len(spreads) == len(strata) + 1 else None


def contrast(data, arm, other, name='M3i', threshold=0, direction='above', kind='real'):
    a = {s: metric(data[s], arm, name, kind) for s in (0, 1)}
    b = {s: metric(data[s], other, name, kind) for s in (0, 1)} if other else None
    spreads = {s: episode_spread(data[s], arm, name, other, kind) for s in (0, 1)}
    return compare(a, b, threshold=threshold, direction=direction, episode_spreads=spreads)


def within_margin(comparison):
    result = json.loads(json.dumps(comparison))
    for row in result['seeds'].values():
        mean = row.get('contrast', {}).get('mean')
        if row.get('power') and row.get('margin') is not None and mean is not None:
            row['status'] = 'holds' if mean <= row['margin'] else 'does_not_hold'
        else:
            row['status'] = 'inconclusive'
    result['holds_both_seeds'] = all(r['status'] == 'holds' for r in result['seeds'].values())
    result['rule'] = 'Paired M3i gain over matched random is no larger than its comparison margin.'
    return result


def curve_points(data, arm):
    points = []
    for gain in CURVE_GAINS:
        values = (data or {}).get('curves', {}).get(f'{arm}:{gain}', {})
        points.append(dict(gain=gain, away=values.get('M4_away'), recovery=values.get('M3i')))
    return points


def curve_at(data, arm, target):
    points = curve_points(data, arm)
    for left, right in zip(points[:-1], points[1:]):
        x = [p['away']['mean'] if p['away'] else None for p in (left, right)]
        y = [p['recovery'] for p in (left, right)]
        if any(v is None for v in x) or any(v is None or v['mean'] is None for v in y):
            continue
        if min(x) <= target <= max(x) and x[0] != x[1]:
            weight = (target - x[0]) / (x[1] - x[0])
            recs = sorted(set(y[0]['recordings']) & set(y[1]['recordings']))
            values = [y[0]['recordings'][r] * (1 - weight) + y[1]['recordings'][r] * weight for r in recs]
            return dict(**interval(values), away=target, gains=[left['gain'], right['gain']],
                        interpolation_weight=weight,
                        limitation='Descriptive fixed-crossing interpolation; not selected on panels')
    return None


def retention(data, arm, kind):
    real = metric(data, arm, 'M3i')
    null = metric(data, arm, 'M3i', kind)
    if not real or not null:
        return None
    recs = sorted(set(real['recordings']) & set(null['recordings']))
    if not recs:
        return None
    a = np.array([real['recordings'][r] for r in recs])
    b = np.array([null['recordings'][r] for r in recs])
    ratio = float(b.mean() / a.mean()) if a.mean() != 0 else None
    rng = np.random.default_rng(20261028)
    ratios = []
    for _ in range(1000):
        ix = rng.integers(len(recs), size=len(recs))
        denominator = a[ix].mean()
        if denominator != 0:
            ratios.append(float(b[ix].mean() / denominator))
    ci = np.quantile(ratios, [.025, .975]).tolist() if len(ratios) == 1000 else None
    return dict(n=len(recs), real=float(a.mean()), null=float(b.mean()), retention=ratio,
                ci95=ci, bootstrap_nonzero_denominators=len(ratios))


def verdicts(data, invariants):
    predictions = dict(
        mel_over_local_010=contrast(data, 'mel-graph', 'local', threshold=.10),
        mel_over_random=contrast(data, 'mel-graph', 'mel-random'),
        mel_away_at_most_020=contrast(data, 'mel-graph', None, name='M4_away', threshold=.20, direction='below'))
    retire = dict(within_random_margin=within_margin(predictions['mel_over_random']),
                  away_over_035=contrast(data, 'mel-graph', None, name='M4_away', threshold=.35))
    complete = all(x and x.get('complete') and not x.get('breaches') for x in invariants.values())
    breach = any(x and x.get('breaches') for x in invariants.values())
    passed = complete and all(x['holds_both_seeds'] for x in predictions.values())
    retired = complete and any(x['holds_both_seeds'] for x in retire.values())
    verdict = ('stopped: invariant breach' if breach else 'incomplete' if not complete else
               'retire' if retired else 'pass' if passed else 'inconclusive')
    oracle_stats = {s: metric(data[s], 'oracle-graph', 'M3i') for s in (0, 1)}
    below = complete and all(v and v['n'] >= 40 and v['mean'] is not None and v['mean'] < .10
                            for v in oracle_stats.values())
    oracle = dict(characterises=True, absolute_recovery_below_010_both=bool(below),
                  over_local=contrast(data, 'oracle-graph', 'local'),
                  over_random=contrast(data, 'oracle-graph', 'oracle-random'),
                  interpretation=('Under the registered dev selection, the fixed cell layer and episode/graph definition '
                                  'bound this exact-repeat propagation: correct repeat edges alone do not produce '
                                  'ten-percentage-point recovery at the selected gain.' if below else
                                  'Read the oracle as a ceiling for the registered exact-repeat graph only.'))
    return dict(verdict=verdict, prediction_holds_both=passed, retire_holds_both=retired,
                predictions=predictions, retire=retire, oracle=oracle)


def cached_mel_parity(data, configs):
    """Check episode reproduction separately from the absolute invariants."""
    result = {}
    for seed in (0, 1):
        if not data[seed] or not configs[seed]:
            result[str(seed)] = dict(assessed=False)
            continue
        old_config = read(OUT / f'P7-config-s{seed}.json')['configs']['mel']
        new_config = configs[seed]['configs']['mel-graph']
        matching_config = all(old_config[k] == new_config[k] for k in ('gain', 'theta', 'K', 'local'))
        old = read(OUT / f'P7-s{seed}.json')
        checked, differences = 0, []
        for group, metrics in data[seed]['summary'].items():
            arm, kind, cell = group.split(':')
            if arm != 'mel-graph' or kind == 'shuffle':
                continue
            previous = old['summary'].get(f'mel:{kind}:{cell}', {})
            for name, value in metrics.items():
                a, b = value['recordings'], previous.get(name, {}).get('recordings', {})
                checked += 1
                if a.keys() != b.keys() or any(abs(a[k] - b[k]) > 1e-12 for k in a.keys() & b.keys()):
                    differences.append(dict(group=group, metric=name))
        result[str(seed)] = dict(assessed=True, matching_selected_config=matching_config,
                                recording_statistic_groups=checked, differences=differences,
                                passed=matching_config and not differences)
        del old
    return result


def figures(data, configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout='constrained')
    styles = [('mel-graph', '#0072B2', '-'), ('oracle-graph', '#D55E00', '-'),
              ('mel-random', '#0072B2', '--'), ('oracle-random', '#D55E00', '--'),
              ('local', '#009E73', '-'), ('graph', '#CC79A7', '-'),
              ('bias', '#777777', '-'), ('mark', '#E6AB02', '-')]
    for seed, ax in enumerate(axes):
        for arm, color, style in styles:
            points = curve_points(data[seed], arm)
            xx, yy = [], []
            for point in points:
                x, y = point['away'], point['recovery']
                valid = x and y and x['mean'] is not None and y['mean'] is not None
                xx.append(x['mean'] if valid else np.nan)
                yy.append(y['mean'] if valid else np.nan)
                if valid:
                    marker = 'o' if min(x['n'], y['n']) >= 40 else 'x'
                    ax.errorbar(x['mean'], y['mean'],
                        yerr=[[max(0, y['mean'] - y['ci95'][0])], [max(0, y['ci95'][1] - y['mean'])]],
                        color=color, marker=marker, markersize=3, capsize=2, linewidth=.7)
            gain = (configs[seed] or {}).get('configs', {}).get(arm, {}).get('gain')
            label = f'{arm} (selected g={gain})'
            if not any(np.isfinite(xx)):
                label += '; no defined harm points'
            ax.plot(xx, yy, style, color=color, linewidth=1, label=label)
            for point, x, y in zip(points, xx, yy):
                if point['gain'] == gain and np.isfinite([x, y]).all():
                    ax.plot(x, y, 's', color=color, markersize=7, markerfacecolor='none')
        for target in (.1, .2):
            ax.axvline(target, color='#999999', linestyle=':', linewidth=.8)
        ax.set_title(f'Feature seed {seed}')
        ax.set_xlabel('Away-share among outside-region MAP changes')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=7, loc='upper left')
    axes[0].set_ylabel('M3i: recovery in other failing ternary regions')
    fig.suptitle('Mapper agreement: single marks, no confirmations; recording means and 95% intervals\n'
                 'Squares: selected gain on this slice. Crosses: fewer than 40 recordings. Undefined harm breaks curves.', fontsize=10)
    fig.savefig(DEST / 'curves.png', dpi=180)
    plt.close(fig)


def fmt(statistic):
    if not statistic or statistic.get('mean') is None:
        return 'undefined (n=0)'
    low, high = statistic['ci95']
    return f"{statistic['mean']:.6f} [{low:.6f}, {high:.6f}], n={statistic['n']}"


def build():
    lock = frozen()
    simulator_path = OUT / 'P6.json'
    simulator = read(simulator_path)
    simulator_status = ('unassessed' if simulator is None else
                        'simulator optimistic' if simulator['simulator_optimistic'] else 'within registered bands')
    data = {s: read(DEST / f'seed-{s}.json') for s in (0, 1)}
    configs = {s: read(DEST / f'config-s{s}.json') for s in (0, 1)}
    invariants = {s: read(DEST / f'invariants-s{s}.json') for s in (0, 1)}
    result = verdicts(data, invariants)
    result.update(prereg=lock, generated_utc=utc(),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  code_sha256={str(p): digest(p) for p in Path(__file__).parent.glob('p7m*.py')},
                  inputs=read(DEST / 'inputs.json'), configurations=configs, invariants=invariants,
                  resources=[json.loads(line) for line in (DEST / 'resources.jsonl').read_text().splitlines()]
                            if (DEST / 'resources.jsonl').exists() else [],
                  deviations=read(DEST / 'deviations.json') or [],
                  failed_paths=read(DEST / 'failed-paths.json') or [],
                  explicit_exclusions=read(DEST / 'explicit-exclusions.json'),
                  simulator=dict(status=simulator_status, path=str(simulator_path),
                                 sha256=digest(simulator_path) if simulator else None),
                  evidence_limit='Mapper agreement only; simulator status is inherited from the prior screen.')
    result['confirming_diagnostics'] = {arm: {kind: contrast(data, arm, None, name='M8',
        threshold=.01, direction='below', kind=kind) for kind in ('confirming-fail', 'confirming-nofail')}
        for arm in NEW_ARMS}
    result['seeds'] = {}
    arms = (*NEW_ARMS, *REFERENCES)
    for seed in (0, 1):
        d = data[seed]
        selected, nulls, strata, curves, equal_gain = {}, {}, {}, {}, {}
        for arm in arms:
            selected[arm] = {name: compact(value) for name, value in
                            (d or {}).get('summary', {}).get(f'{arm}:real:all', {}).items()}
            nulls[arm] = {kind: {name: compact(value) for name, value in
                         (d or {}).get('summary', {}).get(f'{arm}:{kind}:all', {}).items()}
                         for kind in ('self', 'uninformative', 'shuffle', 'confirming-fail', 'confirming-nofail')}
            for kind in ('uninformative', 'shuffle'):
                nulls[arm][kind]['M3_retention'] = retention(d, arm, kind)
            strata[arm] = {cell: {name: compact(value) for name, value in
                          (d or {}).get('summary', {}).get(f'{arm}:real:{cell}', {}).items()}
                          for cell in ('pos0', 'pos1', 'pos2', 'pos3', 'run1', 'run3', 'k0', 'k8', 'ge160', 'lt160')}
            curves[arm] = dict(points=[dict(gain=p['gain'], away=compact(p['away']), recovery=compact(p['recovery']))
                                      for p in curve_points(d, arm)],
                               at_010=curve_at(d, arm, .1), at_020=curve_at(d, arm, .2))
        for arm in ('mel-graph', 'oracle-graph'):
            gain = (configs[seed] or {}).get('configs', {}).get(arm, {}).get('gain')
            a = stat(d, 'curves', f'{arm}:{gain}', 'M3i')
            b = stat(d, 'curves', f"{arm.replace('-graph', '-random')}:{gain}", 'M3i')
            equal_gain[arm] = dict(gain=gain, contrast=paired(a['recordings'], b['recordings'])) if a and b else None
        self_diagnostics = {}
        for arm in NEW_ARMS:
            changed = nulls[arm]['self'].get('changed')
            self_diagnostics[arm] = ('unassessed' if not changed or changed['mean'] is None else
                                     'pass' if changed['mean'] == 0 else 'fail')
        result['seeds'][str(seed)] = dict(complete=bool(invariants[seed] and invariants[seed].get('complete')),
            cohorts=(d or {}).get('cohorts'), regions=(d or {}).get('regions'), selected=selected,
            nulls=nulls, strata=strata, curves=curves, equal_gain_random_on_curve_slice=equal_gain,
            self_label_zero_change=self_diagnostics,
            recording_statistics_path=str(DEST / f'seed-{seed}.json'))
    result['unfinished'] = [f'Seed {s} measurements/invariants' for s in (0, 1)
                            if not result['seeds'][str(s)]['complete']]
    result['cached_mel_parity'] = cached_mel_parity(data, configs)
    figures(data, configs)
    dump(DEST / 'results.json', result)
    lines = ['# P7m: Mel propagation and exact-repeat oracle', '',
             f"Mel screening verdict: **{result['verdict']}**. Measurement only; nothing is adopted.", '',
             result['oracle']['interpretation'], '',
             'All outcomes are agreement with mapper annotations. No measure is validated against the ear. '
             f'The unchanged P7 simulator inherits P6’s status: **{simulator_status}**. The oracle is '
             'restricted to exact non-trivial repeats, the same beat position, a lag of at least eight bars, '
             'and P7’s top-K cap; it is not a ceiling for every possible musical-similarity rule.', '',
             f"Preregistered {lock['time']}; SHA-256 `{lock['sha256']}`. Code `{result['source_commit']}`.", '',
             '## Selected settings and effects', '',
             'M3i is recovery beyond the marked region, excluding each mark’s immediate neighborhood. '
             'M4-away is conditional on outside-region MAP changes; an inactive arm has undefined harm, not zero harm. '
             'Intervals bootstrap original recordings 1,000 times. Reference arms and their selected settings '
             'are reread from corrected P7 caches.', '',
             '| Seed | Arm | Gain | Threshold / K | M3i [95% interval], recordings | M4-away [95% interval], recordings |',
             '| --- | --- | --- | --- | --- | --- |']
    for seed in (0, 1):
        for arm in arms:
            selected = result['seeds'][str(seed)]['selected'][arm]
            cfg = (configs[seed] or {}).get('configs', {}).get(arm, {})
            lines.append(f"| {seed} | {arm} | {cfg.get('gain', 'missing')} | {cfg.get('theta')} / {cfg.get('K')} | {fmt(selected.get('M3i'))} | {fmt(selected.get('M4_away'))} |")
    lines += ['', 'A selected gain of zero disables the arm. At zero objective, the registered tie rule '
              'also selects the smaller run cap; the curves retain that selected topology. This is an '
              'operating-rule ceiling, not proof that every positive-gain exact-repeat update is ineffective.', '']
    for seed in (0, 1):
        for arm in ('mel-graph', 'oracle-graph'):
            diagnostics = (configs[seed] or {}).get('diagnostics', {}).get(arm, {})
            feasible = sum(x['feasible'] and x['config']['gain'] > 0 for x in diagnostics.get('grid', []))
            points = [p for p in result['seeds'][str(seed)]['curves'][arm]['points']
                      if p['recovery'] and p['recovery']['mean'] is not None]
            best = max(points, key=lambda p: p['recovery']['mean']) if points else None
            lines.append(f'Seed {seed}, {arm}: {feasible} feasible positive-gain dev settings. '
                         + (f"Largest sampled curve recovery {fmt(best['recovery'])} at gain {best['gain']}, "
                            f"away-share {fmt(best['away'])}; descriptive only." if best else 'Curve unfinished.'))
    lines += ['', '## Registered comparisons', '',
              '| Rule | Seed | Paired effect or statistic [95% interval] | Margin | Status |',
              '| --- | --- | --- | --- | --- |']
    for section in ('predictions', 'retire'):
        for name, comparison in result[section].items():
            for seed, row in comparison['seeds'].items():
                lines.append(f"| {section}: {name} | {seed} | {fmt(row.get('contrast'))} | {row.get('margin')} | {row['status']} |")
    lines += ['', 'The margin is the maximum of paired-bootstrap half-width, matched-baseline feature-seed '
              'spread, simulator/ordinary-position contrast ranges, and P7’s learned-training-seed range. '
              'Each component and all denominators are in results.json. Fewer than 40 contributing '
              'recordings or a missing component prevents a decision.', '',
              '## Nulls and invariants', '',
              '| Seed | Arm | Self-label mean MAP changes | Confirming M8, failing | Confirming M8, non-failing | Shuffle / real M3 | Uninformative / real M3 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for seed in (0, 1):
        for arm in NEW_ARMS:
            null = result['seeds'][str(seed)]['nulls'][arm]
            ratio = lambda kind: (null[kind].get('M3_retention') or {}).get('retention')
            lines.append(f"| {seed} | {arm} | {fmt(null['self'].get('changed'))} | {fmt(null['confirming-fail'].get('M8'))} | {fmt(null['confirming-nofail'].get('M8'))} | {ratio('shuffle')} | {ratio('uninformative')} |")
    for seed, value in invariants.items():
        lines += ['', f'Seed {seed} invariants: `{json.dumps(value, sort_keys=True)}`.']
    lines += ['', f"Cached Mel reproduction: `{json.dumps(result['cached_mel_parity'])}`."]
    for seed in (0, 1):
        lines += ['', f"Seed {seed} self-label zero-change diagnostics: `{json.dumps(result['seeds'][str(seed)]['self_label_zero_change'])}`."]
    lines += ['', 'Self-label expects exactly zero MAP changes anywhere; any positive count fails that '
              'expectation. Null ratios use shared recordings and remain undefined with a zero real-recovery '
              'denominator. A null failure is not repaired with chart-aware gating.', '',
              '## Gain curves and cohorts', '', '![Gain curves](curves.png)', '',
              'The curve slice uses one mark, no confirmations, both simulators and the three ordinary '
              'positions. Squares show the selected gain on this slice, which differs from the all-episode '
              'table. Missing harm coordinates break the line; crosses indicate fewer than 40 recordings. '
              'Readings at away-shares .10/.20 are descriptive adjacent-gain interpolations without '
              'extrapolation. All sampled points, strata and equal-gain random contrasts are in results.json.', '']
    for seed in (0, 1):
        row = result['seeds'][str(seed)]
        lines.append(f"Seed {seed}: {row['cohorts']}; eligible regions: {row['regions']}.")
    lines += ['', 'The non-failing cohort excludes every original recording that fails in any reference. '
              'Where fewer non-failing recordings are available than failing recordings, all available '
              'non-failing recordings are used. The input audit covers cached labels/tokens/posteriors '
              'only; pruned raw feature files are not accessed.', '',
              '## Limits, resources and unfinished work', '',
              'PCA and M0 were fitted on probe-fit; those recordings were already seen by the trunk. '
              'Graph selection uses probe-dev only, and panels remain design read sets. Both feature '
              'seeds use the same selected checkpoint; neighboring-checkpoint noise is unavailable.', '',
              'The Mel graph uses P7’s three-beat contexts. P5a’s bar retrieval used four-beat bar tokens; '
              'its retrieval advantage does not establish an advantage for this beat-context update rule. '
              'This run does not screen every possible Mel bar-graph design.', '',
              'The low-dimensional frozen cell probe, fixed chart grid, exact-repeat eligibility, run cap, '
              'small or negative mark LLRs, and M0 confidence can each constrain recovery. This comparison '
              'does not isolate the cell model alone. No new training or owner listening validation occurred.', '',
              f"Resource records: `{json.dumps(result['resources'])}`.", '',
              f"Deviations: `{json.dumps(result['deviations'])}`.", '',
              f"Failed paths: `{json.dumps(result['failed_paths'])}`.", '',
              'Unfinished: ' + ('; '.join(result['unfinished']) if result['unfinished'] else 'none within the registered P7m measurement scope.')]
    (DEST / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verdict=result['verdict'], unfinished=result['unfinished'])), flush=True)


if __name__ == '__main__':
    build()
