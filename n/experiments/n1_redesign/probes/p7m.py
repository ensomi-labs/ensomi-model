"""Mel and exact-repeat graph measurements on the frozen P7 episodes.

Run ``python -m n1_redesign.probes.p7m {prepare,seed,report}`` with a frozen
preregistration in the private output directory. Use p7m_guard for bounded runs.
"""
from collections import defaultdict
from dataclasses import replace
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from . import episodes as ep
from .common import (CACHE, OUT, PREV, assert_open, digest, dump, exclusion,
                     interval, labels, prereg_check, stable_seed, units, utc)
from .tokens import token_path

DEST = OUT / 'p7m'
NEW_ARMS = ('mel-graph', 'oracle-graph', 'mel-random', 'oracle-random')
REFERENCES = ('local', 'graph', 'bias', 'mark')
CURVE_GAINS = (0., .0625, .125, .25, .5, 1., 2., 4.)


def log(event, **values):
    row = dict(time=utc(), event=event, **values)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'job.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def frozen():
    prereg_check()
    lock = json.loads((DEST / 'prereg-lock.json').read_text())
    assert digest(DEST / 'prereg.md') == lock['sha256'], 'P7m preregistration changed'
    return lock


def oracle_similarity(s):
    """Exact non-trivial bar repeats, at the same position and lag >=8 bars."""
    z = s.z
    bar = np.searchsorted(z['bar_ids'], z['bars'])
    pair = s.exact & z['nontrivial'][:, None] & z['nontrivial'][None, :]
    pair &= abs(z['bar_ids'][:, None] - z['bar_ids'][None, :]) >= 8
    result = pair[bar[:, None], bar[None, :]] & (z['bib'][:, None] == z['bib'][None, :])
    np.fill_diagonal(result, True)
    return result.astype('float32')


def update(s, marks, arm, config, rng_seed=0):
    if arm in REFERENCES:
        return ep.update(s, marks, arm, config, rng_seed=rng_seed)
    if arm not in NEW_ARMS:
        raise ValueError(f'Unknown graph arm: {arm}')
    if arm.startswith('mel-'):
        similarity = s.mel_similarity
    else:
        if 'oracle_similarity' not in s.metric_cache:
            s.metric_cache['oracle_similarity'] = oracle_similarity(s)
        similarity = s.metric_cache['oracle_similarity']
    view = replace(s, similarity=similarity)
    return ep.update(view, marks, 'random' if arm.endswith('-random') else 'graph',
                     config, rng_seed=rng_seed)


def fixtures():
    original = ep.fixtures()
    z = dict(beats=np.array([0., 500., 1000., 1125.5, 1525.5, 1925.5, 2325.5]),
             length=np.array([500., 500., 500., 400., 400., 400., 400.]),
             red=np.array([[0., 500., 4], [1125.5, 400., 3], [1925.5, 400., 4]]),
             bars=np.array([0, 0, 0, 1, 1, 9, 9]), bib=np.array([0, 1, 2, 0, 1, 0, 1]),
             bar_ids=np.array([0, 1, 9]), nontrivial=np.ones(3, bool),
             y=np.array([0, 1, -1, 1, 0, 1, 0]))
    c = np.eye(7, dtype=np.float32)
    p = np.array([.1, .2, .3, .4, .5, .6, .7], np.float32)
    s = ep.Song({}, z, p, c, c.copy(), c @ c.T, c @ c.T, np.eye(3),
                np.ones((3, 3), bool), [], ep.pulse_hash(z))
    checked = 0
    for arm in NEW_ARMS:
        for marks in ([], [dict(beat=6, L=2., kind=0)], [dict(beat=3, L=-1., kind=0)]):
            new, _, declared = update(s, marks, arm, dict(gain=.5, local=.4, theta=.6, K=3))
            assert ep.pulse_hash(z) == s.hash
            assert np.array_equal(declared, np.flatnonzero(new != p))
            if not marks:
                assert new.tobytes() == p.tobytes()
            checked += 1
    return dict(passed=True, updates=checked, original=original, pulse_sha256=s.hash)


def prepare():
    lock = frozen()
    e = exclusion(DEST / 'exclusion-audit.json')
    selected = [u for u in units() if u['cohort'] in ('probe-dev', 'panel', 'val_hard')]
    manifests = {}
    for seed in (0, 1):
        path = CACHE / f'v01-s{seed}-009001/manifest.jsonl'
        manifests[seed] = {r['sha256']: r for r in map(json.loads, path.read_text().splitlines())}
    files = [OUT / n for n in ('units.json', 'inputs.json', 'prereg-lock.json', 'runtime.json')]
    files += [CACHE / 'v01-s0-009001/PRUNED.md', CACHE / 'v01-s1-009001/DONE']
    for seed in (0, 1):
        files.extend([CACHE / f'v01-s{seed}-009001/manifest.jsonl',
                      OUT / f'P7-config-s{seed}.json', OUT / f'P7-s{seed}.json'])
    for u in selected:
        assert_open(dict(u, audio_sha256=u['sha'], audio_rel=u['audio']), e)
        files.append(OUT / 'labels' / (u['id'] + '.npz'))
        for seed in (0, 1):
            row = manifests[seed][u['sha']]
            assert_open(row, e)
            files.extend([token_path(seed, u), OUT / 'm0' / f's{seed}' / (u['id'] + '.npy')])
    provenance = {str(p): digest(p) for p in sorted(set(files))}
    counts = {cohort: dict(units=sum(u['cohort'] == cohort for u in selected),
              recordings=len({u['rec'] for u in selected if u['cohort'] == cohort}))
              for cohort in ('probe-dev', 'panel', 'val_hard')}
    source = {str(p): digest(p) for p in Path(__file__).parent.glob('*.py')}
    dump(DEST / 'inputs.json', dict(prereg=lock, files=provenance, cohorts=counts,
         sources=source, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         feature_access='Cached labels, tokens and posteriors only; pruned raw features not read'))
    dump(DEST / 'fixtures.json', fixtures())
    log('prepared', cohorts=counts, hashed_files=len(provenance), prereg=lock)


def accumulator():
    return defaultdict(lambda: defaultdict(lambda: [0., 0]))


def summarize(groups):
    result = {}
    for group, metrics in groups.items():
        result[group] = {}
        for metric, recs in metrics.items():
            values = ep.recording_values(recs)
            result[group][metric] = dict(**interval(list(values.values())), recordings=values)
    return result


def choose_config(configs, observations, base):
    stats = []
    for cfg, ob in zip(configs, observations):
        recs = {k: ep.recording_values(g) for k, g in ob.items()}
        means = {k: float(np.mean(list(g.values()))) for k, g in recs.items() if g}
        counts = {k: len(g) for k, g in recs.items()}
        power = all(counts.get(k, 0) >= 40 for k in ('M1e', 'M3i', 'M8'))
        power &= counts.get('M4_away', 0) == 0 or counts['M4_away'] >= 40
        feasible = power and means.get('M8', 1) <= .01 and means.get('M4_away', 0) <= .20
        stats.append(dict(config=cfg, means=means, n=counts, power=power, feasible=feasible,
                          objective=means.get('M1e', 0) + means.get('M3i', 0)))
    allowed = [s for s in stats if s['feasible']]
    selected = sorted(allowed, key=lambda a: (-a['objective'], a['config']['gain'],
                      -a['config']['theta'], a['config']['K'], -a['config']['local']))[0] if allowed else dict(
                      config=dict(base, gain=0.), means={}, n={}, feasible=False, objective=0.)
    return dict(selected=selected, grid=stats)


def tune(seed):
    path = DEST / f'config-s{seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    old = json.loads((OUT / f'P7-config-s{seed}.json').read_text())['configs']
    chosen = {arm: old[arm] for arm in REFERENCES}
    diagnostics = {}
    dev = ep.tuning_units(seed)
    for arm in NEW_ARMS:
        base = dict(local=old['local']['local'], theta=.6, K=3)
        if arm.endswith('-random'):
            base = chosen[arm.replace('-random', '-graph')]
            configs = [dict(base, gain=g) for g in ep.GAINS]
        else:
            configs = [dict(base, theta=t, K=k, gain=g)
                       for t in (.4, .6, .8) for k in (1, 3, 8) for g in ep.GAINS]
        observations = [accumulator() for _ in configs]
        checked = 0
        for num, u in enumerate(dev):
            s = ep.song(seed, u)
            valid = [r for r in s.regions if len(ep.eligible(s.z, r))]
            r = valid[stable_seed(u['id']) % len(valid)] if valid else np.zeros(len(s.p), bool)
            for kind, count in [('real', 0), ('real', 8), ('confirming', 8)]:
                if kind == 'real' and not valid:
                    continue
                marks = ep.simulate(s.z, s.p, r, s.c, simulator=0, position=1,
                         run_length=1, confirming=count, identity=u['id'],
                         null='confirming' if kind == 'confirming' else None)
                if marks is None:
                    continue
                for i, cfg in enumerate(configs):
                    p, loc, _ = update(s, marks, arm, cfg, stable_seed(u['id']))
                    checked += 1
                    values = ep.measures(s, r, marks, p, loc)
                    for key in (['M8'] if kind == 'confirming' else ['M1e', 'M3i', 'M4_away']):
                        ep.add_measure(observations[i], u['rec'], key, values[key], values)
            if num % 100 == 0:
                log('tune_progress', seed=seed, arm=arm, units=num, updates=checked)
        result = choose_config(configs, observations, base)
        result['invariant_checks'] = checked
        chosen[arm] = result['selected']['config']
        diagnostics[arm] = result
        log('tuned', seed=seed, arm=arm, selected=result['selected'])
        dump(DEST / f'tuning-progress-s{seed}.json', dict(configs=chosen, diagnostics=diagnostics))
    result = dict(configs=chosen, diagnostics=diagnostics, dev_units=len(dev))
    dump(path, result)
    return result


def evaluate(seed, configs):
    panel, failing_ids, nofail_ids, recordings_with_failure = ep.panel_partitions(seed)
    summary, curves = defaultdict(accumulator), defaultdict(accumulator)
    chosen_nofail = {u['id'] for u in ep.choose_nonfailing(failing_ids, nofail_ids, recordings_with_failure)}
    checked = skipped = region_count = nofail_count = fail_count = 0
    cohort = dict(panel_units=len(panel), panel_recordings=len({u['rec'] for u in panel}),
                  failing_recordings=len({u['rec'] for u in failing_ids}),
                  nonfailing_recordings=len({u['rec'] for u in panel if u['id'] in chosen_nofail}))

    def record(s, arm, kind, position, sim, run_length, k, values):
        band = 'ge160' if np.median(60000 / s.z['length']) >= 160 else 'lt160'
        groups = [f'{arm}:{kind}:all', f'{arm}:{kind}:run{run_length}', f'{arm}:{kind}:k{k}',
                  f'{arm}:{kind}:pos{position}', f'{arm}:{kind}:sim{sim}', f'{arm}:{kind}:{band}']
        for group in groups:
            for key, v in values.items():
                ep.add_measure(summary[group], s.unit['rec'], key, v, values)

    with gzip.open(DEST / f'episodes-s{seed}.jsonl.gz', 'wt') as raw:
        for num, u in enumerate(panel):
            s = ep.song(seed, u)
            for arm in NEW_ARMS:
                update(s, [], arm, configs[arm]); checked += 1
            regs = [r for r in s.regions if len(ep.eligible(s.z, r))]
            if not regs:
                if u['id'] not in chosen_nofail:
                    continue
                nofail_count += 1
                r = np.zeros(len(s.p), bool)
                for sim in (0, 1):
                    marks = ep.simulate(s.z, s.p, r, s.c, simulator=sim, position=1,
                                       run_length=1, confirming=8, identity=u['id'], null='confirming')
                    for arm in NEW_ARMS:
                        p, loc, _ = update(s, marks, arm, configs[arm], num); checked += 1
                        record(s, arm, 'confirming-nofail', 1, sim, 1, 8, ep.measures(s, r, marks, p, loc))
                continue
            fail_count += 1
            donors = [v for v in failing_ids if v['rec'] != u['rec']]
            donor = ep.song(seed, donors[stable_seed(u['id']) % len(donors)]) if donors else s
            donor_regs = [r for r in donor.regions if len(ep.eligible(donor.z, r))]
            for ri, r in enumerate(regs):
                region_count += 1
                for sim in (0, 1):
                    for position in range(4):
                        for run_length in (1, 3):
                            for k in (0, 8):
                                for kind in ('real', 'self', 'uninformative'):
                                    marks = ep.simulate(s.z, s.p, r, s.c, simulator=sim, position=position,
                                            run_length=run_length, confirming=k, identity=(u['id'], ri),
                                            null=None if kind == 'real' else kind)
                                    if marks is None:
                                        skipped += 1
                                        continue
                                    rng_seed = stable_seed((u['id'], ri, sim, position, k))
                                    for arm in NEW_ARMS:
                                        p, loc, declared = update(s, marks, arm, configs[arm], rng_seed)
                                        checked += 1
                                        values = ep.measures(s, r, marks, p, loc)
                                        record(s, arm, kind, position, sim, run_length, k, values)
                                        raw.write(json.dumps(dict(unit=u['id'], rec=u['rec'], region=ri,
                                            arm=arm, null=kind, position=position, simulator=sim,
                                            run=run_length, confirming=k, declared=len(declared), **values)) + '\n')
                                    if kind == 'real' and run_length == 1 and k == 0 and position < 3:
                                        # P7's cached feature-graph curve uses random seed zero.
                                        for arm in (*NEW_ARMS, 'local', 'bias', 'mark'):
                                            for gain in CURVE_GAINS:
                                                p, loc, _ = update(s, marks, arm, dict(configs[arm], gain=gain))
                                                checked += 1
                                                values = ep.measures(s, r, marks, p, loc)
                                                for key in ('M3i', 'M4_away'):
                                                    ep.add_measure(curves[f'{arm}:{gain}'], u['rec'], key, values[key], values)
                                    if kind == 'real':
                                        dr = donor_regs[ri % len(donor_regs)] if donor_regs else np.ones(len(donor.p), bool)
                                        dm = ep.simulate(donor.z, donor.p, dr, donor.c, simulator=sim,
                                             position=position, run_length=run_length, confirming=k,
                                             identity=(donor.unit['id'], ri))
                                        if dm:
                                            shuffled = ep.transfer_marks(dm, donor.z, s.z)
                                            for arm in NEW_ARMS:
                                                p, loc, _ = update(s, shuffled, arm, configs[arm]); checked += 1
                                                record(s, arm, 'shuffle', position, sim, run_length, k,
                                                       ep.measures(s, r, shuffled, p, loc))
                for sim in (0, 1):
                    marks = ep.simulate(s.z, s.p, r, s.c, simulator=sim, position=1,
                             run_length=1, confirming=8, identity=u['id'], null='confirming')
                    for arm in NEW_ARMS:
                        p, loc, _ = update(s, marks, arm, configs[arm], num); checked += 1
                        record(s, arm, 'confirming-fail', 1, sim, 1, 8, ep.measures(s, r, marks, p, loc))
            if num % 10 == 0:
                log('panel_progress', seed=seed, units=num, regions=region_count, checked=checked)
    log('summarizing', seed=seed, updates=checked)
    result = dict(summary=summarize(summary), curves=summarize(curves), cohorts=cohort,
                  regions=region_count, failing_units=fail_count, nonfailing_units=nofail_count,
                  skipped_mark_draws=skipped, invariant_checks=checked)
    old = json.loads((OUT / f'P7-s{seed}.json').read_text())
    for group, values in old['summary'].items():
        if group.split(':')[0] in (*REFERENCES, 'learned0', 'learned1'):
            result['summary'][group] = values
    result['curves'].update({k: v for k, v in old['curves'].items() if k.split(':')[0] == 'graph'})
    result['reference_provenance'] = dict(path=str(OUT / f'P7-s{seed}.json'),
                                        sha256=digest(OUT / f'P7-s{seed}.json'))
    dump(DEST / f'seed-{seed}.json', result)
    return result


def run(seed):
    frozen()
    assert (DEST / 'inputs.json').exists(), 'Run prepare before seed measurements'
    path = DEST / f'invariants-s{seed}.json'
    dump(path, dict(complete=False, breaches=0))
    try:
        fixture = fixtures()
        dump(DEST / f'fixtures-s{seed}.json', fixture)
        sources = {PREV / 'e2-maps' / f's{seed}' / '009001' / (u['sha'] + '.json')
                   for u in units() if u['cohort'] in ('panel', 'val_hard')}
        before = {str(p): digest(p) for p in sorted(sources)}
        tuned = tune(seed)
        result = evaluate(seed, tuned['configs'])
        after = {str(p): digest(p) for p in sorted(sources)}
        assert before == after, 'Source decoded maps changed during P7m'
        dump(path, dict(complete=True, breaches=0, fixtures=fixture,
             episode_updates_checked=result['invariant_checks'],
             tuning_updates_checked=sum(d['invariant_checks'] for d in tuned['diagnostics'].values()),
             decoded_map_files=len(sources), decoded_maps_unchanged=True,
             decoded_map_manifest_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()))
        log('seed_complete', seed=seed, updates=result['invariant_checks'])
    except AssertionError as exc:
        dump(path, dict(complete=False, breaches=1, failure=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'seed', 'report'))
    parser.add_argument('--seed', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'seed':
        if args.seed is None:
            parser.error('seed requires --seed')
        run(args.seed)
    else:
        from .p7m_report import build
        build()


if __name__ == '__main__':
    main()
