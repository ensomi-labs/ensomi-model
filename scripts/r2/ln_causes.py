"""Mask, target and training-source probes for long-note continuation.

Read an existing fit-only cache and the separation.py census/panel. Run prepare,
census, fr, then analyze. --masks-off opts into playability lesions; grammar is
always enforced. Training sources are explicitly labelled and never pooled with
dev. Outputs belong in an ignored artifact directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from separation import (code_counts, disk_guard, grid_of, jsonable, ln_statistics,
                        mask_statistics, source_objects, write_json)
from ln_analysis import METRICS, interval, pair_spread, scaled, strata
from ensomi_model.r2.bakeoff_panel import load_model
from ensomi_model.r2.cache import load_chart, objects_from_decisions
from ensomi_model.r2.common import action_index
from ensomi_model.r2.data import chart_from_cache
from ensomi_model.r2.splits import role_of
from ensomi_model.r2.state import replay_decisions
from ensomi_model.r2.theta import decode
from ensomi_model.system.arrange import arrange
from ensomi_model.system.controls import Playability
from ensomi_model.system.prior import Prior, kappa_targets, rows_features


def source_levels(actions, K, start=0):
    held, heads, _ = decode(actions[:K])
    return np.array([held[start:].mean(), heads[start:].sum(1).mean(),
                     (held[start:] | heads[start:]).all(1).mean()])


def source_chart(row, cache):
    role = row['role']
    if role not in ('fit_train', 'fit_dev') or role_of(row['group_id']) != role:
        raise ValueError('Only explicitly labelled fit_train and fit_dev sources are allowed')
    return load_chart(Path(cache) / 'charts' / row['file'])


def prepare(args):
    census = pd.read_csv(Path(args.stage1) / 'census.csv')
    panel = json.loads((Path(args.stage1) / 'panel.json').read_text())['charts']
    dev = [dict(r, seen_in_training=False) for r in panel if r['ln_class'] != 'normal']
    train = census[(census.role == 'fit_train') & (census.ln_class == 'inverse')].copy()
    train['order'] = [hashlib.sha256(f'20261108-stage1b:{s}'.encode()).hexdigest() for s in train.sha256]
    train = train.sort_values('order')
    selected, groups, audio = [], set(), set()
    for band, quota in ((2, 1), (3, 3), (4, 4), (5, 4)):
        n = 0
        for r in train[train.band == band].to_dict('records'):
            if r['group_id'] in groups or r['audio_sha256'] in audio:
                continue
            groups.add(r['group_id']); audio.add(r['audio_sha256'])
            selected.append(dict(r, seen_in_training=True)); n += 1
            if n == quota:
                break
    if len(selected) != 12:
        raise ValueError('Training quotas unavailable; amend panel selection before inference')
    write_json(Path(args.out) / 'panel.json', dict(charts=dev + selected))
    print(json.dumps(dict(dev=len(dev), train=len(selected))), flush=True)


def mask_census(chart, config):
    """Replay actual decoder rules, including human decisions they would reject."""
    d = chart.derived()
    guard = Playability(config)
    counts = dict(reattack=0, quad=0, jack=0, union=0, ln_birth_rows=0, ln_birth_heads=0)
    for k in range(chart.K):
        i = action_index(chart.actions[k])
        flags = {name: bool(v[i]) for name, v in guard.rejections(chart, k, d.held[k]).items()}
        for name, value in flags.items():
            counts[name] += value
        bad = any(flags.values())
        counts['union'] += bad
        counts['ln_birth_rows'] += bad and bool(d.ln_head[k].any())
        counts['ln_birth_heads'] += bad * int(d.ln_head[k].sum())
        guard.observe(d.attack[k])
    fast = mask_statistics(chart.head_ms, chart.actions, config)
    assert all(counts[name] == fast['mask_' + name] for name in ('reattack', 'quad', 'jack'))
    return dict(counts, rows=chart.K, heads=int(d.attack[:chart.K].sum()),
                ln_heads=int(d.ln_head[:chart.K].sum()), max_jack_run=fast['max_jack_run'])


def census_job(spec):
    row, cache, config = spec
    dec = source_chart(row, cache)
    return dict(sha=row['sha256'], group=row['group_id'], role=row['role'],
                source_class=row['ln_class'], band=row['band'],
                **mask_census(chart_from_cache(dec), config))


def census(args):
    table = pd.read_csv(Path(args.stage1) / 'census.csv')
    table = table[table.ln_class.isin(['ln_heavy', 'inverse'])]
    masks = json.loads((Path(args.model_dir) / 'calibration.json').read_text())['masks']
    specs = [(r, args.cache, masks[str(r['band'])]) for r in table.to_dict('records')]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        records = list(pool.map(census_job, specs, chunksize=4))
    disk_guard(args.out)
    result = pd.DataFrame(records)
    result.to_csv(Path(args.out) / 'mask-census.csv', index=False)
    summaries = []
    for (role, cls, band), group in result.groupby(['role', 'source_class', 'band']):
        row = dict(role=role, source_class=cls, band=band, charts=len(group), songs=group.group.nunique(),
                   rows=int(group.rows.sum()), ln_heads=int(group.ln_heads.sum()))
        for rule in ('reattack', 'quad', 'jack', 'union', 'ln_birth_rows', 'ln_birth_heads'):
            row[rule] = int(group[rule].sum())
            row[rule + '_charts'] = int((group[rule] > 0).sum())
            row[rule + '_rate'] = row[rule] / row['rows' if rule != 'ln_birth_heads' else 'ln_heads']
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(Path(args.out) / 'mask-summary.csv', index=False)
    print(json.dumps(dict(charts=len(result), rows=int(result.rows.sum()))), flush=True)


_WORKER = None


def initialize(config):
    global _WORKER
    torch.set_num_threads(1)
    models = {n: load_model(config['checkpoint_' + str(n)]) for n in (48, 56, 64)}
    _WORKER = config, models


def fr_job(row):
    config, models = _WORKER
    path = Path(config['out']) / 'fr' / (row['sha256'] + '.jsonl')
    done = path.with_suffix('.done')
    if done.exists():
        return dict(sha=row['sha256'], cached=True)
    dec = source_chart(row, config['cache'])
    grid, start = grid_of(dec), dec.K // 3
    cut = dec.head_ms[start]
    root = Path(config['model_dir'])
    masks = json.loads((root / 'calibration.json').read_text())['masks'][str(row['band'])]
    target = kappa_targets(Prior.load(root / 'prior.json').predict(
        rows_features(dec.head_ms, dec.song_ms, grid), int(row['band'])))
    oracle = source_levels(dec.actions, dec.K)
    base = dict(sha=row['sha256'], group=row['group_id'], role=row['role'], band=row['band'],
                source_class=row['ln_class'], seen_in_training=row['seen_in_training'], mode='prefix', start=start)
    arms = [('baseline', 0, False, target), ('baseline', 3, False, target), ('oracle', 3, False, oracle)]
    if config['masks_off']:
        arms += [('masks_off', 0, True, target), ('masks_off', 3, True, target),
                 ('oracle_masks_off', 3, True, oracle)]
    previous = Path(config['stage1']) / 'fr' / path.name
    reused = {}
    if row['role'] == 'fit_dev':
        for line in previous.read_text().splitlines():
            r = json.loads(line)
            if r['mode'] == 'prefix' and r['arm'] == 'baseline':
                reused[r['checkpoint'], r['kappa'], r['seed']] = r
    human = ln_statistics(source_objects(dec), grid, cut)
    human.update(code_counts(dec.actions, start=start, stop=dec.K))
    human.update(zip(('held_level', 'heads_level', 'busy_level'), source_levels(dec.actions, dec.K, start)))
    started, runs = time.monotonic(), 0
    with torch.inference_mode(), path.open('w') as log:
        log.write(json.dumps(jsonable(dict(base, arm='human', **human))) + '\n')
        for checkpoint, model in models.items():
            for arm, kappa, masks_off, levels in arms:
                for seed in (954, 955, 956):
                    old = reused.get((checkpoint, kappa, seed)) if arm == 'baseline' else None
                    if old:
                        result = dict(old, **{k: v for k, v in base.items() if k not in old})
                        result['reused_stage1'] = True
                        # Stage 1 retained code counts, enough to recover suffix levels.
                        held_n = sum(old[f'held_{c}'] for c in range(5))
                        heads_n = old['free_1'] + old['free_2'] + old['held_3'] + old['held_4']
                        result.update(held_level=held_n / (4 * (dec.K - start)),
                                      heads_level=heads_n / (dec.K - start))
                    else:
                        t0 = time.monotonic()
                        a, g, trace = arrange(model, dec.head_ms, dec.song_ms, grid, seed=seed,
                            target=levels, kappa=kappa, masks=masks, masks_off=masks_off,
                            prefix_actions=dec.actions[:start], prefix_gap=dec.gap_release_ms[:start])
                        final, _ = replay_decisions(dec.head_ms, dec.song_ms, a, g)
                        assert final.finalized and not np.any(final.held)
                        objects = objects_from_decisions(dec.head_ms, a, g, dec.song_ms)
                        result = ln_statistics(objects, grid, cut)
                        result.update(code_counts(a, start=start, stop=dec.K))
                        result.update(mask_statistics(dec.head_ms, a, masks, start))
                        result.update(zip(('held_level', 'heads_level', 'busy_level'), source_levels(a, dec.K, start)))
                        result.update(base, arm=arm, checkpoint=checkpoint, kappa=kappa, seed=seed,
                            target=levels, legal=True, fallbacks=trace['fallback_count'],
                            seconds=time.monotonic() - t0, decisions=dec.K + 1 - start, reused_stage1=False)
                    disk_guard(path.parent)
                    log.write(json.dumps(jsonable(result), allow_nan=False) + '\n'); log.flush()
                    runs += 1
    done.write_text('complete\n')
    return dict(sha=row['sha256'], role=row['role'], runs=runs, seconds=time.monotonic() - started)


def fr(args):
    (Path(args.out) / 'fr').mkdir(exist_ok=True)
    rows = json.loads((Path(args.out) / 'panel.json').read_text())['charts']
    if args.limit:
        rows = rows[:args.limit]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(vars(args),)) as pool:
        for task in as_completed([pool.submit(fr_job, row) for row in rows]):
            print(json.dumps(task.result()), flush=True)


def floors(table, metric):
    s, sn = pair_spread(table[table.checkpoint == 56].pivot(index='sha', columns='seed', values=metric))
    c, cn = pair_spread(table.groupby(['sha', 'checkpoint'])[metric].mean().unstack('checkpoint'))
    return dict(sampling_rms=s, sampling_pairs=sn, checkpoint_rms=c, checkpoint_pairs=cn)


def exposure(args):
    """Reconstruct natural draw membership from the checkpoint's exact RNG state."""
    from ensomi_model.r2.conditions import old_start
    payload = torch.load(args.checkpoint_64, map_location='cpu', weights_only=False)
    cfg, state = payload['config'], payload['state']
    if cfg['phase'] != 'natural' or cfg['draw']['selection'] != 'natural':
        raise ValueError('Exposure reconstruction is defined only for natural phase N')
    index = pd.read_parquet(Path(args.cache) / 'index.parquet')
    index = index[index.role == 'fit_train']
    groups = sorted(index.group_id.unique())
    members = {g: sorted(s) for g, s in index.groupby('group_id').sha256}
    sizes = dict(zip(index.sha256, index.K))
    panel = json.loads((Path(args.out) / 'panel.json').read_text())['charts']
    coverage = {r['sha256']: np.zeros(sizes[r['sha256']], int) for r in panel if r['role'] == 'fit_train'}
    counts = dict.fromkeys(coverage, 0)
    first_exposure = dict.fromkeys(coverage, None)
    checkpoints = (48_000_198, 56_000_574, state['exposures'])
    by_checkpoint = {x: {sha: np.zeros_like(v) for sha, v in coverage.items()} for x in checkpoints}
    windows_by_checkpoint = {x: dict.fromkeys(coverage, 0) for x in checkpoints}
    skips = set(state['skip'])
    rng = np.random.default_rng(cfg['seed_draws'])
    heads = 0
    for j in range(state['windows']):
        group = groups[int(rng.integers(0, len(groups)))]
        charts = members[group]
        sha = charts[int(rng.integers(0, len(charts)))]
        K = sizes[sha]
        start = old_start(rng, K, cfg['window'])
        stop = min(start + cfg['window'], K)
        if j not in skips:
            heads += stop - start
            if sha in coverage:
                counts[sha] += 1
                coverage[sha][start:stop] += 1
                if first_exposure[sha] is None:
                    first_exposure[sha] = heads - (stop - start)
                for x in checkpoints:
                    if heads <= x:
                        by_checkpoint[x][sha][start:stop] += 1
                        windows_by_checkpoint[x][sha] += 1
    rng_matches = rng.bit_generator.state == payload['draw_rng']
    exposure_matches = heads == state['exposures']
    write_json(Path(args.out) / 'training-exposure.json', dict(
        windows=state['windows'], exposures=heads, expected_exposures=state['exposures'],
        rng_matches=rng_matches, exposure_matches=exposure_matches, skip=sorted(skips),
        charts=[dict(sha=sha, windows=counts[sha], scored_head_exposures=int(v.sum()),
                     first_exposure=first_exposure[sha],
                     unique_heads=int((v > 0).sum()), source_heads=len(v),
                     suffix_coverage=float((v[len(v) // 3:] > 0).mean()),
                     checkpoints={x: dict(windows=windows_by_checkpoint[x][sha],
                         scored_head_exposures=int(by_checkpoint[x][sha].sum()),
                         suffix_coverage=float((by_checkpoint[x][sha][len(v) // 3:] > 0).mean()))
                         for x in checkpoints}) for sha, v in coverage.items()]))
    if not rng_matches or not exposure_matches:
        raise ValueError('Draw reconstruction mismatch; cannot certify seen-in-training exposure')
    print(json.dumps(dict(exposure_matches=True, rng_matches=True, selected_windows=counts)), flush=True)


def analyze(args):
    root = Path(args.out)
    records = []
    for p in sorted((root / 'fr').glob('*.jsonl')):
        if not p.with_suffix('.done').exists():
            raise ValueError(f'Incomplete chart {p.name}')
        records += [json.loads(x) for x in p.read_text().splitlines()]
    table = pd.DataFrame(records)
    expected = len(json.loads((root / 'panel.json').read_text())['charts'])
    assert table.sha.nunique() == expected
    rows = []
    for role, by_role in table.groupby('role'):
        for cls, band, sub in strata(by_role, 'source_class'):
            human = sub[sub.arm == 'human'].set_index('sha')
            for (arm, kappa), arm_rows in sub[sub.arm != 'human'].groupby(['arm', 'kappa']):
                refs = ['human'] + (['baseline'] if arm != 'baseline' else [])
                if arm == 'oracle_masks_off':
                    refs += ['oracle', 'masks_off']
                for metric in METRICS + ['held_level', 'heads_level', 'busy_level']:
                    current = arm_rows[arm_rows.checkpoint == 56].groupby('sha')[metric].mean()
                    ff = floors(arm_rows, metric)
                    for ref in refs:
                        fref = {}
                        if ref == 'human':
                            target = human[metric]
                        else:
                            prior = sub[(sub.arm == ref) & (sub.kappa == kappa)]
                            target = prior[prior.checkpoint == 56].groupby('sha')[metric].mean()
                            fref = floors(prior, metric)
                        worst = dict(ff)
                        for key in ('sampling_rms', 'checkpoint_rms'):
                            worst[key] = max(ff[key], fref.get(key, 0))
                        paired = pd.concat([current.rename('generated'), target.rename('source')], axis=1)
                        paired = paired[np.isfinite(paired).all(axis=1)]
                        result = scaled(interval(paired.generated - paired.source), worst)
                        rows.append(dict(role=role, seen_in_training=role == 'fit_train', source_class=cls,
                            band=band, arm=arm, kappa=int(kappa), reference=ref, metric=metric,
                            generated=float(paired.generated.mean()), source=float(paired.source.mean()), **result,
                            **{'arm_' + k: v for k, v in ff.items()},
                            **{'reference_' + k: v for k, v in fref.items()}))
    disk_guard(root)
    pd.DataFrame(rows).to_csv(root / 'effects.csv', index=False)
    transitions = table[table.checkpoint == 56].groupby(
        ['role', 'source_class', 'arm', 'kappa', 'ln_class']).size().rename('n').reset_index()
    transitions.to_csv(root / 'class-transitions.csv', index=False)
    generated = table[table.arm != 'human']
    write_json(root / 'integrity.json', dict(charts=expected, runs=len(generated),
        reused=int(generated.reused_stage1.sum()), illegal=int((~generated.legal.astype(bool)).sum()),
        fallbacks=generated.groupby(['arm', 'kappa']).fallbacks.sum().to_dict(),
        seconds=float(generated.loc[~generated.reused_stage1.astype(bool), 'seconds'].sum())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'census', 'fr', 'analyze', 'exposure'])
    for name in ('cache', 'stage1', 'out', 'model-dir'):
        p.add_argument('--' + name, required=True)
    for n in (48, 56, 64):
        p.add_argument('--checkpoint-' + str(n))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--masks-off', action='store_true')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.command == 'fr' and any(getattr(args, 'checkpoint_' + str(n)) is None for n in (48, 56, 64)):
        p.error('fr requires all three checkpoint paths')
    if args.command == 'exposure' and args.checkpoint_64 is None:
        p.error('exposure requires the full 64M training checkpoint')
    disk_guard(args.out)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
