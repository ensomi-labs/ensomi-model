"""Whole-chart likelihood and own-history style measures for release models.

Uses a fit_dev panel from separation.py. NLL is summed over every head gap,
release opportunity and EOS, never divided by representation-specific decision
counts. Duration hazards use complete generated holds born in the evaluated
suffix: releases in an age bin divided by holds reaching its lower edge.
This is marginal calibration against source durations, not conditional forecast
calibration on human histories. No corpus cache is rebuilt.
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
import torch

from separation import disk_guard, grid_of, jsonable, ln_statistics, source_objects, write_json
from ln_probe import heldout_chart
from ensomi_model.osu_core.hitobjects import ManiaHitObjectKind
from ensomi_model.r2.cache import objects_from_decisions
from ensomi_model.r2.data import chart_from_cache
from ensomi_model.r2.model import R2Model, R2Config
from ensomi_model.r2.release_diagnostics import opportunity_profile, leak_counts
from ensomi_model.r2.release_steps import ReleaseStepModel, continue_release_steps
from ensomi_model.r2.state import replay_decisions
from ensomi_model.r2.theta import decode
from ensomi_model.system.arrange import arrange
from ensomi_model.system.prior import Prior, kappa_targets, rows_features

AGE_EDGES = np.array([0., 60., 125., 250., 500., 1000., 2000., np.inf])


def duration_hazards(objects, start_ms):
    lengths = np.array([o.end_time_ms - o.start_time_ms for o in objects
                        if o.kind is ManiaHitObjectKind.HOLD and o.start_time_ms >= start_ms])
    return [dict(age_lo_ms=float(lo), age_hi_ms=float(hi) if np.isfinite(hi) else None,
                 risk=int((lengths >= lo).sum()), releases=int(((lengths >= lo) & (lengths < hi)).sum()))
            for lo, hi in zip(AGE_EDGES, AGE_EDGES[1:])]


_WORKER = None


def initialize(config):
    global _WORKER
    torch.set_num_threads(1)
    payload = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    representation = payload.get('representation', 'r2')
    if representation not in ('r2', 'visible', 'feature_blind'):
        raise ValueError(f'Unknown checkpoint representation: {representation}')
    cfg = R2Config(**payload['model_config'])
    model = (R2Model(cfg) if representation == 'r2' else
             ReleaseStepModel(cfg, blind=representation == 'feature_blind'))
    model.load_state_dict(payload['model']); model.eval()
    _WORKER = config, model, representation


def job(row):
    args, model, representation = _WORKER
    root = Path(args['out'])
    path = root / (row['sha256'] + '.json')
    if path.exists():
        return dict(sha=row['sha256'], cached=True)
    dec = heldout_chart(row, args['cache'])
    chart, grid = chart_from_cache(dec), grid_of(dec)
    opportunities = opportunity_profile(dec.head_ms, grid)
    system = Path(args['model_dir'])
    masks = json.loads((system / 'calibration.json').read_text())['masks'][str(row['band'])]
    target = kappa_targets(Prior.load(system / 'prior.json').predict(
        rows_features(dec.head_ms, dec.song_ms, grid), int(row['band'])))
    started = time.monotonic()
    nll, code, release, eos = 0., 0., 0., 0.
    with torch.inference_mode():
        for start in range(0, dec.K + 1, 256):
            out = model.window(chart, start, min(dec.K + 1, start + 256))
            nll -= float(out.total.double().sum())
            code -= float(out.action.double().sum())
            release -= float(out.release.double().sum())
            eos -= float(out.total[out.eos].double().sum())
    tf_seconds = time.monotonic() - started
    runs, human = [], []
    original = source_objects(dec)
    sampler = arrange if representation == 'r2' else continue_release_steps
    if not args['nll_only']:
        for mode, cut in [('bos', 0), ('prefix', dec.K // 3)]:
            start_ms = dec.head_ms[cut]
            human.append(dict(mode=mode, **ln_statistics(original, grid, start_ms),
                              leaks=leak_counts(chart, opportunities, cut),
                              duration_hazards=duration_hazards(original, start_ms)))
            for kappa in (0, 3):
                for seed in (954, 955, 956):
                    t0 = time.monotonic()
                    actions, gap, trace = sampler(model, dec.head_ms, dec.song_ms, grid,
                        seed=seed, target=target, kappa=kappa, masks=masks,
                        prefix_actions=dec.actions[:cut], prefix_gap=dec.gap_release_ms[:cut])
                    final, _ = replay_decisions(dec.head_ms, dec.song_ms, actions, gap)
                    assert final.finalized and not np.any(final.held)
                    objects = objects_from_decisions(dec.head_ms, actions, gap, dec.song_ms)
                    generated = chart.with_decisions(actions, gap)
                    held, _, _ = decode(actions)
                    tail = gap[dec.K][held[dec.K]]
                    runs.append(dict(mode=mode, kappa=kappa, seed=seed,
                        **ln_statistics(objects, grid, start_ms),
                        duration_hazards=duration_hazards(objects, start_ms), legal=True,
                        eos_releases=int(held[dec.K].sum()),
                        eos_release_offset_from_last_head_ms=(tail - dec.head_ms[-1]).tolist(),
                        fallbacks=trace['fallback_count'], forced_releases=trace.get('forced_releases'),
                        forced_full_occupancy=trace.get('forced_full_occupancy'),
                        forced_eos=trace.get('forced_eos'), leaks=leak_counts(generated, opportunities, cut),
                        conditional_hazards=trace.get('hazards'), seconds=time.monotonic() - t0))
    write_json(path, dict(sha=row['sha256'], group=row['group_id'], role=row['role'],
        source_class=row['ln_class'], band=row['band'], representation=representation,
        checkpoint=args['checkpoint'], checkpoint_sha256=args['checkpoint_sha256'],
        whole_chart_nll=nll, head_allocation_nll=code, release_nll=release, eos_nll=eos,
        objects=dec.n_objects if hasattr(dec, 'n_objects') else len(original),
        source_head_rows=dec.K, tf_seconds=tf_seconds, human=human, runs=runs))
    return dict(sha=row['sha256'], seconds=time.monotonic() - started, whole_chart_nll=nll, runs=len(runs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'panel', 'model-dir', 'checkpoint', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--nll-only', action='store_true')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    disk_guard(root)
    config = vars(args).copy()
    config['schema_version'] = 2
    config['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    rows = json.loads(Path(args.panel).read_text())['charts']
    if args.limit:
        rows = rows[:args.limit]
    manifest = root / 'evaluation.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Evaluation inputs changed; use a fresh output directory')
    write_json(manifest, config)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(config,)) as pool:
        for task in as_completed([pool.submit(job, row) for row in rows]):
            print(json.dumps(task.result()), flush=True)
    disk_guard(root)
    (root / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    main()
