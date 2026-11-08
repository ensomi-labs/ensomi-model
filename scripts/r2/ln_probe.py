"""Teacher-forced and free-running probes of R2 long-note decisions.

Uses existing fit caches. Pointer interventions are inference lesions, not
trained alternatives. Conditional factor calibration keeps the source action
and earlier source release placements fixed; it is not free-running quality.

The shared --out directory must contain census.csv and panel.json from
separation.py. tf optionally adds --manifest natural windows; fr and
diagnostics require --model-dir for the released masks and prior. Every probe
uses --checkpoint-48, --checkpoint-56 and --checkpoint-64, one CPU thread per
worker, and writes compact per-chart results. Completed charts are skipped on
restart; use a new output directory when changing any input or probe recipe.
joint-pointer samples complete release placements conditional on human codes.
outliers is explicitly a post-score diagnostic. No command trains weights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch

from separation import (code_counts, disk_guard, grid_of, jsonable, ln_statistics,
                        mask_statistics, source_objects, write_json)
from ensomi_model.r2.bakeoff_panel import load_model
from ensomi_model.r2.cache import load_chart, objects_from_decisions
from ensomi_model.r2.common import ACTIONS, action_index
from ensomi_model.r2.data import chart_from_cache
from ensomi_model.r2.features import (decision_factors, lane_query, gap_release_lanes,
                                     row_placements, make_factor)
from ensomi_model.r2.state import action_support_mask, replay_decisions
from ensomi_model.r2.splits import role_of


def heldout_chart(row, cache):
    if row['role'] != 'fit_dev' or role_of(row['group_id']) != 'fit_dev':
        raise ValueError('LN probes require the unsealed fit_dev split')
    return load_chart(Path(cache) / 'charts' / row['file'])


def panel_indices(K):
    starts = (0, K // 3, 2 * K // 3, max(0, K + 1 - 256))
    return np.unique(np.concatenate([np.arange(s, min(K + 1, s + 256)) for s in starts]))


def factors_for(chart, ks):
    lanes = lane_query(chart, ks).astype(np.float32)
    return [f for i, k in enumerate(ks) for orientation in (0, 1)
            for f in decision_factors(chart, int(k), i, chart.actions[k], chart.gap[k], lanes[i], orientation)]


def lesion_inputs(z, factors, chart, lesion):
    """Only the pointer's input view changes; actions, supports and histories do not."""
    altered = []
    for f in factors:
        if lesion == 'same_birth':
            rel = f.relations.copy()
            rel[:, [4, 11, 18, 25]] = 0
            altered.append(replace(f, relations=rel))
        elif lesion == 'placement_relation':
            altered.append(replace(f, relations=np.zeros_like(f.relations)))
        elif lesion == 'hold_age':
            lane, cand = f.lane_feats.copy(), f.cand_feats.copy()
            lane[1:5], cand[:, 8:12] = 0, 0
            altered.append(replace(f, lane_feats=lane, cand_feats=cand))
        elif lesion == 'row_heads':
            held = chart.derived().held[f.k]
            held = held[::-1] if f.orientation else held
            codes = f.codes.copy()
            codes[~held] = 0
            codes[held & (codes >= 3)] = 2
            altered.append(replace(f, codes=codes))
        else:
            altered.append(f)
    if lesion == 'other_hand':
        # Each factor gets its own copy because its releasing hand can differ.
        z = z[[f.owner for f in factors]].clone()
        for i, f in enumerate(factors):
            z[i, 1 if f.lane < 2 else 0] = 0
        altered = [replace(f, owner=i) for i, f in enumerate(altered)]
    return z, altered


def release_probabilities(model, z, factors, chart, size, lesion='none'):
    if not factors:
        return np.zeros(size), np.zeros((size, 2)), []
    zz, changed = lesion_inputs(z, factors, chart, lesion)
    log_target, log_pairs = model.factor_pair_log_probs(zz, changed, chart, ())
    target = log_target.double().numpy()
    sums = np.zeros((size, 2))
    np.add.at(sums, ([f.owner for f in factors], [f.orientation for f in factors]), target)
    mixture = np.logaddexp(sums[:, 0], sums[:, 1]) - np.log(2)
    probabilities = np.split(log_pairs.double().exp().numpy(), np.cumsum([len(f.times) for f in factors])[:-1])
    return mixture, sums, probabilities


def factor_events(chart, f):
    """Binary target events under one directed, gold-conditioned release factor."""
    d = chart.derived()
    birth = d.start[f.k, f.lane]
    short = f.times - birth <= 60
    other = np.flatnonzero(np.any(d.attack[:chart.K, np.arange(4) != f.lane], axis=1))
    times = chart.head_ms[other]
    near = np.searchsorted(times, f.times + 40, side='right') > np.searchsorted(times, f.times + 1, side='left')
    relations = f.relations.reshape(len(f.times), 4, 7)
    same_birth = (relations[:, :, 4] > .5) & (relations[:, :, 5] > .5)
    together = (same_birth & (abs(relations[:, :, 0] * 1000) <= 5.0001)).any(1)
    events = dict(short=short, near=near)
    if same_birth.any():
        events['together'] = together
    return events


def tf_chart(model, chart, ks, lesions=False):
    """Return compact per-decision, held-lane and conditional-factor tables."""
    with torch.inference_mode():
        z = model.window_hands(chart, ks)
        d = chart.derived()
        held = d.held[ks]
        support = np.stack([action_support_mask(h, k == chart.K) for h, k in zip(held, ks)])
        logp = model.action_log_probs(z, support)
        target = np.array([action_index(chart.actions[k]) for k in ks])
        code_nll = -logp[torch.arange(len(ks)), target].double().numpy()
        probabilities = logp.double().exp().numpy()
        marginals = np.stack([probabilities @ (ACTIONS[:, lane, None] == np.arange(5))
                              for lane in range(4)], axis=1)
        factors = factors_for(chart, ks)
        mix, orientations, factor_p = release_probabilities(model, z, factors, chart, len(ks))
        versions = {'none': (mix, orientations, factor_p)}
        if lesions:
            for lesion in ('same_birth', 'hold_age', 'row_heads', 'other_hand'):
                versions[lesion] = release_probabilities(model, z, factors, chart, len(ks), lesion)
        factor_n = np.zeros(len(ks), dtype=int)
        for f in factors:
            factor_n[f.owner] += int(f.orientation == 0)
        decisions, hazards, events = [], [], []
        for i, k in enumerate(ks):
            t = chart.time(k)
            gap = t - chart.time(k - 1) if k else np.nan
            heads = d.attack[k].sum()
            r = dict(k=int(k), eos=k == chart.K, gap_ms=gap, heads=int(heads),
                     code_nll=code_nll[i], pointer_nll=-mix[i], gap_releases=int(factor_n[i]),
                     held=int(held[i].sum()), release_observed=int(np.sum(held[i] & (chart.actions[k] != 0))),
                     release_expected=float(np.sum((1 - marginals[i, :, 0]) * held[i])),
                     birth_free_observed=int(np.sum(~held[i] & (chart.actions[k] == 2))),
                     birth_free_expected=float(np.sum(marginals[i, :, 2] * ~held[i])),
                     birth_held_observed=int(np.sum(held[i] & (chart.actions[k] == 4))),
                     birth_held_expected=float(np.sum(marginals[i, :, 4] * held[i])),
                     forward_nll=-orientations[i, 0], reverse_nll=-orientations[i, 1],
                     orientation_abs_log_ratio=abs(orientations[i, 0] - orientations[i, 1]))
            for lesion, (lm, _, _) in versions.items():
                if lesion != 'none':
                    r[lesion + '_pointer_nll'] = -lm[i]
            for state, lane_mask, codes in (('held', held[i], range(5)), ('free', ~held[i], range(3))):
                for code in codes:
                    r[f'{state}_code{code}_observed'] = int(np.sum(lane_mask & (chart.actions[k] == code)))
                    r[f'{state}_code{code}_expected'] = float(np.sum(marginals[i, :, code] * lane_mask))
            decisions.append(r)
            for lane in np.flatnonzero(held[i]):
                hazards.append(dict(k=int(k), lane=int(lane), eos=k == chart.K,
                    age_ms=t - d.start[k, lane],
                    previous_age_ms=chart.time(k - 1) - d.start[k, lane] if k else np.nan,
                    gap_ms=gap, observed=int(chart.actions[k, lane] != 0), predicted=1 - marginals[i, lane, 0],
                    code=int(chart.actions[k, lane])))
        by_lane = {}
        for fi, f in enumerate(factors):
            by_lane.setdefault((f.owner, f.lane), {})[f.orientation] = fi
            d = chart.derived()
            opposite = slice(2, 4) if f.lane < 2 else slice(0, 2)
            base = dict(k=f.k, lane=f.lane, orientation=f.orientation, eos=f.k == chart.K,
                        age_ms=chart.time(f.k) - d.start[f.k, f.lane],
                        gap_ms=chart.time(f.k) - chart.time(f.k - 1),
                        other_hand_held=int(d.held[f.k, opposite].sum()),
                        row_heads=int(d.attack[f.k].sum()), candidates=len(f.times))
            for event, mask in factor_events(chart, f).items():
                record = dict(base, event=event, observed=int(mask[f.target]), predicted=float(factor_p[fi][mask].sum()))
                for lesion, (_, _, ps) in versions.items():
                    if lesion != 'none':
                        record[lesion + '_predicted'] = float(ps[fi][mask].sum())
                events.append(record)
        tv = {}
        for (owner, lane), orders in by_lane.items():
            a, b = (factor_p[orders[o]] for o in (0, 1))
            tv.setdefault(owner, []).append(float(abs(a - b).sum() / 2))
        for owner, values in tv.items():
            decisions[owner]['conditional_orientation_tv'] = float(np.mean(values))
        return decisions, hazards, events


@contextmanager
def same_birth_lesion(model, enabled):
    def zero_birth(module, inputs):
        x = inputs[0].clone()
        x[..., [4, 11, 18, 25]] = 0
        return (x,)
    handle = model.pointer_relation.register_forward_pre_hook(zero_birth) if enabled else None
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


_WORKER = None


def initialize(config, mode):
    global _WORKER
    torch.set_num_threads(1)
    models = {n: load_model(config['checkpoint_' + str(n)]) for n in (48, 56, 64)}
    _WORKER = config, models


def tf_job(row):
    config, models = _WORKER
    out = Path(config['out']) / 'tf'
    dec = heldout_chart(row, config['cache'])
    chart = chart_from_cache(dec)
    ks = np.asarray(row.get('ks', panel_indices(dec.K)), dtype=int)
    sample = row.get('sample', 'panel')
    done = out / f'{sample}-{row["sha256"]}.done'
    if done.exists():
        return dict(sha=row['sha256'], sample=sample, cached=True)
    started = time.monotonic()
    for checkpoint, model in models.items():
        decisions, hazards, events = tf_chart(model, chart, ks, lesions=checkpoint == 56)
        for name, records in [('decisions', decisions), ('hazards', hazards), ('events', events)]:
            table = pd.DataFrame(records)
            for k, v in dict(sha=row['sha256'], group=row['group_id'], band=row['band'],
                             ln_class=row['ln_class'], checkpoint=checkpoint, sample=sample).items():
                table[k] = v
            disk_guard(out)
            table.to_parquet(out / f'{sample}-{row["sha256"]}-{checkpoint}-{name}.parquet', index=False)
    done.write_text('complete\n')
    return dict(sha=row['sha256'], sample=sample, decisions=len(ks), seconds=time.monotonic() - started)


def run_tf(args):
    out = Path(args.out)
    (out / 'tf').mkdir(exist_ok=True)
    panel = json.loads((out / 'panel.json').read_text())['charts']
    rows = [dict(r, sample='panel') for r in panel]
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())['natural']['windows']
        census = pd.read_csv(out / 'census.csv').set_index('sha256').to_dict('index')
        for sha in sorted({w['sha256'] for w in manifest}):
            windows = [w for w in manifest if w['sha256'] == sha]
            assert len(windows) == 1, 'Natural manifest unexpectedly repeats a chart'
            w = windows[0]
            rows.append(dict(census[sha], sha256=sha, ks=list(range(w['start'], w['stop'])), sample='natural'))
    if args.limit:
        rows = rows[:args.limit]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(vars(args), 'tf')) as pool:
        for task in as_completed([pool.submit(tf_job, row) for row in rows]):
            print(json.dumps(task.result()), flush=True)


def fr_job(row):
    from ensomi_model.system.arrange import arrange
    from ensomi_model.system.prior import Prior, kappa_targets, rows_features
    from ensomi_model.system.measures import tracking
    config, models = _WORKER
    out = Path(config['out']) / 'fr'
    path = out / (row['sha256'] + '.jsonl')
    done = path.with_suffix('.done')
    if done.exists():
        return dict(sha=row['sha256'], cached=True)
    dec = heldout_chart(row, config['cache'])
    grid = grid_of(dec)
    root = Path(config['model_dir'])
    cal = json.loads((root / 'calibration.json').read_text())
    prior = Prior.load(root / 'prior.json')
    target = kappa_targets(prior.predict(rows_features(dec.head_ms, dec.song_ms, grid), int(row['band'])))
    mask = cal['masks'][str(row['band'])]
    base = dict(sha=row['sha256'], group=row['group_id'], band=row['band'], source_class=row['ln_class'])
    original = source_objects(dec)
    started, runs = time.monotonic(), 0
    with torch.inference_mode(), path.open('w') as log:
        for mode, start in [('bos', 0), ('prefix', dec.K // 3)]:
            cut = dec.head_ms[start]
            human = ln_statistics(original, grid, cut)
            human.update(code_counts(dec.actions, start=start, stop=dec.K))
            log.write(json.dumps(jsonable(dict(base, mode=mode, arm='human', start=start, **human))) + '\n')
            for checkpoint, arm in [(c, 'baseline') for c in (48, 56, 64)] + [
                    (56, 'fixed_forward'), (56, 'same_birth')]:
                model = models[checkpoint]
                for kappa in (0, 3):
                    for seed in (954, 955, 956):
                        kwargs = dict(prefix_actions=dec.actions[:start], prefix_gap=dec.gap_release_ms[:start])
                        t0 = time.monotonic()
                        with same_birth_lesion(model, arm == 'same_birth'):
                            a, g, trace = arrange(model, dec.head_ms, dec.song_ms, grid,
                                seed=seed, target=target, kappa=kappa, masks=mask,
                                pointer_orientation=0 if arm == 'fixed_forward' else None, **kwargs)
                        elapsed = time.monotonic() - t0
                        final, _ = replay_decisions(dec.head_ms, dec.song_ms, a, g)
                        assert final.finalized and not np.any(final.held)
                        objects = objects_from_decisions(dec.head_ms, a, g, dec.song_ms)
                        result = ln_statistics(objects, grid, cut)
                        result.update(code_counts(a, start=start, stop=dec.K))
                        result.update(mask_statistics(dec.head_ms, a, mask, start))
                        result.update(tracking(a, dec.K, target))
                        result.update(base, mode=mode, arm=arm, start=start, checkpoint=checkpoint, kappa=kappa,
                                      seed=seed, seconds=elapsed, decisions=dec.K + 1 - start,
                                      target=target, legal=True, fallbacks=trace['fallback_count'],
                                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                        disk_guard(out)
                        log.write(json.dumps(jsonable(result), allow_nan=False) + '\n'); log.flush()
                        runs += 1
    done.write_text('complete\n')
    return dict(sha=row['sha256'], runs=runs, seconds=time.monotonic() - started)


def run_fr(args):
    out = Path(args.out)
    (out / 'fr').mkdir(exist_ok=True)
    rows = json.loads((out / 'panel.json').read_text())['charts']
    if args.limit:
        rows = rows[:args.limit]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(vars(args), 'fr')) as pool:
        for task in as_completed([pool.submit(fr_job, row) for row in rows]):
            print(json.dumps(task.result()), flush=True)


def joint_draws(model, chart, k, z, seed, samples=64, lesion='none'):
    """Sample Q(U | human state, human code) coherently in both lane orders.

    Candidate factors are batched across draws. Earlier releases come from the
    draw, so these are joint samples rather than gold-conditioned factors.
    """
    rng = np.random.default_rng(seed)
    orientation = rng.integers(0, 2, size=samples)
    uniform = rng.random((samples, 4))
    if lesion == 'fixed_forward':
        orientation[:] = 0
    d = chart.derived()
    held, codes = d.held[k], chart.actions[k]
    lane_feats = lane_query(chart, [k]).astype(np.float32)[0]
    release = np.full((samples, 4), np.nan)
    release[:, held & (codes == 1)] = chart.time(k)
    for order in (0, 1):
        indices = np.flatnonzero(orientation == order)
        if not len(indices):
            continue
        placed = [row_placements(chart, k, held, codes) for _ in indices]
        for slot, lane in enumerate(gap_release_lanes(held, codes, order)):
            factors = [make_factor(chart, k, 0, codes, lane, order, p, lane_feats) for p in placed]
            zz, factors = lesion_inputs(z, factors, chart, lesion)
            contexts = model.factor_contexts(zz, factors)
            scores, _, sizes = model.all_pair_scores(contexts, factors, chart, ())
            assert len(set(sizes)) == 1
            probability = torch.softmax(scores.reshape(len(factors), sizes[0]), -1).double().numpy()
            choice = (np.cumsum(probability, axis=1) < uniform[indices, slot, None]).sum(1)
            choice = np.minimum(choice, sizes[0] - 1)
            for i, (idx, f, target) in enumerate(zip(indices, factors, choice)):
                value = float(f.times[target])
                release[idx, lane] = value
                placed[i][lane] = (value, False)
    return release


def mc_events(chart, k, release):
    d = chart.derived()
    held, codes = d.held[k], chart.actions[k]
    gaps = np.flatnonzero(held & (codes >= 2))
    closed = np.flatnonzero(held & (codes != 0))
    pairs = [(a, b) for i, a in enumerate(closed) for b in closed[i + 1:]
             if d.birth[k, a] == d.birth[k, b]]
    out = {}
    if len(gaps):
        out['short'] = float((release[:, gaps] - d.start[k, gaps] <= 60).mean())
        near = []
        for lane in gaps:
            others = chart.head_ms[d.attack[:chart.K, np.arange(4) != lane].any(1)]
            near.append(np.searchsorted(others, release[:, lane] + 40, side='right') >
                        np.searchsorted(others, release[:, lane] + 1, side='left'))
        out['near'] = float(np.mean(near))
    if pairs:
        out['together'] = float(np.mean([abs(release[:, a] - release[:, b]) <= 5 for a, b in pairs]))
    return out


def mc_job(row):
    config, models = _WORKER
    out = Path(config['out']) / 'joint-pointer'
    path = out / (row['sha256'] + '.json')
    if path.exists():
        return dict(sha=row['sha256'], cached=True)
    dec = heldout_chart(row, config['cache'])
    chart = chart_from_cache(dec)
    d = chart.derived()
    eligible = []
    for k in panel_indices(dec.K):
        closed = np.flatnonzero(d.held[k] & (chart.actions[k] != 0))
        shared = any(d.birth[k, a] == d.birth[k, b] for i, a in enumerate(closed) for b in closed[i + 1:])
        if shared and gap_release_lanes(d.held[k], chart.actions[k], 0):
            eligible.append(int(k))
    if len(eligible) > 24:
        eligible = [eligible[i] for i in np.linspace(0, len(eligible) - 1, 24).astype(int)]
    records = []
    with torch.inference_mode():
        for cp, model in models.items():
            if not eligible:
                continue
            z = model.window_hands(chart, eligible)
            for i, k in enumerate(eligible):
                observed = mc_events(chart, k, d.release[k:k + 1])
                for arm in (('none', 'same_birth', 'fixed_forward') if cp == 56 else ('none',)):
                    for seed in (810, 811, 812):
                        draws = joint_draws(model, chart, k, z[i:i + 1], seed ^ k, lesion=arm)
                        predicted = mc_events(chart, k, draws)
                        records.append(dict(sha=row['sha256'], group=row['group_id'], ln_class=row['ln_class'],
                            band=row['band'], k=k, eos=k == dec.K, checkpoint=cp, arm=arm, seed=seed, samples=64,
                            **{event + '_observed': value for event, value in observed.items()},
                            **{event + '_predicted': value for event, value in predicted.items()}))
    write_json(path, records)
    return dict(sha=row['sha256'], decisions=len(eligible), draws=len(records) * 64)


def run_mc(args):
    out = Path(args.out)
    (out / 'joint-pointer').mkdir(exist_ok=True)
    rows = json.loads((out / 'panel.json').read_text())['charts']
    if args.limit:
        rows = rows[:args.limit]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(vars(args), 'mc')) as pool:
        for task in as_completed([pool.submit(mc_job, row) for row in rows]):
            print(json.dumps(task.result()), flush=True)


def diagnostic_job(row):
    from ensomi_model.system.arrange import arrange
    from ensomi_model.system.prior import Prior, kappa_targets, rows_features
    from ensomi_model.r2.features import Chart
    config, models = _WORKER
    out = Path(config['out']) / 'diagnostics'
    done = out / (row['sha256'] + '.done')
    if done.exists():
        return dict(sha=row['sha256'], cached=True)
    dec = heldout_chart(row, config['cache'])
    grid = grid_of(dec)
    root = Path(config['model_dir'])
    cal = json.loads((root / 'calibration.json').read_text())
    prior = Prior.load(root / 'prior.json')
    target = kappa_targets(prior.predict(rows_features(dec.head_ms, dec.song_ms, grid), int(row['band'])))
    age_bins = [0, 30, 60, 125, 250, 500, 1000, 2000, np.inf]
    decisions, hazards, releases, replay_metrics = [], [], [], []
    with torch.inference_mode():
        for cp, model in models.items():
            for mode, start in [('bos', 0), ('prefix', dec.K // 3)]:
                for kappa in (0, 3):
                    for seed in (954, 955, 956):
                        marginals = []
                        original = model.action_log_probs

                        def record(z, masks):
                            logp = original(z, masks)
                            p = logp.double().exp().numpy()
                            marginals.append(np.stack([p @ (ACTIONS[:, l, None] == np.arange(5))
                                                       for l in range(4)], axis=1)[0])
                            return logp

                        model.action_log_probs = record
                        try:
                            a, g, _ = arrange(model, dec.head_ms, dec.song_ms, grid, seed=seed, target=target,
                                kappa=kappa, masks=cal['masks'][str(row['band'])],
                                prefix_actions=dec.actions[:start], prefix_gap=dec.gap_release_ms[:start])
                        finally:
                            model.action_log_probs = original
                        chart = Chart(dec.head_ms, dec.song_ms, grid, a, g)
                        d = chart.derived()
                        common = dict(sha=row['sha256'], group=row['group_id'], ln_class=row['ln_class'],
                                      band=row['band'], checkpoint=cp, mode=mode, kappa=kappa, seed=seed)
                        objects = objects_from_decisions(dec.head_ms, a, g, dec.song_ms)
                        replay_metrics.append(dict(common, **ln_statistics(objects, grid, dec.head_ms[start])))
                        for i, k in enumerate(range(start, dec.K + 1)):
                            t = chart.time(k)
                            gap = t - chart.time(k - 1) if k else np.nan
                            held, m = d.held[k], marginals[i]
                            decisions.append(dict(common, k=k, eos=k == dec.K, dense=gap <= 125,
                                free=int((~held).sum()), held=int(held.sum()), heads=int(d.attack[k].sum()),
                                birth_free=int((~held & d.ln_head[k]).sum()),
                                birth_held=int((held & d.ln_head[k]).sum()),
                                raw_birth_free_expected=float((m[:, 2] * ~held).sum()),
                                raw_birth_held_expected=float((m[:, 4] * held).sum())))
                            for lane in np.flatnonzero(held):
                                age = t - d.start[k, lane]
                                hazards.append(dict(common, k=k, eos=k == dec.K,
                                    age_bin=int(np.searchsorted(age_bins, age, side='right') - 1),
                                    raw_expected=1 - m[lane, 0], observed=int(a[k, lane] != 0)))
                            for lane in np.flatnonzero(held & (a[k] != 0)):
                                end = d.release[k, lane]
                                others = dec.head_ms[d.attack[:dec.K, np.arange(4) != lane].any(1)]
                                near = np.searchsorted(others, end + 40, side='right') > np.searchsorted(others, end + 1)
                                releases.append(dict(common, k=k, lane=int(lane), eos=k == dec.K,
                                    born_in_suffix=d.birth[k, lane] >= start, gap_release=a[k, lane] >= 2,
                                    gap_ms=gap, before_row_ms=t - end, hold_ms=end - d.start[k, lane], near=near,
                                    code=int(a[k, lane])))
    for kind, values in [('decisions', decisions), ('hazards', hazards), ('releases', releases)]:
        pd.DataFrame(values).to_parquet(out / (row['sha256'] + '-' + kind + '.parquet'), index=False)
    write_json(out / (row['sha256'] + '-replay.json'), replay_metrics)
    done.write_text('complete\n')
    return dict(sha=row['sha256'], runs=36)


def run_diagnostics(args):
    out = Path(args.out)
    (out / 'diagnostics').mkdir(exist_ok=True)
    panel = json.loads((out / 'panel.json').read_text())['charts']
    rows = ([r for r in panel if r['ln_class'] == 'inverse'] +
            [r for r in panel if r['ln_class'] == 'ln_heavy'][:4] +
            [r for r in panel if r['ln_class'] == 'normal'][:5])
    if args.diagnostic_band is not None:
        selected = {r['sha256'] for r in rows}
        rows += [r for r in panel if r['band'] == args.diagnostic_band and r['sha256'] not in selected]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(vars(args), 'diagnostics')) as pool:
        for task in as_completed([pool.submit(diagnostic_job, row) for row in rows]):
            print(json.dumps(task.result()), flush=True)


def run_outliers(args):
    """Inspect high-loss source decisions after scoring, without resampling charts."""
    root = Path(args.out)
    table = pd.concat([pd.read_parquet(p) for p in (root / 'tf').glob('panel-*-56-decisions.parquet')])
    selected = pd.concat([table.nlargest(8, 'pointer_nll'), table[table.eos & (table.ln_class == 'inverse')]])
    selected = selected.drop_duplicates(['sha', 'k'])
    torch.set_num_threads(1)
    models = {cp: load_model(getattr(args, 'checkpoint_' + str(cp))) for cp in (48, 56, 64)}
    model = models[56]
    records = []
    with torch.inference_mode():
        for row in selected.to_dict('records'):
            if role_of(row['group']) != 'fit_dev':
                raise ValueError('Pointer diagnostics require the unsealed fit_dev split')
            chart = chart_from_cache(load_chart(Path(args.cache) / 'charts' / (row['sha'] + '.npz')))
            k = row['k']
            z = model.window_hands(chart, [k])
            factors = factors_for(chart, [k])
            nll = {}
            for cp, checkpoint_model in models.items():
                zz = checkpoint_model.window_hands(chart, [k])
                lp, _, _ = release_probabilities(checkpoint_model, zz, factors, chart, 1)
                nll[cp] = float(-lp[0])
            lp, _, _ = release_probabilities(model, z, factors, chart, 1, 'placement_relation')
            lesion_nll = float(-lp[0])
            floor = float(np.sqrt(np.mean([(nll[a] - nll[b]) ** 2 for a, b in [(48, 56), (48, 64), (56, 64)]])))
            target, pairs = model.factor_pair_log_probs(z, factors, chart, ())
            probabilities = np.split(pairs.exp().numpy(), np.cumsum([len(f.times) for f in factors])[:-1])
            d = chart.derived()
            descriptions = []
            for i, (f, probability) in enumerate(zip(factors, probabilities)):
                best = np.argsort(probability)[-3:][::-1]
                descriptions.append(dict(lane=f.lane, orientation=f.orientation, candidates=len(f.times),
                    birth_ms=d.start[k, f.lane], target_ms=f.times[f.target], target_logp=float(target[i]),
                    top_times=f.times[best], top_probabilities=probability[best]))
            records.append(dict(sha=row['sha'], k=k, ln_class=row['ln_class'], eos=row['eos'],
                time_ms=chart.time(k), previous_row_ms=chart.time(k - 1), codes=chart.actions[k],
                factors=descriptions, checkpoint_nll=nll, no_placement_relation_nll=lesion_nll,
                checkpoint_rms=floor, lesion_effect=lesion_nll - nll[56],
                effect_over_floor=abs(lesion_nll - nll[56]) / floor if floor > 0 else None))
    write_json(root / 'pointer-outliers.json', records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['tf', 'fr', 'joint-pointer', 'diagnostics', 'outliers'])
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    for c in (48, 56, 64):
        p.add_argument(f'--checkpoint-{c}', required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--manifest')
    p.add_argument('--model-dir')
    p.add_argument('--limit', type=int)
    p.add_argument('--diagnostic-band', type=int, choices=(2, 3, 4, 5))
    args = p.parse_args()
    {'tf': run_tf, 'fr': run_fr, 'joint-pointer': run_mc, 'diagnostics': run_diagnostics,
     'outliers': run_outliers}[args.command](args)


if __name__ == '__main__':
    main()
