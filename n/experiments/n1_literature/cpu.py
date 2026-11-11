"""CPU measurements using frozen unsealed evaluations; outputs are local research artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import measures as M


def a1(root, out):
    from ensomi_n.eval import score
    from ensomi_n.eval.measures import nearest
    from ensomi_n.features.decode import decode
    from ensomi_n.timing import Reference

    rows, summary, by_seed = [], {}, {}
    for panel in ('panel', 'val_hard'):
        for seed in (0, 1):
            xs = C.scores(root, seed, panel)
            for arm in sorted({r['arm'] for r in xs if r['arm'].startswith('v0@')}):
                arm_rows = [r for r in xs if r['arm'] == arm]
                rec = M.recording_values(arm_rows, 'offset_ms')
                values = list(rec.values())
                summary[f'{panel}/s{seed}/{arm}'] = dict(
                    median=M.interval(values, 'median'), robust_sd=M.interval(values, 'robust_sd'),
                    sd=M.interval(values, 'sd'), above10=M.interval(np.abs(values) > 10),
                    items=len(arm_rows))
                rows.extend(dict(panel=panel, seed=seed, arm=arm, rec=k, offset_ms=v) for k, v in rec.items())
                if arm == 'v0@009001':
                    by_seed[panel, seed] = {r['key']: r for r in arm_rows}
            print('A1 offsets', panel, seed, summary[f'{panel}/s{seed}/v0@009001'], flush=True)
        for metric in ('offset_ms', 'bl20', 'bars_start', 'bars_same', 'level'):
            recs = [M.recording_values(list(by_seed[panel, s].values()), metric) for s in (0, 1)]
            d = [recs[1][k] - recs[0][k] for k in sorted(recs[0].keys() & recs[1].keys())]
            summary[f'{panel}/seed_difference/{metric}'] = dict(mean=M.interval(d), sd=M.interval(d, 'sd'),
                                                                robust_sd=M.interval(d, 'robust_sd'))
        for seed in (0, 1):
            arms = sorted({r['arm'] for r in rows if r['panel'] == panel and r['seed'] == seed})
            final = {r['rec']: r['offset_ms'] for r in rows if r['panel'] == panel and r['seed'] == seed and r['arm'] == arms[-1]}
            for arm in arms[:-1]:
                old = {r['rec']: r['offset_ms'] for r in rows if r['panel'] == panel and r['seed'] == seed and r['arm'] == arm}
                delta = [final[k] - old[k] for k in sorted(final.keys() & old.keys())]
                summary[f'{panel}/s{seed}/final_minus_{arm}'] = dict(sd=M.interval(delta, 'sd'),
                                                                              robust_sd=M.interval(delta, 'robust_sd'))
    C.write_rows(out / 'a1-offset-recordings.jsonl', rows)
    C.write_json(out / 'a1-offset-summary.json', summary)

    data = C.items(root)
    usable = [x for x in data if 'error' not in x]
    C.write_rows(out / 'input-errors.jsonl', [x for x in data if 'error' in x])
    all_maps, verified = {}, []
    for panel in ('panel', 'val_hard'):
        for seed in (0, 1):
            all_maps[panel, seed] = C.maps(root, seed, panel)
            sample = [x for x in usable if x['panel'] == panel and x['key'] in by_seed[panel, seed]][:3]
            for x in sample:
                old = by_seed[panel, seed][x['key']]['offset_ms']
                new = score(x['ref'], all_maps[panel, seed][x['key']]).offset_ms
                if not np.isclose(old, new, atol=1e-4):
                    raise RuntimeError(f'offset reproduction failure {x["key"]}: {old}, {new}')
                verified.append(dict(key=x['key'], seed=seed, saved=old, rescored=new))
    C.write_json(out / 'a1-offset-verification.json', verified)

    pairs, refs, charts = C.pair_inputs(root)
    pair_rows = []
    for p in pairs:
        for x in usable:
            paths = set(x['charts'])
            if not any(str(c.path.relative_to(root)) in paths for k in ('a', 'b') for c in charts[p[k]]):
                continue
            audio_a = charts[p['a']][0].path.parent / refs[p['a']].meta['audio']
            audio_b = charts[p['b']][0].path.parent / refs[p['b']].meta['audio']
            if C.sha256(audio_a) != x['item'] or C.sha256(audio_b) != x['item']:
                continue
            lo, hi = p['lo'], p['hi']
            for seed in (0, 1):
                mmap = all_maps[x['panel'], seed].get(x['key'])
                if mmap is None:
                    continue
                offsets = {}
                for k in ('a', 'b'):
                    r = refs[p[k]]
                    clipped = Reference(r.timing, lo, hi, r.heads[(r.heads >= lo) & (r.heads <= hi)],
                                        r.segments, r.unscorable, r.key, r.meta)
                    offsets[k] = score(clipped, mmap).offset_ms
                pair_rows.append(dict(panel=x['panel'], seed=seed, rec=x['rec'], key=x['key'],
                                      a=p['a'], b=p['b'], differing=not p['flags']['c1'],
                                      offset_a=offsets['a'], offset_b=offsets['b'], pair_offset=p['offset'],
                                      abs_a=abs(offsets['a']), abs_b=abs(offsets['b']), abs_pair=abs(p['offset'])))
    C.write_rows(out / 'a1-pair-comparisons.jsonl', pair_rows)
    C.write_json(out / 'a1-pair-summary.json', {
        f'{panel}/s{seed}/{view}': M.summarize([r for r in pair_rows if r['panel'] == panel and r['seed'] == seed
                                             and (view == 'all' or r['differing'])], ['abs_a', 'abs_b', 'abs_pair'])
        for panel in ('panel', 'val_hard') for seed in (0, 1) for view in ('all', 'differing')})
    for name, rel in [('report', 'artifacts/reports/mapper-spread-20261021/summary.json'),
                      ('run', 'artifacts/bulk/mapper-spread-20261021/run/summary.json')]:
        d = json.loads((root / rel).read_text())['summary']['C1']['all']
        C.write_json(out / f'a1-mapper-floor-{name}.json', dict(source=rel, sha256=C.sha256(root / rel),
                    pair=d['offset'], counts={k:v for k,v in d.items() if k.startswith('n_')},
                    single_independent={k:v / np.sqrt(2) for k,v in d['offset']['rsd_signed'].items()}))

    synth = root / 'artifacts/n1-night-20261018/synth-v0'
    audio_dir = root / 'artifacts/bulk/n1-synth/v0'
    calibration = []
    lag = None
    for setting in ('control_click', 'control_a', 'control_b', 'control_c', 'control_frac'):
        for seed in (0, 1):
            name = f'{setting}-s{seed}'
            truth = json.loads((synth / name / 'onsets.json').read_text())
            onsets = np.unique([t for l in truth['layers'].values() for t in l['onsets_ms']])
            wave = decode(audio_dir / f'{name}.wav').wave
            flux = M.spectral_flux(wave)
            peaks = M.flux_peaks(flux.mean(1))
            e, _ = nearest(onsets, peaks)
            e = e[np.abs(e) <= 100]
            if lag is None:
                lag = float(np.median(e))
            residual = e - lag
            calibration.append(dict(clip=name, role='calibration' if name == 'control_click-s0' else 'validation',
                                    n_onsets=len(onsets), n_matched=len(e), lag_ms=lag,
                                    median_ms=float(np.median(residual)),
                                    p95_abs_ms=float(np.quantile(np.abs(residual), .95)),
                                    within5=float(np.mean(np.abs(residual) <= 5)),
                                    within10=float(np.mean(np.abs(residual) <= 10))))
    C.write_rows(out / 'a1-onset-calibration.jsonl', calibration)
    print('A1 flux calibrated', calibration, flush=True)
    selected = [x for x in usable if any(abs(by_seed[x['panel'], s].get(x['key'], {}).get('offset_ms') or 0) > 10
                                         for s in (0, 1))]
    C.write_json(out / 'a1-onset-cohort.json', [{k:x[k] for k in ('key', 'rec', 'item', 'panel', 'audio_rel')} for x in selected])
    onset_rows, errors, done = [], [], {}
    for ix, x in enumerate(selected):
        C.storage_guard(out)
        try:
            if x['item'] not in done:
                decoded = decode(C.allowed_path(root, x['audio_rel']))
                f = M.spectral_flux(decoded.wave)
                done[x['item']] = ([M.flux_peaks(f.mean(1))] + [M.flux_peaks(f[:, k]) for k in range(3)],
                                   dict(decoder=decoded.decoder, source_rate=decoded.source_rate,
                                        lead_samples=decoded.lead_samples, header=decoded.header))
            peaks, meta = done[x['item']]
            lines = x['ref'].lines()
            weights = x['ref'].weights(lines)
            for seed in (0, 1):
                offset = by_seed[x['panel'], seed].get(x['key'], {}).get('offset_ms')
                if offset is None or abs(offset) <= 10:
                    continue
                result = M.onset_offset(lines.t, weights, peaks[0], offset, lag)
                bands = [M.onset_offset(lines.t, weights, p, offset, lag)['flux_offset_ms'] for p in peaks[1:]]
                spread = float(np.max(bands) - np.min(bands))
                result.update(panel=x['panel'], key=x['key'], rec=x['rec'], item=x['item'], seed=seed,
                              offset_ms=offset, band_offsets_ms=bands, band_range_ms=spread, **meta)
                result['strong_support'] = bool(result['matched_peaks'] >= 20 and np.isfinite(spread) and spread <= 10)
                result.update({v + '_share': float(result['verdict'] == v) for v in ('model', 'chart', 'tie')})
                onset_rows.append(result)
        except Exception as exc:
            errors.append(dict(key=x['key'], error=f'{type(exc).__name__}: {exc}'))
        if (ix + 1) % 10 == 0:
            print('A1 onset', ix + 1, '/', len(selected), 'errors', len(errors), flush=True)
            C.write_rows(out / 'a1-onsets.jsonl', onset_rows)
    C.write_rows(out / 'a1-onsets.jsonl', onset_rows)
    C.write_rows(out / 'a1-onset-errors.jsonl', errors)
    result = {}
    for panel in ('panel', 'val_hard'):
        for seed in (0, 1):
            for subset in ('all', 'strong'):
                rr = [r for r in onset_rows if r['panel'] == panel and r['seed'] == seed and r['verdict'] != 'undefined'
                      and (subset == 'all' or r['strong_support'])]
                result[f'{panel}/s{seed}/{subset}'] = dict(items=len(rr), counts=dict(Counter(r['verdict'] for r in rr)),
                    **M.summarize(rr, ['model_share', 'chart_share', 'tie_share', 'weight_coverage', 'flux_offset_ms']))
    C.write_json(out / 'a1-onset-summary.json', result)
    print('A1 complete', flush=True)


def a2(root, out):
    from ensomi_n.eval import score
    from ensomi_n.labels import read_osu, reference_from_charts
    from ensomi_n.mapformat import MusicMap
    import torch
    from beat_this.model.postprocessor import Postprocessor
    torch.set_num_threads(1)
    pairs, refs, pair_charts = C.pair_inputs(root)
    data = [x for x in C.items(root, ('panel',)) if 'error' not in x]
    classes = {k: M.spacing_class([s.start_ms for s in r.segments]) for k, r in refs.items()}
    chart_rows = []
    for group, groups in [('pairs', pair_charts), ('panel', {x['key']:[read_osu(root / p) for p in x['charts']] for x in data})]:
        for key, cs in groups.items():
            for c in cs:
                chart_rows.append(dict(group=group, key=key, chart=str(c.path.relative_to(root)),
                                       cls=M.spacing_class([r.offset_ms for r in c.red_lines])))
    counts = dict(charts={g:dict(Counter(r['cls'] for r in chart_rows if r['group'] == g)) for g in ('pairs', 'panel')},
                  pair_references=dict(Counter(classes.values())),
                  panel_references=dict(Counter(M.spacing_class([s.start_ms for s in x['ref'].segments]) for x in data)),
                  pair_class={v:dict(Counter(max(classes[p['a']], classes[p['b']]) for p in pairs
                                             if v == 'all' or not p['flags']['c1'])) for v in ('all', 'differing')},
                  cross={v:dict(Counter(f"{classes[p['a']]}-{classes[p['b']]}" for p in pairs
                                        if v == 'all' or not p['flags']['c1'])) for v in ('all', 'differing')})
    C.write_rows(out / 'a2-chart-classes.jsonl', chart_rows)
    C.write_json(out / 'a2-counts-before-scores.json', counts)
    print('A2 COUNTS BEFORE AGREEMENT', json.dumps(counts), flush=True)
    pair_rows = []
    for p in pairs:
        def bars(d):
            return d['bars']['o'] / d['bars']['n'] if d['bars']['n'] else np.nan
        pair_rows.append(dict(rec=p['rec'], a=p['a'], b=p['b'], cls=max(classes[p['a']], classes[p['b']]),
                              class_a=classes[p['a']], class_b=classes[p['b']], differing=not p['flags']['c1'],
                              bl20_o=np.mean([p['ab']['le20_o'], p['ba']['le20_o']]),
                              bl20=np.mean([p['ab']['le20'], p['ba']['le20']]),
                              bars_start_o=np.mean([bars(p['ab']), bars(p['ba'])]),
                              bars_a=p['ab']['bars']['n'], bars_b=p['ba']['bars']['n']))
    C.write_rows(out / 'a2-pairs.jsonl', pair_rows)
    summary = {}
    for cls in (1, 2, 3):
        for view in ('all', 'differing', 'same_class_differing'):
            rr = [r for r in pair_rows if r['cls'] == cls and (view == 'all' or r['differing'])
                  and (view != 'same_class_differing' or r['class_a'] == r['class_b'])]
            summary[f'class{cls}/{view}'] = dict(pairs=len(rr), recordings=len({r['rec'] for r in rr}),
                readable=len(rr) >= 10, **(M.summarize(rr, ['bl20_o', 'bars_start_o', 'bl20']) if len(rr) >= 10 else {}))
    C.write_json(out / 'a2-pair-summary.json', summary)
    pp = Postprocessor(type='minimal')
    tracker, prediction = [], {}
    for ix, x in enumerate(data):
        if x['item'] not in prediction:
            path = root / f'artifacts/bulk/n1-cache/bt-final0-8c328b45f59d/{x["item"]}.npz'
            with np.load(path) as z:
                b, d = pp(torch.from_numpy(z['beat']), torch.from_numpy(z['down']))
            b, d = b * 1000 - 30., d * 1000 - 30.
            prediction[x['item']] = MusicMap.from_beats(b, downbeats=np.searchsorted(b, d))
        m = prediction[x['item']]
        for chart in sorted(x['charts']):
            c = read_osu(root / chart)
            if not c.red_lines or not len(c.heads):
                continue
            r = reference_from_charts([c], chart)
            sc = score(r, m)
            pt = m.timing(r.lo_ms, r.hi_ms)
            rb, pb = r.timing.beats, pt.beats
            rc, pc = r.lines().t, pt.canonical().t
            clip = lambda t: t[(t >= r.lo_ms) & (t <= r.hi_ms)]
            tracker.append(dict(rec=x['rec'], key=x['key'], chart=chart, item=x['item'],
                                cls=M.spacing_class([a.offset_ms for a in c.red_lines]), bl20=sc.bl20,
                                f70_notated=M.matching(clip(rb), clip(pb), 70.)['f'],
                                f70_canonical=M.matching(clip(rc), clip(pc), 70.)['f']))
        if (ix + 1) % 50 == 0:
            print('A2 tracker', ix + 1, '/', len(data), flush=True)
    C.write_rows(out / 'a2-tracker-charts.jsonl', tracker)
    C.write_json(out / 'a2-tracker-summary.json', {
        str(cls):dict(charts=sum(r['cls'] == cls for r in tracker), **M.summarize([r for r in tracker if r['cls'] == cls],
                                               ['bl20', 'f70_canonical', 'f70_notated'])) for cls in (1, 2, 3)})
    print('A2 complete', flush=True)


def a3(root, out):
    from ensomi_n.eval.measures import nearest
    from ensomi_n.v0.labels import V0Labels
    data = [x for x in C.items(root) if 'error' not in x]
    pairs, refs, pair_charts = C.pair_inputs(root)
    reset_of = {k:V0Labels.from_reference(r).bounds[V0Labels.from_reference(r).bound_kind == 1] for k,r in refs.items()}
    manifests = {}
    for seed in (0, 1):
        base = root / C.CACHE / f'v01-s{seed}-009001'
        manifests[seed] = {r['sha256']:r for r in C.read_jsonl(base / 'manifest.jsonl') if r['split'] in ('fit_dev', 'val-n1')}
    results, errors, verified = [], [], {}
    for ix, x in enumerate(data):
        lab = V0Labels.from_reference(x['ref'])
        reset = lab.bounds[lab.bound_kind == 1]
        has_pair, supported = False, np.zeros(len(reset), dtype=bool)
        own = set(x['charts'])
        for p in pairs:
            for a, b, off in [('a', 'b', p['ab']['o']), ('b', 'a', p['ba']['o'])]:
                if any(str(c.path.relative_to(root)) in own for c in pair_charts[p[a]]):
                    has_pair = True
                    e, _ = nearest(reset, reset_of[p[b]] - off)
                    supported |= np.abs(e) <= 20.
        for seed in (0, 1):
            try:
                manifest = manifests[seed][x['item']]
                if (('head_layout' in manifest and manifest['head_layout'][6] != 'boundary')
                        or manifest['fps'] != 100 or manifest['t0_ms'] != 0):
                    raise ValueError('unexpected feature-cache coordinates/layout')
                path = root / C.CACHE / f'v01-s{seed}-009001' / manifest['npz']
                with np.load(path) as z:
                    logits = z['logits']
                    if logits.shape[1] != 23 or z['fps'] != 100 or z['t0_ms'] != 0:
                        raise ValueError('unexpected NPZ coordinates/layout')
                    if seed not in verified:
                        # Later cache manifests omit head_layout. Verify all columns against
                        # the checkpoint's linear projection, allowing float16 storage error.
                        import torch
                        ck = root / f'artifacts/bulk/n1-runs/v01-s{seed}-20261020/ckpt-009001.pt'
                        if C.sha256(ck) != manifest['checkpoint_sha256']:
                            raise ValueError('cache/checkpoint identity mismatch')
                        state = torch.load(ck, map_location='cpu', weights_only=False)['model']
                        w = state['heads.lin.weight'].numpy()
                        b = state['heads.lin.bias'].numpy()
                        expected = z['feat'][:1000].astype(np.float32) @ w.T + b
                        error = np.max(np.abs(expected - logits[:1000].astype(np.float32)), axis=0)
                        if np.max(error) > .1:
                            raise ValueError(f'cached head-layout projection mismatch: {error}')
                        verified[seed] = dict(item=x['item'], max_abs_by_column=error.tolist(),
                                              checkpoint_sha256=manifest['checkpoint_sha256'])
                        C.write_json(out / 'a3-cache-verification.json', verified)
                        del state
                    bound = logits[:, 6].astype(np.float64)
                subsets = dict(tempo=lab.bounds[lab.bound_kind == 0], reset=reset)
                if has_pair:
                    subsets.update(reset_supported=reset[supported], reset_unconfirmed=reset[~supported])
                else:
                    subsets['reset_no_pair'] = reset
                for kind, pos in subsets.items():
                    for radius in (3, 0):
                        r = M.boundary_auc(bound, lab.bounds, pos, lab.lo_ms, lab.hi_ms, radius)
                        results.append(dict(rec=x['rec'], key=x['key'], panel=x['panel'], seed=seed,
                                            kind=kind, radius=radius, **r))
            except Exception as exc:
                errors.append(dict(key=x['key'], seed=seed, error=f'{type(exc).__name__}: {exc}'))
        if (ix + 1) % 50 == 0:
            print('A3', ix + 1, '/', len(data), 'errors', len(errors), flush=True)
    C.write_rows(out / 'a3-boundaries.jsonl', results)
    C.write_rows(out / 'a3-errors.jsonl', errors)
    summary = {}
    for panel in ('panel', 'val_hard'):
        for seed in (0, 1):
            for kind in ('tempo', 'reset', 'reset_supported', 'reset_unconfirmed', 'reset_no_pair'):
                for radius in (3, 0):
                    rows = [r for r in results if r['panel'] == panel and r['seed'] == seed and r['kind'] == kind and r['radius'] == radius]
                    eligible = [r for r in rows if r['n_positive'] and r['n_negative']]
                    summary[f'{panel}/s{seed}/{kind}/radius{radius}'] = dict(
                        items=len(eligible), positive_events=sum(r['n_positive'] for r in eligible),
                        missing_negatives=sum(r['n_positive'] > 0 and r['n_negative'] == 0 for r in rows),
                        auc=M.interval(list(M.recording_values(eligible, 'auc').values())))
    C.write_json(out / 'a3-summary.json', summary)
    print('A3 complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['a1', 'a2', 'a3'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    C.storage_guard(a.out)
    import torch
    torch.set_num_threads(1)
    started = time.time()
    globals()[a.experiment](a.root, a.out)
    C.write_json(a.out / f'{a.experiment}-runtime.json', dict(seconds=time.time() - started, **C.storage_guard(a.out)))


if __name__ == '__main__':
    main()
