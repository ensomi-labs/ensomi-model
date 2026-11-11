"""Paired crop-grid perturbation of one-pass and eight-pass N1 inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import measures as M
from .averaging import align_evidence, mean_evidence, shifted_wave, song_evidence

ARMS = ('single', 'wave8', 'crop8')


def cohort(items, count):
    """One lexical item per recording, round-robin over label strata ordered by rarity."""
    representatives = {}
    for x in sorted(items, key=lambda x:x['key']):
        representatives.setdefault(x['rec'], x)
    rng = np.random.default_rng(M.SEED)
    pools = {}
    for name in sorted({s for x in representatives.values() for s in x['strata']}):
        xs = [x for x in representatives.values() if name in x['strata']]
        pools[name] = [xs[i] for i in rng.permutation(len(xs))]
    order = sorted(pools, key=lambda k:(len(pools[k]), k))
    selected, seen = [], set()
    while len(selected) < min(count, len(representatives)):
        before = len(selected)
        for name in order:
            while pools[name] and pools[name][0]['rec'] in seen:
                pools[name].pop(0)
            if pools[name] and len(selected) < count:
                x = pools[name].pop(0)
                selected.append(x)
                seen.add(x['rec'])
        if len(selected) == before:
            break
    return selected


def map_comparison(a, b, lo, hi):
    clip = lambda t: t[(t >= lo) & (t <= hi)]
    ab = clip(a.timing(lo, hi).beats)
    bb = clip(b.timing(lo, hi).beats)
    matched = M.matching(ab, bb, 5.)
    changed = bool((matched['recall'] < 1 - 1e-6) or (matched['precision'] < 1 - 1e-6))
    topology = bool(len(a.beats) != len(b.beats) or len(a.segments or ()) != len(b.segments or ())
                    or len(a.downbeats) != len(b.downbeats))
    return dict(changed=changed, topology=topology, r5=matched['recall'], p5=matched['precision'],
                n_ref=matched['n_ref'], n_pred=matched['n_pred'])


def reproduction(saved, fresh):
    counts = (len(saved.beats) == len(fresh.beats) and len(saved.segments) == len(fresh.segments)
              and np.array_equal(saved.downbeats, fresh.downbeats))
    delta = float(np.max(np.abs(saved.beats - fresh.beats))) if len(saved.beats) == len(fresh.beats) else None
    return dict(counts_equal=bool(counts), max_beat_ms=delta, passed=bool(counts and delta is not None and delta <= 1.))


def decode_recording(x, seed, evidence, cfg_kw, saved=None, repeat=None):
    from ensomi_n.eval import score
    from ensomi_n.mapformat import MusicMap
    from ensomi_n.v0.decode import DecodeConfig, decode
    from ensomi_n.v0.evaluate import v0_evidence
    t0 = time.monotonic()
    maps, scores = {}, {}
    lo, hi = x['ref'].lo_ms + 1000., x['ref'].hi_ms - 1000.
    for arm, acts in evidence.items():
        m, _ = decode(v0_evidence(acts), DecodeConfig(**cfg_kw))
        maps[arm] = m
        sc = score(x['ref'], m, spans=[(lo, hi)])
        scores[arm] = {f:getattr(sc, f) for f in ('bl20', 'bars_start', 'bars_same', 'level', 'offset_ms')}
    null = None
    if repeat is not None:
        m, _ = decode(v0_evidence(repeat), DecodeConfig(**cfg_kw))
        null = dict(map_comparison(maps['single/0'], m, lo, hi),
                    max_beat_activation=float(np.max(np.abs(evidence['single/0']['beat'] - repeat['beat']))))
        if null['changed'] or null['max_beat_activation'] > 1e-6:
            raise RuntimeError(f'identical-repeat null failed: {null}')
    repro = reproduction(MusicMap.from_json(saved), maps['single/0']) if saved else None
    if repro is not None and not repro['passed']:
        raise RuntimeError(f'saved-map reproduction failed for {x["key"]}/s{seed}: {repro}')
    rows = []
    for arm in ARMS:
        delta = {f: scores[f'{arm}/0'][f] - scores['single/0'][f] for f in scores['single/0']}
        rows.append(dict(key=x['key'], item=x['item'], rec=x['rec'], seed=seed, arm=arm,
                         strata=list(x['strata']), **map_comparison(maps[f'{arm}/0'], maps[f'{arm}/500'], lo, hi),
                         **scores[f'{arm}/0'], delta=delta, fresh=scores[f'{arm}/500']))
    return dict(rows=rows, maps={k:v.to_json() for k,v in maps.items()}, null=null, reproduction=repro,
                decode_seconds=time.monotonic() - t0)


def aggregate(rows):
    result = {}
    rng = np.random.default_rng(M.SEED)
    strata = ['all'] + sorted({s for r in rows for s in r['strata']})
    for seed in (0, 1):
        for stratum in strata:
            base = {r['rec']:r for r in rows if r['seed'] == seed and r['arm'] == 'single'
                    and (stratum == 'all' or stratum in r['strata'])}
            for arm in ARMS:
                rs = [r for r in rows if r['seed'] == seed and r['arm'] == arm and r['rec'] in base]
                entry = M.summarize(rs, ['changed', 'topology', 'r5', 'p5', 'bl20', 'bars_start', 'bars_same', 'level'])
                entry['n_changed'] = sum(r['changed'] for r in rs)
                entry['n'] = len(rs)
                for f in ('bl20', 'bars_start', 'bars_same', 'level'):
                    entry['delta_' + f] = M.interval([r[f] - base[r['rec']][f] for r in rs])
                entry['delta_changed'] = M.interval([int(r['changed']) - int(base[r['rec']]['changed']) for r in rs])
                if rs:
                    a = np.array([r['changed'] for r in rs], float)
                    b = np.array([base[r['rec']]['changed'] for r in rs], float)
                    ix = rng.integers(len(a), size=(2000, len(a)))
                    den = b[ix].mean(1)
                    ok = den > 0
                    reductions = 1 - a[ix].mean(1)[ok] / den[ok]
                    entry['relative_reduction'] = dict(value=float(1 - a.mean() / b.mean()) if b.mean() else None,
                        lo=float(np.quantile(reductions, .025)) if len(reductions) else None,
                        hi=float(np.quantile(reductions, .975)) if len(reductions) else None,
                        denominator_changed=int(b.sum()), n=len(b), valid_bootstraps=int(ok.sum()))
                result[f's{seed}/{stratum}/{arm}'] = entry
    for arm in ARMS:
        ss = [{r['rec']:r for r in rows if r['seed'] == s and r['arm'] == arm} for s in (0, 1)]
        common = sorted(ss[0].keys() & ss[1].keys())
        result[f'seed_null/{arm}'] = {f:M.interval([ss[1][k][f] - ss[0][k][f] for k in common])
                                     for f in ('bl20', 'bars_start', 'bars_same', 'level')}
    return result


def run(a):
    import torch
    from ensomi_n.features.mel import log_mel
    from ensomi_n.features.decode import decode as decode_audio
    from ensomi_n.features.cache import mel_path
    from ensomi_n.v0.songs import memo_from_mel
    from ensomi_n.v0.evaluate import _load_model, pick_settings, inference_song
    from ensomi_n.v0.infer import song_outputs

    torch.set_num_threads(1)
    if a.device != 'mps' or not torch.backends.mps.is_available():
        raise RuntimeError('this measurement requires explicit available MPS')
    started = time.monotonic()
    root, out = a.root, a.out
    data = [x for x in C.items(root, ('panel',)) if 'error' not in x]
    selected = cohort(data, a.count)
    C.write_json(out / 'b-cohort.json', [{k:x[k] for k in ('key', 'rec', 'item', 'strata', 'audio_rel')} for x in selected])
    bulk = root / 'artifacts/bulk/n1-cache'
    index = json.loads((bulk / 'song-index-v0.json').read_text())
    cfg_kw = pick_settings(root / C.NIGHT / 'v0-s0-20261019/eval')
    if cfg_kw != {'reset_penalty':14.}:
        raise RuntimeError(f'unexpected original decoder settings {cfg_kw}')
    from ensomi_n.v0.decode import DecodeConfig
    C.write_json(out / 'b-decoder.json', dict(kw=cfg_kw, full=asdict(DecodeConfig(**cfg_kw))))
    checkpoints = [root / f'artifacts/bulk/n1-runs/v01-s{s}-20261020/ckpt-009001.pt' for s in (0, 1)]
    identities = [dict(seed=s, path=str(p), sha256=C.sha256(p)) for s,p in enumerate(checkpoints)]
    C.write_json(out / 'b-checkpoints-before.json', identities)
    models = [_load_model(p, a.device) for p in checkpoints]
    saved = [C.maps(root, s, 'panel') for s in (0, 1)]
    rows, records = [], []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        for ix, x in enumerate(selected):
            if time.monotonic() - started >= a.minutes * 60:
                break
            C.storage_guard(out)
            t0 = time.monotonic()
            wave = decode_audio(C.allowed_path(root, x['audio_rel'])).wave
            st = index[x['item']]
            source_mel = np.load(mel_path(bulk / 'mel', x['item']), mmap_mode='r')
            features = []
            for j in range(8):
                mel = log_mel(shifted_wave(wave, j * 30)).astype(np.float16)
                features.append((mel, memo_from_mel(mel), float(mel.astype(np.float32).mean()),
                                 float(mel.astype(np.float32).std() + 1e-6)))
            if features[0][0].shape != source_mel.shape or not np.array_equal(features[0][0], source_mel):
                raise RuntimeError(f'fresh Mel differs from cached source for {x["key"]}')
            # Preserve the source normalization down to its stored rounding.
            features[0] = (*features[0][:2], st['mean'], st['std'])
            futures, nulls = [], []
            for seed, model in enumerate(models):
                acts, by_origin = {}, {}
                for fresh in (0, 50):
                    waves = []
                    for j, (mel, memo, mean, std) in enumerate(features):
                        z = song_evidence(model, mel, memo, mean, std, a.device, origin=fresh)
                        if j == 0:
                            by_origin[fresh] = z
                            acts[f'single/{fresh * 10}'] = z
                        waves.append(align_evidence(z, 1.25 * j))
                    acts[f'wave8/{fresh * 10}'] = mean_evidence(waves)
                    del waves
                    crops = []
                    for origin in range(fresh, fresh + 1000, 125):
                        if origin not in by_origin:
                            by_origin[origin] = song_evidence(model, *features[0], a.device, origin=origin)
                        crops.append(by_origin[origin])
                    acts[f'crop8/{fresh * 10}'] = mean_evidence(crops)
                repeat = None
                if ix == 0:
                    repeat = song_outputs(model, inference_song(x['item'], index, bulk), a.device)
                    max_error = max(float(np.max(np.abs(acts['single/0'][k] - repeat[k]))) for k in repeat)
                    if max_error > 1e-6:
                        raise RuntimeError(f'in-memory inference differs from source: {max_error}')
                    nulls.append(dict(seed=seed, source_inference_max_error=max_error))
                futures.append(pool.submit(decode_recording, x, seed, acts, cfg_kw,
                                           saved[seed][x['key']].to_json() if ix < 3 else None, repeat))
                del acts, by_origin
            completed = [f.result() for f in futures]
            for seed, result in enumerate(completed):
                rows.extend(result['rows'])
                # Maps are small and needed to audit geometry; intermediate evidence is discarded.
                C.write_json(out / f'b-maps-{ix:03d}-s{seed}.json', result['maps'])
                records.append(dict(index=ix, key=x['key'], seed=seed, null=result['null'],
                                    reproduction=result['reproduction'], decode_seconds=result['decode_seconds']))
            C.write_rows(out / 'b-recordings.jsonl', rows)
            C.write_json(out / 'b-verification.json', records)
            C.write_json(out / 'b-summary.json', aggregate(rows))
            print('B', ix + 1, '/', len(selected), x['item'][:12], 'seconds', round(time.monotonic() - t0, 2),
                  'total', round(time.monotonic() - started, 2), flush=True)
    for p, before in zip(checkpoints, identities):
        if C.sha256(p) != before['sha256']:
            raise RuntimeError('checkpoint changed during inference')
    C.write_json(out / 'b-runtime.json', dict(seconds=time.monotonic() - started,
        completed_recordings=len(rows) // 6, target=len(selected), minutes_limit=a.minutes,
        checkpoint_hashes_unchanged=True, **C.storage_guard(out)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', required=True, choices=['mps'])
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--minutes', type=float, default=60.)
    run(p.parse_args())


if __name__ == '__main__':
    main()
