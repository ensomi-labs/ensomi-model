"""Score fixed band readouts on the two read-only panels through X5 D2."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import dataclasses
import gzip
import multiprocessing as mp
import subprocess
import time

from common import *
from bands import decode_band, own_probabilities, own_from_logits, pooled_probabilities
from predict import load_head, head_logits
from scipy.special import softmax
from ensomi_n.v0.evaluate import v0_evidence, oracle_evidence
from measurements import structural

ARMS = ('baseline', 'own_d2', 'band_d2', 'band_d2_mlp', 'band_plus_own_d2', 'oracle_d2')


def confusion(truth, pred):
    return np.bincount(7*np.asarray(truth, int)+np.asarray(pred, int), minlength=49).reshape(7,7)


def accuracy(ref, times, probabilities, strata):
    y, valid = targets([ref], times)
    truth, pred = y.argmax(1), probabilities.argmax(1)
    logp = np.log(probabilities+1e-4)
    result = {}
    for cell, spans in [('all', None)]+[(k,v) for k,v in strata.items() if k in ('S0','S5')]:
        mask = valid & base.inside(times, ref, spans)
        if not mask.any():
            continue
        frame = confusion(truth[mask], pred[mask])
        rt, rp, omitted = [], [], 0
        for i, seg in enumerate(ref.segments):
            end = ref.segments[i+1].start_ms if i+1 < len(ref.segments) else float('inf')
            in_regime = mask & (times >= seg.start_ms) & (times < end)
            if not in_regime.any():
                omitted += 1
                continue
            rt.append(int(band_of(seg.beat_ms))-BANDS[0])
            rp.append(int(logp[in_regime].mean(0).argmax()))
        result[cell] = dict(frame=frame, regime=confusion(rt,rp), omitted_regimes=omitted)
    return result


def initialise(seed, rows):
    global HEADS, POOL_WEIGHT, MANIFEST
    folder = OUT / seed / 'heads'
    HEADS = {kind: load_head(folder / f'{kind}.npz') for kind in ('linear', 'mlp')}
    POOL_WEIGHT = json.loads((folder / 'selection.json').read_text())['pool']['weight']
    MANIFEST = rows


def job(args):
    seed, xs = args
    with gzip.open(OUT / 'controls' / seed / (xs[0]['rec']+'.json.gz'), 'rt') as f:
        maps = json.load(f)
    for arm in ('band_d2','band_d2_mlp','band_plus_own_d2'):
        maps[arm] = {}
    acc, diagnostics, inputs = [], [], []
    seen = {}
    for x in xs:
        audio = x['item']
        if audio not in seen:
            ev = v0_evidence(base.evidence(seed, audio))
            row = MANIFEST[audio]
            path = cache_root(seed) / row['npz']
            with np.load(path, allow_pickle=False) as z:
                feat = z['feat']
                if len(feat) != ev.n_frames or float(z['fps']) != 100 or float(z['t0_ms']) != 0:
                    raise ValueError('Feature and E2 frame clocks differ')
                times = float(z['t0_ms'])+10.*np.arange(len(feat))
                own = own_probabilities(ev)
                reconstructed = own_from_logits(z['logits'])
                difference = np.abs(own-reconstructed)
                heads = {}
                for kind, arm in (('linear','band_d2'),('mlp','band_d2_mlp')):
                    logits = head_logits(HEADS[kind], feat).astype(np.float64)
                    p = softmax(logits/float(HEADS[kind]['temperature']), axis=1)
                    if not np.array_equal(p.argmax(1), logits.argmax(1)):
                        raise RuntimeError('Temperature altered frame argmax')
                    heads[arm] = p
            heads['band_plus_own_d2'] = pooled_probabilities(heads['band_d2'], own, POOL_WEIGHT)
            with np.load(base.OUT / 'tracks' / seed / (audio+'.npz')) as z:
                track = tuple(z[k] for k in ('frames','kinds','periods'))
            decoded = {}
            for arm, p in heads.items():
                m, diag = decode_band(ev, p, track)
                decoded[arm] = m.to_json()
                diagnostics.append(dict(audio=audio, arm=arm, **diag))
            seen[audio] = (ev, times, own, heads, decoded)
            inputs.append(dict(audio=audio, manifest_entry=row, feature_sha256=sha(path),
                e2_sha256=sha(base.PREV / 'evidence' / seed / '009001' / (audio+'.npz')),
                reconstruction_max_abs=float(difference.max()), reconstruction_mean_abs=float(difference.mean()),
                reconstruction_argmax_disagreement=float(np.mean(own.argmax(1)!=reconstructed.argmax(1)))))
        ev, times, own, heads, decoded = seen[audio]
        oracle = oracle_evidence(x['ref'], ev.n_frames)
        probs = dict(heads, baseline=own, own_d2=own,
            oracle_d2=own_probabilities(dataclasses.replace(ev, level=oracle.level)))
        for arm in decoded:
            maps[arm][x['key']] = decoded[arm]
        for arm in ARMS:
            for cell, counts in accuracy(x['ref'], times, probs[arm], x['strata']).items():
                acc.append(dict(seed=seed, panel=x['panel'], rec=x['rec'], key=x['key'], arm=arm, cell=cell, **counts))
    rows, structures = [], []
    for arm in ARMS:
        rows += base.score_record((seed+':'+arm, xs, maps[arm]))
        offset, _ = base.global_offset(xs, maps[arm])
        for x in xs:
            m = MusicMap.from_json(maps[arm][x['key']])
            for aligned, off in ((True,offset),(False,0.)):
                structures.append(dict(seed=seed, panel=x['panel'], rec=x['rec'], key=x['key'], arm=arm,
                    aligned=aligned, offset=float(off), strata=x['strata'], **structural(x['ref'],m,off)))
    path = OUT / seed / 'maps' / (xs[0]['rec']+'.json.gz')
    path.parent.mkdir(exist_ok=True)
    with gzip.open(path, 'wt') as f:
        json.dump(dict(maps=maps, diagnostics=diagnostics), f, default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist())
    return dict(seed=seed, panel=xs[0]['panel'], rec=xs[0]['rec'], rows=rows,
        structures=structures, accuracy=acc, inputs=inputs)


def evaluate(seed, workers):
    check_prereg()
    if not json.loads((OUT / 'control-reproduction.json').read_text())['passed']:
        raise RuntimeError('Controls did not reproduce')
    if seed == 's1' and not (cache_root(seed) / 'DONE').exists():
        raise RuntimeError('Seed 1 cache not DONE')
    groups = defaultdict(list)
    for x in base.items():
        groups[(x['panel'], x['rec'])].append(x)
    rows = manifest(seed)
    audios = {x['item'] for xs in groups.values() for x in xs}
    if audios-set(rows):
        raise RuntimeError(f'Missing {len(audios-set(rows))} panel feature entries')
    selected = {audio:rows[audio] for audio in sorted(audios)}
    path = OUT / seed / 'evaluation.jsonl'
    if path.exists():
        raise RuntimeError('Fresh evaluation output required')
    start = time.monotonic()
    identity = dict(seed=seed, workers=workers, utc=datetime.now(timezone.utc).isoformat(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        prereg=json.loads((OUT / 'prereg-identity.json').read_text()),
        manifest_sha256=sha(cache_root(seed) / 'manifest.jsonl'))
    dump(OUT / seed / 'evaluation-identity.json', identity)
    jobs = [(seed,xs) for _,xs in sorted(groups.items())]
    if workers == 1:
        initialise(seed, selected)
        results = map(job, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=initialise, initargs=(seed,selected))
        results = pool.map(job, jobs, chunksize=1)
    try:
        for i, result in enumerate(results, 1):
            append(path, result)
            if i % 10 == 0:
                print(seed, 'evaluation', i, len(jobs), round(time.monotonic()-start), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    identity.update(seconds=time.monotonic()-start, jobs=len(jobs), finished_utc=datetime.now(timezone.utc).isoformat())
    dump(OUT / seed / 'evaluation-identity.json', identity)
    print(seed, 'evaluation complete', identity['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('seed', choices=SEEDS)
    parser.add_argument('--workers', choices=(1,2), type=int, default=2)
    args = parser.parse_args(); evaluate(args.seed, args.workers)
