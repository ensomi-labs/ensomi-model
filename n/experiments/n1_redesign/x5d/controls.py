"""Reproduce the original own-head and oracle D2 maps before fitting new heads."""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import multiprocessing as mp
import time

from common import *
from decoder_measurements import CFG
from ensomi_n.v0.evaluate import oracle_evidence, v0_evidence
from level import decode_level


def job(args):
    seed, xs = args
    with gzip.open(base.OUT / 'maps' / seed / (xs[0]['rec'] + '.json.gz'), 'rt') as f:
        old = json.load(f)['maps']
    maps = {a: {} for a in ('baseline', 'own_d2', 'oracle_d2')}
    comparisons = 0
    for x in xs:
        ev = v0_evidence(base.evidence(seed, x['item']))
        oracle = oracle_evidence(x['ref'], ev.n_frames)
        with np.load(base.OUT / 'tracks' / seed / (x['item'] + '.npz')) as z:
            track = tuple(z[k] for k in ('frames', 'kinds', 'periods'))
        maps['baseline'][x['key']] = base.saved(seed, x['item'])
        for arm in ('own_d2', 'oracle_d2'):
            m, _ = decode_level(ev, oracle, arm, track, CFG)
            maps[arm][x['key']] = m.to_json()
        for arm in maps:
            for field in ('beats', 'segments', 'downbeats'):
                if maps[arm][x['key']].get(field) != old[arm][x['key']].get(field):
                    raise RuntimeError(f'X5 reproduction failed: {seed} {arm} {x["key"]} {field}')
                comparisons += 1
    path = OUT / 'controls' / seed / (xs[0]['rec'] + '.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as f:
        json.dump(maps, f)
    return dict(seed=seed, rec=xs[0]['rec'], comparisons=comparisons, sets=len(xs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'controls.jsonl'
    if path.exists():
        raise RuntimeError('Fresh controls output required')
    groups = defaultdict(list)
    for x in base.items():
        groups[(x['panel'], x['rec'])].append(x)
    jobs = [(s, xs) for _, xs in sorted(groups.items()) for s in SEEDS]
    start = time.monotonic()
    with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as pool:
        for i, result in enumerate(pool.map(job, jobs), 1):
            append(path, result)
            if i % 40 == 0:
                print('controls', i, len(jobs), round(time.monotonic()-start), flush=True)
    dump(OUT / 'control-reproduction.json', dict(passed=True, jobs=len(jobs),
        compared_maps=3*len(base.items())*2, seconds=time.monotonic()-start,
        utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
