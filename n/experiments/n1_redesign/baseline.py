"""Reproduce cached E1 maps and scores before permitting measurements."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime, timezone

from base import *
from decoder_measurements import checked_baseline
from ensomi_n.v0.evaluate import v0_evidence

METRICS = ['ready', 'r5', 'p5', 'n_bars_same', 'n_bars_start', 'n_level',
           'n_reset_recall', 'n_boundary_recall', 'guarded_ready']


def recording_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['panel'], r['cell'], r['arm'], r['rec'])].append(r)
    out = []
    for (panel, cell, arm, rec), rs in sorted(groups.items()):
        row = dict(panel=panel, cell=cell, arm=arm, rec=rec)
        for k in rs[0]:
            if k in row or k == 'key':
                continue
            a = np.array([r[k] for r in rs], float)
            row[k] = float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else math.nan
        row['ready'] = float(row['r5'] >= .9)
        row['guarded_ready'] = float(row['r5'] >= .9 and row['p5'] >= .9)
        out.append(row)
    return out


def job(args):
    seed, xs = args
    maps, reps = {}, []
    seen = {}
    for x in xs:
        audio = x['item']
        if audio not in seen:
            old = saved(seed, audio)
            pm, diag, track, rep = checked_baseline(v0_evidence(evidence(seed, audio)), old)
            # Require exact serialized timing, beyond stage 0's one-ms guard.
            now = pm.to_json()
            for k in ('beats', 'downbeats', 'segments'):
                if now.get(k) != old.get(k):
                    raise RuntimeError(f'Exact baseline mismatch: {seed} {audio} {k}')
            path = OUT / 'tracks' / seed / (audio + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, frames=track[0], kinds=track[1], periods=track[2])
            seen[audio] = now
            reps.append(dict(item=audio, **rep))
        maps[x['key']] = seen[audio]
    rows = score_record((seed + ':baseline', xs, maps))
    return dict(seed=seed, rec=xs[0]['rec'], panel=xs[0]['panel'], rows=rows, reproductions=reps)


def main():
    start = time.monotonic()
    groups = defaultdict(list)
    for x in items():
        groups[(x['panel'], x['rec'])].append(x)
    jobs = [(s, xs) for _, xs in sorted(groups.items()) for s in ('s0', 's1')]
    path = OUT / 'baseline.jsonl'
    if path.exists():
        raise RuntimeError('Fresh baseline output required')
    with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as pool:
        for i, result in enumerate(pool.map(job, jobs, chunksize=1), 1):
            append(path, result)
            if i % 20 == 0:
                print('baseline', i, len(jobs), round(time.monotonic()-start), flush=True)
    rows = [r for line in path.read_text().splitlines() for r in json.loads(line)['rows']]
    recs = recording_rows(rows)
    old = json.loads((PREV / 'e1-summary.json').read_text())
    differences = []
    compared = 0
    for r in recs:
        before = old[r['panel']][r['cell']][r['arm'].split(':')[0]+'-009001']['recordings'][r['rec']]
        for k in METRICS:
            if k not in before:
                continue
            compared += 1
            if not np.isclose(r[k], before[k], atol=1e-9, rtol=0, equal_nan=True):
                differences.append(dict(panel=r['panel'], cell=r['cell'], rec=r['rec'],
                                        arm=r['arm'], metric=k, before=before[k], after=r[k]))
    dump(OUT / 'baseline-recordings.json', recs)
    dump(OUT / 'baseline-reproduction.json', dict(jobs=len(jobs), compared_scores=compared,
        differences=differences, seconds=time.monotonic()-start,
        utc=datetime.now(timezone.utc).isoformat(), passed=not differences))
    if differences:
        raise RuntimeError('Baseline score mismatch; treatments prohibited')
    print('Exact baseline maps and scores reproduced', compared, flush=True)


if __name__ == '__main__':
    main()
