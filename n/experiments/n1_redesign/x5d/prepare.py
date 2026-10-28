"""Prepare bounded, recording-weighted feature samples from the fixed probe cohort."""
import argparse
from datetime import datetime, timezone
from common import *
from numpy.lib.format import open_memmap
from bands import own_from_logits


def prepare(seed):
    check_prereg()
    if not json.loads((OUT / 'control-reproduction.json').read_text())['passed']:
        raise RuntimeError('X5 controls failed')
    if seed == 's1' and not (cache_root(seed) / 'DONE').exists():
        raise RuntimeError('Seed 1 cache is not DONE')
    selected = training_rows()
    rows = manifest(seed)
    missing = set(selected) - set(rows)
    if missing:
        raise RuntimeError(f'{seed}: {len(missing)} probe feature entries missing')
    if seed == 's1':
        source = manifest('s0')
        for audio in selected:
            memberships = lambda row: sorted(tuple(sorted(m['charts'])) for m in row['memberships'])
            if memberships(rows[audio]) != memberships(source[audio]):
                raise RuntimeError('Probe chart memberships differ between evidence seeds')
    destination = OUT / seed / 'data'
    destination.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split, allowed in (('fit', (0, 1)), ('cal', (2,))):
        audios = sorted(a for a in selected if bucket(a) in allowed)
        capacity = 512 * len(audios)
        arrays = {name: open_memmap(destination / f'{split}-{name}.npy', mode='w+', dtype=dtype,
                                   shape=(capacity, *shape))
                  for name, dtype, shape in [('x', 'float32', (160,)), ('y', 'float32', (7,)),
                                             ('own', 'float32', (7,)), ('weight', 'float32', ()),
                                             ('owner', 'int32', ())]}
        offset = 0
        inventory = []
        for owner, audio in enumerate(audios):
            row = rows[audio]
            if row['split'] != 'fit_train':
                raise RuntimeError('Probe entry is not a training recording')
            refs = chart_references(row)
            path = cache_root(seed) / row['npz']
            with np.load(path, allow_pickle=False) as z:
                if float(z['fps']) != 100 or z['feat'].shape[1] != 160:
                    raise ValueError('Unexpected frozen feature schema')
                feat = z['feat']
                times = float(z['t0_ms']) + np.arange(len(feat)) * 10.
                y, valid = targets(refs, times)
                ix = np.flatnonzero(valid)
                if not len(ix):
                    raise ValueError('No mapped target frames for training recording')
                n = min(512, len(ix))
                sample = ix[np.floor((np.arange(n)+.5) * len(ix)/n).astype(int)]
                sl = slice(offset, offset+n)
                arrays['x'][sl] = feat[sample].astype(np.float32)
                arrays['y'][sl] = y[sample]
                arrays['own'][sl] = own_from_logits(z['logits'][sample])
                arrays['weight'][sl] = 1./n
                arrays['owner'][sl] = owner
                offset += n
                inventory.append(dict(audio=audio, samples=n, mapped_frames=int(valid.sum()),
                    ambiguous_frames=int((valid & (np.max(y, axis=1)<1)).sum()),
                    feature_sha256=sha(path), manifest_entry=row,
                    chart_sha256={p:sha(p) for ref in refs for p in ref.meta['difficulties']}))
            if (owner+1) % 100 == 0:
                print(seed, split, owner+1, len(audios), flush=True)
        for array in arrays.values():
            array.flush()
        counts[split] = dict(recordings=len(audios), samples=offset, capacity=capacity)
        dump(destination / f'{split}-inventory.json', inventory)
    # Standardisation is weighted only over the fitted recording sample.
    n = counts['fit']['samples']
    xx = np.load(destination / 'fit-x.npy', mmap_mode='r')
    ww = np.load(destination / 'fit-weight.npy', mmap_mode='r')
    total = 0.; first = np.zeros(160); second = np.zeros(160)
    for lo in range(0, n, 16384):
        x = np.asarray(xx[lo:min(n,lo+16384)], np.float64)
        w = np.asarray(ww[lo:min(n,lo+16384)], np.float64)
        total += w.sum(); first += np.sum(w[:, None]*x, axis=0)
        second += np.sum(w[:, None]*x*x, axis=0)
    mean = first/total
    sd = np.maximum(np.sqrt(np.maximum(second/total-mean*mean, 0.)), 1e-4)
    np.savez(destination / 'normalizer.npz', mean=mean.astype(np.float32), sd=sd.astype(np.float32))
    dump(destination / 'metadata.json', dict(seed=seed, counts=counts,
        exclusions_sha256=sha(OUT / 'exclusions.json'),
        manifest_sha256=sha(cache_root(seed) / 'manifest.jsonl'), utc=datetime.now(timezone.utc).isoformat()))
    print(seed, 'prepared', counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('seed', choices=SEEDS)
    prepare(parser.parse_args().seed)
