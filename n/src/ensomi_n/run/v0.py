"""N1 v0 on the mac: curation, caches, references, training jobs and evaluation.

From the ensomi-model project root with ``PYTHONPATH=<src>/n/src``::

    python -m ensomi_n.run.v0 curate --data D --r0 R0 --src <src>     # curated.json, curation-result.json
    python -m ensomi_n.run.v0 cache  --data D                          # Mel where missing, song stats, memory input
    python -m ensomi_n.run.v0 refs   --data D --run RUN                # kill-rule scale, positive weights
    python -m ensomi_n.run.v0 train  --data D --run RUN --seed 0 --total-min 200 --job-min 50
    python -m ensomi_n.run.v0 eval   --data D --run RUN ...            # see ensomi_n.v0.evaluate

``D`` is the data run directory (synced records of curation and caching), ``RUN`` a training run
directory; bulk outputs go under ``--bulk`` (caches) and ``--bulk-runs`` (checkpoints).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

LABELS = 'labels-v0'


def _progress(path: Path):
    def write(rec: dict):
        with open(path / 'progress.jsonl', 'a') as f:
            f.write(json.dumps(dict(rec, t=round(time.time(), 1))) + '\n')
    return write


def _dump(obj, path: Path) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def cmd_curate(a) -> None:
    from ..v0 import curate
    data, bulk = Path(a.data), Path(a.bulk)
    data.mkdir(parents=True, exist_ok=True)
    res = curate.build(root=Path(a.root), src_root=Path(a.src), r0=Path(a.r0), run=data,
                       labels_dir=bulk / LABELS, processes=a.processes, progress=_progress(data))
    rows = res['rows']
    for r in rows:
        r.pop('trace', None)
    _dump(rows, data / 'curated.json')
    summ = curate.summarise(res)
    summ['inputs'] = res['report_inputs']
    _dump(summ, data / 'curation-result.json')
    _progress(data)(dict(step='curate_done', kept=summ['kept']))
    print(json.dumps({k: summ[k] for k in ('candidates', 'kept', 'excluded_any_reason', 'eval')}, indent=1))


def _rows(data: Path) -> list[dict]:
    return json.load(open(data / 'curated.json'))


def cmd_cache(a) -> None:
    from ..v0 import songs
    data, bulk = Path(a.data), Path(a.bulk)
    rows = [r for r in _rows(data) if r.get('ok') and (r.get('keep') or r['role'] != 'train')]
    res = songs.build(rows, root=Path(a.root), bulk=bulk, run=data, processes=a.processes, progress=_progress(data))
    _dump(res['index'], bulk / 'song-index-v0.json')
    _dump(dict(cache=res['cache'], n_index=len(res['index']), n_errors=res['n_errors'], errors=res['errors']),
          data / 'cache-result.json')
    _progress(data)(dict(step='cache_done', n=len(res['index']), errors=res['n_errors']))


def _index(bulk: Path) -> dict:
    return json.load(open(bulk / 'song-index-v0.json'))


def _probe_sets(data: Path, bulk: Path, root: str, n_crop_songs: int = 150, n_m1: int = 40):
    """Probe songs (val-n1 random), fixed crops, and M1 items for the S0 and hard groups."""
    from ..labels import read_set
    from .. import probe
    from ..v0.data import fixed_crops, load_songs
    rows = [r for r in _rows(data) if r['role'] == 'val_random' and r.get('ok') and not r.get('reader_flags')]
    rows.sort(key=lambda r: r['item'])
    songs = load_songs(rows, _index(bulk), bulk, bulk / LABELS)
    by_item = {r['item']: r for r in rows}
    songs = songs[:n_crop_songs]
    crops = fixed_crops(songs, per_song=1)
    items = []
    counts = {'S0': 0, 'hard': 0}
    charts = {r['item']: r for r in rows}
    split_charts = {}
    for s in songs:
        r = by_item[s.item]
        group = 'S0' if 'S0' in (r.get('strata') or []) else ('hard' if r.get('hard') else None)
        if group is None or counts[group] >= n_m1:
            continue
        split_charts[s.item] = r
        items.append(dict(song=s, group=group, item=s.item))
        counts[group] += 1
    return songs, crops, items, counts


def _charts_of(r0: Path) -> dict:
    split = json.load(open(r0 / 'split.json'))
    out = {}
    for name, lst in split.items():
        for it in lst:
            out.setdefault((name, it['audio_sha256']), it['charts'])
    return out


def cmd_refs(a) -> None:
    from .. import probe
    from ..labels import read_set
    from ..v0.data import Sampler, load_songs
    data, bulk, run = Path(a.data), Path(a.bulk), Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    songs, crops, items, counts = _probe_sets(data, bulk, a.root)
    charts = _charts_of(Path(a.r0))
    f0 = sorted(bulk.glob('bt-final0-*'))
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
    chance, m1_f0 = {}, {}
    for it in items:
        ref = read_set([os.path.join(a.root, c) for c in charts[('val_random', it['item'])]], key=it['item'])
        w = probe.beat_windows(ref)
        it['windows'] = w
        chance.setdefault(it['group'], []).append(probe.chance(w))
        p = f0[0] / f"{it['item']}.npz" if f0 else None
        if p is not None and p.exists():
            z = np.load(p)
            m1_f0.setdefault(it['group'], []).append(probe.m1(probe.Activation(sig(z['beat']), -30.0, 20.0), w))
    train_rows = [r for r in _rows(data) if r['role'] == 'train' and r.get('keep')]
    tr = load_songs(train_rows[:400], _index(bulk), bulk, bulk / LABELS)
    s = Sampler(tr, seed=12345)
    rates: dict[str, list] = {}
    for _ in range(12):
        _, tg, _, _, _ = s.batch(8)
        m = tg['mask']
        for k in ('beat', 'half', 'down', 'bound'):
            rates.setdefault(k, []).append(float((tg[k][m] > 0.75).mean()))
    pos = {k: float(np.clip(np.sqrt((1 - np.mean(v)) / max(np.mean(v), 1e-4)), 1.0, 20.0)) for k, v in rates.items()}
    out = dict(probe_counts=counts, probe_crops=len(crops), probe_songs=len(songs),
               probe_s0_chance=float(np.mean(chance.get('S0', [np.nan]))),
               probe_hard_chance=float(np.mean(chance.get('hard', [np.nan]))),
               probe_s0_final0=float(np.mean(m1_f0.get('S0', [np.nan]))),
               probe_hard_final0=float(np.mean(m1_f0.get('hard', [np.nan]))),
               final0_n={g: len(v) for g, v in m1_f0.items()}, label_rates={k: float(np.mean(v)) for k, v in rates.items()},
               pos_weight=pos, bt_final0_dir=str(f0[0]) if f0 else None)
    (run / 'refs.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_train(a) -> None:
    import resource
    from ..labels import read_set
    from .. import probe
    from ..v0 import train as T
    from ..v0.data import load_songs
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 4096), hard), hard))
    except (ValueError, OSError):
        pass
    data, bulk, run = Path(a.data), Path(a.bulk), Path(a.run)
    refs = json.load(open(run / 'refs.json'))
    rows = [r for r in _rows(data) if r['role'] == 'train' and r.get('keep')]
    if a.sources:
        rows = [r for r in rows if r['source'] in a.sources.split(',')]
    songs = load_songs(rows, _index(bulk), bulk, bulk / LABELS)
    psongs, crops, items, _ = _probe_sets(data, bulk, a.root)
    charts = _charts_of(Path(a.r0))
    for it in items:
        ref = read_set([os.path.join(a.root, c) for c in charts[('val_random', it['item'])]], key=it['item'])
        it['windows'] = probe.beat_windows(ref)
    cfg = dict(seed=a.seed, total_min=a.total_min, job_min=a.job_min, songs=len(songs),
               sources=sorted({s.source for s in songs}), weight_sum=round(sum(s.weight for s in songs), 1),
               model_kw=json.loads(a.model_kw) if a.model_kw else {}, src=a.src_sha, data=str(data),
               peak_lr=T.PEAK_LR, warmup=T.WARMUP, batch=f'{T.SONGS_PER_BATCH}x{T.CROPS_PER_SONG}', crop_s=20.0)
    if not (run / 'config.json').exists():
        (run / 'config.json').write_text(json.dumps(cfg, indent=1))
    ck = Path(a.bulk_runs) / run.name
    status = T.train(songs=songs, probe_crops=crops, probe_songs=psongs, probe_items=items, refs=refs, run=run,
                     ck_dir=ck, seed=a.seed, total_min=a.total_min, job_min=a.job_min, device=a.device,
                     model_kw=cfg['model_kw'])
    print(json.dumps(status, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog='python -m ensomi_n.run.v0')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('curate', 'cache', 'refs', 'train', 'eval'):
        s = sub.add_parser(name)
        s.add_argument('--data', required=True)
        s.add_argument('--run', default='')
        s.add_argument('--r0', default='artifacts/n1-night-20261018/runs/r0-20261019a')
        s.add_argument('--bulk', default='artifacts/bulk/n1-cache')
        s.add_argument('--bulk-runs', default='artifacts/bulk/n1-runs')
        s.add_argument('--root', default='.')
        s.add_argument('--src', default='.')
        s.add_argument('--src-sha', default='')
        s.add_argument('--processes', type=int, default=8)
        s.add_argument('--device', default='mps')
        if name == 'train':
            s.add_argument('--seed', type=int, default=0)
            s.add_argument('--total-min', type=float, default=200.0)
            s.add_argument('--job-min', type=float, default=50.0)
            s.add_argument('--sources', default='')
            s.add_argument('--model-kw', default='')
        if name == 'eval':
            s.add_argument('--arms', default='v0,final0,final1,final2,r2,oracle')
            s.add_argument('--checkpoints', default='last3')
            s.add_argument('--panels', default='panel,val_hard')
            s.add_argument('--out', default='')
            s.add_argument('--decode-kw', default='')
            s.add_argument('--ablation', action='store_true')
            s.add_argument('--limit', type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == 'eval':
        from ..v0 import evaluate
        evaluate.main(a)
        return
    dict(curate=cmd_curate, cache=cmd_cache, refs=cmd_refs, train=cmd_train)[a.cmd](a)


if __name__ == '__main__':
    main()
