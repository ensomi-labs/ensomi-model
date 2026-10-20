"""N1 v0.1: where v0's training throughput goes (data, synchronisation, the memory path), on the mac.

v0 trained at a median 12.6 crops/s; rung 0 measured NE-S at 27.3 crops/s at batch 16, but on random
tensors with a two-term loss and no data loading. This separates, at v0's batch (8 songs x 2 crops):

1. ``data``: `Sampler.batch` alone on one thread, split into the memory input, the crop read and the
   dense labels; then batches per second with 1, 2 and 4 threads drawing at once (the GIL's share);
2. ``model``: training steps on four pre-drawn real batches held on the device, cycled:
   ``exact`` (the training step as written: twelve loss scalars read to the CPU, a synchronise per
   step), ``nosync`` (the same arithmetic, the scalars left on the device, one synchronise at the end),
   ``nomem`` (``nosync`` without the song memory) and ``maxmem`` (``nosync`` with every memory input
   padded to the 480 s cap);
3. ``loop``: the training loop as written (prefetch thread, transfer, exact step) on fresh batches,
   with the time spent waiting for a batch.

    PYTHONPATH=<src>/n/src .venv/bin/python <src>/scripts/n1/v01_throughput.py --data D --refs RUN/refs.json --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import threading
import time

import numpy as np


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--refs', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bulk', default='artifacts/bulk/n1-cache')
    p.add_argument('--device', default='mps')
    p.add_argument('--seconds', type=float, default=40.0)
    a = p.parse_args(argv)

    import torch
    from ensomi_n.run.v0 import LABELS, _index, _rows
    from ensomi_n.v0 import data as D
    from ensomi_n.v0 import model as Mm
    from ensomi_n.v0 import train as T
    from ensomi_n.v0.model import V0

    out = dict(torch=torch.__version__, platform=platform.platform(), device=a.device)
    dev = a.device
    sync = (lambda: torch.mps.synchronize()) if dev == 'mps' else (lambda: None)

    def dump():
        Path(a.out).write_text(json.dumps(out, indent=1))

    bulk = Path(a.bulk)
    t0 = time.time()
    rows = [r for r in _rows(Path(a.data)) if r['role'] == 'train' and r.get('keep')]
    songs = D.load_songs(rows, _index(bulk), bulk, bulk / LABELS)
    out['songs'], out['load_songs_s'] = len(songs), round(time.time() - t0, 1)
    pos_weight = json.load(open(a.refs))['pos_weight']
    n_songs, k = T.SONGS_PER_BATCH, T.CROPS_PER_SONG

    # 1. data on one thread, with its parts
    parts = dict(memo=0.0, mel=0.0, labels=0.0)
    orig = dict(memo=D.memo_input, mel=D.stretch_crop, labels=D.dense_v0)

    def timed(name):
        def f(*args, **kw):
            t = time.perf_counter()
            r = orig[name](*args, **kw)
            parts[name] += time.perf_counter() - t
            return r
        return f
    D.memo_input, D.stretch_crop, D.dense_v0 = timed('memo'), timed('mel'), timed('labels')
    sampler = D.Sampler(songs, seed=1000, crops_per_song=k)
    times, memo_len = [], []
    for _ in range(16):
        t = time.perf_counter()
        b = sampler.batch(n_songs)
        times.append(time.perf_counter() - t)
        memo_len.append(int(b[2].shape[1]))
    D.memo_input, D.stretch_crop, D.dense_v0 = orig['memo'], orig['mel'], orig['labels']
    tot = sum(times)
    out['data_one_thread'] = dict(batch_s=round(tot / len(times), 4), crops_per_s=round(n_songs * k * len(times) / tot, 1),
                                  share={n: round(v / tot, 3) for n, v in parts.items()},
                                  memo_frames=memo_len)
    dump()

    def threads_rate(n_thr: int, secs: float = 12.0) -> float:
        count, stop = [0], [False]
        lock = threading.Lock()

        def run(seed):
            s = D.Sampler(songs, seed=seed, crops_per_song=k)
            while not stop[0]:
                s.batch(n_songs)
                with lock:
                    count[0] += 1
        ths = [threading.Thread(target=run, args=(2000 + i,), daemon=True) for i in range(n_thr)]
        t = time.perf_counter()
        for th in ths:
            th.start()
        time.sleep(secs)
        stop[0] = True
        for th in ths:
            th.join()
        return count[0] * n_songs * k / (time.perf_counter() - t)
    out['data_threads_crops_per_s'] = {str(n): round(threads_rate(n), 1) for n in (1, 2, 4)}
    dump()

    # 2. model steps on pre-drawn batches
    pre = [D.Sampler(songs, seed=3000 + i, crops_per_song=k).batch(n_songs) for i in range(4)]
    dev_batches = []
    for x, tg, memo, mask, owner in pre:
        xt, tt, mt, kt = D.to_torch(x, tg, memo, mask, dev)
        dev_batches.append((xt, tt, mt, kt, torch.from_numpy(owner).to(dev)))

    def pad_max(b):
        xt, tt, mt, kt, ow = b
        cap = int(D.MAX_MEMO_S * 25)
        cap += (-cap) % D.MEMO_BUCKET
        m2 = torch.zeros((mt.shape[0], cap, mt.shape[2]), device=dev)
        k2 = torch.zeros((kt.shape[0], cap), dtype=torch.bool, device=dev)
        m2[:, :mt.shape[1]], k2[:, :kt.shape[1]] = mt, kt
        return xt, tt, m2, k2, ow

    def steps(variant: str, secs: float) -> dict:
        torch.manual_seed(0)
        model = V0().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        model.train()
        batches = [pad_max(b) for b in dev_batches] if variant == 'maxmem' else dev_batches
        nosync = variant != 'exact'
        if nosync:
            Mm.float = lambda v: v                  # loss scalars stay on the device
        n, t_start = 0, None
        try:
            for i in range(10 ** 6):
                if i == 3:
                    sync()
                    t_start = time.perf_counter()
                xt, tt, mt, kt, ow = batches[i % len(batches)]
                if variant == 'nomem':
                    o = model.heads(model.encoder(xt, None, use_memory=False))
                else:
                    mem = model.encoder.song_memory(mt, kt)[ow]
                    o = model.heads(model.encoder(xt, mem))
                loss, _ = V0.loss(o, tt, pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if not nosync:
                    sync()
                    float(loss.detach())
                if t_start is not None:
                    n += 1
                    if n % 5 == 0:
                        sync()
                        if time.perf_counter() - t_start > secs:
                            break
            sync()
            el = time.perf_counter() - t_start
        finally:
            if nosync:
                del Mm.float
        res = dict(steps=n, step_s=round(el / n, 4), crops_per_s=round(n * n_songs * k / el, 2))
        if dev == 'mps':
            res['driver_gb'] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
            torch.mps.empty_cache()
        return res
    out['model'] = {}
    for v in ('exact', 'nosync', 'nomem', 'maxmem'):
        out['model'][v] = steps(v, a.seconds)
        dump()

    # 3. the loop as written, on fresh batches
    torch.manual_seed(0)
    model = V0().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    model.train()
    pf = T.Prefetch(D.Sampler(songs, seed=4000, crops_per_song=k), n_songs)
    wait, n, t_start = 0.0, 0, None
    try:
        for i in range(10 ** 6):
            if i == 3:
                t_start, wait = time.perf_counter(), 0.0
            tw = time.perf_counter()
            x, tg, memo, mask, owner = pf.get()
            wait += time.perf_counter() - tw
            x, tg, memo, mask = D.to_torch(x, tg, memo, mask, dev)
            mem = model.encoder.song_memory(memo, mask)[torch.from_numpy(owner).to(dev)]
            o = model.heads(model.encoder(x, mem))
            loss, _ = V0.loss(o, tg, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sync()
            float(loss.detach())
            if t_start is not None:
                n += 1
                if time.perf_counter() - t_start > a.seconds:
                    break
    finally:
        pf.close()
    el = time.perf_counter() - t_start
    out['loop'] = dict(steps=n, crops_per_s=round(n * n_songs * k / el, 2), wait_share=round(wait / el, 3))
    dump()
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
