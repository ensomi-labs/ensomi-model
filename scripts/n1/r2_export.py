"""R2 as exported, from stored BeatThis ``final0`` logits, for N1's panels (R side).

For each item, ``audio_rows.audio.Listener.listen`` runs unchanged except that BeatThis's forward
pass is replaced by the item's stored logits (from ``bt_logits.py``); ``audio_rows.grid.corrected_grid``
then gives the red lines R2 exports (offset, canonical beat length, meter 4). Output per item:
``<out>/<audio sha256>.json`` with the red lines, the song length, the beat lines (each red line's grid up to
the next, the first extended back to 0, the last to the song's end) and the bar starts (every 4 beats from
each red line). Resumable and time-capped; ``--processes`` workers on CPU.

    PYTHONPATH=src:<src>/n/src .venv/bin/python <src>/scripts/n1/r2_export.py --split <run>/split.json \
        --subset panel,val_hard --logits artifacts/bulk/n1-cache/bt-final0-<sha> --out artifacts/bulk/n1-cache/r2-export
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


class _StoredFrames:
    def __init__(self):
        self.logits = None

    def __call__(self, wave, sr):
        import torch
        b, d = self.logits
        return torch.from_numpy(b), torch.from_numpy(d)


def beats_from_seg(seg: np.ndarray, song_ms: float) -> tuple[np.ndarray, np.ndarray]:
    beats, downs = [], []
    for k, (off, length, meter) in enumerate(seg):
        end = seg[k + 1][0] if k + 1 < len(seg) else song_ms
        j0 = int(np.floor(-off / length)) if k == 0 else 0
        j = np.arange(j0, int(np.ceil((end - off) / length)))
        t = off + j * length
        keep = t < end - 1e-6
        beats.append(t[keep])
        downs.append((j[keep] % int(meter)) == 0)
    return np.concatenate(beats), np.concatenate(downs)


def export_one(args) -> dict:
    item, root, logits_dir, out = args
    target = Path(out) / f"{item['audio_sha256']}.json"
    if target.exists():
        return dict(item=item['item'], status='present')
    try:
        from ensomi_model.audio_rows.audio import Listener
        from ensomi_model.audio_rows.grid import corrected_grid
        from beat_this.inference import load_audio
        z = np.load(Path(logits_dir) / f"{item['audio_sha256']}.npz")
        listener = Listener.__new__(Listener)
        listener.load = load_audio
        listener.frames = _StoredFrames()
        listener.frames.logits = (z['beat'].astype(np.float32), z['down'].astype(np.float32))
        t0 = time.perf_counter()
        song = listener.listen(os.path.join(root, item['audio_rel']))
        _, seg = corrected_grid(song.segments, song.song_ms)
        beats, downs = beats_from_seg(np.asarray(seg), song.song_ms)
        rec = dict(item=item['item'], audio_rel=item['audio_rel'], song_ms=song.song_ms, red_lines=np.asarray(seg).tolist(),
                   raw_segments=np.asarray(song.segments).tolist(), beats=beats.tolist(),
                   downbeats=np.flatnonzero(downs).tolist(), seconds=round(time.perf_counter() - t0, 2))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, target)
        return dict(item=item['item'], status='built', n_lines=len(seg), seconds=rec['seconds'])
    except Exception as exc:
        return dict(item=item['item'], status='error', error=f'{type(exc).__name__}: {exc}',
                    trace=traceback.format_exc()[-600:])


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True)
    p.add_argument('--subset', required=True)
    p.add_argument('--logits', required=True, help='bt-final0-<sha> directory')
    p.add_argument('--out', required=True)
    p.add_argument('--root', default='.')
    p.add_argument('--processes', type=int, default=4)
    p.add_argument('--result', required=True)
    p.add_argument('--max-minutes', type=float, default=50.0)
    a = p.parse_args(argv)
    split = json.load(open(a.split))
    items, seen = [], set()
    for s in a.subset.split(','):
        for it in split[s]:
            if it['audio_sha256'] not in seen and (Path(a.logits) / f"{it['audio_sha256']}.npz").exists():
                seen.add(it['audio_sha256'])
                items.append(it)
    import multiprocessing as mp
    t0 = time.time()
    rows = []
    args = [(it, a.root, a.logits, a.out) for it in items]
    with mp.get_context('spawn').Pool(a.processes) as pool:
        for r in pool.imap_unordered(export_one, args):
            rows.append(r)
            if time.time() - t0 > 60 * a.max_minutes:
                pool.terminate()
                break
    res = dict(n=len(items), counts={s: sum(r['status'] == s for r in rows) for s in ('built', 'present', 'error')},
               errors=[r for r in rows if r['status'] == 'error'][:30], wall_s=round(time.time() - t0, 1),
               logits=a.logits, out=a.out)
    Path(a.result).parent.mkdir(parents=True, exist_ok=True)
    Path(a.result).write_text(json.dumps(res, indent=1))
    print(json.dumps(res['counts']), res['wall_s'])


if __name__ == '__main__':
    main()
