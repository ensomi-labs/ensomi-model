"""Memory-free inference with v0.1's exact window stitching and G1's event scorer."""
import argparse
import json
import os
import time
import numpy as np
import torch
from ensomi_n.v0.data import CROP, Song
from ensomi_n.v0.infer import windows
from ..evaluate import head_rows
from .common import OUT, G1, dump, disk_guard, check_prereg
from .train import EventModel


def song_logits(model, song, device='mps', batch=8):
    mel = np.load(song.mel, mmap_mode='r')
    n = len(mel)
    total = n + (-n) % 4
    x = np.zeros((total, mel.shape[1]), np.float32)
    x[:n] = (np.asarray(mel, np.float32) - song.mean) / song.std
    x[n:] = x[:n].min() if n else 0.
    starts = windows(total)
    best = np.full(total, -1.)
    out = np.empty((total, 7), np.float32)
    with torch.inference_mode():
        for i in range(0, len(starts), batch):
            chunk = starts[i:i+batch]
            xb = torch.from_numpy(np.stack([x[s:s+min(CROP, total)] for s in chunk])).to(device)
            y = model(xb).float().cpu().numpy()
            length = y.shape[1]
            centre = np.minimum(np.arange(length), length-1-np.arange(length))
            for j, start in enumerate(chunk):
                take = centre > best[start:start+length]
                ix = np.flatnonzero(take) + start
                out[ix], best[ix] = y[j,take], centre[take]
    if (best < 0).any() or not np.isfinite(out).all():
        raise ValueError('Inference left uncovered frames or non-finite logits')
    return out[:n]


def infer(seed):
    torch.set_num_threads(1)
    os.nice(10)
    check_prereg()
    name = f'E-s{seed}'
    state = torch.load(OUT/'runs'/name/'final.pt', map_location='cpu', weights_only=True)
    model = EventModel()
    model.load_state_dict(state['model'])
    model.eval().to('mps')
    songs = json.loads((G1/'inference-songs.json').read_text())
    shas = sorted({r['audio_sha256'] for r in json.loads((G1/'cohorts.json').read_text())['panels']})
    dest = OUT/'eval'/name
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frames = 0
    for i, sha in enumerate(shas, 1):
        song = Song(**songs[sha])
        y = song_logits(model, song)
        if y.shape != (song.frames, 7):
            raise ValueError('Panel frame clock changed')
        np.savez_compressed(dest/f'{sha}.npz', new_logits=y, t0_ms=0., fps=100.)
        frames += len(y)
        if i % 16 == 0:
            torch.mps.empty_cache()
        if i % 25 == 0 or i == len(shas):
            disk_guard()
            print(json.dumps(dict(event='inference', run=name, done=i, total=len(shas), wall_s=time.monotonic()-started)), flush=True)
    dump(dest/'inference.json', dict(inputs=len(shas), frames=frames, wall_s=time.monotonic()-started))


def score(seed):
    name = f'E-s{seed}'
    def load(sha):
        with np.load(OUT/'eval'/name/f'{sha}.npz') as z:
            return z['new_logits']
    head_rows(name, seed, logits_loader=load, output_path=OUT/'heads'/f'{name}.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['infer', 'score'])
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    (infer if a.mode=='infer' else score)(a.seed)
