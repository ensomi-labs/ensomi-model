"""Whole-song inference for N1 v0: overlapping 20 s windows under one song memory.

The song memory is computed once from the whole song's memory input (as in training, songs longer
than 8 minutes give it the 8 minutes around the song's middle). The song is cut into windows of the
training crop length with a hop of half a window; each frame takes the output of the window in which
it lies farthest from an edge. Outputs are numpy arrays per frame at 100 fps.
"""
from __future__ import annotations

import numpy as np

from .data import CROP, memo_input
from .songs import MEMO_HOP


def windows(n: int, crop: int = CROP, hop: int = CROP // 2) -> list[int]:
    if n <= crop:
        return [0]
    starts = list(range(0, n - crop, hop)) + [n - crop]
    return sorted(set(starts))


def song_outputs(model, song, device: str, *, memory: str = 'song', other_song=None, batch: int = 8) -> dict:
    """Activations for one `data.Song` (dict of [T] or [T, 4] arrays).

    ``memory``: ``'song'`` (its own memory), ``'zero'`` (cross-attention to the memory skipped) or
    ``'other'`` (``other_song``'s memory).
    """
    import torch
    from .model import V0
    mel = np.load(song.mel, mmap_mode='r')
    n = len(mel)
    T = n + (-n) % 4
    x = np.zeros((T, mel.shape[1]), dtype=np.float32)
    x[:n] = (np.asarray(mel, dtype=np.float32) - song.mean) / song.std
    x[n:] = x[:n].min() if n else 0.0
    starts = windows(T)
    with torch.inference_mode():
        mem = None
        if memory in ('song', 'other') and model.encoder.memory is not None:
            src = song if memory == 'song' else other_song
            m = memo_input(src, 1.0)
            mt = torch.from_numpy(m[None]).to(device)
            mem = model.encoder.song_memory(mt, torch.ones(mt.shape[:2], dtype=torch.bool, device=device))
        acts_all = None
        best = np.full(T, -1.0)
        for i in range(0, len(starts), batch):
            chunk = starts[i:i + batch]
            xb = torch.from_numpy(np.stack([x[s:s + min(CROP, T)] for s in chunk])).to(device)
            mb = None if mem is None else mem.expand(len(chunk), -1, -1)
            out = model.heads(model.encoder(xb, mb, use_memory=mem is not None))
            acts = {k: v.float().cpu().numpy() for k, v in V0.activations(out).items()}
            if acts_all is None:
                acts_all = {k: np.zeros((T,) + v.shape[2:], dtype=np.float32) for k, v in acts.items()}
            L = xb.shape[1]
            centre = np.minimum(np.arange(L), L - 1 - np.arange(L)).astype(np.float64)
            for j, s in enumerate(chunk):
                take = centre > best[s:s + L]
                idx = np.flatnonzero(take) + s
                for k, v in acts.items():
                    acts_all[k][idx] = v[j][take]
                best[idx] = centre[take]
    return {k: v[:n] for k, v in acts_all.items()}
