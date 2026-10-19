"""Crops for training: recording sampler, stretch and gain/tilt augmentation, dense targets (design §3.3, §3.4).

Each draw picks a recording (hard recordings by the interim flags weigh ``hard_weight``), a stretch
``rate`` in [0.88, 1.12], and a crop whose centre lies in the mapped range (so at most half of it is
outside), then builds the standardised log-Mel crop and its targets. Per-song standardisation uses the
whole song's mean and standard deviation, computed once when the song is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..augment import gain_tilt, stretch_crop
from ..features.cache import mel_path
from ..targets import RecLabels, dense

KEYS_F = ('beat', 'half', 'down', 'beat_off', 'half_off', 'down_off', 'weight', 'phase', 'logp', 'bar_phase')
KEYS_B = ('beat_offmask', 'half_offmask', 'down_offmask', 'mask', 'phase_mask', 'bar_mask')


@dataclass
class Song:
    item: str
    path: str
    mean: float
    std: float
    labels: RecLabels
    hard: bool

    @property
    def mel(self) -> np.ndarray:
        """The cached log-Mel, memory-mapped on each access and released with the array.

        A memory map holds a file descriptor, and macOS allows 256 per process by default, so 1,600
        songs cannot all stay mapped (the first mac smoke failed with "Too many open files").
        """
        return np.load(self.path, mmap_mode='r')


def open_songs(items: list[dict], mel_root, labels_dir) -> list[Song]:
    songs = []
    for it in items:
        p = mel_path(mel_root, it['audio_sha256'])
        lp = Path(labels_dir) / it['item'][:2] / f"{it['item']}.json"
        if not p.exists() or not lp.exists():
            continue
        lab = RecLabels.from_json(json.loads(lp.read_text()))
        if len(lab.lines) < 4:
            continue
        full = np.load(p).astype(np.float32)
        songs.append(Song(it['item'], str(p), float(full.mean()), float(full.std() + 1e-6), lab, bool(it.get('hard'))))
        del full
    return songs


class Sampler:
    def __init__(self, songs: list[Song], seed: int, crop_frames: int = 2000, hard_weight: float = 3.0,
                 stretch=(0.88, 1.12), gain=0.5, tilt=0.5):
        self.songs = songs
        self.rng = np.random.default_rng(seed)
        w = np.array([hard_weight if s.hard else 1.0 for s in songs])
        self.p = w / w.sum()
        self.n = crop_frames
        self.stretch, self.gain, self.tilt = stretch, gain, tilt

    def state(self) -> dict:
        return self.rng.bit_generator.state

    def set_state(self, st: dict) -> None:
        self.rng.bit_generator.state = st

    def crop(self, song: Song, rate: float, start_ms: float) -> tuple[np.ndarray, dict]:
        mel = song.mel
        x = stretch_crop(mel, start_ms, self.n, rate)
        del mel
        x = (x - song.mean) / song.std
        return x, dense(song.labels, start_ms, self.n, rate=rate)

    def draw(self) -> tuple[np.ndarray, dict]:
        s = self.songs[self.rng.choice(len(self.songs), p=self.p)]
        r = float(self.rng.uniform(*self.stretch))
        a, b = s.labels.mapped
        half = self.n * 10.0 / 2.0
        centre = float(self.rng.uniform(a * r, max(b * r, a * r + 1.0)))
        start = round((centre - half) / 10.0) * 10.0
        x, tg = self.crop(s, r, start)
        x = gain_tilt(x, float(self.rng.uniform(-self.gain, self.gain)), float(self.rng.uniform(-self.tilt, self.tilt)))
        return x, tg

    def batch(self, size: int) -> tuple[np.ndarray, dict]:
        xs, tgs = zip(*(self.draw() for _ in range(size)))
        tg = {k: np.stack([t[k] for t in tgs]) for k in KEYS_F + KEYS_B + ('level',)}
        return np.stack(xs).astype(np.float32), tg


def to_torch(x: np.ndarray, tg: dict, device):
    import torch
    out = {}
    for k, v in tg.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        if k in KEYS_F:
            t = t.float()
        out[k] = t.to(device, non_blocking=True)
    return torch.from_numpy(x).to(device, non_blocking=True), out


def song_input(song_mel: np.ndarray) -> np.ndarray:
    """A whole song's standardised log-Mel [F, 128] float32 for inference."""
    full = np.asarray(song_mel, dtype=np.float32)
    return (full - full.mean()) / (full.std() + 1e-6)
