"""N1 v0 batches: weighted recording sampler, two crops per song, the whole-song memory input, augmentation.

Each draw picks a recording by its curation weight, a stretch ``rate`` in [0.88, 1.12], and two crops
of 20 s whose centres lie in the mapped range. The crops' log-Mel is read through a memory map and
standardised with the song's statistics; the song's memory input (``memo-v1``) is stretched by the same
rate (linear interpolation along time) so that crop and memory share one clock. Songs longer than
``MAX_MEMO_S`` give the memory a window of that length around the crop pair. Gain and spectral tilt
are drawn per crop. Labels map exactly (`v0.labels.dense_v0`).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..augment import gain_tilt, stretch_crop
from ..features.cache import mel_path
from .labels import V0Labels, dense_v0
from .songs import MEMO_HOP, memo_path

CROP = 2000
MAX_MEMO_S = 480.0
MEMO_BUCKET = 1000      # memory inputs are padded to a multiple of this many frames (40 s): few shapes on MPS
KEYS_F = ('beat', 'half', 'down', 'beat_off', 'half_off', 'down_off', 'weight', 'phase', 'logp', 'bar_phase', 'bound')
KEYS_B = ('beat_offmask', 'half_offmask', 'down_offmask', 'mask', 'phase_mask', 'bar_mask')


@dataclass
class Song:
    item: str
    mel: str
    memo: str
    mean: float
    std: float
    frames: int
    labels: V0Labels
    weight: float = 1.0
    source: str = ''
    strata: tuple = ()


def load_songs(rows: list[dict], index: dict, bulk: Path, labels_dir: Path, *, min_lines: int = 4) -> list[Song]:
    songs = []
    for r in rows:
        st = index.get(r['item'])
        if st is None or 'error' in st:
            continue
        lp = Path(labels_dir) / r['item'][:2] / f"{r['item']}.json"
        if not lp.exists():
            continue
        lab = V0Labels.from_json(json.loads(lp.read_text()))
        if len(lab.lines) < min_lines:
            continue
        songs.append(Song(r['item'], str(mel_path(bulk / 'mel', r['item'])), str(memo_path(bulk, r['item'])),
                          st['mean'], st['std'], st['frames'], lab, float(r.get('weight', 1.0)), r.get('source', ''),
                          tuple(r.get('strata') or ())))
    return songs


def memo_input(song: Song, rate: float = 1.0, centre_ms: float | None = None) -> np.ndarray:
    """The song's standardised memory input on the clock stretched by ``rate`` [S, 64] float32."""
    m = (np.load(song.memo).astype(np.float32) - song.mean) / song.std
    hop_ms = 10.0 * MEMO_HOP
    if abs(rate - 1.0) > 1e-6:
        n = int(len(m) * rate)
        src = np.arange(n) / rate
        i0 = np.clip(np.floor(src).astype(np.int64), 0, len(m) - 1)
        i1 = np.clip(i0 + 1, 0, len(m) - 1)
        f = (src - np.floor(src)).astype(np.float32)[:, None]
        m = m[i0] * (1 - f) + m[i1] * f
    cap = int(MAX_MEMO_S * 1000 / hop_ms)
    if len(m) > cap:
        c = int((centre_ms if centre_ms is not None else len(m) * hop_ms / 2) / hop_ms)
        a = int(np.clip(c - cap // 2, 0, len(m) - cap))
        m = m[a:a + cap]
    return m


def pad_memos(memos: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    s = max(len(m) for m in memos)
    s += (-s) % MEMO_BUCKET
    out = np.zeros((len(memos), s, memos[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(memos), s), dtype=bool)
    for i, m in enumerate(memos):
        out[i, :len(m)] = m
        mask[i, :len(m)] = True
    return out, mask


class Sampler:
    def __init__(self, songs: list[Song], seed: int, crops_per_song: int = 2, stretch=(0.88, 1.12), gain=0.5,
                 tilt=0.5):
        self.songs = songs
        self.rng = np.random.default_rng(seed)
        w = np.array([s.weight for s in songs], dtype=np.float64)
        self.p = w / w.sum()
        self.k = crops_per_song
        self.stretch, self.gain, self.tilt = stretch, gain, tilt

    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, st):
        self.rng.bit_generator.state = st

    def crop(self, song: Song, rate: float, start_ms: float):
        mel = np.load(song.mel, mmap_mode='r')
        x = stretch_crop(mel, start_ms, CROP, rate)
        del mel
        return (x - song.mean) / song.std, dense_v0(song.labels, start_ms, CROP, rate=rate)

    def batch(self, n_songs: int):
        xs, tgs, memos, owner = [], [], [], []
        for i in range(n_songs):
            s = self.songs[self.rng.choice(len(self.songs), p=self.p)]
            r = float(self.rng.uniform(*self.stretch))
            a, b = s.labels.mapped
            starts = []
            for _ in range(self.k):
                centre = float(self.rng.uniform(a * r, max(b * r, a * r + 1.0)))
                start = round((centre - CROP * 5.0) / 10.0) * 10.0
                x, tg = self.crop(s, r, start)
                x = gain_tilt(x, float(self.rng.uniform(-self.gain, self.gain)),
                              float(self.rng.uniform(-self.tilt, self.tilt)))
                xs.append(x)
                tgs.append(tg)
                owner.append(i)
                starts.append(start)
            memos.append(memo_input(s, r, centre_ms=float(np.mean(starts)) + CROP * 5.0))
        memo, mask = pad_memos(memos)
        tg = {k: np.stack([t[k] for t in tgs]) for k in KEYS_F + KEYS_B + ('level',)}
        return np.stack(xs).astype(np.float32), tg, memo, mask, np.asarray(owner)


def fixed_crops(songs: list[Song], per_song: int = 2, seed: int = 7):
    """Deterministic validation crops (no augmentation): (x, targets, memo, mask, owner, song index)."""
    rng = np.random.default_rng(seed)
    out = []
    for si, s in enumerate(songs):
        a, b = s.labels.mapped
        for k in range(per_song):
            centre = a + (b - a) * (k + 0.5) / per_song + float(rng.uniform(-2000, 2000))
            start = round((centre - CROP * 5.0) / 10.0) * 10.0
            mel = np.load(s.mel, mmap_mode='r')
            x = (stretch_crop(mel, start, CROP, 1.0) - s.mean) / s.std
            out.append((x.astype(np.float32), dense_v0(s.labels, start, CROP), si))
    return out


def to_torch(x, tg, memo, mask, device):
    import torch
    t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in tg.items()}
    for k in KEYS_F:
        t[k] = t[k].float()
    return (torch.from_numpy(x).to(device), t, torch.from_numpy(memo).to(device), torch.from_numpy(mask).to(device))
