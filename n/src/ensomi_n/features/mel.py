"""The N line's log-Mel: 24 kHz, 40 ms Hann window, 10 ms hop, 128 Slaney bands.

The configuration follows the Listener's music frontend; the code is N's own
(numpy). Frames are centred: frame ``i`` is the window centred on ``i * hop``
samples, so on a waveform already on osu!'s clock frame ``i`` sits at
``10 * i`` ms. The waveform is zero-padded by half a window at both ends.
Values are ``log(max(mel power, floor))``, not standardised; the loader
standardises per song.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .clock import LEAD_RULE


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 24000
    n_fft: int = 960
    hop: int = 240
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 12000.0
    log_floor: float = 1e-5
    centred: bool = True
    lead_rule: str = LEAD_RULE
    resampler: str = 'torchaudio-sinc-hann-w6'
    version: str = 'mel-v1'

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.hop / self.sample_rate

    def key(self) -> str:
        """Cache directory name: the version and a hash of every field."""
        text = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return f'{self.version}-{hashlib.sha256(text.encode()).hexdigest()[:12]}'


DEFAULT = MelConfig()


def _hz_to_mel(f):
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    lin = f / (200.0 / 3.0)
    log = 15.0 + np.log(np.maximum(f, 1e-12) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log, lin)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    lin = m * (200.0 / 3.0)
    log = 1000.0 * np.exp((m - 15.0) * (np.log(6.4) / 27.0))
    return np.where(m >= 15.0, log, lin)


def filterbank(cfg: MelConfig = DEFAULT) -> np.ndarray:
    """[n_mels, n_fft // 2 + 1] triangular filters with Slaney area normalisation."""
    freqs = np.linspace(0.0, cfg.sample_rate / 2.0, cfg.n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(cfg.fmin), _hz_to_mel(cfg.fmax), cfg.n_mels + 2))
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rise = (freqs[None, :] - lo) / (mid - lo)
    fall = (hi - freqs[None, :]) / (hi - mid)
    weights = np.maximum(0.0, np.minimum(rise, fall))
    weights *= (2.0 / (hi - lo))
    return weights.astype(np.float32)


def frame_count(n_samples: int, cfg: MelConfig = DEFAULT) -> int:
    return int(np.ceil(n_samples / cfg.hop)) if n_samples > 0 else 0


def log_mel(wave: np.ndarray, cfg: MelConfig = DEFAULT, chunk: int = 4096) -> np.ndarray:
    """[F, n_mels] float32 log-Mel of a mono waveform at ``cfg.sample_rate``, F = ceil(N / hop)."""
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    n_frames = frame_count(len(x), cfg)
    if n_frames == 0:
        return np.zeros((0, cfg.n_mels), dtype=np.float32)
    half = cfg.n_fft // 2 if cfg.centred else 0
    need = (n_frames - 1) * cfg.hop + cfg.n_fft
    padded = np.zeros(max(need, half + len(x)), dtype=np.float32)
    padded[half:half + len(x)] = x
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(cfg.n_fft) / cfg.n_fft)).astype(np.float32)
    fb = filterbank(cfg).T                                   # [bins, mels]
    frames = np.lib.stride_tricks.sliding_window_view(padded, cfg.n_fft)[::cfg.hop][:n_frames]
    out = np.empty((n_frames, cfg.n_mels), dtype=np.float32)
    for a in range(0, n_frames, chunk):
        spec = np.fft.rfft(frames[a:a + chunk] * window, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        out[a:a + chunk] = np.log(np.maximum(power @ fb, cfg.log_floor))
    return out


def onset_strength(mel: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux of a log-Mel, summed over bands; [F], value 0 at frame 0.

    Frame ``i`` holds the rise from frame ``i - 1`` to ``i``. Used by the clock
    fixtures and the label checks, not by the model.
    """
    m = np.asarray(mel, dtype=np.float32)
    flux = np.zeros(len(m), dtype=np.float32)
    if len(m) > 1:
        flux[1:] = np.maximum(m[1:] - m[:-1], 0.0).sum(axis=1)
    return flux
