"""Augmentations on the cached log-Mel with exact label mapping (design §3.4; rung 1 uses stretch and gain/tilt).

- `stretch_crop`: a crop of ``n`` frames on the clock stretched by ``rate`` (times x rate): output frame
  ``j`` at ``start_ms + 10 j`` reads the source at ``(start_ms + 10 j) / rate`` by linear interpolation
  between source frames. Labels map exactly with `targets.dense(..., rate=rate)`. Beyond the song the
  crop holds the song's floor value.
- `gain_tilt`: adds a constant (gain) and a linear slope across bands (spectral tilt) to the log-Mel.
"""
from __future__ import annotations

import numpy as np

FRAME_MS = 10.0


def stretch_crop(mel: np.ndarray, start_ms: float, n: int, rate: float = 1.0, fill: float | None = None) -> np.ndarray:
    src = (start_ms + FRAME_MS * np.arange(n)) / rate / FRAME_MS
    i0 = np.floor(src).astype(np.int64)
    f = (src - i0).astype(np.float32)[:, None]
    lo = max(int(i0.min()), 0)
    hi = min(int(i0.max()) + 2, len(mel))
    fill = float(np.min(mel[lo:hi])) if fill is None and hi > lo else (fill if fill is not None else -11.5)
    block = np.asarray(mel[lo:hi], dtype=np.float32) if hi > lo else np.zeros((0, mel.shape[1]), np.float32)

    def rows(idx):
        out = np.full((len(idx), mel.shape[1]), fill, dtype=np.float32)
        ok = (idx >= lo) & (idx < hi)
        out[ok] = block[idx[ok] - lo]
        return out

    return rows(i0) * (1.0 - f) + rows(i0 + 1) * f


def gain_tilt(x: np.ndarray, gain: float, tilt: float) -> np.ndarray:
    return x + gain + tilt * np.linspace(-1.0, 1.0, x.shape[-1], dtype=np.float32)
