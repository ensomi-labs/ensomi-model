"""Decode an audio file to a mono 24 kHz waveform on osu!'s clock.

libsndfile (through ``soundfile``) decodes first. If it fails, the ``ffmpeg``
command line decodes at the file's own rate. Each decoder's output gets the
lead of `clock.lead_samples` for that decoder, at the file's rate, before
resampling, so the result is on osu!'s clock either way. Resampling uses
torchaudio's sinc interpolation (zero phase, so no delay is added).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess

import numpy as np

from .clock import file_lead
from .mel import DEFAULT, MelConfig


@dataclass
class Decoded:
    wave: np.ndarray        # float32 mono at cfg.sample_rate, osu!'s clock
    sample_rate: int
    source_rate: int
    decoder: str            # 'libsndfile' or 'ffmpeg'
    lead_samples: int       # at source_rate
    header: str             # MP3 start kind, '' for other formats


def _libsndfile(path) -> tuple[np.ndarray, int]:
    import soundfile
    data, sr = soundfile.read(str(path), dtype='float32', always_2d=True)
    return data.mean(axis=1), int(sr)


def _ffmpeg_rate(path) -> int:
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                          'stream=sample_rate', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[0].split(',')[0])     # ffprobe may add a trailing field separator


def _ffmpeg(path) -> tuple[np.ndarray, int]:
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found')
    sr = _ffmpeg_rate(path)
    out = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'],
                         capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32).copy(), sr


def resample(wave: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(wave, dtype=np.float32)
    import torch
    import torchaudio.functional as AF
    x = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))
    return AF.resample(x, source_rate, target_rate).numpy()


def decode(path: str | os.PathLike, cfg: MelConfig = DEFAULT) -> Decoded:
    """The waveform of ``path`` on osu!'s clock at ``cfg.sample_rate``."""
    try:
        wave, sr = _libsndfile(path)
        decoder = 'libsndfile'
    except Exception:       # libsndfile rejects a few MP3s; ffmpeg reads them
        wave, sr = _ffmpeg(path)
        decoder = 'ffmpeg'
    lead, _, start = file_lead(path, decoder)
    if lead:
        wave = np.concatenate([np.zeros(lead, dtype=np.float32), wave])
    return Decoded(resample(wave, sr, cfg.sample_rate), cfg.sample_rate, sr, decoder, lead,
                   start.kind if start is not None else '')
