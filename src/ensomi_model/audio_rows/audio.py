"""What the generator reads of a song: the fitter's segments and 10 ms frame features.

``Listener.listen`` runs BeatThis ``final0`` on the audio, fits the legacy grid to its beat and
downbeat probabilities, and builds the frame features: the song's standardised log-Mel (24 kHz,
128 bins, 10 ms hop), then the same BeatThis beat and downbeat probabilities. Mel frame i
covers [10 i, 10 i + 40) ms (no centre padding); the probabilities are linearly interpolated from
BeatThis's 50 Hz frames (frame j centred at 20 j ms) to the Mel frame's centre, 10 i + 20 ms.
The model reads a time t from frame floor(t / 10). Training data and generation both call ``listen``, so
they see the same inputs.

BeatThis reads the file with its own loader (libsndfile); the Mel uses ``load_audio_file``
(pydub, ffmpeg). Each waveform is put on osu!'s clock first: ``decoder.lead_seconds`` of silence
is prepended where that decoder trims an MP3 start that osu!'s BASS keeps (about 2 % of songs,
12 ms), so the grid and the features sit where the mappers' notes sit. When BeatThis's loader
cannot decode a file (it failed on 2 of 300 corpus MP3s), BeatThis reads the Mel waveform instead.
After the legacy fit, ``fitter.peaks.refine_segments`` refits each segment's offset and beat
length to BeatThis's sub-frame peaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from ..features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from .decoder import lead_seconds
from .fitter import fit_segments
from .fitter.peaks import refine_segments

HOP_MS = MUSIC_MEL_CACHE_CONFIG.hop_ms
MEL_BINS = MUSIC_MEL_CACHE_CONFIG.mel_bins
MEL_CENTRE_MS = 500.0 * MUSIC_MEL_CACHE_CONFIG.n_fft / MUSIC_MEL_CACHE_CONFIG.sample_rate   # 20 ms
FEATURE_DIM = MEL_BINS + 2          # log-Mel, then BeatThis beat and downbeat probability
BEATTHIS_FPS = 50.0


@dataclass
class Song:
    features: np.ndarray            # [F, FEATURE_DIM] float16
    segments: np.ndarray            # [S, 2] the fitter's raw (offset_ms, beat_length_ms)
    song_ms: float                  # audio duration
    seconds: dict = field(default_factory=dict)


def features(log_mel: np.ndarray, beat_prob: np.ndarray, downbeat_prob: np.ndarray) -> np.ndarray:
    """[F, FEATURE_DIM] float16 from log-Mel [F, MEL_BINS] and 50 Hz probabilities."""
    frames_ms = np.arange(len(log_mel)) * HOP_MS + MEL_CENTRE_MS
    beat_ms = np.arange(len(beat_prob)) * 1000.0 / BEATTHIS_FPS
    mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    acts = [np.interp(frames_ms, beat_ms, p) for p in (beat_prob, downbeat_prob)]
    return np.column_stack((mel, *acts)).astype(np.float16)


def _probability(logits) -> np.ndarray:
    """Sigmoid as the legacy provider computed it, float64 then float32, so fits match the source."""
    x = logits.detach().cpu().numpy().astype(np.float64)
    p = np.empty_like(x)
    pos = x >= 0.0
    p[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    p[~pos] = e / (1.0 + e)
    return p.astype(np.float32)


def _lead(wave: np.ndarray, seconds: float, sample_rate: int) -> np.ndarray:
    """``wave`` with ``seconds`` of silence prepended along its first axis."""
    n = int(round(seconds * sample_rate))
    if n <= 0:
        return wave
    return np.concatenate((np.zeros((n, *wave.shape[1:]), dtype=wave.dtype), wave))


class Listener:
    """BeatThis loaded once; ``listen`` is then called per audio file."""

    def __init__(self, beatthis_checkpoint, device: str = 'cpu'):
        from beat_this.inference import Audio2Frames, load_audio
        self.frames = Audio2Frames(checkpoint_path=str(beatthis_checkpoint), device=device, float16=False)
        self.load = load_audio

    def listen(self, path) -> Song:
        """Raises ValueError from ``fit_segments`` on audio shorter than 3 s or without beats."""
        from ..features.audio import load_audio_file
        sr_mel = MUSIC_MEL_CACHE_CONFIG.sample_rate
        t0 = time.perf_counter()
        mel_wave = _lead(load_audio_file(path, sr_mel), lead_seconds(path, 'ffmpeg'), sr_mel)
        try:
            wave, sr = self.load(path)
            wave = _lead(np.asarray(wave), lead_seconds(path, 'mpg123'), sr)
        except Exception:   # BeatThis's decoder fails on some MP3s; fall back to the Mel waveform
            wave, sr = mel_wave, sr_mel
        t1 = time.perf_counter()
        beat_logits, downbeat_logits = self.frames(wave, sr)
        beat, downbeat = _probability(beat_logits), _probability(downbeat_logits)
        t2 = time.perf_counter()
        segments = fit_segments(beat, downbeat, BEATTHIS_FPS)
        segments = refine_segments(segments, beat_logits.detach().cpu().numpy(), BEATTHIS_FPS)
        segments = np.array([(s.offset_ms, s.beat_length_ms) for s in segments])
        t3 = time.perf_counter()
        log_mel = compute_log_mel_10ms(mel_wave, sample_rate=sr_mel, config=MUSIC_MEL_CACHE_CONFIG)
        feats = features(log_mel, beat, downbeat)
        t4 = time.perf_counter()
        return Song(feats, segments, 1000.0 * len(wave) / sr,
                    dict(load=t1 - t0, beatthis=t2 - t1, fit=t3 - t2, features=t4 - t3))
