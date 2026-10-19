"""Rendering a `Song` to samples: notes per layer, then edits, then the room and the mix.

- Each note starts at the first sample at or after its onset and is
  evaluated at the exact offset from the onset, so an onset between two
  samples is still exact to well under a sample.
- A monophonic layer's note ends at the earliest of its natural length,
  ``EDGE_MS`` before the layer's next note, and ``EDGE_MS`` before a hard
  stop, fading out over the ``FADE_MS`` before that end. Every onset of such
  a layer therefore follows at least ``EDGE_MS`` of exact silence, which is
  what lets `detect.onsets` find it to the sample on an isolated layer.
- Pads are polyphonic, have no attack and run for their explicit length.
- Noise inside a voice comes from a generator seeded by (clip seed, layer
  name, note index), so a note's sound depends on nothing else.
- Edits (`tempo.EditList`) rewrite each layer's source samples into the
  output timeline. The room (a decaying-noise impulse response) is added to
  the mix after the edits; ``fx=False`` leaves it out, for tests.
- The mix is scaled to a fixed peak and the clip is padded or cut to
  ``CLIP_S``. `to_pcm16` rounds to 16-bit samples.

Everything is numpy and deterministic: the same (setting, seed) gives the
same samples on one platform.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from .tempo import EDGE_MS, FADE_MS
from .voices import NATURAL_S, POLYPHONIC, VOICES

SAMPLE_RATE = 24000
CLIP_S = 60.0
PEAK = 0.89


def _note_rng(seed: int, layer: str, index: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(layer.encode('utf-8')), int(index)])


def render_layer(song, name: str, n_samples: int) -> np.ndarray:
    """One layer of ``song`` on the source timeline, dry, with its gain applied."""
    sr = song.sr
    lay = song.layers[name]
    voice = VOICES[lay.voice]
    notes = sorted((n for n in song.notes if n.layer == name), key=lambda n: n.t)
    stops = np.sort(np.asarray(song.hard_stops, dtype=np.float64))
    out = np.zeros(n_samples)
    poly = lay.voice in POLYPHONIC
    for k, note in enumerate(notes):
        t = note.t
        n0 = int(math.ceil(t * sr / 1000.0 - 1e-9))
        if n0 >= n_samples:
            continue
        if poly:
            dur = note.dur if note.dur is not None else 2000.0
            end = t + dur
        else:
            end = t + 1000.0 * NATURAL_S[lay.voice]
            if note.dur is not None:
                end = min(end, t + note.dur)
            if k + 1 < len(notes):
                end = min(end, notes[k + 1].t - EDGE_MS)
            i = int(np.searchsorted(stops, t + EDGE_MS, side='right'))
            if i < len(stops):
                end = min(end, stops[i] - EDGE_MS)
        n1 = min(int(math.ceil(end * sr / 1000.0 - 1e-9)), n_samples)
        if n1 <= n0:
            continue
        tau = (np.arange(n0, n1) - t * sr / 1000.0) / sr
        params = lay.params
        if poly:
            params = dict(params, dur=(end - t) / 1000.0)
        x = voice(tau, note.vel, note.pitch, params, _note_rng(song.seed, name, k))
        if not poly:
            fade = min(FADE_MS / 1000.0, (end - t) / 4000.0)
            x = x * np.clip(((end - t) / 1000.0 - tau) / fade, 0.0, 1.0)
        out[n0:n1] += lay.gain * x
    return out


def source_samples(song) -> int:
    return int(math.ceil(song.end_ms * song.sr / 1000.0)) + 1


def render(song, layers=None, fx: bool = True, clip: bool = True) -> np.ndarray:
    """The clip's samples (float64): the given layers (default all), after edits, room and peak scaling.

    ``clip`` pads or cuts the output to ``CLIP_S`` seconds.
    """
    names = list(song.layers) if layers is None else list(layers)
    n_src = source_samples(song)
    mix = None
    for name in names:
        x = render_layer(song, name, n_src)
        if song.edits is not None:
            x = song.edits.apply(x, song.sr)
        mix = x if mix is None else mix + x
    if mix is None:
        mix = np.zeros(n_src)
    if fx and song.room is not None:
        mix = mix + room(mix, song.room, song.sr)
    if clip:
        n = int(round(CLIP_S * song.sr))
        mix = mix[:n] if len(mix) >= n else np.concatenate([mix, np.zeros(n - len(mix))])
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0:
        mix = mix * (PEAK / peak)
    return mix


def room(x: np.ndarray, params: dict, sample_rate: int) -> np.ndarray:
    """The wet signal of a simple room: ``x`` convolved with decaying, low-passed noise after a pre-delay."""
    sr = sample_rate
    rng = np.random.default_rng(params['seed'])
    n = int(params['rt60_s'] * sr)
    pre = int(params['predelay_ms'] * sr / 1000.0)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.91 * t / params['rt60_s'])
    a = params['lowpass']
    # One-pole low-pass, as its impulse response truncated where a**k falls below 1e-15.
    taps = (1.0 - a) * a ** np.arange(max(1, int(math.ceil(-15.0 / math.log10(a)))))
    ir = np.convolve(ir, taps)[:n]
    ir = np.concatenate([np.zeros(pre), ir])
    ir *= params['wet'] / max(np.sqrt(np.sum(ir ** 2)), 1e-12)
    m = len(x) + len(ir) - 1
    size = 1 << (m - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:len(x)]
    return y


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype('<i2')


def write_wav(path, x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Write mono 16-bit PCM with the standard library; returns the file's bytes."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(to_pcm16(x).tobytes())
    data = buf.getvalue()
    if path is not None:
        with open(path, 'wb') as f:
            f.write(data)
    return data
