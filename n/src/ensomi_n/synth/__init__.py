"""Synthetic music with exact timing ground truth: suite v0 (N1 evaluation E1).

A clip is rendered deterministically from (setting, seed): a symbolic
program (`song.Song`: tempo map, bars, layers, notes, edits), numpy voices
(`voices`: kicks, snares, hats, clicks, blocks, plucked and pad tones), and
a renderer (`render`). Nothing is downloaded; every sound is computed here,
so the audio is MIT by construction. 60 s, 24 kHz, mono, 16-bit WAV.

The ground truth (`truth`) is read off the program, on the decoded-audio
clock (sample n at 1000 n / 24000 ms): notated beats and downbeats, red
lines where the tempo is piecewise constant, the tempo of every beat, each
beat's subdivision class, every layer's onsets, spans without pulse, and the
stratum spans the measures run on. It is written as a music map
(``map.json``, clock ``decoded``), an onset file (``onsets.json``) and an
``.osu`` file (``truth.osu``, on the decoded clock).

``SETTINGS`` (`settings`) lists the settings, each with its stratum and its
musical meaning. ``python -m ensomi_n.synth`` renders them.

What synthetic audio can show: where a timing system fails on audio whose
structure is known exactly. What it cannot: anything about real music. The
attacks are clean, there is no mix or master and no human performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from .render import CLIP_S, SAMPLE_RATE, render, write_wav
from .settings import SETTINGS, STRATA, Setting
from .song import Song
from .truth import ground_truth, music_map, onset_file, osu_text

GENERATOR = 'ensomi-synth'
VERSION = '0.1.0'

__all__ = ['GENERATOR', 'VERSION', 'SETTINGS', 'STRATA', 'Setting', 'Clip', 'build', 'make', 'write_clip',
           'clip_id', 'SAMPLE_RATE', 'CLIP_S']


def clip_id(setting: str, seed: int) -> str:
    return f'{setting}-s{int(seed)}'


def build(setting: str, seed: int) -> Song:
    """The program of one clip. The generator is seeded by (seed, setting name), nothing else."""
    if setting not in SETTINGS:
        raise KeyError(f'Unknown setting {setting!r}; known: {sorted(SETTINGS)}')
    rng = np.random.default_rng([int(seed), zlib.crc32(setting.encode('utf-8'))])
    song = Song(setting, int(seed), rng, SAMPLE_RATE, target_s=CLIP_S)
    SETTINGS[setting].build(song)
    return song


@dataclass
class Clip:
    id: str
    setting: str
    stratum: str
    seed: int
    song: Song
    samples: np.ndarray
    truth: dict


def make(setting: str, seed: int) -> Clip:
    song = build(setting, seed)
    x = render(song)
    gt = ground_truth(song, 1000.0 * CLIP_S)
    return Clip(clip_id(setting, seed), setting, SETTINGS[setting].stratum, int(seed), song, x, gt)


def provenance(clip: Clip, commit: str = '') -> dict:
    return dict(name=GENERATOR, version=VERSION, commit=commit, setting=clip.setting, seed=clip.seed,
                stratum=clip.stratum, sample_rate=SAMPLE_RATE, clip_s=CLIP_S, soundfont=None,
                voices='numpy (ensomi_n.synth.voices)')


def write_clip(clip: Clip, audio_dir, truth_dir, commit: str = '') -> dict:
    """Write ``<audio_dir>/<id>.wav`` and ``<truth_dir>/<id>/{map.json,onsets.json,truth.osu}``; returns the
    manifest row."""
    audio_dir, truth_dir = Path(audio_dir), Path(truth_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    out = truth_dir / clip.id
    out.mkdir(parents=True, exist_ok=True)
    wav = f'{clip.id}.wav'
    data = write_wav(audio_dir / wav, clip.samples, SAMPLE_RATE)
    prov = provenance(clip, commit)
    music_map(clip.truth, wav, prov).dump(out / 'map.json')
    (out / 'onsets.json').write_text(json.dumps(onset_file(clip.truth, prov), indent=1) + '\n', encoding='utf-8')
    (out / 'truth.osu').write_text(osu_text(clip.truth, wav, clip.id), encoding='utf-8')
    gt = clip.truth
    return dict(id=clip.id, setting=clip.setting, stratum=clip.stratum, seed=clip.seed, audio=wav,
                audio_sha256=hashlib.sha256(data).hexdigest(), audio_bytes=len(data), sample_rate=SAMPLE_RATE,
                n_samples=len(clip.samples), generator=GENERATOR, version=VERSION, commit=commit,
                n_beats=len(gt['beats']['t_ms']), n_segments=len(gt['segments'] or []),
                n_no_pulse=len(gt['no_pulse']), params=_jsonable(clip.song.params),
                meaning=SETTINGS[clip.setting].meaning)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x
