"""The instruments, synthesised in numpy.

Each voice turns an onset into a waveform: ``voice(tau, vel, pitch, p, rng)``
gives the samples at times ``tau`` (seconds since the onset, all >= 0).

- Every voice but ``pad`` has a hard attack: its first sample after the
  onset is far from zero, so an onset sits at a known sample. The renderer
  starts a note at the first sample at or after its onset time, which may
  fall between samples, and evaluates the voice at the exact offset from it.
- ``pad`` rises from silence over its attack, so it has no audible onset;
  it carries beatless spans and drones.
- Noise comes from the ``rng`` the renderer passes per note, so a note's
  sound depends only on the clip's seed and the note's place in its layer.

``NATURAL_S`` gives each voice's length before the renderer cuts it at the
next note of its layer or at a hard stop. Parameters are drawn per clip by
`draw_params`, from ranges that keep each voice recognisable.
"""
from __future__ import annotations

import math

import numpy as np

TRANSIENT_S = 0.0003


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(rng.standard_normal(n) * 0.5, -1.0, 1.0)


def _highpass(x: np.ndarray) -> np.ndarray:
    """First difference, scaled back into [-1, 1]: a gentle high-pass for noise voices."""
    if len(x) < 2:
        return x
    return np.concatenate([x[:1], np.diff(x)]) * 0.5


def kick(tau, vel, pitch, p, rng):
    f0, f1, tf, ta = p['f0'], p['f1'], p['tf'], p['ta']
    phase = 2 * math.pi * (f1 * tau + (f0 - f1) * tf * (1.0 - np.exp(-tau / tf)))
    body = np.cos(phase) * np.exp(-tau / ta)
    click = p['click'] * np.exp(-tau / TRANSIENT_S)
    return vel * (body + click) / (1.0 + p['click'])


def snare(tau, vel, pitch, p, rng):
    body = 0.4 * np.cos(2 * math.pi * p['fb'] * tau) * np.exp(-tau / 0.05)
    noise = 0.5 * _highpass(_noise(len(tau), rng)) * np.exp(-tau / p['tn'])
    hit = np.exp(-tau / TRANSIENT_S)
    return vel * (body + noise + hit) / 1.9


def hat(tau, vel, pitch, p, rng):
    noise = 0.6 * _highpass(_highpass(_noise(len(tau), rng)) * 2.0) * np.exp(-tau / p['th'])
    hit = 0.8 * np.exp(-tau / (0.5 * TRANSIENT_S))
    return vel * (noise + hit) / 1.4


def crash(tau, vel, pitch, p, rng):
    noise = 0.7 * _highpass(_noise(len(tau), rng)) * np.exp(-tau / p['tc'])
    hit = 0.6 * np.exp(-tau / TRANSIENT_S)
    return vel * (noise + hit) / 1.3


def click(tau, vel, pitch, p, rng):
    f = pitch[0] if pitch else p['fc']
    return vel * np.cos(2 * math.pi * f * tau) * np.exp(-tau / 0.004)


def block(tau, vel, pitch, p, rng):
    """A wood block: two inharmonic cosine partials with a fast decay."""
    f = pitch[0] if pitch else p['fw']
    body = np.cos(2 * math.pi * f * tau) + 0.5 * np.cos(2 * math.pi * 2.76 * f * tau)
    return vel * body * np.exp(-tau / p['tw']) / 1.5


def _harmonic(tau, freqs, decay_s, rolloff, sr_nyquist, detune=0.0):
    """Sum of cosine partials, so the first sample after the onset is at full level."""
    out = np.zeros(len(tau))
    total = 0.0
    for f in freqs:
        for k in range(1, 9):
            fk = f * k * (1.0 + detune * (k - 1))
            if fk >= 0.9 * sr_nyquist:
                break
            a = 1.0 / k ** rolloff
            out += a * np.cos(2 * math.pi * fk * tau) * np.exp(-tau * math.sqrt(k) / decay_s)
            total += a
    return out / max(total, 1e-9)


def pluck(tau, vel, pitch, p, rng):
    return vel * _harmonic(tau, pitch, p['decay'], 1.0, p['nyquist'], p.get('detune', 0.0))


def bass(tau, vel, pitch, p, rng):
    return vel * _harmonic(tau, pitch, p['decay'], 1.6, p['nyquist'])


def chord(tau, vel, pitch, p, rng):
    return vel * _harmonic(tau, pitch, p['decay'], 1.3, p['nyquist'], 0.0015)


def pad(tau, vel, pitch, p, rng):
    """Detuned partials under a raised-cosine attack and release; ``p['dur']`` is the note's length."""
    dur, attack, release = p['dur'], p['attack'], p['release']
    env = np.ones(len(tau))
    a = tau < attack
    env[a] = 0.5 - 0.5 * np.cos(math.pi * tau[a] / attack)
    r = tau > dur - release
    env[r] *= np.clip(0.5 + 0.5 * np.cos(math.pi * (tau[r] - (dur - release)) / release), 0.0, 1.0)
    out = np.zeros(len(tau))
    total = 0.0
    for f in pitch:
        for d in (-0.004, 0.0, 0.004):
            for k in range(1, 5):
                fk = f * k * (1.0 + d)
                if fk >= 0.9 * p['nyquist']:
                    break
                a_k = 1.0 / k ** 1.5
                out += a_k * np.sin(2 * math.pi * fk * tau + k * d * 100.0)
                total += a_k
    return vel * env * out / max(total, 1e-9) * 2.0


VOICES = dict(kick=kick, snare=snare, hat=hat, crash=crash, click=click, block=block, pluck=pluck, bass=bass,
              chord=chord, pad=pad)

#: Natural length in seconds, before the renderer cuts a note.
NATURAL_S = dict(kick=0.6, snare=0.5, hat=0.2, crash=2.0, click=0.03, block=0.3, pluck=1.2, bass=1.0, chord=1.2,
                 pad=None)

#: Voices whose notes may overlap within a layer; every other layer is monophonic.
POLYPHONIC = frozenset({'pad'})


def draw_params(voice: str, rng: np.random.Generator, sample_rate: int, family: str = '') -> dict:
    """One clip's parameters for ``voice``, drawn from its ranges.

    ``family`` shifts the ranges: ``band`` gives longer acoustic-like drums
    and a plucked bass, ``soft`` a low, dull kit with long decays; anything
    else the electronic set.
    """
    u = lambda a, b: float(rng.uniform(a, b))
    p: dict = {'nyquist': sample_rate / 2.0}
    soft = family == 'soft'
    band = family == 'band' or soft
    if voice == 'kick':
        p.update(f0=u(95, 125) if band else u(140, 180), f1=u(48, 60) if band else u(42, 52),
                 tf=u(0.025, 0.04), ta=u(0.15, 0.25) if band else u(0.12, 0.2), click=u(0.2, 0.5))
        if soft:
            p.update(f0=u(75, 95), ta=u(0.2, 0.3), click=u(0.05, 0.15))
    elif voice == 'snare':
        p.update(fb=u(170, 220), tn=u(0.12, 0.18) if band else u(0.07, 0.12))
    elif voice == 'hat':
        p.update(th=u(0.03, 0.05) if band else u(0.015, 0.03))
        if soft:
            p.update(th=u(0.05, 0.08))
    elif voice == 'crash':
        p.update(tc=u(0.4, 0.8))
    elif voice == 'click':
        p.update(fc=u(1400, 2400))
    elif voice == 'block':
        p.update(fw=u(600, 1100), tw=u(0.03, 0.06))
    elif voice == 'pluck':
        p.update(decay=u(0.25, 0.6), detune=u(0.0, 0.002))
    elif voice == 'bass':
        p.update(decay=u(0.4, 0.8) if band else u(0.2, 0.4))
    elif voice == 'chord':
        p.update(decay=u(0.6, 1.0) if soft else u(0.3, 0.7))
    elif voice == 'pad':
        p.update(attack=u(0.6, 1.4), release=u(0.8, 1.6))
    else:
        raise ValueError(f'Unknown voice {voice!r}')
    return p
