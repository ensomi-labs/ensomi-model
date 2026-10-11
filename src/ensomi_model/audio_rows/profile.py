"""Predict a relative section workload profile from the existing log-Mel features.

A ridge regression learns within-song variation in log(W_H/s), using only fit_train
charts. It reads section summaries of standardised log-Mel and positive spectral
flux; it does not add channels to the head model. At inference a song-level request
sets the duration-weighted arithmetic mean W_H/s. The audio supplies its distribution
over 16-beat sections, with the same 0.001 floor as training labels. This preserves
requested workload, not a guarantee about realised workload or star rating.

Fit with ``python -m ensomi_model.audio_rows.profile --data <build> --grid chart
--out <profile.json>`` and pass the result to ``audio_rows.generate --profile``.
This does not retrain the head generator. Validate the predicted requests on held-out
songs and ablate them on trained head weights before drawing an input-use conclusion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .audio import HOP_MS, MEL_BINS
from .data import Dataset, LABEL_FLOOR, SECTION_BEATS, section_label, sections
from .lattice import beat_range
from .measures import audio_curves

FEATURE_NAMES = tuple(f'mel_group_{i}' for i in range(8)) + (
    'energy_std', 'energy_p10', 'energy_p90', 'flux_mean', 'flux_p90', 'quiet_fraction', 'active_fraction')


def descriptors(features, edges_ms):
    """Centred [sections,15] audio summaries; centres use physical section durations.

    Quiet and active frames fall below/above this song's energy quartiles. All
    descriptors are relative to the current song, including when it has one section.
    """
    mel = np.asarray(features[:, :MEL_BINS], dtype=np.float32)
    energy, flux = audio_curves(features)
    low, high = np.quantile(energy, [.25, .75])
    rows = []
    for a, b in zip(edges_ms[:-1], edges_ms[1:]):
        start = min(max(int(a / HOP_MS), 0), len(mel) - 1)
        end = min(max(int(b / HOP_MS), start + 1), len(mel))
        e, f = energy[start:end], flux[start:end]
        rows.append([*mel[start:end].mean(0).reshape(8, -1).mean(1), float(e.std()),
                     *np.quantile(e, [.1, .9]), float(f.mean()), float(np.quantile(f, .9)),
                     float((e < low).mean()), float((e > high).mean())])
    x = np.asarray(rows, dtype=np.float64)
    return x - np.average(x, axis=0, weights=np.diff(edges_ms))


def distribute(scores, durations, log_level):
    """Section log rates with a floor and the requested duration-weighted linear mean.

    A request below log(LABEL_FLOOR) is invalid. A constant score gives a constant
    request; adding a constant to every score has no effect.
    """
    level = float(np.exp(log_level))
    if not np.isfinite(level) or level < LABEL_FLOOR - 1e-12:
        raise ValueError('Song workload must be finite and at least LABEL_FLOOR')
    weights = np.exp(np.asarray(scores) - np.max(scores))
    rates = LABEL_FLOOR + max(level - LABEL_FLOOR, 0) * weights / np.average(weights, weights=durations)
    return np.log(rates)


@dataclass
class WorkloadProfile:
    scale: list
    coefficient: list
    section_beats: int = SECTION_BEATS
    version: int = 1

    def scores(self, features, edges_ms):
        return (descriptors(features, edges_ms) / np.asarray(self.scale)) @ np.asarray(self.coefficient)

    def requests(self, features, grid, song_ms, log_level):
        """Raw log(W_H/s) per beat, before any explicit section overrides."""
        b0, nb = beat_range(grid, song_ms)
        beat_ms = np.clip(grid.time_of_beat(b0 + np.arange(nb + 1)), 0.0, song_ms)
        scopes = sections(nb, self.section_beats)
        edges = np.array([beat_ms[a] for a, _ in scopes] + [song_ms])
        values = distribute(self.scores(features, edges), np.diff(edges), log_level)
        return np.repeat(values, [b - a for a, b in scopes])

    @classmethod
    def load(cls, path):
        record = json.loads(Path(path).read_text())
        if record['profile']['version'] != 1 or record['features'] != list(FEATURE_NAMES):
            raise ValueError('Unsupported workload profile')
        return cls(**record['profile'])


def fit(data: Dataset, ridge=0.01):
    """Fit weighted MSE + ridge*||coefficient||² on fit_train only.

    Each audio file has equal total weight, divided among its charts and then by
    section duration. Targets and descriptors are centred within each chart;
    difficulty level therefore cannot explain away the local musical variation.
    The feature scale is fitted on these training rows only.
    """
    if ridge <= 0:
        raise ValueError('ridge must be positive')
    multiplicity = Counter(r['key'] for r in data.train)
    xs, ys, ws = [], [], []
    for row in data.train:
        c = data.charts[row['sha']]
        scopes = sections(len(c['count']))
        edges = np.array([c['beat_ms'][a] for a, _ in scopes] + [c['beat_ms'][-1]])
        weight = np.diff(edges) / edges[-1]
        target = np.array([section_label(c['wh_beat'], c['beat_ms'], a, b) for a, b in scopes])
        xs.append(descriptors(data.features(row['key']), edges))
        ys.append(target - np.average(target, weights=weight))
        ws.append(weight / multiplicity[row['key']])
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    weight /= weight.sum()
    scale = np.maximum(np.sqrt(np.average(x * x, axis=0, weights=weight)), 1e-6)
    x /= scale
    coef = np.linalg.solve(x.T @ (weight[:, None] * x) + ridge * np.eye(x.shape[1]), x.T @ (weight * y))
    error = (x @ coef - y) ** 2
    profile = WorkloadProfile(scale.tolist(), coef.tolist())
    return profile, dict(profile=asdict(profile), features=list(FEATURE_NAMES), ridge=ridge,
                         train_charts=len(data.train), train_audio=len(multiplicity), train_sections=len(y),
                         train_mse=float(np.average(error, weights=weight)),
                         flat_mse=float(np.average(y * y, weights=weight)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', required=True)
    p.add_argument('--grid', choices=('chart', 'fitted'), default='chart')
    p.add_argument('--out', required=True)
    p.add_argument('--ridge', type=float, default=0.01)
    a = p.parse_args(argv)
    _, record = fit(Dataset(a.data, a.grid), a.ridge)
    record['grid'] = a.grid
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
