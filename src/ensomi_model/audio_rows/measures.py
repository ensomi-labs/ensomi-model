"""Audio/chart characterisation; these measures do not decide chart quality.

Intensity is mean standardised log-Mel over the 128 bins. Onset strength is the
mean positive 30 ms difference of those bins, sampled within ±30 ms of a time.
Neither is perceptual loudness or a source-separated estimate of musical salience.
Section statistics include leading/trailing silence and keep full-song strain history.
"""
from __future__ import annotations

import numpy as np

from .audio import HOP_MS, MEL_BINS
from .data import beat_workload, sections
from .lattice import beat_range, slot_times


def ranks(values):
    """Average zero-based ranks, including ties."""
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def correlation(a, b, *, rank=False):
    """Pearson (or Spearman) correlation, None for a constant or insufficient input."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    if rank:
        a, b = ranks(a), ranks(b)
    return float(np.corrcoef(a, b)[0, 1])


def audio_curves(features):
    """Per-frame intensity and positive log-Mel flux, in song-standardised units."""
    mel = np.asarray(features[:, :MEL_BINS], dtype=np.float32)
    energy = mel.mean(1)
    flux = np.zeros(len(mel), dtype=np.float32)
    flux[3:] = np.maximum(mel[3:] - mel[:-3], 0).mean(1)
    return energy, flux


def local_strength(curve, times_ms, radius_ms=30.0):
    """Maximum curve value within a symmetric time tolerance; zero for an empty query."""
    idx = np.floor(np.asarray(times_ms) / HOP_MS).astype(int)
    radius = int(round(radius_ms / HOP_MS))
    return np.max([curve[np.clip(idx + d, 0, len(curve) - 1)]
                   for d in range(-radius, radius + 1)], axis=0)


def section_audio(features, edges_ms):
    """Section means of the two audio curves for consecutive physical-time edges."""
    energy, flux = audio_curves(features)
    result = []
    for a, b in zip(edges_ms[:-1], edges_ms[1:]):
        start = min(max(int(a / HOP_MS), 0), len(energy) - 1)
        end = min(max(int(b / HOP_MS), start + 1), len(energy))
        result.append((float(energy[start:end].mean()), float(flux[start:end].mean())))
    return np.asarray(result)


def chart_measures(head_ms, song_ms, grid, features, section_beats=16):
    """Summary plus section arrays, for human or generated heads on the same audio.

    Low/high intensity means bottom/top quartile of sections by audio, independently
    of heads. Their rates divide workload by duration. Rest fraction uses beats with
    no heads, weighted by physical duration. Head onset rank compares each head's
    tolerated flux with every distinct duple/triple slot in its canonical beat;
    a half-beat circular shift of the head is the density-preserving timing null.
    """
    b0, nb = beat_range(grid, song_ms)
    beat_ms = np.clip(grid.time_of_beat(b0 + np.arange(nb + 1)), 0.0, song_ms)
    scopes = sections(nb, section_beats)
    edges = np.array([beat_ms[a] for a, _ in scopes] + [beat_ms[-1]])
    audio = section_audio(features, edges)
    durations = np.diff(edges) / 1000.0
    wh = beat_workload(head_ms, song_ms, beat_ms)
    workload = np.array([wh[a:b].sum() for a, b in scopes])
    rates = workload / durations
    counts = np.histogram(head_ms, beat_ms)[0]
    rest = np.array([np.sum(np.diff(beat_ms)[a:b] * (counts[a:b] == 0)) / (beat_ms[b] - beat_ms[a])
                     for a, b in scopes])
    n_tail = max(1, len(scopes) // 4)
    order = np.argsort(audio[:, 0], kind='stable')
    low, high = order[:n_tail], order[-n_tail:]
    low_rate = float(workload[low].sum() / durations[low].sum())
    high_rate = float(workload[high].sum() / durations[high].sum())
    _, flux = audio_curves(features)
    slots = slot_times(grid, b0, nb)
    hb = grid.beat(head_ms)
    head_beats = np.clip(np.floor(hb).astype(int) - b0, 0, nb - 1)
    # The union has 24 positions: all duple slots and the eight unshared triple slots.
    reference_times = np.concatenate((slots[:, 0], slots[:, 1, np.flatnonzero(np.arange(12) % 3 != 0)]), axis=1)
    reference = local_strength(flux, reference_times)[head_beats]
    available = ((reference_times >= 0) & (reference_times < song_ms))[head_beats]

    def onset_ranks(times):
        strength = local_strength(flux, times)[:, None]
        return (((strength > reference) + 0.5 * (strength == reference)) * available).sum(1) / available.sum(1)

    selected = onset_ranks(head_ms)
    shifted = onset_ranks(grid.time_of_beat(np.floor(hb) + (hb % 1 + 0.5) % 1))
    return dict(K=len(head_ms), sections=len(scopes),
                intensity_workload_rho=correlation(audio[:, 0], rates, rank=True),
                flux_workload_rho=correlation(audio[:, 1], rates, rank=True),
                low_intensity_wh_s=low_rate, high_intensity_wh_s=high_rate,
                low_high_ratio=low_rate / high_rate if high_rate else None,
                low_intensity_rest=float(np.average(rest[low], weights=durations[low])),
                high_intensity_rest=float(np.average(rest[high], weights=durations[high])),
                wh_s=float(workload.sum() / durations.sum()),
                onset_rank=float(np.mean(selected)) if len(selected) else None,
                shifted_onset_rank=float(np.mean(shifted)) if len(shifted) else None,
                onset_heads=len(selected), section_edges_ms=edges.tolist(),
                section_intensity=audio[:, 0].tolist(), section_flux=audio[:, 1].tolist(),
                section_wh_s=rates.tolist(), section_rest=rest.tolist())


def lattice_measures(target):
    """Triple share of all beats, occupied beats, retained heads and triple-only slots."""
    triple = np.asarray(target['lattice']) == 1
    counts = np.asarray(target['count'])
    n = int(counts.sum())
    exclusive = (np.arange(16) < 12) & (np.arange(16) % 3 != 0)
    return dict(beats=len(counts), occupied_beats=int((counts > 0).sum()), heads=n,
                triple_beats=int(triple.sum()), triple_occupied_beats=int((triple & (counts > 0)).sum()),
                triple_heads=int(counts[triple].sum()),
                triple_only_heads=int(target['mask'][triple][:, exclusive].sum()),
                mean_abs_snap_ms=float(np.abs(target['err_ms']).mean()),
                p95_abs_snap_ms=float(np.quantile(np.abs(target['err_ms']), 0.95)),
                merged=int(target['merged']))
