"""Training data for the figure generator: per chart its beat figures on the chart's own grid (G1),
per audio file its standardised log-Mel.

Build on the mac (reads the R2 cache, the corpus parquet and an earlier full-corpus Mel build)::

    python -m ensomi_model.audio_rows.beatdata --out <dir> --features <dir> --workers 9

``--features`` holds one ``<audio sha256>.npy`` [F, >=128] float16 per audio file whose first 128
channels are ``audio.features``' standardised log-Mel without the osu! decoder lead (the
2026-10-08 build ``audio-rows-20261008/astra/features``). Files whose ffmpeg decode needs that
lead are recomputed with it into ``<dir>/mel``; three files without a lead are recomputed too and
compared (``mel_check`` in the record).

``<dir>`` holds ``charts.npz`` (per-beat arrays of all charts, concatenated) and ``index.json``:
chart rows with their offsets, the figure vocabulary (fit_train figures seen at least 3 times),
the song-request normalisation, the star -> request map and the band requests. A chart's song
request is log(W_H / T) over its beats clipped to [0, T] (``data.section_label`` over all beats):
W_H the ras-v1 rows-only reference workload, T the song duration.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import torch

from ..r2.cache import load_chart
from ..r2.common import GridArrays, grid_from_arrays
from ..r2.strain import StrainTrace
from .audio import HOP_MS, MEL_BINS
from .data import beat_workload, section_label
from .figures import NPOS, POSITIONS, Vocab, beat_figures
from .lattice import snap

STAR_BINS = np.arange(2.0, 6.01, 0.25)
SECTION_BEATS = (8, 16, 32)


def select(cache_root: Path, corpus: Path, audio_root: Path = Path('.')) -> list[dict]:
    """All R2 cache charts whose audio is present, with star and audio path."""
    import pyarrow.parquet as pq
    index = pq.read_table(cache_root / 'index.parquet').to_pandas()
    corp = pq.read_table(corpus, columns=['sha256', 'set_dir', 'audio_filename', 'audio_present']).to_pandas()
    index = index.merge(corp, on='sha256', how='inner')
    index = index[index.audio_present.astype(bool)].sort_values('sha256')
    return [dict(sha=r.sha256, file=r.file, group=r.group_id, role=r.role, band=int(r.band), star=float(r.star),
                 song_ms=float(r.song_ms), audio=str(audio_root / r.set_dir / r.audio_filename), key=r.audio_sha256)
            for r in index.itertuples()]


def chart_arrays(dec, full: bool = False) -> dict:
    """Per-beat figures, workload, unclipped beat edges and BPM of one cached chart on its own grid."""
    grid = GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars))
    t = snap(dec.head_ms, grid, dec.song_ms)
    b0, nb = t['b0'], t['nb']
    edges = grid.time_of_beat(b0 + np.arange(nb + 1))
    beat_ms = np.clip(edges, 0.0, dec.song_ms)
    wh = beat_workload(dec.head_ms, dec.song_ms, beat_ms)
    out = dict(b0=b0, nb=nb, edges=edges, wh=wh, figure=beat_figures(t['lattice'], t['mask']),
               bpm=grid.bpm(beat_ms[:-1]), merged=t['merged'], K=len(dec.head_ms),
               label=section_label(wh, beat_ms, 0, nb))
    if full:
        trace = StrainTrace(dec.head_ms, dec.song_ms).extend(dec.actions, dec.gap_release_ms)
        out.update(W=float(trace.workload_prefix[-1]), W_H=float(trace.reference_prefix[-1]))
    return out


def _chart_job(args):
    row, cache_root = args
    return row['sha'], chart_arrays(load_chart(Path(cache_root) / 'charts' / row['file']), row['role'] == 'fit_dev')


def song_mel(path) -> np.ndarray:
    """[F, 128] float16 standardised log-Mel on osu!'s clock, as ``audio.Listener.listen`` computes it."""
    from ..features.audio import load_audio_file
    from ..features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
    from .audio import _lead
    from .decoder import lead_seconds
    sr = MUSIC_MEL_CACHE_CONFIG.sample_rate
    wave = _lead(load_audio_file(path, sr), lead_seconds(path, 'ffmpeg'), sr)
    log_mel = compute_log_mel_10ms(wave, sample_rate=sr, config=MUSIC_MEL_CACHE_CONFIG)
    return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)).astype(np.float16)


def _mel_job(args):
    from .decoder import lead_seconds
    key, path, out, check = args
    try:
        lead = lead_seconds(path, 'ffmpeg')
        if not (lead > 0 or check):
            return key, lead, None, None
        mel = song_mel(path)
    except Exception as exc:    # an unreadable file is recorded, not fatal
        return key, None, None, f'{type(exc).__name__}: {exc}'[:200]
    if lead > 0:
        np.save(Path(out) / f'{key}.npy', mel)
        return key, lead, None, None
    return key, lead, mel, None


def star_map(stars, labels) -> dict:
    """Median song request per 0.25-star bin of fit_train charts, made non-decreasing."""
    stars, labels = np.asarray(stars), np.asarray(labels)
    centres, medians = [], []
    for a, b in zip(STAR_BINS[:-1], STAR_BINS[1:]):
        sel = (stars >= a) & (stars < b)
        if sel.sum() >= 20:
            centres.append(float((a + b) / 2))
            medians.append(float(np.median(labels[sel])))
    return dict(star=centres, request=np.maximum.accumulate(medians).tolist())


def build(cache_root: Path, corpus: Path, features: Path, out: Path, workers: int, limit=None):
    t0 = time.time()
    rows = select(cache_root, corpus)
    if limit:
        rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:limit]]
    (out / 'mel').mkdir(parents=True, exist_ok=True)
    paths = {r['key']: r['audio'] for r in rows}
    check = set(sorted(paths)[:3])
    leads, failed, mel_check = {}, {}, {}
    with get_context('spawn').Pool(workers) as pool:
        for key, lead, mel, err in pool.imap_unordered(
                _mel_job, [(k, p, str(out / 'mel'), k in check) for k, p in paths.items()], chunksize=8):
            if err:
                failed[key] = err
                continue
            leads[key] = lead
            if mel is not None:
                old = np.load(features / f'{key}.npy', mmap_mode='r')[:, :MEL_BINS]
                n = min(len(old), len(mel))
                mel_check[key] = dict(frames=[len(old), len(mel)],
                                      max_abs=float(np.abs(old[:n].astype(np.float32) - mel[:n]).max()))
        print(json.dumps(dict(stage='mel', files=len(paths), led=sum(v > 0 for v in leads.values()),
                              failed=len(failed), seconds=round(time.time() - t0, 1))), flush=True)
        rows = [r for r in rows if r['key'] in leads]
        charts = {}
        for i, (sha, c) in enumerate(pool.imap_unordered(_chart_job, [(r, str(cache_root)) for r in rows],
                                                         chunksize=16)):
            charts[sha] = c
            if (i + 1) % 2000 == 0:
                print(json.dumps(dict(stage='charts', done=i + 1, total=len(rows),
                                      seconds=round(time.time() - t0, 1))), flush=True)
    train = [r for r in rows if r['role'] == 'fit_train']
    vocab = Vocab.fit(np.concatenate([charts[r['sha']]['figure'] for r in train]))
    labels = np.array([charts[r['sha']]['label'] for r in train])
    arrays = {k: [] for k in ('figure', 'wh', 'bpm', 'edges')}
    start = estart = 0
    for r in rows:
        c = charts[r['sha']]
        r.update(b0=int(c['b0']), nb=int(c['nb']), start=start, estart=estart, K=int(c['K']),
                 merged=int(c['merged']), label=float(c['label']),
                 **({k: c[k] for k in ('W', 'W_H') if k in c}), lead_ms=1000.0 * leads[r['key']])
        for k in arrays:
            arrays[k].append(c[k])
        start += c['nb']
        estart += c['nb'] + 1
    figure = np.concatenate(arrays['figure'])
    np.savez(out / 'charts.npz', figure=figure.astype(np.int32), wh=np.concatenate(arrays['wh']).astype(np.float32),
             bpm=np.concatenate(arrays['bpm']).astype(np.float32), edges=np.concatenate(arrays['edges']))
    known = vocab.encode(figure) >= 0
    is_train = np.concatenate([np.full(r['nb'], r['role'] == 'fit_train') for r in rows])
    record = dict(
        charts=len(rows), train=len(train), dev=len(rows) - len(train), audio_files=len(leads),
        failed_audio=failed, led_audio=sum(v > 0 for v in leads.values()), mel_check=mel_check,
        features=str(features), vocab=vocab.figures.tolist(), positions=POSITIONS.tolist(),
        coverage=dict(train=float(known[is_train].mean()), dev=float(known[~is_train].mean()),
                      beats_train=int(is_train.sum()), beats_dev=int((~is_train).sum())),
        label_mean=float(labels.mean()), label_std=float(labels.std()),
        star_map=star_map([r['star'] for r in train], labels),
        bands={b: float(np.median([charts[r['sha']]['label'] for r in train if r['band'] == b])) for b in range(2, 6)},
        seconds=time.time() - t0)
    (out / 'index.json').write_text(json.dumps(dict(record, rows=rows)))
    print(json.dumps({k: v for k, v in record.items() if k not in ('vocab', 'failed_audio')} | dict(vocab=len(vocab))),
          flush=True)
    return record


class Beats:
    """A build read for training: per-beat arrays in memory, Mel memory-mapped per audio file."""

    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / 'index.json').read_text())
        self.record = {k: v for k, v in index.items() if k != 'rows'}
        self.rows = index['rows']
        with np.load(self.root / 'charts.npz') as z:
            self.figure, self.wh, self.bpm, self.edges = z['figure'], z['wh'], z['bpm'], z['edges']
        self.vocab = Vocab(index['vocab'])
        self.ids = self.vocab.encode(self.figure)
        self.label_mean, self.label_std = index['label_mean'], index['label_std']
        self.features_dir = Path(index['features'])
        self.train = [r for r in self.rows if r['role'] == 'fit_train' and r['nb'] > 16]
        self.dev = [r for r in self.rows if r['role'] == 'fit_dev' and r['nb'] > 16]

    @lru_cache(maxsize=64)      # bounds open memory maps
    def mel(self, key: str) -> np.ndarray:
        led = self.root / 'mel' / f'{key}.npy'
        return np.load(led if led.exists() else self.features_dir / f'{key}.npy', mmap_mode='r')

    def standardise(self, label):
        return (np.asarray(label) - self.label_mean) / self.label_std

    def chart(self, row) -> dict:
        s, e, n = row['start'], row['estart'], row['nb']
        return dict(ids=self.ids[s:s + n], figure=self.figure[s:s + n], wh=self.wh[s:s + n], bpm=self.bpm[s:s + n],
                    edges=self.edges[e:e + n + 1])

    def window(self, row, start: int, length: int, rng, *, shift_ms=0.0, section_rate=0.0, section_prob=0.3,
               request=None, sections=None, roll=0.0) -> dict:
        """Teacher-forcing arrays of beats [start, start + length) of one chart.

        The audio is shifted by a draw uniform in ±``shift_ms``. With probability ``section_rate``
        the window is cut into sections of 8/16/32 beats and each carries its own true log(W_H / s)
        with probability ``section_prob``. ``request`` replaces the chart's song request; ``sections``
        (a list of (a, b, log W_H/s) in window beats) replaces the drawn sections. ``roll`` rotates the song's
        Mel by that fraction of its length (an ablation)."""
        c = self.chart(row)
        n = min(length, row['nb'] - start)
        edges = c['edges'][start:start + n + 1]
        shift = float(rng.uniform(-shift_ms, shift_ms))
        mel = self.mel(row['key'])
        if roll:
            mel = np.roll(mel[:, :MEL_BINS], int(roll * len(mel)), 0)
        pos_ms = edges[:-1, None] + np.diff(edges)[:, None] * POSITIONS[None] / 48.0 + shift
        f0 = min(max(int(edges[0] / HOP_MS) - 64, 0), len(mel) - 1)
        f1 = min(max(int(edges[-1] / HOP_MS) + 64, f0 + 1), len(mel))
        pos_frames = np.clip((pos_ms / HOP_MS).astype(np.int64) - f0, 0, f1 - f0 - 1)
        beat_frames = np.clip(((edges + shift) / HOP_MS).astype(np.int64) - f0, 0, f1 - f0)
        beat_frames = np.stack((beat_frames[:-1], np.maximum(beat_frames[1:], beat_frames[:-1] + 1)), 1).clip(0, f1 - f0)
        song = self.standardise(row['label'] if request is None else request)
        sec = np.zeros((n, 2), dtype=np.float32)
        if sections is None and section_rate and rng.random() < section_rate:
            beat_ms = np.clip(c['edges'], 0.0, row['song_ms'])
            a, sections = 0, []
            while a < n:
                b = min(a + int(rng.choice(SECTION_BEATS)), n)
                if rng.random() < section_prob:
                    sections.append((a, b, section_label(c['wh'], beat_ms, start + a, start + b)))
                a = b
        for a, b, value in sections or ():
            sec[a:b] = (self.standardise(value), 1.0)
        beat_ms = np.clip(edges[:-1], 0.0, row['song_ms'])
        extra = np.column_stack((np.eye(4)[(row['b0'] + start + np.arange(n)) % 4], beat_ms / row['song_ms'],
                                 c['bpm'][start:start + n] / 120.0, np.full(n, song), sec)).astype(np.float32)
        return dict(mel=torch.from_numpy(np.asarray(mel[f0:f1, :MEL_BINS], dtype=np.float32)),
                    pos_frames=torch.from_numpy(pos_frames), beat_frames=torch.from_numpy(beat_frames),
                    extra=torch.from_numpy(extra), ids=torch.from_numpy(c['ids'][start:start + n].copy()))


EXTRA_DIM = 4 + 1 + 1 + 1 + 2       # bar phase, progress, BPM / 120, song request, section request and flag
SONG, SECTION = 6, 7                # ``extra`` columns of the song request and the section request (flag at 8)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cache', type=Path, default=Path('artifacts/r2-cache/v1'))
    p.add_argument('--corpus', type=Path, default=Path('data/r2-corpus.parquet'))
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=9)
    p.add_argument('--limit', type=int, help='random subset of charts, for a smoke run')
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    build(a.cache, a.corpus, a.features, a.out, a.workers, a.limit)


if __name__ == '__main__':
    main()
