"""Training data from the R2 cache: per audio file the generator's inputs, per chart its per-beat targets.

Build on the mac (reads the R2 cache, the corpus parquet and the audio; runs BeatThis and the
fitter once per audio file)::

    python -m ensomi_model.audio_rows.data --out <dir> --beatthis <beat_this-final0.ckpt>

``--audio-root`` resolves relative corpus audio paths. ``--train-audio`` and
``--dev-audio`` bound a seeded audio-file sample, retaining every eligible chart of
each selected file; omitting them keeps the complete split. ``selection.json``
records membership before feature extraction.

``<dir>`` holds:

- ``audio/<key>.npy``: ``audio.Song.features`` per audio file (key: the corpus audio SHA-256);
- ``chart/<sha>.npz`` and ``fitted/<sha>.npz``: per-chart targets on the chart's own red-line
  grid (G1) and on the corrected fitted grid of its audio (G2), both against the cache's song end T;
- ``index.json``: chart rows, per audio file the fitter's raw segments and duration, failed audio
  files, and the label normalisation and band requests both grids share.

The section label of beats [a, b) is log(max(W_H / s, 0.001)). W_H is the ras-v1 reference
workload (``r2.strain.StrainTrace``) of the section's head rows; the reference strain runs from
the chart's first row and is not reset at a section. s is the section's duration, with the
first and last beat clipped to [0, T]. A head belongs to the last beat starting at or before it,
the half-open scopes of ``StrainTrace.row_bounds``. Normalisation and band requests come from
fit_train charts on the red-line grid in 16-beat sections, the last one partial; a band's
request is the log of its median W_H / s.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import torch

from ..r2.cache import load_chart
from ..r2.common import GridArrays, grid_from_arrays
from ..r2.strain import StrainTrace
from .audio import HOP_MS
from .grid import corrected_grid
from .lattice import slot_times, snap
from .model import TOKEN_DIM, extra_inputs, token

GRIDS = ('chart', 'fitted')
SECTION_BEATS = 16
LABEL_FLOOR = 1e-3              # W_H / s of a section without heads
MARGIN_FRAMES = 32              # audio frames beyond a window's ends; covers the encoder's ±30


def sections(nb: int, length: int = SECTION_BEATS) -> list[tuple[int, int]]:
    """Consecutive [a, b) beat ranges of ``length`` beats; the last one may be shorter."""
    return [(a, min(a + length, nb)) for a in range(0, nb, length)]


def beat_workload(head_ms, song_ms: float, beat_ms) -> np.ndarray:
    """W_H per beat [len(beat_ms) - 1] for heads within the beats' clipped edges ``beat_ms``."""
    head_ms = np.asarray(head_ms, dtype=np.float64)
    if not len(head_ms):
        return np.zeros(len(beat_ms) - 1)
    charges = np.diff(StrainTrace(head_ms, song_ms).reference_prefix)
    beat = np.clip(np.searchsorted(beat_ms, head_ms, side='right') - 1, 0, len(beat_ms) - 2)
    return np.bincount(beat, charges, minlength=len(beat_ms) - 1)


def section_label(wh_beat, beat_ms, a: int, b: int) -> float:
    """log(W_H / s) over beats [a, b), floored at LABEL_FLOOR."""
    return float(np.log(max(wh_beat[a:b].sum() / ((beat_ms[b] - beat_ms[a]) / 1000.0), LABEL_FLOOR)))


def chart_targets(head_ms, song_ms: float, grid: GridArrays) -> dict:
    """Per-beat targets and inputs of one chart on ``grid``; ``beat_ms`` are the clipped beat edges."""
    t = snap(head_ms, grid, song_ms)
    b0, nb = t['b0'], t['nb']
    beat_ms = np.clip(grid.time_of_beat(b0 + np.arange(nb + 1)), 0.0, song_ms)
    return dict(b0=b0, lattice=t['lattice'].astype(np.int8), count=t['count'].astype(np.int8), mask=t['mask'],
                wh_beat=beat_workload(head_ms, song_ms, beat_ms), beat_ms=beat_ms, slot_ms=slot_times(grid, b0, nb),
                bpm=grid.bpm(beat_ms[:-1]), err_ms=t['err_ms'], merged=t['merged'])


def frame_indices(beat_ms, slot_ms, f0: int, n: int, align=None):
    """Frame spans [nb, 2] and slot frames [nb, 2, 16] in a frame window [f0, f0 + n).

    ``align`` maps grid time to audio time (identity when None); NaN slots map to frame 0."""
    align = align or (lambda t: t)
    frames = np.clip((align(beat_ms) / HOP_MS).astype(int) - f0, 0, n)
    spans = np.clip(np.stack((frames[:-1], np.maximum(frames[1:], frames[:-1] + 1)), 1), 0, n)
    slots = np.where(np.isnan(slot_ms), 0, (align(np.nan_to_num(slot_ms)) / HOP_MS).astype(int) - f0)
    return torch.from_numpy(spans), torch.from_numpy(np.clip(slots, 0, n - 1))


def select(cache_root: Path, corpus: Path, audio_root: Path = Path('.')) -> list[dict]:
    """fit_train and fit_dev rows of the R2 cache whose audio is present."""
    import pyarrow.parquet as pq
    index = pq.read_table(cache_root / 'index.parquet').to_pandas()
    corp = pq.read_table(corpus, columns=['sha256', 'set_dir', 'audio_filename', 'audio_present']).to_pandas()
    index = index.merge(corp, on='sha256', how='inner')
    index = index[index.audio_present.astype(bool) & index.role.isin(('fit_train', 'fit_dev'))].sort_values('sha256')
    rows = []
    for r in index.itertuples():
        audio = str(audio_root / r.set_dir / r.audio_filename)
        key = r.audio_sha256 if isinstance(r.audio_sha256, str) else hashlib.sha256(audio.encode()).hexdigest()
        rows.append(dict(sha=r.sha256, file=r.file, group=r.group_id, role=r.role, band=int(r.band),
                         song_ms=float(r.song_ms), audio=audio, key=key))
    return rows


def audio_subset(rows, train_audio=None, dev_audio=None, seed=20261009):
    """Seeded audio-file sample per split, retaining every chart of each selected file.

    ``None`` keeps a split in full. Sampling sorted keys makes membership independent of
    row order; fit_train and fit_dev must not share audio. Limits must be positive.
    """
    keys = {role: sorted({r['key'] for r in rows if r['role'] == role})
            for role in ('fit_train', 'fit_dev')}
    if set(keys['fit_train']) & set(keys['fit_dev']):
        raise ValueError('Audio files must not cross fit_train and fit_dev')
    rng = np.random.default_rng(seed)
    keep = set()
    for role, limit in zip(keys, (train_audio, dev_audio)):
        if limit is not None and limit < 1:
            raise ValueError('Audio sample limits must be positive')
        order = rng.permutation(keys[role])
        keep.update(order if limit is None else order[:limit])
    return [r for r in rows if r['key'] in keep]


_LISTENER = None


def _init_worker(beatthis):
    global _LISTENER
    from .audio import Listener
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _LISTENER = Listener(beatthis)


def _listen_job(args):
    key, path, out = args
    try:
        song = _LISTENER.listen(path)
    except Exception as exc:    # one undecodable or beatless file must not stop the build
        return key, None, f'{type(exc).__name__}: {exc}'[:200]
    np.save(Path(out) / f'{key}.npy', song.features)
    return key, dict(segments=song.segments.tolist(), song_ms=song.song_ms, seconds=song.seconds), None


def build(cache_root: Path, corpus: Path, out: Path, beatthis: Path, workers: int, *,
          audio_root: Path = Path('.'), train_audio=None, dev_audio=None, selection_seed=20261009):
    t0 = time.time()
    rows = audio_subset(select(cache_root, corpus, audio_root), train_audio, dev_audio, selection_seed)
    for d in ('audio', *GRIDS):
        (out / d).mkdir(parents=True, exist_ok=True)
    paths = {r['key']: r['audio'] for r in rows}
    selection = dict(train_audio=train_audio, dev_audio=dev_audio, seed=selection_seed, rows=rows)
    (out / 'selection.json').write_text(json.dumps(selection))
    audio, failed = {}, {}
    jobs = [(k, p, str(out / 'audio')) for k, p in paths.items()]
    with get_context('spawn').Pool(workers, initializer=_init_worker, initargs=(str(beatthis),)) as pool:
        for i, (key, info, err) in enumerate(pool.imap_unordered(_listen_job, jobs)):
            if err:
                failed[key] = err
            else:
                audio[key] = info
            if (i + 1) % 50 == 0:
                print(json.dumps(dict(stage='audio', done=i + 1, total=len(jobs), failed=len(failed),
                                      seconds=round(time.time() - t0, 1))), flush=True)
    kept, labels, band_wh = [], [], {b: [] for b in range(2, 6)}
    for i, r in enumerate(rows):
        if r['key'] not in audio:
            continue
        dec = load_chart(cache_root / 'charts' / r['file'])
        grids = dict(chart=GridArrays.from_grid(grid_from_arrays(dec.grid_segments, dec.grid_bars)),
                     fitted=corrected_grid(audio[r['key']]['segments'], dec.song_ms)[0])
        for name, grid in grids.items():
            t = chart_targets(dec.head_ms, dec.song_ms, grid)
            np.savez(out / name / f"{r['sha']}.npz", **t)
            if name == 'chart' and r['role'] == 'fit_train':
                for a, b in sections(len(t['count'])):
                    labels.append(section_label(t['wh_beat'], t['beat_ms'], a, b))
                    band_wh[r['band']].append(np.exp(labels[-1]))
        kept.append(r)
        if (i + 1) % 500 == 0:
            print(json.dumps(dict(stage='targets', done=i + 1, total=len(rows), seconds=round(time.time() - t0, 1))),
                  flush=True)
    summary = dict(charts=len(kept), audio_files=len(audio), failed_audio=len(failed),
                   train=sum(r['role'] == 'fit_train' for r in kept), dev=sum(r['role'] == 'fit_dev' for r in kept),
                   label_mean=float(np.mean(labels)), label_std=float(np.std(labels)),
                   bands={b: float(np.log(np.median(v))) for b, v in band_wh.items() if v},
                   seconds=time.time() - t0)
    (out / 'index.json').write_text(json.dumps(dict(summary, rows=kept, audio=audio, failed=failed)))
    print(json.dumps(summary), flush=True)
    return summary


class Dataset:
    """A build read for one grid: per-chart targets in memory, audio features memory-mapped on demand."""

    def __init__(self, root, grid: str = 'chart'):
        self.root = Path(root)
        index = json.loads((self.root / 'index.json').read_text())
        self.label_mean, self.label_std = index['label_mean'], index['label_std']
        self.bands = {int(b): v for b, v in index['bands'].items()}
        self.audio = index['audio']
        self.rows = index['rows']
        self.charts = {}
        for r in self.rows:
            with np.load(self.root / grid / f"{r['sha']}.npz") as z:
                self.charts[r['sha']] = {k: z[k] for k in ('b0', 'lattice', 'count', 'mask', 'wh_beat', 'beat_ms',
                                                           'slot_ms', 'bpm')}
        self.train = [r for r in self.rows if r['role'] == 'fit_train' and len(self.charts[r['sha']]['count']) > 8]
        self.dev = [r for r in self.rows if r['role'] == 'fit_dev' and len(self.charts[r['sha']]['count']) > 8]

    @lru_cache(maxsize=32)      # bounds open memory maps
    def features(self, key: str) -> np.ndarray:
        return np.load(self.root / 'audio' / f'{key}.npy', mmap_mode='r')

    def standardise(self, label):
        return (np.asarray(label) - self.label_mean) / self.label_std

    def window(self, row, start: int, length: int, rng, *, jitter_ms=0.0, tempo_error=0.0, history_dropout=0.0,
               label_dropout=0.0, section_beats=(16,), label=None) -> dict:
        """Teacher-forcing tensors of beats [start, start + length) of one chart.

        One affine audio alignment per window: a lag uniform in ±``jitter_ms`` and a tempo error
        uniform in ±``tempo_error`` about the song's midpoint; the targets stay on the grid. Both
        draws happen at zero width too, so runs that differ only in widths share every other draw.
        ``label`` (standardised, per beat) replaces the drawn sections of 8/16/32 beats."""
        c = self.charts[row['sha']]
        feats = self.features(row['key'])
        nb = min(length, len(c['count']) - start)
        beat_ms = c['beat_ms'][start:start + nb + 1]
        shift = float(rng.uniform(-jitter_ms, jitter_ms))
        rate = 1.0 + float(rng.uniform(-tempo_error, tempo_error))
        anchor = row['song_ms'] / 2

        def align(t):
            return anchor + (t - anchor) * rate + shift

        f0 = min(max(int(align(beat_ms[0]) / HOP_MS) - MARGIN_FRAMES, 0), len(feats) - 1)
        f1 = min(max(int(align(beat_ms[-1]) / HOP_MS) + MARGIN_FRAMES, f0 + 1), len(feats))
        beat_frames, slot_frames = frame_indices(beat_ms, c['slot_ms'][start:start + nb], f0, f1 - f0, align)
        lattice = torch.from_numpy(c['lattice'][start:start + nb].astype(np.int64))
        count = torch.from_numpy(c['count'][start:start + nb].astype(np.int64))
        prev = torch.zeros(nb, TOKEN_DIM)       # beat i reads the decision of beat i - 1
        before = slice(max(start - 1, 0), start + nb - 1)
        n_prev = before.stop - before.start
        if n_prev > 0:
            prev[nb - n_prev:] = token(torch.from_numpy(c['lattice'][before].astype(np.int64)),
                                       torch.from_numpy(c['count'][before].astype(np.int64)),
                                       torch.from_numpy(c['mask'][before]))
        if history_dropout and rng.random() < history_dropout:
            prev.zero_()
        if label is None and not (label_dropout and rng.random() < label_dropout):
            label = np.zeros(nb)
            a = 0
            while a < nb:
                b = min(a + int(rng.choice(section_beats)), nb)
                label[a:b] = self.standardise(section_label(c['wh_beat'], c['beat_ms'], start + a, start + b))
                a = b
        extra = extra_inputs(c['b0'] + start + np.arange(nb), beat_ms[:-1], c['bpm'][start:start + nb],
                             row['song_ms'], label)
        return dict(feats=torch.from_numpy(feats[f0:f1].astype(np.float32)), beat_frames=beat_frames,
                    slot_frames=slot_frames, prev_tokens=prev, extra=extra, lattice=lattice, count=count,
                    mask=torch.from_numpy(c['mask'][start:start + nb]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--cache', default='artifacts/r2-cache/v1')
    p.add_argument('--corpus', default='data/r2-corpus.parquet')
    p.add_argument('--out', required=True)
    p.add_argument('--beatthis', required=True, help='BeatThis final0 checkpoint')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--audio-root', type=Path, default=Path('.'), help='Root for relative corpus audio paths')
    p.add_argument('--train-audio', type=int, help='Sample this many fit_train audio files; default: all')
    p.add_argument('--dev-audio', type=int, help='Sample this many fit_dev audio files; default: all')
    p.add_argument('--selection-seed', type=int, default=20261009)
    a = p.parse_args(argv)
    build(Path(a.cache), Path(a.corpus), Path(a.out), Path(a.beatthis), a.workers,
          audio_root=a.audio_root, train_audio=a.train_audio, dev_audio=a.dev_audio, selection_seed=a.selection_seed)


if __name__ == '__main__':
    main()
