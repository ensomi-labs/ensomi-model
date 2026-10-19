"""Per-song inputs for N1 v0: the Mel cache (built where missing), song statistics and the song-memory input.

- Mel: `ensomi_n.features.cache` (float16 [F, 128], 100 fps on osu!'s clock), built for any item
  whose array is missing.
- Statistics: the mean and standard deviation of the whole song's log-Mel (the loader
  standardises crops with them) and the frame count, so a training job opens no song in full.
- Memory input (``memo-v1``): the log-Mel averaged over 4 frames (25 fps) and over band pairs (64
  bands), float16 [F // 4, 64], unstandardised, one file per audio sha256 beside the Mel cache.
  The song memory reads the whole song from it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

MEMO_HOP = 4
MEMO_BANDS = 64
MEMO_VERSION = 'memo-v1'


def memo_path(bulk: str | os.PathLike, sha: str) -> Path:
    return Path(bulk) / MEMO_VERSION / sha[:2] / f'{sha}.npy'


def memo_from_mel(mel: np.ndarray) -> np.ndarray:
    m = np.asarray(mel, dtype=np.float32)
    f = (len(m) // MEMO_HOP) * MEMO_HOP
    m = m[:f].reshape(-1, MEMO_HOP, m.shape[1]).mean(axis=1)
    return m.reshape(len(m), MEMO_BANDS, -1).mean(axis=2).astype(np.float16)


def stats_one(args) -> dict:
    sha, bulk = args
    from ..features.cache import mel_path
    try:
        mp = mel_path(Path(bulk) / 'mel', sha)
        mel = np.load(mp).astype(np.float32)
        out = memo_path(bulk, sha)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f'.tmp{os.getpid()}.npy')
            np.save(tmp, memo_from_mel(mel))
            os.replace(tmp, out)
        return dict(sha=sha, frames=int(len(mel)), mean=float(mel.mean()), std=float(mel.std() + 1e-6))
    except Exception as exc:  # noqa: BLE001
        return dict(sha=sha, error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-400:])


def build(items: list[dict], *, root: Path, bulk: Path, run: Path, processes: int = 8, progress=None) -> dict:
    """Mel for missing audio, then statistics and memory input for every item; returns the song index."""
    from ..features import cache
    import multiprocessing as mp
    seen = {}
    for it in items:
        if it.get('audio_sha256'):
            seen.setdefault(it['audio_sha256'], str(root / it['audio_rel']))
    jobs = [(p, s) for s, p in sorted(seen.items())]
    t0 = time.time()
    res = cache.build(jobs, bulk / 'mel', run / 'cache-index.jsonl', processes=processes,
                      progress=(lambda r: progress(dict(step='cache', **r))) if progress else None)
    index = {}
    with mp.get_context('spawn').Pool(processes) as pool:
        for k, r in enumerate(pool.imap_unordered(stats_one, [(s, str(bulk)) for s in seen], chunksize=8), 1):
            index[r['sha']] = r
            if progress and (k % 500 == 0 or k == len(seen)):
                progress(dict(step='song_stats', done=k, total=len(seen), wall_s=round(time.time() - t0, 1)))
    return dict(cache=res, index=index, errors=[r for r in index.values() if 'error' in r][:30],
                n_errors=sum('error' in r for r in index.values()))
