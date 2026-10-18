"""The Mel cache: one float16 [F, 128] array per audio file, keyed by the file's sha256.

Layout: ``<root>/<MelConfig.key()>/<sha[:2]>/<sha>.npy``. Values are the
unstandardised log-Mel of `mel.log_mel` on osu!'s clock (frame ``i`` at
``10 i`` ms). Building is resumable: files already present are skipped, and
each array is written to a temporary name and renamed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .mel import DEFAULT, MelConfig, log_mel


def mel_path(root: str | os.PathLike, sha: str, cfg: MelConfig = DEFAULT) -> Path:
    return Path(root) / cfg.key() / sha[:2] / f'{sha}.npy'


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_mel(root, sha: str, cfg: MelConfig = DEFAULT, mmap: bool = True) -> np.ndarray:
    return np.load(mel_path(root, sha, cfg), mmap_mode='r' if mmap else None)


def build_one(job: tuple[str, str], root: str, cfg: MelConfig = DEFAULT) -> dict:
    """Cache one ``(audio path, expected sha256)``; returns an index record."""
    from .decode import decode
    path, sha = job
    out = mel_path(root, sha, cfg)
    rec = dict(sha=sha, path=path)
    if out.exists():
        arr = np.load(out, mmap_mode='r')
        rec.update(status='present', frames=int(arr.shape[0]))
        return rec
    t0 = time.perf_counter()
    try:
        actual = sha256_file(path)
        if actual != sha:
            rec.update(status='sha_mismatch', actual=actual)
            return rec
        dec = decode(path, cfg)
        mel = log_mel(dec.wave, cfg).astype(np.float16)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f'.tmp{os.getpid()}.npy')
        np.save(tmp, mel)
        os.replace(tmp, out)
        rec.update(status='built', frames=int(mel.shape[0]), decoder=dec.decoder, header=dec.header,
                   lead_samples=dec.lead_samples, source_rate=dec.source_rate,
                   seconds=round(time.perf_counter() - t0, 3))
    except Exception as exc:  # recorded, not raised: one bad file must not stop the cache
        rec.update(status='error', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-800:])
    return rec


def _init_worker():
    try:
        import torch
        torch.set_num_threads(1)
    except ModuleNotFoundError:
        pass


def _run(args):
    job, root = args
    return build_one(job, root)


def build(jobs: list[tuple[str, str]], root: str | os.PathLike, index_path: str | os.PathLike,
          processes: int = 4, progress=None) -> dict:
    """Cache every job with ``processes`` workers; append one JSON line per job to ``index_path``."""
    import multiprocessing as mp
    root = str(root)
    counts: dict[str, int] = {}
    t0 = time.time()
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'a') as index:
        if processes <= 1:
            results = (_run((j, root)) for j in jobs)
            pool = None
        else:
            pool = mp.get_context('spawn').Pool(processes, initializer=_init_worker)
            results = pool.imap_unordered(_run, [(j, root) for j in jobs], chunksize=4)
        try:
            for k, rec in enumerate(results, 1):
                counts[rec['status']] = counts.get(rec['status'], 0) + 1
                index.write(json.dumps(rec) + '\n')
                index.flush()
                if progress is not None and (k % 50 == 0 or k == len(jobs)):
                    progress(dict(done=k, total=len(jobs), counts=dict(counts), wall_s=round(time.time() - t0, 1)))
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    return dict(counts=counts, wall_s=round(time.time() - t0, 1), n=len(jobs))
