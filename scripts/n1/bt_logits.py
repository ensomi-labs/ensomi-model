"""BeatThis logits for N1's rung-0/1 panels (R side: it reads audio exactly as R2's Listener does).

Writes one ``<audio sha256>.npz`` (``beat``, ``down``: float32 logits at 50 fps) per item under
``<bulk>/bt-<ckpt>-<checkpoint sha256[:12]>/``. Modes:

- ``map``: copy stored logits into that layout. ``--stored fitdev`` maps
  ``opt/ft/acc/models/final0/logits/<mel_key>.npz`` through ``fits.jsonl`` (key -> audio_rel) for the
  fit_dev panel items; ``--stored reset`` maps ``repr/reset/logits/<i:04d>.npz`` through
  ``repr/reset/fits.jsonl`` (i -> audio_rel). An item whose audio path disagrees is skipped and listed.
- ``compute``: run BeatThis on each item's audio, as ``audio_rows.audio.Listener.listen`` does: BeatThis's
  loader (libsndfile) with the ``mpg123`` lead of ``audio_rows.decoder``, falling back to the pydub waveform
  with the ``ffmpeg`` lead when libsndfile fails. ``--verify N`` first recomputes N mapped items and reports
  the largest logit difference against the stored ones.

Resumable (existing outputs are skipped) and time-capped (``--max-minutes``). Run from the project root:

    PYTHONPATH=src:<src>/n/src .venv/bin/python <src>/scripts/n1/bt_logits.py compute --split <run>/split.json \
        --subset val_hard,val_random --ckpt final0 --run <run>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

CKPT_DIR = Path.home() / '.cache/torch/hub/checkpoints'
TIMING = Path('artifacts/timing-20261009')


def ckpt_path(name: str) -> Path:
    return CKPT_DIR / f'beat_this-{name}.ckpt'


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def out_dir(bulk: Path, name: str) -> tuple[Path, str]:
    sha = sha256_file(ckpt_path(name))
    return bulk / f'bt-{name}-{sha[:12]}', sha


def items_of(split_path, subsets) -> list[dict]:
    split = json.load(open(split_path))
    out, seen = [], set()
    for s in subsets:
        for it in split[s]:
            if it['audio_sha256'] not in seen:
                seen.add(it['audio_sha256'])
                out.append(it)
    return out


def _save(path: Path, beat, down) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, beat=np.asarray(beat, dtype=np.float32), down=np.asarray(down, dtype=np.float32))
    os.replace(tmp, path)


class Runner:
    def __init__(self, name: str, device: str):
        from beat_this.inference import Audio2Frames, load_audio
        self.frames = Audio2Frames(checkpoint_path=str(ckpt_path(name)), device=device, float16=False)
        self.load = load_audio

    def logits(self, path) -> tuple[np.ndarray, np.ndarray, dict]:
        from ensomi_model.audio_rows.audio import _lead
        from ensomi_model.audio_rows.decoder import lead_seconds
        from ensomi_model.features.audio import load_audio_file
        info = {}
        try:
            wave, sr = self.load(path)
            lead = lead_seconds(path, 'mpg123')
            wave = _lead(np.asarray(wave), lead, sr)
            info.update(loader='libsndfile', lead_ms=1000 * lead)
        except Exception:
            sr = 24000
            lead = lead_seconds(path, 'ffmpeg')
            wave = _lead(load_audio_file(path, sr), lead, sr)
            info.update(loader='pydub', lead_ms=1000 * lead)
        b, d = self.frames(wave, sr)
        return b.detach().cpu().float().numpy(), d.detach().cpu().float().numpy(), info


def cmd_map(a) -> dict:
    bulk = Path(a.bulk)
    dest, sha = out_dir(bulk, a.ckpt)
    items = items_of(a.split, a.subset.split(','))
    done, skipped = 0, []
    if a.stored == 'fitdev':
        src = TIMING / 'opt/ft/acc/models' / a.ckpt
        audio_of = {json.loads(l)['key']: json.loads(l)['audio_rel'] for l in open(src / 'fits.jsonl')}
        for it in items:
            key = it.get('bt_key')
            f = src / 'logits' / f'{key}.npz'
            if key is None or audio_of.get(key) != it['audio_rel'] or not f.exists():
                skipped.append(dict(item=it['item'], key=key, reason='no matching stored file'))
                continue
            target = dest / f"{it['audio_sha256']}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(f, target)
            done += 1
    else:
        src = TIMING / 'repr/reset'
        audio_of = {json.loads(l)['i']: json.loads(l)['audio_rel'] for l in open(src / 'fits.jsonl')}
        for it in items:
            i = it.get('bt_index')
            f = src / 'logits' / f'{i:04d}.npz' if i is not None else None
            if i is None or audio_of.get(i) != it['audio_rel'] or not f.exists():
                skipped.append(dict(item=it['item'], index=i, reason='no matching stored file'))
                continue
            target = dest / f"{it['audio_sha256']}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(f, target)
            done += 1
    return dict(mode='map', stored=a.stored, ckpt=a.ckpt, ckpt_sha256=sha, dest=str(dest), mapped=done,
                skipped=skipped, n=len(items))


def cmd_compute(a) -> dict:
    bulk = Path(a.bulk)
    dest, sha = out_dir(bulk, a.ckpt)
    items = items_of(a.split, a.subset.split(','))
    runner = Runner(a.ckpt, a.device)
    t0 = time.time()
    verify = []
    if a.verify:
        for it in [x for x in items if (dest / f"{x['audio_sha256']}.npz").exists()][:a.verify]:
            b, d, info = runner.logits(os.path.join(a.root, it['audio_rel']))
            z = np.load(dest / f"{it['audio_sha256']}.npz")
            n = min(len(b), len(z['beat']))
            verify.append(dict(item=it['item'], len_new=len(b), len_stored=len(z['beat']),
                               max_abs_beat=float(np.max(np.abs(b[:n] - z['beat'][:n]))),
                               max_abs_down=float(np.max(np.abs(d[:n] - z['down'][:n]))), **info))
    done, errors, infos = 0, [], []
    for it in items:
        target = dest / f"{it['audio_sha256']}.npz"
        if target.exists():
            continue
        if time.time() - t0 > 60 * a.max_minutes:
            break
        try:
            b, d, info = runner.logits(os.path.join(a.root, it['audio_rel']))
            _save(target, b, d)
            infos.append(dict(item=it['item'], frames=len(b), **info))
            done += 1
        except Exception as exc:
            errors.append(dict(item=it['item'], error=f'{type(exc).__name__}: {exc}'))
        if a.run and done % 20 == 0:
            with open(Path(a.run) / 'progress.jsonl', 'a') as f:
                f.write(json.dumps(dict(step=f'bt_{a.ckpt}', done=done, wall_s=round(time.time() - t0, 1),
                                        t=round(time.time(), 1))) + '\n')
    remaining = sum(not (dest / f"{it['audio_sha256']}.npz").exists() for it in items)
    return dict(mode='compute', ckpt=a.ckpt, ckpt_sha256=sha, dest=str(dest), computed=done, remaining=remaining,
                errors=errors, verify=verify, wall_s=round(time.time() - t0, 1), n=len(items),
                loaders={k: sum(i['loader'] == k for i in infos) for k in ('libsndfile', 'pydub')},
                leads=sum(i['lead_ms'] > 0 for i in infos))


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('map', 'compute'))
    p.add_argument('--split', required=True)
    p.add_argument('--subset', required=True, help='comma-separated split.json lists')
    p.add_argument('--ckpt', default='final0')
    p.add_argument('--bulk', default='artifacts/bulk/n1-cache')
    p.add_argument('--root', default='.')
    p.add_argument('--stored', choices=('fitdev', 'reset'), default='fitdev')
    p.add_argument('--device', default='mps')
    p.add_argument('--verify', type=int, default=0)
    p.add_argument('--max-minutes', type=float, default=50.0)
    p.add_argument('--run', default='')
    p.add_argument('--out', required=True, help='result JSON')
    a = p.parse_args(argv)
    res = cmd_map(a) if a.mode == 'map' else cmd_compute(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('skipped', 'errors', 'verify')}))


if __name__ == '__main__':
    main()
