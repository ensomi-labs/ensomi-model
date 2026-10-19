"""E1 baselines on synthetic suite v0: raw BeatThis and R2's timing, per BeatThis checkpoint (R side, AGPL).

For each checkpoint (``final0``, ``final1``, ``final2``), ``audio_rows.audio.Listener`` runs unchanged on
every clip, with BeatThis's frame model wrapped only to keep its logits. From one BeatThis pass per clip
and checkpoint this writes:

- ``<out>/bt/<ckpt>/<id>.json``: raw BeatThis, the package's ``minimal`` postprocessing of those logits
  (what ``File2Beats`` returns): ``beats_s`` and ``downbeats_s``, on the decoded clock;
- ``<out>/r2/<ckpt>/<id>.json``: the fitter's raw segments, ``export`` (``audio_rows.grid.corrected_grid``
  with offsets rounded to whole ms, as ``system.make.write_export`` writes them: canonical beat length,
  meter 4) and ``fitted`` (the same with integer-BPM rounding off, offsets unrounded), on osu!'s clock;
- ``<logits>/<ckpt>/<id>.npz``: the beat and downbeat logits.

Resumable (clips with both files are skipped); one process; errors are recorded per clip.

    PYTHONPATH=<src>/src:<src>/n/src .venv/bin/python <src>/scripts/n1/r2_on_clips.py --audio DIR \
        --manifest <truth>/manifest.jsonl --out <truth>/preds --logits DIR --device mps --result FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


class _Keep:
    """BeatThis's ``Audio2Frames``, keeping the last output."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def __call__(self, wave, sr):
        self.last = self.inner(wave, sr)
        return self.last

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--logits', required=True)
    p.add_argument('--ckpts', default='final0,final1,final2')
    p.add_argument('--ckpt-dir', default=os.path.expanduser('~/.cache/torch/hub/checkpoints'))
    p.add_argument('--device', default='mps')
    p.add_argument('--result', required=True)
    a = p.parse_args(argv)

    import torch
    import beat_this
    from beat_this.model.postprocessor import Postprocessor
    import ensomi_model.audio_rows.grid as grid_mod
    from ensomi_model.audio_rows.audio import Listener

    rows = [json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    out, logits_dir = Path(a.out), Path(a.logits)
    post = Postprocessor(type='minimal', fps=50)
    t_all = time.time()
    result = dict(n_clips=len(rows), ckpts={}, errors=[], torch=torch.__version__,
                  beat_this=getattr(beat_this, '__version__', ''), device=a.device)
    for ck in [c for c in a.ckpts.split(',') if c]:
        path = Path(a.ckpt_dir) / f'beat_this-{ck}.ckpt'
        sha = _sha256(path)
        listener = Listener(path, device=a.device)
        listener.frames = _Keep(listener.frames)
        done = built = 0
        t_ck = time.time()
        for r in rows:
            cid = r['id']
            f_bt, f_r2 = out / 'bt' / ck / f'{cid}.json', out / 'r2' / ck / f'{cid}.json'
            if f_bt.exists() and f_r2.exists():
                done += 1
                continue
            try:
                t0 = time.time()
                song = listener.listen(str(Path(a.audio) / r['audio']))
                beat_logits, down_logits = listener.frames.last
                try:
                    beats, downs = post(beat_logits, down_logits)
                except Exception:      # a postprocessor that wants a batch axis
                    b2, d2 = post(beat_logits[None], down_logits[None])
                    beats, downs = b2[0], d2[0]
                bl = beat_logits.detach().cpu().numpy().astype(np.float32)
                dl = down_logits.detach().cpu().numpy().astype(np.float32)
                (logits_dir / ck).mkdir(parents=True, exist_ok=True)
                np.savez(logits_dir / ck / f'{cid}.npz', beat=bl, down=dl)
                _, seg = grid_mod.corrected_grid(song.segments, song.song_ms)
                export = np.array(seg, dtype=np.float64)
                export[:, 0] = np.round(export[:, 0])
                rounding = grid_mod.BPM_ROUNDING
                try:
                    grid_mod.BPM_ROUNDING = -1.0
                    _, fitted = grid_mod.corrected_grid(song.segments, song.song_ms)
                finally:
                    grid_mod.BPM_ROUNDING = rounding
                secs = round(time.time() - t0, 3)
                common = dict(id=cid, checkpoint=ck, checkpoint_sha256=sha, audio=r['audio'],
                              audio_sha256=r.get('audio_sha256', ''), seconds=secs)
                _write(f_bt, dict(common, beats_s=np.asarray(beats, dtype=np.float64).tolist(),
                                  downbeats_s=np.asarray(downs, dtype=np.float64).tolist(), postprocessor='minimal'))
                _write(f_r2, dict(common, song_ms=float(song.song_ms),
                                  raw_segments=np.asarray(song.segments, dtype=np.float64).tolist(),
                                  export=export.tolist(), fitted=np.asarray(fitted, dtype=np.float64).tolist(),
                                  listener_seconds={k: round(v, 3) for k, v in song.seconds.items()}))
                built += 1
                print(json.dumps(dict(ckpt=ck, id=cid, s=secs, beats=len(beats), lines=len(export))), flush=True)
            except Exception as exc:
                result['errors'].append(dict(ckpt=ck, id=cid, error=f'{type(exc).__name__}: {exc}',
                                             trace=traceback.format_exc()[-800:]))
                print(json.dumps(dict(ckpt=ck, id=cid, error=str(exc))), flush=True)
        result['ckpts'][ck] = dict(sha256=sha, built=built, present=done, wall_s=round(time.time() - t_ck, 1))
        del listener
    result['wall_s'] = round(time.time() - t_all, 1)
    Path(a.result).parent.mkdir(parents=True, exist_ok=True)
    Path(a.result).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'errors'}), 'errors', len(result['errors']))
    return 0 if not result['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
