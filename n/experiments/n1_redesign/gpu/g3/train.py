"""Single-job event training: original crops and auxiliary losses, no pulse or memory path."""
import argparse
import datetime
import json
import os
import time
import numpy as np
import torch
from torch import nn
from ensomi_n.v0.model import NEv0
from ensomi_n.v0.data import load_songs
from ensomi_n.v0 import train as T
from ..finetune import TargetSampler, auxiliary_loss
from ..targets import NEW_KEYS
from .common import OUT, G1, BULK, DEADLINE, dump, disk_guard, check_prereg, log


class EventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NEv0(memory=False)
        self.new = nn.Linear(self.encoder.d, 7)

    def forward(self, x):
        return self.new(self.encoder(x, use_memory=False))


def learning_rate(step, elapsed):
    if step < 500:
        return 1e-3 * (step + 1) / 500
    return 1e-3 * (.05 + .95 * .5 * (1 + np.cos(np.pi * min(elapsed / 3000, 1))))


def run(seed):
    torch.set_num_threads(1)
    os.nice(10)
    lock = check_prereg()
    if datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=60) > DEADLINE:
        raise RuntimeError('Insufficient time for a 50-minute run and evaluation before deadline')
    before = disk_guard(200_000_000)
    name = f'E-s{seed}'
    dest = OUT / 'runs' / name
    if dest.exists():
        raise RuntimeError('Fresh run directory required; no resume')
    rows = json.loads((G1 / 'cohorts.json').read_text())['training']
    index = json.loads((BULK / 'song-index-v0.json').read_text())
    songs = load_songs(rows, index, BULK, BULK / 'labels-v0b')
    if len(songs) != 6196:
        raise RuntimeError('Training cohort changed')
    pw = json.loads((G1 / 'target-rates.json').read_text())['attack_pos_weight']
    dest.mkdir(parents=True)
    started = time.monotonic()
    torch.manual_seed(3000 + seed)
    np.random.seed(3000 + seed)
    model = EventModel().to('mps').train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    # The original sampler also prepares unused CPU memory inputs. Keeping it intact
    # preserves its weighted draws, augmentation and crop stream exactly.
    sampler = TargetSampler(songs, seed=2000 + seed, crops_per_song=2)
    pre = T.Prefetch(sampler, 8)
    config = dict(seed=seed, torch_seed=3000+seed, sampler_seed=2000+seed,
                  memory=False, pulse_losses=False, training_recordings=len(songs),
                  training_minutes=50, peak_lr=1e-3, warmup_steps=500, attack_pos_weight=pw,
                  parameters=sum(p.numel() for p in model.parameters()), disk_before=before,
                  prereg_sha256=lock['sha256'])
    dump(dest / 'config.json', config)
    step = 0
    elapsed = 0.
    status = 'complete'
    final_parts = {}
    losses = []
    parts = {}
    last_check = time.monotonic()
    win_start = last_check
    win_crops = 0
    ref_rate = None
    slow_since = None

    def write(rec):
        rec.update(step=step, training_s=elapsed, wall_s=time.monotonic()-started, utc_s=time.time())
        with (dest / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(dict(run=name, **rec)), flush=True)

    try:
        while elapsed < 3000:
            tick = time.monotonic()
            x, tg, _, _, _ = pre.get()
            x = torch.from_numpy(x).to('mps')
            tg = {k: torch.from_numpy(np.ascontiguousarray(tg[k])).to('mps') for k in (*NEW_KEYS, 'weight')}
            rate = learning_rate(step, elapsed)
            for group in opt.param_groups:
                group['lr'] = rate
            loss, scal = auxiliary_loss(model(x), tg, pw, tg['weight'])
            if not torch.isfinite(loss):
                status = 'killed:nonfinite_loss'
                write(dict(event='kill', reason=status))
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                status = 'killed:nonfinite_gradient'
                write(dict(event='kill', reason=status))
                break
            opt.step()
            torch.mps.synchronize()
            step += 1
            if step % T.EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()
            elapsed += time.monotonic() - tick
            final_parts = dict(total=float(loss.detach()), **scal)
            losses.append(final_parts['total'])
            for key, val in scal.items():
                parts.setdefault(key, []).append(val)
            win_crops += len(x)
            now = time.monotonic()
            if step % 200 == 0 or elapsed >= 3000:
                write(dict(event='loss', loss=float(np.mean(losses)),
                           parts={k:float(np.mean(v)) for k,v in parts.items()}, lr=rate,
                           gradient_norm=float(norm)))
                losses.clear()
                parts.clear()
            if now - last_check >= 60 or elapsed >= 3000:
                driver, allocated = T._mps_gb()
                cps = win_crops / (now - win_start)
                write(dict(event='resource', crops_per_s=cps, mps_driver_gb=driver,
                           mps_alloc_gb=allocated, **disk_guard()))
                if allocated > T.MEM_LIMIT_GB or driver > T.DRIVER_LIMIT_GB:
                    status = 'killed:resource_mps_memory'
                    break
                if ref_rate is None and step > 500:
                    ref_rate = cps
                if ref_rate and cps < .5 * ref_rate:
                    slow_since = slow_since or now
                    if now - slow_since >= 180:
                        status = 'killed:resource_throughput'
                        break
                else:
                    slow_since = None
                last_check = win_start = now
                win_crops = 0
    except RuntimeError as exc:
        if 'G3 disk guard' in str(exc) or 'out of memory' in str(exc).lower():
            status = 'killed:resource:' + str(exc)
            write(dict(event='kill', reason=status))
        else:
            raise
    finally:
        pre.close()
    torch.save(dict(model=model.cpu().state_dict(), seed=seed, memory=False, steps=step,
                    training_s=elapsed, prereg_sha256=lock['sha256']), dest / 'final.pt')
    result = dict(name=name, status=status, steps=step, training_s=elapsed,
                  wall_s=time.monotonic()-started, final_losses=final_parts,
                  kills=[] if status=='complete' else [status], config=config)
    dump(dest / 'result.json', result)
    log('run_complete', **result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    run(p.parse_args().seed)
