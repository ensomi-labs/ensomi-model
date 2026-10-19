"""Rung 0 step (d): training throughput and memory of NE-S and NE-M on random tensors.

For each encoder and batch size: forward, backward and an AdamW step on random
log-Mel crops (20 s for NE-S, 60 s for NE-M with a 4-minute song memory), 3
warm-up steps then up to ``--steps`` timed steps. Reports crops per second and
the peak MPS driver memory. Also times NE-S inference on a whole 4-minute
song, which sets the cost of the in-run probes.

    python -m ensomi_n.run.throughput --out artifacts/n1-night-20261018/runs/<id>/throughput.json
"""
from __future__ import annotations

import argparse
import json
import time


def _sync(device):
    import torch
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def _mem(device) -> float:
    import torch
    if device == 'mps':
        return torch.mps.driver_allocated_memory() / 2 ** 30
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return float('nan')


def measure(name: str, batch: int, device: str, steps: int, max_s: float, frames: int | None = None) -> dict:
    import torch
    from ..models import NEM, NES, Model, n_params
    torch.manual_seed(0)
    enc = NES() if name == 'NE-S' else NEM()
    model = Model(enc, 'A1').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    frames = frames or (2000 if name == 'NE-S' else 6000)
    mel = torch.randn(batch, frames, 128, device=device)
    song = torch.randn(batch, 1500, 128, device=device) if name == 'NE-M' else None
    peak = 0.0
    times = []
    try:
        for k in range(3 + steps):
            t0 = time.perf_counter()
            out = model(mel, song)
            loss = out['logits'].float().pow(2).mean() + out['level'].float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            _sync(device)
            dt = time.perf_counter() - t0
            peak = max(peak, _mem(device))
            if k >= 3:
                times.append(dt)
                if sum(times) > max_s:
                    break
        rate = batch * len(times) / sum(times)
        res = dict(encoder=name, batch=batch, crop_s=frames / 100, params=n_params(enc), steps=len(times),
                   crops_per_s=round(rate, 2), step_s=round(sum(times) / len(times), 4), peak_mps_gb=round(peak, 2))
    except RuntimeError as exc:
        res = dict(encoder=name, batch=batch, error=str(exc)[:300], peak_mps_gb=round(peak, 2))
    del model, opt, mel
    if device == 'mps':
        torch.mps.empty_cache()
    return res


def inference(device: str) -> dict:
    import torch
    from ..models import NES, Model
    model = Model(NES(), 'A1').to(device).eval()
    mel = torch.randn(1, 24000, 128, device=device)
    with torch.inference_mode():
        model(mel)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(3):
            model(mel)
        _sync(device)
    s = (time.perf_counter() - t0) / 3
    return dict(encoder='NE-S', song_minutes=4.0, seconds=round(s, 3), seconds_per_audio_minute=round(s / 4, 4))


def machine(device: str) -> dict:
    """What the job can see of the machine (the inventory could not read it)."""
    import os
    import platform
    import torch
    out = dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count())
    try:
        import psutil
        vm = psutil.virtual_memory()
        out.update(memory_gb=round(vm.total / 2 ** 30, 1), available_gb=round(vm.available / 2 ** 30, 1),
                   load_avg=[round(x, 2) for x in os.getloadavg()])
    except Exception as exc:  # psutil missing or restricted
        out.update(memory_error=str(exc)[:200])
    if device == 'mps':
        try:
            out['mps_recommended_max_gb'] = round(torch.mps.recommended_max_memory() / 2 ** 30, 2)
        except Exception as exc:
            out['mps_recommended_max_error'] = str(exc)[:200]
    out['mps_watermark_ratios'] = [os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO'), os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO')]
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--steps', type=int, default=15)
    p.add_argument('--max-s', type=float, default=45.0)
    p.add_argument('--batches', default='8,16,32')
    p.add_argument('--only', default='', help="'NE-S' or 'NE-M': measure that encoder only (no inference timing)")
    p.add_argument('--frames', type=int, default=0, help='crop frames (default 2000 for NE-S, 6000 for NE-M)')
    a = p.parse_args(argv)
    import torch
    res = dict(device=a.device, torch=torch.__version__, machine=machine(a.device), rows=[], inference=None)

    def write():
        with open(a.out + '.tmp', 'w') as handle:
            json.dump(res, handle, indent=1)
        import os
        os.replace(a.out + '.tmp', a.out)

    # NE-S first (rung 1 needs it), then the whole-song inference, then NE-M, the largest, last; the
    # file is rewritten after each measurement so a failure late keeps what was measured.
    batches = [int(x) for x in a.batches.split(',')]
    if a.only != 'NE-M':
        for b in batches:
            res['rows'].append(measure('NE-S', b, a.device, a.steps, a.max_s, a.frames or None))
            print(json.dumps(res['rows'][-1]), flush=True)
            write()
    if not a.only:
        res['inference'] = inference(a.device)
        print(json.dumps(res['inference']), flush=True)
        write()
    if a.only != 'NE-S':
        for b in batches:
            res['rows'].append(measure('NE-M', b, a.device, a.steps, a.max_s, a.frames or None))
            print(json.dumps(res['rows'][-1]), flush=True)
            write()


if __name__ == '__main__':
    main()
