"""Verify immutable inputs and complete finite outputs before deleting inference arrays."""
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import torch
from .common import OUT, G1, check_prereg, dump, digest, disk_guard
from .statistics import METRICS
from .train import EventModel


def main():
    torch.set_num_threads(1)
    lock = check_prereg()
    inputs = json.loads((OUT/'inputs.json').read_text())['sha256']
    for path, expected in inputs.items():
        if digest(path)!=expected:
            raise ValueError(f'Input changed: {path}')
    queue = json.loads((OUT/'queue-DONE.json').read_text())
    if queue['status']!='complete':
        raise ValueError('Incomplete experiment queue')
    songs = json.loads((G1/'inference-songs.json').read_text())
    shas = {r['audio_sha256'] for r in json.loads((G1/'cohorts.json').read_text())['panels']}
    expected_frames = sum(songs[s]['frames'] for s in shas)
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=dict(platform=platform.platform(),python=sys.version,executable=sys.executable,
            torch=torch.__version__,mps_available=torch.backends.mps.is_available(),
            chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            ram_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))),
        prereg_sha256=lock['sha256'],unchanged_input_files=len(inputs),
        unchanged_experiment_sources=len(lock['source_sha256']),runs={},
        selected_local_checks=json.loads((OUT/'validation-before-training.json').read_text()))
    for seed in (0,1):
        name = f'E-s{seed}'
        root = OUT/'runs'/name
        state = torch.load(root/'final.pt',map_location='cpu',weights_only=True)
        if state['memory'] or state['prereg_sha256']!=lock['sha256']:
            raise ValueError('Checkpoint configuration changed')
        model = EventModel()
        model.load_state_dict(state['model'],strict=True)
        if any(not torch.isfinite(t).all() for t in state['model'].values()):
            raise ValueError('Non-finite final checkpoint')
        checkpoints = list(root.glob('*.pt'))
        if checkpoints!=[root/'final.pt']:
            raise ValueError('Unexpected intermediate checkpoint')
        folder = OUT/'eval'/name
        paths = list(folder.glob('*.npz'))
        if {p.stem for p in paths}!=shas:
            raise ValueError('Panel inference cohort changed')
        frames = 0
        for path in paths:
            with np.load(path,allow_pickle=False) as z:
                y = z['new_logits']
                if y.shape!=(songs[path.stem]['frames'],7) or not np.isfinite(y).all() or z['fps']!=100 or z['t0_ms']!=0:
                    raise ValueError(f'Invalid inference array: {path}')
                frames += len(y)
        if frames!=expected_frames:
            raise ValueError('Panel frame count changed')
        keys = None
        for arm in ('E','Mel','F1','frozen'):
            base = G1 if arm in ('F1','frozen') else OUT
            rows = json.loads((base/'heads'/f'{arm}-s{seed}.json').read_text())['rows']
            current = {(r['key'],r['audio_sha256']) for r in rows}
            if keys is not None and current!=keys:
                raise ValueError('Head score cohorts differ')
            keys = current
            if len(rows)!=472 or len(keys)!=472:
                raise ValueError('Reference count changed')
            for row in rows:
                for metric in METRICS:
                    value = row.get(metric)
                    if value is not None and (not np.isfinite(value) or not (-1 if metric=='salience_spearman' else 0)<=value<=1):
                        raise ValueError(f'Invalid score: {arm}, {metric}')
        progress = [json.loads(v) for v in (root/'progress.jsonl').read_text().splitlines()]
        resources = [r for r in progress if r['event']=='resource']
        losses = [r for r in progress if r['event']=='loss']
        summary = json.loads((root/'result.json').read_text())
        if summary['steps']!=state['steps'] or summary['training_s']!=state['training_s']:
            raise ValueError('Checkpoint/result mismatch')
        if any(not np.isfinite(r['loss']) for r in losses):
            raise ValueError('Non-finite logged loss')
        if any(r['new_bytes']>3e9 or r['free_bytes']<100e9 for r in resources):
            raise ValueError('Disk guard exceeded')
        result['runs'][name] = dict(status=summary['status'],steps=state['steps'],training_s=state['training_s'],
            checkpoint_sha256=digest(root/'final.pt'),checkpoint_files=len(checkpoints),
            parameters=sum(p.numel() for p in model.parameters()),inputs=len(paths),frames=frames,
            references=len(keys),loss_logs=len(losses),driver_peak_gib=max(r['mps_driver_gb'] for r in resources),
            allocation_peak_gib=max(r['mps_alloc_gb'] for r in resources),
            minimum_free_bytes=min(r['free_bytes'] for r in resources),
            final_weighted_loss_error=abs(summary['final_losses']['total']-sum(summary['final_losses'][k]*w for k,w in [('attack',.5),('snap',.5),('salience',.25)])))
    result['disk_before_cleanup'] = disk_guard()
    dump(OUT/'validation-final.json',result)
    print(json.dumps(result['runs']),flush=True)


if __name__=='__main__':
    main()
