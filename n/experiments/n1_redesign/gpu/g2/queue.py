"""Serial MPS stages with at most two scoring workers and final-only training saves."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
from .common import OUT,DEADLINE,dump,disk_guard,check_prereg
from .cleanup import control


def launch(module,args,label):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    with (OUT/f'{label}.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u','-m',f'n.experiments.n1_redesign.gpu.g2.{module}',*map(str,args)],
                           stdout=log,stderr=subprocess.STDOUT,env=env)
    event('started',label=label,pid=p.pid)
    return p


def event(kind,**fields):
    row=dict(event=kind,utc=datetime.now(timezone.utc).isoformat(),**fields)
    with (OUT/'queue.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def wait(p,label):
    code=p.wait();event('finished',label=label,returncode=code)
    if code:raise RuntimeError(f'{label} exited {code}; inspect its log, no scientific rerun')


def main():
    check_prereg();scorer=None;previous=None;status={}
    for seed in (0,1):
        if not (OUT/'control'/f's{seed}/data/metadata.json').exists():
            wait(launch('readout',['samples',seed],f'F0-s{seed}-features'),f'F0-s{seed}-features')
        if scorer is not None:
            wait(scorer,f'score-s{previous}');control(previous);scorer=None
        # The same process owns every accelerator stage; caller must wait for any
        # precomputed standalone feature extraction process to exit before launch.
        remaining=(datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()
        needed=max(3600.,max((r['wall_s']+1200. for r in status.values()),default=0.))
        resources=disk_guard();disk_guard(reserve=max(0,6e9-resources['new_bytes']))
        event('run_budget',seed=seed,remaining_s=remaining,required_s=needed,**resources)
        if remaining<needed:
            event('skipped_deadline',seed=seed);break
        fit=launch('readout',['fit',seed],f'F0-s{seed}-fit')
        start=time.monotonic()
        wait(launch('finetune',[seed],f'F2-s{seed}-train'),f'F2-s{seed}-train')
        status[str(seed)]=json.loads((OUT/'runs'/f'F2-s{seed}/result.json').read_text())
        wait(fit,f'F0-s{seed}-fit')
        for arm in ('F2','F0'):
            wait(launch('evaluate',['infer',seed,'--arm',arm],f'{arm}-s{seed}-inference'),f'{arm}-s{seed}-inference')
        event('accelerator_stages_complete',seed=seed,wall_s=time.monotonic()-start)
        scorer=launch('evaluate',['score',seed],f'score-s{seed}');previous=seed
    if scorer is not None:wait(scorer,f'score-s{previous}');control(previous)
    dump(OUT/'queue-DONE.json',dict(status='complete' if len(status)==2 else 'partial',runs=list(status)))


if __name__=='__main__':main()
