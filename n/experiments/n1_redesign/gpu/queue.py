"""One MPS process at a time; CPU scoring can overlap the next authorised single-job run."""
from __future__ import annotations
import datetime
import json
import subprocess
import sys
import time
from .cohorts import OUT,FEAT,dump
from .audit_inputs import digest

PREFIX='n.experiments.n1_redesign.gpu.'


def execute(module,args,log,background=False):
    f=(OUT/log).open('a')
    p=subprocess.Popen([sys.executable,'-u','-m',PREFIX+module,*args],stdout=f,stderr=subprocess.STDOUT)
    f.close()
    if background:return p
    rc=p.wait()
    if rc:raise RuntimeError(f'{module} {args} exited {rc}; see {log}')


def wait_for(path):
    while not path.exists():
        status=json.loads((OUT/'g0-status.json').read_text()) if (OUT/'g0-status.json').exists() else {}
        if status.get('status')=='failed':raise RuntimeError(f'G0 failed: {status}')
        time.sleep(10)


def run():
    deadline=datetime.datetime.fromisoformat(json.loads((OUT/'resume-budget.json').read_text())['stop_starting_utc'].replace('Z','+00:00')).timestamp()
    wait_for(FEAT/'v01-s1-009001/DONE')
    execute('finetune',['--baseline'],'baseline-probes.log')
    for p in ('baseline-DONE.json','target-rates.json','fits/s0.json','fits/s1.json'):wait_for(OUT/p)
    for seed in (0,1):
        if not (OUT/'heads'/f'frozen-s{seed}.json').exists():
            execute('evaluate',['frozen',str(seed)],f'frozen-s{seed}.log')
    execute('preregister',[],'prereg.log')
    jobs=[];durations=[];deviations=[];score_process=None
    for name,seed,treatment in [('F0-s0',0,False),('F1-s0',0,True),('F0-s1',1,False),('F1-s1',1,True)]:
        if time.time()>=deadline:
            deviations.append(dict(run=name,reason='stop-starting deadline reached'));continue
        estimate=max(durations,default=55*60)
        if name=='F0-s1' and time.time()+2*estimate>deadline:
            deviations.append(dict(run=name,reason='remaining pair cannot finish within budget; preserve F1-s1'))
            dump(OUT/'queue-deviations.json',deviations);continue
        start=time.monotonic()
        execute('finetune',['--seed',str(seed)]+(['--treatment'] if treatment else []),f'{name}.log')
        execute('evaluate',['infer',name],f'{name}-infer.log')
        durations.append(time.monotonic()-start)
        # Never overlap two separate decoder pools; their combined limit is two workers.
        if score_process is not None:
            rc=score_process.wait()
            if rc:raise RuntimeError('Previous CPU evaluation failed')
        score_process=execute('evaluate',['score',name],f'{name}-score.log',True)
        jobs.append(name)
        dump(OUT/'queue-status.json',dict(completed_inference=jobs,wall_costs_s=durations,deviations=deviations))
    if score_process is not None and score_process.wait():raise RuntimeError('Final CPU evaluation failed')
    dump(OUT/'queue-deviations.json',deviations)
    dump(OUT/'queue-DONE.json',dict(runs=jobs,deviations=deviations))


if __name__=='__main__':
    try:run()
    except Exception as exc:
        dump(OUT/'queue-error.json',dict(error=f'{type(exc).__name__}: {exc}'))
        raise
