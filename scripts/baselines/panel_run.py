#!/usr/bin/env python3
"""Execute an explicit private panel manifest with a shared two-thread budget.

Each job supplies argv, environment additions, output receipt, log path, and
compute-thread count. Children have separate process groups; a resource stop
terminates only this queue's children. Completed/failed jobs are logged rather
than silently retried. The manifest and all logs can contain private paths.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def run(plan, root):
    root = Path(root)
    deadline = datetime.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    jobs = list(plan['jobs'])
    active, completed = [], []
    failures = {}
    stop = None
    last_size, retained = 0., 0
    record = root / 'results/jobs.jsonl'
    prior = [json.loads(s) for s in record.read_text().splitlines()] if record.exists() else []
    done = {r['id'] for r in prior if r['status'] in ('completed', 'failed', 'skipped_budget')}
    completed.extend(r for r in prior if r['status'] == 'completed')
    mapper_started = sum(r.get('wall_seconds', 0) for r in prior if r.get('system') == 'mapper')

    def save(row):
        with record.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def request_stop(signum, frame):
        nonlocal stop
        stop = f'signal {signum}'

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while jobs or active:
            now = time.time()
            free = shutil.disk_usage(root).free
            if now - last_size >= 30:
                retained = int(subprocess.check_output(['du', '-sk', plan['retained_root']], text=True).split()[0]) * 1024
                last_size = now
            if free < 100_000_000_000:
                stop = 'free disk below 100 GB'
            if retained > 5_950_000_000:
                stop = 'retained files exceed 5.95 GB safety limit'
            if now >= deadline:
                stop = 'deadline; reserve report time'
            if (root / 'STOP').exists():
                stop = 'STOP file'
            if stop:
                save(dict(status='stopped', reason=stop, time=now, free_bytes=free, retained_bytes=retained))
                break
            for item in active[:]:
                proc, job, started, log = item
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                status = 'completed' if code == 0 and Path(job['receipt']).exists() else 'failed'
                result = dict(id=job['id'], system=job['system'], song=job['song'], arm=job['arm'],
                              seed=job['seed'], source_class=job['source_class'], status=status,
                              returncode=code, wall_seconds=now-started, audio_seconds=job['audio_seconds'],
                              free_bytes=free, retained_bytes=retained, log=job['log'])
                save(result)
                if status == 'completed':
                    completed.append(result)
                    failures[job['system']] = 0
                else:
                    failures[job['system']] = failures.get(job['system'], 0) + 1
                    if failures[job['system']] >= 3:
                        stop = f"three consecutive {job['system']} failures; inspect before continuing"
                if job['system'] == 'mapper':
                    mapper_started += now-started
            while not stop and jobs and sum(x[1]['threads'] for x in active) + jobs[0]['threads'] <= 2:
                job = jobs.pop(0)
                if job['id'] in done:
                    continue
                if Path(job['receipt']).exists():
                    receipt = json.loads(Path(job['receipt']).read_text())
                    row = dict(id=job['id'], system=job['system'], song=job['song'], arm=job['arm'],
                               seed=job['seed'], source_class=job['source_class'], status='completed', reused=True,
                               wall_seconds=receipt['wall_seconds'], audio_seconds=job['audio_seconds'], log=job['log'])
                    save(row); completed.append(row)
                    continue
                # Normal songs were selected before generation. Only runtime, never chart metrics,
                # can remove them. Drop the remaining normal subset if it projects over seven hours.
                if job['system'] == 'mapper' and job['source_class'] == 'normal':
                    measured = [r for r in completed if r['system'] == 'mapper']
                    scale = sum(r['wall_seconds'] for r in measured) / max(1., sum(r['audio_seconds'] for r in measured))
                    remaining = [job] + [j for j in jobs if j['system'] == 'mapper']
                    projection = mapper_started + scale * sum(j['audio_seconds'] for j in remaining)
                    begun = any(r['system'] == 'mapper' and r['song'] == job['song'] for r in completed)
                    if not begun and (projection > 7 * 3600 or now + scale * job['audio_seconds'] * 2 >= deadline):
                        dropped = [job] + [j for j in jobs if j['system'] == 'mapper' and j['song'] == job['song']]
                        jobs = [j for j in jobs if j not in dropped]
                        for j in dropped:
                            save(dict(id=j['id'], system='mapper', source_class='normal', status='skipped_budget',
                                      projected_mapper_seconds=projection, song=j['song'], arm=j['arm'], seed=j['seed']))
                        continue
                env = dict(os.environ, **plan.get('env', {}), **job.get('env', {}))
                for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
                    env[name] = str(job['threads'])
                Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
                log = open(job['log'], 'w')
                proc = subprocess.Popen(job['argv'], env=env, cwd=plan['cwd'], stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                active.append((proc, job, time.time(), log))
                save(dict(id=job['id'], status='started', pid=proc.pid, threads=job['threads'], time=time.time()))
            time.sleep(1)
    finally:
        for proc, job, started, log in active:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc, job, started, log in active:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
            save(dict(id=job['id'], system=job['system'], song=job['song'], arm=job['arm'], seed=job['seed'],
                      status='interrupted', reason=stop, wall_seconds=time.time()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    (args.root / 'results').mkdir(parents=True, exist_ok=True)
    run(json.loads(args.plan.read_text()), args.root)
