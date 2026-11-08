"""Sequential snapshot commands with disk and owned-output limits.

The JSON manifest supplies jobs (name, argv, cwd, env, complete_file), an output
directory, and owned_roots. Commands are argv lists, never shell expressions.
Each child gets its own process group; SIGTERM or STOP requests a clean child
stop. A child which does not exit within 90 seconds is killed. Completed jobs
are skipped on restart. A failed/stopped job prevents later jobs from starting.
Jobs may opt into bounded RSS-guard restarts with state_file and rss_restarts;
only a progressing clean recovery is retried. Disk, explicit stop and numerical
failures never trigger that retry. --detach starts nohup/nice/caffeinate in a new
session; --detach --resume explicitly clears STOP and a verified stale lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def owned_bytes(roots):
    total = 0
    for root in roots:
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(folder) / d).is_symlink()]
            for name in names:
                path = Path(folder) / name
                if not path.is_symlink():
                    try:
                        total += path.stat().st_size
                    except FileNotFoundError:
                        pass
    return total


def guard(root, owned_roots):
    free, used = shutil.disk_usage(root).free, owned_bytes(owned_roots)
    # Leave room to atomically save a final recovery checkpoint before 100 GB.
    if free < 100_100_000_000:
        return f'free disk {free} bytes is below the 100.1 GB stop threshold'
    if used > 5_900_000_000:
        return f'owned outputs {used} bytes exceed the 5.9 GB stop threshold'
    return None


def detach(manifest, spec, resume=False):
    """Launch nohup/nice/caffeinate in a new session; never replace a live queue."""
    root = Path(spec['out'])
    root.mkdir(parents=True, exist_ok=True)
    reason = guard(root, spec['owned_roots'])
    if reason:
        raise SystemExit(reason)
    lock = root / 'queue.lock'
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if not resume:
                raise SystemExit('Stale queue.lock exists; use --detach --resume after checking the run')
            lock.unlink()
        else:
            raise SystemExit(f'Queue lock PID {pid} is still alive')
    stop = root / 'STOP'
    if stop.exists():
        if not resume:
            raise SystemExit('STOP exists; use --detach --resume to resume explicitly')
        stop.unlink()
    commands = [shutil.which(name) for name in ('nohup', 'nice', 'caffeinate')]
    if not all(commands):
        raise SystemExit('Detached launch requires nohup, nice and caffeinate')
    argv = [commands[0], commands[1], '-n', '10', commands[2], '-i', sys.executable,
            str(Path(__file__).resolve()), '--manifest', str(Path(manifest).resolve())]
    with (root / 'queue.log').open('a') as log:
        child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(wrapper_pid=child.pid, wrapper_pgid=child.pid, started=time.time(), argv=argv)
    (root / 'detached.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Explicitly clear STOP/stale lock when detaching')
    args = parser.parse_args()
    spec = json.loads(Path(args.manifest).read_text())
    if args.detach:
        return detach(args.manifest, spec, args.resume)
    if args.resume:
        parser.error('--resume requires --detach; child checkpoint resume belongs in the job argv')
    root = Path(spec['out'])
    root.mkdir(parents=True, exist_ok=True)
    reason = guard(root, spec['owned_roots'])
    if reason:
        raise SystemExit(reason)
    if (root / 'STOP').exists():
        raise SystemExit('STOP exists; remove it explicitly before resuming')
    lock = root / 'queue.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit('queue.lock exists; verify its PID is gone before removing a stale lock')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    stop = []
    def request_stop(signum, frame):
        stop.append(signal.Signals(signum).name)
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    (root / 'queue-pid.json').write_text(json.dumps(dict(pid=os.getpid(), pgid=os.getpgrp(),
                                                      started=time.time()), indent=2))
    status, child = 0, None
    try:
        jobs = list(spec['jobs'])
        retries, progress = {}, {}
        while jobs:
            job = jobs.pop(0)
            if Path(job['complete_file']).exists():
                print(json.dumps(dict(job=job['name'], status='already_complete')), flush=True)
                continue
            reason = guard(root, spec['owned_roots'])
            if stop or (root / 'STOP').exists() or reason:
                print(json.dumps(dict(status='stopped', reason=reason or stop or 'STOP')), flush=True)
                return 75
            env = dict(os.environ, **job.get('env', {}))
            with (root / (job['name'] + '.log')).open('a') as log:
                child = subprocess.Popen(job['argv'], cwd=job['cwd'], env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                print(json.dumps(dict(job=job['name'], pid=child.pid, pgid=child.pid,
                                      started=time.time(), argv=job['argv'])), flush=True)
                stopping = None
                while child.poll() is None:
                    reason = guard(root, spec['owned_roots'])
                    if stop or (root / 'STOP').exists() or reason:
                        if stopping is None:
                            stopping = time.monotonic()
                            print(json.dumps(dict(job=job['name'], status='stopping',
                                                  reason=reason or stop or 'STOP')), flush=True)
                            os.killpg(child.pid, signal.SIGTERM)
                        elif time.monotonic() - stopping > 90:
                            os.killpg(child.pid, signal.SIGKILL)
                    time.sleep(2)
                status = child.returncode
                print(json.dumps(dict(job=job['name'], exited=time.time(), returncode=status)), flush=True)
                state_path = Path(job['state_file']) if job.get('state_file') else None
                if status == 75 and not stopping and state_path is not None and state_path.exists():
                    state = json.loads(state_path.read_text())
                    recovered = state.get('recovery_exposures', 0)
                    name = job['name']
                    if (str(state.get('reason', '')).startswith('RSS ')
                            and recovered > progress.get(name, 0)
                            and retries.get(name, 0) < job.get('rss_restarts', 0)):
                        progress[name] = recovered
                        retries[name] = retries.get(name, 0) + 1
                        print(json.dumps(dict(job=name, status='resume_after_rss_guard',
                                              exposures=recovered, attempt=retries[name])), flush=True)
                        jobs.insert(0, job)
                        continue
                if status or stopping:
                    return status or 75
                if not Path(job['complete_file']).exists():
                    print(f'{job["name"]} exited without its completion file', flush=True)
                    return 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        lock.unlink(missing_ok=True)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
