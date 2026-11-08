"""Sequential snapshot commands with disk and owned-output limits.

The JSON manifest supplies jobs (name, argv, cwd, env, complete_file), an output
directory, and owned_roots. Commands are argv lists, never shell expressions.
Each child gets its own process group; SIGTERM or STOP requests a clean child
stop. A child which does not exit within 90 seconds is killed. Completed jobs
are skipped on restart. A failed/stopped job prevents later jobs from starting.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.manifest).read_text())
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
        for job in spec['jobs']:
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
