"""Supervise P7m with process-tree memory, disk and UTC deadline limits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def tree_rss(pid):
    rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
        ['ps', '-axo', 'pid=,ppid=,rss='], text=True).splitlines()]
    own = {pid}
    while True:
        expanded = own | {p for p, parent, _ in rows if parent in own}
        if expanded == own:
            break
        own = expanded
    return sum(rss * 1024 for p, _, rss in rows if p in own)


def disk_size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def run_child(command, out, deadline):
    name = '-'.join(command).replace('--', '')
    start = time.time()
    peak = 0
    reason = None
    with (out / (name + '.log')).open('a') as log:
        process = subprocess.Popen([sys.executable, '-m', 'n1_redesign.probes.p7m', *command],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            rss = tree_rss(process.pid)
            peak = max(peak, rss)
            now = datetime.now(timezone.utc)
            if rss > 5_500_000_000:
                reason = 'process-tree RSS exceeded conservative 5.5 GB stop'
            elif disk_size(out) > 1_850_000_000:
                reason = 'new output exceeded conservative 1.85 GB stop'
            elif now.timestamp() >= deadline:
                reason = 'UTC measurement/report deadline'
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(1)
        code = process.wait()
    row = dict(utc=datetime.now(timezone.utc).isoformat(), command=command, exit=code,
               seconds=time.time() - start, peak_process_tree_rss_bytes=peak,
               disk_bytes=disk_size(out), stopped=reason)
    with (out / 'resources.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', required=True, help='ISO UTC deadline including timezone')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error('deadline must include timezone')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for command in (['prepare'], ['seed', '--seed', '0'], ['seed', '--seed', '1']):
        if datetime.now(timezone.utc).timestamp() >= deadline.timestamp() - 180:
            failed = True
            break
        if run_child(command, args.out, deadline.timestamp() - 180):
            failed = True
            break
    code = run_child(['report'], args.out, deadline.timestamp())
    raise SystemExit(code or int(failed))


if __name__ == '__main__':
    main()
