"""Launch an unattended R2 CE run from a frozen copy of the code, under a restarting supervisor.

Fresh run (copies ``src/ensomi_model`` to ``artifacts/r2-runs/<run-id>/code/``)::

    PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.launch --run-id <id> --config <cfg.json> [--set key=value ...]

Resume an existing run (reuses its frozen code)::

    PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.launch --run-id <id> --resume

The supervisor runs the trainer with PYTHONPATH pointing at the frozen copy,
appends every exit to ``events.jsonl``, and on a non-zero exit other than the
NaN-limit stop (3) resumes from the latest checkpoint, at most five restarts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MAX_RESTARTS = 5
EXIT_NAN = 3


def tree_sha(root: Path):
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def event(run: Path, **record):
    record.setdefault('time', time.time())
    with (run / 'events.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def prepare(args):
    repo = Path.cwd()
    run = repo / 'artifacts' / 'r2-runs' / args.run_id
    code = run / 'code'
    if args.resume:
        if not (run / 'config.json').exists() or not (code / 'ensomi_model').exists():
            raise SystemExit(f'{run} has no config or frozen code to resume')
        return run, code
    if run.exists() and any((run / 'checkpoints').glob('ckpt-*.pt')):
        raise SystemExit(f'{run} already has checkpoints; use --resume')
    run.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    for item in args.set or []:
        key, value = item.split('=', 1)
        config[key] = parse_value(value)
    config['run_dir'] = str(Path('artifacts') / 'r2-runs' / args.run_id)
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(repo / 'src' / 'ensomi_model', code / 'ensomi_model',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (run / 'config.json').write_text(json.dumps(config, indent=1))
    git = {}
    for name, cmd in (('head', ['git', 'rev-parse', 'HEAD']), ('status', ['git', 'status', '--porcelain', '--', 'src'])):
        try:
            git[name] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            git[name] = f'error: {exc}'
    (run / 'launch.json').write_text(json.dumps(dict(
        command=' '.join(sys.argv), frozen_code=str(code), frozen_code_sha256=tree_sha(code / 'ensomi_model'),
        git_head=git['head'], git_dirty=bool(git['status']), git_status=git['status'].splitlines()[:50],
        git_note='the mac .git may lag the control plane; frozen_code_sha256 identifies the code',
        config=config, ens_job=os.environ.get('ENS_JOB_ID'), time=time.time()), indent=1))
    return run, code


def supervise(run: Path, code: Path, resume: bool):
    env = dict(os.environ, PYTHONPATH=str(code), R2_FROZEN_CODE=str(code))
    restarts = 0
    event(run, event='supervisor_start', resume=resume, pid=os.getpid())
    while True:
        cmd = [sys.executable, '-m', 'ensomi_model.r2.train_ce', '--config', str(run / 'config.json')]
        if resume:
            cmd.append('--resume')
        with (run / 'trainer.log').open('a') as log:
            log.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M:%S")} {" ".join(cmd)}\n')
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        event(run, event='trainer_exit', returncode=rc, restarts=restarts)
        if rc == 0:
            event(run, event='supervisor_done')
            return 0
        if rc == EXIT_NAN:
            event(run, event='supervisor_stop', reason='nan_limit')
            return rc
        if restarts >= MAX_RESTARTS:
            event(run, event='supervisor_stop', reason='restart_limit')
            return rc
        restarts += 1
        resume = True
        event(run, event='restart', restarts=restarts)
        time.sleep(30)


def main(argv=None):
    p = argparse.ArgumentParser(description='R2 launcher')
    p.add_argument('--run-id', required=True)
    p.add_argument('--config', default=None)
    p.add_argument('--set', action='append', help='config override key=value (JSON values)')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--supervise-only', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.supervise_only:
        run = Path.cwd() / 'artifacts' / 'r2-runs' / a.run_id
        return supervise(run, run / 'code', a.resume)
    run, code = prepare(a)
    # Re-exec the supervisor from the frozen copy so later edits to the synced tree cannot reach it.
    env = dict(os.environ, PYTHONPATH=str(code), R2_FROZEN_CODE=str(code))
    args = [sys.executable, '-m', 'ensomi_model.r2.launch', '--run-id', a.run_id, '--supervise-only']
    if a.resume:
        args.append('--resume')
    os.execve(sys.executable, args, env)


if __name__ == '__main__':
    sys.exit(main())
