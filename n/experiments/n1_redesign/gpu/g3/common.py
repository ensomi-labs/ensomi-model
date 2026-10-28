"""G3 artifact ownership, immutable inputs and resource limits."""
import datetime
import json
import shutil
from ..cohorts import ROOT, OUT as G1, BULK, dump
from ..audit_inputs import digest, probe_bucket

OUT = G1 / 'g3'
DEADLINE = datetime.datetime(2026, 10, 29, 1, 15, tzinfo=datetime.timezone.utc)


def disk_guard(reserve=0):
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    free = shutil.disk_usage(OUT).free
    if used + reserve > 3e9 or free - reserve < 100e9:
        raise RuntimeError(f'G3 disk guard: {used} new bytes, {free} free, reserve {reserve}')
    return dict(new_bytes=used, free_bytes=free)


def check_prereg():
    lock = json.loads((OUT / 'prereg-lock.json').read_text())
    if digest(OUT / 'prereg.md') != lock['sha256']:
        raise RuntimeError('G3 preregistration changed')
    for path, sha in lock['source_sha256'].items():
        if digest(path) != sha:
            raise RuntimeError(f'G3 frozen source changed: {path}')
    return lock


def log(event, **values):
    record = dict(event=event, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
    with (OUT / 'job.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
