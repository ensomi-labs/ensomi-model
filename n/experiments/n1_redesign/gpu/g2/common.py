"""G2 artifact ownership, audited inputs and resource guards."""
import json
import shutil
from pathlib import Path
from ..cohorts import ROOT, OUT as G1, BULK, FEAT, RUNS, dump
from ..audit_inputs import digest, probe_bucket

OUT=G1/'g2'
X5D=G1.parent/'decoder/x5d'
DEADLINE='2026-10-29T00:30:00+00:00'


def disk_guard(reserve=0):
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    free=shutil.disk_usage(OUT).free
    if used+reserve>6e9 or free-reserve<100e9:
        raise RuntimeError(f'G2 disk guard: {used} new bytes, {free} free, reserve {reserve}')
    return dict(new_bytes=used,free_bytes=free)


def check_prereg():
    lock=json.loads((OUT/'prereg-lock.json').read_text())
    if digest(OUT/'prereg.md')!=lock['sha256']:raise RuntimeError('G2 preregistration changed')
    if digest(OUT/'target-domain-decision.json')!=lock['domain_decision_sha256']:
        raise RuntimeError('G2 target-domain decision changed')
    return lock
