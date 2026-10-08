"""Run receipts: entry point, configuration, code identity, input hashes, seed and device."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def code_identity():
    """Hash of the imported package sources (the frozen copy when run by the launcher) and git state."""
    package = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        h.update(str(path.relative_to(package)).encode())
        h.update(path.read_bytes())
    r2 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((package / 'r2').glob('*.py'))}
    out = dict(package=str(package), package_sha256=h.hexdigest(), r2_files=r2)
    try:
        out['git_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                         timeout=10).stdout.strip()
        out['git_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src'], capture_output=True,
                                               text=True, timeout=10).stdout.strip())
        out['git_note'] = ('the mac .git may lag the control plane (see .sync/cp/jobs/<id>/provenance.json); '
                           'package_sha256 identifies the code that ran')
    except Exception as exc:  # noqa: BLE001
        out['git_error'] = str(exc)
    return out


def write_receipt(path: Path, **fields):
    data = dict(entry_point=' '.join(sys.argv), code=code_identity(), host=platform.node(), pid=os.getpid(),
                time=time.time(), **fields)
    Path(path).write_text(json.dumps(data, indent=1, default=str))
    return data


if __name__ == '__main__':  # post-hoc receipt for a build: python -m ensomi_model.r2.receipts <path> key=value ...
    write_receipt(Path(sys.argv[1]), **dict(a.split('=', 1) for a in sys.argv[2:]))
