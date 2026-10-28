"""Run individual probe stages using the main checkout's installed environment."""
import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

from .common import OUT, digest, dump, log, utc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['labels', 'freeze', 'tokens', 'P1', 'P2', 'P3', 'P4a', 'P4b', 'P5a', 'P5b', 'P5c', 'P6', 'P7', 'P8', 'packet', 'report'])
    p.add_argument('--seed', type=int, choices=[0, 1], default=0)
    p.add_argument('--shifted', action='store_true')
    p.add_argument('--bootstrap', action='store_true')
    p.add_argument('--template', type=str)
    a = p.parse_args()
    source = {str(f.name): digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    log('stage_start', stage=a.stage, seed=a.seed, source_commit=commit, source_sha256=source,
        runtime_sha256=digest(OUT / 'runtime.json') if (OUT / 'runtime.json').exists() else None)
    if a.stage == 'labels':
        from .labels import prepare
        prepare(a.seed)
    elif a.stage == 'freeze':
        assert not (OUT / 'prereg-lock.json').exists(), 'Preregistration already frozen'
        counts = json.loads((OUT / 'counts-s0.json').read_text())
        body = Path(a.template).read_text() + '\n\n## P0 label counts\n\n```json\n' + json.dumps(counts, indent=2) + '\n```\n'
        path = OUT / 'prereg.md'
        path.write_text(body)
        record = dict(sha256=digest(path), utc=utc())
        dump(OUT / 'prereg-lock.json', record)
        log('prereg_frozen', **record)
    elif a.stage == 'tokens':
        from .tokens import prepare_tokens
        prepare_tokens(a.seed)
    elif a.stage == 'P1':
        from .attacks import fit_models, evaluate
        fit_models(a.seed, a.shifted, a.bootstrap)
        evaluate(a.seed, a.shifted, a.bootstrap)
    elif a.stage == 'P3':
        from .cells import run
        run(a.seed, a.bootstrap)
    elif a.stage == 'P7':
        from .episodes import run
        run(a.seed)
    elif a.stage == 'P2':
        from .salience import run
        run(a.seed)
    elif a.stage == 'packet':
        from .packet import build
        build()
    elif a.stage == 'P8':
        from .session import replay
        replay(a.seed)
    elif a.stage == 'P6':
        from .session import simulator_bands
        simulator_bands()
    elif a.stage in ('P4a', 'P4b'):
        from .layers import run
        run(a.seed, keysounds=a.stage == 'P4b')
    elif a.stage == 'report':
        from .report import build
        build()
    else:
        from .repeats import retrieval, recurrence, downbeat
        {'P5a': retrieval, 'P5b': recurrence, 'P5c': downbeat}[a.stage](a.seed)
    log('stage_done', stage=a.stage, seed=a.seed,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
