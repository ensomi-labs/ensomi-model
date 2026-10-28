"""Run individual probe stages using the main checkout's installed environment."""
import argparse
import json
import resource
import time

from .common import OUT, digest, dump, log, utc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['labels', 'freeze', 'tokens', 'P1', 'P3', 'P5a', 'P5b', 'P5c', 'P7'])
    p.add_argument('--seed', type=int, choices=[0, 1], default=0)
    p.add_argument('--shifted', action='store_true')
    p.add_argument('--bootstrap', action='store_true')
    p.add_argument('--template', type=str)
    a = p.parse_args()
    log('stage_start', stage=a.stage, seed=a.seed)
    if a.stage == 'labels':
        from .labels import prepare
        prepare(a.seed)
    elif a.stage == 'freeze':
        from pathlib import Path
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
    else:
        from .repeats import retrieval, recurrence, downbeat
        {'P5a': retrieval, 'P5b': recurrence, 'P5c': downbeat}[a.stage](a.seed)
    log('stage_done', stage=a.stage, seed=a.seed,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    main()
