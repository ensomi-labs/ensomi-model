"""Check planned N1 GPU cohorts using exclusion metadata before opening labels or features."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from ensomi_n.eval.census import load_exclusion
from ensomi_n.v0.curate import SEAL

from .sealed import build_exclusion


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def probe_bucket(sha):
    return int(hashlib.sha256(('n1r-probe:' + sha).encode()).hexdigest()[:8], 16) % 8


def check_rows(rows, exclusion, sealed):
    """Return identifier-only violations; never open a selected song or chart."""
    blocked = []
    blocked_sets = {str(x) for x in exclusion['set_ids']}
    for row in rows:
        path = Path(row['audio_rel'])
        reasons = []
        if (path.is_absolute() or '..' in path.parts or len(path.parts) != 4
                or path.parts[:2] not in (('dataset', '0'), ('dataset', 'std-taiko'))):
            reasons.append('outside_allowed_sources')
        if str(path.parent) in sealed:
            reasons.append('sealed_folder')
        if row['audio_sha256'] in exclusion['audio_sha256']:
            reasons.append('sealed_audio')
        if row.get('rec') in exclusion['recordings']:
            reasons.append('sealed_recording')
        if str(row.get('set_id')) in blocked_sets:
            reasons.append('sealed_set')
        if row.get('census_key') in exclusion['keys']:
            reasons.append('sealed_key')
        if reasons:
            blocked.append(dict(audio_sha256=row['audio_sha256'], audio_rel=str(path),
                                rec=row.get('rec'), reasons=reasons))
    return blocked


def audit(root):
    import pyarrow.parquet as pq

    run = root / 'artifacts/n1-night-20261018/runs'
    paths = dict(curated=run / 'v0data2-20261019/curated.json',
                 split=run / 'r0-20261019a/split.json',
                 census_run=root / 'artifacts/n1-redesign-20261028/census/run.json')
    args = SimpleNamespace(**json.loads(paths['census_run'].read_text())['args'])
    if (args.hard_holdout, args.hard_holdout_sha256) != SEAL:
        raise ValueError('Census and v0 curation use different hard-holdout seals')
    cols = ['path', 'set_dir', 'beatmap_set_id', 'audio_filename', 'audio_sha256',
            'api_status', 'group_id', 'eval_split', 'star']
    corpus = pq.read_table(root / args.corpus, columns=cols).to_pylist()
    result = {}
    sealed = build_exclusion(args, root, corpus, result)
    exclusion = load_exclusion(root / SEAL[0], SEAL[1])
    curated = json.loads(paths['curated'].read_text())
    split = json.loads(paths['split'].read_text())
    cohorts = dict(training=[r for r in curated if r['role'] == 'train' and r.get('keep')],
                   panel=split['panel'], val_hard=split['val_hard'],
                   val_random=[r for r in curated if r['role'] == 'val_random' and r.get('ok')],
                   std_extension=[r for r in curated if r['source'] == 'std' and
                                  r['role'] == 'train' and not r.get('keep')])
    cohorts['g0_probe'] = [r for r in cohorts['training'] if probe_bucket(r['audio_sha256']) in (0, 1, 2)]
    result['cohorts'] = {name: dict(rows=len(rows), audio=len({r['audio_sha256'] for r in rows}),
                                  violations=check_rows(rows, exclusion, sealed))
                         for name, rows in cohorts.items()}
    result['probe_buckets'] = dict(Counter(probe_bucket(r['audio_sha256']) for r in cohorts['training']))
    result['input_sha256'] = {str(p.relative_to(root)): digest(p) for p in paths.values()}
    for key in ('corpus', 'holdout1', 'holdout2', 'groups', 'inventory'):
        p = root / getattr(args, key)
        result['input_sha256'][str(p.relative_to(root))] = digest(p)
    result['input_sha256'][str(Path(SEAL[0]) / 'membership.json')] = SEAL[1]
    result['completed_scope'] = list(cohorts)
    result['pending_scope'] = ['keysound_extension', 'session1_extension', 'seed0_manifest']
    result['status'] = 'blocked' if any(c['violations'] for c in result['cohorts'].values()) else 'clear'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], cohorts={k: dict(rows=v['rows'], audio=v['audio'],
                         violations=len(v['violations'])) for k, v in result['cohorts'].items()})), flush=True)
    raise SystemExit(2 if result['status'] == 'blocked' else 0)


if __name__ == '__main__':
    main()
