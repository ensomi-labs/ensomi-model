"""Assemble and audit experiment cohorts before chart, audio, feature or label access."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import re

from .audit_inputs import context, check_rows, digest, probe_bucket

ROOT = Path('/Users/l/ensomi/ensomi-model')
OUT = ROOT / 'artifacts/n1-redesign-20261028/gpu'
FEAT = OUT.parent / 'featcache'
BULK = ROOT / 'artifacts/bulk/n1-cache'
RUNS = ROOT / 'artifacts/n1-night-20261018/runs'


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def build():
    paths, args, corpus, report, sealed, exclusion, decision = context(ROOT)
    curated = json.loads(paths['curated'].read_text())
    split = json.loads(paths['split'].read_text())
    recs = json.loads((RUNS / 'r0-20261019a/recordings.json').read_text())
    by_rec = {r['rec']: r for r in recs}
    by_sha = {r['audio_sha256']: r for r in recs if r.get('audio_sha256')}
    by_sha.update({r['audio_sha256']: dict(r, role='panel') for r in split['panel']})
    audit = {}

    def checked(name, rows):
        rows, removed = decision.filter(rows)
        bad = check_rows(rows, exclusion, sealed)
        audit[name] = dict(rows=len(rows), audio=len({r['audio_sha256'] for r in rows}),
                           removed=[r['audio_sha256'] for r in removed], violations=bad)
        dump(OUT / 'full-input-audit.json', audit)
        if bad:
            raise ValueError(f'New sealed input in {name}; see full-input-audit.json')
        return rows

    train = checked('training', [dict(r, split='fit_train') for r in curated if r['role'] == 'train' and r.get('keep')])
    panels = checked('panels', [dict(r, source='4k', split='fit_dev' if name == 'panel' else 'val-n1')
                               for name in ('panel', 'val_hard') for r in split[name]])
    val = checked('val_random', [dict(r, split='val_random') for r in curated if r['role'] == 'val_random' and r.get('ok')])
    std = checked('std_extension', [dict(r, split='ext', ext='std', drop_reasons=r.get('reasons', [])) for r in curated
                                    if r['source'] == 'std' and r['role'] == 'train' and not r.get('keep')])
    occupied = {r['audio_sha256'] for r in train + panels}
    census = list(csv.DictReader((OUT.parent / 'census/charts.csv').open()))
    key_candidates = {(r['set_dir'], r['audio'].casefold()) for r in census if r['source'] == '4k'
                      and r['split'] in ('cal', 'fit_dev') and int(r['n_obj']) >= 10
                      and int(r['ks_obj']) >= 0.5 * int(r['n_obj'])}
    key_rows = []
    for r in recs:
        if r['role'] not in ('cal', 'fit_dev') or r['audio_sha256'] in occupied:
            continue
        p = Path(r['audio_rel'])
        if (str(p.parent), p.name.casefold()) in key_candidates:
            key_rows.append(dict(r, item=r['audio_sha256'], split='ext', source='4k', ext='keysound'))
    key_rows = checked('keysound_candidates', key_rows)
    key_rows = sorted({r['audio_sha256']: r for r in key_rows}.values(), key=lambda r: r['audio_sha256'])[:150]
    keys = checked('keysound_extension', key_rows)
    identities = json.loads((ROOT / 'artifacts/timing-lens-20261024/private/session-01.identities.json').read_text())
    session_shas = set()
    for passage in identities['passages'].values():
        for item in passage.values():
            stem = Path(item.get('source', '')).stem
            if re.fullmatch('[0-9a-f]{64}', stem):
                session_shas.add(stem)
    if len(session_shas) != 15:
        raise ValueError(f'Session-1 identity count differs: {len(session_shas)}')
    session = []
    for sha in sorted(session_shas):
        r = by_sha[sha]
        if r['role'] not in ('fit_dev', 'panel'):
            raise ValueError('Session identity is not fit_dev')
        session.append(dict(r, item=sha, split='ext', source='4k', ext='session1'))
    session = checked('session1_candidates', session)
    # Completion is checked by G0 before this function; the source cache is immutable here.
    manifest = [json.loads(s) for s in (FEAT / 'v01-s0-009001/manifest.jsonl').read_text().splitlines() if s.strip()]
    checked('seed0_manifest', [dict(m, audio_sha256=r['sha256']) for r in manifest for m in r['memberships']])
    existing = {r['sha256'] for r in manifest}
    session = checked('session1_extension', [r for r in session if r['audio_sha256'] not in existing])
    probe = checked('g0_probe', [r for r in train if probe_bucket(r['audio_sha256']) in (0, 1, 2)])
    extension = checked('extension', std + keys + session)
    checked('targets', train + panels)
    checked('g0_s1', probe + panels + extension)
    checked('g0_s0_ext', extension)
    # Only the already-audited set directories may now supply chart headers.
    from ensomi_n.eval.census import read_header
    chart_cache = {}
    def charts(r):
        sha = r['audio_sha256']
        p = Path(r['audio_rel'])
        if 'charts' in r:
            cs = r['charts']
        elif r['source'] == '4k':
            cs = by_rec[r['rec']]['charts']
        else:
            key = (str(p.parent), p.name.casefold(), r['source'])
            if key not in chart_cache:
                cs = []
                for f in sorted((ROOT / p.parent).glob('*.osu')):
                    h = read_header(f)
                    if h.get('AudioFilename', '').casefold() == p.name.casefold() and h.get('Mode', '0') == ('0' if r['source'] == 'std' else '1'):
                        cs.append(str(f.relative_to(ROOT)))
                chart_cache[key] = cs
            cs = chart_cache[key]
        if not cs and not r.get('ext'):
            raise ValueError(f'No reference charts for {sha}')
        for c in cs:
            cp = Path(c)
            if cp.is_absolute() or '..' in cp.parts or cp.parent != p.parent:
                raise ValueError('Chart path outside audited input folder')
        return dict(r, item=r.get('item', sha), charts=cs)
    cohorts = {k: [charts(r) for r in rows] for k, rows in
               dict(training=train, panels=panels, val_random=val, extension=extension, g0_probe=probe).items()}
    dump(OUT / 'cohorts.json', cohorts)
    dump(OUT / 'session1-sha256.json', dict(sha256=sorted(session_shas),added=[r['audio_sha256'] for r in session]))
    report.update(cohorts=audit, decision=json.loads(paths['decision'].read_text()),
                  input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths.values()})
    dump(OUT / 'full-input-audit.json', report)
    return cohorts
