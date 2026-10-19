"""E2 and E3 on the mac in one process: the strata census, the sealed hard holdout, the reader census.

Labels only: `.osu` files, the corpus table, membership lists and set metadata.
Audio files are opened only to hash the bytes of the drawn holdout's audio. Run
from the project root, with ``n/src`` and ``src`` of one code commit on
``PYTHONPATH``::

    nice -n 19 .venv/bin/python scripts/n1/census_run.py --commit SHA \\
        --census-out artifacts/n1-night-20261018/census --holdout-out artifacts/n1-eval/hard-holdout-v1

Inputs (defaults are the project's paths): the 4K corpus table
``data/r2-corpus.parquet`` (ranked and loved charts only), timing holdout-1 and
holdout-2 (``audio_sha256`` columns), the dedup groups, the voted-tag names,
and ``dataset/std-taiko/`` (modes 0 and 1; tags and status from each set's
``metadata.json`` when it has them).

Splits. A 4K set takes, by its audio and group: ``holdout2`` (timing
holdout-2), ``holdout1`` (timing holdout-1, or a dedup member flagged
holdout), ``r2-heldout`` (corpus ``eval_split`` heldout), ``cal``
(calibration), ``fit_dev`` (R2's group hash, ``sha256('r2-fit-dev-v1:' +
group_id)`` mod 10 == 0) or ``fit_train``; every 4K set of a recording takes
the first of these that any of them has. A std-taiko set whose recording also
holds 4K sets is ``std-taiko-linked-sealed`` (when those are in a holdout) or
``std-taiko-linked``; the rest are ``std-taiko``, the hard holdout's pool. After
the draw, pool sets are ``hard-holdout`` or ``std-taiko-design``.

Sealed (counts only, no per-set row anywhere): ``holdout1``, ``holdout2``,
``r2-heldout``, ``std-taiko-linked-sealed``, ``hard-holdout``.

Preflight checks every input before anything is written, and the holdout is
never redrawn: an existing membership stops the run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from ensomi_n.eval import census
from ensomi_n.eval.strata import HARD, STRATA, NAMES

SEALED = ('holdout1', 'holdout2', 'r2-heldout', 'std-taiko-linked-sealed', 'hard-holdout')
ORDER_4K = ('holdout2', 'holdout1', 'r2-heldout', 'cal', 'fit_dev', 'fit_train')
STATUSES = ('ranked', 'loved')
CORPUS_COLUMNS = ('path', 'set_dir', 'beatmap_set_id', 'audio_filename', 'audio_sha256', 'api_status',
                  'group_id', 'eval_split', 'api_tag_ids')
TRIGGER_SHARE = 0.01


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fit_dev(group_id: str) -> bool:
    d = hashlib.sha256(('r2-fit-dev-v1:' + group_id).encode()).digest()
    return int.from_bytes(d[:8], 'big') % 10 == 0


class Union:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def load_tag_names(path) -> dict[int, str]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and 'tags' in data:
        data = data['tags']
    if isinstance(data, dict):
        return {int(k): (v['name'] if isinstance(v, dict) else str(v)) for k, v in data.items()}
    return {int(t['id'] if 'id' in t else t['tag_id']): t['name'] for t in data}


def holdout_shas(path) -> set[str]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'audio_sha256' not in rows[0]:
        raise ValueError(f'{path}: no audio_sha256 column')
    return {r['audio_sha256'].strip() for r in rows if r['audio_sha256'].strip()}


def osu_header(path) -> dict:
    return census.read_header(path)


def std_taiko_units(root: Path, tag_names: dict, report: dict) -> list[dict]:
    """One row per mode-0/1 `.osu` file under dataset/std-taiko, with tags and status from metadata.json."""
    rows = []
    base = root / 'dataset' / 'std-taiko'
    keys_seen: dict[str, int] = {}
    sets_with_tags = sets_with_meta = 0
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(d.glob('*.osu'))
        if not files:
            continue
        meta, by_md5 = None, {}
        mp = d / 'metadata.json'
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text())
                sets_with_meta += 1
                for k in (meta.keys() if isinstance(meta, dict) else []):
                    keys_seen[k] = keys_seen.get(k, 0) + 1
                for b in (meta.get('beatmaps') or []) if isinstance(meta, dict) else []:
                    if b.get('checksum'):
                        by_md5[b['checksum']] = b
            except ValueError:
                meta = None
        audios = {}
        info = []
        for f in files:
            h = osu_header(f)
            try:
                mode = int(float(h.get('Mode', '0') or 0))
            except ValueError:
                continue
            if mode not in (0, 1):
                continue
            audio = h.get('AudioFilename', '')
            audios.setdefault(audio.casefold(), audio)
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            b = by_md5.get(md5)
            tags, known, status = [], False, ''
            if b is not None:
                status = str(b.get('status') or '')
                if 'top_tag_ids' in b:
                    known = bool(tag_names)
                    tags = [tag_names.get(int(t['tag_id']), str(t['tag_id'])) for t in (b.get('top_tag_ids') or [])]
            info.append((f, mode, audio, tags, known, status))
        if any(t[3] for t in info):
            sets_with_tags += 1
        several = len(audios) > 1
        for f, mode, audio, tags, known, status in info:
            rel = d.relative_to(root / 'dataset').as_posix()
            key = f"{rel}/{Path(audio).stem}" if several else rel
            rows.append(dict(file=f.relative_to(root).as_posix(), key=f"{key}@{'std' if mode == 0 else 'taiko'}",
                             source='std-taiko', token='s:' + f"{rel}/{audio}".casefold(), status=status,
                             mode='std' if mode == 0 else 'taiko', set_id=d.name if d.name.isdigit() else '',
                             tags=';'.join(tags), tags_known='1' if known else '0', audio=audio))
    report['std_taiko'] = dict(sets_with_metadata=sets_with_meta, metadata_top_level_keys=keys_seen,
                               sets_with_voted_tags=sets_with_tags, files=len(rows))
    return rows


def build_units(args, report: dict) -> list[census.Unit]:
    import pyarrow.parquet as pq
    root = Path(args.root)
    table = pq.read_table(str(root / args.corpus))
    missing = [c for c in CORPUS_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f'{args.corpus}: missing columns {missing}')
    corpus = table.select(list(CORPUS_COLUMNS)).to_pylist()
    tag_names = load_tag_names(root / args.tags) if args.tags and (root / args.tags).is_file() else {}
    h1, h2 = holdout_shas(root / args.holdout1), holdout_shas(root / args.holdout2)
    groups = json.loads((root / args.groups).read_text())
    u = Union()
    rows = []
    kept = [r for r in corpus if r['api_status'] in STATUSES]
    per_dir: dict[str, set] = {}
    for r in kept:
        per_dir.setdefault(r['set_dir'], set()).add((r['audio_filename'] or '').casefold())
    for r in kept:
        rel = Path(r['set_dir']).relative_to('dataset').as_posix()
        key = f"{rel}/{Path(r['audio_filename'] or '').stem}" if len(per_dir[r['set_dir']]) > 1 else rel
        tags = [tag_names.get(int(t), str(t)) for t in (r['api_tag_ids'] or [])]
        token = 'g:' + r['group_id']
        if r['audio_sha256']:
            u.union(token, 'a:' + r['audio_sha256'])
        u.find(token)
        rows.append(dict(file=r['path'], key=key, source='4k', token=token, status=r['api_status'], mode='mania4k',
                         set_id=str(r['beatmap_set_id'] or ''), tags=';'.join(tags),
                         tags_known='1' if tag_names else '0', audio=r['audio_filename'] or '',
                         sha=r['audio_sha256'] or '', group=r['group_id'], eval_split=r['eval_split']))
    st_rows = std_taiko_units(root, tag_names, report)
    for r in st_rows:
        u.find(r['token'])
    dedup_holdout = set()
    for g in groups:
        ids = []
        for m in g['members']:
            if m.get('source') == '4k-corpus':
                ids.append('a:' + m['sha256'])
                ids += ['g:' + x for x in m.get('corpus_group_ids', [])]
                if m.get('holdout'):
                    dedup_holdout.add(m['sha256'])
            elif m.get('source') == 'std-taiko':
                rel = Path(m['path']).relative_to('dataset').as_posix()
                ids.append('s:' + rel.casefold())
        for a, b in zip(ids, ids[1:]):
            u.union(a, b)
    rec_split: dict[str, int] = {}
    for r in rows:
        if r['sha'] in h2:
            s = 'holdout2'
        elif r['sha'] in h1 or r['sha'] in dedup_holdout:
            s = 'holdout1'
        elif r['eval_split'] == 'heldout':
            s = 'r2-heldout'
        elif r['eval_split'] == 'calibration':
            s = 'cal'
        else:
            s = 'fit_dev' if fit_dev(r['group']) else 'fit_train'
        rec = u.find(r['token'])
        rec_split[rec] = min(rec_split.get(rec, 99), ORDER_4K.index(s))
    units: dict[str, census.Unit] = {}
    for r in rows + st_rows:
        rec = u.find(r['token'])
        if r['source'] == '4k':
            split = ORDER_4K[rec_split[rec]]
        elif rec in rec_split:
            split = 'std-taiko-linked-sealed' if ORDER_4K[rec_split[rec]] in SEALED else 'std-taiko-linked'
        else:
            split = 'std-taiko'
        x = units.get(r['key'])
        if x is None:
            units[r['key']] = census.Unit(r['key'], [r['file']], r['source'], 'rec:' + rec, split, r['status'],
                                          r['mode'], r['set_id'], [t for t in r['tags'].split(';') if t],
                                          r['tags_known'] == '1', r.get('sha', ''))
        else:
            x.files.append(r['file'])
            x.tags = sorted(set(x.tags) | {t for t in r['tags'].split(';') if t})
            if not x.status and r['status']:
                x.status = r['status']
    # std-taiko audio is hashed only where a pair is possible: recordings with sets of several set ids.
    by_rec: dict[str, set] = {}
    for x in units.values():
        by_rec.setdefault(x.recording, set()).add(x.set_id or x.key)
    hashed = 0
    audio_of = {r['key']: r['audio'] for r in st_rows}
    for x in units.values():
        if x.source == 'std-taiko' and len(by_rec[x.recording]) > 1:
            audio = audio_of.get(x.key, '')
            path = census._audio_path(root, x.files, audio) if audio else None
            if path is not None:
                x.audio_sha256 = sha256_file(path)
                hashed += 1
    report['std_taiko']['audio_hashed_for_pairs'] = hashed
    shas_4k = {r['sha'] for r in rows}
    report['inputs'] = dict(
        corpus_rows=len(corpus), corpus_ranked_loved=len(kept), tag_names=len(tag_names),
        holdout1_shas=len(h1), holdout1_matched=len(h1 & shas_4k), holdout2_shas=len(h2),
        holdout2_matched=len(h2 & shas_4k), dedup_groups=len(groups), dedup_holdout_members=len(dedup_holdout),
        sha256={name: sha256_file(root / p) for name, p in (('corpus', args.corpus), ('holdout1', args.holdout1),
                                                            ('holdout2', args.holdout2), ('groups', args.groups))
                if (root / p).is_file()},
        units=len(units), units_by_split=census._count(x.split for x in units.values()))
    return list(units.values())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root', default='.')
    p.add_argument('--corpus', default='data/r2-corpus.parquet')
    p.add_argument('--holdout1', default='artifacts/timing-20261009/opt/holdout/holdout.csv')
    p.add_argument('--holdout2', default='artifacts/timing-20261009/opt/ft/pop/holdout2.csv')
    p.add_argument('--groups', default='artifacts/bulk/music-dedup-20261015/groups.json')
    p.add_argument('--tags', default='artifacts/n1-inventory-20261014/tags.json')
    p.add_argument('--census-out', required=True)
    p.add_argument('--holdout-out', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--seed', type=int, default=census.DRAW_SEED)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--holdout-sha256', help='rerun against an existing seal (its membership sha256) instead of '
                                            'drawing; the seal is checked and never changed')
    args = p.parse_args(argv)
    root = Path(args.root)
    out = root / args.census_out
    hold = root / args.holdout_out
    if (hold / census.MEMBERSHIP).exists() and not args.holdout_sha256:
        print(f'{hold / census.MEMBERSHIP} exists: the holdout is never redrawn; stopping', file=sys.stderr)
        return 2
    if args.holdout_sha256 and not (hold / census.MEMBERSHIP).exists():
        print('--holdout-sha256 needs an existing seal', file=sys.stderr)
        return 2
    report: dict = dict(commit=args.commit, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    t0 = time.time()
    units = build_units(args, report)
    report['t_units_s'] = round(time.time() - t0, 1)
    print(json.dumps(report, indent=1), flush=True)
    if args.preflight_only:
        return 0
    if report['inputs']['holdout2_matched'] < 0.9 * report['inputs']['holdout2_shas']:
        print('holdout-2 audio is mostly unmatched in the corpus; stopping before any output', file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'log.txt').open('a', encoding='utf-8')

    from ensomi_model.evaluation.redlines import musical_grid  # noqa: F401 - fail early if the R line is missing
    import reader_census

    def on_set(unit, ref, charts, paths, st):
        res = reader_census.compare(ref, ref.meta.get('timing_chart'))
        res['causes'] = reader_census.causes(res.pop('changed_t'), st.lines)
        return dict(e3=res)

    def progress(i, n):
        if i % 500 == 0 or i == n:
            log.write(f'{time.strftime("%H:%M:%S")} {i}/{n} sets\n')
            log.flush()

    t1 = time.time()
    results = census.compute(units, root, progress=progress, on_set=on_set)
    report['t_compute_s'] = round(time.time() - t1, 1)

    if args.holdout_sha256:
        sha = args.holdout_sha256
        hard_keys = census.load_exclusion(hold, sha)['keys']
        report['holdout'] = dict(path=str(args.holdout_out), membership_sha256=sha, reused=True,
                                 rule=json.loads((hold / 'rule.json').read_text()))
    else:
        drawn = census.draw(results, pool='std-taiko', seed=args.seed)
        sha = census.seal(drawn, hold, root, commit=args.commit, name=hold.name,
                          date=time.strftime('%Y-%m-%d', time.gmtime()))
        hard_keys = census.load_exclusion(hold, sha)['keys']
        report['holdout'] = dict(path=str(args.holdout_out), membership_sha256=sha, rule=drawn['rule'])

    def split_of(r):
        if r.unit.key in hard_keys:
            return 'hard-holdout'
        return 'std-taiko-design' if r.unit.split == 'std-taiko' else r.unit.split

    flags = {}
    for r in results:
        e3 = r.extra.get('e3')
        if e3 is not None:
            flags[r.unit.key] = {'reader_changed': e3['changed_heads'] > 0,
                                 'reader_changed>1%heads': e3['changed_share'] > 0.01,
                                 'reader_bars_differ': e3['bars_unmatched'] > 0}
        elif r.strata is not None:
            flags[r.unit.key] = {'reader_r_failed': True}
    table = census.tabulate(results, split_of=split_of, set_flags=flags)
    pooled = census.tabulate(results, split_of=lambda r: 'all-data', set_flags=flags)
    status = census.e2_status(table, holdout='holdout2', hard='hard-holdout',
                              floors=('fit_dev', 'std-taiko-design'))
    rows = census.strata_rows(results, sealed=lambda r: split_of(r) in SEALED, split_of=split_of)

    # E3: per hard stratum, the share of recordings whose beat lines change, and the top 10 (unsealed only).
    e3 = {}
    allv = pooled['splits']['all-data']
    triggered = []
    for s in ('all',) + STRATA:
        v = allv[s]
        n_changed = v['flags'].get('reader_changed', 0)
        share = n_changed / v['recordings'] if v['recordings'] else 0.0
        e3[s] = dict(recordings=v['recordings'], changed=n_changed, share=share,
                     changed_over_1pct_heads=v['flags'].get('reader_changed>1%heads', 0),
                     bars_differ=v['flags'].get('reader_bars_differ', 0), r_failed=v['flags'].get('reader_r_failed', 0),
                     by_split={n: dict(recordings=t[s]['recordings'], changed=t[s]['flags'].get('reader_changed', 0))
                               for n, t in table['splits'].items() if '|' not in n})
        if s in HARD and share > TRIGGER_SHARE:
            triggered.append(s)
    cands = []
    for r in results:
        if r.strata is None or split_of(r) in SEALED or 'e3' not in r.extra:
            continue
        res = r.extra['e3']
        if res['changed_heads'] == 0:
            continue
        member = sorted(set(r.strata.members) | ({'S4'} if r.strata.s4_spans and not r.strata.s3 else set()),
                        key=STRATA.index)
        hit = [s for s in member if s in triggered]
        if not hit:
            continue
        kinds = {}
        for line in r.strata.lines:
            kinds[line.kind] = kinds.get(line.kind, 0) + 1
        cands.append(dict(key=r.unit.key, set_id=r.unit.set_id, split=split_of(r), strata=member,
                          recording=r.unit.recording, changed_share=round(res['changed_share'], 4),
                          changed_heads=res['changed_heads'], heads=res['heads'], n_only=res['n_only'],
                          r_only=res['r_only'], causes=res['causes'],
                          why=reader_census.why(res, kinds, res['causes'])))
    cands.sort(key=lambda c: (-c['changed_share'], -c['changed_heads'], c['key']))
    seen, top = set(), []
    for c in cands:  # one chart per recording
        if c['recording'] not in seen:
            seen.add(c['recording'])
            top.append(c)
    cause_heads: dict[str, int] = {}
    cause_sets: dict[str, int] = {}
    for r in results:
        res = r.extra.get('e3')
        if res and res['changed_heads']:
            lead = next(iter(res['causes']), 'none').split('@', 1)[0] + ('@fast' if '@fast' in
                                                                        next(iter(res['causes']), '') else '')
            cause_sets[lead] = cause_sets.get(lead, 0) + 1
            for k, n in res['causes'].items():
                base = k.split(' meter', 1)[0]
                cause_heads[base] = cause_heads.get(base, 0) + n
    reader = dict(rule='E3: if N changes the beat lines of more than 1 % of a hard stratum\'s recordings, the 10 '
                       'most-changed charts go to the human before E4 uses N\'s reader',
                  tolerance_ms=reader_census.TOL_MS, strata=e3, triggered=triggered, top10=top[:10],
                  leading_cause_sets=dict(sorted(cause_sets.items(), key=lambda kv: -kv[1])),
                  cause_heads=dict(sorted(cause_heads.items(), key=lambda kv: -kv[1])),
                  top_by_stratum={s: [c for c in top if s in c['strata']][:5] for s in triggered})
    report['t_total_s'] = round(time.time() - t0, 1)
    report['errors'] = table['errors']
    census.write_census(out, table, status, rows, meta=dict(run=report, sealed=list(SEALED)))
    (out / 'reader-census.json').write_text(json.dumps(census._jsonable(reader), indent=1) + '\n')
    (out / 'run.json').write_text(json.dumps(census._jsonable(report), indent=1) + '\n')
    with (out / 'census.md').open('a') as f:
        f.write('\n## E3\n\n| Stratum | Recordings | N changes beat lines | Share | Bars differ |\n'
                '| --- | ---: | ---: | ---: | ---: |\n')
        for s, v in e3.items():
            f.write(f"| {s} | {v['recordings']} | {v['changed']} | {v['share']:.4f} | {v['bars_differ']} |\n")
        f.write(f'\nTriggered hard strata: {", ".join(triggered) or "none"}\n')
    print(f'done: membership sha256 {sha}; {len(results)} sets; {table["errors"]} unreadable', flush=True)
    log.close()
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
