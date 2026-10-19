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
never redrawn: an existing membership stops the run, unless
``--holdout-sha256`` names its digest, which is checked before anything is
computed; the run then recounts against the seal.

E3 runs twice in the same pass on the same strata: with E0's canonical lines
(``e0_canonical``, frozen from ``be69bf9``) and with N's current lines, each
against R's reader. Outputs besides the census tables: ``reader-census.json``
(E3 before and after per stratum, free spans and BL-20's exact chance per split
and stratum, leading causes, the top lists, the normal-tempo ``same``-set test),
``e3.md`` (the same as tables), ``e3-sets.jsonl`` and ``same-normal.jsonl``
(per set, unsealed splits only) and, with ``--top10-out``, the renders
(``reader_top10``). ``--e3-only`` skips the census tables.
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
from ensomi_n.eval.strata import HARD, NAMES, STRATA, recording_strata

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



def _desc(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def after_reason(new: dict, roles_why: str) -> str:
    """One line on a changed set after the fix: where its changed heads lie and what R drops."""
    if not new['changed_heads']:
        return 'unchanged'
    top = next(iter(new['causes']), '')
    base = top.split('@', 1)[0]
    where = f"{new['causes'][top]} under N's {base} lines" if top else 'no governing line'
    if '@fast' in top:
        where += f" at a notated tempo of 160 BPM or more ({top.split('@fast ', 1)[1]})"
    return (f"{new['changed_heads']} heads changed ({new['changed_free']} in free spans, {new['changed_refined']} "
            f"still changed refined); most: {where}; {roles_why}")


def free_table(results, split_of) -> dict:
    """``free_share`` and BL-20's exact chance per split and stratum, from each set's `free_sums`.

    Per cell: ``w`` (head weight in the stratum's spans: each scorable head weighs 1, split between the
    lines around it), ``wf`` (the part on free lines), ``free_share`` = wf / w, ``chance_free`` (the
    weighted mean chance of BL-20 on free lines, at the spacing they are compared at), ``chance_set`` (on
    the other lines), ``chance_all``, and ``recordings`` / ``recordings_free`` (those with any weight on
    free lines). ``all-data`` pools every split; sealed splits appear only as these counts.
    """
    groups: dict[tuple[str, str], list] = {}
    for r in results:
        if r.strata is None or 'free' not in r.extra.get('e3', {}):
            continue
        for split in (split_of(r), 'all-data'):
            groups.setdefault((split, r.unit.recording), []).append(r)
    zero = dict(w=0.0, wf=0.0, c=0.0, cf=0.0)
    cells: dict[str, dict] = {}
    for (split, rec), sets in groups.items():
        member = recording_strata([r.strata for r in sets])
        member['all'] = {r.unit.key: None for r in sets}
        by_key = {r.unit.key: r for r in sets}
        for stratum, spans_by_set in member.items():
            acc = dict(zero)
            for key, spans in spans_by_set.items():
                fs = by_key[key].extra['e3']['free']
                name = 'whole' if spans is None else ('S4spans' if stratum == 'S4' else stratum)
                for k, v in fs.get(name, zero).items():
                    acc[k] += v
            cell = cells.setdefault(split, {}).setdefault(stratum, dict(zero, recordings=0, recordings_free=0))
            for k, v in acc.items():
                cell[k] += v
            cell['recordings'] += 1
            cell['recordings_free'] += int(acc['wf'] > 0)
    for split in cells.values():
        for cell in split.values():
            w, wf, c, cf = cell['w'], cell['wf'], cell['c'], cell['cf']
            cell['free_share'] = wf / w if w > 0 else None
            cell['chance_free'] = cf / wf if wf > 0 else None
            cell['chance_set'] = (c - cf) / (w - wf) if w - wf > 0 else None
            cell['chance_all'] = c / w if w > 0 else None
    order = ['all-data'] + sorted(k for k in cells if k != 'all-data')
    return {k: {s: cells[k][s] for s in ('all',) + STRATA if s in cells[k]} for k in order if k in cells}


def same_summary(results, split_of) -> dict:
    """§4's test on the sets whose E0 changes lead with redundant lines at normal tempo (``same``).

    Counts pool every split; the per-set reading (``rows``) and its tallies are unsealed only.
    """
    import reader_census
    total = sealed_n = still = 0
    still_lead: dict[str, int] = {}
    rows = []
    tally = dict(cascade=0, stale=0, dropped_between=0, still_changed=0, still_changed_refined=0,
                 prior={}, between={}, cascade_still_changed=0)
    for r in results:
        res = r.extra.get('e3')
        if not res or not res['old']['changed_heads'] or reader_census.lead(res['old']['causes']) != 'same':
            continue
        total += 1
        if res['new']['changed_heads']:
            still += 1
            name = reader_census.lead(res['new']['causes'])
            still_lead[name] = still_lead.get(name, 0) + 1
        if split_of(r) in SEALED:
            sealed_n += 1
            continue
        d = res.get('same_diag')
        if d is None:
            continue
        rows.append(dict(key=r.unit.key, split=split_of(r), changed_after=res['new']['changed_heads'],
                         changed_refined_after=res['new']['changed_refined'], **d))
        tally['cascade'] += int(d['cascade'])
        tally['stale'] += int(d['stale'])
        tally['still_changed'] += int(res['new']['changed_heads'] > 0)
        tally['still_changed_refined'] += int(res['new']['changed_refined'] > 0)
        tally['cascade_still_changed'] += int(d['cascade'] and res['new']['changed_heads'] > 0)
        k = f"{d['prior']['kind']} / R {d['prior']['r_role']}" + (' / R grid stale' if d['stale'] else '')
        tally['prior'][k] = tally['prior'].get(k, 0) + 1
        for name, n in d['between'].items():
            tally['between'][name] = tally['between'].get(name, 0) + n
        tally['dropped_between'] += int(all(x.startswith('expressive') for x in d['between']))
    tally['prior'] = _desc(tally['prior'])
    tally['between'] = _desc(tally['between'])
    return dict(sets=total, sealed=sealed_n, unsealed_read=len(rows), still_changed_after=still,
                still_changed_lead=_desc(still_lead), tally=tally, rows=rows)


def _num(x) -> str:
    return '' if x is None else f'{x:.4f}'


def e3_markdown(reader: dict, report: dict) -> str:
    """E3 before and after, free spans and the §4 test, as tables."""
    out = ['# E3 before and after the reader fix', '', f"commit {report.get('commit')}, mode {report.get('mode')}", '',
           '| Stratum | Recordings | Changed before (E0) | Share | Changed after | Share | Refined after | Share | '
           'Outside free spans | > 1 % heads after | R failed |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s, v in reader['strata'].items():
        out.append(f"| {s} | {v['recordings']} | {v['reader_changed_e0']} | {v['share_e0']:.4f} | "
                   f"{v['reader_changed']} | {v['share']:.4f} | {v['reader_changed_refined']} | "
                   f"{v['share_refined']:.4f} | {v['reader_changed_outside_free']} | {v['reader_changed>1%heads']} | "
                   f"{v['reader_r_failed']} |")
    out += ['', f"Triggered after: {', '.join(reader['triggered']) or 'none'}; before: "
                f"{', '.join(reader['triggered_e0']) or 'none'}", '',
            f"Leading cause per changed set, after: {reader['leading_cause_sets']}", '',
            f"Leading cause per changed set, before: {reader['leading_cause_sets_e0']}", '',
            f"Changed heads by governing N line, after: {reader['cause_heads']}", '',
            f"Changed heads by governing N line, before: {reader['cause_heads_e0']}", '']
    for split, cells in reader['free'].items():
        if split not in ('all-data', 'holdout2', 'hard-holdout'):
            continue
        out += [f'## Free spans: {split}', '',
                '| Stratum | Recordings | With free spans | Head weight | On free lines | free_share | '
                'BL-20 chance, free | chance, phase set | chance, all |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for s, c in cells.items():
            out.append(f"| {s} | {c['recordings']} | {c['recordings_free']} | {c['w']:.0f} | {c['wf']:.0f} | "
                       f"{_num(c['free_share'])} | {_num(c['chance_free'])} | {_num(c['chance_set'])} | "
                       f"{_num(c['chance_all'])} |")
        out.append('')
    d = reader['same_normal']
    out += ['## Normal-tempo `same` sets (§4)', '',
            f"{d['sets']} sets lead with `same` before the fix ({d['sealed']} in sealed splits, counted only); "
            f"{d['unsealed_read']} unsealed read; still changed after the fix: {d['still_changed_after']} "
            f"(leading after: {d['still_changed_lead']}).", '', f"Tally (unsealed): {d['tally']}", '']
    for name in ('top10', 'top10_refined'):
        out += [f'## {name}', '', '| Set | Split | Strata | Changed | Refined | Free | Before | Reason |',
                '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
        for c in reader[name]:
            out.append(f"| `{c['key']}` | {c['split']} | {' '.join(c['strata'])} | {c['changed_heads']} of "
                       f"{c['heads']} | {c['changed_refined']} | {c['changed_free']} | {c['changed_e0']} | "
                       f"{c['reason']} |")
        out.append('')
    if 'renders' in reader:
        out += [f"Renders: {reader['renders']}", '']
    return '\n'.join(out) + '\n'


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
    p.add_argument('--e3-only', action='store_true',
                   help='with --holdout-sha256: E3 before and after the reader fix, free spans and the '
                        'normal-tempo same-set test only; no census tables')
    p.add_argument('--top10-out', help='render the top lists after the fix into this directory (unsealed only)')
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
    if args.e3_only and not args.holdout_sha256:
        print('--e3-only reuses the seal: give --holdout-sha256', file=sys.stderr)
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
    import e0_canonical
    import reader_census

    # A rerun checks the seal before anything is computed, so no sealed set ever gets per-set detail.
    hard_keys: set[str] = set()
    if args.holdout_sha256:
        hard_keys = census.load_exclusion(hold, args.holdout_sha256)['keys']

    def sealed_unit(u) -> bool:
        return u.split in SEALED or u.key in hard_keys or (not args.holdout_sha256 and u.split == 'std-taiko')

    def on_set(unit, ref, charts, paths, st):
        sealed = sealed_unit(unit)
        res = reader_census.e3_set(ref, ref.meta.get('timing_chart'), st, e0_canonical.reference_lines(ref),
                                   with_detail=not sealed)
        if not sealed:
            h = census.read_header(ref.meta.get('timing_chart'))
            res['meta'] = dict(artist=h.get('Artist', ''), title=h.get('Title', ''), version=h.get('Version', ''))
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
    e3_errors: dict[str, int] = {}
    for r in results:
        e3 = r.extra.get('e3')
        if e3 is not None:
            new, old = e3['new'], e3['old']
            flags[r.unit.key] = {'reader_changed': new['changed_heads'] > 0,
                                 'reader_changed_e0': old['changed_heads'] > 0,
                                 'reader_changed_refined': new['changed_refined'] > 0,
                                 'reader_changed_outside_free': new['changed_heads'] > new['changed_free'],
                                 'reader_changed>1%heads': new['changed_share'] > 0.01,
                                 'reader_changed_e0>1%heads': old['changed_share'] > 0.01,
                                 'reader_bars_differ': new['bars_unmatched'] > 0}
        elif r.strata is not None:
            flags[r.unit.key] = {'reader_r_failed': True}
            err = str(r.extra.get('error', 'no result')).split(':', 1)[0]
            e3_errors[err] = e3_errors.get(err, 0) + 1
    report['e3_errors'] = e3_errors
    table = census.tabulate(results, split_of=split_of, set_flags=flags)
    pooled = census.tabulate(results, split_of=lambda r: 'all-data', set_flags=flags)

    # E3 per stratum, before (E0's lines) and after (N's current lines), on the same strata.
    e3 = {}
    allv = pooled['splits']['all-data']
    triggered, triggered_e0 = [], []
    for s in ('all',) + STRATA:
        v = allv[s]
        fl = v['flags']
        n_rec = v['recordings']
        row = dict(recordings=n_rec)
        for name in ('reader_changed', 'reader_changed_e0', 'reader_changed_refined', 'reader_changed_outside_free',
                     'reader_changed>1%heads', 'reader_changed_e0>1%heads', 'reader_bars_differ', 'reader_r_failed'):
            row[name] = fl.get(name, 0)
        row['share'] = row['reader_changed'] / n_rec if n_rec else 0.0
        row['share_e0'] = row['reader_changed_e0'] / n_rec if n_rec else 0.0
        row['share_refined'] = row['reader_changed_refined'] / n_rec if n_rec else 0.0
        row['by_split'] = {n: dict(recordings=t[s]['recordings'], changed=t[s]['flags'].get('reader_changed', 0),
                                   changed_e0=t[s]['flags'].get('reader_changed_e0', 0))
                           for n, t in table['splits'].items() if '|' not in n}
        e3[s] = row
        if s in HARD and row['share'] > TRIGGER_SHARE:
            triggered.append(s)
        if s in HARD and row['share_e0'] > TRIGGER_SHARE:
            triggered_e0.append(s)

    free = free_table(results, split_of)

    # Per-set rows (unsealed only), top lists after the fix, leading causes before and after.
    def member_strata(r):
        return sorted(set(r.strata.members) | ({'S4'} if r.strata.s4_spans and not r.strata.s3 else set()),
                      key=STRATA.index)

    rows_out, cands = [], []
    for r in results:
        if r.strata is None or split_of(r) in SEALED or 'e3' not in r.extra:
            continue
        res = r.extra['e3']
        new, old = res['new'], res['old']
        roles_why = reader_census._why_roles(new, {})
        row = dict(key=r.unit.key, set_id=r.unit.set_id, split=split_of(r), recording=r.unit.recording,
                   strata=member_strata(r), heads=new['heads'], changed_heads=new['changed_heads'],
                   changed_share=round(new['changed_share'], 4), changed_free=new['changed_free'],
                   changed_refined=new['changed_refined'], free_heads=new['free_heads'],
                   changed_e0=old['changed_heads'], changed_share_e0=round(old['changed_share'], 4),
                   lead_e0=reader_census.lead(old['causes']) if old['changed_heads'] else '',
                   lead=reader_census.lead(new['causes']) if new['changed_heads'] else '',
                   causes=new['causes'], causes_e0=old['causes'], n_only=new['n_only'], r_only=new['r_only'],
                   roles=new['roles'], reason=after_reason(new, roles_why),
                   artist='', title='', version='')
        row.update(res.get('meta', {}))
        row['refined_share'] = round(row['changed_refined'] / row['heads'], 4) if row['heads'] else 0.0
        rows_out.append(row)
        if new['changed_heads'] and any(s in triggered for s in row['strata']):
            cands.append(row)

    def top(key):
        seen, out_rows = set(), []
        for c in sorted(cands, key=lambda c: (-c[key], -c['changed_heads'], c['key'])):
            if c['recording'] not in seen:
                seen.add(c['recording'])
                out_rows.append(c)
        return out_rows

    top_strict = top('changed_share')[:10]
    top_refined = [c for c in top('refined_share')[:10] if c['changed_refined']]

    def leads(which):
        sets: dict[str, int] = {}
        heads: dict[str, int] = {}
        for r in results:
            res = r.extra.get('e3')
            if res and res[which]['changed_heads']:
                name = reader_census.lead(res[which]['causes'])
                sets[name] = sets.get(name, 0) + 1
                for k, n in res[which]['causes'].items():
                    base = k.split(' meter', 1)[0]
                    heads[base] = heads.get(base, 0) + n
        return _desc(sets), _desc(heads)

    lead_sets_e0, cause_heads_e0 = leads('old')
    lead_sets, cause_heads = leads('new')
    same_diag = same_summary(results, split_of)
    same_rows = same_diag.pop('rows')

    reader = dict(rule='E3: if N changes the beat lines of more than 1 % of a hard stratum\'s recordings, the 10 '
                       'most-changed charts go to the human before E4 uses N\'s reader',
                  tolerance_ms=reader_census.TOL_MS, strata=e3, triggered=triggered, triggered_e0=triggered_e0,
                  free=free, top10=top_strict, top10_refined=top_refined,
                  leading_cause_sets=lead_sets, cause_heads=cause_heads,
                  leading_cause_sets_e0=lead_sets_e0, cause_heads_e0=cause_heads_e0, same_normal=same_diag)

    if args.top10_out:
        try:
            import reader_top10
            from ensomi_n.eval.strata import set_strata
            from ensomi_n.labels import read_osu, reference_from_charts
            by_key = {u.key: u for u in units}
            picked, seen = [], set()
            for c in top_strict + top_refined:
                if c['key'] not in seen:
                    seen.add(c['key'])
                    picked.append(c)
            items = []
            for c in picked:
                u = by_key[c['key']]
                paths = sorted(root / f for f in u.files if (root / f).is_file())
                charts = [read_osu(p_) for p_ in paths]
                ref = reference_from_charts(charts, u.key)
                st = set_strata(ref, charts, tags=u.tags, tags_known=u.tags_known, key=u.key)
                items.append((c, reader_top10.chart_item(ref, st, ref.meta['timing_chart'],
                                                         e0_canonical.reference_lines(ref), c, c)))
            rendered = reader_top10.render(items, root / args.top10_out,
                                           strict=[c['key'] for c in top_strict])
            reader['renders'] = dict(dir=str(args.top10_out), backend=rendered['backend'],
                                     files=[r_['png'] or r_['svg'] for r_ in rendered['rows']],
                                     errors=rendered['errors'])
        except Exception as exc:  # noqa: BLE001 - the counts stand without the renders
            reader['renders'] = dict(error=f'{type(exc).__name__}: {exc}')

    report['t_total_s'] = round(time.time() - t0, 1)
    report['errors'] = table['errors']
    report['mode'] = 'e3-only' if args.e3_only else 'census+e3'
    if not args.e3_only:
        status = census.e2_status(table, holdout='holdout2', hard='hard-holdout',
                                  floors=('fit_dev', 'std-taiko-design'))
        srows = census.strata_rows(results, sealed=lambda r: split_of(r) in SEALED, split_of=split_of)
        census.write_census(out, table, status, srows, meta=dict(run=report, sealed=list(SEALED)))
    (out / 'reader-census.json').write_text(json.dumps(census._jsonable(reader), indent=1) + '\n')
    (out / 'run.json').write_text(json.dumps(census._jsonable(report), indent=1) + '\n')
    with (out / 'e3-sets.jsonl').open('w', encoding='utf-8') as f:
        for row in rows_out:
            f.write(json.dumps(census._jsonable(row)) + '\n')
    with (out / 'same-normal.jsonl').open('w', encoding='utf-8') as f:
        for row in same_rows:
            f.write(json.dumps(census._jsonable(row)) + '\n')
    (out / 'e3.md').write_text(e3_markdown(reader, report), encoding='utf-8')
    print(f'done: membership sha256 {sha}; {len(results)} sets; {table["errors"]} unreadable; E3 '
          f"{e3['all']['reader_changed']}/{e3['all']['recordings']} after, "
          f"{e3['all']['reader_changed_e0']}/{e3['all']['recordings']} before", flush=True)
    log.close()
    return 0

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
