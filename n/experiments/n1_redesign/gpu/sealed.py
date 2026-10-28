"""Sealed-folder rule copied from supervision-census b14d9f5 (identifier metadata only)."""
import csv
import hashlib
import json
import os
from pathlib import Path

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

def build_exclusion(args, root: Path, corpus: list[dict], report: dict) -> set[str]:
    """Set folders (paths relative to the root) linked to any sealed set. No chart is opened here."""
    def shas(path):
        with open(root / path, newline='') as f:
            rows = list(csv.DictReader(f))
        out = {r['audio_sha256'].strip() for r in rows if (r.get('audio_sha256') or '').strip()}
        if not out:
            raise SystemExit(f'{path}: no audio_sha256 values')
        return out

    h1, h2 = shas(args.holdout1), shas(args.holdout2)
    mpath = root / args.hard_holdout / 'membership.json'
    data = mpath.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.hard_holdout_sha256:
        raise SystemExit(f'{mpath}: sha256 does not match the seal; refusing to run')
    members = json.loads(data.decode('utf-8'))['members']
    u = Union()
    seeds: list[str] = []
    for r in corpus:
        g, d = 'g:' + str(r['group_id']), 'd:' + r['set_dir']
        u.union(g, d)
        if r['audio_sha256']:
            u.union(g, 'a:' + r['audio_sha256'])
            if r['audio_sha256'] in h1 or r['audio_sha256'] in h2:
                seeds.append(g)
        if r['eval_split'] == 'heldout':
            seeds.append(g)
    n_dedup_holdout = 0
    for grp in json.loads((root / args.groups).read_text()):
        ids = []
        for m in grp['members']:
            if m.get('sha256'):
                ids.append('a:' + m['sha256'])
            ids += ['g:' + x for x in m.get('corpus_group_ids', []) or []]
            if m.get('path'):
                ids.append('d:' + os.path.dirname(m['path']))
            if m.get('holdout'):
                n_dedup_holdout += 1
                seeds.append(ids[-1])
        for a, b in zip(ids, ids[1:]):
            u.union(a, b)
    n_inv = 0
    with open(root / args.inventory) as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if x.get('path') and x.get('sha256'):
                u.union('d:' + os.path.dirname(x['path']), 'a:' + x['sha256'])
                n_inv += 1
                if x.get('holdout'):
                    seeds.append('a:' + x['sha256'])
    hard_sets = 0
    for m in members:
        for s in m.get('sets', []):
            hard_sets += 1
            if s.get('set_id'):
                seeds.append('d:dataset/std-taiko/' + str(s['set_id']))
            if s.get('audio_sha256'):
                seeds.append('a:' + s['audio_sha256'])
            if s.get('key'):
                parts = str(s['key']).split('@')[0].split('/')
                if len(parts) >= 2:
                    seeds.append('d:dataset/' + '/'.join(parts[:2]))
            for v in s.values():
                for p in (v if isinstance(v, list) else [v]):
                    if isinstance(p, str) and p.startswith('dataset/'):
                        seeds.append('d:' + '/'.join(p.split('/')[:3]))
    sealed_roots = {u.find(x) for x in seeds}
    sealed_dirs = {k[2:] for k in list(u.p) if k.startswith('d:') and u.find(k) in sealed_roots}
    report['exclusion'] = dict(holdout1_shas=len(h1), holdout2_shas=len(h2), dedup_holdout_members=n_dedup_holdout,
                               inventory_rows=n_inv, hard_holdout_recordings=len(members), hard_holdout_sets=hard_sets,
                               corpus_heldout_rows=sum(r['eval_split'] == 'heldout' for r in corpus),
                               sealed_folders=len(sealed_dirs),
                               sha256=dict(membership=args.hard_holdout_sha256))
    if not (h1 and h2 and members and report['exclusion']['corpus_heldout_rows']):
        raise SystemExit('an exclusion list is empty; refusing to run')
    return sealed_dirs
