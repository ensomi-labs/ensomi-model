"""The training manifest: recordings, their reference set and their split role (design §3.3).

A recording is a dedup group of 4K corpus charts, joined by any of:

- the corpus's ``group_id``;
- one audio file (sha256);
- a dedup group of ``music-dedup-20261015/groups.json`` (its 4K members, by sha256
  and corpus group ids).

Roles, in order of precedence (a recording takes the first that applies):

1. ``excluded:heldout``: a chart with ``eval_split = heldout``;
2. ``excluded:holdout``: an audio file of timing holdout-1 or holdout-2 (sha256), or a
   dedup member flagged holdout;
3. ``cal``: a chart with ``eval_split = calibration``;
4. ``fit_dev``: a corpus group that hashes to fit_dev under R2's rule
   (``sha256('r2-fit-dev-v1:' + group_id)``, first 8 bytes big-endian, mod 10 == 0);
   ``panel`` when its audio is in the rung-1 fit_dev panel;
5. ``reset_panel``: a fit_train recording whose audio is in the stored-logit
   phase-reset population (held out of R1-train and val-n1);
6. ``n_train``: everything else with ``eval_split = fit``.

Within a recording the reference audio is the file with the most 4K charts
(ties: the lowest set id, then sha256); its set is those charts. No play counts
are on hand, so "most-played" (design §3.1) is replaced by "most charts". Other
audio files of the recording (rate edits, re-encodes) are not used tonight.
Only ranked and loved charts count.

`select` draws val-n1 and R1-train from ``n_train`` by a salted hash order,
using the interim hard flags (`strata.flags`).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

FIT_DEV_SALT = 'r2-fit-dev-v1:'
STATUSES = ('ranked', 'loved')
CORPUS_COLUMNS = ('path', 'set_dir', 'sha256', 'beatmap_set_id', 'audio_filename', 'audio_present', 'audio_sha256',
                  'api_status', 'group_id', 'eval_split', 'star', 'creator', 'n_red_lines')


def r2_role(group_id: str) -> str:
    digest = hashlib.sha256((FIT_DEV_SALT + group_id).encode('utf-8')).digest()
    return 'fit_dev' if int.from_bytes(digest[:8], 'big') % 10 == 0 else 'fit_train'


def _order(salt: str, key: str) -> str:
    return hashlib.sha256(f'{salt}:{key}'.encode()).hexdigest()


def load_corpus(path) -> list[dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(str(path), columns=list(CORPUS_COLUMNS))
    return table.to_pylist()


def read_csv_shas(path) -> set[str]:
    with open(path, newline='') as handle:
        return {row['audio_sha256'] for row in csv.DictReader(handle)}


class _Union:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_recordings(rows: list[dict], *, groups: list | None = None, holdout_shas: set[str] = frozenset(),
                     panel_audio: set[str] = frozenset(), reset_audio: set[str] = frozenset()) -> list[dict]:
    """One manifest row per recording (see the module docstring)."""
    u = _Union()
    for r in rows:
        if r['audio_sha256']:
            u.union('g:' + r['group_id'], 'a:' + r['audio_sha256'])
        else:
            u.find('g:' + r['group_id'])
    dedup_holdout = set()
    for g in groups or ():
        ids = []
        for m in g['members']:
            if m.get('source') != '4k-corpus':
                continue
            ids.append('a:' + m['sha256'])
            ids += ['g:' + x for x in m.get('corpus_group_ids', [])]
            if m.get('holdout'):
                dedup_holdout.add(m['sha256'])
        for a, b in zip(ids, ids[1:]):
            u.union(a, b)
    by_root: dict[str, list[dict]] = {}
    for r in rows:
        by_root.setdefault(u.find('g:' + r['group_id']), []).append(r)
    out = []
    for root, members in by_root.items():
        splits = {r['eval_split'] for r in members}
        shas = {r['audio_sha256'] for r in members if r['audio_sha256']}
        audio_rels = {f"{r['set_dir']}/{r['audio_filename']}" for r in members}
        usable = [r for r in members if r['api_status'] in STATUSES and r['audio_present'] and r['audio_sha256']]
        if 'heldout' in splits:
            role = 'excluded:heldout'
        elif shas & (set(holdout_shas) | dedup_holdout):
            role = 'excluded:holdout'
        elif 'calibration' in splits:
            role = 'cal'
        elif any(r2_role(r['group_id']) == 'fit_dev' for r in members):
            role = 'panel' if audio_rels & set(panel_audio) else 'fit_dev'
        elif audio_rels & set(reset_audio):
            role = 'reset_panel'
        else:
            role = 'n_train'
        if not usable:
            role = role if role.startswith('excluded') else 'excluded:no_usable_chart'
        ref = None
        if usable:
            per_audio: dict[str, list[dict]] = {}
            for r in usable:
                per_audio.setdefault(r['audio_sha256'], []).append(r)
            best = min(per_audio.items(), key=lambda kv: (-len(kv[1]), min(x['beatmap_set_id'] for x in kv[1]), kv[0]))
            sha, charts = best
            set_ids = sorted({c['beatmap_set_id'] for c in charts})
            set_id = max(set_ids, key=lambda s: (sum(c['beatmap_set_id'] == s for c in charts), -s))
            charts = sorted((c for c in charts if c['beatmap_set_id'] == set_id), key=lambda c: c['path'])
            ref = dict(audio_sha256=sha, audio_rel=f"{charts[0]['set_dir']}/{charts[0]['audio_filename']}",
                       set_dir=charts[0]['set_dir'], set_id=int(set_id), charts=[c['path'] for c in charts],
                       status=sorted({c['api_status'] for c in charts}),
                       creators=sorted({c['creator'] for c in charts}))
        rec_id = ref['audio_sha256'] if ref else 'none:' + root[2:]
        out.append(dict(rec=rec_id, role=role, groups=sorted({r['group_id'] for r in members}),
                        n_audio=len(shas), n_charts=len(members), splits=sorted(splits),
                        audio_rels=sorted(audio_rels), **(ref or {})))
    out.sort(key=lambda r: r['rec'])
    return out


def select(recs: list[dict], hard: dict[str, bool], *, excluded: set[str] = frozenset(), n_val_hard: int = 150,
           n_val_random: int = 150, n_train: int = 1600, n_train_hard: int = 800, salt: str = 'n1-r1-v1',
           version: str = 'n1-r1-v2') -> dict:
    """val-n1 (hard then random) and R1-train (up to ``n_train_hard`` hard, the rest random) from ``n_train``.

    ``hard`` maps recording id to its interim hard flag; recordings without
    labels or in ``excluded`` (failed label checks) are not drawn. The hard
    draws and each random draw use their own salted order: drawn in one order,
    the first recordings not taken by the hard draw are exactly the non-hard
    ones, so a "random" half would hold no hard recording (version
    ``n1-r1-v1`` did this). Version ``n1-r1-v2`` keeps v1's order (``salt``)
    for the hard draws and orders each random draw by ``version``.
    """
    order = lambda tag: (lambda k: _order(f'{version}:{tag}', k))
    pool = sorted((r['rec'] for r in recs if r['role'] == 'n_train' and r['rec'] in hard and r['rec'] not in excluded),
                  key=lambda k: _order(salt + ':order', k))
    hard_pool = [k for k in pool if hard[k]]
    val_hard = hard_pool[:n_val_hard]
    taken = set(val_hard)
    val_random = [k for k in sorted(pool, key=order('val_random')) if k not in taken][:n_val_random]
    taken |= set(val_random)
    rest = [k for k in pool if k not in taken]
    train_hard = [k for k in rest if hard[k]][:n_train_hard]
    chosen = set(train_hard)
    train_rest = [k for k in sorted(rest, key=order('train_random')) if k not in chosen][:max(n_train - len(train_hard), 0)]
    return dict(val_hard=val_hard, val_random=val_random, train=train_hard + train_rest, train_hard=train_hard,
                pool=len(pool), pool_hard=len(hard_pool), salt=salt, version=version,
                val_random_hard=sum(hard[k] for k in val_random), train_rest_hard=sum(hard[k] for k in train_rest))


def write_json(obj, path) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
