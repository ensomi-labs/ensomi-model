"""Labels for the manifest's recordings and panels: read, checked, flagged and stored (R0 step c).

An *item* is one audio file with its reference set: the charts of that audio
in one beatmap set, read by `ensomi_n.labels.read_set`. Training and val-n1
use each recording's reference audio; the panels use their own audio lists
(the fit_dev panel's 322 songs from the fine-tune acceptance, and the stored-
logit phase-reset songs), with the recording as the bootstrap cluster.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import time
import traceback

from . import checks, manifest as M, strata
from ..targets import RecLabels

READER_VERSION = 'labels-e0-0898dc0'


def item_key(audio_sha: str) -> str:
    return audio_sha


def panel_items(recs: list[dict], rows: list[dict], panel_json, reset_csv) -> list[dict]:
    """Items of the fit_dev panel and the reset panel, each tagged with its recording."""
    by_audio = {}
    for r in recs:
        for a in r['audio_rels']:
            by_audio[a] = r
    sha_of = {}
    charts_of: dict[str, list[str]] = {}
    for row in rows:
        a = f"{row['set_dir']}/{row['audio_filename']}"
        if row['audio_sha256']:
            sha_of[a] = row['audio_sha256']
        if row['api_status'] in M.STATUSES:
            charts_of.setdefault(a, []).append(row['path'])
    items = []
    for p in json.load(open(panel_json)):
        a = p['audio_rel']
        rec = by_audio.get(a)
        items.append(dict(item=item_key(sha_of[a]), audio_rel=a, audio_sha256=sha_of[a],
                          charts=sorted(c['path'] for c in p['charts']), rec=rec['rec'] if rec else a,
                          panel='fit_dev', bt_key=p['mel_key'], pop_stratum=p.get('stratum')))
    with open(reset_csv, newline='') as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            a = row['audio_rel']
            rec = by_audio.get(a)
            if rec is None or rec['role'] != 'reset_panel' or a not in sha_of:
                continue
            items.append(dict(item=item_key(sha_of[a]), audio_rel=a, audio_sha256=sha_of[a],
                              charts=sorted(charts_of.get(a, [])), rec=rec['rec'], panel='reset',
                              bt_index=int(row['i'])))
    return items


def recording_items(recs: list[dict], roles=('n_train',)) -> list[dict]:
    return [dict(item=item_key(r['audio_sha256']), audio_rel=r['audio_rel'], audio_sha256=r['audio_sha256'],
                 charts=r['charts'], rec=r['rec'], panel='') for r in recs if r['role'] in roles and r.get('charts')]


def read_item(args) -> dict:
    """Read one item's reference; store its labels; return its summary row."""
    item, root, labels_dir = args
    from ..labels import read_set
    out = dict(item=item['item'], rec=item['rec'], panel=item.get('panel', ''))
    try:
        ref = read_set([os.path.join(root, c) for c in item['charts']], key=item['item'])
        lab = RecLabels.from_reference(ref)
        path = Path(labels_dir) / item['item'][:2] / f"{item['item']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.tmp{os.getpid()}')
        tmp.write_text(json.dumps(lab.to_json()))
        os.replace(tmp, path)
        fl = strata.flags(ref)
        ex = checks.expressive(ref)
        segs = strata.musical_segments(ref.segments or ())
        out.update(ok=True, on_grid=round(checks.on_grid(ref), 5), n_heads=int(len(lab.heads)),
                   n_lines=int(len(lab.lines)), n_bars=int(len(lab.bars)), lo_ms=lab.lo_ms, hi_ms=lab.hi_ms,
                   n_resets=sum(s["reset"] for s in segs),
                   meters=sorted({int(s.meter) for s in (ref.segments or ())}), expressive=ex,
                   timing_consistent=bool(ref.meta.get('timing_consistent')), **fl)
    except Exception as exc:
        out.update(ok=False, error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-600:])
    return out


def read_all(items: list[dict], root, labels_dir, processes: int = 4, progress=None) -> list[dict]:
    import multiprocessing as mp
    args = [(it, str(root), str(labels_dir)) for it in items]
    rows = []
    t0 = time.time()
    if processes <= 1:
        it = map(read_item, args)
        pool = None
    else:
        pool = mp.get_context('spawn').Pool(processes)
        it = pool.imap(read_item, args, chunksize=8)
    try:
        for k, row in enumerate(it, 1):
            rows.append(row)
            if progress is not None and (k % 200 == 0 or k == len(args)):
                progress(dict(step='labels', done=k, total=len(args), wall_s=round(time.time() - t0, 1)))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return rows


def load_labels(labels_dir, item: str) -> RecLabels:
    return RecLabels.from_json(json.loads((Path(labels_dir) / item[:2] / f'{item}.json').read_text()))
