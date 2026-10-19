"""N1 v0 curated training data and evaluation items (mac side; labels only, plus audio hashes).

Sources and roles:

- **4K N-train**: rung 0's manifest (``recordings.json``) roles ``n_train`` and ``reset_panel``
  (the reset panel was held out only for rung 1's probe, which was cancelled), minus val-n1's 300
  recordings. One reference set per recording, as the manifest chose it.
- **std-taiko**: the census units (``scripts/n1/census_run.build_units``) of split ``std-taiko``
  (no link to the 4K corpus), minus every set id, census key and audio sha256 of the sealed hard
  holdout (`ensomi_n.eval.census.load_exclusion`, which refuses a changed seal). Linked sets
  (``std-taiko-linked*``) are left out whole. Per recording the reference is the set key with the
  most `.osu` files (std before taiko on ties).
- Excluded by construction: holdout-1, holdout-2, the R2 heldout, the sealed hard holdout and its
  linked sets, all of fit_dev (the panel included), cal, and val-n1.
- **Evaluation items**: the fit_dev panel (322 songs), val-n1 hard and random (rung 0's
  ``split.json``), and a 200-recording cal subset for the decoder's settings.

Quality filter on training candidates (reasons recorded, every failing reason kept):

- ``read``: the set reads and has 16 or more canonical lines over 30 s or more;
- ``status``: ranked, approved or loved (4K: guaranteed by the manifest);
- ``grid``: 0.95 or more of heads within 2 ms of the standard subdivisions (``checks.on_grid``).
  Rung 0 found 13 % of 4K sets under 0.99, mostly technical charts whose off-grid heads sit a
  median 4.3 ms from the grid in one difficulty; under 0.95 the misses grow to a fifth to a half
  of the heads across difficulties, which is red lines not explaining the notes;
- ``inconsistent``: difficulties with different red lines;
- ``expressive``: an implausible (scroll-art) line, or a segment shorter than a beat holding no
  heads (barline art);
- ``reader``: `ensomi_n.data.strata.reader_flags` (redundant lines, fast odd bars, fast tempo
  lines), interim until the reader fix lands. The coordinator's literal rule (any segment notated
  at 160 BPM or more, or a redundant line) is recorded as ``literal`` for the counts.

Weights: hard strata (census `ensomi_n.eval.strata.flags`) up: 1 for S0 or none, 1.5 when only S1
or S6 hold, 3 when any of S2, S3, S4, S5, S7, S8, S9 holds; times 1.5 when a second mapper's set on
the same audio (no shared creator) agrees (BL-20 >= 0.9), 0.5 when it disagrees (< 0.6).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ON_GRID_MIN = 0.95
MIN_LINES = 16
MIN_SPAN_MS = 30000.0
STATUSES = ('ranked', 'approved', 'loved')
STRONG = ('S2', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9')
MILD = ('S1', 'S6')
AGREE, DISAGREE = 0.9, 0.6
CAL_N = 200
SEAL = ('artifacts/n1-eval/hard-holdout-v1', '00771c6491ced746bc1dd536c0cd3ef453a01798a658ebe3330ab89edec327bb')


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def strata_weight(strata) -> float:
    s = set(strata or ())
    if s & set(STRONG):
        return 3.0
    if s & set(MILD):
        return 1.5
    return 1.0


def _census_run(src_root: Path):
    path = src_root / 'scripts' / 'n1' / 'census_run.py'
    spec = importlib.util.spec_from_file_location('census_run', path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def read_item(args) -> dict:
    """Read one item's charts; store its v0 labels; return its row (flags, strata, checks)."""
    item, root, labels_dir = args
    from ..data import checks
    from ..data.strata import reader_flags
    from ..eval.strata import flags as census_flags, is_rate_edit
    from ..eval.census import read_header
    from ..labels import read_osu, reference_from_charts
    from ..timing import canonical_beat_ms
    from .labels import V0Labels, literal_flag
    out = dict(item['row'])
    try:
        paths = sorted(Path(root) / c for c in item['charts'])
        paths = [p for p in paths if p.is_file()]
        charts = [read_osu(p) for p in paths]
        ref = reference_from_charts(charts, item['row']['item'])
        if item.get('hash_audio'):
            ap = Path(root) / item['row']['audio_rel']
            out['audio_sha256'] = sha256_file(ap)
            out['item'] = out['audio_sha256']
        lab = V0Labels.from_reference(ref, out['item'])
        if item.get('store', True):
            p = Path(labels_dir) / out['item'][:2] / f"{out['item']}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f'.tmp{os.getpid()}')
            tmp.write_text(json.dumps(lab.to_json()))
            os.replace(tmp, p)
        headers = [read_header(p) for p in paths]
        rate = is_rate_edit(*[h.get('Version', '') for h in headers], ref.meta.get('audio', ''))
        cf = census_flags(ref, charts, tags=item.get('tags', ()), tags_known=bool(item.get('tags_known')))
        ex = checks.expressive(ref)
        in_range = (lab.lines >= ref.lo_ms) & (lab.lines <= ref.hi_ms)
        lv = np.bincount(lab.line_level[in_range], minlength=4) if in_range.any() else np.zeros(4)
        segs = [s for s in (ref.segments or ()) if s.plausible]
        out.update(ok=True, on_grid=round(float(checks.on_grid(ref)), 5), expressive=ex,
                   timing_consistent=bool(ref.meta.get('timing_consistent')), reader_flags=reader_flags(ref),
                   literal=bool(literal_flag(ref)), strata=cf['strata'], hard=bool(cf['hard']),
                   census_flags={k: v for k, v in cf['flags'].items() if v}, rate_edit=bool(rate),
                   n_lines=int(in_range.sum()), span_ms=float(ref.hi_ms - ref.lo_ms), n_heads=int(len(lab.heads)),
                   n_bounds=int(len(lab.bounds)), n_resets=int((lab.bound_kind == 1).sum()),
                   level_lines=[int(x) for x in lv], n_red=len(ref.segments or ()),
                   bpm_notated=sorted({round(s.bpm, 3) for s in segs})[:8],
                   creators=sorted({h.get('Creator', '').casefold() for h in headers if h.get('Creator')}))
    except Exception as exc:  # noqa: BLE001 - recorded per item
        out.update(ok=False, error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-600:])
    return out


def agreement(args) -> dict:
    """BL-20 of each other set's timing (same audio, no shared creator) against the reference."""
    key, root, ref_charts, others = args
    from ..eval import score
    from ..labels import read_osu, reference_from_charts
    try:
        ref = reference_from_charts([read_osu(Path(root) / c) for c in sorted(ref_charts)], key)
        vals = []
        for charts in others:
            o = reference_from_charts([read_osu(Path(root) / c) for c in sorted(charts)], key + ':other')
            vals.append(float(score(ref, o.timing).bl20))
        return dict(item=key, bl20=vals)
    except Exception as exc:  # noqa: BLE001
        return dict(item=key, error=f'{type(exc).__name__}: {exc}')


def _pool_map(fn, args, processes, progress=None, label=''):
    import multiprocessing as mp
    rows = []
    t0 = time.time()
    with mp.get_context('spawn').Pool(processes) as pool:
        for k, r in enumerate(pool.imap(fn, args, chunksize=4), 1):
            rows.append(r)
            if progress is not None and (k % 250 == 0 or k == len(args)):
                progress(dict(step=label, done=k, total=len(args), wall_s=round(time.time() - t0, 1)))
    return rows


def _order(salt: str, key: str) -> str:
    return hashlib.sha256(f'{salt}:{key}'.encode()).hexdigest()


def build(*, root: Path, src_root: Path, r0: Path, run: Path, labels_dir: Path, processes: int = 8,
          progress=None, corpus='data/r2-corpus.parquet', groups='artifacts/n1-night-20261018/inputs/groups.json',
          holdout1='artifacts/timing-20261009/opt/holdout/holdout.csv',
          holdout2='artifacts/timing-20261009/opt/ft/pop/holdout2.csv',
          tags='artifacts/n1-inventory-20261014/tags.json') -> dict:
    from ..eval.census import load_exclusion, _audio_path
    import pyarrow.parquet as pq
    cr = _census_run(src_root)
    ns = type('A', (), dict(root=str(root), corpus=corpus, groups=groups, holdout1=holdout1, holdout2=holdout2,
                            tags=tags))()
    report: dict = {}
    units = cr.build_units(ns, report)
    excl = load_exclusion(root / SEAL[0], SEAL[1])
    recs = json.load(open(r0 / 'recordings.json'))
    split = json.load(open(r0 / 'split.json'))
    corpus_rows = pq.read_table(str(root / corpus), columns=['path', 'audio_sha256', 'beatmap_set_id', 'creator',
                                                             'api_status']).to_pylist()
    unit_of_file = {}
    for u in units:
        for f in u.files:
            unit_of_file[f] = u
    val_recs = {it['rec'] for it in split['val_hard'] + split['val_random']}
    items = []

    def tags_for(charts):
        for c in charts:
            u = unit_of_file.get(c)
            if u is not None:
                return u.tags, u.tags_known
        return [], False

    for r in recs:
        if r['role'] not in ('n_train', 'reset_panel') or not r.get('charts') or r['rec'] in val_recs:
            continue
        t, known = tags_for(r['charts'])
        items.append(dict(charts=r['charts'], tags=t, tags_known=known,
                          row=dict(item=r['audio_sha256'], audio_sha256=r['audio_sha256'], audio_rel=r['audio_rel'],
                                   rec=r['rec'], role='train', source='4k', set_id=r.get('set_id'),
                                   status=','.join(r.get('status', [])), manifest_role=r['role'])))
    # std-taiko: the census pool minus the seal, one reference set per recording
    pool: dict[str, list] = {}
    dropped = dict(seal_key=0, seal_set_id=0, linked=0, seal_audio=0)
    for u in units:
        if u.source != 'std-taiko':
            continue
        if u.split != 'std-taiko':
            dropped['linked'] += 1
            continue
        if u.key in excl['keys']:
            dropped['seal_key'] += 1
            continue
        if u.set_id and u.set_id in excl['set_ids']:
            dropped['seal_set_id'] += 1
            continue
        pool.setdefault(u.recording, []).append(u)
    for rec, us in sorted(pool.items()):
        u = max(us, key=lambda x: (len(x.files), x.mode == 'std', x.key))
        audio = u.key  # placeholder; resolved below
        ap = None
        try:
            hdr = cr.osu_header(root / u.files[0])
            ap = _audio_path(root, u.files, hdr.get('AudioFilename', ''))
        except Exception:  # noqa: BLE001
            ap = None
        if ap is None:
            continue
        items.append(dict(charts=u.files, tags=u.tags, tags_known=u.tags_known, hash_audio=True,
                          row=dict(item=u.key, audio_rel=str(Path(ap).relative_to(root)), rec=rec, role='train',
                                   source=u.mode, set_id=u.set_id, status=u.status, census_key=u.key)))
    # evaluation items (labels read for flags and strata; the panel's items are not stored by sha: two
    # panel songs can share one audio file)
    for name in ('panel', 'val_hard', 'val_random'):
        for it in split[name]:
            t, known = tags_for(it['charts'])
            items.append(dict(charts=it['charts'], tags=t, tags_known=known, store=(name != 'panel'),
                              row=dict(item=it['audio_sha256'], audio_sha256=it['audio_sha256'],
                                       audio_rel=it['audio_rel'], rec=it['rec'], role=name, source='4k',
                                       bt_key=it.get('bt_key'), eval_key=f"{name}:{it['audio_sha256']}:"
                                       f"{hashlib.sha1(','.join(sorted(it['charts'])).encode()).hexdigest()[:8]}")))
    cal = sorted((r for r in recs if r['role'] == 'cal' and r.get('charts')), key=lambda r: _order('n1-v0-cal', r['rec']))
    for r in cal[:CAL_N]:
        t, known = tags_for(r['charts'])
        items.append(dict(charts=r['charts'], tags=t, tags_known=known,
                          row=dict(item=r['audio_sha256'], audio_sha256=r['audio_sha256'], audio_rel=r['audio_rel'],
                                   rec=r['rec'], role='cal', source='4k')))
    if progress:
        progress(dict(step='curate_items', n=len(items), dropped=dropped))
    rows = _pool_map(read_item, [(it, str(root), str(labels_dir)) for it in items], processes, progress, 'curate_read')
    # std-taiko audio shas against the seal
    for r in rows:
        if r.get('source') in ('std', 'taiko') and r.get('audio_sha256') in excl['audio_sha256']:
            r['seal_audio'] = True
            dropped['seal_audio'] += 1
    # second mappers (4K training candidates)
    by_audio: dict[str, dict[int, list]] = {}
    for c in corpus_rows:
        if c['api_status'] in ('ranked', 'loved') and c['audio_sha256']:
            by_audio.setdefault(c['audio_sha256'], {}).setdefault(int(c['beatmap_set_id'] or -1), []).append(c)
    ag_args = []
    charts_of = {it['row']['item']: it['charts'] for it in items if it['row']['role'] == 'train'}
    for r in rows:
        if r.get('role') != 'train' or r.get('source') != '4k' or not r.get('ok'):
            continue
        sets = by_audio.get(r['audio_sha256'], {})
        mine = set(r.get('creators', []))
        others = [[c['path'] for c in cs] for sid, cs in sets.items()
                  if sid != int(r.get('set_id') or -2) and not ({(c['creator'] or '').casefold() for c in cs} & mine)]
        if others:
            ag_args.append((r['item'], str(root), charts_of[r['item']], others[:3]))
    ag = {a['item']: a for a in _pool_map(agreement, ag_args, processes, progress, 'curate_agree')} if ag_args else {}
    # decisions
    for r in rows:
        if r.get('role') != 'train':
            r['reasons'] = []
            continue
        why = []
        if not r.get('ok') or r.get('n_lines', 0) < MIN_LINES or r.get('span_ms', 0) < MIN_SPAN_MS:
            why.append('read')
        if r.get('ok'):
            st = {s.strip().lower() for s in str(r.get('status') or '').split(',') if s.strip()}
            if r['source'] != '4k' and not (st & set(STATUSES)):
                why.append('status')
            if not (r['on_grid'] >= ON_GRID_MIN):
                why.append('grid')
            if not r['timing_consistent']:
                why.append('inconsistent')
            if r['expressive']['implausible'] or r['expressive']['tiny']:
                why.append('expressive')
            if r['reader_flags']:
                why.append('reader')
            if r.get('rate_edit'):
                why.append('rate_edit')
            if r.get('seal_audio'):
                why.append('seal_audio')
        r['reasons'] = why
        a = ag.get(r['item'])
        r['second_bl20'] = a.get('bl20') if a else None
        w = strata_weight(r.get('strata'))
        if a and a.get('bl20'):
            best = max(a['bl20'])
            w *= 1.5 if best >= AGREE else (0.5 if best < DISAGREE else 1.0)
        r['weight'] = w
        r['keep'] = not why
    # one std-taiko reference per audio sha (two set dirs can share audio outside a dedup group)
    seen = set()
    for r in rows:
        if r.get('role') == 'train' and r.get('keep'):
            if r['item'] in seen:
                r['keep'] = False
                r['reasons'] = ['duplicate_audio']
            seen.add(r['item'])
    return dict(rows=rows, dropped=dropped, report_inputs=report.get('inputs'), seal_sha256=SEAL[1])


def summarise(res: dict) -> dict:
    from collections import Counter
    rows = res['rows']
    tr = [r for r in rows if r.get('role') == 'train']
    kept = [r for r in tr if r.get('keep')]
    out = dict(seal_sha256=res['seal_sha256'], std_taiko_dropped=res['dropped'],
               candidates=dict(Counter(r['source'] for r in tr)), kept=dict(Counter(r['source'] for r in kept)))
    out['excluded_any_reason'] = {src: dict(Counter(x for r in tr if r['source'] == src for x in r['reasons']))
                                  for src in sorted({r['source'] for r in tr})}
    out['excluded_only_reason'] = {src: dict(Counter(r['reasons'][0] for r in tr if r['source'] == src
                                                     and len(r['reasons']) == 1)) for src in sorted({r['source'] for r in tr})}
    ok = [r for r in tr if r.get('ok')]
    out['literal_rule_would_drop'] = {src: int(sum(r['literal'] for r in ok if r['source'] == src))
                                      for src in sorted({r['source'] for r in ok})}
    out['reader_rule_drops'] = {src: int(sum(bool(r['reader_flags']) for r in ok if r['source'] == src))
                                for src in sorted({r['source'] for r in ok})}
    strata = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9')
    table = {}
    for s in strata + ('hard', 'all'):
        row = {}
        for src in sorted({r['source'] for r in tr}):
            def has(r):
                return True if s == 'all' else (r.get('hard') if s == 'hard' else s in (r.get('strata') or []))
            row[src] = dict(candidates=sum(1 for r in ok if r['source'] == src and has(r)),
                            kept=sum(1 for r in kept if r['source'] == src and has(r)),
                            kept_weight=round(sum(r['weight'] for r in kept if r['source'] == src and has(r)), 1))
        table[s] = row
    out['per_stratum'] = table
    out['second_mapper'] = dict(with_pair=sum(1 for r in kept if r.get('second_bl20')),
                                agree=sum(1 for r in kept if r.get('second_bl20') and max(r['second_bl20']) >= AGREE),
                                disagree=sum(1 for r in kept if r.get('second_bl20') and max(r['second_bl20']) < DISAGREE))
    ev = [r for r in rows if r.get('role') in ('panel', 'val_hard', 'val_random', 'cal')]
    out['eval'] = {name: dict(n=sum(1 for r in ev if r['role'] == name),
                              ok=sum(1 for r in ev if r['role'] == name and r.get('ok')),
                              reader_flagged=sum(1 for r in ev if r['role'] == name and r.get('reader_flags')),
                              literal=sum(1 for r in ev if r['role'] == name and r.get('literal')))
                   for name in ('panel', 'val_hard', 'val_random', 'cal')}
    out['kept_hours_estimate'] = round(sum(r['span_ms'] for r in kept) / 3.6e6, 1)
    return out
