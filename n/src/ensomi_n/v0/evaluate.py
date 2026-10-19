"""N1 v0 evaluation through `ensomi_n.eval`, per stratum, on the fit_dev panel and val-n1 hard.

Arms (each a music map per item, scored by `ensomi_n.eval.score` against the item's reference):

- ``v0@<ckpt>``: v0's evidence decoded (`v0.decode`), for each of the last checkpoints (their spread is
  the checkpoint noise of every statistic); ``v0f1@<ckpt>`` decodes the F1 heads alone (no phase or period
  input); ``v0zero``/``v0other`` decode with the song memory skipped or taken from another song (the
  input-use check, guardrail 3);
- ``final0``/``final1``/``final2``: BeatThis logits (sigmoid, shifted earlier by 30.0/31.9/31.3 ms, fixed,
  not fitted) through the same decoder, no level input (notated level = canonical); ``final1``/``final2``
  exist for the panel only;
- ``r2``: R2 as exported (rung 0's ``r2-export``: canonical red lines, meter 4);
- ``oracle``: the reference's own canonical lines, bar starts, level and boundaries rendered as evidence
  through the decoder (the decoder's ceiling; it reads the reference, so it never competes).

Every readable item is scored with the fixed reader (`n1/eval-harness` `cb41e0f`). Until the E3 recount
confirms the fix on the corpus, every table is also given without the coordinator's literal-rule sets (any
segment notated at 160 BPM or more, or a redundant same-tempo line): ``<arm>|literal-safe``. BL-20 is
split into non-free lines (the main number, ``bl20_nonfree``) and the lines of free bars (``bl20_free``,
bars that hold between one and two canonical beats, compared at the refined level), with their weight
share; whether BL-20 decides in free spans is open for the human. Strata are the census's (`ensomi_n.eval.strata.set_strata` with the corpus's voted
tags, combined per recording by `recording_strata`), with their spans.

Outputs under ``--out`` (default ``<run>/eval``): ``tables-<panel>.json`` (per arm, per stratum and per
``stratum|ranked/loved``: `ensomi_n.eval.report.build_strata` with 1,000 recording resamples),
``paired-<panel>.json`` (v0 minus R2 and minus each BeatThis arm, per stratum, paired recording bootstrap),
``items-<panel>.jsonl`` (per item and arm: the scalar summary) and ``ablation.json`` (teacher-forced loss
with the memory zeroed or swapped, on val-n1 crops, with recording-bootstrap intervals).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

BT_SHIFT_MS = dict(final0=30.0, final1=31.9, final2=31.3)
BOOT = 1000
MEASURES = ('bl20', 'bl10', 'bl5', 'bl20_beat', 'chance20', 'level', 'bars_start', 'bars_same', 'bars_precision',
            'boundary_recall', 'boundary_precision', 'reset_recall', 'edits_per_min')


# ------------------------------------------------------------------ items and references

def _tag_lookup(root: Path, corpus: str, tags: str):
    import pyarrow.parquet as pq
    names = {}
    tp = root / tags
    if tp.is_file():
        data = json.loads(tp.read_text())
        if isinstance(data, dict) and 'tags' in data:
            data = data['tags']
        if isinstance(data, dict):
            names = {int(k): (v['name'] if isinstance(v, dict) else str(v)) for k, v in data.items()}
        else:
            names = {int(t['id'] if 'id' in t else t['tag_id']): t['name'] for t in data}
    rows = pq.read_table(str(root / corpus), columns=['path', 'api_tag_ids', 'api_status']).to_pylist()
    return {r['path']: ([names.get(int(t), str(t)) for t in (r['api_tag_ids'] or [])], r['api_status']) for r in rows}, bool(names)


def eval_items(split: dict, panels, root: Path, corpus='data/r2-corpus.parquet',
               tags='artifacts/n1-inventory-20261014/tags.json') -> list[dict]:
    """Items of the named panels with reference, strata spans per recording, flags and status."""
    from ..data.strata import reader_flags
    from ..eval.census import read_header
    from ..eval.strata import is_rate_edit, recording_strata, set_strata
    from ..labels import read_osu, reference_from_charts
    from .labels import literal_flag
    tag_of, known = _tag_lookup(root, corpus, tags)
    items = []
    for name in panels:
        for it in split[name]:
            key = f"{name}:{it['audio_sha256']}:{hashlib.sha1(','.join(sorted(it['charts'])).encode()).hexdigest()[:8]}"
            paths = sorted(root / c for c in it['charts'])
            try:
                charts = [read_osu(p) for p in paths]
                ref = reference_from_charts(charts, key)
                tg = sorted({t for c in it['charts'] for t in tag_of.get(c, ([], ''))[0]})
                status = sorted({tag_of.get(c, ([], ''))[1] for c in it['charts']} - {''})
                rate = is_rate_edit(*[read_header(p).get('Version', '') for p in paths], ref.meta.get('audio', ''))
                st = set_strata(ref, charts, tags=tg, tags_known=known, rate_edit=rate, key=key)
                items.append(dict(key=key, panel=name, item=it['audio_sha256'], rec=it['rec'], ref=ref, st=st,
                                  reader_flags=reader_flags(ref), literal=bool(literal_flag(ref)),
                                  source='loved' if status == ['loved'] else 'ranked', charts=it['charts'],
                                  bt_key=it.get('bt_key')))
            except Exception as exc:  # noqa: BLE001
                items.append(dict(key=key, panel=name, item=it['audio_sha256'], rec=it['rec'], error=str(exc)))
    by_rec: dict[str, list] = {}
    for x in items:
        if 'st' in x:
            by_rec.setdefault((x['panel'], x['rec']), []).append(x)
    for xs in by_rec.values():
        rs = recording_strata([x['st'] for x in xs])
        for x in xs:
            x['strata'] = {name: spans[x['key']] for name, spans in rs.items() if x['key'] in spans}
    return items


# ------------------------------------------------------------------ evidence

def bt_evidence(path: Path, shift_ms: float, n_frames: int):
    from .decode import Evidence
    z = np.load(path)
    t = (10.0 * np.arange(n_frames) + shift_ms) / 20.0

    def interp(x):
        x = 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
        return np.interp(t, np.arange(len(x)), x, left=x[0] if len(x) else 0.0, right=x[-1] if len(x) else 0.0)
    return Evidence(beat=interp(z['beat']), down=interp(z['down']))


def oracle_evidence(ref, n_frames: int):
    from .decode import impulse_evidence
    from .labels import V0Labels
    lab = V0Labels.from_reference(ref)
    lines, lev = lab.lines, lab.line_level
    ratio = np.array([0.5, 1.0, 2.0, 4.0])
    t = 10.0 * np.arange(n_frames)
    if len(lines):
        k = np.clip(np.searchsorted(lines, t, side='right') - 1, 0, len(lines) - 1)
        r = ratio[lev[k]]
    else:
        r = np.ones(n_frames)
    return impulse_evidence(n_frames, lines, lab.bars, r, lab.bounds)


def v0_evidence(acts: dict, f1_only: bool = False):
    from .decode import Evidence
    kw = dict(beat=acts['beat'], beat_off=acts['beat_off'], down=acts['down'], half=acts['half'],
              level=acts['level'], boundary=acts['bound'])
    if not f1_only:
        kw.update(phase=acts['phase'], kappa=acts['kappa'], logp=acts['logp'], logp_sigma=acts['logp_sigma'])
    return Evidence(**{k: np.asarray(v, dtype=np.float32) for k, v in kw.items()})


def r2_map(path: Path):
    from ..mapformat import MusicMap
    from ..timing import Segment
    d = json.loads(Path(path).read_text())
    segs, last = [], -math.inf
    for off, length, meter in d['red_lines']:
        if off > last:
            segs.append(Segment(float(off), float(length), int(meter)))
            last = off
    return MusicMap(segments=tuple(segs), model={'name': 'r2-as-exported'})


# ------------------------------------------------------------------ scoring

def free_split(ref, mmap, spans=None, tau: float = 20.0) -> dict:
    """BL-20 on non-free and on free (refined) reference lines, and the free weight share.

    Recomputes the comparison of `ensomi_n.eval.measures.score` (``compared_lines``, ``nearest`` and the
    head weights, restricted to ``spans`` as ``score`` does) to keep the per-line refinement factor that
    ``SetScore`` does not carry. A line is free when either side's bar refines it.
    """
    from ..eval.measures import _Region, _prediction, compared_lines, nearest, share
    from ..timing import Reference
    p = _prediction(mmap, ref.lo_ms, ref.hi_ms, None)
    lines, pred_lines, _ = compared_lines(ref.lines(), p.timing.canonical(), ref.timing.exact_steps,
                                          p.timing.exact_steps, ref_all=ref.timing.canonical(),
                                          ref_end=float(ref.timing.beats[-1]) if len(ref.timing) else -math.inf,
                                          pred_end=float(p.timing.beats[-1]) if len(p.timing) else -math.inf)
    e, _ = nearest(lines.t, pred_lines.t)
    region = _Region(ref.lo_ms, ref.hi_ms, spans)
    if region.whole:
        w = ref.weights(lines)
    else:
        h = ref.heads
        sub = Reference(ref.timing, ref.lo_ms, ref.hi_ms, None if h is None else h[region.contains(h)], ref.segments,
                        ref.unscorable, ref.key)
        w = sub.weights(lines)
        if h is None:
            w = w * region.contains(lines.t)
    ok = np.isfinite(e) & (np.abs(e) <= tau + 1e-9)
    free = lines.factor > 1
    return dict(bl20_nonfree=share(ok[~free], w[~free]) if (w[~free] > 0).any() else math.nan,
                bl20_free=share(ok[free], w[free]) if (w[free] > 0).any() else math.nan,
                free_w=float(share(free, w)) if w.sum() > 0 else math.nan)


def score_item(ref, strata: dict, mmap):
    """{'all': SetScore, stratum: SetScore (on its spans), 'free': {name: free_split}}."""
    from ..eval import score
    out = {'all': score(ref, mmap)}
    fr = {'all': free_split(ref, mmap)}
    for name, spans in strata.items():
        out[name] = out['all'] if spans is None else score(ref, mmap, spans=spans)
        fr[name] = fr['all'] if spans is None else free_split(ref, mmap, spans)
    out['free'] = fr
    return out


def free_table(items: list[dict], scores: dict, n: int = BOOT, seed: int = 0) -> dict:
    """Per stratum: recording means of bl20_nonfree and bl20_free with recording-bootstrap intervals."""
    rng = np.random.default_rng(seed)
    names = sorted({k for x in items for k in x['strata']} | {'all'})
    out = {}
    for name in names:
        row = {}
        for m in ('bl20_nonfree', 'bl20_free', 'free_w'):
            per = {}
            for x in items:
                s = scores.get(x['key'])
                if s is None or (name != 'all' and name not in x['strata']):
                    continue
                v = s['free'][name][m]
                if math.isfinite(v):
                    per.setdefault(x['rec'], []).append(v)
            if not per:
                continue
            d = np.array([np.mean(v) for v in per.values()])
            boot = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
            row[m] = dict(value=float(d.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                          n=len(d))
        out[name] = row
    return out


def _decode_job(args):
    """Decode (unless given a map) and, with a reference, score in the worker: (arm, key, result, diag, error).

    The result is the map when no reference is given, else `score_item`'s dict with the per-line arrays
    dropped (the aggregation reads only summaries and pooled counts).
    """
    arm, key, ev, cfg_kw = args[:4]
    ref, strata = (args[4], args[5]) if len(args) > 5 else (None, None)
    from ..mapformat import MusicMap
    from .decode import DecodeConfig, decode
    try:
        t0 = time.perf_counter()
        if isinstance(ev, MusicMap):
            mmap, diag = ev, {}
        else:
            mmap, diag = decode(ev, DecodeConfig(**cfg_kw), model=arm)
            diag = dict(diag, decode_s=round(time.perf_counter() - t0, 3))
        if ref is None:
            return arm, key, mmap, diag, None
        sc = score_item(ref, strata, mmap)
        for k, v in sc.items():
            if k != 'free':
                v.beats = {}
        return arm, key, sc, diag, None
    except Exception as exc:  # noqa: BLE001
        return arm, key, None, None, f'{type(exc).__name__}: {exc} {traceback.format_exc()[-300:]}'


def aggregate_arm(items: list[dict], scores: dict, bootstrap: int = BOOT) -> dict:
    from ..eval.report import Entry, build_strata
    entries, by_span = [], {}
    for x in items:
        s = scores.get(x['key'])
        if s is None:
            continue
        strata = {k: v for k, v in x['strata'].items()}
        entries.append(Entry(s['all'], x['rec'], x['source'], None, strata))
        for name in strata:
            by_span[(s['all'].key, name)] = s[name]
    return build_strata(entries, by_span, bootstrap=bootstrap)


def paired(items: list[dict], a: dict, b: dict, measure: str = 'bl20', n: int = 2000, seed: int = 0) -> dict:
    """Per stratum: mean over recordings of arm a minus arm b, with a paired recording bootstrap."""
    rng = np.random.default_rng(seed)
    names = sorted({k for x in items for k in x['strata']} | {'all'})
    out = {}
    for name in names:
        per = {}
        for x in items:
            if x['key'] not in a or x['key'] not in b or (name != 'all' and name not in x['strata']):
                continue
            if measure.startswith('bl20_') :
                va, vb = a[x['key']]['free'][name][measure], b[x['key']]['free'][name][measure]
            else:
                va, vb = a[x['key']][name].summary()[measure], b[x['key']][name].summary()[measure]
            if va is None or vb is None or not (math.isfinite(va) and math.isfinite(vb)):
                continue
            per.setdefault(x['rec'], []).append(va - vb)
        if not per:
            continue
        d = np.array([np.mean(v) for v in per.values()])
        boot = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
        out[name] = dict(delta=float(d.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                         n_recordings=len(d))
    return out


def compact(table: dict, measures=MEASURES) -> dict:
    out = {}
    for name, agg in table.items():
        row = dict(n=agg.get('n_recordings', 0))
        for m in measures:
            if m in agg:
                row[m] = agg[m]
        for m in ('offset_median', 'offset_robust_sd', 'offset_within_5', 'offset_within_10', 'offset_within_20', 's_bl20'):
            if m in agg:
                row[m] = agg[m]
        out[name] = row
    return out


# ------------------------------------------------------------------ main

def _paths(a) -> dict:
    return dict(corpus=getattr(a, 'corpus', None) or 'data/r2-corpus.parquet',
                tags=getattr(a, 'tags', None) or 'artifacts/n1-inventory-20261014/tags.json')


def with_cal(split: dict, data: Path, r0: Path) -> dict:
    """Add the curated cal subset (charts from rung 0's manifest) as split ``cal``."""
    recs = {r['rec']: r for r in json.load(open(r0 / 'recordings.json'))}
    rows = [r for r in json.load(open(data / 'curated.json')) if r['role'] == 'cal' and r.get('ok')]
    split = dict(split)
    split['cal'] = [dict(audio_sha256=r['audio_sha256'], audio_rel=r['audio_rel'], rec=r['rec'],
                         charts=recs[r['rec']]['charts']) for r in rows if r['rec'] in recs]
    return split


TUNE_GRID = {
    'base': {},
    'step6': dict(step_penalty=6.0), 'step13': dict(step_penalty=13.0),
    'reset7': dict(reset_penalty=7.0), 'reset14': dict(reset_penalty=14.0),
    'phase0': dict(phase_weight=0.0, period_weight=0.0), 'phase3x': dict(phase_weight=0.03, period_weight=0.03),
    'sigma15': dict(sigma_cont=0.015), 'sigma40': dict(sigma_cont=0.04),
}


def tune(a, split, index, out: Path) -> None:
    """Decoder settings on the cal subset with v0's last checkpoint: BL-20 per stratum for each setting."""
    import multiprocessing as mp
    from .infer import song_outputs
    run, bulk, root = Path(a.run), Path(a.bulk), Path(a.root)
    items = [x for x in eval_items(split, ['cal'], root, **_paths(a)) if 'error' not in x]
    items = [x for x in items if x['item'] in index and 'error' not in index[x['item']]]
    if a.limit:
        items = items[:a.limit]
    ck = _checkpoints(Path(a.bulk_runs) / run.name, 'last1')[0]
    model = _load_model(ck, a.device)
    ev = {x['key']: v0_evidence(song_outputs(model, inference_song(x['item'], index, bulk), a.device)) for x in items}
    del model
    grid = json.loads(a.decode_kw) if a.decode_kw else TUNE_GRID
    res = {}
    by_key = {x['key']: x for x in items}
    with mp.get_context('spawn').Pool(a.processes) as pool:
        for name, kw in grid.items():
            jobs = [(name, k, e, kw, by_key[k]['ref'], by_key[k]['strata']) for k, e in ev.items()]
            sc = {}
            for arm, key, res_k, diag, err in pool.imap_unordered(_decode_job, jobs, chunksize=2):
                if res_k is not None:
                    sc[key] = res_k
            t = aggregate_arm(items, sc, bootstrap=0)
            res[name] = dict(kw=kw, **{s: dict(bl20=t[s]['bl20']['value'], n=t[s]['n_recordings'],
                                                 level=t[s]['level']['value'], bars=t[s]['bars_start']['value'])
                                       for s in t if '|' not in s})
            print(name, json.dumps({s: round(v['bl20'], 4) for s, v in res[name].items() if s != 'kw' and v['bl20'] is not None}), flush=True)
    (out / 'tune.json').write_text(json.dumps(dict(checkpoint=ck.name, n_items=len(items), results=res), indent=1))


def inference_song(item: str, index: dict, bulk: Path):
    from ..features.cache import mel_path
    from .data import Song
    from .songs import memo_path
    st = index[item]
    return Song(item, str(mel_path(bulk / 'mel', item)), str(memo_path(bulk, item)), st['mean'], st['std'],
                st['frames'], None)


def _checkpoints(ck_dir: Path, which: str) -> list[Path]:
    cks = sorted(ck_dir.glob('ckpt-*.pt'))
    if which.startswith('last'):
        return cks[-int(which[4:] or 1):]
    return [ck_dir / f'ckpt-{w}.pt' for w in which.split(',')]


def _load_model(path: Path, device: str):
    import torch
    from .model import V0
    st = torch.load(path, map_location=device, weights_only=False)
    m = V0(**(st.get('model_kw') or {})).to(device)
    m.load_state_dict(st['model'])
    return m.eval()


def main(a) -> None:
    import multiprocessing as mp
    from .infer import song_outputs
    data, run, bulk, root = Path(a.data), Path(a.run), Path(a.bulk), Path(a.root)
    out = Path(a.out) if a.out else run / 'eval'
    out.mkdir(parents=True, exist_ok=True)
    split = with_cal(json.load(open(Path(a.r0) / 'split.json')), data, Path(a.r0))
    index = json.load(open(bulk / 'song-index-v0.json'))
    arms = a.arms.split(',')
    if a.tune:
        return tune(a, split, index, out)
    cfg_kw = json.loads(a.decode_kw) if a.decode_kw else {}
    log = open(out / 'log.txt', 'a')

    def say(msg):
        log.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
        log.flush()
        print(msg, flush=True)

    for panel in a.panels.split(','):
        items = eval_items(split, [panel], root, **_paths(a))
        if a.limit:
            items = items[:a.limit]
        bad = [x for x in items if 'error' in x]
        scored = [x for x in items if 'error' not in x]
        say(f'{panel}: {len(items)} items, {len(bad)} unreadable, {sum(1 for x in scored if x["literal"])} literal-rule')
        songs = {x['item']: inference_song(x['item'], index, bulk) for x in scored if x['item'] in index
                 and 'error' not in index[x['item']]}
        ck_dir = Path(a.bulk_runs) / run.name
        cks = _checkpoints(ck_dir, a.checkpoints) if any(arm.startswith('v0') for arm in arms) else []
        n_frames = {x['key']: int(index[x['item']]['frames']) if x['item'] in index else 0 for x in scored}
        def static_jobs():
            for arm in arms:
                if arm in BT_SHIFT_MS:
                    d = sorted(bulk.glob(f'bt-{arm}-*'))
                    for x in scored:
                        p = d[0] / f"{x['item']}.npz" if d else None
                        if p is not None and p.exists() and n_frames[x['key']]:
                            yield (arm, x['key'], bt_evidence(p, BT_SHIFT_MS[arm], n_frames[x['key']]), cfg_kw, x['ref'], x['strata'])
                elif arm == 'oracle':
                    for x in scored:
                        if n_frames[x['key']]:
                            yield (arm, x['key'], oracle_evidence(x['ref'], n_frames[x['key']]), cfg_kw, x['ref'], x['strata'])
                elif arm == 'r2':
                    for x in scored:
                        p = bulk / 'r2-export' / f"{x['item']}.json"
                        if p.exists():
                            yield ('r2', x['key'], r2_map(p), cfg_kw, x['ref'], x['strata'])

        maps: dict[str, dict] = {}
        diags: dict[str, dict] = {}
        errors = []

        def collect(res):
            arm, key, sc, diag, err = res
            if err:
                errors.append(dict(arm=arm, key=key, error=err))
                return
            maps.setdefault(arm, {})[key] = sc
            diags.setdefault(arm, {})[key] = diag

        with mp.get_context('spawn').Pool(a.processes) as pool:
            pending = []

            def submit(job):
                pending.append(pool.apply_async(_decode_job, (job,)))
                while len(pending) > 4 * a.processes:
                    collect(pending.pop(0).get())

            for job in static_jobs():
                submit(job)
            # v0 evidence on the GPU, decoded in the pool as it comes
            if cks:
                device = a.device
                for ck in cks:
                    model = _load_model(ck, device)
                    tag = ck.stem.split('-')[1]
                    keys = [x for x in scored if x['item'] in songs]
                    for i, x in enumerate(keys):
                        s = songs[x['item']]
                        acts = song_outputs(model, s, device)
                        submit((f'v0@{tag}', x['key'], v0_evidence(acts), cfg_kw, x['ref'], x['strata']))
                        if ck == cks[-1]:
                            if 'v0f1' in arms:
                                submit((f'v0f1@{tag}', x['key'], v0_evidence(acts, f1_only=True), cfg_kw, x['ref'], x['strata']))
                            if 'v0zero' in arms:
                                submit(('v0zero', x['key'], v0_evidence(song_outputs(model, s, device, memory='zero')), cfg_kw, x['ref'], x['strata']))
                            if 'v0other' in arms:
                                o = songs[keys[(i + 7) % len(keys)]['item']]
                                submit(('v0other', x['key'], v0_evidence(song_outputs(model, s, device, memory='other',
                                                                                     other_song=o)), cfg_kw, x['ref'], x['strata']))
                    del model
                    say(f'{panel}: evidence from {ck.name} done')
            while pending:
                collect(pending.pop(0).get())
        say(f'{panel}: {sum(len(v) for v in maps.values())} maps scored; {len(errors)} errors')
        by_key = {x['key']: x for x in scored}
        scores = maps
        tables, compact_tables = {}, {}
        for arm, sc in sorted(scores.items()):
            tables[arm] = aggregate_arm(scored, sc)
            lit = [x for x in scored if not x['literal']]
            tables[arm + '|literal-safe'] = aggregate_arm(lit, sc)
            compact_tables[arm] = compact(tables[arm])
            compact_tables[arm + '|literal-safe'] = compact(tables[arm + '|literal-safe'])
            for nm, sub in ((arm, scored), (arm + '|literal-safe', lit)):
                for st, row in free_table(sub, sc).items():
                    compact_tables[nm].setdefault(st, {}).update(row)
        v0_arms = [k for k in scores if k.startswith('v0@')]
        pair = {}
        for va in v0_arms + [k for k in scores if k.startswith('v0f1@')] + [k for k in ('v0zero', 'v0other') if k in scores]:
            for other in ('r2', 'final0', 'final1', 'final2', 'oracle') + tuple(v0_arms[-1:]):
                if other in scores and other != va:
                    for m in ('bl20_nonfree', 'bl20', 'level', 'bars_start', 'reset_recall'):
                        pair[f'{va}-{other}:{m}'] = paired(scored, scores[va], scores[other], m)
        with open(out / f'items-{panel}.jsonl', 'w') as f:
            for arm, sc in scores.items():
                for k, s in sc.items():
                    x = by_key[k]
                    f.write(json.dumps(dict(arm=arm, key=k, rec=x['rec'], strata=sorted(x['strata']), literal=x['literal'],
                                            **{m: s['all'].summary().get(m) for m in MEASURES + ('offset_ms', 'free_share')},
                                            **s['free']['all'],
                                            diag=diags.get(arm, {}).get(k)), default=float) + '\n')
        meta = dict(panel=panel, n_items=len(items), unreadable=len(bad),
                    reader_flagged=sum(1 for x in scored if x['reader_flags']), scored=len(scored),
                    literal_safe=sum(1 for x in scored if not x['literal']),
                    recordings=len({x['rec'] for x in scored}), checkpoints=[c.name for c in cks], decode_kw=cfg_kw,
                    errors=errors[:20], n_errors=len(errors),
                    strata_items={s: sum(1 for x in scored if s in x['strata']) for s in
                                  ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9')})
        (out / f'tables-{panel}.json').write_text(json.dumps(dict(meta=meta, tables=tables), default=float))
        (out / f'compact-{panel}.json').write_text(json.dumps(dict(meta=meta, tables=compact_tables), indent=1, default=float))
        (out / f'paired-{panel}.json').write_text(json.dumps(pair, indent=1))
        say(f'{panel}: written')
    if a.ablation:
        ablation(a, out, say)


def ablation(a, out: Path, say) -> None:
    """Teacher-forced loss on val-n1 crops (2 per song) with the memory zeroed or swapped (guardrail 3)."""
    from .data import fixed_crops, load_songs
    from .train import probe_loss
    data, run, bulk = Path(a.data), Path(a.run), Path(a.bulk)
    rows = [r for r in json.load(open(data / 'curated.json')) if r['role'] in ('val_hard', 'val_random') and r.get('ok')]
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: r['item']):
        if r['item'] not in seen:
            seen.add(r['item'])
            uniq.append(r)
    songs = load_songs(uniq, json.load(open(bulk / 'song-index-v0.json')), bulk, bulk / 'labels-v0b')
    crops = fixed_crops(songs, per_song=2)
    refs = json.load(open(run / 'refs.json'))
    ck = _checkpoints(Path(a.bulk_runs) / run.name, 'last1')[0]
    model = _load_model(ck, a.device)
    n = len(songs)
    res = {}
    base, tot_base, parts_base = probe_loss(model, crops, songs, refs['pos_weight'], a.device)
    owner = np.array([c[2] for c in crops])
    rec_of = np.array([songs[i].item for i in owner])
    rng = np.random.default_rng(0)

    def interval(d):
        recs = sorted(set(rec_of))
        per = np.array([d[rec_of == r].mean() for r in recs])
        boot = per[rng.integers(0, len(per), (2000, len(per)))].mean(axis=1)
        return dict(delta=float(per.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                    n_recordings=len(per))
    res['own'] = dict(loss=base, parts={k: float(v.mean()) for k, v in parts_base.items()})
    variants = dict(zero=('zero', None), other1=('other', {i: (i + 1) % n for i in range(n)}),
                    other7=('other', {i: (i + 7) % n for i in range(n)}))
    tots = {}
    for name, (mode, other) in variants.items():
        lv, tot, parts = probe_loss(model, crops, songs, refs['pos_weight'], a.device, memory=mode, other=other)
        tots[name] = tot
        res[name] = dict(loss=lv, delta_total=interval(tot - tot_base),
                         delta_parts={k: interval(v - parts_base[k]) for k, v in parts.items()})
    res['null_other1_minus_other7'] = interval(tots['other1'] - tots['other7'])
    res['checkpoint'] = ck.name
    res['crops'] = len(crops)
    (out / 'ablation.json').write_text(json.dumps(res, indent=1))
    say('ablation written')
