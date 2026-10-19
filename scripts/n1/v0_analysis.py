"""N1 v0 against R2 beyond N's reading: reader-neutral spans, R's reading as reference, phase-tolerant BL-20.

The question (research notes, ``n1-night-20261018/v0-analysis.md``): does v0's lead over R2 hold where N's
and R's readers agree, and under a scoring that forgives a constant phase convention? This script is outside
``n/`` because it imports R's reader (`reader_census.r_lines`, R's ``musical_grid``).

Subcommands:

- ``maps``: decode a v0 run's checkpoints on the CPU for the evaluation panels, as the v0 evaluation does
  (`ensomi_n.v0.evaluate`), and keep the maps (gzip JSON lines per checkpoint and panel). The evaluation
  scored its maps without keeping them.
- ``items``: per item and arm (R2 as exported and each v0 checkpoint), BL-20 on non-free lines and its
  phase-tolerant form, for every stratum's spans, against

  - ``N``: N's reference (the v0 evaluation's number; the ``all`` value is checked against its items file),
  - ``Nn``: N's reference on the reader-neutral heads only (below),
  - ``R``: R's canonical lines as the reference, with N's heads as weights, and ``Rn`` its neutral part.

  A head is changed when E3's matched rule (`reader_census.changed_matched`) flags it on the raw lines or on
  the refined comparison (free spans refined on both sides, as E3's ``changed_refined``). Each changed head
  excludes the interval from the earlier of the readers' lines before it to the later of their lines after
  it; the heads left are the neutral heads, and every one of them has both readers' lines around it within
  20 ms.

  Phase-tolerant BL-20: the reference's red lines are cut into segments at every line whose kind
  (`ensomi_n.eval.strata`) is not ``same``; in each segment the prediction is shifted by s x B for s in
  `SHIFTS` (B the canonical beat at each reference line), the shift passing the most head weight is kept, and
  the passes are summed over segments. Its chance per line is the measure of the union of the 20 ms windows
  around the shifted phases (`union_chance`).

- ``agg``: tables from the items files (numpy only): per panel, group (all, fast, not fast by the literal
  rule) and stratum, each arm's value and chance, v0 - R2 with a paired recording bootstrap, the recordings
  and head weight kept, and v0's spread over its checkpoints.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

SHIFTS = (0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0)
TAU = 20.0
HERE = Path(__file__).resolve().parent
TESTS = ('N', 'Nn', 'R', 'Rn')
FIELDS = ('w', 'bl', 'pt', 'nz', 'ch', 'chp')


def _rc():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import reader_census
    return reader_census


# ---------------------------------------------------------------- pieces with absolute tests

def merge_closed(intervals) -> list[tuple[float, float]]:
    """Closed intervals merged where they touch or overlap."""
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b >= a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def exclusion(heads: np.ndarray, changed: np.ndarray, n_lines: np.ndarray, r_lines: np.ndarray):
    """Per changed head, the closed interval from the earlier of the readers' lines before it to the later of
    their lines after it (the head itself where neither reader has a line on that side), merged."""
    rc = _rc()
    h = heads[changed]
    if not len(h):
        return []
    n_prev, n_next = rc._around(n_lines, h)
    r_prev, r_next = rc._around(r_lines, h)
    a = np.fmin(n_prev, r_prev)
    b = np.fmax(n_next, r_next)
    a = np.where(np.isnan(a), h, a)
    b = np.where(np.isnan(b), h, b)
    return merge_closed(zip(np.minimum(a, h), np.maximum(b, h)))


def outside(t: np.ndarray, intervals) -> np.ndarray:
    ok = np.ones(len(t), dtype=bool)
    for a, b in intervals:
        ok &= ~((t >= a) & (t <= b))
    return ok


def union_chance(spacing: np.ndarray, factor: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Per line, the share of phases passing BL-tau for some shift in `SHIFTS`: the union of arcs of half
    width tau / spacing around the residues s x factor (mod 1) on the unit circle (B / spacing = factor)."""
    spacing = np.asarray(spacing, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.int64)
    out = np.zeros(len(spacing))
    a2 = 2.0 * tau / np.where(spacing > 0, spacing, np.inf)
    for f in np.unique(factor):
        r = np.unique(np.round(np.mod(np.array(SHIFTS) * f, 1.0), 9) % 1.0)
        gaps = np.diff(np.append(r, r[0] + 1.0))
        idx = factor == f
        out[idx] = np.minimum(1.0, np.minimum(a2[idx][:, None], gaps[None, :]).sum(axis=1))
    return out


def segment_index(t: np.ndarray, starts: np.ndarray) -> np.ndarray:
    from ensomi_n.timing import MERGE_MS
    if not len(starts):
        return np.zeros(len(t), dtype=np.int64)
    return np.clip(np.searchsorted(starts, t + MERGE_MS, side='right') - 1, 0, None).astype(np.int64)


def scores_for(R, pred_timing, heads_masks: dict, seg_starts: np.ndarray) -> dict:
    """{span-set name: [w, bl, pt, nz, ch, chp]} for one reference and one predicted timing.

    ``w`` the non-free head weight, ``bl`` BL-20 on non-free lines, ``pt`` its phase-tolerant form, ``nz`` the
    weight share whose best shift is not 0, ``ch`` and ``chp`` the chance of ``bl`` and ``pt``.
    """
    from ensomi_n.eval.measures import compared_lines, nearest
    p = pred_timing
    lines, pl, _ = compared_lines(R.lines(), p.canonical(), R.timing.exact_steps, p.exact_steps,
                                  ref_all=R.timing.canonical(),
                                  ref_end=float(R.timing.beats[-1]) if len(R.timing) else -math.inf,
                                  pred_end=float(p.beats[-1]) if len(p) else -math.inf)
    nonfree = lines.factor == 1
    B = lines.beat_ms * lines.factor
    oks = []
    for s in SHIFTS:
        e, _ = nearest(lines.t - s * B, pl.t)
        oks.append(np.isfinite(e) & (np.abs(e) <= TAU + 1e-9))
    seg = segment_index(lines.t, seg_starts)
    nseg = int(seg.max()) + 1 if len(seg) else 1
    ch0 = np.minimum(1.0, 2.0 * TAU / lines.beat_ms) if len(lines) else np.zeros(0)
    chp = union_chance(lines.beat_ms, lines.factor)
    out = {}
    for name, hm in heads_masks.items():
        if not hm.any() or not len(lines):
            continue
        w = dataclasses.replace(R, heads=R.heads[hm]).weights(lines) * nonfree
        W = float(w.sum())
        if W <= 0:
            continue
        passes = np.array([np.bincount(seg, weights=w * ok, minlength=nseg) for ok in oks])
        segw = np.bincount(seg, weights=w, minlength=nseg)
        arg = passes.argmax(axis=0)
        out[name] = [round(W, 4), float(passes[0].sum() / W), float(passes.max(axis=0).sum() / W),
                     float(segw[arg != 0].sum() / W), float((w * ch0).sum() / W), float((w * chp).sum() / W)]
    return out


def r_reference(ref, r_t: np.ndarray):
    """R's canonical lines as a reference over N's range, weighted by N's heads (N's unscorable spans kept);
    and the count of lines in range that differ from R's (expected 0)."""
    from ensomi_n.timing import Reference
    if len(r_t) < 2:
        return None, -1
    rref = Reference.from_beats(r_t, ref.heads, lo_ms=ref.lo_ms, hi_ms=ref.hi_ms, key=ref.key)
    rref = dataclasses.replace(rref, unscorable=ref.unscorable)
    rl = rref.lines().t
    if not len(rl):
        return rref, len(r_t)
    inr = r_t[(r_t >= rl[0] - 1e-6) & (r_t <= rl[-1] + 1e-6)]
    return rref, _unmatched(inr, rl) + _unmatched(rl, inr)


def _unmatched(a: np.ndarray, b: np.ndarray, tol: float = 0.5) -> int:
    """Lines of ``a`` with no line of ``b`` within ``tol`` ms."""
    if not len(a):
        return 0
    if not len(b):
        return int(len(a))
    i = np.clip(np.searchsorted(b, a), 0, len(b) - 1)
    d = np.minimum(np.abs(b[i] - a), np.abs(b[np.maximum(i - 1, 0)] - a))
    return int((d > tol).sum())


def neutral_heads(ref, r_t: np.ndarray) -> tuple[np.ndarray, int, list]:
    """Mask over ``ref.heads`` of the reader-neutral heads, the count of changed heads, and the exclusions."""
    from ensomi_n.eval.measures import compared_lines
    rc = _rc()
    heads = ref.heads
    lo, hi = ref.lo_ms, ref.hi_ms
    inside = (heads >= lo) & (heads <= hi)
    h = heads[inside]
    nl = ref.lines()
    changed = rc.changed_matched(h, nl.t, r_t)
    if len(r_t):
        end = float(ref.timing.beats[-1]) if len(ref.timing) else -math.inf
        n_c, r_c, _ = compared_lines(nl, rc._r_as_lines(r_t), True, False, ref_all=ref.timing.canonical(),
                                     ref_end=end, pred_end=float(r_t[-1]))
        changed = changed | rc.changed_matched(h, n_c.t, r_c.t)
    excl = exclusion(h, changed, nl.t, r_t)
    return inside & outside(heads, excl), int(changed.sum()), excl


def item_record(x: dict, arm_maps: dict) -> dict:
    """Every test for one evaluation item (`ensomi_n.v0.evaluate.eval_items`) and its arms' maps."""
    from ensomi_n.eval.measures import _Region, _prediction
    rc = _rc()
    ref, st = x['ref'], x['st']
    lo, hi = ref.lo_ms, ref.hi_ms
    heads = ref.heads
    r_t = rc.r_lines(ref.meta['timing_chart'], lo, hi)[0]
    neutral, n_changed, excl = neutral_heads(ref, r_t)
    rref, mismatch = r_reference(ref, r_t)
    masks = {'all': np.ones(len(heads), dtype=bool)}
    for name, spans in x['strata'].items():
        masks[name] = _Region(lo, hi, spans).contains(heads)
    masks_n = {k: v & neutral for k, v in masks.items()}
    seg_starts = np.array(sorted(line.start_ms for line in st.lines if line.kind != 'same'))
    arms = {}
    for arm, mmap in arm_maps.items():
        p = _prediction(mmap, lo, hi, None)
        res = {'N': scores_for(ref, p.timing, masks, seg_starts), 'Nn': scores_for(ref, p.timing, masks_n, seg_starts)}
        if rref is not None:
            res['R'] = scores_for(rref, p.timing, masks, seg_starts)
            res['Rn'] = scores_for(rref, p.timing, masks_n, seg_starts)
        arms[arm] = {t: {k: [round(v, 6) for v in vals] for k, vals in d.items()} for t, d in res.items()}
    inside = (heads >= lo) & (heads <= hi)
    return dict(key=x['key'], rec=x['rec'], item=x['item'], literal=x['literal'], fast57=x['fast57'],
                strata=sorted(x['strata']), heads=int(inside.sum()), changed=n_changed,
                neutral=int(neutral.sum()), excluded_ms=round(sum(b - a for a, b in excl), 1),
                r_lines=int(len(r_t)), r_mismatch=mismatch, n_segments=int(len(seg_starts)), arms=arms)


# ---------------------------------------------------------------- mac stages

def _setup(a):
    from ensomi_n.v0 import evaluate as ev
    data, r0, bulk = Path(a.data), Path(a.r0), Path(a.bulk)
    split = ev.with_cal(json.load(open(r0 / 'split.json')), data, r0)
    index = json.load(open(bulk / 'song-index-v0.json'))
    return ev, split, index


def _scored(ev, split, index, panel, a):
    items = ev.eval_items(split, [panel], Path(a.root), corpus=a.corpus, tags=a.tags)
    scored = [x for x in items if 'error' not in x and x['item'] in index and 'error' not in index[x['item']]]
    return items, scored


def _log(out: Path, msg: str) -> None:
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    with open(out / 'log.txt', 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def cmd_maps(a) -> None:
    import torch
    torch.set_num_threads(a.threads)
    from ensomi_n.v0.decode import DecodeConfig, decode
    from ensomi_n.v0.infer import song_outputs
    ev, split, index = _setup(a)
    out, maps = Path(a.out), Path(a.maps)
    out.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    cfg_kw = ev.pick_settings(Path(a.tune_dir))
    cks = ev._checkpoints(Path(a.bulk_runs) / a.run_name, a.checkpoints)
    _log(out, f'maps: run {a.run_name}, checkpoints {[c.name for c in cks]}, decode {cfg_kw}, threads {a.threads}')
    for panel in a.panels.split(','):
        items, scored = _scored(ev, split, index, panel, a)
        songs = {x['item']: ev.inference_song(x['item'], index, Path(a.bulk)) for x in scored}
        _log(out, f'maps {panel}: {len(items)} items, {len(scored)} with audio, {len(songs)} songs')
        for ck in cks:
            tag = ck.stem.split('-')[1]
            path = maps / f'maps-{tag}-{panel}.jsonl.gz'
            if path.exists():
                _log(out, f'maps {panel} {tag}: exists, skipped')
                continue
            model = ev._load_model(ck, 'cpu')
            t0, cache = time.time(), {}
            tmp = path.with_name(path.name + '.tmp')
            with gzip.open(tmp, 'wt') as f:
                for i, x in enumerate(scored):
                    if x['item'] not in cache:
                        acts = song_outputs(model, songs[x['item']], 'cpu')
                        mmap, diag = decode(ev.v0_evidence(acts), DecodeConfig(**cfg_kw), model=f'v0@{tag}')
                        cache[x['item']] = (mmap.to_json(), diag)
                    f.write(json.dumps(dict(key=x['key'], item=x['item'], map=cache[x['item']][0],
                                            diag=cache[x['item']][1]), default=float) + '\n')
                    if (i + 1) % 50 == 0:
                        _log(out, f'maps {panel} {tag}: {i + 1}/{len(scored)} ({time.time() - t0:.0f} s)')
            os.replace(tmp, path)
            del model
            _log(out, f'maps {panel} {tag}: done in {time.time() - t0:.0f} s')


def cmd_items(a) -> None:
    from ensomi_n.mapformat import MusicMap
    ev, split, index = _setup(a)
    out, maps = Path(a.out), Path(a.maps)
    out.mkdir(parents=True, exist_ok=True)
    for panel in a.panels.split(','):
        t0 = time.time()
        items, scored = _scored(ev, split, index, panel, a)
        arm_maps: dict[str, dict] = {}
        for p in sorted(maps.glob(f'maps-*-{panel}.jsonl.gz')):
            arm = 'v0@' + p.name.split('-')[1]
            with gzip.open(p, 'rt') as f:
                for line in f:
                    d = json.loads(line)
                    arm_maps.setdefault(arm, {})[d['key']] = MusicMap.from_json(d['map'])
        for x in scored:
            p = Path(a.bulk) / 'r2-export' / f"{x['item']}.json"
            if p.exists():
                arm_maps.setdefault('r2', {})[x['key']] = ev.r2_map(p)
        ref_vals = {}
        ip = Path(a.eval_dir) / f'items-{panel}.jsonl'
        if ip.exists():
            for line in open(ip):
                d = json.loads(line)
                ref_vals[(d['arm'], d['key'])] = d.get('bl20_nonfree')
        _log(out, f'items {panel}: {len(scored)} items, arms {sorted(arm_maps)}')
        n_err, diffs = 0, {}
        with open(out / f'items-{panel}.jsonl', 'w') as f:
            for x in scored:
                try:
                    rec = item_record(x, {arm: m[x['key']] for arm, m in arm_maps.items() if x['key'] in m})
                    for arm, res in rec['arms'].items():
                        v = ref_vals.get((arm, x['key']))
                        mine = res.get('N', {}).get('all', [0, math.nan])[1]
                        if v is not None and v == v:
                            diffs.setdefault(arm, []).append(abs(mine - v))
                except Exception as exc:  # noqa: BLE001
                    n_err += 1
                    rec = dict(key=x['key'], rec=x['rec'], error=f'{type(exc).__name__}: {exc} '
                                                                   f'{traceback.format_exc()[-400:]}')
                f.write(json.dumps(rec) + '\n')
        repro = {arm: dict(n=len(d), max=float(max(d)), over_1e6=int(sum(v > 1e-6 for v in d)),
                           mean=float(np.mean(d))) for arm, d in diffs.items()}
        (out / f'repro-{panel}.json').write_text(json.dumps(repro, indent=1))
        _log(out, f'items {panel}: written, {n_err} errors, {time.time() - t0:.0f} s; repro {json.dumps(repro)}')


# ---------------------------------------------------------------- aggregation (control plane)

def load_items(path: Path) -> list[dict]:
    return [d for d in (json.loads(line) for line in open(path)) if 'error' not in d]


def per_recording(items, arm, test, stratum, field='bl', group=None):
    """{rec: mean over its items of field} for items with weight in the stratum, in the group."""
    k = FIELDS.index(field)
    per: dict[str, list] = {}
    for d in items:
        if group == 'fast' and not d['literal'] or group == 'notfast' and d['literal']:
            continue
        v = d['arms'].get(arm, {}).get(test, {}).get(stratum)
        if v is None or not math.isfinite(v[k]):
            continue
        per.setdefault(d['rec'], []).append(v[k])
    return {r: float(np.mean(v)) for r, v in per.items()}


def boot_mean(d: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_delta(a: dict, b: dict, recs=None, n=2000, seed=0):
    keys = sorted(set(a) & set(b) if recs is None else set(a) & set(b) & set(recs))
    if not keys:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    lo, hi = boot_mean(d, n, seed)
    return dict(delta=float(d.mean()), lo=lo, hi=hi, n=len(keys))


def weight_sum(items, arm, test, stratum, group=None):
    k = FIELDS.index('w')
    s = 0.0
    for d in items:
        if group == 'fast' and not d['literal'] or group == 'notfast' and d['literal']:
            continue
        v = d['arms'].get(arm, {}).get(test, {}).get(stratum)
        if v is not None:
            s += v[k]
    return s


def neutral_recordings(items) -> set:
    by: dict[str, bool] = {}
    for d in items:
        by[d['rec']] = by.get(d['rec'], True) and d['changed'] == 0
    return {r for r, ok in by.items() if ok}


def aggregate(items, v0_arms: list[str], last: str) -> dict:
    """Per group, test and stratum: arm means, chance, paired v0 - R2 (last checkpoint), checkpoint spread,
    and the matched comparisons of 1a and 1b."""
    strata = sorted({s for d in items for s in d['strata']}, key=lambda s: int(s[1:])) + ['all']
    neutral_recs = neutral_recordings(items)
    out = {}
    for group in (None, 'fast', 'notfast'):
        g = group or 'all'
        out[g] = {}
        for test in TESTS + ('N1a',):
            src = 'N' if test == 'N1a' else test
            out[g][test] = {}
            for st in strata:
                row = {}
                for field in ('bl', 'pt'):
                    r2 = per_recording(items, 'r2', src, st, field, group)
                    v0 = {arm: per_recording(items, arm, src, st, field, group) for arm in v0_arms}
                    recs = None
                    if test == 'N1a':
                        recs = neutral_recs
                        r2 = {k: v for k, v in r2.items() if k in recs}
                        v0 = {arm: {k: v for k, v in d.items() if k in recs} for arm, d in v0.items()}
                    common = set(r2) & set(v0[last])
                    if not common:
                        continue
                    vals = {arm: float(np.mean([v0[arm][k] for k in common if k in v0[arm]])) for arm in v0_arms}
                    pd = paired_delta(v0[last], r2)
                    row[field] = dict(r2=float(np.mean([r2[k] for k in common])), v0=vals[last],
                                      v0_ckpts=vals, spread=float(max(vals.values()) - min(vals.values())),
                                      delta=pd, n=len(common))
                    if field == 'bl':
                        chance = {}
                        for arm, cf in (('r2', 'ch'), (last, 'ch')):
                            c = per_recording(items, arm, src, st, cf, group)
                            chance[arm] = float(np.mean([c[k] for k in common if k in c]))
                        row['chance'] = chance[last]
                        cp = per_recording(items, last, src, st, 'chp', group)
                        row['chance_pt'] = float(np.mean([cp[k] for k in common if k in cp]))
                        nz = {arm: per_recording(items, arm, src, st, 'nz', group) for arm in ('r2', last)}
                        row['nonzero_shift'] = {arm: float(np.mean([nz[arm][k] for k in common if k in nz[arm]]))
                                                for arm in nz}
                if not row:
                    continue
                if test in ('Nn', 'Rn'):
                    full = test[0]
                    wn, wf = weight_sum(items, last, test, st, group), weight_sum(items, last, full, st, group)
                    row['weight_kept'] = wn / wf if wf > 0 else math.nan
                    # matched: the full reference on the same recordings
                    for field in ('bl', 'pt'):
                        if field not in row:
                            continue
                        recs = set(per_recording(items, last, test, st, field, group))
                        a = per_recording(items, last, full, st, field, group)
                        b = per_recording(items, 'r2', full, st, field, group)
                        row[field]['matched_full'] = paired_delta(a, b, recs)
                        an = per_recording(items, last, test, st, field, group)
                        bn = per_recording(items, 'r2', test, st, field, group)
                        keys = sorted(set(a) & set(b) & set(an) & set(bn))
                        if keys:
                            dd = np.array([(an[k] - bn[k]) - (a[k] - b[k]) for k in keys])
                            lo, hi = boot_mean(dd)
                            row[field]['shrink'] = dict(delta=float(-dd.mean()), lo=float(-hi), hi=float(-lo), n=len(keys))
                if test == 'N1a':
                    for field in ('bl',):
                        a = per_recording(items, last, 'N', st, field, group)
                        b = per_recording(items, 'r2', 'N', st, field, group)
                        others = set(a) - neutral_recs
                        row[field]['others'] = paired_delta(a, b, others)
                out[g][test][st] = row
    return out


def cmd_agg(a) -> None:
    res = {}
    for panel in a.panels.split(','):
        items = load_items(Path(a.items) / f'items-{panel}.jsonl')
        v0_arms = sorted({arm for d in items for arm in d['arms'] if arm.startswith('v0@')})
        res[panel] = dict(meta=dict(items=len(items), recordings=len({d['rec'] for d in items}),
                                    neutral_recordings=len(neutral_recordings(items)),
                                    r_mismatch_items=sum(1 for d in items if d['r_mismatch'] != 0),
                                    changed_items=sum(1 for d in items if d['changed'] > 0),
                                    v0_arms=v0_arms),
                          tables=aggregate(items, v0_arms, v0_arms[-1]))
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    print(f'written {a.out_json}')


# ---------------------------------------------------------------- markdown tables and margins (control plane)

STRATA_ORDER = ['all'] + [f'S{k}' for k in range(10)]


def _f(x, nd=3):
    return '' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.{nd}f}'


def _d(d):
    if not d:
        return ''
    return f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def margin_rows(aggs: list[dict], panel: str, group: str, test: str, field: str) -> dict:
    """Per stratum: per seed v0, R2, delta, checkpoint spread; seed spread, m_s, mean delta and the class
    (1.6 of the analysis rules). With one seed, m is that seed's checkpoint spread (provisional)."""
    out = {}
    for st in STRATA_ORDER:
        rows = [a[panel]['tables'][group][test].get(st, {}).get(field) for a in aggs]
        if any(r is None for r in rows):
            continue
        sigma = abs(rows[0]['v0'] - rows[1]['v0']) if len(rows) > 1 else math.nan
        c = [r['spread'] for r in rows]
        m = max([x for x in [sigma] + c if math.isfinite(x)])
        deltas = [r['delta'] for r in rows]
        dbar = float(np.mean([d['delta'] for d in deltas]))
        n = min(r['n'] for r in rows)
        if n < 10:
            cls = 'n<10'
        elif dbar > m and all(d['lo'] > 0 for d in deltas):
            cls = 'v0 ahead'
        elif dbar < -m and all(d['hi'] < 0 for d in deltas):
            cls = 'R2 ahead'
        elif abs(dbar) <= m:
            cls = 'level'
        else:
            cls = 'unresolved'
        out[st] = dict(n=n, r2=rows[0]['r2'], v0=[r['v0'] for r in rows], deltas=deltas, c=c, sigma=sigma, m=m,
                       dbar=dbar, cls=cls if len(rows) > 1 else cls + ' (provisional)')
    return out


def md_tables(aggs: list[dict], panel: str) -> str:
    seeds = len(aggs)
    lines = []
    for test, title in (('N', 'N reference, full spans (the v0 report\'s number)'),
                        ('Nn', 'Test 1b: N reference on reader-neutral heads'),
                        ('N1a', 'Test 1a: recordings with no changed head'),
                        ('R', 'Test 2: R\'s canonical lines as the reference')):
        for field, fname in (('bl', 'BL-20 non-free'), ('pt', 'phase-tolerant BL-20')):
            for group in ('all', 'fast', 'notfast'):
                rows = margin_rows(aggs, panel, group, test, field)
                if not rows:
                    continue
                head = f'**{title}; {fname}; {group}** ({panel})'
                cols = ['Stratum', 'n', 'R2', 'v0 s0'] + (['v0 s1'] if seeds > 1 else []) + ['Δ s0 [95 %]'] + \
                       (['Δ s1 [95 %]', 'Δ̄'] if seeds > 1 else []) + ['c s0'] + (['c s1', 'σ seed'] if seeds > 1 else []) + \
                       ['m', 'class']
                extra = []
                if test in ('Nn',):
                    cols[2:2] = ['kept']
                if field == 'bl':
                    cols += ['chance']
                elif field == 'pt':
                    cols += ['chance', 'shift≠0 R2 / v0']
                if test == 'Nn':
                    cols += ['Δ full, same rec.', 'shrink [95 %]']
                if test == 'N1a':
                    cols += ['Δ other rec.']
                lines += [head, '', '| ' + ' | '.join(cols) + ' |', '|' + ' --- |' * len(cols)]
                for st, r in rows.items():
                    src = aggs[0][panel]['tables'][group][test][st]
                    f0 = src[field]
                    cells = [st, str(r['n'])]
                    if test == 'Nn':
                        cells.append(_f(src.get('weight_kept'), 2))
                    cells += [_f(r['r2']), _f(r['v0'][0])] + ([_f(r['v0'][1])] if seeds > 1 else [])
                    cells += [_d(r['deltas'][0])] + ([_d(r['deltas'][1]), f"{r['dbar']:+.3f}"] if seeds > 1 else [])
                    cells += [_f(r['c'][0])] + ([_f(r['c'][1]), _f(r['sigma'])] if seeds > 1 else [])
                    cells += [_f(r['m']), r['cls']]
                    if field == 'bl':
                        cells.append(_f(src.get('chance')))
                    else:
                        nz = src.get('nonzero_shift', {})
                        cells += [_f(src.get('chance_pt')), ' / '.join(_f(v, 2) for v in nz.values())]
                    if test == 'Nn':
                        cells += [_d(f0.get('matched_full')), _d(f0.get('shrink'))]
                    if test == 'N1a':
                        cells.append(_d(f0.get('others')))
                    lines.append('| ' + ' | '.join(cells) + ' |')
                lines.append('')
    return '\n'.join(lines)


def cmd_md(a) -> None:
    aggs = [json.loads(Path(p).read_text()) for p in a.agg]
    text = []
    for panel in a.panels.split(','):
        meta = [g[panel]['meta'] for g in aggs]
        text.append(f'### {panel}\n\n' + '; '.join(json.dumps(m) for m in meta) + '\n')
        text.append(md_tables(aggs, panel))
    Path(a.out_md).write_text('\n'.join(text))
    print(f'written {a.out_md}')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('maps', 'items'):
        s = sub.add_parser(name)
        s.add_argument('--data', default='artifacts/n1-night-20261018/runs/v0data2-20261019')
        s.add_argument('--r0', default='artifacts/n1-night-20261018/runs/r0-20261019a')
        s.add_argument('--bulk', default='artifacts/bulk/n1-cache')
        s.add_argument('--bulk-runs', default='artifacts/bulk/n1-runs')
        s.add_argument('--root', default='.')
        s.add_argument('--corpus', default='data/r2-corpus.parquet')
        s.add_argument('--tags', default='artifacts/n1-inventory-20261014/tags.json')
        s.add_argument('--panels', default='panel,val_hard')
        s.add_argument('--run-name', required=True)
        s.add_argument('--maps', required=True)
        s.add_argument('--out', required=True)
        if name == 'maps':
            s.add_argument('--checkpoints', default='last3')
            s.add_argument('--tune-dir', default='artifacts/n1-night-20261018/runs/v0-s0-20261019/eval')
            s.add_argument('--threads', type=int, default=4)
        else:
            s.add_argument('--eval-dir', required=True)
    s = sub.add_parser('agg')
    s.add_argument('--items', required=True)
    s.add_argument('--panels', default='panel,val_hard')
    s.add_argument('--out-json', required=True)
    s = sub.add_parser('md')
    s.add_argument('--agg', action='append', required=True, help='agg JSON per seed, seed 0 first')
    s.add_argument('--panels', default='panel,val_hard')
    s.add_argument('--out-md', required=True)
    a = ap.parse_args(argv)
    dict(maps=cmd_maps, items=cmd_items, agg=cmd_agg, md=cmd_md)[a.cmd](a)


if __name__ == '__main__':
    main()
