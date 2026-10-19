"""Rung 1 (design §5.1 R1): references, training, scoring and the verdict.

From the project root with ``PYTHONPATH=<src>/n/src``::

    python -m ensomi_n.run.r1 refs     --run R1 --r0 R0          # kill-rule scale and positive weights
    python -m ensomi_n.run.r1 train    --run R1 --r0 R0 --seed 0 --arms A1,A2 --budget-min 11
    python -m ensomi_n.run.r1 score    --run R1 --r0 R0 --seed 0 --arms A1,A2
    python -m ensomi_n.run.r1 controls --run R1 --r0 R0          # BeatThis final0/1/2 (+-5 ms), R2, chance
    python -m ensomi_n.run.r1 verdict  --run R1 --r0 R0

``R0`` is rung 0's run directory (its ``split.json``, ``labels-result.json``, ``throughput.json``);
``R1`` is this rung's. Bulk outputs (checkpoints, activations) go under ``--bulk``. Every output that
uses strata says ``interim_strata: true``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .. import probe

BT_SHIFT_MS = dict(final0=30.0, final1=31.9, final2=31.3)
RULE_STRATA = ('S2', 'S3', 'S4', 'S5', 'S9')
ALL_STRATA = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S9')
MIN_RECORDINGS = 20
PANELS = ('panel', 'val_hard', 'val_random', 'reset')


def _split(r0) -> dict:
    return json.load(open(Path(r0) / 'split.json'))


def _labels_dir(r0) -> Path:
    return Path(json.load(open(Path(r0) / 'labels-result.json'))['labels_dir'])


def _refs_for(items: list[dict], root: str) -> dict:
    from ..labels import read_set
    out = {}
    for it in items:
        try:
            ref = read_set([os.path.join(root, c) for c in it['charts']], key=it['item'])
            out[it['item']] = dict(ref=ref, beat=probe.beat_windows(ref), bar=probe.bar_windows(ref))
        except Exception as exc:
            out[it['item']] = dict(error=str(exc))
    return out


def _bt_dir(bulk: Path, ckpt: str) -> Path | None:
    found = sorted(bulk.glob(f'bt-{ckpt}-*'))
    return found[0] if found else None


def _bt_acts(path: Path, shift_ms: float) -> tuple[probe.Activation, probe.Activation]:
    z = np.load(path)
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
    return (probe.Activation(sig(z['beat']), -shift_ms, 20.0), probe.Activation(sig(z['down']), -shift_ms, 20.0))


def _measures(beat_act, down_act, w: dict, peaks_ms, period=None) -> dict:
    ref = w['ref']
    return dict(m1=probe.m1(beat_act, w['beat']), m1_bar=probe.m1(down_act, w['bar']) if w['bar'] else float('nan'),
                m2=probe.m2(beat_act, w['beat'], period=period), m3=probe.m3(peaks_ms, ref),
                chance=probe.chance(w['beat']), chance_bar=probe.chance(w['bar']) if w['bar'] else float('nan'),
                weight=float(sum(x['weight'] for x in w['beat'])))


def cmd_refs(a) -> None:
    from ..train.data import Sampler, open_songs
    run, r0, bulk = Path(a.run), Path(a.r0), Path(a.bulk)
    run.mkdir(parents=True, exist_ok=True)
    split = _split(r0)
    val = [it for it in split['val_hard'] + split['val_random'] if 'S0' in it.get('strata', [])]
    refs = _refs_for(val, a.root)
    f0 = _bt_dir(bulk, 'final0')
    chance, m1s, used = [], [], 0
    for it in val:
        w = refs[it['item']]
        if 'error' in w or not w['beat']:
            continue
        chance.append(probe.chance(w['beat']))
        p = f0 / f"{it['audio_sha256']}.npz" if f0 else None
        if p is not None and p.exists():
            m1s.append(probe.m1(_bt_acts(p, BT_SHIFT_MS['final0'])[0], w['beat']))
            used += 1
    songs = open_songs(split['train'][:200], bulk / 'mel', _labels_dir(r0))
    s = Sampler(songs, seed=12345)
    rates = {}
    for _ in range(10):
        _, tg = s.batch(32)
        for k in ('beat', 'half', 'down'):
            m = tg['mask']
            rates.setdefault(k, []).append(float(tg[k][m].mean()))
    pos = {k: float(np.sqrt((1 - np.mean(v)) / np.mean(v))) for k, v in rates.items()}
    thr = json.load(open(r0 / 'throughput.json')) if (r0 / 'throughput.json').exists() else {}
    r0_rate = next((r['crops_per_s'] for r in thr.get('rows', []) if r.get('encoder') == 'NE-S' and r.get('batch') == 32
                    and 'crops_per_s' in r), None)
    out = dict(val_s0_items=len(val), val_s0_scored=len(chance), val_s0_chance=float(np.mean(chance)),
               val_s0_final0=float(np.mean(m1s)) if m1s else float('nan'), val_s0_final0_n=used,
               label_rates={k: float(np.mean(v)) for k, v in rates.items()}, pos_weight=pos,
               r0_ne_s_b32_crops_per_s=r0_rate, bulk_run=str(Path(a.bulk_runs) / run.name), interim_strata=True,
               bt_final0_dir=str(f0) if f0 else None, bt_shift_ms=BT_SHIFT_MS)
    (run / 'refs.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _val_s0(split, root, bulk) -> list[dict]:
    from ..features.cache import mel_path
    items = [it for it in split['val_hard'] + split['val_random'] if 'S0' in it.get('strata', [])]
    refs = _refs_for(items, root)
    out = []
    for it in items:
        w = refs[it['item']]
        p = mel_path(bulk / 'mel', it['audio_sha256'])
        if 'error' in w or not w['beat'] or not p.exists():
            continue
        out.append(dict(item=it['item'], mel=np.load(p, mmap_mode='r'), windows=w['beat']))
    return out


def cmd_train(a) -> None:
    from ..train import train_arm
    from ..train.data import open_songs
    run, r0, bulk = Path(a.run), Path(a.r0), Path(a.bulk)
    refs = json.load(open(run / 'refs.json'))
    split = _split(r0)
    t0 = time.time()
    songs = open_songs(split['train'], bulk / 'mel', _labels_dir(r0))
    val = _val_s0(split, a.root, bulk)
    cfg = dict(seed=a.seed, arms=a.arms, budget_min=a.budget_min, n_train_songs=len(songs), n_val_s0=len(val),
               r0=str(r0), split_sha256=json.load(open(r0 / 'labels-result.json'))['split_sha256'],
               code=os.environ.get('N1_CODE_COMMIT', ''), encoder='NE-S', batch=32, crop_s=20, lr=2e-3, warmup=300,
               augment=dict(stretch=[0.88, 1.12], gain=0.5, tilt=0.5), hard_weight=3.0,
               bt_ckpt_sha256=None, load_s=round(time.time() - t0, 1),
               guard_reference_crops_per_s=a.guard_rate or refs.get('r0_ne_s_b32_crops_per_s'))
    (run / f'config-s{a.seed}.json').write_text(json.dumps(cfg, indent=1))
    for arm in a.arms.split(','):
        out = run / f'{arm}-s{a.seed}'
        st_path = out / 'status.json'
        if st_path.exists() and json.load(open(st_path)).get('verdict') != 'running':
            print(f'{arm} s{a.seed} already done')
            continue
        st = train_arm(arm=arm, seed=a.seed, songs=songs, val_s0=val, refs=refs, out_dir=out,
                       budget_s=60.0 * a.budget_min, device=a.device, resume=True,
                       r0_crops_per_s=a.guard_rate or refs.get('r0_ne_s_b32_crops_per_s'))
        print(json.dumps({k: v for k, v in st.items() if k != 'history'}))


def _score_items(split, panels) -> list[tuple[str, dict]]:
    return [(p, it) for p in panels for it in split[p]]


def cmd_score(a) -> None:
    import torch
    from ..features.cache import mel_path
    from ..models import NES, Model
    from ..train import infer_song
    run, r0, bulk = Path(a.run), Path(a.r0), Path(a.bulk)
    split = _split(r0)
    # The newest checkpoint is scored on --panels; the earlier ones (which give only the checkpoint
    # spread) on --panels-early, by default the same.
    early = a.panels_early or a.panels
    items_all = _score_items(split, sorted(set(a.panels.split(',')) | set(early.split(',')), key=PANELS.index))
    refs = _refs_for([it for _, it in items_all], a.root)
    for arm in a.arms.split(','):
        st = json.load(open(run / f'{arm}-s{a.seed}' / 'status.json'))
        ck_dir = Path(st['ckpt_dir'])
        ckpts = sorted(ck_dir.glob('ckpt-*.pt'))[-a.last:]
        for ck in ckpts:
            out_path = run / f'scores-{arm}-s{a.seed}-{ck.stem}.json'
            if out_path.exists():
                continue
            items = _score_items(split, (a.panels if ck == ckpts[-1] else early).split(','))
            model = Model(NES(), arm).to(a.device)
            model.load_state_dict(torch.load(ck, map_location=a.device, weights_only=False)['model'])
            model.eval()
            rows = []
            t0 = time.time()
            for panel, it in items:
                w = refs[it['item']]
                p = mel_path(bulk / 'mel', it['audio_sha256'])
                if 'error' in w or not w['beat'] or not p.exists():
                    rows.append(dict(item=it['item'], panel=panel, missing=True))
                    continue
                acts = infer_song(model, np.load(p, mmap_mode='r'), a.device)
                beat = probe.Activation(acts['beat'], 0.0, 10.0)
                down = probe.Activation(acts['down'], 0.0, 10.0)
                if arm == 'A1':
                    pk = probe.peaks(beat, 0.5)
                    per = None
                else:
                    pk = probe.phase_wraps(probe.Activation(acts['phase'], 0.0, 10.0),
                                           probe.Activation(acts['kappa'], 0.0, 10.0))
                    per = probe.Activation(acts['period_ms'], 0.0, 10.0)
                rows.append(dict(item=it['item'], panel=panel, rec=it['rec'], strata=it.get('strata', []),
                                 **_measures(beat, down, w, pk, per)))
            out_path.write_text(json.dumps(dict(arm=arm, seed=a.seed, ckpt=ck.name, rows=rows, interim_strata=True,
                                                wall_s=round(time.time() - t0, 1))))
            print(arm, a.seed, ck.name, round(time.time() - t0, 1), 's')


def cmd_controls(a) -> None:
    run, r0, bulk = Path(a.run), Path(a.r0), Path(a.bulk)
    split = _split(r0)
    items = _score_items(split, a.panels.split(','))
    refs = _refs_for([it for _, it in items], a.root)
    r2_dir = bulk / 'r2-export'
    out = {}
    for ckpt in ('final0', 'final1', 'final2'):
        d = _bt_dir(bulk, ckpt)
        for delta in (0.0, -5.0, 5.0):
            name = f'bt-{ckpt}' + ('' if delta == 0 else f'{delta:+.0f}ms')
            rows = []
            for panel, it in items:
                w = refs[it['item']]
                p = d / f"{it['audio_sha256']}.npz" if d else None
                if 'error' in w or not w['beat'] or p is None or not p.exists():
                    continue
                beat, down = _bt_acts(p, BT_SHIFT_MS[ckpt] + delta)
                rows.append(dict(item=it['item'], panel=panel, rec=it['rec'], strata=it.get('strata', []),
                                 **_measures(beat, down, w, probe.peaks(beat, 0.5))))
            out[name] = rows
    rows = []
    for panel, it in items:
        w = refs[it['item']]
        p = r2_dir / f"{it['audio_sha256']}.json"
        if 'error' in w or not w['beat'] or not p.exists():
            continue
        r2 = json.load(open(p))
        beats = np.asarray(r2['beats'])
        downs = beats[np.asarray(r2['downbeats'], dtype=np.int64)] if r2['downbeats'] else np.zeros(0)
        rows.append(dict(item=it['item'], panel=panel, rec=it['rec'], strata=it.get('strata', []),
                         **_measures(probe.Activation.impulses(beats), probe.Activation.impulses(downs), w, beats)))
    out['r2'] = rows
    (run / 'scores-controls.json').write_text(json.dumps(dict(rows=out, bt_shift_ms=BT_SHIFT_MS, interim_strata=True)))
    print({k: len(v) for k, v in out.items()})


def cmd_verdict(a) -> None:
    from .verdict import verdict
    res = verdict(Path(a.run), Path(a.r0))
    print(json.dumps({k: res[k] for k in ('reading', 'a2_vs_a1', 'next')}, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog='python -m ensomi_n.run.r1')
    p.add_argument('cmd', choices=('refs', 'train', 'score', 'controls', 'verdict'))
    p.add_argument('--run', required=True)
    p.add_argument('--r0', required=True)
    p.add_argument('--bulk', default='artifacts/bulk/n1-cache')
    p.add_argument('--bulk-runs', default='artifacts/bulk/n1-runs')
    p.add_argument('--root', default='.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arms', default='A1,A2')
    p.add_argument('--budget-min', type=float, default=11.0)
    p.add_argument('--device', default='mps')
    p.add_argument('--last', type=int, default=3)
    p.add_argument('--panels', default=','.join(PANELS))
    p.add_argument('--guard-rate', type=float, default=0.0,
                   help='train: crops/s the throughput guard compares with (default: rung 0 NE-S batch 32)')
    p.add_argument('--panels-early', default='', help='score: panels for all but the newest checkpoint')
    a = p.parse_args(argv)
    dict(refs=cmd_refs, train=cmd_train, score=cmd_score, controls=cmd_controls, verdict=cmd_verdict)[a.cmd](a)


if __name__ == '__main__':
    main()
