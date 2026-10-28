"""G0 feature caches, with audited cohorts and unchanged v0.1 inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from .cohorts import ROOT, OUT, FEAT, BULK, build, dump
from .audit_inputs import digest, probe_bucket
from .feature_capture import capture_song, write_npz, reproduce


def checkpoint(seed):
    return ROOT / f'artifacts/bulk/n1-runs/v01-s{seed}-20261020/ckpt-009001.pt'


def disk_guard(reserve=0):
    paths = [OUT, FEAT / 'v01-s1-009001', FEAT / 'v01-s0-009001-ext']
    used = sum(p.stat().st_size for d in paths if d.exists() for p in d.rglob('*') if p.is_file())
    if used + reserve > 25e9 or shutil.disk_usage(OUT).free - reserve < 80e9:
        raise RuntimeError(f'Disk guard: {used} new bytes; {shutil.disk_usage(OUT).free} free')
    return used


def prepare_song(row, index):
    from ensomi_n.features.cache import mel_path, build_one
    from ensomi_n.v0.songs import memo_path, memo_from_mel
    from ensomi_n.v0.data import Song
    sha = row['audio_sha256']
    mp, memo = mel_path(BULK / 'mel', sha), memo_path(BULK, sha)
    st = index.get(sha)
    if not mp.exists():
        private = OUT / 'input-cache'
        res = build_one((str(ROOT / row['audio_rel']), sha), str(private / 'mel'))
        if res['status'] not in ('present', 'built'):
            raise ValueError(f'Unreadable extension: {res}')
        mp = mel_path(private / 'mel', sha)
        memo = memo_path(private, sha)
    if not st or 'error' in st or not memo.exists():
        m = np.load(mp).astype(np.float32)
        st = dict(frames=len(m), mean=float(m.mean()), std=float(m.std() + 1e-6))
        if not memo.exists():
            memo = memo_path(OUT / 'input-cache', sha)
            memo.parent.mkdir(parents=True, exist_ok=True)
            np.save(memo, memo_from_mel(m))
        del m
    return Song(sha, str(mp), str(memo), st['mean'], st['std'], st['frames'], None)


def member(row):
    from ensomi_n.eval.census import read_header
    ids, sets = set(), set()
    for c in row['charts']:
        h = read_header(ROOT / c)
        for k, dst in [('BeatmapID', ids), ('BeatmapSetID', sets)]:
            try:
                v = int(h.get(k, '-1'))
                if v > 0:
                    dst.add(v)
            except ValueError:
                pass
    if row.get('set_id'):
        sets.add(int(row['set_id']))
    return dict(rec=row['rec'], audio_rel=row['audio_rel'], charts=row['charts'],
                beatmap_ids=sorted(ids), set_ids=sorted(sets), source=row['source'],
                audio_sha256=row['audio_sha256'], ext=row.get('ext'), drop_reasons=row.get('drop_reasons'))


def run():
    started = time.monotonic()
    while not (FEAT / 'summary.md').exists():
        dump(OUT / 'g0-status.json', dict(status='waiting_for_seed0',wall_s=time.monotonic()-started))
        time.sleep(15)
    original = json.loads((FEAT / 'summary.json').read_text())
    if original['status'] != 'complete':
        raise ValueError('Seed-0 cache did not complete')
    cohorts = build()
    rows = cohorts['g0_probe'] + cohorts['panels'] + cohorts['extension']
    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r['audio_sha256']].append(r)
    index = json.loads((BULK / 'song-index-v0.json').read_text())
    songs, skipped = {}, []
    for i, (sha, rs) in enumerate(sorted(by_sha.items()), 1):
        try:
            songs[sha] = prepare_song(rs[0], index)
        except Exception as exc:
            if not all(r.get('ext') for r in rs):
                raise
            skipped.append(dict(sha256=sha, ext=[r['ext'] for r in rs], error=str(exc)))
        if i % 50 == 0:
            disk_guard()
            dump(OUT / 'g0-status.json', dict(status='preparing',done=i,total=len(by_sha),skipped=skipped))
            print(json.dumps(dict(event='prepare', done=i,total=len(by_sha))),flush=True)
    by_sha = {s:rs for s,rs in by_sha.items() if s in songs}
    dump(OUT / 'inference-songs.json', {s:asdict(v) for s,v in songs.items()})
    members = {s:[member(r) for r in rs] for s,rs in by_sha.items()}
    ext = sorted(s for s,rs in by_sha.items() if any(r.get('ext') for r in rs))
    panels = sorted({r['audio_sha256'] for r in cohorts['panels']})
    five = [panels[i] for i in np.linspace(0,len(panels)-1,5,dtype=int)]
    order = five + sorted(set(songs)-set(five))
    import torch
    from ensomi_n.v0.model import V0
    torch.set_num_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    os.nice(10)
    checks, records, cut = [], {0:[],1:[]}, []
    prep_s = time.monotonic() - started
    first_rate = None
    for seed, selected in [(1, order), (0, ext)]:
        out = FEAT / (f'v01-s{seed}-009001' + ('-ext' if seed == 0 else ''))
        out.mkdir(parents=True, exist_ok=True)
        if (out/'manifest.jsonl').exists():
            raise ValueError('Use a fresh G0 destination; existing manifest is not overwritten')
        ck = checkpoint(seed)
        state = torch.load(ck,map_location='cpu',weights_only=True)
        model = V0(**(state.get('model_kw') or {}))
        model.load_state_dict(state['model']);model.eval().to('mps')
        provenance = dict(checkpoint_id=f'v01-s{seed}-20261020/ckpt-009001',checkpoint_sha256=digest(ck),
                          script_sha256=digest(Path(__file__)),capture_sha256=digest(Path(__file__).with_name('feature_capture.py')),
                          decision_sha256=digest(OUT/'decision-exclusions.json'),t0_ms=0.,fps=100.,lead_rule='osu-lead-v1')
        infer_start = time.monotonic()
        with (out/'manifest.jsonl').open('w') as handle:
            for number, sha in enumerate(selected,1):
                if sha in cut:
                    continue
                song=songs[sha]
                if number%25 == 1:
                    disk_guard(song.frames*366*1.01)
                feat,logits,acts=capture_song(model,song,'mps')
                if feat.shape != (song.frames,160) or logits.shape != (song.frames,23):
                    raise ValueError('Feature clock or width mismatch')
                path=out/sha[:2]/f'{sha}.npz'
                write_npz(path,feat,logits)
                if seed==1 and sha in five:
                    checks.append(reproduce(sha,logits,acts,path,ROOT/'artifacts/astra-consult-20261021/astra/exp/evidence/s1/009001'))
                    dump(OUT/'g0-reproduction.json',checks)
                rs=by_sha[sha]
                kinds=sorted({r['ext'] for r in rs if r.get('ext')})
                rec=dict(sha256=sha,split=rs[0]['split'],ext=bool(kinds),ext_kinds=kinds,
                         frames=song.frames,duration_s=song.frames/100.,npz=str(path.relative_to(out)),
                         memberships=members[sha],set_ids=sorted({i for m in members[sha] for i in m['set_ids']}),
                         beatmap_ids=sorted({i for m in members[sha] for i in m['beatmap_ids']}),**provenance)
                handle.write(json.dumps(rec)+'\n');handle.flush();records[seed].append(rec)
                del feat,logits,acts
                if number%16==0:torch.mps.empty_cache()
                if seed==1 and number==50:
                    first_rate=(time.monotonic()-infer_start)/50
                    projected=prep_s+first_rate*(len(selected)+len(ext))
                    if projected>70*60:
                        cut=[s for s,rr in by_sha.items() if all(r['split']=='fit_train' for r in rr) and probe_bucket(s)==2]
                        if any(r['sha256'] in cut for r in records[seed]):
                            # Already computed rows remain available; the cut only removes pending work.
                            cut=[s for s in cut if s not in {r['sha256'] for r in records[seed]}]
                    dump(OUT/'g0-rate.json',dict(first50_s=first_rate*50,seconds_per_input=first_rate,
                                               projected_wall_s=projected,h2_pending_cut=len(cut)))
                if number%25==0 or number==len(selected):
                    status=dict(status='inference',seed=seed,done=len(records[seed]),total=len(selected)-len(cut)*(seed==1),wall_s=time.monotonic()-started)
                    dump(OUT/'g0-status.json',status);print(json.dumps(status),flush=True)
        del model,state
        torch.mps.empty_cache()
    summary=dict(status='complete',wall_s=time.monotonic()-started,prep_s=prep_s,first50_seconds_per_input=first_rate,
                 probe_h2_cut=len(cut),skipped=skipped,reproduction=checks,counts={},bytes=disk_guard())
    for seed,rs in records.items():
        summary['counts'][str(seed)]=dict(inputs=len(rs),splits=dict(Counter(r['split'] for r in rs)),
                                          extension=dict(Counter(k for r in rs for k in r['ext_kinds'])),
                                          audio_hours=sum(r['duration_s'] for r in rs)/3600,
                                          bytes=sum((FEAT/(f'v01-s{seed}-009001'+('-ext' if seed==0 else ''))/r['npz']).stat().st_size for r in rs))
    dump(OUT/'g0-results.json',summary)
    dump(FEAT/'v01-s1-009001/DONE',summary['counts'])
    (FEAT/'g0-summary.md').write_text('# G0 complete\n\n'+json.dumps({k:v for k,v in summary.items() if k!='reproduction'},indent=2)+'\n\nFive seed-1 E2 inputs reproduced; detailed maxima are in `../gpu/g0-reproduction.json`.\nThe authorised folder/group exclusion removed one original training/probe recording. Prior v0.1 checkpoint influence remains.\n')
    dump(OUT/'g0-status.json',dict(status='complete',wall_s=summary['wall_s']))


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        dump(OUT/'g0-status.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}'))
        raise
