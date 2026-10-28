"""Panel inference and paired D2 arms with the copied stage-0 and X5d scorecards."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import time
import numpy as np
import torch
from scipy.special import softmax
from ensomi_n.v0.data import Song
from ensomi_n.v0.evaluate import v0_evidence
from .common import OUT,G1,ROOT,dump,disk_guard,check_prereg
from .readout import load_model,fold
from .bands import decode_band,CFG
from .level import dec,decode_level
from .accuracy import accuracy
from .structure import structural
from .structural_rows import structural_rows
from .. import base
from ..scoring import panel_items,recording_rows
from ..feature_capture import capture_song

ARMS=('F2-base','F2-own-D2','F2-band-D2','F0-frozen-band-D2')


def infer(seed,arm):
    check_prereg();torch.set_num_threads(1);os.nice(10)
    model,state=load_model(seed,arm)
    if arm=='F0':
        with np.load(OUT/'control'/f's{seed}'/'linear.npz') as z:w,b=fold({k:z[k] for k in z.files})
    else:w,b=model.new.weight.detach().cpu().numpy(),model.new.bias.detach().cpu().numpy()
    songs=json.loads((G1/'inference-songs.json').read_text());dest=OUT/'eval'/f'{arm}-s{seed}';dest.mkdir(parents=True,exist_ok=False)
    shas=sorted({x['item'] for x in panel_items()});start=time.monotonic();repro=[]
    for i,sha in enumerate(shas,1):
        feat,_,acts=capture_song(model,Song(**songs[sha]),'mps')
        band=feat@w.T+b
        if arm=='F0' and i<=5:
            with np.load(G1/'eval'/f'F0-s{seed}'/f'{sha}.npz') as z:
                error=max(float(abs(acts[k]-z[k]).max()) for k in acts)
            if error>1e-6:raise RuntimeError('F0 evidence reproduction failed')
            repro.append(dict(audio_sha256=sha,max_abs=error))
        np.savez_compressed(dest/f'{sha}.npz',band_logits=band,t0_ms=0.,fps=100.,**(acts if arm=='F2' else {}))
        if i%16==0:torch.mps.empty_cache()
        if i%25==0 or i==len(shas):
            disk_guard();print(json.dumps(dict(event='inference',seed=seed,arm=arm,done=i,total=len(shas),wall_s=time.monotonic()-start)),flush=True)
    dump(dest/'inference.json',dict(inputs=len(shas),wall_s=time.monotonic()-start,reproduction=repro))


def evidence(path):
    with np.load(path) as z:
        ev=v0_evidence({k:z[k] for k in ('beat','half','down','beat_off','down_off','bound','level','phase','kappa','logp','logp_sigma','bar_phase','bar_kappa')})
    return ev


def structures_for(seed,arm,xs,maps):
    offset,_=base.global_offset(xs,maps);rows=[]
    for x in xs:
        m=base.MusicMap.from_json(maps[x['key']])
        for aligned,off in [(True,offset),(False,0.)]:
            rows.append(dict(seed=f's{seed}',arm=arm,panel=x['panel'],rec=x['rec'],key=x['key'],
                             strata=x['strata'],aligned=aligned,**structural(x['ref'],m,off)))
    return rows


def add_structure(rows,structures,seed,arm):
    for cell in ('all','S0','S5'):
        ss=structural_rows(structures,cell)
        for aligned,metric in [(True,'structural_ready'),(False,'raw_structural_ready')]:
            lookup={(r['panel'],r['rec']):r['ready'] for r in ss if r['aligned']==aligned and r['arm']==f's{seed}:{arm}'}
            for r in rows:
                if r['cell']==cell:r[metric]=lookup.get((r['panel'],r['rec']),float('nan'))
    return rows


def groups():
    g=defaultdict(list)
    for x in panel_items():g[(x['panel'],x['rec'])].append(x)
    return list(g.values())


def baseline_job(args):
    name,seed,ck,xs=args
    if name=='F0-base':
        maps={x['key']:json.loads((G1/'eval'/f'F0-s{seed}'/'maps'/f"{x['item']}.json").read_text())['map'] for x in xs}
    else:
        maps={x['key']:json.loads((base.PREV/'e2-maps'/f's{seed}'/ck/f"{x['item']}.json").read_text())['full'] for x in xs}
    return structures_for(seed,name,xs,maps)


def baseline():
    gg=groups()
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        for seed,cks in [(0,('009001','008137','008400')),(1,('009001','007973','008401'))]:
            for ck in (*cks,'F0'):
                arm='F0-base' if ck=='F0' else f'v01-{ck}'
                source=f'F0-s{seed}' if ck=='F0' else f'v01-s{seed}-{ck}'
                rows=json.loads((G1/'scores'/f'{source}.json').read_text())['recordings']
                ss=[]
                for result in pool.map(baseline_job,[(arm,seed,ck,g) for g in gg]):ss+=result
                rows=add_structure(rows,ss,seed,arm)
                base.dump(OUT/'baseline'/f'{arm}-s{seed}.json',dict(recordings=rows,structure=ss))
                print(json.dumps(dict(event='baseline',arm=arm,seed=seed,recordings=412)),flush=True)
    dump(OUT/'baseline-DONE.json',dict(status='complete'))


def score_job(args):
    seed,xs=args;maps={a:{} for a in ARMS};acc=[];diagnostics=[];seen={}
    for x in xs:
        sha=x['item']
        if sha not in seen:
            ev=evidence(OUT/'eval'/f'F2-s{seed}'/f'{sha}.npz')
            original=dec._track;capture={}
            def track(e,c):capture['track']=original(e,c);return tuple(a.copy() for a in capture['track'])
            dec._track=track
            try:m,d=dec.decode(ev,CFG)
            finally:dec._track=original
            local={'F2-base':m.to_json()};diagnostics.append(dict(audio=sha,arm='F2-base',**d))
            m,d=decode_level(ev,None,'own_d2',capture['track'],CFG);local['F2-own-D2']=m.to_json()
            with np.load(OUT/'eval'/f'F2-s{seed}'/f'{sha}.npz') as z:p2=softmax(z['band_logits'].astype(np.float64),axis=1)
            m,d=decode_band(ev,p2,capture['track'],CFG);local['F2-band-D2']=m.to_json()
            ev0=evidence(G1/'eval'/f'F0-s{seed}'/f'{sha}.npz');t0=dec._track(ev0,CFG)
            with np.load(OUT/'eval'/f'F0-s{seed}'/f'{sha}.npz') as z:p0=softmax(z['band_logits'].astype(np.float64),axis=1)
            m,d=decode_band(ev0,p0,t0,CFG);local['F0-frozen-band-D2']=m.to_json()
            seen[sha]=(local,p2,p0)
        local,p2,p0=seen[sha]
        for arm in ARMS:maps[arm][x['key']]=local[arm]
        for arm,prob in [('F2-band-D2',p2),('F0-frozen-band-D2',p0)]:
            for cell,v in accuracy(x['ref'],10.*np.arange(len(prob)),prob,x['strata']).items():
                acc.append(dict(seed=seed,arm=arm,panel=x['panel'],rec=x['rec'],key=x['key'],cell=cell,**v))
    rows={};ss={}
    for arm in ARMS:
        rows[arm]=base.score_record((arm,xs,maps[arm]));ss[arm]=structures_for(seed,arm,xs,maps[arm])
    key=hashlib.sha256((xs[0]['panel']+':'+xs[0]['rec']).encode()).hexdigest()[:20]
    path=OUT/'eval/maps'/f's{seed}'/f'{key}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:json.dump(dict(maps=maps,diagnostics=diagnostics),f,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
    return dict(rows=rows,structures=ss,accuracy=acc)


def score(seed):
    check_prereg();start=time.monotonic();jobs=[]
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        for i,r in enumerate(pool.map(score_job,[(seed,g) for g in groups()]),1):
            jobs.append(r)
            if i%25==0:print(json.dumps(dict(event='score',seed=seed,done=i,total=412,wall_s=time.monotonic()-start)),flush=True)
    for arm in ARMS:
        rr=recording_rows([r for j in jobs for r in j['rows'][arm]])
        ss=[r for j in jobs for r in j['structures'][arm]]
        add_structure(rr,ss,seed,arm)
        base.dump(OUT/'scores'/f'{arm}-s{seed}.json',dict(recordings=rr,structure=ss))
    base.dump(OUT/'scores'/f'accuracy-s{seed}.json',[r for j in jobs for r in j['accuracy']])
    dump(OUT/f'evaluation-s{seed}-DONE.json',dict(status='complete',recordings=412,wall_s=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','infer','score']);p.add_argument('seed',type=int,nargs='?');p.add_argument('--arm',choices=['F0','F2']);a=p.parse_args()
    if a.mode=='baseline':baseline()
    elif a.mode=='infer':infer(a.seed,a.arm)
    else:score(a.seed)
