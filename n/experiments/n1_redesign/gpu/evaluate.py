"""Whole-panel inference, CPU decode, pulse/structure scoring and matched new-head readouts."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import json
import multiprocessing as mp
import os
import time
import numpy as np
from scipy.special import expit,softmax
from scipy.stats import rankdata,spearmanr

from ensomi_n.v0.data import Song
from ensomi_n.v0.decode import decode,DecodeConfig
from ensomi_n.v0.evaluate import v0_evidence
from .cohorts import ROOT,OUT,FEAT,dump
from .cache import disk_guard
from .feature_capture import capture_song
from .finetune import Model
from .scoring import panel_items,score_maps
from .targets import chart_key,load_target,dense_new,nearest_indices
from . import base


def infer(name):
    import torch
    torch.set_num_threads(1);os.nice(10)
    state=torch.load(OUT/'runs'/name/'final.pt',map_location='cpu',weights_only=True)
    model=Model(state['floor'],state['treatment'],**state['model_kw'])
    model.load_state_dict(state['model']);model.eval().to('mps')
    fit=torch.load(OUT/'fits'/f"s{state['seed']}.pt",map_location='cpu',weights_only=True)['head']
    head=state['model'] if state['treatment'] else None
    w=(head['new.weight'] if head else fit['weight']).numpy()
    b=(head['new.bias'] if head else fit['bias']).numpy()
    songs=json.loads((OUT/'inference-songs.json').read_text())
    xs=panel_items();shas=sorted({x['item'] for x in xs})
    dest=OUT/'eval'/name;dest.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    for i,sha in enumerate(shas,1):
        song=Song(**songs[sha])
        feat,logits,acts=capture_song(model,song,'mps')
        new=feat@w.T+b
        np.savez_compressed(dest/f'{sha}.npz',**acts,new_logits=new,logits=logits.astype(np.float16),t0_ms=0.,fps=100.)
        if i%16==0:torch.mps.empty_cache()
        if i%25==0 or i==len(shas):
            disk_guard();print(json.dumps(dict(event='inference',run=name,done=i,total=len(shas),wall_s=time.monotonic()-started)),flush=True)
    dump(dest/'inference.json',dict(inputs=len(shas),wall_s=time.monotonic()-started,readout='trained new head' if state['treatment'] else 'frozen initialiser on F0 features'))


def decode_one(job):
    name,sha=job
    path=OUT/'eval'/name/f'{sha}.npz'
    with np.load(path,allow_pickle=False) as z:
        ev={k:z[k] for k in ('beat','half','down','beat_off','down_off','bound','level','phase','kappa','logp','logp_sigma','bar_phase','bar_kappa')}
    mmap,diag=decode(v0_evidence(ev),DecodeConfig(reset_penalty=14.))
    dump(OUT/'eval'/name/'maps'/f'{sha}.json',dict(map=mmap.to_json(),diag=diag))
    return sha,mmap.to_json()


def auc(y,score):
    y=np.asarray(y,bool);score=np.asarray(score,float)
    n=int(y.sum());m=len(y)-n
    if not n or not m:return None
    ranks=rankdata(score,method='average')
    return float((ranks[y].sum()-n*(n+1)/2)/(n*m))


def maxima(values,starts,ends):
    out=[]
    for a,b in zip(starts,ends):
        lo=max(0,int(np.ceil(a/10)));hi=min(len(values),int(np.ceil(b/10)))
        out.append(float(values[lo:hi].max()) if hi>lo else float('nan'))
    return np.asarray(out)


def variant_key(x):
    cs=[str(p)[str(p).index('dataset/'):] for p in x['ref'].meta['difficulties']]
    return chart_key(cs)


def head_rows(name,seed,frozen=False):
    import torch
    fit=torch.load(OUT/'fits'/f's{seed}.pt',map_location='cpu',weights_only=True)['head']
    w,b=fit['weight'].numpy(),fit['bias'].numpy()
    cache=FEAT/f'v01-s{seed}-009001'
    manifest={r['sha256']:r for r in map(json.loads,(cache/'manifest.jsonl').read_text().splitlines())}
    baseline=json.loads((OUT/'scores'/f'v01-s{seed}-009001.json').read_text())
    offsets={(r['panel'],r['rec']):r['offset_ms'] for r in baseline['recordings'] if r['cell']=='all'}
    rows=[];last_sha=None;y=None
    for x in sorted(panel_items(),key=lambda x:(x['item'],x['key'])):
        sha=x['item']
        if sha!=last_sha:
            if frozen:
                with np.load(cache/manifest[sha]['npz'],allow_pickle=False) as z:y=z['feat'].astype(np.float32)@w.T+b
            else:
                with np.load(OUT/'eval'/name/f'{sha}.npz',allow_pickle=False) as z:y=z['new_logits']
            last_sha=sha
        t=load_target(sha,variant_key(x));tg=dense_new(t,0,len(y));times=10.*np.arange(len(y))
        off=nearest_indices(times,t['lines'])[1]>30.
        mask=off&(tg['attack_w']>0)
        positive=mask&(tg['attack_y']>0);negative=mask&(tg['attack_y']==0)
        row=dict(key=x['key'],panel=x['panel'],rec=x['rec'],audio_sha256=sha,
                 attack_pos=int(positive.sum()),attack_neg=int(negative.sum()),attack_auc=None,
                 cell_auc_chart=None,cell_auc_decoded=None,salience_spearman=None)
        if positive.sum()>=50 and negative.sum()>=50:row['attack_auc']=auc(tg['attack_y'][mask]>0,y[mask,0])
        score=expit(y[:,0])*softmax(y[:,1:4],axis=-1)[:,2]
        use=t['cell']>=0
        classes=t['cell'][use]
        row['cell_binary']=int((classes==0).sum());row['cell_ternary']=int((classes==1).sum())
        if row['cell_binary']>=4 and row['cell_ternary']>=4:
            values=maxima(score,t['lines'][use],t['ends'][use])
            good=np.isfinite(values)
            row['cell_auc_chart']=auc(classes[good],values[good])
            mmap=base.MusicMap.from_json(base.saved(f's{seed}',sha))
            timing=mmap.timing(x['ref'].lo_ms-6000,x['ref'].hi_ms+6000)
            delta=offsets[(x['panel'],x['rec'])]
            pred=timing.beats-delta
            ri,pj=base.one_to_one(t['lines'][use],pred,15.)
            row['cell_unmatched']=int(use.sum()-len(ri));row['cell_total']=int(use.sum())
            end=np.minimum(np.r_[timing.beats[1:],timing.beats[-1]+timing.step_ms[-1]],timing.beats+timing.step_ms)
            values=maxima(score,timing.beats[pj],end[pj])
            good=np.isfinite(values)
            row['cell_auc_decoded']=auc(classes[ri][good],values[good])
        st=t['sal_times'];ix=np.round(st/10).astype(int);valid=(ix>=0)&(ix<len(y))
        if valid.sum()>=2:
            expected=expit(y[ix[valid],4:7]).sum(-1);r=t['retention'][valid]
            if np.ptp(expected)>0 and np.ptp(r)>0:row['salience_spearman']=float(spearmanr(expected,r).statistic)
        rows.append(row)
    dump(OUT/'heads'/f'{name}.json',dict(name=name,seed=seed,rows=rows))
    return rows


def score(name):
    xs=panel_items();shas=sorted({x['item'] for x in xs});maps={}
    started=time.monotonic()
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        for i,(sha,mmap) in enumerate(pool.map(decode_one,[(name,s) for s in shas]),1):
            maps[sha]=mmap
            if i%25==0:print(json.dumps(dict(event='decode',run=name,done=i,total=len(shas))),flush=True)
    score_maps(name,{x['key']:maps[x['item']] for x in xs})
    head_rows(name,int(name[-1]))
    dump(OUT/'eval'/name/'DONE.json',dict(status='complete',wall_s=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['infer','score','frozen']);p.add_argument('name');a=p.parse_args()
    if a.mode=='infer':infer(a.name)
    elif a.mode=='score':score(a.name)
    else:head_rows(f'frozen-s{a.name}',int(a.name),True)
