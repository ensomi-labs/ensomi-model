"""Seed-specific frozen linear initialisers fitted only on audited h=0,1 recordings."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ensomi_n.v0.labels import V0Labels,dense_v0
from .cohorts import OUT,FEAT,BULK,dump
from .targets import load_target,dense_new
from .audit_inputs import probe_bucket,digest


def fit(seed):
    torch.set_num_threads(1)
    while not (OUT/'target-rates.json').exists():time.sleep(5)
    if seed==1:
        while not (FEAT/'v01-s1-009001/DONE').exists():time.sleep(5)
    rows=json.loads((OUT/'cohorts.json').read_text())['training']
    rows=[r for r in rows if probe_bucket(r['audio_sha256']) in (0,1)]
    cache=FEAT/f'v01-s{seed}-009001'
    manifest={r['sha256']:r for r in map(json.loads,(cache/'manifest.jsonl').read_text().splitlines())}
    rng=np.random.default_rng(3000+seed)
    xs=[];ys={k:[] for k in ('attack_y','attack_w','snap_y','snap_w','sal_y','sal_w')}
    started=time.monotonic()
    for number,row in enumerate(rows,1):
        sha=row['audio_sha256'];mr=manifest[sha]
        with np.load(cache/mr['npz'],allow_pickle=False) as z:
            feat=z['feat']
        tg=dense_new(load_target(sha),0,len(feat))
        lab=V0Labels.from_json(json.loads((BULK/'labels-v0b'/sha[:2]/f'{sha}.json').read_text()))
        tg['attack_w']*=dense_v0(lab,0,len(feat))['weight']
        eligible=np.flatnonzero((tg['attack_w']>0)|(tg['snap_w']>0)|(tg['sal_w']>0))
        ix=rng.choice(eligible,min(len(eligible),512),replace=False)
        xs.append(feat[ix])
        for k in ys:ys[k].append(tg[k][ix])
        if number%200==0:print(json.dumps(dict(event='fit_data',seed=seed,done=number,total=len(rows))),flush=True)
    x=torch.from_numpy(np.concatenate(xs).astype(np.float32));del xs
    tg={k:torch.from_numpy(np.concatenate(v)) for k,v in ys.items()};del ys
    pw=json.loads((OUT/'target-rates.json').read_text())['attack_pos_weight']
    head=nn.Linear(160,7);nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    opt=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=120,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-6)
    calls=0;first=None;last=None;fit_start=time.monotonic()
    def weighted(v,w):return (v*w).sum()/w.sum().clamp_min(1.)
    def closure():
        nonlocal calls,first,last
        if time.monotonic()-fit_start>600:raise TimeoutError('Linear fit exceeded ten optimisation minutes')
        opt.zero_grad();y=head(x)
        attack=weighted(F.binary_cross_entropy_with_logits(y[:,0],tg['attack_y'],reduction='none',pos_weight=torch.tensor(pw)),tg['attack_w'])
        snap=weighted(F.cross_entropy(y[:,1:4],tg['snap_y'],reduction='none'),tg['snap_w'])
        sal=weighted(F.binary_cross_entropy_with_logits(y[:,4:7],tg['sal_y'],reduction='none').mean(-1),tg['sal_w'])
        loss=.5*attack+.5*snap+.25*sal
        if not torch.isfinite(loss):raise ValueError('Non-finite linear fit loss')
        loss.backward();calls+=1;last=float(loss.detach())
        if first is None:first=last
        return loss
    failure=None
    try:opt.step(closure)
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    dest=OUT/'fits';dest.mkdir(exist_ok=True)
    info=dict(seed=seed,recordings=len(rows),frames=len(x),sample_cap_per_recording=512,optimiser='LBFGS',max_iter=120,
              closure_calls=calls,initial_loss=first,final_loss=last,failed=failure,zero_initialised=bool(failure),
              wall_s=time.monotonic()-started,optimisation_s=time.monotonic()-fit_start,
              decision_sha256=digest(OUT/'decision-exclusions.json'))
    torch.save(dict(head=head.state_dict(),seed=seed),dest/f's{seed}.pt')
    dump(dest/f's{seed}.json',info)
    print(json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();fit(a.seed)
