"""Single-job v0.1 fine-tunes with unchanged sampling/probes and explicit auxiliary heads."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ensomi_n.v0.model import V0, Heads
from ensomi_n.v0 import train as T
from ensomi_n.v0.data import Sampler, load_songs, fixed_crops, to_torch
from ensomi_n.labels import read_set
from ensomi_n import probe
from .cohorts import ROOT, OUT, BULK, RUNS, dump
from .cache import checkpoint, disk_guard
from .targets import dense_new, load_target, NEW_KEYS
from .audit_inputs import digest


class FloorHeads(Heads):
    def __init__(self,d,floor):
        super().__init__(d);self.floor=floor

    def forward(self,h):
        out=super().forward(h)
        out['logp_sigma']=out['logp_sigma']+(self.floor-.001)
        return out


class Model(V0):
    def __init__(self, floor, treatment=False, **kw):
        super().__init__(**kw)
        self.heads=FloorHeads(self.encoder.d,floor)
        self.new=nn.Linear(self.encoder.d,7) if treatment else None


class TargetSampler(Sampler):
    def crop(self,song,rate,start_ms):
        x,tg=super().crop(song,rate,start_ms)
        self.new_targets.append(dense_new(load_target(song.item),start_ms,len(x),rate))
        return x,tg

    def batch(self,n_songs):
        self.new_targets=[]
        x,tg,memo,mask,owner=super().batch(n_songs)
        tg.update({k:np.stack([r[k] for r in self.new_targets]) for k in NEW_KEYS})
        return x,tg,memo,mask,owner


def auxiliary_loss(y,tg,pos_weight,break_weight=None):
    def mean(v,w):return ((v*w).sum(1)/w.sum(1).clamp_min(1.)).mean()
    w=tg['attack_w']
    if break_weight is not None:w=w*break_weight
    attack=mean(F.binary_cross_entropy_with_logits(y[...,0],tg['attack_y'],reduction='none',
                                                  pos_weight=torch.as_tensor(pos_weight,device=y.device)),w)
    snap=mean(F.cross_entropy(y[...,1:4].transpose(1,2),tg['snap_y'],reduction='none'),tg['snap_w'])
    sal=mean(F.binary_cross_entropy_with_logits(y[...,4:7],tg['sal_y'],reduction='none').mean(-1),tg['sal_w'])
    return .5*attack+.5*snap+.25*sal,dict(attack=float(attack.detach()),snap=float(snap.detach()),salience=float(sal.detach()))


def data():
    cohorts=json.loads((OUT/'cohorts.json').read_text())
    index=json.loads((BULK/'song-index-v0.json').read_text())
    songs=load_songs(cohorts['training'],index,BULK,BULK/'labels-v0b')
    if len(songs)!=len(cohorts['training']):raise ValueError('Filtered training loader dropped a recording')
    rows=sorted(cohorts['val_random'],key=lambda r:r['item'])
    psongs=load_songs(rows,index,BULK,BULK/'labels-v0b')[:150]
    crops=fixed_crops(psongs,per_song=1)
    by_item={r['item']:r for r in rows}
    items=[];counts={'S0':0,'hard':0}
    for s in psongs:
        r=by_item[s.item]
        group='S0' if 'S0' in r.get('strata',[]) else ('hard' if r.get('hard') else None)
        if group is None or counts[group]>=40:continue
        ref=read_set([ROOT/c for c in r['charts']],key=s.item)
        items.append(dict(song=s,group=group,item=s.item,windows=probe.beat_windows(ref)))
        counts[group]+=1
    if counts!={'S0':40,'hard':40}:raise ValueError(f'Probe cohort changed: {counts}')
    refs=json.loads((RUNS/'v01-s0-20261020/refs.json').read_text())
    return songs,psongs,crops,items,refs


def baseline_probes():
    torch.set_num_threads(1);os.nice(10)
    songs,psongs,crops,items,refs=data()
    result={}
    for seed in (0,1):
        saved=torch.load(checkpoint(seed),map_location='cpu',weights_only=True)
        model=V0(**(saved.get('model_kw') or {}));model.load_state_dict(saved['model']);model.to('mps')
        lv,_,pv=T.probe_loss(model,crops,psongs,refs['pos_weight'],'mps')
        m1=T.probe_m1(model,items,'mps')
        result[str(seed)]=dict(loss=lv,m1=m1,parts={k:float(v.mean()) for k,v in pv.items()})
        del model,saved;torch.mps.empty_cache()
        dump(OUT/'baseline-probes.json',result)
    return result


def run(seed,treatment):
    torch.set_num_threads(1);os.nice(10)
    name=f'F{int(treatment)}-s{seed}';run_dir=OUT/'runs'/name
    if run_dir.exists():raise ValueError('Each run requires a fresh directory; no resume is allowed')
    prereg=json.loads((OUT/'prereg-lock.json').read_text())
    if digest(OUT/'prereg.md')!=prereg['sha256']:raise ValueError('Preregistration changed after freezing')
    floor=json.loads((OUT/'sigma-baseline.json').read_text())['floor']
    pw=json.loads((OUT/'target-rates.json').read_text())['attack_pos_weight']
    baseline=json.loads((OUT/'baseline-probes.json').read_text())[str(seed)]
    songs,psongs,crops,items,refs=data()
    run_dir.mkdir(parents=True)
    t_job=time.monotonic()
    torch.manual_seed(2000+seed);np.random.seed(2000+seed)
    state=torch.load(checkpoint(seed),map_location='cpu',weights_only=True)
    model=Model(floor,treatment,**(state.get('model_kw') or {}))
    missing,unexpected=model.load_state_dict(state['model'],strict=False)
    if set(missing)!=( {'new.weight','new.bias'} if treatment else set()) or unexpected:raise ValueError('Checkpoint schema changed')
    if treatment:
        fit=torch.load(OUT/'fits'/f's{seed}.pt',map_location='cpu',weights_only=True)
        model.new.load_state_dict(fit['head'])
    model.to('mps')
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    sampler=(TargetSampler if treatment else Sampler)(songs,seed=2000+seed,crops_per_song=2)
    pre=T.Prefetch(sampler,8)
    history=[];elapsed=0.;step=0;nonfinite=0;last_probe=0;last_ck=0;ref_rate=None;slow_since=None
    win_t=time.monotonic();win_n=0;losses=[];parts={};verdict=None;final_parts={}
    def lr():
        if step<200:return 3e-4*(step+1)/200
        return 3e-4*(.05+.95*.5*(1+np.cos(np.pi*min(elapsed/2400,1))))
    def write(rec):
        with (run_dir/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(rec,utc_s=time.time()))+'\n')
        print(json.dumps(rec),flush=True)
    def save():
        path=run_dir/'final.pt'
        torch.save(dict(model=model.state_dict(),model_kw=state.get('model_kw') or {},floor=floor,
                        treatment=treatment,seed=seed,steps=step,elapsed_s=elapsed,history=history,
                        prereg_sha256=prereg['sha256']),path)
        return path
    dump(run_dir/'config.json',dict(seed=seed,sampler_seed=2000+seed,torch_seed=2000+seed,treatment=treatment,
                                   sigma_floor=floor,training_recordings=len(songs),training_minutes=40,
                                   peak_lr=3e-4,warmup_steps=200,prereg_sha256=prereg['sha256']))
    model.train()
    try:
        while elapsed<2400:
            t0=time.monotonic()
            x,tg,memo,mask,owner=pre.get()
            x,tg,memo,mask=to_torch(x,tg,memo,mask,'mps')
            for g in opt.param_groups:g['lr']=lr()
            mem=model.encoder.song_memory(memo,mask)[torch.from_numpy(owner).to('mps')]
            h=model.encoder(x,mem);out=model.heads(h)
            loss,p=V0.loss(out,tg,refs['pos_weight'])
            if treatment:
                aux,ap=auxiliary_loss(model.new(h),tg,pw,tg['weight']);loss=loss+aux;p.update(ap)
            if not torch.isfinite(loss):
                nonfinite+=1;opt.zero_grad(set_to_none=True);write(dict(event='nonfinite_loss',step=step))
                if nonfinite>=2:verdict='killed:nonfinite_loss';break
                continue
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            torch.mps.synchronize();step+=1
            if step%T.EMPTY_CACHE_EVERY==0:torch.mps.empty_cache()
            elapsed+=time.monotonic()-t0
            win_n+=len(x);losses.append(float(loss.detach()));final_parts=p
            for k,v in p.items():parts.setdefault(k,[]).append(v)
            now=time.monotonic()
            if now-win_t>=60:
                rate=win_n/(now-win_t);driver,allocated=T._mps_gb()
                write(dict(event='progress',step=step,elapsed_s=elapsed,loss=float(np.mean(losses)),
                           parts={k:float(np.mean(v)) for k,v in parts.items()},lr=lr(),crops_per_s=rate,
                           mps_driver_gb=driver,mps_alloc_gb=allocated))
                win_t=now;win_n=0;losses=[];parts={};disk_guard()
                if allocated>T.MEM_LIMIT_GB or driver>T.DRIVER_LIMIT_GB:verdict='guard:mps_memory';break
                if ref_rate is None and step>200:ref_rate=rate
                if ref_rate and rate<.5*ref_rate:
                    slow_since=slow_since or now
                    if now-slow_since>=180:verdict='guard:throughput';break
                else:slow_since=None
            if elapsed//240>last_probe or elapsed>=2400:
                last_probe=elapsed//240;te=time.monotonic()
                lv,_,pv=T.probe_loss(model,crops,psongs,refs['pos_weight'],'mps')
                m1=T.probe_m1(model,items,'mps');torch.mps.empty_cache()
                entry=dict(step=step,minute=round(elapsed/60,2),probe_loss=round(lv,5),
                           probe_parts={k:round(float(v.mean()),4) for k,v in pv.items()},
                           m1={k:round(v,4) for k,v in m1.items()},probe_s=time.monotonic()-te)
                history.append(entry)
                verdict=T.kill_verdict(history,refs,lr()/3e-4)
                if len(history)>=3 and all(hh['m1']['S0']<.9*baseline['m1']['S0'] for hh in history[-3:]):
                    verdict='killed:s0_m1_below_90pct_v01_three_probes'
                write(dict(event='probe',**entry,verdict=verdict))
                if verdict:break
            if elapsed//600>last_ck:
                last_ck=elapsed//600;save()
    finally:
        pre.close()
    path=save()
    result=dict(name=name,status=verdict or 'complete',steps=step,training_s=elapsed,wall_s=time.monotonic()-t_job,
                kills=[verdict] if verdict and verdict.startswith('killed:') else [],final_losses=final_parts,
                sigma_floor=floor,history=history,checkpoint=str(path),prereg_sha256=prereg['sha256'])
    dump(run_dir/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--treatment',action='store_true');p.add_argument('--baseline',action='store_true')
    a=p.parse_args()
    if a.baseline:baseline_probes()
    else:run(a.seed,a.treatment)
