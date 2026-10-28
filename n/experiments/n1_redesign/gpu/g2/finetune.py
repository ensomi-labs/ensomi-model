"""F0 single-job recipe with the G2 absolute-band loss and final-only weights."""
import json
import os
import time
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from ensomi_n.v0.model import V0
from ensomi_n.v0 import train as T
from ensomi_n.v0.data import Sampler,to_torch
from ..finetune import Model,data
from ..cache import checkpoint
from ..audit_inputs import digest
from .common import OUT,G1,dump,disk_guard,check_prereg
from .targets import dense_band

class TargetSampler(Sampler):
    def crop(self,song,rate,start_ms):
        x,tg=super().crop(song,rate,start_ms)
        self.band_targets.append(dense_band(song.item,start_ms,len(x),rate))
        return x,tg
    def batch(self,n_songs):
        self.band_targets=[]
        x,tg,memo,mask,owner=super().batch(n_songs)
        tg.update({k:np.stack([r[k] for r in self.band_targets]) for k in ('band_y','band_w')})
        return x,tg,memo,mask,owner

def auxiliary_loss(y,tg,pos_weight=None,break_weight=None):
    w=tg['band_w']*break_weight
    ce=-(tg['band_y']*F.log_softmax(y,dim=-1)).sum(-1)
    loss=((ce*w).sum(1)/w.sum(1).clamp_min(1.)).mean()
    return .5*loss,dict(band=float(loss.detach()))

def run(seed,treatment):
    torch.set_num_threads(1);os.nice(10)
    name=f'F2-s{seed}';run_dir=OUT/'runs'/name
    disk_guard(reserve=max(0,6e9-disk_guard()['new_bytes']))
    if run_dir.exists():raise ValueError('Each run requires a fresh directory; no resume is allowed')
    prereg=check_prereg()
    floor=json.loads((G1/'sigma-baseline.json').read_text())['floor']
    pw=None
    baseline=json.loads((G1/'baseline-probes.json').read_text())[str(seed)]
    songs,psongs,crops,items,refs=data()
    run_dir.mkdir(parents=True)
    t_job=time.monotonic()
    torch.manual_seed(2000+seed);np.random.seed(2000+seed)
    state=torch.load(checkpoint(seed),map_location='cpu',weights_only=True)
    model=Model(floor,treatment,**(state.get('model_kw') or {}))
    missing,unexpected=model.load_state_dict(state['model'],strict=False)
    if set(missing)!=( {'new.weight','new.bias'} if treatment else set()) or unexpected:raise ValueError('Checkpoint schema changed')
    if treatment:
        fit=torch.load(OUT/'initialisers'/f's{seed}.pt',map_location='cpu',weights_only=True)
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
    finally:
        pre.close()
    path=save()
    result=dict(name=name,status=verdict or 'complete',steps=step,training_s=elapsed,wall_s=time.monotonic()-t_job,
                kills=[verdict] if verdict and verdict.startswith('killed:') else [],final_losses=final_parts,
                sigma_floor=floor,history=history,checkpoint=str(path),prereg_sha256=prereg['sha256'])
    dump(run_dir/'result.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);a=p.parse_args()
    run(a.seed,True)
