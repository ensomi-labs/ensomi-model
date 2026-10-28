"""Fold X5d initialisers and refit its linear control on streamed F0 features."""
import argparse
import json
import os
import time
import numpy as np
import torch
from numpy.lib.format import open_memmap
from .common import OUT,G1,X5D,BULK,ROOT,dump,digest,probe_bucket,disk_guard,check_prereg
from .targets import references
from .x5d_targets import targets
from .linear import fit_head,logits_for,temperature,nll
from ..finetune import Model,data
from ..feature_capture import capture_song


def fold(head,calibrated=True):
    scale=float(head['temperature']) if calibrated else 1.
    w=head['w0']/head['sd'][None,:]
    b=head['b0']-w@head['mean']
    return (w/scale).astype(np.float32),(b/scale).astype(np.float32)


def initialise():
    records={};(OUT/'initialisers').mkdir(parents=True,exist_ok=True)
    for seed in (0,1):
        path=X5D/f's{seed}/heads/linear.npz'
        if path.exists():
            with np.load(path) as z:h={k:z[k] for k in z.files}
            w,b=fold(h)
            x=np.random.default_rng(6).normal(size=(128,160)).astype(np.float32)*h['sd']+h['mean']
            expected=((x-h['mean'])/h['sd']@h['w0'].T+h['b0'])/h['temperature']
            error=float(abs((x@w.T+b)-expected).max())
            if not np.allclose(x@w.T+b,expected,atol=2e-5,rtol=2e-5):raise RuntimeError('Standardisation folding failed')
            records[str(seed)]=dict(source=str(path),sha256=digest(path),zero=False,
                                    temperature=float(h['temperature']),fold_max_abs=error)
        else:
            w=np.zeros((7,160),np.float32);b=np.zeros(7,np.float32)
            records[str(seed)]=dict(source=str(path),zero=True,reason='Saved X5d linear parameters absent')
        torch.save(dict(head=dict(weight=torch.from_numpy(w),bias=torch.from_numpy(b))),OUT/'initialisers'/f's{seed}.pt')
    dump(OUT/'initialisers.json',records)


def load_model(seed,arm):
    root=G1 if arm=='F0' else OUT
    state=torch.load(root/'runs'/f'{arm}-s{seed}'/'final.pt',map_location='cpu',weights_only=True)
    model=Model(state['floor'],state['treatment'],**state['model_kw'])
    model.load_state_dict(state['model']);model.eval().to('mps')
    return model,state


def samples(seed):
    torch.set_num_threads(1);os.nice(10)
    # Feature extraction is a control-input measurement; fits wait for the preregistration.
    c=json.loads((G1/'cohorts.json').read_text());songs,*_=data();by_sha={s.item:s for s in songs}
    selected={r['audio_sha256'] for r in c['g0_probe']}
    dest=OUT/'control'/f's{seed}'/'data';dest.mkdir(parents=True,exist_ok=False)
    model,_=load_model(seed,'F0');counts={};start=time.monotonic()
    for split,allowed in [('fit',(0,1)),('cal',(2,))]:
        shas=sorted(s for s in selected if probe_bucket(s) in allowed);capacity=512*len(shas)
        arrays={k:open_memmap(dest/f'{split}-{k}.npy',mode='w+',dtype=dtype,shape=(capacity,*shape))
                for k,dtype,shape in [('x','float16',(160,)),('y','float32',(7,)),('weight','float32',()),('owner','int32',())]}
        offset=0;inventory=[]
        for owner,sha in enumerate(shas):
            feat,_,_=capture_song(model,by_sha[sha],'mps')
            y,valid=targets(references(sha),10.*np.arange(len(feat)))
            ix=np.flatnonzero(valid);n=min(512,len(ix))
            if not n:raise RuntimeError('Empty X5d control target')
            take=ix[np.floor((np.arange(n)+.5)*len(ix)/n).astype(int)];sl=slice(offset,offset+n)
            arrays['x'][sl]=feat[take].astype(np.float16);arrays['y'][sl]=y[take]
            arrays['weight'][sl]=1./n;arrays['owner'][sl]=owner;offset+=n
            inventory.append(dict(audio_sha256=sha,samples=n,mapped_frames=len(ix)))
            del feat,y
            if (owner+1)%16==0:torch.mps.empty_cache()
            if (owner+1)%50==0:
                disk_guard();print(json.dumps(dict(event='F0_features',seed=seed,split=split,done=owner+1,total=len(shas),wall_s=time.monotonic()-start)),flush=True)
        for a in arrays.values():a.flush()
        counts[split]=dict(recordings=len(shas),samples=offset)
        dump(dest/f'{split}-inventory.json',inventory)
    xx=np.load(dest/'fit-x.npy',mmap_mode='r');ww=np.load(dest/'fit-weight.npy',mmap_mode='r')
    total=0.;first=np.zeros(160);second=np.zeros(160);n=counts['fit']['samples']
    for lo in range(0,n,16384):
        x=np.asarray(xx[lo:min(n,lo+16384)],np.float64);w=np.asarray(ww[lo:min(n,lo+16384)],np.float64)
        total+=w.sum();first+=(w[:,None]*x).sum(0);second+=(w[:,None]*x*x).sum(0)
    mean=first/total;sd=np.maximum(np.sqrt(np.maximum(second/total-mean*mean,0)),1e-4)
    np.savez(dest/'normalizer.npz',mean=mean.astype(np.float32),sd=sd.astype(np.float32))
    dump(dest/'metadata.json',dict(seed=seed,counts=counts,wall_s=time.monotonic()-start,feature_dtype='float16'))


def fit(seed):
    check_prereg();torch.set_num_threads(1);os.nice(10)
    dest=OUT/'control'/f's{seed}';path=dest/'data';meta=json.loads((path/'metadata.json').read_text())
    ds={s:{k:np.load(path/f'{s}-{k}.npy',mmap_mode='r')[:meta['counts'][s]['samples']]
           for k in ('x','y','weight','owner')} for s in ('fit','cal')}
    with np.load(path/'normalizer.npz') as z:mean,sd=z['mean'],z['sd']
    start=time.monotonic();model,info=fit_head('linear',ds['fit'],ds['cal'],mean,sd,20261029+seed)
    logits=logits_for(model,ds['cal']['x'],mean,sd);temp=temperature(logits,ds['cal']['y'],ds['cal']['weight'])
    np.savez(dest/'linear.npz',mean=mean,sd=sd,temperature=temp,w0=model[0].weight.detach().numpy(),b0=model[0].bias.detach().numpy())
    info.update(seed=seed,temperature=temp,wall_s=time.monotonic()-start,cohort=meta,
                calibrated_nll=float(np.average(nll(logits,ds['cal']['y'],temp),weights=ds['cal']['weight'])))
    dump(dest/'fit.json',info)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialise','samples','fit']);p.add_argument('seed',type=int,nargs='?');a=p.parse_args()
    if a.mode=='initialise':initialise()
    elif a.mode=='samples':samples(a.seed)
    else:fit(a.seed)
