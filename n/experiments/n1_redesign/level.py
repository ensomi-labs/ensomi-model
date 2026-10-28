"""Experimental octave-band level selection and parity projection on fixed tracks."""
import dataclasses
import importlib
import math
import numpy as np
from scipy.special import ndtr

dec=importlib.import_module('ensomi_n.v0.decode')


def band_ratios(ev,groups,periods,cfg):
    """Integrate the level/period posterior, then map the tracked pulse to its band."""
    if ev.level is None or ev.logp is None:
        raise ValueError('D2 requires both level and period heads')
    bands=np.arange(-2,5)
    logbpm=np.log(60.)-ev.logp
    fold=np.floor((logbpm-np.log(80.))/np.log(2.))
    canonical=logbpm-fold*np.log(2.)
    sigma=np.maximum(ev.logp_sigma if ev.logp_sigma is not None else .05,.001)
    probs=np.zeros((ev.n_frames,len(bands)))
    lev=ev.level/np.maximum(ev.level.sum(1,keepdims=True),1e-12)
    for k,mult in enumerate(dec.LEVEL_CLASSES):
        mu=canonical+np.log(mult)
        low=np.log(80.)+bands*np.log(2.)
        high=low+np.log(2.)
        mass=ndtr((high[None,:]-mu[:,None])/sigma[:,None])-ndtr((low[None,:]-mu[:,None])/sigma[:,None])
        probs+=lev[:,k,None]*mass
    means=[];durs=[]
    for f0,f1,_ in groups:
        a=int(np.clip(f0,0,ev.n_frames-1));b=int(np.clip(f1,a+1,ev.n_frames))
        means.append(np.log(probs[a:b]+cfg.level_eps).mean(0));durs.append(b-a)
    means=np.asarray(means)
    song=int(np.argmax((means*np.array(durs)[:,None]).sum(0)))
    ratios=[]
    for m,period,(_,_,free) in zip(means,periods,groups):
        scores=m.copy();scores[song]+=cfg.level_prior
        choices=[]
        for k in range(0 if free else -2,4):
            bpm=60000./period*2.**k
            band=int(np.floor(np.log2(bpm/80.)))
            where=np.flatnonzero(bands==band)
            if len(where):choices.append((float(scores[where[0]]),-abs(k),k))
        ratios.append(2.**max(choices)[2])
    return ratios


def project(ev,regs,t,conf,cfg,mode):
    canon=t.copy()
    for r in regs:
        if r.kind=='constant':canon[r.i0:r.i1]=r.a+r.L*np.arange(r.i1-r.i0)
    groups=[];members=[];periods=[]
    k=0
    while k<len(regs):
        r=regs[k];j=k+1
        if r.kind=='free':
            while j<len(regs) and regs[j].kind=='free':j+=1
        i0,i1=regs[k].i0,regs[j-1].i1
        period=r.L if r.kind=='constant' else (float(np.median(np.diff(canon[i0:i1+1]))) if i1>i0+1 else 500.)
        end=canon[i1] if i1<len(canon) else canon[i1-1]+period
        groups.append((int(dec._frame(canon[i0])),int(dec._frame(end)),r.kind=='free'))
        members.append((k,j));periods.append(period);k=j
    ratios=band_ratios(ev,groups,periods,cfg) if mode=='d2' else dec._levels(ev,groups,cfg)
    segs=[];nb=len(canon)
    for (k,j),ratio in zip(members,ratios):
        r=regs[k]
        if r.kind=='constant':
            i0,i1=r.i0,r.i1;nxt=canon[i1] if i1<nb else None
            c=canon[i0:i1];cc=conf[i0:i1]
            if ratio>=1:
                q=int(round(ratio));step=r.L/q
                n=(len(c)-1)*q+1
                if nxt is not None:n=max(n,int(math.floor((nxt-.5*step-r.a)/step+1e-9))+1)
                jj=np.arange(n);bt=r.a+jj*step
                segs.append(dec._Seg(float(r.a),float(step),list(bt),list(cc[np.minimum(jj//q,len(cc)-1)]),ratio,False,True))
            else:
                stride=int(round(1/ratio));parity=0
                d=dec._down_at(ev,c,cfg)
                if d is not None and len(c)>1:
                    reducer=np.mean if mode=='d4' else np.sum
                    scores=[float(reducer(d[p::stride])) if len(d[p::stride]) else -np.inf for p in range(stride)]
                    parity=int(np.argmax(scores))
                step=stride*r.L;start=r.a+parity*r.L;first=True
                if parity:
                    # Preserve the prefix at the tracked level, as the baseline does for odd parity.
                    segs.append(dec._Seg(float(r.a),float(r.L),list(c[:parity]),list(cc[:parity]),1.,False,True));first=False
                n=max((len(c)-1-parity)//stride+1,1)
                if nxt is not None:n=max(n,int(math.floor((nxt-.5*step-start)/step+1e-9))+1)
                bt=start+np.arange(n)*step;idx=np.minimum(parity+stride*np.arange(n),len(cc)-1)
                segs.append(dec._Seg(float(start),float(step),list(bt),list(cc[idx]),ratio,False,first))
        else:
            q=int(round(ratio))
            for ii,rr in enumerate(regs[k:j]):
                i=rr.i0
                gap=canon[i+1]-canon[i] if i+1<nb else (canon[i]-canon[i-1] if i>0 else 500.)
                step=gap/q;bt=canon[i]+step*np.arange(q if i+1<nb else 1)
                segs.append(dec._Seg(float(canon[i]),float(step),list(bt),[float(conf[i])]*len(bt),ratio,True,ii==0))
    return segs,canon,ratios


def decode_level(ev,oracle,arm,track,cfg):
    e=dataclasses.replace(ev,level=oracle.level) if arm in ('oracle_d2','oracle_original') else ev
    oldtrack,oldnotated=dec._track,dec._notated
    dec._track=lambda e,c:tuple(a.copy() for a in track)
    if arm!='oracle_original':
        mode='d4' if arm=='d4' else 'd2'
        dec._notated=lambda e,r,t,conf,c:project(e,r,t,conf,c,mode)
    try:return dec.decode(e,cfg)
    finally:dec._track,dec._notated=oldtrack,oldnotated
