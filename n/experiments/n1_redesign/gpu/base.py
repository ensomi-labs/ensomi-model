"""E1 geometry adapter, copied from the named consultation's exp/common.py.

The scoring functions are preserved; all input access is restricted to its
cached open-panel references. This file uses this worktree's N implementation.
"""
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / 'src'
sys.path.insert(0, str(SOURCE))
ROOT = Path('/Users/l/ensomi/ensomi-model')
PREV = ROOT / 'artifacts/astra-consult-20261021/astra/exp'
OUT = ROOT / 'artifacts/n1-stage0-20261028'
BULK = ROOT / 'artifacts/bulk/n1-stage0-20261028'
import hashlib
import json
import math
import pickle
import time
from types import SimpleNamespace
import numpy as np
from ensomi_n.mapformat import MusicMap
from ensomi_n.eval import score
from ensomi_n.eval.measures import nearest, weighted_median

def items():
    with (PREV / 'references.pkl').open('rb') as f:
        xs = pickle.load(f)
    assert len(xs) == 472 and set(x['panel'] for x in xs) == {'panel', 'val_hard'}
    return xs

def saved(seed, item):
    return json.loads((PREV / 'e2-maps' / seed / '009001' / (item + '.json')).read_text())['full']

def evidence(seed, item):
    with np.load(PREV / 'evidence' / seed / '009001' / (item + '.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')

def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')

def shifted(m,delta):
    d=m.to_json() if isinstance(m,MusicMap) else dict(m)
    d=dict(d)
    if 'beats' in d:d['beats']=[float(t)-delta for t in d['beats']]
    if 'segments' in d:d['segments']=[dict(s,start_ms=s['start_ms']-delta) for s in d['segments']]
    if 'heads' in d:d['heads']=[t-delta for t in d['heads']]
    return MusicMap.from_json(d)

def inside(t,ref,spans=None):
    t=np.asarray(t)
    keep=(t>=ref.lo_ms-1e-9)&(t<=ref.hi_ms+1e-9)&ref.scorable(t)
    if spans is not None:
        valid=np.zeros(len(t),dtype=bool)
        for a,b in spans:valid|=(t>=a-1e-9)&(t<b-1e-9)
        keep &= valid
    return keep

def notated(ref,m):
    r=ref.timing.beats
    r=r[inside(r,ref)]
    p=m.timing(ref.lo_ms-6000,ref.hi_ms+6000).beats
    return r,p

def global_offset(xs, maps):
    vals=[];weights=[]
    for x in xs:
        ref=x['ref'];m=MusicMap.from_json(maps[x['key']]);r,p=notated(ref,m)
        p=p[(p>=ref.lo_ms-50)&(p<=ref.hi_ms+50)]
        if not len(r) or not len(p):continue
        e,j=nearest(r,p);_,i=nearest(p,r)
        w=ref.weights(SimpleNamespace(t=r))
        ok=(np.abs(e)<=50+1e-9)&(i[j]==np.arange(len(r)))&(w>0)
        if ok.any():
            vals.extend(e[ok]);weights.extend(w[ok]/w[ok].sum())
    if not vals:return 0.0,False
    return float(np.clip(weighted_median(vals,weights),-50,50)),True

def one_to_one(r,p,tau):
    i=j=0;ri=[];pj=[]
    while i<len(r) and j<len(p):
        d=p[j]-r[i]
        if abs(d)<=tau+1e-9:ri.append(i);pj.append(j);i+=1;j+=1
        elif d < -tau:j+=1
        else:i+=1
    return np.asarray(ri,dtype=int),np.asarray(pj,dtype=int)

def geometry(r,p,w=None):
    out={'nr':len(r),'np':len(p)}
    for tau in (5,20):
        ri,pj=one_to_one(r,p,tau)
        recall=len(ri)/len(r) if len(r) else math.nan
        precision=len(pj)/len(p) if len(p) else 0.0
        out.update({f'm{tau}':len(ri),f'r{tau}':recall,f'p{tau}':precision,
                    f'f{tau}':2*recall*precision/(recall+precision) if recall+precision>0 else 0.0})
    e,_=nearest(r,p)
    for tau in (5,20):
        ok=np.isfinite(e)&(np.abs(e)<=tau+1e-9)
        out[f'nn{tau}']=float(ok.mean()) if len(r) else math.nan
        if w is not None:out[f'hw{tau}']=float(np.sum(w*ok)/w.sum()) if w.sum()>0 else math.nan
    return out

def score_record(job):
    arm,xs,maps=job
    delta,identified=global_offset(xs,maps)
    rows=[]
    for x in xs:
        ref=x['ref'];m=MusicMap.from_json(maps[x['key']]);aligned=shifted(m,delta)
        r,p=notated(ref,m);_,pa=notated(ref,aligned)
        score_cache={}
        for cell,spans in [('all',None)]+list(x['strata'].items()):
            rr=r[inside(r,ref,spans)]
            if not len(rr):continue
            pp=p[inside(p,ref,spans)];paa=pa[inside(pa,ref,spans)]
            # Heads restricted to a stratum for the interpolation-weighted companion.
            h=ref.heads[inside(ref.heads,ref,spans)] if ref.heads is not None else None
            from ensomi_n.timing import Reference
            sub=Reference(ref.timing,ref.lo_ms,ref.hi_ms,h,ref.segments,ref.unscorable,ref.key)
            w=sub.weights(SimpleNamespace(t=rr))
            row={'arm':arm,'key':x['key'],'panel':x['panel'],'rec':x['rec'],'cell':cell,'offset_ms':delta,'offset_identified':int(identified)}
            row.update({'raw_'+k:v for k,v in geometry(rr,pp,w).items()})
            row.update(geometry(rr,paa,w))
            tag=repr(spans)
            if tag not in score_cache:
                score_cache[tag]=(score(ref,aligned,spans=spans).summary(),score(ref,m,spans=spans).summary())
            s,raw_s=score_cache[tag]
            for k in ('bl20','bl5','level','bars_start','bars_same','boundary_recall','boundary_precision','reset_recall'):
                row['n_'+k]=s.get(k,math.nan)
                row['raw_n_'+k]=raw_s.get(k,math.nan)
            row['ready']=int(row['r5']>=.9);row['guarded_ready']=int(row['r5']>=.9 and row['p5']>=.9)
            rows.append(row)
    return rows
