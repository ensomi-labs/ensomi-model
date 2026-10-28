"""Compact attack, notated-subdivision and ladder targets with exact crop stretching."""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from ensomi_n.labels import read_set
from ensomi_n.targets import RecLabels
from ensomi_n.v0.labels import V0Labels, dense_v0
from .cohorts import ROOT, OUT, BULK, dump
from .osu_notes import parse_osu, onsets_and_events

DIVS = (1, 2, 3, 4, 6, 8, 12, 16)
NEW_KEYS = ('attack_y', 'attack_w', 'snap_y', 'snap_w', 'sal_y', 'sal_w')


def chart_key(charts):
    return hashlib.sha256('\n'.join(sorted(charts)).encode()).hexdigest()[:16]


def nearest_indices(times, points):
    times, points = np.asarray(times), np.asarray(points)
    if not len(points):
        return np.zeros(len(times),int), np.full(len(times),np.inf)
    right=np.clip(np.searchsorted(points,times),0,len(points)-1)
    left=np.maximum(right-1,0)
    ix=np.where(abs(times-points[left])<=abs(times-points[right]),left,right)
    return ix,abs(times-points[ix])


def compact(ref, sal_times=(), retention=()):
    lab=RecLabels.from_reference(ref)
    heads=lab.heads
    lines=ref.timing.beats
    steps=ref.timing.step_ms
    end=np.minimum(np.r_[lines[1:],lines[-1]+steps[-1]],lines+steps)
    k=np.searchsorted(lines,heads+2.,side='right')-1
    kk=np.clip(k,0,len(lines)-1)
    valid=(k>=0)&(heads<end[kk]+2.)
    phase=(heads-lines[kk])/steps[kk]
    div=np.zeros(len(heads),int)
    for d in DIVS:
        take=(div==0)&valid&(abs(phase*d-np.round(phase*d))/d*steps[kk]<=2.+1e-9)
        div[take]=d
    cls=np.full(len(heads),-1,np.int8)
    cls[div==1]=0;cls[np.isin(div,(2,4,8,16))]=1;cls[np.isin(div,(3,6,12))]=2
    count=np.bincount(kk[valid],minlength=len(lines))
    cell=np.full(len(lines),-1,np.int8)
    tern=np.bincount(kk[valid&(cls==2)],minlength=len(lines))
    duple=np.bincount(kk[valid&(cls==1)],minlength=len(lines))
    unsnapped=np.bincount(kk[valid&(cls<0)],minlength=len(lines))
    cell[(count>=2)&(tern>0)]=1
    cell[(count>=2)&(tern==0)&(duple>0)&(unsnapped==0)]=0
    r=np.asarray(retention,float)
    bins=np.where(r==0,0,np.where(r<.5,1,np.where(r<1,2,3))).astype(np.int8)
    return dict(heads=heads,snap=cls,lines=lines,steps=steps,ends=end,count=count,cell=cell,
                sal_times=np.asarray(sal_times,float),retention=r,sal_bin=bins)


def dense_new(t, start_ms, n, rate=1.):
    """Heads stretch in milliseconds before rounding to the fixed 10 ms crop clock."""
    u=start_ms+10.*np.arange(n)
    heads=t['heads']*rate
    centres=np.round((heads-start_ms)/10.)
    hi, hd=nearest_indices(np.arange(n),centres)
    _, exact_distance=nearest_indices(u,heads)
    k=np.searchsorted(t['lines']*rate,u,side='right')-1
    kk=np.clip(k,0,len(t['lines'])-1)
    active=(k>=0)&(u<t['ends'][kk]*rate)&(t['count'][kk]>=2)
    aw=np.where((exact_distance>40.)&active,1.,0.)
    ay=np.zeros(n,np.float32)
    pos=hd<=2
    ay[pos]=1.;aw[pos]=np.where(hd[pos]<=1,1.,.5)
    snapped=t['snap']>=0
    si,sd=nearest_indices(np.arange(n),centres[snapped])
    sy=t['snap'][snapped][si] if snapped.any() else np.zeros(n,np.int8)
    st=np.round((t['sal_times']*rate-start_ms)/10.)
    bi,bd=nearest_indices(np.arange(n),st)
    bins=t['sal_bin'][bi] if len(st) else np.zeros(n,np.int8)
    return dict(attack_y=ay,attack_w=aw.astype(np.float32),snap_y=sy.astype(np.int64),
                snap_w=(sd<=1).astype(np.float32),sal_y=(bins[:,None]>np.arange(3)).astype(np.float32),
                sal_w=(bd<=1).astype(np.float32))


@lru_cache(maxsize=256)
def load_target(sha, key=None):
    with np.load(OUT/'targets'/sha[:2]/f'{sha}.npz',allow_pickle=False) as z:
        keys=z['keys'].tolist()
        i=keys.index(key) if key is not None else 0
        return {k[len(str(i))+1:]:z[k] for k in z.files if k.startswith(f'{i}_')}


def build():
    while not (OUT/'cohorts.json').exists():time.sleep(5)
    cohorts=json.loads((OUT/'cohorts.json').read_text())
    census=list(csv.DictReader((OUT.parent/'census/charts.csv').open()))
    ranks={(r['set_dir'],r['audio'].casefold(),r['version'],r['mode']):int(r['ladder_rank']) for r in census}
    by_sha=defaultdict(dict)
    train={r['audio_sha256'] for r in cohorts['training']}
    for r in cohorts['training']+cohorts['panels']:
        by_sha[r['audio_sha256']].setdefault(chart_key(r['charts']),r)
    index=json.loads((BULK/'song-index-v0.json').read_text())
    totals=Counter();classes=Counter();started=time.monotonic()
    for number,(sha,variants) in enumerate(sorted(by_sha.items()),1):
        arrays={};keys=[]
        for i,(key,row) in enumerate(variants.items()):
            ref=read_set([ROOT/c for c in row['charts']],key=sha)
            parsed=[parse_osu((ROOT/c).read_bytes()) for c in row['charts']]
            mode={'4k':'mania4k','std':'std','taiko':'taiko'}[row['source']]
            ranked=[]
            for c,p in zip(row['charts'],parsed):
                heads,ev=onsets_and_events(p)
                rank=ranks.get((str(Path(c).parent),p['audio'].casefold(),p['version'],mode))
                ranked.append(dict(rank=rank,heads=heads,events=ev))
            if all(c['rank'] is not None for c in ranked):ranked.sort(key=lambda c:c['rank'])
            else:ranked.sort(key=lambda c:len(c['heads']))
            st,ret=[],[]
            if len(ranked)>=3:
                st=ranked[-1]['heads']
                ret=sum((nearest_indices(st,c['events'])[1]<=2.+1e-9) for c in ranked[:-1])/(len(ranked)-1)
            t=compact(ref,st,ret)
            for k,v in t.items():arrays[f'{i}_{k}']=v
            keys.append(key)
            if sha in train:
                # Label rates use the same full-audio frame clock and break/detour weights as training.
                n=index[sha]['frames']
                new=dense_new(t,0,n)
                old=dense_v0(V0Labels.from_reference(ref),0,n)
                w=new['attack_w']*old['weight']
                totals['attack_pos']+=float((w*new['attack_y']).sum())
                totals['attack_neg']+=float((w*(1-new['attack_y'])).sum())
                totals['attack_frames']+=int((w>0).sum())
                totals['snap_frames']+=int(new['snap_w'].sum())
                totals['sal_frames']+=int(new['sal_w'].sum())
                totals['all_frames']+=n
                classes.update({str(c):int(((new['snap_y']==c)&(new['snap_w']>0)).sum()) for c in range(3)})
        path=OUT/'targets'/sha[:2]/f'{sha}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,keys=np.asarray(keys),**arrays)
        if number%100==0 or number==len(by_sha):
            status=dict(done=number,total=len(by_sha),wall_s=time.monotonic()-started)
            dump(OUT/'target-status.json',status);print(json.dumps(status),flush=True)
    result=dict(recordings=len(by_sha),training_recordings=len(train),rates=dict(totals),snap_classes=dict(classes),
                attack_pos_weight=float(np.clip(np.sqrt(totals['attack_neg']/totals['attack_pos']),1,20)),
                wall_s=time.monotonic()-started)
    dump(OUT/'target-rates.json',result)


if __name__=='__main__':build()
