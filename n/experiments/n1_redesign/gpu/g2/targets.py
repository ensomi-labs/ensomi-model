"""Compact chart regimes with X5d targets on the stretched crop clock."""
from dataclasses import dataclass,replace
from functools import lru_cache
from types import SimpleNamespace
import json
import pickle
import numpy as np
from .common import OUT,G1,ROOT,BULK,dump
from .x5d_targets import targets,band_of,regime_steps
from .. import base
from ..audit_inputs import context,check_rows,digest,probe_bucket
from ..targets import chart_key


@dataclass
class CompactReference:
    lo_ms: float
    hi_ms: float
    segments: tuple
    unscorable: tuple

    def scorable(self,times):
        ok=np.ones(len(times),bool)
        for a,b in self.unscorable:ok &= ~((times>=a)&(times<b))
        return ok


def compact(ref):
    return dict(lo_ms=ref.lo_ms,hi_ms=ref.hi_ms,
                segments=[(s.start_ms,s.beat_ms) for s in ref.segments],unscorable=ref.unscorable)


def expand(d,rate=1.):
    return CompactReference(d['lo_ms']*rate,d['hi_ms']*rate,
        tuple(SimpleNamespace(start_ms=a*rate,beat_ms=b*rate) for a,b in d['segments']),
        tuple((a*rate,b*rate) for a,b in d['unscorable']))


@lru_cache(maxsize=1)
def database():
    with (OUT/'targets.pkl').open('rb') as f:return pickle.load(f)


def references(sha,key=None,rate=1.):
    variants=database()[sha]
    return [expand(variants[key],rate)] if key is not None else [expand(d,rate) for d in variants.values()]


def mask_unsupported(refs):
    """Mask only band supervision where a stretched red line leaves the vocabulary."""
    result=[]
    for ref in refs:
        spans=list(ref.unscorable)
        for i,seg in enumerate(ref.segments):
            band=int(band_of(seg.beat_ms))
            if band < -2 or band > 4:
                end=ref.segments[i+1].start_ms if i+1<len(ref.segments) else float('inf')
                spans.append((seg.start_ms,end))
        result.append(replace(ref,unscorable=tuple(spans)))
    return result


@lru_cache(maxsize=1)
def domain_decision():
    return json.loads((OUT/'target-domain-decision.json').read_text())['mode']


def dense_band(sha,start,n,rate=1.):
    refs=references(sha,rate=rate)
    mode=domain_decision()
    if mode=='mask':refs=mask_unsupported(refs)
    elif mode!='strict':raise RuntimeError('Unsupported target-domain decision')
    y,valid=targets(refs,start+10.*np.arange(n))
    return dict(band_y=y,band_w=valid.astype(np.float32))


def build():
    from ensomi_n.labels import read_set
    OUT.mkdir(parents=True,exist_ok=True)
    if json.loads((G1/'results.json').read_text())['status']!='complete':raise RuntimeError('G1 incomplete')
    c=json.loads((G1/'cohorts.json').read_text())
    paths,args,corpus,audit,sealed,exclusion,decision=context(ROOT)
    lists=dict(c,probe_fit=[r for r in c['g0_probe'] if probe_bucket(r['audio_sha256'])<2],
               probe_cal=[r for r in c['g0_probe'] if probe_bucket(r['audio_sha256'])==2])
    audit['cohorts']={}
    for name,rows in lists.items():
        kept,removed=decision.filter(rows)
        bad=check_rows(kept,exclusion,sealed)
        audit['cohorts'][name]=dict(rows=len(rows),audio=len({r['audio_sha256'] for r in rows}),
                                    removed=len(removed),violations=bad)
        if bad or removed:
            dump(OUT/'input-audit.json',audit);raise RuntimeError(f'G2 audit stopped at {name}')
    audit['source_cohorts_sha256']=digest(G1/'cohorts.json')
    audit['decision_sha256']=digest(G1/'decision-exclusions.json')
    dump(OUT/'input-audit.json',audit)
    data={};rates=np.zeros(7,np.int64);frames=0;outside=[]
    index=json.loads((BULK/'song-index-v0.json').read_text())
    for i,r in enumerate(c['training']+c['panels']):
        sha=r['audio_sha256'];key=chart_key(r['charts'])
        if key in data.get(sha,{}):continue
        ref=read_set([ROOT/p for p in r['charts']],key=sha)
        d=compact(ref);data.setdefault(sha,{})[key]=d
        if i<len(c['training']):
            times=10.*np.arange(index[sha]['frames'])
            frames+=len(times)
            for rate in (1.,.88,1.12):
                rr=expand(d,rate);tt=times*rate;steps=regime_steps(rr,tt)
                valid=base.inside(tt,rr)&np.isfinite(steps)&(steps>0)
                ix=np.flatnonzero(valid);bands=band_of(steps[ix]);bad=(bands < -2)|(bands>4)
                if bad.any():outside.append(dict(audio_sha256=sha,rate=rate,frames=int(bad.sum()),
                    mapped_frames=len(ix),bpm_min=float((60000/steps[ix[bad]]).min()),
                    bpm_max=float((60000/steps[ix[bad]]).max())))
                if rate==1.:rates+=np.bincount(bands[~bad]+2,minlength=7)
        if (i+1)%500==0:print(json.dumps(dict(event='targets',done=i+1,total=len(c['training'])+len(c['panels']))),flush=True)
    with (OUT/'targets.pkl').open('wb') as f:pickle.dump(data,f)
    dump(OUT/'target-rates.json',dict(audio=len(data),training=len(c['training']),frames=frames,
                                   mapped_frames_in_support=int(rates.sum()),band_counts=rates.tolist(),
                                   out_of_support=outside))


if __name__=='__main__':build()
