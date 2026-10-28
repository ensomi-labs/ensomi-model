"""Stage-0 pulse scoring and the decoder line's structural scorecard on audited panels."""
from __future__ import annotations
from collections import defaultdict
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np

from . import base
from .cohorts import ROOT, OUT, dump
from .structure import structural

METRICS=('guarded_ready','raw_guarded_ready','n_bars_same','raw_n_bars_same','n_level','raw_n_level',
         'n_bl20','raw_n_bl20','r5','p5','raw_r5','raw_p5','offset_ms')


def panel_items():
    c=json.loads((OUT/'cohorts.json').read_text())
    allowed={r['audio_sha256'] for r in c['panels']}
    xs=base.items()
    if {x['item'] for x in xs} != allowed:
        raise ValueError('Reference cache differs from the audited panel list')
    return xs


def finite_mean(vals):
    v=np.asarray(vals,float);v=v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float('nan')


def recording_rows(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['panel'],r['cell'],r['rec'])].append(r)
    result=[]
    for (panel,cell,rec),rs in sorted(groups.items()):
        row=dict(panel=panel,cell=cell,rec=rec)
        for m in METRICS:row[m]=finite_mean([r.get(m,np.nan) for r in rs])
        row['guarded_ready']=float(row['r5']>=.9 and row['p5']>=.9)
        row['raw_guarded_ready']=float(row['raw_r5']>=.9 and row['raw_p5']>=.9)
        result.append(row)
    return result


def interval(vals):
    v=np.asarray(vals,float);v=v[np.isfinite(v)]
    if not len(v):return dict(n=0,value=None,ci95=None)
    rng=np.random.default_rng(20261028)
    boot=v[rng.integers(0,len(v),(1000,len(v)))].mean(1)
    return dict(n=len(v),value=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['panel'],r['cell'])].append(r)
        groups[('pooled',r['cell'])].append(r)
    return {f'{p}/{c}':{m:interval([r[m] for r in rs]) for m in METRICS} for (p,c),rs in groups.items()}


def score_group(job):
    name,xs,maps=job
    rows=base.score_record((name,xs,maps))
    offset,_=base.global_offset(xs,maps)
    structures=[dict(key=x['key'],panel=x['panel'],rec=x['rec'],
                     **structural(x['ref'],base.MusicMap.from_json(maps[x['key']]),offset)) for x in xs]
    return rows,structures


def score_maps(name, maps, workers=2):
    xs=panel_items();groups=defaultdict(list)
    for x in xs:groups[(x['panel'],x['rec'])].append(x)
    jobs=[(name,g,{x['key']:maps[x['key']] for x in g}) for g in groups.values()]
    rows=[];structures=[]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        for i,(rs,ss) in enumerate(pool.map(score_group,jobs),1):
            rows+=rs;structures+=ss
            if i%100==0:print(json.dumps(dict(event='score',arm=name,done=i,total=len(jobs))),flush=True)
    recs=recording_rows(rows)
    result=dict(name=name,recordings=recs,summary=summarize(recs),structure=structures)
    dump(OUT/'scores'/f'{name}.json',result)
    return result


def baseline():
    while not (OUT/'cohorts.json').exists():time.sleep(5)
    xs=panel_items()
    sig=[]
    for sha in sorted({x['item'] for x in xs}):
        with np.load(base.PREV/'evidence/s0/009001'/f'{sha}.npz',allow_pickle=False) as z:sig.append(z['logp_sigma'])
    sig=np.concatenate(sig)
    low=float(np.mean(sig<.01));p5=float(np.quantile(sig,.05))
    dump(OUT/'sigma-baseline.json',dict(frames=len(sig),share_below_001=low,p5=p5,floor=p5 if low>.05 else .01))
    old=json.loads((base.PREV/'e1-summary.json').read_text())
    discrepancies=[];compared=0
    for seed,checkpoints in [(0,['009001','008400','008137']),(1,['009001','008401','007973'])]:
        for ck in checkpoints:
            name=f'v01-s{seed}-{ck}'
            maps={x['key']:json.loads((base.PREV/f'e2-maps/s{seed}'/ck/f"{x['item']}.json").read_text())['full'] for x in xs}
            result=score_maps(name,maps)
            for r in result['recordings']:
                expected=old.get(r['panel'],{}).get(r['cell'],{}).get(f's{seed}-{ck}',{}).get('recordings',{}).get(r['rec'],{})
                for metric in ('r5','p5','n_bars_same','n_level','offset_ms'):
                    if metric in expected and np.isfinite(r[metric]) and np.isfinite(expected[metric]):
                        compared+=1
                        if abs(r[metric]-expected[metric])>1e-8:
                            discrepancies.append(dict(arm=name,rec=r['rec'],cell=r['cell'],metric=metric,actual=r[metric],expected=expected[metric]))
            dump(OUT/'baseline-reproduction.json',dict(compared=compared,differences=discrepancies))
    dump(OUT/'baseline-DONE.json',dict(status='complete',compared=compared,differences=len(discrepancies)))


if __name__=='__main__':baseline()
