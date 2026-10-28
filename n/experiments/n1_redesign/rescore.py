"""Recompute structural scores from saved maps without repeating a decoder arm."""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import multiprocessing as mp
from base import *
from measurements import structural,null_map
from run import check_authority


def job(args):
    seed,xs=args
    with gzip.open(OUT/'maps'/seed/(xs[0]['rec']+'.json.gz'),'rt') as f:maps=json.load(f)['maps']
    structures=[];nulls=[]
    for arm,mm in maps.items():
        offset,_=global_offset(xs,mm)
        for x in xs:
            m=MusicMap.from_json(mm[x['key']])
            for aligned,o in ((True,offset),(False,0.)):
                structures.append(dict(seed=seed,panel=x['panel'],rec=x['rec'],key=x['key'],arm=arm,
                    aligned=aligned,offset=float(o),strata=x['strata'],**structural(x['ref'],m,o)))
    if seed=='s0':
        for arm in ('random_phase','random_level','x2'):
            nm={x['key']:null_map(x['ref'],x['key'],arm).to_json() for x in xs}
            offset,_=global_offset(xs,nm)
            for x in xs:
                m=MusicMap.from_json(nm[x['key']])
                for aligned,o in ((True,offset),(False,0.)):
                    nulls.append(dict(seed='reference',panel=x['panel'],rec=x['rec'],key=x['key'],arm=arm,
                        aligned=aligned,offset=float(o),**structural(x['ref'],m,o)))
    return dict(seed=seed,panel=xs[0]['panel'],rec=xs[0]['rec'],structures=structures,nulls=nulls)


def main():
    check_authority()
    path=OUT/'scorecards.jsonl'
    if path.exists():raise RuntimeError('Fresh scorecard output required')
    groups=defaultdict(list)
    for x in items():groups[(x['panel'],x['rec'])].append(x)
    jobs=[(s,xs) for _,xs in sorted(groups.items()) for s in ('s0','s1')]
    start=time.monotonic()
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(job,jobs,chunksize=1),1):
            append(path,result)
            if i%40==0:print('scorecard',i,len(jobs),round(time.monotonic()-start),flush=True)
    append(OUT/'job.log',dict(event='structural_rescore_complete',seconds=time.monotonic()-start,
        utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
