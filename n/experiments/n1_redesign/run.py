"""Run preregistered decode-free and fixed-track measurements on open caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import multiprocessing as mp

from base import *
from baseline import recording_rows
from decoder_measurements import CFG, dec
from ensomi_n.v0.evaluate import v0_evidence, oracle_evidence
from measurements import meter_runs, same_bars, candidates, structural, null_map, cluster_ratio, interval
from level import decode_level


def check_authority():
    rep=json.loads((OUT/'baseline-reproduction.json').read_text())
    if not rep['passed']:raise RuntimeError('Baseline did not reproduce')
    identity=json.loads((OUT/'prereg-identity.json').read_text())
    if hashlib.sha256((OUT/'prereg.md').read_bytes()).hexdigest()!=identity['sha256']:
        raise RuntimeError('Preregistration hash changed')


def x4_job(args):
    seed,xs=args
    maps={x['key']:saved(seed,x['item']) for x in xs}
    offset,_=global_offset(xs,maps)
    rows=[]
    for x in xs:
        ref=x['ref'];ev=v0_evidence(evidence(seed,x['item']))
        m=MusicMap.from_json(maps[x['key']]);pt=m.timing(ref.lo_ms-6000,ref.hi_ms+6000)
        for run in meter_runs(ref):
            a,b=run['start_index'],run['end_index'];meter=run['meter']
            times=ref.timing.beats[a:b+1]
            bar=times[::meter]
            if same_bars(bar[:-1],bar[1:],pt.bar_starts()-offset).all():continue
            chart=times[:-1]
            d=dec._down_at(ev,chart,CFG);d=np.clip(d,CFG.eps,1-CFG.eps)
            winners,best=candidates(np.log(d)-np.log1p(-d))
            right=(meter,0) in winners
            lo,hi=times[0],times[-1]
            pp=pt.beats[(pt.beats-offset>=lo-20)&(pt.beats-offset<hi-20)]
            decoded_right=False;dw=[]
            if len(pp):
                d=dec._down_at(ev,pp,CFG);d=np.clip(d,CFG.eps,1-CFG.eps)
                dw,_=candidates(np.log(d)-np.log1p(-d))
                # The end beat is only for checking bar closure, not for the sum.
                after=pt.beats[pt.beats>pp[-1]+1e-6][:8]
                extended=np.r_[pp,after]-offset
                decoded_right=any(same_bars(bar[:-1],bar[1:],extended[p::mm]).all() for mm,p in dw)
            g=geometry(chart,pp-offset)
            if min(g['r20'],g['p20'])<.9:cls='grid or level'
            elif right:cls='evidence right, decoder wrong'
            elif winners[0][0]!=meter:cls='meter'
            else:cls='systematic evidence error'
            rows.append(dict(seed=seed,panel=x['panel'],rec=x['rec'],key=x['key'],
                **run,lo=float(lo),hi=float(hi),chart_right=int(right),decoded_right=int(decoded_right),
                one=1,cls=cls,winners=winners,decoded_winners=dw,tied=int(len(winners)>1),
                phase_error=int(winners[0][1]),r20=g['r20'],p20=g['p20']))
    return rows


def panels_job(args):
    seed,xs=args
    arms=['baseline','oracle_original','oracle_d2','own_d2','d4']
    maps={a:{} for a in arms};diags=[]
    for x in xs:
        ev=v0_evidence(evidence(seed,x['item']))
        oracle=oracle_evidence(x['ref'],ev.n_frames)
        with np.load(OUT/'tracks'/seed/(x['item']+'.npz')) as z:
            track=tuple(z[k] for k in ('frames','kinds','periods'))
        maps['baseline'][x['key']]=saved(seed,x['item'])
        for arm in arms[1:]:
            m,diag=decode_level(ev,oracle,arm,track,CFG)
            maps[arm][x['key']]=m.to_json();diags.append(dict(key=x['key'],arm=arm,**diag))
    rows=[];structures=[]
    for arm in arms:
        offset,identified=global_offset(xs,maps[arm])
        if arm!='baseline':rows+=score_record((seed+':'+arm,xs,maps[arm]))
        for x in xs:
            m=MusicMap.from_json(maps[arm][x['key']])
            for aligned,o in ((True,offset),(False,0.)):
                structures.append(dict(seed=seed,panel=x['panel'],rec=x['rec'],key=x['key'],arm=arm,
                    aligned=aligned,offset=float(o),strata=x['strata'],**structural(x['ref'],m,o)))
    nulls=[]
    if seed=='s0':
        for arm in ('random_phase','random_level','x2'):
            nm={x['key']:null_map(x['ref'],x['key'],arm).to_json() for x in xs}
            offset,_=global_offset(xs,nm)
            for x in xs:
                m=MusicMap.from_json(nm[x['key']])
                for aligned,o in ((True,offset),(False,0.)):
                    nulls.append(dict(seed='reference',panel=x['panel'],rec=x['rec'],key=x['key'],
                        arm=arm,aligned=aligned,offset=float(o),**structural(x['ref'],m,o)))
    path=OUT/'maps'/seed/(xs[0]['rec']+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:json.dump(dict(maps=maps,diagnostics=diags),f)
    return dict(seed=seed,panel=xs[0]['panel'],rec=xs[0]['rec'],rows=rows,structures=structures,nulls=nulls)


def main():
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['x4','panels']);args=a.parse_args()
    check_authority()
    groups=defaultdict(list)
    for x in items():groups[(x['panel'],x['rec'])].append(x)
    jobs=[(s,xs) for _,xs in sorted(groups.items()) for s in ('s0','s1')]
    path=OUT/(args.phase+'.jsonl')
    if path.exists():raise RuntimeError('Fresh output required: '+str(path))
    start=time.monotonic()
    append(OUT/'job.log',dict(event='start',phase=args.phase,utc=datetime.now(timezone.utc).isoformat()))
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(x4_job if args.phase=='x4' else panels_job,jobs,chunksize=1),1):
            append(path,result)
            if i%20==0:print(args.phase,i,len(jobs),round(time.monotonic()-start),flush=True)
    append(OUT/'job.log',dict(event='complete',phase=args.phase,seconds=time.monotonic()-start,
        utc=datetime.now(timezone.utc).isoformat()))
    if args.phase=='x4':
        rows=[r for line in path.read_text().splitlines() for r in json.loads(line)]
        selected={r['rec'] for r in json.loads((ROOT/'artifacts/n1-stage0-20261028/events-cohort.json').read_text())['selected']}
        summary={}
        for panel in ('panel','val_hard','pooled','cohort40'):
            summary[panel]={}
            for seed in ('s0','s1'):
                rs=[r for r in rows if r['seed']==seed and (panel=='pooled' or r['panel']==panel or panel=='cohort40' and r['rec'] in selected)]
                classes={c:sum(r['cls']==c for r in rs) for c in sorted({r['cls'] for r in rs})}
                recs=defaultdict(list)
                for r in rs:recs[r['rec']].append(r['chart_right'])
                summary[panel][seed]=dict(chart=cluster_ratio(rs,'chart_right','one'),
                    decoded=cluster_ratio(rs,'decoded_right','one'),failure_classes=classes,
                    recording_mean=interval([np.mean(v) for v in recs.values()]),
                    runs=len(rs),ties=sum(r['tied'] for r in rs))
        gate=all(summary['pooled'][s]['chart']['value']>=.5 for s in ('s0','s1'))
        dump(OUT/'x4-summary.json',dict(cohorts=summary,d3_gate=gate))
        print('X4',json.dumps(summary['pooled']), 'D3 gate',gate,flush=True)


if __name__=='__main__':main()
