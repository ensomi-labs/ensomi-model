"""Open mapper-pair and listening-mark characterisation; writes only artifacts."""
from collections import defaultdict
from functools import lru_cache
from base import *
from baseline import recording_rows
from measurements import structural, cluster_ratio, interval, lattice_chance
from ensomi_n.labels import read_osu, reference_from_charts
from ensomi_n.timing import Reference
from ensomi_n.eval.census import read_header
from run import check_authority


@lru_cache(None)
def pair_reference(key):
    # Existing pairs are the allowlist; no corpus crawl or holdout inspection.
    parts=key.split('/')
    if len(parts) not in (2,3) or parts[0]!='0' or not parts[1].isdigit():
        raise ValueError('Unsupported mapper-pair set identity: '+key)
    paths=[]
    for p in sorted((ROOT/'dataset'/parts[0]/parts[1]).glob('*.osu')):
        header=read_header(p)
        if header.get('Mode')!='3' or float(header.get('CircleSize','0'))!=4:continue
        if len(parts)==3 and Path(header.get('AudioFilename','')).stem!=parts[2]:continue
        paths.append(p)
    return reference_from_charts([read_osu(p) for p in paths],key)


def mapper_pairs():
    source=ROOT/'artifacts/bulk/mapper-spread-20261021/run/pairs.jsonl'
    pairs=[json.loads(s) for s in source.read_text().splitlines()]
    pairs=[p for p in pairs if p['flags']['c1']]
    xs=[];maps={}
    for p in pairs:
        if set(p['split'].split('|'))-{'fit_train','fit_dev','cal'}:
            raise ValueError('Pair outside allowed open splits')
        ar,br=pair_reference(p['a']),pair_reference(p['b'])
        lo,hi=p['lo'],p['hi']
        ref=Reference.from_segments(ar.segments,ar.heads[(ar.heads>=lo)&(ar.heads<=hi)],lo_ms=lo,hi_ms=hi)
        key=p['a']+'|'+p['b']
        xs.append(dict(key=key,rec=p['rec'],panel='C1',ref=ref,strata={}))
        maps[key]=MusicMap(segments=br.segments).to_json()
    groups=defaultdict(list)
    for x in xs:groups[x['rec']].append(x)
    rows=[];structures=[]
    for rec,xx in sorted(groups.items()):
        delta,_=global_offset(xx,maps)
        rows+=score_record(('mapper',xx,maps))
        for x in xx:
            for aligned,o in ((True,delta),(False,0.)):
                structures.append(dict(panel='C1',rec=rec,key=x['key'],arm='mapper',seed='human',
                    aligned=aligned,offset=float(o),**structural(x['ref'],MusicMap.from_json(maps[x['key']]),o)))
    dump(OUT/'mapper-pairs.json',dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pairs=len(xs),recordings=len(groups),rows=rows,recording_rows=recording_rows(rows),structures=structures))
    print('Mapper pairs',len(xs),len(groups),flush=True)


def pulse_labels(proposals,times):
    keep=np.zeros(len(times),bool)
    for q in proposals:
        grid=np.asarray(q['beats_ms']);e,idx=nearest(times,grid);gaps=np.diff(grid)
        left=np.where(idx>0,gaps[np.clip(idx-1,0,len(gaps)-1)],np.nan)
        right=np.where(idx<len(grid)-1,gaps[np.clip(idx,0,len(gaps)-1)],np.nan)
        gate=np.minimum(50.,.2*np.nanmean(np.vstack([left,right]),axis=0))
        keep|=abs(e)<=gate
    return keep


def session_coverage():
    root=ROOT/'artifacts/timing-lens-20261024'
    sources=[root/'sessions/session-01.timing.marks.jsonl',root/'sessions/session-01.timing.json',
             root/'private/session-01.identities.json']
    latest={}
    for s in sources[0].read_text().splitlines():
        r=json.loads(s)
        if r['type']=='save' and (r['passage_id'] not in latest or r['time']>=latest[r['passage_id']]['time']):
            latest[r['passage_id']]=r
    passages=json.loads(sources[1].read_text())['passages']
    identities=json.loads(sources[2].read_text())['passages']
    rows=[];counts=[]
    for p in passages:
        pid=p['id'];r=latest.get(pid)
        if r is None:continue
        marks=sorted((m for m in r['marks'] if m['kind']=='beat'),key=lambda m:m['time_ms'])
        times=np.array([m['time_ms'] for m in marks])
        pulse=pulse_labels(p['proposals'],times)
        chart=next(v['source'] for v in identities[pid].values() if v['kind']=='chart')
        heads=read_osu(ROOT/chart).heads
        residual,_=nearest(times-7.7,heads)
        covered=abs(residual)<=20+1e-9
        audio=p['audio']['sha256']
        counts.append(dict(rec=audio,passage=pid,covered=int(covered[~pulse].sum()),off_pulse=int((~pulse).sum()),
            beat_marks=len(marks),pulse=int(pulse.sum())))
        for i,m in enumerate(marks):
            rows.append(dict(rec=audio,passage=pid,mark_id=m['id'],pulse=bool(pulse[i]),
                covered=bool(covered[i]),residual_ms=float(residual[i])))
    result=dict(counts=counts,marks=rows,coverage=cluster_ratio(counts,'covered','off_pulse'),
        passages=len(counts),recordings=len({r['rec'] for r in counts}),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    dump(OUT/'session-coverage.json',result)
    print('Session coverage',result['coverage'],flush=True)


def main():
    check_authority()
    mapper_pairs();session_coverage()
    dump(OUT/'lattice-chance.json',{str(bpm):{name:lattice_chance(bpm,divs)
        for name,divs in [('half',[2]),('half_third_quarter',[2,3,4]),('through_eighth_sixth',[2,3,4,6,8])]}
        for bpm in (80,120,160)})


if __name__=='__main__':main()
