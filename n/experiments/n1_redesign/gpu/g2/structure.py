"""Recording intervals, complete meter runs and one-to-one structural scoring."""
from collections import Counter, defaultdict
import hashlib
import math
import numpy as np

from ..base import inside, one_to_one, geometry, nearest, MusicMap
from ensomi_n.eval.measures import _bar_terms


def interval(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, value=None, ci95=None)
    rng = np.random.default_rng(20261028)
    b = a[rng.integers(0, len(a), (2000, len(a)))].mean(1)
    return dict(n=len(a), value=float(a.mean()), ci95=np.quantile(b, [.025,.975]).tolist())


def cluster_ratio(rows, numerator, denominator):
    groups = defaultdict(lambda: np.zeros(2))
    for r in rows:
        groups[r['rec']] += [r[numerator], r[denominator]]
    a = np.array([groups[k] for k in sorted(groups)])
    if not len(a):
        return dict(n=0, value=None, ci95=None)
    rng = np.random.default_rng(20261028)
    b = a[rng.integers(0, len(a), (2000, len(a)))].sum(1)
    q = np.divide(b[:,0], b[:,1], out=np.full(2000,np.nan), where=b[:,1]>0)
    return dict(n=len(a), numerator=float(a[:,0].sum()), denominator=float(a[:,1].sum()),
                value=float(a[:,0].sum()/a[:,1].sum()), ci95=np.nanquantile(q,[.025,.975]).tolist())


def margin(s0, s1):
    a, b = interval(s0), interval(s1)
    return max(abs(a['value']-b['value']), *(0.5*(x['ci95'][1]-x['ci95'][0]) for x in (a,b)))


def meter_runs(ref):
    """Maximal chains of complete bars with equal numbers of beat intervals."""
    t = ref.timing
    ds = np.flatnonzero(t.downbeat)
    runs, run = [], None
    for a, b in zip(ds[:-1], ds[1:]):
        m = int(b-a)
        valid = 2 <= m <= 8 and bool(inside(t.beats[a:b+1], ref).all())
        if not valid:
            if run is not None:
                runs.append(run)
            run = None
            continue
        if run is not None and run['meter']==m and run['end_index']==a:
            run['end_index']=int(b)
            run['bars']+=1
        else:
            if run is not None:
                runs.append(run)
            run=dict(start_index=int(a),end_index=int(b),meter=m,bars=1)
    if run is not None:
        runs.append(run)
    return [r for r in runs if r['bars']>=4]


def same_bars(starts, ends, predicted):
    starts, ends, predicted = map(np.asarray, (starts, ends, predicted))
    keep = np.ones(len(starts),bool)
    return _bar_terms(starts, ends, predicted, predicted, keep, keep)[0]


def candidates(logodds):
    scores = [(float(np.sum(logodds[p::m])),m,p) for m in range(2,9) for p in range(m)]
    best = max(s[0] for s in scores)
    return [(m,p) for s,m,p in scores if s>=best-1e-10], best


def spans8(ref):
    bars = ref.timing.bar_starts()
    bars = bars[(bars>=ref.lo_ms)&(bars<=ref.hi_ms)]
    cuts = [ref.lo_ms]
    if len(bars):
        cuts += [float(v) for v in bars[::8] if ref.lo_ms < v < ref.hi_ms]
    cuts += [ref.hi_ms+1e-7]
    return list(zip(cuts[:-1],cuts[1:]))


def boundaries(timing):
    result=[]
    for i in np.flatnonzero(timing.seg_start):
        if i==0:
            continue
        before,after=timing.step_ms[i-1:i+1]
        gap=timing.beats[i]-timing.beats[i-1]
        phase=abs(gap-round(gap/before)*before)
        if abs(after/before-1)>.005 or phase>15:
            result.append(float(timing.beats[i]))
    return np.asarray(result)


CLASSES = ('correct','offset-only','fine','inserted lines','missing lines','phase',
           'tempo regime','level x2','level half','level 3:2','bar phase','meter','boundary')
PRIORITY = {k:i for i,k in enumerate(('level x2','level half','level 3:2','tempo regime',
                                    'phase','inserted lines','missing lines','meter','bar phase','boundary'))}


def structural(ref, mmap, offset=0.):
    """First error per eight-bar span; predicted insertions count against readiness."""
    rt=ref.timing
    pt=mmap.timing(ref.lo_ms-6000,ref.hi_ms+6000)
    rb=rt.beats
    pb=pt.beats-offset
    rbar=rt.bar_starts()
    pbar=pt.bar_starts()-offset
    rbnd,pbnd=boundaries(rt),boundaries(pt)-offset
    # Match on the song first. A matched line belongs to its reference span,
    # even when tolerance places the prediction across an eight-bar cut.
    scored=np.flatnonzero(inside(rb,ref))
    ri_all,pj_all=one_to_one(rb[scored],pb,15)
    matched=np.full(len(rb),-1,dtype=int)
    matched[scored[ri_all]]=pj_all
    used=np.zeros(len(pb),bool);used[pj_all]=True
    rows=[]
    for lo,hi in spans8(ref):
        ridx=np.flatnonzero((rb>=lo)&(rb<hi)&ref.scorable(rb))
        pidx=np.flatnonzero((pb>=lo)&(pb<hi)&ref.scorable(pb))
        r=rb[ridx];p=pb[pidx]
        if not len(r):
            continue
        events=[]
        def event(time,cls,**kw):
            events.append(dict(time=float(time),cls=cls,**kw))
        # Use local four-interval patterns so a level change within a span is visible.
        pattern=p
        if 0<len(p)<5:
            j=int(np.searchsorted(pb,p[0]))
            pattern=pb[max(0,j-4):min(len(pb),j+5)]
        if len(pattern)>=5:
            gaps=np.diff(pattern)
            for j in range(len(pattern)-4):
                g=gaps[j:j+4]
                step=float(np.median(g))
                target=float(rt.step_at([pattern[j]])[0])
                if not np.isfinite(target) or target<=0 or np.max(abs(g/step-1))>.02:
                    continue
                ratio=step/target
                cls=None
                for q,c in ((.5,'level x2'),(2.,'level half'),(2/3,'level 3:2'),(1.5,'level 3:2')):
                    if abs(ratio/q-1)<=.02:
                        cls=c;break
                if cls is None and abs(ratio-1)>.02:
                    cls='tempo regime'
                if cls:
                    when=max(lo,min(float(r[0]),float(pattern[j]))) if j==0 else max(lo,pattern[j])
                    if when<hi:event(when,cls,ratio=ratio)
                    break
        if len(p) and len(r)>=4:
            e,_=nearest(r,p)
            med=float(np.median(e))
            mad=float(np.median(abs(e-med)))
            rp=float(np.median(np.diff(r)))
            pp=float(np.median(np.diff(p))) if len(p)>1 else math.inf
            if abs(pp/rp-1)<=.02 and abs(med)>15 and mad<=15:
                event(r[0],'phase',beat_fraction=med/rp)
        missing=np.flatnonzero(matched[ridx]<0)
        inserted=np.flatnonzero(~used[pidx])
        if len(inserted):
            event(p[inserted[0]],'inserted lines')
        if len(missing):
            event(r[missing[0]],'missing lines')
        # Bar geometry remains a distinct check after notated-line matching.
        rs=rbar[(rbar>=lo)&(rbar<hi)] if not len(missing) and not len(inserted) else []
        for a in rs:
            e,j=nearest(np.array([a]),pbar)
            if len(pbar)>1:
                k=int(np.clip(np.searchsorted(pbar,a,side='right')-1,0,len(pbar)-2))
                pred_meter=int(np.count_nonzero((pb>=pbar[k]-1e-6)&(pb<pbar[k+1]-1e-6)))
                q=int(np.searchsorted(rbar,a))
                if q+1<len(rbar):
                    ref_meter=int(np.count_nonzero((rb>=a-1e-6)&(rb<rbar[q+1]-1e-6)))
                    if pred_meter!=ref_meter:
                        event(a,'meter',reference=ref_meter,predicted=pred_meter)
                        break
            if abs(e[0])>15:
                event(a,'bar phase');break
        for a,other in ((rbnd,pbnd),(pbnd,rbnd)):
            aa=a[(a>=lo)&(a<hi)]
            if len(aa):
                e,_=nearest(aa,other)
                bad=aa[abs(e)>15]
                if len(bad):event(bad[0],'boundary')
        if events:
            first=min(events,key=lambda e:(round(e['time'],6),PRIORITY[e['cls']]))
            row=dict(lo=float(lo),hi=float(hi),**first)
        else:
            residual=pb[matched[ridx]]-r
            raw=residual+offset
            cls='correct' if np.all(abs(raw)<=5) else ('offset-only' if np.all(abs(residual)<=5) else 'fine')
            row=dict(lo=float(lo),hi=float(hi),cls=cls)
        rows.append(row)
    return dict(ready=bool(rows) and all(r['cls'] in CLASSES[:3] for r in rows),
                counts=dict(Counter(r['cls'] for r in rows)),spans=rows)


def null_map(ref,key,arm):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256((key+arm).encode()).digest()[:8],'little'))
    t=ref.timing
    beats=[];downs=[]
    for lo,hi in spans8(ref):
        ix=np.flatnonzero((t.beats>=lo)&(t.beats<hi))
        if not len(ix):continue
        phase=rng.random() if arm=='random_phase' else 0.
        ratio=float(rng.choice([.5,1.,2.])) if arm=='random_level' else (2. if arm=='x2' else 1.)
        for local,i in enumerate(ix):
            if ratio==.5 and local%2:continue
            n=2 if ratio==2 else 1
            for k in range(n):
                v=t.beats[i]+t.step_ms[i]*(phase+k/n)
                if v>=hi:continue
                beats.append(float(v));downs.append(bool(t.downbeat[i] and k==0))
    order=np.argsort(beats,kind='stable')
    b=np.array(beats)[order];d=np.array(downs)[order]
    keep=np.r_[True,np.diff(b)>2] if len(b) else np.zeros(0,bool)
    return MusicMap.from_beats(b[keep],np.flatnonzero(d[keep]))


def lattice_chance(bpm,divisors,tolerance=20.):
    period=60000./bpm
    cells=sorted({round(k/d,12) for d in divisors for k in range(d)})
    gaps=np.diff(cells+[1.+cells[0]])*period
    return float(np.minimum(gaps,2*tolerance).sum()/period)
