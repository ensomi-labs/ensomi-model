"""Exact-clock workload, recovery and multi-scale contrasts from complete rows.

Scopes select observations, never reset physical history. Occupied time and
recovery remain distinct from attack counts and LN-head fraction. No statistic
here is a calibrated physiological strain or an automatic style/BAD label.
"""
from dataclasses import dataclass

import numpy as np

from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..scoped_style_modeling.dataset import ContractError


@dataclass(frozen=True)
class Scope:
    name: str
    start_ms: int
    end_ms: int

    def __post_init__(self):
        if (not self.name or type(self.start_ms) is not int or type(self.end_ms) is not int or
                not 0<=self.start_ms<self.end_ms):
            raise ContractError('Evaluation scopes require an identity and positive half-open native clock')


def _counts(times,left,right):
    return np.searchsorted(times,right,side='left')-np.searchsorted(times,left,side='left')


class ChartTrace:
    """Read a complete prefix from BOS through explicit observed coverage.

    Rows can end with active LNs; their observed occupation ends at coverage,
    without creating release events. Input rows after coverage are rejected.
    Coverage is an exclusive observation boundary, not necessarily audio EOF.
    """
    def __init__(self,rows,coverage_ms):
        if type(coverage_ms) is not int or coverage_ms<=0:
            raise ContractError('Trace coverage must be a positive native clock')
        rows=tuple(rows)
        self.coverage_ms=coverage_ms
        self.times=np.asarray([r.time_ms for r in rows],dtype=np.float64)
        self.actions=np.asarray([r.actions for r in rows],dtype=np.int8).reshape(-1,4)
        if len(rows) and rows[-1].time_ms>=coverage_ms:
            raise ContractError('Evaluation rows must precede exclusive observed coverage')
        state=ExactReplayState()
        holds=[[] for _ in range(4)]
        rests=[[] for _ in range(5)]
        free_start=[0.]*5
        for row in rows:
            occupied=state.occupancy
            for lane,action in enumerate(row.actions):
                if action:
                    if not occupied[lane]:rests[lane].append((free_start[lane],row.time_ms))
                    free_start[lane]=row.time_ms
                if action==3:
                    holds[lane].append((state.open_ln_start_ms[lane],row.time_ms))
            if not any(occupied):rests[4].append((free_start[4],row.time_ms))
            free_start[4]=row.time_ms
            state=commit(state,row)
        for lane,start in enumerate(state.open_ln_start_ms):
            if start is None:rests[lane].append((free_start[lane],coverage_ms))
            else:holds[lane].append((start,coverage_ms))
        if not any(state.occupancy):rests[4].append((free_start[4],coverage_ms))
        self.holds=tuple(np.asarray(h,dtype=float).reshape(-1,2) for h in holds)
        self.rests=tuple(np.asarray(r,dtype=float).reshape(-1,2) for r in rests)
        head=np.isin(self.actions,(1,2))
        self.heads=tuple(self.times[head[:,c]] for c in range(4))
        self.releases=tuple(self.times[self.actions[:,c]==3] for c in range(4))
        self.longs=tuple(self.times[self.actions[:,c]==2] for c in range(4))
        self.H=self.times[head.any(-1)]

    def _scope(self,scope):
        if scope.end_ms>self.coverage_ms:
            raise ContractError('Scope extends beyond observed chart coverage')

    def held_area(self,time_ms):
        """Integrate actual per-column occupied milliseconds on [0,time)."""
        time=np.asarray(time_ms,dtype=float)
        if np.any((time<0)|(time>self.coverage_ms)):
            raise ContractError('Held-time integration needs observed nonnegative clocks')
        result=[]
        for intervals in self.holds:
            if not len(intervals):
                result.append(np.zeros_like(time));continue
            starts,ends=intervals.T
            prefix=np.r_[0.,np.cumsum(ends-starts)]
            n=np.searchsorted(ends,time,side='right')
            partial=np.where(n<len(starts),np.maximum(0,time-starts[n.clip(max=len(starts)-1)]),0.)
            result.append(prefix[n]+partial)
        return np.stack(result,-1)

    def interval_features(self,left,right):
        """Return H, four attack, four held-duty, LN-head and release rates.

        Counts use [left,right). Holding is integrated exactly; rates use seconds.
        Caller-selected elapsed windows can be used for causal or retrospective
        evaluation, and are never injected into the generation model.
        """
        left,right=np.broadcast_arrays(np.asarray(left,float),np.asarray(right,float))
        if np.any((left<0)|(right>self.coverage_ms)|(right<=left)):
            raise ContractError('Activity windows must have positive observed duration')
        width=right-left
        rates=lambda events: np.stack([_counts(t,left,right)*1000/width for t in events],-1)
        h=_counts(self.H,left,right)*1000/width
        attack=rates(self.heads)
        held=(self.held_area(right)-self.held_area(left))/width[...,None]
        ln=rates(self.longs).sum(-1)
        release=rates(self.releases).sum(-1)
        return np.concatenate((h[...,None],attack,held,ln[...,None],release[...,None]),-1)

    def scope_report(self,scope,*,recovery_ms=(250,500,1000),attack_windows_ms=(500,1000,2000,4000,8000,16000)):
        """Return exact scope facts plus prefix-aware recovery and column peaks.

        Recovery credit starts only after a truly free/no-action interval has
        lasted the stated duration. A range boundary never restarts that clock.
        Peak trailing attack windows also retain pre-scope attacks.
        """
        self._scope(scope)
        a,b=scope.start_ms,scope.end_ms;duration=b-a
        active=(self.times>=a)&(self.times<b)
        selected=self.actions[active]
        heads=int(np.isin(selected,(1,2)).sum());longs=int((selected==2).sum())
        result=dict(scope=dict(name=scope.name,start_ms=a,end_ms=b),
            H=int(_counts(self.H,a,b)),heads=heads,LN_heads=longs,releases=int((selected==3).sum()),
            LN_head_fraction=longs/heads if heads else None,
            held_fraction_per_column=((self.held_area(b)-self.held_area(a))/duration).tolist(),
            recovery={},column_attack_peaks={},short_attacks=[])
        free=self.rests[4]
        free_time=np.maximum(0,np.minimum(free[:,1],b)-np.maximum(free[:,0],a)).sum()
        result['any_held_fraction']=float(1-free_time/duration)
        result['mean_held_columns']=sum(result['held_fraction_per_column'])
        for threshold in recovery_ms:
            fractions=[]
            for intervals in self.rests:
                credit=np.maximum(0,np.minimum(intervals[:,1],b)-np.maximum(intervals[:,0]+threshold,a)).sum()
                fractions.append(float(credit/duration))
            result['recovery'][str(threshold)]=dict(per_column=fractions[:4],all_columns=fractions[4])
        for width in attack_windows_ms:
            maxima=[]
            for c,times in enumerate(self.heads):
                ends=np.r_[a,times[(times>=a)&(times<b)]]
                # The point observation includes an attack at its timestamp.
                n=np.searchsorted(times,ends,side='right')-np.searchsorted(times,ends-width,side='right')
                k=int(np.argmax(n));t=float(ends[k])
                maxima.append(dict(column=c,Hz=float(n[k]*1000/width),at_ms=t,
                    history_start_ms=t-width,attacks=int(n[k])))
            result['column_attack_peaks'][str(width)]=max(maxima,key=lambda v:v['Hz'])
        for c,times in enumerate(self.heads):
            for i in np.flatnonzero((np.diff(times)<20)&(times[1:]>=a)&(times[1:]<b)):
                result['short_attacks'].append(dict(column=c,previous_ms=float(times[i]),
                    attack_ms=float(times[i+1]),gap_ms=float(times[i+1]-times[i])))
        return result

    def contrasts(self,scope,*,half_windows_ms=(500,1000,2000,4000,8000,16000),step_ms=250):
        """Locate pressure/texture changes using adjacent elapsed-time windows.

        Both halves must lie within this same named control range. Values are
        signed right-minus-left contrasts; large values are not a quality reward.
        Column-share change is reported separately from total activity change.
        """
        self._scope(scope)
        reports={}
        for width in half_windows_ms:
            centers=np.arange(scope.start_ms+width,scope.end_ms-width+1,step_ms)
            if not len(centers):
                reports[str(width)]=dict(status='insufficient_scope',centers_ms=[],differences=[])
                continue
            left=self.interval_features(centers-width,centers)
            right=self.interval_features(centers,centers+width)
            def shares(x):
                total=x[:,1:5].sum(-1)
                return np.divide(x[:,1:5],total[:,None],out=np.zeros_like(x[:,1:5]),where=total[:,None]>0)
            differences=right-left
            transfer=np.abs(shares(right)-shares(left)).sum(-1)/2
            reports[str(width)]=dict(status='measured',centers_ms=centers.tolist(),
                differences=differences.tolist(),column_share_transfer=transfer.tolist(),
                median_absolute_contrast=np.median(np.abs(differences),axis=0).tolist(),
                p90_absolute_contrast=np.quantile(np.abs(differences),.9,axis=0).tolist())
        return reports
