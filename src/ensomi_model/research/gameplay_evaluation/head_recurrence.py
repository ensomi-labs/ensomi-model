"""Consecutive-head membership facts, not a jack/style/demand classifier."""
import numpy as np


def _observations(times, actions, start_ms, end_ms):
    """Count prefix run ages over H rows; pure releases do not reset membership.

    An age is the number of consecutive H groups containing this head's column.
    Read history before start_ms, but no event at/after end_ms. A gap has no
    automatic reset: witness durations and gap statistics carry the real clock.
    """
    times=np.asarray(times)
    heads=np.isin(np.asarray(actions),(1,2))
    keep=heads.any(-1)&(times<end_ms)
    clocks=times[keep];heads=heads[keep]
    n=len(clocks);index=np.arange(n)[:,None]
    if n:
        absent=np.maximum.accumulate(np.where(heads,-1,index),axis=0)
        age=(index-absent)*heads
    else:age=np.zeros((0,4),dtype=int)
    inside=clocks>=start_ms
    selected=heads&inside[:,None]
    opportunity=selected&(index>0)
    repeats=selected&(age>=2)
    values=age[selected]
    unique,count=np.unique(values,return_counts=True)
    total=int(selected.sum());eligible=int(opportunity.sum());repeated=int(repeats.sum())
    moments=None if not len(values) else dict(zip(('q50','q90','q99'),map(float,np.quantile(values,(.5,.9,.99)))))
    candidates=[]
    if n:
        run_ends=heads&~np.vstack((heads[1:],np.zeros((1,4),dtype=bool)))
        rows,columns=np.where(run_ends&inside[:,None])
        starts=rows-age[rows,columns]+1
        spans=clocks[rows]-clocks[starts]
        order=np.lexsort((columns,clocks[rows],spans,-age[rows,columns]))[:8]
        for chosen in order:
            i,k,s=int(rows[chosen]),int(columns[chosen]),int(starts[chosen])
            t0,t1=float(clocks[s]),float(clocks[i]);gaps=np.diff(clocks[s:i+1])
            candidates.append(dict(column=k,run_start_ms=t0,last_observed_head_ms=t1,
                observed_consecutive_H=int(age[i,k]),span_ms=t1-t0,
                median_HH_gap_ms=None if not len(gaps) else float(np.median(gaps)),
                maximum_HH_gap_ms=None if not len(gaps) else float(gaps.max()),
                started_before_scope=bool(t0<start_ms),
                future_membership_unobserved=bool(i==n-1)))
    return dict(heads=total,heads_with_previous_H=eligible,heads_repeating_previous_H_column=repeated,
        repeated_head_fraction=None if eligible==0 else repeated/eligible,
        repeat_heads_per_column=list(map(int,repeats.sum(0))),
        prefix_H_run_age_head_counts={str(int(k)):int(v) for k,v in zip(unique,count)},
        prefix_H_run_age=moments,maximum_observed_prefix_H_run_age=None if not len(values) else int(values.max()),
        witnesses=candidates,
        interpretation='Consecutive-H column membership with elapsed gaps; not a style label, bad-pattern threshold or physiological strain.')


def head_recurrence(trace, scope):
    """Return prefix-aware head membership and timed witnesses on one scope.

    Ages count consecutive head-bearing rows containing a column. They retain
    pre-scope history and never use events at/after the exclusive end. Release-
    only rows do not reset head membership; no elapsed-gap threshold is imposed.
    Durations and HH gaps must accompany any interpretation of a long run.
    These observations do not label Jack/Stream style or assign player demand.
    """
    trace._scope(scope)
    return dict(scope=dict(name=scope.name,start_ms=scope.start_ms,end_ms=scope.end_ms),
        **_observations(trace.times,trace.actions,scope.start_ms,scope.end_ms))
