"""Typed TAP group continuity and TAP bodies under one continuing held role.

Counts preserve every H, including those with no TAP. They describe organization
at an explicit scope; they do not assign Stream/Jumpstream or BAD-pattern labels.
"""
from collections import Counter

import numpy as np


def _columns(mask):
    return [c for c in range(4) if mask & (1<<c)]


def _runs(records,key):
    runs=[];current=[];previous=None
    def finish():
        if not current:return
        times=np.array([r['time_ms'] for r in current]);masks=[r['tap_mask'] for r in current]
        sizes=[len(_columns(m)) for m in masks];gaps=np.diff(times)
        pairs=list(zip(masks,masks[1:]));size_pairs=list(zip(sizes,sizes[1:]))
        value=dict(start_ms=float(times[0]),last_H_ms=float(times[-1]),span_ms=float(times[-1]-times[0]),
            H_rows=len(current),TAP_heads=sum(sizes),TAP_group_sizes=dict(Counter(sizes)),
            first_TAP_groups=[_columns(m) for m in masks[:16]],
            median_H_gap_ms=float(np.median(gaps)) if len(gaps) else None,
            maximum_H_gap_ms=float(gaps.max()) if len(gaps) else None,
            same_TAP_group_pairs=sum(a==b for a,b in pairs),
            moving_single_pairs=sum(a.bit_count()==b.bit_count()==1 and a!=b for a,b in pairs),
            single_double_alternations=sum({a,b}=={1,2} for a,b in size_pairs))
        if key=='anchor':value['held_origin']=dict(column=previous[0],start_ms=previous[1])
        runs.append(value)
    for r in records:
        identity=r[key]
        if identity is None or (current and identity!=previous):
            finish();current=[]
        if identity is not None:current.append(r)
        previous=identity
    finish()
    return sorted(runs,key=lambda r:(-r['H_rows'],-r['TAP_heads'],r['start_ms']))


def tap_organization(trace,scope):
    """Read scoped H groups with incoming occupancy and true release chronology.

    A single-held TAP body requires exactly one pre-existing continuing LN,
    no new LN in the row, and at least one TAP on the other fingers. Its birth
    and release rows are excluded. A changed held origin or an intervening H
    without that relation breaks the body. Pure R rows update occupancy but
    are not H groups. No tail at/after scope end is read or fabricated.

    Runs contain only H rows inside this one scope. No elapsed-gap threshold
    converts them into semantic episodes: elapsed spans and gap statistics must
    accompany interpretation. Mask transitions include zero-TAP H rows, so
    unrelated TAP groups cannot be joined by dropping LN-only events.
    """
    trace._scope(scope)
    active=[None]*4;records=[]
    for time,actions in zip(trace.times,trace.actions):
        if time>=scope.end_ms:break
        taps=sum(1<<c for c,a in enumerate(actions) if a==1)
        starts=sum(1<<c for c,a in enumerate(actions) if a==2)
        if time>=scope.start_ms and (taps or starts):
            continuing=[(c,t) for c,t in enumerate(active) if t is not None and actions[c]!=3]
            anchor=continuing[0] if len(continuing)==1 and not starts and taps else None
            records.append(dict(time_ms=float(time),tap_mask=taps,LN_mask=starts,
                TAP_body=True if taps and not starts else None,anchor=anchor))
        for c,a in enumerate(actions):
            if a==2:active[c]=float(time)
            elif a==3:active[c]=None
    pairs=Counter((a['tap_mask'],b['tap_mask']) for a,b in zip(records,records[1:]))
    ordinary=_runs(records,'TAP_body');held=_runs(records,'anchor')
    return dict(H_rows=len(records),TAP_bearing_H=sum(bool(r['tap_mask']) for r in records),
        heads_only_TAP_H=sum(r['TAP_body'] is not None for r in records),
        single_held_TAP_H=sum(r['anchor'] is not None for r in records),
        TAP_mask_counts={str(k):v for k,v in sorted(Counter(r['tap_mask'] for r in records).items())},
        TAP_mask_pairs=[dict(previous=_columns(a),current=_columns(b),count=n)
            for (a,b),n in sorted(pairs.items(),key=lambda p:(-p[1],p[0]))],
        longest_heads_only_TAP_H_run=max((r['H_rows'] for r in ordinary),default=0),
        longest_single_held_TAP_H_run=max((r['H_rows'] for r in held),default=0),
        heads_only_TAP_witnesses=ordinary[:5],single_held_TAP_witnesses=held[:5],
        interpretation='Typed scoped group relations, with no semantic style, tempo-grid or player-pressure verdict.')
