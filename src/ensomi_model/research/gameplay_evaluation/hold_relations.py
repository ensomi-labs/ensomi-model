"""LN timing relations in real time and relative to the actual head sequence.

These retrospective descriptors separate H-interval variation from release-span
choices. They are neither a beat grid nor a calibrated player-response cost.
Short, unequal and irregular holds remain part of the chart language.
"""
import numpy as np


def _quantiles(values):
    if not len(values):return None
    return dict(zip(('q10','q50','q90'),map(float,np.quantile(values,(.1,.5,.9)))))


def hold_relations(trace,scope,*,predecessor_gap_ms=2000):
    """Describe holds starting in a scope without clipping or inventing tails.

    A completed LN keeps its actual duration even when it ends beyond the scope.
    An open hold at observed coverage is censored and supplies no duration. A
    preceding LN onset group may be before the scope: boundaries do not reset
    chart context. Simultaneous lengths are a separate within-group relation.

    H coordinates linearly interpolate the observed, distinct H timestamps.
    Their difference measures a hold's span across that sequence, not beats.
    Tails beyond the last observed H have no H coordinate. All quantities use
    only this trace's observed coverage, which can be a private future horizon.
    A lower variation descriptor is not an acceptance objective.
    """
    trace._scope(scope)
    if not np.isfinite(predecessor_gap_ms) or predecessor_gap_ms<=0:
        raise ValueError('LN predecessor relation needs a positive elapsed-time gap')
    heads=trace.H
    groups={};censored=0;selected_count=0;latest_tail=None
    for lane,holds in enumerate(trace.holds):
        for start,end in holds:
            if start>=scope.end_ms:continue
            selected=start>=scope.start_ms
            if selected:selected_count+=1
            group=groups.setdefault(float(start),[])
            if end==trace.coverage_ms:
                if selected:censored+=1
                continue
            span=(float(np.interp(end,heads,np.arange(len(heads)))-np.searchsorted(heads,start))
                  if len(heads)>1 and end<=heads[-1] else None)
            group.append((float(end-start),span,lane))
            if selected:latest_tail=float(end) if latest_tail is None else max(latest_tail,float(end))
    durations=[];spans=[];changes_ms=[];changes_H=[];within=[];witnesses=[];previous=None
    selected_groups=0;groups_with_tails=0;missing_H=0
    for start,notes in sorted(groups.items()):
        lengths=np.array([n[0] for n in notes]);hs=np.array([n[1] for n in notes if n[1] is not None])
        if start>=scope.start_ms:
            selected_groups+=1;durations.extend(lengths);spans.extend(hs)
            groups_with_tails+=bool(len(lengths))
            missing_H+=len(notes)-len(hs)
            if len(lengths)>1:within.append(float(np.log(lengths.max()/lengths.min())))
            if previous is not None and len(previous[1]) and start-previous[0]<=predecessor_gap_ms:
                distance=np.abs(np.log(lengths[:,None]/previous[1][None]))
                nearest=distance.argmin(-1);changes=distance[np.arange(len(lengths)),nearest]
                changes_ms.extend(changes)
                if len(hs) and len(previous[2]):
                    changes_H.extend(np.abs(np.log(hs[:,None]/previous[2][None])).min(-1))
                for i,(duration,span,lane) in enumerate(notes):
                    witnesses.append(dict(start_ms=start,column=lane,previous_group_ms=previous[0],
                        duration_ms=duration,nearest_previous_duration_ms=float(previous[1][nearest[i]]),
                        abs_log_change_ms=float(changes[i]),H_span=span))
        previous=(start,lengths,hs)
    return dict(LN_heads=selected_count,completed_LNs=len(durations),censored_LNs=censored,
        LN_onset_groups=selected_groups,onset_groups_with_observed_tails=groups_with_tails,
        latest_selected_tail_ms=latest_tail,
        observed_through_ms=trace.coverage_ms,predecessor_gap_ms=predecessor_gap_ms,
        duration_ms=_quantiles(durations),H_span=_quantiles(spans),missing_tail_H_coordinate=missing_H,
        nearest_previous_group_log_change_ms=_quantiles(changes_ms),
        nearest_previous_group_log_change_H=_quantiles(changes_H),
        compared_lengths_ms=len(changes_ms),compared_lengths_H=len(changes_H),
        within_group_log_duration_spread=_quantiles(within),
        largest_ms_changes=sorted(witnesses,key=lambda v:-v['abs_log_change_ms'])[:5],
        interpretation='Observed LN timing relationships; no shape mask, style label or player-response verdict.')
