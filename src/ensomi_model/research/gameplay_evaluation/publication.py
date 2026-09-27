"""Playback deadlines from actual immutable-publication wall-clock traces."""
import math

import numpy as np

from ..scoped_style_modeling.dataset import ContractError


def publication_report(events,duration_ms,*,lead_ms=2000,startup_seconds=None):
    """Measure startup needed to maintain a chosen settled-chart lookahead.

    Each event contains elapsed_seconds and inclusive settled-through coverage_ms.
    Time zero must be declared by the caller (for example, loaded weights plus
    cached Mel). Only published coverage counts, not private H or row proposals.
    Future LN endpoints need not be materialized before their protocol permits.
    An incomplete trace yields a prefix-only bound and cannot certify playback.
    """
    events=tuple(events)
    if (type(duration_ms) is not int or duration_ms<=0 or not math.isfinite(lead_ms) or lead_ms<0 or
            (startup_seconds is not None and (not math.isfinite(startup_seconds) or startup_seconds<0))):
        raise ContractError('Publication evaluation needs positive duration and nonnegative timing requirements')
    if not events:
        return dict(status='no_publication',duration_ms=duration_ms,lead_ms=lead_ms,
                    required_startup_seconds=None,coverage_complete=False)
    wall=np.asarray([e['elapsed_seconds'] for e in events],float)
    coverage=np.asarray([e['coverage_ms'] for e in events],float)
    if (not np.isfinite(wall).all() or not np.isfinite(coverage).all() or
            np.any(wall<0) or np.any(np.diff(wall)<0) or np.any(np.diff(coverage)<0) or
            np.any((coverage<0)|(coverage>duration_ms)) or np.any(coverage!=np.rint(coverage))):
        raise ContractError('Published clocks and wall time must be finite, monotone and inside audio coverage')
    previous=np.r_[-1.,coverage[:-1]]
    # g+1 is the first unobserved native millisecond. Before playback starts,
    # incomplete initial lookahead requires waiting for the next publication.
    credit=np.maximum(0.,(previous+1-lead_ms)/1000)
    lower_bound=wall-credit
    index=int(np.argmax(lower_bound))
    required=float(max(0,lower_bound[index]))
    complete=bool(coverage[-1]==duration_ms)
    result=dict(status='complete_trace' if complete else 'incomplete_trace',
        duration_ms=duration_ms,coverage_complete=complete,observed_through_ms=int(coverage[-1]),
        lead_ms=lead_ms,required_startup_seconds=required,
        generation_seconds=float(wall[-1]),whole_generation_to_audio_ratio=float(wall[-1]*1000/duration_ms),
        witness=dict(publication_index=index,elapsed_seconds=float(wall[index]),
            previous_coverage_ms=int(previous[index]),new_coverage_ms=int(coverage[index])),
        interpretation='A bound for this observed publication trace and declared clock origin, not a future-load guarantee.')
    if startup_seconds is not None:
        late=np.maximum(0.,lower_bound-startup_seconds)
        result.update(requested_startup_seconds=startup_seconds,
            deadline_misses=int(np.sum(late>1e-9)),maximum_lateness_seconds=float(late.max()),
            trace_meets_deadlines=bool(complete and not np.any(late>1e-9)))
    return result
