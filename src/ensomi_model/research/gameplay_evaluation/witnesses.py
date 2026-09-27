"""Locate sustained attack-pressure episodes without assigning semantic BAD labels."""
import math

import numpy as np

from ..scoped_style_modeling.dataset import ContractError


def sustained_attack_witnesses(trace, scope, envelope, stars, *, limit_per_window=8):
    """Locate exact positive-excess intervals against a declared corpus envelope.

    Attack and expiry clocks partition each column's trailing rate. Adjacent
    positive intervals form an episode; a real interval of zero excess separates
    them. Scope boundaries clip observation, never clear incoming attacks. Each
    witness includes simultaneous attack/holding context on all four columns.

    Report every window's total integral before retaining its largest episodes.
    The integral matches the attack envelope's squared-relative-excess cost,
    averaged over its windows. It is a localization of that existing diagnostic,
    not independent quality evidence, physiological strain, or a jack ban.
    """
    trace._scope(scope)
    if type(limit_per_window) is not int or limit_per_window <= 0:
        raise ContractError('Attack witnesses need a positive per-window result limit')
    if stars is None:
        return dict(status='unknown_difficulty',windows={},excess_seconds=None)
    if not math.isfinite(stars):
        raise ContractError('Attack witnesses need a finite difficulty request')
    reference = envelope.rates(stars)
    reports = {}
    total = 0.
    a,b = scope.start_ms,scope.end_ms
    for width,maximum in zip(envelope.windows_ms,reference):
        episodes = []
        for column,times in enumerate(trace.heads):
            expiry = times+width
            edges = np.unique(np.r_[a,b,times[(times>a)&(times<b)],expiry[(expiry>a)&(expiry<b)]])
            left,right = edges[:-1],edges[1:]
            counts = (np.searchsorted(times,left,side='right')-
                      np.searchsorted(times,left-width,side='right'))
            rates = counts*1000/width
            positive = rates>maximum
            starts = np.flatnonzero(positive & ~np.r_[False,positive[:-1]])
            stops = np.flatnonzero(positive & ~np.r_[positive[1:],False])+1
            for first,last in zip(starts,stops):
                start,end = float(left[first]),float(right[last-1])
                intensity = (rates[first:last]/maximum-1.)**2
                excess = float((intensity*(right[first:last]-left[first:last])/1000).sum()/len(reference))
                peak = first+int(np.argmax(rates[first:last]))
                context = trace.interval_features(start,end)
                peak_time=float(left[peak])
                peak_counts=[int(np.searchsorted(t,peak_time,side='right')-
                                 np.searchsorted(t,peak_time-width,side='right')) for t in trace.heads]
                peak_held=(trace.held_area(peak_time)-trace.held_area(max(0.,peak_time-width)))/width
                episodes.append(dict(column=column,start_ms=start,end_ms=end,duration_ms=end-start,
                    starts_at_scope_boundary=bool(start==a),ends_at_scope_boundary=bool(end==b),
                    history_start_ms=start-width,excess_seconds=excess,
                    peak_Hz=float(rates[peak]),peak_at_ms=float(left[peak]),
                    peak_window=dict(start_exclusive_ms=peak_time-width,end_inclusive_ms=peak_time,
                        attacks_per_column=peak_counts,attack_Hz_per_column=[n*1000/width for n in peak_counts],
                        held_fraction_per_column=peak_held.tolist()),
                    H_Hz=float(context[0]),attack_Hz_per_column=context[1:5].tolist(),
                    held_fraction_per_column=context[5:9].tolist(),
                    LN_head_Hz=float(context[9]),release_Hz=float(context[10])))
        episodes.sort(key=lambda e:(-e['excess_seconds'],e['start_ms'],e['column']))
        integral=sum(e['excess_seconds'] for e in episodes)
        total+=integral
        key=str(int(width)) if float(width).is_integer() else str(width)
        reports[key]=dict(reference_Hz=float(maximum),excess_seconds=integral,
            episode_count=len(episodes),omitted_episodes=max(0,len(episodes)-limit_per_window),
            episodes=episodes[:limit_per_window])
    return dict(status='measured',scope=dict(name=scope.name,start_ms=a,end_ms=b),
        requested_stars=stars,envelope_reference=envelope.reference,excess_seconds=total,windows=reports,
        interpretation='Corpus-reference pressure witnesses; inspect organization and peer activity before judging quality.')
