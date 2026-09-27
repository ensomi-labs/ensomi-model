"""Select inspection contexts from measured pressure, across all response scales."""
from ..scoped_style_modeling.dataset import ContractError


def pressure_review_contexts(witnesses, coverage_ms, *, limit=3, margin_ms=1000):
    """Rank retained episodes by integrated excess, including their causal window.

    A modest four-second peak can conceal sustained eight/sixteen-second load.
    Ranking therefore uses the measured contribution across all reported windows,
    not a fixed scale or the largest instantaneous rate. Context may precede the
    scored scope to expose incoming pressure; its scope identity remains attached.
    Returned entries may overlap and are inspection priorities, not BAD labels.
    """
    if witnesses['status']!='measured':return []
    scope=witnesses['scope']
    if coverage_ms<scope['end_ms'] or limit<=0 or margin_ms<0:
        raise ContractError('Pressure review requires observed coverage and positive selection bounds')
    ranked=[]
    for width,window in witnesses['windows'].items():
        for episode in window['episodes']:
            peak=episode['peak_window']
            ranked.append(dict(window_ms=float(width),column=episode['column'],
                excess_seconds=episode['excess_seconds'],reference_Hz=window['reference_Hz'],
                requested_stars=witnesses['requested_stars'],scored_scope=dict(scope),
                start_ms=max(0,int(min(episode['start_ms'],peak['start_exclusive_ms'])-margin_ms)),
                end_ms=min(coverage_ms,int(max(episode['end_ms'],peak['end_inclusive_ms'])+margin_ms)),
                episode=episode,review_status='unreviewed'))
    ranked.sort(key=lambda r:(-r['excess_seconds'],r['episode']['start_ms'],r['window_ms'],r['column']))
    return ranked[:limit]
