"""Local shared timing coordinates inferred from distinct H timestamps.

This descriptive fit reads no BPM, redline, audio or row contents. It is not a
quality gate: tempo changes, deliberate displacement and fine subdivisions
can all reduce coverage. Fit separate local/control scopes, not whole-song
averages that erase a change of rhythmic organization.
"""
import numpy as np


def lattice_coverage(head_times, *, minimum_heads=12, minimum_unit_ms=40.,
                     maximum_unit_ms=2000., tolerance_ms=3.,
                     coverage_levels=(.5, .8, .95)):
    """Fit finite shared-period/phase candidates to the supplied observations.

    A chord contributes one distinct H. Up to 64 time-spread observed anchors
    propose units from their index-lag 1/2/4/8 differences divided by 1..16,
    rounded to .001 ms. The phase is anchored at an observed H. Coarsest units
    meeting each requested coverage are returned separately from the maximum
    coverage at any candidate. This is an in-scope fit, not a beat estimate or
    held-out extrapolation; the finite search is not a continuous optimum.

    Bounds and tolerance are observation coordinates, never timing legality.
    Insufficient events or an empty candidate bank stays unobserved. Results
    should not be pooled across different controls or interpreted as a
    calibrated probability of bad rhythm.
    """
    times = np.asarray(head_times, dtype=np.float64)
    levels = tuple(coverage_levels)
    if (times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0)
            or type(minimum_heads) is not int or minimum_heads < 2
            or not np.isfinite([minimum_unit_ms, maximum_unit_ms, tolerance_ms]).all()
            or not 0 < minimum_unit_ms <= maximum_unit_ms
            or not 0 < tolerance_ms < minimum_unit_ms/2
            or not levels or len(set(levels)) != len(levels)
            or any(not np.isfinite(q) or not 0 < q <= 1 for q in levels)):
        raise ValueError('Lattice coverage needs finite native times and valid observation coordinates')
    times = np.unique(times)
    result = dict(heads=len(times), status='insufficient_events',
        minimum_heads=minimum_heads, minimum_unit_ms=minimum_unit_ms,
        maximum_unit_ms=maximum_unit_ms, tolerance_ms=tolerance_ms,
        coarsest_at_coverage={str(q): None for q in levels},
        maximum_coverage=None, maximum_coverage_unit_ms=None, candidate_units=0)
    if len(times) < minimum_heads:
        return result
    anchors = times[np.unique(np.linspace(0, len(times)-1, min(64, len(times))).round().astype(int))]
    deltas = np.concatenate([anchors[lag:]-anchors[:-lag]
                             for lag in (1, 2, 4, 8) if len(anchors) > lag])
    units = np.unique(np.round((deltas[:, None]/np.arange(1, 17)[None]).reshape(-1), 3))
    units = units[(units >= minimum_unit_ms) & (units <= maximum_unit_ms)]
    if not len(units):
        result['status'] = 'no_candidates'
        return result
    differences = times[None]-anchors[:, None]
    # Bound the temporary candidate × anchor × observation arrays.
    batch = max(1, min(64, 1_000_000//differences.size))
    best, phase = [], []
    for first in range(0, len(units), batch):
        p = units[first:first+batch, None, None]
        distance = np.abs(differences[None]-np.round(differences[None]/p)*p)
        counts = (distance <= tolerance_ms).sum(-1)
        chosen = counts.argmax(-1)
        best.extend(counts[np.arange(len(p)), chosen]/len(times))
        phase.extend(anchors[chosen])
    best, phase = np.asarray(best), np.asarray(phase)
    for required in levels:
        indices = np.flatnonzero(best >= required)
        if len(indices):
            i = int(indices[-1])
            result['coarsest_at_coverage'][str(required)] = dict(unit_ms=float(units[i]),
                anchor_ms=float(phase[i]), covered_fraction=float(best[i]))
    maximum = int(best.argmax())
    result.update(status='observed', maximum_coverage=float(best[maximum]),
        maximum_coverage_unit_ms=float(units[maximum]), candidate_units=len(units))
    return result


def head_lattice(trace, scope, **options):
    """Observe only distinct H in a declared local, half-open scope.

    No event outside the scope participates in this retrospective fit. Pure
    releases and the number of simultaneous fingers do not alter the result.
    The caller owns scope selection and keeps differing controls separate.
    """
    trace._scope(scope)
    times = trace.H[(trace.H >= scope.start_ms) & (trace.H < scope.end_ms)]
    return dict(start_ms=scope.start_ms, end_ms=scope.end_ms,
                **lattice_coverage(times, **options))
