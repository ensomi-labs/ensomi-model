"""Offline short-LN exposure against a declared ranked-chart population.

The duration cut is an observation coordinate. Only its corpus-calibrated
prevalence is compared; individual short holds are never rejected by this tool.
"""
from collections import Counter
import math

import numpy as np


def short_ln_exposure(objects, start_ms, end_ms, *, threshold_ms=80.):
    """Describe heads in [start,end), using their fully resolved actual tails.

    This is an offline head-cohort measurement, not a causal frontier value.
    resolved_through_ms declares the latest tail it reads, including beyond the
    scope end. Objects must retain their true endpoints; never fabricate crop
    closures or silently pass unresolved holds as taps. Burden is short LN
    heads / all heads, while fraction_of_LNs has a distinct LN-only denominator.
    Empty denominators remain None. TAPs cannot contribute short LN exposure.
    """
    if not (math.isfinite(start_ms) and math.isfinite(end_ms) and end_ms > start_ms
            and math.isfinite(threshold_ms) and threshold_ms > 0):
        raise ValueError('LN exposure needs a positive scope and duration coordinate')
    selected = [o for o in objects if start_ms <= o.start_time < end_ms]
    holds = [o for o in selected if o.end_time > o.start_time]
    short = sum(o.end_time-o.start_time <= threshold_ms for o in holds)
    return dict(start_ms=start_ms, end_ms=end_ms, threshold_ms=threshold_ms,
        heads=len(selected), LN_heads=len(holds), short_LNs=short,
        burden=short/len(selected) if selected else None,
        fraction_of_LNs=short/len(holds) if holds else None,
        LN_fraction=len(holds)/len(selected) if selected else None,
        resolved_through_ms=max([end_ms, *(o.end_time+1 for o in holds)]))


def ln_usage_band(fraction):
    """Descriptive amount strata, not semantic style labels."""
    if fraction is None:
        return 'unspecified'
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('LN fraction must lie between zero and one')
    return 'none' if fraction == 0 else 'tap_majority' if fraction < .2 else 'mixed' if fraction < .6 else 'LN_majority'


def fit_fragmentation_reference(records, *, stars_min=3.5, stars_max=4.5,
                                requested_LN_fraction=None, quantile=.99):
    """Fit an upper prevalence reference with equal song, then chart weight.

    Each record describes one complete native-time reference chart: group_id,
    source_sha256, stars, heads, LN_heads, short_LNs and threshold_ms. Caller
    owns ranked status, provenance and split selection. Filter on declared
    difficulty and requested amount, never the generated chart's realized
    amount. Missing LN control uses the natural mixture, including TAP charts.
    This in-sample reference is not a held-out false-positive guarantee.
    """
    if not 0 < quantile < 1 or not stars_min <= stars_max:
        raise ValueError('Fragmentation reference requires a valid band and quantile')
    band = ln_usage_band(requested_LN_fraction)
    chosen = [r for r in records if stars_min <= r['stars'] <= stars_max and r['heads'] > 0
              and (band == 'unspecified' or ln_usage_band(r['LN_heads']/r['heads']) == band)]
    if not chosen:
        raise ValueError('The requested fragmentation reference has no observations')
    thresholds = {r['threshold_ms'] for r in chosen}
    if len(thresholds) != 1:
        raise ValueError('Reference charts must share one duration coordinate')
    counts = Counter(r['group_id'] for r in chosen)
    values = np.array([r['short_LNs']/r['heads'] for r in chosen])
    weights = np.array([1/counts[r['group_id']] for r in chosen])
    order = np.argsort(values, kind='stable')
    index = np.searchsorted(np.cumsum(weights[order]), quantile*weights.sum())
    limit = float(values[order[min(int(index), len(order)-1)]])
    return dict(stars_min=stars_min, stars_max=stars_max, amount_band=band,
        threshold_ms=next(iter(thresholds)), quantile=quantile, maximum_burden=limit,
        charts=len(chosen), song_groups=len(counts),
        weighting='Within the declared cohort: equal song group, then equal chart.',
        reference_exceedance_weight=float(weights[values > limit].sum()/weights.sum()))


def fragmentation_check(observation, reference):
    """Compare an offline observation; this alone cannot qualify a chart.

    Amount, excessive lengths/coverage, musical organization and native
    runtime remain independent guards. Reducing LN amount or lengthening all
    tails can reduce this coordinate without improving overall generation.
    """
    if observation['threshold_ms'] != reference['threshold_ms']:
        raise ValueError('Observation and reference use different duration coordinates')
    value = observation['burden']
    return dict(name='short_LN_exposure', status='unobserved' if value is None else
        'passed' if value <= reference['maximum_burden'] else 'failed',
        observed=value, limit=reference['maximum_burden'],
        threshold_ms=reference['threshold_ms'], heads=observation['heads'],
        LN_heads=observation['LN_heads'], short_LNs=observation['short_LNs'])
