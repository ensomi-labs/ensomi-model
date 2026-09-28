"""Conditional LN decisions on an explicit observed history and query clock.

Reference actions can assess calibration. A policy's own sampled actions cannot
establish quality through self-calibration; their reports describe decision
ownership and alternatives instead. Pure-R marks condition on an event time
already chosen upstream and do not evaluate the release timing law.
"""
from itertools import combinations
import math

import numpy as np

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..scoped_style_modeling.replay import HAND_COLUMNS


_ACTIONS = np.asarray(ROW_ACTIONS)
_HEADS = np.isin(_ACTIONS, (1, 2))
_RELEASES = _ACTIONS == 3
_HAS_HEAD = _HEADS.any(-1)
_NONEMPTY = (_ACTIONS != 0).any(-1)
_SUBSETS = _RELEASES @ (1 << np.arange(4))
_INDEX = {tuple(row): i for i, row in enumerate(ROW_ACTIONS)}
_HAND = {column: hand for hand, columns in enumerate(HAND_COLUMNS) for column in columns}


def _binary_summary(records):
    count = len(records)
    expected = sum(r['probability'] for r in records)
    observed = sum(r['observed'] for r in records)
    squared = sum((r['probability']-r['observed'])**2 for r in records)
    bins = []
    for i in range(10):
        selected = [r for r in records if min(9, int(r['probability']*10)) == i]
        bins.append(dict(lower=i/10, upper=(i+1)/10, count=len(selected),
                         predicted_sum=sum(r['probability'] for r in selected),
                         observed_sum=sum(r['observed'] for r in selected)))
    return dict(count=count, predicted_sum=expected, observed_sum=observed,
                brier_sum=squared, mean_brier=squared/count if count else None,
                mean_probability_error=(expected-observed)/count if count else None,
                bins=bins)


def release_calibration(trace, scope, query_times_ms, log_probabilities, *, origin):
    """Observe conditional release choices on exactly the rows in [start,end).

    query_times_ms must match the scope's row clocks in order, with one
    normalized 256-column log-probability vector in ROW_ACTIONS order per row.
    Scores must already include the evaluated policy's support/preferences.
    Finite entries define its available row support; physical/role violations
    are rejected. Normalization errors up to 2e-5 are reported and normalized
    away as arithmetic roundoff; larger errors, NaNs and positive infinity fail.

    origin is reference, generated or synthetic. Reference observations
    must be independent of the evaluated policy's draws to support calibration.
    No output establishes player demand or labels a pattern BAD.

    Incoming holds keep their real origins. No later endpoint is read and an
    open hold is censored. Per-hold H continuation products use probabilities
    at factual prefixes where that hold continued, not a free-running survival
    law: different sampled rows would create different later states.
    """
    if origin not in ('reference', 'generated', 'synthetic'):
        raise ValueError('Release observations require reference, generated or synthetic origin')
    trace._scope(scope)
    a, b = scope.start_ms, scope.end_ms
    clocks = trace.times[(trace.times >= a) & (trace.times < b)]
    queries = np.asarray(query_times_ms)
    lp = np.asarray(log_probabilities, dtype=np.float64)
    if not np.array_equal(queries, clocks) or lp.shape != (len(clocks), len(ROW_ACTIONS)):
        raise ValueError('Release probabilities must align with every scoped row clock and ROW_ACTIONS')
    if np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError('Release log probabilities contain a nonfinite value other than masked -infinity')
    probabilities = np.exp(lp)
    mass = probabilities.sum(-1)
    errors = np.abs(mass-1)
    if np.any(errors > 2e-5):
        raise ValueError('Release log probabilities must be normalized')
    probabilities /= mass[:, None]

    active = [None]*4
    holds, marginal, pairs, subsets = {}, [], [], []
    query_index = 0
    unsupported_rows = 0
    entered = False

    def register(column, start):
        key = (column, float(start))
        if key not in holds:
            holds[key] = dict(column=column, start_ms=float(start), incoming=bool(start < a),
                release_ms=None, release_role=None, release_probability=None,
                H_continue_opportunities=0, H_continue_survival_product=None,
                H_continue_surprisal_sum=0., certain_premature_H_hazards=0,
                _log_continue=0.)
        return holds[key]

    def enter():
        for column, start in enumerate(active):
            if start is not None:
                register(column, start)

    for time, actions in zip(trace.times, trace.actions):
        if time >= b:
            break
        if time >= a:
            if not entered:
                enter()
                entered = True
            probability = probabilities[query_index]
            supported = np.isfinite(lp[query_index])
            query_index += 1
            held = np.asarray([v is not None for v in active])
            head = bool(np.isin(actions, (1, 2)).any())
            role = 'H' if head else 'R'
            physical = (_NONEMPTY & ~(_HEADS & held).any(-1)
                        & ~(_RELEASES & ~held).any(-1) & (_HAS_HEAD == head))
            if np.any(supported & ~physical):
                raise ValueError('Finite row support contradicts entering LN occupancy or the event role')
            actual_index = _INDEX[tuple(actions)]
            unsupported_rows += int(not supported[actual_index])
            release_probability = probability @ _RELEASES
            actual_release = actions == 3
            # (time-1000,time): only strict-past H contributes.
            history_H_hz = float(np.searchsorted(trace.H, time, side='left')
                                 - np.searchsorted(trace.H, time-1000, side='right'))
            subset_probabilities = np.bincount(_SUBSETS, weights=probability, minlength=16)
            actual_subset = int(actual_release @ (1 << np.arange(4)))
            error = subset_probabilities.copy()
            error[actual_subset] -= 1
            subset_support = sorted(map(int, np.unique(_SUBSETS[supported])))
            subsets.append(dict(time_ms=float(time), role=role,
                entering_hold_mask=int(held @ (1 << np.arange(4))),
                observed_subset=actual_subset, probabilities=subset_probabilities.tolist(),
                brier=float(error @ error), support=subset_support,
                any_release_required=bool(0 not in subset_support),
                observed_row_supported=bool(supported[actual_index])))
            for column in np.flatnonzero(held).tolist():
                p = float(np.clip(release_probability[column], 0., 1.))
                observed = bool(actual_release[column])
                can_continue = bool(np.any(supported & ~_RELEASES[:, column]))
                can_release = bool(np.any(supported & _RELEASES[:, column]))
                marginal.append(dict(time_ms=float(time), role=role, column=column,
                    start_ms=float(active[column]), age_ms=float(time-active[column]),
                    previous_H_hz=history_H_hz, other_held=int(held.sum()-1),
                    probability=p, observed=observed,
                    can_continue=can_continue, can_release=can_release))
                obj = register(column, active[column])
                if observed:
                    obj.update(release_ms=float(time), release_role=role, release_probability=p)
                elif head:
                    obj['H_continue_opportunities'] += 1
                    obj['_log_continue'] += math.log1p(-p) if p < 1. else -math.inf
                    obj['certain_premature_H_hazards'] += int(p == 1.)
            for left, right in combinations(np.flatnonzero(held).tolist(), 2):
                joint = float(probability @ (_RELEASES[:, left] & _RELEASES[:, right]))
                pairs.append(dict(time_ms=float(time), role=role, columns=[left, right],
                    same_hand=bool(_HAND[left] == _HAND[right]),
                    same_origin=bool(active[left] == active[right]),
                    ages_ms=[float(time-active[left]), float(time-active[right])],
                    probability=joint, observed=bool(actual_release[left] and actual_release[right]),
                    independent_probability=float(release_probability[left]*release_probability[right])))
            for column, action in enumerate(actions):
                if action == 2:
                    register(column, time)
        for column, action in enumerate(actions):
            if action == 2:
                active[column] = float(time)
            elif action == 3:
                active[column] = None
    if not entered:
        enter()
    objects = list(holds.values())
    for obj in objects:
        value = obj.pop('_log_continue')
        if obj['H_continue_opportunities']:
            obj['H_continue_survival_product'] = math.exp(value)
        obj['H_continue_surprisal_sum'] = -value if math.isfinite(value) else None
        obj['censored'] = obj['release_ms'] is None
    summary = {}
    for role in ('H', 'R'):
        risks = [r for r in marginal if r['role'] == role]
        joint = [r for r in pairs if r['role'] == role]
        events = [r for r in subsets if r['role'] == role]
        # Joint subset calibration only has a release question when a hold
        # enters. Free rows have a trivial all-zero release mask.
        exposed = [r for r in events if r['entering_hold_mask']]
        summary[role] = dict(marginal=_binary_summary(risks), pair=_binary_summary(joint),
            rows=len(events), subset_rows=len(exposed),
            subset_brier_sum=sum(r['brier'] for r in exposed),
            subset_mean_brier=(sum(r['brier'] for r in exposed)/len(exposed) if exposed else None),
            required_release_rows=sum(r['any_release_required'] for r in exposed),
            observed_releases_without_continue_option=sum(r['observed'] and not r['can_continue'] for r in risks))
    return dict(scope=dict(name=scope.name, start_ms=a, end_ms=b), origin=origin,
        maximum_normalization_error=float(errors.max()) if len(errors) else 0.,
        unsupported_observed_rows=unsupported_rows, summary=summary,
        marginal_risks=marginal, pair_risks=pairs, release_subsets=subsets, holds=objects,
        interpretation=('Conditional decisions and teacher-path survival; pure-R timing is not assessed. '
                        'Self-calibration on generated actions is not quality evidence. '
                        'No tail cutoff or player-demand verdict.'))
