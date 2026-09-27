"""Causal typed interactions between continuing holds and observed row actions.

Pair counts describe arrangement roles, not physiological strain or BAD labels.
One action under two continuing holds contributes two pairs. Co-start release
pairs are a subset of hold/release pairs and must not be added as extra workload.
"""
import numpy as np

from ..scoped_style_modeling.replay import HAND_COLUMNS


_HAND = {column: hand for hand, columns in enumerate(HAND_COLUMNS) for column in columns}
_KINDS = ('hold_tap', 'hold_LN_start', 'hold_release', 'co_start_hold_release',
          'release_tap', 'release_LN_start')


def hold_interactions(trace, scope):
    """Measure event relations on [start,end), keeping incoming real hold origins.

    A continuing hold started before the row and does not release on that row.
    Heads at its start or release clock are not interior companion heads.
    Release/head coincidences and release while another LN continues are counted
    separately, using the canonical hand mapping. Scope boundaries never close
    holds. Events and tails at/after the exclusive end are not consulted.

    Per-hold counts describe only observed interactions inside this scope, not
    full lifetime counts. Involved holds include those open just before the
    scope and those starting inside it, including a release at the start.
    """
    trace._scope(scope)
    a, b = scope.start_ms, scope.end_ms
    active = [None] * 4
    objects = {}
    pairs = {kind: dict(same_hand=0, opposite_hand=0) for kind in _KINDS}
    counts = dict(head_rows=0, tap_heads=0, LN_heads=0, releases=0,
                  head_rows_with_continuing_hold=0, tap_heads_with_continuing_hold=0,
                  LN_heads_with_continuing_hold=0, release_rows_with_continuing_hold=0)
    entered = False

    def register(column, start):
        key = (column, float(start))
        if key not in objects:
            objects[key] = dict(column=column, start_ms=float(start), started_before_scope=bool(start < a),
                release_ms_in_scope=None, companion_head_rows=0, companion_tap_heads=0,
                companion_LN_heads=0, companion_releases=0,
                first_interaction_ms=None, last_interaction_ms=None)
        return objects[key]

    def enter():
        for column, start in enumerate(active):
            if start is not None:
                register(column, start)

    def pair(kind, left, right):
        relation = 'same_hand' if _HAND[left] == _HAND[right] else 'opposite_hand'
        pairs[kind][relation] += 1

    for time, actions in zip(trace.times, trace.actions):
        if time >= b:
            break
        if time >= a:
            if not entered:
                enter()
                entered = True
            taps = [c for c, action in enumerate(actions) if action == 1]
            starts = [c for c, action in enumerate(actions) if action == 2]
            releases = [c for c, action in enumerate(actions) if action == 3]
            continuing = [c for c, start in enumerate(active) if start is not None and actions[c] != 3]
            has_heads = bool(taps or starts)
            counts['head_rows'] += int(has_heads)
            counts['tap_heads'] += len(taps)
            counts['LN_heads'] += len(starts)
            counts['releases'] += len(releases)
            if continuing:
                counts['head_rows_with_continuing_hold'] += int(has_heads)
                counts['tap_heads_with_continuing_hold'] += len(taps)
                counts['LN_heads_with_continuing_hold'] += len(starts)
                counts['release_rows_with_continuing_hold'] += int(bool(releases))
            for column in continuing:
                obj = register(column, active[column])
                obj['companion_head_rows'] += int(has_heads)
                obj['companion_tap_heads'] += len(taps)
                obj['companion_LN_heads'] += len(starts)
                obj['companion_releases'] += len(releases)
                if taps or starts or releases:
                    if obj['first_interaction_ms'] is None:
                        obj['first_interaction_ms'] = float(time)
                    obj['last_interaction_ms'] = float(time)
                for other in taps:
                    pair('hold_tap', column, other)
                for other in starts:
                    pair('hold_LN_start', column, other)
                for other in releases:
                    pair('hold_release', column, other)
                    if active[column] == active[other]:
                        pair('co_start_hold_release', column, other)
            for column in releases:
                register(column, active[column])['release_ms_in_scope'] = float(time)
                for other in taps:
                    pair('release_tap', column, other)
                for other in starts:
                    pair('release_LN_start', column, other)
            for column in starts:
                register(column, time)
        for column, action in enumerate(actions):
            if action == 2:
                active[column] = float(time)
            elif action == 3:
                active[column] = None
    if not entered:
        enter()
    values = list(objects.values())
    sizes = [v['companion_head_rows'] for v in values]
    moments = None if not sizes else dict(zip(('q10', 'q50', 'q90'), map(float, np.quantile(sizes, (.1, .5, .9)))))
    witnesses = [v for v in values if v['first_interaction_ms'] is not None]
    witnesses.sort(key=lambda v: (-(v['companion_tap_heads'] + v['companion_LN_heads'] + v['companion_releases']),
                                  v['start_ms'], v['column']))
    seconds = (b - a) / 1000
    return dict(scope=dict(name=scope.name, start_ms=a, end_ms=b), counts=counts,
        pair_counts=pairs, pair_rates_per_second={k: {r: v / seconds for r, v in p.items()} for k, p in pairs.items()},
        involved_holds=len(values), incoming_holds=sum(v['started_before_scope'] for v in values),
        holds_without_observed_release=sum(v['release_ms_in_scope'] is None for v in values),
        observed_companion_head_rows_per_hold=moments,
        tap_fraction_with_continuing_hold=(counts['tap_heads_with_continuing_hold'] / counts['tap_heads']
                                          if counts['tap_heads'] else None),
        largest_hold_contexts=witnesses[:5],
        interpretation='Typed event relations within this scope; no duration cutoff, style label or player-demand verdict.')
