"""Raw R2 state, action support and replay-checked advance.

``R2State`` holds only committed facts: the exact replay state (open LN starts,
last attack/release per lane), the birth row of each open LN and the next
decision index. ``advance`` expands a decision into chronological complete rows
(gap releases sorted and coalesced, then the row at t_k) and commits each one
through the oracle replay, so an illegal decision raises. Mirroring reverses
every lane-indexed field; ``advance(Ms, Md) == M advance(s, d)``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..research.chart.actions import EMPTY, LN_CLOSE, LN_START, TAP
from ..research.oracle_time_continuation.replay import ExactReplayState, commit
from ..research.oracle_time_continuation.schema import CompleteRow
from .common import ACTIONS, ContractError

_CODES = ACTIONS  # [625,4]
_HEAD = {False: np.array([False, True, True, False, False]), True: np.array([False, False, False, True, True])}


@dataclass(frozen=True)
class Decision:
    codes: tuple[int, int, int, int]
    releases: tuple  # four optional gap release times (float or None)

    def __post_init__(self):
        if len(self.codes) != 4 or any(type(c) is not int or not 0 <= c <= 4 for c in self.codes):
            raise ContractError('A decision has four integer codes in 0..4')
        if len(self.releases) != 4:
            raise ContractError('A decision has four release slots')


@dataclass(frozen=True)
class R2State:
    k: int = 0
    replay: ExactReplayState = ExactReplayState()
    birth_row: tuple = (None, None, None, None)
    finalized: bool = False

    @property
    def held(self):
        return self.replay.occupancy


def action_support_mask(held, eos: bool) -> np.ndarray:
    """bool[625]: local legality, at least one head on a head row; EOS: held->2, free->0."""
    held = np.asarray(held, dtype=bool)
    if eos:
        want = np.where(held, 2, 0)
        return (_CODES == want[None]).all(-1)
    allowed = np.where(held[None], np.ones_like(_CODES, dtype=bool), _CODES <= 2)
    heads = np.where(held[None], _CODES >= 3, (_CODES == 1) | (_CODES == 2))
    return allowed.all(-1) & heads.any(-1)


def action_support(state: R2State, K: int) -> np.ndarray:
    if state.finalized or state.k > K:
        raise ContractError('No decision remains')
    return action_support_mask(state.held, state.k == K)


def expand(state: R2State, decision: Decision, head_ms, song_ms: float):
    """Chronological complete rows for one decision; validates codes against held lanes and gaps."""
    K = len(head_ms)
    k = state.k
    if state.finalized or k > K:
        raise ContractError('Chart already finalized')
    eos = k == K
    held = state.held
    if not action_support_mask(held, eos)[np.ravel_multi_index(decision.codes, (5, 5, 5, 5))]:
        raise ContractError(f'Decision {decision.codes} is outside the support at decision {k}')
    lo = float(head_ms[k - 1]) if k > 0 else None
    hi = float(head_ms[k]) if not eos else float(song_ms)
    gap = {}
    for lane, (c, r) in enumerate(zip(decision.codes, decision.releases)):
        needs = held[lane] and c in (2, 3, 4)
        if needs:
            if r is None or not lo < float(r) < hi:
                raise ContractError(f'Lane {lane} gap release {r} outside ({lo}, {hi})')
            gap.setdefault(float(r), []).append(lane)
        elif r is not None:
            raise ContractError(f'Lane {lane} has a release time without a gap-release code')
    rows = []
    for t in sorted(gap):
        actions = [EMPTY] * 4
        for lane in gap[t]:
            actions[lane] = LN_CLOSE
        rows.append(CompleteRow(t, tuple(actions)))
    if not eos:
        actions = [EMPTY] * 4
        for lane, c in enumerate(decision.codes):
            if held[lane]:
                actions[lane] = (EMPTY, LN_CLOSE, EMPTY, TAP, LN_START)[c]
            else:
                actions[lane] = (EMPTY, TAP, LN_START)[c]
        rows.append(CompleteRow(hi, tuple(actions)))
    return rows


def advance(state: R2State, decision: Decision, head_ms, song_ms: float):
    rows = expand(state, decision, head_ms, song_ms)
    K = len(head_ms)
    eos = state.k == K
    replay = state.replay
    for i, row in enumerate(rows):
        replay = commit(replay, row, is_terminal=eos and i == len(rows) - 1)
    births = list(state.birth_row)
    for lane, c in enumerate(decision.codes):
        if state.held[lane] and c in (1, 2, 3, 4):
            births[lane] = None
        if (not state.held[lane] and c == 2) or (state.held[lane] and c == 4):
            births[lane] = state.k
    if eos and any(replay.occupancy):
        raise ContractError('EOS left an open hold')
    return R2State(state.k + 1, replay, tuple(births), finalized=eos), tuple(rows)


def mirror_row(row: CompleteRow) -> CompleteRow:
    return CompleteRow(row.time_ms, tuple(reversed(row.actions)))


def mirror_state(state: R2State) -> R2State:
    r = state.replay
    replay = ExactReplayState(r.row_count, r.note_count, r.first_time_ms,
                              None if r.last_row is None else mirror_row(r.last_row),
                              tuple(reversed(r.open_ln_start_ms)), tuple(reversed(r.last_lane_attack_ms)),
                              tuple(reversed(r.last_lane_release_ms)), r.is_complete)
    return R2State(state.k, replay, tuple(reversed(state.birth_row)), state.finalized)


def mirror_decision(d: Decision) -> Decision:
    return Decision(tuple(reversed(d.codes)), tuple(reversed(d.releases)))


def replay_decisions(head_ms, song_ms, actions, gap_release_ms, upto=None):
    """Replay decisions [0, upto) from BOS; returns the final state and all physical rows."""
    state = R2State()
    rows = []
    n = len(actions) if upto is None else upto
    for k in range(n):
        d = decision_at(actions, gap_release_ms, k)
        state, new = advance(state, d, head_ms, song_ms)
        rows.extend(new)
    return state, rows


def decision_at(actions, gap_release_ms, k) -> Decision:
    codes = tuple(int(c) for c in actions[k])
    rel = tuple(None if np.isnan(x) else float(x) for x in gap_release_ms[k])
    return Decision(codes, rel)


def rows_to_objects(rows):
    from ..osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
    held = [None] * 4
    out = []
    for row in rows:
        for lane, a in enumerate(row.actions):
            if a == TAP:
                out.append(ManiaHitObject(row.time_ms, row.time_ms, lane, ManiaHitObjectKind.TAP))
            elif a == LN_START:
                held[lane] = row.time_ms
            elif a == LN_CLOSE:
                out.append(ManiaHitObject(held[lane], row.time_ms, lane, ManiaHitObjectKind.HOLD))
                held[lane] = None
    if any(h is not None for h in held):
        raise ContractError('Rows leave an open hold')
    return out
