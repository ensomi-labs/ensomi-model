"""Event-time reference state and legal-continuation response observations.

Finite time windows define observable attack/release rates and held-time
fractions. Their box kernels are measurement definitions, not inferred human
recovery laws. Ordered recent rows are retained for coordination modeling;
neither these coordinates nor an aggregate star rating are a complete C0.
"""
from bisect import bisect_right
from dataclasses import dataclass, field, replace
import math

import numpy as np

from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..oracle_time_continuation.schema import CompleteRow, checked_time
from ..scoped_style_modeling.dataset import ContractError

EMPTY_TIMES = ((), (), (), ())
DEFAULT_WINDOWS_MS = (500, 1000, 2000, 4000, 8000, 16000)


@dataclass(frozen=True)
class CommittedPlayState:
    """Immutable facts through an explicit row/no-row boundary.

    Construct from BOS and observe complete rows in order. ``advance`` fixes a
    silent interval; later observations cannot insert rows into that interval.
    Retention is measured in milliseconds, never row count. Open LN origins
    survive arbitrarily long silence, and no future endpoint is accepted.
    """
    time_ms: float = -1.
    retention_ms: float = 32000.
    replay: ExactReplayState = field(default_factory=ExactReplayState)
    rows: tuple[CompleteRow, ...] = ()
    row_times: tuple[float, ...] = ()
    attacks: tuple[tuple[float, ...], ...] = EMPTY_TIMES
    releases: tuple[tuple[float, ...], ...] = EMPTY_TIMES
    closed_holds: tuple[tuple[float, float, int], ...] = ()
    last_head_groups: tuple[tuple[float, int], ...] = ()

    def __post_init__(self):
        if not math.isfinite(self.retention_ms) or self.retention_ms <= 0:
            raise ContractError('Player-state retention requires positive milliseconds')

    def advance(self, end_ms):
        end = checked_time(end_ms)
        if end < self.time_ms:
            raise ContractError('Player-state time cannot move backwards')
        left = end-self.retention_ms
        first = bisect_right(self.row_times, left)
        trim = lambda collection: tuple(times[bisect_right(times, left):] for times in collection)
        return replace(self, time_ms=end, rows=self.rows[first:], row_times=self.row_times[first:],
            attacks=trim(self.attacks), releases=trim(self.releases),
            closed_holds=tuple(hold for hold in self.closed_holds if hold[1] > left))

    def observe(self, row: CompleteRow):
        if row.time_ms <= self.time_ms:
            raise ContractError('Observed row must follow all committed row/no-row decisions')
        replay = commit(self.replay, row)
        previous = self.advance(row.time_ms)
        attacks = tuple(times+(row.time_ms,) if action in (1, 2) else times
                        for times, action in zip(previous.attacks, row.actions))
        releases = tuple(times+(row.time_ms,) if action == 3 else times
                         for times, action in zip(previous.releases, row.actions))
        closed = previous.closed_holds+tuple((self.replay.open_ln_start_ms[lane], row.time_ms, lane)
            for lane, action in enumerate(row.actions) if action == 3)
        mask = sum(1 << lane for lane, action in enumerate(row.actions) if action in (1, 2))
        groups = (*self.last_head_groups, (row.time_ms, mask))[-2:] if mask else self.last_head_groups
        return replace(previous, replay=replay, rows=previous.rows+(row,),
            row_times=previous.row_times+(row.time_ms,), attacks=attacks, releases=releases,
            closed_holds=closed, last_head_groups=groups)

    @classmethod
    def from_rows(cls, rows, end_ms, *, retention_ms=32000.):
        state = cls(retention_ms=retention_ms)
        for row in rows:
            state = state.observe(row)
        return state.advance(end_ms)

    def window_facts(self, windows_ms=DEFAULT_WINDOWS_MS):
        """Return [window,column] rates in Hz and occupied-time fractions.

        Windows are (time-width,time]. A tap and an LN press each count as one
        attack; releases are separate. Holding has no invented release event.
        """
        widths = np.asarray(windows_ms, dtype=np.float64)
        if (widths.ndim != 1 or not len(widths) or not np.isfinite(widths).all() or
                np.any(widths <= 0) or np.any(widths > self.retention_ms)):
            raise ContractError('Response windows must fit the retained time history')
        rates = lambda values: np.array([[(len(times)-bisect_right(times, self.time_ms-width))*1000/width
                                         for times in values] for width in widths])
        held = np.zeros((len(widths), 4))
        intervals = (*self.closed_holds, *((start, self.time_ms, lane)
            for lane, start in enumerate(self.replay.open_ln_start_ms) if start is not None))
        for start, end, lane in intervals:
            held[:, lane] += np.maximum(0., np.minimum(end, self.time_ms)-np.maximum(start, self.time_ms-widths))/widths
        return dict(attack_hz=rates(self.attacks), release_hz=rates(self.releases), held_fraction=held)


@dataclass(frozen=True)
class CoordinationEvent:
    time_ms: float
    actions: tuple[int, ...]
    entering_holds: tuple[bool, ...]
    resulting_holds: tuple[bool, ...]
    previous_head_groups: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class ResponseObservations:
    start_ms: float
    end_ms: float
    windows_ms: tuple[float, ...]
    peak: dict[str, np.ndarray]
    terminal: dict[str, np.ndarray]
    attack_exposure: np.ndarray
    release_exposure: np.ndarray
    held_ms: np.ndarray
    events: tuple[CoordinationEvent, ...]
    state_at_end: CommittedPlayState


def observe_continuation(state, continuation, end_ms, *, windows_ms=DEFAULT_WINDOWS_MS):
    """Evaluate a private legal continuation over (state.time_ms,end_ms].

    No input is mutated. Rows after the horizon are rejected, and an empty
    continuation still advances time and integrates holding. Outputs are source
    observations for response calibration, not difficulty or quality scores.
    Local occupancy is checked; true audio-end closure remains caller-owned.
    Exposure integrates each moving rate over the requested future duration;
    units are head/release equivalents, distinct from its terminal or peak Hz.
    """
    end = checked_time(end_ms)
    if end <= state.time_ms:
        raise ContractError('A continuation response requires a positive time horizon')
    widths = tuple(float(value) for value in windows_ms)
    initial = state.window_facts(widths)
    peak = {name: values.copy() for name, values in initial.items()}
    current = state
    attacks, releases = [list(times) for times in state.attacks], [list(times) for times in state.releases]
    held_ms = np.zeros(4)
    events = []
    for row in continuation:
        if row.time_ms > end:
            raise ContractError('A proposed row lies after the response horizon')
        updated = current.observe(row)
        held_ms += (row.time_ms-current.time_ms)*np.asarray(current.replay.occupancy)
        events.append(CoordinationEvent(row.time_ms, row.actions, current.replay.occupancy,
                                        updated.replay.occupancy, current.last_head_groups))
        for lane, action in enumerate(row.actions):
            if action in (1, 2):
                attacks[lane].append(row.time_ms)
            elif action == 3:
                releases[lane].append(row.time_ms)
        current = updated
        for name, values in current.window_facts(widths).items():
            peak[name] = np.maximum(peak[name], values)
    held_ms += (end-current.time_ms)*np.asarray(current.replay.occupancy)
    current = current.advance(end)
    terminal = current.window_facts(widths)
    for name, values in terminal.items():
        peak[name] = np.maximum(peak[name], values)

    def exposure(collection):
        result = np.zeros((len(widths), 4))
        for i, width in enumerate(widths):
            for lane, times in enumerate(collection):
                times = np.asarray(times)
                overlap = np.maximum(0., np.minimum(end, times+width)-np.maximum(state.time_ms, times))
                result[i, lane] = overlap.sum()/width
        return result

    return ResponseObservations(state.time_ms, end, widths, peak, terminal,
        exposure(attacks), exposure(releases), held_ms, tuple(events), current)
