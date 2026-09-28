"""Committed action-transition responses, independent of the requested level.

Exponential memory preserves load across intervening events and silent time.
Reciprocal-gap impulses are a convex speed-response hypothesis, not measured
human force or fatigue. Ranked references calibrate their scale separately.
"""
from dataclasses import dataclass, field, replace
from collections import Counter

import numpy as np

from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..oracle_time_continuation.schema import checked_time

TAUS_MS = (250., 1000., 4000., 16000.)
KINDS = ('attack', 'HH_speed', 'RH_speed', 'HR_speed', 'release',
         'hand_events', 'hand_turnover', 'partner_held_attack')
SLICES = (slice(0,4), slice(4,8), slice(8,12), slice(12,16),
          slice(16,20), slice(20,22), slice(22,24), slice(24,28))


def _impulse(replay, last_hand, row):
    values = np.zeros(28)
    now, actions = row.time_ms, row.actions
    heads = np.isin(actions, (1, 2))
    speed = lambda previous: 0. if previous is None else 100./(now-previous)
    for lane, action in enumerate(actions):
        head, release = replay.last_lane_attack_ms[lane], replay.last_lane_release_ms[lane]
        if heads[lane]:
            values[lane] = 1
            values[4+lane] = speed(head)
            values[8+lane] = speed(release) if release is not None and (head is None or release > head) else 0
            partner = lane ^ 1
            values[24+lane] = float(replay.occupancy[partner] and actions[partner] != 3)
        elif action == 3:
            values[12+lane] = speed(replay.open_ln_start_ms[lane])
            values[16+lane] = 1
    hands = list(last_hand)
    for hand in range(2):
        count = sum(a != 0 for a in actions[2*hand:2*hand+2])
        if count:
            values[20+hand] = 1
            values[22+hand] = count*speed(last_hand[hand])
            hands[hand] = now
    return values, tuple(hands)


@dataclass(frozen=True)
class ActionResponseState:
    """A branchable exact prefix and request-independent transition memory.

    TAP release times are never invented. RH refers only to the first attack
    after an actual LN release; HR is the actual LN's age at release. An open
    hold remains in exact occupancy while transient responses decay.
    """
    time_ms: float = -1.
    replay: ExactReplayState = field(default_factory=ExactReplayState)
    last_hand: tuple = (None, None)
    values: np.ndarray = field(default_factory=lambda: np.zeros((len(TAUS_MS),28)))

    def advance(self, end_ms):
        end = checked_time(end_ms)
        if end < self.time_ms:
            raise ValueError('Action response cannot move behind committed coverage')
        values = self.values*np.exp(-(end-self.time_ms)/np.asarray(TAUS_MS))[:,None]
        return replace(self, time_ms=end, values=values)

    def observe(self, row):
        if row.time_ms <= self.time_ms:
            raise ValueError('Action response needs a row after committed coverage')
        replay = commit(self.replay, row)
        impulse, hands = _impulse(self.replay, self.last_hand, row)
        decayed = self.advance(row.time_ms)
        values = decayed.values+impulse[None]*1000/np.asarray(TAUS_MS)[:,None]
        return replace(decayed, replay=replay, last_hand=hands, values=values)

    @classmethod
    def from_rows(cls, rows, end_ms):
        state = cls()
        for row in rows:
            state = state.observe(row)
        return state.advance(end_ms)


def transition_impulses(times, actions):
    """Vectorized full-source impulses for corpus calibration, from true rows.

    The caller supplies a verified legal chart from BOS. Returns [row,28]
    with the same coordinates as ActionResponseState; no window resets.
    """
    times, actions = np.asarray(times, float), np.asarray(actions)
    if (times.ndim != 1 or actions.shape != (len(times),4) or
            not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Transition impulses require increasing complete-row times')
    result = np.zeros((len(times),28))
    if not len(times):
        return result
    indices = np.arange(len(times))
    def prior(mask):
        return np.r_[-1, np.maximum.accumulate(np.where(mask, indices, -1))[:-1]]
    def speed(previous):
        gaps = times-times[np.maximum(previous,0)]
        return np.divide(100., gaps, out=np.zeros_like(gaps), where=(previous >= 0) & (gaps > 0))
    starts = []
    for lane in range(4):
        a = actions[:,lane]
        heads, releases = np.isin(a,(1,2)), a == 3
        h, r, birth = prior(heads), prior(releases), prior(a == 2)
        result[:,lane] = heads
        result[:,4+lane] = heads*speed(h)
        result[:,8+lane] = heads*(r > h)*speed(r)
        result[:,12+lane] = releases*speed(birth)
        result[:,16+lane] = releases
        starts.append((birth >= 0) & (birth > r))
    for lane in range(4):
        result[:,24+lane] = np.isin(actions[:,lane],(1,2))*starts[lane ^ 1]*(actions[:,lane ^ 1] != 3)
    for hand in range(2):
        count = (actions[:,2*hand:2*hand+2] != 0).sum(-1)
        result[:,20+hand] = count > 0
        result[:,22+hand] = count*speed(prior(count > 0))
    return result


def source_peaks(times, actions):
    """Exact whole-prefix maxima [memory scale, response kind].

    Positive impulse responses only decrease between rows. Eight-second
    numerical blocks bound exponent magnitudes; they do not reset memory.
    """
    times = np.asarray(times, float)
    impulses = transition_impulses(times, actions)
    tau = np.asarray(TAUS_MS)[:,None,None]
    values = np.zeros((len(TAUS_MS),28))
    peaks = values.copy()
    previous, first = -1., 0
    while first < len(times):
        stop = int(np.searchsorted(times, times[first]+8000, side='right'))
        relative = times[first:stop]-times[first]
        entering = values*np.exp(-(times[first]-previous)/tau[:,0])
        exponent = np.exp(relative[None,:,None]/tau)
        path = (entering[:,None]+np.cumsum(exponent*impulses[None,first:stop]*1000/tau,axis=1))/exponent
        peaks = np.maximum(peaks, path.max(1))
        values, previous, first = path[:,-1], times[stop-1], stop
    return np.stack([peaks[:,s].max(-1) for s in SLICES], -1)


@dataclass(frozen=True)
class ActionEnvelope:
    """Ranked-chart response references, not an identified physiological C0."""
    stars: tuple
    maximum: tuple
    reference: str

    def __post_init__(self):
        values = np.asarray(self.maximum)
        if (len(self.stars) < 2 or np.any(np.diff(self.stars) <= 0)
                or values.shape != (len(self.stars),len(TAUS_MS),len(KINDS))
                or not np.isfinite(values).all() or np.any(values < 0)):
            raise ValueError('Action reference requires ordered levels and finite response coordinates')

    def limits(self, stars):
        table = np.asarray(self.maximum).reshape(len(self.stars),-1)
        values = np.array([np.interp(stars,self.stars,c) for c in table.T])
        kinds = values.reshape(len(TAUS_MS),len(KINDS))
        result = np.empty((len(TAUS_MS),28))
        for i,s in enumerate(SLICES):
            result[:,s] = kinds[:,i,None]
        return result


def fit_action_envelope(stars, groups, peaks, *, knots, reference, quantile=.99, band_width=1.):
    """Equal-song then equal-chart reference quantiles, monotone over level.

    Inputs must be verified source charts, never generated failures. Return
    raw quantiles/counts as well as the monotone reference. This is in-sample
    calibration, not a held-out false-positive guarantee.
    """
    stars, peaks = np.asarray(stars), np.asarray(peaks)
    if (stars.ndim != 1 or len(groups) != len(stars)
            or peaks.shape != (len(stars),len(TAUS_MS),len(KINDS))
            or not np.isfinite(stars).all() or not np.isfinite(peaks).all()
            or not 0 < quantile < 1 or band_width <= 0):
        raise ValueError('Action calibration needs aligned finite source observations')
    raw, counts = [], []
    for knot in knots:
        selected = np.flatnonzero((stars >= knot-band_width/2) & (stars < knot+band_width/2))
        if not len(selected):
            raise ValueError('No source charts in an action-reference band')
        frequencies = Counter(groups[i] for i in selected)
        weights = np.array([1/frequencies[groups[i]] for i in selected])
        columns = peaks[selected].reshape(len(selected),-1).T
        values = []
        for column in columns:
            order = np.argsort(column,kind='stable')
            i = min(int(np.searchsorted(np.cumsum(weights[order]),quantile*weights.sum())),len(order)-1)
            values.append(float(column[order[i]]))
        raw.append(np.array(values).reshape(len(TAUS_MS),len(KINDS)))
        counts.append(dict(stars=knot,charts=len(selected),groups=len(frequencies)))
    monotone = np.maximum.accumulate(raw,axis=0)
    return ActionEnvelope(tuple(knots),tuple(map(lambda r: tuple(map(tuple,r)),monotone)),reference), dict(
        quantile=quantile,band_width=band_width,raw=np.asarray(raw).tolist(),bands=counts,
        taus_ms=TAUS_MS,kinds=KINDS)


def action_response(state, continuation, end_ms, envelope, ranges):
    """Integrate excess over private futures, including inherited transitions.

    Every range owns its level; changing it never resets the state. Exact
    exponential integrals include no-row time. Per-kind results remain
    visible instead of treating one global scalar as complete playability.
    Missing level requests are unscored, not evidence of zero demand.
    """
    future, ranges = tuple(continuation), tuple(ranges)
    if (not ranges or ranges[0][0] != state.time_ms or ranges[-1][1] != end_ms
            or any(b <= a for a,b,d in ranges)
            or any(x[1] != y[0] for x,y in zip(ranges,ranges[1:]))
            or any(r.time_ms <= state.time_ms or r.time_ms > end_ms for r in future)):
        raise ValueError('Action response needs a complete real-time horizon and ordered private rows')
    current, index, reports = state, 0, []
    tau = np.asarray(TAUS_MS)[:,None]/1000
    for begin,end,level in ranges:
        costs = np.zeros_like(current.values)
        peak = np.zeros_like(current.values)
        limits = None if level is None else np.maximum(envelope.limits(level),1e-9)
        while current.time_ms < end:
            next_row = future[index] if index < len(future) else None
            stop = min(end,next_row.time_ms) if next_row is not None else end
            seconds = (stop-current.time_ms)/1000
            if limits is not None:
                ratio = current.values/limits
                peak = np.maximum(peak,ratio)
                active = np.minimum(seconds,tau*np.log(np.maximum(ratio,1.)))
                costs += np.maximum(0., tau*(.5*ratio**2*(-np.expm1(-2*active/tau))
                    -2*ratio*(-np.expm1(-active/tau)))+active)
            if next_row is not None and stop == next_row.time_ms:
                current = current.observe(next_row)
                index += 1
                if limits is not None:
                    peak = np.maximum(peak,current.values/limits)
            else:
                current = current.advance(stop)
        by_kind = {k:float(costs[:,s].sum(-1).mean()) for k,s in zip(KINDS,SLICES)}
        reports.append(dict(start_ms=begin,end_ms=end,stars=level,
            excess_seconds=sum(by_kind.values()),excess_by_kind=by_kind,
            peak_ratio=None if limits is None else float(peak.max()),scored=level is not None))
    fully_scored = all(r['scored'] for r in reports)
    return current,dict(excess_seconds=sum(r['excess_seconds'] for r in reports),ranges=reports,
                        fully_scored=fully_scored,
                        acceptable=fully_scored and all(r['peak_ratio'] <= 1 for r in reports))
