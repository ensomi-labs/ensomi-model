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

    def observe(self, row, *, is_terminal=False):
        if row.time_ms <= self.time_ms:
            raise ValueError('Action response needs a row after committed coverage')
        replay = commit(self.replay, row, is_terminal=is_terminal)
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


def recovery_potential(values, limits):
    """Future excess area if no additional actions occur, coordinate by coordinate."""
    ratio = np.maximum(values/np.maximum(limits,1e-9)-1.,0.)
    tau = np.asarray(TAUS_MS)/1000
    tau = tau.reshape((len(TAUS_MS),)+(1,)*(np.ndim(values)-1))
    return np.maximum(0.,tau*(.5*ratio**2-ratio+np.log1p(ratio)))


def candidate_work(state, times, actions, limits):
    """Vectorized recovery work for hypothetical rows, without mutating history.

    Return [clock,candidate]. Actions need not be legal: the row owner retains
    its exact legality mask. The virtual all-empty release-clock action adds
    zero work and is never committed as a physical row.
    """
    times,actions = np.asarray(times,float),np.asarray(actions)
    if (times.ndim != 1 or actions.ndim != 2 or actions.shape[1] != 4
            or np.any(times <= state.time_ms)):
        raise ValueError('Candidate work requires future clocks and complete four-lane choices')
    heads,releases = np.isin(actions,(1,2)),actions == 3
    impulse = np.zeros((len(times),len(actions),28))
    def speed(previous):
        previous=np.asarray(previous,float)
        gap=times[:,None]-previous[None]
        return np.divide(100.,gap,out=np.zeros_like(gap),where=np.isfinite(gap)&(gap>0))
    h=np.asarray(state.replay.last_lane_attack_ms,float)
    r=np.asarray(state.replay.last_lane_release_ms,float)
    impulse[:,:,:4]=heads
    impulse[:,:,4:8]=heads[None]*speed(h)[:,None]
    prior_release=np.isfinite(r)&(~np.isfinite(h)|(r>h))
    impulse[:,:,8:12]=heads[None]*prior_release[None,None]*speed(r)[:,None]
    impulse[:,:,12:16]=releases[None]*speed(state.replay.open_ln_start_ms)[:,None]
    impulse[:,:,16:20]=releases
    hand_counts=(actions != 0).reshape(len(actions),2,2).sum(-1)
    impulse[:,:,20:22]=hand_counts>0
    impulse[:,:,22:24]=hand_counts[None]*speed(state.last_hand)[:,None]
    partners=np.arange(4)^1
    impulse[:,:,24:28]=heads[None]*np.asarray(state.replay.occupancy)[partners][None,None]*(actions[:,partners] != 3)[None]
    taus=np.asarray(TAUS_MS)[:,None,None,None]
    before=state.values[:,None,None]*np.exp(-(times[None,:,None,None]-state.time_ms)/taus)
    after=before+impulse[None]*1000/taus
    reference=np.asarray(limits)[:,None,None]
    work=np.maximum(0.,recovery_potential(after,reference)-recovery_potential(before,reference))
    return work.sum(-1).mean(0)


def source_work(times, actions, envelope, stars):
    """Per-row added recovery potential from a verified full source prefix.

    This uses the same pre/post-action states as the online continuation.
    Positive jumps include the consequence after a finite observation horizon,
    without inventing any subsequent release or adding future source actions.
    """
    times = np.asarray(times,float)
    impulses = transition_impulses(times,actions)
    tau = np.asarray(TAUS_MS)[:,None,None]
    limits = envelope.limits(stars)[:,None]
    values = np.zeros((len(TAUS_MS),28))
    work = np.zeros(len(times))
    previous,first = -1.,0
    while first < len(times):
        stop = int(np.searchsorted(times,times[first]+8000,side='right'))
        relative = times[first:stop]-times[first]
        entering = values*np.exp(-(times[first]-previous)/tau[:,0])
        exponent = np.exp(relative[None,:,None]/tau)
        increments = impulses[None,first:stop]*1000/tau
        after = (entering[:,None]+np.cumsum(exponent*increments,axis=1))/exponent
        before = np.maximum(0.,after-increments)
        work[first:stop] = np.maximum(0.,recovery_potential(after,limits)-
            recovery_potential(before,limits)).sum(-1).mean(0)
        values,previous,first = after[:,-1],times[stop-1],stop
    return work


def window_work_maxima(times, work, windows_ms):
    """Maximum added work on each real-time (t-width,t] window, with inherited state."""
    times,work = np.asarray(times,float),np.asarray(work,float)
    if times.shape != work.shape or np.any(work < 0):
        raise ValueError('Window work requires aligned nonnegative row charges')
    cumulative = np.r_[0.,np.cumsum(work)]
    return [float(np.max(cumulative[1:]-cumulative[
        np.searchsorted(times,times-width,side='right')],initial=0.)) for width in windows_ms]


@dataclass(frozen=True)
class ActionEnvelope:
    """Ranked-chart response references, not an identified physiological C0."""
    stars: tuple
    maximum: tuple
    reference: str
    work_windows_ms: tuple = ()
    maximum_work: tuple = ()

    def __post_init__(self):
        values = np.asarray(self.maximum)
        if (len(self.stars) < 2 or np.any(np.diff(self.stars) <= 0)
                or values.shape != (len(self.stars),len(TAUS_MS),len(KINDS))
                or not np.isfinite(values).all() or np.any(values < 0)):
            raise ValueError('Action reference requires ordered levels and finite response coordinates')
        if bool(self.work_windows_ms) != bool(self.maximum_work):
            raise ValueError('Action work windows and budgets must be supplied together')
        if self.work_windows_ms:
            budgets=np.asarray(self.maximum_work)
            if (budgets.shape != (len(self.stars),len(self.work_windows_ms))
                    or not np.isfinite(budgets).all() or np.any(budgets < 0)
                    or np.any(np.diff(self.work_windows_ms) <= 0) or self.work_windows_ms[0] <= 0):
                raise ValueError('Action work calibration requires ordered positive horizons and finite budgets')

    def limits(self, stars):
        table = np.asarray(self.maximum).reshape(len(self.stars),-1)
        values = np.array([np.interp(stars,self.stars,c) for c in table.T])
        kinds = values.reshape(len(TAUS_MS),len(KINDS))
        result = np.empty((len(TAUS_MS),28))
        for i,s in enumerate(SLICES):
            result[:,s] = kinds[:,i,None]
        return result

    def work_limit(self, stars, horizon_ms):
        """Use a calibrated enclosing horizon; beyond it use a covering-window bound.

        No linear extrapolation turns a tiny request scope into a zero impulse
        budget. This conservative duration lookup is explicit in each report.
        """
        if not self.work_windows_ms or horizon_ms <= 0:
            raise ValueError('Added-work selection requires calibrated real-time horizons')
        index = min(int(np.searchsorted(self.work_windows_ms,horizon_ms)),len(self.work_windows_ms)-1)
        factor = max(1,int(np.ceil(horizon_ms/self.work_windows_ms[index])))
        value = np.interp(stars,self.stars,np.asarray(self.maximum_work)[:,index])
        return float(value*factor),float(self.work_windows_ms[index])


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
    work_sums = np.zeros((len(ranges),len(TAUS_MS),28))
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
                before = current.advance(stop).values
                current = current.observe(next_row)
                index += 1
                if limits is not None and (stop < end or end == end_ms):
                    peak = np.maximum(peak,current.values/limits)
                # A row on a control boundary belongs to the newly active
                # half-open range; a final-horizon row uses its terminal range.
                owner = min(int(np.searchsorted([r[1] for r in ranges],stop,side='right')),len(ranges)-1)
                request = ranges[owner][2]
                if request is not None:
                    reference = envelope.limits(request)
                    work_sums[owner] += np.maximum(0.,recovery_potential(current.values,reference)-
                                                  recovery_potential(before,reference))
            else:
                current = current.advance(stop)
        by_kind = {k:float(costs[:,s].sum(-1).mean()) for k,s in zip(KINDS,SLICES)}
        reports.append(dict(start_ms=begin,end_ms=end,stars=level,
            excess_seconds=sum(by_kind.values()),excess_by_kind=by_kind,
            peak_ratio=None if limits is None else float(peak.max()),scored=level is not None))
    fully_scored = all(r['scored'] for r in reports)
    for i,report in enumerate(reports):
        by_kind = {k:float(work_sums[i,:,s].sum(-1).mean()) for k,s in zip(KINDS,SLICES)}
        report.update(added_work=sum(by_kind.values()),work_by_kind=by_kind)
        if envelope.work_windows_ms and report['scored']:
            limit,width=envelope.work_limit(report['stars'],report['end_ms']-max(0.,report['start_ms']))
            report.update(work_limit=limit,reference_horizon_ms=width,
                          work_accepted=report['added_work'] <= limit+1e-12)
    acceptable = (fully_scored and all(r['work_accepted'] for r in reports) if envelope.work_windows_ms
                  else fully_scored and all(r['peak_ratio'] <= 1 for r in reports))
    return current,dict(excess_seconds=sum(r['excess_seconds'] for r in reports),ranges=reports,
                        fully_scored=fully_scored,
                        added_work=sum(r['added_work'] for r in reports),
                        selection_cost=sum(r['added_work'] for r in reports),
                        acceptable=acceptable)
