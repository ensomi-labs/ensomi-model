"""A mandatory-head lower bound for the whole-chart 20241007 star algorithm."""
import math
from numbers import Real

from ...osu_core.difficulty import ManiaStrain, RawHitObject, create_difficulty_hit_objects


class _MandatoryHeadStrain(ManiaStrain):
    def strain_value_of(self, current):
        # Keep the minimum contribution of each mandatory head. The actual
        # selected individual strain is >=2 even when it is not the busiest lane.
        self.overall_strain = self._apply_decay(
            self.overall_strain, current.delta_time, self.overall_decay_base) + 1.
        self.highest_individual_strain = 2.
        return 2. + self.overall_strain - self.current_strain


def head_timing_star_lower_bound_20241007(head_times_ms, *, clock_rate=1.):
    """Return a conservative whole-chart star floor for mandatory H timestamps.

    H is one timestamp per head-bearing row, independent of chord size, lane,
    TAP/LN type and eventual LN tails. Times must be finite, nonnegative,
    strictly increasing integer milliseconds; invalid times or a nonpositive,
    nonfinite clock rate raise ValueError. Integral real values are accepted.

    The same floor also bounds a completion that appends later heads to this
    prefix. Open LN tails need not be guessed. Fewer than two H return zero,
    respecting the algorithm's omitted first raw hit object; a chord at that
    first H can still have positive actual difficulty.

    This is neither an attainable minimum nor an upper bound, a scoped
    difficulty, or a playability score. It cannot be compared with a different
    control range. The proof depends on the named algorithm's strain minima,
    decay, section accounting and positive sorted-peak weights.
    """
    if (isinstance(clock_rate, bool) or not isinstance(clock_rate, Real)
            or not math.isfinite(clock_rate) or clock_rate <= 0):
        raise ValueError('Clock rate must be finite and positive')
    times = tuple(head_times_ms)
    if any(isinstance(t, bool) or not isinstance(t, Real)
           or not math.isfinite(t) or t < 0 or int(t) != t for t in times):
        raise ValueError('H timestamps must be finite nonnegative integer milliseconds')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('H timestamps must be strictly increasing, with chords grouped')
    raw = [RawHitObject(float(t), float(t), 0) for t in times]
    skill = _MandatoryHeadStrain(4)
    # Reuse the exact first-object and 400-ms section rules of the named scorer.
    for obj in create_difficulty_hit_objects(raw, 4, float(clock_rate)):
        skill.process(obj)
    return .018 * skill.difficulty_value()
