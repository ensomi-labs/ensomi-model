"""The R2 demo's control semantics: transitions, scope statistics, class-preserving tilts, playability masks
and the band lookups that turn a request quantile into a target.

Ported unchanged in behaviour from the demo build of 2026-10-09.

- Jack J: a row shares a lane with the row before. S': disjoint motion, excluding avoidable exact returns.
- A scope owns its destination rows; earlier rows give transition context (``scope_statistics``).
- ``tilt`` moves probability towards a feature inside each class of actions while keeping every class's mass,
  so a concept request cannot change the row's head count or LN use.
- ``Playability`` holds the per-band masks (re-attack floor, quad runs, jack-run caps by gap tercile).
- ``lookup_band`` / ``lookup_subtype_band`` read targets from the 256 nearest human scopes of the same band and
  length bin (by log median gap; subtypes also by chord share).
- ``difficulty_multiplier`` maps a target r = sqrt(W / W_H) to the heads-per-row multiplier m, given the
  r of the base arrangement, with the per-band fit; m is clamped to [0.8, ``DIFFICULTY_M_EDGE``].
"""
from __future__ import annotations

import math

import numpy as np

from ..r2.common import ACTIONS
from ..r2.properties import row_range
from ..r2.strain import heads_from_codes

CHANCE = np.zeros((5, 5))
for _a in range(1, 5):
    for _b in range(1, 5):
        CHANCE[_a, _b] = 1 - (math.comb(4 - _a, _b) if _b <= 4 - _a else 0) / math.comb(4, _b)

SUBTYPES = {'jumpstream': dict(concept='stream', share='jump_share'),
            'chordstream': dict(concept='stream', share='hand_share'),
            'chordjack': dict(concept='jack', share='chord_share')}

# Upper edge of the difficulty multiplier, set by the demo's main thread: the largest m the demo panel's own runs
# support at no more than 1 fallback per 1,000 head rows (q .75 arm at the 1.3 grid edge: 2.2 per 1,000 at m near
# 1.3, about .2 at m <= 1.2). The map is unchanged inside it.
DIFFICULTY_M_EDGE = 1.2


def transition_features(heads, previous, before_previous=None):
    """(jack, S') of rows ``heads`` after ``previous``: S' is disjoint motion, excluding avoidable exact returns
    (after a three-head hand exempt)."""
    heads = np.asarray(heads, bool)
    previous = np.asarray(previous, bool)
    jack = (heads & previous).any(-1)
    back = np.zeros_like(jack)
    if before_previous is not None and previous.sum() != 3 and not np.any(previous & before_previous):
        back = (heads == before_previous).all(-1)
    return jack, ~jack & ~back


def scope_statistics(head_ms, heads, a_ms, b_ms, song_ms):
    """Destination-row ownership; earlier rows supply transition context."""
    heads = np.asarray(heads, bool)
    lo, hi = row_range(head_ms, a_ms, b_ms, song_ms)
    hi = min(hi, len(heads))
    js, ss, chance = [], [], []
    for k in range(max(1, lo), hi):
        j, s = transition_features(heads[k], heads[k - 1], heads[k - 2] if k >= 2 else None)
        js.append(j)
        ss.append(s)
        chance.append(CHANCE[heads[k - 1].sum(), heads[k].sum()])
    n = len(js)
    count = heads[lo:hi].sum(1)
    chord_pairs = [k for k in range(max(1, lo), hi) if heads[k].sum() >= 2 and heads[k - 1].sum() >= 2]
    identical = sum(np.array_equal(heads[k], heads[k - 1]) for k in chord_pairs)
    return dict(transitions=n, J=float(np.mean(js)) if n else None,
                J_chance=float(np.mean(chance)) if n else None,
                J_ex=float(np.mean(js) - np.mean(chance)) if n else None,
                S=float(np.mean(ss)) if n else None,
                heads_per_row=float(count.mean()) if len(count) else None,
                single_note_share=float((count == 1).mean()) if len(count) else None,
                jump_share=float((count == 2).mean()) if len(count) else None,
                hand_share=float((count >= 3).mean()) if len(count) else None,
                chord_share=float((count >= 2).mean()) if len(count) else None,
                arity_mix={str(i): float((count == i).mean()) for i in range(1, 5)} if len(count) else {},
                identical_consecutive_chord_share=identical / n if n else None,
                identical_consecutive_chord_count=int(identical), consecutive_chord_transitions=len(chord_pairs),
                identical_given_consecutive_chord_share=identical / len(chord_pairs) if chord_pairs else None)


def log_normalize(logp):
    finite = np.isfinite(logp)
    m = logp[finite].max()
    return logp - (m + np.log(np.exp(logp[finite] - m).sum()))


CLASSES = {}
FEATURES = {}
for _pattern in range(16):
    _held = ((_pattern >> np.arange(4)) & 1).astype(bool)
    _heads = heads_from_codes(ACTIONS, _held)
    _ln = np.where(_held, ACTIONS == 4, ACTIONS == 2)
    _features = np.column_stack((_ln.sum(1), _heads.sum(1), (_held | _heads).all(1)))
    FEATURES[_pattern] = (_heads, _ln, _features)
    for _concept, _values in [('jack', _features),
                              ('stream_subtype', np.column_stack((_ln.any(1), _features[:, 1:]))),
                              ('stream', np.column_stack((_ln.any(1), _features[:, 2]))),
                              ('chord', np.column_stack((_ln.any(1), _features[:, 2])))]:
        _, _groups = np.unique(_values, axis=0, return_inverse=True)
        CLASSES[_pattern, _concept] = [np.flatnonzero(_groups == i) for i in range(_groups.max() + 1)]


def tilt(logp, feature, groups, w):
    """Keep each supplied class mass; return logits and maximum marginal error."""
    if w == 0:
        return logp, 0.
    result = logp.copy()
    error = 0.
    for group in groups:
        ix = group[np.isfinite(logp[group])]
        if not len(ix):
            continue
        old = logp[ix]
        new = old + w * feature[ix]
        old_z = np.logaddexp.reduce(old)
        new_z = np.logaddexp.reduce(new)
        result[ix] = new + old_z - new_z
        error = max(error, abs(np.exp(result[ix]).sum() - np.exp(old).sum()))
    return result, float(error)


class Playability:
    """Same row rejection rules for corpus replay and generation.

    Config keys: reattack_floor_ms, quad_run_cap, gap_tercile_edges_ms, jack_run_caps. Tercile uses the current
    inter-head-row gap. Call observe once after each head row, including rows forbidden in a human replay.
    """

    def __init__(self, config):
        self.config = config
        self.runs = np.zeros(4, int)
        self.quad_run = 0

    def rejections(self, chart, k, held):
        heads = heads_from_codes(ACTIONS, held)
        elapsed = chart.head_ms[k] - chart.derived().last_attack[k]
        floor = (heads & (elapsed < self.config['reattack_floor_ms'])).any(1)
        quad = heads.all(1) & (self.quad_run >= self.config['quad_run_cap'])
        gap = chart.head_ms[k] - chart.head_ms[k - 1] if k else np.inf
        tercile = np.searchsorted(self.config['gap_tercile_edges_ms'], gap, side='right')
        cap = self.config['jack_run_caps'][tercile]
        jack = (heads & (self.runs >= cap)).any(1)
        return dict(reattack=floor, quad=quad, jack=jack)

    def observe(self, heads):
        self.runs = np.where(heads, self.runs + 1, 0)
        self.quad_run = self.quad_run + 1 if heads.all() else 0


def validate_requests(requests, song_ms):
    """Validate external scope requests; all overlaps are rejected, adjacency allowed."""
    for r in requests:
        if r['concept'] not in ('stream', 'jack'):
            raise ValueError('Concept must be stream or jack')
        subtype = r.get('subtype')
        if subtype is not None and (subtype not in SUBTYPES or r['concept'] != SUBTYPES[subtype]['concept']):
            raise ValueError('Subtype must be jumpstream/chordstream with stream or chordjack with jack')
        if r.get('chord_only', False) and subtype is None:
            raise ValueError('Chord-only requires a subtype preset')
        if 'chord_target' in r and (not np.isfinite(r['chord_target']) or not 0 <= r['chord_target'] <= 1):
            raise ValueError('Chord target must be in [0,1]')
        a, b = r['a_ms'], r['b_ms']
        if not np.isfinite([a, b]).all() or not 0 <= a < b <= song_ms or not 8000 <= b - a <= 32000:
            raise ValueError('Scopes must be 8–32 seconds within the song')
        if 'q' in r and (not np.isfinite(r['q']) or not 0 <= r['q'] <= 1):
            raise ValueError('q must be a quantile in [0,1]')
        if 'w' in r and (not np.isfinite(r['w']) or abs(r['w']) > 8):
            raise ValueError('Open-loop w must be finite and in [-8,8]')
        if 'target' in r and not np.isfinite(r['target']):
            raise ValueError('Resolved target must be finite')
    ordered = sorted(requests, key=lambda r: r['a_ms'])
    if any(a['b_ms'] > b['a_ms'] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Stream and jack scopes may be adjacent but must not overlap')


def scope_gap(head_ms, a_ms, b_ms, song_ms):
    lo, hi = row_range(head_ms, a_ms, b_ms, song_ms)
    gaps = np.diff(head_ms[lo:hi])
    return float(np.median(gaps)) if len(gaps) else None


def chord_feature(head_count, subtype):
    if subtype == 'jumpstream':
        return head_count == 2
    return head_count >= (3 if subtype == 'chordstream' else 2)


def _length_bin(length_ms):
    centres = np.array([8., 16., 32.])
    return centres[np.argmin(abs(np.log(centres) - np.log(length_ms / 1000)))]


def matched_scopes(scopes, band, length_ms, gap_ms, subtype=None, chord_target=None):
    length = _length_bin(length_ms)
    rows = scopes[(scopes.band == band) & (scopes.length_s == length)].copy()
    loggap = np.log(rows.median_gap_ms.to_numpy())
    if subtype is None:
        distance = abs(loggap - np.log(gap_ms))
        scale = None
    else:
        share = rows[SUBTYPES[subtype]['share']].to_numpy()
        values = np.column_stack((loggap, share))
        sd = values.std(0)
        sd = np.where(sd > 0, sd, 1.)
        distance = np.linalg.norm((values - np.array([np.log(gap_ms), chord_target])) / sd, axis=1)
        scale = dict(population_scopes=len(rows), mean=values.mean(0).tolist(), sd=sd.tolist(),
                     ddof=0, constant_coordinate_scale=1.)
    return rows.assign(distance=distance).sort_values('distance', kind='stable').head(256), float(length), scale


def lookup_subtype_band(scopes, subtype, band, length_ms, gap_ms, q, calibration):
    preset = calibration['subtypes'][subtype]['bands'][str(band)]
    original, length, _ = matched_scopes(scopes, band, length_ms, gap_ms)
    shares = original[SUBTYPES[subtype]['share']].to_numpy()
    chord_target = float(np.quantile(shares, preset['q_sub']))
    rows, _, scale = matched_scopes(scopes, band, length_ms, gap_ms, subtype, chord_target)
    metric = 'S' if SUBTYPES[subtype]['concept'] == 'stream' else 'J_ex'
    values = rows[metric].to_numpy()
    floor = calibration['masks'][str(band)]['reattack_floor_ms']
    reasons = []
    if subtype in ('jumpstream', 'chordstream') and gap_ms > calibration['stream_eligibility_gap_ms']:
        reasons.append('rows do not carry a stream: median gap above stream p90')
    if subtype == 'chordstream' and 2 * gap_ms < floor:
        reasons.append('two median gaps below re-attack floor')
    if subtype == 'chordjack' and gap_ms < max(floor, preset['reference_gap_p10_ms']):
        reasons.append('median gap below re-attack floor or chordjack reference p10')
    return dict(target=float(np.quantile(values, q)), band=np.quantile(values, [.05, .95]).tolist(),
                p90=float(np.quantile(values, .9)), neighbors=len(rows), length_bin_s=length,
                chord_target=chord_target, chord_band=np.quantile(shares, [.05, .95]).tolist(),
                chord_q=preset['q_sub'], chord_neighbors=len(original), preset_in_band=preset['in_band'],
                preset_out_of_band_reasons=preset['out_of_band_reasons'], concept_match_standardization=scale,
                temporal_eligible=not reasons, eligibility_reasons=reasons,
                reattack_floor_ms=floor, reference_gap_p10_ms=preset['reference_gap_p10_ms'])


def lookup_band(scopes, concept, band, length_ms, gap_ms, q):
    """Nearest 256 by log median gap in the same star and nearest length bin.

    scopes is a DataFrame with band,length_s,median_gap_ms,J_ex,S. Length bins use nearest 8/16/32-second centre
    in log duration, ties shorter.
    """
    length = _length_bin(length_ms)
    rows = scopes[(scopes.band == band) & (scopes.length_s == length)].copy()
    rows['distance'] = abs(np.log(rows.median_gap_ms) - np.log(gap_ms))
    values = rows.sort_values('distance', kind='stable').head(256)['S' if concept == 'stream' else 'J_ex'].to_numpy()
    return dict(target=float(np.quantile(values, q)), band=np.quantile(values, [.05, .95]).tolist(),
                p90=float(np.quantile(values, .9)), neighbors=len(values), length_bin_s=float(length))


def difficulty_multiplier(target_r, source_r, fit):
    return float(np.exp((np.log(target_r) - fit['alpha'] - fit['gamma'] * np.log(source_r)) / fit['beta']))


def logit(value):
    p = np.clip(value, .01, .99)
    return np.log(p / (1 - p))


def fallback_action(logp, heads, elapsed):
    """Empty union support: legal single head on the oldest lane, R2 tie-break.

    logp is plain R2 restricted by legality and the optional tap-only mask. Held-lane codes 3/4 remain candidates:
    gap release, then tap/LN head.
    """
    candidates = np.flatnonzero(np.isfinite(logp) & (heads.sum(1) == 1))
    lanes = heads[candidates].argmax(1)
    candidates = candidates[elapsed[lanes] == elapsed[lanes].max()]
    return int(candidates[np.argmax(logp[candidates])])
