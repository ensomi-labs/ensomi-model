#!/usr/bin/env python
"""Panel measurements across the collected mania random_10000 dumps. Aggregates only.

Reads the Parquet tables that dump_collector.py writes under artifacts/player-data-dumps/<date>/
and writes artifacts/player-data-dumps/panel-report.json and panel-report.md. Neither output
holds a user id or username: only counts, shares and quantiles (p01..p99, never min or max).

Measured for every pair of consecutive dumps (A, B):

- sample overlap: users in both, users who left, users who entered, and whether the sets match;
- for users in both: pp (rank_score) change, playcount change, and per (user, beatmap) pair the
  difference of lifetime attempt counts (osu_user_beatmap_playcount): pairs whose count rose,
  stayed, fell, appeared or vanished, and the attempts added, on all maps and on corpus maps;
- best-score rows that are new in B (score id absent from A) in `scores` (lazer table) and
  `osu_scores_mania_high` (stable best scores), by month of the score date, and whether the
  score date falls inside the snapshot interval;
- attempted pairs (count rose) with and without a new best-score row.

Usage: .venv/bin/python scripts/player_data/dump_panel.py [--job-id ID]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import functools
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT_REL = os.path.join('artifacts', 'player-data-dumps')
CORPUS_REL = os.path.join('data', 'r2-corpus.parquet')

BEATMAP_SPAN = 1 << 24          # beatmap_id is a MySQL mediumint unsigned (< 2**24)
SAME, INCREASED, DECREASED, NEW, VANISHED = 0, 1, 2, 3, 4
STATUS_NAMES = {SAME: 'same', INCREASED: 'increased', DECREASED: 'decreased', NEW: 'new', VANISHED: 'vanished'}
PLAYCOUNT_CAP = 65535           # osu_user_beatmap_playcount.playcount is smallint unsigned
QS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


# ---- pure functions (tested) --------------------------------------------------------------------
def pair_keys(user_ids, beatmap_ids):
    """One int64 key per (user, beatmap)."""
    return np.asarray(user_ids, dtype=np.int64) * BEATMAP_SPAN + np.asarray(beatmap_ids, dtype=np.int64)


def key_users(keys):
    return np.asarray(keys, dtype=np.int64) // BEATMAP_SPAN


def key_beatmaps(keys):
    return np.asarray(keys, dtype=np.int64) % BEATMAP_SPAN


def qsummary(x, decimals=2):
    """Count, mean, quantiles p01..p99 and the shares above, at and below zero."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': round(float(x.mean()), decimals),
            'quantiles': {f'p{round(q * 100):02d}': round(float(np.quantile(x, q)), decimals) for q in QS},
            'share_positive': round(float((x > 0).mean()), 4), 'share_zero': round(float((x == 0).mean()), 4),
            'share_negative': round(float((x < 0).mean()), 4)}


def overlap(labels, user_sets):
    """Sample overlap across snapshots: sizes, consecutive overlap, the full matrix, users in all,
    and how many users show each presence pattern ('1' present, '0' absent, in label order)."""
    sets = [np.unique(np.asarray(u, dtype=np.int64)) for u in user_sets]
    out = {'n_users': {lab: int(s.size) for lab, s in zip(labels, sets)}, 'consecutive': []}
    for i in range(len(sets) - 1):
        a, b = sets[i], sets[i + 1]
        both = int(np.intersect1d(a, b, assume_unique=True).size)
        out['consecutive'].append({
            'from': labels[i], 'to': labels[i + 1], 'both': both, 'left': int(a.size - both),
            'entered': int(b.size - both), 'share_of_from_still_present': round(both / a.size, 4) if a.size else None,
            'jaccard': round(both / (a.size + b.size - both), 4) if (a.size + b.size) else None,
            'same_set': bool(a.size == b.size == both),
            'implied_population_if_independent_draws': int(round(a.size * b.size / both)) if both else None})
    out['matrix_both'] = {labels[i]: {labels[j]: int(np.intersect1d(sets[i], sets[j], assume_unique=True).size)
                                      for j in range(len(sets))} for i in range(len(sets))}
    if sets:
        in_all = functools.reduce(lambda x, y: np.intersect1d(x, y, assume_unique=True), sets)
        everyone = functools.reduce(lambda x, y: np.union1d(x, y), sets)
        out['in_all'] = int(in_all.size)
        out['in_any'] = int(everyone.size)
        bits = np.zeros(everyone.size, dtype=np.int64)
        for s in sets:
            bits = bits * 2 + np.isin(everyone, s)
        counts = collections.Counter(bits.tolist())
        width = len(sets)
        out['presence_patterns'] = {format(k, f'0{width}b'): int(v)
                                    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
        out['all_consecutive_same_set'] = all(c['same_set'] for c in out['consecutive'])
    return out


def diff_counts(prev_keys, prev_counts, curr_keys, curr_counts):
    """Difference of two snapshots of lifetime counts keyed by (user, beatmap).

    Keys must be unique within each snapshot. Returns keys (sorted union), prev and curr counts
    (0 where absent) and a status per key: same, increased, decreased, new (absent before),
    vanished (absent after).
    """
    prev_keys = np.asarray(prev_keys, dtype=np.int64)
    curr_keys = np.asarray(curr_keys, dtype=np.int64)
    for name, k in (('prev', prev_keys), ('curr', curr_keys)):
        if np.unique(k).size != k.size:
            raise ValueError(f'{name} keys are not unique')
    keys = np.union1d(prev_keys, curr_keys)
    prev = np.zeros(keys.size, dtype=np.int64)
    curr = np.zeros(keys.size, dtype=np.int64)
    in_prev = np.zeros(keys.size, dtype=bool)
    in_curr = np.zeros(keys.size, dtype=bool)
    pi = np.searchsorted(keys, prev_keys)
    ci = np.searchsorted(keys, curr_keys)
    prev[pi] = np.asarray(prev_counts, dtype=np.int64)
    curr[ci] = np.asarray(curr_counts, dtype=np.int64)
    in_prev[pi] = True
    in_curr[ci] = True
    status = np.select([~in_prev, ~in_curr, curr > prev, curr < prev], [NEW, VANISHED, INCREASED, DECREASED],
                       SAME).astype(np.int8)
    return {'keys': keys, 'prev': prev, 'curr': curr, 'status': status}


def restrict_to_users(keys, users):
    """Mask of keys whose user is in `users`."""
    return np.isin(key_users(keys), np.asarray(users, dtype=np.int64))


def summarize_diff(d, mask=None, interval_days=None):
    """Counts of pairs by status and attempts added or lost, for the keys selected by mask."""
    status, prev, curr = d['status'], d['prev'], d['curr']
    if mask is None:
        mask = np.ones(status.size, dtype=bool)
    delta = curr - prev
    pos = np.where(mask, np.clip(delta, 0, None), 0)
    neg = np.where(mask, np.clip(-delta, 0, None), 0)
    out = {
        'pairs_in_prev': int((mask & (status != NEW)).sum()),
        'pairs_in_curr': int((mask & (status != VANISHED)).sum()),
        'pairs_by_status': {name: int((mask & (status == code)).sum()) for code, name in STATUS_NAMES.items()},
        'pairs_with_attempts_added': int((mask & (delta > 0)).sum()),
        'attempts_added': int(pos.sum()),
        'attempts_added_on_existing_pairs': int(pos[status == INCREASED].sum()),
        'attempts_added_on_new_pairs': int(pos[status == NEW].sum()),
        'attempts_lost_on_decreased_pairs': int(neg[status == DECREASED].sum()),
        'attempts_lost_on_vanished_pairs': int(neg[status == VANISHED].sum()),
        'net_change': int(np.where(mask, delta, 0).sum()),
        'pairs_at_count_cap_65535': int((mask & (curr >= PLAYCOUNT_CAP)).sum()),
    }
    if interval_days:
        out['attempts_added_per_30_days'] = round(out['attempts_added'] * 30 / interval_days, 1)
        out['pairs_with_attempts_added_per_30_days'] = round(out['pairs_with_attempts_added'] * 30 / interval_days, 1)
    return out


def align(users, values, wanted):
    """values for each user in `wanted` (NaN where the user is absent)."""
    users = np.asarray(users, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    order = np.argsort(users)
    su, sv = users[order], values[order]
    idx = np.searchsorted(su, wanted)
    idx = np.clip(idx, 0, max(su.size - 1, 0))
    out = np.full(len(wanted), np.nan)
    if su.size:
        hit = su[idx] == wanted
        out[hit] = sv[idx[hit]]
    return out


def classify_attempted(attempt_keys, new_score_keys, prior_score_keys):
    """Attempted pairs split by whether a best-score row is new, existed before, or never existed."""
    has_new = np.isin(attempt_keys, new_score_keys)
    had_prior = np.isin(attempt_keys, prior_score_keys)
    return {'attempted_pairs': int(np.asarray(attempt_keys).size),
            'with_new_best_score_row': int(has_new.sum()),
            'no_new_row_but_best_score_before': int((~has_new & had_prior).sum()),
            'no_best_score_row_in_either_snapshot': int((~has_new & ~had_prior).sum())}


def date_agreement(dates, snap_prev, snap_curr):
    """Where score dates of rows first seen in the later snapshot fall relative to the interval."""
    d = np.asarray(dates, dtype='datetime64[s]')
    valid = ~np.isnat(d)
    sp, sc = np.datetime64(snap_prev, 's'), np.datetime64(snap_curr, 's')
    month = np.timedelta64(30 * 86400, 's')
    out = {'rows': int(d.size), 'date_missing': int((~valid).sum()),
           'inside_interval': int((valid & (d > sp) & (d <= sc)).sum()),
           'before_interval': int((valid & (d <= sp)).sum()),
           'before_interval_by_more_than_30_days': int((valid & (d <= sp - month)).sum()),
           'after_later_snapshot': int((valid & (d > sc)).sum())}
    n = int(valid.sum())
    out['share_inside'] = round(out['inside_interval'] / n, 4) if n else None
    return out


def month_histogram(dates, first_month):
    """{YYYY-MM: n} from first_month on; earlier dates grouped by year."""
    d = np.asarray(dates, dtype='datetime64[s]')
    d = d[~np.isnat(d)]
    if d.size == 0:
        return {}
    months = d.astype('datetime64[M]')
    cut = np.datetime64(first_month, 'M')
    out = {}
    early = months[months < cut]
    if early.size:
        years, counts = np.unique(early.astype('datetime64[Y]').astype(str), return_counts=True)
        out.update({f'{y} (year)': int(c) for y, c in zip(years, counts)})
    late = months[months >= cut]
    if late.size:
        ms, counts = np.unique(late.astype(str), return_counts=True)
        out.update({m: int(c) for m, c in zip(ms, counts)})
    return out


# ---- loading ------------------------------------------------------------------------------------
def complete_dumps(out_dir):
    dumps = []
    for d in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        p = os.path.join(out_dir, d, 'manifest.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            man = json.load(f)
        if man.get('status') == 'complete' and man['dump']['kind'] == 'random_10000':
            dumps.append((d, man))
    dumps.sort(key=lambda x: x[1]['dump']['dump_date'])
    return dumps


def read_cols(dump_dir, table, cols):
    import pyarrow.parquet as pq
    return pq.read_table(os.path.join(dump_dir, f'{table}.parquet'), columns=cols)


def to_np(col, dtype=None):
    arr = col.to_numpy(zero_copy_only=False)
    return arr.astype(dtype) if dtype is not None else arr


def ts_np(col):
    """Timestamp column -> numpy datetime64[s] (NaT for null).

    Parquet has no second unit, so columns written as timestamp[s] read back as timestamp[ms];
    the stored unit is honoured here rather than assumed.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    per_second = {'s': 1, 'ms': 1000, 'us': 1_000_000, 'ns': 1_000_000_000}[col.type.unit]
    ints = pc.cast(col, pa.int64())
    a = ints.to_numpy(zero_copy_only=False)
    out = np.full(a.shape, np.datetime64('NaT'), dtype='datetime64[s]')
    valid = pc.is_valid(ints).to_numpy(zero_copy_only=False)
    out[valid] = (a[valid].astype('int64') // per_second).astype('datetime64[s]')
    return out


def snapshot_time(man, table):
    t = ((man.get('tables') or {}).get(table) or {}).get('dump_completed_utc')
    return (t or man['dump']['dump_date'] + 'T00:00:00Z').rstrip('Z')


class Small:
    """Per-dump user-level data, kept for all dumps."""

    def __init__(self, dump_dir, man):
        self.dir, self.man = dump_dir, man
        self.date = man['dump']['dump_date']
        self.sample = np.unique(to_np(read_cols(dump_dir, 'sample_users', ['user_id'])['user_id'], np.int64))
        st = read_cols(dump_dir, 'osu_user_stats_mania', ['user_id', 'rank_score', 'playcount', 'last_played'])
        self.stats_users = to_np(st['user_id'], np.int64)
        self.pp = to_np(st['rank_score'], float)
        self.playcount = to_np(st['playcount'], float)
        self.last_played = ts_np(st['last_played'])
        self.snap_stats = np.datetime64(snapshot_time(man, 'osu_user_stats_mania'), 's')


def load_big(dump_dir, man, users):
    """Pair-level and score-level arrays for the given users."""
    import pyarrow as pa
    import pyarrow.compute as pc
    uset = pa.array(users, type=pa.int64())

    def sel(t):
        return t.filter(pc.is_in(t['user_id'], value_set=uset))

    pc_t = sel(read_cols(dump_dir, 'osu_user_beatmap_playcount', ['user_id', 'beatmap_id', 'playcount']))
    sc_t = sel(read_cols(dump_dir, 'scores', ['id', 'user_id', 'beatmap_id', 'ended_at', 'legacy_score_id',
                                              'in_corpus']))
    hi_t = sel(read_cols(dump_dir, 'osu_scores_mania_high', ['score_id', 'user_id', 'beatmap_id', 'date']))
    bm = to_np(read_cols(dump_dir, 'osu_beatmaps', ['beatmap_id'])['beatmap_id'], np.int64)
    legacy = to_np(pc.fill_null(sc_t['legacy_score_id'], 0), np.int64)
    return {
        'pc_keys': pair_keys(to_np(pc_t['user_id']), to_np(pc_t['beatmap_id'])),
        'pc_counts': to_np(pc_t['playcount'], np.int64),
        'sc_ids': to_np(sc_t['id'], np.int64),
        'sc_keys': pair_keys(to_np(sc_t['user_id']), to_np(sc_t['beatmap_id'])),
        'sc_dates': ts_np(sc_t['ended_at']),
        'sc_legacy': legacy > 0,
        'hi_ids': to_np(hi_t['score_id'], np.int64),
        'hi_keys': pair_keys(to_np(hi_t['user_id']), to_np(hi_t['beatmap_id'])),
        'hi_dates': ts_np(hi_t['date']),
        'mania_beatmaps': np.unique(bm),
        'snap_pc': snapshot_time(man, 'osu_user_beatmap_playcount'),
        'snap_sc': snapshot_time(man, 'scores'),
        'snap_hi': snapshot_time(man, 'osu_scores_mania_high'),
    }


def load_corpus_ids():
    import pyarrow.parquet as pq
    path = os.path.join(REPO, CORPUS_REL)
    t = pq.read_table(path, columns=['api_beatmap_id', 'beatmap_id']).to_pydict()
    ids = {int(a) if a else int(b) for a, b in zip(t['api_beatmap_id'], t['beatmap_id']) if a or (b and b > 0)}
    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return np.array(sorted(ids), dtype=np.int64), {'path': CORPUS_REL, 'sha256': digest, 'distinct_ids': len(ids)}


# ---- per-interval measurement -------------------------------------------------------------------
def score_table_interval(ids_a, keys_a, dates_a, ids_b, keys_b, dates_b, snap_a, snap_b, corpus, first_month,
                         legacy_b=None):
    new = ~np.isin(ids_b, ids_a)
    removed = ~np.isin(ids_a, ids_b)
    new_keys = np.unique(keys_b[new])
    removed_keys = keys_a[removed]
    out = {
        'rows_prev': int(ids_a.size), 'rows_curr': int(ids_b.size), 'rows_new': int(new.sum()),
        'rows_new_on_corpus_maps': int(np.isin(key_beatmaps(keys_b[new]), corpus).sum()),
        'rows_removed': int(removed.sum()),
        'rows_removed_with_a_new_row_on_the_same_pair': int(np.isin(removed_keys, new_keys).sum()),
        'pairs_with_new_rows': int(new_keys.size),
        'new_rows_score_date_vs_interval': date_agreement(dates_b[new], snap_a, snap_b),
        'prev_rows_dated_after_prev_snapshot': int((~np.isnat(dates_a) & (dates_a > np.datetime64(snap_a, 's'))).sum()),
        'new_rows_by_score_month': month_histogram(dates_b[new], first_month),
    }
    if legacy_b is not None:
        out['rows_new_with_legacy_score_id'] = int((new & legacy_b).sum())
        out['rows_new_lazer_native'] = int((new & ~legacy_b).sum())
    return out, new_keys


def days_between(a, b):
    return (np.datetime64(b, 's') - np.datetime64(a, 's')).astype('timedelta64[s]').astype(float) / 86400


def interval_report(sa, sb, ba, bb, corpus, first_month):
    both = np.intersect1d(sa.sample, sb.sample, assume_unique=True)
    left = np.setdiff1d(sa.sample, sb.sample, assume_unique=True)
    entered = np.setdiff1d(sb.sample, sa.sample, assume_unique=True)
    nominal_days = days_between(sa.date + 'T00:00:00', sb.date + 'T00:00:00')
    pc_days = days_between(ba['snap_pc'], bb['snap_pc'])
    rep = {'from': sa.date, 'to': sb.date, 'days_nominal': round(nominal_days, 2),
           'days_between_playcount_snapshots': round(pc_days, 3),
           'snapshots_utc': {'prev': {'playcount': ba['snap_pc'], 'scores': ba['snap_sc'], 'high': ba['snap_hi']},
                             'curr': {'playcount': bb['snap_pc'], 'scores': bb['snap_sc'], 'high': bb['snap_hi']}},
           'users_both': int(both.size), 'users_left': int(left.size), 'users_entered': int(entered.size)}

    # user level
    pp_a, pp_b = align(sa.stats_users, sa.pp, both), align(sb.stats_users, sb.pp, both)
    pc_a, pc_b = align(sa.stats_users, sa.playcount, both), align(sb.stats_users, sb.playcount, both)
    dpp, dpc = pp_b - pp_a, pc_b - pc_a
    rep['pp_change'] = qsummary(dpp)
    rep['pp_change_per_30_days'] = qsummary(dpp * 30 / pc_days) if pc_days else None
    rep['playcount_change'] = qsummary(dpc, 1)
    rep['share_users_with_playcount_increase'] = round(float((dpc > 0).mean()), 4) if both.size else None
    inactive = dpc == 0
    rep['pp_change_users_with_no_new_plays'] = qsummary(dpp[inactive])
    rep['users_missing_from_stats'] = {'prev': int(np.isnan(pp_a).sum()), 'curr': int(np.isnan(pp_b).sum())}

    def who(small, users):
        pp = align(small.stats_users, small.pp, users)
        lp_all = small.last_played.astype('datetime64[s]')
        lp_sec = np.where(np.isnat(lp_all), np.nan, lp_all.astype('int64').astype(float))
        lp = align(small.stats_users, lp_sec, users)
        days_idle = (float(small.snap_stats.astype('int64')) - lp) / 86400
        return {'n': int(users.size), 'pp': qsummary(pp)['quantiles'] if users.size else None,
                'days_since_last_played': qsummary(days_idle, 1)['quantiles'] if users.size else None}
    rep['who_left_vs_stayed_at_prev'] = {'left': who(sa, left), 'stayed': who(sa, both)}
    rep['who_entered_vs_stayed_at_curr'] = {'entered': who(sb, entered), 'stayed': who(sb, both)}

    # pair level
    ma = restrict_to_users(ba['pc_keys'], both)
    mb = restrict_to_users(bb['pc_keys'], both)
    d = diff_counts(ba['pc_keys'][ma], ba['pc_counts'][ma], bb['pc_keys'][mb], bb['pc_counts'][mb])
    corpus_mask = np.isin(key_beatmaps(d['keys']), corpus)
    listed_mask = np.isin(key_beatmaps(d['keys']), bb['mania_beatmaps'])
    rep['attempts'] = {'all_maps': summarize_diff(d, None, pc_days),
                       'corpus_maps': summarize_diff(d, corpus_mask, pc_days),
                       'maps_in_curr_dump_mania_osu_beatmaps': summarize_diff(d, listed_mask, pc_days)}
    delta = d['curr'] - d['prev']
    users_of = key_users(d['keys'])
    uidx = np.searchsorted(both, users_of)
    added_per_user = np.bincount(uidx, weights=np.clip(delta, 0, None), minlength=both.size)
    rep['attempts_added_per_user'] = qsummary(added_per_user, 1)
    rep['pair_sum_vs_user_playcount_change'] = {
        'sum_user_playcount_change': int(np.nansum(dpc)),
        'sum_pair_attempts_added': int(added_per_user.sum()),
        'ratio_pairs_over_user': round(float(added_per_user.sum() / np.nansum(dpc)), 4) if np.nansum(dpc) else None,
        'share_users_equal': round(float((added_per_user == dpc).mean()), 4) if both.size else None,
        'user_playcount_change_minus_pair_sum': qsummary(dpc - added_per_user, 1)}

    # score level
    ka, kb = restrict_to_users(ba['sc_keys'], both), restrict_to_users(bb['sc_keys'], both)
    sc_rep, sc_new = score_table_interval(ba['sc_ids'][ka], ba['sc_keys'][ka], ba['sc_dates'][ka],
                                          bb['sc_ids'][kb], bb['sc_keys'][kb], bb['sc_dates'][kb],
                                          ba['snap_sc'], bb['snap_sc'], corpus, first_month, bb['sc_legacy'][kb])
    ha, hb = restrict_to_users(ba['hi_keys'], both), restrict_to_users(bb['hi_keys'], both)
    hi_rep, hi_new = score_table_interval(ba['hi_ids'][ha], ba['hi_keys'][ha], ba['hi_dates'][ha],
                                          bb['hi_ids'][hb], bb['hi_keys'][hb], bb['hi_dates'][hb],
                                          ba['snap_hi'], bb['snap_hi'], corpus, first_month)
    rep['scores_lazer_table'] = sc_rep
    rep['scores_stable_high_table'] = hi_rep
    new_keys = np.union1d(sc_new, hi_new)
    prior_keys = np.union1d(ba['sc_keys'][ka], ba['hi_keys'][ha])
    attempted = d['keys'][delta > 0]
    attempted_corpus = d['keys'][(delta > 0) & corpus_mask]
    rep['attempted_pairs'] = {'all_maps': classify_attempted(attempted, new_keys, prior_keys),
                              'corpus_maps': classify_attempted(attempted_corpus, new_keys, prior_keys)}
    new_without_attempt = new_keys[~np.isin(new_keys, attempted)]
    rep['pairs_with_new_best_but_no_attempt_increase'] = {
        'pairs': int(new_without_attempt.size),
        'pair_absent_from_curr_playcount': int((~np.isin(new_without_attempt, d['keys'][d['status'] != VANISHED])).sum())}
    return rep


def trajectories(smalls):
    users = functools.reduce(lambda x, y: np.intersect1d(x, y, assume_unique=True), [s.sample for s in smalls])
    pp = np.stack([align(s.stats_users, s.pp, users) for s in smalls], axis=1)
    pc = np.stack([align(s.stats_users, s.playcount, users) for s in smalls], axis=1)
    steps = np.diff(pp, axis=1)
    return {'users_in_all_dumps': int(users.size), 'dumps': [s.date for s in smalls],
            'median_pp_by_dump': [round(float(np.nanmedian(pp[:, i])), 2) for i in range(len(smalls))]
            if users.size else None,
            'pp_change_first_to_last': qsummary(pp[:, -1] - pp[:, 0]),
            'playcount_change_first_to_last': qsummary(pc[:, -1] - pc[:, 0], 1),
            'share_pp_never_decreasing': round(float((steps >= 0).all(axis=1).mean()), 4) if users.size else None,
            'share_pp_decreasing_at_least_once': round(float((steps < 0).any(axis=1).mean()), 4) if users.size else None,
            'share_pp_unchanged_throughout': round(float((steps == 0).all(axis=1).mean()), 4) if users.size else None,
            'share_with_playcount_increase_in_every_interval':
                round(float((np.diff(pc, axis=1) > 0).all(axis=1).mean()), 4) if users.size else None}


# ---- report -------------------------------------------------------------------------------------
def build(out_dir, job_id=None):
    dumps = complete_dumps(out_dir)
    if len(dumps) < 2:
        raise SystemExit(f'need at least two complete random_10000 dumps, found {len(dumps)}')
    corpus, corpus_info = load_corpus_ids()
    smalls = [Small(os.path.join(out_dir, d), man) for d, man in dumps]
    first_month = smalls[0].date[:7]
    labels = [s.date for s in smalls]
    report = {
        'schema': 'ensomi.player-dump-panel/1',
        'generated_at': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'job_id': job_id or None,
        'script_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
        'dumps': [{'dump_date': man['dump']['dump_date'], 'dir': d, 'tarball_sha256': man['tarball']['sha256'],
                   'collected_by_job': (man.get('provenance') or {}).get('job_id')} for d, man in dumps],
        'corpus': corpus_info,
        'privacy': 'Aggregates only: counts, shares and quantiles p01..p99. No user ids, usernames, minima or maxima.',
        'overlap': overlap(labels, [s.sample for s in smalls]),
        'stats_table_matches_sample': {s.date: bool(np.array_equal(np.unique(s.stats_users), s.sample)) for s in smalls},
        'intervals': [],
    }
    for i in range(len(smalls) - 1):
        sa, sb = smalls[i], smalls[i + 1]
        both = np.intersect1d(sa.sample, sb.sample, assume_unique=True)
        ba = load_big(sa.dir, sa.man, both)
        bb = load_big(sb.dir, sb.man, both)
        report['intervals'].append(interval_report(sa, sb, ba, bb, corpus, first_month))
        del ba, bb
    report['trajectories_users_in_all_dumps'] = trajectories(smalls)
    report['new_best_rows_by_score_month_all_intervals'] = {
        table: dict(sorted(sum((collections.Counter(iv[table]['new_rows_by_score_month']) for iv in report['intervals']),
                               collections.Counter()).items()))
        for table in ('scores_lazer_table', 'scores_stable_high_table')}
    with open(os.path.join(out_dir, 'panel-report.json'), 'w') as f:
        json.dump(report, f, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
        f.write('\n')
    with open(os.path.join(out_dir, 'panel-report.md'), 'w') as f:
        f.write(render_md(report))
    return report


def fmt(x, nd=0):
    if x is None:
        return 'n/a'
    if isinstance(x, float) and nd:
        return f'{x:,.{nd}f}'
    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x == int(x) and not nd):
        return f'{int(x):,}'
    return f'{x:,.2f}'


def pct(x):
    return 'n/a' if x is None else f'{100 * x:.1f}%'


def render_md(r):
    ov, ivs = r['overlap'], r['intervals']
    L = []
    w = L.append
    w('# Panel report: mania random_10000 dumps\n')
    w(f"Generated {r['generated_at']} by job `{r['job_id']}` from {len(r['dumps'])} dumps: "
      + ', '.join(d['dump_date'] for d in r['dumps']) + '. Aggregates only; per-player tables stay in Parquet '
      'on the mac. Every number below comes from these dumps (tarball SHA-256 values in `panel-report.json`).\n')

    w('## Sample overlap\n')
    w('| From | To | Days | Users in From | Users in To | In both | Left | Entered | Share of From still present | Same set |')
    w('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    for c, iv in zip(ov['consecutive'], ivs):
        w(f"| {c['from']} | {c['to']} | {iv['days_nominal']:g} | {fmt(ov['n_users'][c['from']])} | "
          f"{fmt(ov['n_users'][c['to']])} | {fmt(c['both'])} | {fmt(c['left'])} | {fmt(c['entered'])} | "
          f"{pct(c['share_of_from_still_present'])} | {'yes' if c['same_set'] else 'no'} |")
    w('')
    w(f"- Users in every dump: {fmt(ov['in_all'])} of {fmt(ov['in_any'])} distinct users seen in any dump.")
    labs = list(ov['matrix_both'])
    far = [ov['matrix_both'][a][b] for i, a in enumerate(labs) for b in labs[i + 2:]]
    if far:
        w(f"- Overlap between non-consecutive dumps: {fmt(min(far))} to {fmt(max(far))} users.")
    impl = [ov['n_users'][c['from']] * ov['n_users'][c['to']] / c['both'] for c in ov['consecutive'] if c['both']]
    if impl:
        w(f"- If each sample were an independent draw, the overlaps would imply a population of "
          f"{fmt(round(min(impl), -3))} to {fmt(round(max(impl), -3))} eligible users [inference].")
    top = list(ov['presence_patterns'].items())[:8]
    w('- Most common presence patterns (one digit per dump, oldest first; 1 = in the sample): '
      + ', '.join(f'`{k}` {fmt(v)}' for k, v in top) + '.')
    w(f"- `osu_user_stats_mania` covers exactly `sample_users` in: "
      + ', '.join(f"{k} {'yes' if v else 'no'}" for k, v in r['stats_table_matches_sample'].items()) + '.\n')

    w('## Users present in consecutive dumps\n')
    w('| Interval | Users | pp change p10 / p50 / p90 | pp up / same / down | Playcount change p50 / p90 | '
      'Users with new plays | pp change p50 of users with no new plays (n) |')
    w('| --- | ---: | --- | --- | --- | ---: | --- |')
    for iv in ivs:
        q, pcq, qi = iv['pp_change']['quantiles'], iv['playcount_change']['quantiles'], iv['pp_change_users_with_no_new_plays']
        w(f"| {iv['from']} to {iv['to']} | {fmt(iv['users_both'])} | {q['p10']} / {q['p50']} / {q['p90']} | "
          f"{pct(iv['pp_change']['share_positive'])} / {pct(iv['pp_change']['share_zero'])} / "
          f"{pct(iv['pp_change']['share_negative'])} | {pcq['p50']:g} / {pcq['p90']:g} | "
          f"{pct(iv['share_users_with_playcount_increase'])} | "
          f"{qi.get('quantiles', {}).get('p50', 'n/a')} ({fmt(qi['n'])}) |")
    w('')

    w('## Attempts added (differences of lifetime attempt counts per user and beatmap)\n')
    w('| Interval | Pairs with attempts added | of which new pairs | Attempts added | per 30 days | '
      'Corpus: pairs | Corpus: attempts added | Corpus: per 30 days | Pairs decreased | Pairs vanished | '
      'Attempts lost |')
    w('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for iv in ivs:
        a, c = iv['attempts']['all_maps'], iv['attempts']['corpus_maps']
        w(f"| {iv['from']} to {iv['to']} | {fmt(a['pairs_with_attempts_added'])} | {fmt(a['pairs_by_status']['new'])} | "
          f"{fmt(a['attempts_added'])} | {fmt(a.get('attempts_added_per_30_days'))} | "
          f"{fmt(c['pairs_with_attempts_added'])} | {fmt(c['attempts_added'])} | {fmt(c.get('attempts_added_per_30_days'))} | "
          f"{fmt(a['pairs_by_status']['decreased'])} | {fmt(a['pairs_by_status']['vanished'])} | "
          f"{fmt(a['attempts_lost_on_decreased_pairs'] + a['attempts_lost_on_vanished_pairs'])} |")
    w('')
    w('Consistency of the pair differences with the per-user playcount in `osu_user_stats_mania`:\n')
    w('| Interval | Sum of user playcount change | Sum of pair attempts added | Ratio | Users where they are equal |')
    w('| --- | ---: | ---: | ---: | ---: |')
    for iv in ivs:
        p = iv['pair_sum_vs_user_playcount_change']
        w(f"| {iv['from']} to {iv['to']} | {fmt(p['sum_user_playcount_change'])} | {fmt(p['sum_pair_attempts_added'])} | "
          f"{p['ratio_pairs_over_user']} | {pct(p['share_users_equal'])} |")
    w('')

    w('## Attempted pairs and best scores\n')
    w('A pair is attempted in an interval when its attempt count rose. "New best-score row" means a score id in '
      'the later dump that the earlier dump did not have, in `scores` or `osu_scores_mania_high`.\n')
    w('| Interval | Maps | Attempted pairs | With a new best-score row | No new row, had a best before | '
      'No best-score row in either dump |')
    w('| --- | --- | ---: | ---: | ---: | ---: |')
    for iv in ivs:
        for k, name in (('all_maps', 'all'), ('corpus_maps', 'corpus')):
            a = iv['attempted_pairs'][k]
            n = a['attempted_pairs'] or 1
            w(f"| {iv['from']} to {iv['to']} | {name} | {fmt(a['attempted_pairs'])} | "
              f"{fmt(a['with_new_best_score_row'])} ({pct(a['with_new_best_score_row'] / n)}) | "
              f"{fmt(a['no_new_row_but_best_score_before'])} ({pct(a['no_new_row_but_best_score_before'] / n)}) | "
              f"{fmt(a['no_best_score_row_in_either_snapshot'])} ({pct(a['no_best_score_row_in_either_snapshot'] / n)}) |")
    w('')
    w('| Interval | Pairs with a new best row but no attempt increase | of which absent from the later attempt table |')
    w('| --- | ---: | ---: |')
    for iv in ivs:
        p = iv['pairs_with_new_best_but_no_attempt_increase']
        w(f"| {iv['from']} to {iv['to']} | {fmt(p['pairs'])} | {fmt(p['pair_absent_from_curr_playcount'])} |")
    w('')

    w('## Score dates against the snapshot interval\n')
    w('For rows first seen in the later dump: where the score date (`scores.ended_at`, '
      '`osu_scores_mania_high.date`) falls relative to the two table snapshots (the mysqldump completion times).\n')
    w('| Interval | Table | New rows | Inside interval | Before it | Before by > 30 days | After later snapshot | '
      'Earlier dump rows dated after its own snapshot |')
    w('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for iv in ivs:
        for k, name in (('scores_lazer_table', 'scores'), ('scores_stable_high_table', 'osu_scores_mania_high')):
            s = iv[k]
            a = s['new_rows_score_date_vs_interval']
            w(f"| {iv['from']} to {iv['to']} | {name} | {fmt(s['rows_new'])} | {fmt(a['inside_interval'])} "
              f"({pct(a['share_inside'])}) | {fmt(a['before_interval'])} | {fmt(a['before_interval_by_more_than_30_days'])} | "
              f"{fmt(a['after_later_snapshot'])} | {fmt(s['prev_rows_dated_after_prev_snapshot'])} |")
    w('')
    w('| Interval | Table | Rows removed | of which replaced on the same pair | New rows on corpus maps | '
      'New rows with a legacy score id | New lazer-native rows |')
    w('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
    for iv in ivs:
        for k, name in (('scores_lazer_table', 'scores'), ('scores_stable_high_table', 'osu_scores_mania_high')):
            s = iv[k]
            w(f"| {iv['from']} to {iv['to']} | {name} | {fmt(s['rows_removed'])} | "
              f"{fmt(s['rows_removed_with_a_new_row_on_the_same_pair'])} | {fmt(s['rows_new_on_corpus_maps'])} | "
              f"{fmt(s.get('rows_new_with_legacy_score_id'))} | {fmt(s.get('rows_new_lazer_native'))} |")
    w('')

    w('## New best-score rows by month of the score date\n')
    hist = r['new_best_rows_by_score_month_all_intervals']
    months = sorted(set(hist['scores_lazer_table']) | set(hist['scores_stable_high_table']))
    w('Summed over all intervals; a row is counted in the interval where it first appears.\n')
    w('| Month | scores | osu_scores_mania_high |')
    w('| --- | ---: | ---: |')
    for m in months:
        w(f"| {m} | {fmt(hist['scores_lazer_table'].get(m, 0))} | {fmt(hist['scores_stable_high_table'].get(m, 0))} |")
    w('')

    t = r['trajectories_users_in_all_dumps']
    w('## Users present in every dump\n')
    if not t['users_in_all_dumps']:
        w('No user is present in every dump, so there are no trajectories across the whole span.\n')
        w(supports_section(r))
        return '\n'.join(L) + '\n'
    q = t['pp_change_first_to_last'].get('quantiles', {})
    w(f"- {fmt(t['users_in_all_dumps'])} users. Median pp by dump: "
      + ', '.join(f'{d} {v}' for d, v in zip(t['dumps'], t['median_pp_by_dump'] or [])) + '.')
    w(f"- pp change from the first to the last dump: p10 {q.get('p10')}, p50 {q.get('p50')}, p90 {q.get('p90')}.")
    w(f"- pp never decreasing: {pct(t['share_pp_never_decreasing'])}; decreasing at least once: "
      f"{pct(t['share_pp_decreasing_at_least_once'])}; unchanged throughout: {pct(t['share_pp_unchanged_throughout'])}.")
    w(f"- Playcount rising in every interval: {pct(t['share_with_playcount_increase_in_every_interval'])}.\n")

    w(supports_section(r))
    return '\n'.join(L) + '\n'


def supports_section(r):
    ov, ivs = r['overlap'], r['intervals']
    shares = [c['share_of_from_still_present'] or 0 for c in ov['consecutive']]
    lo = min(shares) if shares else 0
    is_panel = lo >= 0.5
    if ov.get('all_consecutive_same_set'):
        verdict = ('The consecutive samples are the same set of users, so the dumps form a balanced panel over '
                   'the measured span.')
    elif lo >= 0.9:
        verdict = (f'The consecutive samples overlap heavily (at least {pct(lo)} of each sample is still present '
                   'in the next), so the dumps form a panel with some turnover.')
    elif is_panel:
        verdict = (f'The consecutive samples overlap only partly (as low as {pct(lo)}), so the dumps form an '
                   'unbalanced panel: a core of repeat users plus a changing remainder.')
    else:
        verdict = (f'The consecutive samples overlap little (at most {pct(max(shares))} of a sample is still '
                   'present in the next), so the dumps are a series of fresh cross-sections, not a panel of the '
                   'same players.')
    days = ', '.join(f"{iv['from']} to {iv['to']} {iv['days_nominal']:g} d" for iv in ivs)
    dec = sum(iv['attempts']['all_maps']['pairs_by_status']['decreased'] for iv in ivs)
    van = sum(iv['attempts']['all_maps']['pairs_by_status']['vanished'] for iv in ivs)
    inact = [iv['pp_change_users_with_no_new_plays'] for iv in ivs]
    inact_txt = '; '.join(f"{iv['from']} to {iv['to']}: {pct(x.get('share_zero'))} unchanged of {fmt(x['n'])}"
                          for iv, x in zip(ivs, inact))
    both = [iv['users_both'] for iv in ivs]
    inside = [iv[k]['new_rows_score_date_vs_interval']['share_inside'] for iv in ivs
              for k in ('scores_lazer_table', 'scores_stable_high_table')]
    inside = [x for x in inside if x is not None]
    inside_txt = f'{pct(min(inside))} to {pct(max(inside))}' if inside else 'n/a'
    lines = ['## What this panel can and cannot support for a time-varying skill model\n',
             f'**Verdict on the panel.** {verdict} Measured, not assumed: see the overlap table.\n',
             '**What it supports.**\n']
    if is_panel:
        lines += [
            '- Monthly skill snapshots per user: `rank_score` (pp) at each dump date, for every user present in '
            'two or more dumps, plus playcount, play time and the counts of fails and quits.',
            '- Dated exposure: for users present in consecutive dumps, the attempts added on each beatmap within '
            'each interval (difference of lifetime counts). This dates attempts to the interval, not to the day.',
            '- Dated outcomes: each new best-score row carries its own timestamp (`ended_at` or `date`), so the '
            'improvements a user made are dated to the second, on known maps, with accuracy, judgement counts '
            'and mods.',
            '- The pairing of the two: attempts on a map in an interval, with or without a new best on that map in '
            'the same interval. Attempts without a new best are the closest thing to "tried and did not pass or '
            'did not improve" that these dumps give.\n']
    else:
        entered = [c['entered'] for c in ov['consecutive']]
        lines += [
            f"- Breadth, not follow-up: {fmt(ov['in_any'])} distinct users across {len(r['dumps'])} dumps; each "
            f'dump brings {fmt(min(entered))} to {fmt(max(entered))} users absent from the previous one. Collecting '
            'every month grows the number of players observed, not the length of their records.',
            '- Within one dump: skill at the dump date (pp, accuracy, playcount, play time, fail and quit counts), '
            'lifetime attempts per beatmap, and the surviving best scores, each dated to the second and going back '
            'years. A time-varying skill model can use the dates of a player\'s surviving bests as the time axis '
            '(with the survivorship limit below); it cannot use repeated pp snapshots of the same player.',
            '- Population drift: comparing cross-sections (pp distribution, attempts per map, new bests per month) '
            'measures change in the population between dump dates, not in individuals.',
            f'- Repeat users: {fmt(sum(both))} user-intervals over {len(ivs)} intervals ({fmt(min(both))} to '
            f'{fmt(max(both))} per interval) have real differences: attempts added per beatmap, pp change, new '
            f'bests. On them lifetime counts never fell and {inside_txt} of new best rows are dated inside the '
            'interval, so the differencing mechanics work; the numbers are too small for a population model of '
            'skill change [inference about adequacy].\n']
    lines += [
        '**What it cannot support, and where differencing breaks down.**\n',
        '- Users leaving or entering the sample: differences exist only for users present at both ends. A user who '
        'enters brings lifetime counts that cannot be dated; a user who leaves stops being observed. The who-left '
        'and who-entered rows of `panel-report.json` show whether turnover is selective (pp and days since last '
        'played) [inference from those aggregates only].',
        f'- Counts that fall or vanish: across all intervals {fmt(dec)} pairs decreased and {fmt(van)} vanished. A '
        'lifetime count should only grow, so any such pair would mark a deletion, merge or reset on the server side '
        '[inference]. Differencing treats them as zero attempts added and reports the lost attempts separately.',
        f'- Irregular intervals ({days}). June 2026 has no random dump, so 2026-05-01 to 2026-07-13 is one long '
        'interval and 2026-07-13 to 2026-08-01 a short one. Per-30-day rates are given, but attempts inside an '
        'interval stay undated; the long interval blurs anything faster than ten weeks.',
        '- Outcomes of single attempts: only best scores are stored. A pass that does not beat the existing best, '
        'and every failed attempt, leaves no row. Attempts are lifetime counts over all mods (and, as far as the '
        'dumps show, all clients); the mod and outcome of each attempt are unknown.',
        f'- pp is the pp system of the dump date: a recalculation or rework changes everyone\'s pp without play. '
        f'Users with no new plays in an interval show it: {inact_txt}. Any change there is not skill [inference].',
        '- History before the first dump: best scores carry dates going back years, but superseded bests are gone, '
        'so earlier skill can be read only from surviving bests (survivorship).',
        '- Snapshot boundaries: each table is dumped at a slightly different time on the dump date '
        '(`snapshots_utc`); scores set in the minutes between table dumps can land on either side.',
    ]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description='panel report over collected dumps')
    ap.add_argument('--job-id', default='')
    args = ap.parse_args(argv)
    out_dir = os.path.join(REPO, OUT_REL)
    rep = build(out_dir, args.job_id or None)
    print(json.dumps({'dumps': [d['dump_date'] for d in rep['dumps']],
                      'overlap_consecutive': rep['overlap']['consecutive'], 'in_all': rep['overlap']['in_all']},
                     indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
