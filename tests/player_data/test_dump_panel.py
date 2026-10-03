"""Tests for snapshot differencing and the panel measurements."""

import unittest

import numpy as np

from ._fixtures import load

pn = load('dump_panel')


def snapshot(rows):
    """[(user, beatmap, count)] -> (keys, counts)."""
    u, b, c = zip(*rows)
    return pn.pair_keys(u, b), np.array(c)


class DiffCountsTest(unittest.TestCase):
    def setUp(self):
        # user 1: beatmap 10 increases 5 -> 8, beatmap 11 stays at 2, beatmap 12 decreases 4 -> 3,
        #         beatmap 13 is new (0 -> 6), beatmap 14 vanishes (1 -> absent).
        # user 2: beatmap 10 increases 1 -> 2.
        # user 3 is in the earlier sample only (leaves): beatmap 10 at 9.
        self.prev = snapshot([(1, 10, 5), (1, 11, 2), (1, 12, 4), (1, 14, 1), (2, 10, 1), (3, 10, 9)])
        self.curr = snapshot([(1, 10, 8), (1, 11, 2), (1, 12, 3), (1, 13, 6), (2, 10, 2)])
        self.users_both = np.array([1, 2])

    def diff(self):
        pk, pc = self.prev
        ck, cc = self.curr
        mp_, mc = pn.restrict_to_users(pk, self.users_both), pn.restrict_to_users(ck, self.users_both)
        return pn.diff_counts(pk[mp_], pc[mp_], ck[mc], cc[mc])

    def status_of(self, d, user, beatmap):
        i = np.searchsorted(d['keys'], pn.pair_keys([user], [beatmap])[0])
        return pn.STATUS_NAMES[int(d['status'][i])], int(d['prev'][i]), int(d['curr'][i])

    def test_status_per_pair(self):
        d = self.diff()
        self.assertEqual(self.status_of(d, 1, 10), ('increased', 5, 8))
        self.assertEqual(self.status_of(d, 1, 11), ('same', 2, 2))
        self.assertEqual(self.status_of(d, 1, 12), ('decreased', 4, 3))
        self.assertEqual(self.status_of(d, 1, 13), ('new', 0, 6))
        self.assertEqual(self.status_of(d, 1, 14), ('vanished', 1, 0))
        self.assertEqual(self.status_of(d, 2, 10), ('increased', 1, 2))

    def test_user_who_leaves_is_not_differenced(self):
        d = self.diff()
        self.assertNotIn(3, set(pn.key_users(d['keys']).tolist()))
        # without the restriction, the leaver would look like a vanished pair
        pk, pc = self.prev
        ck, cc = self.curr
        raw = pn.diff_counts(pk, pc, ck, cc)
        self.assertEqual(self.status_of(raw, 3, 10), ('vanished', 9, 0))

    def test_summary_counts(self):
        s = pn.summarize_diff(self.diff(), interval_days=15)
        self.assertEqual(s['pairs_by_status'], {'same': 1, 'increased': 2, 'decreased': 1, 'new': 1, 'vanished': 1})
        self.assertEqual(s['pairs_with_attempts_added'], 3)
        self.assertEqual(s['attempts_added'], 3 + 1 + 6)
        self.assertEqual(s['attempts_added_on_existing_pairs'], 4)
        self.assertEqual(s['attempts_added_on_new_pairs'], 6)
        self.assertEqual(s['attempts_lost_on_decreased_pairs'], 1)
        self.assertEqual(s['attempts_lost_on_vanished_pairs'], 1)
        self.assertEqual(s['net_change'], 8)
        self.assertEqual(s['attempts_added_per_30_days'], 20.0)
        self.assertEqual((s['pairs_in_prev'], s['pairs_in_curr']), (5, 5))

    def test_summary_with_a_map_mask(self):
        d = self.diff()
        corpus = np.isin(pn.key_beatmaps(d['keys']), [10])
        s = pn.summarize_diff(d, corpus)
        self.assertEqual(s['pairs_with_attempts_added'], 2)
        self.assertEqual(s['attempts_added'], 4)
        self.assertEqual(s['attempts_lost_on_decreased_pairs'], 0)

    def test_duplicate_keys_are_rejected(self):
        k = pn.pair_keys([1, 1], [10, 10])
        with self.assertRaises(ValueError):
            pn.diff_counts(k, [1, 2], k[:1], [3])


class OverlapTest(unittest.TestCase):
    def test_overlap_and_patterns(self):
        o = pn.overlap(['a', 'b', 'c'], [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]])
        c0, c1 = o['consecutive']
        self.assertEqual((c0['both'], c0['left'], c0['entered'], c0['same_set']), (3, 1, 1, False))
        self.assertEqual((c1['both'], c1['same_set']), (4, True))
        self.assertEqual(o['in_all'], 3)
        self.assertEqual(o['in_any'], 5)
        self.assertEqual(o['presence_patterns'], {'111': 3, '100': 1, '011': 1})
        self.assertEqual(o['matrix_both']['a']['c'], 3)
        self.assertFalse(o['all_consecutive_same_set'])


class ScoresAndDatesTest(unittest.TestCase):
    def test_classify_attempted(self):
        attempted = pn.pair_keys([1, 1, 2], [10, 11, 10])
        new_rows = pn.pair_keys([1], [10])
        prior = pn.pair_keys([1, 2], [11, 99])
        c = pn.classify_attempted(attempted, new_rows, prior)
        self.assertEqual(c, {'attempted_pairs': 3, 'with_new_best_score_row': 1,
                             'no_new_row_but_best_score_before': 1, 'no_best_score_row_in_either_snapshot': 1})

    def test_date_agreement(self):
        dates = np.array(['2026-08-15T00:00:00', '2026-07-01T00:00:00', '2026-05-01T00:00:00',
                          '2026-09-02T00:00:00', 'NaT'], dtype='datetime64[s]')
        a = pn.date_agreement(dates, '2026-08-01T05:48:00', '2026-09-01T05:48:00')
        self.assertEqual((a['inside_interval'], a['before_interval'], a['before_interval_by_more_than_30_days'],
                          a['after_later_snapshot'], a['date_missing']), (1, 2, 2, 1, 1))
        self.assertEqual(a['share_inside'], 0.25)

    def test_month_histogram(self):
        dates = np.array(['2019-03-04T00:00:00', '2026-08-15T00:00:00', '2026-08-20T00:00:00'], dtype='datetime64[s]')
        self.assertEqual(pn.month_histogram(dates, '2026-04'), {'2019 (year)': 1, '2026-08': 2})

    def test_score_table_interval(self):
        corpus = np.array([10])
        ids_a, keys_a = np.array([1, 2]), pn.pair_keys([1, 1], [10, 11])
        ids_b, keys_b = np.array([2, 3, 4]), pn.pair_keys([1, 1, 2], [11, 10, 12])
        dates_a = np.array(['2026-07-01T00:00:00', '2026-07-02T00:00:00'], dtype='datetime64[s]')
        dates_b = np.array(['2026-07-02T00:00:00', '2026-08-10T00:00:00', '2026-06-01T00:00:00'], dtype='datetime64[s]')
        rep, new_keys = pn.score_table_interval(ids_a, keys_a, dates_a, ids_b, keys_b, dates_b,
                                                '2026-08-01T05:00:00', '2026-09-01T05:00:00', corpus, '2026-04',
                                                legacy_b=np.array([False, True, False]))
        self.assertEqual(rep['rows_new'], 2)
        self.assertEqual(rep['rows_removed'], 1)
        self.assertEqual(rep['rows_removed_with_a_new_row_on_the_same_pair'], 1)   # id 1 replaced by id 3
        self.assertEqual(rep['rows_new_on_corpus_maps'], 1)
        self.assertEqual(rep['rows_new_with_legacy_score_id'], 1)
        self.assertEqual(rep['new_rows_score_date_vs_interval']['inside_interval'], 1)
        self.assertEqual(rep['new_rows_score_date_vs_interval']['before_interval'], 1)
        self.assertEqual(sorted(new_keys.tolist()), sorted(pn.pair_keys([1, 2], [10, 12]).tolist()))

    def test_timestamps_survive_a_parquet_round_trip(self):
        # Parquet has no second unit: a timestamp[s] column reads back as timestamp[ms].
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('pyarrow not installed')
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 't.parquet')
            col = pa.array([1788241701, None], type=pa.int64()).cast(pa.timestamp('s', tz='UTC'))
            pq.write_table(pa.table({'t': col}), path)
            out = pn.ts_np(pq.read_table(path)['t'])
        self.assertEqual(str(out[0]), '2026-09-01T05:48:21')
        self.assertTrue(np.isnat(out[1]))

    def test_align_and_qsummary(self):
        out = pn.align([5, 3, 9], [50.0, 30.0, 90.0], np.array([3, 4, 9]))
        self.assertEqual(out[0], 30.0)
        self.assertTrue(np.isnan(out[1]))
        self.assertEqual(out[2], 90.0)
        q = pn.qsummary([-1, 0, 0, 2])
        self.assertEqual((q['n'], q['share_negative'], q['share_zero'], q['share_positive']), (4, 0.25, 0.5, 0.25))


if __name__ == '__main__':
    unittest.main()
