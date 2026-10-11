"""End-to-end extraction of a small dump-shaped tarball, plus listing and licence helpers."""

import os
import tempfile
import unittest

from . import _fixtures as fx

dc = fx.load('dump_collector')


class ListingTest(unittest.TestCase):
    def test_parse_listing_and_entries(self):
        html = ("<a href='2026_07_13_performance_mania_random_10000.tar.bz2'>x</a>"
                '<a href="2026_07_13_performance_mania_top_10000.tar.bz2">y</a>'
                "<a href='2026_07_13_performance_osu_random_10000.tar.bz2'>z</a><a href='LICENCE.txt'>l</a>")
        names = dc.parse_listing(html)
        self.assertEqual(len(names), 3)
        e = dc.DumpEntry('2026_07_13_performance_mania_random_10000.tar.bz2')
        self.assertEqual((e.dump_date, e.kind, e.dir_name), ('2026-07-13', 'random_10000', '2026-07-13'))
        t = dc.DumpEntry('2026_07_13_performance_mania_top_10000.tar.bz2')
        self.assertEqual(t.dir_name, '2026-07-13_top_10000')
        self.assertIsNone(dc.NAME_RE.match('2026_07_13_performance_osu_random_10000.tar.bz2'))

    def test_licence_comparison(self):
        same = dc.licence_record(dc.REFERENCE_LICENCE_TEXT.encode(), 'test')
        self.assertTrue(same['same_bytes_as_reference'])
        self.assertTrue(same['same_text_as_reference'])
        crlf = dc.licence_record(dc.REFERENCE_LICENCE_TEXT.replace('\n', '\r\n').encode(), 'test')
        self.assertFalse(crlf['same_bytes_as_reference'])
        self.assertTrue(crlf['same_text_as_reference'])
        changed = dc.licence_record(b'All data may be used for anything.\n', 'test')
        self.assertFalse(changed['same_text_as_reference'])
        self.assertIn('anything', changed['text'])


class ExtractTest(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest('pyarrow not installed')
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_finds_members_and_licence(self):
        tar = fx.write_tarball(os.path.join(self.dir, 'd.tar.bz2'),
                               extra={'LICENCE.txt': dc.REFERENCE_LICENCE_TEXT.encode()})
        members, licences = dc.scan_members(tar)
        self.assertEqual(len(members), len(fx.MEMBERS) + 1)
        self.assertEqual(len(licences), 1)
        self.assertTrue(licences[0]['same_text_as_reference'])

    def test_extracts_the_seven_tables_with_filters(self):
        import pyarrow.parquet as pq
        tar = fx.write_tarball(os.path.join(self.dir, 'd.tar.bz2'))
        stage = os.path.join(self.dir, 'stage')
        os.makedirs(stage)
        stats = dc.extract_tables(tar, stage, fx.CORPUS_IDS, batch_rows=2)
        self.assertEqual(sorted(k for k in stats if not k.startswith('_')), sorted(dc.TABLES))
        self.assertFalse(os.path.exists(os.path.join(stage, '_osu_beatmap_failtimes.all_modes.parquet')))

        su = pq.read_table(os.path.join(stage, 'sample_users.parquet')).to_pydict()
        self.assertEqual(su['username'], ['alpha', "b'eta, (two)", 'NULL'])

        st = pq.read_table(os.path.join(stage, 'osu_user_stats_mania.parquet')).to_pydict()
        self.assertEqual(st['rank_score'], [1234.56, 1e-05, 0.0])
        self.assertIsNone(st['last_played'][1])
        self.assertEqual(stats['osu_user_stats_mania']['timestamps_unparsed_to_null'], {'last_played': 1})

        sc = pq.read_table(os.path.join(stage, 'scores.parquet')).to_pydict()
        self.assertEqual(sc['id'], [7001, 7002])                  # ruleset 0 row dropped
        self.assertEqual(sc['in_corpus'], [True, False])
        self.assertEqual(sc['data'][0], '{"mods": [{"acronym": "DT"}], "note": "a),(b"}')
        self.assertEqual(stats['scores']['rows_seen'], 3)
        self.assertEqual(stats['scores']['rows_kept'], 2)
        self.assertEqual(stats['scores']['rows_in_corpus'], 1)

        bm = pq.read_table(os.path.join(stage, 'osu_beatmaps.parquet')).to_pydict()
        self.assertEqual(bm['beatmap_id'], [101, 103])            # mania only
        self.assertEqual(bm['version'], ['4K Hard (x, y)', "Kyle's 4K"])

        ft = pq.read_table(os.path.join(stage, 'osu_beatmap_failtimes.parquet')).to_pydict()
        self.assertEqual(list(zip(ft['beatmap_id'], ft['type'])), [(101, 'fail'), (101, 'exit'), (103, 'exit')])
        self.assertEqual(stats['osu_beatmap_failtimes']['rows_seen'], 5)

        pcount = pq.read_table(os.path.join(stage, 'osu_user_beatmap_playcount.parquet'))
        self.assertEqual(pcount.num_rows, 4)
        hi = pq.read_table(os.path.join(stage, 'osu_scores_mania_high.parquet')).to_pydict()
        self.assertEqual(hi['pp'], [123.5, None])
        self.assertEqual(stats['osu_user_beatmap_playcount']['dump_completed_utc'], '2026-09-01T05:48:37Z')
        self.assertEqual(stats['scores']['rows_with_user_not_in_sample_users'], 0)
        self.assertFalse(stats['scores']['schema_vs_reference']['same_as_reference'])

    def test_missing_table_blocks(self):
        members = dict(fx.MEMBERS)
        del members['osu_user_beatmap_playcount']
        tar = fx.write_tarball(os.path.join(self.dir, 'd.tar.bz2'), members)
        stage = os.path.join(self.dir, 'stage')
        os.makedirs(stage)
        with self.assertRaises(dc.Blocked):
            dc.extract_tables(tar, stage, fx.CORPUS_IDS)

    def test_missing_required_column_blocks(self):
        members = dict(fx.MEMBERS)
        members['osu_user_beatmap_playcount'] = fx.member(
            'osu_user_beatmap_playcount', ['`user_id` int unsigned NOT NULL', '`beatmap_id` mediumint unsigned NOT NULL'],
            ['(1,101)'])
        tar = fx.write_tarball(os.path.join(self.dir, 'd.tar.bz2'), members)
        stage = os.path.join(self.dir, 'stage')
        os.makedirs(stage)
        with self.assertRaises(dc.Blocked):
            dc.extract_tables(tar, stage, fx.CORPUS_IDS)


if __name__ == '__main__':
    unittest.main()
