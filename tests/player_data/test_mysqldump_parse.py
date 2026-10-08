"""Tests for the mysqldump INSERT parser used by the dump collector."""

import io
import unittest

from ._fixtures import MEMBERS, load

mp = load('mysqldump_parse')


class ParseValuesTest(unittest.TestCase):
    def test_numeric_rows_take_the_fast_path(self):
        rows, fast = mp.parse_values('(1,101,5),(1,102,2),(2,103,7);\n')
        self.assertTrue(fast)
        self.assertEqual(rows, [['1', '101', '5'], ['1', '102', '2'], ['2', '103', '7']])

    def test_simple_quoted_strings_and_null_on_the_fast_path(self):
        rows, fast = mp.parse_values("(5001,101,'S','2026-08-15 12:00:00',123.5),(5002,103,'A','2019-03-04 05:06:07',NULL);")
        self.assertTrue(fast)
        self.assertEqual(rows[0], ['5001', '101', 'S', '2026-08-15 12:00:00', '123.5'])
        self.assertEqual(rows[1], ['5002', '103', 'A', '2019-03-04 05:06:07', None])

    def test_escapes_commas_and_parentheses_inside_strings(self):
        body = "(1,'b\\'eta, (two)',0),(2,'back\\\\slash\\nnew',-1.5e-05),(3,'',NULL);"
        rows, fast = mp.parse_values(body)
        self.assertFalse(fast)
        self.assertEqual(rows, [['1', "b'eta, (two)", '0'], ['2', 'back\\slash\nnew', '-1.5e-05'], ['3', '', None]])

    def test_quoted_null_is_a_string_and_bare_null_is_none(self):
        rows, _ = mp.parse_values("(1,'NULL',NULL),(2,'x',NULL);")
        self.assertEqual(rows, [['1', 'NULL', None], ['2', 'x', None]])

    def test_json_with_escaped_quotes_and_row_separator_text(self):
        body = "(7001,'{\\\"note\\\": \\\"a),(b\\\", \\\"k\\\": [1, 2]}',200.25),(7002,'{}',NULL);"
        rows, fast = mp.parse_values(body)
        self.assertFalse(fast)
        self.assertEqual(rows[0], ['7001', '{"note": "a),(b", "k": [1, 2]}', '200.25'])
        self.assertEqual(rows[1], ['7002', '{}', None])

    def test_fast_and_slow_paths_agree(self):
        body = "(1,'S','2026-01-01 00:00:00',NULL,3.25),(2,'XH','2025-12-31 23:59:59',7,-1);\n"
        fast, used = mp.parse_values(body)
        self.assertTrue(used)
        self.assertEqual(fast, mp._slow_rows(body))

    def test_malformed_body_raises(self):
        with self.assertRaises(mp.DumpFormatError):
            mp.parse_values("(1,'unterminated);")
        with self.assertRaises(mp.DumpFormatError):
            mp._slow_rows('(1,2)x(3,4);')

    def test_unescape(self):
        self.assertEqual(mp.unescape('a\\0b\\tc\\Zd\\\\e\\"f'), 'a\0b\tc\x1ad\\e"f')


class SchemaAndTrailerTest(unittest.TestCase):
    def test_parse_create_skips_key_lines(self):
        create = ('CREATE TABLE `t` (\n  `user_id` int unsigned NOT NULL,\n  `rank` mediumint NOT NULL,\n'
                  "  `country_acronym` char(2) NOT NULL DEFAULT '',\n  PRIMARY KEY (`user_id`),\n"
                  '  KEY `rank` (`rank`)\n) ENGINE=InnoDB;')
        self.assertEqual(mp.parse_create(create), [('user_id', 'int unsigned NOT NULL'), ('rank', 'mediumint NOT NULL'),
                                                   ('country_acronym', "char(2) NOT NULL DEFAULT ''")])

    def test_trailer_is_read_as_utc(self):
        self.assertEqual(mp.trailer_utc('-- Dump completed on 2026-09-01  5:48:21'), '2026-09-01T05:48:21Z')
        self.assertIsNone(mp.trailer_utc('-- something else'))


class MemberReaderTest(unittest.TestCase):
    def read(self, data):
        seen = {}
        rows = []

        def on_schema(cols, types):
            seen['cols'], seen['types'] = cols, types

        reader = mp.MemberReader('fixture').read(io.BytesIO(data), on_schema, rows.extend)
        return reader, seen, rows

    def test_reads_schema_rows_and_trailer(self):
        reader, seen, rows = self.read(MEMBERS['sample_users'])
        self.assertEqual(seen['cols'], ['user_id', 'username', 'user_warnings'])
        self.assertEqual(rows, [['1', 'alpha', '0'], ['2', "b'eta, (two)", '0'], ['3', 'NULL', '1']])
        s = reader.summary()
        self.assertEqual((s['table'], s['rows_in_member'], s['insert_lines']), ('sample_users', 3, 1))
        self.assertEqual(s['dump_completed_utc'], '2026-09-01T05:46:12Z')
        self.assertEqual(s['server_version'], '8.4.9')

    def test_counts_rows_across_insert_lines(self):
        reader, _, rows = self.read(MEMBERS['osu_user_beatmap_playcount'])
        self.assertEqual(reader.rows, 4)
        self.assertEqual(reader.insert_lines, 2)
        self.assertEqual(reader.fast_lines, 2)
        self.assertEqual(rows[-1], ['3', '101', '1'])

    def test_row_width_mismatch_raises(self):
        bad = MEMBERS['osu_user_beatmap_playcount'].replace(b'(3,101,1)', b'(3,101)')
        with self.assertRaises(mp.DumpFormatError):
            self.read(bad)

    def test_insert_before_create_raises(self):
        with self.assertRaises(mp.DumpFormatError):
            mp.MemberReader('x').read([b'INSERT INTO `t` VALUES (1);\n'])


class ArrowTypingTest(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest('pyarrow not installed')

    def test_types(self):
        import pyarrow as pa
        self.assertEqual(mp.arrow_type('mediumint unsigned NOT NULL'), pa.int64())
        self.assertEqual(mp.arrow_type('float unsigned NOT NULL'), pa.float64())
        self.assertEqual(mp.arrow_type('timestamp NOT NULL'), pa.timestamp('s', tz='UTC'))
        self.assertEqual(mp.arrow_type("enum('fail','exit') NOT NULL"), pa.string())
        self.assertEqual(mp.arrow_type('json NOT NULL'), pa.string())

    def test_columns_cast_and_zero_dates_become_null(self):
        import pyarrow as pa
        arr, bad = mp.to_arrow_column(['2026-09-01 05:48:21', '0000-00-00 00:00:00', None], pa.timestamp('s', tz='UTC'))
        self.assertEqual(bad, 1)
        self.assertEqual(arr.null_count, 2)
        self.assertEqual(str(arr[0].as_py()), '2026-09-01 05:48:21+00:00')
        ints, _ = mp.to_arrow_column(['1', '-2', None], pa.int64())
        self.assertEqual(ints.to_pylist(), [1, -2, None])
        floats, _ = mp.to_arrow_column(['1e-05', '3'], pa.float64())
        self.assertEqual(floats.to_pylist(), [1e-05, 3.0])


if __name__ == '__main__':
    unittest.main()
