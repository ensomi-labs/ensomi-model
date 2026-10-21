"""Tests for the R2 provenance table export, on synthetic count and label tables."""

import codecs
import copy
import csv
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'provenance' / 'r2_tables.py'
_spec = importlib.util.spec_from_file_location('r2_tables', SCRIPT)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

# Synthetic labels that a careless export would change: formula starts, surrounding whitespace, a
# comma with quotes, a slash, non-ASCII text, and two labels that differ only in case.
LABELS = ['Mapper One', '=SUM(A1)', '+plus', '-minus', '@at', ' padded ', 'comma, "quoted"', 'AC/DC',
          'Ünïcødé ラベル', 'mapper one']
CHARTS = 4                      # every synthetic stage has four charts


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_bytes(header, rows):
    """CSV as the extraction wrote it: csv.writer defaults, CRLF line endings."""
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def fixture():
    """key -> [header, rows] for the four inputs, over the 38 real stage ids."""
    stages = rt.CORE_STAGES + rt.SUPPLEMENTAL_STAGES
    labels = {'creator_labels': {}, 'artist_labels': {}}
    for i, stage in enumerate(stages):
        split = [[3, 1], [2, 1, 1]][i % 2]
        labels['creator_labels'][stage] = [(LABELS[(i + k) % len(LABELS)], n) for k, n in enumerate(split)]
        labels['artist_labels'][stage] = [(LABELS[(i + 5 + k) % len(LABELS)], 1) for k in range(CHARTS)]
    core = []
    for stage in rt.CORE_STAGES:
        roles = {'fit_train': 2, 'fit_dev': 1, 'not_in_cache': 1} if stage == 'frozen_corpus_inventory' else \
            {'fit_train': CHARTS}
        core.append([stage, 8, CHARTS, 0, 3, 0, 3, 3, 0, 1, len(labels['creator_labels'][stage]),
                     len(labels['artist_labels'][stage]), 0, 0, 0, json.dumps(roles), '{"fit": 4}', '"synthetic"',
                     'synthetic rule'])
    supplemental = []
    for stage in rt.SUPPLEMENTAL_STAGES:
        supplemental.append([stage, CHARTS, 3, 3, len(labels['creator_labels'][stage]),
                             len(labels['artist_labels'][stage]), 8, 0, 0, 3, 1, 4, 0, 0, 'synthetic#path'])
    tables = {'core_counts': [rt.CORE_HEADER, core], 'supplemental_counts': [rt.SUPPLEMENTAL_HEADER, supplemental]}
    for key, by_stage in labels.items():
        rows = []
        for stage in stages:
            population = 'core' if stage in rt.CORE_STAGES else 'supplemental'
            rows += [[stage, label, n, CHARTS, 1, 1, population] for label, n in by_stage[stage]]
        tables[key] = [rt.LABEL_HEADER, rows]
    return tables


def encode(tables):
    return {key: csv_bytes(header, rows) for key, (header, rows) in tables.items()}


def run(tables):
    inputs = encode(tables)
    return rt.export(inputs, {key: sha(data) for key, data in inputs.items()})


def parse(data):
    return list(csv.reader(io.StringIO(data.decode('utf-8'), newline='')))


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tables = fixture()

    def test_label_tables_keep_every_row_and_label_byte_for_byte(self):
        out = run(self.tables)
        for key, name in rt.LABEL_OUTPUTS.items():
            rows = self.tables[key][1]
            parsed = parse(out[name])
            self.assertEqual(parsed[0], list(rt.LABEL_OUT_HEADER))
            self.assertEqual(parsed[1:], [[str(v) for v in row] for row in rows])
            self.assertEqual([r[1].encode('utf-8') for r in parsed[1:]], [r[1].encode('utf-8') for r in rows])
            self.assertFalse(out[name].startswith(codecs.BOM_UTF8))
            self.assertNotIn(b'\r', out[name])

    def test_stage_counts_has_one_row_per_stage_from_both_tables(self):
        rows = list(csv.DictReader(io.StringIO(run(self.tables)['stage_counts.csv'].decode('utf-8'), newline='')))
        self.assertEqual([r['stage'] for r in rows], list(rt.CORE_STAGES + rt.SUPPLEMENTAL_STAGES))
        frozen, last = rows[0], rows[-1]
        self.assertEqual((frozen['population'], frozen['charts_fit_train'], frozen['charts_fit_dev'],
                          frozen['charts_outside_fit']), ('core', '2', '1', '1'))
        self.assertEqual((last['population'], last['charts_fit_train'], last['charts_fit_dev'],
                          last['charts_outside_fit'], last['positive_beatmap_set_ids']), ('supplemental', '3', '1', '0', ''))
        self.assertEqual(last['creator_labels'], str(len([r for r in self.tables['creator_labels'][1]
                                                          if r[0] == last['stage']])))
        self.assertEqual(last['source_records'], '8')

    def test_record_gives_each_output_its_rows_and_hash(self):
        out = run(self.tables)
        record = json.loads(out['export_record.json'])
        files = {f['name']: f for f in record['outputs']}
        self.assertEqual(files['stage_counts.csv'], {'name': 'stage_counts.csv', 'rows': 38,
                                                     'sha256': sha(out['stage_counts.csv'])})
        for key, name in rt.LABEL_OUTPUTS.items():
            self.assertEqual(files[name]['rows'], len(self.tables[key][1]))
            self.assertEqual(files[name]['sha256'], sha(out[name]))
        self.assertEqual([s['line_endings'] for s in record['sources']], ['CRLF'] * 4)

    def test_without_label_tables_the_record_marks_them_pending(self):
        inputs = encode({k: v for k, v in self.tables.items() if k.endswith('counts')})
        out = rt.export(inputs, {**{k: sha(v) for k, v in inputs.items()}, 'creator_labels': 'a' * 64,
                                 'artist_labels': 'b' * 64})
        self.assertEqual(sorted(out), ['export_record.json', 'stage_counts.csv'])
        files = {f['name']: f for f in json.loads(out['export_record.json'])['outputs']}
        self.assertEqual(files['creator_labels_by_stage.csv']['status'], 'pending')
        self.assertEqual(files['artist_labels_by_stage.csv']['expected_rows'], len(self.tables['artist_labels'][1]))

    def test_dropped_label_row_is_refused(self):
        tables = copy.deepcopy(self.tables)
        del tables['creator_labels'][1][5]
        with self.assertRaisesRegex(rt.ExportError, 'label rows, expected'):
            run(tables)

    def test_denominator_mismatch_is_refused(self):
        tables = copy.deepcopy(self.tables)
        tables['artist_labels'][1][7][3] = CHARTS + 1
        with self.assertRaisesRegex(rt.ExportError, 'stage_chart_denominator'):
            run(tables)

    def test_label_charts_that_miss_the_denominator_are_refused(self):
        tables = copy.deepcopy(self.tables)
        tables['creator_labels'][1][0][2] += 1
        with self.assertRaisesRegex(rt.ExportError, 'label charts sum to'):
            run(tables)

    def test_altered_label_in_output_is_detected(self):
        values = [[str(v) for v in row] for row in self.tables['creator_labels'][1]]
        data = rt.encode_csv(rt.LABEL_OUT_HEADER, values)
        rt.verify_output('labels', data, rt.LABEL_OUT_HEADER, values, 1)
        with self.assertRaisesRegex(rt.ExportError, 'parse back'):
            rt.verify_output('labels', data.replace(b' padded ', b'padded', 1), rt.LABEL_OUT_HEADER, values, 1)

    def test_altered_input_fails_its_hash(self):
        inputs = encode(self.tables)
        expected = {key: sha(data) for key, data in inputs.items()}
        inputs['artist_labels'] = inputs['artist_labels'].replace(b' padded ', b'padded', 1)
        with self.assertRaisesRegex(rt.ExportError, 'SHA-256'):
            rt.export(inputs, expected)

    def test_byte_order_mark_is_refused(self):
        inputs = encode(self.tables)
        inputs['core_counts'] = codecs.BOM_UTF8 + inputs['core_counts']
        with self.assertRaisesRegex(rt.ExportError, 'byte-order mark'):
            rt.export(inputs, {key: sha(data) for key, data in inputs.items()})

    def test_empty_and_repeated_labels_are_refused(self):
        for value, message in (('', 'empty raw_label'), (None, 'repeated label')):
            with self.subTest(message=message):
                tables = copy.deepcopy(self.tables)
                rows = tables['creator_labels'][1]
                rows[1][1] = rows[0][1] if value is None else value
                with self.assertRaisesRegex(rt.ExportError, message):
                    run(tables)

    def test_more_groups_than_charts_is_refused(self):
        tables = copy.deepcopy(self.tables)
        tables['creator_labels'][1][1][5] = 9
        with self.assertRaisesRegex(rt.ExportError, 'distinct_groups 9 exceeds'):
            run(tables)

    def test_path_like_text_is_refused_outside_the_label_column(self):
        rows = [['stage/one', 'AC/DC', '1', '1', '1', '1', 'core']]
        rt.verify_output('labels', rt.encode_csv(rt.LABEL_OUT_HEADER, [['s', 'AC/DC', '1', '1', '1', '1', 'core']]),
                         rt.LABEL_OUT_HEADER, [['s', 'AC/DC', '1', '1', '1', '1', 'core']], 1)
        with self.assertRaisesRegex(rt.ExportError, 'path-like'):
            rt.verify_output('labels', rt.encode_csv(rt.LABEL_OUT_HEADER, rows), rt.LABEL_OUT_HEADER, rows, 1)

    def test_label_report_counts_labels_a_spreadsheet_may_misread(self):
        report = rt.label_report([{'raw_label': label} for label in ['=a', '+b', ' c', 'd ', 'Ü', 'plain', '=a']])
        self.assertEqual(report['rows'], {'leading_formula_character': 3, 'leading_or_trailing_whitespace': 2,
                                          'non_ascii': 1})
        self.assertEqual(report['distinct_labels']['leading_formula_character'], 2)

    def test_failed_check_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = {}
            for key, data in encode(self.tables).items():
                paths[key] = tmp / f'{key}.csv'
                paths[key].write_bytes(data)
            out = tmp / 'out'
            status = rt.main(['--core-counts', str(paths['core_counts']), '--supplemental-counts',
                              str(paths['supplemental_counts']), '--out', str(out)])
            self.assertEqual(status, 1)          # the synthetic tables fail the recorded hashes
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
