"""End to end: the harness on a tiny synthetic corpus, through the worker processes and the report."""
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ensomi_model.evaluation.harness import run

from ._harness_charts import busy_chart, ln_chart, osu_file


def corpus(tmp: Path) -> Path:
    rows = []
    specs = [('fit', busy_chart(24 + 2 * i, 500.0 - 20 * (i % 3)), 3.0 + 0.04 * i) for i in range(8)]
    specs += [('fit', ln_chart(20 + 2 * i, 450.0 + 10 * i), 3.1 + 0.05 * i) for i in range(4)]
    specs += [('calibration', busy_chart(30, 480.0), 3.2), ('calibration', ln_chart(24, 470.0), 3.3),
              ('heldout', busy_chart(10), 3.0)]
    for i, (split, c, star) in enumerate(specs):
        path = osu_file(c, tmp / f'c{i}.osu', title=f'chart {i}')
        rows.append(dict(path=path, sha256=f'{i:064x}', group_id=f'g{i}', star=star, eval_split=split,
                         canonical_bpm=120.0, api_status='ranked' if i % 2 else 'loved'))
    pq.write_table(pa.Table.from_pylist(rows), tmp / 'corpus.parquet')
    return tmp / 'corpus.parquet'


class RunTests(unittest.TestCase):
    def test_harness_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / 'out'
            self.assertEqual(run.main(['--corpus', str(corpus(tmp)), '--out', str(out), '--workers', '2']), 0)
            report = json.loads((out / 'report.json').read_text())
            self.assertTrue((out / 'report.md').read_text().startswith('# Calibration harness v0'))
            c = report['counts']
            self.assertEqual((c['fit_charts'], c['calibration_charts']), (12, 2))
            self.assertEqual((c['fit_errors'], c['fit_null_errors'], c['calibration_errors']), (0, 0, 0))
            self.assertEqual(c['case_errors'], [])
            cov = report['coverage']['original whole']
            self.assertTrue(cov['events'])
            self.assertEqual((cov['events_scored'], cov['heads_ranked'], cov['not_ok']), (cov['events'], cov['heads'], 0))
            hashed = [m for m in report['must_not_flag'] if m['judged_by'] == 'full output hash' and m['condition'] == 'timing']
            self.assertTrue(all(m['compared'] and not m['mismatches'] and not m['errors'] for m in hashed), hashed)
            log = [json.loads(x) for x in (out / 'access_log.jsonl').read_text().splitlines()]
            self.assertEqual([e['split'] for e in log], ['fit', 'calibration'])
            model = json.loads((out / 'model.json').read_text())
            self.assertTrue(model['selected_before_calibration'])
            receipt = json.loads((out / 'receipt.json').read_text())
            self.assertIn('report.md', receipt['outputs_sha256'])
            self.assertEqual(run.main(['--corpus', str(tmp / 'corpus.parquet'), '--out', str(out), '--report-only']), 0)
            self.assertEqual(json.loads((out / 'report.json').read_text())['must_not_flag'], report['must_not_flag'])


if __name__ == '__main__':
    unittest.main()
