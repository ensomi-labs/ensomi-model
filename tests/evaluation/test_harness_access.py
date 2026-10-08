import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ensomi_model.evaluation.harness.access import CorpusAccess, HeldoutRefused, load_chart


def corpus(path: Path):
    rows = []
    for i, (split, status) in enumerate([('fit', 'ranked'), ('fit', 'loved'), ('fit', 'graveyard'),
                                         ('calibration', 'ranked'), ('heldout', 'ranked')]):
        rows.append(dict(path=f'x{i}.osu', sha256=f'{i:064x}', group_id=f'g{i}', star=3.0, eval_split=split,
                         canonical_bpm=120.0, api_status=status))
    pq.write_table(pa.Table.from_pylist(rows), path)


class AccessTests(unittest.TestCase):
    def test_refuses_heldout_and_logs_every_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus(tmp / 'c.parquet')
            access = CorpusAccess(tmp / 'c.parquet', tmp / 'out' / 'access_log.jsonl')
            fit = access.rows('fit', caller='test')
            self.assertEqual(sorted(r['api_status'] for r in fit), ['loved', 'ranked'])
            self.assertEqual(len(access.rows('calibration', caller='test')), 1)
            with self.assertRaises(HeldoutRefused):
                access.rows('heldout', caller='test')
            log = [json.loads(line) for line in (tmp / 'out' / 'access_log.jsonl').read_text().splitlines()]
            self.assertEqual([(e['split'], e['count'], e['refused']) for e in log],
                             [('fit', 2, False), ('calibration', 1, False), ('heldout', 0, True)])
            self.assertTrue(all(e['caller'] == 'test' for e in log))
        with self.assertRaises(HeldoutRefused):
            load_chart(dict(path='x.osu', eval_split='heldout'))


if __name__ == '__main__':
    unittest.main()
