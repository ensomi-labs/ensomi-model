import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

from pulsefield_model.evals.c3_raw_structure_audit import (
    _topk_summary,
    parse_raw_token_text,
    raw_bucket_keys,
    train_frequency_bucket,
)


class C3RawStructureAuditTests(unittest.TestCase):
    def test_parse_raw_token_without_tail(self) -> None:
        parsed = parse_raw_token_text("RAW|D:12:2:0:0")

        self.assertEqual(parsed.payload, "D:12:2:0:0")
        self.assertEqual(parsed.family, "D")
        self.assertEqual(parsed.fields, ("12", "2", "0", "0"))
        self.assertEqual(parsed.numeric_fields, ("12", "2", "0", "0"))
        self.assertFalse(parsed.has_tail)
        self.assertEqual(parsed.tail, "")

    def test_parse_raw_token_with_order_tail(self) -> None:
        parsed = parse_raw_token_text("RAW|D:0:0:0:15:OE3,E0,E1,E2")

        self.assertEqual(parsed.family, "D")
        self.assertEqual(parsed.numeric_fields, ("0", "0", "0", "15"))
        self.assertTrue(parsed.has_tail)
        self.assertEqual(parsed.tail, "OE3,E0,E1,E2")

    def test_parse_raw_token_rejects_non_raw(self) -> None:
        with self.assertRaises(ValueError):
            parse_raw_token_text("REF|12")

    def test_raw_bucket_keys_include_frequency_and_tail(self) -> None:
        buckets = dict(raw_bucket_keys("RAW|D:0:0:0:15:OE3,E0,E1,E2", train_count=37))

        self.assertEqual(buckets["family"], "D")
        self.assertEqual(buckets["numeric_field_count"], "4")
        self.assertEqual(buckets["field_1"], "0")
        self.assertEqual(buckets["field_4"], "15")
        self.assertEqual(buckets["has_tail"], "yes")
        self.assertEqual(buckets["tail_prefix"], "order")
        self.assertEqual(buckets["train_frequency"], "25to49")

    def test_train_frequency_bucket_boundaries(self) -> None:
        self.assertEqual(train_frequency_bucket(0), "0")
        self.assertEqual(train_frequency_bucket(1), "1")
        self.assertEqual(train_frequency_bucket(4), "2to4")
        self.assertEqual(train_frequency_bucket(9), "5to9")
        self.assertEqual(train_frequency_bucket(24), "10to24")
        self.assertEqual(train_frequency_bucket(49), "25to49")
        self.assertEqual(train_frequency_bucket(99), "50to99")
        self.assertEqual(train_frequency_bucket(249), "100to249")
        self.assertEqual(train_frequency_bucket(250), "250plus")

    def test_topk_summary_reports_full_and_raw_only_recall(self) -> None:
        summary = _topk_summary(
            rank_ks=(20,),
            target_count=10,
            model_hits_by_k={20: 3},
            unigram_hits_by_k={20: 5},
            model_raw_only_hits_by_k={20: 6},
            unigram_raw_only_hits_by_k={20: 4},
        )

        row = summary["20"]
        self.assertAlmostEqual(row["model_recall"], 0.3)
        self.assertAlmostEqual(row["unigram_recall"], 0.5)
        self.assertAlmostEqual(row["model_minus_unigram_recall"], -0.2)
        self.assertAlmostEqual(row["model_raw_only_recall"], 0.6)
        self.assertAlmostEqual(row["unigram_raw_only_recall"], 0.4)
        self.assertAlmostEqual(row["model_raw_only_minus_unigram_raw_only_recall"], 0.2)


if __name__ == "__main__":
    unittest.main()
