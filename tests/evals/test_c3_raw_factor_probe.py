import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.evals.c3_raw_factor_probe import (
    _FieldAccumulator,
    build_raw_factor_vocab,
    field_value_scores,
    parse_raw_factor_fields,
    top_field_values,
)


class C3RawFactorProbeTests(unittest.TestCase):
    def test_parse_raw_factor_fields_uses_four_numeric_fields(self) -> None:
        fields = parse_raw_factor_fields("RAW|D:12:8:0:4")

        self.assertEqual(
            fields,
            {
                "field_1": "12",
                "field_2": "8",
                "field_3": "0",
                "field_4": "4",
            },
        )

    def test_parse_raw_factor_fields_rejects_short_raw_token(self) -> None:
        with self.assertRaises(ValueError):
            parse_raw_factor_fields("RAW|D:12:8")

    def test_build_raw_factor_vocab_indexes_raw_tokens_only(self) -> None:
        vocab = build_raw_factor_vocab(
            {
                1: "RAW|D:12:8:0:0",
                2: "RAW|D:6:4:0:0",
                3: "REF|x",
            }
        )

        self.assertEqual(set(vocab.token_fields), {1, 2})
        self.assertEqual(vocab.value_token_indexes["field_1"]["6"], (1,))
        self.assertEqual(vocab.value_token_indexes["field_1"]["12"], (0,))
        self.assertEqual(vocab.value_order["field_1"], ("6", "12"))

    def test_field_value_scores_support_max_and_logsumexp(self) -> None:
        vocab = build_raw_factor_vocab(
            {
                1: "RAW|D:12:8:0:0",
                2: "RAW|D:12:4:0:0",
                3: "RAW|D:6:4:0:0",
            }
        )
        logits = torch.tensor([1.0, 3.0, 2.0])

        max_scores = field_value_scores(logits, factor="field_1", vocab=vocab, aggregation="max")
        logsumexp_scores = field_value_scores(logits, factor="field_1", vocab=vocab, aggregation="logsumexp")

        self.assertEqual(max_scores["12"], 3.0)
        self.assertEqual(max_scores["6"], 2.0)
        self.assertGreater(logsumexp_scores["12"], max_scores["12"])

    def test_top_field_values_returns_sorted_highest_values(self) -> None:
        vocab = build_raw_factor_vocab(
            {
                1: "RAW|D:12:8:0:0",
                2: "RAW|D:6:4:0:0",
                3: "RAW|D:24:2:0:0",
            }
        )
        logits = torch.tensor([1.0, 5.0, 3.0])

        top_values = top_field_values(logits, vocab=vocab, aggregation="max", limit=2)

        self.assertEqual(top_values["field_1"], ("6", "24"))
        self.assertEqual(top_values["field_2"], ("4", "2"))

    def test_field_accumulator_scores_multilabel_targets(self) -> None:
        accumulator = _FieldAccumulator(factors=("field_1",), top_ks=(1, 2))

        accumulator.update(
            target_values={"field_1": frozenset({"6", "12"})},
            predicted_values={"field_1": ("6", "24")},
        )

        result = accumulator.to_dict()["field_1"]
        self.assertEqual(result["target_count"], 2)
        self.assertEqual(result["topk"]["1"]["hits"], 1)
        self.assertEqual(result["topk"]["2"]["hits"], 1)
        self.assertAlmostEqual(result["topk"]["1"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
