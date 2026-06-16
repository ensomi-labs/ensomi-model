import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.evals.c3_auxiliary_diagnostics import (
    C3TopKAccumulator,
    token_kind,
    token_kind_lookup,
)


class C3AuxiliaryDiagnosticsTests(unittest.TestCase):
    def test_token_kind_uses_prefix(self) -> None:
        self.assertEqual(token_kind("RAW|D:0:0:0:10"), "RAW")
        self.assertEqual(token_kind("REF|12"), "REF")
        self.assertEqual(token_kind("RES|x:y"), "RES")
        self.assertEqual(token_kind(""), "UNKNOWN")

    def test_token_kind_lookup_maps_ids(self) -> None:
        self.assertEqual(
            token_kind_lookup({1: "RAW|a", 2: "REF|b", 3: "RES|c"}),
            {1: "RAW", 2: "REF", 3: "RES"},
        )

    def test_topk_accumulator_compares_model_and_unigram(self) -> None:
        accumulator = C3TopKAccumulator(
            top_ks=(1, 2),
            kind_by_id={1: "RAW", 2: "REF", 3: "RES", 4: "RAW"},
        )
        logits = torch.tensor(
            [
                [0.0, 4.0, 3.0, 1.0],
                [5.0, 2.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        tokens = torch.tensor(
            [
                [2, 3, 0],
                [1, 0, 0],
            ],
            dtype=torch.long,
        )
        mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ],
            dtype=torch.bool,
        )

        accumulator.update(
            logits=logits,
            tokens=tokens,
            token_mask=mask,
            available=torch.tensor([True, True], dtype=torch.bool),
            unigram_top_ids=(1, 4),
        )

        result = accumulator.to_dict()
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["positive_sample_count"], 2)
        self.assertEqual(result["positive_label_count"], 3)
        self.assertEqual(result["target_kind_counts"], {"RAW": 1, "REF": 1, "RES": 1})
        self.assertAlmostEqual(result["model"]["topk"]["1"]["micro_recall"], 2 / 3)
        self.assertAlmostEqual(result["model"]["topk"]["2"]["micro_recall"], 1.0)
        self.assertAlmostEqual(result["unigram"]["topk"]["2"]["micro_recall"], 1 / 3)
        self.assertEqual(result["model"]["topk"]["2"]["kind_hits"], {"RAW": 1, "REF": 1, "RES": 1})
        self.assertEqual(result["unigram"]["topk"]["2"]["kind_hits"], {"RAW": 1})

    def test_topk_accumulator_ignores_unavailable_samples(self) -> None:
        accumulator = C3TopKAccumulator(top_ks=(1,), kind_by_id={1: "RAW"})
        accumulator.update(
            logits=torch.tensor([[10.0], [10.0]], dtype=torch.float32),
            tokens=torch.tensor([[1], [1]], dtype=torch.long),
            token_mask=None,
            available=torch.tensor([True, False], dtype=torch.bool),
            unigram_top_ids=(1,),
        )

        result = accumulator.to_dict()
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["available_sample_count"], 1)
        self.assertEqual(result["positive_sample_count"], 1)
        self.assertEqual(result["positive_label_count"], 1)


if __name__ == "__main__":
    unittest.main()
