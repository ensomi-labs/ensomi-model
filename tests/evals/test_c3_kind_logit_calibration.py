import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.evals.c3_kind_logit_calibration import (
    CalibrationTransform,
    PredictionSample,
    _decision,
    apply_transform,
    candidate_transforms,
    score_samples,
    score_unigram_samples,
)


class C3KindLogitCalibrationTests(unittest.TestCase):
    def test_apply_transform_scales_and_biases_kind_indexes(self) -> None:
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        transform = CalibrationTransform(
            name="x",
            family="test",
            kind_biases={"RAW": 1.0, "RES": -1.0},
            kind_scales={"RAW": 2.0, "RES": 0.5},
        )

        transformed = apply_transform(
            logits,
            transform,
            {
                "RAW": torch.tensor([0, 1], dtype=torch.long),
                "REF": torch.tensor([2], dtype=torch.long),
                "RES": torch.tensor([3], dtype=torch.long),
            },
        )

        self.assertEqual(transformed.tolist(), [3.0, 5.0, 3.0, 1.0])
        self.assertEqual(logits.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_score_samples_reports_kind_recall_and_predictions(self) -> None:
        samples = (
            PredictionSample(sample_index=0, logits=torch.tensor([1.0, 4.0, 3.0]), target_ids=frozenset({2, 3})),
            PredictionSample(sample_index=1, logits=torch.tensor([5.0, 0.0, 1.0]), target_ids=frozenset({1})),
        )
        transform = CalibrationTransform(name="identity", family="identity", kind_biases={}, kind_scales={})

        metrics = score_samples(
            samples,
            transform=transform,
            kind_by_id={1: "RAW", 2: "REF", 3: "RES"},
            kind_index_by_name={
                "RAW": torch.tensor([0], dtype=torch.long),
                "REF": torch.tensor([1], dtype=torch.long),
                "RES": torch.tensor([2], dtype=torch.long),
            },
            unigram_top_ids=(1, 2),
            top_k=2,
        )

        self.assertEqual(metrics["hits"], 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["kind_hits"], {"RAW": 1, "REF": 1, "RES": 1})
        self.assertEqual(metrics["predicted_kind_counts"], {"RAW": 1, "REF": 1, "RES": 2})

    def test_score_unigram_samples_uses_static_top_ids(self) -> None:
        samples = (
            PredictionSample(sample_index=0, logits=torch.tensor([0.0, 0.0, 0.0]), target_ids=frozenset({2, 3})),
            PredictionSample(sample_index=1, logits=torch.tensor([0.0, 0.0, 0.0]), target_ids=frozenset({1})),
        )

        metrics = score_unigram_samples(
            samples,
            kind_by_id={1: "RAW", 2: "REF", 3: "RES"},
            unigram_top_ids=(1, 2),
            top_k=2,
        )

        self.assertEqual(metrics["hits"], 2)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertEqual(metrics["kind_hits"], {"RAW": 1, "REF": 1})

    def test_candidate_transforms_include_expected_families(self) -> None:
        families = {candidate.family for candidate in candidate_transforms()}

        self.assertIn("identity", families)
        self.assertIn("raw_bias", families)
        self.assertIn("kind_bias", families)
        self.assertIn("scale_bias", families)

    def test_decision_positive_when_hits_or_raw_gap_improve_and_non_raw_preserved(self) -> None:
        summary = _summary_for_decision(
            identity_hits=10,
            selected_hits=15,
            unigram_hits=18,
            oracle_hits=16,
            identity_raw=0.10,
            selected_raw=0.20,
            unigram_raw=0.30,
            identity_non_raw=0.50,
            selected_non_raw=0.46,
        )

        decision = _decision(summary)

        self.assertEqual(decision["route"], "TEST_NEXT")
        self.assertTrue(decision["positive_signal_observed"])

    def test_decision_mutates_to_raw_split_when_calibration_has_no_headroom(self) -> None:
        summary = _summary_for_decision(
            identity_hits=10,
            selected_hits=10,
            unigram_hits=18,
            oracle_hits=11,
            identity_raw=0.10,
            selected_raw=0.11,
            unigram_raw=0.30,
            identity_non_raw=0.50,
            selected_non_raw=0.50,
        )

        decision = _decision(summary)

        self.assertEqual(decision["route"], "MUTATE_TO_RAW_SPLIT")
        self.assertFalse(decision["positive_signal_observed"])


def _summary_for_decision(
    *,
    identity_hits: int,
    selected_hits: int,
    unigram_hits: int,
    oracle_hits: int,
    identity_raw: float,
    selected_raw: float,
    unigram_raw: float,
    identity_non_raw: float,
    selected_non_raw: float,
) -> dict[str, object]:
    return {
        "identity": {
            "heldout": {
                "hits": identity_hits,
                "kind_recall": {"RAW": identity_raw},
                "non_raw_recall": identity_non_raw,
            }
        },
        "selected": {
            "heldout": {
                "hits": selected_hits,
                "kind_recall": {"RAW": selected_raw},
                "non_raw_recall": selected_non_raw,
            }
        },
        "unigram_baseline": {
            "heldout": {
                "hits": unigram_hits,
                "kind_recall": {"RAW": unigram_raw},
            }
        },
        "unconstrained_best_on_calibration": {
            "heldout": {
                "hits": selected_hits,
                "kind_recall": {"RAW": selected_raw},
                "non_raw_recall": selected_non_raw,
            }
        },
        "oracle_best_on_heldout": {
            "heldout": {
                "hits": oracle_hits,
                "non_raw_recall": selected_non_raw,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
