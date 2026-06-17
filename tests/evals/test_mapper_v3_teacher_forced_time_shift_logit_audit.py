import math
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.evals.mapper_v3_teacher_forced_time_shift_logit_audit import (
    TimeShiftLogitAccumulator,
    decision_from_metrics,
    jensen_shannon_divergence,
)
from pulsefield_model.models.mapper.v3 import MapperV3Vocab


class MapperV3TeacherForcedTimeShiftLogitAuditTests(unittest.TestCase):
    def test_accumulator_scores_only_time_shift_rows(self) -> None:
        vocab = MapperV3Vocab()
        shift_60 = vocab.time_shift_token_id(60)
        shift_100 = vocab.time_shift_token_id(100)
        shift_200 = vocab.time_shift_token_id(200)
        event = vocab.event_token_id_from_signature("T...")
        target = torch.tensor([[shift_60, event, shift_100]], dtype=torch.long)
        mask = torch.ones_like(target, dtype=torch.bool)
        logits = torch.full((1, 3, vocab.size), -5.0, dtype=torch.float32)
        logits[0, 0, shift_60] = 4.0
        logits[0, 0, shift_100] = 3.0
        logits[0, 1, event] = 8.0
        logits[0, 2, shift_200] = 5.0
        logits[0, 2, shift_100] = 4.0

        accumulator = TimeShiftLogitAccumulator(vocab=vocab, top_ks=(1, 2, 5))
        accumulator.update(logits_final=logits, target_tokens=target, target_mask=mask)
        metrics = accumulator.to_dict()

        self.assertEqual(metrics["time_shift_row_count"], 2)
        self.assertEqual(metrics["scored_token_count"], 3)
        self.assertAlmostEqual(metrics["recall_at_k"]["1"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_k"]["2"], 1.0)
        self.assertEqual(metrics["target_shift_counts"], {60: 1, 100: 1})
        self.assertEqual(metrics["argmax_shift_counts"], {60: 1, 200: 1})
        self.assertEqual(metrics["argmax_top_shift_ms"], 60)
        self.assertTrue(math.isfinite(metrics["target_time_shift_nll"]))

    def test_jensen_shannon_divergence_is_zero_for_matching_distributions(self) -> None:
        self.assertEqual(jensen_shannon_divergence({"60": 0.5, "100": 0.5}, {"60": 0.5, "100": 0.5}), 0.0)
        self.assertGreater(jensen_shannon_divergence({"60": 1.0}, {"100": 1.0}), 0.0)

    def test_decision_routes_collapsed_teacher_forced_logits_to_timing_mutation(self) -> None:
        metrics = {
            "time_shift_row_count": 10,
            "target_time_shift_nll": 3.0,
            "target_time_shift_prob_mean": 0.05,
            "recall_at_k": {"5": 0.2},
            "rank": {"median": 8},
            "argmax_top_shift_share": 0.8,
            "argmax_rigid_proxy_piece_share_60_100_200": 0.9,
        }
        checks = {
            "checkpoint_exists": True,
            "training_report_exists": True,
            "checkpoint_model_v3": True,
            "training_report_contract_v3": True,
            "dataset_non_empty": True,
            "time_shift_rows_positive": True,
            "rank_metrics_finite": True,
            "no_training_or_rollout": True,
        }

        decision = decision_from_metrics(metrics, checks=checks)

        self.assertEqual(decision["route"], "MUTATE_TIMING_EMBEDDING_OR_LOSS")


if __name__ == "__main__":
    unittest.main()
