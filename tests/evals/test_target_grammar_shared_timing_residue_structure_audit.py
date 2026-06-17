from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

from pulsefield_model.evals.target_grammar_shared_timing_residue_structure_audit import (
    TimingResidueAccumulator,
    checks_from_comparison,
    decision_from_comparison,
    distribution_stats,
    event_intervals_from_times,
    event_times_from_v3_tokens,
    time_shift_values_from_v2_1_tokens,
    time_shift_values_from_v3_tokens,
)
from pulsefield_model.models.mapper.shared.vocab import LaneAction
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.conversion import v2_1_tokens_to_v3_event_tokens
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


class TargetGrammarSharedTimingResidueStructureAuditTests(unittest.TestCase):
    def test_time_shift_parity_and_event_intervals_from_converted_stream(self) -> None:
        source = MapperV21Vocab()
        target = MapperV3Vocab(source.time_shift_values_ms)
        token_ids = (
            source.time_shift_token_id(100),
            source.time_shift_token_id(60),
            source.lane_action_token_id(0, LaneAction.TAP),
            source.time_shift_token_id(100),
            source.time_shift_token_id(60),
            source.lane_action_token_id(1, LaneAction.TAP),
            source.time_shift_token_id(200),
            source.lane_action_token_id(2, LaneAction.HOLD_START),
        )
        converted = v2_1_tokens_to_v3_event_tokens(token_ids, source_vocab=source, target_vocab=target)

        self.assertEqual(
            time_shift_values_from_v2_1_tokens(token_ids, vocab=source),
            time_shift_values_from_v3_tokens(converted.token_ids, vocab=target),
        )
        self.assertEqual(event_times_from_v3_tokens(converted.token_ids, vocab=target), (160, 320, 520))
        self.assertEqual(event_intervals_from_times((160, 320, 520)), (160, 200))

    def test_accumulator_records_timing_counts_and_parity(self) -> None:
        source = MapperV21Vocab()
        target = MapperV3Vocab(source.time_shift_values_ms)
        accumulator = TimingResidueAccumulator(split_name="test", source_vocab=source, target_vocab=target)

        accumulator.update(
            token_ids=(
                source.time_shift_token_id(100),
                source.time_shift_token_id(60),
                source.lane_action_token_id(0, LaneAction.TAP),
                source.time_shift_token_id(200),
                source.lane_action_token_id(1, LaneAction.TAP),
            ),
            dataset_index=0,
        )

        self.assertEqual(accumulator.time_shift_parity_mismatches, 0)
        self.assertEqual(accumulator.v3_reconstruction_mismatches, 0)
        self.assertEqual(accumulator.v2_1_time_shift_counts, {100: 1, 60: 1, 200: 1})
        self.assertEqual(accumulator.event_interval_counts, {200: 1})
        self.assertEqual(accumulator.to_dict()["event_intervals"]["share_200ms"], 1.0)

    def test_distribution_stats_reports_effective_vocab_and_top_share(self) -> None:
        stats = distribution_stats({160: 2, 200: 2, 400: 4})

        self.assertEqual(stats["count"], 8)
        self.assertEqual(stats["vocab_size"], 3)
        self.assertAlmostEqual(stats["top1_share"], 0.5)
        self.assertGreater(stats["effective_vocab"], 2.0)

    def test_decision_routes_to_timing_calibration_when_targets_are_rich(self) -> None:
        decision = decision_from_comparison(
            {
                "time_shift_parity_mismatches": 0,
                "v3_reconstruction_mismatches": 0,
                "missing_comparator_keys": [],
                "event_interval_effective_vocab": 8.0,
                "event_interval_160_200_share": 0.40,
                "target_rigid_window_share_ge_0_95": 0.20,
                "generated_rigid_evidence": True,
            }
        )

        self.assertEqual(decision["route"], "TEST_TIMING_LOSS_OR_EMBEDDING_CALIBRATION")
        self.assertTrue(decision["positive_gate_passed"])

    def test_decision_mutates_when_targets_are_grid_dominated(self) -> None:
        comparison = {
            "time_shift_parity_mismatches": 0,
            "v3_reconstruction_mismatches": 0,
            "missing_comparator_keys": [],
            "event_interval_effective_vocab": 2.0,
            "event_interval_160_200_share": 0.90,
            "target_rigid_window_share_ge_0_95": 0.90,
            "generated_rigid_evidence": True,
        }

        self.assertFalse(checks_from_comparison(comparison)["target_timing_residue_rich"])
        self.assertEqual(decision_from_comparison(comparison)["route"], "MUTATE_TO_TIMING_GRAMMAR_REPAIR")

    def test_decision_kills_when_parity_fails(self) -> None:
        decision = decision_from_comparison(
            {
                "time_shift_parity_mismatches": 1,
                "v3_reconstruction_mismatches": 0,
                "missing_comparator_keys": [],
                "event_interval_effective_vocab": 8.0,
                "event_interval_160_200_share": 0.40,
                "target_rigid_window_share_ge_0_95": 0.20,
                "generated_rigid_evidence": True,
            }
        )

        self.assertEqual(decision["route"], "KILL_TIMING_RESIDUE_AUDIT_INPUTS")
        self.assertIn("time_shift_parity_mismatch", decision["kill_criteria"])


if __name__ == "__main__":
    unittest.main()
