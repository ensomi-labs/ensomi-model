from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

from pulsefield_model.evals.mapper_v3_factorized_event_signature_proxy import (
    SplitFactorizedProxyAccumulator,
    _decision_from_comparison,
    factorize_event_signature,
    reconstruct_event_signature,
)
from pulsefield_model.models.mapper.shared.vocab import LaneAction
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


class MapperV3FactorizedEventSignatureProxyTests(unittest.TestCase):
    def test_factorization_round_trips_event_signatures(self) -> None:
        for signature in ("T...", ".T.T", "S..E", "TTTT", "SE.T"):
            with self.subTest(signature=signature):
                factors = factorize_event_signature(signature)
                self.assertEqual(reconstruct_event_signature(factors), signature)

    def test_reconstruction_rejects_inconsistent_factors(self) -> None:
        with self.assertRaises(ValueError):
            reconstruct_event_signature(("EV_COUNT_2", "EV_MASK_1000", "EV_ACTION_T"))

    def test_accumulator_counts_factorized_event_tokens(self) -> None:
        source = MapperV21Vocab()
        target = MapperV3Vocab(source.time_shift_values_ms)
        accumulator = SplitFactorizedProxyAccumulator(split_name="test", source_vocab=source, target_vocab=target)
        token_ids = (
            source.time_shift_token_id(100),
            source.lane_action_token_id(0, LaneAction.TAP),
            source.lane_action_token_id(1, LaneAction.TAP),
            source.time_shift_token_id(100),
            source.lane_action_token_id(2, LaneAction.HOLD_START),
        )

        accumulator.update(token_ids=token_ids, dataset_index=0)

        self.assertEqual(accumulator.current_v3_token_count, 4)
        self.assertEqual(accumulator.factorized_token_count, 9)
        self.assertEqual(accumulator.event_token_count, 2)
        self.assertEqual(accumulator.event_factor_token_count, 7)
        self.assertEqual(accumulator.reconstruction_mismatches, 0)
        self.assertEqual(accumulator.current_event_signature_counts, {"TT..": 1, "..S.": 1})
        self.assertEqual(accumulator.lane_count_factor_counts, {"EV_COUNT_2": 1, "EV_COUNT_1": 1})
        self.assertEqual(accumulator.lane_mask_factor_counts, {"EV_MASK_1100": 1, "EV_MASK_0010": 1})
        self.assertEqual(accumulator.action_factor_counts, {"EV_ACTION_T": 2, "EV_ACTION_S": 1})

    def test_decision_tests_factorized_grammar_when_close_and_event_bits_fall(self) -> None:
        decision = _decision_from_comparison(
            {
                "reconstruction_mismatches": 0,
                "current_v3_reconstruction_mismatches": 0,
                "factor_reconstruction_mismatches": 0,
                "v2_1_total_bits": 100.0,
                "current_v3_total_bits": 92.0,
                "factorized_total_bits": 94.0,
                "factorized_total_bit_delta_ratio_vs_current_v3": 2.0 / 92.0,
                "event_bits_per_original_event_reduction_ratio": 0.20,
            }
        )

        self.assertEqual(decision["route"], "TEST_FACTORIZE_V3_EVENT_SIGNATURE_GRAMMAR")
        self.assertFalse(decision["kill_criteria_triggered"])

    def test_decision_kills_factorization_when_total_bits_lose_to_v2_1(self) -> None:
        decision = _decision_from_comparison(
            {
                "reconstruction_mismatches": 0,
                "current_v3_reconstruction_mismatches": 0,
                "factor_reconstruction_mismatches": 0,
                "v2_1_total_bits": 100.0,
                "current_v3_total_bits": 92.0,
                "factorized_total_bits": 101.0,
                "factorized_total_bit_delta_ratio_vs_current_v3": 9.0 / 92.0,
                "event_bits_per_original_event_reduction_ratio": 0.20,
            }
        )

        self.assertEqual(decision["route"], "KILL_FACTORIZE_EVENT_SIGNATURE")
        self.assertIn("factorized_total_bits_not_below_v2_1", decision["kill_criteria"])


if __name__ == "__main__":
    unittest.main()
