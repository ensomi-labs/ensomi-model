import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

from pulsefield_model.evals.target_grammar_v3_event_smoke import (
    _conversion_totals,
    _decision_from_comparison,
    _event_signature_counts,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.conversion import v2_1_tokens_to_v3_event_tokens
from pulsefield_model.models.mapper.v3.vocab import LaneAction, MapperV3Vocab


class TargetGrammarV3EventSmokeTests(unittest.TestCase):
    def test_event_signature_counts_and_totals(self) -> None:
        source = MapperV21Vocab()
        target = MapperV3Vocab()
        conversion = v2_1_tokens_to_v3_event_tokens(
            (
                source.time_shift_token_id(100),
                source.lane_action_token_id(0, LaneAction.TAP),
                source.lane_action_token_id(1, LaneAction.TAP),
                source.time_shift_token_id(100),
                source.lane_action_token_id(2, LaneAction.HOLD_START),
            ),
            source_vocab=source,
            target_vocab=target,
        )

        self.assertEqual(_conversion_totals((conversion,)), {
            "event_token_count": 2,
            "multi_lane_event_count": 1,
            "lane_action_token_count": 3,
        })
        self.assertEqual(_event_signature_counts((conversion,), vocab=target), {"TT..": 1, "..S.": 1})

    def test_decision_requires_exact_reconstruction_and_pressure(self) -> None:
        positive = _decision_from_comparison(
            {
                "reconstruction_mismatches": 0,
                "total_bit_reduction_ratio": 0.04,
                "token_reduction_ratio": 0.12,
            }
        )
        mismatch = _decision_from_comparison(
            {
                "reconstruction_mismatches": 1,
                "total_bit_reduction_ratio": 0.10,
                "token_reduction_ratio": 0.20,
            }
        )
        weak = _decision_from_comparison(
            {
                "reconstruction_mismatches": 0,
                "total_bit_reduction_ratio": 0.01,
                "token_reduction_ratio": 0.20,
            }
        )

        self.assertEqual(positive["route"], "TEST_NEXT")
        self.assertEqual(mismatch["route"], "KILL")
        self.assertEqual(weak["route"], "MUTATE")


if __name__ == "__main__":
    unittest.main()
