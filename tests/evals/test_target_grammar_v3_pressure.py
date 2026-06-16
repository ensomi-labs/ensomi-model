import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

from pulsefield_model.evals.target_grammar_v3_pressure import (
    GROUP_TOKEN_PREFIX,
    candidate_stream_from_v2_tokens,
    extract_group_runs,
    group_signature_from_token_ids,
    lane_action_symbol,
    top_group_signatures,
    unigram_nll_bits,
)
from pulsefield_model.models.mapper.v2_1.vocab import LaneAction, MapperV21Vocab


class TargetGrammarV3PressureTests(unittest.TestCase):
    def test_lane_action_symbol_maps_sparse_actions(self) -> None:
        self.assertEqual(lane_action_symbol(LaneAction.NONE), ".")
        self.assertEqual(lane_action_symbol(LaneAction.TAP), "T")
        self.assertEqual(lane_action_symbol(LaneAction.HOLD_START), "S")
        self.assertEqual(lane_action_symbol(LaneAction.HOLD_END), "E")

    def test_group_signature_uses_four_lane_slots(self) -> None:
        vocab = MapperV21Vocab()
        token_ids = (
            vocab.lane_action_token_id(1, LaneAction.TAP),
            vocab.lane_action_token_id(3, LaneAction.HOLD_START),
        )

        self.assertEqual(group_signature_from_token_ids(token_ids, vocab=vocab), ".T.S")

    def test_extract_group_runs_splits_on_time_shifts_and_specials(self) -> None:
        vocab = MapperV21Vocab()
        token_ids = (
            vocab.time_shift_token_id(100),
            vocab.lane_action_token_id(0, LaneAction.TAP),
            vocab.lane_action_token_id(2, LaneAction.HOLD_START),
            vocab.time_shift_token_id(200),
            vocab.lane_action_token_id(1, LaneAction.HOLD_END),
            vocab.eos_id,
        )

        runs = extract_group_runs(token_ids, vocab=vocab)

        self.assertEqual([run.signature for run in runs], ["T.S.", ".E.."])
        self.assertEqual([run.start_index for run in runs], [1, 4])

    def test_candidate_stream_reconstructs_exact_v2_tokens_with_fallback(self) -> None:
        vocab = MapperV21Vocab()
        token_ids = (
            vocab.time_shift_token_id(100),
            vocab.lane_action_token_id(0, LaneAction.TAP),
            vocab.lane_action_token_id(2, LaneAction.HOLD_START),
            vocab.time_shift_token_id(200),
            vocab.lane_action_token_id(1, LaneAction.HOLD_END),
            vocab.eos_id,
        )
        baseline_names = tuple(vocab.token_name(token_id) for token_id in token_ids)

        candidate = candidate_stream_from_v2_tokens(
            token_ids,
            vocab=vocab,
            group_dictionary={"T.S."},
        )

        self.assertEqual(
            candidate.token_names,
            (
                "TS_100",
                f"{GROUP_TOKEN_PREFIX}T.S.",
                "TS_200",
                "LANE_2_HOLD_END",
                "EOS",
            ),
        )
        self.assertEqual(candidate.reconstructed_token_names, baseline_names)
        self.assertEqual(candidate.total_groups, 2)
        self.assertEqual(candidate.replaced_groups, 1)
        self.assertEqual(candidate.fallback_groups, 1)
        self.assertEqual(candidate.token_savings, 1)

    def test_top_group_signatures_are_frequency_then_signature_sorted(self) -> None:
        self.assertEqual(
            top_group_signatures({"T...": 2, ".T..": 3, "TT..": 3}, limit=2),
            (".T..", "TT.."),
        )

    def test_unigram_nll_bits_scores_candidate_streams(self) -> None:
        baseline = unigram_nll_bits(
            train_streams=(("TS_100", "LANE_1_TAP", "LANE_2_TAP"),),
            eval_streams=(("TS_100", "LANE_1_TAP", "LANE_2_TAP"),),
        )
        candidate = unigram_nll_bits(
            train_streams=(("TS_100", f"{GROUP_TOKEN_PREFIX}TT.."),),
            eval_streams=(("TS_100", f"{GROUP_TOKEN_PREFIX}TT.."),),
        )

        self.assertEqual(candidate["eval_token_count"], 2)
        self.assertLess(candidate["total_bits"], baseline["total_bits"])
        self.assertGreater(candidate["bits_per_token"], 0.0)


if __name__ == "__main__":
    unittest.main()
