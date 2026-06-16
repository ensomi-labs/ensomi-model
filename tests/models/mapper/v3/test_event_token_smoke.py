import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.models.mapper.v3.conversion import (
    v2_1_tokens_to_v3_event_tokens,
    v3_event_tokens_to_v2_1_tokens,
)
from pulsefield_model.models.mapper.v3.grammar import build_grammar_mask
from pulsefield_model.models.mapper.v3.replay import replay_terminal_state
from pulsefield_model.models.mapper.v3.tokenizer import (
    MapperTimepoint,
    encode_full_chart_tokens,
    encode_mapper_window,
    final_full_chart_token_before,
)
from pulsefield_model.models.mapper.v3.vocab import LaneAction, MapperV3Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV3EventTokenSmokeTests(unittest.TestCase):
    def test_event_signature_roundtrip(self) -> None:
        vocab = MapperV3Vocab()

        token_id = vocab.event_token_id_from_signature("T.SE")

        self.assertTrue(vocab.is_event_token(token_id))
        self.assertEqual(vocab.event_signature(token_id), "T.SE")

    def test_v2_1_sparse_run_converts_to_one_v3_event_and_back(self) -> None:
        source = MapperV21Vocab()
        target = MapperV3Vocab()
        token_ids = (
            source.time_shift_token_id(100),
            source.lane_action_token_id(0, LaneAction.TAP),
            source.lane_action_token_id(2, LaneAction.HOLD_START),
            source.time_shift_token_id(300),
            source.lane_action_token_id(2, LaneAction.HOLD_END),
            source.eos_id,
        )

        converted = v2_1_tokens_to_v3_event_tokens(token_ids, source_vocab=source, target_vocab=target)

        self.assertEqual(converted.reconstructed_v2_1_token_ids, token_ids)
        self.assertEqual(v3_event_tokens_to_v2_1_tokens(converted.token_ids, source_vocab=target, target_vocab=source), token_ids)
        self.assertEqual(target.event_signature(converted.token_ids[1]), "T.S.")
        self.assertEqual(converted.token_reduction, 1)

    def test_terminal_padded_window_places_eos_at_chart_end(self) -> None:
        vocab = MapperV3Vocab()

        tokenized = encode_mapper_window(
            [
                MapperTimepoint(8500, _actions(LaneAction.TAP)),
                MapperTimepoint(9000, _actions(LaneAction.NONE, LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=8000,
            write_end_ms=16000,
            chart_end_ms=9000,
        )

        self.assertTrue(tokenized.is_full_chart_end)
        self.assertEqual(tokenized.write_end_ms, 16000)
        self.assertEqual(tokenized.chart_end_ms, 9000)
        self.assertEqual(vocab.token_name(tokenized.target_fragment_ids[0]), "TS_500")
        self.assertEqual(vocab.event_signature(tokenized.target_fragment_ids[1]), "T...")
        self.assertEqual(vocab.token_name(tokenized.target_fragment_ids[2]), "TS_500")
        self.assertEqual(vocab.event_signature(tokenized.target_fragment_ids[3]), ".T..")
        self.assertEqual(vocab.token_name(tokenized.target_fragment_ids[4]), "EOS")

        terminal = replay_terminal_state(
            tokenized.target_fragment_ids,
            vocab=vocab,
            write_start_ms=tokenized.write_start_ms,
            write_end_ms=tokenized.write_end_ms,
            chart_end_ms=tokenized.chart_end_ms,
            ln_carry_in=tokenized.ln_carry_in,
            ln_carry_out=tokenized.ln_carry_out,
            is_full_chart_end=tokenized.is_full_chart_end,
        )
        self.assertEqual(terminal.current_ms, 9000)

        grammar_mask = build_grammar_mask(
            current_ms=tokenized.target_fragment_current_ms.unsqueeze(0),
            open_mask=tokenized.target_fragment_open_mask.unsqueeze(0),
            open_start_ms=tokenized.target_fragment_open_start_ms.unsqueeze(0),
            open_age_ms=tokenized.target_fragment_open_age_ms.unsqueeze(0),
            write_start_ms=torch.tensor([tokenized.write_start_ms]),
            write_end_ms=torch.tensor([tokenized.write_end_ms]),
            chart_end_ms=torch.tensor([tokenized.chart_end_ms]),
            ln_carry_in=tokenized.ln_carry_in,
            ln_carry_out=tokenized.ln_carry_out,
            is_full_chart_start=torch.tensor([tokenized.is_full_chart_start]),
            is_full_chart_end=torch.tensor([tokenized.is_full_chart_end]),
            vocab=vocab,
        )
        self.assertEqual(float(grammar_mask[0, -1, vocab.eos_id].item()), 0.0)

    def test_cross_window_hold_is_boundary_carry_not_future_target(self) -> None:
        vocab = MapperV3Vocab()

        tokenized = encode_mapper_window(
            [
                MapperTimepoint(500, _actions(LaneAction.HOLD_START)),
                MapperTimepoint(9500, _actions(LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=1000,
            write_end_ms=9000,
        )

        self.assertEqual([vocab.token_name(token_id) for token_id in tokenized.target_fragment_ids], ["TS_4000", "TS_4000"])
        self.assertTrue(tokenized.ln_carry_in.open_mask[0])
        self.assertTrue(tokenized.ln_carry_out.open_mask[0])
        self.assertEqual(tokenized.ln_carry_out.open_start_ms[0], 500)

    def test_full_chart_tokens_use_one_event_group_per_same_time_chord(self) -> None:
        vocab = MapperV3Vocab()

        tokens = encode_full_chart_tokens(
            [MapperTimepoint(1000, _actions(LaneAction.TAP, LaneAction.TAP))],
            vocab=vocab,
            chart_start_ms=0,
            chart_end_ms=1000,
        )

        self.assertEqual(vocab.token_name(tokens[0]), "BOS")
        self.assertEqual(vocab.token_name(tokens[1]), "TS_1000")
        self.assertEqual(vocab.event_signature(tokens[2]), "TT..")
        self.assertEqual(vocab.token_name(tokens[3]), "EOS")

    def test_left_context_keeps_autoregressive_time_shift_to_boundary(self) -> None:
        vocab = MapperV3Vocab()

        token_id = final_full_chart_token_before(
            [MapperTimepoint(7990, _actions(LaneAction.TAP, LaneAction.TAP))],
            vocab=vocab,
            chart_start_ms=0,
            boundary_ms=8000,
        )

        self.assertEqual(vocab.token_name(token_id), "TS_10")


if __name__ == "__main__":
    unittest.main()
