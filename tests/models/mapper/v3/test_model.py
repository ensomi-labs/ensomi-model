import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.models.mapper.v3 import (
    MapperTimepoint,
    MapperV3Config,
    MapperV3LossConfig,
    MapperV3Model,
    MapperV3ModelLoss,
    MapperV3Vocab,
    encode_mapper_window,
    ln_carry_state_tensors,
)
from pulsefield_model.models.mapper.v3.vocab import LaneAction
from pulsefield_model.training.mapper_v3 import _mapper_v3_training_spec


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _batched_carry(carry: object) -> dict[str, torch.Tensor]:
    tensors = ln_carry_state_tensors(carry)  # type: ignore[arg-type]
    return {
        "current_ms": tensors["current_ms"].reshape(1),
        "open_mask": tensors["open_mask"].reshape(1, 4),
        "open_start_ms": tensors["open_start_ms"].reshape(1, 4),
        "open_age_ms": tensors["open_age_ms"].reshape(1, 4),
    }


def _batch_for_window(tokenized: object, *, control_dim: int = 16) -> dict[str, object]:
    return {
        "decoder_input_tokens": tokenized.decoder_input_tensor().unsqueeze(0),  # type: ignore[attr-defined]
        "target_fragment_tokens": tokenized.target_fragment_tensor().unsqueeze(0),  # type: ignore[attr-defined]
        "target_fragment_mask": torch.ones((1, tokenized.seq_len), dtype=torch.bool),  # type: ignore[attr-defined]
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms.unsqueeze(0),  # type: ignore[attr-defined]
            "open_mask": tokenized.target_fragment_open_mask.unsqueeze(0),  # type: ignore[attr-defined]
            "open_start_ms": tokenized.target_fragment_open_start_ms.unsqueeze(0),  # type: ignore[attr-defined]
            "open_age_ms": tokenized.target_fragment_open_age_ms.unsqueeze(0),  # type: ignore[attr-defined]
        },
        "ln_carry_in": _batched_carry(tokenized.ln_carry_in),  # type: ignore[attr-defined]
        "ln_carry_out": _batched_carry(tokenized.ln_carry_out),  # type: ignore[attr-defined]
        "close_labels": tokenized.close_labels.unsqueeze(0),  # type: ignore[attr-defined]
        "close_label_mask": tokenized.close_label_mask.unsqueeze(0),  # type: ignore[attr-defined]
        "write_start_ms": torch.tensor([tokenized.write_start_ms], dtype=torch.long),  # type: ignore[attr-defined]
        "write_end_ms": torch.tensor([tokenized.write_end_ms], dtype=torch.long),  # type: ignore[attr-defined]
        "chart_end_ms": torch.tensor([tokenized.chart_end_ms], dtype=torch.long),  # type: ignore[attr-defined]
        "is_full_chart_start": torch.tensor([tokenized.is_full_chart_start], dtype=torch.bool),  # type: ignore[attr-defined]
        "is_full_chart_end": torch.tensor([tokenized.is_full_chart_end], dtype=torch.bool),  # type: ignore[attr-defined]
        "difficulty": torch.tensor([[3.2]], dtype=torch.float32),
        "normalized_difficulty": torch.tensor([[0.1]], dtype=torch.float32),
        "density_target_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
        "density_confidence_8s": torch.ones((1, 400, 1), dtype=torch.float32),
        "control_memory_8s": torch.zeros((1, 400, control_dim), dtype=torch.float32),
        "density_teacher_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
    }


def _small_config(**overrides: object) -> MapperV3Config:
    values = {
        "control_dim": 16,
        "d_model": 16,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 32,
        "dropout": 0.0,
        "max_seq_len": 64,
        "state_prior_hidden_dim": 16,
        "ln_close_hidden_dim": 16,
        "lane_embedding_dim": 4,
        "age_embedding_dim": 4,
        "use_global_context": False,
        "global_conv_blocks": 0,
    }
    values.update(overrides)
    return MapperV3Config(**values)


class MapperV3ModelTests(unittest.TestCase):
    def test_forward_and_loss_use_event_token_grammar_without_sparse_state(self) -> None:
        torch.manual_seed(20260622)
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(1000, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        model = MapperV3Model(_small_config(), vocab=vocab)

        output = model(batch)

        target = batch["target_fragment_tokens"]
        positions = torch.arange(target.shape[1])
        self.assertTrue(torch.isfinite(output.logits_final[0, positions, target[0]]).all().item())
        event_token = tokenized.target_fragment_ids[1]
        self.assertEqual(vocab.event_signature(event_token), "T.T.")
        self.assertEqual(float(output.grammar_mask[0, 1, event_token].item()), 0.0)

        loss_fn = MapperV3ModelLoss(
            MapperV3LossConfig(lambda_density=0.0, lambda_ln_close=0.0, lambda_adapter_reg=1e-5),
            vocab=vocab,
        )
        loss = loss_fn(output, batch)
        loss.total_loss.backward()

        self.assertTrue(torch.isfinite(loss.total_loss).item())
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum().item()), 0.0)

    def test_training_spec_runs_v3_batch_loss_adapter(self) -> None:
        torch.manual_seed(20260623)
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(1000, _actions(LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        config = _small_config()
        loss_config = MapperV3LossConfig(lambda_density=0.0, lambda_ln_close=0.0, lambda_adapter_reg=0.0)
        model = MapperV3Model(config, vocab=vocab)
        loss_fn = MapperV3ModelLoss(loss_config, vocab=vocab)
        spec = _mapper_v3_training_spec(
            model_config=config,
            control_model_config=None,
            loss_config=loss_config,
        )

        loss = spec.batch_loss_adapter(model, loss_fn, batch, torch.device("cpu"))

        self.assertTrue(torch.isfinite(loss.total_loss).item())
        self.assertEqual(loss.metrics["phase/lambda_density"], 0.0)

    def test_incremental_decode_is_not_silently_inherited_from_v2(self) -> None:
        model = MapperV3Model(_small_config())

        with self.assertRaisesRegex(ValueError, "mapper v3 incremental decode is not implemented yet"):
            model.incremental_decode_next_token()


if __name__ == "__main__":
    unittest.main()
