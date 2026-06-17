import unittest
from types import SimpleNamespace

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
from pulsefield_model.models.mapper.v3.factor_target import (
    collate_delta_event_factor_targets,
    delta_event_factor_target_from_v3_tokens,
)
from pulsefield_model.models.mapper.shared.loss import (
    conditioned_event_distribution_loss,
    continuation_jump_guard_loss,
    event_budget_auxiliary_loss,
    time_shift_distance_loss,
)
from pulsefield_model.models.mapper.v3.replay import initial_replay_state, transition_replay_state
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


def _factor_target_for_window(tokenized: object, *, vocab: MapperV3Vocab) -> dict[str, torch.Tensor]:
    return collate_delta_event_factor_targets(
        [
            delta_event_factor_target_from_v3_tokens(
                tokenized.target_fragment_ids,  # type: ignore[attr-defined]
                vocab=vocab,
                write_start_ms=int(tokenized.write_start_ms),  # type: ignore[attr-defined]
                write_end_ms=int(tokenized.write_end_ms),  # type: ignore[attr-defined]
                chart_end_ms=int(tokenized.chart_end_ms),  # type: ignore[attr-defined]
                is_full_chart_end=bool(tokenized.is_full_chart_end),  # type: ignore[attr-defined]
            ).as_tensor_mapping()
        ]
    )


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

    def test_delta_event_auxiliary_target_is_default_off(self) -> None:
        torch.manual_seed(20260626)
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(80, _actions(LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=80,
        )
        batch = _batch_for_window(tokenized)
        model = MapperV3Model(_small_config(), vocab=vocab)

        output = model(batch)
        loss = MapperV3ModelLoss(
            MapperV3LossConfig(lambda_density=0.0, lambda_ln_close=0.0, lambda_adapter_reg=0.0),
            vocab=vocab,
        )(output, batch)

        self.assertIsNone(output.delta_event_delta_logits)
        self.assertIsNone(output.delta_event_signature_logits)
        self.assertIsNone(output.delta_event_end_gap_logits)
        self.assertIsNone(output.delta_event_factor_kind_logits)
        self.assertIsNone(output.delta_event_factor_delta_logits)
        self.assertIsNone(output.delta_event_factor_signature_logits)
        self.assertIsNone(output.delta_event_factor_end_gap_logits)
        self.assertEqual(loss.metrics["phase/lambda_delta_event_auxiliary"], 0.0)
        self.assertEqual(loss.metrics["phase/lambda_delta_event_factor_target"], 0.0)
        self.assertEqual(float(loss.delta_event_auxiliary_loss.item()), 0.0)
        self.assertEqual(float(loss.delta_event_factor_target_loss.item()), 0.0)

    def test_delta_event_auxiliary_target_head_receives_gradients(self) -> None:
        torch.manual_seed(20260627)
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(80, _actions(LaneAction.TAP)),
                MapperTimepoint(240, _actions(LaneAction.NONE, LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        model = MapperV3Model(
            _small_config(
                use_delta_event_auxiliary_target=True,
                delta_event_delta_max_ms=8000,
                delta_event_end_gap_max_ms=8000,
            ),
            vocab=vocab,
        )

        output = model(batch)

        self.assertIsNotNone(output.delta_event_delta_logits)
        self.assertIsNotNone(output.delta_event_signature_logits)
        self.assertIsNotNone(output.delta_event_end_gap_logits)
        assert output.delta_event_delta_logits is not None
        assert output.delta_event_signature_logits is not None
        assert output.delta_event_end_gap_logits is not None
        self.assertEqual(tuple(output.delta_event_delta_logits.shape), (1, tokenized.seq_len, 801))
        self.assertEqual(tuple(output.delta_event_signature_logits.shape), (1, tokenized.seq_len, len(vocab.event_token_ids)))
        self.assertEqual(tuple(output.delta_event_end_gap_logits.shape), (1, tokenized.seq_len, 801))

        loss_fn = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
                lambda_delta_event_auxiliary=0.25,
            ),
            vocab=vocab,
        )
        loss = loss_fn(output, batch)
        loss.total_loss.backward()

        self.assertTrue(torch.isfinite(loss.total_loss).item())
        self.assertIn("loss/delta_event_auxiliary", loss.metrics)
        self.assertGreater(loss.metrics["loss/delta_event_auxiliary"], 0.0)
        self.assertEqual(loss.metrics["delta_event_auxiliary/event_label_count"], 2.0)
        self.assertEqual(loss.metrics["delta_event_auxiliary/end_gap_label_count"], 1.0)
        self.assertIsNotNone(model.delta_event_delta_head)
        self.assertIsNotNone(model.delta_event_signature_head)
        self.assertIsNotNone(model.delta_event_end_gap_head)
        assert model.delta_event_delta_head is not None
        assert model.delta_event_signature_head is not None
        assert model.delta_event_end_gap_head is not None
        self.assertIsNotNone(model.delta_event_delta_head.weight.grad)
        self.assertIsNotNone(model.delta_event_signature_head.weight.grad)
        self.assertIsNotNone(model.delta_event_end_gap_head.weight.grad)
        self.assertGreater(float(model.delta_event_delta_head.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(model.delta_event_signature_head.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(model.delta_event_end_gap_head.weight.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum().item()), 0.0)

    def test_delta_event_factor_target_head_receives_gradients(self) -> None:
        torch.manual_seed(20260631)
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(80, _actions(LaneAction.TAP)),
                MapperTimepoint(240, _actions(LaneAction.NONE, LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        batch["delta_event_factor_target"] = _factor_target_for_window(tokenized, vocab=vocab)
        model = MapperV3Model(
            _small_config(
                use_delta_event_factor_target=True,
                delta_event_factor_delta_max_ms=8000,
                delta_event_factor_end_gap_max_ms=8000,
            ),
            vocab=vocab,
        )

        output = model(batch)

        self.assertIsNotNone(output.delta_event_factor_kind_logits)
        self.assertIsNotNone(output.delta_event_factor_delta_logits)
        self.assertIsNotNone(output.delta_event_factor_signature_logits)
        self.assertIsNotNone(output.delta_event_factor_end_gap_logits)
        assert output.delta_event_factor_kind_logits is not None
        assert output.delta_event_factor_delta_logits is not None
        assert output.delta_event_factor_signature_logits is not None
        assert output.delta_event_factor_end_gap_logits is not None
        self.assertEqual(tuple(output.delta_event_factor_kind_logits.shape), (1, 3, 2))
        self.assertEqual(tuple(output.delta_event_factor_delta_logits.shape), (1, 3, 801))
        self.assertEqual(tuple(output.delta_event_factor_signature_logits.shape), (1, 3, len(vocab.event_token_ids)))
        self.assertEqual(tuple(output.delta_event_factor_end_gap_logits.shape), (1, 3, 801))

        loss_fn = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
                lambda_delta_event_factor_target=0.5,
            ),
            vocab=vocab,
        )
        loss = loss_fn(output, batch)
        loss.total_loss.backward()

        self.assertTrue(torch.isfinite(loss.total_loss).item())
        self.assertIn("loss/delta_event_factor_target", loss.metrics)
        self.assertGreater(loss.metrics["loss/delta_event_factor_target"], 0.0)
        self.assertEqual(loss.metrics["delta_event_factor/kind_label_count"], 3.0)
        self.assertEqual(loss.metrics["delta_event_factor/delta_label_count"], 2.0)
        self.assertEqual(loss.metrics["delta_event_factor/signature_label_count"], 2.0)
        self.assertEqual(loss.metrics["delta_event_factor/end_gap_label_count"], 1.0)
        self.assertIsNotNone(model.delta_event_factor_kind_head)
        self.assertIsNotNone(model.delta_event_factor_delta_head)
        self.assertIsNotNone(model.delta_event_factor_signature_head)
        self.assertIsNotNone(model.delta_event_factor_end_gap_head)
        assert model.delta_event_factor_kind_head is not None
        assert model.delta_event_factor_delta_head is not None
        assert model.delta_event_factor_signature_head is not None
        assert model.delta_event_factor_end_gap_head is not None
        self.assertIsNotNone(model.delta_event_factor_kind_head.weight.grad)
        self.assertIsNotNone(model.delta_event_factor_delta_head.weight.grad)
        self.assertIsNotNone(model.delta_event_factor_signature_head.weight.grad)
        self.assertIsNotNone(model.delta_event_factor_end_gap_head.weight.grad)
        self.assertGreater(float(model.delta_event_factor_kind_head.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(model.delta_event_factor_delta_head.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(model.delta_event_factor_signature_head.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(model.delta_event_factor_end_gap_head.weight.grad.abs().sum().item()), 0.0)

    def test_delta_event_factor_target_requires_nested_batch_field_when_enabled(self) -> None:
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(80, _actions(LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=80,
        )
        batch = _batch_for_window(tokenized)
        model = MapperV3Model(_small_config(use_delta_event_factor_target=True), vocab=vocab)

        with self.assertRaisesRegex(ValueError, "delta_event_factor_target batch field is required"):
            model(batch)

    def test_delta_event_auxiliary_target_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "delta_event_delta_max_ms must be a positive 10ms-grid value"):
            MapperV3Model(
                _small_config(
                    use_delta_event_auxiliary_target=True,
                    delta_event_delta_max_ms=805,
                ),
                vocab=MapperV3Vocab(),
            )
        with self.assertRaisesRegex(ValueError, "lambda_delta_event_auxiliary must be non-negative"):
            MapperV3ModelLoss(MapperV3LossConfig(lambda_delta_event_auxiliary=-0.1), vocab=MapperV3Vocab())
        with self.assertRaisesRegex(ValueError, "delta_event_factor_delta_max_ms must be a positive 10ms-grid value"):
            MapperV3Model(
                _small_config(
                    use_delta_event_factor_target=True,
                    delta_event_factor_delta_max_ms=805,
                ),
                vocab=MapperV3Vocab(),
            )
        with self.assertRaisesRegex(ValueError, "lambda_delta_event_factor_target must be non-negative"):
            MapperV3ModelLoss(MapperV3LossConfig(lambda_delta_event_factor_target=-0.1), vocab=MapperV3Vocab())

    def test_event_budget_loss_prefers_matching_event_mass(self) -> None:
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(1000, _actions(LaneAction.TAP)),
                MapperTimepoint(5000, _actions(LaneAction.TAP, LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=5000,
        )
        batch = _batch_for_window(tokenized)
        target = batch["target_fragment_tokens"]
        mask = batch["target_fragment_mask"]
        states = batch["target_fragment_states"]
        vocab_size = vocab.size
        matching_logits = torch.full((1, tokenized.seq_len, vocab_size), -5.0)
        suppressed_logits = torch.full_like(matching_logits, -5.0)
        time_shift_id = vocab.time_shift_token_id(100)
        for step, token_id in enumerate(tokenized.target_fragment_ids):
            if vocab.is_event_token(token_id):
                matching_logits[0, step, int(token_id)] = 5.0
                suppressed_logits[0, step, time_shift_id] = 5.0
            else:
                matching_logits[0, step, int(token_id)] = 5.0
                suppressed_logits[0, step, int(token_id)] = 5.0

        matching_loss = event_budget_auxiliary_loss(
            logits_final=matching_logits,
            target_tokens=target,
            current_ms=states["current_ms"],
            write_start_ms=batch["write_start_ms"],
            write_end_ms=batch["write_end_ms"],
            vocab=vocab,
            target_mask=mask,
        )
        suppressed_loss = event_budget_auxiliary_loss(
            logits_final=suppressed_logits,
            target_tokens=target,
            current_ms=states["current_ms"],
            write_start_ms=batch["write_start_ms"],
            write_end_ms=batch["write_end_ms"],
            vocab=vocab,
            target_mask=mask,
        )

        self.assertLess(float(matching_loss.item()), float(suppressed_loss.item()))

    def test_event_budget_loss_is_default_off_and_reports_metric_when_enabled(self) -> None:
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(1000, _actions(LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        logits = torch.zeros((1, tokenized.seq_len, vocab.size), dtype=torch.float32)
        output = SimpleNamespace(logits_final=logits)
        off_loss = MapperV3ModelLoss(
            MapperV3LossConfig(lambda_density=0.0, lambda_event_budget=0.0, lambda_ln_close=0.0, lambda_adapter_reg=0.0),
            vocab=vocab,
        )(output, batch)
        on_loss = MapperV3ModelLoss(
            MapperV3LossConfig(lambda_density=0.0, lambda_event_budget=0.5, lambda_ln_close=0.0, lambda_adapter_reg=0.0),
            vocab=vocab,
        )(output, batch)

        self.assertEqual(off_loss.metrics["phase/lambda_event_budget"], 0.0)
        self.assertEqual(float(off_loss.event_budget_loss.item()), 0.0)
        self.assertGreater(float(on_loss.event_budget_loss.item()), 0.0)
        self.assertEqual(on_loss.metrics["phase/lambda_event_budget"], 0.5)
        self.assertGreater(float(on_loss.total_loss.item()), float(off_loss.total_loss.item()))

    def test_conditioned_event_distribution_penalizes_high_difficulty_overproduction(self) -> None:
        vocab = MapperV3Vocab()
        event_id = vocab.event_token_id_from_signature("T...")
        shift_id = vocab.time_shift_token_id(100)
        target = torch.tensor([[shift_id, shift_id, shift_id, shift_id]], dtype=torch.long)
        current_ms = torch.tensor([[1000, 2000, 5000, 6000]], dtype=torch.long)
        write_start_ms = torch.tensor([0], dtype=torch.long)
        write_end_ms = torch.tensor([8000], dtype=torch.long)
        mask = torch.ones_like(target, dtype=torch.bool)
        logits = torch.full((1, target.shape[1], vocab.size), -5.0, dtype=torch.float32)
        logits[:, :, shift_id] = 5.0
        logits[:, 2, event_id] = 6.0

        low_difficulty_loss = conditioned_event_distribution_loss(
            logits_final=logits,
            target_tokens=target,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            normalized_difficulty=torch.tensor([[0.1]], dtype=torch.float32),
            vocab=vocab,
            target_mask=mask,
            zero_target_over_weight=2.0,
            high_difficulty_over_weight=4.0,
            high_difficulty_min=0.75,
        )
        high_difficulty_loss = conditioned_event_distribution_loss(
            logits_final=logits,
            target_tokens=target,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            normalized_difficulty=torch.tensor([[0.9]], dtype=torch.float32),
            vocab=vocab,
            target_mask=mask,
            zero_target_over_weight=2.0,
            high_difficulty_over_weight=4.0,
            high_difficulty_min=0.75,
        )

        self.assertTrue(torch.isfinite(high_difficulty_loss).item())
        self.assertGreater(float(high_difficulty_loss.item()), float(low_difficulty_loss.item()))

    def test_conditioned_event_distribution_loss_is_default_off_and_reports_metric_when_enabled(self) -> None:
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(1000, _actions(LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        logits = torch.zeros((1, tokenized.seq_len, vocab.size), dtype=torch.float32)
        output = SimpleNamespace(logits_final=logits)

        off_loss = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_event_budget=0.0,
                lambda_conditioned_event_distribution=0.0,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
            ),
            vocab=vocab,
        )(output, batch)
        on_loss = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_event_budget=0.0,
                lambda_conditioned_event_distribution=0.5,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
            ),
            vocab=vocab,
        )(output, batch)

        self.assertEqual(off_loss.metrics["phase/lambda_conditioned_event_distribution"], 0.0)
        self.assertEqual(float(off_loss.conditioned_event_distribution_loss.item()), 0.0)
        self.assertEqual(on_loss.metrics["phase/lambda_conditioned_event_distribution"], 0.5)
        self.assertGreater(float(on_loss.conditioned_event_distribution_loss.item()), 0.0)
        self.assertGreater(float(on_loss.total_loss.item()), float(off_loss.total_loss.item()))

    def test_conditioned_event_distribution_requires_normalized_difficulty_when_enabled(self) -> None:
        vocab = MapperV3Vocab()
        event_id = vocab.event_token_id_from_signature("T...")
        output = SimpleNamespace(logits_final=torch.zeros((1, 1, vocab.size), dtype=torch.float32))
        batch = {
            "target_fragment_tokens": torch.tensor([[event_id]], dtype=torch.long),
            "target_fragment_mask": torch.ones((1, 1), dtype=torch.bool),
            "target_fragment_states": {
                "current_ms": torch.tensor([[1000]], dtype=torch.long),
            },
            "write_start_ms": torch.tensor([0], dtype=torch.long),
            "write_end_ms": torch.tensor([8000], dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "normalized_difficulty is required"):
            MapperV3ModelLoss(
                MapperV3LossConfig(
                    lambda_density=0.0,
                    lambda_conditioned_event_distribution=0.5,
                    lambda_ln_close=0.0,
                    lambda_adapter_reg=0.0,
                ),
                vocab=vocab,
            )(output, batch)

    def test_event_token_loss_weight_increases_event_error_pressure(self) -> None:
        vocab = MapperV3Vocab()
        event_id = vocab.event_token_id_from_signature("T...")
        time_shift_id = vocab.time_shift_token_id(100)
        target = torch.tensor([[event_id, time_shift_id]], dtype=torch.long)
        logits = torch.full((1, 2, vocab.size), -5.0, dtype=torch.float32)
        logits[0, 0, time_shift_id] = 5.0
        logits[0, 1, time_shift_id] = 5.0
        output = SimpleNamespace(logits_final=logits)
        batch = {
            "target_fragment_tokens": target,
            "target_fragment_mask": torch.ones_like(target, dtype=torch.bool),
        }

        default_loss = MapperV3ModelLoss(
            MapperV3LossConfig(lambda_density=0.0, lambda_ln_close=0.0, lambda_adapter_reg=0.0),
            vocab=vocab,
        )(output, batch)
        weighted_loss = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
                event_token_loss_weight=4.0,
            ),
            vocab=vocab,
        )(output, batch)

        self.assertGreater(float(weighted_loss.token_loss.item()), float(default_loss.token_loss.item()))
        self.assertEqual(weighted_loss.metrics["phase/event_token_loss_weight"], 4.0)
        self.assertTrue(torch.equal(weighted_loss.total_loss, weighted_loss.token_loss))

    def test_event_token_loss_weight_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "event_token_loss_weight must be positive"):
            MapperV3ModelLoss(MapperV3LossConfig(event_token_loss_weight=0.0), vocab=MapperV3Vocab())

    def test_event_budget_loss_ignores_padded_all_invalid_rows(self) -> None:
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(1000, _actions(LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        target = torch.cat(
            [
                batch["target_fragment_tokens"],
                torch.tensor([[vocab.pad_id]], dtype=torch.long),
            ],
            dim=1,
        )
        mask = torch.cat(
            [
                batch["target_fragment_mask"],
                torch.tensor([[False]], dtype=torch.bool),
            ],
            dim=1,
        )
        current_ms = torch.cat(
            [
                batch["target_fragment_states"]["current_ms"],
                torch.tensor([[0]], dtype=torch.long),
            ],
            dim=1,
        )
        logits = torch.zeros((1, target.shape[1], vocab.size), dtype=torch.float32)
        logits[:, -1, :] = -torch.inf

        loss = event_budget_auxiliary_loss(
            logits_final=logits,
            target_tokens=target,
            current_ms=current_ms,
            write_start_ms=batch["write_start_ms"],
            write_end_ms=batch["write_end_ms"],
            vocab=vocab,
            target_mask=mask,
        )

        self.assertTrue(torch.isfinite(loss).item())

    def test_continuation_jump_guard_penalizes_skipping_next_gold_event(self) -> None:
        vocab = MapperV3Vocab()
        event = vocab.encode_event(_actions(LaneAction.TAP))
        shift_1000 = vocab.time_shift_token_id(1000)
        shift_2000 = vocab.time_shift_token_id(2000)
        shift_4000 = vocab.time_shift_token_id(4000)
        target = torch.tensor(
            [[shift_1000, event, shift_2000, event, vocab.eos_id]],
            dtype=torch.long,
        )
        current_ms = torch.tensor([[0, 1000, 1000, 3000, 3000]], dtype=torch.long)
        mask = torch.ones_like(target, dtype=torch.bool)
        local_logits = torch.full((1, target.shape[1], vocab.size), -5.0)
        jump_logits = torch.full_like(local_logits, -5.0)
        local_logits[0, 0, shift_1000] = 5.0
        local_logits[0, 2, shift_2000] = 5.0
        jump_logits[0, 0, shift_4000] = 5.0
        jump_logits[0, 2, shift_4000] = 5.0

        local_loss = continuation_jump_guard_loss(
            logits_final=local_logits,
            target_tokens=target,
            current_ms=current_ms,
            vocab=vocab,
            target_mask=mask,
            tolerance_ms=100,
        )
        jump_loss = continuation_jump_guard_loss(
            logits_final=jump_logits,
            target_tokens=target,
            current_ms=current_ms,
            vocab=vocab,
            target_mask=mask,
            tolerance_ms=100,
        )

        self.assertLess(float(local_loss.item()), float(jump_loss.item()))
        self.assertTrue(torch.isfinite(jump_loss).item())

    def test_continuation_jump_loss_is_default_off_and_reports_metric_when_enabled(self) -> None:
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(1000, _actions(LaneAction.TAP)),
                MapperTimepoint(3000, _actions(LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=3000,
        )
        batch = _batch_for_window(tokenized)
        logits = torch.zeros((1, tokenized.seq_len, vocab.size), dtype=torch.float32)
        output = SimpleNamespace(logits_final=logits)

        off_loss = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_event_budget=0.0,
                lambda_continuation_jump=0.0,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
            ),
            vocab=vocab,
        )(output, batch)
        on_loss = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_event_budget=0.0,
                lambda_continuation_jump=0.5,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
            ),
            vocab=vocab,
        )(output, batch)

        self.assertEqual(off_loss.metrics["phase/lambda_continuation_jump"], 0.0)
        self.assertEqual(float(off_loss.continuation_jump_loss.item()), 0.0)
        self.assertEqual(on_loss.metrics["phase/lambda_continuation_jump"], 0.5)
        self.assertGreater(float(on_loss.continuation_jump_loss.item()), 0.0)
        self.assertGreater(float(on_loss.total_loss.item()), float(off_loss.total_loss.item()))

    def test_time_shift_distance_loss_prefers_matching_shift_values(self) -> None:
        vocab = MapperV3Vocab()
        shift_60 = vocab.time_shift_token_id(60)
        shift_80 = vocab.time_shift_token_id(80)
        shift_100 = vocab.time_shift_token_id(100)
        event = vocab.event_token_id_from_signature("T...")
        target = torch.tensor([[shift_80, shift_100, shift_60, event]], dtype=torch.long)
        mask = torch.ones_like(target, dtype=torch.bool)
        matching_logits = torch.full((1, target.shape[1], vocab.size), -5.0, dtype=torch.float32)
        rigid_logits = torch.full_like(matching_logits, -5.0)
        for step, token_id in enumerate(target[0].tolist()):
            matching_logits[0, step, int(token_id)] = 5.0
            rigid_logits[0, step, shift_100] = 5.0

        matching = time_shift_distance_loss(
            logits_final=matching_logits,
            target_tokens=target,
            vocab=vocab,
            target_mask=mask,
        )
        rigid = time_shift_distance_loss(
            logits_final=rigid_logits,
            target_tokens=target,
            vocab=vocab,
            target_mask=mask,
        )

        self.assertTrue(torch.isfinite(matching).item())
        self.assertTrue(torch.isfinite(rigid).item())
        self.assertLess(float(matching.item()), float(rigid.item()))

    def test_time_shift_distance_loss_filters_non_target_rows_before_softmax(self) -> None:
        vocab = MapperV3Vocab()
        shift_80 = vocab.time_shift_token_id(80)
        shift_100 = vocab.time_shift_token_id(100)
        event = vocab.event_token_id_from_signature("T...")
        target = torch.tensor([[shift_80, event, vocab.pad_id]], dtype=torch.long)
        mask = torch.tensor([[True, True, False]], dtype=torch.bool)
        logits = torch.zeros((1, target.shape[1], vocab.size), dtype=torch.float32)
        logits[:, :, list(vocab.time_shift_token_ids)] = -torch.inf
        logits[0, 0, shift_80] = 2.0
        logits[0, 0, shift_100] = 0.0
        logits.requires_grad_(True)

        loss = time_shift_distance_loss(
            logits_final=logits,
            target_tokens=target,
            vocab=vocab,
            target_mask=mask,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(float(logits.grad[0, 0].abs().sum().item()), 0.0)
        self.assertEqual(float(logits.grad[0, 1].abs().sum().item()), 0.0)
        self.assertEqual(float(logits.grad[0, 2].abs().sum().item()), 0.0)

    def test_time_shift_distance_loss_is_default_off_and_reports_metric_when_enabled(self) -> None:
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(80, _actions(LaneAction.TAP)),
                MapperTimepoint(240, _actions(LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=240,
        )
        batch = _batch_for_window(tokenized)
        logits = torch.zeros((1, tokenized.seq_len, vocab.size), dtype=torch.float32)
        output = SimpleNamespace(logits_final=logits)

        off_loss = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_time_shift_distance=0.0,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
            ),
            vocab=vocab,
        )(output, batch)
        on_loss = MapperV3ModelLoss(
            MapperV3LossConfig(
                lambda_density=0.0,
                lambda_time_shift_distance=0.5,
                lambda_ln_close=0.0,
                lambda_adapter_reg=0.0,
            ),
            vocab=vocab,
        )(output, batch)

        self.assertEqual(off_loss.metrics["phase/lambda_time_shift_distance"], 0.0)
        self.assertEqual(float(off_loss.time_shift_distance_loss.item()), 0.0)
        self.assertEqual(on_loss.metrics["phase/lambda_time_shift_distance"], 0.5)
        self.assertGreater(float(on_loss.time_shift_distance_loss.item()), 0.0)
        self.assertGreater(float(on_loss.total_loss.item()), float(off_loss.total_loss.item()))

    def test_time_shift_distance_config_requires_positive_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_shift_distance_scale_ms must be positive"):
            MapperV3ModelLoss(
                MapperV3LossConfig(time_shift_distance_scale_ms=0.0),
                vocab=MapperV3Vocab(),
            )

    def test_incremental_decode_matches_cached_full_forward_logits(self) -> None:
        torch.manual_seed(20260624)
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(100, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.HOLD_START)),
                MapperTimepoint(300, _actions(LaneAction.NONE, LaneAction.TAP, LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=500,
        )
        batch = _batch_for_window(tokenized)
        batch["projected_control_memory_8s"] = torch.zeros((1, 400, 16), dtype=torch.float32)
        batch.pop("control_memory_8s")
        batch["density_teacher_8s"] = torch.zeros((1, 400, 1), dtype=torch.float32)
        batch["global_memory"] = torch.randn((1, 5, 16), dtype=torch.float32) * 0.05
        batch["global_memory_padding_mask"] = torch.tensor([[False, False, False, True, True]], dtype=torch.bool)
        batch["global_position_features"] = torch.tensor([[0.0, 0.25, 0.5, 0.75]], dtype=torch.float32)
        model = MapperV3Model(_small_config(layers=2, use_global_context=True), vocab=vocab)
        model.eval()

        with torch.no_grad():
            batch["control_attention_kv_cache"] = model.control_attention_kv_cache(
                batch["projected_control_memory_8s"],
            )
            batch["global_attention_kv_cache"] = model.global_attention_kv_cache(batch["global_memory"])
            full = model(batch)
            decode_state = model.create_empty_decode_state(
                batch_size=int(batch["decoder_input_tokens"].shape[0]),
                device=batch["decoder_input_tokens"].device,
            )
            states = batch["target_fragment_states"]
            for step in range(int(batch["decoder_input_tokens"].shape[1])):
                output = model.incremental_decode_next_token(
                    decode_state=decode_state,
                    decoder_input_token=batch["decoder_input_tokens"][:, step],
                    current_ms=states["current_ms"][:, step],
                    open_mask=states["open_mask"][:, step],
                    open_start_ms=states["open_start_ms"][:, step],
                    open_age_ms=states["open_age_ms"][:, step],
                    write_start_ms=batch["write_start_ms"],
                    write_end_ms=batch["write_end_ms"],
                    chart_end_ms=batch["chart_end_ms"],
                    is_full_chart_start=batch["is_full_chart_start"],
                    is_full_chart_end=batch["is_full_chart_end"],
                    ln_carry_in=batch["ln_carry_in"],
                    ln_carry_out=batch["ln_carry_out"],
                    density_teacher_8s=batch["density_teacher_8s"],
                    projected_control_memory_8s=batch["projected_control_memory_8s"],
                    control_attention_kv_cache=batch["control_attention_kv_cache"],
                    normalized_difficulty=batch["normalized_difficulty"],
                    global_memory=batch["global_memory"],
                    global_memory_padding_mask=batch["global_memory_padding_mask"],
                    global_position_features=batch["global_position_features"],
                    global_attention_kv_cache=batch["global_attention_kv_cache"],
                )

                self.assertTrue(
                    torch.allclose(output.logits_final, full.logits_final[:, step], atol=1e-5, rtol=1e-5),
                    msg=f"incremental logits mismatch at step {step}",
                )
                self.assertTrue(torch.allclose(output.base_logits, full.base_logits[:, step], atol=1e-5, rtol=1e-5))
                self.assertTrue(
                    torch.allclose(output.grammar_mask, full.grammar_mask[:, step], atol=0.0, rtol=0.0)
                )
                self.assertTrue(
                    torch.allclose(output.state_prior_bias, full.state_prior_bias[:, step], atol=1e-6, rtol=1e-6)
                )
                self.assertTrue(
                    torch.allclose(output.ln_close_event_bias, full.ln_close_event_bias[:, step], atol=1e-6, rtol=1e-6)
                )
                self.assertTrue(
                    torch.allclose(
                        output.ln_close_time_shift_bias,
                        full.ln_close_time_shift_bias[:, step],
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )
                decode_state = output.decode_state
                self.assertEqual(decode_state.sequence_length, step + 1)
                self.assertIsNotNone(output.global_attention_gates)

    def test_incremental_decode_accepts_prefix_replayed_state_without_future_targets(self) -> None:
        torch.manual_seed(20260625)
        vocab = MapperV3Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(100, _actions(LaneAction.HOLD_START)),
                MapperTimepoint(400, _actions(LaneAction.HOLD_END, LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=400,
        )
        model = MapperV3Model(_small_config(), vocab=vocab)
        model.eval()
        decode_state = model.create_empty_decode_state(batch_size=1, device=torch.device("cpu"))
        prefix_state = initial_replay_state(tokenized.ln_carry_in)
        projected_control = torch.zeros((1, 400, 16), dtype=torch.float32)
        density_teacher = torch.zeros((1, 400, 1), dtype=torch.float32)
        carry_in = _batched_carry(tokenized.ln_carry_in)
        carry_out = _batched_carry(tokenized.ln_carry_out)

        with torch.no_grad():
            for step, (decoder_input_id, target_id) in enumerate(
                zip(tokenized.decoder_input_ids, tokenized.target_fragment_ids, strict=True),
            ):
                output = model.incremental_decode_next_token(
                    decode_state=decode_state,
                    decoder_input_token=torch.tensor([decoder_input_id], dtype=torch.long),
                    current_ms=torch.tensor([prefix_state.current_ms], dtype=torch.long),
                    open_mask=torch.tensor([prefix_state.open_mask], dtype=torch.bool),
                    open_start_ms=torch.tensor([
                        [-1 if value is None else int(value) for value in prefix_state.open_start_ms]
                    ], dtype=torch.long),
                    open_age_ms=torch.tensor([prefix_state.open_age_ms], dtype=torch.long),
                    write_start_ms=torch.tensor([tokenized.write_start_ms], dtype=torch.long),
                    write_end_ms=torch.tensor([tokenized.write_end_ms], dtype=torch.long),
                    chart_end_ms=torch.tensor([tokenized.chart_end_ms], dtype=torch.long),
                    is_full_chart_start=torch.tensor([tokenized.is_full_chart_start], dtype=torch.bool),
                    is_full_chart_end=torch.tensor([tokenized.is_full_chart_end], dtype=torch.bool),
                    ln_carry_in=carry_in,
                    ln_carry_out=carry_out,
                    density_teacher_8s=density_teacher,
                    projected_control_memory_8s=projected_control,
                    normalized_difficulty=torch.tensor([[0.1]], dtype=torch.float32),
                )

                self.assertEqual(int(output.position.item()), step)
                self.assertTrue(torch.isfinite(output.logits_final[0, target_id]).item())
                self.assertEqual(float(output.grammar_mask[0, target_id].item()), 0.0)
                decode_state = output.decode_state
                prefix_state = transition_replay_state(
                    prefix_state,
                    int(target_id),
                    position=step,
                    vocab=vocab,
                    write_start_ms=tokenized.write_start_ms,
                    write_end_ms=tokenized.write_end_ms,
                    chart_end_ms=tokenized.chart_end_ms,
                    ln_carry_out=tokenized.ln_carry_out,
                    is_full_chart_start=tokenized.is_full_chart_start,
                    is_full_chart_end=tokenized.is_full_chart_end,
                )


if __name__ == "__main__":
    unittest.main()
