from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects
from pulsefield_model.data.control_windows import FRAME_HOP_MS, normalize_difficulty
from pulsefield_model.data.mapper_tuple_windows import (
    MapperTupleWindowDataset,
    MapperTupleWindowRecord,
    collate_mapper_tuple_windows,
    control_teacher_cache_key,
    extract_mapper_density_8s,
    is_mapper_tuple_window_start_allowed,
    load_control_teacher_cache_entry,
)
from pulsefield_model.models.mapper.v2_1.replay import NO_EMITTED_LANE_INDEX, ln_carry_state_tensors
from pulsefield_model.models.mapper.v2_1.tokenizer import (
    MAPPER_WRITE_MS,
    TokenizedMapperWindow,
    UnsupportedMapperActionError as MapperV21UnsupportedMapperActionError,
    encode_mapper_window,
    hitobjects_to_mapper_timepoints,
    mapper_chart_end_ms,
)
from pulsefield_model.models.mapper.shared.tokenizer import UnsupportedMapperActionError as MapperTupleUnsupportedMapperActionError
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab


MAPPER_V21_RECORD_CACHE_SCHEMA_VERSION = 2
MAPPER_V21_TOKENIZER_CACHE_VERSION = 2
C3_SIDE_STREAM_METADATA_CONTRACT = "r0_delta_main_plus_c3_fallback_side_stream_v1"
C3_SIDE_STREAM_TOKEN_PAD_ID = 0
DEFAULT_C3_SIDE_STREAM_MAX_TOKENS = 256
MAPPER_WRITE_FRAMES = MAPPER_WRITE_MS // FRAME_HOP_MS
MAPPER_CONTEXT_FRAMES = MAPPER_WRITE_FRAMES


class MapperV21WindowDataset(MapperTupleWindowDataset):
    """Mapper V2.1 sparse-token 8s window dataset.

    The source control windows, full-song context, and optional control-teacher
    cache layout are shared with Mapper tuple/V2. The tokenization contract is not:
    v2.1 uses sparse lane tokens and carries same-time lane replay state.
    """

    def __init__(
        self,
        *args: Any,
        vocab: MapperV21Vocab | None = None,
        include_c3_side_stream_metadata: bool = False,
        c3_side_stream_metadata_by_beatmap_path: Mapping[str, Mapping[str, Any]] | None = None,
        include_c3_side_stream_token_tensors: bool = False,
        c3_side_stream_token_ids_by_beatmap_path: Mapping[str, Mapping[int | str, Sequence[int]]] | None = None,
        c3_side_stream_max_tokens: int = DEFAULT_C3_SIDE_STREAM_MAX_TOKENS,
        c3_side_stream_contract: str = C3_SIDE_STREAM_METADATA_CONTRACT,
        **kwargs: Any,
    ) -> None:
        resolved_vocab = MapperV21Vocab() if vocab is None else vocab
        if not isinstance(resolved_vocab, MapperV21Vocab):
            raise TypeError(f"vocab must be a MapperV21Vocab, got {type(resolved_vocab).__name__}")
        self.include_c3_side_stream_metadata = bool(include_c3_side_stream_metadata)
        self.c3_side_stream_contract = str(c3_side_stream_contract)
        self.c3_side_stream_metadata_by_beatmap_path = {
            str(key): dict(value)
            for key, value in (c3_side_stream_metadata_by_beatmap_path or {}).items()
        }
        self.include_c3_side_stream_token_tensors = bool(include_c3_side_stream_token_tensors)
        self.c3_side_stream_token_ids_by_beatmap_path = _normalize_c3_side_stream_token_lookup(
            c3_side_stream_token_ids_by_beatmap_path or {},
        )
        self.c3_side_stream_max_tokens = _validate_c3_side_stream_max_tokens(c3_side_stream_max_tokens)
        super().__init__(*args, vocab=resolved_vocab, **kwargs)

    def _record_cache_validity_metadata(self) -> dict[str, Any]:
        metadata = super()._record_cache_validity_metadata()
        metadata.update(
            {
                "schema_version": MAPPER_V21_RECORD_CACHE_SCHEMA_VERSION,
                "tokenizer_cache_version": MAPPER_V21_TOKENIZER_CACHE_VERSION,
                "mapper_token_contract": "v2.1_sparse_lane_actions",
                "vocab_size": int(self.vocab.size),
                "time_shift_values_ms": list(self.vocab.time_shift_values_ms),
            }
        )
        return metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        mapper_record = self.records[index]
        record = mapper_record.control_record
        tokenized = self._tokenize_record(record)
        cache_path = self.control_teacher_cache_path(record)
        cache_entry = None
        if cache_path is not None and cache_path.exists():
            cache_entry = load_control_teacher_cache_entry(cache_path, record=record)
        elif self.require_control_teacher_cache and cache_path is not None:
            raise FileNotFoundError(f"missing mapper v2.1 control teacher cache entry: {cache_path}")
        elif self.require_control_teacher_cache:
            raise ValueError("require_control_teacher_cache=True requires control_teacher_cache_dir")

        metadata = {
            "beatmap_path": record.beatmap_path.as_posix(),
            "audio_path": record.audio_path.as_posix(),
            "difficulty": record.difficulty,
            "source_frame_count": record.frame_count,
            "inference_frame_count": mapper_v2_1_padded_frame_count(record),
            "target_start_frame": record.target_start_frame,
            "target_start_ms": record.target_start_ms,
            "chart_end_ms": tokenized.chart_end_ms,
            "control_record_index": mapper_record.control_record_index,
            "mapper_token_contract": "v2.1_sparse_lane_actions",
        }
        if self.include_c3_side_stream_metadata:
            metadata["c3_side_stream"] = self._c3_side_stream_metadata(record)
        if self.include_c3_side_stream_token_tensors:
            metadata["c3_side_stream_tokens"] = self._c3_side_stream_token_metadata(record)
        if cache_path is not None:
            metadata["control_teacher_cache_key"] = control_teacher_cache_key(record)
            metadata["control_teacher_cache_path"] = cache_path.as_posix()
            metadata["control_teacher_cache_hit"] = cache_entry is not None

        sample: dict[str, Any] = {
            "difficulty": torch.tensor([record.difficulty], dtype=torch.float32),
            "normalized_difficulty": torch.tensor([normalize_difficulty(record.difficulty)], dtype=torch.float32),
            "decoder_input_tokens": tokenized.decoder_input_tensor(),
            "target_fragment_tokens": tokenized.target_fragment_tensor(),
            "target_fragment_states": {
                "current_ms": tokenized.target_fragment_current_ms,
                "open_mask": tokenized.target_fragment_open_mask,
                "open_start_ms": tokenized.target_fragment_open_start_ms,
                "open_age_ms": tokenized.target_fragment_open_age_ms,
                "emitted_lane_mask": tokenized.target_fragment_emitted_lane_mask,
                "last_lane_index": tokenized.target_fragment_last_lane_index,
            },
            "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
            "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
            "close_labels": tokenized.close_labels,
            "close_label_mask": tokenized.close_label_mask,
            "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
            "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
            "chart_end_ms": torch.tensor(tokenized.chart_end_ms, dtype=torch.long),
            "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
            "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
            "metadata": metadata,
        }
        if self.include_c3_side_stream_token_tensors:
            sample.update(self._c3_side_stream_token_fields(record))
        density_target_8s, density_confidence_8s = extract_mapper_density_8s(
            self._load_control_v3_target_8s(record),
        )
        if cache_entry is not None:
            sample["control_memory_8s"] = cache_entry["control_memory_8s"]
            sample["density_teacher_8s"] = cache_entry["density_teacher_8s"]
            sample["density_target_8s"] = density_target_8s
            sample["density_confidence_8s"] = density_confidence_8s
            if self.include_full_song_context:
                sample.update(self._load_full_song_context_fields(mapper_record, record))
            return sample

        sample.update(self._load_full_song_context_fields(mapper_record, record))
        sample["density_target_8s"] = density_target_8s
        sample["density_confidence_8s"] = density_confidence_8s
        return sample

    def _c3_side_stream_metadata(self, record: Any) -> dict[str, Any]:
        beatmap_path = record.beatmap_path.as_posix()
        summary = self.c3_side_stream_metadata_by_beatmap_path.get(beatmap_path)
        return {
            "enabled": True,
            "contract": self.c3_side_stream_contract,
            "beatmap_path": beatmap_path,
            "window_start_ms": int(record.target_start_ms),
            "window_end_ms": int(record.target_start_ms) + MAPPER_WRITE_MS,
            "summary_available": summary is not None,
            "summary": summary,
        }

    def _c3_side_stream_token_metadata(self, record: Any) -> dict[str, Any]:
        token_ids = self._c3_side_stream_token_ids(record)
        clipped_count = 0 if token_ids is None else min(len(token_ids), self.c3_side_stream_max_tokens)
        return {
            "enabled": True,
            "contract": self.c3_side_stream_contract,
            "beatmap_path": record.beatmap_path.as_posix(),
            "window_start_ms": int(record.target_start_ms),
            "window_end_ms": int(record.target_start_ms) + MAPPER_WRITE_MS,
            "available": token_ids is not None,
            "token_count": 0 if token_ids is None else len(token_ids),
            "clipped_token_count": 0 if token_ids is None else clipped_count,
            "truncated": token_ids is not None and len(token_ids) > self.c3_side_stream_max_tokens,
            "pad_id": C3_SIDE_STREAM_TOKEN_PAD_ID,
            "max_tokens": self.c3_side_stream_max_tokens,
        }

    def _c3_side_stream_token_fields(self, record: Any) -> dict[str, torch.Tensor]:
        token_ids = self._c3_side_stream_token_ids(record)
        available = token_ids is not None
        original_count = 0 if token_ids is None else len(token_ids)
        clipped = () if token_ids is None else token_ids[: self.c3_side_stream_max_tokens]
        truncated = original_count > len(clipped)
        return {
            "c3_side_stream_tokens": torch.tensor(clipped, dtype=torch.long),
            "c3_side_stream_token_mask": torch.ones(len(clipped), dtype=torch.bool),
            "c3_side_stream_available": torch.tensor(available, dtype=torch.bool),
            "c3_side_stream_token_count": torch.tensor(original_count, dtype=torch.long),
            "c3_side_stream_truncated": torch.tensor(truncated, dtype=torch.bool),
        }

    def _c3_side_stream_token_ids(self, record: Any) -> tuple[int, ...] | None:
        beatmap_path = record.beatmap_path.as_posix()
        windows = self.c3_side_stream_token_ids_by_beatmap_path.get(beatmap_path)
        if windows is None:
            return None
        return windows.get(int(record.target_start_ms))

    def _tokenize_record(self, record: Any) -> TokenizedMapperWindow:
        write_start_ms = int(record.target_start_ms)
        write_end_ms = write_start_ms + MAPPER_WRITE_MS
        try:
            timepoints = self._load_timepoints(record.beatmap_path)
            chart_end_ms = mapper_chart_end_ms(timepoints)
            if chart_end_ms < write_start_ms:
                raise MapperTupleUnsupportedMapperActionError(
                    f"mapper v2.1 write window starts after chart_end_ms: {write_start_ms} > {chart_end_ms}",
                )
            return encode_mapper_window(
                timepoints,
                vocab=self.vocab,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                chart_start_ms=0,
                chart_end_ms=chart_end_ms,
            )
        except MapperV21UnsupportedMapperActionError as exc:
            raise MapperTupleUnsupportedMapperActionError(str(exc)) from exc

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        key = beatmap_path.as_posix()
        cached = self._timepoints_by_beatmap.get(key)
        if cached is not None:
            self._timepoints_by_beatmap.move_to_end(key)
            return cached
        cached = tuple(hitobjects_to_mapper_timepoints(parse_mania_hit_objects(beatmap_path, expected_key_count=4)))
        if self.max_cached_timepoint_maps > 0:
            self._timepoints_by_beatmap[key] = cached
            while len(self._timepoints_by_beatmap) > self.max_cached_timepoint_maps:
                self._timepoints_by_beatmap.popitem(last=False)
        return cached


def collate_mapper_v2_1_windows(samples: Sequence[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_mapper_v2_1_windows requires at least one sample")
    batch = collate_mapper_tuple_windows(samples, pad_id=pad_id)
    batch_size, max_seq_len = batch["target_fragment_tokens"].shape
    emitted_lane_mask = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)
    last_lane_index = torch.full((batch_size, max_seq_len), NO_EMITTED_LANE_INDEX, dtype=torch.long)

    for batch_index, sample in enumerate(samples):
        length = int(sample["target_fragment_tokens"].shape[0])
        states = sample["target_fragment_states"]
        if "emitted_lane_mask" not in states or "last_lane_index" not in states:
            raise ValueError("mapper v2.1 samples must include emitted_lane_mask and last_lane_index states")
        emitted = states["emitted_lane_mask"].to(dtype=torch.bool)
        last = states["last_lane_index"].to(dtype=torch.long)
        if tuple(emitted.shape) != (length, 4):
            raise ValueError(f"sample {batch_index} emitted_lane_mask must have shape {(length, 4)}")
        if tuple(last.shape) != (length,):
            raise ValueError(f"sample {batch_index} last_lane_index must have shape {(length,)}")
        emitted_lane_mask[batch_index, :length] = emitted
        last_lane_index[batch_index, :length] = last

    batch["target_fragment_states"] = dict(batch["target_fragment_states"])
    batch["target_fragment_states"]["emitted_lane_mask"] = emitted_lane_mask
    batch["target_fragment_states"]["last_lane_index"] = last_lane_index
    if any("c3_side_stream_tokens" in sample for sample in samples):
        batch.update(_collate_c3_side_stream_token_fields(samples))
    return batch


def is_mapper_v2_1_window_start_allowed(
    record: Any,
    *,
    mapper_stride_frames: int = MAPPER_WRITE_FRAMES,
) -> bool:
    return is_mapper_tuple_window_start_allowed(record, mapper_stride_frames=mapper_stride_frames)


def mapper_v2_1_padded_frame_count(record: Any) -> int:
    return max(int(record.frame_count), int(record.target_start_frame) + MAPPER_WRITE_FRAMES)


def load_c3_side_stream_token_sidecar(
    sidecar_path: str | Path,
) -> dict[str, dict[int, tuple[int, ...]]]:
    """Load a mapper-window C3 side-stream token sidecar.

    Expected JSON schema:

    {
      "schema_version": 1,
      "contract": "r0_delta_main_plus_c3_fallback_side_stream_v1",
      "windows": [
        {"beatmap_path": "map.osu", "window_start_ms": 0, "token_ids": [1, 2, 3]}
      ]
    }
    """

    path = Path(sidecar_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid C3 side-stream token sidecar JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("C3 side-stream token sidecar must be a JSON object")
    windows = payload.get("windows")
    if not isinstance(windows, list):
        raise ValueError("C3 side-stream token sidecar must contain a windows list")
    lookup: dict[str, dict[int, tuple[int, ...]]] = {}
    for row_number, row in enumerate(windows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"C3 side-stream sidecar row {row_number} must be an object")
        if "beatmap_path" not in row:
            raise ValueError(f"C3 side-stream sidecar row {row_number} missing beatmap_path")
        if "window_start_ms" not in row:
            raise ValueError(f"C3 side-stream sidecar row {row_number} missing window_start_ms")
        if "token_ids" not in row:
            raise ValueError(f"C3 side-stream sidecar row {row_number} missing token_ids")
        beatmap_path = str(row["beatmap_path"])
        window_start_ms = _normalize_c3_side_stream_window_key(row["window_start_ms"])
        token_ids = _normalize_c3_side_stream_token_ids(row["token_ids"], context=f"sidecar row {row_number}")
        by_window = lookup.setdefault(beatmap_path, {})
        if window_start_ms in by_window:
            raise ValueError(
                "duplicate C3 side-stream sidecar window "
                f"for beatmap_path={beatmap_path!r} window_start_ms={window_start_ms}",
            )
        by_window[window_start_ms] = token_ids
    return lookup


def _collate_c3_side_stream_token_fields(samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    token_sequences: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    available: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    truncated: list[torch.Tensor] = []
    max_len = 0
    for sample_index, sample in enumerate(samples):
        tokens = sample.get("c3_side_stream_tokens")
        if tokens is None:
            tokens = torch.empty(0, dtype=torch.long)
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(f"sample {sample_index} c3_side_stream_tokens must be a torch.Tensor")
        tokens = tokens.to(dtype=torch.long).reshape(-1)
        mask = sample.get("c3_side_stream_token_mask")
        if mask is None:
            mask = torch.ones(int(tokens.shape[0]), dtype=torch.bool)
        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"sample {sample_index} c3_side_stream_token_mask must be a torch.Tensor")
        mask = mask.to(dtype=torch.bool).reshape(-1)
        if int(mask.shape[0]) != int(tokens.shape[0]):
            raise ValueError(f"sample {sample_index} c3_side_stream_token_mask must match token length")
        token_sequences.append(tokens)
        masks.append(mask)
        available.append(_sample_bool_tensor(sample, "c3_side_stream_available"))
        token_counts.append(_sample_long_tensor(sample, "c3_side_stream_token_count"))
        truncated.append(_sample_bool_tensor(sample, "c3_side_stream_truncated"))
        max_len = max(max_len, int(tokens.shape[0]))

    batch_size = len(samples)
    padded = torch.full((batch_size, max_len), C3_SIDE_STREAM_TOKEN_PAD_ID, dtype=torch.long)
    padded_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for index, tokens in enumerate(token_sequences):
        length = int(tokens.shape[0])
        if length:
            padded[index, :length] = tokens
            padded_mask[index, :length] = masks[index]
    return {
        "c3_side_stream_tokens": padded,
        "c3_side_stream_token_mask": padded_mask,
        "c3_side_stream_available": torch.stack(available).reshape(batch_size),
        "c3_side_stream_token_count": torch.stack(token_counts).reshape(batch_size),
        "c3_side_stream_truncated": torch.stack(truncated).reshape(batch_size),
    }


def _sample_bool_tensor(sample: Mapping[str, Any], key: str) -> torch.Tensor:
    value = sample.get(key)
    if value is None:
        return torch.tensor(False, dtype=torch.bool)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a torch.Tensor")
    return value.to(dtype=torch.bool).reshape(())


def _sample_long_tensor(sample: Mapping[str, Any], key: str) -> torch.Tensor:
    value = sample.get(key)
    if value is None:
        return torch.tensor(0, dtype=torch.long)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be a torch.Tensor")
    return value.to(dtype=torch.long).reshape(())


def _normalize_c3_side_stream_token_lookup(
    lookup: Mapping[str, Mapping[int | str, Sequence[int]]],
) -> dict[str, dict[int, tuple[int, ...]]]:
    normalized: dict[str, dict[int, tuple[int, ...]]] = {}
    for beatmap_path, windows in lookup.items():
        if not isinstance(windows, Mapping):
            raise ValueError(f"C3 side-stream windows for {beatmap_path!r} must be a mapping")
        normalized_windows: dict[int, tuple[int, ...]] = {}
        for window_start_ms, token_ids in windows.items():
            normalized_windows[_normalize_c3_side_stream_window_key(window_start_ms)] = (
                _normalize_c3_side_stream_token_ids(
                    token_ids,
                    context=f"beatmap_path={beatmap_path!r} window_start_ms={window_start_ms!r}",
                )
            )
        normalized[str(beatmap_path)] = normalized_windows
    return normalized


def _normalize_c3_side_stream_window_key(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"C3 side-stream window_start_ms must be an integer, got {value!r}")
    try:
        window_start_ms = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"C3 side-stream window_start_ms must be an integer, got {value!r}") from exc
    if window_start_ms < 0:
        raise ValueError(f"C3 side-stream window_start_ms must be non-negative: {window_start_ms}")
    return window_start_ms


def _normalize_c3_side_stream_token_ids(token_ids: object, *, context: str) -> tuple[int, ...]:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError(f"C3 side-stream token_ids must be a sequence for {context}")
    normalized: list[int] = []
    for index, value in enumerate(token_ids):
        if isinstance(value, bool):
            raise ValueError(f"C3 side-stream token id {index} must be a positive integer for {context}")
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"C3 side-stream token id {index} must be a positive integer for {context}") from exc
        if token_id <= C3_SIDE_STREAM_TOKEN_PAD_ID:
            raise ValueError(
                f"C3 side-stream token id {index} must be > pad id {C3_SIDE_STREAM_TOKEN_PAD_ID} for {context}",
            )
        normalized.append(token_id)
    return tuple(normalized)


def _validate_c3_side_stream_max_tokens(value: int) -> int:
    max_tokens = int(value)
    if max_tokens <= 0:
        raise ValueError(f"c3_side_stream_max_tokens must be positive: {value}")
    return max_tokens
