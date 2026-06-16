from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects
from pulsefield_model.data.control_windows import FRAME_HOP_MS, normalize_difficulty
from pulsefield_model.data.mapper_tuple_windows import (
    MapperTupleWindowDataset,
    collate_mapper_tuple_windows,
    control_teacher_cache_key,
    extract_mapper_density_8s,
    is_mapper_tuple_window_start_allowed,
    load_control_teacher_cache_entry,
)
from pulsefield_model.models.mapper.shared.tokenizer import (
    UnsupportedMapperActionError as MapperTupleUnsupportedMapperActionError,
)
from pulsefield_model.models.mapper.v3.replay import ln_carry_state_tensors
from pulsefield_model.models.mapper.v3.tokenizer import (
    MAPPER_WRITE_MS,
    TokenizedMapperWindow,
    UnsupportedMapperActionError as MapperV3UnsupportedMapperActionError,
    encode_mapper_window,
    hitobjects_to_mapper_timepoints,
    mapper_chart_end_ms,
)
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


MAPPER_V3_RECORD_CACHE_SCHEMA_VERSION = 1
MAPPER_V3_TOKENIZER_CACHE_VERSION = 1
MAPPER_WRITE_FRAMES = MAPPER_WRITE_MS // FRAME_HOP_MS


class MapperV3WindowDataset(MapperTupleWindowDataset):
    """Mapper v3 event-token 8s window dataset.

    The source control/full-song context contract mirrors v2.1, but target
    fragments use local event-group tokens and do not carry sparse lane replay
    state.
    """

    def __init__(self, *args: Any, vocab: MapperV3Vocab | None = None, **kwargs: Any) -> None:
        resolved_vocab = MapperV3Vocab() if vocab is None else vocab
        if not isinstance(resolved_vocab, MapperV3Vocab):
            raise TypeError(f"vocab must be a MapperV3Vocab, got {type(resolved_vocab).__name__}")
        super().__init__(*args, vocab=resolved_vocab, **kwargs)

    def _record_cache_validity_metadata(self) -> dict[str, Any]:
        metadata = super()._record_cache_validity_metadata()
        metadata.update(
            {
                "schema_version": MAPPER_V3_RECORD_CACHE_SCHEMA_VERSION,
                "tokenizer_cache_version": MAPPER_V3_TOKENIZER_CACHE_VERSION,
                "mapper_token_contract": "v3_event_groups",
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
            raise FileNotFoundError(f"missing mapper v3 control teacher cache entry: {cache_path}")
        elif self.require_control_teacher_cache:
            raise ValueError("require_control_teacher_cache=True requires control_teacher_cache_dir")

        metadata = {
            "beatmap_path": record.beatmap_path.as_posix(),
            "audio_path": record.audio_path.as_posix(),
            "difficulty": record.difficulty,
            "source_frame_count": record.frame_count,
            "inference_frame_count": mapper_v3_padded_frame_count(record),
            "target_start_frame": record.target_start_frame,
            "target_start_ms": record.target_start_ms,
            "chart_end_ms": tokenized.chart_end_ms,
            "control_record_index": mapper_record.control_record_index,
            "mapper_token_contract": "v3_event_groups",
        }
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

    def _tokenize_record(self, record: Any) -> TokenizedMapperWindow:
        write_start_ms = int(record.target_start_ms)
        write_end_ms = write_start_ms + MAPPER_WRITE_MS
        try:
            timepoints = self._load_timepoints(record.beatmap_path)
            chart_end_ms = mapper_chart_end_ms(timepoints)
            if chart_end_ms < write_start_ms:
                raise MapperTupleUnsupportedMapperActionError(
                    f"mapper v3 write window starts after chart_end_ms: {write_start_ms} > {chart_end_ms}",
                )
            return encode_mapper_window(
                timepoints,
                vocab=self.vocab,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                chart_start_ms=0,
                chart_end_ms=chart_end_ms,
            )
        except MapperV3UnsupportedMapperActionError as exc:
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


def collate_mapper_v3_windows(samples: Sequence[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    return collate_mapper_tuple_windows(samples, pad_id=pad_id)


def is_mapper_v3_window_start_allowed(
    record: Any,
    *,
    mapper_stride_frames: int = MAPPER_WRITE_FRAMES,
) -> bool:
    return is_mapper_tuple_window_start_allowed(record, mapper_stride_frames=mapper_stride_frames)


def mapper_v3_padded_frame_count(record: Any) -> int:
    return max(int(record.frame_count), int(record.target_start_frame) + MAPPER_WRITE_FRAMES)
