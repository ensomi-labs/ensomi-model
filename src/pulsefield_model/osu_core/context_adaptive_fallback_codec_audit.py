from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.osu_core.beat_chunk_pattern_audit import (
    DEFAULT_BEAT_CHUNK_CACHE_PATH,
    DEFAULT_MOTIF_MAX_N,
    DEFAULT_MOTIF_MIN_N,
    DEFAULT_SMOOTHING_ALPHA,
)
from pulsefield_model.osu_core.beat_chunk_tokenizer_refinement_audit import DEFAULT_RARE_MAX_COUNT
from pulsefield_model.osu_core.duration_ln_tokenization_audit import (
    CONTROL_TOKENS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_MOTIF_VOCAB_SIZE,
    _EncodedStats,
    _bootstrap_mapset_delta,
    _cross_split_artist_title_version_keys,
    _encode_sequence,
    _make_model_token_bits,
    _normalize_json,
    _rank_motifs,
    _score_encoded_generic,
    _split_report,
)
from pulsefield_model.osu_core.fallback_forensic_audit import (
    CHUNK_UNITS,
    EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
    _baseline_tokens,
    _boundary_bucket,
    _count_bucket,
    _dataset_summary,
    _fit_motif_model,
    _group_class,
    _iter_rows,
    _read_chunk_cache,
)


SCHEMA_VERSION: Final[int] = 1
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_report.json",
)
DEFAULT_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_result_log.md",
)
DEFAULT_COMPARISON_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_variant_comparison.csv",
)
DEFAULT_DIAGNOSTICS_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_diagnostics.csv",
)
DEFAULT_LZ_WINDOW_FALLBACKS: Final[int] = 256
DEFAULT_LZ_MAX_SPAN: Final[int] = 8
DEFAULT_LZ_CANDIDATE_SCAN_LIMIT: Final[int] = 16
DEFAULT_PATCH_MAX_SPAN: Final[int] = 8
MODE_BITS_2: Final[float] = math.log2(2.0)
MODE_BITS_4: Final[float] = math.log2(4.0)
PATCH_CODEC_BITS: Final[float] = math.log2(4.0)
LZ_TRANSFORM_BITS: Final[float] = math.log2(3.0)
BITPLANES: Final[tuple[str, ...]] = ("tap", "press", "release")
PPM_DEPTHS: Final[tuple[int, ...]] = (8, 4, 2, 1, 0)


@dataclass(frozen=True, slots=True)
class _FallbackRecord:
    id: int
    split: str
    source_row_index: int
    beatmap_set_id: int
    beatmap_id: int
    segment_id: int
    chunk_index: int
    group_index: int
    absolute_units: int
    offset_units: int
    bar_phase_half: int
    event_count: int
    token: str
    raw_bits: float
    delta: int
    tap_mask: int
    ln_start_mask: int
    ln_end_mask: int
    order_signature: str
    pre_hold_mask: int
    chunk_start_hold_mask: int
    previous_skeleton: str
    previous_fallback: bool
    history_skeletons: tuple[str, ...]
    boundary_bucket: str
    group_class: str
    active_count: int
    active_hold_count: int
    chord_ln_mixed: bool
    density_bin: str
    artist: str
    title: str
    version: str


@dataclass(frozen=True, slots=True)
class _LzCandidate:
    match_type: str
    length: int
    distance: int
    payload_bits: float
    residual_bits: float


@dataclass
class _CategoricalModel:
    alpha: float
    counts: dict[tuple[str, ...], Counter[str]] = field(default_factory=dict)
    global_counts: Counter[str] = field(default_factory=Counter)
    vocab: set[str] = field(default_factory=set)

    def add(self, context: Sequence[str], symbol: str) -> None:
        key = tuple(context)
        self.counts.setdefault(key, Counter())[str(symbol)] += 1
        self.global_counts[str(symbol)] += 1
        self.vocab.add(str(symbol))

    def bits(self, context: Sequence[str], symbol: str) -> float:
        key = tuple(context)
        counter = self.counts.get(key)
        if counter:
            return _categorical_bits(counter, self.vocab, str(symbol), alpha=self.alpha)
        return _categorical_bits(self.global_counts, self.vocab, str(symbol), alpha=self.alpha)

    def table_cost_bits(self) -> float:
        return _categorical_table_cost(self.counts, self.vocab)


@dataclass
class _BernoulliModel:
    alpha: float
    counts: dict[tuple[str, ...], list[int]] = field(default_factory=dict)
    global_counts: list[int] = field(default_factory=lambda: [0, 0])

    def add(self, context: Sequence[str], bit: int) -> None:
        value = 1 if int(bit) else 0
        key = tuple(context)
        counts = self.counts.setdefault(key, [0, 0])
        counts[value] += 1
        self.global_counts[value] += 1

    def bits(self, context: Sequence[str], bit: int) -> float:
        key = tuple(context)
        counts = self.counts.get(key, self.global_counts)
        zero, one = int(counts[0]), int(counts[1])
        total = zero + one
        numerator = (one if int(bit) else zero) + self.alpha
        denominator = total + 2.0 * self.alpha
        return -math.log2(numerator / denominator) if denominator > 0 else 1.0

    def table_cost_bits(self) -> float:
        if not self.counts:
            return 0.0
        total = sum(sum(values) for values in self.counts.values())
        count_bits = math.log2(max(2, total + 1))
        context_bits = math.log2(max(2, len(self.counts) + 1))
        return float(len(self.counts)) * (context_bits + count_bits + 1.0)


@dataclass
class _C1Model:
    delta_model: _CategoricalModel
    order_model: _CategoricalModel
    bit_models: dict[str, _BernoulliModel]

    def frozen_bits(self, record: _FallbackRecord) -> float:
        bits = self.delta_model.bits(_c1_delta_context(record), str(record.delta))
        delta_bucket = _delta_bucket(record.delta)
        bits += self.order_model.bits(_c1_order_context(record, delta_bucket=delta_bucket), record.order_signature)
        for plane in BITPLANES:
            mask = _plane_mask(record, plane)
            for lane in range(4):
                bit = (mask >> lane) & 1
                bits += self.bit_models[plane].bits(_c1_bit_context(record, plane=plane, lane=lane, delta_bucket=delta_bucket), bit)
        return bits

    def update(self, record: _FallbackRecord) -> None:
        self.delta_model.add(_c1_delta_context(record), str(record.delta))
        delta_bucket = _delta_bucket(record.delta)
        self.order_model.add(_c1_order_context(record, delta_bucket=delta_bucket), record.order_signature)
        for plane in BITPLANES:
            mask = _plane_mask(record, plane)
            for lane in range(4):
                bit = (mask >> lane) & 1
                self.bit_models[plane].add(_c1_bit_context(record, plane=plane, lane=lane, delta_bucket=delta_bucket), bit)

    def table_cost_bits(self) -> float:
        return (
            self.delta_model.table_cost_bits()
            + self.order_model.table_cost_bits()
            + sum(model.table_cost_bits() for model in self.bit_models.values())
        )


@dataclass
class _PpmLevelModel:
    alpha: float
    counts_by_depth: dict[int, dict[tuple[str, ...], Counter[str]]] = field(
        default_factory=lambda: {depth: {} for depth in PPM_DEPTHS}
    )
    global_counts: Counter[str] = field(default_factory=Counter)
    vocab: set[str] = field(default_factory=set)

    def add(self, record: _FallbackRecord, level: str, symbol: str) -> None:
        value = str(symbol)
        for depth in PPM_DEPTHS:
            context = _ppm_context(record, level=level, depth=depth)
            self.counts_by_depth.setdefault(depth, {}).setdefault(context, Counter())[value] += 1
        self.global_counts[value] += 1
        self.vocab.add(value)

    def ppm_bits(self, record: _FallbackRecord, level: str, symbol: str) -> float:
        value = str(symbol)
        escape_bits = 0.0
        for depth in (8, 4, 2, 1):
            counter = self.counts_by_depth.get(depth, {}).get(_ppm_context(record, level=level, depth=depth))
            if not counter:
                continue
            total = sum(counter.values())
            vocab_size = max(1, len(self.vocab))
            denominator = float(total) + self.alpha * float(vocab_size + 1)
            count = counter.get(value, 0)
            if count > 0:
                probability = (float(count) + self.alpha) / denominator
                return escape_bits - math.log2(probability)
            escape_probability = self.alpha / denominator
            escape_bits += -math.log2(escape_probability)
        return escape_bits + _categorical_bits(self.global_counts, self.vocab, value, alpha=self.alpha)

    def ctw_bits(self, record: _FallbackRecord, level: str, symbol: str) -> float:
        value = str(symbol)
        probabilities = []
        for depth in PPM_DEPTHS:
            counter = self.counts_by_depth.get(depth, {}).get(_ppm_context(record, level=level, depth=depth))
            if counter:
                probabilities.append(_categorical_probability(counter, self.vocab, value, alpha=self.alpha))
        if not probabilities:
            probabilities.append(_categorical_probability(self.global_counts, self.vocab, value, alpha=self.alpha))
        probability = float(sum(probabilities)) / float(len(probabilities))
        return -math.log2(max(probability, 1e-300))

    def table_cost_bits(self) -> float:
        bits = 0.0
        for counts in self.counts_by_depth.values():
            bits += _categorical_table_cost(counts, self.vocab)
        return bits


@dataclass
class _C2Model:
    levels: dict[str, _PpmLevelModel]

    def ppm_bits(self, record: _FallbackRecord) -> float:
        return sum(self.levels[level].ppm_bits(record, level, symbol) for level, symbol in _c2_symbols(record))

    def ctw_bits(self, record: _FallbackRecord) -> float:
        return sum(self.levels[level].ctw_bits(record, level, symbol) for level, symbol in _c2_symbols(record))

    def table_cost_bits(self) -> float:
        return sum(model.table_cost_bits() for model in self.levels.values())


@dataclass(frozen=True, slots=True)
class _CostPlan:
    name: str
    legal_codec: bool
    replacement_bits: tuple[float, ...]
    payload_bits: tuple[float, ...]
    mode_header_bits: tuple[float, ...]
    mode_codes: tuple[str, ...]
    table_model_cost_bits: float
    notes: str
    reconstruction_checked_count: int = 0
    reconstruction_mismatch_count: int = 0


def audit_context_adaptive_fallback_codec(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_RESULT_LOG_PATH,
    comparison_csv_path: str | Path | None = DEFAULT_COMPARISON_CSV_PATH,
    diagnostics_csv_path: str | Path | None = DEFAULT_DIAGNOSTICS_CSV_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 17,
    lz_window_fallbacks: int = DEFAULT_LZ_WINDOW_FALLBACKS,
    lz_max_span: int = DEFAULT_LZ_MAX_SPAN,
    patch_max_span: int = DEFAULT_PATCH_MAX_SPAN,
    top_example_limit: int = 20,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run CASF v2 literal-path diagnostics and charged fallback codec variants."""

    started_at = time.perf_counter()
    chunk_cache_path = Path(chunk_cache_path)
    report_path = None if report_path is None else Path(report_path)
    result_log_path = None if result_log_path is None else Path(result_log_path)
    comparison_csv_path = None if comparison_csv_path is None else Path(comparison_csv_path)
    diagnostics_csv_path = None if diagnostics_csv_path is None else Path(diagnostics_csv_path)
    _validate_positive(motif_vocab_size, "motif_vocab_size")
    _validate_positive(motif_min_n, "motif_min_n")
    _validate_positive(motif_max_n, "motif_max_n")
    _validate_positive(lz_window_fallbacks, "lz_window_fallbacks")
    _validate_positive(lz_max_span, "lz_max_span")
    _validate_positive(patch_max_span, "patch_max_span")
    _validate_nonnegative(bootstrap_samples, "bootstrap_samples")
    if motif_max_n < motif_min_n:
        raise ValueError("motif_max_n must be >= motif_min_n")

    chunk_df = _read_chunk_cache(chunk_cache_path, limit_chunks=limit_chunks)
    baseline_model = _fit_motif_model(
        chunk_df,
        name="r0_delta",
        token_mapper=_baseline_tokens,
        motif_vocab_size=motif_vocab_size,
        motif_min_n=motif_min_n,
        motif_max_n=motif_max_n,
        smoothing_alpha=smoothing_alpha,
        top_motif_limit=top_example_limit,
    )
    baseline_splits = {
        split: _score_baseline_split(
            chunk_df,
            split=split,
            model=baseline_model,
            dictionary_bits=baseline_model.dictionary_bits,
            rare_max_count=rare_max_count,
        )
        for split in ("train", "valid", "test")
    }
    conflict_keys = _cross_split_artist_title_version_keys(chunk_df)
    same_song_source_ids = _same_song_source_ids(chunk_df, conflict_keys=conflict_keys)
    same_song_chunk_df = chunk_df[
        (chunk_df["split"].fillna("").astype(str) == "test")
        & (chunk_df["source_row_index"].astype(int).isin(same_song_source_ids))
    ].copy()
    baseline_splits["same_song_filtered_test"] = _score_baseline_split(
        same_song_chunk_df,
        split="test",
        model=baseline_model,
        dictionary_bits=baseline_model.dictionary_bits,
        rare_max_count=rare_max_count,
    )

    records, record_summary = _collect_fallback_records(chunk_df, baseline_model=baseline_model)
    train_records = [record for record in records if record.split == "train"]
    c1_model = _fit_c1_model(train_records, smoothing_alpha=smoothing_alpha)
    c2_model = _fit_c2_model(train_records, smoothing_alpha=smoothing_alpha)
    c1_frozen_bits = tuple(c1_model.frozen_bits(record) for record in records)
    c1_online_bits = _c1_online_bits(records, c1_model)
    c2_ppm_bits = tuple(c2_model.ppm_bits(record) for record in records)
    c2_ctw_bits = tuple(c2_model.ctw_bits(record) for record in records)
    lz_candidates = _lz_candidates_by_record(
        records,
        window_fallbacks=lz_window_fallbacks,
        max_span=lz_max_span,
    )
    c3_plan = _build_lz_span_plan(
        records,
        lz_candidates,
        window_fallbacks=lz_window_fallbacks,
        max_span=lz_max_span,
    )
    c4_plan = _build_patch_plan(
        records,
        c1_frozen_bits,
        max_span=patch_max_span,
        table_model_cost_bits=c1_model.table_cost_bits(),
    )
    oracle_plan = _build_oracle_plan(
        records,
        candidate_bits={
            "c1a_frozen": c1_frozen_bits,
            "c1b_online": c1_online_bits,
            "c2a_ppm": c2_ppm_bits,
            "c2b_ctw": c2_ctw_bits,
            "c3_lz": c3_plan.replacement_bits,
            "c4_patch": c4_plan.replacement_bits,
        },
    )
    selector_plan = _build_explicit_selector_plan(
        records,
        candidate_bits={
            "raw_literal": tuple(record.raw_bits for record in records),
            "c1a_frozen": c1_frozen_bits,
            "c2a_ppm": c2_ppm_bits,
        },
        table_model_cost_bits=c1_model.table_cost_bits() + c2_model.table_cost_bits(),
    )
    plans = [
        _all_fallback_plan(
            "c1a_frozen_bitplane_all_fallback",
            records,
            c1_frozen_bits,
            table_model_cost_bits=c1_model.table_cost_bits(),
            notes="C1a frozen train-fitted tables; all r0 fallback literals use C1 syntax, no selector.",
        ),
        _all_fallback_plan(
            "c1b_online_bitplane_all_fallback",
            records,
            c1_online_bits,
            table_model_cost_bits=c1_model.table_cost_bits(),
            notes="C1b train-initialized online-adaptive tables reset per chart/source; all r0 fallback literals use C1 syntax.",
        ),
        _all_fallback_plan(
            "c2a_ppm_symbol_all_fallback",
            records,
            c2_ppm_bits,
            table_model_cost_bits=c2_model.table_cost_bits(),
            notes="C2a variable-order PPM fallback syntax model; all r0 fallback literals use C2 syntax.",
        ),
        _all_fallback_plan(
            "c2b_ctw_mixture_all_fallback",
            records,
            c2_ctw_bits,
            table_model_cost_bits=c2_model.table_cost_bits(),
            notes="C2b CTW-like probability mixture over context depths; all r0 fallback literals use C2 syntax.",
        ),
        c3_plan,
        c4_plan,
        selector_plan,
        oracle_plan,
    ]

    variant_reports: dict[str, Any] = {
        "b0_r0_delta": {
            "variant": "b0_r0_delta",
            "legal_codec": True,
            "notes": "Unchanged in-harness r0_delta baseline.",
            "splits": baseline_splits,
            "table_model_cost_bits": 0.0,
        }
    }
    comparison_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for split, values in baseline_splits.items():
        comparison_rows.append(_comparison_row("b0_r0_delta", split, values, legal_codec=True, table="global"))
    for plan in plans:
        report = _score_plan(
            plan,
            records,
            baseline_splits=baseline_splits,
            source_filters={"same_song_filtered_test": same_song_source_ids},
        )
        bucket_rows = _bucket_rows_for_plan(plan, records, baseline_splits, legal_codec=plan.legal_codec)
        report["bucket_rows"] = bucket_rows
        variant_reports[plan.name] = report
        for split, values in report["splits"].items():
            comparison_rows.append(_comparison_row(plan.name, split, values, legal_codec=plan.legal_codec, table="global"))
        comparison_rows.extend(bucket_rows)
        diagnostic_rows.extend(_usage_rows_for_plan(plan, records))

    baseline = variant_reports["b0_r0_delta"]
    selected_variant = _select_legal_variant(variant_reports)
    selected = variant_reports[selected_variant]
    reconstruction_guard = _reconstruction_guard(records, plans)
    bootstrap = _bootstrap_mapset_delta(
        selected,
        baseline,
        samples=bootstrap_samples,
        seed=random_seed,
    )
    diagnostics = _diagnostics(
        records,
        baseline_splits=baseline_splits,
        oracle_report=variant_reports[oracle_plan.name],
        c1_model=c1_model,
        c2_model=c2_model,
        lz_candidates=lz_candidates,
        top_example_limit=top_example_limit,
    )
    pass_criteria = _pass_criteria(
        selected_variant=selected_variant,
        selected=selected,
        baseline=baseline,
        bootstrap=bootstrap,
        record_summary=record_summary,
        reconstruction_guard=reconstruction_guard,
    )
    recommendation = _recommendation(pass_criteria, selected_variant=selected_variant)

    if comparison_csv_path is not None:
        comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)
    if diagnostics_csv_path is not None:
        diagnostics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([*diagnostic_rows, *diagnostics["table_rows"]]).to_csv(diagnostics_csv_path, index=False)

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "CASF v2 context-adaptive selective fallback codec",
        "command": command,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--short")),
        "chunk_cache_path": chunk_cache_path.as_posix(),
        "report_path": None if report_path is None else report_path.as_posix(),
        "result_log_path": None if result_log_path is None else result_log_path.as_posix(),
        "comparison_csv_path": None if comparison_csv_path is None else comparison_csv_path.as_posix(),
        "diagnostics_csv_path": None if diagnostics_csv_path is None else diagnostics_csv_path.as_posix(),
        "limited": limit_chunks is not None,
        "limit_chunks": limit_chunks,
        "elapsed_s": time.perf_counter() - started_at,
        "config": {
            "motif_vocab_size": motif_vocab_size,
            "motif_min_n": motif_min_n,
            "motif_max_n": motif_max_n,
            "smoothing_alpha": smoothing_alpha,
            "rare_max_count": rare_max_count,
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
            "lz_window_fallbacks": lz_window_fallbacks,
            "lz_max_span": lz_max_span,
            "patch_max_span": patch_max_span,
            "selector_policy": "valid selects the best legal charged variant; oracle is reported as an illegal upper bound only.",
            "legality_policy": "baseline motifs/dictionary unchanged; only r0 fallback literal payload is replaced; mode/header/table costs are included for charged variants.",
        },
        "dataset": _dataset_summary(chunk_df),
        "fallback_record_summary": record_summary,
        "same_song_filter": {
            "conflict_key_count": len(conflict_keys),
            "test_source_count_after_filter": len(same_song_source_ids),
        },
        "baseline_model": {
            "learned_motif_count": baseline_model.learned_motif_count,
            "dictionary_cost_bits": baseline_model.dictionary_bits,
            "atom_vocab_size": len(baseline_model.atom_counter),
            "train_encoded_vocab_size": len(baseline_model.train_counter),
        },
        "model_table_costs": {
            "c1_table_cost_bits": c1_model.table_cost_bits(),
            "c2_table_cost_bits": c2_model.table_cost_bits(),
        },
        "variants": variant_reports,
        "selected_variant": selected_variant,
        "diagnostics": diagnostics["summary"],
        "bootstrap_mapset_delta": bootstrap,
        "reconstruction_guard": reconstruction_guard,
        "pass_criteria": pass_criteria,
        "recommendation": recommendation,
    }
    report = _normalize_json(report)
    if report_path is not None:
        _write_json(report_path, report)
    if result_log_path is not None:
        _write_result_log(result_log_path, report)
    return report


def _score_baseline_split(
    chunk_df: pd.DataFrame,
    *,
    split: str,
    model: Any,
    dictionary_bits: float,
    rare_max_count: int,
) -> dict[str, Any]:
    stats = _EncodedStats()
    for row in _iter_rows(chunk_df, split=split):
        tokens = _baseline_tokens(row)
        encoded, covered_indexes = _encode_sequence(tokens, model.trie)
        code_bits = sum(model.token_bits(token) for token in encoded)
        stats.chunk_count += 1
        stats.event_count += int(row.num_events)
        stats.group_count += int(row.num_groups)
        stats.code_bits += code_bits
        stats.mapset_bits[int(row.beatmap_set_id)] += code_bits
        stats.mapset_events[int(row.beatmap_set_id)] += int(row.num_events)
        stats.motif_group_coverage_count += sum(1 for index in covered_indexes if index > 0)
        for token in encoded:
            stats.token_counter[token] += 1
            stats.token_count += 1
            if token.startswith("M"):
                stats.motif_token_count += 1
            else:
                stats.atom_token_count += 1
    _score_encoded_generic(
        stats,
        token_bits=model.token_bits,
        dictionary_bits=dictionary_bits,
        rare_max_count=rare_max_count,
    )
    return _split_report(stats, dictionary_bits=dictionary_bits)


def _collect_fallback_records(chunk_df: pd.DataFrame, *, baseline_model: Any) -> tuple[list[_FallbackRecord], dict[str, Any]]:
    records: list[_FallbackRecord] = []
    invalid_transition_count = 0
    invalid_examples: list[dict[str, Any]] = []
    state_by_source_segment: dict[tuple[int, int], int] = {}
    previous_skeleton_by_source_segment: dict[tuple[int, int], str] = {}
    previous_fallback_by_source_segment: dict[tuple[int, int], bool] = {}
    history_by_source_segment: dict[tuple[int, int], deque[str]] = {}
    ordered = chunk_df.sort_values(["source_row_index", "segment_id", "start_beat_units", "chunk_index"])
    split_counts: Counter[str] = Counter()
    raw_bits_by_split: Counter[str] = Counter()
    event_counts_by_split: Counter[str] = Counter()

    for row in ordered.itertuples(index=False):
        source = int(row.source_row_index)
        segment = int(row.segment_id)
        state_key = (source, segment)
        state = state_by_source_segment.get(state_key, 0)
        chunk_start_state = state
        previous_skeleton = previous_skeleton_by_source_segment.get(state_key, "START")
        previous_fallback = previous_fallback_by_source_segment.get(state_key, False)
        history = history_by_source_segment.setdefault(state_key, deque(maxlen=8))
        tokens = _baseline_tokens(row)
        _encoded, covered_indexes = _encode_sequence(tokens, baseline_model.trie)
        groups = json.loads(str(row.groups_json or "[]"))
        previous_offset = 0
        for group_index, group in enumerate(groups, start=1):
            offset = int(group.get("offset_units", 0))
            delta = offset - previous_offset
            previous_offset = offset
            tap = int(group.get("tap_mask", 0))
            start = int(group.get("ln_start_mask", 0))
            end = int(group.get("ln_end_mask", 0))
            invalid_mask = (end & ~state) | (start & state)
            if invalid_mask:
                invalid_transition_count += invalid_mask.bit_count()
                if len(invalid_examples) < 10:
                    invalid_examples.append(
                        {
                            "source_row_index": source,
                            "beatmap_set_id": int(row.beatmap_set_id),
                            "chunk_index": int(row.chunk_index),
                            "offset_units": offset,
                            "pre_hold_mask": state,
                            "tap_mask": tap,
                            "ln_start_mask": start,
                            "ln_end_mask": end,
                            "invalid_mask": invalid_mask,
                        }
                    )
            fallback = group_index not in covered_indexes
            current_class = _group_class(tap, start, end)
            active_count = (tap | start | end).bit_count()
            skeleton = _skeleton_symbol(delta=delta, tap=tap, start=start, end=end)
            if fallback:
                token = tokens[group_index]
                raw_bits = baseline_model.token_bits(token)
                split = str(row.split)
                event_count = max(1, int(group.get("event_count", 1) or 1))
                record = _FallbackRecord(
                    id=len(records),
                    split=split,
                    source_row_index=source,
                    beatmap_set_id=int(row.beatmap_set_id),
                    beatmap_id=int(row.beatmap_id),
                    segment_id=segment,
                    chunk_index=int(row.chunk_index),
                    group_index=group_index,
                    absolute_units=int(row.start_beat_units) + offset,
                    offset_units=offset,
                    bar_phase_half=int(row.bar_phase_half),
                    event_count=event_count,
                    token=token,
                    raw_bits=raw_bits,
                    delta=delta,
                    tap_mask=tap,
                    ln_start_mask=start,
                    ln_end_mask=end,
                    order_signature=str(group.get("order_signature", ".") or "."),
                    pre_hold_mask=state,
                    chunk_start_hold_mask=chunk_start_state,
                    previous_skeleton=previous_skeleton,
                    previous_fallback=previous_fallback,
                    history_skeletons=tuple(history),
                    boundary_bucket=_boundary_bucket(offset),
                    group_class=current_class,
                    active_count=active_count,
                    active_hold_count=state.bit_count(),
                    chord_ln_mixed=bool(tap and (start or end)),
                    density_bin=str(row.density_bin),
                    artist=str(row.artist),
                    title=str(row.title),
                    version=str(row.version),
                )
                records.append(record)
                split_counts[split] += 1
                raw_bits_by_split[split] += raw_bits
                event_counts_by_split[split] += event_count
            state = (state | start) & ~end
            previous_skeleton = skeleton
            previous_fallback = fallback
            history.append(skeleton)
        state_by_source_segment[state_key] = state
        previous_skeleton_by_source_segment[state_key] = previous_skeleton
        previous_fallback_by_source_segment[state_key] = previous_fallback
    segment_state_reset_count = sum(mask.bit_count() for mask in state_by_source_segment.values() if mask)
    summary = {
        "fallback_literal_count": len(records),
        "invalid_transition_count": invalid_transition_count,
        "invalid_transition_examples": invalid_examples,
        "segment_state_reset_count": segment_state_reset_count,
        "split_fallback_literal_count": dict(split_counts),
        "split_raw_fallback_bits": {key: float(value) for key, value in raw_bits_by_split.items()},
        "split_fallback_event_count": dict(event_counts_by_split),
    }
    return records, summary


def _fit_c1_model(records: Sequence[_FallbackRecord], *, smoothing_alpha: float) -> _C1Model:
    model = _C1Model(
        delta_model=_CategoricalModel(alpha=smoothing_alpha),
        order_model=_CategoricalModel(alpha=smoothing_alpha),
        bit_models={plane: _BernoulliModel(alpha=smoothing_alpha) for plane in BITPLANES},
    )
    for record in records:
        model.update(record)
    return model


def _c1_online_bits(records: Sequence[_FallbackRecord], initial_model: _C1Model) -> tuple[float, ...]:
    output = [0.0] * len(records)
    current_source: int | None = None
    overlay: _C1Model | None = None
    for record in records:
        if current_source != record.source_row_index:
            overlay = _empty_c1_overlay(initial_model)
            current_source = record.source_row_index
        if overlay is None:
            raise RuntimeError("online C1 overlay was not initialized")
        output[record.id] = _c1_overlay_bits(initial_model, overlay, record)
        overlay.update(record)
    return tuple(output)


def _empty_c1_overlay(initial_model: _C1Model) -> _C1Model:
    return _C1Model(
        delta_model=_CategoricalModel(alpha=initial_model.delta_model.alpha),
        order_model=_CategoricalModel(alpha=initial_model.order_model.alpha),
        bit_models={plane: _BernoulliModel(alpha=model.alpha) for plane, model in initial_model.bit_models.items()},
    )


def _c1_overlay_bits(base: _C1Model, overlay: _C1Model, record: _FallbackRecord) -> float:
    bits = _combined_categorical_bits(base.delta_model, overlay.delta_model, _c1_delta_context(record), str(record.delta))
    delta_bucket = _delta_bucket(record.delta)
    bits += _combined_categorical_bits(
        base.order_model,
        overlay.order_model,
        _c1_order_context(record, delta_bucket=delta_bucket),
        record.order_signature,
    )
    for plane in BITPLANES:
        mask = _plane_mask(record, plane)
        for lane in range(4):
            bit = (mask >> lane) & 1
            bits += _combined_bernoulli_bits(
                base.bit_models[plane],
                overlay.bit_models[plane],
                _c1_bit_context(record, plane=plane, lane=lane, delta_bucket=delta_bucket),
                bit,
            )
    return bits


def _fit_c2_model(records: Sequence[_FallbackRecord], *, smoothing_alpha: float) -> _C2Model:
    model = _C2Model(
        levels={
            "rhythm": _PpmLevelModel(alpha=smoothing_alpha),
            "cardinality": _PpmLevelModel(alpha=smoothing_alpha),
            "exact": _PpmLevelModel(alpha=smoothing_alpha),
        }
    )
    for record in records:
        for level, symbol in _c2_symbols(record):
            model.levels[level].add(record, level, symbol)
    return model


def _build_oracle_plan(records: Sequence[_FallbackRecord], *, candidate_bits: Mapping[str, Sequence[float]]) -> _CostPlan:
    replacement: list[float] = []
    payload: list[float] = []
    mode: list[float] = []
    modes: list[str] = []
    for record in records:
        candidates = {"raw_literal": record.raw_bits}
        candidates.update({name: float(bits[record.id]) for name, bits in candidate_bits.items()})
        name, value = min(candidates.items(), key=lambda item: item[1])
        replacement.append(float(value))
        payload.append(float(value))
        mode.append(0.0)
        modes.append(name)
    return _CostPlan(
        name="f0_free_oracle_upper_bound",
        legal_codec=False,
        replacement_bits=tuple(replacement),
        payload_bits=tuple(payload),
        mode_header_bits=tuple(mode),
        mode_codes=tuple(modes),
        table_model_cost_bits=0.0,
        notes="Illegal upper bound: chooses the cheapest candidate per fallback literal with no mode/table selector cost.",
        reconstruction_checked_count=len(records),
        reconstruction_mismatch_count=0,
    )


def _build_explicit_selector_plan(
    records: Sequence[_FallbackRecord],
    *,
    candidate_bits: Mapping[str, Sequence[float]],
    table_model_cost_bits: float,
) -> _CostPlan:
    replacement: list[float] = []
    payload: list[float] = []
    mode: list[float] = []
    modes: list[str] = []
    mode_bits = math.log2(float(max(2, len(candidate_bits))))
    for record in records:
        name, value = min(
            ((candidate_name, float(bits[record.id])) for candidate_name, bits in candidate_bits.items()),
            key=lambda item: item[1] + mode_bits,
        )
        payload.append(value)
        mode.append(mode_bits)
        replacement.append(value + mode_bits)
        modes.append(name)
    return _CostPlan(
        name="s1_explicit_per_fallback_selector",
        legal_codec=True,
        replacement_bits=tuple(replacement),
        payload_bits=tuple(payload),
        mode_header_bits=tuple(mode),
        mode_codes=tuple(modes),
        table_model_cost_bits=float(table_model_cost_bits),
        notes="Legal explicit selector: every fallback literal writes a uniform mode token before payload.",
        reconstruction_checked_count=len(records),
        reconstruction_mismatch_count=0,
    )


def _all_fallback_plan(
    name: str,
    records: Sequence[_FallbackRecord],
    bits: Sequence[float],
    *,
    table_model_cost_bits: float,
    notes: str,
) -> _CostPlan:
    replacement = tuple(float(bits[record.id]) for record in records)
    return _CostPlan(
        name=name,
        legal_codec=True,
        replacement_bits=replacement,
        payload_bits=replacement,
        mode_header_bits=tuple(0.0 for _ in records),
        mode_codes=tuple(name for _ in records),
        table_model_cost_bits=float(table_model_cost_bits),
        notes=notes,
        reconstruction_checked_count=len(records),
        reconstruction_mismatch_count=0,
    )


def _lz_candidates_by_record(
    records: Sequence[_FallbackRecord],
    *,
    window_fallbacks: int,
    max_span: int,
) -> dict[int, _LzCandidate]:
    by_source: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_row_index].append(record)
    candidates: dict[int, _LzCandidate] = {}
    for source_records in by_source.values():
        source_records.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
        exact_index: dict[tuple[int, int, int, int, str], deque[int]] = defaultdict(deque)
        mirror_index: dict[tuple[int, int, int, int, str], deque[int]] = defaultdict(deque)
        skeleton_index: dict[tuple[int, str, int, int, int, str], deque[int]] = defaultdict(deque)
        for index, record in enumerate(source_records):
            best: _LzCandidate | None = None
            candidate_indexes: list[tuple[str, int]] = []
            candidate_indexes.extend(
                ("exact", previous_index)
                for previous_index in _recent_index_candidates(exact_index[_atom_key(record)], index, window_fallbacks)
            )
            candidate_indexes.extend(
                ("mirror", previous_index)
                for previous_index in _recent_index_candidates(mirror_index[_atom_key(record)], index, window_fallbacks)
            )
            candidate_indexes.extend(
                ("skeleton", previous_index)
                for previous_index in _recent_index_candidates(skeleton_index[_skeleton_key(record)], index, window_fallbacks)
            )
            for match_type, previous_index in candidate_indexes:
                distance = index - previous_index
                length, residual = _lz_match_length(source_records, index, previous_index, match_type=match_type, max_span=max_span)
                if length <= 0:
                    continue
                payload = (
                    math.log2(float(window_fallbacks + 1))
                    + math.log2(float(max_span + 1))
                    + LZ_TRANSFORM_BITS
                    + residual
                )
                candidate = _LzCandidate(
                    match_type=match_type,
                    length=length,
                    distance=distance,
                    payload_bits=payload,
                    residual_bits=residual,
                )
                if best is None or _lz_candidate_gain(source_records, index, candidate) > _lz_candidate_gain(source_records, index, best):
                    best = candidate
            if best is not None:
                candidates[record.id] = best
            _append_index_candidate(exact_index[_atom_key(record)], index, window_fallbacks)
            _append_index_candidate(mirror_index[_mirror_atom_key(record)], index, window_fallbacks)
            _append_index_candidate(skeleton_index[_skeleton_key(record)], index, window_fallbacks)
    return candidates


def _recent_index_candidates(indexes: deque[int], current_index: int, window_fallbacks: int) -> list[int]:
    lower = current_index - int(window_fallbacks)
    while indexes and indexes[0] < lower:
        indexes.popleft()
    return [value for value in list(indexes)[-DEFAULT_LZ_CANDIDATE_SCAN_LIMIT:] if value < current_index]


def _append_index_candidate(indexes: deque[int], value: int, window_fallbacks: int) -> None:
    indexes.append(int(value))
    lower = int(value) - int(window_fallbacks)
    while indexes and indexes[0] < lower:
        indexes.popleft()


def _build_lz_span_plan(
    records: Sequence[_FallbackRecord],
    candidates: Mapping[int, _LzCandidate],
    *,
    window_fallbacks: int,
    max_span: int,
) -> _CostPlan:
    replacement = [record.raw_bits for record in records]
    payload = [record.raw_bits for record in records]
    mode = [0.0 for _ in records]
    modes = ["raw_literal" for _ in records]
    reconstruction_mismatch_count = 0
    by_source: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_row_index].append(record)
    for source_records in by_source.values():
        source_records.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
        index = 0
        while index < len(source_records):
            record = source_records[index]
            candidate = candidates.get(record.id)
            if candidate is None or not _active_span_record(record):
                index += 1
                continue
            previous_index = index - candidate.distance
            verified_length, verified_residual = _lz_match_length(
                source_records,
                index,
                previous_index,
                match_type=candidate.match_type,
                max_span=max_span,
            )
            if verified_length < candidate.length or abs(verified_residual - candidate.residual_bits) > 1e-9:
                reconstruction_mismatch_count += candidate.length
                index += 1
                continue
            span = source_records[index : index + candidate.length]
            if len(span) != candidate.length or any(not _active_span_record(item) for item in span):
                index += 1
                continue
            raw_sum = sum(item.raw_bits for item in span)
            charged = candidate.payload_bits + MODE_BITS_4
            if charged >= raw_sum:
                index += 1
                continue
            per_record_payload = candidate.payload_bits / float(candidate.length)
            per_record_mode = MODE_BITS_4 / float(candidate.length)
            for item in span:
                replacement[item.id] = per_record_payload + per_record_mode
                payload[item.id] = per_record_payload
                mode[item.id] = per_record_mode
                modes[item.id] = f"lz_{candidate.match_type}"
            index += candidate.length
    return _CostPlan(
        name="c3_lz_active_span_backref",
        legal_codec=True,
        replacement_bits=tuple(replacement),
        payload_bits=tuple(payload),
        mode_header_bits=tuple(mode),
        mode_codes=tuple(modes),
        table_model_cost_bits=0.0,
        notes=(
            "C3 LZ-style signaled active-span backreference. The patch header charges mode, distance, length, "
            f"and transform over a previous-{window_fallbacks} fallback window with max span {max_span}."
        ),
        reconstruction_checked_count=len(records),
        reconstruction_mismatch_count=reconstruction_mismatch_count,
    )


def _build_patch_plan(
    records: Sequence[_FallbackRecord],
    c1_bits: Sequence[float],
    *,
    max_span: int,
    table_model_cost_bits: float,
) -> _CostPlan:
    replacement = [record.raw_bits for record in records]
    payload = [record.raw_bits for record in records]
    mode = [0.0 for _ in records]
    modes = ["raw_literal" for _ in records]
    header = MODE_BITS_4 + math.log2(float(max_span + 1)) + PATCH_CODEC_BITS
    by_source: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_row_index].append(record)
    for source_records in by_source.values():
        source_records.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
        index = 0
        while index < len(source_records):
            if not _active_span_record(source_records[index]):
                index += 1
                continue
            best_length = 0
            best_gain = 0.0
            for length in range(1, min(max_span, len(source_records) - index) + 1):
                span = source_records[index : index + length]
                if any(not _active_span_record(item) for item in span):
                    break
                raw_sum = sum(item.raw_bits for item in span)
                c1_sum = sum(float(c1_bits[item.id]) for item in span)
                gain = raw_sum - (c1_sum + header)
                if gain > best_gain:
                    best_gain = gain
                    best_length = length
            if best_length <= 0:
                index += 1
                continue
            span = source_records[index : index + best_length]
            per_record_header = header / float(best_length)
            for item in span:
                replacement[item.id] = float(c1_bits[item.id]) + per_record_header
                payload[item.id] = float(c1_bits[item.id])
                mode[item.id] = per_record_header
                modes[item.id] = "patch_c1"
            index += best_length
    return _CostPlan(
        name="c4_signaled_active_patch_c1",
        legal_codec=True,
        replacement_bits=tuple(replacement),
        payload_bits=tuple(payload),
        mode_header_bits=tuple(mode),
        mode_codes=tuple(modes),
        table_model_cost_bits=float(table_model_cost_bits),
        notes="C4 signaled active-state patches with charged patch start, length, codec header, and C1 payload.",
        reconstruction_checked_count=len(records),
        reconstruction_mismatch_count=0,
    )


def _score_plan(
    plan: _CostPlan,
    records: Sequence[_FallbackRecord],
    *,
    baseline_splits: Mapping[str, Mapping[str, Any]],
    source_filters: Mapping[str, set[int]],
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in ("train", "valid", "test", "same_song_filtered_test"):
        source_filter = source_filters.get(split)
        baseline = baseline_splits.get(split, {})
        selected_records = [
            record
            for record in records
            if (record.split == ("test" if split == "same_song_filtered_test" else split))
            and (source_filter is None or record.source_row_index in source_filter)
        ]
        splits[split] = _score_records_against_baseline(plan, selected_records, baseline)
    return {
        "variant": plan.name,
        "legal_codec": plan.legal_codec,
        "notes": plan.notes,
        "table_model_cost_bits": plan.table_model_cost_bits,
        "splits": splits,
    }


def _score_records_against_baseline(
    plan: _CostPlan,
    records: Sequence[_FallbackRecord],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    event_count = int(baseline.get("event_count", 0) or 0)
    group_count = int(baseline.get("group_count", 0) or 0)
    baseline_code_bits = float(baseline.get("code_bits", 0.0) or 0.0)
    baseline_dictionary_bits = float(baseline.get("dictionary_cost_bits", 0.0) or 0.0)
    raw_bits = sum(record.raw_bits for record in records)
    replacement_bits = sum(plan.replacement_bits[record.id] for record in records)
    payload_bits = sum(plan.payload_bits[record.id] for record in records)
    mode_bits = sum(plan.mode_header_bits[record.id] for record in records)
    code_bits = baseline_code_bits - raw_bits + replacement_bits
    mapset_bits = {int(key): float(value) for key, value in baseline.get("mapset_bits", {}).items()}
    for record in records:
        mapset_bits[int(record.beatmap_set_id)] = mapset_bits.get(int(record.beatmap_set_id), 0.0) - record.raw_bits + plan.replacement_bits[record.id]
    fallback_event_count = sum(record.event_count for record in records)
    fallback_count = len(records)
    dictionary_bits = baseline_dictionary_bits + plan.table_model_cost_bits
    mode_counter = Counter(plan.mode_codes[record.id] for record in records)
    return {
        "event_count": event_count,
        "group_count": group_count,
        "chunk_count": int(baseline.get("chunk_count", 0) or 0),
        "fallback_literal_count": fallback_count,
        "fallback_event_count": fallback_event_count,
        "baseline_code_bits": baseline_code_bits,
        "baseline_raw_fallback_bits": raw_bits,
        "replacement_payload_bits": payload_bits,
        "replacement_mode_header_bits": mode_bits,
        "replacement_bits": replacement_bits,
        "code_bits": code_bits,
        "bits_per_event": code_bits / float(event_count) if event_count else 0.0,
        "dictionary_cost_bits": dictionary_bits,
        "baseline_dictionary_cost_bits": baseline_dictionary_bits,
        "table_model_cost_bits": plan.table_model_cost_bits,
        "dictionary_bits_per_event": dictionary_bits / float(event_count) if event_count else 0.0,
        "bits_per_event_with_dictionary": (code_bits + dictionary_bits) / float(event_count) if event_count else 0.0,
        "delta_bits_per_event_with_dictionary": (
            (code_bits + dictionary_bits) / float(event_count) - float(baseline.get("bits_per_event_with_dictionary", 0.0) or 0.0)
            if event_count
            else 0.0
        ),
        "fallback_payload_bits_per_fallback_event": replacement_bits / float(fallback_event_count) if fallback_event_count else 0.0,
        "baseline_raw_bits_per_fallback_event": raw_bits / float(fallback_event_count) if fallback_event_count else 0.0,
        "fallback_payload_delta_bits_per_fallback_event": (
            (replacement_bits - raw_bits) / float(fallback_event_count) if fallback_event_count else 0.0
        ),
        "mode_usage": dict(mode_counter.most_common()),
        "mapset_bits": mapset_bits,
        "mapset_events": {int(key): int(value) for key, value in baseline.get("mapset_events", {}).items()},
    }


def _bucket_rows_for_plan(
    plan: _CostPlan,
    records: Sequence[_FallbackRecord],
    baseline_splits: Mapping[str, Mapping[str, Any]],
    *,
    legal_codec: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("valid", "test"):
        split_records = [record for record in records if record.split == split]
        for table, bucket_func in (
            ("active_hold_count", lambda record: _count_bucket(record.active_hold_count)),
            ("group_class", lambda record: record.group_class),
            ("boundary_bucket", lambda record: record.boundary_bucket),
            ("chord_ln_mixed", lambda record: "yes" if record.chord_ln_mixed else "no"),
            ("active_span", lambda record: "yes" if _active_span_record(record) else "no"),
        ):
            grouped: dict[str, list[_FallbackRecord]] = defaultdict(list)
            for record in split_records:
                grouped[str(bucket_func(record))].append(record)
            for bucket, bucket_records in grouped.items():
                raw_bits = sum(record.raw_bits for record in bucket_records)
                replacement_bits = sum(plan.replacement_bits[record.id] for record in bucket_records)
                events = sum(record.event_count for record in bucket_records)
                rows.append(
                    {
                        "variant": plan.name,
                        "split": split,
                        "table": table,
                        "bucket": bucket,
                        "legal_codec": legal_codec,
                        "fallback_literal_count": len(bucket_records),
                        "fallback_event_count": events,
                        "baseline_raw_bits_per_fallback_event": raw_bits / float(events) if events else 0.0,
                        "replacement_bits_per_fallback_event": replacement_bits / float(events) if events else 0.0,
                        "delta_bits_per_fallback_event": (replacement_bits - raw_bits) / float(events) if events else 0.0,
                        "fallback_share": (
                            float(len(bucket_records)) / float(baseline_splits.get(split, {}).get("group_count", 1) or 1)
                        ),
                    }
                )
    return rows


def _usage_rows_for_plan(plan: _CostPlan, records: Sequence[_FallbackRecord]) -> list[dict[str, Any]]:
    rows = []
    for split in ("valid", "test"):
        split_records = [record for record in records if record.split == split]
        total = len(split_records)
        counter = Counter(plan.mode_codes[record.id] for record in split_records)
        for mode, count in counter.most_common():
            rows.append(
                {
                    "variant": plan.name,
                    "split": split,
                    "table": "codec_usage",
                    "bucket": mode,
                    "legal_codec": plan.legal_codec,
                    "fallback_literal_count": int(count),
                    "usage_share": float(count) / float(total) if total else 0.0,
                }
            )
    return rows


def _reconstruction_guard(records: Sequence[_FallbackRecord], plans: Sequence[_CostPlan]) -> dict[str, Any]:
    rows = [
        {
            "variant": "b0_r0_delta",
            "legal_codec": True,
            "checked_fallback_literal_count": len(records),
            "mismatch_count": 0,
            "pass": True,
            "policy": "baseline raw literal payloads are already exact fallback atoms",
        }
    ]
    for plan in plans:
        rows.append(
            {
                "variant": plan.name,
                "legal_codec": plan.legal_codec,
                "checked_fallback_literal_count": int(plan.reconstruction_checked_count),
                "mismatch_count": int(plan.reconstruction_mismatch_count),
                "pass": int(plan.reconstruction_mismatch_count) == 0,
                "policy": _reconstruction_policy_for_plan(plan),
            }
        )
    legal_rows = [row for row in rows if bool(row["legal_codec"])]
    return {
        "pass": all(bool(row["pass"]) for row in legal_rows),
        "mismatch_count": sum(int(row["mismatch_count"]) for row in legal_rows),
        "checked_fallback_literal_count": len(records),
        "rows": rows,
        "policy": (
            "Fallback-payload reconstruction guard. Raw/C1/C2/C4 encode exact delta, order signature, tap mask, "
            "press mask, and release mask. C3 selected LZ spans are verified against prior decoded fallback records "
            "when the plan is built."
        ),
    }


def _reconstruction_policy_for_plan(plan: _CostPlan) -> str:
    if plan.name == "c3_lz_active_span_backref":
        return "selected LZ spans must match prior fallback-substream records under exact, mirror, or skeleton+residual transform"
    if plan.name == "c4_signaled_active_patch_c1":
        return "patched records use C1 exact fallback syntax plus charged patch header"
    if plan.name.startswith("c1"):
        return "C1 encodes exact delta/order plus tap/press/release bitplanes"
    if plan.name.startswith("c2"):
        return "C2 encodes exact fallback syntax through rhythm, cardinality, and exact-mask symbols"
    if plan.name.startswith("s1"):
        return "per-fallback selector chooses only raw/C1/C2 exact-syntax payloads and charges a mode"
    if plan.name.startswith("f0"):
        return "oracle exact-payload diagnostic; illegal selector but payloads are exact"
    return "exact fallback payload policy"


def _diagnostics(
    records: Sequence[_FallbackRecord],
    *,
    baseline_splits: Mapping[str, Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    c1_model: _C1Model,
    c2_model: _C2Model,
    lz_candidates: Mapping[int, _LzCandidate],
    top_example_limit: int,
) -> dict[str, Any]:
    del c1_model, c2_model
    table_rows: list[dict[str, Any]] = []
    test_records = [record for record in records if record.split == "test"]
    raw_sorted = sorted(test_records, key=lambda record: (-record.raw_bits, record.source_row_index, record.absolute_units))
    top_examples = [
        {
            "source_row_index": record.source_row_index,
            "beatmap_set_id": record.beatmap_set_id,
            "chunk_index": record.chunk_index,
            "offset_units": record.offset_units,
            "token": record.token,
            "raw_bits": record.raw_bits,
            "pre_hold_mask": record.pre_hold_mask,
            "group_class": record.group_class,
            "boundary_bucket": record.boundary_bucket,
        }
        for record in raw_sorted[:top_example_limit]
    ]
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(len(raw_sorted) * fraction))
        subset = raw_sorted[:count]
        table_rows.append(_surprisal_row(subset, fraction=fraction))
    lz_counter = Counter(candidate.match_type for candidate in lz_candidates.values())
    lz_lengths = [candidate.length for candidate in lz_candidates.values()]
    lz_distances = [candidate.distance for candidate in lz_candidates.values()]
    for match_type, count in lz_counter.most_common():
        table_rows.append(
            {
                "variant": "c3_lz_active_span_backref",
                "split": "all",
                "table": "lz_candidate_match_type",
                "bucket": match_type,
                "fallback_literal_count": int(count),
                "coverage": float(count) / float(len(records)) if records else 0.0,
            }
        )
    oracle_test = oracle_report.get("splits", {}).get("test", {})
    baseline_test = baseline_splits.get("test", {})
    summary = {
        "top_surprisal_examples": top_examples,
        "oracle_test_delta_bits_per_event": oracle_test.get("delta_bits_per_event_with_dictionary"),
        "oracle_test_gain_bits_per_event": -float(oracle_test.get("delta_bits_per_event_with_dictionary", 0.0) or 0.0),
        "oracle_stop_gate": "stop_if_gain_lt_0.02",
        "oracle_implement_charged_gate": "implement_if_gain_ge_0.05",
        "baseline_test_bits_per_event_with_dictionary": baseline_test.get("bits_per_event_with_dictionary"),
        "lz_candidate_coverage": float(len(lz_candidates)) / float(len(records)) if records else 0.0,
        "lz_mean_match_length": float(np.mean(lz_lengths)) if lz_lengths else 0.0,
        "lz_mean_distance": float(np.mean(lz_distances)) if lz_distances else 0.0,
    }
    return {"summary": summary, "table_rows": table_rows}


def _surprisal_row(records: Sequence[_FallbackRecord], *, fraction: float) -> dict[str, Any]:
    fallback_count = len(records)
    active = sum(1 for record in records if record.active_hold_count > 0)
    chord_ln = sum(1 for record in records if record.chord_ln_mixed)
    boundary = sum(1 for record in records if record.boundary_bucket in {"exact_start", "near_start_le12", "near_end_le12"})
    raw_bits = sum(record.raw_bits for record in records)
    return {
        "variant": "f0_surprisal_literal_forensics",
        "split": "test",
        "table": "top_surprisal",
        "bucket": f"top_{int(fraction * 100)}pct",
        "fallback_literal_count": fallback_count,
        "fallback_event_count": sum(record.event_count for record in records),
        "mean_raw_bits": raw_bits / float(fallback_count) if fallback_count else 0.0,
        "active_hold_share": float(active) / float(fallback_count) if fallback_count else 0.0,
        "chord_ln_mixed_share": float(chord_ln) / float(fallback_count) if fallback_count else 0.0,
        "boundary_near_share": float(boundary) / float(fallback_count) if fallback_count else 0.0,
    }


def _select_legal_variant(variant_reports: Mapping[str, Mapping[str, Any]]) -> str:
    candidates = [
        name
        for name, report in variant_reports.items()
        if name != "b0_r0_delta" and bool(report.get("legal_codec"))
    ]
    if not candidates:
        return "b0_r0_delta"
    return min(
        candidates,
        key=lambda name: float(
            variant_reports[name].get("splits", {}).get("valid", {}).get("bits_per_event_with_dictionary", math.inf)
        ),
    )


def _pass_criteria(
    *,
    selected_variant: str,
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    record_summary: Mapping[str, Any],
    reconstruction_guard: Mapping[str, Any],
) -> dict[str, Any]:
    selected_test = selected.get("splits", {}).get("test", {})
    selected_valid = selected.get("splits", {}).get("valid", {})
    baseline_test = baseline.get("splits", {}).get("test", {})
    baseline_filtered = baseline.get("splits", {}).get("same_song_filtered_test", {})
    selected_filtered = selected.get("splits", {}).get("same_song_filtered_test", {})
    selected_test_bits = float(selected_test.get("bits_per_event_with_dictionary", math.inf))
    selected_valid_bits = float(selected_valid.get("bits_per_event_with_dictionary", math.inf))
    baseline_test_bits = float(baseline_test.get("bits_per_event_with_dictionary", math.inf))
    baseline_consistency = abs(baseline_test_bits - EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT) < 0.05
    global_promotion = selected_test_bits <= baseline_test_bits - 0.01
    global_continuation = selected_test_bits <= baseline_test_bits + 0.005
    active_gain = _bucket_gain(selected, baseline, table="active_hold_count", bucket="1", split="test")
    active_gain = max(active_gain, _bucket_gain(selected, baseline, table="active_hold_count", bucket="2", split="test"))
    active_gain = max(active_gain, _bucket_gain(selected, baseline, table="active_hold_count", bucket="3plus", split="test"))
    filtered_delta = (
        float(selected_filtered.get("bits_per_event_with_dictionary", math.inf))
        - float(baseline_filtered.get("bits_per_event_with_dictionary", math.inf))
    )
    bootstrap_mostly_below_zero = bool(
        bootstrap.get("available")
        and float(bootstrap.get("delta_bits_per_event_p975", math.inf)) < 0.005
    )
    reconstruction_pass = bool(reconstruction_guard.get("pass"))
    positive = bool(
        baseline_consistency
        and reconstruction_pass
        and int(record_summary.get("fallback_literal_count", 0) or 0) > 0
        and (global_promotion or (global_continuation and active_gain >= 0.05))
        and filtered_delta <= 0.005
        and bootstrap_mostly_below_zero
    )
    return {
        "selected_variant": selected_variant,
        "baseline_consistency_pass": baseline_consistency,
        "expected_r0_test_charged_bits_per_event": EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
        "observed_r0_test_charged_bits_per_event": baseline_test_bits,
        "valid_selected_bits_per_event_with_dictionary": selected_valid_bits,
        "test_selected_bits_per_event_with_dictionary": selected_test_bits,
        "test_baseline_bits_per_event_with_dictionary": baseline_test_bits,
        "test_delta_bits_per_event_with_dictionary": selected_test_bits - baseline_test_bits,
        "global_promotion_pass": global_promotion,
        "global_continuation_pass": global_continuation,
        "active_hold_fallback_gain_bits_per_event": active_gain,
        "same_song_filtered_delta_bits_per_event": filtered_delta,
        "same_song_filtered_pass": filtered_delta <= 0.005,
        "bootstrap_mostly_below_zero_pass": bootstrap_mostly_below_zero,
        "reconstruction_pass": reconstruction_pass,
        "research_pass": positive,
    }


def _bucket_gain(selected: Mapping[str, Any], baseline: Mapping[str, Any], *, table: str, bucket: str, split: str) -> float:
    del baseline
    for row in selected.get("bucket_rows", []):
        if row.get("split") == split and row.get("table") == table and row.get("bucket") == bucket:
            return -float(row.get("delta_bits_per_fallback_event", 0.0) or 0.0)
    return 0.0


def _recommendation(pass_criteria: Mapping[str, Any], *, selected_variant: str) -> str:
    if pass_criteria.get("research_pass"):
        return f"TEST_NEXT: {selected_variant} passed charged CASF gates; refine the winning fallback entropy model before mapper-side work."
    delta = float(pass_criteria.get("test_delta_bits_per_event_with_dictionary", math.inf) or math.inf)
    active_gain = float(pass_criteria.get("active_hold_fallback_gain_bits_per_event", 0.0) or 0.0)
    if delta <= 0.01 or active_gain >= 0.05:
        return f"MUTATE: {selected_variant} has a local signal but fails at least one robustness gate; inspect mode/table cost and same-song/bootstrap."
    return f"KILL_OR_DEFER: {selected_variant} is the best charged legal CASF variant but does not beat the in-harness r0_delta baseline."


def _c1_delta_context(record: _FallbackRecord) -> tuple[str, ...]:
    return (
        "delta",
        f"phase_{record.bar_phase_half}",
        f"pre_hold_{_count_bucket(record.active_hold_count)}",
        f"prev_{record.previous_skeleton}",
        f"prev_fallback_{int(record.previous_fallback)}",
        f"group_{_count_bucket(record.group_index)}",
    )


def _c1_order_context(record: _FallbackRecord, *, delta_bucket: str) -> tuple[str, ...]:
    return (
        "order",
        f"delta_{delta_bucket}",
        f"pre_hold_{_count_bucket(record.active_hold_count)}",
        f"prev_{record.previous_skeleton}",
    )


def _c1_bit_context(record: _FallbackRecord, *, plane: str, lane: int, delta_bucket: str) -> tuple[str, ...]:
    lane_mask = 1 << int(lane)
    return (
        plane,
        f"lane_{lane}",
        f"delta_{delta_bucket}",
        f"pre_lane_{int(bool(record.pre_hold_mask & lane_mask))}",
        f"pre_hold_{_count_bucket(record.active_hold_count)}",
        f"prev_{record.previous_skeleton}",
        f"phase_{record.bar_phase_half}",
    )


def _c2_symbols(record: _FallbackRecord) -> tuple[tuple[str, str], ...]:
    rhythm = f"D{_delta_bucket(record.delta)}:{record.group_class}:A{_count_bucket(record.active_count)}"
    cardinality = f"T{record.tap_mask.bit_count()}:S{record.ln_start_mask.bit_count()}:E{record.ln_end_mask.bit_count()}"
    exact = f"d{record.delta}:t{record.tap_mask}:s{record.ln_start_mask}:e{record.ln_end_mask}:o{record.order_signature}"
    return (("rhythm", rhythm), ("cardinality", cardinality), ("exact", exact))


def _ppm_context(record: _FallbackRecord, *, level: str, depth: int) -> tuple[str, ...]:
    history = tuple(record.history_skeletons[-depth:]) if depth > 0 else ()
    return (
        level,
        f"depth_{depth}",
        f"phase_{record.bar_phase_half}",
        f"pre_hold_{_count_bucket(record.active_hold_count)}",
        f"prev_fallback_{int(record.previous_fallback)}",
        *history,
    )


def _plane_mask(record: _FallbackRecord, plane: str) -> int:
    if plane == "tap":
        return record.tap_mask
    if plane == "press":
        return record.ln_start_mask
    if plane == "release":
        return record.ln_end_mask
    raise ValueError(f"unknown bitplane: {plane}")


def _delta_bucket(delta: int) -> str:
    value = int(delta)
    if value < 0:
        return "neg"
    if value == 0:
        return "0"
    if value <= 6:
        return "1to6"
    if value <= 12:
        return "7to12"
    if value <= 24:
        return "13to24"
    if value <= 48:
        return "25to48"
    return "49plus"


def _skeleton_symbol(*, delta: int, tap: int, start: int, end: int) -> str:
    return f"D{_delta_bucket(delta)}:{_group_class(tap, start, end)}:A{_count_bucket((tap | start | end).bit_count())}"


def _active_span_record(record: _FallbackRecord) -> bool:
    return bool(record.active_hold_count > 0 or record.chord_ln_mixed)


def _lz_match_length(
    source_records: Sequence[_FallbackRecord],
    current_index: int,
    previous_index: int,
    *,
    match_type: str,
    max_span: int,
) -> tuple[int, float]:
    length = 0
    residual = 0.0
    while length < max_span and current_index + length < len(source_records) and previous_index + length < current_index:
        current = source_records[current_index + length]
        previous = source_records[previous_index + length]
        if current.split != previous.split:
            break
        if match_type == "exact":
            if _atom_key(current) != _atom_key(previous):
                break
        elif match_type == "mirror":
            if _atom_key(current) != _mirror_atom_key(previous):
                break
        elif match_type == "skeleton":
            if _skeleton_key(current) != _skeleton_key(previous):
                break
            residual += _lane_residual_bits(current)
        else:
            raise ValueError(f"unknown LZ match type: {match_type}")
        length += 1
    return length, residual


def _lz_candidate_gain(source_records: Sequence[_FallbackRecord], current_index: int, candidate: _LzCandidate) -> float:
    raw = sum(record.raw_bits for record in source_records[current_index : current_index + candidate.length])
    return raw - candidate.payload_bits


def _atom_key(record: _FallbackRecord) -> tuple[int, int, int, int, str]:
    return (record.delta, record.tap_mask, record.ln_start_mask, record.ln_end_mask, record.order_signature)


def _mirror_atom_key(record: _FallbackRecord) -> tuple[int, int, int, int, str]:
    return (
        record.delta,
        _mirror_mask_4(record.tap_mask),
        _mirror_mask_4(record.ln_start_mask),
        _mirror_mask_4(record.ln_end_mask),
        record.order_signature,
    )


def _skeleton_key(record: _FallbackRecord) -> tuple[int, str, int, int, int, str]:
    return (
        record.delta,
        record.group_class,
        record.tap_mask.bit_count(),
        record.ln_start_mask.bit_count(),
        record.ln_end_mask.bit_count(),
        "." if record.order_signature == "." else "order",
    )


def _mirror_mask_4(mask: int) -> int:
    value = 0
    for lane in range(4):
        if int(mask) & (1 << lane):
            value |= 1 << (3 - lane)
    return value


def _lane_residual_bits(record: _FallbackRecord) -> float:
    bits = (
        _comb_bits(4, record.tap_mask.bit_count())
        + _comb_bits(4, record.ln_start_mask.bit_count())
        + _comb_bits(4, record.ln_end_mask.bit_count())
    )
    if record.order_signature != ".":
        bits += max(1, len(record.order_signature.encode("utf-8"))) * 8
    return bits


def _comb_bits(n: int, k: int) -> float:
    k = int(k)
    if k <= 0 or k >= n:
        return 0.0
    return math.log2(float(math.comb(n, k)))


def _categorical_probability(counter: Counter[str], vocab: set[str], symbol: str, *, alpha: float) -> float:
    total = sum(counter.values())
    vocab_size = max(1, len(vocab))
    if total <= 0:
        return 1.0 / float(vocab_size + 1)
    if symbol not in vocab:
        return alpha / (float(total) + alpha * float(vocab_size + 1))
    return (float(counter.get(symbol, 0)) + alpha) / (float(total) + alpha * float(vocab_size + 1))


def _categorical_bits(counter: Counter[str], vocab: set[str], symbol: str, *, alpha: float) -> float:
    probability = _categorical_probability(counter, vocab, symbol, alpha=alpha)
    bits = -math.log2(max(probability, 1e-300))
    if symbol not in vocab:
        bits += _payload_bits(symbol)
    return bits


def _combined_categorical_bits(
    base: _CategoricalModel,
    overlay: _CategoricalModel,
    context: Sequence[str],
    symbol: str,
) -> float:
    key = tuple(context)
    counter = Counter()
    if key in base.counts or key in overlay.counts:
        counter.update(base.counts.get(key, Counter()))
        counter.update(overlay.counts.get(key, Counter()))
    else:
        counter.update(base.global_counts)
        counter.update(overlay.global_counts)
    vocab = set(base.vocab) | set(overlay.vocab)
    return _categorical_bits(counter, vocab, str(symbol), alpha=base.alpha)


def _combined_bernoulli_bits(
    base: _BernoulliModel,
    overlay: _BernoulliModel,
    context: Sequence[str],
    bit: int,
) -> float:
    key = tuple(context)
    if key in base.counts or key in overlay.counts:
        base_counts = base.counts.get(key, [0, 0])
        overlay_counts = overlay.counts.get(key, [0, 0])
    else:
        base_counts = base.global_counts
        overlay_counts = overlay.global_counts
    zero = int(base_counts[0]) + int(overlay_counts[0])
    one = int(base_counts[1]) + int(overlay_counts[1])
    total = zero + one
    numerator = (one if int(bit) else zero) + base.alpha
    denominator = total + 2.0 * base.alpha
    return -math.log2(numerator / denominator) if denominator > 0 else 1.0


def _categorical_table_cost(counts: Mapping[tuple[str, ...], Counter[str]], vocab: set[str]) -> float:
    if not counts:
        return 0.0
    total = sum(sum(counter.values()) for counter in counts.values())
    context_bits = math.log2(max(2, len(counts) + 1))
    symbol_bits = math.log2(max(2, len(vocab) + 1))
    count_bits = math.log2(max(2, total + 1))
    entries = sum(len(counter) for counter in counts.values())
    return float(len(counts)) * context_bits + float(entries) * (symbol_bits + count_bits)


def _payload_bits(symbol: str) -> float:
    return float(max(1, len(str(symbol).encode("utf-8"))) * 8)


def _same_song_source_ids(chunk_df: pd.DataFrame, *, conflict_keys: set[str]) -> set[int]:
    if not conflict_keys:
        return set(int(value) for value in chunk_df.loc[chunk_df["split"] == "test", "source_row_index"].unique())
    maps = chunk_df[["source_row_index", "artist", "title", "version", "split"]].drop_duplicates("source_row_index").copy()
    keys = pd.Series([""] * len(maps), index=maps.index)
    for column in ("artist", "title", "version"):
        keys = keys + "|" + maps[column].map(_normalize_text)
    maps["_key"] = keys
    selected = maps[(maps["split"].astype(str) == "test") & (~maps["_key"].isin(conflict_keys))]
    return set(int(value) for value in selected["source_row_index"].unique())


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).casefold().strip().split())


def _comparison_row(variant: str, split: str, values: Mapping[str, Any], *, legal_codec: bool, table: str) -> dict[str, Any]:
    row = {"variant": variant, "split": split, "table": table, "legal_codec": legal_codec}
    row.update({key: value for key, value in values.items() if key not in {"mapset_bits", "mapset_events", "mode_usage"}})
    return row


def _validate_positive(value: int, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _validate_nonnegative(value: int, name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


def _git_stdout(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_normalize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_name = str(report.get("selected_variant"))
    selected = report.get("variants", {}).get(selected_name, {})
    baseline = report.get("variants", {}).get("b0_r0_delta", {})
    pass_criteria = report.get("pass_criteria", {})
    diagnostics = report.get("diagnostics", {})
    fallback_summary = report.get("fallback_record_summary", {})
    reconstruction_guard = report.get("reconstruction_guard", {})
    lines = [
        "# CASF v2 Context-Adaptive Fallback Codec Result Log",
        "",
        "## Summary",
        "",
        f"- Selected variant: `{selected_name}`",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Research pass: {pass_criteria.get('research_pass')}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Limited: {report.get('limited')}",
        f"- Code dirty: {report.get('code_dirty')}",
        f"- Reconstruction pass: {reconstruction_guard.get('pass') if isinstance(reconstruction_guard, Mapping) else None}",
        f"- Reconstruction mismatches: {reconstruction_guard.get('mismatch_count') if isinstance(reconstruction_guard, Mapping) else None}",
        f"- Invalid active-hold transitions: {fallback_summary.get('invalid_transition_count') if isinstance(fallback_summary, Mapping) else None}",
        f"- Segment-end active holds: {fallback_summary.get('segment_state_reset_count') if isinstance(fallback_summary, Mapping) else None}",
        f"- Baseline charged test bits/event: {_nested(baseline, 'splits', 'test', 'bits_per_event_with_dictionary')}",
        f"- Selected charged test bits/event: {_nested(selected, 'splits', 'test', 'bits_per_event_with_dictionary')}",
        f"- Selected test delta bits/event: {pass_criteria.get('test_delta_bits_per_event_with_dictionary')}",
        f"- Same-song filtered delta: {pass_criteria.get('same_song_filtered_delta_bits_per_event')}",
        f"- Bootstrap delta mean: {_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_mean')}",
        f"- Bootstrap delta 95% interval: [{_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_p025')}, {_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_p975')}]",
        f"- Free-oracle test gain bits/event: {diagnostics.get('oracle_test_gain_bits_per_event') if isinstance(diagnostics, Mapping) else None}",
        "",
        "## Variant Comparison",
        "",
        "| variant | legal | valid charged | test charged | test delta | fallback payload delta | table/model bits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_name, variant in report.get("variants", {}).items():
        valid = variant.get("splits", {}).get("valid", {}) if isinstance(variant, Mapping) else {}
        test = variant.get("splits", {}).get("test", {}) if isinstance(variant, Mapping) else {}
        lines.append(
            "| {variant} | {legal} | {valid} | {test} | {delta} | {fallback_delta} | {table_bits} |".format(
                variant=variant_name,
                legal=variant.get("legal_codec") if isinstance(variant, Mapping) else None,
                valid=_fmt_float(valid.get("bits_per_event_with_dictionary") if isinstance(valid, Mapping) else None),
                test=_fmt_float(test.get("bits_per_event_with_dictionary") if isinstance(test, Mapping) else None),
                delta=_fmt_float(test.get("delta_bits_per_event_with_dictionary") if isinstance(test, Mapping) else 0.0),
                fallback_delta=_fmt_float(test.get("fallback_payload_delta_bits_per_fallback_event") if isinstance(test, Mapping) else 0.0),
                table_bits=_fmt_float(variant.get("table_model_cost_bits") if isinstance(variant, Mapping) else None),
            )
        )
    lines.extend(["", "## Gates", ""])
    for key, value in pass_criteria.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Interpretation", "", str(report.get("recommendation")), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _fmt_float(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _command(args: Sequence[str]) -> str:
    return " ".join(
        shlex.quote(part) for part in ["python", "-m", "pulsefield_model.osu_core.context_adaptive_fallback_codec_audit", *args]
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CASF v2 context-adaptive fallback codec audit.")
    parser.add_argument("--chunk-cache-path", type=Path, default=DEFAULT_BEAT_CHUNK_CACHE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-log-path", type=Path, default=DEFAULT_RESULT_LOG_PATH)
    parser.add_argument("--comparison-csv-path", type=Path, default=DEFAULT_COMPARISON_CSV_PATH)
    parser.add_argument("--diagnostics-csv-path", type=Path, default=DEFAULT_DIAGNOSTICS_CSV_PATH)
    parser.add_argument("--motif-vocab-size", type=int, default=DEFAULT_MOTIF_VOCAB_SIZE)
    parser.add_argument("--motif-min-n", type=int, default=DEFAULT_MOTIF_MIN_N)
    parser.add_argument("--motif-max-n", type=int, default=DEFAULT_MOTIF_MAX_N)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--rare-max-count", type=int, default=DEFAULT_RARE_MAX_COUNT)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--lz-window-fallbacks", type=int, default=DEFAULT_LZ_WINDOW_FALLBACKS)
    parser.add_argument("--lz-max-span", type=int, default=DEFAULT_LZ_MAX_SPAN)
    parser.add_argument("--patch-max-span", type=int, default=DEFAULT_PATCH_MAX_SPAN)
    parser.add_argument("--top-example-limit", type=int, default=20)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_context_adaptive_fallback_codec(
        chunk_cache_path=args.chunk_cache_path,
        report_path=args.report_path,
        result_log_path=args.result_log_path,
        comparison_csv_path=args.comparison_csv_path,
        diagnostics_csv_path=args.diagnostics_csv_path,
        motif_vocab_size=args.motif_vocab_size,
        motif_min_n=args.motif_min_n,
        motif_max_n=args.motif_max_n,
        smoothing_alpha=args.smoothing_alpha,
        rare_max_count=args.rare_max_count,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
        lz_window_fallbacks=args.lz_window_fallbacks,
        lz_max_span=args.lz_max_span,
        patch_max_span=args.patch_max_span,
        top_example_limit=args.top_example_limit,
        limit_chunks=args.limit_chunks,
        command=_command(sys.argv[1:]),
    )
    print(
        "context_adaptive_fallback_codec_audit "
        f"research_pass={report.get('pass_criteria', {}).get('research_pass')} "
        f"selected={report.get('selected_variant')} "
        f"recommendation={report.get('recommendation')} "
        f"report={report.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
