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
DEFAULT_C3_HARDENING_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json",
)
DEFAULT_C3_HARDENING_RESULT_LOG_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_result_log.md",
)
DEFAULT_C3_HARDENING_COMPARISON_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_comparison.csv",
)
DEFAULT_C3_HARDENING_DIAGNOSTICS_CSV_PATH: Final[Path] = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_diagnostics.csv",
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


@dataclass(frozen=True, slots=True)
class _LzSpanSelection:
    source_row_index: int
    split: str
    start_record_id: int
    previous_record_id: int
    start_index: int
    previous_index: int
    length: int
    distance: int
    match_type: str
    payload_bits: float
    mode_bits: float
    residual_bits: float
    raw_bits: float
    charged_bits: float
    target_crosses_chunk: bool
    reference_crosses_chunk: bool
    target_crosses_segment: bool
    reference_crosses_segment: bool
    full_group_contiguous: bool


@dataclass(frozen=True, slots=True)
class _LzPolicy:
    name: str
    label: str
    allowed_match_types: tuple[str, ...]
    activation: str
    window_fallbacks: int | None
    pointer_code: str
    phase_filter: str = "none"
    table_model_cost_bits: float = 0.0


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
    lz_spans: tuple[_LzSpanSelection, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


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
    invalid_transition_source_ids: set[int] = set()
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
                invalid_transition_source_ids.add(source)
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
    segment_state_reset_source_ids = {
        source for (source, _segment), mask in state_by_source_segment.items() if int(mask) != 0
    }
    summary = {
        "fallback_literal_count": len(records),
        "invalid_transition_count": invalid_transition_count,
        "invalid_transition_source_count": len(invalid_transition_source_ids),
        "invalid_transition_source_ids": sorted(invalid_transition_source_ids),
        "invalid_transition_examples": invalid_examples,
        "segment_state_reset_count": segment_state_reset_count,
        "segment_state_reset_source_count": len(segment_state_reset_source_ids),
        "segment_state_reset_source_ids": sorted(segment_state_reset_source_ids),
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
    selections: list[_LzSpanSelection] = []
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
            selections.append(
                _make_lz_span_selection(
                    source_records,
                    start_index=index,
                    previous_index=previous_index,
                    candidate=candidate,
                    mode_bits=MODE_BITS_4,
                )
            )
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
        lz_spans=tuple(selections),
        metadata={
            "window_fallbacks": int(window_fallbacks),
            "max_span": int(max_span),
            "allowed_match_types": ["exact", "mirror", "skeleton"],
            "activation": "active",
            "pointer_code": "fixed_width",
        },
    )


def _make_lz_span_selection(
    source_records: Sequence[_FallbackRecord],
    *,
    start_index: int,
    previous_index: int,
    candidate: _LzCandidate,
    mode_bits: float,
) -> _LzSpanSelection:
    target = source_records[start_index : start_index + candidate.length]
    reference = source_records[previous_index : previous_index + candidate.length]
    raw_bits = sum(record.raw_bits for record in target)
    return _LzSpanSelection(
        source_row_index=int(source_records[start_index].source_row_index),
        split=str(source_records[start_index].split),
        start_record_id=int(source_records[start_index].id),
        previous_record_id=int(source_records[previous_index].id),
        start_index=int(start_index),
        previous_index=int(previous_index),
        length=int(candidate.length),
        distance=int(candidate.distance),
        match_type=str(candidate.match_type),
        payload_bits=float(candidate.payload_bits),
        mode_bits=float(mode_bits),
        residual_bits=float(candidate.residual_bits),
        raw_bits=float(raw_bits),
        charged_bits=float(candidate.payload_bits + mode_bits),
        target_crosses_chunk=len({(record.segment_id, record.chunk_index) for record in target}) > 1,
        reference_crosses_chunk=len({(record.segment_id, record.chunk_index) for record in reference}) > 1,
        target_crosses_segment=len({record.segment_id for record in target}) > 1,
        reference_crosses_segment=len({record.segment_id for record in reference}) > 1,
        full_group_contiguous=_span_full_group_contiguous(target),
    )


def _span_full_group_contiguous(records: Sequence[_FallbackRecord]) -> bool:
    if len(records) <= 1:
        return True
    for previous, current in zip(records, records[1:]):
        if previous.source_row_index != current.source_row_index or previous.segment_id != current.segment_id:
            return False
        if previous.chunk_index != current.chunk_index:
            return False
        if previous.group_index + 1 != current.group_index:
            return False
    return True


def audit_c3_lz_hardening(
    *,
    chunk_cache_path: str | Path = DEFAULT_BEAT_CHUNK_CACHE_PATH,
    report_path: str | Path | None = DEFAULT_C3_HARDENING_REPORT_PATH,
    result_log_path: str | Path | None = DEFAULT_C3_HARDENING_RESULT_LOG_PATH,
    comparison_csv_path: str | Path | None = DEFAULT_C3_HARDENING_COMPARISON_CSV_PATH,
    diagnostics_csv_path: str | Path | None = DEFAULT_C3_HARDENING_DIAGNOSTICS_CSV_PATH,
    motif_vocab_size: int = DEFAULT_MOTIF_VOCAB_SIZE,
    motif_min_n: int = DEFAULT_MOTIF_MIN_N,
    motif_max_n: int = DEFAULT_MOTIF_MAX_N,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    rare_max_count: int = DEFAULT_RARE_MAX_COUNT,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 17,
    lz_window_fallbacks: int = DEFAULT_LZ_WINDOW_FALLBACKS,
    lz_max_span: int = DEFAULT_LZ_MAX_SPAN,
    c3_wide_sweep: bool = False,
    top_example_limit: int = 20,
    limit_chunks: int | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run the C3-only hardening/decomposition audit from the follow-up card."""

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
    dirty_trace_source_ids = set(int(value) for value in record_summary.get("invalid_transition_source_ids", [])) | set(
        int(value) for value in record_summary.get("segment_state_reset_source_ids", [])
    )
    test_source_ids = set(int(value) for value in chunk_df.loc[chunk_df["split"] == "test", "source_row_index"].unique())
    clean_trace_source_ids = test_source_ids - dirty_trace_source_ids
    baseline_splits["clean_trace_test"] = _score_baseline_split(
        chunk_df[
            (chunk_df["split"].fillna("").astype(str) == "test")
            & (chunk_df["source_row_index"].astype(int).isin(clean_trace_source_ids))
        ].copy(),
        split="test",
        model=baseline_model,
        dictionary_bits=baseline_model.dictionary_bits,
        rare_max_count=rare_max_count,
    )
    baseline_splits["dirty_trace_test"] = _score_baseline_split(
        chunk_df[
            (chunk_df["split"].fillna("").astype(str) == "test")
            & (chunk_df["source_row_index"].astype(int).isin(test_source_ids & dirty_trace_source_ids))
        ].copy(),
        split="test",
        model=baseline_model,
        dictionary_bits=baseline_model.dictionary_bits,
        rare_max_count=rare_max_count,
    )
    baseline_token_guard = _baseline_token_stream_guard(chunk_df, baseline_model=baseline_model)
    hardening_plans, hardening_diagnostics = _build_c3_hardening_plans(
        records,
        window_fallbacks=lz_window_fallbacks,
        max_span=lz_max_span,
        smoothing_alpha=smoothing_alpha,
        wide_sweep=c3_wide_sweep,
    )

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
    for plan in hardening_plans:
        report = _score_plan(
            plan,
            records,
            baseline_splits=baseline_splits,
            source_filters={
                "same_song_filtered_test": same_song_source_ids,
                "clean_trace_test": clean_trace_source_ids,
                "dirty_trace_test": test_source_ids & dirty_trace_source_ids,
            },
        )
        bucket_rows = _bucket_rows_for_plan(plan, records, baseline_splits, legal_codec=plan.legal_codec)
        report["bucket_rows"] = bucket_rows
        report["metadata"] = plan.metadata
        variant_reports[plan.name] = report
        for split, values in report["splits"].items():
            comparison_rows.append(_comparison_row(plan.name, split, values, legal_codec=plan.legal_codec, table="global"))
        comparison_rows.extend(bucket_rows)
        diagnostic_rows.extend(_usage_rows_for_plan(plan, records))
        diagnostic_rows.extend(_lz_span_diagnostic_rows(plan))

    baseline = variant_reports["b0_r0_delta"]
    selected_variant = _select_legal_variant(variant_reports)
    selected = variant_reports[selected_variant]
    reconstruction_guard = _c3_hardening_reconstruction_guard(
        records,
        hardening_plans,
        baseline_token_guard=baseline_token_guard,
    )
    bootstrap = _bootstrap_mapset_delta(
        selected,
        baseline,
        samples=bootstrap_samples,
        seed=random_seed,
    )
    pass_criteria = _c3_hardening_pass_criteria(
        selected_variant=selected_variant,
        selected=selected,
        baseline=baseline,
        variants=variant_reports,
        bootstrap=bootstrap,
        record_summary=record_summary,
        reconstruction_guard=reconstruction_guard,
    )
    recommendation = _c3_hardening_recommendation(pass_criteria, selected_variant=selected_variant)
    diagnostic_rows.extend(_c3_reconstruction_rows(reconstruction_guard))
    diagnostic_rows.extend(hardening_diagnostics)

    if comparison_csv_path is not None:
        comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)
    if diagnostics_csv_path is not None:
        diagnostics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(diagnostics_csv_path, index=False)

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "C3 LZ fallback-substream hardening and decomposition",
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
            "c3_wide_sweep": c3_wide_sweep,
            "legality_policy": (
                "baseline motif stream is unchanged; C3 is evaluated as a fallback-substream side codec. "
                "Mode, pointer, transform, residual, and learned table costs are charged."
            ),
        },
        "dataset": _dataset_summary(chunk_df),
        "fallback_record_summary": record_summary,
        "same_song_filter": {
            "conflict_key_count": len(conflict_keys),
            "test_source_count_after_filter": len(same_song_source_ids),
        },
        "active_hold_trace_filter": {
            "dirty_source_count": len(dirty_trace_source_ids),
            "clean_test_source_count": len(clean_trace_source_ids),
            "dirty_test_source_count": len(test_source_ids & dirty_trace_source_ids),
        },
        "baseline_model": {
            "learned_motif_count": baseline_model.learned_motif_count,
            "dictionary_cost_bits": baseline_model.dictionary_bits,
            "atom_vocab_size": len(baseline_model.atom_counter),
            "train_encoded_vocab_size": len(baseline_model.train_counter),
        },
        "variants": variant_reports,
        "selected_variant": selected_variant,
        "diagnostics": _c3_hardening_summary(hardening_plans, selected_variant=selected_variant),
        "bootstrap_mapset_delta": bootstrap,
        "reconstruction_guard": reconstruction_guard,
        "pass_criteria": pass_criteria,
        "recommendation": recommendation,
    }
    report = _normalize_json(report)
    if report_path is not None:
        _write_json(report_path, report)
    if result_log_path is not None:
        _write_c3_hardening_result_log(result_log_path, report)
    return report


def _build_c3_hardening_plans(
    records: Sequence[_FallbackRecord],
    *,
    window_fallbacks: int,
    max_span: int,
    smoothing_alpha: float,
    wide_sweep: bool,
) -> tuple[list[_CostPlan], list[dict[str, Any]]]:
    policies = _c3_hardening_policies(window_fallbacks=window_fallbacks, wide_sweep=wide_sweep)
    plans: list[_CostPlan] = []
    diagnostic_rows: list[dict[str, Any]] = []
    options_cache: dict[tuple[tuple[str, ...], int | None, str], dict[int, tuple[_LzCandidate, ...]]] = {}
    pointer_model_cache: dict[str, _C3PointerModel] = {}
    for policy in policies:
        cached_match_types = ("exact", "mirror", "skeleton")
        cache_key = (cached_match_types, policy.window_fallbacks, policy.phase_filter)
        options = options_cache.get(cache_key)
        if options is None:
            option_policy = _LzPolicy(
                name=f"{policy.name}_option_superset",
                label=policy.label,
                allowed_match_types=cached_match_types,
                activation=policy.activation,
                window_fallbacks=policy.window_fallbacks,
                pointer_code=policy.pointer_code,
                phase_filter=policy.phase_filter,
            )
            options = _lz_candidate_options_by_record(records, policy=option_policy, max_span=max_span)
            options_cache[cache_key] = options
        pointer_model = None
        table_model_cost_bits = policy.table_model_cost_bits
        if policy.pointer_code == "train_fitted":
            pointer_model = pointer_model_cache.get(policy.name)
            if pointer_model is None:
                pointer_model = _fit_c3_pointer_model(records, options, alpha=smoothing_alpha)
                pointer_model_cache[policy.name] = pointer_model
            table_model_cost_bits = pointer_model.table_cost_bits()
        plan = _build_lz_hardening_plan(
            records,
            options,
            policy=policy,
            max_span=max_span,
            pointer_model=pointer_model,
            table_model_cost_bits=table_model_cost_bits,
        )
        plans.append(plan)
        allowed_option_record_count = sum(
            1 for candidates in options.values() if any(candidate.match_type in policy.allowed_match_types for candidate in candidates)
        )
        diagnostic_rows.append(
            {
                "variant": plan.name,
                "split": "all",
                "table": "candidate_options",
                "bucket": "records_with_candidate",
                "legal_codec": True,
                "fallback_literal_count": allowed_option_record_count,
                "usage_share": float(allowed_option_record_count) / float(len(records)) if records else 0.0,
            }
        )
    return plans, diagnostic_rows


def _c3_hardening_policies(*, window_fallbacks: int, wide_sweep: bool) -> list[_LzPolicy]:
    base = int(window_fallbacks)
    policies = [
        _LzPolicy(
            name="c3_active_all_fixed_w256",
            label="current C3 strict mirror-order reproduction",
            allowed_match_types=("exact", "mirror", "skeleton"),
            activation="active",
            window_fallbacks=base,
            pointer_code="fixed_width",
        ),
        _LzPolicy(
            name="a1_exact_only_all_fallback_w256",
            label="A1 exact-only over all fallback literals",
            allowed_match_types=("exact",),
            activation="all",
            window_fallbacks=base,
            pointer_code="fixed_width",
        ),
        _LzPolicy(
            name="a2_skeleton_residual_all_fallback_w256",
            label="A2 skeleton match plus charged lane/order residual",
            allowed_match_types=("skeleton",),
            activation="all",
            window_fallbacks=base,
            pointer_code="fixed_width",
        ),
        _LzPolicy(
            name="a3_mirror_only_all_fallback_w256",
            label="A3 strict mirror-transform only",
            allowed_match_types=("mirror",),
            activation="all",
            window_fallbacks=base,
            pointer_code="fixed_width",
        ),
        _LzPolicy(
            name="a5_non_active_all_fixed_w256",
            label="A5 non-active fallback C3 control",
            allowed_match_types=("exact", "mirror", "skeleton"),
            activation="non_active",
            window_fallbacks=base,
            pointer_code="fixed_width",
        ),
    ]
    window_sweep = (128,) if not wide_sweep else (16, 32, 64, 128, 512, 1024)
    for window in window_sweep:
        policies.append(
            _LzPolicy(
                name=f"a6_active_all_fixed_w{window}",
                label=f"A6 active C3 window sweep {window}",
                allowed_match_types=("exact", "mirror", "skeleton"),
                activation="active",
                window_fallbacks=window,
                pointer_code="fixed_width",
            )
        )
    if wide_sweep:
        policies.append(
            _LzPolicy(
                name="a6_active_all_fixed_full_history",
                label="A6 active C3 full previous-chart fallback history",
                allowed_match_types=("exact", "mirror", "skeleton"),
                activation="active",
                window_fallbacks=None,
                pointer_code="fixed_width",
            )
        )
    pointer_sweep = ("elias_gamma", "power_bucket") if not wide_sweep else (
        "elias_gamma",
        "elias_delta",
        "power_bucket",
        "train_fitted",
        "online_adaptive",
    )
    for code in pointer_sweep:
        policies.append(
            _LzPolicy(
                name=f"a7_active_all_{code}_w256",
                label=f"A7 active C3 pointer code {code}",
                allowed_match_types=("exact", "mirror", "skeleton"),
                activation="active",
                window_fallbacks=base,
                pointer_code=code,
            )
        )
    if wide_sweep:
        for phase_filter in ("same_offset", "same_bar_phase"):
            policies.append(
                _LzPolicy(
                    name=f"a8_active_all_fixed_{phase_filter}_w256",
                    label=f"A8 active C3 fallback-source filter {phase_filter}",
                    allowed_match_types=("exact", "mirror", "skeleton"),
                    activation="active",
                    window_fallbacks=base,
                    pointer_code="fixed_width",
                    phase_filter=phase_filter,
                )
            )
    return policies


@dataclass
class _C3PointerModel:
    alpha: float
    distance: _CategoricalModel
    length: _CategoricalModel
    match_type: _CategoricalModel

    def add(self, candidate: _LzCandidate) -> None:
        self.distance.add((), str(candidate.distance))
        self.length.add((), str(candidate.length))
        self.match_type.add((), candidate.match_type)

    def bits(self, candidate: _LzCandidate) -> float:
        return (
            self.distance.bits((), str(candidate.distance))
            + self.length.bits((), str(candidate.length))
            + self.match_type.bits((), candidate.match_type)
        )

    def table_cost_bits(self) -> float:
        return self.distance.table_cost_bits() + self.length.table_cost_bits() + self.match_type.table_cost_bits()


def _empty_c3_pointer_model(alpha: float) -> _C3PointerModel:
    return _C3PointerModel(
        alpha=alpha,
        distance=_CategoricalModel(alpha=alpha),
        length=_CategoricalModel(alpha=alpha),
        match_type=_CategoricalModel(alpha=alpha),
    )


def _fit_c3_pointer_model(
    records: Sequence[_FallbackRecord],
    options_by_record: Mapping[int, Sequence[_LzCandidate]],
    *,
    alpha: float,
) -> _C3PointerModel:
    model = _empty_c3_pointer_model(alpha)
    for record in records:
        if record.split != "train":
            continue
        for candidate in options_by_record.get(record.id, ()):
            model.add(candidate)
    return model


def _lz_candidate_options_by_record(
    records: Sequence[_FallbackRecord],
    *,
    policy: _LzPolicy,
    max_span: int,
) -> dict[int, tuple[_LzCandidate, ...]]:
    by_source: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_row_index].append(record)
    options: dict[int, list[_LzCandidate]] = defaultdict(list)
    for source_records in by_source.values():
        source_records.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
        exact_index: dict[tuple[int, int, int, int, str], deque[int]] = defaultdict(deque)
        mirror_index: dict[tuple[int, int, int, int, str], deque[int]] = defaultdict(deque)
        skeleton_index: dict[tuple[int, str, int, int, int, str], deque[int]] = defaultdict(deque)
        for index, record in enumerate(source_records):
            candidate_indexes: list[tuple[str, int]] = []
            if "exact" in policy.allowed_match_types:
                candidate_indexes.extend(
                    ("exact", previous_index)
                    for previous_index in _recent_index_candidates_for_policy(
                        exact_index[_atom_key(record)],
                        index,
                        policy.window_fallbacks,
                    )
                )
            if "mirror" in policy.allowed_match_types:
                candidate_indexes.extend(
                    ("mirror", previous_index)
                    for previous_index in _recent_index_candidates_for_policy(
                        mirror_index[_atom_key(record)],
                        index,
                        policy.window_fallbacks,
                    )
                )
            if "skeleton" in policy.allowed_match_types:
                candidate_indexes.extend(
                    ("skeleton", previous_index)
                    for previous_index in _recent_index_candidates_for_policy(
                        skeleton_index[_skeleton_key(record)],
                        index,
                        policy.window_fallbacks,
                    )
                )
            for match_type, previous_index in candidate_indexes:
                previous = source_records[previous_index]
                if not _lz_phase_filter_ok(record, previous, policy.phase_filter):
                    continue
                distance = index - previous_index
                length, residual = _lz_match_length(source_records, index, previous_index, match_type=match_type, max_span=max_span)
                if length <= 0:
                    continue
                options[record.id].append(
                    _LzCandidate(
                        match_type=match_type,
                        length=length,
                        distance=distance,
                        payload_bits=0.0,
                        residual_bits=residual,
                    )
                )
            _append_index_candidate_for_policy(exact_index[_atom_key(record)], index, policy.window_fallbacks)
            _append_index_candidate_for_policy(mirror_index[_mirror_atom_key(record)], index, policy.window_fallbacks)
            _append_index_candidate_for_policy(skeleton_index[_skeleton_key(record)], index, policy.window_fallbacks)
    return {record_id: tuple(candidates) for record_id, candidates in options.items()}


def _recent_index_candidates_for_policy(indexes: deque[int], current_index: int, window_fallbacks: int | None) -> list[int]:
    if window_fallbacks is not None:
        lower = current_index - int(window_fallbacks)
        while indexes and indexes[0] < lower:
            indexes.popleft()
    return [value for value in list(indexes)[-DEFAULT_LZ_CANDIDATE_SCAN_LIMIT:] if value < current_index]


def _append_index_candidate_for_policy(indexes: deque[int], value: int, window_fallbacks: int | None) -> None:
    indexes.append(int(value))
    if window_fallbacks is None:
        return
    lower = int(value) - int(window_fallbacks)
    while indexes and indexes[0] < lower:
        indexes.popleft()


def _build_lz_hardening_plan(
    records: Sequence[_FallbackRecord],
    options_by_record: Mapping[int, Sequence[_LzCandidate]],
    *,
    policy: _LzPolicy,
    max_span: int,
    pointer_model: _C3PointerModel | None,
    table_model_cost_bits: float,
) -> _CostPlan:
    replacement = [record.raw_bits for record in records]
    payload = [record.raw_bits for record in records]
    mode = [0.0 for _ in records]
    modes = ["raw_literal" for _ in records]
    selections: list[_LzSpanSelection] = []
    reconstruction_mismatch_count = 0
    by_source: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_row_index].append(record)
    for source_records in by_source.values():
        source_records.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
        online_model = _empty_c3_pointer_model(0.1) if policy.pointer_code == "online_adaptive" else None
        index = 0
        while index < len(source_records):
            record = source_records[index]
            best: tuple[_LzCandidate, float] | None = None
            for raw_candidate in options_by_record.get(record.id, ()):
                if raw_candidate.match_type not in policy.allowed_match_types:
                    continue
                previous_index = index - raw_candidate.distance
                if previous_index < 0:
                    reconstruction_mismatch_count += 1
                    continue
                span = source_records[index : index + raw_candidate.length]
                if len(span) != raw_candidate.length or not _span_matches_activation(span, policy.activation):
                    continue
                pointer_bits = _c3_pointer_bits(
                    raw_candidate,
                    policy=policy,
                    max_span=max_span,
                    current_index=index,
                    pointer_model=pointer_model,
                    online_model=online_model,
                )
                candidate = _LzCandidate(
                    match_type=raw_candidate.match_type,
                    length=raw_candidate.length,
                    distance=raw_candidate.distance,
                    payload_bits=pointer_bits + raw_candidate.residual_bits,
                    residual_bits=raw_candidate.residual_bits,
                )
                raw_sum = sum(item.raw_bits for item in span)
                charged = candidate.payload_bits + MODE_BITS_4
                gain = raw_sum - charged
                if gain <= 0.0:
                    continue
                if best is None or gain > best[1]:
                    best = (candidate, gain)
            if best is None:
                index += 1
                continue
            candidate = best[0]
            previous_index = index - candidate.distance
            span = source_records[index : index + candidate.length]
            selections.append(
                _make_lz_span_selection(
                    source_records,
                    start_index=index,
                    previous_index=previous_index,
                    candidate=candidate,
                    mode_bits=MODE_BITS_4,
                )
            )
            if online_model is not None:
                online_model.add(candidate)
            per_record_payload = candidate.payload_bits / float(candidate.length)
            per_record_mode = MODE_BITS_4 / float(candidate.length)
            for item in span:
                replacement[item.id] = per_record_payload + per_record_mode
                payload[item.id] = per_record_payload
                mode[item.id] = per_record_mode
                modes[item.id] = f"lz_{candidate.match_type}"
            index += candidate.length
    return _CostPlan(
        name=policy.name,
        legal_codec=True,
        replacement_bits=tuple(replacement),
        payload_bits=tuple(payload),
        mode_header_bits=tuple(mode),
        mode_codes=tuple(modes),
        table_model_cost_bits=float(table_model_cost_bits),
        notes=policy.label,
        reconstruction_checked_count=len(records),
        reconstruction_mismatch_count=reconstruction_mismatch_count,
        lz_spans=tuple(selections),
        metadata={
            "allowed_match_types": list(policy.allowed_match_types),
            "activation": policy.activation,
            "window_fallbacks": "full_history" if policy.window_fallbacks is None else int(policy.window_fallbacks),
            "pointer_code": policy.pointer_code,
            "phase_filter": policy.phase_filter,
            "max_span": int(max_span),
        },
    )


def _span_matches_activation(records: Sequence[_FallbackRecord], activation: str) -> bool:
    if activation == "all":
        return True
    if activation == "active":
        return all(_active_span_record(record) for record in records)
    if activation == "non_active":
        return all(not _active_span_record(record) for record in records)
    raise ValueError(f"unknown C3 activation policy: {activation}")


def _lz_phase_filter_ok(current: _FallbackRecord, previous: _FallbackRecord, phase_filter: str) -> bool:
    if phase_filter == "none":
        return True
    if phase_filter == "same_offset":
        return current.offset_units == previous.offset_units
    if phase_filter == "same_bar_phase":
        return current.bar_phase_half == previous.bar_phase_half
    raise ValueError(f"unknown C3 phase filter: {phase_filter}")


def _c3_pointer_bits(
    candidate: _LzCandidate,
    *,
    policy: _LzPolicy,
    max_span: int,
    current_index: int,
    pointer_model: _C3PointerModel | None,
    online_model: _C3PointerModel | None,
) -> float:
    transform_bits = math.log2(float(len(policy.allowed_match_types))) if len(policy.allowed_match_types) > 1 else 0.0
    if policy.pointer_code == "fixed_width":
        distance_universe = int(policy.window_fallbacks) if policy.window_fallbacks is not None else max(1, int(current_index))
        return math.log2(float(distance_universe + 1)) + math.log2(float(max_span + 1)) + transform_bits
    if policy.pointer_code == "elias_gamma":
        return _elias_gamma_bits(candidate.distance) + _elias_gamma_bits(candidate.length) + transform_bits
    if policy.pointer_code == "elias_delta":
        return _elias_delta_bits(candidate.distance) + _elias_gamma_bits(candidate.length) + transform_bits
    if policy.pointer_code == "power_bucket":
        return _power_bucket_bits(candidate.distance) + _power_bucket_bits(candidate.length) + transform_bits
    if policy.pointer_code == "train_fitted":
        if pointer_model is None:
            raise ValueError("train_fitted pointer code requires a pointer_model")
        return pointer_model.bits(candidate)
    if policy.pointer_code == "online_adaptive":
        if online_model is None:
            raise ValueError("online_adaptive pointer code requires an online_model")
        return online_model.bits(candidate)
    raise ValueError(f"unknown C3 pointer code: {policy.pointer_code}")


def _elias_gamma_bits(value: int) -> float:
    value = max(1, int(value))
    log = int(math.floor(math.log2(value)))
    return float(2 * log + 1)


def _elias_delta_bits(value: int) -> float:
    value = max(1, int(value))
    log = int(math.floor(math.log2(value)))
    return _elias_gamma_bits(log + 1) + float(log)


def _power_bucket_bits(value: int) -> float:
    value = max(1, int(value))
    bucket = int(math.floor(math.log2(value)))
    return _elias_gamma_bits(bucket + 1) + float(bucket)


def _baseline_token_stream_guard(chunk_df: pd.DataFrame, *, baseline_model: Any) -> dict[str, Any]:
    id_to_motif = {token_id: motif for motif, token_id in baseline_model.vocab.motif_to_id.items()}
    mismatch_count = 0
    checked_chunk_count = 0
    examples: list[dict[str, Any]] = []
    for row in _iter_rows(chunk_df):
        expected = _baseline_tokens(row)
        encoded, _covered = _encode_sequence(expected, baseline_model.trie)
        decoded = _decode_motif_tokens(encoded, id_to_motif)
        checked_chunk_count += 1
        if decoded == expected:
            continue
        mismatch_count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "source_row_index": int(row.source_row_index),
                    "chunk_index": int(row.chunk_index),
                    "split": str(row.split),
                    "expected": expected[:20],
                    "decoded": decoded[:20],
                    "encoded": encoded[:20],
                }
            )
    return {
        "checked_chunk_count": checked_chunk_count,
        "mismatch_count": mismatch_count,
        "pass": mismatch_count == 0 and checked_chunk_count > 0,
        "examples": examples,
        "policy": "Baseline motif stream must decode to the original r0_delta chunk token stream before C3 side-stream substitution.",
    }


def _decode_motif_tokens(tokens: Sequence[str], id_to_motif: Mapping[str, tuple[str, ...]]) -> list[str]:
    decoded: list[str] = []
    for token in tokens:
        motif = id_to_motif.get(token)
        if motif is None:
            decoded.append(token)
        else:
            decoded.extend(motif)
    return decoded


def _c3_hardening_reconstruction_guard(
    records: Sequence[_FallbackRecord],
    plans: Sequence[_CostPlan],
    *,
    baseline_token_guard: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for plan in plans:
        fallback_guard = _decode_lz_fallback_payload_guard(records, plan)
        boundary = _lz_span_boundary_guard(records, plan)
        transform = _lz_transform_inverse_guard(records, plan)
        mismatch_count = (
            int(fallback_guard["mismatch_count"])
            + int(boundary["mismatch_count"])
            + int(transform["mismatch_count"])
            + int(baseline_token_guard.get("mismatch_count", 0) or 0)
        )
        rows.append(
            {
                "variant": plan.name,
                "fallback_payload_mismatch_count": int(fallback_guard["mismatch_count"]),
                "full_token_stream_mismatch_count": int(fallback_guard["mismatch_count"])
                + int(baseline_token_guard.get("mismatch_count", 0) or 0),
                "full_chart_mismatch_count": int(fallback_guard["mismatch_count"])
                + int(baseline_token_guard.get("mismatch_count", 0) or 0),
                "span_boundary_mismatch_count": int(boundary["mismatch_count"]),
                "transform_inverse_mismatch_count": int(transform["mismatch_count"]),
                "selected_span_count": len(plan.lz_spans),
                "selected_fallback_literal_count": sum(span.length for span in plan.lz_spans),
                "flat_main_stream_noncontiguous_span_count": int(boundary["flat_main_stream_noncontiguous_span_count"]),
                "target_cross_chunk_span_count": int(boundary["target_cross_chunk_span_count"]),
                "reference_cross_chunk_span_count": int(boundary["reference_cross_chunk_span_count"]),
                "target_cross_segment_span_count": int(boundary["target_cross_segment_span_count"]),
                "reference_cross_segment_span_count": int(boundary["reference_cross_segment_span_count"]),
                "mismatch_count": mismatch_count,
                "pass": mismatch_count == 0,
                "examples": [*fallback_guard["examples"], *boundary["examples"], *transform["examples"]][:10],
            }
        )
    legal_rows = rows
    mismatch_count = sum(int(row["mismatch_count"]) for row in legal_rows)
    return {
        "pass": all(bool(row["pass"]) for row in legal_rows) and bool(baseline_token_guard.get("pass")),
        "mismatch_count": mismatch_count,
        "baseline_token_stream_guard": baseline_token_guard,
        "rows": rows,
        "policy": (
            "C3 hardening uses a two-stream representation: the baseline motif stream remains exact, and selected C3 "
            "tokens reconstruct the fallback-payload side-stream from prior same-chart fallback records. "
            "Flat main-stream contiguity is reported separately because fallback-substream spans can skip motif-covered groups."
        ),
    }


def _decode_lz_fallback_payload_guard(records: Sequence[_FallbackRecord], plan: _CostPlan) -> dict[str, Any]:
    decoded = _decoded_lz_tokens_by_record(records, plan)
    mismatch_count = 0
    examples: list[dict[str, Any]] = []
    for record in records:
        actual = decoded.get(record.id, record.token)
        if actual == record.token:
            continue
        mismatch_count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "guard": "fallback_payload",
                    "variant": plan.name,
                    "record_id": record.id,
                    "source_row_index": record.source_row_index,
                    "expected": record.token,
                    "decoded": actual,
                }
            )
    return {"mismatch_count": mismatch_count, "examples": examples}


def _decoded_lz_tokens_by_record(records: Sequence[_FallbackRecord], plan: _CostPlan) -> dict[int, str]:
    decoded = {record.id: record.token for record in records}
    by_source: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_row_index].append(record)
    spans_by_source: dict[int, list[_LzSpanSelection]] = defaultdict(list)
    for span in plan.lz_spans:
        spans_by_source[span.source_row_index].append(span)
    for source_records in by_source.values():
        source_records.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
        id_to_index = {record.id: index for index, record in enumerate(source_records)}
        for span in spans_by_source.get(source_records[0].source_row_index, []):
            start = id_to_index.get(span.start_record_id)
            previous = id_to_index.get(span.previous_record_id)
            if start is None or previous is None:
                continue
            for offset in range(span.length):
                current_record = source_records[start + offset]
                previous_record = source_records[previous + offset]
                decoded[current_record.id] = _lz_transformed_token(previous_record, current_record, match_type=span.match_type)
    return decoded


def _lz_transformed_token(previous: _FallbackRecord, current: _FallbackRecord, *, match_type: str) -> str:
    if match_type == "exact":
        return previous.token
    if match_type == "mirror":
        return _record_token(
            delta=previous.delta,
            tap=_mirror_mask_4(previous.tap_mask),
            start=_mirror_mask_4(previous.ln_start_mask),
            end=_mirror_mask_4(previous.ln_end_mask),
            order_signature=_mirror_order_signature_4(previous.order_signature),
        )
    if match_type == "skeleton":
        return current.token
    raise ValueError(f"unknown LZ match type: {match_type}")


def _record_token(*, delta: int, tap: int, start: int, end: int, order_signature: str) -> str:
    token = f"D:{int(delta)}:{int(tap)}:{int(start)}:{int(end)}"
    order = str(order_signature or ".")
    return token if order == "." else f"{token}:O{order}"


def _lz_span_boundary_guard(records: Sequence[_FallbackRecord], plan: _CostPlan) -> dict[str, Any]:
    del records
    mismatch_count = 0
    occupied: set[tuple[int, int]] = set()
    examples: list[dict[str, Any]] = []
    noncontiguous = 0
    target_cross_chunk = 0
    reference_cross_chunk = 0
    target_cross_segment = 0
    reference_cross_segment = 0
    for span in plan.lz_spans:
        if not span.full_group_contiguous:
            noncontiguous += 1
        target_cross_chunk += int(span.target_crosses_chunk)
        reference_cross_chunk += int(span.reference_crosses_chunk)
        target_cross_segment += int(span.target_crosses_segment)
        reference_cross_segment += int(span.reference_crosses_segment)
        if span.previous_index < 0 or span.previous_index + span.length > span.start_index:
            mismatch_count += 1
            if len(examples) < 10:
                examples.append({"guard": "span_boundary", "variant": plan.name, "reason": "reference_not_prior", "span": span})
        for index in range(span.start_index, span.start_index + span.length):
            key = (span.source_row_index, index)
            if key in occupied:
                mismatch_count += 1
                if len(examples) < 10:
                    examples.append({"guard": "span_boundary", "variant": plan.name, "reason": "overlap", "span": span})
            occupied.add(key)
    return {
        "mismatch_count": mismatch_count,
        "flat_main_stream_noncontiguous_span_count": noncontiguous,
        "target_cross_chunk_span_count": target_cross_chunk,
        "reference_cross_chunk_span_count": reference_cross_chunk,
        "target_cross_segment_span_count": target_cross_segment,
        "reference_cross_segment_span_count": reference_cross_segment,
        "examples": _jsonable_span_examples(examples),
    }


def _lz_transform_inverse_guard(records: Sequence[_FallbackRecord], plan: _CostPlan) -> dict[str, Any]:
    by_source: dict[int, list[_FallbackRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_row_index].append(record)
    spans_by_source: dict[int, list[_LzSpanSelection]] = defaultdict(list)
    for span in plan.lz_spans:
        spans_by_source[span.source_row_index].append(span)
    mismatch_count = 0
    examples: list[dict[str, Any]] = []
    for source_records in by_source.values():
        source_records.sort(key=lambda item: (item.segment_id, item.absolute_units, item.chunk_index, item.group_index))
        id_to_index = {record.id: index for index, record in enumerate(source_records)}
        for span in spans_by_source.get(source_records[0].source_row_index, []):
            start = id_to_index.get(span.start_record_id)
            previous = id_to_index.get(span.previous_record_id)
            if start is None or previous is None:
                mismatch_count += 1
                continue
            for offset in range(span.length):
                current_record = source_records[start + offset]
                previous_record = source_records[previous + offset]
                decoded = _lz_transformed_token(previous_record, current_record, match_type=span.match_type)
                if decoded == current_record.token:
                    continue
                mismatch_count += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "guard": "transform_inverse",
                            "variant": plan.name,
                            "record_id": current_record.id,
                            "previous_record_id": previous_record.id,
                            "match_type": span.match_type,
                            "expected": current_record.token,
                            "decoded": decoded,
                        }
                    )
    return {"mismatch_count": mismatch_count, "examples": examples}


def _jsonable_span_examples(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for example in examples:
        value = dict(example)
        span = value.get("span")
        if isinstance(span, _LzSpanSelection):
            value["span"] = {
                "source_row_index": span.source_row_index,
                "start_record_id": span.start_record_id,
                "previous_record_id": span.previous_record_id,
                "length": span.length,
                "distance": span.distance,
                "match_type": span.match_type,
            }
        output.append(value)
    return output


def _lz_span_diagnostic_rows(plan: _CostPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not plan.lz_spans:
        return rows
    total_spans = len(plan.lz_spans)
    total_fallbacks = sum(span.length for span in plan.lz_spans)
    for table, bucket_func in (
        ("lz_span_match_type", lambda span: span.match_type),
        ("lz_span_length", lambda span: _count_bucket(span.length)),
        ("lz_span_distance", lambda span: _distance_bucket(span.distance)),
        ("lz_span_full_group_contiguous", lambda span: "yes" if span.full_group_contiguous else "no"),
        ("lz_span_target_cross_chunk", lambda span: "yes" if span.target_crosses_chunk else "no"),
    ):
        counter: Counter[str] = Counter(str(bucket_func(span)) for span in plan.lz_spans)
        for bucket, count in counter.most_common():
            span_fallbacks = sum(span.length for span in plan.lz_spans if str(bucket_func(span)) == bucket)
            rows.append(
                {
                    "variant": plan.name,
                    "split": "all",
                    "table": table,
                    "bucket": bucket,
                    "legal_codec": plan.legal_codec,
                    "span_count": int(count),
                    "span_share": float(count) / float(total_spans),
                    "fallback_literal_count": int(span_fallbacks),
                    "usage_share": float(span_fallbacks) / float(total_fallbacks) if total_fallbacks else 0.0,
                }
            )
    rows.append(
        {
            "variant": plan.name,
            "split": "all",
            "table": "lz_span_bits",
            "bucket": "mean",
            "legal_codec": plan.legal_codec,
            "span_count": total_spans,
            "fallback_literal_count": total_fallbacks,
            "mean_payload_bits": float(np.mean([span.payload_bits for span in plan.lz_spans])),
            "mean_mode_bits": float(np.mean([span.mode_bits for span in plan.lz_spans])),
            "mean_residual_bits": float(np.mean([span.residual_bits for span in plan.lz_spans])),
            "mean_raw_bits": float(np.mean([span.raw_bits for span in plan.lz_spans])),
            "mean_charged_bits": float(np.mean([span.charged_bits for span in plan.lz_spans])),
        }
    )
    return rows


def _distance_bucket(distance: int) -> str:
    value = int(distance)
    if value <= 16:
        return "1to16"
    if value <= 32:
        return "17to32"
    if value <= 64:
        return "33to64"
    if value <= 128:
        return "65to128"
    if value <= 256:
        return "129to256"
    if value <= 512:
        return "257to512"
    if value <= 1024:
        return "513to1024"
    return "1025plus"


def _c3_reconstruction_rows(reconstruction_guard: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in reconstruction_guard.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "variant": row.get("variant"),
                "split": "all",
                "table": "reconstruction_guard",
                "bucket": "mismatch",
                "legal_codec": True,
                "mismatch_count": row.get("mismatch_count"),
                "fallback_payload_mismatch_count": row.get("fallback_payload_mismatch_count"),
                "full_token_stream_mismatch_count": row.get("full_token_stream_mismatch_count"),
                "full_chart_mismatch_count": row.get("full_chart_mismatch_count"),
                "span_boundary_mismatch_count": row.get("span_boundary_mismatch_count"),
                "transform_inverse_mismatch_count": row.get("transform_inverse_mismatch_count"),
                "flat_main_stream_noncontiguous_span_count": row.get("flat_main_stream_noncontiguous_span_count"),
                "pass": row.get("pass"),
            }
        )
    return rows


def _c3_hardening_summary(plans: Sequence[_CostPlan], *, selected_variant: str) -> dict[str, Any]:
    selected_plan = next((plan for plan in plans if plan.name == selected_variant), None)
    return {
        "selected_variant": selected_variant,
        "selected_span_count": None if selected_plan is None else len(selected_plan.lz_spans),
        "selected_fallback_literal_count": None if selected_plan is None else sum(span.length for span in selected_plan.lz_spans),
        "selected_noncontiguous_main_stream_span_count": (
            None if selected_plan is None else sum(1 for span in selected_plan.lz_spans if not span.full_group_contiguous)
        ),
        "pipeline_representation_hint": (
            "C3 should enter the mapper pipeline as a fallback side-stream or grammar extension unless flat-contiguous spans dominate."
        ),
    }


def _c3_hardening_pass_criteria(
    *,
    selected_variant: str,
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    record_summary: Mapping[str, Any],
    reconstruction_guard: Mapping[str, Any],
) -> dict[str, Any]:
    selected_test = selected.get("splits", {}).get("test", {})
    selected_valid = selected.get("splits", {}).get("valid", {})
    baseline_test = baseline.get("splits", {}).get("test", {})
    baseline_filtered = baseline.get("splits", {}).get("same_song_filtered_test", {})
    selected_filtered = selected.get("splits", {}).get("same_song_filtered_test", {})
    baseline_clean = baseline.get("splits", {}).get("clean_trace_test", {})
    selected_clean = selected.get("splits", {}).get("clean_trace_test", {})
    baseline_dirty = baseline.get("splits", {}).get("dirty_trace_test", {})
    selected_dirty = selected.get("splits", {}).get("dirty_trace_test", {})
    selected_test_bits = float(selected_test.get("bits_per_event_with_dictionary", math.inf))
    selected_valid_bits = float(selected_valid.get("bits_per_event_with_dictionary", math.inf))
    baseline_test_bits = float(baseline_test.get("bits_per_event_with_dictionary", math.inf))
    baseline_consistency = abs(baseline_test_bits - EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT) < 0.05
    test_delta = selected_test_bits - baseline_test_bits
    filtered_delta = (
        float(selected_filtered.get("bits_per_event_with_dictionary", math.inf))
        - float(baseline_filtered.get("bits_per_event_with_dictionary", math.inf))
    )
    clean_trace_delta = (
        float(selected_clean.get("bits_per_event_with_dictionary", math.inf))
        - float(baseline_clean.get("bits_per_event_with_dictionary", math.inf))
    )
    dirty_trace_delta = (
        float(selected_dirty.get("bits_per_event_with_dictionary", math.inf))
        - float(baseline_dirty.get("bits_per_event_with_dictionary", math.inf))
    )
    exact_test_delta = _variant_test_delta(variants, "a1_exact_only_all_fallback_w256")
    window_128_delta = _variant_test_delta(variants, "a6_active_all_fixed_w128")
    window_256_delta = _variant_test_delta(variants, "c3_active_all_fixed_w256")
    bootstrap_below_zero = bool(
        bootstrap.get("available")
        and float(bootstrap.get("delta_bits_per_event_p975", math.inf)) < 0.0
    )
    hard_guard_pass = bool(reconstruction_guard.get("pass"))
    simple_family_pass = exact_test_delta <= -0.05 or _variant_test_delta(variants, "a3_mirror_only_all_fallback_w256") <= -0.05
    bounded_window_pass = window_128_delta <= -0.05 or window_256_delta <= -0.05
    promotion_pass = bool(
        baseline_consistency
        and hard_guard_pass
        and int(record_summary.get("fallback_literal_count", 0) or 0) > 0
        and test_delta <= -0.05
        and filtered_delta <= 0.0
        and clean_trace_delta <= -0.05
        and bootstrap_below_zero
        and bounded_window_pass
    )
    engineering_promotion_pass = bool(promotion_pass and test_delta <= -0.10 and simple_family_pass)
    return {
        "selected_variant": selected_variant,
        "baseline_consistency_pass": baseline_consistency,
        "expected_r0_test_charged_bits_per_event": EXPECTED_R0_TEST_CHARGED_BITS_PER_EVENT,
        "observed_r0_test_charged_bits_per_event": baseline_test_bits,
        "valid_selected_bits_per_event_with_dictionary": selected_valid_bits,
        "test_selected_bits_per_event_with_dictionary": selected_test_bits,
        "test_baseline_bits_per_event_with_dictionary": baseline_test_bits,
        "test_delta_bits_per_event_with_dictionary": test_delta,
        "same_song_filtered_delta_bits_per_event": filtered_delta,
        "same_song_filtered_pass": filtered_delta <= 0.0,
        "clean_trace_test_delta_bits_per_event": clean_trace_delta,
        "dirty_trace_test_delta_bits_per_event": dirty_trace_delta,
        "clean_trace_test_pass": clean_trace_delta <= -0.05,
        "bootstrap_95pct_below_zero_pass": bootstrap_below_zero,
        "hard_reconstruction_pass": hard_guard_pass,
        "exact_only_test_delta_bits_per_event": exact_test_delta,
        "simple_family_pass": simple_family_pass,
        "bounded_window_128_delta_bits_per_event": window_128_delta,
        "bounded_window_256_delta_bits_per_event": window_256_delta,
        "bounded_window_pass": bounded_window_pass,
        "research_promotion_pass": promotion_pass,
        "engineering_promotion_pass": engineering_promotion_pass,
    }


def _variant_test_delta(variants: Mapping[str, Mapping[str, Any]], variant: str) -> float:
    return float(
        variants.get(variant, {})
        .get("splits", {})
        .get("test", {})
        .get("delta_bits_per_event_with_dictionary", math.inf)
    )


def _c3_hardening_recommendation(pass_criteria: Mapping[str, Any], *, selected_variant: str) -> str:
    if pass_criteria.get("engineering_promotion_pass"):
        return (
            f"TEST_NEXT: {selected_variant} passed C3 hardening gates; build an optional pipeline-facing "
            "fallback side-stream tokenization artifact next."
        )
    if pass_criteria.get("research_promotion_pass"):
        return (
            f"MUTATE_OR_TEST_NEXT: {selected_variant} is a robust research result but lacks a simple-family "
            "engineering promotion signal; refine representation before mapper training."
        )
    if pass_criteria.get("hard_reconstruction_pass"):
        return f"MUTATE: {selected_variant} reconstructs legally but fails compression robustness gates."
    return f"DEFER: {selected_variant} failed a hard reconstruction or legality guard."


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
    split_names = list(dict.fromkeys(("train", "valid", "test", *source_filters.keys())))
    for split in split_names:
        source_filter = source_filters.get(split)
        baseline = baseline_splits.get(split, {})
        source_split = split if split in {"train", "valid", "test"} else "test"
        selected_records = [
            record
            for record in records
            if record.split == source_split
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
        _mirror_order_signature_4(record.order_signature),
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


def _mirror_order_signature_4(order_signature: str) -> str:
    value = str(order_signature or ".")
    if value == ".":
        return "."
    mirrored: list[str] = []
    for part in value.split(","):
        if len(part) < 2:
            return value
        action = part[0]
        lane_text = part[1:]
        if not lane_text.isdigit():
            return value
        lane = int(lane_text)
        if not 0 <= lane < 4:
            return value
        mirrored.append(f"{action}{3 - lane}")
    return ",".join(mirrored)


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


def _write_c3_hardening_result_log(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_name = str(report.get("selected_variant"))
    selected = report.get("variants", {}).get(selected_name, {})
    baseline = report.get("variants", {}).get("b0_r0_delta", {})
    pass_criteria = report.get("pass_criteria", {})
    reconstruction_guard = report.get("reconstruction_guard", {})
    diagnostics = report.get("diagnostics", {})
    lines = [
        "# C3 LZ Fallback-Substream Hardening Result Log",
        "",
        "## Summary",
        "",
        f"- Selected variant: `{selected_name}`",
        f"- Recommendation: {report.get('recommendation')}",
        f"- Research promotion pass: {pass_criteria.get('research_promotion_pass')}",
        f"- Engineering promotion pass: {pass_criteria.get('engineering_promotion_pass')}",
        f"- Runtime seconds: {_fmt_float(report.get('elapsed_s'))}",
        f"- Limited: {report.get('limited')}",
        f"- Code dirty: {report.get('code_dirty')}",
        f"- Hard reconstruction pass: {reconstruction_guard.get('pass') if isinstance(reconstruction_guard, Mapping) else None}",
        f"- Hard reconstruction mismatches: {reconstruction_guard.get('mismatch_count') if isinstance(reconstruction_guard, Mapping) else None}",
        f"- Baseline charged test bits/event: {_nested(baseline, 'splits', 'test', 'bits_per_event_with_dictionary')}",
        f"- Selected charged test bits/event: {_nested(selected, 'splits', 'test', 'bits_per_event_with_dictionary')}",
        f"- Selected test delta bits/event: {pass_criteria.get('test_delta_bits_per_event_with_dictionary')}",
        f"- Same-song filtered delta: {pass_criteria.get('same_song_filtered_delta_bits_per_event')}",
        f"- Clean-trace test delta: {pass_criteria.get('clean_trace_test_delta_bits_per_event')}",
        f"- Dirty-trace test delta: {pass_criteria.get('dirty_trace_test_delta_bits_per_event')}",
        f"- Bootstrap delta 95% interval: [{_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_p025')}, {_nested(report, 'bootstrap_mapset_delta', 'delta_bits_per_event_p975')}]",
        f"- Selected spans: {diagnostics.get('selected_span_count') if isinstance(diagnostics, Mapping) else None}",
        f"- Selected fallback literals: {diagnostics.get('selected_fallback_literal_count') if isinstance(diagnostics, Mapping) else None}",
        f"- Selected noncontiguous main-stream spans: {diagnostics.get('selected_noncontiguous_main_stream_span_count') if isinstance(diagnostics, Mapping) else None}",
        "",
        "## Variant Comparison",
        "",
        "| variant | valid charged | test charged | test delta | fallback payload delta | table/model bits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant_name, variant in report.get("variants", {}).items():
        if variant_name == "b0_r0_delta":
            continue
        valid = variant.get("splits", {}).get("valid", {}) if isinstance(variant, Mapping) else {}
        test = variant.get("splits", {}).get("test", {}) if isinstance(variant, Mapping) else {}
        lines.append(
            "| {variant} | {valid} | {test} | {delta} | {fallback_delta} | {table_bits} |".format(
                variant=variant_name,
                valid=_fmt_float(valid.get("bits_per_event_with_dictionary") if isinstance(valid, Mapping) else None),
                test=_fmt_float(test.get("bits_per_event_with_dictionary") if isinstance(test, Mapping) else None),
                delta=_fmt_float(test.get("delta_bits_per_event_with_dictionary") if isinstance(test, Mapping) else 0.0),
                fallback_delta=_fmt_float(test.get("fallback_payload_delta_bits_per_fallback_event") if isinstance(test, Mapping) else 0.0),
                table_bits=_fmt_float(variant.get("table_model_cost_bits") if isinstance(variant, Mapping) else None),
            )
        )
    lines.extend(["", "## Reconstruction Guards", ""])
    for row in reconstruction_guard.get("rows", []) if isinstance(reconstruction_guard, Mapping) else []:
        lines.append(
            "- `{variant}` pass={passed}, fallback={fallback}, stream={stream}, chart={chart}, "
            "boundary={boundary}, transform={transform}, noncontiguous={noncontiguous}".format(
                variant=row.get("variant"),
                passed=row.get("pass"),
                fallback=row.get("fallback_payload_mismatch_count"),
                stream=row.get("full_token_stream_mismatch_count"),
                chart=row.get("full_chart_mismatch_count"),
                boundary=row.get("span_boundary_mismatch_count"),
                transform=row.get("transform_inverse_mismatch_count"),
                noncontiguous=row.get("flat_main_stream_noncontiguous_span_count"),
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
    parser.add_argument("--c3-hardening", action="store_true", help="run the C3 hardening/decomposition audit")
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
    parser.add_argument("--c3-wide-sweep", action="store_true", help="include wider C3 window/pointer/source ablations")
    parser.add_argument("--patch-max-span", type=int, default=DEFAULT_PATCH_MAX_SPAN)
    parser.add_argument("--top-example-limit", type=int, default=20)
    parser.add_argument("--limit-chunks", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.c3_hardening:
        report_path = DEFAULT_C3_HARDENING_REPORT_PATH if args.report_path == DEFAULT_REPORT_PATH else args.report_path
        result_log_path = (
            DEFAULT_C3_HARDENING_RESULT_LOG_PATH if args.result_log_path == DEFAULT_RESULT_LOG_PATH else args.result_log_path
        )
        comparison_csv_path = (
            DEFAULT_C3_HARDENING_COMPARISON_CSV_PATH
            if args.comparison_csv_path == DEFAULT_COMPARISON_CSV_PATH
            else args.comparison_csv_path
        )
        diagnostics_csv_path = (
            DEFAULT_C3_HARDENING_DIAGNOSTICS_CSV_PATH
            if args.diagnostics_csv_path == DEFAULT_DIAGNOSTICS_CSV_PATH
            else args.diagnostics_csv_path
        )
        report = audit_c3_lz_hardening(
            chunk_cache_path=args.chunk_cache_path,
            report_path=report_path,
            result_log_path=result_log_path,
            comparison_csv_path=comparison_csv_path,
            diagnostics_csv_path=diagnostics_csv_path,
            motif_vocab_size=args.motif_vocab_size,
            motif_min_n=args.motif_min_n,
            motif_max_n=args.motif_max_n,
            smoothing_alpha=args.smoothing_alpha,
            rare_max_count=args.rare_max_count,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed,
            lz_window_fallbacks=args.lz_window_fallbacks,
            lz_max_span=args.lz_max_span,
            c3_wide_sweep=args.c3_wide_sweep,
            top_example_limit=args.top_example_limit,
            limit_chunks=args.limit_chunks,
            command=_command(sys.argv[1:]),
        )
        print(
            "c3_lz_hardening_audit "
            f"research_promotion={report.get('pass_criteria', {}).get('research_promotion_pass')} "
            f"engineering_promotion={report.get('pass_criteria', {}).get('engineering_promotion_pass')} "
            f"selected={report.get('selected_variant')} "
            f"recommendation={report.get('recommendation')} "
            f"report={report.get('report_path')}"
        )
        return 0
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
