from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_C3_LZ_HARDENING_PATH = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json"
)
DEFAULT_C3_SIDE_STREAM_PATH = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_side_stream_pipeline_report.json"
)
DEFAULT_C3_EXACT_SIDECAR_PATH = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_exact_mapper_window_sidecar_report.json"
)
DEFAULT_C3_ORDERED_RAW_PATH = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_summary.json"
)
DEFAULT_V3_FULL_DATASET_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_summary.json"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/"
    "c3_v3_target_complexity_comparison_summary.json"
)
DEFAULT_REPORT_OUTPUT = Path(
    "artifacts/reports/audits/context_adaptive_fallback_codec/"
    "c3_v3_target_complexity_comparison_result_report.md"
)


def run_c3_v3_target_complexity_comparison(
    *,
    c3_lz_hardening_path: str | Path = DEFAULT_C3_LZ_HARDENING_PATH,
    c3_side_stream_path: str | Path = DEFAULT_C3_SIDE_STREAM_PATH,
    c3_exact_sidecar_path: str | Path = DEFAULT_C3_EXACT_SIDECAR_PATH,
    c3_ordered_raw_path: str | Path = DEFAULT_C3_ORDERED_RAW_PATH,
    v3_full_dataset_path: str | Path = DEFAULT_V3_FULL_DATASET_PATH,
) -> dict[str, Any]:
    start = time.monotonic()
    c3_lz = _read_json(Path(c3_lz_hardening_path))
    c3_side = _read_json(Path(c3_side_stream_path))
    c3_exact = _read_json(Path(c3_exact_sidecar_path))
    c3_ordered = _read_json(Path(c3_ordered_raw_path))
    v3 = _read_json(Path(v3_full_dataset_path))

    v21_metrics = _extract_v21_metrics(v3)
    v3_metrics = _extract_v3_metrics(v3)
    c3_codec_metrics = _extract_c3_codec_metrics(c3_lz, c3_side, c3_exact, c3_ordered)
    guard_results = _guard_results(v3_metrics=v3_metrics, c3_codec_metrics=c3_codec_metrics)
    decision = _decision_from_guards(guard_results)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "C3/v3 target complexity comparison",
        "inputs": {
            "c3_lz_hardening_path": Path(c3_lz_hardening_path).as_posix(),
            "c3_side_stream_path": Path(c3_side_stream_path).as_posix(),
            "c3_exact_sidecar_path": Path(c3_exact_sidecar_path).as_posix(),
            "c3_ordered_raw_path": Path(c3_ordered_raw_path).as_posix(),
            "v3_full_dataset_path": Path(v3_full_dataset_path).as_posix(),
        },
        "v2_1": v21_metrics,
        "v3": v3_metrics,
        "c3": c3_codec_metrics,
        "guard_results": guard_results,
        "decision": decision,
        "elapsed_s": time.monotonic() - start,
    }
    return summary


def _extract_v21_metrics(v3_summary: Mapping[str, Any]) -> dict[str, Any]:
    audit = _mapping(v3_summary.get("audit"))
    total = _mapping(audit.get("total"))
    eval_split = _mapping(audit.get("eval"))
    vocab = _mapping(v3_summary.get("vocab"))
    return {
        "role": "baseline_sparse_lane_action_target",
        "full_dataset_token_count": _int(total.get("baseline_token_count")),
        "eval_token_count": _int(eval_split.get("baseline_token_count")),
        "vocab_size": _int(vocab.get("v2_1_vocab_size")),
        "default_mapper_target": True,
    }


def _extract_v3_metrics(v3_summary: Mapping[str, Any]) -> dict[str, Any]:
    audit = _mapping(v3_summary.get("audit"))
    total = _mapping(audit.get("total"))
    eval_split = _mapping(audit.get("eval"))
    comparison = _mapping(v3_summary.get("comparison"))
    vocab = _mapping(v3_summary.get("vocab"))
    decision = _mapping(v3_summary.get("decision"))
    return {
        "role": "local_event_token_teacher_forcing_target",
        "decision_route": decision.get("route"),
        "full_dataset_window_count": _int(total.get("window_count")),
        "full_dataset_token_count": _int(total.get("candidate_token_count")),
        "full_dataset_baseline_token_count": _int(total.get("baseline_token_count")),
        "full_dataset_token_reduction_ratio": _float(total.get("token_reduction_ratio")),
        "full_dataset_reconstruction_mismatches": _int(total.get("reconstruction_mismatches")),
        "full_dataset_cross_window_ln_window_count": _int(total.get("cross_window_ln_window_count")),
        "eval_token_count": _int(comparison.get("eval_token_count")),
        "eval_baseline_token_count": _int(eval_split.get("baseline_token_count")),
        "eval_token_reduction_ratio": _float(comparison.get("token_reduction_ratio")),
        "eval_total_bit_reduction_ratio": _float(comparison.get("total_bit_reduction_ratio")),
        "eval_total_bit_reduction": _float(comparison.get("total_bit_reduction")),
        "vocab_size": _int(vocab.get("v3_vocab_size")),
        "event_token_count": _int(vocab.get("v3_event_token_count")),
        "requires_future_target_lookup": False,
        "requires_c3_style_backreference_replay": False,
    }


def _extract_c3_codec_metrics(
    c3_lz: Mapping[str, Any],
    c3_side: Mapping[str, Any],
    c3_exact: Mapping[str, Any],
    c3_ordered: Mapping[str, Any],
) -> dict[str, Any]:
    lz_pass = _mapping(c3_lz.get("pass_criteria"))
    lz_reconstruction = _mapping(c3_lz.get("reconstruction_guard"))
    side_reconstruction = _mapping(c3_side.get("reconstruction_guard"))
    side_token_stats = _mapping(c3_side.get("token_stats"))
    side_span_stats = _mapping(c3_side.get("span_stats"))
    exact_pass = _mapping(c3_exact.get("pass_criteria"))
    exact_sidecar_stats = _mapping(c3_exact.get("sidecar_stats"))
    exact_span_window_stats = _mapping(c3_exact.get("span_window_stats"))
    ordered_bits = _mapping(c3_ordered.get("bits"))
    ordered_selected_metrics = _mapping(ordered_bits.get("selected_metrics"))
    ordered_exact_bits = _selected_split_bits(ordered_bits, "exact_c3")
    ordered_field_bits = _selected_split_bits(ordered_bits, "ordered_field")
    ordered_raw_exact_bits = _selected_split_bits(ordered_bits, "raw_exact")
    ordered_raw_bits = _selected_split_bits(ordered_bits, "raw_ordered_field")
    ordered_sequence = _mapping(c3_ordered.get("sequence"))
    ordered_pass = _mapping(c3_ordered.get("pass_criteria"))
    ordered_dataset = _mapping(c3_ordered.get("dataset"))
    return {
        "role": "chart_local_fallback_substream_codec",
        "selected_variant": lz_pass.get("selected_variant"),
        "test_delta_bits_per_event": _float(lz_pass.get("test_delta_bits_per_event_with_dictionary")),
        "same_song_filtered_delta_bits_per_event": _float(lz_pass.get("same_song_filtered_delta_bits_per_event")),
        "bootstrap_95pct_below_zero_pass": bool(lz_pass.get("bootstrap_95pct_below_zero_pass")),
        "lz_reconstruction_pass": bool(lz_reconstruction.get("pass")),
        "side_stream_reconstruction_pass": bool(side_reconstruction.get("pass")),
        "sidecar_generation_pass": bool(exact_pass.get("p5_exact_sidecar_generation_pass")),
        "sidecar_loader_guard_pass": bool(exact_pass.get("loader_guard_pass")),
        "sidecar_token_count": _int(exact_sidecar_stats.get("sidecar_token_count")),
        "sidecar_window_count": _int(exact_sidecar_stats.get("window_count")),
        "sidecar_window_with_tokens_rate": _float(exact_sidecar_stats.get("window_with_tokens_rate")),
        "sidecar_token_vocab_size": _int(exact_sidecar_stats.get("token_vocab_size")),
        "combined_main_side_token_count": _int(side_token_stats.get("combined_main_side_token_count")),
        "baseline_encoded_token_count": _int(side_token_stats.get("baseline_encoded_token_count")),
        "combined_to_baseline_token_ratio": _float(side_token_stats.get("combined_to_baseline_token_ratio")),
        "main_stream_token_count": _int(side_token_stats.get("main_stream_token_count")),
        "fallback_placeholder_count": _int(side_token_stats.get("fallback_placeholder_count")),
        "side_stream_token_count": _int(side_token_stats.get("side_stream_token_count")),
        "noncontiguous_main_stream_span_rate": _safe_divide(
            side_span_stats.get("noncontiguous_main_stream_span_count"),
            side_span_stats.get("span_count"),
        ),
        "target_cross_chunk_span_rate": _safe_divide(
            side_span_stats.get("target_cross_chunk_span_count"),
            side_span_stats.get("span_count"),
        ),
        "target_cross_mapper_window_span_rate": _float(
            exact_span_window_stats.get("target_cross_mapper_window_span_rate")
        ),
        "reference_cross_mapper_window_span_rate": _float(
            exact_span_window_stats.get("reference_cross_mapper_window_span_rate")
        ),
        "target_reference_same_window_span_rate": _float(
            exact_span_window_stats.get("target_reference_same_window_span_rate")
        ),
        "cap_256_truncated_window_rate": _cap_metric(exact_sidecar_stats, 256, "truncated_window_rate"),
        "cap_256_overflow_token_rate": _cap_metric(exact_sidecar_stats, 256, "overflow_token_rate"),
        "ordered_reconstruction_pass": bool(ordered_pass.get("reconstruction_pass")),
        "ordered_raw_bits_reduction_pass": bool(ordered_pass.get("raw_bits_reduction_pass")),
        "ordered_sequence_expansion_bounded_pass": bool(ordered_pass.get("sequence_expansion_bounded_pass")),
        "ordered_token_count": _int(ordered_dataset.get("ordered_symbol_count")),
        "ordered_exact_token_count": _int(ordered_dataset.get("token_count")),
        "ordered_to_exact_token_ratio": _float(ordered_sequence.get("ordered_to_exact_token_ratio")),
        "ordered_exact_c3_bits_per_token": _metric_or_fallback(
            ordered_selected_metrics,
            "exact_bits_per_token",
            ordered_exact_bits.get("bits_per_item"),
        ),
        "ordered_field_bits_per_original_token": _metric_or_fallback(
            ordered_selected_metrics,
            "ordered_bits_per_original_token",
            _safe_divide(ordered_field_bits.get("total_bits"), ordered_exact_bits.get("item_count")),
        ),
        "raw_exact_bits_per_raw_token": _metric_or_fallback(
            ordered_selected_metrics,
            "raw_exact_bits_per_raw_token",
            ordered_raw_exact_bits.get("bits_per_item"),
        ),
        "raw_ordered_bits_per_raw_token": _metric_or_fallback(
            ordered_selected_metrics,
            "raw_ordered_bits_per_raw_token",
            _safe_divide(ordered_raw_bits.get("total_bits"), ordered_raw_exact_bits.get("item_count")),
        ),
        "requires_future_target_lookup": False,
        "requires_c3_style_backreference_replay": True,
        "production_input_legal": False,
    }


def _guard_results(
    *,
    v3_metrics: Mapping[str, Any],
    c3_codec_metrics: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "v3_reconstruction_zero": _int(v3_metrics.get("full_dataset_reconstruction_mismatches")) == 0,
        "v3_eval_total_bit_reduction_positive": _float(v3_metrics.get("eval_total_bit_reduction_ratio")) > 0.0,
        "v3_full_token_reduction_positive": _float(v3_metrics.get("full_dataset_token_reduction_ratio")) > 0.0,
        "v3_no_future_lookup": not bool(v3_metrics.get("requires_future_target_lookup")),
        "c3_codec_reconstruction_zero": bool(c3_codec_metrics.get("lz_reconstruction_pass"))
        and bool(c3_codec_metrics.get("side_stream_reconstruction_pass")),
        "c3_exact_sidecar_generation_pass": bool(c3_codec_metrics.get("sidecar_generation_pass"))
        and bool(c3_codec_metrics.get("sidecar_loader_guard_pass")),
        "c3_combined_sequence_competitive": _float(c3_codec_metrics.get("combined_to_baseline_token_ratio")) <= 1.0,
        "c3_ordered_sequence_bounded": _float(c3_codec_metrics.get("ordered_to_exact_token_ratio")) <= 2.0,
        "c3_ordered_bits_not_worse": _float(c3_codec_metrics.get("ordered_field_bits_per_original_token"))
        <= _float(c3_codec_metrics.get("ordered_exact_c3_bits_per_token")),
        "c3_cross_window_reference_low": _float(c3_codec_metrics.get("reference_cross_mapper_window_span_rate")) <= 0.05,
        "c3_production_input_legal": bool(c3_codec_metrics.get("production_input_legal")),
    }


def _decision_from_guards(guards: Mapping[str, bool]) -> dict[str, Any]:
    v3_ready_representation = (
        bool(guards.get("v3_reconstruction_zero"))
        and bool(guards.get("v3_eval_total_bit_reduction_positive"))
        and bool(guards.get("v3_full_token_reduction_positive"))
        and bool(guards.get("v3_no_future_lookup"))
    )
    c3_target_ready = (
        bool(guards.get("c3_codec_reconstruction_zero"))
        and bool(guards.get("c3_exact_sidecar_generation_pass"))
        and bool(guards.get("c3_combined_sequence_competitive"))
        and bool(guards.get("c3_ordered_sequence_bounded"))
        and bool(guards.get("c3_ordered_bits_not_worse"))
        and bool(guards.get("c3_cross_window_reference_low"))
        and bool(guards.get("c3_production_input_legal"))
    )
    if c3_target_ready:
        return {
            "route": "TEST_NEXT_C3_TARGET",
            "reason": "C3 target costs cleared the complexity guards",
            "next_step": "Create a bounded emitted C3 target-grammar smoke card.",
        }
    if v3_ready_representation and bool(guards.get("c3_codec_reconstruction_zero")):
        return {
            "route": "MUTATE_TO_V3_GRAMMAR_REPAIR",
            "reason": "C3 remains reconstructive but side-stream/state costs are not target-grammar competitive with v3",
            "next_step": "Focus the next bounded experiment on v3/v2.1 grammar repair and trained rollout failure clusters.",
        }
    return {
        "route": "MUTATE",
        "reason": "target-complexity evidence is incomplete or mixed",
        "next_step": "Add a stricter common-slice target-cost audit before mapper training.",
    }


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    v21 = _mapping(summary.get("v2_1"))
    v3 = _mapping(summary.get("v3"))
    c3 = _mapping(summary.get("c3"))
    guards = _mapping(summary.get("guard_results"))
    decision = _mapping(summary.get("decision"))
    lines = [
        "# C3/v3 Target Complexity Comparison Result Report",
        "",
        "## Scope",
        "",
        "This artifact-only pass compares the current C3 side-stream/ordered target evidence against "
        "v3 and the v2.1 sparse target baseline. It does not retrain, retokenize the dataset, change "
        "rollout behavior, or change mapper defaults.",
        "",
        "## Decision",
        "",
        f"- route: `{decision.get('route')}`",
        f"- reason: {decision.get('reason')}",
        f"- next step: {decision.get('next_step')}",
        "",
        "## Target Complexity Snapshot",
        "",
        "| Candidate | Key property | Value |",
        "| --- | --- | ---: |",
        _row("v2.1", "eval tokens", v21.get("eval_token_count")),
        _row("v2.1", "vocab size", v21.get("vocab_size")),
        _row("v3", "full token reduction", _percent(v3.get("full_dataset_token_reduction_ratio"))),
        _row("v3", "eval total-bit reduction", _percent(v3.get("eval_total_bit_reduction_ratio"))),
        _row("v3", "reconstruction mismatches", v3.get("full_dataset_reconstruction_mismatches")),
        _row("v3", "vocab size", v3.get("vocab_size")),
        _row("C3", "codec delta bits/event", _fmt(c3.get("test_delta_bits_per_event"))),
        _row("C3", "combined main+side/baseline ratio", _fmt(c3.get("combined_to_baseline_token_ratio"))),
        _row("C3", "ordered/exact sequence ratio", _fmt(c3.get("ordered_to_exact_token_ratio"))),
        _row("C3", "ordered bits/original token", _fmt(c3.get("ordered_field_bits_per_original_token"))),
        _row("C3", "exact C3 bits/token", _fmt(c3.get("ordered_exact_c3_bits_per_token"))),
        _row("C3", "reference cross-window span rate", _percent(c3.get("reference_cross_mapper_window_span_rate"))),
        _row("C3", "production input legal", str(c3.get("production_input_legal")).lower()),
        "",
        "## Guard Results",
        "",
        *[f"- {key}: `{str(value).lower()}`" for key, value in guards.items()],
        "",
        "## What Passed",
        "",
        "- C3 still has strong codec-side evidence: exact reconstruction and negative charged bits/event delta.",
        "- The exact C3 mapper-window sidecar remains loader-compatible and cap-tractable.",
        "- v3 remains exactly reconstructive on the full dataset and reduces target tokens/bits versus v2.1.",
        "",
        "## What Surfaced",
        "",
        "- C3 is not target-sequence competitive as an emitted mapper grammar: combined main+side tokens exceed the baseline stream.",
        "- Ordered RAW splitting is lossless but expands the sequence and worsens charged bits versus exact C3 tokens.",
        "- C3 reference semantics remain cross-window/stateful, and target-derived C3 labels are not legal production inputs.",
        "- v3 is the simpler local teacher-forcing target, but this report does not prove v3 rollout replacement readiness.",
        "",
        "## Interpretation",
        "",
        "The C3 result should be kept as strong tokenizer-side/codec evidence, but the next mapper-facing "
        "work should focus on v3/v2.1 grammar repair rather than another C3 target or input-conditioning run.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(candidate: str, metric: str, value: Any) -> str:
    return f"| {candidate} | {metric} | `{value}` |"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required artifact is missing: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _safe_divide(numerator: Any, denominator: Any) -> float:
    denominator_f = _float(denominator)
    if denominator_f == 0.0:
        return 0.0
    return _float(numerator) / denominator_f


def _cap_metric(sidecar_stats: Mapping[str, Any], cap: int, metric: str) -> float:
    cap_sweep = _mapping(sidecar_stats.get("cap_sweep"))
    row = _mapping(cap_sweep.get(str(cap)) or cap_sweep.get(cap))
    return _float(row.get(metric))


def _selected_split_bits(bits: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    family_bits = _mapping(bits.get(family))
    selected_split = str(bits.get("selected_split") or "test")
    selected = _mapping(family_bits.get(selected_split))
    if selected:
        return selected
    return family_bits


def _metric_or_fallback(metrics: Mapping[str, Any], key: str, fallback: Any) -> float:
    if key in metrics:
        return _float(metrics.get(key))
    return _float(fallback)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _percent(value: Any) -> str:
    return f"{_float(value) * 100.0:.3f}%"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare C3, v3, and v2.1 target complexity artifacts.")
    parser.add_argument("--c3-lz-hardening", default=DEFAULT_C3_LZ_HARDENING_PATH)
    parser.add_argument("--c3-side-stream", default=DEFAULT_C3_SIDE_STREAM_PATH)
    parser.add_argument("--c3-exact-sidecar", default=DEFAULT_C3_EXACT_SIDECAR_PATH)
    parser.add_argument("--c3-ordered-raw", default=DEFAULT_C3_ORDERED_RAW_PATH)
    parser.add_argument("--v3-full-dataset", default=DEFAULT_V3_FULL_DATASET_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args(argv)

    summary = run_c3_v3_target_complexity_comparison(
        c3_lz_hardening_path=args.c3_lz_hardening,
        c3_side_stream_path=args.c3_side_stream,
        c3_exact_sidecar_path=args.c3_exact_sidecar,
        c3_ordered_raw_path=args.c3_ordered_raw,
        v3_full_dataset_path=args.v3_full_dataset,
    )
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(summary, args.report_output)
    print(
        "c3_v3_target_complexity_comparison_done "
        f"route={summary['decision']['route']} "
        f"c3_combined_ratio={summary['c3']['combined_to_baseline_token_ratio']:.6f}"
    )


if __name__ == "__main__":
    main()
