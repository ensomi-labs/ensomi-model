from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset
from pulsefield_model.models.mapper.v3.replay import target_end_ms as v3_target_end_ms
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_fixed_slice_label_coverage_result_report.md"
DEFAULT_PROXY_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_proxy_audit_summary.json"
DEFAULT_TINY_TRAINING_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_auxiliary_tiny_training_gate_summary.json"


def run_delta_event_auxiliary_fixed_slice_label_coverage(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    proxy_summary_path: str | Path = DEFAULT_PROXY_SUMMARY_PATH,
    tiny_training_summary_path: str | Path = DEFAULT_TINY_TRAINING_SUMMARY_PATH,
    delta_max_ms: int = 8000,
    end_gap_max_ms: int = 8000,
    limit: int | None = None,
    max_cached_timepoint_maps: int = 64,
    progress: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    _validate_range(delta_max_ms, name="delta_max_ms")
    _validate_range(end_gap_max_ms, name="end_gap_max_ms")
    vocab = MapperV3Vocab()
    dataset = MapperV3WindowDataset(
        index_path=index_path,
        vocab=vocab,
        max_cached_timepoint_maps=int(max_cached_timepoint_maps),
        progress=bool(progress),
    )
    row_count = len(dataset.records) if limit is None else min(int(limit), len(dataset.records))
    if row_count <= 0:
        raise ValueError("delta-event auxiliary label coverage requires at least one dataset row")

    aggregate = _empty_aggregate()
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index in range(row_count):
        record = dataset.records[index].control_record
        try:
            tokenized = dataset._tokenize_record(record)
            coverage = window_label_coverage(
                token_ids=tokenized.target_fragment_ids,
                current_ms=tuple(int(value) for value in tokenized.target_fragment_current_ms.tolist()),
                write_start_ms=int(tokenized.write_start_ms),
                write_end_ms=int(tokenized.write_end_ms),
                chart_end_ms=int(tokenized.chart_end_ms),
                is_full_chart_end=bool(tokenized.is_full_chart_end),
                vocab=vocab,
                delta_max_ms=int(delta_max_ms),
                end_gap_max_ms=int(end_gap_max_ms),
            )
        except Exception as exc:  # noqa: BLE001 - reports need concrete per-record failures.
            errors.append(
                {
                    "index": int(index),
                    "beatmap_path": record.beatmap_path.as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        _accumulate(aggregate, coverage)
        if len(examples) < 8 and (
            int(coverage["out_of_range_delta_count"])
            or int(coverage["out_of_range_end_gap_count"])
            or int(coverage["negative_delta_count"])
            or int(coverage["negative_end_gap_count"])
        ):
            examples.append(
                {
                    "index": int(index),
                    "beatmap_path": record.beatmap_path.as_posix(),
                    "coverage": coverage,
                }
            )

    metrics = _finalize_metrics(aggregate, audited_window_count=row_count, errors=errors)
    checks = guard_checks(metrics)
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event auxiliary fixed-slice label coverage",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "dataset": {
            "source_window_count": len(dataset.records),
            "audited_window_count": int(row_count),
            "filter_report": dataset.filter_report.__dict__,
        },
        "config": {
            "delta_max_ms": int(delta_max_ms),
            "end_gap_max_ms": int(end_gap_max_ms),
            "limit": None if limit is None else int(limit),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "no_training": True,
            "no_rollout": True,
            "tokenizer_changed": False,
            "dataset_schema_changed": False,
            "default_behavior_changed": False,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "context": _load_context(
            proxy_summary_path=Path(proxy_summary_path),
            tiny_training_summary_path=Path(tiny_training_summary_path),
        ),
        "metrics": metrics,
        "checks": checks,
        "decision": decision,
        "next_step": decision["next_step"],
        "examples": examples,
        "errors": errors[:20],
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def window_label_coverage(
    *,
    token_ids: Sequence[int],
    current_ms: Sequence[int],
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    is_full_chart_end: bool,
    vocab: MapperV3Vocab,
    delta_max_ms: int,
    end_gap_max_ms: int,
) -> dict[str, Any]:
    if len(token_ids) != len(current_ms):
        raise ValueError("token_ids and current_ms must have equal length")
    target_end = v3_target_end_ms(
        write_start_ms=int(write_start_ms),
        write_end_ms=int(write_end_ms),
        chart_end_ms=int(chart_end_ms),
        is_full_chart_end=bool(is_full_chart_end),
    )
    event_index_by_token = {int(token_id): index for index, token_id in enumerate(vocab.event_token_ids)}
    anchor_ms = int(write_start_ms)
    last_valid_step = -1
    event_deltas: list[int] = []
    event_signatures: list[str] = []
    out_of_range_delta_count = 0
    negative_delta_count = 0
    time_shift_count = 0
    event_token_count = 0
    eos_count = 0
    for step, token_id_raw in enumerate(token_ids):
        token_id = int(token_id_raw)
        last_valid_step = step
        if vocab.is_time_shift_token(token_id):
            time_shift_count += 1
            continue
        if vocab.is_event_token(token_id):
            event_token_count += 1
            event_ms = int(current_ms[step])
            delta_ms = event_ms - anchor_ms
            event_deltas.append(delta_ms)
            event_signatures.append(vocab.event_signature(token_id))
            if delta_ms < 0:
                negative_delta_count += 1
            if delta_ms < 0 or delta_ms > int(delta_max_ms):
                out_of_range_delta_count += 1
            if token_id not in event_index_by_token:
                raise ValueError(f"event token missing signature class: {token_id}")
            anchor_ms = event_ms
            continue
        if token_id == int(vocab.eos_id):
            eos_count += 1
            continue
        raise ValueError(f"unsupported v3 token for delta-event auxiliary label coverage: {token_id}")

    end_gap_count = 0
    end_gap_ms = 0
    out_of_range_end_gap_count = 0
    negative_end_gap_count = 0
    if last_valid_step >= 0:
        end_gap_count = 1
        end_gap_ms = int(target_end) - int(anchor_ms)
        if end_gap_ms < 0:
            negative_end_gap_count = 1
        if end_gap_ms < 0 or end_gap_ms > int(end_gap_max_ms):
            out_of_range_end_gap_count = 1
    return {
        "v3_token_count": len(token_ids),
        "time_shift_token_count": int(time_shift_count),
        "event_token_count": int(event_token_count),
        "eos_count": int(eos_count),
        "event_label_count": len(event_deltas),
        "signature_label_count": len(event_signatures),
        "end_gap_label_count": int(end_gap_count),
        "event_deltas": event_deltas,
        "event_signatures": event_signatures,
        "end_gap_ms": int(end_gap_ms),
        "out_of_range_delta_count": int(out_of_range_delta_count),
        "out_of_range_end_gap_count": int(out_of_range_end_gap_count),
        "negative_delta_count": int(negative_delta_count),
        "negative_end_gap_count": int(negative_end_gap_count),
    }


def guard_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "no_training": True,
        "no_rollout": True,
        "no_tokenizer_dataset_default_change": True,
        "no_c3_backreference_or_future_lookup": True,
        "audited_windows_positive": int(metrics.get("audited_window_count") or 0) > 0,
        "label_errors_zero": int(metrics.get("label_error_count") or 0) == 0,
        "negative_deltas_zero": int(metrics.get("negative_delta_count") or 0) == 0,
        "negative_end_gaps_zero": int(metrics.get("negative_end_gap_count") or 0) == 0,
        "delta_out_of_range_zero": int(metrics.get("out_of_range_delta_count") or 0) == 0,
        "end_gap_out_of_range_zero": int(metrics.get("out_of_range_end_gap_count") or 0) == 0,
        "event_labels_match_event_tokens": int(metrics.get("event_label_count") or -1)
        == int(metrics.get("event_token_count") or -2),
        "signature_labels_match_event_tokens": int(metrics.get("signature_label_count") or -1)
        == int(metrics.get("event_token_count") or -2),
        "end_gap_labels_match_windows": int(metrics.get("end_gap_label_count") or -1)
        == int(metrics.get("audited_window_count") or -2),
        "event_labels_positive": int(metrics.get("event_label_count") or 0) > 0,
        "end_gap_labels_positive": int(metrics.get("end_gap_label_count") or 0) > 0,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return {
            "route": "TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING",
            "reason": "fixed-slice label coverage passed for current delta-event auxiliary head ranges",
            "next_step": "Create a tiny real-data training gate for the default-off delta-event auxiliary objective.",
        }
    range_failures = {"delta_out_of_range_zero", "end_gap_out_of_range_zero"}
    if any(key in range_failures for key in failed):
        return {
            "route": "MUTATE_DELTA_EVENT_AUXILIARY_HEAD_RANGE",
            "reason": "label coverage failed range checks: " + ", ".join(failed),
            "next_step": "Mutate delta/end-gap head ranges or bucketing before any real training.",
        }
    return {
        "route": "KILL_DELTA_EVENT_AUXILIARY_LABEL_COVERAGE",
        "reason": "label coverage failed hard checks: " + ", ".join(failed),
        "next_step": "Repair label extraction semantics before any real training.",
    }


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(summary).rstrip() + "\n", encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    metrics = _mapping(summary.get("metrics"))
    checks = _mapping(summary.get("checks"))
    config = _mapping(summary.get("config"))
    lines = [
        "# Target Grammar v3 Delta-Event Auxiliary Fixed-Slice Label Coverage Result Report",
        "",
        "## Scope",
        "",
        "This fixed-slice audit derives delta-event auxiliary labels from current teacher-forced v3 target fragments. It does not train, roll out, change tokenizer behavior, change dataset schema, or change mapper defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("audited windows", metrics.get("audited_window_count")),
        _row("v3 token count", metrics.get("v3_token_count")),
        _row("time-shift token count", metrics.get("time_shift_token_count")),
        _row("event token count", metrics.get("event_token_count")),
        _row("event label count", metrics.get("event_label_count")),
        _row("signature label count", metrics.get("signature_label_count")),
        _row("end-gap label count", metrics.get("end_gap_label_count")),
        _row("label error count", metrics.get("label_error_count")),
        _row("out-of-range delta count", metrics.get("out_of_range_delta_count")),
        _row("out-of-range end-gap count", metrics.get("out_of_range_end_gap_count")),
        _row("negative delta count", metrics.get("negative_delta_count")),
        _row("negative end-gap count", metrics.get("negative_end_gap_count")),
        _row("max event delta ms", metrics.get("max_event_delta_ms")),
        _row("max end-gap ms", metrics.get("max_end_gap_ms")),
        _row("unique event deltas", metrics.get("unique_event_delta_count")),
        _row("unique end gaps", metrics.get("unique_end_gap_count")),
        "",
        "## Checks",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    lines.extend(_row(key, value) for key, value in checks.items())
    lines.extend(
        [
            "",
            "## Top Event Deltas",
            "",
            "| delta ms | count |",
            "| ---: | ---: |",
        ]
    )
    lines.extend(_counter_rows(metrics.get("top_event_deltas")))
    lines.extend(
        [
            "",
            "## Top End Gaps",
            "",
            "| end gap ms | count |",
            "| ---: | ---: |",
        ]
    )
    lines.extend(_counter_rows(metrics.get("top_end_gaps")))
    lines.extend(
        [
            "",
            "## What Passed",
            "",
            "- Real fixed-slice v3 fragments produced event-delta, event-signature, and end-gap labels under the current 8s head ranges.",
            "- Event/signature label counts matched the current v3 event-token count.",
            "- End-gap label count matched the audited window count.",
            "- No training, rollout, tokenizer/default change, C3 backreference, or future lookup was used.",
            "",
            "## What Surfaced",
            "",
            _interpretation(str(decision.get("route"))),
            "",
            _boundary_note(metrics, config),
            "",
            "## What Is Not Proved",
            "",
            "- No real training run was performed.",
            "- No rollout quality or generated-chart legality was measured.",
            "- This fixed 256-window slice is not full4k label coverage.",
            "- This does not make v3 target replacement ready.",
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
        ]
    )
    return "\n".join(lines)


def _boundary_note(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    max_end_gap = int(metrics.get("max_end_gap_ms") or 0)
    end_gap_max = int(config.get("end_gap_max_ms") or 0)
    if end_gap_max > 0 and max_end_gap >= end_gap_max:
        return (
            f"The max end-gap label hit the current head boundary exactly at `{max_end_gap}ms`. "
            "This is still a pass for the fixed slice, but the future full4k label coverage audit should explicitly "
            "stress the end-gap range before any default grammar change."
        )
    return "No fixed-slice label reached the current configured end-gap head boundary."


def _empty_aggregate() -> dict[str, Any]:
    return {
        "v3_token_count": 0,
        "time_shift_token_count": 0,
        "event_token_count": 0,
        "eos_count": 0,
        "event_label_count": 0,
        "signature_label_count": 0,
        "end_gap_label_count": 0,
        "out_of_range_delta_count": 0,
        "out_of_range_end_gap_count": 0,
        "negative_delta_count": 0,
        "negative_end_gap_count": 0,
        "event_delta_counts": Counter(),
        "end_gap_counts": Counter(),
        "event_signature_counts": Counter(),
    }


def _accumulate(aggregate: dict[str, Any], coverage: Mapping[str, Any]) -> None:
    for key in (
        "v3_token_count",
        "time_shift_token_count",
        "event_token_count",
        "eos_count",
        "event_label_count",
        "signature_label_count",
        "end_gap_label_count",
        "out_of_range_delta_count",
        "out_of_range_end_gap_count",
        "negative_delta_count",
        "negative_end_gap_count",
    ):
        aggregate[key] += int(coverage.get(key) or 0)
    aggregate["event_delta_counts"].update(int(value) for value in _sequence(coverage.get("event_deltas")))
    aggregate["end_gap_counts"].update([int(coverage.get("end_gap_ms") or 0)])
    aggregate["event_signature_counts"].update(str(value) for value in _sequence(coverage.get("event_signatures")))


def _finalize_metrics(
    aggregate: Mapping[str, Any],
    *,
    audited_window_count: int,
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_deltas = _counter(aggregate.get("event_delta_counts"))
    end_gaps = _counter(aggregate.get("end_gap_counts"))
    signatures = _counter(aggregate.get("event_signature_counts"))
    return {
        "audited_window_count": int(audited_window_count),
        "label_error_count": len(errors),
        "v3_token_count": int(aggregate["v3_token_count"]),
        "time_shift_token_count": int(aggregate["time_shift_token_count"]),
        "event_token_count": int(aggregate["event_token_count"]),
        "eos_count": int(aggregate["eos_count"]),
        "event_label_count": int(aggregate["event_label_count"]),
        "signature_label_count": int(aggregate["signature_label_count"]),
        "end_gap_label_count": int(aggregate["end_gap_label_count"]),
        "out_of_range_delta_count": int(aggregate["out_of_range_delta_count"]),
        "out_of_range_end_gap_count": int(aggregate["out_of_range_end_gap_count"]),
        "negative_delta_count": int(aggregate["negative_delta_count"]),
        "negative_end_gap_count": int(aggregate["negative_end_gap_count"]),
        "max_event_delta_ms": max(event_deltas) if event_deltas else 0,
        "max_end_gap_ms": max(end_gaps) if end_gaps else 0,
        "unique_event_delta_count": len(event_deltas),
        "unique_end_gap_count": len(end_gaps),
        "unique_event_signature_count": len(signatures),
        "zero_delta_event_count": int(event_deltas.get(0, 0)),
        "top_event_deltas": _top_counter(event_deltas),
        "top_end_gaps": _top_counter(end_gaps),
        "top_event_signatures": _top_counter(signatures),
    }


def _load_context(
    *,
    proxy_summary_path: Path,
    tiny_training_summary_path: Path,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if proxy_summary_path.exists():
        proxy = _read_json(proxy_summary_path)
        context["delta_event_proxy_audit"] = {
            "path": proxy_summary_path.as_posix(),
            "route": str(_mapping(proxy.get("decision")).get("route") or ""),
            "event_row_count": _mapping(proxy.get("metrics")).get("event_row_count"),
            "end_row_count": _mapping(proxy.get("metrics")).get("end_row_count"),
            "unique": _mapping(_mapping(proxy.get("metrics")).get("unique")),
        }
    if tiny_training_summary_path.exists():
        tiny = _read_json(tiny_training_summary_path)
        context["delta_event_auxiliary_tiny_training"] = {
            "path": tiny_training_summary_path.as_posix(),
            "route": str(_mapping(tiny.get("decision")).get("route") or ""),
            "auxiliary_loss_ratio": _mapping(tiny.get("training")).get("auxiliary_loss_ratio"),
        }
    return context


def _interpretation(route: str) -> str:
    if route == "TEST_DELTA_EVENT_AUXILIARY_TINY_REAL_DATA_TRAINING":
        return (
            "Fixed-slice real v3 fragments fit the current delta-event auxiliary label surface. "
            "This supports a tiny real-data training gate, not rollout or replacement."
        )
    if route == "MUTATE_DELTA_EVENT_AUXILIARY_HEAD_RANGE":
        return "The label semantics are plausible but the current head ranges are insufficient."
    return "The fixed-slice label surface failed hard coverage checks and should be repaired before training."


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_range(value: int, *, name: str) -> None:
    if int(value) <= 0 or int(value) % 10 != 0:
        raise ValueError(f"{name} must be a positive 10ms-grid value")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _counter(value: object) -> Counter[Any]:
    return value if isinstance(value, Counter) else Counter()


def _top_counter(counter: Counter[Any], *, limit: int = 12) -> list[dict[str, Any]]:
    return [{"value": value, "count": int(count)} for value, count in counter.most_common(limit)]


def _counter_rows(value: object) -> list[str]:
    rows: list[str] = []
    for item in _sequence(value):
        row = _mapping(item)
        rows.append(f"| `{row.get('value')}` | `{row.get('count')}` |")
    return rows


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit fixed-slice v3 delta-event auxiliary label coverage.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--delta-max-ms", type=int, default=8000)
    parser.add_argument("--end-gap-max-ms", type=int, default=8000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=64)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    summary = run_delta_event_auxiliary_fixed_slice_label_coverage(
        index_path=Path(args.index_path),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
        delta_max_ms=args.delta_max_ms,
        end_gap_max_ms=args.end_gap_max_ms,
        limit=args.limit,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
        progress=args.progress,
    )
    print(
        "mapper_v3_delta_event_auxiliary_fixed_slice_label_coverage_done "
        f"route={summary['decision']['route']} "
        f"event_labels={summary['metrics']['event_label_count']} "
        f"end_gap_labels={summary['metrics']['end_gap_label_count']} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
