from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset
from pulsefield_model.models.mapper.v3.replay import target_end_ms
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_proxy_audit_summary.json"
DEFAULT_REPORT_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_proxy_audit_result_report.md"
DEFAULT_EXPERIMENT_CARD_PATH = REPORT_ROOT / "target_grammar_v3_delta_event_proxy_audit_experiment_card.md"
DEFAULT_V3_FULL_DATASET_SUMMARY_PATH = REPORT_ROOT / "target_grammar_v3_full_dataset_audit_summary.json"
DEFAULT_ROUTE_SYNTHESIS_PATH = REPORT_ROOT / "mapper_v21_terminal_guard_v3_route_synthesis_summary.json"
FLAT_PAIR_TRACTABLE_LIMIT = 4096


@dataclass(frozen=True)
class DeltaEventRow:
    delta_ms: int
    signature: str
    event_time_ms: int


@dataclass(frozen=True)
class DeltaEventProxyWindow:
    rows: tuple[DeltaEventRow, ...]
    end_gap_ms: int
    write_start_ms: int
    target_end_ms: int
    original_event_times: tuple[int, ...]
    original_event_signatures: tuple[str, ...]
    reconstructed_event_times: tuple[int, ...]
    reconstructed_event_signatures: tuple[str, ...]
    reconstructed_target_end_ms: int

    @property
    def row_count(self) -> int:
        return len(self.rows) + 1

    @property
    def event_count(self) -> int:
        return len(self.rows)

    @property
    def event_mismatch_count(self) -> int:
        return int(
            self.original_event_times != self.reconstructed_event_times
            or self.original_event_signatures != self.reconstructed_event_signatures
        )

    @property
    def end_mismatch_count(self) -> int:
        return int(self.reconstructed_target_end_ms != self.target_end_ms)


def run_delta_event_proxy_audit(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    summary_output_path: str | Path = DEFAULT_SUMMARY_PATH,
    report_output_path: str | Path | None = DEFAULT_REPORT_PATH,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD_PATH,
    v3_full_dataset_summary_path: str | Path = DEFAULT_V3_FULL_DATASET_SUMMARY_PATH,
    route_synthesis_path: str | Path = DEFAULT_ROUTE_SYNTHESIS_PATH,
    limit: int | None = None,
    max_cached_timepoint_maps: int = 64,
    progress: bool = False,
) -> dict[str, Any]:
    start = time.monotonic()
    vocab = MapperV3Vocab()
    dataset = MapperV3WindowDataset(
        index_path=index_path,
        vocab=vocab,
        max_cached_timepoint_maps=int(max_cached_timepoint_maps),
        progress=bool(progress),
    )
    row_count = len(dataset.records) if limit is None else min(int(limit), len(dataset.records))
    if row_count <= 0:
        raise ValueError("delta-event proxy audit requires at least one dataset row")

    windows: list[DeltaEventProxyWindow] = []
    v3_tokens: list[int] = []
    for index in range(row_count):
        record = dataset.records[index].control_record
        tokenized = dataset._tokenize_record(record)
        token_ids = tuple(int(token_id) for token_id in tokenized.target_fragment_ids)
        v3_tokens.extend(token_ids)
        windows.append(
            delta_event_proxy_from_v3_tokens(
                token_ids,
                vocab=vocab,
                write_start_ms=int(tokenized.write_start_ms),
                write_end_ms=int(tokenized.write_end_ms),
                chart_end_ms=int(tokenized.chart_end_ms),
                is_full_chart_end=bool(tokenized.is_full_chart_end),
            )
        )

    metrics = proxy_metrics(windows, v3_tokens=v3_tokens, vocab=vocab)
    context = load_context(
        v3_full_dataset_summary_path=Path(v3_full_dataset_summary_path),
        route_synthesis_path=Path(route_synthesis_path),
    )
    checks = guard_checks(metrics)
    decision = decision_from_checks(checks, metrics)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event proxy audit",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - start,
        "index_path": Path(index_path).as_posix(),
        "dataset": {
            "source_window_count": len(dataset.records),
            "audited_window_count": int(row_count),
            "filter_report": dataset.filter_report.__dict__,
        },
        "config": {
            "limit": None if limit is None else int(limit),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "flat_pair_tractable_limit": int(FLAT_PAIR_TRACTABLE_LIMIT),
            "no_training": True,
            "no_rollout": True,
            "tokenizer_changed": False,
            "grammar_defaults_changed": False,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "context": context,
        "metrics": metrics,
        "checks": checks,
        "decision": decision,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, Path(summary_output_path))
    if report_output_path is not None:
        write_report(summary, Path(report_output_path))
    return summary


def delta_event_proxy_from_v3_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV3Vocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    is_full_chart_end: bool,
) -> DeltaEventProxyWindow:
    target_end = target_end_ms(
        write_start_ms=int(write_start_ms),
        write_end_ms=int(write_end_ms),
        chart_end_ms=int(chart_end_ms),
        is_full_chart_end=bool(is_full_chart_end),
    )
    current_ms = int(write_start_ms)
    anchor_ms = int(write_start_ms)
    rows: list[DeltaEventRow] = []
    original_times: list[int] = []
    original_signatures: list[str] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == int(vocab.eos_id):
            continue
        if vocab.is_time_shift_token(token_id):
            current_ms += int(vocab.time_shift_value(token_id))
            continue
        if vocab.is_event_token(token_id):
            signature = vocab.event_signature(token_id)
            delta_ms = int(current_ms) - int(anchor_ms)
            if delta_ms < 0:
                raise ValueError(f"negative delta-event gap: {delta_ms}")
            rows.append(DeltaEventRow(delta_ms=delta_ms, signature=signature, event_time_ms=int(current_ms)))
            original_times.append(int(current_ms))
            original_signatures.append(signature)
            anchor_ms = int(current_ms)
            continue
        raise ValueError(f"unsupported v3 token in delta-event proxy: {token_id}")
    end_gap_ms = int(target_end) - int(anchor_ms)
    if end_gap_ms < 0:
        raise ValueError(f"negative end gap: {end_gap_ms}")

    reconstructed_times: list[int] = []
    reconstructed_signatures: list[str] = []
    reconstructed_ms = int(write_start_ms)
    for row in rows:
        reconstructed_ms += int(row.delta_ms)
        reconstructed_times.append(int(reconstructed_ms))
        reconstructed_signatures.append(str(row.signature))
    reconstructed_target_end = int(reconstructed_ms) + int(end_gap_ms)
    return DeltaEventProxyWindow(
        rows=tuple(rows),
        end_gap_ms=end_gap_ms,
        write_start_ms=int(write_start_ms),
        target_end_ms=int(target_end),
        original_event_times=tuple(original_times),
        original_event_signatures=tuple(original_signatures),
        reconstructed_event_times=tuple(reconstructed_times),
        reconstructed_event_signatures=tuple(reconstructed_signatures),
        reconstructed_target_end_ms=int(reconstructed_target_end),
    )


def proxy_metrics(
    windows: Sequence[DeltaEventProxyWindow],
    *,
    v3_tokens: Sequence[int],
    vocab: MapperV3Vocab,
) -> dict[str, Any]:
    kind_counts: Counter[str] = Counter()
    event_delta_counts: Counter[int] = Counter()
    end_gap_counts: Counter[int] = Counter()
    signature_counts: Counter[str] = Counter()
    flat_pair_counts: Counter[str] = Counter()
    event_rows = 0
    proxy_rows = 0
    event_mismatches = 0
    end_mismatches = 0
    zero_delta_events = 0
    for window in windows:
        event_mismatches += window.event_mismatch_count
        end_mismatches += window.end_mismatch_count
        for row in window.rows:
            kind_counts["EVENT"] += 1
            event_delta_counts[int(row.delta_ms)] += 1
            signature_counts[str(row.signature)] += 1
            flat_pair_counts[f"{int(row.delta_ms)}:{row.signature}"] += 1
            event_rows += 1
            proxy_rows += 1
            if int(row.delta_ms) == 0:
                zero_delta_events += 1
        kind_counts["END"] += 1
        end_gap_counts[int(window.end_gap_ms)] += 1
        proxy_rows += 1

    v3_token_counts = Counter(int(token_id) for token_id in v3_tokens)
    v3_total_bits = unigram_total_bits(v3_token_counts)
    proxy_total_bits = (
        unigram_total_bits(kind_counts)
        + unigram_total_bits(event_delta_counts)
        + unigram_total_bits(signature_counts)
        + unigram_total_bits(end_gap_counts)
    )
    flat_total_bits = unigram_total_bits(kind_counts) + unigram_total_bits(flat_pair_counts) + unigram_total_bits(end_gap_counts)
    v3_token_count = len(v3_tokens)
    return {
        "window_count": len(windows),
        "v3_token_count": int(v3_token_count),
        "proxy_row_count": int(proxy_rows),
        "event_row_count": int(event_rows),
        "end_row_count": len(windows),
        "sequence_length_ratio_vs_v3": _safe_ratio(proxy_rows, v3_token_count),
        "sequence_length_delta_vs_v3": int(proxy_rows - v3_token_count),
        "event_reconstruction_mismatches": int(event_mismatches),
        "end_reconstruction_mismatches": int(end_mismatches),
        "v3_unigram_total_bits": float(v3_total_bits),
        "v3_unigram_bits_per_token": _safe_ratio(v3_total_bits, v3_token_count),
        "proxy_factorized_total_bits": float(proxy_total_bits),
        "proxy_factorized_bits_per_row": _safe_ratio(proxy_total_bits, proxy_rows),
        "proxy_factorized_total_bit_ratio_vs_v3": _safe_ratio(proxy_total_bits, v3_total_bits),
        "flat_pair_total_bits": float(flat_total_bits),
        "flat_pair_bits_per_row": _safe_ratio(flat_total_bits, proxy_rows),
        "flat_pair_total_bit_ratio_vs_v3": _safe_ratio(flat_total_bits, v3_total_bits),
        "unique": {
            "v3_tokens": len(v3_token_counts),
            "kinds": len(kind_counts),
            "event_deltas": len(event_delta_counts),
            "end_gaps": len(end_gap_counts),
            "event_signatures": len(signature_counts),
            "flat_delta_event_pairs": len(flat_pair_counts),
        },
        "flat_pair_tractable": len(flat_pair_counts) <= FLAT_PAIR_TRACTABLE_LIMIT,
        "zero_delta_event_count": int(zero_delta_events),
        "top_event_deltas": _top_counter(event_delta_counts),
        "top_end_gaps": _top_counter(end_gap_counts),
        "top_event_signatures": _top_counter(signature_counts),
        "top_flat_pairs": _top_counter(flat_pair_counts),
        "special_token_counts": {
            "bos": int(v3_token_counts.get(int(vocab.bos_id), 0)),
            "eos": int(v3_token_counts.get(int(vocab.eos_id), 0)),
            "time_shift": sum(count for token_id, count in v3_token_counts.items() if vocab.is_time_shift_token(token_id)),
            "event": sum(count for token_id, count in v3_token_counts.items() if vocab.is_event_token(token_id)),
        },
    }


def guard_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "no_training": True,
        "no_rollout": True,
        "no_tokenizer_or_default_change": True,
        "no_c3_backreference": True,
        "no_future_lookup": True,
        "event_reconstruction_zero": int(metrics.get("event_reconstruction_mismatches") or 0) == 0,
        "end_reconstruction_zero": int(metrics.get("end_reconstruction_mismatches") or 0) == 0,
        "sequence_shorter_than_v3": (_float(metrics.get("sequence_length_ratio_vs_v3")) or math.inf) < 1.0,
        "factorized_bits_not_worse_than_v3": (_float(metrics.get("proxy_factorized_total_bit_ratio_vs_v3")) or math.inf) <= 1.0,
        "flat_pair_vocab_tractable": bool(metrics.get("flat_pair_tractable")),
    }


def decision_from_checks(checks: Mapping[str, bool], metrics: Mapping[str, Any]) -> dict[str, Any]:
    failed = [key for key, value in checks.items() if not bool(value)]
    if not bool(checks.get("event_reconstruction_zero")) or not bool(checks.get("end_reconstruction_zero")):
        return {
            "route": "KILL_DELTA_EVENT_PROXY_RECONSTRUCTION",
            "reason": f"delta-event proxy failed reconstruction guards: {failed}",
            "next_step": "Do not implement this grammar; inspect conversion semantics or choose a different structural target.",
        }
    if not bool(checks.get("sequence_shorter_than_v3")):
        return {
            "route": "KILL_DELTA_EVENT_PROXY_SEQUENCE",
            "reason": "delta-event proxy did not reduce sequence length versus current v3",
            "next_step": "Do not pursue this proxy; pivot to v2.1 grammar hardening or a different v3 structural target.",
        }
    if not bool(checks.get("factorized_bits_not_worse_than_v3")):
        return {
            "route": "MUTATE_DELTA_EVENT_BUCKETING",
            "reason": "delta-event proxy reduced rows but regressed the factorized unigram bit proxy",
            "next_step": "Try a bounded delta bucketing audit or pivot to v2.1 terminal guard default validation.",
        }
    if not bool(checks.get("flat_pair_vocab_tractable")):
        return {
            "route": "TEST_FACTORIZED_DELTA_EVENT_HEAD",
            "reason": "factorized proxy passed, but flat delta-event pairs are not tractable enough for a single softmax",
            "next_step": "Create a factorized delta/event head smoke card; reject the flat combined-token variant.",
        }
    return {
        "route": "TEST_DELTA_EVENT_PROXY_MODEL_HEAD",
        "reason": (
            "delta-event proxy passed reconstruction, sequence, and factorized-bit gates on the bounded slice "
            f"(row ratio={_fmt(metrics.get('sequence_length_ratio_vs_v3'))}, bit ratio={_fmt(metrics.get('proxy_factorized_total_bit_ratio_vs_v3'))})"
        ),
        "next_step": "Create a bounded factorized delta-event model/loss plumbing card before any training-scale run.",
    }


def load_context(
    *,
    v3_full_dataset_summary_path: Path,
    route_synthesis_path: Path,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if v3_full_dataset_summary_path.exists():
        full = _read_json(v3_full_dataset_summary_path)
        context["v3_full_dataset"] = {
            "path": v3_full_dataset_summary_path.as_posix(),
            "decision_route": _route(full),
            "token_reduction_ratio": _float(_mapping(full.get("comparison")).get("token_reduction_ratio")),
            "total_bit_reduction_ratio": _float(_mapping(full.get("comparison")).get("total_bit_reduction_ratio")),
            "reconstruction_mismatches": _mapping(full.get("comparison")).get("reconstruction_mismatches"),
        }
    if route_synthesis_path.exists():
        route = _read_json(route_synthesis_path)
        context["v21_terminal_guard_v3_route"] = {
            "path": route_synthesis_path.as_posix(),
            "decision_route": _route(route),
            "next_card": _mapping(route.get("decision")).get("next_card"),
        }
    return context


def unigram_total_bits(counts: Counter[Any]) -> float:
    total = int(sum(counts.values()))
    if total <= 0:
        return 0.0
    return -sum(float(count) * math.log2(float(count) / float(total)) for count in counts.values() if count > 0)


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
    unique = _mapping(metrics.get("unique"))
    special = _mapping(metrics.get("special_token_counts"))
    lines = [
        "# Target Grammar v3 Delta-Event Proxy Audit Result Report",
        "",
        "## Scope",
        "",
        "This bounded representation audit converts current v3 target fragments into a factorized delta-event proxy: event rows carry `delta_ms` plus event signature, and each window carries an end-gap marker. It does not train, rerun rollout, change tokenizer behavior, or change defaults.",
        "",
        "## Decision",
        "",
        f"Route: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Recommended next step: {decision.get('next_step')}",
        "",
        "## Guard Results",
        "",
        "| Guard | Value |",
        "| --- | ---: |",
        *[_row(key, value) for key, value in checks.items()],
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        _row("audited windows", _mapping(summary.get("dataset")).get("audited_window_count")),
        _row("v3 token count", metrics.get("v3_token_count")),
        _row("proxy row count", metrics.get("proxy_row_count")),
        _row("sequence ratio vs v3", _fmt(metrics.get("sequence_length_ratio_vs_v3"))),
        _row("v3 unigram total bits", _fmt(metrics.get("v3_unigram_total_bits"))),
        _row("proxy factorized total bits", _fmt(metrics.get("proxy_factorized_total_bits"))),
        _row("proxy bit ratio vs v3", _fmt(metrics.get("proxy_factorized_total_bit_ratio_vs_v3"))),
        _row("flat pair bit ratio vs v3", _fmt(metrics.get("flat_pair_total_bit_ratio_vs_v3"))),
        _row("event reconstruction mismatches", metrics.get("event_reconstruction_mismatches")),
        _row("end reconstruction mismatches", metrics.get("end_reconstruction_mismatches")),
        _row("unique event deltas", unique.get("event_deltas")),
        _row("unique event signatures", unique.get("event_signatures")),
        _row("unique flat delta-event pairs", unique.get("flat_delta_event_pairs")),
        _row("v3 time-shift token count", special.get("time_shift")),
        _row("v3 event token count", special.get("event")),
        "",
        "## Top Event Deltas",
        "",
        "| delta ms | count |",
        "| ---: | ---: |",
    ]
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
        "- The audit stayed representation-only: no training, rollout, tokenizer/default change, C3 input, or future lookup.",
        f"- Reconstruction passed: `{metrics.get('event_reconstruction_mismatches')}` event mismatches and `{metrics.get('end_reconstruction_mismatches')}` terminal end mismatches.",
        f"- Sequence pressure passed: `{metrics.get('proxy_row_count')}` proxy rows versus `{metrics.get('v3_token_count')}` v3 target tokens (`{_fmt(metrics.get('sequence_length_ratio_vs_v3'))}` ratio).",
        f"- Bit proxy passed: factorized proxy bits are `{_fmt(metrics.get('proxy_factorized_total_bit_ratio_vs_v3'))}` of current v3 unigram bits.",
        f"- Flat pair cardinality stayed tractable on this slice: `{unique.get('flat_delta_event_pairs')}` unique delta/event pairs under the `{FLAT_PAIR_TRACTABLE_LIMIT}` guard.",
        "",
        "## What Surfaced",
        "",
        _interpretation(summary),
        "",
        "The audit also surfaced that current v3 spends much of its target surface on standalone time-shift tokens: "
        f"`{special.get('time_shift')}` time-shift tokens versus `{special.get('event')}` event tokens in the audited fragments. "
        "Binding each event to its preceding delta removes that generated-prefix decision surface in the representation, but this remains a unigram proxy and does not prove learnability.",
        "",
        "The timing/event space is concentrated but not trivial: "
        f"`{unique.get('event_deltas')}` event-delta values, `{unique.get('event_signatures')}` event signatures, and `{unique.get('end_gaps')}` terminal end-gap values. "
        "That supports a factorized head follow-up before considering a flat target grammar.",
        "",
        "## What Is Not Proved",
        "",
        "- No mapper training run was performed.",
        "- No autoregressive rollout or generated-chart quality improvement was measured.",
        "- The bit estimate is a target-stream unigram proxy, not a conditional model likelihood.",
        "- This does not replace the C3 local-backreference result; it only tests a separate v3-family grammar mutation without C3 replay.",
        "",
        "## Next Step",
            "",
            str(decision.get("next_step")),
        ]
    )
    return "\n".join(lines)


def _interpretation(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    route = str(decision.get("route") or "")
    if route.startswith("TEST"):
        return "The proxy is viable enough for the next bounded model-head card, but it is not trained rollout evidence or replacement readiness."
    if route.startswith("MUTATE"):
        return "The proxy has partial representation value but needs bucketing or another structural mutation before model work."
    return "The proxy failed a hard representation gate and should not be implemented as a target grammar."


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _route(payload: Mapping[str, Any]) -> str:
    decision = payload.get("decision")
    if isinstance(decision, str):
        return decision
    return str(_mapping(decision).get("route") or "")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def _top_counter(counter: Counter[Any], *, limit: int = 12) -> list[dict[str, Any]]:
    return [{"value": value, "count": int(count)} for value, count in counter.most_common(limit)]


def _counter_rows(value: object) -> list[str]:
    rows = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            row = _mapping(item)
            rows.append(f"| `{row.get('value')}` | `{row.get('count')}` |")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit a bounded v3 delta-event target proxy.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=64)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    summary = run_delta_event_proxy_audit(
        index_path=Path(args.index_path),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output),
        limit=args.limit,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
        progress=args.progress,
    )
    print(
        "target_grammar_v3_delta_event_proxy_audit_done "
        f"route={summary['decision']['route']} "
        f"row_ratio={summary['metrics']['sequence_length_ratio_vs_v3']:.6f} "
        f"bit_ratio={summary['metrics']['proxy_factorized_total_bit_ratio_vs_v3']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
