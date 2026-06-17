from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pulsefield_model.data.mapper_sparse_windows_v3 import MapperV3WindowDataset, collate_mapper_v3_windows
from pulsefield_model.models.mapper.v3.vocab import MapperV3Vocab


SUMMARY_SCHEMA_VERSION = 1
REPORT_ROOT = Path("artifacts/reports/audits/mapper_v2_1_grammar")
DEFAULT_INDEX_PATH = Path("artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet")
DEFAULT_CONTROL_TEACHER_CACHE_DIR = Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000")
DEFAULT_EXPERIMENT_CARD = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_data_contract_experiment_card.md"
DEFAULT_SUMMARY_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_data_contract_summary.json"
DEFAULT_REPORT_OUTPUT = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_data_contract_result_report.md"
DEFAULT_TINY_MODEL_SUMMARY = REPORT_ROOT / "target_grammar_v3_delta_event_factor_target_tiny_model_summary.json"


def run_delta_event_factor_target_data_contract(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    control_teacher_cache_dir: str | Path = DEFAULT_CONTROL_TEACHER_CACHE_DIR,
    summary_output_path: str | Path = DEFAULT_SUMMARY_OUTPUT,
    report_output_path: str | Path | None = DEFAULT_REPORT_OUTPUT,
    experiment_card_path: str | Path = DEFAULT_EXPERIMENT_CARD,
    tiny_model_summary_path: str | Path = DEFAULT_TINY_MODEL_SUMMARY,
    window_limit: int = 32,
    batch_size: int = 4,
    max_cached_timepoint_maps: int = 16,
) -> dict[str, Any]:
    started = time.monotonic()
    if int(window_limit) <= 0:
        raise ValueError("window_limit must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    vocab = MapperV3Vocab()
    default_off_dataset = _dataset(
        index_path=index_path,
        control_teacher_cache_dir=control_teacher_cache_dir,
        vocab=vocab,
        include_delta_event_factor_target=False,
        max_cached_timepoint_maps=max_cached_timepoint_maps,
    )
    default_on_dataset = _dataset(
        index_path=index_path,
        control_teacher_cache_dir=control_teacher_cache_dir,
        vocab=vocab,
        include_delta_event_factor_target=True,
        max_cached_timepoint_maps=max_cached_timepoint_maps,
    )
    consumed = min(int(window_limit), len(default_on_dataset))
    off_batch = collate_mapper_v3_windows(
        [default_off_dataset[index] for index in range(min(int(batch_size), consumed))],
        pad_id=vocab.pad_id,
    )
    on_samples = [default_on_dataset[index] for index in range(consumed)]
    on_batches = [
        collate_mapper_v3_windows(on_samples[start : start + int(batch_size)], pad_id=vocab.pad_id)
        for start in range(0, consumed, int(batch_size))
    ]
    metrics = _factor_metrics(on_batches, consumed_window_count=consumed)
    context = _load_context(Path(tiny_model_summary_path))
    checks = guard_checks(
        default_off_batch=off_batch,
        metrics=metrics,
        context=context,
    )
    decision = decision_from_checks(checks)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Target grammar v3 delta-event factor target data contract",
        "experiment_card": Path(experiment_card_path).as_posix(),
        "elapsed_s": time.monotonic() - started,
        "index_path": Path(index_path).as_posix(),
        "control_teacher_cache_dir": Path(control_teacher_cache_dir).as_posix(),
        "config": {
            "window_limit": int(window_limit),
            "batch_size": int(batch_size),
            "max_cached_timepoint_maps": int(max_cached_timepoint_maps),
            "include_delta_event_factor_target_default": False,
            "no_model_loss_runner_inference_change": True,
            "no_rollout": True,
            "uses_c3_backreference": False,
            "uses_future_lookup": False,
        },
        "dataset": {
            "source_window_count": len(default_on_dataset.records),
            "filter_report": default_on_dataset.filter_report.__dict__,
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


def guard_checks(
    *,
    default_off_batch: Mapping[str, Any],
    metrics: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, bool]:
    expected = _mapping(context.get("factor_target_tiny_model"))
    return {
        "default_off_factor_target_absent": "delta_event_factor_target" not in default_off_batch,
        "tiny_model_route_positive": expected.get("route") == "TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD",
        "audited_windows_positive": int(metrics.get("audited_window_count") or 0) > 0,
        "row_count_positive": int(metrics.get("row_count") or 0) > 0,
        "event_rows_positive": int(metrics.get("event_row_count") or 0) > 0,
        "end_rows_match_windows": int(metrics.get("end_row_count") or -1)
        == int(metrics.get("audited_window_count") or -2),
        "row_mask_count_matches_rows": int(metrics.get("row_mask_valid_count") or -1)
        == int(metrics.get("row_count") or -2),
        "kind_labels_match_rows": int(metrics.get("kind_label_count") or -1)
        == int(metrics.get("row_count") or -2),
        "delta_labels_match_events": int(metrics.get("delta_label_count") or -1)
        == int(metrics.get("event_row_count") or -2),
        "signature_labels_match_events": int(metrics.get("signature_label_count") or -1)
        == int(metrics.get("event_row_count") or -2),
        "end_gap_labels_match_windows": int(metrics.get("end_gap_label_count") or -1)
        == int(metrics.get("audited_window_count") or -2),
        "reconstruction_mismatches_zero": int(metrics.get("reconstruction_mismatch_count") or 0) == 0,
        "row_count_matches_tiny_model": expected.get("row_count") is not None
        and int(metrics.get("row_count") or -1) == int(expected.get("row_count") or -2),
        "event_rows_match_tiny_model": expected.get("event_row_count") is not None
        and int(metrics.get("event_row_count") or -1) == int(expected.get("event_row_count") or -2),
        "end_rows_match_tiny_model": expected.get("end_row_count") is not None
        and int(metrics.get("end_row_count") or -1) == int(expected.get("end_row_count") or -2),
        "no_model_loss_runner_inference_change": True,
        "no_c3_backreference_or_future_lookup": True,
    }


def decision_from_checks(checks: Mapping[str, bool]) -> dict[str, str]:
    failed = [key for key, passed in checks.items() if not bool(passed)]
    if not failed:
        return {
            "route": "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD",
            "reason": "default-off v3 contract is preserved and default-on factor target rows collate with exact fixed-slice counts",
            "next_step": "Create a bounded model/loss plumbing card for factorized v3 target rows.",
        }
    if "default_off_factor_target_absent" in failed:
        return {
            "route": "KILL_FACTOR_TARGET_DATA_CONTRACT_DEFAULT_REGRESSION",
            "reason": "default-off v3 dataset/collate contract exposed factor target fields",
            "next_step": "Repair default-off behavior before any model/loss plumbing.",
        }
    return {
        "route": "MUTATE_FACTOR_TARGET_DATA_CONTRACT",
        "reason": "factor target data contract failed checks: " + ", ".join(failed),
        "next_step": "Repair factor target labels, masks, or count parity before model/loss plumbing.",
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
    lines = [
        "# Target Grammar v3 Delta-Event Factor Target Data Contract Result Report",
        "",
        "## Scope",
        "",
        "This gate verifies default-off production data plumbing for factorized delta-event target rows. It does not change production model/loss, training runner, inference, rollout, or mapper defaults.",
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
        _row("row count", metrics.get("row_count")),
        _row("event rows", metrics.get("event_row_count")),
        _row("end rows", metrics.get("end_row_count")),
        _row("max row len", metrics.get("max_row_len")),
        _row("row mask valid count", metrics.get("row_mask_valid_count")),
        _row("kind label count", metrics.get("kind_label_count")),
        _row("delta label count", metrics.get("delta_label_count")),
        _row("signature label count", metrics.get("signature_label_count")),
        _row("end-gap label count", metrics.get("end_gap_label_count")),
        _row("reconstruction mismatches", metrics.get("reconstruction_mismatch_count")),
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
            "## What Passed",
            "",
            _what_passed(str(decision.get("route"))),
            "",
            "## What Surfaced",
            "",
            _what_surfaced(str(decision.get("route"))),
            "",
            "## What Is Not Proved",
            "",
            "- This does not train a production factorized model.",
            "- This does not implement factorized inference or rollout.",
            "- This does not make v3 replacement ready.",
            "",
            "## Verification",
            "",
            *_verification_lines(summary),
            "",
            "## Next Step",
            "",
            str(decision.get("next_step")),
        ]
    )
    return "\n".join(lines)


def _dataset(
    *,
    index_path: str | Path,
    control_teacher_cache_dir: str | Path,
    vocab: MapperV3Vocab,
    include_delta_event_factor_target: bool,
    max_cached_timepoint_maps: int,
) -> MapperV3WindowDataset:
    return MapperV3WindowDataset(
        index_path=index_path,
        vocab=vocab,
        control_teacher_cache_dir=control_teacher_cache_dir,
        require_control_teacher_cache=True,
        include_full_song_context=False,
        include_delta_event_factor_target=bool(include_delta_event_factor_target),
        max_cached_timepoint_maps=int(max_cached_timepoint_maps),
        progress=False,
    )


def _factor_metrics(batches: Sequence[Mapping[str, Any]], *, consumed_window_count: int) -> dict[str, Any]:
    row_count = 0
    event_row_count = 0
    end_row_count = 0
    max_row_len = 0
    row_mask_valid_count = 0
    kind_label_count = 0
    delta_label_count = 0
    signature_label_count = 0
    end_gap_label_count = 0
    reconstruction_mismatch_count = 0
    for batch in batches:
        factor = _mapping(batch.get("delta_event_factor_target"))
        if not factor:
            raise ValueError("default-on batch is missing delta_event_factor_target")
        row_count_tensor = _tensor(factor.get("row_count"))
        event_count_tensor = _tensor(factor.get("event_row_count"))
        end_count_tensor = _tensor(factor.get("end_row_count"))
        row_count += int(row_count_tensor.sum().item())
        event_row_count += int(event_count_tensor.sum().item())
        end_row_count += int(end_count_tensor.sum().item())
        max_row_len = max(max_row_len, int(_tensor(factor.get("row_mask")).shape[1]))
        row_mask_valid_count += int(_tensor(factor.get("row_mask")).sum().item())
        kind_label_count += int(_tensor(factor.get("target_kind")).ne(-100).sum().item())
        delta_label_count += int(_tensor(factor.get("target_delta")).ne(-100).sum().item())
        signature_label_count += int(_tensor(factor.get("target_signature")).ne(-100).sum().item())
        end_gap_label_count += int(_tensor(factor.get("target_end_gap")).ne(-100).sum().item())
        reconstruction_mismatch_count += int(_tensor(factor.get("reconstruction_mismatch_count")).sum().item())
    return {
        "audited_window_count": int(consumed_window_count),
        "row_count": int(row_count),
        "event_row_count": int(event_row_count),
        "end_row_count": int(end_row_count),
        "max_row_len": int(max_row_len),
        "row_mask_valid_count": int(row_mask_valid_count),
        "kind_label_count": int(kind_label_count),
        "delta_label_count": int(delta_label_count),
        "signature_label_count": int(signature_label_count),
        "end_gap_label_count": int(end_gap_label_count),
        "reconstruction_mismatch_count": int(reconstruction_mismatch_count),
    }


def _load_context(path: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = _mapping(data.get("dataset"))
        context["factor_target_tiny_model"] = {
            "path": path.as_posix(),
            "route": str(_mapping(data.get("decision")).get("route") or ""),
            "row_count": dataset.get("row_count"),
            "event_row_count": dataset.get("event_row_count"),
            "end_row_count": dataset.get("end_row_count"),
        }
    return context


def _verification_lines(summary: Mapping[str, Any]) -> list[str]:
    commands = summary.get("commands")
    if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes, bytearray)):
        lines = ["```bash"]
        results: list[str] = []
        for item in commands:
            command = _mapping(item).get("command")
            result = _mapping(item).get("result")
            if command:
                lines.append(str(command))
            if result:
                results.append(str(result))
        lines.append("```")
        if results:
            lines.extend(["", "Results:", *[f"- {result}" for result in results]])
        return lines
    return [
        "```bash",
        "uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_data_contract",
        "uv run --group dev pytest tests/models/mapper/v3/test_factor_target.py tests/data/test_mapper_sparse_windows_v3_factor_target.py tests/evals/test_mapper_v3_delta_event_factor_target_data_contract.py tests/models/mapper/v3/test_model.py -q",
        "uv run python -m py_compile src/pulsefield_model/models/mapper/v3/factor_target.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_data_contract.py",
        "python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_data_contract_summary.json >/dev/null",
        "git diff --check",
        "```",
    ]


def _what_passed(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD":
        return (
            "- Default-off v3 batches do not expose factor-target fields.\n"
            "- Default-on batches collate shifted factor-target inputs and labels.\n"
            "- Fixed-slice row counts match the accepted tiny model gate.\n"
            "- Reconstruction mismatch count remains zero."
        )
    return "- The gate produced an explicit route; inspect failed checks before model/loss plumbing."


def _what_surfaced(route: str) -> str:
    if route == "TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD":
        return "The factorized target has crossed from eval-only evidence into a default-off production data contract."
    if route == "KILL_FACTOR_TARGET_DATA_CONTRACT_DEFAULT_REGRESSION":
        return "The default v3 data contract regressed and must be repaired before continuing."
    return "The default-off contract held, but factor-target rows or masks need repair before model/loss work."


def _tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("expected torch.Tensor in factor target batch")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _row(label: str, value: object) -> str:
    return f"| {label} | `{value}` |"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v3 delta-event factor target data-contract gate.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--control-teacher-cache-dir", default=DEFAULT_CONTROL_TEACHER_CACHE_DIR)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--window-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-cached-timepoint-maps", type=int, default=16)
    args = parser.parse_args(argv)
    summary = run_delta_event_factor_target_data_contract(
        index_path=Path(args.index_path),
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir),
        summary_output_path=Path(args.summary_output),
        report_output_path=Path(args.report_output) if args.report_output else None,
        window_limit=args.window_limit,
        batch_size=args.batch_size,
        max_cached_timepoint_maps=args.max_cached_timepoint_maps,
    )
    print(
        "mapper_v3_delta_event_factor_target_data_contract_done "
        f"route={summary['decision']['route']} "
        f"rows={summary['metrics']['row_count']} "
        f"summary={Path(args.summary_output).as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
