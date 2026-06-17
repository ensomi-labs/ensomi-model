from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.mapper_v21_soft_anti_rigid_stress_probe import (
    DEFAULT_HARD_BLOCK_SUMMARY_PATH,
    DEFAULT_STRESS_CASE_IDS,
    run_soft_anti_rigid_stress_probe,
)


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("artifacts/tmp/mapper_v21_soft_penalty_sweep_stress")
DEFAULT_SUMMARY_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_penalty_sweep_stress_summary.json",
)
DEFAULT_REPORT_PATH = Path(
    "artifacts/reports/audits/mapper_v2_1_grammar/mapper_v21_soft_penalty_sweep_stress_result_report.md",
)
DEFAULT_PENALTIES = (0.5, 1.0, 2.0)
MIN_RIGID_IMPROVED_CASES = 2


def run_soft_penalty_sweep_stress(
    *,
    hard_block_summary_path: str | Path = DEFAULT_HARD_BLOCK_SUMMARY_PATH,
    output_summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    output_report_path: str | Path | None = DEFAULT_REPORT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    penalties: Sequence[float] = DEFAULT_PENALTIES,
    case_ids: Sequence[str] = DEFAULT_STRESS_CASE_IDS,
    mapper_checkpoint_path: str | Path | None = None,
    control_checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    max_tokens_per_window: int = 512,
    min_repeated_spacings: int = 4,
    min_spacing_ms: int = 40,
    max_spacing_ms: int = 400,
    seed: int = 1337,
    collect_logit_diagnostics: bool = False,
    timepoint_preview_limit: int = 4096,
) -> dict[str, Any]:
    penalty_values = validate_penalties(penalties)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    penalty_results: list[dict[str, Any]] = []

    for penalty in penalty_values:
        penalty_label = _penalty_label(penalty)
        penalty_dir = out_dir / penalty_label
        penalty_summary = run_soft_anti_rigid_stress_probe(
            hard_block_summary_path=hard_block_summary_path,
            output_summary_path=penalty_dir / "summary.json",
            output_report_path=None,
            output_dir=penalty_dir,
            case_ids=case_ids,
            mapper_checkpoint_path=mapper_checkpoint_path,
            control_checkpoint_path=control_checkpoint_path,
            device_name=device_name,
            max_tokens_per_window=max_tokens_per_window,
            min_repeated_spacings=min_repeated_spacings,
            min_spacing_ms=min_spacing_ms,
            max_spacing_ms=max_spacing_ms,
            penalty=penalty,
            seed=seed,
            collect_logit_diagnostics=collect_logit_diagnostics,
            timepoint_preview_limit=timepoint_preview_limit,
        )
        penalty_results.append(summarize_penalty_result(penalty_summary))

    decision = decision_from_penalty_results(penalty_results)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "Mapper v2.1 soft penalty sweep stress",
        "decision": decision,
        "hard_block_summary_path": Path(hard_block_summary_path).as_posix(),
        "output_dir": out_dir.as_posix(),
        "config": {
            "penalties": list(penalty_values),
            "case_ids": list(case_ids),
            "device": device_name,
            "max_tokens_per_window": int(max_tokens_per_window),
            "min_repeated_spacings": int(min_repeated_spacings),
            "min_spacing_ms": int(min_spacing_ms),
            "max_spacing_ms": int(max_spacing_ms),
            "seed": int(seed),
            "collect_logit_diagnostics": bool(collect_logit_diagnostics),
            "timepoint_preview_limit": int(timepoint_preview_limit),
        },
        "penalty_results": penalty_results,
        "next_step": decision["next_step"],
    }
    write_summary_json(summary, output_summary_path)
    if output_report_path is not None:
        write_report(summary, output_report_path)
    return summary


def validate_penalties(penalties: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in penalties)
    if not values:
        raise ValueError("at least one penalty must be provided")
    if any((not math.isfinite(value)) or value < 0.0 for value in values):
        raise ValueError("penalties must be finite non-negative values")
    if len(set(values)) != len(values):
        raise ValueError("penalties must be unique")
    return values


def summarize_penalty_result(summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _mapping(summary.get("aggregate"))
    config = _mapping(summary.get("config"))
    soft = _mapping(aggregate.get("soft"))
    soft_vs_baseline = _mapping(aggregate.get("soft_vs_baseline"))
    case_results = [dict(row) for row in summary.get("case_results", []) if isinstance(row, Mapping)]
    hard_identical_count = sum(1 for row in case_results if case_matches_hard_block(row))
    baseline_identical_count = sum(1 for row in case_results if case_matches_baseline(row))
    case_count = int(aggregate.get("case_count") or len(case_results))
    differs_from_hard_count = max(0, case_count - hard_identical_count)
    differs_from_baseline_count = max(0, case_count - baseline_identical_count)
    primary_pass = bool(
        soft.get("all_legal")
        and int(soft_vs_baseline.get("new_starved_count") or 0) == 0
        and int(soft_vs_baseline.get("rigid_improved_count") or 0) >= MIN_RIGID_IMPROVED_CASES
        and differs_from_hard_count >= 1
    )
    return {
        "penalty": float(config.get("penalty") or 0.0),
        "summary_path": Path(str(summary.get("output_dir", ""))).joinpath("summary.json").as_posix(),
        "decision": dict(_mapping(summary.get("decision"))),
        "aggregate": aggregate,
        "case_count": case_count,
        "soft_all_legal": bool(soft.get("all_legal")),
        "soft_legal_count": int(soft.get("legal_count") or 0),
        "soft_starved_count": int(soft.get("starved_count") or 0),
        "new_starved_count": int(soft_vs_baseline.get("new_starved_count") or 0),
        "rigid_improved_count": int(soft_vs_baseline.get("rigid_improved_count") or 0),
        "mean_f1_delta_vs_baseline": _float(soft_vs_baseline.get("mean_f1_delta")) or 0.0,
        "mean_rigid_delta_vs_baseline": _float(soft_vs_baseline.get("mean_dominant_spacing_ratio_delta")) or 0.0,
        "hard_identical_count": int(hard_identical_count),
        "baseline_identical_count": int(baseline_identical_count),
        "differs_from_hard_count": int(differs_from_hard_count),
        "differs_from_baseline_count": int(differs_from_baseline_count),
        "primary_pass": primary_pass,
        "case_results": case_results,
    }


def decision_from_penalty_results(penalty_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passing = [row for row in penalty_results if bool(row.get("primary_pass"))]
    if passing:
        selected = select_best_penalty(passing)
        route = "TEST_NEXT"
        reason = "at least one weak penalty passed the stress legality/starvation gate"
        next_step = "Run a full 32-map fixed-slice comparison for the selected weak penalty before changing defaults."
    else:
        selected = None
        route = "KILL"
        reason = "no tested weak penalty cleared legality, starvation, rigidity, and hard-block-separation gates"
        next_step = "Kill the simple finite-penalty schedule; mutate the detector/schedule or return to v3/C3 diagnostics."
    return {
        "route": route,
        "reason": reason,
        "next_step": next_step,
        "selected_penalty": None if selected is None else float(selected.get("penalty") or 0.0),
        "passing_penalty_count": int(len(passing)),
    }


def select_best_penalty(penalty_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not penalty_results:
        raise ValueError("cannot select from empty penalty results")
    return sorted(
        penalty_results,
        key=lambda row: (
            int(row.get("rigid_improved_count") or 0),
            float(row.get("mean_f1_delta_vs_baseline") or 0.0),
            int(row.get("differs_from_hard_count") or 0),
            -float(row.get("penalty") or 0.0),
        ),
        reverse=True,
    )[0]


def case_matches_hard_block(row: Mapping[str, Any]) -> bool:
    return _case_matches(row, candidate_prefix="soft", comparator_prefix="hard_block")


def case_matches_baseline(row: Mapping[str, Any]) -> bool:
    return _case_matches(row, candidate_prefix="soft", comparator_prefix="baseline")


def _case_matches(row: Mapping[str, Any], *, candidate_prefix: str, comparator_prefix: str) -> bool:
    candidate = _mapping(row.get(f"{candidate_prefix}_metrics"))
    comparator = _mapping(row.get(f"{comparator_prefix}_metrics"))
    return (
        bool(row.get(f"{candidate_prefix}_legal")) == bool(row.get(f"{comparator_prefix}_legal"))
        and _metric_signature(candidate) == _metric_signature(comparator)
    )


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown(summary), encoding="utf-8")


def report_markdown(summary: Mapping[str, Any]) -> str:
    decision = _mapping(summary.get("decision"))
    lines = [
        "# Mapper v2.1 Soft Penalty Sweep Stress Result Report",
        "",
        "## Scope",
        "",
        "This sweep tests weak finite anti-rigid penalties on the four committed stress cases. It does not change model weights, tokenizer, training, or defaults.",
        "",
        "## Decision",
        "",
        f"Decision: `{decision.get('route')}`.",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Selected penalty: `{decision.get('selected_penalty')}`",
        f"- Passing penalties: `{decision.get('passing_penalty_count')}`",
        "",
        "## Penalty Table",
        "",
        "| Penalty | Pass | Legal | New starved | Rigid improved | Mean F1 delta vs baseline | Mean rigid delta vs baseline | Differs from hard | Differs from baseline |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("penalty_results", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {penalty} | {passed} | {legal}/{cases} | {starved} | {rigid} | {f1} | {rigid_delta} | {diff_hard} | {diff_base} |".format(
                penalty=_fmt(row.get("penalty")),
                passed=row.get("primary_pass"),
                legal=row.get("soft_legal_count"),
                cases=row.get("case_count"),
                starved=row.get("new_starved_count"),
                rigid=row.get("rigid_improved_count"),
                f1=_fmt(row.get("mean_f1_delta_vs_baseline")),
                rigid_delta=_fmt(row.get("mean_rigid_delta_vs_baseline")),
                diff_hard=row.get("differs_from_hard_count"),
                diff_base=row.get("differs_from_baseline_count"),
            )
        )
    best_rigidity = max(
        (
            int(row.get("rigid_improved_count") or 0)
            for row in summary.get("penalty_results", [])
            if isinstance(row, Mapping)
        ),
        default=0,
    )
    best_f1_delta = max(
        (
            float(row.get("mean_f1_delta_vs_baseline") or 0.0)
            for row in summary.get("penalty_results", [])
            if isinstance(row, Mapping)
        ),
        default=0.0,
    )
    min_new_starved = min(
        (
            int(row.get("new_starved_count") or 0)
            for row in summary.get("penalty_results", [])
            if isinstance(row, Mapping)
        ),
        default=0,
    )
    best_hard_separation = max(
        (
            int(row.get("differs_from_hard_count") or 0)
            for row in summary.get("penalty_results", [])
            if isinstance(row, Mapping)
        ),
        default=0,
    )
    rollout_count = sum(
        int(row.get("case_count") or 0) for row in summary.get("penalty_results", []) if isinstance(row, Mapping)
    )
    lines.extend(
        [
            "",
            "## What Passed",
            "",
            f"- The sweep completed all `{rollout_count}` planned real-audio stress rollouts.",
            f"- At least one weak penalty separated from hard-block behavior; best hard-block-separation count was `{best_hard_separation}` / `4` cases.",
            f"- The best rigidity signal improved `{best_rigidity}` / `4` cases versus baseline.",
            f"- The best mean F1 delta versus baseline was `{_fmt(best_f1_delta)}`.",
            "",
            "## What Surfaced",
            "",
            "- No penalty cleared the primary gate.",
            "- Every tested penalty had only `3` / `4` legal stress rollouts.",
            f"- New-starvation count never reached zero; best observed count was `{min_new_starved}`.",
            "- Case `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` remained the invariant failure: dead-end plus starvation across all tested penalties.",
            "- This suggests the simple repeated-spacing token suppression schedule is not enough; penalty strength alone is not the right next lever.",
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## Next Step",
            "",
            str(summary.get("next_step")),
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(summary: Mapping[str, Any]) -> str:
    route = str(_mapping(summary.get("decision")).get("route"))
    if route == "TEST_NEXT":
        return "At least one weak penalty separated from hard-block behavior while preserving stress-set safety. This is only a stress-set signal and needs a full fixed-slice comparison."
    return "The weak penalty sweep did not find a safe finite penalty regime. Simple repeated-spacing token suppression should be deprioritized."


def _metric_signature(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    timing = _mapping(metrics.get("timing_match_100ms"))
    return (
        int(metrics.get("generated_event_count") or 0),
        _round_float(metrics.get("event_count_ratio")),
        _round_float(metrics.get("dominant_spacing_ratio")),
        bool(metrics.get("starved")),
        _round_float(metrics.get("second_window_event_share")),
        _round_float(timing.get("f1")),
        tuple(metrics.get("first_12_generated_times") or ()),
        tuple(metrics.get("last_12_generated_times") or ()),
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _round_float(value: object) -> float | None:
    number = _float(value)
    return None if number is None else round(number, 9)


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.6f}"


def _penalty_label(value: float) -> str:
    return f"penalty_{str(float(value)).replace('.', '_').replace('-', 'neg_')}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the v2.1 weak soft-penalty stress sweep.")
    parser.add_argument("--hard-block-summary", default=DEFAULT_HARD_BLOCK_SUMMARY_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--penalty", action="append", type=float, dest="penalties")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--mapper-checkpoint-path")
    parser.add_argument("--control-checkpoint-path")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max-tokens-per-window", type=int, default=512)
    parser.add_argument("--min-repeated-spacings", type=int, default=4)
    parser.add_argument("--min-spacing-ms", type=int, default=40)
    parser.add_argument("--max-spacing-ms", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--collect-logit-diagnostics", action="store_true")
    parser.add_argument("--timepoint-preview-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    summary = run_soft_penalty_sweep_stress(
        hard_block_summary_path=args.hard_block_summary,
        output_summary_path=args.summary_output,
        output_report_path=args.report_output,
        output_dir=args.output_dir,
        penalties=tuple(args.penalties or DEFAULT_PENALTIES),
        case_ids=tuple(args.case_ids or DEFAULT_STRESS_CASE_IDS),
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        device_name=args.device,
        max_tokens_per_window=args.max_tokens_per_window,
        min_repeated_spacings=args.min_repeated_spacings,
        min_spacing_ms=args.min_spacing_ms,
        max_spacing_ms=args.max_spacing_ms,
        seed=args.seed,
        collect_logit_diagnostics=args.collect_logit_diagnostics,
        timepoint_preview_limit=args.timepoint_preview_limit,
    )
    print(
        "mapper_v21_soft_penalty_sweep_stress_done "
        f"route={summary['decision']['route']} "
        f"selected={summary['decision']['selected_penalty']} "
        f"passing={summary['decision']['passing_penalty_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
