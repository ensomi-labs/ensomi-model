from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.evals.c3_auxiliary_diagnostics import (
    build_datasets_from_config,
    iter_dataset_window_keys,
    load_token_vocab,
    token_kind_lookup,
)


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_TOP_PER_KIND = 512
DEFAULT_KINDS = ("RAW", "REF", "RES")


def train_token_counts_from_config(config_path: Path) -> tuple[Counter[int], dict[str, Any]]:
    datasets = build_datasets_from_config(config_path)
    counts: Counter[int] = Counter()
    for beatmap_path, window_start_ms in iter_dataset_window_keys(datasets.train_dataset):
        token_ids = datasets.token_lookup.get(beatmap_path, {}).get(window_start_ms)
        if not token_ids:
            continue
        counts.update({int(token_id) for token_id in token_ids if int(token_id) > 0})
    coverage_context = {
        "train_dataset": datasets.train_dataset,
        "eval_dataset": datasets.eval_dataset,
        "token_lookup": datasets.token_lookup,
        "token_by_id": datasets.token_by_id,
    }
    return counts, coverage_context


def select_top_tokens_per_kind(
    token_counts: Mapping[int, int],
    token_by_id: Mapping[int, str],
    *,
    top_per_kind: int,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[int, int]:
    if int(top_per_kind) <= 0:
        raise ValueError("top_per_kind must be positive")
    allowed_kinds = tuple(str(kind) for kind in kinds)
    kind_by_id = token_kind_lookup(token_by_id)
    by_kind: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for token_id, count in token_counts.items():
        kind = kind_by_id.get(int(token_id), "UNKNOWN")
        if kind in allowed_kinds and int(count) > 0:
            by_kind[kind].append((int(token_id), int(count)))

    remap: dict[int, int] = {}
    next_id = 1
    for kind in allowed_kinds:
        rows = sorted(by_kind.get(kind, ()), key=lambda item: (-item[1], item[0]))
        for token_id, _count in rows[: int(top_per_kind)]:
            remap[token_id] = next_id
            next_id += 1
    return remap


def reduce_sidecar_payload(
    source_payload: Mapping[str, Any],
    *,
    token_remap: Mapping[int, int],
    token_by_id: Mapping[int, str],
    top_per_kind: int,
    source_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    windows = source_payload.get("windows")
    if not isinstance(windows, list):
        raise ValueError("source sidecar must contain a windows list")
    selected_original_ids = set(int(token_id) for token_id in token_remap)
    reduced_vocab = {
        token_by_id[original_id]: int(new_id)
        for original_id, new_id in sorted(token_remap.items(), key=lambda item: item[1])
        if original_id in token_by_id
    }
    reduced_windows: list[dict[str, Any]] = []
    source_raw_token_count = 0
    retained_raw_token_count = 0
    source_positive_label_count = 0
    retained_positive_label_count = 0
    retained_window_count = 0
    for row_number, row in enumerate(windows):
        if not isinstance(row, Mapping):
            raise ValueError(f"source sidecar row {row_number} must be an object")
        token_ids_raw = row.get("token_ids")
        if not isinstance(token_ids_raw, list):
            raise ValueError(f"source sidecar row {row_number} missing token_ids list")
        token_ids = [int(token_id) for token_id in token_ids_raw if int(token_id) > 0]
        reduced_ids = [int(token_remap[token_id]) for token_id in token_ids if token_id in selected_original_ids]
        source_raw_token_count += len(token_ids)
        retained_raw_token_count += len(reduced_ids)
        source_positive_label_count += len(set(token_ids))
        retained_positive_label_count += len(set(reduced_ids))
        if reduced_ids:
            retained_window_count += 1
        reduced_row = dict(row)
        reduced_row["token_ids"] = reduced_ids
        reduced_windows.append(reduced_row)

    payload = dict(source_payload)
    payload["contract"] = f"{source_payload.get('contract', 'unknown')}_reduced_per_kind_top{int(top_per_kind)}"
    payload["token_vocab"] = reduced_vocab
    payload["token_id_base"] = 1
    payload["token_pad_id"] = 0
    payload["windows"] = reduced_windows
    payload["reduction"] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source_sidecar": source_path.as_posix(),
        "top_per_kind": int(top_per_kind),
        "kinds": list(DEFAULT_KINDS),
        "source_vocab_size": int(len(token_by_id)),
        "reduced_vocab_size": int(len(reduced_vocab)),
        "token_remap": {str(original_id): int(new_id) for original_id, new_id in sorted(token_remap.items())},
    }
    stats = {
        "source_window_count": int(len(windows)),
        "retained_window_count": int(retained_window_count),
        "source_raw_token_count": int(source_raw_token_count),
        "retained_raw_token_count": int(retained_raw_token_count),
        "source_positive_label_count": int(source_positive_label_count),
        "retained_positive_label_count": int(retained_positive_label_count),
        "raw_token_coverage": _safe_divide(retained_raw_token_count, source_raw_token_count),
        "positive_label_coverage": _safe_divide(retained_positive_label_count, source_positive_label_count),
    }
    return payload, stats


def dataset_coverage(
    dataset: Any,
    token_lookup: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    selected_token_ids: set[int],
    kind_by_id: Mapping[int, str],
) -> dict[str, Any]:
    source_positive_label_count = 0
    retained_positive_label_count = 0
    source_kind_counts: Counter[str] = Counter()
    retained_kind_counts: Counter[str] = Counter()
    positive_sample_count = 0
    retained_positive_sample_count = 0
    for beatmap_path, window_start_ms in iter_dataset_window_keys(dataset):
        token_ids = token_lookup.get(beatmap_path, {}).get(window_start_ms)
        if not token_ids:
            continue
        unique_ids = {int(token_id) for token_id in token_ids if int(token_id) > 0}
        if not unique_ids:
            continue
        positive_sample_count += 1
        retained_ids = unique_ids.intersection(selected_token_ids)
        if retained_ids:
            retained_positive_sample_count += 1
        source_positive_label_count += len(unique_ids)
        retained_positive_label_count += len(retained_ids)
        for token_id in unique_ids:
            source_kind_counts[kind_by_id.get(int(token_id), "UNKNOWN")] += 1
        for token_id in retained_ids:
            retained_kind_counts[kind_by_id.get(int(token_id), "UNKNOWN")] += 1
    kind_coverage = {
        kind: _safe_divide(retained_kind_counts.get(kind, 0), source_kind_counts.get(kind, 0))
        for kind in sorted(source_kind_counts)
    }
    return {
        "positive_sample_count": int(positive_sample_count),
        "retained_positive_sample_count": int(retained_positive_sample_count),
        "source_positive_label_count": int(source_positive_label_count),
        "retained_positive_label_count": int(retained_positive_label_count),
        "positive_label_coverage": _safe_divide(retained_positive_label_count, source_positive_label_count),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "retained_kind_counts": dict(sorted(retained_kind_counts.items())),
        "kind_coverage": kind_coverage,
    }


def generate_reduced_sidecar(
    *,
    config_path: Path,
    source_sidecar_path: Path,
    output_sidecar_path: Path,
    top_per_kind: int = DEFAULT_TOP_PER_KIND,
) -> dict[str, Any]:
    start = time.monotonic()
    source_payload = json.loads(source_sidecar_path.read_text(encoding="utf-8"))
    token_by_id = load_token_vocab(source_sidecar_path)
    token_counts, coverage_context = train_token_counts_from_config(config_path)
    token_remap = select_top_tokens_per_kind(
        token_counts,
        token_by_id,
        top_per_kind=top_per_kind,
    )
    reduced_payload, sidecar_stats = reduce_sidecar_payload(
        source_payload,
        token_remap=token_remap,
        token_by_id=token_by_id,
        top_per_kind=top_per_kind,
        source_path=source_sidecar_path,
    )
    output_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    output_sidecar_path.write_text(json.dumps(reduced_payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    kind_by_id = token_kind_lookup(token_by_id)
    selected_ids = set(token_remap)
    selected_by_kind: Counter[str] = Counter()
    for token_id in selected_ids:
        selected_by_kind[kind_by_id.get(int(token_id), "UNKNOWN")] += 1
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "config_path": config_path.as_posix(),
        "source_sidecar_path": source_sidecar_path.as_posix(),
        "output_sidecar_path": output_sidecar_path.as_posix(),
        "top_per_kind": int(top_per_kind),
        "source_vocab_size": int(len(token_by_id)),
        "reduced_vocab_size": int(len(token_remap)),
        "selected_by_kind": dict(sorted(selected_by_kind.items())),
        "sidecar_stats": sidecar_stats,
        "train_coverage": dataset_coverage(
            coverage_context["train_dataset"],
            coverage_context["token_lookup"],
            selected_token_ids=selected_ids,
            kind_by_id=kind_by_id,
        ),
        "eval_coverage": dataset_coverage(
            coverage_context["eval_dataset"],
            coverage_context["token_lookup"],
            selected_token_ids=selected_ids,
            kind_by_id=kind_by_id,
        ),
        "elapsed_s": time.monotonic() - start,
    }
    return summary


def write_summary(summary: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a reduced per-kind C3 sidecar.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-sidecar", required=True, type=Path)
    parser.add_argument("--output-sidecar", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--top-per-kind", type=int, default=DEFAULT_TOP_PER_KIND)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = generate_reduced_sidecar(
        config_path=args.config,
        source_sidecar_path=args.source_sidecar,
        output_sidecar_path=args.output_sidecar,
        top_per_kind=args.top_per_kind,
    )
    write_summary(summary, args.summary_output)
    print(
        "c3_reduced_sidecar "
        f"vocab={summary['reduced_vocab_size']} "
        f"eval_coverage={summary['eval_coverage']['positive_label_coverage']:.6f} "
        f"output={args.output_sidecar.as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
