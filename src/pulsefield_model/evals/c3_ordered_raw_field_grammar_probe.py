from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = 1
RAW_PREFIX = "RAW|"
REF_PREFIX = "REF|"
RES_PREFIX = "RES|"
RAW_BEGIN = "RAW_BEGIN"
RAW_END = "RAW_END"
DEFAULT_SMOOTHING_ALPHA = 0.1


def normalize_token_vocab(token_vocab: Mapping[str, Any]) -> dict[int, str]:
    """Normalize sidecar vocab formats into token-id -> token text."""
    id_to_token: dict[int, str] = {}
    for key, value in token_vocab.items():
        if isinstance(value, int):
            token_id = int(value)
            token_text = str(key)
        else:
            token_id = int(key)
            token_text = str(value)
        if token_id <= 0:
            raise ValueError(f"C3 token ids must be positive, got {token_id}")
        if token_id in id_to_token and id_to_token[token_id] != token_text:
            raise ValueError(f"duplicate C3 token id {token_id}")
        id_to_token[token_id] = token_text
    if not id_to_token:
        raise ValueError("C3 sidecar token_vocab is empty")
    return id_to_token


def encode_ordered_field_symbols(token: str) -> tuple[str, ...]:
    if token.startswith(RAW_PREFIX):
        payload = token[len(RAW_PREFIX) :]
        parts = payload.split(":") if payload else [""]
        return (RAW_BEGIN, *(f"RAW_PART|{index}|{part}" for index, part in enumerate(parts)), RAW_END)
    if token.startswith(REF_PREFIX):
        return (f"REF_TOKEN|{token[len(REF_PREFIX):]}",)
    if token.startswith(RES_PREFIX):
        return (f"RES_TOKEN|{token[len(RES_PREFIX):]}",)
    return (f"OTHER_TOKEN|{token}",)


def decode_ordered_field_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    decoded: list[str] = []
    index = 0
    while index < len(symbols):
        symbol = str(symbols[index])
        if symbol == RAW_BEGIN:
            index += 1
            parts: list[str] = []
            expected_part_index = 0
            while index < len(symbols) and symbols[index] != RAW_END:
                part_symbol = str(symbols[index])
                prefix = f"RAW_PART|{expected_part_index}|"
                if not part_symbol.startswith(prefix):
                    raise ValueError(f"invalid ordered RAW part at symbol {index}: {part_symbol!r}")
                parts.append(part_symbol[len(prefix) :])
                expected_part_index += 1
                index += 1
            if index >= len(symbols) or symbols[index] != RAW_END:
                raise ValueError("unterminated ordered RAW token")
            decoded.append(f"{RAW_PREFIX}{':'.join(parts)}")
            index += 1
            continue
        if symbol.startswith("REF_TOKEN|"):
            decoded.append(f"{REF_PREFIX}{symbol[len('REF_TOKEN|'):]}")
        elif symbol.startswith("RES_TOKEN|"):
            decoded.append(f"{RES_PREFIX}{symbol[len('RES_TOKEN|'):]}")
        elif symbol.startswith("OTHER_TOKEN|"):
            decoded.append(symbol[len("OTHER_TOKEN|") :])
        else:
            raise ValueError(f"invalid ordered C3 symbol at {index}: {symbol!r}")
        index += 1
    return tuple(decoded)


def token_kind(token: str) -> str:
    if token.startswith(RAW_PREFIX):
        return "RAW"
    if token.startswith(REF_PREFIX):
        return "REF"
    if token.startswith(RES_PREFIX):
        return "RES"
    return "OTHER"


def raw_payload_parts(token: str) -> tuple[str, ...]:
    if not token.startswith(RAW_PREFIX):
        raise ValueError(f"expected RAW token, got {token!r}")
    payload = token[len(RAW_PREFIX) :]
    return tuple(payload.split(":") if payload else [""])


def run_ordered_raw_field_grammar_probe(
    *,
    sidecar_path: Path,
    max_windows: int | None = None,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> dict[str, Any]:
    start = time.monotonic()
    sidecar = _load_sidecar(sidecar_path)
    id_to_token = normalize_token_vocab(sidecar["token_vocab"])
    windows = list(sidecar["windows"])
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    if not windows:
        raise ValueError("C3 sidecar has no windows to audit")

    exact_by_split: dict[str, list[str]] = defaultdict(list)
    ordered_by_split: dict[str, list[str]] = defaultdict(list)
    raw_exact_by_split: dict[str, list[str]] = defaultdict(list)
    raw_ordered_by_split: dict[str, list[str]] = defaultdict(list)
    token_kind_counts: Counter[str] = Counter()
    ordered_symbol_counts: Counter[str] = Counter()
    raw_part_counts_by_position: dict[int, Counter[str]] = defaultdict(Counter)
    ref_distance_counts: Counter[str] = Counter()
    ref_length_counts: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    missing_token_ids: Counter[int] = Counter()
    token_count = 0
    boundary_risk_token_count = 0

    for window_index, window in enumerate(windows):
        split = str(window.get("split") or "unknown")
        token_ids = tuple(int(token_id) for token_id in window.get("token_ids", ()))
        exact_tokens: list[str] = []
        ordered_symbols: list[str] = []
        boundary_risk_token_count += int(window.get("boundary_risk_token_count") or 0)

        for token_id in token_ids:
            token = id_to_token.get(token_id)
            if token is None:
                missing_token_ids[token_id] += 1
                continue
            exact_tokens.append(token)
            encoded = encode_ordered_field_symbols(token)
            ordered_symbols.extend(encoded)
            kind = token_kind(token)
            token_kind_counts[kind] += 1
            ordered_symbol_counts.update(encoded)
            if kind == "RAW":
                raw_exact_by_split[split].append(token)
                raw_ordered_by_split[split].extend(encoded)
                for position, part in enumerate(raw_payload_parts(token)):
                    raw_part_counts_by_position[int(position)][part] += 1
            elif kind == "REF":
                _update_ref_stats(token, distance_counts=ref_distance_counts, length_counts=ref_length_counts)

        token_count += len(exact_tokens)
        exact_by_split[split].extend(exact_tokens)
        ordered_by_split[split].extend(ordered_symbols)
        try:
            reconstructed = decode_ordered_field_symbols(ordered_symbols)
        except ValueError as exc:
            reconstructed = ()
            if len(mismatches) < 10:
                mismatches.append(
                    {
                        "window_index": window_index,
                        "reason": str(exc),
                        "beatmap_path": window.get("beatmap_path"),
                        "window_start_ms": window.get("window_start_ms"),
                    }
                )
        if tuple(exact_tokens) != reconstructed and len(mismatches) < 10:
            mismatches.append(
                {
                    "window_index": window_index,
                    "reason": "decoded_tokens_mismatch",
                    "beatmap_path": window.get("beatmap_path"),
                    "window_start_ms": window.get("window_start_ms"),
                    "expected": exact_tokens[:12],
                    "decoded": list(reconstructed[:12]),
                }
            )

    exact_by_split["all"] = _flatten_split_sequences(exact_by_split)
    ordered_by_split["all"] = _flatten_split_sequences(ordered_by_split)
    raw_exact_by_split["all"] = _flatten_split_sequences(raw_exact_by_split)
    raw_ordered_by_split["all"] = _flatten_split_sequences(raw_ordered_by_split)

    exact_bits = _split_bits(exact_by_split, alpha=smoothing_alpha)
    ordered_bits = _split_bits(ordered_by_split, alpha=smoothing_alpha)
    raw_exact_bits = _split_bits(raw_exact_by_split, alpha=smoothing_alpha)
    raw_ordered_bits = _split_bits(raw_ordered_by_split, alpha=smoothing_alpha)
    selected_split = _selected_eval_split(exact_by_split)
    selected_exact = exact_bits.get(selected_split, {})
    selected_ordered = ordered_bits.get(selected_split, {})
    selected_raw_exact = raw_exact_bits.get(selected_split, {})
    selected_raw_ordered = raw_ordered_bits.get(selected_split, {})

    raw_token_count = int(token_kind_counts.get("RAW", 0))
    ordered_symbol_count = int(sum(len(values) for split, values in ordered_by_split.items() if split != "all"))
    exact_vocab = set(exact_by_split["all"])
    ordered_vocab = set(ordered_by_split["all"])
    raw_exact_vocab = set(raw_exact_by_split["all"])
    raw_ordered_vocab = set(raw_ordered_by_split["all"])
    raw_exact_bits_per_raw_token = _float_or_none(selected_raw_exact.get("bits_per_item"))
    raw_ordered_bits_per_raw_token = _safe_divide(
        selected_raw_ordered.get("total_bits"),
        selected_raw_exact.get("item_count"),
    )
    ordered_bits_per_original_token = _safe_divide(
        selected_ordered.get("total_bits"),
        selected_exact.get("item_count"),
    )

    pass_criteria = {
        "reconstruction_pass": len(mismatches) == 0 and not missing_token_ids,
        "raw_bits_reduction_pass": (
            raw_exact_bits_per_raw_token is not None
            and raw_ordered_bits_per_raw_token is not None
            and raw_ordered_bits_per_raw_token <= raw_exact_bits_per_raw_token - 0.05
        ),
        "sequence_expansion_bounded_pass": _safe_divide(ordered_symbol_count, token_count) is not None
        and _safe_divide(ordered_symbol_count, token_count) <= 2.0,
        "label_space_reduction_pass": len(raw_ordered_vocab) < len(raw_exact_vocab),
        "no_missing_token_ids_pass": not missing_token_ids,
    }
    route = _route_from_pass_criteria(pass_criteria)

    elapsed_s = time.monotonic() - start
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "C3 ordered RAW field grammar probe",
        "sidecar_path": str(sidecar_path),
        "max_windows": max_windows,
        "elapsed_s": elapsed_s,
        "sidecar": {
            "contract": sidecar.get("contract"),
            "schema_version": sidecar.get("schema_version"),
            "token_id_base": sidecar.get("token_id_base"),
            "token_pad_id": sidecar.get("token_pad_id"),
            "mapper_window_ms": sidecar.get("mapper_window_ms"),
            "timing_contract": sidecar.get("timing_contract"),
            "window_anchor_mode": sidecar.get("window_anchor_mode"),
        },
        "dataset": {
            "window_count": len(windows),
            "token_count": token_count,
            "ordered_symbol_count": ordered_symbol_count,
            "raw_token_count": raw_token_count,
            "ref_token_count": int(token_kind_counts.get("REF", 0)),
            "res_token_count": int(token_kind_counts.get("RES", 0)),
            "other_token_count": int(token_kind_counts.get("OTHER", 0)),
            "boundary_risk_token_count": boundary_risk_token_count,
            "boundary_risk_token_share": _safe_divide(boundary_risk_token_count, token_count),
            "split_token_counts": {split: len(values) for split, values in sorted(exact_by_split.items())},
        },
        "reconstruction": {
            "mismatch_count": len(mismatches),
            "missing_token_id_count": int(sum(missing_token_ids.values())),
            "missing_token_ids": {str(token_id): count for token_id, count in missing_token_ids.most_common(20)},
            "examples": mismatches,
        },
        "label_space": {
            "exact_c3_vocab_size": len(exact_vocab),
            "ordered_symbol_vocab_size": len(ordered_vocab),
            "raw_exact_vocab_size": len(raw_exact_vocab),
            "raw_ordered_symbol_vocab_size": len(raw_ordered_vocab),
            "raw_part_position_vocab_sizes": {
                str(position): len(counter) for position, counter in sorted(raw_part_counts_by_position.items())
            },
        },
        "bits": {
            "selected_split": selected_split,
            "smoothing_alpha": smoothing_alpha,
            "exact_c3": exact_bits,
            "ordered_field": ordered_bits,
            "raw_exact": raw_exact_bits,
            "raw_ordered_field": raw_ordered_bits,
            "selected_metrics": {
                "exact_bits_per_token": selected_exact.get("bits_per_item"),
                "ordered_bits_per_symbol": selected_ordered.get("bits_per_item"),
                "ordered_bits_per_original_token": ordered_bits_per_original_token,
                "raw_exact_bits_per_raw_token": raw_exact_bits_per_raw_token,
                "raw_ordered_bits_per_symbol": selected_raw_ordered.get("bits_per_item"),
                "raw_ordered_bits_per_raw_token": raw_ordered_bits_per_raw_token,
            },
        },
        "sequence": {
            "ordered_to_exact_token_ratio": _safe_divide(ordered_symbol_count, token_count),
            "raw_ordered_to_raw_token_ratio": _safe_divide(len(raw_ordered_by_split["all"]), raw_token_count),
        },
        "raw_fields": {
            str(position): {
                "value_count": len(counter),
                "entropy_bits": _entropy_bits(counter),
                "top_values": [
                    {"value": value, "count": count}
                    for value, count in counter.most_common(12)
                ],
            }
            for position, counter in sorted(raw_part_counts_by_position.items())
        },
        "ref_stats": {
            "distance_buckets": dict(ref_distance_counts.most_common()),
            "length_buckets": dict(ref_length_counts.most_common()),
            "cross_window_reference_share": None,
            "cross_window_reference_note": (
                "Exact cross-window reference share is not encoded in the mapper sidecar rows; "
                "use the P5 trace report for that statistic. This probe preserves REF tokens exactly."
            ),
        },
        "pass_criteria": pass_criteria,
        "decision": {
            "route": route,
            "reason": _decision_reason(pass_criteria),
            "next_step": _next_step(route),
        },
    }


def write_ordered_raw_field_report(summary: Mapping[str, Any], path: Path) -> None:
    selected = summary["bits"]["selected_metrics"]
    dataset = summary["dataset"]
    reconstruction = summary["reconstruction"]
    label_space = summary["label_space"]
    sequence = summary["sequence"]
    criteria = summary["pass_criteria"]
    decision = summary["decision"]
    lines = [
        "# C3 Ordered RAW Field Grammar Probe Result Report",
        "",
        "## Scope",
        "",
        "This pass tests whether exact C3 side-stream tokens can be decomposed into an ordered RAW field grammar without losing reconstruction. It is an artifact-only target-design probe: no mapper training, rollout, or inference input path is changed.",
        "",
        "## Result",
        "",
        f"Decision: `{decision['route']}`.",
        "",
        f"- sidecar: `{summary['sidecar_path']}`",
        f"- windows: `{dataset['window_count']}`",
        f"- exact C3 tokens: `{dataset['token_count']}`",
        f"- ordered symbols: `{dataset['ordered_symbol_count']}`",
        f"- ordered/exact sequence ratio: `{_fmt(sequence['ordered_to_exact_token_ratio'])}`",
        f"- reconstruction mismatches: `{reconstruction['mismatch_count']}`",
        f"- missing token ids: `{reconstruction['missing_token_id_count']}`",
        f"- selected split: `{summary['bits']['selected_split']}`",
        f"- exact C3 bits/token: `{_fmt(selected['exact_bits_per_token'])}`",
        f"- ordered bits/original token: `{_fmt(selected['ordered_bits_per_original_token'])}`",
        f"- RAW exact bits/RAW token: `{_fmt(selected['raw_exact_bits_per_raw_token'])}`",
        f"- RAW ordered bits/RAW token: `{_fmt(selected['raw_ordered_bits_per_raw_token'])}`",
        "",
        "## Label Space",
        "",
        f"- exact C3 vocab size: `{label_space['exact_c3_vocab_size']}`",
        f"- ordered symbol vocab size: `{label_space['ordered_symbol_vocab_size']}`",
        f"- RAW exact vocab size: `{label_space['raw_exact_vocab_size']}`",
        f"- RAW ordered symbol vocab size: `{label_space['raw_ordered_symbol_vocab_size']}`",
        f"- RAW part-position vocab sizes: `{label_space['raw_part_position_vocab_sizes']}`",
        "",
        "## Pass Criteria",
        "",
    ]
    for key, value in criteria.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _interpretation(summary),
            "",
            "## What Passed",
            "",
        ]
    )
    passed = _passed_items(criteria)
    lines.extend(f"- {item}" for item in passed)
    lines.extend(["", "## What Surfaced", ""])
    surfaced = _surfaced_items(summary)
    lines.extend(f"- {item}" for item in surfaced)
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            decision["next_step"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_sidecar(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid C3 sidecar JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("C3 sidecar must be a JSON object")
    if not isinstance(payload.get("token_vocab"), dict):
        raise ValueError("C3 sidecar must contain token_vocab")
    if not isinstance(payload.get("windows"), list):
        raise ValueError("C3 sidecar must contain windows")
    return payload


def _update_ref_stats(token: str, *, distance_counts: Counter[str], length_counts: Counter[str]) -> None:
    parts = token[len(REF_PREFIX) :].split("|")
    if len(parts) != 3:
        distance_counts["malformed"] += 1
        length_counts["malformed"] += 1
        return
    _match_type, distance, length = parts
    distance_counts[_count_bucket(int(distance))] += 1
    length_counts[_count_bucket(int(length))] += 1


def _count_bucket(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 4:
        return "2to4"
    if value <= 16:
        return "5to16"
    if value <= 64:
        return "17to64"
    if value <= 256:
        return "65to256"
    return "257plus"


def _flatten_split_sequences(by_split: Mapping[str, Sequence[str]]) -> list[str]:
    values: list[str] = []
    for split, sequence in by_split.items():
        if split == "all":
            continue
        values.extend(sequence)
    return values


def _selected_eval_split(by_split: Mapping[str, Sequence[str]]) -> str:
    if by_split.get("test"):
        return "test"
    if by_split.get("valid"):
        return "valid"
    return "all"


def _split_bits(by_split: Mapping[str, Sequence[str]], *, alpha: float) -> dict[str, dict[str, Any]]:
    train_sequence = tuple(by_split.get("train") or by_split.get("all") or ())
    train_counts = Counter(train_sequence)
    result: dict[str, dict[str, Any]] = {}
    for split, sequence in sorted(by_split.items()):
        if not sequence:
            continue
        total_bits = _nll_bits(sequence, train_counts, alpha=alpha)
        result[split] = {
            "item_count": len(sequence),
            "vocab_size": len(set(sequence)),
            "total_bits": total_bits,
            "bits_per_item": _safe_divide(total_bits, len(sequence)),
        }
    return result


def _nll_bits(sequence: Sequence[str], train_counts: Mapping[str, int], *, alpha: float) -> float:
    if not sequence:
        return 0.0
    vocab = set(train_counts)
    vocab.update(sequence)
    vocab_size = len(vocab) + 1
    total = float(sum(train_counts.values()))
    denominator = total + float(alpha) * float(vocab_size)
    bits = 0.0
    for item in sequence:
        probability = (float(train_counts.get(item, 0)) + float(alpha)) / denominator
        bits -= math.log2(probability)
    return bits


def _entropy_bits(counter: Mapping[str, int]) -> float | None:
    total = float(sum(counter.values()))
    if total <= 0:
        return None
    entropy = 0.0
    for count in counter.values():
        probability = float(count) / total
        entropy -= probability * math.log2(probability)
    return entropy


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _route_from_pass_criteria(criteria: Mapping[str, bool]) -> str:
    if not criteria.get("reconstruction_pass"):
        return "KILL"
    if criteria.get("raw_bits_reduction_pass") and criteria.get("sequence_expansion_bounded_pass"):
        return "TEST_NEXT"
    if criteria.get("raw_bits_reduction_pass") or criteria.get("label_space_reduction_pass"):
        return "MUTATE"
    return "KILL"


def _decision_reason(criteria: Mapping[str, bool]) -> str:
    failed = [key for key, value in criteria.items() if not value]
    if not failed:
        return "all ordered RAW field grammar gates passed"
    return "failed checks: " + ", ".join(failed)


def _next_step(route: str) -> str:
    if route == "TEST_NEXT":
        return "Create a bounded ordered C3 teacher-forcing smoke card before any full grammar replacement."
    if route == "MUTATE":
        return "Do not train yet; refine the ordered grammar or compare against v3/v2.1 target complexity before spending model runtime."
    return "Stop this ordered C3 target-grammar path and pivot to v3 continuation/grammar repair or v2.1 grammar improvement."


def _interpretation(summary: Mapping[str, Any]) -> str:
    criteria = summary["pass_criteria"]
    if not criteria["reconstruction_pass"]:
        return (
            "The ordered field representation is not lossless, so it cannot be used as a C3 target grammar candidate."
        )
    if criteria["raw_bits_reduction_pass"] and criteria["sequence_expansion_bounded_pass"]:
        return (
            "The ordered field representation preserves C3 side-stream tokens and reduces RAW cost within the sequence "
            "budget, so it deserves one bounded teacher-forcing smoke."
        )
    if criteria["label_space_reduction_pass"]:
        return (
            "The ordered field representation is lossless and reduces label space, but its bit or sequence budget is not "
            "yet good enough for training. Treat this as a mutation signal, not promotion."
        )
    return (
        "The ordered field representation is lossless but does not improve the target complexity enough to justify "
        "more C3 mapper-side work."
    )


def _passed_items(criteria: Mapping[str, bool]) -> list[str]:
    items: list[str] = []
    if criteria.get("reconstruction_pass"):
        items.append("Exact side-stream reconstruction is lossless under ordered RAW field encoding.")
    if criteria.get("no_missing_token_ids_pass"):
        items.append("All sidecar token ids resolved through the token vocabulary.")
    if criteria.get("label_space_reduction_pass"):
        items.append("RAW field splitting reduces the distinct RAW-side symbol space.")
    if criteria.get("raw_bits_reduction_pass"):
        items.append("RAW field splitting reduces charged RAW bits per original RAW token.")
    if criteria.get("sequence_expansion_bounded_pass"):
        items.append("Ordered sequence expansion stays within the configured budget.")
    return items or ["No positive gate passed."]


def _surfaced_items(summary: Mapping[str, Any]) -> list[str]:
    selected = summary["bits"]["selected_metrics"]
    sequence = summary["sequence"]
    criteria = summary["pass_criteria"]
    items = [
        "Ordered RAW field splitting is target-design evidence only; it does not prove mapper learnability.",
        (
            "Ordered/exact sequence ratio is "
            f"{_fmt(sequence['ordered_to_exact_token_ratio'])}, so any training card must budget sequence length explicitly."
        ),
        (
            "RAW ordered bits per RAW token are "
            f"{_fmt(selected['raw_ordered_bits_per_raw_token'])} versus exact RAW { _fmt(selected['raw_exact_bits_per_raw_token']) }."
        ),
    ]
    if not criteria.get("raw_bits_reduction_pass"):
        items.append("The RAW bit-reduction gate did not pass.")
    if not criteria.get("sequence_expansion_bounded_pass"):
        items.append("The sequence-expansion gate did not pass.")
    if summary["ref_stats"]["cross_window_reference_share"] is None:
        items.append("The mapper sidecar preserves REF tokens but does not itself expose exact cross-window reference share.")
    return items


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a C3 ordered RAW field grammar target-design probe.")
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    args = parser.parse_args(argv)

    summary = run_ordered_raw_field_grammar_probe(
        sidecar_path=args.sidecar,
        max_windows=args.max_windows,
        smoothing_alpha=args.smoothing_alpha,
    )
    _write_json(args.summary_output, summary)
    write_ordered_raw_field_report(summary, args.report_output)
    print(
        "c3_ordered_raw_field_grammar_probe_done "
        f"route={summary['decision']['route']} "
        f"mismatches={summary['reconstruction']['mismatch_count']} "
        f"ordered_ratio={_fmt(summary['sequence']['ordered_to_exact_token_ratio'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
