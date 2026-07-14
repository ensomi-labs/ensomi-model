from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pulsefield_model.evals.inference_bundle_profiler import _sha256_json
from pulsefield_model.evals.mapper_render_reamber import (
    plan_named_spans,
    render_named_spans,
)
from pulsefield_model.events.canonical import CanonicalTimepoint, LaneAction
from pulsefield_model.inference.audio_probe import audio_length_ms_from_file
from pulsefield_model.inference.defaults import DEFAULT_MAPPER_MODEL_ID
from pulsefield_model.inference.osu_export import OsuExportMetadata, format_osu_export
from pulsefield_model.inference.routed_backend import RoutedInferenceBackend
from pulsefield_model.inference.stream_with_cache import DecoderWindow, StreamWithCacheConfig
from pulsefield_model.osu_core.hitobjects import ManiaHitObjectKind, parse_mania_hit_objects
from pulsefield_model.osu_core.metadata import parse_osu_metadata


DEFAULT_SUMMARY_PATH = Path("artifacts/evals/inference_bundle_mps_profile/summary.json")
DEFAULT_SOURCE_BEATMAP_PATH = Path(
    "dataset/0/1086533/Shiggy Jr. - Oyasumi (Cut Ver.) (Pairoxd) [Hard].osu"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/evals/inference_bundle_mps_profile/beatmap")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce a profiled bundle token stream and render it as an osu!mania beatmap.",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--source-beatmap", type=Path, default=DEFAULT_SOURCE_BEATMAP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


async def _replay_bundle(
    config: dict[str, Any],
) -> tuple[list[Any], Any]:
    audio_path = Path(config["audio_path"]).expanduser().resolve()
    audio_length_ms = int(config["audio_length_ms"])
    observed_length_ms = audio_length_ms_from_file(audio_path)
    if observed_length_ms != audio_length_ms:
        raise RuntimeError(
            f"audio duration changed: summary={audio_length_ms}, observed={observed_length_ms}",
        )

    stream_config = StreamWithCacheConfig(
        mapper_checkpoint_path=Path(config["mapper_checkpoint_path"]),
        control_checkpoint_path=Path(config["control_checkpoint_path"]),
        mapper_profile=str(config["mapper_profile"]),
        device=str(config["device"]),
        beatthis_checkpoint=str(config["beatthis_checkpoint"]),
        beatthis_device=str(config["beatthis_device"]),
        token_send_interval_s=float(config["token_send_interval_s"]),
        max_tokens=int(config["max_tokens"]),
        temperature=float(config["temperature"]),
        top_p=None,
        use_incremental_mapper_decode=bool(config["use_incremental_mapper_decode"]),
        seed=None if config["seed"] is None else int(config["seed"]),
    )
    backend = RoutedInferenceBackend(stream_config)
    session_id = "bundle-profile-beatmap-replay"
    started = False
    prepared = False
    tokens: list[Any] = []
    timing_grid = None
    try:
        print("beatmap_replay status=startup", flush=True)
        await backend.startup()
        started = True
        await backend.mount_model(DEFAULT_MAPPER_MODEL_ID)
        print("beatmap_replay status=prepare_audio", flush=True)
        await backend.prepare_audio(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            difficulty=float(config["difficulty"]),
            route="mapper",
        )
        prepared = True
        print("beatmap_replay status=stream", flush=True)
        async for token in backend.iter_hitobject_tokens(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            window=DecoderWindow(start_ms=0, end_ms=8_000),
        ):
            tokens.append(token)

        mapper_backend = backend.mapper_backend
        session_runtime = mapper_backend._session_runtimes[session_id]
        if session_runtime.audio_cache is not None:
            timing_grid = session_runtime.audio_cache.timing_grid
        print(f"beatmap_replay status=stream_done protocol_tokens={len(tokens)}", flush=True)
    finally:
        if prepared:
            await backend.reset_session(session_id)
        if started:
            await backend.shutdown()
    return tokens, timing_grid


def _canonical_timepoints(tokens: list[Any]) -> list[CanonicalTimepoint]:
    return [
        CanonicalTimepoint(
            time_ms=int(token.ms_in_ref_audio),
            lane_actions=tuple(LaneAction(str(action)) for action in token.actions),
        )
        for token in tokens
    ]


def _object_summary(beatmap_path: Path) -> dict[str, Any]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    lane_counts = Counter(int(obj.lane) for obj in hitobjects)
    kind_counts = Counter(str(obj.kind.value) for obj in hitobjects)
    return {
        "hitobject_count": len(hitobjects),
        "tap_count": int(kind_counts[ManiaHitObjectKind.TAP.value]),
        "hold_count": int(kind_counts[ManiaHitObjectKind.HOLD.value]),
        "lane_hitobject_counts": [int(lane_counts[lane]) for lane in range(4)],
        "first_hitobject_ms": min(float(obj.start_time_ms) for obj in hitobjects),
        "last_hitobject_end_ms": max(float(obj.end_time_ms) for obj in hitobjects),
    }


def _write_outputs(
    *,
    summary_path: Path,
    source_beatmap_path: Path,
    output_dir: Path,
    summary: dict[str, Any],
    tokens: list[Any],
    timing_grid: Any,
) -> dict[str, Any]:
    config = summary["config"]
    expected_hash = str(config["expected_protocol_token_sha256"])
    hash_payload = [
        (int(token.token_id), int(token.ms_in_ref_audio), tuple(str(action) for action in token.actions))
        for token in tokens
    ]
    observed_hash = _sha256_json(hash_payload)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"protocol token hash mismatch: expected={expected_hash}, observed={observed_hash}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = parse_osu_metadata(source_beatmap_path)
    audio_path = Path(config["audio_path"]).expanduser().resolve()
    beatmap_path = output_dir / "oyasumi_bundle_v2_1_sparse_diff4_seed0.osu"
    token_path = output_dir / "protocol_tokens.json"
    manifest_path = output_dir / "render_manifest.json"
    render_dir = output_dir / "reamber"

    timepoints = _canonical_timepoints(tokens)
    audio_filename = os.path.relpath(audio_path, start=beatmap_path.parent)
    beatmap_path.write_text(
        format_osu_export(
            timepoints,
            metadata=OsuExportMetadata(
                audio_filename=audio_filename,
                title=source_metadata.title or audio_path.stem,
                artist=source_metadata.artist or "Unknown Artist",
                creator="Pulsefield",
                version=(
                    f"bundle v2.1 sparse diff {float(config['difficulty']):.2f} "
                    f"seed {config['seed']}"
                ),
                difficulty=float(config["difficulty"]),
                hp_drain_rate=source_metadata.hp_drain_rate or 5.0,
                overall_difficulty=source_metadata.overall_difficulty or 5.0,
            ),
            timing_grid=timing_grid,
        ),
        encoding="utf-8",
    )
    token_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_summary": str(summary_path.resolve()),
                "protocol_token_sha256": observed_hash,
                "tokens": [
                    {
                        "token_id": int(token.token_id),
                        "token_name": str(token.token_name),
                        "ms_in_ref_audio": int(token.ms_in_ref_audio),
                        "actions": [str(action) for action in token.actions],
                    }
                    for token in tokens
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("beatmap_render status=reamber", flush=True)
    # A 30 s crop at the default 5 ms/px is exactly 6,000 px high.  The
    # repository fold helper emits an extra empty column when max_height divides
    # that height exactly, so use a nearby non-divisor for this artifact.
    rendered_paths = render_named_spans(beatmap_path, render_dir, fold_max_height=2_100)
    spans = plan_named_spans(beatmap_path)
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "source_summary": str(summary_path.resolve()),
        "source_beatmap_metadata": str(source_beatmap_path.resolve()),
        "audio_path": str(audio_path),
        "audio_length_ms": int(config["audio_length_ms"]),
        "protocol_token_count": len(tokens),
        "protocol_token_sha256": observed_hash,
        "expected_protocol_token_sha256": expected_hash,
        "hash_matches_profile_run": True,
        "canonical_timepoint_count": len(timepoints),
        "beatmap_path": str(beatmap_path.resolve()),
        "protocol_tokens_path": str(token_path.resolve()),
        "rendered_spans": {
            name: {
                "start_ms": float(spans[name].start_ms),
                "end_ms": float(spans[name].end_ms),
                "path": str(path.resolve()),
            }
            for name, path in rendered_paths.items()
        },
        "timing_segments": (
            []
            if timing_grid is None
            else [
                {
                    "offset_ms": float(segment.offset_ms),
                    "beat_length_ms": float(segment.beat_length_ms),
                    "meter": int(segment.meter),
                }
                for segment in timing_grid.segments
            ]
        ),
        **_object_summary(beatmap_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = _parser().parse_args()
    summary_path = args.summary.expanduser().resolve()
    source_beatmap_path = args.source_beatmap.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        raise RuntimeError(f"source profiling run did not pass: {summary.get('status')!r}")

    tokens, timing_grid = asyncio.run(_replay_bundle(summary["config"]))
    manifest = _write_outputs(
        summary_path=summary_path,
        source_beatmap_path=source_beatmap_path,
        output_dir=output_dir,
        summary=summary,
        tokens=tokens,
        timing_grid=timing_grid,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
