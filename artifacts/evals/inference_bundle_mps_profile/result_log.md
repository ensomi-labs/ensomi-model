# Result Log: inference bundle profiling on Apple Silicon

## Status

`TEST` completed successfully. All runtime, completion, and fixed-output hash guards passed.

Recommended next-loop action: `MUTATE` into one bounded incremental-mapper host/device synchronization experiment. Do not optimize the control model first on the evidence from this fixture.

## Machine and software

- MacBook Air, Apple M5 (10 CPU cores), 24 GB unified memory
- macOS 26.5.1, arm64
- Python 3.10.20
- PyTorch 2.11.0
- Device: MPS for control and mapper; CPU for BeatThis
- `torch.profiler.supported_activities()`: CPU only on this build

## Fixture and configuration

- Audio: `dataset/0/1086533/audio.mp3`
- Audio duration: 47.151 seconds
- Packed-mel cache: hit before the first measured request
- Mapper: default `v2_1_sparse` checkpoint
- Control: default learned `ControlDemoGlobalEncoder` checkpoint
- Greedy decode, seed 0, 512-token/window limit
- Incremental mapper decode enabled
- Online control policy; no full-song control precompute
- Token send pacing disabled (`token_send_interval_s=0`) so the result measures compute rather than intentional delivery sleep
- Concurrency: 1
- One warmup session, three unprofiled measured sessions, one separate diagnostic/profiler session

## Commands

```sh
uv run --extra mps --group dev pytest -q tests/evals/test_inference_bundle_profiler.py

uv run --extra mps --group dev pytest -q \
  tests/inference/test_model_bundles.py \
  tests/inference/test_model_runtime.py \
  tests/inference/test_session_runtime.py \
  tests/inference/test_mapper_v2_1_rollout.py

uv run --extra mps python -m pulsefield_model.evals.inference_bundle_profiler \
  --audio-path dataset/0/1086533/audio.mp3 \
  --device mps --repeat 3 --warmup 1 \
  --expected-raw-token-sha256 4cf134ebd0ed770699948255d568e3ae040a8e4c722ad4243470f9f93d238e8a \
  --expected-protocol-token-sha256 219e9f6518bbd57a6af196c0405e0891596dd7881ae2099a03f4dbc1de6ad531 \
  --output-dir artifacts/evals/inference_bundle_mps_profile
```

## Verification

- New benchmark helper tests: 6 passed
- Existing bundle/runtime/session/rollout guards: 38 passed
- Expected windows: 6 per session; observed: 6 for every measured session
- Every window completed
- No dead ends
- No max-token overflow
- Final terminal time reached the audio end
- Every measured run emitted 2,088 raw mapper tokens and 472 protocol tokens
- Raw and protocol hashes were identical across repeats and matched the fixed fail-fast baseline hashes

## Primary result

Warm-bundle request compute includes `prepare_audio` plus the complete online token stream, but excludes cold model mount and the deliberate 20 ms/token delivery pacing.

| Metric | Mean | P50 | P95 | Worst observed |
| --- | ---: | ---: | ---: | ---: |
| Request compute time | 17.829 s | 17.492 s | 18.707 s | 18.842 s |
| Request compute RTF | 0.378 | 0.371 | 0.397 | 0.400 |
| Audio seconds / compute second | 2.649x | 2.696x | — | 2.502x slowest |
| `prepare_audio` | 1.074 s | 1.022 s | 1.176 s | 1.193 s |
| Complete stream compute | 16.755 s | 16.484 s | 17.533 s | 17.649 s |
| First protocol token | 2.227 s | 2.107 s | 2.465 s | 2.504 s |
| Per-window latency | 2.792 s | 2.906 s | 3.449 s | 3.458 s |

The system is faster than real time for this fixed single-session fixture. The slowest measured request still processed audio at 2.50x real time, and every 8-second generation window finished within 3.46 seconds.

## Throughput

- Raw mapper tokens: mean 124.8 tokens/s of stream compute
- Protocol hitobject tokens: mean 28.2 tokens/s of stream compute
- Fixed output volume: 2,088 raw tokens and 472 protocol tokens/session

Tokens/s is output-dependent, so the primary capacity metric remains audio-seconds/compute-second with fixed output hashes.

## Control versus mapper

The diagnostic run synchronizes nested stages and is intentionally separate from headline throughput.

- Control `prepare_control_batch` synchronized latency: 75.9-98.2 ms/window; P50 84.3 ms, P95 95.4 ms.
- Online control throughput: P50 96.2 equivalent audio-seconds/compute-second; minimum observed 81.5.
- Mapper context setup after subtracting control: about 6.4-7.3 ms/window on non-profiled diagnostic windows.
- Mapper decode: 2.08-3.45 s/window on non-profiled diagnostic windows.

The trace-selected second window is explicitly excluded from latency decomposition: `torch.profiler` inflated its wall time to 14.74 seconds. It remains useful for CPU operator structure, not performance totals.

Result: autoregressive mapper decode dominates the current single-session compute path. Control is measurable but is not the first optimization target for this fixture.

## Memory

- MPS driver allocation after cold mount: 168 MB
- MPS driver allocation after full stream: 2.220 GB
- MPS current tensor allocation after stream: approximately 131 MB
- Process maximum RSS during the complete run, including the diagnostic trace: 3.48 GB
- Recommended MPS working-set limit reported by PyTorch: 19.07 GB

The RSS value includes profiler/trace overhead and is not a clean baseline-only peak.

## Profiler interpretation

This PyTorch/MPS build exposes only `ProfilerActivity.CPU`. The 145 MB Chrome trace therefore shows host-side PyTorch operators, `record_function` scopes, dispatch, and synchronization behavior; it does not provide Metal kernel duration.

The committed artifact stores this trace as
`bundle_diagnostic_cpu_trace.json.gz` (9.8 MB). Decompress it back to
`bundle_diagnostic_cpu_trace.json` before opening it in a Chrome trace viewer.

Synchronized `time.perf_counter()` wall time is the source of truth for MPS stage and request latency. The selected profiler window is marked `profiler_active=true`, `latency_breakdown_eligible=false`, and has no reported mapper decode latency.

## Result interpretation

Positive signal:

- RTF is comfortably below 1.0.
- P95 and maximum window latency remain below the 8-second generation lead at concurrency 1.
- Output and completion guards are stable and match the fixed baseline.
- Control and mapper costs are separable without changing production inference code.

Limitations:

- This is the public bundle compute path, including registry, lease, session cache, online control, mapper, and protocol translation. It does not include protobuf serialization, WebSocket framing, socket drain, network, or the production 20 ms/token pacing.
- One short song and concurrency 1 do not establish capacity under concurrent sessions.
- The packed-mel path was warm-cache; cold audio preprocessing remains unmeasured.
- The fanless machine and small repeat count make tail percentiles descriptive, not population estimates.

## Next-loop action

`MUTATE`: define one new Experiment Card around mapper incremental-decode host/device synchronization. The closest code-level candidates are repeated prefix/state tensor construction and per-token `.item()` synchronization. Compare only one bounded change against this fixed-output baseline before considering any control-global-encoder rewrite.

This result is engineering characterization, not a novelty claim.

## Beatmap visualization follow-up

The fixed bundle configuration was replayed once through the same public
`RoutedInferenceBackend` path, with the protocol-token hash checked before any
artifact was accepted. The replay emitted 472 protocol tokens and reproduced
the profiling-run digest exactly:

`219e9f6518bbd57a6af196c0405e0891596dd7881ae2099a03f4dbc1de6ad531`

The repository's canonical osu!mania exporter and Reamber span renderer then
produced:

- `beatmap/oyasumi_bundle_v2_1_sparse_diff4_seed0.osu`
- `beatmap/protocol_tokens.json`
- `beatmap/render_manifest.json`
- five standard Reamber span images under `beatmap/reamber/`

Export/render verification: 8 focused tests passed. Manual image inspection
confirmed that the first-30-second and last-30-second views are non-empty and
together cover the generated 0-47.1-second chart.

The visualization exposes a qualitative failure mode that the throughput
guards did not detect:

- all 471 adjacent protocol-token intervals are exactly 100 ms;
- all 1,615 exported hitobjects are TAPs and there are no holds;
- chord sizes are 2K at 40 timepoints, 3K at 193, and 4K at 239;
- only five distinct protocol event tokens occur;
- lane hitobject counts are `[409, 367, 367, 472]`, so lane 4 is active at
  every timepoint;
- average density is 34.25 hitobjects/second.

This does not invalidate the fixed-output performance measurement or its hash
guards. It does limit interpretation: the reported throughput characterizes
the current deterministic, highly repetitive token stream, not a
product-quality or distribution-representative beatmap. Treat output-quality
sanity as a prerequisite guard in the next performance experiment.
