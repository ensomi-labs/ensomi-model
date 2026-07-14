# Experiment Card: inference bundle profiling on Apple Silicon

## Hypothesis

The production `v2_1_sparse` mapper bundle can generate a fixed real 47.151-second song on MPS faster than real time, and synchronized stage measurements can separate the cost of control preparation from mapper window setup and autoregressive decode without changing production inference code.

## Root objective

Characterize the current control-plus-mapper inference bundle on this Mac, using the real public bundle path and production checkpoints, so later optimization work has a reproducible latency, throughput, memory, and correctness baseline.

## Goal decomposition

1. Verify the real lazy-mount, session prepare, online window generation, protocol translation, and reset path on MPS.
2. Separate cold bundle load, warm audio preparation, control work, mapper window setup, and mapper decode costs.
3. Record output and lifecycle guards so a faster failure, truncated decode, or changed token stream cannot look like an optimization.

## Candidate variants

1. **Isolated model microbench**: reuse `mapper_v21_decoder_profiler.py` for control/mapper calls. Fast, but misses registry, lease, session cache, online control, and protocol translation.
2. **Instrumented production bundle (selected)**: run `RoutedInferenceBackend` through its public `prepare_audio` and `iter_hitobject_tokens` API, injecting benchmark-only subclasses for synchronized stage timing and trace scopes.
3. **Full WebSocket client/server**: includes framing and socket drain, but the default 20 ms token pacing confounds model throughput and the repository has no sample client.
4. **MPS Instruments capture**: useful for Metal kernel analysis, but it is a second diagnostic surface; `torch.profiler` on this PyTorch build exposes CPU activity only.

## Local verification matrix

| Variant | Smallest local check | Pass evidence | Decision |
| --- | --- | --- | --- |
| Isolated model | Existing decoder-profiler unit test | Model operators execute | Reject as primary: not a bundle test |
| Instrumented bundle | One 47.151 s session, pacing disabled | Six windows complete via public API; stage rows and tokens are emitted | Select |
| WebSocket | Existing endpoint tests | Protocol state machine works | Defer: pacing/network obscure model baseline |
| Instruments | MPS profiler API availability | Signpost capture can start | Defer: optional device-side follow-up |

## Selected variant

Instrumented production bundle, concurrency 1, default `v2_1_sparse` checkpoint, default learned control checkpoint, BeatThis on CPU, control and mapper on MPS, greedy decoding, incremental cache enabled, and token pacing disabled.

## Selection pressure

The selected variant is the smallest option that preserves the requested bundle lifecycle and online control/mapper interaction. It beats the isolated probe on representativeness, the WebSocket variant on metric validity, and Instruments on implementation/runtime cost. The guard is successful deterministic completion through the public bundle API; the runtime budget is 15 minutes.

## Minimal change

Add one eval/benchmark module that:

- injects a measured `SessionRuntime` and measured v2.1 stream into the existing bundle;
- uses `torch.mps.synchronize()` at measurement boundaries;
- runs unprofiled baseline sessions separately from one profiler diagnostic session;
- writes a structured JSON summary and CPU Chrome trace;
- does not modify production inference behavior.

## Files likely to change

- `src/pulsefield_model/evals/inference_bundle_profiler.py`
- `tests/evals/test_inference_bundle_profiler.py`
- `artifacts/evals/inference_bundle_mps_profile/experiment_card.md`
- `artifacts/evals/inference_bundle_mps_profile/result_log.md`
- generated JSON/trace files in the same artifact directory

Read-only context:

- `src/pulsefield_model/inference/model_runtime.py`
- `src/pulsefield_model/inference/session_runtime.py`
- `src/pulsefield_model/inference/model_bundles/`
- `src/pulsefield_model/inference/stream_with_cache.py`
- `src/pulsefield_model/evals/mapper_v21_decoder_profiler.py`

## Dataset slice

- Audio: `dataset/0/1086533/audio.mp3`
- Duration: 47.151 seconds
- Expected online windows: 6 x 8-second decoder windows
- Existing packed-mel cache: record hit/miss status in the result
- Difficulty: 4.0
- Decode: greedy, seed 0, max 512 tokens/window

## Baseline / comparator

The primary baseline is the current checkout running without `torch.profiler` and with synchronization only at response/window measurement boundaries. The profiler run is diagnostic and must not supply headline throughput numbers. Historical Riria measurements are context only because the decoder and runtime have since changed.

## Primary metric

`warm_request_compute_rtf = (prepare_audio + complete online token stream wall seconds) / 47.151`.

Also report its inverse, `audio_seconds_per_compute_second`.

## Secondary metric

- cold bundle mount latency
- warm `prepare_audio` latency
- reference-to-first-protocol-token latency
- end-to-end stream latency
- per-window P50/P95/max latency
- synchronized control batch latency and control audio-seconds/s
- mapper window setup and decode latency
- decoded tokens/s and emitted protocol tokens/s
- completed/dead-end/max-token window counts
- MPS allocated and driver memory by phase
- CPU profiler scope/operator totals for the diagnostic run

## Verify command or evaluation procedure

```sh
uv run --extra mps --group dev pytest -q tests/evals/test_inference_bundle_profiler.py
uv run --extra mps python -m pulsefield_model.evals.inference_bundle_profiler \
  --audio-path dataset/0/1086533/audio.mp3 \
  --device mps --repeat 3 --warmup 1 \
  --expected-raw-token-sha256 4cf134ebd0ed770699948255d568e3ae040a8e4c722ad4243470f9f93d238e8a \
  --expected-protocol-token-sha256 219e9f6518bbd57a6af196c0405e0891596dd7881ae2099a03f4dbc1de6ad531 \
  --output-dir artifacts/evals/inference_bundle_mps_profile
```

## Guard check

```sh
uv run --extra mps --group dev pytest -q \
  tests/inference/test_model_bundles.py \
  tests/inference/test_model_runtime.py \
  tests/inference/test_session_runtime.py \
  tests/inference/test_mapper_v2_1_rollout.py
```

Runtime guards:

- resolved device is exactly `mps`;
- all six windows are observed;
- every generated window is complete;
- no window is a dead end or exceeds `max_tokens`;
- terminal time advances through the song;
- output token count and SHA-256 are recorded per repeat and match the fail-fast baseline digests;
- bundle/session returns to a valid state after reset and shutdown.

## Qualitative check

Inspect the Chrome trace for the expected nesting of bundle window, mapper-window preparation, control batch, and v2.1 incremental decoder operators. Treat trace times as CPU-side activity only on MPS.

## Positive signal

- All guards pass on MPS.
- At least three unprofiled measured sessions complete.
- Warm request compute RTF is below 1.0.
- Stage measurements identify control and mapper costs without profiler data being used as the throughput source.

## Negative signal

- RTF is 1.0 or higher, or tail window latency prevents keeping an 8-second lead.
- Control or mapper stage dominates enough that the current online policy has little headroom.
- Repeat outputs or completion guards differ under fixed greedy settings.

## Kill criteria

Stop this experiment rather than broadening it if MPS is unavailable, either production checkpoint cannot load, one fail-fast session cannot finish within 5 minutes, memory pressure destabilizes the host, or the public bundle path cannot complete without modifying the research question. Return `MUTATE` to a shorter fixture or isolated boundary if that happens.

## Expected failure modes

- first request hides lazy model loading inside `prepare_audio`;
- mel cache hit/miss changes prepare latency;
- MPS asynchronous execution under-reports unsynchronized timers;
- profiler in the asyncio parent thread misses `asyncio.to_thread` work;
- per-stage synchronization perturbs command-buffer overlap;
- output-dependent autoregressive token counts distort tokens/s;
- fanless thermal throttling increases later repeats;
- default token pacing overwhelms compute time if not disabled.

## Expected runtime / runtime budget

- Focused tests: under 2 minutes
- Smoke plus one diagnostic trace: under 5 minutes
- One warmup plus three unprofiled sessions: under 8 minutes
- Hard budget: 15 minutes; fail-fast stop at 5 minutes for a single session

## Confounders

Power state, background load, thermal throttling, warm filesystem cache, packed-mel cache state, BeatThis CPU execution, MPS allocator state, profiler overhead, synchronized diagnostic boundaries, and output-dependent decode length. Record machine/software/config metadata with the result.

## Result interpretation plan

- **Positive**: keep this runner as the baseline and use the dominant synchronized stage to define the next bounded optimization experiment.
- **Negative**: preserve the measurements; select the dominant stage rather than proposing a broad rewrite.
- **Ambiguous**: if variance or output changes dominate, increase repetitions or pin an exact output fixture before changing code.

## Result log template

```text
Status:
Machine/software:
Commands:
Fixture/config:
Cold mount:
Warm request RTF and inverse throughput:
Prepare-audio:
Control:
Mapper setup/decode:
Window percentiles:
Memory:
Guards/checksum:
Profiler trace caveat:
Positive/negative/ambiguous interpretation:
Next-loop action:
```

## Next-loop action

`TEST` this selected variant. After results, choose exactly one of: keep the baseline, `MUTATE` into a control-global-encoder experiment, or `MUTATE` into an incremental mapper host/device synchronization experiment.

## Closest analogies and novelty layer

Closest analogies are standard serving latency decomposition, PyTorch synchronized accelerator microbenchmarks, and the repository's existing mapper decoder profiler. This experiment makes no novelty claim. Its contribution is engineering observability at the production bundle boundary, not a new representation or model method.
