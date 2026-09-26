# Local controlled-generation service

The local demo serves the controlled H/R skeleton and complete-row R1 runtime
on loopback HTTP. It is separate from the legacy mapper/protobuf service.
The corresponding macOS client entry is **Live Demo → Listen & Play** in
Ensomi; its operation manual lives in the client repository at
`docs/live_generation_demo.md`.

## Launch and model identity

```sh
./scripts/demo-service.sh
curl http://127.0.0.1:8766/health
```

The launcher resolves the primary Git checkout to find its existing `.venv` and
local research weights, including when invoked from a managed worktree. Override
`UV_PROJECT_ENVIRONMENT` or `ENSOMI_DEMO_CHECKPOINT` for different local paths.
It uses `uv run --no-sync --extra mps` and the current checkout's Python source.
Install dependencies first if no usable environment exists. The generator uses
one CPU intra-op thread; the accelerator extra supplies the model dependencies.

The packaged `configs/inference/controlled_demo.yaml` pins the outcome checkpoint
reported by source commit `a99519ccee60925dce10a4200f88a6c049d37964`:

- `20260926-common-prefix-outcomes-r1-v1/actor-128/step-128.pt`
- SHA-256 `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3`

These are local research assets. The model remains experimental; service
availability does not establish reliable style control or musical playability.
The service verifies the checkpoint hash before accepting traffic. Health is
available only after the model loads. FFmpeg must be installed.

Typed settings are projected in `ensomi_model.serving.hydra`; runtime and HTTP
modules do not import Hydra. Overrides include `port`, `cache_dir`, `cache_mib`,
`max_sessions`, `session_idle_seconds` and `request_budget_seconds`.
The host is restricted to `127.0.0.1` or `localhost`. Default port is 8766.
Changing the model requires an explicit checkpoint file and matching hash.

## HTTP contract

JSON responses use protocol `ensomi-demo/v1`. POST requests require
`Content-Type: application/json`, a JSON object and at most 32 KiB of body data.
Errors return a non-200 status and an `error` string. Audio paths are local
absolute paths accessible to the service; the protocol does not upload audio.

| Request | Behavior |
| --- | --- |
| `GET /health` | Returns ready state, protocol, checkpoint hash, style vocabulary, 1000-ms publication unit and active session count. |
| `POST /sessions` | Accepts a unique client-owned `session_id`, `audio_path`, optional `controls` and optional integer `seed` (default 17). Returns identity, decoded duration and an in-memory feature-cache hit flag. |
| `POST /sessions/{id}/window` | Accepts exactly `after_sequence` and `through_ms`. Returns the next window with sequence, complete rows, confirmed coverage and EOS. |
| `DELETE /sessions/{id}` | Cancels and releases a session; repeating deletion is harmless. |

`controls` accepts nullable `difficulty` in [1,8], nullable `ln_fraction` in
[0,1], and a `style` dictionary with trained vocabulary names and values in
[-1,1]. Omission means unspecified. The demo UI supplies one optional prominent
style and an optional difficulty target. Controls remain fixed for that session;
this adapter does not expose in-place control revision.

The first window request may generate a prefix from BOS through a late-join
target. Later requests advance at most 1000 ms, even if a larger target is
requested. Generation retains the real audio duration; window boundaries do not
close holds. Every row contains an integer millisecond timestamp and four
actions: 0 unchanged, 1 TAP, 2 LN start, 3 LN end. Returned rows occur strictly
after previous coverage and at or before the new coverage. Empty rows arrays
can advance confirmed silence. EOS occurs only at true audio termination with
all holds closed.

Sequences start at one. Repeating the immediately preceding request cursor
returns the same retained response, allowing recovery from a lost response
without consuming randomness twice. Other stale/out-of-order cursors fail.
Only the most recent response is retained; this is not durable resumption.

## Scheduling, caches and lifetime

One compute lock serializes model operations. HTTP health and cancellation remain
independent of that lock. Cancellation is visible to the research generator's
step guard, including while a request is generating a long prefix. Compute-budget
accounting resets at each window request so playback idle time does not consume
the generation budget. A failed window invalidates its session.

A session owns its replay, LN feedback, timing queues, history caches and RNGs.
The client requests one-second windows when its buffer drops to roughly six
seconds, refilling toward eight seconds. The server does not substitute requested
read time for generated coverage. Client pagination must also hold back a
watermark when events at that timestamp remain undelivered.

The disk Mel cache includes audio-content and frontend identities and stores the
exact decoded sample count, preserving the true millisecond audio duration.
Audio encoding is cached only within one frozen model runtime. An LRU bounds
retained Mel/encoding storage to `cache_mib` (256 MiB by default); active sessions
can retain encodings after LRU eviction. The cache limit is therefore not a
process-memory ceiling. There are at most two active sessions by default, with
120-second idle expiration, pruned on subsequent health or generation requests.
Stop, EOS and replacement should release sessions
explicitly. Restarting the service discards generation state.

The client must retain an anchored audio clock, choose an entry with no incoming
open hold, and require sufficient confirmed future coverage. Transport stalls
must not be interpreted as empty chart intervals. This service supplies the
producer contract; physical capture, identification and synchronization belong
to the client.

## Verification

`tests/serving/test_demo_runtime.py` covers actual model-backed window generation,
cache reuse, sequenced retry, cancellation, malformed controls and an HTTP
round trip. Constructor parity checks compare cached and uncached full-audio
encodings. These tests use a small synthetic model and audio; the selected
research checkpoint requires a separate live integration run.
