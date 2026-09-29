# What became a realtime system, and what remained a quality problem

The [H/R/R1 separation](full-audio-and-r1-transfer.md) made it possible to
generate from BOS, plan head-bearing times, generate physical releases online
and publish complete rows with settled coverage. The original request for an
opening of at least thirty rows did not remain a requirement to copy a seed
chart. The system learned/generated its own beginning.

## Timing and publication evidence

Products `e04b349`, `fdc6c13`, `1c95f69` and `9b99800` establish the planned
prototype and audio-file streaming entrypoint. The
[playback budget study](/Users/l/projects/ensomi-model/docs/research/audio_playback_system.md)
reports resident-model readiness of .553–1.052 s over nine native cases,
including fresh audio decode, Mel and thirty rows plus eight seconds of settled
coverage. A 585.495-s source-H workload takes 1.899 s. Imports/model loading are
separate; source-H stress tests establish service behavior, not autonomous H
quality. Eleven traces had no virtual deadline miss under the declared buffer
and stall/scaled-cost replays. These are Apple M5 observations, not client or
OS worst-case guarantees.

The important invariant is that the head plan, settled chart clock and player
clock are different. A latest row timestamp does not prove the following span
empty. An LN head can be published before its tail, as the human permitted;
unpublished lookahead must not rewrite committed facts or dependencies.
The exported `.osu` is not the first usable output: incremental rows and
coverage arrive before completion. The repository does not contain a deployed
client transport.

This evidence deferred speculative models and graceful-degradation machinery:
no measured bottleneck justified them in those workloads. It did not defer the
quality problem. The [later benchmark branch](/Users/l/.codex/worktrees/stream-generation-benchmark/ensomi-model/)
at `4ec631ef71d1ca71e36e5c383d4997efffdebb05` has distinct startup/session/coverage
contracts and is a separate integration target. Later two-second qualifiers
must not be represented as a pass of that benchmark.

## Shared profiles and unpublished continuation screening

Shared arrangement profiles (`10ddaa8`) gave H, R and rows a common condition
for H rate, chord width and LN fraction. Multi-arrangement audio motivated this
joint condition, but those descriptors were not yet numerical difficulty or
semantic style controls. Poor profile calibration persisted on fresh audio.

The first local candidate correction could consider a legal release that the
actual R/R1 rollout never chose. `452abc6` therefore added bounded private
continuation screening: sample R/rows on fixed H, inspect a window plus halo,
retry privately, then publish only an accepted prefix. Finite attempts can fail
incompletely; they cannot invent missing proposal support. It is neither exact
whole-song rejection sampling nor an unchanged neural law.

| Historical comparison | Established result | Remaining failure |
| --- | --- | --- |
| Fifteen paired outputs on the profile checkpoint | All complete; same H; strict sub-20-ms HH 3→0 and RH≤20-ms 12→0. Seven no-retry outputs remain identical. Sampling time rises 3.64%. | Two LN-fraction preservation guards fail. Resampled history/RNG changes propagate into later regions. A 1-ms LN passes both gap screens. |
| Eight additional audio contexts × three requests | All 24 screened charts complete; paired H unchanged; two HH and eight RH witnesses disappear. Sampling rises 3.2%; fresh-input consumer readiness .623–1.625 s. | Large profile errors remain, including weak high-LN realization and missing inspected role/exchange relationships. |
| Longest fresh-audio CLI repetition | Ready in 2.921 s from process start; rows and streamed output match API receipts | An integration reproduction, not an independent quality sample or a player trial. |

The [screen study](/Users/l/projects/ensomi-model/docs/research/unpublished_continuation_screen.md)
preserves exact proposal/state ownership and composition failures. The
[state/RNG follow-up](/Users/l/projects/ensomi-model/docs/research/continuation_state_dependence.md)
matters because one changed suffix did not prove a uniform persistent-collapse
mechanism. The [fresh-audio study](/Users/l/projects/ensomi-model/docs/research/fresh_audio_system_evaluation.md)
at `e28435f` records 209 read time pages over 75 new scopes, plus reused
identical scopes. Rendered files alone would not establish that inspection.
No listening or human playtest verdict is inferred.

The provisional RH threshold was not simply a human-approved universal rule.
The later answer to the HH/RH question restated attack semantics; it did not
select an offered release policy. The following human message required ranked
2–6★ grounding and player-response reasoning. See [answer and context, private,
local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0393).

## Why the next work moved to controls and ownership

Successful playback and removal of narrow collision witnesses left the system
unable to choose or maintain the requested arrangement. Head-profile routing,
mixed H/materializer crosses and count factorization explored that mismatch:
[profile routing](/Users/l/projects/ensomi-model/docs/research/profile_density_routing_evaluation.md),
[head drift](/Users/l/projects/ensomi-model/docs/research/head_factor_drift.md),
[component composition](/Users/l/projects/ensomi-model/docs/research/head_materializer_composition.md)
and [future continuation mass](/Users/l/projects/ensomi-model/docs/research/count_continuation_mass.md).
These records distinguish a currently legal row from a likely legal future;
they are component studies, not a selected final system.

The [scoped-control trajectory](scoped-controls-and-ownership.md) explains the
ensuing typed-resource detour and the human correction that restored complete
row decisions to R1. The later [player-response work](player-response-and-memory.md)
returns to the planner with actual-time sustained-load evidence rather than
assuming a local gap check represents the frontier.

Historical evidence owners are `artifacts/joint-audio/20260924-playback-budget-v1/run-v2`,
`20260925-unpublished-continuation-v1` and `20260925-fresh-audio-system-v1`.
The latter two result digests are respectively
`3fff511d78ef4d0fd3400f33853d2e79219a69dbd034e67bee9d1263b5cb2ca9`
and `9f3c4580adaa6239c44e04bb5d95a0ed921b8fd958b1eee26ccd8cedfa91e862`.
This recovery read the committed reports; it did not rerun playback or inspect
all their images again.
