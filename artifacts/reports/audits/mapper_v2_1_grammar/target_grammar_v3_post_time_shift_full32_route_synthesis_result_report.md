# Target Grammar v3 Post-Time-Shift Full32 Route Synthesis Result Report

## Scope

This report reconciles the row-filtered v3 time-shift-distance full32 gate with the current CASF/C3 evidence boundary. It does not retrain, rerun rollout, change tokenizer behavior, or change mapper defaults.

Primary sources:

- `target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_result_report.md`
- `target_grammar_v3_time_shift_distance_row_filtered_repair_full32_gate_summary.json`
- `context_adaptive_fallback_codec/casf_c3_current_audit_target_report.md`
- `context_adaptive_fallback_codec/casf_c3_evidence_boundary_report.md`
- `context_adaptive_fallback_codec/c3_lz_hardening_final_report.md`
- `context_adaptive_fallback_codec/c3_side_stream_pipeline_final_report.md`
- `context_adaptive_fallback_codec/c3_v3_post_branch_route_synthesis_result_report.md`

## Decision

Route: `MUTATE_TIME_SHIFT_DISTANCE_AND_TEST_C3_TARGET_SIDE_STREAM`.

Reason: the repaired time-shift-distance objective trained stably but failed the full32 rollout gate, while C3 remains the strongest tokenizer-side representation result and has already passed stricter reconstruction hardening.

Recommended next action:

- Do not scale the current time-shift-distance objective as-is.
- If continuing v3, first diagnose why the objective worsened rigid/starved rollout behavior.
- For tokenizer-side exploration, keep the audit target on legal C3 fallback-substream target/side-stream integration, not on a larger global vocabulary.

## What Passed

### Time-Shift Row Filtering Passed As A Plumbing Repair

The row-filtered loss no longer hits the earlier invalid-row/NAN path and trains stably.

- baseline steps: `500`
- enabled steps: `500`
- enabled `loss/time_shift_distance`: `0.017952436666573316`
- baseline eval total loss: `2.2627727390410297`
- enabled eval total loss: `2.258821244279302`
- total-loss delta: `-0.003951494761727847`
- baseline eval token loss: `2.176626054738027`
- enabled eval token loss: `2.172651960832534`
- token-loss delta: `-0.003974093905493081`
- all training contract, completion, finite-loss, and valid-token checks passed

Interpretation: the repair is valid as loss plumbing. It does not prove rollout quality.

### C3 Passed The Strongest Tokenizer-Side Representation Gates

CASF v2 proved robust chart-local fallback repetition:

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- C3 test charged bits/event: `4.198613101989439`
- delta: `-0.2231011052018621` bits/event
- same-song-filtered delta: `-0.22353525270533403`
- bootstrap 95% interval: `[-0.258092, -0.186757]`
- fallback-payload reconstruction mismatches: `0`

C3 hardening then strengthened the result:

- selected variant: `a2_skeleton_residual_all_fallback_w256`
- test charged bits/event: `4.0368045441872065`
- delta versus B0: `-0.3849096630040947` bits/event
- same-song-filtered delta: `-0.38592890137466185`
- clean active-hold-trace delta: `-0.3809793375551669`
- dirty active-hold-trace delta: `-0.43255187244188775`
- fallback-payload mismatches: `0`
- full token-stream mismatches: `0`
- full chart mismatches: `0`
- span-boundary mismatches: `0`
- transform-inverse mismatches: `0`
- baseline motif-stream mismatches: `0`

Interpretation: C3's representation/compression result is real under the current charged audit harness and does not depend on dirty active-hold traces.

## What Failed Or Surfaced

### The Time-Shift Full32 Rollout Gate Failed

The enabled model remained legal, but the rollout quality moved the wrong way:

- rollout pairs: `32`
- baseline starved cases: `5`
- enabled starved cases: `8`
- new starved cases: `3`
- baseline rigid cases: `10`
- enabled rigid cases: `13`
- mean dominant-spacing-ratio delta: `+0.06219724054108845`
- mean F1 delta: `-0.030968231114384386`
- mean second-window-share delta: `-0.0343462851188464`
- `no_new_starved_cases`: `false`
- `mean_rigid_not_worse`: `false`

Interpretation: the current expected-shift distance objective at lambda `0.5` and scale `1000ms` should not be promoted, defaulted, or sent to a larger run unchanged. The loss can improve while generated timing structure worsens.

### C3 Still Is Not A Flat Mapper Replacement

The selected C3 hardening variant is side-stream-shaped:

- selected spans: `417208`
- selected fallback literals: `1619653`
- noncontiguous main-stream spans: `199906` (`47.92%`)
- noncontiguous selected fallback literals: `945902` (`58.40%`)
- target cross-chunk spans: `167597` (`40.17%`)
- target cross-chunk selected fallback literals: `826257` (`51.01%`)

Interpretation: C3 works because it references prior fallback-substream history. A naive flat token replacement would hide the structure that makes the win possible.

### Active-Hold Causality Remains Unproved

CASF v2 recorded dirty active-hold traces, and C3 hardening shows the gain survives after clean/dirty filtering. This means dirty traces are not required for the win. It does not prove the gain is caused by active-hold semantics.

Correct claim:

`C3 wins on chart-local fallback-payload repetition under the charged fallback-substream harness.`

Avoided claim:

`C3 wins because it models active-hold causality.`

## What This Proves

- Row filtering repaired the time-shift-distance loss implementation enough for stable training.
- Stable auxiliary-loss training is not enough; rollout gates remain decisive.
- The current time-shift-distance scalar objective failed full32 rollout validation as tested.
- The strongest tokenizer-side positive evidence remains chart-local fallback-substream repetition.
- C3 passed stricter full reconstruction hardening, so its current blocker is integration/legality for the mapper, not basic codec reconstruction.

## What This Does Not Prove

- It does not prove v3 replacement readiness.
- It does not prove that all time-shift or timing-distribution objectives are bad.
- It does not prove C3 improves mapper-generated chart quality.
- It does not prove C3 can be used as inference-time input conditioning.
- It does not prove active-hold causality.

## Current Issue

The branch now has a clean split:

- v3 time-shift distance: useful plumbing repair, failed rollout objective as tested.
- C3: strongest representation/compression result, but side-stream and target-history shaped.
- v2.1/v3 defaults: should remain unchanged until mapper-quality gates pass.

The research issue is no longer whether fallback has structure. It does. The issue is how to expose chart-local fallback references to the mapper without target leakage, unbounded replay, or excessive sequence/state cost.

## Current Audit Target

Audit target:

`Can C3's chart-local fallback reference structure become a legal target-side side stream or grammar extension with deterministic reconstruction, bounded sequence/state cost, and no inference-time target leakage, while v3 scalar timing objectives remain gated by full32 rollout quality?`

Required guardrails:

- preserve full-chart reconstruction mismatches at `0`;
- forbid target-derived C3 sidecar conditioning as production input;
- report sequence-length, batching, cross-chunk reference, noncontiguous-span, distance, length, residual, active/non-active bucket costs;
- keep v2.1 and v3 defaults unchanged until mapper training and rollout evidence passes;
- do not rerun or scale time-shift distance unless the objective, scale, lambda, or diagnostic target changes.

## Result Interpretation

The time-shift branch should return `MUTATE`, not promotion. The C3 branch should stay `TEST_NEXT` only as a legal target-side/side-stream representation family, not as a global tokenizer, active-hold semantic model, or production mapper input.

One-line status:

`r0_delta + LZ fallback-substream codec` remains the best tokenizer-side research route; row-filtered time-shift distance is stable but failed full32 rollout quality as tested.
