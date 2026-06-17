# C3/v3 Post-Branch Route Synthesis Result Report

## Scope

This is an artifact-only synthesis over the completed C3, v3, and v2.1 anti-rigid reports. It does not retrain, rerun rollouts, change source code, or change mapper defaults.

Source families:

- `casf_v2_final_report.md`
- `c3_lz_hardening_final_report.md`
- `c3_side_stream_pipeline_final_report.md`
- `c3_full_pipeline_role_audit_result_report.md`
- `target_grammar_v3_full_dataset_audit_result_report.md`
- `target_grammar_v3_500step_fixed_slice_wide_audit_result_report.md`
- `target_grammar_v3_500step_failure_cluster_diagnostic_result_report.md`
- `target_grammar_v3_event_distribution_calibration_result_report.md`
- `target_grammar_v3_event_budget_training_gate_result_report.md`
- `target_grammar_v3_continuation_jump_training_gate_result_report.md`
- `target_grammar_v3_selective_event_gate_smoke_result_report.md`
- recent v2.1 anti-rigid hard-block, soft-penalty, and tap-only detector commits

## Decision

Decision: `TEST_NEXT_C3_SIDE_STREAM`.

The strongest current route is:

1. Keep `r0_delta` as the corpus-global motif layer.
2. Promote C3 only as `r0_delta + LZ fallback-substream codec`, not as a generic global tokenizer and not as active-hold semantics.
3. Build the next bounded artifact around a legal pipeline-facing C3 side stream or target-side grammar extension.
4. Keep v3 as a reversible local target-grammar candidate, but do not treat current v3 rollouts as replacement-ready.
5. Close the v2.1 anti-rigid suppression branch unless a new mechanism addresses legality, continuation, and F1 together.

## What Passed

### C3 Compression And Reconstruction

C3 hardening is the strongest positive result in this branch.

- Baseline B0 `r0_delta` test charged bits/event: `4.421714207191301`
- Selected C3 variant: `a2_skeleton_residual_all_fallback_w256`
- Selected C3 test charged bits/event: `4.0368045441872065`
- Delta versus B0: `-0.3849096630040947` bits/event
- Same-song-filtered delta: `-0.38592890137466185`
- Bootstrap 95% interval: `[-0.4235772949889358, -0.34584630062559635]`
- Clean active-hold-trace delta: `-0.3809793375551669`
- Dirty active-hold-trace delta: `-0.43255187244188775`

All hard reconstruction guards passed with zero mismatches:

- fallback-payload reconstruction
- full token-stream reconstruction
- full chart reconstruction
- span-boundary reconstruction
- transform inverse
- baseline motif-stream reconstruction

This proves a legal, charged chart-local fallback repetition codec under the audit harness.

### C3 Decomposition

The result does not depend on the most complex variant.

- Exact-only all-fallback C3 delta: `-0.2785859452434707`
- Mirror-only all-fallback C3 delta: approximately `-0.290085`
- Active fixed-window C3 delta: `-0.24256829527744017`
- Bounded 128-record active window delta: `-0.23783549837040585`
- Skeleton-residual all-fallback C3 delta: `-0.3849096630040947`
- Non-active all-fallback C3 delta: `-0.092664`

Exact-only winning strongly is the critical scientific result: the fallback payload stream contains repeated same-chart material even before skeleton residual abstraction.

### C3 Side-Stream Artifact Legality

The P0 side-stream artifact passed lossless materialization:

- side-stream reference errors: `0`
- side-stream payload mismatches: `0`
- full token-stream mismatches: `0`
- group signature mismatches: `0`
- mapper-timepoint-compatible mismatches: `0`

This proves the representation can be materialized and decoded as an artifact.

### v3 Grammar Audit

v3 passed the representation audit but not the full rollout replacement gate.

- Full-dataset windows: `174515`
- Reconstruction mismatches: `0`
- Token reduction: `20.54%`
- Eval total-bit reduction: `5.43%`
- Cross-window LN windows: `80316`

This keeps v3 alive as a clean teacher-forcing target grammar. It does not by itself prove generated mapper quality.

## What Surfaced

### C3 Is Not A Flat Replacement Tokenizer

The selected C3 variant is side-stream-shaped:

- selected spans: `417208`
- selected fallback literals: `1619653`
- noncontiguous main-stream spans: `199906` (`47.92%`)
- noncontiguous selected fallback literals: `945902` (`58.40%`)
- target cross-chunk spans: `167597` (`40.17%`)
- target cross-chunk selected fallback literals: `826257` (`51.01%`)

This is the main implementation issue. C3 works by referencing prior fallback-substream records, often across ordinary main-stream or chunk boundaries. A naive flat vocabulary replacement would erase the structure that makes it work.

### C3 Input Conditioning Remains Illegal As Implemented

The current C3 mapper-window sidecar is target-derived. It is legal as an offline target or auxiliary label, but not as a production inference input. The pooled input-conditioning route should stay killed unless a generation-time C3 source is introduced.

### v3 Rollout Quality Is Not Replacement-Ready

The 32-case fixed-slice v3 rollout was legal but failed quality gates:

- Mean F1: `0.640`
- Starved cases: `11`
- Rigid cases: `7`
- Dominant failure: second-window starvation coupled to a `160ms` grid in mid/high difficulty charts.

Later v3 mutations did not fix this:

- Density weight `0.20`: legality regressed, starvation worsened `11 -> 17`, median second-window share collapsed `49.1% -> 4.5%`.
- Event-budget `0.05`: `4` max-token cases, starvation worsened `11 -> 18`, mean F1 fell by `0.126`.
- Continuation-jump `0.05`: `3` dead ends, starvation `19`, rigid cases `24`.
- Selective event gate smoke: helped low-bias cases but regressed a pass-like control, so the eval-only gate was killed.
- Global event bonus viability: killed because most starved cases required too large a bias for a safe global decode rule.

Interpretation: v3 remains a good grammar candidate, but current scalar training/decode fixes are not enough for replacement.

### v2.1 Anti-Rigid Suppression Is Closed For Now

The recent v2.1 anti-rigid branch found a useful diagnostic but not a viable replacement mutation.

- Six-case hard-block gate: rigidity improved in all cases and mean F1 rose, so the idea deserved a fixed-slice test.
- 32-case hard-block comparison: rigidity improved `30/32`, but mean F1 dropped by `0.036` and a new hard-block dead end appeared.
- Soft penalties `0.5/1.0/2.0/4.0`: failed to remove the legality/starvation issue; case `05_usao_knight_rider_kuo_kyoka_dnm_s_normal` stayed invariantly bad for suppression.
- Tap-only detector: fixed case `05`, but reverted case `04_oomori_seiko_justadice_tv_size_remu_normal` to the baseline dead-end/starvation path.

Interpretation: simple anti-rigid suppression cannot distinguish when LN suppression should be skipped versus when LN-pattern intervention is necessary.

## What Is Proved

- The remaining tokenizer-side compression space is chart-local fallback-substream repetition, not primarily a larger global vocabulary.
- C3 is reconstructive at full chart level under stricter guards.
- C3 gain survives clean active-hold-trace filtering, so dirty active-hold traces are not required for the result.
- Exact local repetition alone is strong enough to justify C3 side-stream hardening.
- v3 is a reversible, shorter local target grammar, but not yet a trained replacement.
- Current v2.1 anti-rigid suppression improves rigidity by trading away F1, legality, or continuation in important cases.

## What Is Not Proved

- C3 has not proved active-hold causality.
- C3 has not proved mapper-generated chart quality.
- C3 has not proved legal production input conditioning.
- v3 has not proved full-pipeline replacement readiness.
- v2.1 anti-rigid suppression has not proved a viable default decode policy.
- None of these results prove algorithmic novelty beyond the representation-policy layer.

## Current Issue

The central issue has moved from "does fallback contain structure?" to "how can chart-local fallback references be exposed to the mapper without target leakage or complex replay?"

C3 has the strongest compression and reconstruction evidence, but it is side-stream and target-history shaped. v3 has a smaller compression win, but it is local and teacher-forcing friendly. v2.1 is still the safer default mapper grammar, but the anti-rigid branch did not produce a replacement-quality mutation.

## Next Audit Target

Target:

`Can the hardened C3 fallback-substream codec become a legal pipeline-facing target-side side stream or grammar extension with deterministic reconstruction, bounded sequence/state cost, and no inference-time target leakage?`

Required gates for the next C3 card:

- preserve full-chart reconstruction mismatches at `0`;
- expose main stream, fallback placeholders, and side-stream references deterministically;
- measure sequence-length and batching impact versus v2.1/v3;
- forbid target-derived C3 conditioning as production input;
- report per-window statistics on cross-chunk references, noncontiguous spans, distance, length, residual cost, and active/non-active buckets;
- keep v2.1 defaults until mapper training and rollout evidence exists.

## Result Interpretation

This synthesis does not approve a full replacement. It promotes C3 to the next tokenizer-side integration audit because the representation result is now hard enough, while v3 and v2.1 suppression probes have exposed mapper-side failure modes that need a different mechanism.

The clean one-line status is:

`r0_delta + LZ fallback-substream codec` is the current best tokenizer-side research route; v3 remains a local target grammar candidate; v2.1 anti-rigid suppression is closed until a more selective mechanism is defined.
