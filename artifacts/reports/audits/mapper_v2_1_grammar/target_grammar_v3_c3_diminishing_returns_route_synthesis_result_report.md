# Target Grammar v3 / C3 Diminishing-Returns Route Synthesis Result Report

## Scope

This artifact-only synthesis reconciles the current C3 mapper-side evidence with the v3 timing-failure evidence after the post-time-shift full32 gate. It does not retrain, rerun rollout, change tokenizer behavior, or change mapper defaults.

Primary source artifacts:

- `context_adaptive_fallback_codec/casf_c3_current_audit_target_report.md`
- `context_adaptive_fallback_codec/c3_lz_hardening_final_report.md`
- `context_adaptive_fallback_codec/c3_full_pipeline_role_audit_result_report.md`
- `context_adaptive_fallback_codec/c3_auxiliary_target_learnability_result_report.md`
- `context_adaptive_fallback_codec/c3_kind_heads_1000step_result_report.md`
- `context_adaptive_fallback_codec/c3_raw_structure_audit_result_report.md`
- `context_adaptive_fallback_codec/c3_kind_logit_calibration_result_report.md`
- `context_adaptive_fallback_codec/c3_raw_factor_probe_result_report.md`
- `context_adaptive_fallback_codec/c3_raw_context_bucket_probe_result_report.md`
- `context_adaptive_fallback_codec/c3_ordered_raw_field_grammar_probe_result_report.md`
- `target_grammar_v3_generated_state_continuation_diagnostic_result_report.md`
- `target_grammar_shared_timing_residue_structure_audit_result_report.md`
- `target_grammar_v3_post_time_shift_full32_route_synthesis_result_report.md`

## Decision

Route: `MUTATE_C3_MAPPER_PATH_TEST_V3_TIMING_LOGIT_AUDIT`.

Reason: C3 remains a strong codec-side representation result, but the tested mapper-side C3 integration families now show diminishing marginal returns. The v3 target grammar remains reversible and local, while its dominant failure is generated-state timing/ranking collapse. The next bounded experiment should audit v3 teacher-forced timing logits before adding another C3 target grammar or scalar loss.

Recommended next card:

`target_grammar_v3_teacher_forced_time_shift_logit_audit`

## What Passed

### C3 Compression And Reconstruction Still Pass

C3 should not be killed as a representation family.

- CASF v2 B0 `r0_delta` test charged bits/event: `4.421714207191301`
- CASF v2 C3 test charged bits/event: `4.198613101989439`
- CASF v2 delta: `-0.2231011052018621`
- C3 hardening selected variant: `a2_skeleton_residual_all_fallback_w256`
- C3 hardening test charged bits/event: `4.0368045441872065`
- C3 hardening delta: `-0.3849096630040947`
- clean active-hold-trace delta: `-0.3809793375551669`
- fallback-payload mismatches: `0`
- full token-stream mismatches: `0`
- full chart mismatches: `0`
- span-boundary mismatches: `0`
- transform-inverse mismatches: `0`
- baseline motif-stream mismatches: `0`

Interpretation: C3 has proven chart-local fallback-substream repetition and legal reconstruction under the audit harness.

### C3 Auxiliary Target Initially Passed A Learnability Smoke

The legal target-side auxiliary path produced a real but limited positive signal.

- P13 auxiliary eval loss: `0.684974 -> 0.546028`
- relative auxiliary-loss drop: `20.285%`
- final eval token-loss delta versus baseline: `+0.000188`
- token-loss regression gate: not triggered

Interpretation: C3 labels are not impossible to learn as target-side structure.

### v3 Grammar Still Passes Representation Requirements

The v3 event-token target remains the cleanest local replacement candidate.

- full-dataset scored windows: `174515`
- full-dataset reconstruction mismatches: `0`
- full-audit token reduction: `20.54%`
- eval token reduction: `19.96%`
- eval total-bit reduction: `5.43%`
- no C3-style cross-window reference or future target lookup required

Interpretation: v3 remains alive as a teacher-forcing target grammar even though trained rollout quality is not ready.

## What Failed Or Surfaced

### C3 Mapper-Side Path Shows Diminishing Returns

The later C3 mapper-side probes no longer justify another training run without a new formulation.

- P19 reduced kind heads: `MUTATE`; model recall@20 `0.074406`, reduced unigram `0.089710`, only `82.94%` of unigram.
- P19 RAW blocker: RAW recall@20 `0.084309` versus unigram `0.141686`; `REF` tied unigram and `RES` beat unigram, so RAW became the limiting subspace.
- P20 RAW structure: `MUTATE`; model RAW recall@20 trails unigram by `-0.057377`, with rank-near rate `0.324356`.
- P21 kind-logit calibration: `MUTATE_TO_RAW_SPLIT`; selected held-out hits worsened `68 -> 64`; unconstrained calibration helped only by overpromoting RAW and damaging non-RAW recovery.
- P22 RAW factor probe: `MUTATE`; joint factor coverage@5 `0.553864`, but only `field_3` beat unigram at K=3.
- P23 context buckets: `MUTATE`; only `1` supported positive context bucket, too weak for new heads.
- Ordered RAW field grammar: `MUTATE`; reconstruction passed, but ordered/exact sequence ratio was `3.185742` and RAW ordered bits/token `30.189757` versus exact RAW `9.475498`.

Interpretation: C3 is still a good codec, but the current mapper-side paths are no longer the cheapest route to full pipeline progress.

### v3 Timing Collapse Is Better Grounded Than Another C3 Mutation

The v3 failure surface points to timing/ranking calibration.

- Generated-state diagnostic route: `TEST_DECODE_TIMING_CALIBRATION`.
- Shared timing-residue audit route: `TEST_TIMING_LOSS_OR_EMBEDDING_CALIBRATION`.
- Target event-interval effective vocab: `22.690931`.
- Target interval `160/200ms` share: `9.65%`.
- Target rigid-window share at `>=0.95`: `1.58%`.
- Generated v3 mean dominant-spacing ratio in the compared rollout: `0.993197`.

Interpretation: teacher-forcing targets preserve richer timing residue than generated rollouts. This makes timing-logit calibration a smaller and more direct next test than another C3 RAW split.

### The Current Time-Shift Objective Is Not The Fix As Tested

The row-filtered time-shift-distance objective repaired loss plumbing, but full32 rollout quality regressed.

- baseline starved cases: `5`
- enabled starved cases: `8`
- new starved cases: `3`
- baseline rigid cases: `10`
- enabled rigid cases: `13`
- mean dominant-spacing-ratio delta: `+0.06219724054108845`
- mean F1 delta: `-0.030968231114384386`
- mean second-window-share delta: `-0.0343462851188464`

Interpretation: the next timing audit should not rerun the same expected-shift distance objective. It should first inspect teacher-forced time-shift logits and calibration failure modes.

## What This Proves

- C3's representation/compression result is strong and reconstructive.
- C3 target-derived input conditioning is not legal for production inference.
- C3 auxiliary/RAW mapper-side experiments have hit diminishing marginal returns under the current formulations.
- v3 remains the cleaner local grammar candidate because it is reversible, shorter, and online-decodable.
- Current v3 generated failures are dominated by timing/ranking collapse, not by basic grammar irreversibility.

## What This Does Not Prove

- It does not prove C3 should be permanently abandoned.
- It does not prove v3 replacement readiness.
- It does not prove all timing objectives are bad.
- It does not prove the next timing-logit audit will improve rollouts.
- It does not prove novelty beyond representation-policy and engineering integration choices.

## Current Issue

The active issue is no longer "does C3 find structure?" It does.

The active issue is that C3's proven structure is side-stream and target-history shaped, while current mapper-side ways to expose it are either illegal as inference input or weak as learnable auxiliary targets.

For full-pipeline progress, the smaller next question is:

`Does the existing v3 checkpoint already rank diverse teacher-forced time-shift targets well, or is the model's time-shift posterior itself collapsed toward a small rigid-grid prior?`

That question distinguishes:

- target/logit calibration failure;
- decode/exposure failure;
- timing embedding weakness;
- need for grammar-level timing changes.

## Next Audit Target

Target:

`Audit teacher-forced v3 time-shift logits on the fixed-slice checkpoint before adding new training losses or C3 grammar integration.`

Required guardrails:

- no training;
- no rollout rerun;
- no tokenizer or default changes;
- use the existing 500-step v3 checkpoint and fixed-slice eval/data artifacts;
- compare model time-shift posterior concentration with target timing residue;
- report target-shift recall/rank, predicted-shift concentration, and 160/200ms over-selection;
- route to timing embedding/loss mutation only if teacher-forced logits are already collapsed.

## Result Interpretation

Pause C3 mapper integration after P23/P24-style RAW probes. Keep C3 as a codec-side representation family and future target-side research path. Move the immediate full-pipeline effort to a v3 teacher-forced timing-logit audit, because v3's grammar requirements are satisfied but generated timing behavior is not.

One-line status:

`C3 is proven as a codec; v3 is the current pipeline candidate; the next bottleneck is v3 timing-logit calibration, not another C3 RAW-side training run.`
