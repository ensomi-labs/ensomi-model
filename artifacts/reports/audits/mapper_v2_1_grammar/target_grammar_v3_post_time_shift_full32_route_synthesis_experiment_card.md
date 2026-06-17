# Target Grammar v3 Post-Time-Shift Full32 Route Synthesis Experiment Card

## Hypothesis

After the row-filtered time-shift-distance full32 gate, the current evidence should stop the scalar expected-shift objective as tested and keep the next tokenizer-side audit target on legal C3 fallback-substream integration, not on another global tokenizer or a larger time-shift run.

## Idea Quality

Medium-high as a route-control audit. The time-shift branch repaired a real loss-plumbing bug and trained stably, but the full32 rollout gate regressed. In parallel, CASF/C3 has the strongest tokenizer-side positive evidence but remains side-stream-shaped. A synthesis is needed before launching another branch.

## Related Work / Analogies

- Expected-position and distance-style auxiliary objectives for sequence calibration.
- LZ77-style chart-local backreference for C3 fallback-substream repetition.
- Transform and residual coding for exact, mirror, and skeleton-residual fallback spans.
- Teacher-forced target grammar compression for v3 event-token targets.

The possible novelty remains at the representation-policy layer: combining a corpus-global motif stream with an online chart-local fallback side codec. This card does not claim novelty for LZ, transform coding, or scalar auxiliary losses.

## Implementation Families

- Stop or mutate time-shift-distance objective family: keep the row-filter repair, but do not scale the current lambda/objective as-is.
- C3 side-stream family: preserve `r0_delta`, add legal chart-local fallback references as a target-side side stream or grammar extension.
- v3 local target family: keep the reversible event-token grammar, but require new rollout-quality evidence before replacement.
- Killed/deferred families: broad global vocabulary growth, table-heavy fallback context models, C3 target-derived input conditioning, and simple scalar/decode rules that already failed gates.

## Minimal Experiment

Run an artifact-only synthesis over existing committed summaries. Do not retrain, rerun rollouts, change tokenizer behavior, or change mapper defaults.

## Minimal Code Change

None. This is report synthesis only.

## Dataset Slice

Existing artifacts only:

- 500-step baseline and time-shift-distance-enabled full32 v3 training reports.
- 32 matched baseline/enabled real-audio rollout pairs from the row-filtered time-shift full32 gate.
- CASF v2 full-cache context-adaptive fallback codec report.
- C3 LZ hardening full-cache report.
- C3 side-stream and role/target feasibility reports.
- Existing v3 full-dataset and post-branch route synthesis artifacts.

## Metric

Decision consistency across:

- full32 rollout gates for the time-shift-distance objective;
- token/total loss deltas and finite auxiliary-loss checks;
- C3 charged bits/event deltas;
- C3 full reconstruction guards;
- C3 side-stream legality and target-leakage boundaries.

## Positive Signal

- The synthesis does not promote the failed time-shift objective.
- It preserves the row-filter repair as a valid plumbing fix.
- It keeps C3's positive compression/reconstruction evidence separate from unproved mapper-quality claims.
- It names one current audit target and one stopped branch.

## Negative Signal

- The route treats small token-loss improvements as enough to override failed rollout gates.
- The route promotes target-derived C3 sidecar inputs as production inputs.
- The route claims active-hold causality from C3 evidence.
- The route launches another full32 time-shift run without a changed objective, scale, or diagnostic reason.

## Kill Criteria

Kill this synthesis if any source artifact contradicts the reported metrics, if the report cannot distinguish representation compression from mapper quality, or if the next audit target requires inference-time target leakage.

## Expected Runtime

Less than one minute. This is report generation and verification only.

## Files Likely To Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_time_shift_full32_route_synthesis_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_time_shift_full32_route_synthesis_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_post_time_shift_full32_route_synthesis_summary.json`

## Verification / Failure Modes

Verify that all numbers come from committed artifacts and that no code or default settings changed. Main failure modes are overclaiming C3 as a flat tokenizer, overclaiming v3 as replacement-ready, or treating the current time-shift-distance objective as solved because training loss moved in the right direction.

## Result Interpretation

If the source artifacts are consistent, the branch route is:

`MUTATE_TIME_SHIFT_DISTANCE_AND_TEST_C3_TARGET_SIDE_STREAM`

That means the current time-shift-distance objective is stopped as-is, while the C3 family remains the strongest tokenizer-side research route under a legal target-side/side-stream audit.
