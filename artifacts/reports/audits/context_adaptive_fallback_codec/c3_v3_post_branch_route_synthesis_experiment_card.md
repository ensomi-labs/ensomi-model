# C3/v3 Post-Branch Route Synthesis Experiment Card

## Hypothesis

After C3 hardening, v3 rollout diagnostics, and the v2.1 anti-rigid suppression probes, the current evidence should promote `r0_delta + LZ fallback-substream codec` as the tokenizer-side research route while keeping v3 as a local target-grammar candidate and closing the v2.1 anti-rigid suppression branch.

## Idea Quality

High as a route-control audit. The repository now contains several completed bounded experiments with conflicting surface-level implications: C3 has the strongest compression result, v3 has the cleanest teacher-forcing shape, and v2.1 anti-rigid probes improved rigidity but failed F1/legal continuation gates. A synthesis is needed before launching another large run.

## Related Work / Analogies

- LZ77-style online backreference for C3 fallback-substream repetition.
- Residual and transform coding for exact, mirror, and skeleton-plus-residual fallback spans.
- Local target grammar compression for v3 event-token targets.
- Decode-time constraint and auxiliary-loss calibration for the failed v3/v2.1 rollout mutations.

The representation novelty, if any, is at the policy layer: combining a corpus-global motif layer with an online chart-local fallback side codec. This does not claim novelty for LZ, transform coding, or auxiliary losses.

## Implementation Families

- C3 side-stream family: keep `r0_delta`, add legal C3 fallback references as target-side side-stream or grammar extension.
- v3 local target family: keep the reversible event-token grammar, but do not treat current 500-step rollout quality as replacement-ready.
- v2.1 suppression family: anti-rigid decode/logit suppression, including hard block, soft penalty, and tap-only detector variants.
- Killed/deferred global tokenizer family: broad dictionary expansion, table-heavy bitplane/PPM/CTW fallback codecs, and target-derived C3 input conditioning.

## Minimal Experiment

Run an artifact-only route audit over existing reports. Do not retrain, rerun rollouts, or change mapper defaults.

## Dataset Slice

Existing completed artifacts only:

- Full LE<=3 C3/CASF cache audits.
- Full-cache C3 hardening audit.
- C3 side-stream P0 pipeline artifact audit.
- v3 full-dataset grammar audit.
- v3 32-case fixed-slice rollout diagnostics and training gates.
- v2.1 anti-rigid 6-case and 32-case fixed-slice probes.

## Metric

Decision consistency across:

- C3 charged bits/event delta and reconstruction guard results.
- C3 side-stream integration shape and target-leakage boundary.
- v3 full-dataset reconstruction and 32-case rollout quality gates.
- v2.1 anti-rigid legality, starvation, rigidity, and F1 deltas.

## Positive Signal

- C3 hardening passes full reconstruction and clean/dirty trace guards.
- Exact-only C3 remains strongly positive, proving simple chart-local repetition.
- v3 remains reversible and shorter than v2.1 on the full dataset.
- v3/v2.1 negative rollout probes expose bounded failure mechanisms without corrupting defaults.
- The resulting route names one next target and closes the weak branches.

## Negative Signal

- Any source report contradicts the stated C3 reconstruction or compression result.
- The route requires target-derived C3 input conditioning at inference time.
- The route treats v3 current rollouts as replacement-ready despite failed gates.
- The route keeps v2.1 anti-rigid suppression alive without a new mechanism.

## Kill Criteria

Kill this route synthesis if it cannot identify a single bounded next tokenizer-side target, if it ignores a failed legality/reconstruction gate, or if it promotes a target-derived artifact as production input.

## Expected Runtime

Less than one minute. This is report synthesis only.

## Files Likely To Change

- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_post_branch_route_synthesis_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_post_branch_route_synthesis_result_report.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_v3_post_branch_route_synthesis_summary.json`

## Verification / Failure Modes

Verify by checking that each conclusion is backed by an existing result artifact and that no mapper default, tokenizer default, or training path changes. Main failure mode is overclaiming C3 as a flat tokenizer or v3 as replacement-ready.

## Result Interpretation

If C3 hardening remains the strongest positive result and v3/v2.1 rollout mutations remain negative or incomplete, the next loop should focus on a pipeline-facing C3 side-stream/target artifact rather than another global tokenizer or suppression probe.
