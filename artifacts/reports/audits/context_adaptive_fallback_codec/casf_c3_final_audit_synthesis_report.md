# CASF/C3 Final Audit Synthesis Report

## Scope

This report synthesizes the CASF v2, C3 hardening, and C3 side-stream pipeline artifacts. It does not introduce a new experiment and does not change mapper defaults.

Source artifacts:

- `casf_v2_final_report.md`
- `casf_v2_report.json`
- `c3_lz_hardening_final_report.md`
- `c3_lz_hardening_report.json`
- `c3_side_stream_pipeline_final_report.md`
- `c3_side_stream_pipeline_report.json`
- `casf_c3_evidence_boundary_report.md`
- `casf_c3_current_audit_target_report.md`
- `casf_c3_to_v3_route_update_report.md`

## Result

Decision: `TEST_NEXT`, but only for the representation family and side-stream artifact path.

The correct name for the current positive result is:

`r0_delta + LZ fallback-substream codec`

or:

`LZ-backed fallback payload side codec`

The result should not be described as a new global tokenizer, an active-hold semantic model, or a production/default mapper format.

## What Passed

### CASF v2 Research Gate

CASF v2 showed that the remaining compressible structure is chart-local and online-referential, not primarily corpus-global vocabulary structure.

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- C3 test charged bits/event: `4.198613101989439`
- Delta versus B0: `-0.2231011052018621` bits/event
- Same-song-filtered delta: `-0.22353525270533403` bits/event
- Bootstrap mean delta: `-0.22282466691710426`
- Bootstrap 95% interval: `[-0.2580919949410291, -0.18675671602064908]`
- Fallback-payload reconstruction mismatches: `0`

Interpretation: the original C3 result is a real charged compression signal on fallback payloads. The same-song filter and bootstrap interval make duplicate-song leakage or a few outlier mapsets unlikely explanations.

### C3 Hardening Gate

C3 hardening closed the stricter reconstruction concern. The selected hardening variant was `a2_skeleton_residual_all_fallback_w256`.

- Selected test charged bits/event: `4.0368045441872065`
- Delta versus B0: `-0.3849096630040947` bits/event
- Same-song-filtered delta: `-0.38592890137466185`
- Clean active-hold-trace delta: `-0.3809793375551669`
- Dirty active-hold-trace delta: `-0.43255187244188775`
- Bootstrap 95% interval: `[-0.4235772949889358, -0.34584630062559635]`
- Research promotion pass: `true`
- Engineering promotion pass: `true`

All hard reconstruction guards passed with zero mismatches:

- fallback-payload reconstruction
- full token-stream reconstruction
- full chart reconstruction
- span-boundary reconstruction
- transform inverse
- baseline motif-stream reconstruction

Interpretation: the audit no longer only proves that fallback payloads can be reconstructed. It proves that the unchanged baseline motif stream plus the C3 fallback side stream reconstructs the full audited chart/token stream with zero mismatches.

### Local-Repetition Ablations

The hardening ablations support the local-repetition thesis.

- Exact-only all-fallback delta: `-0.2785859452434707`
- Mirror-only all-fallback delta: `-0.2900851303531917`
- Active fixed-window C3 delta: `-0.24256829527744017`
- Active 128-record-window delta: `-0.23783549837040585`
- Skeleton-residual all-fallback delta: `-0.3849096630040947`
- Non-active all-fallback delta: `-0.09266425164250869`

Exact-only already wins, so repeated fallback payloads exist within charts even before mirror or skeleton-residual abstraction. Mirror and skeleton-residual improve the gain, so the stream also contains repeated structure under simple transforms.

### P0 Side-Stream Artifact

The side-stream pipeline pass proved lossless materialization as an artifact.

- side-stream reference errors: `0`
- side-stream payload mismatches: `0`
- full token-stream mismatches: `0`
- group signature mismatches: `0`
- mapper-timepoint-compatible mismatches: `0`
- side-stream tokens: `3,203,904`
- fallback placeholders: `2,786,696`
- combined main+side tokens: `8,856,468`
- combined/main baseline token ratio: `1.566805`

Interpretation: C3 can be represented as a deterministic side stream. This is an artifact-level pass, not a compact learned sequence-format pass.

## What Failed

### C1 Bitplane Codec

C1 had payload signal but failed after table/model cost.

- C1a frozen test delta: `+1.0176756781547605` bits/event
- C1b online test delta: `+1.0152451725879983` bits/event
- C1 table/model cost: `2,683,934` bits

Decision: do not promote C1 as implemented. It is only worth revisiting later as a tiny residual model inside a C3-backed path.

### C2 PPM/CTW

C2 failed from context/table explosion.

- C2a PPM test delta: `+227.67500573862475` bits/event
- C2b CTW-like test delta: `+225.29754649980495` bits/event
- C2 table/model cost: `380,332,350` bits

Decision: keep this formulation killed or deferred unless the symbol and context spaces are radically narrowed.

### C4 Dynamic Patch With C1 Payload Codec

C4 improved fallback payloads but failed globally.

- C4 test delta: `+1.309988256498733` bits/event
- C4 fallback payload delta: `-0.840208` bits/fallback-event

Decision: dynamic patching is not killed in principle, but this implementation inherits C1's table-cost failure. Any revisit should use a C3-style LZ payload codec.

### S1 Explicit Selector

S1 failed because it selected among table-heavy payload codecs.

- S1 test delta: `+225.94302312901377` bits/event
- S1 payload delta: `-1.003025` bits/fallback-event

Decision: this does not disprove selector logic. A future selector should choose among raw fallback, exact LZ, mirror LZ, skeleton-residual LZ, and residual literal, not C1/C2-heavy candidates.

## What Is Not Proved

### Active-Hold Causality

CASF v2 had dirty active-hold traces:

- invalid active-hold transitions: `1,789`
- segment-end active holds: `1,788`

C3 hardening shows the gain survives on clean-trace maps, so dirty active-hold traces are not required for the win. That still does not prove active-hold state is the causal mechanism.

Correct claim: C3 wins on chart-local fallback-payload repetition under the charged fallback-substream harness.

Avoided claim: C3 wins because it models active-hold causality.

### End-To-End Mapper Benefit

The audit proves representation compression and reconstruction. It does not prove:

- mapper training loss improves;
- generated chart quality improves;
- the model can emit or use C3 references during autoregressive decode;
- side-stream semantics are learnable from audio/control inputs;
- sequence length and batching cost are acceptable for default training.

### Global Tokenizer Replacement

C3 is side-stream-shaped, not flat-token-shaped.

For the selected hardening variant:

- selected spans: `417,208`
- selected fallback literals: `1,619,653`
- noncontiguous main-stream spans: `199,906` (`47.92%`)
- noncontiguous selected fallback literals: `945,902` (`58.40%`)
- target cross-chunk spans: `167,597` (`40.17%`)
- target cross-chunk selected fallback literals: `826,257` (`51.01%`)

Interpretation: C3 should not be promoted as a simple global tokenizer replacement. Its gain depends on fallback-substream history and often crosses ordinary chunk/window boundaries.

## Main Issue Surfaced

The main issue is no longer whether fallback has structure. It does.

The remaining issue is how to expose chart-local fallback backreferences to the mapper without target leakage, illegal decode order, impractical replay state, or unacceptable sequence/state cost.

Current status by gate:

- Research gate: passed.
- Representation/reconstruction gate: passed in the audit harness.
- Pipeline artifact gate: passed for lossless offline side-stream materialization.
- Mapper-side learnability/quality gate: not passed.
- Production/default gate: not passed.

## Current Audit Target

The next useful audit target is:

`Can C3's chart-local fallback reference structure be turned into a legal mapper-side target or auxiliary target without target leakage, while preserving deterministic reconstruction and acceptable sequence/state cost?`

This target should stay smaller than a full mapper rewrite. It should test one legal exposure path at a time: auxiliary target, emitted target grammar, or two-stage predicted plan.

## Recommendation

Promote cautiously:

- `r0_delta + LZ fallback-substream codec` as a research-backed representation family.
- C3 side-stream artifacts as legal offline target-side material.

Kill or defer:

- broad global vocabulary expansion as the next answer;
- C1/C2 table-heavy fallback codecs;
- C4=C1 dynamic patching;
- current C3 pooled input-conditioning promotion;
- active-hold causality claims;
- default production enablement.

Next bounded direction:

- keep v3 as the simpler current teacher-forcing target-grammar route;
- preserve C3 as a codec-side and target-side representation family;
- revisit C3 through auxiliary target, emitted target grammar, or a two-stage predicted C3 plan only after a bounded Experiment Card defines the legal decode path and metrics.
