# CASF/C3 Current Audit Target Report

## Scope

This synthesis records the stronger CASF/C3 interpretation from the latest review and reconciles it with the artifacts now present in the repository. It does not introduce a new experiment or change mapper defaults.

Source artifacts:

- `casf_v2_final_report.md`
- `casf_v2_report.json`
- `c3_lz_hardening_final_report.md`
- `c3_lz_hardening_report.json`
- `c3_side_stream_pipeline_final_report.md`
- `c3_full_pipeline_role_audit_result_report.md`
- `c3_target_role_feasibility_result_report.md`
- `casf_c3_evidence_boundary_report.md`
- `casf_c3_to_v3_route_update_report.md`

## Current Thesis

The remaining compressible structure is chart-local and online-referential, not primarily corpus-global and vocabulary-based.

The correct name for the proven family is:

`r0_delta + LZ fallback-substream codec`

or:

`LZ-backed fallback payload side codec`

It should not be described as a new global tokenizer, and it should not be described as an active-hold semantic model.

The positive thesis is narrower and stronger:

A corpus-trained motif tokenizer plus an online chart-local backreference codec outperforms either static tokenization or table-heavy fallback modeling alone under the current charged fallback-substream harness.

## What CASF v2 Proved

CASF v2 proved that the fallback payload stream contains robust chart-local repetition.

Key metrics:

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- C3 test charged bits/event: `4.198613101989439`
- Delta versus B0: `-0.2231011052018621` bits/event
- Same-song-filtered delta: `-0.22353525270533403` bits/event
- Bootstrap mean delta: `-0.222825`
- Bootstrap 95% interval: `[-0.258092, -0.186757]`
- Fallback-payload reconstruction mismatches: `0`

Interpretation:

- The win is too large and too stable to treat as duplicate-song leakage or a few mapset outliers.
- The result is not evidence for a larger global motif dictionary.
- The result is evidence that fallback residuals repeat within the same chart and can be encoded by prior same-chart fallback records.

## What Later Hardening Added

The latest review correctly notes that CASF v2 alone did not prove full chart reconstruction. That gap is now closed by the later C3 hardening pass.

Selected hardening variant:

- `a2_skeleton_residual_all_fallback_w256`

Hardening metrics:

- Test charged bits/event: `4.0368045441872065`
- Delta versus B0: `-0.3849096630040947` bits/event
- Same-song-filtered delta: `-0.38592890137466185`
- Clean active-hold-trace delta: `-0.3809793375551669`
- Dirty active-hold-trace delta: `-0.43255187244188775`
- Bootstrap 95% interval: `[-0.4235772949889358, -0.34584630062559635]`

Hard reconstruction guards all passed with zero mismatches:

- fallback-payload reconstruction
- full token-stream reconstruction
- full chart reconstruction
- span-boundary reconstruction
- transform inverse
- baseline motif-stream reconstruction

Interpretation:

- CASF v2 passed the research gate.
- C3 hardening passed the stricter representation/reconstruction gate.
- The earlier concern that C3 might only reconstruct fallback payloads while hiding chart-level mismatch is no longer the blocker in the audit harness.

## Why C3 Wins

C3 wins for three concrete reasons.

First, it has no large trained table cost. C1/C2 had payload signal, but the context/table accounting erased or overwhelmed it. C3 pays distance, length, transform, and residual costs only when a local reference is worthwhile.

Second, it avoids explaining the residual whenever a prior fallback payload can be referenced. It asks whether the difficult fallback structure appeared earlier in the same chart, not why the structure exists.

Third, osu!mania charts naturally contain repeated local material: phrase repetition, mirrored hand patterns, repeated LN/chord alternation, repeated difficulty patterns, chorus structure, and copy-modified sections. These may be too local to become corpus-global motifs but are well matched to an LZ-style online codec.

## What Failed Or Should Stay Deprioritized

### C1 Bitplane Codec

C1 found payload signal but failed after table cost.

- C1a frozen test delta: `+1.017676` bits/event
- C1b online test delta: `+1.015245` bits/event
- C1 table/model cost: `2,683,934` bits

Decision: do not promote C1 as implemented. Revisit only as a small residual model inside a C3-backed path if future residuals need it.

### C2 PPM/CTW

C2 failed from context/table explosion.

- C2a PPM test delta: `+227.675006` bits/event
- C2b CTW-like test delta: `+225.297546` bits/event
- C2 table/model cost: `380,332,350` bits

Decision: keep killed/deferred unless the symbol and context spaces are radically narrowed.

### C4 Dynamic Patch With C1 Payload Codec

C4 improved payloads but failed globally.

- C4 test delta: `+1.309988` bits/event
- C4 fallback payload delta: `-0.840208` bits/fallback-event

Decision: dynamic patching is not killed in principle, but the C4=C1-patch implementation inherits C1's cost problem. If revisited, the patch codec should be C3-style LZ, not C1.

### S1 Explicit Selector

S1 failed because it selected among table-heavy payload codecs.

- S1 test delta: `+225.943023` bits/event
- S1 payload delta: `-1.003025` bits/fallback-event

Decision: the failed S1 does not disprove selector logic. A future selector should choose among raw fallback, exact LZ, mirror LZ, skeleton-residual LZ, and residual literal rather than C1/C2-heavy candidates.

## What Is Not Proved

### Active-Hold Causality

CASF v2 had dirty active-hold traces:

- Invalid active-hold transitions: `1,789`
- Segment-end active holds: `1,788`

C3 hardening shows the gain survives on clean-trace maps, so dirty active-hold traces are not required for the win. That still does not prove that active-hold state is the causal mechanism.

Correct claim:

`C3 wins on chart-local fallback-payload repetition under the charged fallback-substream harness.`

Avoided claim:

`C3 wins because it models active-hold causality.`

### End-To-End Mapper Benefit

The audit proves representation/compression and reconstruction value. It does not prove:

- mapper training loss improves;
- generated chart quality improves;
- the model can emit or use C3 references during autoregressive decode;
- side-stream semantics are learnable from audio/control inputs;
- sequence length and batching cost are acceptable for default training.

### Global Tokenizer Replacement

The selected C3 structure is side-stream-shaped, not flat-token-shaped.

For the hardened selected variant:

- selected spans: `417,208`
- selected fallback literals: `1,619,653`
- noncontiguous main-stream spans: `199,906` (`47.92%`)
- noncontiguous selected fallback literals: `945,902` (`58.40%`)
- target cross-chunk spans: `167,597` (`40.17%`)
- target cross-chunk selected fallback literals: `826,257` (`51.01%`)

Interpretation: C3 should not be promoted as a simple global tokenizer replacement. Its value depends on fallback-substream history and often crosses ordinary chunk/window boundaries.

## Pipeline Issue Surfaced

The issue is no longer whether fallback has structure. It does.

The issue is how to expose chart-local fallback backreferences to the mapper without leaking target information or requiring an impractical replay path.

The P0 side-stream pipeline artifact proved lossless materialization:

- side-stream reference errors: `0`
- side-stream payload mismatches: `0`
- full token-stream mismatches: `0`
- group signature mismatches: `0`
- mapper-timepoint-compatible mismatches: `0`
- side-stream tokens: `3,203,904`
- fallback placeholders: `2,786,696`
- combined main+side tokens: `8,856,468`
- combined/main baseline token ratio: `1.566805`

This is a strong artifact-level pass, but not a default sequence-format pass. P0 keeps main-stream placeholders and emits explicit side-stream tokens, so it is larger than the baseline encoded stream.

## Current Audit Target

The audit target should now be:

`Can C3's chart-local fallback reference structure be turned into a legal mapper-side target or auxiliary target without target leakage, while preserving deterministic reconstruction and acceptable sequence/state cost?`

This target separates five checkable claims:

1. Codec legality: already passed in C3 hardening.
2. Artifact round-trip: already passed in P0 side-stream tokenization.
3. Inference legality: failed for current pooled input conditioning because the sidecar is target-derived.
4. Learnability/usefulness: not yet proved; later auxiliary/kind diagnostics surfaced weak RAW-side learnability.
5. Default replacement readiness: not proved; C3 remains disabled by default and should not replace v2.1 or v3 target grammar yet.

## Current Decision

Decision: `TEST_NEXT`, but not as a global tokenizer and not as target-derived mapper input conditioning.

Promote cautiously:

- `r0_delta + LZ fallback-substream codec` as a research-backed representation family.
- C3 side-stream artifacts as legal offline target-side material.

Kill/defer:

- broad global vocabulary expansion as the next answer;
- C1/C2 table-heavy fallback codecs;
- C4=C1 dynamic patching;
- current C3 pooled input-conditioning promotion;
- active-hold causality claims;
- default production enablement.

Next bounded direction:

- Keep v3 as the current simpler teacher-forcing target-grammar route.
- Preserve C3 as a codec-side and target-side representation family.
- Revisit C3 through auxiliary target, emitted target grammar, or a two-stage predicted C3 plan only after a bounded card defines the legal decode path and metrics.
