# CASF/C3 Evidence Boundary Report

## Scope

This report synthesizes the existing CASF v2 and C3 LZ hardening artifacts. It does not introduce a new experiment. The purpose is to state what the current evidence proves, what it does not prove, and what issues remain before full pipeline promotion.

Source artifacts:

- `casf_v2_final_report.md`
- `casf_v2_report.json`
- `c3_lz_hardening_final_report.md`
- `c3_lz_hardening_report.json`
- `c3_lz_hardening_comparison.csv`
- `c3_lz_hardening_diagnostics.csv`

## Core Finding

The remaining compressible structure is chart-local and online-referential, not primarily corpus-global and vocabulary-based.

The strongest current representation is:

1. Keep the corpus-global `r0_delta` motif stream unchanged.
2. Add a chart-local LZ-style fallback-substream side codec.
3. Reconstruct selected fallback literals by referencing prior fallback records from the same chart, with exact, mirror, or skeleton-residual transforms.

This should be described as an `r0_delta + LZ fallback-substream codec`, not as a new global tokenizer and not as an active-hold semantic tokenizer.

## What Passed

### CASF v2 Passed The Research Gate

CASF v2 selected `c3_lz_active_span_backreference_w256` as the only legal charged variant that passed the original gates.

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- C3 test charged bits/event: `4.198613`
- Delta: `-0.223101` bits/event
- Same-song-filtered delta: `-0.223535` bits/event
- Bootstrap mean delta: `-0.222825`
- Bootstrap 95% interval: `[-0.258092, -0.186757]`
- Reconstruction guard: pass
- Fallback-payload reconstruction mismatches: `0`

Interpretation: the C3 gain is not explained by duplicate-song leakage or a small set of outlier mapsets. It is a robust charged compression gain on the fallback payload stream.

### C3 Hardening Passed The Stricter Promotion Audit

The C3 hardening pass selected `a2_skeleton_residual_all_fallback_w256`.

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- Selected test charged bits/event: `4.0368045441872065`
- Delta: `-0.3849096630040947` bits/event
- Same-song-filtered delta: `-0.38592890137466185`
- Clean active-hold-trace delta: `-0.3809793375551669`
- Dirty active-hold-trace delta: `-0.43255187244188775`
- Bootstrap 95% interval: `[-0.4235772949889358, -0.34584630062559635]`
- Research promotion pass: `true`
- Engineering promotion pass: `true`

Hard reconstruction guards all passed:

- Fallback-payload mismatch count: `0`
- Full token-stream mismatch count: `0`
- Full chart mismatch count: `0`
- Span-boundary mismatch count: `0`
- Transform-inverse mismatch count: `0`
- Baseline motif-stream mismatch count: `0`

Interpretation: the stricter decode gate is satisfied in the current harness. The result is no longer only "fallback payload seems right"; it is "unchanged baseline motif stream plus C3 fallback side-stream reconstructs the full chart/token stream with zero mismatches under the audit model."

### Ablations Support The Local-Repetition Thesis

Charged test deltas versus B0:

- `a1_exact_only_all_fallback_w256`: `-0.2785859452434707`
- `a3_mirror_only_all_fallback_w256`: `-0.290085`
- `c3_active_all_fixed_w256`: `-0.24256829527744017`
- `a6_active_all_fixed_w128`: `-0.23783549837040585`
- `a7_active_all_elias_gamma_w256`: `-0.248259`
- `a7_active_all_power_bucket_w256`: `-0.235131`
- `a2_skeleton_residual_all_fallback_w256`: `-0.3849096630040947`

Exact-only winning is the key signal: repeated fallback payloads already exist within charts. Mirror and skeleton-residual improve this further, which suggests repeated structure with simple transforms, not just byte-for-byte repetition.

## What Failed Or Should Stay Killed

### C1 Bitplane Codec

C1 had payload signal but failed after table cost.

- C1a frozen test delta: `+1.017676` bits/event
- C1b online test delta: `+1.015245` bits/event
- C1 table/model cost: `2,683,934` bits

Interpretation: the current bitplane family is not promotable. It may only be worth revisiting later as a very small residual model inside a C3-backed path.

### C2 PPM/CTW Context Model

C2 failed due to context/table explosion.

- C2a PPM test delta: `+227.675006` bits/event
- C2b CTW-like test delta: `+225.297546` bits/event
- C2 table/model cost: `380,332,350` bits

Interpretation: this formulation should remain killed. The symbol/context space is too fragmented for the current fallback stream.

### C4 Dynamic Patch With C1 Payload Codec

C4 improved fallback payloads but failed globally.

- C4 test delta: `+1.309988` bits/event
- C4 fallback payload delta: `-0.840208` bits/fallback-event

Interpretation: dynamic patching is not killed in principle, but the C4=C1-patch implementation inherits the C1 table-cost failure. If revisited, it should use a C3-style LZ payload codec, not C1.

### S1 Explicit Selector

S1 failed because its candidate family included table-heavy C1/C2 payload codecs.

- S1 test delta: `+225.943023` bits/event
- S1 payload delta: `-1.003025` bits/fallback-event

Interpretation: selector logic is not disproved. The failed selector should not be used as evidence against a future selector over raw literal, exact LZ, mirror LZ, skeleton-residual LZ, and residual literal.

## What Is Not Proved

### Active-Hold Causality Is Not Proved

The CASF v2 artifact still records dirty active-hold traces:

- Invalid active-hold transitions: `1,789`
- Segment-end active holds: `1,788`

The hardening pass reduces this risk because the clean-trace subset still wins by `-0.3809793375551669` bits/event. That proves C3 does not depend on dirty active-hold traces. It does not prove that active-hold state itself is the causal reason for the gain.

Correct claim: C3 wins on chart-local fallback-payload repetition under the current charged fallback-substream harness.

Avoided claim: C3 wins because it has learned active-hold causality.

### End-To-End Mapper Benefit Is Not Proved

The audit proves a representation/compression win, not a mapper training win.

Not yet proven:

- mapper loss improves with this representation,
- generation quality improves,
- conditioning or target-side auxiliary C3 helps full rollout,
- sequence length and batching cost remain acceptable at full scale,
- the model can use side-stream semantics during autoregressive decoding.

### Global Tokenizer Replacement Is Not Proved

The selected C3 structure is side-stream-shaped. It is not a clean flat token replacement.

For `a2_skeleton_residual_all_fallback_w256`:

- selected spans: `417,208`
- selected fallback literals: `1,619,653`
- noncontiguous main-stream spans: `199,906` spans, `47.92%`
- noncontiguous selected fallback literals: `945,902`, `58.40%`
- target cross-chunk spans: `167,597`, `40.17%`
- target cross-chunk selected fallback literals: `826,257`, `51.01%`

Interpretation: C3 should enter the full pipeline as a fallback side-stream or grammar extension. A flat global tokenizer replacement would throw away the structural fact that made C3 work.

## Main Issue Surfaced

The issue is no longer "can we find more global motifs?" The issue is how to expose chart-local fallback backreferences to the model without breaking stream legality, reconstruction, windowing, or decode order.

The pipeline target should be:

- unchanged `r0_delta` motif stream,
- optional C3 fallback side-stream,
- deterministic interleaving/reconstruction,
- exact round-trip to beat-chunk groups and mapper timepoints,
- explicit sequence-length and cross-window span statistics,
- disabled-by-default integration until mapper-side benefit is measured.

## Current Decision

Decision: `TEST_NEXT`

Next bounded card: build or harden the pipeline-facing C3 fallback side-stream tokenization artifact, not a global tokenizer replacement and not another broad mapper probe.

Kill/defer:

- broad global vocabulary expansion,
- C1/C2 table-heavy fallback codecs,
- C4=C1 patching,
- active-hold causality claims,
- default production enablement.

Promote cautiously:

- `r0_delta + LZ fallback-substream codec` as a research-backed representation family.

