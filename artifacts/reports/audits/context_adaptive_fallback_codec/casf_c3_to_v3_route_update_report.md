# CASF/C3 To v3 Route Update Report

## Scope

This report reconciles the attached CASF v2 interpretation with the artifacts that now exist in the repository. It is a synthesis report only; it does not introduce a new experiment or change mapper defaults.

Source artifacts:

- `casf_v2_final_report.md`
- `c3_lz_hardening_final_report.md`
- `c3_side_stream_pipeline_final_report.md`
- `c3_full_pipeline_role_audit_result_report.md`
- `c3_target_role_feasibility_result_report.md`
- `c3_raw_structure_audit_result_report.md`
- `c3_raw_factor_probe_result_report.md`
- `../mapper_v2_1_grammar/target_grammar_v3_full_dataset_audit_result_report.md`

## Current Decision

Decision: `TEST_NEXT`, but not as broad C3 input conditioning.

The current route is:

1. Keep `r0_delta` as the corpus-global motif layer.
2. Treat C3 as a legal chart-local fallback-substream codec and target-side representation family.
3. Do not promote target-derived C3 sidecar tokens as mapper inference inputs.
4. Use the v3 event-token grammar as the current local teacher-forcing replacement candidate because it is reversible, shorter than v2.1, and does not require C3-style cross-window references.

## What Is Proved

### CASF v2 Proved Chart-Local Fallback Repetition

The original CASF v2 run showed that the remaining compressible structure is chart-local and online-referential, not primarily a missing global vocabulary.

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- C3 test charged bits/event: `4.198613101989439`
- Delta: `-0.2231011052018621` bits/event
- Same-song-filtered delta: `-0.22353525270533403`
- Bootstrap 95% interval: `[-0.2580919949410291, -0.18675671602064908]`
- Fallback-payload reconstruction mismatches: `0`

This proves a robust fallback-substream compression signal under charged scoring. It does not prove active-hold causality or mapper-generation benefit.

### C3 Hardening Proved The Codec Is Legally Reconstructive

The full C3 hardening pass strengthened the CASF result.

- selected variant: `a2_skeleton_residual_all_fallback_w256`
- selected test charged bits/event: `4.0368045441872065`
- delta versus B0: `-0.3849096630040947` bits/event
- same-song-filtered delta: `-0.38592890137466185`
- clean active-hold-trace delta: `-0.3809793375551669`
- dirty active-hold-trace delta: `-0.43255187244188775`
- bootstrap 95% interval: `[-0.4235772949889358, -0.34584630062559635]`
- research promotion pass: `true`
- engineering promotion pass: `true`

All hard reconstruction guards passed with zero mismatches:

- fallback-payload reconstruction
- full token-stream reconstruction
- full chart reconstruction
- span-boundary reconstruction
- transform inverse
- baseline motif stream

This removes the earlier concern that C3 only reconstructed fallback payloads while hiding stream or chart mismatches.

### C3 Decomposition Proved Exact Repetition Is Already Enough

The hardening ablations show that C3 is not only a complex skeleton-residual trick.

- exact-only all-fallback delta: `-0.2785859452434707`
- mirror-only all-fallback delta: `-0.290085`
- active fixed-window C3 delta: `-0.24256829527744017`
- bounded 128-record active window delta: `-0.23783549837040585`
- skeleton-residual all-fallback delta: `-0.3849096630040947`

Exact-only winning is the key result. The fallback stream contains repeated chart-local payloads even before residual abstraction.

### Side-Stream P0 Proved Lossless Pipeline Artifact Feasibility

The P0 side-stream pipeline artifact passed full-cache round-trip.

- selected C3 variant: `a2_skeleton_residual_all_fallback_w256`
- side-stream reference errors: `0`
- side-stream payload mismatches: `0`
- full token-stream mismatches: `0`
- group signature mismatches: `0`
- mapper-timepoint-compatible mismatches: `0`
- side-stream tokens: `3,203,904`
- fallback placeholders: `2,786,696`
- combined main+side tokens: `8,856,468`
- combined/main baseline token ratio: `1.566805`

This proves the representation can be materialized losslessly as an artifact. It also shows why it is not automatically a compact learned sequence format.

### v3 Proved A Simpler Local Target Grammar Is Viable

The v3 full-dataset audit passed as a local event-token target grammar.

- full-audit windows: `174515`
- full-audit reconstruction mismatches: `0`
- full-audit token reduction: `20.54%`
- eval token reduction: `19.96%`
- eval total-bit reduction: `5.43%`
- full-chart terminal windows: `9360`
- terminal event-at-chart-end windows: `9360`
- cross-window LN windows: `80316`

The important difference from C3 is that v3 needs no target-derived side stream, no cross-window backreference replay, and no future target lookup. It is therefore a better current candidate for teacher-forced mapper target replacement.

## What Surfaced

### C3 Is Side-Stream-Shaped

For the selected hardening variant:

- selected spans: `417,208`
- selected fallback literals: `1,619,653`
- noncontiguous main-stream spans: `199,906` (`47.92%`)
- noncontiguous selected fallback literals: `945,902` (`58.40%`)
- target cross-chunk spans: `167,597` (`40.17%`)
- target cross-chunk selected fallback literals: `826,257` (`51.01%`)

This is the main integration issue. C3 works because it references fallback-substream history, often across chunk and main-stream boundaries. Flattening it into ordinary mapper tokens would hide the structure that made it useful.

### Current C3 Sidecar Is Target-Derived

The full-pipeline role audit classified the current exact C3 mapper-window side stream as an offline target-side artifact, not a legal production input.

Current role classification:

- current path: offline target-derived conditioning probe
- legal production input: no
- current pooled input-conditioning promotion: kill
- next family: auxiliary target or target grammar candidate

This kills the current C3 input-conditioning path for production. More training on target-derived pooled sidecar conditioning would not prove deployability.

### C3 Auxiliary Probes Did Not Clear The Mapper-Side Gate

The C3 auxiliary path remained legal because labels are target-side only, but the later RAW/kind diagnostics exposed weak mapper-side learnability:

- RAW model recall@20: `0.084309`
- unigram RAW recall@20: `0.141686`
- RAW-only model recall@20: `0.185012`
- RAW-only unigram recall@20: `0.201405`
- RAW factor max-marginal joint coverage@5: `0.553864`
- too few factor fields beat unigram at K=3

Interpretation: C3 is strong as a codec-side representation, but the tested auxiliary mapper path has not yet shown enough predictive leverage to justify full integration.

## What Is Not Proved

- C3 has not proved active-hold causality. Clean-trace wins show the result does not depend on dirty active-hold traces, but not that active-hold state causes the gain.
- C3 has not proved end-to-end mapper quality improvement.
- Current C3 sidecar conditioning has not proved inference legality.
- v3 has not yet proved trained mapper quality; it has proved reversible target grammar pressure and full-dataset reconstruction.
- None of these reports prove novelty beyond the representation-policy layer. The closest analogies remain LZ-style local backreference, transform coding, residual coding, and teacher-forced target grammar compression.

## Current Issue

The research issue is no longer whether fallback has structure. It does.

The issue is which structure can be exposed to the mapper without leaking target information or requiring complex replay:

- C3 exposes the strongest compression signal, but it is side-stream and target-history shaped.
- v3 exposes a smaller local grammar gain, but it is directly teacher-forcing friendly and online-decodable.

That is why the immediate pipeline replacement path should focus on v3, while preserving C3 as a codec-side/auxiliary target research family rather than forcing it into mapper input conditioning.

## Recommended Next Loop

Proceed with the v3 full-pipeline replacement card only as a bounded implementation experiment:

- implement v3 dataset/model/loss/training/inference surfaces behind explicit v3 entrypoints;
- preserve v2.1 defaults until v3 training and rollout are validated;
- keep C3 reports as evidence for chart-local fallback structure;
- do not reintroduce C3 input conditioning unless a legal generation-time C3 source exists;
- revisit C3 only through target grammar, auxiliary target, or two-stage predicted-plan cards.

Kill/defer:

- broad global vocabulary expansion;
- C1/C2 table-heavy fallback codecs;
- current C3 pooled input-conditioning promotion;
- claims that C3 is an active-hold semantic model;
- default mapper replacement before trained v3 evidence exists.
