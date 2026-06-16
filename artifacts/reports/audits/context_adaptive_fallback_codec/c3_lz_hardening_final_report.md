# C3 LZ Fallback-Substream Hardening Final Report

## Scope

This pass hardened the CASF v2 C3 result without changing the `r0_delta` motif path. The tested representation is a two-layer codec:

- Layer 1: unchanged corpus-global `r0_delta` motif tokenizer.
- Layer 2: chart-local fallback-substream LZ side codec for residual fallback literals.

The full-cache run used `limit_chunks=None`.

## Baseline Reproduction

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- Observed B0 in hardening harness: `4.421714207191301`
- Baseline consistency: pass
- Runtime: `430.970s`
- Code dirty during run: `true`

## What Passed

The selected hardening variant was `a2_skeleton_residual_all_fallback_w256`.

- Valid charged bits/event: `4.088714`
- Test charged bits/event: `4.036805`
- Test delta versus B0: `-0.384910` bits/event
- Same-song-filtered delta: `-0.385929` bits/event
- Bootstrap 95% interval: `[-0.423577, -0.345846]`
- Clean active-hold-trace test delta: `-0.380979` bits/event
- Dirty active-hold-trace test delta: `-0.432552` bits/event

The clean-subset result means the C3 gain does not depend on maps with invalid active-hold trace transitions or segment-end active holds.

## Reconstruction Guards

All hard guards passed with zero mismatches for every tested C3 variant:

- fallback-payload reconstruction mismatches: `0`
- full token-stream reconstruction mismatches: `0`
- full chart reconstruction mismatches: `0`
- span-boundary mismatches: `0`
- transform-inverse mismatches: `0`
- baseline motif-stream reconstruction mismatches: `0`

## Ablation Results

Charged test deltas versus B0:

- `a2_skeleton_residual_all_fallback_w256`: `-0.384910`
- `a3_mirror_only_all_fallback_w256`: `-0.290085`
- `a1_exact_only_all_fallback_w256`: `-0.278586`
- `a7_active_all_elias_gamma_w256`: `-0.248259`
- `c3_active_all_fixed_w256`: `-0.242568`
- `a6_active_all_fixed_w128`: `-0.237835`
- `a7_active_all_power_bucket_w256`: `-0.235131`
- `a5_non_active_all_fixed_w256`: `-0.092664`

Interpretation:

- Exact-only already wins strongly, so pure chart-local repetition is real.
- Mirror-only also wins after strict mirror-order handling.
- Skeleton+residual wins most, so abstract pattern reuse beyond exact payload repetition matters.
- Non-active fallback also wins, but much less than active/all fallback. Do not over-claim active-hold causality; the result is broader chart-local fallback repetition.
- A 128 fallback-record window keeps most of the active C3 gain.
- Elias gamma and power-bucket pointer codes remain positive, so the original fixed pointer cost is not the only reason C3 wins.

## Selected Span Structure

For the selected skeleton-residual variant:

- selected spans: `417,208`
- selected fallback literals: `1,619,653`
- mean raw bits/span: `41.347963`
- mean charged bits/span: `25.467779`
- mean residual bits/span: `12.292229`
- span length `3plus`: `82.27%` of selected fallback literals
- distance `1to16`: `34.16%` of selected fallback literals

Main integration caveat:

- noncontiguous main-stream spans: `199,906` spans (`47.92%`)
- noncontiguous selected fallback literals: `945,902` (`58.40%`)
- target cross-chunk spans: `167,597` (`40.17%`)
- target cross-chunk selected fallback literals: `826,257` (`51.01%`)

This means C3 should not be promoted as a simple flat token replacement. It should enter the full pipeline as an optional fallback side-stream or grammar extension that interleaves with the unchanged motif stream.

## What Surfaced

1. Strict mirror order matters.
   The hardening code now mirrors lane labels in `order_signature`, not only tap/start/end masks.

2. C3 is stronger after decomposition than the original active-span-only result.
   CASF v2 C3 was `-0.223101`; hardening found active fixed-window C3 at `-0.242568` and skeleton-residual all-fallback at `-0.384910`.

3. The active-hold trace caveat no longer blocks C3 promotion.
   Clean-trace maps still show `-0.380979` bits/event.

4. The representation is side-stream-shaped.
   Many selected spans skip over motif-covered groups or cross chunk boundaries, so the next pipeline step must preserve side-stream decode semantics.

5. Wide A6/A7/A8 sweep was not run by default.
   The full hardening run uses the promotion-focused subset. The wider sweep remains available through `--c3-wide-sweep` but was too expensive as a default full-cache gate.

## Recommendation

`TEST_NEXT`: build an optional pipeline-facing fallback side-stream tokenization artifact.

Do not replace the mapper vocabulary yet. The next bounded card should implement an encode/decode artifact that exposes:

- unchanged motif stream tokens,
- C3 fallback side-stream tokens,
- deterministic interleaving/reconstruction,
- per-window mapper dataset statistics,
- sequence-length/runtime impact,
- exact round-trip back to beat-chunk groups and mapper timepoints.

## Artifacts

- `c3_lz_hardening_experiment_card.md`
- `c3_lz_hardening_report.json`
- `c3_lz_hardening_result_log.md`
- `c3_lz_hardening_comparison.csv`
- `c3_lz_hardening_diagnostics.csv`
- `c3_lz_hardening_final_report.md`

## Verification

Commands run:

```bash
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit --c3-hardening
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py -q
```

Observed:

- focused C3/CASF tests: `8 passed`
- full C3 hardening run completed with `limited=false`
- final focused regression suite: `19 passed in 0.91s`
