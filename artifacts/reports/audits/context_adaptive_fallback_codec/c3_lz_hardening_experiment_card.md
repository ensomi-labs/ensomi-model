# C3 LZ Fallback-Substream Hardening Experiment Card

## Hypothesis

The CASF v2 C3 gain is a legal chart-local fallback-payload repetition signal, not a side effect of duplicate-song leakage, dirty active-hold diagnostics, uncharged transforms, or fallback-only reconstruction. If C3 is ready to move toward pipeline tokenization, the gain should survive stricter decode guards, simple ablations, bounded windows, and fully charged pointer syntax.

## Root Objective

Harden `r0_delta + LZ fallback-substream codec` enough to decide whether it should become the next tokenizer-side representation layer for the full mapper pipeline.

## Idea Quality

High as a follow-up. CASF v2 gave a strong charged result (`-0.223101` test bits/event), while C1/C2/C4 showed that broader table-heavy tokenizer replacement is not currently viable. The remaining question is legality and decomposition, not novelty or a larger rewrite.

## Related Work / Analogies

- LZ77/DEFLATE: online local backreferences over already decoded history.
- Grammar-coded residual streams: keep a global dictionary for common motifs and a side-stream for local residual repetition.
- Transform coding: exact, mirror, and skeleton-plus-residual matches are engineering variants of known transform/pointer coding.

Novelty, if any, is at the beatmap representation policy layer: combining a corpus-trained `r0_delta` motif tokenizer with a chart-local fallback-substream backreference codec. This is not a claim that LZ, transform coding, or residual coding are novel.

## Goal Decomposition

- Reproduce B0 and C3 under the same full-cache harness.
- Upgrade reconstruction from fallback-payload-only to token-stream, full-chart, span-boundary, and transform-inverse guards.
- Decompose C3 gain by exact, skeleton residual, mirror transform, active-only, non-active, window size, and pointer coding.
- Check whether C3 survives clean/dirty active-hold trace splits.
- Decide whether the next loop should be pipeline integration, C3 mutation, or active-hold trace repair.

## Candidate Variants

- A1 exact-only LZ: only exact fallback payload matches.
- A2 skeleton-plus-residual LZ: same skeleton, charged lane/order residual.
- A3 mirror-transform LZ: mirrored prior fallback payloads with charged transform.
- A4 active-only C3: C3 only when active-hold/chord-LN fallback structure is present.
- A5 non-active C3: C3 only when active-hold/chord-LN structure is absent.
- A6 window-size sweep: previous 16, 32, 64, 128, 256, 512, 1024 fallback records, plus full previous chart.
- A7 pointer-code sweep: fixed-width, Elias gamma, Elias delta, power-bucket, train-fitted categorical, and online adaptive categorical pointer costs.
- A8 match-source diagnostics: fallback-substream baseline plus phase-restricted fallback-source ablations.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
|---|---|---|
| guards | token-stream, full-chart, span-boundary, transform-inverse mismatches are zero | Any mismatch blocks promotion. |
| A1 | exact-only remains below B0 by at least `0.05` bits/event | If yes, most signal is pure local repetition. |
| A2 | skeleton residual improves beyond exact-only after residual cost | If no, residual abstraction is not worth promoting yet. |
| A3 | mirror-only or all-transform gain survives transform cost | If no, keep mirror only as a diagnostic. |
| A4/A5 | active and non-active split gains are both reported | Separates generic repetition from active/LN-specific interpretation. |
| A6 | bounded window retains most C3 gain | If only full-history wins, defer pipeline integration. |
| A7 | pointer code remains charged and positive | If gain depends on undercharged pointer syntax, mutate. |
| A8 | phase-restricted sources do not contradict fallback-substream result | If gain is only a narrow phase artifact, report that scope. |

## Selected Variant

Run the promotion-critical C3 ablations in one cache-only audit. Keep the baseline motif path unchanged and do not modify the mapper training pipeline in this card. The wider A6/A7/A8 sweep remains available through `--c3-wide-sweep`; it is not the default full-cache gate because candidate materialization is expensive.

## Selection Pressure

Promote only a simple legal C3 family that passes full reconstruction and stays positive with a bounded window and charged pointer syntax. Prefer exact-only or exact+mirror over skeleton residual if the simpler variant keeps most of the gain.

## Minimal Code Change

Extend the CASF audit with C3-specific plan metadata, stricter reconstruction checks, C3 ablation builders, and result artifacts. Do not change default mapper tokenization or training data until this card passes.

## Files Likely To Change

- `src/pulsefield_model/osu_core/context_adaptive_fallback_codec_audit.py`
- `tests/osu_core/test_context_adaptive_fallback_codec_audit.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_result_log.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_comparison.csv`
- `artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_diagnostics.csv`

## Dataset Slice

Use the full existing LE<=3 beat-chunk cache by default with `limit_chunks=None`. Smoke tests may use synthetic rows or an explicit limit, but final results must have `"limited": false`.

## Baseline / Comparator

B0 is the unchanged in-harness `r0_delta` scorer. All variants may only replace fallback literal payloads; motif vocabulary, motif path, dictionary cost, chunks, groups, events, and splits must remain unchanged.

## Primary Metric

Charged test bits/event delta versus B0.

## Secondary Metric

Same-song-filtered delta, mapset-bootstrap confidence interval, fallback payload delta, mode/pointer/residual bits, exact/mirror/skeleton usage, match coverage, mean length/distance, active-only and non-active gains, clean/dirty active-hold split deltas, and reconstruction mismatch counts.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit --c3-hardening
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit --c3-hardening --c3-wide-sweep
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py -q
```

## Guard Check

- Fallback-payload reconstruction mismatches must be `0`.
- Full token-stream reconstruction mismatches must be `0`.
- Full chart reconstruction mismatches must be `0`.
- Span-boundary reconstruction mismatches must be `0`.
- Transform-inverse mismatches must be `0`.
- Selected LZ spans must reference only prior fallback-substream records from the same chart and split.
- Any mode, pointer distance, length, transform, residual, learned table, or selector must be charged.
- Same-song-filtered and bootstrap outputs must be present for the promoted C3 variant.

## Qualitative Check

Inspect usage and gain by exact, skeleton, mirror, active-only, non-active, window size, pointer code, and clean/dirty active-hold trace buckets. Verify that improvements are not concentrated in invalid-transition maps.

## Positive Signal

- All reconstruction guards pass.
- Selected C3-family test delta is `<= -0.05` bits/event.
- Same-song-filtered delta remains close to the unfiltered delta.
- Bootstrap 95% interval is entirely below `0`.
- At least one simple family, exact-only or exact+mirror, retains a meaningful gain.
- Bounded window does not lose most of the gain.

## Negative Signal

- Any reconstruction mismatch.
- Gain disappears under simple exact or exact+mirror coding.
- Gain depends on uncharged transform, future context, cross-chart history, or dirty active-hold traces.
- Pointer cost ablation shows the original C3 was materially undercharged.
- Same-song-filtered delta collapses or bootstrap interval crosses zero.

## Kill Criteria

Kill or defer C3 promotion if full-chart reconstruction fails, selected spans overlap illegally, selected spans cross source/split boundaries, bounded windows lose nearly all gain, or clean-subset C3 delta collapses.

## Expected Failure Modes

- Skeleton residual cost erases the abstraction gain.
- Mirror transform usage is too small after transform cost.
- Active-only gain is confounded by dirty active-hold trace states.
- Larger windows win but are too expensive for pipeline use.
- Train-fitted pointer priors overfit distance/length distributions.

## Expected Runtime / Runtime Budget

Smoke tests should finish in seconds. Full-cache C3 hardening should finish in 5 to 20 minutes. Stop if B0 does not reproduce, if any hard reconstruction guard fails, or if selected C3 no longer beats B0.

## Confounders

The active-hold trace still has invalid transitions and segment-end active holds. Active-state bucket gains are diagnostics only until clean/dirty trace splits prove the C3 result is not dependent on dirty trace rows.

## Result Interpretation Plan

- If exact-only wins strongly, promote a simple LZ fallback-substream codec.
- If exact+mirror wins but skeleton does not, keep transform coding narrow.
- If skeleton residual is required, run a separate residual-tokenization card before pipeline changes.
- If only active-only wins, repair active-hold trace before making causal claims.
- If both active and non-active win, describe C3 as generic chart-local fallback repetition, not active-hold semantics.
- If hardening passes, next card may integrate C3 as an optional beat-chunk tokenization artifact feeding the full mapper pipeline.

## Result Log Template

Record command, commit, dirty flag, runtime, limited flag, B0/C3 reproduction, reconstruction guards, ablation table, window sweep, pointer-code sweep, active trace split, selected C3 family, recommendation, and next-loop action.

## Next-Loop Action

If this card passes, implement a pipeline-facing optional tokenization artifact. If it fails, mutate C3 according to the failed guard or run Active-Hold Trace Sanity Repair before further promotion.
