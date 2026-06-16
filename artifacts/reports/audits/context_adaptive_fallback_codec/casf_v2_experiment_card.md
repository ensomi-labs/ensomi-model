# CASF v2 Experiment Card

## Title

Context-Adaptive Selective Fallback Codec for Active-Hold Beatmap Structure

## Hypothesis

The previous static-token experiments failed because exact residuals were too expensive, not because fallback structure is random. Keeping `r0_delta` motifs unchanged and replacing only fallback literals with charged context-adaptive residual codecs should reduce fallback and active-hold/chord-LN cost. A valid promotion requires the gain to survive mode cost, model/table cost, same-song filtering, and mapset bootstrap.

## Root Objective

Given `r0_delta`'s decoded context, find the cheapest legal code for the next hard residual without replacing the baseline motif path.

## Idea Quality

High as a follow-up audit because E1 showed structural fallback concentration and E3/E4 showed that whole-token residuals are too costly. The idea is bounded: only the literal fallback path changes, and every selector/mode/header/model cost is charged.

## Related Work / Analogies

- DEFLATE/LZ77: keep literals/matches separate and use local backreferences for repeated spans.
- PPM/CTW: model residual symbols with variable-order decoded history rather than a fixed token vocabulary.
- CABAC: binarize syntax elements and use context-adaptive probability tables.
- Dynamic patching/BLT analogy: spend more coding machinery only in high-entropy or active-state regions.
- Novelty layer, if any, is representation/coding-policy adaptation for osu!mania beatmap residuals, not a claim that CABAC, PPM, CTW, LZ77, or entropy patching are novel.

## Goal Decomposition

- Reproduce in-harness `r0_delta` baseline on the full existing beat-chunk cache.
- Build Phase 0 surprisal/literal forensics for fallback literals.
- Estimate a free-oracle upper bound across CABAC, PPM/CTW, LZ, and patch candidates.
- Evaluate charged legal variants with explicit mode/header/table/model cost.
- Report global, fallback-only, active-hold, chord/LN mixed, same-song-filtered, and bootstrap results.

## Candidate Variants

- F0: surprisal/literal forensics plus free-oracle upper bound.
- C1a: frozen train-fitted CABAC-style bitplane fallback codec.
- C1b: train-initialized online-adaptive CABAC-style bitplane fallback codec.
- C2a: PPM fallback-symbol codec with variable-order backoff.
- C2b: CTW-like mixture over context depths.
- C3: LZ-style chart-local active-span backreference codec.
- C4: entropy/active-state patched fallback spans with charged patch header and payload.
- C5: tiny neural entropy model is deferred unless C1/C2/C3 show a credible charged signal.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
|---|---|---|
| F0 | Baseline reproduces known `r0_delta`; high-surprisal fallback buckets are concentrated. | If oracle gain is `<0.02` bits/event, stop or kill CASF. |
| C1 | Bitplane context cross-entropy beats raw literal on active-hold fallback after table cost. | If mode/table cost erases the gain, mutate context or kill C1. |
| C2 | Variable-order history predicts fallback skeleton/masks better than global raw literal. | If same-song-filtered gain vanishes, treat as chart/source confound. |
| C3 | Same-chart previous decoded spans cover fallback payload after distance/length/mode cost. | If matches are short or residual-heavy, kill LZ-active-span. |
| C4 | Patch headers amortize payload gains over active/high-surprisal spans. | If patch gain is only oracle-only or tiny-bucket-only, defer C4. |

## Selected Variant

Run F0 first, then charged C1, C3, C2, and C4 in the recommended order. Do not run C5 unless at least one of C1/C2/C3 has a charged continuation signal.

## Selection Pressure

Prefer codecs that answer a specific failure mode from the previous cards while preserving the `r0_delta` motif path. Select by valid split only, report final once on test, and compare against the full in-harness baseline.

## Minimal Code Change

Add a cache-only audit module that reuses the existing train-only motif harness, identifies baseline fallback atom positions, computes decoder-known contexts, fits context models on train only, scores valid/test with charged CASF variants, writes JSON/CSV/Markdown artifacts, and includes unit tests for legality and reconstruction invariants.

## Implementation Families

- Diagnostic decomposition: group-level surprisal, fallback literal buckets, active-state/chord-LN attribution.
- Bitplane entropy model: delta/order symbols plus tap/start/end lane bits under decoder-known context.
- Variable-order context model: PPM backoff and CTW-like mixture over previous decoded skeletons.
- Local match model: chart-local exact, mirror, and skeleton backreference candidates.
- Patch model: deterministic or signaled active-state fallback spans with charged header.

## Files Likely To Change

- `src/pulsefield_model/osu_core/context_adaptive_fallback_codec_audit.py`
- `tests/osu_core/test_context_adaptive_fallback_codec_audit.py`
- `artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_experiment_card.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_report.json`
- `artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_result_log.md`
- `artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_variant_comparison.csv`
- `artifacts/reports/audits/context_adaptive_fallback_codec/casf_v2_diagnostics.csv`

## Dataset Slice

Use the full existing LE<=3 beat-chunk cache by default, with `limit_chunks=None`. Smoke tests may use synthetic data or an explicit limit, but final results must have `"limited": false`.

## Baseline / Comparator

B0 is the in-harness `r0_delta` scorer:

- keep baseline motif vocabulary, token model, dictionary cost, chunks, groups, events, and train/valid/test split unchanged;
- replace only literal fallback atom costs in CASF variants;
- compare charged bits/event with dictionary/model/table cost included.

## Primary Metric

Global charged test bits/event versus B0.

## Secondary Metrics

- fallback-only charged bits/event
- active-hold fallback charged bits/event
- chord/LN mixed fallback charged bits/event
- mode/header cost bits/event
- residual payload cost bits/event
- dictionary/model/table cost
- reconstruction mismatches
- codec usage share
- codec win rate before and after mode/header cost
- mapset-bootstrap confidence interval
- same-song-filtered test delta
- non-LN regression
- active-hold-count stratified result
- chart-local repetition rate
- patch length distribution

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py -q
```

## Guard Check

- Reconstruction mismatches must be `0`.
- No current raw atom or future group may be used for deterministic gates.
- Any mode, patch boundary, transform, residual, or model/table parameter must be charged.
- Context tables, thresholds, and selector policies must be fit on train/valid only, never test.
- Same-song-filtered and mapset-bootstrap outputs must be present for any promoted continuation signal.

## Qualitative Check

Inspect high-surprisal examples, codec usage, active-hold-count buckets, chord/LN mixed buckets, LZ match lengths/distances, and patch length distribution. Confirm that improvements are not isolated to a few mapsets or tiny event buckets.

## Positive Signal

- Reconstruction mismatches `== 0`.
- Global charged bits/event `<= B0 - 0.01`, or global charged bits/event `<= B0 + 0.005` with active-hold fallback gain `>= 0.05`.
- Active-hold/chord-LN gain survives mode/header/table cost.
- Same-song-filtered result holds.
- Mapset-bootstrap delta is mostly below zero.

## Negative Signal

- Free oracle wins but charged selector loses.
- Mode/header/table cost erases the payload gain.
- Bitplane residual cost matches raw literal cost.
- LZ matches are short or distance/length cost is too high.
- PPM/CTW gains vanish on same-song-filtered test.
- Dynamic patches mostly learn chart-specific noise.

## Kill Criteria

- Any reconstruction mismatch.
- Any uncharged mode, patch boundary, transform, residual, or model parameter.
- Deterministic gate uses current raw atom/future groups or test-tuned thresholds.
- Global charged regression is `> +0.01` bits/event with no active-hold continuation signal.
- Improvement concentrates in tiny buckets or a few mapsets.

## Expected Failure Modes

- Context fragmentation makes train-fitted tables sparse.
- Explicit mode cost overwhelms residual gains.
- Online adaptation helps train/valid but not test.
- Chart-local matches are too short after charging distance and length.
- Patch headers fail to amortize because fallback spans are short.
- Same-song duplicates inflate apparent context predictability.

## Expected Runtime / Runtime Budget

Smoke/unit tests should finish in seconds. Full-cache run is expected to take 5 to 20 minutes. Stop if the baseline cannot be reproduced within tolerance or if reconstruction fails.

## Confounders

The previous O1 result showed harness mismatch on the order of `+0.023267` bits/event, so this card must not compare against an external R0. Free-oracle bounds are diagnostics only and cannot be reported as legal compression wins.

## Result Interpretation Plan

- If F0 oracle gain is weak, kill/defer CASF before building larger models.
- If C1 wins, continue toward better bitplane contexts and table-cost accounting.
- If C3 wins, prioritize chart-local repetition over larger train dictionaries.
- If C2 wins but C1/C3 do not, mutate toward variable-order probability modeling.
- If only C4 wins, inspect whether patch headers are legal and robust.
- If all charged variants fail but oracle is strong, selector/mode coding is the bottleneck.
- If all charged variants and oracle fail, tokenizer-side lossless compression is likely near the current motif-family boundary.

## Result Log Template

Record:

- command, commit, dirty flag, runtime, limited flag, dataset summary;
- B0 global and fallback decomposition;
- F0 high-surprisal/oracle summary;
- charged variant comparison table;
- C1 bitplane diagnostics;
- C2 PPM/CTW diagnostics;
- C3 LZ diagnostics;
- C4 patch diagnostics;
- same-song-filtered result;
- bootstrap CI;
- recommendation: `KILL`, `MUTATE`, or `TEST_NEXT`.

## Next-Loop Action

Run the full-cache audit. Promote only if a charged legal variant satisfies the positive signal. Otherwise report the failure mode and next mutation, if any.
