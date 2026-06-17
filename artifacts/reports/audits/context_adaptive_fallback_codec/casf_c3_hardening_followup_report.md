# CASF/C3 Hardening Follow-Up Report

## Scope

This follow-up reads the current CASF v2 and C3 hardening artifacts against the June 17, 2026 pasted audit brief. It does not change mapper defaults, tokenizer behavior, training data, or the C3 harness.

Primary artifacts checked:

- `casf_v2_report.json`
- `c3_lz_hardening_report.json`
- `c3_lz_hardening_result_log.md`
- `casf_c3_final_audit_synthesis_report.md`

Verification run in this pass:

```bash
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py -q
python3 -m json.tool artifacts/reports/audits/context_adaptive_fallback_codec/c3_lz_hardening_report.json >/dev/null
python3 -m json.tool artifacts/reports/audits/context_adaptive_fallback_codec/casf_c3_final_audit_synthesis_summary.json >/dev/null
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit --c3-hardening --c3-wide-sweep
```

The focused tests passed: `8 passed in 0.75s`.

The wide sweep command was interrupted after more than 12 minutes because it was still in mirror candidate construction. No completed wide-sweep report was produced in this pass.

## Decision

Decision: `TEST_NEXT`, but only for the representation family and the offline side-stream path.

Correct name:

`r0_delta + LZ fallback-substream codec`

Avoided names:

- new global tokenizer
- active-hold tokenizer
- active-hold semantic model
- production/default mapper format

## What Is Proved

CASF v2 proved a real charged compression signal in the fallback substream.

- B0 `r0_delta` test charged bits/event: `4.421714207191301`
- C3 test charged bits/event: `4.198613101989439`
- C3 delta versus B0: `-0.2231011052018621`
- Same-song-filtered delta: `-0.22353525270533403`
- Bootstrap mean delta: `-0.22282466691710426`
- Bootstrap 95% interval: `[-0.2580919949410291, -0.18675671602064908]`
- Fallback-payload reconstruction mismatches: `0`

Interpretation: the remaining compressible structure is chart-local and online-referential, not primarily corpus-global vocabulary structure.

## What Passed

The stricter C3 hardening report passed both research and engineering promotion gates for the selected variant `a2_skeleton_residual_all_fallback_w256`.

- Selected test charged bits/event: `4.0368045441872065`
- Delta versus B0: `-0.3849096630040947`
- Same-song-filtered delta: `-0.38592890137466185`
- Clean-trace test delta: `-0.3809793375551669`
- Dirty-trace test delta: `-0.43255187244188775`
- Bootstrap 95% interval: `[-0.4235772949889358, -0.34584630062559635]`
- Research promotion pass: `true`
- Engineering promotion pass: `true`

Hard reconstruction passed with zero mismatches across the audited guards:

- fallback-payload reconstruction
- full token-stream reconstruction
- full chart reconstruction
- span-boundary reconstruction
- transform inverse
- baseline motif-stream reconstruction

The decomposition supports the local-repetition thesis:

| Variant | Test Delta |
| --- | ---: |
| active C3, fixed window 256 | `-0.24256829527744017` |
| A1 exact-only all fallback | `-0.2785859452434707` |
| A2 skeleton-residual all fallback | `-0.3849096630040947` |
| A3 mirror-only all fallback | `-0.2900851303531917` |
| A5 non-active all fallback | `-0.09266425164250869` |
| A6 active fixed window 128 | `-0.23783549837040585` |

Exact-only already wins, so pure chart-local fallback repetition is sufficient for a strong positive result. Skeleton-residual and mirror variants add more gain, but they are not required to prove the existence of the signal.

## What Surfaced

The main scientific caveat remains active-hold causality.

- Invalid active-hold transitions: `1789`
- Segment-state resets: `1788`
- Dirty trace source count: `711`
- Clean test source count: `904`
- Dirty test source count: `61`

The clean-trace test delta is still strongly negative, so dirty active-hold traces are not required for the C3 win. But the result still does not prove that active-hold semantics caused the gain.

Correct claim:

`C3 wins on chart-local fallback-payload repetition under a charged fallback-substream harness.`

Unsupported claim:

`C3 wins because it models active-hold causality.`

The wide `--c3-wide-sweep` path also surfaced an engineering issue. The current full-cache wide sweep is not runtime-bounded enough for a routine promotion gate. It was still inside `_lz_candidate_options_by_record` / `_lz_match_length` mirror matching when interrupted after more than 12 minutes. The bounded non-wide hardening gate remains valid, but A6/A7/A8 full wide decomposition needs a faster implementation or a bounded dataset/run mode before it should be used as a standard gate.

The completed hardening report records `code_dirty=true`. Treat that flag cautiously for this artifact because the audit writes output CSVs before building the JSON report metadata. The current follow-up started from a clean git status before adding these report files.

## What Is Not Proved

C3 is not yet an end-to-end mapper win. The current artifacts do not prove:

- mapper training loss improves;
- generated chart quality improves;
- the model can legally emit C3 references during autoregressive decode;
- side-stream semantics are learnable from audio/control inputs;
- sequence length, replay state, and batching costs are acceptable for default training;
- active-hold state is the causal source of the compression gain.

C3 is also not a simple flat global tokenizer replacement. The selected hardening variant has many noncontiguous and cross-window spans, so the correct promotion target is a fallback side stream or a grammar extension, not broad replacement of `r0_delta`.

## Updated Audit Target

Next target:

`Can C3's chart-local fallback reference structure be exposed as a legal mapper-side target or auxiliary target without target leakage, while preserving deterministic reconstruction and acceptable sequence/state cost?`

Recommended next bounded work:

- keep the current `r0_delta + LZ fallback-substream codec` result as promoted research evidence;
- optimize or bound the wide A6/A7/A8 sweep before relying on it;
- run Active-Hold Trace Sanity Repair only as a diagnostic, not as a prerequisite for the local-repetition claim;
- test one legal mapper-facing exposure path at a time: auxiliary target, emitted target grammar, or two-stage predicted C3 plan.
