# CASF v2 Full-Cache Final Report

## Scope

This run tested the new framing from the attached request: not broad tokenizer replacement, but a context-adaptive selective fallback codec. The baseline motif path stayed unchanged. Only `r0_delta` fallback literal payloads were replaced or diagnosed.

Full-cache dataset:

- chunks: `1,199,632`
- events: `16,064,569`
- groups: `9,719,273`
- maps: `9,242`
- mapsets: `3,573`
- test fallback literals: `276,741`
- full-run runtime: `474.346s`
- `limited`: `false`
- code dirty: `true`

## Baseline Check

The in-harness B0 baseline reproduced exactly:

- expected `r0_delta` test charged bits/event: `4.421714207191301`
- observed B0 test charged bits/event: `4.421714207191301`
- reconstruction guard: pass
- fallback-payload reconstruction mismatches: `0`
- mapset bootstrap keys: fixed and verified, `362` overlapping test mapsets

## What Passed

### C3 LZ Active-Span Backreference

C3 is the only legal charged variant that passed all gates.

- valid charged bits/event: `4.257266`
- test charged bits/event: `4.198613`
- test delta versus B0: `-0.223101` bits/event
- same-song-filtered delta: `-0.223535` bits/event
- bootstrap mean delta: `-0.222825`
- bootstrap 95% interval: `[-0.258092, -0.186757]`
- active-hold fallback gain: `1.195931` bits/fallback-event
- research pass: `true`

C3 usage on test fallback literals:

- raw literal unchanged: `195,109` (`70.50%`)
- LZ skeleton: `46,695` (`16.87%`)
- LZ exact: `18,614` (`6.73%`)
- LZ mirror: `16,323` (`5.90%`)

This supports a real chart-local repetition signal in the fallback payload stream under the current charged scoring harness. The positive result is not a static vocabulary result; it is a selective local backreference result.

## What Failed

### C1 CABAC-Style Bitplane

C1 payloads were locally cheaper, but table cost erased the gain.

- C1a frozen test delta: `+1.017676` bits/event
- C1b online test delta: `+1.015245` bits/event
- C1 table/model cost: `2,683,934` bits
- payload-only fallback delta was favorable: around `-1.73` bits/fallback-event

Interpretation: bitplane syntax has signal, but the current train-fitted table accounting is too expensive. Do not promote C1 as implemented.

### C2 PPM / CTW

C2 failed hard under charged table accounting.

- C2a PPM test delta: `+227.675006` bits/event
- C2b CTW-like test delta: `+225.297546` bits/event
- C2 table/model cost: `380,332,350` bits

Interpretation: the current variable-order context family fragments badly. Kill this C2 formulation unless the table representation is radically compressed or the context space is narrowed.

### C4 Signaled Active Patch

C4 improved fallback payloads but failed globally.

- C4 test delta: `+1.309988` bits/event
- C4 test fallback payload delta: `-0.840208` bits/fallback-event
- patch usage: `168,250` test fallback literals (`60.80%`)

Interpretation: patching with C1 inherits C1's table-cost failure. The patch idea is not killed generally, but this C4=C1-patch implementation fails.

### S1 Explicit Per-Fallback Selector

The explicit selector failed because it combined expensive C1/C2 model costs.

- S1 test delta: `+225.943023` bits/event
- S1 payload delta: `-1.003025` bits/fallback-event

Interpretation: do not mix table-heavy codecs in an explicit selector without first solving model cost. The corrected S1 selector only chooses raw/C1/C2 exact-syntax payloads; C3/C4 span codecs are not used as per-fallback selector candidates.

## F0 Oracle Bound

The free oracle remains a diagnostic only.

- oracle test charged bits/event: `3.753078`
- oracle gain versus B0: `0.668637` bits/event
- oracle fallback payload delta: `-2.038573` bits/fallback-event

This says there is still headroom, but the oracle is illegal because it chooses the cheapest payload without paying a selector/table policy. It should be used only to guide mutations.

## Structural Findings

High-surprisal fallback is concentrated in the expected hard structure:

- top 1% fallback literals: active-hold share `0.6910`, chord/LN mixed share `0.4062`
- top 5%: active-hold share `0.7155`, chord/LN mixed share `0.4697`
- top 10%: active-hold share `0.7282`, chord/LN mixed share `0.4428`

C3 bucket gains on test:

- active-hold count `3plus`: `-1.195931` bits/fallback-event
- active-hold count `2`: `-1.020280`
- active-hold count `1`: `-0.835705`
- active-hold count `0`: `-0.197055`
- active-span yes: `-0.919762`
- active-span no: `0.000000`
- chord/LN mixed yes: `-0.685777`
- chord/LN mixed no: `-0.676991`

LZ diagnostics:

- candidate coverage: `0.859126`
- mean match length: `3.152901`
- mean distance: `69.400918` fallback positions

Active-hold trace caveat:

- invalid active-hold transitions: `1,789`
- segment-end active holds: `1,788`

These are inherited from the prior fallback forensic pass. They do not invalidate C3's local-backreference result, because C3 references prior fallback payloads rather than relying only on active-hold state. They do weaken any strong causal interpretation of active-hold state itself.

## Result Interpretation

The previous five-card conclusion should be updated carefully:

- Static token vocabulary replacement remains unsupported.
- Table-heavy context models are not yet viable under charged accounting.
- The fallback stream is compressible by chart-local backreference under the current fallback-substream scoring harness.
- The strongest evidence now favors a CASF/LZ residual side-stream, not a larger motif dictionary.

C5 tiny neural entropy modeling was not run in this card. It was explicitly optional and should be a separate Experiment Card if the owner wants it. The current positive result already identifies a simpler winning family: refine C3 first.

## Artifacts

- `casf_v2_experiment_card.md`
- `casf_v2_report.json`
- `casf_v2_result_log.md`
- `casf_v2_variant_comparison.csv`
- `casf_v2_diagnostics.csv`
- smoke artifacts: `smoke_casf_v2_*`

## Verification

Commands run:

```bash
uv run --group dev pytest tests/osu_core/test_context_adaptive_fallback_codec_audit.py tests/osu_core/test_fallback_forensic_audit.py tests/osu_core/test_duration_ln_tokenization_audit.py -q
uv run python -m pulsefield_model.osu_core.context_adaptive_fallback_codec_audit
```

Observed:

- focused tests: `17 passed in 0.94s`
- full no-limit CASF run completed
- final report has `limited=false`
- baseline consistency pass: true
- reconstruction pass: true
- mapset bootstrap available: true

## Next Loop

Recommended next card: C3 refinement audit.

Keep the C3 fallback side-stream, then ablate:

- fallback-substream distance definition,
- exact versus mirror versus skeleton transform accounting,
- span contiguity policy,
- universal versus learned distance/length codes,
- train-free table cost remains zero unless a learned model is added.

Do not move to mapper-side probes yet; the next question is whether the C3 codec remains valid under stricter stream reconstruction, fallback-substream contiguity, and transform accounting.
