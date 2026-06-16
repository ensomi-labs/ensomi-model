# Target Grammar v3 Matched v2.1 Timing Baseline Result Report

## Scope

This pass trains a matched 200-step v2.1 checkpoint on the same fixed slice as the v3 gate and runs the same six real-audio timing cases. It is a baseline audit, not a tokenizer or architecture change.

## Experiment Card

- Source card: `target_grammar_v3_matched_v21_timing_baseline_experiment_card.md`
- Selected variant: matched v2.1 200-step training plus same three cases x two prefixes.
- Comparator: `target_grammar_v3_multicase_timing_quality_summary.json`
- Date: `2026-06-17`

## Result

Decision: `MUTATE`.

- Reason: matched v2.1 also collapses to rigid grids, but v3 has stronger second-window starvation in sparse/dense 16s cases
- v2.1 checkpoint: `artifacts/tmp/mapper_v21_200step_timing_baseline/train/run/checkpoint.pt`
- Final eval total loss: `1.730002`
- Final eval token loss: `1.641958`
- Runs: `6`
- All legal: `True`
- Mean timing F1 @100ms: `0.745` vs v3 `0.643`
- Mean dominant-spacing ratio: `1.000` vs v3 `0.993`
- 16s second-window starvation count: `0` vs v3 `2`

## Case Results

| Case | Prefix | Norm diff | Gen/Ref events | F1 @100ms | Dominant spacing | Rigid ratio | Boundary | 2nd-window share | Legal | First generated times |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sparse` | `8000` | `-0.455` | `49/28` | `0.701` | `160` | `1.000` | `0.000` | `0.000` | `True` | `[160, 320, 480, 640, 800, 960]` |
| `sparse` | `16000` | `-0.455` | `99/57` | `0.692` | `160` | `1.000` | `0.000` | `0.495` | `True` | `[160, 320, 480, 640, 800, 960]` |
| `moderate` | `8000` | `-0.080` | `50/46` | `0.917` | `160` | `1.000` | `0.000` | `0.000` | `True` | `[0, 160, 320, 480, 640, 800]` |
| `moderate` | `16000` | `-0.080` | `100/86` | `0.882` | `160` | `1.000` | `0.000` | `0.490` | `True` | `[0, 160, 320, 480, 640, 800]` |
| `dense` | `8000` | `0.325` | `50/64` | `0.596` | `160` | `1.000` | `0.000` | `0.000` | `True` | `[0, 160, 320, 480, 640, 800]` |
| `dense` | `16000` | `0.325` | `100/144` | `0.680` | `160` | `1.000` | `0.000` | `0.490` | `True` | `[0, 160, 320, 480, 640, 800]` |

## v2.1 vs v3 Aggregate Comparison

| Metric | v2.1 | v3 | Delta |
| --- | ---: | ---: | ---: |
| Mean F1 @100ms | `0.745` | `0.643` | `0.102` |
| Mean event-count ratio | `1.202` | `0.918` | `0.284` |
| Mean dominant-spacing ratio | `1.000` | `0.993` | `0.007` |
| 16s mean second-window share | `0.492` | `0.178` | `0.314` |
| 16s starved case count | `0` | `2` | `-2` |

## What Passed

- Training completed the matched 200-step budget with finite metrics.
- All six v2.1 real-audio rollouts completed legally with no dead-end or max-token failure.
- The v2.1 runtime boundary is now auditable through the same session-backed rollout path as v3.
- The comparator uses the same selected maps, prefix lengths, normalized difficulties, and timing-match proxy as the v3 multicase audit.

## What Surfaced

- v2.1 also collapses to fixed grids: all six runs use a `160ms` dominant spacing.
- v2.1 overproduces regular events in all 16s cases; its mean F1 is higher here because the continuous grid hits more nearby reference events, not because timing quality is solved.
- v3 sparse/dense second-window starvation is not explained by the shared undertrained setting alone; v2.1 continues across the second window.
- Timing F1 remains a weak quality proxy because dense regular grids can match many reference events without learning beatmap structure.

## Interpretation

This is a MUTATE result, not a replacement approval. The fair v2.1 comparator proves the rigid-grid collapse is not unique to v3, while the v3 multicase audit still shows a worse continuation artifact on sparse/dense 16s prefixes.

## Verification

- Focused harness check before rollouts: `8 passed`.
- Runtime/v2.1 harness guard: `15 passed`.
- Training-comparison guard: `18 passed`.
- Inference/model guard: `26 passed`.
- Summary JSON validation: passed.

## Next Step

Define a bounded shared timing/decode calibration card, with an explicit v3 continuation guard for sparse/dense 16s cases.
