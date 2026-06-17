# Target Grammar v3 Time-Shift Distance Masked-Logit NaN Diagnostic Result Report

## Scope

This diagnostic inspects real v3 training batches without optimizer steps. It compares grammar-masked `logits_final` time-shift rows against a finite pre-mask candidate surface.

## Decision

Route: `PASS_ROW_FILTERED_TIME_SHIFT_DISTANCE_REPAIR`.

- Reason: masked loss is finite after row filtering despite all-nonfinite non-target or padded rows; target time-shift rows retain finite nonzero gradients
- Next step: Run the bounded tiny training gate and record whether enabled training remains finite.

## Metrics

- total rows: `392`
- valid rows: `321`
- target time-shift rows: `185`
- target all-nonfinite masked rows: `0`
- target all-nonfinite masked share: `0.00%`
- any-row all-nonfinite masked rows: `71`
- any-row all-nonfinite masked share: `18.11%`
- valid all-nonfinite masked rows: `0`
- valid non-target all-nonfinite masked rows: `0`
- invalid all-nonfinite masked rows: `71`
- finite masked candidate share: `100.00%`
- finite pre-mask candidate share: `100.00%`
- gold masked finite share: `100.00%`
- masked loss values: `[0.07376536726951599]`
- masked loss all finite: `True`
- pre-mask loss values: `[0.11251014471054077]`
- skip-masked loss values: `[0.07376536726951599]`
- row-filtered masked loss values: `[0.07376536726951599]`
- pre-mask gradient abs sums: `[0.2623143792152405]`
- pre-mask gradient all finite: `True`
- pre-mask gradient any nonzero: `True`
- row-filtered masked gradient abs sums: `[0.17154835164546967]`
- row-filtered masked gradient all finite: `True`
- row-filtered masked gradient any nonzero: `True`

## Examples

| Batch | Step | ms | Target | Masked finite TS | Pre-mask finite TS | Gold masked finite | Gold pre-mask finite |
| ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| 0 | 0 | 0 | `TS_30` | 22 | 22 | True | True |
| 0 | 2 | 30 | `TS_300` | 22 | 22 | True | True |
| 0 | 3 | 330 | `TS_20` | 22 | 22 | True | True |
| 0 | 5 | 350 | `TS_200` | 22 | 22 | True | True |
| 0 | 6 | 550 | `TS_40` | 22 | 22 | True | True |
| 0 | 8 | 590 | `TS_200` | 22 | 22 | True | True |
| 0 | 9 | 790 | `TS_30` | 22 | 22 | True | True |
| 0 | 11 | 820 | `TS_200` | 22 | 22 | True | True |

## Interpretation

The row-filtered repair is active on this diagnostic slice. Target time-shift rows remain finite, padded or non-target all -inf rows are no longer allowed to poison the auxiliary softmax, and the next required evidence is an enabled tiny training gate.
