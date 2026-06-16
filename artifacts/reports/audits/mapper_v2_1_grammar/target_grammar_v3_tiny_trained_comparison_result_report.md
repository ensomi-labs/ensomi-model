# Target Grammar v3 Tiny Trained Comparison Result Report

## Scope

This report compares matched tiny v2.1 and v3 mapper training reports produced by the shared mapper runner. It is a comparability and target-length gate, not a mapper-quality result.

## Result

Decision: `TEST_NEXT`.

- Reason: all tiny trained-comparison gates passed
- v2.1 report: `artifacts/tmp/mapper_v3_tiny_trained_comparison/v21/run/report.json`
- v3 report: `artifacts/tmp/mapper_v3_tiny_trained_comparison/v3/run/report.json`
- v2.1 completed steps: `1`
- v3 completed steps: `1`
- v2.1 final eval loss: `3.557616`
- v3 final eval loss: `4.679502`
- v3 eval loss delta: `1.121886`
- v2.1 eval valid tokens: `138.000000`
- v3 eval valid tokens: `125.000000`
- v3 valid-token ratio: `0.905797`
- v3 valid-token reduction: `9.42%`

## Checks

| Check | Passed |
| --- | ---: |
| `v21_contract` | `True` |
| `v21_dataset_contract` | `True` |
| `v3_contract` | `True` |
| `v3_dataset_contract` | `True` |
| `v21_completed_steps` | `True` |
| `v3_completed_steps` | `True` |
| `v21_complete_flag` | `True` |
| `v3_complete_flag` | `True` |
| `v21_eval_loss_finite` | `True` |
| `v3_eval_loss_finite` | `True` |
| `v21_eval_valid_tokens_positive` | `True` |
| `v3_eval_valid_tokens_positive` | `True` |
| `v3_eval_valid_tokens_lower` | `True` |

## Interpretation

The tiny paired comparison passed: both reports use the expected contracts, both completed, loss/token metrics are finite, and v3 preserves a lower valid-token count. This supports a longer bounded trained comparison, but not replacement yet.

## What This Does Not Prove

- It does not prove trained v3 mapper quality.
- It does not prove convergence on the full 4K dataset.
- It does not justify session-runtime/default replacement by itself.

## Next Step

Run a longer bounded v3-vs-v2.1 trained comparison with fixed split and comparable compute.
