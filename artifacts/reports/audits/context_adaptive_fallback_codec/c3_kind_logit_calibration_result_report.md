# C3 Kind-Logit Calibration Result Report

## Scope

This P21 pass tests whether P20's RAW gap can be explained by post-hoc kind-logit calibration. It reuses the P19 checkpoint and reduced sidecar without retraining.

## Result

Decision: MUTATE_TO_RAW_SPLIT.

- candidate transforms: `284`
- calibration samples: `71`
- held-out samples: `71`
- selected transform: `scale_bias_rs1.25_fs1.00_ss1.25_rb+0.5_sb+0.5`
- held-out identity hits@20: `68` / `883`
- held-out selected hits@20: `64` / `883`
- held-out unconstrained-calibration hits@20: `78` / `883`
- held-out unigram hits@20: `85` / `883`
- held-out oracle-calibration hits@20: `81` / `883`
- held-out identity RAW recall@20: `0.091139`
- held-out selected RAW recall@20: `0.096203`
- held-out unigram RAW recall@20: `0.156962`
- RAW gap closure: `0.076923`
- non-RAW preservation ratio: `0.812500`
- elapsed: `14.93s`

## Held-Out Comparison

| System | Hits | Recall | RAW recall | REF recall | RES recall | Pred RAW | Pred REF | Pred RES |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 68 | 0.077010 | 0.091139 | 0.069231 | 0.064246 | 581 | 167 | 632 |
| selected | 64 | 0.072480 | 0.096203 | 0.023077 | 0.064246 | 649 | 41 | 690 |
| unconstrained calibration | 78 | 0.088335 | 0.192405 | 0.015385 | 0.000000 | 1354 | 26 | 0 |
| unigram | 85 | 0.096263 | 0.156962 | 0.076923 | 0.036313 | 828 | 138 | 414 |
| oracle heldout | 81 | 0.091733 | 0.179747 | 0.061538 | 0.005587 | 1274 | 78 | 28 |

## Selected Transform

- family: `scale_bias`
- kind biases: `{'RAW': 0.5, 'REF': 0.0, 'RES': 0.5}`
- kind scales: `{'RAW': 1.25, 'REF': 1.0, 'RES': 1.25}`
- selection policy: `best calibration hits with non-RAW preservation >= 0.90 versus identity`

## Unconstrained Diagnostic Transform

- transform: `scale_bias_rs1.25_fs0.75_ss0.75_rb+0.5_sb+0.0`
- family: `scale_bias`
- kind biases: `{'RAW': 0.5, 'REF': 0.0, 'RES': 0.0}`
- kind scales: `{'RAW': 1.25, 'REF': 0.75, 'RES': 0.75}`

## Top Calibration Candidates

| Candidate | Family | Calibration hits | Calibration RAW recall |
| --- | --- | ---: | ---: |
| `scale_bias_rs1.25_fs0.75_ss0.75_rb+0.5_sb+0.0` | scale_bias | 85 | 0.176471 |
| `scale_bias_rs1.25_fs0.75_ss1.00_rb+0.5_sb-0.5` | scale_bias | 85 | 0.176471 |
| `scale_bias_rs1.25_fs0.75_ss0.75_rb+0.5_sb-0.5` | scale_bias | 85 | 0.176471 |
| `scale_bias_rs1.25_fs0.75_ss1.25_rb+0.5_sb-0.5` | scale_bias | 85 | 0.176471 |
| `scale_bias_rs0.75_fs1.00_ss1.00_rb+1.0_sb+0.0` | scale_bias | 85 | 0.174292 |
| `scale_bias_rs0.75_fs1.00_ss0.75_rb+1.0_sb+0.0` | scale_bias | 85 | 0.174292 |
| `scale_bias_rs0.75_fs1.00_ss1.00_rb+1.0_sb-0.5` | scale_bias | 85 | 0.174292 |
| `scale_bias_rs0.75_fs1.00_ss0.75_rb+1.0_sb-0.5` | scale_bias | 85 | 0.174292 |
| `scale_bias_rs0.75_fs1.00_ss1.25_rb+1.0_sb-0.5` | scale_bias | 85 | 0.174292 |
| `scale_bias_rs0.75_fs1.00_ss0.75_rb+1.0_sb+0.5` | scale_bias | 84 | 0.169935 |

## Top Held-Out Candidates

| Candidate | Family | Held-out hits | Held-out RAW recall |
| --- | --- | ---: | ---: |
| `scale_bias_rs1.25_fs0.75_ss1.00_rb+0.0_sb-0.5` | scale_bias | 81 | 0.179747 |
| `scale_bias_rs1.25_fs1.25_ss1.00_rb+0.5_sb+0.0` | scale_bias | 81 | 0.179747 |
| `scale_bias_rs1.25_fs0.75_ss1.00_rb+0.5_sb+0.0` | scale_bias | 80 | 0.189873 |
| `scale_bias_rs1.25_fs1.00_ss1.00_rb+0.5_sb+0.0` | scale_bias | 80 | 0.182278 |
| `scale_bias_rs1.00_fs1.25_ss0.75_rb+0.5_sb+0.0` | scale_bias | 80 | 0.182278 |
| `scale_bias_rs0.75_fs1.00_ss1.00_rb+0.5_sb-0.5` | scale_bias | 80 | 0.182278 |
| `scale_bias_rs1.00_fs1.25_ss1.00_rb+0.5_sb-0.5` | scale_bias | 80 | 0.182278 |
| `scale_bias_rs1.25_fs0.75_ss0.75_rb+0.0_sb-0.5` | scale_bias | 80 | 0.182278 |
| `scale_bias_rs1.25_fs1.25_ss0.75_rb+0.5_sb+0.0` | scale_bias | 80 | 0.182278 |
| `scale_bias_rs0.75_fs1.00_ss0.75_rb+0.5_sb-0.5` | scale_bias | 80 | 0.182278 |

## What Passed

- The P19 checkpoint, config, reduced sidecar, and eval split are loadable.
- Calibration is post-hoc only; no target-derived C3 labels are used as model inputs.
- Identity, selected calibration, held-out oracle, and reduced unigram are all reported separately.

## What Surfaced

Unconstrained calibration can buy held-out hits only by overpromoting RAW and damaging non-RAW recovery. That is not a viable calibration path for the full C3 target; the next mutation should factor RAW rather than globally bias kind logits.

## Next Step

Create a RAW split/factorization Experiment Card based on the P20 field buckets.
