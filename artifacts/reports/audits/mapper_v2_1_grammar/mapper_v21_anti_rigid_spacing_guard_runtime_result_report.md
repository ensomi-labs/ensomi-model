# Mapper v2.1 Anti-Rigid Spacing Guard Runtime Result Report

## Scope

This gate reruns the six matched v2.1 timing-baseline cases with the opt-in hard-block anti-rigid spacing transform. It does not change mapper defaults, model weights, tokenization, or training.

## Decision

Decision: `TEST_NEXT`.

- Reason: rigidity improved on at least two cases without legality, F1, or starvation guard failure
- Cases: `6`
- All legal: `True`
- Transform blocked count: `332`
- Rigid-improved cases: `6`
- Mean F1 delta: `0.011526`
- New starved cases: `0`
- Mean dominant-spacing ratio: `1.000000` -> `0.794255`

## Case Table

| Case | Prefix | Legal | Blocks | Rigid base->guard | F1 base->guard | 2nd share base->guard | Events base->guard | Dominant spacing |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sparse_1000722_asha_hd` | 8000 | True | 18 | 1.000000->0.800000 | 0.701299->0.753623 | 0.000000->0.000000 | 49->41 | `160` |
| `sparse_1000722_asha_hd` | 16000 | True | 50 | 1.000000->0.804878 | 0.692308->0.685714 | 0.494949->0.506024 | 99->83 | `160` |
| `moderate_1004416_ash_insane` | 8000 | True | 44 | 1.000000->0.800000 | 0.916667->0.901961 | 0.000000->0.000000 | 50->56 | `160` |
| `moderate_1004416_ash_insane` | 16000 | True | 88 | 1.000000->0.792793 | 0.881720->0.848485 | 0.490000->0.500000 | 100->112 | `160` |
| `dense_1008095_lone_sonorous` | 8000 | True | 44 | 1.000000->0.800000 | 0.596491->0.616667 | 0.000000->0.000000 | 50->56 | `160` |
| `dense_1008095_lone_sonorous` | 16000 | True | 88 | 1.000000->0.767857 | 0.680328->0.731518 | 0.490000->0.504425 | 100->113 | `160` |

## Interpretation

The hard-block guard produced enough runtime signal to justify a wider decode timing comparison, but it remains opt-in and is not a default grammar change.

## Next Step

Run a wider fixed-slice v2.1/v3 decode timing comparison before changing defaults.
