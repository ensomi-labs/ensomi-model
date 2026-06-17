# Target Grammar v3 Delta-Event Auxiliary Head Smoke Result Report

## Scope

This bounded smoke implements the next step after the delta-event proxy audit: a default-off v3 auxiliary head that predicts event delta, event signature, and terminal end-gap labels from existing decoder hidden states. It does not change tokenizer behavior, dataset schema, default v3 training behavior, or rollout.

## Decision

Route: `TEST_DELTA_EVENT_AUXILIARY_TINY_TRAINING_GATE`.

- Reason: default-off v3 delta-event auxiliary model/loss plumbing passed focused smoke checks with finite loss and nonzero gradients.
- Recommended next step: Create a bounded tiny-training auxiliary objective gate before replacing v3 target grammar or touching rollout.

## Guard Results

| Guard | Value |
| --- | ---: |
| experiment card exists before code | `True` |
| default-off auxiliary logits absent | `True` |
| enabled auxiliary logits shape valid | `True` |
| enabled auxiliary loss finite | `True` |
| event labels positive | `True` |
| end-gap labels positive | `True` |
| delta head gradient nonzero | `True` |
| signature head gradient nonzero | `True` |
| end-gap head gradient nonzero | `True` |
| shared decoder gradient nonzero | `True` |
| training config accepts v3 model/loss fields | `True` |
| focused tests pass | `True` |
| py_compile pass | `True` |

## Smoke Metrics

| Metric | Value |
| --- | ---: |
| synthetic sequence length | `8` |
| delta logits shape | `[1, 8, 801]` |
| signature logits shape | `[1, 8, 255]` |
| end-gap logits shape | `[1, 8, 801]` |
| event label count | `2` |
| end-gap label count | `1` |
| total loss | `8.434008` |
| delta-event auxiliary loss | `13.809816` |
| delta head grad abs | `2.958763` |
| signature head grad abs | `2.948094` |
| end-gap head grad abs | `6.343822` |
| token embedding grad abs | `1.620438` |

Target tokens in the smoke window:

```text
TS_80 EV_1000 TS_100 TS_60 EV_0100 TS_700 TS_60 EOS
```

## What Passed

- The auxiliary heads are default-off and absent from normal v3 forward output.
- When enabled, delta/event/end-gap logits have expected shapes.
- Teacher-forced labels can be derived from existing v3 target fragments and replay state.
- Auxiliary loss is finite and produces nonzero gradients for all new heads and the shared decoder path.
- Mapper v3 run config accepts the new model/loss knobs.

## What Surfaced

The current v3 sequence still includes standalone time-shift rows. This smoke attaches auxiliary labels only to event rows and a sample end row, so it proves plumbing feasibility but not final target grammar quality.

This is not trained rollout evidence. It does not prove learnability, generated-chart quality, full-dataset label coverage, or replacement readiness.

## Verification

```bash
uv run python -m py_compile src/pulsefield_model/models/mapper/v3/model.py src/pulsefield_model/models/mapper/v3/loss.py src/pulsefield_model/training/mapper_v3.py
uv run --group dev pytest tests/models/mapper/v3/test_model.py tests/training/test_mapper_v3.py -q
uv run --group dev pytest tests/models/mapper/v3 tests/evals/test_target_grammar_v3_delta_event_proxy_audit.py tests/training/test_mapper_v3.py -q
```

Result: focused tests passed, `27 passed in 0.90s`; broader mapper-v3/proxy checks passed, `40 passed in 0.83s`.

## Next Step

Create a bounded tiny-training auxiliary objective gate. Keep the auxiliary path default-off until it shows trainability beyond this smoke.
