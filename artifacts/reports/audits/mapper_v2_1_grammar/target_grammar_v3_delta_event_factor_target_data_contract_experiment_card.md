# Target Grammar v3 Delta-Event Factor Target Data Contract Experiment Card

## Hypothesis

The factorized delta-event target can be exposed through the production v3 dataset/collate contract as default-off optional batch fields without changing current v3 token-stream behavior: rows are shifted for teacher forcing, labels reconstruct current v3 events and terminal target end, real fixed-slice samples collate with masks and control-teacher fields, and default-off batches remain byte-for-byte contract-compatible at the key level.

## Root Objective

Move the factorized v3 target from eval-only evidence toward full pipeline replacement. Full4k representation audits proved reversibility and lower bits; the tiny model gate proved trainability. This card tests the next necessary production boundary: data/batch plumbing that future model/loss/training code can consume.

## Goal Decomposition

- Subgoal 1: Move factorized-row construction into production v3 code, not eval-only code.
- Subgoal 2: Add an optional dataset flag for factorized target fields with default-off behavior.
- Subgoal 3: Collate factorized row inputs, labels, masks, and metadata only when present.
- Subgoal 4: Verify real fixed-slice batches reproduce the factorized row counts and reconstruction guarantees from the tiny model gate.
- Subgoal 5: Verify existing default v3 dataset/collate tests and model tests still pass.

## Candidate Variants

- Variant A: Add default-off optional factor-target fields to `MapperV3WindowDataset` and `collate_mapper_v3_windows`, backed by a production `models.mapper.v3.factor_target` utility module.
- Variant B: Keep factorized rows only inside eval modules until the full model is ready.
- Variant C: Immediately replace `target_fragment_tokens` with factorized rows in the v3 dataset.
- Variant D: Add production model/loss/training runner integration in the same change.

## Local Verification Matrix

| Variant | Pass evidence | Fail evidence |
| --- | --- | --- |
| A | Default-off key set unchanged; default-on real batch has valid factor rows, masks, labels, reconstruction counts, and control teacher fields | Default-off regression, missing labels, mask mismatch, reconstruction mismatch |
| B | No production risk | Does not move toward full pipeline replacement |
| C | Direct replacement path | Too much blast radius before model/loss runner contract is validated |
| D | Faster apparent progress | Conflates data contract, model architecture, loss, and training runner failures |

## Selected Variant

Variant A: default-off production data contract for factorized delta-event target rows.

## Selection Pressure

Variant A is the smallest production-facing step after the tiny model gate. It makes future model/loss integration possible while preserving current v3 behavior. Variant B stalls at eval-only evidence. Variants C and D are rejected as too broad for the current gate.

## Minimal Change

- Add `src/pulsefield_model/models/mapper/v3/factor_target.py` with factorized row construction and tensor helpers.
- Add `include_delta_event_factor_target: bool = False` to `MapperV3WindowDataset`.
- When enabled, emit a `delta_event_factor_target` sample mapping with shifted inputs, labels, masks, row counts, and reconstruction counters.
- Extend `collate_mapper_v3_windows` to collate these fields when every sample includes them.
- Add a bounded evaluator that compares default-off/default-on contracts on the fixed slice.
- Add focused unit tests.

## Files Likely To Change

- `src/pulsefield_model/models/mapper/v3/factor_target.py`
- `src/pulsefield_model/data/mapper_sparse_windows_v3.py`
- `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_data_contract.py`
- `tests/models/mapper/v3/test_factor_target.py`
- `tests/data/test_mapper_sparse_windows_v3_factor_target.py`
- `tests/evals/test_mapper_v3_delta_event_factor_target_data_contract.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_data_contract_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_data_contract_result_report.md`

## Read-Only Context

- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`
- `artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_tiny_model_summary.json`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_full4k_bit_proxy_summary.json`
- `src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_tiny_model.py`

## Dataset Slice

Fixed 32-song / 256-window v3 comparison index:

`artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

The default contract gate audits the first `32` real windows and collates them with batch size `4`.

## Baseline / Comparator

- Factor-target tiny model route: `TEST_DELTA_EVENT_FACTOR_TARGET_PRODUCTION_PLUMBING_CARD`.
- Tiny model fixed-slice rows: `1479` rows, `1447` event rows, `32` end rows, `0` reconstruction mismatches.
- Full4k bit proxy route: `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_CARD`.
- Current v3 dataset/collate default contract without factorized fields.

## Primary Metric

Default-off contract preservation and default-on factor target reconstruction:

- default-off batch does not include `delta_event_factor_target`
- default-on audited row counts match expected fixed-slice counts
- reconstruction mismatch count is `0`

## Secondary Metrics

- event row count
- end row count
- max factor row length
- row mask valid count
- label counts for kind, delta, signature, and end-gap
- collated tensor shapes

## Verify Command Or Evaluation Procedure

```bash
uv run python -m pulsefield_model.evals.mapper_v3_delta_event_factor_target_data_contract
uv run --group dev pytest tests/models/mapper/v3/test_factor_target.py tests/data/test_mapper_sparse_windows_v3_factor_target.py tests/evals/test_mapper_v3_delta_event_factor_target_data_contract.py tests/models/mapper/v3/test_model.py -q
uv run python -m py_compile src/pulsefield_model/models/mapper/v3/factor_target.py src/pulsefield_model/evals/mapper_v3_delta_event_factor_target_data_contract.py
python3 -m json.tool artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_delta_event_factor_target_data_contract_summary.json >/dev/null
git diff --check
```

## Guard Check

- `include_delta_event_factor_target` defaults to false.
- Existing `mapper_token_contract` remains `v3_event_groups`.
- Existing `target_fragment_tokens` and masks remain present.
- No production model/loss/training runner/inference default change.
- No C3 backreference or future lookup.

## Qualitative Check

The report must state this is a data-contract gate only. It must not claim model integration, full training, rollout quality, or replacement readiness.

## Positive Signal

Default-off behavior is preserved; default-on fixed-slice batches carry complete factorized rows and labels; row counts match the tiny model gate; reconstruction mismatches are zero.

## Negative Signal

Default-off key regression, missing factor labels, mask/count mismatch, reconstruction mismatch, or incompatible collate behavior.

## Kill Criteria

- Any factorized row reconstruction mismatch.
- Default-off v3 dataset/collate changes observable batch keys.
- Real fixed-slice row counts diverge from the accepted tiny model gate without explanation.

## Expected Failure Modes

- Collate path may mishandle nested factor target tensors.
- End-row labels may be placed off by one.
- Default-off metadata/cache validity may need an explicit factor-target flag to avoid mixed caches.

## Expected Runtime / Runtime Budget

Expected runtime: under 2 minutes with fixed-slice cached control teacher tensors.

Stop condition: fail on default-off regression, missing fixed-slice cache, or reconstruction/count mismatch.

## Confounders

- This does not train a production model.
- This does not implement inference or autoregressive row generation.
- This does not prove full4k training stability.
- It keeps current v3 token-stream fields present for compatibility.

## Result Interpretation Plan

- If pass: route to `TEST_DELTA_EVENT_FACTOR_TARGET_MODEL_LOSS_PLUMBING_CARD`.
- If default-off regresses: route to `KILL_FACTOR_TARGET_DATA_CONTRACT_DEFAULT_REGRESSION`.
- If factor rows fail: route to `MUTATE_FACTOR_TARGET_DATA_CONTRACT`.

## Result Log Template

```text
route:
audited_windows:
default_off_has_factor_target:
row_count:
event_row_count:
end_row_count:
max_row_len:
reconstruction_mismatch_count:
label_counts:
checks:
next_step:
```

## Next-Loop Action

Use the result to decide whether to add production model/loss plumbing for factorized rows or repair the data contract before touching the model/training runner.

## Closest Analogies And Novelty Layer

Closest analogies are optional dataset targets, multi-head autoregressive training contracts, and staged target-representation migrations. There is no novelty claim here. The tested layer is production data plumbing for an already-audited representation.
