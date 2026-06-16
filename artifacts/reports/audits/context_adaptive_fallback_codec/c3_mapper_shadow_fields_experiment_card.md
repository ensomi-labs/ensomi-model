# C3 Mapper Shadow Fields Experiment Card

## Hypothesis

The P0 C3 side-stream artifact can be exposed to the mapper data pipeline as optional shadow fields without changing default training samples or model inputs. If disabled-path behavior remains identical and enabled smoke samples carry bounded side-stream metadata, C3 is ready for a mapper-side probe card.

## Root Objective

Move from cache-level P0 tokenization to full-pipeline visibility by adding disabled-by-default mapper dataset shadow fields for C3 side-stream statistics/artifacts.

## Goal Decomposition

- Keep current mapper v2.1 dataset outputs unchanged by default.
- Add an explicit opt-in flag for C3 shadow metadata.
- Attach side-stream summary fields to mapper samples without feeding them to the model.
- Verify collation and training tests still pass with the flag disabled.
- Run a small enabled smoke check to prove sample-level metadata can be emitted.

## Candidate Variants

- P1a metadata-only shadow fields: attach counts/flags in `metadata`.
- P1b tensor shadow fields: attach compact side-stream tensors to samples.
- P1c external sidecar lookup: keep dataset samples unchanged but expose a sidecar artifact path/key.

## Local Verification Matrix

| candidate | local check | pass/fail interpretation |
|---|---|---|
| P1a | disabled mapper tests unchanged; enabled sample has C3 metadata | Best first step if metadata is enough for probe design. |
| P1b | collation supports tensors with padding/masks | Needed later, but premature if P1a is enough. |
| P1c | sidecar lookup works without loading huge state per sample | Useful if in-sample metadata is too heavy. |

## Selected Variant

Start with P1a metadata-only shadow fields.

## Selection Pressure

P0 showed combined main+side tokens are `1.566805x` baseline. Tensorizing the full side stream before knowing mapper-window ergonomics is premature. Metadata-only shadow fields expose the relevant structure without changing model input contracts.

## Minimal Change

Add disabled-by-default constructor args to the mapper v2.1 dataset for optional C3 side-stream metadata. The enabled path may compute lightweight per-beatmap/per-window summaries from an existing P0 report or sidecar summary, but must not alter default sample keys except optional metadata fields.

## Files Likely To Change

- `src/pulsefield_model/data/mapper_sparse_windows_v2_1.py`
- `tests/models/mapper/v2_1/test_data_windows.py`
- optionally `src/pulsefield_model/osu_core/c3_side_stream_tokenization.py`

## Dataset Slice

Use existing mapper v2.1 synthetic dataset tests first. Do not require the full LE<=3 cache for unit tests.

## Baseline / Comparator

Baseline is the current mapper v2.1 dataset output with no C3 args. Default sample keys, tensor shapes, metadata contract, and collation behavior must remain unchanged.

## Primary Metric

Disabled-path mapper dataset and training tests pass unchanged.

## Secondary Metric

Enabled sample exposes C3 metadata fields, optional metadata does not break collation, and no default model input tensors change.

## Verify Command Or Evaluation Procedure

```bash
uv run --group dev pytest tests/models/mapper/v2_1/test_data_windows.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/osu_core/test_c3_side_stream_tokenization.py tests/models/mapper/v2_1/test_data_windows.py -q
```

## Guard Check

- No default mapper sample key changes.
- No default tensor shape changes.
- No model forward/training code changes.
- Enabled C3 shadow fields stay in `metadata` or an explicitly named optional field.
- Tests must prove disabled and enabled dataset construction paths.

## Qualitative Check

Inspect one enabled sample metadata object. It should be obvious that C3 is a side-stream artifact and not yet a model input.

## Positive Signal

- Disabled-path tests pass.
- Enabled sample includes C3 shadow metadata.
- Collation still works.
- No mapper model/training defaults change.

## Negative Signal

- Default sample metadata changes unexpectedly.
- Enabled path requires full-cache global state in unit tests.
- Shadow fields are too vague to support a mapper probe.

## Kill Criteria

Stop if P1a requires invasive mapper dataset changes, changes default sample schemas, or needs full-cache C3 state just to instantiate ordinary mapper tests.

## Expected Failure Modes

- Mapper records do not carry a stable key for joining P0 side-stream summaries.
- Window-level side-stream state crosses sample boundaries and cannot be summarized locally.
- Metadata-only fields are insufficient for actual mapper probe design.

## Expected Runtime / Runtime Budget

Unit tests should finish in seconds.

## Confounders

P1a is not a training improvement and should not be interpreted as model evidence. It only exposes C3 to the pipeline safely.

## Result Interpretation Plan

- If P1a passes, design P1b/P2 mapper probe with tensors or sidecar lookup.
- If P1a fails due to missing join keys, add a sidecar-key audit before mapper code.
- If disabled path changes, revert and redesign.

## Result Log Template

Record changed constructor args, disabled-path tests, enabled sample metadata, collation behavior, and recommendation.

## Next-Loop Action

Implement P1a metadata-only shadow fields.
