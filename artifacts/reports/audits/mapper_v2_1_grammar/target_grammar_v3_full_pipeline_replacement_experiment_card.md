# Target Grammar v3 Full-Pipeline Replacement Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: v3 event-token grammar passed the full dataset audit and now needs integration into planner/mapper training and inference.
- Acceptance source, if any: active goal requires v3 final state before replacing planner/mapper full pipeline from training to inference.
- Source snapshot / evidence grade: strong representation evidence from full dataset audit; missing trained mapper and inference evidence.

## Hypothesis

Replacing v2.1 sparse lane-action targets with v3 event-token targets in the mapper pipeline will preserve train/inference legality while reducing target length and simplifying same-time replay state, without requiring future target tokens or C3-style cross-window references.

## Root Objective

Move v3 from audited representation to full-pipeline candidate:

- dataset emits v3 targets,
- mapper trains with v3 vocab and grammar mask,
- inference decodes v3 online,
- generated v3 tokens can reconstruct beatmap events.

## Goal Decomposition

- Subgoal 1: Add a v3 mapper dataset/collate path that mirrors v2.1 inputs while emitting v3 target tokens and v3 replay states.
- Subgoal 2: Add a v3 model/config/training entrypoint using v3 vocab size and v3 grammar/loss contract.
- Subgoal 3: Add v3 inference decode/expansion to beatmap events.
- Subgoal 4: Run a small train/inference smoke before any long training run.

## Candidate Variants

- Variant A: Add v3 as a parallel mapper contract, then switch training/inference configs after smoke passes.
- Variant B: Mutate v2.1 files in place.
- Variant C: Reuse old tuple mapper pipeline directly.
- Variant D: Keep v2.1 targets and use v3 only as an auxiliary side target.

## Local Verification Matrix

- Variant A: Best balance; enables comparison and rollback while moving toward replacement.
- Variant B: Rejected for first implementation because it risks breaking v2.1 before trained v3 evidence exists.
- Variant C: Rejected because old tuple path lacks the exact v2.1 chart-end/carry semantics now encoded in v3 wrappers.
- Variant D: Rejected because the goal requires replacing the target grammar, not adding another auxiliary.

## Selected Variant

- Selected: Variant A, parallel v3 mapper contract with smoke gates, then replacement config.

## Selection Pressure

- Must preserve exact v3 tokenizer/replay/grammar tests.
- Must produce batches with v3 vocab size and valid masks.
- Must run at least one small training step or loss computation with v3 targets.
- Must run one inference/decode smoke that expands v3 tokens back to beatmap events or v2.1-equivalent sparse events.

## Minimal Change

Implement the smallest v3 full-pipeline smoke:

- v3 dataset/collate wrapper,
- v3 model/training config path reusing existing mapper architecture where possible,
- v3 loss/grammar wiring,
- v3 inference expansion helper,
- smoke tests and one bounded train/loss command.

Do not remove v2.1 code in this experiment; replacement becomes a config/default decision after smoke evidence.

## Files Likely to Change

- `src/pulsefield_model/data/mapper_sparse_windows_v3.py`
- `src/pulsefield_model/models/mapper/v3/model.py`
- `src/pulsefield_model/models/mapper/v3/__init__.py`
- `src/pulsefield_model/training/mapper_v3.py`
- `src/pulsefield_model/training/mapper_common.py`
- `src/pulsefield_model/inference/session_runtime.py`
- `tests/models/mapper/v3/test_data_windows.py`
- `tests/models/mapper/v3/test_model.py`
- `tests/inference/test_mapper_v3_inference.py`
- v3 training config/artifact files under `artifacts/reports/audits/mapper_v2_1_grammar/`

## Dataset Slice

- Unit tests use synthetic 4K timepoints.
- Smoke train/loss uses the current mapper v2.1 config dataset with a small bounded batch count.

## Baseline / Comparator

- Current v2.1 mapper dataset/model/training/inference path.
- v3 representation audit results:
  - full-audit reconstruction mismatches: `0`,
  - full-audit token reduction: `20.54%`,
  - eval total-bit reduction: `5.43%`.

## Primary Metric

Successful v3 full-pipeline smoke:

- v3 batch creation passes,
- v3 forward/loss step passes,
- v3 inference/decode expansion passes.

## Secondary Metric

- v3 target sequence length versus v2.1 on the same sample batch.
- v3 grammar mask validity for gold targets.
- v3 generated/teacher-forced expansion mismatch count.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest \
  tests/models/mapper/v3/test_event_token_smoke.py \
  tests/models/mapper/v3/test_data_windows.py \
  tests/models/mapper/v3/test_model.py \
  tests/evals/test_target_grammar_v3_event_smoke.py \
  tests/models/mapper/v2_1/test_tokenizer_replay_grammar.py -q
```

Add a bounded smoke command after the training entrypoint exists.

## Guard Check

- Existing v2.1 tokenizer/replay/grammar tests must keep passing.
- Full-dataset v3 audit artifact must remain unchanged unless explicitly rerun.

## Qualitative Check

Inspect one decoded v3 sample with:

- chord event,
- LN start/end,
- terminal chart-end event,
- cross-window LN carry.

## Positive Signal

- All smoke tests pass.
- Gold v3 targets are legal under v3 grammar masks.
- One forward/loss smoke completes.
- One inference/decode smoke completes without future target dependency.

## Negative Signal

- v3 batch/model shape mismatch,
- grammar masks reject gold v3 targets,
- inference requires future target tokens,
- decoded v3 events do not match teacher-forced events.

## Kill Criteria

Kill immediate pipeline replacement if v3 cannot run a minimal forward/loss smoke without weakening legality checks or introducing future-target dependency.

## Expected Failure Modes

- Existing model/loss contracts may assume v2.1 sparse lane-action state fields.
- Inference runtime may need a v3-specific decode branch.
- Training config loaders may require explicit token-contract selection.
- Old tuple mapper code may be tempting to reuse but may not match v3 chart-end semantics.

## Expected Runtime / Runtime Budget

Unit/smoke tests should run under 2 minutes. Bounded train/loss smoke should run under 10 minutes.

## Confounders

- Passing smoke does not prove trained quality.
- Reusing architecture isolates grammar effects but does not optimize v3-specific model capacity.
- Full replacement needs a longer training run after smoke passes.

## Result Interpretation Plan

- Positive: run bounded v3 training, then compare mapper eval/inference outputs against v2.1.
- Negative: repair v3 integration or mutate representation only if the failure is representational.
- Ambiguous: add narrower diagnostics around the failing pipeline layer.

## Result Log Template

- Experiment: Target grammar v3 full-pipeline replacement smoke
- Date:
- Commit / run id:
- Dataset/config:
- Tests:
- Forward/loss smoke:
- Inference/decode smoke:
- Target length comparison:
- Grammar legality:
- Failed checks:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: run bounded v3 mapper training and inference comparison.
- If negative: repair the specific integration layer before training.
- If blocked by architecture assumptions: create a smaller adapter Experiment Card.

## Closest Analogies and Novelty Layer

- Closest analogies: existing v2.1 sparse mapper pipeline, older tuple/event mapper pipeline, teacher-forced event-token sequence models.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation unless trained v3 results demonstrate a new modeling advantage.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: final replacement requires trained mapper and inference evidence, not only smoke integration.
