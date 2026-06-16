# Target Grammar v3 Trained Runtime Rollout Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: after the real-config cache-backed v3 training gate, verify that a trained v3 checkpoint can be loaded by the inference runtime and used for online full-song rollout.
- Acceptance source, if any: active goal requires v3 to move from audited target grammar into the full mapper pipeline from training to inference.
- Source snapshot / evidence grade: strong representation evidence, moderate real-config training evidence, weak runtime/checkpoint-load evidence.

## Hypothesis

If the inference runtime can detect and reconstruct v3 mapper checkpoints, then an already trained real-config v3 checkpoint can be loaded without embedded control-encoder weights and used by the v3 online full-song rollout path with no future target dependency.

## Root Objective

Bridge v3 from training artifact to inference artifact:

- load v3 mapper checkpoint through `ModelRuntime`,
- preserve separate control runtime loading,
- expose a `MapperV3Vocab`,
- run `generate_full_song_rollout_v3(...)` with a runtime-backed window provider,
- verify the rollout terminates legally and can export beatmap timepoints.

## Goal Decomposition

- Subgoal 1: Add v3 checkpoint detection and model reconstruction to the inference runtime.
- Subgoal 2: Add a unit test that proves v3 runtime loading filters embedded `control_encoder.*` keys and does not read optimizer/history state.
- Subgoal 3: Add a bounded trained-checkpoint rollout eval using the existing real-config v3 checkpoint and session/window provider path.
- Subgoal 4: Record result metrics and keep existing v2/v2.1/v3 guards green.

## Candidate Variants

- Variant A: Extend `ModelRuntime` to support v3 checkpoints and add a small eval script that loads a trained checkpoint and runs v3 full-song rollout.
- Variant B: Keep runtime untouched and write a standalone checkpoint loader inside the eval script.
- Variant C: Start session-runtime replacement directly by switching the default stream path to v3.
- Variant D: Use zero-control v3 rollout only, without loading a trained checkpoint through runtime.

## Local Verification Matrix

- Variant A: Tests the actual inference-runtime boundary needed for replacement while keeping default behavior unchanged.
- Variant B: Faster, but bypasses the runtime layer that must eventually be replaced.
- Variant C: Too broad because v3 quality and full-song runtime evidence are not mature enough.
- Variant D: Already mostly covered by existing inference smoke; it does not verify trained checkpoint loading.

## Selected Variant

- Selected: Variant A.

## Selection Pressure

Variant A is selected because it targets the next real blocker: trained v3 checkpoints are produced, but runtime loading still recognizes only v2/v2.1. Adding v3 support behind detection is small, testable, and directly advances the replacement path without changing defaults.

## Minimal Change

- Add v3 detection to `src/pulsefield_model/inference/model_runtime.py`.
- Instantiate `MapperV3Config`, `MapperV3Model`, and `MapperV3Vocab` when a checkpoint is identified as v3.
- Keep filtering embedded `control_encoder.*` state keys.
- Add unit coverage in `tests/inference/test_model_runtime.py`.
- Add a bounded eval script under `src/pulsefield_model/evals/` if no existing script can load a v3 checkpoint and run runtime-backed rollout.

## Files Likely to Change

- `src/pulsefield_model/inference/model_runtime.py`
- `tests/inference/test_model_runtime.py`
- `src/pulsefield_model/evals/mapper_v3_trained_runtime_rollout.py`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trained_runtime_rollout_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_trained_runtime_rollout_summary.json`

Read-only context:

- `src/pulsefield_model/inference/mapper_v3_rollout.py`
- `src/pulsefield_model/inference/session_runtime.py`
- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/v3/run/checkpoint.pt`
- `artifacts/tmp/mapper_v3_real_config_cache_backed_comparison/fixed_32song_256_eligible_index.parquet`

## Dataset Slice

- Use one short selected beatmap from the real-config comparison slice.
- Use the local audio/control path through `SessionRuntime.prepare_audio(...)` and `prepare_full_control(...)` if feasible.
- Runtime budget may cap chart length or windows for smoke, but rollout must use v3 online window sequencing and not teacher-force future target tokens.

## Baseline / Comparator

- Baseline: current runtime loader supports v2/v2.1 only; v3 checkpoint inference is not represented in `ModelRuntime`.
- Comparator evidence: real-config v3 checkpoint from the prior gate:
  - d384/l4 mapper,
  - `3` MPS optimizer steps,
  - cache-backed training,
  - v3 target token reduction `18.49%` on the training comparison slice.

## Primary Metric

- `ModelRuntime.load(...)` reconstructs a v3 mapper checkpoint as `MapperV3Model` with `MapperV3Vocab`, loads state strictly after filtering embedded control-encoder keys, and reports mapper metadata version `v3`.

## Secondary Metric

- Trained v3 rollout smoke completes at least one full-song or bounded full-song-prefix rollout window through `generate_full_song_rollout_v3(...)`.
- Generated output can be converted to v3 timepoints and v2.1-equivalent tokens without replay errors.
- Rollout reports:
  - completed/dead-end/max-token status,
  - window count,
  - generated token count,
  - event timepoint count.

## Verify Command or Evaluation Procedure

1. Run focused runtime tests:

```bash
uv run --group dev pytest tests/inference/test_model_runtime.py tests/inference/test_mapper_v3_rollout.py -q
```

2. Run the trained runtime rollout eval against the existing real-config v3 checkpoint.
3. Run existing guards:

```bash
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
```

## Guard Check

- Existing v2/v2.1 runtime model loading tests must keep passing.
- v3 training/model/inference tests must keep passing.
- Runtime loader must not read optimizer state, history, or training RNG state for inference.

## Qualitative Check

Inspect the eval summary for:

- mapper version `v3`,
- filtered embedded control-encoder key count > 0,
- finite generated token/timepoint counts,
- no `dead_end`,
- no `max_tokens_exceeded`,
- no use of teacher-forced future target tokens.

## Positive Signal

- Runtime loader detects v3 and loads it strictly.
- Unit tests pass.
- Trained v3 checkpoint can execute at least one runtime-backed rollout.
- Rollout exports generated timepoints or explicitly completes an empty-event legal chart prefix without replay errors.

## Negative Signal

- v3 checkpoint is misdetected as v2/v2.1,
- strict state load fails after filtering control encoder keys,
- session/runtime provider cannot supply v3-compatible control batches,
- rollout dead-ends or hits max-token limits immediately,
- existing v2/v2.1 runtime tests regress.

## Kill Criteria

Kill immediate runtime replacement if v3 cannot be loaded by the inference runtime or cannot run a bounded online rollout without weakening legality checks.

## Expected Failure Modes

- Checkpoint lacks an explicit `model_version`, requiring detection by run name or vocab size.
- Runtime type annotations and metadata may assume only v2/v2.1 vocab classes.
- Session window batches may miss fields needed by v3 global context.
- A 3-step checkpoint may produce poor samples; that is not a loader failure unless it causes grammar dead-end/max-token failure.

## Expected Runtime / Runtime Budget

- Unit tests: under 2 minutes.
- Rollout eval: target under 10 minutes.
- Stop condition: loader failure, strict state-load failure, rollout hard failure, or guard regression.

## Confounders

- The checkpoint is only 3 optimizer steps, so generated chart quality is not meaningful.
- A successful short rollout does not prove full-song quality or convergence.
- Runtime loader support does not mean defaults should switch to v3 yet.

## Result Interpretation Plan

- Positive: proceed to longer v3 training and session-runtime replacement gate.
- Negative loader result: repair `ModelRuntime` support before more training.
- Negative rollout result with successful loader: isolate session provider fields or generation policy before replacement.
- Ambiguous poor samples: do not interpret as quality until longer training exists.

## Result Log Template

- Experiment: Target grammar v3 trained runtime rollout
- Date:
- Commit / run id:
- Runtime loader test:
- Checkpoint:
- Mapper version:
- Filtered control keys:
- Rollout command:
- Beatmap / audio:
- Chart/window scope:
- Generated windows:
- Generated tokens:
- Generated timepoints:
- Completed / dead-end / max-token status:
- Guards:
- Failed checks:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: run a longer real-config v3 training checkpoint and a fuller v3 session-runtime inference comparison.
- If negative: repair the runtime loader or session provider layer before longer v3 training.

## Closest Analogies and Novelty Layer

- Closest analogies: existing v2/v2.1 `ModelRuntime` checkpoint loading, v2.1 online rollout, v3 grammar-constrained online generation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering integration of the audited v3 representation into the inference runtime.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: generated quality and default replacement remain unproven.
