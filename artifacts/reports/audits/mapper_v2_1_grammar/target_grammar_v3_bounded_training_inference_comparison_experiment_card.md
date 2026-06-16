# Target Grammar v3 Bounded Training/Inference Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: v3 passed full-dataset representation, dataset/model/loss, incremental decode, and inference-helper smoke gates. The remaining replacement blocker is trained mapper evidence.
- Acceptance source, if any: active goal requires v3 planner/mapper full pipeline from training to inference before default replacement.
- Source snapshot / evidence grade: strong representation evidence; medium integration evidence; missing trained-quality evidence.

## Hypothesis

A parallel v3 Phase B mapper training entrypoint can train and evaluate with the same control-teacher/full-song context surfaces as v2.1, produce a valid checkpoint/report, and feed the v3 online inference helper without future target dependency.

## Root Objective

Move v3 from smoke integration to a bounded trained pipeline candidate:

- train v3 on a small bounded mapper slice,
- evaluate v3 loss metrics with the shared runner,
- prove the checkpoint/report contract records `v3_event_groups`,
- run a trained/inference-compatible decode smoke path,
- keep v2.1 unchanged as the comparator and rollback path.

## Goal Decomposition

- Subgoal 1: Add a v3 training function/CLI that mirrors v2.1 where the contracts are identical.
- Subgoal 2: Preserve v3-specific dataset, vocab, grammar, and loss wiring.
- Subgoal 3: Run a minimal training/report smoke under bounded runtime.
- Subgoal 4: Keep v2/v2.1 compatibility tests green.
- Subgoal 5: Defer default replacement until trained v3 quality is compared against v2.1.

## Candidate Variants

- Variant A: Add a parallel `mapper_v3` training CLI/function using the shared mapper runner and v3 dataset/collate path.
- Variant B: Mutate the v2.1 CLI into a token-contract switch.
- Variant C: Run only unit-level batch-loss tests without a real runner/checkpoint smoke.
- Variant D: Jump directly to session-runtime default replacement.

## Local Verification Matrix

- Variant A: Best balance; proves the missing full-pipeline training surface while preserving v2.1.
- Variant B: Rejected for this pass because trained v3 evidence does not yet justify merging runtime flags into the stable v2.1 CLI.
- Variant C: Rejected because it cannot prove checkpoint/report/training-run behavior.
- Variant D: Rejected because the trained-quality gate is still missing.

## Selected Variant

- Selected: Variant A, a parallel v3 training CLI/function and bounded smoke report.

## Selection Pressure

- Must reuse the shared mapper runner rather than adding a bespoke training loop.
- Must emit reports/checkpoints with `mapper_token_contract: v3_event_groups`.
- Must keep v2.1 tests passing.
- Must not implement resume or default replacement in this experiment.

## Minimal Change

Add the smallest v3 training surface:

- config loading for v3 model/loss/control sections,
- `run_mapper_v3_phase_b_training(...)`,
- dataset/split/loader construction using `MapperV3WindowDataset`,
- CLI mirroring the v2.1 options that are legal for v3,
- focused tests that mock the runner and validate config forwarding/report contract.

## Files Likely to Change

- `src/pulsefield_model/training/mapper_v3.py`
- `tests/training/test_mapper_v3.py`
- `configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_bounded_training_inference_comparison_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_bounded_training_inference_comparison_result_report.md`

## Dataset Slice

- Unit tests use synthetic/mocked training calls.
- Bounded smoke should use current mapper config paths with `max_steps <= 1`, `batch_size <= 2`, and small eval/final-train-eval sizes if local data/cache availability permits.

## Baseline / Comparator

- Existing v2.1 Phase B training CLI/function and mapper tests.
- Existing v3 smoke evidence:
  - full-dataset reconstruction mismatches: `0`,
  - full-audit token reduction: `20.54%`,
  - inference smoke: `5 passed`,
  - v3 guard: `23 passed`,
  - v2/v2.1 guard: `41 passed`.

## Primary Metric

The v3 training surface can be invoked through a public function/CLI and produces a shared-runner training report contract with `mapper_token_contract: v3_event_groups`.

## Secondary Metric

- v3 batch/loss smoke still passes.
- v3 inference helper tests still pass.
- v2.1 training CLI tests still pass.

## Verify Command / Evaluation Procedure

```bash
uv run --group dev pytest tests/training/test_mapper_v3.py tests/models/mapper/v3/test_model.py tests/inference/test_mapper_v3_rollout.py -q
uv run --group dev pytest tests/training/test_mapper_v2_1.py tests/models/mapper/v2_1/test_model.py tests/inference/test_mapper_v2_1_rollout.py -q
```

If local dataset/cache is available, additionally run:

```bash
uv run python -m pulsefield_model.training.mapper_v3 --config configs/training/stage2_mapper_v3_phase_b_event_global_mps.yaml --max-steps 1 --eval-every 1
```

## Guard Check

- No v2.1 CLI defaults or model semantics may change.
- v3 resume remains explicitly unsupported unless a separate card defines compatibility.
- The training contract must remain `v3_event_groups`.
- No session-runtime default switch in this experiment.

## Qualitative Check

Inspect the generated or mocked training config/report and confirm:

- phase is `B`,
- mapper token contract is `v3_event_groups`,
- dataset report names train/eval/source window counts,
- model config uses v3 vocab-compatible shape.

## Positive Signal

- v3 training CLI/function tests pass.
- Shared runner receives v3 model/loss config and v3 dataset/collate path.
- v3 inference and model smoke tests still pass.
- v2.1 compatibility tests still pass.

## Negative Signal

- v3 cannot construct loaders from the existing mapper dataset surfaces.
- shared runner assumes v2.1-only fields not present in v3.
- checkpoint/report contract cannot distinguish v3 from v2.1.
- v2.1 tests regress.

## Kill Criteria

Kill immediate training-surface promotion if a minimal shared-runner invocation requires weakening v3 grammar/replay legality, introducing v2.1 sparse state into v3, or changing v2.1 behavior.

## Expected Failure Modes

- v3 config loader may miss fields available in v2.1.
- cached control-teacher paths may need a v3-specific output root.
- local full dataset/cache may be unavailable in this environment, limiting execution to mocked runner/unit smoke.
- real trained quality may still underperform v2.1 after the runner is wired.

## Expected Runtime / Runtime Budget

Unit tests should run under 2 minutes. Optional one-step local training smoke should run under 10 minutes if caches are present.

## Confounders

- Passing this experiment proves trainability and report plumbing, not mapper quality.
- A one-step run cannot compare convergence.
- Mocked runner tests prove call contract, not dataset/cache availability.

## Result Interpretation Plan

- Positive: run bounded real v3 training and compare eval/inference metrics against v2.1.
- Negative: repair the training integration layer before changing representation or runtime defaults.
- Ambiguous: add a narrower diagnostic around the failing dataset, runner, or checkpoint layer.

## Result Log Template

- Experiment: Target grammar v3 bounded training/inference comparison
- Date:
- Commit / run id:
- Config:
- Tests:
- One-step training smoke:
- Report/checkpoint:
- Mapper token contract:
- Inference smoke:
- v2.1 guard:
- Failed checks:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: run a bounded v3-vs-v2.1 trained comparison with real checkpoints.
- If negative: fix the smallest failing training layer.
- If blocked by missing data/cache: record the missing artifact and keep the code-level runner smoke as partial evidence.

## Closest Analogies and Novelty Layer

- Closest analogies: v2.1 Phase B sparse mapper training, shared mapper runner, event-token sequence training.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering integration for the already audited v3 representation.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: trained v3 quality still requires a real bounded comparison after this runner exists.
