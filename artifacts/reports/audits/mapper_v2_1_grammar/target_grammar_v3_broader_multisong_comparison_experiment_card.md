# Target Grammar v3 Broader Multi-Song Comparison Experiment Card

## Mode

- Mode: planner
- Route: TEST
- Source idea: the 16-row fixed-split comparison passed, but it used adjacent windows from a narrow local slice. The next replacement gate needs a broader, still cheap, multi-song comparison.
- Acceptance source, if any: active goal requires v3 mapper full pipeline evidence before replacement.
- Source snapshot / evidence grade: strong representation evidence; medium pipeline evidence; bounded trained comparison still narrow.

## Hypothesis

On a deterministic multi-song local slice, v3 and v2.1 will both train through repeated shared-runner steps, and v3 will retain a lower valid-token count in report metrics while keeping finite train/eval losses.

## Root Objective

Move from adjacent-window evidence toward a broader fixed-split trained comparison without starting a full 4K training run.

## Goal Decomposition

- Subgoal 1: Build a deterministic local index slice spanning multiple beatmaps.
- Subgoal 2: Run v2.1 and v3 with comparable tiny CPU configs and the same split/training settings.
- Subgoal 3: Compare reports with the existing report harness.
- Subgoal 4: Record contracts, completed steps, finite losses, and token-count ratio.
- Subgoal 5: Keep relevant mapper guard tests green.

## Candidate Variants

- Variant A: 4 beatmaps x 16 windows, 10 CPU steps, tiny d_model=16 configs.
- Variant B: 16 beatmaps x 16 windows, 50 CPU steps.
- Variant C: full 4K d384 run.
- Variant D: session-runtime route before broader trained comparison.

## Local Verification Matrix

- Variant A: Selected. It broadens beyond one adjacent slice while staying fast and reproducible.
- Variant B: Rejected for this turn because it is a larger runtime jump before A is measured.
- Variant C: Rejected for now because full 4K training should follow broader bounded evidence.
- Variant D: Rejected because trained evidence is still incomplete.

## Selected Variant

- Selected: Variant A, deterministic 4-song/64-row slice with 10 CPU steps per mapper.

## Selection Pressure

- Must use the same fixed slice for v2.1 and v3.
- Must include multiple beatmaps.
- Must validate expected report contracts.
- Must not claim final quality or switch defaults.

## Minimal Change

No production code change is required. Run the existing v2.1/v3 training functions and comparison harness; add result artifacts.

## Files Likely to Change

- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_broader_multisong_comparison_experiment_card.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_broader_multisong_comparison_result_report.md`
- `artifacts/reports/audits/mapper_v2_1_grammar/target_grammar_v3_broader_multisong_comparison_summary.json`

## Dataset Slice

- Deterministically select the first 4 beatmaps in the 4K index whose `.osu` files exist locally and that have at least 16 windows.
- Take the first 16 rows per selected beatmap.
- Local dataset root: `dataset`.
- Output root: `artifacts/tmp/mapper_v3_broader_multisong_comparison`.

## Baseline / Comparator

- v2.1 sparse lane-action mapper, tiny CPU config.
- v3 event-group mapper, same tiny CPU config.

## Primary Metric

Comparison harness route must be `TEST_NEXT`.

## Secondary Metric

- v3 eval valid-token ratio versus v2.1.
- completed steps for both reports.
- final eval loss/total for both reports.
- selected beatmap/window counts.

## Verify Command / Evaluation Procedure

Run paired training:

```bash
uv run python - <<'PY'
# Build deterministic 4-song/64-row local index and run v2.1/v3 for max_steps=10.
PY
```

Run comparison:

```bash
uv run python -m pulsefield_model.evals.mapper_v3_training_comparison \
  --v2-1-report artifacts/tmp/mapper_v3_broader_multisong_comparison/v21/run/report.json \
  --v3-report artifacts/tmp/mapper_v3_broader_multisong_comparison/v3/run/report.json \
  --summary-output artifacts/tmp/mapper_v3_broader_multisong_comparison/comparison_summary.json \
  --report-output artifacts/tmp/mapper_v3_broader_multisong_comparison/comparison_report.md \
  --min-completed-steps 10
```

Guard tests:

```bash
uv run --group dev pytest tests/evals/test_mapper_v3_training_comparison.py tests/training/test_mapper_v3.py tests/training/test_mapper_v2_1.py -q
uv run --group dev pytest tests/inference/test_mapper_v3_rollout.py tests/inference/test_mapper_v2_1_rollout.py tests/models/mapper/v3/test_model.py tests/models/mapper/v2_1/test_model.py -q
```

## Guard Check

- No code/default replacement in this experiment.
- No v3 resume claim.
- No quality claim from 10 CPU steps.
- v2.1 comparator remains unchanged.

## Qualitative Check

Inspect selected beatmaps and report text. Confirm the result is multi-song, not adjacent windows from a single beatmap.

## Positive Signal

- Both runs complete 10 steps.
- Both reports validate expected contracts.
- Both final eval losses are finite.
- v3 eval valid-token count remains lower than v2.1.

## Negative Signal

- Either run fails.
- v3 target-token count is not lower.
- selected slice cannot be built from local files.
- guard tests regress.

## Kill Criteria

Do not escalate to larger/full training if v3 loses the target-token reduction or the shared-runner comparison cannot validate contracts on the broader slice.

## Expected Failure Modes

- Some selected windows may exceed `max_seq_len=1024`.
- Local data may not contain enough matching beatmap files on other machines.
- Tiny CPU losses are not quality-comparable across vocabularies.

## Expected Runtime / Runtime Budget

Paired training, comparison, and guards should run under 15 minutes locally.

## Confounders

- Tiny configs are capacity-limited.
- The slice is broader but still small.
- Per-token loss differs across vocabularies and should not be used alone as a quality claim.

## Result Interpretation Plan

- Positive: proceed to a larger bounded run with more songs and/or real d384 cache-backed config.
- Negative: inspect token-count distribution and training/report layer before session-runtime work.
- Ambiguous: increase slice diversity before making a decision.

## Result Log Template

- Experiment: Target grammar v3 broader multi-song comparison
- Date:
- Commit / run id:
- Selected beatmaps:
- Slice rows:
- Steps:
- v2.1 report:
- v3 report:
- Comparison route:
- Completed steps:
- Final eval losses:
- Valid-token counts:
- Token ratio:
- Guard tests:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Next-Loop Action

- If positive: run a larger bounded or real-config comparison.
- If negative: diagnose representation/training mismatch before broader runs.

## Closest Analogies and Novelty Layer

- Closest analogies: shared-runner report comparisons and v2.1 Phase B training smoke.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering validation for audited v3 representation.

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: final quality still needs larger and eventually real-config training.
