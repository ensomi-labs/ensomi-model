# Handoff, 2026-10-07 closeout (main session b87b7677)

For the next main session. Start with `RESEARCH.md`, then this note, then
[r2-phasec-frontier-20261007](r2-phasec-frontier-20261007.md) (phase-C redesign, decisions, code audit, costs) and
[o-phasen-rss-cause](r2-phasen-and-lens-20261006.md#o-phasen-rss-cause) (what happened to phase N today). Human
inputs of this session are in the private, local file
`private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md` (prompts 1-7, answers 1-2, and both of Astra's
answers verbatim); another clone does not have it. This session ran from the workspace root `~/ensomi`, so relay
hooks did not fire in it.

## Running on the mac at closeout (06:05 UTC)

**Phase-N run `r2-phaseN-20261006`.**
- Resumed 03:41 UTC under ens job `20261007-034113-r2-phaseN-resume` (tmux session of the same name), from
  `ckpt-0041380351-safe.pt`, after the supervisor had stopped on its restart limit. Its frozen `launch.py` was
  replaced by the fixed one and its `config.json` has `rss_growth_limit_gib` 6.0 (event `operator_patch` in
  `events.jsonl`). The frozen trainer is unchanged (no TCN activation checkpointing in this run).
- Health at 05:56 UTC: 52.0M of 64M exposures, about 1,860-2,000 decisions/s, no resource stop since the resume,
  RSS about 1 GiB with transient peaks up to about 4.1 GiB. Expected end: around 08:00 UTC, then the final
  checkpoint and evaluation.
- Check with `ens ps`, `ens cat ensomi-model/artifacts/r2-runs/r2-phaseN-20261006/run.json`, the tail of
  `events.jsonl`, and `logs/resources-*.jsonl` (per-step logs are on the mac only).
- After it ends: checkpoint selection with `select.py` under the phase-N guards; the selected checkpoint is
  phase C's frozen base (plan v5 decision 7 proposal, not yet confirmed by the human).

No subagent or Astra job of this session is running. The phase-C C0 work is the human's and Astra's.

## Code state (`ensomi-model`, branch `r2/train`)
- `7885133`: supervisor restart budget reads regular checkpoints from the directory; `rss_growth_limit_gib`
  6.0 in the phase N and C configs; `tests/r2/test_launch.py`.
- `3520b71`: per-block activation checkpointing of the TCN (`R2Config.checkpoint_temporal`, default on), exact
  same values and gradients (`tests/r2/test_train_step.py`), `DEVIATIONS.md` items 8 and 10.
- Neither is pushed. Tests: 153 passed, 4 skipped on the mac (`tests/r2` plus both temporal suites, job
  `20261007-041657-r2-ckpt-temporal-suite`).
- Not this session's and left alone: a modified `.gitignore` and untracked `src/ensomi_model/evaluation/operators/`
  with four `tests/evaluation/test_operators_*.py`.
- The phase-C configs still say `star_conditions: on`, `star_value: residual`, which the decisions below
  supersede in intent; nothing in code reflects the new difficulty design yet.
- Scratch scripts (untracked, workspace `~/ensomi/.sync/cp/scratch/`): `r2-rss-window-probe.py`,
  `r2-rss-shape.py`, `r2-phaseN-resume.sh`, `r2-phasec-cost/phasec_cost.py`. Cost numbers:
  `artifacts/r2-phasec-cost-20261007/timings.json` on bings-mac.

## Decided this session (all anchored)
- [d-difficulty-proxy](r2-phasec-frontier-20261007.md#d-difficulty-proxy): no star-v2 relabel; the original
  osu!mania star calculation is not the scoped difficulty metric (whole-song only); a skeleton-derived difficulty
  proxy is accepted, preferably extending the osu!mania strain algorithm to short intervals. The first record,
  [d-no-relabel-baseline](r2-phasec-frontier-20261007.md#d-no-relabel-baseline), is superseded.
- [d-strain-proxy-v1](r2-phasec-frontier-20261007.md#d-strain-proxy-v1): Astra's relative attack strain
  ([a-astra-strain-proxy](r2-phasec-frontier-20261007.md#a-astra-strain-proxy)) with a skeleton-relative request;
  attacks only; blind comparisons by the human and Astra through the beatmap-lens harness on the mac; C0 starts.
- Engineering (human instructions): the supervisor fix, the eased RSS guard, the TCN checkpointing.

## Phase C: where it stands
- **C0 scope as offered to the human** ([a-strain-proxy-reading](r2-phasec-frontier-20261007.md#a-strain-proxy-reading)):
  1. the proxy module (O(K) strain trace, ΔW for the 15 head masks, W_H from the cyclic reference, prefix sums)
     with the mechanical tests Astra lists (split and recombine, carry-in, half-open ownership, a head just
     before the end, mirror invariance, chord and same-lane responses);
  2. the natural distribution of r per scope length (2/4/8/16 s, then 32/60 s) on fit_dev source charts;
  3. a fixed-η scan and the analytic budget controller (the counterfactual teacher used directly, no training)
     on the selected or latest phase-N checkpoint, against phase-N continuations;
  4. the 40-60 comparison pairs, synced to the mac and judged there.
  The agent's suggestion to test the analytic controller before training a network to imitate it is a
  suggestion, not a decision.
- **Facts the C0 work needs** ([o-phasec-code-audit](r2-phasec-frontier-20261007.md#o-phasec-code-audit)):
  the 625 actions and their codes, `governed_split`, the frame channels, where the 30 s floor and the star draw
  live, and that `continue_chart` is single-sample with a per-step cost growing along the chart.
- **Costs** ([o-phasec-rollout-cost](r2-phasec-frontier-20261007.md#o-phasec-rollout-cost)): about 700 completed
  60 s scopes per hour per process, sampling about 90% of it; a faster sampler is the first lever if outcome
  training is pursued.
- **Still open for the human:** the remaining items of [v5-open](r2-condition-plan-v5.md#v5-open) that the new
  design does not settle: frozen or KL (Astra's plan keeps the base frozen), phase-C budgets and learning rates,
  checkpoint selection for phase C, conditioning size and the style path (style-module choices,
  [a-style-module](r2-phasen-and-lens-20261006.md#a-style-module)), and whether the four-arm LN study runs as
  Astra proposes. Items about λ_star, μ_star and absolute versus residual star no longer apply in their old form.

## Next steps, in order
1. Monitor phase N to its end; run checkpoint selection and record it.
2. Review what the human and Astra produce in C0 against [a-astra-strain-proxy](r2-phasec-frontier-20261007.md#a-astra-strain-proxy)
   and [d-strain-proxy-v1](r2-phasec-frontier-20261007.md#d-strain-proxy-v1); commit their code changes from
   the control plane (workspace rule 1).
3. Update `docs` or the phase-C configs only once the human accepts the C0 results.
