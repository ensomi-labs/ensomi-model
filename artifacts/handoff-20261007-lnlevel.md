# Handoff, 2026-10-07 closeout (main session 31d15d7a)

For the next main session. Start with `RESEARCH.md`, then this note. The session's materials:
- [r2-phasec-c0-20261007](r2-phasec-c0-20261007.md): the C0 stage of phase C;
- [r2-phasen-and-lens-20261006](r2-phasen-and-lens-20261006.md), from [o-phasen-finished](r2-phasen-and-lens-20261006.md#o-phasen-finished)
  to the end: the end of phase N, selection, the short-hold question and the decisions;
- [r2-ln-length-20261007](r2-ln-length-20261007.md): the short-hold investigation report.

Human inputs of this session are in the private, local file
`private/human-inputs/31d15d7a-049a-4750-95a5-fab3de17f164.md` (prompts 1-5, answers 1-5); another clone does not have it.

## Running at closeout (12:38 UTC)

**Astra job `20261007-123218-r2-guard-lnlevel`**, started 12:32 UTC with `--rw` on the `ensomi-model` working tree. Not
stopped, because it was mid-edit. Brief: `~/ensomi/.sync/cp/jobs/20261007-123218-r2-guard-lnlevel/brief.md`. Watch it with
`ens watch 20261007-123218-r2-guard-lnlevel`; its final message lands in
`~/ensomi/.sync/mac/jobs/20261007-123218-r2-guard-lnlevel/last.md`. What it does
([r-guard-lnlevel-job](r2-phasen-and-lens-20261006.md#r-guard-lnlevel-job)):
- **Part A.**
  - Guard (iv) v2: only model-made holds count; a short hold is strictly under 60 ms; per-band references from
    fit_train; the observed count may be at most 1.25 × the expected count.
  - Guard (v): |mean LN-share drift| ≤ 0.05 over `natural_bos`.
  - Phase-N selection rerun on the regenerated panels of 16M-64M.
  - Outputs: `artifacts/r2-guard-v2-20261007/`.
- **Part B.**
  - The `ln_level` flag (whole-song LN-share input, zero-initialised, dropout 0.3), a strict phase-N
    `warm_start`, a fit_train prior, and level modes in sampling and evaluation.
  - Config `ce_v2_n_lnlevel_ft.json` (12M from 48M) and a 0.2M smoke run `r2-lnlevel-smoke-20261007`.
  - Outputs: `artifacts/r2-lnlevel-20261007/report.md`. Its final message gives the exact `launch.py` command.

Rule 3 applies until it ends: do not edit `ensomi-model` tracked files from the control plane meanwhile. At 12:38 UTC
the working tree already had its edits in `data.py`, `evaluate.py`, `features.py`, `model.py`, `operating_point.py`,
`sampling.py`, `select.py` and `train_ce.py`, plus new `defect_references.py`, `ln_level.py`, the config and
`tests/r2/test_guard_v2.py`.

Nothing else is running: no Claude subagent, and no training (phase N finished at 64M).

## Code state (`ensomi-model`, branch `r2/train`, not pushed)

- `7985cf9`: C0. `r2/strain.py` (metric `ras-v1`), the `head_mask_bias` hook in `sampling.continue_chart`,
  `r2/c0.py`, `tests/r2/test_strain.py`.
- `01aacba`: `min_hold_ms` decode mask (default off, kept off by decision), evaluator plumbing, and the
  `c0.py allocation` subcommand. `tests/r2`: 177 passed, 4 skipped.
- Not this line of work and left alone: the modified `.gitignore`, and the untracked
  `src/ensomi_model/evaluation/operators/` with four `tests/evaluation/test_operators_*.py`.

## What this session established (anchored)

- **C0** ([c0-job](r2-phasec-c0-20261007.md#c0-job)):
  - Natural r has a median of 1.43 at every scope length, with a floor of about 1
    ([o-c0-natural-r](r2-phasec-c0-20261007.md#o-c0-natural-r)).
  - The tilt moves r in order, steeply upward ([o-c0-tilt](r2-phasec-c0-20261007.md#o-c0-tilt)).
  - The analytic controller attains requests, but its reference-clock split departs from natural even at the
    median ([o-c0-controller](r2-phasec-c0-20261007.md#o-c0-controller)). A phase-N-relative split with a band
    (NB) cuts that departure by about a third at 8-16 s
    ([o-c0-allocation](r2-phasec-c0-20261007.md#o-c0-allocation)).
  - 48 blind-screen pairs are ready; the human and Astra judge them
    ([o-c0-screen](r2-phasec-c0-20261007.md#o-c0-screen)). The key is `screen/KEY-sealed.json`; keep it from any judge.
  - These measurements used checkpoint 52M (part 2 and the screen) and 48M with the mask (allocation).
- **Phase N.**
  - It finished at 64M, and selection picked nothing ([o-phasen-no-selection](r2-phasen-and-lens-20261006.md#o-phasen-no-selection)).
  - The mask removes generated short holds but not near-head releases
    ([o-minhold-mask](r2-phasen-and-lens-20261006.md#o-minhold-mask)).
  - Cause ([o-lnlen-cause](r2-phasen-and-lens-20261006.md#o-lnlen-cause)): the LN level is not tied to the chart
    from the start of a song, and the model's own history amplifies it. Release decisions are calibrated, the
    representation is not at fault, and the human's keep hypothesis is refuted.
  - Real charts do contain short holds, so guard (iv) and the 60 ms mask rested on a false premise
    ([c-no-short-holds-claim](r2-phasen-and-lens-20261006.md#c-no-short-holds-claim)).
- **Decisions** ([d-lnlen-next](r2-phasen-and-lens-20261006.md#d-lnlen-next)):
  - a fine-tune test of the LN-level input first, a full retrain only if it works;
  - guard (iv) corrected and guard (v) added;
  - the mask off by default.
  - Earlier: [d-c0-allocation-test](r2-phasec-c0-20261007.md#d-c0-allocation-test),
    [d-phasen-base-mask](r2-phasen-and-lens-20261006.md#d-phasen-base-mask) (its mask premise since withdrawn).

## Next steps, in order

1. When the Astra job ends, read `last.md` and both reports. Check:
   - the old-definition counts against `evals.jsonl`;
   - flag-off byte identity;
   - the zero-init identity;
   - the strict warm start;
   - the test count.
   Then commit the code from the control plane and record the corrected selection result in the notes.
2. **Changed after the closeout (12:50 UTC).** The whole-song LN-level input is a diagnostic only, not the fix
   ([d-lnlen-hint-diagnostic](r2-phasen-and-lens-20261006.md#d-lnlen-hint-diagnostic)). Do not launch the 12M
   fine-tune as a fix. A fresh Opus subagent is investigating why phase N lacks a normal LN distribution without a
   hint ([q-lnlen-hintfree](r2-phasen-and-lens-20261006.md#q-lnlen-hintfree)).
   - Outputs: `artifacts/r2-ln-level-20261007/` on bings-mac; report to this session's scratchpad
     `ln-level-dynamics-report.md`.
   - If this session ends before the subagent reports, the report may be lost with the scratchpad. Then check
     the outputs directory and rerun the investigation from the question.
3. Use the oracle-level fine-tune once as a probe of placement within a level, only if the investigation says it
   adds information.
4. Phase C stays on hold for its base: rerun C0 parts 3-4 and the allocation on whichever checkpoint is selected.
   Open for the human: [v5-open](r2-condition-plan-v5.md#v5-open), the style-module choices, and the blind-screen
   judgments.
5. Open design point, not raised with the human yet: a whole-song LN-level input in phase N overlaps with phase C's
   LN-share request. Decide how a scoped LN request composes with the chart-level level.
