# Handoff, 2026-10-07 closeout (main session 022ac2c4, about 17:15 UTC)

For the next main session. Start with `RESEARCH.md`, then [r2-collapse-20261007](r2-collapse-20261007.md), the question material, which links everything below. Human inputs of this session are in the private, local file `private/human-inputs/022ac2c4-6b2a-4347-9894-9f142365340e.md` (prompts 1-11, answers 1-5); another clone does not have it.

## Where the question stands

- **Direction (human, 16:05 UTC).** Pattern collapse in long-range self-generation is the primary problem. Controls come later. Experiments should compare complete alternative systems toward the working system, and the comparison should cover network families as well ([d-collapse-primary](r2-collapse-20261007.md#d-collapse-primary), [d-system-comparison](r2-collapse-20261007.md#d-system-comparison)).
- **Diagnosis, measured.** The three components F1-F3 and the architecture links are in [r2-collapse-synthesis-20261007](r2-collapse-synthesis-20261007.md) and [s-arch-constraints](r2-bakeoff-plan-20261007.md#s-arch-constraints).
- **Proposed, awaiting the human: the system bake-off** [r2-bakeoff-plan-20261007](r2-bakeoff-plan-20261007.md).
  - Nested arms: B0 baseline 56M, B1 fine-tune control, B2 self-anchor with history dropout, B3 = B2 + a drawn chart vector θ, and an optional B4 two-rate encoder.
  - Enablers: an incremental sampler, an evaluation module, and the long-chart panel.
  - Nothing is implemented. The human said they will keep discussing; implementation goes to Astra (fast tier, plain code, a stated runtime budget per `~/ensomi/AGENTS.md`) only after the human approves.

## Unfinished at closeout

- **X0, human review.** The pack is ready on bings-mac at `artifacts/r2-collapse-20261007/phase0-astra/x0/`. Start the server with `.venv/bin/python artifacts/r2-collapse-20261007/phase0-astra/x0/serve.py --port 8766` from `~/ensomi/ensomi-model`, then open `http://127.0.0.1:8766/review/`.
  - The human agreed to do it; about 57 minutes of audio.
  - Key: `x0-sealed/KEY.json`. Keep it from the judge until the judgments are done.
  - Then score each measure against the human's marks (window-level AUC ≥ 0.75) as in [p-collapse-experiments](r2-collapse-synthesis-20261007.md#p-collapse-experiments). The generated arrays and row-to-time maps are in `x0-sealed/`.
- **X1 and X3 (phase-0 Opus subagent).** It was asked to stop and return partial results at closeout.
  - Its two mac jobs were still running at 17:12 UTC and were left to finish: `20261007-165741-p0opus-x1a` (start against drift) and `20261007-165827-p0opus-x3` (skeleton → θ R²). Both are analysis only, no training.
  - Outputs: `artifacts/r2-collapse-20261007/phase0-opus/x1/` and `x3/`. Scripts: `~/ensomi/.sync/cp/scratch/r2-collapse/phase0-opus/`.
  - Read the outputs against the fixed pass/fail rules in [p-collapse-experiments](r2-collapse-synthesis-20261007.md#p-collapse-experiments).
  - If the subagent's report arrived, it is recorded in [r2-collapse-20261007](r2-collapse-20261007.md). Otherwise only the files exist.
- **Leftover drafts in `ensomi-model`, untracked and untested:** `src/ensomi_model/r2/c1_difficulty.py`, `lnlevel_eval.py`, `tests/r2/test_c1_difficulty.py`, `test_lnlevel_eval.py`, from the killed night-chain job.
  - Not committed and not reviewed. The difficulty pilot is deprioritised by the human.
  - Also not this line of work: `.gitignore` and `src/ensomi_model/evaluation/operators/`.
- **Code committed this session on `r2/train`, not pushed:**
  - `827e306`: guard (iv) v2 and (v), `ln_level`.
  - `8bd2cdd`: `ln_length`, with config `ce_v2_n_lnlevel2_ft.json` warm-starting from 56M.
  - Both inputs are off by default. `8bd2cdd` carries defensive validation that should be slimmed under the new `AGENTS.md` rule.
- **Morning item for the human:** the blind-screen scoring ([o-c0-screen-scored](r2-phasec-c0-20261007.md#o-c0-screen-scored)) and the C0 difficulty work are parked behind the collapse question.

## Nothing else running

- No training. The 12M fine-tune `r2-lnlevel2-12m-20261007` was killed at about 16:06 UTC. Any early checkpoints are in `artifacts/r2-runs/r2-lnlevel2-12m-20261007/`; they are not evaluated and not needed.
- No Astra job. No Claude subagent besides the phase-0 Opus, which was told to stop.
- Workspace repo `~/ensomi` (pushed):
  - `43da7f5`: `ens watch` no longer shows recovered stream drops as FAILED.
  - `df14292`: the `AGENTS.md` note that Astra leans over-defensive.
  - The uncommitted `--app-server` draft was deleted at the human's request.
