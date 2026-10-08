# Handoff, 2026-10-08 about 06:10 UTC (main session 00e97a87)

**For the next main session.**
- **Start with:** `RESEARCH.md`, the current movement, then [r2-representation-20261008](r2-representation-20261008.md) from [d-projection-spaces](r2-representation-20261008.md#d-projection-spaces) onward.
- **Human inputs of this session** are in the private, local file `private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md`: prompts 1-10 and answers 1-2. Another clone does not have it.

## Where the question stands

- **Focus (human):**
  - collapse in self-generation first;
  - within it, representations first, then measures and evaluation;
  - `control` after that;
  - the style formulation in parallel ([d-control-after-eval](entry-point-resynthesis-20261008.md#d-control-after-eval), [d-style-parallel](entry-point-resynthesis-20261008.md#d-style-parallel)).
- **Representation, the current direction (human, 05:00 UTC; [d-projection-spaces](r2-representation-20261008.md#d-projection-spaces)):**
  - project chart sections into several spaces, each defined by what it encodes and what it leaves indistinguishable;
  - fit mappings, constants and metrics to the real corpus.
- **What the human rejected:** the three-block R and its Conv1d embedding (trivial, coupled, hidden priors). The step-1 numbers about R are no longer evidence for the new design. The same-rows human-pair null and the mapper split carry over as facts about the data ([s-step1-answer](r2-representation-20261008.md#s-step1-answer)).
- **Data (human, [d-lens-pool-primary](r2-representation-20261008.md#d-lens-pool-primary)):** the Lens-annotated pool is the primary data. The human's own screens (X0, C0) are too small.
  - Pool on disk: 6,039 sections; 2,860 complete over all five concepts; 4,143 joined to the R2 cache; 600 human observations ([o-lens-pool-inventory](r2-representation-20261008.md#o-lens-pool-inventory), design §9.1).
- **The design study:** [r2-represent-design-20261008](r2-represent-design-20261008.md) proposes eight spaces, S1-S8.
- **The human's answers on it ([d-design-answers](r2-representation-20261008.md#d-design-answers)):**
  - shape keeps position;
  - mass is modelled jointly, per row and per second; the agent's reading is still to be confirmed;
  - the hand difference is kept in S3;
  - the calibration and heldout charts stay closed;
  - the whole-song unit is undecided, so the agent defaults to fixed metrical windows;
  - speed counts, so S8 is built;
  - **new:** a VQ-VAE-like learned representation that distinguishes the five concepts on the annotation pool;
  - tuning may use the whole fit_train corpus (11,368 charts).
- **Not built:** S7, which nobody addressed.

## First step for the next session

**Relaunch the two subagents unchanged from their briefs:**
- **spaces-build (Opus):** `~/ensomi/.sync/cp/scratch/r2-spaces-build/brief.md`.
  - Builds S1-S6 and S8, with invariance unit tests first.
  - `prereg.md` comes before any validation result.
  - Validation runs D, then V1, then V2, plus the batch swap, under the main thread's stricter pass rule.
  - Measurement only. Mac ≤ 3 threads, about 60 minutes.
- **vqvae (Fable):** `~/ensomi/.sync/cp/scratch/r2-vqvae/brief.md`.
  - The decoder is conditioned on the rows.
  - Hyperparameters are chosen by label-free criteria.
  - Delivers the design, the code and a smoke run of at most about 3 minutes.
  - **Its full training run waits for the human's check.**
- **State of the stopped launch:** both were delegated at about 06:05 UTC and stopped at 06:05:50 for this handoff. They had written no file and started no mac job.
- **Launch prompt:** "read the brief file and follow it, including the files it tells you to read; return the full report as your final message".

When they return, save each report into the notes, check headline numbers against the mirrored outputs, and bring the results to the human.

## Waiting on the human

- **Style formulation:** six questions ([q-style-20261008](style-formulation-recheck-20261008.md#q-style-20261008)). The human said to look at representation first.
- **Joint mass units:** confirm the agent's reading ([d-design-answers](r2-representation-20261008.md#d-design-answers), item 2).
- **Whole-song unit (Q5):** undecided.
- **S7:** not addressed. It is a count-based model of familiar patterns.
- **Not authorised:** the clip screen (step 2 of the old plan), now on hold; the decode-controller plan.

## Code and machines

- **Code:** `r2/train` at `3692ca2`: the guardrails skill and the AGENTS.md section. Committed, not pushed.
  - Untracked and unrelated, left alone: `src/ensomi_model/r2/c1_difficulty.py`, `lnlevel_eval.py`, their tests, `src/ensomi_model/evaluation/operators/`, and the `.gitignore` change.
- **Style draft:** branch `docs/style-formulation` at `8da2bda`, worktree `~/wt/ensomi-model-formulation`. Not reviewed by the human.
- **Mac venv:** scikit-learn 1.7.2 was added by `uv pip install`. It is not in `pyproject.toml`, so `uv sync` would remove it.
- **Outputs this session, on bings-mac in `artifacts/r2-represent-20261008/`:**
  - `opus-skeleton/`, `fable-represent/`;
  - `step1/`, including 240 R2 generations on human-pair rows and `encoder.pt`;
  - `design/`: the inventory rerun and statistics.
  - Scripts are in `~/ensomi/.sync/cp/scratch/r2-represent*/`.
- **Running:** nothing. No mac jobs, no Astra job, no Claude subagents.
