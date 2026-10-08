# Handoff, 2026-10-08 about 03:12 UTC (main session 28c38740)

For the next main session. Start with `RESEARCH.md` (current movement), then [r2-representation-20261008](r2-representation-20261008.md), the open question. Human inputs of this session are in the private, local file `private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md` (prompts 1-11, answers 1-2). An external review the human pasted is in `private/external/chatgpt-pro-review-20261007.md`. Another clone has neither.

## Where the question stands

- **Direction (human, 2026-10-08):** representations of chart sections first, then measures derived from them ([d-represent-first](r2-representation-20261008.md#d-represent-first)).
  - The recurring failure is measurement: X0 collapse and the five Lens concepts both went unrecognised.
  - The agents also never examined what the supplied head rows imply. The human's premise: audio decides the rows, the rows carry enough choreography and control, and that separation lets R2 achieve some things without others.
  - The main thread agreed, with two conditions. A representation counts only if a simple probe recognises the labels we already hold. It should describe R2's choices given the rows ([a-represent-agree](r2-representation-20261008.md#a-represent-agree)).
- **Before that, the night and its diagnosis:**
  - Round 1 ([o-bakeoff-r1](r2-bakeoff-night-20261007.md#o-bakeoff-r1), with corrections [c-bakeoff-r1-readings](r2-bakeoff-night-20261007.md#c-bakeoff-r1-readings)) measured mostly checkpoint noise.
  - The diagnosis ([s-diagnose-answer](r2-diagnose-synthesis-20261008.md#s-diagnose-answer)): the model copies its recent level and has no chart-level choice; identity inputs are unused under teacher forcing; decoding holds the level given an external target; local texture is not in the proposals.
  - Reports: [arms](r2-diagnose-arms-20261008.md), [decode](r2-diagnose-decode-20261008.md), [Fable](r2-diagnose-fable-20261008.md).
  - What the agents got wrong: [r2-round1-agent-failures-20261008](r2-round1-agent-failures-20261008.md).
- **On hold, not authorised:** the decode-controller plan [p-diagnose-plan](r2-diagnose-synthesis-20261008.md#p-diagnose-plan).

## First step for the next session

**Relaunch the representation round.** It was delegated at about 03:06 UTC and stopped at 03:11 UTC for this handoff, before any worker had written a file or started a mac job. The briefs are self-contained; launch them unchanged as three fresh subagents (Opus, Fable, Opus):
- `~/ensomi/.sync/cp/scratch/r2-represent/context.md` (shared; it points to `../r2-diagnose/common.md`);
- `brief-opus-history.md`: recurring agent failure patterns across the record; why the 2026-10-02 list did not stop them; at most five structural changes. Read-only.
- `brief-fable-represent.md`: counterfactual analysis of representation spaces and measures. The pilot probes window statistics against a structural representation and a learned section embedding, on the Lens sections and X0. At most 4 mac threads.
- `brief-opus-skeleton.md`: what the supplied rows carry and assume, how much they determine (pilot), deployment shift, counterfactual. At most 2 threads.

Outputs go to `artifacts/r2-represent-20261008/<worker>/` on bings-mac. Save each report into the notes, synthesise, and bring the answer to the human before running anything else.

## State of code, runs and machines

- **Code:** `r2/train` at `df4458c` (bake-off round 1, committed, not pushed).
  - Untracked and unrelated, left alone: `src/ensomi_model/r2/c1_difficulty.py`, `lnlevel_eval.py`, their tests, `src/ensomi_model/evaluation/operators/` and the `.gitignore` change.
- **Round 1 outputs on bings-mac:**
  - `artifacts/r2-bakeoff-20261007/` (panels, `comparison.md`, `b3-probe/`, `night.sh`);
  - checkpoints in `artifacts/r2-runs/r2-bo-{b1,b2,b3}-20261007/`.
  - The smoke folders `artifacts/r2-bakeoff-20261007/review-smoke*` can be deleted; one holds a 37 MB checkpoint.
- **Diagnosis outputs:** `artifacts/r2-diagnose-20261008/{opus-arms,opus-decode,fable-picture}/`. Scripts in `~/ensomi/.sync/cp/scratch/r2-diagnose/`.
  - `opus-arms` left a pilot checkpoint (B3 + 2M, history dropout 0-32) in `train-hd032/`, and a weight average in `swa-ckpt/`.
- **Nothing is running:** no mac jobs (`ens ps`, 03:10 UTC), no Astra job, no Claude subagents.
- **Memory added:** `pilot-before-night`: measure the noise floor and run a cheap ablation before any training night; put evidence that contradicts the plan to the human.
