# Ensomi V3 research

Code and document paths resolve at the baseline commit: `git show 5c56e28:<path>` in any ensomi-model clone. Recurring agent failures and their guardrails: `.agents/skills/ensomi-research-guardrails/SKILL.md` on `r2/train` (from `3692ca2`), with the evidence in [agent-failure-modes](artifacts/agent-failure-modes.md) and [r2-represent-history-20261008](artifacts/r2-represent-history-20261008.md). Section rules and the review policy are in the research-relay convention, "Entry point". The fuller account this file gave before 2026-10-06, including the path from the lineage review to R2 v2: [entry-point-before-20261006](artifacts/entry-point-before-20261006.md).

## Vision

Connect listening to music, playing it and creating for it by changing how music and choreography are joined. ensomi does not start from a new four-key rule set. Three scenarios carry the vision:

- **Ambient play.** Music is already playing around you. ensomi recognises it, aligns to the current position, and a chart generated in real time follows the music without interrupting it.
- **Mapping.** A mapper selects one section and asks for candidate choreography while the surrounding chart stays in place. Candidates are compared, kept, retried or undone. Style and existing charts steer them. Choice and expression stay with the mapper.
- **Training.** A player picks music they like and a practice target. Verified conditions shape the arrangement around the movement to practise.

Standing constraints:

- A capability is claimed only once verified: no condition is offered as a control before it works, and latency is quoted only as measured.
- Complete audio may be used in training and inference; chart decisions and publication stay causal.
- Output is playable 4K over 2–6★, covering ordinary organisation and expressive styles.

_Review: agent draft 2026-09-30, not yet reviewed. Summarised from the source the human designated ([private, local](artifacts/private/human-inputs/7657b130-e91d-453a-a46c-8d5aa017b74e.md#prompt-2)): ensomi-web on bings-mac, outside Git, files `docs/product-direction.md`, `src/config/scenarios.json`, `src/pages/about.astro`. Text unchanged in the 2026-09-30 reset except for one removed link._

## Position

Baseline: `main` at `5c56e28`, with the released model at tag `r1-restored-6.75m` (`8f13103`; weights `sed-i/pulsefield-r1-restored` on Hugging Face). The system is **R1**, a chart continuation model (3.1M parameters). From a seed prefix it samples one complete four-lane row at each remaining event time, and those times come from an existing chart ([s-r1-interface](artifacts/lineage-review/synthesis.md#s-r1-interface)). Package `src/ensomi_model/research/bounded_typed_continuation/`; design and results in `docs/research/bounded_typed_continuation.md`.

| Component | What it does | Where | Checked by |
| --- | --- | --- | --- |
| Chart contract and exact state | Row actions, legality, long-note occupancy and clocks by replay; a support mask removes illegal rows | `contract.py`, `support.py`, `features.py` | Tests |
| Local history encoder | Causal dilated convolutions over the last 511 committed rows | `temporal.py` | Tests, including crop invariance |
| Seed conditioning | The encoded seed's mean, added to the hand vectors through a zero-initialised residual | `model.py` | Tests |
| Landmark memory | A GRU over all committed rows; the hand vectors attend to a landmark every 64 head rows | `long_memory.py` | Tests |
| Joint row head | Mirror-equivariant per-hand scores with a rank-16 coupling over the 256 complete rows | `model.py` | Tests |
| Correction residuals | Three score corrections fitted on a frozen parent to hand-written rules on generated trajectories; a policy residual, not a player-response model | `routing.py`, `consequence.py`, `response.py`, `recovery.py` | Tests and paired generations; the rules by agent inspection only |
| Training | Teacher-forced source cross-entropy on bounded windows, in six stages (`docs/research/r1_staged_restoration.md`) | `data.py`, `corpus.py`, `train_run.py`, `src/ensomi_model/research/r1_restore/` | Frozen inputs and stage audits |
| Generation and recovery | Sampling inside the support mask, snapshots rebuilt by replay, osu! export, an output check | `generation.py`, `generate_run.py`, `verification.py`, `src/ensomi_model/osu_core/export.py` | Tests; published examples reproduce at the tag |

For R1, legal rows, exact replay and export are established. Playability and quality are not, and the human judged R1 to fail substantially ([d-r1-fails](artifacts/r1-verdict.md#d-r1-fails)). The baseline also holds packages outside R1: comparison arms, a 35M teacher profile, earlier research packages, an unused audio frontend and the pre-V3 stack ([list](artifacts/entry-point-before-20261006.md#old-position)).

Not in the baseline:

- Event times from audio. R and H come from an existing chart.
- Generation without a seed chart.
- Any request input for difficulty, style or long-note amount.
- Regeneration of a section between fixed past and future context.
- Measured publication ahead of playback; no client or service.
- A validated evaluator of chart quality. Checks are mechanical, plus inspection against Beatmap Lens examples.
- A chart-level identity: a choice of the chart's character beyond what the rows imply, held over the song.

_Review: the human set the baseline and section scope on 2026-09-30 ([private, local](artifacts/private/human-inputs/0e81052c-94ac-435b-92a1-44c2f5d4be7f.md#prompt-1)). The lines are an agent draft, corrected against the code after the lineage review ([audit, 4a](artifacts/lineage-review/fable/synthesis-audit.md)). The quality sentence records the human's judgment of 2026-10-03 ([private, local](artifacts/private/human-inputs/41ba879c-870c-4368-9fa0-fb9fdf2ccce3.md#prompt-1)). It was condensed on 2026-10-06 without changing a claim; the earlier wording is in [old-position](artifacts/entry-point-before-20261006.md#old-position). The agent added one capability line (chart-level identity) on 2026-10-08._

## Problem map

The map reads as three layers.
- **What a chart is and what is known about it:** `contract`, `data`, `response`.
- **How a chart is described and judged:** `representation`, `evaluation`.
- **How one is made:**
  - `time`: the rows, that is, when notes fall;
  - `identity`: the chart-level choice the rows leave open;
  - `proposal`: the arrangement row by row;
  - `rollout`: holding quality over the model's own history;
  - `selection`, `control`, `style`: steering what is made;
  - `realtime`, `local-edit`: the product settings.

| Node | Problem | Needs | Status | Best current understanding |
| --- | --- | --- | --- | --- |
| <a id="contract"></a>`contract` | What a legal chart, a legal continuation and a committed prefix are. | None | held | `docs/formulation/notation.md` |
| <a id="data"></a>`data` | What the corpus and its human judgments can teach: which arrangements, styles and conditions are present, missing or distorted by sampling. | `contract` | open | Target: the 2–6★ ranked and loved distribution ([d-target-distribution](artifacts/evaluation-first.md#d-target-distribution)). Human labels on hand: X0 calls, the C0 screen, and the Lens pool, the primary data ([d-lens-pool-primary](artifacts/r2-representation-20261008.md#d-lens-pool-primary)). The pool holds 6,039 sections, 2,860 complete across all five concepts and 4,143 joined to the R2 cache; 598 cells are human ([o-lens-pool-inventory](artifacts/r2-representation-20261008.md#o-lens-pool-inventory)). About 2,170 same-song pairs of human charts on near-identical rows serve as a natural experiment and a noise floor. 95 % of them are one mapper's two difficulties ([p-skeleton-benchmark](artifacts/r2-represent-skeleton-20261008.md#p-skeleton-benchmark), [o-step1-null-same-mapper](artifacts/r2-represent-step1-20261008.md#o-step1-null-same-mapper)). |
| <a id="response"></a>`response` | How a player responds to a continuation given the past: which legal futures are comfortable, demanding or unplayable. | `contract` | open | Undefined at the baseline ([s-response-undefined](artifacts/lineage-review/synthesis.md#s-response-undefined)). Physiology is in scope, with corpus and priors ([d-response-priors](artifacts/evaluation-first.md#d-response-priors)). Proposed: a demand machine ([p-demand-machine](artifacts/operator-properties.md#p-demand-machine)); what player data can identify ([a-fable-views](artifacts/player-data.md#a-fable-views)). |
| <a id="representation"></a>`representation` (added 2026-10-08) | How to describe the arrangement of a chart section, given its rows, so that the distinctions the human makes are recognisable. | `contract`, `data` | open | Every representation used so far encoded levels; the human's complaints and the Lens concepts are relations ([o-fable-level-vs-relation](artifacts/r2-represent-fable-20261008.md#o-fable-level-vs-relation)). The human wants projections into several spaces, each defined by what it encodes and leaves indistinguishable, with corpus-tuned constants ([d-projection-spaces](artifacts/r2-representation-20261008.md#d-projection-spaces)); proposed design: [represent-design](artifacts/r2-represent-design-20261008.md#represent-design). Describe the arrangement against a reference on the same rows ([s-represent-answer](artifacts/r2-representation-20261008.md#s-represent-answer)). On a same-rows human null, the LN-relations block tells R2 from a second human without labels, but it locates the human's marks only on one chart ([s-step1-answer](artifacts/r2-representation-20261008.md#s-step1-answer)). Of the designed spaces, S3 (metrical recurrence) recognises the human's jack and trill labels beyond mass; the others characterise, and they do not separate factors as designed ([s-spaces-reading](artifacts/r2-spaces-20261008.md#s-spaces-reading)). |
| <a id="evaluation"></a>`evaluation` | Telling whether a generated chart is good, with observers that see the distinction in dispute. | `contract`, `response`, `time`, `representation` | open | No measure has been validated against the human; deciding with unvalidated measures is the most damaging recurring failure ([s-history-worst](artifacts/r2-represent-history-20261008.md#s-history-worst)). The targets are separate: marks the human places, mostly LN relations, and whole-song impressions such as jacks and repetition ([o-fable-two-targets](artifacts/r2-represent-fable-20261008.md#o-fable-two-targets), [d-x0-whole-song](artifacts/r2-representation-20261008.md#d-x0-whole-song)). Any threshold has to name its null: the null between two mappers is about 1.8× wider than within one ([s-step1-answer](artifacts/r2-representation-20261008.md#s-step1-answer)). Earlier design, paused: [p-eval-object](artifacts/evaluation-first.md#p-eval-object), [operator-properties](artifacts/operator-properties.md). |
| <a id="time"></a>`time` | How event times (the rows) are represented, inferred from audio (tempo and phase included) and scored, and what the rows decide about the chart. | `contract`, `data` | open | R2 takes rows from an existing chart ([d-r2v2-scope](artifacts/r2-v2-stage0-state.md#d-r2v2-scope)). Those rows fix density and section dynamics and most of the difficulty, and little of the chart's identity ([s-skeleton-achieves](artifacts/r2-represent-skeleton-20261008.md#s-skeleton-achieves)). Rows from the legacy timing model lose the difficulty that human rows carry (4 songs; [o-skeleton-shift](artifacts/r2-represent-skeleton-20261008.md#o-skeleton-shift)). |
| <a id="identity"></a>`identity` (added 2026-10-08) | What chart-level choice a chart makes beyond its rows (levels and character held across the song), and how it is chosen and held. | `data`, `time`, `representation` | open | The rows fix at most 41 % of identity and none of hand balance; R2 reproduces the rows' part and draws the rest ([o-skeleton-identity](artifacts/r2-represent-skeleton-20261008.md#o-skeleton-identity)). LN level and LN-head placement are mostly one mapper's convention, so they belong to identity ([c-ln-placement-mapper](artifacts/r2-representation-20261008.md#c-ln-placement-mapper)). With no chart-level choice, the level random-walks ([s-diagnose-answer](artifacts/r2-diagnose-synthesis-20261008.md#s-diagnose-answer)). θ must be drawn per chart ([o-x3-skeleton-theta](artifacts/r2-collapse-20261007.md#o-x3-skeleton-theta)). The human agrees that a whole-chart identity vector is probably needed ([d-represent-first](artifacts/r2-representation-20261008.md#d-represent-first)). |
| <a id="proposal"></a>`proposal` | A distribution over complete rows, given the rows' times, audio and history, that covers real ordinary and expressive arrangements. | `contract`, `data`, `time`, `identity` | open | R1 fails ([d-r1-fails](artifacts/r1-verdict.md#d-r1-fails)). R2 makes one decision per head row and places releases with a pointer ([d-r2-row-decision](artifacts/r2-ln-design.md#d-r2-row-decision)). It places chords and LN heads only as well as the local rows allow ([o-skeleton-placement](artifacts/r2-represent-skeleton-20261008.md#o-skeleton-placement)). Different mappers still agree on chord placement beyond the rows; LN placement agrees mainly within one mapper ([o-step1-mapper](artifacts/r2-represent-step1-20261008.md#o-step1-mapper)). Architecture: [r2-architecture-20261007](artifacts/r2-architecture-20261007.html). |
| <a id="rollout"></a>`rollout` | Keeping proposal quality on histories the model produced itself, not only on source histories. | `proposal`, `identity` | open | Capacity and exposure did not remove free-running collapse ([s-likelihood-vs-rollout](artifacts/lineage-review/synthesis.md#s-likelihood-vs-rollout)). R2 is calibrated on real histories, copies its recent level and holds no chart-level anchor ([s-collapse-agree](artifacts/r2-collapse-synthesis-20261007.md#s-collapse-agree), [s-diagnose-answer](artifacts/r2-diagnose-synthesis-20261008.md#s-diagnose-answer)). The start bias was measured against the wrong reference ([c-band-reference](artifacts/r2-representation-20261008.md#c-band-reference)). Row cross-entropy alone is not enough ([d-beyond-row-ce](artifacts/r1-verdict.md#d-beyond-row-ce)). |
| <a id="selection"></a>`selection` | Choosing among proposed futures by their response. | `proposal`, `response`, `evaluation` | open | The human's direction: rank whole sequences with a stateful score ([d-decode-direction](artifacts/r2-bakeoff-night-20261007.md#d-decode-direction)). Decoding holds the level given a target, but the local texture is not in the proposals ([s-diagnose-answer](artifacts/r2-diagnose-synthesis-20261008.md#s-diagnose-answer)). Its "identity r 0.94" copied the source ([c-band-reference](artifacts/r2-representation-20261008.md#c-band-reference)). Options: [decoding-stage](artifacts/decoding-stage.md). |
| <a id="style"></a>`style` | What a style request asks for, as organisation separate from chart properties, and how it composes with property requests, release and the return to natural generation. | `contract`, `data`, `representation` | open | Formulation draft `8da2bda`, not reviewed ([open-questions](artifacts/style-formulation-rethink.md#open-questions)); decisions so far: [d-formulation-answers](artifacts/r2-style-formulation-check.md#d-formulation-answers), [d-target-strength](artifacts/r2-style-formulation-check.md#d-target-strength). The five Lens concepts become recognisable only with relational representations ([o-fable-lens](artifacts/r2-represent-fable-20261008.md#o-fable-lens)). The baseline ρ is close to `identity` ([o-rho-identity](artifacts/style-formulation-recheck-20261008.md#o-rho-identity)). |
| <a id="control"></a>`control` | Making a scoped request for difficulty, style or long-note amount change the output in the requested way. | `proposal`, `response`, `data`, `style`, `identity`, `evaluation` | open | Still part of R2's goal, solved after the representation, measure and evaluation work ([d-control-after-eval](artifacts/entry-point-resynthesis-20261008.md#d-control-after-eval), [d-collapse-primary](artifacts/r2-collapse-20261007.md#d-collapse-primary)). History: [s-controls-order](artifacts/lineage-review/synthesis.md#s-controls-order); target interface [d-final-conditions](artifacts/r2-ln-design.md#d-final-conditions); plan v5 and C0 ([r2-condition-plan-v5](artifacts/r2-condition-plan-v5.md), [o-c0-screen-scored](artifacts/r2-phasec-c0-20261007.md#o-c0-screen-scored)). |
| <a id="realtime"></a>`realtime` | Publishing settled coverage ahead of playback within a latency budget. | `proposal`, `selection` | open | The commit rule is in `docs/formulation/notation.md`, "Provisional branches and prefix commit". Nothing measured. |
| <a id="local-edit"></a>`local-edit` | Regenerating one section between fixed past and future context. | `proposal`, `control` | open | No work; the formulation has no notation for a window bounded by a fixed later chart. |

_Review: agent draft. The map dates from 2026-09-30 and was re-synthesised on 2026-10-08 at the human's request ([private, local](artifacts/private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#prompt-4)). The human has not reviewed the map._
- _Added by the agent: the nodes `representation` and `identity`, and new edges into `evaluation`, `proposal`, `rollout`, `selection`, `style` and `control`. No edge was removed._
- _The human rejected parking `control` on 2026-10-08: it stays open and comes after representation, measures and evaluation ([private, local](artifacts/private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#prompt-5))._
- _Older proposals still unreviewed: node `time` (2026-09-30) and node `style` (2026-10-06)._
- _The previous map and its reasons: [old-map-20261008](artifacts/entry-point-before-20261008.md#old-map-20261008), [entry-point-resynthesis-20261008](artifacts/entry-point-resynthesis-20261008.md)._

## Current movement

**Focus, set by the human:**
- 2026-10-06: finish R2 v2 phase N and phase C, and finalise the style formulation ([private, local](artifacts/private/human-inputs/da66cd9e-81c2-4ad1-a989-4fe6ef7c81fe.md#prompt-6)).
- 2026-10-07: pattern collapse in long-range self-generation by the natural model is the primary problem; controls come later ([d-collapse-primary](artifacts/r2-collapse-20261007.md#d-collapse-primary)).
- 2026-10-08: representations of chart sections first, then measures derived from them ([d-represent-first](artifacts/r2-representation-20261008.md#d-represent-first)).

1. **`representation` and `evaluation`: find a description of arrangements that recognises what the human sees, then measures from it.**
   - **Question:** which representation of the arrangement given its rows separates the human's marks, and the Lens concepts, beyond the spread between two human charts on the same rows? Which measures derived from it pass a check against the human, kept apart from the labels used to design them?
   - **Why now:** every failed round decided with measures that never recognised the human's judgments ([s-history-worst](artifacts/r2-represent-history-20261008.md#s-history-worst)).
   - **Where it stands:** a candidate representation exists ([p-fable-representation](artifacts/r2-represent-fable-20261008.md#p-fable-representation)), and its X0 result is post hoc ([s-represent-gaps](artifacts/r2-representation-20261008.md#s-represent-gaps)).
   - **Step 1 returned at 04:30 UTC, measurement only and authorised by the human ([r-step1](artifacts/r2-representation-20261008.md#r-step1)):** [s-step1-answer](artifacts/r2-representation-20261008.md#s-step1-answer).
     - The null is mostly within one mapper.
     - A one-sided LN-excess flag locates the X0 marks, but on one chart only.
     - Jacks and repetition separate generated from real only unpaired.
     - LN placement agreement between humans is mostly one mapper's convention.
   - **Redirected by the human, 05:00 UTC ([d-projection-spaces](artifacts/r2-representation-20261008.md#d-projection-spaces)):**
     - Project charts into several designed spaces, each defined by what it encodes and what it leaves indistinguishable, with mappings, constants and metrics tuned to the corpus.
     - The three-block R and its Conv1d embedding are not the design.
     - The clip screen is on hold.
   - **Design study returned ([o-design-returned](artifacts/r2-representation-20261008.md#o-design-returned)), and the human answered its questions ([d-design-answers](artifacts/r2-representation-20261008.md#d-design-answers)).**
   - **Relaunched unchanged after the 06:10 handoff** ([r-spaces-vqvae](artifacts/r2-representation-20261008.md#r-spaces-vqvae), [r-spaces-vqvae-relaunch](artifacts/r2-representation-20261008.md#r-spaces-vqvae-relaunch)):
     - S1-S6 and S8 built and checked against a preregistered pass rule on the Lens pool ([o-spaces-returned](artifacts/r2-representation-20261008.md#o-spaces-returned)). S3 recognises jack and trill beyond mass, on human cells too; S4 recognises the labeller's LN coordination, not shown on human cells; most blindness claims fail;
     - a VQ-VAE-like learned representation: design, code and smoke run returned ([o-vqvae-returned](artifacts/r2-representation-20261008.md#o-vqvae-returned)). The pipeline works; nothing is established about the concepts yet. Its full run and three design questions wait for the human ([q-vqvae](artifacts/r2-vqvae-20261008.md#q-vqvae)).
   - **Would redirect (proposed):**
     - if the human-pair null absorbs the X0 LN result, the representation question reopens;
     - if the screen disagrees with the measures, they stay characterisation only.
2. **`identity`, `rollout` and `proposal`: what the model must decide beyond the rows. Next; no work is authorised.**
   - **Question:** what chart-level choice must be made and held over the song, and what information places chords and LN heads beyond the rows? Possibly audio, which would reopen R2's scope.
   - **Why it waits:** the choice has to be described in a validated representation first. Step 1 points LN toward identity, a mapper's convention, and chord placement toward song information ([c-ln-placement-mapper](artifacts/r2-representation-20261008.md#c-ln-placement-mapper)).
   - **Shelved by the human:** the decode-controller plan ([p-diagnose-plan](artifacts/r2-diagnose-synthesis-20261008.md#p-diagnose-plan)), not started.
3. **`style`: the style formulation, a separate branch pushed in parallel as formulation work** (human, 2026-10-06, confirmed 2026-10-08: [d-style-parallel](artifacts/entry-point-resynthesis-20261008.md#d-style-parallel)).
   - **Question:** what do the draft's open questions resolve to, and where does it still disagree with `notation.md`, `gameplay-state.md` and the R2 v2 plan?
   - **Where it stands:** draft `8da2bda` on branch `docs/style-formulation`, not reviewed by the human ([open-questions](artifacts/style-formulation-rethink.md#open-questions), [conflicts](artifacts/style-formulation-rethink.md#conflicts)).
   - **Re-examined against the 2026-10-07/08 findings ([style-formulation-recheck-20261008](artifacts/style-formulation-recheck-20261008.md)):** the draft's baseline ρ is close to `identity`, but leaves chart-level LN and chord levels unowned and has no supplied rows ([o-rho-identity](artifacts/style-formulation-recheck-20261008.md#o-rho-identity)).
   - **Waiting on the human:** six questions ([q-style-20261008](artifacts/style-formulation-recheck-20261008.md#q-style-20261008)). One of them asks whether ρ should still be built after the conditions.
   - **Would redirect (proposed):** a resolution that changes what a style or property request means reopens the control plan.
4. **`control`: after representation, measures and evaluation.** Phase C and its open decisions ([v5-open](artifacts/r2-condition-plan-v5.md#v5-open)) and the style module ([a-style-module](artifacts/r2-phasen-and-lens-20261006.md#a-style-module)) wait for that work.

**Where things stand:**
- Phase N finished; no checkpoint selected ([o-phasen-no-selection](artifacts/r2-phasen-and-lens-20261006.md#o-phasen-no-selection)).
- Bake-off round 1 measured mostly checkpoint noise ([o-bakeoff-r1](artifacts/r2-bakeoff-night-20261007.md#o-bakeoff-r1), [s-diagnose-answer](artifacts/r2-diagnose-synthesis-20261008.md#s-diagnose-answer)).
- What the agents got wrong, and the guardrails now in agents' context: [r2-round1-agent-failures-20261008](artifacts/r2-round1-agent-failures-20261008.md), [d-guardrails-in-context](artifacts/r2-represent-history-20261008.md#d-guardrails-in-context).
- Parallel discussion sessions, started at the human's request on 2026-10-08 ([private, local](artifacts/private/human-inputs/6d71c094-25a1-4a33-8638-8cde30b9b79b.md#prompt-2)): the style formulation and the whole-formulation update (worktree `~/wt/ensomi-model-formulation`, branch `docs/style-formulation`), and head rows from audio (worktree `~/wt/ensomi-model-audio-rows`, branch `docs/audio-to-rows`). They record their findings as separate materials; this file is edited by one main session.
- Handoffs: [handoff-20261008-spaces](artifacts/handoff-20261008-spaces.md) (latest), [handoff-20261008-represent](artifacts/handoff-20261008-represent.md), [handoff-20261007-collapse](artifacts/handoff-20261007-collapse.md). The earlier account of this section: [old-movement-20261008](artifacts/entry-point-before-20261008.md#old-movement-20261008).

_Review: the focus is the human's: the redirects of 2026-10-07 and 2026-10-08, and the authorisation of step 1 ([private, local](artifacts/private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#answer-1)). The human reviewed the sequencing on 2026-10-08: `control` comes after representation, measures and evaluation, and the style formulation continues in parallel ([private, local](artifacts/private/human-inputs/00e97a87-25ca-4fab-a49a-668fbf454be4.md#prompt-5)). The questions, the why-now lines and the redirect lines are an agent draft._

## Views

None yet.
