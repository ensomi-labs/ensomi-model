# Ensomi V3 research

Code and document paths resolve at the baseline commit: `git show 5c56e28:<path>` in any ensomi-model clone. Section rules and the review policy are in the research-relay convention, "Entry point". The fuller account this file gave before 2026-10-06, including the path from the lineage review to R2 v2: [entry-point-before-20261006](artifacts/entry-point-before-20261006.md).

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

_Review: the human set the baseline and section scope on 2026-09-30 ([private, local](artifacts/private/human-inputs/0e81052c-94ac-435b-92a1-44c2f5d4be7f.md#prompt-1)). The lines are an agent draft, corrected against the code after the lineage review ([audit, 4a](artifacts/lineage-review/fable/synthesis-audit.md)). The quality sentence records the human's judgment of 2026-10-03 ([private, local](artifacts/private/human-inputs/41ba879c-870c-4368-9fa0-fb9fdf2ccce3.md#prompt-1)). It was condensed on 2026-10-06 without changing a claim; the earlier wording is in [old-position](artifacts/entry-point-before-20261006.md#old-position)._

## Problem map

| Node | Problem | Needs | Status | Best current understanding |
| --- | --- | --- | --- | --- |
| <a id="contract"></a>`contract` | What a legal chart, a legal continuation and a committed prefix are. | None | held | `docs/formulation/notation.md` |
| <a id="response"></a>`response` | How a player responds to a continuation given the past: which legal futures are comfortable, demanding or unplayable. | `contract` | open | Undefined at the baseline ([s-response-undefined](artifacts/lineage-review/synthesis.md#s-response-undefined)). Physiology is in scope, with the corpus plus priors ([d-response-priors](artifacts/evaluation-first.md#d-response-priors)). A demand machine (proposed): [p-demand-machine](artifacts/operator-properties.md#p-demand-machine). What player data can identify (proposed): [a-fable-views](artifacts/player-data.md#a-fable-views). |
| <a id="data"></a>`data` | What the corpus and its style judgments can teach: which arrangements, styles and conditions are present, missing or distorted by sampling. | `contract` | open | The target is the 2–6★ ranked and loved distribution ([d-target-distribution](artifacts/evaluation-first.md#d-target-distribution)). Corpus census: [s-not-ordinary](artifacts/lineage-review/synthesis.md#s-not-ordinary). Style labels available: [s-style-module](artifacts/r2-phasen-and-lens-20261006.md#s-style-module). |
| <a id="time"></a>`time` | How event times are represented, inferred from audio (tempo and phase included) and scored. | `contract`, `data` | open | The lineage had no time coordinate ([s-no-time-coordinate](artifacts/lineage-review/synthesis.md#s-no-time-coordinate), [d-grid-cause](artifacts/lineage-review/synthesis.md#d-grid-cause)). R2 v2 takes head times as given ([d-r2v2-scope](artifacts/r2-v2-stage0-state.md#d-r2v2-scope)). |
| <a id="proposal"></a>`proposal` | A distribution over timed complete rows, given audio and history, that covers real ordinary and expressive arrangements. | `contract`, `data`, `time` | open | R1 fails, and the generator gets no release times ([d-r1-fails](artifacts/r1-verdict.md#d-r1-fails), [d-no-release-input](artifacts/r1-verdict.md#d-no-release-input)). R2 makes one decision per head row and places releases with a pointer ([d-r2-row-decision](artifacts/r2-ln-design.md#d-r2-row-decision), [d-decision-unit-kept](artifacts/r2-phasen-and-lens-20261006.md#d-decision-unit-kept)). R2 v2 phase N is training ([r-phasen-run](artifacts/r2-phasen-and-lens-20261006.md#r-phasen-run)). Architecture: [r2-architecture-20261006](artifacts/r2-architecture-20261006.html). |
| <a id="rollout"></a>`rollout` | Keeping proposal quality on histories the model produced itself, and not only on source histories. | `proposal` | open | Capacity and exposure did not remove the free-running collapse ([s-likelihood-vs-rollout](artifacts/lineage-review/synthesis.md#s-likelihood-vs-rollout)). Row cross-entropy is not enough; preference optimisation comes later ([d-beyond-row-ce](artifacts/r1-verdict.md#d-beyond-row-ce)). For R2 v1, exposure bias comes before representation ([a-r2-fable-judgment](artifacts/r2-average-and-control.md#a-r2-fable-judgment)). |
| <a id="selection"></a>`selection` | Choosing among proposed futures by their response. | `proposal`, `response` | open | Nothing selects at the baseline. Where a selector worked: [s-response-not-in-loop](artifacts/lineage-review/synthesis.md#s-response-not-in-loop). Decoding-time options for R2: [decoding-stage](artifacts/decoding-stage.md). |
| <a id="style"></a>`style` | What a style request asks for, as organisation separate from chart properties, and how it composes with property requests, release and the return to natural generation. | `contract`, `data` | open | Formulation draft on branch `docs/style-formulation` (`8da2bda`), not reviewed by the human. Its open questions: [open-questions](artifacts/style-formulation-rethink.md#open-questions). Decisions so far: [d-formulation-answers](artifacts/r2-style-formulation-check.md#d-formulation-answers), [d-formulation-answers-2](artifacts/r2-style-formulation-check.md#d-formulation-answers-2), [d-target-strength](artifacts/r2-style-formulation-check.md#d-target-strength). The five Lens concepts become condition kinds ([d-r2v2-scope](artifacts/r2-v2-stage0-state.md#d-r2v2-scope)). |
| <a id="control"></a>`control` | Making a scoped request for difficulty, style or long-note amount change the output in the requested way. | `proposal`, `response`, `data`, `style` | open | Only LN amount ever followed a request ([s-controls-order](artifacts/lineage-review/synthesis.md#s-controls-order)); in R2 v1 LN share partly did, star did not ([a-r2-fable-judgment](artifacts/r2-average-and-control.md#a-r2-fable-judgment)). Target interface: [d-final-conditions](artifacts/r2-ln-design.md#d-final-conditions). R2 v2 trains natural structure first, then conditions ([d-two-phase-recipe](artifacts/r2-v2-stage0-state.md#d-two-phase-recipe)). Plan (proposed): [r2-condition-plan-v5](artifacts/r2-condition-plan-v5.md). Style module (proposed): [a-style-module](artifacts/r2-phasen-and-lens-20261006.md#a-style-module). |
| <a id="evaluation"></a>`evaluation` | Telling whether a generated chart is good, with observers that see the distinction in dispute. | `contract`, `response`, `time` | open | Existing evaluators have no measured sensitivity ([s-eval-sensitivity](artifacts/lineage-review/synthesis.md#s-eval-sensitivity)). Working hypothesis: rate, not presence ([h-rate-not-presence](artifacts/evaluation-first.md#h-rate-not-presence)). Design (proposed): [p-eval-object](artifacts/evaluation-first.md#p-eval-object). Operator properties (proposed): [operator-properties](artifacts/operator-properties.md). Paused, to be tuned on R2 output: [d-eval-reserve](artifacts/evaluation-first.md#d-eval-reserve). |
| <a id="realtime"></a>`realtime` | Publishing settled coverage ahead of playback within a latency budget. | `proposal`, `selection` | open | The commit rule is in `docs/formulation/notation.md`, "Provisional branches and prefix commit". Nothing measured. |
| <a id="local-edit"></a>`local-edit` | Regenerating one section between fixed past and future context. | `proposal`, `control` | open | No work; the formulation has no notation for a window bounded by a fixed later chart. |

_Review: agent draft of 2026-09-30, not yet reviewed. The whole map, its edges and its statuses are (proposed). So are two later additions: node `time` with its edges (2026-09-30), and node `style` with the edge `control` → `style` (2026-10-06). Human decisions in the cells are cited at their anchors, where the private sources are given. The earlier cell text is in [old-map](artifacts/entry-point-before-20261006.md#old-map)._

## Current movement

Focus, set by the human on 2026-10-06 ([private, local](artifacts/private/human-inputs/da66cd9e-81c2-4ad1-a989-4fe6ef7c81fe.md#prompt-6)):

1. **Finish R2 v2 phase N and phase C** (`proposal`, `control`).
   - Question: does phase N give a natural generator that passes its guards? Can phase C then make scoped requests change the output inside their scopes, while natural decisions stay as phase N left them?
   - Why now: the recipe, the scope of R2 v2 and the phase-N config are decided ([d-two-phase-recipe](artifacts/r2-v2-stage0-state.md#d-two-phase-recipe), [d-r2v2-scope](artifacts/r2-v2-stage0-state.md#d-r2v2-scope), [d-phasen-memory](artifacts/r2-phasen-and-lens-20261006.md#d-phasen-memory)).
   - Phase N finished at 64M on 2026-10-07 ([o-phasen-finished](artifacts/r2-phasen-and-lens-20261006.md#o-phasen-finished)). `select.py` selects no checkpoint, because every candidate fails guard (iv) on short holds and near-head releases ([o-phasen-no-selection](artifacts/r2-phasen-and-lens-20261006.md#o-phasen-no-selection)). The redirect condition below is met. The human chose 48M with a 60 ms minimum-hold mask at decoding as the provisional base ([d-phasen-base-mask](artifacts/r2-phasen-and-lens-20261006.md#d-phasen-base-mask)). With the mask, generated short holds disappear, but releases 1-40 ms before a head stay at 1.63% against 0.42%, so guard (iv) still fails ([o-minhold-mask](artifacts/r2-phasen-and-lens-20261006.md#o-minhold-mask)). Cause, from an Opus subagent: the LN level is not tied to the chart from the start of a song, and the model's own history amplifies it. Release decisions are calibrated, the representation is not at fault, and the keep hypothesis is refuted ([o-lnlen-cause](artifacts/r2-phasen-and-lens-20261006.md#o-lnlen-cause)). Real charts do contain short holds, so guard (iv) and the 60 ms mask rest on a false premise ([c-no-short-holds-claim](artifacts/r2-phasen-and-lens-20261006.md#c-no-short-holds-claim)). Proposed: a chart-level LN input in phase N and a corrected guard (iv) ([a-lnlen-reading](artifacts/r2-phasen-and-lens-20261006.md#a-lnlen-reading)). A budget split relative to phase N cuts the controller's departure by about a third ([o-c0-allocation](artifacts/r2-phasec-c0-20261007.md#o-c0-allocation)).
   - Waiting on the human: the phase-C decisions ([v5-open](artifacts/r2-condition-plan-v5.md#v5-open)) and the style-module choices ([a-style-module](artifacts/r2-phasen-and-lens-20261006.md#a-style-module)).
   - Handoff of 2026-10-07: [handoff-20261007-phasec](artifacts/handoff-20261007-phasec.md). Phase C stage C0 (a relative attack-strain difficulty proxy, [d-strain-proxy-v1](artifacts/r2-phasec-frontier-20261007.md#d-strain-proxy-v1)) ran as Astra job `20261007-061042-r2-phasec-c0`, code `7985cf9` ([c0-job](artifacts/r2-phasec-c0-20261007.md#c0-job)). Natural r has a median of 1.43 at every length, with a floor of about 1 ([o-c0-natural-r](artifacts/r2-phasec-c0-20261007.md#o-c0-natural-r)). A head-mask tilt moves r as requested ([o-c0-tilt](artifacts/r2-phasec-c0-20261007.md#o-c0-tilt)). The analytic controller attains requests, but its reference-clock allocation departs from natural even at the median ([o-c0-controller](artifacts/r2-phasec-c0-20261007.md#o-c0-controller); proposal in [a-c0-reading](artifacts/r2-phasec-c0-20261007.md#a-c0-reading)). The 48 blind-screen pairs await judging by the human and Astra ([o-c0-screen](artifacts/r2-phasec-c0-20261007.md#o-c0-screen)).
   - Phase C is being redesigned with Astra: the human rejected the difficulty relabel and the whole-song star calculation as the scoped metric, and accepts a skeleton-derived, preferably strain-based short-interval proxy ([d-difficulty-proxy](artifacts/r2-phasec-frontier-20261007.md#d-difficulty-proxy)); Astra's frontier-controller proposal is checked against the code and costs in [r2-phasec-frontier-20261007](artifacts/r2-phasec-frontier-20261007.md#o-phasec-code-audit).
   - Would redirect (proposed): a selected phase-N checkpoint that fails its natural guards, or a frozen phase C that misses the onset guards at every conditioning size tried.
2. **Finalise the style formulation and resolve its conflicts with the existing formulation** (`style`).
   - Question: what do the draft's open questions resolve to, and where does it still disagree with `notation.md`, `gameplay-state.md` and the R2 v2 plan?
   - Why now: phase C and the style module need the meaning of a request fixed.
   - Where it stands: draft `8da2bda`, not reviewed ([open-questions](artifacts/style-formulation-rethink.md#open-questions), [conflicts](artifacts/style-formulation-rethink.md#conflicts)).
   - Would redirect (proposed): a resolution that changes what a property or style request means for phase C reopens the phase-C plan.

The previous focus, `evaluation` (2026-10-02), is paused ([d-eval-reserve](artifacts/evaluation-first.md#d-eval-reserve)). Its account is in [old-movement](artifacts/entry-point-before-20261006.md#old-movement).

_Review: the focus was set by the human on 2026-10-06 ([private, local](artifacts/private/human-inputs/da66cd9e-81c2-4ad1-a989-4fe6ef7c81fe.md#prompt-6)); the questions, the why-now lines and the redirect lines are an agent draft._

## Views

None yet.
