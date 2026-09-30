# Ensomi V3 research

Code and document paths resolve at the baseline commit: `git show 5c56e28:<path>` in any ensomi-model clone. Section rules and the review policy are in the research-relay convention, "Entry point".

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

Baseline: `main` at `5c56e28`, with the released model at tag `r1-restored-6.75m` (`8f13103`, the same code before the package rename; weights `sed-i/pulsefield-r1-restored` on Hugging Face).

The system today is **R1**, a chart continuation model with 3,084,432 parameters. Given a seed prefix of an existing chart, the remaining event times R and the subset H that must carry a note head, it samples one complete four-lane row per time until the song ends. It does not read audio. Its package is `src/ensomi_model/research/bounded_typed_continuation/`; bare file names below are in that package. Preset: `src/ensomi_model/configs/hydra/bounded_typed_r1_response.yaml`. Design and results: `docs/research/bounded_typed_continuation.md`.

| Component | What it does | Where | Checked by |
| --- | --- | --- | --- |
| Chart contract and exact state | Row actions, legality, long-note occupancy and action clocks, all derived by replay. A support mask removes illegal rows and rows that leave no free lane before the next required head. | `contract.py`, `support.py`, `features.py` | Tests |
| Local history encoder | Causal dilated convolutions, width 128, dilations 1 to 128, over the last 511 committed rows in shared hand coordinates. | `temporal.py` | Tests, including crop invariance |
| Seed conditioning | Mean of the encoded seed, added to each hand vector through a residual that starts at zero. | `model.py` | Tests |
| Landmark memory | A GRU of width 256 reads every committed row from the start and stores a landmark every 64 head rows; the current hand vectors attend to earlier landmarks. | `long_memory.py` | Tests |
| Joint row head | Per-hand scores plus a rank-16 coupling between hands, mirror-equivariant, over the 256 complete rows; the support mask is applied before sampling. | `model.py` | Tests |
| Correction residuals | Three score corrections, each trained alone on a frozen parent: by head mask, by release mask, and by the exact consequences of each candidate row up to the second next head (`frontier2`). Their training signal is machine preferences computed on model-generated trajectories. | `routing.py`, `consequence.py`, `response.py`, `recovery.py` | Tests; paired development generations |
| Training | Source cross-entropy on bounded windows with teacher forcing, in six stages to 6.75M source-onset exposures on one CPU thread. Corpus: 11,563 TRAIN charts in 3,169 song groups. | `data.py`, `corpus.py`, `train_run.py`, `src/ensomi_model/research/r1_restore/`, `scripts/r1-restore.sh`; `docs/research/r1_staged_restoration.md`, `docs/research/r1_training_distribution.md` | Frozen inputs and stage audits |
| Generation and recovery | Sampling at temperature 1 inside the support mask, durable snapshots rebuilt by replay, osu! export, and an independent check of finished outputs. | `generation.py`, `generate_run.py`, `verification.py`, `src/ensomi_model/osu_core/export.py` | Tests; six published examples reproduce byte for byte at the tag |

What is established for R1 as a whole: legal rows, exact replay, and exact osu! export and reparse. Playability, difficulty consistency and musical quality are not established; the baseline document records the candidate as still to refine.

Also on the baseline, outside R1: the comparison arms R0 and O1 in the same package; a 35M-parameter teacher profile and its training queue (`src/ensomi_model/research/vacation_training/`); the earlier oracle-time, source-action and scoped-style research packages, from which R1 imports storage, runtime and parsing helpers; a log-Mel audio frontend (`src/ensomi_model/features/mel_base.py`) that no V3 model consumes; and the pre-V3 mapper, timing and control stack, which the baseline README rules out as a reference.

Not in the baseline:

- Event times from audio. R and H come from an existing chart.
- Generation without a seed chart.
- Any request input for difficulty, style or long-note amount.
- Regeneration of a section between fixed past and future context.
- Measured publication ahead of playback; no client or service.
- A validated evaluator of chart quality. Checks are mechanical, plus inspection against Beatmap Lens examples.

_Review: baseline and section scope set by the human on 2026-09-30 ([private, local](artifacts/private/human-inputs/0e81052c-94ac-435b-92a1-44c2f5d4be7f.md#prompt-1)); the lines are an agent draft of the same date, not yet reviewed. They were read from the baseline documents and code; no test or generation was re-run._

## Problem map

| Node | Problem | Needs | Status | Best current understanding |
| --- | --- | --- | --- | --- |
| <a id="contract"></a>`contract` | What a legal chart, a legal continuation and a committed prefix are. | None | held | `docs/formulation/notation.md` |
| <a id="response"></a>`response` | How a player responds to a continuation given the past: which legal futures are comfortable, demanding or unplayable. | `contract` | open | `docs/formulation/gameplay-state.md`, "Target response and frontier", names the function and leaves its quantities and comparison rules undefined. |
| <a id="data"></a>`data` | What the corpus and its style judgments can teach: which arrangements, styles and conditions are present, missing or distorted by sampling. | `contract` | open | `docs/formulation/gameplay-state.md`, "Style observations"; `docs/research/r1_training_distribution.md` |
| <a id="proposal"></a>`proposal` | A distribution over timed complete rows, given audio and history, that covers real ordinary and expressive arrangements. | `contract`, `data` | open | R1 answers the half that chooses rows at supplied times. Nothing on the baseline chooses times from audio. `docs/research/bounded_typed_continuation.md` |
| <a id="rollout"></a>`rollout` | Keeping proposal quality on histories the model produced itself, and not only on source histories. | `proposal` | open | R1's correction residuals target this. `docs/research/bounded_typed_continuation.md`, "Native-prefix recovery training" and "Native response recovery result (6.75M)" |
| <a id="selection"></a>`selection` | Choosing among proposed futures by their response. | `proposal`, `response` | open | `docs/formulation/notation.md`, "Provisional branches and prefix commit", leaves the rule to the implementation. R1 samples one future and selects nothing. |
| <a id="control"></a>`control` | Making a scoped request for difficulty, style or long-note amount change the output in the requested way. | `proposal`, `response`, `data` | open | `docs/formulation/gameplay-state.md`, "Controls" |
| <a id="evaluation"></a>`evaluation` | Telling whether a generated chart is good, with observers that see the distinction in dispute. | `contract`, `response` | open | `docs/formulation/gameplay-state.md`, "Evaluation questions"; `docs/research/bounded_typed_continuation.md`, "Quality coverage required beyond mechanical verification" |
| <a id="realtime"></a>`realtime` | Publishing settled coverage ahead of playback within a latency budget. | `proposal`, `selection` | open | The commit rule is in `docs/formulation/notation.md`, "Provisional branches and prefix commit". No measurement on the baseline. |
| <a id="local-edit"></a>`local-edit` | Regenerating one section between fixed past and future context. | `proposal`, `control` | open | No work; the formulation has no notation for a window bounded by a fixed later chart. |

_Review: agent draft 2026-09-30, not yet reviewed. The whole map, its edges and its statuses are (proposed). Node ids are carried over from before the reset; `evaluation` and `realtime` moved from held to open because the grounds for held were outside the baseline._

## Current movement

(proposed) No focus is chosen. On 2026-09-30 the human reset these notes to the baseline above. The audio-conditioned joint lineage of 2026-09-23 to 2026-09-29 closed without a qualified model and is not a basis for position or for the map. Its code is at tag `audio-joint-2026-09` and its notes at tag `relay-notes-audio-joint-2026-09`; read them only when the human asks.

The baseline documents leave one review open: the quality of the restored R1 across whole charts, difficulty levels and seeds (`docs/research/r1_staged_restoration.md`, "Interpretation and provenance").

Running work: none. `ens ps` showed no live job on bings-mac on 2026-09-30.

_Review: agent draft 2026-09-30, not yet reviewed. The reset is the human's decision ([private, local](artifacts/private/human-inputs/0e81052c-94ac-435b-92a1-44c2f5d4be7f.md#prompt-1)); the absence of a focus is (proposed)._

## Views

None yet.
