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

The system today is **R1**, a chart continuation model with 3,084,432 parameters. Given a seed prefix of an existing chart, the remaining event times R and the subset H that must carry a note head, it samples one complete four-lane row per time until the song ends. R is the union of the source chart's head times and its release-only times, and R1 reads the next 16 candidates with their roles and the schedule counts up to 64 s ahead, so rhythm, the number of head times, rests and the moments where long notes may end come from the source chart ([s-r1-interface](artifacts/lineage-review/synthesis.md#s-r1-interface)). It does not read audio and has no difficulty input. Its package is `src/ensomi_model/research/bounded_typed_continuation/`; bare file names below are in that package. Preset: `src/ensomi_model/configs/hydra/bounded_typed_r1_response.yaml`. Design and results: `docs/research/bounded_typed_continuation.md`.

| Component | What it does | Where | Checked by |
| --- | --- | --- | --- |
| Chart contract and exact state | Row actions, legality, long-note occupancy and action clocks, all derived by replay. A support mask removes illegal rows and rows that leave no free lane before the next required head. | `contract.py`, `support.py`, `features.py` | Tests |
| Local history encoder | Causal dilated convolutions, width 128, dilations 1 to 128, over the last 511 committed rows in shared hand coordinates. | `temporal.py` | Tests, including crop invariance |
| Seed conditioning | Mean of the encoded seed, added to each hand vector through a residual that starts at zero. | `model.py` | Tests |
| Landmark memory | A GRU of width 256 reads every committed row from the start and stores a landmark every 64 head rows; the current hand vectors attend to earlier landmarks. | `long_memory.py` | Tests |
| Joint row head | Per-hand scores plus a rank-16 coupling between hands, mirror-equivariant, over the 256 complete rows; the support mask is applied before sampling. | `model.py` | Tests |
| Correction residuals | Three score corrections, each trained alone on a frozen parent: by head mask, by release mask, and by the exact consequences of each candidate row up to the second next head (`frontier2`). Their training signal is three hand-written rules applied to model-generated trajectories (a lane in 28 of 32 head rows; three holds unchanged through 12 heads; a same-lane gap under 30 ms), on 80, 72 and 92 harvested queries. `frontier2` is a residual in the policy, not a model of player response. | `routing.py`, `consequence.py`, `response.py`, `recovery.py` | Tests; paired development generations. The rules themselves were checked by agent inspection only |
| Training | Source cross-entropy on bounded windows with teacher forcing, in six stages to 6.75M source-onset exposures on one CPU thread (8,931 updates, about 0.6 passes over the corpus, one seed). Corpus: 11,563 TRAIN charts in 3,169 song groups, of which about 60% are verified ranked 2-6 star. | `data.py`, `corpus.py`, `train_run.py`, `src/ensomi_model/research/r1_restore/`, `scripts/r1-restore.sh`; `docs/research/r1_staged_restoration.md`, `docs/research/r1_training_distribution.md` | Frozen inputs and stage audits |
| Generation and recovery | Sampling at temperature 1 inside the support mask, durable snapshots rebuilt by replay, osu! export, and an independent check of finished outputs. | `generation.py`, `generate_run.py`, `verification.py`, `src/ensomi_model/osu_core/export.py` | Tests; six published examples reproduce byte for byte at the tag |

What is established for R1 as a whole: legal rows, exact replay, and exact osu! export and reparse. Playability, difficulty consistency and musical quality are not established; the baseline document records the candidate as still to refine, and no human judgment of its output is on record. What R1 does with real times, measured in the review: [s-r1-quality](artifacts/lineage-review/synthesis.md#s-r1-quality).

Also on the baseline, outside R1: the comparison arms R0 and O1 in the same package; a 35M-parameter teacher profile and its training queue (`src/ensomi_model/research/vacation_training/`; the teacher finished training, its checkpoints and readouts are outside the repository and no baseline document reports them, see [s-likelihood-vs-rollout](artifacts/lineage-review/synthesis.md#s-likelihood-vs-rollout)); the earlier oracle-time, source-action and scoped-style research packages, from which R1 imports storage, runtime and parsing helpers; a log-Mel audio frontend (`src/ensomi_model/features/mel_base.py`) that no V3 model consumes; and the pre-V3 mapper, timing and control stack, which the baseline README rules out as a reference.

Not in the baseline:

- Event times from audio. R and H come from an existing chart.
- Generation without a seed chart.
- Any request input for difficulty, style or long-note amount.
- Regeneration of a section between fixed past and future context.
- Measured publication ahead of playback; no client or service.
- A validated evaluator of chart quality. Checks are mechanical, plus inspection against Beatmap Lens examples.

_Review: baseline and section scope set by the human on 2026-09-30 ([private, local](artifacts/private/human-inputs/0e81052c-94ac-435b-92a1-44c2f5d4be7f.md#prompt-1)); the lines are an agent draft of the same date, not yet reviewed. They were read from the baseline documents and code; no test or generation was re-run. Later the same day five lines were corrected to match the baseline code and artifacts, from the lineage review, and an audit then checked all five against the code ([audit, 4a](artifacts/lineage-review/fable/synthesis-audit.md)): what R contains and what R1 reads ahead, no difficulty input, what the correction residuals were trained on and how that was checked, the training budget and corpus share, and the state of the 35M teacher._

## Problem map

| Node | Problem | Needs | Status | Best current understanding |
| --- | --- | --- | --- | --- |
| <a id="contract"></a>`contract` | What a legal chart, a legal continuation and a committed prefix are. | None | held | `docs/formulation/notation.md` |
| <a id="response"></a>`response` | How a player responds to a continuation given the past: which legal futures are comfortable, demanding or unplayable. | `contract` | open | `docs/formulation/gameplay-state.md`, "Target response and frontier", names the function and leaves its quantities and comparison rules undefined. Nothing built since is that response or was checked against a judgment, and ranked charts contain what the human rejected ([s-ranked-contains-rejected](artifacts/lineage-review/synthesis.md#s-ranked-contains-rejected)): [s-response-undefined](artifacts/lineage-review/synthesis.md#s-response-undefined), [s-response-blind](artifacts/lineage-review/synthesis.md#s-response-blind). |
| <a id="data"></a>`data` | What the corpus and its style judgments can teach: which arrangements, styles and conditions are present, missing or distorted by sampling. | `contract` | open | `docs/formulation/gameplay-state.md`, "Style observations"; `docs/research/r1_training_distribution.md`. Corpus census and how the lineage sampled it: [s-not-ordinary](artifacts/lineage-review/synthesis.md#s-not-ordinary). |
| <a id="proposal"></a>`proposal` | A distribution over timed complete rows, given audio and history, that covers real ordinary and expressive arrangements. | `contract`, `data`, `time` | open | R1 chooses rows at supplied times, and those times include the source's release moments ([s-r1-interface](artifacts/lineage-review/synthesis.md#s-r1-interface)). Nothing on the baseline chooses times from audio. The lineage's audio models were too little trained to judge ([s-undertrained](artifacts/lineage-review/synthesis.md#s-undertrained)); what separates generated long notes from ranked ones is release timing and LN allocation ([s-ln-open-state](artifacts/lineage-review/synthesis.md#s-ln-open-state)). `docs/research/bounded_typed_continuation.md` |
| <a id="time"></a>`time` | How event times are represented, inferred from audio (tempo and phase included) and scored. | `contract`, `data` | open | The contract allows internal beat coordinates (`docs/formulation/notation.md`, "Chart object and time"). Ranked charts sit on their grid; the lineage's generated heads do not (pooled at chance, up to twice chance by song); real head times did not reduce short holds in three saved cases; releases just before another head are the clearest timing-side defect; no experiment separates representation from training: [s-no-time-coordinate](artifacts/lineage-review/synthesis.md#s-no-time-coordinate), [d-grid-cause](artifacts/lineage-review/synthesis.md#d-grid-cause). The pre-V3 grid fitter is mostly a constant 29 ms off ([s-pre-v3](artifacts/lineage-review/synthesis.md#s-pre-v3)). |
| <a id="rollout"></a>`rollout` | Keeping proposal quality on histories the model produced itself, and not only on source histories. | `proposal` | open | R1's correction residuals target this. `docs/research/bounded_typed_continuation.md`, "Native-prefix recovery training" and "Native response recovery result (6.75M)". In the one scale test on the row half, capacity and exposure did not remove the free-running collapse that R1's rule-trained residuals remove: [s-likelihood-vs-rollout](artifacts/lineage-review/synthesis.md#s-likelihood-vs-rollout). |
| <a id="selection"></a>`selection` | Choosing among proposed futures by their response. | `proposal`, `response` | open | `docs/formulation/notation.md`, "Provisional branches and prefix commit", leaves the rule to the implementation. R1 samples one future and selects nothing. A time-horizon selector worked where its cost saw the defect, and was never part of a played or qualified system; a per-gap rarity preference was: [s-response-not-in-loop](artifacts/lineage-review/synthesis.md#s-response-not-in-loop). |
| <a id="control"></a>`control` | Making a scoped request for difficulty, style or long-note amount change the output in the requested way. | `proposal`, `response`, `data` | open | `docs/formulation/gameplay-state.md`, "Controls". Only LN amount ever followed a request, through an inference-time controller; difficulty and style did not: [s-controls-order](artifacts/lineage-review/synthesis.md#s-controls-order). |
| <a id="evaluation"></a>`evaluation` | Telling whether a generated chart is good, with observers that see the distinction in dispute. | `contract`, `response`, `time` | open | `docs/formulation/gameplay-state.md`, "Evaluation questions"; `docs/research/bounded_typed_continuation.md`, "Quality coverage required beyond mechanical verification". Existing evaluators have a false-alarm rate and no measured sensitivity to what the human rejects: [s-eval-sensitivity](artifacts/lineage-review/synthesis.md#s-eval-sensitivity). |
| <a id="realtime"></a>`realtime` | Publishing settled coverage ahead of playback within a latency budget. | `proposal`, `selection` | open | The commit rule is in `docs/formulation/notation.md`, "Provisional branches and prefix commit". No measurement on the baseline. |
| <a id="local-edit"></a>`local-edit` | Regenerating one section between fixed past and future context. | `proposal`, `control` | open | No work; the formulation has no notation for a window bounded by a fixed later chart. |

_Review: agent draft 2026-09-30, not yet reviewed. The whole map, its edges and its statuses are (proposed). Node ids are carried over from before the reset; `evaluation` and `realtime` moved from held to open because the grounds for held were outside the baseline. Added after the lineage review, also (proposed): the node `time`, the edges from `proposal` and `evaluation` to `time`, and the review links in the last column._

## Current movement

Focus, set by the human on 2026-09-30: distil what is useful from the `codex/audio-skeleton` lineage since `r1-restored-6.75m`, whose experiments mostly failed and probably began from wrong directions ([private, local](artifacts/private/human-inputs/c7186900-b717-412f-a49e-d277b65ed0c4.md#prompt-2)). The lineage is not a basis for position. Its code is at tag `audio-joint-2026-09`, its earlier notes at tag `relay-notes-audio-joint-2026-09`.

Three keys the human kept for the work ahead ([private, local](artifacts/private/human-inputs/c7186900-b717-412f-a49e-d277b65ed0c4.md#answer-1); evidence in the [feedback index](artifacts/audio-skeleton-human-feedback-index.md)):

1. A player response state is the novel core relative to `ref-proj/`: player profile, stimulus-response, style readout and difficulty modelled together, as in the V3 formulation.
2. The proposal distribution is not the ordinary ranked one; learn ordinary 2-6 star charts first, then add expressive range.
3. Evaluation is extremely important. No evaluation design is chosen.

Dropped as keys: ownership of hold versus release for LN, and a hierarchical end-to-end audio skeleton.

**Done: the skeptical review of the lineage** the human asked for. Nine Astra slices, eight independent Opus reviews covering every slice, and an audit of the first synthesis against sources; reports and how the runs were set up in [lineage-review](artifacts/lineage-review/README.md#state); the main thread's reading, second version, in [synthesis](artifacts/lineage-review/synthesis.md). What it says, one line each, grounds behind the links:

- R1 fills lanes into a real mapper's rhythm, release moments included; the lineage stopped using R1 itself within hours ([s-r1-interface](artifacts/lineage-review/synthesis.md#s-r1-interface), [s-release-sensitivity](artifacts/lineage-review/synthesis.md#s-release-sensitivity), [s-lineage-r1-is-not-r1](artifacts/lineage-review/synthesis.md#s-lineage-r1-is-not-r1)).
- No audio model was trained enough to judge its architecture; scale was never a controlled variable; "ordinary first" was never actually run ([s-undertrained](artifacts/lineage-review/synthesis.md#s-undertrained), [s-not-ordinary](artifacts/lineage-review/synthesis.md#s-not-ordinary)).
- On the row half, a 35M model without the correction residuals kept the fixed-lane collapse that small R1 shows before its corrections (16 charts, one seed) ([s-likelihood-vs-rollout](artifacts/lineage-review/synthesis.md#s-likelihood-vs-rollout)).
- Ranked charts contain things the human rejected in generated ones, so "ordinary" is not corpus membership ([s-ranked-contains-rejected](artifacts/lineage-review/synthesis.md#s-ranked-contains-rejected)).
- The lineage had no musical time coordinate and its head times are far from the song's grid; the project had recorded the same lesson before V3. Whether that causes the bad charts is open ([s-no-time-coordinate](artifacts/lineage-review/synthesis.md#s-no-time-coordinate), [d-grid-cause](artifacts/lineage-review/synthesis.md#d-grid-cause)).
- Generated long notes differ from ranked ones in releases placed just before another head, in holds under 40 ms, and in too many LN on dense heads; not in ending at the next head ([s-ln-open-state](artifacts/lineage-review/synthesis.md#s-ln-open-state)).
- No time-horizon response scorer was in the system the human played, only a per-gap rarity preference; the scorers are blind to how often and how long; the response target was never defined ([s-response-not-in-loop](artifacts/lineage-review/synthesis.md#s-response-not-in-loop), [s-response-blind](artifacts/lineage-review/synthesis.md#s-response-blind), [s-response-undefined](artifacts/lineage-review/synthesis.md#s-response-undefined)).
- The evaluators were never measured against what the human rejects ([s-eval-sensitivity](artifacts/lineage-review/synthesis.md#s-eval-sensitivity)).
- Only LN amount ever followed a request, through a hand-coded controller at inference; difficulty and style never worked on free-running output ([s-controls-order](artifacts/lineage-review/synthesis.md#s-controls-order)).
- The search was mostly local and upstream moves mostly followed the human; the reviewers name different sharper causes ([s-local-search](artifacts/lineage-review/synthesis.md#s-local-search)).
- Reference generators differ in scale, a time grid at output and output repair; their quality on these songs is unknown. The pre-V3 system made charts from audio badly at half a pass of training; its timing stack is mostly a constant 29 ms off and no reason for setting it aside is recorded ([s-outside](artifacts/lineage-review/synthesis.md#s-outside), [s-pre-v3](artifacts/lineage-review/synthesis.md#s-pre-v3)).

**Waiting for the human.** Questions the review could not settle, each with its evidence and options behind the link. H11 comes first because H1 to H3 depend on it.

| Id | Question |
| --- | --- |
| [H11](artifacts/lineage-review/synthesis.md#h11) | Is the standard the ranked corpus, or stricter? Would the ranked source charts of the panel, and a ranked 4 star chart with a 28-attack anchor, be accepted as ordinary? |
| [H1](artifacts/lineage-review/synthesis.md#h1) | What "ordinary" means (star band, ranked only, which styles excluded), and whether it applies first to the row half with real times or to the whole audio-to-chart system. |
| [H2](artifacts/lineage-review/synthesis.md#h2) | Which response state key 1 means: the formulation's canonical response from mapper evidence, physiology excluded, or a hand-load model calibrated on physiology and corpus. If the reference is corpus rarity, key 1 has no target independent of key 2 ([d-key1-vs-key2](artifacts/lineage-review/synthesis.md#d-key1-vs-key2)). |
| [H3](artifacts/lineage-review/synthesis.md#h3) | Whether a set of the human's own accept and reject judgments, kept as files, comes before any new scorer or evaluator, and how much judging the human will do. Does an LN-amount request also govern where the LN go? Which part of complaint B is the defect? |
| [H12](artifacts/lineage-review/synthesis.md#h12) | Which controls the first system must have; whether an inference-time feedback controller counts as a control; what a scoped difficulty means; whether style is out of scope for now. |
| [H4](artifacts/lineage-review/synthesis.md#h4) | Time coordinate: whether to test a musical coordinate with residual offsets, and whether BeatThis or local tempo and phase are wanted as inputs. The evidence does not show it fixes the charts. |
| [H5](artifacts/lineage-review/synthesis.md#h5) | A long note with a duration at birth is an untested alternative; is it worth a test. Touches a dropped key. |
| [H6](artifacts/lineage-review/synthesis.md#h6) | Run an existing generator on the same songs as a yardstick? Does the legacy boundary cover the timing stack as a component or baseline? Merge `cleanup/main-r1` before deciding? |
| [H7](artifacts/lineage-review/synthesis.md#h7) | Is "R1 chooses rows at supplied times" an acceptable position when the times include release moments? Was the 09-21 approval about quality? What was reported about the 35M teacher? |
| [H8](artifacts/lineage-review/synthesis.md#h8) | "No hard-coded rules" against the ranked minimum gaps: support, learned target, or evaluation only? |
| [H9](artifacts/lineage-review/synthesis.md#h9) | Method and budget: small runs or scale, which machine and how long, may an agent initiate a change of direction. |
| [H10](artifacts/lineage-review/synthesis.md#h10) | Tooling: relay hooks on the mac apply to Astra worker jobs; Codex quota used; notes branch not pushed. |

Still open between reviewers: how local the agent's search was, by count ([d-locality-counts](artifacts/lineage-review/synthesis.md#d-locality-counts)). The first synthesis was audited and corrected in nine places where it named causes more strongly than the sources allow; what changed is in [synthesis, section 9](artifacts/lineage-review/synthesis.md#revision-2). The two second-round Opus reviews were not audited.

What would redirect the work: the human's answers to H11 and H1 to H3 decide what is built or measured first. The reviewers named cheap checks that were never run ([synthesis, section 7](artifacts/lineage-review/synthesis.md#checks-not-run)); none is authorised.

The baseline documents leave one review open: the quality of the restored R1 across whole charts, difficulty levels and seeds (`docs/research/r1_staged_restoration.md`, "Interpretation and provenance"). The review adds first measurements, not a judgment ([s-r1-quality](artifacts/lineage-review/synthesis.md#s-r1-quality)).

Running work: none. All eighteen reviewers have returned; no job is live on either machine. Session `2a66b88f` closed out at the context limit after this update; the next session starts from here.

_Review: agent draft 2026-09-30, not yet reviewed. The reset, the focus and the review task are the human's instructions ([private, local](artifacts/private/human-inputs/0e81052c-94ac-435b-92a1-44c2f5d4be7f.md#prompt-1), [focus](artifacts/private/human-inputs/c7186900-b717-412f-a49e-d277b65ed0c4.md#prompt-2), [review](artifacts/private/human-inputs/05329633-5f3b-4149-b6da-8c599388a7c5.md#prompt-1), [workers](artifacts/private/human-inputs/2a66b88f-f40f-4219-a4c8-3f46bc5511ae.md#prompt-1), [supplement and audit](artifacts/private/human-inputs/2a66b88f-f40f-4219-a4c8-3f46bc5511ae.md#prompt-3)); the three keys and their wording are the human's answer. The reading of the review, the choice of six problems for independent review, and every line of the table are the agent's. Nothing in the table is decided._

## Views

None yet.
