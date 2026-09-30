# Ensomi V3 research

A reference written `commit:path` resolves with `git show` in any ensomi-model clone. Section rules and the review policy are in the research-relay convention, "Entry point".

## Vision

Connect listening to music, playing it and creating for it by changing how music and choreography are joined. ensomi does not start from a new four-key rule set. Three scenarios carry the vision:

- **Ambient play.** Music is already playing around you. ensomi recognises it, aligns to the current position, and a chart generated in real time follows the music without interrupting it.
- **Mapping.** A mapper selects one section and asks for candidate choreography while the surrounding chart stays in place. Candidates are compared, kept, retried or undone. Style and existing charts steer them. Choice and expression stay with the mapper.
- **Training.** A player picks music they like and a practice target. Verified conditions shape the arrangement around the movement to practise.

Standing constraints:

- A capability is claimed only once verified: no condition is offered as a control before it works, and latency is quoted only as measured.
- Complete audio may be used in training and inference; chart decisions and publication stay causal.
- Output is playable 4K over 2–6★, covering ordinary organisation and expressive styles.
- [Human decisions that bound the model work](artifacts/intent-and-human-direction.md).

_Review: agent draft 2026-09-30, not yet reviewed. Summarised from the source the human designated ([private, local](artifacts/private/human-inputs/7657b130-e91d-453a-a46c-8d5aa017b74e.md#prompt-2)): ensomi-web on bings-mac, outside Git, files `docs/product-direction.md`, `src/config/scenarios.json`, `src/pages/about.astro`._

## Position

| Capability the vision asks for | Implemented and verified today | Evidence | Grounds |
| --- | --- | --- | --- |
| Playable, musically coherent charts | Not reached. No model is qualified: every final candidate failed numerical qualification, and human inspection keeps finding sustained jacks, little breathing and irregular short LNs. | Receipts and checkpoint hashes rechecked 2026-09-29; human inspection | [Completed fits and stopping point](artifacts/clean-joint-and-current-state.md) |
| Generation from audio alone | Works mechanically: generation from BOS with no seed chart, with exact replay, export and reparse. | Mechanical checks | [First joint model](artifacts/timing-and-joint-baseline.md) |
| Publication ahead of playback | Incremental rows with settled coverage met their deadlines on the measured workloads, on an Apple M5 with a resident model. No client or browser runtime exists. | Measurement | [Playback and planning](artifacts/playback-and-planning.md) |
| Conditions over declared ranges (difficulty, style, LN amount) | Scoped requests are plumbed and replay exactly. LN amount followed a request only under a feedback controller that the latest recipe removed. Difficulty response is weak and style requests do not change output reliably. | Paired native generation | [Scope and ownership](artifacts/scoped-controls-and-ownership.md), [conditional learning](artifacts/conditional-learning.md) |
| Section-local candidates for mapping | Nothing verified. No material in these notes studies regeneration between fixed past and future context. | None | None |
| Recognition and alignment of ambient music | Nothing in this repository; it belongs to the client and web side. | None | None |
| Knowing whether a chart is good | Scoped observers and a qualification runner catch the known failures and block promotion. They cannot certify playability. | Tests; recorded counterexamples | [Evaluation and its limits](artifacts/evaluation-and-evidence-boundaries.md) |

_Review: agent draft 2026-09-30, not yet reviewed. Lines restate the 2026-09-29 recovery; nothing was re-run._

## Problem map

| Node | Problem | Needs | Status | Best current understanding |
| --- | --- | --- | --- | --- |
| <a id="contract"></a>`contract` | What a legal chart, a legal continuation and a committed prefix are. | None | held | `4a8a478:docs/formulation/notation.md` |
| <a id="response"></a>`response` | How a player responds to a continuation given the past: which legal futures are comfortable, demanding or unplayable, across acute transitions, sustained load, holding and coordination. | `contract` | open | The formulation names the function without defining it. [A 25-ms LN costs nothing under the current reference](artifacts/ordinary-patterns-and-frontier.md#o-25ms-ln-zero-work). |
| <a id="data"></a>`data` | What the corpus can teach: which arrangements, styles and conditions are present, excluded, or distorted by sampling and missing labels. | `contract` | open | [Sparse semantic coverage](artifacts/conditional-learning.md), [support and exposure](artifacts/broader-proposal-learning.md) |
| <a id="proposal"></a>`proposal` | A distribution over timed complete rows, given audio and history, that covers real ordinary and expressive arrangements. | `contract`, `data` | open | [Near-duplicate native head times](artifacts/clean-joint-and-current-state.md#o-native-h-doublets), [unused consequence path](artifacts/clean-joint-and-current-state.md#o-unused-consequence-path). Decomposition into timing and rows: [implementation view](views/implementation.md). |
| <a id="rollout"></a>`rollout` | Keeping proposal quality on histories the model produced itself, and not only on source histories. | `proposal` | open | [Full audio and transfer](artifacts/full-audio-and-r1-transfer.md), [response and memory](artifacts/player-response-and-memory.md) |
| <a id="selection"></a>`selection` | Choosing among proposed futures by their response. | `proposal`, `response` | open | [A selector cannot rank futures its score does not see](artifacts/ln-feedback-and-proposal-diagnosis.md) |
| <a id="control"></a>`control` | Making a scoped request for difficulty, style or LN amount change the output in the requested way. | `proposal`, `response`, `data` | open | [Scope and ownership](artifacts/scoped-controls-and-ownership.md), [conditional learning](artifacts/conditional-learning.md) |
| <a id="evaluation"></a>`evaluation` | Telling whether a generated chart is good, with observers that see the distinction in dispute. | `contract`, `response` | held | [Evaluation and its limits](artifacts/evaluation-and-evidence-boundaries.md) |
| <a id="realtime"></a>`realtime` | Publishing settled coverage ahead of playback within a latency budget. | `proposal`, `selection` | held | [Playback and planning](artifacts/playback-and-planning.md) |
| <a id="local-edit"></a>`local-edit` | Regenerating one section between fixed past and future context. | `proposal`, `control` | open | No work yet; the node comes from the mapping scenario. |

_Review: agent draft 2026-09-30, not yet reviewed. The whole map, its edges and its statuses are (proposed)._

## Current movement

(proposed) The human has not chosen a focus since the 2026-09-29 recovery. Three bounded questions are carried over from it, each with a concrete witness, as candidates to check before another long fit or a fusion:

- `proposal`: do the near-duplicate native head times produce the extreme short tails? [Witness](artifacts/clean-joint-and-current-state.md#o-native-h-doublets).
- `proposal`: does restoring the candidate-time consequence path to the segment decoder change its choices? [Witness](artifacts/clean-joint-and-current-state.md#o-unused-consequence-path).
- `response`: what must the response see so that an isolated 25-ms LN is not free? [Witness](artifacts/ordinary-patterns-and-frontier.md#o-25ms-ln-zero-work). A proposal to assess: `96f84fd:docs/research/event_time_gameplay_response_zh.md`.

What would redirect the work is not yet stated; set it when the focus is chosen.

Running work: none was found at the recovery check on 2026-09-29, and it has not been rechecked. The interrupted continuous-context fit is not to be restarted mechanically. Code state at recovery: `4a8a478` (primary), `96f84fd` (`codex/release-calibration`, unmerged), `4ec631e` (realtime benchmark).

_Review: agent draft 2026-09-30, not yet reviewed. The focus is (proposed)._

## Views

- [History](views/history.md): the questions asked in order, what each taught, the lessons that outlived them, and the provenance of these notes.
- [Formulation](views/formulation.md): each map node against the V3 formulation, and what the formulation leaves undefined.
- [Implementation](views/implementation.md): the components in use, the variants tried for each, why they were set aside, and where the code lives.
