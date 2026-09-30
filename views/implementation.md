# Implementation view

The components that exist, the map node each serves, the variants tried in its place and why they were set aside. Nothing here is a selected V3 architecture: no model is qualified, and every choice below is revisable. Commits resolve with `git show <commit>` in any ensomi-model clone.

## How `proposal` is decomposed today

The proposal is factored as event time, then a complete row at that time. Three owners share it:

- **H** plans head-bearing times from audio and controls.
- **R** times pure releases, aware of open LNs.
- **R1** owns the complete row: cardinality, columns, TAP or LN, and which holds close.

This split is a human decision about ownership, not a result: [scope and ownership](../artifacts/scoped-controls-and-ownership.md). Timing and rows stay coupled through it: [lesson](history.md#l-timing-and-rows-couple).

## Components and their variants

| Component | Serves | In use | Variants tried, and why set aside | Reopen if |
| --- | --- | --- | --- | --- |
| Audio front end | `proposal` | Small Mel encoder (24 kHz, 128 bins, 10-ms hop) trained jointly with timing and rows. | Frozen BeatThis features: head timing transferred, release-only prediction did not, and the human redirected to joint learning. Audio pyramid with causal memory readers: integrated, then regressed native panels. [Timing](../artifacts/timing-and-joint-baseline.md), [memory](../artifacts/player-response-and-memory.md) | A phrasing failure is traced to missing long-range audio relations and not to the fitted H base. |
| Timing, H and R | `proposal` | Native integer-millisecond hazards; H for heads, R for pure releases. | An independent onset and release schedule for a frozen R1: release opportunities are not neutral. A typed skeleton owning counts and types: R1 could not move a tail fixed upstream. A rhythm hierarchy with an internal clock: designed, never implemented. [Timing](../artifacts/timing-and-joint-baseline.md), [ordinary patterns](../artifacts/ordinary-patterns-and-frontier.md) | [Near-duplicate native head times](../artifacts/clean-joint-and-current-state.md#o-native-h-doublets) turn out to drive the short tails. |
| Release decision | `proposal` | On the primary branch, R commits a release before R1 can prefer waiting. | Joint R1 wait and release scoring (`965d670`, release-calibration only): a pilot cut short tails on one witness and failed overall checks. [Ordinary patterns](../artifacts/ordinary-patterns-and-frontier.md) | A longer fit of the joint law is judged worth its cost. |
| Row decoder, R1 | `proposal` | Layout normalised within count families, then a candidate consequence energy (`frontier2`) before normalising across rows. | Segment decoder with latent action plans (`7d31b1e`): all prior selections chose one code, and it [drops the candidate consequence inputs](../artifacts/clean-joint-and-current-state.md#o-unused-consequence-path). Count-only refits: cannot repair release identity or columns. [LN diagnosis](../artifacts/ln-feedback-and-proposal-diagnosis.md) | The consequence path is restored and changes choices. |
| Condition paths | `control` | Independent per-attribute scope encoding; conditions reach H, R and R1. | Additive conditions before an affine head: cancel from relative odds. Multiplicative modulation: removes that limit, mixed native result. LN integral feedback: stabilises amount and adds a prefix-balance bias; the clean recipe dropped it, and deletion alone had been rejected earlier. [Conditional learning](../artifacts/conditional-learning.md), [LN diagnosis](../artifacts/ln-feedback-and-proposal-diagnosis.md) | A request must hold amount without a controller. |
| Selection | `selection` | Default generation and qualification call the session directly; no planner runs. | Private continuation screening (`452abc6`): removed narrow collision witnesses, cannot invent missing support. Response planner with a four-scale action-response state (`02e522d`): refuses when nothing passes, and [its reference cannot see short holds](../artifacts/ordinary-patterns-and-frontier.md#o-25ms-ln-zero-work). [Playback](../artifacts/playback-and-planning.md), [ordinary patterns](../artifacts/ordinary-patterns-and-frontier.md) | `response` gains a definition the planner can score. |
| Publication | `realtime` | Buffered incremental rows with separate head-plan, settled-chart and player clocks. | Speculative models and graceful degradation: deferred, since no measured bottleneck called for them. [Playback](../artifacts/playback-and-planning.md) | A client or browser runtime shows a deadline miss. |
| Evaluation | `evaluation` | `ChartTrace` and `evaluate_scopes`; `run_qualification` from BOS with scoped numerical gates, then a required semantic review. | Single scalars (stars, NLL, LN fraction) as acceptance: each hid a failure the next observer exposed. [Evaluation and its limits](../artifacts/evaluation-and-evidence-boundaries.md) | Not applicable; observers accumulate. |

## Training and inference differ

Training scores factual source histories under teacher forcing. Inference starts from BOS and runs on the model's own history, optionally through a planner, and publishes prefixes. Results under one do not transfer to the other by default: [lesson](history.md#l-execution-is-not-quality), [`rollout`](../RESEARCH.md#rollout). The two latest recipes are the three-arm clean joint comparison and the fresh ordinary expert, both in [completed fits and stopping point](../artifacts/clean-joint-and-current-state.md).

## Where the code lives

| Line of work | Branch and commit at the 2026-09-29 recovery | State |
| --- | --- | --- |
| Primary prototype | `codex/audio-skeleton`, `4a8a478` | Source under `src/ensomi_model/research/joint_audio_continuation/`. |
| Release, response and ordinary-arrangement work | `codex/release-calibration`, `96f84fd` | Unmerged. Holds the joint release law, response planner, segment decoder and fresh expert. |
| Realtime benchmark | `codex/stream-generation-benchmark`, `4ec631e` | Separate startup and coverage contract; integration unqualified. |

Checkpoints, receipts and generated charts are on bings-mac under `artifacts/joint-audio/` in the owning checkout and are not in Git.
