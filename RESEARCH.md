# Playable audio-to-chart generation

The goal is a musically coherent, playable 4-key chart generated from complete
audio, with usable chart coverage published ahead of playback. The work began
with released R1 and questions about its timing skeleton, source-free opening,
future controls and dense-passage service. R1's internals were explicitly
revisable. The refined target is playable 2–6★ output, including ordinary
approximately-four-star organization and expressive Tech, Jack, LN and other
styles, with independently scoped difficulty/style/LN controls.

Complete audio may be used in both training and inference. Generated history
and published rows remain causal; open LN tails may be supplied later. Rows
need direct audio, and R1 owns cardinality, columns, TAP/LN and complete-row
release choices. Source note placement supplies supervision; redlines and a
unique beat grid are not oracle inputs. Controls concern the requested range,
not identical pressure or composition at every prefix. The
[human-direction account](artifacts/intent-and-human-direction.md) preserves
how these constraints were clarified and which hypotheses remain tentative.

## Where the research actually stands

There are working source-free generation and publication prototypes, reproducible
learning experiments, meaningful partial improvements and reusable evaluation.
There is **no qualified generally playable model**. User inspection repeatedly
found sustained jacks, insufficient breathing, irregular short LNs and weak
ordinary TAP/LN organization after favorable scalar results.

The primary product document stops at step 512 of a clean three-initialization
comparison, but that run actually completed 4,096 updates. Its final inherited,
early and fresh arms each completed 28 native cases and failed 19, 18 and 17
numerical cases. All remain unpromoted; full final semantic review is incomplete.
The separate fresh ordinary expert fitted sixteen units from four songs for
512 updates. It learned some source-conditioned held/TAP roles, but all eight
BOS outputs failed the extreme-short-LN reference. It did not enter fusion.
Both conclusions are supported by receipts and checkpoint hashes rechecked
during this recovery. [Completed fits and exact stopping point](artifacts/clean-joint-and-current-state.md).

The most recent concrete finding is narrower than absence of time awareness:
H/R1 already receive elapsed-time information, but the new segment decoder
does not use the supplied candidate `local`/`timing` consequences. The independent
response state has exponential time scales, yet its scalar work reference can
accept an isolated 25-ms LN with zero cost. Native H also produces near-duplicate
events associated with many extreme short tails. Proposal timing, action consequences and
response discrimination remain separate unresolved questions.

## Why the work took this path

Read the connection relevant to the question; this is not an instruction to
replay every old experiment or execute its next-step list.

| Research question | What was learned and where to follow it |
| --- | --- |
| Could audio supply a neutral onset/release schedule for fixed R1? | Added release opportunities materially changed LN composition; alternate real charts made release-only targets ambiguous. [Timing and first joint model](artifacts/timing-and-joint-baseline.md) |
| Would full audio or restored R1 initialization solve native drift? | Wider data and bounded history helped some failures; the transfer omitted final `frontier2`, so it was not the released R1 policy. [Full audio and transfer identity](artifacts/full-audio-and-r1-transfer.md) |
| Could the system deliver useful coverage in time? | Buffered publication and private continuation screening worked on measured workloads; quality and profile control still failed. [Playback and planning](artifacts/playback-and-planning.md) |
| Which module should own difficulty, counts and release choices? | The typed skeleton restricted R1's ability to respond; human feedback restored complete-row ownership and separate control scopes. [Scope and ownership](artifacts/scoped-controls-and-ownership.md) |
| Could conditional/source/outcome learning repair the actor? | Specific condition-path defects and useful calibration gains were found; semantic control and native organization remained weak. [Conditional learning](artifacts/conditional-learning.md) |
| Would time-based selection or audio/history attention fix pressure and phrasing? | Sustained-load selection helped its covered channel; bank learning and a larger joint-memory fit regressed native panels. [Response and memory](artifacts/player-response-and-memory.md) |
| Were LN errors just amount feedback or forced deadlines? | Feedback adds prefix bias but also stabilizes amount. Most inspected short tails were optional; attack-only selection was blind to most LN alternatives. [LN/proposal diagnosis](artifacts/ln-feedback-and-proposal-diagnosis.md) |
| Would broader learning, scope state, recovered support or H modulation suffice? | Routing/quantity gains coexisted with new failures. Some H plans made low difficulty impossible; missing-condition exposure was distorted. [Broader proposal learning](artifacts/broader-proposal-learning.md) |
| Could release timing and an operational frontier support ordinary arrangements? | Joint wait/release scoring, independent response state and segment plans were implemented; short-tail and selection failures persisted, and one context fit aborted. [Ordinary patterns and frontier](artifacts/ordinary-patterns-and-frontier.md) |
| What completed after the apparent handoff? | The three-arm 4,096-step endpoints, fresh-expert failure, unused consequence path and terminal process state are preserved. [Current evidence](artifacts/clean-joint-and-current-state.md) |

## Conclusions that remain useful

1. **Execution and quality are different achievements.** Exact replay/export,
   legitimate joint likelihood and fast publication do not establish musical
   organization. The [evaluation account](artifacts/evaluation-and-evidence-boundaries.md)
   distinguishes mechanical checks, native diagnostics, read images and human
   judgment, and explains the counterexamples each observer now detects.
2. **Timing and rows constrain each other without sharing every responsibility.**
   Required H times set unavoidable attack load; R1 can still misuse valid H
   through chord, release and column choices. Source-H substitutions do not
   allocate a universal percentage of blame. A release penalty applied only
   after a forced R event may be too late to prefer waiting.
3. **Available information is not necessarily effective information.** Shared
   additive conditions can cancel from important relative odds, a constant
   latent can block plan inputs, and accepted consequence tensors can be unused.
   Conversely, architectural access or a nonzero gradient does not prove native
   use sufficient for quality.
4. **The training distribution includes support, sampling and missingness.**
   Hard recovery exclusions remove real supervision; balancing rare LN cases
   then hiding the request changes the default prior. Annotation extents and
   current-policy reached states matter. Real suffix labels cannot be reused
   after inventing a different prefix.
5. **A response must see the distinction it is asked to enforce.** Attack counts,
   star values, LN fractions and marginal occupancy can agree while coordination
   differs. More candidate samples cannot help a selector that assigns all
   relevant alternatives equal cost. Corpus rarity is not physiological truth,
   and expressive ranked counterexamples must remain inspectable.

## A worthwhile next starting point

Reassess the preserved fresh-expert failures together with the
[event-time proposal](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/event_time_gameplay_response_zh.md):
near-duplicate H, unused candidate-time consequences and an acute-response blind
spot have concrete witnesses. They offer bounded questions before another long
fit or fusion. Hierarchical H, improved response curves and restoring a candidate
consequence path are hypotheses, not fixes already verified. The interrupted
continuous-context comparison also remains incomplete, not a job to restart
mechanically. The human's fresh ordinary-expert direction remains part of the
broader goal, not approval to narrow success to a capacity test.

No training process was identified at the recovery check. Product worktrees stay
at `4a8a478` (primary), `96f84fd` (release calibration, unmerged) and `4ec631e`
(realtime benchmark). Existing primary user changes and old Agent Notes were
preserved. The benchmark loader/policy integration and final playability review
remain open; a two-second research qualifier is not that benchmark's acceptance.

## Provenance and continuation

Recovered on 2026-09-29 from the explicitly named chat **Design R1 realtime
generation** (`01a0cd3b-6bcb-7e61-bc04-b3f62df3fe65`), relevant product code/docs,
existing `agent-notes` at `4afc915`, and named local receipts. The product
[formulation](/Users/l/projects/ensomi-model/docs/formulation/README.md) owns V3
definitions. These `relay-notes` are a revisable connected account; no existing
notes were migrated or product documentation replaced.

One [private, local input record](artifacts/private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md)
preserves 78 substantive historical records and two imported advice texts,
distinguishing original steering, answers, agent-authored selections/options,
application-reported goal edits and unknown authorship. The exact chart behind
the first human D4-jack report was not supplied. The private file is ignored,
untracked and not backed up by these commits. If it or other local assets are
missing in a fresh clone, mark the missing evidence rather than silently
recovering private originals from an old transcript.

This session recovered and organized evidence; it launched no research fit,
subagent or successor session. Relay's doctor reports protected mode unavailable.
The old chat's automatic continuation is not authority for a new loop. A human
opens the next fresh session manually and uses these materials to reassess the
question, evidence and next direction.
