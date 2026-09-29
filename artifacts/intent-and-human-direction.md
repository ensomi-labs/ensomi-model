# What the playable-system goal requires

The starting problem was to turn the tagged R1 action generator into a usable
audio-to-realtime chart system. Required pieces included a plausible timing
skeleton, an opening sufficient for R1's historical context, later style and
difficulty conditioning, low startup delay and stable service during dense
passages. Scheduling, speculation and graceful degradation were possibilities
to investigate, not selected architectures. Original goal: [private,
local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0001).

The initial full-song assumption was later explicitly supported: training and
inference may both use all audio; final note placement is the target. Source
redlines do not supply timing truth. Causality applies to chart decisions and
publication, rather than requiring an unknown live audio stream. The initial
hardware/startup questions have no recovered submitted answer. The Mac and
measured latency bounds are experimental conditions, not human-specified product
SLOs. Sources: [full-audio clarification, private,
local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0095)
and [unanswered questions, private,
local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#unanswered-initial-questions).

## Decisions that shaped the work

| Human direction, paraphrased | Effect and limit |
| --- | --- |
| Incremental LN delivery can be supported; optimize the whole system, including R1 | An unknown tail need not block publishing its head. This does not prove a client implementation or authorize arbitrary changes to published content. |
| Begin with small probes; overnight computation is available | The answer concerned the first training pass. It was not a quality acceptance or an instruction to scale every failed candidate. |
| Ground architectural changes in target structure, dependencies and research reasoning | Enumeration of superficially plausible variants was not the desired method. Tech, regular subdivision, elaborated jacks and LN coordination must remain expressible. |
| Use the already checked repository Mel frontend and jointly develop audio, skeleton and row learning | Encoder sufficiency was rejected as a prerequisite. New long-range musical memory was deferred initially, then reconsidered after concrete phrasing failures. |
| Rows need direct audio; skeleton history should be separated from row-content history, with necessary LN state available | The LN qualification followed the earlier simplification. The agent's first absolute claim of no feedback was too strong. |
| R1 must own chord size, columns, TAP/LN choices and complete-row consequences | A later correction rejected moving cardinality/layout control into a typed skeleton merely to repair a symptom. |
| Controls concern independently declared time ranges and overall responses | Difficulty need not be numerically exact. Before, override and restored ranges must be evaluated separately. Global LN amount does not prescribe every prefix's composition. |
| Seek ordinary playable 2–6★ arrangements as well as expressive styles | The objective became clearer than visual plausibility alone. Ranked charts are comparative examples, not a proof that every local pattern is acceptable in every context. |
| Explain failures through proposal learning and player-response selection | Broad real-distribution coverage and continuation/frontier discrimination are distinct problems. Hardcoded exclusions of entire pattern families are not the intended solution. |
| Develop reusable regression evaluations and inspect concrete witnesses | A metric improvement must survive native generation and preserve legitimate counterexamples; Lens, corpus evidence and human play remain complementary. |

Sources are retained with the surrounding proposals and actual answers:
[early scope, private, local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0007),
[budget answer](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0009),
[Mel/joint-learning priority](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0038),
[LN-state qualification](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0184),
[R1 ownership correction](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0509),
[scope semantics](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0453),
[2–6★ corpus direction](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0394),
[proposal/selection distinction](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0805),
and [evaluation priority](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0738).

## Corrections that must survive a summary

The strongest playability reports concern excessive long jacks under Stream/D4,
flat pressure with inadequate breathing, fragmented and irregular LN entry/tails,
close release-to-attack relationships, and insufficient recognizable TAP patterns.
These are human assessments of inspected/generated material. The first D4
long-jack report identifies product `a99519c` but supplies no exact chart or seed;
the answer only identifies Stream control. Later agent reproductions are related
counterexamples, not the user's exact sample. Sources: [report, private,
local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0687),
[answer](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0689),
[breathing](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0693),
[LN priority](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0800),
and [ordinary organization](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1022).

Avoid turning the human's examples into universal rules. High LN coverage,
short LNs, jacks, uneven finger activity and irregular timing can be stylistic.
The issue is their coordination, context and difficulty. The answer about
sub-20-ms attacks did not select either proposed release-gap policy. Later
comments broadened attention to release burden and ranked examples; they did
not authorize a permanent physiology constant for all patterns. See [the
question and answer, private,
local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0393).

Attention, weak audio conditioning, inadequate elapsed-time perception and
exponential response curves were proposed as hypotheses. None becomes a proven
cause because the human suggested it. Conversely, the explicit instruction to
improve ordinary arrangements and willingness to reconsider modules/recipe must
not be weakened into another minor threshold sweep. The latest clean-learning
request proposed fresh initialization followed by possible fusion; fusion was
not already approved as a successful model. Sources: [attention intuition,
private, local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0730),
[ordinary-pattern instruction](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1013),
[fresh recipe proposal](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1050),
and [elapsed-time hypothesis](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1062).

## Provenance and limits

Recovery retained 78 substantive historical records, plus two imported advice
documents, in a single ignored private file. Unsolicited comments, answers,
selected agent text and application-reported goal edits are identified separately.
The imported first expert reply is not human-authored technical direction; the
second critique's original authorship is unverified. Repeated automatic
continuation messages and compaction summaries were not treated as additional
human decisions. Exact old dirty source states are not generally recoverable
from conversational statements alone.

This account does not transfer the old chat's automatic continuation authority
to a new session. The present request is to recover and commit materials.
No new training, promotion, remote publication or successor chat is implied.
