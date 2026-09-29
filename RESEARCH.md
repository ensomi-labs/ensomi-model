# Playable audio-to-chart generation

The research asks how to generate a musically coherent, playable 4-key chart
from complete audio while producing usable chart coverage ahead of playback.
R1 was the starting action model, not an architecture that had to remain fixed.
The later target is ordinary playable 2–6★ arrangements with expressive style
variation, independently scoped difficulty/style/LN controls, and stable timing
and publication. A source-free beginning must work as well as later passages.

The original priorities and their refinements are recovered in
[intent and human direction](artifacts/intent-and-human-direction.md).
They explain why timing accuracy, likelihood, exact export, average star rating,
or one corrected bad pattern cannot establish success. User feedback repeatedly
identified sustained jacks, insufficient breathing, irregular short LNs and weak
ordinary TAP/LN organization even after numerical improvements.

The research has built working audio-to-chart prototypes and a substantial
evaluation toolkit. It has **not established a generally playable model**.
The latest committed product account describes a three-initialization joint
learning comparison at 512 of 4,096 planned updates; every arm fails seven of
eight numerical native cases. Lower pressure in the fresh arm is useful evidence,
but does not establish musical quality or control fidelity. See the product
[clean joint learning account](/Users/l/projects/ensomi-model/docs/research/clean_joint_proposal_learning.md)
at `4a8a47805a8fdbcb4fa27802063c55c649d41020`.

## Reading and authority

These are revisable working materials on `relay-notes`, recovered on 2026-09-29
from the explicitly requested chat **Design R1 realtime generation**
(`01a0cd3b-6bcb-7e61-bc04-b3f62df3fe65`), product sources and existing Agent Notes.
The product repository's [formulation](/Users/l/projects/ensomi-model/docs/formulation/README.md)
remains the owner of V3 definitions. Existing `agent-notes` records remain in
place; this is a connected account of the research, not a migration of that branch.

Original inputs, agent questions/options and imported advice are saved in one
[private, local source record](artifacts/private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md).
It is ignored and untracked. A clone without that file cannot independently
check the wording or provenance of the human decisions; it must not silently
reconstruct originals from old chats. Shareable notes paraphrase only the
decisions and interpretations needed to understand the work.

This recovery is organizing existing evidence. No new learning experiment or
protected research loop has been started. Relay's doctor reports protected mode
unavailable. Detailed evidence connections and the final process/state check are
being recovered in separate focused commits.
