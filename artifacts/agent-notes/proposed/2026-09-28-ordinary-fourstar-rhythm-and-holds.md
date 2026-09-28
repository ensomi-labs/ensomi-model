# Agent Note: Ordinary four-star rhythm and held-finger roles

Note ID: 2026-09-28-ordinary-fourstar-rhythm-and-holds
Status: proposed
Kind: investigation
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 2c67bb6b70df425d7bca68a455488833a6720372
Scope: Typical ranked 3.5–4.5-star timing and LN/TAP organization
Related: 2026-09-28-coordination-frontier-and-ranked-contrasts, 2026-09-28-ln-risk-calibration

## Research question

The intended ordinary four-star distribution has recognizable low-order rhythm,
LN entry/release with sufficient execution room, and arrangements such as a
held finger accompanying TAP motion on the others. Previous deliberately
retrieved short-LN counterexamples show that absolute bans would be wrong;
they do not establish the center of the normal distribution. The question is
which temporal and held-role relations the generator should learn, and how
this differs from retaining support for Tech or specialist LN arrangements.

The user explicitly requests a subagent to confirm real corpus organization.
Agent /root/ranked_four_star_arrangement owns read-only corpus statistics and
Lens source reading in artifacts/joint-audio/20260928-ranked-fourstar-arrangement-v1.
It may write its own scripts/evidence but does not edit product source or notes.
The primary agent owns selected full-audio Mel views and synthesis. Neither
role claims new human judgments or a measured physiological threshold.

## Bounded exploratory evidence plan

Revision: 2
Accepted revision: none
Execution authority: continuing research goal and explicit corpus/subagent request.

Use the previously pinned natural TRAIN pairs (SHA
13e4d8b7910f7d33ef18c078a27d12406317a2b7ecaf01fa9843752ef00dae9d)
and complete census of 6924 charts. Recomputed 20241007 stars select 1973
charts /1614 song groups in [3.5,4.5]. Separate TAP-majority, mixed and
LN-majority descriptive strata; they are not human style labels. Retain
per-chart statistics and song-group weighting rather than pooling every
object across long charts. Read normal examples and contrasting specialist
contexts through Lens with actual entry/exit state and all LN tails.

The primary first audio case is Celestial Horizon [Heavenly], beatmap 2131277,
set 1018454, source SHA
729bd30d01d76837f3acc556efb8398b09f06aa8f33726afa275c7fe6efb4cdc,
reported whole stars 4.00355 and LN-head fraction .07485. Inspect [8800,19500)
ms with the canonical Lens audio command. The candidate's four held roles
move across columns while other fingers carry TAP motion; it was selected
as an organizational example, not randomly as a prevalence sample.

Lens source is 22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60; model frontend
source remains ef42095. Audio is whole-file decoded/peak-normalized by the
existing music frontend (24 kHz,128 Mel,10 ms hop,40 ms support). Frame i
has support [10i,10i+40) and center 10i+20 ms. Mel changes can support
spectral correspondence but are not listening or instrument/intent labels.
Do not equate every prominent transient with a required head.

Prepare a fresh Lens learning bundle for the source set in
artifacts/joint-audio/20260928-fourstar-audio-reading-v1; use one CPU worker,
existing environments, no training/downloads, 600 seconds and 4 GiB process
footprint for an audio view. Preserve output/evidence hashes. Stop on source
mismatch, failed harness output or resource bound; no overwrite. Additional
selected audio examples require a recorded revision before execution.

Descriptive grid analysis may use source redlines as a reference, but final
note placement remains the learning target. Coverage by a 1/16 grid does
not mean every subdivision is occupied. Corpus rarity does not identify
human physiology. Reports must state whether each claim comes from the
population, a deliberately selected example, Mel inspection, or a hypothesis.

The live three-arm fit remains unchanged and owns all accelerator work.
Future architectural choices should preserve timing/R1 ownership: H gives
head-bearing times, R gives pure-release times, R1 chooses head counts,
columns/types and release identities while seeing audio and the skeleton.

Revision two adds a second audio view before extraction: Kimi no Bouken
(TV Size) [Create Your Adventure], beatmap 1969732/set 866848, source SHA
8e07788cc8271d6eb08b907b2f06a84fdf2a5daa283593b5a77fa1f7f9ee3134,
reported 4.01555 stars and .1308 LN fraction. Inspect [80000,84500) ms.
Its column-0 LN spans 80858–83858 with 19 internal TAP rows/28 TAP actions,
including nine jump rows across the other three fingers. This is a selected
contrast with denser free-finger organization, not additional prevalence
evidence. Prepare separate fresh bundle-kimi and kimi-audio outputs in the
same owner; all frontend, resource and interpretation conditions stay fixed.

## Completed corpus and music reading

The explicitly delegated subagent completed the 1973-chart study, preserving
source SHA checks and metadata ranked status. Its actual selection uses a
closed [3.5,4.5] interval; the prior half-open spelling here is corrected.
No source has exactly 4.5 stars, so cohort membership is identical.
The 134/1140/636/63 no-LN/TAP-majority/mixed/LN-majority charts correspond
to 6.40/57.65/32.89/3.06 percent under equal-song then equal-chart exposure.
Within strata, opportunity-conditioned song→chart→object weighting gives
LN-duration medians 196/162/132 ms, and next-same-finger recovery medians
242/183/167 ms. These are conditional empirical distributions, not physiology.

TAP-majority LN duration <=80/60/40 ms has 5.8107/1.2073/.1362 percent mass.
Same-finger release→head <=40 ms has .0065 percent mass (five pooled events
among 164052 opportunities). The primary agent recomputed the three duration
fractions from raw per-LN observations. The song- and chart-weighted fractions
of same-entry LN groups with identical tails are 73.4/52.1/34.2 percent in
the three strata. 92.3 percent of TAP-majority LN heads lie within 2 ms of whole/half-beat
positions under source redlines. This does not mean every grid position is
occupied, and unclassified events are not automatically Tech.

Nine complete selected source charts were cross-checked against Lens canonical
normalization for every sourceLine/start/end/column/kind. The subagent read
31 time-proportional pages. The primary additionally read Celestial/Kimi
source pages and both Mel views. Kimi's sustained harmonic bands accompany
the 80858–83858-ms column-0 hold; the other fingers sustain 150-ms jump/single
alternation. Celestial transfers a held role across the fingers while TAP
motion continues. Arakajime supplies a held-finger plus repeated-double-key
contrast. Short specialist LN examples remain separate from these ordinary
examples and from population prevalence claims. No audio listening occurred.

The full report, five source-linked figures and model-ownership implications
are committed at 2c67bb6b70df425d7bca68a455488833a6720372 on the isolated
codex/release-calibration branch:
docs/research/ordinary_fourstar_rhythm_and_holds_zh.md.
Links and whitespace checks pass. The document distinguishes existing
HoldAudioCues from a proposed richer held-role representation, and private
arrangement memory from committed player state. It does not claim a new
architecture has been validated. No main executable source was changed.

Evidence identities:

- Corpus plan: 89aa6eec5cdc27b37a2f1f5938f68bed9a22daced3d51d258072a6daae54c9b5.
- Recommended summary-v2: e630cd022cb19d96aabc8eb839dc0cf9fe25c2a90fc0c41db1fb95cede1966fa.
- Source selections: 24bb92db1f5fe0bd884c9a80a2c2e9e3ca27cd1887858f768701490a798ee91f.
- 31-page reading record: a152796288f19d88e085bc96eb3218b112ca24380d1f269863b887b3253785c6.
- Celestial Mel evidence: 1e119809c1168bf56a45c3a24cf8818372efe22aea421f641a4242928037fc8e.
- Kimi Mel evidence: 1d9637b413c3606e06f8f667a8e95f5295f4240ded91d1674ce583c6dccae5e9.
- Parent Mel reading record: a6f0e377614ab3c2e83476ef9044338ebb5adbe41adef2cccf60d48bc6e54baf.

Original summary remains intact. Its no-opportunity proportions in the no-LN
stratum were mistakenly zero; summary-v2 and a separate receipt change these
to null, without changing the quoted nonempty-stratum numbers. The subagent
is complete. Audio preparation/extraction handles 28458,18770,96336 are all
terminal exit zero; do not restart them. Their fresh owners remain available.

Decision: REFINE. Ordinary low-difficulty organization needs a distinct
conditional distribution and source-linked regression witnesses; preserving
rare complex support is insufficient. Use the new normal examples alongside
previous failures and specialist positives when evaluating held-role retention,
shared closures, recurrence and audio correspondence. No hard duration/grid
mask, independent human label or playable-model promotion follows from this study.
