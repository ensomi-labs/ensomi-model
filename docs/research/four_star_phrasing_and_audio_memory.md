# Four-star phrasing and audio-conditioned history

The small audio/H/R/R1 model already follows some large musical changes, but
matched ranked-chart comparisons expose missing organization at shorter and
intermediate scales. Lower note density does not reliably produce relief:
fewer attacks can become longer simultaneous holds, and a similar total count
can lose a repeated rhythmic figure. A sustained-load ceiling cannot supply
the missing musical structure.

The next architecture direction is a shared multiscale audio representation
and history attention queried by current musical context. This is a proposal,
not an implemented or qualified replacement. The evidence below motivates the
information path; it does not prove that attention or more parameters will fix
generation.

## Matched ranked sources

Four ranked TRAIN charts are selected from cached audio lasting 90–240 seconds,
with recomputed native difficulty in [3.8,4.2). Within four LN-fraction bands,
choose the chart closest to four stars. This supplies different arrangements,
not a representative population sample or an untouched test set.

| Source difficulty | Native stars | LN-head fraction |
| --- | ---: | ---: |
| Max Burning!! — Linde vs. F_C's EXHAUST Lv.15 | 4.0005 | .0429 |
| Classic Pursuit — mint's Another | 4.0000 | .2172 |
| STYX HELIX — ReStart | 4.0044 | .4853 |
| Blizzard Heights — Vivi's Insane | 3.9277 | .8380 |

Core2500 and the reported actor128 each generate one complete chart per audio,
requesting four stars and the source's whole-chart LN fraction. Style fields are
unknown. Two additional actor outputs request prominent Stream on Max Burning
and Classic Pursuit, keeping the other controls and seeds. These are separate
conditions; Stream is not assigned as an observed label to the source or output.

All ten exports complete in 111.55 seconds of command time on the M5/24 GiB Mac,
using one CPU thread. Individual native runs take 8.13–14.45 seconds from cached
Mel. Core and actor H streams match exactly in all four matched conditions,
separating their R1 differences from H timing. No weights are updated and no
response planner is used.

The source is a valid alternative arrangement, not the unique correct answer
for its audio. Nevertheless, its concrete relationships show what a generated
chart's density or scalar rating can fail to describe.

## What the sources express

Whole-song Mel/activity plots are read for all four cases. Lens time views are
read on five-second source/generated comparisons within sixteen-second contexts:
Max Burning [68510,73510), Classic Pursuit [88589,93589), STYX HELIX [6800,11800),
and Blizzard Heights [45342,50342) ms. These are 16 of 56 rendered Lens pages;
unread pages receive no semantic judgment. Mel inspection is not literal audio
listening or a human playtest.

**A pause in attacks can preserve a musical voice.** In Max Burning, changing
single/chord flow leads into one sustained LN at 70,878 ms, with no new head until
72,142 ms, then returns to broader movement. It is not an all-finger rest.
The actor inserts five H events inside that interval and develops overlapping
holds around it. Its source-active total H count is only 748 versus the source's
736. A small overall count difference hides a different articulation of this
transition. Under the separate Stream request, the largest H gap shrinks from
the source's 1,264 ms to 467 ms.

**Continuous activity can still have recognizable phrasing.** Classic Pursuit's
source has no H gap longer than 429 ms in its active body. In the inspected
passage, recurring chord accents and short holds form repeated figures with
changing participating columns. The actor contains a run of repeated attacks
on one column, followed by a less regular overlapping-hold passage. Literal
half-second silence is therefore not a necessary definition of breathing.
On the common source-active clock, the actor has fewer H rows, 1,042 versus
1,132, but eight-second H-rate coefficient of variation falls from .1586 to
.0745. Reduced total count and flattened phrase-scale pacing coexist.

**Fewer attacks can produce more continuous occupation.** STYX HELIX's source
moves from taps into an articulated held passage. On [1800,11800) ms, its 102
heads include 20 LNs; mean occupied columns are .3719 and any column is held for
24.34% of the interval. The actor has only 78 heads, but 62 are LNs; mean occupied
columns rise to 1.9642, with some column held for 95.37% of the interval. Both
share the whole-chart request .4853. The source's local mix is allowed to differ
substantially from that global fraction. Occupation is a factual observation,
not a calibrated fatigue score.

**An LN-heavy chart can deliberately change texture.** Blizzard Heights has
whole-chart LN fraction .8380, yet on [40342,56342) ms the source uses 39 LNs
among 82 heads. Short paired holds release together between more separated taps.
The actor uses 96 LNs among 101 heads and keeps at least one column held for
94.51% of that interval, versus the source's 69.31%. Its whole-chart fraction
.9409 also overshoots the request. A high LN request does not justify maintaining
the same held texture through a musical contrast.

The broader result is mixed rather than a uniform loss of variability:

| Common source-active clock | Source H rows | Actor H rows | Source H-rate CV, 8 s | Actor |
| --- | ---: | ---: | ---: | ---: |
| Max Burning | 736 | 748 | .1707 | .1311 |
| Classic Pursuit | 1132 | 1042 | .1586 | .0745 |
| STYX HELIX | 795 | 563 | .0963 | .2257 |
| Blizzard Heights | 887 | 914 | .3221 | .2641 |

The STYX increase is a counterexample to simply maximizing a variation statistic.
Whole-song plots also show several broad changes following spectral development.
Useful variation concerns where and how an arrangement changes, not a high CV,
large gap count or equal use of all fingers.

## The current information bottlenecks

The implementation at `565d5589bb2dea56292ab3853846d7abf928c604` has several
distinct paths:

| Path | What it retains | Relevant limitation |
| --- | --- | --- |
| Fine audio | 100 Hz, width-96 TCN, about 2.5 seconds of waveform support | Local spectral/rhythmic context alone does not expose longer arrangements. |
| Full-song audio | One learned linear 500-ms reduction per Mel frequency, then two width-128 bidirectional attention layers | Global access begins after a narrow linear compression of within-cell structure. |
| H history | Causal TCN over previous H gaps, 127-event receptive field | No audio attached to historical timing tokens; no query-dependent retrieval. |
| R history | Causal TCN over H/R timing roles, plus exact active-LN clocks | Prior musical context of the selected starts is not generally retained in this history. |
| R1 history | Causal TCN over complete rows and gaps, 511-row receptive field | A fixed compressed history is read before current audio/preview/control fusion. |
| R1 lookahead | Next 16 H times | The represented duration shrinks in dense passages. |

The source owners are the
[full-audio encoder](../../src/ensomi_model/research/joint_audio_continuation/context_model.py),
[causal temporal encoder](../../src/ensomi_model/research/bounded_typed_continuation/temporal.py),
[history features](../../src/ensomi_model/research/planned_audio_continuation/features.py),
and [complete-row fusion](../../src/ensomi_model/research/controlled_audio_continuation/model.py).

There is already attention in the audio encoder. The missing direct relation is
between a historical **musical context and the arrangement chosen there**.
The audio branch knows the song but not generated choices. The history branch
knows chosen actions and gaps but does not bind them to audio at their original
times. Their late fusion has to reconstruct that correspondence from compressed
vectors. A sufficiently expressive model could approximate it; the observation
is a structural burden, not a theorem of impossibility.

R1 has direct current audio. The actor also has layout modulation, introduced to
avoid cancellation of shared additive conditions from some mirrored layout odds.
An attention proposal must preserve this distinction: another identical vector
added to both hands can repeat that ineffective route. Hand-specific queries
with shared mirror-consistent weights can condition relative layout choices.

Training and sampling remain alternative causes. Most recent R1 fits froze
audio/H/R and therefore could not improve H phrasing or audio representations.
The [player-state learning comparison](player_state_conditioning.md) demonstrates
training-bank improvement without native generalization. Global controls must
also remain compatible with locally contrasting source passages.

The finite LN-amount controller updates its offset by `(rho * heads - LNs) / 8`,
clipped to [-2,2]. An all-TAP passage under a known positive global LN request
therefore pushes later decisions toward LN starts even if the intended musical
contrast would defer them. This is a finite preference, not a hard quota; local
variation remains possible. Its causal contribution to the observed texture
flattening is unmeasured and should not be blamed on encoder capacity alone.

## Proposed audio and history memory

Preserve the native timed-row interface and module ownership. Add two connected
information paths rather than replacing expressive timing support with a grid
or hard section labels:

1. **Learned multiscale audio memory.** Build medium/coarse tokens from nonlinear
   fine-audio features so temporal changes can survive reduction. Combine local
   detail, several-second development and full-song relations. Encode complete
   audio once; internal token spacing does not quantize H or release times.
2. **Musically queried committed history.** Associate historical timing/action
   representations with audio at their original times. Query them using current
   audio, exact state, H preview and controls, with relative elapsed-time features.
   Keep the fast local path for immediate coordination; attention supplies
   context-dependent retrieval instead of requiring one summary to serve every
   future query.

For the row path, a possible memory item and read are

$$
m_j = [h_j^Y, E(A,t_j)],\qquad
r_t^Y = \operatorname{Attention}
\big(q(E(A,t),x_t,H_{>t},C_t),\{m_j:t_j<t\}\big).
$$

Here $h_j^Y$ contains only rows committed through $t_j$. Complete audio is
available on both training and inference paths, including future audio. Future
chart actions, releases or source-only endpoints never enter this memory.
The query is hand-specific where layout requires it. Continuous-time biases
must distinguish elapsed duration from event count.

H receives a separate memory of H timing and its aligned audio; it does not read
R1 materialization or its content encoder. R receives its timing/role history,
H preview and permitted LN occupancy/origin information. R1 reads the chosen
skeleton and remains responsible for chord size, columns, TAP/LN and release
subsets. Shared audio does not erase those dependencies. Canonical player state
and explicit future-response evaluation remain separate from musical memory.

A practical memory can retain recent event detail and summarize older history
on an elapsed-time clock. Such summaries are approximations: silence needs an
explicit clock position, events within a cell retain order through the local
encoder, and active LN origins cannot disappear with an old cell. Summaries
must contain only already observed chart content, even when the audio features
have global support. Control changes alter queries and future choices without
resetting physical state or rewriting old memory.

The closest musical analogue is relative attention for motifs and repeated
structure in [Music Transformer](https://arxiv.org/abs/1809.04281). The transferable
mechanism is selective access to historical structure. Its symbolic music tokens
do not solve the external-audio/4K-action correspondence or gameplay constraints.
[Transformer Hawkes Process](https://proceedings.mlr.press/v119/zuo20a.html)
provides a closer event-time attention analogue, but its event prediction results
do not validate musical phrasing or our H/R factorization. This proposal is an
adaptation of established primitives, not a novelty claim.

A slower learned arrangement process remains a distinct branch if retrieval
alone cannot maintain a phrase choice. [MusicVAE](https://proceedings.mlr.press/v80/roberts18a.html)
illustrates hierarchical subsequence embeddings. Transferring that principle
would require an audio/control-conditioned prior available at inference and
evaluation of sampled prior trajectories, not only a target-derived posterior.
It would differ from the earlier [constant chart profiles](shared_arrangement_profiles.md).
No fixed chorus taxonomy, direct future head-count command to the skeleton, or
forced reset at an internal block boundary is proposed.

## Learning and qualification consequences

The first architectural comparison should jointly train the new memory and audio
path against a matched continuation of the existing model. Source arrangements
remain separate examples. Scored spans must include musical development across
multiple seconds and valid quiet time; a training interval must not become a
chart endpoint. Whole-chart controls and genuinely scoped controls must both
appear with their actual scopes, rather than turning a global LN/difficulty
request into a constant local target. Human style supervision stays attached to
its original source and range.

Likelihood supplies a proper sequence-learning signal but cannot select the
playable endpoint alone. Native comparisons need sustained repetitions and their
endings, TAP/LN texture contrasts, free and held relief, rhythmic transfer among
fingers, and their relation to the full audio. Multi-scale facts locate passages
for inspection; they do not define quality by maximizing variation. Distinct
control ranges remain separate. Generated-state coverage and future-response
learning must use the actual visited history rather than borrowed source suffixes.

Capacity may increase in the audio and attention paths, including a several-fold
increase over the roughly 4.6M-parameter prototype if profiling supports it.
There is no reason to preserve the old size as a quality ceiling. Complete-audio
encoding, query cost in dense passages, first-thirty-row latency and publication
deadlines must be measured on the actual implementation. The existing speed
margin permits investigation; it is not a prediction of the new model's speed.

## Evidence identity

Study owner: `20260927-four-star-phrasing-v1`.
Product source: `565d5589bb2dea56292ab3853846d7abf928c604`.
Plan SHA-256: `9ed61800275342fa03e8282b54985c6d33930e2ac3666b124d9f177b333d24d9`.
Core2500 checkpoint: `0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`.
Actor128 checkpoint: `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3`.
Lens harness: `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`.
Ranked source identities, in table order:

- `e043e52f9c5989c31642c937d79cfc5a233820e3cb288962df47c397477c5a5c`
- `899515366f3aab0962a927f947135903503c6cb5322749a15fa6cc451431b5c9`
- `0cc766925a6c98343ffe812927bcec6f62ba923d99d4c7cd9e9a16d23da43412`
- `2b77660caa71c5ae0b11e1e4b6b9d94cc675984ac88d47b4a3b5e80320fd412d`

The artifacts retain charts, full-song plots, exact scoped observations and
viewed-page coverage. They are local research assets; the observations and
information contracts needed to assess this proposal are stated here.
