# Causal player-state input and short-future R1 learning

An explicit player-state input and four-second response objective improve R1's
continuations on the training replay bank, but fail to repair complete-song
generation on the reserved developmental panel. Both fitted endpoints produce
new sustained single-column overload. They are not promoted.

The comparison asks whether R1 can learn the less overloaded choices found by
the [sustained-response planner](sustained_response_planning.md), while retaining
genuine annotated organization. The causal input and open-ended trace scorer
support that comparison; they do not define the final V3 response specification.

## Information supplied to R1

[`PlayerCondition`](../../src/ensomi_model/research/player_response/conditioning.py)
reads 24 deterministic observations per column:

- Attack and release rates over .5/1/2/4/8/16 seconds, encoded as `asinh(Hz/4)`.
- Held-time fractions over those same windows.
- The previous two complete attack masks, their ages in seconds encoded with
  `asinh`, and availability bits.

These observations come from committed rows and elapsed time. A TAP and an LN
press each count as an attack; a release remains separate. No future LN endpoint,
difficulty request or style label enters the observations. Open LN origins survive
the 32-second recent-event buffer. Advancing through no-row time preserves holding
and ages the attack/release history. Control changes do not reset this physical
state; the policy's separate LN-amount feedback has its own control-episode rules.

The four columns are arranged in the model's existing two mirrored hand
coordinate systems. One shared, zero-initialized linear map projects each
96-dimensional vector into R1's 128-dimensional context, adding 12,288 parameters.
That context informs complete-row layout, R1-owned count composition and the
candidate-consequence residual. Existing direct audio, exact clocks, H preview,
controls and learned row history remain available.

Zero initialization reproduces the old probability law. The optional
`player_state` probability option owns the new checkpoint parameters; old
checkpoints leave it disabled. The tested model grows from 4,600,369 to 4,612,657
parameters. This representation preserves declared observations, not every
coordination distinction or an inferred physiological fatigue law.

The native session and row-trace scorer gather these observations for enabled
models. The shared planned-interval collator requires `player_state=True` when
scoring such a checkpoint; its default skips that extra prefix replay for older
models.

The added input goes only to R1. H keeps its timing/history/audio/control inputs.
R retains its timing history, audio, controls, H preview and LN state. R1's actual
choices still change subsequent exact occupancy and execution-feasibility bounds;
its learned content encoder is not added to the skeleton factors.

## A private future need not close its holds

Earlier outcome-learning adapters constructed a complete `SourceChart` before
scoring a short generated interval. That required terminally closed objects and
often generated the rest of the song even when the outcome needed only a few
seconds. A local response endpoint is not the true audio endpoint.

[`collate_row_trace` and `score_row_trace`](../../src/ensomi_model/research/controlled_audio_continuation/trace.py)
take actual complete prefix rows, a proposed continuation, the complete timing-only
H plan, controls, real audio duration and a full-song audio encoding. They score
rows in a half-open native interval `[a,b)` without requiring later releases.
Its response counterpart is the continuation `(a-1,b-1]`, including elapsed time
after its last row. A true audio-end action still obeys terminal occupancy rules.

Each row query uses its own pre-action replay and content history. Future rows
cannot supply features to an earlier query. Full H timing preview is retained
even when it extends beyond the response endpoint; truncating it would change
R1 support and likelihood. This does not provide future materialized columns,
counts or hold endpoints. Complete source audio remains available on both the
training and inference paths.

The scorer reconstructs the native recovery and LN-amount preferences using the
actual prefix. Their arithmetic runs in differentiable CPU float64, matching
native sampling, while gradients return to the MPS model. The trace contains
raw rows and deterministic observations rather than stale learned caches.
Neural history is recomputed under current weights. Full-audio encodings can be
reused only because this experiment freezes the audio encoder.

These are native row probabilities. They are not probabilities of the previous
planner's selected trajectories, whose selection step changes the distribution.

## Source imitation and generated-response supervision

The comparison begins from the reported actor-128 checkpoint. Both arms use the
new zero-initialized input and identical genuine source draws. One learns only
source row likelihood; the other adds expected sustained-response cost on sampled
four-second continuations. Audio/H/R weights remain frozen. Existing R1 parameters
and the new projection are trainable.

Factual examples retain their own source history and actual actions. Difficulty
and LN fraction are defined on the exact training scope; outside it, whole-chart
conditions remain. Human style claims keep their original scopes and known/unknown
status. This differs from conditioning a small source window only on a whole-map
LN fraction: both are meaningful conditions, but the local construction directly
trains the scoped-control interface.

The generated branch uses a fixed replay bank of actual baseline-generated
prefixes. It samples new futures from the current R1, on each bank's unchanged H
times and controls. It never attaches old source suffix labels to a changed
history. Requested style is a policy condition, not an observed style label for
the generated result.

For a fixed replay prefix $h$, let $J(h,Y)$ be the corpus-reference sustained
excess integral over a four-second future, and let $q_\theta$ denote the native
R1 row factors along that future. The added objective is

$$
100\,\mathbb E_{h\sim B}\mathbb E_{Y\mid h}
\left[\frac{J(h,Y)}{4\ \mathrm{s}}\right].
$$

Three independent continuations estimate its score-function gradient. Each uses
the other two costs as an action-independent baseline. Prefix-only costs shared
by all continuations cancel. An empty future with no R1 decisions contributes no
R1 score gradient, while elapsed time still contributes to its response.

The release process remains state dependent. For a fixed materialized trace,
however, its parameters, exact state transitions and feasibility calculations
have no direct dependence on the updated R1 parameters. The R1 row score terms
therefore supply the parameter-dependent part of the sampled-future score.
This does not assert that the release probabilities are constant across different
futures, or that row scores alone are the complete absolute trajectory probability.

This optimizes continuations from the declared replay distribution. It does not
differentiate through how the old policy generated that fixed bank, nor establish
full-song on-policy optimality. Native generation and Lens inspection remain
necessary qualifications. Sustained attacks are only one response channel;
holding, coordination, semantic style and H breathing remain independent concerns.

## Implemented checks and bounded evidence

Eighteen focused tests cover zero-initialized compatibility, H/R separation,
causal observation gathering, mirrored hand coordinates, checkpoint loading,
native/source probability agreement, and open-ended trace probabilities and
gradients on CPU and MPS. Private trace scoring also agrees across interval
partitions, including empty intervals. An initial timestamp-type mismatch at the
exact-replay call was corrected before these checks passed.

The replay bank contains 20 distinct TRAIN song groups/audio identities, excluding
the existing reserved native/control/style songs and their groups. It covers five
requested style families and nominal difficulty 2.5/3.5/4.5/5.5, with 69 high-excess
and 80 ordinary contexts. The selected Stream and LN songs have no positive
baseline excess; the hard contexts come from Jack/Tech/Trill requests. This is
targeted training coverage, not an estimate of population failure frequency.

An eight-update integration run completes in 37.76 seconds on the M5/24 GiB Mac.
Five of 24 sampled futures have positive cost, two updates receive nonzero response
gradients, and eight futures end with active holds. Maximum sampled/rescored row
log-probability discrepancy is $1.89\times10^{-5}$. Frozen audio/H/R tensors are
unchanged; peak sampled footprint is 3.03 GiB. The smoke weights are discarded.

One smoke update has response-gradient norm 283.5 versus source-gradient norm
20.8 before clipping; another has .79 versus 42.7. A source likelihood term alone
does not prove semantic preservation when the added objective dominates an update.
The smoke establishes integration and a nonzero learning signal, not quality.
After the main experiment, player-state preparation in the older interval
collator became opt-in. Twelve affected condition/trace/sampling checks pass in
7.23 seconds; that change does not alter the native-session or trace-scoring
paths used by these fits.

## Complete-song qualification rejects both endpoints

Both arms complete 384 updates with the same 768 factual draws: one annotated
scope and one population example per update. These include 197 distinct annotated
scopes and 300 population identities; none overlap the reserved song groups.
Audio/H/R tensor hashes remain unchanged. The source-only fit takes 401.50 seconds
and the response fit 1,492.60 seconds. They partly overlap after a memory check,
so these are command durations rather than isolated throughput measurements.
Peak sampled footprints are 4.98 and 5.17 GiB respectively.

The response fit samples 1,152 private futures; 445 have positive cost and 162
updates receive nonzero response gradients. Of those futures, 272 end with open
LNs. Maximum sampled/rescored row log-probability discrepancy is
$2.30\times10^{-5}$. Old R1 parameters use learning rate $3\times10^{-5}$;
the player projection and designated control/composition parameters use
$3\times10^{-4}$, with gradient norm clipped to one. These facts verify the
declared update, not successful gameplay control.

Qualification generates 52 complete exports: nine Stream requests and five
other/control-switch cases per arm, plus three seeds on each of four genuine
style scopes per arm. Native requests share the original seeds and full audio;
source-style comparisons share source H. All matched H streams are identical.
The panel has been used repeatedly for development and is not an untouched test
set. No planner is used in this comparison.

| Nine full-song Stream cases, four-star request | Initial actor | Source-only | Source + response |
| --- | ---: | ---: | ---: |
| Mean integrated sustained excess $J$, seconds | .027516 | .144229 | .107160 |
| Mean absolute whole-star proxy error | .47893 | .85255 | .74113 |

The response objective reduces excess 25.7% relative to source-only training,
but both fits are worse than the initial actor. The required 50% reduction
against both comparators fails, as does the allowed .15 increase in star error.
NLL, the fitted excess cost and the star proxy are insufficient quality claims.

The response endpoint's Zenithfall seed-zero chart has 38 attacks on one column
within four seconds, versus 6/4/1 on the others; its hottest eight-second window
has 70 on that column versus 10/11/5 elsewhere. Other columns are not held.
A chain allowing at most 170 ms between same-column attacks contains 41 attacks
over 4,412 ms. Lens inspection of the full ten-second context confirms a prolonged
single-column sequence. The response Hysteric seed-two chart also reaches 33
attacks on one column in four seconds. These are new failure locations, not just
an aggregate-score fluctuation or the two original crops remaining unchanged.

Speed is adequate on this bounded panel: source-only first-thirty-row latency is
.662–1.338 seconds and response latency .673–1.298 seconds; the slowest two-second
publication takes .489 and .500 seconds respectively. Measurements use one CPU
thread and loaded weights, include complete-audio encoding from cached Mel, and
exclude waveform decoding, Mel construction and model loading. They do not
establish end-to-end cold-start latency. The four qualification commands consume
1,357.07 seconds in total; intervening analysis is outside this active-command
accounting.

## Scoped controls and genuine style comparisons

Mixed improvements remain useful evidence. With LN coordination requested at
four stars and LN fraction .6, the initial output has proxy 4.803 and fraction
.7634. Source-only produces 4.512/.6396; response training produces 4.399/.6556.
That amount improvement does not offset the separate Stream failure.

The live control change is submitted after publication through 63,999 ms.
The three ranges remain separate:

| Range | Request: stars / LN fraction | Initial | Source-only | Source + response |
| --- | --- | --- | --- | --- |
| Before [0,64000) | 3 / .2 | 2.2268 / .2992 | 2.2461 / .1353 | 2.6007 / .1518 |
| Override [64000,96000) | 4.5 / .6 | 3.9010 / .5836 | 3.2746 / .5828 | 4.2277 / .5743 |
| Restored [96000,357797) | 3 / .2 | 4.4218 / .2064 | 3.5679 / .1937 | 4.4360 / .1845 |

Response training improves the override's difficulty proxy while leaving a large
restored-range overshoot. Source-only improves the restored range but undershoots
the override more. A pooled score would hide these different errors.

Genuine prominent style scopes supply a more specific organization comparison.
The following star values are measured only within each labeled scope, over
three generated seeds, with source H fixed:

| Human scope, ms | Source target | Source-only outputs | Response outputs |
| --- | ---: | --- | --- |
| Tech [158638,165038) | 3.414 | 4.165, 4.378, 4.234 | 3.363, 4.259, 4.256 |
| Jack [44257,50924) | 4.170 | 3.526, 3.623, 3.556 | 3.580, 3.760, 3.179 |
| Stream [79290,85290) | 4.590 | 5.288, 4.562, 5.119 | 4.316, 4.219, 5.069 |
| Trill [288156,290040) | 5.076 | 4.861, 5.077, 4.363 | 4.832, 4.814, 4.773 |

All 24 generated scopes have zero LN fraction; the source Tech target is .0118
and the other three are zero. Scalar agreement does not establish style fidelity.

Matched Lens review covers seed zero, with seven reference pages and seventeen
generated pages read in total, including the native overload context. Both Jack
outputs retain substantial repeated and overlapping chord groups. Stream movement
remains, although the response version mixes more paired/changing chords and
short repeated columns. The human Trill scope has a distinct fixed two-plus-two
exchange; generated group membership varies more and the episode is less distinct.
This does not establish preserved prominence, nor justify declaring the style
absent from geometry alone. Tech retains irregular source timing by construction,
which cannot itself validate the generated action organization. Unread pages and
other seeds are not assigned semantic judgments. No listening or human playtest
was performed in this qualification.

## What the failed fit did learn

A follow-up holds the old replay prefixes fixed and chooses one context per bank
song: the highest-cost old context when available, otherwise its first ordinary
context. Three fresh suffix draws per arm yield 180 private futures. Mean
four-second $J$ is .008769 for the initial actor, .019478 for source-only and
.003440 for response training. The latter improves 60.8% against initial and 82.3%
against source-only. This uses the training bank and is not held-out evidence.

Two complete native rollouts on those training songs further separate short
conditional performance from full rollout performance. The Stream 4.5-star case
stays at zero excess. The Trill 4.5-star case falls from .385476 to .009320,
with identical H. Consequently, fixed old prefixes alone cannot explain all
long-generation failure. Limited audio/control/state coverage, conflicting
source gradients and generalization remain plausible causes; their contributions
are not identified by this probe. In particular, all four selected Stream bank
songs already had zero baseline excess.

A second diagnostic freezes the actual failed response Zenithfall history on
[118000,126000), keeping audio, H, controls, legality and native preferences fixed:

| Player input lesion | Mean probability of attacking the overloaded column |
| --- | ---: |
| Full response model | .69704 |
| Rates and held fractions zeroed | .69429 |
| Last-two attack masks zeroed | .69517 |
| Entire added input zeroed | .69246 |
| Initial weights, same physical history | .64427 |

The whole added input changes this probability by only .46 percentage points;
most of the fitted-policy increase remains without it. The new recency features
alone are therefore not a sufficient explanation on this trace. These are
network-input lesions, not physical counterfactuals, and they do not determine
the effect on a separately generated trajectory.

The result favors retaining an explicit calibrated response in generation
decisions while improving coverage of states encountered by the learner.
[DAgger](https://proceedings.mlr.press/v15/ross11a.html) and
[interactive cost-based learning](https://arxiv.org/abs/1406.5979) motivate attention
to learner-induced state distributions. Their guarantees do not transfer to this
partial response cost and sparse style supervision. Genuine source actions or
style labels cannot simply be attached to altered generated histories.

Neither more training of this unchanged setup nor removal of the new projection
is justified as the next repair. H breathing, held demand, coordination and
stable scoped style/difficulty control remain unresolved system requirements.

## Evidence identity

Implementation: `94d0b082282ae886709c72ffdc885b2f4e045252`.
Initial checkpoint: `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3`.
Study owner: `20260927-player-state-r1-learning-v1`.
Source panel: `7f8d56c8aeb3b5e33dd3dfe3d993a7311e97d9bea1c983ee2bd0a76d991d9c10`.
Replay bank records: `53ed0a0bf953439ebfbe83a4cf6eff8c203d19232c9b102f7c94428a0f5381d1`.
Factual draw order: `348a344dcb9245e474d6cd3fdc4320ad2773d908bc6ed4e48ebcfd7dcefd0524`.
Smoke checkpoint: `2a4c7ddd038c47f293afb8c67d0084cf6abcbca0c0a183616f8197c592cc2d64`.
Source-only endpoint: `7e0c5dba508af1968bed38b915246c822037c92c519846c1e3e0fb21b4693237`.
Response endpoint: `a91791fb7fda45190ddb0a2f1f17dc160b4e55e6cf1030cf3b17424d1a147ad6`.
Frozen audio/H/R tensors: `8c21a572ba08b5c8f847e818f318d8c92ef7d8f070616eb5ce0d42d911f626a5`.
Lens harness: `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`.
The bank, source pool, calibration, weights and run snapshots are local research
assets; the code, input semantics and bounded evidence are preserved here.
