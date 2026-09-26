# Player-response state for sustained load and recovery

The audio-conditioned model can turn a four-star Stream request into a sustained
single-column jack. Immediate recovery preferences and a scalar difficulty
readout do not prevent this failure. The existing `frontier2` remains useful as
a learned row preference, but it does not implement the formulation's
time-horizon continuation-response interface.

The implementation direction is an explicit state derived from committed rows,
with real-time evolution and an independently calibrated response operator over
proposed continuations. R1 retains cardinality, columns, TAP/LN and release
subsets. H must also supply appropriate timing variation and recovery intervals;
redistributing columns cannot create gaps in a dense mandatory H plan.

## The failure is reproducible on identical head times

A playtest of common-prefix actor-128 reported excessive long jacks under a
four-star Stream request. Its original audio, seed and chart are unavailable.
Nine diagnostic outputs reproduce that control combination on Zenithfall,
Hysteric and Take: difficulty 4, Stream prominent, other styles and LN amount
unspecified. These reproduce the failure family, not the unavailable original
chart.

In Zenithfall seed 271201, [41094,45094) ms contains 32 H rows and 34 heads.
Their column counts are **[1,31,1,1]**. The main same-column sequence lasts
3,907 ms, with 31 attacks. Other columns are not occupied by holds at those
attacks. Lens context shows movement concentrating into the second column,
continuing for several seconds, then moving into another same-column sequence.
Hysteric seed 271212 has counts **[4,4,28,1]** in [8221,12221) ms, also without
other held columns forcing that allocation.

The row problem can be separated from head timing in these examples. The
following checkpoints generate exactly the same H streams under the same audio,
controls and seeds, but different arrangements in the identified windows:

| Checkpoint | Zenithfall column heads | Hysteric column heads |
| --- | --- | --- |
| Core2500 | [11,14,10,14] | [16,15,14,15] |
| Modulated-128 | [5,15,21,3] | [18,16,17,16] |
| Reported actor-128 | [1,31,1,1] | [4,4,28,1] |
| Subsequent paired-128 | [14,3,5,10] | [11,8,8,12] |

The paired endpoint redistributes the attacks in these two windows. In Hysteric
it has 39 heads versus the reported candidate's 37, so that improvement is not
simply removal of notes. Two cases do not establish a general repair. The paired
endpoint still fails the reserved difficulty-response comparison described in
[the outcome study](common_prefix_outcomes.md#continuing-outcome-learning-did-not-repair-control).

## What the current frontier actually represents

[`RowConsequence`](../../src/ensomi_model/research/bounded_typed_continuation/consequence.py)
returns a scalar energy for each complete candidate row. Its exact coordinates
include candidate post-occupancy, attack/release intervals, LN age, the next H
and the second H. The planned-model
[feature builder](../../src/ensomi_model/research/planned_audio_continuation/features.py)
passively advances the immediate post-action state to those opportunities. It
does not simulate intervening future actions.

Its learned hand context can contain longer history, audio and controls. The
history TCN may therefore encode repetition or coordination; this inspection
does not prove a representational impossibility. The concrete deficiency is
that the module has no continuation $Y$, explicit endpoint $e$, separately
updated player state, or calibrated response targets. Its scalar is trained as
part of row likelihood. A preference for continuing a familiar pattern can
therefore receive a favorable score without demonstrating acceptable demand.

The separate
[recovery preference](../../src/ensomi_model/research/typed_audio_continuation/response_preference.py)
uses immediate HH/RH/HR intervals. At difficulty 4, an HH gap above 107 ms has
zero HH rarity cost. Its head-pressure term counts recent heads across columns
within that short window. It does not distinguish a persistently overloaded
finger. The same locally unpenalized repetition can consequently continue for
seconds without accumulating a distinct recovery penalty.

The scalar strain proxy is also insufficient as an acceptance criterion. A
synthetic 16-second passage at twelve H rows per second receives scoped values
4.141 for one-column repetition and 2.600 for four-column cycling. It orders the
two correctly but still assigns an extreme sustained jack a value near four.
These are diagnostic constructions, not empirical human difficulty labels.

## Corpus evidence for a rate-duration response

The reference contains 6,923 native-1x ranked TRAIN charts, with difficulty
recomputed by `compute_mania_star_rating_20241007`. The [3.5,4.5) band contains
1,972 charts in 1,613 song groups. For each chart and window duration, compute
the maximum number of attacks on any column in a moving time window, divided
by its duration. TAP and LN press are attacks; releases are measured separately.
Quantiles give each song group equal total weight, shared among its charts in
that band.

| Window | Median chart maximum, attacks/s | 99% | 99.9% |
| --- | ---: | ---: | ---: |
| .5 s | 8.000 | 10.000 | 10.000 |
| 1 s | 6.000 | 8.000 | 9.000 |
| 2 s | 5.500 | 7.000 | 8.000 |
| 4 s | 4.750 | 6.000 | 6.500 |
| 8 s | 4.375 | 5.375 | 5.750 |
| 16 s | 4.000 | 5.000 | 5.375 |

The generated Zenithfall passage reaches 7.75 attacks/s over four seconds.
Its 31 of 34 heads also show strong concentration. Duration, concentration and
context matter together; an isolated short-gap percentile misses that distinction.
These descriptive quantiles are not adopted physiological cutoffs.

Real exceptions prevent a blanket ban on long jacks. Lens inspection of
`Extra Mode [cie_n VS. Spy's Extra Mode]`, rated 4.069 here, shows a sustained
outer-column anchor with changing accompanying groups. Its longest same-column
chain under a 170 ms gap criterion has 28 attacks over 4,154 ms. A 4.289-rated
`3#006 wyax03 [Happy Love Expert]` reaches 7.5 attacks/s in its strongest
four-second window through shorter repeating groups with interruptions. The
human-prominent Stream episode in `Singularity [FISSH's Beyond]`, rated 3.987,
instead moves across the four columns with chord accents. These are different
organizations; unequal use of columns alone is not a BAD-pattern definition.

Breathing also needs more than one aggregate density measure. Across the nine
generated charts, H-rate variation in eight- and thirty-two-second windows is
not uniformly low. Nevertheless, several charts have almost no fully free
interval beyond 500 ms inside their active bodies. Activity variation, actual
recovery opportunities, LN occupancy and musical context must be considered
separately. Neither high coefficient of variation nor low average density
establishes useful breathing, and a sustained musical passage need not contain
arbitrary periodic rests.

## A reference state before committing to a decay law

The [player-response prototype](../../src/ensomi_model/research/player_response/state.py)
provides `CommittedPlayState` and `observe_continuation(state, Y, end_ms)`.
It retains exact replay, time-indexed attacks/releases, closed-hold intervals,
open LN origins, complete recent rows and the last two attack groups. Recent
rows are retained by elapsed time, with a 32-second reference buffer, rather
than a fixed event count. Open holds survive beyond that buffer.

The state has two distinct operations:

$$
d^+=U(d^-,x^-,y),\qquad d(t+\Delta)=V(d(t),x(t),\Delta).
$$

Both operate on the actual clock. Observing a row cannot insert an action into
already committed no-row time. A private continuation returns its own terminal
state without mutating the committed one. Local occupancy is checked; true
audio-end closure remains owned by the generation caller.

The response observations include peak and terminal attack/release rates over
declared time windows, integrals of those rates over the future duration, actual
held milliseconds, and ordered coordination events with entering/resulting holds
and preceding attack groups. The endpoint matters even after the last proposed
row. An empty continuation can reduce attack pressure while leaving a held key
occupied throughout the horizon.

This is a full-event reference for the declared recent-time observations, not a
claim that window counts are sufficient for every coordination judgment. Keeping
event order prevents equal per-column counts from erasing the distinction between
repetition and exchange. The initial box kernels define measurable time-window
quantities; they are not asserted to be human recovery curves.

Six focused tests cover histories with identical exact replay but different
sustained load, empty futures with held notes, silence after the last event,
composed time advances, retained coordination order and mirror symmetry. On the
reproduced eight-second Zenithfall context, the reference evaluates 66 future
rows from a 219-row recent prefix in about 5.3 ms on an Apple M5 with 24 GiB
unified memory, using Python 3.10. That one query excludes prefix construction
and is not a general runtime bound.

## Response calibration and generation ownership

The next response specification should distinguish loaded versus recovered
histories under the same future, concentrated versus distributed futures with
the same timestamps/cardinality, short bursts versus sustained repetition, and
free versus held intervals. Ranked continuations and mapper judgments supply
the supporting and contrasting cases. Fit response curves and any compact
event-time decay representation against those comparisons, including recovery
after pauses. Chart files alone do not uniquely identify a physiological law.

The canonical state and response depend on the executed chart and fixed profile,
not the requested difficulty. A changed request selects a different acceptable
response region; it does not reset accumulated history or make the same physical
continuation intrinsically easier. Control fields retain their separate scopes.

R1 can reuse its learned arrangement preference while reading the explicit player
state and a forecast over real-time horizons. A proposed future must be evaluated
through its end time, including gaps and unresolved holds. A future-response
predictor trained on sampled continuations must be distinguished from the
canonical response evaluator itself: audio and controls help predict which
continuation the policy will produce, while the evaluator assesses a specified
continuation under the fixed profile.

The scheduler must judge private H/R/R1 continuations before publication. If a
head plan cannot be materialized within the requested response region, shifting
all pressure into a long jack is not an acceptable way to match a scalar rating.
H may need a different timing proposal with musically justified gaps. This
preserves its timing responsibility and R1's spatial/count responsibility.
Recovery state and open holds persist across control boundaries; no fictitious
release or edit to committed rows is permitted.

The reference state is implemented and tested. A subsequent
[bounded planning study](sustained_response_planning.md) connects a corpus-fitted
sustained-attack reference to real-time continuation selection and reduces the
reproduced concentration. Learned coordination responses, broader style quality
and H activity planning remain unresolved. The reference state alone and the two
improved lineage cases do not establish a complete playability repair.

## Evidence identity

Model/source inspection baseline: `ba406ef827807e3a7045623dff0afa4e6084873b`;
its added outcome-learning helper changes no inference behavior from `a99519c`.
Diagnostic owner: `20260927-player-response-frontier-v1`.
Reported checkpoint: `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3`.
Paired checkpoint: `66216f4745837a185ce8b935ead0724f7ce4aa142150bad6b6b1e0fd67920805`.
Ranked manifest: `4cea2672387b6293a4da0846be479d8bd9c857e55535dc8143bd11d65b06d2c4`.
Human Stream source: `d4ec78823737c03c2e8702873bcbb2f1998e1be829e1998fe3e7ffa37075f02c`,
scope [105114,107943) ms, annotation revision
`b22a7a443783e05fee4db4b1d22b8e573ad448ae`; source cohort SHA-256
`252fe593514adc5498011b55dbf62224cba55651025233fc1ed9235078bb95b8`.
