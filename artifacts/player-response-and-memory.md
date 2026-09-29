# Sustained pressure, breathing and the failed memory repair

The [conditional-learning candidate](conditional-learning.md) prompted the
human's concrete report that D4 Stream could collapse into severe long jacks,
with flat pressure and inadequate breathing. The requested response state was
based on committed gameplay, actual elapsed time and finite future durations,
including coordination and releases. These requirements were broader than
column balance or an arbitrary number of future rows. Sources: [report,
private, local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0687),
[time horizon](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0694)
and [state proposal](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0696).

## Response selection helped a covered failure channel

`7ea3b95` introduced committed-state observations and actual-time attack
responses; `24e786b` added a four-second forecast/two-second publication planner
with up to four private R/R1 candidates on unchanged H. In nine matched Stream
outputs, integrated sustained attack excess fell 98.46%, from mean .027516 to
.000425 seconds; two original extreme single-column contexts were visibly broken.
Two zero-cost Take outputs and the LN control output were exactly unchanged.
Whole-star MAE barely moved (.47893→.48392), and Tech/Trill still exhausted
candidate budgets with positive cost. [Planner evidence](/Users/l/projects/ensomi-model/docs/research/sustained_response_planning.md).

This demonstrates useful proposal alternatives and selection for a measured
channel. It does not establish Stream semantics, held burden, musical rests or
complete difficulty calibration. Fixed H cannot create absent attack gaps.
Rated real examples with nonzero excess prevent calling every exceedance BAD.

## Teaching the proposal that response did not generalize

`94d0b08` added a 12,288-parameter player-state input and open-ended trace
scoring, keeping real holds open beyond the response endpoint. Eighteen focused
historical tests checked causality, native probability/gradient agreement,
initial identity and ownership. These are implementation checks.

Matched 384-update source-only and source-plus-response fits used identical
768 factual draws, plus 1,152 sampled four-second futures for the latter.
The fixed replay bank covered twenty TRAIN songs, but its Stream contexts
already had zero initial excess. Training-bank response improved 60.8% versus
initial; nine reserved developmental full-song Stream cases worsened:

| Model | Mean attack excess, seconds | Whole-star MAE |
| --- | ---: | ---: |
| Initial | .027516 | .47893 |
| Source-only | .144229 | .85255 |
| Source + response | .107160 | .74113 |

A new native witness had 38 attacks on one finger in four seconds while other
fingers were available. Lesioning the entire new input changed hot-column
probability by only .46 percentage points on that history; the fitted shared
policy still carried most of the change. Neither removing that input nor
extending unchanged fitting was established as a repair. The
[player-state study](/Users/l/projects/ensomi-model/docs/research/player_state_conditioning.md)
at `565d558` preserves the successful bank behavior and failed transfer together.

## Why musical memory was reconsidered

Human feedback shifted attention back to real ranked approximately-four-star
phrasing, not just limiting one peak. The suggestion of attention was explicitly
an intuition to analyze. [Private, local sources](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0726)
and [attention hypothesis](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0730).

The [four-song phrasing study](/Users/l/projects/ensomi-model/docs/research/four_star_phrasing_and_audio_memory.md)
(`f65de37`) found that fewer attacks need not create relief. In STYX's inspected
interval the source had 102 heads with any-held time 24.34%; the actor had
78 heads with any-held time 95.37%. Classic had fewer H but much flatter
eight-second H-rate variation. High global LN amount did not explain away
sources' local TAP-rich contrasts. Some large musical changes were already
followed; uniform absence of musical response was not the finding.

The proposed new path binds earlier chosen timing/actions to their original
audio and retrieves them using current musical context. Existing full-audio
attention alone did not provide that arrangement relation. `b130dfb` built a
7.616M audio pyramid and separate causal H/R/R1 readers. Early zero-initialized
trajectory equality, padding/gradient checks and bounded MPS probes established
integration and cost, not trained musical benefit.

## Joint memory fitting and the misleading zero-pressure repair

The [matched memory study](/Users/l/projects/ensomi-model/docs/research/audio_memory_joint_fit.md)
trained all audio/H/R/R1 paths for 384 updates on 768 shared examples. Every
arm generated 28 cases. Memory mean Stream excess .38230 was better than
continued baseline .78553 but much worse than common initialization .04447;
star MAE and restored control also failed. All 84 exports completed. Memory's
slowest loaded/cached-Mel first-thirty publication was 1.59063 s and service
1.14145 s, so bounded throughput was not the principal failure.

Follow-up fixed-prefix probes found insufficient explanatory power in long
scope clocks alone and a substantial contribution from the fitted H audio/
control base. Replacing only that base with the unfitted model's own encoder/
base reduced mean Stream excess to zero, but median H count fell to .31832
of memory. Requested D4 Zenithfall became about 2.2–2.4 stars, three startup
bounds failed and separate before/override difficulty ranges failed. A local
Classic type-mix improvement survived; Blizzard held texture remained poor.
Products `fc641aa`, `e588a69` and `14ee153` preserve this diagnostic rather
than selecting it.

This is an especially useful regression counterexample: zero overload and
improved average error can coexist with underfilled requests, substituted held
texture and startup failure. No conclusion that attention is useless follows;
neither does the result justify expanding memory first.

## Evaluation and the next redirection

The human explicitly prioritized reusable evaluations beyond single Lens crops.
`138a1f5` and `251ba0f` added exact-time scoped observations, multi-scale
pressure/recovery, audio correspondence and publication deadlines. Later
witness selection retained all response scales; a 47.662-s distributed-load
episode would be missed by inspecting only the hottest four-second finger.
The framework supplements semantic review rather than reducing all quality to
one scalar. [Evaluation recovery](evaluation-and-evidence-boundaries.md).

Imported critique then challenged prefix-balance LN feedback, support closure,
request exposure and fixed-bank learning. [The following diagnosis](ln-feedback-and-proposal-diagnosis.md)
tested those objections rather than assuming additional memory would resolve
them. Primary local owners are `20260927-sustained-response-planning-v1`,
`20260927-player-state-r1-learning-v1`, `20260927-audio-memory-joint-fit-v1`
and `20260927-head-base-native-replay-v1` under `artifacts/joint-audio/`.
