# Agent Note: Musical phrasing and full-audio conditioning near four stars

Note ID: 2026-09-27-four-star-musical-phrasing
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: f65de370416255477f81993bfd594680ba40cbd6
Scope: Ranked four-star source/generated musical phrasing, multi-scale timing and finger-load variation, audio information flow into H and R1
Related: 2026-09-27-player-response-frontier, 2026-09-23-audio-skeleton-r1-integration

## Question and live explanations

The user reports that generation lacks breathing, multi-scale variation and
musical response, particularly near four stars. Distributed finger pressure also
has rhythm; suppressing a long-jack maximum does not provide it. Model capacity
may grow when the mechanism warrants it. The selected default core2500 and the
reported actor128 share H/audio, while their R1 action laws differ. Recent player-
state fitting improves a replay-bank objective but fails native quality and does
not change H. The previous load study remains useful but insufficient.

Inspect actual ranked sources before choosing another architecture or objective.
Live explanations are impoverished audio representations/readout, inadequate
training of their musical use, missing persistent phrase-level arrangement
choices, and R1 history/response objectives overriding variation. FullSongContext
currently compresses each Mel frequency's 50 frames linearly into a 500-ms token
before two global attention layers; the fine branch has about 2.54 seconds of
support. This is an identifiable compression, not proof of causation. Existing
whole-chart descriptor profiles are not a learned time-varying phrase plan;
their earlier mixed results must not be ignored when considering new hierarchy.

## Experiment Card: four-star-phrasing-observation-v1

Revision: 1. Accepted revision: none. Exploratory descriptive comparison executed
under the user's standing authorization to inspect, experiment and iterate.

Question: Which timing/action/phrase differences occur on matched full audio
between genuine ranked near-four-star charts and native generation, and which
model path could represent and learn those distinctions? Decision: select the
next justified audio/temporal-organization intervention; no runtime adoption.

Baseline: clean product 565d5589bb2dea56292ab3853846d7abf928c604. Model core2500
SHA 0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8 and actor128
SHA 364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3.
No fitting or model intervention. Source and generated organization are compared;
source charts are alternative valid arrangements, not unique timing targets.
The additional Stream request is a separate native condition, not a source label.

Data: four ranked TRAIN charts with cached full audio, native D in [3.8,4.2),
duration 90–240 seconds, nearest D4 within LN-fraction bands [0,.1),[.1,.4),
[.4,.8),[.8,1]. Fixed plan
artifacts/joint-audio/20260927-four-star-phrasing-v1/plan.json, SHA
9ed61800275342fa03e8282b54985c6d33930e2ac3666b124d9f177b333d24d9.
Cases: Max Burning 4.0005/rho .0429; Classic Pursuit 4.0000/.2172;
STYX HELIX 4.0044/.4853; Blizzard Heights 3.9277/.8380. Source/Mel/row hashes are
verified in preparation. This is a small developmental sample, not a population
or held-out claim. Prior native comparisons have no matched source audio in the
ranked TRAIN reference, so these provide new direct comparisons.

Procedure: generate core and actor once per case, D4 and source whole-chart LN
fraction, styles unknown; generate actor Stream-prominent on first two cases.
Seeds 273100+10*case index, same across model/control comparators. Ten complete
exports maximum. Measure source and generation on the same source-active time
range: H and per-column attack activity at several second scales, actual gaps
and fully free time, changing attack groups, and source-aligned Mel. Inspect whole-
chart plots and Lens time views with entry/exit context before interpreting local
motifs. Do not require literal reproduction of source rhythm, impose uniform
lane allocation, infer instruments from Mel, or claim audio listening.

No scalar pass/adoption threshold applies to this descriptive probe. Evidence
must name specific temporal/action relationships and their source-time scopes.
If H already fills the source's musical relief regions before R1 acts, prefer an
H/audio temporal intervention over more lane-load penalties. If comparable H
variation survives but action burden becomes constant/concentrated, investigate
R1's audio/history routing. Mixed findings retain both paths. Capacity alone is
not established by either result; decoder/learning/distribution confounders remain.

Command: uv run --extra mps python
artifacts/joint-audio/20260927-four-star-phrasing-v1/generate.py.
Apple M5, 24 GiB; one CPU thread; existing frozen model/Mel assets; per-song limit
90 seconds and summed generation-command bound 720 seconds. No network dataset
or checkpoint download; at most 200 MB new outputs with audio symlinks. Fresh
native output directory, no overwrite or implicit resume. Stop on native contract
failure, incomplete export or bound. Rendering/analysis is separate elapsed work.
Primary analogues are the local full-audio/history and shared-profile studies;
new hierarchy analogues will be researched only after the observed distinction.


## Result Log: matched source/native phrasing observations

Card four-star-phrasing-observation-v1 revision 1; Accepted revision none.
Generation handle 21894 is terminal, exit 0, with ten complete exports in
111.5518 seconds. Per-song runs are 8.129–14.446 seconds. Core/actor H streams
match in all four unknown-style comparisons. Analysis44979, rendering75236 and
held-scope calculation8969 are terminal, exit 0. No fitting or other live process
remains. Source code and immutable model/plan identities are in native/config.json;
script/result hashes are in evidence-digests.json. The analysis reports actual
source-active scopes; figures additionally show the entire audio clock.

Product f65de370416255477f81993bfd594680ba40cbd6 preserves the self-contained
report and architecture direction in
 docs/research/four_star_phrasing_and_audio_memory.md.
All local links and git diff --check pass; no new product behavior is introduced
by that report. The earlier opt-in collation change retains its twelve-test
check from 565d5589bb2dea56292ab3853846d7abf928c604.

Source/actor H counts on the same source-active scope are Max Burning736/748,
Classic Pursuit1132/1042, STYX795/563, Blizzard887/914. Eight-second H-rate CV is
.1707/.1311, .1586/.0745, .0963/.2257 and .3221/.2641 respectively. Hence variation
is not uniformly lost; increasing a CV would not establish better phrasing.
Classic has fewer H yet flatter intermediate pacing. Max Burning's source
70.878–72.142s sustains one voice without new heads; the actor inserts five H
and develops overlapping holds. The separate Stream request's largest H gap is
467ms versus source1264ms, with its own controls and output retained separately.

On STYX [1800,11800), source102 heads/20 LN become actor78/62. Mean held columns
.3719 -> 1.9642; any-held time24.34 -> 95.37 percent. Fewer attacks do not imply
relief. On Blizzard [40342,56342), the source's globally.838 LN chart locally uses
39 LN among82 heads (rho.4756); actor96 among101 (rho.9505), with any-held time
69.31 -> 94.51 percent. This loses a local texture contrast within the same
whole-chart request. The generated whole fraction.9409 also overshoots .8380.
These are observed arrangement/occupation quantities, not calibrated strain.

Viewed all four whole-song overview plots. Lens read pages2 and3 of every
source/actor context:16 of56 rendered pages, covering five seconds per chart
side. Exact viewed names and observations are in lens-review.json. Original
Max Burning changes moving chord/single groups into a sustained voice and back;
Classic repeats short-LN/chord figures without long blanks; STYX develops from
taps toward holds; Blizzard's quieter texture mixes separated taps with short
paired holds. The actor often replaces these with repeated-column taps or more
continuous held texture. Source need not be the only valid arrangement. No
human labels, literal listening, semantic pass or population coverage is claimed.

## Architecture interpretation and selected research direction

The user additionally suspects missing attention-like historical understanding
and permits scaling when needed. Current global audio already has attention,
but H/R/R1 history uses query-independent causal TCN summaries. Their committed
tokens do not bind audio at historical times to the selected pattern. Current
music is fused after compression. This imposes a concrete information-routing
burden; it does not prove a TCN cannot learn the behavior or that attention alone
will fix it. R1's additive mirrored-head cancellation was partly addressed by
actor layout modulation and must not be reintroduced by an identical shared
attention vector.

Selected Explore result: TEST the joint audio/history information path, rather
than another narrowly adjusted sustained-load penalty. Develop nonlinear
multiscale audio tokens plus current-music-queried committed history; retain
fast local decoding and exact/player state. H memory contains only H and aligned
audio. R memory contains timing roles and allowed LN occupancy/origin facts.
R1 memory contains complete committed actions/aligned audio and reads H preview;
R1 still owns counts, columns, TAP/LN and release subsets. Real elapsed time,
causal chart masks and complete audio in both train/infer are required. No hard
chorus labels or uniform per-lane activity target. New model capacity can be
several times4.6M if actual startup/dense-passage profiling warrants it.

Primary analogues checked: Music Transformer https://arxiv.org/abs/1809.04281
for relative retrieval of repeated musical structure; Transformer Hawkes Process
https://proceedings.mlr.press/v119/zuo20a.html for asynchronous event-history
attention; MusicVAE https://proceedings.mlr.press/v80/roberts18a.html for a separate
possible slower arrangement process. None establishes playable 4K results or
our factorization. This is an adaptation of known primitives, no novelty claim.
The slower sampled plan remains a distinct future branch, not implemented.

Keep supervision/control confounders visible. Recent R1 fits froze audio/H/R;
they cannot test improved musical audio representation or H pacing. A global
LN target can coexist with locally TAP-heavy passages. The finite integral LN
feedback pushes toward a global fraction after such a passage and may compete
with musical texture, but its causal contribution is unmeasured. Do not replace
the architectural question with another low-level controller sweep, and do not
claim the attention proposal already resolves that issue.

Next Design task must set a concrete memory/encoder implementation, matched
training comparator, actual history/training-window scope, source/annotation
coverage, native multi-scale/Lens guards, and Mac startup/dense-service budget.
Jointly train the new audio/history path; preserve full-song conditioning and
actual scoped controls. No such new architecture fit has been launched yet.
Current default model/sampler and scoped interface are unchanged. The complete
playable-system goal remains active and unachieved.
