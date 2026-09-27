# Agent Note: Musical phrasing and full-audio conditioning near four stars

Note ID: 2026-09-27-four-star-musical-phrasing
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 565d5589bb2dea56292ab3853846d7abf928c604
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
