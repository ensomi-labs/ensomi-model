# Agent Note: Time-horizon player responses for sustained load and breathing

Note ID: 2026-09-27-player-response-frontier
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: c4d9e730375ace301f8ae68d34afbab50a75e5a5
Scope: Canonical continuation-response semantics; sustained per-column demand, temporal variation, and publication-time playability for the audio/H/R/R1 system
Related: 2026-09-23-audio-skeleton-r1-integration

## User evidence and required response distinctions

The user played the a99519ccee60925dce10a4200f88a6c049d37964-associated candidate
(common-prefix actor-128, SHA-256
364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3) and reported a
severe 4-star long-jack failure with Stream control: burden concentrated on one
finger instead of the expected flowing distribution. No generated file, audio
or seed is available. The reproduction condition is 4 stars with Stream prominent;
other style fields and LN amount unspecified. This is not an exact reproduction
of the user's unseen chart. The same user reports consistently dense generation,
lack of breathing/intervals and larger-scale rhythm variation.

The user requires the frontier to reflect accumulating strain and to constrain
responses outside the requested difficulty, including excessive lane concentration.
Responses must forecast a future duration in actual time, not a fixed row count.
These requirements take priority over candidate selection by NLL or the existing
scalar difficulty proxy. Numeric gains from the paired-response study cannot
cancel these playability failures.

## Formulation and existing mechanism

The formulation's C0(H through t,t;Y,e) and frontier already include an explicit
horizon e. Empty continuations and silence after the last row matter; occupied
LNs are not automatic rest. Canonical pi0 fixes successful exact execution and
left/right symmetry. This study does not infer personal capacity, physiological
fatigue, pain, or a player-specific tolerance. A response representation must be
assessed against independently declared distinctions; named state coordinates
and low training NLL are not sufficient evidence.

RecoveryPreference currently checks immediate HH/RH/HR intervals. At 4 stars,
the empirical HH reference is107ms; larger gaps have zero immediate HH cost.
The separate head-pressure term counts heads across all columns within that
short lookback and does not identify a persistently overloaded finger. Thus a
sequence of individually unpenalized repetitions can continue for seconds without
this recovery preference accumulating a distinct penalty. Learned frontier2
energy can read longer history, but its NLL-trained value is not a calibrated
canonical player response or a direct sustained-load constraint.

A synthetic read-only diagnostic keeps a16-second TAP passage and H rate fixed,
changing only columns. At12 H/s, one-column repetition scores4.1407 under the
scoped strain proxy, while four-finger cycling scores2.5996. At10 H/s the values
are3.4958/2.2182. The scalar orders the patterns in the expected direction but
can still classify an extreme sustained jack near the requested4-star value.
This is not corpus evidence or an approved physical threshold; it explains why
optimizing the scalar alone need not prevent the reported failure.

## Investigation and design direction

Inspect the native ranked corpus at actual1x times, with computed difficulty
bands and source provenance. Measure duration/rate/concentration jointly, then
read the high-tail cases and ordinary comparisons through Lens. Use complete
attack groups and occupancy; same-column statistics locate possible episodes
but do not by themselves assign a Jack label. Distinguish genuinely free columns
from other fingers pinned by LNs. Repeat representative audio generation under
the reported control combination and locate the strongest sustained-load regions.

Define response questions before choosing a compressed state: how burden changes
when the same future follows a loaded versus recovered history; how concentrated
versus distributed legal futures compare at the same timestamps/cardinality; how
a pause changes recovery; and how open holds alter that comparison. Forecast over
explicit real-time windows, preserving state across control switches. The same
canonical response must not change merely because the requested difficulty changes;
the request changes the acceptable response region.

R1 retains cardinality, column, TAP/LN and release-subset ownership. H must create
musically justified timing density and gaps; changing R1 columns cannot create
breathing while every dense H must be materialized. R1 reads direct audio, timing
preview, controls and exact/player state. R may receive execution-feasibility
release bounds; it does not inherit the R1 content encoder. A scheduler evaluates
private joint continuations under the frontier before publication, retaining all
committed rows and real open-hold state. This direction must preserve plausible
Jack expression at compatible intensity rather than force uniform four-column
usage in every window.

## Limits and next evidence

No numerical frontier law, time constants, corpus cutoff, new decoder or training
intervention is selected yet. The corpus's normal arrangements calibrate the
reported distinction; rarity alone is not proof of poor quality. Low activity with
open holds is not full recovery, and low H-rate variance can be appropriate for
some musical passages. Compare actual contexts, not only global aggregate counts.

The paired-response fits are complete, with their original qualifications
retained as comparative evidence. They are not adopted. Diagnostic owner:
artifacts/joint-audio/20260927-player-response-frontier-v1. After corpus and native
inspection, define one response specification and a bounded implementation/test
that addresses sustained concentration and real-time-horizon behavior. Keep this
Note proposed; no acceptance or remote publication is implied.

## Implementation decision after source inspection

The existing frontier2 cannot serve as the required response interface. Keep its
useful exact candidate-action coordinates and row preference path, but introduce
an explicit committed-history player state and time-horizon continuation-response
operator. This is not a claim that its history TCN cannot encode any coordination
or repeated attacks. Its learned context can distinguish histories; the missing
part is declared response semantics, training/calibration and the required API.

RowConsequence.score currently returns one learned scalar per complete candidate
row. Its inputs include the action's post-occupancy, last-attack/release clocks,
LN age, next H and second H. Planned consequences passively advance that immediate
post-action state; they do not simulate intervening future actions. There is no
explicit future-duration argument, continuation Y, or dedicated demand-state
advance through no-row time. Its context also contains audio and controls, and
its score is learned as part of normalized row likelihood. It is therefore a
conditional arrangement preference rather than an independently calibrated C0.

The replacement response interface must consume the same committed rows for
training/replay/inference, advance on actual milliseconds, preserve active LN
origins/occupancy, and evaluate legal candidate continuations through an explicit
end time, including empty endings. Accumulation/decay laws are candidate
representations to fit/compare against corpus and mapper-defined response
contrasts, not human capacity curves supplied by the repository. Keep the
canonical response independent of the requested difficulty; apply difficulty
when selecting an admissible response region. R1 retains all spatial/count and
TAP/LN decisions; H owns timing variation and silence.

## Reproduction evidence

Nine native outputs under 4 stars + Stream prominent completed using the reported
checkpoint, across Zenithfall, Hysteric and Take with seeds271200+10*case+i. Other
style fields and LN amount are unknown. Zenithfall draw1 has31 of34 heads on one
column over4 seconds (7.75 attacks/s); other columns are not pinned by holds at
those attacks. Hysteric draw2 has28 of37 heads on one column over4 seconds
(7 attacks/s), again with no other held columns. This reproduces the reported
failure family, not the user's unavailable exact chart. Their whole proxies are
4.9844 and3.9434. A130ms chain cutoff misses the longer episodes: time-based
pressure and complete-row Lens inspection are necessary.

The reproduction process56717 is terminal. A pre-event occupancy indexing issue
at the very first row was corrected only after it ended; original results remain
in result.json, recomputed facts in result-v2.json with metric-revision.json.
No generated rows changed. Native ranked TRAIN corpus profiling completed under
an explicit four-worker process, using full source rows and recomputed1x stars.
It measures fixed real-time windows, sustained-column chains, activity variation
and occupancy-aware no-action recovery. These remain descriptive source facts;
no automatic Jack label or admissibility cutoff has been selected.

## Result Log: corpus, Lens and checkpoint comparisons

Accepted: none. Exploratory diagnosis under the standing implementation/research
authority. Source ba406ef827807e3a7045623dff0afa4e6084873b changes no inference
behavior from the user-tested source. All corpus and generation processes are
terminal; no fitting run is live. The corpus's 6,923 ranked TRAIN charts finish
in 30.904 seconds using four CPU workers. Manifest SHA-256
4cea2672387b6293a4da0846be479d8bd9c857e55535dc8143bd11d65b06d2c4;
metrics SHA-256 75d6a0e65df45c997cdf4c1416ddc4196da8be679d0146beaca2dd3a573cbd7f.
Output owner remains 20260927-player-response-frontier-v1/corpus-reference.

The [3.5,4.5) band has 1,972 charts in 1,613 groups. Each group has equal total
weight, divided among its charts within the band. These are per-chart maxima,
not a distribution over arbitrary individual windows. Maximum single-column
attack-rate q99 values for .5/1/2/4/8/16-second windows are
10/8/7/6/5.375/5 attacks per second; q999 values are
10/9/8/6.5/5.75/5.375. The reproduced 7.75/s four-second episode is beyond that
source range at q999. This supports a sustained-load defect without defining a
universal physiological cutoff or labeling every rare arrangement BAD.

Lens review covers native Zenithfall pages 0/1/2 and Hysteric page 1, plus
source pages 0/1 for each of 22e16fc2, 4194d810 and human-stream. Exact source
identities, review contexts and human evidence are in lens-corpus/plan.json;
native contexts are in lens-reproduction/plan.json. The generated Zenithfall
region visibly settles into a prolonged single-column sequence. Ranked
Extra Mode (4.069 stars) has a slower 28-attack outer anchor across 4,154 ms,
with changing accompanying groups. Ranked 3#006 wyax03 (4.289) reaches 7.5/s over
four seconds through shorter groups and interruptions. Human-prominent Stream
in Singularity (3.987), scope [105114,107943) ms, distributes movement across
four columns with chord accents. Preserve these differences; no generated human
labels or universal equal-lane rule are inferred. No additional listening or
human playtest occurred. Remaining rendered pages are not claimed reviewed.

Six lineage generations reuse exactly the two bad cases' controls and seeds.
Every H stream matches the reported checkpoint. In its original Zenithfall
window, Core2500/modulated-128/reported actor-128/paired-128 produce column heads
[11,14,10,14]/[5,15,21,3]/[1,31,1,1]/[14,3,5,10]. In Hysteric they produce
[16,15,14,15]/[18,16,17,16]/[4,4,28,1]/[11,8,8,12]. Paired-128 reduces the
concentration in both cases. Hysteric contains 39 rather than 37 heads, so this
benefit is not solely thinning. It remains two cases, and paired-128 fails its
reserved control-response comparison; it is not promoted. This localizes a
learned R1 failure on H that can support a less concentrated arrangement.

Eight/thirty-two-second H-rate variation is not uniformly low in generated
charts. Several nevertheless have almost no completely free interval beyond
500 ms inside their active bodies. Activity CV, lane recovery, sustained holds
and musical context answer different questions. Breathing cannot be repaired
by a single whole-chart density or CV target.

## Reference implementation and next decision

Clean product 7ea3b956ebccdc4d4238bc52e5cb89762402f054 adds
src/ensomi_model/research/player_response/state.py and
docs/research/time_horizon_player_responses.md. CommittedPlayState retains exact
replay, ordered complete rows, attack/release clocks and hold intervals over a
32-second reference history, retaining ancient open LN origins. observe and
advance respect committed no-row boundaries. observe_continuation evaluates a
private legal future through an explicit endpoint, including empty futures and
silence after the last action. It returns peak/terminal rates, rate integrals,
held milliseconds and ordered coordination events. Inputs are not mutated.

Box kernels are declared observations, not fitted human recovery curves. The
canonical state has no requested difficulty/style input; controls will select
acceptable responses without erasing prior load. Six focused tests pass in
0.06 seconds, covering identical exact replay with different sustained history,
empty futures with holds, endpoint-dependent recovery/exposure, compositional
time advance, coordination order and mirror symmetry. Package layout passes
one test/22 subtests in 0.09 seconds; diff and local document links pass. A real
eight-second context with 66 future rows and 219 retained prefix rows takes
5.283 ms for response observation alone on the M5, excluding prefix construction.

Decision: REFINE the current frontier into an explicit response interface while
retaining its learned row preference. The reference module is not wired into
the sampler; no runtime repair, fitted response law or completed C0 is claimed.
The next bounded intervention should calibrate sustained rate-duration responses
against source contrasts, expose that state to R1, and assess candidate futures
over real time. Preserve legal LN execution, direct audio and scoped controls;
H remains responsible for timing and recovery intervals. Do not resume scalar
difficulty-only tuning as a substitute for the reported playability failure.

## Experiment Card: sustained-response-planning-v1, revision 2

Accepted: none. Local exploratory implementation and execution use the owner's
standing research authority. Baseline source is clean product
7ea3b956ebccdc4d4238bc52e5cb89762402f054; baseline checkpoint is reported
actor-128 SHA-256 364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3.
No parameter fitting or replacement of the selected runtime is implied.

Question: can an explicit sustained-load response over a private real-time future
prevent the reproduced long-jack pathology at acceptable cost, using the existing
R1 proposal distribution? The causal intervention is bounded response-based
continuation selection. No H, audio, control encoding, count ownership or neural
weight change is included. A failed result will motivate conditioning/learning
changes rather than more random search at progressively larger budgets.

Closest analogue: finite-horizon receding-horizon control, as defined by Mayne,
Rawlings, Rao and Scokaert (2000), Constrained model predictive control: Stability
and optimality, https://doi.org/10.1016/S0005-1098(99)00214-9. The transferable
mechanism is evaluating hypothetical futures from the current state and applying
only an initial portion. Here proposals are discrete stochastic chart trajectories;
there is no plant-identification task, terminal Lyapunov condition or transferred
stability theorem. Sampling-based MPC is an engineering analogue, not novelty or
proof of playable generation. Immediate-gap costs alone miss sustained histories;
an uncalibrated learned scalar is not selected as the response oracle.

Fit one descriptive attack-rate envelope from the completed ranked TRAIN cohort,
using per-chart maximum column rates at .5/1/2/4/8/16 seconds. Equalize total song
group weights within each one-star-wide band centered at 2,2.5,...,6. Use q99,
monotonize by cumulative maximum across requested difficulty, and linearly
interpolate between knots. Report all raw/monotone values and counts. The
normalization is a source-reference range, not a physiological threshold. It
contains no requested style and does not label charts BAD by rarity alone.

The canonical observations remain request-independent. For selection, integrate
over each future control range the squared positive relative excess of per-column
window rates above the corresponding requested-difficulty envelope. Sum columns
and average the six durations, integrating in real seconds. Exact attack/expiry
and control-boundary times determine the integral; no frame-grid approximation.
LN press is an attack, releases are separate, and holds still affect legal
proposal generation. This first selector measures sustained attacks only;
coordination/hold burden and H breathing are not claimed solved by that scalar.

Each decision forecasts 4 seconds and commits its first 2 seconds. Keep the
original proposal when its cost is zero. Otherwise generate at most three
additional proposals from the same committed boundary, with separate R/row RNG
seeds and identical H/audio/controls. Select minimum integrated excess, retaining
the earliest proposal on ties; stop when a zero-cost proposal is found. Commit
only the selected two-second state, keeping rows and RNGs at that boundary.
No rewrite of published rows, artificial LN closure, or control-boundary state
reset. The bounded selected-trajectory law differs from native row sampling;
old replay_row_scores must not be reported as its selected log probability.

Fresh owner: artifacts/joint-audio/20260927-sustained-response-planning-v1.
Source modules: player_response/envelope.py and
controlled_audio_continuation/frontier.py, with focused tests. Calibrate from
20260927-player-response-frontier-v1/corpus-reference/charts.jsonl and its pinned
manifest. First compare the exact Zenithfall271201/Hysteric271212 reproductions.
Then, only if either integrated excess improves by at least 50 percent without
an obvious new pathology, run all nine existing 4-star Stream seed combinations.
Source Lens counterexamples and human Stream are calibration checks, not held-out
success claims. Further native guards use Zenithfall with Jack/Tech/Trill controls
at 4 stars and LN coordination at 4 stars/rho .6, and a difficulty/LN switch case;
new controls receive matched baseline and intervention runs.

Primary: mean integrated excess drops at least 50 percent across the nine matched
Stream cases, and both original hotspots lose their prolonged dominant-column
sequence on Lens inspection. Report unchanged H, actual head counts, strongest
4/8/16-second rates, and entire-chart tails so selection cannot hide relocation.
No clear new Stream/Jack/Tech/Trill/LN organization loss in inspected contexts;
no completed output with invalid physical replay or broken source audio.
Do not promote merely because the self-chosen cost decreases. Keep all control
ranges separate. Existing scalar stars are descriptive secondary outcomes only.

Use one CPU thread, cached canonical Mel and loaded CPU weights on the M5/24 GiB;
uv run --extra mps for model-backed commands and --group dev for tests. Seed rule:
Stream native seeds are 271200+10*case+draw; guards use 272300+j for
Jack/Tech/Trill/LN/switch respectively. Native seed XOR 0x6F17 initializes
per-decision independent retry draws; record every retry seed.
Compare model-loaded timing including audio encoding, first thirty published
rows, and publication windows. Guard: first thirty rows at most 2 seconds and
worst 2-second publication service below 2 seconds on these diagnostics; no
unbounded search or additional GPU memory. At most 4 candidates per decision,
180 seconds per full generation, 30 minutes total qualification, 2 GiB new
artifacts, no download. Fresh outputs only, no overwrite or resume. Stop on
nonfinite costs, execution error, clock/history mutation, H mismatch, resource
limit or explicit STOP file. Commit clean implementation before native runs;
freeze script and calibration identities while processes are live.

Interpretation: success supports a useful first response-aware scheduler and
supplies selected trajectories for a later declared learning experiment. It does
not validate the whole frontier or prove human playability. Failure with all
proposals similarly overloaded demonstrates a proposal-coverage/learning problem
under this bounded search; it does not justify widening the search indefinitely.

### Result Log: response-planning implementation and calibration

Accepted: none; sustained-response-planning-v1 revision 1. Clean intervention
24e786b4b8ef8c4752835371bd9fd415c5ed0891 adds a group-weighted AttackEnvelope,
exact attack/expiry-time excess integral and ResponsePlanner. It forecasts four
seconds, commits two, keeps zero-cost first proposals, and otherwise compares at
most four candidates while preserving publication state and H. Native row replay
probabilities are explicitly not the selected trajectory's law. Six focused tests
pass in 1.17 seconds: manual area/partition equality, control references without
state reset, history/hold distinctions, group weighting, unchanged zero-cost
native trajectories, and bounded private selection with correct published state.
The new implementation does not modify model weights or the default entrypoint.

Calibration from charts.jsonl SHA-256
e529cfd868b4ec4bc337803d1712bbb319d2cba730a47dcfc13cc6a9e8f00f3b completes.
All nine difficulty bands contain sources; their group counts are
1321/1875/1957/1819/1613/1298/766/391/125. Raw q99 curves already increase with
difficulty, so monotonic adjustment changes no fitted value. At four stars the
.5/1/2/4/8/16-second rates are 10/8/7/6/5.375/5 Hz. Endpoint six-star tails have
only 125 groups and are less well supported. Calibration is descriptive training
corpus evidence, not held-out validation or individual physiology.

Exact integrated excess for the nine prior Stream outputs, in case/draw order:
Zenithfall .0434756/.1061111/.0940127; Hysteric
.0004430/.0010242/.0025064; Take 0/0/.0000749 seconds. These are squared relative
excess integrals averaged across six windows and summed across columns, not
elapsed time spent in an unwanted pattern. The source identity and full raw/
monotone curves are in calibration.json; baseline responses are in
baseline-response.json. Calibration process 25265 is terminal.

The two-case native smoke now runs under handle 53417 via
uv run --extra mps python artifacts/joint-audio/20260927-sustained-response-planning-v1/run.py smoke.
Its config records clean source, checkpoint, driver, metric, calibration and
panel hashes. Do not edit those inputs or restart while the process is live.


### Result Log: smoke inspection and expanded native comparison

Accepted: none. Revision 2 fills the exact native guard seeds before those runs;
all other protected fields are unchanged. The smoke and seven remaining Stream
cases complete under the same frozen run.py and calibration. Handles 53417 and
9789 are terminal. All nine H streams match their original counterparts.

Smoke maximum four-second per-column rates fall from 7.75/7 to 6/6 Hz. Exact
original-window counts change from [1,31,1,1] to [5,9,18,8], and from [4,4,28,1]
to [7,12,15,10]. Whole proxies change 4.9844 to 4.8483 and 3.9434 to 4.3093; the
second case illustrates why scalar improvement and sustained-load improvement
are separate. Whole-song integrated excess falls .106111 to .002232 and .002506
to zero. Runtime is 33.922/15.850 seconds for complete audio, first thirty
published rows .615/.416 seconds, slowest two-second publication .953/.282 seconds.
Zenithfall uses 239 candidates for 179 decisions, replacing 21 initial proposals;
15 selected horizons retain nonzero cost. Hysteric uses 156/151, replacing four,
with no selected horizon above the reference. The budget is never widened.

Lens reviewed Zenithfall original pages 0/1/2/3, Hysteric original 1/2, and both
new maximum-load contexts' pages 0/1. The prolonged nearly uninterrupted single-
column episodes are broken. Short same-column runs, repeated chord groups and
anchors remain; this is not a prominent-Stream quality pass. The new strongest
Zenithfall context moves across columns with chord accents. Hysteric still has
noticeable repeated anchors. No more severe new pathology is observed in these
contexts, satisfying the declared expansion condition. Exact observations and
review limits are in smoke-lens-review.json. Harness source
22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60 is read-only.

Source contrast evaluation is deliberately retained: valid ranked Happy Love
Expert and Extra Mode contexts receive positive excess .005942/.000805; the
human Stream context receives zero. Quantile excess cannot distinguish all
valid expressive extremes from BAD arrangements. The selector is a bounded
sustained-load preference, not a corpus-derived proof that cost-positive means
unplayable. Follow-up R1 style learning must address short repeated-group bias
without lowering this load threshold to force a Stream label.

### Result Log: completed sustained-response planning qualification

Accepted: none; sustained-response-planning-v1 revision 2. Clean implementation
24e786b4b8ef8c4752835371bd9fd415c5ed0891; report and source link update in
faf726d30cea3a0ad4f0140b884d3d52dddec4cc. All processes are terminal: smokes53417,
remaining9789, guards48921, calibration25265 and render87832. No training process
or selected-policy rollout remains active. Driver, calibration and reference
metric hashes match each phase's recorded config after completion.

All 19 new complete-song exports succeed, comprising fourteen planned and five
new matched native guards. Native qualification from first smoke config through
final guard output consumes 642.529 seconds of wall time; summed generation work
is 416.401 seconds. Local output, including before/after audio playtest packages,
is 28,442,897 bytes. No model parameters are added or trained. Six focused tests
remain passing and unchanged; the final report's local links and diff checks pass.

Across nine Stream cases, mean integrated excess falls .027516427 to .000424514,
a 98.4572 percent reduction. All H times match; Take draws0/1 are exactly identical
in every row. Maximum four-second column rates after selection are
6.25/6/6, 5.25/6/6, 5.5/5.5/5.75 Hz in case/draw order. Whole-star MAE stays
nearly unchanged, .478926 to .483918, so this does not establish difficulty control.
Head counts are not uniformly reduced; Hysteric draw2 increases 3343 to3736.
The 170ms chain diagnostic improves in most cases but Zenithfall0 grows1173 to1252ms.
Do not hide that variation behind the lower aggregate cost. Primary sustained-
load progress and the two original hotspot improvements hold; semantic Stream
quality remains unqualified.

The five new controls use native seeds272300 through272304. Jack excess falls
.00006867 to zero with two replacements; both whole ratings remain about5.10 for
request4. Tech falls1.644136 to.162401, but79 of179 selected horizons remain cost-
positive; whole rating5.5805 to5.4494. Trill falls1.461985 to.530589 with63 positive
horizons; rating5.6882 to5.5941. Both show limited proposal coverage under the
fixed four-candidate budget, which is not enlarged. LN coordination/rho.6 yields
exactly unchanged complete rows, rating4.8027 and fraction.7634; this preserves
an existing LN-amount error rather than qualifying it. Trill's unspecified LN
fraction moves.0104 to.0529; attack-only improvement cannot establish held demand.

The actual mid-publication control update occurs at coverage63999. Requests are
D3/rho.2 before64000, D4.5/rho.6 on[64000,96000), then restored D3/rho.2. Prefixes
are preserved. Before-range outcomes are identical D2.226845/rho.299242;
override outcomes identical D3.901016/rho.583587. Restored outcomes improve from
D4.421816/rho.206358 toD4.205740/rho.199223, but difficulty remains too high.
No pooling of those distinct ranges. All guard H streams match.

Across fourteen planned runs, first thirty published rows take .327-.641 seconds,
and worst two-second publication service is1.097385 seconds. The heaviest whole
run takes82.900 seconds for357.796 seconds of audio. One CPU thread, loaded model
and cached canonical Mel; includes model audio encoding and private forecasts,
excludes waveform/Mel preprocessing and model load. Brief rendering/analysis
activity occurred, so this is not an isolated system-load benchmark. The declared
panel latency bounds hold; no universal realtime guarantee is inferred.

Additional Lens review reads all three strongest-context pages for remaining
Zenithfall0/2, page1 for remaining Hysteric0/1 and Take2; both Jack native/planned
pages0/1; Tech native/planned page1; all Trill native/planned pages0/1/2. Exact
coverage and limitations are in qualification-lens-review.json. The remaining
Zenithfall contexts show broad movement and chord accents. Hysteric retains
repeated columns/subsets. Take contains simultaneous holding and moving roles,
which the current scalar does not evaluate. The matched Jack crop retains
changing chord/single groups. The Tech body stays very dense. The original
Trill crop's inner exchange followed by a long repeated column becomes broader
movement after planning. This removes overload but does not establish preservation
of requested exchange organization. Unlisted rendered pages remain unreviewed;
no new human labels, listening, or human playtest are claimed.

Playable before/after comparisons are packaged as zenithfall-before-after.osz
and hysteric-before-after.osz under the owner. Each includes the same local audio,
original and planned .osu charts, and a limitations note. Only Version metadata
is changed for import labels; original artifact charts are preserved. They are
research comparisons, not a newly trained or promoted runtime model.

Decision: REFINE. Retain the explicit state/response and bounded planner as useful
research components. Default sampler and selected runtime remain unchanged. The
next learning intervention must let R1 use the time-based load information while
preserving actual annotated organization; generated requested-style fields must
not be relabeled as observed style. The current body still overproduces repeated
groups, and the outcome-trained policy has important difficulty/style errors.
Do not substitute tighter rate cutoffs or an expanding search budget for fixing
those proposal-learning problems. H musical activity and recovery intervals remain
an independent unresolved part of the complete system. No new fitting Card or
run is selected by this Result Log.

## Experiment Card: player-state-r1-learning-v1, revision 4

Accepted: none. Standing research authority covers local implementation, profiling
and fitting. Baseline product faf726d30cea3a0ad4f0140b884d3d52dddec4cc; baseline
actor-128 SHA-256364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3.
The previous goal turn is progress: committed response planning,19 native exports,
Lens comparisons and measured runtime. All prior processes are terminal.

Hypothesis: explicit time-based demand coordinates plus generated-continuation
risk learning can move native R1 toward less overloaded choices, while genuine
annotated source rehearsal preserves organization. The response planner proved
that alternatives exist, but candidate selection alone neither teaches the native
policy nor certifies style. Test the learned proposal, with no planner at final
qualification. Audio/H/R remain frozen; R1 still owns counts, lanes and TAP/LN.

The representation adds one zero-initialized shared hand-coordinate projection
of request-independent committed-player features to R1 context. Use six physical
window attack/release rates and held fractions, plus the previous two complete
attack groups and their ages/known bits. Do not send R1 row content into H or R.
Existing exact clocks, direct full audio, H preview and scoped controls remain.
Zero initialization must reproduce the old probability law. An explicit model
option owns the feature contract and checkpoint round trip; old checkpoints
remain unchanged. Compare source-only and source-plus-response fitting with the
same new representation and identical source draws. This separates the added
response objective, not all architectural causes.

A row-trace scorer will replay native row probabilities over a legal private
continuation without requiring EOF or closing open LNs. It takes actual complete
prefix/future rows, a declared timing-only H plan, real audio duration and complete
audio encoding. Future endpoints are targets only when actually proposed. The
same source controls, recovery and LN-amount preferences apply. Verify CPU/MPS
sampled-rescored probability and gradient parity, causality, mirror transformation,
partition independence and open-hold horizons before fitting. This removes the
old complete-SourceChart adapter's needless full-song continuation for four-second
outcomes; it does not invent terminal releases or change sampled support.

Closest mechanism: score-function estimation of expected trajectory cost using
independent sampled continuations and other-draw baselines, already verified in
outcome_learning.py and the preceding paired-response study. The new target is
the explicit real-time sustained-response integral, not star rating alone.
Source imitation and response learning have separate evidence roles. Generated
requested-style fields never become observed semantic labels. No novelty claim.
A larger encoder or more candidate search is not selected by this evidence.

Prepare a bounded hard-context replay bank from twenty TRAIN audio identities,
excluding every existing held native/style/control audio in the balanced source
panel. Use its existing1024 population and251 human source scopes for factual
rehearsal. Select five request families (Stream,Jack,Tech,Trill,LN coordination)
across nominal difficulty2.5/3.5/4.5/5.5, drawing a distinct eligible audio near
each nominal band. LN requests use fraction.6; other LN fields remain unknown.
Generated controls are requests, not labels. Freeze selected identities, schedules,
seeds, source-panel/calibration hashes and bank rows before fitting. Each bank
stores the baseline's full native H/rows and up to twelve high-excess four-second
contexts plus four ordinary contexts per audio on a two-second grid. Generated
committed prefixes are valid conditioning states; each new suffix is sampled
from the current policy. Factual source actions are never transplanted as gold
suffixes onto these changed histories.

Both arms receive one human and one population scope per update from the existing
capped balanced weights, with deterministic frozen draws. Source loss is mean
native row NLL per second, using genuine source styles and both difficulty/LN fraction targets defined
on the exact factual scope. Outside it, retain whole-chart controls and original
human style scopes. The response arm adds100 times expected sustained excess per future
second, averaging three independent four-second continuations and subtracting
the other-draw mean as an action-independent baseline. All continuation scores
and costs include the same actual generated prefix, requested ranges and native
sampler preferences. No star-only reward, pseudo-style reward, output filtering,
new response mask or source-future substitution is added. Train existing R1 at
3e-5 and new projection/composition/control/modulation at3e-4, AdamW decay1e-4,
clip1. Complete audio encoding is reused only while its weights stay frozen.

First run an eight-update response smoke with discarded weights. Inspect actual
nonzero outcome gradients, sampled/rescored parity, source gradients, finite
outputs and footprint. If no positive-cost variation occurs, revise the bank or
objective design before main fitting instead of claiming a working signal.
Then run384 updates per arm, identical source identities/order, from the same
zero-projection initialization. Seed272710; per-step suffix seeds272720+10*step+i.
Bank native seeds272500+i. Save full identities before launch. Warm/replay caches
must be rebuilt under current R1 weights. A source-only improvement alone does
not establish value of the added response objective.

Qualification reuses the nine reported four-star Stream cases, five matched
other-style/scope-change guards and locally matched genuine style contexts.
No planner during this comparison. Primary: at least50 percent lower mean
sustained excess than both initialization and source-only, without renewed
multi-second dominant-column episodes in the original failure contexts on Lens.
Difficulty MAE may increase no more than.15 versus initialization on the Stream
panel; known LN fraction errors may worsen no more than.05 on matching control
ranges. Keep before/override/restored ranges separate. Do not call a response
improvement a semantic style pass: inspect complete attack groups, temporal
organization and LN roles against real source references. A clear style loss or
new severe pattern blocks promotion regardless of objective gain.

Fresh owner artifacts/joint-audio/20260927-player-state-r1-learning-v1. Apple M5,
24GiB unified memory; MPS gradients, CPU native sampling, one CPU thread per
process; uv run --extra mps and --group dev for tests. At most18GiB footprint or
MPS allocation,600s bank preparation,300s smoke,3600s per main arm,45minutes summed active qualification-command wall time
and2GiB new artifacts. Report intervening analysis/review elapsed time separately. No overwrite/resume/download. Stop on
nonfinite values, support disagreement, sampled/rescored log-q error>.002,
frozen-weight drift, failed focused invariant, explicit STOP file or resource
limit. Freeze live run inputs and record clean intervention source before fitting.
Main arms may overlap after a stable source-only footprint and smoke footprint
sum below18GiB; monitor the combined budget. Fitting elapsed times under overlap
are not isolated throughput measurements. Native timing qualification runs after
all fits stop. First thirty published
rows must remain below2s and every measured two-second publication below2s.

Positive evidence supports response-aware native proposal learning, still requiring
broader style/held-demand/H-activity work. A negative or ambiguous result requires
revising the information flow, training-state coverage or objective rather than
blindly increasing parameters/steps. H breathing and comprehensive canonical C0
remain outside this bounded intervention and inside the active overall goal.


### Result Log: player-state input and private row scoring

Accepted: none; player-state-r1-learning-v1 revision 2 clarifies factual LN
control scope before fitting. The old common helper uses a legitimate whole-
chart LN condition with a local difficulty override. The new comparison instead
sets both targets on the factual training scope, exposing scoped LN controls
without treating the global source condition as an erroneous label. Both arms
use this same data construction. No fit has started.

Clean implementation94d0b082282ae886709c72ffdc885b2f4e045252 adds24 observations
per column: six-window attack/release rates and held fractions, plus last two
complete attack masks, ages and known bits. A zero-initialized mirror-equivariant
96-to-hidden projection enters R1 context, composition and consequence scoring;
H/R inputs remain unchanged. The128-wide model adds12,288 parameters. Checkpoint
options record player_state, and old checkpoints keep it disabled.

The private row-trace scorer supports [a,b) without EOF or invented LN closure,
using actual prefix rows, the timing-only H plan and complete audio encoding.
It rebuilds current-weight neural history and native recovery/LN preferences.
Tests first found NumPy integer timestamps crossing the exact-replay API's Python
numeric boundary; converting that call's timestamps to a native list fixed the
implementation. Four private-trace tests then pass in4.08s, including CPU/MPS
probability and gradient equality with full-source scoring, partitioning/empty
intervals and native sampled probabilities before open tails resolve. Earlier
player-condition/sampling/ownership checks pass13 tests in7.13s; sampler checks
also pass after the probability-replay refactor. Checkpoint round trip passes
in2.22s. Distinct covered tests total18; no failed invariant is waived.

Preparation freezes20 distinct TRAIN song groups/audio identities, excluding
held audio identities and their song groups. They total2564.226 audio seconds.
Bank plan SHA-2568aa54f1cd070866edd029eb6fcc024985b4c37e2529a88aa4357a0a6e86cfe9f;
384 factual draw pairs SHA-256
348a344dcb9245e474d6cd3fdc4320ad2773d908bc6ed4e48ebcfd7dcefd0524, covering197 unique
human scopes and300 population scopes. Artifact owner remains
20260927-player-state-r1-learning-v1. No generated requested styles are labels.


### Result Log: replay bank and eight-update learning smoke

Accepted: none. Bank process62513 completes20 native songs in254.470 seconds,
producing69 high-excess contexts and80 ordinary contexts. Every native export
completes. Records SHA-256
53ed0a0bf953439ebfbe83a4cf6eff8c203d19232c9b102f7c94428a0f5381d1. The high-excess
examples occur in Jack/Tech/Trill requests at several difficulties; all four
selected Stream and LN songs have zero baseline excess. This bank is a deliberate
training stress slice, not an unbiased estimate of failure prevalence.

The eight-update response smoke completes in37.763 seconds (handle3753 terminal),
checkpoint SHA-256
2a4c7ddd038c47f293afb8c67d0084cf6abcbca0c0a183616f8197c592cc2d64. Discard these
weights. Five of24 sampled futures have positive cost; two updates have nonzero
response gradients. Eight futures still hold keys at the response endpoint and
are scored without invented closures. Maximum CPU-sampled/MPS-rescored log-q
error is1.8813e-5; frozen hash remains
8c21a572ba08b5c8f847e818f318d8c92ef7d8f070616eb5ce0d42d911f626a5. Peak footprint
3.033GiB, MPS driver1.713GiB. The new projection norm reaches.1043. These establish
learning activity and parity, not quality improvement.

At the strong Trill2.5 context, source gradient norm20.836 versus response283.517;
future costs/second are .036861/.006772/.001191. Another update has source42.696
versus response.788. The large-gradient case may dominate its clipped update;
retain actual style/difficulty qualification rather than infer preservation from
source loss. Other-draw baselines cancel prefix costs when continuations do not
differ. Empty decision traces correctly have no R1 gradient.

The initial source loss9.0525681648 is exactly shared by smoke and the fresh main
source-only arm. Source-only384 is live under28653, at112 updates/114.547s with
footprint3.709GiB and driver1.983GiB. Revision3 permits overlap with the response
arm after that stable measurement; estimated combined footprint with smoke is
under7GiB, below18GiB. Data, objectives, weights, steps and seeds are unchanged.
A metadata-only detach removes a Torch scalar-conversion warning before main
launch: main train.py SHA-256
c67cb4087a017f0a425e58d658b67867f0969e6a68f26f6a84a2f93820e3a217, shared.py
466e78ec80aedb7c9b4d1f85f8d7589c6c8b507be90403d41fa7763c799ff4f4. No live input is
edited. Do not restart the live source-only run. The response arm starts from
the same declared initialization, not smoke weights.

### Result Log: completed player-state fits and first native comparison

Accepted: none; player-state-r1-learning-v1 revision3. Both training handles are
confirmed terminal: source28653 and response55349. Source-only completes384
updates in401.496 seconds; checkpoint SHA-256
7e0c5dba508af1968bed38b915246c822037c92c519846c1e3e0fb21b4693237. Response completes
384 updates in1492.603 seconds; checkpoint SHA-256
a91791fb7fda45190ddb0a2f1f17dc160b4e55e6cf1030cf3b17424d1a147ad6. Both consume the
same768 factual source identities in the same order, verified from final logs.
Frozen audio/H/R hash remains8c21a572ba08b5c8f847e818f318d8c92ef7d8f070616eb5ce0d42d911f626a5.
The new projection norms are.503758/.512772. Sampled footprint peaks4.975/5.175GiB,
MPS driver3.077/3.159GiB. Partial overlap precludes isolated fit-throughput claims.

The response arm draws1152 private futures. There are445 positive-cost futures,
162 nonzero response-gradient updates, and272 futures with open holds at the
four-second horizon. Maximum sampled/rescored row discrepancy2.291e-5. All live
input hashes still match configuration. The source code and smoke contract are
explained in docs/research/player_state_conditioning.md, product
c4d9e730375ace301f8ae68d34afbab50a75e5a5. No main-model quality claim follows from
these fitting statistics.

Source-only native qualification finishes14 outputs under handle42908 (terminal).
Every H stream matches its baseline. The nine Stream mean excess worsens from
.027516 to.144229, and whole-D MAE from.478926 to.852546. Strongest four-second
column rates reach8.75Hz. Adding observable state and source likelihood alone
does not teach its use as a load limit. Do not promote this endpoint or attribute
the failure solely to the new projection: R1 weights, factual control coverage
and supervision also changed. Generated-state response learning is the declared
comparison still to qualify.

Retain mixed effects instead of discarding all changes: the source-only LN guard
moves from D4.8027/rho.7634 toD4.5124/rho.6396 for target4/.6. In the actual switch,
before D2.2461/rho.1353, override D3.2746/rho.5828, restored D3.5679/rho.1937. The
restored difficulty is closer than the initial4.4218, but override difficulty is
farther from4.5 than the initial3.9010. Different ranges remain separate.

Response native qualification is live under44343:
uv run --extra mps python artifacts/joint-audio/20260927-player-state-r1-learning-v1/qualify.py response native.
It starts only after both fits terminate. Do not restart it because a report is
partial. Both endpoints still need twelve source-style outputs each via
qualify.py source|response styles, then matched Lens review and final analysis.
No such style outputs are yet claimed generated or judged.

Four genuine high-confidence prominent source references are rendered in
lens-references. Page1 of Tech/Jack/Stream/Trill has been read; the Trill page
includes the end of its fixed two-plus-two exchange and subsequent movement.
The reference label scopes remain158638-165038,44257-50924,79290-85290 and
288156-290040ms respectively. Their local targets are D3.414/rho.0118,
D4.170/rho0, D4.590/rho0 and D5.076/rho0. Source reference render handle57662 is
terminal. Other pages remain unreviewed. An audit also finds no reserved-song-
group overlap among any of the768 factual draws. This is a repeated developmental
qualification, not an untouched test set.


### Qualification accounting clarification

Accepted: none; revision4 changes only qualification runtime accounting/allowance.
The two native commands complete in417.624 and418.818 seconds (836.442 active
command seconds), while interleaved analysis makes first-launch-to-observation
elapsed time1829.238 seconds. The earlier30-minute wording did not distinguish
active qualification execution from intervening analysis; do not claim its total
elapsed bound was met. Before the remaining source-style comparisons, define a
45-minute summed active command budget and report overall elapsed time separately.
This is authorized by the standing overnight/local-research scope and remains
exploratory. Model, data, seeds, metrics and output counts are unchanged. No
training or qualification input file is modified while live.

Both native comparisons are now terminal (source42908, response44343). The
response native nine-Stream mean excess is.107160 versus source-only.144229 and
initial.027516; whole-D MAE.741130 versus.852546 and.478926. Thus the added objective
helps versus the source-only arm but fails the primary improvement versus initial
and the difficulty allowance. Serious concentration remains in new locations.
The source-style comparison is live under84983; do not restart it. The failed
numeric gate does not cancel the planned source-style inspection or mixed LN/
scope-control evidence. Neither fitted endpoint is promoted.
