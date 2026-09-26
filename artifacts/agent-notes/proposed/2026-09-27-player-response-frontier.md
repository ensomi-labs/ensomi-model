# Agent Note: Time-horizon player responses for sustained load and breathing

Note ID: 2026-09-27-player-response-frontier
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 7ea3b956ebccdc4d4238bc52e5cb89762402f054
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
