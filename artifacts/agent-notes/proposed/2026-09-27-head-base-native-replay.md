# Agent Note: Native replay with an unfitted H base and fitted memory policy

Note ID: 2026-09-27-head-base-native-replay
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: fc641aa740a7528decbb0b5b522aa91039d06ff1
Scope: One H additive-path substitution in complete native rollouts, with reusable scoped regression evaluation
Related: 2026-09-27-head-control-hazard-probe, 2026-09-27-audio-memory-joint-fit, 2026-09-27-playability-regression-evaluation

### Experiment Card: head-base-native-replay-v1

Revision: 1
Accepted revision: none
Authority: exploratory execution under the user's ongoing research/experiment/
local-commit request; no implicit acceptance, adoption or remote publication.

The fixed-prefix probe identifies an H base contribution to faster next events:
replacing memory's base with the unfitted base increases restricted wait at least
20percent in8/12 generated contexts, median25percent. Restoring only the residual
does not recover the unfitted distribution. This does not yet explain complete
rollout counts because every new H changes subsequent history.

Question: does this additive path account for substantial sustained-pressure and
pacing regression under actual renewed generation? The native intervention uses
memory checkpoint7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81
for its fitted H residual, H memory, R/R1, row memory and complete-audio context.
Only H's additive base is replaced with the output of core2500
0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8's own full-audio,
head-control and head-base layers at the same query clock and control vector.
No learned feature vector is transferred between encoders, no R1 content enters H,
and no parameter is updated. This is an explicit two-model diagnostic, not a
new checkpoint or proposed serving architecture.

The analogue is additive component substitution in the already implemented
bounded H law. It preserves the fitted policy's history feedback under a different
input base. Alternative explanations include residual dependence on generated
history, R1 response choices and common data/objective bias; unchanged complete
quality after substitution would limit the base-drift explanation.

The fixed14 cases are the existing9 Stream seeds, four ordinary development
conditions and the live control switch. Cases, audio hashes, controls, seed and
recovery60/50/40 are copied without alteration from the original qualification.
Plan SHA8c76ae534b46709ac89b81b1b98f6560b27149fa5c0282c4da887a220fb7ae83.
Paired native memory/unfitted outputs already exist. No reserved validation,
style guard or source-H case is promoted into a claimed native pass.

Primary diagnostic: Stream mean exact multiscale pressure J, memory baseline
.3823017792s over9 cases; call reduction substantial at<=50percent of this value.
Also require median paired H-count ratio<=.90 to support timing reduction as a
major contributor. These are diagnostic thresholds, not new adoption gates.
Report all9 values, whole-star MAE, maximum and durations of pressure episodes,
occupation/recovery/contrasts and conditional audio correspondence independently.
Actual first30 and2s service bounds remain2s; publication replay uses2s lookahead.
No below20ms attacks, incomplete output or mutated committed prefix is allowed.
The original stricter unfitted-pressure and star-MAE guards remain visible; a
diagnostic improvement cannot silently relax them or certify playable quality.

Switch scopes before[0,64000), override[64000,96000), restored[96000,357797)
remain separate with carried state, requested D3/4.5/3 and rho.2/.6/.2. Retain
the original difficulty error<=1 and LNfraction error<=.10 guards. Developmental
scope inspection uses the same previously viewed source passages, then all-scale
pressure witnesses as warranted. Never infer gain solely from amount or H count.

Procedure: use the existing qualifier's generate/inspect functions and native
MemorySession; a scoped wrapper replaces the H base output only. Include the
additional complete-audio base encoding in publication-start and audio accounting.
Record both model hashes and wrapper/source hashes in every evaluation identity.
Check additive formula equality on actual native queries, reparse exported osu
rows, then run the same temporal/publication/pressure evaluation. This probe must
not overwrite or edit the frozen384-fit qualification outputs.

Command `uv run --extra mps python artifacts/joint-audio/20260927-head-base-native-replay-v1/run.py`.
Fresh output run-v1, CPU one thread, Apple M5/24GiB; no accelerator fit concurrently.
Budget1800s,300s per case,8GiB process peak; one attempt, no automatic retry or
overwrite. Stop on owner STOP, input drift, nonfinite hazard, export/ownership
failure or resource limit. Reuse exact case seeds; helper checks seed274800.
Unrelated AGENTS.md and audio_architecture_walkthrough_zh.md edits remain excluded.

Interpretation: substantial pressure/H reductions with retained relief/control
guards support investigating base/residual training coupling. Unchanged/worse
native pressure despite local waiting effects rejects local component repair as
sufficient. Improvements in J with worse held occupation, rhythm or switch
response are a tradeoff requiring separate diagnosis, not an overall gain.
No default model/interface changes and no next fit are authorized by a pass alone;
the user's broader research goal remains the execution authority.


## Execution handoff

Wrapper SHA345b01fd92842305fc526caaa909b9ecb59abc1604209d7d80dfb6575ca7df5c; syntax compilation passes.
The original qualifier is pinned tobce8addba20e903581d5fae84f3a2d773494717abc2dda1fc33db44fc39ea8af.
All required library code is committed atfc641aa740a7528decbb0b5b522aa91039d06ff1;
the wrapper is experiment-owned instrumentation. Native queries1/50/500 compare
the substituted formula to captured base/residual factors. The extra full-audio
encoding is included in session start and audio timing. Start the bounded run;
no further fitting or model promotion.


## Result Log: complete native substitution and local inspection

Run51637 is terminal/complete:14 verified exports in307.48959s, peakfootprint
.87753GiB. Numerical analysis48957, Lens renderer29302 and fixed-facts/review
writer60114 are terminal. No fitting or generation remains live. Result SHA
14051839450611e4cdeb039bb2a87e384fb49e7a3832700cbfd84f374c6dd3c7;
comparison SHA9cf604ed0b0fe03fcf17f7f44a90e8b9c57bf7734ee05a5903326c54e6b9d796;
visual-review SHA70dca47c44564c79a3a9435b2aa0bdf4c5093d52adf715540624790a6050969f.
All checkpoints/seeds/control schedules/recovery and input hashes follow Card1.
The run is exploratory with acceptance none, no promotion.

Nine-case mean/worst J falls from memory.382302/1.725083 to0/0. Median paired H
ratio is.318318, meeting both primary diagnostic criteria. StarMAE1.162118 versus
memory1.334522 and unfitted1.152140 passes the original+.15 aggregate guard.
This alone would look like a repair, but independent guards reject promotion:
first30 reaches2.267115s, with Zenithfall0/2 and switch over2s. Both complete-audio
encodings are included. Slowest2sservice.548252, no actual deadline misses after
operational startup, no below20ms attacks and no changed published prefix.

Switch D becomes1.808978/2.740275/2.228661 versus3/4.5/3. Restored difficulty
improves but before/override fail ±1. Every LNfraction remains within.10. Separate
ranges are essential: no pooled cancellation. Three Zenithfall wholeD values
are2.405078/2.346692/2.194507 against4. H0 falls5514 to995, heads5968 to1258;
anyheld rises.057 to.387 with LNamount unspecified. Full endogenous history
amplifies the local component sensitivity; this is not evidence for simply
choosing a global interpolation coefficient or keeping two encoders in serving.

All9 prepared Lens pages are actually read:2 Classic on[88589,93589),7 Blizzard
on[40342,56342), harness22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60. Classic interleaves
TAP/chords and held voices across both pages instead of the prior memory's
TAP-to-LN blocks. This is a limited local type-organization improvement. It has
18H/28heads/10LN, meanheld.7264, anyheld.6414 versus memory62H/76heads,
anyheld.4770 and source29H/47heads,anyheld.4212. Recovery and workload cannot be
inferred solely from the visual mix.

Blizzard remains predominantly chained LNs, with occasional TAP/chord accents
and longer paired holds near48s. It has87H/99heads/86LN, fraction.8687,
meanheld1.18531,anyheld.922 and zero all-column250ms recovery. Memory has125H/
155heads/122LN,anyheld.9523; source65H/82heads/39LN,anyheld.6931. No complete
breathing/texture repair follows. Alternate arrangements remain allowed; no
new human labels, audio listening or player test. Other passages/style labels
are unreviewed.

Evaluation: REFINE. H-base output is a demonstrated contributor to this dense
native regime, but restoring it overshoots several requests and fails startup.
The corrected EVAL workflow rejects pressure-only/mean-star-only success while
retaining local gains and exact tradeoffs. Full conclusion and identities are
curated in docs/research/audio_memory_joint_fit.md at14ee1534fbffcd6e809482f9ce748383358cb5a1.

Next research decision: examine how training anchors the audio/control base
and history residual jointly, including the annotation-centered source mixture,
then define a bounded repair evaluated on renewed native histories. A logit
bound is not a difficulty or breathing invariant. Do not launch an alpha sweep,
blindly freeze the full audio encoder, move R1 count/lane ownership into H, or
repeat an unchanged384 fit. Existing global rate feedback and player-state
studies must be read before proposing analogous controls. A secondary live
coupling question is scoped LNfeedback versus local TAP/LN texture; current
holding evidence does not alone identify that controller as the cause. The
ultimate playable-system goal remains active and no default checkpoint changes.
