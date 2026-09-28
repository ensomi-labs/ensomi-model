# Agent Note: Release support and joint R/R1 learning

Note ID: 2026-09-28-release-support-and-learning
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 784ac6fc0e03aa6600c2c805c778ab788d6b126d
Scope: Ranked TRAIN release support, factual interval admission, R/R1 likelihood and native playability
Related: 2026-09-28-scoped-ln-allocation-learning, 2026-09-28-broader-full-row-learning, 2026-09-27-ln-continuation-preference-diagnosis

## Question and current evidence

The preceding study found a genuine 4.485903-star LN-jack with six 38–48-ms
holds excluded by the current 50-ms head-to-release floor. The broader R1
candidate still has native amount/organization failures. The goal is to recover
appropriate real expression and learn its actual distribution, not simply lower
floors or reward shorter LNs.

Code inspection confirms that ControlledSession applies empirical release-age
preferences in the complete-row law, after the R event time is chosen. Unlike
the earlier typed generator, it does not apply release_clock_cost to its R
hazard. A row preference cannot postpone an already selected pure-release event,
especially when only one release identity is eligible. Relaxed support therefore
needs the R timing law and its no-event supervision to be considered alongside
R1, not assumed safe because row preferences remain.

Actual interval admission also needs inspection. Empty-row intervals can still
contain an ongoing LN and supervise release survival. A row-only recipe's
empty-window rejection must not be copied blindly into joint R/R1 learning.
Scope-control program coverage and own-policy outcome learning remain separate
open requirements; the preceding source-only adapter is not adopted.

## Experiment Card: release-support-audit-v1

Revision: 1
Accepted revision: none
Execution authority: continuing user goal authorizes local research, experiments
and suitable commits. Exploratory, with no acceptance or remote publication.

Baseline source 3591df29b824cf4340c5425da81c5338cd7c49aa. Read only the byte-pinned
prepared ranked corpus manifest with SHA
4cea2672387b6293a4da0846be479d8bd9c857e55535dc8143bd11d65b06d2c4,
using its TRAIN charts with metadata stars 2–6. This prepared set already has
an initial 37/25/21 support selection and is not an unfiltered corpus. Do not
inspect TEST or fit a new calibration from validation.

Measure actual HH, first release-to-next-head RH and LN duration HR gaps, with
transition types and independent per-chart/star-band counts. Compare fixed
support profiles 60/50/50, 60/40/35 and 60/25/21; the middle profile describes
coverage, not a separate model-training branch. These are implementation
supports, not physiological thresholds. Preserve at least the existing HH
restriction during this release-focused comparison so native H can remain paired.

Count affected eight-second target intervals and genuine no-row intervals with
entering holds. Label interval counts based only on relations as predictions;
check actual collate_interval acceptance on a deterministic bounded selection
of recovered, still-excluded, ordinary and empty-holding intervals. Record all
exceptions rather than treating a gap census as full admission equivalence.

The decision is whether and how to change release support for a joint R/R1 fit.
No improvement claim follows from greater support alone. Source reference
articulation, full native control/pressure/organization and real-time publication
must be checked in any subsequent learning experiment.

CPU one thread, explicit mps environment, at most 180 seconds for the gap scan
and 600 seconds for bounded actual interval checks, 4-GiB process footprint.
Fresh owner artifacts/joint-audio/20260928-release-support-learning-v1; no
overwrite or model mutation. Stop on STOP, input/hash drift or resource bounds.
Pin the script hash and exact command before execution. Raw source clocks and
objects remain unchanged. Descriptive metadata-band results are not recomputed
star or source-style judgments.

## Next condition

After this audit, freeze one support/learning comparison with actual sample
coverage, matching source-time likelihood factors and native regressions. A
profile change cannot by itself establish playability, and a NLL change cannot
substitute for the full generated task. Keep the entire 2–6-star, expressive,
scoped-control, real-time goal active.

## Result: completed support audit

The byte-pinned prepared TRAIN cohort has 6,923 charts and 131,611 nonempty
eight-second windows. Profiles 60/50/50, 60/40/35 and 60/25/21 affect respectively
457, 117 and 27 charts, with 2,313, 425 and 135 relationship-predicted target
windows. The last profile retains HH=60 and has no remaining RH/HR violations
in this already selected cohort. It is not a physiological calibration.

Actual collator checks on 64 predicted recovered windows accept 0 under the
strict profile and 64 under 60/25/21. All 64 ordinary windows remain accepted;
all 64 HH-excluded windows remain rejected. Sixteen no-row windows with
continuing holds are accepted and carry release survival. Audit summary SHA
6c9a0dad11d626421225e2c268e2c4957ef21c6b49d18494631fef1ad89c76cb;
admission result 934125adddafc89025e0d84355e364c32edaf538c350e6e209d79a6cc3fd6838.
The diagnostic Card above is completed history; the learning Card below is the
single active experiment. Neither is accepted or adopted.

## Experiment Card: release-support-joint-learning-v1

Revision: 2
Accepted revision: none
Execution authority: the continuing user goal explicitly authorizes research,
training, suitable product/Note commits and available Mac resources. Execution
is exploratory; this authority does not imply Note acceptance or promotion.

### Hypothesis and intervention

Restoring observed release articulation alone cannot repair a learned timing
law. R/R1 likelihood on broader factual intervals should learn both the
release/no-release process and complete rows. A matched full-audio/H/R/R1 arm
tests whether frozen upstream music/timing parameters constrain improvement
in requested difficulty, LN organization and breathing.

Closest analogues remain full marked-event likelihood and the established
full-song coarse/local-fine encoder. This tests ownership of joint learning,
not a novel likelihood factorization or a new player-response definition.
Current frontier's missing LN response is not claimed repaired by this study.

Runtime source is 265358048481fcd69c75fd4b0de5302853f616a5, a documentation-only
descendant of 3591df29b824cf4340c5425da81c5338cd7c49aa. Preserve unrelated
AGENTS.md and untracked architecture prose. Require executable source,
tests, dependencies and lockfile to match this commit. Parent checkpoint is
the unqualified scoped-progress candidate SHA
8898c51714474f82171b570cd2c8867bb07e911a59dfbae91b2642687de6a4ca.
Both trained arms use HH/RH/HR=60/25/21 and LN feedback off, retaining the
existing empirical row preference and scope-allocation module.

Arm rr1 trains complete R1, scope allocation, skeleton_temporal, release_clock
and release_control; audio and H parameters remain fixed. Arm joint trains all
model parameters, with shared fine/coarse audio and H at lower learning rate.
The model architecture and row/count ownership are unchanged. A profile-only
parent isolates support changes. The parent with its original 60/50/50 support
remains the quantity/control reference; its existing twenty outputs are reused
by immutable identity and eight D2/D6 cases are added.

### Data, objective and learning budget

Freeze 1,024 new factual eight-second draws, seed 280033, from the pinned
prepared corpus, excluding the five existing native-panel audios. No TEST.
Population draws: metadata-star/LN strata, then uniform group/chart and
interval, with 10% BOS emphasis. Human branch: original annotated windows.
Draw branch with 75/25 probability and keep it through rejection retries.
Require recomputed source stars 2–6 and actual restored-support admission.
Keep genuine empty intervals. Record all acceptance, empty, transition and
condition exposure; no recovered-relation oversampling.

Use the existing true controls: 50% whole-song SR/LN; otherwise 16/32/64-second
source-scoped proxy/LN; human style scopes are not expanded. D/LN missingness
and per-style missingness remain independent as in the broader factual sampler.
These local proxies are not renamed as whole-song star labels.

Both arms optimize H survival/event + R survival/event + deployed complete-row
NLL; frozen factors have zero parameter gradient in rr1. Population weights
use IntervalExample.weight_per_second, human weights use 1000/actual interval
milliseconds. BOS/annotation/strata/rejection weighting means this is an
explicit research measure, not an unbiased all-corpus likelihood claim.
Factual prefixes, controls and targets remain in one real chart world.

512 optimizer updates, batch two, fresh AdamW, weight decay .0001, clip norm
one. R1 composition/row_control/preview/layout/scope paths use .0001; remaining
R/R1 paths .00003; full arm audio/H paths .00001. Same draws and initialization.
Full coarse encoding is differentiable and recomputed at current weights;
fine queries use halo-complete source crops. No stale learned audio cache.
Both models use train mode during fitting and eval mode during validation.

Execute serial 16-update workers with optimizer/state checkpoint continuity.
The first worker is the resource/learning pilot; stop on nonfinite loss or
gradients, source/hash drift, explicit STOP, 16-GiB process footprint, 600
seconds per worker or 7,200 total fitting seconds. Preparation max 600 seconds.
Do not restart a live process. Freeze scripts and plan hashes before execution.
Fresh fit-v1/supervisor-v1 outputs in the existing release-support owner;
failed outputs remain immutable. The same 22 factual validation windows are
diagnostic only; evaluate before/after and do not select by validation NLL.

### Native evaluation and decision

Retain the existing twenty complete native cases, seeds, source contexts and
live override. Add Classic and Zenithfall at D2/D6 with two paired seeds each,
LN/style unknown. Three restored-support arms use the same 28-case plan.
Generate eight added cases for the strict parent; reuse its twenty existing
cases. Pin both plans, response calibration and parent identities.

Primary comparison is nineteen whole-D4 cases' mean absolute recomputed-star
error, parent .58448888462181. A useful joint-learning result must improve by
at least .10 against both the profile-only parent and rr1 comparator. If rr1
alone improves by .10 against profile-only, retain that bounded result without
claiming upstream learning value. Six rich-LN whole-request fraction MAE,
parent .09911720339364642, may worsen by at most .025. No newly failed existing
numeric qualification case; no new completed/legality/less-than-20-ms attack
failure. Report every control scope separately. Existing failures are not
declared solved by average improvement.

For each scope, flag attack excess increases above max(.005 seconds, 10% of
its comparator), and inspect head recurrence with held-finger and TAP/LN
articulation context. A flag needs analysis, not a universal shape ban.
Report counts/density, occupation, survival/span relations, interaction
witnesses, multiscale dynamics and audio correspondence to expose sparse or
LN-substitution wins. Added D2/D6 cases must retain ordered response and
per-case star error <=1 to count as range-control evidence.

Inspect frozen source contexts for Classic, STYX and Blizzard, both seeds and
both trained arms with beatmap-lens, plus new worst pressure/recurrence and
control-failure witnesses. Unreviewed or ambiguous organization cannot be
marked semantically passed. Corpus counterexamples remain reference evidence,
not scalar BAD labels. The existing 2-second native qualifier checks startup
and service; any candidate retained for promotion still needs the separate
30-row/8-second stream benchmark and client contract qualification.

Each native arm has max 1,500 seconds, 180 seconds per case, 4-GiB footprint,
CPU one thread; runs are serial. Record all output identities and separate
mechanism findings from adoption. A NLL-only gain, density collapse, degraded
scoped controls or unresolved timing failure does not justify a success claim.

### Frozen execution identities and preparation receipt

Learning plan SHA aa0c50f563b58b5d555677207e927bcfdc5878b2dc34e96df3a3c75455816e08
pins all five worker/likelihood/validation/native scripts and their exact
commands before fitting. The source plan SHA is
1b59345d6760294685e9475dc6431797f6a6094417ae24f9c89eb04425a15871.
Preparation completes in 411.99 seconds with 1,024 accepted windows, 736 TRAIN
charts, 257 human draws, 26 no-row intervals and 40 windows rejected by the
old strict profile but accepted by restored support. One proposed window is
rejected by the restored profile. These are actual exposure counts, not a
coverage claim for all possible source relations.

Native 28-case plan SHA
f67b6dbadd0ee4e2a67575a569bd69f56fc3387cfb551f629546eae5ca97ea9c;
strict parent's eight-case added range plan
2b581db7ec3f47f8d63eee75f9c1bc4108cff30e1f72a929c86e9b0a5ef86303.
The twenty reused parent cases are byte-verified at
daecd5bfb7613ea48bda480b5da89d4a697a6f8350cf73cd16be3d2dc66049e6.
No training or quality conclusion is attached to this preparation receipt.

### Revision 2: correct the padded-query scoring wrapper

Initial validation terminated before any optimizer update: the new wrapper
passed padded row queries into replay_row_scores, which correctly requires
exactly the actual rows in the interval. Correct the wrapper to slice by
len(batch.row_index), matching the established factual scorer. Preserve the
failed validation-initial-v1/failure.json and original plan.json.

Use plan-v2.json and fresh validation-initial-v2 / validation-terminal-v2.
Fit-v1, supervisor-v1 and native outputs are still unused. All scientific
inputs, arms, budgets, losses, criteria and seeds remain unchanged; the new
plan pins corrected script bytes before rerunning. This was a study wrapper
error, not evidence against the model or a checkpoint result.

## Interim execution evidence

Revision-2 plan SHA
210cbf5320e370e8d6c4679322555bab635c2ab215eb68b8d7e8636adcbfc6a4.
The corrected initial validation reproduces strict-parent row NLL
1.571176095629661 versus prior 1.571176097399668. Restored-support row NLL is
1.571162529652034; H/R/row per-second means are
32.0191259384 / 1.3190457032 / 11.4171410547. This checks the full-coarse,
halo-fine and scoped-progress factual path, not native quality.

The first 16-step worker completes in 34.83 seconds at 4,045,737,776-byte
peak footprint. Trainable counts are 3,274,110 for rr1 and 4,670,322 for joint.
The pilot observes nonzero gradients in the declared audio/H/R/R1 paths and
verifies all rr1-frozen tensors remain bitwise unchanged. Later segments are
running serially; a completed pilot is not a completed fit.

Actual 1,024-draw exposure: 85,246 heads, 22,837 LN starts and 22,761 releases;
661 song groups. Recomputed [2,3)/[3,4)/[4,5)/[5,6] window counts are
205/272/263/284; recovered windows 1/3/11/25. All 26 sampled no-row intervals
have no eligible R milliseconds, so they supply H silence rather than an
additional empty-held training slice. Other windows retain actual R survival.
Known-style exposure is 534.694 seconds and Stream-only-prominent exposure
15.372 seconds. These are repeated training seconds, not independent annotated
episodes or evidence of broad Stream-only conditional coverage.

Analysis plan revision 2 SHA
87c2ba7b307cf55e674d6aee338c1d6ca734e5dc243ce1bfc01e73d5c68fd9ed
pins the scope-separated comparison and frozen Lens contexts before native
results. Existing multiscale contrast quantiles and feature witnesses are
retained; greater variation is not assigned a positive label.

## Realtime integration requirement found during the fit

Read-only inspection of codex/stream-generation-benchmark at
4ec631ef71d1ca71e36e5c383d4997efffdebb05 confirms a loader/behavior distinction:
its ControlledAudioModel constructor does not support the newer player_state
or scope_allocation options, and benchmark.py constructs ControlledSession
without overriding its default LN feedback. Current study models contain the
scope module and are evaluated with LN feedback off. The old benchmark cannot
therefore be treated as a measurement of these exact endpoints by merely
swapping checkpoint filenames. Its worktree remains untouched.

A retained candidate needs the current probability implementation and explicit
sampling settings integrated with the benchmark's 30-row/eight-second,
watermark, private-state and control-update contracts. Do not silently ignore
unknown checkpoint options or add a second quantity controller during that
integration. Current two-second qualification is useful distinct evidence,
not a substitute for this compatibility and runtime work.

## Completed matched fit

Both arms complete 512 updates in 1,338.75 seconds. Maximum sampled worker
footprint is 6,091,968,880 bytes. Every rr1 segment verifies frozen tensors;
optimizer moments and verified parent checkpoint identities carry through
all 32 serial workers. Terminal checkpoints under fit-v1/496-512:

- rr1.pt: 6537341d698071ac0c7dfc41d46ffa6bac39ddfc42cb2e703c11fdb9e62a2583.
- joint.pt: 209aa9c29b97b9850ad50928418bc1830ef5dea552b853dfdebf5fb43872b69c.

Fit summary SHA
926d7949b2fb6c9960f9e2632fb1cb18cf4978ad27a65bbe67fffecebcb4f58d.
Across successive 128-update blocks, joint-minus-rr1 same-draw H NLL per second
averages -.0597, -.2380, -.3277 and -.4553. Row differences stay small and
mixed. These are online comparisons on changing examples, not a fixed-data
learning curve or evidence of native quality. Validation/native review remain
required; neither terminal is promoted.

## Initial validation and low-difficulty reachability evidence

Terminal validation row NLL is rr1 1.5696064050 / joint 1.5643044454; joint
H NLL per second is 31.53779420 versus 32.01912594 for frozen H. R means are
1.30952077 / 1.31357227. Validation receipt SHA
a6cda7b7003cf4e5d9ea1fecb41447599b787b4eac43a5b6b5a1fb114af7d7e5.
These improvements do not change native criteria.

The strict parent's added D2 cases return Classic 3.282442/3.289651 and
Zenithfall 3.664578/3.375850 stars. D6 cases remain ordered and within the
declared one-star tolerance. This exposes an existing low-control deficit.

A post-hoc diagnostic constructs one cyclic-column TAP per unchanged H,
with no LN, solely as an existence witness; it is not a proposed sampler.
Classic admits 2.113209/2.188426 stars, showing unused row-materialization
space. Zenithfall's witnesses are 3.527865/3.193356; this alone does not prove
infeasibility. Result SHA
6e68b33b24e7b1e0c8de66292262103b162f378d606128a50deab894a1a02816.

A stronger read-only bound follows from the implemented 20241007 algorithm:
each processed head contributes >=1 overall strain and >=2 selected individual
strain. Keeping one head per H, using these minima and the same decays,
400-ms section peaks and descending .9 weighting gives a whole-chart SR
lower bound for any materialization of those H. It ignores column recurrence
and LN additions, so is conservative and need not be attainable. Classic
bounds are 1.841016/1.900875; Zenithfall 2.928770/2.674991. Thus exact D2 is
already unavailable for the latter fixed H. These bounds do not prove violation
of the looser +/-1 qualification tolerance, nor establish musical/playability
quality. They must not be compared with unrelated scoped controls.

The exploratory implementation head_floor.py uses current official-variant
section accounting and checks the inequality on 256 legal randomized mixed
TAP/LN/chord charts and all eight strict-parent cases. Result SHA
ead4c8344c0e32b847d8f8e445c3d0250791cb28fe3bb8a9d2b68c43a9d6f831.
This post-hoc diagnostic was not a prespecified winning criterion. Preserve
the proof and develop a reusable evaluator after pinned native runs finish;
do not modify the running experiment's product source or sampling policy.

## Final result: neither trained endpoint qualifies

Evaluation outcome: REFINE. Both 512-update fits, final validation, all four
serial native arms and every rendering/analysis process are terminal. No model
is promoted, no note lifecycle transition is made, and the overall playable
2–6-star generation goal remains active. All training/native executions used
265358048481fcd69c75fd4b0de5302853f616a5; the later evaluator and report are
separate, committed post-hoc work.

Ninety-two new complete maps plus twenty byte-verified strict-parent outputs
give 112 comparison records. New native work takes 1,596.83 seconds. Every
case completes with export/reparse equality and no less-than-20-ms same-column
attack failure. Profile-only/RR1/joint startup maxima are .931/1.023/.897s;
service maxima are .410/.422/.345s. These are the two-second qualifier's
measurements, not the outstanding 30-row/eight-second client qualification.

| Arm | D4 star MAE, 19 whole-song cases | Rich-LN fraction MAE, six cases |
| --- | ---: | ---: |
| Strict parent | .584488885 | .099117203 |
| Profile only | .565434894 | .106349848 |
| R/R1 | .583464252 | .061129150 |
| Joint | .566762641 | .089384423 |

Both arms fail the primary .10 improvement criterion. R/R1 adds four numeric
qualification failures, joint three. The LN mean improves, but STYX seed 0
in R/R1 undershoots its request (.369538 versus .485281); joint seed 1 also
undershoots (.362486). Other references still overshoot. Do not explain all
LN failures as a single upward shift. The live .6 override reaches .795597
in R/R1 and .763736 in joint, versus .897163 for profile-only; it is improved
but unresolved. Its difficulty proxy is respectively 3.554666/4.196048,
against requested 4.5.

D2/D6 responses stay ordered, but only joint Classic seed 0 meets the declared
one-star tolerance at both ends. Joint D2 Zenithfall returns 4.460224/4.199297.
R/R1 and joint Classic D6 seed 1 fall to 4.657215/4.807386, new underestimation
failures. This defeats qualification even with improved factual NLL.

Final comparison SHA
8bd6920ab2557f7d923bc35c778615ea99ec1a2a5e5d34e967f324c5815d9857.
Native cases SHA, profile/RR1/joint respectively:
6ac009c49a065c492415d8dc1a679d25e3703133a56e9db1cac1191a64abdd22,
8d96488b4743a4bdbde5874870717b48523875242224580657dddaee5212abf1,
5ecb95f59e36159e178c22390190495046fdde15028897afd45018e0d052213c.

### Read organization, not only aggregate passes

Beatmap-lens review is complete for 76 pages: forty frozen reference/generated
pages, twenty-seven failure/context pages, six slower-anchor pages, and three
same-support profile-only comparator pages. Source Classic/STYX/Blizzard
recompute to 4.0000/4.0044/3.9277 stars. Zenithfall source is 5.8735 stars and
is only a same-audio organization reference, not a matched four-star chart.
There is no new human label, audio-listening or playtest claim.

The R/R1 Stream Zenithfall seed 271201 case regresses to a column-3 run of
21 heads (20 TAP and one LN press), 31789–34357ms, median/max HH 127/162ms,
with only two companion heads. Other columns are not all held, and the run
continues after the earlier column-1 hold releases. Same-support profile-only
has exactly the same H, distributes attacks around a different hold and has
whole-map maximum run seven. R/R1 learning is the controlled intervention;
it does not isolate R parameters, R1 parameters and induced histories.

Joint distributes that specific window, but another D4 Stream seed has only
maximum run five and still incurs .080000 seconds of attack excess. Its
entire inspected 235440–251441ms span is continuous cross-column TAP/chord
flow with no LN. The 16-second peak counts are 67/91/86/66 and a column-1
excess episode lasts 30.205s. Removing isolated long jack does not establish
recovery or music-responsive phrasing. R/R1 STYX seed 0 instead has eleven
TAPs at 7358–8923ms with up to three other held fingers and 25 continuing-
hold/head pairs while its whole-map attack cost is zero. This exposes the
attack-only observer's different blind spot; neither statistic is a universal
pattern prohibition.

Blizzard references show a persistent anchor with TAP accents and paired short
LNs. Both learned endpoints' inspected outputs remain dominated by changing
holds; amount improvement does not demonstrate recovery of that organization.
Classic has a locally all-TAP R/R1 case despite better whole-song fraction.
Such local purity can satisfy a whole-song request and is not an automatic BAD
label. A slower joint Classic thirteen-TAP anchor has a different cadence and
six companion heads; preserve ranked anchored-jack counterexamples.

Reading receipts, in the same four-part order:
aacc8367d062dad22c0985ac81b12dabab04eab4762190225fca7e82b56c9cc3,
b8b1cc4a3abce9b361414a6ceb156631358482115c1cbb8123e8b41228a4f91b,
e92d68858b8a518686b369f8dfb7e2ac9f17050739e28818813c75bdae2b4a5e,
3430edf6ab236526f5d2035a5c1b7f26f6f5748424a0f21162320821589212e5.

### Separate timing feasibility, protocol and allocation

The reusable H-only bound and eleven focused tests are committed at
dcc21bdc3a1f7c40d36d8302d26a706b30ed5bc2. The selected test command is
`uv run --extra mps --extra render --group dev pytest -q tests/research/gameplay_evaluation/test_head_difficulty.py`;
all eleven pass in .39 seconds. Tests cover first-object/first-chord accounting,
all 256 layouts for four H with three hold choices and two rates, randomized
legal mixed charts, quiet section advances and prefix completions. The bound
is diagnostic, not a new sampler mask or a scoped difficulty target.

Joint D2 Zenithfall seed 1 reduces H count 1899 to 1399 but the floor changes
2.674991 to 2.685472, with actual stars rising 3.375850 to 4.199297. Fewer total
H has not removed the necessary peak burden. Classic's cyclic same-H witness
still proves lower-star row choices exist. This does not show that the joint
NLL improvement came only from silence or assign causal percentages.

R/R1 and profile-only H match in 28/28 cases. Against the strict parent they
match in 27/28. The live override is announced at 63999ms and starts at 64000ms;
support-dependent retention keeps old H/no-H decisions to 64099ms versus
64059ms, moving one resampled H from 64260 to 64256ms. The remaining H match.
This is update_controls feasibility retention, not an action-embedding input
to the H network. The original post-hoc assertion of universal pairing failed
and is preserved; corrected panel-v2 records this exception explicitly.
Panel result SHA
cecdaa4d43747f1aee704b9ea5db49c153525168a8e8d687c2317ebe81aec06e.

### Recipe and architectural hypotheses now made concrete

The 6906 metadata 2–6 TRAIN charts after panel exclusion have whole-chart
LN fraction >=.5 in 4.94% under equal-chart weighting and 6.09% under equal-
group-then-chart weighting. Accepted population exposure is 33.90%, and among
217 windows with LN control hidden it remains 30.41%. Independent hiding
does not undo the balanced sampler. Learning rare conditions and choosing an
unknown-control prior require distinct decisions. This is measured exposure
reweighting, not an identified causal share of a particular failure or proof
that the desired default must equal the corpus prior. Prior audit SHA
288b163235f058b856cf223b430928a25402bb4248d79a3b03dd36f05c5fb5be.

The bounded H audio base is linear in F_A + W_c c, so its audio/control mixed
partial is zero. The nonlinear bounded history residual can still interact,
but its gate is zero at BOS and decays with time since H. This establishes a
particular expressivity limit, not full-model audio blindness or a proven cause
of the peak-pressure failure. Conditional audio queries, multiplicative
modulation or a nonlinear base are candidate primitives. The nearest repository
analogue is row_condition_interactions.md; no novelty or tested benefit claim.

The useful next direction changes the conditional path or learning/selection
target, rather than scaling this unchanged factual recipe. Preserve broad
corpus imitation while learning independently justified scope/gameplay future
consequences on actual generated histories. Continuation must distinguish LN
entry/release/coordination and retained holds from attack-only cost. Scope
allocation, musical organization and player response keep separate semantics.
A new intervention still needs its own bounded design; none is trained here.

## Durable result and publication state

Product report docs/research/release_support_joint_learning.md and five
inspected figures are committed at 784ac6fc0e03aa6600c2c805c778ab788d6b126d.
The formal native-pattern analysis retains the user's original views and
adds the result to its evidence/hypothesis mapping. Its preceding mathematical
extension is 265358048481fcd69c75fd4b0de5302853f616a5: information projection,
joint timing/row reweighting, LN origin/survival mixture, state equivalence,
distributional propagation and response/selection distinctions are analytical
claims under stated assumptions, not new human measurements.

All three touched documentation files pass local path/fragment checks and
math delimiter/brace checks; ten named result hashes were reverified. Final
owned diffs pass whitespace checks. User-owned AGENTS.md changes and the
untracked audio_architecture_walkthrough_zh.md remain untouched and uncommitted.
No remote push. The benchmark worktree remains unchanged; retained-candidate
integration is still required before any realtime product claim.
