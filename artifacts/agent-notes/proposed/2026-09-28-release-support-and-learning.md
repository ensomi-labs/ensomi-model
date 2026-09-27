# Agent Note: Release support and joint R/R1 learning

Note ID: 2026-09-28-release-support-and-learning
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 265358048481fcd69c75fd4b0de5302853f616a5
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
