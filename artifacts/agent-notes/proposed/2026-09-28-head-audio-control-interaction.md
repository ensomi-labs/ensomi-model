# Agent Note: Conditional audio evidence in the head-time base

Note ID: 2026-09-28-head-audio-control-interaction
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 0a74a198f73dec42753e0c92fb9c98ee298e4710
Scope: H audio/control interaction, matched H learning and complete native regression
Related: 2026-09-28-release-support-and-learning, 2026-09-27-head-control-hazard-probe, 2026-09-27-head-base-native-replay

## Mechanism and alternatives

The current bounded H base is affine in F_A + W_c c. Its audio/control mixed
partial is zero. The nonlinear residual can interact, but disappears at BOS
and decays with time since the last H. Thus the dominant base can shift hazard
log-odds with a request but cannot change which encoded audio features matter
for that request. This restriction is separate from capacity and data exposure.

The prior full joint fit reduced Zenithfall D2 seed-1 H count from 1899 to 1399
while its mandatory-H star floor stayed near 2.68 and actual stars increased.
The earlier H-base substitution also changed complete native density strongly
but overshot controls. These findings motivate conditional evidence selection;
they do not identify the affine restriction as the sole cause of either failure.

The selected primitive is feature-wise conditional scaling of the audio base:

$$
b_j(F,c)=w_j^\top F+\beta_j+w_j^\top W_c c
          +w_j^\top\!\left(F\odot\tanh(Vc)\right).
$$

V starts at zero. The old base is recovered exactly; the residual receives its
old additive condition and retains its existing bound/decay. There is no new
row input to H, no count/lane plan, and no player-response or pattern mask.
The ordinary R/R1 paths are unchanged. Controls keep their full per-field scope
encoding, including unknown semantics and future announced boundaries.

Closest analogue: [FiLM](https://arxiv.org/abs/1709.07871), which conditions
feature-wise affine transformations; the repository's row_condition_interactions.md
already applies a related mechanism inside R1. This is an adaptation to the H
audio base, not general-method novelty. A nonlinear base, a control-dependent
audio query and outcome-supervised planning remain alternatives. Do not combine
them in the first comparison or infer their failure from a negative result here.

## Experiment Card: head-audio-control-interaction-v1

Revision: 1
Accepted revision: none
Execution authority: the continuing user goal authorizes local implementation,
research experiments and commits. The Card remains proposed; no acceptance,
adoption or remote publication is implied.

### Baseline, intervention and learning

Product baseline 784ac6fc0e03aa6600c2c805c778ab788d6b126d. Parent checkpoint
artifacts/joint-audio/20260928-scoped-ln-allocation-v1/fit-v2/progress.pt,
SHA 8898c51714474f82171b570cd2c8867bb07e911a59dfbae91b2642687de6a4ca.
Both arms restore release support to 60/25/21, retain existing row preference
and disable LN feedback, matching the prior profile-only comparator.

The matched arms are continued additive H and H with zero-initialized conditional
audio scaling. Both train head_temporal, head_condition, timing, context_timing,
head_base and head_control; only the second also trains head_audio_modulation.
There are 513108 shared trainable H parameters and 75264 added parameters
(336 control coordinates to 224 audio features). Audio, R, R1 and scoped LN
allocation stay frozen and must remain bitwise unchanged at segment receipts.

Freezing audio is a deliberate attribution choice: shared-audio updates would
also change frozen R/R1 inputs. Both arms still read complete-song coarse audio
and full-halo fine features; the comparison cannot decide whether a jointly
adapted encoder would yield a larger final benefit.

Use the preceding 1024 factual eight-second draws exactly once: source-plan SHA
1b59345d6760294685e9475dc6431797f6a6094417ae24f9c89eb04425a15871 in
artifacts/joint-audio/20260928-release-support-learning-v1. It contains 736 TRAIN
charts, 661 groups, excludes the five native panel audios, and retains independent
missing controls and genuine annotations. Do not merge same-audio charts, attach
source suffixes to generated prefixes, or change recovery admission and sampling.
The balanced exposure/unknown-prior caveat remains common to both arms.

Minimize factual H event-plus-survival NLL with the frozen per-second weights,
512 updates, batch two, AdamW weight decay .0001, gradient cap one. Learning
rate .0001 for head_base/head_control/the new modulation; .00003 for the other
H modules. Matched draws, initialization and optimizer settings. R/row losses
may be recorded but produce no training gradient. Model mode is eval with only
trainable H modules in train mode so frozen audio does not acquire stochastic
training-only behavior.

Run serial 16-update workers carrying optimizer state. The first worker is
the bounded resource/learning pilot. Verify nonzero gradients in intended H
paths and exact frozen tensors. No intermediate checkpoint selection.
The same 22 validation windows provide before/after H/R/row diagnostics only.
Initial identity and trained H changes must agree with actual native queries.

### Complete generation decision

Use the exact preceding 28-case native plan, SHA
f67b6dbadd0ee4e2a67575a569bd69f56fc3387cfb551f629546eae5ca97ea9c.
Same seeds, audio, scoped controls, recovery and feedback-off settings. The
frozen profile-only cases SHA is
6ac009c49a065c492415d8dc1a679d25e3703133a56e9db1cac1191a64abdd22.
All comparisons are from BOS and retain each control range separately.

Primary question: does conditional audio scaling improve low-difficulty native
control beyond continued additive H? Four D2 cases have profile-only mean
absolute star error 1.38534246. A useful result requires at least .20 reduction
against both profile-only and the matched additive fit. The mean mandatory-H
floor excess max(floor-2,0), baseline .40094021, must decrease by at least .10
against profile-only to attribute part of the improvement to H feasibility.
The floor is a conservative whole-chart diagnostic, not a quality score or
a new rejection rule.

Guards: nineteen whole-D4 star MAE cannot worsen by more than .10 from profile
.56543489; four D6 MAE cannot worsen by more than .10; six rich-LN fraction MAE
cannot worsen by more than .025 from .10634985. Report every newly failed numeric
case and do not promote an endpoint with one. No incomplete output, invalid
export, changed committed prefix or less-than-20-ms same-column attack.

Flag any scoped attack-excess increase above max(.005 seconds, 10% of its
comparator). Keep LN interaction, occupied recovery, recurrence, multi-scale
contrast and audio correspondence separate. Check genuine ranked examples when
a flag's meaning is unclear. No credit for uniformly thinning maps, deleting LN,
losing Tech irregularity or replacing jack by sustained four-finger pressure.

Read the same fixed Classic/STYX/Blizzard contexts for both seeds and both
trained arms with beatmap-lens, plus both D2 audios' newly worst pressure and
largest changed contexts and all new recurrence/control regressions. Preserve
the earlier 21-head Stream window as a regression witness. Visually different
from a reference is not by itself BAD, and unreviewed cases remain unreviewed.

### Execution, bounds and interpretation

Fresh owner artifacts/joint-audio/20260928-head-audio-control-interaction-v1;
exclusive output creation, no overwrite or automatic retry. Before fitting,
commit product implementation/tests and pin that full source OID, source-plan,
parent, scripts, native plan and response calibration in the run plan.
Exact worker/validation/native commands are frozen in that plan before launch.

Apple M5/24 GiB, Torch 2.11, explicit mps environment, one CPU thread.
Training bound 3600s, 300s per worker, 12 GiB sampled process footprint.
Validation max 600s/4 GiB. Each native arm max 1500s/180s per case/4 GiB,
serial CPU. Owner STOP, nonfinite valid loss/gradient, input drift, frozen-path
mutation or any bound terminates the run and preserves its failed output.
No model-backed training is concurrent with native generation.

Current two-second startup/service qualification is necessary but not sufficient.
Any retained endpoint also needs the separate 30-row/eight-second benchmark
with its exact current loader/probability settings and client protocol.
Do not change or silently bypass the benchmark worktree during this comparison.

If native control/peak feasibility improves without independent regressions,
retain that bounded mechanism result and then consider joint adaptation and
actual generated-state response learning. If NLL improves but native quality
does not, do not scale the unchanged fit or declare audio conditioning solved.
If gains require harmful LN or pressure changes, the result is a tradeoff.
No result here resolves missing LN/coordination semantics in continuation.

## Implementation and execution receipt

The optional H interaction, strict checkpoint option and diagnostic factor
method are committed at de5d560ce06ca0185087488b982e15cab394b136. Sixteen focused
CPU/MPS checks pass in 6.62 seconds with the new head_audio_modulation tests,
the existing head-recovery tests and controlled ownership tests. These check
zero-initialized identity, learnable mixed effects at BOS, unchanged residual
and R/R1 at fixed conditions, and actual native/dense scoring. No quality claim.

Frozen plan SHA
5ddf7ca7607ec50027267e7d8b0089af1d5a8d9023a74b01ccc8720efedd5176
pins all five scripts, source, old factual helper, draw plan, baseline and
native plan before execution. Script syntax compilation passes. Commands use
`uv run --extra mps --extra render --group dev python` with, in sequence,
the new owner's validate.py initial, run.py, validate.py terminal,
evaluate.py additive and evaluate.py modulated. Initial validation must
reproduce profile-only H/row means and exact two-arm probability identity
before the optimizer run starts. Existing artifacts are only read.

## Initial validation and pilot

Initial validation completes in 16.32s and both arms exactly reproduce the
profile-only means: H/R/row nats per second 32.01912594/1.31904570/11.41714105,
macro row NLL 1.57116253. Receipt SHA
cfb8fb4aa4dbd60c5df47af0898b965050fa472b408f9dcf98eef71613e6dff6.

The first 16-update worker completes in 25.82s, peak process footprint
2760166616 bytes, with 513108/588372 trainable H parameters. Both H paths and
the new interaction have nonzero gradients; all frozen tensors pass exact
comparison. The live serial run continues; no endpoint or native result yet.

Analysis plan SHA
686ece3d797e6bb8507d89504e1490333b6ec3702a94c110a177d424578f4940
pins the scoped comparison and fixed Lens rendering scripts before native
results. Guards and criteria are unchanged; attack excess, interaction and
variation remain independent descriptive channels, not one quality scalar.

## Completed fit and diagnostic evidence

Both arms finish 512 updates in 1123.78s, maximum sampled footprint 3396128296
bytes. Additive checkpoint SHA
2d2f1d87f59e391f014c2f37aec83f9e70eee6c8f3660715ad5de373bde8f461;
modulated SHA
a08005e84755ba7e10e5b375d59c22422d7a7c835cf0d55badbc6cfe8597e13a.
Supervisor result SHA
71555b60c44c6b4ca17b14cde68d08d4abbf2c8a4d6c960ffa87093f1b0e4c5d.
Every segment verifies frozen weights and carries optimizer state.

Terminal H NLL is 31.65408468/31.62287877 nats per second, versus initial
32.01912594. R and row scores remain exactly the initial values on the same
factual states. Validation result SHA
4e64e0938826e783cd08e9724b5448a1957358e0c9084b570a73423c98b0f369.
No quality/promotion conclusion follows; complete native generation is pending.

A read-only post-hoc base probe on all five panel audios holds query controls
fixed while centering only the encoded audio. Its mixed D6-minus-D2 contrast
has RMS .00580–.00802 logits in the modulated model and zero in the additive
base. This shows a learned but small interaction; it is not an event law,
musical-quality score or endpoint-selection metric. Pooled base contrast std
also includes the ten different millisecond phases and must not be described
as purely temporal/music variation. Interaction plan/result SHA respectively
23afc26e8ec07ba648745c33d49eeda3e9d31b53b02fb2521b229e5c55868252 and
1d0057b837e557c2b32ce07bd44589e086a79aa8de6b588ba090d2c75b41e8be.

### Same-audio alternative exposure is sparse in this recipe

The actual draw ledger contains 736 charts on 666 byte-distinct audios: 601
audios have one observed chart, sixty have two, five have three. Only eight
draw pairs from different charts on identical audio overlap in time, totalling
64 seconds. Five pairs have difficulty known throughout their common interval;
only three use whole-song difficulty on both sides, all in [0,8000). Other
controls and prior chart histories need not match. Exposure result SHA
3fb711682d0dfae422b4cb69b99e141d866a8abcc0e1957c89f0b72d94e309d1;
the owner's audit_draw_pairs.py records input/script identity and exact draw
indices. The earlier unextended exposure receipt remains preserved.

This supports considering deliberate same-audio alternative sampling. It does
not prove generalization requires paired examples, that conditional variation
is unidentifiable in the full corpus, or that this sparsity caused the native
failures. It describes the frozen draw measure, not all parent training.

Any such follow-up must preserve the many-to-one H projection: a higher-star
source H can admit lower-star R1 materializations. Do not turn every alternative
source H into a negative under the other chart's difficulty. Paired factual
likelihoods keep each chart/prefix/control consistent; a difficulty discrimination
target belongs to justified complete-outcome labels or a proven infeasibility
witness, not assumed one-to-one skeleton labels. This is a design constraint,
not an additional intervention in the running study.

## Completed native result and research decision

All 56 new complete native cases finish, 28 per endpoint; the comparison
reuses the 28 profile-only cases. Additive/modulated elapsed generation is
568.94/561.38 seconds. All new cases preserve export/reparse identity and the
sub-20-ms attack guard. Their case-ledger SHAs are
32412569f7a85ae22041eb10e1e8175d5bdca6d2c278812751ca67e8f7d9d018
and 1a7c4e9b59360c303a0cdf45a276e79496c9e44613488125365107b22200db11.
The comparison SHA is
0588d5f261ed887bb459f501e477f24bb41e318679d9c64d17e23edcf9260ee2.

Profile/additive/modulated D2 whole-star MAE is
1.38534246/1.61353541/1.78828426; mean mandatory-H floor excess above two is
.40094021/.60124812/.53201561. Both primary criteria fail. D4 MAE is
.56543489/.58943952/.51703959 and six rich-LN fraction MAE is
.10634985/.08260302/.07479278, but D6 MAE worsens from .40625427 to
.65667281/.72654674, failing the guard. Additive has four newly failed
numeric checks, modulated two. No endpoint is promoted.

Additive D2 Zenithfall seeds zero/one have H floors 3.27837775/3.12661473;
modulated seed zero has 3.16875630. Even the upper edge of a 2±1 star band
is infeasible for any row realization of those mandatory H plans. Modulated
seed one's floor is 2.95930615, which does not establish that same band claim.
The restored D3 range after the temporary D4.5/LN .6 override has scoped
proxies 4.100745/4.527584. Without a static-D3 comparator, do not attribute
the discrepancy specifically to lingering override effects.

The main Lens review reads all forty pages, source plus both arms/two seeds
for Classic, STYX and Blizzard. Receipt SHA
3e655e5c36c855b0ec1e0b71602781da314efc85018690fa7494242717fba8d6.
Persistent anchors and mixed TAP/LN still occur, but do not certify surrounding
coordination. Blizzard often becomes nearly pure LN. Additional witnesses
include modulated Stream's fourteen-head column-0 recurrence and additive
Trill's 29 solo TAPs over 3390 ms, median head gap 120.5 ms. The modulated
Trill run is shorter yet sustained attack excess is much worse; maximum run
length is not a sufficient quality order.

The original worst-D2/additional-review queue was not completed. Human
steering prioritized deeper matched-ranked coordination analysis and allows
restarting the architecture/recipe instead of repairing inherited weights.
The follow-up Note 2026-09-28-coordination-frontier-and-ranked-contrasts owns
that work, including these LN and recurrence contexts. This is not a full
semantic pass or fulfilment of every originally planned review item.

Before comparison execution, its whole-rating bound assertion was found to
apply to a mixed-control case with no whole-rating field. Version two omits
that inapplicable assertion; original scripts/plans remain preserved. The
revised analysis-plan SHA is
f73b31d965e7cca208e54e4f09b02da62b96dac2770aee8204befad9b794b47f.
No acceptance criterion, checkpoint or output was changed.

Maximum two-second startup/service is .910074/.393722 seconds additive and
.909352/.478564 modulated. Rendering overlaps some late modulated cases;
do not treat elapsed differences as isolated speed gains. The separate
30-row/eight-second benchmark was not run and its worktree is unchanged.

Recommendation: REFINE, not SUPPORTED. The matched H-only recipe did not
establish low-difficulty or feasibility benefit despite better validation
NLL and a learned mixed audio/control effect. Do not scale it unchanged or
promote the D4 average. Frozen audio and sparse genuine-alternative exposure
remain limits of this intervention; broader joint conditioning is not
disproved. Curated result owner: docs/research/head_audio_control_interaction.md.
No new fit is selected or running at this boundary.

Completed result prose is committed at
0a74a198f73dec42753e0c92fb9c98ee298e4710; executable intervention and tests
remain de5d560ce06ca0185087488b982e15cab394b136. The documentation commit
does not change weights, probability law, selected model or benchmark.
Local-only publication; this Note remains proposed with no accepted revision.
