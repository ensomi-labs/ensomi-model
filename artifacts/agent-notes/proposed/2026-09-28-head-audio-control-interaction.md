# Agent Note: Conditional audio evidence in the head-time base

Note ID: 2026-09-28-head-audio-control-interaction
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 784ac6fc0e03aa6600c2c805c778ab788d6b126d
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
