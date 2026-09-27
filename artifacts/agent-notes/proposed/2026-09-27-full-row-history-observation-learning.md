# Agent Note: Full-row learning with factual missing-history observations

Note ID: 2026-09-27-full-row-history-observation-learning
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-28
Product revision: 0882315ef23097e44e031707abd382d971b8c82c
Scope: Matched full-R1 factual learning, optional content-observation masking, native organization and style guards
Related: 2026-09-27-native-prefix-weight-state-decomposition, 2026-09-27-contextual-ln-count-conditioning

## Mechanism and alternatives

The count refit cannot change within-count routing at a fixed history, while
changed actual histories dominate its LN-law difference in four matched LN
cases. This motivates two separate changes: permit the actual row/history/layout
owners to learn, then test whether a factual missing-observation objective helps
their conditioning remain useful beyond detailed teacher content.

The closest analogue is modality dropout: ModDrop randomly omits input channels
while retaining sample identity and targets (Neverova et al.,
https://arxiv.org/abs/1501.00102). This adapts that observation principle to
learned chart-content context; it is not a novel algorithm or evidence that
gesture results transfer to native chart generation. The distinction from
replacing the past with sampled actions follows the consistency concern in
https://arxiv.org/abs/1511.05101. Exact state, including last-row facts, remains
visible, so this does not remove all historical information or guarantee that
audio becomes dominant.

Under a randomly selected visibility mask, supervision remains a conditional
likelihood of the same factual world with less observation. Generated-history
outcome learning remains a separate problem. Healthy LN continuity, jacks and
motifs may rely on history, so native/source guards must be able to reject this
branch if it flattens them. Full audio/H/R training and richer relationship
representations remain available; freezing them here isolates row-side learning.

## Experiment Card: full-row-factual-views-v1

Revision: 1
Accepted revision: none
Execution authority: standing user research, training, compute and local-commit
goal. No inferred exact-Card acceptance, model adoption or remote push.

Implementation source 0882315ef23097e44e031707abd382d971b8c82c; ordinary native
calls preserve the pre-mask probability law. Reuse the frozen 256 factual draws
and 22 validation charts from source plan
a766382082fa7e0b8674f88d9c7be128b41cf16d1fa7e881beb367336dbaddf4.
Both arms initialize actor128 SHA
364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3,
use reference_tilt, saved 60/50/50 recovery, default recovery preference and LN
feedback OFF. No new source draw or target/control change. This compares against
the previous count-only result but has its own matched full-R1 control arm.

Train row temporal, exact/fuse, audio-residual/context-condition projections,
joint layout, route/release residuals, row control/preview, composition,
layout modulation and row consequence. Freeze complete audio encoder and every
H/R neural parameter. Assert the frozen tensor set at saves. Actual R outcomes
can change through the model's own row decisions; freezing its network does not
make the whole release process independent of R1.

Baseline: every genuine source content observation is visible. Intervention:
exactly 64 of 256 windows, chosen without inspecting outcomes by a fixed seed,
hide their learned content observation at non-BOS row queries using TRUNCATED.
Actual BOS remains BOS. Physical replay, last-row facts, LN occupation/ages,
audio, H preview, controls, masks and action targets remain unchanged. H/R factors
never receive this visibility flag. All evaluation uses full ordinary history.
This is one observation-objective intervention, not a synthetic source suffix.

Run 128 updates with batch two, mean deployed source-row NLL per second, AdamW,
weight decay 1e-4 and gradient norm cap 1. Composition/row-control/preview/layout
modulation learning rate 1e-4; other trainable R1 parameters 3e-5. Keep the same
logical draws for both arms and record actual masked rows/gradients/losses.
No optimizer-state sharing between arms. Seed 279812. Preserve a visibility list
and parameter ownership in the frozen plan before fitting.

Use MPS, CPU one thread, serial fresh processes of 16 updates to bound the
observed heterogeneous-shape memory growth. Each process restores both model
and optimizer states from the preceding completed segment. Save both arms and
their exact draw index. Do not restart from an inferred live-state file: only
advance after the previous worker is confirmed terminal and its completion
receipt passes. At most 3600 seconds total fit, 600 per segment, 12 GiB task
footprint. A guard/failure preserves the segment; no automatic parameter or data
change. Stop on STOP, source/hash drift, nonfinite loss/gradient, altered frozen
weights, or memory/time limits. Final source evaluation uses both full and
missing views as separate diagnostics, not a merged NLL.

Native evaluation retains the eleven-case plan
48379544b1b012fec14c8223a16f23275bc476a967a798d0c2c4cb6e6389d863
and adds three prominent-style guards (Jack, Tech, Trill) on the same verified
Zenithfall audio, stars 4, unspecified LN, seeds 279701/279702/279703. The added
guards require an unfitted baseline and both fitted arms. Existing eleven
baseline outputs may be reused with pinned row hashes because ordinary law and
weights are unchanged. All fitted cases start from BOS, use complete audio,
feedback off, and preserve scoped publication. No future source heads or rows.

Primary quality decision: independent Lens/source review must find a repeated
improvement in at least two LN-rich songs across both seeds, without judged
organization loss in the third or in the three style guards. Numeric guards:
no new under-20-ms attack or publication failure; no additional declared LN
amount failures versus the initial actor in the original nine LN/control cases;
star absolute-error macro mean may not exceed the matched full-R1 control by
.1; Stream excess may not exceed that control by .005 seconds per case. Keep
each scope/seed/style distinct and inspect witnesses. Source full-view NLL may
not exceed baseline full-R1 NLL by .05 nats/row. Lower likelihood, longer median
holds, lower head counts or balanced columns alone cannot qualify a model.

The source/learning checks discriminate observation robustness, not calibrated
player-response semantics. If both full-row arms improve similarly, credit
decision capacity/source learning rather than masking. If masking improves only
its auxiliary condition, or hurts native motifs/control, do not scale it. If all
native quality remains bad, record REFINE and inspect actual input/state and
trajectory targets rather than promoting on NLL. Unaccepted exploratory evidence
cannot be called SUPPORTED.

Fresh owner artifacts/joint-audio/20260927-full-row-history-views-v1; freeze its
plan, source-data identity and scripts before launching. Native CPU one-thread
evaluation has a separate 2400-second total, 300 seconds per case, 8 GiB footprint.
The realtime benchmark worktree is unchanged; native observations do not replace
its 30-row/eight-second readiness contract. The full playable-system goal remains
active and unproven.

## Frozen execution inputs

Prepared plan SHA 94bb0e42bcc352a97f0f3ad25063ddf50b745773390ddf517a2884bb3b6e5645
pins all five executable experiment scripts and the reused factual source plan.
The resolved trainable set is 2,753,715 of 4,600,369 parameters. The visibility
list contains exactly 64 independently selected window indices. Native plan SHA
fdddd43cbd945ad7de14abb257e54fb71f8a3f4d858b3b72421b395bb2c484d8
passes the actual qualification plan validator for all fourteen cases, including
scope-owned amount gates.

Commands use `uv run --extra mps python` followed by the owner directory's
run.py, validate.py and evaluate.py. The supervisor starts and waits for each
worker as a real subprocess, verifies its completion and checkpoint identities,
then advances to the next declared segment. It does not infer liveness from a
receipt or retry a failed worker. Validation uses a separate bounded CPU process
to avoid retaining training allocations. Native evaluation produces 31 new
outputs: three initial style guards and fourteen per fitted arm; it reuses the
eleven already pinned ordinary baseline cases.

## Fitting and source validation complete

All eight declared workers are confirmed terminal and complete. Both arms reach
128 updates in 290.011 total seconds. No memory restart or source/optimizer
change was needed. Final model SHAs: full
32afd48e5dbad6714401735e01200963c9d447e9f8ca92057bf6c7b67ab36406,
missing 8ddf378e3e728f6b6a84c34fc5477b2cb3276782cfe40bf4da9222ce80350dde.
Segment receipt SHA 3532ec0e3f8cd560bf4de23abd38df520dc452b5d3281833a88cab88ef182a0b.
Frozen weights stay tensor-equal at every completed segment.

Separate CPU validation on the same 22 charts yields full/missing-view row NLL:
initial 1.730817/2.957741; full-R1 fit 1.618333/3.008688;
missing-view fit 1.623110/2.335572. Missing-view robustness improves in its own
condition, with only .00478 higher ordinary-source NLL than full-R1 fitting.
This does not establish any native-quality gain; the complete comparison remains
necessary. No checkpoint is promoted from these measurements.

Before native evaluation, product 13eefbc67d3e7668910efc612e2f35ba50a30798 adds a
behavior-neutral LN interaction observer to scope reports. Training/model source
remains 0882315ef23097e44e031707abd382d971b8c82c. The descendant changes only
evaluation, tests and its guide; it does not change any probability/sampler or
decision criterion. The observer distinguishes continuing-hold interactions
with TAPs, LN starts and releases, including canonical hand relationships and
co-start staggered releases. It reads only events within the observed scope,
retains incoming origins and assigns no BAD/style/strain label.

Four new semantic tests plus existing temporal/LN/actual-native qualification
owners pass 24 tests in 12.20 seconds. An initial NumPy boolean aggregation
caused non-JSON integers in qualification output; the real qualifier tests caught
it before any research generation. Explicit Python booleans fixed the report,
and all selected tests pass without exclusions. These added descriptors are
diagnostic, not a post-hoc replacement for the frozen primary comparison.

### Result Log: full-row-factual-views-native-v1

#### Experiment and reproduction

Owner 2026-09-27-full-row-history-observation-learning; Card
full-row-factual-views-v1 revision 1; accepted revision none. This is exploratory
execution under standing local research authority, not an accepted or adopted
model. Training source 0882315ef23097e44e031707abd382d971b8c82c and evaluation
source 13eefbc67d3e7668910efc612e2f35ba50a30798 are recoverable commits; the
latter's observer-only difference is recorded above. Unrelated AGENTS.md and an
untracked architecture walkthrough were outside the execution source scope.

The frozen fourteen-case plan, factual data/checkpoint identities, seeds and
commands are unchanged from the Card and frozen inputs. evaluate.py invokes
the native qualification runner on initial-guards, full and missing, preserving
whole-audio generation from BOS, feedback off, original recovery and full history.
Eleven initial cases are reused from the pinned count-study baseline, with three
new initial style guards and fourteen outputs per fitted arm. No output was
overwritten or retried. All 31 new generations complete normally in 492.669s
within the 2400s native budget, on Apple M5 CPU one thread. Fitting separately
completed both 128-update arms in 290.011s on MPS. There was no new training in
the subsequent formal-document revision.

Evidence owner artifacts/joint-audio/20260927-full-row-history-views-v1:
native-v1/full/cases.json SHA
6baf9ae418289c21d294ad4c5c6a2a71def08acba89794b690aaaa95742bef5b;
native-v1/missing/cases.json SHA
191d61b8f613b2260e9c08f27fe8a2ae10380f2d87a9cad319f5251b818bc6ad;
native-v1/initial-guards/cases.json SHA
5aaf6c4ce3cd174b3badee9fe78cd2087daf9c4ea64afe4e8d5cef96a371af8b.
All 31 export hashes are verified against these records. Complete H timestamp
sequences parsed from every output match the corresponding initial baseline
for both fitted arms, including the live-control case.

#### Results

Whole-song LN fractions, with seeds separate:

| Case | Initial | Full R1 | Missing view |
| --- | --- | --- | --- |
| Classic s0 / s1 | .145 / .191 | .217 / .154 | .198 / .040 |
| STYX s0 / s1 | .939 / .933 | .519 / .487 | .438 / .428 |
| Blizzard s0 / s1 | .993 / .993 | .972 / .953 | .947 / .908 |

Requests are .2171533, .4852810 and .8380463 respectively, all D4 with styles
unknown. Ordinary full R1 preserves both Classic amount passes and restores
both STYX passes, but fails both Blizzard amount checks. Missing view loses
Classic s1 and passes only Blizzard s1. The control override is still a separate
case; its fragments are not pooled into whole-song ratios or new quotas.

Stream-only Zenithfall, D4 and LN unknown:

| Seed | Initial stars / attack excess seconds | Full R1 | Missing view |
| --- | --- | --- | --- |
| 271200 | 4.861 / .043476 | 5.071 / .011005 | 5.012 / .030037 |
| 271201 | 4.984 / .106111 | 5.134 / .023828 | 5.236 / .087103 |

Both missing-view excess values exceed full R1 by more than the declared .005s
guard. Its auxiliary source NLL gain therefore does not establish a native win.
Tech guard initial/full/missing excess is 1.30685/2.95179/3.50569s, with stars
5.801/5.964/6.082. Full R1 Jack/Trill guards are 5.069/5.644 stars. Every added
style guard uses Zenithfall, so style causality remains confounded with audio.

The new outputs retain semantic_review pending. No new Lens, listening or
playtest judgment is inferred from these values. The primary replicated
organization criterion is not established. Maximum qualifier startup/service
seconds are .894/.338 for full and .778/.344 for missing; these are not the
separate 30-row/eight-second benchmark. Neither checkpoint is promoted.

#### Plan conformance

No protected field was changed. The declared evaluation-only source descendant
adds LN interactions without altering generation or the comparison criteria.
Reuse of the eleven initial outputs and pending semantic work are explicit.
This result log does not mark incomplete semantic qualification as complete.

#### Evaluation and decision

REFINE. Full R1 learning can improve some ordinary LN amount and Stream attack
outcomes without changing H, whereas count-only repair cannot directly learn
within-count routing. This does not isolate the contribution of any one trained
module. Attack excess and whole difficulty move in opposite directions in the
Stream cases, while Tech pressure worsens. Missing-history robustness improves
only its declared source observation task and fails native pressure guards
relative to ordinary full R1; do not scale it on auxiliary NLL alone.

Retain ordinary full R1 as a research parent, not a qualified playable release.
Complete semantic inspection before claiming improved LN relationships; use
the separate proposed 2026-09-28-style-difficulty-response-probe to distinguish
audio-specific failure from style/control response. Independent LN/coordination
response remains necessary because the attack-only planner is unchanged.
No Note lifecycle transition, human-label creation or remote publication occurs.

The numerical findings and their limits are incorporated into the self-contained
product report docs/research/native_pattern_failure_analysis_zh.md at
0f5ec10b34c43fe21a6143d371a2f2bc36070e47.

## Lens follow-up on ordinary full R1

Twenty-four time-proportional pages have now been viewed across source and both
ordinary-fit seeds, at Classic [88589,93589), STYX [1800,7800) and Blizzard
[40342,46342). Action/context records retain entering holds and complete actual
endpoints. Review record SHA
1c8324d43f3ab49c3928475ce4befbf3c18a9301c357ff5c7e09f099bda0f19d,
under the declared owner's review-v1. No new human labels/listening/playtesting.

Classic regains mixed LN/TAP activity in both seeds, with paired endings in one
and sustained holds under other-column TAPs in the other. STYX remains mostly
LN locally although its whole-song amount improves; its reference passage is
TAP flow, which does not make every alternative LN interpretation invalid.
Blizzard remains near-continuous changing holds; its reference sustained-anchor/
TAP/short-LN contrast is not restored. Continuing-hold/TAP pairs are source 12,
full seeds 0/1. A longer hold in seed 1 is accompanied mostly by LN entries,
demonstrating why duration alone is inadequate.

The replicated two-song organization criterion is still not established. The
missing-view arm and broader style guards have no new semantic approval.
Ordinary full R1 remains a research parent only. Its broader factual exposure
comparison is owned by 2026-09-28-broader-full-row-learning.
