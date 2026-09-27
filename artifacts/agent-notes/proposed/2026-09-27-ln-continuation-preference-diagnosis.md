# Agent Note: Why native continuations prefer fragmented holds and long jacks

Note ID: 2026-09-27-ln-continuation-preference-diagnosis
Status: proposed
Kind: investigation
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 24061b918be7075f1efc70341675fe066c2e45b6
Scope: Native release preference, R1 consequence scores, continuation response coverage and ranked four-star temporal organization
Related: 2026-09-27-controller-semantics-and-native-qualification, 2026-09-27-player-response-frontier, 2026-09-27-playability-regression-evaluation

## Objective and authority

The user reports that roughly four-star outputs still have fragmented irregular
LN lengths and severe long jacks. Real high-coverage LN streams remain valid;
coverage or a short LN alone is not the fault. The immediate task is to identify
the specific formation mechanism, improve continuation/frontier selection and
establish regression evaluation against playable ranked organization. The user
explicitly rejects new hard-coded shape filters: proposed bad continuations
should lose through their predicted gameplay response in context.

Standing local research, implementation, run and commit authority applies. No
exact Note/Card acceptance, human annotations, remote push or model adoption is
inferred. Full playability remains the active goal.

## Code facts and discriminating hypotheses

The default RowConsequence frontier2 is a learned complete-row residual over
immediate post-action facts and passive clocks through two future H times. It
does not evaluate a realized hypothetical continuation. In the native features,
next_r is now + 1, representing a possible event clock rather than a predicted
release. The short recovery-support check is existence of a legal TAP future,
not a musical/gameplay response. The optional ResponsePlanner does actually
expand private elapsed-time futures, but its current cost covers sustained
attack excess only; LN organization is absent. No default planner integration
is asserted. These are capability boundaries, not yet attribution of each failure.

Three explanations predict different observations:

1. **Release support compression:** native H and prior LN choices leave a short
   feasible release window. Short releases should cluster in conditioned waits
   whose deadlines are close to LN starts. A low rate of such windows defeats
   this as the main explanation; release-only events far before nonbinding
   deadlines point elsewhere.
2. **Learned release/type preference:** free choices remain but R proposes early
   releases or R1 chooses unnecessary/multiple closures. Inspect raw versus
   conditioned waiting laws and complete-row score contributions on actual
   reached states. Disabling amount feedback alone already fails, so do not
   repeat that intervention or treat global fraction as the outcome.
3. **Insufficient continuation comparison:** alternatives with better sustained
   organization exist, but the default local consequence score fails to rank
   them. Compare selected rows and real private futures at matched prefixes,
   preserving timing ownership and committed publication. Candidate sampling
   limitations and a missing LN response must be distinguished from poor scoring.

The closest analogue for this audit is model-predictive continuation selection:
the existing ResponsePlanner already samples actual future actions and commits
only a prefix. This is an investigation of that implementation and its missing
response channels, not an algorithmic novelty claim. A new learned response or
training method requires its own bounded comparison after this diagnosis.

## Exploratory diagnostic card: ln-release-origin-v1

Revision: 2
Accepted revision: none
Execution authority: the user's explicit diagnosis, improvement and evaluation
request plus standing local experiment authority.

Baseline: product 24061b918be7075f1efc70341675fe066c2e45b6; the 36 complete native
outputs from 20260927-controller-semantics-v1, their pinned four-arm ledger and
the frozen 8,774-chart ranked reference. No new model sampling or weight change
in this first pass. Use corpus charts in [3.5,4.5) stars, source hashes and
metadata hashes verified. Keep song identities, control scopes and LN coverage
strata separate; broad chart-level facts are diagnostic, not local BAD labels.

Measure complete LN duration distributions, observed head/tail relationships,
held coverage and neighboring LN duration changes. Short-duration cut points
are descriptive channels only, never masks or automatic penalties. For actual
generated rows reconstruct prior exact state, R-only versus H-role releases,
the earliest/latest feasible release window, conditioned versus unconditioned
waits and whether a deadline is actually hit. Preserve open-hold and scope
boundary semantics. Do not equate a conditioned wait with a forced event.

Primary discriminant: what proportion of short observed holds close at H rows,
unconditioned release proposals or deadline-conditioned proposals, and how far
the applicable deadline lies from their starts. This descriptive audit has no
adoption gate. It selects the next causal intervention. Corpus contrasts must
include high-coverage references rather than rewarding fewer LNs or uniformity.

CPU one thread, at most 600 seconds and 4 GiB; fresh output under
artifacts/joint-audio/20260927-ln-continuation-diagnosis-v1. No overwrite, fit,
post-generation pattern removal or reference retiming. Stop on STOP file,
source drift, invalid replay or resource bound; retain failures explicitly.
Before starting, record the diagnostic script hash and input ledger hashes.

Next causal probe must use current-policy reached prefixes, full audio and
matched real futures. A probability change is not a quality gain; inspect the
identified contexts with Lens and retain independent full native evaluation.

## Revision 2 execution correction and task decomposition

Attempt 46728 terminates after extracting the complete 2,506-chart [3.5,4.5)
corpus. It fails in the source-reference adapter because the live-switch case
has no source reference. Preserve run-v1/identity.json and corpus.json and the
failure receipt. Skip source-reference extraction only for a case with no
declared source, and use fresh run-v2. All 36 native cases remain in the audit;
the generated switch is still evaluated by its own ranges. The population,
measurements, bounds and lack of model intervention are unchanged.

The user clarifies two separate success conditions: the proposal model must
learn the broad real distribution, and suitable legal proposals must be
accepted by continuation/frontier. Audit the recipe for representation and
exposure caveats rather than assuming the learned distribution is adequate.
Continuation is intended to model committed player state and actual hypothetical
future response, including LN entry/release, coordination and accumulated jack
strain. Corpus similarity alone is not a replacement for that response. Modules,
capacity and recipes can change within the formulation and final interfaces;
no shape-specific hard mask is selected.

## Release-origin audit result

Run-v2 completes in 24.679 seconds with 0.310 GiB peak RSS, 2,506 verified ranked
charts and all 36 native outputs. Source clocks and original endpoints are kept.
Native LN starts choose diagnostic scope membership; holds are not clipped at
the scope boundary. Corpus body scopes use first head through last object end.
Those duration conventions are reported separately, not silently pooled.

Short holds are not by themselves anomalous: among 108 high-coverage references
with at least 50 LNs and any-held body fraction at least .75, the median of
chart-median LN duration is 164 ms. This is a descriptive unweighted chart
summary, not a training calibration, BAD threshold or player-capacity law.

The inspected native examples largely do not hit release deadlines. Memory
Classic seed 0 has 310 holds below 200 ms: 298 close at H rows, 11 at
unconditioned R proposals and 1 at a conditioned R proposal. None hits the
deadline. All 298 H closures have some legal same-head-count row with zero
releases; layouts/types in that alternative are not yet held fixed. Memory
Blizzard has 1,060 below-200-ms holds: 917 H, 120 unconditioned R, 23 conditioned
R, only one exact deadline hit. Thus recovery-window compression does not
explain most observed short tails in these two cases. It can still affect
which states are reached and which particular alternatives are good.

Organization differs even when median duration is plausible. Source Blizzard
median duration is 89 ms, actor 171 ms, memory 101 ms; simply lengthening every
LN would miss the issue. Median neighboring-LN absolute log-duration change is
.0113 in source versus .4689 actor and .3165 memory. Source STYX is naturally
more variable (.4096), demonstrating why this descriptor is not a universal
penalty. H-interval irregularity and R1 release-span choices remain confounded.

Result identities: corpus SHA d8388e2fd7a7ef7782916f32e1dc67604b620294e63d42bca459f10f0bbe799c;
native SHA 178adea6d39134a6d7e954c2929fb714c52993e4451cbc8f0961e4bc850cefe0;
release-ledger SHA 6c2f00740a099514e72bd05aa4bca9a382071054e79bb52cdbc1f9eb9139e95c.
This executed diagnostic is complete; no new semantic review or improvement
claim follows from these descriptors alone.

## Experiment Card: source-head-ln-diagnosis-v1

Revision: 2
Accepted revision: none
Execution authority: standing explicit local research and causal-improvement
request; exploratory, no adoption or lifecycle transition.

Question: is LN temporal irregularity primarily induced by generated H timing,
or does R/R1 retain the preference on musically authored H times? Baselines are
actor/memory feedback-on from the complete paired LN ablation, eight ordinary
cases per checkpoint. One intervention replaces H with each ranked source's
actual integer H timestamps. No source row counts, lane choices, LN types or
endpoints are passed to the model. Audio, checkpoints, controls, recovery,
seeds, LN feedback, R and R1 remain unchanged. The live-switch case is omitted
because source H would not provide native response to a control update.

Observe per-song/per-seed LN duration relationships in milliseconds and H-relative
coordinates, release-role choices, held occupation, actual recovery and named
local contexts. Improved regularity alone is not a reward; source-compatible
variety and musical organization require Lens. A consistent change toward the
reference organization across both seeds of at least two LN-rich songs supports
an H contribution, but does not prove R1 is correct or prescribe source timing
at deployment. Mixed outcomes require refining the timing/action interaction.
All native amount/difficulty/publication checks remain in force and visible;
this diagnostic is not a checkpoint promotion attempt.

Use the packaged qualification CLI through source_head.py, source
24061b918be7075f1efc70341675fe066c2e45b6. Pin the prepared plan and script hashes
 before execution. Fresh source-head-v2 output, 16 complete exports, CPU one
thread, 4 GiB process bound, 540 seconds per checkpoint, 180 per case and 1,200
overall. No training, overwrite or automatic retry; stop on STOP, source drift,
noninteger source times, unsupported H capacity or incomplete generation.
Exit 2 numeric failure and exit 3 pending review remain unpromoted outcomes.

Revision 2 corrects the supervisor's footprint override from a nonexistent GiB
field to the actual typed footprint_limit_bytes=4294967296 field. Attempt
source-head-v1 exits at Hydra validation before any model case executes; preserve
its plan, identity and actor log. No scientific input or bound changes. Use
fresh source-head-v2 rather than overwriting or claiming native evidence from
the failed startup.
