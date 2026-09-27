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

Revision: 1
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
