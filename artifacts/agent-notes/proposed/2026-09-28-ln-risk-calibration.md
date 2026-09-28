# Agent Note: Conditional LN release calibration and decision ownership

Note ID: 2026-09-28-ln-risk-calibration
Status: proposed
Kind: investigation
Created: 2026-09-28
Updated: 2026-09-28
Product revision: dc50ce2b4849688f3cf0a33220c68a73ef55258c
Scope: Independent factual LN-risk calibration and generated release-decision anatomy
Related: 2026-09-28-coordination-frontier-and-ranked-contrasts, 2026-09-28-clean-joint-proposal-learning

## Question and hypothesis

Joint training is live on the primary worktree. The preceding goal turn made
progress through a new probability mode, passing checks, actual optimizer
updates and a verified resumed segment. No playable model is qualified.

Short generated LNs can arise because R1 already overpredicts release on true
histories, or because generated H/row histories create different opportunities
and states. Factual release calibration can separate these mechanisms without
rewarding longer LNs indiscriminately. Joint co-release probabilities can also
be wrong while each finger's marginal release probability is correct.

The implemented observer will consume externally supplied complete-row
probabilities and a declared chart scope. It introduces no sampler change,
pattern mask, new physiological label or gradient intervention. Main training
source remains fixed. A managed worktree, release-calibration, starts from
ef42095a6e764b0374edbaa36b8ddf87c32364c9 and uses branch
codex/release-calibration for the evaluator.

## Experiment Card: ln-release-risk-calibration-v1

Revision: 4
Accepted revision: none
Execution authority: the continuing user goal includes research, useful
regression evaluations, local code and note commits.

For each factual head-bearing row, inspect every LN open before the row.
Record its true age, strict-past H frequency, other held fingers, predicted
release marginal and observed release indicator. Keep pure-R event mark
calibration separate: those rows condition on a release time that already
occurred and cannot assess R's timing law. Retain joint pair release
probabilities, actual co-release labels, hand relation and shared-origin facts.

Also record finite-support freedom: could the sampled/referenced hold have
continued at this event? Singleton R support and forced release subsets are
decision-ownership evidence. Do not infer that the neural row model chose a
release time which was already fixed upstream.

For each real hold, multiply no-release probabilities along H rows at which
the factual trajectory actually continues holding. This is a teacher-path
diagnostic, not free-running lifetime probability: alternative row choices
would alter subsequent state and R events. Preserve incoming origins, censor
open holds, and never read later endpoints to complete a crop.

The origin must distinguish reference, generated and synthetic traces.
Calibration against a policy's own sampled actions is tautological evidence
for fidelity; generated traces are used only for decision anatomy. Brier and
calibration summaries remain per role and scope. Undefined denominators stay
undefined. No marginal or joint statistic is an automatic BAD label.

Focused tests will compare two joint row laws with identical per-finger
marginals but different co-release mass; verify incoming context and censoring,
scope additivity, mirror transformation, H/R support interpretation and exact
scope/probability alignment. These tests establish observation semantics.

After committing the observer, score the three clean-joint initializations
and step-32/step-512 checkpoints on the eight previously read ranked LN contexts:
Shizuku, Non-breath oblige, Hot Chocolate, Esper, The Last Page, Someone In The
Crowd, Until the end of time and Bedroom community. Freeze source hashes,
scope bounds, whole-chart numeric controls, model identities and commands
before evaluation. These are TRAIN diagnostic references, not unseen test
quality or new human labels. Keep actual human style annotations separate;
do not infer a positive style control merely from a case's description.

Compare predicted versus observed release mass, Brier error, joint pair
calibration and per-hold continuation-risk traces at matched factual queries.
If the error is already present there, inspect representation/objective
allocation. If factual calibration is reasonable but native organization
fails, inspect generated timing/history exposure rather than extending every
hold. Interpretation remains exploratory; the small selected contexts do not
estimate a population causal percentage.

Use the existing MPS/render/dev environment with one CPU inference process,
at most 900 seconds, 4 GiB process footprint and 1 GiB new outputs. No second
accelerator fit or native generation overlaps main training. Run source work
in the isolated evaluator worktree and use the existing local dataset by
explicit immutable paths. New owner is
artifacts/joint-audio/20260928-ln-risk-calibration-v1 in the primary workspace.
Stop on source/hash drift, inconsistent support/probability alignment,
nonfinite measurements, resource limits or owner STOP; preserve failures and
do not overwrite/retry automatically.

## Exploratory execution and revision-two scope

The observer was implemented at ef90a21943ed4b56070f2679c3d3c9a2883be699
in the managed release-calibration worktree. Nine focused tests passed in
0.89 seconds: joint-law distinctions, true incoming holds and censoring,
scope additivity, mirror symmetry, H/R support, exact query alignment and
float normalization. Main training code remains pinned to ef42095.

Version one scored eight TRAIN contexts against initial and step-32 states
of all three arms in 19.175429 seconds. Version two included step 512 and
separated the exact head-signature and conditional release-subset NLL. Its
plan-v2.json was frozen before execution; this Note records the protected
scope extension afterwards. Neither revision is accepted. Both executions
are exploratory under the continuing user authorization, not conforming
runs of an accepted Card. No generated chart was used as a quality label.

Version two completed in 42.856003 seconds with peak sampled footprint
777733536 bytes, CPU single-threaded. All factual target rows were supported.
The diagnostic introduced no fit updates or additional accelerator workload. The source
head signature is a diagnostic condition, not a new inference input.
For each row, the numerical chain-rule identity was checked to 1e-10:
joint row NLL = head signature NLL + release subset NLL given that signature.

On Shizuku, early initialization to step 512 changes the eight-second summed
head NLL from 101.37 to 85.81 nats and release-given-head NLL from 62.86 to
74.73, while total row NLL improves from 164.23 to 160.54. On Non-breath
oblige the inherited arm changes 138.30 to 105.98 for heads and 72.85 to
82.19 for conditional release, improving total from 211.15 to 188.17.
This demonstrates a loss-allocation tradeoff at these factual states, not a
population causal percentage or proof that every alternative is bad.

The error is not uniformly excessive release. Initial inherited Shizuku
predicts total H-release mass 57.94 for 65 actual releases, yet assigns 17.09
release mass to 32 factual continuation opportunities and underpredicts
joint release (9.84 versus 22 observed pairs). Until the end of time further
distinguishes a held anchor from a short-LN train. Fresh at step 512 assigns
.788/.748/.722 release probabilities to three continued anchor events when
conditioned on factual heads; inherited gives .186/.148/.122. Untrained fresh
appears to retain that anchor on an extremely improbable head branch, so it
must not be credited with learning the organization.

Full report and derivations are committed in docs/research/ln_release_calibration.md
at 5b0dbeb6a5c5c5240bde0cf8325e54fee365d6c0. The evaluator remains on
codex/release-calibration, not merged into the live model worktree. Scripts
pin the earlier evaluator code commit ef90a219; rerunning under the later
documentation HEAD needs a new explicit execution plan, never silent bypass.

Evidence in artifacts/joint-audio/20260928-ln-risk-calibration-v1:

- Source/scope SHA ce40ee32c6a98c8a5834640f4b2df7562939ba6d56276911a11c81b78d683ba3.
- v1 result SHA 29a2e1a97eeca51109a203d012661979b42c877e0fef2d00f2b2d82406fd9ef2.
- v2 plan SHA f234a12a9439eb2b3d3f9a565df7e36e06a773e75493be27c87fa71749903be0.
- v2 result SHA 1f80b8e24b1d3ea9c62d26c77210864e99034f4af3373f29d7cea8d7bfd31de1.
- v2 cases SHA 013c4512171f2e57e5023dbe05c914f7403735092fb68a3101fe8eb74ea11e64.

Decision: REFINE. A positive-weight conditional release loss preserves the
unrestricted data-distribution optimum while changing finite-model allocation;
it is a proposed follow-up, not an executed intervention or a repair claim.
The current live three-arm recipe must remain unchanged. Keep actual native
control, musical organization and gameplay evidence as separate requirements.

## Revision three: release cardinality versus identity

The row composition mark is (head count, LN-head count, release count), not
only a head-count pair. Therefore fixing the observed head signature does
not cancel composition's release-count contribution. Fixing both that
signature U and release cardinality K does cancel the mark probability and
within-mark normalizer. The conditional identity law then depends on layout,
row consequence and deployed row preference. This mathematical distinction
changes attribution and is the reason for a further probability decomposition.

Use the already frozen scores-v2 reports, with no new model sampling or fit.
For every exact reference query decompose release-subset NLL as cardinality
NLL plus identity-given-cardinality NLL, by summing the existing 16-mask
probabilities over equal popcount. Preserve H/R role and true incoming holds;
count singleton identity support explicitly. Compare initial/32/512 in every
arm and context. Keep sums, contributing opportunities and per-event evidence.
Check the chain rule and agreement with the preceding row loss to 2e-5 nats,
allowing documented floating-point normalization in the observer. Do not call
a singleton identity decision learned coordination.

The decision is where a candidate release-focused objective needs supervision:
release cardinality, selection among held roles, or both. An increasing
conditional total with decreasing identity loss localizes the measured
regression to cardinality at those factual queries; it does not identify
which shared parameters or training examples caused the change. Conversely,
identity errors at fixed K expose a role-selection issue. No pass threshold
certifies playability, and unchanged factual probabilities would be a null
diagnostic result. This is exploratory secondary analysis, not an accepted
Card or an independent new dataset.

Freeze a script/plan before execution under the existing LN-risk owner, write
new release-factorization.json only, refuse overwrite, and verify every input
report hash from scores-v2/cases.json (SHA 013c4512171f2e57e5023dbe05c914f7403735092fb68a3101fe8eb74ea11e64).
Use standard-library CPU analysis, no network or checkpoint loads; bounds
60 seconds, 512 MiB new memory and 64 MiB output. Stop on hash drift,
nonfinite probabilities, an unsupported actual mask, identity mismatch or STOP.

## Revision-three result and final-law attribution plan

The exact secondary analysis completed in .179959 seconds, maximum RSS
32063488 bytes; maximum chain-rule error was 4.27e-14 nats. Plan SHA
958e7c3e8ca61c5533d6924ca540cb68c0d83b1071b3763f9bdb3d06abe95ca4;
release-factorization.json SHA
0821580897f76e57b6f771dccb80349f18913dc3c3845cffc96d438072218c30.
Early Shizuku initial→512 cardinality NLL is 56.938→69.034 while identity
NLL improves 5.921→5.697. Inherited Non-breath cardinality is 51.922→63.106
while identity improves 20.928→19.086. The earlier total release regressions
therefore localize to cardinality in these contexts, not worsening selection
among a fixed number of fingers. Until the end of time also exposes an
identity error: at 72260 ms inherited step512 assigns .902659 to releasing
one finger, but only .269213 of that mass to the short-LN finger rather than
the held anchor. These remain factual conditional diagnostics.

Revision four keeps the same eight sources and nine model states, but reads
the exact row composition and RowConsequence energies in a CPU forward pass.
Compare the deployed law against three fixed-prefix counterfactuals: removing
only learned row consequence, removing only empirical recovery preference,
and removing both. All clocks, source histories, controls and finite support
remain identical. Verify reconstruction of the unmodified neural law to
2e-5 and reproduce prior whole/conditional NLL to 2e-5 nats per row. This
is an inference-component attribution study, not a retrained ablation or
native quality experiment; NLL's nonidentifiable split forbids assigning
unique historical training blame from these removals.

Record the same exact U/K/identity decomposition and selected Until anchor
probabilities in all four laws. A material conditional error improvement
under removal identifies a current-score contribution on the factual path;
opposite signs across examples reject a universal removal recommendation.
No metric threshold automatically changes the sampler. Main fit and its
milestones remain untouched. Counterfactual probabilities must preserve
actual support and source factuality, including genuine singleton decisions.

Freeze score-components.py and components-plan.json before execution, pin
the evaluator worktree at 5b0dbeb6a5c5c5240bde0cf8325e54fee365d6c0 and use
the exact model/checkpoint/source hashes from plan-v2.json. New output is
components/ under the existing LN-risk owner, refusing overwrite. CPU one
thread only, 900 seconds, 4 GiB footprint and 1 GiB new output; no fit or
network. Stop on prior hash/support/resource/STOP conditions. This further
extension is proposed and exploratory under existing execution authority.

## Revision-four result and reusable evaluator

The component run completed all eight contexts × nine model states × four
laws in 27.856856 seconds, footprint 875923016 bytes and 21650352 output bytes.
Tool session 58760 is terminal exit zero. Baseline reconstruction and matching
prior factual losses passed. No main source or training recipe changed.
Components cases SHA
9a85f7a14baf5858109ca114177613f485deb8843d8a70eac3793f85e6ff7036;
result SHA 1522b41d640b99424d4e667a6c720dd471f042e5bb98278fb7e0cee004f18f6c;
plan SHA b76a6df9a5a36fadfbf54ac3ebee9a1b77871148c3970a61d0280b139b03858b.
Each case retains exact composition/consequence arrays and all four laws in
hashed npz files, with query-level U/K/identity losses and release marginals.

At early-512 Shizuku, removing learned row consequence worsens cardinality
NLL from 69.034 to 75.264 while identity changes 5.697 to 5.612. In inherited
Non-breath it changes 63.106 to 67.707 and 19.086 to 18.689 respectively.
Thus the residual is compensating for some current cardinality error; it is
not a universally harmful inherited module. In inherited Until, whole-scope
identity NLL worsens 23.822 to 25.252 on removal, but the specific 72260-ms
correct-finger probability rises .2692 to about .3370. Local and integrated
effects can disagree, and the latter still does not certify playability.

Removing empirical recovery preference leaves conditional cardinality and
identity losses unchanged to <1e-13 on the inspected Shizuku, Non-breath and
Until factual queries. This excludes that preference as the source of those
conditional errors at those states. Other contexts do change, including a
Bedroom model/role component by 4.265 nats. Do not generalize the exclusion
to all source states or generated histories.

The reusable row_likelihood_parts observer, six new focused tests and the
expanded self-contained analysis are committed at
dc50ce2b4849688f3cf0a33220c68a73ef55258c on codex/release-calibration.
All 15 selected tests pass in .18 seconds in the existing MPS/render/dev
environment. Comparison with the 288 saved laws and 23616 queries matches
all factors to at most 3.2453e-7 nats, attributable to explicit normalization
roundoff. The code uses log-space arithmetic, exposes identity support size,
keeps singleton/empty cases distinct and refuses unsupported target conditions.
Documentation links resolve and diff whitespace checks pass. The evaluator
worktree is clean; no remote push or merge into the live main checkout.

The exact derivation now prevents a misleading module attribution: conditioning
on U and K cancels count-family mass and its layout normalizer, but conditioning
on U alone does not. The K diagnostic is still influenced by layout and later
row energies. Neither this factorization nor an inference-time energy removal
identifies a unique historical training cause.

Decision remains REFINE. Proposed loss allocation is now explicitly
L_H+L_R+L_U+lambda_K L_K|U+lambda_I L_V|U,K with positive weights. It preserves
the unrestricted conditional-distribution optimum, but has not been trained.
Proposal role memory and independently calibrated gameplay response remain
separate questions. The current three-arm fit continues unchanged. This new
evaluator must not be merged into its guarded executable tree while live.
