# Agent Note: Conditional LN release calibration and decision ownership

Note ID: 2026-09-28-ln-risk-calibration
Status: proposed
Kind: investigation
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 5b0dbeb6a5c5c5240bde0cf8325e54fee365d6c0
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

Revision: 2
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
