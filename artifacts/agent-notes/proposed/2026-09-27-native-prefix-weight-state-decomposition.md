# Agent Note: Separate direct count changes from reached-history changes

Note ID: 2026-09-27-native-prefix-weight-state-decomposition
Status: proposed
Kind: investigation
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 80bbde1090e58faf4795fa6ff88d83f3c6082186
Scope: Crossed row-law evaluation on actual native histories after the matched LN-count refit
Related: 2026-09-27-contextual-ln-count-conditioning, 2026-09-27-candidate-supply-and-response-selection

## Question and authority

The previous goal turn is progress: two matched 128-update fits, 44 complete
native outputs, semantic inspection and a committed negative result. Source NLL
improves about 3.75%, while Classic loses LN amount and one Stream seed gains
pressure. Both learned count laws generate the same rows in the six LN cases.
No live experiment handle remains and no endpoint is promoted.

The next decision is whether local weight changes on identical reached states
explain the type shift, or whether changed reached histories dominate the
conditional-law difference. This is a counterfactual model-kernel diagnostic,
not a claim that two long rollouts differ only in their content history.
Standing user authority covers local research, runs and commits; acceptance and
remote publication are not inferred.

## Experiment Card: crossed-native-prefix-laws-v1

Revision: 1
Accepted revision: none

Baseline source 80bbde1090e58faf4795fa6ff88d83f3c6082186, executable tree clean;
unrelated AGENTS.md and architecture prose are preserved. Read only the completed
20260927-contextual-ln-fit-v1/native-v2 reference_tilt-0 and reference_tilt-128
outputs and their pinned models. Model SHAs are
6c3e726397b5167836048f61f3de7753a38abe744f4151082bbf35815ad13cc3 and
d1c7d334e15607e78b3b98c4e780bf381647580bbb32caa7eb67568d8bbc61a8.
Use all eleven frozen cases, complete full audio and actual controls, recovery
60/50/50, default recovery preference and LN feedback off. No new sampling,
training, target substitution, shape filtering or support relaxation.

For every actual H query, evaluate both weight endpoints on both complete
observed histories. Reconstruct row law with the existing factual-prefix scorer
and deployed preference replay. Preserve original row times and actions in each
history, shared H times and physical support. Different histories include their
own R events, LN state, exact clocks and content; do not call this an isolated
content-memory intervention. Synthetic metadata only identifies generated row
arrays and never becomes a predictor input or a source training label.

Record expected heads, LN heads, releases, expected LN/head ratio, current
occupation, legal support and prior-two-second type use. At common H times form
F00, F10, F01, F11 (weight index first, history index second). Report direct
weight effects at each history and history effects at each weight, plus symmetric
telescoping components whose sum equals F11-F00. Keep cases and control scopes
separate. Report all H and the subset with no occupied columns in either history
and identical complete row support.

The closest analogue is a crossed response-surface comparison; no novel causal
attribution estimator is claimed. Primary diagnostic: mean absolute change in
expected LN/head ratio along the two dimensions. If the symmetric history term
exceeds the weight term by at least 2x in at least two LN cases with at least
20 eligible common-support queries each, prioritize actual reached-state/type
organization work over another count calibration. Direct effects dominating
instead favor inspecting the factual count objective. This criterion guides
research, not checkpoint acceptance; cancellation, correlated times, stochastic
history selection and support changes limit causal interpretation.

Verify the count-only invariant with the KL chain rule: at fixed history,
complete-row KL separates into count-family KL plus conditional-layout KL.
The latter should be below 1e-7 nats up to scorer arithmetic. Stop and inspect a
larger discrepancy before interpreting family-specific responsibility. Check
non-composition tensor equality, both weights' support equality on each fixed
history, row normalization, matching H query clocks and telescoping residual
below 1e-10. No probability effect is itself a BAD-pattern label.

CPU one thread; at most 900 seconds and 4 GiB task footprint. Fresh owner
artifacts/joint-audio/20260927-crossed-prefix-laws-v1. Pin script, cases, rows,
models and audio/Mel byte identities before execution. Preserve failures, no
automatic retry/resume; stop on source/input drift, nonfinite observations,
probability/invariant failure, STOP or budget. Output compact per-H evidence
and per-case/per-scope summaries. Existing Lens witnesses supply context; new
semantic claims require further views. Outcome interpretation remains REFINE
for this exploratory unaccepted Card. Full playability remains unproven.

## Completed result

Run-v1 completes all eleven cases and 15,642 common H clocks in 264.202 seconds,
maximum observed task footprint 690,537,864 bytes. Frozen plan SHA:
df593608e2370d22629794d7892f8dec6c4331d5b91894f5b42df9e0dfec8097.
Summary SHA: dfa1ce6e800ec082c7d09c23812099d04794dbde5cb10f102bddfa9d91b72dda.
Each H clock has all four crossed laws. All support, normalization, frozen
non-composition and telescoping checks pass. The maximum conditional-layout KL
is 5.8391e-14 nats, confirming the count-only update's fixed-history limit.

On shared-support/all-free queries, mean absolute history/weight LN-ratio effect
is 6.42 and 7.57 for Classic (696/703 queries), and 25.89 and 6.35 for STYX
(21/26 queries). All four exceed the diagnostic criterion. These are not
independent statistical samples or fractions of causal responsibility. Exact
clocks, cumulative row/head counts and content still differ across histories.
Blizzard has only seven/six shared-free queries and remains near saturated LN
preference; its full-query history effect is not interpreted as an isolated
memory effect.

At old fixed prefixes the refit raises expected releases per H by .00551/.00995
in Classic and .03307/.03819 in STYX. Their larger realized median holds cannot
alone establish better retention preferences: starts, reached states and the
observed hold population also change. Specific object retention remains open.

Evaluation: REFINE. Prioritize full-row decision learning and actual reached
states over another count-only calibration. Do not turn history sensitivity
into a claim that all history dependence is bad. The next bounded branch is
factual missing-observation training compared with ordinary full-R1 learning;
it keeps real physical facts and source targets, without borrowing suffixes for
generated histories. Its own owner is
2026-09-27-full-row-history-observation-learning.

Product 0882315ef23097e44e031707abd382d971b8c82c records this analysis in
docs/research/contextual_ln_count_learning.md and adds the optional R1 observation
mask. CPU/MPS observation-independence/gradient, true-BOS, factual support,
skeleton ownership, cached sampling and layout checks pass 18 tests in 9.04s.
No model or decoding policy is promoted. Note remains proposed; accepted
revision none. The original unrelated product edits remain untouched.
