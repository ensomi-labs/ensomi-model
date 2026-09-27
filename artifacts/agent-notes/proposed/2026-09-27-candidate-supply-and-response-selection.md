# Agent Note: Suitable continuation supply versus response selection

Note ID: 2026-09-27-candidate-supply-and-response-selection
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 9f5ed1d9d5d1a596578f8e4e242cf49e175d2bb8
Scope: Actual reached-prefix candidate sets, LN organization, sustained pressure and current frontier contribution
Related: 2026-09-27-ln-continuation-preference-diagnosis, 2026-09-27-player-response-frontier, 2026-09-27-playability-regression-evaluation

## Progress and current question

The previous goal turn is progress: committed a self-contained formal diagnosis,
completed 16 source-H interventions and added reusable LN relationship evaluation.
Current product HEAD and note HEAD are verified, and no relevant Python job is
live. Full playability remains unproven; no checkpoint is promoted.

The next decision is whether suitable native R/R1 alternatives are available
under the same reached state and H, or whether existing response selection misses
them. Long-jack relief already exists in some proposal sets, so one reproduced
Stream failure is a positive control, not a new claim that attack-only planning
solves quality. The primary unknown is LN organization under identical H and
history. Descriptive duration regularity is not an automatic good-pattern label.

## Experiment Card: native-candidate-supply-v1

Revision: 2
Accepted revision: none
Execution authority: the user's standing goal authorizes local research,
experiments, implementation and commits. No inferred acceptance or remote push.

Baseline source: 9f5ed1d9d5d1a596578f8e4e242cf49e175d2bb8, executable source clean;
unrelated AGENTS.md and untracked architecture prose remain untouched. Actor128
SHA 364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3 and memory384
SHA 7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81 retain their
saved recovery and default deployed LN/recovery preferences. No weight change.

Six native reached states:

- Actor and memory Classic seed 273110, boundary 88588 ms.
- Actor and memory Blizzard seed 273130, boundary 40341 ms.
- Memory STYX seed 273120, boundary 1799 ms.
- Actor Zenithfall prominent Stream seed 271201, boundary 41093 ms.

Each is reached by current-policy BOS generation with full original audio and
controls. Verify the complete prefix and unchanged candidate-zero future against
the pinned existing native rows. Do not restore a fabricated prefix or borrow a
source suffix. Stop and investigate any mismatch. H is naturally identical
across candidates because only R/R1 retry RNGs change.

Generate 32 private futures per state, 192 total, recording snapshots at 2/4/8/16
seconds. Candidate zero preserves current RNGs; later retry seeds use the current
ResponsePlanner's NumPy RNG convention, initialized by the native seed XOR
0x6F17. The first four candidates therefore expose the existing four-second
attack-only selection rule, without stopping further diagnostic sampling when
cost first becomes zero. This is a bounded diagnostic candidate set, not a new
sampler or selected-policy training dataset.

Record separate horizon pressure integrals, explicit continuation occupation,
heads/LN/releases, recovery, LN duration and H-span relationships, and retained
history. Record native row log probability and the local frontier2 contribution
by observing its unchanged score output; the row distribution with that term
subtracted is a same-state diagnostic, not an executed policy. Candidate-zero
row equality supplies an instrumentation check. Do not equate cumulative learned
energy with calibrated future response or compare different controls by pooling.

Use the existing corpus envelope only for its existing attack response. Calibration
SHA 0f614ec85d08a54b72f9fb15ab57f5762a822965ca6589003b1aeec72c184f1a. Report the
first-four/four-second choice, candidate-zero and longer-horizon alternatives.
The numeric comparison selects inspection contexts; it cannot automatically
label overall LN quality. Review candidate zero, the existing planner choice
when different, pressure-best alternatives and contrasting LN relationships.
Keep review-based improvement, unresolved tradeoffs and unreviewed candidates
distinct. Do not report a complete semantic coverage rate from a reviewed subset.

Private horizons retain open holds. For Lens only, continue a candidate for at
most 12 additional seconds until the real tails of holds started by 16 seconds
are resolved. A review projection may include those genuine tails but no invented
crop-end closure. It is not a complete-song export or input to earlier response
scores. If tails remain unresolved, keep censoring and mark that view unavailable.

Budget: CPU one thread, 2400 seconds overall, 360 per reached-state group,
4 GiB peak process RSS on this CPU run. Fresh owner
artifacts/joint-audio/20260927-candidate-supply-v1, no overwrite, training or
automatic retry. Stop on STOP, source/hash drift, nonfinite probability,
prefix/candidate-zero/H mismatch, illegal replay or resource bound. Preserve
every attempted candidate and any failure; sample reports are not a production
latency benchmark. The named realtime benchmark worktree remains unchanged.

Interpretation: if reviewed candidates improve the declared defect while the
current selector retains a worse one, strengthen response/selection work. If
the candidate set preserves the defect, refine proposal/representation/recipe
work rather than tightening shape masks. A finite sample cannot prove absence
of suitable support. Cross-horizon deterioration indicates a horizon/state
problem even if a four-second score improves. All branches still require full
native qualification before any deployment or checkpoint promotion.

## Execution adapter correction

Frozen plan SHA 1f4cd14092f59189d1117235eabc5aaf696cd54ba1506e1fe2a5bdc1dfc06ca3.
First attempt 57128 is terminal/failed during the first candidate's optional
offline difficulty parsing: the minimal source-free header omits AudioFilename
until the complete export wrapper supplies it. The private review projection
uses the lower-level exporter, so it must explicitly reference the original
audio. The native prefix, candidate zero, physical validation and projection
row equivalence had already passed before this metadata failure.

Preserve run-v1 and failure.json; correct only the projection's General header
and use fresh run-v2. No model, seed, case, response, candidate budget or endpoint
changes. Do not describe this failed adapter attempt as complete evaluation.

## Completed exploratory result: 192 native candidate futures

Accepted revision remains none. The standing research authority permitted the
run; completion is not Card acceptance, a SUPPORTED decision or promotion.
Run-v2 completes six contexts and 192 futures in 432.593369 seconds, peak RSS
0.783875 GiB, CPU one thread. Every reached prefix, unchanged candidate zero
through 16 seconds, shared H and exported/reparsed review projection passes its
recorded parity checks. Source remains the frozen product
9f5ed1d9d5d1a596578f8e4e242cf49e175d2bb8; no model weights changed.

Each of the five LN contexts has zero four-second attack cost for its first
four candidates. The existing planner would stop at candidate zero in all five.
Across their 160 candidates, 159 have zero 16-second attack excess; the exception
is actor Classic candidate 7 at 0.0004418993291761148. This corrects an early
progress inference that all LN future costs were zero. It is evidence that
this response channel does not distinguish most LN alternatives, not evidence
that those alternatives are playable.

Eight-second LN-head min/median/max and number of candidates whose median H span
equals one are: actor Classic 3/23/32 and 16/32; memory Classic 18/36/42 and
30/32; actor Blizzard 39/47/54 and 18/32; memory Blizzard 35/57/64 and 21/32;
memory STYX 19/27/31 and 29/32. These are separate control/context populations.
The one-H preference often persists despite candidate variation; this does not
prove good support is absent or that a one-H hold is inherently bad.

The Stream positive control selects candidate 2 from the first four. Candidate
0 has J4/J16 0.006502976/0.014369201, candidate 2 has 0/0.000867, candidate 3
has 0.0000281/0.0000281. Four of 32 candidates have zero J16. These are
un-replanned proposal tails; a real planner replans after two seconds, so this
is not a measured closed-loop deployment regret.

Same-query removal and renormalization of the frontier2 energy shows expected
release count increasing when g is added: actor Classic .7443 to .8690, memory
Classic .8059 to .8664, actor Blizzard .6885 to .7638, memory Blizzard .6957 to
.8428, memory STYX .7536 to .8125. Existing release preference is substantial
before g. This local contribution analysis does not execute an ablated policy
and cannot establish that disabling g improves native organization.

### Same-support history analysis

The behavior-neutral analysis refinement groups all-free H queries by exact H
time and complete row support mask. Insufficient future-H horizon is excluded.
The neural LN ratio is E[LN heads]/E[heads] after g but before recovery preference
and amount feedback. Exact clocks and content history can still differ.

At actor Classic 100474 ms, candidates 27 and 6 have the same 80-row support,
but ratios .0097176 and .896589, with feedback +.258896 and -.979699. At memory
Classic 91522 ms, candidates 6 and 28 similarly give .0179256 and .947934, with
feedback +2 and +.320255. Audio, controls, H time, occupancy and support match.
The feedback favors the low-LN history more, so its sign does not explain this
neural preference difference. This is not an isolated intervention on memory
and does not prove an attractor, semantic badness or a unique hidden-path cause.

### Independent visual record and limits

All 24 rendered Classic pages were viewed: actor 0/4/23 and memory 0/8/19, four
pages each over [88588,96588). The retained agent observations are now stored in
lens-classic-v1/review.json. They are not human annotation, listening or player
trial evidence. Other contexts/candidates lack equivalent review.

The longer-LN alternatives shift or extend TAP/LN blocks. Memory candidate 19
has longer relations and more late mixing, but at the same 114 heads per eight
seconds it increases LN heads from 27 to 41 and any-held fraction from .31625
to .63375. No overall superiority or complete semantic coverage rate is
assigned. Lower duration variation is not treated as a quality label.

Result identities, all under artifacts/joint-audio/20260927-candidate-supply-v1:

- run-v2/cases.json: d7013ab48171cc5a2edcbdfc6f5d7a23d9c8869a072e9180b5ca90fe5a1041df.
- analysis-v1/cases.json: d1971e102d6082f4d94ecb34a823c8ef711c777d44231d886f0f1bb4ef7ad29b.
- support-relation-v1/cases.json: bc42317941815e9b4cd40fd7a5a31d352582b278aeb2e778812bb03312feb28a.
- lens-classic-v1/review.json: c3f6258e5dec05bb0a10f2bfb3b2c55d0794c402d9830aa63c8c43ce9bb5175a.

Evaluation: REFINE. Preserve this candidate set for independent LN/coordination
comparisons; broader candidate search alone does not fill the missing response
semantics. Improve proposal conditioning/coverage and independent response
learning as distinct, possibly coupled interventions. Product document
docs/research/native_pattern_failure_analysis_zh.md at
a96049fb1d734e8dbf0d8cb2fda3f4eb43169562 contains the self-contained evidence,
formal invariance of attack-only cost to release changes, attribution limits
and further discriminating comparisons. No new fit or model promotion follows
from this result log. The contextual LN-count prototype has its separate owner,
2026-09-27-contextual-ln-count-conditioning; matched learning remains pending.
