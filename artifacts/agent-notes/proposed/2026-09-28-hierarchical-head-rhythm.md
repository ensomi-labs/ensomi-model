# Agent Note: Shared rhythmic coordinates for H

Note ID: 2026-09-28-hierarchical-head-rhythm
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 965d6702640a9dc3d6331ddd776daee4e7341d4b plus scoped observer/document implementation
Scope: Primary subdivision, ornament structure and end-to-end audio-to-chart rhythm
Related: 2026-09-28-rh-r1-fragmentation-repair, 2026-09-28-ordinary-fourstar-rhythm-and-holds

## User direction and implementation boundary

The user adds that difficulty acts on skeleton organization conditional on
local musical speed. Tech may retain a coarse main layer with finer ornament.
An accurate hard BPM/phase tracker is not necessarily the right intermediate;
training and inference must stay full-audio-to-chart. No oracle source timing
input may be supplied only in training. R1 retains chord size, columns, TAP/LN
and release responsibility, with its own direct audio condition.

Explicit ongoing research/code/training authority permits these changes, but
does not constitute acceptance of an Experiment Card or adoption of the model.
No new subagent was spawned. The separately authorized ranked corpus agent had
already finished. Product work remains in the release-calibration worktree;
the main baseline supervisor's executable tree remains untouched.

## Current mechanism and competing explanations

At965d670 H has a full-audio affine base with multiplicative control modulation
and a bounded history residual, bound4 and elapsed decay1000ms. The history is
a finite temporal network over previous H gaps. Its explicit clocks are age
since last H and absolute time. A main period/phase or ornament-versus-main
advance is not an explicit state.

This does not prove the network cannot infer rhythm. It motivates a structural
hypothesis: local insertion and sampled timing errors can change the same
recursive reference used for main rhythm. The elapsed decay also removes
useful timing memory along with historical veto after long gaps. Full-song
audio already has fine CNN and500ms-cell bidirectional attention; do not claim
attention is absent.

Alternative explanations remain audio feature resolution/phase uncertainty,
joint training allocation, generated-prefix exposure and source distribution.
The16/80-update release changes do not isolate these explanations.
Their short-LN improvement is not a successful H repair.

## Chart-only exploratory probe

Owner: artifacts/joint-audio/20260928-ln-fragmentation-repair-v1.
Frozen plan8bec90b7435097adace2b6cc36b1024854d21d182a2bd8593fc1cbd110a27ff8;
script rhythm_probe.py, result rhythm-probe-result.json.
21fixed contexts:9previously inspected source sections,3matched source
contexts,3parent2048,3pilot16and3baseline4096. No new native sampling.
The CPU observation completed in.29633s within120s bound.

Distinct H only. Up to64time-spread observed anchors propose periods from
index lags1/2/4/8 divided by1..16, rounded.001ms, retaining40–2000ms.
At each candidate unit and observed phase anchor, count events within3ms of
the nearest lattice position. Report maximum coverage and the coarsest
candidate reaching50/80/95%. Minimum12distinct H; insufficient cases stay
unobserved. It reads no audio, BPM or redline, and fits within its observed
scope. This is not an extrapolation or calibrated quality threshold.

Maximum coverage in matched fixed STYX/Blizzard/Zenithfall contexts:
source.9231/1/.9762; parent2048.3091/.3636/.3696;
pilot16.4359/.5161/.4583; unchanged baseline4096.3778/.4222/.3053.
Source coarsest80%units≈122/176.667/65.333ms.
Among9ordinary source contexts,8have enough events and maxcoverage.9286–1;
Arakajime has10events and remains unobserved.
TellMe has187.667ms covering.9118 and93.833ms covering1;
StarTrip202.833ms/.8571 versus101.429ms/1.
This is a concrete main-plus-finer relationship recoverable without oracle beats.

Finite candidate/anchor search, selected contexts, differing counts and
in-scope fitting all limit interpretation. Tech, tempo changes and intentional
displacement can lower this coordinate without a quality failure.
Do not apply the scalar as a hard generation rule.

## Reusable observer

Implemented gameplay_evaluation/rhythm_lattice.py with lattice_coverage and
head_lattice(trace,scope). It keeps declared scope boundaries, counts chords
once, ignores pure releases and returns missing observations explicitly.
Candidate temporaries are bounded; no generator or qualification threshold
changes. It is intentionally not automatically fitted over whole songs.

Six focused tests pass: main layer plus sparse subdivisions, same count with
different phase organization, nonbinary native rounding, translation/chord
invariance, missing observations and H-only scope selection.
The library reproduces all measured fields of all21original contexts exactly;
receipt rhythm-observer-transfer.json.

## Architectural direction, not an implemented model

The self-contained design is docs/research/audio_rhythm_hierarchy_zh.md.
Separate a latent monotone musical coordinate phi from arrangement subdivision
d; actual unit is1/(d phi_dot). Multiplying phi and inversely scaling d can be
unidentifiable, so actual output geometry matters more than a unique BPM label.
Difficulty controls main layer, optional slots and ornament mixture jointly;
it must not become only a global rate shift or decide chord counts for R1.

Prefer a plan that persists across several events. Coarse/main slots and
relative-position ornament slots can be omitted; an emitted ornament does not
reset the main phase. Each slot emits at most one H, avoiding repeated Poisson
events around one narrow phase peak. For optional slot probability q and
normalized native-time mass f, survival hazard is q f(k)/(1-q F(k-1)).
Keep free-time support for genuinely non-lattice events.

Mixture over whole local trajectories is essential. Per-millisecond independent
averaging of plan hazards can mix incompatible phases within one trajectory.
A filtering implementation must update plan belief after both hits and survival;
publication cuts must not resample the plan. Neighboring plans need continuity
through actual H history, not independent fixed-window restarts.

Finite-hypothesis marginal likelihood can train against note placement without
BPM labels. If a posterior proposal sees the chart, treat it as target-side
inference with a prior-matching term, not an oracle decoder input; evaluate
prior native samples, not only reconstruction. Full audio and R1 remain jointly
trained. R1 may consume the chosen rhythmic reference but owns actual row/LN
decisions. Exact slot/free merging, collisions and alignment marginalization
remain unresolved before implementation; do not claim an existing exact model.

Closest analogues:
- https://arxiv.org/abs/1106.4863: joint continuous tempo/discrete position
  uncertainty; our target is controlled choreography, not transcription.
- https://arxiv.org/abs/1404.2314: ornament states distinct from ordinary score
  progress; no piano labels or known-score input transfers to our model.
These are structural adaptations, not an established novelty claim.

## Research recommendation

REFINE toward one bounded finite-plan/optional-slot implementation. The next
Card must fix candidate proposal from full audio, slot/free probability and
alignment approximation, temporal memory and scope-revision semantics, and
matched data/compute before a model run. Do not launch a broad architecture
grid. Compare primary-unit persistence and post-ornament return at matched
audio/control, then native difficulty/style/LN/pressure/runtime. NLL remains
an optimization proxy; no passing reconstruction or rhythm coordinate can
replace actual generated-chart review.
