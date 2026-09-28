# Agent Note: Restore continuous segment information to R/R1

Note ID: 2026-09-28-continuous-segment-context
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 9c6d53198457d2a66c6158af1c8a431cf24c2896
Scope: Matched R/R1 learning with per-hand continuous plan context
Related: 2026-09-28-action-segment-r1; 2026-09-28-response-blindspot

## Progress, hypothesis and authority

Previous goal turn made progress: an actual planner published a25ms LN at zero
work, the scalar quadratic repair failed a6924chart comparison, rolling budgets
were fixed, and all six pilot charts failed a pinned extreme-tail regression.
All processes are terminal; the overall goal remains active and unmet. Standing
user authority permits implementation, bounded local learning and local commits.
No exact Card acceptance, note lifecycle transition, remote push or promotion
is inferred.

The segment factorization p(Y|x,g)=sum_z pi(z|g)q(Y|x,z) blocks g entirely when
K=1. K4selected only code1in85native plans. Thus the long factual history and
future-audio summary have no demonstrated useful path into the decoder. An
additive continuous code u(g) into its two FiLM layers restores that path,
without changing H ownership or replacing independent player response.

Preserve two relative-hand history views in u; averaging hands first would
remove which-hand information. The same shared network processes each hand,
with common future audio, H and controls. The categorical prior remains
reflection-invariant and unchanged. Continuous context is fixed for a private
plan and identical for R hazard and R1 mark. Exact occupancy survives cuts;
no hypothetical endpoint becomes factual. This is deterministic hierarchical
conditioning related to the earlier MusicVAE/ACT analogues, not a novelty claim.

## Experiment Card: continuous-segment-context-v1

Revision: 3
Accepted revision: none

Baseline source9c6d53198457d2a66c6158af1c8a431cf24c2896 and K1step32 SHA
9c60831599bbda301d75c28c161630e1ec2fbdb0a654149608624443a14bb7d9.
Both arms resume its identical tensors with K1. The intervention adds a zero-
initialized projection from per-hand segment summary into the decoder code.
Initial native laws must match exactly. Keep audio/H/long temporal encoding
frozen; train the same materializer paths in both arms, plus the new projection
and the already-present plan trunk through its restored likelihood gradient.
No response energy or quantity feedback enters imitation.

Matched learning:1024new original mixture draws beginning at ledger index5128,
using the existing equal-song/chart natural branch, condition-balanced branch,
and actual human-style branch. Exclude the pinned STYX/Kimi/Celestial audio.
For each visible query view choose one <=4s/control-bound segment with the
existing partition-count importance factor. Original style and style-known/
LN-hidden views share mass; hidden boundaries never become plan clocks.

Both arms additionally receive one real-BOS piece per update, drawn from the
natural-branch charts of this fixed draw set. Select the actual plan piece
containing that source's first H, preserving preceding silence and empty exact
history. Original controls and genuinely observed styles remain unchanged.
Never move human-oversampled examples into an unconditional BOS population.
This auxiliary contributes .25times its NLL per second, separately from the
four-draw mean source objective; it is explicitly a startup reweighting, not an
unbiased estimate of the original time population. Unsupported source pieces
must be recorded and replaced from the declared natural pool before fitting.

Prepare fixed factual and BOS checks from existing validation draws; these
measure learning, not musical quality. Freeze preparation outputs, scripts,
checkpoint identities and a clean intervention source before model execution.
No source suffix is attached to altered generated history. All R survival,
event and complete-row factors use the same continuous plan context.

Training bound:256updates,4general draws plus1BOS auxiliary per update, seed
281901. MPS with explicit extra, CPU2,16-update fresh-process segments carrying
complete optimizer/RNG state; lr3e-4for new decoder/plan paths,3e-5for inherited
condition/history paths and1e-3for release scale, AdamW wd1e-4,clip1. A first
16-update segment is the resource/learning profile. At most3600s total,
16GiB task footprint per worker and8GiB saved output. Stop on nonfinite values,
source/hash drift, native replay inconsistency, STOP or resource bounds.
No overwrite/restart after terminal failure. A controller resumes only verified
successful saved optimizer/RNG boundaries; it does not restart a timed-out
unknown worker. No model source changes while its pinned worker is live.

Native evaluation after64and256updates: both arms, three pinned actual4star
sources, source H diagnostic first seed at64; both seeds and separately labeled
audio-only H at256if the learning run remains valid. Raw actor is assessed
before independent guidance/acceptance. Apply the frozen40/80ms prevalence
references, report every scope, exact attack/release recurrence and work, actual
whole/scoped difficulty and amount, continuing-held TAP groups, and Lens fixed
scopes. Prior outputs are retained negative controls. Candidate selection does
not turn raw-proposal probabilities into selected-policy likelihoods.

A promising context effect requires at least50%lower40ms burden on two of three
paired source-H cases with no >10%relative worsening on the third, and no new
sustained same-finger failure. This cannot qualify a model by suppressing LN:
requested amount error<=.10, D4difficulty error<=.5, recognizable TAP/held-role
organization in both seeds, native-H quality and actual2s publication service
remain independent requirements. First-window latency includes full model audio
encoding from cached Mel; cold audio preprocessing must be measured separately
before product qualification. Merely lower NLL or better local average is not
success. Persist all failures and distinguish small-exposure uncertainty from
a rejected information-path hypothesis.

## Necessary implementation checks

Old checkpoints load unchanged. At zero projection the new path reproduces the
parent law. With nonzero projection, changing long/future context while fixing
local decoder inputs changes row/release probabilities and receives gradients
for K1. Reflection swaps hand context; compact R and full-row probabilities
agree. Native cached sampling and dense joint scoring agree across plan cuts,
including open holds. Fork/control changes preserve immutable context ownership.
Partial scoring retains the original plan condition; it must not recompute a
new context from a shorter observed future.

## Revision-two implementation and resource contract

The actual host has24GiB physical memory and10logical CPUs. Use16updates per
worker and16GiB footprint, retaining the256update/3600s total bound. Both arms
start from parent tensors with fresh matched AdamW state; subsequent worker
boundaries restore the complete optimizer and RNG states. Resetting both initial
optimizers is shared preparation, not the causal architecture difference.

Implementation adds a per-hand continuous projection to the existing two FiLM
layers, initialized at zero. Native and dense scoring share its original plan
extent. Old checkpoints keep the original law.14segment checks passed3.10s,
including CPU/MPS K1context gradients, fixed-local-input future-context effect,
reflection, actual native/dense R/R1 agreement and censored plan-prefix scoring.
The new path uses the already-existing plan trunk, with shared hand weights;
there is no extra per-row audio pass, invented release endpoint or H action head.

## Revision-three supervision ownership correction

Audit of the frozen earlier pilot data finds36of60human-branch views have no
active style field after random plan subdivision;24retain an actual field.
For example source draw5000 selects[71220,72000) after the retained annotation
ends at71220. This is not missing data: the wrong subinterval inherited the
style-selected chart distribution. The original full crop having a label does
not label every resulting piece. Natural and balanced numeric-control branches
retain their distinct sampling meaning. No causal share of all historical bad
patterns is assigned from this audit alone.

In both arms, restrict human-branch segment sampling to pieces with at least
one actually known style coordinate, including observed zero labels. Normalize
that branch's NLL over its eligible annotated duration, then divide mass across
visibility views. For uniform eligible-piece selection the weight is
1000 * eligible_piece_count / eligible_duration_ms / view_count. Other branches
retain their existing importance weights. Record removed unlabelled duration
and every exclusion in preparation. BOS auxiliaries remain natural-branch
draws only. This common recipe correction is not attributed to continuous
context in the matched comparison.

## Frozen preparation

Product4c8463a2028cc451a06aeabc302e20673522ed77 includes the context path
and supervision-owned partition helper.15segment checks pass3.26s, plus
16joint-release/guidance/frontier checks3.30s. No model process was live during
these source edits. Frozen prepare plan SHA
ae57ef344d0bd17536635cb204c05e3c5af13415f4c1936c13455f90be84f748
owns artifacts/joint-audio/20260928-continuous-segment-context-v1/prepared-v1.
Preparation validates actual segment support before fitting, records excluded
source pieces and supervises true BOS pieces only from natural draws. It also
pins the imported factual-data helper. CPU2,1200s preparation bound, exclusive
output, no overwrite or automatic restart. Model training has not started.
