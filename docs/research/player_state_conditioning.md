# Causal player-state input and short-future R1 learning

The [sustained-response planner](sustained_response_planning.md) finds less
overloaded continuations in the existing R1 proposal distribution. The next
learning question is whether R1 can prefer such continuations itself, while
retaining the organization represented by genuine chart annotations. An optional
player-state input and a scorer for open-ended private continuations support
that comparison. They do not change the V3 formulation or define its final
response specification.

## Information supplied to R1

[`PlayerCondition`](../../src/ensomi_model/research/player_response/conditioning.py)
reads 24 deterministic observations per column:

- Attack and release rates over .5/1/2/4/8/16 seconds, encoded as `asinh(Hz/4)`.
- Held-time fractions over those same windows.
- The previous two complete attack masks, their ages in seconds encoded with
  `asinh`, and availability bits.

These observations come from committed rows and elapsed time. A TAP and an LN
press each count as an attack; a release remains separate. No future LN endpoint,
difficulty request or style label enters the observations. Open LN origins survive
the 32-second recent-event buffer. Advancing through no-row time preserves holding
and ages the attack/release history. Control changes do not reset this physical
state; the policy's separate LN-amount feedback has its own control-episode rules.

The four columns are arranged in the model's existing two mirrored hand
coordinate systems. One shared, zero-initialized linear map projects each
96-dimensional vector into R1's 128-dimensional context, adding 12,288 parameters.
That context informs complete-row layout, R1-owned count composition and the
candidate-consequence residual. Existing direct audio, exact clocks, H preview,
controls and learned row history remain available.

Zero initialization reproduces the old probability law. The optional
`player_state` probability option owns the new checkpoint parameters; old
checkpoints leave it disabled. The tested model grows from 4,600,369 to 4,612,657
parameters. This representation preserves declared observations, not every
coordination distinction or an inferred physiological fatigue law.

The added input goes only to R1. H keeps its timing/history/audio/control inputs.
R retains its timing history, audio, controls, H preview and LN state. R1's actual
choices still change subsequent exact occupancy and execution-feasibility bounds;
its learned content encoder is not added to the skeleton factors.

## A private future need not close its holds

Earlier outcome-learning adapters constructed a complete `SourceChart` before
scoring a short generated interval. That required terminally closed objects and
often generated the rest of the song even when the outcome needed only a few
seconds. A local response endpoint is not the true audio endpoint.

[`collate_row_trace` and `score_row_trace`](../../src/ensomi_model/research/controlled_audio_continuation/trace.py)
take actual complete prefix rows, a proposed continuation, the complete timing-only
H plan, controls, real audio duration and a full-song audio encoding. They score
rows in a half-open native interval `[a,b)` without requiring later releases.
Its response counterpart is the continuation `(a-1,b-1]`, including elapsed time
after its last row. A true audio-end action still obeys terminal occupancy rules.

Each row query uses its own pre-action replay and content history. Future rows
cannot supply features to an earlier query. Full H timing preview is retained
even when it extends beyond the response endpoint; truncating it would change
R1 support and likelihood. This does not provide future materialized columns,
counts or hold endpoints. Complete source audio remains available on both the
training and inference paths.

The scorer reconstructs the native recovery and LN-amount preferences using the
actual prefix. Their arithmetic runs in differentiable CPU float64, matching
native sampling, while gradients return to the MPS model. The trace contains
raw rows and deterministic observations rather than stale learned caches.
Neural history is recomputed under current weights. Full-audio encodings can be
reused only because this experiment freezes the audio encoder.

These are native row probabilities. They are not probabilities of the previous
planner's selected trajectories, whose selection step changes the distribution.

## Source imitation and generated-response supervision

The comparison begins from the reported actor-128 checkpoint. Both arms use the
new zero-initialized input and identical genuine source draws. One learns only
source row likelihood; the other adds expected sustained-response cost on sampled
four-second continuations. Audio/H/R weights remain frozen. Existing R1 parameters
and the new projection are trainable.

Factual examples retain their own source history and actual actions. Difficulty
and LN fraction are defined on the exact training scope; outside it, whole-chart
conditions remain. Human style claims keep their original scopes and known/unknown
status. This differs from conditioning a small source window only on a whole-map
LN fraction: both are meaningful conditions, but the local construction directly
trains the scoped-control interface.

The generated branch uses a fixed replay bank of actual baseline-generated
prefixes. It samples new futures from the current R1, on each bank's unchanged H
times and controls. It never attaches old source suffix labels to a changed
history. Requested style is a policy condition, not an observed style label for
the generated result.

For a fixed replay prefix $h$, let $J(h,Y)$ be the corpus-reference sustained
excess integral over a four-second future, and let $q_\theta$ denote the native
R1 row factors along that future. The added objective is

$$
100\,\mathbb E_{h\sim B}\mathbb E_{Y\mid h}
\left[\frac{J(h,Y)}{4\ \mathrm{s}}\right].
$$

Three independent continuations estimate its score-function gradient. Each uses
the other two costs as an action-independent baseline. Prefix-only costs shared
by all continuations cancel. An empty future with no R1 decisions contributes no
R1 score gradient, while elapsed time still contributes to its response.

The release process remains state dependent. For a fixed materialized trace,
however, its parameters, exact state transitions and feasibility calculations
have no direct dependence on the updated R1 parameters. The R1 row score terms
therefore supply the parameter-dependent part of the sampled-future score.
This does not assert that the release probabilities are constant across different
futures, or that row scores alone are the complete absolute trajectory probability.

This optimizes continuations from the declared replay distribution. It does not
differentiate through how the old policy generated that fixed bank, nor establish
full-song on-policy optimality. Native generation and Lens inspection remain
necessary qualifications. Sustained attacks are only one response channel;
holding, coordination, semantic style and H breathing remain independent concerns.

## Implemented checks and bounded evidence

Eighteen focused tests cover zero-initialized compatibility, H/R separation,
causal observation gathering, mirrored hand coordinates, checkpoint loading,
native/source probability agreement, and open-ended trace probabilities and
gradients on CPU and MPS. Private trace scoring also agrees across interval
partitions, including empty intervals. An initial timestamp-type mismatch at the
exact-replay call was corrected before these checks passed.

The replay bank contains 20 distinct TRAIN song groups/audio identities, excluding
the existing reserved native/control/style songs and their groups. It covers five
requested style families and nominal difficulty 2.5/3.5/4.5/5.5, with 69 high-excess
and 80 ordinary contexts. The selected Stream and LN songs have no positive
baseline excess; the hard contexts come from Jack/Tech/Trill requests. This is
targeted training coverage, not an estimate of population failure frequency.

An eight-update integration run completes in 37.76 seconds on the M5/24 GiB Mac.
Five of 24 sampled futures have positive cost, two updates receive nonzero response
gradients, and eight futures end with active holds. Maximum sampled/rescored row
log-probability discrepancy is $1.89\times10^{-5}$. Frozen audio/H/R tensors are
unchanged; peak sampled footprint is 3.03 GiB. The smoke weights are discarded.

One smoke update has response-gradient norm 283.5 versus source-gradient norm
20.8 before clipping; another has .79 versus 42.7. A source likelihood term alone
does not prove semantic preservation when the added objective dominates an update.
The implementation and learning signal are verified; the main fitted checkpoints
and native/style qualification are not established by this smoke result.

## Evidence identity

Implementation: `94d0b082282ae886709c72ffdc885b2f4e045252`.
Initial checkpoint: `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3`.
Study owner: `20260927-player-state-r1-learning-v1`.
Source panel: `7f8d56c8aeb3b5e33dd3dfe3d993a7311e97d9bea1c983ee2bd0a76d991d9c10`.
Replay bank records: `53ed0a0bf953439ebfbe83a4cf6eff8c203d19232c9b102f7c94428a0f5381d1`.
Factual draw order: `348a344dcb9245e474d6cd3fdc4320ad2773d908bc6ed4e48ebcfd7dcefd0524`.
Smoke checkpoint: `2a4c7ddd038c47f293afb8c67d0084cf6abcbca0c0a183616f8197c592cc2d64`.
The bank, source pool, calibration, weights and run snapshots are local research
assets; the code, input semantics and bounded evidence are preserved here.
