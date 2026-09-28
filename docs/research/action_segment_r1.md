# An exact action-segment mixture for R1

The [ordinary-arrangement diagnosis](r1_ordinary_arrangement_redesign_zh.md)
found that the current row policy can enter an LN-dominated trajectory even on
real H times and with a persistent low-LN composition condition. This prototype
changes the unit of the shared arrangement choice. It is not a qualified model.

## Probability and information paths

For actual-time segment $W_j$, let $P_j$ be its complete committed prefix, $H$
the proposed head-time skeleton, $A$ the full audio and $c$ the visible controls.
Choose one private categorical state, then generate the whole segment:

$$
p(F_j\mid P_j,H,A,c)=
\sum_{z=1}^{K}\pi_\psi(z\mid P_j,H,A,c)
\,q_\theta(F_j\mid z,P_j,H,A,c).
$$

The four-state prototype enumerates each conditional R/R1 likelihood. Its loss
is one logsumexp after summing every row and release survival/event log factor
in the segment. The posterior is obtained exactly from those sums; there is
no target recognition network, variational approximation, or independent
mixture at each event. The one-state version supplies a matched decoder control.

The initial implementation uses at most four seconds per plan and splits at
visible control-owner boundaries. These are computation and control clocks,
not musical sections or a beat grid. A hidden LN field's old source boundaries
do not become implicit plan-reset signals. Two training visibility views may
therefore partition the same source interval differently; sampling weights
must account for their respective partitions.

The prior reads eight future audio observations from the complete encoding,
all H times inside the segment through a learned time-feature pool, head count,
duration, actual controls and incoming history/state. Longer row history reaches
this prior once per plan. No future row actions enter it. Source H is a training
condition or separately labeled diagnostic; inference uses the generated H.

The local decoder uses a short causal history, exact replay, direct audio,
future H preview and actual controls. A code modulates two nonlinear layers
that score complete four-column rows. Shared hand views preserve reflection
symmetry. This replaces the old count/layout/residual-consequence actor heads;
cardinality and all lane/type decisions remain R1 responsibilities. It does
not move them into H.

The same code and row energies determine waiting versus releasing before R's
clock is sampled, then the release subset after that clock is selected. The
virtual empty action means survival and is never appended to physical history.
Raw source learning does not include the independent response-guidance energy.

### Continuous context alongside the categorical choice

`SegmentConfig.continuous_context` adds a deterministic path from the same
incoming facts and future audio/H observations. It preserves two relative-hand
views instead of averaging their histories. A shared plan network produces
$u_j\in\mathbb R^{2\times d}$; the decoder's two FiLM layers receive
$e_{z_j}+u_j$ for every R and R1 query in that plan. Reflection swaps the two
context views. The categorical prior remains invariant and unchanged.

This path addresses a specific bottleneck: with one state, a categorical prior
cannot transmit any varying input to the decoder. It does not assert that a
deterministic summary alone learns musical organization or player response.
`with_continuous_context` copies an existing segment actor and zero-initializes
only the new projection, preserving the starting law for a matched comparison.
Old checkpoints default to the original code-only path and load unchanged.

The session computes context once at the private plan boundary. Forks share
that immutable tensor; a new plan or scoped-control cut replaces it only on
that branch. Dense source scoring reconstructs it from the true incoming
prefix. `score_segment(..., plan_end_ms=...)` retains the original private end
when scoring an observed prefix beginning at that same plan boundary. Shorter
observation coverage must not silently change its future-audio condition or
close an open hold. Scoring from an arbitrary interior start is not supplied
by this helper; it requires retaining the earlier plan and local history.

## State across plan boundaries

Only the local learned content cache resets, using TRUNCATED when a real prefix
exists. Exact occupancy, attack/release clocks, pressure state and the longer
prior history persist. A boundary does not create a row or close a hold.

Each active LN also retains its actual birth-row geometry and its original
audio observation. These can distinguish an isolated held role from a collective
LN entry after the local cache resets. They are factual birth context, not
the persistence of its older latent code or a predicted future endpoint.
Whether this context suffices for stable cross-boundary held roles remains an
empirical question; the implementation does not claim to encode complete intent.

`SegmentSession` owns its plan RNG, current code, expiry and birth contexts.
Forks share frozen weights and immutable caches, while owning their RNG and
event log. A midstream control change truncates the affected future plan at
the new scope; published rows remain untouched. A selected code otherwise
persists through partial publication and empty observed time.

`GuidedSegmentSession` adds the existing independent action-response energy.
`ResponsePlanner` can inspect its actual private futures and retain rolling
committed-work budgets. Neither the scalar code nor source likelihood is itself
a player-response frontier. The independent response's discrimination remains
an empirical qualification requirement.

## Learning and qualification

The initial comparison freezes the inherited full-audio encoder, H policy and
long prior-history encoder. Train the new decoder/prior, short local history,
R1 condition paths and release flow scale. A fixed full-audio encoding is valid
only while its producing parameters are frozen. Shared-audio or H updates need
their corresponding likelihood factors and refreshed encodings.

The source teacher supplies genuine complete prefixes and partial futures with
their true open-hold state. A source crop never forces an endpoint. Both original
and style-known/LN-hidden conditions use the same source sampling mass, with
correct importance weights for their visible segment partitions. Unannotated
style-balanced data must not silently become unconditional supervision.

`training_intervals(..., require_style=True)` retains only pieces with an
actually observed style field, including explicit zero labels. A human-selected
eight-second crop can overlap an annotation while its sampled shorter piece
falls entirely outside it: 36 of 60 human-branch views in the initial pilot did
so. That piece must not inherit the expert sampling weight as unlabelled data.
Condition the human branch on its eligible annotated duration and keep the
corresponding importance factor. Natural-population draws can retain unlabelled
pieces; genuine BOS auxiliaries must come from that natural branch.

Measure posterior-versus-prior information and aggregate code use separately:
uniform use can mean useful diverse plans or a completely ignored code. Native
prior samples, including complete BOS rollouts, are the quality evidence.
Forced-code samples and posterior reconstruction are diagnostics only.

The main organization comparisons use actual ranked charts near four stars:
ordinary TAP, sustained LN with other-finger TAP groups, and their transitions.
Keep generated H and source-H diagnostics distinct; the older 5.873-star
Zenithfall source is not a four-star feasibility reference. Check multiple seeds,
scoped difficulty/style/amount, jack and LN-transition pressure, recognizable
TAP organization, musical variation and actual publication latency. A lower
NLL, longer median LN, or lower LN fraction cannot independently qualify a model.

The design is related to shared subsequence variables in
[MusicVAE](https://proceedings.mlr.press/v80/roberts18a.html) and action-chunk
prediction in [ACT](https://tonyzhaozh.github.io/aloha/). It preserves exact
discrete rows and cross-boundary physical state; it does not copy independent
subsequence execution or averaging of continuous action predictions.

## Bounded pilot outcome

The matched one-state/four-state pilot trained 32 updates on 158 factual
segments: 3,960 target rows and 160,831 native release-risk clocks. Only one
segment supervised the actual first source row; the others began from factual
prefixes. Resets marked TRUNCATED are not substitute BOS examples. This exposure
does not establish decoder convergence or adequate startup learning.

Six complete source-H generations used three actual approximately four-star
sources, with one seed per arm and source: STYX, Kimi and Celestial. These audio
bytes were excluded from this pilot, not declared unseen by the inherited
backbone. Requests used D4 and each source's whole-chart LN fraction. All outputs
remain unqualified. They contain 21–25 ms holds, generated LN fractions cluster
around .25–.30 despite source requests .485/.131/.075, and Celestial outputs
measure approximately 5.05 stars. Some continuing held roles appear, but their
companion TAP organization and short-hold behavior still fail inspection.

All 85 four-state prior selections used code1. The longer history and future
audio observations reach the decoder through that code, so this result does
not show meaningful segment-plan conditioning. The one-state prior is constant
by construction; it cannot transmit its varying inputs through its categorical
output. The decoder still receives direct current full-audio features and H
preview. A deterministic plan-context path and adequate source learning are
distinct follow-up hypotheses; more codes or forced entropy are not justified
by this result.

The [acceptance diagnosis](r1_short_hold_acceptance_zh.md) separately checks
whether the independent frontier catches these proposals. Raw pilot generation
did not use that planner. A later actual planner execution does publish a 25 ms
hold with zero scored work, establishing a separate selection failure.

Implementation:
[model](../../src/ensomi_model/research/segment_audio_continuation/model.py),
[segment likelihood](../../src/ensomi_model/research/segment_audio_continuation/segments.py),
[generation](../../src/ensomi_model/research/segment_audio_continuation/generation.py).
