# Learned LN allocation from scoped progress

The broader R1 fit improves some Stream routing while overshooting explicit LN
fractions. This optional research module tests whether factual allocation
progress helps retain the requested amount without requiring every prefix to
match the final fraction. It is not a player-response model or a qualified
generation improvement.

H still owns head times. R1 owns complete rows. The new 64-hidden MLP reads the
current full-audio query, R1 context, H preview, controls and scoped observations,
and returns one conditional LN-count log-odds shift. A matched context arm has
the same architecture with accounting observations zeroed. Both are disabled
unless explicitly selected in checkpoint probability options.

## Factual state and control ownership

Each LN-bearing control span has a stable identity, even when another span
uses the same numeric fraction. Partial style/difficulty overrides do not create
a new LN request. The last active LN-bearing assignment is the effective owner.
An interrupted request resumes its existing counters.

The immutable state keeps two pairs of counts for every announced LN request:

- Declared counts include all heads and LN starts inside its original interval,
  including intervals controlled by an override.
- Owned counts include only heads and LN starts while this request was the
  effective LN owner.

The current owner's readout contains asinh-scaled head/LN counts, observed
fraction and a nonempty flag for both pairs. It also contains elapsed owned time
as a fraction of total announced owned time and asinh-scaled remaining owned
seconds. Count scale is 32 heads; time scale is one second. The ordinary control
tensor already contains the original interval clocks and missingness.

These are descriptions of what happened and what control program is announced.
Neither count pair imposes a prefix quota, and the module does not redefine how
whole or interrupted scopes are evaluated. It may learn that local pure TAP
or pure LN passages are compatible with the requested full-range result.

Future control announcements are unavailable until received. A later update can
change remaining owned time but never rewrite counters or committed rows.
Factual scoring must use the control program available at the query; replaying
an earlier query with a subsequently announced override is a different input.
Teacher examples whose complete schedule is announced initially can use it
throughout. Full audio is available in both cases.

## Probability law

Let $q(a)$ be the neural row law after local frontier normalization,
$\gamma(a)=(n_H(a),n_R(a))$, and let $b_\psi$ be the new readout. For a known LN
request the module returns

$$
\widetilde q(a)
=q(\gamma(a))
\frac{q(a\mid\gamma(a))\exp[b_\psi n_{LN}(a)]}
{\mathbb E_{q(\cdot\mid\gamma(a))}\exp[b_\psi n_{LN}]}.
$$

This preserves neural head/release-count mass and relative layout odds inside
each full head/LN/release-count family. Recovery preference follows this layer
and can change final deployed family mass through layout/cost correlations.
There is no assertion that the whole generated trajectory preserves counts:
new LN choices change occupation and subsequent histories.

The final MLP projection is initialized to zero. Unknown LN requests return the
original neural probabilities exactly, regardless of learned weights. Thus an
adapter-only fit cannot change an unknown-LN native trajectory with identical
base tensors, controls and random streams. This is particularly relevant to the
existing Stream-only gains. The layer neither masks patterns nor supplies an
independent LN-coordination judgment.

## Source scoring and fixed-base learning

Checkpoint option `scope_allocation` is `none` by default and omitted from old
probability metadata. `context` and `progress` instantiate the matched arms.
`ControlledSession` maintains the immutable accounting state, including private
forks and live control changes.

Training supplies `allocation_features` through `score_interval`'s
`history_options` callback for row queries. Use
`features_before_rows(actual_rows, announced_controls, query_times)`: current
target rows and future LN endpoints never enter their own features. The helper
preserves query order and supports repeated times.

The initial comparison freezes the entire inherited model and trains only this
MLP. Its pre-layer probabilities, audio/context/preview/control inputs and
recovery preference are therefore constant on a factual source query and may
be cached. The adapter and its normalization must be recomputed at current
weights. This shortcut becomes invalid if any upstream weights or the factual
history change. Recover the actual deployed row law after the learned layer;
do not optimize an unnormalized tilt as though it were a likelihood.

## What the current checks establish

Two legal prefixes can have identical exact replay, head/row totals and more
than 511 common final rows, while their earlier scope LN-start counts differ
by 64. This supplies a concrete observability gap for the existing finite
history. It does not establish that the gap caused native quantity drift.

Focused checks cover this collision, override/resumption identities, strict
pre-query observations, immutable private state, zero initialization, unknown
condition identity, CPU/MPS gradients, neural family invariance, checkpoint
round-tripping and native versus partitioned factual probability reconstruction.
Quality still requires the matched fit, actual native scope results and Lens
inspection. Larger R1/audio learning and independent continuation response
remain separate needs described in the
[formal diagnosis](native_pattern_failure_analysis_zh.md).
