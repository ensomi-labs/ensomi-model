# Conditional audio evidence in H generation

The experimental head_audio_modulation option lets a scoped control change
which encoded audio features contribute to H's base hazard. It preserves the
separate bounded history residual and keeps chord size, columns, TAP/LN type
and release identity inside R1. It is disabled by default; expressing this
interaction does not establish a playable model.

## Why the additive base is restrictive

For full-audio query $F_A(t)$ and per-field control vector $c_t$, the existing
base logit for millisecond phase $j$ is

$$
b_j(F,c)=w_j^\top(F+W_c c)+\beta_j.
$$

Its audio/control mixed partial is zero. Changing the control changes a bias
in that phase's logit, independent of the audio feature values. This is a
restriction of the base, not of the entire model: the nonlinear historical
residual still receives both inputs, and survival changes first-event
probabilities. Its recency gate is zero at BOS and decays with time since H,
so that history cannot indefinitely veto audio evidence.

The [matched release-support study](release_support_joint_learning.md)
showed that joint likelihood improvement and fewer total H did not eliminate
low-difficulty peak burden. Another H-base substitution had strong native
density effects but overshot requests, as recorded in
[audio-memory learning](audio_memory_joint_fit.md). Neither observation proves
that the additive restriction caused those failures. They motivate testing
conditional evidence selection with an unchanged comparator.

## Interaction and ownership

Add a zero-initialized control-to-audio matrix $V$:

$$
\begin{aligned}
\widetilde b_j(F,c)
&=b_j(F,c)+w_j^\top\!\left(F\odot\tanh(Vc)\right),\\
\ell_j(t)&=\widetilde b_j(F_A(t),c_t)
           +B g_t\tanh r_j(F_A(t)+W_c c_t,H_{<t}).
\end{aligned}
$$

The effective audio feature scale is $1+\tanh(Vc)$, between zero and two.
At $V=0$, the old probability law is recovered. The base now has an explicit
audio/control interaction without a fixed beat grid, section vocabulary,
head-count target or hand-state input to H. Its total logit is not bounded:
$B g_t$ bounds only the historical correction.

This adapts [FiLM's conditional feature transformation](https://arxiv.org/abs/1709.07871)
to the H base. The repository's
[R1 layout modulation](row_condition_interactions.md) is a related application
at a different decision. These analogues motivate the primitive, not its
effectiveness for musical timing or gameplay.

The current 224-feature, 336-control model adds 75,264 parameters. All control
coordinates keep their existing scope/unknown semantics. Audio is still
available for the complete song during both training and inference. The
intervention neither resets H history at scope changes nor alters publication,
H capacity, release windows or incremental LN representation.

Implementation:
[ControlledAudioModel](../../src/ensomi_model/research/controlled_audio_continuation/model.py).
The controlled_head_parts method returns the actual modified base, unchanged
residual and gate. The inherited unconditioned head_parts method should not be
used to describe a controlled query. Native generation and interval scoring
both call head_logits, so there is one deployed probability path.

The option requires bounded_head and is recorded in probability_options.
Old checkpoints omit it and load the original architecture. Importing old
tensors into the experimental model has exactly one missing zero-initialized
matrix; saved experimental checkpoints load strictly.

## What the comparison must establish

A matched study trains H modules with and without the added interaction from
the same weights and factual windows. Audio and R/R1 are frozen to avoid
changing downstream inputs through a shared encoder. This isolates H adaptation;
it does not make freezing audio a final-system recommendation.

Validation likelihood is diagnostic. From-BOS generation must demonstrate
better low-difficulty control and lower necessary peak burden while retaining
D4/D6 behavior, LN controls, style expression and scoped state continuity.
The H-only star lower bound diagnoses timing feasibility; it is neither a
sampling mask nor a substitute for continuation/player response. Source
matching, genuine ranked counterexamples and Lens review remain necessary.

Sixteen focused CPU/MPS checks pass, including identity initialization of all
three factors, learnable audio/control interaction at BOS, preserved historical
decay, unchanged R/R1 under the same factual state, native/dense H scoring and
publication partition parity, checkpoint loading and existing ownership tests.
These establish implementation properties, not a trained quality result.
