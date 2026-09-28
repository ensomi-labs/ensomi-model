# Joint proposal learning with direct scoped conditions

The experimental `ln_conditioning="direct"` mode makes R1 learn its LN-count
preference from the actual scoped request without an additional analytic
log-odds tilt. It supports a clean joint audio/H/R/R1 learning comparison;
no playable-quality improvement is established by introducing the option.
The [coordination investigation](coordination_frontier_hypotheses_zh.md)
explains why proposal fidelity and independent continuation responses both
remain necessary.

## Probability law

Within a head/release-count family, let $l$ count new LNs, $u$ be factual row
context, $\rho$ the known amount request and $\rho_0$ the reference fraction.
The three supported modes have the following type scores:

$$
\begin{aligned}
\ell_l^{\rm reference}&=f_l(u,\rho_0)
 +l[\operatorname{logit}\rho-\operatorname{logit}\rho_0],\\
\ell_l^{\rm contextual}&=f_l(u,\rho)
 +l[\operatorname{logit}\rho-\operatorname{logit}\rho_0],\\
\ell_l^{\rm direct}&=f_l(u,\rho).
\end{aligned}
$$

Unknown requests retain their separate known-bit encoding and receive no
analytic shift. Direct mode is independent of the stored reference fraction.
It still permits an explicit caller-provided `ln_shift` as a separate policy;
the clean learning comparison sets that shift to zero and disables LN amount
feedback and scope allocation.

When no count prior is enabled, direct mode's count-family composition reduces
to the learned normalized distribution over all supported count marks.
Conditional layout normalization and the actor's complete-row consequence
energy follow as before. R1 retains every cardinality, column, TAP/LN and
release-subset decision. H/R support, native timing, committed history and
incremental LN publication do not change.

Reference tilt remains the checkpoint default. Both older modes retain their
existing code paths; the explicit direct option is persisted in
`probability_options` and uses the existing strict loader. It adds no tensors.

This is an ordinary conditional probability parameterization, not a new demand
model. Removing the analytic tilt does not itself teach difficulty, release
coordination or long-jack response. It also differs from the earlier
[contextual-count comparison](contextual_ln_count_learning.md), which retained
the analytic tilt and trained only count parameters.

## Joint learning comparison

The proposed comparison uses one architecture and factual draw sequence with
three initializations: the inherited scoped-allocation endpoint with that
unused module omitted, an earlier row-owned checkpoint, and fresh weights.
Both audio and H/R/R1 train; constructor-zero layout and H modulation matrices
provide the same available paths. Imported, omitted and initialized parameters
must be reported separately. Old checkpoint laws are not claimed identical
after changing the probability mode.

The sampling measure distinguishes natural default learning from explicitly
requested rare conditions. Natural equal-song-group draws permit independent
missing numeric controls. Difficulty/LN-balanced draws keep both numeric
conditions known. Annotation-centered draws retain at least one genuine style
judgment and keep numeric conditions known; annotation extents and unknown
dimensions are preserved. This avoids hiding the LN condition while continuing
to oversample high-LN charts as default answers. Accepted exposure and support
rejections still need auditing.

All arms receive current-weight full-song coarse audio and complete local
fine-audio halos, genuine source prefixes, event/survival likelihood and the
same deployed row preferences. No stale learned audio cache crosses an update.
The initialization comparison is causal only under this common new recipe;
comparison with earlier published results also changes law and data.

Learning curves and complete native generation determine whether an
initialization is useful. Short pilots verify gradients and resources, not
fresh-model quality. Scope-separated D2/D4/D6, LN/style, sustained pressure,
release relationships and Lens review remain required. No star or likelihood
scalar substitutes for the missing calibrated continuation response.

## Implementation evidence

Twenty-seven focused CPU/MPS checks pass across LN conditioning, ownership
and H modulation. They verify learned context/request interaction without
the analytic LN shift, independence from the reference prior, explicit-shift
behavior, strict checkpoint loading, native/replayed probabilities through
scope changes, reflection symmetry and gradients. Legacy reference/contextual
identity assertions remain exact; direct-mode agreement at the float32-encoded
reference fraction permits only the old logit transform's numerical roundoff.

These are probability and execution checks. No trained endpoint or generation
quality result is implied.
