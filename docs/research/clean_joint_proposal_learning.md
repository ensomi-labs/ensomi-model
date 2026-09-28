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

## Fixed intermediate evidence at 512 updates

The three 4,675,633-parameter arms complete the first 512 matched updates and
eight native cases each. They remain intermediate checkpoints in the fixed
4,096-update comparison; none qualifies for promotion.

The completed data ledger contains 8,192 factual windows and 3,736 distinct
TRAIN charts, plus 22 validation identities. Preparation takes 2,330.10 s and
records six support rejections. Of 4,100 natural draws, 255 are from charts
with whole LN fraction at least .5. Among 820 draws explicitly hiding LN
control, 59 are high-LN charts: 7.20%, compared with 30.41% in the preceding
balanced/dropout recipe. Balanced and human draws do not deliberately hide
numeric controls; an empty source scope can still have undefined LN fraction.
These are actual exposure counts, not proof that the resulting default law
is calibrated.

| Whole-chart result, same audio/seed/control | Inherited | Early | Fresh |
| --- | ---: | ---: | ---: |
| Stream Zenithfall, request 4 | 6.0442 | 5.7915 | 3.6993 |
| Classic, request 2 | 3.8003 | 3.3487 | 3.5458 |
| Classic, request 6 | 5.5070 | 5.1619 | 4.1202 |
| Zenithfall, request 2 | 5.3366 | 5.6762 | 3.3022 |
| Zenithfall, request 6 | 7.2415 | 7.0430 | 4.0920 |
| Blizzard LN fraction, request .8380 | .0075 | .0303 | .2408 |

Inherited and early each fail seven of eight numeric cases, passing only
Classic D6. Fresh also fails seven, passing only the Stream case. Its lower
pressure and star value do not establish style expression or musical timing.
Its D2/D6 response range remains compressed. The live-switch and LN controls
also fail; scopes remain separate in their case records.

The mandatory-H star floors for D2 Zenithfall are 3.4590, 3.4041 and 1.9596.
The first two cannot enter even the upper edge of the requested 2±1 band
through any R1 realization of those H plans. Fresh's floor does not establish
the same impossibility; its R1 realization still reaches 3.3022. Timing and
row materialization therefore remain separate actionable causes.

Validation H NLL improves from 32.0191 to 31.1879/31.1987 in the inherited/early
arms. Fresh improves from 45.5510 to 39.0281. Macro row NLL changes from
1.5728/1.6492/4.3274 to 1.6346/1.6241/2.4043. These changes coexist with the
native failures above and do not select an initialization.

### Inspected organization

All 24 pages of nine fixed contexts are read: STYX [1800,7800), Blizzard
[40342,46342) and the earlier Stream recurrence context [18335,23335), for
each arm. This is an intermediate visual review, not complete-song semantic
qualification, audio listening or a human playtest.

The inherited and early Blizzard views are entirely TAP in the selected six seconds.
Local TAP purity can be valid, but their almost absent whole-chart LN cannot
satisfy the high amount request. A longer median among the few surviving
holds would not count as an articulation repair.

Fresh Blizzard has overlapping held roles, common releases and TAP accents,
then a sparse TAP passage. Fresh STYX includes an approximately one-second
anchor while other fingers play, followed by changing held groups and lighter
TAP flow. Those are useful capabilities to retain, with the failed controls
still explicit.

The former Stream solo-jack window becomes LN-heavy changing holds in inherited,
mostly moving TAPs with one longer hold in early, and chord accents followed
by LN roles in fresh. No old solo run appears in that exact context. This
does not certify the rest of the song: inherited/early remain well above the
request, and replacing a jack with many short LNs is not a universal improvement.

The three Stream outputs have 4,634/4,720/1,582 H and sustained attack excess
.21830/.89997/0 seconds under the declared D4 reference. Maximum consecutive-H
membership is only 7/6/5, illustrating again why a short maximum run cannot
replace actual-time pressure. Whole LN fractions are .4952/.2142/.2565.

### Identity and remaining work

Executable learning source is `ef42095a6e764b0374edbaa36b8ddf87c32364c9`.
The artifact owner is `20260928-clean-joint-proposal-v1` under
`artifacts/joint-audio/`. Initializations, optimizer settings and source draw
sequence stay fixed after the pilot; no stage-512 quality observation changes
later training examples.

| Evidence | SHA-256 |
| --- | --- |
| Complete source ledger | `a48c9cc55f60fd6353295513f060ac23fd7f5487cf798907eb3eca8b6d716862` |
| Inherited 512 checkpoint | `08d36f79f9c65496c6e162ad497f855d5ac0644128d733946121303b5377f022` |
| Early 512 checkpoint | `82edf55f9cb778a92d8ce8477f40981076bdd855622112f046328d6dfbe2a013` |
| Fresh 512 checkpoint | `2cbdc4b91a6ce2b86a1199de2fdd96e7379b3e28a248703307cb578442602f43` |
| Scoped comparison and H floors | `8180941965597b175847365d89a1b5b1dc51ef9f9c8c703bcd76d2a29a4a875e` |
| Twenty-four-page reading receipt | `d18ebf79f22fba2fe59871535046af17dae86d3234cfb298d7b2c9639128c54e` |

Further fixed evaluations remain at 2,048 and 4,096 updates, with the complete
28-case panel at the endpoint. Final model selection, broader semantic review
and the exact-loader playback benchmark remain unproven.
