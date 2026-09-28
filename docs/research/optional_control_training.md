# Training the control combinations that generation actually receives

The controlled generator exposes independently optional difficulty, style and
LN-fraction requests. The clean joint recipe did not cover their missingness
patterns independently: its `schedule` enabled `drop_ln` only in the natural
sampling branch, while style supervision mainly came from the human branch.
Every effective Stream-conditioned piece in that frozen 8,192-example ledger
retained an explicit LN fraction.

Resolving controls separately inside each scored interval gives:

| Effective prominent Stream supervision | All levels | Requested 3.5–4.5 |
| --- | ---: | ---: |
| Distinct training draws | 221 | 68 |
| Source charts | 21 | 10 |
| Constant-control pieces | 235 | 74 |
| Pieces with LN fraction unobserved | **0** | **0** |
| Median actual LN-head fraction | 0 | 0 |
| Pieces with actual LN-head fraction ≥.5 | 16 | 2 |

The ledger is `20260928-clean-joint-proposal-v1/source-plan.json`, SHA-256
`a48c9cc55f60fd6353295513f060ac23fd7f5487cf798907eb3eca8b6d716862`.
Counts describe exposure in this recipe, not independent examples or overall
annotation-corpus coverage. A style recorded elsewhere in a chart does not
count as supervision inside the selected interval. Raw counts of stored spans
therefore differ from this effective-query census.

This is a concrete gap for the user request “4★ + Stream, LN ratio unspecified.”
It contradicts an explanation that the matching source examples were mostly
LN charts. It does not by itself prove how much of the native LN collapse or
jack behavior this gap causes.

## A missing input is a different conditional distribution

Learning $p_\theta(Y\mid A,D,S,\rho)$ does not automatically identify

$$
p(Y\mid A,D,S)
=\int p(Y\mid A,D,S,\rho)\,p(\rho\mid A,D,S)\,d\rho.
$$

The implementation represents missing requests with values, availability bits
and scope clocks. A network can use those bits without obeying the marginal
identity. Observing missing LN requests only when style is unobserved does not
train the needed style-known/LN-unknown conditional.

[style_ln_views](../../src/ensomi_model/research/controlled_audio_continuation/conditional_views.py)
provides a paired training view on the same actual source example: retain the
original conditions, and hide LN value/availability/field clocks **only at
queries where a style field is observed**. Average the two losses so that a
source does not gain twice its sampling weight. Audio encoding, source rows,
exact history and physical support remain shared.

Unannotated portions keep their original numeric conditions. This prevents
style-balanced examples outside their annotations from silently changing the
unconditional LN prior. Supporting and absent style judgments are still
observed conditions; numerical zero does not mean an unobserved style.

The view is for neural likelihood queries, not an executable control schedule.
It has no spans to feed into a quota/amount feedback controller. LN feedback
must be disabled for its likelihood evaluation; no hidden source amount can
be reintroduced by that controller.

## Evidence still required

First compare known-versus-hidden LN conditions with identical source audio,
timing and history to measure the learned conditional response. Then compare
matched training runs with and without this one view change and inspect native
style-only requests, explicit LN requests and ordinary held-LN/TAP roles.
Teacher-forced odds and NLL are diagnostics; native pressure, rhythm, control
accuracy and runtime remain the qualification criteria.

The first matched probe used the joint-release80 checkpoint on all 14 distinct
source/intervals represented by the 4★-near prominent Stream subset, reporting
15 effective control ranges separately. H times, full audio and physical/neural
histories were factual and identical between the two views.
Hiding LN amount changed the expected LN-head fraction by a median **+.26
percentage points**, with maximum **+2.54 points** and minimum **−.44 points**.
The checkpoint still predicted predominantly TAP content on the TAP source
states. This does not support missing LN visibility as a sufficient explanation
for the large native LN collapse. The coverage defect remains real, while the
next causal question concerns generated histories, timing and proposal support.

A second matched probe used D4, only prominent Stream and missing LN in both
views. Changing the style extent from its annotation bounds to the whole song
changed expected LN fraction by median +.051 percentage points, with range
−1.175 to +1.494. The same 14 intervals and 15 ranges were retained; this was
another factual-history probe, not autonomous generation. Its result likewise
does not explain the large native change by input missingness/scope alone.

An autonomous follow-up supplied a persistent private LN-composition condition
to R/R1 while keeping H's public request unchanged. Even the sampled zero-LN
condition left 27 LN heads among 49 heads in the fixed source-H Stream scope.
The [ordinary-arrangement analysis](r1_ordinary_arrangement_redesign_zh.md)
distinguishes a coherent latent plan from a scalar composition request and
records why the latter is insufficient for this decoder.

This recipe repair does not replace the independent
[continuation response](action_response_frontier.md), a persistent rhythmic
plan, or learning coherent held roles. Those mechanisms address other
confirmed and still-open parts of the generation failure.
