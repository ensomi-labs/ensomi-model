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

## Matched learning did not improve native low-difficulty control

At source revision `de5d560ce06ca0185087488b982e15cab394b136`, both arms
completed 512 updates on the same 1,024 factual eight-second draws. These cover
736 TRAIN charts and 661 song groups, excluding the five native-panel audios.
Both start from the same scoped-LN-allocation checkpoint, use restored
HH/RH/HR support of 60/25/21 ms, disable LN amount feedback and retain the
existing row preference. Only H parameters train: 513,108 for the additive
arm and 588,372 for the modulated arm. Audio, R, R1 and scope allocation remain
bitwise unchanged. Full audio remains available.

AdamW uses learning rates 1e-4 for the base/control/modulation projections and
3e-5 for other H parameters, weight decay 1e-4, batch two and gradient cap one.
Serial workers preserve optimizer state; there is no intermediate endpoint
selection. On Apple M5 / 24 GiB / PyTorch 2.11 with MPS, the paired fit takes
1,123.78 s and reaches a sampled process footprint of 3,396,128,296 bytes.

The 22 fixed validation windows improve H NLL from 32.01913 nats/s to
31.65408 for additive continuation and 31.62288 for modulation. R and row
likelihoods are exactly unchanged on those factual states. A post-hoc probe
finds a nonzero but small learned interaction: the audio-centered D6-minus-D2
base contrast has RMS .00580–.00802 logits across the five panel audios,
versus zero for the additive base. This is a base-logit measurement, not the
complete first-event law or musical-quality evidence.

Each endpoint generates the same 28 complete native cases from BOS. The
comparison also reuses the 28 profile-only parent cases. Whole-chart stars,
control scopes and sustained attack exposure remain separate measurements.
All 56 new outputs complete and pass export/reparse and sub-20-ms same-column
attack checks.

| Measurement | Profile-only parent | Additive fit | Modulated fit |
| --- | ---: | ---: | ---: |
| Four D2 requests: whole-star MAE | 1.38534 | 1.61354 | 1.78828 |
| D2 mean necessary-H floor excess above 2 | .40094 | .60125 | .53202 |
| Nineteen D4 requests: whole-star MAE | .56543 | .58944 | .51704 |
| Four D6 requests: whole-star MAE | .40625 | .65667 | .72655 |
| Six LN requests: LN-fraction MAE | .10635 | .08260 | .07479 |

The modulated arm fails both primary requirements: a .20 D2-error improvement
over each comparator and a .10 floor-excess improvement over the parent.
Both fits also fail the D6 non-regression allowance of .10. Better D4 means
and LN fractions do not qualify either endpoint.

The new H plans make three low requests particularly decisive. Additive
Zenithfall seeds zero/one have necessary star floors 3.27838/3.12661;
modulated seed zero has floor 3.16876. No R1 realization of those mandatory
H plans can reach even the upper edge of a 2±1-star band under the declared
rating. The floor for modulated seed one is 2.95931, so that case does not
establish the same band-level infeasibility. H must remain part of the repair;
R1 cannot compensate for every timing proposal.

The scoped-switch case also remains unsuccessful. After the D4.5/LN .6
override on [64000,96000) ms, the restored D3 range has scoped proxies
4.10075/4.52758 for additive/modulated fits. There is no matched static-D3
run here, so this does not isolate a lingering causal effect of the override.
It does establish failure on that declared control range.

### Sequence quality and interpretation

The fixed Classic, STYX and Blizzard review reads all 40 rendered pages.
Mixed TAP/LN and persistent anchors remain possible, but they do not certify
the surrounding release organization. Blizzard remains almost entirely LN
in the inspected generated passage, unlike the reference's mixture of
anchors, TAP accents and shorter groups. Further
[ranked coordination comparisons](coordination_frontier_hypotheses_zh.md)
examine these outputs at similar realized readouts.

New recurrence witnesses include fourteen consecutive H containing column 0
under modulated Stream control, and 29 solo column-0 TAPs under additive
Trill control. The latter spans 3,390 ms at median head spacing 120.5 ms.
Fewer repeated rows alone would not establish a gain: the modulated Trill
case has a shorter maximum run but much larger sustained attack excess.
The additional worst-D2 review queue was not completed; no full semantic
qualification is claimed. The numeric failures already prevent promotion.

The factual exposure also leaves a control-learning question. Among the
1,024 draws, only eight different-chart pairs on byte-identical audio overlap
in time, totaling 64 s. Five have known difficulty on both sides; only three
use whole-song difficulty on both sides, all at BOS. This describes this
recipe, not all parent training, and does not prove paired data is necessary.
Deliberate genuine-alternative exposure is worth testing. An alternative
chart's different difficulty is not an automatic negative label for H:
the same H can support different R1 difficulties.

This bounded comparison rejects scaling the unchanged H-only fit as a
demonstrated solution. It does not reject richer audio/control interaction,
joint encoder adaptation or complete-future response learning. No checkpoint
is promoted. Native CPU generation takes 568.94/561.38 s per arm; maximum
two-second startup/service times are .910/.394 s and .909/.479 s. Some
rendering overlaps late modulated cases, so these are observed service
records, not an isolated speed comparison. They do not qualify the separate
30-row/eight-second playback benchmark.

## Reproduction identities

The local owner is `20260928-head-audio-control-interaction-v1` under
`artifacts/joint-audio/`. The source revision above owns executable behavior;
the following SHA-256 identities locate derivative evidence if available.

| Evidence | SHA-256 |
| --- | --- |
| Frozen run plan | `5ddf7ca7607ec50027267e7d8b0089af1d5a8d9023a74b01ccc8720efedd5176` |
| Parent checkpoint | `8898c51714474f82171b570cd2c8867bb07e911a59dfbae91b2642687de6a4ca` |
| Additive endpoint | `2d2f1d87f59e391f014c2f37aec83f9e70eee6c8f3660715ad5de373bde8f461` |
| Modulated endpoint | `a08005e84755ba7e10e5b375d59c22422d7a7c835cf0d55badbc6cfe8597e13a` |
| Native comparison | `0588d5f261ed887bb459f501e477f24bb41e318679d9c64d17e23edcf9260ee2` |
| Forty-page reading receipt | `3e655e5c36c855b0ec1e0b71602781da314efc85018690fa7494242717fba8d6` |
| Same-audio exposure audit | `3fb711682d0dfae422b4cb69b99e141d866a8abcc0e1957c89f0b72d94e309d1` |

The comparison script was corrected before execution to omit a whole-star
bound assertion for the mixed-control case, whose whole-star field is absent.
Both script versions and plans are retained; no criterion or generated output
changed.
