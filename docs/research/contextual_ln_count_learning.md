# Contextual LN counts: matched learning improves NLL but not native quality

Directly exposing the requested LN fraction to R1's type-count readout did not
produce a material benefit in this bounded comparison. Both count-only source
refits improved validation likelihood and both regressed native control cases.
Neither checkpoint is qualified or promoted. The default probability law remains
unchanged.

The result narrows a hypothesis from the
[formal pattern-failure analysis](native_pattern_failure_analysis_zh.md): an
explicit control-path restriction exists, but removing it is not an established
repair for fragmented LN organization or sustained single-column pressure.
The system still needs broader arrangement learning and independently grounded
continuation responses.

## Intervention and what it can change

The baseline readout uses reference LN fraction $\rho_0$ for its local type
preferences and adds the requested log-odds shift. The alternative feeds actual
$\rho$ to that readout and retains the shift:

$$
\ell_l^{\rm ref}=f_l(u,\rho_0)+l\delta(\rho),\qquad
\ell_l^{\rm ctx}=f_l(u,\rho)+l\delta(\rho),\qquad
\delta(\rho)=\operatorname{logit}\rho-\operatorname{logit}\rho_0.
$$

Here $l$ is new-LN count within a fixed head/release-count family, and $u$ is
the remaining row context. Other complete-policy paths can already depend on
the actual control. This is a path-specific restriction, not a claim that the
whole baseline ignores LN requests.

Both arms start from the same actor128 weights. Only `composition.*` trains:
143,011 of 4,600,369 parameters. Audio, H, R, row history, layout and frontier
weights remain frozen and are checked against their saved values. The models
retain 60/50/50 ms recovery and the default empirical recovery preference.
LN-amount feedback is off in both training and generation. Each source target
uses its own factual prefix, with full audio available throughout.

This intervention has an important capability boundary. For two legal complete
rows $a,b$ with identical counts $m=(n_H,n_{LN},n_R)$, at the same fixed history,
the composition factor cancels:

$$
\log\frac{\pi_\theta(a\mid h)}{\pi_\theta(b\mid h)}
=\ell_0(a,h)-\ell_0(b,h)+g_0(a,h)-g_0(b,h)
-r_0(a,h)+r_0(b,h).
$$

$\ell_0$, $g_0$ and $r_0$ are frozen layout, consequence and recovery terms.
Thus this fit cannot directly learn a better column or release-identity choice
inside a count family. It can change counts and consequently future reached
states. Those indirect changes must not be mistaken for new learned routing
capacity. A complete repair can change the frozen components too; they were
frozen here to isolate the proposed count intervention.

## Factual exposure and optimization

The frozen training sequence contains 256 eight-second windows from 224 charts
and 214 song groups, all with recomputed whole-chart difficulty in 2–6 stars.
The accepted draws comprise 180 population and 76 human-annotation windows.
Population proposals balance metadata-star/LN-fraction strata, then sample
song groups and charts. LN-fraction strata are not LN coverage or style labels.
There are 69/56/79/52 accepted draws in the recomputed 2/3/4/5-star bands.

Controls use 122 whole-song schedules and otherwise factual 16/32/64-second
scopes. Whole-song difficulty uses `compute_mania_star_rating_20241007`; finite
scopes use the existing source strain proxy. Difficulty, LN and individual style
conditions can be independently missing. Human style labels keep their original
scopes; missing or unreviewed dimensions are not negative labels. There are 216
draws with a known LN request.

The five audio identities in the earlier candidate-supply panel are excluded
from this fit. Twenty-two validation charts are held out relative to this fit;
absence of ancestor exposure or previous research inspection is not claimed.
Preparation rejects four source release-window conflicts, four row-support
conflicts and nine windows without row supervision across training/validation.
The accepted distribution is therefore explicitly recorded, not called uniform.

Each arm receives 128 matched updates, two windows per update, AdamW learning
rate $10^{-4}$, weight decay $10^{-4}$ and gradient-norm cap 1. The objective is
mean deployed source-row NLL per second over the sampled windows. This is an
intentional sampling-weighted row objective, not an unbiased whole-corpus joint
likelihood. Timing losses do not update H or R.

## Likelihood and whole native outputs

Validation uses macro mean NLL per row across the 22 fixed chart windows:

| Readout | Before fit | After 128 updates |
| --- | ---: | ---: |
| Reference tilt | 1.730817 | 1.665880 |
| Contextual tilt | 1.730633 | 1.665792 |

Both improve by approximately 3.75%. The contextual advantage after fitting is
only **0.000088 nats/row**, below the predeclared .02 learning discriminator.
It is not evidence of a substantive improvement from the extra interaction.

Native evaluation completes four arms—both unfitted laws and both fitted
models—on the same eleven cases: four songs with two seeds, a live control
override, and two prominent-Stream Zenithfall seeds. All use full audio, native
H and feedback off. **H timestamps are exactly identical across all four arms
for each case.** The fitted models also produce exactly identical complete row
sequences in all six Classic/STYX/Blizzard cases and the control-switch case.

The following whole-song fractions keep songs and seeds separate. Both fitted
arms have the same values in these six cases:

| Song | Unfitted reference LN fraction, s0 / s1 | Fitted LN fraction, s0 / s1 |
| --- | --- | --- |
| Classic Pursuit | .145 / .191 | .048 / .019 |
| STYX HELIX | .939 / .933 | .223 / .439 |
| Blizzard Heights | .993 / .993 | .989 / .989 |

Classic changes from passing both amount checks to failing both. Its whole-song
median LN H span rises from 1 to 2 in both seeds, while most LNs disappear. This
is a concrete counterexample to rewarding a more source-like duration/span
statistic in isolation. STYX loses its near-all-LN behavior, but one seed still
fails amount control. Blizzard retains near-all-LN generation and one-H median
spans; its median durations decrease from 166/163 to 147/145.5 ms.

The Stream condition has unspecified LN fraction. Whole-song sustained-attack
excess and actual osu!mania stars are:

| Seed | Unfitted reference excess / stars | Fitted reference excess / stars | Fitted contextual excess / stars |
| --- | --- | --- | --- |
| 271200 | .043476 / 4.861 | .022885 / 4.962 | .022988 / 4.965 |
| 271201 | .106111 / 4.984 | .119669 / 5.167 | .134512 / 5.275 |

The second seed regresses; contextual excess also exceeds fitted reference by
.014843 seconds, beyond the declared .005 guard. At unchanged H, head counts
increase from 4,108/4,292 to 4,612/4,734 in fitted reference and 4,581/4,766 in
fitted contextual. Reduced LN use is not free relief when additional attacks
increase demand elsewhere.

Each unfitted arm fails numerical checks in 5/11 cases; each fitted arm fails
7/11. The live override still fails its complete-scope LN amount check. The
before/restored fragments remain diagnostic ranges, not new amount quotas.
No under-20-ms same-column attack or publication failure appears in this panel.
These numerical results alone are already insufficient for qualification.

## Lens reading against real arrangements

Twenty-four distinct pages were actually inspected, covering source, unfitted
reference and fitted reference at seed 0. Eight contextual pages are byte-for-byte
identical to their inspected fitted-reference counterparts. The ranges are
Classic `[88589,93589)`, STYX `[1800,7800)` and Blizzard `[40342,46342)` ms.

Classic's reference alternates short LN accents, longer holds and TAP/chord
responses. The unfitted output is already predominantly TAP; the fitted view
contains only TAP/chords. This is not an established LN-organization repair.

The inspected STYX reference is a TAP episode, which is compatible with its
nonzero whole-song LN fraction. The unfitted model overlays predominantly LN
motion. The fit introduces more TAP accents and longer carried holds. This is
a changed texture, with possible local benefits, not evidence that matching
the source's local LN count is required or that whole-song organization passes.

Blizzard's reference combines sustained anchor voices with TAP accents and
paired short holds, then thins out. The unfitted model produces overlapping
LN-only flow; the fit moves toward serial LN flow with occasional paired entries.
It does not restore the inspected anchor/TAP relationship. Simpler overlap is
not automatically better, nor must a valid alternative copy that reference.

These are agent observations of time, action and organization. They are not
new human labels, listening or player-trial evidence. Seed-1 views and other
passages remain unreviewed; the replicated semantic-improvement requirement is
not met and no broader style-preservation claim is made.

## Crossed weights and actual reached histories

A subsequent read-only diagnostic evaluates both reference weight endpoints on
both actual native histories at 15,642 matching H clocks across the eleven cases.
It reconstructs the same feedback-off deployed row law, with each history's true
LN state, clocks, content and support. This is not a rollout in which one history
was substituted into the other policy.

Let $F_{wh}(t)$ be expected new-LN count divided by expected head count, with
weight endpoint $w\in\{0,1\}$ and history source $h\in\{0,1\}$. A symmetric
decomposition is

$$
\Delta_w=\tfrac12[(F_{10}-F_{00})+(F_{11}-F_{01})],\qquad
\Delta_h=\tfrac12[(F_{01}-F_{00})+(F_{11}-F_{10})],
\qquad \Delta_w+\Delta_h=F_{11}-F_{00}.
$$

The history term includes changed occupation, clocks, counts and support; it is
not a causal percentage attributable to the TCN alone. At queries where both
histories have no held lane and exactly the same complete-row support, the ratio
of mean absolute history effect to mean absolute weight effect is:

| Case | Eligible queries | History / weight effect |
| --- | ---: | ---: |
| Classic s0 | 696 | 6.42 |
| Classic s1 | 703 | 7.57 |
| STYX s0 | 21 | 25.89 |
| STYX s1 | 26 | 6.35 |

All four meet the declared exploratory criterion of at least 20 matched queries
and a ratio above two. Exact clocks and historical content still differ inside
this subset. The observation supports investigating behavior on reached states;
it does not establish that dependence on history is intrinsically harmful.

At fixed old histories, the refit increases expected releases per H query by
.00551/.00995 in Classic and .03307/.03819 in STYX. Longer realized median holds
therefore cannot alone establish a generalized preference to preserve current
holds. Which LNs start, which histories are reached and which holds survive can
change the observed duration population. Specific anchor-retention effects need
their own conditional comparison.

Blizzard remains near a saturated LN preference: on seed-zero old histories,
the expected LN/head ratio is .99399 before fitting and .98567 after fitting;
on the fitted histories it is .99582 and .98992 respectively. Only seven/six
queries in its two seeds meet the all-free/shared-support restriction, so that
subset is not used as broad evidence for Blizzard.

The KL chain decomposition numerically confirms the count-only limit. The
largest conditional-layout KL at a fixed history is $5.84\times10^{-14}$ nats;
the complete-row change is accounted for by family mass, up to arithmetic.
This is a mechanism check, not a quality score. The diagnostic completes in
264.20 seconds on CPU one thread, with maximum observed footprint 690,537,864
bytes. Its owner is `20260927-crossed-prefix-laws-v1`; frozen plan SHA is
`df593608e2370d22629794d7892f8dec6c4331d5b91894f5b42df9e0dfec8097`
and summary SHA is
`dfa1ce6e800ec082c7d09c23812099d04794dbde5cb10f102bddfa9d91b72dda`.

## Consequences for the next model change

The evidence favors moving beyond count-only repair. The new explicit LN
condition is available as a research option, but is not a demonstrated missing
ingredient. Increasing its training budget alone has no current quality support.
This small experiment also does not prove broader joint source learning useless.

Two distinctions must guide the next intervention:

- **Direct decision capacity versus reached-state drift.** At fixed history,
  this fit cannot alter within-count column/release choices. A change intended
  to repair those relations must train or redesign their actual owner. Compare
  changed weights on common native prefixes separately from complete rollouts.
- **Source likelihood versus organization under generated conditions.** A
  broader factual sample can improve teacher prediction while driving native
  LN usage toward different extremes in different songs. Future learning must
  address the H/history conditions it actually reaches and the continuation
  consequences, retaining factual imitation without inventing source suffixes
  for generated histories.

There is also a distinct control-state question. The plain actor's
[exact replay](../../src/ensomi_model/research/oracle_time_continuation/replay.py)
retains total rows and heads, but not cumulative LN starts. Its finite content
history cannot retain an arbitrarily old LN allocation. The separate
[amount state](../../src/ensomi_model/research/controlled_audio_continuation/allocation.py)
stores a clipped offset and is disabled in this comparison. Giving the row
readout the requested fraction therefore does not give it the actual historical
amount already allocated to that request.

A candidate learned control state could expose factual scope progress such as

$$
\Omega_W(t)=\bigl(N_{\rm head}(W\cap(-\infty,t]),\ N_{LN}(W\cap(-\infty,t]),\ t-a,\ b-t\bigr),
\qquad W=[a,b).
$$

Here $N_{\rm head}$ counts individual TAP/LN presses, not skeleton H rows.
These observations need not impose a uniform prefix ratio: the network could
condition allocation on music and remaining scope. They are control-accounting
facts, separate from player demand state. Scope interruption/restoration needs
explicit ownership, not new quotas for clipped fragments. This is an untested
missing-information hypothesis; no history-aliasing experiment or claim that it
caused the measured regression is made here. It explains why direct control
values and a larger count readout alone may still omit a necessary variable.

The missing LN/coordination response remains independent of this result: the
earlier attack-only candidate selector has no signal for most LN alternatives.
Larger models, audio/history interactions and explicit persistent LN relationships
remain possible interventions. Their objective must describe the missing relation
or response; extra parameters or a new count target do not supply it by themselves.

The reusable regression lesson is to report quantities and temporal relations
together, per requested scope and seed. Keep actual H parity, head count, LN
amount, occupancy, span/duration relationships, sustained pressure and viewed
witnesses separate. The paired qualification catches an apparently improved
NLL/span result that loses LN use and increases another seed's pressure.

## Execution and reproducibility

Implementation source is `1434924868879f6bef31f5d5ce1db332415df95e` on Apple M5,
24 GiB, Torch 2.11.0. CPU/MPS model, mirror, serialization and replay checks pass
21 tests in 14.05 seconds. Fitting uses MPS with one CPU thread. A 12-GiB task
footprint guard stops the first process during step 128 after 264.001 seconds;
no incomplete optimizer update is saved as a final model. An explicit restart
from both saved step-64 weights and optimizers replays the same remaining draws,
validates the saved source scores, and completes in 121.829 seconds. The latter
process peaks at 9,041,733,880 observed footprint bytes. Total fitting work,
including the discarded repeated updates, is about 385.83 seconds.

All 44 native outputs complete in 502.89 seconds on CPU one thread. The maximum
observed qualifier startup is .988 seconds and service interval .323 seconds.
This is the qualifier's two-second lead contract, not a rerun of the separate
30-row/eight-second realtime benchmark or a production client guarantee.

The experiment owner is `20260927-contextual-ln-fit-v1`. The source plan,
training/recovery scripts, identities, checkpoints, exports and figures are local
generated assets; they are not required to understand this report. Initial
adapter failures are retained: a recovery-script parse error before fitting and
a native-plan rejection before generation, caused by invalid amount gates on
interrupted scope fragments. Neither is counted as a completed experiment.

| Identity | SHA-256 |
| --- | --- |
| Source examples and conditions | `a766382082fa7e0b8674f88d9c7be128b41cf16d1fa7e881beb367336dbaddf4` |
| Valid native plan | `48379544b1b012fec14c8223a16f23275bc476a967a798d0c2c4cb6e6389d863` |
| Fitted reference checkpoint | `d1c7d334e15607e78b3b98c4e780bf381647580bbb32caa7eb67568d8bbc61a8` |
| Fitted contextual checkpoint | `ef4acc8eb34c60ec006cb0bc2976cd4b14b810b9ca8843cdc5ddc0c599fa8ea1` |
| Four native arm records | `9860a8154e08af64539dbd5b6676b2d365be5ad85ca563a8080ba75c29c1d908` |
| Per-case/scope analysis | `0015cee6a626bd4a8fd8a3eccd23d2894918ad0514d0fc35334879ee09e4e1de` |
| Lens inspection record | `dcb813c6db657887fbcbfbb9b5376b1f31ddd37616bf21d70863921d391afbaa` |

Lens harness revision: `22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`.
