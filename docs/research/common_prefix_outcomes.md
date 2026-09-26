# Common-prefix difficulty learning in the full R1 policy

Training the full R1 policy on generated difficulty outcomes improved absolute
calibration, but did not sufficiently separate low and high requests. It also
produced useful full-audio LN-amount changes despite worse results on some
reserved controls. The checkpoint remains a research candidate; it does not
replace the selected runtime model.

This study follows [difficulty-stratified style learning](scoped_style_discrimination.md).
Balancing genuine examples supplies coverage, but their histories already contain
the arrangement being learned. A generated continuation must respond to a new
request from the history that actually exists. The question here is whether
outcome learning on common physical prefixes can teach that response without
removing R1's arrangement responsibilities.

## Executable targets from the same physical prefix

The initial model is the 4.60M-parameter
[condition/history-modulated R1 candidate](row_condition_interactions.md).
H supplies head timestamps; R supplies release-event times using actual LN state;
R1 chooses complete rows, including cardinality, TAP/LN, release subsets and
columns. Full audio is available to timing and row generation. This experiment
changes weights and training data, with no new network or inference policy.

The starting pool contains 128 real arrangement pairs sharing audio and local
H timestamps. Their whole-song H plans need not match. Restricting the target
bodies and their entering source states to TAP-only leaves 56 pairs. For each
pair, the initial model generates one common prefix on the low source's H plan,
under midpoint difficulty, LN fraction zero and unspecified styles.

Each original body is replayed after that generated prefix. Admission requires
exact replay and the existing row/recovery support, preserving every body
action. Scoped difficulty is recomputed with the common prefix's incoming
strain; both results must lie in [2,6), with a gap of at least .6. One high body
fails support, leaving 55 pairs: 47 for fitting and eight reserved for this
phase. Reserved audios are excluded from current-phase source anchors but may
have appeared in earlier training.

These constructions demonstrate executable target values from the particular
common state. They are neither human-approved hybrid charts nor supervised
suffixes attached to generated histories. Factual imitation always uses the
source body's own genuine history. Generated branches receive outcome credit.

Difficulty $D$ is the repository's
[scoped strain readout](../../src/ensomi_model/research/controlled_audio_continuation/outcomes.py),
with full incoming history and real endpoints for relevant LNs. It is an offline
proxy, not an official fragment rating or canonical player frontier $C_0$.
LN fraction $\rho$ counts LN heads divided by all heads within the requested
half-open time scope. Neither scalar establishes playability.

## Learning signal and dependencies

Both arms freeze the Mel encoder, H and R timing parameters. They train all
2,753,715 remaining R1 parameters, including row history, composition, direct
audio/control projections, layout modulation and learned frontier energy. The
learned energy is not a calibrated canonical player model. R's fixed conditional
law can still produce different times after R1 changes actual LN state.

Each of 128 updates uses three genuine-source anchors: the original low and high
examples on their own histories, plus an alternating population or prominent
human-assessed scope. The broad pools contain 878 population scopes and forty
human cells after exclusions. Difficulty bins and available concepts are cycled;
unknown styles remain unknown. Both arms receive the same 384 source identities
in the same order and minimize mean deployed row NLL per second.

The outcome arm additionally draws three independent continuations per request
from the same fixed prefix. It rebuilds current-model caches from committed raw
rows. The requested difficulty changes only on [a,b); midpoint difficulty resumes
afterward. LN fraction stays zero. Within each request, the two other draws form
a leave-one-out baseline for each cost:

$$
c_D=\max(|D-D_{\mathrm{request}}|-.25,0)^2,
\qquad
c_{\mathrm{LN}}=100\max(\rho-.03,0)^2.
$$

The score-function surrogate has weight ten and is averaged over the six draws.
Low/high requests share random seeds, but their costs and baselines are separate;
no coupled cross-request reward is used. This is an application of
[REINFORCE-style sequence-outcome optimization](https://arxiv.org/abs/2402.14740),
not a new estimator or a transfer of that paper's task-specific results.

Difficulty credit scores R1 choices through the last actual LN endpoint that
can affect the measured scope. LN-amount credit scores only [a,b). The real
remainder is generated to resolve endpoints; future tails never become model
inputs. R has no direct parameter derivative in this experiment because its
parameters and shared audio encoding are frozen. Its state-mediated effects
remain in the sampled outcomes. Sampled and rescored row distributions share
deployed support; maximum log-probability discrepancy was 3.42e-5.

All 768 training target continuations were TAP-only. The implementation's
future-tail horizon was therefore not empirically exercised by this fit.
Subsequent LN changes cannot be described as direct LN outcome learning.

Both arms use AdamW, inherited-parameter learning rate 3e-5, rate 3e-4 for
composition/control/preview/modulation, weight decay .0001 and clipping at one.
Four-update integrations precede fresh 128-update fits. On the 24 GiB M5 Mac,
source-only fitting took 116 seconds and outcome fitting 1,184 seconds. The
outcome arm had nonzero R1, composition and modulation gradients on every update.
Peak sampled footprint was 3.86 GiB; peak MPS driver allocation was 1.86 GiB.
Those counters overlap and must not be added. Frozen parameter hashes remained
unchanged. CPU qualification overlapped part of fitting, so the two elapsed
times are resource records, not isolated throughput comparisons.

## Calibration improved more than conditional separation

Each endpoint generates eight reserved prefixes × two requests × three seeds,
or 48 continuations. Targets, committed prefixes, H and future seeds are shared
across endpoints. Gap error is the absolute error of generated high-minus-low
difficulty relative to the executable-reference gap, averaged over 24 pairs.

| Reserved measure | Initial | Source-only | Plus outcomes |
| --- | ---: | ---: | ---: |
| Mean absolute difficulty error | .51248 | .59182 | .38894 |
| Mean absolute gap error | .73596 | .53646 | .68529 |
| High output exceeds low | 21/24 | 24/24 | 23/24 |
| Target continuations with LN fraction above .03 | 0/48 | 0/48 | 0/48 |

The requested progress was at least .20 lower difficulty error and .15 lower gap
error than both comparators, with positive response in at least 80% of pairs.
The outcome arm passes the absolute-error improvement against source-only and
the sign requirement. It misses the improvement against initialization and both
gap-error requirements. One fitting seed and eight reserved contexts do not
establish generalization or a robust response distribution.

Head cardinality often declines under both requests. In one seed-zero context,
low-request heads per H fall from 2.070 to 1.625, toward an executable reference
at 1.516. High-request heads per H also fall, from 2.484 to 2.258, away from its
reference at 2.672. Lens views retain moving singles and chord accents after
thinning. This is useful difficulty reduction, but weaker evidence of learning
how the requested change should alter the continuation.

The separate thirty-continuation, five-case control panel improves mean absolute
difficulty error from .73150 to .67563; source-only gives .73809. Three LN-amount
comparisons exceed their allowed .03 error regression, and one restored-range
difficulty comparison exceeds its .25 allowance. For example, Starry's low-request
restored error rises from .13186 to .38999. These failures remain failures even
where complete-song behavior improves.

## Style presence, physical similarity and control are different measurements

The locally matched style comparison holds each reference's local difficulty
and LN fraction fixed. Its
[physical short-block kernel](physical_trajectory_matching.md) mean rises from
.04772 initially to .06584 with outcomes; source-only is .04881. This fails the
allowed .01 regression. The distance measures similarity to particular physical
arrangements, not semantic style correctness or BAD patterns.

Trill shows why that distinction matters. On [288156,290040) ms, one outcome
draw maintains an inner-column exchange for seventeen successive H rows at
78–79 ms spacing, with outer TAP accents before transitioning to other flow.
Another alternates complete disjoint groups `[012]` and `[3]` for sixteen rows,
then varies the grouping. Columns here are zero-based. A 2+2-only complementary
pair counter misses the second form, while a physical kernel also distinguishes
it from the reference's geometry. The initial model already produces a 3+1
exchange in one matched draw, so this is not evidence of a newly acquired motif.

Extended Lens context and full attack groups support a provisional reading of
recognizable exchange, rather than treating every changed group as style loss.
The human-assessed Tear Rain example provides a related comparison: inner
alternation continues while outer notes become holds. Its human comment is
“according to context we can see 2-3 row alternation lasted for long, it's trill.”
The generated outer accents are TAPs, not those sustained background LNs. The
human judgment applies to its source episode and does not label generated charts
or endorse inherited machine rationale.

Control remains weaker than presence. With the actor's own generated prefix,
fixed H, local difficulty 5.076 and local LN request zero, switching only Trill
absent/prominent changes 0, 14 and 0 of 24 target rows over three paired seeds.
Two pairs are exactly identical. A separate inherited-prefix diagnostic changes
11, 0 and 4 rows without establishing reliable knob-induced exchange. Those two
diagnostic panels also differ in LN request; their between-panel differences
cannot be attributed solely to prefix provenance.

The bounded review viewed 54 pages, including one page from each of sixteen
seed-zero paired contexts for both initial and outcome models, selected style/LN
contexts, extended Trill context and native control/density views. Take's dense
view was corrected to [103970,108970) ms, the maximum five-second H count, after
detecting a renderer selection error. The previous [94500,99500) view was an
ordinary context. No model or metric changed. No generated human labels, audio
listening judgment or human playtest is claimed.

## Complete-audio behavior and remaining R1 action space

Six native runs use three audios, each with static and switching controls.
Static requests are difficulty 3 and LN fraction .2. Switching runs publish
through 63999 ms, then install difficulty 4.5 and LN fraction .6 on [64000,96000)
before restoring the defaults. Existing rows and open holds are preserved.
This exercises a runtime update, rather than supplying the entire schedule at
initialization.

All six H streams are identical between initialization and the outcome endpoint.
Thus differences are downstream of H; changed LN state can still affect the
frozen R law. The following errors average the three audios separately for each
scope. Different control ranges are not pooled.

| Scope | Initial difficulty MAE | Outcome difficulty MAE | Initial LN-fraction MAE | Outcome LN-fraction MAE |
| --- | ---: | ---: | ---: | ---: |
| Static whole request | .97346 | .84666 | .05879 | .02165 |
| Switching: before override | .69016 | .18854 | .05132 | .00956 |
| Switching: override | .45477 | .57158 | .06148 | .04278 |
| Switching: restored | 1.08290 | .64095 | .05596 | .01154 |

LN amounts improve in all four categories; difficulty improves in three and
worsens during the harder override. For Zenithfall, realized LN fractions before,
during and after the override are .2248/.6016/.2099, against .2/.6/.2. Its
corresponding difficulties are 3.021/3.614/3.696, so the restored difficulty bias
and undershooting of the high request remain visible.

These native results compare the outcome endpoint with initialization, without
a native source-only arm. They therefore do not isolate the added outcome loss
from continued factual learning. Each audio/mode has one draw; static and switch
modes use different seeds. Their within-model differences are not a causal
estimate of the control update.

Lens shows moving singles, occasional chords and a sustained LN in Take's peak;
the initial peak contains more multi-head rows and short LNs. Zenith's override
has overlapping LN organization, and its restored context retains older holds
across the control boundary. Some very short isolated LNs remain. These are
bounded visual observations, not a whole-song playability verdict.

Offline constructions also retain every native H timestamp while choosing one
existing head per H and approximately .2 LN fraction. Selected LNs keep their
actual endpoints; other objects may be omitted or converted to TAP. Exact replay
and the unchanged 60/50/50 ms experimental recovery support are verified.

| Native static H | Outcome difficulty | Executable construction | Construction LN fraction |
| --- | ---: | ---: | ---: |
| Zenithfall | 3.84376 | 3.28073 | .20029 |
| Hysteric | 3.77043 | 2.81558 | .19989 |
| Take | 3.92580 | 2.89065 | .19955 |

These are witnesses of remaining R1 action space, not lower bounds, learned
outputs, human-approved charts or an adopted deletion policy. Their true future
tails are used only offline. Zenith's value above 3 does not prove 3 unreachable.
The evidence does not justify moving head cardinality into H to repair R1.

Native CPU generation used one thread with the model already loaded and canonical
Mel cached. First-thirty-row publication took .214–.428 seconds, including audio
encoding from that Mel; the slowest publication window of up to eight seconds
took .335 seconds. No concurrent fit ran during these measurements. Waveform
decoding, Mel construction and model loading are excluded, so these are not
cold-audio startup times or a latency guarantee under load.

## Consequence for the next learning change

Difficulty/style stratification remains a data-coverage measure. Full-R1 outcome
learning improves absolute difficulty calibration relative to matched source-only
learning on common prefixes. Complete-audio results are also encouraging, with
the attribution limits described above. Much of the observed change still shifts
both requests toward lighter charts. Repeatedly increasing the independent
per-request costs' weight is not supported by this comparison.

The next question is how to calibrate the conditional response while retaining
the arrangement prior on generated histories. A direct paired-response objective
and explicit preservation on those histories are candidate mechanisms, not
validated repairs. Genuine source imitation, semantic organization, scoped
control accuracy and physical feasibility must remain distinct evidence.
Future joint H/R/R1 work should change H when timing is the limitation; this
experiment demonstrates cases with unused R1 action space on fixed H.

The frozen Mel encoder was not trained only on the small diagnostic panels: its
ranked-training ancestor updated all parameters with a pool of 6,923 charts on
2,573 groups. Pool size does not establish realized coverage, convergence or
audio-information sufficiency. This experiment neither tests those questions
nor rules out subsequent joint audio adaptation.

Executable source: `e48e4ba210a51951d530e6ff3989f41ec9794455`.
Artifact owner: `20260926-common-prefix-outcomes-r1-v1`.
Initial checkpoint: `b8aecd3e1f43339d2c1e3aff6d245a41e32a12008ddc20fd47e9eccb99544546`.
Frozen panel SHA-256: `8fbeb1ae026bf65c1feda5667d3a98a54808183b4b12e5dff236765e8bcc99f7`.
Source-only checkpoint: `ff9e1e8f9d8443e4543189e92ce6da9084ecb98a058a766dcd6bd45ada68d8f7`.
Outcome checkpoint: `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3`.
Python 3.10.20 / PyTorch 2.11 / MPS fitting; CPU generation with one thread.
Human comparison: source `1fe462e0773df58dd965d659f7c80da22fd37cf38c0f26f17e494717e6eb2ac1`,
scope [112953,113968) ms, Foundation `f-15fa68913bdb2bf3`, annotation revision
`b22a7a443783e05fee4db4b1d22b8e573ad448ae`.
