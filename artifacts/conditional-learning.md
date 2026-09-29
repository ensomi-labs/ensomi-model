# Learning a request is different from exposing its input

After [R1 regained complete-row decisions](scoped-controls-and-ownership.md),
the work tested whether difficulty/style/amount conditions could control those
decisions from the history actually reached. The human repeatedly required
mechanistic explanations and actual chart inspection; NLL was explicitly a proxy.
Sources: [private, local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0176)
and [mechanism requirement](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0178).

## What the bounded learning sequence established

| Direction and source | Useful result | Why it did not close the task |
| --- | --- | --- |
| [Causal response projection](/Users/l/projects/ensomi-model/docs/research/causal_response_control.md), `b3fd694` | Made the actual generated attack history part of a row-probability intervention; localized remaining action space | Low-LN calibration gains did not solve high-LN, timing-floor or short-scope behavior. A projected row law has a different training meaning from the unmodified neural softmax. |
| [Active LN birth-audio cues](/Users/l/projects/ensomi-model/docs/research/active_ln_audio_cues.md), `c1260d3`/`2b20c4f` | In a regular annotated passage, H-coincident releases were 71/83 with cues versus 21/88 under matched continuation, preserving useful 110/220/331-ms relations | Primary median/short-LN improvement criteria failed; ordinary continued learning accounted for much of the gain. Another source's long anchor roles remained absent. No native-H expansion followed. |
| [Whole-chart outcome learning](/Users/l/projects/ensomi-model/docs/research/outcome_learning_and_control_response.md), `c3befb2`/`0c057af` | Replayed deployed probabilities on genuine generated histories and made outcome credit explicit | Scalar difficulty/amount targets did not supply missing semantic organization or stable request response. |
| [Restricted paired-scope adaptation](/Users/l/projects/ensomi-model/docs/research/paired_scope_control_learning.md), `2c2b8af`/`5bf146e` | MAE .80865→.75583; high-minus-low response .16037→.25390 | Missed .20 MAE improvement and .40 response criteria. Only 12,288 effective input weights trained; failure did not reject the capacity of all R1 architectures. |
| [Balanced condition alignment](/Users/l/projects/ensomi-model/docs/research/balanced_condition_alignment.md), `b2a5351`/`6520ba1` | Correct-condition ranking reached 20/24 genuine scopes | Generated error improved only .068; repeated groups and LN organization remained. Factual discrimination is not sustained control. |
| [Physical trajectory matching](/Users/l/projects/ensomi-model/docs/research/physical_trajectory_matching.md), `0e5ca77`/`720c632` | Compared actual short-interval physical continuations rather than treating style as a count | Kernel similarity remains a particular geometric diagnostic, not a semantic or playability oracle. |
| [Conditional layout modulation](/Users/l/projects/ensomi-model/docs/research/row_condition_interactions.md), `24fc4f1`/`0591f90` | Reserved style-kernel distance .06534→.04953 and scoped MAE .85922→.73150 versus matched unmodulated training | Some LN medians shortened; absent/prominent Trill still produced repeated groups rather than reliable exchange. Local and relative guards were not overall acceptance. |
| [Factual style discrimination](/Users/l/projects/ensomi-model/docs/research/scoped_style_discrimination.md), `e48e4ba` | Improved the source comparator and corrected mismatched reference/request scopes | A shared generated prefix still did not reliably respond to the style knob. |

These were distinct interventions with different trainable parameter sets,
data and outcomes. They must not be narrated as successive quality-approved
releases. The referenced product reports preserve their paired controls and
local artifact identities; they were read as historical evidence in recovery,
not rerun.

## Two concrete explanations survived the comparisons

First, the data supplied useful conditional variation but sparse semantic
coverage. An exact same-audio/same-H inventory found 1,339 filtered local
arrangement pairs across 424 audios. On 24 genuine scopes, the original model
preferred the lower difficulty in 22, including ten genuinely harder examples.
This was a factual-condition bias before any generated-history explanation.
Prominent Tech had only five assessed TRAIN examples in the broad audit, all
with local readouts in 3–4; balanced sampling cannot create missing difficulty/
style combinations. [Coverage study](/Users/l/projects/ensomi-model/docs/research/control_condition_learning.md).

Second, an affine main row head following a shared additive condition cannot
change reflected-candidate relative odds through that condition alone. At a
repeat/alternate witness, the main head supplied +2.510 log-odds, versus +.062
routing and +.024 `frontier2`. Numerical perturbation verified the algebra.
Other nonlinear paths still existed, so the whole policy was not condition-blind.
Multiplicative modulation removed this path restriction with 16,384 parameters;
its mixed native result shows why a proven capacity repair is not a proven
musical repair.

## Full-R1 common-prefix outcome learning

The [common-prefix study](/Users/l/projects/ensomi-model/docs/research/common_prefix_outcomes.md)
reported by `a99519c` trained all 2,753,715 R1 parameters while audio/H/R stayed
frozen. It used executable low/high targets from the same generated physical
prefix, without copying source labels onto that prefix. Of 55 admitted pairs,
47 fitted and eight were reserved; all 768 target training continuations were
TAP-only, so the future-tail credit path was not exercised by this fit.

On 48 reserved continuations, initial/source-only/outcome difficulty MAE was
.51248/.59182/.38894, while gap error was .73596/.53646/.68529. Absolute
calibration improved more than conditional separation; the declared full
criteria failed. Native static/before/restored difficulty improved while the
harder override worsened. LN amounts improved in those four categories,
without a native source-only comparator to isolate the outcome term.

Another matched 128 updates compared continued independent costs with paired
response plus trajectory KL (`ba406ef`). Reserved MAE became .55270/.53281
against the starting .38894; gap MAE .57377/.76815 against .68529. The combined
intervention failed its primary comparison. It nevertheless improved two later
Stream witnesses, so failed scalar gates do not erase useful local evidence.

The human's playtest report then identified severe D4 Stream long jacks and
missing breathing. It names `a99519c` but no exact output file/seed. The private
record preserves that limit and the follow-up answer rather than claiming
later reproductions were the exact tested chart. This redirected work to
[time-based player response and phrasing](player-response-and-memory.md).

## Constraints for reusing these results

Factual source imitation always has its own real prefix. Generated branches
need outcome evidence on their actual state. Native proposal likelihood is not
the selected planner policy's likelihood. Frozen H/R weights do not make their
realized outcomes constant when R1 changes LN state. Difficulty, physical
kernel distance, semantic style and inspected organization are different
measurements. Unknown style is not a negative label, and annotation scopes
cannot be expanded to fill a convenient training window.
