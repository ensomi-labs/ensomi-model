# Lineage review 07: evaluation

## 1. Scope and sources

The strongest surviving result is a collection of exact observers and reproducible counterexamples.
The reviewed evidence does not establish an evaluator that recognizes ordinary, playable, musically
organized charts. Several narrow checks pass outputs from experiments the human rejected; later code
usually acknowledges this explicitly. This is an inference from the code and artifacts below, not a
claim that every numerical improvement was misrepresented as a completed system.

Review ran 2026-09-30 12:59–13:21 UTC. Scope: R1 baseline through `audio-joint-2026-09`, including RC.
`doc-claim` means a document/commit/feedback-index assertion; `checked-code` means implementation read;
`checked-artifact` means saved raw evidence read or a calculation reproduced here. Judgments state their
confidence separately. “Not recorded” means absent from the evidence examined, not proven absent everywhere.

Citation abbreviations: `D/` expands to `/tmp/lineage-review/trees/audio-joint/docs/research/`;
`S/` expands to committed `099cb66:src/ensomi_model/research/`; `A/` expands to
`artifacts/joint-audio/`; `R/` expands to `/tmp/lineage-review/07-evaluation/`.
Thus `S/gameplay_evaluation/report.py:14` identifies the committed source and line, not the checkout.

Read first: `.git/research-relay/notes/RESEARCH.md`, the feedback index sections 2–4, baseline README,
and `/tmp/lineage-review/trees/main/docs/formulation/gameplay-state.md:368`.
Read the nine requested research documents, including RC's complete evaluation guide and the evaluation
sections of `native_pattern_failure_analysis_zh.md`; consulted the named supporting studies below.
The recovered `evaluation-and-evidence-boundaries.md` was treated as `doc-claim`, not independent validation.

Checked implementation: chart replay/scope reporting, spacing predicates, profile transforms, whole/scoped
stars, pressure envelopes, LN prevalence, recurrence, interactions, TAP organization, lattice fitting,
audio correspondence, publication checks, probability diagnostics, trajectory kernels and qualification.
Read raw ranked census, temporal replay, recurrence census, blindspot, prevalence, sampling-variation,
qualification plans/cases and two Lens images. Replayed 100 real charts and 19 generated charts on CPU.
Forty-three selected frozen observer tests passed; no model training, inference, GPU/MPS use or downloads.

The human's judgments remain `doc-claim` via the paraphrased index. No private wording was opened.
No new listening, player test or human labels were obtained. The first live-demo failure has no retained
chart/seed, so later witnesses cannot be asserted to be that exact sample. Untracked historical scripts,
all 202 diffs and all 65 Agent Notes were not exhaustively audited; section 9 bounds inventory coverage.

## 2. Attempts

**Questions 1 and 2 — evaluator inventory and validation.** These are evaluator families; separate
output fields are grouped when they share an implementation and evidence boundary. Dates are UTC first
addition/use in the lineage, read from Git; inherited instruments are identified as such.
No trained quality classifier was found. Threshold fitting is distinguished from neural-model training.

| Evaluator; addition | Inputs, measurement and thresholds; checked implementation | Validation and claims resting on it |
| --- | --- | --- |
| Exact legality/replay/export; inherited R1, audio integration `2ae8e3a`, 09-23 | Rows, timing roles, occupancy, seed and actual LN endpoints; discrete contract violations, no musical threshold. `S/bounded_typed_continuation/verification.py:12`; later export/reparse equality at `S/gameplay_evaluation/qualification.py:136`. | Mechanical consistency only. Initial 48 native outputs passed (`doc-claim`, feedback V10–V13); rejected playability followed. Here all 119 inputs parse/replay (`checked-artifact`). No semantic discrimination validation. |
| Skeleton event F1 and absence guards; `135e181`/`68a48aa`, 09-23 | Predicted H/R-only times versus one reference; one-to-one ordered matching at 10/20/40/70 ms. Decoder thresholds .1… .9 maximize calibration F1 at 20 ms; empty-reference cases separately count predictions. `S/audio_skeleton/data.py:79`, `training.py:51`. | Six assessment songs; H F1 .677→.732 (`doc-claim`, `D/audio_conditioned_choreography.md:87`). No real-chart quality pass rate or human-label validation. Different valid arrangements need not match one reference's release timing. |
| Query/full-gap/interval NLL; `09b919c`, 09-23; `ce25a7b`, 09-24 | Factual prefix, target times/rows, audio; average query nats versus complete-interval nats/second are different units. `S/joint_audio_continuation/training.py:120`, `context_training.py:125`. Early “best” uses fixed VAL query NLL (`training.py:169`). | 48-query development panel; later 36 validation songs/22 fixed windows. Validates conditional fit, not native quality. Real-chart acceptance rate and independent human agreement not established. Used for coverage, history, transfer, H and row learning claims. |
| Native activity/short transitions; `caee76b`/`b757da3`, 09-24 | Rows through actual coverage; first/last head, reaching 30 heads, LN amount/duration, TAP/LN transition-specific gaps, occupancy, maximum one-second attacks. Gap coordinates 5/10/20/40/80 ms. `S/joint_audio_continuation/evaluation.py:24`. | Exposed silent-tail collapse and avoids disguising TAP→LN substitutions. Activity ≥30 and last-head >85% were operational panel guards, not ranked-derived quality rules. No aggregate semantic validation. |
| HH/RH screen, current-row and preview support; `452abc6`, 09-24; `ac7fa3a`/`7329b0e`, 09-25 | Same-column attacks and actual release→next head. Historical HH `<20`, RH `≤20`; 8-second windows, 20-ms halo, four proposals. `S/planned_audio_continuation/buffering.py:21`; later configurable recovery support at `spacing.py:90`. | Human criterion anchors strict short attacks; RH interpretation remained experimental. `checked-artifact`: ranked census 0/8,774 charts with HH or RH ≤20. `doc-claim`: zero excluded target rows among 683,341 rows/651 admitted charts under original policies. Does not certify longer-gap jacks or LN organization. |
| Ranked action-relation census; `bcb6f3a`, 09-25 | Byte-matched ranked metadata and complete objects; HH, HR, RH, distinct-H intervals, chord/occupancy/density strata. Exact `<`, `=`, `≤20` counts; no learned quality score. Raw owner `A/20260925-ranked-2to6-reference-v1`. | `checked-artifact`: 8,774 parsed charts/3,387 beatmapsets; 762 metadata-eligible byte mismatches excluded. Eight HR≤20 instances in one valid chart, but 8,960 adjacent-H≤20 intervals across 566 charts. Strong evidence against global onset-spacing or individual-short-LN bans. |
| Whole-chart stars; inherited, explicitly calibrated 09-25 | Complete objects and clock rate. Repository Python implementation `099cb66:src/ensomi_model/osu_core/difficulty.py:517`, named 20241007; no external star library in this path. | Historical 17-chart comparison claimed max error .00000437 (`doc-claim`). Here 100 verified ranked charts match saved official metadata to max .000004973 (`checked-artifact`). Valid scalar implementation; no validation as complete playability. |
| Requested profile descriptor error; `10ddaa8`, 09-24 | Full-chart H/s, heads/H and LN/head transformed by log/log/arcsin-sqrt and TRAIN standardization: `S/planned_audio_continuation/profiles.py:12`. Experiment gates include 15% improvement and 10% regression allowances. | Compared to requested TRAIN representatives, not human quality labels. Native panels 8 songs × 2 or 4 requests, one seed/song. Aggregation can reward removing undesired LNs while destroying desired LN-rich responses. Real-chart BAD/pass calibration never demonstrated. |
| Scoped difficulty/amount outcome; `8687c35`, 09-25 | Complete-prefix strain, actual future tails, normalized weighted 400-ms peaks; `S/typed_audio_continuation/difficulty_targets.py:20`, `S/controlled_audio_continuation/outcomes.py:14`. Qualification defaults absolute difficulty error≤1 and LN-fraction error≤.1. | Explicitly an offline proxy, not official fragment SR or causal player state. Tolerances are experiment settings, not established human error bounds. Here whole-scope error>1 flags 0/100 real and 2/19 generated; narrower-scope validity is unestablished. |
| Short physical trajectory kernel/style distance; `0e5ca77`, 09-26 | H-to-H action/state cells, blocks 1/2/4/8, RBF bandwidths .125/.25/.5, mirror averaging and rollout U-statistic. `S/controlled_audio_continuation/trajectory_kernel.py:26,92,126`. | Four reserved style scopes ×3 draws and three LN scopes ×3 draws in a modulation comparison; .005 absolute/15% relative improvement gate (`doc-claim`, `D/row_condition_interactions.md:140`). Not semantic style recognition; no calibrated real-chart pass rate or blind judgment agreement. |
| Sustained attack envelope; `24e786b`, 09-26 | Prefix attacks and .5/1/2/4/8/16-second rates; equal-song/equal-chart q99 maxima, one-star bands, monotone interpolation; squared positive excess integrated in seconds. `S/player_response/envelope.py:36,66`. | `checked-artifact`: fit population 6,923 ranked TRAIN charts, D4 band 1,972/1,613 groups. Three real positive guards include two positive-excess charts. Nine generated Stream cases improve optimized cost 98.46%; no independent semantic pass follows. |
| Exact temporal workload/recovery/contrasts; `138a1f5`, 09-27 | Complete prefix and half-open scopes; counts, held fractions, free recovery after .25/.5/1 s, peaks and adjacent-window contrasts. `S/gameplay_evaluation/temporal.py:31,115`; `report.py:14`. | 41 historical reports, four real sources, three further positive guards. Tests establish additive/prefix semantics. No quality thresholds: real/pass and generated/flag rates are undefined for these descriptive fields. |
| Mel/chart correspondence; `138a1f5`, 09-27 | Canonical Mel, exact chart clock, 1/4/16-second feature windows; linear CKA against five circular shifts. `S/gameplay_evaluation/alignment.py:19,57`. | Four initial source comparisons; here 5 matched real versus 19 generated observations. No listening/semantic calibration, significance test or pass threshold. Higher alignment can reward loudness following; valid steady or elaborative charts can score lower. |
| Publication deadlines; `251ba0f`, 09-27, earlier streaming audits 09-24 | Actual elapsed wall time and settled coverage; declared lookahead/startup. `S/gameplay_evaluation/publication.py:9`. Standard qualification uses 2-second lead/startup/service, whereas earlier readiness required 30 rows and 8 seconds coverage. | Replays retained traces and injected stalls; here 19/19 meet their declared deadlines. Correctness/runtime test, not ranked-chart quality test; cache-ready origin excludes some startup costs and all client/network behavior. |
| Pressure witnesses/review localization; `5832191`/`ef49d63`, 09-27 | Exact attack/expiry edges and same envelope; all-scale episodes with peer heads/holds. `S/gameplay_evaluation/witnesses.py:9`. | One newly located Max Burning regression, six inspected source/generated pages; positive ranked exceptions retained. This localizes the optimized cost; it is not an independent second quality metric. |
| First-event waiting-law diagnostic; `fc641aa`, 09-27 | Hazard logits, valid/forced clocks; event probability, conditional delay, survival/censor mass and law comparisons. `S/gameplay_evaluation/waiting.py:33,58`. | Synthetic/native-sampler parity and control lesions; no real-chart quality threshold or human validation. Distinguishes timing shifts from total event probability, not good musical timing from bad. |
| LN timing relations; `9f5ed1d`, 09-27 | Actual duration and interpolated H-span, onset-group changes/spread, censored holds; predecessor context default 2 s. `S/gameplay_evaluation/hold_relations.py:15`. | Source/native and synthetic timing perturbation comparisons. No quality gate: short, unequal or changing lengths are not intrinsically BAD. Never shown to classify the human's full LN judgments. |
| Typed hold interactions; `13eefbc`, 09-27 | Continuing holds × TAP/LN/release, same/opposite hand, origin and coincidence witnesses. `S/gameplay_evaluation/hold_interactions.py:17`. | Equal-marginal/different-tail synthetic counterexample and ranked examples. Tested relations distinguish what counts/occupation miss; no calibrated preferred interaction distribution or pass rate. |
| Consecutive-H recurrence; `6fa3a54`, 09-27; context extensions through `3591df2` | Per-finger membership age, elapsed gaps, companion attacks/holds/releases; no elapsed-gap reset or quality cutoff. `S/gameplay_evaluation/head_recurrence.py:5,86`. | `checked-artifact`: 1,972 real charts/1,613 groups and 12 native cases in recurrence census. Synthetic equal-count, zero-excess pair has maximum ages 1 versus 8. Human BAD detection/false-rejection rates not defined. |
| Mandatory-H star floor; `dcc21bd`, 09-28 | Strictly increasing H times; conservative floor for the same whole-chart algorithm, `S/gameplay_evaluation/head_difficulty.py:18`. Compare only to a whole-chart request. | Mathematical bound and synthetic tests; here below actual SR for real sample, above exact D4 request for 1/19 generated plans. Proves some timing requests impossible; low floor says nothing about ordinary organization. |
| Release-risk calibration/NLL factors; `ef90a21`/`dc50ce2`, 09-28 | Exact row clocks, factual occupancy, deployed [query,256] log laws; marginal/pair/subset Brier and head/count/identity NLL. `S/gameplay_evaluation/release_calibration.py:44`, `row_likelihood.py:17`. | Eight selected TRAIN snippets, three initializations and 32/512-step endpoints (`doc-claim`, `D/ln_release_calibration.md`). No held-out population calibration; pure-R conditioning cannot validate R event time. Normalization tolerance 2e-5 is numerical, not a quality threshold. |
| Short-LN prevalence; `e1fecbd`, 09-28; qualification wiring `822f34f` | Resolved ≤40/≤80-ms holds/all heads; song-weighted q99 inside star/requested-LN strata. `S/gameplay_evaluation/ln_fragmentation.py:12,45,79`. Whole-chart calibration must not become a crop threshold. | `checked-artifact`: six generated outputs all fail 40 and pass 80; three source witnesses pass both. Held-out real errors: mixed 1/145 at 40, 5/145 at 80; TAP-majority 3/208 and 7/208. Useful partial discrimination, not zero false rejection. |
| H-lattice and TAP-role organization; `a6c912f`/`3ea1827`, 09-28 | H lattice searches periods 40–2000 ms within 3 ms, ≥12 heads; coverage coordinates .5/.8/.95. TAP runs preserve all H and entering held roles. `S/gameplay_evaluation/rhythm_lattice.py:11,78`, `tap_organization.py:42`. | Synthetic properties and source witnesses; here chart-only observations on 119 charts. Parameters are observation choices, no validated BAD threshold. Counts alone do not identify ordinary TAP, Tech or musical intention. |
| Independent action-response/work budgets; `02e522d`/`4334e67`, 09-28 | Eight action channels, exponential scales .25/1/4/16 s, reciprocal-gap features; q99 source references and work budgets. `S/player_response/action_response.py:147,155,311,345`. | Fit 5,556/held-out 1,368 ranked charts. Pointwise marginal limits reject 113/1,368 (`doc-claim`); 4-second work rejects 21/1,368 (`checked-artifact`). Five of six known short-LN outputs pass work; all six pass quadratic variant. Not physiology or a complete acceptance rule. |
| Native qualification composition; `226725c`, 09-27; corrected thereafter | Generates, verifies, measures scopes, optional envelopes/prevalence, startup/service and actual publication. `S/gameplay_evaluation/qualification.py:136,203,276`; defaults `qualification_config.py:7`. | A failed number fails the candidate; otherwise status is `review_required`, always `promoted=false`. Thus code deliberately withholds final semantic certification. There is no end-to-end validated acceptance rate on ranked versus human-rejected charts. |

The `src/ensomi_model/evals/` directory contains MIR-anchor and legacy mapper/render/profiling tools.
Git shows no lineage changes there except the package rename; no reviewed result established those
as the quality evaluator for this lineage. Their presence must not be counted as validation coverage.

**Attempt chronology, exposures and contemporaneous interpretation.** All numbers below are
`doc-claim` unless marked `checked-artifact`; comparator and budget matter more than an instrument's name.
Fixed-weight audits have zero training steps; model parameter count is irrelevant to the observer itself.

| Attempt; human problems | Hypothesis/change and training scale | Evaluation/result, interpretation and next step |
| --- | --- | --- |
| 09-23 skeleton bridge, `135e181`–`068988e`; A/B/E/H | Learn H/R-only from audio, feed frozen R1; timing F1 and mechanical checks. Training steps/wall/model parameters/chart-song counts: not recorded in examined brief evidence. | Six-song timing assessment, 48 native integrations; generation seeds not recorded here. Legality succeeded but V12–V19 rejected playability; shifted to joint audio/event/row learning. F1 gains did not test the disputed row consequences. |
| 09-23/24 query, expanded interval and prior-27 studies; A/B/C/E/H | Initial 2,950,458 parameters; 121 TRAIN charts/48 songs, 12 VAL songs. Expanded 585/240, 36 VAL; 3,461,828-parameter candidate. Four interval cells each 4,800 intervals/37,300,370 ms/339,728 heads; training wall time not recorded here. | Coverage raises additional-24 query NLL 6.4852→5.4883, yet 3/84 samples have <30 heads. Four cells ×42 songs×2 seeds=336; no cell meets 3% likelihood gain. Prior reduces located ≤20-ms pairs to zero across 84 retained outcomes, but only 25 resampled; source/native LN mismatch persists. Interpretation already bounded, then prefix and transfer diagnostics. |
| 09-24 transfer comparison, `4db2335`; B/H/J | Three initializations, same 1,200 updates, 615 charts/240 songs; 37,253,258 ms and 345,613 heads per arm. Parameters and training wall: not recorded here. | NLL/s 40.40358/40.05814/40.06002. Native cap 1,800 s stops at 246/276 attempts, 245 complete; two arms have 84+8 cases, plain only 62. No complete fair three-arm native ranking; documents do not select a playability winner. |
| 09-24 UTC state/RNG diagnostic, `579c0a8`; B/H/J | No training, unchanged weights. Two selected song/state pairs; four saved-stream cells each plus 16 paired fresh seeds 239251–239266,72 continuations. | `checked-artifact`:199.197 s, paired bootstrap 10,000 resamples. Prom Queen apparent -.3501 LN change becomes -.0406, CI[-.1267,.0510]; Airborne -.0818, CI[-.1300,-.0300]. Both declared unresolved. Strong correction of an overinterpretable single pair. Artifact directory date is 09-25. |
| 09-24 fresh-audio screen, `0b35ece`/`e28435f`; A/B/H | No training; fixed checkpoint. Eight additional audios ×3 profiles ×2 policies; paired one seed/audio, parameters not recorded in this comparison. | 48 charts; HH 2→0, RH 8→0; sampling 83.982→86.680 s, full driver 217.125 s. Bounded success, LN control poor; retained screening option, not promoted. `D/fresh_audio_system_evaluation.md`. |
| 09-24 density routing/count-layout, `9e6dbc6`/`151f9c2`; A/B/F/H/I | Each arm 1,200 updates, 615 charts/240 groups,36 VAL. Routing 4,250,174 parameters,906.859/1,041.464 s; count arm 4,416,513 versus 4,250,174, fitting wall not recorded. | Routing 80 direct outputs, near-equal NLL40.1827/40.3493 but errors 6.1967/10.5420. Count 32 direct outputs: width+LN error 6.3338→4.9092 (22.5%), HH2→24; screened run stops 9 successes then exhaustion. No seed variance; both documents reject complete quality improvement. |
| 09-25 current/preview and ranked census, `ac7fa3a`/`bcb6f3a`; A/B/H/K | No training. Same endpoints,8 audios×2 profiles×2 models×2 policies=64 calls, one seed/audio. 8,774-chart reference,3,387 beatmapsets. |62 complete; count retries 6→3 meets 50% gate but exactly 20-ms HH and HR omitted. Census exposes them; constraints strengthened. `checked-artifact` census preserves legitimate short-H and rare short-LN exceptions. |
| 09-26 kernel/control actor, `0e5ca77`/`0591f90` and common-prefix work; C/D/G/H/I | Modulation fit 1,209 s; parameter/step/training-chart counts not recorded in this slice's checked sections. Related common-prefix result uses 8 reserved contexts×2 requests×3 seeds=48 continuations/endpoint. | Four style scopes kernel.06534→.04953, difficulty MAE.85922→.73150; LN distance.03087→.03248. Native 6 cases/3 songs, one draw/mode. Local inspected organization mixed; subsequent live-demo A/D complaint is not pinned to an exact output. |
| 09-26/27 pressure planning, temporal and memory diagnostics; A/B/D/E/G/H | No training for planner/replay; trained player-state/memory arms have budgets not recorded here. Envelope fit 6,923 charts;3 songs×3 seeds for native Stream; initial replay 41 reports. | `checked-artifact` replay 17.877 s: mean excess.0275164→.0004245 under planner, failed response fit.1071601. Later H substitution claimed zero pressure with median H ratio.318 and 3 startup failures. Multiple independent guards reject, rather than accepting the optimized scalar alone. |
| 09-28 H-control comparison, `de5d560`/`57cd13f`; A/B/C/E/H/I |512 updates/1,024 factual 8-s draws;736 TRAIN charts/661 groups excluding five panel audios;22 VAL windows.513,108/588,372 trainable H parameters, total not recorded here; paired fit 1,123.78 s. |28 cases/endpoint plus 28 reused parent. H NLL32.01913→31.65408/31.62288; D4 MAE.56543→.51704, six LN requests MAE.10635→.07479. D2/D6 regress; all 56 new exports/HH checks pass. V68 rejects LN/jack organization;40 Lens pages/15 contexts are evidence, not a pass. |
| 09-28 RC release/prevalence/action observers; B/C/G/H/K | Release calibration:8 TRAIN snippets,3 initializations,32/512-step checkpoints; models 4,675,633 parameters per clean arm (`doc-claim`). Segment K1/K4 pilots 32 updates,3,960 target rows each; parameter/wall/chart-song training counts not recorded here. | Six source-H native outputs/3 songs/seeds 281100,281110,281120 plus 3 source positives. `checked-artifact`: all six fail 40-ms prevalence, pass 80-ms; actual planner publishes 25-ms hold with zero work. Independent action budgets also miss 5/6; quadratic variant misses 6/6. No ordinary expert promoted. |

## 3. Commentary

**Question 3 — metric gains followed by rejected or unestablished quality.** The feedback index is the
source of human chronology. “Family-linked” below means a later complaint concerns that experiment
family; it does not manufacture a human label for every saved generated file. Related agent inspections
are identified separately. Effect sizes are conditional on the listed sample, without a general noise floor.

| Case and human link | Metric/result and n | What inspection found; why the metric missed it |
| --- | --- | --- |
| V12–V19,09-23; direct human reset |48 legality/export successes; F1 gain on 6 assessment songs (`doc-claim`). |12-ms release/repress and local burden spike. Contract and matching source timestamps do not measure the consequences of a legal row sequence. |
| V26–V28,09-24; family-linked NLL objection |Three 1,200-step transfer arms near-equal VAL NLL; incomplete 246-attempt native comparison (`doc-claim`). |Long-horizon/LN behavior remained disputed. Factual histories conceal the native states whose continuation preferences are failing; incomplete arms also prevent a fair quality rank. |
| V33–V38,09-24/25; direct short-attack objection |Screened zero `<20` HH:24 fresh screened outputs;62 complete current/preview outputs (`doc-claim`). |Exactly 20-ms attacks and 21–36-ms residues survive; HR was unscreened. Predicates answered narrower questions than requested comfort. The census explicitly corrected the boundary distinction. |
| V46–V52,09-26; live-demo report, exact seed missing |Related scoped outcome study: static difficulty MAE.97346→.84666, LN error.05879→.02165,3 audios/one draw per mode (`doc-claim`). |D4 Stream becomes one-column jack; uniform pressure/no breathing. Scalar star and amount reward totals, not sustained finger organization or phrase placement. Cannot assign that missing demo sample these exact numbers. |
| V53–V56,09-27; temporal follow-up |Pressure-selected 9 cases improve 98.46%; later fitted response actor has mean.1071601 versus.0275164 initial (`checked-artifact`,41-report replay). |One native regression reaches 9.5 attacks/s on a finger over 4 s despite improved training-bank cost (`doc-claim`). Here native evaluation caught the failure; a bank-score improvement was not sufficient transfer evidence. |
| V60/V68,09-27/28; family-linked |Memory H substitution claimed pressure.38230→0 and star MAE1.33452→1.16212 on 9 cases (`doc-claim`). |H-count ratio.318,3 startup failures, near-continuous holding and undershot controls. Reducing attacks changes the metric without supplying relief or ordinary organization; other guards correctly stayed red. |
| V68–V69,09-28; explicit 40-page pointer |H-modulated 19 D4 outputs: star MAE.56543→.51704; six LN controls.10635→.07479;56 new outputs pass strictHH/export (`doc-claim`; D4 endpoint independently reproduced here). |Very short/irregular holds, almost-all-LN Blizzard and missing regular LN+TAP. Whole-song totals and a minimum attack gap do not constrain arrangement relationships. |
| V75–V78,09-28; family-linked short-LN repair |Six K1/K4 outputs pass 80-ms prevalence; all six fail 40-ms (`checked-artifact`). K1 STYX short 80/all-head is 5.25% versus source 9.81%, but short 40 is 2.42% versus 0. |A broad duration bucket hides a more extreme tail. This is a diagnostic collision within a human-rejected family, not proof that K1 was ever human-approved or numerically promoted. |
| V79–V81,09-28; final failure remains red |Fresh and inherited factual NLL can improve; release decomposition examples:164.23→160.54 total while release-given-heads 62.86→74.73;211.15→188.17 while 72.85→82.19 (`doc-claim`,two of 8 snippets). |Normal held roles and TAP organization still fail. Summed likelihood lets improved head factors conceal worse release factors; later evaluation exposes this rather than explaining it away. |

Count/layout's22.5% descriptor gain and kernel modulation's.01581 style-distance gain are additional
agent-inspected proxy failures in this sequence, but the feedback index does not identify a unique human
rejection of each exact endpoint. They should not be inflated into extra independently labeled cases.

**Question 6 — what “red first” actually became.** `checked-artifact` evidence establishes the following:

- `A/20260927-gameplay-evaluation-v1/results/completion.json`:41 reports, comprising 4 source,
  10 source-matched generated and 3×9 historical generated reports; elapsed 17.877 s. This is a replay
  panel, not 41 independently human-labeled quality examples.
- `new-fault-lens-review.json`: one newly localized Max Burning failure, column 2 with 33 attacks/4 s
  despite whole-star proxy 3.8617; six source/generated pages were recorded as inspected. The witness
  has retained bytes and seed 273100; it is agent-confirmed, not the missing initial human demo trace.
- `positive-guards.json`: three inspected ranked controls; two have excess.00593949/.000804769 and
  the human-prominent Stream reference has zero. A universal “any positive pressure is BAD” rule
  would reject legitimate references. This limitation was preserved, not hidden.
- `A/20260928-recurrence-observations-v1/census-v1/summary.json`:1,972 charts/1,613 groups and 12
  native cases; it adds visibility, not an executable semantic red/green boundary.
- `A/20260928-response-blindspot-v1/quality-replay-v1/{checks,references,source-positives}.json`:
  six generated failures at 40 ms, three source passes, and nonzero held-out source rejection above.
  At80 ms the same six are all green. This is the clearest checked, useful but partial red-first result.
- `accepted-prefix-v1/result.json`: actual planner commits a25-ms STYX hold; first four decisions
  accept proposal 0 with zero work. `probe-v1/result.json`: four-second excess/quadratic budgets reject
  21/1,368 and 15/1,368 held-out sources, while permitting 5/6 and 6/6 segment outputs respectively.

Thus red-first work was real and some known failures became executable regressions. It did not produce
a unified suite that reliably stayed green on real charts while rejecting all the human's bad patterns.
The code's `review_required` result is consistent with that limited accomplishment.

**Question 7 — new CPU comparison.** Ran `R/compare.py` with lineage-end `PYTHONPATH`, followed by
`R/auxiliary.py` and `R/summarize.py`; exact command form is in section 7. No model weights were used.
Inputs: byte-verified census `A/20260925-ranked-2to6-reference-v1/charts.json`, source observations
`A/20260928-coordination-corpus-v1/scan-v1/charts.jsonl`, historical split flags in the blindspot
`probe-v1/charts.jsonl`, and `A/20260928-head-audio-control-interaction-v1/native-modulated-v1`.

Real selection: first 100 source-SHA-ordered charts in inclusive official 3.5–4.5 stars with historical
`held_out=true`;98 song groups. Generated selection: all 19 constant-D4 cases, five audios, original
seeds retained. Source bytes, metadata SHA, MD5 checksum, ranked/native-mania/4K status and official
snapshot rating were rechecked. This is historical calibration-held-out data, not untouched lineage TEST.
The 19 outputs belong to the experiment explicitly criticized atV68; they are not 19 separate human BAD labels.

Natural-mixture fragmentation references fit 1,580 historical fitting charts/1,288 groups with the existing
q99 routine; bounds are.002301496 at 40 ms and.195141700 at 80 ms. Generated requested-stratum checks
use the declared LN request, never realized LN amount. Unspecified requests use the natural mixture.
Primary run 11.763 s; auxiliary run 6.129 s. `R/manifest.json`, per-chart JSON and summaries retain all inputs/results.

| Existing evaluator/readout | Real, n=100 | Generated, n=19 | Meaning |
| --- | ---: | ---: | --- |
| Exact parse/physical replay |100 pass |19 pass |Mechanical validity only; no new export was needed. |
| `scope_report` strict below 20-ms attacks |0 flags |0 flags |Passes the entire criticized cohort. |
| Sustained attack excess>0 |0 positive |8 positive (42.1%) |A diagnostic exceedance, not an authorized universal rejection rule;11 are silent. |
| Natural-mixture short 40 prevalence |2 flags (2%) |11 flags (57.9%) |Partial discrimination; real exceptions remain. |
| Natural-mixture short 80 prevalence |1 flag (1%) |0 flags |Broad bucket misses every generated case here. |
| Requested-stratum short 40 / short 80 |Not comparable without assigned requests |10/19 /0/19 flags |Request conditioning changes the reference; no post-generation stratum selection. |
| Whole stars within 3.5–4.5 |100 pass |10 pass |Real selection used this range; not an independent real acceptance test. |
| Whole-star error versus source metadata/request>1 |0 flags |2 flags |17 criticized-experiment outputs meet this loose scalar tolerance. |
| Existing scoped-difficulty error>1, whole scope |0 flags |2 flags |Same operational tolerance, not a local semantic validation. |
| Mandatory-H floor above exact target |0 |1 |Necessary timing feasibility only; remaining 18 not certified. |
| Recurrence, hold relations/interactions, TAP organization, recovery, contrasts |100 observed; no gate |19 observed; no gate |No defensible “pass rate” exists for descriptive outputs. |
| Lattice maximum coverage, middle 16-s window |Median 1.000 |Median.308642 |No quality threshold; strong descriptive timing difference, not a Tech ban. |
| Audio CKA minus shift median,1/4/16 s |Matched-source n=5: medians.368/.427/.420 |n=19: medians.192/.220/.141 |No pass threshold; unequal conditions and repeated songs preclude a quality accuracy estimate. |
| Actual publication deadlines |Not applicable |19/19 pass |Historical recorded wall time, not new runtime measurement. |
| Release Brier, row NLL factors, waiting law |Not run |Not run |Require aligned policy logits/hazards, unavailable from osu rows alone. |

Five generated cases remain clear even under the conjunction of strictHH, zero attack excess,
requested 40 prevalence and whole-star error≤1; this is an illustrative conjunction, not a proposed gate.
Conversely, recurrence cannot be turned into “smaller is better”: real and generated maximum-age medians
are both 5 and maxima 17; generated repeated-head fraction median.087 is below real.165. The requested
styles and amounts differ, so these descriptive aggregates do not establish either group's semantic quality.

**Question 5 — star calculator.** The scoring path is a repository Python reimplementation of the named
20241007 algorithm, including its own strain and section-peak accumulation; it is not a call to an
installed official osu library. `checked-code`: `099cb66:src/ensomi_model/osu_core/difficulty.py:517`.
`checked-artifact`: our 100-source comparison against byte-matched saved official metadata gives mean
absolute error.00000238057, max.00000497279. Historical 17-chart agreement is therefore credible, but
only for this snapshot/version/rate; no current-server comparison was made. The scoped proxy changes
aggregation and normalization and must not inherit this whole-chart accuracy claim.

**Question 8 — what Lens supplied and what it could not.** The paired Blizzard pages newly read here,
`A/20260928-head-audio-control-interaction-v1/lens-main-v1/source-blizzard-heights-0.png` and
`modulated-four-blizzard-heights-s0-0.png`, show a concrete distinction (`checked-artifact`).
The reference holds a persistent role while other columns tap and paired holds release together;
the generated page largely cycles short holds and irregular tails. This is visible in the organization
of simultaneous and successive actions, even when a whole-song amount or duration summary looks plausible.
It does not require copying the reference to recognize that the claimed held-role behavior was not shown.

Historical Lens feedback likewise distinguished long single-column runs, continuing anchors, complementary
TAP groups, independently staggered releases and changing subdivision. Later relational observers can now
measure some of those differences; “no metric saw it” describes the earlier inventory, not a permanent
inability to instrument them. Neither the images nor these counts alone confer a semantic quality label.

Static selected pages cannot establish whole-song failure frequency, robustness across seeds/songs,
publication latency, perceptual audio correspondence, player comfort or a calibrated response curve.
A crop can also omit incoming/future context unless endpoint/action records accompany it. The temporal,
publication and corpus tools answered parts of the request for things a Lens page cannot reveal;
blind listening and actual player response remained unmeasured in the reviewed evaluation evidence.

## 4. Direction

The following judgments address each attempt family; confidence is confidence in the bounded judgment,
not confidence that an alternative architecture would work. Evidence is the graded inventory and comparisons above.

| Attempt family | Strongest case for the direction | Strongest case against | Call |
| --- | --- | --- | --- |
| Legality, F1, NLL and completion |Necessary executable foundations; source fit and timing matching expose concrete bugs/underfitting. |They evaluate a different target from native ordinary organization; early small panels invited overinterpretation. |Keep their narrow role; insufficient as progression evidence alone. High confidence. |
| Local screens and profile/amount/star optimization |Targeted interventions remove observed extreme conflicts and measure control use under matched conditions. |Boundary changes, missing LN relations and compensating aggregates let bad arrangements pass; stronger support excludes real styles. |Reasonable bounded symptom repair; does not establish the full direction. High confidence. |
| Kernels and sustained attack selection |Adds actual physical sequence information and real-time histories; nine-case pressure reduction is concrete. |Chosen features and selected objective lack independent semantics; a planner reducing its own cost is not external validation. |Useful diagnosis/selection hypothesis, unvalidated quality evaluator. High confidence. |
| Temporal, LN, recurrence, lattice and audio observers |Exact scoped facts and counterexamples expose distinctions earlier metrics erased. |An expanding list of witnesses is not a calibrated criterion for which arrangements are ordinary or musically apt. |Right instrumentation direction; evidence insufficient for acceptance. High confidence. |
| Release factorization and short-LN prevalence |Separates hidden subtask failures; six-bad/three-source contrast plus held-out error estimates is meaningful. |Only selected TRAIN snippets for probability calibration; duration prevalence remains easy to improve by moving the defect. |Strongest bounded regression evidence, limited to the named failure coordinate. High confidence. |
| Independent action-response budgets and qualification |Separates actor scores from evaluation; records false rejections and refuses numerical promotion. |Scalar budgets miss obvious accepted short holds; default execution does not necessarily run the planner. |Correctly exposes inadequacy of current response/acceptance mapping; not validated player demand. High confidence. |

**Question 4 — evaluation hygiene and reuse.** `checked-artifact`: `R/reuse.py` inspected 97 narrowly
selected plan files, found 39 with audio-bearing cases across 17 experiment directories:563 case records,
20 distinct audio hashes. These are counts of declared plans, not 563 independent completed experiments.
Repeated nested copies are deduplicated by experiment directory in the table. Counts are lower bounds
because older bespoke plan formats were not decoded.

| Development audio | Distinct experiment directories | Common repeated seeds |
| --- | ---: | --- |
| STYX HELIX (`dataset/0/1933843`) |16 |273120,273121 |
| Classic Pursuit (`dataset/0/730295`) |15 |273110,273111 |
| Blizzard Heights (`dataset/0/2071574`) |15 |273130,273131 |
| Zenithfall (`dataset/0/2293949`) |14 |271200,271201,271202, plus later diagnostic seeds |
| Max Burning (`dataset/0/320905`) |13 |273100,273101 |

The four closest-to-D4 source references are TRAIN selections (`D/four_star_phrasing_and_audio_memory.md`,
`doc-claim`; matching reference split fields are retained in plans). Excluding five panel audios from
one late fit does not erase ancestor exposure or repeated human/agent selection on those same audios.
The earlier fresh eight-audio panel is reused in density routing, materializer and row-support studies.
Its selected joint corpus excludes matching group/audio/waveform hashes, but R1 ancestor exposure and
perceptual duplicates were explicitly unknown (`doc-claim`, `D/fresh_audio_system_evaluation.md:10`).

Held-out data did exist: skeleton calibration versus assessment, source VAL, reserved factual prefixes,
and song-group-held-out response calibration. The V2 guide says TEST did not select that candidate.
No examined source demonstrates a final quality comparison on an untouched TEST panel, independent of
all checkpoint ancestry and iterative selection. It would be incorrect to call the repeated development
panel “held out” merely because its current fitting minibatches omitted those songs.

Run-to-run variation was measured in one unusually useful case. `checked-artifact`:
`A/20260925-continuation-state-rng-v1/result.json` has 72 continuations, two state pairs,16 fresh seeds,
paired SE.04747/.02613 and 10,000-resample intervals. Prom Queen's fixed-pair LN change-.3501 collapses
to mean-.0406 with an interval spanning zero; Airborne mean-.0818 remains below its declared-.10
material-effect requirement. This directly weakens single-pair causal stories. Most later gains have
one training seed and one/two/three generation seeds, without comparable uncertainty over songs or
training runs. One cannot transfer this LN noise floor numerically to stars, kernels or pressure cost.

## 5. What was overlooked or never questioned

**Question 9 — comparison with the formulation's six evaluation questions.** Source:
`/tmp/lineage-review/trees/main/docs/formulation/gameplay-state.md:368` (`doc-claim` of the specification).
The implementation coverage below is an inference from the checked instruments, not a changed formulation.

| Formulation question | Lineage coverage |
| --- | --- |
| Recover declared style concepts/ordinal strengths on held-out scoped judgments? |Not established. Human reference contexts and physical kernels exist; no reviewed semantic recognizer validates explicit negatives, unresolved labels and ordinal strength on held-out groups. |
| Preserve independently specified target responses on legal continuations/horizons? |Partial mechanical probes and response summaries, but the independent target-response specification remains undefined. Optimized envelope cost cannot validate its own sufficiency. |
| Demand contribution to recognition: chart-only, demand-only, joint plus density/count/difficulty baselines? |No such held-out recognition comparison found. Player-state ablations against actor losses address a different question. |
| Style request changes intended semantic organization at fixed audio/history? |Partial fixed-prefix and three-seed kernel comparisons, plus native/Lens witnesses. No independent validated semantics or blind human comparison establishes success; Source-H differs from native BOS. |
| Demand control changes intended response while tracking style/tradeoffs? |Partial stars, scope amounts, pressure and occupation comparisons. Tradeoffs are often retained, but “intended response” was replaced by empirical proxies, without independent validation. |
| Canonical symmetry of histories and legal continuations? |Substantial algebraic/property-test coverage in observers/kernels; selected mirror tests passed here. This establishes transformations, not mapper agreement, frontier sufficiency or gameplay quality. |

**Question 10 — missing pieces, ranked by consequence.** These are missing evidence requirements, not
an evaluation design or roadmap. “Missing” means not demonstrated in the reviewed lineage sources.

1. An independent meaning for acceptable ordinary organization and player response. The six-short-LN
   acceptance failure and undefined formulation response show why a precise computed scalar is insufficient.
   This question was named repeatedly, but its answer was never supplied by the observers themselves.
2. Joint validation of the actual acceptance decision. The ≤80-ms metric passes 6/6 witnesses and 19/19
   criticized-experiment outputs; work budgets pass 5/6 or 6/6. No complete rule has a demonstrated known-bad
   detection rate together with a representative real-chart pass rate and independent labels.
3. A final untouched comparison unit with ancestry and selection exposure tracked. Five audios reused
   across 13–16 plans cannot independently substantiate generalization, even when excluded from the last fit.
4. A noise floor for each claimed gain at the song, sampling and training-run levels. The checked RNG
   reversal undermines single-pair inference; most model-ranking deltas have no corresponding uncertainty.
5. Independent human agreement on the dimensions being optimized. The Lens records are predominantly
   agent observations of selected contexts; there is no blind, repeated judgment validating kernel/CKA,
   star, recurrence or prevalence as a substitute for the human's musical/playability assessment.
6. A matched observation scale and denominator for every criterion. Whole-song LN amount does not
   require uniform local texture; whole-chart prevalence is not short-window capacity; source event NLL
   is not rollout behavior. Scope corrections fixed examples but do not prove all old results comparable.
7. Evidence that ordinary valid structure is central rather than merely possible. Ranked exceptions
   correctly falsify universal bans, but do not establish the typical conditional distribution. Conversely,
   low recurrence or high variation alone cannot define ordinary organization; the 119-chart replay shows overlap.
8. Separation of response detection from actual invocation and final acceptance. `ControlledSession`
   qualification does not instantiate `ResponsePlanner`; even a real planner accepted a25-ms hold atzero
   cost. The existence of observer code, or a frontier-named module, cannot establish deployed protection.

## 6. Worth keeping

- **Verified reference population:** the 8,774-chart census retains checksum mismatches, exact endpoints,
  difficulty strata and valid counterexamples. `checked-artifact`, census `result.json`/`charts.json`.
- **Exact prefix/scope semantics:** incoming holds, attack history, open-tail censoring and additive
  recovery are explicit. `checked-code`, `S/gameplay_evaluation/temporal.py:31,115`;43 selected tests
  passed here (`R/observer-tests.log`). Those properties survive a change of generator.
- **Separating aggregate amounts from relations:** hold interactions, recurrence, TAP roles and timing
  relations expose real information loss. `checked-code` and synthetic regression tests; retain as observers.
- **Versioned star computation and H infeasibility floor:** the 100-chart metadata comparison is strong
  numerical evidence. Keep named version/clock and whole-versus-scoped distinction; no quality inflation.
- **Red witnesses with positive controls:** six 40-ms prevalence failures, three positive sources and
  held-out error counts; Max Burning pressure witness and ranked positive-excess exceptions. These are
  auditable regression evidence, although no conjunction is yet a complete semantic classifier.
- **Execution that cannot silently promote:** hashed plans/outputs, retained failures, publication traces,
  separate numeric/semantic status. `checked-code`, `S/gameplay_evaluation/qualification.py:276`.
- **The two-state/16-seed diagnostic:** a concrete example of retracting a persuasive single-pair story
  after measuring sampling variation (`checked-artifact`, continuation-state-RNG `result.json`).

## 7. Claims worth re-verifying

Reproduce this review from repository root with each command prefixed by
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1`:

```sh
PYTHONPATH=/tmp/lineage-review/trees/audio-joint/src .venv/bin/python /tmp/lineage-review/07-evaluation/compare.py
PYTHONPATH=/tmp/lineage-review/trees/audio-joint/src .venv/bin/python /tmp/lineage-review/07-evaluation/auxiliary.py
.venv/bin/python /tmp/lineage-review/07-evaluation/reuse.py
.venv/bin/python /tmp/lineage-review/07-evaluation/summarize.py
```

The scripts and manifests identify every computed input and n; reruns write only this slice's scratch.
Pivotal outstanding claims and the exact evidence needed:

- **Factual release calibration magnitudes:** `A/20260928-ln-risk-calibration-v1/scores-v1`, later score
  outputs, `factorization-plan.json`, `score.py`, `score-v2.py`, `factorize.py`. I read implementing code
  and document tables, not every aligned logit array. Rechecking requires the pinned checkpoints and
  source assets; historical drivers contain relocated worktree paths and must not be run blindly.
- **Current complete-qualification recall:** compare hashed `native-modulated-v1/plan.json` and
  `cases.json` above with `A/20260928-clean-joint-proposal-v1/lens-2048` and the retained rejected scopes.
  Missing independent BAD labels prevent a defensible population sensitivity number; more arithmetic
  cannot recover the absent judgment.
- **All claimed held-out exclusions:** inspect ancestor corpus manifests named by each plan, especially
  R1 restoration versus the eight “fresh” audios. Missing ancestor exposure/perceptual-duplicate evidence
  prevents calling the whole lineage's quality panel untouched; this review did not reconstruct it.
- **Final 84-case semantic review and missing first demo:** exact completed review records for the final
  three-arm outputs, and the original a99519c chart/seed/publication trace, respectively. The feedback
  index says the latter was not retained; it cannot be replaced by an agent reproduction.

## 8. Cross-slice notes

Data/recipe slice: q99 responses, the 8,774-chart census and 1,972-chart recurrence reference are different
populations; inherited source filters and song weights affect what “ordinary” means. I did not audit training
sampling sufficiently to assign responsibility for native distribution failure.
Architecture/player-response slices: the H floor can establish a timing limitation, while the 25-ms
zero-work acceptance demonstrates a response-to-decision limitation. Neither licenses a global blame
percentage for H versus R/R1. Default planner non-invocation is a separate integration fact.
Realtime slice: preserved publication traces substantiate their declared cache-ready clock only; no
result here establishes fresh process, network or client rendering performance under future contention.

## 9. Failed paths and unfinished work

The initial comparison stopped after 100 real charts because some generated controls omit `ln_fraction`;
the script now preserves that as unspecified. The first-error log is retained. The reuse collector also
needed to handle explicit null references; corrected without dropping those cases. A few guessed source
filenames were absent; `rg --files` located the owning modules. These were review-script issues, not model bugs.

No model-backed reproduction, all-checkpoint hash audit, full 200-commit causal reconstruction or blind
human validation was attempted. Only two Lens images were newly inspected; historical claims of 40 or 209
pages are recorded claims/artifact review records, not this reviewer's image coverage. Source metadata
and 119 osu inputs were available; no requested CPU comparison was blocked by deleted caches.
Aligned policy distributions were not reconstructed, and the final ordinary-expert 84-case semantic audit
remains outside this completed bounded review. I did not change any tracked source, Git state or other report.

Created/changed: this report; `R/compare.py`, `auxiliary.py`, `reuse.py`, `summarize.py`;119 per-chart JSON
reports; manifests, summaries, command logs, selected-test log and commit listing under `R/`.
`R/files-created.txt` lists the exact scratch files and this report. Review complete at the stated evidence
boundary; no outstanding background command or subagent work.

## 10. Inconsistencies and items for the human

1. **A pass means different things in different generations.** StrictHH `<20`, inclusive census `≤20`,
   60/50/50 recovery support and 40/80-ms prevalence are not interchangeable (`checked-code`, buffering:21;
   `checked-artifact`, census and replay). Earlier zero counts can be numerically correct while the intended problem persists.
2. **Real-chart support is not preserved by all “playability” constraints.** `D/joint_action_spacing.md:146`
   reports 60/50/50 excluding relationships in 558/8,774 charts, including 281/2,047 at 4–5 stars (`doc-claim`).
   These were later audited, but an inherited constrained training/evaluation population cannot stand for all ranked arrangements.
3. **Total likelihood and release quality can move oppositely.** Two documented factual examples improve
   total NLL while worsening conditional release NLL (`doc-claim`, `D/ln_release_calibration.md:109`;
   `checked-code`, row_likelihood:17). This disproves that particular inference, not the value of likelihood training.
4. **Response availability is not response enforcement.** Qualification imports/constructs the ordinary
   session, while the optional planner is a separate path (`checked-code`, qualification:17,203).
   The actual 25-ms zero-work commit further shows enforcement alone would not validate the current score (`checked-artifact`).
5. **A broader short-tail count can improve while extreme tails worsen.** All six historical segment
   outputs pass 80 ms but fail 40 ms; all 19 reviewed D4 outputs also pass 80 ms (`checked-artifact`).
   Which remaining arrangements count as unacceptable needs human semantics, not another unlabeled numerical success.
6. **Named “style” gains have weaker semantic authority than the formulation requires.** The trajectory
   kernel is a physical block-distance statistic (`checked-code`, trajectory_kernel:92), while held-out
   ordinal concept recovery and independent generated judgments are required (`doc-claim`, formulation:368).
7. **The late documents are often more cautious than an overall success narrative would imply.**
   `review_required` and `promoted=false` are enforced in code (`checked-code`, qualification:276).
   The human can distinguish dissatisfaction with the research direction from a claim that the final runner falsely certified success.
8. **Ordinary versus expressive acceptance remains a human specification question.** Ranked short-LN
   and positive-excess witnesses invalidate blanket bans, but do not define the preferred ordinary mixture
   (`checked-artifact`, census/positive guards). The reviewed evidence cannot settle that preference or player-response target.
