# Lineage review: 06-controls-ln

Review date: 2026-09-30. Scope: difficulty/style/LN-amount controls and LN allocation/release/fragmentation in the audio-skeleton lineage.

## 1. Scope and sources

The strongest positive result is bounded native LN-amount calibration with explicit feedback. Neither that result nor improved likelihood establishes ordinary LN/TAP organization, semantic style control, or reliable difficulty control across 2–6 stars. The late collapse findings are real, but the segment experiment was too small to support an architectural rejection.

Evidence grades used throughout:

- **doc-claim**: historical document or commit assertion, not independently verified here.
- **checked-code**: implementing source inspected; citations include line numbers and revision.
- **checked-artifact**: saved machine-readable result, ledger, or exported chart inspected. A saved evaluator result is evidence of its measurement, not automatically of playability.
- Judgments are explicitly identified as this review's inference from those grades.

Path abbreviations: `J/` means `artifacts/joint-audio/`; `D/` means `/tmp/lineage-review/trees/audio-joint/docs/research/`; `M/` means `/tmp/lineage-review/trees/main/docs/research/`. Code citations use the lineage-end commit `099cb66` unless an earlier revision is named.

Read the baseline research entry point, README, and sections 2–3 of `.git/research-relay/notes/artifacts/audio-skeleton-human-feedback-index.md`. Its A–K identifiers describe the requested problems; its experiment interpretations remain doc-claims. No private material or Codex session files were opened.

Read the named control/LN research documents, the three pre-lineage style documents, and relevant portions of `typed_audio_continuation.md`, `row_condition_interactions.md`, `common_prefix_outcomes.md`, `outcome_learning_and_control_response.md`, and `action_segment_r1.md`. Inspected the read-only commit timeline and the e04b349/c5b7db8 implementation transitions. No training, model inference, accelerator execution, installs, Git mutation, or source changes were performed.

Principal raw evidence owners:

| ID | Artifact owner and decisive files |
| --- | --- |
| E1 | `J/20260925-scoped-control-target-v1/{joint-1000/result.json,native-1000-summary.json,train.py}` |
| E2 | `J/20260925-ln-allocation-feedback-v1/native-feedback-summary.json` |
| E3 | `J/20260926-control-coverage-v1/{result.json,condition-audit.json,coverage.py}` |
| E4 | `J/20260926-paired-scope-controls-v1/{comparison.json,training-comparison.json,source-96/config.json}` |
| E5 | `J/20260926-balanced-condition-r1-v1/{comparison.json,condition-audit.json,balanced-1200/result.json,aligned-1200/result.json}` |
| E6 | `J/20260927-controller-semantics-v1/{assessment-plan-v2.json,scope-assessment-v2.json,analysis/result.json}` and four on/off arms' `cases.json`/exports |
| E7 | `J/20260927-contextual-ln-fit-v1/{source-plan.json,fit-resumed-v1/result.json}` |
| E8 | `J/20260928-scoped-ln-allocation-v1/{fit-v2/result.json,progress-exposure.json,training-program-coverage.json,native-progress-v2/cases.json}` |
| E9 | `J/20260928-release-support-learning-v1/{fit-summary.json,plan-v2.json}` |
| E10 | `J/20260928-ln-risk-calibration-v1/{risk-decomposition.json,scores-v2/result.json}` |
| E11 | `J/20260928-ln-fragmentation-repair-v1/{ownership-pilot.json,ownership-pilot-v2-plan.json,ownership_pilot_v2.py,pilot-v1/result.json,continuation-v1/result.json,native-80-result.json}` |
| E12 | `J/20260928-optional-control-repair-v1/{probe-v1/result.json,scope-probe-v1/result.json}` |
| E13 | `J/20260928-style-difficulty-response-v1/{native-v1/cases.json,analysis-v1/contrasts.json}` |
| E14 | `J/20260928-action-segment-r1-v1/{source-data.json,fit-plan.json,pilot-v1/initialization.json,pilot-v1/result.json,native-source-H-v1/result.json}` and six exports |
| E15 | `J/20260928-ranked-fourstar-arrangement-v1/{charts.jsonl,summary-v2.json,study.py,correct_summary.py}` |
| E16 | `J/20260928-coordination-corpus-v1/scan-v1/result.json`; `J/20260925-ranked-2to6-reference-v1/result.json` |

Independent CPU reductions are in `/tmp/lineage-review/06-controls-ln/check_evidence.py`, with results in `checked-evidence.json`: six segment exports, nine attribution exports, 32 feedback on/off whole-song exports, the 1,973-chart observation ledger, and 15 optional-control probe records. Export hashes were verified for the six segment and nine attribution charts. The script checks LN ratios and H/R tail classification directly from osu objects; it does not regenerate charts.

`/tmp/lineage-review/06-controls-ln/check_support.py` additionally reran the original support observer on the three parent attribution charts, using the exported lineage code, and exactly reproduced the saved short-tail summaries. Results: `checked-support.json`. This is a legal-support replay, with no neural checkpoint or model forward pass.

Not all named artifacts or historical checkpoints were opened. In particular, early profile results, active-origin cue results, most late release-support native aggregates, and common-prefix/modulation results below remain doc-claims. Existing charts establish geometry, not listening quality or human playability. “Not recorded” below means not recorded in the sources inspected for that row, not a claim that no other file contains it.

## 2. Attempts

**Question 1 — label provenance and granularity.** Numeric labels, community tags, and human section assessments are distinct sources.

| Control / source | Label definition and scope | Available coverage | Evidence |
| --- | --- | --- | --- |
| Difficulty | Training recomputes `compute_mania_star_rating_20241007(objects,4,1.)`, the repository's implementation of the 20241007 mania calculation. Metadata stars also select/filter some populations. This is not a measured player-response target. | Ranked control pool: 6,923 TRAIN charts / 2,573 song groups; metadata bins 2–3: 2,558; 3–4: 2,300; 4–5: 1,611; 5–6: 454. | checked-code: E1 `train.py:91–94`; `099cb66:src/ensomi_model/osu_core/difficulty.py:517`; checked-artifact E3 bins; groups are doc-claim, `D/typed_audio_continuation.md:400`. |
| Scoped difficulty | Initially repeats whole-chart difficulty. From 8687c35, full-source strain is computed first, then local 400-ms peaks are weighted with .9 decay and .018 scaling, normalized for scope duration. Explicitly an approximate local proxy, not official local stars. | 61,446 eligible 16-second windows with ≥32 H events in E3. Early relabel audit: 256 charts/groups, 2,940 scopes, 24.3% >1 unit below chart label (doc-claim). | checked-code: `099cb66:src/ensomi_model/research/typed_audio_continuation/difficulty_targets.py:23`, `:68`; E3 checked-artifact; relabel audit in `D/typed_audio_continuation.md:554`. |
| LN amount | Count LN starts / (TAP starts + LN starts) in the requested half-open range. Does not count occupancy-time, tails, or “LN coordination.” Whole-song and 8/16/32/64-second variants occur. Empty-head scopes yield unknown, not zero. | Computable for all valid source charts; no special annotation population. Known-LN exposure varies with dropout and recipe. | checked-code: `099cb66:src/ensomi_model/research/typed_audio_continuation/controls.py:103–120`. |
| Semantic style | Human `assessment-cohort.jsonl` cells, preserving original time scopes; absent/supporting/prominent map to −1/0/+1. Missing/conflicting assessments remain unknown. Controls use Jack, Stream, Trill, Tech, LN coordination; concepts can coexist. | E3: 289 human cells on 112 TRAIN charts. Per-concept counts below. | checked-code: E1 `train.py:53–70`; checked-artifact E3. |
| Community tags | osu user-voted `top_tag_ids` joined to the August 7 tag catalogue; whole-difficulty weak evidence, not local targets. | 14,787 local 4K difficulties; 2,688 have style tags; chordjack 833, jumpstream 736, LN coordination 497, longjack 190. | doc-claim: `M/osu_mania_4k_style_tag_reference.md`. The inspected control sampler uses human cells, not blanket community-tag propagation. |
| Shared profiles | Sixteen medoids of whole-chart H rate, heads/H, and LN fraction, song-group weighted. Neither stars nor semantic styles. | 615 TRAIN arrangements / 240 audio groups in the profile comparison. | doc-claim: `D/shared_arrangement_profiles.md:9–30,144–153`. |

Human control coverage from E3 (`cells/charts`; each cell category happens to have one chart per cell):

| Concept | Absent | Supporting | Prominent |
| --- | ---: | ---: | ---: |
| Jack | 14/14 | 21/21 | 11/11 |
| Stream | 7/7 | 18/18 | 21/21 |
| Trill | 35/35 | 14/14 | 13/13 |
| Tech | 49/49 | 18/18 | 5/5 |
| LN coordination | 40/40 | 15/15 | 8/8 |

Charts overlap across concepts. All five prominent Tech cells have local difficulty in 3–4; prominent LN coordination has four in 2–3, two in 3–4, two in 4–5. Balancing cannot supply missing support at the other difficulties. **checked-artifact**, E3.

Whole-chart labels can supervise an aggregate target, but do not identify which internal range should express it. The lineage recognized and partially repaired this: source difficulty became scoped, human style spans stayed scoped, and per-field clocks replaced a shared scope clock. None of those representation fixes proves that a local intervention is learned. **Inference, high confidence**, from the checked code and E1/E3.

The pre-lineage warning was already concrete. `M/scoped_style_seed17_results_and_next_questions.md` reported 2,701 updates/43,096 draws per arm, 3,526 machine training cells, 79 human-validation cells/26 groups: lower machine NLL did not repair human Trill/Tech positive recall (zero). `M/scoped_style_probe_postmortem.md` then used 3,200 training cells (382 human), 61 primary human-validation cells/18 groups, seed 17 only; B/C ran 272/423 updates, not to established convergence. Better readout loss still did not repair human Trill/LN detection or Jack/Stream selectivity. These are **doc-claims**, not rerun here. They warn against using that classifier as an already validated semantic control evaluator; they do not prove that all whole-chart conditioning is useless.

Chronological inventory follows. Changes are grouped by experiment, rather than treating every documentation commit as a new attempt. A/B/C/F/H/I/J/K refer to the complaint index. P denotes parameter count; n denotes independent sources/cases, not event queries. Unless stated otherwise, no between-training-seed effect estimate was recorded.

| Attempt / commits | Hypothesis and change; problem IDs | Training budget and exposure | Evaluation/result, interpretation and next step |
| --- | --- | --- | --- |
| 09-24 separate R and full-held conditioning: e04b349, 12d80eb, 7c316e6 | Learn release-event times using LN ages/occupancy and H preview; condition a finite release wait consistently. B/F. | Steps, wall time, P, charts/songs, fit seeds: not recorded here. | Separate clock is checked-code below; improvement numbers not audited. Release feasibility later moved through R1 windows. No duration-object representation introduced. |
| 09-24 shared profiles: 10ddaa8, d80de49; routing/count follow-ups a2bce68, 4567d87 | Make H/R/rows share a chart-level arrangement request. I/B. | 1,200 updates per arm; 964.4/920.7 s; P=4,250,174 conditioned, +2,736; 615 charts/240 audio groups; fit seed not recorded here. | Five audios/nine fixed audio-seed cases, four explicit medoids plus two automatic arms=54 outputs; standardized descriptor error 13.2592→7.5809. Cross-coupling and near-tail attacks persisted. Routing removal later worsened native control; no qualified profile solution. **doc-claim**, `D/shared_arrangement_profiles.md`, `profile_density_routing_evaluation.md` link. |
| 09-25 typed controls and explicit LN law: ca3dc65, 27b9526, 929138b | Typed planner chooses TAP/LN counts/release identities, R1 materializes lanes; explicit binomial LN base counters weak neural conditioning. I/B/F. | 1,200 updates/413 s +400/147 s; P=3,895,879; 614 TRAIN/36 VAL; songs and fit seed not recorded here. | Three audios × five requests=15 outputs/arm; strong LN response appeared but difficulty worsened. Both factorization and extra training changed. Led to timing-base repair and broader fitting. **doc-claim**, `D/typed_audio_continuation.md:180–227`. |
| 09-25 repaired/broader typed fit: 4adcdd9, 4142ba4, 2e7e7bb | Preserve unconstrained local LN preferences, improve coverage and scoped evaluation. I/B/H. | P=3,947,227; repaired 1,200/451 s; broad 6,000/2,813 s, 4,641 charts actually drawn from 6,923/2,573 groups; 36 VAL. One fit seed, ID not recorded here. | Eight audios/40 outputs each at 2,400 and 6,000 steps: star MAE .553→.959, ≤40-ms LN 1.72%→3.18%; later lower-loss endpoint rejected. Seven-case style probe showed response, not semantic calibration. **doc-claim**, same document:398–504. |
| 09-25 local difficulty targets: 8687c35, 466e21f | Replace chart stars at local queries with complete-prefix scoped strain. I/H. | 1,000 updates/511.819 s; P=3,947,227; 1,395 drawn charts; same 6,923 pool; seed 251928. Drawn songs not recorded. | Fifteen native outputs/three audios, one draw per request. Static star MAE .488→1.644; LN MAE .0777→.1018; ≤40-ms LN 2.05%→9.38%. Rejected despite override scalar improvement. **checked-artifact** E1 endpoint; baseline comparison **doc-claim**. |
| 09-25 bounded allocation feedback: 466d470, 8cbe86a | Correct cumulative LN surplus/deficit by bounded odds tilt. I/B. | No training; same retained 2,400-step model and P. | Same 15-case panel: static LN MAE .0777→.0380; star MAE .488→.580. Numeric amount improved, local durations/Jack control did not. **checked-artifact** E2 endpoint; comparator in document. |
| 09-25 R1 ownership and integral feedback: c5b7db8, bbd1a7f, 298c17c | Return full row/count/TAP-LN choices to R1; coordinate release feasibility, apply amount feedback inside R1 families. F/I/B. | Core 2,500 updates; P/wall time/actual source counts/fit seed not recorded in inspected account. Optional mean-head predictor: 2,000 separate updates. | Three audio-seed pairs per LN cell: .2/.7 errors .0017/.0089 with integral feedback, but star MAE 1.0559/1.4433. Scoped .7 overrides .7410/.7225/.7126. Useful amount mechanism; no general qualification. **doc-claim**, `D/controlled_audio_continuation.md:333–386`. |
| 09-26 active LN audio origins: 2662c17, c1260d3, 2b20c4f | Carry birth audio into R and R1, so tails can relate to an ongoing role. B/E. | 1,500 updates/arm, 3,000 intervals/1,889 charts; cue/ordinary 841.7/789.7 s; +73,920 P; songs/fit-seed ID not recorded here. | Five source-H songs, one seed each, ten outputs. One LN case ≤80-ms share 4.53%→3.38%; source-like coincident tails 21→71 against source 73 in a selected range. Sustained role still failed elsewhere. **doc-claim**, `D/active_ln_audio_cues.md`. |
| 09-26 whole-chart outcomes then paired scopes: 0c057af, 2c2b8af, 5bf146e | Optimize completed difficulty/LN outcomes rather than source loss alone. I/H. | Whole-chart: P=4,583,985, trainable 2,737,331; 128 updates; 12 charts/distinct audios; 27.64/1,065.57 s. Paired: 96 updates; trainable 12,288; 12 charts/24 scopes; 36.26/635.69 s; seed 261270. | Whole-chart seven source-H cases/arm worsened singles MAE 1.272→1.337 (**doc-claim**). Paired five prefixes × two requests × three seeds/arm: MAE .80865→.75583; contrast .16037→.25390, failed gates. **checked-artifact** E4; led to condition audit/balancing. |
| 09-26 balanced condition alignment: b2a5351, 6520ba1 | Use same-H alternative arrangements and correct-versus-swapped conditions. I/K. | 1,200 updates/arm; 819.02/906.02 s; 703 charts/2,400 draws; 128 paired-audio pool +1,024 population scopes +251 human cells; trainable 3,187,773; seed 261410; actual distinct songs not recorded here. | Source correct-condition ranking 14/24→20/24; generated five-prefix × two-request × three-seed MAE .78434→.71638, missed .20 improvement gate. Correct inference was better condition discrimination, insufficient generation. **checked-artifact** E5; history/modulation work followed. |
| 09-26 conditional modulation and common-prefix outcomes: 24fc4f1, 0591f90, a99519c | Make request interact with history in main row head; then train difficulty on reachable alternative continuations. I/A/H. | Modulation +16,384 P, 128 updates/1,209 s, 384 source examples. Common-prefix: P≈4.60M, trainable 2,753,715; 47 fitting pairs/eight reserved; 128 updates, 116/1,184 s. Fit seeds/unique chart totals not recorded here. | Modulation style-kernel distance .06534→.04953; this is not semantic accuracy. Common-prefix MAE .59182→.38894 versus source-only, but gap MAE .53646→.68529. Three-audio native static star MAE .84666, LN MAE .02165. Continued outcome fits regressed. **doc-claim**, `D/row_condition_interactions.md`, `common_prefix_outcomes.md`. |
| 09-27 feedback semantics ablation: 24061b9 | Test whether amount feedback causes unnatural local uniformity, and retain whole-scope semantics. I/B/C. | No fitting: actor128 and memory384, each on/off. Inherited training/P outside this ablation. | Four songs × two seeds + one live override × four arms=36 outputs; all 18 H pairs equal. Actor STYX .504/.496 on versus .939/.933 off. Feedback helps amount but can oppose legitimate TAP phrases. **checked-artifact** E6; contextual/progress alternatives followed. |
| 09-27 contextual LN count: 1434924, 80bbde1 | Learn count tilt using row/audio/hold context instead of a static reference tilt. I/B. | 128 updates, two arms; 256 windows/224 charts/214 groups; P=4,600,369, trainable 143,011; seed 279611; 264.00 s aborted +121.83 s resumed work. | 22 validation windows; 11 native cases × four endpoints=44 outputs. NLL improves similarly; Classic LN .145/.191→.048/.019, STYX .939/.933→.223/.439, Blizzard stays .989/.989. Native quality not qualified. Budgets **checked-artifact** E7; response figures **doc-claim**, `D/contextual_ln_count_learning.md`. |
| 09-27/28 learned scoped progress: c0db49c, 3591df2 | Add factual count progress, declared and effective ownership clocks. I/B. | 256 updates/344.93 s; 512 windows/425 charts/392 groups; frozen P=4,600,369, trainable +69,953; seed 280031; 22 VAL windows. | Two endpoints ×20 native cases, two seeds on central songs. LN MAE benefit .054679 vs parent, only .015759 vs context-only; concentrated in STYX seed 1. Override .6 still yields .897163. **checked-artifact** E8 budgets/coverage; comparison values **doc-claim**, `D/scoped_ln_allocation.md`. |
| 09-28 release-support joint learning: 784ac6f | Restore valid source release support; train R/R1 versus full joint audio/H/R/R1. B/I/K. | 512 updates, 1,338.75 s combined; 1,024 windows/736 charts/661 groups; trainable P=3,274,110 /4,670,322; seed 280033. | 22 validation windows; original 20 native cases +eight D2/D6 cases per arm. Six LN-request MAE .099117→.061129 R/R1, .089384 joint; new amount failures and long-jack regression. Not promoted. Budget **checked-artifact** E9; results **doc-claim**, `D/release_support_joint_learning.md`. |
| 09-28 release conditional calibration: ef90a21, 5b0dbeb, dc50ce2 | Decompose H release risks, release cardinality/identity, and whole-row likelihood. B/H. | Diagnostic only; inherited/early/fresh at initial, 32, 512 steps; eight selected TRAIN scopes. Those checkpoints' training exposure/P not re-audited here. | Source Shizuku predicts 57.94 releases vs 65, yet allocates 17.09 releases to 32 continued risks; expected simultaneous pairs 9.84 vs22. Aggregate totals hide relationship errors. **checked-artifact** E10; method checked-code below. |
| 09-28 joint wait/release + cues: 965d670, a6c912f | Score wait and release subsets together before R time is selected. B/F/H. | Eight flow-calibration examples; 16 updates/32 new factual examples/91.34 s, then64 updates/321.31 s to 80. P=4,711,570; seed280929; charts/songs not recorded here. | Three same-seed native cases, one per song: B80 passes while Blizzard amount and Stream difficulty fail. New cues, law, loss weight and joint fitting changed together. Not a single-factor causal experiment. **checked-artifact** E11 budgets/exports; interpretation `D/joint_r1_release_decisions.md`. |
| 09-28 optional conditions and style/difficulty probes: 1658123 | Remove leaked known LN amount from style-known training views; inspect scope and numeric interactions. I/C. | No matched repair-training result found. Probe:14 distinct source intervals/10 charts,15 effective ranges; one existing checkpoint. Native diagnostic:two songs ×two seeds ×three views=12 outputs. | Hiding LN changes factual expected fraction by median +.002579, max+.02536; inadequate as sole explanation of native collapse. Stream D4 outputs Zenithfall5.071/5.134; lowering neural D by1 gives4.388/4.586 without demonstrated style repair. **checked-artifact** E12/E13. |
| 09-28 action segments: 7d31b1e, 9c6d531, e4c4453 | Persistent four-second plan code and LN birth context, joint R/R1 segment likelihood; later deterministic context path. B/C/I. | 32 updates/146.18 s for K1/K4; 128 source examples,126 charts/126 groups,158 segments/3,960 rows; seed280930. Trainable/frozen P K1=1,963,814/3,917,438; K4=1,964,777/3,917,438. | Three source-H songs ×one matched seed ×two arms=six outputs; ratios24.51–30.26%, all 85 K4 choices code1. Added continuous context is implemented, not established by these earlier results. **checked-artifact** E14 and independent reduction. |

**Question 2 — requested versus achieved controls.** The following separates controls and history regimes. `native` means generated H and rows from BOS; `source-H/native rows` still supplies real head times; `source` means factual histories. Multiple requests for one song are not independent songs.

| Attempt / control | Request | Achieved | n and history; grade |
| --- | --- | --- | --- |
| Profile / LN descriptor | First four TRAIN medoids from a 16-profile bank | LN-response coefficient .177; standardized three-descriptor error13.2592→7.5809 | Five audios/nine seed cases; native; doc-claim. No isolated star/style target. |
| Typed first / difficulty | D3/D5, LN.2 | Hysteric3.010/4.729; Zenithfall4.625/5.080; As It Was2.574/2.972 | Three audios, one draw/cell; native; doc-claim. |
| Typed explicit base / LN | .2/.7 at D3 | Zenithfall.085/.748; Hysteric.102/.791; As It Was.171/.739 | Same three-audio panel; native; doc-claim. |
| Typed style probe / style | Jack/Stream/Tech absent versus prominent, D4/LN.2 in32s | Jack repeated masks3→12; Stream visual motion changes; Tech inconclusive | Seven cases; six paired contexts; native; doc-claim; no semantic tolerance. |
| Local-label fit / difficulty and LN | D3/.2 → D5/.7 on[105000,137000) → D3/.2 | Zenithfall D4.013/5.145/5.250 and LN.265/.880/.182; Hysteric D4.279/5.056/5.083 and LN.147/.901/.138 | Three audios, one switched draw each; native; checked-artifact E1. |
| Typed feedback / LN | .2/.7, D3/D5 static; .7 override | Static MAE.0380; overrides .7485/.8144/.7994; restored .1510/.1547/.1365 | 12 static+three switch outputs/three audios; native; checked-artifact E2. |
| R1 integral feedback / LN | .2/.7, D3 | Mean errors.0017/.0089; three .7 overrides .7410/.7225/.7126 | Three audio-seed pairs/cell; native; doc-claim. |
| Paired scopes / difficulty | Low/high1.5 apart; fast singles2.190/3.690 |3.738/4.354; slower chords2/3.5→2.171/2.278; MAE.75583 overall | Five prefixes ×three seeds/request; source-H/native histories; E4 checked-artifact aggregate, per-case document. |
| Balanced alignment / difficulty | Local low/high1.5 apart | MAE.71638; mean spread.31325 versus requested1.5 | Five prefixes ×three seeds/request; source-H/native; E5 checked-artifact. |
| Balanced native switch / D,LN | D3/.2→D4.5/.6→D3/.2 | Zenithfall D3.024/3.650/3.690, LN.199/.631/.185; Take D4.009/4.178/4.184,LN.185/.774/.222 | Three audios ×one switched seed; native; doc-claim `D/balanced_condition_alignment.md:194–230`. |
| Modulation / style | Four reserved source-style scopes | Kernel distance.06534→.04953, meeting relative proxy gates |12 source-H/native style outputs, three seeds/scope; doc-claim; no semantic accuracy. |
| Common-prefix / D,LN | Static D3/.2;32s D4.5/.6 override | Static MAE D.84666/LN.02165; override D.57158/LN.04278 | Three audios, one seed/mode; native; doc-claim. |
| Feedback ablation / LN | Max.042931, Classic.217153, STYX.485281, Blizzard.838046 | Actor on: .0330/.0226; .2253/.2147; .5045/.4958; .9409/.9460. Memory on: .0209/.0214; .2056/.2117; .4805/.4858; .8517/.8490 | Four songs/two seeds each; native; E6 exports independently counted. |
| Feedback ablation / difficulty | D4 for same eight outputs | Actor on3.850–4.473; memory on3.366–5.123 | Same seeds273100/101,273110/111,273120/121,273130/131; native; checked-artifact E6. |
| Contextual count / LN | Classic.217, STYX.485, Blizzard.838 | Fitted both arms approximately .048/.019, .223/.439, .989/.989 respectively | Three songs/two seeds, within11-case panel; native; doc-claim. |
| Learned progress / LN | .217153/.485281/.838046 | .319927/.390197; .448020/.538945; .962733/.941320 | Same three songs/two seeds; native; E8/document. |
| Learned progress / D,LN override | D4.5/.6 on[64000,96000) | D3.285719/LN.897163 | One switched case; native; doc-claim. |
| Release-support / D | D2/D6 | Only joint Classic seed0 met both ±1 gates; joint Zenithfall D2=4.4602/4.1993 | Two songs/two seeds/request; native; doc-claim; not a 2–6 control success. |
| Joint-release16 / LN,D | STYX D4/.4853; Blizzard D4/.8380; Stream D4/LN unknown | .4690/3.53; .2871/4.05; .5115/5.79 | Three songs/one seed each; native; E11 exports +document stars. |
| Joint-release80 / LN,D | Same three requests after 64 more updates | STYX LN.6875/D3.691; Blizzard LN.6451/D4.359; Stream LN.4245/D5.825; all B80 pass but both explicit amount gates fail | Same three native songs/seeds; checked-artifact E11 `native-80-result.json`. |
| Optional views / LN missingness | Same factual Stream conditions, known→hidden LN | Expected fraction delta−.004447 to+.025360; median +.002579 |15 ranges/14 intervals/10 charts; source; E12 independently reduced. |
| Late style probe / D,style | D4 Stream prominent; LN unspecified | Zenithfall5.071/5.134, Classic4.366/4.360. Stream-minus-unspecified star changes .153/−.073 and−.084/−.093 | Two songs/two seeds; native; E13. Style itself has no validated achieved ordinal value. |
| Segments K1/K4 / D,LN | STYX4/.485281; Kimi4/.130778; Celestial4/.074850 | STYX D4.467/4.624,LN.27948/.24511; Kimi D4.038/3.973,LN.29017/.30260; Celestial D5.045/5.059,LN.29544/.29658 | Three songs, seeds281100/281110/281120 paired across arms; source-H/native rows; E14 +export checks. |

The “25–30% regardless of request” shorthand is approximately correct for those six outputs (actual range24.51–30.26%). It is not an intra-song sweep over different LN requests: each song received its own source fraction. All85 K4 selections were code1, as logged in generation metrics. This is a collapsed latent plan choice, not the separate integral LN controller. **checked-artifact**, E14; this distinction corrects an easy ambiguity in Question2.

**Question 3 — did a control ever work within a stated tolerance?** Yes, narrowly. E6 declares ±.10 LN fraction and ±1 difficulty gates. Independently counted native whole-song outputs pass LN in 8/8 memory-on cases and 6/8 actor-on cases, versus 2/8 and 4/8 off. Actor-on difficulty passes 8/8 at the single requested D4; memory-on passes 5/8. These demonstrate finite-panel amount calibration and D4 agreement, not a difficulty response curve or all-controls success. E6 still marks the candidates failed; style/playability are separate.

No inspected result establishes semantic style control within a validated ordinal tolerance. Relative kernel/style-distance gates, changed row counts, source-label likelihood, and visual anecdotes are not such a tolerance. No general native 2–6 difficulty guarantee is established; early D3/D5 and late D2/D6 counterexamples directly limit it. **Inference, high confidence**, from E4–E6/E13 and the explicitly bounded documents.

## 3. Commentary

**Question 4 — LN representation.** All successful code paths examined here represent an ongoing hold as a start plus a later release action. Action segments add a shared decision variable, not a duration attached once at LN birth.

| Form and transition | What is explicit | What it makes easier; what remains hard for B | Evidence |
| --- | --- | --- | --- |
| R1 baseline / first skeleton bridge | Complete lane row actions with replayed occupancy; supplied or learned event times, release eventually closes an open start | Exact causal replay, overlapping holds, simultaneous/subset releases; no learned commitment to a future held role from the birth alone | Baseline entry point is doc-claim; lineage row vocabulary reused in checked `typed_audio_continuation/program.py:11–20`. |
| Planned H/R/R1, e04b349 | H proposes head times; independent R hazard selects release-only time; R1 chooses complete row/identities. H rows can also release | Separates clock and lane decisions and supports millisecond release placement. At a pure-R event with one held LN, R1 cannot choose “continue”; normalizing one nonempty mark erases its preference | checked-code: `e04b349:src/ensomi_model/research/planned_audio_continuation/generation.py:128–165`; E11 replay. |
| Typed planner, ca3dc65 | Marks `(tap count, LN count, release mask)` on four abstract resources; starts remain open until mask release | Exact occupancy/count feasibility and direct amount conditioning; duration/role coherence still emergent, while upstream counts constrain R1 choices | checked-code: `099cb66:src/ensomi_model/research/typed_audio_continuation/program.py:14–20,33–74`. |
| Controlled R1, c5b7db8/bbd1a7f | R1 resumes all complete-row decisions; R1 feasibility window constrains R; separate pure-R hazard persists | Repairs ownership and future-head feasibility; amount tilt still alters births without supplying lifetime or relational intent | checked-code: `099cb66:src/ensomi_model/research/controlled_audio_continuation/generation.py:56–92`; historical diff c5b7db8. |
| Joint wait/release,965d670 | R1 scores empty wait plus feasible nonempty release subsets; hazard=`logsumexp(release)-wait+flow scale`; wait is not a committed row | Fixes the singleton pure-R decision bottleneck. Still must learn when a held role should persist; more long holds can create deadline pressure/occupied-finger constraints | checked-code: `099cb66:src/ensomi_model/research/controlled_audio_continuation/joint_release.py:46–75,79–124`. |
| Action segments,7d31b1e | Persistent code for ≤4s, exact mixture likelihood over joint R/R1 traces; factual birth row/audio remains attached to active holds across plans | Creates a path for shared temporal organization; code collapse blocks that path, and existing per-event decoder can still emit 21ms holds. No duration-object guarantee | checked-code: `099cb66:src/ensomi_model/research/segment_audio_continuation/model.py:88–109`; `segments.py:138–159`; `generation.py:27–45`. |

This representation makes exact online continuation straightforward. It also leaves musical release grouping, persistence through later H events, and TAP companionship to a sequence policy. That is a challenge, not proof that event representation cannot learn them. The lineage's late segment pilot did not test a converged duration-object alternative. **Inference, medium confidence**.

**Question 5 — attribution of short tails.** E11's observer reconstructs rows from exported objects, classifies each release time as H if any head occurs at that time, otherwise R, replays exact LN starts, and defines short as≤80ms. It enumerates legal rows under 60/25/21 recovery and fixed H preview, including alternatives that keep the same current head signature while retaining each released LN. It separately checks whether an R event equals a nonterminal forced deadline. Implementation: E11 `ownership_pilot_v2.py:23–90` (**checked-code**).

| Parent inherited-2048 output | Seed | All LNs | ≤80ms LNs | At H / pure R | H shorts with legal keep, same heads | Nonterminal deadline shorts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stream Zenithfall, D4/LN unknown |271201|4,382|2,670|2,524 /146|2,524|0|
| STYX, D4/LN.485281 |273121|758|84|13 /71|13|0|
| Blizzard, D4/LN.838046 |273131|824|71|50 /21|50|0|

All three rows were independently recomputed from chart objects and through the original support observer; nine trajectories across parent/pilot/4096 were checked for duration and H/R classification. The headline attribution is two contrasting charts; the available parent census is three charts, one seed each. **checked-artifact** E11 and both scratch outputs.

The method is sound for proximate decision ownership. Stream's 94.53% H share means a purely delayed R clock cannot eliminate those tails with that same birth/head behavior. STYX's 84.52% pure-R share motivates allowing wait before selecting R. Neither percentage is an estimate across songs or a causal allocation of blame to trained modules. Legal retention of each LN separately is not a demonstration that retaining all of them simultaneously, or for a musically useful duration, leads to a good continuation. Earlier H density, births, occupancy, and learned-history drift remain possible causes. **Inference, high confidence**.

**Question 6 — ranked3.5–4.5-star LN.** E15 contains 1,973 ranked TRAIN charts/1,614 song groups, selected using recomputed20241007 stars. It has 3,430,578 heads and 628,071 LN starts: pooled head share 18.31%; song-equal median chart fraction 13.62%. These two weightings answer different questions. The independent reduction reproduced counts and pooled distributions from all 1,973 ledger records.

| Source stratum | Charts / groups | Song-equal chart share | LNs | LN duration median [Q25,Q75],ms | Release→next same-column head median [Q25,Q75],ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| No LN |134 /110|6.40%|0|Undefined|Undefined|
| LN fraction(0,.2) |1,140 /962|57.65%|165,601|196 [150,337]|242 [158,375]|
| LN fraction[.2,.6) |636 /564|32.89%|387,033|162 [98,250]|183 [125,319]|
| LN fraction[.6,1] |63 /57|3.06%|75,437|132 [94,220]|167 [98,261]|
| All eligible LN charts |1,839 /1,517|—|628,071|176 [135,316]|214 [150,343]|

These quantiles give each eligible song group equal weight, then its charts, then relevant objects; group counts across strata overlap. RH opportunities exist in 1,831 charts/1,512 groups. **checked-artifact**, E15 `summary-v2.json`; weighting implementation inspected in `study.py:118–119` and `correct_summary.py:18–29`.

Pooled-object duration Q10/Q25/Q50/Q75/Q90=82/100/164/258/370ms; corresponding same-column RH quantiles=86/118/183/321/484ms over 624,956 opportunities. The source song-equal mean≤80ms-LN share among LN is 7.04%;≤40ms RH share is .00726%. These descriptive tails are not hard exclusion rules. **checked-artifact**, E15 and independent reduction.

| Generated parent chart, same three E11 cases | LN share | Duration median [Q25,Q75],ms | Same-column RH median [Q25,Q75],ms | ≤80ms / all LN |
| --- | ---: | ---: | ---: | ---: |
| STYX |74.17%|130 [114,234.75]|171.5 [121,258.75]|11.08%|
| Blizzard |66.03%|93 [87,171]|177 [94,267]|8.62%|
| Stream Zenithfall |90.67%|73 [60,98]|160 [119,250]|60.93%|

This is a descriptive comparison, not source-matched causal estimation; Stream has unspecified amount and a source above4stars. The strong failure is its mass of short, LN-dominated heads, even though median RH does not look as extreme. “Release near any H” is a different statistic from same-finger recovery; do not interchange them. **checked-artifact**, independent export parsing.

E16 supplies broader context: the09-25 reference parsed 8,774 verified ranked2–6 charts/3,387 beatmapsets; pooled LN durations median168ms and RH208ms. The09-28 coordination scan covers 6,924 TRAIN charts/125,590 windows; it is a relationship search pool, not 1,973 additional independent four-star charts. Its pre-support population differs by one chart from the 6,923 control pool. **checked-artifact**, E16.

`D/ordinary_fourstar_rhythm_and_holds_zh.md` gives useful contrasting examples: long anchors with companion TAP, common tails from staggered births, and legitimate88/89ms or52/53ms LN groups. These remain **doc-claim** visual readings here; the mechanical population does establish that “ordinary” is predominantly low-LN by chart share, not that every ordinary LN must last seconds or that all short LN are bad.

The factual release evaluator explicitly distinguishes reference actions from policy-generated actions and conditions pure-R marks on an already selected event time. Its per-hold product follows factual prefixes, not the free-running lifetime distribution; that prevents a stronger causal interpretation of the calibration scores (**checked-code**, `099cb66:src/ensomi_model/research/gameplay_evaluation/release_calibration.py:44–61`).

**Question 7 — regression suite and metrics.** There were repeated panels, but no single unchanged lineage-wide suite. The repeated three-audio15-case panel, five-prefix/three-seed paired panel, four-song/two-seed feedback panel, later20-case native plan, and late three-song single-seed segment panel answer different questions. Static/switch seeds and older publication-relative override starts sometimes differed. Treating their averages as one learning curve would be invalid. **checked-artifact** E1/E4–E6/E14; timing caveat doc-claim in `D/typed_audio_continuation.md:479–491`.

| Claimed gain | Metric actually used | What it cannot establish |
| --- | --- | --- |
| Shared profiles; typed/control allocation | Standardized descriptor distance; LN fraction MAE; whole-star/scoped-strain MAE; response slope | Semantic style, coherent lifetime/companion roles, or local physical demand |
| Source condition alignment | Correct-versus-swapped row log probability; row NLL | Native stability or attainable requested difficulty from generated prefixes |
| Outcome/control modulation | Scalar outcome error/deadbands; high-minus-low contrast; short-block trajectory-kernel U-statistic | Semantic requested strength; causal benefit on native panels lacking source-only comparator |
| Active LN cues | Tail-on-H counts, nearest-H offsets, duration distribution | Complete hold organization or audio causality across songs |
| Feedback/context/progress | Amount error; row NLL; H parity; prefix offsets/derivatives; LN H-span | Scope-total meaning alone, recovered role structure, or robustness to large late errors |
| Release-support/joint release | H/R/row NLL; control MAE; B80 short-LN heads/all heads; duration medians; deadline counts | Ordinary arrangement; B80 can improve simply by losing LNs or shifting≤80ms mass toward≤40ms |
| Conditional release calibration | Brier scores, joint release subsets, NLL split by heads/cardinality/identity | Free-running survival or independent playability verdict |
| Late segments/acceptance | B40 plus B80, continuing holds/TAP relations, response-work budgets, code usage, service deadlines | A converged architecture comparison or general semantic control |

The late B40 addition matters: `D/r1_short_hold_acceptance_zh.md` reports all six segment outputs passing B80 but failing B40; source STYX has 110≤80ms LNs and zero≤40ms, while K1 has 65≤80ms including 30≤40ms. I independently counted K1's30≤40ms exported holds. The declared gate outcomes remain doc-claims because the full response-blindspot evaluator was not rerun. This is evidence of an observer blind spot, not evidence that the agent promoted those six outputs.

## 4. Direction

**Question 8 — was this the right order?** The best case for early controls is substantial. Requested difficulty and LN amount are needed to define which source outputs should be compared; averaging arbitrary corpus arrangements can obscure ordinary 2–6-star behavior. The user explicitly requested controls on09-25 and balancing on09-26 (feedback index I). Condition audits found measurable source-history bias, so investigating controls was not merely avoiding native failures. Explicit LN feedback demonstrably corrected large errors, and genuine same-audio/H alternatives supplied useful discriminative evidence. **Inference from checked artifacts**, high confidence on the rationale, not on system success.

The best case against the order is that most available “control success” measurements could be met while the proposal remained structurally wrong. Whole-scope LN share cannot specify sustained roles; approximate star error cannot distinguish Stream from a long jack. Sparse semantic labels and a previously failing readout did not support a broad style-control claim. Learned progress was then trained almost entirely on already-correct factual prefixes, with no overlapping LN requests, while deployment asked for correction on generated histories and live overrides. These are concrete target/exposure gaps, not a generic objection to conditional modeling. **Inference from E3/E6/E8/E14**, high confidence.

My call: implementing scoped inputs, preserving their semantics, and auditing factual/native response was reasonable; treating repeated count/condition refinements as the principal path to ordinary native playability was not supported. A narrow mechanism was repeatedly improved before the system had established the arrangement distribution those mechanisms were meant to control. This supports the “nearest-module local optimum” concern, but only partly: the lineage also broadened data, tested native failures, rejected endpoints, diagnosed ownership, and finally changed the sequence unit. Confidence: **medium-high**.

For explicit feedback, the strongest positive case is the fixed-weight native ablation: it really controls amount. The strongest negative case is the source-valid TAP-prefix witness and the remaining fragmented output. Call: keep the measured mechanism and its limits; do not equate amount correction with LN organization. Confidence: **high**.

For learned context/progress and condition alignment, the positive case is preserved genuine histories, matched draws, held-out phase panels, and nonzero measured sensitivity. The negative case is small native panels, little recovery-state exposure, reused development songs, and tiny gains compared with seed-specific variation. Call: useful diagnostics, insufficient evidence of a successful learned controller. Confidence: **high** on observed qualification failure, **low** on ultimate architectural merit.

For joint wait/release, the positive case is a verified decision bottleneck and direct ownership replay. The negative case is a bundled 16/80-update intervention that also changed cues, losses, audio/H and learned state, and often reduced LN mass drastically. Call: the probability-law repair is well motivated; its musical benefit and long-run tradeoffs remain unestablished. Confidence: **high** on mechanism, **medium** on direction.

For action segments, the positive case is a concrete shared-time context path with exact likelihood and preserved holds. The negative case is32 updates,126 charts, one seed/song, only one true first-row training segment (doc-claim), and code collapse. Call: failure of this pilot, not rejection of segment planning or proof that duration objects are required. Confidence: **high** on that evidential limit.

## 5. What was overlooked or never questioned

**Question 9 — ranked by consequence.** “Never” below means not demonstrated or settled in the inspected lineage evidence; several issues were eventually named, so they should not be called literally unasked.

1. **What must a successful control preserve besides its scalar?** Amount changes alter occupancy, companions, release grouping and difficulty. E6 can improve amount while a legitimate source TAP phrase is tilted toward LN; E11 can pass B80 while missing amount/style. No joint human-validated acceptance condition for ordinary organization was established. This was eventually acknowledged, not solved. **checked-artifact/code plus inference, high confidence**.
2. **What distribution should unspecified controls select?** The clean-joint8192-draw recipe had zero effective prominent-Stream pieces with LN unknown, although deployment requested that combination (`D/optional_control_training.md`, doc-claim). The factual hidden-LN probe is small (+.26percentage points median), so the coverage defect is real but insufficient as the sole causal explanation. A validated native conditional-marginal interpretation was never demonstrated. **checked-artifact E12 +doc-claim ledger census**.
3. **Where does correction-state supervision come from?** E8 has zero overlapping LN-bearing programs and only 28/11,398 later-half queries with absolute fraction error>.2. Thus progress supervision mostly teaches consistency on genuine arrangements, not recovery from a substantial generated surplus. Implementing clocks/counters did not create that evidence. **checked-artifact E8**, high confidence.
4. **What is a meaningful scoped difficulty unit?** The implemented proxy uses actual future LN endpoints offline and duration-normalized strain peaks; it is not the undefined V3 player response. E1's improved override scalar came with larger preceding/restored errors. No validation that a ±1 proxy tolerance matches the desired player experience was found. **checked-code/E1 +inference**, high confidence.
5. **How much native failure is inherited versus caused by the new control?** Reused songs, sparse independent training seeds, and interventions changing law+loss+cue+optimizer together leave that attribution open. The separate-H replay is diagnostic but cannot validate a full audio generator. The late 32-update segment result especially cannot identify a capacity or representation limit. **checked-artifact E4/E11/E14**, high confidence.
6. **Does “ordinary” mean the natural ranked mixture or a curated subset?** E15 shows low-LN charts dominate but legitimate short-LN organization exists. Rebalancing rare LN/style cases can change unspecified-control priors. Neither all-short-LN rejection nor equal style prevalence follows from the population. The actual intended ordinary distribution was not empirically fixed in these experiments. **checked-artifact E15 +inference**, medium-high confidence.
7. **Can the same release decision preserve a role through future heads?** Audio-origin cues and segment codes eventually addressed this; therefore it was asked late. What remains unmeasured is a sufficiently trained, multi-seed comparison of representation choices with the same source/target/evaluation distribution. No duration-object comparison in this slice justifies preferring or rejecting that form. **checked-code +doc-claim budgets**, medium confidence.

## 6. Worth keeping

- Scope resolution with independently known fields, fixed prefixes, preserved entering holds, and per-range evaluation. `099cb66:src/ensomi_model/research/typed_audio_continuation/controls.py:43–99` and `controlled_audio_continuation/scope_allocation.py:63–86` are concrete reusable semantics (**checked-code**), though their eventual controller policy is not settled.
- The frozen corpus measurements and human-label inventory, retaining weights, denominators, scope, unknown labels and source provenance. E3/E15/E16 are useful evidence independent of a failed generator (**checked-artifact**).
- The on/off amount ablation and paired-source condition audits. They separate a real inference-policy effect from a claim that neural conditioning has been learned (**checked-artifact**, E3–E6).
- Exact release-ownership replay and conditional release diagnostics. Singleton pure-R support and head-versus-release likelihood decomposition expose failures hidden by totals (**checked-code/artifact**, E10/E11).
- Joint wait/release as an executable hypothesis with correct empty-wait semantics, plus the warning that it did not qualify the system. The conditional-mark singleton issue is real even if a later representation supersedes this implementation (**checked-code**).
- B40/B80 and LN/companion observations as descriptive regression witnesses, not an accepted playability evaluator. Their value is the documented failure they expose, not the universality of 40/80ms (**checked-artifact exports; gate results doc-claim**).

## 7. Claims worth re-verifying

Exact commands already executed, from repository root:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python /tmp/lineage-review/06-controls-ln/check_evidence.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/lineage-review/trees/audio-joint/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python /tmp/lineage-review/06-controls-ln/check_support.py
```

The first takes inputs named in section1 and records each generated osu path in `checked-evidence.json`; n=6 segment,9 attribution,32 feedback exports,1,973 corpus records,15 optional-probe ranges. The second uses E11's parent cases and declared durations, n=3; it verifies legal alternatives and deadline classification under the existing observer, not long-horizon quality.

Pivotal outstanding checks, with exact inputs:

- Early profile control: `J/20260924-shared-profile-v1` and `J/20260925-profile-path-crossover-v1`; independently recompute selected versus realized descriptors. The current report relies on `D/shared_arrangement_profiles.md` for those effect sizes. Training-seed uncertainty was not reported.
- Active-origin cues: `D/active_ln_audio_cues.md:158–181` owns endpoint/artifact identities; the 21→71 tail-coincidence result needs export-level replication and more than one seed per chart. No architectural conclusion here depends on accepting it.
- Release-support native aggregate: `J/20260928-release-support-learning-v1/{native-strict-v1,native-profile-v1,native-rr1-v1,native-joint-v1}/cases.json` and `comparison-v1/{result.json,scopes.json}`. Verify per-seed .1 amount/±1 difficulty gates and the named 21-head regression before treating the published aggregate as a confirmed effect.
- Learned allocation native gain: `J/20260928-scoped-ln-allocation-v1/{comparison-v3.json,native-context-v2/cases.json,native-progress-v2/cases.json}`. The training coverage is checked; the .054679/.015759 gain decomposition was not independently recomputed.
- Scope-missingness census: `J/20260928-clean-joint-proposal-v1/source-plan.json` and E12 `probe.py`/`scope_probe.py`; distinguish the 8192-draw coverage claim from the 15 factual-probe effects checked here.
- Segment gate replication: `J/20260928-response-blindspot-v1` and its plans, linked by `D/r1_short_hold_acceptance_zh.md:181–193`. Saved export counts verify ultra-short holds; the corpus-derived B40/B80 thresholds and planner's25ms zero-work acceptance were not replayed in this review.

No claim in this report requires a fresh model run to be believed. Where architectural/generalization claims would require retraining or additional generation, the necessary frozen comparison and evaluation contract are not specified well enough to give an honest exact command; inventing one would design a new experiment, outside this review's scope.

## 8. Cross-slice notes

The difficulty proxy, support filters, native response evaluator and proposal-learning recipe overlap other slices. Relevant findings here: early source labels and later source-H diagnostics do not establish native audio control; strict support can reject real release relationships; B80 and scalar stars do not identify ordinary LN/TAP organization. H-only attainability bounds and physiological response modeling are not independently reviewed here.

The formulation permits broader provisional future reasoning than the fixed event implementation. This review finds no evidence that millisecond event sampling itself was proved necessary, but also no fair trained comparison proving it wrong. The very small segment pilot should not carry that architectural conclusion.

## 9. Failed paths and unfinished work

The report is complete for all nine numbered questions at the evidence depth stated. This is not an exhaustive audit of every control-related commit or every raw native result; grouped early experiments and several later aggregate comparisons remain explicitly doc-claims.

Some initial source guesses (`controlled_audio_continuation/data.py`, `corpus.py`, typed `train_run.py`) do not exist; the operative training sampler is in the named artifact scripts. A guessed modulation document filename was replaced by the actual `row_condition_interactions.md`. These were discovery failures, not missing experiment evidence.

The independent reduction initially assumed every case had a whole-star value and every LN had a subsequent same-lane head. It was corrected to preserve undefined values and restrict the feedback pass count to the eight complete static requests per arm. The final reported counts exclude the live override case. An initial training-group lookup was corrected to read each identity's nested `entry`.

Saved paths point to an older repository location; scripts rebase only their `artifacts/` suffix to this repository, retaining hash checks where stated. No evidence files were altered. Checkpoint tensors were unnecessary for the performed replays; deletion of a particular checkpoint was not established and is not offered as an explanation for unperformed work.

No listening, new human annotation, image-by-image Lens review, training-seed replication, blind test, or neural forward pass was performed. The style postmortem and visual interpretations are read as historical claims. No external literature search was needed for this local historical audit.

## 10. Inconsistencies and items for the human

1. **Whole-scope amount versus prefix regulation.** E6's source-valid STYX TAP prefix reaches the feedback bound even though the entire source meets.485281; the local LN alternative gets an odds multiplier of e². The code/document distinction is explicit, but whether that local preference is acceptable under the requested amount semantics remains a product/research choice (**doc-claim witness, checked-artifact ablation; checked-code scope counters**).
2. **Control capability versus scalar agreement.** Native LN amounts and single D4 stars sometimes pass, while candidates still fail organization; calling controls uniformly ineffective would discard a real bounded success, while calling them solved would exceed it. The intended acceptance tolerance for semantic style and coordinated quality was not fixed (**checked-artifact**, E6/E13/E14).
3. **Late collapse scope.** “Six charts” means six generated outputs from three real H schedules, not six independent source charts; “whatever requested” is three cross-song requests, not a within-song sweep. The four-state collapse refers to latent segment codes, not the LN feedback controller (**checked-artifact**, E14).
4. **Source coverage versus optional-control interface.** Independently optional fields existed, but the clean-joint recipe reportedly supplied no style-known/LN-unknown prominent-Stream pieces. Hiding LN on factual histories has only a small effect, so this is a supervision inconsistency without proof that it caused native collapse (**doc-claim** coverage, **checked-artifact** E12).
5. **A narrow short-tail pass concealed a worse tail.** K1 STYX has fewer≤80ms LN than the source yet30≤40ms holds where the source has none. The late report explicitly rejects these outputs; this is a weakness of the earlier metric, not evidence of a hidden acceptance claim (**checked-artifact** generated count; **doc-claim** source/gate comparison).
6. **Local target differs from official difficulty.** Scoped strain normalization and complete future endpoint use are documented in code; it is an offline proxy. Whether±1 of this value is a useful user-facing range remains unsettled by its numerical agreement with whole-chart stars (**checked-code**, difficulty_targets.py:23–81).
7. **Correct implementation did not match training exposure.** Declared/owned progress counters support overlapping controls, but the 512-window training ledger has zero overlapping LN requests and very few large late errors. Their existence cannot be cited as evidence that live-override correction was learned (**checked-artifact**, E8; **checked-code**, scope_allocation.py:76–102).
8. **Ordinary is not synonymous with no short LN.** The ranked distribution is mostly TAP-led charts, yet valid short-LN group structure and LN-dominant charts exist. The human must settle the intended ordinary target mixture; neither a universal duration floor nor rare-style equalization follows from these corpus statistics (**checked-artifact**, E15; example semantics remain **doc-claim**).
