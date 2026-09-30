# Player response state in the audio-skeleton lineage: independent review

Reviewer: Claude (Opus), control plane, 2026-09-30. This review is independent of the Astra slice 04 run. I read nothing under `artifacts/reports/lineage-review/` or `.sync/*/jobs/*lr-*`, nothing in `artifacts/private/`, and nothing under `~/.codex/`.

Evidence grades used below:

- **doc-claim**: a lineage document, commit message or agent note. Not checked.
- **checked-code**: I read the code in the named tree.
- **checked-artifact**: I read a raw artifact on the mac.
- **re-computed**: I ran the lineage's own functions on its frozen inputs, or on synthetic inputs. These runs were light, single-threaded CPU, under a minute each. Scripts are listed at the end.

## 1. Summary

1. **The response scorers were never part of the system that was played or evaluated.**
   - The demo server and the standard qualification runner both build `ControlledSession` directly, with no planner and no guidance (checked-code).
   - The checkpoint the human playtested (`a99519c`, 09-26 16:06Z) predates the first player state and the first planner (`7ea3b95` and `24e786b`, 18:03Z and 18:16Z).
   - After a scorer existed, each fix tried to bake the response into the proposal by training. That did not transfer: the 384-update run learned it on the training bank but not on the reserved panel, where the 38-attack run appeared. The scorer was not then made a required step of generation.
2. **The late scorers catch the extreme long jacks. They are blind to the ordinary-looking failures the human kept reporting.**
   - Both late scorers reject a 16 s single-column run at 12 Hz and a 38-attack column in 4 s, both by large margins (re-computed).
   - The final unified "added work" scorer accepts the two reproduced playtest windows ([2,31,1,1]: 24% of the 4 s budget; [4,4,28,1]: 6%).
   - It also accepts an isolated 25 ms LN at exactly zero work, and did so in a real planner run (re-computed and checked-artifact).
   - Structural reason: the references are the 99th percentile of each chart's *peak* response in a star band. Anything below the extreme that 1% of charts reach once is free. So the scorer cannot express how often or how long something happens, and fragmentation and sustained concentration are exactly frequency and duration properties.
3. **Every response quantity was calibrated to ranked-corpus rarity inside official star bands. None was validated against a human judgment of difficulty or playability.**
   - The action-response state largely re-encodes the official difficulty. Its chart peaks correlate with official stars at Spearman 0.97 for attack and HH coordinates. A linear fit on its 32 log-peaks predicts held-out stars with R² 0.965 and MAE 0.135 (re-computed; this analysis is not in the lineage).
   - The star labels come from the same official strain model that scores a 16 s jack at 4.14 (re-computed, 4.139).
   - The formulation's target response $\mathcal C_0$ was never defined. The lineage's documents say so repeatedly, and it went ahead on proxies without asking the human for the mapper-facing definition the formulation calls for.

The key-1 versus key-2 split for the three failures is in section 5. In short, the long jacks are mostly a key-1 failure, of definition and of integration: better candidates existed in the proposal when a scorer asked for them. The short-LN tails are both a key-1 failure (the scorer is blind to them) and a key-2 failure (the proposal prefers them).

## 2. What I read and checked

- **Formulation.** `main:docs/formulation/gameplay-state.md` and `notation.md`, in full. `git log r1-restored-6.75m..audio-joint-2026-09 -- docs/formulation` is empty, so the lineage never changed the formulation (checked).
- **Baseline R1 code**, `main` tree: `bounded_typed_continuation/{consequence,response,recovery,routing}.py`, the preset `bounded_typed_r1_response.yaml`, and the R1 doc sections "Native-prefix recovery training", "Native response recovery result (6.75M)" and "R1 candidate action consequences".
- **Lineage code**, `audio-joint` tree:
  - package `player_response` (`state.py`, `envelope.py`, `conditioning.py`, `action_response.py`);
  - `typed_audio_continuation/{response_preference,program,demand,difficulty_targets}.py`;
  - `controlled_audio_continuation/{outcomes,sampling}.py`;
  - `planned_audio_continuation/{attack_response,buffering}.py` (headers);
  - `joint_audio_continuation/head_spacing.py` (header);
  - `segment_audio_continuation/model.py`;
  - `gameplay_evaluation/qualification.py`.
- **Lineage documents**, in full: `joint_action_spacing.md`, `ranked_2to6_action_reference.md`, `sustained_response_planning.md`, `time_horizon_player_responses.md`, `player_state_conditioning.md`, `action_response_frontier.md`, `event_time_gameplay_response_zh.md`, `r1_short_hold_acceptance_zh.md`, `common_prefix_outcomes.md`.
- **Lineage documents**, in part: `ordinary_expert_from_scratch_zh.md`, `typed_audio_continuation.md` (at `ca3dc65`).
- **Notes**: relay note `player-response-and-memory.md`, and agent notes `2026-09-27-player-response-frontier.md` and `2026-09-27-candidate-supply-and-response-selection.md`, in part.
- **Mac artifacts**, read or re-computed:
  - `20260927-player-response-frontier-v1` (`lineage/`, `native-reproduction/`, `response-state-check.json`);
  - `20260927-player-state-r1-learning-v1/native-response`, `feature-probe`;
  - `20260927-sustained-response-planning-v1` (`calibration.json`, `stream-comparison.json`);
  - `20260928-action-response-frontier-v1` (`work-pooled-v1/reference.json`, `calibration-v1/charts.jsonl`);
  - `20260928-response-blindspot-v1` (`probe-v1`, `accepted-prefix-v1`).
- **Demo branch**: `refs/archive/heads/codex/stream-generation-benchmark`, grepped for session construction.

## 3. What the formulation asks for (reference points)

All checked against `main:docs/formulation/gameplay-state.md`.

- **The target response is named but not defined.** It is $\mathcal C_0(H_{\le t},t;Y,e)$, a function of the committed history and a legal continuation over an explicit horizon. Its quantities, scales and comparison rules "remain to be defined". They are to be established "from a mapper's perspective after the initial chart dataset is complete" (lines 7-12, 53-56, 78-92).
- **The demand state is only a representation.** $d_\psi$ must preserve what $\mathcal C_0$ distinguishes. Sufficiency needs "comparison with target responses whose basis is independent of the candidate compression" (lines 100-167).
- **The canonical profile excludes physiology.** It excludes "individual capacity, misses, timing noise, adaptive fingering, handedness-specific profiles, physiological fatigue, and subjective pain". Mapper judgments "are not observations of a particular player's internal state" (lines 25-28).
- **Demand requests have no interface yet.** A demand request refers to response-spec quantities, and its interface "remains open until those semantics are defined" (lines 344-349).
- **The state-update interfaces are fixed; the dynamics are not.** They are $d^+=U(d^-,x^-,y)$ and $d(t+\Delta)=V(\dots)$, with no accumulation law or decay kernel chosen (lines 132-151).

## 4. Inventory of response objects

"Persistent" means carried forward through time across rows, silent intervals and publications, rather than recomputed from a fixed window at each query.

| Object (where, when) | What it computes | Learned or hand-written | Horizon | Where it acts | Calibration | Persistent | Distance from $\mathcal C_0$ / $d_\psi$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `response.py::response_costs` (R1 baseline, 09-20) | Count of heads within 30 ms of a same-lane head or release, over the current row and the next two H, with an optimistic future (one TAP per H, earliest releases) | Hand | Current row plus two H | Training only: 92 machine preferences at weight 0.25; not at inference | 30 ms chosen by the agent; no other calibration | No | A local legality-like preference, not a response. checked-code |
| `consequence.py::RowConsequence` `frontier2` (R1 baseline, 27,648 parameters) | A scalar residual logit per complete row, from exact clocks advanced passively to the next and second-next H | Learned by source CE plus KL plus the preferences above | Two H; no continuation $Y$ | Inside the policy logits at inference | None beyond NLL | Only through R1's learned history encoders | Not a separate scorer; NLL cannot identify it as demand (lineage agrees). checked-code. Deleted in the final `SegmentAudioModel` (checked-code) |
| `attack_response.py` (`7ea2e82`, 09-24) | Minimum count of same-lane attacks under 20 ms for a candidate and the previewed H | Hand | Preview H | Optional inference selection | 20 ms from the human | No | Hard local rule. checked-code |
| `spacing.py` minimum action gap $g$ (`7329b0e`, 09-25) | A hard support mask on HH, HR and RH ≥ g, with an existence proof over $t+2g$ | Hand | $t+L$ ms | Support mask for H, R and rows | g=21 from the ranked census (zero ≤20 ms) | No | Legality narrowing, not response. doc-claim plus code headers |
| `Recovery(hh,rh,hr)` (`ca3dc65`, 09-25) | Hard resource readiness | Hand | Per transition | Inference support and training-data admission | Default 37/25/21 = ranked minima. "Trial" 60/50/50 set after the human's 21-36 ms complaint | No | Excludes real ranked relations in 6.36% of charts (13.7% at 4-5★) (doc-claim). Relaxed to 60/25/21 on 09-28 (doc-claim) |
| `RecoveryPreference` (`82436bd`, 09-25) | Soft cost $4\log(\text{threshold}/\text{gap})$ below the ranked 1st-percentile gap per star; optional head-pressure term | Hand form, corpus thresholds | Per transition (HH 107 ms at 4★) | Default native sampling (`replay_row_scores`, head_pressure 4) | Ranked TRAIN q01 | Recent-heads window only | Per-gap rarity. A 125 ms jack costs nothing. checked-code |
| `AudioDemand` / `DemandFeedback` (typed line) | Learned head rate from audio; 4 s exponential count balance that shifts logits | Learned rate, hand feedback | 4 s memory | Inference | Source rates | Yes (decayed count) | An activity controller, not a response. checked-code |
| `ScopeStrainTrace` / `scoped_difficulty` (`2c2b8af`, 09-26) | Official osu!mania strain replayed over the full prefix, then the scoped weighted-peak "star" level | Hand (official) | Scope, with incoming strain | REINFORCE target $c_D$ (common-prefix outcomes); difficulty metric in every qualification | Official by construction | Yes (strain decays) | The only difficulty quantity in the loop. Blind to concentration (§5). checked-code |
| `CommittedPlayState` + `observe_continuation` (`7ea3b95`, 09-26 18:03Z) | Exact box-window per-column attack and release rates, held fractions and coordination events | Hand observations | 0.5-16 s windows, 32 s buffer | Input to the next two rows | None (observations only) | Yes | Structurally matches $U$/$V$ and branch isolation. checked-code |
| `AttackEnvelope` + `sustained_response` $J$ (`24e786b`, 09-26 18:16Z) | Squared excess of per-column attack rate over the ranked 99th percentile per star, integrated over a 4 s future | Hand form, corpus thresholds | 4 s forecast | `ResponsePlanner` (study scripts only); REINFORCE cost in the player-state fit | 6,923 ranked TRAIN charts, group-weighted | Uses the persistent state | Attack only; no hold or coordination terms. checked-code |
| `PlayerCondition` (`94d0b08`, 09-26 19:02Z) | 24 state features per column projected into R1 context (12,288 parameters) | Learned projection | Same | R1 input | Trained with source CE ± $J$ REINFORCE | Yes | Lesioning the whole input changes the hot-column probability by 0.46 pp (doc-claim; numbers seen in `feature-probe/result.json`) |
| `ResponsePlanner` (09-26, extended 09-28) | Forecast 4 s, commit 2 s, up to 4 R/R1 resamples on a fixed H; later rejects when no candidate passes, plus a committed work ledger | Hand | 4 s | Study scripts only; not in qualification or demo | n/a | Keeps committed state and ledger | Closest to "selection among proposed futures". checked-code |
| `ActionResponseState` + `ActionEnvelope` + added work (`02e522d`, 09-28 09:24Z) | 28 per-finger and per-hand impulses (attack; HH/RH/HR as 100/gap; hand turnover; attack with held partner) with exponential memory at τ = 250, 1000, 4000 and 16000 ms; recovery potential Ψ over ranked 99th-percentile peaks; added work against pooled budgets | Hand form (taus and 1/gap chosen), corpus thresholds | 4 s forecast; windows 0.5-8 s | Planner and `ResponseGuidedSession` in studies | 5,556 fit and 1,368 held-out ranked charts; 21/1,368 held-out false rejections at 4 s (checked-artifact) | Yes, request-independent, mirror-symmetric | The best structural match to the formulation's interfaces. Semantics are corpus rarity. checked-code |
| `ResponseGuidedSession` (`b3f0184`, 09-28) | Adds $\alpha\,\Delta\Psi$ to candidate row energy before release timing | Hand | Immediate | Studies (`GuidedSegmentSession`) | Same | Yes | Reactive; not a prospective value (lineage agrees). checked-code |
| Short-LN prevalence observers (09-28) | Share of ≤40 and ≤80 ms LNs among all heads, against a ranked reference at the requested LN level | Hand | Scope | Evaluation (qualification "fragmentation references") | Ranked 3.5-4.5★ | No | The first frequency-type quantity. Evaluation only. doc-claim |

Answer to "is any of them a persistent state carried through time": yes, several. `CommittedPlayState`, `ActionResponseState`, the strain trace and the demand balance all advance through silent time, survive control changes and are branch-isolated. None is learned. None was shown to be sufficient for any declared response, because none was declared.

## 5. Three documented failures: which of the three explanations held

The three candidate explanations are: the scorer liked the bad candidate, the scorer was not applied, or no better candidate was proposed. All four scorers below are the lineage's own code, evaluated at a 4★ request on the actual output files (re-computed). $J$ is attack-rate excess in seconds. "Work" is hot-window added work divided by the 4 s budget of 0.0499. "Proxy" is the scoped official-strain star level over the 4 s window.

| Case | Columns in hot 4 s | Proxy | $J$ | Work / budget | Scorer in generation path? |
| --- | --- | ---: | ---: | ---: | --- |
| Synthetic, one column, 12 Hz, 16 s | all on one | 4.14 (whole 4.08) | 7.51 (16 s) | 487× (4 s window) | n/a |
| Playtest reproduction, Zenithfall s271201 (actor-128) | [2,31,1,1] | 2.86 | 0.008 | **0.24 (accepts)** | No |
| Playtest reproduction, Hysteric s271212 | [4,4,28,1] | 2.92 | 0.001 | **0.06 (accepts)** | No |
| Same H, parent `b8aecd3e` (before outcome RL), Zenithfall window | [5,15,21,3] | 3.24 | not computed | not computed | No |
| 38-attack run, player-state response endpoint, Zenithfall seed 0 | [6,38,4,1] | 3.75 | 0.120 | 16.3 (rejects) | No: "No planner is used" (doc-claim); qualification path unassisted (checked-code) |
| Same fit, Hysteric seed 2 | [6,2,33,1] | 3.19 | 0.036 | 2.6 (rejects) | No |
| 25 ms LN, STYX, action-segment R1, real planner run | n/a | n/a | 0 | **0 (accepts)** | Yes: first proposal accepted at every decision (checked-artifact `accepted-prefix-v1`) |

### Failure A: a 16 s single-column run near 4.1★

This is a synthetic diagnostic, not a generated candidate. I reproduce 4.139.

**Which explanation held: the scorer liked it.** The official strain proxy puts the run inside a 4★ target band. The same proxy was the REINFORCE target in the common-prefix outcome learning ($c_D$) and the difficulty metric in every qualification (checked-code).

It is not true, though, that the proxy *rewarded* the actual collapse. In both reproduced playtest windows, the child checkpoint scores lower than its parent on identical H (2.86 against 3.24 at Zenithfall; 2.91 against 3.88 at Hysteric). The children also have fewer heads (34 against 44; 37 against 67; checked-artifact).

My reading is that the proxy was *blind*. It supplied no gradient against concentration while the outcome RL reduced chord cardinality, a drift the lineage itself reports. The parent and child identities are `b8aecd3e` and `364c7b71`; I matched them to `common_prefix_outcomes.md` (checked-artifact). This is two windows, so the evidence is moderate.

### Failure B: the playtested long jacks

The human's own playtest chart and seed were not kept (index, V47). These are the lineage's reproductions of the same control combination, not that sample.

**Which explanation held: the scorer was not applied.** The planner did not exist when the human played. Later demo and qualification paths never call it (checked-code).

**Better candidates existed.** On the same H, the parent distributes the attacks (checked-artifact). With up to 4 resamples, the planner cut mean $J$ from 0.0275 to 0.00042 on nine Stream outputs. It replaced 52 of 1,209 publication decisions, and 32 decisions still exhausted the budget (checked-artifact `stream-comparison.json`).

**But the final scorer accepts these windows.** The 09-28 unified work scorer passes both reproduced windows, so its introduction was a regression for this channel. It is consistent with the lineage's own probe: 32 same-finger TAPs at 125 ms give 0.0199, under the 0.0499 budget (checked-artifact).

**Attribution: mostly key 1.** That is a key-1 integration failure plus a key-1 definition failure. Key 2 is less involved here.

### Failure C: the 38-attack run in 4 s

**Which explanation held: the scorer was not applied.** Every response scorer rejects it, but none was used.

The model was the attempt to teach $J$ to R1 by REINFORCE on a 20-song replay bank. It improved $J$ on the bank by 60.8% and worsened it on the reserved panel (doc-claim).

The strain proxy calls this window 3.75: "on target" within the 0.25 tolerance of the 4★ request.

**Attribution: key 1 (integration).** It also shows that internalizing a hand-written cost into a small policy did not transfer.

### Failure D: short LN tails ending at the next head

**Which explanation held: the scorer liked it.** An isolated LN of 23, 25, 40, 80, 150 or 300 ms gets zero work, and a 21 ms LN gets 1.5e-5 (checked-artifact `probe-v1`). The HR coordinate at τ=250 is 16 for a 25 ms LN, against a 4★ reference of 17.44 (re-computed from `reference.json`).

Other channels:

- The attack-only $J$ is invariant to releases. The agent's candidate-supply note reports that 159 of 160 LN-context candidates have zero 16 s $J$ (doc-claim).
- The NLL-trained `frontier2` energy *raises* expected release counts in those contexts, for example from .74 to .87 (doc-claim, agent note `2026-09-27-candidate-supply-...`).
- In the from-scratch expert (155 of 191 short tails at the next H), the new decoder deletes the consequence module. Its `local` and `timing` arguments are unused (checked-code). No scorer was in that loop.

**Were better candidates proposed?** Undetermined. The planner stops at the first zero-work proposal. Candidate LN counts do vary widely under identical H (3 to 32 LN heads per 8 s; doc-claim).

**Attribution: both key 1 and key 2.** The proposal's short-LN rate is 4-9% of LNs at ≤40 ms, against 0.106% in ranked 3.5-4.5★ charts (doc-claim).

## 6. Hard thresholds: origin and fate

| Value | Origin | Fate |
| --- | --- | --- |
| 30 ms | Agent choice in R1's `response.py` (baseline, 09-20); also the pool admission criterion | Kept in R1; never calibrated |
| 20 ms | The human said same-lane attacks under 20 ms are essentially always bad (index V33, V35). The census found zero HH or RH ≤20 ms in 8,774 ranked charts (doc-claim; census counts not re-checked) | `SHORT_ATTACK_MS=20`, close-pair screen, spacing law g=21. The generated "zero at <20" claims missed exactly-20 ms cases (lineage's own correction) |
| 37 / 25 / 21 | Ranked census minima for HH, RH and HR | Default `Recovery` |
| 40 ms | R1 diagnostic column; later "ranked 3.5-4.5★ has no HH ≤40 ms" (doc-claim) | ≤40 ms short-LN prevalence red eval |
| 60 / 50 / 50 | A "trial profile" by the agent (`ca3dc65`, 09-25) after the human rejected 21-36 ms attacks (V38) | Hard support in all controlled models. Filtered relations in 457 TRAIN charts. Relaxed to 60/25/21 on 09-28 (doc-claim). The 20→60 ms step for HH has no corpus or human basis beyond "no ranked HH <60 ms at 2-3★" |
| q01 gaps (HH 107 ms at 4★, etc.) | Ranked TRAIN first percentile per star | Soft sampler cost; still default |
| 170 ms | Diagnostic chain criterion | Reporting only |

**Pattern.** Each threshold answered the most recent complaint at the nearest module. None was re-examined against a declared response. The human's 20 ms rule was treated as a quantitative anchor, although the human framed it as an example of badness (V37 also asked for fewer defensive checks).

## 7. The physiology request against the formulation: the gap stated, not resolved

The README statement checks out: the formulation leaves $\mathcal C_0$ undefined and excludes physiological fatigue and individual capacity from the canonical profile. Per the feedback index, the human asked on 09-26 for a player state with corpus-derived decay (V52), and on 09-28 for response curves based on hand physiology and the corpus (V80).

The precise gap:

1. **Per-finger and decay are not in conflict.** The canonical profile fixes the lane-to-hand-role mapping, so a four-finger coordinate system is inside it. $V_\psi$ allows any decay law.
2. **The conflict is the target and the calibration source.**
   - The formulation says the target is mapper-facing and derived from mapper evidence. A physiology-derived kernel could only be a representation choice $\psi$, still judged against $\mathcal C_0$.
   - The human's wording makes physiology plus corpus the calibration authority. It also leaves open whether the state models a *player's* load, which the formulation explicitly says mapper judgments do not observe.
3. **It changes what a difficulty control means.** In the formulation a demand request refers to $\mathcal C_0$ quantities. Under the physiological reading it would refer to modelled hand load.
4. **The lineage did not settle it either.** It cited physiology papers for *structure* only (Häger-Ross and Schieber 2000; Kelso 1984; Bächinger et al. 2019) and said none supplies osu-specific constants (doc-claim, `event_time_gameplay_response_zh.md` §4). I agree with that restraint.

## 8. Can such a state be identified from chart data alone?

Partly, and not the part the human asked for.

**What charts do identify.** Ranked charts identify what mappers and ranking review accept at a given official star level and context. That is a typicality model, useful for key 2 and for weak references.

**What charts do not identify.** They carry no player outcome. The lineage's taus (250-16000 ms) and its 1/gap impulse are hand choices; only the thresholds are fitted (checked-code). Many (kernel, impulse, quantile) triples would accept the same corpus.

**The star bands are circular.** The official strain model computes the star label. The re-computed Spearman 0.97 and R² 0.965 show the lineage's state mostly re-encodes that model. Conditioning references on star therefore inherits the official model's blind spots, rather than supplying an independent view.

**Rate counts cannot separate acceptable from unacceptable concentration.**

- A ranked 4.07★ chart (Extra Mode) has a 28-attack anchor over 4.15 s with changing accompaniment (doc-claim).
- The Hysteric failure has 28 attacks in 4 s on one column.
- The distinction is in organization (what the other fingers do). The formulation places organization in the style–demand relationship, and the lineage deliberately kept it out of demand.

**Sources never considered.** No lineage document uses player-performance data (osu! replays, hit errors, pass statistics); a grep over docs and both note trees finds only "no input trace or player test was performed". Such data is the only external observable of stimulus-response. Mapper or human pairwise judgments are the other source, and the one the formulation names. Whether either is in scope is for the human (§11).

## 9. Judgment, both sides

**For keeping the key-1 direction as the lineage framed it** (an explicit, persistent, request-independent state judging private continuations before publication):

- The implementation is careful:
  - exact event-time $U$/$V$ with composition;
  - branch isolation;
  - no reset at control changes;
  - a committed work ledger across publications;
  - held-out false-rejection accounting.
- It is the closest thing to the formulation's frontier in the repository.
- Selection demonstrably worked where the scorer saw the defect: $J$ fell 98% with a 4-sample budget on fixed H.
- The negative results are real knowledge for a future $\mathcal C_0$:
  - the peak-quantile blind spot;
  - the 25 ms LN at zero work;
  - the strain proxy's rating of concentration;
  - that REINFORCE internalization on a small bank does not transfer.

**Against:**

- No object was the formulation's response. The target was never defined, and every calibration is corpus rarity inside official-star bands, which is largely the official difficulty model again (R² 0.965).
- No quantity was checked against a human judgment.
- The scorers were never in the played or evaluated system, so the lineage has no evidence about what a response-gated system would do.
- The work went from frontier2 → spacing law → 60/50/50 → q01 preference → strain RL → $J$ → player-state RL → unified work → guidance → ledger → quadratic potential. Each step answered the previous failure at the nearest module. The two whole-system questions were never taken up:
  - making the gate mandatory in generation and evaluation;
  - defining what it must distinguish, with the human, from contrast cases.
- This supports the human's local-optimum suspicion for this slice.

**My call.** The lineage's response *machinery* is partly reusable. Its response *semantics* should not be carried forward. The next step from the baseline is not another scorer. It is to settle, with the human, which distinctions the response must see, using the retained witnesses as contrast pairs; the formulation's own procedure in "Defining the response from mapper evidence" describes this. Every candidate scorer should then be measured on that set, for both sensitivity and false rejections, before it is wired into anything.

Confidence:

- High that the lineage's scorers are not the target and were not integrated.
- Moderate that the long-jack failures would largely have been caught by a gate with duration-aware semantics, given that the proposal contained alternatives in the two reproduced cases.
- Low on how much of the short-LN problem a response state can fix, as opposed to the proposal (key 2).

## 10. Overlooked or never questioned

1. **Integration.** Neither the demo nor the qualification runner ever called the response gate. All human and qualification evidence is of the unassisted proposal.
2. **Frequency and duration.** Every scorer compares a peak or a per-transition value against a threshold. Prevalence (how often short LNs occur) and persistence (how long a jack lasts at a moderate rate) are not representable by peak-quantile references. The prevalence observers added on 09-28 are the first exception, and they are evaluation-only.
3. **Circular star conditioning.** The official star rating was used as the control variable, as the band for every reference and as the RL target. Its known blindness was diagnosed (the 4.14 synthetic) but never propagated to the references built on its bands.
4. **No negative-labeled validation set.** Sensitivity was shown on a handful of generated failures chosen by the agent. The human's playtest failure left no file (V47), and no judged set of accepted and rejected windows was assembled.
5. **Proposal coverage was never measured.** A 4-resample budget on fixed H was the only probe of "does an acceptable candidate exist". This is the measurement that separates key 1 from key 2.
6. **Organization.** The difference between an acceptable anchor and a failed jack, at equal counts, is organizational. The lineage kept demand and style apart, which is formally right, but it never built the joint comparison the formulation describes.
7. **Player-performance data** was not considered as a calibration source.
8. **The formulation's order was skipped.** It asks for the response definition from mapper evidence before a demand representation. The lineage built representations first. It also never checked whether the initial annotated dataset the formulation waits on was far enough along to start. I did not check this either.

## 11. Worth keeping

- **The event-time state machinery.** `player_response/state.py` and the `ActionResponseState` update and advance logic, with its tests: exact composition through silent time, open-hold semantics, branch isolation, request independence and mirror symmetry. It is a candidate $d_\psi$ substrate, not a validated one.
- **The planner's publication discipline.** `ResponsePlanner`: private forecasts, committed-prefix-only state, the `NoAcceptableContinuation` refusal and the committed work ledger with per-control-range accounting.
- **The ranked census and its caveats.** In `ranked_2to6_action_reference.md` and `joint_action_spacing.md`: the zero ≤20 ms HH/RH result, the per-star minima and the coverage cost of stronger profiles (doc-claim; the census itself is not re-run).
- **Witness cases for a future response definition:**
  - Zenithfall s271201 [41094,45094) and Hysteric s271212 [8221,12221), with the parent `b8aecd3e` outputs on identical H;
  - the 38-attack window at 118849 ms;
  - STYX's 25 ms LN at 1122 ms;
  - the ranked counterexamples Extra Mode, Happy Love Expert, Singularity FISSH's Beyond, Infinite Jest and lost memory.
- **Negative results.** The strain proxy rates concentration as ordinary difficulty. The unified work scorer accepts 7-8 Hz single-column runs and short LNs. A small-bank REINFORCE on a response cost did not transfer. Lesioning the player-state input barely moves the policy.

## 12. Inconsistencies and questions only the human can settle

1. **Which response state does key 1 mean?**
   - (a) The formulation's canonical chart-demand response, defined from mapper evidence, with physiology excluded.
   - (b) A model of player hand load, calibrated on physiology and the corpus (V52, V80).
   - Both can share machinery, but they differ in target, validation and the meaning of a difficulty control.
2. **Is a difficulty request an official star value?** If it is, every reference inherits the official model. If not, what is it before $\mathcal C_0$ exists?
3. **Is corpus rarity acceptable as a stand-in for "comfortable, demanding, unplayable"?** Key 2 makes the ranked distribution the proposal target. Using the same distribution as the response would make key 1 redundant with key 2. Was that intended?
4. **Is the Hysteric [4,4,28,1] window unplayable at 4★?** A ranked 4.07★ chart has a 28-attack anchor in about 4 s. Is the difference the accompaniment, the context, or the human's standard being stricter than ranked?
5. **Was the 20 ms statement a threshold, or an example of a class of badness?** The lineage used it as the former.
6. **Is player-performance data (replays, hit errors) in scope?** If so, as calibration for the canonical profile, or for individual profiles, which the formulation excludes?

## 13. Not verified, and how to check

- **The 38-attack and paired-128 outcomes as training effects.** I checked outputs and checkpoint hashes only. To check the fits, read `artifacts/joint-audio/20260927-player-state-r1-learning-v1/{fit-comparison.json,analysis.json}` and `20260927-paired-response-r1-v1/`.
- **The ranked census counts** (zero ≤20 ms, minima table). Re-run `ssh bings-mac 'cd ~/ensomi/ensomi-model/artifacts/joint-audio/20260925-ranked-2to6-reference-v1 && cat result.json'` and compare with the doc. The census script is `audit.py` there (about 130 s CPU; not run, to stay light).
- **The 60/25/21 relaxation and which final checkpoints use which profile.** Grep `recovery` in the `model.recovery` of checkpoint configs under `20260928-*/` (not done).
- **Planner outcome on the 38-attack checkpoint.** Unknown whether an acceptable candidate would exist. This needs a `ResponsePlanner` run with `a91791fb` on Zenithfall seed 0: model inference on CPU, a few minutes. Not run under the no-compute rule.
- **Whether short-LN alternatives exist under identical H.** `20260927-candidate-supply-v1/run-v2/cases.json` has 32 candidates per context. Score them with `probe.py`'s `describe` against the ≤40 ms prevalence reference.
- **Human feedback wording.** I relied on the index paraphrases V33-V81, not the private verbatim file.
- **The official-star correlation** used chart peaks at every star and a linear fit. It shows re-encoding, not causation. A per-window comparison would be stronger.

## Reproduction of my computations

- Control plane: `python3 <scratchpad>/opus-player-response/strain_probe.py`. It builds synthetic runs at 6-12 Hz over 16 s with `ScopeStrainTrace` and `compute_mania_star_rating_20241007` from the `audio-joint` export. One column at 12 Hz gives 4.139 scoped and 4.082 whole; a four-column cycle gives 2.599.
- Mac, `/tmp/lineage-review/opus-player-response/`, run with `~/ensomi/ensomi-model/.venv/bin/python`:
  - `window_strain.py`: scoped proxy in the reproduced windows for the reported, core2500, modulated and paired128 arms;
  - `synthetic_work.py`: $J$ and work on synthetic runs;
  - `hot_window.py <osu...>`: hottest 4 s column window, proxy, $J$ and work;
  - an inline Spearman and linear fit over `20260928-action-response-frontier-v1/calibration-v1/charts.jsonl`.
- The scratch scripts are disposable. Their inputs are the frozen artifact paths above.
