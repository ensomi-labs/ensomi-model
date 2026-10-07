# Phase C as a frontier controller: the human's steer, Astra's proposal, and whether it fits R2 (2026-10-07)

Question: can Astra's proposal of 2026-10-07 be used for the actual R2 v2 phase C, and what would change in the
plan of [r2-condition-plan-v5](r2-condition-plan-v5.md) (open decisions in [v5-open](r2-condition-plan-v5.md#v5-open))?

## Human steer

<a id="h-relabel-baseline-fail"></a>**Human judgment, 2026-10-07 ([private, local](private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md#prompt-5)).**
The human judges that the relabel and the baseline, the two stage-0 steps phase C's difficulty set-up needs
(`labels.py` star-v2 cells; `baseline.py` b(S) and g), fail detrimentally. The agent read this as rejecting the
planned difficulty request (residual star, Difficulty_nu minus a skeleton baseline, on fixed 30/60/120 s cells)
before it is built. Which part fails in the human's view is not stated: the exact cell labels, the residual
centring, or only building them before the design is settled. The stage-0 chain was therefore not started.
Phase-C configs at `ensomi-model` `3520b71` still set `star_conditions: on` and `star_value: residual`.

<a id="d-no-relabel-baseline"></a>**Decision (human, 2026-10-07, answer to the agent's question, [private, local](private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md#answer-1)); superseded the same day by [d-difficulty-proxy](#d-difficulty-proxy), the human having misread the question. Kept as first recorded:**
Both are rejected: exact difficulty labels of real-chart spans (the star-v2 cell relabel) and the skeleton
baseline b(S) with the residual target. The agent reads this as b(S) in any role, including as a controller
input (agent extension, not the human's words). Open: whether the star calculator may still score generated
output. The phase-C configs' `star_conditions: on` / `star_value: residual` are now superseded in intent but
unchanged in code.

<a id="d-difficulty-proxy"></a>**Decision (human, 2026-10-07, correcting the answer above, [private, local](private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md#prompt-6)).**
- **Out:** the star-v2 relabel (exact difficulty labels of real-chart cells), and the original osu!mania star
  calculation as the scoped difficulty metric, because it only works well for a whole song (`properties.difficulty`
  tiles a scope to 240 s and is undefined under 30 s, [o-phasec-code-audit](#o-phasec-code-audit)).
- **In:** a skeleton-derived difficulty proxy, needed to tune the difficulty response without the skeleton
  module; preferably an extension of the osu!mania difficulty algorithm (strain-based) that works on shorter
  intervals, for condition-control tuning.
- **Agent reading, unconfirmed:** the proxy takes the role b(S) had (difficulty implied by the head-time skeleton)
  but is rebuilt on the strain algorithm instead of a ridge fit to tiled star on fixed cells. Not resolved:
  whether it reads head times only or head times plus the generated arrangement, and what "the skeleton module"
  is (possibly a future head-time generator, which R2 v2 does not have).
- Effect: the stage-0 chain (relabel, ridge baseline, `draw_sim` on the current C configs) stays off; the
  correction prompt for Astra was revised ([p-astra-correction](#p-astra-correction)).

<a id="h-phasec-design-asks"></a>**Human design asks, 2026-10-07 (same source; the human's words to Astra, agent paraphrase).**
- LN share and difficulty are two different kinds of control, each accumulating an exact metric, and need separate recipes.
- A control should respond well over varied scope lengths.
- Playability and control strength must be balanced.
- Use the style data: 5,000+ sections of about 10 s, annotated on five style concepts with a three-level strength and a separate high/low confidence (the style-module job counted 4,714: [s-style-module](r2-phasen-and-lens-20261006.md#s-style-module)).
- With head times fixed, the surface through which difficulty can be controlled is limited, and the design must account for it.
- Doubt about whether a generation frontier should aggregate what has been generated, plus its residue, to approximate the control target.
- Asked for staged phase-C targets, recipe, architecture, losses, and desired and undesired properties.

## Astra's proposal

<a id="a-astra-frontier"></a>**Proposal (Astra, through the human's chat, 2026-10-07; not decided; the verbatim response is [private, local](private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md#astra-response)).**
Its four web citations (osu!mania difficulty source "version 20241007", CORN, hindsight relabelling, FUDGE) were
placeholders in the paste and are not checked. Agent summary:

1. **Phase C as a finite-horizon controller trained on the outcomes of its own continuations.** Source-conditioned
   CE stays as the first stage but is not the whole recipe. Critique of the current plan: L_star is one more
   source-CE term and does not teach generated scopes to reach the target; the relaxed linear star proxy is a
   different objective from tiled star; a frozen base does not keep controlled outputs natural; in frozen mode,
   source CE on decisions the identity gate bypasses has no gradient for the conditioning path.
2. **A per-scope frontier state outside the natural encoder:** an exact ledger (heads and LN heads in scope, held
   lanes and ages, owned unresolved LNs, progress), a remaining-skeleton summary (rows, seconds, beats, multiscale
   density and gaps of the rest of the scope), a completion forecast under a named continuation policy (natural or
   controlled), the request with a presence bit, and a feasibility estimate. Three residuals kept apart:
   d* − b(H,S) (skeleton-centred), d* − E[D_final | F_k, π] (predicted completion error), r*·H_k − L_k (LN deficit).
3. **LN:** a count problem with a generated denominator (chord size changes the share). First interface: a
   governed tap/LN split, p_N(G|F)·p_N(T|G,F)·exp(η·n_LN(T))/Z, so an LN request keeps the natural head masks and
   release categories and only moves taps to LN heads; natural release pointer kept at first. Train on terminal
   per-scope error with a tolerance, not on every prefix ratio; at the last in-scope row the error can be averaged
   exactly over legal actions.
4. **Difficulty:** star is not additive (strain with decay and maxima), so control needs completion prediction.
   Inputs d*, b, d* − b, frontier, remaining skeleton, natural-completion quantiles. Estimate a demonstrated
   playable range per skeleton and prefix by sampling continuations, and draw training requests inside it.
   An LN's release can fall after the scope, so exact difficulty settles late; recompute the full scorer once
   the owned objects are complete.
5. **Style:** two roles, source-conditioned CE for meaning, and a completed-passage observer (five masked ordinal
   heads, confidence-weighted). Keep intensity, confidence and adherence strength separate. No relabelling of
   10 s labels onto crops; no star labels under 30 s; long-scope style defined as a sustained regime.
6. **Architecture:** frozen natural branch; frontier added to the conditioning path; LN governed split,
   difficulty modulation or residual action preferences, per-concept style adapters, and a zero-initialised joint
   term for co-active kinds; optional gated residuals on row or pointer scores.
7. **Stages C0–C5:** C0 measurement (exact replay, observer, continuation panels, control ranges); C1 meaning
   (source-conditioned CE, governed split, style requests); C2 LN completion (exact outcomes, achieved-goal replay,
   KL on generated states); C3 difficulty completion (exact scorer outcomes, value learning); C4 composition and
   lengths; C5 planning only if needed.
8. **Outcome training loop:** generate the scope without autograd, continue naturally until the metric settles,
   compute the exact outcome, re-score the trajectory in 256-decision chunks with gradients through readable
   decisions only; actor loss = −Σ stopgrad(A_k) log π + β·KL + α·source CE. Hindsight-relabelled examples only in
   the supervised replay path. Starting learning rates 1e-4 and 3e-4 (CE), 3e-5 and 1e-4 (outcome).
9. **Length robustness:** KL per scope divided by the fixed head-row count of the scope, so longer scopes are not
   penalised more for the same terminal gain; coverage over durations, densities and offsets.
10. **Operating trade-off:** tolerance-band error per kind; adherence strength limits departure, never changes the
    target; playability judged at matched difficulty and style.
11. **Limits:** a FUDGE-style critic trained on source prefixes is a heuristic on a conditioned policy; global
    reranking selects among natural decisions outside the scope, so under strict rule L planning must act only
    on permitted decisions.
12. **First experiment:** four LN arms (A current FiLM; B FiLM plus ledger and remaining skeleton; C governed split
    plus frontier; D C plus exact outcome training), on matched skeletons, prefixes, targets and length buckets.

## Can it be used for the actual phase C?

<a id="o-phasec-code-audit"></a>**Observation: the proposal checked against the code (subagent audit, read-only, `ensomi-model` `3520b71`, 2026-10-07; paths under `src/ensomi_model/r2/`).**
- **Frontier ledger, partly present.** The FiLM input is 2 roles × 2 kinds × 17 channels = 68 (`model.py:94`): the value, offsets to a and b in ms and beats, time progress, log1p of remaining head rows in scope, and four statistics per kind (`features.py:328-369`). LN statistics are committed heads and LN heads in scope and their ratio (`features.py:292-300`). Difficulty statistics are the proxies (mean chord/4, same-lane repeat rate, held-lane occupancy, LN share; `features.py:303-317`). Missing: held lanes, LN ages and owned unresolved LNs (natural inputs only, `lane_query`, `features.py:213-230`), and density of the remaining scope. γ and δ depend on the frame only, not on z (`model.py:105-112`), so state-dependent modulation needs a new input path.
- **Natural lookahead** (`query_features`, `features.py:233-262`): head counts in the next 1-32 beats, the next 16 gaps, song time left; not limited to the scope.
- **Governed split is exactly computable.** 625 = 5⁴ actions; on a free lane the codes are 0 empty, 1 tap, 2 LN head, and on a held lane 0 keep, 1 release at t_k, 2 gap release, 3 gap release + tap, 4 gap release + LN head (`common.py:7-18`). `governed_split` (`model.py:442-469`) already groups actions by head mask and release types and is the L_ln term (`loss.py:76`). So p(G|F)·p(T|G,F) with an exponential tilt on the LN count is a small change. Caveat: the release pointer conditions on the full code, T included (`model.py:356-360`).
- **Losses are all teacher-forced source CE.** L_ln (governed split) and L_star (whole-decision CE) on decisions reading the kind; the relaxed proxy is a pathwise relaxation through g and needs the residual target (`train_ce.py:476-477, 629-649`; `proxy.py:167-197`). In frozen mode a decision with no readable request has an all-zero frame, the gate returns z, and its CE has no gradient for FiLM (`model.py:107-112`; `train_ce.py:473-475, 598`). Astra's critique on this point is correct by construction. No outcome-reward training exists anywhere; `train_dpo.py` is soft-label DPO on real preference pairs, which do not exist.
- **Difficulty values come only from source-chart cells.** Draws take LN share from the chart's own counts and star from the cached cell labels (`conditions.py:141, 166`; `data.py:90-100`). Star is undefined under 30 s (`properties.py:31, 174-175`). `difficulty()` is a full recomputation of the 20241007 strain port on tiled head-owned objects (`properties.py:146-183`); no incremental form. `continue_chart(close_scope=...)` already continues until a scope's LNs close (`sampling.py:40-43, 108-112`).
- **Rollouts.** `continue_chart` is no-grad, rule-L aware, single sample, batch 1, and rebuilds derived state and history tokens each step, so its cost per decision grows with the prefix (`sampling.py:46-121`). No batched path. Outcome training at scale needs a faster sampler.
- **Style.** Only `ln_share` and `difficulty` exist as kinds (`features.py:28`). Lens pool: 4,714 sections, median 5.5 s, mean 10.5 s, 90th percentile 10 s; levels absent / supporting / prominent; confidence exists on human labels only (171 High, 23 Low, 406 unspecified), machine labels have none; 3,268 sections join the R2 cache (2,995 fit_train, 273 fit_dev) on 1,021 charts (`artifacts/r2-style-module-20261006/report.md`).
- **What the decision changes in code** ([d-difficulty-proxy](#d-difficulty-proxy)): without star-v2 labels, `star_conditions: on` cannot start, so no difficulty request can be drawn from source charts; `star_value: residual` needs `baseline.py`'s b and g (`train_ce.py:371`), and the relaxed proxy needs the residual (`train_ce.py:476-477`). A strain-based short-interval proxy would replace both `properties.difficulty` (as the scoped metric) and b(S); the draw, the frames' difficulty statistics, `request_set.py:120-121` (the 30 s floor), `tests/r2/test_phases.py:277` and the draw hash would follow. With no source values, source-conditioned CE for difficulty has no target unless the proxy supplies one; [p-astra-correction](#p-astra-correction) asks Astra.

<a id="o-phasec-rollout-cost"></a>**Observation: cost of an outcome-training loop on bings-mac CPU (subagent, 2026-10-07; jobs `20261007-052749-r2-phasec-probe`, `20261007-052837-r2-phasec-cost`, `20261007-053211-r2-phasec-cost-t1`; raw numbers in `artifacts/r2-phasec-cost-20261007/timings.json` on bings-mac; checkpoint `ckpt-0048000198.pt` of the running phase-N run; measured while phase N trained, about ±20% noise).**
- **Sampling** (`continue_chart`, empty track, 2 threads): 196-222 decisions/s on a 502-decision chart, about 170/s on the median fit_dev chart (949), 92-100/s on a 2,202-decision chart. The rate falls along the chart (162-237/s near the start, 57-67/s after decision 2,000), consistent with rebuilding derived state and history tokens over the whole prefix each step (read from code, not profiled). One thread is as fast as two. A non-empty track costs 12-16% more. Earlier full-evaluation free runs at 4 threads: 229 decisions/s ([s-phasen-tune runtime files](r2-phasen-and-lens-20261006.md#s-phasen-tune)).
- **Exact difficulty**: 0.011-0.058 s per call at any scope length (scopes are tiled to 240 s). LN share 1-12 ms. In 8 sampled 60 s scopes the natural continuation added no decisions to close tails (LN-heavy requests not tested).
- **Teacher-forced 256-decision window with gradients** (HEAD, TCN checkpointing on): 0.14-0.19 s at a 255-row prefix to 0.63 s at 2,202; frozen base with the conditioner only, 0.13-0.23 s.
- **Throughput:** sample + score + re-score a 60 s scope is 3.1-6.4 s, i.e. about 570-1,170 completed scopes per hour per process, 705 on the median chart: about 99 re-scored heads/s against phase N's 2,000 at 4 threads, 20× fewer. Four single-thread processes would give about 2,800 scopes/h if they scale (not measured). Whole-song scopes: about 136/h. Sampling is about 90% of the cost, so a faster sampler (incremental state, batching several samples) is the first lever; not attempted.

<a id="p-astra-correction"></a>**Correction prompt for Astra (agent draft, 2026-10-07, given to the human to send; the human may edit it).**
The text below is the agent's wording, third version. Version 1 stated a rejection of the relabel and of the
baseline in any role (from answer-1) and said the median section was "5.5 s, not about 10 s". Version 2 corrected
point 2 after [o-phasec-code-audit](#o-phasec-code-audit). Version 3 restates point 1 after
[d-difficulty-proxy](#d-difficulty-proxy), adds the measured loop cost from
[o-phasec-rollout-cost](#o-phasec-rollout-cost), and asks for the proxy's design first. Style-pool facts are from
[s-style-module](r2-phasen-and-lens-20261006.md#s-style-module).

````text
Corrections to your answer. Please revise it with these in force.

1. Difficulty: what is out and what is in.
   - Out: the relabel. No exact difficulty labels precomputed on real charts over fixed cells (30/60/120 s cells at offsets 0/10/20 s, plus the whole song), and no conditioning on such source-chart cell values.
   - Out: the original osu!mania star calculation as the control metric, because it only works well for a whole song. (Our wrapper tiles a scope's objects out to 240 s and is undefined below 30 s.)
   - In: a skeleton-derived difficulty proxy. We need it to tune the difficulty response without the skeleton module. Preferably it extends the osu!mania difficulty calculation algorithm (strain-based) and works on shorter intervals, for condition-control tuning. Our current baseline b(H,S) is a ridge fit of typical tiled star on 23 head-time features over the fixed cells; the proxy should preferably be strain-based instead.
   Your answer relied on the rejected items in several places: C1 "use the full chart corpus for exact LN and difficulty labels", the difficulty inputs [d*, b, d* - b, ...], "if difficulty is included, measure a valid containing scope", exact completed-scope star as the outcome, and the comparison with b(S) +/- 0.5. Redo the difficulty part on the proxy.

2. Data correction. The style pool is 4,714 sections after admission (15,948 concept cells; 2,523 sections carry all five concepts). Section length: median 5.5 s, mean 10.5 s (a long tail), 90% at or under 10 s. Levels are absent / supporting / prominent. The high/low confidence exists only on human labels (230 sections; 171 high, 23 low and 406 unspecified judgments); machine labels carry no confidence. 3,268 sections join charts in our corpus (2,995 in the training split, 273 in the dev split).

3. Constraints, measured on our machine. Training and generation run on one Apple M5 CPU, no GPU. The phase-N model (2.4M parameters, long-range memory off) is the frozen base for phase C. Teacher-forced phase-N training runs at about 2,000 head decisions/s on 4 threads. Our sampler is single-sample and sequential: about 100-220 decisions/s per process, slowing along the chart. A loop that samples a 60 s scope, scores it and re-scores it with gradients completes about 700 scopes per hour per process, about 20 times fewer heads per hour than teacher-forced training; sampling is about 90% of that cost.

Please answer:
a. Design the difficulty proxy: what it reads (head times only, or head times plus the generated arrangement), how it extends the strain algorithm to short intervals, which interval lengths it supports, how it is validated, and what it costs to compute. Say how you read "skeleton-derived" so that I can confirm it.
b. How a difficulty request is defined with that proxy, and where its training signal comes from.
c. The staged phase-C targets again, for LN, difficulty and style, marking what changed from your previous answer and why.
d. For each stage: the training signal, the architecture change, the loss, the properties wanted and not wanted, and the evidence required to advance.
e. The first experiment for difficulty, with a rough compute estimate under (3). Keep the four-arm LN study unless (1), (2) or (3) changes it, and say if it does.
f. Which of your earlier critiques of the current plan still hold, and which fall with (1).
````

## Astra's revision (second answer)

<a id="a-astra-strain-proxy"></a>**Proposal (Astra, answer to correction prompt version 3, pasted by the human 2026-10-07; not decided; verbatim [private, local](private/human-inputs/b87b7677-58c6-4a19-875a-6ccd15205a4f.md#astra-response-2)).** Agent summary:
- **Reading of "skeleton-derived", for the human to confirm:** the head times define a deterministic reference workload and the remaining opportunities; the generated arrangement determines the workload produced; the request is their relation. A head-only quantity alone cannot measure phase C's response.
- **Proxy v1, "relative attack strain":** four lane states x and one overall state g, decayed with osu!mania's bases (τ_I = 0.481 s from 0.125, τ_G = 0.831 s from 0.30); a row's head mask q (chord size m) adds q and m. Each row is charged the rise in the potential Ψ = (w_I τ_I/2)Σx² + (w_G τ_G/2)g²: ΔW(q) = w_I τ_I (x⁻·q + m/2) + w_G τ_G (g⁻m + m²/2), w_I = 1, w_G = 0.25 to start. W(S) is the sum of charges of in-scope head rows; strain runs continuously, never reset at a boundary. Taps and LN heads count the same; holds, releases and reading are out of v1.
- **Reference and request:** W_H(S) from one tap per head row on a fixed lane cycle (0, 2, 1, 3); request r* on r(S) = sqrt(W/W_H), skeleton-relative, not a cross-song scale. Exact remaining budget B_k = r*²W_H − W_committed; W and W_H add over adjacent scopes. Defined for any scope with a head; first supported range 2-16 s with at least 8 head rows, then 32 and 60 s. Settles when the last in-scope head row is committed.
- **Actuator:** a tilt on the 15 non-empty head masks, P(q) ∝ P_0(q)·exp(η ΔW(q)/s_S), added as a head-mask bias to the 625 row logits; keeps p_N(action | head mask), so tap/LN typing and release categories stay natural.
- **Training:** no source-target CE for difficulty. A counterfactual teacher on source states with independently drawn targets allocates the budget by the reference clock and solves for η per state (KL-limited); a small controller learns it by KL; then a little outcome training (Huber on log r, score-function gradient). States and the 15 mask probabilities cached in a bank.
- **Stages:** C0 proxy and data contracts (mechanical tests, a 40-60 pair blinded screen, style label masks); C1 meaning (LN and style source CE, difficulty teacher KL); C2 short-scope completion with KL normalised by the scope's head-row count; C3 combination (difficulty tilts head masks, LN tilts the tap/LN split within them, style adapters underneath); C4 lengths and adherence strength.
- **First difficulty pilot:** a fixed-η scan (64 fitting-split continuations), 2M counterfactual exposures, 192 held-out requested continuations (8 anchors × 2/4/8/16 s × 3 targets × 2 seeds) plus 64 phase-N ones, 128 outcome scopes, re-evaluation: about 72 min at the measured 60 s cost, possibly about 24 min for short scopes. The four-arm LN study stays (about 100 min).
- **Withdrawn from the first answer:** source-cell CE for difficulty, tiled star as the outcome, the 30 s floor, b(H,S) and d* − b, waiting for LN tails, L_star. Kept: source-state success need not transfer to generated histories; a frozen base does not keep active outputs natural; bypassed CE gives no gradient; length-normalised regularisation; adapters do not compose by default.

<a id="a-strain-proxy-reading"></a>**Agent reading (main thread, 2026-10-07; checked against the code at `3520b71` and [o-phasec-rollout-cost](#o-phasec-rollout-cost), nothing run).**
- **Mathematically sound.** ΔW is the potential difference as stated; summed over a whole chart it equals the integral of w_I Σx² + w_G g² over time, so each row is charged the future squared strain its attack causes. That is why the charge is additive and immune to placement just before a boundary. Decay bases match the port (`osu_core/difficulty.py:437-438`).
- **How far it "extends" osu!mania** ([d-difficulty-proxy](#d-difficulty-proxy) prefers an extension): it keeps the decay bases and the individual/overall split, but replaces the 400 ms section peaks and top-weighted aggregation with the workload potential, squares and sums lane states where osu! takes the highest lane, and drops osu!'s hold terms (hold factor 1.25 for a head under a longer hold, the logistic release addition; `osu_core/difficulty.py:384-433`). LN-heavy passages therefore read as easy in v1. Whether attack-only is acceptable for the first version is the human's call.
- **Scale:** the cyclic single-tap reference is near the lowest workload a skeleton allows (one head per row, least-recently-used lanes), so r below about 1 will rarely be reachable; requests effectively run from about 1 upward, and "easier" means fewer chords and fewer same-lane repeats than natural. The natural range of r per scope length is unmeasured.
- **Fits the code with small, local changes.** Every row places at least one head, so the 15 masks cover the choice; a per-mask bias on the joint head keeps the identity gate exact (η = 0) and rule L (bias only on decisions that read the request). It bypasses FiLM for difficulty. New code: an O(K) strain trace and mask costs, a request kind in proxy units (replacing the 30 s floor in `request_set.py:120-121` and the star draw), the bias in `action_log_probs` and sampling, tests. LN arms B-D also need the ledger features and the governed-split tilt ([o-phasec-code-audit](#o-phasec-code-audit)).
- **Simplification for the pilot:** the counterfactual teacher (budget allocation plus a one-dimensional η solve over 15 costs) is already a complete controller with no parameters. Evaluating it directly, before training a network to imitate it, tests the proxy and the actuator at almost no cost; a learned controller earns its place only if outcome training beats the analytic one.
- **Compute:** the estimates use our measured rates and are plausible; short scopes are cheaper than the 60 s pipeline but anchors late in long charts pay prefix replay (about 400 decisions/s) and the slower sampler.
- **For the human:** (1) confirm the reading of "skeleton-derived" and a skeleton-relative request r; (2) attack-only v1, or osu!'s hold terms from the start; (3) who judges the 40-60 pair blinded screen; (4) whether to start C0 now, while phase N finishes: proxy module and mechanical tests, the natural r distribution per scope length on fit_dev, and a fixed-η plus analytic-controller scan on the phase-N checkpoint (no training).
