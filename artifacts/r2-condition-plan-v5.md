# R2 v2 training plan (condition plan v5): natural phase, then conditions

Written 2026-10-06 by an Opus subagent (Claude, control plane) for the human's code check. It supersedes [r2-condition-plan-v4](r2-condition-plan-v4.md) where this document says so. Section 7 lists which v4 sections still hold unchanged; for those, v4 (with its [v4-amendments](r2-condition-plan-v4.md#v4-amendments)) remains the reference. The formulation `docs/formulation/style-conditions-and-control.md` (`8da2bda`, branch `docs/style-formulation`) still takes precedence over both plans.

Code: `ensomi-model`, branch `r2/train`, HEAD `7d9640a`, with the uncommitted stage-0 change set plus this revision's edits (section 8). Paths are relative to `src/ensomi_model/r2/` unless they start with `tests/`. Line numbers refer to the working tree after this revision. Nothing is committed.

Mac jobs run for this plan:

- `20261006-122217-r2-window-scopes`: a read-only measurement of scope lengths against the 256-decision window (section 5).
- `20261006-123753-r2-v-share`: a read-only measurement of the share of windows with no visible decision under the phase-C draw.
- `20261006-123305-r2-v5-tests`, `20261006-124124-r2-v5-tests-2`, `20261006-124543-r2-v5-phases`, `20261006-124607-r2-v5-phases-2`, `20261006-124635-r2-v5-tests-3`: the test suite (section 8).

No training ran.

Evidence tags: **[code]** checked at the cited line; **[run]** measured by one of the jobs above; **[inferred]** reasoning, not run; **[proposal]** a value or design the human has not decided.

<a id="v5-direction"></a>
## 0. The direction and the recipe in brief

**The human's direction (2026-10-06).** Training condition-following together with chart structure is not good. Instead:

1. **Phase N** learns natural chart structure by teacher-forced cross-entropy, with no conditions.
2. **Phase C** then tunes the conditions specifically. The natural model is either **frozen** or held close to itself by a **KL** term. The human named both options and has not chosen.

This supersedes Q-E (condition terms on with λ = 1 from the start in arm A1, compared with A0) and the stage-1 two-arm design built on it.

**Further human input, the same day (relayed):**

- Music (audio) input, head-row (timing) generation and any memory work are out of scope for R2 v2. The landmark memory currently cannot be trained effectively; its intended future use is to recall similar earlier patterns when the song returns to them (a chorus).
- R2 v2's condition inputs will include the five Beatmap Lens foundation concepts, in addition to LN share and difficulty: seven condition kinds. Condition-control training needs more parameters than the current conditioning path.

**The recipe in ten lines:**

1. Phase N: fit_train windows of 256 decisions by the v1 start rule, empty tracks, uniform per-decision CE on a fixed divisor; it trains every parameter outside the conditioner modules.
2. Phase N is selected by natural-manifest NLL with guards legal, (i), (iii) and (iv); which phase-N checkpoint starts phase C is open.
3. Phase C loads that checkpoint's natural parameters. The conditioning path keeps its own initialisation, with a zero output layer, so phase C starts exactly at the phase-N model.
4. An identity gate makes FiLM the identity on every decision and candidate pair that reads no interval. Natural decisions therefore never depend on the conditioning parameters.
5. Phase C draws v4's aligned windows (dropout first, onsets in the window, LN runs and whole-song scopes) with residual difficulty against b(S).
6. Its task loss is CE on the decisions that read an interval, plus λ_ln·L_ln (governed LN split) and λ_star·L_star, plus λ_star·μ_star times the approved relaxed-proxy star term.
7. Frozen mode trains the conditioning path alone (42,112 parameters at the default size). Natural decisions stay bit-identical to phase N.
8. KL mode trains every parameter and adds β·KL to a frozen copy of the phase-N model. The direction, the decisions the KL covers, and whether natural decisions keep their CE are config switches. All are null.
9. The conditioning path's width and depth are configs (`film_width`, `film_layers`), with today's size as the default. A per-kind adapter form sized for seven kinds is proposed in section 6, not built.
10. Every undecided value (schedules, budgets, λ, μ, β, the KL switches, `init_from`, N̄) is null in the configs, and the trainer refuses to start until it is set.

---

<a id="v5-standing"></a>
## 1. What stands, what is superseded

**Standing decisions, unchanged by this plan:**

- Requests are targets only; the default strength is the generator's trained operating point. Requests are valid only when added before their start, cannot be cancelled or changed after it, and two targets for one property may not overlap ([d-target-strength](r2-style-formulation-check.md#d-target-strength), [d-formulation-answers-2](r2-style-formulation-check.md#d-formulation-answers-2), [d-plan-v4-answers](r2-style-formulation-check.md#d-plan-v4-answers)).
- A condition's loss acts only where the condition is visible to the decision (ownership = visibility, rule L, `locality.py`). The model sees only the current scope's controls; history tokens carry notes only ([d-condition-scoped-loss](r2-average-and-control.md#d-condition-scoped-loss)).
- Difficulty is a residual against the skeleton baseline b(S) (`baseline.py`). The relaxed-proxy term that pushes the realised star of generated sections toward the request is approved (Q-G (B)). An own-sample LN term is not (Q-I).
- LN training scopes include runs of 2-8 pieces and the whole song (Q-B).
- The learning-rate schedule is Astra's task (Q-F). It now applies per phase.
- Process: no build-time estimates, no AI review gates, no invented training details.

**Superseded:**

- Q-E's arms A0/A1 and λ = 1 from the start.
- The stage-1 claim and thresholds (v4 §13 stage 1). The readiness note's O2, a restatement of that claim, is therefore moot.
- O1 (per-arm condition manifests) is moot: both phase-C modes share one draw, so they share one manifest version (`data.manifest_version` keys on the draw hash).
- The configs `ce_v2_a0.json` and `ce_v2_a1.json` are **retired and deleted**. The phase configs replace them (section 8). The `per_song` selection stays in `conditions.py` but no config uses it.

**Out of scope for R2 v2 (human, 2026-10-06):**

- Audio input and head-row timing generation.
- Memory work. The memory path is unchanged: landmarks every 64 TCN outputs, read by single-head attention through a zero-initialised `lm_out` (`model.py`, `read_landmarks`). Whether to keep it as it is or set `memory='none'` is open decision 11.

---

<a id="v5-phase-n"></a>
## 2. Phase N: natural structure

**Data and draw.**
- Charts come from fit_train: song group uniform, then chart uniform in the group (`data.Corpus.pick`).
- The draw is `DrawConfig(selection='natural', p_align=0, ipw=False)` (`conditions.py:266`). It uses the v1 start rule (j = 0 with p 0.125, j = max(0, K - 255) with p 0.125, else uniform on [0, K]), the window [j, min(j + 256, K + 1)), an empty track and weight 1.
- No candidates are drawn, so no star labels and no baseline are needed (`star_conditions: off`).
- Batches hold 4 windows (`accumulate`).

**Loss.**
- L = L_base = (1/N̄) Σ_j ℓ_j over every scored decision, ℓ_j = -(log P(A_j) + log Q(U_j)). EOS contributes only its releases.
- N̄ is the expected number of scored decisions per batch, from `draw_sim` on `ce_v2_n.json` (N̄_ln = N̄_star = 0 for this draw).
- There is no window mean, no IPW (weight 1) and no condition term. `check_config` refuses λ, μ and any phase-C key in phase N (`train_ce.py:165`).

**What trains.** `trainable_parameters` (`train_ce.py:211`) returns `R2Model.parameter_split()`'s natural list (`model.py:194`): every parameter outside the two conditioner modules.
- With the default model that is 2,404,724 - 42,112 (`film`) - 102,144 (`tokens`) = 2,260,468 parameters [inferred from the architecture note's counts].
- The conditioner modules get `requires_grad=False` and are not in the optimiser.
- On an empty track FiLM is never evaluated (section 3.2), so it stays at its initialisation. `test_train_step.py` asserts that no gradient reaches it.

**Evaluation.** `Evaluator(conditions=False)` (`evaluate.py:188`, `:348`) runs only the natural parts:
- teacher-forced natural-manifest NLL at every checkpoint;
- at full evaluations: natural from BOS, the prefix panel (guard (i)), own-history calibration (guard (iii)), legality, defects (guard (iv)) and G3 (c) at `g3c_exposures`.

The condition manifests, the counterfactual, the probe, the onset, whole-song, switch and residual panels, and G3 (a1), (a2), (b), (f) are skipped. A phase-N model's outputs are identical with and without a request (test `test_film_capacity_is_configurable_and_starts_at_identity`), so those numbers would be constants and a cost of about 15 minutes per full evaluation [inferred from v4 §9.8].

**Checkpoint selection.** `select.py` reads `phase` from the run's `config.json` (`select.py:57`, `:70`).
- Candidates, the primary (natural-manifest NLL, mean of the checkpoint and its two predecessors, song-group bootstrap) and the rule (the earliest passing candidate within 2 SE of the passing minimum) are v4's.
- The guards are legal, (i), (iii) and (iv); (ii) and (a1) need conditions and are dropped.
- No adherence report is written: a natural model has no targets.
- Which phase-N checkpoint initialises phase C is open decision 7. Proposal: the one `select.py` picks.

**Budget, schedule.** Both are null. The schedule is Astra's task. The phase-N budget is open decision 6, with a proposal there.

---

<a id="v5-phase-c"></a>
## 3. Phase C: conditions

### 3.1 Start

`init_from` names a phase-N checkpoint. `load_phase_n` (`train_ce.py:244`):
- refuses a checkpoint whose config is not `phase: natural`, or whose natural architecture (`hidden, levels, expansion, rank, memory, stride, code_dim`) differs;
- loads only the natural parameters (the strict check covers them);
- leaves the conditioning path at the phase-C model's own initialisation, whose output layer is zero.

So the conditioning path may be sized independently of phase N (section 6), and at step 0 phase C computes exactly the phase-N model on every decision (tests `test_kl_is_zero_when_the_model_equals_its_reference`, `test_checkpoint_round_trip_across_phases`). Exposures restart at 0. The init record (path, SHA-256, phase-N exposures, phase-N recipe hash, model config) goes into every phase-C checkpoint payload and receipt. A resume refuses a checkpoint of another phase, another base mode or another init file (`train_ce.py:538`).

<a id="v5-gate"></a>
### 3.2 The identity gate

FiLM computes z + γ(c) ⊙ LayerNorm(z) + δ(c) from a frame row c. A decision that reads no interval has c = 0, so before this revision its output was z + γ(0) ⊙ LN(z) + δ(0), a learned constant transform. Training FiLM, even with a frozen base, therefore moved every natural decision.

The gate (`model.py:104`):
- A frame row that is all zero passes through unchanged: FiLM(z, 0) = z, exactly, through `torch.where`. When no row in the call reads anything, the MLP is not evaluated at all.
- The gate acts per row: on the two hand vectors of a decision, and on each candidate pair of the pointer.
- A decision that reads no interval has all-zero rows, because under presence `none` and rule L the frames take only V_k (`features.frames`). Conversely, an interval in V_k is active at every candidate time of that decision, and at its row time except at EOS when b < T (rule L). An active interval sets the `active` channel to 1, so its rows are non-zero. The only way a non-visible decision can carry a non-zero row is presence `anywhere`, which is test-only. [code, `features.py` `frames`; `locality.py`]

**Consequences:**
1. Natural decisions never depend on the conditioning parameters, in any phase. Under a frozen base, every decision with V_k empty, and every decision on an empty track, equals the phase-N model bit for bit. Tested in CPU float64: `test_gate_conditioned_model_equals_phase_n_on_an_empty_track` (teacher-forced and a free run) and `test_gate_decisions_outside_v_equal_phase_n_and_inside_v_differ`. Power check: with `identity_gate=False` (a test-only switch) the same perturbation moves natural decisions (`test_gate_power_without_it_training_film_moves_natural_decisions`).
2. Rule L's locality conditions are unchanged, and both now hold with respect to the conditioning parameters as well.
3. This changes the v2 stage-0 model, where natural decisions did pass through FiLM(z, 0). Every existing locality, lesion, mirror and normalisation test passes with the gate (section 8).
4. "Explicit 0" still differs from "no request": an explicit LN 0 has value channel -1 and active 1, so its row is not zero (T-Z passes).

### 3.3 Frozen base (`base_mode: frozen`, `configs/ce_v2_c_frozen.json`)

**Trainable set.** The configured conditioner module, defined in one place (`R2Model.parameter_split`, `CONDITIONERS` at `model.py:31`). Under FiLM that is `film.mlp.*` and `film.norm.*`:
- at the default size: Linear(68→128), GELU, Linear(128→256), LayerNorm(128) = 42,112 parameters;
- for width w and L hidden layers: 68w + w + (L - 1)(w² + w) + 256w + 256 + 256.

Everything else, including the unused token conditioner, has `requires_grad=False` and is not in the optimiser (`trainable_parameters`, `make_optimizer`). Tests:
- `test_frozen_phase_c_trains_exactly_the_conditioning_path` (with a widened path, `film_width=64, film_layers=2`): the trainable set equals the `film.*` set, every other parameter is bit-identical after an AdamW step with weight decay, and `film` moved;
- the cache-backed round trip: the natural state stays bit-identical to phase N across two trainer steps.

**What the base being frozen means:**
- The natural NLL, the natural panels and guards (i) and (iii) on natural runs are those of the selected phase-N checkpoint.
- A request can change only decisions in its V. It does so through an affine modulation of the two hand vectors (which feed the 625-way head and the pointer base) and of each pointer query.
- A window whose decisions all have V_k empty has no gradient. The trainer skips its backward pass (`train_ce.py:597`; `test_a_frozen_window_without_visible_decisions_has_no_gradient`).
- Under the phase-C draw with LN only, 46.3 % of windows have no visible decision and 38.9 % of scored head decisions are in V [run, `20261006-123753-r2-v-share`, 1,000 draws; star conditions were off because the v2 labels do not exist yet, so with star on both numbers change]. In frozen mode those windows cost a forward pass and teach nothing. Rejecting them in the draw is a throughput option that changes N̄ and the draw hash; it is not built.

### 3.4 KL-held base (`base_mode: kl`, `configs/ce_v2_c_kl.json`)

**Trainable set.** Natural plus conditioning parameters: everything except the unused conditioner.

**Reference.** A frozen copy of the `init_from` phase-N model, built from its own model config and loaded strictly (`reference_model`, `train_ce.py:268`). It is in eval mode with `requires_grad=False` and is evaluated on the same window with the empty track. It reads no condition; its own conditioning path is at its zero initialisation anyway.

**KL per decision** (`loss.decision_kl`, `loss.py:90`):

KL_j = KL over the 625 legal actions of the action factor + ½ Σ over both orientations of Σ over that orientation's release factors of KL over the factor's candidates.

- The action part is exact.
- The release part is the KL of each directed pointer factor on the teacher-forced context: the source's codes and its earlier placements in that orientation. It is averaged over the two orientations because the release likelihood is their mixture.
- The KL of the mixture itself is not decomposable, so this is the per-factor teacher-forced KL, the analogue of a per-token KL on a data sequence. `window(..., pairs=True)` keeps every candidate's log-probability for it (`model.py:297`, `:338`, `:415`). Masked actions enter as 0 before any arithmetic, so no NaN reaches a gradient (checked in `test_kl_scope_power_and_value`).

**Switches.** All are null until decided (open decision 2):

| Key | Values | What it changes |
| --- | --- | --- |
| `kl_direction` | `forward` = KL(reference ‖ model), `reverse` = KL(model ‖ reference) | Forward is CE against the reference's distribution on the data's states: it penalises losing mass the reference has, so it keeps the phase-N variety. Reverse penalises putting mass where the reference has little: mode-seeking, the RLHF form. Both are exact here and zero iff the factors agree |
| `kl_decisions` | `natural` (V_k empty) or `all` | `natural` holds natural decisions only; under the gate they move only through shared natural parameters. `all` also penalises the conditioned decisions' distance from the natural prediction, so it opposes the condition terms. It acts like a strength knob (a smaller β admits more response), which the formulation reserves for strength levels above the default |
| `natural_ce` | true / false | Whether L_base still scores natural decisions by CE in phase C. False keeps phase C to "tune the conditions"; true continues fitting phase N's data under the KL leash |
| `kl_weight` β | ≥ 0 | Strength of the hold. No evidence for a value yet |

**How it combines.**

L = L_base^B + λ_ln L_ln + λ_star L_star + λ_star μ_star L_proxy + β L_kl, with L_kl = (1/N̄) Σ_{j∈D} w_j KL_j.

- D is set by `kl_decisions`.
- w_j is the IPW weight u on natural decisions and 1 on decisions in V.
- B is every decision if `natural_ce`, else the decisions in V (`loss.window_loss`, `loss.py:124`).
- Using N̄ as the KL divisor makes β "per expected scored decision". Another divisor would only rescale β.

Tests:
- `test_kl_is_zero_when_the_model_equals_its_reference`: four cases (2 directions × 2 decision sets) give exactly 0 in CPU float64, on a phase-C start model with a non-empty track.
- `test_kl_scope_power_and_value`: with only the conditioning path perturbed, `natural` gives 0 and `all` gives > 0. With a natural parameter moved, every natural decision has KL > 0, and the action KL of one decision equals a direct numpy computation within 1e-10.
- The cache-backed round trip: the KL of the first KL-mode step is below 1e-6.

**Trade-off between the two modes** [inferred; nothing has been trained]:

| | Frozen | KL |
| --- | --- | --- |
| Natural behaviour | Identical to phase N by construction; guard (i) and natural NLL need no re-check | Drifts by an amount set by β; natural guards must be re-measured at every phase-C checkpoint |
| Capacity for conditions | Only the conditioning path (affine modulation at two sites) | Every parameter can serve the condition response, e.g. hand representations that encode what conditions need |
| Plug-in of later kinds | A new kind trained alone cannot disturb natural decisions; with per-kind adapters (section 6) not the other kinds either | Retraining shared parameters for a new kind moves earlier kinds and natural decisions within the KL bound |
| Cost per step | No backward through the TCN; windows without V are free of gradient | One extra reference forward per window (no grad) and a full backward; also the per-candidate log-probabilities |
| Failure mode | Under-fitting conditions if the path is too small (the human's capacity concern, section 6) | Natural quality traded for condition response; β must be chosen |
| What it says about the human's goal | Answers "can conditions be learned on top of a fixed natural model?" | Answers "how much natural change does condition-following need?" |

A natural sequence [proposal]: frozen first, because it is cheaper and its natural guard is free. KL follows only if frozen mode's conditioned-decision NLL or onset-panel adherence stalls while the path is already at its proposed size. Open decision 1.

<a id="v5-terms"></a>
### 3.5 L_base, the condition terms, IPW and the aligned draw in phase C

- **Draw.** v4 §8.1 with the amendments, unchanged.
  - Candidates: LN pieces of 8/16/32/64 beats, runs of 2-8 pieces (p 0.25), the whole song (p 0.10); star cells of 30/60/120 s at 3 phases, or the whole song (p 0.10).
  - Dropout first (0.20 all, 0.25 per kind, 0.20 per candidate).
  - Alignment with p 0.6 and lead ≤ 32.
  - 1-3 intervals per kind per window, consecutive with p 0.5.
  - These are v4's stated defaults, kept behind `DrawConfig` (the human has not ruled on them individually).
  - Both phase-C configs use the same draw, hence the same N̄ key and the same condition manifest.
- **L_base.** Restricted to decisions in V unless `natural_ce`. Under a frozen base, natural decisions have no gradient, so including them would change only the logged value; `check_config` keeps `natural_ce` null there.
- **Condition terms.** Unchanged from v4 §6.2-6.3: L_ln is the governed tap-versus-LN split on Ω_LN, L_star the whole decision on Ω_star, ownership = visibility. λ_ln and λ_star are null (open decision 3; proposal 1, Q-E's value for "terms on").
- **IPW.** Its weights touch only decisions with V_k empty. Frozen mode: no effect. KL mode: they weight the KL on natural decisions, and with `natural_ce` their CE, so that both see the natural state distribution of the v1 start rule (the design rule `design.md:303`).
- **Divisors.** N̄, N̄_ln and N̄_star come from one `draw_sim` run on `ce_v2_c_frozen.json`. The trainer refuses phase C without N̄_ln, and without N̄_star when star conditions are on. Before this revision a missing N̄_κ silently zeroed its term.

### 3.6 Difficulty: the b(S) residual and the relaxed-proxy term

- **Residual value.** The phase-C configs set `star_value: residual`: the frame value is target - b(S), clipped at ±1.5 star and scaled by /0.5 (`features.encode_value`). This is the standing difficulty design (H6g).
  - It needs the stage-0 baseline fit (`baseline.py`, not yet run) and the v2 labels (`labels.py`, not yet run). The trainer refuses without them.
  - `star_conditions: on` now refuses when the v2 label file is incomplete (`train_ce.star_setting`, `train_ce.py:359`). Before, `on` and `auto` both trained without star intervals silently (readiness O5).
  - v4's stage-2 comparison of absolute against residual star (B0/B1) is not built into the phase configs. Whether it is wanted is open decision 12.
- **Relaxed proxy.** The approved term (Q-G (B)) stays as built:
  - On one window in `proxy_every` (4), sample the first difficulty scope a scored decision reads, from the real prefix with the current model.
  - Re-score that scope teacher-forced and add λ_star μ_star (g(E_θ[proxies]) - v_res)².
  - The gradient flows through the action probabilities of decisions in V (`proxy.py`). Under a frozen base it therefore reaches only the conditioning path, and its own samples differ from phase N's only inside V.
  - True F3 stays a no-gradient measurement every 16 steps.
  - μ_star is null (open decision 4): no weight was ever set.
  - Readiness O3 still applies: the proxy and F3 draw their sampling seeds from the training draw RNG, so a config with μ > 0 sees a different window sequence from one without.
- **Cost on long scopes.** The proxy samples and re-scores the whole scope. Difficulty scopes are long: 60 s cells have a median of 405 head rows, 120 s cells 858, whole songs 979 (p90 2,616, max 12,401) [run, section 5]. A whole-song or 120 s scope means a sampled continuation of hundreds to thousands of decisions plus a teacher-forced window of the same length inside one training step, on one window in four. That is readiness O7, now quantified. Whether proxy scopes are limited (by class or by length) is part of open decision 4. The cost needs the pilot.

### 3.7 Evaluation in phase C

v4 §9 unchanged: both manifests, per-kind and per-stratum NLL, null contrast, counterfactual, probe, all free-run panels, calibration, G3. In frozen mode the natural-manifest NLL and the natural panels must reproduce the init checkpoint's numbers. A difference means the gate or the freeze is broken. This is an engineering check, not a claim.

<a id="v5-operating-point"></a>
### 3.8 The default operating point and checkpoint selection now

The default strength is the generator's trained and validated operating point (formulation, "Target strength"). With two phases that generator is defined by:

1. the phase-N recipe and the phase-N checkpoint used as `init_from`;
2. the phase-C recipe (config hash);
3. the phase-C selection rule with its guards;
4. the decoding (Gumbel maximum, temperature 1, fair orientation, one lane at a time);
5. the conditioning (rule L, presence `none`, default η, FiLM with its size).

Changing any of them defines a different default whose adherence is measured again.

**Code status.** The phase-C checkpoint payload and receipt carry the init record (SHA-256 and recipe hash of the phase-N checkpoint). `operating_point.recipe_hash` still hashes only the phase-C `config.json`, which names `init_from` by path, not by content. Including the init record in the operating-point hash is not done (section 9); it waits on open decision 8.

**Phase-C selection** is v4's rule as implemented. Its primary, natural-manifest NLL, is constant across phase-C checkpoints under a frozen base (section 3.3). There the rule reduces to "the earliest checkpoint passing every guard". Under KL it varies with β. What phase C's primary should be is open decision 8, with three options there.

---

<a id="v5-tests-items"></a>
## 4. Stage-0 items and tests: what changes

| Item | v4 / readiness status | v5 |
| --- | --- | --- |
| Three failing causes (19 tests) | open | **fixed**: `generate.py:108` record keys; `features.py:393` token value via `encode_value` (with the run's `star_value`); the short-window draw crash (section 8) |
| 0.1-0.2 loss | implemented | extended: `base_conditioned`, `natural_ce`, KL term (`loss.py`) |
| 0.3 draw, `draw_sim` | implemented, not run | + `selection='natural'`; lead capped at window - 1; the aligned candidate is always selectable. `draw_sim` runs once for `ce_v2_n.json` and once for the phase-C draw, not for A0/A1 |
| 0.4 relabel, 0.5 baseline refit | not run | now needed before phase C (residual star; `star_conditions: on`) |
| 0.7 manifests | per-arm (O1) | one natural manifest; one condition manifest shared by both phase-C modes |
| 0.8 evaluation | implemented, untested | phase-aware (`Evaluator(conditions=...)`); still no test covers the evaluator |
| 0.9 guard, restart budget, logs | implemented | unchanged |
| 0.10 DPO anchor on L_base | missing | still missing; DPO waits for real pairs |
| 0.15 rule L, birth role, tokens raise | partial (DEVIATIONS rewrite missing) | unchanged; DEVIATIONS still not rewritten |
| 0.19 operating point | implemented | selection rule notes phase N; init record not yet hashed into the operating point |
| 0.21 style naming, 0.22 tail-share script | missing | still missing |
| new 0.23 identity gate | none | built, tested (section 3.2) |
| new 0.24 phases N and C in the trainer, `init_from`, `base_mode`, KL | none | built, tested |
| new 0.25 phase configs, `check_config` | none | built, tested; A0/A1 deleted |
| new 0.26 conditioning capacity (`film_width`, `film_layers`) | none | built, default unchanged (42,112), tested |
| new 0.27 phase-aware selection | none | built (no test, like the rest of `select.py`) |

**Tests changed:**
- `tests/r2/test_draw.py`: the A0/A1 config-diff test is replaced by a natural-draw test and a short-window test, and the stored-N̄ test runs over the three phase configs.
- `tests/r2/test_train_step.py`: runs phase N; asserts no gradient reaches the conditioning path; the draw-change refusal uses the natural draw.
- New `tests/r2/test_phases.py`: 19 tests (section 8).

**Stage-0 mac measurements still to run, in order:**
1. relabel;
2. baseline refit;
3. `draw_sim` for phase N and for phase C;
4. pilots for phase N, frozen C and KL C;
5. the tail share (script still missing);
6. the baseline rows on ckpt-0061432779.

---

<a id="v5-windows"></a>
## 5. Do the 256-decision windows limit phase N or phase C?

**What the window does.** A window is the set of decisions a draw scores.
- It does not bound the context: `window_hands` encodes the history tokens from decision 0 up to the window's last decision for every window (`model.py`, `window_hands`/`encode_history`).
- Every decision therefore sees the TCN state over its full prefix: a receptive field of 511 tokens, plus landmark reads every 64 tokens of the whole prefix.
- Its condition frame describes the whole scope: offsets to a and b, progress, the scope's committed counters since a, and the remaining head rows. These are computed from the chart, not from the window (`features.frames`).
- The CE terms (phase N's L_base, phase C's L_base^V, L_ln, L_star) and the KL term are sums of per-decision quantities. A decision contributes the same term whichever window it falls in; windows are minibatches of decisions.

**Measured scope lengths** [run, `20261006-122217-r2-window-scopes`: 11,368 fit_train charts in 4,167 groups; scope classes on 1,000 charts picked as the draw picks them; head rows in the scope, plus EOS when b = T]:

| Scope class | Head rows p10 / p50 / p90 (max) | Share > 256 |
| --- | --- | --- |
| LN piece 8 beats | 12 / 29 / 56 (109) | 0 % |
| LN piece 16 beats | 26 / 57 / 108 (255) | 0 % |
| LN piece 32 beats | 48 / 112 / 209 (407) | 2.6 % |
| LN piece 64 beats | 86 / 214 / 402 (796) | 35.2 % |
| LN run of 2-8 pieces | 116 / 387 / 962 (3,043) | 69.6 % |
| Whole song (LN or difficulty) | 430 / 979 / 2,616 (12,401) | 98.0 % |
| Difficulty cell 30 s | 119 / 198 / 341 (688) | 29.2 % |
| Difficulty cell 60 s | 252 / 405 / 667 (1,140) | 89.2 % |
| Difficulty cell 120 s | 518 / 858 / 1,318 (2,105) | 99.3 % |

- Whole charts: K + 1 > 256 for 97.8 % of charts (97.5 % weighted as the draw picks them). The draw-weighted K quantiles are 413 / 962 / 2,516.
- Head rows per canonical beat: median 3.5 (p10 2.0, p90 5.8).

**Under the phase-C aligned draw** [run, same job, 600 draws, LN only because the v2 star labels do not exist]:
- Of 556 LN intervals with a visible decision in their window, 23.4 % have more visible decisions than a window holds.
- The share of an interval's visible decisions that lie inside the window has median 0.95 and p10 0.11.
- By class:
  - pieces (n 462): 12 % longer than the window, mean share inside 0.75;
  - runs (n 66): 68 % longer, median share inside 0.50;
  - whole song (n 28): all longer, median share inside 0.25.

**Phase N.** The window does not limit what phase N can learn. Per-decision CE is the same whatever window a decision is scored in, and the context is the full prefix. Three practical effects:
1. The start rule fixes how often each position is scored. It oversamples BOS and the last 256 decisions by 0.125 each, as in v1, and so does the natural manifest.
2. Encoding from decision 0 makes a late window cost O(k) TCN work. That is a throughput fact for the pilot.
3. Gradients within a window are correlated (one chart).

The limits that matter for natural structure are elsewhere: teacher forcing (the own-history gap, 0.129 ± 0.044 at 61.4M in v1) and the memory path, which is out of scope.

**Phase C, teacher-forced terms.** No limit either, for the same reason: a decision deep inside a 1,000-row scope has the same frame and the same term in any window. Two effects of the window:
1. **Where long scopes get scored.** Aligned windows score the first ≤ 256 visible decisions after an onset (onset lead ≤ 32). The interior and end of runs, whole songs and 60/120 s cells are scored only by the other 40 % of windows and by aligned windows of other intervals that happen to reach them. Measured: the median aligned window covers half of a run and a quarter of a whole song. This is a question of where exposure goes, and the onset/middle/end strata of v4 §9.1 report it. It is not a limit on what can be learned.
2. **Aggregates the window cannot see.** The target of a request is an aggregate: the scope's LN share or tiled star of the generated chart. Teacher-forced CE scores the source's decisions given the source's history, and the frame's committed counters are the source's own. No window length changes that: a window spanning the whole scope would still score only source decisions. What CE cannot see is whether the model's own continuation across the scope reaches the target. That is a teacher-forcing limit, not a window limit.

**Phase C, own-sample terms.** These are where aggregates are trained.
- For difficulty, the relaxed proxy already ignores the 256-decision window: it samples the scope from its onset with the model and re-scores the whole scope in one window of the scope's length. `model.window` accepts any length.
- Its cost scales with the scope: up to thousands of decisions for whole songs (section 3.6).
- The LN analogue (an own-sample LN term, v4 stage 3) is the only route to the realised LN aggregate, and Q-I has not approved it.

**Conclusion.** The 256-decision window limits neither phase. Long scopes are learned from their frames per decision. What the window choice affects is where exposure lands inside long scopes, and that is measured. What no teacher-forced recipe of any window length can train is the realised aggregate. For difficulty the relaxed proxy covers it, at a cost that grows with scope length. For LN only an own-sample term would, which waits on Q-I.

---

<a id="v5-capacity"></a>
## 6. Conditioning capacity and new condition kinds

### 6.1 What the code allows now

- One shared FiLM module (`model.py:82`). Its input is 2 roles (row, candidate) × 2 kinds × 17 channels = 68, then a hidden layer of 128 and an output of 256 = (γ, δ) for the 128-wide hidden vector, plus the LayerNorm.
- The same weights modulate the two hand vectors (after fuse and the landmark read) and every pointer query.
- **New:** `film_width` and `film_layers` (R2Config and TrainConfig; default 128 and 1, the v2 size, 42,112 parameters; the default build is bit-identical to before). `film_width=512, film_layers=2` gives 68·512 + 512 + 512² + 512 + 512·256 + 256 + 256 = 429,568 parameters [inferred arithmetic; the 256/2 case is asserted in `test_film_capacity_is_configurable_and_starts_at_identity`].
- Frozen mode trains exactly this module at any size; the test checks this with a widened path.
- Per-kind adapters, more insertion points and Lens inputs are **not built**.

**Widening for a new kind.** A new kind adds 2 × 17 input columns to the first linear layer, zero-initialised, so the model is unchanged at the start. Two ways to train it:
- Train the whole module. The new kind can then shift the shared hidden units that the earlier kinds use, so earlier kinds' responses move.
- Train only the new columns (a gradient mask). Every decision without the new kind stays exact, because its columns multiply zero inputs. The new kind then has to work through hidden units fitted for the earlier kinds, which limits it.

### 6.2 Seven kinds: form and size [proposal]

The direction: seven condition kinds (LN share, difficulty, five Lens foundation concepts) need more conditioning capacity than one 42,112-parameter FiLM.

**Proposed form: per-kind adapters.**
- Each kind κ gets its own MLP from its own two-role frame (34 inputs for a 17-channel frame; a Lens frame's width depends on the levels per concept, which are undecided) to (γ_κ, δ_κ) for each insertion site.
- The output layer is zero-initialised, and a per-kind gate opens only when that kind's frame row is non-zero.
- The modulations of the active kinds are summed:

  z' = z + Σ_κ g_κ (γ_κ ⊙ LN(z) + δ_κ)

  with separate (γ, δ) for the hand site and the pointer-query site. Today one set serves both.

Why this form:
1. **Plug-in.** A kind added later is trained alone with everything else frozen. Decisions without that kind stay bit-identical, by the same argument as the identity gate, and earlier kinds are untouched.
2. **Separate data.** The Lens concepts' labels may arrive at different times from different sources; one adapter can train when its data exists.
3. **Separate sites.** The hand vectors and the pointer query are different spaces; separate outputs per site add capacity where releases matter (LN coordination, tech).

The cost: interactions between co-active kinds are additive in (γ, δ). The model's downstream non-linearity (the bilinear joint head, the pointer) gives some interaction, but the conditioning itself does not learn, say, "LN 0.6 and jacks prominent" as a pair.

**Proposed size.**
- Two hidden layers of width 128 per kind:
  - 34·128 + 128 = 4,480;
  - 128·128 + 128 = 16,512;
  - 128·512 + 512 = 66,048;
  - per kind 87,040.
- Seven kinds give 609,280, plus two LayerNorms (512) = **≈ 0.61M**, about 14 times today's FiLM.
- With 2,260,468 natural parameters and the unused token conditioner (102,144), the model has ≈ 2.97M, under the 4.5M cap.
- For the two kinds that exist now, the same form is ≈ 0.17M.
- Larger alternative: width 256 gives 206,336 per kind, ≈ 1.44M for seven, ≈ 3.8M in total.

**Alternatives** [proposal]:
- **Wider shared FiLM** for seven kinds: input 2 × 7 × 17 = 238, two hidden layers of 512 = 516,608 parameters. It learns interactions between kinds but gives up plug-in isolation. It is buildable now with the knobs, except for the input widening.
- **More insertion points.** Add FiLM sites at the pointer base (`pointer_in` output) and at the candidate embedding, or a residual adapter MLP([z, e_c]) that is not affine in z.
  - More sites let a condition change the release side directly. That matters for the LN-coordination and tech concepts, which concern releases.
  - Under rule L, any site inside a visible decision keeps locality.
  - Each site needs its own gate and lesion test.

**A measurement that would test the size, using the knobs that exist** [proposal]: frozen-mode phase C on LN and difficulty at matched exposure, with today's FiLM (42k) against `film_width=512, film_layers=2` (430k). Compare condition-manifest Ω NLL by stratum and the onset-panel slope. A gain over 2 SE says capacity binds at today's size. The human's direction already says it does, so this is a check, not a gate.

### 6.3 Lens condition kinds

Not built: no frames, labels, draws, validator kinds or data. The plan's stage 4 named-attribute path (v4) becomes the route for them. Two open items:
- where the five concepts' section labels come from (a Lens labeller, human labels, deterministic query evidence, or a mix);
- how many levels each concept has (v4's Q-K default was three: absent, supporting, prominent; not decided).

Both are open decision 10. The validator, ν and the overlap rules (same attribute → reject) of v4 §5 and the formulation apply unchanged once the kinds exist.

---

<a id="v5-v4-map"></a>
## 7. Plan v4: what survives

| v4 section | v5 status |
| --- | --- |
| Standing rules 1-5, Terms | unchanged |
| §1 plan paragraph | replaced by §0 here |
| §2 facts | background (line numbers are `7d9640a`'s) |
| §3 rulings | unchanged; the Q-E amendment is superseded |
| §4 locality, rule L, T-V, T-P, T-L, T-C, T-A, T-E | unchanged; the identity gate (§3.2 here) adds the guarantee for the conditioning parameters |
| §5 requests, ν, schema, η, presence, strength, ρ slot, record | unchanged |
| §6 loss | §6.1, 6.3, 6.4, 6.6-6.8 unchanged; §6.2's total is extended by the phase-dependent base set and the KL term (§3.4 here); §6.5 unchanged |
| §7 difficulty | §7.1, 7.2, 7.4, 7.5 unchanged; §7.3's relaxed proxy moves into phase C with μ_star null |
| §8 draw | phase C uses it unchanged; phase N uses the natural draw |
| §9 diagnostics | §9.1-9.5 and 9.7 unchanged for phase C; phase N runs the natural subset; §9.6 is amended by §3.8 here; §9.8's costs drop for phase N |
| §10 exposure bias | unchanged (an own-sample LN term waits on Q-I) |
| §11 infrastructure | unchanged |
| §12 formulation map | unchanged |
| §13 stages | stage 0 revised (§4 here); stages 1 and 2 replaced by phases N and C; stage 3 (own-sample LN, behind Q-I) and stage R (ρ) unchanged; stage 4 becomes the Lens route (§6.3) |
| §14 ledger | unchanged; L29's "≤ 4 passes" belongs to the retired stage 1 and becomes part of open decision 6 |
| §15 questions | Q-E superseded; the others as answered |

---

<a id="v5-code"></a>
## 8. Code changes of this revision (working tree, uncommitted)

**Bugs fixed:**
- `generate.py:108` built each effective-track record entry with `kind`, `a` and `b` passed twice: the provenance dict already holds all three, not only `kind`. It now uses `dict(p, frame_value=iv.value)`.
- `features.tokens` called the removed `_norm_value`. It now calls `encode_value(iv, star_value)`, and the model passes its `star_value`.
- `conditions.draw_window` with a window shorter than the lead raised `StopIteration`. The fix:
  - the lead is capped at window - 1, in the draw and in `align_distribution`, so the IPW weights stay exact;
  - `_intersecting` treats a scope with b = T as containing T;
  - the aligned candidate is always added to the selectable set.

  At window 256 with lead ≤ 32 only the EOS-only-window edge changes.

**New mechanisms:**

| Module | Change |
| --- | --- |
| `model.py` | `CONDITIONERS`, `R2Model.parameter_split`; FiLM width and depth and the identity gate (`R2Config.film_width`, `film_layers`, `identity_gate`); `window(..., pairs=True)` and `release_and_pairs` / `factor_pair_log_probs` (the old methods keep their signatures) |
| `loss.py` | `base_conditioned`, `LossConfig.natural_ce`, `kl_weight`; `decision_kl`, `window_kl`; `window_loss(..., kl)` |
| `conditions.py` | `selection='natural'`, `SELECTIONS`, `lead_max`, the three draw fixes |
| `train_ce.py` | `TrainConfig.phase, init_from, base_mode, natural_ce, kl_weight, kl_direction, kl_decisions, film_width, film_layers`. Undecided keys default to `None`: λ, μ, the budget and the lr keys (the old class defaults 3e-4 and 8M are gone). New `check_config`, `trainable_parameters`, `make_optimizer`, `load_phase_n`, `reference_model`; KL in `step`; phase and init checks before `load` touches any state; phase fields in the payload and receipt; `star_conditions: on` refuses without labels |
| `evaluate.py` | `Evaluator(conditions=...)`; the conditioned panels moved into `condition_panels`; the order of random draws is unchanged |
| `select.py`, `operating_point.py` | phase-N guard set; the rule text notes it |
| `draw_sim.py` docstring, `README.md` | phase configs; README "Training recipe: two phases", pilot and launch commands |
| `configs/` | `ce_v2_n.json`, `ce_v2_c_frozen.json`, `ce_v2_c_kl.json` added; `ce_v2_a0.json`, `ce_v2_a1.json` deleted; `ce_v1.json` untouched (it now fails `check_config`, as it already failed the N̄ check) |

**Tests** (`tests/r2/test_phases.py`, 19 new):
- gate identity on an empty track (teacher-forced and free run);
- outside V equal and inside V moved;
- gate power;
- capacity knobs and identity at start;
- frozen trainable set and bit-identical natural parameters after an AdamW step;
- phase N leaving the conditioning path and KL mode training both;
- no gradient from a frozen window without V;
- KL exactly zero at equality (4 cases);
- KL scope power and value (2);
- refusal of null keys for each config (3);
- refusal of inapplicable keys;
- configs differ only where intended;
- checkpoint round trip across phases (cache: phase N → frozen C with exact resume and refusals → KL C with KL < 1e-6 on the first step).

Plus the test changes listed in section 4.

**Mac test jobs** (`pytest tests/r2 tests/evaluation`, the command of the readiness check):

| Job | Result |
| --- | --- |
| `20261006-123305-r2-v5-tests` | 254 passed, 5 skipped, 0 failed; 1 warning (a test called `float()` on a tensor that requires grad) |
| `20261006-124124-r2-v5-tests-2` | 254 passed, 5 skipped, 0 failed; 1 warning (the same, another line) |
| `20261006-124543-r2-v5-phases` | `test_phases.py` with warnings as errors: 18 passed, 1 failed (the same kind of warning in the round-trip test) |
| `20261006-124607-r2-v5-phases-2` | `test_phases.py` and `test_train_step.py` with warnings as errors: 22 passed. `Stats.add` and the proxy log now detach before `float()` |
| `20261006-124635-r2-v5-tests-3` | **254 passed, 5 skipped, 0 failed, no warnings**: the final tree |

The 5 skips are the three stored-N̄ tests (N̄ is null) and the two label tests (no v2 labels). Before this revision the same suite gave 19 failed, 215 passed, 4 skipped (`20261006-104814`).

<a id="v5-log"></a>
### Failed paths and surprises

- Changing `release_from_factors` / `factor_log_probs` to return tuples would have broken three tests that monkeypatch them (mirror and DPO controls). New methods carry the per-candidate log-probabilities instead.
- In the first draft of `Trainer.load` the phase checks ran after `load_state_dict`. A phase-N checkpoint's optimiser state would then have failed with a param-group size error instead of the intended refusal. The checks now run first, phase before draw key.
- `generate.py:108` duplicated `a` and `b` as well as `kind`.
- `star_conditions: on` was as silent as `auto` when labels were missing, because `Corpus` gets an empty cell table.
- Under the phase-C draw with LN only, 46 % of windows have no visible decision [run], so frozen mode spends almost half its forward passes on windows with no gradient.
- Scope lengths: 89 % of 60 s cells, 70 % of LN runs and 98 % of whole songs exceed one window [run].

---

<a id="v5-not-done"></a>
## 9. Not done

- The operating-point hash does not yet include the phase-N init record (section 3.8).
- `scripts/r2/smoke.sh` still launches `ce_v1.json`, which `check_config` refuses. It is outside this revision's edit scope.
- No test covers `evaluate.py`'s phase switch or `select.py`'s phase-N guards. No test exists for those modules at all.
- `DEVIATIONS.md` is not rewritten (v4 item 0.15 and the gate).
- No per-kind adapter, extra insertion point or Lens input is built.
- No stage-0 measurement: relabel, baseline, `draw_sim`, pilots, tail share, baseline rows.
- O3 (proxy seeds from the draw RNG) and O6 (in-run records hash the operating point without the recipe) are untouched.
- The DPO anchor is still the v1 window mean.

---

<a id="v5-open"></a>
## 10. Open decisions for the human

1. **Frozen or KL in phase C.** Options: frozen only; KL only; both at matched exposure as a comparison; frozen first, then KL if frozen stalls [proposal: the last].
2. **KL settings, if KL runs:**
   - `kl_direction`: forward KL(ref ‖ model) or reverse [proposal: forward, to keep the phase-N variety];
   - `kl_decisions`: natural only or all [proposal: natural; "all" also penalises the condition response];
   - `natural_ce`: keep CE on natural decisions or not [proposal: no];
   - `kl_weight` β: no proposal until a pilot shows the KL's scale against the condition terms.
3. **λ_ln, λ_star in phase C.** [proposal: 1 each, Q-E's value for "terms on"].
4. **μ_star**, the relaxed-proxy weight (0 switches it off), and whether proxy scopes are limited by class or length because whole-song and 120 s scopes are hundreds to thousands of decisions per sample. No proposal for μ.
5. **Learning-rate schedules per phase (Astra's task).** Also: do the inherited AdamW settings (β 0.9/0.95, weight decay 0.01, clip 1) stay, or go into Astra's brief?
6. **Budgets per phase:** `total_exposures`, `checkpoint_every`, `g3c_exposures`. Phase-C exposures count all head decisions of drawn windows, of which about 39 % are in V with LN only [run]. [proposal: phase N 50M with checkpoints every 4.39M and G3 (c) at 25M and 50M, as v4's stage 1; no proposal for phase C].
7. **Which phase-N checkpoint initialises phase C** (`init_from`). [proposal: the one `select.py` picks under the phase-N guards].
8. **Phase-C checkpoint selection, which fixes the default operating point.** Options:
   - (a) v4's rule as is; under a frozen base it becomes "earliest passing all guards";
   - (b) primary = condition-manifest NLL on Ω (teacher-forced);
   - (c) primary = free-run adherence on a selection panel kept separate from the reported panels (standing rule 5).

   Also: should the operating-point hash cover the init checkpoint?
9. **Conditioning path size and form.** For now (two kinds): keep 42,112, or widen with `film_width`/`film_layers`. For seven kinds: per-kind adapters ≈ 0.61M [proposal], a wider shared FiLM ≈ 0.52M, or more insertion points.
10. **Lens condition data.** Where the five concepts' section labels come from, and how many levels each concept has. Nothing is built until this is decided.
11. **Memory.** Keep the landmark read as it is, or switch it off with `memory='none'` (the architecture of phase N and C must then match).
12. **Difficulty arms.** Residual only, as the phase-C configs set it, or also an absolute-star run for comparison (v4's B0/B1).
13. **Agent choices to confirm:**
    - phase-C configs use `star_conditions: on` (refuse without labels) and `star_value: residual`;
    - phase N uses the v1 start rule;
    - phase C keeps v4's draw defaults;
    - both phases reuse training seeds 171/471;
    - in frozen mode, windows without a visible decision are drawn and cost a forward pass (a draw that rejects them is possible but changes N̄).

Standing and unchanged: Q-I (own-sample LN term), q7 and ρ (stage R), Q-K (the style vocabulary, now tied to item 10).
