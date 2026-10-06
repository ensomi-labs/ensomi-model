# R2 v2 readiness: evidence for the pre-run code check

What this report covers: branch `r2/train` at HEAD `7d9640a`, plus the uncommitted working tree as of 2026-10-06 about 10:55 UTC. The plan is `r2-condition-plan-v4.md`, with the `v4-amendments` overriding the body.

Mac jobs:
- I ran one job: `20261006-104814` (pytest; exit code 1; finished 10:52:52Z).
- No other mac command was run. Mac file contents were checked only through the `ens ls` index, which runs on the control plane.
- Nothing in the repository or the notes was edited.

Tags: **[inferred]** means reasoning from the code, not something that was run. A passing test is engineering evidence that a mechanism exists. It is not evidence that the model learns.

---

## 1. Test status now

The command was `PYTHONPATH=src .venv/bin/python -m pytest tests/r2 tests/evaluation -q -p no:cacheprovider`. The last line of the log was:

`19 failed, 215 passed, 4 skipped, 1 warning, 264 subtests passed in 268.40s (0:04:28)`

All 19 failures come from three root causes. Every cause line below is quoted from `.sync/mac/jobs/20261006-104814/log`.

| # | Test id | Cause (log) | Root cause |
| --- | --- | --- | --- |
| 1 | tests/r2/test_conditions.py::test_film_frames_hide_future_values_tokens_show_them | `NameError: name '_norm_value' is not defined` | A |
| 2 | tests/r2/test_conditions.py::test_condition_counts_use_committed_heads_only | `NameError: name '_norm_value' is not defined` | A |
| 3 | tests/r2/test_dpo.py::test_delta_is_mirror_invariant[tokens] | `NameError: name '_norm_value' is not defined` | A |
| 4 | tests/r2/test_dpo.py::test_cli_start_and_resume_on_the_cache | `StopIteration` (`src/ensomi_model/r2/conditions.py:307`) | B |
| 5 | tests/r2/test_locality.py::test_t_p3b_token_conditioner_breaks_locality | `NameError: name '_norm_value' is not defined` | A |
| 6 | tests/r2/test_mirror.py::test_mirror_identity_cpu_float64[tokens] | `NameError: name '_norm_value' is not defined` | A |
| 7 | tests/r2/test_normalisation.py::test_decision_probabilities_sum_to_one | `NameError: name '_norm_value' is not defined` | A |
| 8 | tests/r2/test_requests.py::test_t_a_validity | `TypeError: dict() got multiple values for keyword argument 'kind'` | C |
| 9 | tests/r2/test_requests.py::test_t_c_withdrawal_and_no_cancellation_after_the_start | same TypeError | C |
| 10-15 | tests/r2/test_requests.py::test_t_l1_law_before_a[954-5, 954-20, 955-5, 955-20, 956-5, 956-20] | same TypeError | C |
| 16-18 | tests/r2/test_requests.py::test_t_l2_law_after_b[954, 955, 956] | same TypeError | C |
| 19 | tests/r2/test_requests.py::test_record_completeness_and_no_echo | same TypeError | C |

### Root causes

**A: the token conditioner path is broken.**
- `features.py:392` (`tokens()`) still calls `_norm_value(iv)`, but the edit to `features.py` removed that function. It was replaced by `encode_value` at `features.py:320`.
- Only the token conditioner path is affected. The plan keeps the token conditioner, behind its lead-in flag, as the instrument for a future lead-in policy (plan §13, line 730).
- **Six** tests fail from this. The default FiLM path is not affected.

**B: the aligned draw crashes for windows shorter than the lead.**
- The crash is at `conditions.py:307`: `at = next(i for i, c in enumerate(avail) if c is aligned)` raises because the aligned candidate is not in `avail`.
- `Corpus` now uses the aligned `per_window` draw by default. The DPO helper calls `corpus.draw(rng, window)` with `window = cfg.horizon`, which is 8 in this test (`train_dpo.py:510, 562`).
- The lead `u` can be up to 32 (`conditions.py:61, 284`), so the aligned onset can fall after a short window. `_intersecting` (`conditions.py:248-249, 302`) then drops the aligned candidate. [inferred from the traceback and code]
- For CE training at window 256 this can only happen in one edge case: the aligned candidate is first read by EOS (onset = K, so `chart.time(K) = T`) and u = 0. Then `t_lo = T` and `c.iv.b > t_lo` is false. [inferred, not observed]

**C: every generation that carries a request crashes.**
- The crash is at `generate.py:108`: `dict(kind=KIND_NAME[iv.kind], ..., **p)`. The provenance dict `p` already contains the key `kind` (`request_set.py:247`).
- No generation that carries a request has run, so **T-A, T-C, T-L1, T-L2 and the record-completeness / no-echo test were never actually checked.** Each fails before its first assertion.

### Skipped tests (4)
From their position in the progress line and their skip conditions [inferred]:
- `test_draw.py::test_stored_n_bar_matches_a_fresh_estimate[a0, a1]` skip because `n_bar` is null in both configs (`test_draw.py:99-100`).
- `test_properties.py::test_labels_equal_readouts_on_the_cache` and `test_draw_ln_values_equal_nu_on_the_cache` skip because the v2 label file is absent (`test_properties.py:104, 134`). The mac index shows `artifacts/r2-cache/v1/labels/` holds only `receipt.json`, `star.json.gz` and `star_summary.json`, all dated 2026-10-03.

### Warning
One, quoted from the log:
- `train_ce.py:274: UserWarning: Converting a tensor with requires_grad=True to a scalar`

### What passed
- In `tests/r2`: T-V(1)-(4), T-E, T-G, T-Z, T-P1, T-P2, T-P3a, T-P3b(i)(ii), T-R, T-U, T-M on synthetic charts, all frame-channel and role lesions, the baseline-slot test, the IPW exactness test, the A0/A1 config-diff test, and the train-step, resume and NaN tests.
- All `tests/evaluation` tests passed. None of them is in the failure list.

### Earlier jobs
- **`20261006-091458-r2v2-baseline-tests`**: 49 passed at 09:14-09:18. That was before most of today's edits.
- **`20261006-094149-r2v2-import`**: printed `imports ok`, then `ERROR collecting tests/r2/test_conditions.py ... ImportError: cannot import name 'draw_track'` and `1 error`. Its exit code is still 0, because the command piped pytest through `| tail -30`.
- The new test files were written at 09:46-09:48 UTC (file mtimes), after both jobs: `test_locality.py`, `test_requests.py`, `test_properties.py`, `test_draw.py` and `locality_fixture.py`. Job `20261006-104814` is the first run of them.
- No implementation report survives. That the last edits (`draw_sim.py`, `test_draw.py` at 09:48:42) were never followed by a test run suggests the agent stopped before verifying its work. [inferred]

### Code that no test exercises
I grepped `tests/` for these:
- `evaluate.py`: the in-run evaluation. Train-step tests run with `freerun=False` (`test_train_step.py:32`).
- `select.py`, `operating_point.py` (beyond the record), `baseline.py`, `proxy.py`.
- `Trainer.relaxed_proxy` and `Trainer.f3_calibration` (`train_ce.py:450-512`).
- `labels.relabel` and the `generate` CLI.

---

## 2. Stage-0 coverage (plan §13 as amended; tests of §4.6, §6.4 and the §13 test list)

### Code items

| # | Item | Status | Evidence |
| --- | --- | --- | --- |
| 0.1 | Ownership by visibility; per-factor terms; governed split | implemented | `loss.py:41-68`; `model.py:282-287` (`visible` kinds per decision), `model.py:384-411` (`_lane_groups`, `governed_split`); T-V3, T-G, T-P1, T-P2 pass |
| 0.2 | Uniform base CE, fixed divisors, IPW | implemented | `loss.py:47-68` (weight on decisions with V empty); `conditions.py:295-297`; `test_draw.py:38` passes |
| 0.3 | §8.1 draw behind config; `draw_sim.py` with D1-D8 | implemented, **not run**; draw crash (cause B) | `conditions.py:47-322`; `draw_sim.py:34` (thresholds as plan §8.3), `:72-167`. No `artifacts/r2-stage0/` on the mac index |
| 0.4 | Relabel at new lengths, from the cache, through ν | implemented, **not run** | `labels.py` `cells_of`, `relabel`, `load_cells` (ν-hash check); no `star-v2.json.gz` on the mac |
| 0.5 | Baseline refit, residual quantiles, within-chart correlation | implemented, **not run**; no tests | `baseline.py:128-221`; `star_informative_rule` reported at `baseline.py:207`. The §7.4 lever (star weight 3 when the rule fires) is not in the draw: `conditions.py:215-216` gives star weight 1 always |
| 0.6 | Frame: one value channel per kind, star proxies, residual encoding, FRAME_DIM 17 | implemented | `features.py:26, 303-318, 320-325, 328-370`; `conditions.py:339` allows a negative residual; lesion tests pass |
| 0.7 | Two manifests with version hashes | implemented, with a deviation | `data.py:123-238`. The condition manifest is keyed on each arm's own draw hash (`data.py:124`), so A0 and A1 build different ones. See agent observation O1 |
| 0.8 | Teacher-forced and free-run evaluation, §9.1-9.5 | implemented, **untested**; breaks at runtime [inferred] | `evaluate.py:202-354` (TF, counterfactual, probe), `:382-551` (panels), `:554-568` (calibration). The onset panel calls `generate` with a request (`evaluate.py:466-467`), so it raises the cause-C TypeError. `eval_and_log` then replaces the whole record with `error` (`train_ce.py:714-721`) and `select.load_evals` drops it (`select.py:34`) |
| 0.9 | Guard, restart budget, log segmentation | implemented | `train_ce.py:60` (swap trip disabled), `:605-620` (RSS growth over 2 GiB between checkpoints), `:346-349` (segmented logs), `:675-677` (`available_bytes`, `pressure_level` in `resource_limit` events); `launch.py:29-30, 90-133` (5 restarts in 6 h, cleared by a new checkpoint) |
| 0.10 | DPO anchor on L_base; DEVIATIONS entry for the panel (L46) | **missing** | `train_dpo.py` unchanged (`git status`); `train_dpo.py:13` still says "the window mean of decision NLL"; `DEVIATIONS.md` unchanged since 2026-10-03 |
| 0.11 | `properties` module with ν: undefined readouts, b = T, open holds, mirror, T-U, T-M | implemented; labels-equal-readouts **skipped** | `properties.py:36-58, 109-199`; T-U, T-M, row/object agreement pass (`test_properties.py:30-88`). The agreement test uses 200 synthetic charts, not the "200 source charts" of §5.1 |
| 0.12 | Request schema and validator; request set across calls; `replace_interval` retired | implemented; T-R passes; **T-A, T-C fail (cause C)** | `request_set.py:87-197`; `replace_interval` and `draw_track` are gone from `src/` (grep). `README.md:23` still lists `replace_interval` |
| 0.13 | Absolute difficulty targets through the validator | implemented | `request_set.py:232-243` (v_res = target − b(S), flagged and clipped beyond 1.5) |
| 0.14 | Presence switch, default `none`; T-Z | implemented | `features.py:340-347`; `model.py:47`; T-Z passes (`test_locality.py:165`) |
| 0.15 | Rule L in `window()` and `continue_chart`; birth role removed; activity closed at T; tokens raise; DEVIATIONS items 2, 5, 6 | **partial** | `locality.py:26-47`; `model.py:46, 206-221, 311-329`; `features.py:280-283`; `model.py:54-56`. The DEVIATIONS rewrite is missing: items 5, 6 and 8 still describe the birth role, the announced whole-track token conditioner and the v1 guard |
| 0.16 | Locality tests T-V (with power), T-P1-T-P3, T-L1, T-L2, T-E | **partial**: T-V, T-P, T-E pass; T-L1, T-L2 never reach an assertion (cause C) | `test_locality.py:47-265`; `test_requests.py:136-178` |
| 0.17 | Generation record with every §5.7 field; completeness test | **broken** (cause C) | `generate.py:94-136`; `test_requests.py:181` fails |
| 0.18 | Prefix panel; guard (i) binding; G3 (a1), (a2), (b), (c), (f); CLI | implemented, **untested**, breaks at runtime (cause C) [inferred] | `evaluate.py:150-165, 434-453, 570-699`; CLI `generate.py:139-193` |
| 0.19 | Operating-point definition; adherence report, hashed | implemented, untested | `operating_point.py:17-65`; `select.py:70-135` |
| 0.20 | `baseline` slot | implemented | `sampling.py:32-37`; `generate.py:118`; test passes |
| 0.21 | Style-attribute naming in the frame code | **missing** | `features.py:28`: "stage-4 style attributes are not framed" |
| 0.22 | Tail-share measurement script (§7.5) | **missing** | No script. Δ_tail appears only in the stage-2 free-run panel (`evaluate.py:530-540`), not as the stage-0 source-chart measurement of 2,000 cells per length |

### Stage-0 mac measurements (§13 table)
None has been run. The mac index has no `r2-stage0/`, no `star-v2*` and no `baseline-v1*`:
- `draw_sim` for A0 and A1, which must produce N̄, D1-D8 and the u distribution;
- the relabel;
- the baseline refit;
- the tail share (script missing);
- the 200-window pilot;
- the baseline rows on ckpt-0061432779 (no v1-code baseline-row script exists; grep for "v1 conditioning" or the checkpoint name in `src/` finds nothing).

The stage-0 suite on the mac is job `20261006-104814`: 19 failures.

### Other tests listed in §13 that are missing
- b(S) is skeleton-only, mirror-invariant and span-local: no test imports `baseline`.
- The weight-sum and stored-N̄ tests exist (`test_draw.py:95`) but skip.
- The stage-1 exposure-accounting test (both arms' 50M checkpoints within 1 % in head decisions) has no code.
- The stage-2/3 engineering tests for the own-sample path (T-P2 on that path, no gradient through sampling) are absent.

### Amendment checks

| Amendment | Status | Evidence |
| --- | --- | --- |
| Q-E: λ = 1 for every kind present; A0 against A1 | implemented in configs | `ce_v2_a1.json:56-57` (1.0, 1.0); `ce_v2_a0.json:56-57` (0.0, 0.0); `loss.py:67`; `test_draw.py:82-90` passes. Caveat: λ_star acts only if star conditions are on, and `"star_conditions": "auto"` (`ce_v2_a*.json:9`) turns them on only when the v2 label file is complete (`train_ce.py:215-218`). It is not built |
| Q-B: LN runs of 2-8 pieces (p 0.25) and the whole song (p 0.10) | implemented | `conditions.py:50-52, 132-164`; D7 in `draw_sim.py:124-129`; request flag range `request_set.py:30` (8-512 beats, or the whole song) |
| q10: no cancel or change once the scope has started; withdraw or replace before it | implemented; **test unverified** | `request_set.py:159-175`. T-C fails at its first `generate` call (cause C), before its raise checks |
| A0 = current recipe on stage-0 code; A1 = new draw + terms | implemented, with caveats | `ce_v2_a0.json:51-57` against `ce_v2_a1.json:51-57`. Both have `lr 0.0003` and `lr_schedule null` (section 3). `n_bar*` are null, and the Trainer refuses to start until draw_sim values are pasted in (`train_ce.py:315-317`) |
| Q-G(B): stage-2 relaxed-proxy star term, true F3 as calibration | code present, untested, no stage-2 config | `train_ce.py:85-88, 420-426, 450-512`; `proxy.py:33-63`; g map `baseline.py:170-173`. It needs `star_value='residual'` and the baseline (`train_ce.py:320-321`); `mu_star` defaults to 0.0 |
| Lr schedule is Astra's task, fixed before arm comparisons | open | the `lr_schedule` mechanism exists (`train_ce.py:140-188`), with no schedule set |

---

## 3. Defaults standing in for undecided training and inference details

Source codes:
- **P**: the plan states the value as its default.
- **H**: the human decided it.
- **V1**: inherited from v1 or the overnight run; the plan is silent.
- **N**: appears nowhere in the plan or notes. ⚑ marks an item to flag.

### Optimiser, schedule, budget (`ce_v2_a0.json` / `ce_v2_a1.json`, identical unless noted; `TrainConfig` in `train_ce.py:65-119`)

| Key | Value | Where | Source |
| --- | --- | --- | --- |
| `lr` | **3e-4** | `ce_v2_a*.json:12`; `train_ce.py:95` | ⚑ Neither the plan's default (1e-3, plan §13 line 737, Q-F(a) line 892) nor the human's decision (Astra sets the schedule). It equals the v1 `TrainConfig` and `ce_v1.json:12` default. The overnight run used 1e-3 (`artifacts/r2-runs/r2-ce-overnight-20261004/config.json` `"lr": 0.001`) |
| `lr_schedule` | null, which falls back to warm-up plus cosine | `ce_v2_a*.json:40`; `train_ce.py:181-188` | open (H: Astra's task) |
| `lr_min` | 3e-5 | `:13` | P (Q-F(a)), now under Astra's task |
| `warmup_exposures` | 50,000 | `:11` | V1 |
| `total_exposures` | 50,000,000 | `:10` | P (§13 stage 1) |
| `weight_decay` | 0.01 | `:17` | P (Q-F(a)) |
| optimiser AdamW, `beta1` 0.9, `beta2` 0.95, `eps` 1e-8 | | `:14-16`; `train_ce.py:326-328` | V1 |
| `clip` | 1.0 | `:18` | V1 |
| `accumulate` 4, `window` 256 | | `:19-20` | P (§7.3, §8.1) |
| `hidden` 128, `expansion` 4, `rank` 16, `levels` 8, `memory` landmarks, `max_parameters` 4.5M | | `train_ce.py:72-77`; `:7-8` | V1 (the plan defers a capacity re-tune until after stage 1, §13 "Deferred") |
| `checkpoint_every` | 4,388,000 | `:21` | P |
| `full_eval_every` | 2 | `:36` | P |
| `g3c_exposures` | [25M, 50M] | `:47-50` | P (§9.7(c)) |
| `seed_weights` / `seed_draws` / `seed_validation` / `freerun_seeds` | 171 / 471 / 954 / 954-956 | `:23-31` | P. The second pair (172/472) has no config yet; the plan runs it after the first read-out |
| `rss_limit_gib` 12, `rss_growth_limit_gib` 2, swap trip off | | `:34, :46`; `train_ce.py:60` | P (Q-D default) |
| `presence` | none | `:37` | H (d-formulation-answers) |
| `rule_l` | true | `:38` | P |
| `star_value` | absolute | `:39` | P (stage 1) |
| `star_conditions` | auto | `:9` | V1. Its consequence is agent observation O5 |
| `lambda_ln`, `lambda_star` | A0 0/0, A1 1/1 | `:56-57` | H (Q-E answer, v4-amendments) |
| `n_bar`, `n_bar_ln`, `n_bar_star`, `n_bar_key` | null | `:42-45` | P (measured by draw_sim) |
| `mu_star` | 0.0 | `train_ce.py:85` | open: the plan gives no stage-2 weight (μ = 1 appears only for the stage-3 arm C1, line 767); no stage-2 config exists |
| `proxy_every` | 4 | `train_ce.py:86` | P (§7.3 "one window in four") |
| `f3_every` 16, `f3_samples` 4 | | `train_ce.py:87-88` | P (Q-G(B) line 898; k = 4, §7.3) |
| `conditioner` | film | `:6` | P / V1 |

### Draw (`DrawConfig`, `conditions.py:47-71`)

| Key | Value | Line | Source |
| --- | --- | --- | --- |
| `ln_beats` | (8, 16, 32, 64) | 49 | P |
| `p_whole_ln` | 0.10 | 50 | H (Q-B(a)) |
| `p_long`, `run_pieces` | 0.25, (2, 8) | 51-52 | H (Q-B(a)) |
| `star_lengths_s`, `star_phases_s` | (30, 60, 120), (0, 10, 20) | 53-54 | P (Q-C default) |
| `p_whole_star` | 0.10 | 55 | P (§8.1, Q-H default) |
| `drop_all`, `drop_kind`, `drop_interval` | 0.20, 0.25, 0.20 | 56-58 | P (§8.1 step 3) |
| `selection` | per_window (A1) / per_song (A0) | 59 | P |
| `p_align` | 0.6 | 60 | P |
| `lead_max` | 32 | 61 | P |
| `informative_rows` 64, `min_heads` 20, `z_min` 2, `informative_weight` 3 | | 62-65 | P |
| `per_window_max` | 3 | 66 | P |
| `per_song_ln_max` 4, `per_song_star_max` 3 | | 67-68 | V1 (plan §2.1) |
| `p_consecutive` | 0.5 | 69 | P |
| `ipw` | true (A1), false (A0) | 70 | P |
| Order in A0: dropout on candidates, then per-song selection | | `conditions.py:258-278` | ⚑ N. v1 applied dropout to the selected per-song track (plan §2.1 row "Dropout … on the per-song track"). The plan does not say which order A0 uses. See agent observation O4 |

### Loss, encoding, request flags

| Item | Value | Where | Source |
| --- | --- | --- | --- |
| Loss form | L_base/N̄ + λ_LN·L_LN/N̄_LN + λ_star·L_star/N̄_star | `loss.py:62-68` | P (§6.2) |
| LN term on the governed split; star term on the whole decision | | `loss.py:55-56` | P (§6.3) |
| Residual frame encoding | v_res/0.5, clipped to ±3 | `features.py:324` | P (§6.8) |
| Absolute star encoding | /4 | `features.py:325` | V1 |
| Residual flag | 1.5 star | `request_set.py:29` | P |
| LN trained-length range | 8-512 beats, or the whole song | `request_set.py:30` | P (§8.2) |
| Star trained lengths ±1 ms | | `request_set.py:233` | N (minor tolerance) |

### Baseline and relaxed proxy

| Item | Value | Where | Source |
| --- | --- | --- | --- |
| b(S) form | quadratic ridge, λ = 1, 23 head-time features | `baseline.py:33-60, 106-114, 163` | P (§7.1, "as A fitted"; A's `probe.py:475` has `ridge_lambda=1`) |
| kNN-32 check | inverse-distance weighted | `baseline.py:166-169` | P names kNN-32; the weighting was not compared with A's |
| g map | linear ridge, λ = 1, on chord/4, same-lane repeat, LN share; held occupancy left out | `baseline.py:37-38, 170-173` | P for the form (§7.3: hold-length features out of g). ⚑ N for λ = 1 on g |
| Within-chart correlation rule | threshold 0.5 | `baseline.py:207` | P (§7.4). The lever is not implemented |
| Relaxed-proxy form | (g(E_θ[proxies over S]) − v_res)², gradient through the action probabilities of decisions in V, one own sample from the real prefix | `train_ce.py:459-479`; `proxy.py:33-63` | P (§7.3) |
| Relaxed-proxy scope choice | the first difficulty interval that a scored decision reads | `train_ce.py:450-457` | N (detail) |
| Relaxed-proxy normalisation | none; added as λ_star·μ_star·own per window | `train_ce.py:423` | P formula (§6.2); no divisor stated |

### Operating point, selection, evaluation

| Item | Value | Where | Source |
| --- | --- | --- | --- |
| Decoding | Gumbel max, temperature 1, fair orientation, no truncation | `operating_point.py:17-18` | P (§9.6, V12) |
| Selection guards | as §9.6 | `select.py:55-67`; `operating_point.py:20-33` | P. One difference: guard (iv)'s release rate compares with the prefix-panel source rate (`select.py:64-65`), where the plan says "the source band's rate" |
| Panel values | random seeds 954-956; onset values {0.05, 0.3, 0.6, 0.9}; whole-song {0, 0.1, 0.3, 0.6, 0.9}; residuals ±0.5/±0.25/0; 2,000 bootstrap resamples | `evaluate.py:31-44` | P (§9.4) |
| Calibration | 96 states, horizon 64 | `evaluate.py:554` | 96 is P (§9.5). ⚑ N for horizon 64: not stated in plan v4 |
| Prefix panel | A's 8 plus two charts per star band 2-5 (one at or below the median length, one above) | `evaluate.py:150-165` | P in outline; the selection rule is N (detail) |
| Residual-star scope start | rounded to 10 s near 1/3 of the song | `evaluate.py:518` | N (detail) |
| Condition manifest | pool of 4,000 draws, greedy quota fill; extra fit_train manifest | `data.py:166-194, 212` | P quotas; mechanism N (detail) |
| draw_sim random seed | 20261006 | `draw_sim.py:182` | N (detail) |

### Not present
- No model dropout layer exists. The plan's "dropout" means condition dropout, covered in the draw table.
- No guidance or classifier-free scale exists anywhere: decoding is plain, as the plan specifies.

---

## 4. Open ML-core decisions

### (a) Blocks stage 1

| Question | Why it matters | Source |
| --- | --- | --- |
| What learning-rate schedule, possibly staged, do both stage-1 arms use? | It sets the optimiser path in both arms. It must be fixed before the arm comparison and be identical across arms. The configs now carry 3e-4 with no schedule | [d-plan-v4-answers](r2-style-formulation-check.md#d-plan-v4-answers) Q-F; [v4-amendments](r2-condition-plan-v4.md#v4-amendments) |
| Agent observation O2: under the amended arms, what are the stage-1 claim and thresholds? | The plan's claim, its config-diff test and the threshold A1 − A0 ≥ +0.15 slope were written for arms that differ only in the draw (plan §13 lines 734-743). With Q-E, A1 also has λ = 1, and the notes record no restated claim, threshold or stop rule. Standing rule 1 requires them before any run | `ce_v2_a1.json:56-57`; plan lines 20, 734-743 |
| Agent observation O1: should both arms share one condition manifest? | The stage-1 threshold "onset-decision NLL paired difference A1 − A0" needs common windows. The code builds a per-arm manifest keyed on each arm's draw (`data.py:124, 216-217`). Plan §8.3 calls for one manifest from the §8.1 draw, frozen before stage 1 | `data.py:123-126`; plan line 520 |

### (b) Blocks stage 2 or later

| Question | Why it matters | Source |
| --- | --- | --- |
| Stage 2 under Q-G(B): what weight μ_star does the relaxed-proxy term get, and is it a separate arm (B2) or part of the stage-2 recipe? | It sets the strength of the own-sample term and the arm design. The plan gives μ only for stage 3 (C1, μ = 1) | [v4-amendments](r2-condition-plan-v4.md#v4-amendments) (Q-G); plan §13 stage 2, lines 749, 767 |
| Trimmed difficulty ν, if the tail rule fires | If the median Δ_tail on 30 s cells exceeds 0.10 star, the human decides on a trimmed ν before stage 2. That would change the labels and readouts. The measurement is not built (0.22 missing) | plan §7.5, [v4-tails](r2-condition-plan-v4.md#v4-tails) |
| Q-I for LN: may an own-sample LN term run before real preference pairs? | Gates the stage-3 LN term. It can move ahead of stage 2 if stage 1 passes teacher-forced but not free-run | [v4-amendments](r2-condition-plan-v4.md#v4-amendments); plan §13 stage-1 outcome rules, line 741; §15 Q-I |
| Star informativeness (§7.4), if within-chart residual correlation exceeds 0.5 | Changes star draw weights. The rule is stated but the lever is not built | plan §7.4 |
| Q-H default: whole-song star at p 0.10 as a residual; requests within ±0.5 star | Range of stage-2 requests and draws. A plan default, not confirmed by the human | plan §15 "Defaults" |
| Stage R: what is ρ (vector, tokens, reference prefix or latent), and is it trained by CE alone or with a free-run objective? | Identity and return to natural. The plan says "Not decided" | plan §13 stage R; [q-r2-after-judgment](r2-average-and-control.md#q-r2-after-judgment) item 5 |
| q7: does a supplied baseline governs over a chart seed? | ρ semantics, stage R. At its written default | [style-formulation-rethink](style-formulation-rethink.md#q7) |
| q3 and q9 for style: strength for style directives; the overlap ban for the same attribute (the agent's reading, unconfirmed) | Stage 4 validator and style-directive semantics | [style-formulation-rethink](style-formulation-rethink.md#q3), [#q9](style-formulation-rethink.md#q9); [d-formulation-answers-2](r2-style-formulation-check.md#d-formulation-answers-2) |
| Q-M: does the stage-3 emphasis and own-sample terms use the governed factor or the whole decision? | Stage-3 loss. Plan default: governed factor | plan §6.5, §15 |
| Q-K: style vocabulary | Stage 4 attribute set. Plan default: the 5 Lens names, 3 levels | plan §15 |
| Q-L: landmark ablation | Memory path. Plan default: after stage 2 | plan §15 |
| Capacity and lr/decay re-tune | Model size. Deferred to after stage 1 | plan §13 "Deferred" |
| Token form against FiLM comparison | The human's 2026-10-03 decision asked for both forms to be built and compared. The plan defers the comparison until a lead-in policy exists | [d-r2-conditions](r2-implementation.md#d-r2-conditions); plan §13 "Deferred" |

### (c) Not blocking

| Question | Why it matters | Source |
| --- | --- | --- |
| q10 detail: may a request be withdrawn or replaced before its start? | Interface. This part is the agent's reading, not confirmed | [d-plan-v4-answers](r2-style-formulation-check.md#d-plan-v4-answers) |
| q2: are there strength levels below the default? | Nothing is built either way | [style-formulation-rethink](style-formulation-rethink.md#q2) |
| q8: may different properties overlap without priority? | Built at the written default (a). Option (b) would reject every overlapping LN and difficulty pair | [style-formulation-rethink](style-formulation-rethink.md#q8) |
| Q-C, Q-D, the natural guard ±0.05, rule L's cost rule, unattainable-request handling | Stage-0 defaults already applied in code | plan §15 "Defaults" |
| Q-J: constructed alternatives | "Not now" | plan §15 |
| Review of the v1 overnight agent decisions and of the loosened DPO integration test | Listed as awaiting the human in the 2026-10-04 handoff. The notes do not record an answer | [r2-handoff](r2-implementation.md#r2-handoff) item 5; [d-r2-v1-spec](r2-implementation.md#d-r2-v1-spec) |

### Agent observations not recorded in the notes (code-level)
- **O1**: per-arm condition manifest. See (a).
- **O2**: stage-1 claim and thresholds not restated after Q-E. See (a).
- **O3**: own-sample sampling consumes the training draw RNG. The relaxed proxy and F3 draw their sampling seeds from `self.rng` (`train_ce.py:473, 496`), which is the generator that draws training windows (`train_ce.py:329, 399`). An arm with `mu_star > 0` therefore sees a different window sequence from one without, against plan §13 stage 2's config-diff rule ("arms differ in the value encoding, and B2's term only"). [inferred]
- **O4**: A0 is not the v1 condition exposure. A0 drops candidates before per-song selection (`conditions.py:258-278`). v1 dropped from the selected track. Per-interval dropout therefore no longer lowers the number of intervals A0 shows. [inferred] The plan does not specify which.
- **O5**: star conditions can switch off silently. `star_conditions: "auto"` turns them off when the v2 label file is absent (`train_ce.py:215-218`). Nothing refuses: the receipt records it (`train_ce.py:568-569`), but A1's λ_star would then be inert.
- **O6**: in-run records lack the recipe. In-run generation records pass `train_config=None` (`evaluate.py:371`), so the operating-point hash in panel records omits the recipe component (`operating_point.py:62`). Plan §9.6 item 1 makes the recipe part of that hash.
- **O7**: relaxed-proxy cost on whole-song scopes. The proxy free-runs the whole scope from the real prefix (`train_ce.py:469-475`). For a whole-song difficulty interval (p 0.10) that is a whole-song sample inside one training step. The pilot that would measure this cost has not run. [inferred]

---

## 5. What stands between the current tree and a stage-1 run

I give no go or no-go verdict. These are the items, with evidence.

### 1. Failing stage-0 suite
Job `20261006-104814`: 19 failed, 215 passed, 4 skipped. The three causes are in section 1:
- A: `features.py:392`;
- B: `conditions.py:307`;
- C: `generate.py:108`.

Cause C leaves the free-run locality tests T-L1 and T-L2 unverified, along with T-A, T-C and record completeness. The plan's selection rule makes "code passes the stage-0 suite, the locality tests included" a candidate condition (§9.6 item 1).

### 2. In-run evaluation breaks [inferred]
Cause C makes every full evaluation error out:
- at the onset panel (`evaluate.py:466-467`);
- the error is caught and the whole record replaced (`train_ce.py:714-721`);
- so `select.py` would find no candidates (`select.py:34, 77`).

`evaluate.py`, `select.py`, `baseline.py`, `proxy.py` and the relaxed-proxy and F3 code have no tests.

### 3. Draw crash
`conditions.py:307` raises for windows shorter than the 32-row lead. This is deterministic in the DPO path. In CE at window 256 it is possible only in an EOS edge case [inferred]. A crash inside `next_batch` would resume from the checkpoint with the same draw RNG state. [inferred: it would re-crash at the same window until the restart budget stops the run]

### 4. Missing stage-0 code items
- 0.10: DPO anchor on L_base, and the DEVIATIONS L46 entry.
- 0.15: the DEVIATIONS rewrite.
- 0.21: style-attribute naming.
- 0.22: the tail-share script.
- The b(S) tests: skeleton-only, mirror and span-local.
- The stage-1 exposure-accounting test.
- The baseline-row script for ckpt-0061432779, labelled "v1 conditioning".
- The §7.4 star-informativeness lever. It is needed only if that rule fires.

### 5. Stage-0 mac measurements, none run, with the thresholds the plan states before simulation

| Measurement | Status and threshold |
| --- | --- |
| Relabel at 30/60/120 s × 3 phases plus the whole song | ≈ 36 min on 4 workers. Required for star conditions to be on under `auto` (O5) and for the labels-equal-readouts tests |
| Baseline refit (`baseline.py`) | Residual quantiles. Within-chart correlation: if above 0.5, star informativeness applies (§7.4) |
| `draw_sim.py` for **each** of A0 and A1 | The Trainer refuses to start without its N̄ values and matching `n_bar_key` (`train_ce.py:315-317`). Thresholds (`draw_sim.py:34` = plan §8.3): D1 ≥ 0.70; D2 slope 2-SE interval contains 0; D3 ≥ 2 %; D4 LN ≥ 25 % and star ≥ 25 %; D6 ≥ 50 %; D7 ≥ 20 %; **D8 ≤ 5 % on 16-beat LN scopes, otherwise a split boundary decision is designed and costed before stage 1** (plan §4.4 line 268). After the values are pasted in, the stored-N̄ tests (`test_draw.py:95`) must pass within 5 % |
| Tail share (§7.5) | Script missing. The rule (median Δ_tail on 30 s cells over 0.10 star leads to a trimmed-ν question) gates stage 2, not stage 1 |
| 200-window pilot (`train_ce --pilot 200`) | Throughput, peak RSS and rule-L overhead. These set the per-run cost (plan §9.8 ≈ 8.1 h at 2,500 decisions/s) |
| Baseline rows on ckpt-0061432779 | Guard (i) on the prefix panel and the G3 natural rows, as baseline rows required by plan rule 4 (line 580) |

### 6. Learning-rate schedule from Astra
Not yet delivered. The configs currently say `lr 0.0003` (⚑ not the plan's or the human's value) with no `lr_schedule`.

### 7. Stage-1 pre-statement under the amended arms
Claim, metric, thresholds, stop rule (standing rule 1). The plan's text assumes draw-only arms (O2). Two related points:
- whether the arms share a condition manifest (O1);
- only the first training-seed pair (171/471) has configs. The plan runs the second pair (172/472) after the first read-out.

### 8. The human's code check
Required before any training run (v4-amendments, "Process").

### 9. Repository state
- Everything above is uncommitted working tree.
- `src/ensomi_model/evaluation/operators/` and `tests/evaluation/test_operators_*.py` are untracked but dated 2026-10-03 08:38-09:04, so they predate this change set. Whether they belong to R2 v2 is not determined; their tests passed in this job.
- `.gitignore` gains `.claude/` entries unrelated to R2.

### What I could not determine
- Whether the implementation agent considered itself finished: no report exists.
- The skip reasons for the 4 skipped tests: these were inferred from their position and skip conditions, not from the log.
- How often the window-256 StopIteration edge case occurs.
- Whether the kNN weighting in `baseline.py` matches A's.
- B's original M1H horizon.
