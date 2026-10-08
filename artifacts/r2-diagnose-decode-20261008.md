# Diagnosis round after bake-off round 1: decode-time selection (Opus, 2026-10-08)

Report of the fresh Opus subagent "opus-decode", 01:33-02:17 UTC, one of three analysts after round 1 ([o-bakeoff-r1](r2-bakeoff-night-20261007.md#o-bakeoff-r1)). Brief: `~/ensomi/.sync/cp/scratch/r2-diagnose/common.md` plus `brief-opus-decode.md`. The main thread saved its returned text here unchanged in content, and spot-checked the source rows in `charts.md`.

Tags: [M] measured, [I] inferred, [P] proposed.

**Setup.** All generation is 56M, BOS, seed 954, T = 1. Every arm is scored on rows below min(K, 2,560), because the bake-off BOS runs stop at panel `max_rows` = 2,560. Pilot subset: the first 6 panel charts per band, 24 in all.

## Headline

1. **The wanted mass is in the proposals for chart identity and LN level** [M].
   - A source-oracle best-of-4 per 64-row block (tilt ≤ 0.64 nats per block) reproduces each chart's LN level, chord rate and sectional drift.
   - Within-band identity r, oracle vs plain: held 0.94 vs 0.40, c3 0.93 vs 0.58, LN-head share 0.91 vs 0.45.
   - The proposals do **not** hold the source's local organisation: the adjacency-MI excess closes about 17 % of the gap.
   - They are narrower than real charts, so selection makes charts more uniform than the source and pushes envelope exits below the source rate.
2. **Both d0 scores fail for lack of a target, not lack of search** [M/I].
   - `phi` anchors to the chart's own past (slope 0.948 on running held). It locks in the LN-poor BOS start, and its regression fixed point pulls fast charts' held down.
   - `env` is inactive: it picks candidate 0 in 90/96/98/98 % of blocks (bands 2-5).
3. **Two of the stated failures are reading artefacts** [M].
   - "Band-5 stay far below source": `d0-phi` is *above* the source (0.50 vs 0.32). The signed −27 comes from a 0.007 denominator.
   - "`d0-env` lowers band-5 LN": the b0 row in `comparison.md` pools seeds 954-956 with prefix continuations (band-5 prefix held 0.177). Against its matched seed, `d0-env` is 0.091 and plain 0.085.
4. **G for one seed is about as noisy as the arm effects** [M].
   - Plain seed against plain seed gives G = 0.93-1.19.
   - `d0` gives 0.74-0.91 against the three plain baselines.
   - One system across seeds: b1 1.13-1.47, b2 0.99-1.40.
5. **Fixed variant** (flagged blocks only, satisficing, two-sided chart-relative score) [M].
   - It cuts flagged windows 6× (0.062 → 0.010) with LN share unchanged, at a tilt of 0.0073 nats per block.
   - It is a guard, not a fix: identity, drift and adjacency do not move, and it overshoots the source's legitimate exit rate of 0.079.

## Failure 1: `d0-phi` strips LN in bands 4-5 (and the band-5 stay rate)

Facts [M], matched seed 954, 25 charts per band:

| | Plain | `d0-phi` | `d0-env` | Source | Plain seed range (954-956) |
|---|---|---|---|---|---|
| Band 4 held | 0.091 | 0.066 | 0.111 | 0.116 | 0.091-0.137 |
| Band 5 held | 0.085 | 0.047 | 0.091 | 0.176 | 0.083-0.126 |
| Band 5 held by thirds | .047/.099/.107 | .047/.048/.046 | .115/.096/.059 | .18/.17/.18 | |
| Band 5 bus4 / c3 | .038 / .050 | .011 / .017 | .039 / .039 | .118 / .080 | |
| Band 5 stay (transitions) | 0.31 | 0.50 (few) | 0.235 | 0.32 | |

Band 2 goes the other way under `d0-phi`: held 0.108 → 0.131.

| Rank | Hypothesis | Prediction | Evidence | Discriminating pilot | Tuning |
|---|---|---|---|---|---|
| 1 | **Self-anchor lock-in.** The phi target is a ridge prediction from the chart's own running Phi, with slope 0.948 on held. It follows the chart with no pull toward band or source, so it freezes the BOS start, which is LN-poor in bands 4-5 (F1) | Flat thirds at plain's first-third level; drift below the source; charts more uniform than real ones | [M] Thirds flat at 0.047 = plain's first third. Drift \|t3−t1\| 0.051 vs source 0.084 and plain 0.128. Chart-relative score −3.08 vs source −4.67 (more uniform) | `d0-phi` from a real 512-row band-5 prefix: lock-in keeps held near 0.18, a downward bias decays it (not run) | Do not self-anchor. Score distance to an external chart-level target, or drop running Phi from the predictors |
| 2 | **Regression fixed point from between-chart confounds, amplified 19×.** h* = (μ − 0.948h)/0.052 depends on timing terms (fast charts have fewer LNs in fit_train) | Change has the sign of (h* − plain) per band and per chart | [M] `pull.py`: median h* by band 0.287 / 0.167 / 0.068 / −0.039 vs plain 0.110 / 0.121 / 0.092 / 0.086. Direction matches in bands 2, 4, 5 (not 3). Per-chart corr(h* − plain, change) +0.25 to +0.51, +0.41 over 99 charts | Done | Fit phi within-chart, without level-carrying timing predictors, or with band as a predictor |
| 3 | **Asymmetric per-block selection against LN candidates.** A pooled Gaussian (held residual SD 0.08) on a zero-bounded, right-skewed statistic | Negative held selection differential in bands 4-5 on any trajectory | [M] phi argmax − candidate mean, held: −0.022 / −0.013 (bands 4/5, oracle trajectories, 202/265 blocks); −0.015 / −0.019 on its own trajectory; +0.004 in band 2. rho(score, held) −0.09 to −0.21 | Done (`cands.py`) | Two-sided z-distance with per-band within-chart scales, not a pooled density |

Rejected [M]:
- **Candidates differ mostly in LN.** The spread of 4 candidates over the real within-chart block SD is 0.42-0.77 for held, about the same as nh, c3, pent and jack.
- **`bus4` in env.** env decides only 2 % of band 4-5 blocks, and LN is unchanged against the matched seed.
- **Mode-seeking under p_θ.** The chosen block's log p_θ is within about 1 nat of the candidate mean. The mode-seeking is within phi's own Gaussian, i.e. hypothesis 3.
- **Tilt size.** The oracle uses the same ≤ 0.64 nats per block and reaches the source LN. The direction of selection is wrong, not its size.

## Failure 2: lowest G, but no visibly better charts

| Rank | Hypothesis | Evidence | Discriminating test | Tuning |
|---|---|---|---|---|
| 1 | **Mostly a statistic artefact** | [M] Null G (`null_g.py`): plain 955 / 956 vs 954 → 1.02 / 0.93; 954 vs 955 / 956 → 0.975 / 1.19. `d0-phi` / `d0-env` vs plain 954, 955, 956 → 0.91 / 0.87, 0.80 / 0.74, 0.82 / 0.87, so d0 is roughly 1-2 seed SDs better. The d0 rows are matched to 99 runs at one seed and b1-b3 to 396, so the rows are not comparable. Plain band-3 stay rate ranges 0.24-0.61 across seeds | Done | ≥ 3 seeds per arm; a seed-matched baseline (`cand0` = plain with the block seeds); drop the per-term ratio where the denominator is tiny |
| 2 | **G rewards blandness and statistics selected on directly** | [M] `d0-env` selects on the lock/hlock/bus4/jack statistics behind m1, m2 and two guards. `d0-phi` charts are sparser (nh −0.09), have half the chords, are LN-poor in bands 4-5 and more uniform than real. All selection arms push m1 below the source (0.010-0.030 vs 0.079) | Done | Score m1 two-sided against the source rate; add a "not more uniform than real" guard |
| 3 | **G does not see what the human sees** | [M] Adjacency excess MI stays at 0.33-0.36 under every arm, against the source's 0.49. [I] With X0, no G term covers LN placement relative to the chart, hand balance or mechanical jacks | Human paired screen (not run) | Keep G as a guard, not a selection target |

## Oracle: is the wanted mass in the proposals?

Same 24 charts. `cand0` is plain sampling with the block seeds, the paired baseline for the oracle and fix arms.

| Arm | m1 | Stay (n) | Held r within band | c3 r | LN-head r | Drift | Chart-relative | Adjacency | Closeness to source rows |
|---|---|---|---|---|---|---|---|---|---|
| Source | 0.079 | 0.35 (52) | 1 | 1 | 1 | 0.084 | −4.67 | 0.487 | 0 |
| Plain b0 | 0.076 | 0.45 (51) | 0.34 | 0.40 | 0.13 | 0.128 | −5.50 | 0.337 | −12.9 |
| cand0 | 0.062 | 0.39 (41) | 0.40 | 0.58 | 0.45 | 0.121 | −6.26 | 0.339 | −11.8 |
| `d0-phi` | 0.030 | 0.16 (19) | 0.28 | 0.07 | 0.27 | 0.051 | −3.08 | 0.329 | −10.7 |
| `d0-env` | 0.027 | 0.22 (18) | 0.44 | 0.61 | 0.50 | 0.112 | −5.15 | 0.337 | −10.9 |
| **Oracle, N4, B64** | 0.022 | 0.13 (15) | **0.94** | **0.93** | **0.91** | **0.085** | −3.36 | 0.364 | −4.4 |
| Fix | 0.010 | 0.67 (6) | 0.39 | 0.60 | 0.42 | 0.115 | −5.25 | 0.341 | −11.4 |

- **Per-band LN** [M], oracle / source / plain: band 3 0.047 / 0.047 / 0.154; band 4 0.116 / 0.111 / 0.081; band 5 0.127 / 0.140 / 0.057.
- **Headroom** [M]. The best of 4 is as close to the source block as the source's own neighbouring block: −3.5 to −5.6, against −3.9 to −6.7. A random candidate scores −6.2 to −9.9.
- **Proposal spread** [M]: 0.37-0.84 of real within-chart block variability (raw 4-sample SD; about 0.45-1.0 after small-sample correction).
- **Answer** [M/I]:
  - Yes, at block level, for chart identity, LN level and sectional drift.
  - No, for local organisation (F3) and for the source's legitimate extremes.
  - The oracle's charts are smoother than the source.
  - Decoding needs the target; the proposals can realise it.
- **Budget and block size** [M, bands 2 and 5, 3 charts each, n = 6]:

  | Arm | Closeness | Chart-relative (source −5.36) | Held (source 0.187) | Adjacency (source 0.511) |
  |---|---|---|---|---|
  | N4, B64 | −5.59 | −3.67 | 0.170 | 0.333 |
  | N4, B16 (same compute) | −5.33 | −4.59 | 0.180 | 0.389 |
  | N8, B64 | −3.85 | −3.70 | 0.166 | 0.352 |

  [I] Shorter blocks keep more variety for the same compute; more candidates track more tightly.
- **Constant chart-level target** (the "theta" arm: source chart means as a perfect θ, 12 charts) [M]:
  - Within-band identity r: held 0.84, LN-head 0.83.
  - Over-uniform: drift 0.061 against the source's 0.089; chart-relative −3.07 against −4.62.
  - Flagged stretches last long: stay 0.87 over 15 transitions, against the source's 0.27.
  - [I] A constant θ is not enough; the target needs a section plan.

## Fixed variant (flagged-only, satisficing)

- **Rule.** Draw candidate 0. If it does not exit the envelope, keep it. Otherwise draw 3 more, and pick at random among those that are unflagged and whose chart-relative z-score (9 coordinates, pseudo-block at the band median) is at least the real charts' q05.
- **Result** [M]:
  - It intervened in 51 of 758 blocks (6.7 %) and replaced 78 % of them; the mean acceptable fraction was 0.58.
  - Tilt: 0.108 nats per flagged block, 0.0073 nats per block overall (binary-KL estimate).
  - Against cand0: m1 0.062 → 0.010; held 0.147 → 0.144; LN-head share 0.274 → 0.268.
  - Identity, drift and adjacency are unchanged; the stay rate cannot be estimated (6 transitions).
- **Verdict** [I]: it lowers degenerate stretches without moving LN share. It also removes legitimate exits (0.010 against the source's 0.079), and touches nothing else the human named.

## Verdict: can decode-time selection solve the collapse?

- **Partly, and only with a target** [I from M]. Given a correct chart-level, section-aware target, it can carry identity, LN level, difficulty divergence and absorbing stretches. The oracle shows the proposal mass is enough at ≤ 0.64 nats per block.
- **It cannot** [M]:
  - invent that target;
  - fix mechanical local organisation (the adjacency gap, F3);
  - restore real within-chart variety.
  - Those live in the proposal distribution itself, so they need training or capacity.

**What it would take [P].**
- **Target:** a predicted per-chart trajectory, identity plus a section plan, from the skeleton (and audio where allowed). First test: B3's donor-θ prior used as the *selection target*, not as an input. This bypasses the finding that B3 barely reads θ.
- **Score:** two-sided z-distance to the target over nh, held, LN-head share, LN length, c3, jack, rep1, pent and hmax, with per-band within-chart scales.
  - No self-anchoring, no pooled density.
  - Satisficing: a random pick among candidates within the source's neighbour-block distance.
- **Budget:** 4 candidates per 16-row block (d0's compute), or SMC with the same increments.
- **Mac cost:** measured 3.3 ms per row for one candidate, and 5-7.5 ms per row-candidate in this harness (not 1.4 ms). Most of it is `IncrementalState` rebuilt from the full prefix for each candidate.
- **Safeguards:** m1 two-sided against the source rate; not more uniform than the source (chart-relative score, drift); LN share per band; the adjacency gap; the log p_θ shift; ≥ 3 seeds with seed-matched baselines; a human paired screen as the judge.

## Jobs, scripts, outputs

- **Scripts** in `~/ensomi/.sync/cp/scratch/r2-diagnose/opus-decode/`:
  - `select_pilot.py` (all arms; logs every candidate's statistics, action log p and phi/env/oracle/rel/theta scores);
  - `analyze.py`, `cands.py`, `headroom.py`, `pull.py`;
  - `null_g.py` and `saved_summary.py`, run on the control plane from mirrored JSON.
- **Jobs:**
  - `20261008-014059-opdec-oracle` (24 charts, 1,104 s);
  - `20261008-014131-opdec-fix`;
  - `20261008-014712-opdec-chain` (phi re-run bands 4-5 ×3, 431 s; oracle-b16; oracle-n8, 473 s);
  - `20261008-020037-opdec-analyze2`;
  - `20261008-020136-opdec-theta`;
  - `20261008-020432-opdec-final`;
  - `20261008-021518-opdec-cands`.
- **Outputs:** `artifacts/r2-diagnose-20261008/opus-decode/` (mirrored except `runs/*.npz`):
  - `charts.md` / `charts.json`, `pull.md` / `pull.json`;
  - `cands-{oracle,fix,phi,theta,oracle-b16,oracle-n8}.md` / `.json`, `headroom.md`;
  - per-arm directories with `*.blocks.json`, `tables.json` and `runs/`.
- **Harness check** [M]: the phi re-run with the log-p hook is byte-identical to the saved `d0-phi` runs on 6 of 6 charts.

**Failed paths.**
- `20261008-015939-opdec-analyze` crashed on the 2,560-row cap of the saved runs. It was fixed by evaluating below min(K, 2,560); the first identity check read 3/6 for the same reason.
- `cands.py` and `headroom.py` were first run on the control plane, writing into the read-only replica. That breaks the workspace rule. Mutagen removed the files, and they were regenerated on the mac.
- The `null_g.py` output exists only in the worker's scratchpad; its numbers are in this report.

**Not done:** `d0-phi` from a real prefix (the lock-in vs bias test); extra seeds for the oracle, fix and theta arms (single seed throughout; 6-chart band means have a seed SD of about 0.03-0.05 in held); a section-aware target arm; donor θ as the selection target; SMC; any human screen.
