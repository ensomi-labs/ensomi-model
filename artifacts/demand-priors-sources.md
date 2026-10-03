<!-- Provenance: written 2026-10-03 by a fresh docs-researcher subagent of main session 10fd41cc (Claude, control plane), from a brief asking for sourced physiological and calculator priors for the demand machine in operator-properties.md. Copied unchanged from the session scratchpad. The main thread checked section B6 against 2044c2b:src/ensomi_model/osu_core/difficulty.py (constants match) and did not re-check the peer-reviewed sources. Agent reading of what it changes: operator-properties.md#s-demand-priors-sources. -->

# Demand-state priors for an osu!mania 4K evaluator: sources

Compiled 2026-10-03. Read-only research. Tags used:
**[PR]** peer-reviewed, **[CODE]** official source code, **[DOC]** official docs or news, **[COMM]** community (forum, wiki, community repo).
Confidence: **H** (number read in the primary text or code), **M** (from an abstract or a reliable summary, not read in the full text), **L** (secondary or partial).

A caution on tooling: a WebFetch summary of Häger-Ross and Schieber (2000) produced invented index values (0.54, 0.52, ...). The real table values are about 0.98. Every number below was checked against the raw text, an abstract, or code. Values that could not be checked are labelled as such.

---

## A. Human motor physiology

### A1. Maximal single-finger tapping rate

| Population / task | Rate | ITI | Source | Conf |
|---|---|---|---|---|
| 24 expert pianists, fastest single-finger tapping | mean 6.9 ± 0.6 Hz (range 5.5 to 8.1) | ~145 ms (range 123 to 182) | Furuya, Oku, Miyazaki, Kinoshita 2015, *Sci Rep* 5:15750, Table 1 ("Tapping rate: finger (Hz) 8.1 5.5 6.9 0.6"). https://doi.org/10.1038/srep15750 (PMC4621510) [PR] | H |
| Same pianists, maximal repetitive piano keystroke rate | 6.0 to 7.8 Hz | 128 to 167 ms | ibid., Results | H |
| 10 healthy adults, index finger, 3 min maximal rate | peak 5.7 Hz (SEM 0.26) and 6.0 Hz (SEM 0.25) in two sessions | ~167 to 175 ms | Madinabeitia-Mancebo et al. 2020, *Sci Rep* 10:3063, Results "Motor behaviour". https://doi.org/10.1038/s41598-020-60043-0 (PMC7035251) [PR] | H |
| Aged 50 to 70 (n=176), smartphone, 30 s single finger | mean 5.2 Hz | ~192 ms | Heimhofer et al. 2024, *Front Hum Neurosci* 18:1427336, Discussion. https://doi.org/10.3389/fnhum.2024.1427336 (PMC11461208) [PR] | H |

**Order by finger:** index fastest, then middle, little, ring (Aoki, Francis, Kinoshita 2003, *Exp Brain Res* 152:270-280, abstract; https://doi.org/10.1007/s00221-003-1552-z) [PR, M]. Heimhofer 2024 (n=370, index, middle and little only): index > middle > little, with every pair significantly different [PR, H]. Aoki and Fukuoka 2010 (*Med Sci Sports Exerc* 42:449): index and middle ITIs are significantly shorter than ring and little ITIs [PR, H, Results text]. Aoki et al. 2005 (*Motor Control* 9:23, https://doi.org/10.1123/mcj.9.1.23): pianists show much less ring and little slowness than controls, so training narrows the gap [PR, M].

**Burst vs sustained:** the figures above are maxima taken in the first seconds. The decline over 10 to 30 s is covered in A3.

**Gap:** I could not get absolute per-finger Hz values (index, middle, ring) from Aoki 2003, 2005 or 2010. They are in figures behind paywalls. A search snippet gave "ring 221 ± 40 ms (4.5 Hz)", but I could not trace it to a primary source, so it is **unverified and not used**.

### A2. Within-hand alternation (trill) vs between-hand alternation

- **Within-hand trill, index and middle, 7 s:** the combined rate of the two fingers is **about 50% higher** than single-finger tapping (Aoki and Kinoshita 2001, *Ergonomics* 44:1368-1383, abstract: "double-finger mode had ... a 50% faster tapping frequency than the single-finger mode"; https://doi.org/10.1080/00140130110107452) [PR, M]. Single-finger and double-finger temporal data were uncorrelated, which suggests a different motor strategy.
- **Per-finger cadence inside a trill is lower than that finger's single-finger cadence.** The loss depends on the pair: smallest for index-middle, largest for ring-little. Freeing the fingers (no resting contact) raised cadence markedly (Aoki 2003 abstract) [PR, M]. So a two-finger hand reaches about 1.5x, not 2x, the single-finger rate, and each finger's own recovery interval stretches during a trill.
- **Between-hand alternation:** I found **no primary source that gives a maximal finger rate for alternation between the hands, compared with alternation within a hand.** What exists:
  - Coordination dynamics: anti-phase bimanual finger oscillation switches spontaneously to in-phase above a critical frequency (Kelso 1984; Haken, Kelso and Bunz 1985, *Biol Cybern* 51:347) [PR, M]. I did not obtain the critical frequency value. This is about continuous oscillation, not discrete key presses.
  - Drummers with sticks (wrist movement, not fingers): about 10 Hz with one hand (Fujii et al. 2009, *Neurosci Lett*). I saw this only in a search summary, so it is [L] and not finger-relevant.
- **Answer to "is between-hand faster, and by how much": no sourced number.** Treat it as an open prior.

### A3. Fatigue time course and recovery

| Task | Decline | Onset / shape | Recovery | Source | Conf |
|---|---|---|---|---|---|
| Index flexion-extension at max voluntary rate, 20 s (n=10) | to **73% of baseline** at 20 s | decline **starts at 7 to 9 s** | not measured. MVC force and ballistic speed were unchanged after the task, so no peripheral fatigue | Rodrigues, Mastaglia, Thickbroom 2009, *Exp Brain Res* 196:557-563, abstract. https://doi.org/10.1007/s00221-009-1886-2 [PR] | H (abstract) |
| Alternating index and middle taps (within-hand trill), 30 s max | **21.6 ± 6.4 %** slowing over 30 s (foot 13.9%, eyes 18.3%) | gradual over 30 s | speed **recovers within the first 20 s of a break**, roughly **linearly** over 25 to 30 s. Breaks of 5 to 30 s were tested (N=17). Recovery slope correlates with the amount of slowing. Virtually no accumulation across trials with 30 to 50 s breaks | Bächinger et al. 2019, *eLife* 8:e46750, Results (Exp. 1 to 5). https://doi.org/10.7554/eLife.46750 (PMC6746551) [PR] | H |
| Single finger, 30 s max, smartphone (n=370) | **17.0 ± 6.9 %** (young), 16.5 ± 7.5 % (aged); depends on finger | 6 × 5 s bins, steady decline | 30 s rest between trials | Heimhofer et al. 2024 [PR] | H |
| Index, 3 min max rate (n=10) | **≈40 %** total | **rapid in minute 1, shallower in minute 2, plateau in minute 3** | the paper notes the CNS "recovers very quickly" after the task | Madinabeitia-Mancebo et al. 2020 [PR] | H |

Mechanism: the authors attribute the early decline to central causes (breakdown of reciprocal flexor-extensor timing into co-contraction; loss of surround inhibition), not to failure of muscle force (Rodrigues 2009; Bächinger 2019) [PR, H].

**Time constants:** no paper reports a fitted exponential time constant for tapping-rate decline or recovery. From the sources: onset at about 8 s, about 20% loss by 30 s, about 40% at a plateau after about 2 to 3 min, recovery roughly linear and complete in about 20 to 30 s.

### A4. Functional forms near the limit; critical power for small muscles

- **Critical power family** (survey: Sreedhara et al. 2019, *Sports Med Open* 5:54, PMC6934642 "A survey of mathematical models of human performance using power and energy") [PR, H]:
  - Two-parameter hyperbola (Monod and Scherrer 1965, *Ergonomics* 8:329): `t_lim = W' / (P - CP)`.
  - Three-parameter (Morton): `t = W'/(P - CP) + k`, which gives a finite P_max at t=0.
  - Exponential (Hopkins, Morton): `P(t) = CP + (P_max - CP)·e^(-t/τ)`.
  - W' balance (Skiba): `W'bal = W' - ∫ W'exp·e^(-(t-u)/τ_W') du`, with `τ_W' = 546·e^(-0.01·D_CP) + 316` s, where D_CP = CP minus the mean sub-CP power. This was fitted on cycling.
- **Small-muscle applications:** Monod and Scherrer's original paper was titled "work capacity of a synergic muscular group" [PR, M]. A forearm critical force test (rhythmic handgrip with maximal contractions, 1 s on and 2 s off for 600 s) gave a plateau (fCF) and a W' that predicted time to exhaustion above fCF well (Kellawan and Tschakovsky 2014, *PLoS One* 9:e93481, PMC3974771) [PR, H]. There is also finger-flexor critical force in rock climbers (Fryer et al. 2019, title only) [PR, L].
- **Applied to tapping rate: no source found.** Every small-muscle CP study I found uses force or impulse, not the rate of an unloaded movement. The tapping literature shows the rate limit is central (A3). Mapping W' onto rate (r_crit, a finite reserve above it) is therefore an analogy, not an established result.
- **Exponential, power-law or logistic cost as the gap approaches the limit:** no physiology source. The only "steep near the limit" forms I found are in community calculators (B6 to B8), where they were set by hand.

### A5. Finger independence; chord timing

- **Enslaving (isometric force):** fingers not instructed to press produced forces up to **67.5 % of their own single-finger maximum**. The effect is larger for neighbouring fingers and is nonadditive (Zatsiorsky, Li, Latash 2000, *Exp Brain Res* 131:187-195, abstract) [PR, M].
- **Individuation of movement** (Häger-Ross and Schieber 2000, *J Neurosci* 20:8542, Tables 1 and 2, self-paced ~2 Hz, right hand) [PR, H, read from the PDF text]:
  - Individuation index: thumb 0.983, index 0.982, middle 0.937, **ring 0.907**, little 0.943.
  - Stationarity index (how still a finger stays while others move): thumb 0.994, index 0.967, middle 0.941, **ring 0.908**, little 0.943.
  - Externally paced movement at 3 Hz was less individuated than self-paced movement at ~2 Hz.
- **Holding a key while a neighbour taps:** in Aoki 2003 the non-tapping fingers rested on keys during tapping. Their key-contact force changed in parallel with the tapping finger, with the **largest range when the ring finger tapped**. The authors attribute the ring finger's low cadence to neural independence more than to mechanical coupling. Freeing the fingers raised cadence "markedly" [PR, M]. No source gives a cost for holding versus resting.
- **Chord simultaneity:** among 22 skilled pianists, the ~30 ms melody lead seen at hammer-string level falls to **about 0 at finger-key level within the right hand**. The left hand tends to lead the right (Goebl 2001, *JASA* 110:563, Results) [PR, H]. I found no SD of within-hand or between-hand chord asynchrony for non-musicians or game players.
- **Temporal pointing error model** (Lee and Oulasvirta 2016, CHI): it predicts the mean and spread of response timing from the time to the target and the window width, combining internal timekeeping, motor execution and input-latency noise [PR, M, abstract-level].

---

## B. Community difficulty calculators

### B6. osu!mania star rating (ppy/osu, deployed)

Source: `ppy/osu` master at **c834803ea9988159d67e1dc5d99cfaccada685d9** (2026-10-02), `osu.Game.Rulesets.Mania/Difficulty/`. `ManiaDifficultyCalculator.Version => 20241007` [CODE, H].

- **Skill** (`Skills/Strain.cs`): `individual_decay_base = 0.125`, `overall_decay_base = 0.30`, applied as `value · base^(Δt/1000)`.
  - This gives an individual (per-column) τ = 1/ln 8 = **0.481 s** (half-life 333 ms) and an overall τ = 1/ln(10/3) = **0.831 s** (half-life 576 ms). These are my arithmetic.
  - The strain value is `highestIndividualStrain + overallStrain`. The chord max is taken when `DeltaTime <= 1` ms.
  - Section 400 ms, `DecayWeight 0.9` (`osu.Game/Rulesets/Difficulty/Skills/StrainSkill.cs`). `difficulty_multiplier = 0.018`.
- **Per note:** `IndividualStrainEvaluator` adds **2.0** for every note (tap or LN head), multiplied by **1.25** if the note starts and ends inside another column's held LN. Δt in the column is head to head (`ColumnStrainTime = StartTime - PrevInColumn.StartTime`). Tails are not separate objects.
- `OverallStrainEvaluator` adds `(1 + holdAddition)·holdFactor`. holdFactor is 1.25 under a hold. If an LN overlaps a previous one, `holdAddition = 1/(1+exp(0.27·(30 - Δend_ms)))` (`release_threshold = 30`, `DiffUtils.Logistic`). In words, a release that falls close to another LN's release earns almost nothing; one about 45 ms or more apart earns about +1.
- **Not modelled:** hands, finger coupling, the tail-to-next-head gap. Cost per note is constant. Rate enters only through exponential decay, so steady-state per-column strain for period Δ is `2/(1 - 0.125^(Δ/1000))`, which goes as ~1/Δ for small Δ (my derivation). There is no finite-gap divergence.
- **Deployment history** (osu-wiki `news/`) [DOC, H]:
  - 2024-10-28: LN overlap bonus needs larger overlaps; chords of simultaneous LN starts no longer buffed ("30 ms corresponds to 250 BPM 1/8").
  - 2025-10-29: mania refactor only (PR #33411).
  - 2025-03-06 and 2026-07-03: no mania section.
- **Sunny rework (SR Rebirth)** is **not deployed**. ppy/osu PR #36342 "Implement Sunnyxxy's Mania Rework" was closed unmerged on 2026-02-22 ("Closing pending ongoing refactor work (Sunny rework is NOT CANCELLED)"). Infrastructure PRs #36556 (tail processing) and #36557 (ManiaStrainSkill) were closed unmerged in 2026-06 and 2026-09 [CODE, H]. Algorithm details are in B8.

### B7. Etterna MinaCalc

Source: `etternagame/etterna` develop at **df42fdf132855bf682cbb3002d90f5d28aba7f16** (2026-10-01), `src/Etterna/MinaCalc/`, `mina_calc_version = 527` [CODE, H].

- **Hands are modelled separately.** Intervals are **0.5 s** (`UlbuAcolytes.h: interval_span = 0.5F`). Every per-interval difficulty, pattern modifier, point count and stamina pass runs per hand (`both_hands`). For 4K the left hand is columns 0 and 1, the right hand 2 and 3 (`oversimplified_jacks::init`).
- **Skillsets** (`Models/NoteData/NoteDataStructures.h`): Overall, Stream, Jumpstream, Handstream, Stamina, JackSpeed, Chordjack, Technical. There are many pattern mods (`Agnostic/HA_PatternMods`, `Dependent/HD_PatternMods`: Roll, OHJ, OHT, WideRangeJumptrill, Balance, Chaos, ...).
- **Speed to difficulty is linear in rate.** `ms_to_scaled_nps(ms) = (1000/ms)·finalscaler`, with `finalscaler = 3.632·1.06` (`SequencingHelpers.h`). The per-finger ms estimate in `nps::actual_cancer` (`SequencedBaseDiffCalc.h`) averages the 3 to 5 shortest per-finger intervals (missing intervals are filled with 360 ms dummies).
- **Jack speed:** the per-column sequence ms estimate is `(total_ms + 30 + 1.5·avg_ms)/(len-1)`, floored at **95 ms** (and at 180 ms ×1.1 for length-2 jacks). The interval jack difficulty is `ms_to_scaled_nps(ms)·1.01`. **Cost does not diverge near a limit: it is hyperbolic in ms and clamped at 95 ms (about 10.5 Hz per column).**
- **Failure model (difficulty to score):**
  - Per interval and hand, when difficulty d exceeds skill x, points lost = `pts·(1 - (x/d)^p)`, with p = 1.7 by default, 1.8 for Chordjack and 2.0 for Technical (`CalcInternal`).
  - For jacks, points lost per row = `max(0, 12·erf(0.04·(d - x)))` (`jack_pointloser_func`).
  - The rating is the skill x at which retained points reach `default_score_goal = 0.93` (`MinaCalcHelpers.h`; cap 0.965). Points per interval = 2 × notes per hand.
- **Stamina** (`StamAdjust`), per 0.5 s interval:
  - `mod += ((avg_d/(0.69424·x)) - 1)/243`. The floor starts at 0.95 and rises by `(mod-0.95)/500` while mod exceeds 0.95. mod is clamped to `[floor, 1.075234·floor]` and capped at 1.09. Difficulty is multiplied by mod.
  - So load accumulates linearly whenever local difficulty is above 69.4% of player skill and drains linearly below it. The floor ratchets upward only, an irreversible component.
  - There is no exponential time constant. For example, at d = x, mod climbs about 0.0018 per interval, so a full 0.125 swing takes about 35 s (my arithmetic).
  - Jack stamina uses separate constants: prop 0.49424, mag 23, ceil 1.05234, fscale 750, cap 1.01. A source comment says the threshold "has shown to be around 0.8", but the code uses 0.694.

### B8. Quaver and SR Rebirth (calculators that differ materially)

**Quaver** (`Quaver/Quaver.API` at **a921d561b2ece7f6bf3682446696c06c17b81649**, 2026-07-27, `Maps/Processors/Difficulty/Rulesets/Keys/`, `Version "0.0.5"`) [CODE, H]:
- Explicit **hand and finger states**. Each object is classed against the next object on the **same hand** as Roll (trill), SimpleJack, TechnicalJack or Bracket.
- Coefficient: `1 + (max-1)·r^exp` with `r = max(0, 1 - (d - xMin)/(xMax - xMin))`, where d = same-hand action duration in ms:

  | Action | xMin to xMax (ms) | max | exp |
  |---|---|---|---|
  | SJack | 40 to 320 | 68 | 1.17 |
  | TJack | 40 to 330 | 70 | 1.14 |
  | Roll | 30 to 230 | 55 | 1.13 |
  | Bracket | 30 to 230 | 56 | 1.13 |

  The code comments it "todo: temp. Linear for now". Nearly linear in gap, with no divergence.
- LN handling: `LnBaseMultiplier 0.6`, layer threshold 93.7 ± 60 ms, `LnEndThresholdMs 42`, release-before 1.3, release-after 1.0, tap-inside-LN 1.05. Chord tolerance 8 ms.
- **No time-accumulated strain or stamina:** per-object strains are averaged over 1 s bins, then a continuity adjustment (0.90 to 1.05) and a short-map nerf below 60 s are applied.

**SR Rebirth (sunnyxxy)**: `sunnyxxy/Star-Rating-Rebirth` at **b6c1d8915fc06060804389a1120424d493a4f06e** (2025-04-15), `algorithm.py`, plus `Star_Rating_Rebirth.pdf` (Feb 2025) [COMM, H]:
- Times are in seconds (δ = 0.001·Δms). Hit leniency `x = 0.3·sqrt((64.5 - ceil(3·OD))/500)`, then `min(x, 0.6(x-0.09)+0.09)`. At OD 8 this gives about 0.085 s (my arithmetic).
- **Same column J:** `1/(δ·(δ + 0.11·x^(1/4)))·(1 - 7e-5·(0.15+|δ-0.08|)^-4)`. It goes as δ^-2 for small δ and diverges only at δ→0.
- **Cross-column X** (adjacent column pairs): `0.16·max(x,δ)^-2`, capped at δ = x. There is also a "fast_cross" term `max(0, 0.4·max(δ, 0.06, 0.75x)^-2 - 80)`, which switches on below **δ ≈ 70.7 ms** and saturates at 60 ms. This is the only explicit threshold-type cost I found.
- The 4K cross weights are all 0.125 (`cross_matrix[4]`), so there is **no hand-boundary distinction in 4K**. 5K and above weight the middle at 0.05.
- **Releases are actions:** `R = 0.08·δ_r^-0.5/x·(1+0.8(I_i+I_{i+1}))`, where δ_r is the tail-to-tail gap and I is a logistic of |hold length - 80 ms| and |tail to next head - 80 ms|. LN bodies weight pressing intensity (`1 + 6·∫LN`, with 1.3x in the first 60 to 120 ms of a body).
- Smoothing ±500 ms (unevenness ±250 ms). There is no fatigue state. The final rating mixes the 93rd and 83rd weighted percentiles with a 5-norm.

### B9. Calibration against player data

- **SR Rebirth:** the author first fitted parameters to a "ground-truth model of ranked beatmaps (Evening and Keytoix's websites)", then **"decided to abandon the systematic approach and set the values by hand ... with strong reliance on my prior beliefs"**. Results were checked against huismetbenen ranking lists (PDF §3.2) [COMM, H]. The final parameters are (w0, w1, p1, w2, p0) = (0.4, 2.7, 1.5, 0.27, 1.0) and norm 5.
- **osu! (ppy):** changes are reviewed by the PP committee using community spreadsheets ("Smoogisheet", cited in PR #36556) and huismetbenen. I found no published fit to scores or replays [DOC/CODE, M].
- **Etterna:** MSD is defined operationally as the skill needed for 93% Wife (code), and its point-loss law is a psychometric-style form. I found no published calibration study, and the constants are hand-tuned (the code comments say so, for example "tuned for jacks, dunno what this functionally means, yet") [CODE, H for the definition, L for the absence].
- **Academic:** Franks et al. 2023 (arXiv:2301.09485) fit ordinal regression to StepMania labels made by chart authors, not player performance [PR-preprint, M]. Lee and Oulasvirta 2016 is a timing-error model validated on game scores (a related cue-integration model reached R² = 0.86 on a different game) [PR, M]. **I found no study relating osu!mania or Etterna pattern parameters to player accuracy from replays.**

---

## Prior → best sourced value or form → strength

| Prior | Best sourced value / form | Strength |
|---|---|---|
| Single-finger max rate | ~5.7 to 6.9 Hz typical (ITI ~145 to 175 ms); pianists up to 8.1 Hz (123 ms); order index > middle > little > ring | Strong (PR, several studies), but per-finger absolute values are missing |
| Within-hand trill | combined ≈ 1.5 × single-finger rate; each finger slower than when tapping alone; index-middle pair best, ring-little worst | Moderate (PR abstracts) |
| Between-hand vs within-hand alternation | **no number**; anti-phase coordination destabilises at high frequency (HKB) | Weak / gap |
| Sustained decline | onset 7 to 9 s; −17 to −27 % by 20 to 30 s; −40 % plateau by ~3 min; central, not muscular | Strong (PR, 4 studies) |
| Recovery | ≈ linear, mostly done in ~20 s, complete in 25 to 30 s; no accumulation with ≥30 s breaks | Moderate (one PR study, N=17) |
| "Exponential" rise near a limit | no physiology source. CP hyperbola `t = W'/(P−CP)` is established for force or power, not tapping rate. Calculators use hyperbolic 1/δ (Etterna, osu!), δ⁻² (SR Rebirth), a near-linear clamp (Quaver), a 95 ms floor (Etterna) or a ~71 ms threshold (SR Rebirth cross-column) | Weak for physiology; the forms in code are hand-tuned |
| Strain accumulated over time | osu!: exponential decay τ 0.48 s (column) and 0.83 s (overall); Etterna: linear accumulator above 69% of skill on 0.5 s intervals with a ratcheting floor; Skiba W'bal τ ≈ 316 to 862 s (cycling only) | Code-exact but not physiological; physiology suggests a fast-recovering central state (A3) |
| Coordination within a hand | enslaving up to 67.5 %; ring finger least independent (II 0.907); neighbouring key force disturbed most when the ring finger taps | Strong (PR) as direction; no cost function |
| Tap ≡ LN head load | osu!: both +2.0; SR Rebirth: heads in J/P, bodies add load | Code convention only, no physiology source |
| Hold occupies finger | osu!: ×1.25 for notes inside a hold; Quaver: LN layering multipliers; SR Rebirth: LN-body integral | Code convention only |
| Release is an action | SR Rebirth R term (tail gaps, 80 ms logistic); osu!: logistic release-separation bonus (30 ms midpoint); Quaver release-before 1.3 | Code convention only |
| Chord timing precision | skilled pianists: finger-key asynchrony ≈ 0 within the right hand | Weak (one PR study, musicians, piano) |

---

## Failed searches and open gaps

1. Absolute per-finger maximal rates for index, middle and ring (Aoki 2003, 2005, 2010): the values are in paywalled figures. Springer, PubMed and Medscape pages did not expose them. The "221 ms ring" snippet is unverified.
2. Maximal finger rate for alternation between hands versus within a hand: no primary source found in three searches. The HKB critical frequency was not retrieved.
3. A fitted time constant for decline or recovery of tapping rate: none reported. Bächinger 2019 describes recovery as roughly linear.
4. Critical power / W' applied to tapping or another movement-rate task: none found. Only force-based small-muscle studies (handgrip, climbers).
5. A cost or penalty for pressing with one finger while the neighbouring finger holds a key down (as opposed to resting on it): none found.
6. SD of chord asynchrony in non-musicians or game players: none found. Goebl 2001 covers piano only.
7. Calibration of osu!, Etterna or Quaver difficulty against replays or accuracy: none published. SR Rebirth explicitly abandoned fitting.
8. Blocked fetches: pmc.ncbi.nlm.nih.gov (reCAPTCHA), elifesciences.org, pubmed, the 2024 osu news page. I worked around them with Europe PMC full text, the osu-wiki repo and direct PDF stream extraction.

Sources consulted: about 38 (code repositories counted once per repo and commit).
