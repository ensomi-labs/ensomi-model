# R2 bake-off round 1: the night of 2026-10-07

Question material for the first system comparison on the collapse problem ([r2-collapse-20261007](r2-collapse-20261007.md)). The plan it executes: [r2-bakeoff-plan-20261007](r2-bakeoff-plan-20261007.md). Main session 28c38740.

<a id="o-x0-first-read"></a>**Observation (agent read of the sheet against the key, unscored), 2026-10-07 about 18:06 UTC.**
- The human marked 6 of 8 generated songs as collapsed and 1 of 4 real source foils: x0-08, a band-5 chart of 451 s, at high confidence ("repetitive patterns, poor hand balance, lack of stream style feeling").
- Not marked: generated x0-05 (band 3) and x0-10 (band 4).
- The human's descriptions name LN distribution and organisation, jack organisation (same-lane, mechanical repetitive jacks), repetition and lack of variance, hand balance, and difficulty diverging between parts.
- Most first marks fall at about 20-85 s, which the agent estimated (K / song length, not the row maps) at rows 170-500, inside the encoder's 511-row history.
- Inferred: what the human calls collapse is mostly local organisation (F3) and absorption, appearing early, rather than identity lost past 511 rows (F2). Round 1 as planned targets F1 and F2 only. To be checked by the scoring ([r-x0-scoring](#r-x0-scoring)).
- The human added: the marked ranges are not precise, and only parts of each chart were checked ([private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#answer-1)). Unmarked windows are therefore not reliable negatives.

<a id="r-x0-scoring"></a>**Delegated 2026-10-07 18:05 UTC: X0 scoring and the X1 analysis** (fresh general-purpose subagent; brief `~/ensomi/.sync/cp/scratch/r2-collapse/x0-score/brief.md`; outputs `artifacts/r2-collapse-20261007/x0-score/` on bings-mac).
- Window-level AUC per candidate measure, 64-row windows, primary label "half the window inside a mark", with sensitivities: any overlap; negatives only from unmarked items; windows after the last mark dropped; marks widened by ±3 s and ±6 s (added after the human's caveat). Item-bootstrap 90 % CIs. Differences under about 0.05 AUC are read as noise.
- Also: the false-positive source, the row of each first mark, and the type notes mapped to statistics.
- X1 per the fixed start-dominant or drift-dominant rule ([p-collapse-experiments](r2-collapse-synthesis-20261007.md#p-collapse-experiments)).

<a id="d-bakeoff-night"></a>**Decision (human, 2026-10-07 about 18:08-18:10 UTC, [private, local](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#answer-1), [prompt-2](private/human-inputs/28c38740-5d34-43bb-ad36-ff8ec6d0299f.md#prompt-2)).**
- Build now and fix the arm list after X0 is scored: incremental sampler, evaluation module, long-chart panel, B1-B3 behind flags.
- Decode-time block selection runs tonight as an arm for the local defects. The on-policy fine-tune is not ruled out; it is not built tonight.
- For tonight only, the pre-launch check is handed to a fresh Opus subagent, which reviews and corrects Astra's code; then training starts without waiting for the human. The human is asleep until morning in Japan.
- Agent reading, within the plan: B4 (two-rate encoder) is not built tonight (not in the chosen option). The plan's early-kill rules only freed a slot for B4, so the mid-run mini-panels are dropped; reader weight norms are logged instead. Training length is equal wall-clock, 1.25 h per arm, with each arm's cosine schedule set from its measured throughput.

<a id="r-bakeoff-build"></a>**Delegated 2026-10-07 18:12 UTC: Astra builds round 1** (ens job `20261007-181225-r2-bakeoff-build`, fast tier, `--rw`; brief `~/ensomi/.sync/cp/scratch/r2-bakeoff/brief-astra-build.md`).
- Arms: configs `ce_v2_bo_b1`, `b2` (about 16 cumulative self-anchor channels through a zero-initialised reader, plus history dropout on 25 % of windows truncated to 64-128 rows), and `b3` (B2 plus θ, 10 standardised whole-chart coordinates with a known bit, noised, dropped on 30 % of windows; a donor prior drawn from the 32 nearest fit_train charts in skeleton space). All warm-start from 56M with LN level and length off.
- Enablers: an incremental sampler state with an equality test against the current sampler; `collapse_eval.py` with m1-m7, gap closure G against B0 and the guards; the long-chart panel (25 groups per band, K ≥ 1,000 in band 2 and ≥ 1,500 above, X0 charts excluded; BOS at three seeds capped at 2,560 rows, plus one prefix continuation).
- `d0`: decode-time selection on 56M, best of N = 4 candidates per 64-row block, with a pluggable block score (`phi`: corpus ridge log-density; `envelope`: distance outside the band's source envelope). The score and the G measure subset are fixed after X0 scoring.
- Night runner `artifacts/r2-bakeoff-20261007/night.sh`, written, not run: B1, B2 and B3 train in series; panels run alongside the next training.
