# Agent Note: Release decision ownership and hierarchical rhythm repair

Note ID: 2026-09-28-rh-r1-fragmentation-repair
Status: proposed
Kind: investigation
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 2c67bb6b70df425d7bca68a455488833a6720372
Scope: A currently failing fragmentation regression, joint release decisions and difficulty-conditioned primary rhythmic units
Related: 2026-09-28-clean-joint-proposal-learning, 2026-09-28-ordinary-fourstar-rhythm-and-holds, 2026-09-28-ln-risk-calibration

## Direction and authority

The user explicitly requests a red evaluation for generated fragmented LNs,
then targeted architectural changes and fine-tuning. Difficulty should act on
the skeleton's primary subdivision conditional on local musical speed; Tech
can retain a coarser main layer with finer decoration. A mandatory accurate
BPM/phase detector is not required. Both training and inference must remain
audio-to-chart with full audio available, without oracle rhythm inputs that
are missing at deployment. LN release should be reconsidered at the decision
that still has the ability to continue holding, not only after R commits a time.

Prior goal turn made progress: natural corpus study, audio/source figures and
reusable likelihood diagnostics were completed and committed. The current
three-arm baseline remains live and unchanged; its main executable tree is
pinned to ef42095a6e764b0374edbaa36b8ddf87c32364c9. New code uses the separate
release-calibration worktree. No new subagent is authorized by this continuation;
the explicitly requested corpus subagent already completed its assignment.

## New observations before choosing the repair

The fixed stage2048 Lens comparison is complete: all24pages across9contexts.
Review SHA ac2fa42fbb8edf3c8cdcd1c304d172409fa4bbe1f8efbe18ef58f6ffe72325a1,
plan SHA fd33807a7f5b8769ecde212420bcbcb7b9ab5a00117fce62cf14bcf9b9ce32fb.
Some inherited Blizzard handoffs/shared closures improve locally; early
Blizzard remains mostly TAP; fresh has longer local held roles but fails
whole LN requests. Stage2048 inherited Stream locally returns to TAP motion,
yet whole-song LN fraction is .90668 and difficulty6.275. No model qualifies.

Exact physical replay of nine generated stage2048 charts separates current
event and mark ownership. In inherited Stream,2670/4382LNs are<=80ms (60.93%);
2524of those end at H (94.53% of the short set). Every one could have continued
with the same current head signature under unchanged support. Only146 end
at pureR. In inherited STYX,71/84short tails instead end at pureR;26 have just
one entering hold and no row-level keep option. None of these short releases
lands on a nonterminal forced deadline. This rejects R-only repair as a
sufficient answer to Stream, while confirming real R/R1 decision asymmetry.

Ownership script/plan are in the clean-joint owner, with plan SHA
ca1f08fea740f9901bf435cd1240ff999509639cb4c1e65bc535f98ef2ad0b32.
Its completed result is analysis-2048/release-ownership.json; session57337
is terminal exit0. This is proximate support attribution, not a unique
historical training-cause decomposition. H density, birth choices and release
policies remain coupled even when the final release is voluntary.

## Experiment Card: fragmented-ln-regression-v1

Revision: 1
Accepted revision: none
Execution authority: explicit user instruction to implement the red eval,
change architecture and fine-tune, plus the continuing system goal.

First establish an offline exposure coordinate: heads whose actual LN duration
is<=80ms divided by all heads in a fully resolved scope. Also retain its LN-only
denominator and counts. The80ms coordinate was already used in the independent
ordinary-fourstar study; it is not an individual minimum-duration rule.
Compare prevalence with equal-song then equal-chart ranked3.5–4.5star
references at quantile.99. Known LN requests select the corresponding source
amount stratum; unknown LN requests use the natural mixture, never a stratum
selected by the generated output. Preserve no-opportunity semantics and actual
tails beyond a scope. This is not a causal frontier value or a full quality score.

Baseline: fixed stage2048 checkpoint/export identities already recorded by the
clean-joint study. Compare its inherited/early/fresh Stream, STYX and Blizzard
cases. Source sanity controls include the ordinary corpus study's nine charts
and the earlier short-LN references that lie inside the declared difficulty
band, using their actual LN amounts. Exact sources, code OID, plan and output
identities will be frozen before the criterion run. Preserve a failed positive
as a measurement defect to investigate; do not silently relax the criterion.

The first intended red witness is concentrated short-LN generation under D4
Stream. Baseline raw counts are known, so this is a deliberately constructed
regression, not a blind discovery or held-out generalization result. A reduced
fragmentation coordinate alone cannot pass a replacement model: retain explicit
LN amount, excessive duration/coverage, difficulty, styles, source organization
and native runtime guards. Do not fix the red result by suppressing LNs,
uniformly lengthening tails or inserting a decoder duration mask.

New owner: artifacts/joint-audio/20260928-ln-fragmentation-repair-v1.
Reference construction and baseline measurement are CPU-only, no new model
samples,60seconds per statistical pass,1GiB process memory and64MiB outputs.
Fresh files only; stop on source/hash drift, mismatched scope or failed source
comparison. The library observer has five focused checks for group weighting,
true tail dependencies, request-conditioned cohorts, missing denominators and
individual short-LN admissibility. The implementation does not change any sampler.

The subsequent architecture Card revision must specify the actual joint
release law, H rhythmic representation, learning data and parameter/update
budget before training. The user's authorization covers execution; neither
this Note nor any passing diagnostic is human acceptance or model promotion.
