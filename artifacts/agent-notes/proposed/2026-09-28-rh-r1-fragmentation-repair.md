# Agent Note: Release decision ownership and hierarchical rhythm repair

Note ID: 2026-09-28-rh-r1-fragmentation-repair
Status: proposed
Kind: investigation
Created: 2026-09-28
Updated: 2026-09-28
Product revision: 965d6702640a9dc3d6331ddd776daee4e7341d4b
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

Revision: 3
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

## Red baseline and implemented decision law

The observer was committed at e1fecbda2a5d6b041f32cffd34cbc160fd0787a4;
five focused checks passed. The frozen baseline run completed in1.872684s,
session34582 exit2 as intended: inherited and early Stream short-LN burden
is .55245189/.43932891 against natural-mixture99th percentile .20013755.
All16real source sanity controls pass, including ordinary held roles and
specialist short-LN/non-binary examples. Other source amount references are
.08096591(TAP-majority),.24214418(mixed),.37607450(LN-majority).
Plan SHA93401047b3aeaea0c8fe799c42f276b48583b990acea296f89d642c0b76a2429.
Artifacts baseline.json/reference-records.json retain all checks and existing
native guards. A low fragmentation score alone is explicitly insufficient.

Implementation965d670 adds release_policy=r1_joint. R1 scores a virtual wait
and all feasible nonempty release subsets at each queried native millisecond;
R hazard is logsumexp(nonempty scores)-wait score+learned flow log scale.
After an event, its conditional mark law is the actual R1 row law. No virtual
wait enters physical or neural event history. The release scale starts at
log(.001), expressing a native-ms flow coordinate rather than a hold floor.
The separate R MLP is unused in this mode. H decisions retain their boundary.

Joint mode uses raw survival with an explicit last-clock deadline atom, not
finite-wait conditional normalization. This preserves the effect of a global
wait preference and bounds inference queries to publication/chunk time. All
physical recovery support remains60/25/21ms. The current implementation
requires flat rows with no count prior, scoped allocation or player adapter;
these are not enabled by the parent under comparison. Candidate scoring is
reduced from256to16release/wait rows with matching values and gradients.

Nine joint-policy checks include one-held-finger wait choice, actual event/mark
factorization, row-history dependence without H-content dependence, shared
short-hold preference in event odds, native/replay agreement across control
boundaries and publication partitions, raw deadline atoms, strict checkpoints
and CPU/MPS loss/gradient parity. The final relevant command passes26tests;
an earlier broader affected-owner command passes36tests. No quality claim
follows. The self-contained architecture is docs/research/joint_r1_release_decisions.md.

## Revision-two resource and learning pilot

This revision implements the first release-decision branch, not the required
hierarchical H subdivision redesign. The latter remains necessary: local
musical speed/difficulty should control the main time unit, with separate
finer decoration, learned end-to-end without oracle BPM/phase. Do not describe
the new release law as having solved H or the majority-H short-tail mechanism.

Initialize from inherited step2048, checkpointSHA
8f3eda8c5e206230838f172c9ee8d32015572d1740b4fa7a19860408357195eb.
Load every original tensor exactly, add joint release mode and32-wide existing
hold-audio cues with zero output projections. New cue tensors and the scalar
flow scale are the only missing parent parameters. The initial H/row law is
preserved; R event law changes. Fresh AdamW,lr1e-4,weight decay1e-4,clip1.

Use32already frozen training examples at source-plan indices4096:4128 from
the clean-joint ledger(SHAa48c9cc55f60fd6353295513f060ac23fd7f5487cf798907eb3eca8b6d716862).
This parent had consumed0:4096, but the separately continuing baseline may
subsequently consume them; this is not a new held-out evaluation set.
Keep all source prefixes, masks, controls and sample weights. Full-song fine
and coarse audio is recomputed with current weights during every update.
No beat/redline labels or future LN ends enter model queries.

First calibrate only the new scalar flow coordinate by maximum likelihood on
raw R logits from the first8ofthese training examples, with all other weights
fixed and no optimizer state. Report the original/new scalar and event count.
This is initialization exposure, not validation or gameplay improvement.
Then execute16batch-twoupdates jointly over audio/H/R1 using H timing NLL,
joint R timing NLL and row loss L_U+2L_release|U. The extra conditional release
weight targets the measured loss-allocation conflict; it is a simultaneous
recipe change, so no isolated architectural causal percentage may be claimed.

The pilot uses CPU2threads while the old MPS fit continues. Bounds:1200s,
12GiB sampled process footprint,2GiB output, fresh pilot-v1 directory, no
automatic overwrite/restart. Every step records scalar losses, gradient roots,
elapsed time and footprint. Require finite updates, active audio/context/H/
R1/hold-cue/flow gradients across the pilot and exact checkpoint reload.
Unused legacy R/skeleton parameters are not required gradient roots.
Stop on source/checkpoint/data drift, nonfinite values, owner STOP or bounds.
Preserve failures. Pin scripts, source OID, source slice and commands before run.

After the learning/resource pilot, measure real native startup/service and
fixed behavioral guards before selecting any larger run. No second MPS fit
starts while the old supervisor remains active. Execution is exploratory
under explicit user authority; Card acceptance remains none.

## Completed learning pilot and native probe

Pilot session92030 completed normally:16updates,32factual draws,91.337938s,
peak Darwin footprint3,471,429,448bytes. All required shared audio/H/R1/cue/
flow gradient roots were active and strict checkpoint reload was exact.
The calibrated flow scale was-8.32499437 from36true R events at23,802valid
non-forced native clocks in the first8draws; after learning it was-8.32426548.
Different batches' losses are not a monotone-improvement measure.

Checkpoint pilot-v1/step-16.pt SHA
1ad052688ab39398614cc8c3b1e1946bacce88520d60ce6f71283832eb575a3b;
pilot plan7ec1cae92eb0027cc89b836c8cc89fc0dc665ee71ea438b42e32d3f6f6b7a241;
source slice3376ede1d7e1240626a0c4b5464eb0c9c6bbcf214ae9e4876dd7fa776bb7ded8;
flow receiptf5c0f28ed01dff50ef691ff7eeda5d008eed68cb489c5894aef40de870ed7d37.
No native-quality conclusion follows from this resource/learning pilot.

Next frozen native probe uses the same three complete-song D4 witnesses:
STYX, Blizzard and Stream Zenithfall. Controls/seeds/assets match the prior
interim plan, with no gold H or LN endpoints. Startup requires30actual rows
and8s settled coverage, including whole-audio model encoding from cached Mel;
2s service windows retain the existing2s startup/service limits. Source965d670,
CPU2threads,360s per case,1200s whole probe,6GiB footprint. Partial failures
preserve open holds and do not undergo complete-chart fragmentation scoring.
The same frozen corpus reference and existing difficulty/amount/pressure
checks remain visible. No suppression or hard duration floor is added.
Native plan SHA7264cf453a8cbb9f86528d016d7197f94b62dc09aff9f42cca429531a439aa91.

The separate old three-arm baseline reached4096updates and entered final
validation/native evaluation. Its supervisor29252 remains authoritative;
do not modify its main executable tree or start a second MPS fit yet.

## Native pilot result and revision-three continuation

All three native charts completed, with immutable exports/reparse and eight
fixed Lens pages read. The wrapper then failed in postprocessing because it
indexed an absent ln_fraction on the intentionally unspecified Stream request.
No generation was repeated: finish_native_probe.py used .get(...), verified
unchanged outputs and wrote native-probe-result.json. Repair plan SHA
782a4669ded470cbedb6db30c7c151e3d8560fcbed61ebe8f8727f41c75c43d7.
The original failed script and plan remain preserved.

Short-LN/all-head burden: STYX24/806=.02978; Blizzard25/1038=.02408;
Stream248/3601=.06887. All pass this coordinate, but no model qualifies.
Stream remains5.79123stars and LN fraction.51152; Blizzard LN.28709 fails
its.83805request. STYX whole.46898 is close to its.48528request, but the fixed
1.8–7.8s source comparison is entirely TAP. Blizzard40.342–46.342s begins
with recurrent col0TAPs then has only isolated holds/handoffs. Stream18.335–
23.335s has broad overlapping holds, initial all-four shared closures and
later irregular exchanges, rather than the requested prominent Stream.
This is an explicit counterexample to treating short-tail reduction as a win.

CPU2thread cached-Mel startup with30rows and8s coverage: .970/.697/2.812s;
maximum2s-window service1.228/1.052/1.745s. Stream misses the2s startup bound.
STYX also has one14.6ms steady-lookahead deadline miss immediately after8s;
required startup from the complete trace is.985s. Do not silently change the
criterion. Concurrent old baseline CPU evaluation makes these measured traces
exploratory performance evidence, not an isolated machine benchmark.

Revision3 continues exactly the same learning recipe/optimizer from step16
for64additional batch-two updates, using the next immutable128draws at4128:
4256. This expands observed factual exposure rather than changing the model
or retuning the regression to the outputs. Full audio receives gradients.
The64-update segment is CPU2threads with1800s/12GiB/2GiB limits, fresh
continuation-v1 output, no overwrite or automatic restart. It preserves the
same three native witnesses, source comparisons and all other quality/runtime
guards for subsequent evaluation; no milestone is an automatic promotion.
No MPS fit begins while the old supervisor remains active. The follow-up is
exploratory and explicitly user-authorized; Accepted revision remains none.

## Revision-three result

Continuation session49229 completed normally:64additional updates, total80,
321.306296s, peakfootprint4,132,098,296bytes. Strict reload and shared gradient
roots passed. CheckpointSHA b57934728a77abfef0d4bd1ea6387d7cb6f8582c6e450bdd890bfeb1d380ebd6;
plan347be2ef030c9538ef6248fc49fecf04087b1d914d5e98ad286602ea32385635;
source slice b77b818756fdec067f614f59f0bc0f57b8a8a92f9ddc592ff039c417f670af4a.
No run remains in this segment; do not restart it.

Native80 session92085 completed all3cases,202.49s, all overall failed.
Plan288c19603adf75674828e8e63dd552f891a0f562d663cc70af598cffaaa7d389.
STYX/Blizzard/Stream stars3.69055/4.35911/5.82521, rho.6875/.64506/.42449,
B80.03864/.07374/.06561. All B80pass; both specified LN amounts fail, and
Stream difficulty fails. First30rows+8s startup2.247/1.838/2.270s; all traces
have2lookahead deadline misses; max2s service1.270/1.210/1.330s. Keep failures.
No source/native generation was rerun merely to obtain a lower latency.

All8fixed Lens80pages were read; session59082 is terminal. STYX regains
short held roles and repeated col1 entries; Blizzard has more sustained LN
interplay than step16 but some quick returns/unequal endings. Stream remains
short/medium LN interplay with later longer overlaps, not requested prominent
TAP Stream. Whole B80can pass while local LN organization remains poor.
Do not apply whole-chart reference quantiles directly to short windows.

The pilot16 support replay completed after a summary-key renaming defect was
repaired in a separate preserved v2script. Ownership plan
1e6f3a4e31bdb4facd746620b6f156b15e97e83d5142f679c709b3144f2cd19f
is the original; actual v2plan/outputs remain under the same owner.
Pilot Stream medianLN191ms versus parent73ms;130/1842LN tails land at nonterminal
forced deadlines versus0/4382before. The old conditional law lacked the same
explicit atom, so this difference alone is not a count of bad patterns.
All 21/14/130nonterminal forced tails in STYX/Blizzard/Stream include both short
and long holds. Counts are per LN object, not number of release rows.

Decision: REFINE. The shared wait/mark law fixes a real decision asymmetry and
learns, but this recipe is not qualified. Do not scale it automatically or
promote on B80. H's missing shared rhythmic relationships and birth/release
control calibration remain live causes. Hierarchical-H reasoning and its
chart-only diagnostic now have a separate proposed owning Note.

Product observer and self-contained docs committed at
a6c912f5c31cd843103048640febf0d9927d04ea. The neural implementation exercised
by all above checkpoints is965d670; this descendant adds only the H-rhythm
observer/tests and docs/images. The research worktree is clean. Do not rerun
old source-guarded scripts against a newer executable tree without a fresh
declared plan. No remote push, main merge or benchmark update occurred.

Final native80 result SHA5bff5abb42c865135061d66ec7cb8bcafe0ff6207bdfa82630daeb1c51f3adf8;
Lens80review446d5759a05cd02d7b522e43f3e6f644e2817230b1388c4afc6974d840569466.
The earlier supervisor29252 has now also terminated exit0 after all84final
native cases; its own result receipt confirms completion. No training or
evaluation process from this Note remains live. The overall playable-system
goal remains active and unmet.
