# Agent Note: Learn an ordinary four-star expert from fresh initialization

Note ID: 2026-09-28-ordinary-expert-from-scratch
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-29
Product revision: d6eba238ff3872613c6bfe84b75c5e867c881634
Scope: Fresh ordinary-chart recipe, positive corpus, independent failure evaluation and continuation-level fusion
Related: 2026-09-28-continuous-segment-context; 2026-09-28-response-blindspot

## User direction and remaining goal

The user explicitly asks for a new recipe, fresh relevant modules, learning
ordinary approximately4star arrangements with simple LN/rhythmic organization
and strict long-jack control, then trying fusion. This authorizes local data
preparation, implementation, bounded fresh training and evaluation. It does not
approve an exact Card revision, a lifecycle transition or remote publication.
The ultimate goal remains playable2–6star generation, expressive optional styles,
scoped controls and the existing realtime/CompleteRow interface. The ordinary
expert is one component, not a narrower replacement success criterion.

The previous turn made progress: implemented a direct hand-relative context
path, found36/60expert-selected pilot views outside their actual style labels,
ran matched learning to112durable updates and actual64-step generation, and
found weak fixed-prefix LN-control response. The paired run later aborted inside
MPS at update122; it is terminal and preserved. No qualified model exists.

## Main design decisions

Start learned audio, H, R/R1 and control pathways from new initialization;
retain deterministic Mel, exact replay, row syntax, likelihood machinery and
publication interfaces. Hidden layers use normal random initialization; zero
initialization applies only to suitable output/residual projections. Do not
zero every neuron or load actor tensors from a previous checkpoint. Input
normalization comes from the new training split. Full audio is available during
both training and inference; learned audio encodings cannot be reused across
optimizer updates.

Use one interpretable full-audio encoder with local Mel context and coarse
bidirectional audio context, a timing-only H owner, and a complete-row R1 whose
energies own both release survival/marks and H-row choices. R1 owns cardinality,
columns and LN entry/exit; H does not prescribe chord counts. Keep direct audio,
future H, actual hold origins and persistent factual history visible. Independent
player-response evaluation is outside actor NLL. A simple specialist distribution
should be learned before adding expert routing.

Difficulty semantics are fixed: source whole-chart approximately4star labels
remain the reference while learning their quiet and dense passages. The older
recipe mixed whole labels with local strain-derived labels at16/32/64s scopes;
that can teach a different conditional distribution and is a live contributor
to uniformly pressured output, not a proven exclusive cause. Do not relabel
every ordinary quiet crop as a request for a different difficulty. Source
annotations belong only to their actual ranges. Default samples come from the
natural selected population, and visible-condition balancing must not contaminate
missing-condition training.

Ordinary includes clear TAP groups, breath/transition structure, a sustained LN
with other-finger TAPs, and regular LN streams/high coverage where sources support
them. It does not mean no LN, no chords or one fixed pattern template. Offline
selection of a teaching population does not declare other ranked styles bad or
introduce runtime duration bans.

## Evaluation and fusion

Use failure evals in three separate roles: characterize/select the positive
teaching population; retain a disjoint native regression set; later supply
training-side negatives for a separate continuation evaluator and actual
on-policy learning. The old scalar work reference still misses25ms LN and some
sustained repetitions. Its green flag is not a reward oracle. Preserve source
positives and compare short-LN tail coordinates, RH recovery, time-based finger
load/recurrence, rhythm organization, amount, scope controls, music and runtime.
A good mean must not compensate a severe failure. Train-side signals and sealed
regression cases require separate identities.

Teacher forcing supplies one valid source trajectory. It does not label an
arbitrarily changed generated prefix with the old source suffix. The DAgger
analogue motivates visiting policy-induced states, but this project lacks an
expert action oracle there (Ross et al., PMLR15,2011). Actual candidate futures
can instead receive independent response/control judgments, with the correct
joint H/R/R1 likelihood when those paths are optimized. NLL remains a warm-start
objective and diagnostic, not the final quality criterion.

Fusion is over coherent private continuations from the same committed state.
A router selects an expert for a real-time interval; each candidate includes
its compatible H and R/R1 decisions. The frontier evaluates each actual future,
and only a selected prefix publishes. Open LNs, player state and controls persist
across expert changes; private future endpoints are not public facts. The options
analogue is Sutton/Precup/Singh1999. Once this system qualifies, policy distillation
(Rusu et al., arXiv1511.06295) may reduce runtime cost. Keep the ordinary expert's
regressions protected during fusion; output selection is not parameter averaging
and proposal log probabilities are not selected-policy probabilities.

## Experiment Card: ordinary-fresh-bootstrap-v1

Revision: 4
Accepted revision: none

First establish a reproducible positive population from the existing1973ranked
3.5–4.5star charts. Split connected song-group/audio components before selection;
reserve all previous native-case audio. Record exact source clocks, LF strata,
short-LN/RH tails and real-time per-column attack peaks. These are descriptors
and candidate-selection evidence, not adopted physiological thresholds. Follow
with actual Lens/rhythm inspection for selected teaching anchors. Include ordinary
LN organization as well as TAP and quiet passages.

Frozen scouting script/plan live in artifacts/joint-audio/20260928-ordinary-scratch-v1,
plan SHAcdbe5f2d3d9d939f51a6daec238abc6cc8a8389ab567cb0b4be05ee10a12d6d5.
Inputs are the pinned corpus census and full Mel/row manifest. One CPU process,
600s, exclusive scout-v1, no overwrite or automatic restart. Verify source and
cached-row hashes/counts. Fit any selection/reference cutoffs on training groups
only; held-out groups retain their independent meaning. Do not call the candidate
population the completed ordinary subset before those decisions are frozen.

The first actual fresh fit must use a clean source implementing fresh initialization
and a joint H/R/R1 learning score, then a fixed small set of verified ordinary
sources as a capacity/backend probe. Freeze its data, architecture, initialization,
controls, optimizer, native seeds and resources in a revised Card before execution.
Start with a short CPU/MPS stability comparison on the actual work units after
the prior MPS assertion. Do not inherit cached learned audio or a silent pretrained
H. A source-H diagnosis can isolate materialization but does not replace native
end-to-end audio evaluation. Scaling and fusion require real generated evidence;
no checkpoint is qualified by the bootstrap fit alone.

## Census cache discrepancy and revised scout

scout-v1 terminated with KeyError before producing population output: one of
1973valid census sources is absent from the6959-entry cached manifest. Preserve
that failure; do not silently drop the source or call the cache the full corpus.
Revision2resolves its actual AudioFilename and byte hash from the named source
.osu, parses its true rows directly, and marks cache availability separately.
All other inputs, source band, split and metrics remain unchanged. New exclusive
scout-v2 and scout-plan-v2.json pin this fallback; this is an explicit revised
run after a diagnosed missing-entry error, not automatic restart.

## Source scout and inspection results

scout-v2 failed because the fallback reused the preceding chart's timing/action
arrays; this was a script error, not a source disagreement. scout-v3 fixed that
assignment and completed all1973sources. Its plan SHA is
697cb524498cb9c57e22fdb4642d81e36a56d9092d4792268611e4d1039ab1fc.
Connected song/audio components yield1602train charts/1313components,
361validation/295components and10reserved/6components. One uncached source
remains in the population with explicit missing cache fields. No label-based
selection preceded this split.

Training-component weighted maximum same-finger4s attack counts have q75=21,
q99=23;8s counts q75=37,q99=43. These describe this star band, not universal
physiological limits. The provisional comfortable pool has1139cached train
charts with noLN or same-columnRH at most40ms and attack peaks below both q75
values. This is a teaching selection; it does not reject other ranked styles.

Thirteen Lens pages were inspected, including a replacement TAP source. Selected
capacity anchors are Dawn4.00255star, Sulyvahn3.99303star, mumei3.98959star and
Kill The Beat3.76671star. They exhibit moving TAPs, a sustained LN with other
finger TAPs, role transfer and regular high-coverage LN units. The observations
cover named source scopes, not every event in these charts; they are not human
gold labels or audio audition. HEAVENLY MOON4.00130star is retained as a deliberate
moderate chordjack contrast:16consecutive H on one column at162–163ms. Its4s
peak19is lower than Dawn's20. Counts alone therefore cannot identify the
desired flowing default, and a real ranked jack is not automatically BAD.

Frozen capacity-data-v1 has4songs and16four-second units: onset from empty
history, two consecutive inspected-organization blocks, and a low-activity
block per song. Whole-song stars and LN-head fraction remain constant conditions;
style is missing. Equal-song complete-Mel moments supply new normalization.
No learned tensor or optimizer comes from an old checkpoint.
Data SHA931740ffde5785f14a2c247609d6a56e5b15e0ef9a68683538dad9434bc29322;
normalization SHAf37ec1c8d4ce724a5e9e692a55243fddb860da06b84fbb47078d3531093dee5f;
inspection record SHA38b0b56baa2b91a116638cea01903a3ccdad8fd4d86fc56e8d53c0ede1b22a15.

## Revision3 backend and capacity entry

Clean product d6eba238ff3872613c6bfe84b75c5e867c881634 implements fresh
audio/H/R1 constructors, joint source H plus complete R/R1 likelihood, optional
retained local row history across plans, and causal padded long-prefix encoding.
The fresh default uses128hidden/64local-audio/96global-audio widths,2global
attention layers,192row hidden,32continuous plan width and one categorical state.
H is unbounded; R1 owns releases and row geometry. All learned parameters are
trainable. HH20/RH1/HR1describe syntactic support; RH/HR1are not acceptable-play
claims. Independent failure evals remain necessary.

The first comparison runs the same16updates in separate fresh CPU4thread and
MPS2thread processes, constructor seed290029, AdamW3e-4/weight decay1e-4,
joint negative log probability per elapsed second, gradient clip5. Recompute
full-audio encodings for every update. Use no count feedback, response energy,
pretrained H, local difficulty proxy or source endpoint hints. Each process
has240s and16GiB physical-footprint bounds and a STOP file. Outputs are exclusive
profile-cpu-v1/profile-mps-v1; no automatic retry or overwrite. Any nonfinite
loss/gradient, backend assertion or resource guard terminates that arm.
CPU plan SHAa461d81a19a5338235c999ad295a4558c63deda4afdaa27831f038873a8db29c;
MPS plan SHA314c443ae71ccdc764c7fbedbabd95573675f78911674a4f958aadb162b527f7.
Compare matching objective trajectories, update time and physical/active/driver
memory, not just final NLL. Finite completed updates within bounds only establish
this backend slice. A further capacity fit requires its own pinned plan and
actual native/source-H output inspection; neither these16units nor their NLL
are a held-out quality result.

Selected local checks:38tests passed in6.18s across segment continuation and
planned distribution tests, including fresh initialization, both H modes,
joint gradients and dense/native scoring with retained history. git diff--check
passed. No remote publication, promoted model or benchmark-path change occurred.
Card remains proposed with no accepted revision; local work proceeds under the
user's explicit training authority, so resulting evidence is exploratory.

## Backend results and revision4 capacity fit

Both16-update profiles completed. CPU took8.9145s with peak1,854,753,168bytes;
MPS took13.9925s with peak3,877,784,576bytes. Maximum paired per-step H and R1
NLL/second differences were .000349 and .000774. MPS driver memory settled at
1,648,427,008bytes on the repeated shape cycle. This is not proof the earlier
variable-shape assertion is fixed. CPU4threads is selected for the small fit.
The actual model has4,782,754learned parameters, all trainable.

The capacity run starts fresh again, not from either profile checkpoint. It
uses32seeded permutations of all16units (512updates), draw seed290030 and the
same constructor seed, optimizer, full-audio joint loss and conditions. Resource
bounds are900s/8GiB, output capacity-fit-v1, no automatic retry, one final
durable checkpoint with optimizer and RNG. Before/after diagnostics score all
16fit units. The capacity-fit-plan.json SHA is
fdddc82c3537833d5f789e8e0d3a9a8b2aa1df270f293ab80153a11a145cca51.

A20percent reduction in both mean H and row factual NLL/second is a capacity
screen, not success. Inspect actual source-H and native-H generated continuations
and keep short-tail, RH, repetition, role continuity and rhythm observations
separate. Teacher-prefix/source-H continuations can distinguish source learning
from policy-induced-prefix drift. No generated-chart quality is established by
the backend comparison or by fitting these deliberately oversampled units.
Do not scale or fuse solely because the loss falls.
