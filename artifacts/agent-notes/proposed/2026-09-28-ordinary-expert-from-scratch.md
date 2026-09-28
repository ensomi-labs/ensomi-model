# Agent Note: Learn an ordinary four-star expert from fresh initialization

Note ID: 2026-09-28-ordinary-expert-from-scratch
Status: proposed
Kind: research
Created: 2026-09-28
Updated: 2026-09-29
Product revision: 96f84fd32218e39ff809c651d73e6edbd39a3500 (executable intervention d6eba238ff3872613c6bfe84b75c5e867c881634)
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

Revision: 5
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

## Capacity512 result and revision5 generated evidence

Fresh512fit completed in235.73s, peak1,892,403,720bytes. Mean factual H NLL/s
on the16fit units changed31.01186to10.72888(-65.4percent); R/R1 changed
23.07761to2.29225(-90.1percent). Both pass the declared learning screen only.
Checkpoint SHA is f7b2037ce5338c593b4f3b7ff01e3c5d20a1cea79b7d3f3461043e6457898167.

Generate12whole-song completions:4sources times3modes, one fixed seed290031.
Modes are source-prefix/source-H at the aligned inspected-plan start,
BOS/source-H, and BOS/native-H. Source prefix is replayed as actual committed
rows with open holds and neural history; no future row materialization enters.
Full audio and the same global source-star/LF controls are used in all modes.
No response guidance, count feedback or post hoc chart correction is enabled.
Only generated-prefix runs receive whole-chart qualification checks; a teacher
prefix must not dilute its continuation's failures into a whole-chart pass.
Inspect the fixed source ranges independently, retaining unresolved facts until
their actual generated tails. Source-H is a diagnosis, not the audio task.

The plan SHA is19c951f2e34948cb092f80edb31f018da7e62318f31b9119f00046b72cf1c495,
exclusive output capacity-native-512-v1, CPU4threads,900s total/90s per case,
no overwrite or automatic retry. Numeric checks include whole-star error<=.5,
whole-LF error<=.1, whole-chart40/80ms LN burden and same-finger4/8s maxima.
Existing evaluators additionally report RH/HR, recurring column membership,
TAP/held-role organization and local H-lattice coverage. No aggregate score can
cancel a failed tail coordinate. Actual Lens review remains required.

New references use the current train components only and keep validation
exceedances visible. capacity-references-v1.json SHA is
9605dd3769b13fc2f7e7fe02f3209fc21ce04657efc678970c7f00103fec18a1.
40ms LN burden limits: TAP-majority0 (5/215validation charts exceed), mixed
.00596125(0/113), LN-majority.00604230(0/11). The LN-majority validation is small.
80ms limits:.0845461(2/215),.253020(0/113),.319486(1/11).
Same-finger4s maximum23has12/361held-out exceedances;8s maximum43has2/361.
These are empirical tail warnings with known false positives, not physiology
or universal style bans. Their observation unit is a whole chart; use scoped
descriptive evidence instead of applying these prevalence cutoffs to short crops.

## Generated result, actual inspection and next causal question

All12outputs completed in98.03s total. All8BOS outputs failed the40ms LN
burden reference and passed the80ms reference. Source-H versus native-H
short-LN counts are Dawn1/31, Sulyvahn9/47, mumei10/68, Kill The Beat9/45.
The4source controls contain zero such LNs. Whole native stars are5.2086,
4.6092,3.8569,4.5527 respectively. mumei passes whole-star and LN-fraction
checks while its68extreme short tails still fail. Native Dawn also exceeds
the4s/8s same-finger limits at28/47. No candidate qualifies or enters fusion.

All30generated Lens pages were actually viewed; source scopes and exact
output hashes are recorded in capacity-review-512-v1.json. There was no
audio audition or human playtest. Source-prefix mumei learns the sustained
column2 role with13other-finger TAP-H groups and a later transfer to column0.
BOS/source-H has only3consecutive companion groups in that scope. Native
mumei produces5ms and6ms LNs. Sulyvahn has not reproduced the long held role;
other patterns partly recover. Alternative TAP arrangements are not automatically
BAD; failure to reproduce a teaching structure and demonstrated player-pressure
failures remain distinct. Most whole-song regions were outside these16fit
units, so this is not a verdict on a fully trained ordinary population.

Post hoc timing inspection found native H pairs at most10ms apart:
152/80/154/77 versus zero in each real source. In Dawn's trained6s window,
all69native H fall within10ms of some source H, yet14adjacent pairs are at
most10ms; the source has61H and zero such pairs. Onset-nearness alone hides
duplicate events. Across native outputs,155/191short LN tails equal the
immediately following H; none are audio-terminal closures. Relaxing local
lattice tolerance from3ms to6–15ms recovers the approximate source periods,
so the evidence concerns jitter/doublets and coupling, not total absence of
rhythm. H substitution also changes later states and RNG consumption; do not
claim per-object causal attribution from this one-seed comparison.

Result SHA a20acd1e0a65906add77632f97962ca2bee2ee198587c293c62b00cb3ce808f8;
visual review SHA09021a8d46876d2c415a9b7d76646ef99f23c375d0dd0815ada5468a5220e2be;
timing interpretation SHA0eb17d5854544865f80b7a83b6b1c0450d98453e570d0f544810fbb0f6a41a5e.

## Real-time information and response curves

The user specifically suspects inadequate awareness of next-row elapsed
real time and asks about exponential/custom response curves grounded in hand
constraints and real corpus. Source audit confirms existing time inputs:
H reads elapsed-since-H and absolute bin time; R/R1 read exact attack/release
clocks, LN ages and future H distances. H shares a bin query across10native
offset hazards. Time is not entirely absent.

The segment decoder removed RowConsequence and ignores the still-supplied
local/timing candidate-consequence tensors. A read-only4window ablation on
checkpoint512 confirms zero distribution change when those tensors are erased.
Erasing exact clocks changes mean TV by.0017–.0200; erasing future preview
changes it by.0906–.2679. This is an input-path ablation, not a legal retiming
or physical correctness metric. Probe SHA
453f1e0ace8f84e2f8159e6de7a573fa87913479aa71a830a17e80be64793c01.

ActionResponseState already uses exponential tau250/1000/4000/16000ms and
100/gap transition impulses. These are engineered hypotheses; corpus fitted
reference limits, not identified physiological recovery. An isolated25ms LN
has250ms HR increment16below the reference17.44, explaining its zero work.
Separate acute transition response from accumulated regular workload. Occupancy
persists exactly but does not supply a tonic input to the decaying impulse bank.
Coarse hand turnover/partner-held coordinates do not establish ordered
coordination sufficiency. ResponsePlanner has true4s horizons/2s publications,
but retries only R/R1 with shared H; a bad private H requires upstream whole-
candidate regeneration rather than indefinitely resampling row geometry.

An additional exact transition census on1602train ranked3.5–4.5star charts
found no HH<=40ms among2,766,890same-finger attack intervals. There are25/1490
charts with HR<=40ms and10/1484with RH<=40ms. Equal eligible component then
chart weighting gives transition fractions0/.00050651/.0000736915 forHH/HR/RH.
This is ranked support evidence, not a measured human limit. The corpus audit
SHA is5d2f88537935f55ad18accb2d1c4724e87db0dfaebf6a480ba274b24099300e1.

Primary hand research checked: Häger-Ross/Schieber2000(PMID11069962), finger
coupling/frequency; Kelso1984(DOI10.1152/ajpregu.1984.246.6.R1000), frequency-
dependent bimanual coordination; Bächinger et al.2019(eLife46750), motor slowing
and recovery. These motivate channels/multiple scales, not transplantable mania
thresholds. Canonical V3 excludes measured individual physiological fatigue;
use physiological evidence as a structural prior without changing that scope.

Selected next direction: explicit real-time advance then candidate action;
separate monotone acute HH/HR/RH curves, multi-scale per-finger/hand memory,
tonic occupation and ordered coordination; expose candidate consequences to R1
and preserve vector/time-window distinctions in frontier acceptance. Test clock
dilation, absolute-translation invariance, equivalent split waits and matched
lane/time contrasts. Restoring an action-time branch alone is untested and is
not declared a fix. Larger source training and on-policy correction remain
necessary candidates; no new physiological model or repaired planner is trained.

Product documentation now records these findings self-containedly in
docs/research/ordinary_expert_from_scratch_zh.md and
docs/research/event_time_gameplay_response_zh.md. Documentation links/math
delimiters and git diff--check passed. Executable tests remain the38selected
passes on d6eba238ff3872613c6bfe84b75c5e867c881634; subsequent product edits
are prose only. All profile, fit, generation and inspection processes are
terminal. No overnight run is live, no model was promoted, no remote push or
benchmark-path mutation occurred. Recommendation is REFINE, not SUPPORTED;
the proposed Card remains unaccepted and the overall playable2–6star goal
remains unmet.
Do not scale or fuse solely because the loss falls.
