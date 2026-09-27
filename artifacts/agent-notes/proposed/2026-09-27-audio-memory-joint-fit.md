# Agent Note: Matched joint fitting of audio-history memory

Note ID: 2026-09-27-audio-memory-joint-fit
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: baed4d720f3efc0c03ab8c2249bb96110497a586
Scope: Matched baseline/memory full-audio H/R/R1 fitting and fixed temporal, semantic-control and publication regression qualification
Related: 2026-09-27-audio-history-memory, 2026-09-27-playability-regression-evaluation, 2026-09-23-audio-skeleton-r1-integration

## Design and preparation

Previous goal turn is progress: implemented and verified memory architecture,
completed8+24 discarded joint integration/resource updates and reusable EVAL
with real historical failures and ranked positive exceptions. All those handles
are terminal. The next substantive task is actual matched quality fitting, not
another unchanged microbenchmark or rejected player-state fit.

Baseline is core2500, SHA0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8.
Compare ordinary controlled baseline continuation to AudioMemoryModel with the
same inherited tensors plus zero-output new modules. Both update audio/H/R/R1;
no frozen-audio encoding cache. Main learning will use384 updates with two source
microbatches (one population, one human-supported) per update. Inherited lr3e-5,
new memory/pyramid modules3e-4, AdamW decay1e-4, clip1. Final endpoint only; earlier
checkpoints serve recovery rather than selection by NLL. Integration weights are
not an initialization. A complete bounded Card will pin prepared inputs/gates
before either fit starts; this section authorizes only preparation under the
user's standing research instructions.

Preparation scope: ranked TRAIN charts with recomputed whole stars2–6 and complete
cached audio60–480s. Limiting to240s leaves5855 population charts and162 human
scopes; allowing480s gives6640/241 before final profile admission and additional
positive-reference exclusions. Human coverage remains sparse, particularly
prominent Tech; do not fabricate labels or claim full semantic coverage.
Exclude song groups/audio identities for the old fresh-audio panel, four genuine
style guards, four exposed near-four-star sources and inspected positive response
references. Population draws choose a difficulty band, then song group/chart,
then a uniform32s clock-partition interval. Human draws use the existing capped
inverse-square-root concept/assessment/difficulty-cell weights, then a32s window
containing its original annotation when possible. Original label scopes remain.

A proposal's group/branch or human selection is retained while retrying declared
profile-incompatible windows; every rejection is logged. Freeze the same accepted
768 examples and controls before either arm learns. Population importance retains
its clock-partition inclusion weight; human windows use per-second loss. Both
families have equal microbatch weight. Per-field controls use actual whole-chart
values by default, with independently sampled30% local difficulty and30% local
LN overrides on the scored window; local difficulty outside2–6 falls back to the
whole value, not clipping. No-head local LN fraction is undefined and likewise
falls back. Drop each control family independently with probability.15. Human
style assessments keep their original scopes/known bits. Shared control seeds
are frozen in prepared draws.

Preparation uses CPU source replay/collation only, not model fitting; maximum
1200 active seconds, fresh preparation output, no overwrite. It must preserve
actual endpoints and report profile exclusions. Before fitting, record accepted
coverage, input/script hashes, all reserved groups, exact evaluation cases/seeds,
per-range guards and the Mac runtime/memory stop. Reserved canonical validation
songs will be selected separately from the exposed developmental diagnostics;
no claim of being unseen by every inherited checkpoint is allowed without audit.

### Preparation exposed a target-support conflict

Initial preparation39869 terminates after96 logged update pairs because a retained
human Tech-supporting anchor has no admitted32s context among200 proposals.
Keep the partial preparation directory; no fit has started. The source is ranked
Senbonzakura [Oriental Cherry], SHA48becb6ae141b18562e17de67239ee0a38f9bd424ddf1acc5567c9a5f0dee04f,
wholeD5.8448, local annotatedD4.4017 on[173990,175704). Nearby true LNs last42–43ms;
HH minimum85ms and RH minimum57ms. Thus the rejection is not a too-fast repeated
attack. Four tested covering contexts fail the old60/50/50 support, and relaxing
HR to40 alone admits a534-row32s context. Probes26426 and50396 are terminal.

Revise the common experiment profile to Recovery(hh=60,rh=50,hr=40) for both
baseline and memory arms, including their untrained comparison. Preserve the
actual source endpoints; do not relabel short LNs as TAP or retime them. This
common protocol adaptation is separate from the architecture intervention and
must be reported against the old selected60/50/50 runtime. It is not a claimed
cause or repair of all previous held-texture problems. No default profile changes.

Fresh preparation-v2 completed under99761. It pre-admits one32s context per original
human anchor, trying up to8 positions while retaining its actual annotation
scope; exclusions and reasons are explicit. Recompute capped human sampling
weights over admitted anchors. Both arms use the same frozen context/controls;
population clock-partition sampling is unchanged. This avoids silently replacing
an unreachable anchor or narrowing its scored context to make a test pass. The
initial preparation is not the final input identity. Any further coverage loss,
especially rare prominent styles, must be reported before the fit Card is fixed.

## Final preparation evidence

Preparation99761 completed in446.2784s with6552 eligible population charts,
236 admitted human anchors and3 explicit exclusions. Population proposals had20
rejections:14 row-spacing/continuation,5 release-likelihood,1 no eligible release.
All15 concept/assessment cells remain. Two real prominent-Tech anchors are a
coverage limitation, even when oversampled. No source timing or endpoint changes.

The draft random human sampler produced only one prominent-Tech draw. Finalizer
seed274002 instead balances the15 concept/assessment cells at25–26 draws each,
then samples an available local-difficulty integer band, group and anchor within
the cell. It retains the pre-admitted contexts and original scoped human labels.
Population identities are unchanged. Numeric control overrides now occupy8/16s
interior ranges, with at least4s scored before and after: this trains restoration
after expiration. Difficulty and LN attempts remain independent30% with factual
targets and independent15% family dropout. Undefined/out-of-range local targets
fall back to whole-chart conditions. Before dropout,175 difficulty and222 LN
overrides cover348 of768 examples. This supersedes the draft sampling/control
procedure above, and is common to both fitted arms.

Final inputs are preparation-final-v2; handle95218 completed in50.1174s.
The earlier preparation-final output67979 has identical draw bytes but superseded
metadata. There are417 selected sources from383 song groups and768 examples.
Qualification builder12951 completed with28 cases. All preparation handles are
terminal; no learning updates have run when this Card is committed.

## Experiment Card: audio-memory-joint-fit-v1

Revision: 3. Accepted revision: none. Owning Note remains proposed. The user's
standing authorization explicitly includes model implementation, local experiments
and overnight compute. This is an exploratory comparison, with no automatic
runtime adoption, Note acceptance, remote push or final-system completion.

### Question, alternatives and intervention

Does jointly trainable multiscale full-audio attention and queried elapsed-time
history improve native rhythmic/held-texture organization while retaining
playability, scoped controls and publication latency? Compare ordinary core
continuation against AudioMemoryModel continuation from identical inherited
tensors. The combined architecture is the single between-arm intervention;
its audio and three history readers are not separately attributed by this test.

Closest analogues are Music Transformer relative/history attention, Transformer
Hawkes Process's history-conditioned event intensity, and MusicVAE's separation
of time scales, as discussed in the owning audio-history-memory Note and product
analysis. The transfer is content-dependent retrieval plus multiple temporal
resolutions; the difference is full-audio conditioning, exact typed execution,
separate H/R/R1 ownership and online publication. This is an adaptation, not a
novelty claim. Supervision coverage, source likelihood versus rollout mismatch,
and ineffective condition use remain live explanations. Reject more unchanged
player-state-only fits: their source/response pilots already failed native guards.

Both arms train audio, H, R and R1 jointly. Memory adds the committed .5/2/8s
audio pyramid and separate64s H/R/R1 retrieval paths. H does not receive generic
row content; R may read its exact permitted LN state; R1 owns lane/count/TAP/LN
materialization and directly reads audio and planned skeleton. Full-song Mel
remains available and differentiable during each microbatch in both arms.

### Fixed identities and procedure

Executable source OID: baed4d720f3efc0c03ab8c2249bb96110497a586. Model, tests,
dependency files are committed. At launch the shared workspace also has unrelated
AGENTS.md evaluation-routing edits and untracked
docs/research/audio_architecture_walkthrough_zh.md. Preserve them and record
workspace status; do not claim the entire checkout is clean. No executable source
diff is allowed. The comparison is exploratory under this documented provenance.

Initial core2500 SHA0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8,
4,583,985 parameters; memory7,616,517. Discard all integration-fit weights.
Common Recovery(hh=60,rh=50,hr=40), inherited head-pressure and LN preferences.
The previously selected runtime uses HR50. HR40 is a common target-support
adaptation, not part of the between-arm architecture effect; measure an unfitted
HR40 comparator and do not equate it to the old default profile.

Inputs in artifacts/joint-audio/20260927-audio-memory-joint-fit-v1:

- preparation-final-v2/result.json SHA71e9e21850331eaa7bc9f02ac61922a4cec2efe296be52b94de3e1559c9b7fdc;
- draws.json SHA6ed9e3609d31d19e459eb4703766e55f1c49d95466ffdbc008f22fc205ce77fd;
- sources.json SHA69f625902ad45cae45d4afc20d796e8bebd3e49fc3644a2fad4bf8ba73697975;
- evaluation.json SHA4dde12ece2b96a29a745ae1f68f70c1a1e266ac349f146bd5de79fd49d11239b;
- qualification-plan.json SHAcf19396e4977f6d3bd4857359ae516391838d15ae967f720779bed59408780bd;
- train.py SHA7a66b3c494eb14fa9a3f0250c00f41996fed8ae1987f8560ca0364b221538af7.

TRAIN population: ranked native2–6stars, complete audio60–480s; reserved audio
identities/song groups excluded. Manifest SHA4cea2672387b6293a4da0846be479d8bd9c857e55535dc8143bd11d65b06d2c4,
human cohort SHA252fe593514adc5498011b55dbf62224cba55651025233fc1ed9235078bb95b8.
Freeze384 updates, one population and one balanced human microbatch per update,
each32s scored interval. Source likelihood uses exact interval survival/release
support and native R1 recovery/LN-amount preferences. Population inclusion weights
and per-second human loss are retained; each microbatch contributes one half.
AdamW inherited lr3e-5, added branches3e-4, decay1e-4, gradient clip1. Seed274100,
step seed274100+step. Full audio padded to3200-frame buckets. No frozen encodings,
learned cache across updates, fabricated endpoints or loss-based selection.

Commands, sequentially from product root:
uv run --extra mps python artifacts/joint-audio/20260927-audio-memory-joint-fit-v1/train.py baseline
and the same command with memory. Fresh baseline-384 and memory-384 outputs;
no overwrite or implicit restart. Save every64 updates and final endpoint with
optimizer/RNG identity. Earlier checkpoints are recovery evidence, not alternative
quality candidates. A changed restart/resume procedure requires a Card revision.

Environment Apple M5/24GiB, Python3.10, Torch2.11.0, MPS, one CPU thread.
Maximum10800 active seconds per arm,18GiB sampled process footprint or MPS driver
memory, at most8GiB new study outputs, no network training. Sample after audio,
forward and backward as well as update; stop before another microbatch when a
bound is exceeded. Stop on nonfinite loss/gradient, source identity mismatch or
execution contract failure. Observation timeouts are not failures or permission
to restart. A STOP marker ends at an update boundary. Store the last completed
update on bounded stop; do not claim a partial update completed.

### Frozen evaluation and decision rules

Run28 matched cases per arm and unfitted common-profile reference: nine historical
Stream trajectories (three songs, three seeds); four style requests; one live
control switch; four exposed near4star source comparisons plus two Stream
variants; four reserved validation songs; four genuine style guards with fixed
source H. Full exact assets, seeds, controls, independent ranges and reference
rows are frozen in qualification-plan.json. The fixed-H cases test R/R1 on their
own and cannot establish improved native H generation. Canonical validation is
reserved for this comparison, not proven unseen by every inherited checkpoint.

Baseline historical evidence: under old HR50, nine Stream generations have mean
integrated sustained-column excess J=.0275164272, worst=.1061111; a4s private
response planner lowered mean to.0004245142 while difficulty MAE remained about
.48. Those are context, not numerical substitutes for the new HR40 paired runs.
Nine seeds span only three independent songs; no precise population confidence
bound is claimed. A failed response fit had mean J=.1071601211, showing why source
NLL or small conditional probes cannot certify native quality.

Primary improvement requires an inspectable musical/texture gain on at least
two of four developmental source comparisons, with no contradicted severe new
failure on four reserved cases. Review both the frozen problem windows and
new diagnostic maxima, not a favorable excerpt selected after training. Ranked
source charts are valid examples, not unique target arrangements. Keep labeled
Jack/Tech/Trill/LN behavior distinct; do not enforce equal lanes or ban genuine
style-appropriate repetition. Lens inspection and listening/playtest gaps must
be stated, with automated facts separated from semantic judgments.

Mandatory quantitative guards:

- All exports complete and physically valid; no same-column attack below20ms.
- Nine-Stream memory mean J <= max(.001,1.1 * continued-baseline mean J), and
  <= max(.001,1.1 * unfitted-common mean J).
- Nine-Stream whole-star MAE no more than.15 above either paired comparator.
  Stars are insufficient without temporal/semantic guards.
- Switch before/override/restored ranges scored separately: absolute local
  difficulty-proxy error <=1.0 and known LN fraction error <=.10 in each range.
- Loaded-model/cached-Mel first30 publication <=2s; maximum2s publication service
  <=2s. Actual deadline replay uses2s lookahead and startup waits for both30 rows
  and settled coverage through1999ms. Record necessary startup and any misses.
- No new severe sustained-jack, fictitious-relief or held-texture regression in
  inspected fixed witnesses and newly found maxima. Any such failure defeats
  adoption regardless of aggregate averages.

Use ChartTrace/evaluate_scopes on reparsed exports. Preserve committed pre-range
strain and open-LN occupancy; never close holds at control boundaries. Report
trailing .5–16s attack maxima, recovery at.25/.5/1s, held fractions and multiscale
activity/finger-transfer contrasts. Compare Mel correspondence against circular
shifts only as a diagnostic. No maximizing CKA, CV, contrast, low density, low
LN occupancy or low J in isolation. Whole/native, fixed-H, and differently
controlled ranges are separate populations. Qualification runtime <=7200 active
seconds total with fresh per-arm outputs; failure/partial exports remain visible.

Positive: paired qualitative improvement with every guard met supports a larger
diversity/seed study, not final-system promotion. Negative: more NLL improvement
without native gains, lost styles, difficult/unplayable concentration, flattened
textures or missed publication deadlines rejects this fit as a candidate.
Ambiguous: partial resource stop, underfit new modules, sparse label coverage or
mixed seed/song outcomes motivates a specific revision, not another unchanged
microbenchmark. Remaining confounders include only384updates, repeated rare
human anchors, factual rather than counterfactual controls, inherited checkpoint
exposure, matched seeds with diverging stochastic paths, and both capacity changes
bundled together. NLL is an optimization diagnostic, not the research endpoint.

## Execution handoff: matched baseline running

Baseline384 launches under74054 with the frozen trainerSHA7a66b3c494eb14fa9a3f0250c00f41996fed8ae1987f8560ca0364b221538af7
and command caffeinate -i uv run --extra mps python
artifacts/joint-audio/20260927-audio-memory-joint-fit-v1/train.py baseline.
At observed step222/384,722.45s elapsed, sampled footprint11.1335GiB and driver
peak4.9619GiB. No contract/nonfinite/resource failure. NLL does not qualify it.
Do not restart74054 on a short observation timeout. Memory fit has not started.
Preparation builder12951 and import-only driver check36048 are terminal.

Behavior-neutral evaluation descendants58321912c22dd2f6bd0dabf50e52a116e1203bae
andb69d360bd044ff11327c98b50a4d514f11728470 touch only the evaluation module,
its tests and guide; model/trainer/dependency contents are identical to the
recorded baseline. The memory arm may run this exact descendant after checking
that audit, recording its actual source OID. This does not alter the causal
intervention, controls, exposure, optimizer or fit budget. Unrelated AGENTS.md
and architecture-walkthrough changes remain preserved and unstaged.

Qualification and analysis drivers are implemented and import-checked, not run:
qualify.py SHA9d253994ef50a2d02f6e5cff8609e2f1c33e3e75e89adc79df5266f7032081de;
analyze.py SHA52acef65639869383eb62b471902b3e7cd3396aaa4cd2c6f4ce04e1c2e937a74.
The qualifier waits for both384-step endpoints, then runs unfitted/baseline/memory
sequentially on CPU with one thread, fresh qualification-<arm> destinations.
Command uv run --extra mps python with the owner/qualify.py path and arm argument.
Per-case300s budget stays inside the declared7200s total across all arms.
It saves actual row/coverage publication, onset-preserving fixed-H checks,
unchanged published prefixes across the live switch, full reparsed osu reports,
scoped proxy/pressure/occupation/texture/correspondence, and precise overload
witnesses. Endpoint/asset/plan hashes are verified. Partial failures stay visible.

Analyze.py makes paired independent-range reports and guard results, retaining
semantic/meaningful-improvement status as pending. Its review plan includes the
frozen seven developmental/historical windows, top two new Stream pressure
contexts, all four original fixed-H style scopes, and each reserved song's
largest added/removed8s occupation and maximum4s pressure contexts. These are
inspection priorities, not automatic BAD labels. Neither driver updates weights
or selects earlier fit checkpoints. Model evaluation and Lens review remain
outstanding. Source/example evidence and every run stay exploratory; no Note
status change or model/default promotion.

## Result Log and Card revision2: avoid expanded attention products

Baseline74054 completes384 updates in1227.9036s. Final checkpoint
baseline-384/step-384.pt SHAa6916403daf6c658f0952d1181703732156f4ac37a8e5b773007c5dc4e2c5c24.
Sampled footprint12.3900GiB, active MPS2.6185GiB, driver5.1481GiB. These ledgers
are overlapping, not additive. No qualification has run and no quality claim.

Memory62347, sourceb69d360bd044ff11327c98b50a4d514f11728470, stops normally at
its recorded memory guard after10 completed updates, during update11, in90.7736s.
Footprint reaches18.3099GiB, active sampled peak5.3297GiB, driver14.4726GiB.
Its terminal partial checkpoint SHA b45bc0a00e1cac057223622973eeea34a858d09de733dc83f8d7681e4c155f9a
is retained only as failed-run evidence, not a quality candidate or restart seed.
The driver did not tag the exact failing phase; do not infer byte ownership from
counter differences. Update11 begins with391.732s full audio. Prior short-input
stability did not establish diverse-corpus memory stability. Both fits are now
terminal; no current learning process remains at this revision's creation.

Revision2 preserves the architecture, information flow, parameters, training
law, all768 examples, controls, seeds, inherited initialization,384-update endpoint,
18GiB limit and frozen qualification. It changes only the implementation of the
history attention contraction and the fresh retry destination. Replace explicit
query-times-key and attention-times-value elementwise products with PyTorch2.11
scaled_dot_product_attention, dropout0, retaining the learned age bias, invalid
mask, two-hand sharing and zero null key/value. API reference:
https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html.
The avoidable query/cell/channel intermediates are identifiable in code; that
alone does not prove they explain all observed memory growth.

Before the retry, compare nonzero outputs and input/parameter gradients against
the old analytic contraction on CPU and MPS, and run the affected memory/native
law tests. Preserve zero-law, causality and scoped rollback. Record a clean code
commit and script hash. Start again from core2500, not the partial10 updates.
Keep original train.py; create train_sdpa.py and fresh memory-sdpa-384. The
baseline endpoint remains the comparator because the changed contraction belongs
only to the added module; no baseline math changed. The retry has10709 active
seconds, so original90.7736s plus retry stays inside the previous10800s arm budget.
Add phase/branch/step-tagged resource observations to explain any subsequent
limit without attributing overlapping counters. Do not silently raise the limit,
shorten full audio, or remove model history. Another resource failure requires
its own concrete adjustment; no unchanged retry loop.

Qualification resolves the memory endpoint from memory-sdpa-384 after both arms
complete. Same28-case plan, seeds, scope guards and7200s evaluation budget.
This is an exploratory numerical-equivalence/performance repair within the user
research task. It does not establish the causal benefit of memory or accept the
Note. Original revision1 procedure and terminal evidence remain above.

Revision2 implementation ec9c1efe5282cb106d6d9302fcf33af6b5672cb4 replaces only
the added history reader's contraction with standard SDPA, retaining the masked
age bias and explicit null values. Fifteen memory-owner tests pass in10.06s,
including new nonzero CPU/MPS output and all input/parameter gradient comparisons
against the expanded reference. Native H/R/row parity, ownership, zero law,
checkpoint, fork/rollback and padding tests remain covered. No claim that this
already resolves large-corpus memory: that is measured by the retry.

Frozen fresh trainer train_sdpa.py SHA6e7a80119222fe6d81e30e4a3e70e1c160bd0adfedddb1cde8d0dce1e6a40543;
updated endpoint resolver qualify.py SHA0e3911648d172fee6aa959c621452b336e17fb6e0b6082a39261d804b5e521bd.
Retry command caffeinate -i uv run --extra mps python
artifacts/joint-audio/20260927-audio-memory-joint-fit-v1/train_sdpa.py memory.
Fresh output memory-sdpa-384, no prior partial-weight resume. Resource samples
now record audio/scoring/backward phase and exact step/branch. The completed
baseline remains its original source/driver/checkpoint; all model math shared
with baseline is unchanged. No qualification output exists yet.


## Card revision3: bounded activation materialization

Standard SDPA retry41712 is terminal:10 updates,81.3772s, then the same18GiB
limit during update11/population/scoring. Exact final sample:footprint19.3968GiB,
active4.7980GiB, driver15.5376GiB; run-wide active peak5.3408GiB. Partial checkpoint
SHAd6fbb630c27646eca0e89f26b30b777b79a76306fe55dea19c3023015bb4e273 remains
failed-run evidence. Standard contraction alone did not solve diverse-input
memory. Do not claim the allocator or a specific buffer owns the driver residual.

The reader still gathers K/V separately for every query/cell/hand/channel and
retains them for backward. Bound that materialization: project shared K/V tables
once, evaluate query groups of128 under non-reentrant activation checkpointing,
and recompute those small gathers during backward. Concatenate every query output
and retain full gradients into query, historical states, full audio, age bias and
all parameters. Inference without gradients keeps the direct SDPA path. This
changes computation/storage, not history span, audio coverage, row law, target
examples or optimizer objective. Compare257-query nonzero outputs and all input/
parameter gradients to the original analytic formula on CPU/MPS, alongside
existing native-law tests, before another main run.

Fresh retry train_checkpoint.py writes memory-checkpoint-384, again initialized
from core2500. Preserve both failed runs and original scripts.10709s is superseded
by10627s for this retry, keeping both failed attempts plus the retry inside the
original10800s memory-arm budget. All other fixed comparison and qualification
fields remain. Update only the memory checkpoint resolver after both384-update
arms complete. Do not reduce complete-song audio or the64s memory to make the
budget pass. Source commit, tests and driver hashes must be pinned before launch.

Revision3 source7fdad44331c75da035fdb9720d0875aa02bdd732 retains shared projected
K/V tables and checkpointed128-query gathers;17 memory-owner tests pass in10.36s.
The257-query CPU/MPS cases exercise multiple checkpoint groups and compare all
input/parameter gradients to the original expanded formula, alongside direct
four-query cases. Existing teacher/native and ownership/rollback checks pass.
Only this model family's implementation and its tests/guide change; no baseline
factor, training objective or frozen data changes.

train_checkpoint.py SHA186df999c9a2dbd7acf9833dc58405835ca749fcabb95e492401a27f96d93470;
qualify.py SHA357e08aa283b6d5d2872f46a9513be2d2df00c84111b79287aa58cba3551c817.
Command caffeinate -i uv run --extra mps python
artifacts/joint-audio/20260927-audio-memory-joint-fit-v1/train_checkpoint.py memory.
Fresh memory-checkpoint-384 endpoint,10627s remaining arm time and18GiB guard.
Original and SDPA attempts are terminal, baseline384 is complete; qualification
still awaits a completed matched memory endpoint. Implementation equivalence is
verified, while large-input resource benefit is still an empirical question.

## Live revision3 handoff

Only main fit handle96244 is live. It runs the pinned train_checkpoint.py memory
command and writes memory-checkpoint-384. Do not restart it or launch another
accelerator fit on an observation timeout. At last observed step21/384,
162.5240s elapsed; sampled footprint peak13.1952GiB, active4.5291GiB,
driver8.2419GiB. No resource/contract/nonfinite failure so far. The previous
step11 failure is passed. This does not establish stability for all384 examples.

Matched update11/population/scoring, identical source
2d4720a783bc89740ecd12493183e6a2352ad29a3d58873c8c77f36040494bbb:
standard SDPA footprint19.3968GiB, active4.7980GiB, driver15.5376GiB;
checkpointed groups footprint10.3399GiB, active3.0202GiB, driver6.4943GiB.
The changed saved-activation policy has a measured resource benefit at this
same input/phase. Do not turn it into a full allocator root-cause attribution or
a whole-corpus guarantee. All source/audio/history information and gradients are
retained, with parity already tested. Earlier failed10-step weights are not used.

Next actions: poll96244; if it completes384, verify final result/checkpoint hash,
then run qualifier arms unfitted,baseline,memory sequentially with no concurrent
fit, followed by analyze.py and the frozen Lens review scopes. Keep per-control
ranges separate. No native model quality qualification has run yet. If another
resource stop occurs, use its tagged phase and active-versus-driver behavior to
choose a concrete execution repair; do not silently retry unchanged or raise the
18GiB bound. Periodic exact optimizer/RNG checkpoint/resume in fresh processes is
an available implementation alternative if accumulated process state proves the
remaining constraint, but is not yet selected or authorized by this Card revision.

Terminal handles in this goal turn:12951 qualification-plan builder;15135 trainer
syntax/runtime check;74054 completed baseline fit;96809 failed witness audit
adapter then45835 successful audit;36048 qualifier/analyzer import;76365 final
witness audit;73558 nine-page Lens render;62347 original memory bound stop;
66456 local SDPA docs;54620 fifteen memory checks;41712 SDPA bound stop;70713
seventeen checkpointed-memory checks. Only96244 needs resumption.
