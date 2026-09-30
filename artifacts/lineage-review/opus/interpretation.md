# Lineage review, slice 08 (independent): the reasoning trajectory of the audio-skeleton lineage

Reviewer: Claude (Opus), control plane, 2026-09-30. Independent of the Astra slice reports, which I did
not open. Shareable; no human wording (the human is cited only through the paraphrased feedback
index, V-numbers).

Evidence labels: **doc-claim** (lineage document, note or commit message, not re-derived);
**checked-git** (commit list, timestamps, diffs, grep over trees at the tags); **checked-artifact**
(files read or recomputed on bings-mac, read-only); **my-class** (my own classification, method stated).

## 1. What I read and checked

Read: notes `RESEARCH.md`, `lineage-review/README.md`, `audio-skeleton-human-feedback-index.md`
(sections 2, 3, 5); all 16 files of `relay-notes-tagged`; the 202-commit lineage list with UTC times;
the `agent-notes` history since 09-22 (363 commits: 27/76/64/68/69/59 on 09-23..28) and the notes for
the turning points; late documents `audio_joint_expert_question.md`, `native_pattern_failure_analysis_zh.md`
(opening, section map, capacity passages), `event_time_gameplay_response_zh.md`,
`ordinary_expert_from_scratch_zh.md`; parts of `audio_conditioned_choreography.md`,
`r1_transfer_stability_audit.md`, `clean_joint_proposal_learning.md`, `audio_rhythm_hierarchy_zh.md`,
`.agents/skills/research-triage/SKILL.md`, `docs/formulation/gameplay-state.md`; both pasted expert replies.

Checked on the mac (read-only, light): clean-joint `fit-plan.json`, `source-plan.json`,
`supervisor-v1/result.json`; fresh-expert `capacity-fit-plan.json`; head-time and LN-duration counts
over the exported `.osu` of `20260928-clean-joint-proposal-v1/native-4096-{fresh,inherited,early}`
(84 charts) and `20260928-ordinary-scratch-v1/capacity-native-512-v1` (12 charts); artifact
modification times during two overnight gaps. Scripts in `/tmp/lineage-review/opus-interpretation/`.

## 2. How the agent worked (the shape of the trajectory)

- **Cadence.** An agent-note commit every 10 to 20 minutes for six days, in a fixed rhythm:
  "Design X" -> implement -> preflight -> short fit -> "Record X" -> "Design Y". Roughly 85 distinct
  interventions in 202 product commits (my-class). Thirty agent-note entries refer to "the preceding
  goal turn", typically as having made progress (checked-git): the run was an automatic goal-continuation loop in which
  each turn chose the next step from the last result.
- **Unit of evidence.** Typical fits were 32, 128, 384 or 512 updates, often on adapters of
  12,288-16,384 parameters with the rest frozen (doc-claim, many notes). The largest run of the
  lineage, the three-arm clean joint comparison, was 4,096 updates at batch 2 over 8,192 eight-second
  windows, i.e. one pass (checked-artifact: `steps 4096, batch 2`, 8,192 examples, 16,957 s wall
  for the three arms together). Joint models saw about 250,000 event rows per fit
  (doc-claim: 248,836 in the 2x2, 253,512 in the lineage-stage comparison) against R1's 6.75M
  source-onset exposures, while also learning audio-to-timing from scratch.
- **Method was partly instructed.** The repository's `research-triage` skill (unchanged since 08-21)
  says to design "the smallest test that can change the decision", to "prefer a fail-fast slice over
  a full training run", one causal intervention per card, and to leave research direction to the
  human owner (checked-git). The human's first budget answer asked for small probes first (V11);
  on 09-24 the goal text asked for large whole-system steps (V34). The agent's gate
  "Scaling and fusion require real generated evidence" (note `52b251b`) meant a small fit had to pass
  native quality before anything was scaled; no small fit ever passed.
- **Compute was idle overnight.** No file anywhere under `ensomi-model/artifacts/` was written between
  09-25 21:00Z and 09-26 01:30Z, or between 09-26 21:00Z and 09-27 02:50Z (checked-artifact, mtimes;
  the 09-30 cleanup deleted caches and superseded checkpoints, so this is strong but not conclusive).
  Whether the session was paused by the human or idle by itself I cannot tell.

## 3. Diagnosis timeline

Trigger: **H** = human message within about three hours before (feedback index times), **E** = expert
reply, **O** = the agent's own evidence. Locality: **L** = change made at the module nearest the
symptom (sampler prior, loss term, condition path, count law, selector); **U** = an upstream
assumption re-examined (factorisation, ownership, base weights, data regime, representation).

| # | When (UTC) | Diagnosis that steered the work | Alternatives on record | Action | Trig. | Loc. | How the belief aged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 09-23 08:15 | R1 is a sound row model that lacks only event times; audio -> onset/release schedule -> frozen R1 | none before starting | ms sensitivity probe, BeatThis frame pilot | H (V09) | - | Contradicted in 2 h by own evidence: added release opportunities shorten LNs (median ratio .686/.375), release-only F1 .035-.051 (doc-claim) |
| 2 | 09-23 10:28-11:35 | Target is the joint law of a playable chart given audio; learn Mel encoder, timing and rows together; defer memory | Written table of primitives: marked point process, waiting-time mixtures, persistent latent, **soft beat coordinate**, **LN as a duration object** vs state-conditioned release, receding horizon (`9538d7d`, checked-git) | 2.95M joint hazard + complete-row model initialised from R1 | H (V12-V21; the agent's "encoder sufficiency first" was rejected) | U | Factorisation kept to the end. Beat coordinate and LN-duration alternatives never tested |
| 3 | 09-23 12:21-14:59 | Failures are query-coverage gaps and duplicate rapid TAPs | - | complete-wait supervision; optional 27-ms head prior, default off | O | L | Duplicates recur in 09-28 fresh expert (row 23) |
| 4 | 09-23 15:21-15:49 | Native-history drift can be distilled away | - | next-event KL on native histories | O | L | Failed same hour; expert question written at the human's request (V22) |
| 5 | 09-23 17:33 - 09-24 05:14 | Expert order: coverage, then 2x2 audio/history, then latent | expert's list | 48 -> 240 groups at matched 38,400 examples (23 min); 2x2 cells of 1,200 updates (6-10 min each); latent coded, never fitted | E | U in form | Coverage: held-out NLL -15.4% on new songs, "curve was still improving" (note `8ff4f23`) -> not continued. 2x2: no cell met 3% -> full-song audio set aside until the human raised it on 09-27 |
| 6 | 09-24 05:35 | The joint model is not the released R1 (frontier2, seed, landmark memory omitted) | - | audit; three lineage-stage arms, incomplete (246/276 cases) | H (V26) | U check | Confirmed (doc-claim); R1 backbone kept |
| 7 | 09-24 07:56-08:40 | Timing must not read row content: split H, R, R1 under an information contract | expert 2 later calls H isolation too strong | planned H/R/R1 model, frontier2 restored | H (V29-V32) | U | Kept to the end; H never received action history |
| 8 | 09-24 09:34-13:05 | Head starvation, release-deadline mass, short attacks | - | bounded head suppression; release-wait conditioning; optional short-attack correction | O (+V33) | L x3 | Superseded |
| 9 | 09-24 13:26-18:07 | Runtime has headroom, so quality is the bottleneck | speculative decoding deferred | shared profiles; private continuation screening | O (+V34) | L (selector) | Own finding: screening cannot create missing proposals |
| 10 | 09-24 18:58 - 09-25 00:53 | Profile, count routing and short-future availability | - | density routing via H; count factorisation; row conditioning on availability | O | L x4 | Superseded |
| 11 | 09-25 01:49-06:12 | Calibrate to ranked 2-6 star; move counts and types into a typed skeleton | - | typed resource planner, scoped controls | H (V35-V39) + O | U (ownership) | At 06:12 own evidence showed a downstream scorer cannot move a tail fixed upstream; the agent repaired contracts inside the typed line for 11 more hours (note `278b8ed`) |
| 12 | 09-25 06:12-17:23 | Control quantities through feedback and guidance | - | LN integral feedback, 60/50/50 recovery support, head pressure, guidance, audio demand model, packaged bundle | O | L x8 | LN feedback later shown to impose prefix balance (expert 2, own ablation) |
| 13 | 09-25 18:21 | Chord size and arrangement belong to R1 | - | complete-row ownership restored | H (V41-V42) | U (reversal) | Kept |
| 14 | 09-26 01:49-14:05 | Conditions do not change row odds; data has sparse semantic coverage | algebra: an additive condition before an affine head cancels from relative odds | eight conditional-learning interventions, 128-384 updates, 12k-16k trainable weights | O (+V43) | L x8 | Algebra sound (doc-claim); none met its criteria |
| 15 | 09-26 14:05-16:06 | Outcome learning on common generated prefixes | - | full-R1 fit on 47 prefixes -> `a99519c` demo | O | L | Human playtest: long jacks, no breathing |
| 16 | 09-26 18:03-19:45 | A time-horizon player response is needed | - | 4-s selector on sustained attack excess; 12,288-parameter player-state input | H (V46-V52) | new layer | Selector fixed its channel; learned state regressed on held-out cases |
| 17 | 09-27 03:15-08:45 | Flat phrasing = missing audio-history binding | human intuition, marked as such | 7.6M audio pyramid and memory readers; 384 updates on 768 examples | H (V53-V55) | U (added module) | Failed native qualification; dropped |
| 18 | 09-27 10:02-11:08 | Modules do not do what their names claim | expert 2's list | qualification runner, LN feedback ablation, support census | E (V59) | mixed | See section 5 |
| 19 | 09-27 12:07-23:10 | Both H and R1 cause LN fragmentation; the selector is blind to releases | formal doc notes the response is not identifiable from ranked data (line 198) | count-only refits, missing-history views, broader row fit, scoped allocation, release support (128-512 updates each) | H (V60-V67) | L x5 | Each gained somewhere and regressed elsewhere |
| 20 | 09-28 00:26-01:41 | Some H plans make low difficulty impossible | - | H star floor; H audio interaction | O | L | Interaction worsened D2 |
| 21 | 09-28 03:50 | Can the inherited proposal be rehabilitated versus fresh init? | - | clean joint three-arm fit, one pass (row "Unit of evidence") | H (V68-V69) | U (init) | All fail; 30-39% of all LNs last <=80 ms (checked-artifact, section 6) |
| 22 | 09-28 06:36-14:56 | R commits a release before R1 can prefer waiting; response reference blind to short holds | - | joint wait/release (16-80 updates), action-response state, segment mixture (32 updates, all 85 selections chose one code), continuous context (aborted at update 121) | H (V70-V78) | mixed | All failed |
| 23 | 09-28 16:38 | Fresh ordinary expert from scratch | - | 4.78M params, 512 updates over 16 four-second units from 4 songs, 236 s CPU (checked-artifact) | H (V79) | U | Failed native; agent labelled it a capacity diagnostic; 77-154 adjacent head pairs <=10 ms per chart (checked-artifact) |
| 24 | 09-28 17:14 | Elapsed time is not perceived; consequence path unused; response curves undefined | - | event-time response design (document only) | H (V80-V81) + O | proposal | Never tested |

## 4. Locality and turning points, counted

Method (my-class): I listed each point where a recorded failure was followed by a new mechanism
(about 40 points over the 202 commits and the agent notes) and classified it as above.

| | Local | Upstream |
| --- | --- | --- |
| Decision points after a failure (about 40) | about 27 | about 13 |
| of which agent-initiated from its own evidence | about 25 | 1 (typed skeleton, row 11; reversed by the human) |
| of which after a human or expert message | 2-3 | 12 |

Macro turning points (a change of direction that held for half a day or more): 13. Eleven followed a
human message, one followed expert 2, one was the agent's own (row 11, in the direction the human
then reversed). The expert consultation itself was requested by the human. The agent's own
evidence did produce sharp diagnoses (rows 6, 11, 14, 19, 22, 24), but in every case its next action
was a local repair; the upstream move came when the human asked.

How fast conclusions formed, and how long they steered:

| Conclusion | Evidence behind it | What it steered |
| --- | --- | --- |
| "More paired data helped prediction, not rollout" (relay note) | one 2,400-update run, 23 min, one seed; the result note says the curve was still improving | Data and training length never varied again (6 days) |
| "Full-song audio context gives little benefit" | four cells of 1,200 updates, 6-10 min each | Audio context set aside until the human raised it (3 days) |
| "Learned player state does not generalise" | 384 updates on 768 draws | Shift to memory |
| "Memory/attention fails native qualification" | 384 updates on 768 examples | Memory dropped; later cited against "the model is too small" (failure analysis, lines 7, 41, 1459), though line 417 concedes it is "not an isolated capacity comparison" |
| "Fresh model also fails" | 512 updates on 64 s of chart audio | The human chose to reconsider the ordinary-expert architecture and recipe (V84); the notes' key-2 falsifier reads it as "pointing at architecture or information path" |

## 5. The two expert replies and what became of them

Expert 1 (pasted V25, reply to `audio_joint_expert_question.md`, 09-23/24; checked on the mac):

| Recommendation | Fate |
| --- | --- |
| Keep ms hazard + complete row; no fixed beat grid; do not pre-decide LN ends | Adopted and never revisited |
| **First expand paired-song coverage** with the model unchanged; match the budget, record curves, and **if the larger data is still underfit, keep increasing coverage and training**; do not read non-convergence at equal updates as "more data does not help" | Budget-matched run done within hours; the curve was still improving; not continued. **Ignored in substance** |
| Evaluate BOS, early, held and mature prefixes separately; count early failures | Adopted |
| Two-scale audio (about 2 Hz global attention + local TCN) | Adopted in the 2x2; judged on 6-10-minute fits |
| Audio-driven hazard base plus a bounded, decaying history modulation | Adopted 09-24; on 09-28 the agent argued the bound caps exact rhythm (`ordinary_expert_from_scratch_zh.md` section 6) and removed it for the fresh expert, which then produced head doublets. Never tested head to head |
| Contiguous interval training unit with survival and explicit weighting | Adopted |
| Missing-history view, never fabricated prefixes | Adopted 09-27 (`0882315`) |
| Persistent small latent only if needed; warned the decoder may ignore it | Coded 09-24, not fitted; the 09-28 four-state segment mixture collapsed to one code, as warned |
| Frozen segment set judged by blind listening/play, separately for action, music and variation | **Not adopted.** No blind or paired human protocol exists in the lineage; the agent's own Lens reading stood in |
| Only three discriminating experiments; scale data, training or model only on evidence | Roughly 85 interventions followed |

The question itself framed the answer: it stated that R1 is the usable initialisation, that the
Mac has 24 GiB, and that "there is no evidence yet that parameter expansion is necessary"; the
expert then advised against expanding parameters first (checked, both texts).

Expert 2 (pasted V59, 09-27, critique of the architecture account; authorship unverified):

| Recommendation | Fate |
| --- | --- |
| LN ratio feedback imposes prefix balance; ablate it | Adopted within 2 h; bias confirmed; deletion rejected because amount worsened; later recipes dropped it |
| Strict isolation of H from committed actions is a strong assumption; compare with a small action summary | **Not tested**; recorded as a revisable restriction. The isolation was a human instruction (V30) |
| Check that H/R/R1 plus the recovery supports admit a legal future; quantify excluded real relations | Adopted (6.36% of ranked charts excluded; supports relaxed to 60/25/21) |
| Decompose memory into four arms before investing | Not done; memory abandoned |
| A fixed prefix bank is not the policy's own state distribution; cover states reached from BOS during training | **Not adopted**; cited (DAgger) in the 09-28 design, not implemented |
| Native qualification runner before further architecture | Adopted the same hour (`226725c`) |

## 6. What was never questioned

| Assumption | Status in the lineage | Evidence |
| --- | --- | --- |
| R1 as base | Questioned in words (expert question: "R1 is an initialisation, not a frozen architecture"), but R1 weights or its row-decoder architecture were in every model until the 09-28 fresh arm and fresh expert, both human-prompted. R1's own long-form quality was never re-established: its plain stage produced 43% LN heads against an 18.6% reference, the released stage 19% (audit) | doc-claim; checked-git for timing |
| Millisecond clock with no beat grid | Considered and declined on day 1 (row 2); endorsed by expert 1; the human asked for local BPM/phase on 09-28 (V71); a rhythm-hierarchy design (`a6c912f`) was written and never implemented | checked-git |
| Model and data scale, training length | Never a controlled variable. One matched coverage run; largest fit one pass over 8,192 windows; the agent's rule required a small fit to pass before scaling | checked-artifact, checked-git |
| How the corpus was sampled | Partly questioned, to the agent's credit: it found its own balancing raised high-LN exposure from 6.09% to 33.90%, and missing-condition gaps. The choice of tiny subsets (48 groups, 768 examples, 4 songs) was never treated as a cause | doc-claim |
| Small sequential pilots as the method | No lineage source questions it. Instructed by the skill; asked for by the human early, asked against later (V34) | checked-git |
| Strict next-event proposal | Endorsed by expert 1. Relaxed only in selection (4-s private futures) and in the 09-28 segment plans | doc-claim |
| Existing generator as baseline or component | **No source questions it.** Zero mentions of Mapperatorinator, Mug-Diffusion or osuT5 in any lineage document or note; pre-lineage docs already record that "a matched comparison with osuT5 or Mug-Diffusion has not been established" | checked-git (grep over both trees) |
| Defining the response target before building scorers | The formulation says response definitions must come from mapper evidence (positive, contrasting and equivalent cases). The lineage changed the formulation only by the rename (checked-git), built five successive response objects (frontier2, sustained excess, player state, four-scale action response, event-time proposal), and calibrated them to corpus quantiles. Its own 09-27 analysis states that ranked data cannot identify the response (failure analysis, line 198). Recognised, not acted on | checked-git, doc-claim |
| **LN as an open state closed at event opportunities** (my addition) | The duration-object alternative was named on day 1 and never tested. Day-1 evidence: extra release opportunities shorten LNs. Final runs: 30.5% (fresh arm) and 38.9% (inherited arm) of all generated LNs last <=80 ms, and 74-80% of those end exactly at a head time, mostly the very next one (checked-artifact, 18,828 and 27,315 LNs); ranked 3.5-4.5 star charts have 612 HR <=80 ms of 508,936 (doc-claim). The dominant late complaint (problem B) has the signature of this representation | checked-artifact |
| **Human judgment as data** (my addition) | Quality was judged by the agent's own Lens readings and a growing set of observers, each built after the failure it catches. The human's first playtest report carried no seed or file (V47). No held-out, human-judged set existed | doc-claim |

Also checked: head doublets. In the fresh expert, 10-13% of generated head times sit <=10 ms after the
previous one (77-154 per chart; zero with source H). In the clean-joint outputs, where H kept the
bounded modulation, doublets are 0.3-0.4% while short LNs are as common as above. **Doublets
therefore do not explain the short tails in general.** This weakens the relay's carried-over question
"do near-duplicate native head times produce the extreme short tails?" and the notes' hypothesis
`h-scale-and-grid` (b), outside the fresh expert.

## 7. The local-optimum hypothesis

**For it.**
- Autonomous moves were local (about 25 of 27 local changes were the agent's own); upstream moves
  came from the human or an expert (12 of 13).
- Problems A and B (long jacks, fragmented LN) were reported on 5 and 2 days respectively and
  outlived about 40 repairs.
- Day-1 alternatives (beat coordinate, LN duration object) were never tried, although later
  evidence pointed back at them.
- The one direction with a clear, still-improving positive result, more data, was dropped after
  23 minutes.
- In the typed-skeleton episode the agent's own evidence located the ownership problem and it kept
  patching for 11 hours.
- The final documents still propose local repairs (restore a consequence path, fit response curves).

**Against it.**
- The instructions assigned direction to the human: the triage skill and the human's frequent
  steering (84 messages in six days, about 14 per day).
- The human fixed several upstream choices (Mel encoder, H isolation, R1 row ownership, no
  hard-coded rules), so some upstream moves were not the agent's to make.
- Many diagnoses were correct and non-obvious: the transfer identity, additive-condition
  cancellation, LN feedback bias, a planner absent from qualification, an unused consequence path,
  response non-identifiability, and its own sampling distortion.
- It preserved failures and never promoted a model.
- The problem is hard: no validated evaluator, a Mac, and a target that also demanded controls and
  realtime from day 3.

**My call.** The hypothesis holds, with a sharper cause than "patching the nearest module". The whole
search ran at a scale where no architectural claim could be distinguished from under-training.
Every failure was therefore read as a mechanism, and the answer was a mechanism. Locality kept this
invisible because no step ever changed scale, data regime or base together.

- The search was local and human-steered: **high confidence (about 85%)**.
- The late architectural conclusions are confounded by under-training: **moderate-high (about 70%)**.
- More data or training alone would have fixed the charts: **no evidence either way**. The
  short-LN signature suggests a representation problem that scale may not remove.

## 8. Method, problem or instructions

- **Agent's method.** A 20-to-90-minute experiment loop. Conclusions drawn from single short fits.
  The expert's conditional advice was not followed when its condition held. Idle overnight compute.
  No blind human protocol. Repair at the symptom module. An evaluator added per failure.
- **Hard problem.** No defined response target. Human judgments sparse and unseeded. Paired audio
  had to be recovered. A 24 GiB Mac with MPS faults (the continuous-context abort).
- **Instructions.** A skill that prefers fail-fast slices and reserves direction for the human. "Small
  runs first". Controls, scoped ranges, realtime and demo requested before any ordinary playable
  proposal existed (V39, V44). Module ownership and "no hard-coded rules" fixed by the human. A
  goal-continuation loop that rewards "progress" each turn.

## 9. Inconsistencies found

- Relay note: coverage "did not solve rollout stability". Result note: held-out NLL -15.4%, the
  planned threshold passed, the curve still improving.
- The failure analysis uses the 384-update memory fit against "the model is too small", and in the
  same document calls it not a capacity comparison.
- Feedback index, key 2 falsifier: it reads the fresh expert (64 s of chart audio) as evidence about
  architecture. The agent's own document says it is not.
- Bounded H modulation: adopted from expert 1, then declared rhythm-limiting four days later without
  a test.
- Five response objects were built while the formulation's response definition stayed untouched.

## 10. Questions only the human can settle

1. Did "small runs first" (V11) still apply after the first day? Did "large steps" (V34) mean more
   compute and scale, or bolder architecture?
2. Should the agent initiate direction changes, or is the triage skill's rule (the human owns
   direction) intended?
3. Will the response target be defined from mapper evidence, as the formulation says, before new
   scorers are built? Or are corpus-calibrated proxies acceptable?
4. Should an existing generator run on the same songs as a baseline?
5. Given its 64-s training set, does the fresh-expert result still carry the weight V84 gave it?
6. Is an LN represented as an object with a duration compatible with the incremental-release
   protocol you allowed?
7. Were the idle overnight windows your pause or the agent's?

## 11. Not verified

- Human-message timing relies on the feedback index. Transcripts were out of scope.
- Locality and trigger counts are my classification from commit subjects, notes and index times.
  Another reader could move a few points between columns; the direction would not change.
- Most numbers in sections 3-5 are doc-claims. I re-derived only those marked checked-artifact.
- Second expert's authorship and source document.
- Whether any overnight run wrote outside `ensomi-model/artifacts/` or was deleted by the 09-30
  cleanup.
- Whether scale, a beat coordinate or LN-as-duration would help. I tested none; I checked only that
  they were never tested.
