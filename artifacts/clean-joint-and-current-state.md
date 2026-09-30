# Completed fits, the fresh expert and the actual stopping point

The latest state is **no qualified playable model and no identified research
training process still running**. Two different fresh-initialization efforts
must not be collapsed into one: the three-arm clean joint comparison on the
primary branch, and the later four-song ordinary-expert capacity probe on
`codex/release-calibration`.

## The three-arm comparison finished beyond the primary document

`ef42095` added direct learned LN conditioning without the analytic ratio tilt.
The common new recipe jointly trained audio/H/R/R1 from inherited, early and
fresh initializations, distinguished natural missing-control draws from balanced
known-control and human-annotation draws, and disabled amount feedback/allocation.
All three models had 4,675,633 parameters. This compares initialization under
a new shared recipe; it does not isolate every recipe change from old results.

Preparation produced 8,192 factual windows from 3,736 distinct TRAIN charts,
plus 22 validation identities, with six support rejections. Only natural draws
deliberately hid numeric controls. High-LN share among hidden-LN draws fell
from the prior recipe's 30.41% to 7.20%, while subsequent analysis still found
missing style-known/LN-unknown exposure. Both observations remain relevant.

At primary HEAD `4a8a478`, the
[product account](/Users/l/projects/ensomi-model/docs/research/clean_joint_proposal_learning.md)
ends at step 512, when each arm failed seven of eight numerical cases. The
[owning Agent Note](/Users/l/projects/ensomi-model-agent-notes/artifacts/agent-notes/proposed/2026-09-28-clean-joint-proposal-learning.md)
at notes commit `4afc915c5bb15e4ce371a534b012c3332cbae939` records later milestones:

| Milestone | Inherited | Early | Fresh | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 512 updates, failures / 8 | 7 | 7 | 7 | Fresh reduced pressure but compressed D2/D6 response; no selection |
| 2,048 updates, failures / 8 | 4 | 6 | 7 | Intermediate calibration gains coexisted with Stream/LN failures |
| 4,096 updates, failures / 28 | 19 | 18 | 17 | All 84 exports complete; every candidate failed; full final semantic review pending |

The supervisor completed normally in 16,957.51 s, with all 8,192 draws matching
the frozen ledger. It was not waiting at 512/2,048, interrupted, or silently
promoted. Final macro row NLL was 1.61091/1.60646/1.84232 and H NLL/s
30.11214/30.14575/32.76122. Those values do not rank playability.

Recovery independently read the final supervisor and three qualification
receipts, counted their failure lists, and hashed the actual four final model
files below. No tensor deserialization or new generation was required.
All paths in this table are under
`/Users/l/projects/ensomi-model/artifacts/joint-audio/`.

| File | SHA-256 verified during recovery |
| --- | --- |
| `20260928-clean-joint-proposal-v1/supervisor-v1/result.json` | `0368262cb45ea335122c0c9b16722f186b8080ef48afca7aa817727baf399729` |
| `20260928-clean-joint-proposal-v1/fit-v1/4064-4096/inherited.pt` | `eaa17a02606ee08542a20a33547bd35b5aaa02964cf6db52ab5e5aa54d74427f` |
| `20260928-clean-joint-proposal-v1/fit-v1/4064-4096/early.pt` | `1138d679fbff2f30f8a3e1add3b5f4ac2f7cac1b7b669595cd529de2fefa06e5` |
| `20260928-clean-joint-proposal-v1/fit-v1/4064-4096/fresh.pt` | `2eacf8328135dcd1b729a096522a5dc1569a929cda9fe83c61805900330bd559` |
| `20260928-clean-joint-proposal-v1/native-4096-inherited/result.json` | `a08e03cdceb39c96c03c658fe473ea7a38222d0d36bf9335c0fc804957fd8a9b` |
| `20260928-clean-joint-proposal-v1/native-4096-early/result.json` | `784d68c9564828a89d2121d10dabe616baf875ee37254756eb90833a67484341` |
| `20260928-clean-joint-proposal-v1/native-4096-fresh/result.json` | `9602996936755a4fa5251456b8dbb7bb66960ddf0f3a8f2047658f40f49c0fa4` |

Each final qualification receipt says execution complete, candidate failed,
promotion false and semantic review pending. The direct-mode learning source
was `ef42095`; final evaluator source was `4a8a478`. Later branch documents
have more results than the primary document. Treating primary HEAD as the
whole historical state would lose substantial completed work.

## Fresh ordinary expert: learned a small sample, failed native qualification

The human's later request authorized a new ordinary-four-star recipe with
fresh learned modules and eventual fusion as a possibility. It did not narrow
the ultimate 2–6★/style/control/realtime goal to four-song memorization.
[Private, local request](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1050).

After the [continuous-context fit aborted](ordinary-patterns-and-frontier.md),
the ordinary scout split 1,973 ranked 3.5–4.5★ charts by connected song/audio
components before selection: 1,602 train, 361 validation and ten reserved.
Two failed scouts were preserved: a missing cached chart and a mistaken reused
array in its fallback. The corrected third scout included all sources.
The provisional ordinary teaching subset was not the complete learned population.

Four inspected anchors—Dawn, Sulyvahn, mumei and Kill The Beat—provided sixteen
four-second fit units covering entry, organization and lower activity. The
recipe retained each whole-song rating across quiet/dense units, instead of
relabeling each crop with local strain. This tests a different conditional
meaning from prior mixed local/whole targets; it does not prove that earlier
labels caused all flat-pressure behavior.

`d6eba23` built a 4,782,754-parameter fresh audio/H/R1 model with direct
continuous plan context and local history retained across plan boundaries.
Hidden weights were randomly initialized; only appropriate residual outputs
were zeroed. Normalization came from the new sources. No old actor or audio
weights were imported. A sixteen-update CPU/MPS comparison took 8.91/13.99 s
with similar losses, so the small capacity fit used CPU. This did not establish
that the prior variable-shape MPS assertion was fixed.

The fresh 512-update fit completed in 235.73 s. Fit-unit H NLL/s fell
31.01186→10.72888 and R/R1 23.07761→2.29225. These were deliberately tiny
capacity data. Twelve completions then compared real prefix+source H,
BOS+source H and BOS+native H for each song:

- All eight BOS outputs failed the 40-ms LN-burden reference while passing
  the broader 80-ms reference. Source-H/native-H short-tail counts were
  Dawn 1/31, Sulyvahn 9/47, mumei 10/68 and Kill The Beat 9/45.
- Native stars were 5.2086, 4.6092, 3.8569 and 4.5527. mumei passed star and
  LN-amount checks but still had 68 extreme short tails.
- Thirty generated Lens pages were historically read. In real-prefix/source-H
  mumei, a persistent held role with thirteen other-finger TAP groups was
  learned; BOS/source-H retained only three, and native H produced 5–6-ms LNs.
  This is useful conditional capacity with failed autonomous behavior.
- <a id="o-native-h-doublets"></a>**Observation, fresh expert at 512 updates, agent inspection.** Native H contained many <=10-ms adjacent pairs absent from the four sources.
  Of 191 short tails, 155 ended at the immediately following H. Onset-nearness
  could remain high while duplicate H events appeared. Source-H substitution
  changes later state/RNG too; this is not an objectwise causal percentage.

The local owner is `20260928-ordinary-scratch-v1`. Recovery verified:

| File under that owner | SHA-256 |
| --- | --- |
| `capacity-fit-v1/step-512.pt` | `f7b2037ce5338c593b4f3b7ff01e3c5d20a1cea79b7d3f3461043e6457898167` |
| `capacity-native-512-v1/result.json` | `a20acd1e0a65906add77632f97962ca2bee2ee198587c293c62b00cb3ce808f8` |
| `capacity-review-512-v1.json` | `09021a8d46876d2c415a9b7d76646ef99f23c375d0dd0815ada5468a5220e2be` |
| `capacity-interpretation-512-v1.json` | `0eb17d5854544865f80b7a83b6b1c0450d98453e570d0f544810fbb0f6a41a5e` |
| `time-path-probe-512-v1.json` | `453f1e0ace8f84e2f8159e6de7a573fa87913479aa71a830a17e80be64793c01` |
| `corpus-time-gaps-v1.json` | `5d2f88537935f55ad18accb2d1c4724e87db0dfaebf6a480ba274b24099300e1` |

The review receipt explicitly records agent inspection, no audio audition and
no human playtest. No checkpoint qualified or entered fusion.

## The last elapsed-time hypothesis is specific and still open

The human suspected inadequate perception of the next action's elapsed real
time and asked about response curves grounded in hand constraints and corpus.
This was a hypothesis/request, not an established cause. [Private, local
sources](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1062)
and [curve question](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m1065).

<a id="o-unused-consequence-path"></a>**Observation, source read at `96f84fd` and a four-window ablation.** The source audit found existing time inputs in H and R1, and exponential
response scales 250/1,000/4,000/16,000 ms with reciprocal-gap impulses. The
segment decoder nevertheless deletes `row_consequence` and accepts `local`/
`timing` candidate features without using them. Recovery re-read that function
at `96f84fd`; the saved four-window ablation reports zero probability change
when those features are erased. Exact-clock and future-preview lesions do
change probabilities. Time is therefore not wholly absent; one candidate
consequence path is missing.

The response state retains exact occupancy but has no tonic held input to its
decaying impulse bank. Its engineered curves and corpus limits are not measured
individual physiology. The 1,602-chart census has zero HH≤40-ms intervals among
2,766,890 attacks, but HR/RH have rare positive occurrences. A 25-ms isolated
LN can still receive zero work under the current response reference. Acute
transition and accumulated regular workload should not be conflated.

The latest proposal is explicit time advance before applying a candidate,
separate acute HH/HR/RH responses, multiscale finger/hand accumulation and
ordered coordination/holding state, with actual-future frontier evaluation.
It is documented in [event-time response](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/event_time_gameplay_response_zh.md)
and [fresh-expert results](/Users/l/.codex/worktrees/release-calibration/ensomi-model/docs/research/ordinary_expert_from_scratch_zh.md).
Restoring one path, training response curves, hierarchical H and fusion remain
unverified. A fresh session should reassess those proposals against these
counterexamples before selecting a bounded next experiment.

## Checkout and process state at recovery

On 2026-09-29 around 15:09 Asia/Shanghai, the historical chat was idle and
process inspection found no matching training/supervisor/`caffeinate` job;
only Python language-server processes and the recovery probe were present.
Old tool-session IDs in notes are historical, not live obligations. No process
was killed or relaunched. No subagent was spawned in this recovery.

| Checkout | Confirmed revision and disposition |
| --- | --- |
| Primary `/Users/l/projects/ensomi-model` | `codex/audio-skeleton`, `4a8a47805a8fdbcb4fa27802063c55c649d41020`, 15 ahead of its recorded upstream; existing modified `AGENTS.md`, untracked relay-skill link and architecture walkthrough preserved |
| Release calibration worktree | `codex/release-calibration`, `96f84fd32218e39ff809c651d73e6edbd39a3500`, clean; later code/docs remain unmerged |
| Realtime benchmark worktree | `codex/stream-generation-benchmark`, `4ec631ef71d1ca71e36e5c383d4997efffdebb05`; separate loader/policy integration remains unqualified |
| Existing Agent Notes worktree | `agent-notes`, `4afc915c5bb15e4ce371a534b012c3332cbae939`; read as evidence and left unchanged |

The new relay materials do not restart the old automatic research goal. Source
and tests were read selectively; no old experiment or test suite was rerun.
Large generated assets and the ignored private human-input file remain local.
Their hashes establish identity where checked, not a new scientific replication.
