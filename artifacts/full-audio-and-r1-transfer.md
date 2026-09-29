# Full audio, generated history and the identity of R1

The early failures did not invalidate the chain factorization of event time
followed by a complete action row. Source-time row likelihood need not
differentiate through sampled event times to be a valid joint likelihood.
The harder question was reliability on the model's own evolving histories.
The [expert question](/Users/l/projects/ensomi-model/docs/research/audio_joint_expert_question.md)
at `10a16fe` and imported advice motivated wider paired coverage, a full-song
audio branch and a controlled history path. Imported advice is preserved as
[external text, private, local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#imported-expert-20260924),
not attributed to the human as original wording.

## Wider coverage and matched information paths

Expanding 48 song groups/121 arrangements to 240/585 at the same 38,400 logical
examples improved additional-24 validation query NLL from 6.4852 to 5.4883,
but left three below-30-head outputs among 84 native samples. More paired data
helped conditional prediction and did not solve rollout stability.

The subsequent local/global × original/bounded-history study held a new
interval recipe and exposure fixed: 4,800 intervals, 37,300,370 ms and
248,836 event rows per arm. All 336 outputs completed, exceeded 30 heads and
ended after 85% of the song. No arm met the planned 3% additional-panel NLL
improvement. Same-key <=20-ms pairs were 17/35/11/4 respectively. The
global/bounded arm was less active and more LN-heavy, so the lower count
required composition review.

Full-song conditioning influenced predictions: additional-panel NLL/s was
37.7294 with correct coarse context, 41.5211 when zeroed, and 37.7460 after a
half-song shift. This supports context use but weakly identifies temporal
musical relations. It does not establish that attention or full audio alone
solved phrasing. Details and execution identities are in
[audio context/history](/Users/l/projects/ensomi-model/docs/research/audio_context_history.md)
and [v2 candidate evidence](/Users/l/projects/ensomi-model/docs/research/audio_joint_playtest_v2.md),
especially product `b5c5ee3` and `98013ee`.

Adding the fixed 27-ms prior to that candidate removed observed <=20-ms pairs
on its 84-output panel, with 80 timed-row sequences unchanged. Follow-up style
contexts still exposed major gaps: native death-piano and Prom-Queen passages
were strongly LN-based where their real-prefix predictions were nearly TAP-only.
Forty-eight observed-prefix continuations demonstrated useful conditional TAP
capacity, yet early reference priming did not consistently prevent later drift.
Source-prefix performance therefore did not solve audio-only arrangement choice.

Pairing repair recovered 490 chart copies lacking adjacent audio via verified
byte-identical source aliases; thirty TRAIN alternatives expanded the chosen
corpus to 615 arrangements. This was a data-location correction with pinned
identities, not permission to merge alternatives or claim new held-out evidence.

## The transferred model was not released R1

The [transfer audit](/Users/l/projects/ensomi-model/docs/research/r1_transfer_stability_audit.md)
at `4b22613` and `0848d8d` is decisive about identity:

- Released `r1-restored-6.75m` had 3,084,432 parameters; 6.75M denotes exposure.
- The audio model copied 2,444,688 parameters, omitted 523,776 in seed memory,
  landmark memory and row consequences, and discarded 115,968 projection weights.
- Passing 6.5M release and 6.75M response checkpoints through the same transfer
  produced bit-identical initial tensors. The last stage trained only the
  omitted `frontier2` component, so it had zero direct parameter effect here.

This does not give a percentage attribution for bad audio generations. The
conditions also changed from supplied R/H timing plus an observed seed to
learned timing and BOS. The actual released lineage improved mean LN fraction
from 43.12% to 19.25% in sixteen fixed-condition outputs; its final local
correction mainly reduced release-to-head <30-ms counts, 67 to seven, while
head-to-head counts changed two to five. Those are distinct actions.
Its ledger still required long-form review.

A sixteen-output ablation of seed/memory neural readouts did not support uniform
deterioration, though regional arrangements changed sharply. A later matched
plain/memory/release initialization comparison completed only 246 of 276
attempts within its global budget: 245 complete, one partial, thirty unattempted.
Release and memory each covered their panels; plain did not. There was no
complete three-arm quality winner. Preserve this incompleteness rather than
turning near-equal validation NLL into a selected model.

## Ownership corrected by human feedback

The human required direct audio to rows, proposed separate skeleton history,
then explicitly retained necessary LN occupancy feedback. The older flat
timing residual still changed when only past TAP layout changed, even with
audio, skeleton and LN state fixed. At one held prefix, next-50-ms event
probability moved from 61.36% to 73.08%. This demonstrates an information-path
violation under that proposed contract, not causation of every quality failure.

The resulting H/R/R1 prototype separates head planning, LN-aware release timing
and audio-conditioned complete rows, and restores candidate consequences.
`frontier2` means a particular local candidate-energy implementation; it is not
the full player-response frontier in V3. Product `b5a664c`, `e32e6e1` and
`e04b349` document and implement this transition. See
[the information contract](/Users/l/projects/ensomi-model/docs/research/audio_skeleton_information_contract.md)
and [planned continuation](/Users/l/projects/ensomi-model/docs/research/planned_audio_continuation.md).
The later critique reopened strict timing-preference isolation as a modeling
choice; this was not an immutable V3 law.

Human sources, with selected agent statements intact: [direct row audio,
private, local](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0180),
[skeleton simplification](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0182),
[LN qualification](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0184),
and [frontier query](private/human-inputs/01a0ebe8-d368-7ba3-9d56-48da0524c5b6.md#m0188).

The four-state persistent-intent prototype (`6031291`) is implemented, with an
audio prior and reference-only training recognition network. The recovery found
no completed quality result selecting it in its owning product document;
lineage/ownership investigations superseded treating it as the immediate remedy.
Do not conflate it with later arrangement-profile or segment-latent experiments.
Its [own account](/Users/l/projects/ensomi-model/docs/research/audio_persistent_intent.md)
remains an implementation/open-experiment record.

The next practical evidence is [buffered playback and private continuation
planning](playback-and-planning.md): ample measured runtime headroom moved the
bottleneck toward arrangement quality, rather than automatically justifying
speculative decoding.
