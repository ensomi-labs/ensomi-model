# Agent Note: Full-row learning with factual missing-history observations

Note ID: 2026-09-27-full-row-history-observation-learning
Status: proposed
Kind: research
Created: 2026-09-27
Updated: 2026-09-27
Product revision: 0882315ef23097e44e031707abd382d971b8c82c
Scope: Matched full-R1 factual learning, optional content-observation masking, native organization and style guards
Related: 2026-09-27-native-prefix-weight-state-decomposition, 2026-09-27-contextual-ln-count-conditioning

## Mechanism and alternatives

The count refit cannot change within-count routing at a fixed history, while
changed actual histories dominate its LN-law difference in four matched LN
cases. This motivates two separate changes: permit the actual row/history/layout
owners to learn, then test whether a factual missing-observation objective helps
their conditioning remain useful beyond detailed teacher content.

The closest analogue is modality dropout: ModDrop randomly omits input channels
while retaining sample identity and targets (Neverova et al.,
https://arxiv.org/abs/1501.00102). This adapts that observation principle to
learned chart-content context; it is not a novel algorithm or evidence that
gesture results transfer to native chart generation. The distinction from
replacing the past with sampled actions follows the consistency concern in
https://arxiv.org/abs/1511.05101. Exact state, including last-row facts, remains
visible, so this does not remove all historical information or guarantee that
audio becomes dominant.

Under a randomly selected visibility mask, supervision remains a conditional
likelihood of the same factual world with less observation. Generated-history
outcome learning remains a separate problem. Healthy LN continuity, jacks and
motifs may rely on history, so native/source guards must be able to reject this
branch if it flattens them. Full audio/H/R training and richer relationship
representations remain available; freezing them here isolates row-side learning.

## Experiment Card: full-row-factual-views-v1

Revision: 1
Accepted revision: none
Execution authority: standing user research, training, compute and local-commit
goal. No inferred exact-Card acceptance, model adoption or remote push.

Implementation source 0882315ef23097e44e031707abd382d971b8c82c; ordinary native
calls preserve the pre-mask probability law. Reuse the frozen 256 factual draws
and 22 validation charts from source plan
a766382082fa7e0b8674f88d9c7be128b41cf16d1fa7e881beb367336dbaddf4.
Both arms initialize actor128 SHA
364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3,
use reference_tilt, saved 60/50/50 recovery, default recovery preference and LN
feedback OFF. No new source draw or target/control change. This compares against
the previous count-only result but has its own matched full-R1 control arm.

Train row temporal, exact/fuse, audio-residual/context-condition projections,
joint layout, route/release residuals, row control/preview, composition,
layout modulation and row consequence. Freeze complete audio encoder and every
H/R neural parameter. Assert the frozen tensor set at saves. Actual R outcomes
can change through the model's own row decisions; freezing its network does not
make the whole release process independent of R1.

Baseline: every genuine source content observation is visible. Intervention:
exactly 64 of 256 windows, chosen without inspecting outcomes by a fixed seed,
hide their learned content observation at non-BOS row queries using TRUNCATED.
Actual BOS remains BOS. Physical replay, last-row facts, LN occupation/ages,
audio, H preview, controls, masks and action targets remain unchanged. H/R factors
never receive this visibility flag. All evaluation uses full ordinary history.
This is one observation-objective intervention, not a synthetic source suffix.

Run 128 updates with batch two, mean deployed source-row NLL per second, AdamW,
weight decay 1e-4 and gradient norm cap 1. Composition/row-control/preview/layout
modulation learning rate 1e-4; other trainable R1 parameters 3e-5. Keep the same
logical draws for both arms and record actual masked rows/gradients/losses.
No optimizer-state sharing between arms. Seed 279812. Preserve a visibility list
and parameter ownership in the frozen plan before fitting.

Use MPS, CPU one thread, serial fresh processes of 16 updates to bound the
observed heterogeneous-shape memory growth. Each process restores both model
and optimizer states from the preceding completed segment. Save both arms and
their exact draw index. Do not restart from an inferred live-state file: only
advance after the previous worker is confirmed terminal and its completion
receipt passes. At most 3600 seconds total fit, 600 per segment, 12 GiB task
footprint. A guard/failure preserves the segment; no automatic parameter or data
change. Stop on STOP, source/hash drift, nonfinite loss/gradient, altered frozen
weights, or memory/time limits. Final source evaluation uses both full and
missing views as separate diagnostics, not a merged NLL.

Native evaluation retains the eleven-case plan
48379544b1b012fec14c8223a16f23275bc476a967a798d0c2c4cb6e6389d863
and adds three prominent-style guards (Jack, Tech, Trill) on the same verified
Zenithfall audio, stars 4, unspecified LN, seeds 279701/279702/279703. The added
guards require an unfitted baseline and both fitted arms. Existing eleven
baseline outputs may be reused with pinned row hashes because ordinary law and
weights are unchanged. All fitted cases start from BOS, use complete audio,
feedback off, and preserve scoped publication. No future source heads or rows.

Primary quality decision: independent Lens/source review must find a repeated
improvement in at least two LN-rich songs across both seeds, without judged
organization loss in the third or in the three style guards. Numeric guards:
no new under-20-ms attack or publication failure; no additional declared LN
amount failures versus the initial actor in the original nine LN/control cases;
star absolute-error macro mean may not exceed the matched full-R1 control by
.1; Stream excess may not exceed that control by .005 seconds per case. Keep
each scope/seed/style distinct and inspect witnesses. Source full-view NLL may
not exceed baseline full-R1 NLL by .05 nats/row. Lower likelihood, longer median
holds, lower head counts or balanced columns alone cannot qualify a model.

The source/learning checks discriminate observation robustness, not calibrated
player-response semantics. If both full-row arms improve similarly, credit
decision capacity/source learning rather than masking. If masking improves only
its auxiliary condition, or hurts native motifs/control, do not scale it. If all
native quality remains bad, record REFINE and inspect actual input/state and
trajectory targets rather than promoting on NLL. Unaccepted exploratory evidence
cannot be called SUPPORTED.

Fresh owner artifacts/joint-audio/20260927-full-row-history-views-v1; freeze its
plan, source-data identity and scripts before launching. Native CPU one-thread
evaluation has a separate 2400-second total, 300 seconds per case, 8 GiB footprint.
The realtime benchmark worktree is unchanged; native observations do not replace
its 30-row/eight-second readiness contract. The full playable-system goal remains
active and unproven.
