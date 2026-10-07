# R2 v1: deviations from the runnable v1 spec

The spec is section "Runnable v1 spec" of the R2 design review (relay notes,
`artifacts/r2-design-review-fable.md`), with `artifacts/r2-ml-design-20261003/design.md`
for the parts it keeps and the main-thread overrides of 2026-10-03. Each entry gives the
nearest faithful version that was built and the reason.

1. **Song length from `soundfile.info`, not ffprobe (main-thread override).** The mac has
   no ffprobe. T is the container duration read by libsndfile; missing or unreadable audio
   excludes the chart with its reason. No API length or hit-object-derived length is used.

2. **Release pointer reuses R1's mathematics, not the `EndpointPointer` class.** The v1
   query includes per-candidate placement relations (28) and per-candidate condition frames
   (row, candidate and birth roles), so the query differs per candidate. R1's
   `EndpointPointer.log_prob` takes one context per factor and cannot express that. `R2Model`
   keeps the same score (dot / sqrt(128) + linear candidate bias), the same exact streaming
   normaliser over ragged candidate sets with activation checkpointing in blocks of 8192
   pairs, and the same CPU float64 Gumbel-maximum sampler. The first query layer is split
   into a per-factor part and a per-candidate relation part (`pointer_in` + `pointer_relation`),
   which is the same affine map as one layer over the concatenation.

3. **The pointer query also receives a one-hot of the releasing lane in the oriented frame
   (4 inputs).** Without it, two lanes releasing in the same decision with equal codes and
   equal lane clocks are indistinguishable to the directed pointer. The directed pointer is
   only used inside the two-orientation mixture, so the mirror identity is unaffected (test 2).

4. **Cache index and star labels are not plain JSON files.** The per-chart index is
   `index.parquet` (its SHA-256 is in `summary.json`) and the labels are
   `labels/star.json.gz`, because JSON files under `artifacts/` are mirrored to the control
   plane and files above 4 MB stall that mirror. Content is as specified.

5. **Condition frame counters.** For every role (row, candidate, birth) the committed counts
   are the head objects of decisions before k whose head time lies in the interval, and the
   remaining count is the uncommitted input head rows in it; only offsets and progress depend
   on the role's own time. The presence bit of a kind is set when the track has any interval
   of that kind, active or not; values, bounds and counters of inactive intervals are zero.

6. **Token conditioner roles.** Token mode reads the whole announced track, one token per
   interval (18 inputs: kind one-hot, value, start and end offsets in ms and beats, active,
   progress, committed head and LN counts, LN ratio, remaining rows, started) plus a null
   token. Row queries use offsets from t_k; pointer queries use offsets from the candidate
   time. There is no separate birth-role token in this form.

7. **Generation uses the TCN online cache** (`FiniteTemporal.append`, fixed weights) rather
   than re-running the dense TCN over the prefix at each decision. R1 established the
   dense/cache equality; landmarks are the cached outputs at token indices divisible by 64.

8. **Supervisor and the resource guard.** The trainer stops on `ResourceGuard` (exit 4,
   after a safe checkpoint), as spec section 7 says. The supervisor, as the brief asks,
   resumes after any non-zero exit except the NaN-limit stop (exit 3), at most five times in
   six hours, so a resource stop is followed by a restart from the latest checkpoint in a fresh
   process. A restart that reaches a new regular checkpoint clears the count; the supervisor
   reads that from the checkpoint directory, since a resource stop points `latest.json` at its
   safe checkpoint.

9. **Checkpoint cadence versus measured throughput (not changed, flagged).** The spec's
   250k-exposure checkpoint and per-checkpoint evaluation assume a few hundred decisions per
   second. Measured on the mac (FiLM, landmarks, 8 levels, 4 threads): about 3,700 head
   decisions/s on CPU and about 700 on MPS (job `20261003-182254-r2-pilot-both`, 48 windows).
   At the CPU rate a 250k checkpoint comes every ~70 s; each one costs ~19 s of evaluation
   (fit_dev manifest ~2 s, twelve free-runs ~16 s) and 28 MB of disk, i.e. about 25 %
   overhead and ~1,000 checkpoints (~30 GB) in 20 hours. `checkpoint_every` is a config value;
   the tuning agent should set it from the pilot and record the change.

10. **The TCN is activation-checkpointed per block in training** (`R2Config.checkpoint_temporal`,
    on by default). A window's history is the whole chart prefix up to its end, so the
    activations kept for backward grow with the prefix: about 0.14 GiB per 1,000 positions,
    4.8 GiB for a window ending the longest fit_train chart (K = 43,661). Those transient peaks
    tripped the 2 GiB RSS growth guard eight times in run `r2-phaseN-20261006`. Recomputing each
    block in backward gives the same values and gradients (`test_train_step.py`), keeps
    1.9 GiB at that length, and costs one extra forward of the blocks. Runs launched before
    the change keep their frozen code.

11. **Optional whole-song LN-level input in phase N.** The model can receive a chart's LN
    share before its first generated decision. Three shared query channels encode known,
    level, and a logit clipped at 1e-6; a separate zero-initialized linear reader preserves
    the phase-N checkpoint's function at warm start. The default off path keeps the original
    feature and parameter shapes. Training drops the source level independently for 30% of
    windows by default. Generation can use unknown, oracle, a fixed value, or a fit_train
    empirical prior conditioned on star band and within-band head-row-density tercile.
    Selection uses prior levels; oracle and unknown evaluations show the effect of supplied
    source information. This input addresses the missing whole-song LN commitment at BOS.

12. **Guard (iv) v2 and BOS drift guard (v).** Short holds use strict duration <60 ms.
    Defect counts and denominators exclude holds closed inside copied prefixes. Each defect
    count is compared with 1.25 times the expected count from fit_train rates in the source
    star band, weighted by generated hold count. This replaces the fixed 0.5% short-hold cap
    and the pooled source-panel near-head rate; original metrics and guard `iv_v1` remain
    nonbinding diagnostics. Guard (v) additionally bounds the absolute mean BOS LN-share
    drift from the first to last song third at 0.05. BOS/source correlation, SD ratio, and
    dense-row LN-birth rates are reported without binding selection.
