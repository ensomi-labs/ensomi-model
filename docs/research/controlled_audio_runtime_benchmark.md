# H/R/R1 generation latency on the M5

On an Apple M5 MacBook Air with 24 GiB RAM, the controlled H/R skeleton and
complete-row R1 runtime has substantial compute margin for local buffered
playback. The aligned checkpoint's 90 full-song runs produced 3,060 publication
windows: eight seconds of chart coverage took 0.300 s at p95 and at most 0.474 s.
Resident-model readiness from cached Mel was 0.374 s at the median and 0.589 s
at the maximum. One-second publication units took 44 ms at p95 in the separate
paired window panel, with unchanged generated rows.

These are generator and preprocessing measurements. They do not establish
network delivery, Swift rendering, semantic style accuracy or musical
playability. The current client needs an explicit producer-watermark path before
these compute margins can become a reliable downstream streaming guarantee.

## Models, inputs and environment

The model source baseline is `e48e4ba210a51951d530e6ff3989f41ec9794455`.
The behavior-neutral benchmark entrypoint and tests are committed at
`cf4de217367cfafa42d35678e5f4348d1b090757`. The model is the
[controlled H/R/R1 runtime](controlled_audio_continuation.md), with full-song
audio, 16-head lookahead, native millisecond events and HH/RH/HR recovery
intervals of 60/50/50 ms. No source chart, supplied H plan or seed rows enter
these runs. Native generation begins at BOS.

| Label | Checkpoint | SHA-256 |
| --- | --- | --- |
| core | Retained row-owned, frontier-2500 | `0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8` |
| modulated | Conditional layout modulation, step 128 | `b8aecd3e1f43339d2c1e3aff6d245a41e32a12008ddc20fd47e9eccb99544546` |
| aligned | Scoped style discrimination, aligned step 400 | `a867547cf39d1284ca66daaca83e7f58f70851c3e8706c9971a1e7a665790214` |

The later candidates have unresolved quality and control failures documented in
[condition/history interactions](row_condition_interactions.md) and
[scoped style discrimination](scoped_style_discrimination.md). Timing this
candidate does not promote it over the retained core model.

The primary inputs are Take (144.236 s), Hysteric Night Girl (301.008 s), and
Operation: Zenithfall (357.796 s). Yomi yori (498.989 s) supplies a separate long
case. The panel pins each audio and canonical Mel file by SHA-256. Its own
SHA-256 is `1f089af145b45c232c0ab95de77f03cee8b6a6a1b319f2335ac38f59eb79051b`.

Runtime: macOS 27.0, build 26A428; Python 3.10.20; PyTorch 2.11.0;
NumPy 1.26.4; SciPy 1.15.3; FFmpeg 8.1; float32 model tensors. CPU intra-op and
inter-op threads are one except the explicit thread comparison. MPS fallback
and fast-math environment overrides were unset. Runs were serial on AC power.
OS file caches were not evicted, and desktop activity, frequency and temperature
were not controlled. Small differences between separately run model panels are
not causal estimates of an architectural cost.

The measured suite contains 252 primary generation records: 190 complete songs
and 62 explicitly bounded prefixes. All 190 complete songs passed independent
row verification and `.osu` export/reparse equality. Prefix tests preserve open
holds and never treat the prefix endpoint as audio termination. First-use warmup
passes and synchronized diagnostic reruns are separate from these counts.

## Measurement boundaries

- **Fresh process readiness:** parent process launch through the child's first
  ready record, including imports, model load, audio decode, full Mel and
  generation. It excludes a websocket server, input rehashing and chart export.
- **Resident readiness:** already loaded model and in-memory canonical Mel
  through a published boundary with at least 30 physical rows and 8 s of
  settled coverage. It includes complete-song audio encoding. This is not the
  client's separate clean-LN-entry readiness test.
- **Window service:** elapsed generation needed to advance the settled clock by
  the configured publication unit. The final window and the live-control
  boundary can be shorter. Silent windows still advance coverage. Export,
  reparsing and result-file writes occur outside this measurement.
- **RTF:** generation seconds, including encoding, divided by generated audio
  seconds. Lower is faster; RTF .025 means 40 audio seconds per wall second.
- **Stage diagnostics:** a separate identical-seed rollout, synchronizing each
  named stage. Its row hash must equal the ordinary run. Ordinary MPS timing
  synchronizes at publication and first-thirty-row boundaries, not every stage.
- **Memory:** sampled process RSS at initialization and run endpoints, plus MPS
  active/driver counters when applicable. These are not peak allocation
  measurements, and the overlapping counters are not added together.

Each primary condition uses three seeds on each of three audios. Seed assignment
is `261900 + 1000 * selected_case_index + repeat`; warm jobs are shuffled within
one checkpoint/device process. The device and publication comparisons use the
same Take audio and matching seeds. Quantiles are descriptive; correlated
windows are not independent population samples or an OS latency bound.

## Controls and complete-song throughput

The base request is difficulty 3 and LN-head fraction .2, with styles unspecified.
Low/high difficulty replace only the difficulty with 1.5/6. LN-high replaces
only the fraction with .7. Style rows request that single trained attribute at
prominent strength; other style attributes remain unspecified. Unspecified omits
all controls. The live override is submitted at published coverage 63,999 ms:
difficulty 4.5, LN fraction .6 and prominent trill on [64,96) seconds, after
which the base request resumes. Published rows and active holds remain intact.
These values are requests, not claims about the realized musical arrangement.

Each row below contains nine full generations. Window p95 pools that row's
three audios and three seeds.

| Request | Ready median / max (s) | 8-s service p95 / max (s) | Median RTF |
| --- | --- | --- | --- |
| Unspecified | 0.361 / 0.399 | 0.354 / 0.474 | 0.025 |
| D=3, LN=.2 | 0.362 / 0.438 | 0.242 / 0.376 | 0.020 |
| D=1.5 | 0.385 / 0.394 | 0.200 / 0.246 | 0.017 |
| D=6 | 0.358 / 0.508 | 0.332 / 0.416 | 0.026 |
| LN=.7 | 0.402 / 0.512 | 0.309 / 0.359 | 0.029 |
| Jack prominent | 0.373 / 0.435 | 0.183 / 0.249 | 0.014 |
| Stream prominent | 0.387 / 0.457 | 0.242 / 0.355 | 0.022 |
| Trill prominent | 0.379 / 0.402 | 0.287 / 0.418 | 0.022 |
| Tech prominent | 0.373 / 0.482 | 0.338 / 0.455 | 0.026 |
| Live override | 0.367 / 0.589 | 0.280 / 0.351 | 0.021 |

Across all 90 runs, RTF ranged from .0097 to .0325: about 31–103 audio seconds
per compute second. Whole generation took 2.773–11.644 s for the 144–358 s
inputs. The six long-song runs became ready in 0.501–0.600 s from Mel, with a
maximum eight-second service time of 0.386 s and maximum RTF .0275.

For the matched base and LN-high panel, each checkpoint has 18 full-song runs:

| Checkpoint | Parameters | Ready median (s) | Median RTF | 8-s service p95 / max (s) |
| --- | --- | --- | --- | --- |
| core | 4,583,985 | 0.404 | 0.024 | 0.304 / 0.459 |
| modulated | 4,600,369 | 0.434 | 0.025 | 0.309 / 0.458 |
| aligned | 4,600,369 | 0.396 | 0.023 | 0.298 / 0.376 |

Different weights and controls produce different row counts. These totals
measure the resulting workloads; they do not isolate the incremental cost of
the 16,384-parameter modulation matrix.

## Where generation time goes

The following diagnostic uses complete Take audio, seed 263900, and the aligned
checkpoint. All times are seconds. H plan includes H-history updates and hazard
sampling. R network and R1 scores are their neural calls; R1 scores include
composition, placement and frontier scoring. Row/R history combines the two
append operations. Other includes support construction, recovery/LN preferences,
release survival sampling, tensor preparation, exact commit and loop overhead.
It is a timing residual, not an attribution to one implementation component.

| Request | Rows | Audio | H plan | R network | R1 scores | Row/R history | Other | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unspecified | 1043 | 0.094 | 1.055 | 0.006 | 0.526 | 0.898 | 0.635 | 3.214 |
| D=3, LN=.2 | 966 | 0.099 | 1.066 | 0.020 | 0.497 | 0.870 | 0.890 | 3.442 |
| D=1.5 | 865 | 0.100 | 0.963 | 0.020 | 0.444 | 0.769 | 0.870 | 3.168 |
| D=6 | 1135 | 0.096 | 1.150 | 0.024 | 0.568 | 0.974 | 0.992 | 3.804 |
| LN=.7 | 1331 | 0.098 | 1.073 | 0.055 | 0.677 | 1.192 | 1.543 | 4.639 |
| Jack prominent | 814 | 0.095 | 0.915 | 0.016 | 0.413 | 0.705 | 0.738 | 2.882 |
| Stream prominent | 1009 | 0.096 | 1.179 | 0.013 | 0.562 | 0.928 | 0.859 | 3.637 |
| Trill prominent | 983 | 0.096 | 1.074 | 0.015 | 0.500 | 0.849 | 0.807 | 3.341 |
| Tech prominent | 1136 | 0.095 | 1.301 | 0.012 | 0.591 | 1.001 | 0.873 | 3.873 |
| Live override | 1055 | 0.099 | 1.139 | 0.030 | 0.556 | 0.969 | 1.125 | 3.918 |

For the base request, H planning and the two history append paths together take
1.936 s of the 3.442 s diagnostic, versus .497 s in the row scorer and .020 s
in release logits. The short neural calls and repeated state updates make
end-to-end runtime a more useful backend criterion than full-audio encoding alone.

## CPU, MPS and thread counts

This comparison encodes the same complete 144.236-s Take audio, then generates
only its first 32 seconds. Each table cell is the median of six matched
base/LN-high runs. This is not an MPS full-song throughput measurement.

| Model | CPU total (s) | MPS total (s) | CPU encode (s) | MPS encode (s) |
| --- | --- | --- | --- | --- |
| core | 0.929 | 4.100 | 0.095 | 0.023 |
| modulated | 0.826 | 3.812 | 0.095 | 0.023 |
| aligned | 0.906 | 3.759 | 0.098 | 0.023 |

MPS saves audio-encoding time but loses end-to-end on this sequential runtime.
Across the six matched pairs, median MPS/CPU latency ratios are 4.52 for core
and 4.52 for modulated. The aligned panel additionally tests high difficulty;
its nine matched ratios have median 4.01, range 3.45–4.72. All 21 CPU/MPS pairs
produce identical row hashes. The result does not identify a particular Metal
kernel or transfer as the sole cause.

The following Take prefix comparison retains the aligned checkpoint and six
base/LN-high cases. Additional intra-op threads help the large encoding pass
but do not improve the overall median in this panel:

| CPU intra-op threads | 32-s generation median (s) | Encoding median (s) |
| --- | --- | --- |
| 1 | 0.906 | 0.098 |
| 2 | 1.050 | 0.074 |
| 4 | 1.073 | 0.063 |

## Preprocessing, startup and cache reuse

Full canonical Mel is computed from the decoded, globally peak-normalized
24 kHz mono waveform. Each preprocessing row below is a median of three fresh
computations with a resident process; cached reads use `.npy` on the local disk.
Every fresh Mel exactly matched its pinned cached array.

| Audio (duration) | Decode (s) | Mel (s) | Mel cache read (ms) |
| --- | --- | --- | --- |
| Zenithfall (357.796 s) | 0.505 | 0.142 | 2.148 |
| Hysteric (301.008 s) | 0.384 | 0.110 | 1.063 |
| Take (144.236 s) | 0.212 | 0.050 | 0.946 |
| Yomi yori (498.989 s) | 0.803 | 0.206 | 2.817 |

A separate probe starts a fresh Python process for each request. Each row below
is three repetitions of the base request. Component medians need not sum to
the independently measured total. The final column includes audio encoding;
it must not be added to a separate encoding counter.

| Device / audio | Process to ready (s) | Imports (s) | Model load (s) | Decode (s) | Mel (s) | Encoding + first coverage (s) |
| --- | --- | --- | --- | --- | --- | --- |
| cpu/take | 1.454 | 0.814 | 0.032 | 0.248 | 0.071 | 0.270 |
| cpu/yomi-yori | 2.514 | 0.769 | 0.032 | 0.811 | 0.221 | 0.613 |
| mps/take | 2.272 | 0.756 | 0.147 | 0.233 | 0.065 | 1.064 |

The startup data favors a resident model and reused audio features. An encoded
float32 song occupies 12.33 MiB for Take, 25.72 MiB for Hysteric and 30.57 MiB
for Zenithfall. The serial reuse probe returns the same immutable encoding to
new sessions; it retains the existing constructor's Mel copy and is not an
implemented concurrent cache. All 18 complete-song reuse pairs have identical
rows to their uncached counterparts. Their resident readiness median is .198 s,
with maximum .311 s; separately timed full-song totals do not establish a
throughput improvement after controlling for run-order effects.

A further within-process probe alternates hit/miss order over six Take prefix
pairs. Ready medians are 0.305 s with re-encoding and 0.213 s on a hit;
the median paired saving is 0.124 s. All six pairs preserve the row hash.
This establishes a startup benefit without assuming that the decoder becomes
faster. Across primary-run endpoints, sampled process RSS reached 1.357 GiB;
transient peaks, cache eviction and concurrent sessions were not measured.

## Publication granularity and a bounded buffer

The aligned Take panel uses base/LN-high requests and three matched seeds,
generating complete songs. Each 1/2/4-second result exactly equals its
8-second counterpart, including real LN endpoints. Smaller publication units
therefore preserve these sampled trajectories while reducing service bursts.

| Publication unit | Windows | Service median / p95 / max (ms) | Whole-song median (s) |
| --- | --- | --- | --- |
| 1 s | 870 | 27.228 / 44.288 / 79.504 | 4.126 |
| 2 s | 438 | 53.348 / 83.739 / 116.668 | 3.992 |
| 4 s | 222 | 109.625 / 166.046 / 183.985 | 3.993 |
| 8 s | 114 | 202.097 / 305.286 / 349.961 | 3.904 |

All 90 main traces have zero virtual presentation misses at 2 s lead; minimum
unscaled slack is 5.739 s. The virtual consumer starts only when the declared
30-row/8-second requirement is met. This calculation uses the **previous**
published watermark before each next delivery, never the future watermark.

A separate paced simulation replays the six one-second traces with a 6 s low
watermark and 8 s refill target, stopping work between refill episodes. Its
minimum presentation slack is 3.942 s. Inserting one 1 s stall at the
slowest post-start service leaves 2.942 s; multiplying all service
costs by ten leaves 3.469 s. These are deterministic simulations over
measured costs, not a paced server or background-contention test.

## Serving design indicated by the measurements

A practical first configuration is one CPU generation worker with a resident
model, one-second publication units, eight seconds of initial coverage, and
refill from six toward eight seconds ahead. Keep full-audio preparation outside
the publication deadline loop. The measured budgets support this configuration;
its transport and real wall-clock scheduling still need integration tests.

Cache immutable Mel by audio identity plus frontend identity. Cache the complete
encoding by those identities, checkpoint, device and dtype. A bounded LRU can
manage these shared assets. Each session separately owns exact replay, LN
feedback, H lookahead, H/R/row RNGs, survival residuals and finite history caches.
A different seed or control schedule may reuse audio features; it must not
reuse another session's generated state. Control revisions act strictly after
the published clock through the model's scoped-update policy. Durable recovery
needs its own state/replay contract; it is not supplied by these in-memory probes.

The inspected Ensomi client revision is
`e4ae9e80f2b811fcddb265b134d917d7e6d149ce`. Its
`BufferedInferenceMania4KHitObjectStream.read` currently returns the requested
`throughChartTimeMs` as `completeThroughChartTimeMs` without consulting a
producer coverage watermark. The play session rejects later events at or below
that watermark. Integration must carry confirmed coverage from the producer and
advance the consumer watermark only after all events through it have actually
been delivered, including when a read is limited. An empty completed interval
must remain distinct from a pending interval. Final EOS requires true audio
termination and closed holds.

The client's existing startup rule additionally needs a clean LN entry, a 1 s
first-object lead and 5 s of ready chart. The benchmark's 30-row/8-second
readiness is not a substitute for that rule. Neither a last note timestamp nor
an available H plan certifies settled chart coverage. Final system validation
must include silent intervals, long holds, future control changes, late joins,
seeks, cancellation, limited reads and stalled delivery.

## Reproduction and evidence

The packaged entrypoint is
`ensomi_model.research.controlled_audio_continuation.benchmark_hydra`; its
[typed settings](../../src/ensomi_model/research/controlled_audio_continuation/benchmark_config.py)
and [runner](../../src/ensomi_model/research/controlled_audio_continuation/benchmark.py)
own the measurement procedure. For example:

```sh
uv run --extra mps python -m ensomi_model.research.controlled_audio_continuation.benchmark_hydra \
  checkpoint_file=/absolute/path/to/step-400.pt \
  checkpoint_sha256=a867547cf39d1284ca66daaca83e7f58f70851c3e8706c9971a1e7a665790214 \
  panel_file=/absolute/path/to/panel.json \
  panel_sha256=1f089af145b45c232c0ab95de77f03cee8b6a6a1b319f2335ac38f59eb79051b \
  output_dir=artifacts/performance/new-run \
  'case_names=[zenithfall,hysteric,take]' \
  'conditions=[base,ln_high,switch]' preprocess=true
```

The output directory must be fresh. Checkpoints, audio and features are local
research assets, not included with the repository. Each output records the
resolved configuration, source and checkpoint identity, frontend identity,
runtime versions, first-use pass, shuffled job order, per-window traces, row
hashes and independent verification. Stage diagnostics are stored separately
and never substituted for ordinary latency records.

The raw owner is `artifacts/performance/20260926-stream-generation-v1`.
`commands.json` and `supplement-commands.json` record exact generation
invocations; `cold_start.jsonl` records each fresh-process invocation.
`paired_cache.py` and `buffer_replay.py` retain the alternating cache comparison
and paced simulation. `aggregate.json` contains the derived statistics.
The evidence-file manifest SHA-256 is
`45be27ce9c32dc26e050902d5dce0ade3948a8ffa29a29a8c5f87f14dc2be1fb`.

The locked SciPy wheel failed loading in a newly created environment on this
macOS build because of a Mach-O zero-fill section offset. Measurements used the
existing working research environment, explicitly selected with
`UV_PROJECT_ENVIRONMENT`, `PYTHONPATH=src`, `uv run --no-sync --extra mps`.
`host.json` records its SciPy binary and lockfile hashes. This environment
limitation must be resolved or the compatible environment reused for reproduction.

Focused checks cover watermark slack accounting, strict Hydra projection,
bitwise trajectory preservation under stage instrumentation and encoding reuse,
and package resource inclusion. The entrypoint help path was also exercised.
