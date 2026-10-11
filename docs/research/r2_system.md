# Ensomi R2: the end-to-end chart system

<!-- SKELETON: facts only, for the main thread to write up. The "Source" columns and bracketed anchors name the
relay-note file (artifacts/<file>.md) and anchor each figure came from; remove them before publishing, the notes are
not public. -->

## What it is

- Audio file + star request in [2, 6) + optional controls in; legal osu!mania 4K `.osz` (chart, audio, `receipt.json`) out.
- Code: `src/ensomi_model/system/` (entry `System`, `make`), `src/ensomi_model/audio_rows/` (timing system, rows model), `src/ensomi_model/r2/`.
- Whole song at a time; the rows model needs the complete audio and the head-row count before R2's first row. Not streaming. [handoff-20261011-e2e, "It generates a whole song at a time"]
- Tags `rows-1.0`, `timing-1.0`, `r2-1.0` on the release commit (to be created by the caller). Hugging Face: `sed-i/ensomi-r2`.

## Components

| Component | Tag | Parameters | Training data | Training commit |
| --- | --- | --- | --- | --- |
| Timing system (Listener) | `timing-1.0` | none of its own; BeatThis `final0` (third party, MIT, not redistributed) | no training | code `audio_rows/audio.py`, `audio_rows/fitter/` |
| Rows model | `rows-1.0` | 2,105,199 trainable + 8,853-figure buffer; 9,622,005 bytes; no optimizer state | 11,368 train charts, 1,163 dev | `cf753aa`; run `config.json` is authoritative (two later commits changed `fit.py` defaults; the job passed settings explicitly) [release-r2-20261011, `o-release-inventory`] |
| R2 (frozen, phase N) | `r2-1.0` | 2,404,724; slim file 9,655,631 bytes (`model`, `model_config`); training checkpoint 27.3 MB incl. 17.6 MB optimizer state | R2 cache v1: 12,531 charts, 13.97M head rows; fit_train 11,368 charts in 4,167 groups | `3954031` |

Sources: parameter counts, bytes: `artifacts/release-r2/ckpt_inspect.json` (mac) and [r2-v2-stage0-state, `s-v2-stage0-state`] for 2,404,724; corpus counts: [handoff-20261008-audio-rows.md, `summary.json` line "12,531 charts, 13.97M head rows"], rows record `train_charts` 11368, `dev_charts` 1163 (`artifacts/release-r2/rec_inspect.txt`).

### Timing system facts

- Pipeline (`audio_rows/audio.py` docstring): BeatThis `final0` beat/downbeat probabilities -> legacy grid fit -> `refine_segments` to BeatThis sub-frame peaks -> `downbeat_phase` -> `mel_phase` (segment offset to the song's own log-Mel onsets) -> `mel_tempo` (tempo refit to their drift). Model inputs: standardised 128-bin log-Mel at 10 ms hop (24 kHz) + BeatThis beat/downbeat probabilities.
- Log-Mel flux onsets about 2.7 ms rSD from mappers' beats within a chart vs 6.5 ms for BeatThis peaks; Mel steps need no training, add about 0.003 s per audio-minute. [timing-phase-20261009, `s-timing-final`]
- Sealed holdout, 519 charts of 289 songs (one song failed BeatThis's loader, excluded), paired on identical `final0` logits [timing-phase-20261009, `o-opt-final`]:
  - charts with 90 % of heads within 20 ms (S1-20): 0.784 -> 0.850 (+0.066 [+0.033, +0.101]);
  - heads within 20 ms (H-20): 0.897 -> 0.931 (+0.034 [+0.018, +0.055]);
  - S1-10: 0.514 -> 0.674.
- Runtime 1.499 s per audio-minute, quiet mac, CPU, 2 threads [`o-opt-final`].
- No human has screened the grids by ear [`o-opt-final`, "Evidence strength"]. Out of reach chart-free: per-set offset at 10 ms (ridge on 166 descriptors, R² at most 0.08); short tempo sections and phase resets (85 % of those spans hold fewer than four BeatThis peaks) [`s-timing-final`].

### Rows model facts

- Architecture (`audio_rows/net.py` docstring, record `net`): frame encoder (dilated convs, about ±310 ms), transformer over beats (window ±64, 4 heads, 2 layers), GRU hidden 256, figure vocabulary 8,853 over 24 positions, 4 audio-keyed copy heads over 64 lags, request = log(W_H / s).
- Training record (`rec_inspect.txt`): 5,000 steps, batch 8, window 512, lr 2e-3 -> 2e-4, warmup 100, seed 1, device mps, `copy_rests` true; label_mean 2.392, label_std 0.627; `star_map` of 16 star/request points.
- Chosen model `full2-s1` step 5,000 vs the old generator, 96 held-out fit_dev songs, 231 human charts, 2 seeds, characterising only [rows-e2e-20261010, `o-p1-stageb`]:

  | Measure | Rows | Old | Human |
  | --- | ---: | ---: | ---: |
  | onsets over a half-beat shift | .050 | .024-.028 | .052 |
  | rows repeating where the audio repeats | .19 | .05 | .234 |
  | head F1 at 20 ms | .74 | .66 | pairs .868 |
  | figure agreement | .71 | .61 | pairs .978 |

  - Input ablations: request from another chart of the same audio +.068 nats/beat; copy heads off +.41; position audio off +.72.
  - Attained density rises with the request on 100 % of songs. End third minus first third +.20 to +.23 (human -.02, wide).

### R2 facts

- Architecture: TCN (82.8 % of parameters), hidden 128, 8 levels, FiLM conditioner width 128, memory `none`, `rule_l` on, 625-way lane-action code per head row plus gap releases. Does not read audio. [r2-v2-stage0-state; `ckpt_inspect.json` `model_config`]
- Phase N: teacher-forced CE, no conditions. Config: lr 3e-4, warmup 20k, cosine to 3e-5, window 256, accumulate 4, checkpoints every 4M, 64M budget, CPU 4 threads, seed 171. [r2-phasen-and-lens-20261006, `s-phasen-tune`, `r-phasen-run`; run `config.json`]
- Run ended at 64,000,573 exposures; fit_dev natural NLL per decision: 2.130 (24M), 2.089 (32M), 2.071 (40M), 2.070 (48M), 2.075 (56M), 2.084 (64M). [`o-phasen-finished`]
- Selection of 56M (56,000,574 exposures): corrected guards (iv) v2 and (v); 56M passes everything ((iv) 0.454 / 0.746, (v) +0.034); 24M also passes but lies outside the two-SE NLL plateau; 48M fails (iv) near-head releases (1.355); 32M fails (iv); 16M, 40M, 64M fail (v). [`o-selection-v2`; code `827e306`, report `artifacts/r2-guard-v2-20261007/report.md` on the mac]
  - Caveat: the run directory's `selection.json` predates the correction and records "no selection: no candidate passes every guard". The guards were calibrated on a short panel and are not validated against the human (guardrail 1 in `.agents/skills/ensomi-research-guardrails`).
- 56,000,574 exposures about 4.4 passes over 12.7M fit_train head rows (121.7M exposures = about 9.6 passes, so 12.7M head rows) [r2-condition-plan.md].

## Decode-time parts

- **Star to request:** `star_request` through the record's `star_map`.
- **Section tilt** (`rowsgen.py`): in a section each figure's log-probability gains head count x tilt; tilt = `SECTION_GAIN` 3.0 x (request - realised), clipped to ±`TILT_MAX` 2.0; realised = log W_H/s over the section's beats so far, extended back to the last 4. Learned section input alone moves 12-17 % of a request; tilt moves 80-92 %, right direction 96-98 %, spill .09-.15. [`o-p1-stageb`]
- **Copy heads off empty beats** (`copy_rests` off at decode): on pilot weights the last fifth of long songs was .89 below the request, +.01 with the heads kept off empty beats (human -.27). A decode-time patch on pilot-recipe weights, not a training fix; one untested arm would separate the causes. [`rowsgen.py` docstring; `o-drift-rca`]
- **Quiet-rests rule** `quiet_rests` 0.40 (`QUIET_RESTS`, `REST_WINDOW` 4): copy heads may attend to empty beats below the song's 40th percentile of local intensity. 96-song panel: quiet-section rests .053 -> .076 [.057, .098] (human .085), paired +.023 [.013, .035]; long-song last fifth -.095 (human -.146); gain in bands 4-5 (band 5 overshoots .113 vs .079), bands 2-3 barely move; precision .409 -> .372, recall .314 -> .363. Adopted as default. [rows-e2e-20261010, `o-rests`, `s-rests-adopt`]
- **Prior** (`system/prior.py`, `prior.json`): kNN, 64 neighbours, fit_train 11,368 charts (per band 3,754 / 3,530 / 2,819 / 1,265 train charts); features log rows/s, log median gap, log W_H/s, log BPM; targets held share, heads per row, busy-free, quad share, log r. Replaces the source chart's values the R2 demo used. [`o-p2`; prior.json]
- **Restoring force** (`system/arrange.py`): candidate bias = -(candidate values of [LN heads, heads, all-busy] . (kappa x (running - target))), running over the last 64 rows, kappa (3, 3, 3); acts after masks, before the Gumbel draw.
- **Masks** (calibration `masks`): re-attack floor ms by band 2-5 = 145 / 105 / 85 / 75; quad run cap 0 / 0 / 1 / 1; jack run caps by gap tercile [2,3,5], [2,4,6], [2,4,8], [1,3,14]. No surviving action -> legal single head on the oldest lane, counted as a fallback.
- **Four controls:**
  - LN amount `tap|half|natural` (natural = prior's held share);
  - whole-song difficulty: r at quantile q of human charts in the band (values per band 3,568 / 3,479 / 2,807 / 1,264); default q .5 (bands 2-4), .467 (band 5); q -> heads-per-row multiplier m by per-band power map (fit: 40 observations, 4 charts per band, seed SD .039 / .033 / .020 / .011), clamped to [0.8, 1.2] (`DIFFICULTY_M_EDGE`), one correction pass; tested q range .1-.75;
  - stream and jack sections, 8-32 s, presets jumpstream / chordstream / chordjack, targets looked up on `scopes.parquet` (132,053 rows);
  - section difficulty: absolute stars only, applied in the rows model [rows-e2e-20261010, `d-absolute-sections`].
- **Whole milliseconds:** heads rounded before R2, gap releases after (kept strictly between neighbouring rows), timing points at export; export is parsed back and replayed (objects, head times, holds closed).
- **R2 copy path** (`system/copy*.py`): off, weights not shipped. Rule: no candidate raised recurrence beyond seed noise .013 (C1 +.007, C2 +.008, C3 +.009, C4 +.004) [`system.json` reason; `o-p2`].

## Verified, and how

All figures one listener, one seed per side unless stated. [rows-e2e-20261010]

| Item | Result | Source |
| --- | --- | --- |
| Early packet, new rows vs old, 6 pairs | new 4, old 1, can't tell 1 | `h-early-verdict` |
| Final packet, rows new vs old | new 6, old 2 | `h-final-verdict` |
| Combined | new 10, old 3, 1 can't tell | `h-final-verdict` |
| Whole-song star pairs 015, 016 | both called correctly | `h-final-verdict` |
| Jumpstream, chordstream | work | `h-final-verdict` |
| Chordjack | out of band | `h-final-verdict` |
| LN | can't tell; "long notes didn't perform well" | `h-final-verdict` |
| Copy | on 1, off 2, no difference 1 | `h-final-verdict` |
| Human chart vs generated, bands 2-4 | human chart chosen 3 of 3, no notes | `h-final-verdict` |
| Seed nulls | a side picked in both | `h-final-verdict` |
| Rests | pair 001 "too empty in the silence part"; early 005 wanted more breath (sub-beat spacing: rows stream at quarter beats where the mapper used half beats, 0 of 32 empty beats in all three) | `h-final-verdict`, `o-rests` |
| Absolute-section pairs | not yet screened | handoff-20261011-e2e |

Measured, characterising (phase 2 [`o-p2`], measured with `quiet_rests` 0; the end-to-end runs used .40):

- 11 of 11 end-to-end runs legal, whole-ms times; difficulty within seed SD on 9 of 11; 84 tests passed at that time.
- Controls, 24 held-out songs x 2 seeds per arm, copy off, attained in band: stream 48/48, jack 35/48, jumpstream 36/48, chordstream 37/48, chordjack 18/28; fallbacks .15-.59 per 1,000 head rows; LN half removes about a third of the LN; difficulty multiplier clamps in a third of runs at q .1 and .75.
- Default q: W_H/W near human medians; band 2 .527 vs .535.
- Section gain test [`o-section-fix`]: gain 3 up residual -.055, down +.150; gain 4.5 -.045 / +.098 (fails up artifacts); gain 6 -.030 / +.057 (fails artifacts and precision); 12+-row beats .090 at gain 3 vs .018 in human charts.
- Open rows items [handoff-20261011-e2e, "Open, for later"]: quarter-beat stream tendency, rest level pulls both ways, long-song stability rests on a decode-time rule.

Claims allowed as working: jumpstream and chordstream requests; whole-song difficulty by star. Not offered: LN amount on generated rows; absolute section difficulty; copy; (chordjack, by the screen).

## Release check

- Parity record: `artifacts/release-r2/parity.md` (mac; text mirrored). P1-P4 regenerated from the built directory, cwd outside the repo, empty `TORCH_HOME` so BeatThis `final0` is downloaded: hit objects identical, 1,214 / 1,215 / 1,192 / 2,557. Old `--config` path for P1 identical. Stream + jack (4 requests) identical between the original calibration/scope table and the shipped ones (1,179 objects). Downloaded `final0` SHA-256 equals the cached file's (`8c328b45...8331`).
- Clean-room `uv sync --python 3.10 --extra mps`: needed `pandas==2.3.3` added to `pyproject.toml` (commit `431452a`); on this mac the scipy 1.15.3 wheel fails to dlopen (macOS 27.0 build 26A428) and the repo `.venv`'s patched copy was substituted; the check passed with that.
- Tests on the mac at `431452a`: `tests/system` 33 passed, `tests/audio_rows` 51 passed, `tests/test_package_layout.py` 1 passed (10 subtests).

## Runtime

- Release check (macOS arm64, Apple M5, Python 3.10.20, torch 2.11.0, CPU, 1 thread), process wall time: P1 (2.8 min) 19 s cold incl. 77.3 MB BeatThis download (77 s in the clean-room run on a slow link); P2 (2.2 min) 8 s; P3 (2.0 min) 8 s; P4 (3.3 min) 12 s.
- In process (listener / rows / R2 / total s): P1 5.5 / 0.19 / 1.19 / 7.0; P2 4.3 / 0.12 / 2.06 / 6.6; P3 4.1 / 0.11 / 2.20 / 6.5; P4 7.2 / 0.21 / 2.44 / 10.0.
- Earlier measurement, idle mac, model loads included: 7.8 / 11.6 / 14.6 s for 2.0 / 3.3 / 5.4 min songs [`o-p2`]. Not run on Linux or CUDA.

## Download and run (commands actually run)

```sh
git clone https://github.com/ensomi-labs/ensomi-model.git && cd ensomi-model
git checkout --detach r2-1.0                       # tag to be created; the checks ran on release/r2
uv sync --python 3.10 --extra mps                  # Linux NVIDIA: --extra cuda (not run)
hf download sed-i/ensomi-r2 --local-dir artifacts/hf-r2   # not run: the repository is not uploaded yet
uv run --python 3.10 --extra mps python -m ensomi_model.system.make \
  --model-dir artifacts/hf-r2 --audio /path/to/song.mp3 --star 3.5 \
  --title "Song title" --artist "Artist" --seed 0 --out artifacts/my-chart.osz
```

The run checks used `uv sync` in a clean environment and `--model-dir` on the locally built directory (`artifacts/release-r2/bulk/hf/ensomi-r2/`). `hf download` has not been run.

## Files and licences

- File table with sizes and SHA-256: the model card (`hf/README.md` in the scratch directory, filled in by the build) and `SHA256SUMS` in the directory.
- Not shipped: BeatThis weights (MIT; fetched by `beat_this` from its own source), copy weights, cache index (paths expose titles and mappers), audio of any song.
- HF repo content (weights, prior, calibration, scope table, card, examples, previews): CC BY-NC-SA 4.0 (`LICENSE` is the official legal code). GitHub code: AGPL-3.0-only. Example charts carry song titles and artists; no audio.
- `.pt` files are PyTorch pickles loaded by the project's code.
- Removed from the shipped calibration: `difficulty_fit.monotone_per_chart` (16 chart hashes), `step2a.status_file`, `subtype_calibration.grade_landmarks`; the scope table keeps 8 columns (`band`, `length_s`, `median_gap_ms`, `J_ex`, `S`, `jump_share`, `hand_share`, `chord_share`), 132,053 rows, 2,447,837 bytes (from 9.9 MB). Example receipts: source audio reduced to a file name, config paths release-relative, copy weights null; their `r2.sha256` is the full training checkpoint's (`e67f388d2f10...`), the regenerated `.osu` comment carries `r2/model.pt`'s prefix (`8dedefe4b11d`).
