# Codex brief: Ardour's detectors and openDAW's TempoCNN on the Smart Tempo baseline's 48 songs

**Status:** written by a research worker on the control plane, for review by the main thread before it is
submitted. This brief grants nothing beyond the permissions the task already has.

**Guardrails:** read `.agents/skills/ensomi-research-guardrails/SKILL.md` in the source snapshot below. The ones
that apply here: 1 (every number characterises; nothing here decides), 2 (the determinism null in step 6), 4
(the fixed inputs listed under "Arms"), 6 (the probe canary in step 4), 8 (report only what the tables carry)
and 10 (the absolute checks in steps 2, 3 and 7).

## Why

Ensomi N1 is a music timing model: from audio it infers beat lines, downbeats, tempo changes, meter and offset.
On 2026-10-21 the human decided that Ardour's detectors, openDAW's TempoCNN and Logic Pro's Smart Tempo are all
reference baselines for automatic tempo maps and beat tracking. Smart Tempo has already been run and scored on 48
hard songs (`artifacts/reports/logic-tempo-baseline-20261020/`, run 2026-10-20). This run puts the two open tools
on the same 48 songs, reading the same audio, scored by the same code and clock rule, so that all of them sit in
one table with N1 v0 and R2. Nothing here changes the Smart Tempo results.

## Rules

- **Audio stays on this Mac.** Do not upload, publish, sync or share any audio or any file derived from it to any
  hosted service. The results directory syncs to the control plane as the Smart Tempo results did; it holds JSON
  and Markdown only, no audio.
- **Allowed:** Node 24 (installed), the openDAW checkout with its npm installs already done, the Ardour 9.8.0
  demo app already copied out, ffmpeg already installed at `/opt/homebrew/bin/ffmpeg`, the project's existing
  Python environment, and `git fetch` and `git archive` of ensomi-model.
- **Not allowed without the human's go-ahead:** Rust, mkcert, Homebrew packages, pip installs, any new toolchain,
  and the MiniBPM compile in Part M unless its go-ahead line is filled in.
- **Do not modify** openDAW's or Ardour's code, the Smart Tempo results
  (`artifacts/reports/logic-tempo-baseline-20261020/` and `artifacts/bulk/logic-tempo-baseline-20261020/`), or
  the canonical WAVs. You may fix a small mistake in this run's own scripts, such as a Lua binding name or an
  import path: record each fix as a diff with its reason in `$OUT/NOTES.md` and in your reply.
- **Never fit an offset per song, never choose a tool parameter from a chart.** The scripts already hold the only
  chart-reading arm, `qm-bbt-oracle`, which is labelled an oracle.
- **No GUI steps.** The optional GUI check under Part M is the human's.

## Paths

```sh
cd ~/ensomi/ensomi-model                      # the project root; every command below runs from here
COMMIT=<filled in by the main thread: the n1/open-baselines commit to run>
SRC=artifacts/bulk/n1-src/${COMMIT:0:7}       # a source snapshot of that commit
OUT=artifacts/reports/open-baselines-20261021 # results, synced to the control plane
BULK=artifacts/bulk/open-baselines-20261021   # per-song tool outputs, logs, sessions; stays on the Mac
LOGIC=artifacts/reports/logic-tempo-baseline-20261020
ARDOUR_LUA=ref-proj/ardour-demo/Ardour9.app/Contents/MacOS/ardour9-lua
OD=ref-proj/opendaw                           # openDAW checkout, 6087ee55a3b90220d5d2d46f13563e969f5f50c1
MODEL=artifacts/bulk/open-daws-20261021/opendaw/tempo-cnn-v0.onnx
OB="$SRC/scripts/n1/open_baselines/open_baselines.py"
PY=<the Python of the project environment that ran scripts/n1/logic_baseline.py for the Smart Tempo run>
export PYTHONPATH=$SRC/n/src PYTHONDONTWRITEBYTECODE=1
```

Use `$ARDOUR_LUA` exactly as written: the `Contents/MacOS/` wrapper sets `VAMP_PATH` to the bundle's plugins.

## 0. Source snapshot and versions

```sh
git fetch origin n1/open-baselines
git rev-parse FETCH_HEAD                      # must equal $COMMIT; otherwise stop and report both
mkdir -p $SRC && git archive $COMMIT | tar -x -C $SRC
mkdir -p $OUT $BULK
```

Record in your reply, and in `$OUT/NOTES.md`:
- `$PY -c 'import sys, numpy, pyarrow; print(sys.version, numpy.__version__, pyarrow.__version__)'`;
- `node --version` (must be 23.5 or later, for `module.registerHooks`);
- `plutil -p ref-proj/ardour-demo/Ardour9.app/Contents/Info.plist | grep -i version` (expect 9.8.0);
- `git -C $OD rev-parse HEAD` (expect `6087ee55a3b90220d5d2d46f13563e969f5f50c1`);
- `shasum -a 256 $MODEL` (expect `38b4915b35c46f72e072a7c93ab4b7e280404133e2ca94f9cbcf5ace15c7a321`);
- that `$OD/packages/lib/inference/dist/tasks/TempoDetectionTask.js` and `$OD/packages/lib/dsp/dist/` exist
  (both were built in the trial of 2026-10-21).

A missing tool stops only its own part; record it and go on with the others.

## 1. Arms (what each one is; nothing to run here)

Every arm reads the canonical decode the Smart Tempo capture read, channel 0 (left), at its own sample rate, with
the tool's default parameters, on the whole file.

| Arm | Tool and route | Output scored |
| --- | --- | --- |
| `qm-bbt` | Ardour's bundled Vamp `qm-barbeattracker`, 4 beats per bar (its default), through `ardour9-lua` (`ARDOUR.Readable.load`, `ARDOUR.LuaAPI.Vamp:analyze`) | beats and bar starts |
| `qm-bbt-oracle` | the same tracker at each beats-per-bar value the chart states; best run per song | an oracle, never a baseline |
| `qm-tempo` | Ardour's bundled Vamp `qm-tempotracker`, the same route | beats, no bars |
| `mbpm-clip` | Ardour 9.8's MiniBPM on the whole file, rounded as Ardour rounds a clip's tempo | one tempo, 4/4 from the file start, as a clip trigger plays it |
| `mbpm-raw`, `mbpm-source` | MiniBPM before the rounding; MiniBPM on 10 s from the middle, rounded over 10 s | one tempo each |
| `tcnn` | openDAW's own `TempoDetectionTask` (TempoCNN, onnxruntime-web, wasm) | one integer BPM |

The clock rule is Smart Tempo's: one decoded-to-osu! constant per run, c20 = -20.0 ms and c287 = -28.7 ms, both
scored, never a per-song offset.

## 2. Prep: the songs and their canonical decodes

```sh
$PY $OB prep 2>&1 | tee $BULK/prep.txt
```

- It reads the 48 songs from `$LOGIC/analysis/songs.json`, the capture's `manifest.json` and its
  `raw-resu/NNN__MODE.capture.json` records in `$LOGIC/bulk/`, and checks each canonical WAV (by default the
  manifest's `prepared_input_path`, `/Users/l/Downloads/hard-songs/logic-baseline/inputs-wav/NNN.wav`) against
  them: 32-bit float, frame count, sample rate, channels, and the SHA-256 of the sample payload. These are the
  values the Smart Tempo preflight checked on 2026-10-20 (`$LOGIC/analysis/preflight-facts.md`, section F4).
- It writes two synthetic 120 BPM click probes (`P44`, `P48`) to `$BULK/probes/`, `$BULK/jobs.json`, and
  `$OUT/open-songs.json`.
- **If it reports problems, stop** and report them. If the `inputs-wav/` directory has moved, find it, rerun with
  `--wav-dir <dir>` and report the path. If it is gone, decode each source again with the exact ffmpeg command
  recorded in the manifest (`conversion.command`), writing to `$BULK/inputs-wav/NNN.wav`, and rerun with
  `--wav-dir $BULK/inputs-wav`; the hash check decides whether that decode is the canonical one. Do not use any
  WAV whose hash differs.

## 3. Preflight: the Smart Tempo results reproduce

```sh
$PY $OB preflight 2>&1 | tee $BULK/preflight.txt
```

It re-scores the Smart Tempo run's arms (Smart Tempo, v0, R2) from that run's own inputs with
`scripts/n1/logic_baseline.py` (unchanged from `ba5e538`) into `$OUT/preflight/`, and compares the new
`results.json` and `items.json` with `$LOGIC/analysis/`. **If any number, flag or structure differs, it exits
non-zero: stop the run** and report `$OUT/preflight/compare.json`. Differing strings alone (such as a path
written another way) are listed and do not stop it; report them.

## 4. Probe canary

```sh
$PY $OB ardour   --ids P44,P48 --ardour-lua $ARDOUR_LUA 2>&1 | tee $BULK/ardour-probes.txt
H=$BULK/tempocnn-harness && mkdir -p $H && ln -sfn "$(cd $OD && pwd)/node_modules" $H/node_modules \
  && cp $SRC/scripts/n1/open_baselines/tempocnn_one.mjs $SRC/scripts/n1/open_baselines/node_resolve.mjs $H/
$PY $OB tempocnn --ids P44,P48 --opendaw $OD --model $MODEL --harness $H 2>&1 | tee $BULK/tempocnn-probes.txt
$PY $OB status
```

- Each probe should give status `ok` for `qm-bbt`, `qm-tempo` and `tcnn` (`$OUT/run-status.json`, `probes`).
  `P48` checks the 48 kHz path (step size 557, block 1114).
- **If a tool fails on both probes, do not run it on the songs.** Report the raw output
  (`$BULK/raw/<tool>/main/P44*`) and its log, and go on with the other tools. Do not change a tool to make it run.
- The probes' median offsets are a fact about each tracker's timestamps on clicks. They are reported, never
  applied to the songs.

## 5. Song runs

```sh
$PY $OB ardour   --ardour-lua $ARDOUR_LUA 2>&1 | tee $BULK/ardour.txt
$PY $OB tempocnn --opendaw $OD --model $MODEL --harness $H 2>&1 | tee $BULK/tempocnn.txt
```

- Each tool runs once per song as a separate process with a 900 s timeout (`--timeout`). A crash, a timeout or a
  refused input is written as a `failed` output with its reason, and the run goes on. A run that completes
  without beats or a tempo is `empty`, and scores 0, since that is what a user gets.
- Expected time: Ardour about 10 to 60 s per song (it runs `qm-bbt` once per beats-per-bar value the song needs,
  then `qm-tempo`), TempoCNN about 5 to 20 s per song.
- **If the first five songs all fail for a tool, stop that tool** and report; do not retry with changed settings.

### Part M: MiniBPM (only with the human's go-ahead)

`MINIBPM_GO_AHEAD: <not given>` (the main thread replaces this with the human's decision and its date)

Ardour's Lua cannot reach MiniBPM (`luabindings.cc` binds no tempo estimator, segment descriptor or clip
trigger), and the GUI shows only the rounded value, so the route is to compile Ardour 9.8's own `minibpm.cc`
(a standard-library-only file) with this Mac's existing Apple clang. While the line above says not given, run only
the check below, report its output, and skip the rest of Part M.

```sh
xcode-select -p; ls -l "$(xcode-select -p 2>/dev/null)/usr/bin/clang++"   # never run clang or xcrun if this fails:
                                                                          # that opens an install prompt
```

With the go-ahead:

```sh
MB=$BULK/minibpm && mkdir -p $MB/src/ardour
curl -fsSL -o $MB/src/minibpm.cc https://raw.githubusercontent.com/Ardour/ardour/9.8/libs/ardour/minibpm.cc
curl -fsSL -o $MB/src/ardour/minibpm.h https://raw.githubusercontent.com/Ardour/ardour/9.8/libs/ardour/ardour/minibpm.h
shasum -a 256 $MB/src/minibpm.cc $MB/src/ardour/minibpm.h
# At Ardour master d66bd6c (just after 9.8) they are 34c0bdf475d91bc4be7c2c1cff9277b7c139c2aa33f5be07100cc56b351509d8
# and 86fec5f9ce966ae05d435ee01f6f460464d1aca49ba5533ec708516c5251927e; report whether tag 9.8 matches.
clang++ --version
clang++ -std=c++11 -O3 -ffast-math -fno-finite-math-only -I $MB/src -o $MB/minibpm_cli \
  $SRC/scripts/n1/open_baselines/minibpm_cli.cc $MB/src/minibpm.cc
$PY $OB minibpm --ids P44,P48 --minibpm-bin $MB/minibpm_cli --minibpm-src-sha256 "$(shasum -a 256 $MB/src/minibpm.cc | cut -c1-64)"
$PY $OB minibpm --minibpm-bin $MB/minibpm_cli --minibpm-src-sha256 "$(shasum -a 256 $MB/src/minibpm.cc | cut -c1-64)"
```

The flags are those Ardour's `wscript` uses for clang on macOS. Optional, for the human at the GUI: drop three
canonical WAVs (for example 013, 022 and 030) into clip slots of a session at each file's rate and read the
clip's tempo tooltip ("Estimated tempo (from file) was: ..."); it should equal `mbpm-clip`'s BPM for that song.

## 6. Determinism null

```sh
$PY $OB ardour   --tag null --ids 013,022,030 --ardour-lua $ARDOUR_LUA
$PY $OB tempocnn --tag null --ids 013,022,030 --opendaw $OD --model $MODEL --harness $H
# with Part M: $PY $OB minibpm --tag null --ids 013,022,030 --minibpm-bin $MB/minibpm_cli
$PY $OB status
```

Each tool is expected to give identical outputs on a second run (`run-status.json`, `nulls`). Report any that
does not; do not average runs.

## 7. Score and report

```sh
$PY $OB score  2>&1 | tee $BULK/score.txt
$PY $OB report 2>&1 | tee $BULK/report.txt
```

- `score` rescores two old arms on its own spans and stops if either differs from the preflight
  (`$OUT/score-checks.json`); report that file if it stops.
- An arm whose runs failed on more than 4 of the 48 songs is not scored; `RESULTS.md` says which.
- Outputs in `$OUT`: `RESULTS.md`, `results.json`, `items.json`, `arms.json`, `run-status.json`,
  `open-songs.json`, `score-checks.json`, `progress.json`, `NOTES.md`, and `preflight/`.

## Stopping rule, in one place

Stop the whole run on: a prep problem (step 2), a preflight difference (step 3), or a source snapshot that is not
`$COMMIT` (step 0). Stop one tool when it fails on both probes or on the first five songs. Stop at about three
hours of wall time and report what finished. Never work around a failure by changing a tool, its parameters, the
audio or the clock rule.

## What to return

1. The commit run, the versions listed in step 0, and the wall time of each step.
2. Prep and preflight results (counts; any problems or differences, in full).
3. From `RESULTS.md`: the BL-20, level, bars_start and offset_ms rows of the tier tables (tier A hard span, tier
   B whole song, tier C hard span, both clocks), the paired-difference rows for `qm-bbt`, `qm-tempo` and `tcnn`,
   and the one-tempo arms' tempo-terms tables.
4. `run-status.json` counts, every failed run with its reason, the probe table and the determinism null.
5. Part M: the clang check output, and if it ran, the source hashes and whether tag 9.8 matched.
6. Every deviation from this brief and every script fix, with its diff and reason.
7. Anything surprising: for example a tracker whose probe offset is far from 0, a 48 kHz failure, or openDAW's
   WAV decoder rejecting the canonical WAVs (`opendaw_wav_decoder` in the TempoCNN outputs).
