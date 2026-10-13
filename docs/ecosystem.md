# Facts the Ensomi repositories share

Six repositories and one Hugging Face repository make up Ensomi. Several facts
are stated in more than one of them: the released model's files and revision,
licences, versions, and the rules of generation. Each such fact has one owner,
the place that defines it. This page names the owner and every place that
restates the fact.

Change a fact at its owner, then every place listed here that restates it, in
the same round of pull requests;
[`ensomi-cross-repo-changes`](../.agents/skills/ensomi-cross-repo-changes/SKILL.md)
says how. A change that states a shared fact somewhere new adds that place
here, or links to the owner instead.

Paths are relative to each repository's root; `chart.proto` is protocol
`proto/ensomi/protocol/v1/chart.proto`, and the app's Swift files are under its
`Sources/`. In the Ensomi workspace the repositories are checked out side by
side: `../ensomi-model`, `../ensomi-runtime`, `../ensomi-protocol`, `../ensomi`,
`../ensomi-web` and `../beatmap-lens`.

## Repositories

| Repository | Holds | Default branch | Licence |
| --- | --- | --- | --- |
| [ensomi-model](https://github.com/ensomi-labs/ensomi-model) | models, training, the released system, the model service, the ONNX exporter, this page | `main` | AGPL-3.0-only |
| [runtime](https://github.com/ensomi-labs/runtime) | the Rust core every client binds to, the model-package format, the Swift package `EnsomiRuntime` and the npm package `@ensomi/runtime` | `main` | Apache-2.0 |
| [protocol](https://github.com/ensomi-labs/protocol) | EnsomiProtocol, the messages programs exchange | `main` | Apache-2.0 |
| [ensomi](https://github.com/ensomi-labs/ensomi) | the macOS and iOS app | `dev/swift` | Apache-2.0 |
| [ensomi-web](https://github.com/ensomi-labs/ensomi-web) | the site at [ensomi.app](https://ensomi.app) | `main` | MIT |
| [beatmap-lens](https://github.com/ensomi-labs/beatmap-lens) | the chart renderer and annotation tools | `main` | MIT |
| [`sed-i/ensomi-r2`](https://huggingface.co/sed-i/ensomi-r2) (Hugging Face) | the released model's files and its card | `main`, tag `r2-1.0` | CC BY-NC-SA 4.0; BeatThis's files MIT |

## The released model

| Fact | Owner | Restated in |
| --- | --- | --- |
| Identity: id `ensomi-r2`, name "Ensomi R2", version `r2-1.0` | ensomi-model `src/ensomi_model/export/package.py` (`MODEL_ID`, `MODEL_NAME`, `MODEL_VERSION`) | the manifest's `model`; ensomi-model `src/ensomi_model/service/mapping.py` (imports them); runtime `crates/generate/src/capabilities.rs` (a copy); prose in every repository |
| The release's code tags | ensomi-model tags `r2-1.0`, `rows-1.0` and `timing-1.0` | ensomi-model `README.md`, `docs/research/r2_system.md`; the card; ensomi-web `public/index.html` |
| The revision clients download | the Hub tag `r2-1.0` | app `Sources/EnsomiCore/Services/Beatmap/ChartModelDownloader.swift` (`ChartModelSource.revision`); ensomi-web `site/app.ts` (`HUB`) with the cache name in `site/worker.ts`; `hf download ... --revision r2-1.0` in ensomi-model `README.md` and `docs/research/r2_system.md`, the card and ensomi-web `public/index.html` |
| The package's files, their sizes and hashes | the manifest, `onnx/ensomi-package.json` on the Hub | `SHA256SUMS` on the Hub; the card's "Files"; the total, "about 120 MB", in the card, app `README.md` and `docs/solo-device.md`, ensomi-web `README.md` and `public/index.html` |
| The manifest format, `ensomi-package/1` | runtime `docs/model-package.md` and its reader | ensomi-model `src/ensomi_model/export/package.py` (the writer) and `docs/guides/onnx_runtime.md` |
| The config format, `ensomi-onnx-runtime-v1` (`runtime.json`) | ensomi-model `src/ensomi_model/export/onnx.py` (the writer) | runtime `crates/generate/src/models.rs` and `js/src/models.js` (the readers) |
| The files `runtime.json` names that a package needs | runtime `Models::required_files` and `Models::notice_files` (`crates/generate/src/models.rs`), `model_files` (`crates/ffi/src/generator.rs`) | ensomi-model `export/package.py` `config_files`, which writes the manifest of a package exported before manifests existed |
| Release figures: parameters, training corpus, holdout timing, blind screens, lineage, check environment | ensomi-model `docs/research/r2_system.md` | the card; ensomi-model `README.md` and `docs/research/README.md` |

## Licences

| Fact | Owner | Restated in |
| --- | --- | --- |
| The model's files are CC BY-NC-SA 4.0, except BeatThis's ONNX export and front-end arrays (`onnx/beatthis.onnx`, `onnx/beatthis-mel-filter.bin`, `onnx/beatthis-hann-window.bin`), which keep BeatThis's MIT licence | the card's "Licence" and the manifest's `license` (`CC-BY-NC-SA-4.0 AND MIT`) | ensomi-model `README.md` and `docs/research/r2_system.md`; runtime `README.md` and `js/README.md`; app `README.md`, `docs/solo-device.md` and Settings (`ChartModelManagementView.swift`); ensomi-web `README.md` and `public/index.html` |
| BeatThis's licence notice, `BeatThis-LICENSE.txt` | ensomi-model `src/ensomi_model/export/beatthis_graph.py` (copies it and names it in `runtime.json`) | the manifest's `notices`; the card; runtime `notice_files`; the app and ensomi.app show its text |
| Each repository's code licence | that repository's `LICENSE` | the table above; the card; ensomi-web `README.md` and `public/index.html` |
| ONNX Runtime's licence and notices | runtime `licenses/onnxruntime/` | runtime `swift/README.md`; the app's acknowledgements; ensomi-web `public/index.html` for onnxruntime-web |

## Versions and pins

| Fact | Owner | Pinned or restated in |
| --- | --- | --- |
| Runtime version | runtime `Cargo.toml` and `js/package.json` (the release workflow requires them equal), tag `vX.Y.Z` | app `project.yml` (`exactVersion`), `Package.resolved` and `README.md`; ensomi-web `package.json` (exact); runtime `swift/README.md` and `RELEASING.md` (`from:`), the generator line of `conformance/generate/native-charts/*.osu`; ensomi-model `docs/research/r2_system.md` |
| Protocol version | protocol `package.json` and `pyproject.toml`, tag `vX.Y.Z` | ensomi-model `pyproject.toml` (`service` extra, by commit); app `project.yml` and `Package.resolved`; protocol `README.md`; the runtime's copies of the messages (`crates/generate/src/capabilities.rs`, `crates/ffi/src/generator.rs`, `js/src/types.js`, `js/src/index.d.ts`) |
| protobuf range of the Python package | protocol `pyproject.toml` | ensomi-model `pyproject.toml`, whose `service` extra pins one version inside it and inside onnx's range |
| onnx and onnxruntime for Python | ensomi-model `pyproject.toml` (`export` extra) | — |
| ONNX Runtime, native | runtime `crates/generate-ort/Cargo.toml` (`ort`) | runtime `crates/generate-ort/examples/providers.rs` and `crates/generate-ort/src/bin/parity.rs` (printed) |
| onnxruntime-web | runtime `js/package.json` | ensomi-web `package.json`, `site/worker.ts` (the CDN URL of the `.wasm`, which must match the bundled JavaScript) and `README.md` |
| Python | ensomi-model `pyproject.toml` (`requires-python`) | `--python 3.10` in ensomi-model `README.md` and the report, the card and ensomi-web `public/index.html` |
| Platforms: macOS 14, iOS 17, Apple silicon; browsers | runtime `Package.swift` and `README.md` (browsers); app `project.yml` | runtime `swift/README.md` and `js/README.md`; app `README.md`; ensomi-web `README.md` and `public/index.html` |
| ONNX opset 17 | ensomi-model `src/ensomi_model/export/onnx.py` | the card; ensomi-model `docs/guides/onnx_runtime.md` |

## Generation

| Fact | Owner | Restated or applied in |
| --- | --- | --- |
| Star range: at least 2, below 6 | ensomi-model `src/ensomi_model/system/make.py` | ensomi-model `service/mapping.py`; runtime `capabilities.rs` and `crates/generate/src/generate.rs`; protocol `chart.proto`; ensomi-model `README.md` and the report; the card; app `docs/architecture.md`; ensomi-web `public/index.html` |
| Controls and their statuses | ensomi-model `docs/research/r2_system.md`, which holds the evidence | ensomi-model `service/mapping.py` (`CONTROLS`) and runtime `capabilities.rs`, word for word; the card; ensomi-model `README.md`; the clients read statuses from the capabilities |
| Sections last 8 to 32 s, lie inside the song and do not overlap | protocol `chart.proto` ("R2: 8 to 32 seconds"); runtime `generate.rs` enforces it for every section | ensomi-model `system/controls.py` (pattern scopes) and `service/mapping.py`; app `Mania4KChartGenerationModel.swift` and `docs/architecture.md`; ensomi-web `site/app.ts` and `public/index.html` |
| Arrangement passes: one in the runtime and the service, two in the release's `make` | runtime `generate.rs`; ensomi-model `system/make.py` (`DIFFICULTY_PASSES`) and `service/mapping.py` (`max_difficulty_passes=1`) | the card; the report; runtime `README.md`, `js/README.md`, `swift/README.md` and `conformance/generate/README.md`; ensomi-web `packages/runtime/README.md` |
| Random numbers: the runtime's `mulberry32-u53-v1`; R2's seed offset 954 | runtime `crates/generate/src/random.rs`; ensomi-model `system/make.py` (`R2_SEED`, exported into `runtime.json`) | runtime `generate.rs` (954 written again); the card; ensomi-web `public/index.html` and `packages/runtime/README.md`; ensomi-model `docs/guides/onnx_runtime.md` |
| Seed 0 | protocol `chart.proto`: the default seed, a seed like any other | the runtime, the service and the app treat 0 as a seed (the app's blank field means random); ensomi-web draws a random seed for 0 |
| Whole-millisecond times | protocol `chart.proto`; ensomi-model `system/make.py` | ensomi-model `README.md`, the report and `service/mapping.py`; the card; app `docs/architecture.md` |
| Phases: loading, timing, rows, arrangement, export | protocol `chart.proto` (`GenerationPhase`) | ensomi-model `service/server.py`; runtime `crates/ffi/src/generator.rs`, `js/src/types.js` and `swift/README.md`; the report; app `docs/solo-device.md`; ensomi-web `site/app.ts` and `public/index.html` (which label loading "Prepare") |
| Error codes | protocol `chart.proto` (`ErrorEvent`) | ensomi-model `service/mapping.py`, `service/server.py` and the report; runtime `generate.rs`, `crates/ffi/src/generator.rs` and `swift/README.md`; app `docs/solo-device.md` and `ChartGenerationClient.swift` |
| `.osz` layout: `generated.osu`, `audio.<ext>`, `receipt.json` | ensomi-model `system/make.py` | runtime `crates/osz` and `js/src/export.js`, which each write it; runtime `swift/README.md`; app `docs/solo-device.md` |
| Export name `Artist - Title (ensomi R2 s<seed>)` | ensomi-web `site/app.ts` | app `Mania4KChartGenerationModel.swift` and `docs/solo-device.md` |

## The model service

| Fact | Owner | Restated in |
| --- | --- | --- |
| Command, host `127.0.0.1`, port 8765, cache directory | ensomi-model `src/ensomi_model/service/server.py` | the report; app `README.md`, `docs/architecture.md` and `ChartGenerationClient.swift` (`ENSOMI_SERVICE_URL`, then `ws://localhost:8765`) |
| Port 8765 is also the default of the older inference endpoint on `legacy/v2`, which the app's `InferenceEndpointWebSocketClient` uses | ensomi-model branch `legacy/v2` | ensomi-model `README.md` and `AGENTS.md` |

## Claims and their measurements

A size, speed or compatibility is stated with the measurement or check behind
it. A claim that nothing checks is not made.

| Claim | Backed by | Stated in |
| --- | --- | --- |
| 8-12 s for a 2-3.3 minute song, Python, one CPU thread of an Apple M5 | ensomi-model `docs/research/r2_system.md` | ensomi-model `README.md`; the card |
| About 6 s, about 13 s without WebGPU, for a 3:20 song on ensomi.app | measured on the retired TypeScript runtime and on a core older than runtime 0.2.0; not measured since | ensomi-web `public/index.html` |
| 7.30 s to decode and generate a 133 s song in the app (CPU) | app `docs/solo-device.md` | — |
| A generated chart opens in osu! | nothing checks it | ensomi-model `docs/research/r2_system.md`; ensomi-web `public/index.html` |

## Two sources today

These facts are defined in two places. Until each has one owner, a change
edits both.

- **Model identity, controls and statuses, star range:** ensomi-model's service
  and the runtime's `capabilities.rs` each define them. They are to move into
  the package, read by the runtime and the service.
- **Section rules:** ensomi-model `system/controls.py` checks pattern scopes;
  the runtime checks every section. The service and the app follow the
  runtime's rule.
- **R2's seed offset 954:** the runtime writes the literal instead of reading
  `constants.system.make.R2_SEED` from `runtime.json`.
- **The `.osz` layout:** written by Python, Rust and JavaScript.
- **onnxruntime-web:** ensomi-web's CDN URL repeats the version the runtime's
  package depends on.
