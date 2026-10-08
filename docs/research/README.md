# Research documents

These documents describe code on this branch or on the research line they name.
They record implementations and measurements. The
[formulation](../formulation/README.md) owns the V3 specification.

| Document | Covers |
| --- | --- |
| [Bounded typed continuation](bounded_typed_continuation.md) | the R1 task, model, training stages and generation commands |
| [Staged R1 restoration](r1_staged_restoration.md) | the six-stage recipe behind the released checkpoint |
| [Vacation training queue](vacation_training.md) | unattended multi-day queue: audio cache, 35M teacher, stress runs |
| [R1 training distribution](r1_training_distribution.md) | difficulty and long-note exposure of the training plan |
| [Oracle-time continuation](oracle_time_continuation.md) | row replay, corpus cache and the causal backbone R1 builds on |
| [Head-time decomposition](head_time_decomposition.md) | measurements on the R2 line (branch `r2/train`) of what supplied head times carry, and the evidence on the decomposed problem's three assumptions |
| [4K style tag reference](osu_mania_4k_style_tag_reference.md), [community tag semantics](osu_mania_community_tag_semantics.md) | the style vocabulary behind the formulation's style observations |

## Earlier work

| Period | Line of work | Where |
| --- | --- | --- |
| 2026-05 to 2026-08 | mapper v2 and v2.1, timing grid fitting, Control V3, websocket inference service | branch `legacy/v2` |
| 2026-09-13 to 2026-09-15 | scoped style probes, source-action modeling | branch `legacy/v2`, `docs/research/scoped_style_*` and `source_action_*` |
| 2026-09-15 to 2026-09-21 | oracle-time continuation, R1 stages and release | this branch, tag `r1-restored-6.75m` |
| 2026-09-23 to 2026-09-29 | audio-conditioned joint model | tag `audio-joint-2026-09` |

Experiments on one problem are usually spread over many commits, so tags mark
only the end of a line of work. The tag `audio-joint-2026-09` points at a commit
whose parents are all three branch tips of that period, and its message lists
the phases with their commit ranges. The index by research question, with the
human decisions that steered it, is
[`RESEARCH.md` on `relay-notes`](https://github.com/ensomi-labs/ensomi-model/blob/relay-notes/RESEARCH.md).

## R1 stage commits

Each stage of the R1 lineage ended at a commit on this branch's history.

| Commit | Stage |
| --- | --- |
| `89d5379` | oracle-time runtime and formulation review |
| `ca28511` | bounded typed continuation, difficulty coverage at 4M exposures |
| `4a98c02` | candidate action-consequence residual |
| `7fdeb7a` | persistent observed seed |
| `c1cd14c` | long-form landmark memory and head routing |
| `dfdbc75` | release routing |
| `db0a9b8` | row response |
| `3500eac` | vacation training queue |
| `cdbc687` | staged restoration; training revision of the released checkpoint |
| `8f13103` | release, tag `r1-restored-6.75m` |

## Archived branches

Branches that ended without a result on this branch are kept on GitHub under
`refs/archive/`, outside the branch and tag lists. Fetch them with:

```sh
git fetch origin 'refs/archive/*:refs/archive/*'
git log refs/archive/heads/codex/release-calibration
```

| Ref under `refs/archive/heads/` | Contents |
| --- | --- |
| `codex/audio-skeleton`, `codex/release-calibration`, `codex/stream-generation-benchmark` | the 2026-09-23 to 09-29 audio lineage, also reachable from `audio-joint-2026-09` |
| `agent-notes` | 64 agent-written experiment notes, 2026-09-14 to 09-29 |
| `codex/r1-response-calibration` | one calibration experiment on the R1 response stage |
| `exp/mel-reconstruct` | timing v3 experiment, 2026-08, recorded as failed |
| `research/beatmap`, `research/beatmap-structure`, `research/token-LN-BPE` | 2026-06 tokenization and BPM-ramp research |
| `diffusion-test` | 2026-09-10 diffusion probe |
| `codex/inference-bundle-profile-results`, `codex/skill-workflow-repair`, `skills/borrow-dsh`, `chore/cleanup`, `refactor`, `refactor-better-readability-backup`, `ref/prepare-for-protocol` | engineering branches from 2026-05 to 2026-08 |

`refs/archive/tags/` keeps the former `archive/*` tags as markers.
