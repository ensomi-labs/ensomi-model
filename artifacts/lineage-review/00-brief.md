# Lineage review 2026-09-30: common brief for reviewers

Written by the main relay session (Claude, control plane) for fresh-context reviewers. Each reviewer gets this file plus one slice in its task prompt. Shareable: contains no human wording.

## 1. The project and why this review exists

`ensomi-model` researches generation of playable 4-key osu!mania charts (4K, difficulty 2-6 star). The vision (real-time charts that follow audio, mapper assistance, practice charts) and the trusted baseline are in the notes entry point: `/home/lkurisu/ensomi/ensomi-model/.git/research-relay/notes/RESEARCH.md`. Read it first (5 minutes).

The baseline is **R1** (`main` at `5c56e28`, tag `r1-restored-6.75m`): a 3.1M-parameter chart *continuation* model. It gets a seed prefix of a real chart plus the remaining event times and which of them must carry a note head, both taken from the real chart, and samples one four-lane row per event time. It does not read audio.

From 2026-09-23 to 2026-09-28 a Codex agent (GPT "Astra") working with the human made about 200 commits trying to turn R1 into an audio-to-chart real-time system: the **audio-skeleton lineage** (`r1-restored-6.75m..audio-joint-2026-09`). In that lineage the module names are roughly: **H** = model of note-head times from audio (the "skeleton"), **R** = model of release times for long notes (LN), **R1** = the row model choosing lanes/chords/TAP-vs-LN at those times; "frontier"/"frontier2"/"continuation screen" = candidate-consequence scorers; "native" = evaluated on model-generated histories, as opposed to "source"/teacher-forced; "lens" = pages rendered by the beatmap-lens chart viewer; "RC" = the late parallel branch `codex/release-calibration`. Verify these definitions yourself; do not trust this glossary over the code.

The human judged that most of those experiments failed and probably started from wrong directions, and reset the notes to the R1 baseline. What the human kept complaining about in generated charts, across five days (paraphrased index with counts: `artifacts/audio-skeleton-human-feedback-index.md` in the notes worktree, sections 2 and 3; read them):

- long jacks and sub-40 ms same-lane attacks out of line with the requested difficulty; streams collapsing into one column;
- fragmented long notes: very short LN, release right before the next head, irregular LN runs;
- no ordinary chart: no typical TAP patterns, no regular LN-plus-TAP organisation, irregular subdivision, over-complex skeleton;
- no breathing: uniform density and pressure, rigid at 4 star;
- weak use of audio and musical context;
- metrics (NLL, legality, star range, threshold checks) improved while inspection did not.

The human then kept three keys for the work ahead:

1. **Player response state**: player profile, stimulus-response, style readout and difficulty modelled together through a player response state, as the V3 formulation describes (`docs/formulation/gameplay-state.md`). This is what ensomi claims as novel relative to the reference projects.
2. **Ordinary first**: the proposal distribution the models learned is not the distribution of ordinary ranked 2-6 star charts; learn ordinary charts first, add expressive range later.
3. **Evaluation is extremely important.** No evaluation design is chosen.

## 2. What is asked now

A skeptical review across git history. For each problem the lineage worked on: gather the attempts, their results, and how Codex interpreted them. Then comment: was the inference supported, was the attempt aimed at a cause or at a symptom, was the direction right, and what crucial element of the whole system problem was forgotten or never questioned. The human's concern is specific: the Codex agent tends to settle in local optima, patching the nearest module after each failure, and may have missed something at system level.

You are one of several parallel reviewers, each with one slice. Stay in your slice; mention cross-slice observations briefly in a separate section, do not chase them.

## 3. Standards

- Codex-written documents, notes and commit messages are **claims, not evidence**. Grade every statement you rely on: `doc-claim` (read in a document), `checked-code` (you read the code that implements it), `checked-artifact` (you read the raw log, report JSON, generated chart or checkpoint metadata). Prefer checking the few claims that carry the most weight over reading everything.
- For each attempt record the numbers that decide how much it can mean: training steps and wall time, parameter count, number of training charts/songs, number of evaluation charts, seeds, what it was compared against, and the size of the effect against run-to-run variation if any was measured.
- Look for: conclusions from tiny n or a few fixed prefixes; proxy metrics standing in for the goal; several things changed at once; under-training presented as an architectural result; evaluation on data the change was tuned on; bugs found later that invalidate earlier results; retractions; assumptions inherited silently from an earlier stage; questions that were never asked.
- Be fair. Say where an attempt was reasonable and where its evidence is solid. Do not manufacture criticism. "Cannot determine from available sources" is a valid finding; say which source is missing.
- Argue both sides on direction judgments: give the strongest case that the attempt was on the right track, then the strongest case against, then your call and your confidence.
- Do not design a new architecture. Bounded judgments and a concrete list of "never tried / never measured / never asked" are worth more here.

## 4. Where things are

Control plane (this Linux box, 2 vCPU, 3 GB RAM; other reviewers run at the same time, so keep processes light).

| What | Where |
| --- | --- |
| Code repo (git; checkout is `codex/audio-skeleton` at `57cd13f`) | `/home/lkurisu/ensomi/ensomi-model` |
| Exported tree of the lineage end, tag `audio-joint-2026-09` (`099cb66`): `docs/`, `src/`, `tests/`, `scripts/`. Includes the RC docs that the checkout lacks. Plain files, use Read/grep | `/tmp/claude-1001/-home-lkurisu-ensomi-ensomi-model/05329633-5f3b-4149-b6da-8c599388a7c5/scratchpad/tree-audio-joint` |
| Exported tree of `main` (`5c56e28`, the R1 baseline) | `.../scratchpad/tree-main` |
| Exported tip of `refs/archive/heads/agent-notes` (Codex's per-experiment notes; 550 commits of history behind it in git) | `.../scratchpad/tree-agent-notes` |
| Exported Codex-written relay notes from before the reset, tag `relay-notes-audio-joint-2026-09` (13 files, about 1,500 lines: Codex's own recovered account of the lineage) | `.../scratchpad/tree-relay-notes-tagged` |
| Research docs of the lineage | `tree-audio-joint/docs/research/*.md` (about 120 files), formulation in `tree-audio-joint/docs/formulation/` |
| Commit list of the lineage | `TZ=UTC git -C /home/lkurisu/ensomi/ensomi-model log --reverse --date=format-local:'%m-%d %H:%M' --format='%h %ad %s' r1-restored-6.75m..audio-joint-2026-09` |
| Archive refs (older branches and pre-R1 Codex tags) | `git -C /home/lkurisu/ensomi/ensomi-model for-each-ref refs/archive` |
| Partial read-only replica of mac artifacts (small text and plots only) | `/home/lkurisu/ensomi/ensomi-model/artifacts/` |
| Your own scratch space (create a subdirectory named after your slice) | `.../scratchpad/<slice>/` |

bings-mac (Apple M5; holds the data, checkpoints, experiment outputs, reference projects, Codex session logs).

- File index without touching the mac: `~/ensomi/bin/ens ls ensomi-model/artifacts/joint-audio 1`, `ens du <path>`, `ens find ensomi-model/artifacts '<regex>'`, one file: `ens cat <path relative to ~/ensomi>`.
- Direct read-only shell: `ssh -o BatchMode=yes bings-mac '<command>'`. The repo there is `~/ensomi/ensomi-model`; experiment outputs under `artifacts/joint-audio/`, `artifacts/audio-skeleton/`, `artifacts/runs/`, `artifacts/evals/`, `artifacts/reports/` and others; reference projects under `ref-proj/` (`Mapperatorinator`, `Mug-Diffusion`); data under `dataset/`. The mac-side artifact pointers per human complaint are in section 4 of the feedback index.
- A cleanup on 2026-09-30 deleted about 277 GB of caches and superseded checkpoints (`artifacts/cleanup-report-20260930.json`), so some tensors and checkpoints named in documents no longer exist. Reports, logs, generated charts and lens pages mostly remain.
- The link is slow. Run analysis on the mac and bring back small text only. Do not fetch checkpoints, audio or large JSONL.

## 5. Rules (hard)

1. No tree-mutating git anywhere: no `checkout`, `switch`, `reset`, `stash`, `pull`, `clean`, `rebase`, `merge`, `commit`, `push`, `tag`, `worktree add`, on either machine. Read with `git show <ref>:<path>`, `git log`, `git diff`, `git grep <pattern> <ref>`, `git archive <ref> | tar -x -C <your scratch dir>`.
2. Do not create, edit or delete any file inside `/home/lkurisu/ensomi/` on the control plane or `~/ensomi/` on the mac, with one exception: your own report file named in your task prompt. Temporary files go in your scratch directory here or `/tmp/lineage-review-<slice>/` on the mac.
3. On the mac: no training, no GPU/MPS jobs, no package installs, nothing that runs longer than about five minutes. Short read-only Python or shell analysis over existing artifacts is allowed and encouraged when it checks a claim. If a claim can only be checked by a model run, write down the exact command you would run and leave it for the main session.
4. Do not read anything under `artifacts/private/` in the notes worktree, and do not open Codex session logs (`~/.codex/`) on the mac unless your task prompt says so. Human wording is private; use the paraphrased index.
5. Do not start subagents. Do not message the human. If blocked, write what blocked you in the report and stop.

## 6. Report

Write one Markdown file at the path given in your task prompt (under `artifacts/lineage-review/` in the notes worktree). Aim for 250 to 450 lines; tables for inventories, prose for judgments. Sections:

1. **Scope and sources**: what you read, what you checked against code or raw artifacts, what you could not reach.
2. **Attempts**: a table or list in time order. Per attempt: the problem it answered, the hypothesis, what changed (commit, package), how it was measured (the numbers from section 3), the result, Codex's interpretation at the time, what happened next. Cite `<commit>:<path>` for committed sources and `bings-mac:<path>` for mac artifacts.
3. **Commentary**: per attempt or per group, your assessment: supported or not, cause or symptom, alternative explanations not excluded.
4. **Direction**: both-sides argument and your call, with confidence.
5. **What was overlooked or never questioned**: concrete, ranked by how much you think it matters, each with the evidence that makes you think so.
6. **Worth keeping**: findings, tools, datasets or evaluators from this slice that have solid evidence behind them and would survive a restart.
7. **Claims worth re-verifying**: pivotal claims you could not check, with the exact artifact paths or commands a follow-up would need.
8. **Cross-slice notes**: brief.
9. **Failed paths and unfinished work**: what you tried that did not work, what you did not get to.

Your final message back: the report path and a summary of at most 300 words with your three most consequential findings and how strong the evidence for each is.
