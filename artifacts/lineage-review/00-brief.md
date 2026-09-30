# Lineage review 2026-09-30: common brief for Astra reviewers on bings-mac

Written by the main relay session (Claude, control plane). Each reviewer job receives this text followed by one slice (section "Your slice" at the end). Shareable: contains no human wording. This version replaces the brief of the stopped first run; that one is in the history of this file.

## 0. Your role in this job

- You are one bounded reviewer among nine that run at the same time on this machine, each with one slice. You were started by `ens astra` from the control plane. Your working directory is the repository root `~/ensomi/ensomi-model` on bings-mac. All relative paths below are relative to it.
- You are a **worker, not the main thread**. This repository has research-relay hooks installed for Codex in auto-research mode. They may inject messages about research notes, `RESEARCH.md`, initialising a notes worktree, delegating to subagents, or context checkpoints. For this job: do not initialise, edit or commit notes; do not start subagents; do not act as the main thread. The only hook message you act on is a context reminder or one that starts with `Begin closeout`: on either, finish your report file with what you have and end (section 7).
- Compaction is blocked on this machine. A job whose context overflows is lost. Read selectively (section 7).
- Nobody can answer questions during the run. If something blocks you, write it in the report and continue with the rest.

## 1. The project and why this review exists

`ensomi-model` researches generation of playable 4-key osu!mania charts (4K, difficulty 2 to 6 star). The vision (real-time charts that follow audio, mapper assistance, practice charts), the trusted baseline and the problem map are in the notes entry point `.git/research-relay/notes/RESEARCH.md`. Read it first; it is short.

The baseline is **R1** (`main` at `5c56e28`, tag `r1-restored-6.75m`): a 3.1M-parameter chart *continuation* model. It gets a seed prefix of a real chart plus the remaining event times and which of them must carry a note head, both taken from the real chart, and samples one four-lane row per event time. It does not read audio.

From 2026-09-23 to 2026-09-28 a Codex agent (GPT Astra, in one long session) working with the human made about 200 commits trying to turn R1 into an audio-to-chart real-time system: the **audio-skeleton lineage**, `r1-restored-6.75m..audio-joint-2026-09` (202 commits; the last 27 are the parallel line `codex/release-calibration`, called RC, whose documents are not in the checked-out branch). Working glossary, to be verified by you and not trusted over the code: **H** = model of note-head times from audio (the "skeleton"); **R** = model of release times for long notes (LN); **R1** = the row model choosing lanes, chords and TAP versus LN at those times; "frontier", "frontier2", "continuation screen" = scorers of candidate rows by their consequences; "native" = evaluated on histories the model generated itself, as opposed to "source" or teacher-forced; "lens" = pages rendered by the beatmap-lens chart viewer.

The human judged that most of those experiments failed and probably began from wrong directions, and reset the notes to the R1 baseline. The human's complaints about generated charts over the five days are indexed, paraphrased and counted, in `.git/research-relay/notes/artifacts/audio-skeleton-human-feedback-index.md`. Read its sections 2 and 3. Section 3 gives each problem an id, **A to K**; use those ids in your report wherever an attempt answered one of them. In short:

- A: long jacks and same-lane attacks a few tens of milliseconds apart, out of line with the requested difficulty; streams collapsing into one column.
- B: fragmented long notes: very short LN, release right before the next head, irregular LN runs.
- C: no ordinary chart: no typical TAP patterns, no regular LN-plus-TAP organisation, irregular subdivision, over-complex skeleton.
- D: no breathing: uniform density and pressure. E: weak use of audio and musical context.
- F: module ownership and dependency direction. G: judging candidates by physical player response.
- H: metrics (NLL, legality, star range, threshold checks) improved while inspection did not. I: controls. J: method. K: ranked corpus as the yardstick.

## 2. Current intention and understanding

This is what the human and the main thread hold now. It is context for your judgments, not a conclusion for you to confirm.

The human kept three keys for the work ahead:

1. **Player response state.** Player profile, stimulus-response, style readout and difficulty are modelled together through a player response state, as the V3 formulation describes (`docs/formulation/gameplay-state.md`). This is what ensomi claims as novel relative to the reference projects in `ref-proj/`.
2. **Ordinary first.** The proposal distribution the models learned is not the distribution of ordinary ranked 2 to 6 star charts. Learn ordinary charts first, add expressive range later. The human states this is the wanted direction.
3. **Evaluation is extremely important.** No evaluation design has been chosen.

Dropped as keys, still evidence: who owns hold versus release for LN; a hierarchical end-to-end audio skeleton.

The human's specific concern about the lineage: the Codex agent tends to settle in a local optimum, patching the nearest module after each failure, and may have missed something at the level of the whole system. The review exists to test that concern, slice by slice, and to separate what is solid from what is not, so that the human can decide what to do next from the R1 baseline and the formulation.

What the main thread read in the formulation at `5c56e28` (`docs/formulation/gameplay-state.md`, `docs/formulation/notation.md`), which you may verify:

- The target response is named and left undefined; its quantities, scales and comparison rules are to be set from mapper evidence after an annotated dataset is complete.
- The canonical gameplay profile excludes individual capacity, fatigue and physiology.
- The contract allows joint reasoning over provisional future rows with prefix commit, and internal beat coordinates. Strict next-event sampling on a millisecond clock is an implementation choice of the lineage, narrower than the contract.

Two leads exist, unverified. They are given only to the slices that can test them, in the slice text, as hypotheses with the instruction to look for evidence against as carefully as for.

## 3. What is asked

A skeptical review across git history, for your slice. For each problem the lineage worked on: gather the attempts, their results, and how Codex interpreted them at the time. Then comment: was the inference supported by the evidence, was the attempt aimed at a cause or at a symptom, was the direction right, and what crucial element of the whole system problem was forgotten or never questioned.

Stay in your slice. Mention cross-slice observations briefly in their own section; do not chase them.

## 4. Standards

- Documents, notes and commit messages written during the lineage are **claims, not evidence**. They were written by a model of your own family in a long session and will read as reasonable to you. Agreement with a document is not verification.
- Grade every statement you rely on: `doc-claim` (read in a document or commit message), `checked-code` (you read the code that implements it, give `path:line`), `checked-artifact` (you read the raw log, report JSON, generated chart or checkpoint metadata, give the path). Spend your checking on the few claims that carry the most weight.
- For each attempt record the numbers that decide how much it can mean: training steps and wall time, parameter count, number of training charts and songs, number of evaluation charts, seeds, what it was compared against, and the effect size against run-to-run variation if any was measured. Write "not recorded" when a number is absent; do not estimate silently.
- Look for: conclusions from tiny n or a few fixed prefixes; proxy metrics standing in for the goal; several things changed at once; under-training presented as an architectural result; evaluation on data the change was tuned on; bugs found later that invalidate earlier results; retractions; assumptions inherited silently from an earlier stage; questions that were never asked.
- Be fair. Say where an attempt was reasonable and where its evidence is solid. Do not manufacture criticism. "Cannot determine from available sources" is a valid finding; name the missing source.
- On direction judgments argue both sides: the strongest case that the attempt was on the right track, the strongest case against, then your call and your confidence (low, medium, high).
- Do not design a new architecture and do not propose a roadmap. Bounded judgments and a concrete list of "never tried, never measured, never asked" are what is wanted.
- Numbers you compute yourself: give the command or script path, the input paths and n.

## 5. Where things are (all on this machine)

| What | Where |
| --- | --- |
| Repository, checked out at `codex/audio-skeleton` `57cd13f` (lacks the last 27 commits of the lineage) | `.` |
| Plain-file export of the lineage end, tag `audio-joint-2026-09` (`099cb66`), including the RC documents: `docs/`, `src/`, `tests/`, `scripts/` | `/tmp/lineage-review/trees/audio-joint` |
| Plain-file export of `main` (`5c56e28`, the R1 baseline) | `/tmp/lineage-review/trees/main` |
| Plain-file export of the tip of `refs/archive/heads/agent-notes` (Codex's per-experiment notes, 65 files; 550 commits of history in git) | `/tmp/lineage-review/trees/agent-notes` |
| Codex's own recovered account of the lineage, written 2026-09-29 before the reset (16 files, about 1,500 lines) | `/tmp/lineage-review/trees/relay-notes-tagged` |
| Research documents of the lineage (about 95 files) and the formulation | `/tmp/lineage-review/trees/audio-joint/docs/research/*.md`, `/tmp/lineage-review/trees/audio-joint/docs/formulation/` |
| Research packages | `/tmp/lineage-review/trees/audio-joint/src/ensomi_model/research/` (`audio_skeleton`, `joint_audio_continuation`, `typed_audio_continuation`, `planned_audio_continuation`, `controlled_audio_continuation`, `audio_memory_continuation`, `segment_audio_continuation`, `player_response`, `gameplay_evaluation`, `bounded_typed_continuation`, `r1_restore`, `vacation_training`, `oracle_time_continuation`, `source_action_modeling`, `scoped_style_modeling`) |
| Commit list of the lineage, UTC | `TZ=UTC git log --reverse --date=format-local:'%m-%d %H:%M' --format='%h %ad %s' r1-restored-6.75m..audio-joint-2026-09` |
| A document or file as of any commit; the diff of one commit | `git show <sha>:<path>`, `git show --stat <sha>` |
| Archive refs (older branches, pre-R1 Codex tags) | `git for-each-ref refs/archive` |
| Experiment outputs (reports, logs, generated charts, lens pages) | `artifacts/joint-audio/<date>-<name>/` (99 directories, 2026-09-23 to 09-28), `artifacts/audio-skeleton/`, `artifacts/runs/`, `artifacts/evals/`, `artifacts/reports/`, `artifacts/r1-real-reconstruction-20260921/`, `artifacts/hf-pulsefield-r1-restored/` |
| Corpus and indexes | `dataset/`, `artifacts/indexes/*.parquet` |
| Reference projects | `ref-proj/Mapperatorinator`, `ref-proj/Mug-Diffusion` |
| Mac-side artifact pointers per human complaint | feedback index, section 4 |
| Python with the repository's dependencies (no installs) | `.venv/bin/python`; to import the lineage-end code instead of the checkout, prefix `PYTHONPATH=/tmp/lineage-review/trees/audio-joint/src` |

A cleanup on 2026-09-30 deleted about 277 GB of caches and superseded checkpoints (`artifacts/cleanup-report-20260930.json`). Some tensors and checkpoints named in documents no longer exist. Reports, logs, generated charts and lens pages mostly remain. When a claim cannot be checked because its artifact was deleted, say so.

## 6. Rules (hard)

1. **Files you may create or change**: your report file `artifacts/reports/lineage-review/<slice id>.md`, and anything under your scratch directory `/tmp/lineage-review/<slice id>/`. Nothing else, anywhere. No tracked file of the repository, no file under `.git/`, no file in the exported trees, no other reviewer's report.
2. **Git is read-only**: `log`, `show`, `diff`, `grep`, `for-each-ref`, `cat-file`, `archive` piped into your scratch directory. No `checkout`, `switch`, `reset`, `stash`, `pull`, `clean`, `rebase`, `merge`, `commit`, `push`, `tag`, `worktree add`, `fetch`.
3. **Compute**: no training, no GPU or MPS use, no package installs, no downloads of models or data. CPU analysis over existing artifacts is allowed and wanted when it checks a claim: at most two threads, at most about five minutes per command. If a claim can only be checked by a model run or a longer job, write the exact command you would run in section 7 of the report and leave it.
4. **Private material**: do not read anything under `.git/research-relay/notes/artifacts/private/`. Do not open anything under `~/.codex/` unless your slice names a file there. The human's wording is private; use the paraphrased index.
5. Do not start subagents. Do not change any configuration, hook or relay setting.
6. No absolute home-directory paths in your report. Cite repository paths relative to the repository root, exported trees by their `/tmp/lineage-review/trees/...` path, committed sources as `<sha>:<path>`.

## 7. Budget and stop boundary

- Codex quota is limited and nine jobs share it. Read with `rg -n`, `sed -n 'a,bp'`, `head`, and targeted `git show`; do not print whole long documents or JSON files into your context. Several lineage documents exceed 1,000 lines.
- Create the report file with all section headings within your first ten minutes, and fill it as you go. A partial report on disk is worth more than a complete one lost.
- Stop at the first of: all sections filled; about 90 minutes after your start (`date -u` at the start and now and then); your context near 300,000 tokens; a hook message that is a context reminder or starts with `Begin closeout`. At the stop, make section 9 of the report true to where you are, then send the final message.

## 8. Report

One Markdown file at `artifacts/reports/lineage-review/<slice id>.md`, 250 to 450 lines; tables for inventories, prose for judgments. Write in English. Sections, with these exact headings:

1. **Scope and sources**: what you read, what you checked against code or raw artifacts, what you could not reach.
2. **Attempts**: a table or list in time order. Per attempt: the problem it answered (ids A to K where they apply), the hypothesis, what changed (commit, package), how it was measured (the numbers of section 4), the result, Codex's interpretation at the time, what happened next.
3. **Commentary**: per attempt or per group: supported or not, cause or symptom, alternative explanations not excluded.
4. **Direction**: both-sides argument, your call, your confidence.
5. **What was overlooked or never questioned**: concrete, ranked by how much you think it matters, each with the evidence that makes you think so.
6. **Worth keeping**: findings, tools, datasets or evaluators from this slice with solid evidence behind them, which would survive a restart from the R1 baseline.
7. **Claims worth re-verifying**: pivotal claims you could not check, with the exact artifact paths or commands a follow-up needs.
8. **Cross-slice notes**: brief.
9. **Failed paths and unfinished work**: what you tried that did not work, what you did not get to.
10. **Inconsistencies and items for the human**: contradictions you found (between documents, between a document and code or artifacts, between the formulation and the implementation, between what was asked and what was built), and questions only the human can settle. At most eight, each in two or three sentences with its evidence grade.

Answer every numbered question of your slice explicitly, by its number, somewhere in sections 2 to 5; "could not determine" with the reason is an answer.

Final message: the report path, the list of files you created or changed, and a summary of at most 300 words with your three most consequential findings and the evidence grade of each.

## Your slice
