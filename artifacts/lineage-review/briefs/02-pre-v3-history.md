Slice id: `02-pre-v3-history`. Report file: `artifacts/reports/lineage-review/02-pre-v3-history.md`. Scratch: `/tmp/lineage-review/02-pre-v3-history/`.

**Question.** Which paradigms came before V3 and R1 in this repository (May to mid-September 2026), how far each got, and whether leaving each was decided on evidence. The baseline README forbids using the pre-V3 mapper, timing and control stack as a reference. The review needs to know what was set aside and why, and which assets exist that V3 does not use.

**Read first.** `git log --format='%h %ad %s' --date=short main` (133 commits from 2026-05-17). `git for-each-ref refs/archive` and the logs of `refs/archive/heads/research/beatmap`, `research/beatmap-structure`, `research/token-LN-BPE`, `diffusion-test`, `exp/mel-reconstruct`, `refactor`, `codex/inference-bundle-profile-results`, `ref/prepare-for-protocol`. In `/tmp/lineage-review/trees/main`: `README.md`, `docs/formulation/README.md`, `docs/research/mel_frontend_metamer_result.md`, `docs/research/scoped_style_probe_postmortem.md`, `docs/research/source_action_representation_directions.md`, `docs/research/oracle_time_expert_question.md`; packages `src/ensomi_model/models`, `timing`, `training`, `inference`, `events`, `features`. Artifacts: `artifacts/gold_diffusion`, `artifacts/mel_metamer*`, `artifacts/runs`, `artifacts/evals`, `artifacts/reports/timing`, `artifacts/reports/evals`. The task brief names these artifact directories, so you may list and sample them; do not scan them exhaustively.

**Numbered questions.**

1. Give a table of paradigms in time order: name, dates, representation (tokens, rows, diffusion target), time representation (beat grid, tick, millisecond), audio input, model size, training data and compute, how far it got, where its code and results are now.
2. Did any earlier system produce a chart from audio end to end? For each that did: what evaluation exists (n charts, metric, any human judgment), and what quality was claimed.
3. For each paradigm that was left: was the reason evidence (which result), a bug, a resource limit, or a preference? Cite the commit, document or note that records the reason. "No recorded reason" is an answer.
4. What did the pre-V3 timing stack do (beat, tempo, onset or note-timing prediction)? What was its measured accuracy, on what data? Is it usable today as a component or as a baseline, and what would that need?
5. Which assets exist that V3 and the audio-skeleton lineage did not use: trained checkpoints that still exist on disk, tokenizers, conditioning schemes, datasets and indexes, evaluation scripts? Check existence on disk for each.
6. What did the "failed timing v3 experiment" on `exp/mel-reconstruct` and the diffusion work on `diffusion-test` try, and what do their records say about why they failed?
7. Why does the baseline README rule out the pre-V3 stack as a reference? Find the recorded reasoning. Is it a judgment about quality, about contract mismatch, or undocumented?
8. Which lessons recorded in the pre-V3 period (in documents or commit messages) were repeated as mistakes in the audio-skeleton lineage, if any? Only with a citation on both sides.
