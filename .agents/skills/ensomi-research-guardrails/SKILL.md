---
name: ensomi-research-guardrails
description: Recurring failure patterns of agents in Ensomi research, each with the guardrail that answers it. Use when planning, briefing, launching or reading an experiment, training run, evaluation or design round; when choosing a measure, baseline, checkpoint or decision rule; when writing a claim others will cite; and when writing a brief for a subagent or Astra.
---

# Ensomi research guardrails

From the September lineage to the R2 bake-off of 2026-10-07, agents repeated the same failures under new names. A written list of them stopped working once it dropped out of the agents' context. This skill keeps the patterns in context.

The guardrails support judgment; they are not a form to fill. Apply each one whose trigger holds, and say in the plan or report how you applied it. Nothing here authorises a run. Training runs still need the human's check.

## Guardrails

1. **A measure decides only after it is shown to see what the human sees.**
   - Trigger: a measure, guard, score or evaluator is used to select, rank, stop or judge.
   - Do: test it against the human labels on hand (X0 collapse calls, Lens concept labels, the C0 screen) and report AUC or F1 with an interval.
   - Keep the labels used to design a measure apart from those used to validate it.
   - If no measure is validated, the measures only characterise, and the human's screen decides.
   - Seen: gates tuned per complaint passed 7 of 12 charts the human rejected (09-28). Guards calibrated on a 16-chart short panel selected the 56M checkpoint (10-07). Round 1 was judged on G after X0 validated no measure (10-07).
2. **Know the noise before setting a margin.**
   - Trigger: a comparison between arms, checkpoints, seeds or recipes.
   - Do: measure the seed spread and the neighbouring-checkpoint spread of the decision statistic. Run the statistic on a null: seed against seed, source against source.
   - Never use a selected checkpoint as the only baseline.
   - Seen: round 1 set a 0.15 G margin against a checkpoint SD of 0.24-0.29, with the selected 56M as the only baseline (10-07).
3. **Show that a new input is used before training on it.**
   - Trigger: a new conditioning input, reader, latent or memory path.
   - Do: run a teacher-forced ablation on trained weights and report ΔNLL. Initialisation-time tests show only that the path is connected.
   - Seen: the star condition was unused after 158M exposures (10-04). θ carried 0.0002 nats in B3 (10-08).
4. **Work out what the fixed inputs imply.**
   - Trigger: a design, a diagnosis, or a new measure.
   - Do: list each fixed input or choice: supplied head rows, warm start, selected baseline, panel, decision unit, scope exclusions. For each, state:
     - what the model can and cannot learn under it;
     - what the evaluation can and cannot see;
     - which of the human's complaints it could cause or hide.
   - Out of scope to build is not out of scope to analyse.
   - Seen: R1 read release times supplied by the source chart (09-23). Head rows predicted star with R² 0.75, so the star input went unused (10-04). Nobody analysed what supplied head rows imply until the human asked (10-08).
5. **Put the human's contradicting evidence to the human.**
   - Trigger: before a launch or a consequential decision, including one inside a delegated night.
   - Do: when the human's latest evidence contradicts the plan's premise, state the contradiction in a few lines and let the human decide.
   - If the human cannot be reached, run only measurements (nulls, ablations, noise runs), not arms that decide.
   - Labelling a choice "for the human's review in the morning" does not make it the human's decision.
   - Seen: the R2 v1 spec and a 20-hour run were launched while the human slept (10-03). The X0 marks were early and local while round 1 targeted identity past row 511; the main thread noticed and launched unchanged (10-07).
6. **Turn your own warning into a test before you spend on the action.**
   - Trigger: a risk you or a reviewer recorded next to a planned action.
   - Do: run the cheap test that would show the risk, or drop the action.
   - Seen: "the reader may stay at zero" was written, and B2 and B3 were trained as written (10-07).
7. **Repair upstream when the cause is upstream.**
   - Trigger: a third repair in the module nearest a symptom, or one new input per symptom.
   - Do: stop and name the upstream choice that produces the symptoms. Keep each recognised open question visible until it is settled or the human parks it.
   - Seen: a 60 ms mask, then `ln_level`, `ln_length` and θ for one LN-drift problem (10-06/07). "Whether head times stay given" was open from 10-03 and never settled.
8. **Claim no more than the comparison carries.**
   - Trigger: a causal sentence, a shareable page, or a report.
   - Do:
     - compare matched rows only;
     - give no ratio without its denominator;
     - name the strongest alternative explanation;
     - check a claim for necessity, not for its presence in the notes;
     - keep "input used", "process improves" and "charts improve" apart.
   - Seen: "own history amplifies" was measured against unmatched history (10-07). A signed ratio sat on a 0.007 denominator (10-08). "CE never penalises" was caught only by an external review (10-07).
9. **Keep the early warning.**
   - Trigger: cutting in-run evaluation, mini-panels or kill checks to fit wall-clock.
   - Do: keep at least one kill rule that can fire, with a reference scale.
   - Seen: round 1 dropped its mid-run panels, and B3 trained to the end with an unused reader (10-07).

A review that checks code against its brief cannot catch what the brief left out. Pre-launch reviews passed the signed G and the per-resample interval, and an external review caught them (10-07). Apply the guardrails in the plan itself; a review step is no substitute.

## In briefs

Name this skill in every brief for a subagent or Astra, and say which guardrails apply to the task. Do not copy the list into the brief.

## Keeping it current

When a failure recurs, add the instance to the guardrail it belongs to. Add a guardrail only for a pattern none of these covers. Keep the list short enough to apply.

The evidence behind each instance is in the research notes on the `relay-notes` branch: `artifacts/agent-failure-modes.md` (2026-10-02), `artifacts/r2-round1-agent-failures-20261008.md` and `artifacts/r2-represent-history-20261008.md`.
