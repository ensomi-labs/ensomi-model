<a id="result-<result-log-id>"></a>
### Result Log: <result-log-id>

#### Experiment and Reproduction

- Experiment Card ID and accepted revision (`none` if exploratory or no Card):
- Baseline source revision and worktree state:
- Intervention source revision and worktree state:
- Baseline run or evidence IDs:
- Intervention run IDs:
- Baseline command or procedure:
- Intervention command or procedure:
- Environment, dependencies, and hardware:
- Baseline and intervention seeds:
- Dataset slice and identity:
- Baseline config and checkpoint identities:
- Intervention identity:
- Output artifact paths or stable IDs:
- Output collision, overwrite, and resume disposition:
- Budget consumed:
- Stop reason:

#### Results

- Baseline primary-metric value, aggregation, sample count, and uncertainty:
- Intervention primary-metric value, aggregation, sample count, and uncertainty:
- Decision threshold result, including the out-of-metric prediction for a research claim:
- Regression guard value and bound:
- Qualitative observations:
- Sensitivity check result (passed | failed | not run), failures, and missing evidence:

#### Plan Conformance

- Planned-versus-actual deviations:
- Protected fields changed, if any:
- Deviation disposition: none | behavior-neutral | material

#### Evaluation

Leave this section pending for a run-only request. Fill or revise it during Evaluate.

- Observation:
- Interpretation:
- Strongest alternative explanation (for a research claim, whether the matched baseline rules out the generic explanation):
- Confounders:
- Conclusion scope and claim kind supported (research | engineering | none):

#### Decision

Leave this section pending for a run-only request. Fill or revise it during Evaluate.

- Recommended outcome: DROP | REFINE | REPEAT | SUPPORTED
- Evidence supporting the outcome:
- Revised question or next discriminating test:
- Human direction still required:

Without an accepted card and identifiable baseline and guard, or after a material protected-field deviation, do not recommend `SUPPORTED`. Do not recommend `DROP` on a negative or null result whose sensitivity check has not passed.
