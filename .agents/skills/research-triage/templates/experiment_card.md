<a id="card-<card-id>"></a>
### Experiment Card: <card-id>

#### Identity and Authority

- Card ID:
- Revision:
- Status: proposed | accepted (link the accepting decision)

Only a human's explicit acceptance of this exact revision makes it `accepted`. Any protected-field edit increments the revision and returns the card to `proposed`.

#### Question and Hypothesis

- Research question:
- Selected hypothesis, claim kind, and mechanism:
- Problem property and out-of-metric prediction, for a research claim:
- Decision the result should change:

#### Analogues and Branch Selection

- Domain and technique analogues, with sources:
- Shared mechanism:
- Remaining difference:
- Alternative branches rejected:
- Selection criterion:

#### Fixed Comparison

- Baseline product source revision (full clean OID):
- Dataset slice and identity:
- Baseline config and checkpoint identities:
- Baseline run or evidence identity (a matched baseline for a research claim):
- Baseline metric value, aggregation, sample count, and uncertainty:
- One causal intervention:
- Behavior-neutral instrumentation:
- Files likely to change:
- Read-only context:

#### Evidence and Decision Rule

- Primary metric definition and direction:
- Decision threshold (for a research claim, also on the out-of-metric prediction):
- Regression guard and bound:
- Qualitative check:
- Positive result:
- Negative or null result, given a passed sensitivity check:
- Ambiguous result:

#### Reproduction and Bounds

- Exact command or procedure:
- Comparison design: pairing or independence, matched baseline, and the sensitivity check's controls (mechanism ablation, null control or known-effect control, noise estimate and strata):
- Seeds:
- Environment, dependencies, and hardware:
- Runtime, compute, storage, and network bounds:
- Fresh output destination:
- Overwrite and resume policy:
- Early-stop or kill condition:
- Expected failure modes:
- Confounders:
- Conclusion scope:

Protected fields are the question, baseline identity and value, dataset slice, causal intervention, primary metric, decision threshold, regression guard, procedure, comparison design, seeds, environment, resource bounds, output and resume policy, and stop condition.
