# Temporal gameplay regression evaluation

The evaluation package supplements scoped Lens inspection with repeatable
whole-trajectory and cross-scale observations. Its immediate targets are known
failure families: sustained single-column overload, lost relief under similar
head counts, excessive occupation under a plausible LN fraction, flattened
phrasing, and control-range drift. It returns evidence and witness times rather
than a single score that could hide one failure behind another improvement.

## Exact prefix and scope semantics

[`ChartTrace`](../../src/ensomi_model/research/gameplay_evaluation/temporal.py)
reads complete rows from BOS through an explicit exclusive coverage boundary.
Replay validates their physical execution. A local boundary does not close an
LN: observed holding is integrated through coverage while release count stays
unchanged. Every named `Scope` is half-open. Pre-scope attacks, holding and
recovery remain available; a control change cannot reset them.

Each scope reports H/head/LN/release counts, LN-head fraction, exact mean held
columns and any-held fraction. It also records per-column and all-column time
spent freely recovered after .25/.5/1 seconds without an action. Recovery credit
begins at the actual previous action plus that duration, not at the scope start.
Adjacent scope credits therefore add without a fictitious restart.

Trailing per-column attack peaks use .5/1/2/4/8/16-second windows, retain prefix
attacks and return witness clocks. Successive same-column attacks below 20 ms
have explicit endpoint witnesses. These attack checks count TAP and LN press;
releases remain distinct. The existing
[sustained-response envelope](sustained_response_planning.md) owns calibrated
corpus-reference excess and can be attached separately. This package does not
replace it with another invented strain scale.

These distinctions catch failures that an average cannot:

- Identical H times and head count can concentrate demand on one finger.
- Identical LN-head fraction can conceal much longer simultaneous holding.
- Fewer heads can coexist with much less actual recovery.
- A new control range can inherit already accumulated attack pressure.

## Multi-scale phrasing observations

For each half-window length .5/1/2/4/8/16 seconds, compare adjacent left/right
intervals at a 250-ms analysis stride. Each vector contains H rate, four column
attack rates, four held fractions, LN-head rate and release rate. Signed
right-minus-left differences preserve the direction and time of change.
Column-share transfer is also reported separately from total activity.

Both halves must lie in the same named control range. A short range is explicitly
insufficient for larger scales; other ranges cannot be pooled to manufacture
coverage. Exact event counts and occupied-time integrals supply the interval
features. The stride samples contrast locations, so it is not an exact maximum
detector; rare attack peaks have their separate event-time calculation.

The measurements locate changed pacing and texture. Higher contrast or greater
finger redistribution is not inherently better: steady streams, anchors and
chordjacks are valid. Ranked examples and labeled organization determine which
changes matter, and each control condition keeps its own comparison.

## Audio correspondence is a diagnostic

[`audio_correspondence`](../../src/ensomi_model/research/gameplay_evaluation/alignment.py)
uses the confirmed canonical full-song Mel clock: 128 bins, 10-ms hop, 40-ms
window, centers at `20 + 10*i` ms. Eight log-power bands and their positive
frame-to-frame changes describe spectral development. Their means over 1/4/16
seconds are paired with the chart's interval-feature vectors on the same clock.

Within each scope and scale, center and standardize feature dimensions, then
measure linear centered kernel alignment. This adapts the representation-
correspondence statistic discussed by
[Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a.html); their neural
representation results do not establish musical quality. The implementation uses
feature-space products rather than allocating a time-by-time kernel matrix.

Report zero-lag alignment beside five within-scope circular shifts of the chart
features. Shifts preserve marginal values while disrupting temporal pairing.
They are descriptive comparators, not exchangeable samples or p-values.
Constant inputs or fewer than eight windows are explicitly unevaluable.

A loudness-following generator can score well while producing poor arrangements.
Deliberate dumps or steady passages can score poorly while being valid. Low
alignment must not become an automatic BAD label, and this statistic must not
be optimized as a musical-quality reward. Its role is to help distinguish a
changed temporal relationship from a changed global mean, with source comparisons
and actual listening/Lens judgments supplying the missing interpretation.

## Use and qualification status

```python
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace, Scope
from ensomi_model.research.gameplay_evaluation.report import evaluate_scopes

trace = ChartTrace(complete_rows, coverage_ms=audio_duration_ms + 1)
report = evaluate_scopes(trace, [Scope("override", 64000, 96000)], mel=full_mel,
                         identity=verified_run_identity)
```

The report format is `gameplay-temporal-evaluation/v1`. Caller-owned identity
should record audio/source/model byte hashes, controls, seed and generation
completion. Reports retain separate scopes and contain no pooled pass verdict.
`feature_order` names every contrast coordinate. Artifact arrays are diagnostic
evidence, not new human annotations or source labels.

Eight focused tests cover concentrated versus distributed attacks at equal
timing/counts, smeared LN tails at equal head fractions, carried recovery and
open holds, flattened temporal activity at equal totals, cross-boundary short
attacks, constant-input handling, and canonical-Mel/chart alignment. These verify
algorithmic distinctions. Real ranked exceptions and historical generated failures
must still calibrate regression rules before the framework can qualify a model.
Latency, runtime stress, semantic style and player experience remain separate
evaluation dimensions; none is certified by these tests.
