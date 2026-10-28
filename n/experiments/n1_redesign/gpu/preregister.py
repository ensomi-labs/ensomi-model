"""Freeze definitions, measured baseline facts and decision rules before any fine-tune."""
import datetime
import json
import subprocess
from pathlib import Path
import numpy as np
from .cohorts import OUT,dump
from .audit_inputs import digest
from .scoring import METRICS


def freeze():
    if (OUT/'prereg-lock.json').exists():
        lock=json.loads((OUT/'prereg-lock.json').read_text())
        if digest(OUT/'prereg.md')!=lock['sha256']:raise ValueError('Frozen preregistration was changed')
        return lock
    if any((OUT/'runs').glob('*/result.json')):raise ValueError('Cannot preregister after a treatment result')
    audit=json.loads((OUT/'full-input-audit.json').read_text())
    labels=json.loads((OUT/'target-rates.json').read_text())
    sigma=json.loads((OUT/'sigma-baseline.json').read_text())
    probes=json.loads((OUT/'baseline-probes.json').read_text())
    fits={str(s):json.loads((OUT/'fits'/f's{s}.json').read_text()) for s in (0,1)}
    scores={}
    for seed,cks in [(0,['008137','008400','009001']),(1,['007973','008401','009001'])]:
        for ck in cks:scores[f's{seed}-{ck}']=json.loads((OUT/'scores'/f'v01-s{seed}-{ck}.json').read_text())
    noise={}
    for cell in ('pooled/all','pooled/S0'):
        noise[cell]={}
        for metric in METRICS:
            c=cell.split('/')[1]
            tables={k:{(r['panel'],r['rec']):r[metric] for r in v['recordings'] if r['cell']==c and np.isfinite(r[metric])}
                    for k,v in scores.items()}
            matched=set.intersection(*(set(v) for v in tables.values()))
            values={k:float(np.mean([v[r] for r in sorted(matched)])) for k,v in tables.items()}
            noise[cell][metric]=dict(n=len(matched),values=values,seed_difference=abs(values['s0-009001']-values['s1-009001']),
                                     checkpoint_ranges={str(s):max(v for k,v in values.items() if k.startswith(f's{s}-'))-
                                                        min(v for k,v in values.items() if k.startswith(f's{s}-')) for s in (0,1)})
    dump(OUT/'baseline-noise.json',noise)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    source=Path(__file__).parent
    hashes={p.name:digest(p) for p in sorted(source.glob('*.py'))}
    facts=dict(utc=now,counts=audit['cohorts'],sigma=sigma,label_rates=labels,probe_baselines=probes,
               linear_fits=fits,noise=noise,code_sha256=hashes)
    from .report import head_table, HEAD_METRICS
    facts['frozen_head_baselines']={}
    for metric in HEAD_METRICS:
        tables=[head_table(f'frozen-s{s}',metric) for s in (0,1)]
        matched=sorted(set(tables[0])&set(tables[1]))
        values=[float(np.mean([t[k] for k in matched])) if matched else None for t in tables]
        facts['frozen_head_baselines'][metric]=dict(n=len(matched),values=values,
            seed_difference=abs(values[0]-values[1]) if matched else None)
    dump(OUT/'prereg-facts.json',facts)
    text=f'''# G1 preregistration

Frozen UTC: {now}. This records the authorised measurement design before any F0 or F1 fine-tune.
The accompanying `prereg-facts.json` contains input counts, label rates, both baseline probe values,
all selected and neighbouring-checkpoint panel statistics, seed spread and checkpoint ranges.
Its SHA-256 is `{digest(OUT/'prereg-facts.json')}`.

## Scope and exclusions

The explicit resumption decision excludes the folder and corpus group recorded in
`decision-exclusions.json` from every new cohort. Training has {audit['cohorts']['training']['audio']}
recordings. The G0 probe subset has {audit['cohorts']['g0_probe']['audio']} recordings before any runtime cut.
The panels have {audit['cohorts']['panels']['rows']} reference entries and {audit['cohorts']['panels']['audio']}
audio inputs. The extension has {audit['cohorts']['std_extension']['audio']} std candidates and
{audit['cohorts']['keysound_extension']['audio']} keysound candidates. The session-1 extension adds
{audit['cohorts']['session1_extension']['audio']} recordings not already in seed-0's cache.
All lists, including the source manifest after this decision filter, passed both the copied census
sealed-folder rule and `load_exclusion`. Any new unexplained match stops the job.

The original v0.1 checkpoints were trained on the excluded recording. Filtering a fine-tune does
not remove that prior influence. This authorised deviation is recorded; holdout content remains
closed. No private listening marks are used. These measurements characterise the design; adoption
and perceptual quality remain for the owner. No new conditioning input is introduced (guardrail 3).

## Targets and clock

Targets are cached once per audio, with separate reference variants where panel sets share audio.
N's Reference and RecLabels provide notated beats and the scorable union of difficulty heads.
A head is snapped within 2 ms, divisors tried in order 1,2,3,4,6,8,12,16: line, duple-off or ternary;
unsnapped heads receive no snap supervision. Fractional boundaries and tempos, crop stretching,
song ends and slider events have absolute synthetic tests (guardrail 10).

Attack is binary BCE weighted 0.5. Round each stretched head to its nearest 10 ms frame; frames
within one frame have target 1 and weight 1, two frames away have target 1 and weight 0.5.
Negatives are strictly more than 40 ms from every head inside notated beats with at least two heads.
Other frames have weight zero; multiply attack weights by v0.1's break/detour/mapped-range weights.
Positive weight is sqrt(weighted negatives / weighted positives), clipped to [1,20], namely
{labels['attack_pos_weight']:.12g}. Snap CE has coefficient 0.5 on frames within one frame of a snapped head.
Ties choose the nearer head, earlier on equal distance. Salience uses the hardest chart of a ladder
of at least three charts on the same audio and mode, ordered by the census rank (onset count fallback
when absent). Easier standard slider repeats/tails count as retention. Bins are r=0, 0<r<0.5,
0.5<=r<1 and r=1; three binary cumulative targets 1[bin>k], k=0,1,2, have mean BCE coefficient 0.25
on frames within one frame of a binned head.

## Fits and runs

Separate frozen linear fits use each seed's own features, only training recordings with h in {{0,1}},
where h is the prescribed SHA-256 hash modulo 8. At most 512 uniformly sampled labelled frames
per recording are fitted by CPU LBFGS, at most 120 iterations and ten optimisation minutes, with
the same target masks and loss coefficients. These limits and fit loss diagnostics are recorded.
Any failed fit becomes a zero head and is reported. Sampling limits are a potential comparator
weakness; a gain over an underfit initialiser cannot establish that a fully optimised frozen probe
would be insufficient (guardrail 8).

Runs in order F0-s0, F1-s0, F0-s1, F1-s1. Each starts from that seed's ckpt-009001 and a fresh
AdamW (weight decay .01, clip 1). Both torch and sampler seeds are 2000+seed. Preserve weighted
sampling, 8 songs x 2 crops x 20 seconds, augmentation, memory and prefetch; each run is one job,
without resume. Peak LR 3e-4, 200 linear warmup steps, then v0.1's seconds-based cosine to 5%
over 40 training minutes. F0 has only pulse losses; F1 adds Linear(160,7) initialised from its fit.
All runs use sigma=softplus(raw)+{sigma['floor']:.12g}. Baseline share below .01 was
{sigma['share_below_001']:.12g}, above 5%, so the floor is seed-0's fifth percentile, not .01.

Keep validation crop loss and decoder-free M1 on 40 S0 and 40 hard val-random songs every four
training minutes, all original nonfinite, stalled, divergence, memory and throughput guards, plus
kill after three consecutive S0 M1 probes below 90% of the same seed's measured ckpt-009001 M1.
A scientific F1 kill is an interference result and is not rerun. One retry is allowed only for an
infrastructure failure within budget. Keep allocated MPS below 14 GiB, driver below 17 GiB,
80 GB free disk and at most 25 GB newly written. Only one MPS process; nice 10 when permitted.

## Readouts and aggregation

Infer on all panel audio through the copied feature-capture path. Store pulse evidence and seven
auxiliary logits, then use the original decoder with reset_penalty=14. CPU decode/scoring uses
at most two processes and may overlap the next training job. F0's seven evaluation logits are the
unchanged frozen initialiser applied to F0's features; they are not training heads. This permits the
prescribed F1-F0 AUC noise calculation. The principal frozen comparator is the initialiser on v0.1
cached features, independently for each seed. Those cached features are fp16; treatment readouts
use float32 features. Quantisation is a small potential alternative explanation, quantified by G0.

Use the copied stage-0 harness. R5 and P5 each >=.9 define guarded readiness. Report before and
after the recording's oracle global offset, same bars, level and BL-20; same bars additionally on S0.
Aggregate reference sets within recording as stage 0 does; recompute readiness from recording-mean
R5 and P5. Report panels separately and pooled, plus stratum tables and the copied decoder structural
scorecard. Baseline scores reproduce E1; tiny offset-only floating-point differences are disclosed.
Neighbouring checkpoints (s0: 008137,008400; s1: 007973,008401) are noise characterisations, never
substitutes selected after seeing a treatment (guardrail 2).

Attack AUC is per reference on eligible off-line frames >30 ms from every notated line, with at
least 50 positives and 50 negatives. Positive shoulders count as positives; AUC itself is unweighted.
Cell cohorts contain notated beats with at least two heads, in references with at least four beats
of each class. Any ternary head makes a ternary beat, mixed beats included; binary-fine requires only
line/duple heads and at least one duple-off head. Unsnapped heads prevent the binary-fine label.
Score a beat by max(sigmoid(attack)*softmax(snap)[ternary]) over its frames. Report AUC on chart
beats and on original v0.1 decoded notated beats matched one-to-one within 15 ms after the original
recording offset; use the original predicted intervals for pooling frame scores. Report unmatched
share and class counts. Salience Spearman compares sum(sigmoid(cumulative logits)) with retention
at nearest hardest-chart head frames; constant-valued recordings are omitted and counted. It only
characterises. Average reference values within recordings before song/recording summaries.

All intervals resample recordings, 1,000 draws, RNG seed 20261028, paired on matched finite rows.
A margin is max(abs(v0.1 s0 - v0.1 s1), half the paired F1-F0 bootstrap 95% interval width).
For AUC the v0.1 values are the corresponding frozen linear fits. Missing or unpaired statistics
are unavailable, never zero. Report sample counts and denominators for every statistic.

## Predictions and rules

Predictions were specified on 2026-10-28 before any run: F1 attack AUC and cell AUC exceed the
same-seed frozen readout by at least .05; pulse is preserved relative to F0. For a gain prediction,
pass on a seed requires gain >= .05+margin. Read cells on both chart and matched decoded grids.
Pulse preservation requires abs(F1-F0)<=margin for raw/aligned guarded readiness, same bars,
level and BL-20 on pooled/all, and for same bars on pooled/S0. F0-v0.1 characterises fine-tuning drift.
Both-seed rule: pass only when the prediction holds beyond its prescribed margin on both seeds;
retire only when the retiring condition holds on both; otherwise inconclusive.

Retire extend-the-trunk for interference if F1-F0 < -margin on guarded readiness, same bars or
BL-20 on both seeds (pooled/all, with S0 same bars also checked). Retire because frozen probes
suffice if the attack and cell AUC gains stay below .02 on both seeds. Intervals remain visible.
A single-seed F1 kill is reported as interference evidence; it cannot silently become a clean
both-seed comparison. No result selects a new margin, cohort, checkpoint or rerun.

## Budget and deviations

The restarted clock is 2026-10-28 15:00:03 UTC. Stop starting new runs at 21:30:03 UTC.
Before F0-s1, if both remaining runs cannot finish within that window using observed wall costs,
skip F0-s1 and run F1-s1 against v0.1-s1; its sigma floor and lack of F0 control are explicit confounds,
so a controlled both-seed conclusion is unavailable. G0 DONE precedes any G1 run.

Known deviations: the authorised cohort exclusion; bounded sampling in frozen linear fits; the
small fp16/fp32 feature storage difference. Initial G0 metadata assembly needed a panel-alias lookup
fix before inference; no treatment was run. Further deviations go in the report, never in this file.
'''
    (OUT/'prereg.md').write_text(text)
    lock=dict(sha256=digest(OUT/'prereg.md'),utc=now,facts_sha256=digest(OUT/'prereg-facts.json'))
    dump(OUT/'prereg-lock.json',lock)
    with (OUT/'job.log').open('a') as f:f.write(json.dumps(dict(event='preregistered',**lock))+'\n')
    print(json.dumps(lock),flush=True)
    return lock


if __name__=='__main__':freeze()
