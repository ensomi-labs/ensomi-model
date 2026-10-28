"""Freeze G2 definitions and measured baseline noise before either F2 run."""
import datetime
import json
import subprocess
from pathlib import Path
from .common import OUT,G1,X5D,DEADLINE,dump,digest,disk_guard
from .statistics import noise


def freeze():
    if (OUT/'prereg-lock.json').exists():raise RuntimeError('G2 preregistration already frozen')
    if not (OUT/'baseline-DONE.json').exists():raise RuntimeError('Baseline measurements incomplete')
    reproduction=json.loads((OUT/'decoder-reproduction.json').read_text())
    if not reproduction['passed']:raise RuntimeError('Copied decoder reproduction failed')
    decision=json.loads((OUT/'target-domain-decision.json').read_text())
    from ..cache import checkpoint
    from .. import base
    paths=[G1/'results.json',G1/'cohorts.json',G1/'decision-exclusions.json',G1/'sigma-baseline.json',
           OUT/'targets.pkl',OUT/'target-rates.json',OUT/'target-domain-decision.json',base.PREV/'references.pkl']
    for seed in (0,1):
        paths += [checkpoint(seed),G1/'runs'/f'F0-s{seed}/final.pt',OUT/'initialisers'/f's{seed}.pt',
                  X5D/f's{seed}/heads/linear.npz']
    inputs=dict(files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in paths if p.exists()],
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                copied_source_commit='a00b9b6',
                code_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    dump(OUT/'inputs.json',inputs)
    facts=dict(noise=noise(),rates=json.loads((OUT/'target-rates.json').read_text()),
               audit=json.loads((OUT/'input-audit.json').read_text()),
               initialisers=json.loads((OUT/'initialisers.json').read_text()),
               sigma=json.loads((G1/'sigma-baseline.json').read_text()),
               domain_decision=decision,resources=disk_guard(),decoder_reproduction=reproduction)
    dump(OUT/'prereg-facts.json',facts)
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    cohorts=facts['audit']['cohorts'];outside=facts['rates']['out_of_support']
    text=f'''# G2 preregistration

Frozen UTC: {utc}. Facts SHA-256: {digest(OUT/'prereg-facts.json')}.
G1's four runs, their scoring and report are complete. G2 is authorised measurement;
no perceptual validation or model adoption is claimed. Chart bands are supervision,
not a new conditioning input. Fixed inputs remain audio, song memory, existing open
panels and the original seed checkpoints. Prior influence from the excluded training
recording remains in those checkpoints. No sealed audio/charts or listening marks are read.

## Cohort and targets

Reuse the audited {cohorts['training']['audio']} training recordings and
{cohorts['panels']['audio']} panel audio inputs / {cohorts['panels']['rows']} reference sets.
The original folder/group decision applies to every list, including the
{cohorts['probe_fit']['audio']} fit and {cohorts['probe_cal']['audio']} calibration recordings. The independent census folder
and load_exclusion audits have passed again. A new unexplained match stops the job.

Copy X5d a00b9b6 targets, half-open red-line lookup, band aggregation and D2 as files.
b=floor(log2(notated BPM/80)), b=-2..4. Targets use active chart red lines within mapped,
scorable ranges; reference memberships average one-hot labels as in X5d. Stretch the
chart times AND beat periods before applying the band function, preserving F0's audio
augmentation. Band CE is weighted 0.5 and multiplied by the original break/detour/range
weights. A zero mask contributes no gradient.

Target-domain decision: {json.dumps(decision)}.
The boundary audit found {len(outside)} recording/rate cases outside support,
totalling {sum(r['frames'] for r in outside)} mapped frames across the audited rates.
Exact affected metadata and rates are in prereg-facts.json. Tests cover
fractional red-line boundaries, mapped ends, unscorable spans, stretch-dependent octave
changes, absolute D2 projections and identical sampler random streams.

## Runs and controls

F2-s0 then F2-s1 each starts from that seed's ckpt-009001, never from F0. Fresh AdamW,
weight decay .01, clip 1; peak LR 3e-4, 200 warmup steps, original seconds-based cosine
to 5% over 40 training minutes. Torch/sampler seeds 2000+seed; weighted sampler, 8 songs
x 2 crops x 20 seconds, augmentations, memory and prefetch are unchanged. Sigma floor
is {facts['sigma']['floor']:.12g}, as in G1. Each run is single-job with no resume.

Add Linear(160,7), with the saved same-seed X5d linear parameters. Fold feature mean,
standard deviation AND its positive calibration temperature into weights/bias, so
initial posterior equals the calibrated X5d posterior. Folding was numerically checked.
Missing parameters initialise zeros and are recorded; both saved fits are present.
F2's final head is used directly, with no new post-training temperature or panel tuning.

Keep the original validation loss/M1 probes and every original kill/resource rule, plus
three consecutive S0 M1 probes below 90% of the same-seed original baseline. Probes are
every four training minutes. Scientific kills are results and not retried. Save ONLY
the final checkpoint, including a killed endpoint; no intermediate weights.

Arm 5 refits X5d's linear algorithm on each final F0's own features: h=0/1 fit, h=2
epoch selection and temperature calibration, using the same probe recording identities.
At most 512 evenly spaced mapped frames per recording, equal total recording weight.
Keep sampled features float16, normalise from fit rows only. X5d AdamW linear fit uses
LR .01, weight decay 1e-4, batch 4096, at most 60 epochs, patience 8 with 1e-4 progress;
select earliest strict calibration minimum, then positive temperature in [.05,20].
No MLP, mixture tuning or panel labels select this control. Full features are streamed;
only these bounded samples are cached, and deleted after arm 5 is scored. The bounded
linear fit remains an alternative explanation for gains; it does not prove necessity.

## Evaluation and decisions

Arms: F0-base (G1 maps/scores reused), F2-base (original decoder), F2-own-D2,
F2-band-D2, F0-frozen-band-D2. All decoders use reset_penalty=14. Each checkpoint supplies
its own tracked pulse; D2 arms share that checkpoint's exact track. Band D2 uses X5d's
mean-log regime aggregation and unchanged song prior. Save final pulse/band outputs for
466 inputs; retain scored maps and tables after cleanup. CPU scoring uses at most two
workers. F0 regenerated evidence is checked against five G1 panel dumps per seed.

Cells: fit_dev/all (panel), val_hard/all, pooled/S0, pooled/S5. Measures: level,
same bars, guarded R5 AND P5 >= .9 with/without oracle offset, BL-20, and copied
structural readiness. Structural stratum spans use X5d's overlap rule and all reference
spans within a recording must be ready. These structural values characterise only.
Band accuracy for arms 4/5: all mapped frames, and chart red-line regimes (same-band
changes retained); regime choice is mean log posterior without song bonus, as X5d.
Report confusion counts, denominators and equal-reference means within recording.

Every comparison uses matched finite recording rows. Intervals use 1,000 paired
recording bootstrap resamples, RNG 20261028. Margin=max(abs(v0.1 s0-s1), paired difference
95% interval half-width), computed for that cell/statistic and compared pair. Selected
and neighbouring-checkpoint baselines and seed nulls are already recorded in facts.

Per seed, prediction requires F2-band-D2 minus F0-base: fit_dev/all level > margin;
val_hard/all level >= -margin; same bars >= -margin on both panels. It additionally
requires F2-band-D2 minus F0-frozen-band-D2 val_hard/all level >= 0 (the brief's literal
nonnegative comparison; its interval/margin are reported). Both seeds must satisfy
all clauses for pass. Retire the band training target only if val_hard/all level
F2-band-D2 minus F0-base < -margin on BOTH seeds. Otherwise inconclusive. F2-base versus
F0-base guarded readiness, bars and BL-20 characterises interference; no extra rule.
Scientific kills remain visible alongside endpoint verdicts. Nothing chooses a new
margin, cohort, decoder setting or rerun after seeing F2 numbers.
G1's F0-s0 stopped at its scientific kill boundary before the full training duration;
F2 keeps the same rules. Different endpoint durations limit causal attribution if they occur.

## Resources, retention and deadline

One MPS process at a time, nice 10, original memory/throughput guards. At most 6 GB new;
check before each run with reserve for the remaining 6 GB budget, ensuring at least
100 GB remains. Do not start a run unless its observed/estimated training and panel
inference can finish by {DEADLINE}. The queue conservatively reserves at least 60 minutes.
After the G2 report is written, delete gpu/input-cache and per-audio G1/G2 inference
dumps while retaining final checkpoints, scored maps, summaries and numeric probes.
Record every deleted path and its bytes in the report. Do not delete G0 feature caches
or another job's artifacts. Further deviations are recorded separately; never edit this file.
'''
    (OUT/'prereg.md').write_text(text)
    lock=dict(utc=utc,sha256=digest(OUT/'prereg.md'),facts_sha256=digest(OUT/'prereg-facts.json'),
              domain_decision_sha256=digest(OUT/'target-domain-decision.json'),inputs_sha256=digest(OUT/'inputs.json'))
    dump(OUT/'prereg-lock.json',lock)
    with (OUT/'job.log').open('a') as f:f.write(json.dumps(dict(event='preregistered',**lock))+'\n')
    print(json.dumps(lock),flush=True)


if __name__=='__main__':freeze()
