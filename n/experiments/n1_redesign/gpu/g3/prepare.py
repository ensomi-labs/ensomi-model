"""Audit metadata and freeze G3 definitions before any event model is trained."""
import datetime
import json
from pathlib import Path
import subprocess
from ensomi_n.features.cache import mel_path
from ..audit_inputs import context, check_rows
from .common import OUT, G1, ROOT, BULK, dump, digest, probe_bucket, disk_guard, log
from .statistics import METRICS, table, paired


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'prereg-lock.json').exists():
        raise RuntimeError('Preregistration already frozen')
    if json.loads((G1/'results.json').read_text())['status']!='complete' or json.loads((G1/'g2/results.json').read_text())['status']!='complete':
        raise RuntimeError('Preceding experiments are incomplete')
    paths, args, corpus, audit, sealed, exclusion, decision = context(ROOT)
    cohorts = json.loads((G1/'cohorts.json').read_text())
    cohorts['probe_fit'] = [r for r in cohorts['training'] if probe_bucket(r['audio_sha256'])<2]
    audit['cohorts'] = {}
    for name, rows in cohorts.items():
        kept, removed = decision.filter(rows)
        violations = check_rows(kept, exclusion, sealed)
        audit['cohorts'][name] = dict(rows=len(rows), audio=len({r['audio_sha256'] for r in rows}),
                                      removed=len(removed), violations=violations)
        if removed or violations:
            dump(OUT/'input-audit.json',audit)
            raise RuntimeError(f'G3 input audit stopped at {name}')
    dump(OUT/'input-audit.json',audit)
    songs = json.loads((G1/'inference-songs.json').read_text())
    train = {r['audio_sha256'] for r in cohorts['training']}
    panel = {r['audio_sha256'] for r in cohorts['panels']}
    for sha in train:
        if not mel_path(BULK/'mel',sha).exists():
            raise RuntimeError(f'Missing training Mel: {sha}')
    for sha in panel:
        if not Path(songs[sha]['mel']).exists():
            raise RuntimeError(f'Missing panel Mel: {sha}')
    if len(train)!=6196 or len(panel)!=466 or len(cohorts['probe_fit'])!=1544:
        raise RuntimeError('G1 cohort counts changed')
    files = [G1/p for p in ['cohorts.json','inference-songs.json','decision-exclusions.json','target-rates.json',
        'prereg.md','results.json','heads/F1-s0.json','heads/F1-s1.json','heads/frozen-s0.json','heads/frozen-s1.json']]
    files += [BULK/'song-index-v0.json']
    inputs = {str(p):digest(p) for p in files}
    target_paths = list((G1/'targets').glob('*.npz'))
    if not target_paths:
        target_paths = list((G1/'targets').rglob('*.npz'))
    # Inventory only the target files referenced by the audited cohorts.
    used_targets = [p for p in target_paths if p.stem in train|panel]
    if {p.stem for p in used_targets} != train|panel:
        raise RuntimeError('Missing G1 target files')
    inputs.update({str(p):digest(p) for p in used_targets})
    dump(OUT/'inputs.json',dict(sha256=inputs,training=len(train),panel_inputs=len(panel),panel_references=len(cohorts['panels']),
        probe_fit=len(cohorts['probe_fit']),target_files=len(used_targets),disk=disk_guard(),
        main_checkout_mutations='Private G3 artifacts only; shared Mel, labels and G1 targets read-only'))
    baseline = {}
    for kind in ['F1','frozen']:
        rows = [json.loads((G1/'heads'/f'{kind}-s{s}.json').read_text())['rows'] for s in (0,1)]
        baseline[kind] = {m:paired(table(rows[0],m),table(rows[1],m)) for m in METRICS}
    dump(OUT/'baseline-noise.json',baseline)
    text = '''# G3 preregistration: event training from random initialization

Measurement only; no model or decoder is adopted. G1 and G2 are complete before this experiment.

## Inputs, exclusions and fixed choices

Reuse G1's 6,196 audited training recordings, 466 panel audio inputs (472 reference sets), targets and chart variants. Reaudit every inherited cohort and the 1,544 h=0,1 fit recordings against load_exclusion, sealed folders and the owner's folder/group decision before opening training or panel content. The closed folder/group and all aliases remain excluded. The original v0.1 checkpoints had prior influence from the excluded training recording; their G1 comparisons retain that influence. G3 starts randomly and does not load those weights. No private listening marks or sealed chart/audio content is read. Input hashes and cohort counts are in inputs.json and input-audit.json.

G1 targets are unchanged: union of difficulty heads; notated red-line beats; coarsest legal divisors 1,2,3,4,6,8,12,16 with 2 ms tolerance. Attack positives within one frame have weight 1, two frames weight .5; negatives are farther than 40 ms in beats with at least two heads; missing notes supply no negative evidence. Snap CE uses one-frame head neighborhoods. Salience uses the original ladder retention bins and slider repeat/tail retention rule. Reuse G1's attack pos_weight {ATTACK_POS_WEIGHT} and break mask exactly. Sigma floor is inapplicable because G3 has no period head or pulse loss.

## Runs and guards

E-s0 then E-s1, one MPS process at a time. Instantiate NEv0(memory=False) with its default v0.1 encoder dimensions, random torch seed 3000+seed, followed only by Linear(160,7). No warm start, pulse head, pulse loss or song-memory module. Reuse G1 TargetSampler with sampler seed 2000+seed so its crops and augmentation sequence match G1; eight weighted songs, two 20 s crops per song, rate .88..1.12 and original gain/tilt. The original sampler prepares unused CPU memory arrays; none enters the event model.

Fresh AdamW, weight decay .01, gradient clip 1.0, peak LR .001, 500 step linear warm-up, then v0.1's training-seconds cosine to 5 percent at 3,000 training seconds. Auxiliary loss is exactly G1: attack weighted BCE times .5, snap CE times .5, cumulative salience BCE times .25, including G1's per-crop normalization. No pulse probes or loss guards. Kill on any nonfinite loss or gradient, allocated MPS memory above 14 GiB or driver above 17 GiB, or throughput below half its first post-warm-up reference for 180 s. Check resources each minute, loss means every 200 steps and at the end. Save only final weights. No resume or scientific reruns.

Use nice 10 and one Torch/BLAS thread. New disk at most 3 GB and at least 100 GB free. Do not start a run unless 60 wall minutes remain until 2026-10-29 01:15 UTC (50 training minutes plus evaluation allowance). Delete only G3 regenerable per-recording inference arrays after scores are durable, and retain their size ledger. No shared inputs are removed.

## Comparators and scoring

Reuse G1's exact head_rows implementation, targets, eligibility and score aggregation. Compare E against same-seed G1 F1 and frozen-linear saved recording scores. Mel-linear has five 128-band log-Mel frames, offsets -2..2 on the 100 fps clock, song mean/std normalization as the encoder receives, endpoint replication at song ends, and no augmentation at fit/evaluation. Fit one comparator per seed on the same audited h=0,1 recordings, the same eligible frames and RNG 3000+seed, cap 512 frames per recording, zero-initialized Linear(640,7), G1 losses and LBFGS configuration (120 iterations, 10 optimization minutes, one CPU thread). No feature standardization beyond song normalization. A failed fit is zeroed and reported, as in G1. No panel fitting. Samples are held in RAM; no additional feature cache.

Attack: within-song AUC on frames farther than 30 ms from chart lines, using G1 positives/negatives, at least 50 of each. Cells: beats with at least two heads, songs with at least four ternary and four binary-fine beats; mixed ternary beats count ternary. Score maximum sigmoid(attack)*softmax(snap)[ternary] over each beat. Report chart beats and original same-seed v0.1 decoded beats matched within 15 ms after the G1 song offset; retain G1's exact decoded interval readout and report unmatched share. Salience: within-song Spearman of expected cumulative bin versus retention. Average reference variants within recording, then mean recordings; all paired comparisons intersect recording keys. Report pooled and panel-specific characterization, counts, intervals and missingness.

## Noise, prediction and retirement

Baseline seed-null statistics are measured before training in baseline-noise.json. G1 F1 chart-cell AUCs and seed spread are computed from the saved recording rows in baseline-noise.json. New-head neighboring-checkpoint estimates are unavailable because only final G1 heads were scored; this is an evidence limit, not a zero-noise assumption. The prescribed margin replaces G1's seed-spread floor: max(.01, paired recording-bootstrap 95 percent half-width for E minus F1), 1,000 resamples with the existing G1 bootstrap seed. No panel-, seed- or checkpoint-based selection.

Prediction: chart-beat cell AUC gain E minus F1 is at least .05 on both seeds. Report this literal point threshold and, following the inherited beyond-margin pass rule, call a noise-qualified pass only when the gain is at least .05+margin on both seeds. Retire dedicated events trunk recovers thirds if, on both seeds, E cell AUC is at most F1+margin and E is below .70. Otherwise inconclusive. A missing seed cannot pass or retire. Attack, decoded-cell AUC, salience and the training-loss curve characterize only. The primary comparison uses the matched E/F1 recording set; comparator-specific pairs also retain explicit denominators.

This comparison changes initialization, memory, losses, learning rate and training duration together. It tests this recipe, not the isolated causal effect of removing pulse supervision. G1's saved run metadata identifies any earlier stops that limit the matched comparison. Chart-derived metrics have not been validated against private human labels, so these are preregistered numerical verdicts for owner review, not perceptual quality claims. Guardrails 2,4,8,9,10 apply; guardrail 3 introduces no new conditioning path here.
'''
    text = text.replace('{ATTACK_POS_WEIGHT}',str(json.loads((G1/'target-rates.json').read_text())['attack_pos_weight']))
    (OUT/'prereg.md').write_text(text)
    source = Path(__file__).parent
    frozen = [source/p for p in ['common.py','train.py','infer.py','mel.py','statistics.py']]
    frozen += [source.parent/p for p in ['evaluate.py','finetune.py','targets.py','scoring.py','decision.py','sealed.py']]
    frozen += [Path('n/src/ensomi_n')/p for p in ['v0/model.py','v0/data.py','v0/infer.py','v0/train.py','augment/__init__.py']]
    lock = dict(sha256=digest(OUT/'prereg.md'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={str(p.resolve()):digest(p) for p in frozen},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    dump(OUT/'prereg-lock.json',lock)
    log('preregistered',prereg_sha256=lock['sha256'],prereg_utc=lock['utc'],
        source_sha256=lock['source_sha256'],code_commit=lock['code_commit'])


if __name__=='__main__':
    prepare()
