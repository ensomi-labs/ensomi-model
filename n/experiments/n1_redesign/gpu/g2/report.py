"""Render G2's fixed paired comparisons, verdicts and artifact-retention record."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import numpy as np
from .common import OUT,G1,dump,check_prereg,disk_guard,digest
from .statistics import CELLS,METRICS,read,select,paired
from ..scoring import interval

ARMS=('F0-base','F2-base','F2-own-D2','F2-band-D2','F0-frozen-band-D2')


def compare(seed,arm,control='F0-base'):
    result={}
    for p,c in CELLS:
        result[f'{p}/{c}']={m:paired(*[select(read(name,s),p,c,m) for name,s in
            [('v01-009001',0),('v01-009001',1),(control,seed),(arm,seed)]]) for m in METRICS}
    return result


def accuracy_summary(seed):
    rows=json.loads((OUT/'scores'/f'accuracy-s{seed}.json').read_text());result={}
    for arm in ('F2-band-D2','F0-frozen-band-D2'):
        result[arm]={}
        for p,c in CELLS:
            rr=[r for r in rows if r['arm']==arm and r['cell']==c and (p=='pooled' or r['panel']==p)]
            cell={}
            for kind in ('frame','regime'):
                groups=defaultdict(list);conf=np.zeros((7,7),np.int64)
                for r in rr:
                    matrix=np.asarray(r[kind]);conf+=matrix
                    if matrix.sum():groups[(r['panel'],r['rec'])].append(float(matrix.trace()/matrix.sum()))
                cell[kind]=dict(recording_accuracy=interval([np.mean(x) for x in groups.values()]),
                                confusion=conf.tolist(),total=int(conf.sum()),correct=int(conf.trace()))
            cell['omitted_regimes']=sum(r['omitted_regimes'] for r in rr)
            result[arm][f'{p}/{c}']=cell
    return result


def decision(comp,matched):
    def stat(cell,metric):return comp['F2-band-D2'][cell][metric]
    fit=stat('panel/all','n_level');hard=stat('val_hard/all','n_level')
    bars=[stat(c,'n_bars_same') for c in ('panel/all','val_hard/all')]
    control=matched['val_hard/all']['n_level']
    clauses=dict(fit_level_gain=fit['delta']['value']>fit['margin'],
                 hard_level_no_loss=hard['delta']['value']>=-hard['margin'],
                 fit_bars_no_loss=bars[0]['delta']['value']>=-bars[0]['margin'],
                 hard_bars_no_loss=bars[1]['delta']['value']>=-bars[1]['margin'],
                 hard_level_at_least_frozen=control['delta']['value']>=0)
    return dict(clauses=clauses,prediction=all(clauses.values()),retire=hard['delta']['value'] < -hard['margin'])


def load(path):return json.loads(path.read_text())


def build():
    lock=check_prereg()
    for seed in (0,1):
        if not (OUT/f'evaluation-s{seed}-DONE.json').exists():raise RuntimeError('Both evaluations required')
    results=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),preregistration=lock,
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 baseline=load(OUT/'prereg-facts.json'),decoder_reproduction=load(OUT/'decoder-reproduction.json'),seeds={})
    for seed in (0,1):
        comp={arm:compare(seed,arm) for arm in ARMS}
        matched=compare(seed,'F2-band-D2','F0-frozen-band-D2')
        results['seeds'][str(seed)]=dict(comparisons_to_F0=comp,matched_readout=matched,decision=decision(comp,matched),
            run=load(OUT/'runs'/f'F2-s{seed}/result.json'),F0_run=load(G1/'runs'/f'F0-s{seed}/result.json'),
            control_fit=load(OUT/'control'/f's{seed}/fit.json'),band_accuracy=accuracy_summary(seed),
            inference={arm:load(OUT/'eval'/f'{arm}-s{seed}/inference.json') for arm in ('F0','F2')},
            evaluation=load(OUT/f'evaluation-s{seed}-DONE.json'))
    decisions=[s['decision'] for s in results['seeds'].values()]
    results['verdict']=dict(prediction='pass' if all(d['prediction'] for d in decisions) else 'inconclusive',
                           retire='triggered' if all(d['retire'] for d in decisions) else 'not_triggered',adoption='none')
    results['deviations']=[load(OUT/'target-domain-decision.json')]
    results['limitations']=[
        'The original checkpoints retain influence from the excluded training recording; exclusion cannot undo it.',
        'F0-s0 ended at its G1 scientific kill; differing endpoint durations limit causal attribution.',
        'The frozen readout uses bounded per-recording samples and a fixed linear optimisation; it is not an upper bound on frozen features.',
        'Structural readiness and chart scores characterise this open panel; no owner listening marks or sealed holdouts were used.',
    ]
    results['resources']=disk_guard()
    results['deletions']=load(OUT/'deletions.json') if (OUT/'deletions.json').exists() else []
    results['cleanup_complete']=(OUT/'cleanup-DONE.json').exists()
    results['deleted_bytes']=sum(r['bytes'] for r in results['deletions'] if r['status']=='deleted')
    results['resources']['retained_plus_deleted_G2_bytes']=results['resources']['new_bytes']+sum(
        r['bytes'] for r in results['deletions'] if r['status']=='deleted' and Path(r['path']).is_relative_to(OUT))
    dump(OUT/'results.json',results)
    lines=['# G2: absolute tempo band as a training target','',
           f"Prediction: **{results['verdict']['prediction']}**. Retire rule: **{results['verdict']['retire']}**. Nothing adopted.",'',
           f"Preregistered {lock['utc']}; SHA-256 `{lock['sha256']}`. Code `{results['code_commit']}`.",'',
           'G1 completed before G2. The copied X5d D2 implementation reproduced both own-level and frozen-band maps on five items per seed (60 exact field comparisons). F0 inference reproduction errors appear below. All cohort audits passed, including the prior folder/group exclusion. Baseline seed and neighbouring-checkpoint statistics are in prereg-facts.json and results.json.','',
           'Level differences below are percentage points; parentheses give the margin. The first two comparisons are F2-band-D2 minus F0-base. The last is F2-band-D2 minus the matched frozen readout.','',
           '| Seed | fit_dev level Δ (M) | val_hard level Δ (M) | val_hard Δ versus frozen (M) | Prediction clauses |',
           '|---|---:|---:|---:|---|']
    for seed,s in results['seeds'].items():
        values=[s['comparisons_to_F0']['F2-band-D2'][c]['n_level'] for c in ('panel/all','val_hard/all')]
        values.append(s['matched_readout']['val_hard/all']['n_level'])
        fields=[f"{100*v['delta']['value']:+.2f} ({100*v['margin']:.2f})" for v in values]
        failed=[k for k,v in s['decision']['clauses'].items() if not v]
        lines.append(f"| {seed} | {' | '.join(fields)} | {', '.join(failed) if failed else 'all met'} |")
    lines += ['','## Runs','',
           '| Run | Status | Steps | Training min | Wall min | Sigma floor | F0 repro max abs |',
           '|---|---|---:|---:|---:|---:|---:|']
    for seed,s in results['seeds'].items():
        error=max(x['max_abs'] for x in s['inference']['F0']['reproduction'])
        for name,r in [(f'F0-s{seed} (G1)',s['F0_run']),(f'F2-s{seed}',s['run'])]:
            lines.append(f"| {name} | {r['status']} | {r['steps']} | {r['training_s']/60:.2f} | {r['wall_s']/60:.2f} | {r['sigma_floor']:.10g} | {error:g} |")
    lines+=['','G2 wrote only a final checkpoint for each F2 run. Original four-minute probes, losses, resource guards and kills are retained in runs/F2-s*/progress.jsonl. No scientific kill was rerun. Final component losses:','']
    for seed,s in results['seeds'].items():
        lines.append(f"- F2-s{seed}: `{json.dumps(s['run']['final_losses'],sort_keys=True)}`.")
    lines+=['','## Panel and structure comparisons','',
            'Values and differences are percentages/percentage points. Brackets give paired 95% recording-bootstrap intervals (1,000 resamples). M is max(original seed spread, paired interval half-width). Each row uses the common finite recording cohort for that statistic. “panel” is fit_dev. Offset-aligned and raw measures are both shown.','']
    labels=dict(n_level='Level',n_bars_same='Same bars',guarded_ready='Guarded readiness',raw_guarded_ready='Guarded readiness raw',n_bl20='BL-20',structural_ready='Structural readiness',raw_structural_ready='Structural readiness raw',raw_n_level='Level raw',raw_n_bars_same='Same bars raw',raw_n_bl20='BL-20 raw')
    for seed,s in results['seeds'].items():
        lines += [f'### Seed {seed}','','| Cell | Arm | Measure | n | Value | Δ F0 [95% CI] | M |','|---|---|---|---:|---:|---:|---:|']
        for arm,cc in s['comparisons_to_F0'].items():
            for cell,metrics in cc.items():
                for metric,v in metrics.items():
                    if v is None:continue
                    d=v['delta'];lo,hi=d['ci95']
                    lines.append(f"| {cell} | {arm} | {labels[metric]} | {v['n']} | {100*v['treatment']['value']:.2f} | {100*d['value']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {100*v['margin']:.2f} |")
        v=s['matched_readout']['val_hard/all']['n_level'];d=v['delta'];lo,hi=d['ci95']
        lines+=['',f"F2-band-D2 minus F0-frozen-band-D2, val_hard level: {100*d['value']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}], margin {100*v['margin']:.2f} pp.",'',
                f"Prediction clauses: `{json.dumps(s['decision']['clauses'])}`. Seed retirement condition: `{s['decision']['retire']}`.",'']
    lines+=['## Band accuracy','','Equal-reference means within recordings, then equal-recording averages. Confusions and exact frame/regime denominators are in results.json. Regime accuracy uses chart red-line regimes; the decoder uses its own tracked regimes.','',
            '| Seed | Arm | Cell | Frame accuracy | Regime accuracy | Recordings |','|---|---|---|---:|---:|---:|']
    for seed,s in results['seeds'].items():
        for arm,cc in s['band_accuracy'].items():
            for cell,v in cc.items():
                a=v['frame']['recording_accuracy'];b=v['regime']['recording_accuracy']
                lines.append(f"| {seed} | {arm} | {cell} | {100*a['value']:.2f}% | {100*b['value']:.2f}% | {a['n']} |")
    lines+=['','## Initialisation and matched readout','',
            'Both F2 heads used the saved same-seed X5d linear fit, folding feature standardisation and its positive calibration temperature into weights and bias. Numerical folding checks and input hashes are retained in initialisers.json. The matched control refit uses h=0/1 fit and h=2 calibration on F0 features; fit curves and inventories are retained.','']
    for seed,s in results['seeds'].items():
        f=s['control_fit'];lines.append(f"- Seed {seed}: selected epoch {f['best_epoch']}, temperature {f['temperature']:.6g}, calibrated NLL {f['calibrated_nll']:.6g}; {f['cohort']['counts']}.")
    lines+=['','## Deviations and limits','','```json',json.dumps(results['deviations'],indent=2),'```','']
    lines += ['- '+x for x in results['limitations']]
    lines+=['','## Disk retention','',f"Deleted {results['deleted_bytes']/1e9:.3f} GB. Cleanup complete: {results['cleanup_complete']}. Current G2 artifacts: {results['resources']['new_bytes']/1e9:.3f} GB; free disk: {results['resources']['free_bytes']/1e9:.3f} GB.",'',
            f"Retained plus already deleted G2 artifacts total {results['resources']['retained_plus_deleted_G2_bytes']/1e9:.3f} GB. This accounts for caches removed before the final report; every running disk check also enforced the 6 GB cap and 100 GB free-space floor.",'',
            'The complete per-path deletion ledger (sizes and status) is in deletions.json and results.json. Final checkpoints, numeric probes, scored maps, score tables, fitted readout weights, inventories and reports are retained. G0 caches and other jobs are untouched.','',
            '| Deleted location | Files | GB |','|---|---:|---:|']
    deleted=defaultdict(lambda:[0,0])
    for r in results['deletions']:
        if r['status']!='deleted':continue
        p=Path(r['path']);key=str(p.parent) if p.suffix=='.npz' else str(p)
        deleted[key][0]+=r['files'];deleted[key][1]+=r['bytes']
    lines += [f'| {p} | {n} | {b/1e9:.3f} |' for p,(n,b) in sorted(deleted.items())]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'report-identity.json',dict(utc=results['utc'],report_sha256=digest(OUT/'report.md'),results_sha256=digest(OUT/'results.json')))
    print(json.dumps(results['verdict']),flush=True)


if __name__=='__main__':build()
