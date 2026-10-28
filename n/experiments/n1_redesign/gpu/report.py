"""Matched recording bootstrap comparisons and the preregistered both-seed verdicts."""
from __future__ import annotations
from collections import Counter,defaultdict
import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .cohorts import ROOT,OUT,FEAT,dump
from .audit_inputs import digest
from .scoring import interval,METRICS

HEAD_METRICS=('attack_auc','cell_auc_chart','cell_auc_decoded','salience_spearman')


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def pulse_table(name,cell,metric):
    data=read(OUT/'scores'/f'{name}.json')
    if not data:return {}
    return {(r['panel'],r['rec']):r[metric] for r in data['recordings'] if r['cell']==cell and np.isfinite(r[metric])}


def head_table(name,metric):
    data=read(OUT/'heads'/f'{name}.json')
    if not data:return {}
    groups=defaultdict(list)
    for r in data['rows']:
        if r.get(metric) is not None and np.isfinite(r[metric]):groups[(r['panel'],r['rec'])].append(r[metric])
    return {k:float(np.mean(v)) for k,v in groups.items()}


def comparison(b0,b1,f0,f1,seed):
    keys=sorted(set(b0)&set(b1)&set(f0)&set(f1))
    if not keys:return None
    a,b,c,d=[np.array([t[k] for k in keys]) for t in (b0,b1,f0,f1)]
    baseline=(a,b)[seed]
    delta=interval(d-c);gain=interval(d-baseline)
    width=(delta['ci95'][1]-delta['ci95'][0])/2
    margin=max(abs(float(a.mean()-b.mean())),width)
    return dict(n=len(keys),v01=interval(baseline),F0=interval(c),F1=interval(d),F1_minus_F0=delta,
                F1_minus_v01=gain,F0_minus_v01=interval(c-baseline),margin=margin,
                seed_spread=abs(float(a.mean()-b.mean())),paired_half_width=width,
                preserved=abs(delta['value'])<=margin,harmed=delta['value'] < -margin,
                gain_prediction=gain['value']>=.05+margin,gain_below_002=gain['value']<.02)


def both(values):
    return len(values)==2 and all(values)


def render():
    pulse={};heads={};runs={};missing=[]
    for seed in (0,1):
        pulse[str(seed)]={};heads[str(seed)]={}
        for arm in ('F0','F1'):
            name=f'{arm}-s{seed}'
            runs[name]=read(OUT/'runs'/name/'result.json',dict(status='not_started'))
        has_control=(OUT/'scores'/f'F0-s{seed}.json').exists()
        for cell in ('all','S0'):
            pulse[str(seed)][cell]={}
            for metric in METRICS:
                result=comparison(pulse_table('v01-s0-009001',cell,metric),pulse_table('v01-s1-009001',cell,metric),
                                  pulse_table(f'F0-s{seed}' if has_control else f'v01-s{seed}-009001',cell,metric),
                                  pulse_table(f'F1-s{seed}',cell,metric),seed)
                if result:result['controlled']=has_control
                pulse[str(seed)][cell][metric]=result
        for metric in HEAD_METRICS:
            result=comparison(head_table('frozen-s0',metric),head_table('frozen-s1',metric),
                              head_table(f'F0-s{seed}' if has_control else f'frozen-s{seed}',metric),
                              head_table(f'F1-s{seed}',metric),seed)
            if result:result['controlled']=has_control
            heads[str(seed)][metric]=result
    predictions={}
    for label,metrics in [('attack_gain',('attack_auc',)),('cell_gain',('cell_auc_chart','cell_auc_decoded'))]:
        ok=[]
        for seed in (0,1):
            rs=[heads[str(seed)][m] for m in metrics]
            ok.append(all(r and r['controlled'] and r['gain_prediction'] for r in rs))
        predictions[label]='pass' if both(ok) else 'inconclusive'
    primary=[('all',m) for m in ('guarded_ready','raw_guarded_ready','n_bars_same','raw_n_bars_same',
                                  'n_level','raw_n_level','n_bl20','raw_n_bl20')]+[('S0','n_bars_same'),('S0','raw_n_bars_same')]
    preserved=[]
    for seed in (0,1):
        rs=[pulse[str(seed)][c][m] for c,m in primary]
        preserved.append(all(r and r['controlled'] and r['preserved'] for r in rs))
    predictions['pulse_preservation']='pass' if both(preserved) else 'inconclusive'
    harms=[]
    for cell,metric in primary:
        if 'level' in metric:continue
        rs=[pulse[str(s)][cell][metric] for s in (0,1)]
        if all(r and r['controlled'] and r['harmed'] for r in rs):harms.append(f'{cell}/{metric}')
    sufficient=[]
    for seed in (0,1):
        rs=[heads[str(seed)][m] for m in ('attack_auc','cell_auc_chart','cell_auc_decoded')]
        sufficient.append(all(r and r['gain_below_002'] for r in rs))
    retire=dict(interference='retire' if harms else 'inconclusive',interference_metrics=harms,
                frozen_probes_suffice='retire' if both(sufficient) else 'inconclusive')
    all_kills={name:r['kills'] for name,r in runs.items() if r.get('kills')}
    kills={name:k for name,k in all_kills.items() if name.startswith('F1')}
    storage=[OUT,FEAT/'v01-s1-009001',FEAT/'v01-s0-009001-ext']
    resources=dict(new_bytes=sum(p.stat().st_size for d in storage if d.exists() for p in d.rglob('*') if p.is_file()),
                   free_bytes=shutil.disk_usage(OUT).free,budget=read(OUT/'resume-budget.json'),max_driver_gib={})
    for name in runs:
        path=OUT/'runs'/name/'progress.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        values=[r['mps_driver_gb'] for r in rows if 'mps_driver_gb' in r]
        if values:resources['max_driver_gib'][name]=max(values)
    g0=read(OUT/'g0-results.json');lock=read(OUT/'prereg-lock.json')
    repro=read(OUT/'baseline-reproduction.json',{})
    differences=defaultdict(list)
    for d in repro.get('differences',[]):differences[d['metric']].append(abs(d['actual']-d['expected']))
    reproduction=dict(compared=repro.get('compared'),differences={k:dict(count=len(v),max_abs=max(v)) for k,v in differences.items()})
    coverage={}
    for name in [f'{a}-s{s}' for s in (0,1) for a in ('frozen','F0','F1')]:
        h=read(OUT/'heads'/f'{name}.json')
        if h:
            rows=h['rows'];tot=sum(r.get('cell_total',0) for r in rows);unmatched=sum(r.get('cell_unmatched',0) for r in rows)
            coverage[name]=dict(reference_entries=len(rows),qualified={m:sum(r.get(m) is not None for r in rows) for m in HEAD_METRICS},
                                decoded_unmatched_beats=unmatched,decoded_total_beats=tot,decoded_unmatched_share=unmatched/tot if tot else None)
    structure={}
    for name in [f'v01-s{s}-009001' for s in (0,1)]+[f'F{a}-s{s}' for s in (0,1) for a in (0,1)]:
        scores=read(OUT/'scores'/f'{name}.json')
        if scores:
            counts=Counter();ready=defaultdict(list)
            for r in scores['structure']:
                counts.update(r['counts']);ready[(r['panel'],r['rec'])].append(float(r['ready']))
            structure[name]=dict(first_error_span_counts=dict(counts),recording_ready=interval([np.mean(v) for v in ready.values()]))
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    audit_file=read(OUT/'full-input-audit.json',{})
    audit=audit_file.get('cohorts',{})
    extension=read(OUT/'cohorts.json',{}).get('extension',[])
    drop_reasons=Counter(reason for r in extension if r.get('ext')=='std' for reason in r.get('drop_reasons',[]))
    result=dict(status='complete' if (OUT/'queue-DONE.json').exists() else 'in_progress',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                code_commit=code_commit,preregistration=lock,g0=g0,baseline_reproduction=reproduction,
                input_cohorts=audit,exclusion_decision=audit_file.get('decision'),
                prior_checkpoint_training_removed=audit.get('training',{}).get('removed',[]),
                extension_drop_reason_counts=dict(drop_reasons),
                baseline_noise=read(OUT/'baseline-noise.json'),sigma=read(OUT/'sigma-baseline.json'),
                target_rates=read(OUT/'target-rates.json'),target_head_invariant=read(OUT/'target-head-invariant.json'),
                runs=runs,pulse=pulse,new_heads=heads,head_coverage=coverage,structure=structure,predictions=predictions,
                retire_rules=retire,scientific_kills=all_kills,F1_kills=kills,resources=resources,
                kill_interpretation=('F1 scientific kills are preregistered interference outcomes and were not retried. '
                                     'A matching control kill limits attribution to the auxiliary heads. '
                                     'Endpoint statistics describe the stopped runs, not completed 40-minute treatments.' if kills else None),
                queue_deviations=read(OUT/'queue-deviations.json',[]),
                limitations=['Original checkpoints retain prior influence from the excluded folder/group.',
                             'Frozen initialisers use bounded frame samples and 120 LBFGS iterations; optimiser diagnostics are reported.',
                             'Frozen features are fp16; new-run readouts are float32.',
                             'No perceptual measure was validated against private listening marks; these numerical rules do not adopt a model.'],
                failed_paths=['Initial sealed-input stop, then explicit decision filter and full re-audit.',
                              'Initial G0 metadata lookup lacked non-primary panel audio aliases; fixed before inference.'])
    dump(OUT/'results.json',result)
    def val(x):return '—' if x is None else f'{x:.4f}'
    def ci(x):return '—' if not x else f"{x['value']:.4f} [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"
    lines=['# N1 redesign GPU measurements','',f"Status: {result['status']}. Code commit: `{code_commit}`.",'',
           f"Predictions: {json.dumps(predictions)}.",f"Retire rules: {json.dumps(retire)}.",'',
           f"Scientific kills: {json.dumps(all_kills)}.",result['kill_interpretation'] or 'No F1 scientific kill was recorded.','',
           'These are preregistered measurement verdicts. The owner decides adoption and perceptual quality.', '',
           '## Preregistration and inputs','',f"Preregistration: `{lock['sha256'] if lock else 'not yet frozen'}`, UTC `{lock['utc'] if lock else 'pending'}`.",
           f"The authorised folder/group decision leaves {audit.get('training',{}).get('audio')} training recordings "
           f"and {audit.get('g0_probe',{}).get('audio')} G0 probe recordings. "
           'All assembled cohorts passed the independent sealed-folder audit after this filter. '
           'The original checkpoints retain prior influence from the excluded recording; no sealed audio or chart labels were opened.', '',
           f"Excluded folders: {json.dumps((result['exclusion_decision'] or {}).get('folders',[]))}. "
           f"Excluded corpus groups: {json.dumps((result['exclusion_decision'] or {}).get('corpus_groups',[]))}. "
           f"Prior-checkpoint training recording removed from new training: {json.dumps(result['prior_checkpoint_training_removed'])}.",'',
           'Inputs and cheap hashes are in `inputs.json`; full cohort counts and decision removals are in `full-input-audit.json`. '
           'The fixed input is audio plus unchanged song memory. Heads, snap and retention are supervision only. '
           'Evaluation uses existing open panels; chart-grid cells can conceal errors in the predicted grid, so the second cell readout uses original decoded grids.', '',
           '## G0','']
    if g0:
        lines += [f"Wall: {g0['wall_s']/60:.2f} minutes. New job bytes at G0 completion: {g0['bytes']/1e9:.3f} GB. h=2 inputs cut: {g0['probe_h2_cut']}.",'',
                  '| Seed cache | Inputs | Audio hours | Cache GB | Extension counts |','| --- | ---: | ---: | ---: | --- |']
        for s,r in g0['counts'].items():lines.append(f"| {s} | {r['inputs']} | {r['audio_hours']:.3f} | {r['bytes']/1e9:.3f} | {json.dumps(r['extension'])} |")
        mx=max(v for r in g0['reproduction'] for v in r['full_precision_max_abs'].values())
        lines += ['',f"Five seed-1 E2 reproduction checks: maximum float32 absolute difference {mx:.9g}. "
                  'Detailed field/float16 differences are in g0-reproduction.json.',
                  f"Unreadable extension inputs skipped: {len(g0['skipped'])}; identities and reasons are in `g0-results.json`. "
                  'Curation drop reasons are retained in extension manifest memberships and `cohorts.json`.',
                  f"Std extension drop-reason counts (a recording may have several): {json.dumps(dict(drop_reasons))}.",
                  'All 15 session-1 recordings were already in the seed-0 manifest, so none was added. `DONE` was written before G1.', '']
    else:lines+=['G0 is running.','']
    lines+=['## Runs','', '| Run | Status | Steps | Training min | Wall min | Sigma floor | Final probe loss |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in runs.items():
        last=(r.get('history') or [{}])[-1]
        lines.append(f"| {name} | {r['status']} | {r.get('steps','—')} | {r.get('training_s',0)/60:.2f} | {r.get('wall_s',0)/60:.2f} | {val(r.get('sigma_floor'))} | {val(last.get('probe_loss'))} |")
    lines+=['','Per-run losses, every probe and kill are in `results.json` and `runs/<run>/`. '
            'F0-v0.1 drift and paired intervals are included alongside each comparison.',
            f"New job storage: {resources['new_bytes']/1e9:.3f} GB; free disk: {resources['free_bytes']/1e9:.3f} GB. "
            f"Maximum sampled driver GiB by run: {json.dumps(resources['max_driver_gib'])}.",'',
            '## Baselines and pulse','',f"Baseline reproduction: {reproduction['compared']} comparisons; {json.dumps(reproduction['differences'])}. "
            'All discrepancies are reported rather than hidden by rounding.',
            'Neighbouring-checkpoint ranges and the selected-seed difference are in `baseline-noise.json`. '
            'Margins below are the larger of the baseline seed difference and paired F1-F0 bootstrap half-width. '
            'Registered preservation is two-sided: improvements beyond the margin also fail equivalence. '
            'The harm retirement rule uses only negative differences.',
            f"The seed-0 share of baseline sigma below .01 was {result['sigma']['share_below_001']:.12g}; "
            f"all fine-tunes used its fifth-percentile floor {result['sigma']['floor']:.12g}.", '']
    for seed in (0,1):
        lines += ['',f'### Seed {seed}', '', '| Cohort / statistic | N | v0.1 | F0 | F1 | F1−F0 [95% CI] | Margin |','| --- | ---: | ---: | ---: | ---: | --- | ---: |']
        for c,m in primary:
            r=pulse[str(seed)][c][m]
            if r:lines.append(f"| {c}/{m} | {r['n']} | {val(r['v01']['value'])} | {val(r['F0']['value'])} | {val(r['F1']['value'])} | {ci(r['F1_minus_F0'])} | {val(r['margin'])} |")
    lines += ['', '## New heads','', 'AUCs and Spearman are recording means; comparisons use matched finite recordings. '
              'Intervals resample recordings 1,000 times. F0 readouts apply the frozen initialiser to the fine-tuned F0 features.', '']
    for seed in (0,1):
        lines += ['',f'### Seed {seed}', '', '| Statistic | N | Frozen | F0 readout | F1 | F1−frozen [95% CI] | Margin |','| --- | ---: | ---: | ---: | ---: | --- | ---: |']
        for m,r in heads[str(seed)].items():
            if r:lines.append(f"| {m} | {r['n']} | {val(r['v01']['value'])} | {val(r['F0']['value'])} | {val(r['F1']['value'])} | {ci(r['F1_minus_v01'])} | {val(r['margin'])} |")
    lines += ['', 'Decoded-grid unmatched shares and qualified reference counts:', '', '```json',json.dumps(coverage,indent=2),'```','',
              'The structural scorecard is in `results.json:structure`, with all per-reference spans in `scores/`. '
              'Salience characterises only. No listening marks were used.', '', '## Deviations, failures and limits','',
              *['- '+s for s in result['failed_paths']+result['limitations']],
              *['- '+json.dumps(s) for s in result['queue_deviations']], '',
              'Scientific kills are outcomes; they are not retried. Any missing arm prevents a controlled both-seed conclusion. '
              'The original budget deadline is recorded in `resume-budget.json`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def provenance():
    files=['artifacts/n1-night-20261018/runs/v0data2-20261019/curated.json',
           'artifacts/n1-night-20261018/runs/r0-20261019a/split.json',
           'artifacts/n1-night-20261018/runs/r0-20261019a/recordings.json',
           'artifacts/bulk/n1-cache/song-index-v0.json',
           'artifacts/astra-consult-20261021/astra/exp/references.pkl',
           'artifacts/astra-consult-20261021/astra/exp/e1-summary.json',
           'artifacts/n1-redesign-20261028/census/charts.csv','artifacts/n1-redesign-20261028/census/names.json',
           'artifacts/timing-lens-20261024/private/session-01.identities.json']
    for s in (0,1):
        files += [f'artifacts/bulk/n1-runs/v01-s{s}-20261020/ckpt-009001.pt',
                  f'artifacts/n1-night-20261018/runs/v01-s{s}-20261020/config.json',
                  f'artifacts/n1-redesign-20261028/featcache/v01-s{s}-009001/manifest.jsonl']
    records={p:dict(exists=(ROOT/p).exists(),sha256=digest(ROOT/p) if (ROOT/p).is_file() else None) for p in files}
    for p in ['artifacts/bulk/n1-cache/mel','artifacts/bulk/n1-cache/memo-v1','artifacts/bulk/n1-cache/labels-v0b',
              'artifacts/n1-stage0-20261028','artifacts/bulk/n1-stage0-20261028/maps']:
        records[p]=dict(exists=(ROOT/p).is_dir(),kind='directory')
    for seed in (0,1):
        p=f'artifacts/astra-consult-20261021/astra/exp/evidence/s{seed}/009001'
        records[p]=dict(exists=(ROOT/p).is_dir(),kind='directory')
    dump(OUT/'inputs.json',dict(inputs=records,code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                               decision_sha256=digest(OUT/'decision-exclusions.json'),
                               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                               source_base='b51a9ff',feature_source='codex/n1-featcache',score_source='codex/n1-stage0',
                               structure_source='codex/n1-decoder',parser_source='b14d9f5'))


if __name__=='__main__':
    provenance();render()
