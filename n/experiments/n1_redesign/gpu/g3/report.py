"""Matched G3 evidence, numerical verdicts, loss curves and owned-cache cleanup."""
import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path
import subprocess
import numpy as np
from ..scoring import interval
from .common import OUT, G1, dump, digest, disk_guard, check_prereg
from .statistics import METRICS, table, paired


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(name):
    root = G1 if name.startswith(('F1-', 'frozen-')) else OUT
    return read(root/'heads'/f'{name}.json', {}).get('rows', [])


def coverage(data):
    return dict(references=len(data), recordings=len({(r['panel'],r['rec']) for r in data}),
                eligible_recordings={m:len(table(data,m)) for m in METRICS},
                decoded_beats=sum(r.get('cell_total',0) for r in data),
                decoded_unmatched=sum(r.get('cell_unmatched',0) for r in data),
                decoded_unmatched_share=(sum(r.get('cell_unmatched',0) for r in data)/sum(r.get('cell_total',0) for r in data))
                    if sum(r.get('cell_total',0) for r in data) else None)


def fmt(value):
    if value is None:
        return '—'
    return f'{value:.4f}'


def fmt_interval(value):
    if not value or value['value'] is None:
        return '—'
    return f"{value['value']:.4f} [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]"


def loss_curves():
    points = {}
    for seed in (0,1):
        path = OUT/'runs'/f'E-s{seed}'/'progress.jsonl'
        points[str(seed)] = [r for r in map(json.loads,path.read_text().splitlines()) if r['event']=='loss'] if path.exists() else []
    dump(OUT/'loss-curves.json',points)
    if any(points.values()):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,2,figsize=(10,3.4),sharey=True)
            for s,ax in enumerate(axes):
                rs = points[str(s)]
                for metric in ['total','attack','snap','salience']:
                    ax.plot([r['training_s']/60 for r in rs],
                            [r['loss'] if metric=='total' else r['parts'][metric] for r in rs],marker='.',label=metric)
                ax.set(title=f'E-s{s}',xlabel='Training minutes')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            axes[0].set_ylabel('Mean loss over logging interval')
            fig.tight_layout()
            fig.savefig(OUT/'loss-curves.png',dpi=160)
            plt.close(fig)
        except ImportError:
            pass
    return points


def render():
    lock = check_prereg()
    queue = read(OUT/'queue-DONE.json',{})
    runs = {f'E-s{s}':read(OUT/'runs'/f'E-s{s}'/'result.json',dict(status='missing')) for s in (0,1)}
    controls = {f'F1-s{s}':read(G1/'runs'/f'F1-s{s}'/'result.json',{}) for s in (0,1)}
    comparisons = {}
    aggregates = {}
    covers = {}
    for seed in (0,1):
        names = [f'{arm}-s{seed}' for arm in ['E','F1','frozen','Mel']]
        data = {name:rows(name) for name in names}
        covers[str(seed)] = {name:coverage(rs) for name,rs in data.items()}
        comparisons[str(seed)] = {}
        aggregates[str(seed)] = {}
        for label,panel in [('pooled',None),('fit_dev','panel'),('val_hard','val_hard')]:
            comparisons[str(seed)][label] = {}
            aggregates[str(seed)][label] = {}
            for metric in METRICS:
                ts = {name:table(rs,metric,panel) for name,rs in data.items()}
                aggregates[str(seed)][label][metric] = {name:interval(list(t.values())) for name,t in ts.items()}
                comparisons[str(seed)][label][metric] = {
                    name:paired(ts[name],ts[f'E-s{seed}']) for name in names if name!=f'E-s{seed}'}
    clauses = {}
    for s in (0,1):
        comp = comparisons[str(s)]['pooled']['cell_auc_chart'][f'F1-s{s}']
        clauses[str(s)] = None if comp is None else dict(point_prediction=comp['delta']['value']>=.05,
            noise_qualified_prediction=comp['delta']['value']>=.05+comp['margin'],
            retirement=comp['treatment']['value']<=comp['control']['value']+comp['margin'] and comp['treatment']['value']<.70,
            event_auc=comp['treatment']['value'],F1_auc=comp['control']['value'],gain=comp['delta']['value'],margin=comp['margin'])
    both = lambda key: all(clauses[str(s)] and clauses[str(s)][key] for s in (0,1))
    verdict = dict(prediction='pass' if both('noise_qualified_prediction') else 'inconclusive',
                   literal_point_threshold='met' if both('point_prediction') else 'not_met',
                   retire='retire' if both('retirement') else 'not_triggered',per_seed=clauses)
    curves = loss_curves()
    cleanup = read(OUT/'cleanup.json',dict(status='pending',entries=[],deleted_bytes=0))
    fits = {str(s):{k:v for k,v in read(OUT/'fits'/f'mel-s{s}.json',{}).items() if k!='selection'} for s in (0,1)}
    result = dict(status='complete' if queue.get('status')=='complete' and all(covers[str(s)][f'E-s{s}']['references']==472 and covers[str(s)][f'Mel-s{s}']['references']==472 for s in (0,1)) else 'incomplete',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  preregistration=lock,input_audit=read(OUT/'input-audit.json'),
                  baseline_noise=read(OUT/'baseline-noise.json'),runs=runs,G1_F1_runs=controls,comparisons=comparisons,
                  aggregates=aggregates,coverage=covers,mel_fits=fits,verdict=verdict,loss_curves=curves,
                  queue=queue,resources=disk_guard(),cleanup=cleanup,failed_paths=read(OUT/'failed-paths.json',[]),
                  validations=read(OUT/'validation-final.json',read(OUT/'validation-before-training.json')),
                  limitations=['This recipe jointly changes initialization, memory, losses, learning rate and duration relative to F1.',
                    'G1 run metadata records any earlier loss-guard stops; G3 uses a different training budget.',
                    'Original v0.1/G1 comparators retain prior influence from the excluded training recording; G3 is initialized randomly.',
                    'Chart-derived measures characterize; no private human-label validation or adoption is claimed.',
                    'No neighboring-checkpoint noise estimates exist for these new-head metrics.',
                    'Mel-linear uses the prescribed G1 fit and readout; it does not establish optimal Mel capacity or reproduce other probe methods.',
                    'Decoded-cell scoring intentionally reuses the original v0.1 maps and G1 readout, not an event-only decoder.',
                    'Retirement concerns this architecture and training budget; it cannot exclude other event-training recipes.'])
    dump(OUT/'results.json',result)
    lines = ['# G3: event-only training without song memory','',
             f"Status: **{result['status']}**. Prediction: **{verdict['prediction']}**. Retirement rule: **{verdict['retire']}**. Nothing adopted.",'',
             'The comparison tests this event-training recipe. It does not isolate pulse supervision from random initialization, removed memory, learning rate or training duration.','',
             f"Preregistered {lock['utc']}, SHA-256 `{lock['sha256']}`. Code commit `{result['code_commit']}`.",'',
             '## Cohorts and execution','',
             'Training: 6,196 recordings; probe fit: 1,544; panels: 466 audio inputs, 472 reference sets. The full inherited cohort audit passed with zero new sealed matches. G1 folder/group exclusions remain applied. Targets and data caches were reused read-only. No input-cache regeneration was needed.','',
             '| Run | Status | Steps | Training min | Wall min | Final total | Attack | Snap | Salience |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in runs.items():
        p = r.get('final_losses',{})
        lines.append(f"| {name} | {r['status']} | {r.get('steps','—')} | {r.get('training_s',0)/60:.2f} | {r.get('wall_s',0)/60:.2f} | "+' | '.join(fmt(p.get(k)) for k in ['total','attack','snap','salience'])+' |')
    for name,r in controls.items():
        lines.append(f"\nG1 comparator {name}: {r.get('status','missing')}, {r.get('training_s',0)/60:.2f} training minutes, {r.get('steps',0)} steps.")
    lines += ['', 'The encoder has no song-memory module and only seven event outputs. AdamW, loss weights, targets and masking follow the frozen preregistration. Sigma floor is inapplicable. Final losses are the last batch; interval means and full loss curves are saved separately. Component losses are unweighted; total is 0.5 attack + 0.5 snap + 0.25 salience.','',
              '## Primary chart-cell comparison','',
              '| Seed | E AUC [95% CI] | F1 AUC [95% CI] | Paired gain [95% CI] | Margin | Recordings | Point ≥ .05 | Beyond margin | Retire clause |',
              '|---|---|---|---|---:|---:|---|---|---|']
    for s in (0,1):
        c = comparisons[str(s)]['pooled']['cell_auc_chart'][f'F1-s{s}']
        v = clauses[str(s)]
        if c:
            lines.append(f"| {s} | {fmt_interval(c['treatment'])} | {fmt_interval(c['control'])} | {fmt_interval(c['delta'])} | {c['margin']:.4f} | {c['n']} | {v['point_prediction']} | {v['noise_qualified_prediction']} | {v['retirement']} |")
    lines += ['', 'Margins are max(0.01, paired recording-bootstrap 95% half-width), with 1,000 resamples. Retirement requires both seeds to have E ≤ F1 + margin and E < 0.70. The literal +0.05 threshold and the inherited beyond-margin pass rule are reported separately.','',
              '## All paired head comparisons','',
              '| Seed | Panel | Metric | Comparator | E [95% CI] | Comparator [95% CI] | Paired delta [95% CI] | n |','|---|---|---|---|---|---|---|---:|']
    for s in (0,1):
        for panel,ms in comparisons[str(s)].items():
            for metric,arms in ms.items():
                for name,c in arms.items():
                    if c:
                        lines.append(f"| {s} | {panel} | {metric} | {name} | {fmt_interval(c['treatment'])} | {fmt_interval(c['control'])} | {fmt_interval(c['delta'])} | {c['n']} |")
    lines += ['', 'Attack, decoded-cell AUC and salience characterize only. Comparisons use matched recording keys; reference variants are averaged within recording. Unmatched counts are retained in results.json.','',
              '## Readout coverage and Mel fits','',
              '| Seed | Arm | References | Recordings | Attack eligible | Chart-cell eligible | Decoded unmatched / eligible beats |','|---|---|---:|---:|---:|---:|---|']
    for s in (0,1):
        for name,c in covers[str(s)].items():
            lines.append(f"| {s} | {name} | {c['references']} | {c['recordings']} | {c['eligible_recordings']['attack_auc']} | {c['eligible_recordings']['cell_auc_chart']} | {c['decoded_unmatched']} / {c['decoded_beats']} |")
    lines += ['', 'Decoded unmatched counts pool eligible reference variants; the denominator is beats, not recordings. Each Mel fit uses song-normalized log-Mel, five stacked frames with endpoint replication, h=0,1 only and the same sampling/optimizer as its seed’s G1 frozen fit.','']
    for s,f in fits.items():
        lines.append(f"- Mel-s{s}: {f.get('recordings',0)} recordings, {f.get('frames',0)} sampled frames; {f.get('optimisation_s',0):.1f} optimization seconds; failure: {f.get('failed')}; final fit loss: {fmt(f.get('final_loss'))}.")
    lines += ['', '## Training-loss curve','',
              '![Training loss by elapsed training minutes](loss-curves.png)' if (OUT/'loss-curves.png').exists() else 'Full numeric loss curves: loss-curves.json.', '',
              '## Deviations, failed paths and limits','']
    for entry in result['failed_paths']:
        lines.append(f"- {entry['phase']}: {entry['detail']}")
    for limitation in result['limitations']:
        lines.append('- '+limitation)
    if queue.get('failures') or queue.get('skipped'):
        lines.append('- Queue failures/skips: '+json.dumps(queue))
    lines += ['', '## Storage and cleanup','',
              f"Current G3 bytes: {result['resources']['new_bytes']:,}; free bytes: {result['resources']['free_bytes']:,}. Cleanup status: {cleanup['status']}; removed {cleanup.get('deleted_bytes',0):,} bytes.",'']
    for entry in cleanup.get('entries',[]):
        lines.append(f"- `{entry['path']}`: {entry['files']} files, {entry['bytes']:,} bytes, {entry['status']}.")
    lines += ['', 'Only G3 per-recording inference arrays are regenerable here. Final checkpoints, per-recording scores, fit weights, manifests, preregistration, audits, logs and summary tables are retained. Shared G0/G1/G2 artifacts are untouched by this cleanup.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    return result


def cleanup():
    result = read(OUT/'results.json')
    if not result or result['status']!='complete':
        raise RuntimeError('Complete durable scores and report required before cleanup')
    ledger = read(OUT/'cleanup.json',dict(status='pending',entries=[],deleted_bytes=0))
    if ledger['status']=='complete':
        return
    for seed in (0,1):
        root = OUT/'eval'/f'E-s{seed}'
        paths = sorted(root.glob('*.npz'))
        if len(paths)!=466:
            raise RuntimeError('Unexpected inference dump count')
        info = dict(path=str(root/'*.npz'),files=len(paths),bytes=sum(p.stat().st_size for p in paths),
                    status='pending',sha256={p.name:digest(p) for p in paths})
        ledger['entries'].append(info)
        dump(OUT/'cleanup.json',ledger)
        for path in paths:
            path.unlink()
        info['status']='deleted'
        ledger['deleted_bytes']+=info['bytes']
        dump(OUT/'cleanup.json',ledger)
    ledger['status']='complete'
    dump(OUT/'cleanup.json',ledger)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cleanup',action='store_true')
    args=p.parse_args()
    render()
    if args.cleanup:
        cleanup()
        render()
