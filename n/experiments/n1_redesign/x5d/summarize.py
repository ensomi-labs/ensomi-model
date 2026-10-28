"""Paired recording intervals, frozen decisions and private X5d report."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import math

from common import *
from baseline import recording_rows
from measurements import interval
from analyze import structural_rows, scorecard, fmt
from evaluate import ARMS

PANELS = ('panel','val_hard','pooled')
CELLS = ('all','S0','S5')
METRICS = ('n_bars_same','n_level','r5','p5','guarded_ready','raw_guarded_ready','structural_ready','raw_structural_ready')


def select(rows, panel, cell, arm):
    return {(r['panel'],r['rec']): r for r in rows if r['cell']==cell and r['arm']==arm
            and (panel=='pooled' or r['panel']==panel)}


def comparisons(rows, seeds, margins):
    result = {}
    for panel in PANELS:
        result[panel] = {}
        for cell in CELLS:
            result[panel][cell] = {}
            for arm in ARMS:
                metrics = {}
                for metric in METRICS:
                    stats = {}
                    for seed in seeds:
                        b = select(rows,panel,cell,seed+':baseline')
                        t = select(rows,panel,cell,seed+':'+arm)
                        keys = sorted(set(b)&set(t))
                        keys = [k for k in keys if np.isfinite(b[k].get(metric,np.nan)) and np.isfinite(t[k].get(metric,np.nan))]
                        if not keys:
                            continue
                        bv = np.array([b[k][metric] for k in keys]); tv = np.array([t[k][metric] for k in keys])
                        stats[seed] = dict(baseline=interval(bv), treatment=interval(tv), delta=interval(tv-bv))
                    metrics[metric] = dict(margin=margins[panel][cell].get(metric), seeds=stats)
                result[panel][cell][arm] = metrics
    return result


def accuracy_summary(rows, seeds):
    result = {}
    for panel in PANELS:
        result[panel] = {}
        for cell in CELLS:
            result[panel][cell] = {}
            for seed in seeds:
                result[panel][cell][seed] = {}
                for arm in ARMS:
                    sub = [r for r in rows if r['seed']==seed and r['arm']==arm and r['cell']==cell
                           and (panel=='pooled' or r['panel']==panel)]
                    result[panel][cell][seed][arm] = arm_result = {}
                    for unit in ('frame','regime'):
                        totals = np.zeros((7,7),np.int64)
                        groups = defaultdict(list)
                        omitted = 0
                        for r in sub:
                            counts = np.asarray(r[unit],np.int64)
                            totals += counts
                            omitted += r['omitted_regimes'] if unit=='regime' else 0
                            n = counts.sum()
                            if n:
                                groups[(r['panel'],r['rec'])].append([
                                    counts.trace()/n, np.trace(counts,offset=-1)/n, np.trace(counts,offset=1)/n])
                        recs = np.array([np.mean(groups[k],axis=0) for k in sorted(groups)])
                        arm_result[unit] = dict(sets=len(sub), recordings=len(recs), count=int(totals.sum()),
                            confusion=totals.tolist(), omitted_regimes=omitted,
                            correct=interval(recs[:,0] if len(recs) else []),
                            adjacent_lower=interval(recs[:,1] if len(recs) else []),
                            adjacent_higher=interval(recs[:,2] if len(recs) else []))
    return result


def decide(comp, seeds):
    result = {}
    for panel in PANELS:
        result[panel] = {}
        for cell in CELLS:
            values = comp[panel][cell]
            per_seed = {}
            for seed in seeds:
                level = values['band_d2']['n_level']; bars = values['band_d2']['n_bars_same']
                dl = level['seeds'][seed]['delta']['value']; db = bars['seeds'][seed]['delta']['value']
                per_seed[seed] = dict(level_prediction=dl>=level['margin']+.02,
                    bars_guard=db>=-bars['margin'],
                    retire=all(values[a]['n_level']['seeds'][seed]['delta']['value'] < -values[a]['n_level']['margin']
                               for a in ('band_d2','band_d2_mlp')))
            both = set(seeds)==set(SEEDS)
            passed = both and all(v['level_prediction'] and v['bars_guard'] for v in per_seed.values())
            retired = both and all(v['retire'] for v in per_seed.values())
            verdict = 'prediction supported' if passed else 'retired for these readouts' if retired else 'inconclusive'
            result[panel][cell] = dict(seeds=per_seed, both_seeds=both, prediction=passed, retire=retired, verdict=verdict)
    return result


def summarize(seeds):
    check_prereg()
    raw = []
    for seed in seeds:
        jobs = [json.loads(line) for line in (OUT / seed / 'evaluation.jsonl').read_text().splitlines()]
        if len(jobs)!=412 or len({(j['panel'],j['rec']) for j in jobs})!=412:
            raise RuntimeError('Incomplete or duplicate evaluation jobs')
        raw.extend(jobs)
    rows = recording_rows([r for job in raw for r in job['rows']])
    sr = [r for job in raw for r in job['structures']]
    for cell in CELLS:
        structures = structural_rows(sr,cell)
        for aligned, metric in ((True,'structural_ready'),(False,'raw_structural_ready')):
            lookup = {(r['panel'],r['rec'],r['arm']):r['ready'] for r in structures if r['aligned']==aligned}
            for r in rows:
                if r['cell']==cell:
                    r[metric] = lookup.get((r['panel'],r['rec'],r['arm']),math.nan)
    for r in rows:
        r['raw_guarded_ready'] = int(r['raw_r5']>=.9 and r['raw_p5']>=.9)
    old = json.loads((base.OUT / 'treatment-recordings.json').read_text())
    old = {(r['panel'],r['cell'],r['arm'],r['rec']):r for r in old}
    checked = 0
    for r in rows:
        if r['arm'].split(':')[1] not in ('baseline','own_d2','oracle_d2'):
            continue
        before = old[(r['panel'],r['cell'],r['arm'],r['rec'])]
        for metric in METRICS[:-1]:
            if metric in before and metric in r:
                checked += 1
                if not np.isclose(r[metric],before[metric],atol=1e-10,rtol=0,equal_nan=True):
                    raise RuntimeError(f'Control score mismatch {r["arm"]} {r["rec"]} {metric}')
    frozen = json.loads((OUT / 'frozen-margins.json').read_text())
    comp = comparisons(rows,seeds,frozen['margins'])
    acc = accuracy_summary([r for job in raw for r in job['accuracy']],seeds)
    input_rows = {(job['seed'],r['audio']):r for job in raw for r in job['inputs']}
    reconstruction = {seed:dict(audios=sum(s==seed for s,_ in input_rows),
        max_abs=max(r['reconstruction_max_abs'] for (s,_),r in input_rows.items() if s==seed),
        mean_abs=interval([r['reconstruction_mean_abs'] for (s,_),r in input_rows.items() if s==seed]),
        argmax_disagreement=interval([r['reconstruction_argmax_disagreement'] for (s,_),r in input_rows.items() if s==seed]))
        for seed in seeds}
    ambiguities = {}
    for seed in seeds:
        ambiguities[seed] = {}
        for split in ('fit','cal'):
            inv = json.loads((OUT / seed / 'data' / f'{split}-inventory.json').read_text())
            ambiguous = sum(r['ambiguous_frames'] for r in inv); mapped = sum(r['mapped_frames'] for r in inv)
            ambiguities[seed][split] = dict(recordings=len(inv), ambiguous_frames=ambiguous, mapped_frames=mapped,
                frame_weighted_share=ambiguous/mapped,
                recording_mean=interval([r['ambiguous_frames']/r['mapped_frames'] for r in inv]))
    result = dict(prereg=json.loads((OUT / 'prereg-identity.json').read_text()),
        generated_utc=datetime.now(timezone.utc).isoformat(), seeds=seeds,
        controls=dict(maps=json.loads((OUT / 'control-reproduction.json').read_text()), checked_scores=checked, passed=True),
        cohorts=dict(panel=dict(sets=322,recordings=262), val_hard=dict(sets=150,recordings=150),
            unique_audio=466, fit=len([a for a in training_rows() if bucket(a)<2]),
            calibration=len([a for a in training_rows() if bucket(a)==2])),
        exclusion_identity=json.loads((OUT / 'exclusion-identity.json').read_text()),
        frozen_margins=frozen, comparisons=comp, accuracy=acc, verdicts=decide(comp,seeds),
        structural=scorecard(structural_rows(sr)),
        training={seed:json.loads((OUT / seed / 'heads' / 'selection.json').read_text()) for seed in seeds},
        evaluation={seed:json.loads((OUT / seed / 'evaluation-identity.json').read_text()) for seed in seeds},
        ambiguous_targets=ambiguities, fp16_reconstruction=reconstruction,
        failures=json.loads((OUT / 'failures.json').read_text()) if (OUT / 'failures.json').exists() else [],
        deviations=json.loads((OUT / 'deviations.json').read_text()) if (OUT / 'deviations.json').exists() else [])
    dump(OUT / 'recording-scores.json',rows)
    dump(OUT / 'panel-inputs.json',list(dict(seed=s,**r) for (s,_),r in sorted(input_rows.items())))
    dump(OUT / 'results.json',result)
    write_report(result)
    print('X5d',result['verdicts']['panel']['all']['verdict'],flush=True)


def write_report(r):
    verdict = r['verdicts']['panel']['all']['verdict']
    lines = [f'# X5d: {verdict}', '',
        f"Preregistration SHA256 `{r['prereg']['sha256']}`, frozen {r['prereg']['utc']}.",
        'CPU measurement only; no decoder or model change is adopted. Primary decision cell: fit_dev/all.', '',
        f"Completed seeds: {', '.join(r['seeds'])}. Baseline/own-D2/oracle-D2 reproduced exactly: "
        f"{r['controls']['maps']['compared_maps']} maps before fitting, {r['controls']['checked_scores']} scored control values after evaluation.", '',
        f"The band is floor(log2(notated BPM / 80)), b=-2..4. Fit uses {r['cohorts']['fit']:,} training recordings (hash buckets 0/1); "
        f"{r['cohorts']['calibration']:,} disjoint bucket-2 recordings select epochs, temperatures and pooling weight. Each contributes at most 512 "
        'evenly spaced mapped frames with total weight one. Panel labels never select a parameter.', '',
        'Read sets: fit_dev 322 sets / 262 recordings; val-n1 hard 150 sets / 150 recordings, 466 unique audio files. '
        'All confidence intervals are 2,000 recording-bootstrap percentile intervals. Deltas use paired recordings; '
        'sets have equal weight within each recording. Structural readiness remains a characterisation.', '',
        '## Decision', '',
        'The linear prediction requires level gain at least 2 percentage points above its frozen margin and no bars '
        'loss beyond its margin on both seeds. Retirement requires both linear and MLP level losses beyond the '
        'margin on both seeds. Margins are max(baseline seed difference, larger baseline bootstrap half-width).', '',
        '| Cell | Level margin (pp) | Bars margin (pp) | Prediction | Retire | Verdict |',
        '| --- | ---: | ---: | --- | --- | --- |']
    for panel in PANELS:
        for cell in CELLS:
            d=r['verdicts'][panel][cell]; c=r['comparisons'][panel][cell]['band_d2']
            lines.append(f"| {panel}/{cell} | {100*c['n_level']['margin']:.2f} | {100*c['n_bars_same']['margin']:.2f} | {d['prediction']} | {d['retire']} | {d['verdict']} |")
    lines += ['', '## Scores and paired changes', '', 'Percent; brackets are 95% intervals. Baseline rows show absolute scores and zero change.', '',
        '| Cell | Seed | Arm | Level | Level Δ (pp) | Same bars | Bars Δ (pp) | Structural ready, offset | Structural ready, raw | Guarded ready, offset |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for panel in PANELS:
        for cell in CELLS:
            for seed in r['seeds']:
                for arm in ARMS:
                    c=r['comparisons'][panel][cell][arm]
                    def val(metric,which='treatment'):
                        return fmt(c[metric]['seeds'][seed][which])
                    lines.append(f'| {panel}/{cell} | {seed} | {arm} | '+ ' | '.join([
                        val('n_level'),val('n_level','delta'),val('n_bars_same'),val('n_bars_same','delta'),
                        val('structural_ready'),val('raw_structural_ready'),val('guarded_ready')])+' |')
    lines += ['', '## Band accuracy and adjacent-band confusion', '',
        'Frame accuracy uses every mapped 100-fps frame, then equal set and recording means. Regimes are chart red-line '
        'intervals, including same-band red-line changes; posterior aggregation is mean log probability without the song bonus. '
        'Frame argmax is identical before and after positive temperature calibration (asserted for every panel frame). '
        'Baseline and own_d2 share the same band posterior. Oracle-D2 still uses the model period head. '
        'All 7×7 integer confusion matrices, mapped-frame counts and omitted-regime counts are in results.json; rows are target bands, columns predicted bands, ordered -2..4.', '',
        '| Cell | Seed | Arm | Frame correct | Frame −1 band | Frame +1 band | Regime correct | Regime −1 band | Regime +1 band |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for panel in PANELS:
        for cell in CELLS:
            for seed in r['seeds']:
                for arm in ARMS:
                    a=r['accuracy'][panel][cell][seed][arm]
                    lines.append(f'| {panel}/{cell} | {seed} | {arm} | '+ ' | '.join(fmt(a[u][m])
                        for u in ('frame','regime') for m in ('correct','adjacent_lower','adjacent_higher'))+' |')
    lines += ['', '## Training, calibration and use of features', '',
        '| Seed | Head | Selected / run epochs | Temperature | Cal NLL | Shuffled-feature NLL | Shuffle ΔNLL |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for seed in r['seeds']:
        for head,t in r['training'][seed]['heads'].items():
            lines.append(f"| {seed} | {head} | {t['best_epoch']} / {len(t['curve'])} | {t['temperature']:.4f} | "
                f"{fmt(t['cal_nll'],False)} | {fmt(t['shuffled_nll'],False)} | {fmt(t['shuffle_delta'],False)} |")
        pool=r['training'][seed]['pool']
        lines += ['',f"{seed} pooling band weight: {pool['weight']}; candidates and calibration losses: `{json.dumps(pool['candidates'])}`."]
        for split,a in r['ambiguous_targets'][seed].items():
            lines.append(f"{seed}/{split}: {a['ambiguous_frames']} of {a['mapped_frames']} mapped frames have conflicting chart bands "
                f"({100*a['frame_weighted_share']:.3f}% frame-weighted; recording mean {fmt(a['recording_mean'])}%).")
        q=r['fp16_reconstruction'][seed]
        lines.append(f"{seed} cached fp16 own-head reconstruction vs exact E2: max absolute band-probability difference "
            f"{q['max_abs']:.6g}, mean {fmt(q['mean_abs'],False)}, argmax disagreement {fmt(q['argmax_disagreement'])}%. "
            'Calibration uses cached logits and evaluation uses exact E2; no panel adjustment was made.')
    lines += ['', '## Structural first-error classes', '',
        'Each eight-bar span receives one first-error class under the corrected X6 whole-song one-to-one matcher. '
        'Ready means every span is correct, offset-only or fine. Class counts and cluster intervals are in results.json.', '',
        '| Panel | Arm | Alignment | Ready | Spans | First-error counts |',
        '| --- | --- | --- | ---: | ---: | --- |']
    for panel in PANELS:
        for arm,alignments in r['structural'][panel].items():
            for alignment,s in alignments.items():
                lines.append(f"| {panel} | {arm} | {alignment} | {fmt(s['ready'])} | {s['spans']} | `{json.dumps(s['classes'],sort_keys=True)}` |")
    lines += ['', '## Deviations, failures and limits', '']
    lines += [f'- {d}' for d in r['deviations']] or ['No preregistered method deviation.']
    lines += ['', 'Preserved failed paths:']
    lines += [f'- {d}' for d in r['failures']] or ['None.']
    lines += ['', 'The fixed audio representation cannot identify mapper intent when charts disagree. These heads use a bounded '
        'frame sample and fixed optimisation settings; a failed readout is not proof that all band information is absent. '
        'The fixed tracked path cannot repair missed beats or boundaries. Structural ranking has no independent human '
        'label validation in this follow-up; results characterise charts and do not authorise adoption.', '',
        'Selected curves, model weights, normalisers, feature/chart hashes, decoded maps and per-recording rows are preserved '
        'under this directory. The frozen margin inventory includes X5’s neighbouring-checkpoint spread. '
        'Public commits contain only experiment code and synthetic tests; production N is unchanged.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--seeds',nargs='+',choices=SEEDS,default=list(SEEDS))
    summarize(parser.parse_args().seeds)
