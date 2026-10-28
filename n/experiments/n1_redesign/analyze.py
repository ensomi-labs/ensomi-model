"""Summarise paired decoder measurements into private JSON and Markdown artifacts."""
from collections import Counter,defaultdict
from base import *
from baseline import recording_rows
from measurements import interval,margin,cluster_ratio,CLASSES

METRICS=('n_bars_same','n_level','r5','p5','guarded_ready','raw_guarded_ready','structural_ready')
SEEDS=('s0','s1')
PANELS=('panel','val_hard','pooled')
ARMS=('oracle_original','oracle_d2','own_d2','d4')


def structural_rows(rows,cell='all'):
    groups=defaultdict(list)
    for r in rows:
        if cell=='all':spans=r['spans']
        elif cell in r.get('strata',{}):
            masks=r['strata'][cell]
            spans=r['spans'] if masks is None else [s for s in r['spans'] if any(s['lo']<b and s['hi']>a for a,b in masks)]
        else:continue
        if spans:
            groups[(r['panel'],r['rec'],r['seed']+':'+r['arm'],r['aligned'])].extend(spans)
    return [dict(panel=p,rec=rec,arm=arm,aligned=aligned,cell=cell,
        ready=int(all(s['cls'] in CLASSES[:3] for s in ss)),counts=dict(Counter(s['cls'] for s in ss)),spans=len(ss))
        for (p,rec,arm,aligned),ss in sorted(groups.items())]


def select(rows,panel,cell,arm):
    return {(r['panel'],r['rec']):r for r in rows if r['cell']==cell and r['arm']==arm and (panel=='pooled' or r['panel']==panel)}


def comparisons(rows):
    result={}
    for panel in PANELS:
        result[panel]={}
        for cell in ('all','S0','S5'):
            result[panel][cell]={}
            for arm in ARMS:
                stats={}
                for metric in METRICS:
                    bs=[select(rows,panel,cell,s+':baseline') for s in SEEDS]
                    ts=[select(rows,panel,cell,s+':'+arm) for s in SEEDS]
                    keys=sorted(set.intersection(*(set(d) for d in bs+ts)))
                    keys=[k for k in keys if all(np.isfinite(d[k].get(metric,np.nan)) for d in bs+ts)]
                    if not keys:continue
                    b=np.array([[d[k][metric] for k in keys] for d in bs])
                    t=np.array([[d[k][metric] for k in keys] for d in ts])
                    noise=margin(b[0],b[1])
                    stats[metric]=dict(margin=noise,seeds={s:dict(baseline=interval(b[i]),
                        treatment=interval(t[i]),delta=interval(t[i]-b[i])) for i,s in enumerate(SEEDS)})
                result[panel][cell][arm]=stats
    return result


def scorecard(rows):
    result={}
    for panel in (*PANELS,'C1'):
        result[panel]={}
        for arm in sorted({r['arm'] for r in rows}):
            for aligned in (False,True):
                sub=[r for r in rows if r['arm']==arm and r['aligned']==aligned and
                     (r['panel'] in ('panel','val_hard') if panel=='pooled' else r['panel']==panel)]
                if not sub:continue
                classes=Counter()
                for r in sub:classes.update(r['counts'])
                result[panel].setdefault(arm,{})['aligned' if aligned else 'raw']=dict(
                    ready=interval([r['ready'] for r in sub]),classes=dict(classes),spans=sum(classes.values()),
                    class_intervals={c:cluster_ratio([dict(rec=r['rec'],hit=r['counts'].get(c,0),one=r['spans'])
                        for r in sub],'hit','one') for c in classes})
    return result


def fmt(s,percent=True):
    if not s or s['value'] is None:return 'n/a'
    scale=100 if percent else 1
    return f"{scale*s['value']:.2f} [{scale*s['ci95'][0]:.2f}, {scale*s['ci95'][1]:.2f}]"


def map_changes():
    import gzip
    counts={}
    for seed in SEEDS:
        c=Counter()
        for path in sorted((OUT/'maps'/seed).glob('*.json.gz')):
            with gzip.open(path,'rt') as f:maps=json.load(f)['maps']
            for key,b in maps['baseline'].items():
                c['sets']+=1
                for arm in ('d4','own_d2','oracle_d2'):
                    c[arm+'_changed']+=any(maps[arm][key].get(k)!=b.get(k) for k in ('beats','segments','downbeats'))
        counts[seed]=dict(c)
    return counts


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scorecards')
    args=parser.parse_args()
    baseline=json.loads((OUT/'baseline-recordings.json').read_text())
    raw=[json.loads(line) for line in (OUT/'panels.jsonl').read_text().splitlines()]
    rows=baseline+recording_rows([r for job in raw for r in job['rows']])
    sc=[json.loads(line) for line in (OUT/args.scorecards).read_text().splitlines()] if args.scorecards else raw
    sr=[r for job in sc for r in job['structures']]
    nulls=[r for job in sc for r in job['nulls']]
    mapper=json.loads((OUT/'mapper-pairs.json').read_text())
    for cell in ('all','S0','S5'):
        lookup={(r['panel'],r['rec'],r['arm']):r['ready'] for r in structural_rows(sr,cell) if r['aligned']}
        for r in rows:
            if r['cell']==cell:r['structural_ready']=lookup.get((r['panel'],r['rec'],r['arm']),math.nan)
    for r in rows:
        r['raw_guarded_ready']=int(r['raw_r5']>=.9 and r['raw_p5']>=.9)
    comp=comparisons(rows)
    structures=structural_rows(sr+nulls+mapper['structures'])
    card=scorecard(structures)
    x4=json.loads((OUT/'x4-summary.json').read_text())
    x4rows=[r for line in (OUT/'x4.jsonl').read_text().splitlines() for r in json.loads(line)]
    for panel in ('panel','val_hard','pooled','cohort40'):
        selected={r['rec'] for r in json.loads((ROOT/'artifacts/n1-stage0-20261028/events-cohort.json').read_text())['selected']}
        for seed in SEEDS:
            sub=[r for r in x4rows if r['seed']==seed and (panel=='pooled' or r['panel']==panel or panel=='cohort40' and r['rec'] in selected)]
            target=x4['cohorts'][panel][seed]
            target['class_intervals']={c:cluster_ratio([dict(r,hit=int(r['cls']==c)) for r in sub],'hit','one') for c in target['failure_classes']}
            target['phase_errors']=dict(Counter(f"{r['meter']}:{r['phase_error']}" for r in sub if r['cls']=='systematic evidence error'))
    x4['margins']={}
    for panel,ss in x4['cohorts'].items():
        stats=[ss[s]['chart'] for s in SEEDS]
        x4['margins'][panel]=max(abs(stats[0]['value']-stats[1]['value']),
            *(0.5*(st['ci95'][1]-st['ci95'][0]) for st in stats))
    pooled=comp['pooled']['all']
    verdicts={}
    verdicts['X4a_prediction']=all(x4['cohorts']['pooled'][s]['chart']['value']>=.6 for s in SEEDS)
    verdicts['X4a_retires_main_lever']=all(x4['cohorts']['pooled'][s]['chart']['value']<=.4 for s in SEEDS)
    verdicts['X4b']='gated out' if not x4['d3_gate'] else 'requires D3 result'
    def delta(arm,metric,s,cell='all'):
        return comp['pooled'][cell][arm][metric]['seeds'][s]['delta']['value']
    verdicts['X5a']={s:dict(bars=delta('oracle_d2','n_bars_same',s)>=-.01,
        level=delta('oracle_d2','n_level',s)>=0,guarded=delta('oracle_d2','guarded_ready',s)>=0) for s in SEEDS}
    verdicts['X5b']={s:dict(S5_level=delta('own_d2','n_level',s,'S5')>=.01,
        S0_no_loss=all(delta('own_d2',m,s,'S0')>=-comp['pooled']['S0']['own_d2'][m]['margin']
            for m in ('n_bars_same','n_level','guarded_ready','structural_ready'))) for s in SEEDS}
    verdicts['X5c']={s:all(abs(delta('d4',m,s))<=pooled['d4'][m]['margin'] for m in
        ('n_bars_same','n_level','guarded_ready','structural_ready')) for s in SEEDS}
    verdicts['X5_retires_fold']=any(all(delta('oracle_d2',m,s)<-pooled['oracle_d2'][m]['margin'] for s in SEEDS)
        for m in ('n_bars_same','n_level'))
    baseline_struct={s:{r['rec']:r['ready'] for r in structures if r['arm']==s+':baseline' and r['aligned']} for s in SEEDS}
    keys=sorted(set(baseline_struct['s0'])&set(baseline_struct['s1']))
    disagree=interval([baseline_struct['s0'][k]!=baseline_struct['s1'][k] for k in keys])
    human=card['C1']['human:mapper']['aligned']['ready']['value']
    model={s:card['pooled'][s+':baseline']['aligned']['ready']['value'] for s in SEEDS}
    gap={s:human-model[s] for s in SEEDS}
    human_values=np.array([r['ready'] for r in structures if r['arm']=='human:mapper' and r['aligned']])
    gap_intervals={}
    for s in SEEDS:
        mv=np.array(list(baseline_struct[s].values()))
        rng=np.random.default_rng(20261028)
        draws=human_values[rng.integers(0,len(human_values),(2000,len(human_values)))].mean(1)-mv[rng.integers(0,len(mv),(2000,len(mv)))].mean(1)
        gap_intervals[s]=dict(n_model=len(mv),n_human=len(human_values),value=gap[s],ci95=np.quantile(draws,[.025,.975]).tolist())
    null_pass={a:card['pooled']['reference:'+a]['aligned']['ready']['value']<=.05 for a in ('random_phase','random_level','x2')}
    doubled=card['pooled']['reference:x2']['aligned']
    x2level=doubled['classes'].get('level x2',0)/doubled['spans']
    verdicts['X6']=dict(nulls=null_pass,x2_level_share=x2level,x2_level_pass=x2level>=.95,
        gap=gap,gap_intervals=gap_intervals,x2_level=doubled['class_intervals'].get('level x2'),
        seed_rate_difference=abs(model['s0']-model['s1']),seed_verdict_disagreement=disagree,
        gap_exceeds_disagreement=all(v>disagree['value'] for v in gap.values()))
    verdicts['X6']['retires_ranking']=not(all(null_pass.values()) and x2level>=.95 and verdicts['X6']['gap_exceeds_disagreement'])
    guard={}
    for panel in (*PANELS,'C1'):
        source=mapper['recording_rows'] if panel=='C1' else rows
        arms=['mapper'] if panel=='C1' else [s+':baseline' for s in SEEDS]
        guard[panel]={}
        for arm in arms:
            sub=[r for r in source if r['arm']==arm and r['cell']=='all' and (panel in ('pooled','C1') or r['panel']==panel)]
            guard[panel][arm]={k:interval([int(r['raw_r5']>=.9 and r['raw_p5']>=.9) if k=='raw_guarded_ready' else r[k] for r in sub])
                for k in ('r5','p5','raw_r5','raw_p5','ready','guarded_ready','raw_guarded_ready')}
    session=json.loads((OUT/'session-coverage.json').read_text())
    lattice=json.loads((OUT/'lattice-chance.json').read_text())
    old=json.loads((ROOT/'artifacts/n1-stage0-20261028/baseline-null.json').read_text())
    checkpoint_context={p:{c:{m:old[p][c][m]['checkpoint_ranges']
        for m in ('n_bars_same','n_level','r5') if m in old[p].get(c,{})}
        for c in ('all','S0','S5')} for p in ('panel','val_hard')}
    import csv
    original=list(csv.DictReader((ROOT/'artifacts/n1-stage0-20261028/decoder-recordings.csv').open()))
    original={(q['panel'],q['cell'],q['arm'],q['rec']):q for q in original if q['arm'].endswith(':level')}
    oracle_differences=[];n_oracle=0
    for q in rows:
        if not q['arm'].endswith(':oracle_original'):continue
        key=(q['panel'],q['cell'],q['arm'].replace('oracle_original','level'),q['rec'])
        before=original[key]
        for metric in ('n_bars_same','n_level','r5','p5','guarded_ready'):
            n_oracle+=1
            if not np.isclose(q[metric],float(before[metric]),atol=1e-9,rtol=0,equal_nan=True):
                oracle_differences.append(dict(key=key,metric=metric,before=float(before[metric]),after=q[metric]))
    cohort_counts={p:{c:comp[p][c]['oracle_d2']['n_level']['seeds']['s0']['baseline']['n']
        for c in ('all','S0','S5')} for p in PANELS}
    result=dict(preregistration=json.loads((OUT/'prereg-identity.json').read_text()),scorecard_source=args.scorecards or 'panels.jsonl',
        cohort_recordings=cohort_counts,map_changes=map_changes(),
        baseline=json.loads((OUT/'baseline-reproduction.json').read_text()),X4=x4,X5=comp,
        checkpoint_context=checkpoint_context,original_oracle_reproduction=dict(compared=n_oracle,differences=oracle_differences),
        X6=dict(scorecard=card,guarded=guard,session_coverage={k:v for k,v in session.items() if k not in ('marks','counts')},
                lattice=lattice),verdicts=verdicts)
    dump(OUT/'results.json',result)
    dump(OUT/'treatment-recordings.json',rows)
    dump(OUT/'structural-recordings.json',structures)
    report(result,mapper)
    print(json.dumps(verdicts,indent=2),flush=True)


def report(r,mapper):
    pr=r['preregistration'];v=r['verdicts']
    lines=['# N1 decoder line: X4, X5, X6', '',
        f"Preregistration SHA-256: `{pr['sha256']}`; frozen UTC: `{pr['utc']}`.",
        'Baseline source: `b51a9ff`; evidence: both E2 checkpoint-009001 seeds. CPU only, nice 10, two workers.',
        f"Baseline: {r['baseline']['jobs']} recording/seed jobs, {r['baseline']['compared_scores']} score comparisons; exact timing maps and E1 scores reproduced with zero differences.", '',
        'The fit_dev panel has 322 reference sets in 262 recordings; val-n1 hard has 150 sets in 150 recordings. '
        'The pooled panel has 472 sets in 412 recordings. All model arms use both seeds at the same checkpoint and the same fixed tracked path. '
        'The frozen definitions and thresholds are in [prereg.md](prereg.md).', '',
        'Intervals below are recording-bootstrap 95% intervals (2,000 draws); values and differences are percentages / percentage points. '
        'The margin is the larger baseline seed difference or baseline bootstrap half-width. '
        'The 40-song subset is included in fit_dev and is not counted again in the pool.', '',
        '## X4: evidence and decoder', '',
        '| Cohort | Seed | Failing runs | Recordings | Chart evidence right % [95% CI] | Decoded evidence right % [95% CI] |',
        '| --- | --- | ---: | ---: | --- | --- |']
    lines[lines.index('## X4: evidence and decoder')+1:lines.index('## X4: evidence and decoder')+1]=[
        '', 'A run contains at least four complete consecutive chart bars with the same notated beat count. '
        'Same-bars requires each start and end within 20 ms and no intervening predicted bar start. '
        'On baseline-failing runs, maximise the summed sampled downbeat log odds over meters 2–8 and every phase; '
        'the true chart meter/phase must be a maximum. The primary share pools runs, with recording bootstrap clusters.']
    for panel,ss in r['X4']['cohorts'].items():
        for s,x in ss.items():lines.append(f"| {panel} | {s} | {x['runs']} | {x['chart']['n']} | {fmt(x['chart'])} | {fmt(x['decoded'])} |")
    lines+=['', 'Chart-evidence noise margins (pp): '+', '.join(f"{p} {100*m:.2f}" for p,m in r['X4']['margins'].items())+'.']
    lines+=['',f"Prediction (>=60% both seeds): **{v['X4a_prediction']}**. Retire decoder repair as main lever (<=40% both): **{v['X4a_retires_main_lever']}**.",
        f"D3 gate: **{r['X4']['d3_gate']}**. X4b status: **{v['X4b']}**.", '',
        'Neither the >=60% prediction nor the <=40% retirement threshold holds on both seeds. '
        'D3, its shuffled-emission null and the stage 0c negative control were not run because the preregistered gate failed; D3 itself is untested.', '',
        'The requested unnormalised sum over downbeat log odds is sensitive to candidate meter rate. '
        'This characterises the stated decode-free criterion; it does not prove the head contains no usable bar information.', '',
        '| Cohort | Seed | Failure class | Runs | Share % [95% CI] |','| --- | --- | --- | ---: | --- |']
    for panel,ss in r['X4']['cohorts'].items():
        for s,x in ss.items():
            for c,n in x['failure_classes'].items():lines.append(f"| {panel} | {s} | {c} | {n} | {fmt(x['class_intervals'][c])} |")
    lines+=['','Systematic phase histograms and tie counts are in results.json; classes give grid failures priority while the evidence-right numerator remains independent.', '',
        '## X5: octave bands and parity', '',
        'D2 integrates the level posterior with the period-head distribution into notated octave-band probabilities, '
        'then selects a power-of-two multiplier of the fitted tracked pulse. D4 changes only half-level parity from sums to means. '
        'The original oracle comparator replaces level probabilities without D2.', '',
        '| Cohort | All recordings | S0 recordings | S5 recordings |', '| --- | ---: | ---: | ---: |']
    for panel,counts in r['cohort_recordings'].items():
        lines.append(f"| {panel} | {counts['all']} | {counts['S0']} | {counts['S5']} |")
    lines += ['',
        '| Cohort/cell | Arm | Metric | Margin pp | s0 baseline % | s0 delta pp [95% CI] | s1 baseline % | s1 delta pp [95% CI] |',
        '| --- | --- | --- | ---: | ---: | --- | ---: | --- |']
    for panel,cells in r['X5'].items():
        for cell,arms in cells.items():
            for arm,stats in arms.items():
                for m in ('n_bars_same','n_level','guarded_ready','structural_ready'):
                    if m not in stats:continue
                    x=stats[m];a,b=(x['seeds'][s] for s in SEEDS)
                    lines.append(f"| {panel}/{cell} | {arm} | {m} | {100*x['margin']:.2f} | {100*a['baseline']['value']:.2f} | {fmt(a['delta'])} | {100*b['baseline']['value']:.2f} | {fmt(b['delta'])} |")
    lines+=['',f"X5a predictions: `{json.dumps(v['X5a'])}`.",f"X5b predictions: `{json.dumps(v['X5b'])}`.",
        f"X5c within-noise prediction: `{json.dumps(v['X5c'])}`.",f"Fold diagnosis retirement rule fired: **{v['X5_retires_fold']}**.", '',
        f"Original level-oracle reproduction: {r['original_oracle_reproduction']['compared']} metric comparisons, {len(r['original_oracle_reproduction']['differences'])} differences from stage 0.", '',
        'The oracle replaces only level probabilities; period uncertainty, the tracked path, parity and bar emissions remain model-derived. '
        'The difference from the original oracle is therefore a decoder diagnostic, not deployable oracle performance. '
        'The oracle result supports the fold diagnosis under this intervention. The own-head result rejects this D2 implementation: '
        'its period/band aggregation may interact with the learned level probabilities, so oracle success does not establish a deployable repair.', '',
        'D4 changed timing maps: '+', '.join(f"{s} {x['d4_changed']}/{x['sets']}" for s,x in r['map_changes'].items())+'.', '',
        '## X6: structural scorecard', '',
        'The scorecard uses one-to-one notated matching at 15 ms, partitions the reference into eight-bar spans including final fragments, '
        'and assigns each span its first structural error. Inserted predictions count. Correct, offset-only and fine spans are structurally acceptable; '
        'one structural error in any reference set makes its recording unready. The aligned verdict subtracts one head-weighted mutual-nearest '
        'song offset capped at ±50 ms; raw uses zero. The same offset and 90% R5/P5 guard apply to mapper pairs.', '',
        f"Mapper floor: {mapper['pairs']} C1 same-audio pairs, {mapper['recordings']} recordings. C1 was selected for canonical grid agreement; these are different recordings from the model panels.", '',
        '| Cohort | Arm | Offset | Recordings | Ready % [95% CI] | Spans |','| --- | --- | --- | ---: | --- | ---: |']
    for panel,arms in r['X6']['scorecard'].items():
        for arm,modes in arms.items():
            if arm.split(':')[-1] not in ('baseline','mapper','random_phase','random_level','x2'):continue
            for mode,x in modes.items():lines.append(f"| {panel} | {arm} | {mode} | {x['ready']['n']} | {fmt(x['ready'])} | {x['spans']} |")
    lines+=['','First structural error counts (aligned; all raw and treatment counts are in results.json):','',
        '| Cohort | Arm | Class | Spans | Share % [95% CI] |','| --- | --- | --- | ---: | --- |']
    for panel in ('pooled','C1'):
        for arm,modes in r['X6']['scorecard'][panel].items():
            if arm.split(':')[-1] not in ('baseline','mapper','random_phase','random_level','x2'):continue
            for cls,n in modes['aligned']['classes'].items():
                lines.append(f"| {panel} | {arm} | {cls} | {n} | {fmt(modes['aligned']['class_intervals'][cls])} |")
    lines+=['','Guarded readiness uses the same one-to-one notated R5 and P5 for models and mapper pairs, each >=90%.','',
        '| Cohort | Arm | R5 % [CI] | P5 % [CI] | Guarded % [CI] | Raw guarded % [CI] |',
        '| --- | --- | --- | --- | --- | --- |']
    for panel,arms in r['X6']['guarded'].items():
        for arm,x in arms.items():lines.append(f"| {panel} | {arm} | {fmt(x['r5'])} | {fmt(x['p5'])} | {fmt(x['guarded_ready'])} | {fmt(x['raw_guarded_ready'])} |")
    x=v['X6']
    lines+=['',f"Null ready <=5%: `{json.dumps(x['nulls'])}`. Doubled-level spans: {100*x['x2_level_share']:.2f}% (required >=95%).",
        'Mapper-minus-model gap pp [95% CI]: '+', '.join(f"{s} {fmt(q)}" for s,q in x['gap_intervals'].items())+'. '
        f"Seed ready-rate difference: {100*x['seed_rate_difference']:.2f} pp; paired seed verdict disagreement: {fmt(x['seed_verdict_disagreement'])}%.",
        f"Retire scorecard for ranking: **{x['retires_ranking']}**. Passing nulls and a selected mapper floor does not validate ranking against independent human structural labels; none are available here.", '',
        '### Session 1 chart coverage', '',
        f"Off-pulse marks within 20 ms of a chart head after subtracting +7.7 ms: {fmt(r['X6']['session_coverage']['coverage'])}%.",
        f"Coverage denominator: {r['X6']['session_coverage']['coverage']['denominator']}; covered: {r['X6']['session_coverage']['coverage']['numerator']}. "
        f"{r['X6']['session_coverage']['passages']} passages and {r['X6']['session_coverage']['recordings']} recordings. Details remain only in session-coverage.json.", '',
        '### Uniform-time lattice chance', '', '| BPM | Half | Half/third/quarter | Through eighth/sixth |','| ---: | ---: | ---: | ---: |']
    for bpm,x in r['X6']['lattice'].items():lines.append(f"| {bpm} | {100*x['half']:.2f}% | {100*x['half_third_quarter']:.2f}% | {100*x['through_eighth_sixth']:.2f}% |")
    lines+=['','## Deviations, failed paths and limits','',
        'The first synthetic insertion fixture exposed a meter diagnosis before line matching; this was fixed before any treatment. '
        'Short final spans use up to four neighbouring intervals to identify level, while their error time is restricted to the span. '
        'S0/S5 structural guards include any eight-bar span overlapping the stratum; mixed spans are conservative and may include errors outside it.', '',
        'The initial scorecard matched independently inside each span, falsely declaring missing lines for 1 ms jitter across a cut. '
        'A new absolute fixture failed on that implementation. Matching now runs once over the song; each matched prediction belongs to its reference span, '
        'and unmatched predictions still count in their own time span. Structural scores were recomputed from saved maps without decoding again. '
        'The failed scorecard, report and summaries are preserved under failed-span-local-scorecard/. '
        'The preregistration was not edited; this is a measurement implementation correction made after observing the initial zero-ready result.', '',
        'The first C1 loader stopped on a multi-audio set key. Its retry handles the existing set/audio-stem identities and reads only their named chart files; '
        'the original error is retained in calibration.log. No pair was dropped.', '',
        'Stage 0 neighbouring-checkpoint ranges for bars, level and R5 are retained in results.json as checkpoint_context. '
        'Only the two selected checkpoint-009001 seeds are decoded here, as requested; the margin uses the frozen two-seed/bootstrap rule.', '',
        'No holdouts were opened. No weights were trained or inferred. All production decoder files remain unchanged. '
        'The immutable preregistration, job.log, input identities, per-recording rows, per-span classes and treatment maps are in this directory. '
        'Private listening data and derived results are excluded from the public branch.', '']
    (OUT/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
