#!/usr/bin/env python3
"""Render private panel tables without promoting descriptive differences to causal claims."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

ARMS = [('r2', 'shipped'), ('r2', 'source'), ('mug', 'natural'), ('mug', 'source'),
        ('mapper', 'natural'), ('mapper', 'source'), ('mapper', 'source-inverse')]
CLASSES = ['normal', 'ln_heavy', 'inverse']


def number(value, digits=1):
    return f'{value:.{digits}f}' if pd.notna(value) and np.isfinite(value) else '—'


def render(root):
    records = json.loads((root / 'results/charts.json').read_text())
    runs = pd.read_csv(root / 'results/runs.csv')
    effects = pd.read_csv(root / 'results/effects.csv')
    hazard = pd.read_csv(root / 'results/hazard-effects.csv')
    effects = effects[(effects.sensitivity == 'all_parsed') & (effects.band.astype(str) == 'all')]
    jobs = [json.loads(s) for s in (root / 'results/jobs.jsonl').read_text().splitlines()]
    plan = json.loads((root / 'setup/jobs.json').read_text())
    latest = {j['id']: j for j in jobs if 'id' in j}
    unstarted = sum(j['id'] not in latest for j in plan['jobs'])
    selection = json.loads((root / 'setup/selection.json').read_text())
    panel = json.loads((root / 'results/panel.json').read_text())
    source_counts = Counter(row['ln_class'] for row in panel)
    lines = ['| System / arm | Normal: LN / gap / median | LN-heavy: LN / gap / median | Inverse: LN / gap / median | Legal / outputs | Wall s; peak RSS GB |',
             '| --- | --- | --- | --- | ---: | --- |']
    compact = []
    for system, arm in ARMS:
        sub = runs[(runs.system == system) & (runs.arm == arm)]
        cells = []
        for cls in CLASSES:
            e = effects[(effects.system == system) & (effects.arm == arm) & (effects.source_class == cls)]
            values = {}
            for metric in ('ln_share', 'inverse_gap_share', 'hold_p50_ms'):
                m = e[e.metric == metric]
                values[metric] = None if m.empty else m.iloc[0]
            if all(v is None for v in values.values()):
                cells.append('—')
            else:
                cell = []
                for metric, scale in [('ln_share', 100), ('inverse_gap_share', 100), ('hold_p50_ms', 1)]:
                    v = values[metric]
                    cell.append('—' if v is None or not np.isfinite(v.effect) else f"{number(v.generated_mean*scale)} ({v.effect*scale:+.1f})")
                cells.append(' / '.join(cell))
                compact.append(dict(system=system, arm=arm, source_class=cls,
                                    **{k: None if v is None else v.generated_mean for k, v in values.items()}))
        rr = [r for r in records if r['system'] == system and r['arm'] == arm]
        legal = sum(r['legal'] for r in rr)
        cost = '—' if not rr else f"{min(r['wall_seconds'] for r in rr):.1f}–{max(r['wall_seconds'] for r in rr):.1f}; {max(r['peak_rss_bytes'] for r in rr)/1e9:.2f}"
        lines.append(f"| {system} / {arm} | {' | '.join(cells)} | {legal}/{len(rr)} | {cost} |")
    pd.DataFrame(compact).to_csv(root / 'results/first-table.csv', index=False)
    lines += ['', 'LN and inverse-gap shares are percentages; median hold length is milliseconds. Parentheses give the paired mean difference from the same human source, in percentage points or milliseconds. Each song has equal weight after averaging seeds; length entries average the corresponding per-chart quantiles. Undefined source/generated metric pairs are omitted only from that metric, with counts in `results/effects.csv`. This first table includes parsed illegal charts; `legal_only` rows in that CSV are a separate sensitivity analysis.', '',
              f"The source panel contains {len(panel)} fit_dev songs: {source_counts['normal']} normal, {source_counts['ln_heavy']} LN-heavy and {source_counts['inverse']} inverse. Mapper’s initial subset contains {sum(row['mapper_selected'] for row in panel)} songs; retained counts by class appear below. These are development-panel measurements.", '',
              f"Updated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}. Completed analysis: {len(records)}/{len(plan['jobs'])} planned outputs. "
              f"Generation failures: {sum(j['status']=='failed' for j in jobs)}; budget-skipped jobs: {sum(j['status']=='skipped_budget' for j in jobs)}. "
              f"Interrupted jobs: {sum(j['status']=='interrupted' for j in latest.values())}; never started: {unstarted}. "
              'The requested Mapper normal subset can be reduced by measured runtime; missing outputs are not zero-valued charts.', '',
              '## Reading and scope', '',
              'These are audio-to-chart comparisons with independently generated timing and heads. They describe the released systems and their controls; they do not isolate the LN representation. R2’s oracle receives three source statistics, including heads per row and all-busy share, whereas the external source arms receive LN-head share and, where specified, a tag. Training data, unknown external-model overlap with this panel, model size, requested-star calibration, native postprocessing and head placement remain confounded. Mug generates head and held channels jointly by diffusion; this is not a test of strictly causal or next-head-blind release decisions.', '',
              'For Mapperatorinator, the November generator already has the per-lane exporter fix. Each final event stream is exported twice with the same RNG state: the selected per-lane method and a counterfactual differing only in global versus per-lane pending-head storage. No generation is resampled and no chart is repaired. The old May generator is not substituted. `patches/mapper-per-lane.diff` records the forward fix; per-output sidecars retain both charts, raw events, and both warning lists. Raw-stream errors can include upstream window-merging/resnapping errors, so they are not automatically model prediction errors.', '',
              '## Class attainment and remaining style measures', '',
              '| System / arm | Source class | Songs / runs | Hold p10 / p50 / p90 ms | Co-release % | Near-head % | Output class equals source / runs | Legal / runs |',
              '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for system, arm in ARMS:
        for cls in CLASSES:
            sub = runs[(runs.system==system)&(runs.arm==arm)&(runs.source_class==cls)]
            if sub.empty:
                continue
            mean = sub.groupby('sha')[['hold_p10_ms','hold_p50_ms','hold_p90_ms','co_release_share','near_head_share']].mean().mean()
            lengths = ' / '.join(number(mean[f'hold_p{q}_ms']) for q in (10,50,90))
            lines.append(f"| {system} / {arm} | {cls} | {sub.sha.nunique()} / {len(sub)} | {lengths} | {number(mean.co_release_share*100)} | {number(mean.near_head_share*100)} | {int((sub.ln_class==cls).sum())}/{len(sub)} | {int(sub.legal.sum())}/{len(sub)} |")
    lines += ['', 'Class rules and all eight style metrics come from the frozen stage-1 `separation.ln_statistics`, imported through the same analysis/probe modules. Inverse means LN share ≥80%, at least 32 eligible same-lane LN gaps, and ≥50% gaps ≤max(40 ms, a quarter of the canonical beat at release); otherwise LN-heavy means LN share ≥50%. Terminal LNs have no next-head gap. Stage 1 counts negative overlapping gaps as short; illegal outputs can therefore inflate inverse-gap/class attainment. Co-release pairs are holds with exactly the same birth time and releases within 5 ms. Near-head means an other-lane head 1–40 ms after release. These are fixed definitions, not validated style judgments.', '',
              'The primary generated grid is the heads-only canonical musical grid, using the same grid construction as the source cache. `source_grid_metrics` in `results/charts.json` repeats every style metric with the fixed human grid to expose timing/tempo confounding. The `source-grid-runs.csv`, `source-grid-effects.csv` and `source-grid-comparisons.csv` files in `results/` retain those values, paired intervals and seed spreads. This human grid is used only for scoring. All class-by-band effects, signed song-bootstrap intervals, absolute gaps, seed RMS and valid counts are in `results/effects.csv`; per-chart values and denominators are in `results/runs.csv`.', '',
              '## Marginal release hazard by hold age', '',
              'Stage 2’s unchanged `release_evaluate.duration_hazards` counts releases in [lo, hi) divided by complete holds reaching lo. Bins are 0–60, 60–125, 125–250, 250–500, 500–1000, 1000–2000 and ≥2000 ms. Hazards use exported holds with 0 ≤ head < release ≤ decoded audio end. The filter counts excluded nonpositive or outside-audio exported holds. Events or frame fragments omitted by native conversion never enter the hazard calculation; their diagnostics are reported separately below. This is a marginal duration comparison, not conditional calibration on human histories or comparable release-decision opportunities. The open-ended final bin is necessarily 1 whenever any complete hold reaches 2000 ms; its value alone cannot show tail agreement. `results/hazards.csv` retains risk/release counts; `hazard-effects.csv` reports paired song means, intervals and sampling spread by class and band.', '',
              '| System / arm | Source class | Generated hazard by age bin | Human hazard on matched songs/bins |',
              '| --- | --- | --- | --- |']
    for system, arm in ARMS:
        for cls in CLASSES:
            sub = hazard[(hazard.system==system)&(hazard.arm==arm)&(hazard.source_class==cls)&(hazard.band.astype(str)=='all')].sort_values('age_lo_ms')
            if sub.empty:
                continue
            lines.append(f"| {system} / {arm} | {cls} | {' / '.join(number(v,3) for v in sub.generated_rate)} | {' / '.join(number(v,3) for v in sub.human_rate)} |")
    lines += ['', '## Sampling spread and human same-song nulls', '',
              'Effects use the frozen `ln_analysis.interval`: 2,000 song bootstraps with seed 20261108. R2/Mug sampling RMS is the paired difference of seeds 954/955 on matching songs (`pair_spread`); all counts are retained. Mapper has one seed, so its sampling floor is missing, never zero; its intervals resample songs only. No new checkpoint or training-seed spread is estimated, and no acceptance margin is set. Three inverse songs give weak population evidence. A zero attainment count and a degenerate bootstrap do not bound an unseen future attainment rate.', '',
              '| Source class | Human pair creator stratum | Band match | LN-share pairs / songs | RMS pp | p95 absolute pp |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    nulls = pd.read_csv(root / 'setup/human-null-summary.csv')
    nulls = nulls[(nulls['mode']=='bos')&(nulls.metric=='ln_share')&nulls.ln_class.isin(CLASSES)]
    for row in nulls.itertuples():
        lines.append(f"| {row.ln_class} | {'same' if row.same_creator else 'different'} | {row.bands} | {row.pairs}/{row.songs} | {number(row.rms*100)} | {number(row.p95_abs*100)} |")
    lines += ['', 'The full inherited null table, with metric-specific counts, is `setup/human-null-summary.csv` and the underlying pairs are `setup/human-null.csv`. These near-identical-row same-song pairs come from stage 1’s broader source-pair pool, not just the 48 selected panel songs. They are not a perceptual ceiling or a floor matched to independently generated heads. There is no inverse–inverse null; LN-heavy different-creator nulls cover only one song.', '',
              '## Legality and exporter diagnostics', '',
              '| System / arm | Parse | Grammar | Replay reaches closed terminal state | Inside audio and legal | Raw grammar / replay | Unpatched parse / grammar / replay / legal |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for system, arm in ARMS:
        sub = [r for r in records if r['system']==system and r['arm']==arm]
        if not sub:
            continue
        counts = [sum(r.get('parse',False) for r in sub),sum(r.get('grammar',False) for r in sub),
                  sum(r.get('holds_closed',False) for r in sub),sum(r['legal'] for r in sub)]
        cells = [f'{n}/{len(sub)}' for n in counts]
        if system == 'mapper':
            raw_counts = [sum(r['raw_stream']['grammar'] for r in sub),
                          sum(r['raw_stream']['replay']['complete'] for r in sub)]
            old_counts = [sum(r['unpatched'].get(k, False) for r in sub)
                          for k in ('parse', 'grammar', 'holds_closed', 'legal')]
            cells += ['; '.join(f'{n}/{len(sub)}' for n in counts) for counts in (raw_counts, old_counts)]
        else:
            cells += ['—', '—']
        lines.append(f"| {system} / {arm} | {' | '.join(cells)} |")
    for system in ('r2','mug','mapper'):
        sub = [r for r in records if r['system']==system]
        violations = Counter()
        for r in sub:
            violations.update(r.get('violations',{}))
        lines += ['', f"{system}: output violation totals `{dict(violations)}`; excluded hazard holds {sum(r.get('hazard_excluded_holds',0) for r in sub)}."]
        if system=='mug':
            hold_loss = [r['native_hold_loss'] for r in sub if 'native_hold_loss' in r]
            lines += [f"Orphan held frames: {sum(r['receipt']['orphan_held_frames'] for r in sub)}; held lanes at padded-array end: {sum(r['receipt']['held_lanes_at_array_end'] for r in sub)}. Native minijack processing moved or removed {sum(r['receipt']['postprocess']['removed_or_moved_objects'] for r in sub)} objects, including {sum(r['receipt']['postprocess']['removed_or_moved_holds'] for r in sub)} holds. Those combined counts compare snapped objects with final objects. On {len(hold_loss)}/{len(sub)} parseable charts, raw-to-export hold counts separately show {sum(r['dropped_holds'] for r in hold_loss)} dropped holds out of {sum(r['raw_holds'] for r in hold_loss)}, across {sum(r['dropped_holds'] > 0 for r in hold_loss)} charts. Native gridify preserves object types and minijack processing only moves or removes objects, so this count reduction measures hold loss. Receipts and raw-object sidecars retain the evidence; no validator repair is applied."]
        if system=='mapper':
            for version in ('unpatched','patched'):
                warnings = [w for r in sub for w in r['export_warnings'][version]]
                lines += [f"{version} exporter warnings: incomplete heads {sum('Incomplete hold note' in w for w in warnings)}, unmatched ends {sum('Unmatched hold end' in w for w in warnings)}, untimed groups {sum('without a timestamp' in w for w in warnings)}."]
            differences = Counter()
            raw_errors = Counter()
            for r in sub:
                differences.update(r.get('export_object_difference', {}))
                raw_errors.update(r['raw_stream']['errors'])
            lines += [f"Same-stream object comparison totals: `{dict(differences)}`. Raw event grammar violations: `{dict(raw_errors)}`. A changed hold pairing appears once in each export's exclusive count, so these are not all dropped heads. The unpatched diagnostic uses one dictionary slot; its trailing warning's slot number is not a physical lane."]
    lines += ['', 'Exact replay uses Ensomi’s `ExactReplayState` on the original exported head/release times; it does not snap releases into R2 candidates. A replay failure leaves terminal closure unverified; it is not itself a count of open holds. Grammar checks include positive duration and same-lane occupancy; Ensomi includes both LN endpoints, so a same-lane head exactly at release counts as overlap. Audio span is checked separately. A legal exported chart does not establish that its formatter preserved all generated holds. `results/legality.csv` gives these counts, excluded holds and formatter diagnostics by system, arm, source class and band.', '',
              '## Inputs, controls, reproducibility and budget', '',
              f"Stage-1 snapshot commit `{selection['stage1_commit']}`; input panel SHA-256 `{selection['panel_sha256']}`. All 48 audio hashes and stage-1 source classes/LN shares were verified before generation. `results/panel.json` records source paths, stars, classes, bands, decoded lengths, oracle levels and the frozen Mapper subset. Normal selection: {selection['normal_selection']}. The active stage-2 worktree and queue were not modified.", '',
              'R2 uses the full released `System.make`: own timing and rows, one thread, shipped default controls and up to two difficulty passes. Arrangement seeds 954/955 map to make/rows seeds 0/1. Its source hook replaces only the κ=3 vector by stage-1b `source_levels`; the vector stays fixed even during difficulty correction. This includes source held-lane share, heads/row and all-busy share, not source LN-head share. Both requests and actual targets are in every receipt.', '',
              'Mug uses CPU, two threads, seeds 954/955, 100 DDIM steps, CFG 5, eta 0, native gridify/90 ms minijack processing. Natural leaves LN features unknown. Source sets the exact source LN-head share and adds ln=1 for LN-heavy/inverse, with no rice or hybrid tag for normal. Numeric feature embedding IDs are saved. The held-state grid is about 46.44 ms, and predicted within-frame offsets refine both head and release times before native snapping (`mug/data/convertor.py:array_to_objects`). Full-audio inference extends beyond the 190.218 s training window when needed, without cropping; padding and legality are reported.', '',
              'Mapper uses CPU, two threads, seed 954, the V32 mania weights and November code pinned in the smoke report, fp32, maximum batch size 1, CFG 1 for natural/source. Source sets exact hold_note_ratio; the inverse-only extra arm retains that ratio and adds gimmick/LN inverse at CFG 3. Native audio/timing inference is unchanged. Both same-stream exporters retain every other native postprocessing choice.', '',
              'The external controls quantize the requested level differently: Mug’s feature encoder floors LN ratio into 0.1-wide bins; Mapper’s tokenizer rounds it to the nearest tenth, with dedicated zero/one tokens. The continuous requested values are retained in receipts. This quantization, R2’s different three-statistic target vector, and the inverse arm’s joint descriptor/CFG change also limit attribution to representation alone. The encoding owners are Mug’s `mug/util.py` and `mania_beatmap_features.yaml`, and Mapper’s `Tokenizer.encode_hold_note_ratio`/`ratio_to_value` in the pinned source.', '',
              'Weights, environment paths, exact dependency freezes, source revisions and weight SHA-256 remain those in `../REPORT.md` and `../setup/`. No new weights or environments were downloaded. Exact argv/environment additions are in `setup/jobs.json`; process wall time and peak RSS come from `/usr/bin/time -l` logs and per-chart receipts. R2 may run two one-thread processes; Mug/Mapper run one two-thread process. Shared-Mac scheduling and repeated model loading confound speed comparisons. Peak RSS is process RSS, not all system memory.', '',
              f"Free disk at report time: {shutil.disk_usage(root).free/1e9:.2f} GB. Retained baseline-directory allocation: {int(subprocess.check_output(['du','-sk',str(root.parent)],text=True).split()[0])*1024/1e9:.3f} GB. The queue stops its own children below 100 GB free, above 5.95 GB retained, or at {plan['deadline_utc']}; the nine-hour limit ends ten minutes later. Jobs with native illegal charts remain completed observations; command failures and runtime cuts are separately recorded in `results/jobs.jsonl`.", '',
              'Mug weights and every output/event sidecar remain private and non-commercial. Only public-safe runner/analysis code is committed. No audio copies or .osz bundles are retained by the R2 wrapper. The source data, stage-2 queue, phones and live ports are untouched.', '']
    index = lines.index('## Reading and scope') + 2
    reading = root/'results/reading.md'
    if reading.exists():
        lines[index:index] = [reading.read_text().strip(), '']
    scheduling = root/'setup/scheduling.json'
    if scheduling.exists():
        schedule = json.loads(scheduling.read_text())
        if schedule.get('implemented'):
            index = lines.index('## Inputs, controls, reproducibility and budget') + 2
            lines[index:index] = [schedule['description'], '',
                                  'The original and execution manifests, their hashes, and the dispatcher restart are retained in `setup/scheduling.json` and `setup/dispatcher-restart.json`. The job dictionaries are identical; only their order changes.', '']
            restart = json.loads((root/'setup/dispatcher-restart.json').read_text())
            if restart.get('note'):
                lines[index + 4:index + 4] = [restart['note'], '']
    index = lines.index('## Sampling spread and human same-song nulls') + 2
    spread = ['| System / arm | Source class | Paired songs | LN-share gap pp [95% song CI] | Seed RMS pp / pairs |',
              '| --- | --- | ---: | --- | ---: |']
    for system, arm in ARMS:
        for cls in CLASSES:
            sub = effects[(effects.system==system)&(effects.arm==arm)&(effects.source_class==cls)&(effects.metric=='ln_share')]
            if sub.empty:
                continue
            row = sub.iloc[0]
            spread.append(f"| {system} / {arm} | {cls} | {row.songs} | {number(row.effect*100)} [{number(row.lo*100)}, {number(row.hi*100)}] | {number(row.sampling_rms*100)} / {row.sampling_pairs} |")
    lines[index:index] = spread + ['', 'Cross-system contrasts use only shared songs and are in `results/comparisons.csv`. In particular, R2-versus-Mapper normal comparisons use Mapper’s retained subset. `results/costs.csv` contains class/band runtime and memory summaries.', '']
    if (root/'results/hazards.png').exists():
        index = lines.index('## Marginal release hazard by hold age') + 2
        lines[index:index] = ['![Marginal release hazards on shared songs](results/hazards.png)', '',
                              'The plot uses songs shared by the displayed systems and, within each age bin, only songs with a defined hazard in every curve. Valid song lists and counts are retained in the plotted-data JSON. Bands bootstrap these songs; they do not add a sampling floor for Mapper. SVG and plotted values are retained beside the PNG. The inverse-descriptor extra arm remains in the numeric hazard table below.', '']
    lines += ['## Local provenance and checks', '',
              'Released R2 configuration/weight hashes and the existing BeatThis checkpoint hash are in `setup/r2-model-hashes.json`; the released files match stage 1’s published-hash verification. `setup/r2-environment.json` records the existing main environment (Torch 2.11.0); the external environments retain Torch 2.8.0. R2 generation imports the baseline checkout’s released runtime; the archived stage-1 source supplies metrics and oracle-level calculation only. `setup/r2-runtime-code.json` records the runtime source tree and verifies all R2 receipt weight hashes. No stage-2 source was edited.', '',
              'The crossing-hold exporter fixture preserves both holds with per-lane pairing and reproduces the single-head loss without it (`setup/exporter-fixture-final/`). A separate-lane, nonoverlapping fixture is unchanged, and both preserve the exact input events. The first fixture attempt omitted explicit timing and failed before producing a chart; no generation was affected. Target-hook checks verify exact source vectors and identical generated head-row times for matched R2 seeds; lane assignments and heads per row can change. Focused tests exercise raw replay failures and terminate an owned deadline child while an unrelated sentinel remains running (`logs/panel-tests.log`). CLI checks are in `setup/cli-checks.json`.', '']
    if (root/'setup/completion.json').exists():
        completion = json.loads((root/'setup/completion.json').read_text())
        lines += [f"Final code commit: `{completion['commit']}` on `codex/r2-baselines`. Finish: {completion['finished_utc']}. {completion['status']}", '']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    render(parser.parse_args().root)
