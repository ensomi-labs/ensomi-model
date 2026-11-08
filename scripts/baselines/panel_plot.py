#!/usr/bin/env python3
"""Plot complete-hold marginal hazards on songs shared by the displayed systems."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root):
    sys.path.insert(0, str(root/'stage1-code/scripts/r2'))
    sys.path.insert(0, str(root/'stage1-code/src'))
    from ln_analysis import interval
    from separation import jsonable
    table = pd.read_csv(root/'results/hazards.csv')
    human = {r['sha']: r for r in json.loads((root/'results/human.json').read_text())}
    colors = dict(r2='#0072B2', mug='#D55E00', mapper='#009E73', human='#222222')
    edges = [0,60,125,250,500,1000,2000]
    labels = ['0–60','60–125','125–250','250–500','500–1000','1000–2000','≥2000']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    manifest = []
    for row, mode in enumerate(['natural', 'source']):
        for col, cls in enumerate(['normal','ln_heavy','inverse']):
            ax = axes[row,col]
            series = {}
            for system in ['r2','mug','mapper']:
                arm = 'shipped' if system=='r2' and mode=='natural' else mode
                sub = table[(table.system==system)&(table.arm==arm)&(table.source_class==cls)]
                if not sub.empty:
                    series[system] = sub
            if not series:
                ax.set_title(f'{cls}: no completed runs')
                continue
            common = set.intersection(*(set(v.sha) for v in series.values()))
            ax.set_title(f'{cls.replace("_","-")} · {mode} · {len(common)} shared songs')
            if not common:
                continue
            human_values = []
            for sha in sorted(common):
                for bin in human[sha]['duration_hazards']:
                    human_values.append(dict(sha=sha, age_lo_ms=bin['age_lo_ms'],
                                             rate=bin['releases']/bin['risk'] if bin['risk'] else np.nan))
            series['human'] = pd.DataFrame(human_values)
            for system, sub in series.items():
                sub = sub[sub.sha.isin(common)]
                song = sub.groupby(['sha','age_lo_ms']).rate.mean().unstack('age_lo_ms').reindex(columns=edges)
                summaries = [interval(song[e]) for e in edges]
                mean = np.array([s['effect'] for s in summaries])
                ax.plot(range(7), mean, marker='o', ms=3, label=system, color=colors[system], lw=1.7)
                ax.fill_between(range(7), [s['lo'] for s in summaries], [s['hi'] for s in summaries],
                                color=colors[system], alpha=.1)
                manifest.append(dict(mode=mode, source_class=cls, system=system, songs=sorted(common), bins=summaries))
            ax.set_xticks(range(7), labels, rotation=35, ha='right', fontsize=8)
            ax.set_ylim(-.02,1.02)
            ax.grid(axis='y', alpha=.2)
    axes[0,0].set_ylabel('Releases / holds reaching age-bin start')
    axes[1,0].set_ylabel('Releases / holds reaching age-bin start')
    for ax in axes[1]:
        ax.set_xlabel('Hold-age bin (ms)')
    for ax in axes.flat:
        ax.set_xticks(range(7), labels, rotation=35, ha='right', fontsize=8)
    handles = {}
    for ax in axes.flat:
        hs, ls = ax.get_legend_handles_labels()
        handles.update(zip(ls,hs))
    fig.legend(handles.values(), handles.keys(), loc='upper center', bbox_to_anchor=(.5,.955), ncol=4, frameon=False)
    fig.suptitle('Marginal release hazards · chart means · shared songs · complete holds only', y=.995, fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(root/'results/hazards.png', dpi=180)
    fig.savefig(root/'results/hazards.svg')
    (root/'results/hazard-plot-data.json').write_text(json.dumps(jsonable(manifest),indent=2,allow_nan=False)+'\n')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    plot(p.parse_args().root)
