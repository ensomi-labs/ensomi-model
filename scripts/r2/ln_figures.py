"""Plot source coverage, conditional release hazard and free-running LN share.

Requires matplotlib in the analysis environment. Reads only the compact outputs
of separation.py, ln_probe.py and ln_analysis.py; never opens corpus files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ln_analysis import interval
from separation import disk_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    root = Path(p.parse_args().out)
    disk_guard(root)
    classes = ['normal', 'ln_heavy', 'inverse']
    colors = ['#2369a4', '#d18120', '#9a4380']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    census = pd.read_csv(root / 'representation-summary.csv')
    census = census[census.role == 'fit_dev'].set_index('ln_class').loc[classes]
    ax = axes[0, 0]
    ax.bar(np.arange(3) - .18, census.exact * 100, .36, label='Exact')
    ax.bar(np.arange(3) + .18, census.within5 * 100, .36, label='Within 5 ms')
    ax.set(xticks=range(3), xticklabels=['Normal', 'LN-heavy', 'Inverse (n=3)'],
           ylabel='Source LN releases (%)', ylim=(0, 105), title='A. What the candidates can express')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    hazard = pd.read_csv(root / 'hazard-age_ms.csv')
    hazard = hazard[(hazard['sample'] == 'panel') & (hazard.checkpoint == 56) & ~hazard.eos]
    bins = ['[0.0, 30.0)', '[30.0, 60.0)', '[60.0, 125.0)', '[125.0, 250.0)',
            '[250.0, 500.0)', '[500.0, 1000.0)', '[1000.0, 2000.0)', '[2000.0, inf)']
    for cls, color in zip(classes, colors):
        h = hazard[hazard.ln_class == cls]
        h = h.assign(o=h.observed * h.n, p=h.predicted * h.n).groupby('age_bin')[['o', 'p', 'n']].sum().reindex(bins)
        ax.plot(range(8), h.o / h.n, color=color, marker='o', label=cls.replace('_', '-'))
        ax.plot(range(8), h.p / h.n, color=color, linestyle='--')
    ax.set(xticks=range(8), xticklabels=['<30', '30–60', '60–125', '125–250', '250–500', '500–1k', '1–2k', '≥2k'],
           ylabel='Release probability per held opportunity', ylim=(0, 1),
           title='B. Human history: source (solid), model (dashed)', xlabel='Age at decision (ms; rare ages have small n)')
    ax.tick_params(axis='x', labelrotation=35, labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    fr = pd.read_parquet(root / 'fr-summary.parquet')
    counts = fr.groupby('source_class').sha.nunique()
    labels = [f'{name} ({counts.get(cls, 0)})' for name, cls in
              zip(('Normal', 'LN-heavy', 'Inverse'), classes)]
    for mode, ax, letter in [('bos', axes[1, 0], 'C'), ('prefix', axes[1, 1], 'D')]:
        for j, cls in enumerate(classes):
            s = fr[(fr['mode'] == mode) & (fr.source_class == cls)]
            arms = [('human', None), ('baseline', 0), ('baseline', 3)]
            for i, (arm, kappa) in enumerate(arms):
                selected = s[s.arm == arm]
                if arm != 'human':
                    selected = selected[(selected.kappa == kappa) & (selected.checkpoint == 56)]
                stat = interval(selected.groupby('sha').ln_share.mean())
                x, y = j + (i - 1) * .2, stat['effect']
                error = np.maximum([[y - stat['lo']], [stat['hi'] - y]], 0)
                ax.errorbar(x, y, yerr=error, fmt=['s', 'o', '^'][i],
                            color=colors[i], capsize=3,
                            label=['Human', 'κ=0', 'κ=3'][i] if j == 0 else None)
        ax.set(xticks=range(3), xticklabels=labels,
               ylabel='LN heads / all heads', ylim=(0, 1),
               title=f'{letter}. ' + ('From the start' if mode == 'bos' else 'After a first-third human prefix'))
        ax.axhline(.5, color='gray', linewidth=.6, linestyle=':')
        ax.axhline(.8, color='gray', linewidth=.6, linestyle=':')
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(root / 'ln-contribution.svg')
    fig.savefig(root / 'ln-contribution.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
