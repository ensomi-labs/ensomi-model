"""Standalone figures for the preregistered recording-level measurements."""
import json

from .common import OUT


def write():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def read(name):
        path = OUT / name
        return json.loads(path.read_text()) if path.exists() else {}

    files = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, layout='constrained')
    found = False
    max_x = max_y = 0.
    for seed, ax in enumerate(axes):
        data = read(f'P7-s{seed}.json')
        for arm, color in [('graph', '#276FBF'), ('learned0', '#B44E26'), ('learned1', '#6B7C32')]:
            points = []
            for key, values in data.get('curves', {}).items():
                if key.split(':')[0] != arm:
                    continue
                x, y = values.get('M4_away'), values.get('M3i')
                if x and y and x['mean'] is not None and y['mean'] is not None:
                    points.append((float(key.split(':')[1]), x, y))
            points.sort(key=lambda p: p[0])
            if points:
                found = True
                max_x = max(max_x, max(p[1]['mean'] for p in points))
                max_y = max(max_y, max(p[2]['ci95'][1] for p in points))
                ax.errorbar([p[1]['mean'] for p in points], [p[2]['mean'] for p in points],
                    yerr=[[max(0, p[2]['mean'] - p[2]['ci95'][0]) for p in points],
                          [max(0, p[2]['ci95'][1] - p[2]['mean']) for p in points]],
                    color=color, marker='o', markersize=3, linewidth=1, capsize=2, label=arm)
        for x in (.1, .2):
            ax.axvline(x, color='#999999', linestyle=':', linewidth=.8)
        ax.set_title(f'Feature seed {seed}')
        ax.set_xlabel('Away-share outside the marked region')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Recovery in other failing ternary regions')
    axes[0].set_xlim(0, min(1, max(.25, max_x * 1.05)))
    axes[0].set_ylim(0, min(1, max(.05, max_y * 1.05)))
    if found:
        axes[1].legend(frameon=False)
        fig.suptitle('Single marks, no confirming marks; recording-balanced means and 95% intervals')
        for suffix in ('png', 'svg'):
            path = OUT / f'P7-curves.{suffix}'
            fig.savefig(path, dpi=160)
            files.append(str(path))
    plt.close(fig)

    specifications = [
        ('P1', 'panels:offline', ['mlp-F', 'mlp-M'], 'Off-line attacks'),
        ('P3', 'panels:chart:real', ['mlp-T24-real', 'mlp-Mel24-real'], 'Ternary cells'),
        ('P5a', 'summary', ['all:exact:auc:F', 'all:exact:auc:M'], 'Exact bar repeats')]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True, layout='constrained')
    found = False
    for ax, (experiment, group, keys, title) in zip(axes, specifications):
        for seed, color in [(0, '#276FBF'), (1, '#B44E26')]:
            data = read(f'{experiment}-s{seed}.json').get(group, {})
            for index, key in enumerate(keys):
                v = data.get(key)
                if v and v['mean'] is not None:
                    found = True
                    ax.errorbar(index + (seed - .5) * .12, v['mean'],
                        yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]],
                        color=color, marker='o', capsize=3, label=f'seed {seed}' if index == 0 else None)
        ax.set_xticks([0, 1], ['Frozen features', 'Mel'])
        ax.set_xlim(-.4, 1.4); ax.set_ylim(.45, 1)
        ax.axhline(.5, color='#999999', linestyle=':', linewidth=.8)
        ax.set_title(title); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Within-recording AUC')
    if found:
        axes[-1].legend(frameon=False)
        fig.suptitle('Mapper agreement; intervals bootstrap recordings')
        for suffix in ('png', 'svg'):
            path = OUT / f'probe-comparison.{suffix}'
            fig.savefig(path, dpi=160)
            files.append(str(path))
    plt.close(fig)
    return files
