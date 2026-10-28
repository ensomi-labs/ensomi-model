"""Matched recording statistics and the fixed G3 uncertainty rule."""
from collections import defaultdict
import numpy as np
from ..scoring import interval

METRICS = ('attack_auc', 'cell_auc_chart', 'cell_auc_decoded', 'salience_spearman')


def table(rows, metric, panel=None):
    groups = defaultdict(list)
    for row in rows:
        v = row.get(metric)
        if v is not None and np.isfinite(v) and (panel is None or row['panel']==panel):
            groups[(row['panel'],row['rec'])].append(v)
    return {key:float(np.mean(vals)) for key,vals in groups.items()}


def paired(control, treatment):
    keys = sorted(set(control) & set(treatment))
    if not keys:
        return None
    a = np.array([control[k] for k in keys])
    b = np.array([treatment[k] for k in keys])
    delta = interval(b-a)
    half = (delta['ci95'][1]-delta['ci95'][0])/2
    return dict(n=len(keys),control=interval(a),treatment=interval(b),delta=delta,
                paired_half_width=half,margin=max(.01,half),
                control_unmatched=len(control)-len(keys),treatment_unmatched=len(treatment)-len(keys))
