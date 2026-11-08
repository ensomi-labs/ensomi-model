"""Count schedule and completion leaks on an existing fit_dev panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ln_probe import heldout_chart
from separation import disk_guard, grid_of
from ensomi_model.r2.data import chart_from_cache
from ensomi_model.r2.release_diagnostics import opportunity_profile, leak_counts


def summarize(table):
    columns = ['head_gaps', 'changed_prefix_gaps', 'held_gaps', 'changed_prefix_held_gaps',
               'full_occupancy_gaps', 'completion_pressure_gaps', 'forced_events_two_orders']
    rows = []
    for mode, part in table.groupby('mode'):
        for cls in ['all'] + sorted(part.source_class.unique()):
            selected = part if cls == 'all' else part[part.source_class == cls]
            for band in ['all'] + sorted(selected.band.unique()):
                sub = selected if band == 'all' else selected[selected.band == band]
                counts = sub[columns].sum().to_dict()
                row = dict(mode=mode, source_class=cls, band=band, songs=len(sub), **counts)
                for name, numerator, denominator, scale in [
                    ('schedule_rate', 'changed_prefix_gaps', 'head_gaps', 1),
                    ('held_schedule_rate', 'changed_prefix_held_gaps', 'held_gaps', 1),
                    ('full_occupancy_rate', 'full_occupancy_gaps', 'head_gaps', 1),
                    ('completion_pressure_rate', 'completion_pressure_gaps', 'head_gaps', 1),
                    ('forced_replay_per_gap', 'forced_events_two_orders', 'head_gaps', 2),
                    ('forced_replay_per_full_gap', 'forced_events_two_orders', 'full_occupancy_gaps', 2)]:
                    row[name] = counts[numerator] / (scale * counts[denominator]) if counts[denominator] else None
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['cache', 'panel', 'out']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    disk_guard(root)
    records = []
    for row in json.loads(Path(args.panel).read_text())['charts']:
        dec = heldout_chart(row, args.cache)
        chart = chart_from_cache(dec)
        profile = opportunity_profile(dec.head_ms, grid_of(dec))
        for mode, cut in [('bos', 0), ('prefix', dec.K // 3)]:
            records.append(dict(sha=row['sha256'], group=row['group_id'], source_class=row['ln_class'],
                                band=row['band'], mode=mode, **leak_counts(chart, profile, cut)))
    disk_guard(root)
    table = pd.DataFrame(records)
    table.to_csv(root / 'source-leaks.csv', index=False)
    summarize(table).to_csv(root / 'source-leaks-summary.csv', index=False)
    print(json.dumps(dict(charts=table.sha.nunique(), head_gaps=int(table[table['mode'] == 'bos'].head_gaps.sum()))))


if __name__ == '__main__':
    main()
