"""Training-side data: the manifest and split roles, interim strata flags, label checks and items.

- `manifest`: recordings (dedup groups), reference sets, roles, val-n1 and R1-train selection;
- `strata`: the interim hard-strata flags of design §5.1 (marked interim wherever used);
- `checks`: the label checks of design §3.2;
- `items`: reading, checking and storing labels for the manifest's items.
"""
