"""Calibration harness v0 for the chart evaluator.

- ``access``: the one door to the corpus; refuses the held-out split and logs every call.
- ``arrays``: a chart's objects as arrays, lane-occupancy queries, the harness's placement units.
- ``injections``: defect families D1 to D10 and F1 to F7 at seeded doses, legal by construction where possible.
- ``transforms``: must-not-flag transforms and the time-stretch covariance test.
- ``descriptions``: one targeted scalar per family, for the corpus-plausible dose.
- ``trivial``: the trivial evaluator's one event family (head rate) and its model of normal.
- ``chain``: per-case score chain on the event field (``..field``) and the kernel-weighted song nulls.
- ``cases``: per-chart work in worker processes; ``run``: the entry point; ``report``: analysis and report.

Any family that follows ``field.EventFamily`` runs through ``chain`` unchanged.
The bar-passage placeholder (``..passages``) is no longer on the default path.
"""
