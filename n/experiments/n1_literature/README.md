# Creator-timing diagnostics

These measurements characterize frozen N1 v0.1 outputs. They do not train a
model, modify a decoder or validate a perceptual quality measure. Inputs are
existing local unsealed evaluations, mapper pairs, synthetic controls and
feature caches; none is distributed here.

Run from the repository root with its existing environment. CPU commands use
one thread and `nice`; the shift experiment requires explicit MPS and an idle
GPU. Set `PYTHONDONTWRITEBYTECODE=1` to leave source snapshots unchanged.

```sh
export PYTHONPATH=n/src:n/experiments
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

nice -n 10 python -m pytest n/experiments/n1_literature -q -p no:cacheprovider
nice -n 10 python -m n1_literature.cpu a1 --root /path/to/data-root --out /path/to/new-results
nice -n 10 python -m n1_literature.cpu a2 --root /path/to/data-root --out /path/to/new-results
nice -n 10 python -m n1_literature.cpu a3 --root /path/to/data-root --out /path/to/new-results
nice -n 10 python -m n1_literature.shift_run --root /path/to/data-root \
  --out /path/to/new-results --device mps --count 100 --minutes 60
```

Use a fresh output directory and preregister definitions before running. The
commands write JSON summaries and per-item/recording JSONL tables there. They
check for 100 GiB free space and a 2 GB output limit. The root must contain the
source run records at the relative paths in `common.py`; only the `panel` and
`val_hard` split entries and the pair run's unsealed rows are opened. Existing
seal-filtered manifests remain the authority for membership. These scripts do
not construct or alter an exclusion list.

- A1 uses the harness's signed offset, recording bootstrap, saved neighboring
  checkpoints, exact-audio mapper overlap and a calibrated band-flux reference.
  Synthetic attack calibration does not make flux a human beat annotation.
- A2 writes the three red-line-spacing class counts before agreement scores.
  Pair classes with fewer than ten pairs are not summarized. Beat This uses
  its own minimal postprocessor and the evaluation's fixed 30-ms correction;
  canonical BL-20, canonical F70 and notated F70 remain distinct.
- A3 compares the boundary logit at tempo changes and resets with background
  frames within each song. Both event and background scores use the same
  temporal max filter; an exact-frame sensitivity is also written. A cache
  without the optional head-layout field is checked against the checkpoint's
  linear projection before use.
- B averages eight translated waveforms or eight crop origins. It repeats
  each whole arm on a fresh crop grid and measures one-to-one beat agreement
  at 5 ms, plus chart BL-20, bars and level. It verifies fresh features,
  unchanged inference, identical repeats, initial saved maps and checkpoint
  hashes. A failed reproduction check stops attribution to averaging.

`test_measures.py` and `test_averaging.py` contain absolute fixtures for clock
signs, fractional boundaries, matching multiplicity, AUC ties, class thresholds
and circular/offset alignment. A changed-map rate characterizes sensitivity to
the declared crop grids. Even perfect repeatability can hide a stable error.
