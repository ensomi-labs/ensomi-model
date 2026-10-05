# R2 v1: CE training, generation and data cache

R2 generates a 4-key osu!mania chart over supplied head times, a heads-only beat grid and the
song length. One decision per head row is a joint 4-lane action (5 codes per lane, 625 joint
actions) together with the gap releases of the lanes it closes; a headless EOS closes the
remaining holds. The specification is the "Runnable v1 spec" of the R2 design review in the
relay notes; departures are listed in [DEVIATIONS.md](DEVIATIONS.md).

All commands run on the mac from the `ensomi-model` repository root. From the control plane,
wrap each one in `ens run ensomi-model --name <slug> -- bash -c '<command>'` (tmux and
caffeinate come with `ens run`).

## Modules

| Module | Role |
| --- | --- |
| `splits.py` | Population filter, fit-only assertion, fit_train / fit_dev group hash |
| `cache.py` | Chart to decisions (2 ms row merge, heads-only grid, `soundfile` song length, snapped releases); cache build |
| `labels.py` | LN share; tiled-star intervals and labels; label build |
| `candidates.py` | Release candidates of a gap (`cand-v1`) and their static features |
| `state.py` | Raw state, action support, replay-checked `advance`, mirror |
| `features.py` | History tokens, queries, condition frames and tokens, release factors |
| `conditions.py` | Training condition draws, dropout, `replace_interval` |
| `model.py` | `R2Model`: TCN + landmarks, FiLM / token conditioner, joint head, release pointer; `decision_log_prob`, `sequence_log_prob` |
| `data.py` | Window draws and the fit_dev manifest |
| `train_ce.py` | Trainer: checkpoints, exact resume, NaN recovery, logs, pilot and free-run modes |
| `sampling.py`, `export.py`, `report.py` | Free-running generation, `.osu` export, free-run summary |
| `launch.py` | Frozen-code launcher and restarting supervisor |

## Build the cache

```
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.cache --out artifacts/r2-cache/v1 --workers 6
```

Writes `artifacts/r2-cache/v1/{charts/<sha256>.npz, index.parquet, splits.json, summary.json,
excluded.json}`. `summary.json` has the counts, exclusions by reason, merged rows and the snap
error by star band.

## Build the star labels (after the cache)

```
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.labels --cache artifacts/r2-cache/v1 --workers 4
```

Writes `labels/star.json.gz` and `labels/star_summary.json`. With `star_conditions: "auto"` the
trainer uses star conditions only when this file is complete.

## Tests

```
PYTHONPATH=src .venv/bin/python -m pytest tests/r2 -q
```

## End-to-end smoke

`scripts/r2/smoke.sh <run-id> [cpu|mps] [total_exposures] [checkpoint_every]` launches a short
run through the launcher, kills the trainer after its first checkpoint, lets the supervisor
resume it, and prints the events, checkpoints and per-checkpoint evaluations.

## Pilot (tuning hour)

Same draws on each device; prints JSON with decisions/s, seconds per step, peak RSS and the
MPS driver-memory series:

```
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.train_ce --config src/ensomi_model/r2/configs/ce_v1.json \
  --pilot 200 --device cpu --threads 4 --pilot-out artifacts/r2-runs/pilot/cpu.json \
  --pilot-checkpoint artifacts/r2-runs/pilot/cpu.pt
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.train_ce --config src/ensomi_model/r2/configs/ce_v1.json \
  --pilot 200 --device mps --pilot-out artifacts/r2-runs/pilot/mps.json
```

Overrides for spec section 6 step 3: `--memory none`, `--levels 6`. Free-run sanity from the
pilot checkpoint (four fixed fit_dev charts, seeds 954 to 956):

```
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.train_ce --config src/ensomi_model/r2/configs/ce_v1.json \
  --run-dir artifacts/r2-runs/pilot --freerun-only artifacts/r2-runs/pilot/cpu.pt
```

Then set `total_exposures = 0.9 x decisions/s x 3600 x hours` and choose `checkpoint_every`
(see DEVIATIONS.md item 9).

## Overnight launch

```
ens run ensomi-model --name r2-ce-<run-id> -- bash -c 'PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.launch \
  --run-id <run-id> --config src/ensomi_model/r2/configs/ce_v1.json \
  --set device=cpu --set threads=4 --set total_exposures=<N> --set checkpoint_every=<M>'
```

The launcher copies `src/ensomi_model` to `artifacts/r2-runs/<run-id>/code/`, writes
`config.json` and `launch.json`, and re-executes itself from the copy as the supervisor. The
supervisor runs the trainer with `PYTHONPATH` set to the copy; on a non-zero exit other than
the NaN-limit stop it resumes from the latest checkpoint after 30 s, at most five times.

## Resume

```
ens run ensomi-model --name r2-ce-<run-id>-resume -- bash -c 'PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.launch \
  --run-id <run-id> --resume'
```

## Reading the logs

Everything is in `artifacts/r2-runs/<run-id>/` (JSON and JSONL files are mirrored to the
control plane; checkpoints and `.osu` files are not):

| File | Content |
| --- | --- |
| `run.json` | Status, device, exposures reached, wall time of the training loop (in-loop evaluations included), exposures per wall second, NaN events, restarts |
| `train.jsonl` | Every 2k exposures: action NLL/decision, gap-release NLL/decision and per gap LN, EOS NLL, natural vs conditioned NLL, lr, decisions/s, RSS, MPS memory |
| `evals.jsonl` | Every checkpoint: fit_dev CE on the 64-window manifest (same components) and the free-run report of the 4 fixed charts x 3 seeds, with the source chart's summary |
| `events.jsonl` | Start, resume, trainer exits, restarts, non-finite events, resource stops |
| `receipt-*.json`, `launch.json` | Entry point, config, frozen-code hash, git state, cache hashes, seeds, device |
| `resources.jsonl` | Resource guard snapshots |
| `trainer.log` | Trainer stdout and stderr |
| `checkpoints/ckpt-<exposures>.pt` | All checkpoints (`latest.json` names the newest) |
| `freerun/<exposures>/*.osu` | Free-run charts |

Exit codes of the trainer: 0 finished (exposure budget or `stop_at_unix`), 3 NaN-event limit
(three events; each one reloads the last checkpoint, skips the offending window and halves the
learning-rate multiplier), 4 resource guard.

## Sequence DPO (`train_dpo.py`)

`dpo_loss(policy, reference, pairs, anchor_windows, beta=0.1, lambda_ce=0.2)` is the soft-label
sequence DPO of design section 5: for each pair, both branches are replayed on their own states
and scored with `sequence_log_prob` by the policy and by the frozen reference (no gradient, eval
mode); Delta is the difference of the branch log-ratio sums, the loss is
`-q log sigma(beta Delta) - (1 - q) log sigma(-beta Delta)` averaged over pairs, plus `lambda_ce`
times the CE trainer's window loss on the anchor windows. `backward=True` backpropagates one
pair or window at a time (same gradient, one graph alive).

Pairs come only from a file of real preference pairs (`pairs_file`; schema in `load_pairs`:
fit_train chart sha, prefix length, condition track, the two branches and `q`). No real source
exists yet, so the CLI refuses to start without one, and `load_pairs` refuses records whose
`meta.source` is `synthetic`. The synthetic labellers and pair builder are test fixtures in
`tests/r2/dpo_synthetic.py`: they exist to test the DPO mechanism with a known preferred
direction and are not a preference source. `LNShareLabeller`'s fixed target ignores the state's
condition track and would reward ignoring the condition.

Training (CE checkpoint = initial policy and frozen reference; held states and CE evaluation
windows from fit_dev):

```
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.train_dpo --checkpoint <run>/checkpoints/ckpt-<N>.pt \
  --run-dir artifacts/r2-dpo/<id> --set pairs_file='"<pairs.pt>"' [--set key=value ...]
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.train_dpo --run-dir artifacts/r2-dpo/<id> --resume
```

Defaults follow design section 5: AdamW lr 1e-5, betas (0.9, 0.95), weight decay 0.01 on
matrices, clip 1, 50-step linear warmup, 8 pairs and 8 anchor windows of 256 decisions per
update, beta 0.1, lambda_CE 0.2. The run directory holds `config.json`, `pairs.pt` (validated
copy of the pair file, with its path and sha256) and `pairs_summary.json`, `train.jsonl` (per
update: beta Delta mean/std/min/max, implicit rewards beta R+ and beta R-, preference accuracy,
preference loss, anchor CE, grad norm, lr), `evals.jsonl` (every `eval_every` updates and at the
end: the same statistics over the whole pair pool, fit_dev CE on `eval_windows` windows, and on
fresh policy samples of `horizon` decisions from the held states the Monte Carlo KL to the
reference per decision with its cluster SE), `checkpoints/dpo-<step>.pt` with `latest.json`,
`events.jsonl` and `receipt-*.json`.

Tests: `tests/r2/test_dpo.py` (T1, T2, T5, mirror invariance of Delta, microbatched gradient,
exact resume, labeller counts, an integration run on synthetic charts with synthetic pairs, the
CLI refusal without a pair file, and a CLI start/resume on the cache from a fixture pair file).
