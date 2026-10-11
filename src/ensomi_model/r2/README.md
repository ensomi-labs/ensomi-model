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
| `ln_level.py` | Whole-song LN share and the fit_train empirical prior by star band and density tercile |
| `defect_references.py` | Closure-owned hold defects and fit_train reference rates by star band |
| `train_ce.py` | Trainer: checkpoints, exact resume, NaN recovery, logs, pilot and free-run modes |
| `sampling.py`, `export.py`, `report.py` | Free-running generation with an optional per-decision head-mask bias, `.osu` export, free-run summary |
| `strain.py` | Frozen `ras-v1` replay-backed workload and reference prefix sums, scope ratios and budgets, 15-mask costs and probability masses, fixed-eta sampling bias |
| `c0.py` | Source strain distributions, fixed-tilt scans, bounded analytic workload control, full decision likelihoods, replayable scope trajectories, paired proxy overhead, blind comparison export with sealed provenance, and masked R/N/NB allocation comparisons |
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

## Training recipe: two phases (plan v5)

R2 v2 trains in two phases, each a run of `train_ce.py` with its own config, budget and
learning-rate schedule:

| Phase | Config | What trains | Data and loss |
| --- | --- | --- | --- |
| N (natural) | `configs/ce_v2_n.json` (`phase: natural`) | every parameter outside the conditioner modules (`R2Model.parameter_split`) | `draw.selection: natural`: the v1 start rule, empty tracks; uniform per-decision CE on the fixed divisor N_bar |
| C (conditions), frozen base | `configs/ce_v2_c_frozen.json` (`base_mode: frozen`) | the conditioning path only (`film`; its width and depth are `film_width`, `film_layers`) | the aligned condition draw with residual difficulty; CE on decisions that read an interval (V_k non-empty) plus the LN and difficulty terms and, if `mu_star > 0`, the relaxed-proxy term |
| C, KL-held base | `configs/ce_v2_c_kl.json` (`base_mode: kl`) | every parameter | as above, plus `kl_weight` times the KL to the frozen phase-N model on `kl_decisions` (`natural` or `all`) in `kl_direction` (`forward` KL(reference ‖ model) or `reverse`); `natural_ce` keeps CE on natural decisions |

Phase C starts from `init_from`, a phase-N checkpoint: its natural parameters are loaded and the
conditioning path keeps its own initialisation, whose output layer is zero. A decision that reads
no interval passes through FiLM unchanged, so phase C starts as the phase-N model exactly, and
under a frozen base every natural decision stays exactly the phase-N decision.

Undecided values are `null` in the configs: the budget (`total_exposures`, `checkpoint_every`,
`g3c_exposures`), the learning rate (`lr_schedule`, or `lr`, `lr_min`, `warmup_exposures`),
`n_bar*` from `draw_sim`, and in phase C `init_from`, `lambda_ln`, `lambda_star`, `mu_star` and the
KL keys. The trainer refuses to start until each is set (`train_ce.check_config`), including for a
pilot, so pass them with `--set` there. `star_conditions: on` refuses to start without the complete
v2 label file. The open decisions are listed in plan v5 (`r2-condition-plan-v5` in the relay notes).

Measure the divisors per phase (the two phase-C configs share one draw):

```
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.draw_sim --config src/ensomi_model/r2/configs/ce_v2_n.json \
  --draws 2000 --out artifacts/r2-stage0/draw-sim/n.json
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.draw_sim --config src/ensomi_model/r2/configs/ce_v2_c_frozen.json \
  --draws 2000 --out artifacts/r2-stage0/draw-sim/c.json
```

and copy `config_values` from each output into the configs.

### Whole-song LN-level input

Phase N optionally reads a single LN share from the start of the song through EOS. Set
`ln_level: "on"` to append three shared query channels: known, level, and
`logit(clip(level, 1e-6, 1-1e-6))`. Unknown levels give three zeros. The default, `"off"`,
keeps the 150-channel query and existing parameter shapes. With the input on, the query has
153 channels; a separate bias-free `ln_level_reader` adds the three channels to the first
query layer before GELU. Its weights start at zero, preserving the original model function.

`warm_start` loads every parameter of a phase-N checkpoint, including the conditioner.
The model configuration and tensor shapes must match; enabling the LN input permits only
the new, exactly zero reader weight to be absent. Missing legacy weights, unexpected weights,
or other configuration differences raise. The optimizer and exposure counter start fresh.
Phase C continues to use `init_from`.

Each training window uses its source chart's whole-song LN share (LN heads / all heads).
`ln_level_dropout` defaults to 0.3; a separate, checkpointed RNG (`seed_ln_level`, default 1471)
replaces the level with unknown without changing the window or condition draw streams.
Generation accepts unknown, oracle, prior, or a fixed numeric level in [0, 1]. The prior
samples one fit_train chart's share from the applicable star-band and skeleton-density cell,
using the generation seed and skeleton identity independently of action sampling. Density
is head rows per second over the whole song. Its terciles are fitted within each star band;
threshold ties enter the upper cell. An empty cell raises instead of borrowing another
cell's support.

Pass the mode or numeric value as `sampling.continue_chart(..., ln_level=...)`.
Individual `Evaluator.run` calls use `Evaluator.ln_mode`, which also accepts a numeric value.
Full `Evaluator.panels` calls evaluate oracle, unknown, and prior modes with the input on.

Build both fit_train reference files before full evaluation:

```sh
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.defect_references --cache artifacts/r2-cache/v1
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.ln_level --cache artifacts/r2-cache/v1
```

These write `defect-references-v1.json` and `ln-level-prior-v1.json` beside the cache.
With the LN input on, full evaluation reports natural panels and natural-manifest NLL under
oracle, unknown, and prior levels. Selection uses the prior panels and prior NLL; a chart's
prior level is reproducible across checkpoints. Oracle results measure behavior with the
source level supplied, while prior results match generation without that source information.

`configs/ce_v2_n_lnlevel_ft.json` warm-starts the 48M phase-N checkpoint and trains for 12M
new exposures: 200k linear warmup, peak `lr=1e-4`, cosine decay to `lr_min=3e-5`, and a full
evaluation every 4M exposures. A null `lr_schedule` selects this built-in warmup/cosine
schedule:

```sh
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.launch \
  --run-id r2-lnlevel-ft-20261007 \
  --config src/ensomi_model/r2/configs/ce_v2_n_lnlevel_ft.json
```

Set `ln_length: "on"` alongside `ln_level: "on"` to add two more shared query
channels: a known bit and the whole-source median of `log2` hold lengths in grid
beats. Every hold participates, including snapped gap releases at EOS. Beat
length is `grid.beat(release) - grid.beat(head)`, with a `1e-4` floor before the
logarithm. Nonfinite or nonpositive lengths raise before flooring. Fewer than
ten holds give an unknown length (two zeros), while share may remain known.
Length requires complete source decisions through EOS; it is never estimated
from the prefix or training window. `ln_length_reader` is a separate bias-free,
zero-initialized query projection. Turning it off preserves the share-only
model and RNG stream. Warm start permits only the explicitly added zero reader
weights to be absent. Both inputs use the same existing dropout draw.

Length-enabled prior generation uses `ln-level-prior-v2.json`. Each observation
contains one source chart's share, length or null, and source SHA. One empirical
draw supplies both values. V2 keeps the band and density-cell rules but uses a
versioned seed domain; V1 fitting and seeded sampling remain unchanged. Fit V2
with an explicit output path:

```sh
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.ln_level \
  --cache artifacts/r2-cache/v1 --with-length \
  --out artifacts/r2-lnlevel2-20261007/ln-level-prior-v2.json
```

Pass the loaded prior to `continue_chart` or `generate` as `ln_prior`, or use
`generate --ln-level prior --ln-prior <path>`. Unknown mode clears both values;
oracle uses the explicit whole-source share and length as diagnostics. A fixed
numeric share accepts optional `ln_length` (`--ln-length` on the CLI) in log2
beats; omitting it leaves length unknown. Generation records include length's
known bit, value, and units. `Evaluator.ln_mode` and `Evaluator.ln_length` expose
the same fixed controls, and `TrainConfig.ln_prior` selects an explicit prior
path for evaluation. Neither training nor evaluation fits or writes a prior.

`configs/ce_v2_n_lnlevel2_ft.json` changes the share-only recipe in three places:
it starts from the 56M checkpoint, enables length, and sets `full_eval_every=4`.
Its three checkpoints at 4M, 8M, and 12M new exposures run teacher-forced
evaluation only, including evaluation retried after a resume. Supply the V2
prior path as a runtime `ln_prior` override for these evaluations. All other
training values, including the null `lr_schedule`, are identical.

### Natural-panel selection guards

Guard (iv) v2 counts holds whose closing decision was generated by the model, including
holds opened in a copied prefix and closed after it. Holds closed inside the prefix enter
neither numerator nor denominator. A short hold lasts strictly less than 60 ms. A near-head
release occurs 1–40 ms before another lane's head. For each defect, the expected count sums
each run's model-made holds times its fit_train source rate in `common.band_of(cache star)`.
The overall count must be at most 1.25 times that expectation; per-band ratios are diagnostic.
The original `defects` fields and nonbinding `iv_v1` remain available for comparison.

Guard (v) requires the absolute mean natural-BOS LN-share drift (last third minus first
third) to be at most 0.05. BOS/source correlation and SD ratio use one seed-averaged value
per chart. Dense-row LN-birth rates count LN heads among heads whose next head row occurs
within 60 ms. These readouts are diagnostic. Phase-N selection binds legality and guards
(i), (iii), (iv) v2, and (v), then applies the three-checkpoint NLL plateau rule in `select.py`.

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
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.train_ce --config src/ensomi_model/r2/configs/ce_v2_n.json \
  --pilot 200 --device cpu --threads 4 --pilot-out artifacts/r2-runs/pilot/cpu.json \
  --pilot-checkpoint artifacts/r2-runs/pilot/cpu.pt --set lr=<pilot lr> --set lr_min=<x> \
  --set warmup_exposures=<x> --set total_exposures=<x> --set checkpoint_every=<x> --set g3c_exposures=[]
```

The pilot's values are pilot-only; the same command with `ce_v2_c_frozen.json` or `ce_v2_c_kl.json`
(plus their phase-C keys) measures phase C.

Overrides for spec section 6 step 3: `--memory none`, `--levels 6`. Free-run sanity from the
pilot checkpoint (four fixed fit_dev charts, seeds 954 to 956):

```
PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.train_ce --config src/ensomi_model/r2/configs/ce_v2_n.json \
  --run-dir artifacts/r2-runs/pilot --freerun-only artifacts/r2-runs/pilot/cpu.pt <the pilot's --set values>
```

Each phase's budget is an open decision of plan v5; the pilot's decisions/s converts it to hours
(`total_exposures = 0.9 x decisions/s x 3600 x hours`, DEVIATIONS.md item 9).

## Launch

One run per phase; phase C names the selected phase-N checkpoint in `init_from` (a decided value
in the config, or `--set init_from='"<path>"'`):

```
ens run ensomi-model --name r2-ce-<run-id> -- bash -c 'PYTHONPATH=src .venv/bin/python -m ensomi_model.r2.launch \
  --run-id <run-id> --config src/ensomi_model/r2/configs/ce_v2_n.json --set device=cpu --set threads=4'
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
