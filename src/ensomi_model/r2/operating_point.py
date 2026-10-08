"""The default operating point of R2 v2 (plan v4 section 9.6): what "default strength" means.

The default is fixed by four things, hashed together and named by that hash in every
generation record: the training recipe (the arm's config), the frozen checkpoint selection
rule with its guards (``select.py``), the decoding, and the conditioning. Changing any of
them defines a different generator whose default adherence is measured again. The guards
are the plan's choice of operating point, not a pass threshold on adherence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .locality import RULE_L_VERSION

DECODING = dict(sampler='Gumbel maximum on the masked log-softmax (CPU float64)', temperature=1.0,
                orientation='fair coin', pointer='one lane at a time in the orientation order', truncation=None)

SELECTION_RULE = dict(
    version='r2-select-v2',
    candidates='regular-cadence checkpoints after warm-up (safe checkpoints excluded) whose natural and '
               'conditioned free runs are all legal with every head present',
    primary='natural-manifest per-decision NLL, mean over the checkpoint and its two predecessors, paired '
            'song-group bootstrap SE (2,000 resamples)',
    guards=dict(i='prefix panel: chart-paired mean difference of continuation LN share (generated - real) '
                  'within +-0.05 (binding); SD ratio reported, outside [0.5, 2] flagged',
                ii='span following at onsets: slope >= 0.7 and MAE <= 0.15',
                iii='own-history calibration gap <= 0.05',
                iv='natural BOS and prefix-natural holds closed by model decisions: strict <60 ms and '
                   'another-lane head 1-40 ms after release counts each <=1.25 times the sum of model-owned '
                   'hold counts times fit_train reference rates by common.band_of(cache star)',
                v='absolute chart-paired mean natural-BOS LN-share drift (last third minus first third) <=0.05',
                a1='released-property identity check passes'),
    nonbinding=dict(iv_v1='holds <=60 ms <=0.5% and near-head release rate <=source panel rate',
                    bos='per-chart LN-share correlation and SD ratio against source; generated/source dense-row '
                        'LN-birth rates, where the next head row is <=60 ms later'),
    rule='the earliest candidate passing every binding guard whose primary is within 2 SE of the minimum '
         'among passing candidates; none passing: no selection',
    phase_n='a phase-N run (no conditions) is selected on guards legal, (i), (iii), (iv) and (v)',
    ln_level='when LN level is on, free-run guards and natural-manifest NLL use prior mode; '
             'oracle and unknown modes are diagnostics; NLL uses one prior draw per chart at seed_validation')

CONDITIONING = dict(rule_l=RULE_L_VERSION, presence='none', eta='default (no priority, no transitions)',
                    conditioner='film', baseline_style='none (not implemented)')


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


RECIPE_KEYS_EXCLUDED = ('run_dir', 'device', 'threads', 'stop_at_unix', 'freerun', 'freerun_seeds')


def recipe_hash(config: dict) -> str:
    """Hash of the training recipe: the config without paths, devices and evaluation switches."""
    return _sha({k: v for k, v in sorted(config.items()) if k not in RECIPE_KEYS_EXCLUDED})


def selection_source_sha256() -> str:
    return hashlib.sha256((Path(__file__).parent / 'select.py').read_bytes()).hexdigest()


def operating_point(train_config: dict | None, model_config: dict | None) -> dict:
    conditioning = dict(CONDITIONING)
    if model_config:
        conditioning.update(presence=model_config.get('presence', 'none'),
                            rule_l=RULE_L_VERSION if model_config.get('rule_l', True) else 'off',
                            conditioner=model_config.get('conditioner', 'film'),
                            star_value=model_config.get('star_value', 'absolute'))
    definition = dict(recipe=recipe_hash(train_config) if train_config else None,
                      selection=dict(SELECTION_RULE, source_sha256=selection_source_sha256()),
                      decoding=DECODING, conditioning=conditioning)
    return dict(hash=_sha(definition), definition=definition)
