"""R2 population and the group-disjoint fit_train / fit_dev assignment.

Population: corpus rows with eval_split == 'fit', api_status in {ranked, loved}
and 2 <= star <= 6 (the corpus's ordinary computed star). fit_dev holds a group
when the first eight bytes (big-endian) of SHA-256('r2-fit-dev-v1:' + group_id)
are 0 mod 10. No calibration or heldout row may be opened: ``assert_fit`` runs
before any chart or audio file is touched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import ContractError

FIT_DEV_SALT = 'r2-fit-dev-v1:'
STATUSES = ('ranked', 'loved')


class SplitError(ContractError):
    """A non-fit row reached a code path that opens chart or audio files."""


def assert_fit(row) -> None:
    split = row['eval_split'] if isinstance(row, dict) else getattr(row, 'eval_split')
    if split != 'fit':
        raise SplitError(f'R2 may only open fit rows, got eval_split={split!r}')


def role_of(group_id: str) -> str:
    digest = hashlib.sha256((FIT_DEV_SALT + group_id).encode('utf-8')).digest()
    return 'fit_dev' if int.from_bytes(digest[:8], 'big') % 10 == 0 else 'fit_train'


def population(df):
    """Rows of the corpus DataFrame admitted to R2 before file checks."""
    mask = (df.eval_split == 'fit') & df.api_status.isin(STATUSES) & (df.star >= 2) & (df.star <= 6)
    return df[mask].copy()


def assignment(group_ids) -> dict[str, str]:
    return {g: role_of(g) for g in sorted(set(group_ids))}


def write_assignment(path: Path, groups: dict[str, str]) -> str:
    text = json.dumps(dict(salt=FIT_DEV_SALT, rule='sha256(salt+group_id)[:8] big-endian % 10 == 0 -> fit_dev',
                           groups=groups), sort_keys=True, separators=(',', ':'))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def verify_assignment(path: Path) -> int:
    data = json.loads(Path(path).read_text())
    if data['salt'] != FIT_DEV_SALT:
        raise SplitError('Stored split uses another salt')
    bad = [g for g, r in data['groups'].items() if role_of(g) != r]
    if bad:
        raise SplitError(f'{len(bad)} stored group roles differ from the hash rule')
    return len(data['groups'])
