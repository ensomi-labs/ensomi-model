"""The one door to the corpus for the harness: refuses the held-out split and logs every call.

Rows are ranked and loved charts only (one population; the status is never a
key or a stratum). Every call appends ``{time, split, count, caller}`` to the
run's ``access_log.jsonl``. Chart files are read only from rows this returned,
and ``load_chart`` refuses a held-out row as a second guard.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ..case import Chart

HELDOUT = 'heldout'
POPULATION = ('ranked', 'loved')
COLUMNS = ('path', 'sha256', 'group_id', 'star', 'eval_split', 'canonical_bpm', 'api_status')


class HeldoutRefused(PermissionError):
    pass


class CorpusAccess:
    def __init__(self, corpus: str | Path, log_path: str | Path):
        self.corpus = Path(corpus)
        self.log_path = Path(log_path)

    def rows(self, split: str, *, caller: str) -> list[dict]:
        if split == HELDOUT:
            self._log(split, 0, caller, refused=True)
            raise HeldoutRefused('The harness never reads the held-out split')
        import pyarrow.parquet as pq
        table = pq.read_table(self.corpus, columns=list(COLUMNS)).to_pylist()
        out = [r for r in table if r['eval_split'] == split and r['api_status'] in POPULATION]
        if any(r['eval_split'] == HELDOUT for r in out):
            raise HeldoutRefused('A held-out row passed the split filter')
        self._log(split, len(out), caller)
        return out

    def _log(self, split: str, count: int, caller: str, refused: bool = False) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'), split=split, count=count,
                     caller=caller, refused=refused)
        with self.log_path.open('a') as handle:
            handle.write(json.dumps(entry) + '\n')


def load_chart(row: dict) -> Chart:
    if row.get('eval_split') == HELDOUT:
        raise HeldoutRefused(f"Refusing a held-out chart: {row.get('path')}")
    return Chart.from_osu(row['path'])
