"""Compare lane pairing on one Mapperatorinator event stream without resampling.

The supported November exporter already pairs holds per lane. The counterfactual
below restores only the older single-pending-head behavior. Everything else,
including upstream overlap snapping, is identical. The saved forward diff is
the minimal per-lane fix. Event streams and both charts belong in private results.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import difflib
import gzip
import inspect
import json
import logging
from pathlib import Path
import textwrap


def single_pending_source(source):
    """Invert only the per-lane pairing changes in the pinned generate method."""
    # A single dictionary slot is equivalent to the older global pending head;
    # keeping the surrounding method intact isolates pairing from every other fix.
    changes = [
        ('hold_note_starts.get(column)', 'hold_note_starts.get(0)'),
        ('hold_note_starts[column] = group', 'hold_note_starts[0] = group'),
        ('hold_note_starts.pop(column, None)', 'hold_note_starts.pop(0, None)'),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('Exporter audit requires the pinned November per-lane generate method')
        source = source.replace(old, new)
    return source


class Messages(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def install(output):
    """Wrap one final export, saving the counterfactual and the exact input events."""
    import numpy as np
    from osuT5.osuT5.inference import postprocessor as module

    output = Path(output)
    base = output.with_suffix('')
    paths = [Path(str(base) + suffix) for suffix in
             ('.unpatched.osu', '.events.json.gz', '.export-audit.json', '.per-lane.diff')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Exporter audit sidecar already exists')
    original = module.Postprocessor.generate
    fixed = textwrap.dedent(inspect.getsource(original))
    # Replacements use the indentation of the dedented method.
    legacy = single_pending_source(fixed)
    namespace = dict(vars(module))
    exec(compile(legacy, '<single-pending-exporter>', 'exec'), namespace)
    unpatched = namespace['generate']
    called = False

    def generate(self, events, beatmap_config, timing=None):
        nonlocal called
        if called:
            raise RuntimeError('Exporter audit expects one final chart per process')
        called = True
        output.parent.mkdir(parents=True, exist_ok=True)
        groups, indices = module.get_groups(events, types_first=self.types_first)
        payload = dict(events=[dict(type=e.type.value, value=e.value) for e in events],
                       groups=[dict(type=g.event_type.value, time=float(g.time), x=g.x, y=g.y,
                                    timestamp_present=any(events[i].type == module.EventType.TIME_SHIFT for i in ix))
                               for g, ix in zip(groups, indices)],
                       beatmap=asdict(beatmap_config), timing=[] if timing is None else [t.pack() for t in timing])
        with gzip.open(paths[1], 'wt') as f:
            json.dump(payload, f, default=lambda x: x.item())
        paths[3].write_text(''.join(difflib.unified_diff(legacy.splitlines(True), fixed.splitlines(True),
                                                      fromfile='unpatched/generate.py', tofile='patched/generate.py')))
        state = np.random.get_state()
        diagnostics = {}
        for label, fn in [('unpatched', unpatched), ('patched', original)]:
            np.random.set_state(state)
            handler = Messages()
            self.logger.addHandler(handler)
            try:
                chart = fn(self, copy.deepcopy(events), copy.deepcopy(beatmap_config), copy.deepcopy(timing))
            finally:
                self.logger.removeHandler(handler)
            diagnostics[label] = handler.messages
            if label == 'unpatched':
                paths[0].write_text(chart)
        paths[2].write_text(json.dumps(diagnostics, indent=2) + '\n')
        return chart

    module.Postprocessor.generate = generate
