"""Little-endian array recordings and sampling with recorded random variates.

Array entries address byte ranges of ``arrays.bin``. Graph calls list their input
and output entry names; repeated tensor inputs may share an earlier byte range.
Sampling records the exponential race for multinomial, uniforms for Gumbel, and
the orientation draw. Replay consumes these variates, never the chosen indices.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
import weakref

import numpy as np
import torch


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    return np.ascontiguousarray(a, dtype=a.dtype.newbyteorder('<'))


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = (self.directory / 'arrays.bin').open('wb')
        self.arrays, self.calls, self.events, self.draws = [], [], [], []
        self.saved = {}
        self.counts = defaultdict(int)
        self.scope = ''
        self.timings = defaultdict(lambda: dict(calls=0, seconds=0.0))

    def put(self, name, value):
        old = self.saved.get(id(value))
        tracked = isinstance(value, torch.Tensor) and not value.is_inference()
        if old is not None and old[0]() is value and tracked and old[2] == value._version:
            entry = dict(old[1], name=name)
        else:
            a = array(value)
            offset = self.file.tell()
            self.file.write(a.tobytes())
            entry = dict(name=name, dtype=a.dtype.str, shape=list(np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value).shape),
                         file='arrays.bin', byte_offset=offset, nbytes=a.nbytes)
            if tracked:
                self.saved[id(value)] = (weakref.ref(value), entry, value._version)
        self.arrays.append(entry)
        return len(self.arrays) - 1

    def event(self, name, **values):
        full = f'{self.scope}/{name}' if self.scope else name
        self.events.append(dict(name=full, arrays={k: self.put(f'{full}/{k}', v) for k, v in values.items()}))

    def graph(self, name, inputs, run):
        i = self.counts[name]
        self.counts[name] += 1
        prefix = f'{self.scope}/{name}/{i}'
        feeds = {k: self.put(f'{prefix}/input/{k}', v) for k, v in inputs.items()}
        t = time.perf_counter()
        outputs = run()
        elapsed = time.perf_counter() - t
        self.timings[name]['calls'] += 1
        self.timings[name]['seconds'] += elapsed
        result = {k: self.put(f'{prefix}/output/{k}', v) for k, v in outputs.items()}
        self.calls.append(dict(name=name, scope=self.scope, step=i, inputs=feeds, outputs=result))
        return outputs

    def random(self, name, kind, values, chosen, scores):
        self.draws.append(dict(name=name, kind=kind, values=self.put(name + '/draw', values), chosen=int(chosen),
                               scores=self.put(name + '/scores', scores)))

    def close(self):
        self.file.close()
        index = dict(format='ensomi-runtime-reference-v1', arrays=self.arrays, graphs=self.calls,
                     events=self.events, draws=self.draws)
        (self.directory / 'index.json').write_text(json.dumps(index, separators=(',', ':')) + '\n')
        return dict(arrays=len(self.arrays), graph_calls=dict(self.counts), draws=len(self.draws),
                    bytes=(self.directory / 'arrays.bin').stat().st_size, graph_seconds=dict(self.timings))


class Recording:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.index = json.loads((self.directory / 'index.json').read_text())
        self.data = np.memmap(self.directory / 'arrays.bin', dtype=np.uint8, mode='r')

    def get(self, index):
        e = self.index['arrays'][index]
        return np.ndarray(e['shape'], dtype=e['dtype'], buffer=self.data, offset=e['byte_offset'])

    def event(self, name):
        e = next(x for x in self.index['events'] if x['name'] == name)
        return {k: self.get(v) for k, v in e['arrays'].items()}


class RandomSource:
    def __init__(self, recorder=None, replay=None):
        self.recorder, self.replay = recorder, replay
        self.position = 0
        self.differences = []

    def _draw(self, name, kind, draw):
        if self.replay is None:
            return draw(), None
        row = self.replay.index['draws'][self.position]
        self.position += 1
        if row['name'] != name or row['kind'] != kind:
            raise ValueError(f'Random replay diverged at {name}: expected {row["name"]}')
        return torch.from_numpy(self.replay.get(row['values']).copy()), row

    def _save(self, name, kind, values, chosen, scores, expected):
        if self.recorder is not None:
            self.recorder.random(name, kind, values, chosen, scores)
        if expected is not None and chosen != expected['chosen']:
            old = expected['chosen']
            s = array(scores).reshape(-1)
            reference = self.replay.get(expected['scores']).reshape(-1)
            self.differences.append(dict(name=name, kind=kind, reference=old, replay=chosen,
                                         original_scores=[float(reference[old]), float(reference[chosen])],
                                         replay_scores=[float(s[old]), float(s[chosen])],
                                         original_margin=float(reference[old] - reference[chosen]),
                                         replay_margin=float(s[chosen] - s[old])))

    def multinomial(self, probs, generator, name):
        # A separate generator exposes the race without changing PyTorch's draw or its RNG state.
        def draw():
            clone = torch.Generator().set_state(generator.get_state())
            return torch.empty_like(probs).exponential_(1, generator=clone)
        values, expected = self._draw(name, 'exponential', draw)
        scores = probs / values
        chosen = int(torch.multinomial(probs, 1, generator=generator)) if expected is None else int(scores.argmax())
        self._save(name, 'exponential', values, chosen, scores, expected)
        return chosen

    def gumbel(self, logits, generator, name):
        logits = logits.detach().cpu().to(torch.float64)
        values, expected = self._draw(name, 'uniform', lambda: torch.rand(len(logits), generator=generator,
                                                                        dtype=torch.float64))
        u = values.clamp_min(torch.finfo(torch.float64).tiny)
        scores = logits - (-u.log()).log()
        chosen = int(scores.argmax())
        self._save(name, 'uniform', values, chosen, scores, expected)
        return chosen

    def orientation(self, generator, name):
        values, expected = self._draw(name, 'orientation', lambda: torch.randint(2, (), generator=generator))
        chosen = int(values)
        self._save(name, 'orientation', values, chosen, torch.zeros(2), expected)
        return chosen


class OrtRuntime:
    def __init__(self, directory):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.sessions = {p.stem: ort.InferenceSession(str(p), options, providers=['CPUExecutionProvider'])
                         for p in Path(directory).glob('*.onnx')}
        self.timings = defaultdict(lambda: dict(calls=0, seconds=0.0))
        self.scope = ''

    def graph(self, name, inputs, run):
        session = self.sessions[name]
        feeds = {i.name: array(inputs[i.name]).reshape(tuple(inputs[i.name].shape)) for i in session.get_inputs()}
        t = time.perf_counter()
        values = session.run(None, feeds)
        self.timings[name]['calls'] += 1
        self.timings[name]['seconds'] += time.perf_counter() - t
        return {o.name: torch.from_numpy(v) for o, v in zip(session.get_outputs(), values)}

    def event(self, name, **values):
        pass
