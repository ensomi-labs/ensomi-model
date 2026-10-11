"""R2 prediction, release scoring and one-row history commits with tensor state.

Feature construction, legal supports and sampling stay in Python. A commit
consumes one completed head-row token and carries each TCN dilation buffer;
it never re-encodes an earlier row. Landmark models also carry their stored
hand vectors and occupied-slot mask. The released memory='none' model omits
that unused state.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from ..r2.candidates import CANDIDATE_DIM
from ..r2.features import (ANCHOR_DIM, HISTORY_DIM, LANE_QUERY_DIM, LN_LENGTH_DIM,
                          LN_LEVEL_DIM, QUERY_DIM, RELATION_DIM, THETA_DIM)
from ..r2.sampling import _history_token
from ..research.bounded_typed_continuation.temporal import pointwise


READERS = (('ln_level', LN_LEVEL_DIM), ('ln_length', LN_LENGTH_DIM),
           ('anchor', ANCHOR_DIM), ('theta', THETA_DIM))


def _film(module, values, condition):
    gamma, delta = module.mlp(condition).chunk(2, -1)
    changed = values + gamma * module.norm(values) + delta
    return torch.where((condition != 0).any(-1, keepdim=True), changed, values) if module.gate else changed


class R2Predict(nn.Module):
    """Read the carried history and score all 625 unmasked actions."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_names = ['last', 'query', 'row_condition']
        if model.config.memory == 'landmarks':
            self.input_names += ['landmarks', 'landmark_valid']
        self.output_names = ['hands', 'action_logits']

    def forward(self, last, query, row_condition, *memory):
        m = self.model
        base = pointwise(m.exact[0], query[..., :QUERY_DIM].contiguous())
        offset = QUERY_DIM
        for name, width in READERS:
            if getattr(m.config, name) == 'on':
                base = base + pointwise(getattr(m, name + '_reader'), query[..., offset:offset + width])
                offset += width
        exact = pointwise(m.exact[2], m.exact[1](base))
        hands = pointwise(m.fuse, torch.cat((last[None], exact), -1))
        if m.config.memory == 'landmarks':
            marks, valid = memory
            q, key, value = m.lm_query(hands), m.lm_key(marks), m.lm_value(marks)
            scores = torch.einsum('mhd,lhd->mhl', q, key) / math.sqrt(m.config.hidden)
            weights = scores.masked_fill(~valid[None, None], -1e9).softmax(-1)
            weights = weights * valid.any()
            hands = hands + m.lm_out(torch.einsum('mhl,lhd->mhd', weights, value))
        hands = _film(m.film, hands, row_condition[:, None])
        return hands, m.joint(hands)


class R2Commit(nn.Module):
    """Append one head row, returning the complete carried history state."""

    def __init__(self, model):
        super().__init__()
        self.temporal = model.temporal
        self.stride = model.config.stride
        self.landmarks = model.config.memory == 'landmarks'
        self.input_names = ['history_token', 'rows']
        self.state_names = ['last', 'rows']
        if self.landmarks:
            self.input_names += ['landmarks', 'landmark_valid']
            self.state_names += ['landmarks', 'landmark_valid']
        buffers = [f'buffer_{i}' for i in range(model.config.levels)]
        self.input_names += buffers
        self.state_names += buffers
        self.output_names = ['next_' + name for name in self.state_names]

    def forward(self, history_token, rows, *state):
        buffers = state[2:] if self.landmarks else state
        value = self.temporal.input(history_token)
        updated = []
        for block, buffer in zip(self.temporal.blocks, buffers):
            value, buffer = block.step(value, buffer)
            updated.append(buffer)
        output = [value, rows + 1]
        if self.landmarks:
            marks, valid = state[:2]
            write = torch.arange(marks.shape[0], device=marks.device) * self.stride == rows
            output += [torch.where(write[:, None, None], value[None], marks), valid | write]
        return tuple(output + updated)


class R2Release(nn.Module):
    """Score every candidate for one directed, oriented release factor."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_names = ['hands', 'orientation', 'codes', 'slot', 'lane_features',
                            'relations', 'candidates', 'pair_condition']
        self.output_names = ['release_scores']

    def forward(self, hands, orientation, codes, slot, lane_features,
                relations, candidates, pair_condition):
        m = self.model
        flip = orientation[:, None] == 1
        left = torch.where(flip, hands[:, 1], hands[:, 0])
        right = torch.where(flip, hands[:, 0], hands[:, 1])
        lane_slot = (torch.arange(4, device=slot.device)[None] == slot[:, None]).to(hands.dtype)
        context = m.pointer_in(torch.cat((left, right, m.code_embedding(codes).flatten(1),
                                          lane_slot, lane_features), -1))
        query = m.pointer_out(nn.functional.gelu(context + m.pointer_relation(relations)))
        query = _film(m.film, query, pair_condition)
        return (query * m.candidate(candidates)).sum(-1) / math.sqrt(m.config.hidden) + m.candidate_bias(candidates)[:, 0]


def initial_state(model, max_head_rows):
    """BOS tensors; landmark capacity covers all head rows in this song."""
    like = model.temporal.input.weight
    state = dict(last=model.temporal.boundary[0].expand(2, -1).clone(),
                 rows=torch.zeros((), dtype=torch.int64, device=like.device))
    if model.config.memory == 'landmarks':
        capacity = max(1, math.ceil(max_head_rows / model.config.stride))
        state.update(landmarks=like.new_zeros(capacity, 2, model.config.hidden),
                     landmark_valid=torch.zeros(capacity, dtype=torch.bool, device=like.device))
    state.update({f'buffer_{i}': like.new_zeros(2 * d, 2, model.config.hidden)
                  for i, d in enumerate(model.temporal.config.dilations)})
    return state


def graph_modules(model):
    if model.config.conditioner != 'film':
        raise ValueError('R2 ONNX export supports the released FiLM conditioner')
    return dict(r2_predict=R2Predict(model).eval(), r2_commit=R2Commit(model).eval(),
                r2_release=R2Release(model).eval())


class R2Stepper:
    """Run named tensor graphs through a caller's recorder or runtime callback.

    ``graph_call(name, feeds, fn)`` returns an output dictionary. ``fn()`` runs
    the Torch graph on those feeds. A runtime may return its own tensors using
    the same output names. Call commit after each completed head row, never
    after EOS. The query and release methods do not mutate the carried state.
    """

    def __init__(self, model, graph_call, max_head_rows):
        self.model = model
        self.graph_call = graph_call
        self.modules = graph_modules(model)
        self.state = initial_state(model, max_head_rows)

    def _call(self, name, feeds):
        module = self.modules[name]

        def run():
            values = module(*(feeds[key] for key in module.input_names))
            if len(module.output_names) == 1:
                values = (values,)
            return dict(zip(module.output_names, values))

        return run() if self.graph_call is None else self.graph_call(name, feeds, run)

    @torch.no_grad()
    def predict(self, chart, k, track=(), **query_kwargs):
        feeds = dict(last=self.state['last'], query=self.model.query_features(chart, [k], **query_kwargs),
                     row_condition=self.model.row_condition(chart, track, [k]))
        if self.model.config.memory == 'landmarks':
            feeds.update(landmarks=self.state['landmarks'], landmark_valid=self.state['landmark_valid'])
        output = self._call('r2_predict', feeds)
        return output['hands'], output['action_logits']

    @torch.no_grad()
    def release(self, hands, factor, chart, track=()):
        model = self.model
        condition = model.pair_condition([factor], chart, track)
        if condition is None:
            condition = hands.new_zeros(len(factor.times), model.film.inputs)
        feeds = dict(hands=hands,
                     orientation=torch.tensor([factor.orientation], dtype=torch.int64, device=model.device),
                     codes=torch.as_tensor(factor.codes[None], dtype=torch.int64, device=model.device),
                     slot=torch.tensor([factor.slot], dtype=torch.int64, device=model.device),
                     lane_features=model._t(factor.lane_feats[None]), relations=model._t(factor.relations),
                     candidates=model._t(factor.cand_feats), pair_condition=condition)
        return self._call('r2_release', feeds)['release_scores']

    @torch.no_grad()
    def commit(self, chart, k):
        module = self.modules['r2_commit']
        feeds = {key: self.state[key] for key in module.input_names[1:]}
        feeds['history_token'] = self.model._t(_history_token(chart, k))
        output = self._call('r2_commit', feeds)
        self.state = {name: output['next_' + name] for name in module.state_names}


def _examples(model, max_head_rows):
    state = initial_state(model, max_head_rows)
    like = state['last']
    query_dim = QUERY_DIM + sum(width for name, width in READERS if getattr(model.config, name) == 'on')
    predict = dict(last=state['last'], query=like.new_zeros(1, 2, query_dim),
                   row_condition=like.new_zeros(1, model.film.inputs))
    if model.config.memory == 'landmarks':
        predict.update(landmarks=state['landmarks'], landmark_valid=state['landmark_valid'])
    commit = dict(history_token=like.new_zeros(2, HISTORY_DIM), **{k: v for k, v in state.items() if k != 'last'})
    release = dict(hands=like.new_zeros(1, 2, model.config.hidden),
                   orientation=torch.zeros(1, dtype=torch.int64, device=like.device),
                   codes=torch.zeros(1, 4, dtype=torch.int64, device=like.device),
                   slot=torch.zeros(1, dtype=torch.int64, device=like.device),
                   lane_features=like.new_zeros(1, LANE_QUERY_DIM), relations=like.new_zeros(3, RELATION_DIM),
                   candidates=like.new_zeros(3, CANDIDATE_DIM), pair_condition=like.new_zeros(3, model.film.inputs))
    return dict(r2_predict=predict, r2_commit=commit, r2_release=release)


@torch.no_grad()
def export_graphs(model, out, max_head_rows=32768, opset_version=17):
    """Write float32 ONNX files and return named input/output schemas.

    Landmark capacity and release candidate count are dynamic axes. Every
    other dimension is fixed by the checkpoint. The caller writes schemas and
    initial_state tensors in its artifact format; this function writes ONNX only.
    The supplied model is moved to CPU float32 evaluation mode in place.
    """
    import onnx

    if opset_version < 17:
        raise ValueError('R2 graphs require ONNX opset 17 or newer')
    model = model.cpu().float().eval()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    modules = graph_modules(model)
    examples = _examples(model, max_head_rows)
    schemas = {}
    for name, module in modules.items():
        feeds = examples[name]
        args = tuple(feeds[key] for key in module.input_names)
        values = module(*args)
        if len(module.output_names) == 1:
            values = (values,)
        dynamic = {}
        if name == 'r2_release':
            dynamic.update({key: {0: 'candidates'} for key in
                            ('relations', 'candidates', 'pair_condition', 'release_scores')})
        for key in module.input_names + module.output_names:
            if 'landmark' in key:
                dynamic[key] = {0: 'landmarks'}
        path = out / (name + '.onnx')
        torch.onnx.export(module, args, str(path), input_names=module.input_names,
                          output_names=module.output_names, dynamic_axes=dynamic,
                          opset_version=opset_version, dynamo=False, external_data=False)
        onnx.checker.check_model(str(path))

        def spec(key, tensor):
            shape = list(tensor.shape)
            for axis, symbol in dynamic.get(key, {}).items():
                shape[axis] = symbol
            return dict(name=key, dtype=str(tensor.dtype).removeprefix('torch.'), shape=shape)

        schemas[name] = dict(path=str(path), opset=opset_version,
                             inputs=[spec(key, feeds[key]) for key in module.input_names],
                             outputs=[spec(key, value) for key, value in zip(module.output_names, values)])
        if name == 'r2_commit':
            schemas[name]['state_outputs'] = dict(zip(module.output_names, module.state_names))
    return schemas
