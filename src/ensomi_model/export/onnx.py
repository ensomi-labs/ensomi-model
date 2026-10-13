"""Export the released float32 networks and their Python decode constants.

    python -m ensomi_model.export.onnx --model-dir DIR --out DIR/onnx
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path

import numpy as np
import torch


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        if any(not isinstance(k, (str, int)) for k in value):
            return [dict(key=json_value(k), value=json_value(v)) for k, v in value.items()]
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value).__name__)


def module_constants(name):
    module = importlib.import_module('ensomi_model.' + name)
    values = {}
    for key, value in vars(module).items():
        if key.isupper() and not key.startswith('_'):
            try:
                values[key] = json_value(value)
            except TypeError:
                pass
    return values


def graph_schema(path, known_shapes=None):
    import onnx
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    def spec(value):
        tensor = value.type.tensor_type
        shape = (known_shapes or {}).get(value.name, [d.dim_param or d.dim_value for d in tensor.shape.dim])
        return dict(name=value.name, dtype=str(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)),
                    shape=shape, dynamic_axes={str(i): d for i, d in enumerate(shape) if isinstance(d, str)})
    return dict(file=path.name, bytes=path.stat().st_size,
                opset={x.domain or 'ai.onnx': x.version for x in graph.opset_import},
                inputs=[spec(v) for v in graph.graph.input], outputs=[spec(v) for v in graph.graph.output])


def manifest(system, out, files):
    """The runtime config for the graphs in ``out``. Writes the arrays and notice it names into ``out`` and
    appends their paths to ``files``."""
    from ..audio_rows import decoder
    from ..audio_rows.figures import NPOS
    from ..audio_rows.fitter.config import GridFitterConfig
    from ..features.mel_base import MUSIC_MEL_CACHE_CONFIG
    from .beatthis_graph import beatthis_frontend_constants, write_frontend_arrays
    from .r2_graph import initial_state
    from .recording import array
    import pandas as pd
    out = Path(out)
    modules = ('audio_rows.audio', 'audio_rows.rowsgen', 'audio_rows.figures', 'audio_rows.beatdata',
               'audio_rows.data', 'audio_rows.lattice', 'audio_rows.grid', 'audio_rows.decoder', 'audio_rows.fitter.config',
               'audio_rows.fitter.peaks', 'audio_rows.fitter.mel_phase', 'r2.common', 'r2.candidates',
               'r2.features', 'r2.strain', 'r2.state', 'r2.properties', 'system.make', 'system.arrange',
               'system.controls', 'system.prior', 'evaluation.beats')
    constants = {name: module_constants(name) for name in modules}
    table = pd.read_parquet(system.cal['scope_table'])
    scope_table = dict(rows=len(table), columns=[])
    for name in table.columns:
        values = array(table[name].to_numpy())
        filename = f'scopes-{name}.bin'
        (out / filename).write_bytes(values.tobytes())
        files.append(filename)
        scope_table['columns'].append(dict(name=name, file=filename, dtype=values.dtype.str,
                                          shape=list(values.shape), nbytes=values.nbytes))
    state = {}
    for key, value in initial_state(system.r2, 1).items():
        a = array(value)
        path = out / f'r2_initial_{key}.bin'
        path.write_bytes(a.tobytes())
        files.append(path.name)
        state[key] = dict(file=path.name, dtype=a.dtype.str, shape=list(value.shape))
    arrays = write_frontend_arrays(system.rows.listener.frames, out)
    files += [arrays[side][name]['path'] for side in ('beatthis', 'rows') for name in ('mel_filter', 'hann_window')]
    files.append(arrays['beatthis_license'])
    record = system.rows.record
    cfg = system.rows.model.cfg
    shapes = dict(beatthis={name: [1, 'time'] for name in ('beat_logits', 'downbeat_logits')},
                  rows_encoder=dict(pos=[1, 'beats', 24, cfg.enc], context=[1, 'beats', cfg.ctx],
                                    query=[1, 'beats', cfg.heads, cfg.key], key=[1, 'beats', cfg.heads, cfg.key]))
    result = dict(
        format='ensomi-onnx-runtime-v1', precision='float32',
        graphs={p.stem: graph_schema(p, shapes.get(p.stem)) for p in sorted(out.glob('*.onnx'))},
        state=dict(r2_initial=state, r2_commit_outputs={f'next_{k}': k for k in state},
                   rows_hidden_shape=[1, 1, system.rows.model.cfg.hidden], rows_initial_previous=system.rows.model.cfg.vocab,
                   rows_copy_history='newest lag first; unavailable figure IDs -1; recurrent hidden initially zero'),
        beatthis_frontend=beatthis_frontend_constants(system.rows.listener.frames),
        frontend_arrays=arrays,
        grid_fitter=json_value(GridFitterConfig()),
        mp3=dict(lead_samples=decoder.LEAD_SAMPLES, ffmpeg_encoders=decoder._FFMPEG_ENCODERS,
                 bitrates_v1=decoder._BITRATES_V1, bitrates_v2=decoder._BITRATES_V2,
                 sample_rates=decoder._SAMPLE_RATES, header_scan_bytes=65536,
                 frame_coefficients=dict(mpeg1=144, mpeg2=72),
                 side_info_bytes=dict(mpeg1_mono=17, mpeg1_stereo=32, mpeg2_mono=9, mpeg2_stereo=17),
                 mpg123_early="tag == 'xing' or (tag == 'lame' and delay == 0)",
                 ffmpeg_early="tag == 'lame' and delay == 0 and encoder starts with LAME, Lavf or Lavc",
                 prepend_samples='round(lead_samples / source_sample_rate * target_sample_rate)',
                 non_mp3_lead_samples=0),
        rows_frontend=dict(config=json_value(MUSIC_MEL_CACHE_CONFIG), window='hann_periodic', center=False,
                           power=2., mel_norm=1, mel_scale='slaney', log='natural', floor=1e-5,
                           tail_padding='zeros to ceil(samples / hop) frames',
                           standardization='whole-song scalar mean/std over log-Mel, denominator std + 1e-6',
                           storage_dtype='float16', graph_input_dtype='float32',
                           audio_decode='pydub/ffmpeg, mono, resampled to 24000 Hz, peak-normalized',
                           mp3_lead='audio_rows.decoder.lead_seconds separately for ffmpeg and mpg123'),
        rows=dict(config=json_value(system.rows.model.cfg),
                  record={k: record[k] for k in ('vocab', 'label_mean', 'label_std', 'star_map', 'bands')},
                  members=system.rows.model.members.tolist(), figure_bitmask_width=NPOS, position_beat_denominator=48),
        r2=dict(config=json_value(system.r2.config)), system=system.config,
        calibration=system.cal, prior=system.prior.record, constants=constants,
        sampling=dict(rows='single-sample multinomial; replay exponential variates, choose argmax(probability / draw)',
                      r2='float64 Gumbel maximum: argmax(log_probability - log(-log(max(uniform, tiny))))',
                      orientation='fair integer in {0,1}; consumed even if no releases',
                      difficulty_pass='each pass restarts the R2 generator at R2_SEED + seed',
                      figure_history='committed figure IDs, not sampled positions',
                      graph_outputs='raw networks; all masks, probability mixtures, controls and sampling stay outside graphs'),
        decode_literals=dict(
            rows=dict(copy_equality_history_offsets=[1, 2], section_minimum_history_beats=4,
                      head_rounding_decimal_places=3, intensity_rank_ties='stable'),
            timing=dict(peak_radius_frames=3, peak_logit_threshold=0., mad_scale=1.4826,
                        peak_residual_floor_ms=5., onset_curvature_epsilon=1e-12, onset_parabola_clip=.5,
                        segment_edge_epsilon=1e-9, drift_floor_ms=3., drift_robust_multiplier=3.,
                        beat_probability_threshold=.5),
            feature_transform=dict(linear_clip=32., beat_scale=1., bpm_scale=120., meter_scale=4.,
                                   ln_value_scale=2., ln_value_offset=-1., star_scale=4.,
                                   residual_star_scale=.5, residual_star_clip=3.,
                                   anchor_repeat_lags=[1, 2, 4], hold_length_floor_beats=1e-4, entropy_floor=1e-30),
            candidates=dict(rounding='floor(x + 0.5)', grid_search_margin_ms=1., midpoint_fraction=.5,
                            previous_beat_anchors=[.125, .25], label_rounding='ties_to_even'),
            controls=dict(force_window_rows=64, tilt_clip=8., initial_concept_gain=0.,
                          logit_probability_clip=[.01, .99], scope_bins_seconds=[8., 16., 32.],
                          scope_neighbours=256, scope_duration_seconds=[8., 32.],
                          exact_return_exempt_previous_arity=3, quantile_method='linear',
                          nearest_neighbour_ties='stable'),
            prior=dict(gap_floor_ms=1., workload_floor=1e-6, share_bounds=[0., 1.], head_bounds=[1., 4.]),
            difficulty=dict(max_passes=2, retry_rule='absolute r error greater than band fit seed_sd',
                            correction='clip(m * exp((log(target_r)-log(achieved_r))/beta), lo, hi)'),
            export=dict(head_rounding='ties_to_even', release_rounding='ties_to_even',
                        release_lower_offset_ms=1., release_upper_offset_ms=-1.,
                        final_release_upper='ceil(song_ms) - 1', timing_offset_rounding='ties_to_even'),
            random=dict(gumbel_dtype='float64', uniform_floor=float(np.finfo(np.float64).tiny),
                        fallback_skips_action_draw=True, release_order='orientation chooses ascending or descending lanes')))
    # Model-relative paths are the portable runtime contract.
    result['calibration'] = dict(result['calibration'], scope_table=scope_table)
    result['rows_frontend']['config'].pop('cache_root')
    return json_value(result)


def export(model_dir, out, opset=17):
    from ..system.make import System
    from .beatthis_graph import export_beatthis_graph
    from .package import CONFIG, write_manifest
    from .r2_graph import export_graphs
    from .rows_graph import export_rows_graphs
    model_dir, out = Path(model_dir), Path(out)
    system = System.load(json.loads((model_dir / 'system.json').read_text()), model_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = [export_beatthis_graph(system.rows.listener.frames.model, out / 'beatthis.onnx', opset),
              *export_rows_graphs(system.rows.model, out, opset).values(),
              *(Path(g['path']) for g in export_graphs(system.r2, out, opset_version=opset).values())]
    files = [CONFIG, *(p.name for p in graphs)]
    result = manifest(system, out, files)
    (out / CONFIG).write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    write_manifest(out, files, notices=[result['frontend_arrays']['beatthis_license']])
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--opset', type=int, default=17)
    a = p.parse_args(argv)
    if not (a.model_dir / 'system.json').is_file():
        p.error('--model-dir must contain system.json')
    if a.opset < 17:
        p.error('--opset must be at least 17')
    torch.set_num_threads(1)
    result = export(a.model_dir, a.out, a.opset)
    print(json.dumps({k: dict(bytes=v['bytes'], opset=v['opset']) for k, v in result['graphs'].items()}))


if __name__ == '__main__':
    main()
