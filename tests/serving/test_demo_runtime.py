from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch
from ensomi_model.serving.config import DemoServiceConfig
from ensomi_model.serving.runtime import DemoRuntime, ServiceError, request_controls
from ensomi_model.serving.http import handler_for
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel
from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig


@pytest.fixture
def runtime(tmp_path):
    torch.manual_seed(51)
    from dataclasses import asdict
    model = ControlledAudioModel(PlannedModelConfig(hidden=16, audio_width=8, audio_levels=2,
        history_levels=4, expansion=2, coupling_rank=4, routing_hidden=16, release_hidden=16,
        context_width=16, context_layers=1, context_heads=2, head_hidden=8, head_levels=2,
        skeleton_hidden=8, skeleton_levels=2, lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60), style_names=('tech',))
    path = tmp_path/'model.pt'
    torch.save(dict(format='controlled-audio/v1', model_config=asdict(model.config),
        probability_options=model.probability_options(), model=model.state_dict()), path)
    config = DemoServiceConfig(checkpoint_file=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cache_dir=str(tmp_path/'cache'))
    rt = DemoRuntime(config)
    import soundfile
    audio = tmp_path/'test.wav'
    soundfile.write(audio, np.sin(np.arange(48000)*.025).astype('float32'), 24000)
    return rt, audio


def test_complete_windows_retry_cache_and_controls(runtime):
    rt, audio = runtime
    receipt = rt.create(dict(session_id='first', audio_path=str(audio), controls={'difficulty': 3, 'style': {'tech': 1}}))
    assert not receipt['cache_hit']
    first = rt.window('first', dict(after_sequence=0, through_ms=900))
    assert first == rt.window('first', dict(after_sequence=0, through_ms=900))
    assert first['coverage_ms'] == 900 and not first['end_of_stream']
    second = rt.window('first', dict(after_sequence=1, through_ms=2000))
    assert second['coverage_ms'] == 1900
    assert all(r['time_ms'] > 900 for r in second['rows'])
    last = rt.window('first', dict(after_sequence=2, through_ms=2000))
    assert last['end_of_stream'] and last['coverage_ms'] == 2000
    assert not any(rt.sessions['first'].generator.replay.occupancy)
    rt.cancel('first')
    with pytest.raises(ServiceError):rt.window('first', dict(after_sequence=0, through_ms=900))
    again=rt.create(dict(session_id='second', audio_path=str(audio), controls={'difficulty':3,'style':{'tech':1}}))
    assert again['cache_hit']
    assert first['rows']==rt.window('second',dict(after_sequence=0,through_ms=900))['rows']


def test_encoded_audio_matches_original_constructor(runtime):
    rt, audio = runtime
    with torch.inference_mode():
        feature,_=rt._features(audio)
        controls=request_controls({'difficulty':3},rt.model.style_names,feature.duration_ms)
        base=ControlledSession(rt.model,feature.mel,feature.duration_ms,controls,seed=27)
        cached=ControlledSession(rt.model,feature.mel,feature.duration_ms,controls,seed=27,encoded_audio=feature.encoded)
        base.publish_to(feature.duration_ms);cached.publish_to(feature.duration_ms)
        assert base.rows==cached.rows
        with pytest.raises(ValueError):
            ControlledSession(rt.model,feature.mel,feature.duration_ms,controls,encoded_audio=feature.encoded[:,:1])


def test_unknown_controls_and_stale_sequences_fail(runtime):
    rt,audio=runtime
    for controls in ({'style':{'unknown':1}}, {'difficulty':float('nan')}, {'unused':1}):
        with pytest.raises(ServiceError):rt.create(dict(session_id='invalid',audio_path=str(audio),controls=controls))
    rt.create(dict(session_id='valid',audio_path=str(audio)))
    with pytest.raises(ServiceError):rt.window('valid',dict(after_sequence=3,through_ms=1000))
    rt.cancel('valid')
    assert not rt.sessions


def test_http_health_and_actual_generated_window(runtime):
    rt,audio=runtime
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(rt))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def request(path,body=None):
        req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),
            headers={} if body is None else {'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req))
    try:
        assert request('/health')['ready']
        request('/sessions',dict(session_id='wire',audio_path=str(audio)))
        result=request('/sessions/wire/window',dict(after_sequence=0,through_ms=2000))
        assert result['end_of_stream'] and result['sequence']==1
    finally:
        server.shutdown();server.server_close();thread.join()
