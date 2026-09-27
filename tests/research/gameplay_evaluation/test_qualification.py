from dataclasses import asdict,replace
from importlib.resources import files
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from ensomi_model.research.audio_memory_continuation.model import initialize,MemoryConfig
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel
from ensomi_model.research.gameplay_evaluation.qualification import run_qualification
from ensomi_model.research.gameplay_evaluation.qualification_config import QualificationConfig
from ensomi_model.research.gameplay_evaluation.qualification_hydra import compose_config
from ensomi_model.research.joint_audio_continuation.data import digest
from ensomi_model.research.typed_audio_continuation.program import Recovery
from planned_audio_continuation.test_distribution import config


def fixture(tmp_path,family='controlled'):
    torch.manual_seed(274810)
    net=ControlledAudioModel(replace(config(),lookahead=16,bounded_head=True,condition_full_holds=True,
        minimum_action_gap_ms=21),style_names=('tech',),recovery=Recovery(21,21,21))
    with torch.no_grad():
        net.head_base.weight.zero_();net.head_base.bias.fill_(-2.5)
    if family=='memory':net=initialize(net,memory_config=MemoryConfig(width=32,audio_width=32,audio_layers=1))
    checkpoint=tmp_path/'model.pt'
    payload=(net.checkpoint() if family=='memory' else dict(format='controlled-audio/v1',
        model_config=asdict(net.config),probability_options=net.probability_options(),model=net.state_dict()))
    torch.save(payload,checkpoint)
    audio=tmp_path/'fixture-audio';audio.write_bytes(b'explicit synthetic fixture with independent cached Mel')
    mel=tmp_path/'mel.npy';np.save(mel,np.zeros((100,128),np.float32))
    plan=dict(format='native-gameplay-qualification/v1',cases=[dict(key='synthetic',seed=41,
        asset=dict(audio_file=str(audio),audio_sha256=digest(audio),mel_file=str(mel),mel_sha256=digest(mel),duration_ms=1000),
        controls=[dict(start_ms=0,end_ms=1001,ln_fraction=.5)],
        scopes=[dict(name='before',start_ms=0,end_ms=400),dict(name='override',start_ms=400,end_ms=700),
                dict(name='restored',start_ms=700,end_ms=1001)],
        switch=dict(announce_after_ms=399,span=dict(start_ms=400,end_ms=700,ln_fraction=.2)),
        review_contexts=[dict(start_ms=300,end_ms=800,reason='Cross-scope held-state review')])])
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    settings=QualificationConfig(str(checkpoint),digest(checkpoint),str(path),digest(path),str(tmp_path/'run'),
        max_seconds=30,max_case_seconds=15,startup_seconds_limit=15,service_seconds_limit=15)
    return settings,plan


@pytest.mark.parametrize('family',['controlled','memory'])
def test_actual_native_export_and_scoped_evaluation_cannot_self_promote(tmp_path,family):
    settings,_=fixture(tmp_path,family)
    settings.ln_feedback=False
    result=run_qualification(settings)
    assert result['run_status']=='complete'
    assert result['candidate_status']=='review_required' and not result['promoted']
    assert result['semantic_review']=='pending'
    report=json.loads((tmp_path/'run/synthetic/evaluation.json').read_text())
    assert report['identity']['ln_feedback'] is False
    assert report['identity']['head_source']=='native'
    assert [s['scope']['name'] for s in report['scopes']]==['before','override','restored']
    assert report['publication']['trace_meets_deadlines']
    assert report['review_contexts'][0]['review_status']=='unreviewed'
    assert (tmp_path/'run/synthetic/generated.osu').exists()
    assert (tmp_path/'run/synthetic/publications.json').exists()


def test_numeric_failure_retains_complete_evidence_and_all_planned_cases(tmp_path):
    settings,plan=fixture(tmp_path)
    plan['cases'].append({**plan['cases'][0],'key':'second'})
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));settings.plan_sha256=digest(path)
    settings.startup_seconds_limit=1e-9
    result=run_qualification(settings)
    assert result['run_status']=='complete' and result['cases_completed']==2
    assert result['candidate_status']=='failed' and not result['promoted']
    assert result['failed_cases']==['synthetic','second']
    assert (tmp_path/'run/second/evaluation.json').exists()


def test_incomplete_generation_and_missing_envelope_cannot_pass(tmp_path):
    settings,plan=fixture(tmp_path)
    settings.max_case_seconds=1e-9
    result=run_qualification(settings)
    assert result['run_status']=='incomplete' and result['candidate_status']=='failed'
    assert result['cases_completed']==0
    assert json.loads((tmp_path/'run/cases.json').read_text())[0]['reason']
    settings.output_dir=str(tmp_path/'other')
    plan['cases'][0]['scopes'][0].update(stars=4,maximum_excess_seconds=.1)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));settings.plan_sha256=digest(path)
    with pytest.raises(ValueError,match='pressure bound'):run_qualification(settings)


def test_declared_pressure_bound_runs_and_keeps_witness_context(tmp_path):
    settings,plan=fixture(tmp_path)
    plan['cases'][0]['scopes'][0].update(stars=4,maximum_excess_seconds=0.)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));settings.plan_sha256=digest(path)
    envelope=tmp_path/'envelope.json'
    envelope.write_text(json.dumps(dict(envelope=dict(windows_ms=[250,1000],stars=[2,6],
        maximum_hz=[[1,1],[1,1]],reference='Test boundary, not corpus or player capacity'))))
    settings.envelope_file=str(envelope);settings.envelope_sha256=digest(envelope)
    result=run_qualification(settings)
    assert result['run_status']=='complete' and result['candidate_status']=='failed'
    record=json.loads((tmp_path/'run/cases.json').read_text())[0]
    assert any(c['name']=='before:pressure_excess' and not c['passed'] for c in record['checks'])
    assert record['scopes'][0]['pressure_witnesses']['excess_seconds']>0
    assert any('scored_scope' in context for context in record['review_contexts'])


def test_recovery_override_reaches_runtime_without_desynchronizing_H_capacity(tmp_path):
    settings,plan=fixture(tmp_path)
    plan['recovery']=dict(hh=21,rh=30,hr=10)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));settings.plan_sha256=digest(path)
    result=run_qualification(settings)
    assert result['run_status']=='complete' and result['identity']['recovery']==plan['recovery']
    plan['recovery']['hh']=28
    path.write_text(json.dumps(plan));settings.plan_sha256=digest(path);settings.output_dir=str(tmp_path/'other')
    with pytest.raises(ValueError,match='H-capacity'):run_qualification(settings)


def test_hydra_projection_packaging_and_lightweight_help(tmp_path):
    settings,_=fixture(tmp_path)
    overrides=[f'{k}={v}' for k,v in asdict(settings).items() if k in
               ('checkpoint_file','checkpoint_sha256','plan_file','plan_sha256','output_dir')]
    selected=compose_config([*overrides,'ln_feedback=false','lead_ms=1000','cpu_threads=2'])
    assert not selected.ln_feedback and selected.lead_ms==1000 and selected.cpu_threads==2
    with pytest.raises(ValueError,match='Unknown'):compose_config([*overrides,'+unused=true'])
    assert files('ensomi_model.configs.hydra').joinpath('native_gameplay_qualification.yaml').is_file()
    command=[sys.executable,'-m','ensomi_model.research.gameplay_evaluation.qualification_hydra']
    help_result=subprocess.run([*command,'--help'],capture_output=True,text=True)
    assert help_result.returncode==0 and 'ln_feedback' in help_result.stdout
    legacy=subprocess.run([*command,'--checkpoint-file','old.pt'],capture_output=True,text=True)
    assert legacy.returncode!=0
    actual=subprocess.run([*command,*overrides,'ln_feedback=false','startup_seconds_limit=1e-9'],
                          capture_output=True,text=True)
    assert actual.returncode==2,actual.stdout+actual.stderr
    assert json.loads((tmp_path/'run/result.json').read_text())['candidate_status']=='failed'
    assert json.loads((tmp_path/'run/cases.json').read_text())[0]['identity']['ln_feedback'] is False
