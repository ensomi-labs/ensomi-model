from dataclasses import replace

import numpy as np
import pytest
import torch

from ensomi_model.research.audio_memory_continuation.memory import gather_memory, memory_indices, HistoryAttention
from ensomi_model.research.audio_memory_continuation.model import initialize, load_model, MemoryConfig
from ensomi_model.research.audio_memory_continuation.generation import MemorySession
from ensomi_model.research.audio_memory_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import (
    collate_interval as collate_old,score_interval as score_old,interval_losses,
)
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule,ControlSpan
from ensomi_model.research.oracle_time_continuation.features import TIME_DIM
from controlled_audio_continuation.test_ownership import model
from controlled_audio_continuation.test_sampling import source
from planned_audio_continuation.test_distribution import chart


def network():
    return initialize(model(),memory_config=MemoryConfig(span_ms=2000,cell_ms=50,width=32,heads=4,
                                                         audio_width=32,audio_layers=1))


def activate(net):
    for module in (net.audio_pyramid,net.head_memory,net.release_memory,net.row_memory):
        torch.nn.init.normal_(module.output.weight,std=.03)


def schedule(net,duration=1000):
    return ControlSchedule((ControlSpan(0,duration+1,stars=4,ln_fraction=.5),),net.style_names)


def test_clock_selection_respects_known_prefix_inside_one_hazard_bin_and_expires_by_time():
    times = np.array([3,7,250,500,1000])
    a = memory_indices(times,[0,1,2],[9,9,509],span_ms=1000,cell_ms=500)
    assert a[0].tolist()==[-1,-1,0]
    assert a[1].tolist()==[-1,-1,1]
    assert a[2].tolist()==[-1,2,-1]
    assert (memory_indices(times,[4],[2001],span_ms=1000,cell_ms=500)==-1).all()


def test_attention_is_hand_equivariant_audio_query_sensitive_and_empty_safe():
    torch.manual_seed(41)
    layer = HistoryAttention(12,8,6,width=16,heads=4)
    torch.nn.init.normal_(layer.output.weight,std=.1)
    history,audio = torch.randn(4,2,8),torch.randn(4,6)
    query = torch.randn(2,2,12)
    memory = gather_memory(history,audio,[0,200,400,600],[3,-1],[650,700],span_ms=1000,cell_ms=100)
    value = layer(query,memory)
    assert torch.isfinite(value).all() and not value[1].any()
    mirrored = replace(memory,history=history.flip(1))
    torch.testing.assert_close(layer(query.flip(1),mirrored),value.flip(1),atol=1e-6,rtol=1e-6)
    changed = layer(query+torch.randn_like(query),memory)
    assert not torch.allclose(value[0],changed[0])


def test_zero_initialized_memory_preserves_full_audio_and_all_three_source_factors():
    torch.set_num_threads(1);torch.manual_seed(62)
    old = model().eval()
    net = initialize(old,memory_config=MemoryConfig(width=32,audio_width=32,audio_layers=1)).eval()
    c = source();example=IntervalExample(c,0,1001);controls=schedule(net)
    mel = torch.from_numpy(c.mel)[None]
    old_audio,audio = old.encode_audio(mel),net.encode_audio(mel)
    torch.testing.assert_close(audio,old_audio,atol=0,rtol=0)
    a=score_old(old,collate_old(example,old.config,recovery=old.recovery).inputs,None,
                controls=controls,encoded_full=old_audio)
    b=score_interval(net,collate_interval(example,net),controls,audio)
    for name in ('head','release','row'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),atol=0,rtol=0)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_joint_memory_audio_gradients_and_checkpoint_roundtrip(device,tmp_path):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    torch.set_num_threads(1);torch.manual_seed(51)
    net=network().to(device);activate(net)
    c=source();batch=collate_interval(IntervalExample(c,0,1001),net,device)
    encoded=net.encode_audio(torch.from_numpy(c.mel)[None].to(device))
    result=score_interval(net,batch,schedule(net),encoded)
    loss=interval_losses(result,batch.planned)[-1]
    loss.backward()
    for layer in (net.audio_input,net.audio_pyramid.project,net.audio_pyramid.output,
                  net.head_memory.key,net.release_memory.key,net.row_memory.key):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0
    path=tmp_path/'memory.pt';torch.save(net.checkpoint(),path)
    restored=load_model(path,device=device)
    assert restored.memory_config==net.memory_config
    for key,value in net.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],atol=0,rtol=0)


def test_native_row_probabilities_match_partitioned_current_weight_source_replay():
    torch.set_num_threads(1);torch.manual_seed(72)
    net=network().eval();activate(net)
    controls=schedule(net);mel=np.random.default_rng(9).normal(size=(100,128)).astype(np.float32)
    recorded=[]
    class Recorder(MemorySession):
        def prefer_rows(self,*args):
            q=super().prefer_rows(*args);recorded.append(q.clone());return q
    with torch.inference_mode():
        session=Recorder(net,mel,1000,controls,seed=61,
            head_times=(0,100,200,300,450,550,650,750,900))
        for end in (133,399,712,1000):session.publish_to(end)
    c=replace(chart([r.time_ms for r in session.rows],[r.actions for r in session.rows],1000),mel=mel)
    encoded=net.encode_audio(torch.from_numpy(mel)[None]);actual=[]
    for i in range(IntervalExample(c,0,137).count):
        example=IntervalExample(c,i,137);batch=collate_interval(example,net)
        raw=score_interval(net,batch,controls,encoded)
        actual.append(replay_row_scores(raw.row[:len(batch.planned.row_index)],example,controls))
    torch.testing.assert_close(torch.cat(actual),torch.stack(recorded),atol=3e-5,rtol=3e-6)


def test_native_zero_initialization_and_fork_control_rollback_preserve_owned_memories():
    torch.set_num_threads(1);torch.manual_seed(82)
    old=model().eval();net=initialize(old,memory_config=MemoryConfig(width=32,audio_width=32,audio_layers=1)).eval()
    mel=np.random.default_rng(15).normal(size=(250,128)).astype(np.float32)
    controls=schedule(net,2500)
    with torch.inference_mode():
        base=ControlledSession(old,mel,2500,controls,seed=74)
        current=MemorySession(net,mel,2500,controls,seed=74)
        base.publish_to(650);current.publish_to(650)
        assert base.rows==current.rows
        fork=current.fork()
        kept=tuple(current.rows)
        change=ControlSpan(900,1700,stars=3,ln_fraction=.1)
        base.update_controls(change);current.update_controls(change)
        assert len(current.planner.event_memory.times)==len(current.planner.generated)
        base.publish_to(2500);current.publish_to(2500)
        assert base.rows==current.rows and tuple(current.rows[:len(kept)])==kept
        assert tuple(fork.rows)==kept
        assert len(fork.row_memory.times)==len(kept)
        fork.publish_to(2500)
        assert not any(fork.replay.occupancy)


def test_nonzero_memory_keeps_tap_layout_and_count_out_of_skeleton_factors():
    torch.set_num_threads(1);torch.manual_seed(93)
    net=network().eval();activate(net)
    a=chart([0,100,250,400],[(1,0,0,0),(0,2,0,0),(0,0,0,1),(0,3,0,0)],500)
    b=chart([0,100,250,400],[(0,1,1,1),(0,2,0,0),(1,0,1,0),(0,3,0,0)],500)
    controls=schedule(net,500)
    encoded=net.encode_audio(torch.from_numpy(a.mel)[None])
    scores=[score_interval(net,collate_interval(IntervalExample(c,0,501),net),controls,encoded) for c in (a,b)]
    torch.testing.assert_close(scores[0].head,scores[1].head,atol=0,rtol=0)
    torch.testing.assert_close(scores[0].release,scores[1].release,atol=0,rtol=0)
    assert not torch.equal(scores[0].row,scores[1].row)


def test_nonzero_native_head_and_release_queries_match_source_prefix_scoring():
    torch.set_num_threads(1);torch.manual_seed(94)
    net=network().eval();activate(net)
    mel=np.random.default_rng(17).normal(size=(200,128)).astype(np.float32)
    controls=schedule(net,2000);mode=['native'];ledger={'head':{},'release':{}};matched={'head':0,'release':0}
    def capture(kind,original):
        def call(audio,history,clocks,**options):
            value=original(audio,history,clocks,**options)
            c=clocks.detach().cpu().double()
            if kind=='head':
                now=torch.round(1000*c[:,TIME_DIM+1].sinh()).long()
                previous=now-torch.round(1000*c[:,1].sinh()).long()
                known=c[:,TIME_DIM-1]>0
            else:
                now=2000-torch.round(1000*c[:,0,7*TIME_DIM+1].sinh()).long()
                previous=now-torch.round(1000*c[:,0,4*TIME_DIM+1].sinh()).long()
                known=c[:,0,5*TIME_DIM-1]>0
            for i,(t,p,k) in enumerate(zip(now.tolist(),previous.tolist(),known.tolist())):
                key=(t,p if k else -1)
                if mode[0]=='native':ledger[kind][key]=value[i].detach().cpu()
                elif key in ledger[kind]:
                    torch.testing.assert_close(value[i].detach().cpu(),ledger[kind][key],atol=3e-5,rtol=3e-6)
                    matched[kind]+=1
            return value
        return call
    net.head_logits=capture('head',net.head_logits)
    net.release_logits=capture('release',net.release_logits)
    with torch.inference_mode():
        session=MemorySession(net,mel,2000,controls,seed=43)
        for end in (633,1250,2000):session.publish_to(end)
    c=replace(chart([r.time_ms for r in session.rows],[r.actions for r in session.rows],2000),mel=mel)
    encoded=net.encode_audio(torch.from_numpy(mel)[None]);mode[0]='teacher'
    for i in range(IntervalExample(c,0,377).count):
        batch=collate_interval(IntervalExample(c,i,377),net)
        score_interval(net,batch,controls,encoded)
    assert matched['head']>100 and matched['release']>10


@pytest.mark.parametrize('device',['cpu','mps'])
def test_nonzero_audio_pyramid_padding_cannot_change_real_song_features(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    torch.set_num_threads(1);torch.manual_seed(95)
    net=network().to(device).eval();activate(net)
    mel=torch.randn(1,137,128,device=device)
    padded=torch.nn.functional.pad(mel,(0,0,0,263))
    valid=torch.arange(400,device=device)[None]<137
    a=net.encode_audio(mel)
    b=net.encode_audio(padded,valid)[:,:137]
    torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
