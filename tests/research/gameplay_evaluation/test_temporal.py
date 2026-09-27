import numpy as np
import pytest

from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace,Scope
from ensomi_model.research.gameplay_evaluation.alignment import linear_cka,standardized,audio_correspondence
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def row(time,actions):return CompleteRow(time,tuple(actions))


def taps(times,lanes):
    return [row(int(t),[int(c==lane) for c in range(4)]) for t,lane in zip(times,lanes)]


def test_same_H_and_head_count_cannot_hide_one_finger_accumulation():
    times=np.arange(0,16000,100)
    concentrated=ChartTrace(taps(times,np.zeros(len(times),int)),16001)
    distributed=ChartTrace(taps(times,np.arange(len(times))%4),16001)
    scope=Scope('four_star_stream',4000,16000)
    a,b=[t.scope_report(scope) for t in (concentrated,distributed)]
    assert a['H']==b['H'] and a['heads']==b['heads']
    assert a['column_attack_peaks']['4000']['Hz']==10
    assert b['column_attack_peaks']['4000']['Hz']==2.5
    assert a['column_attack_peaks']['4000']['history_start_ms']==0


def test_identical_LN_amount_and_heads_cannot_hide_smeared_holds():
    def held(duration):
        rows=[]
        for t in range(0,32000,4000):rows.extend((row(t,[2,0,0,0]),row(t+duration,[3,0,0,0])))
        return ChartTrace(rows,32001)
    a,b=[t.scope_report(Scope('same_request',0,32000)) for t in (held(200),held(3900))]
    assert a['heads']==b['heads'] and a['LN_head_fraction']==b['LN_head_fraction']==1
    assert a['held_fraction_per_column'][0]==pytest.approx(.05)
    assert b['held_fraction_per_column'][0]==pytest.approx(.975)
    assert a['recovery']['500']['all_columns']>.8
    assert b['recovery']['500']['all_columns']==0


def test_scope_boundaries_do_not_reset_holding_or_recovery_clocks():
    trace=ChartTrace([row(0,[2,0,0,0]),row(1000,[3,0,0,0]),row(3000,[1,0,0,0])],4000)
    reports=[trace.scope_report(s,recovery_ms=(500,)) for s in
             (Scope('before',0,2000),Scope('after',2000,4000),Scope('whole',0,4000))]
    assert reports[0]['held_fraction_per_column'][0]==.5
    assert reports[1]['held_fraction_per_column'][0]==0
    assert reports[1]['recovery']['500']['all_columns']==.75
    assert sum(r['recovery']['500']['all_columns']*2000 for r in reports[:2])==pytest.approx(
        reports[2]['recovery']['500']['all_columns']*4000)
    assert reports[0]['heads']==reports[2]['heads']-reports[1]['heads']


def test_private_horizon_preserves_open_holds_without_inventing_release():
    trace=ChartTrace([row(100,[2,0,0,0])],5000)
    result=trace.scope_report(Scope('future',2000,4000))
    assert result['heads']==result['releases']==0
    assert result['held_fraction_per_column']==[1,0,0,0]
    assert result['recovery']['500']['all_columns']==0


def test_same_total_activity_can_have_different_phrase_contrasts():
    burst=np.arange(16000,32000,125);uniform=np.arange(0,32000,250)
    traces=[ChartTrace(taps(t,np.arange(len(t))%4),32001) for t in (burst,uniform)]
    scope=Scope('fixed_control',0,32000)
    assert traces[0].scope_report(scope)['heads']==traces[1].scope_report(scope)['heads']
    values=[t.contrasts(scope,half_windows_ms=(8000,),step_ms=8000)['8000'] for t in traces]
    assert max(abs(r[0]) for r in values[0]['differences'])==8
    assert max(abs(r[0]) for r in values[1]['differences'])==0


def test_short_attack_witness_retains_pre_scope_attack():
    trace=ChartTrace(taps([990,1005,1300],[0,0,1]),2000)
    witness=trace.scope_report(Scope('override',1000,2000))['short_attacks']
    assert witness==[dict(column=0,previous_ms=990.,attack_ms=1005.,gap_ms=15.)]


def test_alignment_distinguishes_correspondence_and_declares_constant_unmeasurable():
    rng=np.random.default_rng(5);x=rng.normal(size=(256,6))
    y=x@rng.normal(size=(6,4))
    score=linear_cka(standardized(x),standardized(y))
    shuffled=linear_cka(standardized(x),np.roll(standardized(y),71,axis=0))
    assert score>.5 and shuffled<.1
    assert linear_cka(x,np.ones((256,3))) is None
    assert linear_cka(x,x[:,::-1])==pytest.approx(1)


def test_audio_correspondence_runs_on_the_same_canonical_mel_and_chart_clock():
    rates=np.array([2,8,3,6,1,9,4,7,2,5,9,1,6,3,8,4])
    levels=np.repeat(rates,400)
    mel=np.repeat(levels[:,None],128,axis=1).astype(float)
    times=[]
    for block,rate in enumerate(rates):
        times.extend(block*4000+np.arange(4*rate)*(1000/rate)+50)
    trace=ChartTrace(taps(np.asarray(times).astype(int),np.arange(len(times))%4),64000)
    result=audio_correspondence(trace,mel,Scope('whole',0,64000),windows_ms=(1000,))['1000']
    assert result['status']=='measured'
    assert result['aligned_minus_shift_median']>.3
    assert len(result['shifted'])==5
