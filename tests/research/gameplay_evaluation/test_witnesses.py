import pytest

from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace, Scope
from ensomi_model.research.gameplay_evaluation.witnesses import sustained_attack_witnesses
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.player_response.envelope import AttackEnvelope, sustained_response
from ensomi_model.research.player_response.state import CommittedPlayState


ENVELOPE=AttackEnvelope((500,4000),(2.,6.),((8.,4.),(8.,4.)), 'fixture, not a quality threshold')


def taps(column_at):
    return tuple(CompleteRow(t,tuple(int(c==column_at(i)) for c in range(4)))
                 for i,t in enumerate(range(0,5000,100)))


def test_localizes_concentration_and_retains_free_peer_context():
    scope=Scope('stream',0,6000)
    concentrated=sustained_attack_witnesses(ChartTrace(taps(lambda i:1),6000),scope,ENVELOPE,4)
    distributed=sustained_attack_witnesses(ChartTrace(taps(lambda i:i%4),6000),scope,ENVELOPE,4)
    witness=concentrated['windows']['4000']['episodes'][0]
    assert witness['column']==1
    assert witness['duration_ms']>4000
    assert witness['peak_Hz']==10
    assert witness['attack_Hz_per_column'][0]==0
    assert witness['held_fraction_per_column']==[0.,0.,0.,0.]
    assert concentrated['excess_seconds']>0
    assert distributed['excess_seconds']==0


def test_partition_preserves_incoming_pressure_and_exact_existing_integral():
    rows=taps(lambda i:1);trace=ChartTrace(rows,9000)
    whole=sustained_attack_witnesses(trace,Scope('whole',0,9000),ENVELOPE,4)
    left=sustained_attack_witnesses(trace,Scope('before',0,2500),ENVELOPE,4)
    right=sustained_attack_witnesses(trace,Scope('after',2500,9000),ENVELOPE,4)
    _,reference=sustained_response(CommittedPlayState(),rows,9000,ENVELOPE,[(-1,0,None),(0,9000,4)])
    assert whole['excess_seconds']==pytest.approx(reference['excess_seconds'])
    assert left['excess_seconds']+right['excess_seconds']==pytest.approx(whole['excess_seconds'])
    witness=right['windows']['4000']['episodes'][0]
    assert witness['start_ms']==2500
    assert witness['starts_at_scope_boundary']
    assert witness['history_start_ms']==-1500


def test_real_relief_splits_episodes_and_summary_includes_omitted_witnesses():
    rows=tuple(CompleteRow(t,(1,0,0,0)) for t in (*range(0,1000,100),*range(7000,8000,100)))
    trace=ChartTrace(rows,12000);scope=Scope('two_bursts',0,12000)
    all_episodes=sustained_attack_witnesses(trace,scope,ENVELOPE,4)
    limited=sustained_attack_witnesses(trace,scope,ENVELOPE,4,limit_per_window=1)
    window=limited['windows']['500']
    assert window['episode_count']==2
    assert window['omitted_episodes']==1
    assert limited['excess_seconds']==all_episodes['excess_seconds']
    assert len(window['episodes'])==1


def test_chord_repetition_and_held_peers_remain_distinguishable():
    chord=tuple(CompleteRow(t,(1,1,1,1)) for t in range(0,5000,100))
    held=(CompleteRow(0,(1,2,2,2)),*(CompleteRow(t,(1,0,0,0)) for t in range(100,5000,100)))
    scope=Scope('context',0,6000)
    chord_w=sustained_attack_witnesses(ChartTrace(chord,6000),scope,ENVELOPE,4)['windows']['4000']['episodes'][0]
    held_w=sustained_attack_witnesses(ChartTrace(held,6000),scope,ENVELOPE,4)['windows']['4000']['episodes'][0]
    assert all(rate>0 for rate in chord_w['attack_Hz_per_column'])
    assert chord_w['held_fraction_per_column']==[0.,0.,0.,0.]
    assert held_w['held_fraction_per_column'][1:]==[1.,1.,1.]
    assert held_w['release_Hz']==0


def test_unknown_difficulty_does_not_invent_reference_capacity():
    result=sustained_attack_witnesses(ChartTrace((),1000),Scope('unknown',0,1000),ENVELOPE,None)
    assert result['status']=='unknown_difficulty'
    assert result['excess_seconds'] is None
