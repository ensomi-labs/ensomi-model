from ensomi_model.research.gameplay_evaluation.review import pressure_review_contexts
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace,Scope
from ensomi_model.research.gameplay_evaluation.witnesses import sustained_attack_witnesses
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.player_response.envelope import AttackEnvelope


def test_review_finds_long_scale_load_and_keeps_incoming_history_separate_from_scope():
    envelope=AttackEnvelope((4000,16000),(2.,6.),((6.,4.5),(6.,4.5)), 'fixture, not a quality threshold')
    rows=tuple(CompleteRow(t,(1,0,0,0)) for t in range(0,32000,200))
    trace=ChartTrace(rows,34000);scope=Scope('override',20000,32000)
    witnesses=sustained_attack_witnesses(trace,scope,envelope,4)
    assert witnesses['windows']['4000']['episode_count']==0
    review=pressure_review_contexts(witnesses,trace.coverage_ms,limit=1)
    assert len(review)==1 and review[0]['window_ms']==16000
    assert review[0]['scored_scope']==dict(name='override',start_ms=20000,end_ms=32000)
    assert review[0]['start_ms']<scope.start_ms
    assert review[0]['end_ms']==33000
    assert review[0]['excess_seconds']>0
    assert review[0]['review_status']=='unreviewed'
