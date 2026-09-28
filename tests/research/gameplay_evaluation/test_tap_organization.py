from ensomi_model.research.gameplay_evaluation.tap_organization import tap_organization
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace,Scope
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def report(rows,start=0,end=1000):
    return tap_organization(ChartTrace(tuple(CompleteRow(t,a) for t,a in rows),1001),Scope('one',start,end))


def test_one_held_role_retains_complete_tap_groups_and_real_gaps():
    value=report([(0,(2,0,0,0)),(100,(0,1,1,0)),(250,(0,0,0,1)),
                  (400,(0,1,1,0)),(600,(3,0,0,1))])
    assert value['single_held_TAP_H']==3 and value['longest_single_held_TAP_H_run']==3
    witness=value['single_held_TAP_witnesses'][0]
    assert witness['TAP_heads']==5 and witness['single_double_alternations']==2
    assert witness['held_origin']==dict(column=0,start_ms=0.)
    assert witness['maximum_H_gap_ms']==150 and witness['span_ms']==300


def test_LN_only_H_cannot_be_dropped_to_join_separate_tap_bodies():
    value=report([(0,(2,0,0,0)),(100,(0,1,0,0)),(200,(0,0,2,0)),
                  (250,(0,0,3,0)),(300,(0,1,0,0)),(500,(3,0,0,0))])
    assert value['single_held_TAP_H']==2 and value['longest_single_held_TAP_H_run']==1
    pairs={(tuple(p['previous']),tuple(p['current'])):p['count'] for p in value['TAP_mask_pairs']}
    assert ((1,),()) in pairs and ((),(1,)) in pairs
    assert ((1,),(1,)) not in pairs


def test_incoming_open_hold_uses_no_future_tail_and_scope_runs_do_not_pool():
    rows=[(0,(2,0,0,0)),(100,(0,1,0,0)),(200,(0,0,1,0)),(300,(0,0,0,1))]
    a=report(rows,start=150,end=350)
    b=report(rows+[(400,(3,0,0,0))],start=150,end=350)
    assert a==b
    assert a['longest_single_held_TAP_H_run']==2
    assert a['single_held_TAP_witnesses'][0]['held_origin']['start_ms']==0
    assert a['single_held_TAP_witnesses'][0]['moving_single_pairs']==1


def test_two_continuing_holds_are_not_a_single_held_role():
    value=report([(0,(2,2,0,0)),(100,(0,0,1,1)),(200,(0,0,1,0)),
                  (250,(0,3,0,0)),(300,(0,0,1,1)),(500,(3,0,0,0))])
    assert value['heads_only_TAP_H']==3 and value['single_held_TAP_H']==1
