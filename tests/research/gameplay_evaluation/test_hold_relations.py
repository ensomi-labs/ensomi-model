import numpy as np
import pytest

from ensomi_model.research.gameplay_evaluation.hold_relations import hold_relations
from ensomi_model.research.gameplay_evaluation.report import evaluate_scopes
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace,Scope
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def stream(times):
    rows=[]
    for i,t in enumerate(times):
        actions=[0]*4
        if i:actions[(i-1)%4]=3
        if i<len(times)-1:actions[i%4]=2
        rows.append(CompleteRow(t,tuple(actions)))
    return ChartTrace(rows,int(times[-1])+1)


def test_H_jitter_changes_elapsed_LN_rhythm_without_changing_release_spans():
    traces=[stream(t) for t in ([0,100,200,300,400,500],[0,110,190,310,400,500])]
    reports=[hold_relations(t,Scope('same_duration_choices',0,400)) for t in traces]
    assert reports[0]['nearest_previous_group_log_change_ms']['q50']==0
    assert reports[1]['nearest_previous_group_log_change_ms']['q50']>.2
    for report in reports:
        assert report['H_span']==dict(q10=1.,q50=1.,q90=1.)
        assert report['nearest_previous_group_log_change_H']['q90']==0


def test_scope_keeps_prior_group_and_true_tail_but_censors_unobserved_end():
    trace=ChartTrace([CompleteRow(0,(2,0,0,0)),CompleteRow(100,(3,2,0,0)),
                     CompleteRow(1000,(0,3,0,0))],1001)
    result=evaluate_scopes(trace,[Scope('override',100,200)])['scopes'][0]['LN_timing_relations']
    assert result['duration_ms']['q50']==900
    assert result['latest_selected_tail_ms']==1000
    assert result['largest_ms_changes'][0]['previous_group_ms']==0
    assert result['nearest_previous_group_log_change_ms']['q50']==pytest.approx(np.log(9))
    assert result['missing_tail_H_coordinate']==1
    private=ChartTrace([CompleteRow(0,(2,0,0,0)),CompleteRow(100,(3,2,0,0))],200)
    report=hold_relations(private,Scope('private',100,200))
    assert report['LN_heads']==report['censored_LNs']==1
    assert report['completed_LNs']==0 and report['duration_ms'] is None
    assert report['latest_selected_tail_ms'] is None


def test_unequal_simultaneous_holds_are_not_sequential_variation_and_mirror_agrees():
    rows=[CompleteRow(0,(2,2,0,0)),CompleteRow(100,(3,0,0,0)),CompleteRow(300,(0,3,0,0))]
    scope=Scope('chord',0,301)
    reports=[hold_relations(ChartTrace(rs,301),scope) for rs in
             (rows,[CompleteRow(r.time_ms,tuple(reversed(r.actions))) for r in rows])]
    for r in reports:
        assert r['within_group_log_duration_spread']['q50']==pytest.approx(np.log(3))
        assert r['compared_lengths_ms']==0 and r['nearest_previous_group_log_change_ms'] is None
    assert reports[0]==reports[1]


def test_unresolved_previous_group_is_not_replaced_by_an_older_completed_group():
    trace=ChartTrace([CompleteRow(0,(2,0,0,0)),CompleteRow(50,(3,0,0,0)),
        CompleteRow(100,(0,2,0,0)),CompleteRow(200,(0,0,2,0)),
        CompleteRow(300,(0,0,3,0))],1000)
    report=hold_relations(trace,Scope('private',100,1000))
    assert report['LN_onset_groups']==2 and report['onset_groups_with_observed_tails']==1
    assert report['censored_LNs']==1 and report['completed_LNs']==1
    assert report['compared_lengths_ms']==0 and not report['largest_ms_changes']
