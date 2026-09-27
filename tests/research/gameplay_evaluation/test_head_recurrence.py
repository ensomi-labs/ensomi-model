import numpy as np
import pytest

from ensomi_model.research.gameplay_evaluation.head_recurrence import head_recurrence
from ensomi_model.research.gameplay_evaluation.temporal import ChartTrace, Scope
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.player_response.envelope import AttackEnvelope,sustained_response
from ensomi_model.research.player_response.state import CommittedPlayState


def measure(rows,a=0,b=4000):
    trace=ChartTrace(rows,max(b,1 if not rows else int(rows[-1].time_ms)+1))
    return head_recurrence(trace,Scope('test',a,b))


def rows_for_columns(columns,gap=125):
    return [CompleteRow(i*gap,tuple(int(k==c) for k in range(4))) for i,c in enumerate(columns)]


def test_equal_counts_and_zero_attack_excess_can_hide_repeated_column_blocks():
    blocked=rows_for_columns([i//8 for i in range(32)])
    rotating=rows_for_columns([i%4 for i in range(32)])
    assert [r.time_ms for r in blocked]==[r.time_ms for r in rotating]
    assert np.array_equal(np.asarray([r.actions for r in blocked]).sum(0),
                          np.asarray([r.actions for r in rotating]).sum(0))
    rates=(10.,8.,7.,6.,5.375,5.)
    envelope=AttackEnvelope((500,1000,2000,4000,8000,16000),(3.,5.),(rates,rates),'counterexample-only')
    for rows in (blocked,rotating):
        _,response=sustained_response(CommittedPlayState(),rows,4000,envelope,((-1,4000,4.),))
        assert response['excess_seconds']==0
    a,b=measure(blocked),measure(rotating)
    assert a['heads']==b['heads']==32
    assert a['heads_repeating_previous_H_column']==28 and b['heads_repeating_previous_H_column']==0
    assert a['maximum_observed_prefix_H_run_age']==8 and b['maximum_observed_prefix_H_run_age']==1
    assert a['witnesses'][0]['span_ms']==875
    assert a['witnesses'][0]['maximum_HH_gap_ms']==125


def test_scope_continuity_and_pure_release_are_distinct_from_head_membership():
    rows=[CompleteRow(0,(2,0,0,1)),CompleteRow(100,(3,0,0,0)),CompleteRow(200,(1,0,0,0)),
          CompleteRow(300,(1,0,2,0)),CompleteRow(400,(0,0,3,0)),CompleteRow(500,(0,1,0,0))]
    full,left,right=[measure(rows,a,b) for a,b in [(0,501),(0,300),(300,501)]]
    for key in ('heads','heads_with_previous_H','heads_repeating_previous_H_column'):
        assert full[key]==left[key]+right[key]
    assert measure(rows,150,400)['prefix_H_run_age_head_counts']=={'1':1,'2':1,'3':1}
    first=measure(rows,150,400)['witnesses'][0]
    assert first['observed_consecutive_H']==3 and first['run_start_ms']==0
    assert first['started_before_scope'] and first['future_membership_unobserved']


def test_no_future_events_are_used_for_run_ages_or_witnesses():
    rows=rows_for_columns([0,0,0,1,1])
    assert measure(rows,100,300)==measure(rows[:3],100,300)
    empty=measure(rows,300,350)
    assert empty['heads']==0 and empty['prefix_H_run_age'] is None
    assert empty['repeated_head_fraction'] is None and empty['witnesses']==[]


def test_long_gaps_are_reported_without_an_arbitrary_run_reset_or_bad_label():
    r=measure(rows_for_columns([0]*4,gap=2000),0,7000)
    assert r['maximum_observed_prefix_H_run_age']==4
    assert r['witnesses'][0]['maximum_HH_gap_ms']==2000
    assert r['witnesses'][0]['span_ms']==6000
    assert 'bad' not in r and 'style' not in r


def test_mirror_preserves_global_counts_and_reflects_column_facts():
    rows=rows_for_columns([0,0,0,1,2,2,3])
    a=measure(rows);b=measure([CompleteRow(r.time_ms,r.actions[::-1]) for r in rows])
    for key in ('heads','heads_with_previous_H','heads_repeating_previous_H_column',
                'repeated_head_fraction','prefix_H_run_age_head_counts','prefix_H_run_age'):
        assert a[key]==b[key]
    assert a['repeat_heads_per_column']==b['repeat_heads_per_column'][::-1]


def test_empty_and_actual_scope_report_serialize_without_inventing_recurrence():
    import json
    from ensomi_model.research.gameplay_evaluation.report import evaluate_scopes
    trace=ChartTrace([],500)
    result=evaluate_scopes(trace,[Scope('empty',0,500)])
    observed=result['scopes'][0]['head_recurrence']
    assert observed['heads']==0 and observed['repeat_heads_per_column']==[0,0,0,0]
    assert observed['prefix_H_run_age'] is None and observed['witnesses']==[]
    json.dumps(result,allow_nan=False)


def test_equal_recurrent_finger_timing_can_have_different_companion_roles():
    solo=rows_for_columns([0]*4,gap=150)
    accompanied=[CompleteRow(r.time_ms,(1,int(i%2==0),int(i%2==1),0))
                 for i,r in enumerate(solo)]
    a,b=measure(solo)['witnesses'][0],measure(accompanied)['witnesses'][0]
    for key in ('column','observed_consecutive_H','span_ms','median_HH_gap_ms','maximum_HH_gap_ms'):
        assert a[key]==b[key]
    assert a['companion_heads']==0 and a['recurrent_column_head_share']==1
    assert b['companion_heads']==4 and b['recurrent_column_head_share']==.5
    assert b['companion_heads_per_column']==[0,2,2,0]
    # The witness keeps pre-scope accompaniment just as it keeps its run age.
    incoming=measure(accompanied,300,600)['witnesses'][0]
    assert incoming['companion_heads']==4 and incoming['started_before_scope']


def test_no_companion_heads_can_still_mean_three_other_fingers_are_holding():
    common=[CompleteRow(200+125*i,(0,0,1,0)) for i in range(4)]
    free=[CompleteRow(0,(1,1,0,1)),*common]
    held=[CompleteRow(0,(2,2,0,2)),*common]
    a,b=[measure(rows,200,900)['witnesses'][0] for rows in (free,held)]
    assert a['observed_consecutive_H']==b['observed_consecutive_H']==4
    assert a['companion_heads']==b['companion_heads']==0
    assert a['continuing_other_hold_pairs']==0
    assert b['continuing_other_hold_pairs']==12
    assert b['continuing_other_hold_pairs_per_column']==[4,4,0,4]
    assert b['maximum_continuing_other_holds']==3
    assert b['other_hold_starts_at_run_start_ms']==[0.,0.,None,0.]
    # A later true close cannot change the witness at the earlier horizon.
    later=[*held,CompleteRow(1000,(3,3,0,3))]
    assert measure(held,200,900)==measure(later,200,900)
    released=[*held[:-1],CompleteRow(575,(0,3,1,0))]
    c=measure(released,200,900)['witnesses'][0]
    assert c['continuing_other_hold_pairs']==11
    assert c['companion_releases']==1
    assert c['companion_releases_per_column']==[0,1,0,0]


def test_same_recurrence_can_be_taps_or_ln_presses_with_intervening_releases():
    taps=rows_for_columns([0]*4,gap=200)
    longs=[r for i in range(4) for r in (
        CompleteRow(i*200,(2,0,0,0)),CompleteRow(i*200+100,(3,0,0,0)))]
    a,b=[measure(rows,0,750)['witnesses'][0] for rows in (taps,longs)]
    assert a['observed_consecutive_H']==b['observed_consecutive_H']==4
    assert a['median_HH_gap_ms']==b['median_HH_gap_ms']==200
    assert (a['recurrent_TAP_heads'],a['recurrent_LN_heads'])==(4,0)
    assert (b['recurrent_TAP_heads'],b['recurrent_LN_heads'])==(0,4)
    assert a['releases_in_run_span_per_column']==[0,0,0,0]
    # The true fourth release is after the witness's last head, so is excluded.
    assert b['releases_in_run_span_per_column']==[3,0,0,0]
