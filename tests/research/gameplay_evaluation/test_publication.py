import pytest

from ensomi_model.research.gameplay_evaluation.publication import publication_report


def test_average_faster_than_realtime_does_not_hide_a_late_dense_publication():
    events=[dict(coverage_ms=g,elapsed_seconds=t) for g,t in
            ((1999,.2),(3999,.4),(5999,7.),(10000,7.2))]
    result=publication_report(events,10000,lead_ms=0,startup_seconds=.2)
    assert result['whole_generation_to_audio_ratio']==pytest.approx(.72)
    assert result['required_startup_seconds']==3
    assert not result['trace_meets_deadlines']
    assert result['maximum_lateness_seconds']==pytest.approx(2.8)
    assert result['witness']['previous_coverage_ms']==3999


def test_initial_lookahead_and_later_buffer_drain_both_constrain_startup():
    events=[dict(coverage_ms=g,elapsed_seconds=t) for g,t in
            ((499,.1),(1999,.4),(3999,.6),(10000,3.))]
    result=publication_report(events,10000,lead_ms=2000,startup_seconds=1.)
    assert result['required_startup_seconds']==1.
    assert result['trace_meets_deadlines']
    assert result['deadline_misses']==0


def test_incomplete_generation_cannot_claim_playback_even_with_large_startup():
    result=publication_report([dict(coverage_ms=4999,elapsed_seconds=.3)],10000,startup_seconds=20)
    assert result['status']=='incomplete_trace'
    assert result['deadline_misses']==0 and not result['trace_meets_deadlines']
