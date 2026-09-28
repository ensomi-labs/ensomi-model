import numpy as np

from ensomi_model.research.bounded_typed_continuation.features import TIME_DIM
from ensomi_model.research.controlled_audio_continuation.conditional_views import style_ln_views,StyleWithoutLn
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule,ControlSpan,PER_FIELD_SCOPE


def schedule():
    return ControlSchedule((ControlSpan(0,4000,stars=4,ln_fraction=.6),
        ControlSpan(1000,2000,style={'stream':0.}),
        ControlSpan(2500,3000,style={'tech':-1.})),('stream','tech'))


def test_style_present_supporting_or_absent_is_observed_not_a_numeric_zero_mask():
    controls=schedule();times=[999,1000,1999,2000,2500,2999,3000]
    source,view=style_ln_views(controls,0,4000)
    a=source.at(times,encoding=PER_FIELD_SCOPE)
    b=view.at(times,encoding=PER_FIELD_SCOPE)
    n=4;hidden=[1,2,4,5]
    expected=a.copy()
    expected[hidden,1]=expected[hidden,n+1]=0
    expected[hidden,2*n+2*TIME_DIM:2*n+4*TIME_DIM]=0
    np.testing.assert_array_equal(b,expected)
    np.testing.assert_array_equal(source.at(times,encoding=PER_FIELD_SCOPE),a)


def test_annotation_elsewhere_does_not_hide_unannotated_balanced_data():
    controls=schedule()
    assert style_ln_views(controls,0,1000)==(controls,)
    assert style_ln_views(controls,2000,2500)==(controls,)
    assert len(style_ln_views(controls,999,1001))==2
    view=StyleWithoutLn(controls)
    np.testing.assert_array_equal(view.at([500,2000,3500]),controls.at([500,2000,3500]))


def test_already_unknown_amount_has_no_duplicate_training_view():
    controls=ControlSchedule((ControlSpan(0,2000,stars=4,style={'stream':1.}),),('stream',))
    assert style_ln_views(controls,0,2000)==(controls,)


def test_query_view_cannot_silently_feed_original_amount_to_a_feedback_controller():
    view=StyleWithoutLn(schedule())
    assert not hasattr(view,'spans')
