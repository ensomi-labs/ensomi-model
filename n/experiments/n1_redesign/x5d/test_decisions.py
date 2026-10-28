"""Both-seed decisions, the extra prediction gain and adjacent confusion signs."""
from common import *
from summarize import decide, accuracy_summary, PANELS, CELLS
from evaluate import ARMS, confusion


def comparison_fixture(linear=.07, mlp=.07, bars=0.):
    return {p:{c:{a:{m:dict(margin=.04,seeds={s:dict(delta=dict(value=(bars if m=='n_bars_same' else
                linear if a=='band_d2' else mlp))) for s in SEEDS})
                for m in ('n_level','n_bars_same')} for a in ARMS} for c in CELLS} for p in PANELS}


def test_prediction_requires_extra_two_points_and_both_seeds():
    c=comparison_fixture(linear=.05)
    assert not decide(c,SEEDS)['panel']['all']['prediction']
    c=comparison_fixture()
    assert decide(c,SEEDS)['panel']['all']['prediction']
    c['panel']['all']['band_d2']['n_bars_same']['seeds']['s1']['delta']['value']=-.05
    assert not decide(c,SEEDS)['panel']['all']['prediction']
    assert not decide(comparison_fixture(),('s0',))['panel']['all']['prediction']


def test_retirement_requires_both_heads_and_both_seeds():
    assert decide(comparison_fixture(-.06,-.06),SEEDS)['panel']['all']['retire']
    assert not decide(comparison_fixture(-.06,-.03),SEEDS)['panel']['all']['retire']
    c=comparison_fixture(-.06,-.06)
    c['panel']['all']['band_d2_mlp']['n_level']['seeds']['s1']['delta']['value']=0.
    assert not decide(c,SEEDS)['panel']['all']['retire']


def test_confusion_orientation_and_recording_equal_weights():
    rows=[]
    for rec, truth, pred in [('a',[2]*100,[1]*100),('b',[2],[3])]:
        counts=confusion(truth,pred)
        rows.append(dict(panel='panel',cell='all',seed='s0',arm='band_d2',rec=rec,
            frame=counts,regime=counts,omitted_regimes=0))
    a=accuracy_summary(rows,('s0',))['panel']['all']['s0']['band_d2']['frame']
    assert a['adjacent_lower']['value']==.5
    assert a['adjacent_higher']['value']==.5
    assert a['confusion'][2][1]==100
