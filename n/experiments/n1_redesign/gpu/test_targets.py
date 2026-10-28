"""Absolute target-clock fixtures for resets, fractional tempos, stretching and slider events."""
import numpy as np
from ensomi_n.timing import Reference, Segment
from n.experiments.n1_redesign.gpu.targets import compact, dense_new
from n.experiments.n1_redesign.gpu.osu_notes import parse_osu, onsets_and_events


def test_notated_snap_resets_at_fractional_boundary_and_keeps_song_end():
    a=.5; length=60000/123.45; b=a+2.5*length; second=60000/177.7
    heads=np.array([a,a+length/2,a+length,a+length+length/3,b-1,b+second/2,b+second])
    ref=Reference.from_segments([Segment(a,length,4),Segment(b,second,7)],heads)
    t=compact(ref)
    assert t['snap'].tolist()==[0,1,0,2,0,1,0]
    assert set(t['cell'][t['cell']>=0])=={0,1}
    assert np.all(t['ends']<=np.r_[t['lines'][1:],np.inf]+1e-8)
    assert np.min(abs(t['lines']-b))<1e-8
    n=int(np.round(heads[-1]/10))+1
    d=dense_new(t,0,n)
    assert d['attack_y'][-1]==1 and d['attack_w'][-1]==1


def test_stretch_happens_before_rounding_and_missing_note_is_not_negative():
    ref=Reference.from_segments([Segment(0,500,4)],np.array([45.,250.,1000.]))
    t=compact(ref)
    d=dense_new(t,0,130,rate=1.1)
    assert d['attack_w'][6]==1 and d['attack_y'][6]==1
    assert d['attack_y'][15]==0 and d['attack_w'][15]==1  # dense first beat
    assert d['attack_w'][80]==0  # empty next beat provides no negative evidence
    assert d['attack_w'][8]==0  # between the positive window and 40 ms negative boundary


def test_cumulative_salience_bins_and_slider_retention_clock():
    text='''osu file format v14
[General]
AudioFilename:a.mp3
Mode:0
[Difficulty]
SliderMultiplier:1.4
[TimingPoints]
0,500,4,1,0,100,1,0
499,-50,4,1,0,100,0,0
[HitObjects]
100,100,500,2,0,B|200:100,2,140,0|0|0,0:0|0:0|0:0,0:0:0:0:
'''
    h,e=onsets_and_events(parse_osu(text.encode()))
    assert h.tolist()==[500.]
    assert e.tolist()==[500.,750.,1000.]
    times=np.array([0.,500.,1000.,1500.])
    t=compact(Reference.from_segments([Segment(0,500,4)],times),times,[0,.25,.5,1])
    d=dense_new(t,0,151)
    assert d['sal_y'][[0,50,100,150]].tolist()==[[0,0,0],[1,0,0],[1,1,0],[1,1,1]]
