"""Absolute fixtures for octave folding, parity and structural error controls."""
import importlib
import numpy as np
from base import MusicMap
from ensomi_n.timing import Reference, Segment
from measurements import structural, null_map, lattice_chance, meter_runs, candidates
from level import band_ratios, project

dec=importlib.import_module('ensomi_n.v0.decode')


def reference(segments=None):
    return Reference.from_segments(segments or [Segment(0.,500.,4)],np.arange(0,64001,250.),lo_ms=0.,hi_ms=64000.)


def test_structural_identity_and_double_level():
    ref=reference()
    m=MusicMap(segments=ref.segments)
    assert structural(ref,m)['ready']
    doubled=null_map(ref,'synthetic','x2')
    s=structural(ref,doubled)
    assert not s['ready']
    assert all(r['cls']=='level x2' for r in s['spans'])


def test_fractional_boundary_identity():
    ref=reference([Segment(0.,500.3,4),Segment(18001.7,431.2,3),Segment(39003.2,431.2,4)])
    assert structural(ref,MusicMap(segments=ref.segments))['ready']


def test_insertions_cannot_pass_recall():
    ref=reference();t=ref.timing
    b=np.sort(np.r_[t.beats,10125.])
    down=np.flatnonzero(np.isin(b,t.bar_starts()))
    s=structural(ref,MusicMap.from_beats(b,down))
    assert not s['ready']
    assert s['counts']['inserted lines']==1


def test_offset_only():
    ref=reference()
    m=MusicMap(segments=[Segment(8.,500.,4)])
    assert structural(ref,m,8.)['ready']
    assert 'offset-only' in structural(ref,m,8.)['counts']


def test_fold_does_not_halve_pulse_already_in_band():
    n=200
    e=dec.Evidence(np.ones(n),level=np.tile([.97,.01,.01,.01],(n,1)),
                   logp=np.full(n,np.log(60000/158/1000)),logp_sigma=np.full(n,.005))
    cfg=dec.DecodeConfig()
    assert band_ratios(e,[(0,n,False)],[60000/79],cfg)==[1.]
    assert band_ratios(e,[(0,n,False)],[60000/158],cfg)==[.5]


def test_mean_parity_on_unequal_counts():
    n=300;t=np.arange(5)*500.
    down=np.zeros(n)
    for i,time in enumerate(t):down[int(time/10)]=.6 if i%2 else .5
    e=dec.Evidence(np.ones(n),down=down,level=np.tile([.97,.01,.01,.01],(n,1)))
    # Keyword construction guards against internal dataclass ordering changes.
    r=dec._Regime(i0=0,i1=5,kind='constant',a=0.,L=500.)
    segs,_,_=project(e,[r],t,np.ones(5),dec.DecodeConfig(),'d4')
    assert segs[1].start==500.


def test_projection_preserves_original_without_intervention():
    n=1000;t=np.arange(17)*500.
    e=dec.Evidence(np.ones(n),down=np.linspace(.1,.9,n),
                  level=np.tile([.97,.01,.01,.01],(n,1)))
    regs=[dec._Regime(kind='constant',i0=0,i1=9,a=0.,L=500.),
          dec._Regime(kind='constant',i0=9,i1=17,a=4500.,L=500.)]
    a=dec._notated(e,regs,t,np.ones(17),dec.DecodeConfig())
    b=project(e,regs,t,np.ones(17),dec.DecodeConfig(),'original')
    assert a[0]==b[0]
    np.testing.assert_array_equal(a[1],b[1])
    assert a[2]==b[2]


def test_lattice_union_and_bar_argmax():
    assert abs(lattice_chance(120,[2])-.16)<1e-10
    assert lattice_chance(160,[2,3,4,6,8])<=1.
    d=np.tile([5.,-2.,-2.,-2.],5)
    assert candidates(d)[0]==[(4,0)]
    assert meter_runs(reference())[0]['meter']==4
