"""Multi-scale audio/chart correspondence with explicit shifted comparators.

Linear centered kernel alignment measures a relation between observations; it
does not establish music understanding or arrangement quality. Steady streams,
deliberate dumps and quiet passages can have low or undefined correspondence.
Never use this diagnostic alone as a BAD-pattern detector or training reward.
"""
import numpy as np

from ..scoped_style_modeling.dataset import ContractError


def standardized(values):
    centered=values-values.mean(0,keepdims=True)
    scale=centered.std(0,keepdims=True)
    return np.divide(centered,scale,out=np.zeros_like(centered),where=scale>1e-8)


def linear_cka(left,right):
    """Compare centered Gram geometry; return None when either side is constant.

    Both inputs have the same ordered observations. This implementation uses
    feature-space products and does not allocate a time-by-time kernel matrix.
    """
    x,y=np.asarray(left,float),np.asarray(right,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2:
        raise ContractError('Alignment requires matching nontrivial observation sequences')
    x=x-x.mean(0);y=y-y.mean(0)
    denominator=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return None if denominator<=1e-16 else float(np.linalg.norm(x.T@y)**2/denominator)


class MelDescriptors:
    """Eight log-power bands and their positive frame-to-frame changes.

    Input is the canonical 128-bin, 10-ms-hop, 40-ms-window natural-log Mel.
    Features retain the 20+10*i ms center clock. Frame centers select averaging
    support; the 40-ms analysis window limits onset interpretation.
    """
    def __init__(self,mel):
        mel=np.asarray(mel,dtype=float)
        if mel.ndim!=2 or mel.shape[1]!=128 or not len(mel) or not np.isfinite(mel).all():
            raise ContractError('Audio correspondence requires finite complete canonical Mel')
        bands=np.logaddexp.reduce(mel.reshape(-1,8,16),axis=-1)-np.log(16)
        flux=np.maximum(np.diff(bands,axis=0,prepend=bands[:1]),0.)
        self.centers=20+10*np.arange(len(mel))
        self.prefix=np.vstack((np.zeros((1,16)),np.cumsum(np.c_[bands,flux],axis=0)))

    def average(self,left,right):
        lo=np.searchsorted(self.centers,left,side='left')
        hi=np.searchsorted(self.centers,right,side='left')
        if np.any(hi<=lo) or np.any(np.asarray(right)>self.centers[-1]+20):
            raise ContractError('Audio observation windows must contain available real frame centers')
        return (self.prefix[hi]-self.prefix[lo])/(hi-lo)[...,None]


def audio_correspondence(trace,mel,scope,*,windows_ms=(1000,4000,16000),step_ms=250):
    """Keep every control scope separate and compare zero-lag with clock shifts.

    Shift values are descriptive controls, not exchangeable draws or p-values.
    They preserve each sequence's marginal values but disrupt its alignment.
    No source timing is assumed to be the unique correct arrangement.
    """
    trace._scope(scope)
    descriptor=mel if isinstance(mel,MelDescriptors) else MelDescriptors(mel)
    result={}
    for width in windows_ms:
        left=np.arange(scope.start_ms,scope.end_ms-width+1,step_ms)
        if len(left)<8:
            result[str(width)]=dict(status='insufficient_scope',observations=len(left));continue
        x=standardized(descriptor.average(left,left+width))
        y=standardized(trace.interval_features(left,left+width))
        score=linear_cka(x,y)
        if score is None:
            result[str(width)]=dict(status='constant_audio_or_chart',observations=len(left),aligned=None);continue
        offsets=np.unique(np.round(np.array([.17,.33,.5,.67,.83])*len(left)).astype(int))
        shifted=[dict(shift_ms=int(k*step_ms),alignment=linear_cka(x,np.roll(y,k,axis=0))) for k in offsets]
        values=[r['alignment'] for r in shifted]
        result[str(width)]=dict(status='measured',observations=len(left),aligned=score,
            shifted=shifted,aligned_minus_shift_median=float(score-np.median(values)))
    return result
