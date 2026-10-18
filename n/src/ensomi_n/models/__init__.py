"""N models: the shared encoder (NE-S, NE-M) and the rung-1 heads (F1 frame heads, F2 phase heads).

Torch is needed to build them; without it the modules still import (see `ensomi_n._torch`).
"""
from .encoder import NEM, NES, n_params
from .heads import HEADS, FrameHeads, Model, PhaseHeads

__all__ = ['NES', 'NEM', 'n_params', 'HEADS', 'FrameHeads', 'PhaseHeads', 'Model']
