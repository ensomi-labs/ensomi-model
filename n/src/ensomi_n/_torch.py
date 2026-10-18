"""Torch, imported once, with a stand-in where torch is not installed.

Model and training modules subclass ``nn.Module``. On a machine without torch
(the control plane runs the numpy tests) they must still import, so that the
import-boundary test can load every module; instantiating one then raises.
"""
from __future__ import annotations

import types

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - exercised on the control plane only
    HAVE_TORCH = False

    class _Missing:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError('torch is required for this part of ensomi_n')

    torch = None
    F = None
    nn = types.SimpleNamespace(Module=_Missing)

__all__ = ['torch', 'nn', 'F', 'HAVE_TORCH']
