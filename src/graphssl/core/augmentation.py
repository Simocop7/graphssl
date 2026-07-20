from __future__ import annotations
from typing import Protocol, runtime_checkable
from torch_geometric.data import Data


@runtime_checkable
class BaseAugmentation(Protocol):
    """Structural interface for graph augmentation transforms.

    Any callable Data -> Data satisfies this protocol, matching the
    torchvision-style API used in augmentation/transforms.py.
    """

    def __call__(self, data: Data) -> Data: ...
