"""CombinedLoss: weighted sum of multiple registered losses.

Build programmatically::

    from graphssl.losses import CombinedLoss, NTXentLoss, VICRegLoss

    loss_fn = CombinedLoss([
        (NTXentLoss(tau=0.5), 0.7),
        (VICRegLoss(invariance=25.0, variance=25.0, covariance=1.0), 0.3),
    ])
    total = loss_fn(z1, z2)

Or from a config list (all keys except 'name' and 'weight' are forwarded to the constructor)::

    loss_fn = CombinedLoss.from_config([
        {"name": "nt_xent", "weight": 0.7, "tau": 0.5},
        {"name": "vicreg",  "weight": 0.3, "invariance": 25.0},
    ])
"""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn
from torch import Tensor

from graphssl.registry import LOSSES


class CombinedLoss(nn.Module):
    """Weighted sum of multiple loss functions.

    Weights are plain scalars and need not sum to 1.
    All component losses must share the same call signature.
    """

    def __init__(self, losses: List[Tuple[nn.Module, float]]):
        super().__init__()
        if not losses:
            raise ValueError("CombinedLoss requires at least one (loss, weight) pair.")
        self._loss_modules = nn.ModuleList([m for m, _ in losses])
        self._weights = [w for _, w in losses]

    def forward(self, *args, **kwargs) -> Tensor:
        total: Tensor | float = 0.0
        for module, weight in zip(self._loss_modules, self._weights, strict=True):
            total = total + weight * module(*args, **kwargs)
        return total  # type: ignore[return-value]

    @classmethod
    def from_config(cls, config_list: list) -> "CombinedLoss":
        """Build from a list of dicts with 'name', 'weight', and loss-specific kwargs."""
        pairs: List[Tuple[nn.Module, float]] = []
        for entry in config_list:
            entry = dict(entry)
            name = entry.pop("name")
            weight = float(entry.pop("weight", 1.0))
            pairs.append((LOSSES.build(name, **entry), weight))
        return cls(pairs)

    def __repr__(self) -> str:
        parts = [
            f"  ({w:.3g}) {m.__class__.__name__}"
            for m, w in zip(self._loss_modules, self._weights, strict=True)
        ]
        return "CombinedLoss(\n" + "\n".join(parts) + "\n)"
