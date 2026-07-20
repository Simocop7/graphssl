"""Projection and predictor heads for SSL models."""

from __future__ import annotations
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Generic MLP: Linear → (BN → ReLU → Dropout) × (L-1) → Linear.

    No norm/activation on the final layer so it can serve as raw projector output.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        norm: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()


class Projector(MLP):
    """Projection head: maps encoder output to the SSL loss space."""


class Predictor(MLP):
    """Predictor head: maps online projection to target projection (BGRL/AFGRL)."""
