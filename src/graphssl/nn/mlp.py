"""Projection and predictor heads for SSL models."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

_ACTIVATIONS = {"relu": nn.ReLU, "prelu": nn.PReLU}


class MLP(nn.Module):
    """Generic MLP: Linear → (BN → activation → Dropout) × (L-1) → Linear.

    No norm/activation on the final layer so it can serve as raw projector output.
    ``norm=False`` drops the batch normalization; ``activation`` is 'relu' or 'prelu'
    (one learnable slope per layer).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        assert num_layers >= 1
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be 'relu' or 'prelu', got {activation!r}")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(_ACTIVATIONS[activation]())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if m is self:
                continue
            reset_fn = getattr(m, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()


class Projector(MLP):
    """Projection head: maps encoder output to the SSL loss space."""


class Predictor(MLP):
    """Predictor head: maps online projection to target projection (BGRL/AFGRL).

    The default is Linear → BatchNorm → ReLU → Linear. ``norm=False, activation="prelu"``
    gives Linear → PReLU → Linear, the predictor of the BGRL reference implementation.
    """
