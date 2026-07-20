"""Weight standardization utility for GNN layers."""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn


def weight_standardize(weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalize each output filter independently: (w - mean_i) / sqrt(var_i + eps)."""
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    return (weight - mean) / (var + eps).sqrt()


def apply_weight_standardization(module: nn.Module, eps: float = 1e-5) -> None:
    """In-place weight standardization on all weight parameters of *module*.

    Applies to any parameter whose name contains 'weight' and has >= 2 dims.
    Called inside forward() before the convolution, so gradients remain intact.
    """
    for name, param in module.named_parameters(recurse=False):
        if "weight" in name and param.dim() >= 2:
            param.data = weight_standardize(param.data, eps)
