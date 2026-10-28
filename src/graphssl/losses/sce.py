"""Scaled cosine error (GraphMAE)."""

import torch.nn.functional as F
from torch import Tensor, nn

from graphssl.registry import LOSSES


@LOSSES.register("sce")
class SCELoss(nn.Module):
    """Scaled cosine error of GraphMAE (Hou et al., 2022):
    ``mean_i (1 - cos(x_i, y_i)) ** alpha``.

    Insensitive to the scale of each row, so a reconstruction target needs no normalization;
    ``alpha > 1`` down-weights the rows that are already well reconstructed.

    Args:
        alpha: Exponent, >= 1 (``alpha_l`` in the reference implementation).
    """

    def __init__(self, alpha: float = 3.0):
        super().__init__()
        if alpha < 1.0:
            raise ValueError(f"alpha must be >= 1, got {alpha}")
        self.alpha = alpha

    def forward(self, reconstruction: Tensor, target: Tensor) -> Tensor:
        cos = (F.normalize(reconstruction, dim=-1) * F.normalize(target, dim=-1)).sum(-1)
        return (1.0 - cos).pow(self.alpha).mean()
