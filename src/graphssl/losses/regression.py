"""Cosine regression loss for bootstrapped SSL (BGRL / AFGRL)."""

from __future__ import annotations
import torch.nn.functional as F
from torch import nn, Tensor

from graphssl.registry import LOSSES


@LOSSES.register("cosine_regression")
class CosineRegressionLoss(nn.Module):
    """Cosine similarity regression loss.

    Symmetric (BGRL):
        loss = 2 - cos_sim(p1, t2) - cos_sim(p2, t1)

    Asymmetric / single-sided (AFGRL):
        loss = 1 - cos_sim(p, t).mean()

    Args:
        symmetric: If True (default) expects all four tensors (p1, t1, p2, t2).
    """

    def __init__(self, symmetric: bool = True):
        super().__init__()
        self.symmetric = symmetric

    def forward(
        self,
        p1: Tensor,
        t1: Tensor,
        p2: Tensor | None = None,
        t2: Tensor | None = None,
    ) -> Tensor:
        p1_n = F.normalize(p1, dim=-1)
        t1_n = F.normalize(t1, dim=-1)

        if self.symmetric:
            assert p2 is not None and t2 is not None
            p2_n = F.normalize(p2, dim=-1)
            t2_n = F.normalize(t2, dim=-1)
            return (
                2
                - (p1_n * t2_n).sum(-1).mean()
                - (p2_n * t1_n).sum(-1).mean()
            )
        return 1 - (p1_n * t1_n).sum(-1).mean()
