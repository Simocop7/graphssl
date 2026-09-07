"""NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss for GraphCL."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graphssl.registry import LOSSES


@LOSSES.register("nt_xent")
class NTXentLoss(nn.Module):
    """Symmetric contrastive loss over two augmented views.

    Positive pair: (z1_i, z2_i). All other pairs in the batch are negatives.

    Args:
        tau: Temperature. Lower = harder negatives. Default 0.5 (GraphCL paper).
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        N = z1.size(0)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        sim = (z @ z.T) / self.tau  # [2N, 2N]

        # Mask self-similarities so they don't contribute as negatives
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive indices: (i, i+N) and (i+N, i)
        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels + N, labels])  # [2N]

        return F.cross_entropy(sim, labels)
