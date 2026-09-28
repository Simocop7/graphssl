"""NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss for GraphCL."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from graphssl.registry import LOSSES


@LOSSES.register("nt_xent")
class NTXentLoss(nn.Module):
    """Symmetric contrastive loss over two augmented views.

    Positive pair: (z1_i, z2_i). All other pairs in the batch are negatives.

    Args:
        tau: Temperature. Lower = harder negatives. Default 0.5 (GraphCL paper).
        chunk_size: Max rows of the [2N, 2N] similarity matrix materialized at once. When
            2N exceeds it, the loss is computed in row chunks with gradient checkpointing:
            exactly the same value and gradients, but peak memory O(chunk_size * 2N)
            instead of O((2N)^2). Full-batch PubMed (2N = 39,434) needs well over 16 GB
            unchunked. ``None`` always materializes the full matrix.
    """

    def __init__(self, tau: float = 0.5, chunk_size: Optional[int] = 4096):
        super().__init__()
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 or None, got {chunk_size}")
        self.tau = tau
        self.chunk_size = chunk_size

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        N = z1.size(0)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        # Positive indices: (i, i+N) and (i+N, i)
        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels + N, labels])  # [2N]

        if self.chunk_size is None or 2 * N <= self.chunk_size:
            return self._rows_loss(z, z, labels, 0) / (2 * N)

        total = z.new_zeros(())
        for start in range(0, 2 * N, self.chunk_size):
            end = min(start + self.chunk_size, 2 * N)
            chunk_loss = checkpoint(
                self._rows_loss, z[start:end], z, labels[start:end], start, use_reentrant=False
            )
            total = total + chunk_loss
        return total / (2 * N)

    def _rows_loss(self, rows: Tensor, z: Tensor, labels: Tensor, offset: int) -> Tensor:
        """Summed cross-entropy for ``rows`` = z[offset : offset + len(rows)] against all of z."""
        sim = (rows @ z.T) / self.tau  # [C, 2N]

        # Mask self-similarities so they don't contribute as negatives
        r = torch.arange(rows.size(0), device=z.device)
        sim[r, r + offset] = float("-inf")

        return F.cross_entropy(sim, labels, reduction="sum")
