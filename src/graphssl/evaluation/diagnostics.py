"""Representation diagnostics that accuracy alone can hide."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def effective_rank(embeddings: Tensor, eps: float = 1e-12) -> float:
    """Effective rank of an embedding matrix (Roy & Vetterli, 2007).

    ``exp(H(p))``, where ``p`` are the singular values of the column-centered
    ``embeddings`` [N, D] normalized to sum to 1. It ranges from 1 (all variance along a
    single direction) to ``min(N, D)`` (variance spread evenly), and is 0 when the
    embeddings have no variance at all. A value far below ``D`` signals dimensional
    collapse, a common failure mode of non-contrastive SSL objectives.
    """
    z = embeddings.detach().float()
    s = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
    total = s.sum()
    if total <= eps:
        return 0.0
    p = s / total
    entropy = -(p * p.clamp_min(eps).log()).sum()
    return float(entropy.exp())
