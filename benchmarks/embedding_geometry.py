"""Label-free geometry of node embeddings, shared by representation_metrics.py (saved models)
and run_benchmark.py --trajectory (during training). Random pairs and edges come from private
generators: computing these never touches the global RNG, so training is unchanged."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from graphssl.evaluation import effective_rank


def linear_cka(x: Tensor, y: Tensor) -> float:
    """Linear CKA of two [N, D] matrices of the same nodes, in float64."""
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    cross = (x.T @ y).pow(2).sum()
    norm = (x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()
    return (cross / norm).item() if norm > 0 else float("nan")


def random_pairs(num_nodes: int, n: int) -> tuple:
    """``n`` pairs of distinct nodes, the same for every checkpoint of a dataset."""
    g = torch.Generator().manual_seed(0)
    i = torch.randint(num_nodes, (n,), generator=g)
    j = (i + torch.randint(1, num_nodes, (n,), generator=g)) % num_nodes
    return i, j


def geometry(z: Tensor, edge_index: Tensor, pairs: tuple, n_edges: int) -> Dict[str, float]:
    u = F.normalize(z.double(), dim=1)
    # The trainer moves the graph to the GPU in place while z comes back on the CPU.
    edge_index = edge_index.to(u.device)
    i, j = pairs
    g = torch.Generator().manual_seed(0)
    e = torch.randint(edge_index.size(1), (min(n_edges, edge_index.size(1)),), generator=g)
    src, dst = edge_index[0, e], edge_index[1, e]
    sq = (u[i] - u[j]).pow(2).sum(1)
    return {
        "pair_cos": (u[i] * u[j]).sum(1).mean().item(),
        "edge_cos": (u[src] * u[dst]).sum(1).mean().item(),
        "uniformity": torch.log(torch.exp(-2 * sq).mean()).item(),
        "feature_std_mean": z.std(0).mean().item(),
        "eff_rank": effective_rank(z),
    }
