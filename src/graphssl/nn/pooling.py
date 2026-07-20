"""Graph-level pooling utilities."""

from __future__ import annotations
from typing import Optional
from torch import Tensor
from torch_geometric.nn import global_mean_pool


def pool_graph_embeddings(node_embeddings: Tensor, batch: Optional[Tensor]) -> Tensor:
    """Pool node embeddings to one embedding per graph.

    Falls back to mean over all nodes when batch is None (single-graph input).
    """
    if batch is None:
        return node_embeddings.mean(dim=0, keepdim=True)
    return global_mean_pool(node_embeddings, batch)
