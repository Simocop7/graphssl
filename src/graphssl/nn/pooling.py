"""Graph-level pooling utilities."""

from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

READOUTS = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}


def pool_graph_embeddings(
    node_embeddings: Tensor, batch: Optional[Tensor], readout: str = "mean"
) -> Tensor:
    """Pool node embeddings to one embedding per graph.

    ``readout`` is 'mean', 'sum' or 'max'. The mean does not see how large a graph is; the
    sum does, which matters when the target grows with the number of nodes. With
    ``batch=None`` every node belongs to one graph.
    """
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {sorted(READOUTS)}, got {readout!r}")
    return READOUTS[readout](node_embeddings, batch)
