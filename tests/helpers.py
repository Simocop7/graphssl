"""Shared test helpers for graph data construction."""

import torch
from torch_geometric.data import Data, Batch


def make_batch(n_graphs: int = 4, n_nodes: int = 10, n_feat: int = 7) -> Batch:
    graphs = []
    for _ in range(n_graphs):
        graphs.append(Data(
            x=torch.randn(n_nodes, n_feat),
            edge_index=torch.stack([
                torch.randint(0, n_nodes, (n_nodes * 2,)),
                torch.randint(0, n_nodes, (n_nodes * 2,)),
            ]),
        ))
    return Batch.from_data_list(graphs)


def make_graph(n_nodes: int = 20, n_feat: int = 7) -> Data:
    return Data(
        x=torch.randn(n_nodes, n_feat),
        edge_index=torch.stack([
            torch.randint(0, n_nodes, (n_nodes * 3,)),
            torch.randint(0, n_nodes, (n_nodes * 3,)),
        ]),
    )
