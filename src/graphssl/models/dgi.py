"""DGI: Deep Graph Infomax."""

from __future__ import annotations

from typing import Dict, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from graphssl.core.model import BaseSSLModel
from graphssl.config.schema import DGIConfig


class DGI(BaseSSLModel):
    """Mutual information maximisation between node embeddings and global summary.

    Discriminates real node embeddings from a corrupted-graph counterpart using
    a learnable bilinear discriminator W (DGI, Velickovic et al. 2019).

    The summary vector s is computed as sigmoid(mean(h_pos)).  For graph-level
    mode (encoder.pool=True) one summary per graph is computed and broadcast
    to the corresponding nodes via the batch index.

    Args:
        config: Config dict validated against DGIConfig.
        in_channels: Node feature dimensionality, resolved from the dataset.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__()
        cfg = DGIConfig.from_dict(config)
        self.encoder = cfg.encoder.build(in_channels)
        self.graph_level: bool = cfg.encoder.pool
        self.corruption: str = cfg.corruption
        self.shuffle_ratio: float = cfg.shuffle_ratio
        hidden_dim = cfg.encoder.hidden_dim
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)

    def _corrupt(self, data: Data) -> Data:
        if self.corruption == "shuffle_nodes":
            N = data.x.size(0)
            k = max(1, int(N * self.shuffle_ratio))
            sel = torch.randperm(N, device=data.x.device)[:k]
            perm = torch.randperm(k, device=data.x.device)
            x_corrupt = data.x.clone()
            x_corrupt[sel] = data.x[sel[perm]]
            return Data(x=x_corrupt, edge_index=data.edge_index, batch=data.batch)
        # shuffle_edges: permute destinations of a random subset of edges
        E = data.edge_index.size(1)
        k = max(1, int(E * self.shuffle_ratio))
        sel = torch.randperm(E, device=data.edge_index.device)[:k]
        perm = torch.randperm(k, device=data.edge_index.device)
        dst = data.edge_index[1].clone()
        dst[sel] = data.edge_index[1, sel[perm]]
        return Data(x=data.x, edge_index=torch.stack([data.edge_index[0], dst]), batch=data.batch)

    def forward(self, data: Data) -> Tensor:
        return self.encoder(data.x, data.edge_index, data.batch)

    def compute_loss(self, data: Data) -> Tensor:
        h_pos = self.encoder(data.x, data.edge_index, data.batch)
        corrupted = self._corrupt(data)
        h_neg = self.encoder(corrupted.x, corrupted.edge_index, corrupted.batch)

        if self.graph_level:
            # One summary per graph, broadcast to each node via batch index.
            s = torch.sigmoid(global_mean_pool(h_pos, data.batch))
            Ws = s[data.batch] @ self.W
        else:
            s = torch.sigmoid(h_pos.mean(0, keepdim=True))
            Ws = s @ self.W  # broadcasts over nodes

        pos_logits = (h_pos * Ws).sum(-1)
        neg_logits = (h_neg * Ws).sum(-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        return F.binary_cross_entropy_with_logits(logits, labels)

    def student_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.parameters())
