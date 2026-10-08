"""DGI: Deep Graph Infomax."""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from graphssl.config.schema import DGIConfig
from graphssl.core.encoder import encode
from graphssl.core.model import BaseSSLModel
from graphssl.nn.pooling import pool_graph_embeddings


class DGI(BaseSSLModel):
    """Mutual information maximisation between node embeddings and global summary.

    Discriminates real node embeddings from a corrupted-graph counterpart using
    a learnable bilinear discriminator W (DGI, Velickovic et al. 2019).

    The summary vector s is computed as sigmoid(mean(h_pos)).  For graph-level
    mode (encoder.pool=True) one summary per graph is computed and broadcast
    to the corresponding nodes via the batch index.  On a NeighborLoader batch
    only the seed nodes (the first ``batch_size``) enter the summary and the loss.

    Args:
        config: Config dict validated against DGIConfig.
        in_channels: Node feature dimensionality, resolved from the dataset.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__()
        cfg = DGIConfig.from_dict(config)
        self.encoder = cfg.encoder.build(in_channels)
        self.graph_level: bool = cfg.encoder.pool
        self.readout: str = cfg.encoder.readout
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
        z = encode(self.encoder, data)
        if self.graph_level:
            # One embedding per graph, like every other model's forward().
            z = pool_graph_embeddings(z, data.batch, self.readout)
        return z

    def _embed_pair(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Node embeddings of the real and of the corrupted graph, from one forward pass.

        Two passes would normalise each graph with its own batch statistics (BatchNorm in
        training mode), and the discriminator can tell the passes apart from those alone.
        On NeighborLoader batches it did: 100% accuracy even on nodes with no incoming
        edge, where real and corrupted embeddings are identically distributed, with the
        embeddings ending up worse than an untrained encoder's.
        """
        corrupted = self._corrupt(data)
        n = data.x.size(0)
        x = torch.cat([data.x, corrupted.x])
        edge_index = torch.cat([data.edge_index, corrupted.edge_index + n], dim=1)
        batch = data.batch
        if batch is not None:
            batch = torch.cat([batch, batch + int(batch.max()) + 1])
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr])  # both corruptions keep the edge order
        both = Data(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        h = encode(self.encoder, both)
        return h[:n], h[n:]

    def compute_loss(self, data: Data) -> Tensor:
        h_pos, h_neg = self._embed_pair(data)

        if self.graph_level:
            # One summary per graph, broadcast to each node via batch index.
            s = torch.sigmoid(global_mean_pool(h_pos, data.batch))
            Ws = s[data.batch] @ self.W
        else:
            # Mini-batch: the sampled neighbours exist for message passing only (most have
            # no incoming edge in the batch), so the summary and the loss use the seeds.
            batch_size = getattr(data, "batch_size", None)
            if batch_size is not None:
                h_pos, h_neg = h_pos[:batch_size], h_neg[:batch_size]
            s = torch.sigmoid(h_pos.mean(0, keepdim=True))
            Ws = s @ self.W  # broadcasts over nodes

        pos_logits = (h_pos * Ws).sum(-1)
        neg_logits = (h_neg * Ws).sum(-1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        return F.binary_cross_entropy_with_logits(logits, labels)

    def student_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.parameters())
