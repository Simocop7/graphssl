"""GraphMAE: masked feature reconstruction."""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from graphssl.config.schema import GraphMAEConfig
from graphssl.core.encoder import encode
from graphssl.core.model import BaseSSLModel
from graphssl.losses.sce import SCELoss
from graphssl.nn.gat_decoder import GATDecoder
from graphssl.nn.pooling import pool_graph_embeddings


class GraphMAE(BaseSSLModel):
    """Masked graph autoencoder (Hou et al., KDD 2022), after the reference implementation
    (THUDM/GraphMAE, ``edcoder.py``).

    A fraction ``mask_rate`` of the nodes is masked. Of those, a fraction ``replace_rate``
    gets the features of random nodes; the rest get a learnable mask token in place of their
    features. The encoder embeds the masked graph; a bias-free linear map and *re-masking*
    (the masked nodes' representations set to zero) feed a one-layer, one-head GAT decoder
    (``GATDecoder``: GATConv's function, computed before the map to the feature space), which
    must rebuild each masked node's features from its neighbors. The loss is the scaled
    cosine error on the masked nodes only. ``forward()`` embeds the clean graph.

    Differences from the reference, all outside its default path: at least one node is
    masked on any graph; the token and noise counts add up to the masked count (the
    reference rounds them separately, leaving up to one masked node unchanged, and fails
    when the rounded noise count is 0 with a non-zero rate); on a NeighborLoader batch the
    seed nodes and the sampled neighbors are masked in the same proportion and the loss uses
    the masked seeds, as every model here computes its loss on the seeds.

    Args:
        config: Config dict validated against GraphMAEConfig.
        in_channels: Node feature dimensionality, resolved from the dataset.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__()
        cfg = GraphMAEConfig.from_dict(config)
        self.encoder = cfg.encoder.build(in_channels)
        self.graph_level: bool = cfg.encoder.pool
        self.readout: str = cfg.encoder.readout
        self.mask_rate: float = cfg.mask_rate
        self.replace_rate: float = cfg.replace_rate
        hidden_dim = cfg.encoder.hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels))
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = GATDecoder(hidden_dim, in_channels)
        self.loss_fn = SCELoss(alpha=cfg.alpha)

    def forward(self, data: Data) -> Tensor:
        z = encode(self.encoder, data)
        if self.graph_level:
            z = pool_graph_embeddings(z, data.batch, self.readout)
        return z

    def _masked_nodes(self, num_nodes: int, batch_size: Optional[int], device) -> Tensor:
        """Indices of the masked nodes; with seeds, seeds and neighbors in equal proportion."""
        if batch_size is None or batch_size >= num_nodes:
            k = max(1, int(self.mask_rate * num_nodes))
            return torch.randperm(num_nodes, device=device)[:k]
        k_seeds = max(1, int(self.mask_rate * batch_size))
        seeds = torch.randperm(batch_size, device=device)[:k_seeds]
        n_rest = num_nodes - batch_size
        rest = batch_size + torch.randperm(n_rest, device=device)[: int(self.mask_rate * n_rest)]
        return torch.cat([seeds, rest])

    def _corrupt(self, x: Tensor, masked: Tensor) -> Tensor:
        """Features with the masked nodes replaced: random nodes' features or the mask token."""
        n_noise = int(self.replace_rate * masked.numel())
        order = torch.randperm(masked.numel(), device=x.device)
        noise, token = masked[order[:n_noise]], masked[order[n_noise:]]
        x_in = x.clone()
        x_in[token] = 0.0
        x_in[noise] = x[torch.randperm(x.size(0), device=x.device)[:n_noise]]
        x_in[token] += self.mask_token
        return x_in

    def compute_loss(self, data: Data) -> Tensor:
        batch_size = None if self.graph_level else getattr(data, "batch_size", None)
        masked = self._masked_nodes(data.x.size(0), batch_size, data.x.device)
        corrupted = Data(
            x=self._corrupt(data.x, masked),
            edge_index=data.edge_index,
            batch=getattr(data, "batch", None),
            edge_attr=getattr(data, "edge_attr", None),
        )
        h = encode(self.encoder, corrupted)
        # Re-mask: the decoder sees no representation of a masked node, so it has to
        # rebuild its features from the neighbors'.
        rep = self.encoder_to_decoder(h).index_fill(0, masked, 0.0)
        reconstruction = self.decoder(rep, data.edge_index)
        target = masked if batch_size is None else masked[masked < batch_size]
        return self.loss_fn(reconstruction[target], data.x[target])

    def student_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.parameters())
