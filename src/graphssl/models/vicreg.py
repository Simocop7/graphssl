"""VICReg: Variance-Invariance-Covariance Regularization for graphs."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from graphssl.augmentation.compose import compose
from graphssl.config.schema import VICRegConfig
from graphssl.core.model import BaseSSLModel
from graphssl.losses.vicreg import VICRegLoss
from graphssl.nn.mlp import Projector
from graphssl.nn.pooling import pool_graph_embeddings


class VICReg(BaseSSLModel):
    """Symmetric augmentation with triple VICReg loss (invariance + variance + covariance).

    Args:
        config: Config dict validated against VICRegConfig.
        in_channels: Node feature dimensionality, resolved from the dataset.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__()
        cfg = VICRegConfig.from_dict(config)
        self.encoder = cfg.encoder.build(in_channels)
        hidden_dim = cfg.encoder.hidden_dim
        self.projector = Projector(hidden_dim, hidden_dim * 2, cfg.proj_dim, num_layers=3)
        self.loss_fn = VICRegLoss(
            invariance=cfg.invariance,
            variance=cfg.variance,
            covariance=cfg.covariance,
        )
        self.aug_list: List[Tuple[str, dict]] = [(a.name, a.kwargs) for a in cfg.augment]
        self.graph_level: bool = cfg.encoder.pool

    def forward(self, data: Data) -> Tensor:
        z = self.encoder(data.x, data.edge_index, data.batch)
        if self.graph_level:
            z = pool_graph_embeddings(z, data.batch)
        return z

    def compute_loss(self, data: Data) -> Tensor:
        protected: Optional[Tensor] = None
        batch_size: Optional[int] = None
        if not self.graph_level:
            batch_size = getattr(data, "batch_size", None)
            if batch_size is not None:
                _dev = data.x.device if data.x is not None else torch.device("cpu")
                protected = torch.arange(batch_size, device=_dev)

        v1 = compose(data, self.aug_list, protected_nodes=protected)
        v2 = compose(data, self.aug_list, protected_nodes=protected)

        h1 = self.encoder(v1.x, v1.edge_index, v1.batch)
        h2 = self.encoder(v2.x, v2.edge_index, v2.batch)

        if self.graph_level:
            h1 = pool_graph_embeddings(h1, v1.batch)
            h2 = pool_graph_embeddings(h2, v2.batch)
        elif batch_size is not None:
            h1, h2 = h1[:batch_size], h2[:batch_size]

        return self.loss_fn(self.projector(h1), self.projector(h2))

    def student_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.parameters())
