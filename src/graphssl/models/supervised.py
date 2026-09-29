"""Supervised GNN: encoder + linear classification head."""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from graphssl.config.schema import SupervisedConfig
from graphssl.core.model import BaseSSLModel
from graphssl.nn.pooling import pool_graph_embeddings


class Supervised(BaseSSLModel):
    """Encoder + linear head with CrossEntropyLoss.

    The labels the loss may see depend on the mode:

    - Graph-level (``encoder.pool=True``): node embeddings are mean-pooled per graph and
      every graph in the batch contributes.
    - Node-level mini-batch (``batch.batch_size`` set by ``NeighborLoader``): only the first
      ``batch_size`` seed nodes contribute. Build the loader with ``input_nodes=train_idx``.
    - Node-level full-batch: only ``data.train_mask`` nodes contribute. A full graph without
      ``train_mask`` raises rather than silently training on validation/test labels.

    Args:
        config: Config dict validated against SupervisedConfig.
        in_channels: Node feature dimensionality, resolved from the dataset.
        num_classes: Number of output classes, resolved from the dataset.
    """

    def __init__(self, config: Dict, in_channels: int, num_classes: int):
        super().__init__()
        cfg = SupervisedConfig.from_dict(config)
        self.encoder = cfg.encoder.build(in_channels)
        self.head = nn.Linear(cfg.encoder.hidden_dim, num_classes)
        self.graph_level: bool = cfg.encoder.pool

    def forward(self, data: Data) -> Tensor:
        z = self.encoder(data.x, data.edge_index, data.batch)
        if self.graph_level:
            z = pool_graph_embeddings(z, data.batch)
        return z

    def student_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.parameters())

    def compute_loss(self, data: Data) -> Tensor:
        logits = self.head(self.forward(data))
        y = data.y
        if not self.graph_level:
            batch_size: Optional[int] = getattr(data, "batch_size", None)
            if batch_size is not None:
                logits, y = logits[:batch_size], y[:batch_size]
            else:
                train_mask: Optional[Tensor] = getattr(data, "train_mask", None)
                if train_mask is None:
                    raise ValueError(
                        "Full-batch node-level Supervised training needs `data.train_mask`: "
                        "without it the loss would include validation/test labels."
                    )
                logits, y = logits[train_mask], y[train_mask]
        return F.cross_entropy(logits, y)
