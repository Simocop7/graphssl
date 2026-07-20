"""Supervised GNN: encoder + linear classification head."""

from __future__ import annotations

from typing import Dict, Iterator

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from graphssl.core.model import BaseSSLModel
from graphssl.config.schema import SupervisedConfig


class Supervised(BaseSSLModel):
    """Encoder + linear head with CrossEntropyLoss.

    Supports full-batch and mini-batch node training. In mini-batch mode
    (batch.batch_size is set by NeighborLoader), loss is computed only on
    the seed nodes.

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

    def forward(self, data: Data) -> Tensor:
        return self.encoder(data.x, data.edge_index, data.batch)

    def student_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.parameters())

    def compute_loss(self, data: Data) -> Tensor:
        emb = self.encoder(data.x, data.edge_index, data.batch)
        logits = self.head(emb)
        y = data.y
        if hasattr(data, "batch_size"):
            logits = logits[: data.batch_size]
            y = y[: data.batch_size]
        return F.cross_entropy(logits, y)
