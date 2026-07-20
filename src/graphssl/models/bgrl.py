"""BGRL: Bootstrapped Graph Representation Learning."""

from __future__ import annotations

import copy
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from graphssl.core.model import BaseSSLModel
from graphssl.config.schema import BGRLConfig
from graphssl.utils.ema import update_ema_params
from graphssl.utils.schedulers import CosineEMAScheduler
from graphssl.losses.regression import CosineRegressionLoss
from graphssl.nn.mlp import Predictor
from graphssl.nn.pooling import pool_graph_embeddings
from graphssl.augmentation.compose import compose


class BGRL(BaseSSLModel):
    """Teacher-student SSL with EMA target encoder and cosine regression loss.

    Key invariant: the target encoder is deepcopied then reset via
    reset_parameters(), giving it intentionally different initial weights.
    This asymmetry is critical for convergence (App. B, BGRL paper).

    Args:
        config: Config dict validated against BGRLConfig.
        in_channels: Node feature dimensionality, resolved from the dataset.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__()
        cfg = BGRLConfig.from_dict(config)
        self.online_enc = cfg.encoder.build(in_channels)
        hidden_dim = cfg.encoder.hidden_dim
        self.online_pred = Predictor(hidden_dim, cfg.pred_hidden, hidden_dim)

        # Target encoder: deepcopy + reset — intentionally different from online.
        self.target_enc = copy.deepcopy(self.online_enc)
        reset_fn = getattr(self.target_enc, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()
        for p in self.target_enc.parameters():
            p.requires_grad = False

        self._ema_tau: float = cfg.ema_tau
        self._ema_scheduler: Optional[CosineEMAScheduler] = (
            CosineEMAScheduler(cfg.ema_tau, cfg.ema_tau_end, cfg.total_steps)
            if cfg.total_steps > 0
            else None
        )
        self._step: int = 0
        self.graph_level: bool = cfg.encoder.pool
        self.loss_fn = CosineRegressionLoss(symmetric=True)
        self.aug_list: List[Tuple[str, dict]] = [(a.name, a.kwargs) for a in cfg.augment]

    def forward(self, data: Data) -> Tensor:
        with torch.no_grad():
            z = self.online_enc(data.x, data.edge_index, data.batch)
            if self.graph_level:
                z = pool_graph_embeddings(z, data.batch)
            return z.detach()

    def compute_loss(self, data: Data) -> Tensor:
        # Protect seed nodes from destructive augmentations in mini-batch training.
        protected: Optional[Tensor] = None
        batch_size: Optional[int] = None
        if not self.graph_level:
            batch_size = getattr(data, "batch_size", None)
            if batch_size is not None:
                _dev = data.x.device if data.x is not None else torch.device("cpu")
                protected = torch.arange(batch_size, device=_dev)

        v1 = compose(data, self.aug_list, protected_nodes=protected)
        v2 = compose(data, self.aug_list, protected_nodes=protected)

        z1 = self.online_enc(v1.x, v1.edge_index, v1.batch)
        z2 = self.online_enc(v2.x, v2.edge_index, v2.batch)

        with torch.no_grad():
            t1 = self.target_enc(v1.x, v1.edge_index, v1.batch)
            t2 = self.target_enc(v2.x, v2.edge_index, v2.batch)

        if self.graph_level:
            z1 = pool_graph_embeddings(z1, v1.batch)
            z2 = pool_graph_embeddings(z2, v2.batch)
            t1 = pool_graph_embeddings(t1, v1.batch)
            t2 = pool_graph_embeddings(t2, v2.batch)
        elif batch_size is not None:
            z1, z2 = z1[:batch_size], z2[:batch_size]
            t1, t2 = t1[:batch_size], t2[:batch_size]

        p1 = self.online_pred(z1)
        p2 = self.online_pred(z2)

        # CosineRegressionLoss(p1, t1, p2, t2) = 2 - cos(p1, t2) - cos(p2, t1)
        return self.loss_fn(p1, t1, p2, t2)

    def student_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.online_enc.parameters()
        yield from self.online_pred.parameters()

    def post_step(self) -> None:
        tau = self._ema_scheduler.get(self._step) if self._ema_scheduler else self._ema_tau
        update_ema_params(self.online_enc, self.target_enc, tau)
        self._step += 1
