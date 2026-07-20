"""AFGRL: Augmentation-Free Graph Representation Learning."""

from __future__ import annotations

import copy
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from graphssl.core.model import BaseSSLModel
from graphssl.config.schema import AFGRLConfig
from graphssl.utils.ema import update_ema_params
from graphssl.utils.schedulers import CosineEMAScheduler
from graphssl.utils.positive_miner import PositiveMiner
from graphssl.losses.regression import CosineRegressionLoss
from graphssl.nn.mlp import Predictor
from graphssl.nn.pooling import pool_graph_embeddings


class AFGRL(BaseSSLModel):
    """No-augmentation SSL: positive pairs mined via local adjacency and global k-means.

    Identical to BGRL in the student/teacher setup, but replaces augmented views
    with node pairs identified as positives through:
      - Local:  top-k cosine-similarity neighbours that are also graph-adjacent (kNN ∩ adj).
      - Global: top-k neighbours sharing a k-means cluster in ANY of the independent runs.

    Key difference from BGRL: the target encoder starts with the same weights as the
    online encoder (deepcopy without reset). Diversity comes from the miner, not
    from weight asymmetry.

    Graph-level mode: the PositiveMiner is topology-aware and not meaningful across
    independent graphs. In this mode it is skipped; a simple teacher-student loss on
    pooled embeddings is used instead.

    Requires faiss-cpu: pip install faiss-cpu

    Args:
        config: Config dict validated against AFGRLConfig.
        in_channels: Node feature dimensionality, resolved from the dataset.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__()
        cfg = AFGRLConfig.from_dict(config)
        self.online_enc = cfg.encoder.build(in_channels)
        hidden_dim = cfg.encoder.hidden_dim
        self.online_pred = Predictor(hidden_dim, cfg.pred_hidden, hidden_dim)

        # Target encoder: same initial weights as online (no reset — unlike BGRL).
        self.target_enc = copy.deepcopy(self.online_enc)
        for p in self.target_enc.parameters():
            p.requires_grad = False

        self._ema_tau: float = cfg.ema_tau
        self._ema_scheduler: Optional[CosineEMAScheduler] = (
            CosineEMAScheduler(cfg.ema_tau, cfg.ema_tau_end, cfg.total_steps)
            if cfg.total_steps > 0
            else None
        )
        self._step: int = 0
        self._topk: int = cfg.topk
        self.graph_level: bool = cfg.encoder.pool
        self._graph_loss = CosineRegressionLoss(symmetric=False)
        self.positive_miner = PositiveMiner(
            num_centroids=cfg.num_centroids,
            num_kmeans=cfg.num_kmeans,
            clus_num_iters=cfg.clus_num_iters,
        )

    def forward(self, data: Data) -> Tensor:
        with torch.no_grad():
            z = self.online_enc(data.x, data.edge_index, data.batch)
            if self.graph_level:
                z = pool_graph_embeddings(z, data.batch)
            return z.detach()

    def compute_loss(self, data: Data) -> Tensor:
        z_online = self.online_enc(data.x, data.edge_index, data.batch)

        # Graph-level: PositiveMiner doesn't generalise across graphs — use simple loss.
        if self.graph_level:
            z = pool_graph_embeddings(z_online, data.batch)
            p = self.online_pred(z)
            with torch.no_grad():
                z_target = pool_graph_embeddings(
                    self.target_enc(data.x, data.edge_index, data.batch), data.batch
                )
            return self._graph_loss(p, z_target.detach())

        p_online = self.online_pred(z_online)

        with torch.no_grad():
            z_target = self.target_enc(data.x, data.edge_index, data.batch)

        # Build sparse adjacency for local positive mining (kNN ∩ adj).
        edge_attr = getattr(data, "edge_attr", None)
        adj_vals = (
            edge_attr.float()
            if edge_attr is not None
            else torch.ones(data.edge_index.shape[1], device=data.x.device)
        )
        n = data.x.shape[0]
        adj = torch.sparse_coo_tensor(data.edge_index, adj_vals, (n, n))

        src, dst = self.positive_miner.mine(
            adj,
            F.normalize(z_online.detach(), dim=-1),
            F.normalize(z_target.detach(), dim=-1),
            self._topk,
        )

        # Symmetric regression on mined positive pairs.
        p_src = F.normalize(p_online[src], dim=-1)
        p_dst = F.normalize(p_online[dst], dim=-1)
        t_src = F.normalize(z_target[src].detach(), dim=-1)
        t_dst = F.normalize(z_target[dst].detach(), dim=-1)
        loss1 = 2 - 2 * (p_src * t_dst).sum(dim=-1)
        loss2 = 2 - 2 * (p_dst * t_src).sum(dim=-1)
        return (loss1 + loss2).mean()

    def student_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.online_enc.parameters()
        yield from self.online_pred.parameters()

    def post_step(self) -> None:
        tau = self._ema_scheduler.get(self._step) if self._ema_scheduler else self._ema_tau
        update_ema_params(self.online_enc, self.target_enc, tau)
        self._step += 1
