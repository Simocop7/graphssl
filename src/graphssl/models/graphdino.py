"""GraphDINO: DINO-style self-distillation for graph-level SSL."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, cast

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from graphssl.augmentation import compose
from graphssl.config.schema import GraphDINOConfig, HeadConfig
from graphssl.core.encoder import encode
from graphssl.core.model import BaseSSLModel
from graphssl.losses.dino import DINOLoss
from graphssl.nn.dino_head import DINOHead
from graphssl.nn.pooling import pool_graph_embeddings
from graphssl.registry import HEADS
from graphssl.utils import update_ema_params
from graphssl.utils.schedulers import CosineEMAScheduler


class GraphDINO(BaseSSLModel):
    """
    Args:
        config: Config dict (see graphdino.yaml); validated against GraphDINOConfig.
        in_channels: Node feature size, resolved from the dataset at runtime.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__()
        cfg = GraphDINOConfig.from_dict(config)

        self.student_enc = cfg.encoder.build(in_channels)
        # _build_head() returns nn.Module (it goes through the HEADS registry, which is
        # deliberately generic — see registry pattern). GraphDINO itself only ever
        # requests the "dino" head, so this cast documents a real, narrower invariant
        # rather than papering over one; it's what unlocks DINOHead-specific access
        # below (.center, .set_epoch(), .update_center(), .cancel_last_layer_gradients()).
        self.student_head = cast(DINOHead, self._build_head(cfg.head, cfg.encoder.hidden_dim))

        # Build fresh instances rather than deepcopy: weight_norm creates non-leaf
        # tensors that break copy.deepcopy.
        self.teacher_enc = cfg.encoder.build(in_channels)
        self.teacher_head = cast(DINOHead, self._build_head(cfg.head, cfg.encoder.hidden_dim))
        self.teacher_enc.load_state_dict(self.student_enc.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_enc.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.aug_list_teacher: List[Tuple[str, dict]] = [
            (a.name, a.kwargs) for a in cfg.augment_teacher
        ]
        self.aug_list_student: List[Tuple[str, dict]] = [
            (a.name, a.kwargs) for a in cfg.augment_student
        ]
        self._n_views: int = cfg.n_views
        self._n_global_views: int = cfg.n_global_views
        # pool=True means the encoder returns graph-level embeddings; no batch_size crop needed
        self._graph_level: bool = cfg.encoder.pool
        self.readout: str = cfg.encoder.readout

        self._ema_tau: float = cfg.ema_tau_base
        self._loss_fn = DINOLoss()

        self._freeze_last_layer_epochs: int = cfg.freeze_last_layer_epochs
        self._epoch: int = 0

        if cfg.total_steps > 0 and cfg.ema_tau_base < cfg.ema_tau:
            self._ema_scheduler: CosineEMAScheduler | None = CosineEMAScheduler(
                ema_base=cfg.ema_tau_base,
                ema_end=cfg.ema_tau,
                total_steps=cfg.total_steps,
            )
        else:
            self._ema_scheduler = None
        self._step: int = 0

        self._last_teacher_out: Tensor | None = None
        self._last_teacher_logits: Tensor | None = None

    @property
    def last_teacher_out(self) -> Tensor | None:
        return self._last_teacher_out

    @staticmethod
    def _build_head(head_cfg: HeadConfig, hidden_dim: int) -> nn.Module:
        kwargs = {k: v for k, v in vars(head_cfg).items() if k != "name"}
        return HEADS.build(head_cfg.name, hidden_dim=hidden_dim, **kwargs)

    def forward(self, data: Data) -> Tensor:
        with torch.no_grad():
            z = encode(self.teacher_enc, data)
            if self._graph_level:
                z = pool_graph_embeddings(z, data.batch, self.readout)
            return z.detach()

    def _embed(self, enc: nn.Module, view: Data, batch_size: Optional[int]) -> Tensor:
        h = encode(enc, view)
        if self._graph_level:
            h = pool_graph_embeddings(h, view.batch, self.readout)
        elif batch_size is not None:
            h = h[:batch_size]
        return h

    def compute_loss(self, data: Data) -> Tensor:
        # In mini-batch node training, protect the first batch_size seed nodes
        # from destructive augmentations (e.g. node_drop).
        batch_size: Optional[int] = getattr(data, "batch_size", None)
        if self._graph_level:
            batch_size = None
        protected: Optional[Tensor] = None
        if batch_size is not None:
            _dev = data.x.device if data.x is not None else torch.device("cpu")
            protected = torch.arange(batch_size, device=_dev)

        # Global views: weak augmentation → fed to teacher and student.
        global_views = [
            compose(data, self.aug_list_teacher, protected_nodes=protected)
            for _ in range(self._n_global_views)
        ]
        # Local views: strong augmentation → fed to student only.
        local_views = [
            compose(data, self.aug_list_student, protected_nodes=protected)
            for _ in range(self._n_views - self._n_global_views)
        ]
        all_views = global_views + local_views

        # Head called once per view, so its BatchNorm sees per-view statistics.
        student_out = torch.cat(
            [self.student_head(self._embed(self.student_enc, v, batch_size)) for v in all_views],
            dim=0,
        )

        with torch.no_grad():
            # Raw teacher logits: the center is subtracted from them here and updated
            # with them in post_step(), so both happen in logit space.
            teacher_logits = torch.cat(
                [
                    self.teacher_head.prototype_logits(self._embed(self.teacher_enc, v, batch_size))
                    for v in global_views
                ],
                dim=0,
            )
            teacher_out = self.teacher_head.teacher_probs(teacher_logits)
        self._last_teacher_logits = teacher_logits
        self._last_teacher_out = teacher_out

        return self._loss_fn(student_out, teacher_out, self._n_global_views, self._n_views)

    def student_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.student_enc.parameters()
        yield from self.student_head.parameters()

    def post_backward(self) -> None:
        if self._epoch < self._freeze_last_layer_epochs:
            self.student_head.cancel_last_layer_gradients()

    def post_step(self) -> None:
        if self._ema_scheduler is not None:
            self._ema_tau = self._ema_scheduler.get(self._step)
        self._step += 1

        update_ema_params(self.student_enc, self.teacher_enc, self._ema_tau)
        # Sync student center = teacher center before buffer copy so the copy
        # is a no-op for the center buffer (ema.py copies buffers directly).
        # This preserves the teacher center's history through the EMA sync.
        self.student_head.center.copy_(self.teacher_head.center)
        update_ema_params(self.student_head, self.teacher_head, self._ema_tau)
        if self._last_teacher_logits is not None:
            self.teacher_head.update_center(self._last_teacher_logits)

    def on_epoch_start(self, epoch: int) -> None:
        self.teacher_head.set_epoch(epoch)

    def on_epoch_end(self, epoch: int) -> None:
        self._epoch = epoch + 1
