from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


@runtime_checkable
class BaseEncoder(Protocol):
    """Structural interface for GNN backbone encoders.

    Using Protocol (not ABC) so existing nn.Module encoders conform without
    inheriting from this class — structural subtyping only.
    """

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor: ...

    def reset_parameters(self) -> None: ...


def encode(encoder: nn.Module, data: Data) -> Tensor:
    """Run ``encoder`` on a graph or a batch of graphs, with its edge features if it has any.

    Every model reaches its encoder through this function. Calling the encoder with
    ``(x, edge_index, batch)`` alone drops ``edge_attr``, and an edge-aware encoder then runs
    on zeros without complaining: bond types were ignored that way on molecules.
    """
    batch = getattr(data, "batch", None)
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        return encoder(data.x, data.edge_index, batch)
    return encoder(data.x, data.edge_index, batch, edge_attr=edge_attr)


def pretrained_encoder(model: nn.Module) -> nn.Module:
    """The encoder a model embeds with in ``forward()``: the one to reuse after pre-training.

    ``encoder`` for single-encoder models, the online encoder of BGRL / AFGRL, the teacher of
    GraphDINO. To fine-tune, copy its weights into a ``Supervised`` model built with the same
    encoder config: ``supervised.encoder.load_state_dict(pretrained_encoder(m).state_dict())``.
    """
    for attr in ("encoder", "online_enc", "teacher_enc"):
        encoder = getattr(model, attr, None)
        if isinstance(encoder, nn.Module):
            return encoder
    raise ValueError(f"{type(model).__name__} has no encoder / online_enc / teacher_enc")
