"""GIN encoder: GINConv or GINEConv (edge features) stack with configurable norm."""

from __future__ import annotations

from typing import cast

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GINEConv

from graphssl.registry import ENCODERS


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(dim)
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"norm_type must be 'batch', 'layer', or 'none', got {norm_type!r}")


class GINLayer(nn.Module):
    """GINConv (or GINEConv) with pre-norm, ReLU, dropout, and residual."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 2,
        drop: float = 0.2,
        norm_type: str = "batch",
        edge_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        mlp = Sequential(Linear(dim, hidden_dim), ReLU(), nn.Dropout(drop), Linear(hidden_dim, dim))

        self.norm = _make_norm(norm_type, dim)
        # GINEConv when edge features are present, plain GINConv otherwise.
        self.conv = GINEConv(nn=mlp, edge_dim=edge_dim) if edge_dim is not None else GINConv(nn=mlp)
        self.activation = ReLU()
        self.drop = nn.Dropout(drop)
        self.uses_edge_attr = edge_dim is not None

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        h = self.norm(x)
        if self.uses_edge_attr:
            # GINEConv requires a valid edge_attr tensor; fall back to zeros if absent.
            if edge_attr is None:
                edge_attr = x.new_zeros(edge_index.size(1), h.size(-1))
            h = self.conv(h, edge_index, edge_attr)
        else:
            h = self.conv(h, edge_index)
        h = self.activation(h)
        h = self.drop(h)
        return x + h

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        reset_fn = getattr(self.norm, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()


@ENCODERS.register("gin")
class GINEncoder(nn.Module):
    """GIN backbone with configurable norm, optional edge features, and categorical embeddings.

    Supports both continuous and categorical node/edge features out of the box,
    making it compatible with datasets like ZINC (categorical atoms + bond types)
    and ogbn-arxiv (continuous 128-dim embeddings).

    Args:
        in_channels: Input node feature dimension. Ignored when node_emb_num_classes is set.
        hidden_dim: Hidden dimension throughout the stack.
        out_dim: Final output dimension (defaults to hidden_dim).
        num_layers: Number of GINLayer blocks.
        mlp_ratio: Width multiplier for the inner MLP of each GINConv.
        drop: Dropout rate.
        pool: If True, apply global_add_pool to produce graph-level embeddings.
        norm_type: Normalisation after each layer — 'batch', 'layer', or 'none'.
        edge_dim: Projected edge feature dimension fed to GINEConv. Required when
            edge features are present. If None, edge_attr is ignored.
        node_emb_num_classes: If set, replaces the linear input projection with nn.Embedding
            (needed for integer node labels such as ZINC atom types, 28 classes).
        edge_emb_num_classes: If set, adds an nn.Embedding that maps integer edge labels
            to edge_dim-dimensional vectors before GINEConv. Required for ZINC bond types
            (4 classes). Needs edge_dim to be set.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int | None = None,
        num_layers: int = 3,
        mlp_ratio: int = 2,
        drop: float = 0.2,
        pool: bool = False,
        norm_type: str = "batch",
        edge_dim: int | None = None,
        node_emb_num_classes: int | None = None,
        edge_emb_num_classes: int | None = None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        if edge_emb_num_classes is not None and edge_dim is None:
            raise ValueError("edge_emb_num_classes requires edge_dim to be set.")
        self.pool = pool
        self._drop = drop
        self.edge_dim = edge_dim

        # --- Node input projection ---
        # Categorical (e.g. ZINC atom types)  → nn.Embedding
        # Continuous (e.g. ogbn-arxiv 128-d)  → nn.Linear
        if node_emb_num_classes is not None:
            self.input_proj: nn.Module = nn.Embedding(node_emb_num_classes, hidden_dim)
        else:
            self.input_proj = Linear(in_channels, hidden_dim)

        # --- Edge input projection ---
        # Categorical (e.g. ZINC bond types 0-3) → nn.Embedding → edge_dim vectors
        # Continuous edge features               → passed directly (user must ensure dim matches)
        # No edge features                        → None
        self.edge_proj: nn.Module | None = None
        if edge_emb_num_classes is not None:
            assert edge_dim is not None  # validated above: edge_emb_num_classes requires edge_dim
            self.edge_proj = nn.Embedding(edge_emb_num_classes, edge_dim)

        self.layers = nn.ModuleList(
            [
                GINLayer(
                    hidden_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    norm_type=norm_type,
                    edge_dim=edge_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.activation = ReLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        # Node features
        if isinstance(self.input_proj, nn.Embedding):
            x = self.input_proj(x.squeeze(-1).long())
        else:
            x = self.input_proj(x)

        # Edge features: embed categorical labels if needed
        if self.edge_proj is not None and edge_attr is not None:
            edge_attr = self.edge_proj(edge_attr.squeeze(-1).long())

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr if self.edge_dim is not None else None)

        x = self.activation(self.lin1(x))
        x = F.dropout(x, p=self._drop, training=self.training)
        x = self.lin2(x)
        return x

    def reset_parameters(self) -> None:
        input_reset = getattr(self.input_proj, "reset_parameters", None)
        if callable(input_reset):
            input_reset()
        edge_reset = getattr(self.edge_proj, "reset_parameters", None)
        if callable(edge_reset):
            edge_reset()
        for layer in self.layers:
            # self.layers is a plain nn.ModuleList (yields Module on iteration per its
            # stub), but every element is actually a GINLayer, which does define this.
            cast(GINLayer, layer).reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
