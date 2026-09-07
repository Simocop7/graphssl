"""Graph Transformer encoder: stack of TransformerConv blocks with pre-norm."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv

from graphssl.registry import ENCODERS


def _make_norm(name: str, dim: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm1d(dim)
    if name == "layer":
        return nn.LayerNorm(dim)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm layer: {name!r}. Choose 'batch', 'layer', or 'none'.")


def _make_act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name!r}. Choose 'gelu' or 'relu'.")


class TransformerBlock(nn.Module):
    """Pre-norm TransformerConv + FFN block with residual connections.

    Supports optional edge features: when edge_dim is set, TransformerConv
    conditions the attention scores on edge attributes (edge_attr).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        attn_dropout: float,
        mlp_ratio: float,
        concat: bool,
        norm_type: str,
        act_layer_name: str,
        edge_dim: int | None = None,
    ):
        super().__init__()
        # concat=True: TransformerConv outputs heads * out_channels.
        # We keep the total dim constant, so out_channels = dim // heads.
        attn_out_channels = dim // heads if concat else dim

        self.norm1 = _make_norm(norm_type, dim)
        self.attn = TransformerConv(
            dim,
            attn_out_channels,
            heads=heads,
            dropout=attn_dropout,
            concat=concat,
            edge_dim=edge_dim,  # None → standard attention; set → edge-conditioned attention
        )

        ffn_hidden = int(dim * mlp_ratio)
        self.norm2 = _make_norm(norm_type, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            _make_act(act_layer_name),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout),
        )
        self.uses_edge_attr = edge_dim is not None

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        h = self.attn(h, edge_index, edge_attr) if self.uses_edge_attr else self.attn(h, edge_index)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        for m in self.ffn.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for norm in (self.norm1, self.norm2):
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()


@ENCODERS.register("transformer")
class TransformerEncoder(nn.Module):
    """Stack of pre-norm TransformerConv blocks.

    Aligns with the GINEncoder API: same norm_type values ('batch', 'layer',
    'none'), same edge_dim / node_emb_num_classes / edge_emb_num_classes
    parameters for seamless config swap between encoder types.

    Args:
        in_channels: Input node feature size. Ignored when node_emb_num_classes is set.
        hidden_dim: Uniform hidden/output dimension throughout the stack.
        num_layers: Number of TransformerBlock layers.
        heads: Number of attention heads (hidden_dim must be divisible by heads).
        dropout: Dropout in the FFN.
        attn_dropout: Dropout inside TransformerConv attention.
        mlp_ratio: FFN hidden dim = hidden_dim * mlp_ratio.
        concat: If True, TransformerConv concatenates heads (see TransformerBlock).
        norm_type: 'batch', 'layer', or 'none' — aligned with GINEncoder API.
        act_layer_name: 'gelu' or 'relu'.
        pool: Apply global_mean_pool to produce graph-level embeddings.
        edge_dim: Projected edge feature dimension. When set, each TransformerBlock
            conditions attention on edge_attr. Ignored if None.
        node_emb_num_classes: If set, replaces the linear input projection with
            nn.Embedding (for integer node labels, e.g. ZINC atom types).
        edge_emb_num_classes: If set, adds an nn.Embedding that maps integer edge
            labels to edge_dim vectors before attention. Requires edge_dim to be set.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        concat: bool = True,
        norm_type: str = "batch",
        act_layer_name: str = "gelu",
        pool: bool = True,
        edge_dim: int | None = None,
        node_emb_num_classes: int | None = None,
        edge_emb_num_classes: int | None = None,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        if edge_emb_num_classes is not None and edge_dim is None:
            raise ValueError("edge_emb_num_classes requires edge_dim to be set.")

        self.pool = pool
        self.edge_dim = edge_dim

        # --- Node input projection ---
        if node_emb_num_classes is not None:
            self.input_proj: nn.Module = nn.Embedding(node_emb_num_classes, hidden_dim)
        else:
            self.input_proj = nn.Linear(in_channels, hidden_dim)

        # --- Edge input projection ---
        self.edge_proj: nn.Module | None = (
            nn.Embedding(edge_emb_num_classes, edge_dim)
            if edge_emb_num_classes is not None
            else None
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    mlp_ratio=mlp_ratio,
                    concat=concat,
                    norm_type=norm_type,
                    act_layer_name=act_layer_name,
                    edge_dim=edge_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = _make_norm(norm_type, hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        # Node projection
        if isinstance(self.input_proj, nn.Embedding):
            x = self.input_proj(x.squeeze(-1).long())
        else:
            x = self.input_proj(x)

        # Edge projection
        if self.edge_proj is not None and edge_attr is not None:
            edge_attr = self.edge_proj(edge_attr.squeeze(-1).long())

        for block in self.blocks:
            x = block(x, edge_index, edge_attr if self.edge_dim is not None else None)
        x = self.norm(x)
        return x

    def reset_parameters(self) -> None:
        if hasattr(self.input_proj, "reset_parameters"):
            self.input_proj.reset_parameters()
        if self.edge_proj is not None and hasattr(self.edge_proj, "reset_parameters"):
            self.edge_proj.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()
