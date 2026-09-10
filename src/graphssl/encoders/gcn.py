"""GCN encoder: GCNConv stack with configurable norm, PReLU, weight standardization."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv

from graphssl.nn.norm import apply_weight_standardization
from graphssl.registry import ENCODERS


def _make_norm(norm_type: str, dim: int, batchnorm_mm: float = 0.01) -> nn.Module:
    """Create a normalisation layer by name, aligned with the GIN/Transformer API."""
    if norm_type == "batch":
        return nn.BatchNorm1d(dim, momentum=batchnorm_mm)
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"norm_type must be 'batch', 'layer', or 'none', got {norm_type!r}")


@ENCODERS.register("gcn")
class GCNEncoder(nn.Module):
    """2-layer GCNConv backbone.

    Architecture per layer: GCNConv → Norm → PReLU.

    Uses the same ``norm_type`` API as GINEncoder and TransformerEncoder so that
    switching encoder backbones is a one-line config change.

    Args:
        in_channels: Input node feature dimension.
        hidden_dim: Hidden (and output) dimension for layer 1.
        out_dim: Output dimension for layer 2 (defaults to hidden_dim).
        norm_type: 'batch', 'layer', or 'none' — same API as GINEncoder.
        weight_standardization: Standardize GCNConv weights before layer 2 forward.
        batchnorm_mm: BatchNorm momentum (standard PyTorch convention; default 0.01).
        pool: If True, apply global_mean_pool to produce graph-level embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int | None = None,
        norm_type: str = "batch",
        weight_standardization: bool = False,
        batchnorm_mm: float = 0.01,
        pool: bool = False,
        # Legacy aliases kept for backwards compatibility — map to norm_type internally.
        batchnorm: bool | None = None,
        layernorm: bool | None = None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim

        # --- Backwards-compatible norm resolution ---
        # If the caller uses the old batchnorm/layernorm bool flags, honour them.
        if batchnorm is not None or layernorm is not None:
            if batchnorm and layernorm:
                raise ValueError("batchnorm and layernorm are mutually exclusive.")
            if batchnorm:
                norm_type = "batch"
            elif layernorm:
                norm_type = "layer"
            else:
                norm_type = "none"

        self.weight_standardization = weight_standardization
        self.pool = pool

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.norm1 = _make_norm(norm_type, hidden_dim, batchnorm_mm)
        self.norm2 = _make_norm(norm_type, out_dim, batchnorm_mm)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x, edge_index)))

        if self.weight_standardization:
            apply_weight_standardization(self.conv2)
        x = self.act2(self.norm2(self.conv2(x, edge_index)))

        return x

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for norm in (self.norm1, self.norm2):
            reset_fn = getattr(norm, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()
        self.act1 = nn.PReLU().to(next(self.parameters()).device)
        self.act2 = nn.PReLU().to(next(self.parameters()).device)
