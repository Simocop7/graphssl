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
    """GCNConv backbone (2 layers by default).

    Architecture per layer: GCNConv → Norm → PReLU.

    Uses the same ``norm_type`` API as GINEncoder and TransformerEncoder so that
    switching encoder backbones is a one-line config change.

    Args:
        in_channels: Input node feature dimension.
        hidden_dim: Output dimension of every layer but the last.
        out_dim: Output dimension of the last layer (defaults to hidden_dim).
        norm_type: 'batch', 'layer', or 'none' — same API as GINEncoder.
        weight_standardization: Standardize the GCNConv weights of every layer after the
            first, before each forward (BGRL's setting on ogbn-arxiv, with layer norm).
        batchnorm_mm: BatchNorm momentum (standard PyTorch convention; default 0.01).
        pool: If True, apply global_mean_pool to produce graph-level embeddings.
        num_layers: Number of GCNConv layers.
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
        num_layers: int = 2,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

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

        dims = [in_channels] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(GCNConv(i, o) for i, o in zip(dims[:-1], dims[1:], strict=True))
        self.norms = nn.ModuleList(_make_norm(norm_type, d, batchnorm_mm) for d in dims[1:])
        self.acts = nn.ModuleList(nn.PReLU() for _ in dims[1:])

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        layers = zip(self.convs, self.norms, self.acts, strict=True)
        for i, (conv, norm, act) in enumerate(layers):
            if self.weight_standardization and i > 0:
                apply_weight_standardization(conv)
            x = act(norm(conv(x, edge_index)))
        return x

    def reset_parameters(self) -> None:
        for module in (*self.convs, *self.norms):
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()
        device = next(self.parameters()).device
        self.acts = nn.ModuleList(nn.PReLU() for _ in self.convs).to(device)
