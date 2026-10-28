"""One-head GAT layer that aggregates before its linear map: GraphMAE's decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter, softmax


class GATDecoder(nn.Module):
    """The function of ``torch_geometric.nn.GATConv(in_channels, out_channels, heads=1)``,
    computed in the input dimension.

    With one head, GAT's output at node i is ``W · Σ_j α_ij h_j + b`` and its attention logits
    ``a_srcᵀ W h_j + a_dstᵀ W h_i`` are linear in ``h``: the linear map can follow the
    aggregation instead of preceding it. ``GATConv`` maps first and passes ``out_channels``-wide
    messages along every edge; a decoder that rebuilds the input features has out_channels =
    the feature dimension (6,805 on Coauthor CS: about 5 GB of messages per step). Here the
    messages have ``in_channels`` dimensions. Same parameters, initialization, self-loops
    (existing ones replaced by one per node), leaky-ReLU slope and softmax over each node's
    incoming edges as ``GATConv``; ``tests/test_graphmae.py`` checks the outputs and gradients
    against it.
    """

    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Glorot uniform, as GATConv's weight and attention vectors; zero bias.
        for p, (fan_in, fan_out) in (
            (self.lin.weight, self.lin.weight.shape[::-1]),
            (self.att_src, self.att_src.shape),
            (self.att_dst, self.att_dst.shape),
        ):
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(p, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n)
        src, dst = edge_index
        weight_t = self.lin.weight.t()  # [in, out]
        score_src = x @ (weight_t @ self.att_src[0])
        score_dst = x @ (weight_t @ self.att_dst[0])
        alpha = F.leaky_relu(score_src[src] + score_dst[dst], self.negative_slope)
        alpha = softmax(alpha, dst, num_nodes=n)
        aggregated = scatter(x[src] * alpha.unsqueeze(-1), dst, dim=0, dim_size=n, reduce="sum")
        return self.lin(aggregated) + self.bias
