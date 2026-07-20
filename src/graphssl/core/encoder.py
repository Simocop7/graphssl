from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable
from torch import Tensor


@runtime_checkable
class BaseEncoder(Protocol):
    """Structural interface for GNN backbone encoders.

    Using Protocol (not ABC) so existing nn.Module encoders conform without
    inheriting from this class — structural subtyping only.
    """

    def forward(
        self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None
    ) -> Tensor: ...

    def reset_parameters(self) -> None: ...
