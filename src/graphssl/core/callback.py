from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data


class Callback:
    """Base class for Trainer callbacks with no-op defaults.

    Subclass and override only the hooks you need.
    All methods receive the trainer instance as first argument so callbacks
    can access trainer.device, trainer.optimizer, etc. without coupling.
    """

    def on_train_start(self, trainer: Any, model: torch.nn.Module) -> None:
        pass

    def on_epoch_start(self, trainer: Any, model: torch.nn.Module, epoch: int) -> None:
        pass

    def on_batch_end(
        self,
        trainer: Any,
        model: torch.nn.Module,
        loss: torch.Tensor,
        batch: "Data",
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        model: torch.nn.Module,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        pass

    def on_train_end(self, trainer: Any, model: torch.nn.Module) -> None:
        pass
