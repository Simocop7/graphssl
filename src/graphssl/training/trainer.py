"""Trainer: orchestrates SSL training epochs with optional callbacks."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from graphssl.core.callback import Callback
from graphssl.core.model import BaseSSLModel


class DINOTrainer:
    """Stateless trainer compatible with any BaseSSLModel.

    Per-batch order: compute_loss → backward → clip_grad → post_backward → optimizer.step → post_step
    Per-epoch order: on_epoch_start → batches → on_epoch_end

    Args:
        grad_clip_norm: Clip student gradient norm. ~3.0 stabilises early DINO.
        device: Default device used in train() when none is given per call.
        callbacks: List of Callback instances invoked at epoch/batch boundaries.
    """

    def __init__(
        self,
        grad_clip_norm: Optional[float] = 3.0,
        device: torch.device | str = "cpu",
        callbacks: Optional[List[Callback]] = None,
    ):
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device(device)
        self.callbacks: List[Callback] = callbacks or []

    def train_epoch(
        self,
        model: BaseSSLModel,
        loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device | str | None = None,
        epoch: int = 0,
    ) -> float:
        """Run one training epoch and return the average loss."""
        _device = torch.device(device) if device is not None else self.device
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(_device)
            loss = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.student_parameters()),
                    self.grad_clip_norm,
                )

            model.post_backward()
            optimizer.step()
            model.post_step()

            for cb in self.callbacks:
                cb.on_batch_end(self, model, loss, batch)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        model: BaseSSLModel,
        loader: DataLoader,
        optimizer: Optimizer,
        num_epochs: int,
        device: torch.device | str | None = None,
    ) -> List[float]:
        """Train for num_epochs epochs, fire callbacks, return per-epoch losses."""
        _device = torch.device(device) if device is not None else self.device
        model = model.to(_device)

        for cb in self.callbacks:
            cb.on_train_start(self, model)

        losses: List[float] = []
        for epoch in range(num_epochs):
            model.on_epoch_start(epoch)
            for cb in self.callbacks:
                cb.on_epoch_start(self, model, epoch)

            avg_loss = self.train_epoch(model, loader, optimizer, device=_device, epoch=epoch)
            losses.append(avg_loss)
            metrics = {"loss": avg_loss}

            model.on_epoch_end(epoch)
            for cb in self.callbacks:
                cb.on_epoch_end(self, model, epoch, metrics)

        for cb in self.callbacks:
            cb.on_train_end(self, model)

        return losses
