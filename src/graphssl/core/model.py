from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


class BaseModel(nn.Module, ABC):
    """Base for all models. forward() returns embeddings only."""

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Extract node/graph embeddings from a batch."""


class BaseSSLModel(BaseModel):
    """Extension for self-supervised models.

    The Trainer loop is:
        model.on_epoch_start(epoch)     # teacher temp warmup, etc.
        for batch in loader:
            loss = model.compute_loss(batch)
            loss.backward()
            clip_grad_norm_(...)
            model.post_backward()       # freeze last layer, etc.
            optimizer.step()
            model.post_step()           # EMA, center updates, etc.
        model.on_epoch_end(epoch)       # epoch counter, etc.

    For evaluation:
        embeddings = model(batch)       # calls forward(), no grad needed
    """

    @abstractmethod
    def compute_loss(self, data: Data) -> Tensor:
        """Compute the SSL training loss. Augmentation is handled internally."""

    @abstractmethod
    def student_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters to optimize. Excludes frozen teacher/target parameters."""

    def post_backward(self) -> None:
        """Called after backward() and grad clipping, before optimizer.step().

        Override to cancel gradients on specific parameters (e.g. DINO last layer freeze).
        """

    def post_step(self) -> None:
        """Called after optimizer.step(). Override for EMA updates, center updates, etc."""

    def on_epoch_start(self, epoch: int) -> None:
        """Called by Trainer at the start of each epoch before any batch."""

    def on_epoch_end(self, epoch: int) -> None:
        """Called by Trainer at the end of each epoch after all batches."""
