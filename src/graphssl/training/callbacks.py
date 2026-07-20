"""Built-in Trainer callbacks."""

from __future__ import annotations
import os
from typing import Any, Dict
import torch
import torch.nn as nn

from graphssl.core.callback import Callback


class EmbeddingLoggerCallback(Callback):
    """Save embeddings (and labels) to disk at regular intervals.

    Files are written as .pt dicts with keys 'embeddings' and 'labels'.

    Args:
        datamodule: DataModule used to extract embeddings.
        save_dir: Directory where .pt files are written.
        every_n_epochs: Save frequency.
        device: Inference device.
    """

    def __init__(
        self,
        datamodule,
        save_dir: str = "embeddings",
        every_n_epochs: int = 10,
        device: str = "cpu",
    ):
        self.datamodule = datamodule
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.device = device

    def on_epoch_end(
        self, trainer: Any, model: nn.Module, epoch: int, metrics: Dict[str, Any]
    ) -> None:
        if epoch % self.every_n_epochs != 0:
            return
        from graphssl.evaluation.visualization import extract_embeddings
        os.makedirs(self.save_dir, exist_ok=True)
        embeddings, labels = extract_embeddings(model, self.datamodule, device=self.device)
        path = os.path.join(self.save_dir, f"epoch_{epoch:04d}.pt")
        torch.save({"embeddings": embeddings, "labels": labels}, path)


class LinearEvalCallback(Callback):
    """Run a linear probe evaluation at regular intervals and print results.

    Args:
        datamodule: DataModule with train/val/test splits and labels.
        num_classes: Number of classes for the linear head.
        every_n_epochs: Evaluation frequency.
        device: Inference device.
        multilabel: Use binary CE + average precision (e.g. ogbg-molpcba).
    """

    def __init__(
        self,
        datamodule,
        num_classes: int,
        every_n_epochs: int = 10,
        device: str = "cpu",
        multilabel: bool = False,
    ):
        self.datamodule = datamodule
        self.num_classes = num_classes
        self.every_n_epochs = every_n_epochs
        self.device = device
        self.multilabel = multilabel
        self._history: list[Dict[str, Any]] = []

    @property
    def results_history(self) -> list[Dict[str, Any]]:
        return self._history

    def on_epoch_end(
        self, trainer: Any, model: nn.Module, epoch: int, metrics: Dict[str, Any]
    ) -> None:
        if epoch % self.every_n_epochs != 0:
            return
        from graphssl.evaluation.visualization import extract_embeddings
        from graphssl.evaluation.linear_probe import LogRegEvaluator

        embeddings, labels = extract_embeddings(model, self.datamodule, device=self.device)
        evaluator = LogRegEvaluator(multilabel=self.multilabel)
        results = evaluator.evaluate(
            embeddings.to(self.device),
            labels,
            self.datamodule.train_idx,
            self.datamodule.val_idx,
            self.datamodule.test_idx,
            self.num_classes,
        )
        results["epoch"] = epoch
        self._history.append(results)
        print(f"[LinearEval epoch={epoch}] {results}")


class VisualizationCallback(Callback):
    """Save UMAP scatter plots at regular intervals.

    Requires optional deps: pip install umap-learn matplotlib

    Args:
        datamodule: DataModule used to extract embeddings.
        save_dir: Directory where PNG files are written.
        every_n_epochs: Plot frequency.
        device: Inference device.
    """

    def __init__(
        self,
        datamodule,
        save_dir: str = "plots",
        every_n_epochs: int = 50,
        device: str = "cpu",
    ):
        self.datamodule = datamodule
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.device = device

    def on_epoch_end(
        self, trainer: Any, model: nn.Module, epoch: int, metrics: Dict[str, Any]
    ) -> None:
        if epoch % self.every_n_epochs != 0:
            return
        from graphssl.evaluation.visualization import extract_embeddings, plot_embeddings
        os.makedirs(self.save_dir, exist_ok=True)
        embeddings, labels = extract_embeddings(model, self.datamodule, device=self.device)
        path = os.path.join(self.save_dir, f"epoch_{epoch:04d}.png")
        plot_embeddings(embeddings, labels, save_path=path, title=f"Epoch {epoch}")
