"""Logistic regression linear probe evaluator in pure PyTorch."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from sklearn.metrics import average_precision_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class LogRegEvaluator:
    """Train a linear classifier on frozen embeddings and evaluate accuracy.

    For multilabel tasks (e.g. ogbg-molpcba), uses binary cross-entropy and
    reports average precision score instead of accuracy. Labels are expected
    as float tensors possibly containing NaN for missing values.

    Args:
        lr: Learning rate for the linear head.
        epochs: Number of training epochs.
        weight_decay: L2 regularisation.
        multilabel: If True, treats the task as multilabel classification.
    """

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 100,
        weight_decay: float = 0.0,
        multilabel: bool = False,
    ):
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.multilabel = multilabel

    def evaluate(
        self,
        embeddings: Tensor,
        labels: Tensor,
        train_idx: Tensor,
        val_idx: Tensor,
        test_idx: Tensor,
        num_classes: int,
    ) -> Dict[str, float]:
        device = embeddings.device
        in_dim = embeddings.size(1)
        out_dim = labels.size(1) if (self.multilabel and labels.dim() > 1) else num_classes

        classifier = nn.Linear(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        x_train = embeddings[train_idx]
        y_train = labels[train_idx].to(device)
        if not self.multilabel and y_train.dim() > 1:
            y_train = y_train.squeeze(-1)  # OGB node datasets store single-label y as [N, 1]

        classifier.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = classifier(x_train)
            if self.multilabel:
                nan_mask = ~torch.isnan(y_train)
                loss = F.binary_cross_entropy_with_logits(
                    logits[nan_mask], y_train[nan_mask].float()
                )
            else:
                loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        classifier.eval()
        results: Dict[str, float] = {}
        with torch.no_grad():
            for split, idx in [("val", val_idx), ("test", test_idx)]:
                x_s = embeddings[idx]
                y_s = labels[idx].to(device)
                if not self.multilabel and y_s.dim() > 1:
                    y_s = y_s.squeeze(-1)
                logits = classifier(x_s)

                if self.multilabel:
                    if HAS_SKLEARN:
                        import numpy as np

                        probs = torch.sigmoid(logits).cpu().numpy()
                        y_np = y_s.cpu().numpy()
                        nan_mask = ~torch.isnan(y_s).cpu().numpy()
                        ap_scores = []
                        for i in range(out_dim):
                            col = nan_mask[:, i]
                            if col.sum() > 0:
                                ap_scores.append(
                                    average_precision_score(y_np[col, i], probs[col, i])
                                )
                        results[f"{split}_ap"] = (
                            float(np.mean(ap_scores)) if ap_scores else float("nan")
                        )
                    else:
                        results[f"{split}_ap"] = float("nan")
                else:
                    preds = logits.argmax(dim=-1)
                    results[f"{split}_acc"] = (preds == y_s).float().mean().item()

        return results
