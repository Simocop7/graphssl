"""Logistic regression linear probe evaluator in pure PyTorch."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from sklearn.metrics import average_precision_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# L2 strengths tried by default. Few labels (Planetoid: 60-140) pick the strong end, many
# labels (ogbn-arxiv: 91k) the weak end.
DEFAULT_WEIGHT_DECAYS: Tuple[float, ...] = (10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


class LogRegEvaluator:
    """L2-regularised logistic regression on frozen embeddings.

    The classifier minimises ``mean cross-entropy + weight_decay / 2 * ||W||²`` on the
    training rows, full-batch with L-BFGS, starting from zero: no random initialisation
    and no learning rate, so the same embeddings always give the same result. When several
    ``weight_decay`` values are given, the one with the best validation metric is used
    (the strongest on ties) and the test metric is reported for it. The values are fitted
    from the strongest to the weakest, each starting from the previous solution.

    For multilabel tasks (e.g. ogbg-molpcba), uses binary cross-entropy and
    reports average precision score instead of accuracy. Labels are expected
    as float tensors possibly containing NaN for missing values.

    Args:
        weight_decay: L2 strength, or the values to select among on the validation split.
        max_iter: Budget of L-BFGS iterations and of loss evaluations for each fit.
            ``converged`` in the result is False when the selected fit used it up.
        standardize: Rescale every feature to zero mean and unit variance, with the
            statistics of the training rows only.
        multilabel: If True, treats the task as multilabel classification.

    ``evaluate()`` returns ``val_acc`` / ``test_acc`` (``val_ap`` / ``test_ap`` when
    multilabel), the selected ``weight_decay`` and ``converged``.
    """

    def __init__(
        self,
        *,
        weight_decay: Union[float, Sequence[float]] = DEFAULT_WEIGHT_DECAYS,
        max_iter: int = 5000,
        standardize: bool = True,
        multilabel: bool = False,
    ):
        values = [weight_decay] if isinstance(weight_decay, (int, float)) else list(weight_decay)
        if not values or any(v < 0 for v in values):
            raise ValueError(f"weight_decay must be one or more values >= 0, got {weight_decay!r}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        self.weight_decays: Tuple[float, ...] = tuple(
            sorted({float(v) for v in values}, reverse=True)
        )
        self.max_iter = max_iter
        self.standardize = standardize
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
        if self.multilabel and not HAS_SKLEARN and len(self.weight_decays) > 1:
            raise ImportError(
                "Selecting weight_decay on a multilabel task needs average precision: "
                "install scikit-learn, or pass a single weight_decay."
            )
        device = embeddings.device
        out_dim = labels.size(1) if (self.multilabel and labels.dim() > 1) else num_classes

        x: Dict[str, Tensor] = {}
        y: Dict[str, Tensor] = {}
        for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            x[split] = embeddings[idx]
            y_s = labels[idx].to(device)
            if not self.multilabel and y_s.dim() > 1:
                y_s = y_s.squeeze(-1)  # OGB node datasets store single-label y as [N, 1]
            y[split] = y_s
        if self.standardize:
            mean = x["train"].mean(dim=0, keepdim=True)
            std = x["train"].std(dim=0, keepdim=True)
            std = torch.where(std > 0, std, torch.ones_like(std))  # constant feature: centre only
            x = {split: (x_s - mean) / std for split, x_s in x.items()}

        classifier = nn.Linear(x["train"].size(1), out_dim).to(
            device=device, dtype=x["train"].dtype
        )
        nn.init.zeros_(classifier.weight)
        nn.init.zeros_(classifier.bias)

        metric = "ap" if self.multilabel else "acc"
        results: Dict[str, float] = {}
        for weight_decay in self.weight_decays:
            converged = self._fit(classifier, x["train"], y["train"], weight_decay)
            with torch.no_grad():
                val = self._score(classifier(x["val"]), y["val"])
                # Strict '>' keeps the stronger regularisation on ties (and the only value
                # when the metric is NaN: multilabel without scikit-learn).
                if not results or val > results[f"val_{metric}"]:
                    results = {
                        f"val_{metric}": val,
                        f"test_{metric}": self._score(classifier(x["test"]), y["test"]),
                        "weight_decay": weight_decay,
                        "converged": converged,
                    }
        return results

    def _fit(self, classifier: nn.Linear, x: Tensor, y: Tensor, weight_decay: float) -> bool:
        """Minimise the regularised loss from the classifier's current weights.

        Returns False when the budget ran out before L-BFGS met its tolerances.
        """
        if self.multilabel:
            valid = ~torch.isnan(y)
            target = y[valid].float()
        evals = 0
        optimizer = torch.optim.LBFGS(
            classifier.parameters(),
            max_iter=self.max_iter,
            max_eval=self.max_iter,
            tolerance_grad=1e-6,
            tolerance_change=1e-10,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            nonlocal evals
            evals += 1
            optimizer.zero_grad()
            logits = classifier(x)
            if self.multilabel:
                loss = F.binary_cross_entropy_with_logits(logits[valid], target)
            else:
                loss = F.cross_entropy(logits, y)
            loss = loss + 0.5 * weight_decay * classifier.weight.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        # Every iteration costs at least one evaluation, so staying below the budget means
        # L-BFGS stopped on a tolerance.
        return evals < self.max_iter

    def _score(self, logits: Tensor, y: Tensor) -> float:
        """Accuracy, or mean average precision over the labels for multilabel targets."""
        if not self.multilabel:
            return (logits.argmax(dim=-1) == y).float().mean().item()
        if not HAS_SKLEARN:
            return float("nan")
        import numpy as np

        probs = torch.sigmoid(logits).cpu().numpy()
        y_np = y.cpu().numpy()
        valid_np = ~torch.isnan(y).cpu().numpy()
        ap_scores = []
        for i in range(logits.size(1)):
            col = valid_np[:, i]
            if col.sum() > 0:
                ap_scores.append(average_precision_score(y_np[col, i], probs[col, i]))
        return float(np.mean(ap_scores)) if ap_scores else float("nan")
