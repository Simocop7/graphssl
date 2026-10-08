"""Ridge regression probe evaluator in pure PyTorch."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
from torch import Tensor

# L2 strengths tried by default. They all reuse one eigendecomposition, so a wide grid is free.
DEFAULT_RIDGE_WEIGHT_DECAYS: Tuple[float, ...] = tuple(10.0**e for e in range(3, -7, -1))


class RidgeEvaluator:
    """Ridge regression on frozen embeddings, for node- or graph-level regression targets.

    Fits ``mean squared error + weight_decay * ||W||²`` on the training rows in closed form:
    no optimiser, no random initialisation, so the same embeddings always give the same
    result. When several ``weight_decay`` values are given, the one with the lowest
    validation MAE is used (the strongest on ties) and the test error is reported for it.

    Args:
        weight_decay: L2 strength, or the values to select among on the validation split.
        standardize: Rescale every feature to zero mean and unit variance, with the
            statistics of the training rows only.

    ``evaluate()`` returns ``val_mae`` / ``test_mae``, ``val_rmse`` / ``test_rmse`` and the
    selected ``weight_decay``. Targets are ``[N]`` or ``[N, T]`` (errors averaged over the
    ``T`` targets) and must not contain NaN.
    """

    def __init__(
        self,
        *,
        weight_decay: Union[float, Sequence[float]] = DEFAULT_RIDGE_WEIGHT_DECAYS,
        standardize: bool = True,
    ):
        values = [weight_decay] if isinstance(weight_decay, (int, float)) else list(weight_decay)
        if not values or any(v < 0 for v in values):
            raise ValueError(f"weight_decay must be one or more values >= 0, got {weight_decay!r}")
        self.weight_decays: Tuple[float, ...] = tuple(
            sorted({float(v) for v in values}, reverse=True)
        )
        self.standardize = standardize

    def evaluate(
        self,
        embeddings: Tensor,
        targets: Tensor,
        train_idx: Tensor,
        val_idx: Tensor,
        test_idx: Tensor,
    ) -> Dict[str, float]:
        device = embeddings.device
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        if torch.isnan(targets).any():
            raise ValueError("RidgeEvaluator does not support missing (NaN) targets")

        # float64: the normal equations square the condition number of the embeddings.
        x: Dict[str, Tensor] = {}
        y: Dict[str, Tensor] = {}
        for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            x[split] = embeddings[idx].double()
            y[split] = targets[idx].to(device).double()
        mean = x["train"].mean(dim=0, keepdim=True)
        x = {split: x_s - mean for split, x_s in x.items()}
        if self.standardize:
            std = x["train"].std(dim=0, keepdim=True)
            std = torch.where(std > 0, std, torch.ones_like(std))  # constant feature: centre only
            x = {split: x_s / std for split, x_s in x.items()}
        intercept = y["train"].mean(dim=0, keepdim=True)

        # W(l) = V diag(1 / (s + l)) V^T X^T y / n, with X^T X / n = V diag(s) V^T.
        n = x["train"].size(0)
        evals, evecs = torch.linalg.eigh(x["train"].T @ x["train"] / n)
        evals = evals.clamp_min(0.0)
        projected = evecs.T @ (x["train"].T @ (y["train"] - intercept) / n)

        results: Dict[str, float] = {}
        for weight_decay in self.weight_decays:
            # A zero strength on a rank-deficient problem: the minimum-norm solution.
            scale = torch.where(evals + weight_decay > 1e-12, 1.0 / (evals + weight_decay), 0.0)
            weight = evecs @ (scale.unsqueeze(-1) * projected)
            errors = {s: x[s] @ weight + intercept - y[s] for s in ("val", "test")}
            val_mae = errors["val"].abs().mean().item()
            # Strict '<' keeps the stronger regularisation on ties.
            if not results or val_mae < results["val_mae"]:
                results = {
                    "val_mae": val_mae,
                    "test_mae": errors["test"].abs().mean().item(),
                    "val_rmse": errors["val"].pow(2).mean().sqrt().item(),
                    "test_rmse": errors["test"].pow(2).mean().sqrt().item(),
                    "weight_decay": weight_decay,
                }
        return results
