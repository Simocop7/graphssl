"""k-NN evaluator for node/graph classification."""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from torch import Tensor


class KNNEvaluator:
    """Weighted k-nearest-neighbour classifier on frozen embeddings.

    Uses cosine similarity to find neighbours. Labels are assigned by
    majority vote among the k nearest training nodes.

    Args:
        k: Number of nearest neighbours.
    """

    def __init__(self, k: int = 20):
        self.k = k

    def evaluate(
        self,
        embeddings: Tensor,
        labels: Tensor,
        train_idx: Tensor,
        val_idx: Tensor,
        test_idx: Tensor,
    ) -> Dict[str, float]:
        if labels.dim() > 1:
            labels = labels.squeeze(-1)  # OGB node datasets store single-label y as [N, 1]

        z = F.normalize(embeddings, dim=-1)
        z_train = z[train_idx]
        y_train = labels[train_idx]

        results: Dict[str, float] = {}
        for split, idx in [("val", val_idx), ("test", test_idx)]:
            z_s = z[idx]
            y_s = labels[idx]

            # [|split|, |train|] cosine similarity
            sim = z_s @ z_train.T
            k = min(self.k, z_train.size(0))
            topk_idx = sim.topk(k, dim=-1).indices   # [S, k]
            nn_labels = y_train[topk_idx]            # [S, k]
            pred = nn_labels.mode(dim=-1).values
            results[f"{split}_acc"] = (pred == y_s).float().mean().item()

        return results
