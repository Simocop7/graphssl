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
        chunk_size: Rows of the evaluated split scored at a time. The full
            ``[|split|, |train|]`` similarity matrix is never built (it would take ~17 GB
            for the ogbn-arxiv test split); the predictions are identical.
    """

    def __init__(self, k: int = 20, chunk_size: int = 4096):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.k = k
        self.chunk_size = chunk_size

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

            k = min(self.k, z_train.size(0))
            preds = []
            for start in range(0, z_s.size(0), self.chunk_size):
                # [chunk, |train|] cosine similarity
                sim = z_s[start : start + self.chunk_size] @ z_train.T
                topk_idx = sim.topk(k, dim=-1).indices  # [chunk, k]
                preds.append(y_train[topk_idx].mode(dim=-1).values)
            pred = torch.cat(preds) if preds else y_s.new_empty(0)
            results[f"{split}_acc"] = (pred == y_s).float().mean().item()

        return results
