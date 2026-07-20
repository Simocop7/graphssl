"""FAISS-based positive pair miner for AFGRL: local (kNN∩adj) + global (same cluster)."""

from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class PositiveMiner:
    """Mine positive pairs via local graph neighbours and global k-means clustering.

    Two types of positives are identified and unioned:
    - Local:  top-k cosine-similarity neighbours that are also graph-adjacent (kNN ∩ adj).
    - Global: top-k neighbours that share a k-means cluster in ANY of the independent runs.

    Args:
        num_centroids: Number of k-means clusters per run.
        num_kmeans: Number of independent k-means runs (different random seeds).
        clus_num_iters: k-means iterations per run.
    """

    def __init__(
        self,
        num_centroids: int = 50,
        num_kmeans: int = 4,
        clus_num_iters: int = 20,
    ):
        if not HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required for AFGRL. Install with: pip install faiss-cpu"
            )
        self.num_centroids = num_centroids
        self.num_kmeans = num_kmeans
        self.clus_num_iters = clus_num_iters

    @torch.no_grad()
    def mine(
        self,
        adj: Tensor,
        student: Tensor,
        teacher: Tensor,
        topk: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return (src, dst) positive pair indices.

        Args:
            adj: Sparse COO adjacency [N, N] (values can be 0/1 or edge weights).
            student: L2-normalised student embeddings [N, D].
            teacher: L2-normalised teacher embeddings [N, D].
            topk: Top-k neighbours per node considered as positive candidates.

        Returns:
            src, dst: LongTensors of shape [P] (the positive pair indices).
        """
        device = student.device
        N = student.shape[0]

        # cosine similarity; +10 on diagonal so self is always among top-k
        sim = student @ teacher.T + torch.eye(N, device=device) * 10.0
        _, knn_idx = sim.topk(k=topk, dim=1, largest=True, sorted=True)  # [N, topk]

        row = torch.arange(N, device=device).repeat_interleave(topk)
        col = knn_idx.reshape(-1)  # [N*topk]

        # local positives: kNN ∩ graph adjacency
        knn_vals = torch.ones(N * topk, device=device)
        knn_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), knn_vals, (N, N))
        locality = (knn_sparse * adj).coalesce()

        # global positives: same k-means cluster in any of the num_kmeans runs
        teacher_np = teacher.detach().cpu().float().numpy()
        D = teacher_np.shape[1]
        cluster_labels_runs = []
        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(
                D,
                min(self.num_centroids, N),
                niter=self.clus_num_iters,
                gpu=False,
                seed=seed + 1234,
            )
            kmeans.train(teacher_np)
            _, ids = kmeans.index.search(teacher_np, 1)
            cluster_labels_runs.append(ids[:, 0])  # [N]

        labels_np = np.stack(cluster_labels_runs, axis=0)  # [num_kmeans, N]
        row_np = np.repeat(np.arange(N), topk)
        col_np = knn_idx.cpu().numpy().reshape(-1)

        # a pair is a global positive if they share a cluster in ANY run
        same_cluster = np.zeros(N * topk, dtype=bool)
        for labels in labels_np:
            same_cluster |= (labels[row_np] == labels[col_np])

        mask = torch.from_numpy(same_cluster).to(device)
        g_row = row[mask]
        g_col = col[mask]
        g_vals = torch.ones(g_row.numel(), device=device)
        globality = torch.sparse_coo_tensor(torch.stack([g_row, g_col]), g_vals, (N, N))

        positives = (locality + globality).coalesce()
        idx = positives.indices()  # [2, P]

        if idx.shape[1] == 0:
            src = torch.arange(N, device=device)
            dst = torch.randperm(N, device=device)
            return src, dst

        return idx[0], idx[1]
