"""FAISS-based positive pair miner for AFGRL: local (kNN∩adj) + global (same cluster)."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from torch import Tensor

try:
    import faiss
    import numpy as np

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def sparse_coo_unchecked(indices: Tensor, values: Tensor, size: Tuple[int, int]) -> Tensor:
    """``torch.sparse_coo_tensor`` with invariant checks explicitly (not implicitly) disabled.

    Same behavior as the default, minus the "Sparse invariant checks are implicitly
    disabled" UserWarning recent PyTorch versions emit. Passing ``check_invariants=False``
    does not silence it; only an explicit opt-out via the context manager does. Callers
    guarantee valid indices (they come from ``edge_index`` or ``topk``).
    """
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        return torch.sparse_coo_tensor(indices, values, size)


@torch.no_grad()
def topk_similar(
    student: Tensor, teacher: Tensor, k: int, chunk_size: Optional[int] = 4096
) -> Tensor:
    """Indices [N, k] of each node's ``k`` most similar nodes, itself always included.

    Similarity is ``student @ teacher.T`` (cosine for L2-normalised inputs), with 10 added to
    the self-similarity so that every node is its own top neighbour, as in AFGRL. Rows are
    processed ``chunk_size`` at a time: identical result, peak memory O(chunk_size · N)
    instead of O(N²) — the dense N×N matrix alone would need ~115 GB on ogbn-arxiv.
    ``None`` materialises the full matrix at once.
    """
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0 or None, got {chunk_size}")
    N = student.shape[0]
    step = N if chunk_size is None else chunk_size
    chunks: List[Tensor] = []
    for start in range(0, N, step):
        sim = student[start : start + step] @ teacher.T  # [C, N]
        rows = torch.arange(sim.shape[0], device=sim.device)
        sim[rows, rows + start] += 10.0
        chunks.append(sim.topk(k=k, dim=1, largest=True, sorted=True).indices)
    return torch.cat(chunks)


class PositiveMiner:
    """Mine positive pairs via local graph neighbours and global k-means clustering.

    Two types of positives are identified and unioned:
    - Local:  top-k cosine-similarity neighbours that are also graph-adjacent (kNN ∩ adj).
    - Global: top-k neighbours that share a k-means cluster in ANY of the independent runs.

    Args:
        num_centroids: Number of k-means clusters per run.
        num_kmeans: Number of independent k-means runs (different random seeds).
        clus_num_iters: k-means iterations per run.
        kmeans_threads: OpenMP threads FAISS may use for k-means (capped at the CPU count).
            FAISS defaults to every core, which on many-core machines is dramatically
            slower for the small, per-step k-means AFGRL runs (thread start-up and
            contention dominate). ``None`` leaves FAISS's own setting untouched. The
            previous FAISS thread count is restored after every call.
        knn_chunk_size: Rows of the N×N similarity matrix computed at once for the top-k
            search (see ``topk_similar``). ``None`` builds the whole matrix.
    """

    def __init__(
        self,
        num_centroids: int = 50,
        num_kmeans: int = 4,
        clus_num_iters: int = 20,
        kmeans_threads: Optional[int] = 8,
        knn_chunk_size: Optional[int] = 4096,
    ):
        if not HAS_FAISS:
            raise ImportError(
                "faiss-cpu is required for AFGRL. Install with: pip install faiss-cpu"
            )
        self.num_centroids = num_centroids
        self.num_kmeans = num_kmeans
        self.clus_num_iters = clus_num_iters
        self.kmeans_threads = kmeans_threads
        self.knn_chunk_size = knn_chunk_size

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

        knn_idx = topk_similar(student, teacher, topk, self.knn_chunk_size)  # [N, topk]

        row = torch.arange(N, device=device).repeat_interleave(topk)
        col = knn_idx.reshape(-1)  # [N*topk]

        # local positives: kNN ∩ graph adjacency
        knn_vals = torch.ones(N * topk, device=device)
        knn_sparse = sparse_coo_unchecked(torch.stack([row, col]), knn_vals, (N, N))
        locality = (knn_sparse * adj).coalesce()

        # global positives: same k-means cluster in any of the num_kmeans runs
        labels_np = self._cluster_labels(teacher.detach().cpu().float().numpy())
        row_np = np.repeat(np.arange(N), topk)
        col_np = knn_idx.cpu().numpy().reshape(-1)

        # a pair is a global positive if they share a cluster in ANY run
        same_cluster = np.zeros(N * topk, dtype=bool)
        for labels in labels_np:
            same_cluster |= labels[row_np] == labels[col_np]

        mask = torch.from_numpy(same_cluster).to(device)
        g_row = row[mask]
        g_col = col[mask]
        g_vals = torch.ones(g_row.numel(), device=device)
        globality = sparse_coo_unchecked(torch.stack([g_row, g_col]), g_vals, (N, N))

        positives = (locality + globality).coalesce()
        idx = positives.indices()  # [2, P]

        if idx.shape[1] == 0:
            src = torch.arange(N, device=device)
            dst = torch.randperm(N, device=device)
            return src, dst

        return idx[0], idx[1]

    def _cluster_labels(self, x: np.ndarray) -> np.ndarray:
        """Run ``num_kmeans`` independent k-means on ``x`` [N, D]; return labels [num_kmeans, N]."""
        N, D = x.shape
        prev_threads = faiss.omp_get_max_threads()
        if self.kmeans_threads is not None:
            faiss.omp_set_num_threads(min(self.kmeans_threads, os.cpu_count() or 1))
        try:
            runs: List[np.ndarray] = []
            for seed in range(self.num_kmeans):
                kmeans = faiss.Kmeans(
                    D,
                    min(self.num_centroids, N),
                    niter=self.clus_num_iters,
                    gpu=False,
                    seed=seed + 1234,
                )
                kmeans.train(x)
                _, ids = kmeans.index.search(x, 1)
                runs.append(ids[:, 0])  # [N]
        finally:
            faiss.omp_set_num_threads(prev_threads)
        return np.stack(runs, axis=0)
