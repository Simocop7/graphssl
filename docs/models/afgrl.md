# AFGRL — Augmentation-Free Graph Representation Learning

**Lee et al., AAAI 2022.** Same teacher-student skeleton as [BGRL](bgrl.md), but instead of
relying on stochastic augmentation to create two views, positive pairs are *mined* directly
from the graph's own structure and embedding geometry.

!!! note "Requires FAISS"
    `pip install faiss-cpu` (included in the `full` extra). Tests and the
    [benchmark runner](../benchmarks.md) auto-skip AFGRL when it isn't installed.

## How it works

- Online encoder (student) + predictor MLP; target encoder, updated via EMA, gradient-free.
- **Difference from BGRL**: the target encoder starts with the *same* weights as the online
  encoder — `deepcopy` **without** `reset_parameters()`. Diversity comes from the miner, not
  from weight asymmetry.
- **PositiveMiner** (`utils/positive_miner.py`) unions two kinds of positives per node:
    - *Local*: top-k cosine-similarity neighbors that are **also** adjacent in the graph
      (kNN ∩ adjacency).
    - *Global*: top-k neighbors sharing a k-means cluster in **at least one** of `num_kmeans`
      independent runs.
- Graph-level tasks skip the miner entirely (it's not meaningful across independent graphs) and
  fall back to a simple teacher-student loss on pooled embeddings.
- EMA scheduling is identical to BGRL.

## Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `pred_hidden` | `512` | predictor hidden dimension |
| `pred_norm` / `pred_activation` | `"batch"` / `"relu"` | predictor layout, same semantics as BGRL |
| `ema_tau` / `ema_tau_end` / `total_steps` | `0.99` / `1.0` / `0` | same semantics as BGRL |
| `topk` | `5` | neighbors considered for both local and global mining |
| `num_centroids` | `50` | k-means clusters |
| `num_kmeans` | `4` | independent k-means runs (a node is "global-positive" if it shares a cluster in *any* run) |
| `clus_num_iters` | `20` | k-means iterations |
| `kmeans_threads` | `8` | OpenMP threads FAISS may use for k-means (capped at the CPU count); `None` = FAISS default (all cores) |
| `knn_chunk_size` | `4096` | rows of the N × N similarity matrix computed at once for the top-k search; `None` = whole matrix |

!!! tip "Why `kmeans_threads` exists"
    The miner re-runs `num_kmeans` small k-means every training step. With FAISS's default of
    one thread per core, thread start-up and contention dominate: on a 48-core machine a Cora
    step took ~5.7 s with 48 threads vs ~0.1 s with 4–16. The previous FAISS thread count is
    restored after each call, so this doesn't leak into the rest of your program.

!!! warning "Scaling"
    The miner's top-k neighbor search compares every pair of nodes, every step. Its
    **memory** is chunked: `knn_chunk_size` rows of the N × N similarity matrix at a time
    (`utils.positive_miner.topk_similar`), so peak memory is O(`knn_chunk_size` · N) instead of
    O(N²) — the full matrix would be ~1.5 GB on PubMed and ~115 GB on ogbn-arxiv — with exactly
    the same neighbors. Its **time** stays O(N² d) per step, plus `num_kmeans` k-means runs,
    and mining runs on the whole graph (no mini-batch mode), so very large graphs remain slow.

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 256, "num_layers": 2, "pool": False},
    "pred_hidden": 256,
    "ema_tau": 0.99, "ema_tau_end": 1.0, "total_steps": 300,
    "topk": 5, "num_centroids": 50, "num_kmeans": 4, "clus_num_iters": 20,
    "kmeans_threads": 8, "knn_chunk_size": 4096,
}
model = AFGRL(config, in_channels=dataset.num_features)
```

## Reference

`AFGRLConfig` — see [Config API](../api/config.md#graphssl.config.schema.AFGRLConfig).
