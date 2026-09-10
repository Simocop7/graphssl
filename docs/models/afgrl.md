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
| `ema_tau` / `ema_tau_end` / `total_steps` | `0.99` / `1.0` / `0` | same semantics as BGRL |
| `topk` | `5` | neighbors considered for both local and global mining |
| `num_centroids` | `50` | k-means clusters |
| `num_kmeans` | `4` | independent k-means runs (a node is "global-positive" if it shares a cluster in *any* run) |
| `clus_num_iters` | `20` | k-means iterations |

!!! warning "Memory scaling"
    The FAISS-based k-means runs on the full embedding matrix in memory — this works fine for
    citation-network-scale graphs but becomes a real constraint above roughly 10⁵ nodes.
    Sub-sampled k-means / CPU-offload isn't implemented yet.

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 256, "num_layers": 2, "pool": False},
    "pred_hidden": 256,
    "ema_tau": 0.99, "ema_tau_end": 1.0, "total_steps": 300,
    "topk": 5, "num_centroids": 50, "num_kmeans": 4, "clus_num_iters": 20,
}
model = AFGRL(config, in_channels=dataset.num_features)
```

## Reference

`AFGRLConfig` — see [Config API](../api/config.md#graphssl.config.schema.AFGRLConfig).
