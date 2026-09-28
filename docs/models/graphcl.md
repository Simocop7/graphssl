# GraphCL — Graph Contrastive Learning

**You et al., NeurIPS 2020.** Classic contrastive learning on graphs: two augmented views of
the same input, pulled together and pushed apart from everything else in the batch via
NT-Xent.

## How it works

- NT-Xent loss over two independently augmented views.
- Encoder + a 2-layer projector (`Projector(hidden_dim, hidden_dim, proj_dim)`).
- `protected_nodes` is passed to `compose()` so seed nodes survive augmentation during
  mini-batch node training — see [Augmentation](../augmentation.md#protected-nodes).

## Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `proj_dim` | `128` | projector output dimension |
| `tau` | `0.5` | NT-Xent temperature |
| `loss_chunk_size` | `4096` | max rows of the `[2N, 2N]` similarity matrix built at once; `None` = always full |

!!! tip "Large full-batch graphs"
    NT-Xent compares every node with every other, so the full similarity matrix is
    `(2N)²` floats — ~5.8 GiB for PubMed alone, several times that once autograd keeps its
    intermediates. When `2N > loss_chunk_size` the loss is computed in row chunks with
    gradient checkpointing: **identical value and gradients**, peak memory
    `O(loss_chunk_size · 2N)` instead (PubMed: ~1.9 GiB). Smaller inputs take the unchunked
    path unchanged.

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 128, "num_layers": 2, "pool": True},
    "augment": [{"name": "edge_drop", "p": 0.2}, {"name": "node_drop", "p": 0.1}],
    "proj_dim": 128,
    "tau": 0.5,
}
model = GraphCL(config, in_channels=dataset.num_features)
```

## Reference

`GraphCLConfig` — see [Config API](../api/config.md#graphssl.config.schema.GraphCLConfig).
