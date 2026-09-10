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
