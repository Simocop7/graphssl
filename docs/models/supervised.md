# Supervised

A plain encoder + linear classification head, trained with `CrossEntropyLoss`. It isn't a
self-supervised method — it's included as the baseline every SSL method in this library should
be compared against.

```python
model = Supervised(config, in_channels, num_classes)
```

## How it works

- Encoder + linear head `nn.Linear(hidden_dim, num_classes)`.
- `graph_level` is derived from `cfg.encoder.pool`, like every other model. Which labels the
  loss sees depends on the mode:

| Mode | Loss computed on |
|---|---|
| Graph-level (`pool=True`) | every graph in the batch (node embeddings mean-pooled per graph) |
| Node-level mini-batch (`NeighborLoader`, `batch.batch_size` set) | the first `batch_size` seed nodes — build the loader with `input_nodes=train_idx` |
| Node-level full-batch | the nodes in `data.train_mask` only |

!!! warning "Full-batch node training requires `data.train_mask`"
    Without it `compute_loss()` raises a `ValueError` instead of silently training on the
    validation/test labels too. Planetoid datasets ship with the public split's masks.

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 128, "num_layers": 2, "pool": False},
}
model = Supervised(config, in_channels=dataset.num_features, num_classes=dataset.num_classes)
```

## Reference

`SupervisedConfig` — see [Config API](../api/config.md#graphssl.config.schema.SupervisedConfig).
