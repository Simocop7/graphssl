# Supervised

A plain encoder + linear classification head, trained with `CrossEntropyLoss`. It isn't a
self-supervised method — it's included as the baseline every SSL method in this library should
be compared against.

```python
model = Supervised(config, in_channels, num_classes)
```

## How it works

- Encoder + linear head `nn.Linear(hidden_dim, num_classes)`.
- Supports both full-batch and mini-batch training. In mini-batch mode (`NeighborLoader`,
  `batch.batch_size` set), the loss is cropped to `[:batch_size]` for the seed nodes.
- `graph_level` is derived from `cfg.encoder.pool`, like every other model.

!!! warning "Full-batch training computes the loss over the whole `Data` object"
    `compute_loss()` only restricts to seed nodes when `batch.batch_size` is present (i.e. in
    mini-batch/`NeighborLoader` mode). In full-batch mode there is currently no automatic
    masking to `train_idx` — passing the full graph's `Data` object directly, unfiltered, will
    include validation/test labels in the loss. Use `NeighborLoader(input_nodes=train_idx)` for
    full-batch-sized graphs (Cora/CiteSeer/PubMed) rather than `DataModule.train_dataloader()`
    directly if you need strict train/val/test separation. This was discovered while building
    the [benchmark runner](../benchmarks.md); tracked as a follow-up fix.

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 128, "num_layers": 2, "pool": False},
}
model = Supervised(config, in_channels=dataset.num_features, num_classes=dataset.num_classes)
```

## Reference

`SupervisedConfig` — see [Config API](../api/config.md#graphssl.config.schema.SupervisedConfig).
