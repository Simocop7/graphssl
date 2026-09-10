# DGI — Deep Graph Infomax

**Veličković et al., ICLR 2019.** The simplest method in the library: no augmentations to
tune, no teacher-student pair — just a discriminator learning to tell real node embeddings
apart from corrupted ones.

## How it works

- Discriminates real vs. corrupted embeddings through a discriminator built around a learnable
  matrix `W` (`nn.Parameter`, initialized with `xavier_uniform_`).
- Corruption: either shuffling node features (permuting `x`) or shuffling edge destinations.
- A global summary `s = sigmoid(mean(h_pos))` for node-level tasks, or `global_mean_pool` for
  graph-level tasks.
- Discriminator score: `pos_logits = (h_pos * W @ s).sum(-1)` (and equivalently for negatives).
- Loss: `BCEWithLogitsLoss` over `[pos_logits, neg_logits]` with labels `[1,1,...,0,0,...]`.

## Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `corruption` | `"shuffle_nodes"` | or `"shuffle_edges"` |
| `shuffle_ratio` | `1.0` | fraction of nodes/edges corrupted |

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 128, "num_layers": 2, "pool": False},
    "corruption": "shuffle_nodes",
    "shuffle_ratio": 1.0,
}
model = DGI(config, in_channels=dataset.num_features)
```

## Reference

`DGIConfig` — see [Config API](../api/config.md#graphssl.config.schema.DGIConfig).
