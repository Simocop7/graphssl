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
- The real and the corrupted graph are encoded in **one forward pass**, so they are
  normalised with the same batch statistics.
- On a `NeighborLoader` batch only the seed nodes (the first `batch_size`) enter the summary
  and the loss: the sampled neighbours are there for message passing.

!!! note "Why one forward pass"
    With a batch-normalised encoder, two separate passes give each graph its own batch
    statistics, and the discriminator can tell the passes apart from those alone. In
    mini-batch training on ogbn-arxiv it did exactly that: it reached 100% accuracy even on
    nodes with no incoming edge in the sampled batch, where the real and the corrupted
    embedding are identically distributed, the loss went to zero in two epochs and the
    embeddings ended up worse than an untrained encoder's.

!!! warning "Long mini-batch runs still degrade"
    With the single forward pass the collapse is gone, but on ogbn-arxiv (one seed) the
    embeddings peak after 20–40 epochs of mini-batch training and then decline: at 100
    epochs they are below the untrained encoder again and their effective rank has shrunk
    from 4.8 to 1.6. The cause is not identified. In full-batch training on the citation
    networks there is no such decline up to 3000 steps. Until this is understood, stop DGI
    early when training with `NeighborLoader`.

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
