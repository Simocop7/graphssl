# GraphMAE — masked feature reconstruction

**Hou et al., KDD 2022.** The one generative objective in the library: instead of making two
views agree, it hides the features of some nodes and trains the encoder so that their
features can be rebuilt from the neighbors'.

## How it works

- A fraction `mask_rate` of the nodes is masked. Of those, a fraction `replace_rate` gets the
  features of random nodes; the rest get a learnable mask token (zeros at initialization) in
  place of their features.
- The encoder embeds the corrupted graph. A bias-free linear map follows, then
  **re-masking**: the representations of the masked nodes are set to zero, so the decoder
  cannot copy them.
- A one-layer, one-head GAT decoder rebuilds the input features. The loss is the **scaled
  cosine error** on the masked nodes only, against their clean features:
  `mean (1 − cos(x̂ᵢ, xᵢ))^alpha`.
- `forward()` embeds the clean graph; for graph-level tasks it pools, like every model.
- On a `NeighborLoader` batch the seed nodes and the sampled neighbors are masked in the same
  proportion, and the loss uses the masked seeds.

!!! note "A decoder that scales with the feature dimension"
    The decoder computes exactly what PyG's `GATConv(hidden, num_features, heads=1)`
    computes, but aggregates the neighbors in the hidden dimension and maps to the feature
    space afterwards (`graphssl.nn.GATDecoder`). `GATConv` maps first and sends
    feature-wide messages along every edge: about 5 GB per step on Coauthor CS (6,805
    features), six times slower.

Categorical node features (`encoder.node_emb_num_classes`) are refused: a mask token in
feature space has no meaning for atom-type indices.

## Hyperparameters

Defaults are the reference implementation's Cora configuration.

| Field | Default | Notes |
|---|---|---|
| `mask_rate` | `0.5` | fraction of the nodes masked, in (0, 1) |
| `replace_rate` | `0.05` | fraction of the masked nodes given random nodes' features |
| `alpha` | `3.0` | exponent of the scaled cosine error, ≥ 1 |

## Config example

```python
config = {
    "encoder": {"name": "gcn", "hidden_dim": 256, "num_layers": 2, "pool": False},
    "mask_rate": 0.5,
    "replace_rate": 0.05,
    "alpha": 3.0,
}
model = GraphMAE(config, in_channels=dataset.num_features)
```

## Reference

`GraphMAEConfig` — see [Config API](../api/config.md#graphssl.config.schema.GraphMAEConfig).
