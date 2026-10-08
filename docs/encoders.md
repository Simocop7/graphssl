# Encoders (GNN Backbones)

Three interchangeable backbones — GCN, GIN, Graph Transformer — swappable via one config line
for any model. All expose the same signature:

```python
forward(x, edge_index, batch, edge_attr=None) -> Tensor[N, out_dim]
```

and are registered via `@ENCODERS.register("name")` (see [Registry Pattern](architecture.md#registry-pattern)).

## Shared config (`EncoderConfig`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | str | — | `'gin'`, `'gcn'`, `'transformer'` |
| `hidden_dim` | int | — | hidden/output dimension |
| `num_layers` | int | — | number of layers |
| `norm_type` | str | `"batch"` | `'batch'`, `'layer'`, `'none'` — uniform across all encoders |
| `weight_standardization` | bool | `False` | GCN only: standardize the convolution weights of every layer after the first |
| `pool` | bool | `True` | graph-level task: the model pools the node embeddings per graph |
| `readout` | str | `"mean"` | `'mean'`, `'sum'` or `'max'`: the pooling used when `pool=True` |
| `drop` | float | `0.2` | dropout rate |
| `mlp_ratio` | float | `2.0` | hidden-dim multiplier for the internal MLP |
| `edge_dim` | int \| None | `None` | enables edge-feature-aware convolution (GINEConv / TransformerConv) |
| `node_emb_num_classes` | int \| None | `None` | replaces `Linear` with `nn.Embedding` for categorical node features (e.g. ZINC: 28 atom types) |
| `edge_emb_num_classes` | int \| None | `None` | `nn.Embedding` for categorical edge features. Requires `edge_dim` |

`EncoderConfig.build(in_channels)` uses `inspect.signature` to forward only the kwargs the
specific encoder actually accepts — unsupported fields are silently ignored, so the same
config schema works across encoders that don't share every feature.

## GCN

- `num_layers` × (`GCNConv` → norm → `PReLU`).
- Optional weight standardization (`weight_standardization`) of the convolution weights of
  every layer after the first — BGRL's setting on ogbn-arxiv, together with layer norm.
- Backward compatible with the older `batchnorm=True/False` / `layernorm=True/False` flags.
- Exposes `reset_parameters()` — used by BGRL to give its target encoder different initial
  weights (see [BGRL](models/bgrl.md)).

## GIN (Graph Isomorphism Network)

- A stack of `GINLayer`s with pre-norm + residual connections:
  `norm(x) → GINConv(h) → ReLU → Dropout → x + h`.
- `edge_dim != None` switches to `GINEConv` (edge-feature aware). If `edge_dim` is set but
  `edge_attr=None` at runtime, the layer builds a correctly-shaped zero tensor instead of
  crashing — the same encoder works with or without edge features.
- `node_emb_num_classes` / `edge_emb_num_classes` swap in `nn.Embedding` for categorical
  node/edge features (see the [ZINC example](getting-started.md)).

## Graph Transformer

- A stack of `TransformerBlock`s (PyG's `TransformerConv`), pre-norm + FFN + residual.
- Supports `edge_attr` via `TransformerConv(edge_dim=...)`.
- Same `norm_type` API as GIN (batch/layer/none), and the same `node_emb_num_classes` /
  `edge_emb_num_classes` support.

## Reference

See the [Encoders API](api/encoders.md) for full signatures.
