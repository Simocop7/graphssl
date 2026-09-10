# VICReg — Variance-Invariance-Covariance Regularization

**Bardes et al., ICLR 2022.** No negatives, no teacher-student pair, no stop-gradient trick —
representation collapse is prevented directly by the loss function's variance and covariance
terms.

## How it works

- Encoder + a 3-layer projector (`Projector(hidden_dim, hidden_dim*2, proj_dim)`).
- A triple loss (`VICRegLoss`):
    - **Invariance** — MSE between the two views' embeddings.
    - **Variance** — a hinge penalty keeping each embedding dimension's standard deviation
      above a threshold (this is what prevents collapse).
    - **Covariance** — an off-diagonal penalty decorrelating different embedding dimensions.

## Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `proj_dim` | `256` | projector output dimension |
| `invariance` | `25.0` | weight on the invariance term |
| `variance` | `25.0` | weight on the variance term |
| `covariance` | `1.0` | weight on the covariance term |

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 256, "num_layers": 2, "pool": False},
    "augment": [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}],
    "proj_dim": 256, "invariance": 25.0, "variance": 25.0, "covariance": 1.0,
}
model = VICReg(config, in_channels=dataset.num_features)
```

## Reference

`VICRegConfig` — see [Config API](../api/config.md#graphssl.config.schema.VICRegConfig). Loss
standalone at [`VICRegLoss`](../api/losses.md#graphssl.losses.VICRegLoss).
