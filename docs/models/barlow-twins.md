# Barlow Twins — Cross-Correlation Redundancy Reduction

**Zbontar et al., ICML 2021.** Same family as [VICReg](vicreg.md) — no negatives, no
teacher-student — but collapse is prevented by driving the cross-correlation matrix between
two views' embeddings toward the identity, rather than by an explicit variance term.

## How it works

- Encoder + a 3-layer projector, identical in shape to VICReg's.
- Cross-correlation matrix `C = (z1_norm.T @ z2_norm) / N` between the two (batch-normalized)
  views.
- Loss (`BarlowTwinsLoss`): `sum((1 − C_ii)²) + lambda * sum_{i≠j}(C_ij²)` — the diagonal term
  pushes each dimension to agree between views (invariance), the off-diagonal term pushes
  different dimensions apart (redundancy reduction).
- `lambda_param=None` defaults to `1/proj_dim`, computed inside `BarlowTwinsLoss`.

## Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `proj_dim` | `256` | projector output dimension |
| `lambda_param` | `None` | defaults to `1/proj_dim` when unset |

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 256, "num_layers": 2, "pool": False},
    "augment": [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}],
    "proj_dim": 256,  # lambda_param omitted -> defaults to 1/256
}
model = BarlowTwins(config, in_channels=dataset.num_features)
```

## Reference

`BarlowTwinsConfig` — see [Config API](../api/config.md#graphssl.config.schema.BarlowTwinsConfig).
Loss standalone at [`BarlowTwinsLoss`](../api/losses.md#graphssl.losses.BarlowTwinsLoss).
