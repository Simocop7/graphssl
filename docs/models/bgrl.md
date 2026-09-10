# BGRL — Bootstrapped Graph Representation Learning

**Thakoor et al., ICLR 2022.** Teacher-student bootstrapping with no negative pairs at all: an
online (student) encoder learns to predict a slowly-moving target (teacher) encoder's
representation of a differently-augmented view of the same graph.

## How it works

- Online encoder (student) + a predictor MLP (`Predictor(hidden_dim, pred_hidden,
  hidden_dim)`); target encoder (teacher), gradient-free.
- Loss: `2 - cos(pred1, target2) - cos(pred2, target1)` — `CosineRegressionLoss(symmetric=True)`.
  The correct call is `loss_fn(p1, t1, p2, t2)`; internally the loss pairs `t2` with `p1` and
  `t1` with `p2`.
- The target encoder is updated via EMA in `post_step()`, via `update_ema_params()`.
- Momentum τ is cosine-annealed from `ema_tau` to `ema_tau_end` over `total_steps`
  (`CosineEMAScheduler`). If `total_steps=0`, τ stays fixed.

!!! note "Target-encoder initialization is intentional, not incidental"
    The target encoder is built as `deepcopy(encoder)` **followed by `reset_parameters()`**,
    giving it weights that are deliberately different from the online encoder at
    initialization. This asymmetry is called out explicitly in Appendix B of the BGRL paper as
    critical for convergence — contrast with [AFGRL](afgrl.md), which skips the reset. Getting
    this backwards (or "simplifying" it away) is a common way to silently break BGRL.

## Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `pred_hidden` | `512` | predictor hidden dimension |
| `ema_tau` | `0.99` | starting EMA momentum |
| `ema_tau_end` | `1.0` | final EMA momentum |
| `total_steps` | `0` | EMA annealing horizon; `0` = fixed τ |

## Config example

```python
config = {
    "encoder": {"name": "gin", "hidden_dim": 256, "num_layers": 2, "pool": False},
    "augment": [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}],
    "pred_hidden": 256,
    "ema_tau": 0.99,
    "ema_tau_end": 1.0,
    "total_steps": 300,  # match num_epochs so the EMA schedule spans the full run
}
model = BGRL(config, in_channels=dataset.num_features)
```

## Benchmarked results

See [Benchmarks](../benchmarks.md) for validated Cora/CiteSeer/PubMed numbers — this is
currently the most extensively validated model in the library.

## Reference

`BGRLConfig` — see [Config API](../api/config.md#graphssl.config.schema.BGRLConfig).
