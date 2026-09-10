# Models Overview

Every model is a pure `nn.Module` implementing `forward()` and `compute_loss()`, built the
same way regardless of which SSL paradigm it belongs to:

```python
model = ModelClass(config: dict, in_channels: int)
# Supervised additionally takes num_classes: int
model = Supervised(config, in_channels, num_classes)
```

No model reaches into a trainer, logger, or datamodule — see [Architecture](../architecture.md)
for why that boundary is enforced.

## Which one should I use?

| Situation | Model | Why |
|---|---|---|
| Want a simple baseline, minimal code | [DGI](dgi.md) | One discriminator, no augmentation to tune |
| Need explicit negatives / large batches | [GraphCL](graphcl.md) | Classic contrastive, NT-Xent |
| Want to avoid choosing/tuning structural augmentations | [AFGRL](afgrl.md) | Mines positives from topology (kNN ∩ adjacency + k-means), no augmentation |
| Want the most stable/robust default | [BGRL](bgrl.md) | Teacher-student, no negatives, generally less sensitive to hyperparameters than GraphCL |
| Need embeddings that are decorrelated/non-collapsing by construction | [VICReg](vicreg.md) / [Barlow Twins](barlow-twins.md) | Loss terms explicitly penalize redundancy across dimensions |
| Exploring something closer to ViT-style self-distillation | [GraphDINO](graphdino.md) | Prototypes + teacher temperature warmup |
| Have labels and want a supervised reference point | [Supervised](supervised.md) | Same API, direct CrossEntropyLoss |

## Config dataclasses

Every model's `config` dict is validated by a dedicated dataclass in
[`src/graphssl/config/schema.py`](https://github.com/Simocop7/graphssl/blob/main/src/graphssl/config/schema.py)
before the model is built — a misconfigured run fails at construction time, not mid-training.

| Model | Config dataclass |
|---|---|
| Supervised | `SupervisedConfig` |
| DGI | `DGIConfig` |
| GraphCL | `GraphCLConfig` |
| BGRL | `BGRLConfig` |
| AFGRL | `AFGRLConfig` |
| VICReg | `VICRegConfig` |
| Barlow Twins | `BarlowTwinsConfig` |
| GraphDINO | `GraphDINOConfig` |

See the [Config API reference](../api/config.md) for the full field list of each.
