# Architecture

## Design principles

1. **Config-driven uniform API.** Every model's `__init__(config: dict, in_channels: int)`. The
   config is validated by a dataclass in `config/schema.py`. The encoder is always built via
   `cfg.encoder.build(in_channels)` — no encoder is ever constructed inline inside a model.
2. **Losses as standalone `nn.Module`s** in `losses/`, independently testable and combinable
   via [`CombinedLoss`](losses.md#combining-losses).
3. **Registry pattern** for every extensible component: `@ENCODERS.register("name")`.
4. **Hook-driven trainer.** `post_backward()`, `post_step()`, `on_epoch_start/end()` instead of
   subclassing the trainer itself.
5. **Layered separation.** `core → encoders/models/nn/augmentation/losses → evaluation →
   training → data`. No higher-level module imports from a lower one.

These aren't just conventions — they're what make it possible to swap BGRL for VICReg with a
one-line config change and reuse the exact same training loop, evaluation code, and callbacks.

## Registry pattern

```python
from graphssl.registry import ENCODERS, LOSSES, AUGMENTS, OBJECTIVES, DATASETS, LOADERS

# Register a custom encoder:
@ENCODERS.register("my_gat")
class GATEncoder(nn.Module): ...

# Instantiate from the registry:
enc = ENCODERS.build("my_gat", in_channels=32, hidden_dim=64)
```

Once registered, `"my_gat"` is usable in *any* model's config, for *any* model, with no other
code changes:

```python
config = {"encoder": {"name": "my_gat", "hidden_dim": 128, ...}, ...}
```

Available registries:

| Registry | Purpose |
|---|---|
| `ENCODERS` | GNN backbones |
| `HEADS` | projection/prediction heads |
| `AUGMENTS` | augmentation transforms |
| `LOSSES` | loss functions |
| `LOADERS` | DataLoader factories |
| `OBJECTIVES` | custom SSL objectives (placeholder for extensions) |
| `DATASETS` | custom dataset builders (placeholder for extensions) |

## Trainer and callbacks

[`DINOTrainer`](api/training.md) is generic — it works with any model implementing the
`BaseSSLModel` protocol.

**Per-batch order:**
```
compute_loss → backward → clip_grad_norm → post_backward → optimizer.step → post_step
```

**Per-epoch order:**
```
on_epoch_start → batches → on_epoch_end
```

Models implement the hooks they need; everything else is a no-op:

| Model | `post_backward` | `post_step` | `on_epoch_start` | `on_epoch_end` |
|---|---|---|---|---|
| BGRL | — | EMA teacher | — | — |
| AFGRL | — | EMA teacher | — | — |
| GraphDINO | freeze last layer | EMA + center update | teacher temp warmup | epoch counter |

Or write your own loop directly, since models expose these hooks publicly:

```python
for batch in loader:
    loss = model.compute_loss(batch)
    loss.backward()
    model.post_backward()
    optimizer.step()
    model.post_step()
```

**Callbacks:**

- `EmbeddingLoggerCallback` — saves embeddings every N epochs as `.pt`
- `LinearEvalCallback` — periodic linear probe during training
- `VisualizationCallback` — UMAP scatter plot (`pip install umap-learn matplotlib`)

## Key utilities

| Utility | Location | Purpose |
|---|---|---|
| `update_ema_params(student, teacher, tau)` | `utils/ema.py` | Updates parameters *and* buffers (e.g. BatchNorm running stats). Buffers are copied directly, not averaged. |
| `CosineDecayScheduler` | `utils/schedulers.py` | Cosine decay with linear warmup. |
| `CosineEMAScheduler` | `utils/schedulers.py` | Increasing cosine EMA momentum (BGRL/GraphDINO). |
| `pool_graph_embeddings` | `nn/pooling.py` | `global_mean_pool` with a fallback when `batch=None`. |
| `extract_embeddings` | `evaluation/visualization.py` | Extracts embeddings given a model + `DataModule`; handles graph-level, node full-batch, and node mini-batch (`NeighborLoader`) paths uniformly. |
| `LogRegEvaluator` | `evaluation/linear_probe.py` | Pure-PyTorch linear evaluator; multilabel targets (e.g. ogbg-molpcba) use BCE + average precision. |
| `DataModule` | `data/datamodule.py` | Wraps a PyG dataset for both graph-level (`DataLoader`) and node-level (`Data` + masks + `neighbor_loader()`) tasks. |

## Project structure

```
src/graphssl/
├── core/          # BaseModel, BaseSSLModel, BaseEncoder, BaseAugmentation, Callback, Registry
├── config/        # schema.py (dataclass validation), load.py (load_config, build_model)
├── registry/      # ENCODERS, HEADS, AUGMENTS, LOSSES, LOADERS, OBJECTIVES, DATASETS
├── encoders/      # GCN, GIN, Transformer (auto-registered)
├── models/        # DGI, GraphCL, BGRL, AFGRL, GraphDINO, VICReg, BarlowTwins, Supervised
├── losses/        # nt_xent, dino, vicreg, barlow, regression, combined (CombinedLoss)
├── nn/            # MLP, DINOHead, norm (weight_standardize), pooling
├── augmentation/  # functional.py, transforms.py, compose.py
├── evaluation/    # linear_probe, knn, visualization (extract_embeddings)
├── training/      # trainer.py (DINOTrainer), callbacks.py
├── data/          # pure-Python DataModule
└── utils/         # ema.py, schedulers.py, positive_miner.py
```

## Development tooling

Linting/formatting (`ruff`), type checking (`mypy` — clean, CI-enforced), coverage,
pre-commit hooks and CI are covered in [Contributing](contributing.md) rather than duplicated
here.
