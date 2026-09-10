GraphSSL is a Python library for Graph Machine Learning (GML).
It implements the main SSL algorithms on graphs in pure PyTorch,
with a modular, reusable architecture.

---

## TECH STACK

- Python 3.10+
- PyTorch + PyTorch Geometric (PyG)
- Pure PyTorch only in the core: zero dependencies on PyTorch Lightning, Hydra, OmegaConf
- FAISS-cpu for positive mining (AFGRL) — optional
- UMAP + matplotlib + seaborn for visualization — optional
- ogb + pyyaml for benchmarks and YAML configs — optional

---

## PUBLIC API — UNIFORM CONSTRUCTION

**Every model uses the same public signature:**
```python
model = ModelClass(config: Dict, in_channels: int)
# Supervised adds: num_classes: int
model = Supervised(config, in_channels, num_classes)
```

The `config` dict is validated by a dedicated dataclass in `src/graphssl/config/schema.py`.
`EncoderConfig.build(in_channels)` instantiates the encoder through the `ENCODERS` registry.

**YAML factory:**
```python
from graphssl.config.load import load_config, build_model
cfg = load_config("configs/bgrl.yaml")
model = build_model(cfg, in_channels=dataset.num_features)
```

**Per-model config dataclass:**
| Model        | Config dataclass    |
|--------------|---------------------|
| DGI          | `DGIConfig`         |
| GraphCL      | `GraphCLConfig`     |
| VICReg       | `VICRegConfig`      |
| BarlowTwins  | `BarlowTwinsConfig` |
| BGRL         | `BGRLConfig`        |
| AFGRL        | `AFGRLConfig`       |
| Supervised   | `SupervisedConfig`  |
| GraphDINO    | `GraphDINOConfig`   |

All defined in `src/graphssl/config/schema.py`.

---

## IMPLEMENTED ALGORITHMS

Every model is a pure `nn.Module` with `forward()` and `compute_loss()`.
No access to the trainer, logger, or datamodule from inside a model.

### Supervised
- Encoder + linear head `nn.Linear(hidden_dim, num_classes)`, CrossEntropyLoss
- Supports full-batch and mini-batch (crop to `[:batch_size]` for seed nodes)
- `graph_level` derived from `cfg.encoder.pool`

### DGI (Deep Graph Infomax)
- Discriminates real vs. corrupted embeddings via a discriminator with a learnable matrix W
  (`nn.Parameter`, initialized with `xavier_uniform_`)
- Corruption: node shuffling (permutation of x) or edge-destination shuffling
- Global summary `s = sigmoid(mean(h_pos))` for node-level, `global_mean_pool` for graph-level
- Discriminator: `pos_logits = (h_pos * W@s).sum(-1)`
- Loss: BCEWithLogitsLoss over [pos_logits, neg_logits] with labels [1,1,...,0,0,...]
- Hyperparameters: `corruption="shuffle_nodes", shuffle_ratio=1.0`

### GraphCL (Graph Contrastive Learning)
- NT-Xent loss over 2 augmented views
- Encoder + 2-layer projector `Projector(hidden_dim, hidden_dim, proj_dim)`
- `protected_nodes` passed to `compose()` for seed nodes in mini-batch node training
- Hyperparameters: `proj_dim=128, tau=0.5`

### BGRL (Bootstrapped Graph Representation Learning)
- Teacher-student with an EMA update on the target encoder
- Online encoder (student) + predictor MLP `Predictor(hidden_dim, pred_hidden, hidden_dim)`; target encoder (teacher, no grad)
- Loss: `2 - cos(pred1, target2) - cos(pred2, target1)` — `CosineRegressionLoss(symmetric=True)`
  - Correct call: `loss_fn(p1, t1, p2, t2)` → internally the loss pairs t2 with p1 and t1 with p2
- EMA update of the target encoder in `post_step()` via `update_ema_params()`
- Momentum τ cosine-annealed from `ema_tau` to `ema_tau_end` over `total_steps` — `CosineEMAScheduler`.
  If `total_steps=0`, τ is fixed.
- Target encoder: `deepcopy(encoder)` + `reset_parameters()` — weights intentionally differ
  from the online encoder (critical for convergence, Appendix B of the BGRL paper)
- Hyperparameters: `ema_tau=0.99, ema_tau_end=1.0, total_steps=0, pred_hidden=512`

### AFGRL (Augmentation-Free Graph Representation Learning)
- Like BGRL but without structural augmentation: positive pairs are mined from graph structure
- Online encoder (student) + predictor MLP; target encoder (EMA, no grad)
- **Difference from BGRL**: the target encoder starts with the same weights as the online encoder (`deepcopy` without reset)
- **PositiveMiner** (`utils/positive_miner.py`): union of two kinds of positives per node:
  - Local: top-k cosine-similarity neighbors that are ALSO adjacent in the graph (kNN ∩ adj)
  - Global: top-k neighbors sharing a k-means cluster in AT LEAST ONE of the `num_kmeans` runs
- Graph-level loss: no miner; a simple teacher-student loss on pooled embeddings
- EMA: identical to BGRL — `CosineEMAScheduler` in `post_step()`
- Hyperparameters: `ema_tau=0.99, ema_tau_end=1.0, total_steps=0, topk=5,
  num_centroids=50, num_kmeans=4, clus_num_iters=20, pred_hidden=512`

### GraphDINO
- Adaptation of DINO (ViT) to graphs
- Student encoder + student head; teacher encoder + teacher head (EMA, no grad)
- `n_views` total views; the first `n_global_views` go to the teacher, all views go to the student
- **DINOLoss**: cross-entropy over every pair (teacher_i, student_j) with i ≠ j, averaged over the number of terms
  - Student output: `log_softmax(logits / student_temp)` — computed inside `DINOHead`
  - Teacher output: `softmax((logits − center) / teacher_temp_eff)` — computed inside `DINOHead`
- **Teacher temperature warmup**: `teacher_temp_eff` grows linearly from `warmup_teacher_temp`
  to `teacher_temp` over `warmup_teacher_temp_epochs` epochs. `DINOHead.set_epoch(epoch)` updates the value.
- **Center update** (EMA): `center = c_mom * center + (1 − c_mom) * mean(teacher_out)`
  Handled by `DINOHead.update_center()`, called from `post_step()`.
- **EMA teacher**: momentum grows on a cosine schedule from `ema_tau_base` to `ema_tau` over `total_steps`.
- **freeze_last_layer_epochs**: during the first N epochs, gradients of the prototype layer
  (`student_head.proto`) are zeroed out after backward — in `post_backward()`.
- Hyperparameters: `student_temp=0.1, teacher_temp=0.07, warmup_teacher_temp=0.04,
  warmup_teacher_temp_epochs=30, ema_tau=0.996, freeze_last_layer_epochs=1,
  n_views=2, n_global_views=2`

### VICReg
- Encoder + 3-layer projector `Projector(hidden_dim, hidden_dim*2, proj_dim)`
- Triple loss: invariance (MSE), variance (hinge on std), covariance (off-diagonal) — `VICRegLoss`
- Hyperparameters: `proj_dim=256, invariance=25.0, variance=25.0, covariance=1.0`

### Barlow Twins
- Encoder + 3-layer projector identical to VICReg's
- Cross-correlation matrix `C = (z1_norm.T @ z2_norm) / N`
- Loss: `sum((1−C_ii)²) + lambda * sum_{i≠j}(C_ij²)` — `BarlowTwinsLoss`
- `lambda_param=None` → defaults to `1/proj_dim`, computed inside `BarlowTwinsLoss`
- Hyperparameters: `proj_dim=256, lambda_param=None`

---

## ENCODERS (GNN BACKBONES)

All expose `forward(x, edge_index, batch, edge_attr=None) → Tensor[N, out_dim]`.
All are registered via `@ENCODERS.register("name")`.

**`EncoderConfig` — shared parameters:**
| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | str | — | 'gin', 'gcn', 'transformer' |
| `hidden_dim` | int | — | hidden/output dimension |
| `num_layers` | int | — | number of layers |
| `norm_type` | str | "batch" | 'batch', 'layer', 'none' — **uniform API across all encoders** |
| `pool` | bool | True | global_add/mean_pool for graph-level tasks |
| `drop` | float | 0.2 | dropout rate |
| `mlp_ratio` | float | 2.0 | hidden-dim multiplier for the internal MLP |
| `edge_dim` | int\|None | None | enables edge-feature-aware convolution (GINEConv / TransformerConv) |
| `node_emb_num_classes` | int\|None | None | replaces Linear with nn.Embedding for categorical features (ZINC: 28) |
| `edge_emb_num_classes` | int\|None | None | nn.Embedding for categorical edge features (ZINC: 4). Requires `edge_dim`. |

`EncoderConfig.build(in_channels)` uses `inspect.signature` to forward only the kwargs
accepted by the specific encoder — unsupported fields are silently ignored.

### GCN
- 2-layer GCNConv with norm + PReLU
- Optional weight standardization on the second layer
- Backward compatible: still accepts `batchnorm=True/False` and `layernorm=True/False`
- `reset_parameters()` exposed (used by BGRL for the target encoder)

### GIN (Graph Isomorphism Network)
- Stack of `GINLayer` with pre-norm + residual connections
- `GINLayer`: `norm(x)` → `GINConv(h)` → `ReLU` → `Dropout` → `x + h`
- `edge_dim != None` → uses `GINEConv` (edge-feature aware)
- If `edge_dim` is set but `edge_attr=None` at runtime, `GINEConv` receives zeros (doesn't crash)
- `node_emb_num_classes` → `nn.Embedding` instead of `Linear` for categorical features
- `edge_emb_num_classes` → `nn.Embedding` for categorical edge features before `GINEConv`

### Graph Transformer
- Stack of `TransformerBlock` (PyG's TransformerConv), pre-norm + FFN + residual
- Supports `edge_attr` via `TransformerConv(edge_dim=...)`
- Same `norm_type` API as GINEncoder (batch/layer/none)
- `node_emb_num_classes` and `edge_emb_num_classes` both supported

---

## AUGMENTATION SYSTEM

A torchvision-style composable system in `augmentation/`:
- `functional.py`: pure functions `(data, *, protected_nodes=None, **kwargs) → Data`
- `transforms.py`: callable classes wrapping those functions
- `compose.py`: `compose(data, aug_list, protected_nodes=None)` applies the list in sequence

Available operations:
| Name | Parameters | Description |
|---|---|---|
| `edge_drop` | p | drops edges with probability p |
| `edge_add` | p | adds random edges (fraction p of existing ones) |
| `feat_mask` | p | masks node features with probability p |
| `feat_noise` | std | adds Gaussian noise to features |
| `feat_shuffle` | p | swaps features between random nodes |
| `subgraph` | num_hops | extracts a k-hop subgraph from a random seed |
| `node_drop` | p | drops nodes; `protected_nodes` are never removed |

Two APIs:
1. **Class-based** (torchvision style): `EdgeDrop(p=0.2)(data)` — used with `MultiView`
2. **Registry-based** via `compose()`: `compose(data, [("edge_drop", {"p": 0.2})])` — supports `protected_nodes`

`MultiView(transforms, n_views)` accepts either API and generates n independent views.

---

## LOSS FUNCTIONS

All losses are standalone `nn.Module`s in `losses/`, independently testable.
All are registered in the `LOSSES` registry via `@LOSSES.register("name")`.

| Registry name | Class | Used by |
|---|---|---|
| `nt_xent` | `NTXentLoss` | GraphCL |
| `vicreg` | `VICRegLoss` | VICReg |
| `barlow_twins` | `BarlowTwinsLoss` | BarlowTwins |
| `cosine_regression` | `CosineRegressionLoss` | BGRL, AFGRL |
| — | `DINOLoss` | GraphDINO |

**Combinable losses:**
```python
from graphssl.losses import CombinedLoss

fn = CombinedLoss.from_config([
    {"name": "nt_xent", "weight": 0.7, "tau": 0.5},
    {"name": "vicreg",  "weight": 0.3, "invariance": 25.0, "variance": 25.0, "covariance": 1.0},
])
loss = fn(z1, z2)
```
Or programmatically: `CombinedLoss([(NTXentLoss(tau=0.5), 0.7), (VICRegLoss(), 0.3)])`.

---

## REGISTRY PATTERN

```python
from graphssl.registry import ENCODERS, LOSSES, AUGMENTS, OBJECTIVES, DATASETS, LOADERS

# Register a custom encoder:
@ENCODERS.register("my_gat")
class GATEncoder(nn.Module): ...

# Instantiate from the registry:
enc = ENCODERS.build("my_gat", in_channels=32, hidden_dim=64)
```

Available registries:
- `ENCODERS` — GNN backbones
- `HEADS` — projection/prediction heads
- `AUGMENTS` — augmentation transforms
- `LOSSES` — loss functions
- `LOADERS` — DataLoader factories
- `OBJECTIVES` — custom SSL objectives (placeholder for extensions)
- `DATASETS` — custom dataset builders (placeholder for extensions)

---

## KEY UTILITIES

### Parameter EMA (`utils/ema.py`)
`update_ema_params(student, teacher, tau)` — updates parameters and buffers (e.g. BatchNorm running stats).
Buffers are copied directly (not averaged).

### Schedulers (`utils/schedulers.py`)
- `CosineDecayScheduler(max_val, min_val, total_steps, warmup_steps)` — cosine decay with linear warmup
- `CosineEMAScheduler(ema_base, ema_end, total_steps)` — increasing cosine EMA momentum (BGRL/DINO)

### Pooling (`nn/pooling.py`)
`pool_graph_embeddings(node_embeddings, batch)` — `global_mean_pool` with a fallback when `batch=None`.

### extract_embeddings (`evaluation/visualization.py`)
Extracts embeddings from a model given a datamodule. Handles:
1. Graph-level: standard DataLoader
2. Node full-batch: a single forward pass over the whole graph
3. Node mini-batch: NeighborLoader with a `global_to_local` mapping to reconstruct order

### LogRegEvaluator (`evaluation/linear_probe.py`)
Pure-PyTorch linear evaluator. For multilabel targets (e.g. ogbg-molpcba) uses BCE + average precision.

### DataModule (`data/datamodule.py`)
Wraps a PyG dataset for two modes:
- Graph-level (`is_graph_level=True`): `DataLoader` over a dataset of graphs
- Node-level (`is_graph_level=False`): a single `Data` object with masks + `neighbor_loader()` for large graphs

---

## TRAINER AND CALLBACKS

`DINOTrainer` is generic: it works with any `BaseSSLModel`.

**Per-batch order:**
```
compute_loss → backward → clip_grad_norm → post_backward → optimizer.step → post_step
```
**Per-epoch order:**
```
on_epoch_start → batches → on_epoch_end
```

**Hooks implemented per model:**
| Model | `post_backward` | `post_step` | `on_epoch_start` | `on_epoch_end` |
|---|---|---|---|---|
| BGRL | — | EMA teacher | — | — |
| AFGRL | — | EMA teacher | — | — |
| GraphDINO | freeze last layer | EMA + center update | teacher temp warmup | epoch counter |

**Available callbacks:**
- `EmbeddingLoggerCallback` — saves embeddings every N epochs as `.pt`
- `LinearEvalCallback` — periodic linear probe during training
- `VisualizationCallback` — UMAP scatter plot (requires `pip install umap-learn matplotlib`)

---

## BENCHMARKS — Cora/CiteSeer/PubMed, ZINC AND ogbn-arxiv

### Cora / CiteSeer / PubMed (node-level, citation network, full-batch)
- Multi-seed script: `examples/benchmark_planetoid.py --dataset {Cora,CiteSeer,PubMed} --seeds 10`
- Single-run/smoke-test script: `examples/cora_bgrl.py`, config: `configs/cora_bgrl.yaml`
- BGRL, GIN-2L, `hidden_dim=256`, full-batch for 300 steps, public Planetoid split, mean±std over 10 seeds:

| Dataset | Linear probe (test) | KNN k=5 (test) |
|---|---|---|
| Cora | 62.39 ± 3.71 | 58.61 ± 3.76 |
| CiteSeer | 50.08 ± 1.58 | 43.07 ± 3.61 |
| PubMed | 69.18 ± 2.32 | 66.59 ± 2.88 |

- The config is deliberately lightweight/untuned (no per-dataset hyperparameter search) — its purpose is to validate that the whole pipeline (augmentation → EMA → both evaluation heads) works end-to-end, not to compete with the state of the art. See `paper.tex` for a methodological comparison against BGRL's original numbers (Appendix C, Table 7).

### ZINC (graph-level, molecular regression)
```yaml
encoder:
  name: gin
  node_emb_num_classes: 28  # 28 atom types
  edge_dim: 64
  edge_emb_num_classes: 4   # 4 bond types
  pool: true
```
- `in_channels` is ignored when `node_emb_num_classes` is set
- Script: `examples/zinc_bgrl.py`, config: `configs/zinc_bgrl.yaml`

### ogbn-arxiv (node-level, classification)
```yaml
encoder:
  name: gin
  pool: false               # node-level: no pooling
  hidden_dim: 256
```
- Continuous 128-dim features: no categorical embeddings
- Requires `NeighborLoader` for scalable training
- Script: `examples/ogbn_arxiv_bgrl.py`, config: `configs/ogbn_arxiv_bgrl.yaml`

---

## TESTS

```bash
pytest tests/ -v
```

| File | Model | Notes |
|---|---|---|
| `test_bgrl.py` | BGRL | teacher frozen, reset → different weights, EMA scheduler, step counter |
| `test_dgi.py` | DGI | W learnable, shuffle_nodes/shuffle_edges |
| `test_graphcl.py` | GraphCL | projector dim, NT-Xent ≥ 0 |
| `test_vicreg.py` | VICReg | 3-layer projector, loss ≥ 0 |
| `test_barlow_twins.py` | BarlowTwins | lambda default = 1/proj_dim |
| `test_afgrl.py` | AFGRL | **auto-skipped if faiss isn't installed** |
| `test_supervised.py` | Supervised | head dim, mini-batch crop |
| `test_graphdino.py` | GraphDINO | freeze last layer, teacher temp warmup, DINOTrainer hooks |
| `test_new_features.py` | — | edge_emb_num_classes, norm_type API, CombinedLoss |
| `test_evaluation.py` | — | LogRegEvaluator/KNNEvaluator, OGB-style 2D labels (`[N,1]`) equivalent to 1D |

---

## DEVELOPMENT TOOLING & CI

### Linting & formatting (`ruff`)
Config lives in `pyproject.toml` (`[tool.ruff]`): line-length 100, target py310, rule set
E/F/W/I/UP/B/C4. Typing-modernization rules (`UP006`/`UP007`/`UP035`/`UP037`/`UP045` —
`List`→`list`, `Optional[X]`→`X | None`, etc.) are deliberately excluded: the codebase
intentionally keeps the pre-PEP 604/585 typing style for consistency (see "Implementation
Notes" style below); modernizing it repo-wide is tracked as a separate, deliberate follow-up,
never mixed into an unrelated change. `__init__.py` re-exports are exempt from `F401`
(unused-import), since they exist specifically to re-export names for the public API surface.

### Type checking (`mypy`)
Configured in `pyproject.toml` (`[tool.mypy]`). `src/graphssl` is fully clean (0 errors) and
CI **blocks on regressions** — no `continue-on-error`, no swallowed exit code. Getting there
mostly meant resolving `nn.Module.__getattr__` being typed `Tensor | Module` (mypy can't
statically tell a submodule/buffer access from a tensor/parameter access unless the attribute
is explicitly annotated) via explicit attribute annotations, `getattr(..., None)` +
`callable()` instead of `hasattr()` + direct call, and `cast()` where a registry/container
return type is more generic than what the surrounding code actually guarantees. See
`CONTRIBUTING.md` for the worked examples and the pattern to follow for new code.

### Test coverage
`pytest-cov` measures coverage (`[tool.coverage.run]` / `[tool.coverage.report]` in
`pyproject.toml`); CI uploads the report to Codecov from the Python-3.12 matrix leg. Current
baseline is ~70%. Known gaps: `training/callbacks.py` (~27%) and `utils/schedulers.py`
(~53%) are largely untested — treat behavior there as less battle-tested than the core
model/loss code.

### Pre-commit hooks
`.pre-commit-config.yaml`: `ruff check --fix` + `ruff format`, plus standard hygiene hooks
(trailing-whitespace, end-of-file-fixer, check-yaml/toml, check-merge-conflict,
check-added-large-files). Enabled locally with `pre-commit install` after cloning.

### CI (`.github/workflows/tests.yml`)
Three jobs run on every push/PR to `main`:
- `lint` — `ruff check .` + `ruff format --check .`
- `typecheck` — `mypy src/graphssl`, blocking (see above)
- `pytest` — full matrix (Python 3.10/3.11/3.12) with coverage, uploading to Codecov from
  the 3.12 leg only

### git-blame hygiene
`.git-blame-ignore-revs` lists large, purely mechanical commits (e.g. the initial
`ruff format` baseline) so `git blame` stays useful; GitHub's web UI picks it up
automatically. Only add commits here that are behavior-preserving and mechanical — never one
that also changes behavior.

### Contributing
The full dev workflow (setup, pre-PR checklist, code-style rationale, how to add a
model/encoder/loss) lives in `CONTRIBUTING.md` — don't duplicate it here; update both if the
workflow itself changes.

---

## STRUCTURE

```
src/graphssl/
├── core/          ← ABC/Protocol: BaseModel, BaseSSLModel, BaseEncoder, BaseAugmentation, Callback, Registry
├── config/        ← schema.py (dataclass validation), load.py (load_config, build_model)
├── registry/      ← ENCODERS, HEADS, AUGMENTS, LOSSES, LOADERS, OBJECTIVES, DATASETS
├── encoders/      ← GCN, GIN, Transformer (auto-registered)
├── models/        ← DGI, GraphCL, BGRL, AFGRL, GraphDINO, VICReg, BarlowTwins, Supervised
├── losses/        ← nt_xent, dino, vicreg, barlow, regression, combined (CombinedLoss)
├── nn/            ← MLP, DINOHead, norm (weight_standardize), pooling
├── augmentation/  ← functional.py, transforms.py, compose.py
├── evaluation/    ← linear_probe, knn, visualization (extract_embeddings)
├── training/      ← trainer.py (DINOTrainer), callbacks.py
├── data/          ← pure-Python DataModule
└── utils/         ← ema.py (update_ema_params), schedulers.py, positive_miner.py
```

---

## ARCHITECTURAL PRINCIPLES

1. **Config-driven uniform API**: every model's `__init__(config: Dict, in_channels: int)`.
   The config is validated by a dataclass in `schema.py`. The encoder is always built from
   `cfg.encoder.build(in_channels)` — no encoder is ever built inline inside a model.

2. **Losses as standalone `nn.Module`s** in `losses/`, independently testable and combinable
   via `CombinedLoss`.

3. **Registry pattern** for every extensible component: `@ENCODERS.register("name")`.

4. **Hook-driven trainer**: `post_backward()`, `post_step()`, `on_epoch_start/end()` instead
   of subclassing the trainer.

5. **Layered separation**: `core → encoders/models/nn/augmentation/losses → evaluation →
   training → data`. No higher-level module imports from a lower one.

---

## IMPLEMENTATION NOTES

### graph-level vs. node-level
`self.graph_level` is derived from `cfg.encoder.pool` in the constructor — never passed as a
separate parameter. No model infers it at runtime from the batch.

### Mini-batch: protected nodes
In mini-batch node training, `batch.batch_size` gives the seed-node count.
The loss must be computed only on `z[:batch.batch_size]`.
`compose()` accepts `protected_nodes=torch.arange(batch.batch_size)` and propagates it to
every augmentation; `node_drop` uses this parameter to never remove seed nodes.

### BGRL vs. AFGRL: target-encoder initialization
- **BGRL**: `deepcopy(encoder)` + `reset_parameters()` — weights DIFFER from online (critical, paper Appendix B)
- **AFGRL**: `deepcopy(encoder)` without reset — weights are IDENTICAL to online at init

### GraphDINO: operation order
```
forward → loss → backward → clip_grad → post_backward (freeze last layer)
→ optimizer.step → post_step (EMA teacher + center update)
```

### GINLayer: edge_attr=None with edge_dim set
If `edge_dim` is set (so `GINEConv` is used) but `edge_attr=None` at runtime, the layer builds
a correctly-shaped zero tensor instead of crashing. This lets the same encoder be used with
and without edge features during training.

### LogRegEvaluator: multilabel
For ogbg-molpcba, labels are multilabel floats `[N, 128]` with NaNs.
Uses average precision score instead of accuracy.

### extract_embeddings: global_to_local mapping
In mini-batch mode, NeighborLoader batches arrive in a different order than the original
node ids. A map `global_to_local[node_id] = position in z_cpu` is used to reconstruct order.

---

## DISTRIBUTION

```toml
# pyproject.toml optional-dependencies
viz       = ["umap-learn", "matplotlib", "seaborn"]
benchmark = ["ogb", "pyyaml"]
full      = ["faiss-cpu", "umap-learn", "matplotlib", "seaborn", "ogb", "pyyaml"]
dev       = ["pytest", "pytest-cov", "build", "twine", "pyyaml", "ruff", "mypy", "pre-commit"]
```

**Published on PyPI**: `graphssl` v0.1.0 is live (`pip install graphssl`), released 2026-09-07.

**Release process:** every `v*.*.*` tag on GitHub automatically triggers the
`.github/workflows/publish.yml` workflow, which builds and publishes to PyPI via Trusted
Publishing. Note: Trusted Publishing requires a "pending publisher" to be registered on
pypi.org *before* a brand-new project's very first release — this isn't automatic. The
v0.1.0 tag's first publish attempt failed for exactly this reason; re-running the job after
registering the pending publisher on pypi.org succeeded without needing a new tag.

## HOW TO ADD A NEW MODEL (instructions for Claude)

### 1. Read first, write later
Read both versions (the reference paper and GraphSSL) and list the differences before
modifying any file.

### 2. Use the uniform public signature
Every model uses `__init__(config: Dict, in_channels: int)`, no exceptions.

For each new model:
1. Create `ModelNameConfig` in `src/graphssl/config/schema.py` with `__post_init__`
   validation and a `from_dict(cls, d: dict)` classmethod.
2. The constructor should only do: `cfg = ModelNameConfig.from_dict(config)` and then use `cfg.*`.
3. Use `cfg.encoder.build(in_channels)` — never build the encoder inline.
4. `graph_level` always derives from `cfg.encoder.pool`.

### 3. Dependency checklist
For every modified model, check:
- `losses/` — is the loss testable standalone?
- `utils/ema.py`, `utils/schedulers.py` — are EMA and the scheduler used correctly in `post_step()`?
- `augmentation/` — does `compose()` pass `protected_nodes`?
- `config/schema.py` — was a config dataclass added?
- `models/__init__.py`, `src/graphssl/__init__.py` — are exports updated?

### 4. Update this file (and CONTRIBUTING.md if the dev workflow itself changed)
Describe precisely: the config dataclass, the loss formula, EMA details, default
hyperparameters.
