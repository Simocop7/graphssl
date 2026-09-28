# GraphSSL

[![Tests](https://github.com/Simocop7/graphssl/actions/workflows/tests.yml/badge.svg)](https://github.com/Simocop7/graphssl/actions/workflows/tests.yml)
[![Docs](https://github.com/Simocop7/graphssl/actions/workflows/docs.yml/badge.svg)](https://simocop7.github.io/graphssl/)
[![codecov](https://codecov.io/gh/Simocop7/graphssl/branch/main/graph/badge.svg)](https://codecov.io/gh/Simocop7/graphssl)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A modular Python library for **Self-Supervised Learning on graphs**, built on PyTorch and PyTorch Geometric. No Lightning, no Hydra — clean, readable training loops you can step through with a debugger.

> **Status:** Alpha — all models train end-to-end and pass tests; all 7 SSL methods benchmarked on the citation networks under a shared protocol (see below), large-scale (OGB) benchmarks in progress.

---

## Supported Methods

| Method | Family | Paper |
|---|---|---|
| **DGI** | Mutual Information | Veličković et al., ICLR 2019 |
| **GraphCL** | Contrastive (NT-Xent) | You et al., NeurIPS 2020 |
| **BGRL** | Teacher-Student + EMA | Thakoor et al., ICLR 2022 |
| **AFGRL** | Augmentation-Free Mining | Lee et al., AAAI 2022 |
| **VICReg** | Variance-Invariance-Covariance | Bardes et al., ICLR 2022 |
| **Barlow Twins** | Cross-Correlation | Zbontar et al., ICML 2021 |
| **GraphDINO** | Self-Distillation | Adapted from Caron et al., ICCV 2021 |

Plus a **Supervised** baseline for comparison.

Encoder backbones: **GCN**, **GIN**, **Graph Transformer** — all swappable via one config line.

---

## Installation

```bash
pip install graphssl

# optional extras
pip install "graphssl[viz]"        # UMAP + matplotlib
pip install "graphssl[benchmark]"  # ogb + pyyaml (for ZINC / ogbn-arxiv)
pip install "graphssl[full]"       # faiss-cpu + viz + benchmark
```

> **No GPU?** `pip install graphssl` pulls in PyTorch's default CUDA-enabled build, which is
> large (1-2GB+ with bundled NVIDIA runtime libraries). On a CPU-only machine, install the
> CPU build first — it's a fraction of the size and installs much faster:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install graphssl
> ```

**Development:**
```bash
git clone https://github.com/Simocop7/graphssl.git
cd graphssl
pip install -e ".[dev]"
pre-commit install   # ruff on every commit
pytest tests/ -v
```

All unit tests should pass. AFGRL tests are auto-skipped if `faiss-cpu` is not installed.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev workflow (lint, type checking, adding a model).

**Full documentation:** [simocop7.github.io/graphssl](https://simocop7.github.io/graphssl/) — per-model pages, architecture, and an auto-generated API reference. Build it locally with:
```bash
pip install -e ".[docs]"
mkdocs serve   # http://127.0.0.1:8000
```

---

## Quick Start

Every model uses the same constructor API:

```python
from graphssl.models import BGRL
from graphssl.training import DINOTrainer
from torch.optim import AdamW

config = {
    "encoder": {
        "name": "gin",
        "hidden_dim": 256,
        "num_layers": 3,
        "norm_type": "batch",  # 'batch', 'layer', or 'none'
        "pool": False,         # False = node-level task
    },
    "augment": [
        {"name": "edge_drop", "p": 0.5},
        {"name": "feat_mask", "p": 0.1},
    ],
    "pred_hidden": 512,
    "ema_tau": 0.99,
    "ema_tau_end": 1.0,
    "total_steps": 0,
}

model = BGRL(config, in_channels=dataset.num_features)
optimizer = AdamW(model.student_parameters(), lr=1e-3)
trainer = DINOTrainer(device="cuda")
losses = trainer.train(model, loader, optimizer, num_epochs=1000)
```

Or load config from YAML:

```python
from graphssl.config.load import load_config, build_model

cfg   = load_config("configs/ogbn_arxiv_bgrl.yaml")
model = build_model(cfg, in_channels=128)
```

See `configs/` for reference YAML files and `examples/` for full benchmark scripts:

| Script | Dataset | Task | Notes |
|---|---|---|---|
| `examples/cora_bgrl.py` | Cora | Node classification (7 classes) | Full-batch, CPU-friendly — fastest smoke test |
| `examples/benchmark_planetoid.py` | Cora / CiteSeer / PubMed | Node classification, multi-seed | BGRL-only multi-seed demo; the cross-method numbers below come from `benchmarks/run_benchmark.py` |
| `examples/ogbn_arxiv_bgrl.py` | ogbn-arxiv | Node classification (40 classes) | `NeighborLoader` mini-batch training |
| `examples/zinc_bgrl.py` | ZINC-12k | Molecular property regression | Categorical node/edge embeddings |

### Benchmark results (citation networks)

All 7 SSL methods under one shared protocol: GIN-2L encoder (`hidden_dim=256`), full-batch training for 300 steps, AdamW (lr 5e-4), public Planetoid split. Linear-probe test accuracy (%), mean ± std over 10 seeds; best per dataset in bold:

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| DGI | 68.62 ± 2.77 | 51.10 ± 2.26 | 66.16 ± 2.83 |
| GraphCL | 78.06 ± 1.48 | 59.22 ± 2.50 | **80.11 ± 1.24** |
| BGRL | 64.18 ± 3.04 | 47.59 ± 2.22 | 68.67 ± 2.07 |
| AFGRL | 70.06 ± 2.64 | 47.50 ± 2.56 | 73.48 ± 1.27 |
| VICReg | 77.45 ± 1.78 | 61.44 ± 2.10 | 79.07 ± 1.62 |
| Barlow Twins | **78.13 ± 1.00** | **63.00 ± 1.31** | 77.82 ± 1.03 |
| GraphDINO | 61.54 ± 1.79 | 42.00 ± 3.13 | 70.80 ± 3.40 |

```bash
python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --model all --seeds 10
python benchmarks/render_tables.py      # rebuilds these tables from benchmarks/results/*.json
```

This is a deliberately untuned configuration — no per-method or per-dataset hyperparameter search — so it compares objectives at equal budget rather than reproducing each paper's best reported number. kNN results, per-seed metrics and full provenance (git commit, package versions) are in [the benchmarks docs](https://simocop7.github.io/graphssl/benchmarks/) and `benchmarks/results/`. ogbn-arxiv and OGB graph-level benchmarks (ogbg-molhiv, ogbg-molpcba) are next.

---

## Datasets with Categorical Features (ZINC)

ZINC has integer node and edge features. Use `node_emb_num_classes` and `edge_emb_num_classes`
to replace linear projections with `nn.Embedding`:

```python
config = {
    "encoder": {
        "name": "gin",
        "hidden_dim": 64,
        "num_layers": 4,
        "norm_type": "layer",
        "pool": True,
        "node_emb_num_classes": 28,  # 28 atom types
        "edge_dim": 64,
        "edge_emb_num_classes": 4,   # 4 bond types
    },
    ...
}
model = BGRL(config, in_channels=1)  # in_channels ignored when node_emb_num_classes is set
```

---

## Composable Losses

Mix and weight multiple objectives without touching model code:

```python
from graphssl.losses import CombinedLoss

loss_fn = CombinedLoss.from_config([
    {"name": "nt_xent",  "weight": 0.7, "tau": 0.5},
    {"name": "vicreg",   "weight": 0.3, "invariance": 25.0, "variance": 25.0, "covariance": 1.0},
])
loss = loss_fn(z1, z2)
```

---

## Extending the Library

All major components are registered by name and can be replaced via config:

```python
from graphssl.registry import ENCODERS, LOSSES, AUGMENTS

@ENCODERS.register("my_gat")
class GATEncoder(nn.Module):
    ...

# Then use it in any model config:
config = {"encoder": {"name": "my_gat", "hidden_dim": 128, ...}, ...}
```

Models expose `compute_loss(batch)`, `post_backward()`, and `post_step()` hooks,
so you can write your own training loop if needed:

```python
for batch in loader:
    loss = model.compute_loss(batch)
    loss.backward()
    model.post_backward()
    optimizer.step()
    model.post_step()
```

---

## Project Structure

```
src/graphssl/
├── core/          # BaseSSLModel, Callback, Registry, Protocol interfaces
├── config/        # Dataclass schemas + YAML loading (load_config, build_model)
├── registry/      # ENCODERS, HEADS, AUGMENTS, LOSSES, LOADERS, OBJECTIVES, DATASETS
├── encoders/      # GCN, GIN, Transformer (auto-registered)
├── models/        # All SSL models + Supervised baseline
├── losses/        # NT-Xent, DINO, VICReg, Barlow Twins, CosineRegression, CombinedLoss
├── augmentation/  # 7 transforms, compose(), MultiView
├── nn/            # MLP, Projector, Predictor, DINOHead, pooling
├── evaluation/    # LogRegEvaluator, KNNEvaluator, extract_embeddings
├── training/      # DINOTrainer + callbacks
├── data/          # DataModule (full-batch and NeighborLoader)
└── utils/         # update_ema_params, CosineDecayScheduler, CosineEMAScheduler, PositiveMiner
```

---

## Key Design Choices

- **Uniform API.** Every model is `ModelClass(config: Dict, in_channels: int)`. The config dict is validated by a dedicated dataclass in `schema.py`.
- **Hook-driven training.** The trainer calls `post_backward()` and `post_step()` at fixed points. Models implement these to freeze prototype gradients (GraphDINO) or update the EMA teacher (BGRL, AFGRL) — without subclassing the trainer.
- **Protected-node augmentations.** When using `NeighborLoader` for mini-batch training, seed nodes are shielded from `node_drop` so the loss remains valid.
- **Registry pattern.** All components are registered by name (`@ENCODERS.register("gin")`), so swapping architectures is a one-line config change.
- **Composable losses.** `CombinedLoss` enables weighted mixtures of any registered losses, configurable from YAML or programmatically.

---

## Acknowledgments

Developed at [NECSTLab](https://necst.it), Politecnico di Milano.
