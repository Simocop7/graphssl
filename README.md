# GraphSSL

[![Tests](https://github.com/Simocop7/graphssl/actions/workflows/tests.yml/badge.svg)](https://github.com/Simocop7/graphssl/actions/workflows/tests.yml)
[![Docs](https://github.com/Simocop7/graphssl/actions/workflows/docs.yml/badge.svg)](https://simocop7.github.io/graphssl/)
[![codecov](https://codecov.io/gh/Simocop7/graphssl/branch/main/graph/badge.svg)](https://codecov.io/gh/Simocop7/graphssl)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A modular Python library for **Self-Supervised Learning on graphs**, built on PyTorch and PyTorch Geometric, and the instrument of a controlled study of how graph SSL methods are evaluated. No Lightning, no Hydra — clean, readable training loops you can step through with a debugger.

> **Status:** Alpha — all eight SSL objectives train end-to-end with every backbone and pass the test suite. The library is the instrument of an [evaluation study](#evaluation-study) of how much of a graph SSL result comes from the objective, and how much from the backbone, the probe and the training settings. What changed between releases is in the [changelog](https://github.com/Simocop7/graphssl/blob/main/CHANGELOG.md).

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
| **GraphMAE** | Masked Feature Reconstruction | Hou et al., KDD 2022 |

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

> **macOS and AFGRL:** with the PyPI wheels, `torch`, `faiss-cpu` and `scikit-learn` each ship their own OpenMP library, and AFGRL ends with a segmentation fault. Make FAISS and scikit-learn use PyTorch's copy (reinstalling them undoes it):
> ```bash
> SP=$(python -c "import site; print(site.getsitepackages()[0])")
> for d in faiss/.dylibs sklearn/.dylibs; do
>   mv "$SP/$d/libomp.dylib" "$SP/$d/libomp.dylib.bundled"
>   ln -s "$SP/torch/lib/libomp.dylib" "$SP/$d/libomp.dylib"
> done
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
        "name": "gcn",         # 'gcn', 'gin' or 'transformer'
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
| `examples/ogbn_arxiv_bgrl.py` | ogbn-arxiv | Node classification (40 classes) | `NeighborLoader` mini-batch training |
| `examples/zinc_bgrl.py` | ZINC-12k | Molecular property regression | Categorical node/edge embeddings |

Each example prints its result next to the same encoder left untrained: part of a frozen
GNN's accuracy comes from the architecture alone.

### Evaluate, save, fine-tune

```python
from graphssl.config import load_model, save_model
from graphssl.core import pretrained_encoder
from graphssl.evaluation import LogRegEvaluator, effective_rank, extract_embeddings

z, y = extract_embeddings(model, datamodule)             # frozen embeddings
probe = LogRegEvaluator().evaluate(z, y, train_idx, val_idx, test_idx, num_classes=7)
probe["test_acc"], effective_rank(z)

save_model(model, "bgrl.pt", config, in_channels=dataset.num_features)
model = load_model("bgrl.pt")                            # rebuilt from the saved config
encoder = pretrained_encoder(model)                      # the encoder to fine-tune from
```

The [evaluation guide](https://simocop7.github.io/graphssl/evaluation/) has the full
script: the probes (linear, kNN, ridge for regression), the untrained-encoder reference,
and fine-tuning.

---

## Results

### Evaluation study

How much of a graph SSL result is the objective? The [study](https://simocop7.github.io/graphssl/study/) trains the eight objectives with a GCN, a GIN and a graph Transformer on six node-classification datasets (Cora, CiteSeer, PubMed, Amazon Computers, Amazon Photo, Coauthor CS) and on ogbn-arxiv, under one shared, untuned protocol, saves every model, and scores the same embeddings with the linear probes of public codebases, with kNN and on random splits, always against the same encoder left untrained and against the raw features.

![Raw features, the untrained encoder and the eight objectives on six datasets with three backbones](https://raw.githubusercontent.com/Simocop7/graphssl/main/benchmarks/paper/figures/fig_teaser.png)

- **A frozen representation is mostly its backbone.** Models that share a backbone are more similar (linear CKA 0.43–0.81) than models that share an objective (0.25–0.45), and the backbone explains 68–82% of the variance of the gain over the untrained encoder on four of the six datasets. An untrained GCN is already a good feature map: with a GCN the average objective adds 0.2 to 1.5 points on the Amazon and Coauthor graphs.
- **The probe decides how large the gain looks, in both directions.** With a GCN, the 100-step probe of DGI and PyG-SSL measures a gain of 1.5 points on CiteSeer where a probe regularized on validation measures 8.6, and 9.5 on Coauthor CS where it measures 1.5. Standardizing the embeddings and choosing the L2 penalty on validation, as `LogRegEvaluator` does, removes both biases.
- **What survives is the order within one backbone.** It holds across probes and machines, partly across learning rates, and not across backbones; GraphCL is the best objective in 13 of the 18 dataset–backbone cells.
- **Tests fixed in advance**: the small GCN gains hold against tuning the optimizer and learning rate and against ten times the training budget; of four label-free metrics, only the CKA distance from the untrained encoder identifies the objective that helped most better than chance, and narrowly.

Every number comes from the result files in `benchmarks/paper/`; the [study page](https://simocop7.github.io/graphssl/study/) lists the commands that regenerate them. A paper is in preparation.

### Benchmarks

Per-method tables of the library's benchmark runs, under the same shared protocol (citation networks with GCN and GIN encoders, linear probe and kNN, supervised references; ZINC with a frozen probe and fine-tuning), are in [the benchmarks docs](https://simocop7.github.io/graphssl/benchmarks/) and `benchmarks/results/`:

```bash
D="--dataset Cora CiteSeer PubMed --seeds 10"
python benchmarks/run_benchmark.py $D --model all supervised supervised_reg                 # GIN
python benchmarks/run_benchmark.py $D --model all supervised supervised_reg --encoder gcn
python benchmarks/render_tables.py --summary   # rebuilds the tables from benchmarks/results/*.json
```

**BGRL on ogbn-arxiv, with the protocol of its paper** (Thakoor et al., ICLR 2022: full-graph training for 10,000 steps, 3-layer GCN with layer normalization and weight standardization) reaches 71.32 ± 0.10% test accuracy over 3 seeds, against 71.64 ± 0.12 in the paper; the same encoder left untrained gives 69.53 ± 0.03 (paper: 68.94 ± 0.15). Close to the paper, not an exact match. It needs the predictor of the public BGRL implementation (`pred_norm: none`, `pred_activation: prelu`): with the library's default predictor, which has a batch normalization, the same training ends below the untrained encoder.

```bash
python -u benchmarks/reproduce_bgrl_arxiv.py --seeds 3   # about 4 h per seed on an NVIDIA A2
```

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
├── losses/        # NT-Xent, DINO, VICReg, Barlow Twins, CosineRegression, SCE, CombinedLoss
├── augmentation/  # 7 transforms, compose(), MultiView
├── nn/            # MLP, Projector, Predictor, DINOHead, GATDecoder, pooling
├── evaluation/    # LogRegEvaluator, KNNEvaluator, extract_embeddings
├── training/      # DINOTrainer + callbacks
├── data/          # DataModule (full-batch and NeighborLoader)
└── utils/         # update_ema_params, CosineDecayScheduler, CosineEMAScheduler, PositiveMiner

benchmarks/        # runners, analyses and result files; benchmarks/paper/ holds the evaluation study
```

---

## Key Design Choices

- **Uniform API.** Every model is `ModelClass(config: Dict, in_channels: int)`. The config dict is validated by a dedicated dataclass in `schema.py`.
- **Hook-driven training.** The trainer calls `post_backward()` and `post_step()` at fixed points. Models implement these to freeze prototype gradients (GraphDINO) or update the EMA teacher (BGRL, AFGRL) — without subclassing the trainer.
- **Protected-node augmentations.** When using `NeighborLoader` for mini-batch training, seed nodes are shielded from `node_drop` so the loss remains valid.
- **Registry pattern.** All components are registered by name (`@ENCODERS.register("gin")`), so swapping architectures is a one-line config change.
- **Composable losses.** `CombinedLoss` enables weighted mixtures of any registered losses, configurable from YAML or programmatically.
- **Measured, not assumed.** Every benchmark result records its configuration, git commit and package versions; models can be saved and scored again with another probe or split without retraining; every number has the untrained encoder as its reference.

---

## Acknowledgments

Developed at [NECSTLab](https://necst.it), Politecnico di Milano.
