# Getting Started

## Installation

```bash
pip install graphssl

# optional extras
pip install "graphssl[viz]"        # UMAP + matplotlib
pip install "graphssl[benchmark]"  # ogb + pyyaml (for ZINC / ogbn-arxiv)
pip install "graphssl[full]"       # faiss-cpu + viz + benchmark
```

!!! tip "No GPU?"
    `pip install graphssl` pulls in PyTorch's default CUDA-enabled build, which is large
    (1-2GB+ with bundled NVIDIA runtime libraries). On a CPU-only machine, install the CPU
    build first — it's a fraction of the size and installs much faster:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install graphssl
    ```

**For development:**
```bash
git clone https://github.com/Simocop7/graphssl.git
cd graphssl
pip install -e ".[dev]"
pre-commit install   # runs ruff on every commit
pytest tests/ -v
```
All 107 unit tests should pass. AFGRL tests are auto-skipped if `faiss-cpu` isn't installed.
See [Contributing](contributing.md) for the full dev workflow.

## A complete worked example

This is [`examples/cora_bgrl.py`](https://github.com/Simocop7/graphssl/blob/main/examples/cora_bgrl.py)
end to end — BGRL pretraining on Cora, then a linear-probe evaluation, in about 30 lines. It
runs on CPU in roughly a minute.

```python
from torch.optim import AdamW
from torch_geometric.datasets import Planetoid

from graphssl.config.load import build_model
from graphssl.data import DataModule
from graphssl.evaluation import LogRegEvaluator, extract_embeddings
from graphssl.training import DINOTrainer

# 1. Data — plain PyTorch Geometric, no custom wrapper
dataset = Planetoid(root="data/Cora", name="Cora")
data = dataset[0]
train_idx = data.train_mask.nonzero(as_tuple=True)[0]
val_idx = data.val_mask.nonzero(as_tuple=True)[0]
test_idx = data.test_mask.nonzero(as_tuple=True)[0]

# 2. DataModule — serves the graph to the training loop
dm = DataModule(data=data, is_graph_level=False,
                 train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
loader = dm.train_dataloader()  # full-batch: the whole graph as a single "batch"

# 3. Config -> model (BGRL: teacher-student with EMA, no negatives)
config = {
    "name": "bgrl",
    "encoder": {"name": "gin", "hidden_dim": 128, "num_layers": 2,
                "norm_type": "batch", "pool": False},
    "augment": [{"name": "edge_drop", "p": 0.5}, {"name": "feat_mask", "p": 0.2}],
    "pred_hidden": 128, "ema_tau": 0.99, "ema_tau_end": 1.0, "total_steps": 300,
}
model = build_model(config, in_channels=data.num_features)

# 4. Pretrain WITHOUT labels — compute_loss() never looks at data.y
optimizer = AdamW(model.student_parameters(), lr=5e-4, weight_decay=1e-5)
trainer = DINOTrainer(device="cpu")
losses = trainer.train(model, loader, optimizer, num_epochs=300)

# 5. Now that the encoder is trained, extract embeddings and evaluate WITH labels
z, y = extract_embeddings(model, dm, device="cpu")
results = LogRegEvaluator().evaluate(z, y, train_idx, val_idx, test_idx, num_classes=dataset.num_classes)
print(results["test_acc"])
```

A few things that aren't obvious from the code alone:

- `model.student_parameters()` — only the student/online encoder's weights go to the
  optimizer, never the target/teacher (it updates itself via EMA inside `post_step()`, not
  via backprop).
- The [`DINOTrainer`](api/training.md) is identical for every model. It calls
  `model.compute_loss(batch)` → `.backward()` → `model.post_step()`; what changes is what
  each model's `compute_loss` does internally, not the loop itself.
- `y` comes out of `extract_embeddings` alongside `z` — labels are never passed to the model,
  only to the final evaluation step.

## Loading from YAML instead

```python
from graphssl.config.load import load_config, build_model

cfg = load_config("configs/ogbn_arxiv_bgrl.yaml")
model = build_model(cfg, in_channels=128)
```

See [`configs/`](https://github.com/Simocop7/graphssl/tree/main/configs) for reference YAML
files and [`examples/`](https://github.com/Simocop7/graphssl/tree/main/examples) for the full
set of runnable scripts (Cora smoke test, multi-seed Planetoid benchmark, ogbn-arxiv
mini-batch training, ZINC with categorical features).

## Next steps

- Pick a [model](models/index.md) matching your use case.
- Read [Architecture](architecture.md) for the design principles that make every model behave
  the same way from the outside.
- See [Benchmarks](benchmarks.md) for validated results and how to reproduce them.
