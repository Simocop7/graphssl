# GraphSSL

A modular Python library for **Self-Supervised Learning on graphs**, built on PyTorch and
PyTorch Geometric. No Lightning, no Hydra — clean, readable training loops you can step
through with a debugger.

!!! info "Status: Alpha"
    All models train end-to-end and pass tests; citation-network benchmarks are validated
    (see [Benchmarks](benchmarks.md)), large-scale (OGB) benchmarks are in progress.

## Supported methods

| Method | Family | Paper |
|---|---|---|
| [DGI](models/dgi.md) | Mutual Information | Veličković et al., ICLR 2019 |
| [GraphCL](models/graphcl.md) | Contrastive (NT-Xent) | You et al., NeurIPS 2020 |
| [BGRL](models/bgrl.md) | Teacher-Student + EMA | Thakoor et al., ICLR 2022 |
| [AFGRL](models/afgrl.md) | Augmentation-Free Mining | Lee et al., AAAI 2022 |
| [VICReg](models/vicreg.md) | Variance-Invariance-Covariance | Bardes et al., ICLR 2022 |
| [Barlow Twins](models/barlow-twins.md) | Cross-Correlation | Zbontar et al., ICML 2021 |
| [GraphDINO](models/graphdino.md) | Self-Distillation | Adapted from Caron et al., ICCV 2021 |

Plus a [Supervised](models/supervised.md) baseline for comparison.

Encoder backbones: [GCN, GIN, Graph Transformer](encoders.md) — all swappable via one config
line.

## Why GraphSSL

Comparing SSL paradigms on graphs today usually means stitching together several papers'
official repositories — different frameworks, different encoders, different evaluation
protocols — before you can trust that a difference in reported accuracy reflects the method
and not the harness. GraphSSL puts bootstrapping (BGRL, AFGRL), redundancy-reduction (VICReg,
Barlow Twins), self-distillation (GraphDINO), contrastive (GraphCL) and mutual-information
(DGI) methods behind one identical construction and training API, so swapping one for another
is a one-line config change — see [Architecture](architecture.md) for how that's enforced, not
just documented.

## Quick start

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

Continue with [Getting Started](getting-started.md) for installation and a full worked
example, or jump straight to a [model page](models/index.md) if you already know which method
you want.

## Project

Developed at [NECSTLab](https://necst.it), Politecnico di Milano. Source on
[GitHub](https://github.com/Simocop7/graphssl), package on
[PyPI](https://pypi.org/project/graphssl/).
