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

**Checkpoints and fine-tuning:**
```python
from graphssl.config import save_model, load_model
from graphssl.core import pretrained_encoder
save_model(model, "bgrl.pt", cfg, in_channels=dataset.num_features)
model = load_model("bgrl.pt")            # rebuilt from the saved config, weights loaded
# pre-train -> fine-tune: same encoder config, task head on top
supervised = build_model({"name": "supervised", "encoder": cfg["encoder"], "task": "regression"},
                         in_channels=dataset.num_features, num_classes=1)
supervised.encoder.load_state_dict(pretrained_encoder(model).state_dict())
```
`pretrained_encoder(model)` is the encoder `forward()` embeds with: `encoder`, the online
encoder of BGRL / AFGRL, the teacher of GraphDINO. There was no way to save a model before
2026-10-08.

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
- `task: regression` (`SupervisedConfig`): L1 loss, `num_classes` = number of targets, float
  `data.y` (ZINC)
- `graph_level` derived from `cfg.encoder.pool`: embeddings mean-pooled per graph, loss on every graph
- Node-level mini-batch: crop to `[:batch_size]` seed nodes (build `NeighborLoader` with `input_nodes=train_idx`)
- Node-level full-batch: loss on `data.train_mask` only; **raises** if `train_mask` is missing —
  never trains silently on val/test labels (it used to, until the benchmark runner exposed it)
- Benchmark runner (neither is part of `all`; both save the head's own accuracy as
  `test_acc_head` next to linear probe/kNN). `--model all supervised supervised_reg` runs
  all nine; up to `49e92ee` `all` replaced the whole list and the supervised models were
  dropped without a word:
  - `--model supervised`: shared SSL protocol unchanged (no dropout, AdamW, last checkpoint) —
    overfits the 60–140 labels (head, Cora / CiteSeer / PubMed: 73.2 / 51.8 / 74.5 with GIN,
    72.3 / 57.2 / 73.8 with GCN)
  - `--model supervised_reg`: recipe adapted from Kipf & Welling (dropout 0.5, Adam lr 0.01 +
    L2 5e-4, best-validation checkpoint; recorded in the JSON hyperparameters, selected epoch
    per seed as `best_epoch`) — head 76.5 / 61.8 / 72.7 with GIN (helps on Cora and
    CiteSeer, hurts on PubMed), 78.0 / 65.4 / 74.6 with GCN
  - Both stay below the best SSL linear probe of their table (GCN 82.5 / 69.6 / 81.6, GIN
    81.3 / 65.6 / 80.4). The backbone is **not** what separates them from Kipf & Welling's
    GCN (81.5 / 70.3 / 79.0): with GCN the std. recipe is still 3.5 / 4.9 / 4.4 below. Its
    `best_epoch` is 0–2 on CiteSeer for all ten seeds and for 7 of 10 on Cora (15–100 with
    GIN): validation accuracy peaks after one to three Adam steps and never recovers. Not
    investigated — these are untuned references, don't present them as what supervision
    can reach

### DGI (Deep Graph Infomax)
- Discriminates real vs. corrupted embeddings via a discriminator with a learnable matrix W
  (`nn.Parameter`, initialized with `xavier_uniform_`)
- Corruption: node shuffling (permutation of x) or edge-destination shuffling
- Global summary `s = sigmoid(mean(h_pos))` for node-level, `global_mean_pool` for graph-level
- Discriminator: `pos_logits = (h_pos * W@s).sum(-1)`
- Loss: BCEWithLogitsLoss over [pos_logits, neg_logits] with labels [1,1,...,0,0,...]
- Hyperparameters: `corruption="shuffle_nodes", shuffle_ratio=1.0`
- **One forward pass** for the real and the corrupted graph (`_embed_pair`: nodes
  concatenated, the corrupted `edge_index` shifted by N, `batch` by the number of graphs).
  Two passes normalise each graph with its own BatchNorm statistics and the discriminator can
  read the pass from those alone — never go back to two passes with a batch-normalised
  encoder. Found by the ogbn-arxiv stress test: loss ≈ 0 from epoch 2, 100% discriminator
  accuracy even on nodes with no incoming edge in the sampled batch (79% of the batch nodes,
  real and corrupted identically distributed there; 50% with running statistics), kNN
  55.0 → 48.2 → 33.7 at epochs 10 / 20 / 100 against 53.3 untrained.
- **Mini-batch**: summary and loss on the seed nodes only (`[:batch_size]`).
- With both changes on ogbn-arxiv (one seed, 100 epochs): kNN 57.9 / 58.2 / 58.1 / 56.1 /
  51.5 at epochs 10 / 20 / 40 / 70 / 100. **The collapse is gone, a slow decline is not**:
  past epoch 40 the effective rank shrinks (4.8 → 1.6) and at 100 epochs the embeddings are
  below the untrained encoder again (kNN 51.5 vs 53.3, linear probe 44.2 vs 55.2). Cause not
  identified — don't state one. Until it is, stop DGI early in mini-batch (20–40 epochs).
- Full-batch is unaffected by either problem. One pass vs the committed two-pass benchmark
  row (10 seeds, kNN, Cora / CiteSeer / PubMed): 64.8 ± 2.4 / 44.2 ± 3.4 / 65.8 ± 3.0 vs
  65.4 ± 2.9 / 46.1 ± 3.6 / 64.5 ± 3.0, all within seed noise (a 5-seed check with the old
  probe had the linear accuracy 2–3 points lower with one pass on Cora and CiteSeer, about
  one std). No decline with budget: Cora kNN 64.8 / 63.1 / 65.6 and rank 35 / 43 / 44 at
  300 / 1000 / 3000 steps (CiteSeer 44.2 / 41.6 / 42.6).

### GraphCL (Graph Contrastive Learning)
- NT-Xent loss over 2 augmented views
- Encoder + 2-layer projector `Projector(hidden_dim, hidden_dim, proj_dim)`
- `protected_nodes` passed to `compose()` for seed nodes in mini-batch node training
- `NTXentLoss(tau, chunk_size)`: when `2N > chunk_size` the `[2N, 2N]` similarity matrix is
  processed in row chunks under gradient checkpointing — exact same value/gradients, peak
  memory `O(chunk_size · 2N)`. Needed for full-batch PubMed (unchunked OOMs a 16 GB GPU;
  chunked peaks at ~1.9 GiB).
- Hyperparameters: `proj_dim=128, tau=0.5, loss_chunk_size=4096`

### BGRL (Bootstrapped Graph Representation Learning)
- Teacher-student with an EMA update on the target encoder
- Online encoder (student) + predictor MLP `Predictor(hidden_dim, pred_hidden, hidden_dim)`; target encoder (teacher, no grad)
- Predictor layout (`BGRLConfig` and `AFGRLConfig`): `pred_norm` ('batch' | 'none') and
  `pred_activation` ('relu' | 'prelu'). Default Linear → BatchNorm → ReLU → Linear;
  `pred_norm: none` + `pred_activation: prelu` = Linear → PReLU → Linear, the predictor of
  the reference implementation (nerdslab/bgrl). On ogbn-arxiv with the paper's layer-norm
  encoder the default one makes BGRL lose accuracy — see the reproduction below.
- Loss: `2 - cos(pred1, target2) - cos(pred2, target1)` — `CosineRegressionLoss(symmetric=True)`
  - Correct call: `loss_fn(p1, t1, p2, t2)` → internally the loss pairs t2 with p1 and t1 with p2
- EMA update of the target encoder in `post_step()` via `update_ema_params()`
- Momentum τ cosine-annealed from `ema_tau` to `ema_tau_end` over `total_steps` — `CosineEMAScheduler`.
  If `total_steps=0`, τ is fixed.
- Target encoder: `deepcopy(encoder)` + `reset_parameters()` — weights intentionally differ
  from the online encoder (critical for convergence, Appendix B of the BGRL paper)
- Hyperparameters: `ema_tau=0.99, ema_tau_end=1.0, total_steps=0, pred_hidden=512,
  pred_norm="batch", pred_activation="relu"`

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
  num_centroids=50, num_kmeans=4, clus_num_iters=20, kmeans_threads=8, knn_chunk_size=4096,
  pred_hidden=512, pred_norm="batch", pred_activation="relu"` (predictor layout as in BGRL)
- `knn_chunk_size`: the top-k search (`topk_similar` in `utils/positive_miner.py`) computes the
  N×N similarity in row chunks — identical neighbors, peak memory `O(chunk · N)` instead of
  `O(N²)` (~115 GB on ogbn-arxiv unchunked). Time is still `O(N² d)` per step and mining is
  full-graph only, so AFGRL stays slow on very large graphs. Tested without faiss in
  `tests/test_positive_miner.py`.
- `kmeans_threads` caps FAISS's OpenMP threads inside `PositiveMiner` (restored after each
  call; `None` = FAISS default, i.e. all cores). FAISS's all-cores default made AFGRL ~50×
  slower on a 48-core VM (5.7 s vs 0.1 s per Cora step) — never remove the cap without
  re-measuring on a many-core machine.

### GraphDINO
- Adaptation of DINO (ViT) to graphs
- Student encoder + student head; teacher encoder + teacher head (EMA, no grad)
- `n_views` total views; the first `n_global_views` go to the teacher, all views go to the student
- **DINOLoss**: cross-entropy over every pair (teacher_i, student_j) with i ≠ j, averaged over the number of terms
  - Student output: `log_softmax(logits / student_temp)` — computed inside `DINOHead`
  - Teacher output: `softmax((logits − center) / teacher_temp_eff)` — computed inside `DINOHead`
- **Teacher temperature warmup**: `teacher_temp_eff` grows linearly from `warmup_teacher_temp`
  to `teacher_temp` over `warmup_teacher_temp_epochs` epochs. `DINOHead.set_epoch(epoch)` updates the value.
- **Center update** (EMA): `center = c_mom * center + (1 − c_mom) * mean(teacher_logits)`, over the
  **raw** teacher logits (`DINOHead.prototype_logits`), as in reference DINO — the center is
  subtracted from logits, so it must live in logit space. Handled by `DINOHead.update_center()`,
  called from `post_step()`. (Until `20cce09` it was fed the post-softmax probabilities by
  mistake; rerunning the Planetoid benchmark after the fix moved every number by < 1 std.)
- **EMA teacher**: momentum grows on a cosine schedule from `ema_tau_base` to `ema_tau` over `total_steps`.
- **freeze_last_layer_epochs**: during the first N epochs, gradients of the prototype layer
  (`student_head.proto`) are zeroed out after backward — in `post_backward()`.
- Hyperparameters (code defaults, `HeadConfig`/`GraphDINOConfig`): `student_temp=0.1,
  teacher_temp=0.07, warmup_teacher_temp=0.04, warmup_teacher_temp_epochs=30,
  center_momentum=0.9, ema_tau=0.996, ema_tau_base=0.9, total_steps=0,
  freeze_last_layer_epochs=1, n_views=2, n_global_views=2, norm_last_layer=False`.
  - `ema_tau_base=0.9` and `teacher_temp=0.07` differ from reference DINO (0.996 / 0.04) on
    purpose: see "Stability" below. `from_dict` falls back to `min(0.9, ema_tau)` for the
    base, so a config that only lowers `ema_tau` stays valid.
  - With `total_steps=0` the momentum stays fixed at `ema_tau_base` (0.9). The ablations
    always set `total_steps` to the run length; the fixed-0.9 path is only spot-checked
    (local CPU, Cora, 300 steps: 71.6 ± 3.3 vs 74.0 ± 2.9 with the schedule).
  - **The benchmark runner does not use these defaults**: `run_benchmark.py` pins reference
    DINO's values (`ema_tau_base=0.996`, `head.teacher_temp=0.04`), so the benchmark row stays
    untuned and every committed JSON stays reproducible. `--ema-tau 0.9 --set
    head.teacher_temp=0.07` runs the library defaults.
- `norm_last_layer` (`HeadConfig`, off by default = unchanged behavior): fixes the prototypes'
  weight-norm scale g at 1, so logits are cosines in [-1, 1] (DINO's `norm_last_layer`).
  Without it g is trainable (weight_norm initializes it to the row norms: ~0.57 with the
  benchmark's 64-d bottleneck) and grows during training.
- **Stability** (why the defaults changed): with reference DINO's values GraphDINO gets
  *worse* the longer it trains. 5 seeds, GIN, linear probe at 300 → 1000 → 3000 steps, Cora |
  CiteSeer (`benchmarks/ablations/graphdino_stability`, `--ema-tau 0.9` alone in
  `teacher_student`); tables in `docs/benchmarks.md` (Sensitivity):
  - reference values: 63.3 → 50.0 → 38.9 | 42.2 → 33.3 → 28.8 (below the untrained GIN)
  - `--ema-tau 0.9` (0.9 → 0.996 over the run): 66.5 → 62.9 → 62.9 | 43.7 → 44.0 → 42.7
  - `--ema-tau 0.9` + `head.teacher_temp=0.07` (**the library defaults**): 73.5 → 68.1 → 67.0 |
    47.8 → 45.6 → 45.0 — best in every cell measured, also PubMed at 300 steps (75.6 vs 70.7)
    and with GCN (Cora 78.3 → 75.4 vs 73.3 → 69.1)
  - `head.teacher_temp=0.07` alone, 3000 steps: 54.2 | 36.9. `head.norm_last_layer=true`
    alone: 37.4 | 27.9, no effect (a single-seed local diagnostic had it helping: don't
    decide on one seed)
  - Still open: a milder decline with training length (Cora 73.5 → 67.0), larger seed std
    (up to 4.7 on CiteSeer), and with GCN on CiteSeer GraphDINO stays below the untrained
    encoder (54.5 vs 57.0)

  Local single-seed diagnostics tie the decline to the teacher over-sharpening (output
  entropy → 0); gradient clipping and more prototypes didn't help there. The settings were
  chosen on test accuracy of Cora/CiteSeer: say so wherever the tuned numbers are reported.

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
| `weight_standardization` | bool | False | GCN only: standardize the conv weights of every layer after the first |
| `pool` | bool | True | graph-level task: models pool the node embeddings per graph |
| `readout` | str | "mean" | 'mean', 'sum', 'max' — the pooling used when `pool=True` (read by the models, not by the encoder) |
| `drop` | float | 0.2 | dropout rate |
| `mlp_ratio` | float | 2.0 | hidden-dim multiplier for the internal MLP |
| `edge_dim` | int\|None | None | enables edge-feature-aware convolution (GINEConv / TransformerConv) |
| `node_emb_num_classes` | int\|None | None | replaces Linear with nn.Embedding for categorical features (ZINC: 28) |
| `edge_emb_num_classes` | int\|None | None | nn.Embedding for categorical edge features (ZINC: 4). Requires `edge_dim`. |

`EncoderConfig.build(in_channels)` uses `inspect.signature` to forward only the kwargs
accepted by the specific encoder — unsupported fields are silently ignored.

### GCN
- `num_layers` × (GCNConv → norm → PReLU), 2 by default. Until this was fixed the encoder
  always had two layers: `num_layers` from the config was dropped on the way.
- `weight_standardization` (`EncoderConfig` field, off by default): standardizes the GCNConv
  weights of every layer after the first before each forward, as BGRL does on ogbn-arxiv.
  It used to do nothing — the helper looked for a direct `weight` parameter, GCNConv keeps
  its weight in `lin.weight` — and could not be set from a config.
- State-dict keys are `convs.N` / `norms.N` / `acts.N` (0.1.0: `conv1`, `norm1`, …).
- **BatchNorm momentum 0.1** (`batchnorm_mm`, constructor only; PyTorch's default, as GIN
  and Transformer). It was 0.01 until 2026-10-09: after n forward passes the running
  statistics keep `0.99^n` of their initial values (variance 1) — 4.9% after 300 — while
  the batch variance after the first GCN layer is ~7e-3 on Cora and 1.4e-4 on PubMed, so
  the running variance was 8× / 340× too large and eval mode did not reproduce what
  training saw. Training itself never depended on it (every model, target encoders
  included, runs in train mode: batch statistics). Measured with GCN, 300 steps, 3 seeds,
  eval mode vs batch statistics (= eval mode after the fix): supervised head on PubMed
  48.7 / 58.3 / 65.0 → 73.6 / 73.3 / 74.8; kNN on Cora, AFGRL 67.4 → 73.0 and DGI 65.4 →
  71.3 (AFGRL on CiteSeer: no change, 56.0 vs 55.7); linear probe within ±0.7 everywhere (it refits on whatever it is given); BGRL and
  Barlow Twins on Cora within noise — two views = 600 passes, residual 0.24%. **Every GCN
  kNN, effective-rank and supervised-head number measured before the fix at ≤ 300 steps is
  affected** (single-pass models most: Supervised, DGI, AFGRL), including the GCN JSONs of
  the 2026-10-09 run at `49e92ee`: rerun the GCN rows from a commit with the fix before
  rendering them. The untrained encoder (`--epochs 0`) is unchanged: at initialization
  BatchNorm in eval mode is the identity, with any momentum.
  `tests/test_new_features.py::test_eval_statistics_follow_a_short_training` fails with 0.01.
  The benchmark confirms it on 10 seeds (same seeds and training, `49e92ee` with 0.01 →
  `acae6df` with 0.1; the old JSONs are in `benchmarks/ablations/gcn_bn_momentum_0.01`):
  supervised head 66.9 → 72.3 / 49.8 → 57.1 / 49.7 → 73.8 on Cora / CiteSeer / PubMed (std.
  recipe 77.8 → 78.0 / 63.6 → 65.4 / 68.0 → 74.6); kNN of DGI 64.9 → 73.5 / 54.0 → 59.3 /
  69.8 → 72.6 and of AFGRL on Cora 68.9 → 72.9; every two-view method within ±0.8 kNN and
  every linear probe within ±0.5 (supervised up to 0.9).
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
| `feat_mask` | p | masks feature **columns** with probability p (the same columns on every node) |
| `attr_mask` | p, mask_value | replaces the features of a fraction p of the **nodes** with `mask_value` (molecules: mask token = one index past the atom types, with `node_emb_num_classes` one larger) |
| `feat_noise` | std | adds Gaussian noise to features |
| `feat_shuffle` | p | swaps features between random nodes |
| `subgraph` | num_hops | extracts a k-hop subgraph from a random seed; **raises** if given `protected_nodes` (it cannot keep the seeds of a mini-batch in place) |
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
`pool_graph_embeddings(node_embeddings, batch, readout="mean")` — mean / sum / max pooling per
graph (`batch=None`: one graph). Every model passes `cfg.encoder.readout`. The mean hides
the size of a graph; the sum keeps it.

### extract_embeddings (`evaluation/visualization.py`)
Extracts embeddings from a model given a datamodule (pass the `DataModule`, not a loader).
Handles:
1. Graph-level: standard DataLoader
2. Node full-batch: a single forward pass over the whole graph. It moves a **shallow copy** of
   the graph to the device: PyG's `Data.to()` mutates in place, and moving the datamodule's
   own graph to the GPU made a `NeighborLoader` built on it crash in its worker processes
   ("Cannot re-initialize CUDA in forked subprocess") — found by the ogbn-arxiv stress test.
3. Node mini-batch: NeighborLoader with a `global_to_local` mapping to reconstruct order

Returns CPU tensors: move them explicitly before training a probe on GPU.

### KNNEvaluator (`evaluation/knn.py`)
Cosine kNN, majority vote. `chunk_size=4096` scores the evaluated split in row chunks with
identical predictions; the full `[|split|, |train|]` similarity (~17 GB for the ogbn-arxiv
test split) is never built.

### LogRegEvaluator (`evaluation/linear_probe.py`)
L2-regularised logistic regression on frozen embeddings, pure PyTorch:
`LogRegEvaluator(*, weight_decay=DEFAULT_WEIGHT_DECAYS, max_iter=5000, standardize=True, multilabel=False)`
(keyword-only: 0.1.0's positional `lr, epochs` must fail loudly).
- Features are standardized with the **training rows'** statistics. The classifier starts from
  zero and is fitted full-batch with L-BFGS (history 100, strong Wolfe) on
  `mean CE + weight_decay / 2 * ||W||²`: no seed, no learning rate. Same embeddings, same
  result (CPU and GPU agree within 0.1 points).
- `weight_decay`: one value or several (default 10 … 1e-6 by decades). The value with the best
  **validation** metric is used (the strongest on ties); fits go strongest → weakest, each
  starting from the previous solution. Planetoid selects 1e-4 … 10, ogbn-arxiv 1e-6.
- Returns `val_acc` / `test_acc` (`val_ap` / `test_ap` when multilabel), the selected
  `weight_decay`, and `converged` (False when the selected fit used up `max_iter`).
- **Why it replaced the 0.1.0 probe** (random init, 100 Adam steps on raw features): on
  ogbn-arxiv the same embeddings scored 45–54% depending on the seed, and 8–20 points below
  convergence — most for low-rank embeddings (untrained GIN 35 → 55, BGRL after 10 epochs
  46 → 57, GraphCL 53 → 62), i.e. against the teacher-student methods. On Planetoid the old
  probe was within about ±2.6 points of the new one (seed range up to 4.8 on BGRL).
- **Cost.** The weak end of the grid needs thousands of L-BFGS iterations on low-rank
  embeddings. ogbn-arxiv (91k training nodes): under 90 s per evaluation on the idle A2,
  6–10 min when the GPU is shared, 5 min on the CPU. Planetoid: 1–5 s on the CPU with ≤ 4
  threads, but 20 s with the VM's 48 threads and 30–40 s on its busy GPU — the steps are all
  per-operation overhead. Hence `run_benchmark.py` fits it on the CPU with
  `PROBE_THREADS = 4`, and the stress script only before and after training.
- Measured and dropped — don't retry without new evidence: more Adam steps (unregularised,
  it overfits 60–140 labels: Barlow Twins on CiteSeer 61 → 54; and 1000 steps are still short
  of convergence on arxiv); PCA whitening as the feature map (Cora 46%); L-BFGS history 10
  (stops at a worse objective); scaling the variables by the feature eigenvalues × class
  frequencies (7 iterations instead of 190 at strong L2, no gain at the weak end, where the
  time goes). A faster solver would not change any number: the optimum is unique.

### RidgeEvaluator (`evaluation/regression_probe.py`)
`RidgeEvaluator(*, weight_decay=DEFAULT_RIDGE_WEIGHT_DECAYS, standardize=True)` — the probe
for **regression** targets (ZINC). Closed-form ridge on standardized embeddings (training
rows' statistics), in float64, one eigendecomposition shared by every L2 strength (default
1e3 … 1e-6): deterministic, no optimiser. The strength with the lowest **validation MAE** is
used (the strongest on ties). Returns `val_mae` / `test_mae`, `val_rmse` / `test_rmse`,
`weight_decay`. Targets `[N]` or `[N, T]`, no NaN.

### encode (`core/encoder.py`)
`encode(encoder, data)` runs an encoder with `x, edge_index, batch` and, when the graph has
them, `edge_attr`. **Every model reaches its encoder through it.** Until 2026-10-08 every
model called `encoder(x, edge_index, batch)`: edge-aware encoders fall back to zeros when
`edge_attr` is missing, so the bond types of molecules were silently ignored in training and
in `extract_embeddings`. `tests/test_model_encoder_matrix.py` now checks, for every model,
that the encoder receives them and that changing them changes the embeddings.

### effective_rank (`evaluation/diagnostics.py`)
`effective_rank(z)` = `exp(entropy)` of the normalized singular values of the centered
embeddings (Roy & Vetterli 2007): 1 = all variance along one direction, `min(N, D)` = spread
evenly, 0 = no variance. Exposes dimensional collapse that accuracy can hide; the benchmark
runner records it for every seed (`eff_rank`).

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
- `LinearEvalCallback` — periodic linear probe during training; `evaluator=` takes a
  configured `LogRegEvaluator` (the default one fits 8 classifiers: minutes on a large graph)
- `VisualizationCallback` — UMAP scatter plot (requires `pip install umap-learn matplotlib`)

---

## BENCHMARKS — Cora/CiteSeer/PubMed, ZINC AND ogbn-arxiv

### Cora / CiteSeer / PubMed (node-level, citation network, full-batch)
- Cross-method runner: `benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --model all --seeds 10`
  → one JSON per (dataset, model) in `benchmarks/results/`; `benchmarks/render_tables.py [--latex]`
  rebuilds the tables. **Never retype benchmark numbers by hand** — regenerate them from the JSONs.
- `render_tables.py` keeps the latest run per (dataset, **encoder**, model): a run with
  `--encoder gcn` in `benchmarks/results/` is rendered as its own table ("Cora (GCN
  encoder)"), it does not replace the GIN rows.
- BGRL-only demo: `examples/benchmark_planetoid.py`; smoke test: `examples/cora_bgrl.py` (`configs/cora_bgrl.yaml`)
- Shared protocol (all 7 methods), run with **two backbones**: GCN-2L or GIN-2L,
  `hidden_dim=256`, batch norm, no dropout, full-batch 300 steps, AdamW lr 5e-4 / wd 1e-5,
  `edge_drop` 0.5 + `feat_mask` 0.2 (augmentation-based methods), public Planetoid split,
  10 seeds, current probe. Run 2026-10-09 on the NECSTLab VM (NVIDIA A2 16 GB), clean tree:
  GIN and the untrained encoders at `49e92ee`, GCN at `acae6df` (after the BatchNorm-momentum
  fix, see GCN). About 3.5 h per backbone.
- `render_tables.py --summary [--latex]` prints the tables below (methods × datasets per
  encoder, linear and kNN, best SSL method in bold, supervised heads, and the untrained
  encoder from `benchmarks/ablations/untrained_encoder`). README, `docs/benchmarks.md` and
  `paper.tex` were regenerated from it on 2026-10-10.
- **What is still from the 0.1.0 probe**: the two ablation directories (`teacher_student`,
  `graphdino_stability`, 5 seeds) and everything quoted from them — the GraphDINO
  "Stability" numbers above, the "Sensitivity" sections of the docs and the paper — plus
  the ogbn-arxiv stress test's linear column. They are consistent with each other, not with
  the tables below. Rerun cost: ≈ 10 h.
- Results saved since record the probe's settings (`hyperparameters.linear_probe`) and, per
  seed, `probe_weight_decay` / `probe_converged`. `render_tables.py` and `render_ablation.py`
  print the probe protocol under each table, and say so when its rows mix protocols — a JSON
  without `linear_probe` is a 0.1.0-probe result.
- Linear-probe test accuracy (%), mean ± std over 10 seeds (kNN and the supervised heads in
  `docs/benchmarks.md`). **GCN:**

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| *Untrained encoder* | 73.17 ± 0.99 | 57.69 ± 1.47 | 74.07 ± 1.60 |
| DGI | 78.12 ± 1.22 | 65.07 ± 1.44 | 78.60 ± 1.35 |
| GraphCL | **82.46 ± 1.08** | **69.59 ± 1.00** | 81.28 ± 0.77 |
| BGRL | 80.82 ± 1.23 | 67.05 ± 0.89 | 79.85 ± 1.10 |
| AFGRL | 79.18 ± 1.26 | 62.63 ± 2.01 | 78.43 ± 1.00 |
| VICReg | 81.06 ± 1.10 | 66.78 ± 1.45 | **81.59 ± 0.70** |
| Barlow Twins | 81.65 ± 0.93 | 68.95 ± 1.33 | 81.24 ± 0.72 |
| GraphDINO | 77.53 ± 1.22 | 60.16 ± 1.40 | 76.97 ± 2.08 |
| *Supervised* † | 72.67 ± 0.85 | 56.35 ± 1.52 | 74.12 ± 1.01 |
| *Supervised, std. recipe* † | 78.89 ± 1.42 | 64.84 ± 3.84 | 75.45 ± 1.12 |

  **GIN:**

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| *Untrained encoder* | 41.70 ± 2.44 | 36.06 ± 1.57 | 56.60 ± 1.50 |
| DGI | 65.43 ± 3.03 | 49.18 ± 2.86 | 66.90 ± 3.87 |
| GraphCL | **81.31 ± 1.14** | 65.17 ± 1.44 | **80.36 ± 0.82** |
| BGRL | 63.77 ± 1.60 | 47.68 ± 2.30 | 68.52 ± 2.57 |
| AFGRL | 70.75 ± 1.75 | 49.32 ± 3.02 | 74.51 ± 1.11 |
| VICReg | 77.36 ± 1.88 | 63.52 ± 2.10 | 79.08 ± 1.27 |
| Barlow Twins | 79.33 ± 0.75 | **65.60 ± 1.62** | 77.67 ± 1.49 |
| GraphDINO | 62.53 ± 1.91 | 43.34 ± 3.27 | 70.38 ± 3.23 |
| *Supervised* † | 71.46 ± 1.83 | 51.21 ± 3.73 | 71.49 ± 1.70 |
| *Supervised, std. recipe* † | 75.83 ± 2.33 | 59.32 ± 4.51 | 71.96 ± 1.90 |

- † Supervised references (see the Supervised section): trained on labels, not ranked.
- Deliberately untuned (no per-method/per-dataset search): it compares objectives at equal
  budget, it doesn't compete with each paper's best number. What the two tables say:
  - **With GCN every objective learns and they are close**: all seven are above the
    untrained GCN on all three datasets (+2.5 to +11.9); GraphCL, Barlow Twins, VICReg and
    BGRL are within 1.6 / 2.8 / 1.7 points of each other; AFGRL, DGI and GraphDINO are 3 to
    9 below the best of each column.
  - **With GIN they separate**: GraphCL, Barlow Twins and VICReg lose 1 to 4 points against
    their GCN numbers; BGRL, AFGRL, GraphDINO and DGI are 6 to 22 points below the best.
    BGRL's effective rank is 8–9 of 256 with GIN, 148–190 with GCN. **BGRL trailing is a
    property of the GIN backbone, not of the method** — don't write "teacher-student methods
    trail" without naming the backbone.
  - **Always read a row against the untrained encoder of its table**: the untrained GCN is
    at 73.2 / 57.7 / 74.1, the untrained GIN at 41.7 / 36.1 / 56.6. Most of the GCN table is
    the backbone.
  - Against the BGRL paper's Table 7 (GRACE 83.0 / 71.6 / 86.1, BGRL 83.8 / 72.3 / 86.0 —
    20 random splits and a tuned configuration, not the public split): untuned BGRL + GCN
    is 3.0 / 5.3 / 6.2 points below; with GIN it was 20 / 25 / 18.
  - The 0.1.0 probe had hidden part of this: BGRL + GCN on CiteSeer was 57.8 with it (a
    5-seed ablation) and is 67.1 now, while the untrained GCN barely moved (57.0 → 57.7).
  - From the 0.1.0-probe ablations (5 seeds, GIN), still to be rerun: from 300 to 3000 steps
    Barlow Twins is flat, BGRL gains (Cora 64.5 → 71.2 → 73.5) but stays below, GraphDINO
    with the reference values declines (see GraphDINO).

  Ruled out: GraphDINO's center bug (rerun after the fix at `20cce09`: every number moved
  < 1 std).
- **Alternatives tried on 2026-10-08, taken from the earlier Lightning/Hydra implementation
  of the same models** (diagnostics from an uncommitted tree: 3 seeds, Cora | CiteSeer, GIN,
  300 steps, current probe, linear accuracy — measure again before quoting):
  - *Its GIN block* (BatchNorm after the input projection, inside every GIN MLP and before
    the output, GELU, trainable eps) — **worse than ours for every method**: BGRL 65.0 →
    52.8 | 48.0 → 36.0, Barlow Twins 79.1 → 69.4 | 64.8 → 58.4, GraphDINO 63.2 → 41.5 |
    43.7 → 33.7, DGI 68.5 → 58.0 | 50.7 → 48.6. It has a higher effective rank (BGRL 9 → 40)
    and lower accuracy: rank alone is not the goal. Keep the current block.
  - *Milder augmentation* (`edge_drop` 0.2 instead of 0.5, `feat_mask` 0.2): helps BGRL
    (65.2 → 67.9 | 46.3 → 52.0, rank 9 → 17), hurts Barlow Twins (79.4 → 76.6 | 64.1 →
    61.6), neutral for GraphCL (82.0 → 81.0 | 65.0 → 64.9). Method-dependent: not a shared
    default.
  - *DINO-style asymmetric views for GraphDINO* (4 views, teacher views with `edge_drop` 0.2
    only, student views with `edge_drop` 0.3 + `feat_mask` 0.3), on the library defaults:
    much worse, 72.3 → 57.9 | 47.5 → 37.2.
  - *AFGRL with a faster teacher* (`--ema-tau 0.9`, annealed to 1.0): 71.4 → 74.3 | 48.4 →
    50.9, kNN 67.2 → 71.0 | 46.3 → 47.3. *AFGRL with lr 0.01*: much worse, 58.7 | 38.3
    (55.6 | 38.5 with both). A candidate for a tuned AFGRL config, not for the shared one.
  - *Reference BGRL predictor* (Linear → PReLU → Linear instead of the library's Linear →
    BatchNorm → ReLU → Linear), BGRL and AFGRL, GIN and GCN: within one std in 6 of 8
    cells. With GIN on CiteSeer it gains about 4 points (BGRL 48.9 → 53.1, AFGRL 49.2 →
    53.0) at half the effective rank and, for BGRL, a lower kNN (42.1 → 35.1). Not what
    holds the teacher-student methods back. Keep the library's.
  - With the current probe the top of the table is close to the literature: GraphCL 82.0 ±
    1.0 on Cora (0.1.0 probe: 78.1).
- The 20 JSONs at commit `e295451` say `"dirty": true` — false positive (untracked files were
  counted, fixed in `dae32cb`); the code that ran is exactly `e295451`. GraphDINO and
  Supervised were rerun at `20cce09`, Supervised std. recipe at `558af8e` (all clean).
- Cost: on PubMed GraphCL (~440 s/run with GIN, O(N²) NT-Xent) and AFGRL (~250 s/run, dense
  N×N kNN + per-step k-means; ~500 s before the FAISS thread cap) are 5–10× slower than the
  other methods (35–55 s).
- `paper.tex` compares our BGRL against the original BGRL paper's numbers (Appendix C, Table 7)
  and spells out the protocol differences.
- Every result JSON also has `model_config` (the exact `build_model()` dict), per-seed
  `eff_rank`, and for BGRL/AFGRL/GraphDINO the other encoder's metrics (`*_alt`).
- **Ablations never go in `benchmarks/results/`** (render_tables.py would report them as the
  benchmark): use `--out-dir benchmarks/ablations/<name>` and `benchmarks/render_ablation.py`.
  `benchmarks/ablation_teacher_student.sh` is the grid for the teacher-student gap (budget
  300/1000/3000 × `--ema-tau` default/0.9, Barlow Twins control, BGRL + GCN, AFGRL on Cora).
  `benchmarks/ablation_graphdino.sh` tests GraphDINO's stability levers (see GraphDINO);
  `benchmarks/ablation_graphdino_ema.sh` is its follow-up (faster teacher with GCN / on PubMed /
  with the softer teacher, plus the untrained-encoder reference at `--epochs 0`). Both write
  to `benchmarks/ablations/graphdino_stability`.
- `--set KEY=VALUE` (repeatable) overrides any model-config field, dotted for nested ones
  (`--set head.teacher_temp=0.07`); keys are validated against the config dataclasses, values
  parsed as JSON, and recorded under `hyperparameters.overrides` (a column in render_ablation).

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
- **Cross-method runner**: `benchmarks/run_zinc.py --model all supervised --seeds 3` → one
  JSON per model in `benchmarks/zinc/` + `summary.md`. Shared untuned protocol: ZINC-12k
  (10k / 1k / 1k), GINE-4L, 128 units, batch norm, mean readout, 100 epochs, batch 256,
  AdamW lr 1e-3 / wd 1e-5; augmentations `attr_mask` 0.2 (mask token = index 28, so
  `node_emb_num_classes: 29`) + `edge_drop` 0.2. Evaluation: `RidgeEvaluator` on the frozen
  graph embeddings → test MAE (lower is better) + effective rank. Every run also saves its
  own encoder **untrained** (0.598 MAE with seed 0), and `--model supervised` is the
  end-to-end reference (`task: regression`, test MAE of its head at the best-validation
  epoch). About 5 s per epoch on the A2.
- `--readout mean|sum|max` (default mean) sets `encoder.readout`.
- `--finetune-epochs N` adds the usual protocol for molecules: after pre-training, the
  encoder and a new linear head are trained on the labels (L1, best-validation epoch).
  Its reference is the supervised row — the same training from a random initialisation.
- **Results** (2026-10-09, `49e92ee`, 3 seeds, 100 pre-training + 100 fine-tuning epochs,
  test MAE; `benchmarks/zinc/summary.md` — regenerate from the JSONs, don't retype;
  untrained encoder 0.593 ± 0.010), frozen ridge probe | fine-tuned:

  | Method | Frozen encoder | Fine-tuned | Rank |
  |---|---|---|---|
  | VICReg | 0.591 ± 0.010 | 0.335 ± 0.009 | 18.5 |
  | GraphCL | 0.571 ± 0.006 | 0.341 ± 0.007 | 18.0 |
  | Barlow Twins | 0.574 ± 0.005 | 0.354 ± 0.012 | 15.7 |
  | DGI | 0.932 ± 0.071 | 0.359 ± 0.018 | 9.2 |
  | GraphDINO | 0.578 ± 0.009 | 0.361 ± 0.009 | 15.6 |
  | *Supervised from scratch* † | 0.396 ± 0.028 | 0.364 ± 0.007 | 7.5 |
  | BGRL | 0.700 ± 0.082 | 0.377 ± 0.007 | 5.6 |
  | AFGRL | 0.671 ± 0.046 | 0.379 ± 0.007 | 5.0 |

  - **Frozen**: GraphCL, Barlow Twins and GraphDINO are 0.015–0.02 below the untrained
    encoder, VICReg equals it; BGRL and AFGRL collapse (rank ≈ 5) and DGI is far worse than
    no training. None comes near the supervised encoder (0.396): with a frozen probe SSL
    buys almost nothing here.
  - **Fine-tuned**: VICReg (−0.029) and GraphCL (−0.023) beat the same training from a
    random initialisation by about three of its stds; Barlow Twins, DGI and GraphDINO are
    within one; BGRL and AFGRL are worse than no pre-training. Three seeds, untuned.
  - Sum readout moved little in a one-seed diagnostic (untrained 0.559, supervised head
    0.341, VICReg frozen 0.577).
- Bond types only count since `encode()` (see Key utilities): before, every ZINC run ignored
  them. The example's `feat_mask` hides the single atom-type column for every atom or for
  none — use `attr_mask` on molecules.

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
- `ogb` <= 1.3.6 loads its processed file with a bare `torch.load()`: with PyTorch >= 2.6
  allow-list PyG's `DataEdgeAttr`, `DataTensorAttr`, `GlobalStorage` via
  `torch.serialization.add_safe_globals` first (done in the example and the stress test)
- **Stress test**: `benchmarks/stress_ogbn_arxiv.py` runs all 7 SSL methods through the
  mini-batch path (NeighborLoader, callbacks with a kNN / effective-rank evaluation during
  training, linear probe before and after it, full-graph evaluation, mini-batch extraction
  checked against the full-graph pass). One seed, no tuning: it reports failures, time, memory
  and accuracy curves, it is not a benchmark. Results go to
  `benchmarks/stress/ogbn_arxiv/` (one JSON per model + `summary.md`); a failing model doesn't
  stop the run. `--skip-linear-probe` for smoke runs (two probe fits per model otherwise).
  Each trained model is saved next to its JSON (`<model>.pt`, git-ignored, `load_model()`
  reads it): the 2026-10-06 run kept none, so re-evaluating it means running it again.
  The committed run (2026-10-06, `aaa280f`, 100 epochs) used the 0.1.0 probe: read its kNN
  column, not the linear one. `status: ok` only means the model ran to the end — DGI is `ok`
  there with embeddings worse than the untrained encoder's (kNN 33.7 vs 53.3).
- **Reproduction of the BGRL paper** (`benchmarks/reproduce_bgrl_arxiv.py`, results in
  `benchmarks/reproductions/bgrl_ogbn_arxiv/`): the paper's own protocol, not the shared
  benchmark one. Thakoor et al., ICLR 2022, Table 5 / Table 8 / Appendix F (checked against
  the PDF): symmetrized graph, full-graph training, 10,000 steps; 3 GCN layers × 256 with
  layer norm + weight standardization + PReLU; predictor hidden 256; edges dropped with
  p = 0.6 in both views, no feature masking; AdamW lr 1e-2 (1,000 warm-up steps, cosine
  decay to 0), wd 1e-5; EMA 0.99 → 1.0. Paper (20 seeds, val / test): BGRL 72.53 ± 0.09 /
  71.64 ± 0.12, the same encoder untrained 69.90 ± 0.11 / 68.94 ± 0.15, DGI 71.26 / 70.34,
  MLP on raw features 57.65 / 55.50, supervised GCN 73.00 / 71.74.
  - **The untrained encoder reproduces the paper**: 70.21 / 69.50 with the library probe
    (seed 0). It validates the GCN encoder and the evaluation independently of training.
  - **Trained BGRL does not reproduce it** (seed 0, 10,000 steps, 2026-10-08, uncommitted
    tree, 7.6 h on the shared A2): 68.23 / 67.38, i.e. **−1.98 / −2.12 against the untrained
    encoder** where the paper gains +2.63 / +2.70. The paper's std is 0.12: not seed noise.
    kNN goes 66.7 → 63.2 in the first 1,000 steps and stays at 64.6–64.9 from step 3,000
    on; final loss 0.0041, effective rank 116 → 159; the probe selects the weakest L2 of its
    grid (1e-6). No checkpoint was kept (the run predates the checkpoint saving). That run
    used the library's predictor (BatchNorm): see "What breaks it" below. The script now
    defaults to the reference predictor (`--predictor reference`; `library` for the old one).
  - **The same recipe at Cora scale** (diagnostic, 2026-10-08, uncommitted tree: BGRL + GCN
    through `run_benchmark.py`, constant lr, 3 seeds, linear accuracy Cora | CiteSeer, gain
    over the same encoder untrained), one ingredient at a time from the shared protocol to
    the paper's ogbn-arxiv recipe:

    | Variant (cumulative) | Steps | Trained | Gain | Rank |
    |---|---|---|---|---|
    | batch norm, 2 layers, shared protocol | 300 | 81.4 \| 67.7 | +9.0 \| +9.3 | 171 \| 189 |
    | layer norm | 300 | 78.5 \| 68.5 | +6.5 \| +8.8 | 150 \| 177 |
    | + weight standardization | 300 | 80.7 \| 66.8 | +8.6 \| +7.5 | 169 \| 189 |
    | + 3 layers | 300 | 78.4 \| 64.8 | +6.5 \| +6.4 | 144 \| 170 |
    | + `edge_drop` 0.6 only, predictor 256 | 300 | 76.3 \| 62.4 | +4.4 \| +4.0 | 156 \| 179 |
    | + lr 1e-2 | 300 | 75.3 \| 60.7 | +3.4 \| +2.3 | 29 \| 76 |
    | same, ten times longer | 3000 | 71.3 \| 58.9 | −0.6 \| +0.5 | 11 \| 31 |
    | batch norm instead of layer norm + WS (3 layers, `edge_drop` 0.6, lr 1e-2) | 300 | 79.5 \| 61.3 | +6.8 \| +4.7 | 127 \| 140 |

    No single ingredient breaks BGRL, but the gain shrinks at every step towards the paper's
    recipe and is gone at 3,000 steps, with the embeddings collapsed to rank 11 on Cora.
    The rank drop comes with lr 1e-2 on the layer-norm encoder (156 → 29); the batch-norm
    encoder keeps its rank at the same lr. Not the same symptom as on ogbn-arxiv, where the
    rank grows while accuracy falls — a lead, not the explanation.

    Follow-up at 3,000 steps, still the library predictor (3 seeds, GPU, gain over the same
    encoder untrained): batch-norm encoder at lr 1e-2 76.5 | 61.1, +3.7 | +4.5, rank 167 |
    161; layer norm + WS at lr 5e-4 73.4 | 58.5, +1.4 | +0.1, rank 179 | 193 — the gain goes
    at the low learning rate too, with no rank drop at all; layer norm without WS at lr 1e-2
    72.3 | 55.0, rank 3 | 4, kNN 42.5 | 37.0.
  - **What breaks it: the predictor's BatchNorm on top of the layer-norm encoder**
    (diagnostics, 2026-10-08/09, uncommitted trees, the arxiv rows one seed stopped at
    2,000 of the 10,000 steps — rerun from committed code before quoting; the mechanism is
    not identified, don't state one). ogbn-arxiv, seed 0, the paper's schedule, one change
    per run; linear val / test, gain over the same encoder untrained:

    | Run | Untrained | Step 2,000 | Gain | kNN test | Loss | Pair cos | Rank |
    |---|---|---|---|---|---|---|---|
    | paper recipe, library predictor | 70.20 / 69.49 | 68.31 / 67.92 | −1.89 / −1.57 | 66.70 → 64.43 | 0.006 | 0.92 → 0.99 | 116 → 117 |
    | reference predictor (Linear → PReLU → Linear) | 70.22 / 69.48 | 71.71 / 70.72 | +1.49 / +1.24 | 66.70 → 68.02 | 0.221 | 0.92 → 0.12 | 116 → 135 |
    | batch-norm encoder, no WS, library predictor | 70.22 / 68.96 | 72.05 / 71.03 | +1.83 / +2.07 | 67.34 → 67.76 | 0.197 | 0.96 → 0.21 | 63 → 169 |
    | peak lr 1e-3, library predictor | 70.21 / 69.50 | 70.23 / 69.22 | +0.02 / −0.28 | 66.70 → 65.24 | 0.025 | 0.92 → 0.88 | 116 → 63 |

    "Pair cos" is the mean cosine between the embeddings of random node pairs. In the
    failing runs the loss goes to ≈ 0 and the embeddings stay almost parallel (0.88–0.99):
    a collapse towards one direction that the effective rank, computed on centered
    embeddings, does not show. The runs that gain settle at a loss of ≈ 0.2 with pair cos
    0.1–0.2. Either change alone (reference predictor, or batch norm in the encoder) is
    enough.

    **Full 10,000 steps with the reference predictor** (seed 0, 2026-10-09, a copy of
    `7aa5499` that was not a git checkout, 7.6 h on the shared A2): **72.45 / 71.15**
    against 70.21 / 69.50 untrained, i.e. **+2.24 / +1.65** (paper: 72.53 ± 0.09 / 71.64 ±
    0.12, +2.63 / +2.70). Validation is within the paper's std, test is 0.5 below; the gain
    is smaller also because our probe gives the untrained encoder 0.3 / 0.6 more than the
    paper's. kNN 66.7 → 67.6 at step 1,000, 68.1–68.2 from step 2,000 on (68.05 at the
    end); loss 0.21 throughout; rank 116 → 166; probe L2 1e-3. One seed: the three-seed run
    from committed code (`acae6df`, log `reproduce_bgrl_20261010.log` on the VM, JSON in
    `benchmarks/reproductions/bgrl_ogbn_arxiv/` when the third seed ends) is the one to
    quote. Its seed 0 repeats the result: 72.43 / 71.22, +2.22 / +1.72, in 3.9 h on the
    idle A2.

    The same at Cora scale (BGRL + GCN-3L, layer norm + WS, `edge_drop` 0.6, lr 1e-2, 300
    steps, 3 seeds, laptop CPU, Cora | CiteSeer; untrained 71.9 | 58.4, rank 186 | 199),
    changing the predictor only:

    | Predictor | Linear | Gain | kNN | Rank |
    |---|---|---|---|---|
    | BatchNorm + ReLU (library default) | 77.2 ± 1.3 \| 60.1 ± 2.1 | +5.2 \| +1.7 | 62.2 \| 51.1 | 29 \| 64 |
    | BatchNorm + PReLU | 71.5 ± 2.4 \| 61.3 ± 0.8 | −0.5 \| +2.9 | 60.3 \| 50.6 | 39 \| 105 |
    | no norm + ReLU | 80.4 ± 0.6 \| 66.2 ± 1.3 | +8.5 \| +7.8 | 71.7 \| 61.0 | 98 \| 111 |
    | no norm + PReLU (reference) | 79.9 ± 0.8 \| 66.0 ± 0.4 | +8.0 \| +7.6 | 71.6 \| 60.7 | 108 \| 121 |

    The same four at 3,000 steps (VM CPU, same seeds):

    | Predictor | Linear | Gain | kNN | Rank |
    |---|---|---|---|---|
    | BatchNorm + ReLU (library default) | 74.4 ± 1.4 \| 58.6 ± 1.0 | +2.5 \| +0.2 | 62.7 \| 48.6 | 11 \| 26 |
    | BatchNorm + PReLU | 71.5 ± 1.5 \| 55.6 ± 4.4 | −0.5 \| −2.8 | 62.1 \| 49.2 | 13 \| 92 |
    | no norm + ReLU | 76.5 ± 0.8 \| 59.8 ± 1.0 | +4.6 \| +1.4 | 67.4 \| 53.4 | 180 \| 160 |
    | no norm + PReLU (reference) | 76.8 ± 1.3 \| 60.3 ± 1.5 | +4.9 \| +1.9 | 66.3 \| 51.3 | 196 \| 185 |

    Without the predictor's BatchNorm the rank no longer collapses (180–196 against 11–26),
    but the gain still shrinks with training length (Cora +8.0 → +4.9, CiteSeer +7.6 →
    +1.9): at this scale the BatchNorm explains the collapse, not the whole decline. The
    first row on the GPU gave 71.3 | 58.9 with the same seeds: over 3,000 steps at lr 1e-2
    CPU and GPU runs drift apart by up to 3 points, so compare within one device.

    It is the normalization, not the activation. In the shared protocol (batch-norm
    encoders, lr 5e-4) the same swap changes little — see "Reference BGRL predictor" in the
    Planetoid section — so the library default (`pred_norm="batch"`) is unchanged for now.
    Whether it should become `"none"` is open: it needs the BGRL / AFGRL benchmark rows
    with `--set pred_norm=none` (10 seeds, both encoders).
  - The paper's linear evaluation (L2-normalized rows, 100 AdamW steps at lr 0.01, weight
    decay searched), implemented literally, is far from fitted: 44.1 on the untrained
    encoder, 66.5 after 1,000 steps, 68.8 after 5,000. The script therefore uses the library
    probe and reports BGRL's **gain over the untrained encoder** (paper: +2.63 / +2.70),
    which does not depend on how strong the classifier is.
  - Known difference left: the reference implementation (nerdslab/bgrl) uses PyG's
    graph-mode `LayerNorm`, the library per-node `nn.LayerNorm`. The predictor is no longer
    one (`pred_norm: none`, `pred_activation: prelu`).
  - An untrained GCN-3L (69.5) is far above every GIN-2L result of the stress test (kNN ≤
    59.5; linear ≤ 62 where re-evaluated): on ogbn-arxiv the backbone, not the SSL method,
    sets the level. Read the stress test as a robustness check only.
  - One seed takes hours on the A2 (full-graph, 8 GiB).
- Both example scripts (`examples/ogbn_arxiv_bgrl.py` and `examples/zinc_bgrl.py`) crashed
  before `6493aa3` — examples are not covered by the test suite, so smoke-run them after API
  changes.

---

## TESTS

```bash
pytest tests/ -v
```

| File | Model | Notes |
|---|---|---|
| `test_bgrl.py` | BGRL | teacher frozen, reset → different weights, EMA scheduler, step counter, predictor layout (`pred_norm` / `pred_activation`: default and reference, slope trained, invalid values) |
| `test_dgi.py` | DGI | W learnable, shuffle_nodes/shuffle_edges, one forward pass (an untouched isolated node gets the same embedding in the real and in the corrupted graph), mini-batch loss on the seed nodes only |
| `test_graphcl.py` | GraphCL | projector dim, NT-Xent ≥ 0, chunked NT-Xent == full (value + grads) |
| `test_vicreg.py` | VICReg | 3-layer projector, loss ≥ 0 |
| `test_barlow_twins.py` | BarlowTwins | lambda default = 1/proj_dim |
| `test_afgrl.py` | AFGRL | `kmeans_threads` applied + restored, reference predictor; **auto-skipped if faiss isn't installed** |
| `test_positive_miner.py` | AFGRL | chunked top-k search == full matrix, self always first, `knn_chunk_size` validation (no faiss needed) |
| `test_supervised.py` | Supervised | head dim, mini-batch crop, full-batch loss on `train_mask` only (raises without it), graph-level pooling, regression task (L1) |
| `test_graphdino.py` | GraphDINO | freeze last layer, teacher temp warmup, defaults (EMA 0.9 → `ema_tau`, teacher temp 0.04 → 0.07), center in logit space, `norm_last_layer`, DINOTrainer hooks |
| `test_model_encoder_matrix.py` | all | every model × every encoder (gin/gcn/transformer) builds, trains a step and returns one embedding per node / per graph; edge features reach the encoder in training and in `forward()` (gin/transformer); AFGRL rows need faiss |
| `test_new_features.py` | — | edge_emb_num_classes, norm_type API, GCN eval-mode statistics after a short training (BatchNorm momentum), GCN depth (`num_layers`) and weight standardization, CombinedLoss |
| `test_checkpoint.py` | all | `save_model` / `load_model` give the same embeddings, `pretrained_encoder` is the encoder `forward()` uses, fine-tuning starts from its weights |
| `test_callbacks.py` | — | `EmbeddingLoggerCallback` files, `LinearEvalCallback` history and custom `evaluator`, the model trains again after an evaluation |
| `test_schedulers.py` | — | `CosineDecayScheduler` (linear warm-up, cosine decay, BGRL's formula), `CosineEMAScheduler` (BGRL's formula, monotonic, clamped) |
| `test_augmentation.py` | — | `attr_mask` (whole nodes, categorical dtype kept), `edge_add` with 1-D edge attributes, protected nodes (`node_drop` keeps them in the leading rows, views stay aligned, models pass a mini-batch's seeds, `subgraph` refuses them), each augmentation's basic behaviour |
| `test_evaluation.py` | — | RidgeEvaluator (recovers a linear target, L2 selected on validation MAE, scale invariance), LogRegEvaluator (result independent of the seed, L2 selected on validation, invariant to feature scale/offset, `converged` flag, multilabel selection needs scikit-learn), KNNEvaluator, OGB-style 2D labels (`[N,1]`) equivalent to 1D, chunked kNN == single pass, `extract_embeddings` leaves the datamodule's graph in place, `effective_rank` |

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
baseline is ~83% (2026-10-08; it was ~70% with `training/callbacks.py` at 27%,
`utils/schedulers.py` at 53% and no test of the augmentation module). Remaining gaps:
`VisualizationCallback` (needs umap/matplotlib) and the benchmark/example scripts, which the
suite does not run — smoke-run them after API changes.

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
separate parameter. No model infers it at runtime from the batch. In graph-level mode every
model's `forward()` returns **one embedding per graph** (pooled): `extract_embeddings` pairs it
with graph labels. DGI returned node embeddings until this was enforced by
`tests/test_model_encoder_matrix.py`, which also caught GraphDINO building its encoder inline
(it crashed with GCN/Transformer) — always use `cfg.encoder.build(in_channels)`.

### Mini-batch: protected nodes
In mini-batch node training, `batch.batch_size` gives the seed-node count.
The loss must be computed only on `z[:batch.batch_size]`.
`compose()` accepts `protected_nodes=torch.arange(batch.batch_size)` and propagates it to
every augmentation; `node_drop` uses this parameter to never remove seed nodes.
Only `node_drop` reads it: with `edge_drop` / `feat_mask` (the benchmark's and the stress
test's augmentations) there is nothing to protect, so no experiment exercises it yet.
`tests/test_augmentation.py` checks the mechanism itself (there was no test of it before,
whatever the paper said): protected nodes stay in the leading rows of every view, without
protection they don't, and GraphCL / BGRL pass a mini-batch's seeds to `compose()`.
`subgraph` cannot honour it and raises instead of silently pairing the wrong nodes.
**AFGRL is the exception to the seed crop**: it mines positives and computes its loss over
every node of the batch, sampled neighbours included — it has no proper mini-batch mode.

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
Uses average precision score instead of accuracy. Average precision needs scikit-learn:
without it a single `weight_decay` returns NaN metrics, and selecting among several raises.

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
- the encoder is called through `encode(encoder, data)` (`core/encoder.py`), never as
  `encoder(x, edge_index, batch)` — that drops the edge features
- `losses/` — is the loss testable standalone?
- `utils/ema.py`, `utils/schedulers.py` — are EMA and the scheduler used correctly in `post_step()`?
- `augmentation/` — does `compose()` pass `protected_nodes`?
- `config/schema.py` — was a config dataclass added?
- `models/__init__.py`, `src/graphssl/__init__.py` — are exports updated?

### 4. Update this file (and CONTRIBUTING.md if the dev workflow itself changed)
Describe precisely: the config dataclass, the loss formula, EMA details, default
hyperparameters.
