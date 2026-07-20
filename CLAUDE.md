GraphSSL è una libreria Python per Graph Machine Learning (GML).
Implementa i principali algoritmi SSL su grafi in PyTorch puro,
con un'architettura modulare e riutilizzabile.

---

## STACK TECNOLOGICO

- Python 3.10+
- PyTorch + PyTorch Geometric (PyG)
- Solo PyTorch puro nel core: zero dipendenze da PyTorch Lightning, Hydra, OmegaConf
- FAISS-cpu per il positive mining (AFGRL) — opzionale
- UMAP + matplotlib + seaborn per visualizzazione — opzionale
- ogb + pyyaml per benchmark e config YAML — opzionali

---

## PUBLIC API — COSTRUZIONE UNIFORME

**Tutti i modelli usano la stessa firma pubblica:**
```python
model = ModelClass(config: Dict, in_channels: int)
# Supervised aggiunge: num_classes: int
model = Supervised(config, in_channels, num_classes)
```

Il `config` dict viene validato da un dataclass dedicato in `src/config/schema.py`.
`EncoderConfig.build(in_channels)` istanzia l'encoder tramite il registry `ENCODERS`.

**Factory da YAML:**
```python
from graphssl.config.load import load_config, build_model
cfg = load_config("configs/bgrl.yaml")
model = build_model(cfg, in_channels=dataset.num_features)
```

**Dataclass per modello:**
| Modello      | Config dataclass    |
|--------------|---------------------|
| DGI          | `DGIConfig`         |
| GraphCL      | `GraphCLConfig`     |
| VICReg       | `VICRegConfig`      |
| BarlowTwins  | `BarlowTwinsConfig` |
| BGRL         | `BGRLConfig`        |
| AFGRL        | `AFGRLConfig`       |
| Supervised   | `SupervisedConfig`  |
| GraphDINO    | `GraphDINOConfig`   |

Tutti in `src/graphssl/config/schema.py`.

---

## ALGORITMI IMPLEMENTATI

Tutti i modelli sono `nn.Module` puri con `forward()` e `compute_loss()`.
Nessun accesso a trainer, logger, o datamodule dall'interno del modello.

### Supervised
- Encoder + linear head `nn.Linear(hidden_dim, num_classes)`, CrossEntropyLoss
- Supporto full-batch e mini-batch (crop a `[:batch_size]` per seed nodes)
- `graph_level` derivato da `cfg.encoder.pool`

### DGI (Deep Graph Infomax)
- Discrimina embedding reale vs corrotto tramite un discriminatore con matrice W learnable
  (`nn.Parameter`, inizializzata con `xavier_uniform_`)
- Corruzione: shuffle dei nodi (permutazione di x) o shuffle delle edge destinations
- Summary globale `s = sigmoid(mean(h_pos))` per node-level, `global_mean_pool` per graph-level
- Discriminator: `pos_logits = (h_pos * W@s).sum(-1)`
- Loss: BCEWithLogitsLoss su [pos_logits, neg_logits] con label [1,1,...,0,0,...]
- Iperparametri: `corruption="shuffle_nodes", shuffle_ratio=1.0`

### GraphCL (Graph Contrastive Learning)
- NT-Xent loss su 2 viste augmentate
- Encoder + projector 2-layer `Projector(hidden_dim, hidden_dim, proj_dim)`
- `protected_nodes` passato a `compose()` per i seed nodes in mini-batch node training
- Iperparametri: `proj_dim=128, tau=0.5`

### BGRL (Bootstrapped Graph Representation Learning)
- Teacher-student con EMA update sul target encoder
- Online encoder (student) + predictor MLP `Predictor(hidden_dim, pred_hidden, hidden_dim)`; target encoder (teacher, no grad)
- Loss: `2 - cos(pred1, target2) - cos(pred2, target1)` — `CosineRegressionLoss(symmetric=True)`
  - Chiamata corretta: `loss_fn(p1, t1, p2, t2)` → la loss internamente usa t2 con p1 e t1 con p2
- EMA update del target encoder in `post_step()` via `update_ema_params()`
- Momentum τ cosine-annealed da `ema_tau` a `ema_tau_end` su `total_steps` — `CosineEMAScheduler`.
  Se `total_steps=0`, τ fisso.
- Target encoder: `deepcopy(encoder)` + `reset_parameters()` — pesi intenzionalmente diversi
  dall'online encoder (cruciale per la convergenza, App. B del paper BGRL)
- Iperparametri: `ema_tau=0.99, ema_tau_end=1.0, total_steps=0, pred_hidden=512`

### AFGRL (Augmentation-Free Graph Representation Learning)
- Come BGRL ma senza augmentation strutturale: positive pairs minati da struttura del grafo
- Online encoder (student) + predictor MLP; target encoder (EMA, no grad)
- **Differenza da BGRL**: target encoder inizia con stessi pesi dell'online (`deepcopy` senza reset)
- **PositiveMiner** (`utils/positive_miner.py`): unione di due tipi di positivi per ogni nodo:
  - Local: top-k vicini per cosine similarity CHE SONO ANCHE adiacenti nel grafo (kNN ∩ adj)
  - Global: top-k vicini che condividono un cluster k-means in ALMENO UNA delle `num_kmeans` run
- Loss graph-level: nessun miner; teacher-student semplice su embeddings poolati
- EMA: identico a BGRL — `CosineEMAScheduler` in `post_step()`
- Iperparametri: `ema_tau=0.99, ema_tau_end=1.0, total_steps=0, topk=5,
  num_centroids=50, num_kmeans=4, clus_num_iters=20, pred_hidden=512`

### GraphDINO
- Adattamento di DINO (ViT) ai grafi
- Student encoder + student head; teacher encoder + teacher head (EMA, no grad)
- `n_views` viste totali; le prime `n_global_views` vanno al teacher, tutte allo student
- **DINOLoss**: cross-entropy su tutti i pari (teacher_i, student_j) con i ≠ j, mediata per numero di termini
  - Student output: `log_softmax(logits / student_temp)` — calcolato in `DINOHead`
  - Teacher output: `softmax((logits − center) / teacher_temp_eff)` — calcolato in `DINOHead`
- **Teacher temperature warmup**: `teacher_temp_eff` cresce linearmente da `warmup_teacher_temp`
  a `teacher_temp` in `warmup_teacher_temp_epochs` epoche. `DINOHead.set_epoch(epoch)` aggiorna il valore.
- **Center update** (EMA): `center = c_mom * center + (1 − c_mom) * mean(teacher_out)`
  Gestito da `DINOHead.update_center()`, chiamato da `post_step()`.
- **EMA teacher**: momentum crescente cosine da `ema_tau_base` a `ema_tau` su `total_steps`.
- **freeze_last_layer_epochs**: nelle prime N epoche i gradienti del layer prototipo
  (`student_head.proto`) vengono azzerati dopo il backward — in `post_backward()`.
- Iperparametri: `student_temp=0.1, teacher_temp=0.07, warmup_teacher_temp=0.04,
  warmup_teacher_temp_epochs=30, ema_tau=0.996, freeze_last_layer_epochs=1,
  n_views=2, n_global_views=2`

### VICReg
- Encoder + projector 3-layer `Projector(hidden_dim, hidden_dim*2, proj_dim)`
- Loss tripla: invariance (MSE), variance (hinge su std), covariance (off-diagonal) — `VICRegLoss`
- Iperparametri: `proj_dim=256, invariance=25.0, variance=25.0, covariance=1.0`

### Barlow Twins
- Encoder + projector 3-layer identico a VICReg
- Cross-correlation matrix `C = (z1_norm.T @ z2_norm) / N`
- Loss: `sum((1−C_ii)²) + lambda * sum_{i≠j}(C_ij²)` — `BarlowTwinsLoss`
- `lambda_param=None` → default `1/proj_dim` calcolato dentro `BarlowTwinsLoss`
- Iperparametri: `proj_dim=256, lambda_param=None`

---

## ENCODER (GNN BACKBONES)

Tutti espongono `forward(x, edge_index, batch, edge_attr=None) → Tensor[N, out_dim]`.
Tutti registrati tramite `@ENCODERS.register("nome")`.

**`EncoderConfig` — parametri condivisi:**
| Campo | Tipo | Default | Note |
|---|---|---|---|
| `name` | str | — | 'gin', 'gcn', 'transformer' |
| `hidden_dim` | int | — | dimensione hidden/output |
| `num_layers` | int | — | numero di layer |
| `norm_type` | str | "batch" | 'batch', 'layer', 'none' — **API uniforme tra tutti gli encoder** |
| `pool` | bool | True | global_add/mean_pool per graph-level |
| `drop` | float | 0.2 | dropout rate |
| `mlp_ratio` | float | 2.0 | moltiplicatore hidden del MLP interno |
| `edge_dim` | int\|None | None | abilita edge-feature-aware conv (GINEConv / TransformerConv) |
| `node_emb_num_classes` | int\|None | None | sostituisce Linear con nn.Embedding per feature categoriche (ZINC: 28) |
| `edge_emb_num_classes` | int\|None | None | nn.Embedding per edge features categoriche (ZINC: 4). Richiede `edge_dim`. |

`EncoderConfig.build(in_channels)` usa `inspect.signature` per passare solo i kwargs
accettati dall'encoder specifico — i campi non supportati vengono silenziosamente ignorati.

### GCN
- 2 layer GCNConv con norm + PReLU
- Weight standardization opzionale sul secondo layer
- Backward compat: accetta ancora `batchnorm=True/False` e `layernorm=True/False`
- `reset_parameters()` esposto (usato da BGRL per target encoder)

### GIN (Graph Isomorphism Network)
- Stack di `GINLayer` con pre-norm + residual connections
- `GINLayer`: `norm(x)` → `GINConv(h)` → `ReLU` → `Dropout` → `x + h`
- `edge_dim != None` → usa `GINEConv` (edge-feature aware)
- Se `edge_dim` è impostato ma `edge_attr=None` a runtime, `GINEConv` riceve zeri (non crasha)
- `node_emb_num_classes` → `nn.Embedding` invece di `Linear` per feature categoriche
- `edge_emb_num_classes` → `nn.Embedding` per edge features categoriche prima di `GINEConv`

### Graph Transformer
- Stack di `TransformerBlock` (TransformerConv da PyG), pre-norm + FFN + residual
- Supporta `edge_attr` via `TransformerConv(edge_dim=...)`
- Stessa API `norm_type` di GINEncoder (batch/layer/none)
- `node_emb_num_classes` e `edge_emb_num_classes` supportati

---

## SISTEMA DI AUGMENTATION

Sistema componibile stile torchvision in `augmentation/`:
- `functional.py`: funzioni pure `(data, *, protected_nodes=None, **kwargs) → Data`
- `transforms.py`: classi callable che wrappano le funzioni
- `compose.py`: `compose(data, aug_list, protected_nodes=None)` applica la lista in sequenza

Operazioni disponibili:
| Nome | Parametri | Descrizione |
|---|---|---|
| `edge_drop` | p | rimuove edge con probabilità p |
| `edge_add` | p | aggiunge edge casuali (frazione p degli esistenti) |
| `feat_mask` | p | maschera features dei nodi con probabilità p |
| `feat_noise` | std | aggiunge rumore gaussiano alle features |
| `feat_shuffle` | p | scambia features tra nodi casuali |
| `subgraph` | num_hops | estrae k-hop subgraph da un seed casuale |
| `node_drop` | p | rimuove nodi; i `protected_nodes` non vengono mai rimossi |

Due API:
1. **Class-based** (stile torchvision): `EdgeDrop(p=0.2)(data)` — usata con `MultiView`
2. **Registry-based** via `compose()`: `compose(data, [("edge_drop", {"p": 0.2})])` — supporta `protected_nodes`

`MultiView(transforms, n_views)` accetta entrambe le API e genera n viste indipendenti.

---

## LOSS FUNCTIONS

Tutte le loss sono `nn.Module` standalone in `losses/`, testabili indipendentemente.
Tutte registrate nel `LOSSES` registry tramite `@LOSSES.register("nome")`.

| Nome registry | Classe | Usata da |
|---|---|---|
| `nt_xent` | `NTXentLoss` | GraphCL |
| `vicreg` | `VICRegLoss` | VICReg |
| `barlow_twins` | `BarlowTwinsLoss` | BarlowTwins |
| `cosine_regression` | `CosineRegressionLoss` | BGRL, AFGRL |
| — | `DINOLoss` | GraphDINO |

**Loss combinabili:**
```python
from graphssl.losses import CombinedLoss

fn = CombinedLoss.from_config([
    {"name": "nt_xent", "weight": 0.7, "tau": 0.5},
    {"name": "vicreg",  "weight": 0.3, "invariance": 25.0, "variance": 25.0, "covariance": 1.0},
])
loss = fn(z1, z2)
```
Oppure programmaticamente: `CombinedLoss([(NTXentLoss(tau=0.5), 0.7), (VICRegLoss(), 0.3)])`.

---

## REGISTRY PATTERN

```python
from graphssl.registry import ENCODERS, LOSSES, AUGMENTS, OBJECTIVES, DATASETS, LOADERS

# Registrare un encoder custom:
@ENCODERS.register("my_gat")
class GATEncoder(nn.Module): ...

# Istanziare dal registry:
enc = ENCODERS.build("my_gat", in_channels=32, hidden_dim=64)
```

Registry disponibili:
- `ENCODERS` — backbone GNN
- `HEADS` — projection/prediction heads
- `AUGMENTS` — trasformazioni di augmentation
- `LOSSES` — funzioni di loss
- `LOADERS` — factory di DataLoader
- `OBJECTIVES` — obiettivi SSL custom (placeholder per estensioni)
- `DATASETS` — dataset builder custom (placeholder per estensioni)

---

## UTILITY CHIAVE

### EMA dei parametri (`utils/ema.py`)
`update_ema_params(student, teacher, tau)` — aggiorna parametri e buffer (es. BatchNorm running stats).
I buffer vengono copiati direttamente (non mediati).

### Schedulers (`utils/schedulers.py`)
- `CosineDecayScheduler(max_val, min_val, total_steps, warmup_steps)` — decay cosine con warmup lineare
- `CosineEMAScheduler(ema_base, ema_end, total_steps)` — momentum EMA crescente cosine (BGRL/DINO)

### Pooling (`nn/pooling.py`)
`pool_graph_embeddings(node_embeddings, batch)` — `global_mean_pool` con fallback se `batch=None`.

### extract_embeddings (`evaluation/visualization.py`)
Estrae embedding da un modello dato un datamodule. Gestisce:
1. Graph-level: DataLoader standard
2. Node full-batch: forward sull'intero grafo
3. Node mini-batch: NeighborLoader con `global_to_local` mapping per ricostruire l'ordine

### LogRegEvaluator (`evaluation/linear_probe.py`)
Evaluator lineare PyTorch puro. Per multilabel (es. ogbg-molpcba) usa BCE + average precision.

### DataModule (`data/datamodule.py`)
Wrappa dataset PyG per due modalità:
- Graph-level (`is_graph_level=True`): `DataLoader` su dataset di grafi
- Node-level (`is_graph_level=False`): singolo `Data` object con mask + `neighbor_loader()` per large graphs

---

## TRAINER E CALLBACKS

`DINOTrainer` è generico: funziona con qualsiasi `BaseSSLModel`.

**Loop per batch:**
```
compute_loss → backward → clip_grad_norm → post_backward → optimizer.step → post_step
```
**Loop per epoca:**
```
on_epoch_start → batches → on_epoch_end
```

**Hook implementati per modello:**
| Modello | `post_backward` | `post_step` | `on_epoch_start` | `on_epoch_end` |
|---|---|---|---|---|
| BGRL | — | EMA teacher | — | — |
| AFGRL | — | EMA teacher | — | — |
| GraphDINO | freeze last layer | EMA + center update | teacher temp warmup | epoch counter |

**Callbacks disponibili:**
- `EmbeddingLoggerCallback` — salva embeddings ogni N epoche come `.pt`
- `LinearEvalCallback` — linear probe periodica durante il training
- `VisualizationCallback` — UMAP scatter plot (richiede `pip install umap-learn matplotlib`)

---

## BENCHMARK — ZINC E ogbn-arxiv

### ZINC (graph-level, regressione molecolare)
```yaml
encoder:
  name: gin
  node_emb_num_classes: 28  # 28 tipi di atomi
  edge_dim: 64
  edge_emb_num_classes: 4   # 4 tipi di legame
  pool: true
```
- `in_channels` ignorato quando `node_emb_num_classes` è impostato
- Script: `examples/zinc_bgrl.py`, config: `configs/zinc_bgrl.yaml`

### ogbn-arxiv (node-level, classificazione)
```yaml
encoder:
  name: gin
  pool: false               # node-level: no pooling
  hidden_dim: 256
```
- Feature continue 128-dim: nessun embedding categorico
- Richiede `NeighborLoader` per training scalabile
- Script: `examples/ogbn_arxiv_bgrl.py`, config: `configs/ogbn_arxiv_bgrl.yaml`

---

## TEST

```bash
pytest tests/ -v
```

| File | Modello | Note |
|---|---|---|
| `test_bgrl.py` | BGRL | teacher frozen, reset→pesi diversi, EMA scheduler, step counter |
| `test_dgi.py` | DGI | W learnable, shuffle_nodes/shuffle_edges |
| `test_graphcl.py` | GraphCL | projector dim, NT-Xent ≥ 0 |
| `test_vicreg.py` | VICReg | projector 3-layer, loss ≥ 0 |
| `test_barlow_twins.py` | BarlowTwins | lambda default=1/proj_dim |
| `test_afgrl.py` | AFGRL | **skip automatico se faiss non installato** |
| `test_supervised.py` | Supervised | head dim, mini-batch crop |
| `test_graphdino.py` | GraphDINO | freeze last layer, teacher temp warmup, DINOTrainer hooks |
| `test_new_features.py` | — | edge_emb_num_classes, norm_type API, CombinedLoss |

---

## STRUTTURA

```
src/graphssl/
├── core/          ← ABC/Protocol: BaseModel, BaseSSLModel, BaseEncoder, BaseAugmentation, Callback, Registry
├── config/        ← schema.py (dataclass validate), load.py (load_config, build_model)
├── registry/      ← ENCODERS, HEADS, AUGMENTS, LOSSES, LOADERS, OBJECTIVES, DATASETS
├── encoders/      ← GCN, GIN, Transformer (auto-registrati)
├── models/        ← DGI, GraphCL, BGRL, AFGRL, GraphDINO, VICReg, BarlowTwins, Supervised
├── losses/        ← nt_xent, dino, vicreg, barlow, regression, combined (CombinedLoss)
├── nn/            ← MLP, DINOHead, norm (weight_standardize), pooling
├── augmentation/  ← functional.py, transforms.py, compose.py
├── evaluation/    ← linear_probe, knn, visualization (extract_embeddings)
├── training/      ← trainer.py (DINOTrainer), callbacks.py
├── data/          ← DataModule puro Python
└── utils/         ← ema.py (update_ema_params), schedulers.py, positive_miner.py
```

---

## PRINCIPI ARCHITETTURALI

1. **Config-driven uniform API**: tutti i modelli `__init__(config: Dict, in_channels: int)`.
   Il config è validato da un dataclass in `schema.py`. L'encoder è sempre costruito da
   `cfg.encoder.build(in_channels)` — nessun encoder costruito inline nei modelli.

2. **Loss come `nn.Module` standalone** in `losses/`, testabili indipendentemente e combinabili
   via `CombinedLoss`.

3. **Registry pattern** per tutti i componenti estendibili: `@ENCODERS.register("nome")`.

4. **Hook-driven trainer**: `post_backward()`, `post_step()`, `on_epoch_start/end()` invece di
   subclassing del trainer.

5. **Separazione a strati**: `core → encoders/models/nn/augmentation/losses → evaluation → training → data`.
   Nessun modulo di livello superiore importa da uno inferiore.

---

## NOTE IMPLEMENTATIVE

### graph-level vs node-level
`self.graph_level` è derivato da `cfg.encoder.pool` nel costruttore — mai passato come parametro
separato. Nessun modello lo inferisce runtime dal batch.

### Mini-batch: protected nodes
In mini-batch node training, `batch.batch_size` indica i seed nodes.
La loss va calcolata solo su `z[:batch.batch_size]`.
`compose()` accetta `protected_nodes=torch.arange(batch.batch_size)` e lo propaga a ogni augment;
`node_drop` usa questo parametro per non rimuovere mai i seed nodes.

### BGRL vs AFGRL: inizializzazione del target encoder
- **BGRL**: `deepcopy(encoder)` + `reset_parameters()` — pesi DIVERSI da online (cruciale, App. B paper)
- **AFGRL**: `deepcopy(encoder)` senza reset — pesi IDENTICI a online all'inizio

### GraphDINO: ordine operazioni
```
forward → loss → backward → clip_grad → post_backward (freeze last layer)
→ optimizer.step → post_step (EMA teacher + center update)
```

### GINLayer: edge_attr=None con edge_dim impostato
Se `edge_dim` è impostato (usa `GINEConv`) ma `edge_attr=None` a runtime, il layer costruisce
un tensore di zeri della forma corretta per non crashare. Questo permette di usare lo stesso
encoder con e senza edge features durante il training.

### LogRegEvaluator: multilabel
Per ogbg-molpcba le label sono multilabel float `[N, 128]` con NaN.
Usa average precision score invece di accuracy.

### extract_embeddings: global_to_local mapping
In mini-batch, i batch del NeighborLoader arrivano in ordine diverso dai node_ids originali.
Si usa una mappa `global_to_local[node_id] = posizione in z_cpu` per ricostruire l'ordine.

---

## DISTRIBUZIONE

```toml
# pyproject.toml optional-dependencies
viz       = ["umap-learn", "matplotlib", "seaborn"]
benchmark = ["ogb", "pyyaml"]
full      = ["faiss-cpu", "umap-learn", "matplotlib", "seaborn", "ogb", "pyyaml"]
dev       = ["pytest", "pytest-cov", "build", "twine", "pyyaml"]
```

**Release:** ogni tag `v*.*.*` su GitHub attiva automaticamente il workflow
`.github/workflows/publish.yml` che builda e pubblica su PyPI via Trusted Publishing.

## COME AGGIUNGERE UN NUOVO MODELLO (istruzioni per Claude)

### 1. Leggi prima, scrivi poi
Leggi entrambe le versioni (riferimento e GraphSSL) e lista le differenze prima di
modificare qualsiasi file.

### 2. Usa la firma pubblica uniforme
Tutti i modelli usano `__init__(config: Dict, in_channels: int)`, senza eccezioni.

Per ogni nuovo modello:
1. Crea `ModelNameConfig` in `src/config/schema.py` con `__post_init__` validation
   e `from_dict(cls, d: dict)` classmethod.
2. Il costruttore fa solo: `cfg = ModelNameConfig.from_dict(config)` e usa `cfg.*`.
3. Usa `cfg.encoder.build(in_channels)` — non costruire l'encoder inline.
4. `graph_level` si deriva sempre da `cfg.encoder.pool`.

### 3. Checklist dipendenze
Per ogni modello modificato, verifica:
- `losses/` — la loss è testabile standalone?
- `utils/ema.py`, `utils/schedulers.py` — EMA e scheduler usati correttamente in `post_step()`?
- `augmentation/` — `compose()` passa `protected_nodes`?
- `config/schema.py` — aggiunto dataclass config?
- `models/__init__.py`, `src/graphssl/__init__.py` — export aggiornati?

### 4. Aggiorna questo file
Descrivi con precisione: config dataclass, formula della loss, dettagli EMA, iperparametri di default.
