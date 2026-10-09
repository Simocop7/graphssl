# Changelog

Notable changes to the `graphssl` package. Benchmark results and ablations are documented in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Unreleased (planned as 0.2.0)

### Changed (behavior differs from 0.1.0)

- **GraphDINO defaults.** `ema_tau_base` 0.996 → 0.9 (annealed to `ema_tau` over
  `total_steps`), `head.teacher_temp` 0.04 → 0.07, `head.warmup_teacher_temp_epochs` 0 → 30.
  With reference DINO's values, accuracy dropped as training continued in the
  citation-network ablations (see the Sensitivity section of `docs/benchmarks.md`). When
  `ema_tau_base` is omitted it now falls back to `min(0.9, ema_tau)` instead of `ema_tau`.
  To keep the 0.1.0 behavior set `ema_tau_base` equal to `ema_tau`, `head.teacher_temp=0.04`
  and `head.warmup_teacher_temp_epochs=0`.
- **GraphDINO center.** The center is now an EMA of the raw teacher logits, as in reference
  DINO. It used to be fed the post-softmax probabilities.
- **Supervised.** Full-batch node training computes the loss on `data.train_mask` only and
  raises if the mask is missing. It used to train on every node's label.
- **DGI.** In graph-level mode `forward()` returns one pooled embedding per graph. It used to
  return node embeddings.
- **`subgraph` augmentation.** It raises when given `protected_nodes`, i.e. in mini-batch
  node training: it keeps the neighbourhood of one random node and cannot keep the seed
  nodes in place, so each view silently paired different nodes.
- **GCN encoder BatchNorm momentum.** 0.01 → 0.1, PyTorch's default and the value the GIN
  and Transformer encoders already used. Training is unchanged (it uses batch statistics);
  eval-mode outputs change for models trained for fewer than about a thousand forward
  passes. With 0.01 the running statistics still held 5% of their initial values after 300
  passes — a variance of 1, where the batch variance after the first GCN layer is 1e-4 to
  1e-2 — so the embeddings in eval mode were not the ones the model had been trained with.
  In the citation benchmark's protocol (300 full-batch steps) the linear probe did not move,
  kNN accuracy was about 6 points lower for DGI and AFGRL on Cora, and the supervised
  model's own head fell from 74% to 49–65% on PubMed (3 seeds each).
  `GCNEncoder(batchnorm_mm=0.01)` gives the old behavior.
- **GCN encoder state dict.** The layers live in `convs`, `norms` and `acts`; the keys were
  `conv1`, `norm1`, `act1`, `conv2`, … A GCN checkpoint saved with 0.1.0 needs its keys
  renamed.
- **Linear probe.** `LogRegEvaluator` is now an L2-regularised logistic regression on
  standardized features (statistics of the training rows), fitted full-batch with L-BFGS from
  a zero initialisation. The L2 strength is selected on the validation split among
  `weight_decay` (default: 10 down to 1e-6 by decades). The result also reports the selected
  `weight_decay` and `converged`. The 0.1.0 probe trained a randomly initialised linear layer
  for 100 Adam steps on raw features: on ogbn-arxiv the same embeddings scored 45–54%
  depending on the seed, and 8 to 20 points below the converged probe. Arguments are now
  keyword-only: `lr` and `epochs` are gone, `max_iter` is the L-BFGS budget. Accuracies are
  not comparable with 0.1.0's. The new probe is slower: seconds on Planetoid-size splits,
  minutes on ogbn-arxiv's 91k training nodes.

### Fixed

- Edge features reach the encoder. Every model called its encoder without `edge_attr`, so an
  edge-aware encoder (GIN with `edge_dim`, the Transformer) ran on zeros: on molecules the
  bond types were ignored, in training and in `extract_embeddings`, without any error.
  Models now go through `graphssl.core.encode(encoder, data)`.
- `edge_add` works with one-dimensional edge attributes (one categorical value per edge).
- The GCN encoder honours `num_layers`. It always built two layers, whatever the config said.
- The GCN encoder's `weight_standardization` standardizes the convolution weights. It did
  nothing: the helper looked for a direct `weight` parameter, which `GCNConv` does not have.
  It now applies to every layer after the first.
- DGI encodes the real and the corrupted graph in one forward pass. With two passes each
  graph was normalised with its own BatchNorm statistics, and in mini-batch training the
  discriminator told them apart from those alone: on ogbn-arxiv the loss went to zero in two
  epochs and the embeddings ended up worse than an untrained encoder's (kNN 33.7% vs 53.3%).
  On a `NeighborLoader` batch the summary and the loss now use the seed nodes only; they
  used to include the sampled neighbours, most of which have no incoming edge in the batch.
- GraphDINO builds its encoders through `cfg.encoder.build()`, so it works with the GCN and
  Transformer backbones (it crashed with both).
- AFGRL no longer lets FAISS use every core for k-means: `kmeans_threads` (default 8) caps
  it. The uncapped default was about 50× slower on a 48-core machine.
- `extract_embeddings` (node-level, full-graph pass) no longer moves the `DataModule`'s graph
  to the inference device as a side effect. A `NeighborLoader` built on that graph before the
  call crashed afterwards on GPU ("Cannot re-initialize CUDA in forked subprocess"), e.g. when
  a linear probe ran during mini-batch training.

### Added

- `attr_mask` augmentation (`AttrMask`): replaces the features of a random fraction of the
  nodes with a mask value — the attribute masking used on molecules. `feat_mask` hides the
  same feature columns on every node, which on a single categorical column masks every atom
  or none.
- `RidgeEvaluator`: closed-form ridge regression on frozen embeddings for regression
  targets, L2 strength selected on validation, MAE and RMSE.
- `Supervised` takes `task: regression` (L1 loss on `num_classes` targets).
- `EncoderConfig.readout`: `'mean'` (default, unchanged), `'sum'` or `'max'` pooling of the
  node embeddings into a graph embedding. The mean does not see the size of a graph.
- `graphssl.core.encode(encoder, data)`: runs an encoder on a graph or batch with its edge
  features, if any.
- `save_model(model, path, config, in_channels, num_classes=None)` and `load_model(path)` in
  `graphssl.config`: a checkpoint that carries what is needed to rebuild the model.
- `graphssl.core.pretrained_encoder(model)`: the encoder a model embeds with, to fine-tune
  from (`supervised.encoder.load_state_dict(pretrained_encoder(model).state_dict())`).
- `EncoderConfig.weight_standardization` (GCN only, off by default).
- `BGRLConfig` / `AFGRLConfig`: `pred_norm` (`'batch'` by default, or `'none'`) and
  `pred_activation` (`'relu'` by default, or `'prelu'`) set the predictor's layout. The
  default is unchanged (Linear → BatchNorm → ReLU → Linear); `pred_norm: none` with
  `pred_activation: prelu` is the predictor of the BGRL reference implementation
  (Linear → PReLU → Linear). `MLP` takes the matching `activation` argument.
- `LinearEvalCallback(evaluator=...)`: the probe to run during training, e.g. a
  `LogRegEvaluator` with a single `weight_decay` when the default one is too slow.
- `HeadConfig.norm_last_layer`: fixes the prototypes' weight-norm scale at 1 (off by default).
- `GraphCLConfig.loss_chunk_size`: NT-Xent in row chunks under gradient checkpointing, same
  value and gradients with bounded memory.
- `AFGRLConfig.knn_chunk_size`: chunked top-k neighbor search, same neighbors with memory
  `O(chunk · N)` instead of `O(N²)`.
- `graphssl.evaluation.effective_rank`: effective rank of a set of embeddings.
- `KNNEvaluator(chunk_size=4096)`: the evaluated split is scored in row chunks, with identical
  predictions. The full similarity matrix (about 17 GB for the ogbn-arxiv test split) is no
  longer built.

## 0.1.0 (2026-09-07)

First release on PyPI.
