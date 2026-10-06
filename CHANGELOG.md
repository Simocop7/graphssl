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

### Fixed

- GraphDINO builds its encoders through `cfg.encoder.build()`, so it works with the GCN and
  Transformer backbones (it crashed with both).
- AFGRL no longer lets FAISS use every core for k-means: `kmeans_threads` (default 8) caps
  it. The uncapped default was about 50× slower on a 48-core machine.
- `extract_embeddings` (node-level, full-graph pass) no longer moves the `DataModule`'s graph
  to the inference device as a side effect. A `NeighborLoader` built on that graph before the
  call crashed afterwards on GPU ("Cannot re-initialize CUDA in forked subprocess"), e.g. when
  a linear probe ran during mini-batch training.

### Added

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
