# Benchmarks

## Citation networks: cross-method comparison

All 7 SSL methods, trained and evaluated under **one shared protocol**, so that differences
come from the objective rather than from per-method tuning:

| Setting | Value |
|---|---|
| Encoder | GIN, 2 layers, `hidden_dim=256`, batch norm, no dropout |
| Training | full-batch, 300 steps, AdamW (lr `5e-4`, weight decay `1e-5`) |
| Augmentation | `edge_drop` p=0.5 + `feat_mask` p=0.2, both views (GraphCL, BGRL, VICReg, Barlow Twins, GraphDINO); DGI uses node shuffling, AFGRL none |
| EMA (BGRL, AFGRL) | cosine schedule 0.99 → 1.0 over training |
| Split | public Planetoid train/val/test |
| Evaluation | frozen embeddings → linear probe (Adam-trained) and kNN (k=5, cosine) |
| Seeds | 10; mean ± sample std |
| Hardware | NVIDIA A2 (16 GB), PyTorch 2.11 + CUDA 12.8, PyG 2.8 |

No per-method or per-dataset hyperparameter search is performed: the goal is to compare
objectives at equal budget, not to reproduce each paper's best reported number. Test accuracy
(%), best per column in bold.

### Linear probe

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| DGI | 68.62 ± 2.77 | 51.10 ± 2.26 | 66.16 ± 2.83 |
| GraphCL | 78.06 ± 1.48 | 59.22 ± 2.50 | **80.11 ± 1.24** |
| BGRL | 64.18 ± 3.04 | 47.59 ± 2.22 | 68.67 ± 2.07 |
| AFGRL | 70.06 ± 2.64 | 47.50 ± 2.56 | 73.48 ± 1.27 |
| VICReg | 77.45 ± 1.78 | 61.44 ± 2.10 | 79.07 ± 1.62 |
| Barlow Twins | **78.13 ± 1.00** | **63.00 ± 1.31** | 77.82 ± 1.03 |
| GraphDINO | 61.54 ± 1.79 | 42.00 ± 3.13 | 70.80 ± 3.40 |

### kNN (k=5)

| Method | Cora | CiteSeer | PubMed |
|---|---|---|---|
| DGI | 65.38 ± 2.86 | 46.10 ± 3.63 | 64.47 ± 3.02 |
| GraphCL | 73.70 ± 1.39 | 60.69 ± 2.16 | 77.97 ± 1.33 |
| BGRL | 59.87 ± 3.53 | 43.07 ± 1.29 | 66.97 ± 2.86 |
| AFGRL | 65.13 ± 2.55 | 44.09 ± 2.81 | 70.63 ± 2.31 |
| VICReg | 73.08 ± 1.39 | 58.30 ± 2.50 | **78.28 ± 1.79** |
| Barlow Twins | **75.17 ± 1.06** | **62.23 ± 1.92** | 76.84 ± 1.33 |
| GraphDINO | 54.16 ± 1.33 | 36.58 ± 1.70 | 64.88 ± 3.10 |

### Reading the results

- **Redundancy reduction and contrastive learning lead.** Barlow Twins, VICReg and GraphCL
  form the top group on all three datasets, with seed std ≤ 2.5 points.
- **Teacher-student methods trail under this budget.** BGRL, AFGRL and GraphDINO sit below
  the top group everywhere. Within the family, AFGRL — which uses no augmentation at all —
  beats BGRL on Cora (+5.9) and PubMed (+4.8) and ties on CiteSeer. A plausible explanation
  is the short shared budget and untuned EMA schedule rather than the objectives themselves;
  this has not been verified yet (it needs per-method tuning).
- **Quadratic cost is visible.** On PubMed (≈19.7k nodes) one 300-step run takes ≈440 s for
  GraphCL (pairwise NT-Xent) and ≈500 s for AFGRL (dense kNN + per-step k-means), versus
  ≈35–60 s for the other methods. Wall times are indicative only: some runs shared the GPU.

## Reproduce

```bash
python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --model all --seeds 10
python benchmarks/render_tables.py            # Markdown tables from the saved JSON files
python benchmarks/render_tables.py --latex    # also LaTeX (booktabs)
```

Each run is saved under `benchmarks/results/<Dataset>/<model>__<UTC-timestamp>.json` with
per-seed metrics (including training wall time), aggregate mean/std, the exact
hyperparameters, the git commit (plus whether tracked files had uncommitted changes) and
package versions. Old files are never overwritten, so results accumulate as a reproducible
history; `render_tables.py` uses the latest file per (dataset, model). See
[`benchmarks/README.md`](https://github.com/Simocop7/graphssl/blob/main/benchmarks/README.md)
for the known gaps (only Planetoid datasets are wired up so far). The supervised reference
runs with `--model supervised` (it isn't part of `--model all`).

!!! note "Provenance of the committed results"
    The 20 files recorded at commit `e295451` report `"dirty": true`. That is a false
    positive of the runner at the time — it counted untracked files (logs, the results
    themselves) as modifications; fixed in `dae32cb`. The code that ran is exactly
    `e295451`. GraphCL on PubMed was run at `dae32cb`, whose chunked NT-Xent is numerically
    identical to the unchunked one used for Cora and CiteSeer.

## Comparison with the original BGRL numbers

The paper
([`paper.tex`](https://github.com/Simocop7/graphssl/blob/main/paper.tex), Results section)
places our BGRL numbers next to the original BGRL paper's small-dataset evaluation (Appendix C,
Table 7 of Thakoor et al.) and lists the protocol differences explicitly: random vs. public
splits, GCN vs. GIN encoder, tuned scikit-learn logistic regression vs. an untuned linear
probe. The gap is consistent with those differences; it is not a like-for-like comparison.

## In progress

ogbn-arxiv and OGB graph-level benchmarks (ogbg-molhiv, ogbg-molpcba) need larger compute and
are next. `examples/ogbn_arxiv_bgrl.py` and `examples/zinc_bgrl.py` already implement the
training scripts; `run_benchmark.py` still has to be extended to those datasets.
