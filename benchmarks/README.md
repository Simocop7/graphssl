# Benchmarks

Unified, reproducible benchmark runner — one command sweeps any combination of
datasets/models/seeds with the same encoder and training budget, instead of
maintaining a separate script per dataset.

```bash
# Cross-method comparison: all 7 SSL models on Cora, same encoder/budget
python benchmarks/run_benchmark.py --dataset Cora --model all --seeds 10

# One model, multiple datasets
python benchmarks/run_benchmark.py --dataset Cora CiteSeer PubMed --model bgrl

# Render the results (paste straight into README.md / paper.tex)
python benchmarks/render_tables.py
python benchmarks/render_tables.py --latex
```

Each run writes `results/<Dataset>/<model>__<UTC-timestamp>.json` — per-seed
metrics, aggregate mean/std, the exact hyperparameters, the git commit
(and whether tracked files had uncommitted changes), and package versions. Old files
are never overwritten, so results accumulate as a history; `render_tables.py`
always uses the most recent file per (dataset, model) pair.

This intentionally does not replace `examples/*.py`, which stay as minimal
single-file demos for learning the API — this is for producing citable,
comparable numbers across the model zoo. See `../CLAUDE.md` (Benchmarks
section) for the methodology and current status.

**Current results:** all 7 SSL models × Cora/CiteSeer/PubMed × 10 seeds are committed under
`results/` and rendered in `../docs/benchmarks.md`. Budget for a full sweep: most models take
~10–60 s per seed on an NVIDIA A2, but on PubMed GraphCL (O(N²) NT-Xent) and AFGRL (dense
N×N kNN + per-step k-means) take ~7–9 min per seed, so ~75–85 min each for 10 seeds.

**Supervised references** (not part of `all`): both train the same encoder full-batch on the
public split's `train_mask` only.
- `--model supervised` keeps the shared SSL protocol unchanged (no dropout, AdamW, last
  checkpoint). With 120–140 labels and no regularization it overfits, so it understates what
  the labels are worth.
- `--model supervised_reg` uses the standard Kipf & Welling recipe: dropout 0.5, Adam lr 0.01
  with L2 5e-4, checkpoint with the best validation accuracy. This is the credible reference;
  its recipe is recorded in the JSON's `hyperparameters`.

Both get the same linear probe + kNN evaluation on their encoder embeddings; the accuracy of
the trained head itself is saved as `test_acc_head`.

**Known gaps:**
- `afgrl` is skipped automatically when `faiss-cpu` isn't installed (same
  behavior as `tests/test_afgrl.py`).
- Only Planetoid (Cora/CiteSeer/PubMed) is wired up so far — ogbn-arxiv/ZINC
  support is the next extension (add an entry next to `PLANETOID_DATASETS`).
