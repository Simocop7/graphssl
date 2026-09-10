# Benchmarks

## Citation networks (validated)

BGRL, GIN-2L encoder (`hidden_dim=256`), full-batch training for 300 steps, public Planetoid
split, mean ± std over 10 seeds:

| Dataset | Linear probe (test) | KNN k=5 (test) |
|---|---|---|
| Cora | 62.39 ± 3.71 | 58.61 ± 3.76 |
| CiteSeer | 50.08 ± 1.58 | 43.07 ± 3.61 |
| PubMed | 69.18 ± 2.32 | 66.59 ± 2.88 |

This is a deliberately lightweight, untuned configuration (no per-dataset hyperparameter
search) — its purpose is to validate that the whole pipeline (augmentation → EMA → both
evaluation heads) works correctly end-to-end, not to compete with the state of the art. See
[`paper.tex`](https://github.com/Simocop7/graphssl/blob/main/paper.tex) for a methodological
comparison against BGRL's own original numbers (Appendix C, Table 7).

Reproduce with:

```bash
python examples/benchmark_planetoid.py --dataset Cora --seeds 10
```

## In progress

ogbn-arxiv and OGB graph-level benchmarks (ogbg-molhiv, ogbg-molpcba) require larger compute
and are planned on dedicated infrastructure. `examples/ogbn_arxiv_bgrl.py` and
`examples/zinc_bgrl.py` already implement the training scripts; they just haven't been run to
convergence and reported yet.

## Cross-method comparison

The numbers above are BGRL only. A fair comparison across all 7 SSL methods — same encoder,
same training budget, same evaluation protocol — is exactly what
[`benchmarks/run_benchmark.py`](https://github.com/Simocop7/graphssl/blob/main/benchmarks/run_benchmark.py)
exists to produce:

```bash
# Every SSL model, same encoder/budget, same dataset
python benchmarks/run_benchmark.py --dataset Cora --model all --seeds 10

# Render the results as a Markdown/LaTeX table
python benchmarks/render_tables.py --latex
```

Each run is saved as a JSON file (per-seed metrics, aggregate mean/std, exact hyperparameters,
git commit, package versions) under `benchmarks/results/` — old runs are never overwritten, so
results accumulate as a reproducible history rather than a single point-in-time number. See
[`benchmarks/README.md`](https://github.com/Simocop7/graphssl/blob/main/benchmarks/README.md)
for current known gaps (the supervised baseline isn't wired in yet; only Planetoid datasets are
wired up so far).
