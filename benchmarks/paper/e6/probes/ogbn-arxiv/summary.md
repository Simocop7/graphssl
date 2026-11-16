| Embeddings | adam100 | adam5000 | graphssl | kNN | Rank |
|---|---|---|---|---|---|
| raw features | 46.85 ± 0.24 (46.4–47.2) | 52.46 ± 0.03 (52.4–52.5) | 52.48 | 47.05 | 112.6 |
| propagated A^1X | 53.68 ± 0.82 (51.3–55.0) | 64.69 ± 0.01 (64.7–64.7) | 64.64 | 64.87 | 82.0 |
| propagated A^2X | 55.22 ± 0.61 (53.7–56.2) | 69.13 ± 0.01 (69.1–69.1) | 69.20 | 67.85 | 79.6 |

Test accuracy (%) on ogbn-arxiv. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
