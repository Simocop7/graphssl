| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 69.29 ± 0.27 (68.8–69.7) | 68.80 ± 0.79 (68.2–69.7) | 72.90 | 67.00 | 64.60 | 456.3 |
| propagated A^1X | 73.28 ± 0.30 (72.7–74.1) | 73.27 ± 0.25 (73.0–73.5) | 76.10 | 69.90 | 67.20 | 424.9 |
| propagated A^2X | 75.63 ± 0.21 (75.3–76.0) | 75.57 ± 0.06 (75.5–75.6) | 79.70 | 75.10 | 72.50 | 391.1 |
| PCA-256 of A^2X | 62.58 ± 0.58 (61.5–63.6) | 63.80 ± 0.52 (63.2–64.1) | 79.70 | 54.90 | 71.80 | 213.3 |

Test accuracy (%) on PubMed. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
