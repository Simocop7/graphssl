| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 48.29 ± 0.43 (47.5–49.0) | 48.37 ± 0.90 (47.8–49.4) | 62.40 | 47.50 | 50.60 | 2185.4 |
| propagated A^1X | 62.23 ± 0.47 (61.5–63.4) | 62.60 ± 0.17 (62.5–62.8) | 71.60 | 55.50 | 62.40 | 1572.3 |
| propagated A^2X | 63.57 ± 0.32 (62.7–64.2) | 65.57 ± 0.59 (64.9–66.0) | 72.30 | 46.80 | 65.50 | 1226.8 |
| PCA-256 of A^2X | 43.67 ± 0.83 (42.0–45.0) | 44.10 ± 0.20 (43.9–44.3) | 72.20 | 48.10 | 66.40 | 242.3 |

Test accuracy (%) on CiteSeer. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
