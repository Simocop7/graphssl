| Embeddings | adam100 | adam5000 | liblinear | graphssl | kNN | Rank |
|---|---|---|---|---|---|---|
| raw features | 45.34 ± 0.53 (44.3–46.3) | 46.97 ± 0.12 (46.9–47.1) | 60.20 | 52.60 | 46.30 | 1096.0 |
| propagated A^1X | 71.28 ± 0.45 (70.5–71.9) | 73.40 ± 0.62 (72.7–73.9) | 78.40 | 70.40 | 65.20 | 915.2 |
| propagated A^2X | 77.18 ± 0.23 (76.8–77.7) | 78.47 ± 0.50 (78.0–79.0) | 82.00 | 71.50 | 69.70 | 739.8 |
| PCA-256 of A^2X | 55.94 ± 0.51 (54.9–56.8) | 55.97 ± 0.06 (55.9–56.0) | 81.50 | 57.20 | 76.00 | 230.4 |

Test accuracy (%) on Cora. A protocol fitted more than once reports the mean ± sample std over its random initialisations and, in brackets, the range.

- `adam100`: DGI reference code, PyG-SSL: 100 Adam steps, test at the last step.
- `adam5000`: PyGCL: 5,000 Adam steps, test at the best validation.
- `liblinear`: BGRL reference implementation: liblinear on L2-normalised rows, C on validation.
- `graphssl`: GraphSSL: standardized features, L-BFGS to convergence, L2 on validation.
